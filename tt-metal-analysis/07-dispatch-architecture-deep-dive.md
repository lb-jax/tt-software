# Dispatch 架构深度剖析

## 概述

本文档深入剖析 tt-metal 的命令分发（Dispatch）架构，重点解答以下关键问题：

- **Prefetcher 和 Dispatcher 的本质是什么？它们有什么区别？**
- **Hugepage 在哪里？为什么需要它？**
- **Tensix Core 的 5 个 RISC-V 核心如何分工？**
- **Prefetcher 和 Dispatcher 如何在硬件上运行？**
- **完整的命令执行流程是怎样的？**

本文档适合第一次接触 tt-metal 的开发者，将从硬件架构基础开始，逐步深入到实现细节。

## 目录

1. [硬件架构基础](#1-硬件架构基础)
2. [Host 端的 Hugepage 内存](#2-host-端的-hugepage-内存)
3. [Tensix Core 的 RISC-V 架构](#3-tensix-core-的-risc-v-架构)
4. [Prefetcher 和 Dispatcher 详解](#4-prefetcher-和-dispatcher-详解)
5. [完整的命令执行流程](#5-完整的命令执行流程)
6. [代码实现分析](#6-代码实现分析)
7. [性能考虑](#7-性能考虑)
8. [常见疑问解答](#8-常见疑问解答)

---

## 1. 硬件架构基础

### 1.1 Tenstorrent ASIC 组成

以 Wormhole B0 为例，一个 ASIC 包含：

| 组件 | 数量 | 说明 |
|------|------|------|
| **Tensix Tile** | 80 个 | 计算核心（可能有部分被 fused off） |
| **DRAM Tile** | 18 个 | 总共 12 GiB GDDR6 |
| **Ethernet Tile** | 16 个 | 100 GbE 连接，用于多芯片互联 |
| **PCIe Tile** | 1 个 | PCIe 4.0 x16，连接 Host |
| **ARC Tile** | 1 个 | 芯片管理，用户不直接使用 |
| **NoC** | 2 个 | 片上网络，连接所有 Tile |

### 1.2 整体架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Host System (CPU)                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌────────────────────────────────────────────────────────┐         │
│  │  System Memory (DRAM)                                  │         │
│  │                                                        │         │
│  │  ┌──────────────────────────────────────────┐         │         │
│  │  │  Hugepage (DMA-accessible memory)        │  ◄──────┼─────┐   │
│  │  │  - 命令队列 (Issue Queue)                 │         │     │   │
│  │  │  - 数据缓冲区                             │         │     │   │
│  │  └──────────────────────────────────────────┘         │     │   │
│  │                                                        │     │   │
│  │  HWCommandQueue (构建命令)                             │     │   │
│  │     │                                                  │     │   │
│  │     ├── DeviceCommand (命令构建器)                     │     │   │
│  │     │                                                  │     │   │
│  │     └── SystemMemoryManager (内存管理)                │     │   │
│  └────────────────────────────────────────────────────────┘     │   │
│                                                                 │   │
└─────────────────────────────────────────────────────────────────┼───┘
                                                                  │
                            PCIe 4.0 x16 (DMA)                    │
                                  │                               │
┌─────────────────────────────────┼───────────────────────────────┼───┐
│         Device (Tenstorrent ASIC)                               │   │
├─────────────────────────────────┼───────────────────────────────┼───┤
│                                 │                               │   │
│  ┌──────────────────────────────▼────────────────┐              │   │
│  │  PCIExpressTile                               │              │   │
│  │  - Host MMU (映射 BAR 地址空间)                │              │   │
│  │  - DMA Engines (高速数据传输)                  │              │   │
│  │  - Outbound iATU (地址转换)                    │              │   │
│  └───────────────────┬───────────────────────────┘              │   │
│                      │ NoC (Network on Chip)                    │   │
│  ┌───────────────────┼───────────────────────────┐              │   │
│  │  Prefetcher Core  │ (专用 Tensix Core)        │ ◄────────────┘   │
│  │                   │                           │  通过 PCIe 读取   │
│  │  ┌────────────────▼──────────────┐            │  Hugepage        │
│  │  │  BRISC (控制逻辑)              │            │                  │
│  │  │  - 解析 Prefetch 命令          │            │                  │
│  │  │  - 协调数据预取                 │            │                  │
│  │  └────────────┬───────────────────┘            │                  │
│  │               │                                │                  │
│  │  ┌────────────▼──────────────┐                 │                  │
│  │  │  NCRISC (NoC 数据传输)     │                 │                  │
│  │  │  - 通过 NoC 转发数据        │                 │                  │
│  │  └────────────┬───────────────┘                 │                  │
│  └───────────────┼──────────────────────────────────┘                  │
│                  │ NoC 数据流                                         │
│  ┌───────────────▼──────────────────────────────────┐                  │
│  │  Dispatcher Core (专用 Tensix Core)             │                  │
│  │                                                 │                  │
│  │  ┌────────────────────────────┐                 │                  │
│  │  │  NCRISC (主处理器)          │                 │                  │
│  │  │  - 解析 Dispatch 命令       │                 │                  │
│  │  │  - 写入 Worker Cores        │                 │                  │
│  │  │  - 发送 GO 信号             │                 │                  │
│  │  └────────────┬───────────────┘                 │                  │
│  │               │                                 │                  │
│  │  ┌────────────▼───────────────┐                 │                  │
│  │  │  BRISC (辅助控制)           │                 │                  │
│  │  │  - 协助命令处理             │                 │                  │
│  │  └────────────┬───────────────┘                 │                  │
│  └───────────────┼──────────────────────────────────┘                  │
│                  │ NoC 写入 + GO 信号                                  │
│  ┌───────────────▼──────────────────────────────────┐                  │
│  │  Worker Cores (80x Tensix Cores)                │                  │
│  │                                                 │                  │
│  │  每个 Worker Core:                               │                  │
│  │  ┌────────────────────────────┐                 │                  │
│  │  │  BRISC + NCRISC            │ ← 数据移动       │                  │
│  │  │  TRISC0 (UNPACK)           │                 │                  │
│  │  │  TRISC1 (MATH)             │ ← 计算          │                  │
│  │  │  TRISC2 (PACK)             │                 │                  │
│  │  └────────────────────────────┘                 │                  │
│  │  - 执行用户的 kernel 代码                        │                  │
│  │  - 访问 L1 内存 (1464 KB)                       │                  │
│  │  - 通过 NoC 访问 DRAM                            │                  │
│  └──────────────────────────────────────────────────┘                  │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

### 1.3 关键概念

**NoC (Network on Chip)**
- 片上高速互连网络
- 2D Torus 拓扑（10x12 网格）
- 每个 Tile 通过 (x, y) 坐标寻址
- 256-bit 数据宽度，16 个虚拟通道

**Tensix Core**
- 每个核心包含 5 个 RISC-V 处理器
- 1464 KB L1 共享内存
- Tensix 协处理器（矩阵运算、向量运算）
- 通过 NoC 与其他核心通信

**PCIe Tile**
- 唯一与 Host 直接连接的通路
- 支持 PCIe 4.0 x16（理论带宽 32 GB/s 双向）
- 提供 BAR 地址空间映射
- 包含 DMA 引擎用于高速数据传输

---

## 2. Host 端的 Hugepage 内存

### 2.1 什么是 Hugepage？

**Hugepage 是在 Host CPU 的系统内存（DRAM）中分配的大页内存。**

```
┌───────────────────────────────────────────────────┐
│         Host System Memory (DRAM)                 │
├───────────────────────────────────────────────────┤
│                                                   │
│  普通内存 (4KB 页)                                  │
│  ┌─────┬─────┬─────┬─────┐                        │
│  │ 4KB │ 4KB │ 4KB │ 4KB │ ...                    │
│  └─────┴─────┴─────┴─────┘                        │
│                                                   │
│  Hugepage (2MB 或 1GB 页)                          │
│  ┌─────────────────────────────────────┐          │
│  │          2MB Hugepage               │          │
│  │  ┌──────────────────────────┐       │          │
│  │  │  Issue Queue (命令队列)   │       │          │
│  │  ├──────────────────────────┤       │          │
│  │  │  Data Buffers (数据缓冲)  │       │          │
│  │  ├──────────────────────────┤       │          │
│  │  │  Completion Queue (完成)  │       │          │
│  │  └──────────────────────────┘       │          │
│  └─────────────────────────────────────┘          │
│            ▲                                      │
│            │ 物理地址连续，可被 DMA 直接访问         │
│            │                                      │
└────────────┼───────────────────────────────────────┘
             │
             │ PCIe DMA 读取
             │
      ┌──────▼──────┐
      │ PCIe Tile   │
      │  (设备端)    │
      └─────────────┘
```

### 2.2 为什么需要 Hugepage？

| 特性 | 普通页 (4KB) | Hugepage (2MB/1GB) |
|------|-------------|-------------------|
| **物理连续性** | 不保证 | 保证大块连续 |
| **DMA 友好** | 需要 scatter-gather | 可以直接访问 |
| **TLB 效率** | 低（更多 TLB miss） | 高（更少 TLB 条目） |
| **分配开销** | 低 | 高（但运行时高效） |

**Hugepage 的关键优势：**

1. **物理地址连续**：DMA 引擎可以通过单个物理地址访问大块数据
2. **减少 TLB 压力**：一个 TLB 条目可以映射更大的地址范围
3. **PCIe 传输效率**：减少地址转换开销，提高传输效率

### 2.3 Hugepage 的使用流程

```cpp
// 在 SystemMemoryManager 中分配和管理

// 1. Host 端初始化
SystemMemoryManager::SystemMemoryManager(ChipId device_id, uint8_t num_hw_cqs) {
    // 分配 Hugepage (通常由 kernel driver 预先分配)
    this->cq_sysmem_start = allocate_hugepage(cq_size);

    // 创建 Issue Queue (Host → Device 命令)
    for (uint8_t cq_id = 0; cq_id < num_hw_cqs; cq_id++) {
        cq_interfaces.emplace_back(
            cq_sysmem_start,  // Hugepage 起始地址
            cq_id,
            cq_size,
            channel_offset
        );
    }
}

// 2. Host 写入命令到 Hugepage
void* SystemMemoryManager::issue_queue_reserve(uint32_t cmd_size, uint32_t cq_id) {
    // 返回 Hugepage 中的可用空间指针
    return cq_interfaces[cq_id].get_write_ptr();
}

void SystemMemoryManager::issue_queue_push_back(uint32_t size, uint32_t cq_id) {
    // 更新写入指针
    cq_interfaces[cq_id].advance_write_ptr(size);
}

// 3. 通知设备读取
void SystemMemoryManager::fetch_queue_write(uint32_t size, uint32_t cq_id) {
    // 写入 fetch queue，通知 Prefetcher 有新命令
    // Prefetcher 通过 PCIe 轮询这个地址
}
```

### 2.4 Hugepage 的内存布局

```
Hugepage 内存布局 (每个 Command Queue):

┌─────────────────────────────────────────────────┐
│  Offset 0x0000_0000                             │
├─────────────────────────────────────────────────┤
│                                                 │
│  Issue Queue (Host → Device)                    │
│  - Ring buffer 结构                              │
│  - Host 写入命令和数据                            │
│  - Prefetcher 通过 PCIe DMA 读取                │
│                                                 │
│  ┌────────────┬────────────┬────────────┐       │
│  │ Command 1  │ Command 2  │ Command 3  │ ...   │
│  └────────────┴────────────┴────────────┘       │
│   ▲                                      ▲      │
│   │ rd_ptr                         wr_ptr│      │
│   │ (Prefetcher 读取位置)    (Host 写入位置) │   │
│                                                 │
├─────────────────────────────────────────────────┤
│                                                 │
│  Completion Queue (Device → Host)               │
│  - Dispatcher 写入完成事件                       │
│  - Host 轮询读取                                 │
│                                                 │
├─────────────────────────────────────────────────┤
│                                                 │
│  Prefetch Queue (控制信息)                       │
│  - rd_ptr: Prefetcher 读取位置                  │
│  - wr_ptr: Host 写入位置                        │
│                                                 │
└─────────────────────────────────────────────────┘
```

### 2.5 Prefetcher 如何访问 Hugepage

```cpp
// cq_prefetch.cpp - Prefetcher 在设备上运行

constexpr uint32_t pcie_base = PCIE_BASE;  // 映射到 Host Hugepage 的地址
constexpr uint32_t pcie_size = PCIE_SIZE;

// Prefetcher 的读取指针（在设备 L1 内存中）
static uint32_t pcie_read_ptr = pcie_base;

// 从 Hugepage 读取命令
void read_command_from_host() {
    // 通过 NoC 访问 PCIe Tile，PCIe Tile 通过 DMA 从 Host 读取
    noc_async_read(
        pcie_read_ptr,        // Host Hugepage 中的地址
        local_buffer_addr,    // 设备 L1 内存地址
        cmd_size              // 命令大小
    );

    // 推进读取指针
    pcie_read_ptr += cmd_size;

    // Wrap around (环形缓冲区)
    if (pcie_read_ptr >= pcie_base + pcie_size) {
        pcie_read_ptr = pcie_base;
    }
}
```

**关键理解：**
- `pcie_base` 是一个**设备侧的地址**，映射到 Host 的 Hugepage
- 当 Prefetcher 访问这个地址时，NoC 路由到 PCIe Tile
- PCIe Tile 的 DMA 引擎将请求转换为 PCIe 读取事务
- Host 的 PCIe 控制器响应请求，返回 Hugepage 中的数据

---

## 3. Tensix Core 的 RISC-V 架构

### 3.1 Tensix Core 组成

每个 Tensix Core 包含：

```
┌─────────────────────────────────────────────────────────┐
│                    Tensix Core                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │  L1 Memory (1464 KB 共享)                       │   │
│  │  - 所有 RISC-V 核心共享                          │   │
│  │  - 用户数据、Circular Buffers、临时数据          │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  RISC-V B    │  │  RISC-V NC   │  │  NoC Router  │ │
│  │  (BRISC)     │  │  (NCRISC)    │  │  - NoC 0     │ │
│  │              │  │              │  │  - NoC 1     │ │
│  │  4KB 本地RAM │  │  4KB 本地RAM │  │              │ │
│  │  2KB I-cache │  │  16KB IRAM   │  └──────────────┘ │
│  │              │  │  ½KB I-cache │                   │
│  └──────────────┘  └──────────────┘                   │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Tensix Coprocessor (AI 加速器)                  │  │
│  │                                                  │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐ │  │
│  │  │ RISC-V T0  │  │ RISC-V T1  │  │ RISC-V T2  │ │  │
│  │  │ (TRISC0)   │  │ (TRISC1)   │  │ (TRISC2)   │ │  │
│  │  │            │  │            │  │            │ │  │
│  │  │ 2KB本地RAM │  │ 2KB本地RAM │  │ 2KB本地RAM │ │  │
│  │  │ 2KB I-cache│  │ ½KB I-cache│  │ 2KB I-cache│ │  │
│  │  └────────────┘  └────────────┘  └────────────┘ │  │
│  │       │              │              │           │  │
│  │       ▼              ▼              ▼           │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐        │  │
│  │  │ Unpacker│  │ Matrix  │  │ Packer  │        │  │
│  │  │ (2x)    │  │ Unit    │  │ (4x)    │        │  │
│  │  └─────────┘  │ Vector  │  └─────────┘        │  │
│  │               │ Unit    │                      │  │
│  │               └─────────┘                      │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 3.2 五个 RISC-V 核心详解

| RISC-V 核心 | 全称 | 本地内存 | 主要职责 | 特点 |
|------------|------|---------|---------|------|
| **RISC-V B** | BRISC (Baby RISC-V) | 4KB RAM<br>2KB I-cache | **数据移动控制**<br>- 控制其他核心<br>- 协调 NoC 传输<br>- 处理 Mailbox 通信 | 擅长控制逻辑<br>可访问所有控制寄存器 |
| **RISC-V NC** | NCRISC (Network Controller) | 4KB RAM<br>16KB IRAM<br>½KB I-cache | **NoC 数据传输**<br>- 处理 NoC 读写<br>- DMA 控制<br>- 大规模数据移动 | 最大指令空间<br>适合复杂传输逻辑 |
| **RISC-V T0** | TRISC0 (Tensix 0) | 2KB RAM<br>2KB I-cache | **UNPACK**<br>- 从 L1 加载数据<br>- 解压/转换格式<br>- 送入 Tensix 协处理器 | 直接控制 Unpacker<br>准备计算数据 |
| **RISC-V T1** | TRISC1 (Tensix 1) | 2KB RAM<br>½KB I-cache | **MATH**<br>- 控制矩阵运算<br>- 控制向量运算<br>- FPU 操作 | 直接控制 Matrix/Vector Unit<br>执行数学运算 |
| **RISC-V T2** | TRISC2 (Tensix 2) | 2KB RAM<br>2KB I-cache | **PACK**<br>- 从协处理器读取结果<br>- 打包/转换格式<br>- 写回 L1 | 直接控制 Packer<br>存储计算结果 |

### 3.3 TT-Metalium 的标准分工

根据官方 ISA 文档：

> **If using TT-Metalium, data movement kernels run on RISCV B and RISCV NC, whereas compute kernels run on RISCV T0 (UNPACK part of kernel) and T1 (MATH part of kernel) and T2 (PACK part of kernel).**

```
用户编写的 Kernel 类型与 RISC-V 核心的映射:

┌─────────────────────────────────────────────────────┐
│  Data Movement Kernel (DataFlow API)                │
│                                                     │
│  reader_kernel.cpp  →  BRISC + NCRISC              │
│  writer_kernel.cpp  →  BRISC + NCRISC              │
│                                                     │
│  职责:                                               │
│  - 从 DRAM 读取数据到 L1                             │
│  - 从 L1 写入数据到 DRAM                             │
│  - 通过 NoC 传输数据                                 │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  Compute Kernel (Compute API)                       │
│                                                     │
│  compute_kernel.cpp  →  TRISC0 + TRISC1 + TRISC2   │
│                                                     │
│  职责:                                               │
│  - TRISC0: 从 L1 Unpack 数据                        │
│  - TRISC1: 执行矩阵/向量运算                         │
│  - TRISC2: Pack 结果到 L1                           │
└─────────────────────────────────────────────────────┘
```

### 3.4 RISC-V 核心的协同工作示例

以一个简单的矩阵乘法为例：

```cpp
// Worker Core 执行流程

// === BRISC (数据移动控制) ===
void brisc_main() {
    // 1. 从 DRAM 读取输入矩阵 A 到 L1
    cb_push_back(cb_id_in0, num_tiles);

    // 2. 通知 NCRISC 执行 NoC 传输
    // 3. 通知 TRISC0 数据已准备好
    // 4. 等待 TRISC2 完成
    // 5. 写回结果到 DRAM
}

// === NCRISC (NoC 传输) ===
void ncrisc_main() {
    // 执行实际的 NoC 读写
    noc_async_read(dram_addr, l1_addr, size);
    noc_async_read_barrier();
}

// === TRISC0 (Unpack) ===
void trisc0_main() {
    // 从 L1 加载数据到 Tensix 寄存器
    unpack_tiles(cb_id_in0, num_tiles);
}

// === TRISC1 (Math) ===
void trisc1_main() {
    // 执行矩阵乘法
    matmul_tiles(tile_a, tile_b, tile_out);
}

// === TRISC2 (Pack) ===
void trisc2_main() {
    // 将结果打包回 L1
    pack_tiles(cb_id_out, num_tiles);
}
```

**关键理解：**
- 5 个核心在**同一个 Tensix Core 内并行执行**
- 它们共享 L1 内存，通过 Circular Buffers 通信
- 每个核心有自己的本地 RAM（栈、局部变量）
- 协同工作通过信号量（semaphores）和 mailbox 同步

---

## 4. Prefetcher 和 Dispatcher 详解

### 4.1 核心概念辨析

**最关键的理解：Prefetcher 和 Dispatcher 是两个独立的固件程序，运行在两个不同的 Tensix Core 上。**

```
┌─────────────────────────────────────────────────────┐
│  Prefetcher Core (Tensix Core #1)                  │
│  - 位置: 通常在靠近 PCIe Tile 的位置                  │
│  - 固件: cq_prefetch.cpp (独立编译)                  │
│  - 运行在: BRISC + NCRISC                            │
└─────────────────────────────────────────────────────┘
              │
              │ NoC 数据流
              ▼
┌─────────────────────────────────────────────────────┐
│  Dispatcher Core (Tensix Core #2)                  │
│  - 位置: 独立的 Tensix Core                          │
│  - 固件: cq_dispatch.cpp (独立编译)                  │
│  - 运行在: NCRISC + BRISC                            │
└─────────────────────────────────────────────────────┘
              │
              │ NoC 数据流
              ▼
┌─────────────────────────────────────────────────────┐
│  Worker Cores (多个 Tensix Cores)                   │
│  - 位置: 计算网格                                     │
│  - 固件: 用户编写的 kernels                           │
│  - 运行在: BRISC/NCRISC (数据) + TRISC0/1/2 (计算)   │
└─────────────────────────────────────────────────────┘
```

### 4.2 Prefetcher 详解

#### 4.2.1 职责

**Prefetcher 是命令流的"搬运工"**，负责：

1. **从 Host Hugepage 读取命令**（通过 PCIe DMA）
2. **预取数据**：
   - 从 Host 读取内联数据
   - 从 DRAM 读取数据
3. **转发给 Dispatcher**：通过 NoC 将命令和数据发送到 Dispatcher
4. **流量控制**：通过信号量与 Dispatcher 同步

#### 4.2.2 Prefetcher 命令类型

```cpp
// 位置: tt_metal/impl/dispatch/kernels/cq_commands.hpp

enum CQPrefetchCmdId : uint8_t {
    CQ_PREFETCH_CMD_RELAY_LINEAR = 1,      // 从 DRAM 线性读取数据
    CQ_PREFETCH_CMD_RELAY_LINEAR_H = 2,    // 从 Host 线性读取数据
    CQ_PREFETCH_CMD_RELAY_PAGED = 3,       // 分页读取数据
    CQ_PREFETCH_CMD_RELAY_INLINE = 4,      // 内联数据 (命令中直接包含数据)
    CQ_PREFETCH_CMD_RELAY_INLINE_NOFLUSH = 5,
    CQ_PREFETCH_CMD_STALL = 6,             // 等待 dispatcher 完成
    CQ_PREFETCH_CMD_EXEC_BUF = 7,          // 执行 buffer (trace 回放)
    CQ_PREFETCH_CMD_EXEC_BUF_END = 8,      // 执行 buffer 结束
    CQ_PREFETCH_CMD_SET_OFFSET = 9,        // 设置地址偏移
    CQ_PREFETCH_CMD_DEBUG = 10,            // 调试命令
    CQ_PREFETCH_CMD_TERMINATE = 11,        // 终止 prefetcher
};
```

#### 4.2.3 Prefetcher 的 RISC-V 分工

```
Prefetcher Core 内部:

┌─────────────────────────────────────────────────────┐
│  BRISC (主控制逻辑)                                  │
│  ┌─────────────────────────────────────────────┐   │
│  │  while (true) {                              │   │
│  │    // 1. 从 Hugepage 读取命令头               │   │
│  │    cmd = read_cmd_from_pcie(pcie_read_ptr);  │   │
│  │                                              │   │
│  │    // 2. 根据命令类型处理                     │   │
│  │    switch (cmd.cmd_id) {                     │   │
│  │      case RELAY_LINEAR_H:                    │   │
│  │        // 从 Host 读取数据                    │   │
│  │        noc_read_from_pcie(...);              │   │
│  │        break;                                │   │
│  │      case RELAY_LINEAR:                      │   │
│  │        // 从 DRAM 读取数据                    │   │
│  │        noc_read_from_dram(...);              │   │
│  │        break;                                │   │
│  │      case STALL:                             │   │
│  │        // 等待 Dispatcher 处理完成            │   │
│  │        wait_for_dispatcher();                │   │
│  │        break;                                │   │
│  │    }                                         │   │
│  │                                              │   │
│  │    // 3. 转发给 Dispatcher                   │   │
│  │    forward_to_dispatcher(cmd, data);         │   │
│  │  }                                           │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
              │ 调用 NCRISC 执行 NoC 传输
              ▼
┌─────────────────────────────────────────────────────┐
│  NCRISC (NoC 传输执行)                               │
│  ┌─────────────────────────────────────────────┐   │
│  │  // NoC 写入到 Dispatcher Core              │   │
│  │  noc_async_write(                           │   │
│  │    local_addr,        // 本地 L1 地址        │   │
│  │    dispatcher_noc_xy, // Dispatcher 的坐标   │   │
│  │    dispatcher_addr,   // Dispatcher L1 地址  │   │
│  │    size               // 数据大小            │   │
│  │  );                                         │   │
│  │  noc_async_write_barrier();                 │   │
│  │                                              │   │
│  │  // 更新 Dispatcher 的信号量                 │   │
│  │  noc_semaphore_inc(                         │   │
│  │    dispatcher_noc_xy,                       │   │
│  │    page_ready_sem_id,                       │   │
│  │    1                                        │   │
│  │  );                                         │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

**代码证据：**

```cpp
// cq_prefetch.cpp 注释
// Write cmd buf allocation:
//  - BRISC_WR_CMD_BUF: writes to downstream_noc_xy
//  - BRISC_WR_REG_CMD_BUF: small writes to dispatch_s_noc_xy

// BRISC 负责写入命令到 Dispatcher
static constexpr uint32_t downstream_write_cmd_buf = BRISC_WR_CMD_BUF;

// NCRISC 负责执行 NoC 传输
relay_client.write_any_len<my_noc_index, true, NCRISC_WR_CMD_BUF>(...);
```

### 4.3 Dispatcher 详解

#### 4.3.1 职责

**Dispatcher 是命令的"执行者"**，负责：

1. **接收来自 Prefetcher 的命令和数据**
2. **解析 Dispatch 命令**
3. **写入数据到 Worker Cores 的 L1 内存**
4. **发送控制信号**：
   - GO 信号：启动 Worker Cores 执行 kernel
   - WAIT 信号：等待 Worker Cores 完成
5. **写入完成事件到 Completion Queue**

#### 4.3.2 Dispatcher 命令类型

```cpp
enum CQDispatchCmdId : uint8_t {
    CQ_DISPATCH_CMD_WRITE_LINEAR = 1,      // 线性写入 L1
    CQ_DISPATCH_CMD_WRITE_LINEAR_H = 2,    // 线性写入 Host
    CQ_DISPATCH_CMD_WRITE_LINEAR_H_HOST = 3, // Host 直接写
    CQ_DISPATCH_CMD_WRITE_PAGED = 4,       // 分页写入
    CQ_DISPATCH_CMD_WRITE_PACKED = 5,      // 打包写入 (多目标)
    CQ_DISPATCH_CMD_WRITE_PACKED_LARGE = 6, // 大数据打包写入
    CQ_DISPATCH_CMD_WAIT = 7,              // 等待条件
    CQ_DISPATCH_CMD_GO = 8,                // 发送 go signal
    CQ_DISPATCH_CMD_SINK = 9,              // 数据同步
    CQ_DISPATCH_CMD_DEBUG = 10,            // 调试
    CQ_DISPATCH_CMD_DELAY = 11,            // 延迟
    CQ_DISPATCH_CMD_SET_WRITE_OFFSET = 12, // 设置写偏移
    CQ_DISPATCH_CMD_TERMINATE = 13,        // 终止 dispatcher
};
```

#### 4.3.3 Dispatcher 的 RISC-V 分工

```
Dispatcher Core 内部:

┌─────────────────────────────────────────────────────┐
│  NCRISC (主处理器)                                   │
│  ┌─────────────────────────────────────────────┐   │
│  │  while (true) {                              │   │
│  │    // 1. 等待 Prefetcher 的数据               │   │
│  │    wait_for_semaphore(page_ready_sem_id);    │   │
│  │                                              │   │
│  │    // 2. 从本地 dispatch buffer 读取命令      │   │
│  │    cmd = read_dispatch_cmd(dispatch_cb);     │   │
│  │                                              │   │
│  │    // 3. 根据命令类型处理                     │   │
│  │    switch (cmd.cmd_id) {                     │   │
│  │      case WRITE_LINEAR:                      │   │
│  │        // 写入数据到 Worker Core L1          │   │
│  │        noc_write_to_worker(                  │   │
│  │          worker_noc_xy,                      │   │
│  │          worker_l1_addr,                     │   │
│  │          data, size                          │   │
│  │        );                                    │   │
│  │        break;                                │   │
│  │                                              │   │
│  │      case WRITE_PACKED:                      │   │
│  │        // 批量写入多个 Worker Cores           │   │
│  │        for (core : cores) {                  │   │
│  │          noc_write(core, ...);               │   │
│  │        }                                     │   │
│  │        break;                                │   │
│  │                                              │   │
│  │      case GO:                                │   │
│  │        // 发送 GO 信号启动 Workers            │   │
│  │        send_go_signal(worker_cores);         │   │
│  │        break;                                │   │
│  │                                              │   │
│  │      case WAIT:                              │   │
│  │        // 等待 Workers 完成                   │   │
│  │        wait_for_workers(sem_addr);           │   │
│  │        break;                                │   │
│  │    }                                         │   │
│  │                                              │   │
│  │    // 4. 通知 Prefetcher 可以继续             │   │
│  │    notify_prefetcher(page_done_sem_id);      │   │
│  │  }                                           │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
              │ 调用 NoC 写入
              ▼
┌─────────────────────────────────────────────────────┐
│  BRISC (辅助控制)                                    │
│  - 协助处理一些控制逻辑                               │
│  - 管理本地 buffer                                   │
│  - 处理 Completion Queue 写入                        │
└─────────────────────────────────────────────────────┘
```

**代码证据：**

```cpp
// cq_dispatch.cpp

// NCRISC 负责执行 NoC 写入
relay_client.init_write_state_only<my_noc_index, NCRISC_WR_CMD_BUF>(
    get_noc_addr_helper(downstream_noc_xy, 0)
);

relay_client.write<my_noc_index, true, NCRISC_WR_CMD_BUF>(...);
```

### 4.4 Prefetcher 和 Dispatcher 的同步机制

两个核心通过**信号量 (Semaphores)** 进行同步：

```
Prefetcher Core                        Dispatcher Core
     │                                      │
     │  1. 写入数据到 Dispatcher buffer      │
     │     (通过 NoC)                        │
     ├──────────────────────────────────────>│
     │                                      │
     │  2. 增加 page_ready 信号量            │
     │     noc_semaphore_inc(               │
     │       dispatcher_xy,                 │
     │       page_ready_sem, 1)             │
     ├──────────────────────────────────────>│ 3. 等待信号量
     │                                      │    wait_semaphore(
     │                                      │      page_ready_sem)
     │                                      │
     │                                      │ 4. 处理命令
     │                                      │    - 解析命令
     │                                      │    - 写入 Workers
     │                                      │    - 发送 GO 信号
     │                                      │
     │                                      │ 5. 增加 page_done 信号量
     │                                      │    noc_semaphore_inc(
     │<──────────────────────────────────────┤      prefetcher_xy,
     │                                      │      page_done_sem, 1)
     │                                      │
     │  6. 等待信号量                        │
     │     wait_semaphore(page_done_sem)    │
     │                                      │
     │  7. 可以覆盖 buffer，继续下一批        │
     │                                      │
```

**关键变量：**

```cpp
// cq_prefetch.cpp
constexpr uint32_t my_downstream_cb_sem_id = MY_DOWNSTREAM_CB_SEM_ID;  // page_done
constexpr uint32_t downstream_cb_sem_id = DOWNSTREAM_CB_SEM_ID;        // page_ready

// cq_dispatch.cpp
constexpr uint32_t my_dispatch_cb_sem_id = MY_DISPATCH_CB_SEM_ID;      // page_ready
constexpr uint32_t upstream_dispatch_cb_sem_id = UPSTREAM_DISPATCH_CB_SEM_ID; // page_done
```

### 4.5 为什么需要两级设计？

| 优势 | 说明 |
|------|------|
| **流水线并行** | Prefetcher 预取下一批数据时，Dispatcher 处理当前批次 |
| **解耦读写** | Prefetcher 专注于 PCIe/DRAM 读取，Dispatcher 专注于 Worker 写入 |
| **灵活的数据源** | Prefetcher 可以从多个源获取数据（Host、DRAM、inline） |
| **减少延迟** | 通过预取隐藏 PCIe 和 DRAM 访问延迟 |
| **负载均衡** | 两个独立核心，避免单点瓶颈 |
| **扩展性** | 可以有多个 Dispatcher（分布式 dispatch） |

---

## 5. 完整的命令执行流程

### 5.1 端到端流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│  Step 1: Host 构建命令                                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  HWCommandQueue::enqueue_program(program)                          │
│    │                                                                │
│    ├─ 1.1 编译 kernels (如果还未编译)                                │
│    │   program.compile(device)                                     │
│    │                                                                │
│    ├─ 1.2 生成设备命令序列                                           │
│    │   DeviceCommand cmd;                                          │
│    │   cmd.add_dispatch_wait(...);        // 等待前一程序            │
│    │   cmd.add_prefetch_relay_linear(...); // 加载 kernel 二进制    │
│    │   cmd.add_dispatch_write_linear(...); // 写入运行时参数        │
│    │   cmd.add_dispatch_go(...);           // 启动 kernel           │
│    │                                                                │
│    ├─ 1.3 写入到 Hugepage                                           │
│    │   void* ptr = manager.issue_queue_reserve(cmd.size());        │
│    │   memcpy(ptr, cmd.data(), cmd.size());                        │
│    │   manager.issue_queue_push_back(cmd.size());                  │
│    │                                                                │
│    └─ 1.4 通知 Prefetcher                                           │
│        manager.fetch_queue_write(cmd.size());                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │ Hugepage 中的命令
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 2: Prefetcher 读取和转发                                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Prefetcher Core (BRISC + NCRISC)                                  │
│    │                                                                │
│    ├─ 2.1 轮询 fetch queue                                          │
│    │   检测到新命令                                                  │
│    │                                                                │
│    ├─ 2.2 从 Hugepage 读取命令 (通过 PCIe)                           │
│    │   noc_async_read(pcie_base + offset, local_buf, size);        │
│    │   // PCIe Tile 的 DMA 引擎执行 Host → Device 传输              │
│    │                                                                │
│    ├─ 2.3 处理 Prefetch 命令                                        │
│    │   switch (cmd.cmd_id) {                                       │
│    │     case RELAY_LINEAR_H:                                      │
│    │       // 从 Host 读取数据 (例如 kernel 二进制)                  │
│    │       noc_read_from_pcie(pcie_addr, local_buf, size);         │
│    │       break;                                                  │
│    │     case RELAY_LINEAR:                                        │
│    │       // 从 DRAM 读取数据                                      │
│    │       noc_read_from_dram(dram_addr, local_buf, size);         │
│    │       break;                                                  │
│    │   }                                                           │
│    │                                                                │
│    ├─ 2.4 转发到 Dispatcher (通过 NoC)                              │
│    │   noc_async_write(                                            │
│    │     local_buf,              // 本地 L1                         │
│    │     dispatcher_noc_xy,      // Dispatcher 坐标                │
│    │     dispatcher_cb_addr,     // Dispatcher buffer              │
│    │     size                                                      │
│    │   );                                                          │
│    │                                                                │
│    └─ 2.5 通知 Dispatcher                                           │
│        noc_semaphore_inc(dispatcher_xy, page_ready_sem, 1);        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │ NoC 数据传输
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 3: Dispatcher 解析和分发                                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Dispatcher Core (NCRISC + BRISC)                                  │
│    │                                                                │
│    ├─ 3.1 等待数据准备好                                             │
│    │   noc_semaphore_wait(page_ready_sem, 1);                      │
│    │                                                                │
│    ├─ 3.2 从 dispatch buffer 读取命令                               │
│    │   cmd = read_from_dispatch_cb();                              │
│    │                                                                │
│    ├─ 3.3 处理 Dispatch 命令                                        │
│    │   switch (cmd.cmd_id) {                                       │
│    │                                                                │
│    │     case WRITE_LINEAR:                                        │
│    │       // 写入 kernel 二进制到 Worker Core L1                   │
│    │       noc_async_write(                                        │
│    │         local_data,                                           │
│    │         worker_noc_xy,     // Worker 坐标                     │
│    │         worker_l1_addr,    // Worker L1 地址                  │
│    │         kernel_size                                           │
│    │       );                                                      │
│    │       break;                                                  │
│    │                                                                │
│    │     case WRITE_PACKED:                                        │
│    │       // 批量写入多个 Workers (例如运行时参数)                   │
│    │       for (core : core_list) {                                │
│    │         noc_async_write(                                      │
│    │           runtime_args,                                       │
│    │           core.noc_xy,                                        │
│    │           core.rtargs_addr,                                   │
│    │           args_size                                           │
│    │         );                                                    │
│    │       }                                                       │
│    │       break;                                                  │
│    │                                                                │
│    │     case GO:                                                  │
│    │       // 发送 GO 信号，启动 Workers                             │
│    │       send_go_signal_multicast(worker_grid);                  │
│    │       // Workers 的 BRISC 会检测到这个信号并启动 kernel         │
│    │       break;                                                  │
│    │                                                                │
│    │     case WAIT:                                                │
│    │       // 等待 Workers 完成                                     │
│    │       noc_semaphore_wait(worker_done_sem, num_workers);       │
│    │       break;                                                  │
│    │   }                                                           │
│    │                                                                │
│    └─ 3.4 通知 Prefetcher 完成                                      │
│        noc_semaphore_inc(prefetcher_xy, page_done_sem, 1);         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │ NoC 写入 + GO 信号
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 4: Worker Cores 执行 Kernel                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Worker Core (所有 5 个 RISC-V)                                     │
│    │                                                                │
│    ├─ 4.1 BRISC 等待 GO 信号                                        │
│    │   while (!go_signal_received) { /* spin */ }                  │
│    │                                                                │
│    ├─ 4.2 BRISC 启动其他 RISC-V 核心                                │
│    │   launch_ncrisc();  // 启动数据移动                            │
│    │   launch_trisc0();  // 启动 UNPACK                            │
│    │   launch_trisc1();  // 启动 MATH                              │
│    │   launch_trisc2();  // 启动 PACK                              │
│    │                                                                │
│    ├─ 4.3 执行 Kernel                                               │
│    │   // Data Movement (BRISC + NCRISC)                           │
│    │   - 从 DRAM 读取输入数据到 L1 Circular Buffers                 │
│    │   - 通知 Compute kernels 数据准备好                            │
│    │                                                                │
│    │   // Compute (TRISC0 + TRISC1 + TRISC2)                       │
│    │   - TRISC0: Unpack 数据到 Tensix 寄存器                       │
│    │   - TRISC1: 执行矩阵/向量运算                                  │
│    │   - TRISC2: Pack 结果回 L1                                    │
│    │                                                                │
│    │   // Data Movement (BRISC + NCRISC)                           │
│    │   - 等待 Compute 完成                                          │
│    │   - 将结果从 L1 写回 DRAM                                      │
│    │                                                                │
│    └─ 4.4 通知完成                                                  │
│        noc_semaphore_inc(dispatcher_xy, worker_done_sem, 1);       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │ 完成信号
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 5: 完成通知                                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Dispatcher Core                                                    │
│    │                                                                │
│    ├─ 5.1 检测到所有 Workers 完成                                    │
│    │   noc_semaphore_wait(worker_done_sem, num_workers);           │
│    │                                                                │
│    └─ 5.2 写入完成事件到 Completion Queue (Host 内存)               │
│        noc_async_write(                                            │
│          event_data,                                               │
│          pcie_tile_noc_xy,                                         │
│          completion_queue_addr,  // Host Hugepage 中的地址          │
│          event_size                                                │
│        );                                                          │
│        // PCIe Tile 通过 DMA 写入 Host 内存                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │ PCIe DMA
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 6: Host 读取完成事件                                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Host CPU                                                           │
│    │                                                                │
│    ├─ 6.1 Completion Queue 线程轮询                                 │
│    │   completion_queue_thread.run() {                             │
│    │     while (true) {                                            │
│    │       event = poll_completion_queue();                        │
│    │       if (event) {                                            │
│    │         notify_waiting_threads(event);                        │
│    │       }                                                       │
│    │     }                                                         │
│    │   }                                                           │
│    │                                                                │
│    ├─ 6.2 EnqueueProgram 返回                                       │
│    │   // 或者 Finish() 返回                                         │
│    │                                                                │
│    └─ 6.3 用户代码继续执行                                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 具体示例：运行一个简单的 Add Kernel

假设我们要在设备上执行 `c = a + b`：

```cpp
// 用户代码 (Host)
Program program = CreateProgram();

// 创建 kernel
KernelHandle add_kernel = CreateKernel(
    program,
    "add_kernel.cpp",  // 计算 kernel
    core_coord,
    ComputeConfig{}
);

// 设置运行时参数
SetRuntimeArgs(program, add_kernel, core_coord, {num_tiles});

// 创建数据移动 kernels (reader/writer)
KernelHandle reader = CreateKernel(
    program,
    "reader_kernel.cpp",
    core_coord,
    DataMovementConfig{.processor = DataMovementProcessor::RISCV_0}
);

KernelHandle writer = CreateKernel(
    program,
    "writer_kernel.cpp",
    core_coord,
    DataMovementConfig{.processor = DataMovementProcessor::RISCV_1}
);

// 执行
EnqueueProgram(cq, program, false);
Finish(cq);
```

**生成的命令序列（简化版）：**

```
Host 写入 Hugepage:
┌─────────────────────────────────────────────┐
│ 1. CQ_PREFETCH_CMD_RELAY_LINEAR_H           │
│    - 从 Host 读取 reader_kernel.bin         │
│    - size: 4096 bytes                       │
├─────────────────────────────────────────────┤
│ 2. CQ_DISPATCH_CMD_WRITE_LINEAR             │
│    - 写入 reader_kernel.bin 到 Worker L1    │
│    - noc_xy: (1, 1)                         │
│    - addr: 0x0 (L1 base)                    │
├─────────────────────────────────────────────┤
│ 3. CQ_PREFETCH_CMD_RELAY_LINEAR_H           │
│    - 从 Host 读取 writer_kernel.bin         │
│    - size: 4096 bytes                       │
├─────────────────────────────────────────────┤
│ 4. CQ_DISPATCH_CMD_WRITE_LINEAR             │
│    - 写入 writer_kernel.bin 到 Worker L1    │
│    - noc_xy: (1, 1)                         │
│    - addr: 0x2000 (NCRISC code region)      │
├─────────────────────────────────────────────┤
│ 5. CQ_PREFETCH_CMD_RELAY_LINEAR_H           │
│    - 从 Host 读取 add_kernel.bin            │
│    - size: 8192 bytes                       │
├─────────────────────────────────────────────┤
│ 6. CQ_DISPATCH_CMD_WRITE_LINEAR             │
│    - 写入 add_kernel.bin 到 Worker L1       │
│    - noc_xy: (1, 1)                         │
│    - addr: 0x8000 (TRISC code region)       │
├─────────────────────────────────────────────┤
│ 7. CQ_DISPATCH_CMD_WRITE_PACKED             │
│    - 写入运行时参数到 Worker L1              │
│    - {num_tiles = 1024}                     │
├─────────────────────────────────────────────┤
│ 8. CQ_DISPATCH_CMD_GO                       │
│    - 启动 Worker Core (1, 1)                │
├─────────────────────────────────────────────┤
│ 9. CQ_DISPATCH_CMD_WAIT                     │
│    - 等待 Worker 完成                        │
│    - sem_addr: worker_done_sem              │
└─────────────────────────────────────────────┘
```

### 5.3 时序图

```
Host      Hugepage    Prefetcher    Dispatcher    Worker       DRAM
 │            │            │            │           │            │
 │ 1. 写入命令│            │            │           │            │
 ├───────────>│            │            │           │            │
 │            │            │            │           │            │
 │ 2. 通知    │            │            │           │            │
 ├────────────┼───────────>│            │           │            │
 │            │            │            │           │            │
 │            │ 3. PCIe 读取            │           │            │
 │            │<───────────┤            │           │            │
 │            │            │            │           │            │
 │            │            │ 4. NoC 写入│           │            │
 │            │            ├───────────>│           │            │
 │            │            │            │           │            │
 │            │            │            │ 5. 写 L1  │            │
 │            │            │            ├──────────>│            │
 │            │            │            │           │            │
 │            │            │            │ 6. GO信号 │            │
 │            │            │            ├──────────>│            │
 │            │            │            │           │            │
 │            │            │            │           │ 7. 读数据  │
 │            │            │            │           │<───────────┤
 │            │            │            │           │            │
 │            │            │            │           │ 8. 执行    │
 │            │            │            │           │   计算     │
 │            │            │            │           │            │
 │            │            │            │           │ 9. 写结果  │
 │            │            │            │           ├───────────>│
 │            │            │            │           │            │
 │            │            │            │ 10. 完成  │            │
 │            │            │            │<──────────┤            │
 │            │            │            │           │            │
 │            │ 11. PCIe 写入(completion)           │            │
 │<───────────┼────────────┼────────────┤           │            │
 │            │            │            │           │            │
 │ 12. 返回   │            │            │           │            │
 │            │            │            │           │            │
```

---

## 6. 代码实现分析

### 6.1 Prefetch Kernel 关键代码

```cpp
// cq_prefetch.cpp

// 全局变量
static uint32_t pcie_read_ptr = pcie_base;  // Host Hugepage 读取指针
static uint32_t downstream_data_ptr = downstream_cb_base;  // 转发 buffer

// 主循环
void kernel_main() {
    // 初始化 NoC
    noc_init();

    while (true) {
        // 1. 从 Host 读取命令头
        CQPrefetchCmd cmd;
        noc_async_read(
            get_noc_addr(pcie_read_ptr),  // Host 地址（通过 PCIe）
            (uint32_t)&cmd,                // 本地 L1 地址
            sizeof(CQPrefetchCmd)
        );
        noc_async_read_barrier();

        pcie_read_ptr += sizeof(CQPrefetchCmd);

        // 2. 处理命令
        switch (cmd.base.cmd_id) {
            case CQ_PREFETCH_CMD_RELAY_LINEAR_H: {
                // 从 Host 读取数据
                uint32_t size = cmd.relay_linear.length;
                uint32_t src_addr = cmd.relay_linear.addr;

                // 读取到本地 buffer
                noc_async_read(
                    get_noc_addr(pcie_base + src_addr),
                    downstream_data_ptr,
                    size
                );
                noc_async_read_barrier();

                // 转发到 Dispatcher
                relay_to_dispatcher(downstream_data_ptr, size);

                break;
            }

            case CQ_PREFETCH_CMD_RELAY_LINEAR: {
                // 从 DRAM 读取数据
                uint32_t dram_noc_xy = cmd.relay_linear.noc_xy_addr;
                uint32_t dram_addr = cmd.relay_linear.addr;
                uint32_t size = cmd.relay_linear.length;

                // 读取到本地 buffer
                noc_async_read(
                    get_noc_addr_xy(dram_noc_xy, dram_addr),
                    downstream_data_ptr,
                    size
                );
                noc_async_read_barrier();

                // 转发到 Dispatcher
                relay_to_dispatcher(downstream_data_ptr, size);

                break;
            }

            case CQ_PREFETCH_CMD_STALL: {
                // 等待 Dispatcher 完成
                noc_semaphore_wait(my_downstream_cb_sem_id, 1);
                break;
            }

            case CQ_PREFETCH_CMD_TERMINATE: {
                return;  // 退出
            }
        }
    }
}

// 转发数据到 Dispatcher
void relay_to_dispatcher(uint32_t local_addr, uint32_t size) {
    // 等待 Dispatcher buffer 有空间
    noc_semaphore_wait(my_downstream_cb_sem_id, 1);

    // 通过 NoC 写入到 Dispatcher
    uint64_t dispatcher_addr = get_noc_addr_xy(
        downstream_noc_xy,
        dispatch_cb_base + dispatch_cb_wr_ptr
    );

    noc_async_write(local_addr, dispatcher_addr, size);
    noc_async_write_barrier();

    // 通知 Dispatcher 数据准备好
    noc_semaphore_inc(
        downstream_noc_xy,
        downstream_cb_sem_id,  // page_ready
        1
    );

    // 更新写入指针
    dispatch_cb_wr_ptr = (dispatch_cb_wr_ptr + size) % dispatch_cb_size;
}
```

### 6.2 Dispatch Kernel 关键代码

```cpp
// cq_dispatch.cpp

// 主循环
void kernel_main() {
    noc_init();

    while (true) {
        // 1. 等待 Prefetcher 的数据
        noc_semaphore_wait(my_dispatch_cb_sem_id, 1);

        // 2. 从 dispatch buffer 读取命令
        CQDispatchCmd cmd;
        uint32_t cmd_ptr = dispatch_cb_base + dispatch_cb_rd_ptr;
        memcpy(&cmd, (void*)cmd_ptr, sizeof(CQDispatchCmd));

        cmd_ptr += sizeof(CQDispatchCmd);

        // 3. 处理命令
        switch (cmd.base.cmd_id) {
            case CQ_DISPATCH_CMD_WRITE_LINEAR: {
                // 线性写入 Worker L1
                uint32_t worker_noc_xy = cmd.write_linear.noc_xy_addr;
                uint32_t worker_addr = cmd.write_linear.addr;
                uint32_t size = cmd.write_linear.length;

                // 数据跟随在命令后面
                uint32_t data_ptr = cmd_ptr;

                // 通过 NoC 写入 Worker
                uint64_t worker_addr_full = get_noc_addr_xy(
                    worker_noc_xy,
                    worker_addr
                );

                noc_async_write(data_ptr, worker_addr_full, size);
                noc_async_write_barrier();

                cmd_ptr += size;
                break;
            }

            case CQ_DISPATCH_CMD_WRITE_PACKED: {
                // 批量写入多个 Workers
                uint32_t num_dests = cmd.write_packed.num_dests;
                uint32_t size = cmd.write_packed.size;
                uint32_t data_ptr = cmd_ptr;

                // 读取目标列表
                for (uint32_t i = 0; i < num_dests; i++) {
                    PackedSubCmd subcmd;
                    memcpy(&subcmd, (void*)cmd_ptr, sizeof(PackedSubCmd));
                    cmd_ptr += sizeof(PackedSubCmd);

                    // 写入到每个目标
                    uint64_t dest_addr = get_noc_addr_xy(
                        subcmd.noc_xy,
                        subcmd.addr
                    );

                    noc_async_write(data_ptr, dest_addr, size);
                }

                noc_async_write_barrier();
                cmd_ptr += size;
                break;
            }

            case CQ_DISPATCH_CMD_GO: {
                // 发送 GO 信号启动 Workers
                uint32_t num_workers = cmd.go.num_workers;

                // Multicast GO signal
                for (uint32_t i = 0; i < num_workers; i++) {
                    CoreCoord core = cmd.go.worker_cores[i];

                    // 写入 GO signal 到 Worker 的特定地址
                    uint32_t go_signal = 1;
                    uint64_t signal_addr = get_noc_addr_xy(
                        core.noc_xy,
                        GO_SIGNAL_ADDR
                    );

                    noc_async_write(
                        (uint32_t)&go_signal,
                        signal_addr,
                        sizeof(go_signal)
                    );
                }

                noc_async_write_barrier();
                break;
            }

            case CQ_DISPATCH_CMD_WAIT: {
                // 等待 Workers 完成
                uint32_t wait_addr = cmd.wait.addr;
                uint32_t wait_count = cmd.wait.count;

                // 轮询信号量
                volatile uint32_t* sem_ptr = (volatile uint32_t*)wait_addr;
                while (*sem_ptr < wait_count) {
                    // Spin wait
                }

                break;
            }

            case CQ_DISPATCH_CMD_TERMINATE: {
                return;  // 退出
            }
        }

        // 4. 更新读取指针
        dispatch_cb_rd_ptr = (cmd_ptr - dispatch_cb_base) % dispatch_cb_size;

        // 5. 通知 Prefetcher 可以继续
        noc_semaphore_inc(
            prefetcher_noc_xy,
            upstream_dispatch_cb_sem_id,  // page_done
            1
        );
    }
}
```

### 6.3 Worker Kernel 示例

```cpp
// reader_kernel.cpp - 运行在 Worker Core 的 BRISC

void kernel_main() {
    // 1. 等待 GO 信号
    volatile uint32_t* go_signal_ptr = (volatile uint32_t*)GO_SIGNAL_ADDR;
    while (*go_signal_ptr == 0) {
        // Spin wait
    }

    // 2. 读取运行时参数
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t dram_addr = get_arg_val<uint32_t>(1);

    // 3. 从 DRAM 读取数据到 L1 Circular Buffer
    constexpr uint32_t cb_id = tt::CB::c_in0;
    uint32_t l1_write_addr = get_write_ptr(cb_id);

    for (uint32_t i = 0; i < num_tiles; i++) {
        // 通过 NoC 从 DRAM 读取
        noc_async_read(
            dram_addr,           // DRAM 地址
            l1_write_addr,       // L1 Circular Buffer
            tile_size
        );

        noc_async_read_barrier();

        // 通知 Compute kernel 数据准备好
        cb_push_back(cb_id, 1);

        dram_addr += tile_size;
        l1_write_addr = get_write_ptr(cb_id);  // 下一个 tile
    }

    // 4. 通知 Dispatcher 完成
    noc_semaphore_inc(
        dispatcher_noc_xy,
        worker_done_sem_id,
        1
    );
}
```

```cpp
// add_kernel.cpp - 运行在 Worker Core 的 TRISC1

void kernel_main() {
    constexpr uint32_t cb_in0 = tt::CB::c_in0;
    constexpr uint32_t cb_in1 = tt::CB::c_in1;
    constexpr uint32_t cb_out = tt::CB::c_out0;

    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    for (uint32_t i = 0; i < num_tiles; i++) {
        // 1. 等待输入数据准备好
        cb_wait_front(cb_in0, 1);
        cb_wait_front(cb_in1, 1);

        // 2. Acquire output buffer
        cb_reserve_back(cb_out, 1);

        // 3. TRISC0 自动 Unpack (由 TRISC0 执行)
        // 4. 执行加法运算 (TRISC1 执行)
        add_tiles(cb_in0, cb_in1, 0, 0, 0);  // dst = src0 + src1

        // 5. TRISC2 自动 Pack (由 TRISC2 执行)

        // 6. 释放 buffers
        cb_pop_front(cb_in0, 1);
        cb_pop_front(cb_in1, 1);
        cb_push_back(cb_out, 1);
    }
}
```

---

## 7. 性能考虑

### 7.1 瓶颈分析

| 潜在瓶颈 | 影响 | 缓解措施 |
|---------|------|---------|
| **PCIe 带宽** | Host ↔ Device 数据传输 | - 使用 Hugepage 减少开销<br>- 批量传输<br>- Prefetcher 预取 |
| **NoC 带宽** | Prefetcher → Dispatcher → Workers | - 双 NoC 并行使用<br>- Multicast 写入<br>- 减少小包传输 |
| **Prefetcher 处理** | 命令读取速度 | - BRISC + NCRISC 并行<br>- Ring buffer 减少等待 |
| **Dispatcher 处理** | 命令分发速度 | - 分布式 Dispatcher<br>- WRITE_PACKED 批量写入 |
| **DRAM 访问** | Worker 读写数据 | - L1 作为 cache<br>- 双缓冲重叠计算和传输 |

### 7.2 优化技巧

**1. 使用 Trace 模式**

Trace 模式可以将整个命令序列录制下来，重放时无需重新构建命令：

```cpp
// 第一次执行 - 录制 trace
uint32_t trace_id = BeginTraceCapture(device, cq_id);
EnqueueProgram(cq, program);
EndTraceCapture(device, cq_id, trace_id);

// 后续执行 - 直接重放 trace (快得多)
EnqueueTrace(cq, trace_id);
```

**Trace 的优势：**
- Host 只需发送一个 `CQ_PREFETCH_CMD_EXEC_BUF` 命令
- Prefetcher 从设备 DRAM 读取整个命令序列（比 PCIe 快）
- 减少 Host CPU 开销

**2. 批量操作**

```cpp
// 差 - 多次小写入
for (int i = 0; i < 100; i++) {
    cmd.add_dispatch_write_linear(core, addr + i * size, data[i], size);
}

// 好 - 一次大写入
cmd.add_dispatch_write_linear(core, addr, all_data, total_size);
```

**3. Multicast 写入**

当多个 Worker Cores 需要相同的数据时（例如 weights），使用 multicast：

```cpp
// Dispatcher 支持 multicast
cmd.add_dispatch_write_linear_multicast(
    start_core,  // 起始坐标
    end_core,    // 结束坐标
    addr,
    data,
    size
);

// NoC 会自动将数据广播到矩形区域内的所有核心
// 比逐个写入快得多
```

**4. 双缓冲**

在 Worker Cores 使用双缓冲重叠计算和数据传输：

```cpp
// Circular Buffer 配置为 2 个 pages
CircularBufferConfig cb_config = CircularBufferConfig(
    2 * tile_size,  // 2 pages
    {{cb_id, tile_format}}
);

// Reader kernel
for (tile : tiles) {
    // Buffer 0: NCRISC 读取数据
    // Buffer 1: TRISC 处理上一个 tile
    // 两个操作并行
}
```

### 7.3 性能数据

根据 PCIExpressTile 文档，以下是实测性能：

| 操作 | 吞吐量 | 延迟 |
|------|--------|------|
| Host → Device (DMA) | 24 GB/s | ≥ 1342 ns |
| Device → Host (DMA) | 11 GB/s | ≥ 1052 ns |
| Device-initiated Read | 24.5 GB/s | ≥ 690 ns |
| Device-initiated Write | 26.4 GB/s | ≥ 260 ns |

**关键观察：**
- **Device-initiated 写入最快**：用于将结果写回 Host
- **PCIe 读取比写入慢**：Prefetcher 预取可以隐藏延迟
- **NoC 内部传输更快**：Prefetcher → Dispatcher → Workers 都在设备内部

---

## 8. 常见疑问解答

### Q1: Prefetcher 和 Dispatcher 为什么不合并成一个？

**A:** 分离设计有多个优势：

1. **流水线并行**：Prefetcher 预取下一批数据时，Dispatcher 处理当前批次
2. **职责分离**：Prefetcher 处理 PCIe/DRAM 读取，Dispatcher 处理 Worker 写入
3. **扩展性**：可以有多个 Dispatcher 并行处理（分布式 dispatch）
4. **性能**：两个核心可以真正并行工作，而不是时分复用

### Q2: 为什么需要 Hugepage？普通内存不行吗？

**A:** Hugepage 的关键优势：

1. **物理连续**：DMA 引擎可以直接访问大块数据，无需 scatter-gather
2. **TLB 效率**：减少 TLB miss，提高地址转换效率
3. **PCIe 效率**：减少地址转换开销，提高传输效率

普通 4KB 页面的问题：
- 物理地址不连续，DMA 需要多次传输
- TLB 条目不够，频繁 miss
- PCIe 传输效率低

### Q3: Worker Core 的 5 个 RISC-V 能否执行其他代码？

**A:** 可以，但有限制：

- **BRISC + NCRISC**：运行 Data Movement kernels，也可以运行 Prefetcher/Dispatcher
- **TRISC0/1/2**：运行 Compute kernels，也可以不使用 Tensix 协处理器

但是：
- 5 个核心**必须协同工作**，不能相互干扰
- 每个核心只有 2-4KB 本地 RAM，代码不能太大
- L1 是共享的，需要仔细管理内存

### Q4: Prefetcher/Dispatcher 的固件何时加载？

**A:** 在设备初始化时：

```
Device::Initialize()
  │
  ├─ 1. 配置 PCIe Tile (BAR mapping, DMA engines)
  │
  ├─ 2. 加载 dispatch kernels
  │    │
  │    ├─ 2.1 编译 cq_prefetch.cpp → prefetch.bin
  │    │
  │    ├─ 2.2 编译 cq_dispatch.cpp → dispatch.bin
  │    │
  │    ├─ 2.3 通过 UMD 写入到对应 Tensix Core 的 L1
  │    │     write_to_device(prefetcher_core, prefetch.bin, ...)
  │    │     write_to_device(dispatcher_core, dispatch.bin, ...)
  │    │
  │    └─ 2.4 De-assert RISC 复位，启动 Prefetcher/Dispatcher
  │          deassert_risc_resets(prefetcher_core)
  │          deassert_risc_resets(dispatcher_core)
  │
  └─ 3. 初始化 Hugepage (SystemMemoryManager)
```

之后，Prefetcher 和 Dispatcher 就一直运行，等待命令。

### Q5: 如何调试 Prefetcher/Dispatcher？

**A:** 几种方法：

1. **日志输出**：

```cpp
// 在 kernel 中添加 debug print
DPRINT << "Prefetcher: received cmd " << cmd.cmd_id << ENDL();
```

2. **环境变量**：

```bash
export TT_METAL_LOGGER_LEVEL=DEBUG
export TT_METAL_LOGGER_TYPES=Dispatch
```

3. **Watcher 功能**：

```cpp
// tt-metal 的 watcher 可以监控设备状态
watcher.watch_core(prefetcher_core);
```

4. **Tracy Profiler**：

```bash
export TRACY_NO_INVARIANT_CHECK=1
# Tracy 可以可视化命令流
```

### Q6: 能否有多个 Prefetcher/Dispatcher？

**A:** 可以！tt-metal 支持：

- **多个 Command Queue**：每个 CQ 有独立的 Prefetcher/Dispatcher
- **分布式 Dispatcher**：多个 Dispatcher 并行处理不同的 Worker 区域

```
                      ┌─────────────────┐
                      │  Prefetcher     │
                      └────────┬────────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
         ┌──────▼───┐   ┌─────▼────┐   ┌────▼─────┐
         │Dispatch 0│   │Dispatch 1│   │Dispatch 2│
         └──────┬───┘   └─────┬────┘   └────┬─────┘
                │              │              │
         Workers[0:31]  Workers[32:63] Workers[64:79]
```

### Q7: Prefetcher 如何知道何时从 Host 读取？

**A:** 通过 **Fetch Queue** 轮询：

```
Host 端:
┌─────────────────────────────────────┐
│  SystemMemoryManager                │
│                                     │
│  fetch_queue_write(size, cq_id) {  │
│    // 写入 fetch queue 指针         │
│    uint32_t wr_ptr = ...;           │
│    write_to_sysmem(                 │
│      fetch_queue_wr_ptr_addr,       │
│      &wr_ptr,                       │
│      sizeof(wr_ptr)                 │
│    );                               │
│  }                                  │
└─────────────────────────────────────┘

设备端 (Prefetcher):
┌─────────────────────────────────────┐
│  cq_prefetch.cpp                    │
│                                     │
│  while (true) {                     │
│    // 轮询 fetch queue 写入指针     │
│    wr_ptr = noc_read(               │
│      pcie_base + fetch_q_wr_ptr     │
│    );                               │
│                                     │
│    if (wr_ptr != rd_ptr) {          │
│      // 有新命令，开始读取           │
│      process_commands();            │
│    }                                │
│  }                                  │
└─────────────────────────────────────┘
```

### Q8: Worker Core 如何知道何时开始执行？

**A:** 通过 **GO Signal**：

```
Dispatcher:
  // 发送 GO signal
  uint32_t go_signal = 1;
  noc_async_write(
    &go_signal,
    get_noc_addr_xy(worker_xy, GO_SIGNAL_ADDR),
    sizeof(go_signal)
  );

Worker (BRISC):
  // 等待 GO signal
  volatile uint32_t* go_ptr = (volatile uint32_t*)GO_SIGNAL_ADDR;
  while (*go_ptr == 0) {
    // Spin wait
  }

  // 收到信号，启动其他 RISC-V 核心
  launch_ncrisc();
  launch_trisc0();
  launch_trisc1();
  launch_trisc2();
```

---

## 参考资料

### 源码位置

```
tt-metal/
├── tt_metal/impl/dispatch/
│   ├── kernels/
│   │   ├── cq_prefetch.cpp           # Prefetcher kernel
│   │   ├── cq_dispatch.cpp           # Dispatcher kernel
│   │   ├── cq_commands.hpp           # 命令定义
│   │   └── cq_common.hpp             # 公共定义
│   ├── hardware_command_queue.cpp    # HWCommandQueue 实现
│   ├── device_command.cpp            # DeviceCommand 实现
│   └── system_memory_manager.cpp     # SystemMemoryManager 实现
│
└── tt_metal/third_party/umd/
    ├── device/chip/
    │   ├── chip.hpp                  # Chip 基类
    │   ├── local_chip.cpp            # PCIe 连接实现
    │   └── simulation_chip.cpp       # 模拟器实现
    └── device/pcie/
        └── pci_device.hpp            # PCIe 设备接口
```

### ISA 文档

```
tt-isa-documentation/WormholeB0/
├── README.md                         # 芯片整体架构
├── PCIExpressTile/README.md          # PCIe Tile 详解
├── TensixTile/
│   ├── README.md                     # Tensix Core 架构
│   └── BabyRISCV/README.md           # RISC-V 核心详解
├── NoC/README.md                     # NoC 网络
└── DRAMTile/README.md                # DRAM Tile
```

### 相关文档

- [01-repo-structure.md](./01-repo-structure.md) - tt-metal 仓库结构
- [02-device-detection.md](./02-device-detection.md) - 设备检测流程
- [03-umd-interface.md](./03-umd-interface.md) - UMD 接口详解
- [04-command-submission.md](./04-command-submission.md) - 命令提交流程概览
- [05-memory-management.md](./05-memory-management.md) - 内存管理
- [06-simulator-interface.md](./06-simulator-interface.md) - 模拟器接口

---

*创建时间: 2025-02*
*最后更新: 2025-02*
