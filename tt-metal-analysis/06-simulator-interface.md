# 模拟器接口设计

## 概述

本文档汇总前面的分析，提出实现 Tenstorrent 模拟器的具体设计方案。目标是让 tt-metal 能够在没有物理硬件的情况下运行，支持功能验证和开发调试。

## 设计目标

1. **功能正确性**: 正确模拟内存读写、命令执行
2. **最小侵入性**: 尽量不修改 tt-metal 代码
3. **可扩展性**: 支持逐步添加更复杂的模拟功能
4. **调试友好**: 提供丰富的日志和调试接口

## 集成方案对比

### 方案一: TTSIM 接口 (推荐)

```
优点:
✅ tt-metal 原生支持
✅ 接口简单 (6 个函数)
✅ 无需修改 tt-metal 代码
✅ 通过环境变量启用

缺点:
❌ 需要实现动态库
❌ 需要理解 tile-based 内存访问模式

实现难度: 中等
```

### 方案二: Mock 模式扩展

```
优点:
✅ 现有基础设施
✅ 无需实现完整模拟

缺点:
❌ 不执行实际计算
❌ 只能用于流程测试

实现难度: 低
```

### 方案三: 自定义 Chip 实现

```
优点:
✅ 完全控制所有操作
✅ 可以实现复杂模拟

缺点:
❌ 需要修改 tt-metal 代码
❌ 维护成本高

实现难度: 高
```

### 方案四: QEMU 虚拟 PCIe (同事方案)

```
优点:
✅ 对 tt-metal 完全透明
✅ 可模拟完整硬件行为

缺点:
❌ 实现复杂度极高
❌ 需要内核模块
❌ 性能开销大

实现难度: 极高
```

## 推荐方案: TTSIM 接口实现

### 架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        tt-metal Runtime                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  tt::Cluster                                                        │
│      │                                                              │
│      ▼                                                              │
│  tt::umd::Cluster                                                   │
│      │                                                              │
│      ├── target_type = Simulator                                    │
│      │                                                              │
│      ▼                                                              │
│  TTSimChip                                                          │
│      │                                                              │
│      ▼                                                              │
│  TTSimTTDevice  ←─────── 加载动态库                                 │
│                                                                     │
└──────────────────────┬──────────────────────────────────────────────┘
                       │ dlopen("libttsim.so")
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    libttsim.so (自定义实现)                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  导出函数:                                                          │
│  ├── libttsim_init()                                               │
│  ├── libttsim_exit()                                               │
│  ├── libttsim_pci_config_rd32()                                    │
│  ├── libttsim_tile_rd_bytes()                                      │
│  ├── libttsim_tile_wr_bytes()                                      │
│  └── libttsim_clock()                                              │
│                                                                     │
│  内部实现:                                                          │
│  ├── SimulatorMemory (内存模型)                                    │
│  ├── SimulatorCore (核心模拟)                                      │
│  └── SimulatorKernel (kernel 执行)                                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 接口定义

```cpp
// libttsim.h - 模拟器接口头文件

#ifndef LIBTTSIM_H
#define LIBTTSIM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// 初始化模拟器
// 在任何其他操作之前调用
void libttsim_init(void);

// 清理模拟器
// 程序退出时调用
void libttsim_exit(void);

// 读取 PCI 配置空间
// 用于设备检测和初始化
// bus_device_function: PCIe BDF 编码
// offset: 配置空间偏移
// 返回: 配置值
uint32_t libttsim_pci_config_rd32(uint32_t bus_device_function, uint32_t offset);

// 从设备内存读取数据
// x, y: 核心坐标
// addr: 内存地址
// p: 目标缓冲区
// size: 读取字节数
void libttsim_tile_rd_bytes(uint32_t x, uint32_t y, uint64_t addr,
                             void* p, uint32_t size);

// 向设备内存写入数据
// x, y: 核心坐标
// addr: 内存地址
// p: 源数据
// size: 写入字节数
void libttsim_tile_wr_bytes(uint32_t x, uint32_t y, uint64_t addr,
                             const void* p, uint32_t size);

// 推进模拟器时钟
// n_clocks: 时钟周期数
void libttsim_clock(uint32_t n_clocks);

#ifdef __cplusplus
}
#endif

#endif // LIBTTSIM_H
```

### 完整实现框架

```cpp
// libttsim.cpp - 模拟器实现

#include "libttsim.h"
#include <cstring>
#include <map>
#include <vector>
#include <mutex>
#include <iostream>
#include <fstream>

namespace {

//=============================================================================
// 配置常量
//=============================================================================

// Wormhole B0 配置
constexpr uint32_t GRID_SIZE_X = 10;  // 包括 DRAM 和 ETH 核心
constexpr uint32_t GRID_SIZE_Y = 12;
constexpr uint32_t TENSIX_START_X = 1;
constexpr uint32_t TENSIX_START_Y = 1;
constexpr uint32_t TENSIX_END_X = 8;
constexpr uint32_t TENSIX_END_Y = 10;

constexpr size_t L1_SIZE = 1 * 1024 * 1024;        // 1 MB
constexpr size_t DRAM_SIZE = 8ULL * 1024 * 1024 * 1024;  // 8 GB
constexpr uint32_t NUM_DRAM_CHANNELS = 12;

// PCI 配置空间常量
constexpr uint32_t TENSTORRENT_VENDOR_ID = 0x1E52;
constexpr uint32_t WORMHOLE_DEVICE_ID = 0x401E;

//=============================================================================
// 核心坐标
//=============================================================================

struct CoreCoord {
    uint32_t x;
    uint32_t y;

    bool operator<(const CoreCoord& other) const {
        return (x < other.x) || (x == other.x && y < other.y);
    }

    bool operator==(const CoreCoord& other) const {
        return x == other.x && y == other.y;
    }
};

//=============================================================================
// 内存模型
//=============================================================================

class SimulatorMemory {
public:
    SimulatorMemory() {
        // 初始化 DRAM (延迟分配)
        dram_channels_.resize(NUM_DRAM_CHANNELS);
    }

    // L1 内存访问
    void write_l1(uint32_t x, uint32_t y, uint64_t addr,
                  const void* data, size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);

        CoreCoord core{x, y};
        auto& mem = get_or_create_l1(core);

        if (addr + size > L1_SIZE) {
            log_error("L1 write out of bounds: core (%u,%u) addr 0x%lx size %zu",
                      x, y, addr, size);
            return;
        }

        std::memcpy(mem.data() + addr, data, size);
        log_debug("L1 write: core (%u,%u) addr 0x%lx size %zu", x, y, addr, size);
    }

    void read_l1(uint32_t x, uint32_t y, uint64_t addr,
                 void* data, size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);

        CoreCoord core{x, y};
        auto& mem = get_or_create_l1(core);

        if (addr + size > L1_SIZE) {
            log_error("L1 read out of bounds: core (%u,%u) addr 0x%lx size %zu",
                      x, y, addr, size);
            std::memset(data, 0, size);
            return;
        }

        std::memcpy(data, mem.data() + addr, size);
        log_debug("L1 read: core (%u,%u) addr 0x%lx size %zu", x, y, addr, size);
    }

    // DRAM 访问 (通过 DRAM 核心坐标)
    void write_dram(uint32_t channel, uint64_t addr,
                    const void* data, size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);

        if (channel >= NUM_DRAM_CHANNELS) {
            log_error("Invalid DRAM channel: %u", channel);
            return;
        }

        auto& mem = get_or_create_dram(channel);
        if (addr + size > DRAM_SIZE) {
            log_error("DRAM write out of bounds: ch %u addr 0x%lx size %zu",
                      channel, addr, size);
            return;
        }

        std::memcpy(mem.data() + addr, data, size);
        log_debug("DRAM write: ch %u addr 0x%lx size %zu", channel, addr, size);
    }

    void read_dram(uint32_t channel, uint64_t addr,
                   void* data, size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);

        if (channel >= NUM_DRAM_CHANNELS) {
            log_error("Invalid DRAM channel: %u", channel);
            std::memset(data, 0, size);
            return;
        }

        auto& mem = get_or_create_dram(channel);
        if (addr + size > DRAM_SIZE) {
            log_error("DRAM read out of bounds: ch %u addr 0x%lx size %zu",
                      channel, addr, size);
            std::memset(data, 0, size);
            return;
        }

        std::memcpy(data, mem.data() + addr, size);
        log_debug("DRAM read: ch %u addr 0x%lx size %zu", channel, addr, size);
    }

    void dump_stats() {
        std::lock_guard<std::mutex> lock(mutex_);
        size_t l1_total = l1_memory_.size() * L1_SIZE;
        size_t dram_total = 0;
        for (const auto& ch : dram_channels_) {
            dram_total += ch.size();
        }
        log_info("Memory stats: L1 cores=%zu (%zu MB), DRAM allocated=%zu MB",
                 l1_memory_.size(), l1_total / (1024*1024),
                 dram_total / (1024*1024));
    }

private:
    std::mutex mutex_;
    std::map<CoreCoord, std::vector<uint8_t>> l1_memory_;
    std::vector<std::vector<uint8_t>> dram_channels_;

    std::vector<uint8_t>& get_or_create_l1(CoreCoord core) {
        auto it = l1_memory_.find(core);
        if (it == l1_memory_.end()) {
            auto [new_it, _] = l1_memory_.emplace(core, std::vector<uint8_t>(L1_SIZE, 0));
            return new_it->second;
        }
        return it->second;
    }

    std::vector<uint8_t>& get_or_create_dram(uint32_t channel) {
        auto& mem = dram_channels_[channel];
        if (mem.empty()) {
            // 延迟分配 - 可以根据需要调整大小
            // 这里先分配 1GB，按需增长
            mem.resize(1ULL * 1024 * 1024 * 1024, 0);
        }
        return mem;
    }

    void log_debug(const char* fmt, ...) {
        // 可选: 启用详细日志
#ifdef TTSIM_DEBUG
        va_list args;
        va_start(args, fmt);
        fprintf(stderr, "[TTSIM DEBUG] ");
        vfprintf(stderr, fmt, args);
        fprintf(stderr, "\n");
        va_end(args);
#endif
    }

    void log_info(const char* fmt, ...) {
        va_list args;
        va_start(args, fmt);
        fprintf(stderr, "[TTSIM INFO] ");
        vfprintf(stderr, fmt, args);
        fprintf(stderr, "\n");
        va_end(args);
    }

    void log_error(const char* fmt, ...) {
        va_list args;
        va_start(args, fmt);
        fprintf(stderr, "[TTSIM ERROR] ");
        vfprintf(stderr, fmt, args);
        fprintf(stderr, "\n");
        va_end(args);
    }
};

//=============================================================================
// 地址映射
//=============================================================================

// 判断坐标是否为 DRAM 核心
bool is_dram_core(uint32_t x, uint32_t y) {
    // Wormhole DRAM 核心在 x=0 或 x=9
    return (x == 0 || x == 9) && (y >= 0 && y < 6);
}

// DRAM 核心坐标到通道映射
uint32_t dram_core_to_channel(uint32_t x, uint32_t y) {
    if (x == 0) {
        return y;  // 通道 0-5
    } else {  // x == 9
        return 6 + y;  // 通道 6-11
    }
}

// 判断坐标是否为 Tensix 核心
bool is_tensix_core(uint32_t x, uint32_t y) {
    return x >= TENSIX_START_X && x <= TENSIX_END_X &&
           y >= TENSIX_START_Y && y <= TENSIX_END_Y;
}

//=============================================================================
// 全局状态
//=============================================================================

SimulatorMemory* g_memory = nullptr;
uint64_t g_clock_cycles = 0;
bool g_initialized = false;

} // anonymous namespace

//=============================================================================
// 导出函数实现
//=============================================================================

extern "C" {

void libttsim_init(void) {
    if (g_initialized) {
        return;
    }

    fprintf(stderr, "[TTSIM] Initializing Tenstorrent Simulator\n");
    fprintf(stderr, "[TTSIM] Configuration: Wormhole B0\n");
    fprintf(stderr, "[TTSIM] Grid: %ux%u, L1: %zu KB, DRAM: %u channels\n",
            GRID_SIZE_X, GRID_SIZE_Y, L1_SIZE/1024, NUM_DRAM_CHANNELS);

    g_memory = new SimulatorMemory();
    g_clock_cycles = 0;
    g_initialized = true;
}

void libttsim_exit(void) {
    if (!g_initialized) {
        return;
    }

    fprintf(stderr, "[TTSIM] Shutting down simulator\n");
    fprintf(stderr, "[TTSIM] Total clock cycles: %lu\n", g_clock_cycles);

    if (g_memory) {
        g_memory->dump_stats();
        delete g_memory;
        g_memory = nullptr;
    }

    g_initialized = false;
}

uint32_t libttsim_pci_config_rd32(uint32_t bus_device_function, uint32_t offset) {
    // 返回模拟的 PCI 配置值
    switch (offset) {
        case 0x00:  // Vendor ID + Device ID
            return (WORMHOLE_DEVICE_ID << 16) | TENSTORRENT_VENDOR_ID;

        case 0x08:  // Revision ID + Class Code
            return 0x12000000;  // 处理器类设备

        case 0x2C:  // Subsystem Vendor ID + Subsystem ID
            return (0x0001 << 16) | TENSTORRENT_VENDOR_ID;

        default:
            return 0xFFFFFFFF;
    }
}

void libttsim_tile_rd_bytes(uint32_t x, uint32_t y, uint64_t addr,
                             void* p, uint32_t size) {
    if (!g_initialized || !g_memory) {
        std::memset(p, 0, size);
        return;
    }

    if (is_dram_core(x, y)) {
        uint32_t channel = dram_core_to_channel(x, y);
        g_memory->read_dram(channel, addr, p, size);
    } else {
        g_memory->read_l1(x, y, addr, p, size);
    }
}

void libttsim_tile_wr_bytes(uint32_t x, uint32_t y, uint64_t addr,
                             const void* p, uint32_t size) {
    if (!g_initialized || !g_memory) {
        return;
    }

    if (is_dram_core(x, y)) {
        uint32_t channel = dram_core_to_channel(x, y);
        g_memory->write_dram(channel, addr, p, size);
    } else {
        g_memory->write_l1(x, y, addr, p, size);
    }
}

void libttsim_clock(uint32_t n_clocks) {
    g_clock_cycles += n_clocks;

    // 这里可以添加:
    // - Kernel 执行模拟
    // - 时序事件处理
    // - 同步信号更新
}

} // extern "C"
```

### 构建脚本

```bash
#!/bin/bash
# build_simulator.sh

set -e

# 编译选项
CXX="${CXX:-g++}"
CXXFLAGS="-std=c++17 -O2 -fPIC -Wall -Wextra"

# 调试版本
if [ "$DEBUG" = "1" ]; then
    CXXFLAGS="$CXXFLAGS -g -DTTSIM_DEBUG"
fi

# 输出目录
OUTPUT_DIR="${OUTPUT_DIR:-build}"
mkdir -p "$OUTPUT_DIR"

echo "Building libttsim.so..."
$CXX $CXXFLAGS -shared -o "$OUTPUT_DIR/libttsim.so" libttsim.cpp

echo "Build complete: $OUTPUT_DIR/libttsim.so"

# 生成 SoC 描述符 (如果不存在)
if [ ! -f "$OUTPUT_DIR/wormhole_b0_sim.yaml" ]; then
    echo "Generating SoC descriptor..."
    cat > "$OUTPUT_DIR/wormhole_b0_sim.yaml" << 'EOF'
# Wormhole B0 模拟器 SoC 描述符
device:
  arch: wormhole_b0

grid:
  x_size: 10
  y_size: 12

tensix_cores:
  start: [1, 1]
  end: [8, 10]

dram_cores:
  - [0, 0]
  - [0, 1]
  - [0, 2]
  - [0, 3]
  - [0, 4]
  - [0, 5]
  - [9, 0]
  - [9, 1]
  - [9, 2]
  - [9, 3]
  - [9, 4]
  - [9, 5]

eth_cores:
  - [0, 6]
  - [0, 7]
  - [9, 6]
  - [9, 7]

l1_size: 1048576  # 1 MB
dram_size: 8589934592  # 8 GB per channel
num_dram_channels: 12
EOF
fi

echo "Setup complete!"
echo ""
echo "To use the simulator, set these environment variables:"
echo "  export TT_METAL_SIMULATOR=$OUTPUT_DIR"
echo "  export ARCH_NAME=wormhole_b0"
```

## 使用方法

### 1. 编译模拟器

```bash
# 编译 Release 版本
./build_simulator.sh

# 编译 Debug 版本 (带详细日志)
DEBUG=1 ./build_simulator.sh
```

### 2. 配置环境

```bash
# 设置模拟器路径
export TT_METAL_SIMULATOR=/path/to/build

# 设置架构 (必需)
export ARCH_NAME=wormhole_b0

# 可选: 使用 Mock 集群描述符
export TT_METAL_MOCK_CLUSTER_DESC_PATH=/path/to/tt-metal/tt_metal/third_party/umd/tests/cluster_descriptor_examples/wormhole_N150.yaml
```

### 3. 运行测试

```python
#!/usr/bin/env python3
# test_simulator.py

import os
os.environ['TT_METAL_SIMULATOR'] = '/path/to/build'
os.environ['ARCH_NAME'] = 'wormhole_b0'

try:
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
    print(f"Successfully initialized device: {device}")

    # 简单测试
    import torch
    x = torch.tensor([1.0, 2.0, 3.0], device=device)
    y = x * 2
    print(f"Computation result: {y}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
```

## 扩展功能

### 1. Kernel 执行模拟

```cpp
// 简单的 kernel 执行模拟框架
class KernelSimulator {
public:
    // 执行 element-wise 操作
    void execute_eltwise_binary(
        CoreCoord core,
        uint64_t input_a_addr, uint64_t input_b_addr,
        uint64_t output_addr,
        size_t num_elements,
        std::function<float(float, float)> op)
    {
        std::vector<float> a(num_elements), b(num_elements), c(num_elements);

        // 读取输入
        g_memory->read_l1(core.x, core.y, input_a_addr, a.data(), num_elements * sizeof(float));
        g_memory->read_l1(core.x, core.y, input_b_addr, b.data(), num_elements * sizeof(float));

        // 执行计算
        for (size_t i = 0; i < num_elements; i++) {
            c[i] = op(a[i], b[i]);
        }

        // 写入输出
        g_memory->write_l1(core.x, core.y, output_addr, c.data(), num_elements * sizeof(float));
    }

    // 执行矩阵乘法
    void execute_matmul(
        CoreCoord core,
        uint64_t a_addr, uint64_t b_addr, uint64_t c_addr,
        size_t M, size_t N, size_t K)
    {
        // 实现矩阵乘法逻辑
        // ...
    }
};
```

### 2. 命令解析器

```cpp
// 解析和执行 dispatch 命令
class CommandParser {
public:
    void process_command(const void* cmd_data, size_t size) {
        const auto* base = static_cast<const CQDispatchBaseCmd*>(cmd_data);

        switch (base->cmd_id) {
            case CQ_DISPATCH_CMD_WRITE_LINEAR:
                handle_write_linear(cmd_data);
                break;

            case CQ_DISPATCH_CMD_WAIT:
                handle_wait(cmd_data);
                break;

            case CQ_DISPATCH_CMD_GO:
                handle_go(cmd_data);
                break;

            // ... 其他命令
        }
    }

private:
    void handle_write_linear(const void* cmd_data) {
        const auto* cmd = static_cast<const CQDispatchCmd*>(cmd_data);
        // 解析目标地址
        auto [x, y, addr] = decode_noc_addr(cmd->write_linear.noc_xy_addr);
        size_t size = cmd->write_linear.length;

        // 数据紧跟命令之后
        const void* data = reinterpret_cast<const uint8_t*>(cmd_data) + sizeof(CQDispatchCmd);

        // 写入内存
        g_memory->write_l1(x, y, addr, data, size);
    }

    void handle_wait(const void* cmd_data) {
        // 等待命令 - 在模拟器中可以立即完成
        // 或者实现真正的同步逻辑
    }

    void handle_go(const void* cmd_data) {
        // 触发 kernel 执行
        // 在完整模拟器中，这里会启动 kernel 模拟
    }
};
```

### 3. 调试接口

```cpp
// 调试功能
class SimulatorDebug {
public:
    // 内存转储
    void dump_l1(CoreCoord core, uint64_t start, uint64_t end,
                 const char* filename) {
        std::ofstream file(filename, std::ios::binary);
        size_t size = end - start;
        std::vector<uint8_t> data(size);
        g_memory->read_l1(core.x, core.y, start, data.data(), size);
        file.write(reinterpret_cast<char*>(data.data()), size);
    }

    // 内存比较
    bool compare_l1(CoreCoord core, uint64_t addr,
                    const void* expected, size_t size) {
        std::vector<uint8_t> actual(size);
        g_memory->read_l1(core.x, core.y, addr, actual.data(), size);
        return std::memcmp(actual.data(), expected, size) == 0;
    }

    // 设置断点
    void set_breakpoint(CoreCoord core, uint64_t addr) {
        breakpoints_.insert({core, addr});
    }

private:
    std::set<std::pair<CoreCoord, uint64_t>> breakpoints_;
};
```

## 测试策略

### 单元测试

```cpp
// test_memory.cpp
#include <gtest/gtest.h>
#include "libttsim.h"

TEST(SimulatorMemory, BasicL1ReadWrite) {
    libttsim_init();

    uint32_t data_write = 0xDEADBEEF;
    uint32_t data_read = 0;

    libttsim_tile_wr_bytes(1, 1, 0x1000, &data_write, sizeof(data_write));
    libttsim_tile_rd_bytes(1, 1, 0x1000, &data_read, sizeof(data_read));

    EXPECT_EQ(data_write, data_read);

    libttsim_exit();
}

TEST(SimulatorMemory, DRAMReadWrite) {
    libttsim_init();

    std::vector<float> data_write(1024, 3.14f);
    std::vector<float> data_read(1024);

    // DRAM core at (0, 0)
    libttsim_tile_wr_bytes(0, 0, 0x0, data_write.data(),
                           data_write.size() * sizeof(float));
    libttsim_tile_rd_bytes(0, 0, 0x0, data_read.data(),
                           data_read.size() * sizeof(float));

    EXPECT_EQ(data_write, data_read);

    libttsim_exit();
}
```

### 集成测试

```python
# test_integration.py
import pytest
import os

# 确保设置了模拟器
os.environ['TT_METAL_SIMULATOR'] = '/path/to/build'
os.environ['ARCH_NAME'] = 'wormhole_b0'

def test_device_detection():
    """测试设备检测"""
    import tt_metal
    num_devices = tt_metal.GetNumAvailableDevices()
    assert num_devices > 0, "应该检测到模拟设备"

def test_buffer_allocation():
    """测试 buffer 分配"""
    import tt_metal
    device = tt_metal.CreateDevice(0)

    # 分配 L1 buffer
    l1_buffer = tt_metal.CreateBuffer(device, 1024, tt_metal.BufferType.L1)
    assert l1_buffer is not None

    # 分配 DRAM buffer
    dram_buffer = tt_metal.CreateBuffer(device, 1024*1024, tt_metal.BufferType.DRAM)
    assert dram_buffer is not None

    tt_metal.CloseDevice(device)

def test_simple_computation():
    """测试简单计算"""
    import torch
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()

    # 简单张量操作
    a = torch.tensor([1.0, 2.0, 3.0], device=device)
    b = torch.tensor([4.0, 5.0, 6.0], device=device)
    c = a + b

    # 同步并检查结果
    xm.mark_step()
    result = c.cpu().numpy()
    expected = [5.0, 7.0, 9.0]
    assert all(abs(r - e) < 1e-5 for r, e in zip(result, expected))
```

## 已知限制

1. **计算精度**: 基本模拟器不执行实际 kernel 计算，返回空数据
2. **时序**: 不模拟真实的时序和延迟
3. **并发**: 不模拟多核心并行执行
4. **NOC 带宽**: 不模拟 NOC 拥塞和带宽限制
5. **电源管理**: 不模拟功耗状态

## 下一步改进

1. 添加基本 kernel 执行模拟 (element-wise 操作)
2. 实现 Tile-based 内存访问模式
3. 添加 Circular Buffer 模拟
4. 实现同步原语 (semaphore, barrier)
5. 集成与 TTNN 的端到端测试

---

*更新时间: 2025-02*
