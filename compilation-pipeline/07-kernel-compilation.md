# RISC-V 内核编译和 ELF 生成

## 1. 内核编译概述

### 1.1 什么是 Tensix Core 内核

Tensix Core 内核是在 Tenstorrent 硬件上直接执行的底层代码，负责：
- **数据移动**：在不同内存层次间传输数据（DRAM ↔ L1 ↔ NoC）
- **计算执行**：在 Tensix math 引擎上执行矩阵运算、向量运算
- **同步控制**：管理多核之间的通信和同步

每个内核都是一个独立的 C++ 程序，编译为 RISC-V ELF 可执行文件，在特定的 Tensix 处理器上运行。

### 1.2 Tensix 的 5 个 RISC-V 处理器架构

每个 Tensix Core 包含 **5 个独立的 RISC-V 处理器**，各司其职：

```
Tensix Core
├── BRISC (Base RISC)
│   └── 总控制器，协调其他处理器
├── NCRISC (Network-on-Chip RISC)
│   └── 数据移动：DRAM ↔ L1，NoC 通信
├── TRISC0 (Tensor RISC 0)
│   └── 数学运算：Unpack 单元控制
├── TRISC1 (Tensor RISC 1)
│   └── 数学运算：Math 引擎控制
└── TRISC2 (Tensor RISC 2)
    └── 数学运算：Pack 单元控制
```

**处理器特性**：
- **BRISC/NCRISC**：RV32 架构，主要用于控制和数据传输
- **TRISC0/1/2**：RV32 架构，紧密耦合到 Tensix math 引擎
- **所有处理器共享 L1 内存**，但有各自的指令缓存和寄存器文件

### 1.3 内核类型分类

TTKernel Dialect 定义了三种线程类型（`ThreadType` 枚举）：

```cpp
// TTKernelOpsEnums.td
def TTKernel_ThreadTypeNoc : I32EnumAttrCase<"Noc", 0, "noc">;
def TTKernel_ThreadTypeCompute : I32EnumAttrCase<"Compute", 1, "compute">;
def TTKernel_ThreadTypeEthernet : I32EnumAttrCase<"Ethernet", 2, "ethernet">;
```

**1. Compute Kernels（计算内核）**
- **运行处理器**：TRISC0/1/2
- **功能**：执行数学运算（matmul, eltwise, reduce 等）
- **API**：`compute_kernel_api/*`
- **示例操作**：
  - `matmul_tiles()` - 矩阵乘法
  - `add_tiles()` - 元素级加法
  - `tilize()` / `untilize()` - 数据布局转换

**2. Data Movement Kernels（数据移动内核）**
- **运行处理器**：NCRISC
- **功能**：NoC 通信，DRAM/L1 数据传输
- **API**：`dataflow_api.h`
- **示例操作**：
  - `noc_async_read()` - 异步 NoC 读取
  - `noc_async_write()` - 异步 NoC 写入
  - `cb_push_back()` / `cb_pop_front()` - Circular Buffer 操作

**3. Dispatcher/Prefetcher Kernels（固件内核）**
- **运行处理器**：BRISC
- **功能**：内核调度、命令分发
- **文件位置**：`tt_metal/impl/dispatch/kernels/`
- **示例**：`cq_prefetch.cpp`, `cq_dispatch.cpp`

---

## 2. TTKernel Dialect 到 C++ 代码生成

### 2.1 代码生成流程

TTKernel IR 通过两个步骤转换为 C++ 代码：

```
TTKernel IR
    ↓ [TTKernelToEmitC Pass]
EmitC IR
    ↓ [translateKernelFuncToCpp]
C++ 代码
```

**关键文件**：
- **源码**：`lib/Target/TTKernel/TTKernelToCpp.cpp`
- **转换 Pass**：`lib/Conversion/TTKernelToEmitC/TTKernelToEmitC.cpp`

### 2.2 代码生成机制

#### 2.2.1 ScopedModuleHelper - 生成样板代码

`ScopedModuleHelper` 类负责生成内核的头文件包含和初始化代码：

```cpp
// TTKernelToCpp.cpp
class ScopedModuleHelper {
public:
  ScopedModuleHelper(OpBuilder *builder, Location loc, Region *region,
                     ThreadType threadType, StringRef originalSymbolName = "") {
    // 1. 添加通用头文件
    builder->create<emitc::IncludeOp>(loc, "cstdint", /*isStandard=*/true);
    builder->create<emitc::IncludeOp>(loc, "tools/profiler/kernel_profiler.hpp", false);
    builder->create<emitc::IncludeOp>(loc, "internal/firmware_common.h", false);

    // 2. 根据线程类型添加专用头文件
    if (threadType == ThreadType::Noc) {
      builder->create<emitc::IncludeOp>(loc, "api/dataflow/dataflow_api.h", false);
      emitExperimentalLLKs();  // 生成实验性 Low-Level Kernel APIs
    }
    if (threadType == ThreadType::Compute) {
      builder->create<emitc::IncludeOp>(loc, "compute_kernel_api/matmul.h", false);
      builder->create<emitc::IncludeOp>(loc, "compute_kernel_api/eltwise_binary.h", false);
      // ... 更多计算 API 头文件
      emitExperimentalLLKs();
    }
  }
};
```

#### 2.2.2 translateKernelFuncToCpp - 主转换函数

```cpp
LogicalResult translateKernelFuncToCpp(func::FuncOp entry, llvm::raw_ostream &os) {
  // 1. 检查线程类型属性
  if (!entry->hasAttr(ThreadTypeAttr::name)) {
    return failure();
  }
  ThreadType threadType = entry->getAttrOfType<ThreadTypeAttr>(...).getValue();

  // 2. 克隆函数到独立模块（添加样板代码）
  FailureOr<ModuleOp> kernelModule = cloneEntryIntoStandaloneModule(entry, threadType);

  // 3. 使用 EmitC 后端生成 C++ 代码
  return emitc::translateToCpp(*kernelModule, os);
}
```

### 2.3 生成的 C++ 代码结构

#### 示例 1：Compute Kernel 生成的代码

**输入 TTKernel IR**：
```mlir
func.func @ttkernel_compute() attributes {
  ttkernel.thread = #ttkernel.thread<compute>
} {
  %c4 = arith.constant 4 : i32
  %cb_in = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<...>
  %cb_out = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<...>

  ttkernel.untilize_init(%cb_in) : (...)
  ttkernel.untilize_block(%cb_in, %c4, %cb_out) : (...)
  ttkernel.cb_pop_front(%cb_in, %c4) : (...)
  ttkernel.cb_push_back(%cb_out, %c4) : (...)

  func.return
}
```

**生成的 C++ 代码**（`ttkernel_compute.cpp`）：
```cpp
// 1. 头文件（由 ScopedModuleHelper 生成）
#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"
#include "internal/firmware_common.h"
#include "compute_kernel_api/untilize.h"
#include "compute_kernel_api/common.h"
// ... 更多计算 API 头文件

// 2. 辅助宏定义
inline uint32_t float_to_bits(float f) {
  uint32_t r; __builtin_memcpy(&r, &f, sizeof(r)); return r;
}
#ifndef INFINITY
#define INFINITY __builtin_inff()
#endif

// 3. 主内核函数
void kernel_main() {
  int32_t v1 = 4;  // %c4 常量

  // 编译时参数通过 get_compile_time_arg_val() 访问
  untilize_init(get_compile_time_arg_val(0));
  untilize_block(get_compile_time_arg_val(0), v1, get_compile_time_arg_val(1));
  cb_pop_front(get_compile_time_arg_val(0), v1);
  cb_push_back(get_compile_time_arg_val(1), v1);

  return;
}
```

#### 示例 2：Data Movement Kernel 生成的代码

**输入 TTKernel IR**：
```mlir
func.func @ttkernel_noc() attributes {
  ttkernel.thread = #ttkernel.thread<noc>
} {
  %c32 = arith.constant 32 : i32
  %c0 = arith.constant 0 : index
  %src_addr = arith.constant 262144 : i32
  %dst_addr = arith.constant 262400 : i32

  %noc_addr = ttkernel.get_noc_addr(%c0, %c0, %src_addr) : (index, index, i32) -> i64
  ttkernel.noc_async_read(%noc_addr, %dst_addr, %c32) : (i64, i32, i32)
  ttkernel.noc_async_read_barrier() : ()

  func.return
}
```

**生成的 C++ 代码**（`ttkernel_noc.cpp`）：
```cpp
// 1. 头文件
#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"
#include "internal/firmware_common.h"
#include "api/dataflow/dataflow_api.h"

// 2. 主内核函数
void kernel_main() {
  int32_t v3 = 32;        // 传输大小
  int32_t v6 = 262144;    // 源地址
  int32_t v4 = 262400;    // 目标地址
  size_t v5 = 0;          // 核坐标

  // NoC 地址计算和异步读取
  int64_t v7 = get_noc_addr(v5, v5, v6);
  noc_async_read(v7, v4, v3);

  // 等待所有 NoC 操作完成
  noc_async_read_barrier();

  return;
}
```

### 2.4 Circular Buffer (CB) 管理

Circular Buffer 是 Tensix 核间共享数据的关键机制：

**TTKernel CB 类型**：
```mlir
!ttkernel.cb<size, element_type>
// 示例：
!ttkernel.cb<8, !ttcore.tile<32x32, f32>>    // 8 个 tile 的 CB
!ttkernel.cb<8192, f32>                      // 8KB 字节流 CB
```

**CB 操作生成的 C++ 代码**：
```cpp
// Push 操作（生产者）
cb_reserve_back(cb_id, num_tiles);  // 预留空间
cb_push_back(cb_id, num_tiles);     // 标记数据可用

// Pop 操作（消费者）
cb_wait_front(cb_id, num_tiles);    // 等待数据就绪
cb_pop_front(cb_id, num_tiles);     // 释放空间
```

**CB 详细信息打印**（仅 Compute 内核）：
```cpp
// 由 emitDebugPrint() 生成（当使用 dprint 时）
namespace ttmlir {
  inline void print_cb_details_(DebugPrinter dp, uint32_t cb_id) {
    dp << "cb_id " << cb_id << ": { ";
    dp << "size: " << get_local_cb_interface(cb_id).fifo_size << ", ";
    dp << "page_size: " << get_local_cb_interface(cb_id).fifo_page_size << ", ";
    dp << "num_pages: " << get_local_cb_interface(cb_id).fifo_num_pages << ", ";
    dp << "rd_ptr: " << get_local_cb_interface(cb_id).fifo_rd_ptr << ", ";
    dp << "wr_ptr: " << get_local_cb_interface(cb_id).fifo_wr_ptr << " }";
  }
}
```

### 2.5 Tile 操作实现

Tile 是 Tensix 的基本计算单元（32x32 元素）：

**Tilize 操作**（Row-major → Tile 格式）：
```cpp
// TTKernel IR
ttkernel.tilize_init(%cb_in, %num_tiles, %cb_out)
ttkernel.tilize_block(%cb_in, %num_tiles, %cb_out)

// 生成的 C++
tilize_init(cb_in, num_tiles, cb_out);
tilize_block(cb_in, num_tiles, cb_out);
```

**Untilize 操作**（Tile 格式 → Row-major）：
```cpp
// TTKernel IR
ttkernel.untilize_init(%cb_in)
ttkernel.untilize_block(%cb_in, %num_tiles, %cb_out)

// 生成的 C++
untilize_init(cb_in);
untilize_block(cb_in, num_tiles, cb_out);
```

### 2.6 实验性 Low-Level Kernels (LLKs)

编译器会根据使用的操作，自动内联实验性 LLK 代码：

```cpp
void emitExperimentalLLKs() {
  // 检测是否使用特定操作
  if (hasCall("experimental::tilize")) {
    auto llks = StringRef(experimental_tilize_llks_generated, ...);
    builder->create<emitc::VerbatimOp>(loc, llks);
  }

  if (hasCall("experimental::matmul_block")) {
    auto llks = StringRef(experimental_matmul_llks_generated, ...);
    builder->create<emitc::VerbatimOp>(loc, llks);
  }

  if (hasCall("experimental::fabric_fast_write_any_len")) {
    auto llks = StringRef(experimental_fabric_api_generated, ...);
    builder->create<emitc::VerbatimOp>(loc, llks);
  }
}
```

这些 LLK 代码存储为预编译的头文件字符串：
- `experimental_tilize_llks_generated.h`
- `experimental_matmul_llks_generated.h`
- `experimental_dataflow_api_generated.h`
- `experimental_fabric_api_generated.h`

---

## 3. RISC-V 编译工具链

### 3.1 编译器架构

Tenstorrent 使用定制的 **RISC-V GCC** 工具链编译内核：

```
C++ 源码
    ↓ [riscv32-unknown-elf-g++]
RISC-V 汇编 (.s)
    ↓ [riscv32-unknown-elf-as]
RISC-V 目标文件 (.o)
    ↓ [riscv32-unknown-elf-ld]
ELF 可执行文件
```

**目标架构**：
- **ISA**：RV32IMC（32 位，整数，乘法，压缩指令）
- **ABI**：ilp32（整数/长整型/指针都是 32 位）
- **内存模型**：紧耦合内存（L1 SRAM）

### 3.2 编译选项

**典型编译命令**（推测）：
```bash
riscv32-unknown-elf-g++ \
  -march=rv32imc \          # 目标架构
  -mabi=ilp32 \             # ABI
  -O2 \                     # 优化级别
  -fno-exceptions \         # 禁用异常（嵌入式）
  -fno-rtti \               # 禁用 RTTI
  -ffreestanding \          # 独立环境（无标准库）
  -nostdlib \               # 不链接标准库
  -I/path/to/tt-metal/tt_metal \
  -I/path/to/tt-metal/tt_metal/hw/inc \
  -c ttkernel_compute.cpp \
  -o ttkernel_compute.o
```

**优化级别选择**：
- **-O0**：调试版本，保留符号信息
- **-O2**：生产版本，平衡速度和代码大小
- **-Os**：优化代码大小（L1 内存受限场景）
- **-O3**：激进优化（可能增加代码大小）

### 3.3 链接脚本

**典型链接脚本结构**（`tensix.ld`）：
```ld
MEMORY {
  L1_TEXT   (rx)  : ORIGIN = 0x00000000, LENGTH = 16K   /* 指令内存 */
  L1_DATA   (rw)  : ORIGIN = 0x00010000, LENGTH = 240K  /* 数据内存 */
  L1_STACK  (rw)  : ORIGIN = 0x0004C000, LENGTH = 8K    /* 栈 */
}

SECTIONS {
  .text : {
    *(.text.init)      /* 启动代码 */
    *(.text .text.*)   /* 代码段 */
  } > L1_TEXT

  .rodata : {
    *(.rodata .rodata.*)  /* 只读数据 */
  } > L1_TEXT

  .data : {
    *(.data .data.*)      /* 初始化数据 */
  } > L1_DATA

  .bss : {
    *(.bss .bss.*)        /* 未初始化数据 */
    *(COMMON)
  } > L1_DATA

  .stack : {
    . = ALIGN(16);
    _stack_bottom = .;
    . = . + 8K;
    _stack_top = .;
  } > L1_STACK
}

ENTRY(kernel_main)
```

**内存布局约束**：
- **L1 总容量**：约 1MB（每个 Tensix Core）
  - 指令内存：16-32KB
  - 数据内存：240KB+
  - Circular Buffers：动态分配
  - 栈：8KB 典型
- **DRAM**：通过 NoC 访问，不在链接脚本中

### 3.4 链接命令

```bash
riscv32-unknown-elf-ld \
  -T tensix.ld \              # 链接脚本
  -o ttkernel_compute.elf \   # 输出 ELF
  ttkernel_compute.o \        # 目标文件
  -Map=ttkernel_compute.map   # 生成内存映射文件
```

---

## 4. ELF 文件生成

### 4.1 ELF 格式结构

生成的 ELF 文件包含以下主要部分：

```
ELF Header
  ↓
Program Headers (加载信息)
  ↓
Section Headers (节区信息)
  ↓
.text (代码段)
.rodata (只读数据)
.data (初始化数据)
.bss (未初始化数据)
.symtab (符号表)
.strtab (字符串表)
.debug_* (调试信息)
```

**ELF Header 关键字段**：
```c
typedef struct {
  unsigned char e_ident[16];  // 魔数: 7f 45 4c 46 (ELF)
  uint16_t e_type;            // ET_EXEC (可执行文件)
  uint16_t e_machine;         // EM_RISCV (RISC-V)
  uint32_t e_version;         // EV_CURRENT
  uint32_t e_entry;           // 入口点地址 (kernel_main)
  uint32_t e_phoff;           // Program Header 偏移
  uint32_t e_shoff;           // Section Header 偏移
  uint32_t e_flags;           // RVC (压缩指令)
  // ...
} Elf32_Ehdr;
```

### 4.2 Section 布局

#### .text 段（代码段）
```
地址          内容
0x00000000   kernel_main:
0x00000000     addi sp, sp, -16      # 分配栈帧
0x00000004     sw   ra, 12(sp)       # 保存返回地址
0x00000008     li   a0, 4            # v1 = 4
0x0000000C     li   a1, 0            # cb_in = get_compile_time_arg_val(0)
0x00000010     call untilize_init    # 调用 untilize_init
0x00000014     ...
0x000000XX     lw   ra, 12(sp)       # 恢复返回地址
0x000000YY     addi sp, sp, 16       # 释放栈帧
0x000000ZZ     ret                   # 返回
```

#### .rodata 段（只读数据）
```
地址          内容
0x00010000   .LC0: .float 0.0      # 浮点常量
0x00010004   .LC1: .float 1.0
0x00010008   debug_string: "CB overflow\0"
```

#### .data 段（初始化数据）
```
地址          内容
0x00020000   global_config: .word 0x12345678
0x00020004   tile_size: .word 1024
```

#### .bss 段（未初始化数据）
```
地址          大小
0x00030000   buffer: .space 4096   # 4KB 缓冲区
0x00031000   temp_array: .space 256
```

### 4.3 符号表和重定位

**符号表 (.symtab)**：
```
Num  Value      Size Type    Bind   Vis      Ndx Name
0    00000000   0    NOTYPE  LOCAL  DEFAULT  UND
1    00000000   128  FUNC    GLOBAL DEFAULT  1   kernel_main
2    00000080   64   FUNC    GLOBAL DEFAULT  1   untilize_init
3    00010000   4    OBJECT  GLOBAL DEFAULT  2   .LC0
4    00020000   4    OBJECT  GLOBAL DEFAULT  3   global_config
```

**重定位表 (.rela.text)**：
```
Offset     Type               Symbol          Addend
0x00000010 R_RISCV_CALL_PLT   untilize_init   0
0x00000024 R_RISCV_HI20       .LC0            0
0x00000028 R_RISCV_LO12_I     .LC0            0
```

### 4.4 可在 Tensix Core 上运行的 ELF

**关键要求**：
1. **入口点**：`kernel_main` 函数必须存在
2. **内存约束**：代码+数据必须适配 L1 内存（<256KB）
3. **无标准库**：所有函数调用必须来自 TTMetal API
4. **固定加载地址**：由链接脚本指定的 L1 地址

**ELF 验证命令**：
```bash
# 查看 ELF 头
riscv32-unknown-elf-readelf -h ttkernel_compute.elf

# 查看段信息
riscv32-unknown-elf-readelf -S ttkernel_compute.elf

# 查看符号表
riscv32-unknown-elf-nm ttkernel_compute.elf

# 反汇编
riscv32-unknown-elf-objdump -d ttkernel_compute.elf

# 检查代码大小
riscv32-unknown-elf-size ttkernel_compute.elf
```

---

## 5. 内核类型详解

### 5.1 Compute Kernels（计算内核）

**运行处理器**：TRISC0/1/2（Unpack/Math/Pack 流水线）

**典型操作流程**：
```cpp
void kernel_main() {
  // 1. 初始化操作
  matmul_init(cb_in0, cb_in1);

  // 2. 主计算循环
  for (uint32_t b = 0; b < num_blocks; ++b) {
    // TRISC0: Unpack tiles from CB to register file
    cb_wait_front(cb_in0, tiles_per_batch);
    cb_wait_front(cb_in1, tiles_per_batch);

    // TRISC1: Execute math operation
    matmul_tiles(cb_in0, cb_in1, 0, 0, 0);

    // TRISC2: Pack result from register to CB
    cb_reserve_back(cb_out, 1);
    pack_tile(0, cb_out);
    cb_push_back(cb_out, 1);

    // 释放输入 CB
    cb_pop_front(cb_in0, tiles_per_batch);
    cb_pop_front(cb_in1, tiles_per_batch);
  }
}
```

**重要 API**：
- **Init**：`matmul_init`, `add_tiles_init`, `tilize_init`
- **Compute**：`matmul_tiles`, `add_tiles`, `mul_tiles`
- **Layout**：`tilize_block`, `untilize_block`, `transpose_wh_tiles`
- **Unary**：`exp_tile`, `relu_tile`, `sqrt_tile`

### 5.2 Data Movement Kernels（数据移动内核）

**运行处理器**：NCRISC

**Reader Kernel 示例**：
```cpp
void kernel_main() {
  // 1. 获取运行时参数
  uint32_t src_addr = get_arg_val<uint32_t>(0);
  uint32_t num_tiles = get_arg_val<uint32_t>(1);

  // 2. 计算 NoC 地址
  uint64_t src_noc_addr = get_noc_addr(src_core_x, src_core_y, src_addr);

  // 3. 循环读取数据
  for (uint32_t i = 0; i < num_tiles; ++i) {
    // 预留 CB 空间
    cb_reserve_back(cb_id, 1);

    // 异步读取一个 tile
    uint32_t cb_addr = get_write_ptr(cb_id);
    noc_async_read(src_noc_addr, cb_addr, tile_size_bytes);
    noc_async_read_barrier();

    // 标记数据可用
    cb_push_back(cb_id, 1);

    src_noc_addr += tile_size_bytes;
  }
}
```

**Writer Kernel 示例**：
```cpp
void kernel_main() {
  uint32_t dst_addr = get_arg_val<uint32_t>(0);
  uint32_t num_tiles = get_arg_val<uint32_t>(1);

  uint64_t dst_noc_addr = get_noc_addr(dst_core_x, dst_core_y, dst_addr);

  for (uint32_t i = 0; i < num_tiles; ++i) {
    // 等待数据就绪
    cb_wait_front(cb_id, 1);

    // 异步写入一个 tile
    uint32_t cb_addr = get_read_ptr(cb_id);
    noc_async_write(cb_addr, dst_noc_addr, tile_size_bytes);
    noc_async_write_barrier();

    // 释放 CB 空间
    cb_pop_front(cb_id, 1);

    dst_noc_addr += tile_size_bytes;
  }
}
```

**重要 API**：
- **NoC 操作**：`noc_async_read`, `noc_async_write`, `noc_semaphore_inc`
- **地址转换**：`get_noc_addr`, `get_noc_multicast_addr`
- **CB 管理**：`get_read_ptr`, `get_write_ptr`

### 5.3 Prefetcher/Dispatcher Kernels（固件内核）

**运行处理器**：BRISC

**Prefetcher 功能**（`cq_prefetch.cpp`）：
- 从命令队列（Command Queue）读取命令
- 解析命令类型（kernel launch, data transfer, sync）
- 分发命令到相应的 Dispatcher

**Dispatcher 功能**（`cq_dispatch.cpp`）：
- 配置内核参数（CB、runtime args、semaphores）
- 加载内核 ELF 到各 RISC 处理器
- 启动内核执行
- 管理内核间同步

**文件位置**：
```
tt-metal/tt_metal/impl/dispatch/kernels/
├── cq_prefetch.cpp              # 命令预取
├── cq_dispatch.cpp              # 主调度器
├── cq_dispatch_subordinate.cpp # 从调度器
└── packet_*.cpp                 # 网络数据包处理
```

---

## 6. 完整编译流程

```
                    ┌─────────────────────┐
                    │  TTKernel IR        │
                    │  (MLIR Dialect)     │
                    └──────────┬──────────┘
                               │
                               │ TTKernelToEmitC Pass
                               ↓
                    ┌─────────────────────┐
                    │  EmitC IR           │
                    │  (中间表示)          │
                    └──────────┬──────────┘
                               │
                               │ translateToCpp
                               ↓
                    ┌─────────────────────┐
                    │  C++ 源码           │
                    │  kernel_main() {...}│
                    └──────────┬──────────┘
                               │
                               │ riscv32-g++ -c
                               ↓
                    ┌─────────────────────┐
                    │  RISC-V 汇编        │
                    │  (.s 文件)          │
                    └──────────┬──────────┘
                               │
                               │ riscv32-as
                               ↓
                    ┌─────────────────────┐
                    │  RISC-V Object      │
                    │  (.o 文件)          │
                    └──────────┬──────────┘
                               │
                               │ riscv32-ld + tensix.ld
                               ↓
                    ┌─────────────────────┐
                    │  ELF 可执行文件     │
                    │  (内核二进制)        │
                    └──────────┬──────────┘
                               │
                               │ TTMetal 运行时
                               ↓
                    ┌─────────────────────┐
                    │  加载到 Tensix L1   │
                    │  在 RISC-V 处理器   │
                    │  上执行             │
                    └─────────────────────┘
```

**详细步骤**：

1. **MLIR 编译**（编译时）
   ```bash
   ttmlir-opt --convert-ttkernel-to-emitc -o kernel.emitc.mlir kernel.mlir
   ttmlir-translate --ttkernel-to-cpp -o kernel.cpp kernel.emitc.mlir
   ```

2. **C++ 编译**（编译时）
   ```bash
   riscv32-unknown-elf-g++ \
     -march=rv32imc -mabi=ilp32 -O2 \
     -ffreestanding -nostdlib \
     -I${TT_METAL_HOME}/tt_metal/hw/inc \
     -c kernel.cpp -o kernel.o
   ```

3. **链接**（编译时）
   ```bash
   riscv32-unknown-elf-ld \
     -T tensix.ld \
     -o kernel.elf \
     kernel.o
   ```

4. **打包到 Flatbuffer**（编译时）
   - ELF 文件内容被读取为字节流
   - 嵌入到 Flatbuffer 的 `Program` 消息中
   - 与 CB 配置、运行时参数一起序列化

5. **加载和执行**（运行时）
   - TTMetal 运行时从 Flatbuffer 提取 ELF
   - 通过 PCIe/NoC 加载到 Tensix Core L1 内存
   - 配置 RISC 处理器的 PC 寄存器指向 `kernel_main`
   - 释放 RISC 处理器开始执行

---

## 7. 内核加载和执行

### 7.1 TTMetal 如何加载 ELF 文件

**加载流程**：
```cpp
// 1. Program 对象包含所有内核
class Program {
  std::vector<Kernel> kernels_;
};

// 2. Kernel 对象包含 ELF 二进制
class Kernel {
  std::string name_;
  ThreadType thread_type_;  // noc, compute, ethernet
  std::vector<uint8_t> elf_binary_;
  std::vector<uint32_t> compile_time_args_;
  std::map<uint32_t, std::vector<uint32_t>> runtime_args_;  // per-core
};

// 3. 加载内核到设备
void load_kernel_to_core(
    Device *device,
    const Kernel &kernel,
    CoreCoord core) {

  // 计算 L1 地址
  uint32_t l1_text_base = get_l1_text_base(kernel.thread_type_);

  // 通过 PCIe 写入 ELF 到 L1
  device->write_to_device(
    kernel.elf_binary_.data(),
    kernel.elf_binary_.size(),
    core,
    l1_text_base
  );

  // 写入编译时参数
  device->write_to_device(
    kernel.compile_time_args_.data(),
    kernel.compile_time_args_.size() * sizeof(uint32_t),
    core,
    L1_COMPILE_TIME_ARGS_BASE
  );

  // 写入运行时参数
  auto &rt_args = kernel.runtime_args_[core.id];
  device->write_to_device(
    rt_args.data(),
    rt_args.size() * sizeof(uint32_t),
    core,
    L1_RUNTIME_ARGS_BASE
  );
}
```

### 7.2 内核分发到不同的 Tensix Cores

**分发策略**：
```cpp
// Op 指定核心范围
ttnn::matmul_op {
  input_shards = {{core(0,0), core(0,1)}, {core(1,0), core(1,1)}},
  compute_kernel = "matmul_compute.elf",
  reader_kernel = "matmul_reader.elf",
  writer_kernel = "matmul_writer.elf"
}

// TTMetal 运行时分发
for (auto core : input_shards.cores()) {
  // 每个核心加载三个内核
  load_kernel_to_core(device, compute_kernel, core);  // → TRISC
  load_kernel_to_core(device, reader_kernel, core);   // → NCRISC (Reader)
  load_kernel_to_core(device, writer_kernel, core);   // → NCRISC (Writer)
}
```

**内存布局（单个 Tensix Core）**：
```
L1 内存空间 (1MB)
├── 0x00000 - 0x03FFF: BRISC 指令 (Dispatcher)
├── 0x04000 - 0x07FFF: NCRISC 指令 (Reader)
├── 0x08000 - 0x0BFFF: NCRISC 指令 (Writer)
├── 0x0C000 - 0x0FFFF: TRISC0 指令
├── 0x10000 - 0x13FFF: TRISC1 指令
├── 0x14000 - 0x17FFF: TRISC2 指令
├── 0x18000 - 0x1BFFF: 编译时参数
├── 0x1C000 - 0x1FFFF: 运行时参数
├── 0x20000 - 0x9FFFF: Circular Buffers (动态分配)
└── 0xA0000 - 0xFFFFF: 栈和临时数据
```

### 7.3 内核执行模型

**启动序列**：
```
1. Host (CPU)
   ↓ write_to_device(ELF, L1)
2. PCIe Bridge
   ↓ DMA transfer
3. Tensix Core L1 内存
   ↓ set PC register
4. RISC 处理器
   ↓ fetch instruction
5. 执行 kernel_main()
```

**同步机制**：
```cpp
// Compute Kernel 等待数据
cb_wait_front(cb_in0, num_tiles);  // 阻塞直到 Reader 填充数据

// Reader Kernel 等待空间
cb_reserve_back(cb_in0, num_tiles); // 阻塞直到 Compute 消费数据

// Writer Kernel 等待结果
cb_wait_front(cb_out, num_tiles);   // 阻塞直到 Compute 产生数据
```

**CB 同步原理**：
- **FIFO 语义**：读写指针硬件同步
- **无锁设计**：单生产者单消费者模型
- **自动阻塞**：`cb_wait_front` 和 `cb_reserve_back` 轮询硬件寄存器

### 7.4 内核间通信（NoC）

**NoC（Network-on-Chip）架构**：
```
      NoC Router
     ┌────┴────┐
Core(0,0)   Core(0,1)   Core(0,2)   ...
     │         │            │
Core(1,0)   Core(1,1)   Core(1,2)   ...
     │         │            │
    ...       ...          ...
```

**点对点通信**：
```cpp
// Reader Kernel (Core 0,0) → Compute Kernel (Core 1,1)
uint64_t dst_noc_addr = get_noc_addr(1, 1, L1_CB_BASE);
noc_async_write(local_cb_addr, dst_noc_addr, tile_size);
noc_async_write_barrier();

// Compute Kernel (Core 1,1) 读取
cb_wait_front(cb_in, 1);  // 自动等待 NoC 写入完成
```

**多播通信**：
```cpp
// 向 2x2 核心网格广播数据
uint64_t dst_noc_addr = get_noc_multicast_addr(
  /*start_x=*/0, /*start_y=*/0,
  /*end_x=*/1, /*end_y=*/1,
  L1_CB_BASE
);
noc_async_write(local_data, dst_noc_addr, data_size);
```

---

## 8. 完整示例：矩阵乘法内核

### 8.1 TTKernel IR

```mlir
// MatMul Compute Kernel
func.func @matmul_compute(
  %cb_in0 : !ttkernel.cb<32, !ttcore.tile<32x32, bf16>>,
  %cb_in1 : !ttkernel.cb<32, !ttcore.tile<32x32, bf16>>,
  %cb_out : !ttkernel.cb<16, !ttcore.tile<32x32, bf16>>
) attributes {
  ttkernel.thread = #ttkernel.thread<compute>,
  ttkernel.arg_spec = #ttkernel.arg_spec<
    ct_args = [
      <arg_type = cb_port, operand_index = 0>,
      <arg_type = cb_port, operand_index = 1>,
      <arg_type = cb_port, operand_index = 2>
    ]
  >
} {
  %c32 = arith.constant 32 : i32
  %c1 = arith.constant 1 : i32

  // 初始化 matmul
  ttkernel.matmul_init(%cb_in0, %cb_in1)

  // 主循环
  scf.for %i = %c0 to %c32 step %c1 {
    // 等待输入数据
    ttkernel.cb_wait_front(%cb_in0, %c1)
    ttkernel.cb_wait_front(%cb_in1, %c1)

    // 执行矩阵乘法
    ttkernel.matmul_tiles(%cb_in0, %cb_in1, 0, 0, 0)

    // 输出结果
    ttkernel.cb_reserve_back(%cb_out, %c1)
    ttkernel.pack_tile(0, %cb_out)
    ttkernel.cb_push_back(%cb_out, %c1)

    // 释放输入
    ttkernel.cb_pop_front(%cb_in0, %c1)
    ttkernel.cb_pop_front(%cb_in1, %c1)
  }

  func.return
}
```

### 8.2 生成的 C++ 代码

```cpp
// matmul_compute.cpp
#include <cstdint>
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/common.h"

void kernel_main() {
  // 编译时参数（CB IDs）
  constexpr uint32_t cb_in0 = get_compile_time_arg_val(0);
  constexpr uint32_t cb_in1 = get_compile_time_arg_val(1);
  constexpr uint32_t cb_out = get_compile_time_arg_val(2);

  // 运行时参数
  const uint32_t num_tiles = get_arg_val<uint32_t>(0);

  // 初始化 matmul 操作
  matmul_init(cb_in0, cb_in1);

  // 主计算循环
  for (uint32_t i = 0; i < num_tiles; ++i) {
    // 等待输入 tile 就绪
    cb_wait_front(cb_in0, 1);
    cb_wait_front(cb_in1, 1);

    // Unpack (TRISC0) + Math (TRISC1) + Pack (TRISC2)
    matmul_tiles(cb_in0, cb_in1, 0, 0, 0);

    // 预留输出空间并写入结果
    cb_reserve_back(cb_out, 1);
    pack_tile(0, cb_out);
    cb_push_back(cb_out, 1);

    // 释放输入 CB 空间
    cb_pop_front(cb_in0, 1);
    cb_pop_front(cb_in1, 1);
  }
}
```

### 8.3 编译后的 RISC-V 汇编（部分）

```asm
# matmul_compute.s (TRISC1 代码)
kernel_main:
    # 函数序言
    addi sp, sp, -32          # 分配栈帧
    sw   ra, 28(sp)           # 保存返回地址
    sw   s0, 24(sp)           # 保存寄存器
    sw   s1, 20(sp)

    # 加载编译时参数 (假设在固定地址)
    lui  a0, 0x18000          # cb_in0 地址
    lw   s0, 0(a0)            # s0 = cb_in0 ID
    lw   s1, 4(a0)            # s1 = cb_in1 ID
    lw   s2, 8(a0)            # s2 = cb_out ID

    # 初始化 matmul
    mv   a0, s0
    mv   a1, s1
    call matmul_init

    # 主循环初始化
    li   s3, 0                # i = 0
    li   s4, 32               # num_tiles = 32

.L_loop:
    bge  s3, s4, .L_done      # if (i >= num_tiles) break

    # cb_wait_front(cb_in0, 1)
    mv   a0, s0
    li   a1, 1
    call cb_wait_front

    # cb_wait_front(cb_in1, 1)
    mv   a0, s1
    li   a1, 1
    call cb_wait_front

    # matmul_tiles(cb_in0, cb_in1, 0, 0, 0)
    mv   a0, s0
    mv   a1, s1
    li   a2, 0
    li   a3, 0
    li   a4, 0
    call matmul_tiles         # 调用硬件加速器

    # cb_reserve_back(cb_out, 1)
    mv   a0, s2
    li   a1, 1
    call cb_reserve_back

    # pack_tile(0, cb_out)
    li   a0, 0
    mv   a1, s2
    call pack_tile

    # cb_push_back(cb_out, 1)
    mv   a0, s2
    li   a1, 1
    call cb_push_back

    # cb_pop_front(cb_in0, 1)
    mv   a0, s0
    li   a1, 1
    call cb_pop_front

    # cb_pop_front(cb_in1, 1)
    mv   a0, s1
    li   a1, 1
    call cb_pop_front

    # i++
    addi s3, s3, 1
    j    .L_loop

.L_done:
    # 函数尾声
    lw   ra, 28(sp)
    lw   s0, 24(sp)
    lw   s1, 20(sp)
    addi sp, sp, 32
    ret

# matmul_tiles 的硬件加速实现（伪代码）
matmul_tiles:
    # 配置 Tensix Math 引擎寄存器
    li   t0, MATH_CTRL_REG
    li   t1, MATMUL_OP
    sw   t1, 0(t0)            # 设置操作类型

    # 设置源 CB
    sw   a0, 4(t0)            # cb_in0
    sw   a1, 8(t0)            # cb_in1

    # 启动计算
    li   t1, 1
    sw   t1, 12(t0)           # START_BIT

    # 等待完成（轮询状态寄存器）
.L_wait:
    lw   t1, 16(t0)           # DONE_BIT
    beqz t1, .L_wait

    ret
```

### 8.4 最终 ELF 文件描述

**文件结构**（`readelf -a matmul_compute.elf`）：
```
ELF Header:
  Magic:   7f 45 4c 46 01 01 01 00 00 00 00 00 00 00 00 00
  Class:                             ELF32
  Data:                              2's complement, little endian
  Version:                           1 (current)
  OS/ABI:                            UNIX - System V
  ABI Version:                       0
  Type:                              EXEC (Executable file)
  Machine:                           RISC-V
  Version:                           0x1
  Entry point address:               0x0
  Start of program headers:          52 (bytes into file)
  Start of section headers:          4832 (bytes into file)
  Flags:                             0x3, RVC, soft-float ABI
  Size of this header:               52 (bytes)
  Size of program headers:           32 (bytes)
  Number of program headers:         2
  Size of section headers:           40 (bytes)
  Number of section headers:         10
  Section header string table index: 9

Section Headers:
  [Nr] Name              Type            Addr     Off    Size   ES Flg Lk Inf Al
  [ 0]                   NULL            00000000 000000 000000 00      0   0  0
  [ 1] .text             PROGBITS        00000000 001000 000480 00  AX  0   0  4
  [ 2] .rodata           PROGBITS        00010000 001480 000020 00   A  0   0  4
  [ 3] .data             PROGBITS        00020000 0014a0 000010 00  WA  0   0  4
  [ 4] .bss              NOBITS          00030000 0014b0 001000 00  WA  0   0  4
  [ 5] .symtab           SYMTAB          00000000 0014b0 000150 10      6  12  4
  [ 6] .strtab           STRTAB          00000000 001600 000080 00      0   0  1
  [ 7] .shstrtab         STRTAB          00000000 001680 00004d 00      0   0  1
  [ 8] .debug_info       PROGBITS        00000000 0016d0 000200 00      0   0  1
  [ 9] .debug_line       PROGBITS        00000000 0018d0 000100 00      0   0  1

Program Headers:
  Type           Offset   VirtAddr   PhysAddr   FileSiz MemSiz  Flg Align
  LOAD           0x001000 0x00000000 0x00000000 0x00480 0x00480 R E 0x1000
  LOAD           0x001480 0x00010000 0x00010000 0x00030 0x01030 RW  0x1000

Symbol table '.symtab' contains 21 entries:
   Num:    Value  Size Type    Bind   Vis      Ndx Name
     0: 00000000     0 NOTYPE  LOCAL  DEFAULT  UND
     1: 00000000     0 SECTION LOCAL  DEFAULT    1 .text
     2: 00010000     0 SECTION LOCAL  DEFAULT    2 .rodata
     3: 00020000     0 SECTION LOCAL  DEFAULT    3 .data
     4: 00030000     0 SECTION LOCAL  DEFAULT    4 .bss
    12: 00000000   320 FUNC    GLOBAL DEFAULT    1 kernel_main
    13: 00000140    80 FUNC    GLOBAL DEFAULT    1 matmul_init
    14: 000001c0   160 FUNC    GLOBAL DEFAULT    1 matmul_tiles
    15: 00000260    64 FUNC    GLOBAL DEFAULT    1 cb_wait_front
    16: 000002a0    64 FUNC    GLOBAL DEFAULT    1 cb_reserve_back
    17: 000002e0    48 FUNC    GLOBAL DEFAULT    1 cb_push_back
    18: 00000310    48 FUNC    GLOBAL DEFAULT    1 cb_pop_front
    19: 00000340    96 FUNC    GLOBAL DEFAULT    1 pack_tile
    20: 000003a0   224 FUNC    GLOBAL DEFAULT    1 get_compile_time_arg_val

代码段大小: 1152 bytes (0x480)
数据段大小: 48 bytes (0x30)
BSS 段大小: 4096 bytes (0x1000)
总内存占用: ~5.3 KB
```

**执行时内存映射**：
```
Tensix Core (1,1) L1 内存
├── 0x00000000: kernel_main (320 bytes)
│   ├── 函数序言
│   ├── matmul_init 调用
│   ├── 主循环
│   └── 函数尾声
├── 0x00000140: matmul_init (80 bytes)
├── 0x000001C0: matmul_tiles (160 bytes)
│   └── 硬件加速器配置
├── 0x00000260: cb_* 函数 (256 bytes)
├── 0x00010000: 只读数据 (32 bytes)
├── 0x00020000: 全局变量 (16 bytes)
├── 0x00030000: BSS (4KB)
└── 0x00040000: CB0-CB2 (动态分配)
    ├── CB0 (cb_in0): 32 tiles × 2KB = 64KB
    ├── CB1 (cb_in1): 32 tiles × 2KB = 64KB
    └── CB2 (cb_out): 16 tiles × 2KB = 32KB
```

---

## 9. 内核编译的关键要点

### 9.1 性能优化

**1. CB 大小调优**
- **太小**：频繁阻塞，流水线停顿
- **太大**：浪费 L1 内存，减少可用核心数
- **经验值**：4-32 tiles，取决于操作类型

**2. 循环展开**
```cpp
// 未优化
for (uint32_t i = 0; i < num_tiles; ++i) {
  matmul_tiles(...);
}

// 优化（展开因子 4）
for (uint32_t i = 0; i < num_tiles; i += 4) {
  matmul_tiles(...);
  matmul_tiles(...);
  matmul_tiles(...);
  matmul_tiles(...);
}
```

**3. NoC 流水线**
```cpp
// 双缓冲：异步读取下一批数据
noc_async_read(src_addr, cb_addr, tile_size);      // 启动传输
process_previous_data();                            // 并行处理
noc_async_read_barrier();                           // 等待传输完成
```

### 9.2 调试技巧

**1. 使用 dprint**
```cpp
// TTKernel IR
ttkernel.dprint("CB0 state: ", %cb_id)

// 生成的 C++
ttmlir::dprint("CB0 state: ", cb_id);
DPRINT << "CB0 size: " << get_local_cb_interface(cb_id).fifo_size;
```

**2. 检查 ELF 符号**
```bash
# 确认 kernel_main 存在
nm matmul_compute.elf | grep kernel_main

# 检查代码大小是否超限
size matmul_compute.elf
# 输出：text < 16KB, data+bss < 240KB
```

**3. 反汇编验证**
```bash
objdump -d matmul_compute.elf > matmul_compute.dis
# 检查：
# - 是否有意外的库调用（应该没有 malloc/printf）
# - 循环是否被正确优化
# - 是否使用了 RVC（压缩指令）
```

### 9.3 常见陷阱

**1. 栈溢出**
```cpp
// 危险：大数组在栈上
void kernel_main() {
  uint32_t large_array[1024];  // 4KB 栈空间！
  // ...
}

// 安全：使用 L1 数据段或 CB
constexpr uint32_t buffer_addr = 0x30000;
uint32_t *buffer = reinterpret_cast<uint32_t*>(buffer_addr);
```

**2. CB 死锁**
```cpp
// 死锁：Reader 和 Compute 互相等待
// Reader:
cb_reserve_back(cb_in, 32);  // 等待 32 个空位

// Compute:
cb_wait_front(cb_in, 32);    // 等待 32 个数据

// 解决：分批处理
cb_reserve_back(cb_in, 1);
cb_wait_front(cb_in, 1);
```

**3. NoC 地址错误**
```cpp
// 错误：使用 L1 本地地址
uint64_t addr = 0x00020000;
noc_async_read(addr, ...);  // 无效！

// 正确：转换为 NoC 地址
uint64_t noc_addr = get_noc_addr(core_x, core_y, 0x00020000);
noc_async_read(noc_addr, ...);
```

---

## 10. 总结

### TTKernel 编译流程回顾

1. **高层表示**：TTKernel IR（MLIR Dialect）
2. **代码生成**：EmitC → C++ 源码
3. **编译**：RISC-V GCC → 汇编 → 目标文件
4. **链接**：Linker Script → ELF 可执行文件
5. **部署**：Flatbuffer 打包 → TTMetal 加载 → Tensix 执行

### 关键技术

- **多处理器架构**：5 个 RISC-V 处理器协同工作
- **Circular Buffer**：零拷贝内核间通信
- **NoC 网络**：高带宽多核互连
- **硬件加速**：Tensix Math 引擎直接访问

### 相关文档

- **前序**：[06-ttnn-to-ttmetal.md](06-ttnn-to-ttmetal.md) - TTMetal Dialect 生成
- **后续**：[08-runtime-execution.md](08-runtime-execution.md) - 运行时执行流程
- **参考**：[02-mlir-dialects.md](02-mlir-dialects.md) - TTKernel Dialect 定义

---

**文档版本**：1.0
**最后更新**：2025-02-07
**适用项目**：tt-mlir, tt-xla, tt-forge
