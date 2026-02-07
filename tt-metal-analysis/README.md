# TT-Metal 模拟器适配分析

本目录用于记录 tt-metal 仓库的分析过程，目标是理解其架构并找到模拟器的接入点。

## 目标

将 Tenstorrent 的 tt-metal 硬件抽象层修改为支持自定义模拟器，使得：
- tt-metal 能够检测到"虚拟"的 Tenstorrent 设备
- 计算指令能够被转发到模拟器执行
- 模拟器返回的结果能够被正确处理

## 当前问题

运行 Demo 时报错：
```
TT_FATAL @ tt_cluster.cpp:117: num_chips > 0
info: No chips detected in the cluster
```

原因：没有物理 Tenstorrent 硬件，tt-metal 的设备检测失败。

## 分析计划

| 阶段 | 内容 | 状态 | 文档 |
|------|------|------|------|
| 1 | tt-metal 仓库结构概览 | ✅ 完成 | [`01-repo-structure.md`](./01-repo-structure.md) |
| 2 | 设备检测流程分析 | ✅ 完成 | [`02-device-detection.md`](./02-device-detection.md) |
| 3 | UMD 层接口分析 | ✅ 完成 | [`03-umd-interface.md`](./03-umd-interface.md) |
| 4 | 命令提交流程 | ✅ 完成 | [`04-command-submission.md`](./04-command-submission.md) |
| 5 | 内存管理 | ✅ 完成 | [`05-memory-management.md`](./05-memory-management.md) |
| 6 | 模拟器接口设计 | ✅ 完成 | [`06-simulator-interface.md`](./06-simulator-interface.md) |
| 7 | Dispatch 架构深度剖析 | ✅ 完成 | [`07-dispatch-architecture-deep-dive.md`](./07-dispatch-architecture-deep-dive.md) |
| 8 | Socket 模拟器实现审查 | ✅ 完成 | [`08-socket-simulator-implementation-review.md`](./08-socket-simulator-implementation-review.md) |

## 分析总结

### 关键发现

1. **tt-metal 已有模拟器支持**: 通过 `TT_METAL_SIMULATOR` 环境变量可以启用 Simulator 模式
2. **TTSIM 接口**: 只需实现 6 个函数的动态库 (`libttsim.so`)
3. **Mock 模式**: 可用于快速验证流程 (不执行计算)
4. **UMD 层是最佳接入点**: `Chip` 类提供了清晰的抽象接口

### 推荐方案

**TTSIM 动态库方案** (详见 `06-simulator-interface.md`)

```bash
# 启用模拟器
export TT_METAL_SIMULATOR=/path/to/simulator
export ARCH_NAME=wormhole_b0
```

需要实现的接口：
```c
void libttsim_init(void);
void libttsim_exit(void);
uint32_t libttsim_pci_config_rd32(uint32_t bdf, uint32_t offset);
void libttsim_tile_rd_bytes(uint32_t x, uint32_t y, uint64_t addr, void* p, uint32_t size);
void libttsim_tile_wr_bytes(uint32_t x, uint32_t y, uint64_t addr, const void* p, uint32_t size);
void libttsim_clock(uint32_t n_clocks);
```

### 与 QEMU 方案对比

| 方面 | TTSIM 方案 | QEMU 方案 |
|------|-----------|----------|
| 实现复杂度 | 中等 | 极高 |
| 代码修改 | 无需修改 tt-metal | 无需修改 tt-metal |
| 功能完整性 | 可按需实现 | 完整硬件模拟 |
| 性能 | 较好 | 有开销 |
| 维护成本 | 低 | 高 |

## 关键文件位置

tt-metal 源码位于 tt-xla 构建目录中：
```
/home/ubuntu/work/tt/tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/
```

### 核心文件

| 文件 | 作用 |
|------|------|
| `tt_metal/llrt/tt_cluster.cpp` | 设备检测和集群管理 |
| `tt_metal/llrt/rtoptions.cpp` | 运行时选项 (环境变量解析) |
| `tt_metal/third_party/umd/device/chip/` | Chip 类实现 |
| `tt_metal/third_party/umd/device/simulation/` | 模拟器相关实现 |
| `tt_metal/impl/dispatch/` | 命令提交系统 (详见 [`07-dispatch-architecture-deep-dive.md`](./07-dispatch-architecture-deep-dive.md)) |
| `tt_metal/impl/dispatch/kernels/` | Prefetcher/Dispatcher 固件 (在设备上运行) |
| `tt_metal/impl/allocator/` | 内存分配器 |

### UMD Chip 类继承体系

```
Chip (抽象基类)
├── LocalChip          # PCIe 连接的本地芯片
├── RemoteChip         # Ethernet 连接的远程芯片
├── MockChip           # 测试用 Mock (所有操作为空)
└── SimulationChip     # 模拟基类
    ├── TTSimChip      # TTSIM (.so 库)
    └── RTLSimulationChip  # RTL 仿真 (Zebu)
```

## 调用链路 (从错误堆栈分析)

```
torch_xla._XLAC._xla_get_default_device()
    │
    ▼
torch_xla::runtime::PjRtComputationClient::Initialize()
    │
    ▼
tt::pjrt::ClientInstance::populateDevices()
    │
    ▼
tt::runtime::getCurrentSystemDesc()
    │
    ▼
tt::tt_metal::GetNumAvailableDevices()
    │
    ▼
tt::tt_metal::MetalContext::instance()
    │
    ▼
tt::tt_metal::MetalContext::initialize_base_objects()
    │
    ▼
tt::Cluster::Cluster() → open_driver()
    │
    ├── [Silicon] 检测 PCIe 设备
    ├── [Simulator] 加载 libttsim.so ← 推荐方案
    └── [Mock] 加载 YAML 集群描述符
    │
    ▼
tt::Cluster::get_cluster_type_from_cluster_desc()
    │
    ▼
❌ TT_FATAL(num_chips > 0) 失败 (Silicon 模式无硬件时)
```

## 环境变量

### 关键配置

```bash
# 启用 Simulator 模式 (推荐)
export TT_METAL_SIMULATOR=/path/to/simulator/directory

# 启用 Mock 模式 (仅测试流程)
export TT_METAL_MOCK_CLUSTER_DESC_PATH=/path/to/cluster.yaml

# 指定架构 (Simulator/Mock 模式必需)
export ARCH_NAME=wormhole_b0

# 调试日志
export TT_METAL_LOGGER_LEVEL=DEBUG
export TT_METAL_LOGGER_TYPES=Dispatch,Allocator
```

### 优先级

```
TT_METAL_SIMULATOR > TT_METAL_MOCK_CLUSTER_DESC_PATH > Silicon (默认)
```

## 快速验证

### Mock 模式测试

```bash
# 设置 Mock 模式
export TT_METAL_MOCK_CLUSTER_DESC_PATH=/path/to/tt-metal/tt_metal/third_party/umd/tests/cluster_descriptor_examples/wormhole_N150.yaml
export ARCH_NAME=wormhole_b0

# 运行测试
python -c "
import torch_xla.core.xla_model as xm
device = xm.xla_device()
print(f'Device: {device}')
"
```

### Simulator 模式测试

```bash
# 编译模拟器 (详见 06-simulator-interface.md)
cd /path/to/simulator
./build_simulator.sh

# 设置环境变量
export TT_METAL_SIMULATOR=$(pwd)/build
export ARCH_NAME=wormhole_b0

# 运行测试
python test_simulator.py
```

## 进阶阅读

### 深度理解 tt-metal

**强烈推荐阅读 [`07-dispatch-architecture-deep-dive.md`](./07-dispatch-architecture-deep-dive.md)**

这份文档深入剖析了 tt-metal 的命令分发架构，解答了以下关键问题：
- Prefetcher 和 Dispatcher 的本质区别
- Hugepage 在 Host 还是设备上？为什么需要它？
- Tensix Core 的 5 个 RISC-V 核心如何分工？
- Prefetcher/Dispatcher 固件如何在硬件上运行？
- 完整的端到端命令执行流程

适合希望深入理解 tt-metal 运行机制的开发者。

### 实现审查

**实现者必读：[`08-socket-simulator-implementation-review.md`](./08-socket-simulator-implementation-review.md)**

这份文档详细分析了当前基于 Socket 的模拟器实现（位于 `/home/ubuntu/work/tt/tt-umd`），包括：
- 架构问题：同步阻塞模型、错误处理不完善
- 协议问题：协议过于简单、缺乏流控机制
- 性能问题：小数据传输效率低、全局锁竞争
- 功能缺失：libttsim_clock 未实现、缺少批量传输
- 改进建议：异步批量模式、协议版本控制、缓存层、细粒度锁
- 优先级排序：从高到低的改进路线图

适合正在实现或优化模拟器的开发者。

## 参考资料

- [TT-Forge 架构分析报告](/home/ubuntu/work/tt/TT-FORGE-ARCHITECTURE.md)
- [tt-metal GitHub](https://github.com/tenstorrent/tt-metal)
- [UMD 仿真文档](tt_metal/third_party/umd/README.emu.md)
- [Wormhole B0 ISA 文档](/home/ubuntu/work/tensix/tt-isa-documentation/WormholeB0/) - 硬件架构详细说明

---

*创建时间: 2025-02*
*最后更新: 2025-02 (完成所有阶段分析，新增 Dispatch 架构深度剖析)*
