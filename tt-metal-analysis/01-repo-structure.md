# tt-metal 仓库结构分析

## 概述

tt-metal 是 Tenstorrent 的低层硬件抽象库，负责与硬件直接交互。本文档分析其目录结构和关键组件。

## 源码位置

```
/home/ubuntu/work/tt/tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/
```

## 顶层目录结构

| 目录 | 说明 |
|------|------|
| `tt_metal/` | 核心实现，包含 LLRT、设备管理等 |
| `tt_fabric/` | 多芯片互联 Fabric 实现 |
| `tt_metalium/` | Metalium 运行时扩展 |
| `ttnn/` | TT-NN (神经网络) 高层 API |
| `infra/` | 基础设施和工具 |
| `models/` | 模型实现示例 |
| `tests/` | 测试套件 |
| `scripts/` | 构建和工具脚本 |

## 核心组件：tt_metal/

### 目录结构

```
tt_metal/
├── llrt/                    # Low-Level Runtime (关键!)
│   ├── tt_cluster.cpp       # 集群管理，设备检测失败位置
│   ├── tt_cluster.hpp
│   ├── rtoptions.cpp        # 运行时选项解析
│   ├── rtoptions.hpp        # RunTimeOptions 类定义
│   ├── hal.cpp              # Hardware Abstraction Layer
│   └── llrt.cpp             # LLRT 主入口
├── third_party/
│   └── umd/                 # User Mode Driver (极其重要!)
├── impl/                    # 实现细节
│   └── context/
│       └── metal_context.cpp # MetalContext 单例
├── hw/                      # 硬件相关定义
└── soc_descriptors/         # SoC 描述文件 (YAML)
```

### LLRT (Low-Level Runtime) - 关键模块

LLRT 是 tt-metal 的核心运行时层，负责：
- 设备检测和初始化
- 集群管理
- 硬件抽象

#### 关键文件

| 文件 | 大小 | 说明 |
|------|------|------|
| `tt_cluster.cpp` | 69KB | 集群管理，**错误发生位置 (line 117)** |
| `rtoptions.hpp` | 30KB+ | 运行时选项定义，包含模拟器配置 |
| `rtoptions.cpp` | ~30KB | 环境变量解析 |
| `hal.cpp` | ~20KB | 硬件抽象层实现 |

## UMD (User Mode Driver) - 设备抽象层

UMD 是最底层的设备驱动抽象，是模拟器接入的**最佳位置**。

### 位置

```
tt_metal/third_party/umd/
```

### 目录结构

```
umd/
├── device/
│   ├── api/umd/device/
│   │   ├── cluster.hpp           # Cluster 类定义 (重要!)
│   │   ├── cluster_descriptor.hpp
│   │   ├── chip/
│   │   │   ├── chip.hpp          # Chip 基类 (抽象接口)
│   │   │   ├── local_chip.hpp    # 本地芯片 (PCIe 连接)
│   │   │   ├── remote_chip.hpp   # 远程芯片 (Ethernet 连接)
│   │   │   └── mock_chip.hpp     # Mock 芯片 (测试用)
│   │   ├── simulation/
│   │   │   ├── simulation_chip.hpp    # 仿真芯片基类
│   │   │   ├── tt_sim_chip.hpp        # TTSIM 实现
│   │   │   ├── rtl_simulation_chip.hpp # RTL 仿真
│   │   │   └── simulation_host.hpp
│   │   ├── tt_device/
│   │   │   ├── tt_device.hpp
│   │   │   ├── wormhole_tt_device.hpp
│   │   │   ├── blackhole_tt_device.hpp
│   │   │   ├── tt_sim_tt_device.hpp   # 仿真设备
│   │   │   └── rtl_simulation_tt_device.hpp
│   │   ├── pcie/
│   │   │   └── pci_device.hpp    # PCIe 设备接口
│   │   ├── arch/
│   │   │   ├── architecture_implementation.hpp
│   │   │   ├── wormhole_implementation.hpp
│   │   │   └── blackhole_implementation.hpp
│   │   └── types/
│   │       ├── arch.hpp          # ARCH 枚举
│   │       └── cluster_types.hpp
│   ├── cluster.cpp
│   ├── cluster_descriptor.cpp
│   └── ...
├── README.md
├── README.emu.md            # 仿真文档 (Zebu EP1)
└── emulation_rocky8.def     # 仿真容器定义
```

## 设备类型枚举

### tt_target_device.hpp (tt-metal 层)

```cpp
enum class TargetDevice {
    Silicon,    // 真实硬件
    Simulator,  // 软件仿真器
    Mock        // 测试用 Mock
};
```

### cluster.hpp (UMD 层)

```cpp
enum ChipType {
    SILICON,     // 真实硬件
    SIMULATION,  // 仿真模式
    MOCK,        // Mock 模式
};
```

## Chip 类继承体系

```
Chip (抽象基类)
├── LocalChip          # PCIe 连接的本地芯片
│   ├── WormholeChip
│   └── BlackholeChip
├── RemoteChip         # Ethernet 连接的远程芯片
├── MockChip           # 测试用 Mock
└── SimulationChip     # 仿真基类
    ├── TTSimChip      # TTSIM (.so 库)
    └── RTLSimulationChip  # RTL 仿真
```

## Chip 接口定义 (chip.hpp)

```cpp
class Chip {
public:
    virtual void start_device() = 0;
    virtual void close_device() = 0;
    virtual bool is_mmio_capable() const = 0;

    // 内存读写
    virtual void write_to_device(CoreCoord core, const void* src,
                                  uint64_t l1_dest, uint32_t size) = 0;
    virtual void read_from_device(CoreCoord core, void* dest,
                                   uint64_t l1_src, uint32_t size) = 0;

    // 系统内存操作
    virtual void write_to_sysmem(...) = 0;
    virtual void read_from_sysmem(...) = 0;

    // DMA 操作
    virtual void dma_write_to_device(...) = 0;
    virtual void dma_read_from_device(...) = 0;

    // RISC 复位控制
    virtual void send_tensix_risc_reset(...) = 0;
    virtual void deassert_risc_resets() = 0;

    // 内存屏障
    virtual void l1_membar(...) = 0;
    virtual void dram_membar(...) = 0;

    // ARC 消息
    virtual int arc_msg(...) = 0;
};
```

## 环境变量配置

### 关键环境变量

| 变量 | 说明 | 示例 |
|------|------|------|
| `TT_METAL_SIMULATOR` | 模拟器路径，启用 Simulator 模式 | `/path/to/simulator` |
| `TT_METAL_MOCK_CLUSTER_DESC_PATH` | Mock 集群描述符路径 | `/path/to/mock.yaml` |
| `ARCH_NAME` | 架构名称 (仿真模式必需) | `wormhole_b0`, `blackhole` |
| `TT_METAL_HOME` | tt-metal 根目录 | 通常自动设置 |

### 优先级

```
TT_METAL_SIMULATOR > TT_METAL_MOCK_CLUSTER_DESC_PATH
```

如果同时设置了两者，Simulator 优先。

### 启用 Mock 模式示例

```bash
# 使用预定义的 Mock 集群描述符
export TT_METAL_MOCK_CLUSTER_DESC_PATH=/path/to/tt-metal/tt_metal/third_party/umd/tests/cluster_descriptor_examples/wh_2chip_cluster.yaml
```

## 错误发生位置分析

### tt_cluster.cpp:117

```cpp
const auto num_chips = cluster_desc->get_all_chips().size();
TT_FATAL(num_chips > 0, "No chips detected in the cluster");  // Line 117
```

### 调用链

```
tt::tt_metal::MetalContext::instance()
    │
    ▼
tt::tt_metal::MetalContext::initialize_base_objects()
    │
    ▼
tt::Cluster::get_cluster_type_from_cluster_desc()  // tt_cluster.cpp
    │
    ▼
tt::umd::Cluster::create_cluster_descriptor()
    │
    ▼
检测 PCIe 设备 → 无设备 → cluster_desc->get_all_chips() 返回空
    │
    ▼
num_chips == 0 → TT_FATAL 失败
```

## 现有仿真支持

### 1. Zebu EP1 仿真 (README.emu.md)

- **类型**: 硬件 FPGA 仿真器
- **用途**: RTL 级别验证
- **接口**: `tt_emulation_device` + `libtt_zebu_wrapper.so`
- **限制**: 需要专用硬件 (Zebu 板)

### 2. TTSIM 软件仿真

- **类型**: 软件仿真
- **实现**: `TTSimChip` 类
- **接口**: 动态库 (.so)
- **使用**: 设置 `TT_METAL_SIMULATOR` 环境变量

### 3. Mock 模式

- **类型**: 功能 Mock
- **实现**: `MockChip` 类
- **接口**: 所有函数返回空/默认值
- **使用**: 设置 `TT_METAL_MOCK_CLUSTER_DESC_PATH`

## 模拟器接入点分析

### 最佳接入点: UMD Chip 接口

**优势:**
1. 最底层的设备抽象
2. 接口清晰、定义明确
3. 已有 `SimulationChip` 和 `MockChip` 作为参考
4. 对上层透明

**需要实现的方法:**
- `start_device()` / `close_device()`
- `write_to_device()` / `read_from_device()`
- `write_to_sysmem()` / `read_from_sysmem()`
- `dma_write_to_device()` / `dma_read_from_device()`
- `send_tensix_risc_reset()` / `deassert_risc_resets()`
- `l1_membar()` / `dram_membar()`
- `arc_msg()`

### 次选接入点: tt_cluster.cpp

**优势:**
- 可以快速绕过设备检测
- 修改量较少

**劣势:**
- 侵入性强
- 需要 fork/修改 tt-metal

## 集群描述符示例

UMD 提供了多个测试用的集群描述符:

```
tt_metal/third_party/umd/tests/cluster_descriptor_examples/
├── wh_2chip_cluster.yaml       # Wormhole 2芯片集群
├── galaxy_cluster.yaml         # Galaxy (TG) 集群
├── n300_cluster.yaml           # N300 集群
└── ...
```

## 下一步

1. [02-device-detection.md](./02-device-detection.md) - 设备检测流程详细分析
2. [03-umd-interface.md](./03-umd-interface.md) - UMD 接口完整分析
3. [04-command-submission.md](./04-command-submission.md) - 命令提交流程
4. [05-memory-management.md](./05-memory-management.md) - 内存管理
5. [06-simulator-interface.md](./06-simulator-interface.md) - 模拟器接口设计

---

*更新时间: 2025-02*
