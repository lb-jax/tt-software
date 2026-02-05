# 设备检测流程分析

## 概述

本文档详细分析 tt-metal 的设备检测和初始化流程，以及如何通过配置绕过硬件检测。

## 调用链

### 完整调用链

```
Python: xm.xla_device()
    │
    ▼
torch_xla._XLAC._xla_get_default_device()
    │
    ▼
C++: torch_xla::runtime::PjRtComputationClient::Initialize()
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
tt::tt_metal::MetalContext::instance()  ← 单例初始化
    │
    ▼
tt::tt_metal::MetalContext::initialize_base_objects()
    │
    ▼
tt::Cluster::Cluster(rtoptions, hal)    ← tt_cluster.cpp
    │
    ▼
tt::Cluster::detect_arch_and_target()
    │
    ▼
tt::Cluster::initialize_device_drivers()
    │
    ├── open_driver()
    │   │
    │   ├── [Silicon] tt::umd::Cluster::create_cluster_descriptor()
    │   │       │
    │   │       ▼
    │   │   检测 PCIe 设备 → 无硬件时返回空集群
    │   │
    │   ├── [Simulator] 使用 simulator_directory
    │   │
    │   └── [Mock] 加载 mock_cluster_desc.yaml
    │
    └── generate_cluster_descriptor()
            │
            ▼
        get_cluster_type_from_cluster_desc()
            │
            ▼
        TT_FATAL(num_chips > 0)  ← 错误位置: tt_cluster.cpp:117
```

## 关键代码分析

### 1. MetalContext 单例 (metal_context.cpp)

```cpp
// 位置: impl/context/metal_context.cpp
MetalContext& MetalContext::instance() {
    static MetalContext context;  // 首次调用时初始化
    return context;
}

void MetalContext::initialize_base_objects() {
    // 创建 HAL (Hardware Abstraction Layer)
    hal_ = std::make_unique<tt::tt_metal::Hal>(rtoptions_);

    // 创建 Cluster (设备检测在这里发生)
    cluster_ = std::make_unique<tt::Cluster>(rtoptions_, *hal_);
}
```

### 2. 目标设备检测 (tt_cluster.cpp)

```cpp
// 位置: llrt/tt_cluster.cpp
void Cluster::detect_arch_and_target() {
    // 从 rtoptions 获取目标设备类型
    this->target_type_ = rtoptions_.get_target_device();
    // 获取架构类型
    this->arch_ = tt_metal::get_platform_architecture(rtoptions_);

    // 验证目标类型
    TT_FATAL(
        this->target_type_ == tt::TargetDevice::Silicon ||
        this->target_type_ == tt::TargetDevice::Simulator ||
        this->target_type_ == tt::TargetDevice::Mock,
        "Target type={} is not supported", this->target_type_);
}
```

### 3. 驱动打开 (tt_cluster.cpp:open_driver)

```cpp
void Cluster::open_driver() {
    std::string sdesc_path = get_soc_description_file(this->arch_, this->target_type_, rtoptions_);

    if (this->target_type_ == TargetDevice::Silicon) {
        // 真实硬件模式: 检测 PCIe 设备
        auto temp_cluster_desc = tt::umd::Cluster::create_cluster_descriptor();
        // ... 如果无硬件，temp_cluster_desc 为空

        device_driver = std::make_unique<tt::umd::Cluster>(tt::umd::ClusterOptions{
            .chip_type = tt::umd::ChipType::SILICON,
            // ...
        });

    } else if (this->target_type_ == TargetDevice::Simulator) {
        // 模拟器模式
        device_driver = std::make_unique<tt::umd::Cluster>(tt::umd::ClusterOptions{
            .chip_type = tt::umd::ChipType::SIMULATION,
            .target_devices = {0},  // 默认单设备
            .simulator_directory = rtoptions_.get_simulator_path(),
        });

    } else if (this->target_type_ == TargetDevice::Mock) {
        // Mock 模式: 从 YAML 加载集群描述
        auto mock_cluster_desc = get_mock_cluster_desc(rtoptions_);

        device_driver = std::make_unique<tt::umd::Cluster>(tt::umd::ClusterOptions{
            .chip_type = tt::umd::ChipType::MOCK,
            .sdesc_path = sdesc_path,
            .cluster_descriptor = mock_cluster_desc.get(),
        });
    }
}
```

### 4. 错误发生位置

```cpp
// 位置: llrt/tt_cluster.cpp:117
tt::tt_metal::ClusterType Cluster::get_cluster_type_from_cluster_desc(
    const llrt::RunTimeOptions& rtoptions,
    const umd::ClusterDescriptor* cluster_desc)
{
    // ...
    const auto num_chips = cluster_desc->get_all_chips().size();
    TT_FATAL(num_chips > 0, "No chips detected in the cluster");  // ← LINE 117
    // ...
}
```

## 目标设备类型

### TargetDevice 枚举

```cpp
// 位置: tt_target_device.hpp
enum class TargetDevice {
    Silicon,    // 真实硬件 (默认)
    Simulator,  // 软件模拟器
    Mock        // 测试用 Mock
};
```

### 设置方式

| 目标类型 | 环境变量 | 说明 |
|----------|----------|------|
| Silicon | (默认) | 需要真实 Tenstorrent 硬件 |
| Simulator | `TT_METAL_SIMULATOR=/path/to/sim` | 需要模拟器二进制文件 |
| Mock | `TT_METAL_MOCK_CLUSTER_DESC_PATH=/path/to/yaml` | 只需 YAML 描述文件 |

### 优先级

```
Simulator > Mock > Silicon (默认)
```

如果同时设置 `TT_METAL_SIMULATOR` 和 `TT_METAL_MOCK_CLUSTER_DESC_PATH`，Simulator 优先。

## Mock 模式详解

### 环境变量设置

```bash
# 方式1: 使用预定义的集群描述符
export TT_METAL_MOCK_CLUSTER_DESC_PATH=/path/to/tt-metal/tt_metal/third_party/umd/tests/cluster_descriptor_examples/wormhole_N150.yaml

# 方式2: 使用自定义描述符
export TT_METAL_MOCK_CLUSTER_DESC_PATH=/path/to/custom_cluster.yaml
```

### 集群描述符 YAML 格式

```yaml
# 最小化的 Wormhole N150 配置
arch: {
   0: Wormhole,      # chip_id: 架构类型
}

chips: {
   0: [0,0,0,0],     # chip_id: [ethernet coordinates]
}

ethernet_connections: [
   # 芯片间的以太网连接 (单芯片可为空)
]

chips_with_mmio: [
   0: 0,             # chip_id: mmio_device_id
]

harvesting: {
   0: {noc_translation: true, harvest_mask: 32},
}

boardtype: {
   0: n150,          # 板卡类型
}
```

### 支持的 boardtype 值

- `n150` - N150 单芯片
- `n300` - N300 双芯片
- `p100` - Blackhole P100
- `p150` - Blackhole P150
- `p300` - Blackhole P300
- `galaxy` / `ubb` - Galaxy (TG) 集群

## Simulator 模式详解

### 环境变量设置

```bash
export TT_METAL_SIMULATOR=/path/to/simulator/directory
# 可选: 结合 Mock 集群描述符
export TT_METAL_MOCK_CLUSTER_DESC_PATH=/path/to/cluster.yaml
```

### 要求

Simulator 模式需要:
1. 模拟器目录包含:
   - SoC 描述符 YAML
   - 模拟器动态库 (.so)
2. 模拟器实现 `SimulationChip` 接口

### 相关类

```
SimulationChip (抽象基类)
├── TTSimChip       # TTSIM 软件模拟
└── RTLSimulationChip  # RTL 仿真 (Zebu)
```

## 快速验证 Mock 模式

### 测试脚本

```python
#!/usr/bin/env python3
import os

# 设置 Mock 模式
os.environ['TT_METAL_MOCK_CLUSTER_DESC_PATH'] = \
    '/path/to/tt-metal/tt_metal/third_party/umd/tests/cluster_descriptor_examples/wormhole_N150.yaml'

# 尝试初始化
try:
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
    print(f"Device: {device}")
except Exception as e:
    print(f"Error: {e}")
```

### 预期结果

Mock 模式下:
- 设备初始化成功
- 所有设备操作返回空/默认值
- **不执行实际计算** (仅用于测试流程)

## UMD 层设备检测

### create_cluster_descriptor (Silicon 模式)

```cpp
// 位置: umd/device/cluster_descriptor.cpp
std::unique_ptr<ClusterDescriptor> Cluster::create_cluster_descriptor(
    std::string sdesc_path, IODeviceType device_type)
{
    if (device_type == IODeviceType::PCIe) {
        // 扫描 PCIe 总线
        auto pci_devices = PCIDevice::enumerate_devices();
        if (pci_devices.empty()) {
            // 无设备 → 返回空集群描述符
            return std::make_unique<ClusterDescriptor>();
        }
        // ... 构建集群描述符
    }
}
```

### 芯片创建工厂 (cluster.cpp)

```cpp
// 位置: umd/device/cluster.cpp
std::unique_ptr<Chip> Cluster::construct_chip_from_cluster(...) {
    switch (chip_type) {
        case ChipType::SILICON:
            // 创建真实硬件芯片对象
            return std::make_unique<LocalChip>(...);

        case ChipType::SIMULATION:
            // 创建模拟芯片对象
            return SimulationChip::create(simulator_directory, ...);

        case ChipType::MOCK:
            // 创建 Mock 芯片对象
            return std::make_unique<MockChip>(soc_desc);
    }
}
```

## 绕过设备检测的方法

### 方法 1: Mock 模式 (推荐用于测试)

**优点:**
- 不需要额外软件
- 只需 YAML 配置文件
- 快速验证流程

**缺点:**
- 不执行实际计算
- 所有操作返回空值

### 方法 2: Simulator 模式 (需要模拟器)

**优点:**
- 可以执行实际计算 (取决于模拟器)
- 验证完整执行流程

**缺点:**
- 需要实现模拟器
- 需要 SoC 描述符

### 方法 3: 自定义 Chip 实现 (模拟器开发)

**步骤:**
1. 继承 `SimulationChip` 或 `Chip`
2. 实现所有虚函数
3. 在 `open_driver()` 中集成

## 下一步

详见 [03-umd-interface.md](./03-umd-interface.md) - UMD 接口完整分析

---

*更新时间: 2025-02*
