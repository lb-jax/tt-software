# UMD (User Mode Driver) 接口完整分析

## 概述

UMD (User Mode Driver) 是 tt-metal 的最底层设备抽象层，为模拟器接入提供了最佳的接口点。本文档详细分析 UMD 的类层次结构、接口定义和实现方式。

## 源码位置

```
/home/ubuntu/work/tt/tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/third_party/umd/
```

## 类层次结构

### 完整继承图

```
Chip (抽象基类)
├── LocalChip              # PCIe 连接的本地芯片 (Silicon)
│   ├── WormholeChip
│   └── BlackholeChip
├── RemoteChip             # Ethernet 连接的远程芯片
├── MockChip               # 测试用 Mock (所有方法为空)
└── SimulationChip         # 仿真基类
    ├── TTSimChip          # TTSIM 软件仿真 (.so 动态库)
    └── RtlSimulationChip  # RTL 仿真 (Zebu EP1)
```

### 关键类文件

| 类 | 头文件 | 实现文件 |
|----|--------|----------|
| Chip | `api/umd/device/chip/chip.hpp` | `chip/chip.cpp` |
| LocalChip | `api/umd/device/chip/local_chip.hpp` | `chip/local_chip.cpp` |
| MockChip | `api/umd/device/chip/mock_chip.hpp` | `chip/mock_chip.cpp` |
| SimulationChip | `api/umd/device/simulation/simulation_chip.hpp` | `simulation/simulation_chip.cpp` |
| TTSimChip | `api/umd/device/simulation/tt_sim_chip.hpp` | `simulation/tt_sim_chip.cpp` |
| Cluster | `api/umd/device/cluster.hpp` | `cluster.cpp` |

## Chip 核心接口详解

### 生命周期管理

```cpp
class Chip {
public:
    // 启动设备，初始化硬件/模拟器
    virtual void start_device() = 0;

    // 关闭设备，释放资源
    virtual void close_device() = 0;

    // 是否支持 MMIO (Memory-Mapped I/O)
    virtual bool is_mmio_capable() const = 0;
};
```

### 设备内存操作

```cpp
class Chip {
public:
    // 写数据到设备 L1/DRAM
    // @param core: 目标核心坐标
    // @param src: 源数据指针
    // @param l1_dest: 设备端目标地址
    // @param size: 数据大小 (字节)
    virtual void write_to_device(CoreCoord core, const void* src,
                                  uint64_t l1_dest, uint32_t size) = 0;

    // 从设备 L1/DRAM 读数据
    virtual void read_from_device(CoreCoord core, void* dest,
                                   uint64_t l1_src, uint32_t size) = 0;

    // 寄存器读写 (比普通内存访问更慢但保证完成)
    virtual void write_to_device_reg(CoreCoord core, const void* src,
                                      uint64_t reg_dest, uint32_t size) = 0;
    virtual void read_from_device_reg(CoreCoord core, void* dest,
                                       uint64_t reg_src, uint32_t size) = 0;
};
```

### 系统内存操作 (主机内存)

```cpp
class Chip {
public:
    // 获取主机内存通道数量
    virtual int get_num_host_channels() = 0;

    // 获取指定通道的大小
    virtual int get_host_channel_size(uint32_t channel) = 0;

    // 写数据到系统内存 (设备可访问的主机内存)
    virtual void write_to_sysmem(uint16_t channel, const void* src,
                                  uint64_t sysmem_dest, uint32_t size) = 0;

    // 从系统内存读数据
    virtual void read_from_sysmem(uint16_t channel, void* dest,
                                   uint64_t sysmem_src, uint32_t size) = 0;
};
```

### DMA 操作

```cpp
class Chip {
public:
    // DMA 写入设备
    virtual void dma_write_to_device(const void* src, size_t size,
                                      CoreCoord core, uint64_t addr) = 0;

    // DMA 从设备读取
    virtual void dma_read_from_device(void* dst, size_t size,
                                       CoreCoord core, uint64_t addr) = 0;

    // NOC 多播写入
    virtual void noc_multicast_write(void* dst, size_t size,
                                      CoreCoord core_start, CoreCoord core_end,
                                      uint64_t addr);
};
```

### RISC-V 复位控制

```cpp
class Chip {
public:
    // 发送 Tensix RISC 复位信号到单个核心
    virtual void send_tensix_risc_reset(CoreCoord core,
                                         const TensixSoftResetOptions& soft_resets);

    // 发送复位信号到所有核心
    virtual void send_tensix_risc_reset(const TensixSoftResetOptions& soft_resets);

    // 取消 RISC 复位
    virtual void deassert_risc_resets() = 0;

    // 新 API: 精细控制指定 RISC 核心的复位
    virtual void assert_risc_reset(CoreCoord core, const RiscType selected_riscs);
    virtual void deassert_risc_reset(CoreCoord core, const RiscType selected_riscs,
                                      bool staggered_start);
};
```

### 内存屏障

```cpp
class Chip {
public:
    // L1 内存屏障 - 确保 L1 写入完成
    virtual void l1_membar(const std::unordered_set<CoreCoord>& cores = {}) = 0;

    // DRAM 内存屏障 - 确保 DRAM 写入完成
    virtual void dram_membar(const std::unordered_set<CoreCoord>& cores = {}) = 0;
    virtual void dram_membar(const std::unordered_set<uint32_t>& channels) = 0;

    // 非 MMIO 刷新 (用于 Ethernet 连接的远程芯片)
    virtual void wait_for_non_mmio_flush() = 0;
};
```

### ARC 处理器消息

```cpp
class Chip {
public:
    // 向 ARC 处理器发送消息
    // @param msg_code: 消息代码
    // @param wait_for_done: 是否等待完成
    // @param args: 消息参数
    // @param timeout_ms: 超时时间
    // @param return_3, return_4: 返回值
    virtual int arc_msg(
        uint32_t msg_code,
        bool wait_for_done = true,
        const std::vector<uint32_t>& args = {},
        const std::chrono::milliseconds timeout_ms = timeout::ARC_MESSAGE_TIMEOUT,
        uint32_t* return_3 = nullptr,
        uint32_t* return_4 = nullptr);
};
```

### 其他方法

```cpp
class Chip {
public:
    // 获取设备时钟频率
    virtual int get_clock() = 0;

    // 获取 NUMA 节点
    virtual int get_numa_node() = 0;

    // 设置电源状态
    virtual void set_power_state(DevicePowerState state);

    // 设置远程传输使用的以太网核心
    virtual void set_remote_transfer_ethernet_cores(
        const std::unordered_set<CoreCoord>& cores) = 0;
};
```

## MockChip 实现分析

`MockChip` 是最简单的 `Chip` 实现，所有方法都是空操作或返回默认值：

```cpp
// mock_chip.cpp - 完整实现
MockChip::MockChip(SocDescriptor soc_descriptor)
    : Chip(std::move(soc_descriptor)) {}

bool MockChip::is_mmio_capable() const { return false; }
void MockChip::start_device() {}
void MockChip::close_device() {}
TTDevice* MockChip::get_tt_device() { return nullptr; }
SysmemManager* MockChip::get_sysmem_manager() { return nullptr; }
TLBManager* MockChip::get_tlb_manager() { return nullptr; }
int MockChip::get_num_host_channels() { return 0; }
int MockChip::get_host_channel_size(uint32_t channel) { return 0; }

// 所有内存操作都是空操作
void MockChip::write_to_sysmem(...) {}
void MockChip::read_from_sysmem(...) {}
void MockChip::write_to_device(...) {}
void MockChip::read_from_device(...) {}
void MockChip::write_to_device_reg(...) {}
void MockChip::read_from_device_reg(...) {}
void MockChip::dma_write_to_device(...) {}
void MockChip::dma_read_from_device(...) {}

// ARC 消息返回成功
int MockChip::arc_msg(...) {
    *return_3 = 1;  // 表示成功
    return 0;
}

// 其他方法都是空操作或返回 0
void MockChip::l1_membar(...) {}
void MockChip::dram_membar(...) {}
void MockChip::send_tensix_risc_reset(...) {}
void MockChip::deassert_risc_resets() {}
int MockChip::get_clock() { return 0; }
int MockChip::get_numa_node() { return 0; }
```

**用途**: 测试代码流程，不执行实际计算

## SimulationChip 实现分析

`SimulationChip` 是仿真芯片的基类，提供了通用实现：

### 构造函数

```cpp
SimulationChip::SimulationChip(
    const std::filesystem::path& simulator_directory,
    const SocDescriptor& soc_descriptor,
    ChipId chip_id,
    int num_host_mem_channels)
    : Chip(soc_descriptor),
      arch_name(soc_descriptor.arch),
      chip_id_(chip_id),
      simulator_directory_(simulator_directory)
{
    // 创建仿真用的系统内存管理器
    sysmem_manager_ = std::make_unique<SimulationSysmemManager>(num_host_mem_channels);
}
```

### 工厂方法

```cpp
std::unique_ptr<SimulationChip> SimulationChip::create(
    const std::filesystem::path& simulator_directory,
    const SocDescriptor& soc_descriptor,
    ChipId chip_id,
    size_t num_chips,
    int num_host_mem_channels)
{
    if (simulator_directory.extension() == ".so") {
        // 动态库模式 → TTSimChip
        return std::make_unique<TTSimChip>(...);
    } else {
        // 目录模式 → RTL 仿真
        return std::make_unique<RtlSimulationChip>(...);
    }
}
```

### 默认实现

```cpp
// 寄存器读写 → 委托给普通内存读写
void SimulationChip::write_to_device_reg(...) { write_to_device(...); }
void SimulationChip::read_from_device_reg(...) { read_from_device(...); }

// DMA 操作 → 委托给普通内存读写
void SimulationChip::dma_write_to_device(...) { write_to_device(...); }
void SimulationChip::dma_read_from_device(...) { read_from_device(...); }

// NOC 多播 → 循环执行单播
void SimulationChip::noc_multicast_write(...) {
    for (x = start.x; x <= end.x; ++x) {
        for (y = start.y; y <= end.y; ++y) {
            write_to_device(CoreCoord(x, y, ...), ...);
        }
    }
}

// 内存屏障 → 空操作 (仿真环境中不需要)
void SimulationChip::l1_membar(...) {}
void SimulationChip::dram_membar(...) {}
void SimulationChip::wait_for_non_mmio_flush() {}

// ARC 消息 → 返回成功
int SimulationChip::arc_msg(...) {
    *return_3 = 1;
    return 0;
}
```

### 纯虚函数 (子类必须实现)

```cpp
virtual void start_device() = 0;
virtual void close_device() = 0;
virtual void write_to_device(CoreCoord core, const void* src,
                              uint64_t l1_dest, uint32_t size) = 0;
virtual void read_from_device(CoreCoord core, void* dest,
                               uint64_t l1_src, uint32_t size) = 0;
virtual void send_tensix_risc_reset(...) = 0;
virtual void assert_risc_reset(...) = 0;
virtual void deassert_risc_reset(...) = 0;
```

## TTSIM 接口详解

### TTSimChip 架构

```
TTSimChip
├── SimulationChip (基类)
│   └── sysmem_manager_ (SimulationSysmemManager)
└── tt_device_ (TTSimTTDevice)
    └── libttsim_handle (dlopen 加载的 .so 库)
```

### 动态库接口

TTSimTTDevice 通过 `dlsym` 加载以下函数：

```cpp
// 必须导出的函数签名
void libttsim_init();
void libttsim_exit();
uint32_t libttsim_pci_config_rd32(uint32_t bus_device_function, uint32_t offset);
void libttsim_tile_rd_bytes(uint32_t x, uint32_t y, uint64_t addr, void* p, uint32_t size);
void libttsim_tile_wr_bytes(uint32_t x, uint32_t y, uint64_t addr, const void* p, uint32_t size);
void libttsim_clock(uint32_t n_clocks);
```

### 函数详解

| 函数 | 说明 |
|------|------|
| `libttsim_init()` | 初始化模拟器 |
| `libttsim_exit()` | 关闭模拟器 |
| `libttsim_pci_config_rd32(bdf, offset)` | 读取 PCI 配置空间 |
| `libttsim_tile_rd_bytes(x, y, addr, p, size)` | 从核心 (x,y) 的地址 addr 读取 size 字节到 p |
| `libttsim_tile_wr_bytes(x, y, addr, p, size)` | 向核心 (x,y) 的地址 addr 写入 p 处的 size 字节 |
| `libttsim_clock(n)` | 推进模拟器时钟 n 个周期 |

### PCI 设备 ID

```cpp
// tt_sim_tt_device.cpp 中的设备 ID 检测
void TTSimTTDevice::start_device() {
    pfn_libttsim_init();
    uint32_t pci_id = pfn_libttsim_pci_config_rd32(0, 0);
    uint32_t vendor_id = pci_id & 0xFFFF;      // 应该是 0x1E52 (Tenstorrent)
    libttsim_pci_device_id = pci_id >> 16;     // 设备 ID
}

// 支持的设备 ID:
// 0x401E - Wormhole
// 0xB140 - Blackhole
// 0xFEED - Quasar
```

### 模拟器目录结构

```
simulator_directory/
├── libttsim.so           # 模拟器动态库
└── soc_descriptor.yaml   # SoC 描述文件
```

## SimulationSysmemManager

仿真环境的系统内存管理器：

```cpp
class SimulationSysmemManager : public SysmemManager {
private:
    std::vector<uint8_t> system_memory_;  // 使用普通内存模拟系统内存
};
```

**功能**: 使用 `std::vector<uint8_t>` 模拟设备可访问的主机内存区域。

## SoC 描述文件格式

### wormhole_b0_80_arch.yaml 示例

```yaml
# 芯片网格大小
grid:
  x_size: 10
  y_size: 12

# ARC 核心位置
arc:
  [ 0-10 ]

# PCIe 位置
pcie:
  [ 0-3 ]

# DRAM 核心位置 (每行是一个 DRAM 通道)
dram:
  [
    [0-0, 0-1, 0-11],  # Channel 0
    [0-5, 0-6, 0-7],   # Channel 1
    # ...
  ]

# 以太网核心
eth:
  [ 9-0, 1-0, 8-0, 2-0, ... ]

# 功能核心 (Tensix) - 所有不是特殊用途的核心
functional_workers:
  [
    1-1, 2-1, 3-1, 4-1, 6-1, 7-1, 8-1,
    # ...
  ]
```

## 实现自定义模拟器的步骤

### 方案一：实现 TTSIM 动态库

1. **创建 .so 库**，导出以下函数：

```cpp
extern "C" {
    void libttsim_init();
    void libttsim_exit();
    uint32_t libttsim_pci_config_rd32(uint32_t bdf, uint32_t offset);
    void libttsim_tile_rd_bytes(uint32_t x, uint32_t y, uint64_t addr,
                                 void* p, uint32_t size);
    void libttsim_tile_wr_bytes(uint32_t x, uint32_t y, uint64_t addr,
                                 const void* p, uint32_t size);
    void libttsim_clock(uint32_t n_clocks);
}
```

2. **创建 soc_descriptor.yaml** 描述目标架构

3. **使用方式**：
```bash
export TT_METAL_SIMULATOR=/path/to/libttsim.so
```

### 方案二：继承 SimulationChip

1. 创建新类继承 `SimulationChip`
2. 实现纯虚函数：
   - `start_device()` / `close_device()`
   - `write_to_device()` / `read_from_device()`
   - `send_tensix_risc_reset()`
   - `assert_risc_reset()` / `deassert_risc_reset()`

3. 修改 `SimulationChip::create()` 工厂方法以支持新类型

### 方案三：直接继承 Chip

完全自定义实现，参考 `MockChip` 的结构。

## Cluster 管理

### ClusterOptions

```cpp
struct ClusterOptions {
    ChipType chip_type = ChipType::SILICON;
    uint32_t num_host_mem_ch_per_mmio_device = 0;
    bool perform_harvesting = true;
    std::string sdesc_path = "";
    std::unordered_set<ChipId> target_devices = {};
    ClusterDescriptor* cluster_descriptor = nullptr;
    std::filesystem::path simulator_directory = "";  // 仅用于 SIMULATION
};
```

### Cluster 初始化流程

```cpp
Cluster::Cluster(ClusterOptions options) {
    // 1. 根据 chip_type 创建芯片
    for (auto chip_id : target_devices) {
        chips_[chip_id] = construct_chip_from_cluster(chip_id, chip_type, ...);
    }
}

std::unique_ptr<Chip> Cluster::construct_chip_from_cluster(...) {
    switch (chip_type) {
        case ChipType::SILICON:
            return LocalChip::create(physical_device_id, ...);
        case ChipType::SIMULATION:
            return SimulationChip::create(simulator_directory, ...);
        case ChipType::MOCK:
            return std::make_unique<MockChip>(soc_desc);
    }
}
```

## 核心坐标系统

### CoreCoord 类

```cpp
struct CoreCoord {
    uint32_t x;
    uint32_t y;
    CoreType core_type;      // TENSIX, DRAM, ETH, ARC, PCIE
    CoordSystem coord_system; // PHYSICAL, TRANSLATED, VIRTUAL
};
```

### 坐标系统转换

```cpp
// 在 SocDescriptor 中进行坐标转换
tt_xy_pair SocDescriptor::translate_coord_to(CoreCoord coord, CoordSystem target);
```

## 下一步

详见:
- [04-command-submission.md](./04-command-submission.md) - 命令提交流程
- [05-memory-management.md](./05-memory-management.md) - 内存管理
- [06-simulator-interface.md](./06-simulator-interface.md) - 模拟器接口设计方案

---

*更新时间: 2025-02*
