# 内存管理分析

## 概述

本文档详细分析 tt-metal 的内存管理系统，包括设备内存布局、分配器实现、以及 Buffer 抽象。这是模拟器正确执行计算的关键。

## 设备内存架构

### Tenstorrent 芯片内存层次

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Tenstorrent 芯片 (Wormhole/Blackhole)          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │
│  │  DRAM       │  │  DRAM       │  │  DRAM       │  ... (多通道)   │
│  │  Channel 0  │  │  Channel 1  │  │  Channel 2  │                 │
│  │  (8GB+)     │  │  (8GB+)     │  │  (8GB+)     │                 │
│  └─────────────┘  └─────────────┘  └─────────────┘                 │
│        │               │               │                            │
│        └───────────────┼───────────────┘                            │
│                        │ NoC (Network on Chip)                      │
│        ┌───────────────┼───────────────┐                            │
│        ▼               ▼               ▼                            │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                      │
│  │ Tensix   │    │ Tensix   │    │ Tensix   │  ... (网格布局)      │
│  │ Core     │    │ Core     │    │ Core     │                      │
│  │ [0,0]    │    │ [1,0]    │    │ [2,0]    │                      │
│  │          │    │          │    │          │                      │
│  │ L1: 1MB  │    │ L1: 1MB  │    │ L1: 1MB  │                      │
│  └──────────┘    └──────────┘    └──────────┘                      │
│                                                                     │
│  ┌──────────┐    ┌──────────┐                                      │
│  │ ETH Core │    │ ETH Core │  (以太网核心)                         │
│  │ L1: 256KB│    │ L1: 256KB│                                      │
│  └──────────┘    └──────────┘                                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                        │
                        │ PCIe
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Host (CPU)                                  │
├─────────────────────────────────────────────────────────────────────┤
│  System Memory (Hugepage / DMA 可访问)                              │
│  - 命令队列                                                         │
│  - 数据传输缓冲区                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

### 内存类型枚举

```cpp
// 位置: tt_metal/api/tt-metalium/buffer_types.hpp

enum class BufferType {
    DRAM,           // 设备 DRAM (大容量，高延迟)
    L1,             // Tensix 核心 L1 (1MB，低延迟)
    SYSTEM_MEMORY,  // Host 系统内存
    L1_SMALL,       // L1 小缓冲区 (用于特定分配)
    TRACE,          // Trace 缓冲区 (DRAM 顶部)
};
```

### 内存布局 (TensorMemoryLayout)

```cpp
enum class TensorMemoryLayout {
    INTERLEAVED = 0,      // 交错分布在多个 bank
    HEIGHT_SHARDED = 2,   // 按高度分片到不同核心
    WIDTH_SHARDED = 3,    // 按宽度分片到不同核心
    BLOCK_SHARDED = 4,    // 2D 块分片
};
```

## L1 内存布局

### Tensix Core L1 内存映射

```
┌─────────────────────────────────────────┐ 0x0
│  Firmware / RISC 二进制                  │
│  (BRISC, NCRISC, TRISC0-2)              │
├─────────────────────────────────────────┤ l1_unreserved_base
│                                         │
│  用户可分配区域                          │
│  - Circular Buffers                     │
│  - Runtime Args                         │
│  - 临时数据                              │
│                                         │
├─────────────────────────────────────────┤
│  L1 Small 区域 (可选)                    │
├─────────────────────────────────────────┤ worker_l1_size (1MB)
│  保留区域 (调试/性能监控)                │
└─────────────────────────────────────────┘
```

### DRAM 内存映射

```
┌─────────────────────────────────────────┐ 0x0
│  保留区域                               │
├─────────────────────────────────────────┤ dram_unreserved_base
│                                         │
│  用户可分配区域                          │
│  - 权重/参数                             │
│  - 激活值                                │
│  - 中间结果                              │
│                                         │
├─────────────────────────────────────────┤
│  Trace 缓冲区区域                        │
│  (从顶部向下分配)                        │
└─────────────────────────────────────────┘ dram_bank_size
```

## 分配器架构

### 类层次结构

```
Allocator (API 层接口)
    │
    ▼
AllocatorImpl (实现基类)
    │
    ├── L1BankingAllocator (L1 分配器)
    │
    └── 管理多个 BankManager:
        ├── dram_manager_ (DRAM 分配)
        ├── l1_manager_ (L1 分配)
        ├── l1_small_manager_ (L1 小缓冲区)
        └── trace_buffer_manager_ (Trace 缓冲区)

BankManager
    │
    └── Algorithm (分配算法)
        └── FreeListOpt (优化的空闲链表)
```

### AllocatorConfig 配置

```cpp
// 位置: tt_metal/impl/allocator/allocator_types.hpp

struct AllocatorConfig {
    // DRAM 配置
    size_t num_dram_channels = 0;       // DRAM 通道数
    size_t dram_bank_size = 0;          // 每个 bank 大小
    std::vector<size_t> dram_bank_offsets;  // bank 偏移量
    uint32_t dram_unreserved_base = 0;  // 可用起始地址
    uint32_t dram_alignment = 0;        // 对齐要求

    // Worker (Tensix) L1 配置
    uint32_t l1_unreserved_base = 0;    // L1 可用起始地址
    CoreRangeSet worker_grid;           // Worker 核心网格
    size_t worker_l1_size = 0;          // 每个核心 L1 大小 (1MB)
    size_t l1_small_size = 0;           // L1 小缓冲区大小
    size_t trace_region_size = 0;       // Trace 区域大小

    // 核心类型映射
    std::unordered_map<CoreCoord, AllocCoreType> core_type_from_noc_coord_table;

    // 对齐和其他
    uint32_t l1_alignment = 0;
    bool disable_interleaved = false;   // 禁用交错分配
};
```

### AllocatorImpl 主要方法

```cpp
// 位置: tt_metal/impl/allocator/allocator.hpp

class AllocatorImpl {
public:
    // 构造函数
    explicit AllocatorImpl(const AllocatorConfig& alloc_config);

    // Buffer 分配
    DeviceAddr allocate_buffer(Buffer* buffer);
    void deallocate_buffer(Buffer* buffer);
    void deallocate_buffers();  // 释放所有

    // 查询方法
    uint32_t get_num_banks(const BufferType& buffer_type) const;
    DeviceAddr get_bank_size(const BufferType& buffer_type) const;
    CoreCoord get_logical_core_from_bank_id(uint32_t bank_id) const;
    int32_t get_bank_offset(BufferType buffer_type, uint32_t bank_id) const;

    // 统计
    Statistics get_statistics(const BufferType& buffer_type) const;

    // 状态管理 (用于 Trace)
    AllocatorState extract_state() const;
    void override_state(const AllocatorState& state);

private:
    std::unique_ptr<BankManager> dram_manager_;
    std::unique_ptr<BankManager> l1_manager_;
    std::unique_ptr<BankManager> l1_small_manager_;
    std::unique_ptr<BankManager> trace_buffer_manager_;

    std::unordered_set<Buffer*> allocated_buffers_;
    std::unique_ptr<AllocatorConfig> config_;
};
```

## BankManager 分析

### 功能概述

BankManager 管理一组内存 bank (可以是 DRAM 通道或 L1 核心)，提供统一的分配接口。

```cpp
// 位置: tt_metal/impl/allocator/bank_manager.hpp

class BankManager {
public:
    BankManager(
        const BufferType& buffer_type,
        const std::vector<int64_t>& bank_offsets,  // 每个 bank 的偏移
        DeviceAddr size_bytes,                      // 每个 bank 大小
        uint32_t alignment_bytes,                   // 对齐要求
        DeviceAddr alloc_offset = 0,               // 分配起始偏移
        bool disable_interleaved = false);

    // 分配/释放
    DeviceAddr allocate_buffer(
        DeviceAddr size,
        DeviceAddr page_size,
        bool bottom_up,                    // 分配方向
        const CoreRangeSet& compute_grid,
        std::optional<uint32_t> num_shards);  // 分片数 (可选)

    void deallocate_buffer(DeviceAddr address);

    // 查询
    uint32_t num_banks() const;
    DeviceAddr bank_size() const;
    int64_t bank_offset(uint32_t bank_id) const;

    Statistics get_statistics() const;
};
```

### 分配算法

```cpp
// 位置: tt_metal/impl/allocator/algorithms/free_list_opt.hpp

namespace allocator {

class FreeListOpt : public Algorithm {
public:
    FreeListOpt(DeviceAddr max_size, DeviceAddr min_allocation_size,
                DeviceAddr alignment, DeviceAddr offset);

    // 分配
    std::optional<DeviceAddr> allocate(
        DeviceAddr size,
        bool bottom_up = true,
        DeviceAddr address_limit = 0);  // 0 = 无限制

    // 分配并避开指定范围
    std::optional<DeviceAddr> allocate_avoiding_ranges(
        DeviceAddr size,
        bool bottom_up,
        const std::vector<std::pair<DeviceAddr, DeviceAddr>>& avoid_ranges,
        DeviceAddr address_limit = 0);

    // 释放
    void deallocate(DeviceAddr address);

    // 状态
    bool is_allocated(DeviceAddr address) const;
    Statistics get_statistics() const;
};

}  // namespace allocator
```

## Buffer 类

### Buffer 结构

```cpp
// 位置: tt_metal/api/tt-metalium/buffer.hpp

struct BufferConfig {
    IDevice* device;
    DeviceAddr size;       // 总大小 (字节)
    DeviceAddr page_size;  // 页大小
    BufferType buffer_type;
};

class Buffer {
public:
    // 构造
    Buffer(const BufferConfig& config);
    Buffer(const ShardedBufferConfig& config);  // 分片 Buffer

    // 属性
    IDevice* device() const;
    DeviceAddr address() const;      // 分配的起始地址
    DeviceAddr size() const;         // 总大小
    DeviceAddr page_size() const;    // 页大小
    BufferType buffer_type() const;

    // 分片信息
    bool is_sharded() const;
    const ShardSpec& shard_spec() const;
    TensorMemoryLayout memory_layout() const;

    // 内存映射
    uint32_t num_banks() const;
    uint32_t num_cores() const;
    CoreCoord core_from_bank_id(uint32_t bank_id) const;
};
```

### 分片 Buffer (ShardSpec)

```cpp
struct ShardSpec {
    CoreRangeSet grid;                    // 分片映射的核心集合
    std::array<uint32_t, 2> shape;        // 分片形状 [height, width]
    ShardOrientation orientation;          // ROW_MAJOR 或 COL_MAJOR

    uint32_t num_cores() const;
    uint32_t numel() const;  // shape[0] * shape[1]
};
```

## UMD 层内存接口

### Chip 内存操作

```cpp
// 位置: umd/device/api/umd/device/chip/chip.hpp

class Chip {
public:
    // L1 读写
    virtual void write_to_device(CoreCoord core, const void* src,
                                  uint64_t l1_dest, uint32_t size) = 0;
    virtual void read_from_device(CoreCoord core, void* dest,
                                   uint64_t l1_src, uint32_t size) = 0;

    // 系统内存 (Host) 读写
    virtual void write_to_sysmem(uint16_t channel, const void* src,
                                  uint64_t sysmem_dest, uint32_t size) = 0;
    virtual void read_from_sysmem(uint16_t channel, void* dest,
                                   uint64_t sysmem_src, uint32_t size) = 0;

    // DMA 操作 (大数据传输)
    virtual void dma_write_to_device(const void* src, size_t size,
                                      CoreCoord core, uint64_t addr) = 0;
    virtual void dma_read_from_device(void* dst, size_t size,
                                       CoreCoord core, uint64_t addr) = 0;

    // 多播写入
    virtual void noc_multicast_write(void* dst, size_t size,
                                      CoreCoord core_start, CoreCoord core_end,
                                      uint64_t addr) = 0;

    // 内存屏障
    virtual void l1_membar(const std::unordered_set<CoreCoord>& cores) = 0;
    virtual void dram_membar(const std::unordered_set<CoreCoord>& cores) = 0;
};
```

### MockChip 实现 (参考)

```cpp
// 位置: umd/device/chip/mock_chip.cpp

// 所有内存操作都是空操作 - Mock 不维护实际内存状态
void MockChip::write_to_device(CoreCoord core, const void* src,
                                uint64_t l1_dest, uint32_t size) {}
void MockChip::read_from_device(CoreCoord core, void* dest,
                                 uint64_t l1_src, uint32_t size) {}
void MockChip::write_to_sysmem(uint16_t channel, const void* src,
                                uint64_t sysmem_dest, uint32_t size) {}
void MockChip::read_from_sysmem(uint16_t channel, void* dest,
                                 uint64_t sysmem_src, uint32_t size) {}
void MockChip::dma_write_to_device(const void* src, size_t size,
                                    CoreCoord core, uint64_t addr) {}
void MockChip::dma_read_from_device(void* dst, size_t size,
                                     CoreCoord core, uint64_t addr) {}
```

### TTSimChip 实现

```cpp
// 位置: umd/device/simulation/tt_sim_chip.cpp

void TTSimChip::write_to_device(CoreCoord core, const void* src,
                                 uint64_t l1_dest, uint32_t size) {
    // 转换坐标系统
    auto translated = soc_descriptor_.translate_coord_to(core, CoordSystem::TRANSLATED);
    // 调用模拟器动态库
    tt_device_->write_to_device(src, translated, l1_dest, size);
}

void TTSimChip::read_from_device(CoreCoord core, void* dest,
                                  uint64_t l1_src, uint32_t size) {
    auto translated = soc_descriptor_.translate_coord_to(core, CoordSystem::TRANSLATED);
    tt_device_->read_from_device(dest, translated, l1_src, size);
}
```

## 模拟器内存实现

### 推荐的内存模型

```cpp
class SimulatorMemory {
public:
    // Wormhole B0 配置
    static constexpr size_t L1_SIZE = 1 * 1024 * 1024;      // 1 MB per core
    static constexpr size_t DRAM_SIZE = 8ULL * 1024 * 1024 * 1024;  // 8 GB per channel
    static constexpr size_t NUM_DRAM_CHANNELS = 12;
    static constexpr size_t TENSIX_GRID_X = 8;
    static constexpr size_t TENSIX_GRID_Y = 8;

private:
    // L1 内存: 每个 Tensix core 1MB
    // 使用稀疏映射避免预分配所有内存
    std::map<CoreCoord, std::vector<uint8_t>> l1_memory_;

    // DRAM 内存: 多通道
    std::vector<std::vector<uint8_t>> dram_channels_;

    // 系统内存 (模拟 Hugepage)
    std::vector<uint8_t> sysmem_;

public:
    // L1 操作
    void write_l1(CoreCoord core, uint64_t addr, const void* data, size_t size) {
        auto& mem = get_or_create_l1(core);
        if (addr + size > L1_SIZE) {
            throw std::runtime_error("L1 write out of bounds");
        }
        std::memcpy(mem.data() + addr, data, size);
    }

    void read_l1(CoreCoord core, uint64_t addr, void* data, size_t size) {
        auto& mem = get_or_create_l1(core);
        if (addr + size > L1_SIZE) {
            throw std::runtime_error("L1 read out of bounds");
        }
        std::memcpy(data, mem.data() + addr, size);
    }

    // DRAM 操作
    void write_dram(uint32_t channel, uint64_t addr, const void* data, size_t size) {
        if (channel >= NUM_DRAM_CHANNELS) {
            throw std::runtime_error("Invalid DRAM channel");
        }
        auto& mem = dram_channels_[channel];
        if (addr + size > DRAM_SIZE) {
            throw std::runtime_error("DRAM write out of bounds");
        }
        std::memcpy(mem.data() + addr, data, size);
    }

    void read_dram(uint32_t channel, uint64_t addr, void* data, size_t size) {
        if (channel >= NUM_DRAM_CHANNELS) {
            throw std::runtime_error("Invalid DRAM channel");
        }
        auto& mem = dram_channels_[channel];
        std::memcpy(data, mem.data() + addr, size);
    }

private:
    std::vector<uint8_t>& get_or_create_l1(CoreCoord core) {
        auto it = l1_memory_.find(core);
        if (it == l1_memory_.end()) {
            // 延迟分配
            auto [new_it, _] = l1_memory_.emplace(core, std::vector<uint8_t>(L1_SIZE, 0));
            return new_it->second;
        }
        return it->second;
    }
};
```

### 地址空间映射

```cpp
// NOC 地址编码
// Tenstorrent 使用 NOC (Network on Chip) 地址来标识目标

// NOC 地址格式 (64-bit):
// [63:36] - 保留
// [35:32] - NOC ID (通常为 0 或 1)
// [31:24] - X 坐标
// [23:16] - Y 坐标
// [15:0]  - 本地地址偏移

uint64_t encode_noc_addr(uint8_t x, uint8_t y, uint32_t local_addr) {
    return ((uint64_t)x << 24) | ((uint64_t)y << 16) | local_addr;
}

std::tuple<uint8_t, uint8_t, uint32_t> decode_noc_addr(uint64_t noc_addr) {
    uint8_t x = (noc_addr >> 24) & 0xFF;
    uint8_t y = (noc_addr >> 16) & 0xFF;
    uint32_t local = noc_addr & 0xFFFFFFFF;
    return {x, y, local};
}
```

### DRAM 地址映射

```cpp
// DRAM 通道到核心映射 (Wormhole)
// DRAM 核心在芯片边缘，有特定坐标

CoreCoord dram_channel_to_core(uint32_t channel) {
    // Wormhole DRAM 核心布局
    static const std::vector<CoreCoord> dram_cores = {
        {0, 0}, {0, 1}, {0, 2}, {0, 3},  // 左边缘
        {0, 4}, {0, 5}, {0, 6}, {0, 7},
        {9, 0}, {9, 1}, {9, 2}, {9, 3},  // 右边缘
    };
    return dram_cores[channel % dram_cores.size()];
}

// Bank ID 到地址计算
DeviceAddr calculate_address_in_bank(
    DeviceAddr base_address,
    uint32_t bank_id,
    DeviceAddr page_size,
    uint32_t num_banks)
{
    // 交错分配: 页面轮流分布在各 bank
    uint32_t page_index = base_address / page_size;
    uint32_t bank_index = page_index % num_banks;

    if (bank_index != bank_id) {
        return INVALID_ADDRESS;  // 此页不在此 bank
    }

    uint32_t page_in_bank = page_index / num_banks;
    return page_in_bank * page_size;
}
```

## 内存对齐要求

### 对齐常量

```cpp
// 常见对齐要求
constexpr uint32_t L1_ALIGNMENT = 16;           // L1 基本对齐
constexpr uint32_t DRAM_ALIGNMENT = 32;         // DRAM 对齐
constexpr uint32_t TILE_SIZE = 32 * 32 * 2;     // BFloat16 Tile (2KB)
constexpr uint32_t PAGE_ALIGNMENT = 4096;       // 页对齐 (某些操作)
constexpr uint32_t HUGEPAGE_SIZE = 1 << 30;     // 1GB Hugepage
```

### 地址对齐函数

```cpp
inline DeviceAddr align_up(DeviceAddr addr, uint32_t alignment) {
    return ((addr + alignment - 1) / alignment) * alignment;
}

inline DeviceAddr align_down(DeviceAddr addr, uint32_t alignment) {
    return (addr / alignment) * alignment;
}

// Buffer 大小对齐
DeviceAddr aligned_size(DeviceAddr size, DeviceAddr page_size, uint32_t num_banks) {
    // 确保每个 bank 分配相同大小
    DeviceAddr size_per_bank = align_up(size / num_banks, page_size);
    return size_per_bank * num_banks;
}
```

## 调试技巧

### 分配器统计

```cpp
// 获取内存使用统计
Statistics stats = allocator->get_statistics(BufferType::L1);

std::cout << "Total allocatable: " << stats.total_allocatable_size_bytes << "\n";
std::cout << "Total allocated: " << stats.total_allocated_bytes << "\n";
std::cout << "Total free: " << stats.total_free_bytes << "\n";
std::cout << "Largest free block: " << stats.largest_free_block_bytes << "\n";
```

### 内存块表

```cpp
// 转储内存块状态
allocator_impl->dump_memory_blocks(BufferType::L1, std::cout);

// 输出示例:
// Bank 0: [0x1000, 0x2000) allocated
// Bank 0: [0x2000, 0x10000) free
// Bank 1: [0x1000, 0x3000) allocated
// ...
```

### 环境变量

```bash
# 启用分配器调试日志
export TT_METAL_LOGGER_LEVEL=DEBUG
export TT_METAL_LOGGER_TYPES=Allocator

# 检测内存泄漏
export TT_METAL_DETECT_MEMORY_LEAK=1
```

## 下一步

详见 [06-simulator-interface.md](./06-simulator-interface.md) - 模拟器接口设计

---

*更新时间: 2025-02*
