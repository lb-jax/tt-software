# 命令提交流程分析

## 概述

本文档详细分析 tt-metal 的命令提交流程，从高层 API 到底层硬件命令的完整链路。这是模拟器实现的核心部分。

## 架构概览

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Host (CPU)                                │
├─────────────────────────────────────────────────────────────────────┤
│  Program                                                            │
│     │                                                               │
│     ▼                                                               │
│  CommandQueue (高层接口)                                             │
│     │                                                               │
│     ▼                                                               │
│  HWCommandQueue (硬件命令队列)                                       │
│     │                                                               │
│     ├── DeviceCommand (命令构建器)                                   │
│     │                                                               │
│     ▼                                                               │
│  SystemMemoryManager (系统内存管理)                                  │
│     │                                                               │
│     ▼                                                               │
│  Hugepage (DMA 可访问内存)                                           │
└──────────────────────┬──────────────────────────────────────────────┘
                       │ PCIe DMA
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Device (Tenstorrent)                        │
├─────────────────────────────────────────────────────────────────────┤
│  Prefetcher Core                                                    │
│     │ (从 Hugepage 读取命令)                                        │
│     ▼                                                               │
│  Dispatcher Core                                                    │
│     │ (执行命令，写入 worker cores)                                 │
│     ▼                                                               │
│  Worker Cores (Tensix / ETH)                                        │
│     │ (执行 kernel)                                                 │
│     ▼                                                               │
│  Completion Queue                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

## 命令类型

### 1. Prefetcher 命令 (CQ_PREFETCH_CMD_*)

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

### 2. Dispatcher 命令 (CQ_DISPATCH_CMD_*)

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

### 3. Host 运行时命令

```cpp
// 位置: tt_metal/impl/dispatch/host_runtime_commands.hpp

enum class EnqueueCommandType {
    ENQUEUE_READ_BUFFER,       // 从设备读取 buffer
    ENQUEUE_WRITE_BUFFER,      // 写入 buffer 到设备
    ENQUEUE_RECORD_EVENT,      // 记录事件
    ENQUEUE_WAIT_FOR_EVENT,    // 等待事件
    FINISH,                    // 等待完成
    FLUSH,                     // 刷新队列
};
```

## 核心类分析

### 1. HWCommandQueue

```cpp
// 位置: tt_metal/impl/dispatch/hardware_command_queue.hpp

class HWCommandQueue {
private:
    uint32_t id_;                           // 队列 ID
    IDevice* device_;                       // 设备指针
    SystemMemoryManager& manager_;          // 系统内存管理器
    std::thread completion_queue_thread_;   // 完成队列读取线程

    // Prefetcher 缓存管理
    uint32_t prefetcher_cache_sizeB_;
    std::unique_ptr<RingbufferCacheManager> prefetcher_cache_manager_;

    // 配置缓冲区管理
    std::vector<WorkerConfigBufferMgr> config_buffer_mgr_;

public:
    // 主要方法
    void enqueue_record_event(...);         // 记录事件
    void read_completion_queue();           // 完成队列读取 (后台线程)
    void record_begin(...);                 // 开始 trace 记录
    void record_end();                      // 结束 trace 记录
};
```

### 2. DeviceCommand

```cpp
// 位置: tt_metal/impl/dispatch/device_command.hpp

template <bool hugepage_write>
class DeviceCommand {
private:
    uint32_t cmd_sequence_sizeB;           // 命令序列大小
    uint32_t cmd_write_offsetB = 0;        // 当前写入偏移
    void* cmd_region;                      // 命令缓冲区指针
    vector_aligned<uint32_t> cmd_region_vector;  // 非 hugepage 模式的缓冲区

public:
    // 命令构建方法
    void add_dispatch_wait(uint32_t flags, uint32_t address,
                           uint32_t stream, uint32_t count,
                           uint8_t dispatcher_type = 0);

    void add_dispatch_wait_with_prefetch_stall(...);

    void add_prefetch_relay_linear(uint32_t noc_xy_addr,
                                   DeviceAddr lengthB,
                                   uint32_t addr);

    void add_prefetch_relay_linear_h(...);

    void add_dispatch_terminate(DispatcherSelect selector);

    void add_prefetch_terminate();

    void add_prefetch_exec_buf_end();

    // 数据访问
    uint32_t size_bytes() const;
    void* data() const;
};

// 类型别名
using HugepageDeviceCommand = DeviceCommand<true>;   // 直接写入 hugepage
using HostDeviceCommand = DeviceCommand<false>;      // 写入普通内存
```

### 3. SystemMemoryManager

```cpp
// 位置: tt_metal/impl/dispatch/system_memory_manager.hpp

class SystemMemoryManager {
private:
    ChipId device_id;
    char* cq_sysmem_start;                 // 命令队列系统内存起始地址
    uint32_t cq_size;                      // 命令队列大小

    // Issue Queue (Host → Device 命令)
    std::vector<SystemMemoryCQInterface> cq_interfaces;

    // Completion Queue (Device → Host 完成通知)
    std::vector<uint32_t> completion_byte_addrs;

public:
    // Issue Queue 操作
    void* issue_queue_reserve(uint32_t cmd_size, uint32_t cq_id);
    void issue_queue_push_back(uint32_t size, uint32_t cq_id);

    // Fetch Queue 操作 (通知 prefetcher)
    void fetch_queue_reserve_back(uint32_t cq_id);
    void fetch_queue_write(uint32_t size, uint32_t cq_id);

    // Completion Queue 操作
    void completion_queue_wait_front(uint32_t cq_id,
                                     bool& exit_condition);
    uint32_t get_last_completed_event(uint32_t cq_id);

    // Bypass 模式 (用于 trace 记录)
    void set_bypass_mode(bool enable, bool clear);
    std::vector<uint32_t> get_bypass_data();
};
```

## 命令提交流程

### 1. 程序执行流程

```
用户调用 EnqueueProgram(cq, program, ...)
    │
    ▼
CommandQueue::enqueue_program(program)
    │
    ▼
program.compile(device)               ← 编译 kernel，生成二进制
    │
    ▼
program.generate_dispatch_commands()  ← 生成设备命令序列
    │
    ▼
HWCommandQueue::enqueue_program_impl()
    │
    ├── 分配系统内存: manager.issue_queue_reserve()
    │
    ├── 构建命令序列:
    │   ├── DeviceCommand::add_dispatch_wait()      ← 等待前一程序完成
    │   ├── DeviceCommand::add_prefetch_relay_linear() ← 加载 kernel 二进制
    │   ├── DeviceCommand::add_dispatch_write_linear() ← 写入配置数据
    │   └── DeviceCommand::add_dispatch_go()        ← 发送启动信号
    │
    ├── 提交命令: manager.issue_queue_push_back()
    │
    └── 通知 prefetcher: manager.fetch_queue_write()
```

### 2. Buffer 读写流程

```
EnqueueWriteBuffer(cq, buffer, data, ...)
    │
    ▼
HWCommandQueue::enqueue_write_buffer()
    │
    ├── 分配系统内存
    │
    ├── 复制数据到 hugepage
    │
    ├── 构建命令:
    │   ├── CQ_PREFETCH_CMD_RELAY_LINEAR_H (从 host 读取数据)
    │   └── CQ_DISPATCH_CMD_WRITE_LINEAR/PAGED (写入设备内存)
    │
    └── 提交命令
```

```
EnqueueReadBuffer(cq, buffer, data, ...)
    │
    ▼
HWCommandQueue::enqueue_read_buffer()
    │
    ├── 构建命令:
    │   ├── CQ_DISPATCH_CMD_WRITE_LINEAR_H (设备写入 host)
    │   └── 等待 completion queue
    │
    ├── 提交命令
    │
    └── 等待完成:
        └── completion_queue_thread 读取完成通知
```

### 3. 事件同步流程

```
EnqueueRecordEvent(cq, event)
    │
    ▼
HWCommandQueue::enqueue_record_event()
    │
    ├── 分配 event_id
    │
    ├── 构建命令:
    │   └── CQ_DISPATCH_CMD_WAIT (等待所有 worker 完成)
    │   └── 写入 event_id 到 completion queue
    │
    └── 等待 completion_queue_thread 读取 event
```

## Mock 设备支持

### SystemMemoryManager 中的 Mock 处理

```cpp
// 位置: tt_metal/impl/dispatch/system_memory_manager.cpp

inline bool is_mock_device() {
    return tt::tt_metal::MetalContext::instance()
        .get_cluster().get_target_device_type() == tt::TargetDevice::Mock;
}

SystemMemoryManager::SystemMemoryManager(ChipId device_id, uint8_t num_hw_cqs) {
    // ...

    if (is_mock_device()) {
        // Mock 模式: 使用简化的初始化
        this->cq_size = 65536;
        this->cq_sysmem_start = nullptr;  // 不分配实际内存
        this->channel_offset = 0;

        // 初始化事件计数器
        this->cq_to_event.resize(num_hw_cqs, 0);
        this->cq_to_last_completed_event.resize(num_hw_cqs, 0);

        // 创建接口 stub
        for (uint8_t cq_id = 0; cq_id < num_hw_cqs; cq_id++) {
            this->cq_interfaces.emplace_back(0, cq_id, this->cq_size, 0);
        }
        return;
    }

    // 真实硬件: 分配 hugepage 内存
    // ...
}
```

### Mock 模式的限制

- 不执行实际计算
- 所有读操作返回未定义数据
- 事件同步立即完成
- 用于测试流程，不用于验证正确性

## 模拟器实现要点

### 1. 需要实现的命令处理

对于功能模拟器，需要处理以下命令：

```cpp
// 必须实现
CQ_DISPATCH_CMD_WRITE_LINEAR     // 写入 L1 内存
CQ_DISPATCH_CMD_WRITE_PAGED      // 写入 DRAM
CQ_DISPATCH_CMD_WAIT             // 等待条件
CQ_DISPATCH_CMD_GO               // 启动 worker
CQ_DISPATCH_CMD_TERMINATE        // 终止

// 可选实现 (性能优化)
CQ_DISPATCH_CMD_WRITE_PACKED     // 打包写入
CQ_PREFETCH_CMD_EXEC_BUF         // Trace 回放
```

### 2. 内存模型

模拟器需要维护：

```cpp
class SimulatorMemory {
    // 每个 core 的 L1 内存 (1MB)
    std::map<CoreCoord, std::vector<uint8_t>> l1_memory;

    // DRAM bank (每个 8GB 或更多)
    std::vector<std::vector<uint8_t>> dram_banks;

    // Host 共享内存 (hugepage)
    std::vector<uint8_t> sysmem;

    // 信号量/计数器
    std::map<CoreCoord, std::map<uint32_t, std::atomic<uint32_t>>> semaphores;
};
```

### 3. 命令解析示例

```cpp
void Simulator::process_dispatch_command(const CQDispatchCmd& cmd) {
    switch (cmd.base.cmd_id) {
        case CQ_DISPATCH_CMD_WRITE_LINEAR: {
            // 解析目标地址
            uint32_t noc_xy = cmd.write_linear.noc_xy_addr;
            uint32_t addr = cmd.write_linear.addr;
            uint32_t size = cmd.write_linear.length;

            // 获取数据 (跟随命令的 inline 数据)
            const void* data = cmd_data_ptr;

            // 写入模拟内存
            CoreCoord core = noc_xy_to_core(noc_xy);
            memcpy(&l1_memory[core][addr], data, size);
            break;
        }

        case CQ_DISPATCH_CMD_WAIT: {
            uint32_t addr = cmd.wait.addr;
            uint32_t count = cmd.wait.count;

            // 等待指定地址的值达到 count
            while (read_l1(cmd.wait.stream, addr) < count) {
                // 模拟器可以直接跳过等待
            }
            break;
        }

        case CQ_DISPATCH_CMD_GO: {
            // 启动 worker cores
            trigger_kernel_execution();
            break;
        }
    }
}
```

## 命令数据结构

### CQPrefetchCmd 结构

```cpp
struct CQPrefetchCmd {
    CQPrefetchBaseCmd base;
    union {
        CQPrefetchRelayLinearCmd relay_linear;
        CQPrefetchRelayPagedCmd relay_paged;
        CQPrefetchRelayInlineCmd relay_inline;
        CQPrefetchExecBufCmd exec_buf;
    };
};

struct CQPrefetchRelayLinearCmd {
    uint32_t noc_xy_addr;    // NOC 坐标 (编码的 x,y)
    uint32_t addr;           // 目标地址
    uint32_t length;         // 数据长度
};
```

### CQDispatchCmd 结构

```cpp
struct CQDispatchCmd {
    CQDispatchBaseCmd base;
    union {
        CQDispatchWriteLinearCmd write_linear;
        CQDispatchWritePagedCmd write_paged;
        CQDispatchWritePackedCmd write_packed;
        CQDispatchWaitCmd wait;
        CQDispatchGoCmd go;
        CQDispatchDelayCmd delay;
    };
};

struct CQDispatchWriteLinearCmd {
    uint8_t num_mcast_dests;  // 多播目标数量
    uint8_t write_offset_index;
    uint16_t pad;
    uint32_t noc_xy_addr;     // NOC 坐标
    uint32_t addr;            // L1 地址
    uint32_t length;          // 数据长度
};

struct CQDispatchWaitCmd {
    uint8_t flags;            // 等待标志
    uint8_t stream;           // 流 ID
    uint16_t pad;
    uint32_t addr;            // 轮询地址
    uint32_t count;           // 目标计数
};
```

## 调试提示

### 1. 启用命令跟踪

```bash
export TT_METAL_LOGGER_LEVEL=DEBUG
export TT_METAL_LOGGER_TYPES=Dispatch
```

### 2. 关键日志位置

- `tt_metal/impl/dispatch/hardware_command_queue.cpp` - 命令入队
- `tt_metal/impl/dispatch/system_memory_manager.cpp` - 内存管理
- `tt_metal/impl/dispatch/device_command.cpp` - 命令构建

### 3. Tracy 性能分析

```bash
export TRACY_NO_INVARIANT_CHECK=1
# 运行程序后使用 Tracy profiler 分析
```

## 下一步

详见 [05-memory-management.md](./05-memory-management.md) - 内存管理分析

---

*更新时间: 2025-02*
