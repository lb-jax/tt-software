# 运行时执行流程深度剖析

本文档详细解释 TT-MLIR 编译后的代码如何在运行时被加载和执行，涵盖从 Flatbuffer 加载到硬件内核执行的完整流程。

## 目录

1. [运行时架构概述](#运行时架构概述)
2. [Flatbuffer 加载与验证](#flatbuffer-加载与验证)
3. [设备初始化与管理](#设备初始化与管理)
4. [张量创建与内存分配](#张量创建与内存分配)
5. [TTNN Runtime 执行流程](#ttnn-runtime-执行流程)
6. [TTMetal Dispatch 架构](#ttmetal-dispatch-架构)
7. [内核执行机制](#内核执行机制)
8. [完整端到端执行流程](#完整端到端执行流程)
9. [性能优化机制](#性能优化机制)
10. [调试与监控](#调试与监控)

---

## 运行时架构概述

### 1.1 两层运行时设计

TT-MLIR 采用分层运行时架构，提供不同抽象级别的执行能力：

```
┌─────────────────────────────────────────────────────────────────┐
│                    应用层 (Python/C++ API)                      │
│  - PyTorch/JAX 前端                                             │
│  - 用户代码                                                      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────────┐
│                  TT-MLIR Runtime API                            │
│  - tt::runtime::submit()                                        │
│  - tt::runtime::openMeshDevice()                                │
│  - tt::runtime::createTensor()                                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
┌────────▼────────┐  ┌─────▼──────┐  ┌──────▼──────────┐
│  TTNN Runtime   │  │  Binary     │  │  TTMetal        │
│  (High-level)   │  │  Loader     │  │  Runtime        │
│                 │  │             │  │  (Low-level)    │
│  - 操作执行     │  │  - Schema   │  │  - 命令调度     │
│  - 自动内存管理 │  │    解析     │  │  - 显式内存     │
│  - 算子融合     │  │  - 版本检查 │  │  - 内核加载     │
└────────┬────────┘  └─────┬──────┘  └──────┬──────────┘
         │                 │                 │
         └─────────────────┼─────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────────┐
│                    TT-Metalium 库                               │
│  - TTNN Operations (ttnn::matmul, ttnn::conv2d, ...)           │
│  - TTMetal Core (Device, Program, Buffer, ...)                 │
│  - Hardware Abstraction Layer                                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────────┐
│              Low-Level Runtime (LLRT)                           │
│  - Command Queue Management                                     │
│  - Prefetcher/Dispatcher Firmware                               │
│  - DMA and NoC Control                                          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────────┐
│                  Hardware (Tenstorrent ASIC)                    │
│  - Wormhole N150/N300 (80 Tensix cores, 12 GB DRAM)           │
│  - Blackhole P150B (80 Tensix cores)                           │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 TTNN Runtime vs TTMetal Runtime

| 特性 | TTNN Runtime | TTMetal Runtime |
|------|-------------|----------------|
| **抽象级别** | 高（算子级别） | 低（内核级别） |
| **操作粒度** | matmul, conv2d, attention | EnqueueProgram, CreateBuffer |
| **内存管理** | 自动（运行时分配） | 手动（显式管理） |
| **性能控制** | 有限（依赖库优化） | 完全控制（细粒度调度） |
| **开发难度** | 简单 | 复杂 |
| **适用场景** | 快速原型、标准模型 | 性能调优、自定义内核 |
| **Flatbuffer 格式** | `.ttnn` | `.ttm` |
| **编译 Pass** | TTNNToFlatbuffer | TTMetalToFlatbuffer |

### 1.3 职责划分

**TTNN Runtime**：
- 执行高层操作（matmul、conv2d 等）
- 自动选择最优算法
- 管理临时缓冲区
- 处理布局转换

**TTMetal Runtime**：
- 直接调度内核
- 管理命令队列
- 控制内存分配
- 处理同步和等待

**共同依赖的 TT-Metalium**：
- 设备管理（Device、MeshDevice）
- 缓冲区分配（Buffer、Allocator）
- 命令队列（CommandQueue）
- 底层通信（NoC、DMA）

---

## Flatbuffer 加载与验证

### 2.1 Flatbuffer 文件格式

编译器生成的二进制文件（`.ttnn` 或 `.ttm`）包含以下结构：

```
.ttnn 文件结构
├── Magic Header (4 bytes)                  "TTML"
├── Version Information
│   ├── major: uint8                        编译器主版本
│   ├── minor: uint8                        编译器次版本
│   ├── patch: uint8                        编译器补丁版本
│   └── hash: string                        Git commit hash
│
├── SystemDesc (硬件配置描述)
│   ├── chip_descs: [ChipDesc]
│   │   ├── arch: ChipArch                  Wormhole, Blackhole
│   │   ├── grid_size: Dim2d                {y: 8, x: 10}
│   │   ├── l1_size: uint64                 1499136 (1.4 MB)
│   │   ├── num_dram_channels: uint32       12
│   │   ├── dram_bank_size: uint64          1 GB per channel
│   │   ├── noc_x_size: uint32              12
│   │   ├── noc_y_size: uint32              10
│   │   └── supported_data_types: [DataType]
│   └── chip_desc_by_chip_id: {int: ChipDesc}
│
├── Programs (执行程序列表)
│   ├── Program 0: "forward"
│   │   ├── name: string                    函数名
│   │   ├── inputs: [TensorRef]             输入张量引用
│   │   │   ├── global_id: uint32           全局张量 ID
│   │   │   └── desc: TensorDesc
│   │   │       ├── shape: [int32]          [1, 128, 768]
│   │   │       ├── mesh_shape: [int32]     [1, 1]
│   │   │       └── layout_v2: LayoutV2
│   │   │           ├── tensor_layout: TensorLayout  Tile/RowMajor
│   │   │           └── memory_config: MemoryConfig
│   │   │               ├── tensor_memory_layout: Interleaved/Sharded
│   │   │               ├── buffer_type: DRAM/L1
│   │   │               └── shard_spec: ShardSpec (if sharded)
│   │   │
│   │   ├── outputs: [TensorRef]            输出张量引用
│   │   ├── operations: [Operation]         操作序列
│   │   │   ├── Operation 0
│   │   │   │   ├── type: OpType (union)    ToDeviceOp
│   │   │   │   ├── debug_info: string      MLIR IR 片段
│   │   │   │   └── loc_info: string        源位置信息
│   │   │   ├── Operation 1
│   │   │   │   ├── type: MatmulOp
│   │   │   │   ├── in0: TensorRef
│   │   │   │   ├── in1: TensorRef
│   │   │   │   ├── out: TensorRef
│   │   │   │   └── core_grid: Dim2d
│   │   │   └── ...
│   │   │
│   │   ├── dylibs: [DynamicLib]           自定义内核动态库
│   │   │   ├── file_name: string          "kernel_abc123.so"
│   │   │   └── binary: [uint8]            .so 文件二进制
│   │   │
│   │   ├── mesh_shape: Dim2d              设备网格形状
│   │   └── private: bool                  是否为内部函数
│   │
│   └── Program 1: "const_eval_func"
│       └── ...
│
└── DebugInfo (调试信息)
    ├── mlir_module: string                 完整 MLIR IR
    └── golden_map: {uint32: GoldenTensor} 测试用参考数据
```

### 2.2 加载流程

```cpp
// 代码位置: /home/ubuntu/work/tt/tt-mlir/runtime/lib/runtime.cpp

namespace tt::runtime {

// 1. 从文件加载
Binary Binary::loadFromPath(const char *path) {
  // a. 读取文件到内存
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error("Failed to open file: " + std::string(path));
  }

  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> buffer(size);
  if (!file.read(buffer.data(), size)) {
    throw std::runtime_error("Failed to read file");
  }

  // b. 创建 Flatbuffer 对象
  Flatbuffer fb(buffer.data(), buffer.size());

  // c. 验证 Flatbuffer 完整性
  if (!fb.isValid()) {
    throw std::runtime_error("Invalid flatbuffer binary");
  }

  return Binary(std::move(fb));
}

// 2. 验证版本兼容性
bool Binary::isVersionCompatible() const {
  const auto* fbBinary = as<::tt::target::ttnn::Binary>();
  auto version = fbBinary->version();

  // 主版本必须匹配
  if (version->major() != TTMLIR_VERSION_MAJOR) {
    LOG_WARNING("Major version mismatch: binary {}, runtime {}",
                version->major(), TTMLIR_VERSION_MAJOR);
    return false;
  }

  // 次版本向后兼容
  if (version->minor() > TTMLIR_VERSION_MINOR) {
    LOG_WARNING("Binary compiled with newer compiler");
    return false;
  }

  return true;
}

// 3. 提取设备需求
DeviceRuntime Binary::getDeviceRuntime() const {
  // 根据 schema 类型判断
  if (as<::tt::target::ttnn::Binary>()) {
    return DeviceRuntime::TTNN;
  } else if (as<::tt::target::ttmetal::Binary>()) {
    return DeviceRuntime::TTMetal;
  }
  return DeviceRuntime::Unknown;
}

}  // namespace tt::runtime
```

### 2.3 Schema 验证

Flatbuffer 自动验证以下内容：

1. **结构完整性**：所有必需字段存在
2. **类型安全**：union 类型正确
3. **指针有效性**：所有偏移量在文件范围内
4. **字符串终止**：字符串正确以 null 结尾

**验证代码**：

```cpp
bool Flatbuffer::isValid() const {
  // Flatbuffer 内置验证器
  flatbuffers::Verifier verifier(
      reinterpret_cast<const uint8_t*>(data_),
      size_
  );

  // 根据类型验证
  if (as<::tt::target::ttnn::Binary>()) {
    return ::tt::target::ttnn::VerifyBinaryBuffer(verifier);
  } else if (as<::tt::target::ttmetal::Binary>()) {
    return ::tt::target::ttmetal::VerifyBinaryBuffer(verifier);
  }

  return false;
}
```

---

## 设备初始化与管理

### 3.1 设备检测与集群管理

**代码位置**：`/home/ubuntu/work/tt/tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/llrt/`

```cpp
// tt_cluster.cpp - 设备集群管理

namespace tt::tt_metal::llrt {

// 1. 检测系统中的设备
std::vector<chip_id_t> detectDevices() {
  std::vector<chip_id_t> deviceIds;

  // a. 通过 UMD 扫描 PCIe 总线
  auto pciDevices = tt::umd::PCIDevice::enumerate();

  for (const auto& pciDev : pciDevices) {
    chip_id_t chipId = pciDev.getChipId();
    deviceIds.push_back(chipId);
    LOG_INFO("Detected device {}: {}", chipId, pciDev.getDeviceName());
  }

  return deviceIds;
}

// 2. 创建集群描述
Cluster::Cluster(const std::vector<chip_id_t>& deviceIds) {
  for (auto chipId : deviceIds) {
    // 创建 Chip 对象（真实硬件或模拟器）
    auto chip = createChip(chipId);

    // 读取 SoC 描述符
    auto socDesc = chip->getSocDescriptor();

    // 初始化设备状态
    chips_[chipId] = {
      .chip = std::move(chip),
      .soc_desc = socDesc,
      .state = DeviceState::Uninitialized
    };
  }

  // 检测多芯片连接（以太网）
  detectInterChipConnections();
}

// 3. 检测芯片间连接
void Cluster::detectInterChipConnections() {
  for (auto& [chipId, chipInfo] : chips_) {
    auto& socDesc = chipInfo.soc_desc;

    // 遍历以太网核心
    for (const auto& ethCore : socDesc.ethernet_cores) {
      // 读取远端芯片 ID（通过以太网链路）
      auto remoteChipId = readRemoteChipId(chipId, ethCore);

      if (remoteChipId != INVALID_CHIP_ID) {
        // 记录连接
        connections_[chipId][ethCore] = remoteChipId;
        LOG_INFO("Chip {} eth core {} connected to chip {}",
                 chipId, ethCore, remoteChipId);
      }
    }
  }
}

}  // namespace tt::tt_metal::llrt
```

### 3.2 UMD 层初始化

**Unified Memory Driver (UMD)** 提供设备底层访问：

```cpp
// umd/device/chip/local_chip.cpp

class LocalChip : public Chip {
public:
  LocalChip(chip_id_t id) : chip_id_(id) {
    // 1. 打开 PCIe 设备
    pci_device_ = PCIDevice::open(id);
    if (!pci_device_) {
      throw std::runtime_error("Failed to open PCIe device");
    }

    // 2. 映射 BAR 空间（内存映射 I/O）
    bar0_ = pci_device_->mapBAR(0);  // 寄存器空间
    bar2_ = pci_device_->mapBAR(2);  // L1 内存访问
    bar4_ = pci_device_->mapBAR(4);  // DRAM 访问

    // 3. 初始化 ARC（芯片管理处理器）
    initARC();

    // 4. 去除 RISC-V 核心复位状态
    deassertRISCVResets();

    // 5. 配置 NoC（片上网络）
    configureNoC();

    // 6. 初始化 DRAM
    initDRAM();
  }

  // 写入 L1 内存
  void write_to_device(CoreCoord core, const void* src,
                       uint64_t l1_dest, uint32_t size) override {
    // 计算 NOC 地址
    uint64_t nocAddr = getNoCAddr(core, l1_dest);

    // 通过 BAR2 写入
    memcpy(bar2_ + nocAddr, src, size);

    // 可选：使用 DMA 加速大数据传输
    if (size > DMA_THRESHOLD) {
      pci_device_->dmaWrite(src, size, nocAddr);
    }
  }

private:
  chip_id_t chip_id_;
  std::unique_ptr<PCIDevice> pci_device_;
  void* bar0_;  // 寄存器映射
  void* bar2_;  // L1 访问映射
  void* bar4_;  // DRAM 访问映射
};
```

### 3.3 内存分配器初始化

```cpp
// tt_metal/impl/allocator/allocator.cpp

AllocatorImpl::AllocatorImpl(const AllocatorConfig& config) {
  // 1. 创建 DRAM 分配器
  if (config.num_dram_channels > 0) {
    dram_manager_ = std::make_unique<BankManager>(
        BufferType::DRAM,
        config.dram_bank_offsets,    // 每个 bank 的偏移
        config.dram_bank_size,        // 每个 bank 大小 (1 GB)
        config.dram_alignment,        // 对齐要求 (32 bytes)
        config.dram_unreserved_base,  // 可分配起始地址
        config.disable_interleaved
    );
  }

  // 2. 创建 L1 分配器（每个 Worker 核心）
  std::vector<int64_t> l1_bank_offsets;
  for (const auto& coreRange : config.worker_grid) {
    for (const auto& core : coreRange) {
      l1_bank_offsets.push_back(getL1BankOffset(core));
    }
  }

  l1_manager_ = std::make_unique<BankManager>(
      BufferType::L1,
      l1_bank_offsets,
      config.worker_l1_size,        // 每个核心 L1 大小 (1 MB)
      config.l1_alignment,          // 对齐要求 (16 bytes)
      config.l1_unreserved_base,    // 可分配起始地址
      config.disable_interleaved
  );

  // 3. 创建 L1 Small 分配器（用于小对象）
  if (config.l1_small_size > 0) {
    l1_small_manager_ = std::make_unique<BankManager>(
        BufferType::L1Small,
        l1_bank_offsets,
        config.l1_small_size,
        config.l1_alignment,
        config.worker_l1_size - config.l1_small_size,
        config.disable_interleaved
    );
  }

  // 4. 创建 Trace Buffer 分配器
  if (config.trace_region_size > 0) {
    trace_buffer_manager_ = std::make_unique<BankManager>(
        BufferType::Trace,
        config.dram_bank_offsets,
        config.trace_region_size,
        config.dram_alignment,
        config.dram_bank_size - config.trace_region_size,  // 从顶部分配
        false  // Trace 不使用交错
    );
  }
}
```

### 3.4 打开 MeshDevice

```cpp
// runtime/lib/ttnn/runtime.cpp

namespace tt::runtime::ttnn {

Device openMeshDevice(const MeshDeviceOptions& options) {
  // 1. 验证设备 ID
  for (auto deviceId : options.device_ids) {
    if (!isDeviceAvailable(deviceId)) {
      throw std::runtime_error("Device " + std::to_string(deviceId) +
                               " not available");
    }
  }

  // 2. 创建设备网格
  auto meshShape = ::ttnn::MeshShape{
      options.mesh_shape[0],  // rows
      options.mesh_shape[1]   // cols
  };

  // 3. 配置 L1 Small 和 Trace 大小
  size_t l1_small_size = options.l1_small_size.value_or(
      DEFAULT_L1_SMALL_SIZE  // 1 KB
  );
  size_t trace_region_size = options.trace_region_size.value_or(
      DEFAULT_TRACE_REGION_SIZE  // 256 MB
  );

  // 4. 创建 MeshDevice
  auto meshDevice = std::make_shared<::ttnn::MeshDevice>(
      meshShape,
      std::vector<chip_id_t>(options.device_ids.begin(),
                             options.device_ids.end()),
      options.num_hw_cqs,           // 硬件命令队列数 (default: 2)
      l1_small_size,
      trace_region_size,
      options.dispatch_core_type.value_or(
          ::ttnn::DispatchCoreType::Worker
      )
  );

  // 5. 启用程序缓存（可选）
  if (options.enable_program_cache) {
    ::ttnn::ProgramCache::instance().enable();
  }

  // 6. 初始化 Prefetcher/Dispatcher 固件
  for (auto deviceId : options.device_ids) {
    auto device = meshDevice->get_device(deviceId);
    initializeDispatchKernels(device);
  }

  // 7. 返回设备句柄
  return Device(meshDevice);
}

// 初始化 Dispatch 内核
void initializeDispatchKernels(::ttnn::Device* device) {
  // 编译 Prefetcher 和 Dispatcher 固件
  auto prefetcherBinary = compilePrefetcherKernel(device);
  auto dispatcherBinary = compileDispatcherKernel(device);

  // 选择专用核心
  auto prefetcherCore = selectPrefetcherCore(device);
  auto dispatcherCore = selectDispatcherCore(device);

  // 加载固件到核心 L1
  device->write_to_device(prefetcherCore, prefetcherBinary.data(),
                          0, prefetcherBinary.size());
  device->write_to_device(dispatcherCore, dispatcherBinary.data(),
                          0, dispatcherBinary.size());

  // 启动 Prefetcher/Dispatcher（去除复位）
  device->deassert_risc_reset(prefetcherCore);
  device->deassert_risc_reset(dispatcherCore);

  LOG_INFO("Dispatch kernels initialized on device {}", device->id());
}

}  // namespace tt::runtime::ttnn
```

---

## 张量创建与内存分配

### 4.1 从 TensorDesc 创建张量

```cpp
// runtime/lib/ttnn/utils.cpp

::ttnn::Tensor createDeviceTensor(
    const ::tt::target::ttnn::TensorDesc* desc,
    Device device
) {
  // 1. 提取形状
  std::vector<uint32_t> shape;
  for (auto dim : *desc->shape()) {
    shape.push_back(static_cast<uint32_t>(dim));
  }

  // 2. 转换 Layout
  auto layoutV2 = desc->layout_v2();
  ::ttnn::Layout layout = toTTNNLayout(layoutV2->tensor_layout());

  // 3. 转换 MemoryConfig
  auto memoryConfig = toTTNNMemoryConfig(layoutV2->memory_config());

  // 4. 推断数据类型（从 LayoutV2 或使用默认值）
  ::ttnn::DataType dtype = ::ttnn::DataType::BFLOAT16;  // 默认
  if (auto dtypeAttr = layoutV2->dtype()) {
    dtype = toTTNNDataType(dtypeAttr);
  }

  // 5. 创建设备张量（分配内存）
  auto ttnnDevice = getMeshDevice(device);
  auto tensor = ::ttnn::create_device_tensor(
      ::ttnn::Shape(shape),
      dtype,
      layout,
      *ttnnDevice,
      memoryConfig
  );

  return tensor;
}

// 转换 MemoryConfig
::ttnn::MemoryConfig toTTNNMemoryConfig(
    const ::tt::target::ttnn::MemoryConfig* fbMemConfig
) {
  // 1. Buffer Type
  ::ttnn::BufferType bufferType = ::ttnn::BufferType::DRAM;
  switch (fbMemConfig->buffer_type()) {
    case ::tt::target::BufferType::DRAM:
      bufferType = ::ttnn::BufferType::DRAM;
      break;
    case ::tt::target::BufferType::L1:
      bufferType = ::ttnn::BufferType::L1;
      break;
    case ::tt::target::BufferType::SystemMemory:
      bufferType = ::ttnn::BufferType::SYSTEM_MEMORY;
      break;
  }

  // 2. Tensor Memory Layout
  ::ttnn::TensorMemoryLayout memLayout =
      ::ttnn::TensorMemoryLayout::INTERLEAVED;
  switch (fbMemConfig->tensor_memory_layout()) {
    case ::tt::target::ttnn::TensorMemoryLayout::Interleaved:
      memLayout = ::ttnn::TensorMemoryLayout::INTERLEAVED;
      break;
    case ::tt::target::ttnn::TensorMemoryLayout::HeightSharded:
      memLayout = ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED;
      break;
    case ::tt::target::ttnn::TensorMemoryLayout::WidthSharded:
      memLayout = ::ttnn::TensorMemoryLayout::WIDTH_SHARDED;
      break;
    case ::tt::target::ttnn::TensorMemoryLayout::BlockSharded:
      memLayout = ::ttnn::TensorMemoryLayout::BLOCK_SHARDED;
      break;
  }

  // 3. Shard Spec (if sharded)
  std::optional<::ttnn::ShardSpec> shardSpec = std::nullopt;
  if (fbMemConfig->shard_spec()) {
    shardSpec = toTTNNShardSpec(fbMemConfig->shard_spec());
  }

  // 4. 创建 MemoryConfig
  return ::ttnn::MemoryConfig{
    .memory_layout = memLayout,
    .buffer_type = bufferType,
    .shard_spec = shardSpec
  };
}
```

### 4.2 内存分配流程

```cpp
// ttnn/cpp/ttnn/tensor/tensor.cpp

::ttnn::Tensor create_device_tensor(
    const Shape& shape,
    DataType dtype,
    Layout layout,
    Device& device,
    const MemoryConfig& memory_config
) {
  // 1. 计算张量大小
  size_t numElements = shape.volume();
  size_t elementSize = getDataTypeSize(dtype);
  size_t totalSizeBytes = numElements * elementSize;

  // 2. 计算页大小
  size_t pageSize;
  if (layout == Layout::TILE) {
    // Tile 格式：32x32 元素
    pageSize = TILE_HEIGHT * TILE_WIDTH * elementSize;
  } else {
    // Row-major：使用行大小
    pageSize = shape[-1] * elementSize;
  }

  // 3. 对齐到页边界
  size_t numPages = (totalSizeBytes + pageSize - 1) / pageSize;
  totalSizeBytes = numPages * pageSize;

  // 4. 创建 Buffer 配置
  BufferConfig bufferConfig{
    .device = &device,
    .size = totalSizeBytes,
    .page_size = pageSize,
    .buffer_type = memory_config.buffer_type
  };

  // 5. 根据 Memory Layout 分配
  std::shared_ptr<Buffer> buffer;
  if (memory_config.memory_layout == TensorMemoryLayout::INTERLEAVED) {
    // 交错分配（默认）
    buffer = std::make_shared<Buffer>(bufferConfig);
  } else {
    // 分片分配
    ShardedBufferConfig shardedConfig{
      .device = &device,
      .size = totalSizeBytes,
      .page_size = pageSize,
      .buffer_type = memory_config.buffer_type,
      .shard_parameters = *memory_config.shard_spec
    };
    buffer = std::make_shared<Buffer>(shardedConfig);
  }

  // 6. 通过 Allocator 分配设备内存
  DeviceAddr deviceAddr = device.allocator()->allocate_buffer(buffer.get());

  // 7. 创建 Tensor 对象
  return Tensor(
      TensorStorage{buffer},
      shape,
      dtype,
      layout
  );
}
```

### 4.3 Layout 转换

**Row-Major ↔ Tile 转换**在以下场景发生：

```cpp
// ttnn/cpp/ttnn/operations/core/to_layout/to_layout.cpp

::ttnn::Tensor to_layout(
    const Tensor& input,
    Layout target_layout,
    const std::optional<MemoryConfig>& memory_config
) {
  // 1. 如果已经是目标 layout，直接返回
  if (input.get_layout() == target_layout) {
    return input;
  }

  // 2. Row-Major → Tile
  if (input.get_layout() == Layout::ROW_MAJOR &&
      target_layout == Layout::TILE) {
    return tilize(input, memory_config);
  }

  // 3. Tile → Row-Major
  if (input.get_layout() == Layout::TILE &&
      target_layout == Layout::ROW_MAJOR) {
    return untilize(input, memory_config);
  }

  throw std::runtime_error("Unsupported layout conversion");
}

// Tilize: Row-Major → Tile
Tensor tilize(const Tensor& input,
              const std::optional<MemoryConfig>& memory_config) {
  // a. 创建输出张量（Tile 布局）
  auto output = create_device_tensor(
      input.get_shape(),
      input.get_dtype(),
      Layout::TILE,
      input.device(),
      memory_config.value_or(input.memory_config())
  );

  // b. 创建 tilize kernel
  Program program = CreateProgram();

  // Reader kernel: 从 DRAM 读取 row-major 数据
  auto reader_kernel = CreateKernel(
      program,
      "ttnn/cpp/ttnn/operations/core/to_layout/device/kernels/reader_tilize.cpp",
      workerCores,
      DataMovementConfig{.processor = DataMovementProcessor::RISCV_0}
  );

  // Compute kernel: 执行 tilize 转换
  auto compute_kernel = CreateKernel(
      program,
      "ttnn/cpp/ttnn/operations/core/to_layout/device/kernels/compute_tilize.cpp",
      workerCores,
      ComputeConfig{}
  );

  // Writer kernel: 写入 tile 格式数据
  auto writer_kernel = CreateKernel(
      program,
      "ttnn/cpp/ttnn/operations/core/to_layout/device/kernels/writer_tilize.cpp",
      workerCores,
      DataMovementConfig{.processor = DataMovementProcessor::RISCV_1}
  );

  // c. 设置运行时参数
  SetRuntimeArgs(program, reader_kernel, workerCores[0], {
    input.buffer()->address(),
    output.buffer()->address(),
    input.get_shape()[-2],  // height
    input.get_shape()[-1]   // width
  });

  // d. 执行转换
  EnqueueProgram(input.device().command_queue(), program, false);

  return output;
}
```

**Tilize Kernel 示例**（简化版）：

```cpp
// compute_tilize.cpp - 运行在 Tensix Core 的计算内核

#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/pack.h"

void MAIN {
  constexpr uint32_t cb_in = tt::CB::c_in0;   // Row-major input
  constexpr uint32_t cb_out = tt::CB::c_out0; // Tile output

  uint32_t num_tiles = get_arg_val<uint32_t>(0);

  for (uint32_t i = 0; i < num_tiles; ++i) {
    // 1. 等待输入 tile 准备好（reader kernel 已读取）
    cb_wait_front(cb_in, 1);

    // 2. 获取输出 buffer
    cb_reserve_back(cb_out, 1);

    // 3. TRISC0 自动 unpack (row-major → 寄存器)
    // 4. TRISC1 重排数据为 tile 格式
    tile_regs_acquire();
    copy_tile(cb_in, 0, 0);  // src, src_idx, dst_idx
    tile_regs_commit();

    // 5. TRISC2 自动 pack (寄存器 → tile 格式)

    // 6. 释放 buffers
    cb_pop_front(cb_in, 1);
    cb_push_back(cb_out, 1);
  }
}
```

---

## TTNN Runtime 执行流程

### 5.1 Submit 入口函数

```cpp
// runtime/lib/ttnn/runtime.cpp

namespace tt::runtime::ttnn {

std::vector<Tensor> submit(
    Device deviceHandle,
    Binary executableHandle,
    std::uint32_t programIndex,
    const std::vector<Tensor>& inputs
) {
  // 1. 获取 Flatbuffer Program
  const auto* fbBinary = executableHandle.as<::tt::target::ttnn::Binary>();
  if (!fbBinary) {
    throw std::runtime_error("Invalid binary type");
  }

  const auto* program = fbBinary->programs()->Get(programIndex);
  if (!program) {
    throw std::runtime_error("Invalid program index");
  }

  // 2. 验证输入数量
  if (inputs.size() != program->inputs()->size()) {
    throw std::runtime_error(
        "Input count mismatch: expected " +
        std::to_string(program->inputs()->size()) +
        ", got " + std::to_string(inputs.size())
    );
  }

  // 3. 创建张量映射表（global_id → Tensor）
  std::unordered_map<uint32_t, ::ttnn::Tensor> tensorMap;

  // 4. 填充输入张量
  for (size_t i = 0; i < inputs.size(); ++i) {
    uint32_t globalId = program->inputs()->Get(i)->global_id();
    tensorMap[globalId] = toTTNNTensor(inputs[i]);
  }

  // 5. 按顺序执行操作
  for (const auto* op : *program->operations()) {
    executeOperation(op, tensorMap, deviceHandle);
  }

  // 6. 收集输出张量
  std::vector<Tensor> outputs;
  for (const auto* outputRef : *program->outputs()) {
    uint32_t globalId = outputRef->global_id();
    outputs.push_back(toRuntimeTensor(tensorMap[globalId]));
  }

  return outputs;
}

}  // namespace tt::runtime::ttnn
```

### 5.2 操作调度器

```cpp
void executeOperation(
    const ::tt::target::ttnn::Operation* op,
    std::unordered_map<uint32_t, ::ttnn::Tensor>& tensorMap,
    Device device
) {
  // 根据操作类型分发
  switch (op->type_type()) {
    // === 数据移动操作 ===
    case ::tt::target::ttnn::OpType_ToDeviceOp:
      executeToDeviceOp(op->type_as_ToDeviceOp(), tensorMap, device);
      break;

    case ::tt::target::ttnn::OpType_FromDeviceOp:
      executeFromDeviceOp(op->type_as_FromDeviceOp(), tensorMap, device);
      break;

    case ::tt::target::ttnn::OpType_ToLayoutOp:
      executeToLayoutOp(op->type_as_ToLayoutOp(), tensorMap, device);
      break;

    case ::tt::target::ttnn::OpType_ToMemoryConfigOp:
      executeToMemoryConfigOp(op->type_as_ToMemoryConfigOp(), tensorMap, device);
      break;

    // === 元素级操作 ===
    case ::tt::target::ttnn::OpType_EltwiseBinaryOp:
      executeEltwiseBinaryOp(op->type_as_EltwiseBinaryOp(), tensorMap, device);
      break;

    case ::tt::target::ttnn::OpType_EltwiseUnaryOp:
      executeEltwiseUnaryOp(op->type_as_EltwiseUnaryOp(), tensorMap, device);
      break;

    // === 线性代数 ===
    case ::tt::target::ttnn::OpType_MatmulOp:
      executeMatmulOp(op->type_as_MatmulOp(), tensorMap, device);
      break;

    case ::tt::target::ttnn::OpType_LinearOp:
      executeLinearOp(op->type_as_LinearOp(), tensorMap, device);
      break;

    // === 卷积 ===
    case ::tt::target::ttnn::OpType_Conv2dOp:
      executeConv2dOp(op->type_as_Conv2dOp(), tensorMap, device);
      break;

    // === 归一化 ===
    case ::tt::target::ttnn::OpType_LayerNormOp:
      executeLayerNormOp(op->type_as_LayerNormOp(), tensorMap, device);
      break;

    case ::tt::target::ttnn::OpType_RMSNormOp:
      executeRMSNormOp(op->type_as_RMSNormOp(), tensorMap, device);
      break;

    // === Attention ===
    case ::tt::target::ttnn::OpType_ScaledDotProductAttentionOp:
      executeSdpaOp(op->type_as_ScaledDotProductAttentionOp(), tensorMap, device);
      break;

    // ... 100+ 种操作

    default:
      throw std::runtime_error(
          "Unsupported operation type: " +
          std::to_string(op->type_type())
      );
  }
}
```

### 5.3 操作执行示例

#### ToDeviceOp：将 Host 张量移到设备

```cpp
void executeToDeviceOp(
    const ::tt::target::ttnn::ToDeviceOp* op,
    std::unordered_map<uint32_t, ::ttnn::Tensor>& tensorMap,
    Device device
) {
  // 1. 获取输入张量（Host）
  auto inputTensor = tensorMap[op->in()->global_id()];

  // 2. 转换 MemoryConfig
  auto memoryConfig = toTTNNMemoryConfig(op->memory_config());

  // 3. 获取设备
  auto ttnnDevice = getMeshDevice(device);

  // 4. 调用 TTNN 库将张量移到设备
  auto outputTensor = ::ttnn::to_device(
      inputTensor,
      *ttnnDevice,
      memoryConfig
  );

  // 5. 存储输出
  tensorMap[op->out()->global_id()] = outputTensor;
}

// TTNN 库实现（ttnn/cpp/ttnn/operations/core/to_device/to_device.cpp）
::ttnn::Tensor to_device(
    const Tensor& host_tensor,
    Device& device,
    const MemoryConfig& memory_config
) {
  // a. 创建设备张量（分配内存）
  auto device_tensor = create_device_tensor(
      host_tensor.get_shape(),
      host_tensor.get_dtype(),
      host_tensor.get_layout(),
      device,
      memory_config
  );

  // b. 从 Host 复制数据到设备
  EnqueueWriteBuffer(
      device.command_queue(),
      device_tensor.buffer(),
      host_tensor.data(),
      /*blocking=*/false
  );

  return device_tensor;
}
```

#### MatmulOp：矩阵乘法

```cpp
void executeMatmulOp(
    const ::tt::target::ttnn::MatmulOp* op,
    std::unordered_map<uint32_t, ::ttnn::Tensor>& tensorMap,
    Device device
) {
  // 1. 获取输入张量
  auto inputA = tensorMap[op->in0()->global_id()];
  auto inputB = tensorMap[op->in1()->global_id()];

  // 2. 转换可选参数
  std::optional<::ttnn::MemoryConfig> memoryConfig = std::nullopt;
  if (op->memory_config()) {
    memoryConfig = toTTNNMemoryConfig(op->memory_config());
  }

  std::optional<::ttnn::DataType> dtype = std::nullopt;
  if (op->dtype()) {
    dtype = toTTNNDataType(op->dtype());
  }

  std::optional<::ttnn::CoreGrid> coreGrid = std::nullopt;
  if (op->core_grid()) {
    coreGrid = ::ttnn::CoreGrid{
      op->core_grid()->y(),
      op->core_grid()->x()
    };
  }

  // 3. 调用 TTNN matmul
  auto result = ::ttnn::matmul(
      inputA,
      inputB,
      /*transpose_a=*/false,
      /*transpose_b=*/false,
      memoryConfig,
      dtype,
      /*program_config=*/std::nullopt,
      /*activation=*/std::nullopt,
      /*compute_kernel_config=*/std::nullopt,
      coreGrid
  );

  // 4. 存储结果
  tensorMap[op->out()->global_id()] = result;
}

// TTNN matmul 实现（ttnn/cpp/ttnn/operations/matmul/matmul.cpp）
::ttnn::Tensor matmul(
    const Tensor& input_a,
    const Tensor& input_b,
    bool transpose_a,
    bool transpose_b,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DataType>& dtype,
    const std::optional<MatmulProgramConfig>& program_config,
    const std::optional<std::string>& activation,
    const std::optional<ComputeKernelConfig>& compute_kernel_config,
    const std::optional<CoreGrid>& core_grid
) {
  // a. 选择最优算法
  auto matmul_algorithm = selectMatmulAlgorithm(
      input_a.get_shape(),
      input_b.get_shape(),
      transpose_a,
      transpose_b,
      core_grid
  );

  // b. 根据算法调度
  switch (matmul_algorithm) {
    case MatmulAlgorithm::BMMS:  // Batched Matrix Multiply Sharded
      return bmm_sharded(input_a, input_b, ...);

    case MatmulAlgorithm::BMM_MULTI_CORE:
      return bmm_multi_core(input_a, input_b, ...);

    case MatmulAlgorithm::MATMUL_1D:
      return matmul_1d(input_a, input_b, ...);

    default:
      throw std::runtime_error("Unsupported matmul algorithm");
  }
}
```

#### Conv2dOp：二维卷积

```cpp
void executeConv2dOp(
    const ::tt::target::ttnn::Conv2dOp* op,
    std::unordered_map<uint32_t, ::ttnn::Tensor>& tensorMap,
    Device device
) {
  // 1. 获取输入
  auto input = tensorMap[op->input()->global_id()];
  auto weight = tensorMap[op->weight()->global_id()];

  std::optional<::ttnn::Tensor> bias = std::nullopt;
  if (op->bias()) {
    bias = tensorMap[op->bias()->global_id()];
  }

  // 2. 转换卷积参数
  auto conv_params = ::ttnn::Conv2dConfig{
    .kernel_size = {op->kernel_height(), op->kernel_width()},
    .stride = {op->stride_height(), op->stride_width()},
    .padding = {op->padding_height(), op->padding_width()},
    .dilation = {op->dilation_height(), op->dilation_width()},
    .groups = op->groups(),
    .memory_config = toTTNNMemoryConfig(op->memory_config())
  };

  // 3. 调用 TTNN conv2d
  auto result = ::ttnn::conv2d(
      input,
      weight,
      bias,
      conv_params,
      device
  );

  // 4. 存储结果
  tensorMap[op->out()->global_id()] = result;
}
```

### 5.4 张量生命周期管理

**TTNN Runtime 自动管理张量生命周期**：

```cpp
// 张量引用计数
class Tensor {
  std::shared_ptr<TensorStorage> storage_;  // 使用 shared_ptr 自动管理

public:
  // 析构时自动释放
  ~Tensor() {
    // storage_ 引用计数减 1
    // 当计数为 0 时，自动调用 Buffer::deallocate()
  }
};

// Buffer 释放
Buffer::~Buffer() {
  if (device_) {
    // 通过 Allocator 释放设备内存
    device_->allocator()->deallocate_buffer(this);
  }
}
```

**显式管理（可选）**：

```cpp
// 手动释放张量（立即释放内存）
::ttnn::deallocate(tensor);

// 批量释放所有未使用张量
device.deallocate_buffers();
```

---

## TTMetal Dispatch 架构

本节总结已有文档 `/home/ubuntu/work/tt/tt-software/tt-metal-analysis/07-dispatch-architecture-deep-dive.md` 的核心内容。

### 6.1 Prefetcher 和 Dispatcher 概述

**Prefetcher 和 Dispatcher 是两个独立的固件程序，运行在两个不同的 Tensix Core 上：**

```
┌─────────────────────────────────────────────────────┐
│  Host (CPU)                                         │
│                                                     │
│  ┌─────────────────────────────────────────┐       │
│  │  Hugepage (DMA-accessible memory)       │       │
│  │  - Issue Queue (Host → Device commands) │       │
│  │  - Completion Queue (Device → Host)     │       │
│  └────────────┬────────────────────────────┘       │
└───────────────┼─────────────────────────────────────┘
                │ PCIe DMA
                ▼
┌──────────────────────────────────────────────────────┐
│  Device (Tenstorrent ASIC)                          │
│                                                     │
│  ┌──────────────────────────────────────────────┐  │
│  │  Prefetcher Core (Tensix #1)                 │  │
│  │  - BRISC: 命令解析和控制                      │  │
│  │  - NCRISC: NoC 数据传输                       │  │
│  │  职责:                                        │  │
│  │  1. 从 Host Hugepage 读取命令（通过 PCIe）    │  │
│  │  2. 预取数据（Host/DRAM → L1）               │  │
│  │  3. 转发命令和数据到 Dispatcher               │  │
│  └────────────┬─────────────────────────────────┘  │
│               │ NoC
│               ▼
│  ┌──────────────────────────────────────────────┐  │
│  │  Dispatcher Core (Tensix #2)                 │  │
│  │  - NCRISC: 主处理器                          │  │
│  │  - BRISC: 辅助控制                            │  │
│  │  职责:                                        │  │
│  │  1. 接收 Prefetcher 的命令和数据              │  │
│  │  2. 解析 Dispatch 命令                        │  │
│  │  3. 写入数据到 Worker Cores L1                │  │
│  │  4. 发送 GO 信号启动 Worker Cores             │  │
│  │  5. 等待 Workers 完成                         │  │
│  └────────────┬─────────────────────────────────┘  │
│               │ NoC
│               ▼
│  ┌──────────────────────────────────────────────┐  │
│  │  Worker Cores (80x Tensix)                   │  │
│  │  - 执行用户内核（DataMovement + Compute）     │  │
│  │  - 每个核心 1 MB L1 内存                      │  │
│  └──────────────────────────────────────────────┘  │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 6.2 Hugepage 内存

**Hugepage 是在 Host 系统内存中分配的大页内存（2 MB 或 1 GB），用于高效 PCIe DMA 传输：**

**优势**：
- **物理地址连续**：DMA 引擎可以直接访问
- **减少 TLB 压力**：一个 TLB 条目映射更大范围
- **PCIe 传输效率高**：减少地址转换开销

**内存布局**：

```
Hugepage 内存布局 (per Command Queue):

┌─────────────────────────────────────────────────┐
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
│   │ (Prefetcher)               (Host)    │      │
├─────────────────────────────────────────────────┤
│  Completion Queue (Device → Host)               │
│  - Dispatcher 写入完成事件                       │
│  - Host 轮询读取                                 │
├─────────────────────────────────────────────────┤
│  Prefetch Queue (控制信息)                       │
│  - rd_ptr, wr_ptr                               │
└─────────────────────────────────────────────────┘
```

### 6.3 Prefetcher 工作流程

```cpp
// cq_prefetch.cpp - 运行在 Prefetcher Core 的固件

void kernel_main() {
  // 初始化
  noc_init();
  uint32_t pcie_read_ptr = pcie_base;

  while (true) {
    // 1. 从 Host Hugepage 读取命令头（通过 PCIe）
    CQPrefetchCmd cmd;
    noc_async_read(
        get_noc_addr(pcie_read_ptr),  // Host 地址
        (uint32_t)&cmd,                // 本地 L1
        sizeof(CQPrefetchCmd)
    );
    noc_async_read_barrier();
    pcie_read_ptr += sizeof(CQPrefetchCmd);

    // 2. 处理命令
    switch (cmd.base.cmd_id) {
      case CQ_PREFETCH_CMD_RELAY_LINEAR_H:
        // 从 Host 读取数据
        noc_async_read(
            get_noc_addr(pcie_base + cmd.relay_linear.addr),
            downstream_data_ptr,
            cmd.relay_linear.length
        );
        noc_async_read_barrier();
        relay_to_dispatcher(downstream_data_ptr, cmd.relay_linear.length);
        break;

      case CQ_PREFETCH_CMD_RELAY_LINEAR:
        // 从 DRAM 读取数据
        noc_async_read(
            get_noc_addr_xy(dram_noc_xy, cmd.relay_linear.addr),
            downstream_data_ptr,
            cmd.relay_linear.length
        );
        noc_async_read_barrier();
        relay_to_dispatcher(downstream_data_ptr, cmd.relay_linear.length);
        break;

      case CQ_PREFETCH_CMD_STALL:
        // 等待 Dispatcher 完成
        noc_semaphore_wait(my_downstream_cb_sem_id, 1);
        break;

      case CQ_PREFETCH_CMD_TERMINATE:
        return;  // 退出
    }
  }
}

// 转发数据到 Dispatcher
void relay_to_dispatcher(uint32_t local_addr, uint32_t size) {
  // 1. 等待 Dispatcher buffer 有空间
  noc_semaphore_wait(my_downstream_cb_sem_id, 1);

  // 2. 通过 NoC 写入到 Dispatcher
  noc_async_write(
      local_addr,
      get_noc_addr_xy(dispatcher_noc_xy, dispatcher_cb_addr),
      size
  );
  noc_async_write_barrier();

  // 3. 通知 Dispatcher 数据准备好
  noc_semaphore_inc(dispatcher_noc_xy, downstream_cb_sem_id, 1);
}
```

### 6.4 Dispatcher 工作流程

```cpp
// cq_dispatch.cpp - 运行在 Dispatcher Core 的固件

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
        // 写入 Worker Core L1
        uint32_t data_ptr = cmd_ptr;
        noc_async_write(
            data_ptr,
            get_noc_addr_xy(cmd.write_linear.noc_xy_addr,
                            cmd.write_linear.addr),
            cmd.write_linear.length
        );
        noc_async_write_barrier();
        cmd_ptr += cmd.write_linear.length;
        break;
      }

      case CQ_DISPATCH_CMD_WRITE_PACKED: {
        // 批量写入多个 Workers
        uint32_t data_ptr = cmd_ptr +
            cmd.write_packed.num_dests * sizeof(PackedSubCmd);

        for (uint32_t i = 0; i < cmd.write_packed.num_dests; i++) {
          PackedSubCmd subcmd;
          memcpy(&subcmd, (void*)cmd_ptr, sizeof(PackedSubCmd));
          cmd_ptr += sizeof(PackedSubCmd);

          noc_async_write(
              data_ptr,
              get_noc_addr_xy(subcmd.noc_xy, subcmd.addr),
              cmd.write_packed.size
          );
        }
        noc_async_write_barrier();
        cmd_ptr += cmd.write_packed.size;
        break;
      }

      case CQ_DISPATCH_CMD_GO: {
        // 发送 GO 信号启动 Workers
        for (uint32_t i = 0; i < cmd.go.num_workers; i++) {
          uint32_t go_signal = 1;
          noc_async_write(
              (uint32_t)&go_signal,
              get_noc_addr_xy(cmd.go.worker_cores[i], GO_SIGNAL_ADDR),
              sizeof(go_signal)
          );
        }
        noc_async_write_barrier();
        break;
      }

      case CQ_DISPATCH_CMD_WAIT: {
        // 等待 Workers 完成
        volatile uint32_t* sem_ptr = (volatile uint32_t*)cmd.wait.addr;
        while (*sem_ptr < cmd.wait.count) {
          // Spin wait
        }
        break;
      }

      case CQ_DISPATCH_CMD_TERMINATE:
        return;
    }

    // 4. 更新读取指针
    dispatch_cb_rd_ptr = (cmd_ptr - dispatch_cb_base) % dispatch_cb_size;

    // 5. 通知 Prefetcher 可以继续
    noc_semaphore_inc(prefetcher_noc_xy, upstream_dispatch_cb_sem_id, 1);
  }
}
```

### 6.5 Fast Dispatch 性能数据

根据 PCIExpressTile 文档实测性能：

| 操作 | 吞吐量 | 延迟 |
|------|--------|------|
| Host → Device (DMA) | 24 GB/s | ≥ 1342 ns |
| Device → Host (DMA) | 11 GB/s | ≥ 1052 ns |
| Device-initiated Read | 24.5 GB/s | ≥ 690 ns |
| Device-initiated Write | 26.4 GB/s | ≥ 260 ns |

**优化策略**：
- **Device-initiated 写入最快**：用于将结果写回 Host
- **Prefetcher 预取隐藏 PCIe 延迟**
- **NoC 内部传输更快**：Prefetcher → Dispatcher → Workers

---

## 内核执行机制

### 7.1 Tensix Core 的 5 个 RISC-V 核心

每个 Tensix Core 包含 5 个 RISC-V 处理器，分工协作：

```
┌─────────────────────────────────────────────────────────┐
│                    Tensix Core                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │  L1 Memory (1464 KB 共享)                       │   │
│  │  - Circular Buffers                             │   │
│  │  - Runtime Args                                 │   │
│  │  - 临时数据                                      │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  RISC-V B    │  │  RISC-V NC   │  │  NoC Router  │ │
│  │  (BRISC)     │  │  (NCRISC)    │  │              │ │
│  │              │  │              │  │  - NoC 0     │ │
│  │  控制逻辑    │  │  NoC 传输     │  │  - NoC 1     │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Tensix Coprocessor (AI 加速器)                  │  │
│  │                                                  │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐ │  │
│  │  │ RISC-V T0  │  │ RISC-V T1  │  │ RISC-V T2  │ │  │
│  │  │ (TRISC0)   │  │ (TRISC1)   │  │ (TRISC2)   │ │  │
│  │  │            │  │            │  │            │ │  │
│  │  │ UNPACK     │  │ MATH       │  │ PACK       │ │  │
│  │  └──────┬─────┘  └──────┬─────┘  └──────┬─────┘ │  │
│  │         ▼                ▼                ▼       │  │
│  │  ┌─────────┐      ┌─────────┐      ┌─────────┐  │  │
│  │  │Unpacker │      │ Matrix  │      │ Packer  │  │  │
│  │  │ (2x)    │      │ Unit    │      │ (4x)    │  │  │
│  │  └─────────┘      │ Vector  │      └─────────┘  │  │
│  │                   │ Unit    │                    │  │
│  │                   └─────────┘                    │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

| RISC-V | 职责 | 本地内存 |
|--------|------|---------|
| **BRISC** | 数据移动控制、协调其他核心 | 4KB RAM, 2KB I-cache |
| **NCRISC** | NoC 数据传输、DMA 控制 | 4KB RAM, 16KB IRAM, ½KB I-cache |
| **TRISC0** | UNPACK（从 L1 加载数据到寄存器） | 2KB RAM, 2KB I-cache |
| **TRISC1** | MATH（执行矩阵/向量运算） | 2KB RAM, ½KB I-cache |
| **TRISC2** | PACK（将结果写回 L1） | 2KB RAM, 2KB I-cache |

### 7.2 Worker Kernel 执行流程

**示例：矩阵加法 Kernel**

```cpp
// === Data Movement Kernel (reader_kernel.cpp) ===
// 运行在 BRISC + NCRISC

void kernel_main() {
  // 1. 等待 GO 信号（Dispatcher 发送）
  volatile uint32_t* go_signal = (volatile uint32_t*)GO_SIGNAL_ADDR;
  while (*go_signal == 0) {
    // Spin wait
  }

  // 2. 读取运行时参数
  uint32_t src_addr = get_arg_val<uint32_t>(0);  // DRAM 地址
  uint32_t num_tiles = get_arg_val<uint32_t>(1);

  // 3. 从 DRAM 读取数据到 L1 Circular Buffer
  constexpr uint32_t cb_id = tt::CB::c_in0;
  uint32_t l1_write_addr = get_write_ptr(cb_id);

  for (uint32_t i = 0; i < num_tiles; i++) {
    // NCRISC 通过 NoC 读取
    noc_async_read(
        src_addr,           // DRAM 地址
        l1_write_addr,      // L1 Circular Buffer
        TILE_SIZE
    );
    noc_async_read_barrier();

    // 通知 Compute kernel 数据准备好
    cb_push_back(cb_id, 1);

    src_addr += TILE_SIZE;
    l1_write_addr = get_write_ptr(cb_id);
  }
}
```

```cpp
// === Compute Kernel (add_kernel.cpp) ===
// 运行在 TRISC0 + TRISC1 + TRISC2

#include "compute_kernel_api/eltwise_binary.h"

void kernel_main() {
  constexpr uint32_t cb_in0 = tt::CB::c_in0;
  constexpr uint32_t cb_in1 = tt::CB::c_in1;
  constexpr uint32_t cb_out = tt::CB::c_out0;

  uint32_t num_tiles = get_arg_val<uint32_t>(0);

  // 初始化加法算子
  binary_op_init_common(cb_in0, cb_in1, cb_out);
  add_tiles_init();

  for (uint32_t i = 0; i < num_tiles; i++) {
    // 1. 等待输入数据准备好（Reader kernel 已读取）
    cb_wait_front(cb_in0, 1);
    cb_wait_front(cb_in1, 1);

    // 2. 获取输出 buffer
    cb_reserve_back(cb_out, 1);

    // 3. TRISC0 自动 Unpack（L1 → 寄存器）
    // 4. TRISC1 执行加法运算（调用 Tensix 协处理器）
    add_tiles(cb_in0, cb_in1, 0, 0, 0);  // 寄存器索引

    // 5. TRISC2 自动 Pack（寄存器 → L1）

    // 6. 释放 buffers
    cb_pop_front(cb_in0, 1);
    cb_pop_front(cb_in1, 1);
    cb_push_back(cb_out, 1);
  }
}
```

```cpp
// === Data Movement Kernel (writer_kernel.cpp) ===
// 运行在 BRISC + NCRISC

void kernel_main() {
  uint32_t dst_addr = get_arg_val<uint32_t>(0);  // DRAM 地址
  uint32_t num_tiles = get_arg_val<uint32_t>(1);

  constexpr uint32_t cb_id = tt::CB::c_out0;

  for (uint32_t i = 0; i < num_tiles; i++) {
    // 等待 Compute kernel 完成
    cb_wait_front(cb_id, 1);

    // 获取 L1 地址
    uint32_t l1_read_addr = get_read_ptr(cb_id);

    // NCRISC 通过 NoC 写入 DRAM
    noc_async_write(
        l1_read_addr,   // L1 地址
        dst_addr,       // DRAM 地址
        TILE_SIZE
    );
    noc_async_write_barrier();

    // 释放 buffer
    cb_pop_front(cb_id, 1);

    dst_addr += TILE_SIZE;
  }

  // 通知 Dispatcher 完成
  noc_semaphore_inc(dispatcher_noc_xy, worker_done_sem, 1);
}
```

### 7.3 Circular Buffer（环形缓冲区）

**Circular Buffer 是 L1 内存中的固定大小缓冲区，用于 DataMovement 和 Compute 核心间通信：**

```cpp
// 配置 Circular Buffer
CircularBufferConfig cb_config = CircularBufferConfig(
    2 * TILE_SIZE,  // 2 个 tiles（双缓冲）
    {{CB::c_in0, tt::DataFormat::Float16_b}}
)
.set_page_size(CB::c_in0, TILE_SIZE);

CreateCircularBuffer(program, coreRange, cb_config);
```

**双缓冲机制**：

```
时间 t0:
┌────────┬────────┐
│ Tile 0 │ Tile 1 │
└────────┴────────┘
  ▲        ▲
  │        │
Reader   Compute
读取     处理

时间 t1:
┌────────┬────────┐
│ Tile 2 │ Tile 1 │
└────────┴────────┘
  ▲        ▲
  │        │
Reader   Compute
读取     处理

Reader 和 Compute 并行工作，隐藏延迟
```

### 7.4 NoC（Network-on-Chip）通信

**NoC 是芯片内部高速互连网络，连接所有 Tiles（Tensix、DRAM、PCIe）：**

- **拓扑**：2D Torus（10x12 网格）
- **带宽**：256-bit 数据宽度
- **寻址**：通过 (x, y) 坐标
- **虚拟通道**：16 个，避免死锁

**NoC 操作示例**：

```cpp
// 读取远端核心的 L1 内存
uint64_t remote_addr = get_noc_addr(remote_core_x, remote_core_y, l1_addr);
noc_async_read(remote_addr, local_addr, size);
noc_async_read_barrier();  // 等待完成

// 写入远端核心的 L1 内存
noc_async_write(local_addr, remote_addr, size);
noc_async_write_barrier();

// Multicast 写入（一对多）
noc_async_write_multicast(
    local_addr,
    start_core_x, start_core_y,
    end_core_x, end_core_y,
    remote_l1_addr,
    size
);
```

---

## 完整端到端执行流程

### 8.1 完整流程图

```
┌──────────────────────────────────────────────────────────┐
│  Step 1: 用户代码（Python/C++）                          │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  import torch                                            │
│  import torch_ttnn                                       │
│                                                          │
│  model = MyModel()                                       │
│  model.eval()                                            │
│                                                          │
│  # 编译模型                                              │
│  compiled_model = torch.compile(                         │
│      model,                                              │
│      backend="ttnn"                                      │
│  )                                                       │
│                                                          │
│  # 执行推理                                              │
│  input_tensor = torch.randn(1, 3, 224, 224)             │
│  output = compiled_model(input_tensor)                   │
│                                                          │
└──────────────┬───────────────────────────────────────────┘
               │
┌──────────────▼───────────────────────────────────────────┐
│  Step 2: TT-MLIR 编译器（离线或 JIT）                    │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  PyTorch IR → StableHLO → TTIR → TTNN Dialect           │
│                                                          │
│  优化 Passes:                                            │
│  - 常量折叠                                              │
│  - 算子融合                                              │
│  - Layout 优化                                           │
│  - Sharding 策略                                         │
│                                                          │
│  生成 Flatbuffer:                                        │
│  - model.ttnn                                            │
│                                                          │
└──────────────┬───────────────────────────────────────────┘
               │
┌──────────────▼───────────────────────────────────────────┐
│  Step 3: Runtime 初始化                                  │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  // 1. 加载 binary                                       │
│  Binary binary = Binary::loadFromPath("model.ttnn");    │
│                                                          │
│  // 2. 验证版本                                          │
│  if (!binary.isVersionCompatible()) {                    │
│    throw std::runtime_error("Version mismatch");         │
│  }                                                       │
│                                                          │
│  // 3. 设置运行时                                        │
│  tt::runtime::setCompatibleDeviceRuntime(binary);        │
│  // → 自动选择 TTNN Runtime                              │
│                                                          │
│  // 4. 打开设备                                          │
│  Device device = tt::runtime::openMeshDevice({           │
│    .device_ids = {0},                                    │
│    .mesh_shape = {1, 1},                                 │
│    .num_hw_cqs = 2,                                      │
│    .enable_program_cache = true                          │
│  });                                                     │
│                                                          │
│  // 内部：                                               │
│  // - 检测设备                                           │
│  // - 初始化 UMD                                         │
│  // - 创建 Allocator                                     │
│  // - 加载 Prefetcher/Dispatcher 固件                    │
│  // - 分配 Hugepage                                      │
│                                                          │
└──────────────┬───────────────────────────────────────────┘
               │
┌──────────────▼───────────────────────────────────────────┐
│  Step 4: 创建输入张量                                    │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  // 1. 从 NumPy/Torch 数据创建 Host 张量                 │
│  std::vector<float> inputData = ...;                     │
│  Tensor hostInput = tt::runtime::createOwnedHostTensor(  │
│      inputData.data(),                                   │
│      {1, 3, 224, 224},  // shape                         │
│      {3*224*224, 224*224, 224, 1},  // stride            │
│      sizeof(float),     // itemsize                      │
│      DataType::Float32                                   │
│  );                                                      │
│                                                          │
└──────────────┬───────────────────────────────────────────┘
               │
┌──────────────▼───────────────────────────────────────────┐
│  Step 5: 提交执行                                        │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  std::vector<Tensor> outputs = tt::runtime::submit(      │
│      device,                                             │
│      binary,                                             │
│      0,  // program index (通常为 "forward")             │
│      {hostInput}                                         │
│  );                                                      │
│                                                          │
└──────────────┬───────────────────────────────────────────┘
               │
┌──────────────▼───────────────────────────────────────────┐
│  Step 6: TTNN Runtime 执行（内部）                       │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  // 1. 解析 Flatbuffer Program                           │
│  const auto* program = binary.getProgram(0);             │
│                                                          │
│  // 2. 创建张量映射表                                    │
│  std::unordered_map<uint32_t, ttnn::Tensor> tensorMap;  │
│                                                          │
│  // 3. 按顺序执行操作                                    │
│  for (const auto* op : program->operations()) {          │
│    switch (op->type_type()) {                            │
│                                                          │
│      case OpType_ToDeviceOp:                             │
│        // a. 从 Flatbuffer 读取 MemoryConfig             │
│        // b. 在设备上分配内存（通过 Allocator）          │
│        // c. 通过 CommandQueue 写入数据                  │
│        outputTensor = ttnn::to_device(                   │
│            inputTensor, device, memoryConfig);           │
│        tensorMap[op->out()->global_id()] = outputTensor; │
│        break;                                            │
│                                                          │
│      case OpType_ToLayoutOp:                             │
│        // Layout 转换（Row-Major ↔ Tile）                │
│        // - 调度 tilize/untilize kernel                  │
│        outputTensor = ttnn::to_layout(                   │
│            inputTensor, Layout::TILE);                   │
│        tensorMap[...] = outputTensor;                    │
│        break;                                            │
│                                                          │
│      case OpType_MatmulOp:                               │
│        // 矩阵乘法                                        │
│        // a. 选择算法（BMMS, BMM_MULTI_CORE, ...)        │
│        // b. 编译 kernels（如果未缓存）                  │
│        // c. 通过 CommandQueue 调度执行                  │
│        outputTensor = ttnn::matmul(                      │
│            inputA, inputB, ...);                         │
│        tensorMap[...] = outputTensor;                    │
│        break;                                            │
│                                                          │
│      case OpType_FromDeviceOp:                           │
│        // 从设备读取结果到 Host                          │
│        hostTensor = ttnn::from_device(deviceTensor);     │
│        tensorMap[...] = hostTensor;                      │
│        break;                                            │
│    }                                                     │
│  }                                                       │
│                                                          │
│  // 4. 收集输出                                          │
│  return outputs;                                         │
│                                                          │
└──────────────┬───────────────────────────────────────────┘
               │
┌──────────────▼───────────────────────────────────────────┐
│  Step 7: TTMetal 层执行（每个操作内部）                  │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  // 以 ttnn::matmul 为例                                 │
│                                                          │
│  // 1. 选择最优算法                                      │
│  auto algorithm = selectMatmulAlgorithm(shapeA, shapeB); │
│  // → BMM_MULTI_CORE（多核批量矩阵乘法）                 │
│                                                          │
│  // 2. 编译 kernels（如果未缓存）                        │
│  Program program = CreateProgram();                      │
│                                                          │
│  // a. Reader kernel（从 DRAM 读取输入）                 │
│  auto readerKernel = CreateKernel(                       │
│      program,                                            │
│      "matmul_reader.cpp",                                │
│      workerCores,                                        │
│      DataMovementConfig{...}                             │
│  );                                                      │
│                                                          │
│  // b. Compute kernel（执行矩阵乘法）                    │
│  auto computeKernel = CreateKernel(                      │
│      program,                                            │
│      "matmul_compute.cpp",                               │
│      workerCores,                                        │
│      ComputeConfig{...}                                  │
│  );                                                      │
│                                                          │
│  // c. Writer kernel（写回结果）                         │
│  auto writerKernel = CreateKernel(                       │
│      program,                                            │
│      "matmul_writer.cpp",                                │
│      workerCores,                                        │
│      DataMovementConfig{...}                             │
│  );                                                      │
│                                                          │
│  // 3. 配置 Circular Buffers                             │
│  CreateCircularBuffer(program, workerCores, cb_config);  │
│                                                          │
│  // 4. 设置运行时参数                                    │
│  for (auto core : workerCores) {                         │
│    SetRuntimeArgs(program, readerKernel, core, {        │
│      input_addr, weight_addr, num_tiles, ...            │
│    });                                                   │
│  }                                                       │
│                                                          │
│  // 5. 提交到 CommandQueue                               │
│  EnqueueProgram(device.command_queue(), program, false); │
│                                                          │
└──────────────┬───────────────────────────────────────────┘
               │
┌──────────────▼───────────────────────────────────────────┐
│  Step 8: Command Queue & Dispatch                        │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  // HWCommandQueue 内部                                  │
│                                                          │
│  // 1. Host 构建命令                                     │
│  HugepageDeviceCommand cmd(manager.issue_queue_reserve());│
│                                                          │
│  // a. 等待前一 program 完成                             │
│  cmd.add_dispatch_wait(...);                             │
│                                                          │
│  // b. 加载 kernel 二进制（从 Host 或 Program Cache）    │
│  cmd.add_prefetch_relay_linear_h(                        │
│      kernel_binary_addr, kernel_size);                   │
│  cmd.add_dispatch_write_linear(                          │
│      worker_noc_xy, l1_addr, kernel_binary, kernel_size);│
│                                                          │
│  // c. 写入运行时参数                                    │
│  cmd.add_dispatch_write_packed(                          │
│      workerCores, runtime_args_addr, runtime_args);      │
│                                                          │
│  // d. 发送 GO 信号                                      │
│  cmd.add_dispatch_go(workerCores);                       │
│                                                          │
│  // e. 等待 Workers 完成                                 │
│  cmd.add_dispatch_wait(worker_done_sem, num_workers);    │
│                                                          │
│  // 2. 提交到 Hugepage                                   │
│  manager.issue_queue_push_back(cmd.size_bytes());        │
│                                                          │
│  // 3. 通知 Prefetcher                                   │
│  manager.fetch_queue_write(cmd.size_bytes());            │
│                                                          │
└──────────────┬───────────────────────────────────────────┘
               │
┌──────────────▼───────────────────────────────────────────┐
│  Step 9: Prefetcher 读取命令                             │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  // 运行在 Prefetcher Core (Tensix)                      │
│                                                          │
│  while (true) {                                          │
│    // 1. 从 Hugepage 读取命令头（通过 PCIe DMA）         │
│    noc_async_read(pcie_addr, local_buf, cmd_size);       │
│                                                          │
│    // 2. 处理 Prefetch 命令                              │
│    switch (cmd.cmd_id) {                                 │
│      case RELAY_LINEAR_H:                                │
│        // 从 Host 读取 kernel 二进制                     │
│        noc_async_read(pcie_addr, local_buf, size);       │
│        relay_to_dispatcher(local_buf, size);             │
│        break;                                            │
│    }                                                     │
│                                                          │
│    // 3. 通过 NoC 转发到 Dispatcher                      │
│    noc_async_write(local_buf, dispatcher_addr, size);    │
│    noc_semaphore_inc(dispatcher_xy, page_ready_sem, 1);  │
│  }                                                       │
│                                                          │
└──────────────┬───────────────────────────────────────────┘
               │
┌──────────────▼───────────────────────────────────────────┐
│  Step 10: Dispatcher 分发命令                            │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  // 运行在 Dispatcher Core (Tensix)                      │
│                                                          │
│  while (true) {                                          │
│    // 1. 等待 Prefetcher 数据                            │
│    noc_semaphore_wait(page_ready_sem, 1);                │
│                                                          │
│    // 2. 解析 Dispatch 命令                              │
│    switch (cmd.cmd_id) {                                 │
│      case WRITE_LINEAR:                                  │
│        // 写入 kernel 二进制到 Worker L1                 │
│        noc_async_write(                                  │
│            local_data,                                   │
│            get_noc_addr(worker_xy, l1_addr),             │
│            kernel_size                                   │
│        );                                                │
│        break;                                            │
│                                                          │
│      case WRITE_PACKED:                                  │
│        // 批量写入运行时参数到多个 Workers               │
│        for (core : workerCores) {                        │
│          noc_async_write(runtime_args, core, ...);       │
│        }                                                 │
│        break;                                            │
│                                                          │
│      case GO:                                            │
│        // 发送 GO 信号启动 Workers                       │
│        for (core : workerCores) {                        │
│          noc_async_write(&go_signal, core, GO_ADDR, 4);  │
│        }                                                 │
│        break;                                            │
│                                                          │
│      case WAIT:                                          │
│        // 等待 Workers 完成                              │
│        while (*worker_done_sem < num_workers) {}         │
│        break;                                            │
│    }                                                     │
│                                                          │
│    // 3. 通知 Prefetcher 继续                            │
│    noc_semaphore_inc(prefetcher_xy, page_done_sem, 1);   │
│  }                                                       │
│                                                          │
└──────────────┬───────────────────────────────────────────┘
               │
┌──────────────▼───────────────────────────────────────────┐
│  Step 11: Worker Cores 执行 Kernel                       │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  // 每个 Worker Core (Tensix)                            │
│                                                          │
│  // === BRISC ===                                        │
│  void brisc_main() {                                     │
│    // 1. 等待 GO 信号                                    │
│    while (*go_signal_addr == 0) {}                       │
│                                                          │
│    // 2. 启动其他 RISC-V 核心                            │
│    launch_ncrisc();                                      │
│    launch_trisc0();                                      │
│    launch_trisc1();                                      │
│    launch_trisc2();                                      │
│                                                          │
│    // 3. 执行 Reader kernel 逻辑                         │
│    for (tile : tiles) {                                  │
│      // 从 DRAM 读取到 L1 CB                             │
│      noc_async_read(dram_addr, cb_addr, tile_size);      │
│      cb_push_back(cb_in0, 1);                            │
│    }                                                     │
│  }                                                       │
│                                                          │
│  // === NCRISC ===                                       │
│  void ncrisc_main() {                                    │
│    // 执行实际的 NoC 传输                                │
│    // （BRISC 调用的 noc_async_read 由 NCRISC 执行）     │
│  }                                                       │
│                                                          │
│  // === TRISC0 (UNPACK) ===                              │
│  void trisc0_main() {                                    │
│    for (tile : tiles) {                                  │
│      // 从 L1 CB 加载到 Tensix 寄存器                    │
│      unpack_tiles(cb_in0, 1);                            │
│    }                                                     │
│  }                                                       │
│                                                          │
│  // === TRISC1 (MATH) ===                                │
│  void trisc1_main() {                                    │
│    for (tile : tiles) {                                  │
│      // 执行矩阵乘法                                     │
│      matmul_tiles(in0, in1, out);                        │
│    }                                                     │
│  }                                                       │
│                                                          │
│  // === TRISC2 (PACK) ===                                │
│  void trisc2_main() {                                    │
│    for (tile : tiles) {                                  │
│      // 将结果打包回 L1 CB                               │
│      pack_tiles(cb_out, 1);                              │
│    }                                                     │
│  }                                                       │
│                                                          │
│  // BRISC: Writer kernel 逻辑                            │
│  for (tile : tiles) {                                    │
│    cb_wait_front(cb_out, 1);                             │
│    noc_async_write(cb_addr, dram_addr, tile_size);       │
│    cb_pop_front(cb_out, 1);                              │
│  }                                                       │
│                                                          │
│  // 通知 Dispatcher 完成                                 │
│  noc_semaphore_inc(dispatcher_xy, worker_done_sem, 1);   │
│                                                          │
└──────────────┬───────────────────────────────────────────┘
               │
┌──────────────▼───────────────────────────────────────────┐
│  Step 12: 结果返回                                       │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  // 1. Dispatcher 检测到所有 Workers 完成                │
│  noc_semaphore_wait(worker_done_sem, num_workers);       │
│                                                          │
│  // 2. 写入完成事件到 Completion Queue（Host Hugepage）  │
│  noc_async_write(                                        │
│      event_data,                                         │
│      pcie_tile_addr,                                     │
│      completion_queue_addr,                              │
│      event_size                                          │
│  );                                                      │
│                                                          │
│  // 3. Host Completion Queue 线程轮询                    │
│  while (true) {                                          │
│    event = poll_completion_queue();                      │
│    if (event) {                                          │
│      notify_waiting_threads(event);                      │
│      break;                                              │
│    }                                                     │
│  }                                                       │
│                                                          │
│  // 4. TTNN Runtime submit() 返回输出张量                │
│  return outputs;                                         │
│                                                          │
└──────────────┬───────────────────────────────────────────┘
               │
┌──────────────▼───────────────────────────────────────────┐
│  Step 13: 用户获取结果                                   │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  // Python 代码                                          │
│  output = compiled_model(input_tensor)                   │
│  # output 是 Torch Tensor，可以直接使用                  │
│                                                          │
│  print(output.shape)                                     │
│  print(output.mean())                                    │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## 性能优化机制

### 9.1 Program Cache（程序缓存）

**目的**：避免重复编译相同的 Program。

```cpp
// ttnn/cpp/ttnn/program_cache.cpp

class ProgramCache {
public:
  static ProgramCache& instance();

  void enable();
  void disable();

  // 查找缓存
  std::optional<ProgramWithCallbacks> find(const ProgramCacheKey& key);

  // 插入缓存
  void insert(const ProgramCacheKey& key, ProgramWithCallbacks program);

private:
  std::unordered_map<ProgramCacheKey, ProgramWithCallbacks> cache_;
  bool enabled_ = false;
};

// 使用示例（在 ttnn::matmul 内部）
auto cache_key = createProgramCacheKey(
    input_a.get_shape(),
    input_b.get_shape(),
    program_config
);

if (auto cached = ProgramCache::instance().find(cache_key)) {
  // 使用缓存的 program（跳过编译）
  return executeCachedProgram(*cached, input_a, input_b);
} else {
  // 首次执行：编译并缓存
  auto program = compileMatmulProgram(...);
  ProgramCache::instance().insert(cache_key, program);
  return executeProgram(program, input_a, input_b);
}
```

**性能提升**：
- 首次执行：编译 + 执行（慢）
- 后续执行：直接执行（快 10-100x）

### 9.2 Trace Mode（追踪模式）

**目的**：录制命令序列，重放时无需重新构建。

```cpp
// 录制 trace
uint32_t trace_id = BeginTraceCapture(device, cq_id);

// 执行一次（录制命令）
EnqueueProgram(cq, program_1);
EnqueueProgram(cq, program_2);
EnqueueProgram(cq, program_3);

// 结束录制
EndTraceCapture(device, cq_id, trace_id);

// 后续执行：直接重放（极快）
EnqueueTrace(cq, trace_id);
```

**Trace 存储位置**：
- DRAM 顶部的 Trace Region
- 避免 PCIe 传输，直接从 DRAM 读取

**性能提升**：
- 首次执行：正常速度
- 重放：减少 Host CPU 开销，提升 10x+

### 9.3 Async Execution（异步执行）

**默认行为**：CommandQueue 异步提交，不阻塞 Host。

```cpp
// 异步提交（默认）
EnqueueProgram(cq, program, /*blocking=*/false);
// 立即返回，设备在后台执行

// 后续同步（可选）
Finish(cq);  // 等待所有命令完成
```

**流水线执行**：

```python
# Python 示例
for batch in dataloader:
  # Host 准备下一批数据（与设备执行并行）
  inputs = preprocess(batch)

  # 提交到设备（异步）
  outputs = model(inputs)  # 不阻塞

  # 后处理（如果需要同步）
  results = postprocess(outputs)
```

### 9.4 Multi-Device Coordination（多设备协调）

**MeshDevice 自动管理多设备执行**：

```cpp
// 打开 2x2 设备网格
Device device = openMeshDevice({
  .device_ids = {0, 1, 2, 3},
  .mesh_shape = {2, 2},
  ...
});

// 单个 submit 调用自动分发到多设备
auto outputs = submit(device, binary, 0, {inputs});
```

**内部机制**：
- **数据分片**：根据 ShardSpec 自动分割输入
- **All-Reduce**：通过以太网核心同步梯度
- **负载均衡**：均匀分布计算到所有设备

---

## 调试与监控

### 10.1 环境变量

```bash
# === 日志级别 ===
export TT_METAL_LOGGER_LEVEL=DEBUG      # DEBUG, INFO, WARNING, ERROR
export TT_METAL_LOGGER_TYPES=Dispatch   # 过滤日志类型

# === Dispatch 调试 ===
export TT_METAL_LOGGER_TYPES=Dispatch,Device
# 输出：
# - Prefetcher/Dispatcher 命令
# - Command Queue 状态
# - Kernel 加载信息

# === 内存调试 ===
export TT_METAL_LOGGER_TYPES=Allocator
# 输出：
# - Buffer 分配/释放
# - 内存使用统计
# - 碎片化信息

export TT_METAL_DETECT_MEMORY_LEAK=1
# 程序结束时检测未释放的 buffers

# === 性能分析 ===
export TT_METAL_DEVICE_PROFILER=1
# 启用设备性能分析
# 输出每个操作的耗时

export TRACY_NO_INVARIANT_CHECK=1
# 启用 Tracy Profiler 集成

# === Kernel 调试 ===
export TT_METAL_DPRINT_ENABLE=1
# 启用 kernel 内的 DPRINT 输出

export TT_METAL_WATCHER=1
# 启用 Watcher（监控设备状态）
```

### 10.2 性能分析工具

**Tracy Profiler**：

```bash
# 1. 编译时启用 Tracy
export ENABLE_TRACY=1
cmake -B build && cmake --build build

# 2. 运行程序
./my_app

# 3. 启动 Tracy GUI
tracy

# Tracy 可视化：
# - Host CPU 时间线
# - Command Queue 提交
# - Kernel 执行时间
# - NoC 流量
```

**Device Profiler**：

```cpp
// 启用性能分析
::ttnn::DeviceProfiler::instance().enable();

// 执行操作
auto output = ::ttnn::matmul(input_a, input_b);

// 获取性能数据
auto stats = ::ttnn::DeviceProfiler::instance().get_statistics();
for (const auto& [op_name, duration] : stats) {
  std::cout << op_name << ": " << duration << " us\n";
}

// 输出示例：
// matmul_kernel: 1234 us
// data_movement: 456 us
// total: 1690 us
```

### 10.3 内存使用监控

```cpp
// 查询分配器统计
auto stats = device.allocator()->get_statistics(BufferType::L1);

std::cout << "L1 Memory Usage:\n";
std::cout << "  Total allocatable: "
          << stats.total_allocatable_size_bytes / 1024 << " KB\n";
std::cout << "  Total allocated: "
          << stats.total_allocated_bytes / 1024 << " KB\n";
std::cout << "  Total free: "
          << stats.total_free_bytes / 1024 << " KB\n";
std::cout << "  Largest free block: "
          << stats.largest_free_block_bytes / 1024 << " KB\n";
std::cout << "  Fragmentation: "
          << stats.fragmentation << "\n";

// 输出示例：
// L1 Memory Usage:
//   Total allocatable: 1024 KB
//   Total allocated: 768 KB
//   Total free: 256 KB
//   Largest free block: 128 KB
//   Fragmentation: 0.15
```

### 10.4 错误诊断

**常见错误及解决方案**：

| 错误 | 原因 | 解决方案 |
|------|------|---------|
| **Out of L1 memory** | L1 分配器耗尽 | 减少 Circular Buffer 大小，增加分片数 |
| **Out of DRAM** | DRAM 分配器耗尽 | 释放不再使用的张量，使用 deallocate() |
| **Program timeout** | Kernel 挂起或死循环 | 检查 kernel 代码，启用 Watcher |
| **Invalid NOC address** | NOC 坐标错误 | 验证 core_range，检查设备网格配置 |
| **PCIe DMA error** | Hugepage 访问失败 | 检查 Hugepage 配置，重启设备 |
| **Version mismatch** | Binary 与 Runtime 不兼容 | 重新编译 binary，或更新 Runtime |

**Watcher 使用**：

```bash
# 启用 Watcher
export TT_METAL_WATCHER=1

# 运行程序
./my_app

# 如果 kernel 挂起，Watcher 会输出：
# [WATCHER] Core (1,1) BRISC stuck at PC=0x12345
# [WATCHER] Core (1,1) NCRISC waiting on semaphore 0x8000
# [WATCHER] Dumping L1 memory...
```

### 10.5 调试技巧

**1. 单步调试 Runtime**：

```bash
# 编译 Debug 版本
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build

# 使用 GDB
gdb --args ./my_app
(gdb) break tt::runtime::ttnn::submit
(gdb) run
(gdb) step
```

**2. 检查 Flatbuffer 内容**：

```bash
# 使用 flatc 工具
flatc --raw-binary --json \
  program.fbs -- model.ttnn > model.json

# 查看 JSON
cat model.json | jq '.programs[0].operations[0]'
```

**3. Kernel DPRINT**：

```cpp
// kernel.cpp
#include "debug/dprint.h"

void kernel_main() {
  DPRINT << "Starting kernel, num_tiles=" << num_tiles << ENDL();

  for (uint32_t i = 0; i < num_tiles; i++) {
    DPRINT << "Processing tile " << i << ENDL();
    // ...
  }

  DPRINT << "Kernel finished" << ENDL();
}
```

**4. 保存中间结果**：

```cpp
// 保存操作的输出到文件
auto output = ::ttnn::matmul(input_a, input_b);

// 读取到 Host
auto host_output = ::ttnn::from_device(output);

// 保存为 NumPy 格式
saveTensorToNumpy(host_output, "output.npy");
```

---

## 总结

### 关键点回顾

1. **两层运行时架构**
   - TTNN Runtime：高层算子执行，自动优化
   - TTMetal Runtime：低层内核调度，精细控制

2. **Flatbuffer 加载**
   - 零拷贝访问，快速启动
   - 包含完整执行信息（操作、张量、设备配置）

3. **设备初始化**
   - UMD 层管理 PCIe 和硬件资源
   - Allocator 管理 L1 和 DRAM 内存
   - Prefetcher/Dispatcher 固件常驻运行

4. **张量管理**
   - 从 TensorDesc 创建设备张量
   - 自动内存分配和布局转换
   - 引用计数自动释放

5. **Dispatch 架构**
   - Hugepage：高效 Host ↔ Device 通信
   - Prefetcher：从 Host 预取命令和数据
   - Dispatcher：分发命令到 Worker Cores
   - Workers：5 个 RISC-V 核心并行执行

6. **性能优化**
   - Program Cache：避免重复编译
   - Trace Mode：录制命令序列快速重放
   - Async Execution：流水线执行

7. **调试工具**
   - 环境变量控制日志级别
   - Tracy Profiler 可视化性能
   - Watcher 监控设备状态

### 文件路径速查

```
运行时核心：
  /home/ubuntu/work/tt/tt-mlir/runtime/lib/runtime.cpp
  /home/ubuntu/work/tt/tt-mlir/runtime/lib/ttnn/runtime.cpp
  /home/ubuntu/work/tt/tt-mlir/runtime/include/tt/runtime/runtime.h

设备管理：
  /home/ubuntu/work/tt/tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/llrt/tt_cluster.cpp
  /home/ubuntu/work/tt/tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/impl/device/device.cpp

Dispatch 固件：
  /home/ubuntu/work/tt/tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/impl/dispatch/kernels/cq_prefetch.cpp
  /home/ubuntu/work/tt/tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/impl/dispatch/kernels/cq_dispatch.cpp

内存管理：
  /home/ubuntu/work/tt/tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/impl/allocator/allocator.cpp
  /home/ubuntu/work/tt/tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/impl/allocator/bank_manager.cpp

TTNN 操作：
  /home/ubuntu/work/tt/tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/
```

### 相关文档

- [06-ttnn-to-flatbuffer.md](06-ttnn-to-flatbuffer.md) - Flatbuffer 序列化详解
- [/home/ubuntu/work/tt/tt-software/tt-metal-analysis/07-dispatch-architecture-deep-dive.md](../tt-metal-analysis/07-dispatch-architecture-deep-dive.md) - Dispatch 架构深度剖析
- [/home/ubuntu/work/tt/tt-software/tt-metal-analysis/04-command-submission.md](../tt-metal-analysis/04-command-submission.md) - 命令提交流程
- [/home/ubuntu/work/tt/tt-software/tt-metal-analysis/05-memory-management.md](../tt-metal-analysis/05-memory-management.md) - 内存管理分析

---

*创建时间: 2025-02*
*最后更新: 2025-02*
