# TTNN 到 Flatbuffer 代码生成与运行时

本文档详细解释 TT-MLIR 编译器如何将 TTNN IR 转换为 Flatbuffer 二进制格式，以及运行时如何加载和执行这些二进制文件。

## 目录

1. [为什么使用 Flatbuffer](#为什么使用-flatbuffer)
2. [Flatbuffer Schema 结构](#flatbuffer-schema-结构)
3. [TTNN IR 到 Flatbuffer 转换](#ttnn-ir-到-flatbuffer-转换)
4. [序列化内容详解](#序列化内容详解)
5. [替代代码生成路径](#替代代码生成路径)
6. [运行时加载与执行](#运行时加载与执行)
7. [TTNN vs TTMetal Runtime](#ttnn-vs-ttmetal-runtime)

---

## 为什么使用 Flatbuffer

Flatbuffer 是 Google 开发的高效序列化库，TT-MLIR 选择 Flatbuffer 作为编译器输出格式有以下原因：

### 优势

1. **零拷贝访问**：可以直接访问序列化数据，无需解析步骤
2. **内存高效**：不需要将整个数据结构加载到内存
3. **向后兼容**：Schema 演化支持，旧版本运行时可以加载新版本二进制
4. **跨平台**：与语言和平台无关的二进制格式
5. **快速加载**：直接 mmap 文件即可使用，启动延迟低

### 使用场景

```
编译时                                运行时
---------                             ---------
TTNN IR                               Flatbuffer Binary
   ↓                                      ↓
TTNNToFlatbuffer.cpp      →→→      Load & Deserialize
   ↓                                      ↓
.ttnn 二进制文件                    Runtime Execution
```

---

## Flatbuffer Schema 结构

### 核心 Schema 文件

Flatbuffer schema 定义在以下文件中：

```
/home/ubuntu/work/tt/tt-mlir/runtime/include/tt/runtime/flatbuffer/
└── types.fbs                    # 运行时类型定义

/home/ubuntu/work/tt/tt-mlir/include/ttmlir/Target/
├── Common/types.fbs             # 通用类型（Arch, DataType, Layout）
├── TTNN/types.fbs              # TTNN 特定类型
├── TTNN/program.fbs            # TTNN Program 结构
└── TTMetal/program.fbs         # TTMetal Program 结构
```

### 1. 设备配置（MeshDeviceOptions）

```flatbuffers
table MeshDeviceOptions {
  mesh_offset: [uint32];              // 设备网格偏移
  device_ids: [int];                  // 物理设备 ID 列表
  num_hw_cqs: uint8;                  // 硬件命令队列数量
  enable_program_cache: bool;         // 是否启用程序缓存
  mesh_shape: [uint32];               // 设备网格形状 [rows, cols]
  l1_small_size: uint64 = null;       // L1 小缓冲区大小
  trace_region_size: uint64 = null;   // Trace 区域大小
  dispatch_core_type: DispatchCoreType = null;  // 调度核心类型
}

enum DispatchCoreType: uint {
  Worker,      // 计算核心
  Ethernet     // 以太网核心
}
```

**用途**：配置多设备执行环境，指定设备拓扑和资源分配。

### 2. 张量描述（TensorDesc）

```flatbuffers
table TensorDesc {
  shape: [int32];                    // 张量形状 [N, C, H, W]
  mesh_shape: [int32];               // 多设备网格形状 [x, y]
  layout: Layout (deprecated);       // 旧版 Layout（已弃用）
  layout_v2: LayoutV2;              // 新版 Layout 描述
  sharding_desc: RuntimeTensorShardingDesc;  // 分片描述
}

// Layout 类型
table LayoutV2 {
  tensor_layout: TensorLayout;       // ROW_MAJOR 或 TILE
  memory_config: MemoryConfig;       // 内存配置
}

enum TensorLayout: ushort {
  RowMajor,    // 行主序
  Tile,        // Tile 格式（32x32 块）
  Invalid,
}
```

**用途**：完整描述张量的形状、布局和内存配置，运行时根据此信息分配内存。

### 3. 内存配置（MemoryConfig）

```flatbuffers
table MemoryConfig {
  tensor_memory_layout: TensorMemoryLayout;  // 内存布局策略
  buffer_type: BufferType;                   // 缓冲区类型
  shard_spec: ShardSpec;                     // 分片规格
  nd_shard_spec: NDShardSpec;                // N 维分片规格
}

enum TensorMemoryLayout: ushort {
  Interleaved,      // 交错存储（所有核心共享）
  HeightSharded,    // 按高度分片
  WidthSharded,     // 按宽度分片
  BlockSharded,     // 按块分片（2D）
}

enum BufferType: ushort {
  DRAM,           // DRAM 内存
  L1,             // L1 缓存
  SystemMemory,   // 主机系统内存
  L1Small,        // L1 小缓冲区
  Trace,          // Trace 缓冲区
}

table ShardSpec {
  core_range_set: CoreRangeSet;      // 核心范围集合
  shape: [int32];                    // 分片形状
  orientation: ShardOrientation;     // 分片方向
}
```

**用途**：指定张量在设备上的存储策略，决定数据如何分布在多个核心上。

### 4. 操作（Operation）

```flatbuffers
table Operation {
  type: OpType;         // 操作类型（union）
  debug_info: string;   // 调试信息（MLIR IR）
  loc_info: string;     // 源位置信息
}

union OpType {
  // 数据移动
  ToDeviceOp,
  FromDeviceOp,
  ToLayoutOp,
  ToMemoryConfigOp,

  // 元素级操作
  EltwiseBinaryOp,
  EltwiseUnaryOp,

  // 线性代数
  MatmulOp,
  LinearOp,

  // 卷积
  Conv2dOp,
  Conv3dOp,

  // 归一化
  LayerNormOp,
  RMSNormOp,

  // Attention
  ScaledDotProductAttentionOp,

  // ... 共 100+ 种操作
}
```

### 5. 程序（Program）

```flatbuffers
table Program {
  name: string;                     // 函数名称
  inputs: [TensorRef];              // 输入张量引用
  outputs: [TensorRef];             // 输出张量引用
  operations: [Operation];          // 操作序列
  dylibs: [DynamicLib];            // 动态库（自定义内核）
  debug_info: DebugInfo;           // 调试信息
  private: bool;                   // 是否为内部函数
  mesh_shape: Dim2d;               // 设备网格形状
}

table TensorRef {
  global_id: uint32;                // 全局张量 ID
  desc: TensorDesc;                 // 张量描述
}
```

**用途**：表示一个完整的执行单元（对应 MLIR 中的 func.func），包含输入输出和操作序列。

### 6. 二进制文件顶层结构

```flatbuffers
table Binary {
  version: Version;                 // 编译器版本
  programs: [Program];              // 所有程序
  system_desc: SystemDesc;          // 系统描述（硬件配置）
  debug_info: DebugInfo;           // 全局调试信息
}

table Version {
  major: uint8;
  minor: uint8;
  patch: uint8;
  hash: string;                     // Git commit hash
}
```

---

## TTNN IR 到 Flatbuffer 转换

### 转换流程

```cpp
// 入口：TTNNToFlatbuffer.cpp
namespace mlir::tt::ttnn {

#define GEN_PASS_DEF_TTNNSERIALIZETOBINARY
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

class TTNNSerializeToBinaryPass {
  void runOnOperation() override {
    // 1. 创建 Flatbuffer Builder
    flatbuffers::FlatBufferBuilder fbb(1024);
    FlatbufferObjectCache cache(&fbb);

    // 2. 遍历模块中的所有函数
    for (auto func : getOperation().getOps<func::FuncOp>()) {
      // 3. 转换函数为 Program
      auto program = funcToProgram(cache, func);
      programs.push_back(program);
    }

    // 4. 创建 Binary
    auto binary = CreateBinaryDirect(fbb, version, programs, systemDesc);
    fbb.Finish(binary);

    // 5. 写入文件
    writeToFile(outputPath, fbb.GetBufferPointer(), fbb.GetSize());
  }
};

}
```

### 关键转换函数

#### 1. 张量类型转换（tensorTypeToFlatbuffer）

```cpp
flatbuffers::Offset<::tt::target::ttnn::TensorDesc>
tensorTypeToFlatbuffer(FlatbufferObjectCache &cache, Type type,
                       ttcore::DeviceAttr deviceAttr,
                       ttcore::ShardStatus shardStatusType,
                       mlir::RankedTensorType localShapeType) {
  auto tensorType = mlir::cast<RankedTensorType>(type);

  // 1. 提取 TTNNLayoutAttr
  TTNNLayoutAttr layoutAttr =
      mlir::cast<ttnn::TTNNLayoutAttr>(tensorType.getEncoding());

  // 2. 转换 shape
  auto shapeInt64 = tensorType.getShape();
  std::vector<int32_t> shape;
  std::transform(shapeInt64.begin(), shapeInt64.end(),
                 std::back_inserter(shape),
                 [](int64_t val) { return static_cast<int32_t>(val); });

  // 3. 获取 mesh shape（多设备支持）
  std::vector<int32_t> meshShape = {1, 1};  // 默认单设备
  if (layoutAttr.getTensorMesh()) {
    auto meshShapeInt64 = deviceAttr.getMeshShape();
    meshShape = std::vector<int32_t>(meshShapeInt64.begin(),
                                     meshShapeInt64.end());
  }

  // 4. 转换分片状态
  auto shardStatus = shardStatusType == ttcore::ShardStatus::Presharded
                         ? ::tt::target::ttnn::ShardStatus::Presharded
                         : ::tt::target::ttnn::ShardStatus::Unsharded;

  // 5. 创建 TensorDesc
  return ::tt::target::ttnn::CreateTensorDescDirect(
      *cache.fbb, &shape, &meshShape,
      cache.getOrCreate(layoutAttr, ttnnLayoutAttrToFlatbuffer, deviceAttr),
      runtimeTensorShardingDesc);
}
```

**转换映射**：

```
MLIR Type                         Flatbuffer
---------                         ----------
RankedTensorType<4x3x224x224>  →  TensorDesc {shape: [4,3,224,224]}
TTNNLayoutAttr                 →  LayoutV2 {layout, memory_config}
tensor::encoding               →  layout field
```

#### 2. Layout 转换（ttnnLayoutAttrToFlatbuffer）

```cpp
flatbuffers::Offset<::tt::target::ttnn::LayoutV2>
ttnnLayoutAttrToFlatbuffer(FlatbufferObjectCache &cache,
                          TTNNLayoutAttr layoutAttr,
                          ttcore::DeviceAttr deviceAttr) {
  // 1. 转换 tensor layout
  ::tt::target::TensorLayout tensorLayout;
  switch (layoutAttr.getLayout()) {
    case Layout::RowMajor:
      tensorLayout = ::tt::target::TensorLayout::RowMajor;
      break;
    case Layout::Tile:
      tensorLayout = ::tt::target::TensorLayout::Tile;
      break;
  }

  // 2. 转换 memory config
  auto memoryConfig = memoryConfigAttrToFlatbuffer(
      cache, layoutAttr, deviceAttr);

  // 3. 创建 LayoutV2
  return ::tt::target::ttnn::CreateLayoutV2(
      *cache.fbb, tensorLayout, memoryConfig);
}
```

#### 3. 内存配置转换（memoryConfigAttrToFlatbuffer）

```cpp
flatbuffers::Offset<::tt::target::ttnn::MemoryConfig>
memoryConfigAttrToFlatbuffer(FlatbufferObjectCache &cache,
                             TTNNLayoutAttr layoutAttr,
                             ttcore::DeviceAttr deviceAttr) {
  // 1. 转换 buffer type
  ::tt::target::BufferType bufferType;
  switch (layoutAttr.getBufferType()) {
    case BufferType::DRAM:
      bufferType = ::tt::target::BufferType::DRAM;
      break;
    case BufferType::L1:
      bufferType = ::tt::target::BufferType::L1;
      break;
    case BufferType::SystemMemory:
      bufferType = ::tt::target::BufferType::SystemMemory;
      break;
  }

  // 2. 转换 memory layout
  ::tt::target::ttnn::TensorMemoryLayout memLayout;
  switch (layoutAttr.getMemLayout()) {
    case MemoryLayout::Interleaved:
      memLayout = ::tt::target::ttnn::TensorMemoryLayout::Interleaved;
      break;
    case MemoryLayout::HeightSharded:
      memLayout = ::tt::target::ttnn::TensorMemoryLayout::HeightSharded;
      break;
    case MemoryLayout::WidthSharded:
      memLayout = ::tt::target::ttnn::TensorMemoryLayout::WidthSharded;
      break;
    case MemoryLayout::BlockSharded:
      memLayout = ::tt::target::ttnn::TensorMemoryLayout::BlockSharded;
      break;
  }

  // 3. 转换 shard spec（如果存在）
  flatbuffers::Offset<::tt::target::ttnn::ShardSpec> shardSpec = 0;
  if (auto shardAttr = layoutAttr.getShardSpec()) {
    shardSpec = shardSpecAttrToFlatbuffer(cache, shardAttr, deviceAttr);
  }

  // 4. 创建 MemoryConfig
  return ::tt::target::ttnn::CreateMemoryConfig(
      *cache.fbb, memLayout, bufferType, shardSpec);
}
```

#### 4. 操作序列化（以 MatmulOp 为例）

```cpp
::flatbuffers::Offset<::tt::target::ttnn::MatmulOp>
createMatmulOp(FlatbufferObjectCache &cache, ttnn::MatmulOp op) {
  // 1. 序列化输入张量
  auto in0 = tensorValueToFlatbuffer(cache, op.getA());
  auto in1 = tensorValueToFlatbuffer(cache, op.getB());

  // 2. 序列化输出张量
  auto out = tensorValueToFlatbuffer(cache, op.getResult());

  // 3. 获取内存配置
  auto memoryConfig = getMemoryConfigIfNeeded(cache, op);

  // 4. 获取操作属性
  auto coreGrid = toFlatbuffer(cache, op.getCoreGrid());

  // 5. 创建 MatmulOp flatbuffer
  return ::tt::target::ttnn::CreateMatmulOp(
      *cache.fbb,
      in0,              // 输入 A
      in1,              // 输入 B
      out,              // 输出
      memoryConfig,     // 内存配置
      nullptr,          // dtype（使用默认）
      coreGrid          // 核心网格
  );
}

// 包装为通用 Operation
::flatbuffers::Offset<::tt::target::ttnn::Operation>
createOperation(FlatbufferObjectCache &cache,
                ::flatbuffers::Offset<::tt::target::ttnn::MatmulOp> matmulOp,
                Operation op) {
  // 获取调试信息
  std::string debugInfo = getDebugString(op);
  std::string locInfo = getLocationString(op.getLoc());

  return ::tt::target::ttnn::CreateOperationDirect(
      *cache.fbb,
      ::tt::target::ttnn::OpType_MatmulOp,  // Union type
      matmulOp.Union(),                      // Union value
      debugInfo.c_str(),
      locInfo.c_str()
  );
}
```

### FlatbufferObjectCache

为了避免重复序列化相同的对象，使用缓存机制：

```cpp
class FlatbufferObjectCache {
  flatbuffers::FlatBufferBuilder *fbb;
  std::unordered_map<void*, flatbuffers::Offset<void>> cache;
  uint32_t global_id = 0;

  template<typename T, typename Creator>
  flatbuffers::Offset<T> getOrCreate(void* key, Creator creator) {
    auto it = cache.find(key);
    if (it != cache.end()) {
      return flatbuffers::Offset<T>(it->second.o);
    }
    auto offset = creator(*this, key);
    cache[key] = offset;
    return offset;
  }
};
```

---

## 序列化内容详解

### 完整的二进制文件结构

```
.ttnn 文件
├── Version Information
│   ├── major: 5
│   ├── minor: 0
│   ├── patch: 0
│   └── hash: "abc123def..."
│
├── SystemDesc (硬件描述)
│   ├── chips: [ChipDesc]
│   │   ├── arch: Blackhole
│   │   ├── grid_size: {y: 8, x: 10}
│   │   ├── l1_size: 1499136
│   │   ├── num_dram_channels: 12
│   │   └── supported_data_types: [Float32, BFloat16, ...]
│   └── chip_descs_by_chip_id: {0: ChipDesc, ...}
│
├── Programs (多个函数)
│   ├── Program 0: "forward"
│   │   ├── inputs: [TensorRef]
│   │   │   ├── global_id: 0
│   │   │   └── desc: TensorDesc
│   │   │       ├── shape: [1, 128, 768]
│   │   │       ├── mesh_shape: [1, 1]
│   │   │       └── layout_v2:
│   │   │           ├── tensor_layout: Tile
│   │   │           └── memory_config:
│   │   │               ├── tensor_memory_layout: Interleaved
│   │   │               └── buffer_type: DRAM
│   │   │
│   │   ├── outputs: [TensorRef]
│   │   │   └── (类似结构)
│   │   │
│   │   ├── operations: [Operation]
│   │   │   ├── Operation 0: ToDeviceOp
│   │   │   │   ├── type: ToDeviceOp
│   │   │   │   ├── input: TensorRef(0)
│   │   │   │   ├── output: TensorRef(1)
│   │   │   │   ├── device: DeviceRef(0)
│   │   │   │   ├── memory_config: MemoryConfig(DRAM)
│   │   │   │   └── debug_info: "%1 = ttnn.to_device %0 ..."
│   │   │   │
│   │   │   ├── Operation 1: MatmulOp
│   │   │   │   ├── type: MatmulOp
│   │   │   │   ├── input_a: TensorRef(1)
│   │   │   │   ├── input_b: TensorRef(2)
│   │   │   │   ├── output: TensorRef(3)
│   │   │   │   ├── memory_config: MemoryConfig(L1)
│   │   │   │   ├── core_grid: {y: 8, x: 8}
│   │   │   │   └── debug_info: "%3 = ttnn.matmul %1, %2 ..."
│   │   │   │
│   │   │   └── ... (更多操作)
│   │   │
│   │   ├── dylibs: [DynamicLib]
│   │   │   └── (如果有自定义内核)
│   │   │
│   │   ├── mesh_shape: {y: 1, x: 1}
│   │   └── private: false
│   │
│   └── Program 1: "const_eval_func" (内部函数)
│       ├── private: true
│       └── ...
│
└── DebugInfo (全局调试信息)
    ├── mlir_module: "module { ... }"  // 原始 MLIR IR
    └── golden_map: {0: GoldenTensor, ...}
```

### 序列化的关键内容

#### 1. 张量元数据（不包含数据）

```cpp
// 序列化的是张量的"配方"，不是数据本身
TensorDesc {
  shape: [1, 3, 224, 224]           // 形状
  mesh_shape: [1, 1]                // 设备拓扑
  layout_v2: {
    tensor_layout: Tile             // Tile 格式
    memory_config: {
      tensor_memory_layout: Interleaved
      buffer_type: DRAM
    }
  }
}

// 运行时会根据此描述分配和创建张量
```

#### 2. 操作参数和属性

```cpp
// 例如：Conv2dOp
Conv2dOp {
  input: TensorRef(0)
  weight: TensorRef(1)
  bias: TensorRef(2)
  output: TensorRef(3)

  // 卷积参数
  kernel_height: 3
  kernel_width: 3
  stride_height: 1
  stride_width: 1
  padding_height: 1
  padding_width: 1
  dilation_height: 1
  dilation_width: 1
  groups: 1

  // 内存配置
  memory_config: MemoryConfig(L1)

  // 设备配置
  device: DeviceRef(0)
}
```

#### 3. 动态库信息（自定义内核）

```cpp
DynamicLib {
  file_name: "custom_kernel_abc123.so"
  binary: [0x7f, 0x45, 0x4c, 0x46, ...]  // .so 文件二进制内容
}

// 运行时会：
// 1. 将二进制写入临时文件
// 2. 使用 dlopen() 加载
// 3. 通过 dlsym() 获取内核函数
```

#### 4. 调试信息

```cpp
DebugInfo {
  // 原始 MLIR 模块（文本格式）
  mlir_module: "
    module {
      func.func @forward(%arg0: tensor<1x128x768xbf16>) -> tensor<1x128x768xbf16> {
        %0 = ttnn.to_device %arg0 : tensor<1x128x768xbf16>
        %1 = ttnn.matmul %0, %weight : tensor<1x128x768xbf16>
        return %1
      }
    }
  "

  // Golden tensors（测试参考）
  golden_map: {
    0: GoldenTensor {
      tensor_id: 0
      file_name: "input_0.npy"
      data: [...]  // 可选：嵌入数据
    }
  }
}
```

---

## 替代代码生成路径

除了 Flatbuffer，TT-MLIR 还支持其他代码生成方式用于调试和验证。

### 1. EmitC：生成 C++ 代码

**目的**：将 TTNN IR 转换为可读的 C++ 代码，便于调试和理解。

**实现位置**：`/home/ubuntu/work/tt/tt-mlir/lib/Dialect/TTNN/Transforms/TTNNToCpp.cpp`

```cpp
namespace mlir::tt::ttnn {

LogicalResult emitTTNNAsCpp(ModuleOp origOp, llvm::raw_ostream &os) {
  ModuleOp op = cast<ModuleOp>(origOp->clone());

  // 1. 运行 TTNN -> EmitC 转换 Pass
  auto pm = PassManager::on<ModuleOp>(op.getContext());
  pm.addPass(createConvertTTNNToEmitCPass());

  if (pm.run(op).failed()) {
    return failure();
  }

  // 2. 使用 MLIR EmitC 后端生成 C++
  if (emitc::translateToCpp(op, os).failed()) {
    return failure();
  }

  return success();
}

}
```

**生成的 C++ 代码示例**：

```cpp
// 输入 TTNN IR:
// %0 = ttnn.to_device %arg0
// %1 = ttnn.add %0, %0

// 生成的 C++ 代码:
#include <ttnn/operations/core.hpp>
#include <ttnn/operations/eltwise.hpp>

ttnn::Tensor forward(ttnn::Tensor arg0) {
  // 将输入移到设备
  auto v0 = ttnn::to_device(arg0, device);

  // 执行加法
  auto memory_config = ttnn::MemoryConfig{
    .memory_layout = ttnn::TensorMemoryLayout::INTERLEAVED,
    .buffer_type = ttnn::BufferType::DRAM
  };
  auto v1 = ttnn::add(v0, v0, std::nullopt, memory_config);

  return v1;
}
```

**使用场景**：
- 调试编译器转换逻辑
- 验证操作语义正确性
- 生成参考实现用于测试
- 教学和文档示例

**优点**：
- 人类可读
- 可以手动修改和测试
- 易于调试

**缺点**：
- 需要重新编译
- 启动延迟高
- 不适合生产环境

### 2. EmitPy：生成 Python 代码

**目的**：生成使用 Python ttnn 库的代码。

虽然代码中引用了 EmitPy 的概念，但实际实现可能不完整。理论上会生成类似：

```python
# 输入 TTNN IR:
# %0 = ttnn.to_device %arg0
# %1 = ttnn.matmul %0, %weight

# 生成的 Python 代码:
import ttnn

def forward(arg0, weight):
    # 将输入移到设备
    v0 = ttnn.to_device(arg0)

    # 矩阵乘法
    v1 = ttnn.matmul(v0, weight, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    return v1
```

**使用场景**：
- 快速原型验证
- 与 Python 测试框架集成
- 交互式调试

### 3. 代码生成路径对比

| 特性 | Flatbuffer | EmitC | EmitPy |
|------|-----------|-------|--------|
| 输出格式 | 二进制 | C++ 源码 | Python 源码 |
| 加载速度 | 极快（零拷贝） | 慢（需编译） | 快 |
| 可读性 | 差（二进制） | 好 | 最好 |
| 生产环境 | ✅ | ❌ | ❌ |
| 调试 | 中等 | 好 | 最好 |
| 性能 | 最高 | 高 | 低 |
| 跨平台 | ✅ | ❌（需重编译） | ✅ |

---

## 运行时加载与执行

### 加载流程

```cpp
// 1. 加载二进制文件
Binary binary = Binary::loadFromPath("model.ttnn");

// 2. 设置运行时环境
tt::runtime::setCompatibleDeviceRuntime(binary);  // 自动选择 TTNN 或 TTMetal

// 3. 打开设备
Device device = tt::runtime::openMeshDevice({
  .device_ids = {0},
  .mesh_shape = {1, 1},
  .enable_program_cache = true
});

// 4. 创建输入张量
std::vector<uint32_t> shape = {1, 128, 768};
Tensor input = tt::runtime::createOwnedHostTensor(
    data.data(), shape, stride, itemsize, DataType::BFloat16);

// 5. 提交执行
std::vector<Tensor> outputs = tt::runtime::submit(
    device,
    binary,
    0,  // program index
    {input}
);

// 6. 获取结果
Tensor output = outputs[0];
auto resultData = tt::runtime::getTensorDataBuffer(output);

// 7. 清理
tt::runtime::closeMeshDevice(device);
```

### Binary 类结构

```cpp
struct Binary : public Flatbuffer {
  Binary(Flatbuffer fb);

  static Binary loadFromPath(const char *path);

  // 查询接口
  std::uint32_t getNumPrograms() const;
  std::string getProgramName(std::uint32_t programIndex) const;
  std::uint32_t getNumProgramInputs(std::uint32_t programIndex) const;
  std::uint32_t getNumProgramOutputs(std::uint32_t programIndex) const;

  // 获取 I/O 描述
  std::vector<TensorDesc> getProgramInputs(std::uint32_t programIndex) const;
  std::vector<TensorDesc> getProgramOutputs(std::uint32_t programIndex) const;

  // 调试
  std::string getSystemDescAsJson() const;
  std::string getProgramOpsAsJson(std::uint32_t programIndex) const;

  const std::pair<std::uint32_t, std::uint32_t>
  getProgramMeshShape(std::uint32_t programIndex) const;
};
```

### Submit 执行流程

```cpp
// runtime/include/tt/runtime/runtime.h
std::vector<Tensor> submit(
    Device deviceHandle,
    Binary executableHandle,
    std::uint32_t programIndex,
    const std::vector<Tensor> &inputs
);
```

**内部实现**（runtime/lib/ttnn/runtime.cpp）：

```cpp
namespace tt::runtime::ttnn {

std::vector<Tensor> submit(Device deviceHandle,
                          Binary executableHandle,
                          std::uint32_t programIndex,
                          const std::vector<Tensor> &inputs) {
  // 1. 获取 Program
  const auto* fbBinary = executableHandle.as<::tt::target::ttnn::Binary>();
  const auto* program = fbBinary->programs()->Get(programIndex);

  // 2. 验证输入
  assert(inputs.size() == program->inputs()->size());

  // 3. 创建张量映射表
  std::unordered_map<uint32_t, ::ttnn::Tensor> tensorMap;
  for (size_t i = 0; i < inputs.size(); ++i) {
    uint32_t globalId = program->inputs()->Get(i)->global_id();
    tensorMap[globalId] = toTTNNTensor(inputs[i]);
  }

  // 4. 按顺序执行操作
  for (const auto* op : *program->operations()) {
    executeOperation(op, tensorMap, deviceHandle);
  }

  // 5. 收集输出
  std::vector<Tensor> outputs;
  for (const auto* outputRef : *program->outputs()) {
    uint32_t globalId = outputRef->global_id();
    outputs.push_back(toRuntimeTensor(tensorMap[globalId]));
  }

  return outputs;
}

}
```

### 操作执行（executeOperation）

```cpp
void executeOperation(const ::tt::target::ttnn::Operation* op,
                     std::unordered_map<uint32_t, ::ttnn::Tensor>& tensorMap,
                     Device device) {
  // 根据操作类型分发
  switch (op->type_type()) {
    case ::tt::target::ttnn::OpType_ToDeviceOp: {
      auto toDeviceOp = op->type_as_ToDeviceOp();
      executeToDeviceOp(toDeviceOp, tensorMap, device);
      break;
    }

    case ::tt::target::ttnn::OpType_MatmulOp: {
      auto matmulOp = op->type_as_MatmulOp();
      executeMatmulOp(matmulOp, tensorMap, device);
      break;
    }

    case ::tt::target::ttnn::OpType_EltwiseBinaryOp: {
      auto eltwiseOp = op->type_as_EltwiseBinaryOp();
      executeEltwiseBinaryOp(eltwiseOp, tensorMap, device);
      break;
    }

    // ... 100+ 种操作

    default:
      LOG_FATAL("Unsupported operation type");
  }
}
```

### 操作执行示例：MatmulOp

```cpp
void executeMatmulOp(const ::tt::target::ttnn::MatmulOp* op,
                    std::unordered_map<uint32_t, ::ttnn::Tensor>& tensorMap,
                    Device device) {
  // 1. 获取输入张量
  auto inputA = tensorMap[op->in0()->global_id()];
  auto inputB = tensorMap[op->in1()->global_id()];

  // 2. 转换内存配置
  std::optional<::ttnn::MemoryConfig> memoryConfig = std::nullopt;
  if (op->memory_config()) {
    memoryConfig = toTTNNMemoryConfig(op->memory_config());
  }

  // 3. 转换数据类型
  std::optional<::ttnn::DataType> dtype = std::nullopt;
  if (op->dtype()) {
    dtype = toTTNNDataType(op->dtype());
  }

  // 4. 获取核心网格
  std::optional<::ttnn::CoreGrid> coreGrid = std::nullopt;
  if (op->core_grid()) {
    coreGrid = ::ttnn::CoreGrid{op->core_grid()->y(), op->core_grid()->x()};
  }

  // 5. 调用 TTNN 库
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

  // 6. 存储结果
  tensorMap[op->out()->global_id()] = result;
}
```

### 设备管理

```cpp
// 打开设备
Device openMeshDevice(const MeshDeviceOptions &options) {
  // 1. 创建设备坐标
  std::vector<chip_id_t> deviceIds(options.device_ids.begin(),
                                   options.device_ids.end());

  // 2. 创建 MeshDevice
  auto meshDevice = std::make_shared<::ttnn::MeshDevice>(
      ::ttnn::MeshShape{options.mesh_shape[0], options.mesh_shape[1]},
      deviceIds,
      options.num_hw_cqs,
      options.l1_small_size.value_or(DEFAULT_L1_SMALL_SIZE),
      options.trace_region_size.value_or(DEFAULT_TRACE_REGION_SIZE)
  );

  // 3. 初始化设备
  if (options.enable_program_cache) {
    ::ttnn::ProgramCache::instance().enable();
  }

  // 4. 返回设备句柄
  return Device(meshDevice);
}

// 关闭设备
void closeMeshDevice(Device parentMesh) {
  auto meshDevice = getMeshDevice(parentMesh);
  ::ttnn::close(*meshDevice);
}
```

---

## TTNN vs TTMetal Runtime

TT-MLIR 支持两层运行时，分别针对不同的抽象级别。

### 运行时层次结构

```
┌─────────────────────────────────────┐
│     应用层 (Python/C++)             │
├─────────────────────────────────────┤
│   TT-MLIR Runtime API               │
│   - submit()                        │
│   - openDevice()                    │
│   - createTensor()                  │
├─────────────────────────────────────┤
│                                     │
│   ┌──────────────┐  ┌─────────────┐│
│   │TTNN Runtime  │  │TTMetal      ││
│   │(High-level)  │  │Runtime      ││
│   │              │  │(Low-level)  ││
│   └──────────────┘  └─────────────┘│
│          │                 │        │
│          └─────────┬───────┘        │
├────────────────────┼────────────────┤
│         TT-Metalium                 │
│         (TTNN + TTMetal Library)    │
└─────────────────────────────────────┘
│         Hardware                    │
│   (Wormhole, Blackhole)             │
└─────────────────────────────────────┘
```

### 1. TTNN Runtime

**目标**：执行高层 TTNN 操作（matmul, conv2d, attention 等）

**特点**：
- 操作粒度大（一个操作可能对应多个内核）
- 自动内存管理
- 高层优化（算子融合、布局转换）
- 面向算法工程师

**Flatbuffer Schema**：`/home/ubuntu/work/tt/tt-mlir/include/ttmlir/Target/TTNN/program.fbs`

```flatbuffers
// TTNN Program
table Program {
  operations: [Operation];  // 高层操作
  inputs: [TensorRef];
  outputs: [TensorRef];
}

union OpType {
  MatmulOp,
  Conv2dOp,
  EltwiseBinaryOp,
  // ... 100+ 高层操作
}
```

**执行流程**：

```cpp
// TTNN 操作执行
void executeMatmulOp(...) {
  // 直接调用 TTNN 库
  auto result = ::ttnn::matmul(inputA, inputB, ...);

  // TTNN 库内部会：
  // 1. 选择最优的矩阵乘法算法
  // 2. 分配临时缓冲区
  // 3. 调度多个内核（预处理、计算、后处理）
  // 4. 管理内存布局转换
}
```

**代码位置**：
```
/home/ubuntu/work/tt/tt-mlir/runtime/lib/ttnn/
├── runtime.cpp              # TTNN Runtime 实现
├── operations/              # 操作执行器
│   ├── matmul.cpp
│   ├── conv.cpp
│   └── ...
└── utils.cpp               # 工具函数
```

### 2. TTMetal Runtime

**目标**：执行低层 TTMetal 命令（kernel dispatch, data movement）

**特点**：
- 操作粒度小（直接调度内核）
- 显式内存管理
- 细粒度控制
- 面向性能工程师

**Flatbuffer Schema**：`/home/ubuntu/work/tt/tt-mlir/include/ttmlir/Target/TTMetal/program.fbs`

```flatbuffers
// TTMetal Program
table Program {
  command_queues: [CommandQueue];  // 命令队列
  buffers: [Buffer];               // 显式缓冲区
}

table CommandQueue {
  commands: [Command];
}

union CommandType {
  CreateBufferCommand,      // 创建缓冲区
  DeallocateBufferCommand,  // 释放缓冲区
  EnqueueProgramCommand,    // 调度内核
  EnqueueReadBufferCommand, // 读取缓冲区
  EnqueueWriteBufferCommand,// 写入缓冲区
  FinishCommand,            // 同步
}
```

**执行流程**：

```cpp
// TTMetal 命令执行
void executeCommand(const Command* cmd, ...) {
  switch (cmd->type()) {
    case CommandType_EnqueueProgramCommand: {
      auto progCmd = cmd->as_EnqueueProgramCommand();

      // 1. 获取编译好的内核二进制
      auto kernelBinary = getKernelBinary(progCmd->kernel_id());

      // 2. 创建 TTMetal Program
      ::tt::tt_metal::Program program;

      // 3. 添加内核到 Program
      ::tt::tt_metal::CreateKernel(
          program,
          kernelBinary,
          coreRange,
          ...
      );

      // 4. 设置运行时参数
      ::tt::tt_metal::SetRuntimeArgs(program, coreCoord, args);

      // 5. 调度执行
      ::tt::tt_metal::EnqueueProgram(commandQueue, program, blocking);
      break;
    }

    case CommandType_CreateBufferCommand: {
      // 显式创建缓冲区
      auto bufferCmd = cmd->as_CreateBufferCommand();
      auto buffer = ::tt::tt_metal::CreateBuffer(
          device,
          bufferCmd->size(),
          bufferCmd->page_size(),
          bufferCmd->buffer_type()
      );
      bufferMap[bufferCmd->buffer_id()] = buffer;
      break;
    }
  }
}
```

**代码位置**：
```
/home/ubuntu/work/tt/tt-mlir/runtime/lib/ttmetal/
├── runtime.cpp              # TTMetal Runtime 实现
└── command_executor.cpp     # 命令执行器
```

### 对比表格

| 特性 | TTNN Runtime | TTMetal Runtime |
|------|-------------|----------------|
| 抽象级别 | 高（算子级） | 低（内核级） |
| 操作示例 | `matmul()` | `EnqueueProgram()` |
| 内存管理 | 自动 | 手动 |
| 性能控制 | 有限 | 完全控制 |
| 代码复杂度 | 简单 | 复杂 |
| 适用场景 | 快速开发 | 性能调优 |
| Schema 文件 | `TTNN/program.fbs` | `TTMetal/program.fbs` |
| 编译 Pass | `TTNNToFlatbuffer` | `TTMetalToFlatbuffer` |

### 运行时选择

```cpp
// 运行时自动选择
void setCompatibleDeviceRuntime(const Binary &binary) {
  // 检查 binary 的 schema
  if (binary.isDeviceRuntime(DeviceRuntime::TTNN)) {
    setCurrentDeviceRuntime(DeviceRuntime::TTNN);
  } else if (binary.isDeviceRuntime(DeviceRuntime::TTMetal)) {
    setCurrentDeviceRuntime(DeviceRuntime::TTMetal);
  }
}

// 手动选择
setCurrentDeviceRuntime(DeviceRuntime::TTNN);  // 或 TTMetal
```

### 编译流程差异

```
┌───────────────────────────────────────────────┐
│             MLIR 前端                         │
│   (StableHLO → TTIR → TTNN Dialect)          │
└───────────────┬───────────────────────────────┘
                │
        ┌───────┴────────┐
        │                │
        v                v
┌───────────────┐  ┌──────────────────┐
│TTNN Backend   │  │TTMetal Backend   │
│               │  │                  │
│TTNNToFlatbuf  │  │TTNNToTTMetal     │
│               │  │TTMetalToFlatbuf  │
└───────────────┘  └──────────────────┘
        │                │
        v                v
┌───────────────┐  ┌──────────────────┐
│.ttnn binary   │  │.ttm binary       │
│(高层操作)     │  │(低层命令)        │
└───────────────┘  └──────────────────┘
```

**TTNN 路径**：
1. TTNN Dialect → TTNN Flatbuffer
2. 直接序列化 TTNN 操作
3. 运行时调用 TTNN 库

**TTMetal 路径**：
1. TTNN Dialect → TTMetal Dialect
2. TTMetal Dialect → TTMetal Flatbuffer
3. 包含内核编译和命令生成
4. 运行时直接调度内核

---

## 总结

### 关键点回顾

1. **Flatbuffer 是生产环境的标准输出**
   - 高效、零拷贝、跨平台
   - 包含完整的执行信息（操作、张量、设备配置）

2. **转换过程是类型驱动的**
   - MLIR Type → TensorDesc
   - MLIR Attribute → Flatbuffer Table
   - MLIR Operation → Operation Union

3. **运行时是数据驱动的**
   - 读取 Flatbuffer → 创建张量映射 → 顺序执行操作
   - 通过张量 global_id 跟踪数据流

4. **两层运行时提供灵活性**
   - TTNN：快速开发、自动优化
   - TTMetal：极致性能、精细控制

5. **调试路径辅助开发**
   - EmitC：生成可读 C++ 代码
   - 保留 MLIR IR 在 debug_info 中

### 文件路径速查

```
编译器端：
  /home/ubuntu/work/tt/tt-mlir/lib/Target/TTNN/TTNNToFlatbuffer.cpp
  /home/ubuntu/work/tt/tt-mlir/lib/Target/TTMetal/TTMetalToFlatbuffer.cpp

Schema 定义：
  /home/ubuntu/work/tt/tt-mlir/include/ttmlir/Target/Common/types.fbs
  /home/ubuntu/work/tt/tt-mlir/include/ttmlir/Target/TTNN/types.fbs
  /home/ubuntu/work/tt/tt-mlir/include/ttmlir/Target/TTNN/program.fbs
  /home/ubuntu/work/tt/tt-mlir/include/ttmlir/Target/TTMetal/program.fbs
  /home/ubuntu/work/tt/tt-mlir/runtime/include/tt/runtime/flatbuffer/types.fbs

运行时端：
  /home/ubuntu/work/tt/tt-mlir/runtime/lib/runtime.cpp              # 顶层 API
  /home/ubuntu/work/tt/tt-mlir/runtime/lib/ttnn/runtime.cpp         # TTNN Runtime
  /home/ubuntu/work/tt/tt-mlir/runtime/include/tt/runtime/runtime.h # 公共 API

调试工具：
  /home/ubuntu/work/tt/tt-mlir/lib/Dialect/TTNN/Transforms/TTNNToCpp.cpp  # EmitC
```

### 下一步阅读

- [内核编译](07-kernel-compilation.md)：了解如何编译自定义内核到动态库
- [运行时执行](08-runtime-execution.md)：深入理解运行时内部机制
- [端到端示例](09-end-to-end-example.md)：完整的模型编译和执行流程
