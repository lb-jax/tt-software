# TTIR 到 TTNN 降级流程

## 目录
- [转换概述](#转换概述)
- [TTNN Pipeline 架构](#ttnn-pipeline-架构)
- [Layout 和 Memory Configuration](#layout-和-memory-configuration)
- [TTNN 操作特性](#ttnn-操作特性)
- [转换模式详解](#转换模式详解)
- [优化 Pass](#优化-pass)
- [代码示例](#代码示例)

---

## 转换概述

### TTIR 和 TTNN 的区别

**TTIR (Tenstorrent IR)** 是一个与硬件无关的中间表示：
- 抽象的张量操作，不包含硬件特定的细节
- 支持任意元素类型（FP32, FP16, INT8 等）
- 没有显式的内存布局和位置信息
- DPS (Destination-Passing Style) 风格的操作

**TTNN (TT Neural Network)** 是硬件特定的底层表示：
- 操作映射到 Tenstorrent 硬件的 TTNN 库
- 显式的内存配置（L1, DRAM, System Memory）
- 显式的布局信息（Row-Major, Tile）
- 显式的分片配置（Sharding）
- 硬件特定的数据类型（BFP8, BFP4）

### 为什么需要这一层转换

1. **硬件映射**：将抽象操作映射到硬件支持的具体操作
2. **内存管理**：决定张量在 L1/DRAM/System Memory 中的存储位置
3. **布局优化**：选择最优的内存布局（Tile 用于计算，Row-Major 用于数据传输）
4. **性能调优**：插入 layout 转换、配置分片策略以最大化硬件利用率
5. **硬件约束**：确保操作满足硬件约束（对齐、大小限制等）

### 转换的主要挑战

1. **Layout 决策**：自动决定何时使用 Tile vs Row-Major layout
2. **Memory 分配**：平衡 L1（快但小）和 DRAM（慢但大）的使用
3. **Sharding 策略**：将张量分布到多核心以并行化计算
4. **约束满足**：确保操作满足硬件的各种约束条件
5. **性能权衡**：在转换开销和计算性能之间找到平衡

---

## TTNN Pipeline 架构

### 完整 Pipeline 概览

TTIR 到 TTNN 的转换由多个阶段组成：

```
TTIR Operations (抽象)
        ↓
[TTIR Preparation Passes]
  - Fusing
  - Decomposition
  - Implicit Broadcast Fold
        ↓
[TTNN Lowering Passes]
  - TTNNLayout          (添加 layout 信息)
  - ConvertTTIRToTTNN   (转换操作)
  - RemoveDeadValues    (清理未使用的值)
        ↓
[TTNN Workaround Passes]
  - Layout Workarounds
  - Decomposition Workarounds
        ↓
[TTNN Analysis Passes]
  - RowMajorLayoutPropagation  (传播 Row-Major layout)
  - TTNNOptimizer              (优化 layout 和 memory)
  - OperationValidation        (验证并 fallback)
  - PrepareConv2dWeights       (准备 Conv2d 权重)
        ↓
[TTNN Finalization]
  - DecomposeLayouts    (分解 layout ops)
  - Deallocate          (插入释放操作)
        ↓
TTNN Operations (硬件特定)
```

### 核心 Pipeline 函数

#### 1. TTNN Lowering Passes

```cpp
void createTTNNPipelineLoweringPasses(OpPassManager &pm,
                                      bool removeDeadValuesEnabled) {
  // 添加 layout 信息到所有张量
  pm.addPass(createTTNNLayout());

  // 将 TTIR 操作转换为 TTNN 操作
  pm.addPass(createConvertTTIRToTTNNPass());

  // 移除未使用的值
  if (removeDeadValuesEnabled) {
    pm.addPass(mlir::createRemoveDeadValuesPass());
  }
}
```

**关键职责**：
- `TTNNLayout`：为每个张量添加初始的 layout 属性（默认 DRAM + Tile）
- `ConvertTTIRToTTNN`：1-to-1 操作转换，使用 conversion patterns
- `RemoveDeadValues`：清理中间变量

#### 2. TTNN Analysis Passes

```cpp
void createTTNNPipelineAnalysisPasses(
    OpPassManager &pm, const TTIRToTTNNDevicePipelineOptions &options) {

  if (options.optimizerPassEnabled) {
    // Row-Major layout 传播
    innerPm.addPass(createTTNNRowMajorLayoutPropagation());

    // 优化 layout 和 memory configuration
    innerPm.addPass(createTTNNOptimizer(optimizerOptions));

    // 规范化
    innerPm.addPass(mlir::createCanonicalizerPass());

    // 验证操作并应用 fallback
    innerPm.addPass(createTTNNOperationValidationAndFallback(
        validationOptions));

    // 准备 Conv2d 权重
    innerPm.addPass(createTTNNPrepareConv2dWeightsAndBias());
  }
}
```

**关键职责**：
- `RowMajorLayoutPropagation`：从输入传播 Row-Major layout，减少不必要的转换
- `TTNNOptimizer`：基于性能模型优化 layout 和 sharding
- `OperationValidationAndFallback`：验证硬件约束，应用 fallback 策略
- `PrepareConv2dWeightsAndBias`：预处理 Conv2d 权重以提高性能

---

## Layout 和 Memory Configuration

### Layout 类型

#### 1. Row-Major Layout
```cpp
// Row-Major: 数据按行连续存储
// 适用于：数据传输、Element-wise 操作、某些 reduction
Layout::RowMajor
```

**特点**：
- 内存连续，适合 DMA 传输
- 不需要 padding 到 tile 边界
- Element-wise 操作可以直接在 Row-Major 上执行

**MLIR 表示**：
```mlir
tensor<32x64xbf16, #ttnn.ttnn_layout<
  ...,
  layout = #ttnn.layout<row_major>,
  ...
>>
```

#### 2. Tile Layout (32x32)
```cpp
// Tile: 数据组织成 32x32 的块
// 适用于：MatMul, Conv, 需要 Tensor Core 的操作
Layout::Tile
```

**特点**：
- 32x32 的块，匹配硬件的 Tensor Core
- MatMul 和 Conv 需要 Tile layout
- 需要 padding 到 32 的倍数
- 使用 `TileType` 作为元素类型

**MLIR 表示**：
```mlir
tensor<32x64xbf16, #ttnn.ttnn_layout<
  ...,
  element_type = !ttcore.tile<32, 32, bf16>,
  layout = #ttnn.layout<tile>,
  ...
>>
```

### Memory Configuration

#### Buffer Types

```cpp
enum class BufferType {
  SystemMemory,  // Host 内存（CPU）
  DRAM,          // 设备 DRAM（慢但大，~12-16GB）
  L1,            // 设备 L1 SRAM（快但小，~1-2MB per core）
  L1_SMALL,      // 预留
  TRACE          // Trace 缓冲区
};
```

#### Tensor Memory Layout

```cpp
enum class TensorMemoryLayout {
  Interleaved,      // 交错存储，跨 banks/cores
  SingleBank,       // 单个 bank
  HeightSharded,    // 按高度分片（每个核心处理部分行）
  WidthSharded,     // 按宽度分片（每个核心处理部分列）
  BlockSharded      // 块分片（每个核心处理一个矩形块）
};
```

#### Memory Config 属性

```mlir
#ttnn.memory_config<
  #dram,              // BufferType
  <interleaved>       // TensorMemoryLayout
>

#ttnn.memory_config<
  #l1,                // BufferType
  <block_sharded>,    // TensorMemoryLayout
  #ttnn.shard_spec<   // ShardSpec（仅用于 sharded layouts）
    #ttnn.core_range_set<[#ttnn.core_range<(0,0), (7, 0)>]>,
    <32x128>,         // 每个核心的 shard shape
    <row_major>,      // Shard orientation
    <physical>        // Sharding mode
  >
>
```

### Sharding 策略

Sharding 将张量分布到多个核心以并行化计算：

#### Height Sharded
```
原始张量 [128, 256]
→ 分布到 8 个核心
→ 每个核心: [16, 256]

适用于：行并行的操作（某些 element-wise、reduction）
```

#### Width Sharded
```
原始张量 [128, 256]
→ 分布到 8 个核心
→ 每个核心: [128, 32]

适用于：列并行的操作
```

#### Block Sharded
```
原始张量 [128, 256]
→ 分布到 4x2 grid (8 cores)
→ 每个核心: [32, 128]

适用于：MatMul, Conv（最通用和高效）
```

### 数据类型

#### 标准数据类型
```cpp
DataType::Float32    // FP32 - 32-bit float
DataType::Float16    // FP16 - 16-bit float
DataType::BFloat16   // BF16 - 16-bit bfloat
```

#### 块浮点类型（硬件特定）
```cpp
DataType::BFP_BFloat8  // BFP8 - 8-bit block float (1个指数 + 8个尾数)
DataType::BFP_BFloat4  // BFP4 - 4-bit block float (1个指数 + 8个尾数)
```

**BFP 优势**：
- 压缩存储和带宽
- 保持数值范围（共享指数）
- 适用于权重存储

---

## TTNN 操作特性

### TTNN 操作的关键属性

每个 TTNN 操作都带有以下属性：

```mlir
%result = "ttnn.operation"(%input) <{
  // 1. Layout 信息（在类型中）
  // 2. Memory configuration
  memory_config = #ttnn.memory_config<#dram, <interleaved>>,

  // 3. Data type
  dtype = #ttcore.supportedDataTypes<bf16>,

  // 4. 操作特定参数
  // ...
}> : (tensor<...>) -> tensor<...>
```

### TTNN 操作接口

#### 1. DPS 接口
TTNN 操作不再使用显式的 DPS output 参数，而是通过属性控制输出：

```cpp
// TTIR (DPS style)
%output = ttir.empty() : tensor<32x32xf32>
%result = ttir.add(%a, %b, %output)
  : (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>)
  -> tensor<32x32xf32>

// TTNN (无显式 output)
%result = "ttnn.add"(%a, %b) <{
  memory_config = #ttnn.memory_config<...>
}> : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
```

#### 2. Device 管理
TTNN 操作需要访问设备：

```cpp
// 获取或插入设备
%device = "ttnn.get_device"() : () -> !ttnn.device

// 操作使用设备（隐式或显式）
%result = "ttnn.empty"(%device) <{...}>
  : (!ttnn.device) -> tensor<...>
```

---

## 转换模式详解

### 转换框架

TTIR 到 TTNN 的转换使用 MLIR 的 Dialect Conversion 框架：

```cpp
class TTNNTypeConverter : public TypeConverter {
  // 转换张量类型，添加 TTNNLayoutAttr
  addConversion([=](RankedTensorType type) -> Type {
    TTNNLayoutAttr layout = createLayoutAttr(ctx, deviceGrid, type,
                                             bufferType, isTiled);
    return RankedTensorType::get(type.getShape(),
                                 type.getElementType(), layout);
  });
};
```

### 常见操作转换

#### 1. Element-wise 操作 (Add, Multiply, etc.)

**TTIR**:
```mlir
%2 = ttir.add %0, %1, %out
  : tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>
  -> tensor<32x32xbf16>
```

**TTNN**:
```mlir
%2 = "ttnn.add"(%0, %1)
  : (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>)
  -> tensor<32x32xbf16, #layout>
```

**Conversion Pattern**:
```cpp
class ElementwiseBinaryOpConversionPattern
    : public OpConversionPattern<ttir::AddOp> {
  LogicalResult matchAndRewrite(
      ttir::AddOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ttnn::AddOp>(
        op,
        this->getTypeConverter()->convertType(op.getType()),
        adaptor.getLhs(),
        adaptor.getRhs());
    return success();
  }
};
```

#### 2. MatMul 操作

**TTIR**:
```mlir
%2 = ttir.matmul %0, %1, %out {
  transpose_a = false,
  transpose_b = false
} : tensor<32x64xbf16>, tensor<64x128xbf16>, tensor<32x128xbf16>
  -> tensor<32x128xbf16>
```

**TTNN**:
```mlir
%2 = "ttnn.matmul"(%0, %1) <{
  transpose_a = false,
  transpose_b = false,
  matmul_program_config = null,  // 将由 optimizer 填充
  activation = null
}> : (tensor<32x64xbf16, #layout_a>,
      tensor<64x128xbf16, #layout_b>)
   -> tensor<32x128xbf16, #layout_out>
```

**关键点**：
- MatMul 要求输入为 Tile layout
- `matmul_program_config` 由 optimizer 根据形状和硬件特性设置
- 可以融合 activation（如 ReLU）

**Conversion Pattern**:
```cpp
class MatmulOpConversionPattern
    : public OpConversionPattern<ttir::MatmulOp> {
  LogicalResult matchAndRewrite(
      ttir::MatmulOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ttnn::MatmulOp>(
        op,
        this->getTypeConverter()->convertType(op.getType()),
        adaptor.getA(),
        adaptor.getB(),
        adaptor.getTransposeA(),
        adaptor.getTransposeB(),
        /*matmul_program_config=*/nullptr,  // 稍后填充
        /*activation=*/nullptr);
    return success();
  }
};
```

#### 3. Conv2d 操作

**TTIR**:
```mlir
%output = ttir.conv2d %input, %weight, %bias, %out {
  stride = [1, 1],
  padding = [0, 0],
  dilation = [1, 1],
  groups = 1
} : tensor<1x3x224x224xbf16>,   // NCHW
    tensor<64x3x3x3xbf16>,       // OIHW
    tensor<64xbf16>,
    tensor<1x64x224x224xbf16>
  -> tensor<1x64x224x224xbf16>
```

**TTNN**:
```mlir
%output = "ttnn.conv2d"(%input_flattened, %weight, %bias, %device) <{
  in_channels = 3,
  out_channels = 64,
  batch_size = 1,
  input_height = 224,
  input_width = 224,
  kernel_size = [3, 3],
  stride = [1, 1],
  padding = [0, 0],
  dilation = [1, 1],
  groups = 1,
  conv_config = null  // 将由 optimizer 填充
}> : (tensor<?xbf16, #layout>,         // flattened input
      tensor<64x3x3x3xbf16, #layout>,  // weight
      tensor<64xbf16, #layout>,        // bias
      !ttnn.device)
   -> tensor<?xbf16, #layout>
```

**关键点**：
- TTNN Conv2d 需要 flattened input（通过 FlattenSlidingWindow pass）
- 权重需要预处理（PrepareConv2dWeights）
- Conv2d 是最复杂的操作之一，有多个 program configs

#### 4. Reduction 操作 (Sum, Max, Mean)

**TTIR**:
```mlir
%1 = ttir.sum %0, %out {
  dim_arg = [1 : i32],
  keep_dim = true
} : tensor<32x64xbf16>, tensor<32x1xbf16> -> tensor<32x1xbf16>
```

**TTNN**:
```mlir
%1 = "ttnn.sum"(%0) <{
  dim_arg = 1 : i64,
  keep_dim = true,
  memory_config = null
}> : (tensor<32x64xbf16, #layout>) -> tensor<32x1xbf16, #layout>
```

**关键点**：
- 多维 reduction 需要分解为多个单维 reduction
- 某些 reduction 对 layout 有特殊要求

#### 5. Special 操作

##### Embedding
**TTIR**:
```mlir
%output = ttir.embedding %input, %weight, %out
  : tensor<2x4xi32>,          // indices [batch, seq]
    tensor<512x128xbf16>,     // weight [vocab, embed_dim]
    tensor<2x4x128xbf16>
  -> tensor<2x4x128xbf16>
```

**TTNN**:
```mlir
%output = "ttnn.embedding"(%input, %weight)
  : (tensor<2x4xi32, #layout>,
     tensor<512x128xbf16, #layout>)
  -> tensor<2x4x128xbf16, #layout>
```

##### Softmax
**TTIR**:
```mlir
%output = ttir.softmax %input, %out {
  dimension = 1 : si32
} : tensor<32x64xbf16>, tensor<32x64xbf16> -> tensor<32x64xbf16>
```

**TTNN**:
```mlir
%output = "ttnn.softmax"(%input) <{
  dimension = 1 : si32,
  numeric_stable = true
}> : (tensor<32x64xbf16, #layout>) -> tensor<32x64xbf16, #layout>
```

##### LayerNorm
```mlir
%output = "ttnn.layer_norm"(%input, %weight, %bias) <{
  epsilon = 1.0e-5 : f32,
  memory_config = null
}> : (tensor<32x64xbf16, #layout>,
      tensor<64xbf16, #layout>,
      tensor<64xbf16, #layout>)
   -> tensor<32x64xbf16, #layout>
```

### Empty/Constant 操作转换

#### Empty Op
**TTIR**:
```mlir
%0 = ttir.empty() : tensor<32x32xbf16>
```

**TTNN**:
```mlir
// 设备内存
%device = "ttnn.get_device"() : () -> !ttnn.device
%0 = "ttnn.empty"(%device) <{
  shape = #ttnn.shape<32x32>,
  dtype = #ttcore.supportedDataTypes<bf16>,
  layout = #ttnn.layout<tile>,
  memory_config = #ttnn.memory_config<#dram, <interleaved>>
}> : (!ttnn.device) -> tensor<32x32xbf16, #layout>

// 或 System Memory
%0 = "ttnn.zeros"() <{
  shape = #ttnn.shape<32x32>,
  dtype = #ttcore.supportedDataTypes<bf16>,
  layout = #ttnn.layout<row_major>,
  memory_config = null
}> : () -> tensor<32x32xbf16, #layout>
```

#### Full Op
**TTIR**:
```mlir
%0 = ttir.full %fill_value {
  shape = [32, 32]
} : f32 -> tensor<32x32xf32>
```

**TTNN**:
```mlir
%device = "ttnn.get_device"() : () -> !ttnn.device
%0 = "ttnn.full"(%fill_value, %device)
  : (f32, !ttnn.device) -> tensor<32x32xf32, #layout>
```

### ToLayout 操作

`ToLayout` 是 TTNN 中最重要的操作之一，用于在不同 layout 之间转换：

```mlir
// Row-Major → Tile
%1 = "ttnn.to_layout"(%0) <{
  layout = #ttnn.layout<tile>,
  dtype = #ttcore.supportedDataTypes<bf16>,
  memory_config = #ttnn.memory_config<#dram, <interleaved>>
}> : (tensor<32x64xbf16, #layout_rm>)
   -> tensor<32x64xbf16, #layout_tile>

// Tile → Row-Major
%2 = "ttnn.to_layout"(%1) <{
  layout = #ttnn.layout<row_major>,
  dtype = #ttcore.supportedDataTypes<bf16>,
  memory_config = #ttnn.memory_config<#dram, <interleaved>>
}> : (tensor<32x64xbf16, #layout_tile>)
   -> tensor<32x64xbf16, #layout_rm>

// L1 → DRAM (memory transfer)
%3 = "ttnn.to_memory_config"(%1) <{
  memory_config = #ttnn.memory_config<#dram, <interleaved>>
}> : (tensor<32x64xbf16, #layout_l1>)
   -> tensor<32x64xbf16, #layout_dram>
```

---

## 优化 Pass

### 1. TTNNRowMajorLayoutPropagation

**目标**：从输入传播 Row-Major layout，减少不必要的 Tile 转换

**算法**：
```
1. 识别函数输入参数中的 Row-Major 张量（通常是整数索引）
2. 查找这些输入上的 ToLayout(RM→Tile) 操作
3. 如果可以移除这些 ToLayout，则：
   a. 移除 ToLayout 操作
   b. 传播 RM layout 到下游操作
   c. 在需要 Tile 的操作前插入 ToLayout
```

**示例**：
```mlir
// Before
func @forward(%indices: tensor<2x4xi32, #rm_layout>) {
  %0 = "ttnn.to_layout"(%indices) <{layout = tile}>
    : (tensor<2x4xi32, #rm_layout>) -> (tensor<2x4xi32, #tile_layout>)
  %1 = "ttnn.gather"(%weight, %0)  // gather 可以在 RM 上工作
    : (tensor<512x128xbf16>, tensor<2x4xi32, #tile_layout>)
    -> tensor<2x4x128xbf16>
  return %1
}

// After (ToLayout 被移除)
func @forward(%indices: tensor<2x4xi32, #rm_layout>) {
  %1 = "ttnn.gather"(%weight, %indices)  // 直接使用 RM
    : (tensor<512x128xbf16>, tensor<2x4xi32, #rm_layout>)
    -> tensor<2x4x128xbf16>
  return %1
}
```

**关键代码**：
```cpp
// 识别输入参数
llvm::SmallVector<Value> identifyInputArguments(func::FuncOp func) {
  llvm::SmallVector<Value> rmCandidates;
  for (BlockArgument arg : func.getArguments()) {
    if (isInputArgument(arg, func) && hasIntegerElementType(arg)) {
      rmCandidates.push_back(arg);
    }
  }
  return rmCandidates;
}

// 查找冗余的 ToLayout
llvm::SmallVector<ttnn::ToLayoutOp>
findRedundantToLayoutOps(const llvm::SmallVector<Value> &inputArgs) {
  llvm::SmallVector<ttnn::ToLayoutOp> opsToRemove;
  for (Value arg : inputArgs) {
    for (const auto &user : arg.getUsers()) {
      if (auto toLayoutOp = dyn_cast<ttnn::ToLayoutOp>(user)) {
        // 检查是否只是 RM→Tile 转换
        if (canRemoveToLayout(toLayoutOp)) {
          opsToRemove.push_back(toLayoutOp);
        }
      }
    }
  }
  return opsToRemove;
}
```

### 2. TTNNOptimizer

**目标**：基于性能模型优化 layout、memory config 和 sharding

**优化策略**：
1. **Layout 选择**：为每个操作选择最优 layout（RM vs Tile）
2. **Memory 配置**：决定 L1 vs DRAM
3. **Sharding 决策**：选择 sharding 策略和 grid shape
4. **Program Config**：设置操作特定的配置（如 MatMul 的 block 大小）

**关键步骤**：
```cpp
void TTNNOptimizer::runOnOperation() {
  // 1. 收集所有操作和约束
  collectOperations();

  // 2. 为每个操作生成候选配置
  for (auto op : operations) {
    generateCandidateConfigs(op);
  }

  // 3. 使用性能模型评估每个配置
  for (auto config : candidates) {
    config.performance = evaluateWithOpModel(config);
  }

  // 4. 选择最优配置（考虑全局依赖）
  OptimalConfig bestConfig = selectOptimalConfig(candidates);

  // 5. 应用配置
  applyConfig(bestConfig);
}
```

**OpModel 集成**：
```cpp
// 使用 tt-metal 的 OpModel 评估性能
double evaluateWithOpModel(OpConfig config) {
  // 创建 OpModel 输入
  auto opModelInput = createOpModelInput(config);

  // 调用 OpModel
  auto result = opModel->evaluate(opModelInput);

  // 返回估计的执行时间
  return result.executionTime;
}
```

### 3. TTNNOperationValidationAndFallback

**目标**：验证操作是否满足硬件约束，并应用 fallback 策略

**验证内容**：
1. **内存约束**：操作的输入/输出是否适合 L1
2. **对齐约束**：张量大小是否满足对齐要求
3. **Sharding 约束**：shard 大小是否有效
4. **硬件限制**：操作是否支持当前配置

**Fallback 策略**：
```cpp
LogicalResult validateAndFallback(Operation *op) {
  // 1. 尝试当前配置
  if (validateOperation(op)) {
    return success();
  }

  // 2. Fallback 策略层次
  // Level 1: 更改 layout
  if (tryLayoutFallback(op)) {
    return success();
  }

  // Level 2: 更改 memory config
  if (tryMemoryFallback(op)) {
    return success();
  }

  // Level 3: 更改 sharding
  if (tryShardingFallback(op)) {
    return success();
  }

  // Level 4: 添加中间转换
  if (tryIntermediateConversion(op)) {
    return success();
  }

  // 失败
  return failure();
}
```

**示例**：
```mlir
// Original: MatMul with large tensors in L1 (超出内存)
%result = "ttnn.matmul"(%a, %b) <{
  memory_config = #ttnn.memory_config<#l1, <interleaved>>
}> : (tensor<1024x1024xbf16>, tensor<1024x1024xbf16>)
   -> tensor<1024x1024xbf16>

// After fallback: 使用 DRAM
%result = "ttnn.matmul"(%a, %b) <{
  memory_config = #ttnn.memory_config<#dram, <interleaved>>
}> : (tensor<1024x1024xbf16>, tensor<1024x1024xbf16>)
   -> tensor<1024x1024xbf16>
```

### 4. TTNNPrepareConv2dWeightsAndBias

**目标**：预处理 Conv2d 权重以提高运行时性能

**转换**：
```mlir
// Before
%output = "ttnn.conv2d"(%input, %weight, %bias, %device) <{...}>

// After
%prepared_weight = "ttnn.prepare_conv2d_weights"(%weight, %device) <{
  input_channels = 3,
  output_channels = 64,
  kernel_size = [3, 3],
  ...
}>

%prepared_bias = "ttnn.prepare_conv2d_bias"(%bias, %device) <{
  output_channels = 64,
  ...
}>

%output = "ttnn.conv2d"(%input, %prepared_weight, %prepared_bias, %device) <{...}>
```

**优势**：
- `prepare_conv2d_weights` 可以被 const-eval（常量折叠）
- 预处理的权重格式更适合硬件
- 减少运行时开销

### 5. TTNNWeightBFP8Conversion

**目标**：将权重转换为 BFP8 以减少内存和带宽

**转换**：
```mlir
// Before
%output = "ttnn.matmul"(%activation, %weight) <{...}>
  : (tensor<32x64xbf16>, tensor<64x128xbf16>)
  -> tensor<32x128xbf16>

// After
%weight_bfp8 = "ttnn.typecast"(%weight) <{
  dtype = #ttcore.supportedDataTypes<bfp_bf8>
}> : (tensor<64x128xbf16>) -> (tensor<64x128xbfp8>)

%output = "ttnn.matmul"(%activation, %weight_bfp8) <{...}>
  : (tensor<32x64xbf16>, tensor<64x128xbfp8>)
  -> tensor<32x128xbf16>
```

**规则**：
- 仅转换权重（参数），不转换激活
- 保持输出精度（BF16/FP32）
- 仅在显式启用时应用（`--experimental-bfp8-weights`）

---

## 代码示例

### 完整示例 1: Simple Add

#### TTIR 输入
```mlir
module {
  func.func @add(%arg0: tensor<32x32xbf16>,
                 %arg1: tensor<32x32xbf16>)
      -> tensor<32x32xbf16> {
    %0 = ttir.empty() : tensor<32x32xbf16>
    %1 = ttir.add %arg0, %arg1, %0
      : tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>
      -> tensor<32x32xbf16>
    return %1 : tensor<32x32xbf16>
  }
}
```

#### TTNN 输出
```mlir
#layout = #ttnn.ttnn_layout<
  (d0, d1) -> (d0, d1),
  <32x32>,
  memref<1x1x32x32xbf16, #dram>,
  #ttnn.tensor_memory_layout<interleaved>
>

module {
  func.func @add(%arg0: tensor<32x32xbf16, #layout>,
                 %arg1: tensor<32x32xbf16, #layout>)
      -> tensor<32x32xbf16, #layout> {
    // 获取设备
    %device = "ttnn.get_device"() <{
      mesh_offset = #ttnn<mesh_offset 0x0>,
      mesh_shape = #ttnn<mesh_shape 1x1>
    }> : () -> !ttnn.device

    // Add 操作（无需显式 empty）
    %1 = "ttnn.add"(%arg0, %arg1) <{
      output_dtype = #ttcore.supportedDataTypes<bf16>
    }> : (tensor<32x32xbf16, #layout>,
          tensor<32x32xbf16, #layout>)
       -> tensor<32x32xbf16, #layout>

    return %1 : tensor<32x32xbf16, #layout>
  }
}
```

### 完整示例 2: MatMul with Layout Conversions

#### TTIR 输入
```mlir
module {
  func.func @matmul(%arg0: tensor<32x64xbf16>,
                    %arg1: tensor<64x128xbf16>)
      -> tensor<32x128xbf16> {
    %0 = ttir.empty() : tensor<32x128xbf16>
    %1 = ttir.matmul %arg0, %arg1, %0
      : tensor<32x64xbf16>, tensor<64x128xbf16>, tensor<32x128xbf16>
      -> tensor<32x128xbf16>
    return %1 : tensor<32x128xbf16>
  }
}
```

#### TTNN 输出（初始，TTNNLayout pass 后）
```mlir
#layout_tile = #ttnn.ttnn_layout<
  (d0, d1) -> (d0, d1),
  <32x64>,
  memref<1x1x!tt.tile<32x32, bf16>, !tt.tile<32x32, bf16>>,
  #ttnn.tensor_memory_layout<interleaved>,
  #ttnn.buffer_type<dram>
>

module {
  func.func @matmul(
      %arg0: tensor<32x64xbf16, #layout_tile>,
      %arg1: tensor<64x128xbf16, #layout_tile>)
      -> tensor<32x128xbf16, #layout_tile> {

    %device = "ttnn.get_device"() : () -> !ttnn.device

    // 输入可能需要 ToLayout（如果从 host 传入）
    %arg0_device = "ttnn.to_device"(%arg0, %device)
      : (tensor<32x64xbf16, #layout_tile>, !ttnn.device)
      -> tensor<32x64xbf16, #layout_tile>

    %arg1_device = "ttnn.to_device"(%arg1, %device)
      : (tensor<64x128xbf16, #layout_tile>, !ttnn.device)
      -> tensor<64x128xbf16, #layout_tile>

    // MatMul 操作
    %result = "ttnn.matmul"(%arg0_device, %arg1_device) <{
      transpose_a = false,
      transpose_b = false
    }> : (tensor<32x64xbf16, #layout_tile>,
          tensor<64x128xbf16, #layout_tile>)
       -> tensor<32x128xbf16, #layout_tile>

    return %result : tensor<32x128xbf16, #layout_tile>
  }
}
```

#### TTNN 输出（优化后，Optimizer + Validation 后）
```mlir
#layout_sharded = #ttnn.ttnn_layout<
  (d0, d1) -> (d0, d1),
  <32x128>,
  memref<1x1x!tt.tile<32x32, bf16>, !tt.tile<32x32, bf16>>,
  #ttnn.tensor_memory_layout<block_sharded>,
  #ttnn.buffer_type<l1>,
  #ttnn.shard_spec<
    #ttnn.core_range_set<[#ttnn.core_range<(0,0), (3,1)>]>,
    <8x32>,
    <row_major>,
    <physical>
  >
>

module {
  func.func @matmul(
      %arg0: tensor<32x64xbf16, #layout_tile>,
      %arg1: tensor<64x128xbf16, #layout_tile>)
      -> tensor<32x128xbf16, #layout_tile> {

    %device = "ttnn.get_device"() : () -> !ttnn.device

    // ... (to_device ops)

    // MatMul with sharding config
    %result = "ttnn.matmul"(%arg0_device, %arg1_device) <{
      transpose_a = false,
      transpose_b = false,
      matmul_program_config = #ttnn.matmul_multicore_reuse_mcast_config<
        compute_with_storage_grid_size = (4, 2),
        in0_block_w = 2,
        out_subblock_h = 1,
        out_subblock_w = 1,
        per_core_M = 1,
        per_core_N = 1,
        transpose_mcast = false,
        fuse_batch = true
      >,
      memory_config = #ttnn.memory_config<
        #l1,
        <block_sharded>,
        #ttnn.shard_spec<...>
      >
    }> : (tensor<32x64xbf16, #layout_tile>,
          tensor<64x128xbf16, #layout_tile>)
       -> tensor<32x128xbf16, #layout_sharded>

    // Convert back to interleaved for return
    %output = "ttnn.to_memory_config"(%result) <{
      memory_config = #ttnn.memory_config<#dram, <interleaved>>
    }> : (tensor<32x128xbf16, #layout_sharded>)
       -> tensor<32x128xbf16, #layout_tile>

    return %output : tensor<32x128xbf16, #layout_tile>
  }
}
```

### 完整示例 3: Conv2d

#### TTIR 输入
```mlir
module {
  func.func @conv2d(
      %input: tensor<1x3x224x224xbf16>,    // NCHW
      %weight: tensor<64x3x3x3xbf16>,      // OIHW
      %bias: tensor<64xbf16>)
      -> tensor<1x64x224x224xbf16> {

    %0 = ttir.empty() : tensor<1x64x224x224xbf16>
    %1 = ttir.conv2d %input, %weight, %bias, %0 {
      stride = [1, 1],
      padding = [1, 1],
      dilation = [1, 1],
      groups = 1
    } : tensor<1x3x224x224xbf16>,
        tensor<64x3x3x3xbf16>,
        tensor<64xbf16>,
        tensor<1x64x224x224xbf16>
      -> tensor<1x64x224x224xbf16>

    return %1 : tensor<1x64x224x224xbf16>
  }
}
```

#### TTNN 输出（经过 FlattenSlidingWindow 后）
```mlir
module {
  func.func @conv2d(
      %input: tensor<50176x27xbf16, #layout>,     // flattened
      %weight: tensor<64x3x3x3xbf16, #layout>,
      %bias: tensor<1x1x1x64xbf16, #layout>)
      -> tensor<50176x64xbf16, #layout> {

    %device = "ttnn.get_device"() : () -> !ttnn.device

    // 准备权重（可以 const-eval）
    %prepared_weight = "ttnn.prepare_conv2d_weights"(
        %weight, %device) <{
      in_channels = 3,
      out_channels = 64,
      batch_size = 1,
      input_height = 224,
      input_width = 224,
      kernel_size = [3, 3],
      stride = [1, 1],
      padding = [1, 1],
      groups = 1
    }> : (tensor<64x3x3x3xbf16, #layout>, !ttnn.device)
       -> tensor<?x?x?x?xbf16, #layout>

    // 准备 bias
    %prepared_bias = "ttnn.prepare_conv2d_bias"(
        %bias, %device) <{
      output_channels = 64
    }> : (tensor<1x1x1x64xbf16, #layout>, !ttnn.device)
       -> tensor<1x1x1x64xbf16, #layout>

    // Conv2d 操作
    %output = "ttnn.conv2d"(
        %input, %prepared_weight, %prepared_bias,
        %device) <{
      in_channels = 3,
      out_channels = 64,
      batch_size = 1,
      input_height = 224,
      input_width = 224,
      kernel_size = [3, 3],
      stride = [1, 1],
      padding = [1, 1],
      dilation = [1, 1],
      groups = 1,
      conv_config = #ttnn.optimized_conv_op_parallel_config<
        act_block_h_ntiles = 1,
        act_block_w_ntiles = 1,
        shard_layout = #ttnn.layout<row_major>,
        grid_size = (8, 7)
      >
    }> : (tensor<50176x27xbf16, #layout>,
          tensor<?x?x?x?xbf16, #layout>,
          tensor<1x1x1x64xbf16, #layout>,
          !ttnn.device)
       -> tensor<50176x64xbf16, #layout>

    return %output : tensor<50176x64xbf16, #layout>
  }
}
```

---

## 转换流程图

```
┌─────────────────────────────────────────────────────────────┐
│                        TTIR Module                          │
│  - 抽象操作（add, matmul, conv2d, etc.）                    │
│  - 无 layout 信息                                           │
│  - DPS style                                                │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
          ┌────────────────────────┐
          │  TTIR Preparation      │
          │  - Fusing              │
          │  - Decomposition       │
          │  - FlattenSlidingWindow│
          └────────┬───────────────┘
                   │
                   ▼
          ┌────────────────────────┐
          │   TTNNLayout Pass      │
          │  - 添加初始 layout     │
          │  - 默认: DRAM + Tile   │
          │  - 创建 TTNNLayoutAttr │
          └────────┬───────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────┐
│             ConvertTTIRToTTNN Pass                       │
│  - Pattern-based 转换                                    │
│  - 1-to-1 操作映射                                       │
│  - 移除 DPS output 参数                                  │
│  - 添加 device 操作                                      │
└────────────────────┬─────────────────────────────────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │  TTNN Workarounds    │
          │  - Layout fixes      │
          │  - Decomposition     │
          └──────────┬───────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│           TTNN Analysis & Optimization                  │
│                                                         │
│  ┌────────────────────────────────────────┐            │
│  │ RowMajorLayoutPropagation              │            │
│  │  - 从输入传播 RM layout                │            │
│  │  - 移除冗余 ToLayout                   │            │
│  └────────────┬───────────────────────────┘            │
│               │                                         │
│               ▼                                         │
│  ┌────────────────────────────────────────┐            │
│  │ TTNNOptimizer (需要 OpModel)           │            │
│  │  - Layout 优化                         │            │
│  │  - Memory config 优化                  │            │
│  │  - Sharding 决策                       │            │
│  │  - Program config 设置                 │            │
│  └────────────┬───────────────────────────┘            │
│               │                                         │
│               ▼                                         │
│  ┌────────────────────────────────────────┐            │
│  │ OperationValidationAndFallback         │            │
│  │  - 验证硬件约束                        │            │
│  │  - 应用 fallback 策略                  │            │
│  └────────────┬───────────────────────────┘            │
│               │                                         │
│               ▼                                         │
│  ┌────────────────────────────────────────┐            │
│  │ PrepareConv2dWeightsAndBias            │            │
│  │  - 预处理 Conv2d 权重                  │            │
│  │  - 允许 const-eval                     │            │
│  └────────────────────────────────────────┘            │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
          ┌──────────────────────┐
          │  DecomposeLayouts    │
          │  - ToLayout → 细粒度 ops│
          │  - ToDevice          │
          │  - ToMemoryConfig    │
          └──────────┬───────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │  TTNNDeallocate      │
          │  - 插入 deallocate   │
          └──────────┬───────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                      TTNN Module                            │
│  - 硬件特定操作                                             │
│  - 完整的 layout/memory 信息                                │
│  - Sharding configs                                         │
│  - 准备好降级到 Flatbuffer                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 关键设计决策

### 1. 为何使用 Tile Layout？

Tenstorrent 硬件的 Tensor Core 以 32x32 块操作：
- MatMul 和 Conv 在 Tile layout 上更高效
- Tile 数据在 L1 中已经对齐，减少重排开销
- 某些操作（如 transpose）在 Tile 上更快

### 2. 何时使用 Row-Major？

- 数据传输（Host ↔ Device, DRAM ↔ L1）
- Element-wise 操作（如果避免 layout 转换的开销更大）
- 整数张量（索引、mask 等）
- 某些 reduction 操作

### 3. L1 vs DRAM 权衡

**L1 优势**：
- 极快（~10x faster than DRAM）
- 支持 sharding（多核心并行）

**L1 限制**：
- 小容量（~1-2MB per core）
- 需要显式管理

**策略**：
- 小张量/激活 → L1
- 大权重 → DRAM
- 中间结果 → 根据使用模式决定

### 4. Sharding 何时有效？

**有效场景**：
- MatMul、Conv（计算密集，可并行）
- 足够大的张量（sharding 开销被计算收益抵消）
- 多次使用（分片后的张量被多个操作使用）

**无效场景**：
- 小张量（overhead 大于收益）
- Element-wise 操作（带宽受限）
- 需要频繁 gather/scatter 的操作

---

## 调试技巧

### 1. 查看 Layout 信息
```bash
ttmlir-opt --ttnn-layout input.mlir -o with-layout.mlir
```

### 2. 查看转换结果
```bash
ttmlir-opt --convert-ttir-to-ttnn input.mlir -o ttnn.mlir
```

### 3. 完整 Pipeline
```bash
ttmlir-opt --ttir-to-ttnn-backend-pipeline \
  --system-desc-path=/path/to/system_desc.ttsys \
  input.mlir -o output.mlir
```

### 4. 启用 Optimizer（需要 OpModel）
```bash
ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimizer-pass-enabled=true" \
  input.mlir -o optimized.mlir
```

### 5. 查看性能指标
```bash
ttmlir-opt --ttnn-collect-perf-metrics \
  --ttnn-perf-metrics-verbose-output-enabled \
  input.mlir
```

---

## 总结

TTIR 到 TTNN 的转换是 TT-MLIR 编译器中最关键的阶段：

1. **抽象到具体**：从硬件无关的 TTIR 到硬件特定的 TTNN
2. **多阶段优化**：layout、memory、sharding 的联合优化
3. **约束满足**：确保所有操作满足硬件约束
4. **性能驱动**：基于 OpModel 的性能模型指导决策
5. **可扩展性**：通过 pass 组合支持不同优化级别

这一层的质量直接决定了最终的执行性能，是编译器优化的核心。

---

## 相关文件

### 核心转换
- `/home/ubuntu/work/tt/tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp`
- `/home/ubuntu/work/tt/tt-mlir/include/ttmlir/Conversion/TTIRToTTNN/TTIRToTTNN.h`

### TTNN Transforms
- `/home/ubuntu/work/tt/tt-mlir/lib/Dialect/TTNN/Transforms/TTNNLayout.cpp`
- `/home/ubuntu/work/tt/tt-mlir/lib/Dialect/TTNN/Transforms/OptimizerPasses/RowMajorLayoutPropagation.cpp`
- `/home/ubuntu/work/tt/tt-mlir/lib/Dialect/TTNN/Transforms/OptimizerPasses/Optimizer.cpp`
- `/home/ubuntu/work/tt/tt-mlir/lib/Dialect/TTNN/Transforms/OptimizerPasses/OperationValidationAndFallback.cpp`

### Pipeline
- `/home/ubuntu/work/tt/tt-mlir/lib/Dialect/TTNN/Pipelines/TTNNPipelines.cpp`
- `/home/ubuntu/work/tt/tt-mlir/include/ttmlir/Dialect/TTNN/Transforms/Passes.td`

### 测试
- `/home/ubuntu/work/tt/tt-mlir/test/ttmlir/Conversion/TTIRToTTNN/`
- `/home/ubuntu/work/tt/tt-mlir/test/ttmlir/Dialect/TTNN/`
