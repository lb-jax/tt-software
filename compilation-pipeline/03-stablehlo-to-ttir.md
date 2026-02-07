# StableHLO 到 TTIR 转换详解

## 概述

StableHLO 到 TTIR 的转换是 TT-MLIR 编译器中最核心的环节之一。这个转换过程不仅涉及操作语义的映射，还包含了复杂的多设备并行化处理（Sharding）。本文档详细解释这个转换的完整流程。

## 转换架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    StableHLO + Shardy IR                        │
│  (ML框架生成，包含 sdy.mesh 和 sdy.sharding 注解)                │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │  StableHLO Pipeline (前处理)  │
         │  • Mesh 分析                 │
         │  • Sharding 传播             │
         │  • 集合操作插入               │
         └──────────┬──────────────────┘
                    │
                    ▼
         ┌─────────────────────────────┐
         │  StableHLO to TTIR Pipeline │
         │  • Op 转换                   │
         │  • Shardy 转换               │
         └──────────┬──────────────────┘
                    │
                    ▼
         ┌─────────────────────────────┐
         │       TTIR + Mesh Info      │
         │  (包含 mesh_shard ops)       │
         └─────────────────────────────┘
```

## 完整 Pipeline

### 阶段 1: StableHLO Pipeline（前处理）

在正式转换之前，需要运行 StableHLO Pipeline 处理并行化相关的任务。

**文件位置**: `/tt-mlir/lib/Dialect/StableHLO/Pipelines/StableHLOPipelines.cpp`

```cpp
void createStableHLOPipeline(OpPassManager &pm,
                             const StableHLOPipelineOptions &options) {
  // 1. 内联所有操作，便于分析
  pm.addPass(mlir::createInlinerPass());

  // 2. 填充参数类型信息（来自运行时）
  pm.addPass(mlir::tt::ttcore::createTTPopulateArgumentTypes());

  // 3. 转换 XLA SDY 注解为标准 SDY 注解
  pm.addPass(createConvertXlaSdyToSdyPass());

  // 4. 部分转换 SDY ops 为 StableHLO
  pm.addPass(createPartiallyConvertSdyToStableHLOPass());

  // 5. 应用参数分片状态注解
  pm.addPass(createApplyArgumentShardStatusPass());

  // 6. 分析 Mesh 并更新 sharding 注解
  pm.addPass(createAnalyzeMeshPass(analyzeMeshOptions));

  // 7. 解耦常量扇出
  pm.addPass(createDecoupleConstFanoutPass());

  // 8. 展平复合操作（便于 sharding 传播）
  pm.addPass(createFlattenCompositePass());

  // 9. 注册自定义 sharding 规则
  pm.addPass(createRegisterCustomShardingRulePass());

  // 10. 用户优先级的 sharding 传播
  pm.addPass(mlir::sdy::createUserPriorityPropagationPass());

  // 11. 插入显式 reshard 操作
  pm.addPass(mlir::sdy::createInsertExplicitReshardsPass());

  // 12. 将操作包裹在 manual_computation 中
  pm.addPass(createWrapUnderManualComputationPass());

  // 13. 将 reshard 转换为集合操作（all-gather, reduce-scatter等）
  pm.nest<func::FuncOp>().addPass(
      mlir::sdy::createReshardToCollectivesPass());

  // 14. 规范化 Shardy CCL 操作
  pm.addPass(createShardyCCLCanonicalizationPass());

  // 15. 更新全局形状到本地形状
  pm.addPass(createUpdateGlobalToLocalShapesPass());

  // 16. 重新轮廓化复合操作
  pm.addPass(createReoutlineCompositePass());

  // 17. 关闭 sharding（分析完成）
  pm.addPass(mlir::sdy::createCloseShardingsPass());

  // 18. 规范化
  pm.addPass(mlir::createCanonicalizerPass());
}
```

#### 关键 Pass 详解

**1. `createAnalyzeMeshPass` - Mesh 分析**

- 作用：分析目标设备的网格拓扑，验证或生成设备网格定义
- 输入：`meshShape` 参数（如 `[1, 8]` 表示 1x8 的设备网格）
- 输出：验证/更新 `sdy.mesh` 定义

**2. `createUserPriorityPropagationPass` - Sharding 传播**

- 作用：根据用户标注的 sharding 规则，向整个计算图传播 sharding 信息
- 策略：
  - 从已标注的操作开始
  - 根据操作语义传播（如 matmul 的维度对应关系）
  - 处理冲突（用户优先级）

**3. `createInsertExplicitReshardsPass` - 插入 Reshard**

- 作用：在 sharding 变化的地方插入显式的 reshard 操作
- 示例：如果前一个操作输出 `[{"x"}, {}]` 而后一个操作需要 `[{}, {"y"}]`，插入 reshard

**4. `createWrapUnderManualComputationPass` - 包裹 Manual Computation**

- 作用：将需要在多设备上并行执行的操作包裹在 `sdy.manual_computation` 中
- 重要性：这是 Shardy 到 TTIR 转换的关键结构

**5. `createReshardToCollectivesPass` - Reshard 转集合操作**

- 作用：将抽象的 reshard 操作转换为具体的集合通信操作
- 转换映射：
  - 全局聚合 → `stablehlo.all_gather`
  - 聚合规约 → `stablehlo.reduce_scatter`
  - 全规约 → `stablehlo.all_reduce`

**6. `createUpdateGlobalToLocalShapesPass` - 全局到本地形状**

- 作用：根据 sharding 信息，将全局张量形状转换为每个设备上的本地形状
- 示例：
  - 全局：`tensor<784x128xf32>` with sharding `[{"y"}, {}]` on 8 devices
  - 本地：`tensor<98x128xf32>` (784/8 = 98)

### 阶段 2: StableHLO to TTIR Pipeline（正式转换）

**文件位置**: `/tt-mlir/lib/Dialect/TTIR/Pipelines/TTIRPipelines.cpp`

```cpp
void createStableHLOToTTIRPipeline(
    OpPassManager &pm, const StableHLOToTTIRPipelineOptions &options) {

  // 1. （可选）Arith dialect 转换
  if (options.arithDialectConversionsEnabled) {
    pm.addPass(createConvertArithToStableHLOPass());
  }

  // 2. 合法化 StableHLO composite 为 TTIR
  pm.addPass(createLegalizeStableHLOCompositeToTTIRPass());

  // 3. （可选）合法化 composite 为函数调用
  if (options.legalizeCompositeToCallEnabled) {
    pm.addPass(stablehlo::createStablehloLegalizeCompositeToCallPass());
  }

  // 4. 内联
  pm.addPass(mlir::createInlinerPass());

  // 5. （可选）激进的简化
  if (options.enableAggressiveSimplification) {
    pm.addPass(stablehlo::createStablehloAggressiveSimplificationPass());
  }

  // 6. 核心转换：StableHLO → TTIR
  ttir::ConvertStableHLOToTTIROptions passOptions;
  passOptions.enablePartialConversion = options.enableCPUFallback;
  pm.addPass(createConvertStableHLOToTTIRPass(passOptions));

  // 7. CSE 优化
  pm.addPass(mlir::createCSEPass());

  // 8. （可选）CPU 回退处理
  if (options.enableCPUFallback) {
    pm.addPass(ttcore::createTTCoreWrapDeviceModulePass());
    // ... CPU 相关的 pass
  }
}
```

## 核心转换 Pass: ConvertStableHLOToTTIR

**文件位置**: `/tt-mlir/lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPass.cpp`

### Pass 实现

```cpp
struct ConvertStableHLOToTTIRPass :
    public ttir::impl::ConvertStableHLOToTTIRBase<ConvertStableHLOToTTIRPass> {

  void runOnOperation() final {
    mlir::ConversionTarget target(getContext());

    // 设置合法/非法的 dialect
    target.addLegalDialect<mlir::quant::QuantDialect>();
    target.addLegalDialect<ttir::TTIRDialect>();
    target.addLegalDialect<ttcore::TTCoreDialect>();
    target.addIllegalDialect<mlir::sdy::SdyDialect>();  // Shardy 必须转换
    target.addIllegalOp<mlir::tensor::EmptyOp>();

    if (!enablePartialConversion) {
      target.addIllegalDialect<mlir::stablehlo::StablehloDialect>();
    }

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });

    RewritePatternSet patterns(&getContext());

    // 添加转换模式
    addEmptyOpTypeConversionPattern(&getContext(), patterns, typeConverter);
    populateStableHLOToTTIRPatterns(&getContext(), patterns, typeConverter);
    populateShardyToTTIRPatterns(&getContext(), patterns, typeConverter);

    // 函数类型转换
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    populateCallOpTypeConversionPattern(patterns, typeConverter);

    // 应用转换
    if (enablePartialConversion) {
      applyPartialConversion(getOperation(), target, std::move(patterns));
    } else {
      applyFullConversion(getOperation(), target, std::move(patterns));
    }
  }
};
```

## 转换模式（Conversion Patterns）

### 1. 基础操作转换

**文件位置**: `/tt-mlir/lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`

#### 默认转换模式（直接映射）

对于语义完全一致的操作，使用默认转换模式：

```cpp
template <typename SrcOp, typename DestOp>
class StableHLOToTTIROpDefaultConversionPattern :
    public OpConversionPattern<SrcOp> {
public:
  LogicalResult matchAndRewrite(
      SrcOp srcOp, Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto outputType = mlir::cast<RankedTensorType>(
        this->getTypeConverter()->convertType(srcOp.getResult().getType()));

    rewriter.replaceOpWithNewOp<DestOp>(
        srcOp, outputType, adaptor.getOperands());

    return success();
  }
};
```

**应用的操作对**：
- `stablehlo.add` → `ttir.add`
- `stablehlo.multiply` → `ttir.multiply`
- `stablehlo.subtract` → `ttir.subtract`
- `stablehlo.divide` → `ttir.divide`
- `stablehlo.exp` → `ttir.exp`
- `stablehlo.log` → `ttir.log`
- `stablehlo.sqrt` → `ttir.sqrt`
- `stablehlo.rsqrt` → `ttir.rsqrt`
- `stablehlo.abs` → `ttir.abs`
- `stablehlo.neg` → `ttir.neg`
- `stablehlo.sine` → `ttir.sin`
- `stablehlo.cosine` → `ttir.cos`
- 等等...

#### Transpose 转换（需要语义适配）

```cpp
class StableHLOToTTIRTransposeOpConversionPattern :
    public OpConversionPattern<mlir::stablehlo::TransposeOp> {
public:
  LogicalResult matchAndRewrite(
      mlir::stablehlo::TransposeOp srcOp,
      mlir::stablehlo::TransposeOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    RankedTensorType outputType = mlir::cast<RankedTensorType>(
        this->getTypeConverter()->convertType(srcOp.getResult().getType()));

    // StableHLO 的 transpose 对应 TTIR 的 permute
    rewriter.replaceOpWithNewOp<mlir::tt::ttir::PermuteOp>(
        srcOp, outputType, adaptor.getOperand(), adaptor.getPermutation());

    return success();
  }
};
```

**转换示例**：

```mlir
// StableHLO
%0 = stablehlo.transpose %arg0, dims = [1, 0] :
     (tensor<2x3xf32>) -> tensor<3x2xf32>

// TTIR
%0 = ttir.permute %arg0 {permutation = array<i64: 1, 0>} :
     tensor<2x3xf32> -> tensor<3x2xf32>
```

#### DotGeneral 转换（复杂映射）

```cpp
class StableHLOToTTIRDotGeneralOpConversionPattern :
    public OpConversionPattern<mlir::stablehlo::DotGeneralOp> {
public:
  LogicalResult matchAndRewrite(
      mlir::stablehlo::DotGeneralOp srcOp,
      mlir::stablehlo::DotGeneralOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResult().getType()));

    rewriter.replaceOpWithNewOp<mlir::tt::ttir::DotGeneralOp>(
        srcOp, outputType,
        adaptor.getLhs(), adaptor.getRhs(),
        adaptor.getDotDimensionNumbers().getLhsBatchingDimensions(),
        adaptor.getDotDimensionNumbers().getLhsContractingDimensions(),
        adaptor.getDotDimensionNumbers().getRhsBatchingDimensions(),
        adaptor.getDotDimensionNumbers().getRhsContractingDimensions());

    return success();
  }
};
```

### 2. Reduce 操作转换（模式匹配）

Reduce 操作需要识别 reducer 函数的模式：

```cpp
class StableHLOToTTIRReduceOpConversionPattern :
    public OpConversionPattern<mlir::stablehlo::ReduceOp> {
public:
  LogicalResult matchAndRewrite(
      mlir::stablehlo::ReduceOp srcOp,
      mlir::stablehlo::ReduceOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    const mlir::Operation &innerOp = srcOp.getBody().front().front();

    // 根据 reducer 内部操作类型选择对应的 TTIR op
    if (mlir::isa<mlir::stablehlo::AddOp>(innerOp)) {
      return matchAndRewriteInternal<mlir::tt::ttir::SumOp>(
          srcOp, adaptor, rewriter);
    }
    if (mlir::isa<mlir::stablehlo::MaxOp>(innerOp)) {
      return matchAndRewriteInternal<mlir::tt::ttir::MaxOp>(
          srcOp, adaptor, rewriter);
    }
    if (mlir::isa<mlir::stablehlo::MinOp>(innerOp)) {
      return matchAndRewriteInternal<mlir::tt::ttir::MinOp>(
          srcOp, adaptor, rewriter);
    }
    if (mlir::isa<mlir::stablehlo::MulOp>(innerOp)) {
      return matchAndRewriteInternal<mlir::tt::ttir::ProdOp>(
          srcOp, adaptor, rewriter);
    }

    return failure();
  }
};
```

**转换示例**：

```mlir
// StableHLO (Sum Reduce)
%init = stablehlo.constant dense<0.0> : tensor<f32>
%sum = stablehlo.reduce(%input init: %init) applies stablehlo.add
       across dimensions = [1] : (tensor<32x128xf32>, tensor<f32>) -> tensor<32xf32>

// TTIR
%sum = ttir.sum %input {keep_dim = false, dim = 1} :
       tensor<32x128xf32> -> tensor<32xf32>
```

### 3. BatchNorm 转换（复杂操作）

```cpp
class StableHLOToBatchNormInferenceOpConversionPattern :
    public OpConversionPattern<mlir::stablehlo::BatchNormInferenceOp> {
public:
  LogicalResult matchAndRewrite(
      mlir::stablehlo::BatchNormInferenceOp srcOp,
      mlir::stablehlo::BatchNormInferenceOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    // 验证合法性
    LogicalResult legalityResult =
        checkBatchNormConversionLegality(srcOp, adaptor, rewriter);
    if (!legalityResult.succeeded()) {
      return legalityResult;
    }

    auto outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResult().getType()));

    // 转换 feature_index 为 dimension 属性
    IntegerAttr dimensionAttr =
        mlir::IntegerAttr::get(integerType, srcOp.getFeatureIndex());

    rewriter.replaceOpWithNewOp<mlir::tt::ttir::BatchNormInferenceOp>(
        srcOp, outputType,
        adaptor.getOperand(), adaptor.getScale(), adaptor.getOffset(),
        adaptor.getMean(), adaptor.getVariance(),
        adaptor.getEpsilonAttr(), dimensionAttr);

    return success();
  }
};
```

### 4. 所有支持的转换

`populateStableHLOToTTIRPatterns` 函数注册了以下类别的转换模式：

```cpp
void populateStableHLOToTTIRPatterns(MLIRContext *ctx,
                                     RewritePatternSet &patterns,
                                     TypeConverter &typeConverter) {
  // 元素级一元操作
  addElementwiseUnaryOpsConversionPatterns(ctx, patterns, typeConverter);

  // 元素级二元操作
  addElementwiseBinaryOpsConversionPatterns(ctx, patterns, typeConverter);

  // 量化操作
  addQuantizeOpsConversionPattern(ctx, patterns, typeConverter);

  // Reduce 操作
  addReduceOpsConversionPatterns(ctx, patterns, typeConverter);

  // DotGeneral（矩阵乘法）
  addDotGeneralOpConversionPatterns(ctx, patterns, typeConverter);

  // 获取维度大小
  addGetDimensionSizeOpsConversionPatterns(ctx, patterns, typeConverter);

  // 张量创建操作
  addTensorCreationOpsConversionPatterns(ctx, patterns, typeConverter);

  // 广播操作
  addBroadcastOpConversionPattern(ctx, patterns, typeConverter);

  // 卷积操作
  addConv2dOpConversionPattern(ctx, patterns, typeConverter);

  // Reduce Window（池化等）
  addReduceWindowOpConversionPattern(ctx, patterns, typeConverter);

  // 比较操作
  addCompareOpsConversionPatterns(ctx, patterns, typeConverter);

  // 拼接操作
  addConcatOpsConversionPatterns(ctx, patterns, typeConverter);

  // Transpose
  addTransposeOpConversionPattern(ctx, patterns, typeConverter);

  // Reshape
  addReshapeOpConversionPattern(ctx, patterns, typeConverter);

  // 集合通信操作（CCL）
  addCCLOpsConversionPattern(ctx, patterns, typeConverter);

  // 逻辑和位操作
  addLogicalAndBitwiseOpsConversionPatterns(ctx, patterns, typeConverter);

  // Slice 操作
  addSliceOpConversionPattern(ctx, patterns, typeConverter);
  addDynamicSliceOpConversionPattern(ctx, patterns, typeConverter);

  // Clamp 操作
  addClampOpConversionPattern(ctx, patterns, typeConverter);

  // Gather/Scatter 操作
  addGatherOpConversionPattern(ctx, patterns, typeConverter);
  addScatterOpConversionPatterns(ctx, patterns, typeConverter);

  // Iota 操作
  addIotaOpConversionPattern(ctx, patterns, typeConverter);

  // Reverse 操作
  addReverseOpConversionPattern(ctx, patterns, typeConverter);

  // Pad 操作
  addPadOpConversionPattern(ctx, patterns, typeConverter);

  // SelectAndScatter
  addSelectAndScatterOpConversionPatterns(ctx, patterns, typeConverter);

  // BatchNorm 操作
  addBatchNormOpConversionPattern(ctx, patterns, typeConverter);

  // 随机数生成
  addRngOpConversionPattern(ctx, patterns, typeConverter);
  addRngBitGeneratorOpConversionPattern(ctx, patterns, typeConverter);

  // Erf 函数
  addErfOpConversionPattern(ctx, patterns, typeConverter);

  // 排序操作
  addSortOpConversionPattern(ctx, patterns, typeConverter);

  // 缓存操作（用于 KV Cache）
  addCacheOpsConversionPattern(ctx, patterns, typeConverter);

  // 优化屏障
  addOptimizationBarrierOpConversionPattern(ctx, patterns, typeConverter);

  // 注意力操作
  addScaledDotProductAttentionDecodeOpConversionPattern(
      ctx, patterns, typeConverter);
}
```

## Shardy 到 TTIR 转换

### Manual Computation 转换

**文件位置**: `/tt-mlir/lib/Conversion/StableHLOToTTIR/ShardyToTTIRPatterns.cpp`

这是多设备并行化的核心转换：

```cpp
class ShardyToTTIRManualComputationOpConversionPattern :
    public mlir::OpConversionPattern<mlir::sdy::ManualComputationOp> {
public:
  llvm::LogicalResult matchAndRewrite(
      mlir::sdy::ManualComputationOp srcOp,
      mlir::sdy::ManualComputationOp::Adaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {

    mlir::Location loc = srcOp.getLoc();

    // 1. 分析 shard 状态缓存
    ManualComputationAnalysisCache cache =
        ManualComputationAnalysisCache::generate(srcOp);

    // 2. 获取 mesh 定义
    llvm::SmallVector<mlir::sdy::MeshOp> parsedMeshOps =
        shardy_utils::getMeshOps(module);

    // 3. 为输入创建 mesh_shard ops（Full → Shard）
    rewriter.setInsertionPoint(srcOp);
    llvm::SmallVector<mlir::Value> fullToShardResults;

    for (auto [globalOperand, argSharding, localArgType] :
         llvm::zip_equal(adaptor.getOperands(),
                        srcOp.getInShardings().getShardings(),
                        srcOp.getBody().getArgumentTypes())) {

      // 解析 sharding 信息
      llvm::Expected<shardy_utils::ShardyMeshSharding> shardyMeshSharding =
          shardy_utils::ShardyMeshSharding::generate(
              parsedMeshOps[0].getMeshAttr(), argSharding,
              cache.getShardStatus(globalOperand),
              ttcore::MeshShardDirection::FullToShard);

      // 创建 mesh_shard op
      auto meshShardOp = rewriter.create<ttir::MeshShardOp>(
          loc, outputType, globalOperand,
          shardyMeshSharding->getShardType(),
          shardyMeshSharding->getShardDirection(),
          shardyMeshSharding->getShardShape(),
          shardyMeshSharding->getShardDims());

      fullToShardResults.push_back(meshShardOp.getResult());
    }

    // 4. 为输出创建 mesh_shard ops（Shard → Full）
    rewriter.setInsertionPointAfter(srcOp);
    llvm::SmallVector<mlir::Value> shardToFullResults;

    mlir::Operation *sdyReturn = mlir::sdy::getBodyTerminator(srcOp);
    for (auto [returnOperand, outSharding, opResult] :
         llvm::zip_equal(sdyReturn->getOpOperands(),
                        srcOp.getOutShardings().getShardings(),
                        srcOp.getResults())) {

      // 创建 Shard → Full 的 mesh_shard op
      auto meshShardOp = rewriter.create<ttir::MeshShardOp>(
          loc, outputType, returnOperand.get(),
          ttcore::MeshShardType::Identity,  // 输出使用 Identity
          shardyMeshSharding->getShardDirection(),
          shardyMeshSharding->getShardShape(),
          shardyMeshSharding->getShardDims());

      shardToFullResults.push_back(meshShardOp.getResult());
    }

    // 5. 内联 manual_computation 的内部操作
    rewriter.inlineBlockBefore(&srcOp.getBody().front(), srcOp,
                               fullToShardResults);
    rewriter.eraseOp(sdyReturn);
    rewriter.replaceOp(srcOp, shardToFullResults);

    return llvm::success();
  }
};
```

### Mesh 定义转换

```cpp
class ShardyToTTIRMeshOpConversionPattern :
    public mlir::OpConversionPattern<mlir::sdy::MeshOp> {
public:
  llvm::LogicalResult matchAndRewrite(
      mlir::sdy::MeshOp srcOp,
      mlir::sdy::MeshOp::Adaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {

    // 创建 TTIR mesh 属性
    mlir::tt::ttcore::MeshAttr ttMeshAttr =
        shardy_utils::createTTMeshAttrFromSdyMeshOp(srcOp);

    // 添加到 module 的 meshes 属性中
    mlir::ModuleOp module = srcOp->getParentOfType<mlir::ModuleOp>();
    llvm::SmallVector<mlir::tt::ttcore::MeshAttr> meshes;

    if (auto meshesAttr = module->getAttrOfType<ttcore::MeshesAttr>(
            ttcore::MeshesAttr::name)) {
      meshes = llvm::SmallVector<ttcore::MeshAttr>(meshesAttr.getMeshes());
    }

    // 避免重复添加
    if (llvm::all_of(meshes, [&](ttcore::MeshAttr m) {
          return m.getName() != ttMeshAttr.getName();
        })) {
      meshes.push_back(ttcore::MeshAttr::get(
          getContext(), ttMeshAttr.getName(), ttMeshAttr.getShape()));

      rewriter.modifyOpInPlace(module, [&]() {
        module->setAttr(ttcore::MeshesAttr::name,
                       ttcore::MeshesAttr::get(getContext(), meshes));
      });
    }

    rewriter.eraseOp(srcOp);
    return llvm::success();
  }
};
```

### Shardy 转换注册

```cpp
void populateShardyToTTIRPatterns(MLIRContext *ctx,
                                  RewritePatternSet &patterns,
                                  TypeConverter &typeConverter) {
  patterns.add<ShardyToTTIRManualComputationOpConversionPattern>(
      typeConverter, ctx);
  patterns.add<ShardyToTTIRMeshOpConversionPattern>(
      typeConverter, ctx);
}
```

## 完整转换示例

### 示例 1: 简单元素级操作

**输入 (StableHLO)**:
```mlir
module @jit_eltwise_add {
  func.func public @test_add(%arg0: tensor<64x128xf32>,
                              %arg1: tensor<64x128xf32>)
                              -> tensor<64x128xf32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<64x128xf32>
    return %0 : tensor<64x128xf32>
  }
}
```

**输出 (TTIR)**:
```mlir
module @jit_eltwise_add {
  func.func public @test_add(%arg0: tensor<64x128xf32>,
                              %arg1: tensor<64x128xf32>)
                              -> tensor<64x128xf32> {
    %0 = ttir.add %arg0, %arg1 : tensor<64x128xf32>
    return %0 : tensor<64x128xf32>
  }
}
```

### 示例 2: Transpose

**输入 (StableHLO)**:
```mlir
module @jit_tensor_transpose {
  func.func public @test_transpose(%arg0: tensor<2x3xf32>)
                                   -> tensor<3x2xf32> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] :
         (tensor<2x3xf32>) -> tensor<3x2xf32>
    return %0 : tensor<3x2xf32>
  }
}
```

**输出 (TTIR)**:
```mlir
module @jit_tensor_transpose {
  func.func public @test_transpose(%arg0: tensor<2x3xf32>)
                                   -> tensor<3x2xf32> {
    %0 = ttir.permute %arg0 {permutation = array<i64: 1, 0>} :
         tensor<2x3xf32> -> tensor<3x2xf32>
    return %0 : tensor<3x2xf32>
  }
}
```

### 示例 3: 多设备并行（Tensor Parallelism）

**输入 (StableHLO + Shardy)**:
```mlir
module @jit_loss_tp attributes {
  mhlo.num_partitions = 8 : i32,
  mhlo.num_replicas = 1 : i32
} {
  // 定义设备网格：1x8
  sdy.mesh @mesh = <["x"=1, "y"=8]>

  func.func public @main(
      // 输入数据：在 y 维度分片
      %arg0: tensor<32x784xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>},
      // 权重：在第 0 维度分片（按 y 维度）
      %arg1: tensor<784x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>},
      // 偏置：在 y 维度分片
      %arg2: tensor<128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}
  ) -> tensor<32x128xf32> {

    // Manual Computation 区域：多设备并行执行
    %0 = sdy.manual_computation(%arg0, %arg1, %arg2)
         in_shardings=[
           <@mesh, [{}, {"y"}]>,   // 输入 0 的 sharding
           <@mesh, [{"y"}, {}]>,   // 输入 1 的 sharding
           <@mesh, [{"y"}]>        // 输入 2 的 sharding
         ]
         out_shardings=[<@mesh, [{}, {"y"}]>]  // 输出的 sharding
         manual_axes={"x", "y"}                // 手动管理的轴
         // 内部操作使用本地形状
         (%local_input: tensor<32x98xf32>,      // 784/8 = 98
          %local_weight: tensor<98x128xf32>,
          %local_bias: tensor<16xf32>) {         // 128/8 = 16

      // 本地矩阵乘法
      %mm = stablehlo.dot_general %local_input, %local_weight,
            contracting_dims = [1] x [0] :
            (tensor<32x98xf32>, tensor<98x128xf32>) -> tensor<32x128xf32>

      // Reduce-Scatter：聚合并分发结果
      %rs = "stablehlo.reduce_scatter"(%mm) <{
        channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>,
        replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>,
        scatter_dimension = 1 : i64,
        use_global_device_ids
      }> ({
      ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
        %sum = stablehlo.add %lhs, %rhs : tensor<f32>
        stablehlo.return %sum : tensor<f32>
      }) : (tensor<32x128xf32>) -> tensor<32x16xf32>

      // 添加偏置（广播）
      %bias_bcast = stablehlo.broadcast_in_dim %local_bias,
                    dims = [1] : (tensor<16xf32>) -> tensor<32x16xf32>
      %result = stablehlo.add %rs, %bias_bcast : tensor<32x16xf32>

      sdy.return %result : tensor<32x16xf32>
    } : (tensor<32x784xf32>, tensor<784x128xf32>, tensor<128xf32>)
        -> tensor<32x128xf32>

    return %0 : tensor<32x128xf32>
  }
}
```

**输出 (TTIR + Mesh Info)**:
```mlir
module @jit_loss_tp attributes {
  // Mesh 定义已转换为 TTIR 属性
  ttcore.meshes = #ttcore.meshes<[
    #ttcore.mesh<"mesh", shape=[1, 8]>
  ]>
} {
  func.func public @main(
      %arg0: tensor<32x784xf32> {
        ttcore.runtime_tensor_sharding = #ttcore.runtime_tensor_sharding<
          shard_status = #ttcore.shard_status<sharded>
        >
      },
      %arg1: tensor<784x128xf32> {
        ttcore.runtime_tensor_sharding = #ttcore.runtime_tensor_sharding<
          shard_status = #ttcore.shard_status<sharded>
        >
      },
      %arg2: tensor<128xf32> {
        ttcore.runtime_tensor_sharding = #ttcore.runtime_tensor_sharding<
          shard_status = #ttcore.shard_status<sharded>
        >
      }
  ) -> tensor<32x128xf32> {

    // Mesh Shard: Full → Shard (输入 0)
    %input_local = "ttir.mesh_shard"(%arg0) {
      shard_type = #ttcore.shard_type<identity>,
      shard_direction = #ttcore.shard_direction<full_to_shard>,
      shard_shape = array<i64: 1, 8>,
      shard_dims = array<i64: -1, 1>
    } : (tensor<32x784xf32>) -> tensor<32x98xf32>

    // Mesh Shard: Full → Shard (权重)
    %weight_local = "ttir.mesh_shard"(%arg1) {
      shard_type = #ttcore.shard_type<identity>,
      shard_direction = #ttcore.shard_direction<full_to_shard>,
      shard_shape = array<i64: 8, 1>,
      shard_dims = array<i64: -1, 0>
    } : (tensor<784x128xf32>) -> tensor<98x128xf32>

    // Mesh Shard: Full → Shard (偏置)
    %bias_local = "ttir.mesh_shard"(%arg2) {
      shard_type = #ttcore.shard_type<identity>,
      shard_direction = #ttcore.shard_direction<full_to_shard>,
      shard_shape = array<i64: 8>,
      shard_dims = array<i64: -1, 0>
    } : (tensor<128xf32>) -> tensor<16xf32>

    // 本地矩阵乘法（已转换为 TTIR）
    %mm = ttir.dot_general %input_local, %weight_local {
      lhs_batching_dimensions = array<i64>,
      lhs_contracting_dimensions = array<i64: 1>,
      rhs_batching_dimensions = array<i64>,
      rhs_contracting_dimensions = array<i64: 0>
    } : tensor<32x98xf32>, tensor<98x128xf32> -> tensor<32x128xf32>

    // Reduce-Scatter（已转换为 TTIR CCL op）
    %rs = "ttir.all_reduce"(%mm) {
      op_type = #ttcore.reduce_type<sum>,
      scatter_dimension = 1 : i64,
      replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>
    } : (tensor<32x128xf32>) -> tensor<32x16xf32>

    // 广播（已转换为 TTIR）
    %bias_bcast = ttir.broadcast %bias_local {
      broadcast_dimensions = array<i64: 1>
    } : tensor<16xf32> -> tensor<32x16xf32>

    // 加法
    %result_local = ttir.add %rs, %bias_bcast : tensor<32x16xf32>

    // Mesh Shard: Shard → Full (输出)
    %result = "ttir.mesh_shard"(%result_local) {
      shard_type = #ttcore.shard_type<identity>,
      shard_direction = #ttcore.shard_direction<shard_to_full>,
      shard_shape = array<i64: 1, 8>,
      shard_dims = array<i64: -1, 1>
    } : (tensor<32x16xf32>) -> tensor<32x128xf32>

    return %result : tensor<32x128xf32>
  }
}
```

## Sharding 详解

### Mesh 定义

Mesh 定义了设备的拓扑结构：

```mlir
// Shardy 格式
sdy.mesh @mesh = <["x"=1, "y"=8]>

// TTIR 格式（转换后）
ttcore.meshes = #ttcore.meshes<[
  #ttcore.mesh<"mesh", shape=[1, 8]>
]>
```

**解释**：
- `"x"=1`: x 轴有 1 个设备
- `"y"=8`: y 轴有 8 个设备
- 总共 1 × 8 = 8 个设备

### Sharding 注解

Sharding 注解描述张量如何在设备间分片：

```mlir
// 在 y 维度分片第 0 个张量维度
tensor<784x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}

// 含义：
// - 第 0 维（784）沿 mesh 的 y 轴（8 个设备）分片
// - 每个设备获得 784/8 = 98
// - 第 1 维（128）不分片
// - 本地形状：tensor<98x128xf32>
```

### Shard Status

描述参数是否已经分片：

```cpp
enum class ShardStatus {
  Unsharded,  // 未分片（需要在运行时分片）
  Sharded     // 已分片（输入已经是分片后的数据）
};
```

### MeshShardOp

`ttir.mesh_shard` 操作负责分片和聚合：

```mlir
"ttir.mesh_shard"(%tensor) {
  shard_type = #ttcore.shard_type<identity>,
  shard_direction = #ttcore.shard_direction<full_to_shard>,
  shard_shape = array<i64: 8, 1>,
  shard_dims = array<i64: -1, 0>
} : (tensor<784x128xf32>) -> tensor<98x128xf32>
```

**参数解释**：
- `shard_type`:
  - `identity`: 直接分片/聚合
  - `devices`: 跨设备分片
- `shard_direction`:
  - `full_to_shard`: 全局张量 → 本地张量
  - `shard_to_full`: 本地张量 → 全局张量
- `shard_shape`: 每个维度的分片数量
- `shard_dims`: 哪些维度需要分片（-1 表示不分片）

### 集合通信操作

Sharding 需要集合通信来同步数据：

**1. All-Gather（全聚合）**
```mlir
// 每个设备有部分数据，聚合成完整数据
%full = "ttir.all_gather"(%partial) {
  gather_dimension = 0
} : (tensor<98x128xf32>) -> tensor<784x128xf32>
```

**2. Reduce-Scatter（聚合分发）**
```mlir
// 所有设备规约结果，然后分发
%partial = "ttir.reduce_scatter"(%full) {
  op_type = #ttcore.reduce_type<sum>,
  scatter_dimension = 1
} : (tensor<32x128xf32>) -> tensor<32x16xf32>
```

**3. All-Reduce（全规约）**
```mlir
// 所有设备规约，每个设备都得到完整结果
%reduced = "ttir.all_reduce"(%input) {
  op_type = #ttcore.reduce_type<sum>
} : (tensor<32x128xf32>) -> tensor<32x128xf32>
```

## 流程图

### 单设备转换流程

```
┌──────────────────────┐
│  StableHLO IR        │
│  stablehlo.add       │
│  stablehlo.matmul    │
└──────────┬───────────┘
           │
           ▼
  ┌────────────────────┐
  │ Pattern Matching   │
  │ • 查找对应的 Pattern│
  │ • 验证操作合法性    │
  └────────┬───────────┘
           │
           ▼
  ┌────────────────────┐
  │ Type Conversion    │
  │ • 转换类型         │
  │ • 转换属性         │
  └────────┬───────────┘
           │
           ▼
  ┌────────────────────┐
  │ Create TTIR Op     │
  │ • ttir.add         │
  │ • ttir.matmul      │
  └────────┬───────────┘
           │
           ▼
  ┌────────────────────┐
  │  Replace Original  │
  └────────────────────┘
```

### 多设备转换流程

```
┌────────────────────────────────────────────────────┐
│  StableHLO + Shardy IR                             │
│  sdy.mesh @mesh = <["x"=1, "y"=8]>                │
│  sdy.manual_computation(...) {                     │
│    stablehlo.dot_general                           │
│    stablehlo.reduce_scatter                        │
│  }                                                 │
└─────────────────────┬──────────────────────────────┘
                      │
                      ▼
         ┌────────────────────────────┐
         │  1. Analyze Mesh           │
         │  • 验证设备网格             │
         │  • 提取 sharding 信息       │
         └────────────┬───────────────┘
                      │
                      ▼
         ┌────────────────────────────┐
         │  2. Create Input MeshShard │
         │  Full → Shard              │
         │  • tensor<784x128> →       │
         │    tensor<98x128>          │
         └────────────┬───────────────┘
                      │
                      ▼
         ┌────────────────────────────┐
         │  3. Convert Body Ops       │
         │  • stablehlo.dot_general → │
         │    ttir.dot_general        │
         │  • stablehlo.reduce_scatter│
         │    → ttir.all_reduce       │
         └────────────┬───────────────┘
                      │
                      ▼
         ┌────────────────────────────┐
         │  4. Create Output MeshShard│
         │  Shard → Full              │
         │  • tensor<32x16> →         │
         │    tensor<32x128>          │
         └────────────┬───────────────┘
                      │
                      ▼
         ┌────────────────────────────┐
         │  5. Inline Body            │
         │  移除 manual_computation    │
         └────────────┬───────────────┘
                      │
                      ▼
         ┌────────────────────────────┐
         │  6. Update Module Mesh     │
         │  添加 ttcore.meshes 属性    │
         └────────────────────────────┘
```

## 关键数据结构

### ShardyMeshSharding

封装了 sharding 的所有信息：

```cpp
class ShardyMeshSharding {
public:
  // 从 Shardy 注解生成
  static llvm::Expected<ShardyMeshSharding> generate(
      mlir::sdy::MeshAttr meshAttr,
      mlir::sdy::TensorShardingAttr sharding,
      ttcore::ShardStatus shardStatus,
      ttcore::MeshShardDirection direction);

  // Getters
  ttcore::MeshShardType getShardType() const;
  ttcore::MeshShardDirection getShardDirection() const;
  llvm::ArrayRef<int64_t> getShardShape() const;
  llvm::ArrayRef<int64_t> getShardDims() const;
};
```

### ManualComputationAnalysisCache

缓存 manual computation 的分片状态分析：

```cpp
class ManualComputationAnalysisCache {
public:
  static ManualComputationAnalysisCache generate(
      mlir::sdy::ManualComputationOp &op);

  ttcore::ShardStatus getShardStatus(mlir::Value arg) const;

private:
  llvm::DenseMap<mlir::Value, ttcore::ShardStatus> shardStatusCache;
};
```

## 调试和验证

### 运行转换

```bash
# 运行完整 pipeline
ttmlir-opt --stablehlo-pipeline \
           --stablehlo-to-ttir-pipeline \
           input.mlir -o output.mlir

# 只运行 StableHLO 到 TTIR 的核心转换
ttmlir-opt --convert-stablehlo-to-ttir \
           input.mlir -o output.mlir

# 查看中间结果
ttmlir-opt --stablehlo-pipeline \
           --mlir-print-ir-after-all \
           input.mlir 2>&1 | less
```

### 测试文件位置

- 基础操作: `/tt-mlir/test/python/golden/mlir_snippets/stablehlo/`
- Shardy 相关: `/tt-mlir/test/ttmlir/Conversion/StableHLOToTTIR/ccl/`
- 端到端测试: `/tt-mlir/test/ttmlir/Silicon/StableHLO/`

### 常见问题排查

**1. Conversion Pattern 不匹配**

```
error: failed to legalize operation 'stablehlo.custom_call'
```

解决：检查 `populateStableHLOToTTIRPatterns` 是否注册了对应的 pattern。

**2. Sharding 信息丢失**

```
error: unable to find shard status for value
```

解决：确保运行了完整的 StableHLO pipeline，特别是 `createAnalyzeMeshPass`。

**3. 类型转换失败**

```
error: failed to convert type 'tensor<*xf32>' to 'tensor<?xf32>'
```

解决：检查 TypeConverter 的配置，确保支持动态形状。

## 总结

StableHLO 到 TTIR 的转换是一个复杂的过程，涉及：

1. **操作语义映射**：将 StableHLO 操作转换为等价的 TTIR 操作
2. **多设备并行化**：通过 Shardy dialect 处理张量分片和集合通信
3. **类型和属性转换**：确保所有类型和属性正确映射
4. **Mesh 管理**：维护设备网格信息，指导运行时执行

关键要点：

- **Pipeline 顺序很重要**：必须先运行 StableHLO pipeline 完成 sharding 分析
- **Manual Computation 是核心**：多设备代码都在这个结构中
- **MeshShardOp 管理数据分布**：明确标记 Full ↔ Shard 的转换点
- **类型安全**：使用 TypeConverter 确保转换正确性

下一步：生成的 TTIR 将进入优化阶段（文档 04），然后降级到 TTNN（文档 05）。
