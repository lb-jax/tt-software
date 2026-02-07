# Tenstorrent 编译流程概览

## 目录

- [1. 编译流程全景](#1-编译流程全景)
- [2. 各层架构详解](#2-各层架构详解)
- [3. 关键技术组件](#3-关键技术组件)
- [4. 文档导航](#4-文档导航)

---

## 1. 编译流程全景

### 1.1 从 PyTorch/JAX 到硬件执行的完整路径

```
┌─────────────────────────────────────────────────────────────────┐
│                      Framework Layer                             │
│  PyTorch (torch.compile)     JAX (PJRT plugin)                  │
└────────────────────┬─────────────────────┬──────────────────────┘
                     │                     │
                     ▼                     ▼
           ┌─────────────────────────────────────┐
           │    TT-XLA Frontend                  │
           │  - torch_pass_pipeline              │
           │  - PJRT module_builder              │
           │  - Fusion & Decomposition           │
           └─────────────┬───────────────────────┘
                         │
                         ▼
           ┌─────────────────────────────────────┐
           │    StableHLO + Shardy               │
           │  - Framework-independent IR         │
           │  - Sharding annotations             │
           │  - Composite operations             │
           └─────────────┬───────────────────────┘
                         │
                         ▼ [StableHLOToTTIR Pass]
           ┌─────────────────────────────────────┐
           │    TTIR (TT Intermediate Rep)       │
           │  - Device-agnostic semantics        │
           │  - High-level tensor operations     │
           │  - Fusing, decomposition, layout    │
           └─────────────┬───────────────────────┘
                         │
                         ▼ [TTIRToTTNN Pass]
           ┌─────────────────────────────────────┐
           │    TTNN (TT Neural Network)         │
           │  - TTNN library operations          │
           │  - Layout optimization              │
           │  - Memory configuration             │
           │  - Hardware-specific optimizations  │
           └─────────────┬───────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Flatbuffer   │  │   EmitC      │  │   EmitPy     │
│   Binary     │  │  (C++ code)  │  │ (Py code)    │
└──────┬───────┘  └──────────────┘  └──────────────┘
       │                │                │
       │ (主要路径)      │ (调试)         │ (验证)
       │                │                │
       ▼                ▼                ▼
┌──────────────────────────────────────────────────┐
│         TT-Metalium Runtime                      │
│  ┌────────────────────────────────────────────┐  │
│  │ TTNN Runtime                               │  │
│  │ - Operations execution                     │  │
│  │ - Tensor management                        │  │
│  └────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────┐  │
│  │ TTMetal Runtime                            │  │
│  │ - Kernel dispatch (Prefetcher/Dispatcher)  │  │
│  │ - Device management                        │  │
│  │ - Memory allocation (L1/DRAM)              │  │
│  │ - Command queue management                 │  │
│  └────────────────────────────────────────────┘  │
└──────────────┬───────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────┐
│         Hardware (Wormhole/Blackhole)            │
│  ┌────────────────────────────────────────────┐  │
│  │ Tensix Cores (NOC mesh)                    │  │
│  │  - 5 RISC-V processors per core            │  │
│  │    • 1x Unpacker (NCRISC)                  │  │
│  │    • 3x Math cores (TRISC0/1/2)            │  │
│  │    • 1x Packer                             │  │
│  └────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────┐  │
│  │ Memory Hierarchy                           │  │
│  │  - L1 SRAM (per core, ~1MB)                │  │
│  │  - DRAM (shared, GBs)                      │  │
│  │  - NoC (Network-on-Chip for data movement) │  │
│  └────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────┘
```

### 1.2 编译时间线

| 阶段 | 输入 | 输出 | 主要工作 | 耗时 |
|------|------|------|----------|------|
| **前端转换** | PyTorch/JAX | StableHLO | 图捕获、融合、分解 | ~100ms |
| **StableHLO→TTIR** | StableHLO | TTIR | Sharding传播、集合操作 | ~200ms |
| **TTIR优化** | TTIR | TTIR | Fusing、Decomposition、Layout | ~500ms |
| **TTIR→TTNN** | TTIR | TTNN | 映射到TTNN库操作 | ~100ms |
| **TTNN优化** | TTNN | TTNN | Layout、内存配置、验证 | ~300ms |
| **代码生成** | TTNN | Flatbuffer | 序列化为二进制格式 | ~50ms |
| **运行时加载** | Flatbuffer | Runtime | 加载到设备、初始化 | ~500ms |

**总编译时间**: 约 1-2 秒（对于中等规模模型）

---

## 2. 各层架构详解

### 2.1 TT-XLA Frontend

**位置**: `/home/ubuntu/work/tt/tt-xla/`

**职责**:
- 拦截 PyTorch `torch.compile` 调用或 JAX PJRT 请求
- 执行框架特定的优化（fusion、decomposition、type promotion）
- 转换到框架无关的 StableHLO IR
- 传播 sharding 注解（用于多设备并行）

**关键组件**:
```python
# PyTorch backend
@register_backend(name="tt")
def xla_backend(gm, example_inputs, options={}):
    # Pass pipeline:
    # 1. Fusion passes
    # 2. Handle composite ops
    # 3. Decomposition (torch.export.export)
    # 4. Type promotion
    # 5. Metadata extraction
    return XLAExecutor(module, graph_signature, node_info)
```

**详细文档**: [01-pytorch-to-stablehlo.md](./01-pytorch-to-stablehlo.md)

### 2.2 StableHLO + Shardy

**位置**: MLIR 外部 dialect（由 OpenXLA 维护）

**职责**:
- 提供框架无关的中间表示
- 支持多设备 sharding（通过 Shardy annotations）
- 保留高层语义（composite operations）

**关键特性**:
- **Sharding**: 使用 `sdy.sharding` 属性标注张量分片
- **Composite Ops**: 保留模式匹配的高层操作（如 `LayerNorm`）
- **Control Flow**: 支持条件、循环等控制流

**示例 StableHLO IR**:
```mlir
func.func @matmul(%arg0: tensor<128x256xf32>,
                  %arg1: tensor<256x512xf32>) -> tensor<128x512xf32> {
  %0 = stablehlo.dot_general %arg0, %arg1,
       contracting_dims = [1] x [0]
       : (tensor<128x256xf32>, tensor<256x512xf32>) -> tensor<128x512xf32>
  return %0 : tensor<128x512xf32>
}
```

### 2.3 TTIR (TT Intermediate Representation)

**位置**: `/home/ubuntu/work/tt/tt-mlir/include/ttmlir/Dialect/TTIR/`

**职责**:
- 设备无关的高层语义表示
- 调度工作到 Tenstorrent 硬件
- 执行高层优化（fusion、decomposition、layout planning）

**核心操作类型**:
- **Tensor Ops**: `add`, `multiply`, `matmul`, `conv2d`
- **Layout Ops**: `reshape`, `transpose`, `permute`, `broadcast`
- **Reduction Ops**: `sum`, `max`, `mean`
- **Special Ops**: `embedding`, `softmax`, `layer_norm`

**示例 TTIR**:
```mlir
#layout = #tt.layout<(d0, d1) -> (d0, d1), undef>
func.func @add(%arg0: tensor<32x64xf32, #layout>,
               %arg1: tensor<32x64xf32, #layout>)
            -> tensor<32x64xf32, #layout> {
  %0 = ttir.add %arg0, %arg1 : tensor<32x64xf32, #layout>
  return %0 : tensor<32x64xf32, #layout>
}
```

**详细文档**:
- [03-stablehlo-to-ttir.md](./03-stablehlo-to-ttir.md) - 转换流程
- [04-ttir-optimizations.md](./04-ttir-optimizations.md) - 优化 pass

### 2.4 TTNN (TT Neural Network)

**位置**: `/home/ubuntu/work/tt/tt-mlir/include/ttmlir/Dialect/TTNN/`

**职责**:
- 映射到 TTNN 库的实际操作
- 硬件特定优化（layout、memory config）
- 操作验证和 fallback 策略

**TTNN 操作特点**:
- **Layout**: Row-Major vs Tile layout (32x32 tiles)
- **Memory**: L1 vs DRAM placement
- **Data Type**: FP32, FP16, BFP8, BFP4
- **Sharding**: 张量如何分布到多核心

**示例 TTNN**:
```mlir
%0 = "ttnn.matmul"(%arg0, %arg1) {
  memory_config = #ttnn.memory_config<
    tensor_memory_layout = interleaved,
    buffer_type = dram
  >,
  dtype = #tt.data_type<f32>,
  layout = #ttnn.layout<tile>
} : (tensor<128x256xf32>, tensor<256x512xf32>) -> tensor<128x512xf32>
```

**详细文档**:
- [05-ttir-to-ttnn.md](./05-ttir-to-ttnn.md) - TTIR → TTNN 转换
- [06-ttnn-to-ttmetal.md](./06-ttnn-to-ttmetal.md) - 代码生成

### 2.5 Flatbuffer Binary

**位置**: `/home/ubuntu/work/tt/tt-mlir/runtime/include/tt/runtime/flatbuffer/types.fbs`

**职责**:
- 序列化编译后的计算图
- 包含设备配置、张量描述、操作序列
- 高效的零拷贝反序列化

**Flatbuffer Schema**:
```cpp
table MeshDeviceOptions {
  mesh_offset: [uint32];
  device_ids: [int];
  num_hw_cqs: uint8;
  enable_program_cache: bool;
  mesh_shape: [uint32];
  l1_small_size: uint64;
  trace_region_size: uint64;
  dispatch_core_type: DispatchCoreType;
}

table TensorDesc {
  shape: [int32];
  layout: Layout;
  memory_config: MemoryConfig;
  dtype: DataType;
}

table Program {
  operations: [Operation];
  tensors: [TensorDesc];
  devices: [MeshDeviceOptions];
}
```

### 2.6 TT-Metalium Runtime

**位置**: `/home/ubuntu/work/tt/tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/`

**职责**:
- 加载 Flatbuffer 二进制
- 初始化设备和内存
- 执行操作序列
- 管理命令队列和内核调度

**两层架构**:

#### TTNN Runtime
- 高层操作执行（matmul, conv2d, softmax 等）
- 张量生命周期管理
- 自动 layout 转换

#### TTMetal Runtime
- 低层内核调度（Prefetcher/Dispatcher 架构）
- 设备内存分配（L1 SRAM, DRAM）
- 命令队列管理（Fast Dispatch）
- NoC（Network-on-Chip）通信

**详细文档**: [08-runtime-execution.md](./08-runtime-execution.md)

---

## 3. 关键技术组件

### 3.1 MLIR Dialects 层次结构

```
┌─────────────────────────────────────────────┐
│           Framework Dialects                │
│  torch, mhlo, stablehlo                     │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│      TTIR (High-level TT Operations)        │
│  - Device-agnostic                          │
│  - Semantic-preserving                      │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│      TTNN (TTNN Library Operations)         │
│  - Hardware-aware                           │
│  - Layout & memory annotated                │
└─────────────────┬───────────────────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
        ▼                   ▼
┌──────────────┐    ┌──────────────┐
│  TTKernel    │    │   TTMetal    │
│  (Kernel IR) │    │  (Metal IR)  │
└──────┬───────┘    └──────┬───────┘
       │                   │
       ▼                   ▼
┌──────────────────────────────────┐
│     LLVM IR / RISC-V Assembly    │
└──────────────────────────────────┘
```

**详细文档**: [02-mlir-dialects.md](./02-mlir-dialects.md)

### 3.2 优化 Pass 流水线

#### StableHLO → TTIR Pipeline
```cpp
void createStableHLOToTTIRPipeline(OpPassManager &pm) {
  pm.addPass(createInlinerPass());                      // 1. 内联
  pm.addPass(createTTPopulateArgumentTypes());          // 2. 类型注解
  pm.addPass(createConvertXlaSdyToSdyPass());           // 3. Shardy 转换
  pm.addPass(createAnalyzeMeshPass());                  // 4. Mesh 分析
  pm.addPass(createFlattenCompositePass());             // 5. Flatten composites
  pm.addPass(createRegisterCustomShardingRulePass());   // 6. 注册 sharding 规则
  pm.addPass(createUserPriorityPropagationPass());      // 7. Shardy 传播
  pm.addPass(createInsertExplicitReshardsPass());       // 8. 插入 reshards
  pm.addPass(createWrapUnderManualComputationPass());   // 9. 包装 manual_computation
  pm.addPass(createReshardToCollectivesPass());         // 10. Reshard → 集合操作
  pm.addPass(createUpdateGlobalToLocalShapesPass());    // 11. 更新形状
}
```

#### TTIR Optimization Pipeline
```cpp
void createTTNNPipelineTTIRPasses(OpPassManager &pm) {
  pm.addPass(createTTCoreRegisterDevicePass());         // 1. 注册设备
  pm.addPass(createTTIRFusing());                       // 2. Fusing
  pm.addPass(createTTIRQuantDequantConversion());       // 3. 量化/反量化
  pm.addPass(createTTIRToTTIRDecompositionPass());      // 4. 分解
  pm.addPass(createInlinerPass());                      // 5. 内联
  pm.addPass(createTTIRFlattenSlidingWindow());         // 6. Flatten 滑动窗口
  pm.addPass(createTTIRExplicateTMs());                 // 7. 显式化变换
  pm.addPass(createTTIREraseInverseOps());              // 8. 消除逆操作
  pm.addPass(createTTIRImplicitBroadcastFold());        // 9. 折叠广播
}
```

#### TTNN Optimization Pipeline
```cpp
void createTTNNPipelineAnalysisPasses(OpPassManager &pm) {
  // Layout passes
  pm.addPass(createTTNNRowMajorLayoutPropagation());    // 1. Row-major 传播
  pm.addPass(createTTNNOptimizer());                    // 2. TTNN 优化器
  pm.addPass(createTTNNOperationValidationAndFallback()); // 3. 验证和 fallback
  pm.addPass(createTTNNPrepareConv2dWeightsAndBias());  // 4. Conv2d 权重准备

  // Code generation
  pm.addPass(createTTNNLayout());                       // 5. 添加 layout
  pm.addPass(createConvertTTIRToTTNNPass());            // 6. TTIR → TTNN
  pm.addPass(createTTNNDeallocate());                   // 7. 插入 deallocate
  pm.addPass(createTTNNDecomposeLayouts());             // 8. 分解 layouts
}
```

### 3.3 Sharding 和并行化

**Shardy (Sharding Dialect)**:
- 用于多设备并行的关键技术
- 在 StableHLO 层开始注解 sharding
- 传播到 TTIR 和 TTNN 层
- 自动插入集合操作（all-gather, reduce-scatter, all-reduce）

**Sharding 类型**:
```mlir
// 1. 完全分片（shard 在维度 0）
#sharding1 = #sdy.sharding<@mesh_1, [{?}, {}]>

// 2. 完全复制
#sharding2 = #sdy.sharding<@mesh_1, [{}, {}]>

// 3. 部分分片（维度 0 分片，维度 1 复制）
#sharding3 = #sdy.sharding<@mesh_1, [{0}, {}]>
```

**Mesh 定义**:
```mlir
sdy.mesh @mesh_1 = <["x"=4, "y"=2]>  // 4x2 设备网格
```

**集合操作**:
- `all_gather`: 收集所有分片到完整张量
- `reduce_scatter`: 规约后分散到各设备
- `all_reduce`: 全局规约
- `all_to_all`: 全对全重分布

### 3.4 Layout 和内存管理

#### Layout 类型

**Row-Major Layout** (默认):
```
Tensor shape: [32, 64]
Memory layout: 线性存储，按行优先
```

**Tile Layout** (硬件优化):
```
Tensor shape: [32, 64]
Tile shape: [32, 32]
Tiles: 2 tiles (32x32 each)
```

#### Memory Configuration

**L1 SRAM** (每核心约 1MB):
- 低延迟、高带宽
- 用于频繁访问的数据（激活、权重）

**DRAM** (全局共享，GBs):
- 高容量、较低带宽
- 用于大型张量存储

**Interleaved vs Sharded**:
- **Interleaved**: 张量在多个核心间交错存储
- **Sharded**: 张量在特定核心上独占存储

### 3.5 Data Types

| 类型 | 位宽 | 精度 | 使用场景 |
|------|------|------|----------|
| FP32 | 32 | 单精度浮点 | 训练、高精度推理 |
| FP16 | 16 | 半精度浮点 | 推理、训练（混合精度） |
| BFP8 | 8 | Block Floating Point | 权重压缩、快速推理 |
| BFP4 | 4 | Block Floating Point | 极致压缩 |
| INT8 | 8 | 8位整数 | 量化推理 |

---

## 4. 文档导航

### 按主题浏览

#### 前端转换
- [01-pytorch-to-stablehlo.md](./01-pytorch-to-stablehlo.md) - PyTorch/JAX → StableHLO

#### MLIR 编译器
- [02-mlir-dialects.md](./02-mlir-dialects.md) - MLIR Dialects 详解
- [03-stablehlo-to-ttir.md](./03-stablehlo-to-ttir.md) - StableHLO → TTIR
- [04-ttir-optimizations.md](./04-ttir-optimizations.md) - TTIR 优化
- [05-ttir-to-ttnn.md](./05-ttir-to-ttnn.md) - TTIR → TTNN
- [06-ttnn-to-ttmetal.md](./06-ttnn-to-ttmetal.md) - TTNN → TTMetal

#### 运行时和执行
- [07-kernel-compilation.md](./07-kernel-compilation.md) - 内核编译
- [08-runtime-execution.md](./08-runtime-execution.md) - 运行时执行

#### 实践
- [09-end-to-end-example.md](./09-end-to-end-example.md) - 端到端示例

### 按难度浏览

**入门** (建议阅读顺序):
1. 本文档（00-compilation-overview.md）
2. [02-mlir-dialects.md](./02-mlir-dialects.md)
3. [09-end-to-end-example.md](./09-end-to-end-example.md)

**进阶**:
4. [01-pytorch-to-stablehlo.md](./01-pytorch-to-stablehlo.md)
5. [03-stablehlo-to-ttir.md](./03-stablehlo-to-ttir.md)
6. [05-ttir-to-ttnn.md](./05-ttir-to-ttnn.md)

**高级**:
7. [04-ttir-optimizations.md](./04-ttir-optimizations.md)
8. [06-ttnn-to-ttmetal.md](./06-ttnn-to-ttmetal.md)
9. [07-kernel-compilation.md](./07-kernel-compilation.md)
10. [08-runtime-execution.md](./08-runtime-execution.md)

---

## 5. 关键文件路径速查

### TT-XLA Frontend
```
/home/ubuntu/work/tt/tt-xla/
├── python_package/tt_torch/backend/
│   ├── backend.py                        # PyTorch backend 入口
│   └── torch_pass_pipeline.py            # Pass pipeline
├── pjrt_implementation/
│   ├── module_builder/                   # StableHLO 构建器
│   └── frontend_passes/                  # Frontend passes
└── tests/
    ├── jax/                              # JAX 测试
    └── torch/                            # PyTorch 测试
```

### TT-MLIR Compiler
```
/home/ubuntu/work/tt/tt-mlir/
├── include/ttmlir/Dialect/
│   ├── TTIR/                             # TTIR dialect
│   │   ├── IR/TTIROps.td                 # 操作定义
│   │   └── Transforms/Passes.td          # Pass 定义
│   ├── TTNN/                             # TTNN dialect
│   │   ├── IR/TTNNOps.td
│   │   └── Transforms/Passes.td
│   ├── TTKernel/                         # TTKernel dialect
│   └── TTMetal/                          # TTMetal dialect
├── lib/
│   ├── Conversion/                       # 转换 passes
│   │   ├── StableHLOToTTIR/
│   │   ├── TTIRToTTNN/
│   │   └── TTNNToEmitC/
│   ├── Dialect/                          # Dialect 实现
│   │   ├── TTIR/Pipelines/
│   │   └── TTNN/Pipelines/
│   └── Target/                           # 代码生成
│       ├── TTNN/TTNNToFlatbuffer.cpp
│       └── TTMetal/TTMetalToFlatbuffer.cpp
└── runtime/
    ├── include/tt/runtime/flatbuffer/
    │   └── types.fbs                     # Flatbuffer schema
    └── lib/                              # Runtime 实现
```

### TT-Metal Runtime
```
/home/ubuntu/work/tt/tt-xla/third_party/tt-mlir/src/tt-mlir/
third_party/tt-metal/src/tt-metal/
├── tt_metal/
│   ├── llrt/                             # Low-level runtime
│   │   ├── tt_cluster.cpp                # 集群管理
│   │   └── rtoptions.cpp                 # 运行时选项
│   ├── impl/
│   │   ├── dispatch/                     # 命令调度
│   │   │   ├── command_queue.cpp
│   │   │   └── kernels/                  # Prefetcher/Dispatcher 固件
│   │   └── allocator/                    # 内存分配器
│   └── third_party/umd/device/
│       ├── chip/                         # Chip 抽象
│       └── simulation/                   # 模拟器
└── ttnn/
    └── cpp/                              # TTNN 库实现
```

---

## 6. 相关文档

### TT-Metal Runtime 分析
- [../tt-metal-analysis/README.md](../tt-metal-analysis/README.md) - TT-Metal 运行时完整分析
- [../tt-metal-analysis/07-dispatch-architecture-deep-dive.md](../tt-metal-analysis/07-dispatch-architecture-deep-dive.md) - Dispatch 架构详解

### 官方文档
- [TT-MLIR GitHub](https://github.com/tenstorrent/tt-mlir)
- [TT-Metal GitHub](https://github.com/tenstorrent/tt-metal)
- [TT-XLA GitHub](https://github.com/tenstorrent/tt-xla)

---

*创建时间: 2025-02*
*最后更新: 2025-02*
