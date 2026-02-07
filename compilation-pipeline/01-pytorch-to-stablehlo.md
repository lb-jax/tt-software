# PyTorch/JAX 到 StableHLO 转换前端分析

本文档详细解释 TT-XLA 前端如何将 PyTorch 和 JAX 模型转换为 StableHLO IR，这是整个编译流程的第一阶段。

## 目录

1. [概述](#概述)
2. [PyTorch 集成](#pytorch-集成)
3. [PyTorch Pass Pipeline 详解](#pytorch-pass-pipeline-详解)
4. [JAX 集成](#jax-集成)
5. [StableHLO 前端 Pass](#stablehlo-前端-pass)
6. [完整流程图](#完整流程图)

---

## 概述

TT-XLA 通过两个主要入口点支持 PyTorch 和 JAX：

```
PyTorch Model                    JAX Model
      ↓                               ↓
torch.compile(backend="tt")      jax.jit + PJRT Plugin
      ↓                               ↓
  backend.py                    pjrt_plugin_tt.so
      ↓                               ↓
torch_pass_pipeline            ModuleBuilder (C++)
      ↓                               ↓
torch-xla (dynamo_bridge)         StableHLO IR
      ↓                               ↓
      └────────── StableHLO IR ──────┘
                    ↓
          Frontend SHLO Pipeline
                    ↓
            TT-MLIR Compiler
```

**关键特点：**
- PyTorch: 使用 `torch.compile` backend 机制，通过 `torch-xla` 生成 StableHLO
- JAX: 使用 PJRT 插件机制，直接生成 StableHLO
- 两条路径最终都生成 StableHLO IR，然后进入统一的编译流程

---

## PyTorch 集成

### 1. Backend 注册

**文件路径:** `/home/ubuntu/work/tt/tt-xla/python_package/tt_torch/backend/backend.py`

PyTorch 后端通过 `@register_backend` 装饰器注册：

```python
@register_backend(name="tt")
def xla_backend(gm, example_inputs, options={}):
    """TT backend for torch.compile."""
    # 1. 运行 pass pipeline
    module, graph_signature, node_info = torch_pass_pipeline(
        gm, example_inputs, options
    )

    # 2. 确定是否使用 legacy compile
    legacy_compile_enabled = False
    if "tt_experimental_compile" in options:
        legacy_compile_enabled = not bool(options["tt_experimental_compile"])
    if "tt_legacy_compile" in options:
        legacy_compile_enabled = bool(options["tt_legacy_compile"])

    # 3. 返回 executor
    return XLAExecutor(module, graph_signature, node_info, legacy_compile_enabled)
```

**使用方式：**

```python
import torch
import torch_xla

model = MyModel()
compiled_model = torch.compile(model, backend="tt")
output = compiled_model(input)
```

### 2. XLAExecutor 类

XLAExecutor 负责在 XLA 设备上执行编译后的程序。

**关键职责：**

```python
class XLAExecutor:
    def __init__(self, module, signature, node_info, legacy_compile_enabled):
        self.module = module                    # GraphModule
        self.signature = signature              # ExportGraphSignature
        self.node_info = node_info             # 元数据列表
        self.inject_metadata = ...             # 是否注入调试信息
        self.devices = ...                     # 收集所有设备
        self.legacy_compile_enabled = ...      # 编译模式

    def __call__(self, *args):
        # Legacy 模式：直接执行 + 手动同步
        if self.legacy_compile_enabled:
            output = self.module(*args)
            # 告诉 torch-xla 在输出处切图
            torch_xla._XLAC._xla_sync_multi(list(output), self.devices, wait=False)
            return output

        # Experimental 模式：使用 torch_xla bridge
        else:
            return self._call_experimental_compile(*args)
```

**Experimental Compile 流程：**

```python
def _call_experimental_compile(self, *args):
    if self.compiled_graph is None:
        # 1. 导出程序（lifting 参数和常量）
        program = torch.export.export(self.module, tuple(args), strict=False)

        # 2. 从导出的程序中提取参数和常量
        self.params_and_consts = self._build_params_and_consts(program)

        # 3. 使用 torch_xla bridge 提取编译后的图
        # 这避免了后续执行时的重复追踪
        self.compiled_graph = bridge.extract_compiled_graph(
            program.graph_module, self.params_and_consts + args
        )

    # 4. 执行编译后的图
    full_args = self.params_and_consts + args
    return self.compiled_graph(*full_args)
```

---

## PyTorch Pass Pipeline 详解

**文件路径:** `/home/ubuntu/work/tt/tt-xla/python_package/tt_torch/backend/backend.py:31-82`

`torch_pass_pipeline` 是 PyTorch 前端的核心，将 GraphModule 转换为可编译的形式。

### Pipeline 概览

```python
def torch_pass_pipeline(
    gm: torch.fx.GraphModule,
    example_inputs: Tuple[torch.Tensor],
    options: dict[str, bool] | None,
) -> Tuple[torch.fx.GraphModule, ExportGraphSignature, list[str]]:
```

**流程图：**

```
GraphModule (from torch.compile)
    ↓
[1] run_fusion_passes
    ↓
[2] handle_composite_ops
    ↓
[3] torch.export.export + run_decompositions
    ↓
[4] insert_argument_type_markers
    ↓
[5] bypass_dtype_promotion_and_redundant_cast
    ↓
[6] bypass_redundant_getitem
    ↓
[7] bypass_assert_tensor_metadata
    ↓
[8] compiled_graph.recompile()
    ↓
[9] extract_nodes_info (metadata extraction)
    ↓
Final GraphModule + Signature + Metadata
```

### Pass 详解

#### Pass 1: Fusion Passes

**文件路径:** `/home/ubuntu/work/tt/tt-xla/python_package/tt_torch/backend/passes.py:11-34`

```python
def run_fusion_passes(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """运行所有注册的融合 pass"""
    total_replacements = 0

    for provider_cls in FusionProvider.get_registered_providers():
        provider = provider_cls()
        num_replaced = provider.replace_pattern(gm)
        if num_replaced > 0:
            logger.debug(f"[Fusion] {provider.name}: {num_replaced} match(es)")
            total_replacements += num_replaced

    if total_replacements > 0:
        gm.graph.lint()
        gm.recompile()

    return gm
```

**融合模式示例：**

文件路径: `/home/ubuntu/work/tt/tt-xla/python_package/tt_torch/fusion_providers.py`

```python
class RMSNormFusionProvider(FusionProvider):
    @property
    def name(self) -> str:
        return "RMS Normalization Fusion"

    @staticmethod
    def pattern(input, weight, normalized_shape, eps):
        # 定义要匹配的模式
        variance = input.pow(2).mean(-1, keepdim=True)
        hidden_states = input * torch.rsqrt(variance + eps)
        return weight * hidden_states

    @staticmethod
    def replacement(input, weight, normalized_shape, eps):
        # 替换为 composite op
        return torch.nn.functional.rms_norm(input, normalized_shape, weight, eps)
```

**目的：** 将多个算子的组合模式识别为单一的高级算子（如 RMSNorm、LayerNorm），为后续的 composite ops 处理做准备。

#### Pass 2: Composite Ops Handling

**文件路径:** `/home/ubuntu/work/tt/tt-xla/python_package/tt_torch/backend/passes.py:37-62`

```python
def handle_composite_ops(gm: torch.fx.GraphModule) -> None:
    """将 torch ops 替换为 composite ops"""
    for node in gm.graph.nodes:
        # 1. 处理 call_function 节点
        if node.op == "call_function":
            if node.target in composite_ops.replacements:
                node.target = composite_ops.replacements[node.target]

        # 2. 处理 call_module 节点
        elif node.op == "call_module":
            module = gm.get_submodule(node.target)
            module_type = type(module)
            if module_type in composite_ops.replacements:
                composite_ops.replacements[module_type](gm, node, module)

    gm.graph.lint()
```

**Composite Op 示例：**

文件路径: `/home/ubuntu/work/tt/tt-xla/python_package/tt_torch/composite_ops.py`

```python
def composite_gelu(input: Tensor, approximate: str = "none") -> Tensor:
    """创建 composite gelu 操作"""
    tanh = approximate == "tanh"
    name = "tenstorrent.gelu" + ("_tanh" if tanh else "")
    attr = {"approximate": "tanh"} if tanh else None

    # 使用 StableHLOCompositeBuilder 标记输入输出
    builder = StableHLOCompositeBuilder(name=name, attr=attr)
    input = builder.mark_inputs(input)
    input = torch.nn.functional.gelu(input, approximate=approximate)
    input = builder.mark_outputs(input)

    return input
```

**生成的 StableHLO 结构：**

```mlir
%0 = stablehlo.composite "tenstorrent.gelu" %input {
  composite_attributes = {},
  decomposition = @gelu_decomposition
} : (tensor<f32>) -> tensor<f32>
```

**支持的 Composite Ops：**
- `tenstorrent.gelu` / `tenstorrent.gelu_tanh`
- `tenstorrent.rms_norm`
- `tenstorrent.layer_norm`
- 更多算子可通过 `composite_ops.py` 中的 `replacements` 字典查看

**为什么使用 Composite Ops？**
1. **性能优化**: 后端可以用定制实现替换整个模式，避免分解
2. **降低复杂度**: MLIR 编译器可以将 composite op 直接降级到 TTIR/TTNN 的原生算子
3. **保留语义**: 分解信息被保留，作为备选实现

#### Pass 3: Decomposition

**文件路径:** `/home/ubuntu/work/tt/tt-xla/python_package/tt_torch/backend/backend.py:55-62`

```python
# 1. 填充分解表
decompositions = populate_decompositions()

# 2. 使用 torch.export.export 导出程序
program = torch.export.export(
    gm,
    tuple(example_inputs),
    strict=False,
)

# 3. 运行分解
program = program.run_decompositions(decompositions)
compiled_graph = program.module()
```

**Decomposition 表构建：**

文件路径: `/home/ubuntu/work/tt/tt-xla/python_package/tt_torch/backend/decompositions.py:410-426`

```python
def populate_decompositions() -> DecompositionTable:
    # 1. 获取 PyTorch 核心 aten 分解
    decompositions = torch._decomp.core_aten_decompositions()

    # 2. 移除某些分解（避免不必要的转换）
    # 移除 einsum 分解（我们有自定义的 mm->einsum）
    decompositions.pop(torch.ops.aten.einsum.default)
    # 移除 dot 分解（我们有自定义的 dot->matmul）
    decompositions.pop(torch.ops.aten.dot.default)

    # 3. 添加默认分解
    decompositions.update(get_decompositions(_get_default_decomposition_ops()))

    # 4. 添加自定义分解
    decompositions.update(_get_custom_decompositions())

    return decompositions
```

**自定义分解示例：**

```python
def _get_custom_decompositions() -> DecompositionTable:
    aten = torch.ops.aten
    return {
        # 处理 copy 的广播语义
        aten.copy.default: copy_default,

        # matmul 处理高维张量
        aten.matmul.default: matmul,

        # Interpolation: 使用 matmul 替代 gather
        aten.upsample_bilinear2d.vec: upsample_linear_vec,
        aten.upsample_nearest2d.vec: upsample_nearest_vec,

        # 布尔位运算转换为逻辑运算
        aten.bitwise_and.Tensor: boolean_bitwise_and,
        aten.bitwise_or.Tensor: boolean_bitwise_or,

        # 其他...
    }
```

**关键分解：Bilinear Interpolation**

```python
def upsample_linear(input, output_size, align_corners, scales):
    """使用 matmul 实现插值，避免 gather 操作"""
    res = input
    for i in range(len(scales)):
        # 计算线性插值权重矩阵
        weight = compute_linear_weight(
            input_size[i], output_size[i], scales[i],
            align_corners, input.dtype, input.device
        )
        # 使用 matmul 应用权重
        res = (res.transpose(i - len(scales), -1) @ weight).transpose(
            i - len(scales), -1
        )
    return res
```

**为什么需要自定义分解？**
- PyTorch 的默认分解可能包含 TT-MLIR 难以处理的算子（如 `aten.gather`）
- 某些分解会引入性能问题（如 type promotion）
- 需要保留特定模式用于后续优化

#### Pass 4: Argument Type Markers

**文件路径:** `/home/ubuntu/work/tt/tt-xla/python_package/tt_torch/backend/passes.py:64-143`

```python
def insert_argument_type_markers(
    gm: torch.fx.GraphModule, graph_signature
) -> torch.fx.GraphModule:
    """为每个输入参数添加类型标记"""

    # 1. 收集所有输入节点
    input_nodes = gm.graph.find_nodes(op="get_attr") + \
                  gm.graph.find_nodes(op="placeholder")

    # 2. 根据 signature 确定参数类型
    for in_spec in input_signature:
        type_str = None
        if in_spec.kind == InputKind.USER_INPUT:
            type_str = "input"
        elif in_spec.kind == InputKind.PARAMETER:
            type_str = "parameter"
        elif in_spec.kind == InputKind.CONSTANT_TENSOR:
            type_str = "constant"
        elif in_spec.kind == InputKind.BUFFER:
            # 如果 buffer 被修改，标记为 input；否则标记为 constant
            if in_spec.target in mutated_buffer_targets:
                type_str = "input"
            else:
                type_str = "constant"

    # 3. 插入 marker 节点
    for input_node in input_nodes:
        with gm.graph.inserting_after(input_node):
            new_input = gm.graph.create_node(
                "call_function",
                torch.ops.tt.mark_argument_attributes,
                args=(input_node,),
                kwargs={"argument_type": argument_type, "name": input_node.name},
            )

        # 4. 替换所有使用
        for user in users:
            user.replace_input_with(input_node, new_input)

    return gm
```

**生成的 FX 图结构：**

```python
# 原始:
def forward(self, input, weight):
    x = linear(input, weight)
    ...

# 插入 marker 后:
def forward(self, input, weight):
    input_marked = torch.ops.tt.mark_argument_attributes(
        input, argument_type="input", name="input"
    )
    weight_marked = torch.ops.tt.mark_argument_attributes(
        weight, argument_type="parameter", name="weight"
    )
    x = linear(input_marked, weight_marked)
    ...
```

**在 StableHLO 中的体现：**

这些标记会被 torch-xla 转换为 StableHLO custom_call，然后在前端 pass 中被处理：

```mlir
func.func @main(%arg0: tensor<4x512xf32>, %arg1: tensor<512x256xf32>) {
  %0 = stablehlo.custom_call @tt.mark_argument_attributes(%arg0)
       {argument_type = "input", name = "input"}
  %1 = stablehlo.custom_call @tt.mark_argument_attributes(%arg1)
       {argument_type = "parameter", name = "weight"}
  ...
}
```

#### Pass 5: Dtype Promotion Bypass

**文件路径:** `/home/ubuntu/work/tt/tt-xla/python_package/tt_torch/backend/passes.py:196-231`

```python
def bypass_dtype_promotion_and_redundant_cast(gm, example_inputs):
    """移除不必要的类型转换"""
    removed_non_redundant_casts = False

    for node in gm.graph.nodes:
        if node.target.name() == "prims::convert_element_type":
            # 1. 检测不需要的 dtype promotion
            is_unwanted_dtype_promotion = (
                node.meta["original_aten"]._name != "aten::_to_copy"
                and node.args[1] == torch.float32
            )

            # 2. 检测冗余 cast (输入输出类型相同)
            is_redundant_cast = (
                node.args[0].meta["tensor_meta"].dtype == node.args[1]
            )

            # 3. 移除不必要的 cast
            if is_unwanted_dtype_promotion or is_redundant_cast:
                node.replace_all_uses_with(node.args[0])
                removed_non_redundant_casts |= is_unwanted_dtype_promotion

    gm.graph.eliminate_dead_code()

    # 4. 如果移除了非冗余 cast，重新传播 shape 并递归
    if removed_non_redundant_casts:
        run_shape_prop(gm, example_inputs)
        gm = bypass_dtype_promotion_and_redundant_cast(gm, example_inputs)

    return gm
```

**问题背景：**

PyTorch 在分解过程中会自动将操作提升到 float32，即使用户指定了其他 dtype（如 bfloat16）。这会导致：
- 不必要的内存开销
- 精度不匹配
- 性能下降

**示例：**

```python
# 用户代码 (bfloat16)
x = x.to(torch.bfloat16)
y = torch.nn.functional.gelu(x)

# PyTorch 分解后 (自动提升到 float32)
x_f32 = x.to(torch.float32)  # 不想要的提升
y_f32 = gelu_decomposition(x_f32)
y = y_f32.to(torch.bfloat16)  # 再转回来

# 此 pass 移除后
y = gelu_decomposition(x)  # 保持 bfloat16
```

#### Pass 6-8: 清理 Passes

**Pass 6: bypass_redundant_getitem**

```python
def bypass_redundant_getitem(gm):
    """移除冗余的 getitem 调用"""
    for node in gm.graph.nodes:
        if node.op == "call_function" and "getitem" in node.name:
            if isinstance(node.args[0], tuple):
                idx = node.args[1]
                if isinstance(idx, int):
                    # 直接替换为元组元素
                    node.replace_all_uses_with(node.args[0][idx])
    return gm
```

**Pass 7: bypass_assert_tensor_metadata**

```python
def bypass_assert_tensor_metadata(gm):
    """移除 tensor metadata 断言"""
    for node in gm.graph.nodes:
        if node.target == torch.ops.aten._assert_tensor_metadata.default:
            # 这些断言可能假设在 CPU 上，但我们在 XLA 设备上运行
            gm.graph.erase_node(node)
    return gm
```

**Pass 8: Recompile**

```python
compiled_graph.recompile()
```

确保所有修改反映到执行中。

#### Pass 9: Metadata Extraction

**文件路径:** `/home/ubuntu/work/tt/tt-xla/python_package/tt_torch/backend/metadata_propagation.py:327-366`

```python
def extract_nodes_info(graph_module: torch.fx.GraphModule) -> list[str]:
    """从 FX 图节点中提取位置元数据"""
    emit_locs = []
    op_index = 0

    for node in graph_module.graph.nodes:
        # 只处理 call_function 节点
        if node.op != "call_function":
            continue

        # 跳过 builtin 函数
        if isinstance(node.target, types.BuiltinFunctionType):
            continue

        # 提取源码和模块层次信息
        emit_loc = _extract_source_and_module_hierarchy_info(node, op_index)
        emit_locs.append(emit_loc)
        op_index += 1

    return [emit_loc.to_string() for emit_loc in emit_locs]
```

**提取的信息：**

```python
@dataclass
class EmitLoc:
    modules: list[EmitModuleLoc]  # 模块层次 [Model, Encoder, Layer0, ...]
    func_path: str                 # 文件路径:行号
    func_name: str                 # 函数名
    op_line_num: int              # 操作所在行号
    op_name: str                  # 操作名称
    op_index: int                 # 操作索引
```

**元数据注入：**

如果设置了 `XLA_HLO_DEBUG=1`，元数据会在运行时通过 `MetadataDispatchMode` 注入到 XLA tensors：

```python
class MetadataDispatchMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs={}):
        res = func(*args, **kwargs)

        # 为计算节点设置元数据
        if self.operation_index < len(self.node_info):
            module_hierarchy = self.node_info[self.operation_index]
            torch_xla._XLAC._set_xla_custom_op_name_prefix(
                res, module_hierarchy, 0
            )

        self.operation_index += 1
        return res
```

这使得生成的 MLIR 具有丰富的调试信息：

```mlir
%0 = stablehlo.add %arg0, %arg1
     loc("0|Model[model]|Encoder[encoder]|/path/model.py:42|forward|add")
```

### Pipeline 输出

经过完整的 pipeline 后，返回：

```python
(
    compiled_graph,      # 优化后的 GraphModule
    program.graph_signature,  # 输入输出签名
    node_info           # 元数据列表
)
```

这些会被传递给 `XLAExecutor`，然后通过 `torch-xla` 的 `dynamo_bridge` 转换为 StableHLO。

---

## JAX 集成

JAX 使用 PJRT (Portable JAX Runtime) 插件机制，与 PyTorch 不同，它直接在 C++ 层生成 StableHLO。

### 1. PJRT 插件架构

**插件加载流程：**

```python
import jax
# JAX 自动加载已注册的 PJRT 插件
jax.devices('tt')  # 返回 TT 设备列表
```

**底层机制：**

```
JAX Python API
    ↓
PJRT C API
    ↓
pjrt_plugin_tt.so (动态库)
    ↓
ModuleBuilder::buildModule
    ↓
StableHLO IR
```

**文件结构：**

```
python_package/
├── pjrt_plugin_tt/          # 共享插件包
│   └── pjrt_plugin_tt.so    # 核心动态库
├── jax_plugin_tt/           # JAX 包装器
│   └── __init__.py          # 注册 TT 插件
└── torch_plugin_tt/         # PyTorch/XLA 包装器
    └── __init__.py          # 注册 TT 插件
```

### 2. ModuleBuilder - C++ 编译入口

**文件路径:** `/home/ubuntu/work/tt/tt-xla/pjrt_implementation/inc/api/module_builder/module_builder.h`

ModuleBuilder 是 PJRT 插件的核心，负责将 StableHLO 编译为可执行二进制。

**类定义：**

```cpp
class ModuleBuilder {
public:
    ModuleBuilder();

    // 编译 MLIR 模块
    std::tuple<tt_pjrt_status, std::shared_ptr<ExecutableImage>>
    buildModule(
        const std::string_view &mlir_code,
        const std::string &system_descriptor_path,
        const std::unordered_map<std::string, std::string> &compile_options,
        tt::pjrt::ClientInstance *client_instance
    );

private:
    // VHLO/StableHLO 处理
    tt_pjrt_status createVHLOModule(...);
    tt_pjrt_status convertFromVHLOToSHLO(...);
    tt_pjrt_status runFrontendSHLOPipeline(...);
    tt_pjrt_status runCompilerStableHLOPipeline(...);

    // TTIR/TTNN 转换
    tt_pjrt_status convertFromSHLOToTTIR(...);
    tt_pjrt_status convertFromTTIRToTTNN(...);

    // Flatbuffer 生成
    tt_pjrt_status createFlatbufferBinary(...);

    std::unique_ptr<mlir::MLIRContext> m_context;
};
```

### 3. buildModule 流程

**文件路径:** `/home/ubuntu/work/tt/tt-xla/pjrt_implementation/src/api/module_builder/module_builder.cc:209-370`

```cpp
std::tuple<tt_pjrt_status, std::shared_ptr<ExecutableImage>>
ModuleBuilder::buildModule(
    const std::string_view &mlir_code,
    const std::string &system_descriptor_path,
    const std::unordered_map<std::string, std::string> &compile_options_map,
    ClientInstance *client_instance)
{
    auto compile_options = CompileOptions::parse(compile_options_map);

    // 1. 创建 VHLO module (Versioned StableHLO)
    mlir::OwningOpRef<mlir::ModuleOp> mlir_module;
    status = createVHLOModule(mlir_code, mlir_module,
                             compile_options.export_path,
                             compile_options.export_model_name);

    // 2. VHLO -> StableHLO 转换
    status = convertFromVHLOToSHLO(mlir_module,
                                   compile_options.export_path,
                                   compile_options.export_model_name);

    // 3. 运行前端 StableHLO pipeline
    status = runFrontendSHLOPipeline(mlir_module,
                                    compile_options.export_path,
                                    compile_options.export_model_name);

    // 4. 收集输入/输出 sharding 信息
    std::vector<mlir::tt::sharding_utils::MeshSharding> input_shardings;
    status = collectInputShardings(mlir_module, input_shardings);

    std::vector<mlir::tt::sharding_utils::MeshSharding> output_shardings;
    status = collectOutputShardings(mlir_module, output_shardings);

    // 5. 运行编译器 StableHLO pipeline (来自 tt-mlir)
    status = runCompilerStableHLOPipeline(mlir_module,
                                         compile_options.export_path,
                                         compile_options.export_model_name);

    // 6. StableHLO -> TTIR
    std::string ttir_code;
    status = convertFromSHLOToTTIR(mlir_module, ttir_code,
                                  compile_options.export_path,
                                  compile_options.export_model_name);

    // 7. TTIR -> TTNN
    std::string ttnn_code;
    status = convertFromTTIRToTTNN(system_descriptor_path, mlir_module,
                                   compile_options, client_instance,
                                   devices_mesh_shape, ttnn_code);

    // 8. 生成可执行映像
    if (compile_options.backend == BackendRuntime::TTNNFlatbuffer) {
        return buildModuleForTTNNRuntime(...);
    } else {
        return buildModuleForTTNNCodegen(...);
    }
}
```

**JAX 使用示例：**

```python
import jax
import jax.numpy as jnp

@jax.jit
def model(x, w):
    return jnp.matmul(x, w)

# JAX 自动使用 TT PJRT 插件
# 内部调用 ModuleBuilder::buildModule
x = jnp.ones((4, 512), device=jax.devices('tt')[0])
w = jnp.ones((512, 256), device=jax.devices('tt')[0])
y = model(x, w)
```

---

## StableHLO 前端 Pass

前端 passes 在 `runFrontendSHLOPipeline` 中执行，用于处理 TT-XLA 特定的注解和转换。

### Pass 1: Input Role Propagation

**文件路径:** `/home/ubuntu/work/tt/tt-xla/pjrt_implementation/inc/api/module_builder/frontend_passes/shlo_input_role_propagation.h`

**目的：** 将 PyTorch frontend 插入的 `tt.mark_argument_attributes` 标记传播到函数签名。

```cpp
tt_pjrt_status annotateArgumentAttributes(
    mlir::OwningOpRef<mlir::ModuleOp> &mlir_module
);
```

**转换示例：**

输入 StableHLO (来自 torch-xla):

```mlir
func.func @main(%arg0: tensor<4x512xf32>, %arg1: tensor<512x256xf32>) {
  %0 = stablehlo.custom_call @tt.mark_argument_attributes(%arg0)
       {argument_type = "input"}
  %1 = stablehlo.custom_call @tt.mark_argument_attributes(%arg1)
       {argument_type = "parameter"}
  %2 = stablehlo.dot_general %0, %1 ...
  return %2
}
```

输出 (注解传播后):

```mlir
func.func @main(
    %arg0: tensor<4x512xf32> {ttcore.argument_type = "input"},
    %arg1: tensor<512x256xf32> {ttcore.argument_type = "parameter"}
) {
  // tt.mark_argument_attributes custom_call 被移除并内联
  %0 = stablehlo.dot_general %arg0, %arg1 ...
  return %0
}
```

**内部步骤：**

1. `propagateInputRoleAttributes`: 追踪 custom_call 使用，向上传播到函数参数
2. `inlineTTMarkFunctions`: 内联并移除 marker 函数
3. `setDefaultRoleForUnannotatedArguments`: 为未标记的参数设置默认角色（Input）

### Pass 2: Proper Shardy Mesh Attribute

**文件路径:** `/home/ubuntu/work/tt/tt-xla/pjrt_implementation/inc/api/module_builder/frontend_passes/shlo_set_proper_sdy_mesh_attribute.h`

**目的：** 修复 PyTorch SPMD 模式中的退化 mesh 配置。

```cpp
tt_pjrt_status setProperSdyMeshAttributeInSpmdMode(
    mlir::OwningOpRef<mlir::ModuleOp> &mlir_module
);
```

**问题背景：**

在 PyTorch SPMD 模式下，完全复制的图可能默认为退化的 mesh `[1,1]`，而不是期望的 `[1, num_devices]`。

**转换示例：**

输入 (退化 mesh):

```mlir
sdy.mesh @mesh = <["data"=1, "model"=1]>

func.func @main(%arg0: tensor<4x512xf32>)
    attributes {mhlo.sharding = "{replicated}"} {
  // 完全复制的计算
  ...
}
```

输出 (修正后的 mesh):

```mlir
sdy.mesh @mesh = <["data"=1, "model"=8]>  // 假设有 8 个设备

func.func @main(%arg0: tensor<4x512xf32>)
    attributes {mhlo.sharding = "{devices=[1,8]<=[8]}"} {
  // 现在可以在所有 8 个设备上复制执行
  ...
}
```

**检测逻辑：**

```cpp
namespace internal {
bool isSpmdMode(const mlir::OwningOpRef<mlir::ModuleOp> &module) {
    // 如果参数包含 "mhlo.sharding" 属性，则为 SPMD 模式
    for (auto funcOp : getPublicFuncOps(module)) {
        for (auto arg : funcOp.getArguments()) {
            if (arg.hasAttr("mhlo.sharding")) {
                return true;
            }
        }
    }
    return false;
}
}
```

### Pass 3: Clean for XLA Ingestion

**文件路径:** `/home/ubuntu/work/tt/tt-xla/pjrt_implementation/inc/api/module_builder/frontend_passes/shlo_clean_for_xla_ingestion.h`

**目的：** 将 Shardy 注解的模块转换为 XLA 可解析的 GSPMD 格式。

```cpp
tt_pjrt_status cleanForXlaIngestion(
    mlir::OwningOpRef<mlir::ModuleOp> &mlir_module
);
```

**问题背景：**

XLA 的 MLIR 解析器（通过 `PJRT_OptimizedProgram` 返回）与 Shardy dialect 不兼容，导致输出被静默地全部复制。

**转换步骤：**

1. **移除 ttcore/ttir/sdy dialect 属性**
   ```mlir
   // Before
   func.func @main(%arg0: tensor<f32> {ttcore.argument_type = "input",
                                       sdy.sharding = #sdy.sharding<...>})

   // After
   func.func @main(%arg0: tensor<f32>)
   ```

2. **移除位置信息**
   ```mlir
   // Before
   %0 = stablehlo.add %arg0, %arg1
        loc("model.py:42:forward")

   // After
   %0 = stablehlo.add %arg0, %arg1
        loc(unknown)
   ```

3. **处理 sdy.manual_computation**
   ```mlir
   // Before
   %0 = sdy.manual_computation(%arg0) ... {
     // 复杂的分片计算
     %1 = ...
     sdy.return %1
   }

   // After
   %0 = stablehlo.constant dense<0.0> : tensor<...>  // Dummy output
   ```

4. **注入输出 sharding 为 module 属性**
   ```mlir
   module attributes {
     mhlo.spmd_output_shardings = [
       "{devices=[1,8]<=[8]}",
       "{replicated}"
     ]
   } {
     func.func @main(...) { ... }
   }
   ```

**为什么需要这个 pass？**

- XLA 在解析优化后的程序时期望纯粹的 GSPMD 注解
- Shardy 是更高级的抽象，XLA 不支持
- 此转换使得 `jax.jit(..., out_shardings=...)` 正确工作

### Frontend Pipeline 总结

完整的前端 StableHLO pipeline:

```cpp
tt_pjrt_status ModuleBuilder::runFrontendSHLOPipeline(
    mlir::OwningOpRef<mlir::ModuleOp> &mlir_module,
    const std::optional<std::string> &export_path,
    const std::string &model_name)
{
    // 1. 注解参数属性
    status = frontend_passes::annotateArgumentAttributes(mlir_module);
    if (!tt_pjrt_status_is_ok(status)) return status;

    // 2. 修正 Shardy mesh (如果需要)
    status = frontend_passes::setProperSdyMeshAttributeInSpmdMode(mlir_module);
    if (!tt_pjrt_status_is_ok(status)) return status;

    // 3. 为 XLA ingestion 清理 (如果需要返回给 XLA)
    if (need_xla_compatible_output) {
        status = frontend_passes::cleanForXlaIngestion(mlir_module);
        if (!tt_pjrt_status_is_ok(status)) return status;
    }

    printModule(mlir_module, export_path, "frontend_shlo", model_name);
    return tt_pjrt_status::kSuccess;
}
```

---

## 完整流程图

### PyTorch 完整流程

```
                     torch.compile(model, backend="tt")
                                  ↓
                        torch.fx.symbolic_trace
                                  ↓
                        GraphModule (FX Graph)
                                  ↓
┌────────────────────────────────────────────────────────────────────┐
│                    torch_pass_pipeline                              │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ [1] run_fusion_passes                                        │  │
│  │     - RMSNormFusion                                          │  │
│  │     - LayerNormFusion                                        │  │
│  │     - ...                                                    │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                            ↓                                        │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ [2] handle_composite_ops                                     │  │
│  │     - gelu → tenstorrent.gelu                               │  │
│  │     - rms_norm → tenstorrent.rms_norm                       │  │
│  │     - ...                                                    │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                            ↓                                        │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ [3] torch.export.export + run_decompositions                │  │
│  │     - populate_decompositions()                             │  │
│  │     - matmul → einsum (高维)                                │  │
│  │     - upsample → matmul 链                                  │  │
│  │     - bitwise ops → logical ops                             │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                            ↓                                        │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ [4] insert_argument_type_markers                            │  │
│  │     %0 = tt.mark(arg0, type="input")                        │  │
│  │     %1 = tt.mark(arg1, type="parameter")                    │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                            ↓                                        │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ [5] bypass_dtype_promotion_and_redundant_cast               │  │
│  │     - 移除 f32 promotion                                    │  │
│  │     - 移除冗余 cast                                         │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                            ↓                                        │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ [6-7] bypass_redundant_getitem +                            │  │
│  │       bypass_assert_tensor_metadata                         │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                            ↓                                        │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ [8] compiled_graph.recompile()                              │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                            ↓                                        │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ [9] extract_nodes_info                                      │  │
│  │     - 提取模块层次                                          │  │
│  │     - 提取源码位置                                          │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
                                  ↓
                    Optimized GraphModule + Metadata
                                  ↓
                           XLAExecutor
                                  ↓
           torch_xla.dynamo_bridge.extract_compiled_graph
                                  ↓
                    torch-xla (StableHLO 生成)
                                  ↓
                          StableHLO IR (VHLO)
                                  ↓
┌────────────────────────────────────────────────────────────────────┐
│                   ModuleBuilder (C++ PJRT)                         │
│                                                                    │
│  createVHLOModule → convertVHLOToSHLO → runFrontendSHLOPipeline   │
└────────────────────────────────────────────────────────────────────┘
                                  ↓
                         到 TT-MLIR 编译器
```

### JAX 完整流程

```
                          jax.jit(fn)
                                  ↓
                       JAX Tracer (jaxpr)
                                  ↓
                    JAX StableHLO Lowering
                                  ↓
                    StableHLO IR (VHLO)
                                  ↓
                         PJRT C API
                                  ↓
                    pjrt_plugin_tt.so
                                  ↓
┌────────────────────────────────────────────────────────────────────┐
│                 ModuleBuilder::buildModule                         │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ [1] createVHLOModule                                         │  │
│  │     - 解析 VHLO 字符串                                       │  │
│  │     - 创建 MLIR ModuleOp                                     │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                            ↓                                        │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ [2] convertFromVHLOToSHLO                                    │  │
│  │     - VHLO → StableHLO 版本转换                             │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                            ↓                                        │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ [3] runFrontendSHLOPipeline                                 │  │
│  │     - annotateArgumentAttributes                            │  │
│  │     - setProperSdyMeshAttribute                             │  │
│  │     - cleanForXlaIngestion                                  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                            ↓                                        │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ [4] collectInputShardings + collectOutputShardings          │  │
│  │     - 提取 GSPMD/Shardy sharding 信息                       │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                            ↓                                        │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ [5] runCompilerStableHLOPipeline (from tt-mlir)             │  │
│  │     - StableHLO 优化和转换                                  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                            ↓                                        │
│               到 TTIR 转换 (下一阶段)                              │
└────────────────────────────────────────────────────────────────────┘
```

### 对比总结

| 特性 | PyTorch 路径 | JAX 路径 |
|------|--------------|----------|
| **入口点** | `@register_backend("tt")` | PJRT Plugin |
| **语言** | Python (backend.py) | C++ (module_builder.cc) |
| **中间表示** | FX GraphModule | jaxpr |
| **StableHLO 生成** | torch-xla | JAX 内置 |
| **Pass Pipeline** | torch_pass_pipeline (Python) | ModuleBuilder (C++) |
| **Fusion** | FusionProvider 框架 | 依赖 JAX 转换 |
| **Composite Ops** | 显式标记 (StableHLOCompositeBuilder) | 通过 custom_call |
| **Decomposition** | 自定义分解表 | JAX 分解 |
| **Metadata** | MetadataDispatchMode | MLIR locations |
| **Sharding** | 通过 torch-xla 传播 | GSPMD/Shardy 注解 |

---

## 关键文件索引

### PyTorch Frontend

| 文件 | 行号 | 描述 |
|------|------|------|
| `/home/ubuntu/work/tt/tt-xla/python_package/tt_torch/backend/backend.py` | 212-227 | Backend 注册 |
| | 31-81 | torch_pass_pipeline |
| | 84-209 | XLAExecutor |
| `/home/ubuntu/work/tt/tt-xla/python_package/tt_torch/backend/passes.py` | 11-34 | run_fusion_passes |
| | 37-62 | handle_composite_ops |
| | 64-143 | insert_argument_type_markers |
| | 196-231 | bypass_dtype_promotion |
| `/home/ubuntu/work/tt/tt-xla/python_package/tt_torch/backend/decompositions.py` | 410-426 | populate_decompositions |
| | 371-407 | _get_custom_decompositions |
| | 25-60 | compute_linear_weight |
| `/home/ubuntu/work/tt/tt-xla/python_package/tt_torch/backend/metadata_propagation.py` | 327-366 | extract_nodes_info |
| | 174-324 | _extract_source_and_module_hierarchy_info |
| | 369-451 | MetadataDispatchMode |
| `/home/ubuntu/work/tt/tt-xla/python_package/tt_torch/composite_ops.py` | 30-47 | composite_gelu |
| | 50-79 | composite_rms_norm |
| | 82-126 | composite_layer_norm |
| `/home/ubuntu/work/tt/tt-xla/python_package/tt_torch/fusion_providers.py` | 20-92 | FusionProvider 基类 |
| | 97-159 | RMSNormFusionProvider |

### JAX Frontend / C++ Module Builder

| 文件 | 行号 | 描述 |
|------|------|------|
| `/home/ubuntu/work/tt/tt-xla/pjrt_implementation/inc/api/module_builder/module_builder.h` | 121-359 | ModuleBuilder 类定义 |
| `/home/ubuntu/work/tt/tt-xla/pjrt_implementation/src/api/module_builder/module_builder.cc` | 209-370 | buildModule 实现 |
| | 177-206 | ModuleBuilder 构造函数 |
| `/home/ubuntu/work/tt/tt-xla/pjrt_implementation/inc/api/module_builder/frontend_passes/shlo_input_role_propagation.h` | 23-55 | Input role 传播接口 |
| `/home/ubuntu/work/tt/tt-xla/pjrt_implementation/inc/api/module_builder/frontend_passes/shlo_set_proper_sdy_mesh_attribute.h` | 16-32 | Shardy mesh 修正接口 |
| `/home/ubuntu/work/tt/tt-xla/pjrt_implementation/inc/api/module_builder/frontend_passes/shlo_clean_for_xla_ingestion.h` | 16-36 | XLA ingestion 清理接口 |

---

## 后续阶段

StableHLO IR 生成后，进入统一的 TT-MLIR 编译流程：

```
StableHLO IR (frontend_shlo)
    ↓
runCompilerStableHLOPipeline (tt-mlir)
    ↓
TTIR (tt dialect)
    ↓
TTNN (ttnn dialect)
    ↓
TTMetal Kernels
    ↓
Tenstorrent Hardware
```

详见后续文档：
- `02-mlir-dialects.md` - MLIR Dialects 详解
- `03-stablehlo-to-ttir.md` - StableHLO 到 TTIR 转换
- `04-ttir-optimization.md` - TTIR 优化 Pass
- `05-ttir-to-ttnn.md` - TTIR 到 TTNN 降级

---

## 调试技巧

### 1. 导出中间 IR

**PyTorch:**

```python
import torch
import os

# 导出 FX Graph
os.environ["TORCH_LOGS"] = "+dynamo,+aot,+inductor"

model = torch.compile(model, backend="tt")
output = model(input)
```

**JAX + TT-XLA:**

```python
import os

# 导出所有中间 MLIR
os.environ["TTXLA_LOGGER_LEVEL"] = "VERBOSE"
os.environ["export_path"] = "/tmp/tt_xla_debug"
os.environ["export_model_name"] = "my_model"

# 生成的文件:
# /tmp/tt_xla_debug/my_model_g0_vhlo.mlir
# /tmp/tt_xla_debug/my_model_g0_shlo.mlir
# /tmp/tt_xla_debug/my_model_g0_frontend_shlo.mlir
# ...
```

### 2. 启用元数据注入 (PyTorch)

```bash
export XLA_HLO_DEBUG=1
```

生成的 MLIR 将包含丰富的位置信息：

```mlir
%0 = stablehlo.add %arg0, %arg1
     loc("0|Model[model]|Linear[fc1]|/path/model.py:forward:42|add")
```

### 3. 查看 Composite Ops

```python
# 禁用 composite ops 查看原始分解
compiled_model = torch.compile(
    model,
    backend="tt",
    options={"tt_enable_composite_ops": False}
)
```

### 4. 禁用 Fusion

```python
# 禁用 fusion passes
compiled_model = torch.compile(
    model,
    backend="tt",
    options={"tt_enable_torch_fx_fusion_pass": False}
)
```

### 5. 使用 Legacy Compile 模式

```python
# 使用 legacy compile 模式 (手动同步)
compiled_model = torch.compile(
    model,
    backend="tt",
    options={"tt_legacy_compile": True}
)
```

---

## 常见问题

### Q1: 为什么需要自定义分解？

**A:** PyTorch 的默认分解可能：
- 包含 TT-MLIR 不支持的算子（如 `aten.gather`）
- 引入不必要的 dtype promotion
- 无法利用硬件特性（如 Tensor Core 的 GEMM 融合）

自定义分解让我们控制计算图的形状，使其更适合 TT 硬件。

### Q2: Composite Ops 和 Fusion 的区别？

**A:**
- **Fusion**: 在 FX Graph 级别识别并合并多算子模式（如 RMSNorm = pow + mean + mul + rsqrt）
- **Composite Ops**: 将合并后的模式包装为 StableHLO composite，传递语义信息给 MLIR 编译器

流程: Fusion → Composite Wrapping → MLIR Lowering

### Q3: XLAExecutor 的两种编译模式有什么区别？

**A:**

| | Legacy Compile | Experimental Compile |
|---|----------------|---------------------|
| **追踪** | 每次执行都追踪 | 首次编译后缓存 |
| **同步** | 手动 `_xla_sync_multi` | 自动管理 |
| **性能** | 较慢（重复追踪） | 较快（缓存图） |
| **副作用** | 需要 `sync()` | 正确处理 mutation |
| **默认** | No | Yes (推荐) |

### Q4: 如何调试 "Operation not supported" 错误？

**A:**

1. 导出 StableHLO IR 查看是哪个算子
   ```bash
   export TTXLA_LOGGER_LEVEL=DEBUG
   export export_path=/tmp/debug
   ```

2. 检查是否是自定义分解问题
   ```python
   # 在 decompositions.py 中添加分解
   def my_custom_op_decomposition(input, ...):
       # 使用支持的算子实现
       return ...
   ```

3. 如果是 composite op 问题，检查 MLIR 是否识别
   ```bash
   # 搜索 tt-mlir 中的 composite handler
   grep -r "tenstorrent.my_op" /path/to/tt-mlir/
   ```

### Q5: Shardy 和 GSPMD 的关系？

**A:**

- **GSPMD**: Google's SPMDpartitioner，基于字符串的 sharding 规范
  ```python
  # GSPMD format
  "{devices=[1,8]<=[8]}"  # 在 8 个设备上沿第二维分片
  ```

- **Shardy**: Google Shardy，更高级的 MLIR dialect
  ```mlir
  #sdy.sharding<@mesh, [{}, {"model"}]>
  ```

TT-XLA 支持两者：
- JAX 原生使用 Shardy
- PyTorch/XLA 使用 GSPMD
- Frontend passes 在两种格式间转换

---

## 总结

本文档详细介绍了 TT-XLA 前端如何将 PyTorch 和 JAX 转换为 StableHLO IR：

**PyTorch 路径:**
1. 通过 `torch.compile(backend="tt")` 注册
2. `torch_pass_pipeline` 运行 9 个 passes
3. 使用 `torch-xla` 生成 StableHLO
4. XLAExecutor 管理执行

**JAX 路径:**
1. 通过 PJRT 插件加载 `pjrt_plugin_tt.so`
2. JAX 直接生成 StableHLO (VHLO)
3. `ModuleBuilder::buildModule` 在 C++ 层处理

**共同的前端 passes:**
- Input role propagation
- Shardy mesh 修正
- XLA ingestion 清理

最终输出统一的 StableHLO IR，进入 TT-MLIR 编译器的后续阶段。
