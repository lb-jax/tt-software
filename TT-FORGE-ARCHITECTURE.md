# TT-Forge 架构分析报告

本文档记录 Tenstorrent 编译器生态系统的构建流程和架构分析，目标是理解各仓库如何协同工作，以便用自定义模拟器替换硬件层。

## 目录

1. [工作空间概览](#1-工作空间概览)
2. [构建步骤分析](#2-构建步骤分析)
3. [MLIR 方言层次结构](#3-mlir-方言层次结构)
4. [运行时调用链](#4-运行时调用链)
5. [硬件抽象层接口](#5-硬件抽象层接口)
6. [仓库依赖关系](#6-仓库依赖关系)
7. [端到端执行详解](#7-端到端执行详解)

---

## 1. 工作空间概览

### 1.1 仓库结构

```
/home/ubuntu/work/tt/
├── tt-mlir/           # 核心 MLIR 编译器框架
├── tt-xla/            # PJRT 后端 (JAX/PyTorch 接入)
├── tt-forge/          # 中央集成仓库 (benchmarks, demos)
├── pytorch-xla/       # Tenstorrent fork 的 PyTorch/XLA (提供 _XLAC 模块)
└── vcpkg/             # C++ 包管理器
```

### 1.2 编译器栈架构

```
ML Frameworks (PyTorch, JAX, ONNX)
           ↓
   Frontend Layer
   ├── TT-XLA (JAX/PyTorch via StableHLO)
   └── TT-Forge-ONNX (ONNX/TF via TVM)
           ↓
   TT-MLIR Compiler
   ├── StableHLO-IR → TTIR → TTNN/TTMetal
   └── Passes: layout, fusing, decomposition, sharding
           ↓
   TT-Metalium (TTNN + TTMetal)
           ↓
   Hardware (Wormhole N150/N300, Blackhole P150B)
           ↑
   [模拟器替换点]
```

---

## 2. 构建步骤分析

### 2.0 构建路径概览

tt-mlir 存在**双重构建**的情况，根据目标不同可以选择不同的构建路径：

**完整构建路径** (步骤一到九):
```
步骤 1-5: 构建工具链 (LLVM/MLIR, FlatBuffers)
步骤 6:   构建 tt-mlir 独立版 (可选)
步骤 7-9: 构建 tt-xla (包含完整版 tt-mlir + tt-metal)
```

**最小构建路径** (推荐，跳过步骤六):
```
步骤 1-5: 构建工具链 (必须)
步骤 7-9: 构建 tt-xla (必须)
```

| 步骤 | 内容 | 是否必须 | 说明 |
|------|------|----------|------|
| 1-2 | Clone tt-mlir, 设置目录 | ✅ 必须 | 工具链源码 |
| 3-4 | 构建工具链 (LLVM/MLIR) | ✅ 必须 | tt-xla 依赖此工具链 |
| 5 | 激活环境 | ✅ 必须 | 用于步骤 3-4 |
| **6** | **构建 tt-mlir 独立版** | ❌ **可跳过** | tt-xla 会自己构建完整版 |
| 7-9 | 构建 tt-xla, 运行 Demo | ✅ 必须 | 包含完整运行时 |

**为什么步骤六可以跳过？**

- 步骤 3-4 构建的是**工具链** (`/opt/ttmlir-toolchain/`)，tt-xla 需要它
- 步骤 6 构建的是 tt-mlir **独立版**，但默认 `RUNTIME=OFF`，功能不完整
- 步骤 8 tt-xla 会自己构建**完整版** tt-mlir (`RUNTIME=ON, STABLEHLO=ON`)

**什么时候需要步骤六？**

| 场景 | 是否需要步骤六 |
|------|---------------|
| 运行 Demo / 模型推理 | ❌ 不需要 |
| 开发/调试 tt-mlir 编译器 | ✅ 需要 |
| 使用不同版本的 tt-mlir | ✅ 需要 |
| 只做编译测试 (不需要运行时) | ✅ 需要 |

---

### 步骤一：Clone tt-mlir 仓库

```bash
git clone https://github.com/tenstorrent/tt-mlir.git
# 目标路径: /home/ubuntu/work/tt/tt-mlir
```

### 步骤二：设置环境变量和目录

```bash
cd tt-mlir
export TTMLIR_TOOLCHAIN_DIR=/opt/ttmlir-toolchain/
sudo mkdir -p "${TTMLIR_TOOLCHAIN_DIR}"
sudo chown -R "${USER}" "${TTMLIR_TOOLCHAIN_DIR}"
```

**说明**: `TTMLIR_TOOLCHAIN_DIR` 是工具链安装目录，包含 LLVM/MLIR、FlatBuffers 等依赖。

### 步骤三：配置环境构建

```bash
cmake -B env/build env
```

**执行流程**:

1. 读取 `env/CMakeLists.txt`
2. 配置 ExternalProject 目标（不执行构建）:
   - `python-venv`: Python 虚拟环境
   - `flatbuffers`: 序列化库 (commit: `fb9afba...`)
   - `llvm-project`: LLVM/MLIR (commit: `4efe170...`)
   - `stablehlo`: XLA 标准 HLO (commit: `0a4440a...`)
   - `shardy`: 张量分片库 (commit: `edfd673...`)

### 步骤四：构建工具链

```bash
cmake --build env/build
```

**构建依赖图**:

```
                    python-venv (首先)
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
    flatbuffers    llvm-project    stablehlo (只clone)
                        │              │
                        │         shardy (clone+patch)
                        ▼
                   llvm-lit
```

**产物位置**:

```
/opt/ttmlir-toolchain/
├── venv/                    # Python 虚拟环境
│   └── bin/python
├── bin/                     # 工具链二进制
│   ├── flatc                # FlatBuffers 编译器
│   ├── mlir-opt             # MLIR 优化器
│   ├── mlir-translate       # MLIR 翻译器
│   └── llvm-lit             # 测试框架
├── lib/
│   ├── libMLIR*.a           # MLIR 库
│   └── cmake/mlir/          # CMake 配置
└── include/
    ├── mlir/                # MLIR 头文件
    └── flatbuffers/         # FlatBuffers 头文件

env/build/
├── stablehlo-prefix/src/stablehlo/   # StableHLO 源码
└── shardy-prefix/src/shardy/         # Shardy 源码 (已打补丁)
```

### 步骤五：激活环境

```bash
source env/activate
```

**设置的环境变量**:

| 变量 | 值 | 用途 |
|------|-----|------|
| `TTMLIR_TOOLCHAIN_DIR` | `/opt/ttmlir-toolchain` | 工具链根目录 |
| `TTMLIR_VENV_DIR` | `/opt/ttmlir-toolchain/venv` | Python 虚拟环境 |
| `TTMLIR_ENV_ACTIVATED` | `1` | 环境激活标记 |
| `TT_MLIR_HOME` | `<tt-mlir>` | tt-mlir 根目录 |
| `TT_METAL_HOME` | `<tt-mlir>/third_party/tt-metal/src/tt-metal` | tt-metal 根目录 |

**PATH 修改** (优先级从高到低):
1. `<tt-mlir>/build/bin` - tt-mlir 工具
2. `/opt/ttmlir-toolchain/bin` - 工具链
3. `/opt/ttmlir-toolchain/venv/bin` - Python

**PYTHONPATH 修改**:
```
<tt-mlir>/build/python_packages        # tt-mlir Python 绑定
<tt-mlir>/build/runtime/python         # 运行时接口
<tt-metal>/ttnn                        # TTNN Python 接口
<tt-metal>/tt_eager                    # TT-Eager 接口
```

### 步骤六：构建 tt-mlir (⚠️ 可选)

> **注意**: 如果你的目标是运行 Demo 或使用完整的编译器栈，可以**跳过此步骤**，直接进入步骤七。tt-xla 会自动构建包含完整运行时的 tt-mlir。

```bash
source env/activate
cmake -G Ninja -B build
cmake --build build
```

**默认构建选项**:

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `TTMLIR_ENABLE_RUNTIME` | OFF | 不构建运行时 (tt-metal 不会构建) |
| `TTMLIR_ENABLE_STABLEHLO` | OFF | 不启用 StableHLO |
| `TTMLIR_ENABLE_TTRT` | ON | 构建 ttrt 工具 |
| `TTMLIR_ENABLE_TOOLS` | ON | 构建核心工具 |
| `TTMLIR_ENABLE_BINDINGS_PYTHON` | ON | Python 绑定 |

**重要**: 默认 `TTMLIR_ENABLE_RUNTIME=OFF`，tt-metal 不会被构建！

**构建产物**:

```
tt-mlir/build/
├── bin/
│   ├── ttmlir-opt           # MLIR 优化工具
│   ├── ttmlir-translate     # MLIR 翻译器
│   ├── ttmlir-lsp-server    # LSP 服务
│   └── ttrt                 # 运行时工具
├── lib/
│   ├── libMLIRTTIR.a        # TTIR 方言库
│   ├── libMLIRTTNN.a        # TTNN 方言库
│   └── libMLIRTTMetal.a     # TTMetal 方言库
└── python_packages/
    └── ttmlir/              # Python 包
```

### 步骤七：Clone tt-xla 并初始化子模块

> **如果跳过了步骤六**: 确保步骤 1-4 已完成（工具链已构建），然后直接从这里继续。

```bash
cd /home/ubuntu/work/tt
export TTMLIR_TOOLCHAIN_DIR=/opt/ttmlir-toolchain/
git clone https://github.com/tenstorrent/tt-xla.git
cd tt-xla
git submodule update --init --recursive
```

**tt-xla 目录结构**:

```
tt-xla/
├── src/                        # C++ 源码 (PJRT 插件实现)
├── pjrt_implementation/src/    # PJRT API 实现
├── python_package/             # Python 包
│   ├── jax_plugin_tt/          # JAX 插件包装
│   ├── torch_plugin_tt/        # PyTorch/XLA 插件包装
│   ├── pjrt_plugin_tt/         # 共享 PJRT 插件
│   └── tt_torch/               # PyTorch backend
├── tests/                      # 测试用例
│   ├── jax/                    # JAX 测试
│   └── torch/                  # PyTorch 测试
├── third_party/                # 第三方依赖
│   ├── tt-mlir/                # tt-mlir (ExternalProject，非子模块!)
│   ├── tt_forge_models/        # 模型测试用例 (git 子模块)
│   ├── pjrt_c_api/             # PJRT C API 头文件
│   └── loguru/                 # 日志库
└── venv/                       # 虚拟环境脚本
```

**关键发现**: tt-xla 中的 tt-mlir **不是 git 子模块**，而是通过 CMake ExternalProject 管理：

```cmake
# tt-xla/third_party/CMakeLists.txt
set(TT_MLIR_VERSION "975b80cb68a4367f71e4d94e744af7ca0ee4eff5")

ExternalProject_Add(tt-mlir
    GIT_REPOSITORY https://github.com/tenstorrent/tt-mlir.git
    GIT_TAG ${TT_MLIR_VERSION}
    CMAKE_ARGS
      -DTTMLIR_ENABLE_STABLEHLO=ON      # 启用 StableHLO
      -DTTMLIR_ENABLE_RUNTIME=ON        # 启用运行时
      -DTT_RUNTIME_ENABLE_TTNN=ON       # 启用 TTNN
      -DTTMLIR_ENABLE_OPMODEL=ON        # 启用 OpModel
    ...
)
```

**tt-mlir 双重存在**:

| 位置 | 版本 | 用途 |
|------|------|------|
| `/home/ubuntu/work/tt/tt-mlir/` | 你手动 clone 的 (main) | 独立开发/测试 |
| `tt-xla/third_party/tt-mlir/src/tt-mlir/` | `975b80cb6` | tt-xla 构建时使用 |

### 步骤八：构建 TT-XLA (PJRT 插件)

```bash
cd /home/ubuntu/work/tt/tt-xla
source venv/activate
cmake -G Ninja -B build
cmake --build build
```

#### 8.1 环境激活 (`source venv/activate`)

**设置的环境变量**:

| 变量 | 值 | 说明 |
|------|-----|------|
| `TTMLIR_TOOLCHAIN_DIR` | `/opt/ttmlir-toolchain` | 工具链 |
| `TT_MLIR_HOME` | `tt-xla/third_party/tt-mlir/src/tt-mlir/` | tt-xla 内的 tt-mlir |
| `TTXLA_ENV_ACTIVATED` | `1` | CMake 检查此变量 |
| `TTMLIR_ENV_ACTIVATED` | `1` | 兼容 tt-mlir |
| `ARCH_NAME` | `wormhole_b0` | **目标硬件架构** |

**Python 依赖** (首次激活时安装):

| 包 | 版本 | 来源 |
|----|------|------|
| `jax` | 0.7.1 | PyPI |
| `jaxlib` | 0.7.1 | PyPI |
| `torch` | 2.9.0+cpu | PyTorch CPU wheel |
| `torch-xla` | 2.9.0 | **Tenstorrent 内部 PyPI** |

**重要**: `torch-xla` 来自 Tenstorrent 定制版本：
```
torch-xla@https://pypi.eng.aws.tenstorrent.com/torch-xla/torch_xla-2.9.0+git061c1e7-cp311-cp311-linux_x86_64.whl
```

这就是你 clone 的 `/home/ubuntu/work/tt/pytorch-xla/` 仓库编译的产物，包含 `_XLAC` 模块。

#### 8.2 构建依赖图

```
pjrt_plugin_tt.so (TTPJRTTTDylib)  ← 最终产物，JAX 导入的插件
│
├── TTPJRTBindings                  ← API 绑定逻辑
│   └── TTPJRTApi                   ← PJRT API 核心实现
│       ├── TTMLIRCompiler          ← tt-mlir 编译器库 (.so)
│       ├── TTMLIRRuntime           ← tt-mlir 运行时库 (.so)
│       ├── TTPJRTUtils             ← 工具函数
│       │   └── loguru              ← 日志库
│       └── protobuf::libprotobuf   ← Protocol Buffers
│
└── coverage_config                 ← 代码覆盖率配置
```

#### 8.3 完整构建流程

```
cmake -G Ninja -B build && cmake --build build
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  1. 构建 tt-mlir (ExternalProject)                      │
│     - Clone tt-mlir (commit: 975b80cb6)                │
│     - 启用: STABLEHLO, RUNTIME, TTNN, OPMODEL          │
│     - 触发 tt-metal 构建 (因为 RUNTIME=ON)              │
│     - 安装到 third_party/tt-mlir/install/              │
│     耗时: 30-60 分钟                                    │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  2. 构建 loguru                                         │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  3. 构建 PJRT 插件                                       │
│     TTPJRTUtils → TTPJRTApi → TTPJRTBindings            │
│     → TTPJRTTTDylib (pjrt_plugin_tt.so)                │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  4. 安装插件 (install-dylib-editable)                   │
│     - 软链接 .so 到 python_package/pjrt_plugin_tt/      │
│     - pip install -e python_package (editable 模式)    │
└─────────────────────────────────────────────────────────┘
```

#### 8.4 构建产物

```
tt-xla/
├── build/
│   └── pjrt_implementation/src/
│       └── pjrt_plugin_tt.so        ← PJRT 插件动态库
│
├── third_party/tt-mlir/
│   ├── src/tt-mlir/                 ← tt-mlir 源码 (commit: 975b80cb6)
│   │   ├── build/                   ← tt-mlir 构建目录
│   │   └── third_party/tt-metal/    ← tt-metal (嵌套)
│   │       └── src/tt-metal/
│   │           └── build/lib/       ← tt-metal 库
│   │               ├── libtt_metal.so
│   │               ├── libdevice.so   ← 模拟器替换点
│   │               ├── _ttnn.so
│   │               └── ...
│   │
│   └── install/                     ← tt-mlir 安装目录
│       └── lib/
│           ├── libTTMLIRCompiler.so
│           ├── libTTMLIRRuntime.so
│           └── ...
│
├── python_package/
│   └── pjrt_plugin_tt/
│       └── pjrt_plugin_tt.so        ← 软链接
│
└── venv/                            ← Python 虚拟环境
    └── lib/python3.11/site-packages/
        ├── jax/
        ├── torch/
        ├── torch_xla/               ← Tenstorrent 定制版
        │   └── _XLAC.cpython-311-x86_64-linux-gnu.so
        └── pjrt_plugin_tt/          ← editable install
```

### 步骤九：运行 Demo (端到端执行)

```bash
cd /home/ubuntu/work/tt
git clone https://github.com/tenstorrent/tt-forge.git
cd tt-forge
git submodule update --init --recursive   # 下载 tt_forge_models
export PYTHONPATH=.
python3 demos/tt-xla/cnn/resnet_demo.py
```

**注意**: 运行 Demo 前必须在同一 shell 中先激活 tt-xla 环境 (`source venv/activate`)。

**tt-forge 仓库结构**:

```
tt-forge/
├── demos/                      # 演示脚本
│   └── tt-xla/
│       ├── cnn/resnet_demo.py  ← 运行的文件
│       ├── llm/
│       └── nlp/
├── benchmark/                  # 性能基准测试
├── basic_tests/                # 基础验证测试
└── third_party/
    └── tt_forge_models/        # 模型定义 (git submodule)
```

**Demo 代码核心流程** (`resnet_demo.py`):

```python
import torch_xla.runtime as xr
from tt_torch.backend.backend import xla_backend

# 1. 设置 XLA 运行时设备为 TT
xr.set_device_type("TT")

# 2. 加载模型
model = loader.load_model(dtype_override=torch.bfloat16)

# 3. 使用 TT 后端编译模型 (关键!)
compiled_model = torch.compile(model, backend=xla_backend)

# 4. 移动到 TT 设备
device = xm.xla_device()
compiled_model = compiled_model.to(device)

# 5. 运行推理
output = compiled_model(inputs)
```

**端到端执行流程**:

```
torch.compile(model, backend="tt")
         │
         ▼
┌─────────────────────────────────────────┐
│ xla_backend (tt_torch.backend.backend)  │
│ - torch_pass_pipeline()                 │
│ - 返回 XLAExecutor                       │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ torch_xla (_XLAC 模块)                   │
│ - 生成 StableHLO 计算图                  │
│ - 调用 PJRT 插件                         │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ pjrt_plugin_tt.so                       │
│ → TTMLIRCompiler (StableHLO → TTNN)     │
│ → TTMLIRRuntime (执行 FlatBuffer)        │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ TTNN → tt-metal → libdevice.so          │
│ → 硬件/模拟器                            │
└─────────────────────────────────────────┘
```

---

## 3. MLIR 方言层次结构

### 3.1 方言转换流程

```
                    ┌─────────────┐
                    │  StableHLO  │  ← JAX/PyTorch/XLA 输入
                    └──────┬──────┘
                           │ StableHLOToTTIR Pass
                           ▼
                    ┌─────────────┐
                    │    TTIR     │  ← Tenstorrent 通用 IR
                    └──────┬──────┘
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
       ┌─────────────┐          ┌─────────────┐
       │    TTNN     │          │  TTMetal    │
       │ (高级神经网络)│          │ (底层内核)   │
       └──────┬──────┘          └──────┬──────┘
              │                         │
              └────────────┬────────────┘
                           ▼
                    ┌─────────────┐
                    │  FlatBuffer │  ← 序列化的可执行格式
                    └──────┬──────┘
                           │ ttrt (运行时)
                           ▼
                    ┌─────────────┐
                    │  tt-metal   │  ← 硬件抽象层
                    └─────────────┘
```

### 3.2 方言目录结构

```
lib/Dialect/
├── TTIR/          # Tenstorrent 通用 IR
├── TTNN/          # 神经网络高级 ops
├── TTMetal/       # 底层内核访问
├── TTKernel/      # 内核抽象
├── TTCore/        # 核心计算原语
├── StableHLO/     # StableHLO 支持
├── D2M/           # Device-to-Memory
├── SFPI/          # 固件操作
├── EmitPy/        # Python 代码生成
├── Debug/         # 调试工具
└── LLVM/          # LLVM 集成
```

### 3.3 转换 Pass 目录

```
lib/Conversion/
├── StableHLOToTTIR/    # StableHLO → TTIR
├── TTIRToTTNN/         # TTIR → TTNN
├── TTIRToTTMetal/      # TTIR → TTMetal
├── TTNNToEmitPy/       # TTNN → Python
└── ...
```

---

## 4. 运行时调用链

### 4.1 完整调用链 (PJRT 插件视角)

```
用户代码: jax.numpy.add(a, b) 或 torch.compile(model, backend='tt')
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  JAX 运行时 / PyTorch XLA                               │
│  - 生成 StableHLO 计算图                                 │
│  - 查找 PJRT 插件 (通过 jax.devices('tt'))              │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  pjrt_plugin_tt.so (PJRT 插件)                          │
│  - 接收 StableHLO 模块                                   │
│  - 位置: tt-xla/build/.../pjrt_plugin_tt.so            │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  TTMLIRCompiler (libTTMLIRCompiler.so)                  │
│  - StableHLO → TTIR → TTNN                             │
│  - 位置: tt-xla/third_party/tt-mlir/install/lib/       │
│  - 输出 FlatBuffer                                      │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  TTMLIRRuntime (libTTMLIRRuntime.so)                    │
│  - 加载 FlatBuffer                                      │
│  - 调度执行                                              │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  TTNN (_ttnn.so, _ttnncpp.so)                           │
│  - 神经网络 op 实现                                      │
│  - 位置: tt-metal/build/lib/                           │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  tt-metal (libtt_metal.so)                              │
│  - Metal API 实现                                        │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  libdevice.so (UMD - User Mode Driver)                  │
│  - 最底层设备接口                                        │
│  ← **你的模拟器替换点**                                   │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Tenstorrent 硬件 (或模拟器)                             │
│  - Wormhole B0 (ARCH_NAME=wormhole_b0)                 │
└─────────────────────────────────────────────────────────┘
```

### 4.2 关键库文件

| 库文件 | 所属层 | 位置 | 功能 |
|--------|--------|------|------|
| `pjrt_plugin_tt.so` | tt-xla | `build/pjrt_implementation/src/` | JAX/PyTorch 插件入口 |
| `libTTMLIRCompiler.so` | tt-mlir | `third_party/tt-mlir/install/lib/` | 编译器 |
| `libTTMLIRRuntime.so` | tt-mlir | `third_party/tt-mlir/install/lib/` | 运行时 |
| `_ttnncpp.so` | tt-metal | `tt-metal/build/lib/` | TTNN C++ 库 |
| `_ttnn.so` | tt-metal | `tt-metal/build/lib/` | TTNN Python 绑定 |
| `libtt_metal.so` | tt-metal | `tt-metal/build/lib/` | Metal 核心库 |
| `libdevice.so` | tt-metal | `tt-metal/build/lib/` | **设备抽象层** |
| `_XLAC.*.so` | pytorch-xla | `venv/.../torch_xla/` | PyTorch XLA C++ 绑定 |

---

## 5. 硬件抽象层接口

### 5.1 tt-metal 依赖配置

来自 `tt-mlir/third_party/CMakeLists.txt`:

```cmake
set(TT_METAL_VERSION "27c514cddefb8fafabce7a77f614a68c12c504da")

# 只有启用 runtime 或 opmodel 时才构建
if (TTMLIR_ENABLE_RUNTIME OR TTMLIR_ENABLE_OPMODEL)
    ExternalProject_Add(tt-metal
        GIT_REPOSITORY https://github.com/tenstorrent/tt-metal.git
        GIT_TAG ${TT_METAL_VERSION}
        ...
    )
endif()
```

### 5.2 tt-metal 构建产物

```
third_party/tt-metal/src/tt-metal/build/lib/
├── _ttnncpp.so        # TTNN C++ 库
├── _ttnn.so           # TTNN Python 绑定
├── libtt_metal.so     # TT-Metal 核心库
├── libdevice.so       # 设备抽象层 ← 模拟器替换点
├── libtt_stl.so       # TT STL 库
└── libtracy.so        # 性能分析
```

### 5.3 设备检测与 Mock 模式

#### 5.3.1 错误原因

运行 Demo 时报错 `TT_FATAL @ tt_cluster.cpp:117: num_chips > 0` 的原因是：

```cpp
// tt_cluster.cpp:117
const auto num_chips = cluster_desc->get_all_chips().size();
TT_FATAL(num_chips > 0, "No chips detected in the cluster");
```

没有物理硬件时，PCIe 扫描返回空结果，导致断言失败。

#### 5.3.2 目标设备类型

tt-metal 支持三种目标设备类型：

| 类型 | 环境变量 | 说明 |
|------|----------|------|
| `Silicon` | (默认) | 真实 Tenstorrent 硬件 |
| `Simulator` | `TT_METAL_SIMULATOR=/path` | 软件模拟器 |
| `Mock` | `TT_METAL_MOCK_CLUSTER_DESC_PATH=/path` | 测试用 Mock |

**优先级**: `Simulator > Mock > Silicon`

#### 5.3.3 快速启用 Mock 模式

```bash
# 使用预定义的 Wormhole N150 集群描述符
export TT_METAL_MOCK_CLUSTER_DESC_PATH=/path/to/tt-metal/tt_metal/third_party/umd/tests/cluster_descriptor_examples/wormhole_N150.yaml

# 然后运行程序
python3 demos/tt-xla/cnn/resnet_demo.py
```

**注意**: Mock 模式下所有设备操作返回空值，**不执行实际计算**，仅用于验证代码流程。

#### 5.3.4 集群描述符格式

```yaml
# wormhole_N150.yaml
arch: {
   0: Wormhole,              # chip_id: 架构类型
}
chips: {
   0: [0,0,0,0],            # chip_id: ethernet coordinates
}
ethernet_connections: []     # 芯片间连接 (单芯片可为空)
chips_with_mmio: [
   0: 0,                    # chip_id: mmio_device_id
]
harvesting: {
   0: {noc_translation: true, harvest_mask: 32},
}
boardtype: {
   0: n150,                 # 板卡类型
}
```

### 5.4 UMD (User Mode Driver) 接口

UMD 是模拟器接入的**最佳位置**，提供清晰的设备抽象接口。

#### 5.4.1 接口位置

```
tt-metal/tt_metal/third_party/umd/device/api/umd/device/
├── chip/
│   ├── chip.hpp          # Chip 基类 (核心接口)
│   ├── mock_chip.hpp     # MockChip 实现
│   └── local_chip.hpp    # 本地芯片 (Silicon)
├── simulation/
│   ├── simulation_chip.hpp    # 仿真芯片基类
│   ├── tt_sim_chip.hpp        # TTSIM 软件模拟
│   └── rtl_simulation_chip.hpp # RTL 仿真 (Zebu)
└── cluster.hpp           # Cluster 管理类
```

#### 5.4.2 Chip 核心接口 (chip.hpp)

```cpp
class Chip {
public:
    // 生命周期
    virtual void start_device() = 0;
    virtual void close_device() = 0;
    virtual bool is_mmio_capable() const = 0;

    // 设备内存读写 (L1/DRAM)
    virtual void write_to_device(CoreCoord core, const void* src,
                                  uint64_t l1_dest, uint32_t size) = 0;
    virtual void read_from_device(CoreCoord core, void* dest,
                                   uint64_t l1_src, uint32_t size) = 0;

    // 系统内存操作 (主机内存)
    virtual void write_to_sysmem(uint16_t channel, const void* src,
                                  uint64_t dest, uint32_t size) = 0;
    virtual void read_from_sysmem(uint16_t channel, void* dest,
                                   uint64_t src, uint32_t size) = 0;

    // DMA 操作
    virtual void dma_write_to_device(const void* src, size_t size,
                                      CoreCoord core, uint64_t addr) = 0;
    virtual void dma_read_from_device(void* dst, size_t size,
                                       CoreCoord core, uint64_t addr) = 0;

    // RISC 复位控制
    virtual void send_tensix_risc_reset(CoreCoord core,
                                         const TensixSoftResetOptions& opts) = 0;
    virtual void deassert_risc_resets() = 0;

    // 内存屏障
    virtual void l1_membar(const std::unordered_set<CoreCoord>& cores = {}) = 0;
    virtual void dram_membar(const std::unordered_set<CoreCoord>& cores = {}) = 0;

    // ARC 处理器消息
    virtual int arc_msg(uint32_t msg_code, bool wait_for_done = true,
                        const std::vector<uint32_t>& args = {},
                        uint32_t* return_3 = nullptr,
                        uint32_t* return_4 = nullptr) = 0;
};
```

#### 5.4.3 模拟器实现策略

**方案一：继承 `SimulationChip`**
- 继承现有的 `SimulationChip` 基类
- 只需实现纯虚函数
- 参考 `TTSimChip` 实现

**方案二：继承 `Chip` 基类**
- 完全自定义实现
- 更灵活但工作量更大
- 参考 `MockChip` 实现

**方案三：QEMU 虚拟 PCIe 设备**
- 在系统层面模拟 Tenstorrent PCIe 设备
- 对 tt-metal 完全透明
- 需要内核模块开发

### 5.5 模拟器替换策略

要用模拟器替换硬件，有以下选项:

1. **使用 Mock 模式** (最简单，仅用于测试)
   - 设置 `TT_METAL_MOCK_CLUSTER_DESC_PATH`
   - 不执行实际计算

2. **实现 `Chip` 接口** (推荐用于模拟器)
   - 继承 `SimulationChip` 或 `Chip`
   - 实现设备读写、DMA、复位控制等
   - 需要修改 `open_driver()` 集成

3. **QEMU 虚拟 PCIe** (对 tt-metal 透明)
   - 完全在系统层模拟
   - 难度最高

具体接口定义在:
```
tt-metal/tt_metal/third_party/umd/device/api/  # UMD (User Mode Driver) 接口
```

详细分析见: `/home/ubuntu/work/tt/tt-metal-analysis/`

---

## 6. 仓库依赖关系

### 6.1 完整依赖图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              tt-xla                                      │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  third_party/                                                      │  │
│  │  ├── tt-mlir/ (ExternalProject, commit: 975b80cb6)                │  │
│  │  │   ├── TTMLIR_ENABLE_STABLEHLO=ON                               │  │
│  │  │   ├── TTMLIR_ENABLE_RUNTIME=ON                                 │  │
│  │  │   └── third_party/tt-metal/ (嵌套 ExternalProject)              │  │
│  │  │       └── tt-metal (commit: 由 tt-mlir 定义)                    │  │
│  │  │           ├── libtt_metal.so                                   │  │
│  │  │           ├── libdevice.so  ← 模拟器替换点                      │  │
│  │  │           └── _ttnn.so                                         │  │
│  │  ├── tt_forge_models/ (git submodule)                             │  │
│  │  ├── pjrt_c_api/                                                  │  │
│  │  └── loguru/ (ExternalProject)                                    │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  venv/lib/python3.11/site-packages/                                     │
│  └── torch_xla/ ← 来自 Tenstorrent 定制版 pytorch-xla                   │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              │ 依赖
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  /opt/ttmlir-toolchain/ (由 tt-mlir 步骤三、四构建)                       │
│  ├── LLVM/MLIR                                                          │
│  ├── FlatBuffers                                                        │
│  ├── StableHLO 源码                                                     │
│  └── Shardy 源码                                                        │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              │ 来源
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  /home/ubuntu/work/tt/pytorch-xla/ (手动 clone)                         │
│  └── 编译产物: torch_xla whl (包含 _XLAC 模块)                           │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 tt-mlir 版本对比

| 构建方式 | tt-mlir 版本 | tt-metal 构建 | 用途 |
|----------|-------------|---------------|------|
| 独立构建 (步骤六) | main 分支 | **不构建** (默认) | 开发/测试编译器 |
| tt-xla 内构建 (步骤八) | `975b80cb6` | **构建** | 完整运行时 |

### 6.3 pytorch-xla 与 tt-xla 的关系

```
pytorch-xla (Tenstorrent fork)
     │
     │ 编译
     ▼
torch_xla-2.9.0+git061c1e7.whl
     │
     │ pip install (在 tt-xla venv/activate 时)
     ▼
tt-xla/venv/lib/python3.11/site-packages/torch_xla/
├── __init__.py
├── _XLAC.cpython-311-x86_64-linux-gnu.so  ← 你问的 _XLAC 模块
└── ...
```

`_XLAC` 是 PyTorch/XLA 的 C++ 核心绑定，由 Tenstorrent 定制以支持 TT 硬件。

---

## 7. 端到端执行详解

### 7.1 Demo 模块依赖

```
resnet_demo.py
     │
     ├── torch_xla (Tenstorrent 定制版)
     │   └── _XLAC.cpython-311-x86_64-linux-gnu.so
     │
     ├── tt_torch.backend.backend (来自 tt-xla)
     │   └── xla_backend (@register_backend(name="tt"))
     │
     ├── tt_forge_models (git submodule)
     │   └── resnet/pytorch/ModelLoader
     │
     └── pjrt_plugin_tt.so (运行时加载)
         ├── libTTMLIRCompiler.so
         └── libTTMLIRRuntime.so
             └── libtt_metal.so
                 └── libdevice.so  ← 模拟器替换点
```

### 7.2 关键函数调用链

| 步骤 | 函数/模块 | 作用 |
|------|----------|------|
| 1 | `xr.set_device_type("TT")` | 设置 XLA 使用 TT 设备 |
| 2 | `torch.compile(model, backend="tt")` | 注册 TT 编译后端 |
| 3 | `xla_backend()` | 运行 Torch 图优化 Pass |
| 4 | `model.to(xm.xla_device())` | 移动到 TT 设备 |
| 5 | `compiled_model(inputs)` | 触发 XLAExecutor |
| 6 | `_XLAC._xla_sync_multi()` | 触发实际编译和执行 |

### 7.3 `xla_backend` 优化 Pass

```python
# tt-xla/python_package/tt_torch/backend/backend.py
def torch_pass_pipeline(gm, example_inputs, options):
    gm = run_fusion_passes(gm)              # 算子融合
    handle_composite_ops(gm)                 # 复合算子处理
    program = torch.export.export(gm, ...)   # 导出
    program = program.run_decompositions()   # 分解
    compiled_graph = bypass_dtype_promotion_and_redundant_cast(...)  # 消除冗余
    return compiled_graph, signature, node_info
```

---

## 9. 待分析内容

- [x] tt-xla 构建流程
- [x] pytorch-xla 与 tt-xla 的关系
- [x] 端到端执行流程
- [ ] PJRT 插件接口详解
- [ ] libdevice.so 接口定义
- [ ] FlatBuffer 格式规范
- [ ] 如何用模拟器替换 libdevice.so

---

## 附录

### A. 环境变量速查表

| 变量 | 用途 | 设置位置 |
|------|------|----------|
| `TTMLIR_TOOLCHAIN_DIR` | 工具链目录 | 手动设置或 activate 脚本 |
| `TTMLIR_VENV_DIR` | Python 虚拟环境 | activate 脚本 |
| `TTMLIR_ENV_ACTIVATED` | tt-mlir 环境标记 | activate 脚本 |
| `TTXLA_ENV_ACTIVATED` | tt-xla 环境标记 | venv/activate |
| `TT_MLIR_HOME` | tt-mlir 根目录 | activate 脚本 |
| `TT_METAL_HOME` | tt-metal 根目录 | activate 脚本 |
| `ARCH_NAME` | 目标硬件架构 | venv/activate (默认 wormhole_b0) |
| `SYSTEM_DESC_PATH` | 系统描述符路径 | 测试时设置 |

### B. 常用构建命令

#### 最小构建路径 (推荐)

```bash
# === 1. 构建工具链 ===
cd /home/ubuntu/work/tt
git clone https://github.com/tenstorrent/tt-mlir.git
cd tt-mlir
export TTMLIR_TOOLCHAIN_DIR=/opt/ttmlir-toolchain/
sudo mkdir -p "${TTMLIR_TOOLCHAIN_DIR}"
sudo chown -R "${USER}" "${TTMLIR_TOOLCHAIN_DIR}"
cmake -B env/build env
cmake --build env/build      # 耗时约 30-60 分钟

# === 2. 构建 tt-xla (包含完整版 tt-mlir) ===
cd /home/ubuntu/work/tt
git clone https://github.com/tenstorrent/tt-xla.git
cd tt-xla
git submodule update --init --recursive
source venv/activate
cmake -G Ninja -B build
cmake --build build          # 耗时约 30-60 分钟

# === 3. 运行 Demo ===
cd /home/ubuntu/work/tt
git clone https://github.com/tenstorrent/tt-forge.git
cd tt-forge
git submodule update --init --recursive
export PYTHONPATH=.
python3 demos/tt-xla/cnn/resnet_demo.py
```

#### 完整构建路径 (开发 tt-mlir 时使用)

```bash
# === tt-mlir 工具链构建 ===
cd /home/ubuntu/work/tt/tt-mlir
export TTMLIR_TOOLCHAIN_DIR=/opt/ttmlir-toolchain/
cmake -B env/build env
cmake --build env/build

# === tt-mlir 基础构建 (不含 runtime，可选) ===
source env/activate
cmake -G Ninja -B build
cmake --build build

# === tt-mlir 完整构建 (含 runtime 和 tt-metal，可选) ===
cmake -G Ninja -B build -DTTMLIR_ENABLE_RUNTIME=ON -DTTMLIR_ENABLE_STABLEHLO=ON
cmake --build build

# === tt-xla 构建 ===
cd /home/ubuntu/work/tt/tt-xla
export TTMLIR_TOOLCHAIN_DIR=/opt/ttmlir-toolchain/
source venv/activate
cmake -G Ninja -B build
cmake --build build

# === 验证安装 ===
python -c "import jax; print(jax.devices('tt'))"
```

### C. 版本锁定

| 依赖 | Commit Hash | 使用位置 |
|------|-------------|----------|
| LLVM/MLIR | `4efe170d858eb54432f520abb4e7f0086236748b` | tt-mlir 工具链 |
| FlatBuffers | `fb9afbafc7dfe226b9db54d4923bfb8839635274` | tt-mlir 工具链 |
| StableHLO | `0a4440a5c8de45c4f9649bf3eb4913bf3f97da0d` | tt-mlir 工具链 |
| Shardy | `edfd6730ddfc39da5fbea8b6b202357fdf1cdb90` | tt-mlir 工具链 |
| tt-metal | `27c514cddefb8fafabce7a77f614a68c12c504da` | tt-mlir (独立) |
| tt-mlir | `975b80cb68a4367f71e4d94e744af7ca0ee4eff5` | tt-xla 内置 |
| torch-xla | `2.9.0+git061c1e7` | tt-xla venv |
| loguru | `4adaa185883e3c04da25913579c451d3c32cfac1` | tt-xla |

---

*文档更新时间: 2025-02*
*分析基于: tt-mlir main 分支, tt-xla main 分支, tt-forge main 分支*
