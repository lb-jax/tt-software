# TT-MLIR Dialects Reference

This document provides a comprehensive reference for all MLIR dialects in the TT-MLIR compiler ecosystem. These dialects form the intermediate representation layers that progressively lower high-level ML operations to Tenstorrent hardware instructions.

## Table of Contents

1. [Dialect Architecture Overview](#dialect-architecture-overview)
2. [Dialect Transformation Flow](#dialect-transformation-flow)
3. [Core Dialects](#core-dialects)
   - [TTCore - Common Types and Attributes](#ttcore---common-types-and-attributes)
   - [TTIR - High-Level Dispatch](#ttir---high-level-dispatch)
   - [TTNN - Neural Network Operations](#ttnn---neural-network-operations)
   - [D2M - Direct-to-Metal](#d2m---direct-to-metal)
   - [TTKernel - Kernel Operations](#ttkernel---kernel-operations)
   - [TTMetal - Low-Level Runtime](#ttmetal---low-level-runtime)
   - [SFPI - SFPU Programming Interface](#sfpi---sfpu-programming-interface)
4. [Dialect Details and Operations](#dialect-details-and-operations)
5. [Type System](#type-system)
6. [Code Examples](#code-examples)

---

## Dialect Architecture Overview

The TT-MLIR compiler uses a multi-layered dialect architecture to progressively lower operations from ML framework semantics to hardware-specific instructions:

```
┌─────────────────────────────────────────────────────────┐
│  Frontend (StableHLO, ONNX via TVM)                     │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  TTIR (TT Intermediate Representation)                  │
│  • High-level tensor operations                         │
│  • Platform-agnostic dispatch semantics                 │
│  • Layout and memory space abstractions                 │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  TTNN (TT Neural Network)                               │
│  • Neural network operations                            │
│  • Device-aware operations                              │
│  • Memory configuration and layout management           │
└──────────────────────┬──────────────────────────────────┘
                       │
                ┌──────┴──────┐
                ▼             ▼
┌──────────────────────┐  ┌──────────────────────┐
│  D2M (Direct-to-Metal)│  │  TTKernel            │
│  • Generic dispatch   │  │  • FPU operations    │
│  • Compute/Data ops   │  │  • SFPU operations   │
└──────────┬────────────┘  │  • Tile operations   │
           │               └──────────┬───────────┘
           │                          │
           └──────────┬───────────────┘
                      ▼
┌─────────────────────────────────────────────────────────┐
│  TTMetal (Runtime Operations)                           │
│  • Buffer management (create, allocate, deallocate)     │
│  • Program enqueue operations                           │
│  • Device I/O (read/write buffers)                      │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Hardware (TTNN + TTMetal Runtime)                      │
│  • Wormhole (N150, N300)                                │
│  • Blackhole (P150B)                                    │
└─────────────────────────────────────────────────────────┘
```

### Dialect Roles

| Dialect | Level | Purpose | Key Responsibility |
|---------|-------|---------|-------------------|
| **TTCore** | Foundation | Common infrastructure | Shared types, attributes, traits across all TT dialects |
| **TTIR** | High | Tensor operations | Platform-agnostic tensor operations and dispatch semantics |
| **TTNN** | Mid-High | Neural network ops | Device-aware NN operations with memory/layout management |
| **D2M** | Mid-Low | Generic dispatch | Direct-to-metal lowering with compute/datamovement regions |
| **TTKernel** | Low | Kernel operations | Tile-level FPU/SFPU operations on circular buffers |
| **TTMetal** | Lowest | Runtime operations | Buffer management, program enqueue, device I/O |
| **SFPI** | Low | SFPU instructions | 1:1 mapping to SFPU hardware instructions |

---

## Dialect Transformation Flow

The compilation pipeline transforms operations through these dialect conversions:

```
StableHLO → TTIR → TTNN → TTKernel/D2M → TTMetal → Flatbuffer
```

### Key Transformation Stages

1. **StableHLO to TTIR** (`--stablehlo-to-ttir-pipeline`)
   - Convert framework-level operations to platform-agnostic TTIR
   - Example: `stablehlo.add` → `ttir.add`

2. **TTIR Optimization** (Multiple passes)
   - Layout optimization, operation fusion, constant folding
   - Memory space assignment, sharding decisions

3. **TTIR to TTNN** (`--ttir-to-ttnn-backend-pipeline`)
   - Lower to device-aware operations
   - Insert memory configuration and layout operations
   - Example: `ttir.add` → `ttnn.add` with memory configs

4. **TTNN to TTMetal** (`--ttir-to-ttmetal-pipeline`)
   - Generate kernel operations via TTKernel/D2M
   - Create buffer management code
   - Schedule operations on device

5. **TTMetal to Flatbuffer** (`--ttmetal-to-flatbuffer`)
   - Serialize to runtime executable format

---

## Core Dialects

### TTCore - Common Types and Attributes

**Namespace:** `mlir::tt::ttcore`
**File Location:** `/home/ubuntu/work/tt/tt-mlir/include/ttmlir/Dialect/TTCore/IR/`

#### Purpose

TTCore provides the foundational types, attributes, and traits shared across all TT dialects. It does not define operations but serves as a common infrastructure layer.

#### Key Components

**Memory Spaces:**
- System memory (host)
- DRAM (device)
- L1 (on-chip SRAM)

**Data Types:**
- Standard types: `f32`, `f16`, `bf16`, `i32`, `i8`, `u8`
- BFloat8 variants: `bfp_bf8`, `bfp_bf4`, `bfp_bf2`

**Layout Attributes:**
- `MetalLayoutAttr`: Describes tensor layout in device memory
  - Tile shape, memory space, sharding configuration
  - Grid mapping (logical to physical cores)

**Traits:**
- `TTCoreDuplicateConstEvalTrait`: Mark ops to duplicate in const-eval subgraphs
- `TTCoreCreationOpTrait`: Mark tensor creation ops for const-eval hoisting
- `TTCoreNonCacheableTrait`: Mark non-cacheable ops (e.g., random number generation)

#### Example Attributes

```mlir
// Metal layout attribute specifying device memory layout
#layout = #ttcore.metal_layout<
  8192x128x1,           // Linear size (width x height x depth)
  undef,                // Tile grid offset
  <1x1>,                // Core grid shape
  memref<64x128xf32, #l1_>  // Memory reference with L1 space
>

// Memory space attributes
#system = #ttcore.memory_space<system>
#dram = #ttcore.memory_space<dram>
#l1_ = #ttcore.memory_space<l1>
```

---

### TTIR - High-Level Dispatch

**Namespace:** `mlir::tt::ttir`
**Dialect Name:** `ttir`
**File Location:** `/home/ubuntu/work/tt/tt-mlir/include/ttmlir/Dialect/TTIR/IR/`

#### Purpose

TTIR provides high-level semantics for dispatching work to Tenstorrent hardware. It defines declarative tensor operations that are largely agnostic to backend implementation details. This is the primary IR for optimization passes.

#### Design Principles

- **Platform-agnostic**: Operations work across different TT hardware generations
- **High-level semantics**: Tensor-level operations without hardware-specific details
- **Optimization-friendly**: Designed for compiler transformations and analysis
- **Destination-style**: Most ops implement `DestinationStyleOpInterface`

#### Key Operation Categories

**1. Layout and Memory Operations**
- `ttir.to_layout`: Transition tensors between layouts/memory spaces
- `ttir.ttnn_metal_layout_cast`: Cast between TTNN and Metal layout encodings
- `ttir.alloc`: Allocate tensor memory
- `ttir.dealloc`: Deallocate tensor memory

**2. Element-wise Operations**
- Unary: `abs`, `sqrt`, `exp`, `log`, `sin`, `cos`, `tanh`, `sigmoid`, `neg`, `relu`, `gelu`
- Binary: `add`, `subtract`, `multiply`, `divide`, `maximum`, `minimum`, `power`
- Comparison: `eq`, `ne`, `gt`, `ge`, `lt`, `le`
- Logical: `logical_and`, `logical_or`, `logical_not`, `logical_xor`

**3. Linear Algebra Operations**
- `ttir.dot_general`: Generalized matrix multiplication (batched, contracted dims)
- `ttir.matmul`: Standard matrix multiplication
- `ttir.transpose`: Permute tensor dimensions
- `ttir.reshape`: Change tensor shape
- `ttir.broadcast`: Broadcast tensor to larger shape
- `ttir.squeeze`: Remove singleton dimensions
- `ttir.unsqueeze`: Add singleton dimensions

**4. Reduction Operations**
- `ttir.sum`: Sum reduction along dimensions
- `ttir.mean`: Mean reduction
- `ttir.max`: Maximum reduction
- `ttir.min`: Minimum reduction

**5. Neural Network Operations**
- `ttir.conv2d`: 2D convolution
- `ttir.max_pool2d`: 2D max pooling
- `ttir.avg_pool2d`: 2D average pooling
- `ttir.softmax`: Softmax activation
- `ttir.layer_norm`: Layer normalization
- `ttir.batch_norm`: Batch normalization

**6. Control Flow and Utilities**
- `ttir.generic`: Generic linalg-style operation
- `ttir.constant`: Constant tensor creation
- `ttir.empty`: Allocate empty tensor
- `ttir.get_dimension_size`: Query tensor dimension

#### Types

- `ttir.mem_tx`: Memory transaction type (for DMA operations)
- `ttir.semaphore`: Semaphore primitive for core synchronization

#### Traits

- `TTIRInvolution`: Operation where `f(f(x)) = x` (e.g., `not(not(x))`)
- `TTIRIdempotence`: Operation where `f(f(x)) = f(x)` (e.g., `abs(abs(x))`)
- `TTIRBinaryIdempotence`: Binary operation where `f(x, x) = x` (e.g., `and(x, x)`)
- `Broadcastable`: Operation supports broadcasting semantics

#### Code Example

```mlir
// Simple addition with layout transitions
func.func @add(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // Allocate output tensor
  %0 = ttir.empty() : tensor<64x128xf32>

  // Perform element-wise addition
  %1 = ttir.add %arg0, %arg1, %0 :
    tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>
    -> tensor<64x128xf32>

  return %1 : tensor<64x128xf32>
}

// Matrix multiplication with layout specification
#layout = #ttcore.metal_layout<8192x128x1, undef, <1x1>, memref<64x128xf32, #l1_>>
func.func @matmul(%lhs: tensor<64x64xf32>, %rhs: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %out = ttir.empty() : tensor<64x128xf32>

  // Transition to device layout
  %lhs_device = ttir.to_layout %lhs, %lhs_out :
    tensor<64x64xf32> into tensor<64x64xf32, #layout>

  %rhs_device = ttir.to_layout %rhs, %rhs_out :
    tensor<64x128xf32> into tensor<64x128xf32, #layout>

  // Matmul operation
  %result = ttir.matmul %lhs_device, %rhs_device, %out :
    tensor<64x64xf32, #layout>, tensor<64x128xf32, #layout>, tensor<64x128xf32>
    -> tensor<64x128xf32, #layout>

  return %result : tensor<64x128xf32, #layout>
}
```

---

### TTNN - Neural Network Operations

**Namespace:** `mlir::tt::ttnn`
**Dialect Name:** `ttnn`
**File Location:** `/home/ubuntu/work/tt/tt-mlir/include/ttmlir/Dialect/TTNN/IR/`

#### Purpose

TTNN provides device-aware neural network operations that map closely to the TTNN library API. This dialect includes memory configuration, layout management, and device-specific optimizations.

#### Design Characteristics

- **Device-aware**: Explicit device management with `ttnn.device` type
- **Memory-aware**: All operations specify memory configuration
- **Layout-explicit**: Operations include layout (row-major, tile) specifications
- **Runtime-ready**: Operations map 1:1 to TTNN library calls

#### Key Operation Categories

**1. Device Management**
- `ttnn.get_device`: Acquire device handle with mesh configuration
- `ttnn.to_device`: Send tensor to device
- `ttnn.from_device`: Retrieve tensor from device

**2. Memory and Layout Operations**
- `ttnn.to_memory_config`: Change memory configuration (DRAM ↔ L1, interleaved ↔ sharded)
- `ttnn.to_layout`: Change tensor layout (row-major ↔ tile)
- `ttnn.typecast`: Convert data type on device
- `ttnn.to_dtype`: Convert data type on host

**3. Element-wise Unary Operations**

All inherit from `TTNN_ElementwiseUnaryOp` base class with optional `memory_config`:

- Activation functions: `abs`, `relu`, `gelu`, `sigmoid`, `tanh`, `silu`, `log_sigmoid`
- Math functions: `sqrt`, `rsqrt`, `exp`, `log`, `log1p`, `log2`, `log10`
- Trigonometric: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`
- Hyperbolic: `sinh`, `cosh`, `asinh`, `acosh`, `atanh`
- Rounding: `ceil`, `floor`, `round`, `trunc`
- Other: `neg`, `sign`, `reciprocal`, `expm1`

**4. Element-wise Binary Operations**

All inherit from `TTNN_ElementwiseBinaryOp` with `dtype` and `memory_config`:

- Arithmetic: `add`, `subtract`, `multiply`, `div`, `remainder`, `floor_div`
- Comparison: `eq`, `ne`, `gt`, `ge`, `lt`, `le`
- Logical: `logical_and`, `logical_or`, `logical_xor`
- Advanced: `maximum`, `minimum`, `power`, `squared_difference`
- Bitwise: `bitwise_and`, `bitwise_or`, `bitwise_xor`, `bitwise_not`

**5. Linear Algebra Operations**
- `ttnn.matmul`: Matrix multiplication with core grid configuration
- `ttnn.linear`: Fully connected layer (matmul + bias)
- `ttnn.embedding`: Embedding lookup

**6. Tensor Manipulation**
- `ttnn.reshape`: Change tensor shape
- `ttnn.transpose`: Permute dimensions
- `ttnn.concat`: Concatenate tensors
- `ttnn.slice`: Extract subtensor
- `ttnn.pad`: Pad tensor
- `ttnn.squeeze`: Remove dimensions
- `ttnn.unsqueeze`: Add dimensions
- `ttnn.repeat`: Repeat tensor along dimension
- `ttnn.permute`: Permute dimensions

**7. Reduction Operations**
- `ttnn.sum`: Sum reduction with keepdim support
- `ttnn.mean`: Mean reduction
- `ttnn.max`: Maximum reduction
- `ttnn.min`: Minimum reduction

**8. Convolution and Pooling**
- `ttnn.conv2d`: 2D convolution with extensive configuration
- `ttnn.max_pool2d`: 2D max pooling
- `ttnn.upsample`: Upsampling operation

**9. Normalization**
- `ttnn.softmax`: Softmax along dimension
- `ttnn.layer_norm`: Layer normalization
- `ttnn.group_norm`: Group normalization
- `ttnn.rms_norm`: RMS normalization

#### Types

- `ttnn.device`: Opaque device handle type

#### Attributes

**Memory Configuration** (`TTNN_MemoryConfigAttr`):
```mlir
#memory_config = #ttnn.memory_config<
  buffer_type = dram,         // dram, l1, l1_small, trace
  memory_layout = interleaved, // interleaved, single_bank, height_sharded, width_sharded, block_sharded
  shard_spec = <shape=[32, 64], grid=[1, 2]>  // Optional sharding specification
>
```

**Layout** (`TTNN_LayoutAttr`):
- `row_major`: Standard row-major layout
- `tile`: 32x32 tile layout (required for most operations)

**Sharding Configuration**:
- Interleaved: Data distributed across cores
- Height sharded: Sharding along height dimension
- Width sharded: Sharding along width dimension
- Block sharded: 2D sharding

#### Interfaces

- `TTNN_OpModelInterface`: Performance modeling for operations
- `TTNN_WorkaroundInterface`: Hardware workarounds
- `TTNN_MemoryConfigOpInterface`: Operations with memory configuration
- `TTNN_DtypeOpInterface`: Operations with dtype specification

#### Code Example

```mlir
// Device management and tensor operations
func.func @forward(%arg0: tensor<1x64x128xf32>, %arg1: tensor<128x256xf32>)
    -> tensor<1x64x256xf32> {
  // Get device handle
  %device = ttnn.get_device {mesh_shape=#ttnn.mesh_shape<1x1>} : !ttnn.device

  // Define memory config for L1 interleaved
  #mem_config = #ttnn.memory_config<buffer_type=l1, memory_layout=interleaved>

  // Send input to device with tiled layout
  %arg0_device = ttnn.to_device %arg0, %device {memory_config=#mem_config} :
    tensor<1x64x128xf32> -> tensor<1x64x128xf32>

  // Convert to tile layout (required for matmul)
  %arg0_tiled = ttnn.to_layout %arg0_device {layout=#ttnn.tile_layout} :
    tensor<1x64x128xf32> -> tensor<1x64x128xf32>

  // Prepare weight matrix
  %arg1_device = ttnn.to_device %arg1, %device {memory_config=#mem_config} :
    tensor<128x256xf32> -> tensor<128x256xf32>
  %arg1_tiled = ttnn.to_layout %arg1_device {layout=#ttnn.tile_layout} :
    tensor<128x256xf32> -> tensor<128x256xf32>

  // Matrix multiplication
  %result = ttnn.matmul %arg0_tiled, %arg1_tiled {
    memory_config=#mem_config,
    dtype=f32,
    core_grid=#ttnn.core_grid<2x2>
  } : tensor<1x64x128xf32>, tensor<128x256xf32> -> tensor<1x64x256xf32>

  // Apply activation
  %activated = ttnn.relu %result {memory_config=#mem_config} :
    tensor<1x64x256xf32> -> tensor<1x64x256xf32>

  // Retrieve from device
  %output = ttnn.from_device %activated : tensor<1x64x256xf32>

  return %output : tensor<1x64x256xf32>
}

// Element-wise operations with broadcasting
func.func @elementwise_ops(%input: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %device = ttnn.get_device : !ttnn.device
  #mem_config = #ttnn.memory_config<buffer_type=l1, memory_layout=interleaved>

  %input_device = ttnn.to_device %input, %device {memory_config=#mem_config} :
    tensor<64x128xf32> -> tensor<64x128xf32>

  // Unary operations
  %exp_result = ttnn.exp %input_device {memory_config=#mem_config} :
    tensor<64x128xf32> -> tensor<64x128xf32>

  %sqrt_result = ttnn.sqrt %exp_result {memory_config=#mem_config} :
    tensor<64x128xf32> -> tensor<64x128xf32>

  // Binary operation with scalar broadcast
  %scalar = ttnn.constant dense<2.0> : tensor<1xf32>
  %scalar_device = ttnn.to_device %scalar, %device {memory_config=#mem_config} :
    tensor<1xf32> -> tensor<1xf32>

  %scaled = ttnn.multiply %sqrt_result, %scalar_device {
    dtype=f32,
    memory_config=#mem_config
  } : tensor<64x128xf32>, tensor<1xf32> -> tensor<64x128xf32>

  %output = ttnn.from_device %scaled : tensor<64x128xf32>
  return %output : tensor<64x128xf32>
}
```

---

### D2M - Direct-to-Metal

**Namespace:** `mlir::tt::d2m`
**Dialect Name:** `d2m`
**File Location:** `/home/ubuntu/work/tt/tt-mlir/include/ttmlir/Dialect/D2M/IR/`

#### Purpose

The D2M (Direct-to-Metal) dialect provides the subset of operations used by the direct-to-metal lowering path (TTMetal). It hosts generic dispatch operations and region-based compute/datamovement operations required after the `D2MToD2MGeneric` transformation.

#### Design Characteristics

- **Generic dispatch**: Operations with compute and datamovement regions
- **Region-based**: Separates compute and data movement into distinct regions
- **Low-level**: Close to hardware execution model
- **Thread-aware**: Distinguishes between compute and datamovement threads

#### Key Components

**Generic Region Operations:**
Operations that contain regions for compute and datamovement:
- Compute region: Executes on MATH/FPU cores
- Datamovement region: Executes on data movement cores (READER/WRITER)

**Traits:**
- `D2MGenericRegionComputeOpTrait`: Marks ops inside compute region
- `D2MGenericRegionDatamovementOpTrait`: Marks ops inside datamovement region
- `D2MSkipOpEltwiseFusionTrait`: Skip element-wise fusion pass

#### Code Example

```mlir
// Generic dispatch operation with separate compute and datamovement regions
d2m.dispatch @matmul(%lhs: tensor<64x64xf32>, %rhs: tensor<64x128xf32>)
    -> tensor<64x128xf32> {
  // Datamovement region - handle I/O
  ^datamovement(%lhs_dm: tensor<64x64xf32>, %rhs_dm: tensor<64x128xf32>):
    %lhs_cb = d2m.read_buffer %lhs_dm : tensor<64x64xf32>
    %rhs_cb = d2m.read_buffer %rhs_dm : tensor<64x128xf32>
    d2m.yield %lhs_cb, %rhs_cb

  // Compute region - perform computation
  ^compute(%lhs_cb: !d2m.cb, %rhs_cb: !d2m.cb):
    %result = d2m.matmul %lhs_cb, %rhs_cb : !d2m.cb, !d2m.cb -> !d2m.cb
    d2m.yield %result

  // Write back region
  ^writeback(%result: !d2m.cb):
    %output = d2m.write_buffer %result : tensor<64x128xf32>
    d2m.yield %output
}
```

---

### TTKernel - Kernel Operations

**Namespace:** `mlir::tt::ttkernel`
**Dialect Name:** `ttkernel`
**File Location:** `/home/ubuntu/work/tt/tt-mlir/include/ttmlir/Dialect/TTKernel/IR/`

#### Purpose

TTKernel provides tile-level operations that execute on Tenstorrent hardware compute engines. It represents operations on circular buffers (CB) and destination (DST) registers, mapping closely to the compute kernel API.

#### Design Characteristics

- **Tile-based**: Operates on 32x32 tiles
- **Register-aware**: Explicit DST register management
- **Circular buffers**: Data flows through circular buffers
- **Hardware threads**: Separates MATH (compute) and PACK (output) threads

#### Key Operation Categories

**1. Hardware Initialization**
- `ttkernel.compute_kernel_hw_startup`: Initialize compute kernel hardware

**2. Register Management**

Register lifecycle for MATH thread:
- `ttkernel.tile_regs_acquire`: Acquire DST register lock (MATH thread)
- `ttkernel.tile_regs_commit`: Release DST register lock (MATH thread)

Register lifecycle for PACK thread:
- `ttkernel.tile_regs_wait`: Wait for DST register (PACK thread)
- `ttkernel.tile_regs_release`: Release DST register (PACK thread)

**3. Tile Data Movement**
- `ttkernel.copy_tile_init`: Initialize copy operation
- `ttkernel.copy_tile`: Copy tile from CB to DST
- `ttkernel.pack_tile`: Pack tile from DST to CB

**4. FPU Operations (Floating Point Unit)**

Operate on tiles using hardware FPU, all require init functions:

*Initialization:*
- `ttkernel.unary_op_init_common`: Init unary operations
- `ttkernel.binary_op_init_common`: Init binary operations
- `ttkernel.mm_init`: Init matrix multiplication

*Unary Operations:*
- `ttkernel.unary_bcast`: Broadcast tile

*Binary Operations:*
- `ttkernel.add_tiles`: Add two tiles
- `ttkernel.sub_tiles`: Subtract tiles
- `ttkernel.mul_tiles`: Multiply tiles
- `ttkernel.binary_dest_reuse_tiles`: Binary operation reusing destination

*Matrix Operations:*
- `ttkernel.matmul_tiles`: Matrix multiply tiles
- `ttkernel.mm_init_short`: Short matmul initialization
- `ttkernel.mm_block_init`: Block matmul initialization
- `ttkernel.experimental::matmul_block`: Experimental block matmul

*Reduction Operations:*
- `ttkernel.reduce_init`: Initialize reduction
- `ttkernel.reduce_tile`: Reduce tile along dimension
- `ttkernel.reduce_uninit`: Cleanup reduction state

**5. SFPU Operations (Special Function Processing Unit)**

Complex math operations using specialized hardware:

*Initialization:*
- `ttkernel.init_sfpu`: Initialize SFPU subsystem

*Transcendental Functions:*
- `ttkernel.exp_tile_init`, `ttkernel.exp_tile`: Exponential
- `ttkernel.log_tile_init`, `ttkernel.log_tile`: Natural logarithm
- `ttkernel.sqrt_tile_init`, `ttkernel.sqrt_tile`: Square root
- `ttkernel.rsqrt_tile_init`, `ttkernel.rsqrt_tile`: Reciprocal square root
- `ttkernel.recip_tile_init`, `ttkernel.recip_tile`: Reciprocal
- `ttkernel.power_tile_init`, `ttkernel.power_tile`: Power (scalar exponent)

*Trigonometric Functions:*
- `ttkernel.sin_tile_init`, `ttkernel.sin_tile`: Sine
- `ttkernel.cos_tile_init`, `ttkernel.cos_tile`: Cosine
- `ttkernel.tan_tile_init`, `ttkernel.tan_tile`: Tangent
- `ttkernel.tanh_tile_init`, `ttkernel.tanh_tile`: Hyperbolic tangent

*Other Functions:*
- `ttkernel.abs_tile_init`, `ttkernel.abs_tile`: Absolute value
- `ttkernel.negative_tile_init`, `ttkernel.negative_tile`: Negation
- `ttkernel.typecast_tile_init`, `ttkernel.typecast_tile`: Type conversion

*Rounding Operations:*
- `ttkernel.rounding_op_tile_init`: Initialize rounding
- `ttkernel.ceil_tile`: Ceiling
- `ttkernel.floor_tile`: Floor

*Binary SFPU Operations:*
- `ttkernel.add_binary_tile_init`, `ttkernel.add_binary_tile`
- `ttkernel.sub_binary_tile_init`, `ttkernel.sub_binary_tile`
- `ttkernel.mul_binary_tile_init`, `ttkernel.mul_binary_tile`
- `ttkernel.div_binary_tile_init`, `ttkernel.div_binary_tile`
- `ttkernel.power_binary_tile_init`, `ttkernel.power_binary_tile`
- `ttkernel.binary_max_tile_init`, `ttkernel.binary_max_tile`
- `ttkernel.binary_min_tile_init`, `ttkernel.binary_min_tile`

*Bitwise SFPU Operations:*
- `ttkernel.binary_bitwise_tile_init`: Initialize bitwise operations
- `ttkernel.bitwise_and_binary_tile`: Bitwise AND
- `ttkernel.bitwise_or_binary_tile`: Bitwise OR
- `ttkernel.bitwise_xor_binary_tile`: Bitwise XOR

*Scalar Operations:*
- `ttkernel.binop_with_scalar_tile_init`: Initialize scalar operations
- `ttkernel.add_unary_tile`: Add scalar to tile
- `ttkernel.mul_unary_tile`: Multiply tile by scalar
- `ttkernel.sub_unary_tile`: Subtract scalar from tile
- `ttkernel.div_unary_tile`: Divide tile by scalar

*Ternary Operations:*
- `ttkernel.clamp_tile_init`, `ttkernel.clamp_tile`: Clamp values
- `ttkernel.where_tile_init`, `ttkernel.where_tile`: Conditional select
- `ttkernel.where_fp32_tile`: FP32 conditional select

*Utility Operations:*
- `ttkernel.copy_dest_values_init`, `ttkernel.copy_dest_values`: Copy between DST slots

#### Types

- `ttkernel.cb`: Circular buffer type (opaque handle)

#### Traits

- `TTKernelFPUOpTrait`: Marks FPU operations
- `TTKernelSFPUOpTrait`: Marks SFPU operations
- `TTKernelInitOpTrait`: Marks initialization operations
- `TTKernelUnaryOpTrait`: Marks unary operations
- `TTKernelBinaryOpTrait`: Marks binary operations
- `TTKernelTernaryOpTrait`: Marks ternary operations

#### Code Example

```mlir
// Matrix multiplication kernel using FPU and circular buffers
func.func @matmul_kernel(%cb_in0: !ttkernel.cb, %cb_in1: !ttkernel.cb,
                         %cb_out: !ttkernel.cb) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // Initialize hardware
  ttkernel.compute_kernel_hw_startup(%cb_in0, %cb_in1, %cb_out) :
    (!ttkernel.cb, !ttkernel.cb, !ttkernel.cb) -> ()

  // Initialize matmul operation
  ttkernel.mm_init(%cb_in0, %cb_in1, %cb_out) :
    (!ttkernel.cb, !ttkernel.cb, !ttkernel.cb) -> ()

  // Acquire DST register for computation
  ttkernel.tile_regs_acquire() : () -> ()

  // Copy input tiles from CB to DST
  ttkernel.copy_tile_init(%cb_in0) : (!ttkernel.cb) -> ()
  ttkernel.copy_tile(%cb_in0, %c0, %c0) : (!ttkernel.cb, index, index) -> ()
  ttkernel.copy_tile(%cb_in1, %c0, %c1) : (!ttkernel.cb, index, index) -> ()

  // Perform matrix multiplication
  ttkernel.matmul_tiles(%c0, %c1, %c0) : (index, index, index) -> ()

  // Commit DST register (MATH thread done)
  ttkernel.tile_regs_commit() : () -> ()

  // Wait for DST register (PACK thread)
  ttkernel.tile_regs_wait() : () -> ()

  // Pack result from DST to output CB
  ttkernel.pack_tile(%c0, %cb_out, %c0, false) :
    (index, !ttkernel.cb, index, i1) -> ()

  // Release DST register (PACK thread done)
  ttkernel.tile_regs_release() : () -> ()

  return
}

// SFPU operation kernel (element-wise exp)
func.func @exp_kernel(%cb_in: !ttkernel.cb, %cb_out: !ttkernel.cb) {
  %c0 = arith.constant 0 : index

  // Initialize SFPU
  ttkernel.init_sfpu() : () -> ()
  ttkernel.exp_tile_init() : () -> ()

  ttkernel.tile_regs_acquire() : () -> ()

  // Copy tile to DST
  ttkernel.copy_tile(%cb_in, %c0, %c0) : (!ttkernel.cb, index, index) -> ()

  // Apply exponential using SFPU
  ttkernel.exp_tile(%c0) : (index) -> ()

  ttkernel.tile_regs_commit() : () -> ()
  ttkernel.tile_regs_wait() : () -> ()

  // Pack result
  ttkernel.pack_tile(%c0, %cb_out, %c0, false) :
    (index, !ttkernel.cb, index, i1) -> ()

  ttkernel.tile_regs_release() : () -> ()
  return
}
```

---

### TTMetal - Low-Level Runtime

**Namespace:** `mlir::tt::ttmetal`
**Dialect Name:** `ttmetal`
**File Location:** `/home/ubuntu/work/tt/tt-mlir/include/ttmlir/Dialect/TTMetal/IR/`

#### Purpose

TTMetal provides the lowest-level runtime operations for buffer management, program execution, and device I/O. These operations map directly to the TTMetal runtime API.

#### Key Operations

**1. Buffer Management**
- `ttmetal.create_buffer`: Allocate device buffer at specified address
- `ttmetal.deallocate_buffer`: Free device buffer

**2. Program Execution**
- `ttmetal.enqueue_program`: Enqueue kernel program with buffers and circular buffers
  - Specifies kernel configurations (compute, data reader, data writer)
  - Maps buffers to circular buffer ports
  - Optional fabric connection configuration for multi-chip

**3. Data Transfer**
- `ttmetal.enqueue_write_buffer`: Write data from host to device buffer
- `ttmetal.enqueue_read_buffer`: Read data from device buffer to host

**4. Synchronization**
- `ttmetal.finish`: Wait for all queued operations to complete (global barrier)

**5. Multi-Chip Operations**
- `ttmetal.mesh_shard`: N-dimensional sharding or concatenation
  - `ShardToFull`: Shard tensor across chips
  - `FullToShard`: Concatenate shards from chips

#### Attributes

**Kernel Configuration** (`TTMetal_KernelConfigAttr`):
Specifies kernel file paths and core assignments:
```mlir
#kernel_config = #ttmetal.kernel_config<
  kernel_path = "kernels/compute_kernel.cpp",
  core_range = [[0, 0], [1, 1]],  // Start and end cores
  kernel_type = compute  // compute, data_reader, data_writer
>
```

**Fabric Connection Configuration**:
For multi-chip communication via fabric network.

#### Code Example

```mlir
// Complete program execution with buffer management
func.func @simple_add(%arg0: memref<64x128xf32, #dram>,
                      %arg1: memref<64x128xf32, #dram>)
    -> memref<64x128xf32, #dram> {
  // Create device buffers
  %addr0 = arith.constant 0x10000 : i64
  %addr1 = arith.constant 0x20000 : i64
  %addr_out = arith.constant 0x30000 : i64

  %buf0 = ttmetal.create_buffer {address = %addr0} : memref<64x128xf32, #dram>
  %buf1 = ttmetal.create_buffer {address = %addr1} : memref<64x128xf32, #dram>
  %buf_out = ttmetal.create_buffer {address = %addr_out} : memref<64x128xf32, #dram>

  // Write input data to device
  ttmetal.enqueue_write_buffer %arg0, %buf0 :
    memref<64x128xf32, #dram>, memref<64x128xf32, #dram>
  ttmetal.enqueue_write_buffer %arg1, %buf1 :
    memref<64x128xf32, #dram>, memref<64x128xf32, #dram>

  // Create circular buffers
  %cb0 = memref.alloc() : memref<2x32x32xf32, #l1>  // Input CB 0
  %cb1 = memref.alloc() : memref<2x32x32xf32, #l1>  // Input CB 1
  %cb_out = memref.alloc() : memref<2x32x32xf32, #l1>  // Output CB

  // Define kernel configurations
  #reader0_config = #ttmetal.kernel_config<
    kernel_path = "kernels/reader_unary.cpp",
    core_range = [[0, 0], [0, 0]],
    kernel_type = data_reader
  >

  #reader1_config = #ttmetal.kernel_config<
    kernel_path = "kernels/reader_unary.cpp",
    core_range = [[0, 0], [0, 0]],
    kernel_type = data_reader
  >

  #compute_config = #ttmetal.kernel_config<
    kernel_path = "kernels/eltwise_binary.cpp",
    core_range = [[0, 0], [1, 1]],
    kernel_type = compute
  >

  #writer_config = #ttmetal.kernel_config<
    kernel_path = "kernels/writer_unary.cpp",
    core_range = [[0, 0], [0, 0]],
    kernel_type = data_writer
  >

  // Enqueue program with buffers and CBs
  ttmetal.enqueue_program
    buffers(%buf0, %buf1, %buf_out)
    cbs(%cb0, %cb1, %cb_out)
    cb_ports = [0, 1, 16]  // CB port assignments
    kernel_configs = [#reader0_config, #reader1_config, #compute_config, #writer_config]

  // Read result back to host
  %result = memref.alloc() : memref<64x128xf32, #system>
  ttmetal.enqueue_read_buffer %buf_out, %result :
    memref<64x128xf32, #dram>, memref<64x128xf32, #system>

  // Wait for completion
  ttmetal.finish

  // Cleanup
  ttmetal.deallocate_buffer %buf0
  ttmetal.deallocate_buffer %buf1
  ttmetal.deallocate_buffer %buf_out

  return %result : memref<64x128xf32, #system>
}

// Multi-chip mesh sharding
func.func @mesh_shard_example(%input: memref<128x256xf32, #system>)
    -> memref<64x256xf32, #system> {
  // Shard input across 2 chips (height-wise)
  %sharded = ttmetal.mesh_shard %input {
    shard_type = #ttcore.shard_type<shard_to_full>,
    shard_direction = #ttcore.shard_direction<height>,
    shard_shape = [64, 256],
    shard_dims = [0]
  } : memref<128x256xf32, #system> -> memref<64x256xf32, #system>

  return %sharded : memref<64x256xf32, #system>
}
```

---

### SFPI - SFPU Programming Interface

**Namespace:** `mlir::tt::sfpi`
**Dialect Name:** `sfpi`
**File Location:** `/home/ubuntu/work/tt/tt-mlir/include/ttmlir/Dialect/SFPI/IR/`

#### Purpose

SFPI (SFPU Programming Interface) provides a 1:1 mapping to Tenstorrent SFPU (Special Function Processing Unit) hardware instructions. This dialect enables direct access to SFPU capabilities through MLIR operations, working with 4x8 tile-based floating-point vectors.

#### Design Characteristics

- **Hardware-direct**: Direct mapping to SFPU instructions (see `rvtt-insn.h`)
- **Vector-based**: Operates on 4x8 floating-point vectors
- **Low-level**: Exposes fine-grained hardware control
- **ISA-aligned**: Corresponds to Tenstorrent ISA documentation

#### Key Operation Categories

**1. Unary Operations** (inherit `SFPI_UnaryOpTrait`)
Mathematical functions on single vector operand

**2. Binary Operations** (inherit `SFPI_BinaryOpTrait`)
Operations on two vector operands

**3. Ternary Operations** (inherit `SFPI_TernaryOpTrait`)
Operations on three vector operands (e.g., fused multiply-add)

**4. Comparison Operations** (inherit `SFPI_ComparisonOpTrait`)
Vector comparison operations producing masks

**5. Conversion Operations** (inherit `SFPI_ConversionOpTrait`)
Type conversions and data format changes

#### Traits

- `SFPIUnaryOpTrait`: Marks unary SFPI operations
- `SFPIBinaryOpTrait`: Marks binary SFPI operations
- `SFPITernaryOpTrait`: Marks ternary SFPI operations
- `SFPIComparisonOpTrait`: Marks comparison operations
- `SFPIConversionOpTrait`: Marks conversion operations

#### Usage Context

SFPI operations are typically generated during the TTKernel lowering phase when SFPU-capable operations are decomposed to hardware instructions. They represent the actual instruction stream that will execute on the SFPU hardware.

#### Documentation Reference

For detailed SFPU instruction documentation, refer to:
- [tt-isa-documentation](https://github.com/tenstorrent/tt-isa-documentation)
- SFPI GCC backend: `rvtt-insn.h`

#### Code Example

```mlir
// Example SFPI instruction sequence for vector exponential
// Note: This is typically compiler-generated, not hand-written

func.func @sfpi_exp_vector(%vec: vector<4x8xf32>) -> vector<4x8xf32> {
  // SFPI exponential approximation using hardware instructions
  // (Actual instruction sequence depends on hardware generation)

  // Load vector into SFPU registers
  %reg = sfpi.load %vec : vector<4x8xf32>

  // Perform exponential using SFPU hardware
  %result_reg = sfpi.exp %reg : vector<4x8xf32>

  // Store result back
  %result = sfpi.store %result_reg : vector<4x8xf32>

  return %result : vector<4x8xf32>
}
```

---

## Dialect Details and Operations

### Operation Naming Conventions

All TT dialects follow MLIR naming conventions:

- **Dialect prefix**: `ttir.`, `ttnn.`, `ttkernel.`, `ttmetal.`, `sfpi.`
- **Operation names**: Snake_case (e.g., `to_layout`, `matmul_tiles`)
- **Type names**: CamelCase (e.g., `Device`, `CircularBuffer`)
- **Attribute names**: Snake_case (e.g., `memory_config`, `core_grid`)

### Common Operation Patterns

**Destination-Style Operations (DPS):**
Many TTIR operations follow the destination-style pattern where the output tensor is passed as an operand:

```mlir
%output_storage = ttir.empty() : tensor<64x128xf32>
%result = ttir.add %lhs, %rhs, %output_storage :
  tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>
  -> tensor<64x128xf32>
```

**Memory Effects:**
- `Pure`: No side effects (most element-wise ops)
- `MemoryEffects<[MemRead]>`: Reads memory
- `MemoryEffects<[MemWrite]>`: Writes memory
- `MemoryEffects<[MemAlloc]>`: Allocates memory
- `MemoryEffects<[MemFree]>`: Frees memory

**Interfaces:**
- `OpAsmOpInterface`: Custom assembly format
- `InferTypeOpInterface`: Type inference
- `DestinationStyleOpInterface`: DPS operations
- `SideEffectInterface`: Memory effect specification

---

## Type System

### Tensor Types

**MLIR Built-in:**
```mlir
tensor<64x128xf32>              // Ranked tensor, f32 element type
tensor<*xf32>                   // Unranked tensor
tensor<64x128xf32, #encoding>   // Tensor with layout encoding
```

**Layout Encodings:**

**TTCore Metal Layout:**
```mlir
#layout = #ttcore.metal_layout<
  linear_size,      // Width x Height x Depth
  tile_offset,      // Tile grid offset (or 'undef')
  core_grid,        // <rows x cols>
  memref_type       // memref<...> with memory space
>
```

**TTNN Layout:**
```mlir
#ttnn_layout = #ttnn.ttnn_layout<
  (d0, d1) -> (d0, d1),  // Layout map
  memory_config,          // Memory configuration
  memref<64x128xf32, #l1>
>
```

### Memory Reference Types

```mlir
memref<64x128xf32, #system>     // System (host) memory
memref<64x128xf32, #dram>       // Device DRAM
memref<64x128xf32, #l1>         // Device L1 (SRAM)
```

### Dialect-Specific Types

| Dialect | Type | Purpose |
|---------|------|---------|
| **TTIR** | `ttir.mem_tx` | Memory transaction handle |
| **TTIR** | `ttir.semaphore` | Core synchronization primitive |
| **TTNN** | `ttnn.device` | Device handle |
| **TTKernel** | `ttkernel.cb` | Circular buffer handle |

### Attribute Types

**Memory Configuration:**
```mlir
#memory_config = #ttnn.memory_config<
  buffer_type = l1,              // system, dram, l1, l1_small, trace
  memory_layout = interleaved,    // interleaved, single_bank, height_sharded, etc.
  shard_spec = <shape=[32, 64], grid=[1, 2]>
>
```

**Mesh Configuration:**
```mlir
#mesh_shape = #ttnn.mesh_shape<rows x cols>
#mesh_offset = #ttnn.mesh_offset<row, col>
```

**Core Grid:**
```mlir
#core_grid = #ttnn.core_grid<rows x cols>
```

**Data Types:**
```mlir
#dtype = #ttcore.data_type<f32>
#dtype = #ttcore.data_type<bf16>
#dtype = #ttcore.data_type<bfp_bf8>
```

---

## Code Examples

### End-to-End Example: Simple Neural Network Layer

This example shows a complete transformation from TTIR to TTMetal for a simple linear layer.

See the comprehensive examples in the main document sections above.

---

## Summary

The TT-MLIR dialect system provides a comprehensive multi-level IR for compiling ML models to Tenstorrent hardware:

1. **TTCore**: Foundation layer with common types and attributes
2. **TTIR**: High-level, platform-agnostic tensor operations for optimization
3. **TTNN**: Device-aware neural network operations with memory management
4. **D2M**: Generic dispatch with compute/datamovement separation
5. **TTKernel**: Tile-level operations on circular buffers (FPU/SFPU)
6. **TTMetal**: Low-level runtime operations (buffers, program execution)
7. **SFPI**: Direct hardware instruction mapping for SFPU

Each dialect serves a specific purpose in the compilation pipeline, enabling progressive lowering from high-level ML semantics to hardware-executable code. The dialects are designed with clear separation of concerns, enabling effective optimization at each level while maintaining composability.

For more detailed information:
- Source files: `/home/ubuntu/work/tt/tt-mlir/include/ttmlir/Dialect/`
- Test cases: `/home/ubuntu/work/tt/tt-mlir/test/ttmlir/`
- Pipeline documentation: See `01-stablehlo-to-ttir.md` and other pipeline docs

---

## Appendix: Operation Quick Reference

### TTIR Operations Summary

| Category | Operations |
|----------|------------|
| **Layout** | `to_layout`, `ttnn_metal_layout_cast`, `alloc`, `dealloc` |
| **Arithmetic** | `add`, `subtract`, `multiply`, `divide`, `remainder` |
| **Math** | `abs`, `sqrt`, `exp`, `log`, `sin`, `cos`, `tanh`, `sigmoid` |
| **Linear Algebra** | `matmul`, `dot_general`, `transpose`, `reshape` |
| **Reduction** | `sum`, `mean`, `max`, `min` |
| **Comparison** | `eq`, `ne`, `gt`, `ge`, `lt`, `le` |
| **Logical** | `logical_and`, `logical_or`, `logical_not`, `logical_xor` |
| **NN Ops** | `conv2d`, `max_pool2d`, `softmax`, `layer_norm` |

### TTNN Operations Summary

| Category | Operations |
|----------|------------|
| **Device** | `get_device`, `to_device`, `from_device` |
| **Memory** | `to_memory_config`, `to_layout`, `typecast`, `to_dtype` |
| **Arithmetic** | `add`, `subtract`, `multiply`, `div`, `power`, `maximum`, `minimum` |
| **Math** | `abs`, `sqrt`, `exp`, `log`, `reciprocal`, `sign` |
| **Trigonometric** | `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh` |
| **Activation** | `relu`, `gelu`, `sigmoid`, `tanh`, `silu`, `log_sigmoid` |
| **Linear Algebra** | `matmul`, `linear`, `embedding`, `transpose` |
| **Tensor Ops** | `reshape`, `concat`, `slice`, `pad`, `squeeze`, `unsqueeze` |
| **Reduction** | `sum`, `mean`, `max`, `min` |
| **Conv/Pool** | `conv2d`, `max_pool2d`, `upsample` |
| **Normalization** | `softmax`, `layer_norm`, `group_norm`, `rms_norm` |

### TTKernel Operations Summary

| Category | Operations |
|----------|------------|
| **Hardware Init** | `compute_kernel_hw_startup` |
| **Register Mgmt** | `tile_regs_acquire`, `tile_regs_commit`, `tile_regs_wait`, `tile_regs_release` |
| **Tile Movement** | `copy_tile_init`, `copy_tile`, `pack_tile` |
| **FPU Binary** | `add_tiles`, `sub_tiles`, `mul_tiles`, `matmul_tiles` |
| **FPU Reduce** | `reduce_init`, `reduce_tile`, `reduce_uninit` |
| **SFPU Math** | `exp_tile`, `log_tile`, `sqrt_tile`, `rsqrt_tile`, `recip_tile` |
| **SFPU Trig** | `sin_tile`, `cos_tile`, `tan_tile`, `tanh_tile` |
| **SFPU Binary** | `add_binary_tile`, `mul_binary_tile`, `div_binary_tile` |
| **SFPU Util** | `typecast_tile`, `abs_tile`, `negative_tile`, `ceil_tile`, `floor_tile` |

### TTMetal Operations Summary

| Category | Operations |
|----------|------------|
| **Buffers** | `create_buffer`, `deallocate_buffer` |
| **Programs** | `enqueue_program` |
| **Data Transfer** | `enqueue_write_buffer`, `enqueue_read_buffer` |
| **Synchronization** | `finish` |
| **Multi-Chip** | `mesh_shard` |
