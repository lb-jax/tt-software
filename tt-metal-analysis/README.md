# TT-Metal 模拟器适配分析

本目录用于记录 tt-metal 仓库的分析过程，目标是理解其架构并找到模拟器的接入点。

## 目标

将 Tenstorrent 的 tt-metal 硬件抽象层修改为支持自定义模拟器，使得：
- tt-metal 能够检测到"虚拟"的 Tenstorrent 设备
- 计算指令能够被转发到模拟器执行
- 模拟器返回的结果能够被正确处理

## 当前问题

运行 Demo 时报错：
```
TT_FATAL @ tt_cluster.cpp:117: num_chips > 0
info: No chips detected in the cluster
```

原因：没有物理 Tenstorrent 硬件，tt-metal 的设备检测失败。

## 分析计划

| 阶段 | 内容 | 状态 | 文档 |
|------|------|------|------|
| 1 | tt-metal 仓库结构概览 | ✅ 完成 | `01-repo-structure.md` |
| 2 | 设备检测流程分析 | ✅ 完成 | `02-device-detection.md` |
| 3 | UMD 层接口分析 | ✅ 完成 | `03-umd-interface.md` |
| 4 | 命令提交流程 | ⏳ 待分析 | `04-command-submission.md` |
| 5 | 内存管理 | ⏳ 待分析 | `05-memory-management.md` |
| 6 | 模拟器接口设计 | ⏳ 待分析 | `06-simulator-interface.md` |

## 关键文件位置

tt-metal 源码位于 tt-xla 构建目录中：
```
/home/ubuntu/work/tt/tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/
```

### 已确认的关键文件

| 文件 | 作用 |
|------|------|
| `tt_metal/llrt/tt_cluster.cpp:117` | 设备检测失败的位置 |
| `tt_metal/llrt/` | Low-Level Runtime |
| `tt_metal/third_party/umd/` | User Mode Driver (设备抽象) |

## 调用链路 (从错误堆栈分析)

```
torch_xla._XLAC._xla_get_default_device()
    │
    ▼
torch_xla::runtime::PjRtComputationClient::Initialize()
    │
    ▼
tt::pjrt::ClientInstance::populateDevices()
    │
    ▼
tt::runtime::getCurrentSystemDesc()
    │
    ▼
tt::tt_metal::GetNumAvailableDevices()
    │
    ▼
tt::tt_metal::MetalContext::instance()
    │
    ▼
tt::tt_metal::MetalContext::initialize_base_objects()
    │
    ▼
tt::Cluster::get_cluster_type_from_cluster_desc()
    │
    ▼
❌ assert(num_chips > 0) 失败
```

## 模拟器接入策略 (初步设想)

### 策略一：替换 UMD 层
- 实现自己的 `libdevice.so`
- 模拟设备检测、内存分配、命令提交
- 难度：中等，需要理解 UMD 接口

### 策略二：修改 tt_cluster.cpp
- 在设备检测逻辑中添加模拟器分支
- 当检测到特定环境变量时，返回模拟设备
- 难度：较低，但侵入性强

### 策略三：实现虚拟 PCIe 设备
- 在系统层面模拟 Tenstorrent PCIe 设备
- 对 tt-metal 完全透明
- 难度：较高，需要内核模块

## 环境变量

当前设置：
```bash
ARCH_NAME=wormhole_b0       # 目标架构
TT_METAL_HOME=(未设置)       # tt-metal 根目录
```

## 参考资料

- [TT-Forge 架构分析报告](/home/ubuntu/work/tt/TT-FORGE-ARCHITECTURE.md)
- [tt-metal GitHub](https://github.com/tenstorrent/tt-metal)

---

*创建时间: 2025-02*
