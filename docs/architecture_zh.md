# 架构概述

本文档提供 mHC Ascend 实现的整体架构概述。

---

## 1. 什么是 mHC？

**mHC（流形约束超连接，Manifold-Constrained Hyper-Connections）** 是 DeepSeek-AI 提出的一种技术，用于替代 Transformer 中的标准残差连接。mHC 不使用简单的加法（`output = residual + branch`），而是使用可学习的双随机混合矩阵来组合多个信息流。

核心组件：
- **Sinkhorn-Knopp 归一化**：确保混合矩阵是双随机的
- **动态 H（Dynamic-H）**：通过学习的投影计算输入相关的混合权重
- **多流聚合与分发**：在多个流之间组合和重新分配信息

参考文献：[Hyper-Connections 论文](https://arxiv.org/abs/2409.19606)（DeepSeek-AI，2024）

---

## 2. 项目结构

```
mHC_ascend/
├── src/
│   ├── csrc/                      # C++ 源代码
│   │   ├── include/               # 头文件
│   │   │   └── mhc_types.h        # Tiling 结构体和枚举
│   │   └── kernels/               # Ascend C 内核实现
│   │       ├── sinkhorn_knopp.cpp # 双随机归一化
│   │       ├── rmsnorm.cpp        # RMS 归一化
│   │       ├── stream_ops.cpp     # 流聚合/分发内核
│   │       ├── fused_ops.cpp      # 融合 RMSNorm + MatMul
│   │       ├── compute_rms.cpp    # 独立 RMS 计算
│   │       └── mhc_layer.cpp      # （遗留）层级内核
│   │
│   └── python/
│       ├── bindings.cpp           # PyTorch ↔ C++ 绑定（pybind11）
│       └── mhc/                   # Python 包
│           ├── __init__.py        # 公共 API 导出
│           ├── layer.py           # MHCLayer, MHCResidualWrapper
│           ├── ops.py             # Autograd 函数和算子封装
│           └── ops_pure_pytorch_backward.py  # 纯 PyTorch 回退实现
│
├── scripts/
│   ├── build.sh                   # 编译脚本
│   ├── install.sh                 # 安装脚本
│   └── accuracy_check.py          # 数值精度测试
│
├── CMakeLists.txt                 # CMake 编译配置
├── setup.py                       # Python 包设置
└── pyproject.toml                 # Python 项目元数据
```

---

## 3. 分层架构

实现采用三层架构：

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Python 用户接口                               │
│  MHCLayer, MHCResidualWrapper, sinkhorn_knopp(), rmsnorm() 等       │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      C++ 绑定层（bindings.cpp）                      │
│  - 类型转换（BFloat16 ↔ Float32）                                   │
│  - Tiling 参数计算                                                   │
│  - 通过 ACLNN API 启动内核                                          │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Ascend C 内核（.cpp 文件）                        │
│  - AI 核上的底层计算                                                 │
│  - 内存管理（全局内存 GM ↔ 统一缓冲区 UB）                           │
│  - 向量化和多核并行                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.1 Python 层

**目的**：面向用户的 API、自动微分支持、高层编排。

关键类：
- `MHCLayer`：核心 mHC 计算模块
- `MHCResidualWrapper`：残差连接的直接替换
- Autograd 函数：`SinkhornKnoppFunction`、`RMSNormFunction` 等

### 3.2 C++ 绑定层

**目的**：PyTorch 张量与 Ascend 内核之间的桥梁。

职责：
- 将 PyTorch 张量转换为原始指针
- 根据输入形状计算 tiling 参数
- 处理类型转换（如将 BFloat16 输入转换为 Float32 供内核使用）
- 通过 ACLNN 运行时 API 启动内核

### 3.3 Ascend C 内核层

**目的**：在华为 Ascend NPU 上进行高性能计算。

特点：
- 使用 Ascend C 编写（Ascend NPU 的 C++ 扩展）
- 在全局内存（GM）和统一缓冲区（UB）之间显式管理内存
- 支持向量化（SIMD 操作）和多核并行

---

## 4. 算子组织

### 4.1 独立算子

具有独立 Python 接口的单个算子，适用于单元测试和独立使用。

| 算子 | Forward 内核 | Backward 内核 | Python 接口 |
|----------|---------------|-----------------|------------------|
| Sinkhorn-Knopp | `sinkhorn_knopp_fwd` | `sinkhorn_knopp_bwd` | `sinkhorn_knopp()` |
| RMSNorm | `rmsnorm_fwd` | `rmsnorm_bwd` | `rmsnorm()` |
| Stream Aggregate | `stream_aggregate_fwd` | `stream_aggregate_bwd` | `stream_aggregate()` |
| Stream Distribute Mix Add | `stream_distribute_mix_add_fwd` | `stream_distribute_mix_add_bwd` | `stream_distribute_mix_add()` |
| Compute RMS | `compute_rms_fwd` | N/A | `compute_rms()` |

### 4.2 融合层算子

在 C++ 层级编排，以提高训练效率。多个内核顺序调用，无需在调用之间返回 Python。

| C++ 函数 | 内部内核调用 |
|-------------|----------------------|
| `mhc_layer_fwd_dynamic` | compute_rms → matmul → sinkhorn → stream_aggregate → rmsnorm → stream_distribute |
| `mhc_layer_bwd_dynamic` | stream_distribute_bwd → rmsnorm_bwd → stream_aggregate_bwd → sinkhorn_bwd + ATen 操作 |

**优势**：减少 Python-C++ 边界跨越开销，中间张量保留在 C++ 层。

---

## 5. 数据流

### 5.1 MHCLayer 前向传播（Dynamic-H 模式）

```
输入: x [B, n, C]
       │
       ▼
┌──────────────────┐
│  展平为          │ x_flat [B, n*C]
│  [B, n*C]        │
└──────────────────┘
       │
       ├───────────────────────────────────────┐
       ▼                                       ▼
┌──────────────────┐                   ┌──────────────────┐
│  计算 RMS        │ rms [B]           │  与 phi_concat   │ H_proj [B, 2n + n²]
│                  │                   │  做矩阵乘法      │
└──────────────────┘                   └──────────────────┘
       │                                       │
       │                     ┌─────────────────┼─────────────────┐
       │                     ▼                 ▼                 ▼
       │              H_pre [B, n]      H_post [B, n]      H_res [B, n, n]
       │                     │                 │                 │
       │                     ▼                 │                 ▼
       │              ┌──────────────┐         │          ┌──────────────┐
       │              │ α * H / rms  │         │          │ exp(H_res)   │
       │              │ + b_pre      │         │          │              │
       │              └──────────────┘         │          └──────────────┘
       │                     │                 │                 │
       │                     ▼                 │                 ▼
       │              ┌──────────────┐         │          ┌──────────────┐
       └──────────────│ 流聚合      │         │          │ Sinkhorn     │ M [B, n, n]
                      │             │         │          │ 归一化       │
                      └──────────────┘         │          └──────────────┘
                             │                 │                 │
                             ▼                 │                 │
                      x_agg [B, C]             │                 │
                             │                 │                 │
                             ▼                 │                 │
                      ┌──────────────┐         │                 │
                      │ RMSNorm     │          │                 │
                      └──────────────┘         │                 │
                             │                 │                 │
                             ▼                 ▼                 ▼
                      y_norm [B, C]     H_post [B, n]      M [B, n, n]
                             │                 │                 │
                             └─────────────────┼─────────────────┘
                                               ▼
                                    ┌──────────────────────┐
                                    │ 流分发混合加法        │
                                    │                      │
                                    └──────────────────────┘
                                               │
                                               ▼
                                    输出: [B, n, C]
```

---

## 6. 与 CUDA 参考实现的关系

本实现是原始 CUDA 实现（仓库根目录中的 `src/` 目录）的移植版本。

| 方面 | CUDA | Ascend |
|--------|------|--------|
| 硬件 | NVIDIA GPU | 华为 Ascend NPU |
| 语言 | CUDA C++ | Ascend C（C++ 扩展） |
| 并行模型 | 线程块、warp | AI 核、向量化 |
| 内存 | 共享内存、寄存器 | 统一缓冲区（UB）、全局内存（GM） |
| 向量化 | 原生 float4 类型 | Mask API（SetMaskCount 等） |

**移植挑战**：
1. **32 字节对齐**：Ascend 的 `DataCopy` 要求 32B 对齐传输
2. **多核干扰**：Mask 操作是全局状态，非线程私有
3. **无原生 float4**：必须使用显式 SIMD 操作或标量回退

---

## 7. 构建系统

项目使用 CMake 配合 Ascend CANN 工具包：

```bash
# 编译
cd mHC_ascend
bash scripts/build.sh

# 安装 Python 包
pip install -e .
```

**主要构建依赖**：
- CANN 工具包（`/usr/local/Ascend/ascend-toolkit/latest`）
- 带有 torch_npu 的 PyTorch
- pybind11（用于 Python 绑定）

**输出**：`mhc_ascend.cpython-*.so`（Python 扩展模块）

---

## 8. 设计决策

### 8.1 为什么使用融合层操作？

训练涉及重复的前向-后向传播。在 C++ 中融合多个内核调用可以减少：
- Python 解释器开销
- 张量分配/释放
- 内存带宽（中间结果保留在更快的内存中）

### 8.2 为什么保留独立算子？

独立算子支持：
- 单个组件的单元测试
- 隔离调试数值问题
- 在不需要完整 MHCLayer 的自定义模型中使用

### 8.3 为什么需要多条实现路径？

由于 Ascend 硬件约束（对齐、多核干扰），不同的输入配置需要不同的代码路径：
- **标量路径**：安全的回退，适用于所有输入
- **向量化路径**：更快，但有对齐/多核约束
- **特化路径**：如 N=4 优化的 Sinkhorn 反向传播

实现会根据输入特征自动选择最佳路径。
