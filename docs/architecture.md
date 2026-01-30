# Architecture Overview

This document provides a high-level overview of the mHC Ascend implementation architecture.

---

## 1. What is mHC?

**mHC (Manifold-Constrained Hyper-Connections)** is a technique proposed by DeepSeek-AI to replace standard residual connections in Transformers. Instead of simple addition (`output = residual + branch`), mHC uses a learnable, doubly-stochastic mixing matrix to combine multiple information streams.

Key components:
- **Sinkhorn-Knopp normalization**: Ensures the mixing matrix is doubly-stochastic
- **Dynamic-H**: Input-dependent mixing weights computed via learned projections
- **Multi-stream aggregation and distribution**: Combines and redistributes information across streams

Reference: [Hyper-Connections paper](https://arxiv.org/abs/2409.19606) (DeepSeek-AI, 2024)

---

## 2. Project Structure

```
mHC_ascend/
├── src/
│   ├── csrc/                      # C++ source code
│   │   ├── include/               # Header files
│   │   │   └── mhc_types.h        # Tiling structures and enums
│   │   └── kernels/               # Ascend C kernel implementations
│   │       ├── sinkhorn_knopp.cpp # Doubly-stochastic normalization
│   │       ├── rmsnorm.cpp        # RMS normalization
│   │       ├── stream_ops.cpp     # Stream aggregate/distribute kernels
│   │       ├── fused_ops.cpp      # Fused RMSNorm + MatMul
│   │       ├── compute_rms.cpp    # Standalone RMS computation
│   │       └── mhc_layer.cpp      # (Legacy) Layer-level kernel
│   │
│   └── python/
│       ├── bindings.cpp           # PyTorch ↔ C++ bindings (pybind11)
│       └── mhc/                   # Python package
│           ├── __init__.py        # Public API exports
│           ├── layer.py           # MHCLayer, MHCResidualWrapper
│           ├── ops.py             # Autograd functions and op wrappers
│           └── ops_pure_pytorch_backward.py  # Pure PyTorch fallback
│
├── scripts/
│   ├── build.sh                   # Build script
│   ├── install.sh                 # Install script
│   └── accuracy_check.py          # Numerical accuracy tests
│
├── CMakeLists.txt                 # CMake build configuration
├── setup.py                       # Python package setup
└── pyproject.toml                 # Python project metadata
```

---

## 3. Layered Architecture

The implementation follows a three-layer architecture:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Python User Interface                         │
│  MHCLayer, MHCResidualWrapper, sinkhorn_knopp(), rmsnorm(), etc.    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      C++ Bindings (bindings.cpp)                     │
│  - Type conversion (BFloat16 ↔ Float32)                             │
│  - Tiling parameter computation                                      │
│  - Kernel launch with ACLNN API                                      │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Ascend C Kernels (.cpp files)                     │
│  - Low-level computation on AI cores                                 │
│  - Memory management (Global Memory ↔ Unified Buffer)               │
│  - Vectorization and multi-core parallelism                          │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.1 Python Layer

**Purpose**: User-facing API, autograd support, high-level orchestration.

Key classes:
- `MHCLayer`: Core mHC computation module
- `MHCResidualWrapper`: Drop-in replacement for residual connections
- Autograd Functions: `SinkhornKnoppFunction`, `RMSNormFunction`, etc.

### 3.2 C++ Bindings Layer

**Purpose**: Bridge between PyTorch tensors and Ascend kernels.

Responsibilities:
- Convert PyTorch tensors to raw pointers
- Compute tiling parameters based on input shapes
- Handle type conversions (e.g., BFloat16 inputs to Float32 for kernels)
- Launch kernels via ACLNN runtime API

### 3.3 Ascend C Kernel Layer

**Purpose**: High-performance computation on Huawei Ascend NPUs.

Features:
- Written in Ascend C (C++ extension for Ascend NPUs)
- Explicit memory management between Global Memory (GM) and Unified Buffer (UB)
- Support for vectorization (SIMD operations) and multi-core parallelism

---

## 4. Operator Organization

### 4.1 Standalone Operators

Individual operators with independent Python interfaces, suitable for unit testing and standalone use.

| Operator | Forward Kernel | Backward Kernel | Python Interface |
|----------|---------------|-----------------|------------------|
| Sinkhorn-Knopp | `sinkhorn_knopp_fwd` | `sinkhorn_knopp_bwd` | `sinkhorn_knopp()` |
| RMSNorm | `rmsnorm_fwd` | `rmsnorm_bwd` | `rmsnorm()` |
| Stream Aggregate | `stream_aggregate_fwd` | `stream_aggregate_bwd` | `stream_aggregate()` |
| Stream Distribute Mix Add | `stream_distribute_mix_add_fwd` | `stream_distribute_mix_add_bwd` | `stream_distribute_mix_add()` |
| Compute RMS | `compute_rms_fwd` | N/A | `compute_rms()` |

### 4.2 Fused Layer Operators

Orchestrated at the C++ level for training efficiency. Multiple kernels are called sequentially without returning to Python between calls.

| C++ Function | Internal Kernel Calls |
|-------------|----------------------|
| `mhc_layer_fwd_dynamic` | compute_rms → matmul → sinkhorn → stream_aggregate → rmsnorm → stream_distribute |
| `mhc_layer_bwd_dynamic` | stream_distribute_bwd → rmsnorm_bwd → stream_aggregate_bwd → sinkhorn_bwd + ATen ops |

**Benefit**: Reduced Python-C++ boundary crossing overhead, intermediate tensors stay in C++.

---

## 5. Data Flow

### 5.1 MHCLayer Forward Pass (Dynamic-H Mode)

```
Input: x [B, n, C]
       │
       ▼
┌──────────────────┐
│  Flatten to      │ x_flat [B, n*C]
│  [B, n*C]        │
└──────────────────┘
       │
       ├───────────────────────────────────────┐
       ▼                                       ▼
┌──────────────────┐                   ┌──────────────────┐
│  Compute RMS     │ rms [B]           │  MatMul with     │ H_proj [B, 2n + n²]
│                  │                   │  phi_concat      │
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
       └──────────────│ Stream      │         │          │ Sinkhorn     │ M [B, n, n]
                      │ Aggregate   │         │          │ Normalization│
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
                                    │ Stream Distribute    │
                                    │ Mix Add              │
                                    └──────────────────────┘
                                               │
                                               ▼
                                    Output: [B, n, C]
```

---

## 6. Relationship with CUDA Reference

This implementation is a port of the original CUDA implementation (`src/` directory in the repository root).

| Aspect | CUDA | Ascend |
|--------|------|--------|
| Hardware | NVIDIA GPUs | Huawei Ascend NPUs |
| Language | CUDA C++ | Ascend C (C++ extension) |
| Parallelism | Thread blocks, warps | AI cores, vectorization |
| Memory | Shared memory, registers | Unified Buffer (UB), Global Memory (GM) |
| Vectorization | float4 native type | Mask API (SetMaskCount, etc.) |

**Porting Challenges**:
1. **32-byte alignment**: Ascend's `DataCopy` requires 32B aligned transfers
2. **Multi-core interference**: Mask operations are global state, not thread-local
3. **No native float4**: Must use explicit SIMD operations or scalar fallbacks

---

## 7. Build System

The project uses CMake with the Ascend CANN toolkit:

```bash
# Build
cd mHC_ascend
bash scripts/build.sh

# Install Python package
pip install -e .
```

**Key build dependencies**:
- CANN Toolkit (`/usr/local/Ascend/ascend-toolkit/latest`)
- PyTorch with torch_npu
- pybind11 (for Python bindings)

**Output**: `mhc_ascend.cpython-*.so` (Python extension module)

---

## 8. Design Decisions

### 8.1 Why Fused Layer Operations?

Training involves repeated forward-backward passes. Fusing multiple kernel calls in C++ reduces:
- Python interpreter overhead
- Tensor allocation/deallocation
- Memory bandwidth (intermediate results stay in faster memory)

### 8.2 Why Standalone Operators?

Standalone operators enable:
- Unit testing of individual components
- Debugging numerical issues in isolation
- Use in custom models that don't need full MHCLayer

### 8.3 Why Multiple Implementation Paths?

Due to Ascend hardware constraints (alignment, multi-core interference), different input configurations require different code paths:
- **Scalar path**: Safe fallback, works for all inputs
- **Vectorized path**: Faster, but has alignment/multi-core constraints
- **Specialized paths**: e.g., N=4 optimized Sinkhorn backward

The implementation automatically selects the best path based on input characteristics.
