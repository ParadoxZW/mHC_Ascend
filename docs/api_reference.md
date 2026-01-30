# API Reference

This document provides a complete reference for the mHC Ascend Python API and environment variables.

---

## 1. Python API

### 1.1 Main Classes

#### MHCLayer

The core mHC computation module.

```python
from mhc import MHCLayer

layer = MHCLayer(
    hidden_dim: int,           # Feature dimension (C)
    expansion_rate: int = 4,   # Number of streams (n)
    num_sinkhorn_iters: int = 20,
    sinkhorn_eps: float = 1e-8,
    rmsnorm_eps: float = 1e-5,
    use_dynamic_h: bool = True,
    alpha_init: float = 0.01,
)
```

**Parameters**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hidden_dim` | int | required | Feature dimension (C) |
| `expansion_rate` | int | 4 | Number of streams (n) |
| `num_sinkhorn_iters` | int | 20 | Sinkhorn-Knopp iterations |
| `sinkhorn_eps` | float | 1e-8 | Sinkhorn numerical stability epsilon |
| `rmsnorm_eps` | float | 1e-5 | RMSNorm numerical stability epsilon |
| `use_dynamic_h` | bool | True | Use dynamic (input-dependent) H computation |
| `alpha_init` | float | 0.01 | Initialization scale for alpha parameters |

**Input/Output**:
| Direction | Shape | Type | Description |
|-----------|-------|------|-------------|
| Input | `[B, n, C]` | float32 | Batch of n streams, each with C features |
| Output | `[B, n, C]` | float32 | Processed streams |

**Example**:
```python
import torch
import torch_npu
from mhc import MHCLayer

layer = MHCLayer(hidden_dim=256, expansion_rate=4).npu()
x = torch.randn(32, 4, 256, device='npu')
output = layer(x)  # [32, 4, 256]
```

---

#### MHCResidualWrapper

Drop-in replacement for residual connections in Transformers.

```python
from mhc import MHCResidualWrapper

wrapper = MHCResidualWrapper(
    hidden_dim: int,
    expansion_rate: int = 4,
    residual_mode: str = 'decay',  # 'zero' or 'decay'
    **mhc_kwargs,
)
```

**Parameters**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hidden_dim` | int | required | Feature dimension |
| `expansion_rate` | int | 4 | Number of streams |
| `residual_mode` | str | 'decay' | How to initialize extra streams: 'zero' or 'decay' |
| `**mhc_kwargs` | | | Additional arguments passed to MHCLayer |

**Input/Output**:
| Direction | Shape | Type | Description |
|-----------|-------|------|-------------|
| residual | `[B, C]` or `[B, T, C]` | float32 | Residual input |
| branch_output | same as residual | float32 | Branch output (attention/FFN) |
| Output | same as input | float32 | mHC-processed result |

**Example**:
```python
from mhc import MHCResidualWrapper

# In a Transformer block
mhc = MHCResidualWrapper(hidden_dim=512, expansion_rate=4).npu()

# Replace: output = residual + attention_output
output = mhc(residual, attention_output)
```

---

### 1.2 Standalone Operators

#### sinkhorn_knopp

Doubly-stochastic matrix normalization via Sinkhorn-Knopp algorithm.

```python
from mhc import sinkhorn_knopp

output = sinkhorn_knopp(
    inp: Tensor,        # [B, N, N] float32, non-negative, square matrices
    num_iters: int = 20,
    eps: float = 1e-8,
) -> Tensor  # [B, N, N] float32, doubly-stochastic matrices
```

**Constraints**:
- Input must be **square matrices** (M == N). Non-square input raises `RuntimeError`.
- Input must be **non-negative**. Negative values cause algorithm non-convergence.

**Example**:
```python
from mhc import sinkhorn_knopp

# Create non-negative input (use rand, not randn!)
inp = torch.rand(8, 4, 4, device='npu')
out = sinkhorn_knopp(inp, num_iters=20, eps=1e-8)

# Verify doubly-stochastic property
print(out.sum(dim=-1))  # Should be close to 1
print(out.sum(dim=-2))  # Should be close to 1
```

---

#### rmsnorm

Root Mean Square Layer Normalization.

```python
from mhc import rmsnorm

output = rmsnorm(
    inp: Tensor,     # [B, C] float32
    weight: Tensor,  # [C] float32
    eps: float = 1e-5,
) -> Tensor  # [B, C] BFloat16
```

**Note**: Output is BFloat16 due to internal kernel design.

---

#### stream_aggregate

Weighted aggregation of multiple streams with sigmoid activation.

```python
from mhc import stream_aggregate

output = stream_aggregate(
    inp: Tensor,       # [B, n, C] float32
    H_pre_raw: Tensor, # [B, n] float32
) -> Tensor  # [B, C] BFloat16
```

**Formula**: `output = Σ sigmoid(H_pre_raw[i]) * inp[i]` for i in range(n)

---

#### stream_distribute_mix_add

Distributes normalized features, applies mixing matrix, and adds residual.

```python
from mhc import stream_distribute_mix_add

output = stream_distribute_mix_add(
    y_norm: Tensor,      # [B, C] float32
    H_post_raw: Tensor,  # [B, n] float32
    M: Tensor,           # [B, n, n] float32
    x_inp: Tensor,       # [B, n, C] float32
) -> Tensor  # [B, n, C] float32
```

**Formula**:
- Distribution: `y_dist[i] = 2 * sigmoid(H_post_raw[i]) * y_norm`
- Mixing: `mix_out[i] = Σ M[i,j] * x_inp[j]`
- Output: `output = y_dist + mix_out`

---

#### compute_rms

Standalone RMS (Root Mean Square) computation.

```python
from mhc import compute_rms

rms = compute_rms(
    inp: Tensor,  # [B, K] BFloat16
    eps: float = 1e-5,
) -> Tensor  # [B] float32
```

**Formula**: `rms = sqrt(mean(inp²) + eps)`

---

### 1.3 Factory Functions

#### create_mhc_layer

Factory function for creating MHCLayer.

```python
from mhc import create_mhc_layer

layer = create_mhc_layer(
    hidden_dim: int,
    expansion_rate: int = 4,
    **kwargs,
) -> MHCLayer
```

#### replace_residual_with_mhc

Factory function for creating MHCResidualWrapper.

```python
from mhc import replace_residual_with_mhc

wrapper = replace_residual_with_mhc(
    hidden_dim: int,
    expansion_rate: int = 4,
    residual_mode: str = 'decay',
    **kwargs,
) -> MHCResidualWrapper
```

---

## 2. Environment Variables

### 2.1 Global Settings

#### MHC_MULTICORE

Controls the number of AI cores used for parallel computation.

| Value | Effect |
|-------|--------|
| `1` | Single-core mode (useful for debugging) |
| `2-32` | Multi-core parallel (limited by batch size) |
| Not set | Default: 8 cores |

**Example**:
```bash
# Debug mode (single core)
MHC_MULTICORE=1 python train.py

# Production mode (default 8 cores)
python train.py
```

**Notes**:
- Actual cores used = `min(MHC_MULTICORE, batch_size)`
- Ascend 910B typically has 8 AI cores

---

### 2.2 Sinkhorn Implementation Control

#### MHC_SINKHORN_FWD_IMPL

Controls Sinkhorn forward implementation selection.

| Value | Behavior |
|-------|----------|
| `auto` (default) | N%4==0 or single-core → Vectorized; otherwise → Scalar |
| `scalar` | Force scalar (loop-unrolled) implementation |
| `vectorized` | Force vectorized (Mask API), with safety fallback |

#### MHC_SINKHORN_BWD_IMPL

Controls Sinkhorn backward implementation selection.

| Value | Behavior |
|-------|----------|
| `auto` (default) | M=N=4 → N4 Optimized; N%4==0 or single-core → Vectorized; otherwise → Scalar |
| `scalar` | Force scalar implementation |
| `vectorized` | Force vectorized, with safety fallback |
| `n4_optimized` or `n4` | Force N4 optimized (only effective when M=N=4) |

**Example**:
```bash
# Force scalar implementations for debugging
MHC_SINKHORN_FWD_IMPL=scalar MHC_SINKHORN_BWD_IMPL=scalar python train.py
```

---

### 2.3 MatMul Implementation Control

#### MHC_USE_SCALAR_MATMUL

Controls the fused RMSNorm MatMul implementation.

| Value | Behavior |
|-------|----------|
| `0` (default) | Use PyTorch matmul (optimized CANN ops, ~33x faster) |
| `1` | Use custom scalar kernel (for debugging) |

**Note**: PyTorch matmul is highly optimized and recommended for production.

---

## 3. Environment Variable Summary

| Variable | Default | Values | Purpose |
|----------|---------|--------|---------|
| `MHC_MULTICORE` | `8` | `1-32` | Number of AI cores |
| `MHC_SINKHORN_FWD_IMPL` | `auto` | `auto`, `scalar`, `vectorized` | Sinkhorn forward path |
| `MHC_SINKHORN_BWD_IMPL` | `auto` | `auto`, `scalar`, `vectorized`, `n4`/`n4_optimized` | Sinkhorn backward path |
| `MHC_USE_SCALAR_MATMUL` | `0` | `0`, `1` | MatMul implementation |

---

## 4. Recommended Configurations

### 4.1 Production Training

```bash
# Default settings are optimal
python train.py
```

Recommended hyperparameters:
| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `expansion_rate` | 4 | Well-tested configuration |
| `num_sinkhorn_iters` | 20 | Training; can reduce to 10-15 for inference |
| `use_dynamic_h` | True | Better expressiveness |
| `alpha_init` | 0.01 | Small initialization for stability |

### 4.2 Debugging Numerical Issues

```bash
# Single-core, scalar implementations
MHC_MULTICORE=1 \
MHC_SINKHORN_FWD_IMPL=scalar \
MHC_SINKHORN_BWD_IMPL=scalar \
python debug_script.py
```

### 4.3 Performance Benchmarking

```bash
# Explicit multi-core (same as default)
MHC_MULTICORE=8 python benchmark.py
```

---

## 5. C++ Module Interface

The compiled `mhc_ascend` module exports the following functions (for advanced users):

```python
import mhc_ascend

# Forward functions
mhc_ascend.sinkhorn_knopp_fwd(inp, num_iters, eps)
mhc_ascend.rmsnorm_fwd(inp, weight, eps)
mhc_ascend.stream_aggregate_fwd(inp, H_pre_raw)
mhc_ascend.stream_distribute_mix_add_fwd(y_norm, H_post_raw, M, x_inp)
mhc_ascend.compute_rms_fwd(inp, eps)
mhc_ascend.fused_rmsnorm_matmul_fwd(inp, weight)

# Backward functions
mhc_ascend.sinkhorn_knopp_bwd(grad_out, inp, out, num_iters, eps)
mhc_ascend.rmsnorm_bwd(grad_out, inp, weight, rms)
mhc_ascend.stream_aggregate_bwd(grad_out, inp, H_pre_activated)
mhc_ascend.stream_distribute_mix_add_bwd(grad_out, x_inp, y_norm, M, H_post_activated)

# Fused layer operations
mhc_ascend.mhc_layer_fwd_dynamic(...)  # Returns output + intermediates
mhc_ascend.mhc_layer_bwd_dynamic(...)  # Returns all gradients
```

**Note**: These low-level functions do not provide autograd support. Use the Python API (`mhc.sinkhorn_knopp`, etc.) for automatic differentiation.
