# Operator Implementation Details

This document provides detailed implementation information for all mHC Ascend operators, including computation paths, technical challenges encountered, and solutions adopted.

---

## 1. Operator Overview

### 1.1 Standalone Operators

| Operator | Source File | Description |
|----------|-------------|-------------|
| Sinkhorn-Knopp | `sinkhorn_knopp.cpp` | Doubly-stochastic matrix normalization |
| RMSNorm | `rmsnorm.cpp` | Root Mean Square normalization |
| Stream Aggregate | `stream_ops.cpp` | Weighted stream aggregation with sigmoid |
| Stream Distribute Mix Add | `stream_ops.cpp` | Distribution, mixing, and residual addition |
| Compute RMS | `compute_rms.cpp` | Standalone RMS computation |
| Fused RMSNorm MatMul | `fused_ops.cpp` | BFloat16 matrix multiplication |

### 1.2 Fused Layer Operators

| Function | Description |
|----------|-------------|
| `mhc_layer_fwd_dynamic` | Forward pass orchestrating all kernels |
| `mhc_layer_bwd_dynamic` | Backward pass orchestrating all kernels |

---

## 2. Sinkhorn-Knopp Kernel

### 2.1 Algorithm

The Sinkhorn-Knopp algorithm normalizes a non-negative matrix to become doubly-stochastic (rows and columns sum to 1) through alternating normalizations:

```
for iter in range(num_iters):
    # Row normalization
    A[i, :] = A[i, :] / sum(A[i, :])
    # Column normalization
    A[:, j] = A[:, j] / sum(A[:, j])
```

### 2.2 Forward Implementation Paths

The kernel provides three implementation paths, automatically selected based on input configuration:

| Path | Function | Characteristics |
|------|----------|-----------------|
| **v1_Stable** | `ProcessSingleMatrix_v1_Stable` | Pure scalar, guaranteed correct, used for verification |
| **v2_Optimized** | `ProcessSingleMatrix_v2_Optimized` | Loop unrolling for N=4,8,16, no Mask API |
| **v3_Vectorized** | `ProcessSingleMatrix_v3_Vectorized` | Uses Mask API (SetMaskCount, SetVectorMask) for SIMD |

**Dispatch Logic** (`ProcessSingleMatrix`):

```cpp
// Controlled by MHC_SINKHORN_FWD_IMPL environment variable
if (impl_mode == SCALAR) {
    ProcessSingleMatrix_v2_Optimized(batch_idx);
} else if (impl_mode == VECTORIZED) {
    // Safety fallback for multi-core + N%4!=0
    if ((N % 4 == 0) || (used_core_num == 1)) {
        ProcessSingleMatrix_v3_Vectorized(batch_idx);
    } else {
        ProcessSingleMatrix_v2_Optimized(batch_idx);  // Fallback
    }
} else {  // AUTO
    if ((N % 4 == 0) || (used_core_num == 1)) {
        ProcessSingleMatrix_v3_Vectorized(batch_idx);
    } else {
        ProcessSingleMatrix_v2_Optimized(batch_idx);
    }
}
```

### 2.3 Backward Implementation Paths

| Path | Function | Characteristics |
|------|----------|-----------------|
| **Scalar** | `ProcessSingleBackward_Scalar` | O(T) with history buffers |
| **Vectorized** | `ProcessSingleBackward_Vectorized` | Uses Mask API |
| **N4 Optimized** | `ProcessSingleBackward_N4_Optimized` | Specialized for M=N=4, register-like storage |

**Dispatch Logic**:
```cpp
if (M == 4 && N == 4) {
    ProcessSingleBackward_N4_Optimized(batch_idx);  // Register-optimized
} else if ((N % 4 == 0) || (used_core_num == 1)) {
    ProcessSingleBackward_Vectorized(batch_idx);
} else {
    ProcessSingleBackward_Scalar(batch_idx);
}
```

### 2.4 Key Implementation Details

**Memory Layout**:
- Input: `[B, M, N]` in Global Memory (GM)
- Internal: `[M, N_aligned]` in Unified Buffer (UB), where `N_aligned = ALIGN_UP(N, 8)`
- Each row starts at 32-byte aligned address for vectorization

**Multi-core Work Distribution**:
```cpp
int32_t batch_per_core = CeilingDiv(batch_size, num_cores);
int32_t batch_start = core_idx * batch_per_core;
int32_t batch_count = min(batch_per_core, batch_size - batch_start);
```

**GM Write-back**:
```cpp
// IMPORTANT: Use DataCopyPad instead of SetValue for multi-core safety
AscendC::DataCopyExtParams params = {1, N * sizeof(T), 0, 0, 0};
AscendC::DataCopyPad(outGm[gm_offset], matrix[ub_offset], params);
```

### 2.5 Challenges and Solutions

| Challenge | Solution |
|-----------|----------|
| **Multi-core Mask interference** | N%4!=0 uses scalar path; N%4==0 safe for vectorized |
| **ZeroBuffer multi-core issue** | Use scalar loop instead of ZeroBuffer (Duplicate) |
| **SetValue multi-core writes** | Replace with DataCopyPad for all GM writes |
| **Backward O(T²) complexity** | Phase 2 optimization: save history buffers, achieve O(T) |

### 2.6 Input Constraints

- **Square matrices required**: M must equal N (enforced via `TORCH_CHECK`)
- **Non-negative input**: Negative values cause non-convergence

---

## 3. RMSNorm Kernel

### 3.1 Algorithm

```
rms = sqrt(mean(x²) + eps)
out = weight * (x / rms)
```

### 3.2 Implementation

**Single implementation** with alignment-safe data transfer:

```cpp
// Load with DataCopyPad for alignment safety
DataCopyPadExtParams<DataT> padParams = {false, 0, pad_count, 0};
DataCopyPad(inp, inpGm[offset], copyParams, padParams);

// BF16 → F32 conversion for computation
ConvertBF16ToF32(inpF32, inp, hidden_dim_aligned);

// Compute RMS
Mul(squared, inpF32, inpF32, hidden_dim_aligned);  // x²
sum = ReduceSum(squared, hidden_dim);              // sum(x²)
rms = sqrt(sum / hidden_dim + eps)

// Normalize and scale
Divs(normalized, inpF32, rms, hidden_dim);         // x / rms
Mul(outF32, normalized, weightF32, hidden_dim);    // * weight

// F32 → BF16 conversion for output
ConvertF32ToBF16(out, outF32, hidden_dim);
```

### 3.3 Key Details

- **Input type**: BFloat16
- **Computation type**: Float32 (for precision)
- **Output type**: BFloat16
- **Weight**: BFloat16, loaded once per core and converted to F32

---

## 4. Stream Aggregate Kernel

### 4.1 Algorithm

```
H_activated = sigmoid(H_pre_raw)
out = sum(H_activated[i] * inp[i, :] for i in range(n))
```

### 4.2 Implementation

**Alignment-aware implementation** with automatic path selection:

```cpp
// Determine if C dimension needs padding
C_padded = ALIGN_UP(C, 8);  // 8 floats = 32 bytes
needs_C_padding = (C != C_padded);

// Load input with alignment handling
if (needs_C_padding) {
    // Row-by-row load with padding
    for (int row = 0; row < n; row++) {
        LoadGmToQueue(inp[row * C_padded], inpGm[row * C], C);
        // Zero-fill padding
        for (int c = C; c < C_padded; c++) {
            inp.SetValue(row * C_padded + c, 0.0f);
        }
    }
} else {
    // Fast path: direct DataCopy
    DataCopy(inp, inpGm[offset], n * C);
}

// Weighted sum using C_padded for vector operations
for (int i = 0; i < n; i++) {
    h_val = H_activated.GetValue(i);
    Axpy(outF32, inp[i * row_stride], h_val, vec_len);  // out += h * inp[i]
}
```

### 4.3 LoadGmToQueue Helper

Pattern from official Ascend C sample (`DataCopyPadCustom_GM2UB`):

```cpp
void LoadGmToQueue(LocalTensor dst, GlobalTensor src, int32_t count) {
    const int32_t align_count = 32 / sizeof(T);  // 8 for float32
    int32_t aligned_count = (count / align_count) * align_count;

    if (aligned_count > 0) {
        DataCopy(dst, src, aligned_count);  // Copy aligned portion
    }
    // Scalar copy for remainder
    for (int32_t i = aligned_count; i < count; i++) {
        dst.SetValue(i, src.GetValue(i));
    }
}
```

### 4.4 Challenges and Solutions

| Challenge | Solution |
|-----------|----------|
| **C < 16 causes CRASH** | Use padded buffer layout with LoadGmToQueue |
| **n*C not 32B aligned** | Row-by-row loading with padding |
| **Output not aligned** | Use DataCopyPad for GM write |

---

## 5. Stream Distribute Mix Add Kernel

### 5.1 Algorithm

```
# Distribution
y_dist[i] = 2 * sigmoid(H_post_raw[i]) * y_norm

# Mixing
mix_out[i] = sum(M[i, j] * x_inp[j, :] for j in range(n))

# Output
out[i] = y_dist[i] + mix_out[i]
```

### 5.2 Forward Implementation

**Alignment-safe implementation** supporting arbitrary C and n:

```cpp
// Padded dimensions
C_padded = ALIGN_UP(C, 16);  // 16 for BF16, 8 for F32

// y_norm loading (BF16, C elements)
if (C * 2 < 32) {  // < 32 bytes
    // Scalar load
    for (int c = 0; c < C; c++) {
        y_norm.SetValue(c, y_normGm.GetValue(batch_idx * C + c));
    }
    // Zero-fill padding
    for (int c = C; c < C_padded; c++) {
        y_norm.SetValue(c, 0);
    }
} else {
    LoadGmToQueue(y_norm, y_normGm[offset], C);
}

// M matrix loading (F32, n×n elements)
LoadGmToQueue(M_t, MGm[offset], n * n);

// x_inp loading (BF16, n×C elements)
// Row-by-row with padding
for (int row = 0; row < n; row++) {
    LoadGmToQueue(x_inp[row * C_padded], x_inpGm[row * C], C);
}

// Vector operations use C_padded length
Muls(y_dist, y_norm, 2.0f * sigmoid(H_post), C_padded);

// Output writing (per-row DataCopyPad)
for (int row = 0; row < n; row++) {
    DataCopyExtParams params = {1, C * sizeof(OutT), 0, 0, 0};
    DataCopyPad(outGm[row * C], out[row * C_padded], params);
}
```

### 5.3 Backward Implementation

Similar alignment handling with type conversion:

```cpp
// IMPORTANT: y_norm must be converted from BF16 to F32 before kernel
auto y_norm_f32 = y_norm.to(at::kFloat).contiguous();
// ... pass y_norm_f32.data_ptr() to kernel
```

### 5.4 Challenges and Solutions

| Challenge | Solution |
|-----------|----------|
| **C=4 CRASH** | Scalar GetValue for small C |
| **C=8 wrong results** | Use C_padded=16 minimum |
| **n%4!=0 wrong results** | Per-row loading and padding |
| **y_norm BF16 type mismatch** | Explicit type conversion in bindings.cpp |

---

## 6. Fused RMSNorm MatMul Kernel

### 6.1 Algorithm

```
out = x @ weight.T  # [B, hidden_dim] @ [out_dim, hidden_dim].T -> [B, out_dim]
```

### 6.2 Implementation

**Default**: PyTorch matmul (uses optimized CANN ops)
**Optional**: Custom scalar kernel (for debugging, ~33x slower)

```cpp
// Scalar kernel implementation
for (int i = 0; i < out_dim; i++) {
    float dot_product = 0.0f;
    for (int j = 0; j < hidden_dim; j++) {
        // BF16 -> F32 manual conversion
        uint16_t w_bits = *(uint16_t*)&weight[i * hidden_dim + j];
        float w_f32 = bit_cast<float>(w_bits << 16);
        dot_product += inp_f32[j] * w_f32;
    }
    result[i] = dot_product;
}
```

### 6.3 Why PyTorch Default?

| Implementation | Time (B=64, hidden=256, out=28) |
|---------------|--------------------------------|
| PyTorch matmul | 0.036 ms |
| Custom scalar | 1.082 ms |

PyTorch uses Ascend's Cube Unit (Matrix Engine) which is highly optimized.

---

## 7. Compute RMS Kernel

### 7.1 Algorithm

```
rms = sqrt(mean(x²) + eps)
```

### 7.2 Implementation

Standalone version of the RMS computation from RMSNorm:

```cpp
// BF16 input
DataCopyPad(inp, inpGm[offset], copyParams, padParams);
ConvertBF16ToF32(inpF32, inp, K_aligned);

// Compute
Mul(squared, inpF32, inpF32, K_aligned);
sum = ReduceSum(squared, K);
rms = sqrt(sum / K + eps);

// Output F32 RMS value
rmsGm.SetValue(batch_idx, rms);
```

---

## 8. Fused Layer Operations

### 8.1 Forward (`mhc_layer_fwd_dynamic`)

Orchestrates all forward kernels in sequence:

```cpp
// 1. Flatten input
auto x_flat = x.view({B, n * C});

// 2. Compute RMS
auto rms = compute_rms_fwd(x_flat, eps);

// 3. Project to H values
auto H_proj = fused_rmsnorm_matmul_fwd(x_flat, phi_concat);  // or torch::matmul

// 4. Split projections
auto H_pre = H_proj.slice(...);
auto H_post = H_proj.slice(...);
auto H_res = H_proj.slice(...);

// 5. Compute tilde values: alpha * proj * (1/rms) + b
auto tilde_pre = alpha_pre * H_pre * rms.reciprocal().unsqueeze(-1) + b_pre;

// 6. Sinkhorn normalization
auto M = sinkhorn_knopp_fwd(exp(H_res), num_iters, eps);

// 7. Stream aggregate
auto [x_agg, H_pre_activated] = stream_aggregate_fwd(x, tilde_pre);

// 8. RMSNorm
auto [y_norm, rms_h] = rmsnorm_fwd(x_agg, weight, eps);

// 9. Stream distribute
auto [out, H_post_activated] = stream_distribute_mix_add_fwd(y_norm, tilde_post, M, x);

return {out, rms, x_agg, H_pre_activated, H_post_activated, M, y_norm, x_flat, rms_h};
```

### 8.2 Backward (`mhc_layer_bwd_dynamic`)

Orchestrates backward kernels in reverse order:

```cpp
// 1. Stream distribute backward
auto [grad_x_partial, grad_y_norm, grad_M, grad_H_post] =
    stream_distribute_mix_add_bwd(grad_out, x, y_norm, M, H_post_activated);

// 2. RMSNorm backward
auto [grad_x_agg, grad_weight] = rmsnorm_bwd(grad_y_norm, x_agg, weight, rms_h);

// 3. Stream aggregate backward
auto [grad_x_from_agg, grad_H_pre] =
    stream_aggregate_bwd(grad_x_agg, x, H_pre_activated);

// 4. Sinkhorn backward
auto grad_H_res = sinkhorn_knopp_bwd(grad_M * M, inp_sinkhorn, M, num_iters, eps);

// 5. Combine gradients and compute parameter gradients via ATen ops
// ...

return {grad_x, grad_weight, grad_phi_pre, grad_phi_post, grad_phi_res,
        grad_alpha_pre, grad_alpha_post, grad_alpha_res,
        grad_b_pre, grad_b_post, grad_b_res};
```

---

## 9. Common Implementation Patterns

### 9.1 Alignment-Safe Loading

```cpp
// Pattern: LoadGmToQueue
void LoadGmToQueue(LocalTensor dst, GlobalTensor src, int count) {
    int aligned = (count / 8) * 8;  // 32-byte aligned count
    if (aligned > 0) DataCopy(dst, src, aligned);
    for (int i = aligned; i < count; i++) {
        dst.SetValue(i, src.GetValue(i));
    }
}
```

### 9.2 Alignment-Safe Storing

```cpp
// Pattern: Conditional DataCopy vs DataCopyPad
bool aligned = (count * sizeof(T)) % 32 == 0;
if (aligned) {
    DataCopy(dstGm[offset], src, count);
} else {
    DataCopyExtParams params = {1, count * sizeof(T), 0, 0, 0};
    DataCopyPad(dstGm[offset], src, params);
}
```

### 9.3 Padded Buffer Strategy

```cpp
// Pattern: Padded dimensions for vector operations
int C_padded = ALIGN_UP(C, 8);  // For F32: 8 elements = 32 bytes
bool needs_padding = (C != C_padded);

// Allocate padded buffer
pipe.InitBuffer(buf, n * C_padded * sizeof(float));

// Vector operations use padded length (padding is zero, safe for SIMD)
Muls(dst, src, scalar, C_padded);
```

### 9.4 Multi-core Work Distribution

```cpp
// Pattern: Batch distribution across cores
int batch_per_core = CeilingDiv(batch_size, num_cores);
int batch_start = core_idx * batch_per_core;
int batch_count = min(batch_per_core, batch_size - batch_start);

if (batch_count <= 0) return;  // This core has no work

// Set GM buffer with offset
int offset = batch_start * feature_size;
inputGm.SetGlobalBuffer(ptr + offset, batch_count * feature_size);
```

---

## 10. Source File Index

| File | Content |
|------|---------|
| `src/csrc/kernels/sinkhorn_knopp.cpp` | Sinkhorn forward and backward kernels |
| `src/csrc/kernels/rmsnorm.cpp` | RMSNorm forward and backward kernels |
| `src/csrc/kernels/stream_ops.cpp` | Stream aggregate and distribute kernels |
| `src/csrc/kernels/fused_ops.cpp` | Fused RMSNorm MatMul kernel |
| `src/csrc/kernels/compute_rms.cpp` | Standalone RMS computation kernel |
| `src/csrc/include/mhc_types.h` | Tiling structures and type definitions |
| `src/csrc/include/utils.h` | Helper macros and functions |
| `src/python/bindings.cpp` | PyTorch C++ bindings |
