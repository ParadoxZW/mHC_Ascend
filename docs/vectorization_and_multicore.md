# Vectorization and Multi-core Support

This document describes the vectorization and multi-core parallel computing support in mHC Ascend, including technical challenges encountered and solutions adopted.

---

## 1. Overview

### 1.1 Ascend NPU Architecture

Huawei Ascend 910B NPU features:
- **8 AI Cores**: Independent processing units capable of parallel execution
- **Vector Unit**: SIMD operations on 256-bit (32 bytes) data
- **Unified Buffer (UB)**: Fast on-chip memory per core
- **Global Memory (GM)**: Shared memory accessible by all cores

### 1.2 Key Differences from CUDA

| Aspect | CUDA | Ascend C |
|--------|------|----------|
| Parallelism model | SIMT (threads within warps) | Multi-core (independent cores) |
| Vectorization | Native float4 type | Mask API (SetMaskCount, etc.) |
| Mask state | Thread-private | **Global/Shared** (key issue!) |
| Memory transfer | Coalesced access | **32-byte aligned DataCopy** |

---

## 2. Multi-core Parallelism

### 2.1 Work Distribution

Each kernel distributes work across cores by batch:

```cpp
int32_t core_idx = GetBlockIdx();
int32_t num_cores = tiling.used_core_num;

int32_t batch_per_core = CeilingDiv(batch_size, num_cores);
int32_t batch_start = core_idx * batch_per_core;
int32_t batch_count = min(batch_per_core, batch_size - batch_start);

if (batch_count <= 0) return;  // This core has no work
```

### 2.2 Environment Variable Control

| Variable | Default | Description |
|----------|---------|-------------|
| `MHC_MULTICORE` | 8 | Number of AI cores to use (1-32) |

```bash
# Single-core mode (for debugging)
MHC_MULTICORE=1 python train.py

# Explicit 8-core mode (same as default)
MHC_MULTICORE=8 python train.py
```

### 2.3 Multi-core Support Status

| Operator | Forward | Backward | Notes |
|----------|---------|----------|-------|
| Sinkhorn-Knopp | ✅ All N values | ✅ M=N=4 optimized | Fixed in 2026-01-30 |
| RMSNorm | ✅ | ✅ | No issues |
| Stream Aggregate | ✅ | ✅ | No issues |
| Stream Distribute Mix Add | ✅ | ✅ | No issues |
| Compute RMS | ✅ | N/A | No issues |
| Fused MatMul | ✅ | ✅ | Uses PyTorch ops |

---

## 3. Multi-core Challenges and Solutions

### 3.1 Mask API Global State Interference

**Problem**: Vectorization Mask operations (`SetMaskCount`, `SetVectorMask`, `ResetMask`) modify global hardware state. When multiple cores execute these concurrently, they interfere with each other.

**Symptom**: 40-80% numerical error when N%4!=0 in multi-core mode.

**Example of problematic code**:
```cpp
// DANGEROUS: Mask operations in multi-core mode
SetMaskCount(N);  // Core 0 sets mask for N elements
SetVectorMask<float>(mask, mask);  // Core 1 might overwrite!
Add(dst, src1, src2, MASK_PLACEHOLDER, N);
```

**Solution**: Automatic path selection based on N alignment and core count:

```cpp
bool can_use_vectorized = (N % 4 == 0) || (used_core_num == 1);
if (can_use_vectorized) {
    ProcessVectorized();  // Safe: N%4==0 means fixed mask pattern
} else {
    ProcessScalar();      // Fallback: no Mask API usage
}
```

**Why N%4==0 is safe**: When N is divisible by 4, the mask pattern is always all-ones (full vector), so even if multiple cores set it simultaneously, the result is consistent.

### 3.2 ZeroBuffer (Duplicate) Multi-core Issue

**Problem**: `ZeroBuffer()` internally calls `Duplicate()` which uses Mask operations for non-aligned lengths.

**Symptom**: Inconsistent buffer initialization across cores.

**Solution**: Use scalar loop for buffer clearing:

```cpp
// WRONG: ZeroBuffer uses Duplicate (Mask interference)
ZeroBuffer(colSums, N);

// CORRECT: Scalar loop
for (int32_t j = 0; j < N; ++j) {
    colSums.SetValue(j, 0.0f);
}
```

### 3.3 GlobalTensor::SetValue Multi-core Writes

**Problem**: Multiple cores calling `SetValue()` to write different GM locations can interfere, likely due to cache line conflicts.

**Symptom**: Some batches have zero output (writes not persisted).

**Pattern observed**:
```
Batch 0: correct (written by core 0)
Batch 1: zeros   (core 1 write lost)
Batch 2: correct (written by core 2)
Batch 3: zeros   (core 3 write lost)
```

**Solution**: Use `DataCopyPad()` instead of `SetValue()` for all GM writes:

```cpp
// WRONG: SetValue has multi-core issues
for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
        outGm.SetValue(gm_offset + i * N + j, matrix.GetValue(ub_offset + j));
    }
}

// CORRECT: DataCopyPad is reliable
for (int i = 0; i < M; i++) {
    DataCopyExtParams params = {1, N * sizeof(T), 0, 0, 0};
    DataCopyPad(outGm[gm_offset + i * N], matrix[ub_offset + i * N_aligned], params);
}
```

---

## 4. Vectorization

### 4.1 Ascend C Vector Operations

Available vector instructions:
- **Arithmetic**: `Add`, `Sub`, `Mul`, `Div`, `Muls`, `Adds`
- **Math**: `Exp`, `Log`, `Sqrt`, `Reciprocal`
- **Reduction**: `ReduceSum`, `ReduceMax`, `ReduceMin`
- **Data movement**: `DataCopy`, `DataCopyPad`

**Requirement**: Operations on Unified Buffer require 32-byte aligned addresses and lengths.

### 4.2 Mask API Usage

For non-aligned vector lengths, use Mask API:

```cpp
// Set mask for N elements (N not necessarily aligned)
SetMaskCount(N);
uint64_t mask = (1ULL << N) - 1;
SetVectorMask<float>(mask, mask);

// Execute masked operation
Add(dst, src1, src2, MASK_PLACEHOLDER, N_aligned);

// Reset mask (important!)
ResetMask();
```

**Caution**: Only use Mask API in single-core mode or when N%4==0.

### 4.3 Vectorization Status by Operator

| Operator | Forward | Backward | Notes |
|----------|---------|----------|-------|
| Sinkhorn | ✅ v3_Vectorized | ✅ N4_Optimized for M=N=4 | Auto-dispatch |
| RMSNorm | ✅ | ✅ | Uses DataCopyPad |
| Stream Aggregate | ✅ | ✅ | Uses padded buffers |
| Stream Distribute | ✅ | ✅ | Uses padded buffers |
| Fused MatMul | ❌ Scalar | N/A | PyTorch default |

---

## 5. Intelligent Computation Path Selection

### 5.1 Design Principles

1. **Correctness first**: Prefer safe (scalar) path over potentially incorrect vectorized path
2. **Automatic selection**: Choose optimal path based on input characteristics
3. **User override**: Environment variables allow forcing specific paths

### 5.2 Sinkhorn Forward Dispatch

```cpp
// Environment variable: MHC_SINKHORN_FWD_IMPL
// Values: auto (default), scalar, vectorized

switch (impl_mode) {
case SCALAR:
    ProcessSingleMatrix_v2_Optimized(batch_idx);
    break;
case VECTORIZED:
    // Safety fallback for multi-core + N%4!=0
    if (vectorized_safe) {
        ProcessSingleMatrix_v3_Vectorized(batch_idx);
    } else {
        ProcessSingleMatrix_v2_Optimized(batch_idx);
    }
    break;
case AUTO:
default:
    if (can_use_vectorized) {
        ProcessSingleMatrix_v3_Vectorized(batch_idx);
    } else {
        ProcessSingleMatrix_v2_Optimized(batch_idx);
    }
}
```

### 5.3 Sinkhorn Backward Dispatch

```cpp
// Environment variable: MHC_SINKHORN_BWD_IMPL
// Values: auto (default), scalar, vectorized, n4_optimized

if (M == 4 && N == 4) {
    // Specialized register-optimized version
    ProcessSingleBackward_N4_Optimized(batch_idx);
} else if (can_use_vectorized) {
    ProcessSingleBackward_Vectorized(batch_idx);
} else {
    ProcessSingleBackward_Scalar(batch_idx);
}
```

### 5.4 Stream Ops Alignment Dispatch

```cpp
// Automatic based on dimension alignment
bool needs_C_padding = (C % 8 != 0);  // 8 floats = 32 bytes

if (needs_C_padding) {
    // Row-by-row loading with padding
    // Vector ops use C_padded length
} else {
    // Fast path: direct DataCopy
}
```

---

## 6. Environment Variables Summary

| Variable | Values | Default | Purpose |
|----------|--------|---------|---------|
| `MHC_MULTICORE` | 1-32 | 8 | Number of AI cores |
| `MHC_SINKHORN_FWD_IMPL` | auto, scalar, vectorized | auto | Forward implementation |
| `MHC_SINKHORN_BWD_IMPL` | auto, scalar, vectorized, n4 | auto | Backward implementation |
| `MHC_USE_SCALAR_MATMUL` | 0, 1 | 0 | MatMul implementation |

### 6.1 Recommended Settings

**Production**:
```bash
# Default settings are optimal
python train.py
```

**Debugging numerical issues**:
```bash
MHC_MULTICORE=1 \
MHC_SINKHORN_FWD_IMPL=scalar \
MHC_SINKHORN_BWD_IMPL=scalar \
python debug.py
```

---

## 7. Performance Characteristics

### 7.1 Multi-core Speedup

| Configuration | Tokens/sec | Relative |
|---------------|------------|----------|
| Pure PyTorch | 639 | 1.0x |
| Ascend C (1 core) | ~200 | 0.3x |
| Ascend C (8 cores) | **860** | **1.35x** |

### 7.2 Implementation Comparison

| Sinkhorn N=4 | Implementation | Time |
|--------------|----------------|------|
| Scalar | v2_Optimized | baseline |
| Vectorized | v3_Vectorized | ~same (small matrix overhead dominates) |
| N4 Optimized | Register-like | ~same (memory bandwidth bound) |

**Note**: For small matrices (4x4), kernel launch overhead and memory bandwidth dominate; the computation itself is very fast regardless of implementation.

---

## 8. Best Practices

### 8.1 For Kernel Developers

1. **Avoid Mask API in multi-core code** unless N%4==0 or single-core mode
2. **Use DataCopyPad** instead of SetValue for GM writes
3. **Use scalar loops** for buffer clearing instead of ZeroBuffer
4. **Align buffer sizes** to 32 bytes for vector operations
5. **Test both single-core and multi-core** modes

### 8.2 For Users

1. **Use default settings** for production (8 cores, auto implementation)
2. **Use single-core mode** when debugging numerical issues
3. **Typical configurations** (M=N=4, n=4, C>=16) are fully optimized
4. **Unusual configurations** may fall back to slower but correct scalar paths

---

## 9. Known Limitations

### 9.1 Vectorization Limitations

- Mask API not fully multi-core safe for arbitrary N
- Small matrix sizes (< 32 bytes) require scalar handling
- No native float4 type (unlike CUDA)

### 9.2 Multi-core Limitations

- GlobalTensor::SetValue unreliable for multi-core writes
- ZeroBuffer/Duplicate use Mask internally
- Cache line conflicts possible with concurrent GM access

### 9.3 Future Improvements

- **Wait for Ascend SDK updates**: Thread-safe Mask API
- **Cube Unit integration**: Use Matrix Engine for larger matrices
- **More specialized paths**: N=8, N=16 optimized versions
