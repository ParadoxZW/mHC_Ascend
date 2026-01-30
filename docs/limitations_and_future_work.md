# Limitations and Future Work

This document describes current limitations of the mHC Ascend implementation and planned improvements.

---

## 1. Current Limitations

### 1.1 Input Constraints

#### Sinkhorn-Knopp

| Constraint | Requirement | Reason |
|------------|-------------|--------|
| Square matrices | M == N | Sinkhorn-Knopp algorithm requirement |
| Non-negative input | All elements ≥ 0 | Algorithm non-convergence with negatives |

**Behavior**: Non-square input raises `RuntimeError` at C++ level.

#### Stream Operations

| Constraint | Recommended | Reason |
|------------|-------------|--------|
| C dimension | C ≥ 16 | Alignment requirements for vector ops |
| n dimension | n = 4 or 8 | Well-tested configurations |

**Behavior**: Smaller C values use slower scalar paths; some edge cases may have issues.

### 1.2 Known Edge Cases

| Configuration | Status | Notes |
|---------------|--------|-------|
| C=12, n=2 | ⚠️ Stream Distribute Forward may have ~1.2 error | Single edge case |
| C < 16, various n | ⚠️ MHCLayer may fail for some combinations | Small C not well-tested |
| M ≠ N (Sinkhorn) | ❌ Rejected at runtime | Invalid input |

**Recommendation**: Use typical configurations (C ≥ 64, n = 4) for production.

### 1.3 Performance Limitations

| Component | Current State | Impact |
|-----------|---------------|--------|
| Fused MatMul | Uses PyTorch (not custom kernel) | Optimal; no issue |
| Sinkhorn small matrices | Memory bandwidth bound | Limited speedup potential |
| Scalar fallback paths | ~3-4x slower than vectorized | Affects non-aligned inputs |

### 1.4 Hardware Requirements

- **NPU**: Ascend 910B1 (other variants may work but untested)
- **CANN Toolkit**: Required for compilation
- **torch_npu**: Must be imported before mhc

---

## 2. Verified Working Configurations

### 2.1 Fully Tested (Production Ready)

| Parameter | Values |
|-----------|--------|
| M, N (Sinkhorn) | 4, 8, 16 (M == N required) |
| n (expansion_rate) | 4, 8 |
| C (hidden_dim) | 64, 128, 256, 512, 1024, 2048, 4096 |
| Batch size | 1 to 128 |
| num_sinkhorn_iters | 10-20 |

### 2.2 Partially Tested

| Parameter | Values | Notes |
|-----------|--------|-------|
| C | 16-63 | Some edge cases may have issues |
| n | 2, 3, 5, 6, 7 | Less common, may use scalar paths |
| M=N | 2, 3, 5, 6, 7 | Single-core verified, multi-core varies |

### 2.3 Not Recommended

| Configuration | Issue |
|---------------|-------|
| C < 16 | Frequent scalar fallback, some failures |
| n = 2 with small C | Edge case issues |
| expansion_rate = 2 | Documented boundary problems |

---

## 3. Bug Tracking

### 3.1 Open Issues

| ID | Description | Impact | Workaround |
|----|-------------|--------|------------|
| B1 | Stream Distribute Forward C=12,n=2 error | Low (single config) | Avoid this specific config |
| B2 | MHCLayer small C backward failures | Low (non-typical) | Use C ≥ 16 |

### 3.2 Fixed Issues (Historical Reference)

| ID | Description | Fixed In | Solution |
|----|-------------|----------|----------|
| F1 | Sinkhorn Forward multi-core N%4!=0 | 2026-01-30 | DataCopyPad + scalar ZeroBuffer |
| F2 | Sinkhorn Backward math error | 2026-01-25 | Correct normalization backward formula |
| F3 | Stream Aggregate alignment CRASH | 2026-01-29 | LoadGmToQueue + padded buffers |
| F4 | Stream Distribute alignment CRASH | 2026-01-29 | LoadGmToQueue + padded buffers |
| F5 | Stream Distribute BWD type mismatch | 2026-01-29 | BF16→F32 conversion in bindings |
| F6 | DataCopy 32B alignment truncation | 2026-01-26 | Alignment-safe loading patterns |

---

## 4. Future Work

### 4.1 Priority 1: Correctness

- [ ] **Fix C=12, n=2 edge case** in Stream Distribute Forward
- [ ] **Improve small C support** in MHCLayer (C < 16)
- [ ] **Add input validation** for all operators at Python level

### 4.2 Priority 2: Performance

- [ ] **Sinkhorn N=8 optimized backward** (similar to N=4)
- [ ] **Cube Unit integration** for large matrix operations
- [ ] **Memory optimization** for large batch training

### 4.3 Priority 3: Features

- [ ] **Static H mode** full implementation (currently only Dynamic H tested)
- [ ] **Mixed precision training** (FP16 accumulation option)
- [ ] **Inference optimization** (reduced Sinkhorn iterations)

### 4.4 Priority 4: Infrastructure

- [ ] **Comprehensive unit tests** for all (C, n, M, N) combinations
- [ ] **Continuous integration** with Ascend hardware
- [ ] **Performance benchmarking suite**
- [ ] **Memory profiling** for large models

---

## 5. Contributing Guidelines

### 5.1 Reporting Issues

When reporting issues, please include:

1. **Configuration**: C, n, M, N, batch_size values
2. **Environment**: CANN version, torch_npu version, NPU model
3. **Error message**: Full traceback or numerical error values
4. **Minimal reproduction**: Smallest code that triggers the issue

### 5.2 Testing New Configurations

Before using non-standard configurations:

```python
# Test standalone operators first
from mhc import sinkhorn_knopp, stream_aggregate, stream_distribute_mix_add

# 1. Test Sinkhorn (M must equal N)
inp = torch.rand(8, N, N, device='npu')  # Non-negative!
out = sinkhorn_knopp(inp, num_iters=20)
print(f"Sinkhorn N={N}: row_sum={out.sum(-1).mean():.6f}")  # Should be ~1.0

# 2. Test Stream Aggregate
inp = torch.randn(8, n, C, device='npu')
H = torch.randn(8, n, device='npu')
out = stream_aggregate(inp, H)
print(f"Stream Aggregate n={n}, C={C}: shape={out.shape}")

# 3. Test Stream Distribute
# ... similar pattern
```

### 5.3 Code Style

- Follow existing code patterns in `src/csrc/kernels/`
- Use alignment-safe loading/storing patterns
- Add comprehensive comments for non-obvious code
- Test both single-core and multi-core modes

---

## 6. Version History

| Version | Date | Major Changes |
|---------|------|---------------|
| 0.1.0 | 2026-01-22 | Initial implementation |
| 0.2.0 | 2026-01-25 | Sinkhorn backward fix, Phase 2 optimization |
| 0.3.0 | 2026-01-27 | Fused MatMul, performance optimization |
| 0.4.0 | 2026-01-29 | Stream ops alignment fixes |
| 0.5.0 | 2026-01-30 | Sinkhorn multi-core fix, documentation |

---

## 7. References

- **mHC Paper**: [Hyper-Connections](https://arxiv.org/abs/2409.19606) (DeepSeek-AI, 2024)
- **CUDA Reference**: `src/` directory in repository
- **Ascend C Documentation**: CANN Toolkit documentation
- **Debug Notes**: `debug_notes/` directory for detailed issue analysis
