# Troubleshooting Guide

This document provides solutions for common issues when using mHC Ascend.

---

## 1. Installation Issues

### 1.1 Import Error: "No module named 'mhc_ascend'"

**Symptom**:
```python
>>> from mhc import MHCLayer
ModuleNotFoundError: No module named 'mhc_ascend'
```

**Cause**: The C++ extension is not compiled or installed.

**Solution**:
```bash
cd mHC_ascend
bash scripts/build.sh
pip install -e .
```

### 1.2 Import Error: "torch_npu" Required

**Symptom**:
```python
>>> from mhc import MHCLayer
ImportError: torch_npu must be imported before mhc
```

**Solution**: Always import `torch_npu` first:
```python
import torch
import torch_npu  # MUST be before mhc
from mhc import MHCLayer
```

### 1.3 Build Failure: CANN Toolkit Not Found

**Symptom**:
```
CMake Error: CANN toolkit not found
```

**Solution**: Ensure CANN toolkit is installed and environment is set:
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash scripts/build.sh
```

---

## 2. Runtime Errors

### 2.1 RuntimeError: Sinkhorn requires square matrices

**Symptom**:
```
RuntimeError: Sinkhorn-Knopp requires square matrices (M == N), got M=8, N=4
```

**Cause**: Sinkhorn-Knopp algorithm only works on square matrices.

**Solution**: Ensure input tensor has shape `[B, N, N]` (not `[B, M, N]` where M ≠ N).

### 2.2 NPU Vector Core Exception

**Symptom**:
```
[ERROR] RUNTIME: vector core exception
aclrtSynchronizeDevice failed, error code 507035
```

**Cause**: Memory alignment violation in kernel (e.g., DataCopy with unaligned address).

**Impact**: NPU device enters error state; all subsequent operations fail.

**Solution**:
1. Wait 5-10 seconds for device to recover
2. If using small C values (< 16), try larger C
3. Report the specific (C, n) configuration that caused the crash

### 2.3 Numerical Errors (Large Differences)

**Symptom**: Forward/backward results differ significantly from expected values.

**Diagnosis Steps**:

1. **Check input constraints**:
   ```python
   # Sinkhorn requires non-negative input
   inp = torch.rand(...).npu()  # CORRECT: [0, 1)
   inp = torch.randn(...).npu() # WRONG: may have negatives
   ```

2. **Try single-core mode**:
   ```bash
   MHC_MULTICORE=1 python your_script.py
   ```

3. **Force scalar implementation**:
   ```bash
   MHC_SINKHORN_FWD_IMPL=scalar MHC_SINKHORN_BWD_IMPL=scalar python your_script.py
   ```

4. **Test standalone operators**:
   ```python
   from mhc import sinkhorn_knopp

   # Simple test
   inp = torch.rand(1, 4, 4, device='npu')
   out = sinkhorn_knopp(inp, num_iters=20)

   # Check doubly-stochastic property
   print("Row sums:", out.sum(dim=-1))   # Should be ~1.0
   print("Col sums:", out.sum(dim=-2))   # Should be ~1.0
   ```

### 2.4 NaN or Inf in Output

**Symptom**: Model outputs contain NaN or Inf values.

**Possible Causes**:

1. **Input contains NaN/Inf**: Check input data
2. **Numerical instability**: Try smaller learning rate, gradient clipping
3. **Configuration issue**: Some (C, n) combinations have known issues

**Diagnostic**:
```python
# Check for NaN/Inf
def check_tensor(t, name):
    if torch.isnan(t).any():
        print(f"{name} contains NaN")
    if torch.isinf(t).any():
        print(f"{name} contains Inf")
    print(f"{name}: min={t.min():.6f}, max={t.max():.6f}")

# Use in forward hook
def debug_hook(module, input, output):
    check_tensor(output, type(module).__name__)
```

---

## 3. Performance Issues

### 3.1 Slower Than Expected

**Symptom**: Training speed lower than pure PyTorch implementation.

**Diagnosis**:

1. **Check multi-core setting**:
   ```bash
   echo $MHC_MULTICORE  # Should be 8 or unset
   ```

2. **Check implementation mode**:
   ```bash
   echo $MHC_SINKHORN_FWD_IMPL  # Should be "auto" or unset
   ```

3. **Profile to identify bottleneck**:
   ```python
   import time

   # Time individual operations
   start = time.time()
   out = sinkhorn_knopp(inp, num_iters=20)
   torch.npu.synchronize()
   print(f"Sinkhorn: {(time.time() - start) * 1000:.2f} ms")
   ```

**Expected Performance**:
- 8-core mode: ~860 tokens/sec (1.35x pure PyTorch)
- 1-core mode: ~200 tokens/sec (0.3x pure PyTorch)

### 3.2 Memory Issues

**Symptom**: Out of memory errors.

**Solutions**:

1. **Reduce batch size**
2. **Use gradient checkpointing** (if available in your framework)
3. **Check for memory leaks**:
   ```python
   import torch
   print(f"Allocated: {torch.npu.memory_allocated() / 1e9:.2f} GB")
   print(f"Cached: {torch.npu.memory_reserved() / 1e9:.2f} GB")
   ```

---

## 4. Debugging Techniques

### 4.1 Single-Core Debug Mode

Isolate multi-core issues:
```bash
MHC_MULTICORE=1 python your_script.py
```

### 4.2 Force Scalar Implementations

Eliminate vectorization issues:
```bash
MHC_SINKHORN_FWD_IMPL=scalar \
MHC_SINKHORN_BWD_IMPL=scalar \
MHC_USE_SCALAR_MATMUL=1 \
python your_script.py
```

### 4.3 Test Individual Operators

```python
import torch
import torch_npu
from mhc import sinkhorn_knopp, rmsnorm, stream_aggregate, stream_distribute_mix_add

# Test each operator in isolation
def test_sinkhorn(N=4, B=8):
    inp = torch.rand(B, N, N, device='npu', requires_grad=True)
    out = sinkhorn_knopp(inp, num_iters=20)

    # Forward check
    assert not torch.isnan(out).any(), "NaN in output"
    row_sum_err = (out.sum(-1) - 1.0).abs().max()
    col_sum_err = (out.sum(-2) - 1.0).abs().max()
    print(f"Sinkhorn N={N}: row_err={row_sum_err:.6f}, col_err={col_sum_err:.6f}")

    # Backward check
    loss = out.sum()
    loss.backward()
    assert not torch.isnan(inp.grad).any(), "NaN in gradient"
    print(f"  Gradient: min={inp.grad.min():.6f}, max={inp.grad.max():.6f}")

test_sinkhorn(N=4)
test_sinkhorn(N=8)
```

### 4.4 Compare with PyTorch Reference

```python
def pytorch_sinkhorn(inp, num_iters=20, eps=1e-8):
    """Pure PyTorch reference implementation"""
    A = inp.clone()
    for _ in range(num_iters):
        # Row normalization
        A = A / (A.sum(dim=-1, keepdim=True) + eps)
        # Column normalization
        A = A / (A.sum(dim=-2, keepdim=True) + eps)
    return A

# Compare
inp = torch.rand(8, 4, 4, device='npu')
out_ascend = sinkhorn_knopp(inp, num_iters=20)
out_pytorch = pytorch_sinkhorn(inp.clone(), num_iters=20)

diff = (out_ascend - out_pytorch).abs().max()
print(f"Max difference: {diff:.6e}")
```

### 4.5 Subprocess Isolation for Crash-prone Tests

NPU exceptions can poison the process. Use subprocess for risky tests:

```python
import subprocess
import sys

def test_config_isolated(C, n):
    code = f'''
import torch
import torch_npu
from mhc import stream_aggregate

inp = torch.randn(4, {n}, {C}, device='npu')
H = torch.randn(4, {n}, device='npu')
try:
    out = stream_aggregate(inp, H)
    torch.npu.synchronize()
    print("SUCCESS")
except Exception as e:
    print(f"FAIL: {{e}}")
'''
    result = subprocess.run(
        [sys.executable, '-c', code],
        capture_output=True, text=True, timeout=30
    )
    return "SUCCESS" in result.stdout

# Test various configurations
for C in [4, 8, 16, 32]:
    for n in [2, 4, 8]:
        ok = test_config_isolated(C, n)
        print(f"C={C}, n={n}: {'✓' if ok else '✗'}")
```

---

## 5. Environment Variables Reference

| Variable | Values | Default | Purpose |
|----------|--------|---------|---------|
| `MHC_MULTICORE` | 1-32 | 8 | Number of AI cores |
| `MHC_SINKHORN_FWD_IMPL` | auto, scalar, vectorized | auto | Sinkhorn forward path |
| `MHC_SINKHORN_BWD_IMPL` | auto, scalar, vectorized, n4 | auto | Sinkhorn backward path |
| `MHC_USE_SCALAR_MATMUL` | 0, 1 | 0 | MatMul implementation |

---

## 6. Getting Help

### 6.1 Information to Include in Bug Reports

1. **Hardware/Software Environment**:
   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}')"
   python -c "import torch_npu; print(f'torch_npu: {torch_npu.__version__}')"
   cat /usr/local/Ascend/ascend-toolkit/latest/version.info
   ```

2. **Configuration**:
   - C (hidden_dim)
   - n (expansion_rate)
   - M, N (if using Sinkhorn directly)
   - Batch size
   - Environment variable settings

3. **Error Details**:
   - Full error message / traceback
   - Numerical error values (if applicable)
   - Whether error occurs in forward or backward

4. **Minimal Reproduction Code**:
   ```python
   import torch
   import torch_npu
   from mhc import ...

   # Minimal code that reproduces the issue
   ```

### 6.2 Debug Checklist

- [ ] Is `torch_npu` imported before `mhc`?
- [ ] Are inputs on NPU device (`.npu()`)?
- [ ] For Sinkhorn: Is input non-negative and square?
- [ ] Have you tried single-core mode (`MHC_MULTICORE=1`)?
- [ ] Have you tried scalar implementations?
- [ ] Does the issue reproduce with a minimal test case?
- [ ] Does the issue occur with typical configurations (C=64, n=4)?
