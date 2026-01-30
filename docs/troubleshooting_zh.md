# 故障排除指南

本文档提供使用 mHC Ascend 时常见问题的解决方案。

---

## 1. 安装问题

### 1.1 导入错误："No module named 'mhc_ascend'"

**症状**：
```python
>>> from mhc import MHCLayer
ModuleNotFoundError: No module named 'mhc_ascend'
```

**原因**：C++ 扩展未编译或未安装。

**解决方案**：
```bash
cd mHC_ascend
bash scripts/build.sh
pip install -e .
```

### 1.2 导入错误：需要 "torch_npu"

**症状**：
```python
>>> from mhc import MHCLayer
ImportError: torch_npu must be imported before mhc
```

**解决方案**：始终先导入 `torch_npu`：
```python
import torch
import torch_npu  # 必须在 mhc 之前
from mhc import MHCLayer
```

### 1.3 编译失败：找不到 CANN 工具包

**症状**：
```
CMake Error: CANN toolkit not found
```

**解决方案**：确保 CANN 工具包已安装并设置环境：
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash scripts/build.sh
```

---

## 2. 运行时错误

### 2.1 RuntimeError: Sinkhorn 需要方阵

**症状**：
```
RuntimeError: Sinkhorn-Knopp requires square matrices (M == N), got M=8, N=4
```

**原因**：Sinkhorn-Knopp 算法仅适用于方阵。

**解决方案**：确保输入张量形状为 `[B, N, N]`（而非 M ≠ N 的 `[B, M, N]`）。

### 2.2 NPU 向量核异常

**症状**：
```
[ERROR] RUNTIME: vector core exception
aclrtSynchronizeDevice failed, error code 507035
```

**原因**：内核中的内存对齐违规（如使用未对齐地址的 DataCopy）。

**影响**：NPU 设备进入错误状态；所有后续操作都会失败。

**解决方案**：
1. 等待 5-10 秒让设备恢复
2. 如果使用小 C 值（< 16），尝试使用更大的 C
3. 报告导致崩溃的具体 (C, n) 配置

### 2.3 数值错误（差异较大）

**症状**：前向/反向结果与预期值显著不同。

**诊断步骤**：

1. **检查输入约束**：
   ```python
   # Sinkhorn 需要非负输入
   inp = torch.rand(...).npu()  # 正确：[0, 1)
   inp = torch.randn(...).npu() # 错误：可能有负值
   ```

2. **尝试单核模式**：
   ```bash
   MHC_MULTICORE=1 python your_script.py
   ```

3. **强制使用标量实现**：
   ```bash
   MHC_SINKHORN_FWD_IMPL=scalar MHC_SINKHORN_BWD_IMPL=scalar python your_script.py
   ```

4. **测试独立算子**：
   ```python
   from mhc import sinkhorn_knopp

   # 简单测试
   inp = torch.rand(1, 4, 4, device='npu')
   out = sinkhorn_knopp(inp, num_iters=20)

   # 检查双随机性质
   print("行和:", out.sum(dim=-1))   # 应该 ~1.0
   print("列和:", out.sum(dim=-2))   # 应该 ~1.0
   ```

### 2.4 输出中出现 NaN 或 Inf

**症状**：模型输出包含 NaN 或 Inf 值。

**可能原因**：

1. **输入包含 NaN/Inf**：检查输入数据
2. **数值不稳定**：尝试更小的学习率、梯度裁剪
3. **配置问题**：某些 (C, n) 组合有已知问题

**诊断方法**：
```python
# 检查 NaN/Inf
def check_tensor(t, name):
    if torch.isnan(t).any():
        print(f"{name} 包含 NaN")
    if torch.isinf(t).any():
        print(f"{name} 包含 Inf")
    print(f"{name}: min={t.min():.6f}, max={t.max():.6f}")

# 在前向钩子中使用
def debug_hook(module, input, output):
    check_tensor(output, type(module).__name__)
```

---

## 3. 性能问题

### 3.1 比预期慢

**症状**：训练速度低于纯 PyTorch 实现。

**诊断**：

1. **检查多核设置**：
   ```bash
   echo $MHC_MULTICORE  # 应该是 8 或未设置
   ```

2. **检查实现模式**：
   ```bash
   echo $MHC_SINKHORN_FWD_IMPL  # 应该是 "auto" 或未设置
   ```

3. **分析以识别瓶颈**：
   ```python
   import time

   # 计时单个操作
   start = time.time()
   out = sinkhorn_knopp(inp, num_iters=20)
   torch.npu.synchronize()
   print(f"Sinkhorn: {(time.time() - start) * 1000:.2f} ms")
   ```

**预期性能**：
- 8 核模式：~860 tokens/sec（纯 PyTorch 的 1.35 倍）
- 1 核模式：~200 tokens/sec（纯 PyTorch 的 0.3 倍）

### 3.2 内存问题

**症状**：内存不足错误。

**解决方案**：

1. **减小批次大小**
2. **使用梯度检查点**（如果框架支持）
3. **检查内存泄漏**：
   ```python
   import torch
   print(f"已分配: {torch.npu.memory_allocated() / 1e9:.2f} GB")
   print(f"已缓存: {torch.npu.memory_reserved() / 1e9:.2f} GB")
   ```

---

## 4. 调试技巧

### 4.1 单核调试模式

隔离多核问题：
```bash
MHC_MULTICORE=1 python your_script.py
```

### 4.2 强制标量实现

排除向量化问题：
```bash
MHC_SINKHORN_FWD_IMPL=scalar \
MHC_SINKHORN_BWD_IMPL=scalar \
MHC_USE_SCALAR_MATMUL=1 \
python your_script.py
```

### 4.3 测试单个算子

```python
import torch
import torch_npu
from mhc import sinkhorn_knopp, rmsnorm, stream_aggregate, stream_distribute_mix_add

# 隔离测试每个算子
def test_sinkhorn(N=4, B=8):
    inp = torch.rand(B, N, N, device='npu', requires_grad=True)
    out = sinkhorn_knopp(inp, num_iters=20)

    # 前向检查
    assert not torch.isnan(out).any(), "输出中有 NaN"
    row_sum_err = (out.sum(-1) - 1.0).abs().max()
    col_sum_err = (out.sum(-2) - 1.0).abs().max()
    print(f"Sinkhorn N={N}: 行误差={row_sum_err:.6f}, 列误差={col_sum_err:.6f}")

    # 反向检查
    loss = out.sum()
    loss.backward()
    assert not torch.isnan(inp.grad).any(), "梯度中有 NaN"
    print(f"  梯度: min={inp.grad.min():.6f}, max={inp.grad.max():.6f}")

test_sinkhorn(N=4)
test_sinkhorn(N=8)
```

### 4.4 与 PyTorch 参考对比

```python
def pytorch_sinkhorn(inp, num_iters=20, eps=1e-8):
    """纯 PyTorch 参考实现"""
    A = inp.clone()
    for _ in range(num_iters):
        # 行归一化
        A = A / (A.sum(dim=-1, keepdim=True) + eps)
        # 列归一化
        A = A / (A.sum(dim=-2, keepdim=True) + eps)
    return A

# 对比
inp = torch.rand(8, 4, 4, device='npu')
out_ascend = sinkhorn_knopp(inp, num_iters=20)
out_pytorch = pytorch_sinkhorn(inp.clone(), num_iters=20)

diff = (out_ascend - out_pytorch).abs().max()
print(f"最大差异: {diff:.6e}")
```

### 4.5 子进程隔离测试可能崩溃的配置

NPU 异常会污染进程。对有风险的测试使用子进程：

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

# 测试各种配置
for C in [4, 8, 16, 32]:
    for n in [2, 4, 8]:
        ok = test_config_isolated(C, n)
        print(f"C={C}, n={n}: {'✓' if ok else '✗'}")
```

---

## 5. 环境变量参考

| 变量 | 可选值 | 默认值 | 用途 |
|----------|--------|---------|---------|
| `MHC_MULTICORE` | 1-32 | 8 | AI 核数量 |
| `MHC_SINKHORN_FWD_IMPL` | auto, scalar, vectorized | auto | Sinkhorn 前向路径 |
| `MHC_SINKHORN_BWD_IMPL` | auto, scalar, vectorized, n4 | auto | Sinkhorn 反向路径 |
| `MHC_USE_SCALAR_MATMUL` | 0, 1 | 0 | MatMul 实现 |

---

## 6. 获取帮助

### 6.1 Bug 报告需包含的信息

1. **硬件/软件环境**：
   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}')"
   python -c "import torch_npu; print(f'torch_npu: {torch_npu.__version__}')"
   cat /usr/local/Ascend/ascend-toolkit/latest/version.info
   ```

2. **配置**：
   - C（hidden_dim）
   - n（expansion_rate）
   - M, N（如果直接使用 Sinkhorn）
   - 批次大小
   - 环境变量设置

3. **错误详情**：
   - 完整错误信息 / 堆栈跟踪
   - 数值误差值（如适用）
   - 错误发生在前向还是反向

4. **最小复现代码**：
   ```python
   import torch
   import torch_npu
   from mhc import ...

   # 能复现问题的最小代码
   ```

### 6.2 调试检查清单

- [ ] `torch_npu` 是否在 `mhc` 之前导入？
- [ ] 输入是否在 NPU 设备上（`.npu()`）？
- [ ] 对于 Sinkhorn：输入是否非负且为方阵？
- [ ] 是否尝试过单核模式（`MHC_MULTICORE=1`）？
- [ ] 是否尝试过标量实现？
- [ ] 问题是否能用最小测试用例复现？
- [ ] 问题是否在典型配置（C=64, n=4）下出现？
