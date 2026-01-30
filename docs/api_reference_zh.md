# API 参考手册

本文档提供 mHC Ascend Python API 和环境变量的完整参考。

---

## 1. Python API

### 1.1 主要类

#### MHCLayer

核心 mHC 计算模块。

```python
from mhc import MHCLayer

layer = MHCLayer(
    hidden_dim: int,           # 特征维度（C）
    expansion_rate: int = 4,   # 流的数量（n）
    num_sinkhorn_iters: int = 20,
    sinkhorn_eps: float = 1e-8,
    rmsnorm_eps: float = 1e-5,
    use_dynamic_h: bool = True,
    alpha_init: float = 0.01,
)
```

**参数说明**：
| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `hidden_dim` | int | 必需 | 特征维度（C） |
| `expansion_rate` | int | 4 | 流的数量（n） |
| `num_sinkhorn_iters` | int | 20 | Sinkhorn-Knopp 迭代次数 |
| `sinkhorn_eps` | float | 1e-8 | Sinkhorn 数值稳定性 epsilon |
| `rmsnorm_eps` | float | 1e-5 | RMSNorm 数值稳定性 epsilon |
| `use_dynamic_h` | bool | True | 使用动态（输入相关）H 计算 |
| `alpha_init` | float | 0.01 | alpha 参数的初始化缩放 |

**输入/输出**：
| 方向 | 形状 | 类型 | 描述 |
|-----------|-------|------|-------------|
| 输入 | `[B, n, C]` | float32 | n 个流的批次，每个流有 C 个特征 |
| 输出 | `[B, n, C]` | float32 | 处理后的流 |

**示例**：
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

Transformer 中残差连接的直接替换。

```python
from mhc import MHCResidualWrapper

wrapper = MHCResidualWrapper(
    hidden_dim: int,
    expansion_rate: int = 4,
    residual_mode: str = 'decay',  # 'zero' 或 'decay'
    **mhc_kwargs,
)
```

**参数说明**：
| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `hidden_dim` | int | 必需 | 特征维度 |
| `expansion_rate` | int | 4 | 流的数量 |
| `residual_mode` | str | 'decay' | 额外流的初始化方式：'zero' 或 'decay' |
| `**mhc_kwargs` | | | 传递给 MHCLayer 的额外参数 |

**输入/输出**：
| 方向 | 形状 | 类型 | 描述 |
|-----------|-------|------|-------------|
| residual | `[B, C]` 或 `[B, T, C]` | float32 | 残差输入 |
| branch_output | 与 residual 相同 | float32 | 分支输出（注意力/FFN） |
| 输出 | 与输入相同 | float32 | mHC 处理后的结果 |

**示例**：
```python
from mhc import MHCResidualWrapper

# 在 Transformer 块中使用
mhc = MHCResidualWrapper(hidden_dim=512, expansion_rate=4).npu()

# 替换: output = residual + attention_output
output = mhc(residual, attention_output)
```

---

### 1.2 独立算子

#### sinkhorn_knopp

通过 Sinkhorn-Knopp 算法进行双随机矩阵归一化。

```python
from mhc import sinkhorn_knopp

output = sinkhorn_knopp(
    inp: Tensor,        # [B, N, N] float32，非负方阵
    num_iters: int = 20,
    eps: float = 1e-8,
) -> Tensor  # [B, N, N] float32，双随机矩阵
```

**约束条件**：
- 输入必须是**方阵**（M == N）。非方阵输入会抛出 `RuntimeError`。
- 输入必须是**非负的**。负值会导致算法不收敛。

**示例**：
```python
from mhc import sinkhorn_knopp

# 创建非负输入（使用 rand，不要用 randn！）
inp = torch.rand(8, 4, 4, device='npu')
out = sinkhorn_knopp(inp, num_iters=20, eps=1e-8)

# 验证双随机性质
print(out.sum(dim=-1))  # 应该接近 1
print(out.sum(dim=-2))  # 应该接近 1
```

---

#### rmsnorm

Root Mean Square 层归一化。

```python
from mhc import rmsnorm

output = rmsnorm(
    inp: Tensor,     # [B, C] float32
    weight: Tensor,  # [C] float32
    eps: float = 1e-5,
) -> Tensor  # [B, C] BFloat16
```

**注意**：由于内核内部设计，输出为 BFloat16。

---

#### stream_aggregate

带 sigmoid 激活的多流加权聚合。

```python
from mhc import stream_aggregate

output = stream_aggregate(
    inp: Tensor,       # [B, n, C] float32
    H_pre_raw: Tensor, # [B, n] float32
) -> Tensor  # [B, C] BFloat16
```

**公式**：`output = Σ sigmoid(H_pre_raw[i]) * inp[i]`，其中 i 从 0 到 n-1

---

#### stream_distribute_mix_add

分发归一化特征、应用混合矩阵并添加残差。

```python
from mhc import stream_distribute_mix_add

output = stream_distribute_mix_add(
    y_norm: Tensor,      # [B, C] float32
    H_post_raw: Tensor,  # [B, n] float32
    M: Tensor,           # [B, n, n] float32
    x_inp: Tensor,       # [B, n, C] float32
) -> Tensor  # [B, n, C] float32
```

**公式**：
- 分发：`y_dist[i] = 2 * sigmoid(H_post_raw[i]) * y_norm`
- 混合：`mix_out[i] = Σ M[i,j] * x_inp[j]`
- 输出：`output = y_dist + mix_out`

---

#### compute_rms

独立的 RMS（Root Mean Square）计算。

```python
from mhc import compute_rms

rms = compute_rms(
    inp: Tensor,  # [B, K] BFloat16
    eps: float = 1e-5,
) -> Tensor  # [B] float32
```

**公式**：`rms = sqrt(mean(inp²) + eps)`

---

### 1.3 工厂函数

#### create_mhc_layer

创建 MHCLayer 的工厂函数。

```python
from mhc import create_mhc_layer

layer = create_mhc_layer(
    hidden_dim: int,
    expansion_rate: int = 4,
    **kwargs,
) -> MHCLayer
```

#### replace_residual_with_mhc

创建 MHCResidualWrapper 的工厂函数。

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

## 2. 环境变量

### 2.1 全局设置

#### MHC_MULTICORE

控制用于并行计算的 AI 核数量。

| 值 | 效果 |
|-------|--------|
| `1` | 单核模式（用于调试） |
| `2-32` | 多核并行（受批次大小限制） |
| 未设置 | 默认：8 核 |

**示例**：
```bash
# 调试模式（单核）
MHC_MULTICORE=1 python train.py

# 生产模式（默认 8 核）
python train.py
```

**注意**：
- 实际使用核数 = `min(MHC_MULTICORE, batch_size)`
- Ascend 910B 通常有 8 个 AI 核

---

### 2.2 Sinkhorn 实现控制

#### MHC_SINKHORN_FWD_IMPL

控制 Sinkhorn 前向实现选择。

| 值 | 行为 |
|-------|----------|
| `auto`（默认） | N%4==0 或单核 → 向量化；否则 → 标量 |
| `scalar` | 强制使用标量（循环展开）实现 |
| `vectorized` | 强制使用向量化（Mask API），带安全回退 |

#### MHC_SINKHORN_BWD_IMPL

控制 Sinkhorn 反向实现选择。

| 值 | 行为 |
|-------|----------|
| `auto`（默认） | M=N=4 → N4 优化；N%4==0 或单核 → 向量化；否则 → 标量 |
| `scalar` | 强制使用标量实现 |
| `vectorized` | 强制使用向量化，带安全回退 |
| `n4_optimized` 或 `n4` | 强制使用 N4 优化（仅当 M=N=4 时有效） |

**示例**：
```bash
# 强制使用标量实现进行调试
MHC_SINKHORN_FWD_IMPL=scalar MHC_SINKHORN_BWD_IMPL=scalar python train.py
```

---

### 2.3 MatMul 实现控制

#### MHC_USE_SCALAR_MATMUL

控制融合 RMSNorm MatMul 实现。

| 值 | 行为 |
|-------|----------|
| `0`（默认） | 使用 PyTorch matmul（优化的 CANN 算子，约快 33 倍） |
| `1` | 使用自定义标量内核（用于调试） |

**注意**：PyTorch matmul 已高度优化，建议用于生产环境。

---

## 3. 环境变量汇总

| 变量 | 默认值 | 可选值 | 用途 |
|----------|---------|--------|---------|
| `MHC_MULTICORE` | `8` | `1-32` | AI 核数量 |
| `MHC_SINKHORN_FWD_IMPL` | `auto` | `auto`, `scalar`, `vectorized` | Sinkhorn 前向路径 |
| `MHC_SINKHORN_BWD_IMPL` | `auto` | `auto`, `scalar`, `vectorized`, `n4`/`n4_optimized` | Sinkhorn 反向路径 |
| `MHC_USE_SCALAR_MATMUL` | `0` | `0`, `1` | MatMul 实现 |

---

## 4. 推荐配置

### 4.1 生产训练

```bash
# 默认设置是最优的
python train.py
```

推荐超参数：
| 参数 | 推荐值 | 说明 |
|-----------|-------------|-------|
| `expansion_rate` | 4 | 经过充分测试的配置 |
| `num_sinkhorn_iters` | 20 | 训练时；推理时可减少到 10-15 |
| `use_dynamic_h` | True | 更好的表达能力 |
| `alpha_init` | 0.01 | 较小的初始化更稳定 |

### 4.2 调试数值问题

```bash
# 单核，标量实现
MHC_MULTICORE=1 \
MHC_SINKHORN_FWD_IMPL=scalar \
MHC_SINKHORN_BWD_IMPL=scalar \
python debug_script.py
```

### 4.3 性能基准测试

```bash
# 显式多核（与默认相同）
MHC_MULTICORE=8 python benchmark.py
```

---

## 5. C++ 模块接口

编译后的 `mhc_ascend` 模块导出以下函数（供高级用户使用）：

```python
import mhc_ascend

# 前向函数
mhc_ascend.sinkhorn_knopp_fwd(inp, num_iters, eps)
mhc_ascend.rmsnorm_fwd(inp, weight, eps)
mhc_ascend.stream_aggregate_fwd(inp, H_pre_raw)
mhc_ascend.stream_distribute_mix_add_fwd(y_norm, H_post_raw, M, x_inp)
mhc_ascend.compute_rms_fwd(inp, eps)
mhc_ascend.fused_rmsnorm_matmul_fwd(inp, weight)

# 反向函数
mhc_ascend.sinkhorn_knopp_bwd(grad_out, inp, out, num_iters, eps)
mhc_ascend.rmsnorm_bwd(grad_out, inp, weight, rms)
mhc_ascend.stream_aggregate_bwd(grad_out, inp, H_pre_activated)
mhc_ascend.stream_distribute_mix_add_bwd(grad_out, x_inp, y_norm, M, H_post_activated)

# 融合层操作
mhc_ascend.mhc_layer_fwd_dynamic(...)  # 返回输出 + 中间结果
mhc_ascend.mhc_layer_bwd_dynamic(...)  # 返回所有梯度
```

**注意**：这些底层函数不提供自动微分支持。请使用 Python API（如 `mhc.sinkhorn_knopp` 等）进行自动微分。
