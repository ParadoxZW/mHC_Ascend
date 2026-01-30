# 当前限制与未来工作

本文档描述 mHC Ascend 实现的当前限制和计划的改进。

---

## 1. 当前限制

### 1.1 输入约束

#### Sinkhorn-Knopp

| 约束 | 要求 | 原因 |
|------------|-------------|--------|
| 方阵 | M == N | Sinkhorn-Knopp 算法要求 |
| 非负输入 | 所有元素 ≥ 0 | 负值会导致算法不收敛 |

**行为**：非方阵输入在 C++ 层抛出 `RuntimeError`。

#### Stream 操作

| 约束 | 推荐值 | 原因 |
|------------|-------------|--------|
| C 维度 | C ≥ 16 | 向量操作的对齐要求 |
| n 维度 | n = 4 或 8 | 经过充分测试的配置 |

**行为**：较小的 C 值使用较慢的标量路径；某些边界情况可能有问题。

### 1.2 已知边界情况

| 配置 | 状态 | 说明 |
|---------------|--------|-------|
| C=12, n=2 | ⚠️ Stream Distribute Forward 可能有 ~1.2 误差 | 单一边界情况 |
| C < 16, 各种 n | ⚠️ MHCLayer 对某些组合可能失败 | 小 C 未充分测试 |
| M ≠ N（Sinkhorn） | ❌ 运行时拒绝 | 无效输入 |

**建议**：生产环境使用典型配置（C ≥ 64, n = 4）。

### 1.3 性能限制

| 组件 | 当前状态 | 影响 |
|-----------|---------------|--------|
| Fused MatMul | 使用 PyTorch（非自定义内核） | 最优，无问题 |
| Sinkhorn 小矩阵 | 内存带宽受限 | 加速潜力有限 |
| 标量回退路径 | 比向量化慢 ~3-4 倍 | 影响非对齐输入 |

### 1.4 硬件要求

- **NPU**：Ascend 910B1（其他型号可能工作但未测试）
- **CANN 工具包**：编译必需
- **torch_npu**：必须在 mhc 之前导入

---

## 2. 验证过的工作配置

### 2.1 完全测试（生产就绪）

| 参数 | 值 |
|-----------|--------|
| M, N（Sinkhorn） | 4, 8, 16（要求 M == N） |
| n（expansion_rate） | 4, 8 |
| C（hidden_dim） | 64, 128, 256, 512, 1024, 2048, 4096 |
| 批次大小 | 1 到 128 |
| num_sinkhorn_iters | 10-20 |

### 2.2 部分测试

| 参数 | 值 | 说明 |
|-----------|--------|-------|
| C | 16-63 | 某些边界情况可能有问题 |
| n | 2, 3, 5, 6, 7 | 较少见，可能使用标量路径 |
| M=N | 2, 3, 5, 6, 7 | 单核验证通过，多核情况不一 |

### 2.3 不推荐

| 配置 | 问题 |
|---------------|-------|
| C < 16 | 频繁标量回退，部分失败 |
| n = 2 与小 C | 边界情况问题 |
| expansion_rate = 2 | 有记录的边界问题 |

---

## 3. Bug 跟踪

### 3.1 待解决问题

| ID | 描述 | 影响 | 临时方案 |
|----|-------------|--------|------------|
| B1 | Stream Distribute Forward C=12,n=2 误差 | 低（单一配置） | 避免此特定配置 |
| B2 | MHCLayer 小 C 反向失败 | 低（非典型） | 使用 C ≥ 16 |

### 3.2 已修复问题（历史参考）

| ID | 描述 | 修复时间 | 解决方案 |
|----|-------------|----------|----------|
| F1 | Sinkhorn Forward 多核 N%4!=0 | 2026-01-30 | DataCopyPad + 标量 ZeroBuffer |
| F2 | Sinkhorn Backward 数学错误 | 2026-01-25 | 正确的归一化反向公式 |
| F3 | Stream Aggregate 对齐 CRASH | 2026-01-29 | LoadGmToQueue + 填充缓冲区 |
| F4 | Stream Distribute 对齐 CRASH | 2026-01-29 | LoadGmToQueue + 填充缓冲区 |
| F5 | Stream Distribute BWD 类型不匹配 | 2026-01-29 | bindings 中 BF16→F32 转换 |
| F6 | DataCopy 32B 对齐截断 | 2026-01-26 | 对齐安全加载模式 |

---

## 4. 未来工作

### 4.1 优先级 1：正确性

- [ ] **修复 C=12, n=2 边界情况**（Stream Distribute Forward）
- [ ] **改进小 C 支持**（MHCLayer C < 16）
- [ ] **在 Python 层添加输入验证**（所有算子）

### 4.2 优先级 2：性能

- [ ] **Sinkhorn N=8 优化反向**（类似 N=4）
- [ ] **Cube Unit 集成**（大矩阵操作）
- [ ] **大批次训练的内存优化**

### 4.3 优先级 3：功能

- [ ] **Static H 模式** 完整实现（目前仅测试 Dynamic H）
- [ ] **混合精度训练**（FP16 累加选项）
- [ ] **推理优化**（减少 Sinkhorn 迭代次数）

### 4.4 优先级 4：基础设施

- [ ] **全面的单元测试**（覆盖所有 C, n, M, N 组合）
- [ ] **持续集成**（Ascend 硬件）
- [ ] **性能基准测试套件**
- [ ] **大模型内存分析**

---

## 5. 贡献指南

### 5.1 报告问题

报告问题时，请包含：

1. **配置**：C, n, M, N, batch_size 值
2. **环境**：CANN 版本、torch_npu 版本、NPU 型号
3. **错误信息**：完整的错误堆栈或数值误差值
4. **最小复现**：能触发问题的最小代码

### 5.2 测试新配置

使用非标准配置前：

```python
# 先测试独立算子
from mhc import sinkhorn_knopp, stream_aggregate, stream_distribute_mix_add

# 1. 测试 Sinkhorn（M 必须等于 N）
inp = torch.rand(8, N, N, device='npu')  # 非负！
out = sinkhorn_knopp(inp, num_iters=20)
print(f"Sinkhorn N={N}: row_sum={out.sum(-1).mean():.6f}")  # 应该 ~1.0

# 2. 测试 Stream Aggregate
inp = torch.randn(8, n, C, device='npu')
H = torch.randn(8, n, device='npu')
out = stream_aggregate(inp, H)
print(f"Stream Aggregate n={n}, C={C}: shape={out.shape}")

# 3. 测试 Stream Distribute
# ... 类似模式
```

### 5.3 代码风格

- 遵循 `src/csrc/kernels/` 中的现有代码模式
- 使用对齐安全的加载/存储模式
- 为非显而易见的代码添加详细注释
- 同时测试单核和多核模式

---

## 6. 版本历史

| 版本 | 日期 | 主要变更 |
|---------|------|---------------|
| 0.1.0 | 2026-01-22 | 初始实现 |
| 0.2.0 | 2026-01-25 | Sinkhorn 反向修复，Phase 2 优化 |
| 0.3.0 | 2026-01-27 | Fused MatMul，性能优化 |
| 0.4.0 | 2026-01-29 | Stream ops 对齐修复 |
| 0.5.0 | 2026-01-30 | Sinkhorn 多核修复，文档完善 |

---

## 7. 参考资料

- **mHC 论文**：[Hyper-Connections](https://arxiv.org/abs/2409.19606)（DeepSeek-AI，2024）
- **CUDA 参考**：仓库中的 `src/` 目录
- **Ascend C 文档**：CANN 工具包文档
- **调试笔记**：`debug_notes/` 目录中的详细问题分析
