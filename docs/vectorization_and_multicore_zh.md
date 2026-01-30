# 向量化与多核支持

本文档介绍 mHC Ascend 中的向量化和多核并行计算支持，包括遇到的技术挑战和采用的解决方案。

---

## 1. 概述

### 1.1 Ascend NPU 架构

华为 Ascend 910B NPU 特性：
- **8 个 AI 核**：能够并行执行的独立处理单元
- **向量单元**：对 256 位（32 字节）数据进行 SIMD 操作
- **统一缓冲区（UB）**：每个核的快速片上内存
- **全局内存（GM）**：所有核可访问的共享内存

### 1.2 与 CUDA 的关键差异

| 方面 | CUDA | Ascend C |
|--------|------|----------|
| 并行模型 | SIMT（warp 内的线程） | 多核（独立核心） |
| 向量化 | 原生 float4 类型 | Mask API（SetMaskCount 等） |
| Mask 状态 | 线程私有 | **全局/共享**（关键问题！） |
| 内存传输 | 合并访问 | **32 字节对齐 DataCopy** |

---

## 2. 多核并行

### 2.1 工作分配

每个内核按批次在核心之间分配工作：

```cpp
int32_t core_idx = GetBlockIdx();
int32_t num_cores = tiling.used_core_num;

int32_t batch_per_core = CeilingDiv(batch_size, num_cores);
int32_t batch_start = core_idx * batch_per_core;
int32_t batch_count = min(batch_per_core, batch_size - batch_start);

if (batch_count <= 0) return;  // 此核无工作
```

### 2.2 环境变量控制

| 变量 | 默认值 | 描述 |
|----------|---------|-------------|
| `MHC_MULTICORE` | 8 | 使用的 AI 核数量（1-32） |

```bash
# 单核模式（用于调试）
MHC_MULTICORE=1 python train.py

# 显式 8 核模式（与默认相同）
MHC_MULTICORE=8 python train.py
```

### 2.3 多核支持状态

| 算子 | 前向 | 反向 | 说明 |
|----------|---------|----------|-------|
| Sinkhorn-Knopp | ✅ 所有 N 值 | ✅ M=N=4 优化 | 2026-01-30 修复 |
| RMSNorm | ✅ | ✅ | 无问题 |
| Stream Aggregate | ✅ | ✅ | 无问题 |
| Stream Distribute Mix Add | ✅ | ✅ | 无问题 |
| Compute RMS | ✅ | N/A | 无问题 |
| Fused MatMul | ✅ | ✅ | 使用 PyTorch 操作 |

---

## 3. 多核挑战与解决方案

### 3.1 Mask API 全局状态干扰

**问题**：向量化 Mask 操作（`SetMaskCount`、`SetVectorMask`、`ResetMask`）修改全局硬件状态。当多个核心并发执行这些操作时，会相互干扰。

**症状**：多核模式下 N%4!=0 时出现 40-80% 数值误差。

**问题代码示例**：
```cpp
// 危险：多核模式下的 Mask 操作
SetMaskCount(N);  // 核 0 设置 N 个元素的 mask
SetVectorMask<float>(mask, mask);  // 核 1 可能会覆盖！
Add(dst, src1, src2, MASK_PLACEHOLDER, N);
```

**解决方案**：基于 N 对齐和核数的自动路径选择：

```cpp
bool can_use_vectorized = (N % 4 == 0) || (used_core_num == 1);
if (can_use_vectorized) {
    ProcessVectorized();  // 安全：N%4==0 意味着固定的 mask 模式
} else {
    ProcessScalar();      // 回退：不使用 Mask API
}
```

**为什么 N%4==0 是安全的**：当 N 能被 4 整除时，mask 模式始终是全 1（完整向量），因此即使多个核心同时设置，结果也是一致的。

### 3.2 ZeroBuffer（Duplicate）多核问题

**问题**：`ZeroBuffer()` 内部调用 `Duplicate()`，对于非对齐长度会使用 Mask 操作。

**症状**：核心之间缓冲区初始化不一致。

**解决方案**：使用标量循环清除缓冲区：

```cpp
// 错误：ZeroBuffer 使用 Duplicate（Mask 干扰）
ZeroBuffer(colSums, N);

// 正确：标量循环
for (int32_t j = 0; j < N; ++j) {
    colSums.SetValue(j, 0.0f);
}
```

### 3.3 GlobalTensor::SetValue 多核写入

**问题**：多个核心调用 `SetValue()` 写入不同的 GM 位置会相互干扰，可能与缓存行冲突有关。

**症状**：部分批次输出为零（写入未持久化）。

**观察到的模式**：
```
Batch 0: 正确（核 0 写入）
Batch 1: 零    （核 1 写入丢失）
Batch 2: 正确（核 2 写入）
Batch 3: 零    （核 3 写入丢失）
```

**解决方案**：所有 GM 写入使用 `DataCopyPad()` 代替 `SetValue()`：

```cpp
// 错误：SetValue 有多核问题
for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
        outGm.SetValue(gm_offset + i * N + j, matrix.GetValue(ub_offset + j));
    }
}

// 正确：DataCopyPad 是可靠的
for (int i = 0; i < M; i++) {
    DataCopyExtParams params = {1, N * sizeof(T), 0, 0, 0};
    DataCopyPad(outGm[gm_offset + i * N], matrix[ub_offset + i * N_aligned], params);
}
```

---

## 4. 向量化

### 4.1 Ascend C 向量操作

可用的向量指令：
- **算术**：`Add`、`Sub`、`Mul`、`Div`、`Muls`、`Adds`
- **数学**：`Exp`、`Log`、`Sqrt`、`Reciprocal`
- **归约**：`ReduceSum`、`ReduceMax`、`ReduceMin`
- **数据移动**：`DataCopy`、`DataCopyPad`

**要求**：对统一缓冲区的操作需要 32 字节对齐的地址和长度。

### 4.2 Mask API 使用

对于非对齐的向量长度，使用 Mask API：

```cpp
// 为 N 个元素设置 mask（N 不一定对齐）
SetMaskCount(N);
uint64_t mask = (1ULL << N) - 1;
SetVectorMask<float>(mask, mask);

// 执行带 mask 的操作
Add(dst, src1, src2, MASK_PLACEHOLDER, N_aligned);

// 重置 mask（重要！）
ResetMask();
```

**注意**：仅在单核模式或 N%4==0 时使用 Mask API。

### 4.3 各算子向量化状态

| 算子 | 前向 | 反向 | 说明 |
|----------|---------|----------|-------|
| Sinkhorn | ✅ v3_Vectorized | ✅ M=N=4 的 N4_Optimized | 自动分发 |
| RMSNorm | ✅ | ✅ | 使用 DataCopyPad |
| Stream Aggregate | ✅ | ✅ | 使用填充缓冲区 |
| Stream Distribute | ✅ | ✅ | 使用填充缓冲区 |
| Fused MatMul | ❌ 标量 | N/A | 默认使用 PyTorch |

---

## 5. 智能计算路径选择

### 5.1 设计原则

1. **正确性优先**：优先选择安全（标量）路径，而非可能错误的向量化路径
2. **自动选择**：根据输入特征选择最优路径
3. **用户覆盖**：环境变量允许强制使用特定路径

### 5.2 Sinkhorn 前向分发

```cpp
// 环境变量：MHC_SINKHORN_FWD_IMPL
// 值：auto（默认）、scalar、vectorized

switch (impl_mode) {
case SCALAR:
    ProcessSingleMatrix_v2_Optimized(batch_idx);
    break;
case VECTORIZED:
    // 多核 + N%4!=0 的安全回退
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

### 5.3 Sinkhorn 反向分发

```cpp
// 环境变量：MHC_SINKHORN_BWD_IMPL
// 值：auto（默认）、scalar、vectorized、n4_optimized

if (M == 4 && N == 4) {
    // 专门的寄存器优化版本
    ProcessSingleBackward_N4_Optimized(batch_idx);
} else if (can_use_vectorized) {
    ProcessSingleBackward_Vectorized(batch_idx);
} else {
    ProcessSingleBackward_Scalar(batch_idx);
}
```

### 5.4 Stream Ops 对齐分发

```cpp
// 基于维度对齐自动选择
bool needs_C_padding = (C % 8 != 0);  // 8 个 float = 32 字节

if (needs_C_padding) {
    // 逐行加载并填充
    // 向量操作使用 C_padded 长度
} else {
    // 快速路径：直接 DataCopy
}
```

---

## 6. 环境变量汇总

| 变量 | 可选值 | 默认值 | 用途 |
|----------|--------|---------|---------|
| `MHC_MULTICORE` | 1-32 | 8 | AI 核数量 |
| `MHC_SINKHORN_FWD_IMPL` | auto, scalar, vectorized | auto | 前向实现 |
| `MHC_SINKHORN_BWD_IMPL` | auto, scalar, vectorized, n4 | auto | 反向实现 |
| `MHC_USE_SCALAR_MATMUL` | 0, 1 | 0 | MatMul 实现 |

### 6.1 推荐设置

**生产环境**：
```bash
# 默认设置是最优的
python train.py
```

**调试数值问题**：
```bash
MHC_MULTICORE=1 \
MHC_SINKHORN_FWD_IMPL=scalar \
MHC_SINKHORN_BWD_IMPL=scalar \
python debug.py
```

---

## 7. 性能特性

### 7.1 多核加速

| 配置 | Tokens/sec | 相对性能 |
|---------------|------------|----------|
| 纯 PyTorch | 639 | 1.0x |
| Ascend C（1 核） | ~200 | 0.3x |
| Ascend C（8 核） | **860** | **1.35x** |

### 7.2 实现对比

| Sinkhorn N=4 | 实现 | 时间 |
|--------------|----------------|------|
| 标量 | v2_Optimized | 基准 |
| 向量化 | v3_Vectorized | ~相同（小矩阵开销主导） |
| N4 优化 | 类寄存器 | ~相同（内存带宽受限） |

**注意**：对于小矩阵（4x4），内核启动开销和内存带宽是主要因素；无论使用哪种实现，计算本身都非常快。

---

## 8. 最佳实践

### 8.1 内核开发者

1. **在多核代码中避免使用 Mask API**，除非 N%4==0 或单核模式
2. **使用 DataCopyPad** 代替 SetValue 进行 GM 写入
3. **使用标量循环** 清除缓冲区，而不是 ZeroBuffer
4. **将缓冲区大小对齐** 到 32 字节以支持向量操作
5. **同时测试单核和多核** 模式

### 8.2 用户

1. **使用默认设置** 进行生产（8 核，auto 实现）
2. **使用单核模式** 调试数值问题
3. **典型配置**（M=N=4, n=4, C>=16）已完全优化
4. **非典型配置** 可能回退到较慢但正确的标量路径

---

## 9. 已知限制

### 9.1 向量化限制

- Mask API 对于任意 N 不完全多核安全
- 小矩阵大小（< 32 字节）需要标量处理
- 无原生 float4 类型（与 CUDA 不同）

### 9.2 多核限制

- GlobalTensor::SetValue 对于多核写入不可靠
- ZeroBuffer/Duplicate 内部使用 Mask
- 并发 GM 访问可能存在缓存行冲突

### 9.3 未来改进

- **等待 Ascend SDK 更新**：线程安全的 Mask API
- **Cube Unit 集成**：对较大矩阵使用矩阵引擎
- **更多特化路径**：N=8、N=16 优化版本
