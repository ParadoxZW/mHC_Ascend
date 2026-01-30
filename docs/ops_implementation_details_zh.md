# 算子实现细节

本文档提供所有 mHC Ascend 算子的详细实现信息，包括计算路径、遇到的技术挑战和采用的解决方案。

---

## 1. 算子概览

### 1.1 独立算子

| 算子 | 源文件 | 描述 |
|----------|-------------|-------------|
| Sinkhorn-Knopp | `sinkhorn_knopp.cpp` | 双随机矩阵归一化 |
| RMSNorm | `rmsnorm.cpp` | Root Mean Square 归一化 |
| Stream Aggregate | `stream_ops.cpp` | 加权流聚合（带 sigmoid） |
| Stream Distribute Mix Add | `stream_ops.cpp` | 分发、混合和残差加法 |
| Compute RMS | `compute_rms.cpp` | 独立 RMS 计算 |
| Fused RMSNorm MatMul | `fused_ops.cpp` | BFloat16 矩阵乘法 |

### 1.2 融合层算子

| 函数 | 描述 |
|----------|-------------|
| `mhc_layer_fwd_dynamic` | 编排所有内核的前向传播 |
| `mhc_layer_bwd_dynamic` | 编排所有内核的反向传播 |

---

## 2. Sinkhorn-Knopp 内核

### 2.1 算法

Sinkhorn-Knopp 算法通过交替归一化将非负矩阵归一化为双随机矩阵（行和列的和都为 1）：

```
for iter in range(num_iters):
    # 行归一化
    A[i, :] = A[i, :] / sum(A[i, :])
    # 列归一化
    A[:, j] = A[:, j] / sum(A[:, j])
```

### 2.2 前向实现路径

内核提供三条实现路径，根据输入配置自动选择：

| 路径 | 函数 | 特点 |
|------|----------|-----------------|
| **v1_Stable** | `ProcessSingleMatrix_v1_Stable` | 纯标量，保证正确，用于验证 |
| **v2_Optimized** | `ProcessSingleMatrix_v2_Optimized` | N=4,8,16 的循环展开，无 Mask API |
| **v3_Vectorized** | `ProcessSingleMatrix_v3_Vectorized` | 使用 Mask API（SetMaskCount, SetVectorMask）进行 SIMD |

**分发逻辑**（`ProcessSingleMatrix`）：

```cpp
// 由 MHC_SINKHORN_FWD_IMPL 环境变量控制
if (impl_mode == SCALAR) {
    ProcessSingleMatrix_v2_Optimized(batch_idx);
} else if (impl_mode == VECTORIZED) {
    // 多核 + N%4!=0 的安全回退
    if ((N % 4 == 0) || (used_core_num == 1)) {
        ProcessSingleMatrix_v3_Vectorized(batch_idx);
    } else {
        ProcessSingleMatrix_v2_Optimized(batch_idx);  // 回退
    }
} else {  // AUTO
    if ((N % 4 == 0) || (used_core_num == 1)) {
        ProcessSingleMatrix_v3_Vectorized(batch_idx);
    } else {
        ProcessSingleMatrix_v2_Optimized(batch_idx);
    }
}
```

### 2.3 反向实现路径

| 路径 | 函数 | 特点 |
|------|----------|-----------------|
| **Scalar** | `ProcessSingleBackward_Scalar` | O(T) 复杂度，使用 history 缓冲区 |
| **Vectorized** | `ProcessSingleBackward_Vectorized` | 使用 Mask API |
| **N4 Optimized** | `ProcessSingleBackward_N4_Optimized` | 专为 M=N=4 设计，类寄存器存储 |

**分发逻辑**：
```cpp
if (M == 4 && N == 4) {
    ProcessSingleBackward_N4_Optimized(batch_idx);  // 寄存器优化
} else if ((N % 4 == 0) || (used_core_num == 1)) {
    ProcessSingleBackward_Vectorized(batch_idx);
} else {
    ProcessSingleBackward_Scalar(batch_idx);
}
```

### 2.4 关键实现细节

**内存布局**：
- 输入：全局内存（GM）中的 `[B, M, N]`
- 内部：统一缓冲区（UB）中的 `[M, N_aligned]`，其中 `N_aligned = ALIGN_UP(N, 8)`
- 每行起始于 32 字节对齐地址以支持向量化

**多核工作分配**：
```cpp
int32_t batch_per_core = CeilingDiv(batch_size, num_cores);
int32_t batch_start = core_idx * batch_per_core;
int32_t batch_count = min(batch_per_core, batch_size - batch_start);
```

**GM 写回**：
```cpp
// 重要：使用 DataCopyPad 而不是 SetValue 以确保多核安全
AscendC::DataCopyExtParams params = {1, N * sizeof(T), 0, 0, 0};
AscendC::DataCopyPad(outGm[gm_offset], matrix[ub_offset], params);
```

### 2.5 挑战与解决方案

| 挑战 | 解决方案 |
|-----------|----------|
| **多核 Mask 干扰** | N%4!=0 使用标量路径；N%4==0 向量化安全 |
| **ZeroBuffer 多核问题** | 使用标量循环代替 ZeroBuffer（Duplicate） |
| **SetValue 多核写入** | 所有 GM 写入改用 DataCopyPad |
| **反向 O(T²) 复杂度** | Phase 2 优化：保存 history 缓冲区，实现 O(T) |

### 2.6 输入约束

- **必须为方阵**：M 必须等于 N（通过 `TORCH_CHECK` 强制）
- **非负输入**：负值会导致算法不收敛

---

## 3. RMSNorm 内核

### 3.1 算法

```
rms = sqrt(mean(x²) + eps)
out = weight * (x / rms)
```

### 3.2 实现

使用对齐安全数据传输的**单一实现**：

```cpp
// 使用 DataCopyPad 加载以确保对齐安全
DataCopyPadExtParams<DataT> padParams = {false, 0, pad_count, 0};
DataCopyPad(inp, inpGm[offset], copyParams, padParams);

// BF16 → F32 转换用于计算
ConvertBF16ToF32(inpF32, inp, hidden_dim_aligned);

// 计算 RMS
Mul(squared, inpF32, inpF32, hidden_dim_aligned);  // x²
sum = ReduceSum(squared, hidden_dim);              // sum(x²)
rms = sqrt(sum / hidden_dim + eps)

// 归一化和缩放
Divs(normalized, inpF32, rms, hidden_dim);         // x / rms
Mul(outF32, normalized, weightF32, hidden_dim);    // * weight

// F32 → BF16 转换用于输出
ConvertF32ToBF16(out, outF32, hidden_dim);
```

### 3.3 关键细节

- **输入类型**：BFloat16
- **计算类型**：Float32（为了精度）
- **输出类型**：BFloat16
- **权重**：BFloat16，每核加载一次并转换为 F32

---

## 4. Stream Aggregate 内核

### 4.1 算法

```
H_activated = sigmoid(H_pre_raw)
out = sum(H_activated[i] * inp[i, :] for i in range(n))
```

### 4.2 实现

具有自动路径选择的**对齐感知实现**：

```cpp
// 确定 C 维度是否需要填充
C_padded = ALIGN_UP(C, 8);  // 8 个 float = 32 字节
needs_C_padding = (C != C_padded);

// 对齐处理的输入加载
if (needs_C_padding) {
    // 逐行加载并填充
    for (int row = 0; row < n; row++) {
        LoadGmToQueue(inp[row * C_padded], inpGm[row * C], C);
        // 零填充
        for (int c = C; c < C_padded; c++) {
            inp.SetValue(row * C_padded + c, 0.0f);
        }
    }
} else {
    // 快速路径：直接 DataCopy
    DataCopy(inp, inpGm[offset], n * C);
}

// 使用 C_padded 进行向量操作的加权求和
for (int i = 0; i < n; i++) {
    h_val = H_activated.GetValue(i);
    Axpy(outF32, inp[i * row_stride], h_val, vec_len);  // out += h * inp[i]
}
```

### 4.3 LoadGmToQueue 辅助函数

来自官方 Ascend C 示例（`DataCopyPadCustom_GM2UB`）的模式：

```cpp
void LoadGmToQueue(LocalTensor dst, GlobalTensor src, int32_t count) {
    const int32_t align_count = 32 / sizeof(T);  // float32 为 8
    int32_t aligned_count = (count / align_count) * align_count;

    if (aligned_count > 0) {
        DataCopy(dst, src, aligned_count);  // 复制对齐部分
    }
    // 标量复制余数
    for (int32_t i = aligned_count; i < count; i++) {
        dst.SetValue(i, src.GetValue(i));
    }
}
```

### 4.4 挑战与解决方案

| 挑战 | 解决方案 |
|-----------|----------|
| **C < 16 导致 CRASH** | 使用 LoadGmToQueue 的填充缓冲区布局 |
| **n*C 非 32B 对齐** | 逐行加载并填充 |
| **输出未对齐** | 使用 DataCopyPad 写回 GM |

---

## 5. Stream Distribute Mix Add 内核

### 5.1 算法

```
# 分发
y_dist[i] = 2 * sigmoid(H_post_raw[i]) * y_norm

# 混合
mix_out[i] = sum(M[i, j] * x_inp[j, :] for j in range(n))

# 输出
out[i] = y_dist[i] + mix_out[i]
```

### 5.2 前向实现

支持任意 C 和 n 的**对齐安全实现**：

```cpp
// 填充维度
C_padded = ALIGN_UP(C, 16);  // BF16 为 16，F32 为 8

// y_norm 加载（BF16，C 个元素）
if (C * 2 < 32) {  // < 32 字节
    // 标量加载
    for (int c = 0; c < C; c++) {
        y_norm.SetValue(c, y_normGm.GetValue(batch_idx * C + c));
    }
    // 零填充
    for (int c = C; c < C_padded; c++) {
        y_norm.SetValue(c, 0);
    }
} else {
    LoadGmToQueue(y_norm, y_normGm[offset], C);
}

// M 矩阵加载（F32，n×n 个元素）
LoadGmToQueue(M_t, MGm[offset], n * n);

// x_inp 加载（BF16，n×C 个元素）
// 逐行加载并填充
for (int row = 0; row < n; row++) {
    LoadGmToQueue(x_inp[row * C_padded], x_inpGm[row * C], C);
}

// 向量操作使用 C_padded 长度
Muls(y_dist, y_norm, 2.0f * sigmoid(H_post), C_padded);

// 输出写入（逐行 DataCopyPad）
for (int row = 0; row < n; row++) {
    DataCopyExtParams params = {1, C * sizeof(OutT), 0, 0, 0};
    DataCopyPad(outGm[row * C], out[row * C_padded], params);
}
```

### 5.3 反向实现

类似的对齐处理加上类型转换：

```cpp
// 重要：y_norm 必须在内核调用前从 BF16 转换为 F32
auto y_norm_f32 = y_norm.to(at::kFloat).contiguous();
// ... 将 y_norm_f32.data_ptr() 传递给内核
```

### 5.4 挑战与解决方案

| 挑战 | 解决方案 |
|-----------|----------|
| **C=4 CRASH** | 对小 C 使用标量 GetValue |
| **C=8 结果错误** | 使用 C_padded=16 最小值 |
| **n%4!=0 结果错误** | 逐行加载并填充 |
| **y_norm BF16 类型不匹配** | 在 bindings.cpp 中显式类型转换 |

---

## 6. Fused RMSNorm MatMul 内核

### 6.1 算法

```
out = x @ weight.T  # [B, hidden_dim] @ [out_dim, hidden_dim].T -> [B, out_dim]
```

### 6.2 实现

**默认**：PyTorch matmul（使用优化的 CANN 算子）
**可选**：自定义标量内核（用于调试，约慢 33 倍）

```cpp
// 标量内核实现
for (int i = 0; i < out_dim; i++) {
    float dot_product = 0.0f;
    for (int j = 0; j < hidden_dim; j++) {
        // BF16 -> F32 手动转换
        uint16_t w_bits = *(uint16_t*)&weight[i * hidden_dim + j];
        float w_f32 = bit_cast<float>(w_bits << 16);
        dot_product += inp_f32[j] * w_f32;
    }
    result[i] = dot_product;
}
```

### 6.3 为什么默认使用 PyTorch？

| 实现 | 时间（B=64, hidden=256, out=28） |
|---------------|--------------------------------|
| PyTorch matmul | 0.036 ms |
| 自定义标量 | 1.082 ms |

PyTorch 使用 Ascend 的 Cube Unit（矩阵引擎），已高度优化。

---

## 7. Compute RMS 内核

### 7.1 算法

```
rms = sqrt(mean(x²) + eps)
```

### 7.2 实现

RMSNorm 中 RMS 计算的独立版本：

```cpp
// BF16 输入
DataCopyPad(inp, inpGm[offset], copyParams, padParams);
ConvertBF16ToF32(inpF32, inp, K_aligned);

// 计算
Mul(squared, inpF32, inpF32, K_aligned);
sum = ReduceSum(squared, K);
rms = sqrt(sum / K + eps);

// 输出 F32 RMS 值
rmsGm.SetValue(batch_idx, rms);
```

---

## 8. 融合层操作

### 8.1 前向（`mhc_layer_fwd_dynamic`）

按顺序编排所有前向内核：

```cpp
// 1. 展平输入
auto x_flat = x.view({B, n * C});

// 2. 计算 RMS
auto rms = compute_rms_fwd(x_flat, eps);

// 3. 投影到 H 值
auto H_proj = fused_rmsnorm_matmul_fwd(x_flat, phi_concat);  // 或 torch::matmul

// 4. 拆分投影
auto H_pre = H_proj.slice(...);
auto H_post = H_proj.slice(...);
auto H_res = H_proj.slice(...);

// 5. 计算 tilde 值：alpha * proj * (1/rms) + b
auto tilde_pre = alpha_pre * H_pre * rms.reciprocal().unsqueeze(-1) + b_pre;

// 6. Sinkhorn 归一化
auto M = sinkhorn_knopp_fwd(exp(H_res), num_iters, eps);

// 7. 流聚合
auto [x_agg, H_pre_activated] = stream_aggregate_fwd(x, tilde_pre);

// 8. RMSNorm
auto [y_norm, rms_h] = rmsnorm_fwd(x_agg, weight, eps);

// 9. 流分发
auto [out, H_post_activated] = stream_distribute_mix_add_fwd(y_norm, tilde_post, M, x);

return {out, rms, x_agg, H_pre_activated, H_post_activated, M, y_norm, x_flat, rms_h};
```

### 8.2 反向（`mhc_layer_bwd_dynamic`）

按逆序编排反向内核：

```cpp
// 1. Stream distribute 反向
auto [grad_x_partial, grad_y_norm, grad_M, grad_H_post] =
    stream_distribute_mix_add_bwd(grad_out, x, y_norm, M, H_post_activated);

// 2. RMSNorm 反向
auto [grad_x_agg, grad_weight] = rmsnorm_bwd(grad_y_norm, x_agg, weight, rms_h);

// 3. Stream aggregate 反向
auto [grad_x_from_agg, grad_H_pre] =
    stream_aggregate_bwd(grad_x_agg, x, H_pre_activated);

// 4. Sinkhorn 反向
auto grad_H_res = sinkhorn_knopp_bwd(grad_M * M, inp_sinkhorn, M, num_iters, eps);

// 5. 合并梯度并通过 ATen 操作计算参数梯度
// ...

return {grad_x, grad_weight, grad_phi_pre, grad_phi_post, grad_phi_res,
        grad_alpha_pre, grad_alpha_post, grad_alpha_res,
        grad_b_pre, grad_b_post, grad_b_res};
```

---

## 9. 通用实现模式

### 9.1 对齐安全加载

```cpp
// 模式：LoadGmToQueue
void LoadGmToQueue(LocalTensor dst, GlobalTensor src, int count) {
    int aligned = (count / 8) * 8;  // 32 字节对齐数量
    if (aligned > 0) DataCopy(dst, src, aligned);
    for (int i = aligned; i < count; i++) {
        dst.SetValue(i, src.GetValue(i));
    }
}
```

### 9.2 对齐安全存储

```cpp
// 模式：条件 DataCopy vs DataCopyPad
bool aligned = (count * sizeof(T)) % 32 == 0;
if (aligned) {
    DataCopy(dstGm[offset], src, count);
} else {
    DataCopyExtParams params = {1, count * sizeof(T), 0, 0, 0};
    DataCopyPad(dstGm[offset], src, params);
}
```

### 9.3 填充缓冲区策略

```cpp
// 模式：向量操作的填充维度
int C_padded = ALIGN_UP(C, 8);  // 对于 F32：8 个元素 = 32 字节
bool needs_padding = (C != C_padded);

// 分配填充缓冲区
pipe.InitBuffer(buf, n * C_padded * sizeof(float));

// 向量操作使用填充长度（填充为零，对 SIMD 安全）
Muls(dst, src, scalar, C_padded);
```

### 9.4 多核工作分配

```cpp
// 模式：跨核批次分配
int batch_per_core = CeilingDiv(batch_size, num_cores);
int batch_start = core_idx * batch_per_core;
int batch_count = min(batch_per_core, batch_size - batch_start);

if (batch_count <= 0) return;  // 此核无工作

// 设置带偏移的 GM 缓冲区
int offset = batch_start * feature_size;
inputGm.SetGlobalBuffer(ptr + offset, batch_count * feature_size);
```

---

## 10. 源文件索引

| 文件 | 内容 |
|------|---------|
| `src/csrc/kernels/sinkhorn_knopp.cpp` | Sinkhorn 前向和反向内核 |
| `src/csrc/kernels/rmsnorm.cpp` | RMSNorm 前向和反向内核 |
| `src/csrc/kernels/stream_ops.cpp` | Stream aggregate 和 distribute 内核 |
| `src/csrc/kernels/fused_ops.cpp` | Fused RMSNorm MatMul 内核 |
| `src/csrc/kernels/compute_rms.cpp` | 独立 RMS 计算内核 |
| `src/csrc/include/mhc_types.h` | Tiling 结构体和类型定义 |
| `src/csrc/include/utils.h` | 辅助宏和函数 |
| `src/python/bindings.cpp` | PyTorch C++ 绑定 |
