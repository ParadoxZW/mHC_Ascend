/**
 * @file utils.h
 * @brief Utility functions for mHC Ascend implementation
 *
 * Copyright (C) 2025. All rights reserved.
 *
 * This file provides helper functions used across different kernels,
 * including data conversion, reduction operations, and common patterns.
 */

#ifndef MHC_ASCEND_UTILS_H
#define MHC_ASCEND_UTILS_H

#include "kernel_operator.h"
#include "mhc_types.h"

namespace mhc_ascend {

// =============================================================================
// Inline Utility Functions
// =============================================================================

/**
 * @brief Load tiling data from GM into a local struct (Ascend C style)
 *
 * Mirrors the behavior of generated GET_TILING_DATA macros in Ascend samples.
 */
template <typename T>
__aicore__ inline void InitTilingData(GM_ADDR tiling, T* tiling_data)
{
    const __gm__ uint32_t* src = reinterpret_cast<const __gm__ uint32_t*>(tiling);
    uint32_t* dst = reinterpret_cast<uint32_t*>(tiling_data);
    constexpr int32_t kWords = (sizeof(T) + sizeof(uint32_t) - 1) / sizeof(uint32_t);
    for (int32_t i = 0; i < kWords; ++i) {
        dst[i] = src[i];
    }
}

/**
 * @brief Ceiling division (compile-time compatible)
 * @param a Numerator
 * @param b Denominator
 * @return Ceiling of a/b
 */
__aicore__ inline int32_t CeilingDiv(int32_t a, int32_t b) {
    return (a + b - 1) / b;
}

/**
 * @brief Calculate global offset for current AI core
 * @param core_idx Current core index (from GetBlockIdx())
 * @param total_size Total data size
 * @param num_cores Number of cores
 * @return Offset for this core
 */
__aicore__ inline int32_t CalcCoreOffset(int32_t core_idx, int32_t total_size, int32_t num_cores) {
    int32_t per_core = CeilingDiv(total_size, num_cores);
    return core_idx * per_core;
}

/**
 * @brief Calculate data length for current AI core
 * @param core_idx Current core index
 * @param total_size Total data size
 * @param num_cores Number of cores
 * @return Data length for this core
 */
__aicore__ inline int32_t CalcCoreLength(int32_t core_idx, int32_t total_size, int32_t num_cores) {
    int32_t per_core = CeilingDiv(total_size, num_cores);
    int32_t offset = CalcCoreOffset(core_idx, total_size, num_cores);
    return MIN(per_core, total_size - offset);
}

// =============================================================================
// Reduction Operations
// =============================================================================

/**
 * @brief Efficient reduction sum using Ascend C intrinsics
 *
 * Implements multi-level reduction similar to CUDA warp reductions:
 * 1. WholeReduceSum to reduce within repeat units
 * 2. Iterative reduction until single value
 *
 * @param dst Output tensor (single element)
 * @param src Input tensor (length elements)
 * @param length Number of elements to reduce
 * @param tmpBuffer Temporary calculation buffer
 */
template<typename T>
__aicore__ inline void ReduceSumOptimized(
    AscendC::LocalTensor<T> dst,
    AscendC::LocalTensor<T> src,
    AscendC::LocalTensor<T> tmpBuffer,
    int32_t length)
{
    // Use built-in ReduceSum to support arbitrary lengths safely.
    AscendC::ReduceSum(dst, src, tmpBuffer, length);
}

/**
 * @brief Reduce mean (sum / count)
 * @param dst Output tensor (single element)
 * @param src Input tensor
 * @param tmpBuffer Temporary buffer
 * @param length Number of elements
 */
template<typename T>
__aicore__ inline void ReduceMeanOptimized(
    AscendC::LocalTensor<T> dst,
    AscendC::LocalTensor<T> src,
    AscendC::LocalTensor<T> tmpBuffer,
    int32_t length)
{
    ReduceSumOptimized(dst, src, tmpBuffer, length);

    // Divide by count
    T scale = static_cast<T>(1.0) / static_cast<T>(length);
    AscendC::Muls(dst, dst, scale, 1);
}

// =============================================================================
// Data Conversion Utilities
// =============================================================================

/**
 * @brief Convert float32 to bfloat16
 * @param dst Destination (bfloat16)
 * @param src Source (float32)
 * @param length Number of elements
 */
__aicore__ inline void ConvertF32ToBF16(
    AscendC::LocalTensor<floatX> dst,
    AscendC::LocalTensor<float> src,
    int32_t length)
{
    // Use RINT for stable float->bf16 conversion (matches Ascend samples)
    AscendC::Cast(dst, src, AscendC::RoundMode::CAST_RINT, length);
}

/**
 * @brief Convert bfloat16 to float32
 * @param dst Destination (float32)
 * @param src Source (bfloat16)
 * @param length Number of elements
 */
__aicore__ inline void ConvertBF16ToF32(
    AscendC::LocalTensor<float> dst,
    AscendC::LocalTensor<floatX> src,
    int32_t length)
{
    // BF16 -> F32 should use CAST_NONE (per AscendC samples).
    AscendC::Cast(dst, src, AscendC::RoundMode::CAST_NONE, length);
}

// =============================================================================
// Activation Functions
// =============================================================================

/**
 * @brief Reciprocal: dst = 1 / src
 * @param dst Output tensor
 * @param src Input tensor
 * @param tmpOnes Temporary buffer filled with ones
 * @param length Number of elements
 *
 * Note: Ascend C doesn't have Rec(), use Div(dst, ones, src, length) instead
 */
template<typename T>
__aicore__ inline void Reciprocal(
    AscendC::LocalTensor<T> dst,
    AscendC::LocalTensor<T> src,
    AscendC::LocalTensor<T> tmpOnes,
    int32_t length)
{
    // tmpOnes should be pre-filled with 1.0
    // dst = 1.0 / src
    AscendC::Div(dst, tmpOnes, src, length);
}

/**
 * @brief Sigmoid activation: 1 / (1 + exp(-x))
 * @param dst Output tensor
 * @param src Input tensor
 * @param tmpOnes Temporary buffer filled with ones
 * @param length Number of elements
 */
template<typename T>
__aicore__ inline void Sigmoid(
    AscendC::LocalTensor<T> dst,
    AscendC::LocalTensor<T> src,
    AscendC::LocalTensor<T> tmpOnes,
    int32_t length)
{
    // dst = -src
    AscendC::Muls(dst, src, static_cast<T>(-1.0), length);

    // dst = exp(dst) = exp(-src)
    AscendC::Exp(dst, dst, length);

    // dst = dst + 1 = 1 + exp(-src)
    AscendC::Adds(dst, dst, static_cast<T>(1.0), length);

    // dst = 1 / dst = 1 / (1 + exp(-src))
    // tmpOnes should be pre-filled with 1.0
    AscendC::Div(dst, tmpOnes, dst, length);
}

/**
 * @brief Sigmoid activation using Reciprocal (no temp ones buffer)
 * @param dst Output tensor
 * @param src Input tensor
 * @param length Number of elements
 */
template<typename T>
__aicore__ inline void Sigmoid(
    AscendC::LocalTensor<T> dst,
    AscendC::LocalTensor<T> src,
    int32_t length)
{
    // dst = -src
    AscendC::Muls(dst, src, static_cast<T>(-1.0), length);

    // dst = exp(-src)
    AscendC::Exp(dst, dst, length);

    // dst = 1 + exp(-src)
    AscendC::Adds(dst, dst, static_cast<T>(1.0), length);

    // dst = 1 / (1 + exp(-src))
    AscendC::Reciprocal(dst, dst, length);
}

/**
 * @brief Sigmoid backward: sigmoid(x) * (1 - sigmoid(x))
 * @param grad_input Output gradient w.r.t. input
 * @param grad_output Gradient w.r.t. output
 * @param output Forward pass output (sigmoid(x))
 * @param tmpBuffer Temporary buffer
 * @param length Number of elements
 */
template<typename T>
__aicore__ inline void SigmoidBackward(
    AscendC::LocalTensor<T> grad_input,
    AscendC::LocalTensor<T> grad_output,
    AscendC::LocalTensor<T> output,
    AscendC::LocalTensor<T> tmpBuffer,
    int32_t length)
{
    // tmpBuffer = 1 - output
    AscendC::Adds(tmpBuffer, output, static_cast<T>(-1.0), length);
    AscendC::Muls(tmpBuffer, tmpBuffer, static_cast<T>(-1.0), length);

    // tmpBuffer = output * (1 - output)
    AscendC::Mul(tmpBuffer, output, tmpBuffer, length);

    // grad_input = grad_output * tmpBuffer
    AscendC::Mul(grad_input, grad_output, tmpBuffer, length);
}

// =============================================================================
// Memory Management Helpers
// =============================================================================

/**
 * @brief Initialize buffer with constant value
 * @param tensor Tensor to initialize
 * @param value Constant value
 * @param length Number of elements
 */
template<typename T>
__aicore__ inline void FillConstant(
    AscendC::LocalTensor<T> tensor,
    T value,
    int32_t length)
{
    AscendC::Duplicate(tensor, value, length);
}

/**
 * @brief Initialize buffer with zeros
 * @param tensor Tensor to zero out
 * @param length Number of elements
 */
template<typename T>
__aicore__ inline void ZeroBuffer(
    AscendC::LocalTensor<T> tensor,
    int32_t length)
{
    FillConstant(tensor, static_cast<T>(0.0), length);
}

/**
 * @brief Initialize buffer with ones
 * @param tensor Tensor to fill with ones
 * @param length Number of elements
 */
template<typename T>
__aicore__ inline void OnesBuffer(
    AscendC::LocalTensor<T> tensor,
    int32_t length)
{
    FillConstant(tensor, static_cast<T>(1.0), length);
}

// =============================================================================
// Math Utilities
// =============================================================================

/**
 * @brief Reciprocal with epsilon (1 / (x + eps))
 * @param dst Output tensor
 * @param src Input tensor
 * @param tmpOnes Temporary buffer filled with ones
 * @param eps Epsilon for numerical stability
 * @param length Number of elements
 */
template<typename T>
__aicore__ inline void ReciprocalEps(
    AscendC::LocalTensor<T> dst,
    AscendC::LocalTensor<T> src,
    AscendC::LocalTensor<T> tmpOnes,
    T eps,
    int32_t length)
{
    AscendC::Adds(dst, src, eps, length);
    // tmpOnes should be pre-filled with 1.0
    AscendC::Div(dst, tmpOnes, dst, length);
}

/**
 * @brief Square elements (x^2)
 * @param dst Output tensor
 * @param src Input tensor
 * @param length Number of elements
 */
template<typename T>
__aicore__ inline void Square(
    AscendC::LocalTensor<T> dst,
    AscendC::LocalTensor<T> src,
    int32_t length)
{
    AscendC::Mul(dst, src, src, length);
}

/**
 * @brief Root mean square (RMS): sqrt(mean(x^2))
 * @param rms Output RMS value (single element)
 * @param src Input tensor
 * @param tmpBuffer1 Temporary buffer (same size as src)
 * @param tmpBuffer2 Temporary buffer (for reduction)
 * @param length Number of elements
 * @param eps Epsilon for numerical stability
 */
template<typename T>
__aicore__ inline void ComputeRMS(
    AscendC::LocalTensor<T> rms,
    AscendC::LocalTensor<T> src,
    AscendC::LocalTensor<T> tmpBuffer1,
    AscendC::LocalTensor<T> tmpBuffer2,
    int32_t length,
    T eps)
{
    // Force scalar accumulation to avoid ReduceSum alignment issues while debugging RMS accuracy.
    T sum = static_cast<T>(0.0);
    for (int32_t i = 0; i < length; ++i) {
        T v = src.GetValue(i);
        sum += v * v;
    }
    T mean = sum / static_cast<T>(length);
    rms.SetValue(0, mean + eps);
    AscendC::Sqrt(rms, rms, 1);
}

// =============================================================================
// Scalar GM Read/Write Helpers (alignment-safe)
// =============================================================================

template <typename T>
__aicore__ inline void ReadScalarFromGM(
    AscendC::LocalTensor<T> dst,
    AscendC::GlobalTensor<T> src,
    int32_t idx)
{
    AscendC::DataCopyExtParams copyParams = {1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
    AscendC::DataCopyPad<T>(dst, src[idx], copyParams, padParams);
}

template <typename T>
__aicore__ inline void WriteScalarToGM(
    AscendC::GlobalTensor<T> dst,
    int32_t idx,
    AscendC::LocalTensor<T> src)
{
    AscendC::DataCopyExtParams copyParams = {1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPad<T>(dst[idx], src, copyParams);
}

// =============================================================================
// Debugging Utilities (CPU-side only)
// =============================================================================

#ifndef __aicore__

/**
 * @brief Print tensor info (for debugging on CPU)
 */
template<typename T>
inline void PrintTensorInfo(const char* name, const T* data, int32_t length, int32_t max_print = 10) {
    printf("%s: length=%d, first %d elements: [", name, length, max_print);
    for (int32_t i = 0; i < MIN(length, max_print); ++i) {
        printf("%.6f", static_cast<float>(data[i]));
        if (i < MIN(length, max_print) - 1) printf(", ");
    }
    if (length > max_print) printf(", ...");
    printf("]\n");
}

#endif // __aicore__

} // namespace mhc_ascend

#endif // MHC_ASCEND_UTILS_H
