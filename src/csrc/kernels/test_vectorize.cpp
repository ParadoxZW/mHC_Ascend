/**
 * @file test_vectorize.cpp
 * @brief Test kernel for exploring Ascend C vectorization with small tensors
 *
 * Copyright (C) 2026. All rights reserved.
 *
 * Purpose: Systematically test different vectorization approaches for small tensors
 * (e.g., n=4 float32 = 16 bytes, which is less than 32B alignment requirement)
 *
 * Test scenarios:
 * 1. Add two vectors of size N (N < 8 floats = 32B)
 * 2. Reduce sum of vector of size N
 * 3. Row normalization (sum + divide)
 *
 * Methods to test:
 * A. Scalar baseline (GetValue/SetValue) - known to work
 * B. SetMaskCount + SetVectorMask + MASK_PLACEHOLDER
 * C. Pad to 32B alignment, compute, extract results
 * D. DataCopyPad for load/store
 */

#include "kernel_operator.h"

using namespace AscendC;

// =============================================================================
// Constants
// =============================================================================
constexpr int32_t BLK_LEN = 32;  // Minimum alignment unit (bytes)

// =============================================================================
// Tiling structure for test kernel
// =============================================================================
struct VectorizeTestTiling {
    int32_t batch_size;
    int32_t N;           // Vector length (e.g., 4, 8, 16)
    int32_t test_mode;   // Which test method to use (0=scalar, 1=mask, 2=pad)
    int32_t used_core_num;
};

// =============================================================================
// Test Kernel Class
// =============================================================================

template<typename T = float>
class VectorizeTestKernel {
public:
    __aicore__ inline VectorizeTestKernel() {}

    /**
     * @brief Initialize kernel
     * @param x_gm Input vector 1 [batch, N]
     * @param y_gm Input vector 2 [batch, N]
     * @param out_gm Output vector [batch, N]
     * @param tiling Tiling configuration
     *
     * Test operations:
     * - out = x + y (vector add)
     * - sum(x) stored in out[0] (reduce)
     */
    __aicore__ inline void Init(
        GM_ADDR x_gm,
        GM_ADDR y_gm,
        GM_ADDR out_gm,
        const VectorizeTestTiling& tiling)
    {
        this->N = tiling.N;
        this->test_mode = tiling.test_mode;

        // Calculate aligned N for buffer allocation
        int32_t elems_per_blk = BLK_LEN / sizeof(T);  // 8 for float32
        this->N_aligned = ((N + elems_per_blk - 1) / elems_per_blk) * elems_per_blk;
        if (this->N_aligned < elems_per_blk) {
            this->N_aligned = elems_per_blk;  // Minimum 8 floats (32 bytes)
        }

        int32_t core_idx = GetBlockIdx();
        int32_t num_cores = tiling.used_core_num;

        this->batch_per_core = (tiling.batch_size + num_cores - 1) / num_cores;
        this->batch_start = core_idx * this->batch_per_core;
        if (this->batch_start + this->batch_per_core > tiling.batch_size) {
            this->batch_count = tiling.batch_size - this->batch_start;
        } else {
            this->batch_count = this->batch_per_core;
        }

        if (this->batch_count <= 0) {
            return;
        }

        int32_t offset = this->batch_start * this->N;
        int32_t total_size = this->batch_count * this->N;

        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x_gm) + offset, total_size);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y_gm) + offset, total_size);
        outGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out_gm) + offset, total_size);

        // Allocate aligned buffers
        int32_t buf_bytes = this->N_aligned * sizeof(T);
        pipe.InitBuffer(xBuf, buf_bytes);
        pipe.InitBuffer(yBuf, buf_bytes);
        pipe.InitBuffer(outBuf, buf_bytes);
        pipe.InitBuffer(tmpBuf, buf_bytes);
    }

    /**
     * @brief Main processing function
     */
    __aicore__ inline void Process() {
        if (this->batch_count <= 0) {
            return;
        }

        for (int32_t b = 0; b < this->batch_count; ++b) {
            switch (this->test_mode) {
                case 0:
                    TestScalarAdd(b);
                    break;
                case 1:
                    TestMaskedVectorAdd(b);
                    break;
                case 2:
                    TestPaddedVectorAdd(b);
                    break;
                case 3:
                    TestScalarReduceSum(b);
                    break;
                case 4:
                    TestMaskedReduceSum(b);
                    break;
                case 5:
                    TestRowNormScalar(b);
                    break;
                case 6:
                    TestRowNormMasked(b);
                    break;
                default:
                    TestScalarAdd(b);
            }
        }
    }

private:
    // =========================================================================
    // Test Mode 0: Scalar Add (Baseline)
    // =========================================================================
    __aicore__ inline void TestScalarAdd(int32_t batch_idx) {
        int32_t gm_offset = batch_idx * this->N;

        // Load using GetValue
        LocalTensor<T> x = xBuf.Get<T>();
        LocalTensor<T> y = yBuf.Get<T>();
        LocalTensor<T> out = outBuf.Get<T>();

        for (int32_t i = 0; i < this->N; ++i) {
            x.SetValue(i, xGm.GetValue(gm_offset + i));
            y.SetValue(i, yGm.GetValue(gm_offset + i));
        }
        PipeBarrier<PIPE_V>();

        // Compute: out = x + y
        for (int32_t i = 0; i < this->N; ++i) {
            T sum = x.GetValue(i) + y.GetValue(i);
            out.SetValue(i, sum);
        }
        PipeBarrier<PIPE_V>();

        // Store using DataCopyPad
        DataCopyExtParams copyParams = {1, static_cast<uint32_t>(this->N * sizeof(T)), 0, 0, 0};
        DataCopyPad(outGm[gm_offset], out, copyParams);
    }

    // =========================================================================
    // Test Mode 1: Masked Vector Add
    // Using SetMaskCount + SetVectorMask to control element count
    // =========================================================================
    __aicore__ inline void TestMaskedVectorAdd(int32_t batch_idx) {
        int32_t gm_offset = batch_idx * this->N;

        LocalTensor<T> x = xBuf.Get<T>();
        LocalTensor<T> y = yBuf.Get<T>();
        LocalTensor<T> out = outBuf.Get<T>();

        // Clear buffers first (including padding area)
        for (int32_t i = 0; i < this->N_aligned; ++i) {
            x.SetValue(i, static_cast<T>(0.0));
            y.SetValue(i, static_cast<T>(0.0));
            out.SetValue(i, static_cast<T>(0.0));
        }

        // Load data using GetValue (safe for any size)
        for (int32_t i = 0; i < this->N; ++i) {
            x.SetValue(i, xGm.GetValue(gm_offset + i));
            y.SetValue(i, yGm.GetValue(gm_offset + i));
        }
        PipeBarrier<PIPE_V>();

        // Use masked vector add
        // SetMaskCount enables counter mode
        SetMaskCount();

        // SetVectorMask controls how many elements participate
        // For N=4 float32, we want only first 4 elements
        SetVectorMask<T>(0, static_cast<uint32_t>(this->N));

        // Vector add with mask placeholder
        Add<T, false>(out, x, y, MASK_PLACEHOLDER, 1, {1, 1, 1, 1, 1, 1});
        PipeBarrier<PIPE_V>();

        // Reset mask to normal mode
        ResetMask();
        SetMaskNorm();

        // Store result
        DataCopyExtParams copyParams = {1, static_cast<uint32_t>(this->N * sizeof(T)), 0, 0, 0};
        DataCopyPad(outGm[gm_offset], out, copyParams);
    }

    // =========================================================================
    // Test Mode 2: Padded Vector Add
    // Load with padding, compute on aligned data, store relevant portion
    // =========================================================================
    __aicore__ inline void TestPaddedVectorAdd(int32_t batch_idx) {
        int32_t gm_offset = batch_idx * this->N;

        LocalTensor<T> x = xBuf.Get<T>();
        LocalTensor<T> y = yBuf.Get<T>();
        LocalTensor<T> out = outBuf.Get<T>();

        // Clear entire aligned buffers
        for (int32_t i = 0; i < this->N_aligned; ++i) {
            x.SetValue(i, static_cast<T>(0.0));
            y.SetValue(i, static_cast<T>(0.0));
            out.SetValue(i, static_cast<T>(0.0));
        }

        // Load data to aligned buffer using DataCopyPad
        DataCopyExtParams loadParams = {1, static_cast<uint32_t>(this->N * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};

        DataCopyPad<T>(x, xGm[gm_offset], loadParams, padParams);
        DataCopyPad<T>(y, yGm[gm_offset], loadParams, padParams);
        PipeBarrier<PIPE_MTE2>();

        // Now we have aligned buffers - use standard vector add on full aligned size
        // This is safe because padding is zero
        Add(out, x, y, this->N_aligned);
        PipeBarrier<PIPE_V>();

        // Store only the relevant portion
        DataCopyExtParams storeParams = {1, static_cast<uint32_t>(this->N * sizeof(T)), 0, 0, 0};
        DataCopyPad(outGm[gm_offset], out, storeParams);
    }

    // =========================================================================
    // Test Mode 3: Scalar Reduce Sum
    // =========================================================================
    __aicore__ inline void TestScalarReduceSum(int32_t batch_idx) {
        int32_t gm_offset = batch_idx * this->N;

        LocalTensor<T> x = xBuf.Get<T>();
        LocalTensor<T> out = outBuf.Get<T>();

        // Load using GetValue
        for (int32_t i = 0; i < this->N; ++i) {
            x.SetValue(i, xGm.GetValue(gm_offset + i));
        }
        PipeBarrier<PIPE_V>();

        // Compute sum
        T sum = static_cast<T>(0.0);
        for (int32_t i = 0; i < this->N; ++i) {
            sum += x.GetValue(i);
        }

        // Store result in out[0] using DataCopyPad (SetValue has bugs for GM writes)
        out.SetValue(0, sum);
        PipeBarrier<PIPE_V>();

        // Use DataCopyPad to write single value to GM
        DataCopyExtParams copyParams = {1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        DataCopyPad(outGm[gm_offset], out, copyParams);
    }

    // =========================================================================
    // Test Mode 4: Masked Reduce Sum
    // Using WholeReduceSum with mask
    // =========================================================================
    __aicore__ inline void TestMaskedReduceSum(int32_t batch_idx) {
        int32_t gm_offset = batch_idx * this->N;

        LocalTensor<T> x = xBuf.Get<T>();
        LocalTensor<T> out = outBuf.Get<T>();

        // Clear buffers
        for (int32_t i = 0; i < this->N_aligned; ++i) {
            x.SetValue(i, static_cast<T>(0.0));
            out.SetValue(i, static_cast<T>(0.0));
        }

        // Load data
        for (int32_t i = 0; i < this->N; ++i) {
            x.SetValue(i, xGm.GetValue(gm_offset + i));
        }
        PipeBarrier<PIPE_V>();

        // Use masked reduce
        SetMaskCount();
        SetVectorMask<T>(0, static_cast<uint32_t>(this->N));

        // WholeReduceSum with mask
        constexpr uint32_t DEFAULT_BLK_STRIDE = 1;
        constexpr uint32_t DEFAULT_REP_STRIDE = 8;

        WholeReduceSum<T, false>(out, x, MASK_PLACEHOLDER, 1,
            DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REP_STRIDE);
        PipeBarrier<PIPE_V>();

        ResetMask();
        SetMaskNorm();

        // Store result using DataCopyPad (SetValue has bugs for GM writes)
        // Result is already in out[0] from WholeReduceSum
        DataCopyExtParams copyParams = {1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        DataCopyPad(outGm[gm_offset], out, copyParams);
    }

    // =========================================================================
    // Test Mode 5: Scalar Row Normalization
    // out[i] = x[i] / sum(x)
    // =========================================================================
    __aicore__ inline void TestRowNormScalar(int32_t batch_idx) {
        int32_t gm_offset = batch_idx * this->N;

        LocalTensor<T> x = xBuf.Get<T>();
        LocalTensor<T> out = outBuf.Get<T>();

        // Load
        for (int32_t i = 0; i < this->N; ++i) {
            x.SetValue(i, xGm.GetValue(gm_offset + i));
        }
        PipeBarrier<PIPE_V>();

        // Compute sum
        T sum = static_cast<T>(0.0);
        for (int32_t i = 0; i < this->N; ++i) {
            sum += x.GetValue(i);
        }

        // Normalize
        T inv = (sum > static_cast<T>(1e-8)) ? static_cast<T>(1.0) / sum : static_cast<T>(0.0);
        for (int32_t i = 0; i < this->N; ++i) {
            out.SetValue(i, x.GetValue(i) * inv);
        }
        PipeBarrier<PIPE_V>();

        // Store
        DataCopyExtParams copyParams = {1, static_cast<uint32_t>(this->N * sizeof(T)), 0, 0, 0};
        DataCopyPad(outGm[gm_offset], out, copyParams);
    }

    // =========================================================================
    // Test Mode 6: Masked Row Normalization
    // Combines masked reduce and masked multiply
    // =========================================================================
    __aicore__ inline void TestRowNormMasked(int32_t batch_idx) {
        int32_t gm_offset = batch_idx * this->N;

        LocalTensor<T> x = xBuf.Get<T>();
        LocalTensor<T> out = outBuf.Get<T>();
        LocalTensor<T> tmp = tmpBuf.Get<T>();

        // Clear buffers
        for (int32_t i = 0; i < this->N_aligned; ++i) {
            x.SetValue(i, static_cast<T>(0.0));
            out.SetValue(i, static_cast<T>(0.0));
            tmp.SetValue(i, static_cast<T>(0.0));
        }

        // Load data
        for (int32_t i = 0; i < this->N; ++i) {
            x.SetValue(i, xGm.GetValue(gm_offset + i));
        }
        PipeBarrier<PIPE_V>();

        // Step 1: Compute sum using masked reduce
        SetMaskCount();
        SetVectorMask<T>(0, static_cast<uint32_t>(this->N));

        constexpr uint32_t DEFAULT_BLK_STRIDE = 1;
        constexpr uint32_t DEFAULT_REP_STRIDE = 8;

        WholeReduceSum<T, false>(tmp, x, MASK_PLACEHOLDER, 1,
            DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REP_STRIDE);
        PipeBarrier<PIPE_V>();

        ResetMask();
        SetMaskNorm();

        // Get sum value
        T sum = tmp.GetValue(0);
        T inv = (sum > static_cast<T>(1e-8)) ? static_cast<T>(1.0) / sum : static_cast<T>(0.0);

        // Step 2: Apply normalization using masked Muls
        SetMaskCount();
        SetVectorMask<T>(0, static_cast<uint32_t>(this->N));

        Muls<T, false>(out, x, inv, MASK_PLACEHOLDER, 1, {1, 1, 8, 8});
        PipeBarrier<PIPE_V>();

        ResetMask();
        SetMaskNorm();

        // Store result
        DataCopyExtParams copyParams = {1, static_cast<uint32_t>(this->N * sizeof(T)), 0, 0, 0};
        DataCopyPad(outGm[gm_offset], out, copyParams);
    }

private:
    TPipe pipe;
    TBuf<QuePosition::VECCALC> xBuf, yBuf, outBuf, tmpBuf;
    GlobalTensor<T> xGm, yGm, outGm;

    int32_t N;
    int32_t N_aligned;
    int32_t batch_per_core;
    int32_t batch_start;
    int32_t batch_count;
    int32_t test_mode;
};

// =============================================================================
// Kernel Entry Point
// =============================================================================

extern "C" __global__ __aicore__ void test_vectorize(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR out,
    GM_ADDR tiling)
{
    // Copy tiling data from GM
    VectorizeTestTiling tiling_data;
    auto tiling_ptr = reinterpret_cast<__gm__ uint32_t*>(tiling);
    auto data_ptr = reinterpret_cast<uint32_t*>(&tiling_data);
    for (uint32_t i = 0; i < sizeof(VectorizeTestTiling) / sizeof(uint32_t); ++i) {
        data_ptr[i] = tiling_ptr[i];
    }

    VectorizeTestKernel<float> kernel;
    kernel.Init(x, y, out, tiling_data);
    kernel.Process();
}
