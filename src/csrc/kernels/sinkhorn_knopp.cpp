/**
 * @file sinkhorn_knopp.cpp
 * @brief Sinkhorn-Knopp matrix normalization kernel for Ascend
 *
 * Copyright (C) 2025. All rights reserved.
 *
 * Implements doubly stochastic matrix normalization via iterative row and column
 * normalization (Sinkhorn-Knopp algorithm). This is the core operation in mHC layers.
 *
 * Algorithm:
 * 1. Row normalization: M[i,:] = M[i,:] / sum(M[i,:])
 * 2. Column normalization: M[:,j] = M[:,j] / sum(M[:,j])
 * 3. Repeat for num_iters iterations
 *
 * CUDA Reference: src/csrc/kernels/sinkhorn_knopp.cuh
 */

#include "kernel_operator.h"
#include "../include/mhc_types.h"
#include "../include/utils.h"

using namespace mhc_ascend;
using namespace AscendC;

// Alias for tiling structure used by this kernel
using SinkhornKnoppTilingData = SinkhornTiling;

// =============================================================================
// Sinkhorn-Knopp Kernel Class
// =============================================================================

template<typename T = float>
class SinkhornKnoppKernel {
public:
    __aicore__ inline SinkhornKnoppKernel() {}

    /**
     * @brief Initialize kernel with tiling data
     * @param inp_gm Input matrix in global memory [batch, M, N]
     * @param out_gm Output matrix in global memory [batch, M, N]
     * @param tiling Tiling configuration
     */
    __aicore__ inline void Init(GM_ADDR inp_gm, GM_ADDR out_gm, SinkhornKnoppTilingData& tiling) {
        this->tiling.batch_size = tiling.batch_size;
        this->tiling.M = tiling.M;
        this->tiling.N = tiling.N;
        this->tiling.num_iters = tiling.num_iters;
        this->tiling.eps = tiling.eps;
        this->tiling.used_core_num = tiling.used_core_num;

        // Calculate work for this core
        int32_t core_idx = GetBlockIdx();
        int32_t num_cores = this->tiling.used_core_num;

        // Each core processes a subset of batch
        this->batch_per_core = CeilingDiv(this->tiling.batch_size, num_cores);
        this->batch_start = core_idx * this->batch_per_core;
        this->batch_count = MIN(this->batch_per_core, this->tiling.batch_size - this->batch_start);

        if (this->batch_count <= 0) {
            return;  // This core has no work
        }

        // Matrix size
        this->M = this->tiling.M;
        this->N = this->tiling.N;
        this->matrix_size = M * N;

        // For vectorization: ensure N_aligned is a multiple of 8 (32B for float32)
        // This ensures each row starts at a 32-byte aligned address
        int32_t elems_per_blk = BLK_LEN / static_cast<int32_t>(sizeof(T));  // 8 for float32
        this->N_aligned = ((N + elems_per_blk - 1) / elems_per_blk) * elems_per_blk;
        this->matrix_size_aligned = M * N_aligned;
        this->total_size = this->batch_count * this->matrix_size;

        // Set global buffers
        int32_t offset = this->batch_start * this->matrix_size;
        inpGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(inp_gm) + offset, this->total_size);
        outGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out_gm) + offset, this->total_size);

        // Use a VECIN queue for in-place normalization
        int32_t matrix_bytes = this->matrix_size_aligned * static_cast<int32_t>(sizeof(T));
        pipe.InitBuffer(matrixBuf, ALIGN_UP(matrix_bytes, BLK_LEN));

        // Row/column sum buffers (aligned to BLK_LEN)
        int32_t sum_elems = MAX(M, N_aligned);
        int32_t sum_bytes = ALIGN_UP(sum_elems * static_cast<int32_t>(sizeof(T)), BLK_LEN);
        pipe.InitBuffer(rowSumBuf, sum_bytes);
        pipe.InitBuffer(colSumBuf, sum_bytes);

        // Temporary buffer for vectorized operations
        int32_t tmp_bytes = ALIGN_UP(N_aligned * static_cast<int32_t>(sizeof(T)), BLK_LEN);
        pipe.InitBuffer(tmpBuf, tmp_bytes);
    }

    /**
     * @brief Main processing function
     */
    __aicore__ inline void Process() {
        if (this->batch_count <= 0) {
            return;
        }

        // Process each matrix in the batch
        for (int32_t b = 0; b < this->batch_count; ++b) {
            ProcessSingleMatrix(b);
        }
    }

private:
    /**
     * @brief Process a single matrix through Sinkhorn-Knopp algorithm (STABLE VERSION)
     * @param batch_idx Index within the batch handled by this core
     *
     * This is the original, thoroughly tested implementation.
     * Uses scalar operations for guaranteed correctness.
     */
    __aicore__ inline void ProcessSingleMatrix_v1_Stable(int32_t batch_idx) {
        // Copy input matrix from GM to UB (scalar-safe) for in-place updates
        LocalTensor<T> matrix = matrixBuf.Get<T>();
        for (int32_t i = 0; i < M; ++i) {
            int32_t base = i * this->N_aligned;
            int32_t gm_base = batch_idx * this->matrix_size + i * this->N;
            for (int32_t j = 0; j < N; ++j) {
                matrix.SetValue(base + j, inpGm.GetValue(gm_base + j));
            }
        }

        // Explicitly clear sum buffers once per matrix to avoid stale UB state on first call.
        // NOTE: Use scalar loop instead of ZeroBuffer (Duplicate) to avoid multi-core mask interference
        LocalTensor<T> rowSums = rowSumBuf.Get<T>();
        LocalTensor<T> colSums = colSumBuf.Get<T>();
        for (int32_t i = 0; i < M; ++i) {
            rowSums.SetValue(i, static_cast<T>(0.0));
        }
        for (int32_t j = 0; j < this->N_aligned; ++j) {
            colSums.SetValue(j, static_cast<T>(0.0));
        }
        PipeBarrier<PIPE_V>();

        // Iterative normalization
        for (int32_t iter = 0; iter < tiling.num_iters; ++iter) {
            // Row normalization
            NormalizeRows(matrix, rowSums);

            // Column normalization
            NormalizeColumns(matrix, colSums);
        }

        // Copy result back to GM using DataCopyPad (SetValue has multi-core issues)
        for (int32_t i = 0; i < M; ++i) {
            int32_t base = i * this->N_aligned;
            int32_t gm_base = batch_idx * this->matrix_size + i * this->N;
            AscendC::DataCopyExtParams copyParams = {1, static_cast<uint32_t>(N * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPad(outGm[gm_base], matrix[base], copyParams);
        }
    }

    /**
     * @brief Normalize rows: matrix[i,:] /= sum(matrix[i,:])
     * @param matrix Matrix to normalize (M x N)
     * @param rowSums Buffer for row sums (size M)
     * @param tmpCalc Temporary calculation buffer
     */
    __aicore__ inline void NormalizeRows(
        LocalTensor<T>& matrix,
        LocalTensor<T>& rowSums)
    {
        // Match CUDA flow: compute row sums, then normalize.
        for (int32_t i = 0; i < M; ++i) {
            T sum = static_cast<T>(0.0);
            int32_t base = i * this->N_aligned;
            for (int32_t j = 0; j < N; ++j) {
                sum += matrix.GetValue(base + j);
            }
            rowSums.SetValue(i, sum);
        }
        PipeBarrier<PIPE_V>();

        for (int32_t i = 0; i < M; ++i) {
            T row_sum = rowSums.GetValue(i);
            int32_t base = i * this->N_aligned;
            if (row_sum > tiling.eps) {
                T inv = static_cast<T>(1.0) / row_sum;
                for (int32_t j = 0; j < N; ++j) {
                    T v = matrix.GetValue(base + j);
                    matrix.SetValue(base + j, v * inv);
                }
            }
        }
        PipeBarrier<PIPE_V>();
    }

    /**
     * @brief Normalize columns: matrix[:,j] /= sum(matrix[:,j])
     * @param matrix Matrix to normalize (M x N)
     * @param colSums Buffer for column sums (size N)
     * @param tmpCalc Temporary calculation buffer
     */
    __aicore__ inline void NormalizeColumns(
        LocalTensor<T>& matrix,
        LocalTensor<T>& colSums)
    {
        // Match CUDA flow: compute col sums, then normalize.
        for (int32_t j = 0; j < N; ++j) {
            T sum = static_cast<T>(0.0);
            for (int32_t i = 0; i < M; ++i) {
                int32_t base = i * this->N_aligned;
                sum += matrix.GetValue(base + j);
            }
            colSums.SetValue(j, sum);
        }
        PipeBarrier<PIPE_V>();

        for (int32_t j = 0; j < N; ++j) {
            T col_sum = colSums.GetValue(j);
            if (col_sum > tiling.eps) {
                T inv = static_cast<T>(1.0) / col_sum;
                for (int32_t i = 0; i < M; ++i) {
                    int32_t base = i * this->N_aligned;
                    T v = matrix.GetValue(base + j);
                    matrix.SetValue(base + j, v * inv);
                }
            }
        }
        PipeBarrier<PIPE_V>();
    }

    /**
     * @brief Normalize rows using vectorized operations (VECTORIZED VERSION)
     * Uses SetMaskCount + SetVectorMask + WholeReduceSum/Muls for N < 8
     * @param matrix Matrix to normalize (M x N_aligned)
     * @param rowSums Buffer for row sums (size M)
     * @param tmpVec Temporary buffer for vectorized operations (size N_aligned)
     *
     * IMPORTANT: We copy row data to tmpVec (aligned buffer), perform operations there,
     * and copy back. This avoids alignment issues with matrix[base] row slicing.
     */
    __aicore__ inline void NormalizeRows_Vectorized(
        LocalTensor<T>& matrix,
        LocalTensor<T>& rowSums,
        LocalTensor<T>& tmpVec)
    {
        constexpr uint32_t DEFAULT_BLK_STRIDE = 1;
        constexpr uint32_t DEFAULT_REP_STRIDE = 8;

        // Phase 1: Compute row sums using masked reduce
        for (int32_t i = 0; i < M; ++i) {
            int32_t base = i * this->N_aligned;

            // Copy row to tmpVec (aligned buffer) - avoids slice alignment issues
            for (int32_t j = 0; j < this->N_aligned; ++j) {
                tmpVec.SetValue(j, (j < N) ? matrix.GetValue(base + j) : static_cast<T>(0.0));
            }
            PipeBarrier<PIPE_V>();

            // Use masked reduce sum on tmpVec (guaranteed aligned)
            SetMaskCount();
            SetVectorMask<T>(0, static_cast<uint32_t>(N));
            // Result goes to tmpVec[0] in-place
            WholeReduceSum<T, false>(tmpVec, tmpVec, MASK_PLACEHOLDER, 1,
                DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REP_STRIDE);
            PipeBarrier<PIPE_V>();
            ResetMask();
            SetMaskNorm();

            // Store sum
            rowSums.SetValue(i, tmpVec.GetValue(0));
        }
        PipeBarrier<PIPE_V>();

        // Phase 2: Normalize each row using masked multiply
        for (int32_t i = 0; i < M; ++i) {
            T row_sum = rowSums.GetValue(i);
            if (row_sum > tiling.eps) {
                T inv = static_cast<T>(1.0) / row_sum;
                int32_t base = i * this->N_aligned;

                // Copy row to tmpVec (aligned buffer)
                for (int32_t j = 0; j < this->N_aligned; ++j) {
                    tmpVec.SetValue(j, (j < N) ? matrix.GetValue(base + j) : static_cast<T>(0.0));
                }
                PipeBarrier<PIPE_V>();

                // Use masked multiply on tmpVec
                SetMaskCount();
                SetVectorMask<T>(0, static_cast<uint32_t>(N));
                Muls<T, false>(tmpVec, tmpVec, inv, MASK_PLACEHOLDER, 1, {1, 1, 8, 8});
                PipeBarrier<PIPE_V>();
                ResetMask();
                SetMaskNorm();

                // Copy result back to matrix
                for (int32_t j = 0; j < N; ++j) {
                    matrix.SetValue(base + j, tmpVec.GetValue(j));
                }
            }
        }
        PipeBarrier<PIPE_V>();
    }

    /**
     * @brief Normalize columns using vectorized operations (VECTORIZED VERSION)
     * Column normalization is harder to vectorize due to non-contiguous access
     * This version uses a hybrid approach: scalar sum + masked multiply
     * @param matrix Matrix to normalize (M x N_aligned)
     * @param colSums Buffer for column sums (size N_aligned)
     * @param tmpVec Temporary buffer for vectorized operations (size N_aligned)
     *
     * IMPORTANT: We copy row data to tmpVec (aligned buffer), perform operations there,
     * and copy back. This avoids alignment issues with matrix[base] row slicing.
     */
    __aicore__ inline void NormalizeColumns_Vectorized(
        LocalTensor<T>& matrix,
        LocalTensor<T>& colSums,
        LocalTensor<T>& tmpVec)
    {
        // Phase 1: Compute column sums (scalar - columns are non-contiguous)
        for (int32_t j = 0; j < N; ++j) {
            T sum = static_cast<T>(0.0);
            for (int32_t i = 0; i < M; ++i) {
                int32_t base = i * this->N_aligned;
                sum += matrix.GetValue(base + j);
            }
            // Store 1/sum for vectorized multiply (avoid division in inner loop)
            colSums.SetValue(j, (sum > tiling.eps) ? static_cast<T>(1.0) / sum : static_cast<T>(0.0));
        }
        // Clear padding in colSums
        for (int32_t j = N; j < this->N_aligned; ++j) {
            colSums.SetValue(j, static_cast<T>(1.0));  // Identity for multiplication
        }
        PipeBarrier<PIPE_V>();

        // Phase 2: Apply normalization using masked element-wise multiply
        for (int32_t i = 0; i < M; ++i) {
            int32_t base = i * this->N_aligned;

            // Copy row to tmpVec (aligned buffer)
            for (int32_t j = 0; j < this->N_aligned; ++j) {
                tmpVec.SetValue(j, (j < N) ? matrix.GetValue(base + j) : static_cast<T>(0.0));
            }
            PipeBarrier<PIPE_V>();

            // Use masked multiply: tmpVec[j] = tmpVec[j] * colSums[j]
            SetMaskCount();
            SetVectorMask<T>(0, static_cast<uint32_t>(N));
            Mul<T, false>(tmpVec, tmpVec, colSums, MASK_PLACEHOLDER, 1, {1, 1, 1, 1, 1, 1});
            PipeBarrier<PIPE_V>();
            ResetMask();
            SetMaskNorm();

            // Copy result back to matrix
            for (int32_t j = 0; j < N; ++j) {
                matrix.SetValue(base + j, tmpVec.GetValue(j));
            }
        }
        PipeBarrier<PIPE_V>();
    }

    /**
     * @brief Check convergence by measuring max deviation from doubly stochastic property
     * @param matrix Matrix to check (M x N)
     * @param rowSums Buffer for row sums
     * @param colSums Buffer for column sums
     * @return Maximum error (max deviation from 1.0)
     */
    __aicore__ inline T CheckConvergence(
        LocalTensor<T>& matrix,
        LocalTensor<T>& rowSums,
        LocalTensor<T>& colSums)
    {
        T max_error = static_cast<T>(0.0);

        // Check row sums
        for (int32_t i = 0; i < M; ++i) {
            T sum = static_cast<T>(0.0);
            int32_t base = i * this->N_aligned;
            for (int32_t j = 0; j < N; ++j) {
                sum += matrix.GetValue(base + j);
            }
            T error = (sum > static_cast<T>(1.0)) ? (sum - static_cast<T>(1.0)) : (static_cast<T>(1.0) - sum);
            if (error > max_error) max_error = error;
        }

        // Check column sums
        for (int32_t j = 0; j < N; ++j) {
            T sum = static_cast<T>(0.0);
            for (int32_t i = 0; i < M; ++i) {
                int32_t base = i * this->N_aligned;
                sum += matrix.GetValue(base + j);
            }
            T error = (sum > static_cast<T>(1.0)) ? (sum - static_cast<T>(1.0)) : (static_cast<T>(1.0) - sum);
            if (error > max_error) max_error = error;
        }

        return max_error;
    }

    /**
     * @brief Normalize rows with optimizations (loop unrolling + fused operations)
     * Conservative vectorization - uses scalar ops but with better cache locality
     */
    __aicore__ inline void NormalizeRows_Optimized(
        LocalTensor<T>& matrix,
        LocalTensor<T>& rowSums)
    {
        // FUSED: Compute row sums AND normalize in single pass per row
        // This reduces memory traffic compared to two separate passes
        for (int32_t i = 0; i < M; ++i) {
            int32_t base = i * this->N_aligned;
            T sum = static_cast<T>(0.0);

            // First pass: compute sum with unrolling
            if (N == 4) {
                #pragma unroll 4
                for (int32_t j = 0; j < 4; ++j) {
                    sum += matrix.GetValue(base + j);
                }
            } else if (N == 8) {
                #pragma unroll 8
                for (int32_t j = 0; j < 8; ++j) {
                    sum += matrix.GetValue(base + j);
                }
            } else if (N == 16) {
                #pragma unroll 16
                for (int32_t j = 0; j < 16; ++j) {
                    sum += matrix.GetValue(base + j);
                }
            } else {
                #pragma unroll 4
                for (int32_t j = 0; j < N; ++j) {
                    sum += matrix.GetValue(base + j);
                }
            }

            // Immediately normalize this row (no need to wait for all rows)
            if (sum > tiling.eps) {
                T inv = static_cast<T>(1.0) / sum;

                // Second pass: normalize with unrolling
                if (N == 4) {
                    #pragma unroll 4
                    for (int32_t j = 0; j < 4; ++j) {
                        T v = matrix.GetValue(base + j);
                        matrix.SetValue(base + j, v * inv);
                    }
                } else if (N == 8) {
                    #pragma unroll 8
                    for (int32_t j = 0; j < 8; ++j) {
                        T v = matrix.GetValue(base + j);
                        matrix.SetValue(base + j, v * inv);
                    }
                } else if (N == 16) {
                    #pragma unroll 16
                    for (int32_t j = 0; j < 16; ++j) {
                        T v = matrix.GetValue(base + j);
                        matrix.SetValue(base + j, v * inv);
                    }
                } else {
                    #pragma unroll 4
                    for (int32_t j = 0; j < N; ++j) {
                        T v = matrix.GetValue(base + j);
                        matrix.SetValue(base + j, v * inv);
                    }
                }
            }
            rowSums.SetValue(i, sum);  // Store for potential convergence check
        }
        PipeBarrier<PIPE_V>();
    }

    /**
     * @brief Normalize columns with optimizations (loop unrolling + pre-computed reciprocals)
     */
    __aicore__ inline void NormalizeColumns_Optimized(
        LocalTensor<T>& matrix,
        LocalTensor<T>& colSums)
    {
        // Compute column sums with unrolling
        for (int32_t j = 0; j < N; ++j) {
            T sum = static_cast<T>(0.0);

            // Unroll based on M
            if (M == 4) {
                #pragma unroll 4
                for (int32_t i = 0; i < 4; ++i) {
                    int32_t base = i * this->N_aligned;
                    sum += matrix.GetValue(base + j);
                }
            } else if (M == 8) {
                #pragma unroll 8
                for (int32_t i = 0; i < 8; ++i) {
                    int32_t base = i * this->N_aligned;
                    sum += matrix.GetValue(base + j);
                }
            } else if (M == 16) {
                #pragma unroll 16
                for (int32_t i = 0; i < 16; ++i) {
                    int32_t base = i * this->N_aligned;
                    sum += matrix.GetValue(base + j);
                }
            } else {
                #pragma unroll 4
                for (int32_t i = 0; i < M; ++i) {
                    int32_t base = i * this->N_aligned;
                    sum += matrix.GetValue(base + j);
                }
            }

            // Pre-compute and store reciprocal (optimization: avoid division in inner loop)
            if (sum > tiling.eps) {
                colSums.SetValue(j, static_cast<T>(1.0) / sum);
            } else {
                colSums.SetValue(j, static_cast<T>(0.0));
            }
        }
        PipeBarrier<PIPE_V>();

        // Apply column normalization using pre-computed reciprocals
        for (int32_t i = 0; i < M; ++i) {
            int32_t base = i * this->N_aligned;

            // Unroll based on N
            if (N == 4) {
                #pragma unroll 4
                for (int32_t j = 0; j < 4; ++j) {
                    T inv = colSums.GetValue(j);
                    if (inv > static_cast<T>(0.0)) {
                        T v = matrix.GetValue(base + j);
                        matrix.SetValue(base + j, v * inv);
                    }
                }
            } else if (N == 8) {
                #pragma unroll 8
                for (int32_t j = 0; j < 8; ++j) {
                    T inv = colSums.GetValue(j);
                    if (inv > static_cast<T>(0.0)) {
                        T v = matrix.GetValue(base + j);
                        matrix.SetValue(base + j, v * inv);
                    }
                }
            } else if (N == 16) {
                #pragma unroll 16
                for (int32_t j = 0; j < 16; ++j) {
                    T inv = colSums.GetValue(j);
                    if (inv > static_cast<T>(0.0)) {
                        T v = matrix.GetValue(base + j);
                        matrix.SetValue(base + j, v * inv);
                    }
                }
            } else {
                #pragma unroll 4
                for (int32_t j = 0; j < N; ++j) {
                    T inv = colSums.GetValue(j);
                    if (inv > static_cast<T>(0.0)) {
                        T v = matrix.GetValue(base + j);
                        matrix.SetValue(base + j, v * inv);
                    }
                }
            }
        }
        PipeBarrier<PIPE_V>();
    }

    /**
     * @brief Process a single matrix with optimizations (OPTIMIZED VERSION)
     * @param batch_idx Index within the batch handled by this core
     *
     * Optimizations:
     * 1. Early stopping - check convergence every 5 iterations
     * 2. Vectorization - use VEC instructions for N >= 8
     * 3. Loop unrolling - for small fixed sizes
     * 4. Fused operations - combine sum and normalize
     */
    __aicore__ inline void ProcessSingleMatrix_v2_Optimized(int32_t batch_idx) {
        // Copy input matrix from GM to UB
        LocalTensor<T> matrix = matrixBuf.Get<T>();
        for (int32_t i = 0; i < M; ++i) {
            int32_t base = i * this->N_aligned;
            int32_t gm_base = batch_idx * this->matrix_size + i * this->N;
            for (int32_t j = 0; j < N; ++j) {
                matrix.SetValue(base + j, inpGm.GetValue(gm_base + j));
            }
        }

        // Clear sum buffers using scalar loop to avoid multi-core mask interference
        // NOTE: ZeroBuffer uses Duplicate which has multi-core issues for non-aligned sizes
        // See v1_Stable comments and knowledge/ascendc_vectorization_vs_multicore_conflict.md
        LocalTensor<T> rowSums = rowSumBuf.Get<T>();
        LocalTensor<T> colSums = colSumBuf.Get<T>();
        for (int32_t i = 0; i < M; ++i) {
            rowSums.SetValue(i, static_cast<T>(0.0));
        }
        for (int32_t j = 0; j < this->N_aligned; ++j) {
            colSums.SetValue(j, static_cast<T>(0.0));
        }
        PipeBarrier<PIPE_V>();

        // Iterative normalization (fixed iterations to match backward pass)
        // NOTE: Early stopping was removed to ensure forward and backward use same iteration count
        // This is critical for correct gradient computation in backward pass
        for (int32_t iter = 0; iter < tiling.num_iters; ++iter) {
            // Use optimized version with loop unrolling and fused operations
            NormalizeRows_Optimized(matrix, rowSums);
            NormalizeColumns_Optimized(matrix, colSums);
        }

        // Copy result back to GM using DataCopyPad (SetValue has multi-core issues)
        // See CLAUDE.md: "写回必须用 DataCopyPad"
        for (int32_t i = 0; i < M; ++i) {
            int32_t base = i * this->N_aligned;
            int32_t gm_base = batch_idx * this->matrix_size + i * this->N;
            // Use DataCopyPad for reliable GM write in multi-core mode
            AscendC::DataCopyExtParams copyParams = {1, static_cast<uint32_t>(N * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPad(outGm[gm_base], matrix[base], copyParams);
        }
    }

    /**
     * @brief Process a single matrix using vectorized operations (VECTORIZED VERSION)
     * @param batch_idx Index within the batch handled by this core
     *
     * Uses masked vector operations (WholeReduceSum, Muls, Mul) which work for any N.
     */
    __aicore__ inline void ProcessSingleMatrix_v3_Vectorized(int32_t batch_idx) {
        // Get buffers
        LocalTensor<T> matrix = matrixBuf.Get<T>();
        LocalTensor<T> rowSums = rowSumBuf.Get<T>();
        LocalTensor<T> colSums = colSumBuf.Get<T>();
        LocalTensor<T> tmpVec = tmpBuf.Get<T>();

        // Copy input matrix from GM to UB (with N_aligned padding)
        for (int32_t i = 0; i < M; ++i) {
            int32_t base = i * this->N_aligned;
            int32_t gm_base = batch_idx * this->matrix_size + i * this->N;
            // Copy actual data
            for (int32_t j = 0; j < N; ++j) {
                matrix.SetValue(base + j, inpGm.GetValue(gm_base + j));
            }
            // Clear padding region
            for (int32_t j = N; j < this->N_aligned; ++j) {
                matrix.SetValue(base + j, static_cast<T>(0.0));
            }
        }

        // Clear sum buffers
        ZeroBuffer(rowSums, M);
        ZeroBuffer(colSums, this->N_aligned);
        ZeroBuffer(tmpVec, this->N_aligned);
        PipeBarrier<PIPE_V>();

        // Iterative normalization using vectorized operations
        for (int32_t iter = 0; iter < tiling.num_iters; ++iter) {
            NormalizeRows_Vectorized(matrix, rowSums, tmpVec);
            NormalizeColumns_Vectorized(matrix, colSums, tmpVec);
        }

        // Copy result back to GM using DataCopyPad (SetValue has multi-core issues)
        for (int32_t i = 0; i < M; ++i) {
            int32_t base = i * this->N_aligned;
            int32_t gm_base = batch_idx * this->matrix_size + i * this->N;
            AscendC::DataCopyExtParams copyParams = {1, static_cast<uint32_t>(N * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPad(outGm[gm_base], matrix[base], copyParams);
        }
    }

    /**
     * @brief Dispatch to appropriate implementation based on configuration
     * @param batch_idx Index within the batch handled by this core
     *
     * Vectorized mask operations (SetMaskCount/SetVectorMask) have multi-core
     * interference issues for non-4-aligned N values. But N % 4 == 0 works fine.
     *
     * Strategy (controlled by MHC_SINKHORN_FWD_IMPL environment variable):
     * - auto: N % 4 == 0 or single-core → vectorized; otherwise → scalar
     * - scalar: Force v2_Optimized (loop unrolling, no Mask ops)
     * - vectorized: Force v3_Vectorized (Mask API), with safety fallback
     */
    __aicore__ inline void ProcessSingleMatrix(int32_t batch_idx) {
        int32_t impl_mode = this->tiling.fwd_impl_mode;

        if (impl_mode == 1) {
            // SCALAR: Force scalar (v2_Optimized) implementation
            ProcessSingleMatrix_v2_Optimized(batch_idx);
            return;
        } else if (impl_mode == 2) {
            // VECTORIZED: Force vectorized, but with safety fallback
            bool vectorized_safe = (this->N % 4 == 0) || (this->tiling.used_core_num == 1);
            if (vectorized_safe) {
                ProcessSingleMatrix_v3_Vectorized(batch_idx);
            } else {
                // Safety fallback: multi-core + N%4!=0 would cause interference
                ProcessSingleMatrix_v2_Optimized(batch_idx);
            }
            return;
        }

        // AUTO mode: automatic selection based on N alignment and core count
        bool can_use_vectorized = (this->N % 4 == 0) || (this->tiling.used_core_num == 1);
        if (can_use_vectorized) {
            ProcessSingleMatrix_v3_Vectorized(batch_idx);
        } else {
            // Multi-core with N % 4 != 0: use scalar version
            ProcessSingleMatrix_v2_Optimized(batch_idx);
        }
    }

private:
    TPipe pipe;
    TBuf<QuePosition::VECCALC> matrixBuf;
    TBuf<QuePosition::VECCALC> rowSumBuf, colSumBuf;
    TBuf<QuePosition::VECCALC> tmpBuf;  // For vectorized operations

    GlobalTensor<T> inpGm;
    GlobalTensor<T> outGm;

    SinkhornTiling tiling;

    int32_t M, N;
    int32_t N_aligned;
    int32_t matrix_size;
    int32_t matrix_size_aligned;
    int32_t batch_per_core;
    int32_t batch_start;
    int32_t batch_count;
    int32_t total_size;
};

// =============================================================================
// Sinkhorn-Knopp Forward Kernel Entry Point
// =============================================================================

/**
 * @brief Sinkhorn-Knopp forward kernel entry
 * @param inp Input matrices [batch, M, N]
 * @param out Output normalized matrices [batch, M, N]
 * @param tiling Tiling configuration data
 */
extern "C" __global__ __aicore__ void sinkhorn_knopp_forward(
    GM_ADDR inp,
    GM_ADDR out,
    GM_ADDR tiling)
{
    SinkhornTiling tiling_data;
    InitTilingData(tiling, &tiling_data);

    SinkhornKnoppKernel<float> kernel;
    kernel.Init(inp, out, tiling_data);
    kernel.Process();
}

// =============================================================================
// Sinkhorn-Knopp Backward Kernel Class
// =============================================================================

template<typename T = float>
class SinkhornKnoppBackwardKernel {
public:
    __aicore__ inline SinkhornKnoppBackwardKernel() {}

    /**
     * @brief Initialize backward kernel
     * @param grad_out_gm Gradient w.r.t. output [batch, M, N]
     * @param inp_gm Original input [batch, M, N]
     * @param out_gm Forward output [batch, M, N]
     * @param grad_inp_gm Gradient w.r.t. input [batch, M, N] (output)
     * @param tiling Tiling configuration
     */
    __aicore__ inline void Init(
        GM_ADDR grad_out_gm,
        GM_ADDR inp_gm,
        GM_ADDR out_gm,
        GM_ADDR grad_inp_gm,
        const SinkhornTiling& tiling)
    {
        this->tiling = tiling;

        int32_t core_idx = GetBlockIdx();
        int32_t num_cores = tiling.used_core_num;

        this->batch_per_core = CeilingDiv(tiling.batch_size, num_cores);
        this->batch_start = core_idx * this->batch_per_core;
        this->batch_count = MIN(this->batch_per_core, tiling.batch_size - this->batch_start);

        if (this->batch_count <= 0) {
            return;
        }

        this->M = tiling.M;
        this->N = tiling.N;
        this->matrix_size = M * N;

        // Calculate N_aligned to ensure each row starts at 32-byte aligned address
        // For VEC instructions, need at least 32-byte (8 floats) alignment per row
        // IMPORTANT: Must round UP to nearest multiple of 8, not just ensure minimum
        int32_t elems_per_blk = (32 + sizeof(T) - 1) / sizeof(T);  // 8 for float
        this->N_aligned = ((N + elems_per_blk - 1) / elems_per_blk) * elems_per_blk;
        this->matrix_size_aligned = M * N_aligned;

        this->total_size = this->batch_count * this->matrix_size;

        int32_t offset = this->batch_start * this->matrix_size;
        gradOutGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(grad_out_gm) + offset, this->total_size);
        inpGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(inp_gm) + offset, this->total_size);
        outGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out_gm) + offset, this->total_size);
        gradInpGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(grad_inp_gm) + offset, this->total_size);

        // Initialize buffers with N_aligned for proper row alignment
        int32_t matrix_bytes_aligned = ALIGN_UP(this->matrix_size_aligned * static_cast<int32_t>(sizeof(T)), BLK_LEN);
        pipe.InitBuffer(gradQueue, NUM_BUFFERS, matrix_bytes_aligned);
        pipe.InitBuffer(matrixQueue, NUM_BUFFERS, matrix_bytes_aligned);
        pipe.InitBuffer(outMatrixQueue, NUM_BUFFERS, matrix_bytes_aligned);
        pipe.InitBuffer(tmpBuf1, matrix_bytes_aligned);
        pipe.InitBuffer(tileFwdBuf, matrix_bytes_aligned);
        // Need 4 segments: rowBuf, colBuf, tmpVec, tmpVec2 (for vectorized backward)
        int32_t sum_bytes = ALIGN_UP(4 * MAX(M, N_aligned) * static_cast<int32_t>(sizeof(T)), BLK_LEN);
        pipe.InitBuffer(tmpBuf2, sum_bytes);

        // Phase 2 optimization: History buffers for O(T) complexity
        // Store inv values and A matrices for each iteration to avoid O(T²) recomputation
        int32_t max_iters = 32;  // Maximum iterations we support (typically 20)
        int32_t row_inv_history_bytes = ALIGN_UP(max_iters * M * static_cast<int32_t>(sizeof(T)), BLK_LEN);
        int32_t col_inv_history_bytes = ALIGN_UP(max_iters * N_aligned * static_cast<int32_t>(sizeof(T)), BLK_LEN);
        int32_t a_history_bytes = ALIGN_UP(max_iters * this->matrix_size_aligned * static_cast<int32_t>(sizeof(T)), BLK_LEN);
        pipe.InitBuffer(rowInvHistoryBuf, row_inv_history_bytes);
        pipe.InitBuffer(colInvHistoryBuf, col_inv_history_bytes);
        pipe.InitBuffer(aHistoryBuf, a_history_bytes);
    }

    /**
     * @brief Main backward processing
     *
     * For Sinkhorn-Knopp backward, we need to recompute the forward pass
     * and then apply the chain rule through each normalization step.
     */
    __aicore__ inline void Process() {
        if (this->batch_count <= 0) {
            return;
        }

        for (int32_t b = 0; b < this->batch_count; ++b) {
            ProcessSingleBackward(b);
        }
    }

private:
    /**
     * @brief Process backward for a single matrix (OLD IMPLEMENTATION - alignment issues with small N)
     *
     * Gradient computation through Sinkhorn-Knopp:
     * We recompute forward and apply gradients backwards through each iteration.
     *
     * PROBLEM: Uses VEC instructions (Muls, Mul, Add) on potentially unaligned rows.
     * When N=4, each row is only 16 bytes, causing alignment errors.
     */
    __aicore__ inline void ProcessSingleBackward_OLD(int32_t batch_idx) {
        // Load gradient output
        LocalTensor<T> gradOut = gradQueue.AllocTensor<T>();
        // FIX: Use DataCopyPad for small matrices to avoid alignment issues
        AscendC::DataCopyExtParams copyParams1 = {
            1, static_cast<uint32_t>(this->matrix_size * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams1;
        AscendC::DataCopyPad<T>(
            gradOut,
            gradOutGm[batch_idx * this->matrix_size],
            copyParams1,
            padParams1);
        gradQueue.EnQue(gradOut);

        // Load original input for re-computation
        LocalTensor<T> matrix = matrixQueue.AllocTensor<T>();
        // FIX: Use DataCopyPad for small matrices to avoid alignment issues
        AscendC::DataCopyExtParams copyParams2 = {
            1, static_cast<uint32_t>(this->matrix_size * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams2;
        AscendC::DataCopyPad<T>(
            matrix,
            inpGm[batch_idx * this->matrix_size],
            copyParams2,
            padParams2);
        matrixQueue.EnQue(matrix);

        gradOut = gradQueue.DeQue<T>();
        matrix = matrixQueue.DeQue<T>();

        LocalTensor<T> gradMatrix = tmpBuf1.Get<T>();
        LocalTensor<T> tileFwd = tileFwdBuf.Get<T>();
        LocalTensor<T> sumBuf = tmpBuf2.Get<T>();
        int32_t max_dim = MAX(M, N);
        LocalTensor<T> rowBuf = sumBuf;
        LocalTensor<T> colBuf = sumBuf[max_dim];
        LocalTensor<T> tmpVec = sumBuf[2 * max_dim];

        // Copy grad_out to gradMatrix as starting point
        DataCopy(gradMatrix, gradOut, this->matrix_size);

        // Backward through each iteration (reverse order), matching CUDA reference
        for (int32_t iter = tiling.num_iters - 1; iter >= 0; --iter) {
            // Recompute forward up to current iteration
            DataCopy(tileFwd, matrix, this->matrix_size);

            for (int32_t fwd_iter = 0; fwd_iter < iter; ++fwd_iter) {
                // Row normalization
                for (int32_t r = 0; r < M; ++r) {
                    LocalTensor<T> row = tileFwd[r * N];
                    T sum = static_cast<T>(0.0);
                    for (int32_t c = 0; c < N; ++c) {
                        sum += row.GetValue(c);
                    }
                    T inv = (sum > tiling.eps) ? static_cast<T>(1.0) / sum : static_cast<T>(0.0);
                    rowBuf.SetValue(r, inv);
                }
                for (int32_t r = 0; r < M; ++r) {
                    LocalTensor<T> row = tileFwd[r * N];
                    T inv = rowBuf.GetValue(r);
                    Muls(row, row, inv, N);
                }
                PipeBarrier<PIPE_V>();

                // Column normalization
                ZeroBuffer(colBuf, N);
                for (int32_t r = 0; r < M; ++r) {
                    LocalTensor<T> row = tileFwd[r * N];
                    Add(colBuf, colBuf, row, N);
                }
                for (int32_t c = 0; c < N; ++c) {
                    T sum = colBuf.GetValue(c);
                    T inv = (sum > tiling.eps) ? static_cast<T>(1.0) / sum : static_cast<T>(0.0);
                    colBuf.SetValue(c, inv);
                }
                for (int32_t r = 0; r < M; ++r) {
                    LocalTensor<T> row = tileFwd[r * N];
                    Mul(row, row, colBuf, N);
                }
                PipeBarrier<PIPE_V>();
            }

            // Row normalization for current iteration
            for (int32_t r = 0; r < M; ++r) {
                LocalTensor<T> row = tileFwd[r * N];
                T sum = static_cast<T>(0.0);
                for (int32_t c = 0; c < N; ++c) {
                    sum += row.GetValue(c);
                }
                T inv = (sum > tiling.eps) ? static_cast<T>(1.0) / sum : static_cast<T>(0.0);
                rowBuf.SetValue(r, inv);
            }
            for (int32_t r = 0; r < M; ++r) {
                LocalTensor<T> row = tileFwd[r * N];
                T inv = rowBuf.GetValue(r);
                Muls(row, row, inv, N);
            }
            PipeBarrier<PIPE_V>();

            // Column backward: colBuf[c] = sum_r(gradMatrix[r,c] * tileFwd[r,c])
            ZeroBuffer(colBuf, N);
            for (int32_t r = 0; r < M; ++r) {
                LocalTensor<T> gradRow = gradMatrix[r * N];
                LocalTensor<T> tileRow = tileFwd[r * N];
                Mul(tmpVec, gradRow, tileRow, N);
                Add(colBuf, colBuf, tmpVec, N);
            }

            // gradMatrix -= tileFwd * colBuf
            for (int32_t r = 0; r < M; ++r) {
                LocalTensor<T> gradRow = gradMatrix[r * N];
                LocalTensor<T> tileRow = tileFwd[r * N];
                Mul(tmpVec, tileRow, colBuf, N);
                Muls(tmpVec, tmpVec, static_cast<T>(-1.0), N);
                Add(gradRow, gradRow, tmpVec, N);
            }
            PipeBarrier<PIPE_V>();

            // Row backward: rowBuf[r] = sum_c(gradMatrix[r,c] * tileFwd[r,c])
            for (int32_t r = 0; r < M; ++r) {
                LocalTensor<T> gradRow = gradMatrix[r * N];
                LocalTensor<T> tileRow = tileFwd[r * N];
                T sum = static_cast<T>(0.0);
                for (int32_t c = 0; c < N; ++c) {
                    sum += gradRow.GetValue(c) * tileRow.GetValue(c);
                }
                rowBuf.SetValue(r, sum);
            }

            // gradMatrix -= tileFwd * rowBuf
            for (int32_t r = 0; r < M; ++r) {
                LocalTensor<T> gradRow = gradMatrix[r * N];
                LocalTensor<T> tileRow = tileFwd[r * N];
                T sum = rowBuf.GetValue(r);
                Muls(tmpVec, tileRow, static_cast<T>(-1.0) * sum, N);
                Add(gradRow, gradRow, tmpVec, N);
            }
            PipeBarrier<PIPE_V>();
        }

        // Write gradient back to GM
        // FIX: Use DataCopyPad for small matrices to avoid alignment issues
        AscendC::DataCopyExtParams copyParamsOut = {
            1, static_cast<uint32_t>(this->matrix_size * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPad<T>(
            gradInpGm[batch_idx * this->matrix_size],
            gradMatrix,
            copyParamsOut);

        // Free tensors
        gradQueue.FreeTensor(gradOut);
        matrixQueue.FreeTensor(matrix);
    }

    /**
     * @brief Process backward for a single matrix (SCALAR VERSION)
     *
     * Uses scalar operations and N_aligned layout to avoid VEC alignment issues.
     * Similar strategy as forward kernel.
     */
    __aicore__ inline void ProcessSingleBackward_Scalar(int32_t batch_idx) {
        // Get local buffers with N_aligned layout
        LocalTensor<T> gradMatrix = tmpBuf1.Get<T>();
        LocalTensor<T> tileFwd = tileFwdBuf.Get<T>();
        LocalTensor<T> sumBuf = tmpBuf2.Get<T>();

        int32_t max_dim = MAX(M, N_aligned);
        LocalTensor<T> rowBuf = sumBuf;
        LocalTensor<T> colBuf = sumBuf[max_dim];
        LocalTensor<T> tmpVec = sumBuf[2 * max_dim];  // Third segment for intermediate results

        // Load gradient output from GM to UB (scalar-safe with N_aligned layout)
        for (int32_t i = 0; i < M; ++i) {
            int32_t base_aligned = i * this->N_aligned;
            int32_t gm_base = batch_idx * this->matrix_size + i * this->N;
            for (int32_t j = 0; j < N; ++j) {
                T val = gradOutGm.GetValue(gm_base + j);
                gradMatrix.SetValue(base_aligned + j, val);
            }
        }

        // Load original input for re-computation (scalar-safe with N_aligned layout)
        for (int32_t i = 0; i < M; ++i) {
            int32_t base_aligned = i * this->N_aligned;
            int32_t gm_base = batch_idx * this->matrix_size + i * this->N;
            for (int32_t j = 0; j < N; ++j) {
                T val = inpGm.GetValue(gm_base + j);
                tileFwd.SetValue(base_aligned + j, val);
            }
        }
        PipeBarrier<PIPE_V>();

        // Clear sum buffers (FIX: use correct sizes for each buffer)
        ZeroBuffer(rowBuf, M);         // rowBuf size = M (number of rows)
        ZeroBuffer(colBuf, N_aligned); // colBuf size = N_aligned (aligned columns)
        PipeBarrier<PIPE_V>();

        // Clear padding regions in gradMatrix and tileFwd to avoid garbage values
        for (int32_t i = 0; i < M; ++i) {
            int32_t base = i * this->N_aligned;
            for (int32_t j = N; j < this->N_aligned; ++j) {
                gradMatrix.SetValue(base + j, static_cast<T>(0.0));
                tileFwd.SetValue(base + j, static_cast<T>(0.0));
            }
        }
        PipeBarrier<PIPE_V>();

        // ==============================================================
        // Phase 2 Optimization: Two-phase backward for O(T) complexity
        // ==============================================================

        // PHASE 1: Recompute forward ONCE and save all intermediate values
        // This replaces the O(T²) nested recomputation loop

        LocalTensor<T> rowInvHistory = rowInvHistoryBuf.Get<T>();
        LocalTensor<T> colInvHistory = colInvHistoryBuf.Get<T>();
        LocalTensor<T> aHistory = aHistoryBuf.Get<T>();

        // Load original input matrix
        for (int32_t i = 0; i < M; ++i) {
            int32_t base_aligned = i * this->N_aligned;
            int32_t gm_base = batch_idx * this->matrix_size + i * this->N;
            for (int32_t j = 0; j < N; ++j) {
                T val = inpGm.GetValue(gm_base + j);
                tileFwd.SetValue(base_aligned + j, val);
            }
            // Clear padding region
            for (int32_t j = N; j < this->N_aligned; ++j) {
                tileFwd.SetValue(base_aligned + j, static_cast<T>(0.0));
            }
        }
        PipeBarrier<PIPE_V>();

        // Forward pass: save rowInv, colInv, and A for each iteration
        for (int32_t iter = 0; iter < tiling.num_iters; ++iter) {
            // Row normalization
            for (int32_t r = 0; r < M; ++r) {
                int32_t base = r * this->N_aligned;
                T sum = static_cast<T>(0.0);
                for (int32_t c = 0; c < N; ++c) {
                    sum += tileFwd.GetValue(base + c);
                }
                T inv = (sum > tiling.eps) ? static_cast<T>(1.0) / sum : static_cast<T>(0.0);
                // Save rowInv to history
                rowInvHistory.SetValue(iter * M + r, inv);
            }

            // Apply row normalization: X -> A
            for (int32_t r = 0; r < M; ++r) {
                int32_t base = r * this->N_aligned;
                T inv = rowInvHistory.GetValue(iter * M + r);
                for (int32_t c = 0; c < N; ++c) {
                    T v = tileFwd.GetValue(base + c);
                    tileFwd.SetValue(base + c, v * inv);
                }
            }
            PipeBarrier<PIPE_V>();

            // Save A matrix to history (after row norm, before col norm)
            for (int32_t i = 0; i < M; ++i) {
                int32_t base_aligned = i * this->N_aligned;
                int32_t hist_base = iter * this->matrix_size_aligned + base_aligned;
                for (int32_t j = 0; j < N; ++j) {
                    T val = tileFwd.GetValue(base_aligned + j);
                    aHistory.SetValue(hist_base + j, val);
                }
            }

            // Column normalization
            for (int32_t c = 0; c < N; ++c) {
                T sum = static_cast<T>(0.0);
                for (int32_t r = 0; r < M; ++r) {
                    sum += tileFwd.GetValue(r * this->N_aligned + c);
                }
                T inv = (sum > tiling.eps) ? static_cast<T>(1.0) / sum : static_cast<T>(0.0);
                // Save colInv to history
                colInvHistory.SetValue(iter * N_aligned + c, inv);
            }

            // Apply column normalization for next iteration
            for (int32_t r = 0; r < M; ++r) {
                int32_t base = r * this->N_aligned;
                for (int32_t c = 0; c < N; ++c) {
                    T v = tileFwd.GetValue(base + c);
                    T inv = colInvHistory.GetValue(iter * N_aligned + c);
                    tileFwd.SetValue(base + c, v * inv);
                }
            }
            PipeBarrier<PIPE_V>();
        }

        // PHASE 2: Backward through iterations using saved values (O(T))
        for (int32_t iter = tiling.num_iters - 1; iter >= 0; --iter) {
            // Load saved values for this iteration
            // rowInv[r] = rowInvHistory[iter * M + r]
            for (int32_t r = 0; r < M; ++r) {
                T inv = rowInvHistory.GetValue(iter * M + r);
                rowBuf.SetValue(r, inv);
            }

            // colInv[c] = colInvHistory[iter * N_aligned + c]
            for (int32_t c = 0; c < N; ++c) {
                T inv = colInvHistory.GetValue(iter * N_aligned + c);
                colBuf.SetValue(c, inv);
            }
            // Clear padding in colBuf
            for (int32_t c = N; c < N_aligned; ++c) {
                colBuf.SetValue(c, static_cast<T>(0.0));
            }

            // A[i,j] = aHistory[iter * matrix_size_aligned + i * N_aligned + j]
            for (int32_t i = 0; i < M; ++i) {
                int32_t base_aligned = i * this->N_aligned;
                int32_t hist_base = iter * this->matrix_size_aligned + base_aligned;
                for (int32_t j = 0; j < N; ++j) {
                    T val = aHistory.GetValue(hist_base + j);
                    tileFwd.SetValue(base_aligned + j, val);
                }
            }
            PipeBarrier<PIPE_V>();

            // Now proceed with backward computation using loaded values
            // (No recomputation needed - this is the O(T) improvement!)

            // Column backward: dA = (dY - sCol) * colInv
            // Step 1: Compute dot[c] = sum_r(dY[r,c] * A[r,c])
            for (int32_t c = 0; c < N; ++c) {
                T dot = static_cast<T>(0.0);
                for (int32_t r = 0; r < M; ++r) {
                    int32_t base = r * this->N_aligned;
                    T dY = gradMatrix.GetValue(base + c);
                    T A = tileFwd.GetValue(base + c);
                    dot += dY * A;
                }
                tmpVec.SetValue(c, dot);
            }

            // Step 2: Compute sCol[c] = dot[c] * colInv[c]
            for (int32_t c = 0; c < N; ++c) {
                T dot = tmpVec.GetValue(c);
                T colInv = colBuf.GetValue(c);
                tmpVec.SetValue(c, dot * colInv);
            }

            // Step 3: Compute dA[r,c] = (dY[r,c] - sCol[c]) * colInv[c]
            for (int32_t r = 0; r < M; ++r) {
                int32_t base = r * this->N_aligned;
                for (int32_t c = 0; c < N; ++c) {
                    T dY = gradMatrix.GetValue(base + c);
                    T sCol = tmpVec.GetValue(c);
                    T colInv = colBuf.GetValue(c);
                    gradMatrix.SetValue(base + c, (dY - sCol) * colInv);
                }
            }
            PipeBarrier<PIPE_V>();

            // Row backward: dX = (dA - sRow) * rowInv
            // Step 1: Compute dot[r] = sum_c(dA[r,c] * A[r,c])
            for (int32_t r = 0; r < M; ++r) {
                int32_t base = r * this->N_aligned;
                T dot = static_cast<T>(0.0);
                for (int32_t c = 0; c < N; ++c) {
                    T dA = gradMatrix.GetValue(base + c);
                    T A = tileFwd.GetValue(base + c);
                    dot += dA * A;
                }
                tmpVec.SetValue(r, dot);
            }

            // Step 2: Compute sRow[r] = dot[r] * rowInv[r]
            for (int32_t r = 0; r < M; ++r) {
                T dot = tmpVec.GetValue(r);
                T rowInv = rowBuf.GetValue(r);
                tmpVec.SetValue(r, dot * rowInv);
            }

            // Step 3: Compute dX[r,c] = (dA[r,c] - sRow[r]) * rowInv[r]
            for (int32_t r = 0; r < M; ++r) {
                int32_t base = r * this->N_aligned;
                T sRow = tmpVec.GetValue(r);
                T rowInv = rowBuf.GetValue(r);
                for (int32_t c = 0; c < N; ++c) {
                    T dA = gradMatrix.GetValue(base + c);
                    gradMatrix.SetValue(base + c, (dA - sRow) * rowInv);
                }
            }
            PipeBarrier<PIPE_V>();
        }

        // Write gradient back to GM using DataCopyPad (SetValue has bugs for small tensors!)
        // First copy gradMatrix (N_aligned layout) to a contiguous buffer (N layout)
        LocalTensor<T> gradOut = gradQueue.AllocTensor<T>();
        for (int32_t i = 0; i < M; ++i) {
            int32_t base_aligned = i * this->N_aligned;
            int32_t out_base = i * this->N;
            for (int32_t j = 0; j < N; ++j) {
                gradOut.SetValue(out_base + j, gradMatrix.GetValue(base_aligned + j));
            }
        }
        PipeBarrier<PIPE_V>();

        // Use DataCopyPad to write to GM
        AscendC::DataCopyExtParams copyParams = {
            1, static_cast<uint32_t>(this->matrix_size * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPad(gradInpGm[batch_idx * this->matrix_size], gradOut, copyParams);
        PipeBarrier<PIPE_MTE2>();

        gradQueue.FreeTensor(gradOut);
    }

    /**
     * @brief Process backward for a single matrix (VECTORIZED VERSION)
     *
     * Uses masked vector operations for row-wise computations.
     * Column operations remain scalar due to non-contiguous access.
     *
     * IMPORTANT: Vector mask operations have multi-core interference issues
     * for non-4-aligned N values. Use N%4 dispatch like forward kernel.
     */
    __aicore__ inline void ProcessSingleBackward_Vectorized(int32_t batch_idx) {
        constexpr uint32_t DEFAULT_BLK_STRIDE = 1;
        constexpr uint32_t DEFAULT_REP_STRIDE = 8;

        // Get local buffers with N_aligned layout
        LocalTensor<T> gradMatrix = tmpBuf1.Get<T>();
        LocalTensor<T> tileFwd = tileFwdBuf.Get<T>();
        LocalTensor<T> sumBuf = tmpBuf2.Get<T>();

        int32_t max_dim = MAX(M, N_aligned);
        LocalTensor<T> rowBuf = sumBuf;
        LocalTensor<T> colBuf = sumBuf[max_dim];
        LocalTensor<T> tmpVec = sumBuf[2 * max_dim];
        LocalTensor<T> tmpVec2 = sumBuf[3 * max_dim];  // Additional buffer for vectorized ops

        // Load gradient output from GM to UB (with N_aligned padding)
        for (int32_t i = 0; i < M; ++i) {
            int32_t base_aligned = i * this->N_aligned;
            int32_t gm_base = batch_idx * this->matrix_size + i * this->N;
            for (int32_t j = 0; j < N; ++j) {
                gradMatrix.SetValue(base_aligned + j, gradOutGm.GetValue(gm_base + j));
            }
            for (int32_t j = N; j < this->N_aligned; ++j) {
                gradMatrix.SetValue(base_aligned + j, static_cast<T>(0.0));
            }
        }

        // Load original input for re-computation (with N_aligned padding)
        for (int32_t i = 0; i < M; ++i) {
            int32_t base_aligned = i * this->N_aligned;
            int32_t gm_base = batch_idx * this->matrix_size + i * this->N;
            for (int32_t j = 0; j < N; ++j) {
                tileFwd.SetValue(base_aligned + j, inpGm.GetValue(gm_base + j));
            }
            for (int32_t j = N; j < this->N_aligned; ++j) {
                tileFwd.SetValue(base_aligned + j, static_cast<T>(0.0));
            }
        }
        PipeBarrier<PIPE_V>();

        // Clear buffers
        ZeroBuffer(rowBuf, M);
        ZeroBuffer(colBuf, N_aligned);
        ZeroBuffer(tmpVec, N_aligned);
        ZeroBuffer(tmpVec2, N_aligned);
        PipeBarrier<PIPE_V>();

        // ==============================================================
        // Phase 2 Optimization: Two-phase backward for O(T) complexity
        // ==============================================================

        LocalTensor<T> rowInvHistory = rowInvHistoryBuf.Get<T>();
        LocalTensor<T> colInvHistory = colInvHistoryBuf.Get<T>();
        LocalTensor<T> aHistory = aHistoryBuf.Get<T>();

        // PHASE 1: Forward recomputation with vectorized row operations
        // Reload original input
        for (int32_t i = 0; i < M; ++i) {
            int32_t base_aligned = i * this->N_aligned;
            int32_t gm_base = batch_idx * this->matrix_size + i * this->N;
            for (int32_t j = 0; j < N; ++j) {
                tileFwd.SetValue(base_aligned + j, inpGm.GetValue(gm_base + j));
            }
            for (int32_t j = N; j < this->N_aligned; ++j) {
                tileFwd.SetValue(base_aligned + j, static_cast<T>(0.0));
            }
        }
        PipeBarrier<PIPE_V>();

        for (int32_t iter = 0; iter < tiling.num_iters; ++iter) {
            // Row normalization - VECTORIZED
            // Phase 1a: Compute row sums using masked reduce
            for (int32_t r = 0; r < M; ++r) {
                int32_t base = r * this->N_aligned;

                // Copy row to tmpVec (aligned buffer)
                for (int32_t j = 0; j < this->N_aligned; ++j) {
                    tmpVec.SetValue(j, (j < N) ? tileFwd.GetValue(base + j) : static_cast<T>(0.0));
                }
                PipeBarrier<PIPE_V>();

                // Masked reduce sum
                SetMaskCount();
                SetVectorMask<T>(0, static_cast<uint32_t>(N));
                WholeReduceSum<T, false>(tmpVec, tmpVec, MASK_PLACEHOLDER, 1,
                    DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REP_STRIDE);
                PipeBarrier<PIPE_V>();
                ResetMask();
                SetMaskNorm();

                T sum = tmpVec.GetValue(0);
                T inv = (sum > tiling.eps) ? static_cast<T>(1.0) / sum : static_cast<T>(0.0);
                rowInvHistory.SetValue(iter * M + r, inv);
            }

            // Phase 1b: Apply row normalization using masked multiply
            for (int32_t r = 0; r < M; ++r) {
                int32_t base = r * this->N_aligned;
                T inv = rowInvHistory.GetValue(iter * M + r);

                // Copy row to tmpVec
                for (int32_t j = 0; j < this->N_aligned; ++j) {
                    tmpVec.SetValue(j, (j < N) ? tileFwd.GetValue(base + j) : static_cast<T>(0.0));
                }
                PipeBarrier<PIPE_V>();

                // Masked scalar multiply
                SetMaskCount();
                SetVectorMask<T>(0, static_cast<uint32_t>(N));
                Muls<T, false>(tmpVec, tmpVec, inv, MASK_PLACEHOLDER, 1, {1, 1, 8, 8});
                PipeBarrier<PIPE_V>();
                ResetMask();
                SetMaskNorm();

                // Copy back to tileFwd
                for (int32_t j = 0; j < N; ++j) {
                    tileFwd.SetValue(base + j, tmpVec.GetValue(j));
                }
            }
            PipeBarrier<PIPE_V>();

            // Save A matrix to history
            for (int32_t i = 0; i < M; ++i) {
                int32_t base_aligned = i * this->N_aligned;
                int32_t hist_base = iter * this->matrix_size_aligned + base_aligned;
                for (int32_t j = 0; j < N; ++j) {
                    aHistory.SetValue(hist_base + j, tileFwd.GetValue(base_aligned + j));
                }
            }

            // Column normalization - scalar sum, vectorized apply
            // Column sum (scalar - non-contiguous access)
            for (int32_t c = 0; c < N; ++c) {
                T sum = static_cast<T>(0.0);
                for (int32_t r = 0; r < M; ++r) {
                    sum += tileFwd.GetValue(r * this->N_aligned + c);
                }
                T inv = (sum > tiling.eps) ? static_cast<T>(1.0) / sum : static_cast<T>(0.0);
                colInvHistory.SetValue(iter * N_aligned + c, inv);
                colBuf.SetValue(c, inv);
            }
            for (int32_t c = N; c < N_aligned; ++c) {
                colBuf.SetValue(c, static_cast<T>(1.0));
            }
            PipeBarrier<PIPE_V>();

            // Apply column normalization using masked element-wise multiply
            for (int32_t r = 0; r < M; ++r) {
                int32_t base = r * this->N_aligned;

                // Copy row to tmpVec
                for (int32_t j = 0; j < this->N_aligned; ++j) {
                    tmpVec.SetValue(j, (j < N) ? tileFwd.GetValue(base + j) : static_cast<T>(0.0));
                }
                PipeBarrier<PIPE_V>();

                // Masked element-wise multiply with colBuf
                SetMaskCount();
                SetVectorMask<T>(0, static_cast<uint32_t>(N));
                Mul<T, false>(tmpVec, tmpVec, colBuf, MASK_PLACEHOLDER, 1, {1, 1, 1, 1, 1, 1});
                PipeBarrier<PIPE_V>();
                ResetMask();
                SetMaskNorm();

                // Copy back
                for (int32_t j = 0; j < N; ++j) {
                    tileFwd.SetValue(base + j, tmpVec.GetValue(j));
                }
            }
            PipeBarrier<PIPE_V>();
        }

        // PHASE 2: Backward through iterations using saved values (O(T))
        for (int32_t iter = tiling.num_iters - 1; iter >= 0; --iter) {
            // Load saved values for this iteration
            for (int32_t r = 0; r < M; ++r) {
                rowBuf.SetValue(r, rowInvHistory.GetValue(iter * M + r));
            }
            for (int32_t c = 0; c < N; ++c) {
                colBuf.SetValue(c, colInvHistory.GetValue(iter * N_aligned + c));
            }
            for (int32_t c = N; c < N_aligned; ++c) {
                colBuf.SetValue(c, static_cast<T>(0.0));
            }

            // Load A matrix from history
            for (int32_t i = 0; i < M; ++i) {
                int32_t base_aligned = i * this->N_aligned;
                int32_t hist_base = iter * this->matrix_size_aligned + base_aligned;
                for (int32_t j = 0; j < N; ++j) {
                    tileFwd.SetValue(base_aligned + j, aHistory.GetValue(hist_base + j));
                }
            }
            PipeBarrier<PIPE_V>();

            // Column backward: dA = (dY - sCol) * colInv
            // Step 1: Compute dot[c] = sum_r(dY[r,c] * A[r,c]) - scalar (non-contiguous)
            for (int32_t c = 0; c < N; ++c) {
                T dot = static_cast<T>(0.0);
                for (int32_t r = 0; r < M; ++r) {
                    int32_t base = r * this->N_aligned;
                    dot += gradMatrix.GetValue(base + c) * tileFwd.GetValue(base + c);
                }
                T colInv = colBuf.GetValue(c);
                tmpVec2.SetValue(c, dot * colInv);  // sCol[c] = dot[c] * colInv[c]
            }
            for (int32_t c = N; c < N_aligned; ++c) {
                tmpVec2.SetValue(c, static_cast<T>(0.0));
            }
            PipeBarrier<PIPE_V>();

            // Step 2: dA[r,c] = (dY[r,c] - sCol[c]) * colInv[c] - VECTORIZED per row
            for (int32_t r = 0; r < M; ++r) {
                int32_t base = r * this->N_aligned;

                // Copy gradMatrix row to tmpVec
                for (int32_t j = 0; j < this->N_aligned; ++j) {
                    tmpVec.SetValue(j, (j < N) ? gradMatrix.GetValue(base + j) : static_cast<T>(0.0));
                }
                PipeBarrier<PIPE_V>();

                // tmpVec = tmpVec - sCol (masked sub)
                SetMaskCount();
                SetVectorMask<T>(0, static_cast<uint32_t>(N));
                Sub<T, false>(tmpVec, tmpVec, tmpVec2, MASK_PLACEHOLDER, 1, {1, 1, 1, 1, 1, 1});
                PipeBarrier<PIPE_V>();
                ResetMask();
                SetMaskNorm();

                // tmpVec = tmpVec * colInv (masked mul)
                SetMaskCount();
                SetVectorMask<T>(0, static_cast<uint32_t>(N));
                Mul<T, false>(tmpVec, tmpVec, colBuf, MASK_PLACEHOLDER, 1, {1, 1, 1, 1, 1, 1});
                PipeBarrier<PIPE_V>();
                ResetMask();
                SetMaskNorm();

                // Copy back to gradMatrix
                for (int32_t j = 0; j < N; ++j) {
                    gradMatrix.SetValue(base + j, tmpVec.GetValue(j));
                }
            }
            PipeBarrier<PIPE_V>();

            // Row backward: dX = (dA - sRow) * rowInv - VECTORIZED
            for (int32_t r = 0; r < M; ++r) {
                int32_t base = r * this->N_aligned;
                T rowInv = rowBuf.GetValue(r);

                // Copy gradMatrix row and tileFwd row to buffers
                for (int32_t j = 0; j < this->N_aligned; ++j) {
                    tmpVec.SetValue(j, (j < N) ? gradMatrix.GetValue(base + j) : static_cast<T>(0.0));
                    tmpVec2.SetValue(j, (j < N) ? tileFwd.GetValue(base + j) : static_cast<T>(0.0));
                }
                PipeBarrier<PIPE_V>();

                // Compute dot = sum(dA * A) using masked reduce
                SetMaskCount();
                SetVectorMask<T>(0, static_cast<uint32_t>(N));
                Mul<T, false>(tmpVec2, tmpVec, tmpVec2, MASK_PLACEHOLDER, 1, {1, 1, 1, 1, 1, 1});
                PipeBarrier<PIPE_V>();
                WholeReduceSum<T, false>(tmpVec2, tmpVec2, MASK_PLACEHOLDER, 1,
                    DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REP_STRIDE);
                PipeBarrier<PIPE_V>();
                ResetMask();
                SetMaskNorm();

                T dot = tmpVec2.GetValue(0);
                T sRow = dot * rowInv;

                // Reload gradMatrix row (tmpVec was used)
                for (int32_t j = 0; j < this->N_aligned; ++j) {
                    tmpVec.SetValue(j, (j < N) ? gradMatrix.GetValue(base + j) : static_cast<T>(0.0));
                }
                PipeBarrier<PIPE_V>();

                // tmpVec = tmpVec - sRow (masked adds with -sRow)
                SetMaskCount();
                SetVectorMask<T>(0, static_cast<uint32_t>(N));
                Adds<T, false>(tmpVec, tmpVec, -sRow, MASK_PLACEHOLDER, 1, {1, 1, 8, 8});
                PipeBarrier<PIPE_V>();
                ResetMask();
                SetMaskNorm();

                // tmpVec = tmpVec * rowInv (masked muls)
                SetMaskCount();
                SetVectorMask<T>(0, static_cast<uint32_t>(N));
                Muls<T, false>(tmpVec, tmpVec, rowInv, MASK_PLACEHOLDER, 1, {1, 1, 8, 8});
                PipeBarrier<PIPE_V>();
                ResetMask();
                SetMaskNorm();

                // Copy back to gradMatrix
                for (int32_t j = 0; j < N; ++j) {
                    gradMatrix.SetValue(base + j, tmpVec.GetValue(j));
                }
            }
            PipeBarrier<PIPE_V>();
        }

        // Write gradient back to GM using DataCopyPad (SetValue has bugs for small tensors!)
        // Copy gradMatrix to a contiguous buffer first, then use DataCopyPad
        LocalTensor<T> gradOut = gradQueue.AllocTensor<T>();
        for (int32_t i = 0; i < M; ++i) {
            int32_t base_aligned = i * this->N_aligned;
            int32_t out_base = i * this->N;
            for (int32_t j = 0; j < N; ++j) {
                gradOut.SetValue(out_base + j, gradMatrix.GetValue(base_aligned + j));
            }
        }
        PipeBarrier<PIPE_V>();

        // Use DataCopyPad to write to GM
        AscendC::DataCopyExtParams copyParams = {
            1, static_cast<uint32_t>(this->matrix_size * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPad(gradInpGm[batch_idx * this->matrix_size], gradOut, copyParams);
        PipeBarrier<PIPE_MTE2>();

        gradQueue.FreeTensor(gradOut);
    }

    /**
     * @brief Process backward for M=N=4 using fully register-based computation (NO MASK)
     *
     * This is a specialized implementation for the common case M=N=4 mentioned in the
     * mHC paper (Section 4.3). By keeping all data in local variables and avoiding
     * Mask operations, this implementation is safe for multi-core execution.
     *
     * Strategy (similar to CUDA's float4 register-based approach):
     * 1. Load entire 4x4 matrix into 16 scalar variables
     * 2. Perform all computations using unrolled scalar operations
     * 3. No SetMaskCount/SetVectorMask calls - avoids multi-core interference
     */
    __aicore__ inline void ProcessSingleBackward_N4_Optimized(int32_t batch_idx) {
        // For M=N=4: matrix_size = 16, small enough to fit in registers
        // Using local arrays instead of individual variables for cleaner code
        constexpr int32_t N4 = 4;
        constexpr int32_t M4 = 4;
        constexpr int32_t MAT_SIZE = 16;

        // Gradient matrix (16 floats)
        T grad[MAT_SIZE];
        // Forward intermediate state A (16 floats)
        T A[MAT_SIZE];
        // Input matrix for forward recomputation (16 floats)
        T inp[MAT_SIZE];
        // Row/col inverse values (4 floats each)
        T rowInv[M4];
        T colInv[N4];

        // History storage for O(T) complexity
        // For typical num_iters=20: rowInvHist = 80 floats, colInvHist = 80 floats, aHist = 320 floats
        // Total ~480 floats < 2KB, fits in UB
        LocalTensor<T> rowInvHistory = rowInvHistoryBuf.Get<T>();
        LocalTensor<T> colInvHistory = colInvHistoryBuf.Get<T>();
        LocalTensor<T> aHistory = aHistoryBuf.Get<T>();

        // ============================================================
        // Load gradient output and input from GM
        // ============================================================
        int32_t gm_base = batch_idx * MAT_SIZE;
        #pragma unroll
        for (int32_t i = 0; i < MAT_SIZE; ++i) {
            grad[i] = gradOutGm.GetValue(gm_base + i);
            inp[i] = inpGm.GetValue(gm_base + i);
        }

        // ============================================================
        // PHASE 1: Forward recomputation with history saving
        // ============================================================

        // Copy input to A for forward computation
        #pragma unroll
        for (int32_t i = 0; i < MAT_SIZE; ++i) {
            A[i] = inp[i];
        }

        for (int32_t iter = 0; iter < tiling.num_iters; ++iter) {
            // Row normalization: compute row sums
            T s0 = A[0] + A[1] + A[2] + A[3];
            T s1 = A[4] + A[5] + A[6] + A[7];
            T s2 = A[8] + A[9] + A[10] + A[11];
            T s3 = A[12] + A[13] + A[14] + A[15];

            // Compute row inverses
            rowInv[0] = (s0 > tiling.eps) ? static_cast<T>(1.0) / s0 : static_cast<T>(0.0);
            rowInv[1] = (s1 > tiling.eps) ? static_cast<T>(1.0) / s1 : static_cast<T>(0.0);
            rowInv[2] = (s2 > tiling.eps) ? static_cast<T>(1.0) / s2 : static_cast<T>(0.0);
            rowInv[3] = (s3 > tiling.eps) ? static_cast<T>(1.0) / s3 : static_cast<T>(0.0);

            // Save row inverses to history
            rowInvHistory.SetValue(iter * M4 + 0, rowInv[0]);
            rowInvHistory.SetValue(iter * M4 + 1, rowInv[1]);
            rowInvHistory.SetValue(iter * M4 + 2, rowInv[2]);
            rowInvHistory.SetValue(iter * M4 + 3, rowInv[3]);

            // Apply row normalization
            A[0] *= rowInv[0]; A[1] *= rowInv[0]; A[2] *= rowInv[0]; A[3] *= rowInv[0];
            A[4] *= rowInv[1]; A[5] *= rowInv[1]; A[6] *= rowInv[1]; A[7] *= rowInv[1];
            A[8] *= rowInv[2]; A[9] *= rowInv[2]; A[10] *= rowInv[2]; A[11] *= rowInv[2];
            A[12] *= rowInv[3]; A[13] *= rowInv[3]; A[14] *= rowInv[3]; A[15] *= rowInv[3];

            // Save A matrix to history (after row norm, before col norm)
            #pragma unroll
            for (int32_t i = 0; i < MAT_SIZE; ++i) {
                aHistory.SetValue(iter * MAT_SIZE + i, A[i]);
            }

            // Column normalization: compute column sums
            T c0 = A[0] + A[4] + A[8] + A[12];
            T c1 = A[1] + A[5] + A[9] + A[13];
            T c2 = A[2] + A[6] + A[10] + A[14];
            T c3 = A[3] + A[7] + A[11] + A[15];

            // Compute column inverses
            colInv[0] = (c0 > tiling.eps) ? static_cast<T>(1.0) / c0 : static_cast<T>(0.0);
            colInv[1] = (c1 > tiling.eps) ? static_cast<T>(1.0) / c1 : static_cast<T>(0.0);
            colInv[2] = (c2 > tiling.eps) ? static_cast<T>(1.0) / c2 : static_cast<T>(0.0);
            colInv[3] = (c3 > tiling.eps) ? static_cast<T>(1.0) / c3 : static_cast<T>(0.0);

            // Save column inverses to history
            colInvHistory.SetValue(iter * N4 + 0, colInv[0]);
            colInvHistory.SetValue(iter * N4 + 1, colInv[1]);
            colInvHistory.SetValue(iter * N4 + 2, colInv[2]);
            colInvHistory.SetValue(iter * N4 + 3, colInv[3]);

            // Apply column normalization
            A[0] *= colInv[0]; A[1] *= colInv[1]; A[2] *= colInv[2]; A[3] *= colInv[3];
            A[4] *= colInv[0]; A[5] *= colInv[1]; A[6] *= colInv[2]; A[7] *= colInv[3];
            A[8] *= colInv[0]; A[9] *= colInv[1]; A[10] *= colInv[2]; A[11] *= colInv[3];
            A[12] *= colInv[0]; A[13] *= colInv[1]; A[14] *= colInv[2]; A[15] *= colInv[3];
        }

        // ============================================================
        // PHASE 2: Backward using saved history (O(T) complexity)
        // ============================================================

        for (int32_t iter = tiling.num_iters - 1; iter >= 0; --iter) {
            // Load saved values from history
            rowInv[0] = rowInvHistory.GetValue(iter * M4 + 0);
            rowInv[1] = rowInvHistory.GetValue(iter * M4 + 1);
            rowInv[2] = rowInvHistory.GetValue(iter * M4 + 2);
            rowInv[3] = rowInvHistory.GetValue(iter * M4 + 3);

            colInv[0] = colInvHistory.GetValue(iter * N4 + 0);
            colInv[1] = colInvHistory.GetValue(iter * N4 + 1);
            colInv[2] = colInvHistory.GetValue(iter * N4 + 2);
            colInv[3] = colInvHistory.GetValue(iter * N4 + 3);

            #pragma unroll
            for (int32_t i = 0; i < MAT_SIZE; ++i) {
                A[i] = aHistory.GetValue(iter * MAT_SIZE + i);
            }

            // --------------------------------------------------------
            // Column backward: dA = (dY - sCol) * colInv
            // --------------------------------------------------------

            // Compute dot[c] = sum_r(dY[r,c] * A[r,c])
            T dot0 = grad[0]*A[0] + grad[4]*A[4] + grad[8]*A[8] + grad[12]*A[12];
            T dot1 = grad[1]*A[1] + grad[5]*A[5] + grad[9]*A[9] + grad[13]*A[13];
            T dot2 = grad[2]*A[2] + grad[6]*A[6] + grad[10]*A[10] + grad[14]*A[14];
            T dot3 = grad[3]*A[3] + grad[7]*A[7] + grad[11]*A[11] + grad[15]*A[15];

            // Compute sCol[c] = dot[c] * colInv[c]
            T sCol0 = dot0 * colInv[0];
            T sCol1 = dot1 * colInv[1];
            T sCol2 = dot2 * colInv[2];
            T sCol3 = dot3 * colInv[3];

            // Compute dA[r,c] = (dY[r,c] - sCol[c]) * colInv[c]
            grad[0] = (grad[0] - sCol0) * colInv[0];
            grad[1] = (grad[1] - sCol1) * colInv[1];
            grad[2] = (grad[2] - sCol2) * colInv[2];
            grad[3] = (grad[3] - sCol3) * colInv[3];
            grad[4] = (grad[4] - sCol0) * colInv[0];
            grad[5] = (grad[5] - sCol1) * colInv[1];
            grad[6] = (grad[6] - sCol2) * colInv[2];
            grad[7] = (grad[7] - sCol3) * colInv[3];
            grad[8] = (grad[8] - sCol0) * colInv[0];
            grad[9] = (grad[9] - sCol1) * colInv[1];
            grad[10] = (grad[10] - sCol2) * colInv[2];
            grad[11] = (grad[11] - sCol3) * colInv[3];
            grad[12] = (grad[12] - sCol0) * colInv[0];
            grad[13] = (grad[13] - sCol1) * colInv[1];
            grad[14] = (grad[14] - sCol2) * colInv[2];
            grad[15] = (grad[15] - sCol3) * colInv[3];

            // --------------------------------------------------------
            // Row backward: dX = (dA - sRow) * rowInv
            // --------------------------------------------------------

            // Compute dot[r] = sum_c(dA[r,c] * A[r,c])
            T dotR0 = grad[0]*A[0] + grad[1]*A[1] + grad[2]*A[2] + grad[3]*A[3];
            T dotR1 = grad[4]*A[4] + grad[5]*A[5] + grad[6]*A[6] + grad[7]*A[7];
            T dotR2 = grad[8]*A[8] + grad[9]*A[9] + grad[10]*A[10] + grad[11]*A[11];
            T dotR3 = grad[12]*A[12] + grad[13]*A[13] + grad[14]*A[14] + grad[15]*A[15];

            // Compute sRow[r] = dot[r] * rowInv[r]
            T sRow0 = dotR0 * rowInv[0];
            T sRow1 = dotR1 * rowInv[1];
            T sRow2 = dotR2 * rowInv[2];
            T sRow3 = dotR3 * rowInv[3];

            // Compute dX[r,c] = (dA[r,c] - sRow[r]) * rowInv[r]
            grad[0] = (grad[0] - sRow0) * rowInv[0];
            grad[1] = (grad[1] - sRow0) * rowInv[0];
            grad[2] = (grad[2] - sRow0) * rowInv[0];
            grad[3] = (grad[3] - sRow0) * rowInv[0];
            grad[4] = (grad[4] - sRow1) * rowInv[1];
            grad[5] = (grad[5] - sRow1) * rowInv[1];
            grad[6] = (grad[6] - sRow1) * rowInv[1];
            grad[7] = (grad[7] - sRow1) * rowInv[1];
            grad[8] = (grad[8] - sRow2) * rowInv[2];
            grad[9] = (grad[9] - sRow2) * rowInv[2];
            grad[10] = (grad[10] - sRow2) * rowInv[2];
            grad[11] = (grad[11] - sRow2) * rowInv[2];
            grad[12] = (grad[12] - sRow3) * rowInv[3];
            grad[13] = (grad[13] - sRow3) * rowInv[3];
            grad[14] = (grad[14] - sRow3) * rowInv[3];
            grad[15] = (grad[15] - sRow3) * rowInv[3];
        }

        // ============================================================
        // Write gradient back to GM using DataCopyPad
        // ============================================================
        LocalTensor<T> gradOut = gradQueue.AllocTensor<T>();
        #pragma unroll
        for (int32_t i = 0; i < MAT_SIZE; ++i) {
            gradOut.SetValue(i, grad[i]);
        }
        PipeBarrier<PIPE_V>();

        AscendC::DataCopyExtParams copyParams = {
            1, static_cast<uint32_t>(MAT_SIZE * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPad(gradInpGm[gm_base], gradOut, copyParams);
        PipeBarrier<PIPE_MTE2>();

        gradQueue.FreeTensor(gradOut);
    }

    /**
     * @brief Dispatch to appropriate backward implementation based on configuration
     * @param batch_idx Index within the batch handled by this core
     *
     * Dispatch strategy (controllable via MHC_SINKHORN_BWD_IMPL environment variable):
     * - auto (default): Automatic selection based on M, N, core count
     *   1. M=N=4: Use fully optimized register-based version (NO MASK, multi-core safe)
     *   2. N%4==0 or single-core: Use vectorized version with Mask operations
     *   3. Otherwise: Use scalar version
     * - scalar: Force scalar implementation
     * - vectorized: Force vectorized implementation
     * - n4_optimized: Force N4 optimized implementation (only valid for M=N=4)
     */
    __aicore__ inline void ProcessSingleBackward(int32_t batch_idx) {
        // Check if explicit implementation mode is set
        int32_t impl_mode = this->tiling.bwd_impl_mode;

        // Handle explicit mode selection
        if (impl_mode == 1) {  // SCALAR
            ProcessSingleBackward_Scalar(batch_idx);
            return;
        } else if (impl_mode == 2) {  // VECTORIZED
            // Safety check: vectorized is unsafe with multi-core + N%4!=0
            // In that case, fallback to scalar to avoid 40-60% errors
            bool vectorized_safe = (this->N % 4 == 0) || (this->tiling.used_core_num == 1);
            if (vectorized_safe) {
                ProcessSingleBackward_Vectorized(batch_idx);
            } else {
                // Forced vectorized but unsafe -> fallback to scalar
                ProcessSingleBackward_Scalar(batch_idx);
            }
            return;
        } else if (impl_mode == 3) {  // N4_OPTIMIZED
            // Only valid for M=N=4, fallback to scalar otherwise
            if (this->M == 4 && this->N == 4) {
                ProcessSingleBackward_N4_Optimized(batch_idx);
            } else {
                ProcessSingleBackward_Scalar(batch_idx);
            }
            return;
        }

        // AUTO mode (impl_mode == 0): automatic selection
        // Check if we can use the highly optimized M=N=4 version
        // This version uses no Mask operations, so it's safe for multi-core
        if (this->M == 4 && this->N == 4) {
            ProcessSingleBackward_N4_Optimized(batch_idx);
            return;
        }

        // For other sizes, use the N%4 dispatch strategy like forward kernel
        bool can_use_vectorized = (this->N % 4 == 0) || (this->tiling.used_core_num == 1);

        if (can_use_vectorized) {
            ProcessSingleBackward_Vectorized(batch_idx);
        } else {
            ProcessSingleBackward_Scalar(batch_idx);
        }
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, NUM_BUFFERS> gradQueue, matrixQueue, outMatrixQueue;
    TBuf<QuePosition::VECCALC> tmpBuf1, tmpBuf2, tileFwdBuf;
    // Phase 2 optimization: History buffers for O(T) complexity
    TBuf<QuePosition::VECCALC> rowInvHistoryBuf, colInvHistoryBuf, aHistoryBuf;

    GlobalTensor<T> gradOutGm, inpGm, outGm, gradInpGm;

    SinkhornTiling tiling;

    int32_t M, N;
    int32_t N_aligned;  // Aligned N to ensure each row starts at aligned address
    int32_t matrix_size;
    int32_t matrix_size_aligned;  // Aligned matrix size for internal buffers
    int32_t batch_per_core;
    int32_t batch_start;
    int32_t batch_count;
    int32_t total_size;
};

// =============================================================================
// Sinkhorn-Knopp Backward Kernel Entry Point
// =============================================================================

/**
 * @brief Sinkhorn-Knopp backward kernel entry
 */
extern "C" __global__ __aicore__ void sinkhorn_knopp_backward(
    GM_ADDR grad_out,
    GM_ADDR inp,
    GM_ADDR out,
    GM_ADDR grad_inp,
    GM_ADDR tiling)
{
    SinkhornTiling tiling_data;
    InitTilingData(tiling, &tiling_data);

    SinkhornKnoppBackwardKernel<float> kernel;
    kernel.Init(grad_out, inp, out, grad_inp, tiling_data);
    kernel.Process();
}
