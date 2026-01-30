/**
 * @file mhc_types.h
 * @brief Type definitions for mHC Ascend implementation
 *
 * Copyright (C) 2025. All rights reserved.
 *
 * This file defines common types and constants used across all mHC kernels.
 * It provides Ascend C equivalents to the CUDA types used in the original implementation.
 */

#ifndef MHC_ASCEND_TYPES_H
#define MHC_ASCEND_TYPES_H

#include <stdint.h>

#ifndef MHC_HOST_BUILD
#include "__clang_cce_types.h"  // bfloat16_t definition for device builds
#include "kernel_operator.h"
#endif

namespace mhc_ascend {

// =============================================================================
// Type Aliases
// =============================================================================

/**
 * Primary storage type (bfloat16 for memory efficiency)
 */
#ifndef MHC_HOST_BUILD
using floatX = bfloat16_t;  // Equivalent to nv_bfloat16 in CUDA version
#else
using floatX = uint16_t;  // Placeholder type for host-only builds
#endif

/**
 * Accumulation type (float32 for precision)
 */
using floatN = float;

/**
 * Index type
 */
using index_t = int32_t;

// =============================================================================
// Constants
// =============================================================================

/**
 * Default Sinkhorn-Knopp normalization iterations
 */
constexpr int32_t DEFAULT_SINKHORN_ITERS = 20;

/**
 * Default epsilon for numerical stability
 */
constexpr float DEFAULT_EPSILON = 1e-8f;

/**
 * Default RMSNorm epsilon
 */
constexpr float DEFAULT_RMSNORM_EPS = 1e-5f;

/**
 * Maximum supported expansion rate
 */
constexpr int32_t MAX_EXPANSION_RATE = 64;

/**
 * Maximum supported matrix dimension for Sinkhorn-Knopp
 */
constexpr int32_t MAX_SINKHORN_DIM = 128;

/**
 * Tile size for matrix operations (Ascend-specific)
 */
constexpr int32_t TILE_M = 16;
constexpr int32_t TILE_N = 16;
constexpr int32_t TILE_K = 16;

/**
 * Buffer sizes for queue management
 */
constexpr int32_t NUM_BUFFERS = 2;  // Double buffering

/**
 * Vector processing sizes
 * REP_LEN is the byte size of one repeat (256 bytes on Ascend)
 */
constexpr int32_t REP_LEN = 256;
constexpr int32_t BLK_LEN = 32;

// =============================================================================
// Tiling Data Structures
// =============================================================================

/**
 * Sinkhorn forward implementation mode
 * Controllable via MHC_SINKHORN_FWD_IMPL environment variable
 */
enum class SinkhornFwdImplMode : int32_t {
    AUTO = 0,        // Automatic selection based on N alignment and core count
    SCALAR = 1,      // Force scalar (v2_Optimized) implementation
    VECTORIZED = 2   // Force vectorized (v3_Vectorized) implementation
};

/**
 * Sinkhorn backward implementation mode
 * Controllable via MHC_SINKHORN_BWD_IMPL environment variable
 */
enum class SinkhornBwdImplMode : int32_t {
    AUTO = 0,        // Automatic selection based on M, N, core count
    SCALAR = 1,      // Force scalar implementation
    VECTORIZED = 2,  // Force vectorized implementation (may fail if N%4!=0 && multicore)
    N4_OPTIMIZED = 3 // Force N4 optimized implementation (only valid for M=N=4)
};

/**
 * Tiling configuration for Sinkhorn-Knopp kernel
 */
struct SinkhornTiling {
    int32_t batch_size;        // Number of matrices in batch
    int32_t M;                 // Number of rows
    int32_t N;                 // Number of columns
    int32_t num_iters;         // Number of normalization iterations
    float eps;                 // Epsilon for numerical stability
    int32_t used_core_num;     // Number of AI cores to use
    int32_t fwd_impl_mode;     // Forward implementation mode (SinkhornFwdImplMode)
    int32_t bwd_impl_mode;     // Backward implementation mode (SinkhornBwdImplMode)
};

/**
 * Tiling configuration for RMSNorm kernel
 */
struct RMSNormTiling {
    int32_t batch_size;        // Batch size
    int32_t hidden_dim;        // Hidden dimension size
    float eps;                 // Epsilon for numerical stability
    int32_t used_core_num;     // Number of AI cores
    int32_t tile_size;         // Size of each processing tile
};

/**
 * Tiling configuration for stream aggregation
 */
struct StreamAggregateTiling {
    int32_t batch_size;        // Batch size
    int32_t n;                 // Number of streams (expansion rate)
    int32_t C;                 // Hidden dimension
    int32_t used_core_num;     // Number of AI cores
};

/**
 * Tiling configuration for stream distribute mix add
 */
struct StreamDistributeMixAddTiling {
    int32_t batch_size;        // Batch size
    int32_t n;                 // Number of streams (expansion rate)
    int32_t C;                 // Hidden dimension
    int32_t used_core_num;     // Number of AI cores
};

/**
 * Tiling configuration for general stream operations
 */
struct StreamOpsTiling {
    int32_t batch_size;        // Batch size
    int32_t expansion_rate;    // Expansion rate (n)
    int32_t hidden_dim;        // Hidden dimension (C)
    int32_t used_core_num;     // Number of AI cores
    int32_t tile_size;         // Tile size for processing
};

/**
 * Tiling configuration for fused RMSNorm + MatMul
 */
struct FusedRMSNormMatMulTiling {
    int32_t batch_size;        // Batch size
    int32_t hidden_dim;        // Input hidden dimension
    int32_t out_dim;           // Output dimension
    float eps;                 // RMSNorm epsilon
    int32_t used_core_num;     // Number of AI cores
};

/**
 * Tiling configuration for compute_rms kernel
 */
struct ComputeRMSTiling {
    int32_t batch_size;        // Batch size
    int32_t hidden_dim;        // Hidden dimension
    float eps;                 // Epsilon for numerical stability
    int32_t used_core_num;     // Number of AI cores
};

/**
 * Tiling configuration for full MHC layer
 */
struct MHCLayerTiling {
    int32_t batch_size;           // Batch size
    int32_t expansion_rate;       // Expansion rate (n)
    int32_t hidden_dim;           // Hidden dimension (C)
    int32_t num_sinkhorn_iters;   // Sinkhorn-Knopp iterations
    float sinkhorn_eps;           // Sinkhorn epsilon
    float rmsnorm_eps;            // RMSNorm epsilon
    int32_t used_core_num;        // Number of AI cores
    bool use_dynamic_h;           // Whether to use dynamic H computation
};

// =============================================================================
// Utility Macros
// =============================================================================

/**
 * Ceiling division
 */
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

/**
 * Align to multiple
 */
#define ALIGN_UP(x, align) (((x) + (align) - 1) / (align) * (align))

/**
 * Min/Max macros
 */
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

/**
 * Check if value is power of 2
 */
#define IS_POW2(x) (((x) & ((x) - 1)) == 0)

// =============================================================================
// Error Checking
// =============================================================================

/**
 * Runtime assertion for kernel code
 */
#ifdef __aicore__
#define MHC_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            /* In Ascend C, we can't directly print, but we can trigger error */ \
            return; \
        } \
    } while (0)
#else
#define MHC_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "Assertion failed: %s\n", msg); \
            abort(); \
        } \
    } while (0)
#endif

// =============================================================================
// Memory Position Tags (Ascend C specific)
// =============================================================================

/**
 * Queue position enums are accessed via AscendC::QuePosition
 * Examples:
 *   - QuePosition::VECIN  : Vector input queue
 *   - QuePosition::VECOUT : Vector output queue
 *   - QuePosition::VECCALC: Vector calculation buffer
 *
 * Use directly with TQue, e.g.:
 *   TQue<QuePosition::VECIN, NUM_BUFFERS> inQueue;
 */

} // namespace mhc_ascend

#endif // MHC_ASCEND_TYPES_H
