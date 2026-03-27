//===- gpu_ops.cpp - GPU operator implementations -------------------------===//
//
// Real implementations for the operators declared in gpu_runner.h.  cuBLAS
// is used for GEMM-flavored ops (matmul, linear), cuDNN drives softmax, and
// custom CUDA kernels (gpu_kernels.cu) handle the rest.
//
//===----------------------------------------------------------------------===//

#include "gpu_runner.h"
#include "gpu_kernels.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace joy {
namespace gpu {

namespace {

// ----- error helpers (no exceptions / RTTI used) -----

#define JOY_CUDA_CHECK(call)                                                   \
  do {                                                                         \
    cudaError_t e = (call);                                                    \
    if (e != cudaSuccess) {                                                    \
      fprintf(stderr, "[joy gpu] CUDA error at %s:%d: %s\n", __FILE__,         \
              __LINE__, cudaGetErrorString(e));                                \
    }                                                                          \
  } while (0)

#define JOY_CUBLAS_CHECK(call)                                                 \
  do {                                                                         \
    cublasStatus_t s = (call);                                                 \
    if (s != CUBLAS_STATUS_SUCCESS) {                                          \
      fprintf(stderr, "[joy gpu] cuBLAS error at %s:%d: %d\n", __FILE__,       \
              __LINE__, (int)s);                                               \
    }                                                                          \
  } while (0)

#define JOY_CUDNN_CHECK(call)                                                  \
  do {                                                                         \
    cudnnStatus_t s = (call);                                                  \
    if (s != CUDNN_STATUS_SUCCESS) {                                           \
      fprintf(stderr, "[joy gpu] cuDNN error at %s:%d: %s\n", __FILE__,        \
              __LINE__, cudnnGetErrorString(s));                               \
    }                                                                          \
  } while (0)

// Element type ids (mirror MemrefDesc::element_type)
constexpr int32_t kF16 = 0;
constexpr int32_t kF32 = 1;
constexpr int32_t kF64 = 2;
constexpr int32_t kI32 = 3;
constexpr int32_t kI64 = 4;

inline cudaStream_t streamOf(GpuContext *ctx) {
  return ctx ? reinterpret_cast<cudaStream_t>(ctx->stream) : nullptr;
}

inline cublasHandle_t cublasOf(GpuContext *ctx) {
  return ctx ? reinterpret_cast<cublasHandle_t>(ctx->cublas) : nullptr;
}

inline cudnnHandle_t cudnnOf(GpuContext *ctx) {
  return ctx ? reinterpret_cast<cudnnHandle_t>(ctx->cudnn) : nullptr;
}

// Product of last `keep` dims; returns (outer, inner)
inline void splitOuterInner(const MemrefDesc &m, int keep, int64_t &outer,
                            int64_t &inner) {
  outer = 1;
  inner = 1;
  for (int64_t i = 0; i < m.rank - keep; ++i) outer *= m.shape[i];
  for (int64_t i = m.rank - keep; i < m.rank; ++i) inner *= m.shape[i];
}

// Compute outer = prod(shape[0..rank-2]), last = shape[rank-1].
inline void rowsAndLast(const MemrefDesc &m, int64_t &rows, int64_t &last) {
  rows = 1;
  for (int64_t i = 0; i + 1 < m.rank; ++i) rows *= m.shape[i];
  last = (m.rank == 0) ? 1 : m.shape[m.rank - 1];
}

cudnnDataType_t cudnnDtype(int32_t et) {
  switch (et) {
  case kF16: return CUDNN_DATA_HALF;
  case kF32: return CUDNN_DATA_FLOAT;
  case kF64: return CUDNN_DATA_DOUBLE;
  default:   return CUDNN_DATA_FLOAT;
  }
}

cudaDataType cublasDtype(int32_t et) {
  switch (et) {
  case kF16: return CUDA_R_16F;
  case kF32: return CUDA_R_32F;
  case kF64: return CUDA_R_64F;
  default:   return CUDA_R_32F;
  }
}

// Generic "row-major" GEMM via cuBLAS:
//   C[M,N] = alpha * A[M,K] * B[K,N]^opB + beta * C[M,N]
// where data is laid out in row-major.  We map this to column-major cuBLAS
// using the identity: row_major(C) = column_major(C^T), so we issue
// cublasGemmEx with the operands swapped (B before A) and dimensions
// (m=N, n=M, k=K).
//
// transB: if true, B is shaped [N, K] row-major and the GEMM uses B^T.
//   This is the "linear" case: y = x @ W^T with W row-major [N, K].
void runGemm(GpuContext *ctx, const void *A, const void *B, void *C,
             int64_t M, int64_t N, int64_t K, bool transB, int32_t etype,
             float alpha = 1.0f, float beta = 0.0f) {
  cublasHandle_t handle = cublasOf(ctx);
  if (!handle) {
    fprintf(stderr, "[joy gpu] runGemm: missing cublas handle\n");
    return;
  }
  cudaDataType dt = cublasDtype(etype);
  cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;
  // For f64, use 64F compute.
  if (etype == kF64) computeType = CUBLAS_COMPUTE_64F;

  // In column-major view:
  //   C^T_col(N,M) = (op(B))^T_col(N,K) * (A)^T_col(K,M)
  //   row-major B[K,N] (transB=false) viewed in col-major is N x K, lda=N
  //     → use op = N (no further transpose), m=N
  //   row-major B[N,K] (transB=true)  viewed in col-major is K x N, lda=K
  //     → need to use op = T to get N x K, m=N
  // and row-major A[M,K] viewed in col-major is K x M, ldb=K, op=N → KxM
  // result C^T_col is N x M, written into the same memory as row-major C[M,N]

  cublasOperation_t opA = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = CUBLAS_OP_N;
  int m = (int)N;
  int n = (int)M;
  int k = (int)K;
  int lda = transB ? (int)K : (int)N;
  int ldb = (int)K;
  int ldc = (int)N;

  // Use mixed-precision compute: f16 inputs/outputs but f32 accumulate.
  float alpha32 = alpha, beta32 = beta;
  double alpha64 = alpha, beta64 = beta;
  const void *alpha_ptr =
      (etype == kF64) ? (const void *)&alpha64 : (const void *)&alpha32;
  const void *beta_ptr =
      (etype == kF64) ? (const void *)&beta64 : (const void *)&beta32;

  JOY_CUBLAS_CHECK(cublasSetStream(handle, streamOf(ctx)));
  JOY_CUBLAS_CHECK(cublasGemmEx(handle, opA, opB, m, n, k, alpha_ptr, B, dt,
                                lda, A, dt, ldb, beta_ptr, C, dt, ldc,
                                computeType, CUBLAS_GEMM_DEFAULT));
}

// ----- batched matmul helper -----
// For 3D+ inputs.  A has shape [..., M, K], B has shape [..., K, N] (or K,N
// transposed).  We reduce arbitrary leading dims to a single batch B.
// All batch dims must match.
void runBatchedGemm(GpuContext *ctx, const void *A, const void *B, void *C,
                    int64_t batch, int64_t M, int64_t N, int64_t K,
                    bool transB, int32_t etype) {
  cublasHandle_t handle = cublasOf(ctx);
  if (!handle) return;
  cudaDataType dt = cublasDtype(etype);
  cublasComputeType_t computeType =
      (etype == kF64) ? CUBLAS_COMPUTE_64F : CUBLAS_COMPUTE_32F;

  cublasOperation_t opA = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = CUBLAS_OP_N;
  int m = (int)N, n = (int)M, k = (int)K;
  int lda = transB ? (int)K : (int)N;
  int ldb = (int)K;
  int ldc = (int)N;

  long long strideA = (long long)N * K;
  long long strideB = (long long)M * K;
  long long strideC = (long long)M * N;

  float alpha32 = 1.0f, beta32 = 0.0f;
  double alpha64 = 1.0, beta64 = 0.0;
  const void *alpha_ptr =
      (etype == kF64) ? (const void *)&alpha64 : (const void *)&alpha32;
  const void *beta_ptr =
      (etype == kF64) ? (const void *)&beta64 : (const void *)&beta32;

  JOY_CUBLAS_CHECK(cublasSetStream(handle, streamOf(ctx)));
  JOY_CUBLAS_CHECK(cublasGemmStridedBatchedEx(
      handle, opA, opB, m, n, k, alpha_ptr, B, dt, lda, strideA, A, dt, ldb,
      strideB, beta_ptr, C, dt, ldc, strideC, (int)batch, computeType,
      CUBLAS_GEMM_DEFAULT));
}

} // namespace

// =====================================================================
// Embedding: ids[*shape] (int) + table[V, H] (T) -> out[*shape, H]
// =====================================================================
void GpuEmbeddingOp::compute() {
  // Inputs: 0=ids, 1=table.  Output: 0=out.
  if (numInputs_ < 2 || numOutputs_ < 1) return;
  MemrefDesc &ids = inputs_[0];
  MemrefDesc &table = inputs_[1];
  MemrefDesc &out = outputs_[0];

  int64_t n = ids.numElements();
  int64_t hidden = table.shape[table.rank - 1];
  int64_t vocab = table.shape[0];
  cudaStream_t s = streamOf(ctx_);

  if (table.element_type == kF32 && ids.element_type == kI32) {
    joy_kernel_embedding_f32_i32(
        reinterpret_cast<const float *>(table.data),
        reinterpret_cast<const int32_t *>(ids.data),
        reinterpret_cast<float *>(out.data), n, hidden, vocab, s);
  } else if (table.element_type == kF32 && ids.element_type == kI64) {
    joy_kernel_embedding_f32_i64(
        reinterpret_cast<const float *>(table.data),
        reinterpret_cast<const int64_t *>(ids.data),
        reinterpret_cast<float *>(out.data), n, hidden, vocab, s);
  } else if (table.element_type == kF16 && ids.element_type == kI64) {
    joy_kernel_embedding_f16_i64(table.data,
                                 reinterpret_cast<const int64_t *>(ids.data),
                                 out.data, n, hidden, vocab, s);
  } else {
    fprintf(stderr,
            "[GpuEmbeddingOp] unsupported dtype combo (table=%d, ids=%d)\n",
            table.element_type, ids.element_type);
  }
}

// =====================================================================
// RMSNorm: x [..., H], weight [H] -> y [..., H]
// =====================================================================
void GpuRMSNormOp::compute() {
  // RMSNorm now dispatches to the auto-generated kernels in
  // codegen_kernel.cu (produced from MLIR by joy-emit-cuda at build
  // time), not to a hand-written CUDA kernel.
  if (numInputs_ < 2 || numOutputs_ < 1) return;
  MemrefDesc &x = inputs_[0];
  MemrefDesc &w = inputs_[1];
  MemrefDesc &y = outputs_[0];

  int64_t outer, h;
  rowsAndLast(x, outer, h);
  cudaStream_t s = streamOf(ctx_);

  if (x.element_type == kF32) {
    joy_codegen_rms_norm_f32(reinterpret_cast<const float *>(x.data),
                             reinterpret_cast<const float *>(w.data),
                             reinterpret_cast<float *>(y.data), outer, h,
                             epsilon, s);
  } else if (x.element_type == kF16) {
    joy_codegen_rms_norm_f16(x.data, w.data, y.data, outer, h, epsilon, s);
  } else {
    fprintf(stderr, "[GpuRMSNormOp] unsupported dtype %d\n", x.element_type);
  }
}

// =====================================================================
// Linear (PyTorch nn.Linear convention): y = x @ W^T
//   x: [..., K]    W: [N, K]    y: [..., N]
// =====================================================================
void GpuLinearOp::compute() {
  if (numInputs_ < 2 || numOutputs_ < 1) return;
  MemrefDesc &x = inputs_[0];
  MemrefDesc &W = inputs_[1];
  MemrefDesc &y = outputs_[0];

  int64_t M = 1;
  for (int64_t i = 0; i + 1 < x.rank; ++i) M *= x.shape[i];
  int64_t K = x.shape[x.rank - 1];
  int64_t N = W.shape[0];

  runGemm(ctx_, x.data, W.data, y.data, M, N, K, /*transB=*/true,
          x.element_type);
}

// =====================================================================
// MatMul: y = x @ W
//   - 2D x 2D: [M, K] x [K, N]    -> [M, N]
//   - 3D x 3D: [B, M, K] x [B, K, N] -> [B, M, N] (batched GEMM)
//   - 4D x 4D: [B0, B1, M, K] x [B0, B1, K, N] -> [B0, B1, M, N]
//   No transpose flag here — that's what Linear is for.
// =====================================================================
void GpuMatMulOp::compute() {
  if (numInputs_ < 2 || numOutputs_ < 1) return;
  MemrefDesc &x = inputs_[0];
  MemrefDesc &W = inputs_[1];
  MemrefDesc &y = outputs_[0];

  if (x.rank == 2 && W.rank == 2) {
    int64_t M = x.shape[0];
    int64_t K = x.shape[1];
    int64_t N = W.shape[1];
    runGemm(ctx_, x.data, W.data, y.data, M, N, K, /*transB=*/false,
            x.element_type);
    return;
  }

  // Batched
  if (x.rank >= 3 && x.rank == W.rank) {
    int64_t B = 1;
    for (int64_t i = 0; i + 2 < x.rank; ++i) B *= x.shape[i];
    int64_t M = x.shape[x.rank - 2];
    int64_t K = x.shape[x.rank - 1];
    int64_t N = W.shape[W.rank - 1];
    runBatchedGemm(ctx_, x.data, W.data, y.data, B, M, N, K, /*transB=*/false,
                   x.element_type);
    return;
  }

  fprintf(stderr,
          "[GpuMatMulOp] unsupported input ranks (x.rank=%ld, W.rank=%ld)\n",
          (long)x.rank, (long)W.rank);
}

// =====================================================================
// Softmax along last axis (or axis attr, but cuDNN's softmax operates
// along channel dim — we view the tensor as N x C x 1 x 1 with C = last dim).
// =====================================================================
void GpuSoftmaxOp::compute() {
  if (numInputs_ < 1 || numOutputs_ < 1) return;
  MemrefDesc &x = inputs_[0];
  MemrefDesc &y = outputs_[0];

  // Compute the resolved axis.
  int64_t resolved = axis;
  if (resolved < 0) resolved += x.rank;
  if (resolved != x.rank - 1) {
    fprintf(stderr,
            "[GpuSoftmaxOp] only last-axis softmax is supported (got %ld)\n",
            (long)axis);
    return;
  }

  int64_t outer, last;
  rowsAndLast(x, outer, last);

  cudnnHandle_t handle = cudnnOf(ctx_);
  if (!handle) {
    fprintf(stderr, "[GpuSoftmaxOp] missing cudnn handle\n");
    return;
  }
  JOY_CUDNN_CHECK(cudnnSetStream(handle, streamOf(ctx_)));

  cudnnTensorDescriptor_t desc = nullptr;
  JOY_CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));
  JOY_CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW,
                                             cudnnDtype(x.element_type),
                                             (int)outer, (int)last, 1, 1));

  // alpha/beta need to match compute precision: f32 for FP16/FP32, f64 for F64.
  if (x.element_type == kF64) {
    double alpha = 1.0, beta = 0.0;
    JOY_CUDNN_CHECK(cudnnSoftmaxForward(
        handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha,
        desc, x.data, &beta, desc, y.data));
  } else {
    float alpha = 1.0f, beta = 0.0f;
    JOY_CUDNN_CHECK(cudnnSoftmaxForward(
        handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha,
        desc, x.data, &beta, desc, y.data));
  }

  cudnnDestroyTensorDescriptor(desc);
}

// =====================================================================
// SiLU
// =====================================================================
void GpuSiLUOp::compute() {
  if (numInputs_ < 1 || numOutputs_ < 1) return;
  MemrefDesc &x = inputs_[0];
  MemrefDesc &y = outputs_[0];
  int64_t n = x.numElements();
  cudaStream_t s = streamOf(ctx_);
  if (x.element_type == kF32) {
    joy_kernel_silu_f32(reinterpret_cast<const float *>(x.data),
                        reinterpret_cast<float *>(y.data), n, s);
  } else if (x.element_type == kF16) {
    joy_kernel_silu_f16(x.data, y.data, n, s);
  } else {
    fprintf(stderr, "[GpuSiLUOp] unsupported dtype %d\n", x.element_type);
  }
}

// =====================================================================
// Add
// =====================================================================
void GpuAddOp::compute() {
  if (numInputs_ < 2 || numOutputs_ < 1) return;
  MemrefDesc &a = inputs_[0];
  MemrefDesc &b = inputs_[1];
  MemrefDesc &c = outputs_[0];
  int64_t n = a.numElements();
  cudaStream_t s = streamOf(ctx_);
  if (a.element_type == kF32) {
    joy_kernel_add_f32(reinterpret_cast<const float *>(a.data),
                       reinterpret_cast<const float *>(b.data),
                       reinterpret_cast<float *>(c.data), n, s);
  } else if (a.element_type == kF16) {
    joy_kernel_add_f16(a.data, b.data, c.data, n, s);
  } else {
    fprintf(stderr, "[GpuAddOp] unsupported dtype %d\n", a.element_type);
  }
}

// =====================================================================
// Mul
// =====================================================================
void GpuMulOp::compute() {
  if (numInputs_ < 2 || numOutputs_ < 1) return;
  MemrefDesc &a = inputs_[0];
  MemrefDesc &b = inputs_[1];
  MemrefDesc &c = outputs_[0];
  int64_t n = a.numElements();
  cudaStream_t s = streamOf(ctx_);
  if (a.element_type == kF32) {
    joy_kernel_mul_f32(reinterpret_cast<const float *>(a.data),
                       reinterpret_cast<const float *>(b.data),
                       reinterpret_cast<float *>(c.data), n, s);
  } else if (a.element_type == kF16) {
    joy_kernel_mul_f16(a.data, b.data, c.data, n, s);
  } else {
    fprintf(stderr, "[GpuMulOp] unsupported dtype %d\n", a.element_type);
  }
}

// =====================================================================
// Reshape: it's just a memory view.  We support the form where an output
// buffer is provided (we async-copy) as well as the in-place form (no-op).
// =====================================================================
void GpuReshapeOp::compute() {
  if (numInputs_ < 1 || numOutputs_ < 1) return;
  MemrefDesc &x = inputs_[0];
  MemrefDesc &y = outputs_[0];
  if (x.data == y.data) return; // truly in-place
  size_t bytes = x.sizeInBytes();
  if (bytes == 0) return;
  JOY_CUDA_CHECK(cudaMemcpyAsync(y.data, x.data, bytes,
                                 cudaMemcpyDeviceToDevice, streamOf(ctx_)));
}

// =====================================================================
// Transpose with arbitrary permutation
// =====================================================================
void GpuTransposeOp::compute() {
  if (numInputs_ < 1 || numOutputs_ < 1) return;
  MemrefDesc &x = inputs_[0];
  MemrefDesc &y = outputs_[0];
  int64_t rank = x.rank;
  if ((int64_t)permutation.size() != rank) {
    fprintf(stderr, "[GpuTransposeOp] permutation rank mismatch\n");
    return;
  }
  joy_kernel_transpose(x.data, y.data, x.shape, permutation.data(), rank,
                       (int64_t)x.elementSize(), streamOf(ctx_));
}

// =====================================================================
// Apply rotary embedding
//   inputs: 0 = x [B, H, S, D]
//           1 = cos [S, D]
//           2 = sin [S, D]
//   output: y [B, H, S, D]
// =====================================================================
void GpuApplyRotaryEmbOp::compute() {
  if (numInputs_ < 3 || numOutputs_ < 1) return;
  MemrefDesc &x = inputs_[0];
  MemrefDesc &cosT = inputs_[1];
  MemrefDesc &sinT = inputs_[2];
  MemrefDesc &y = outputs_[0];

  if (x.rank != 4) {
    fprintf(stderr, "[GpuApplyRotaryEmbOp] expected rank-4 x, got %ld\n",
            (long)x.rank);
    return;
  }
  int64_t B = x.shape[0];
  int64_t H = x.shape[1];
  int64_t S = x.shape[2];
  int64_t D = x.shape[3];

  cudaStream_t s = streamOf(ctx_);
  if (x.element_type == kF32) {
    joy_kernel_apply_rotary_emb_f32(
        reinterpret_cast<const float *>(x.data),
        reinterpret_cast<const float *>(cosT.data),
        reinterpret_cast<const float *>(sinT.data),
        reinterpret_cast<float *>(y.data), B, H, S, D, s);
  } else if (x.element_type == kF16) {
    joy_kernel_apply_rotary_emb_f16(x.data, cosT.data, sinT.data, y.data, B, H,
                                    S, D, s);
  } else {
    fprintf(stderr, "[GpuApplyRotaryEmbOp] unsupported dtype %d\n",
            x.element_type);
  }
}

// =====================================================================
// Repeat KV: [B, H_kv, S, D] -> [B, H_kv * n_rep, S, D]
// =====================================================================
void GpuRepeatKVOp::compute() {
  if (numInputs_ < 1 || numOutputs_ < 1) return;
  MemrefDesc &x = inputs_[0];
  MemrefDesc &y = outputs_[0];
  if (x.rank != 4) {
    fprintf(stderr, "[GpuRepeatKVOp] expected rank-4 x, got %ld\n",
            (long)x.rank);
    return;
  }
  int64_t B = x.shape[0];
  int64_t H_kv = x.shape[1];
  int64_t S = x.shape[2];
  int64_t D = x.shape[3];
  cudaStream_t s = streamOf(ctx_);
  if (x.element_type == kF32) {
    joy_kernel_repeat_kv_f32(reinterpret_cast<const float *>(x.data),
                             reinterpret_cast<float *>(y.data), B, H_kv, S, D,
                             n_rep, s);
  } else if (x.element_type == kF16) {
    joy_kernel_repeat_kv_f16(x.data, y.data, B, H_kv, S, D, n_rep, s);
  } else {
    fprintf(stderr, "[GpuRepeatKVOp] unsupported dtype %d\n", x.element_type);
  }
}

// =====================================================================
// Fused (x + residual) -> RMSNorm.
//   inputs:  0=x   1=residual   2=weight
//   outputs: 0=normed_y         (1=add_out, optional)
// =====================================================================
void GpuFuseAddRMSNormOp::compute() {
  if (numInputs_ < 3 || numOutputs_ < 1) return;
  MemrefDesc &x = inputs_[0];
  MemrefDesc &res = inputs_[1];
  MemrefDesc &w = inputs_[2];
  MemrefDesc &y = outputs_[0];
  float *add_out_ptr =
      (numOutputs_ >= 2) ? reinterpret_cast<float *>(outputs_[1].data)
                         : nullptr;

  int64_t outer, h;
  rowsAndLast(x, outer, h);
  cudaStream_t s = streamOf(ctx_);

  if (x.element_type == kF32) {
    // The codegen kernel writes the residual sum into add_out
    // unconditionally, but accepts a separate norm_out for the final
    // value.  When the caller hasn't allocated an add_out buffer, we
    // share the norm_out buffer so the temporary gets a valid place
    // to land (it's overwritten with the normalized value by the
    // second loop, matching the original hand-written semantics).
    float *norm_ptr = reinterpret_cast<float *>(y.data);
    float *add_ptr = add_out_ptr ? add_out_ptr : norm_ptr;
    joy_codegen_fuse_add_rms_norm_f32(
        reinterpret_cast<const float *>(x.data),
        reinterpret_cast<const float *>(res.data),
        reinterpret_cast<const float *>(w.data),
        add_ptr, norm_ptr, outer, h, epsilon, s);
  } else {
    fprintf(stderr, "[GpuFuseAddRMSNormOp] only f32 supported in this build\n");
  }
}

} // namespace gpu
} // namespace joy
