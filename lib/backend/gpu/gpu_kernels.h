//===- gpu_kernels.h - CUDA kernel launcher prototypes ---------*- C++ -*-===//
//
// Joy Compiler - GPU backend custom CUDA kernel launchers.
//
// Each launcher is a host-callable extern "C" function that internally
// configures the launch geometry and invokes a CUDA kernel.  The signatures
// avoid any CUDA header types in their declaration so that they can be
// called from regular C++ translation units that are not compiled by nvcc.
//
//===----------------------------------------------------------------------===//

#ifndef JOY_BACKEND_GPU_GPU_KERNELS_H
#define JOY_BACKEND_GPU_GPU_KERNELS_H

#include <cstddef>
#include <cstdint>

extern "C" {

// stream is a cudaStream_t (opaque void* here). A nullptr stream means default.

// ---- Element-wise binary ops (same shape; flattened to N elements) ----
void joy_kernel_add_f32(const float *a, const float *b, float *c,
                        int64_t n, void *stream);
void joy_kernel_add_f16(const void *a, const void *b, void *c,
                        int64_t n, void *stream);
void joy_kernel_mul_f32(const float *a, const float *b, float *c,
                        int64_t n, void *stream);
void joy_kernel_mul_f16(const void *a, const void *b, void *c,
                        int64_t n, void *stream);

// ---- SiLU: y = x * sigmoid(x) ----
void joy_kernel_silu_f32(const float *x, float *y, int64_t n, void *stream);
void joy_kernel_silu_f16(const void *x, void *y, int64_t n, void *stream);

// ---- RMSNorm / FuseAddRMSNorm: implemented in codegen_kernel.cu, which is
// generated at build time from MLIR by joy-emit-cuda.  These prototypes
// MUST match the launcher names emitted by joy::emitCudaC (see
// joy/lib/optimizer/EmitCudaC.cpp).
//
// Layout matches the hand-written kernels they replace: one CUDA block
// per row, kBlockSize=256 threads, dynamically-sized shared memory used
// for the reduction.
void joy_codegen_rms_norm_f32(const float *input, const float *scale,
                              float *output, int64_t rows, int64_t cols,
                              float epsilon, void *stream);
void joy_codegen_rms_norm_f16(const void *input, const void *scale,
                              void *output, int64_t rows, int64_t cols,
                              float epsilon, void *stream);

void joy_codegen_fuse_add_rms_norm_f32(const float *lhs, const float *rhs,
                                       const float *scale, float *add_out,
                                       float *norm_out, int64_t rows,
                                       int64_t cols, float epsilon,
                                       void *stream);

// ---- Embedding lookup. ids: [N] int32/int64. table: [V, H]. out: [N, H] ----
void joy_kernel_embedding_f32_i32(const float *table, const int32_t *ids,
                                  float *out, int64_t n, int64_t hidden,
                                  int64_t vocab, void *stream);
void joy_kernel_embedding_f32_i64(const float *table, const int64_t *ids,
                                  float *out, int64_t n, int64_t hidden,
                                  int64_t vocab, void *stream);
void joy_kernel_embedding_f16_i64(const void *table, const int64_t *ids,
                                  void *out, int64_t n, int64_t hidden,
                                  int64_t vocab, void *stream);

// ---- Transpose for arbitrary rank up to 8.  Element type-erased copy. ----
void joy_kernel_transpose(const void *src, void *dst, const int64_t *src_shape,
                          const int64_t *perm, int64_t rank,
                          int64_t element_size, void *stream);

// ---- Repeat KV: input [B, H_kv, S, D] -> output [B, H_kv * n_rep, S, D] ----
void joy_kernel_repeat_kv_f32(const float *src, float *dst, int64_t b,
                              int64_t h_kv, int64_t s, int64_t d,
                              int64_t n_rep, void *stream);
void joy_kernel_repeat_kv_f16(const void *src, void *dst, int64_t b,
                              int64_t h_kv, int64_t s, int64_t d,
                              int64_t n_rep, void *stream);

// ---- Apply rotary embedding.
//   x:   [B, H, S, D]
//   cos: [S, D]
//   sin: [S, D]
//   y:   [B, H, S, D]
//   For each (b, h, s) row of length D:
//     half = D / 2
//     rotated[d]      = -x[d + half]   (d < half)
//     rotated[d+half] =  x[d]
//     y[d] = x[d] * cos[s, d] + rotated[d] * sin[s, d]
void joy_kernel_apply_rotary_emb_f32(const float *x, const float *cos,
                                     const float *sin, float *y, int64_t b,
                                     int64_t h, int64_t s, int64_t d,
                                     void *stream);
void joy_kernel_apply_rotary_emb_f16(const void *x, const void *cos,
                                     const void *sin, void *y, int64_t b,
                                     int64_t h, int64_t s, int64_t d,
                                     void *stream);

} // extern "C"

#endif // JOY_BACKEND_GPU_GPU_KERNELS_H
