//===- gpu_kernels.cu - CUDA kernels for the Joy GPU backend --------------===//
//
// Custom CUDA kernels backing operators that don't map cleanly onto cuBLAS or
// cuDNN: element-wise ops, RMSNorm, SiLU, embedding lookup, transpose, RoPE,
// repeat_kv, and a fused add+rmsnorm.  All launchers are exposed as extern "C"
// (see gpu_kernels.h) so the rest of the backend can stay in plain C++.
//
//===----------------------------------------------------------------------===//

#include "gpu_kernels.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>

namespace {

constexpr int kBlockSize = 256;

inline cudaStream_t asStream(void *p) {
  return reinterpret_cast<cudaStream_t>(p);
}

inline int gridFor(int64_t n, int block = kBlockSize) {
  int64_t g = (n + block - 1) / block;
  if (g < 1) g = 1;
  if (g > 65535) g = 65535;
  return static_cast<int>(g);
}

// -------------------- element-wise --------------------

template <typename T>
__global__ void kernel_add(const T *a, const T *b, T *c, int64_t n) {
  int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  int64_t stride = (int64_t)blockDim.x * gridDim.x;
  for (int64_t i = idx; i < n; i += stride) {
    c[i] = a[i] + b[i];
  }
}

template <typename T>
__global__ void kernel_mul(const T *a, const T *b, T *c, int64_t n) {
  int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  int64_t stride = (int64_t)blockDim.x * gridDim.x;
  for (int64_t i = idx; i < n; i += stride) {
    c[i] = a[i] * b[i];
  }
}

__device__ __forceinline__ float silu_f(float x) {
  return x / (1.0f + __expf(-x));
}

__global__ void kernel_silu_f32(const float *x, float *y, int64_t n) {
  int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  int64_t stride = (int64_t)blockDim.x * gridDim.x;
  for (int64_t i = idx; i < n; i += stride) {
    y[i] = silu_f(x[i]);
  }
}

__global__ void kernel_silu_f16(const __half *x, __half *y, int64_t n) {
  int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  int64_t stride = (int64_t)blockDim.x * gridDim.x;
  for (int64_t i = idx; i < n; i += stride) {
    float xv = __half2float(x[i]);
    y[i] = __float2half(silu_f(xv));
  }
}

// -------------------- RMSNorm + FuseAddRMSNorm: removed --------------------
//
// The hand-written rms_norm / fuse_add_rmsnorm kernels that used to live
// here are now generated from MLIR by joy-emit-cuda at build time.  The
// resulting source ends up in ${BUILD}/lib/backend/gpu/codegen_kernel.cu
// and exposes:
//
//   joy_codegen_rms_norm_f32 / _f16
//   joy_codegen_fuse_add_rms_norm_f32
//
// See joy/lib/optimizer/EmitCudaC.cpp for the emitter and
// joy/lib/backend/gpu/CMakeLists.txt for the codegen rule.

// -------------------- Embedding lookup --------------------

template <typename TVal, typename TIdx>
__global__ void kernel_embedding(const TVal *table, const TIdx *ids,
                                 TVal *out, int64_t n, int64_t hidden,
                                 int64_t vocab) {
  int64_t row = blockIdx.x;
  if (row >= n) return;
  TIdx idx = ids[row];
  if (idx < 0 || (int64_t)idx >= vocab) {
    // out-of-range -> zero-fill so we don't crash silently
    for (int64_t c = threadIdx.x; c < hidden; c += blockDim.x)
      out[row * hidden + c] = TVal(0);
    return;
  }
  const TVal *src = table + (int64_t)idx * hidden;
  TVal *dst = out + row * hidden;
  for (int64_t c = threadIdx.x; c < hidden; c += blockDim.x) {
    dst[c] = src[c];
  }
}

// -------------------- Transpose (rank up to 8) --------------------

constexpr int kMaxRank = 8;

template <typename T>
__global__ void kernel_transpose(const T *src, T *dst, int64_t n,
                                 int rank,
                                 const int64_t *__restrict__ src_strides,
                                 const int64_t *__restrict__ dst_shape,
                                 const int *__restrict__ perm) {
  int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  int64_t stride = (int64_t)blockDim.x * gridDim.x;
  for (int64_t i = idx; i < n; i += stride) {
    int64_t coords[kMaxRank];
    int64_t rem = i;
    for (int r = rank - 1; r >= 0; --r) {
      int64_t s = dst_shape[r];
      coords[r] = rem % s;
      rem /= s;
    }
    int64_t src_off = 0;
    for (int r = 0; r < rank; ++r) {
      src_off += coords[r] * src_strides[perm[r]];
    }
    dst[i] = src[src_off];
  }
}

// -------------------- Repeat KV --------------------

template <typename T>
__global__ void kernel_repeat_kv(const T *src, T *dst, int64_t b,
                                 int64_t h_kv, int64_t s, int64_t d,
                                 int64_t n_rep) {
  int64_t total = b * h_kv * n_rep * s * d;
  int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  int64_t stride = (int64_t)blockDim.x * gridDim.x;
  int64_t H = h_kv * n_rep;
  for (int64_t i = idx; i < total; i += stride) {
    int64_t di = i % d;
    int64_t si = (i / d) % s;
    int64_t hi = (i / (d * s)) % H;
    int64_t bi = i / (d * s * H);
    int64_t src_h = hi / n_rep;
    int64_t src_off = ((bi * h_kv + src_h) * s + si) * d + di;
    dst[i] = src[src_off];
  }
}

// -------------------- Rotary embedding --------------------
//
// Layout convention used here matches Hugging Face's rotate_half:
//   x split along last dim into [x1, x2] of size D/2 each.
//   rotated = concat(-x2, x1)
//   y = x * cos + rotated * sin
// cos/sin shape: [S, D] (the same value broadcast across batch and heads).

template <typename T>
__device__ __forceinline__ float toF(T v) { return (float)v; }
template <>
__device__ __forceinline__ float toF<__half>(__half v) {
  return __half2float(v);
}

template <typename T>
__device__ __forceinline__ T fromF(float v) { return (T)v; }
template <>
__device__ __forceinline__ __half fromF<__half>(float v) {
  return __float2half(v);
}

template <typename T>
__global__ void kernel_apply_rotary_emb(const T *x, const T *cosTab,
                                        const T *sinTab, T *y, int64_t b,
                                        int64_t h, int64_t s, int64_t d) {
  int64_t total = b * h * s * d;
  int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  int64_t stride = (int64_t)blockDim.x * gridDim.x;
  int64_t half = d / 2;

  for (int64_t i = idx; i < total; i += stride) {
    int64_t di = i % d;
    int64_t si = (i / d) % s;
    // pair element index along last dim
    int64_t pair_off;
    float sign;
    if (di < half) {
      pair_off = i + half;        // -x2
      sign = -1.0f;
    } else {
      pair_off = i - half;        //  x1
      sign = 1.0f;
    }
    float xv = toF<T>(x[i]);
    float xpair = toF<T>(x[pair_off]);
    float c = toF<T>(cosTab[si * d + di]);
    float sv = toF<T>(sinTab[si * d + di]);
    float out = xv * c + sign * xpair * sv;
    y[i] = fromF<T>(out);
  }
}

} // namespace

// =============================================================
// extern "C" launchers
// =============================================================

#define LAUNCH_BIN(name, T)                                                    \
  extern "C" void joy_kernel_##name##_f32(                                     \
      const float *a, const float *b, float *c, int64_t n, void *stream) {     \
    if (n <= 0) return;                                                        \
    kernel_##name<float>                                                       \
        <<<gridFor(n), kBlockSize, 0, asStream(stream)>>>(a, b, c, n);         \
  }

LAUNCH_BIN(add, float)
LAUNCH_BIN(mul, float)

extern "C" void joy_kernel_add_f16(const void *a, const void *b, void *c,
                                   int64_t n, void *stream) {
  if (n <= 0) return;
  kernel_add<__half><<<gridFor(n), kBlockSize, 0, asStream(stream)>>>(
      reinterpret_cast<const __half *>(a), reinterpret_cast<const __half *>(b),
      reinterpret_cast<__half *>(c), n);
}
extern "C" void joy_kernel_mul_f16(const void *a, const void *b, void *c,
                                   int64_t n, void *stream) {
  if (n <= 0) return;
  kernel_mul<__half><<<gridFor(n), kBlockSize, 0, asStream(stream)>>>(
      reinterpret_cast<const __half *>(a), reinterpret_cast<const __half *>(b),
      reinterpret_cast<__half *>(c), n);
}

extern "C" void joy_kernel_silu_f32(const float *x, float *y, int64_t n,
                                    void *stream) {
  if (n <= 0) return;
  kernel_silu_f32<<<gridFor(n), kBlockSize, 0, asStream(stream)>>>(x, y, n);
}
extern "C" void joy_kernel_silu_f16(const void *x, void *y, int64_t n,
                                    void *stream) {
  if (n <= 0) return;
  kernel_silu_f16<<<gridFor(n), kBlockSize, 0, asStream(stream)>>>(
      reinterpret_cast<const __half *>(x), reinterpret_cast<__half *>(y), n);
}

// joy_kernel_rms_norm_* / joy_kernel_fuse_add_rms_norm_f32 launchers
// removed: those entry points now live in codegen_kernel.cu under the
// joy_codegen_rms_norm_* / joy_codegen_fuse_add_rms_norm_f32 names.

extern "C" void joy_kernel_embedding_f32_i32(const float *table,
                                             const int32_t *ids, float *out,
                                             int64_t n, int64_t hidden,
                                             int64_t vocab, void *stream) {
  if (n <= 0 || hidden <= 0) return;
  int block = (hidden < 256) ? (int)hidden : 256;
  if (block < 32) block = 32;
  kernel_embedding<float, int32_t>
      <<<(int)n, block, 0, asStream(stream)>>>(table, ids, out, n, hidden,
                                               vocab);
}
extern "C" void joy_kernel_embedding_f32_i64(const float *table,
                                             const int64_t *ids, float *out,
                                             int64_t n, int64_t hidden,
                                             int64_t vocab, void *stream) {
  if (n <= 0 || hidden <= 0) return;
  int block = (hidden < 256) ? (int)hidden : 256;
  if (block < 32) block = 32;
  kernel_embedding<float, int64_t>
      <<<(int)n, block, 0, asStream(stream)>>>(table, ids, out, n, hidden,
                                               vocab);
}
extern "C" void joy_kernel_embedding_f16_i64(const void *table,
                                             const int64_t *ids, void *out,
                                             int64_t n, int64_t hidden,
                                             int64_t vocab, void *stream) {
  if (n <= 0 || hidden <= 0) return;
  int block = (hidden < 256) ? (int)hidden : 256;
  if (block < 32) block = 32;
  kernel_embedding<__half, int64_t>
      <<<(int)n, block, 0, asStream(stream)>>>(
          reinterpret_cast<const __half *>(table), ids,
          reinterpret_cast<__half *>(out), n, hidden, vocab);
}

extern "C" void joy_kernel_transpose(const void *src, void *dst,
                                     const int64_t *src_shape,
                                     const int64_t *perm, int64_t rank,
                                     int64_t element_size, void *stream) {
  if (rank <= 0 || rank > kMaxRank) {
    fprintf(stderr, "[joy_kernel_transpose] unsupported rank=%ld\n",
            (long)rank);
    return;
  }

  int64_t src_strides_h[kMaxRank];
  int64_t dst_shape_h[kMaxRank];
  int perm_h[kMaxRank];
  int64_t s = 1;
  for (int r = (int)rank - 1; r >= 0; --r) {
    src_strides_h[r] = s;
    s *= src_shape[r];
  }
  int64_t total = s;
  for (int r = 0; r < rank; ++r) {
    perm_h[r] = (int)perm[r];
    dst_shape_h[r] = src_shape[perm_h[r]];
  }

  // Copy the small index arrays to device.
  int64_t *d_src_strides = nullptr;
  int64_t *d_dst_shape = nullptr;
  int *d_perm = nullptr;
  cudaMallocAsync(&d_src_strides, sizeof(int64_t) * rank, asStream(stream));
  cudaMallocAsync(&d_dst_shape, sizeof(int64_t) * rank, asStream(stream));
  cudaMallocAsync(&d_perm, sizeof(int) * rank, asStream(stream));
  cudaMemcpyAsync(d_src_strides, src_strides_h, sizeof(int64_t) * rank,
                  cudaMemcpyHostToDevice, asStream(stream));
  cudaMemcpyAsync(d_dst_shape, dst_shape_h, sizeof(int64_t) * rank,
                  cudaMemcpyHostToDevice, asStream(stream));
  cudaMemcpyAsync(d_perm, perm_h, sizeof(int) * rank, cudaMemcpyHostToDevice,
                  asStream(stream));

  if (element_size == 4) {
    kernel_transpose<int32_t><<<gridFor(total), kBlockSize, 0,
                                asStream(stream)>>>(
        reinterpret_cast<const int32_t *>(src),
        reinterpret_cast<int32_t *>(dst), total, (int)rank, d_src_strides,
        d_dst_shape, d_perm);
  } else if (element_size == 2) {
    kernel_transpose<int16_t><<<gridFor(total), kBlockSize, 0,
                                asStream(stream)>>>(
        reinterpret_cast<const int16_t *>(src),
        reinterpret_cast<int16_t *>(dst), total, (int)rank, d_src_strides,
        d_dst_shape, d_perm);
  } else if (element_size == 8) {
    kernel_transpose<int64_t><<<gridFor(total), kBlockSize, 0,
                                asStream(stream)>>>(
        reinterpret_cast<const int64_t *>(src),
        reinterpret_cast<int64_t *>(dst), total, (int)rank, d_src_strides,
        d_dst_shape, d_perm);
  } else {
    fprintf(stderr, "[joy_kernel_transpose] unsupported element_size=%ld\n",
            (long)element_size);
  }

  cudaFreeAsync(d_src_strides, asStream(stream));
  cudaFreeAsync(d_dst_shape, asStream(stream));
  cudaFreeAsync(d_perm, asStream(stream));
}

extern "C" void joy_kernel_repeat_kv_f32(const float *src, float *dst,
                                         int64_t b, int64_t h_kv, int64_t s,
                                         int64_t d, int64_t n_rep,
                                         void *stream) {
  int64_t total = b * h_kv * n_rep * s * d;
  if (total <= 0) return;
  kernel_repeat_kv<float>
      <<<gridFor(total), kBlockSize, 0, asStream(stream)>>>(src, dst, b, h_kv,
                                                            s, d, n_rep);
}
extern "C" void joy_kernel_repeat_kv_f16(const void *src, void *dst, int64_t b,
                                         int64_t h_kv, int64_t s, int64_t d,
                                         int64_t n_rep, void *stream) {
  int64_t total = b * h_kv * n_rep * s * d;
  if (total <= 0) return;
  kernel_repeat_kv<__half><<<gridFor(total), kBlockSize, 0,
                             asStream(stream)>>>(
      reinterpret_cast<const __half *>(src), reinterpret_cast<__half *>(dst),
      b, h_kv, s, d, n_rep);
}

extern "C" void joy_kernel_apply_rotary_emb_f32(const float *x,
                                                const float *cosTab,
                                                const float *sinTab, float *y,
                                                int64_t b, int64_t h,
                                                int64_t s, int64_t d,
                                                void *stream) {
  int64_t total = b * h * s * d;
  if (total <= 0) return;
  kernel_apply_rotary_emb<float>
      <<<gridFor(total), kBlockSize, 0, asStream(stream)>>>(x, cosTab, sinTab,
                                                            y, b, h, s, d);
}
extern "C" void joy_kernel_apply_rotary_emb_f16(const void *x, const void *cos,
                                                const void *sin, void *y,
                                                int64_t b, int64_t h,
                                                int64_t s, int64_t d,
                                                void *stream) {
  int64_t total = b * h * s * d;
  if (total <= 0) return;
  kernel_apply_rotary_emb<__half><<<gridFor(total), kBlockSize, 0,
                                    asStream(stream)>>>(
      reinterpret_cast<const __half *>(x),
      reinterpret_cast<const __half *>(cos),
      reinterpret_cast<const __half *>(sin), reinterpret_cast<__half *>(y), b,
      h, s, d);
}
