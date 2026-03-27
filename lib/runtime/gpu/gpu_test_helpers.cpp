//===- gpu_test_helpers.cpp - C ABI helpers for ctypes-based unit tests ---===//
//
// These helpers are intended for the Python unit tests under
// joy/tests/python_tests/test_op.  They expose just enough of the CUDA
// runtime, cuBLAS, and cuDNN to let a Python test allocate a GPU buffer,
// move data on/off the device, build a GpuContext, and synchronize.
//
// They are not used by the JOY compiler itself.
//
//===----------------------------------------------------------------------===//

#include "gpu_runner.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#include <cstdio>
#include <cstdlib>

using joy::gpu::GpuContext;

extern "C" {

// ---- Device memory ----
void *joy_test_device_alloc(size_t bytes) {
  void *p = nullptr;
  cudaError_t e = cudaMalloc(&p, bytes);
  if (e != cudaSuccess) {
    fprintf(stderr, "[joy_test] cudaMalloc failed: %s\n",
            cudaGetErrorString(e));
    return nullptr;
  }
  return p;
}

void joy_test_device_free(void *ptr) {
  if (ptr) cudaFree(ptr);
}

int joy_test_memcpy_h2d(void *dst, const void *src, size_t bytes) {
  cudaError_t e = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
  if (e != cudaSuccess) {
    fprintf(stderr, "[joy_test] H2D failed: %s\n", cudaGetErrorString(e));
    return (int)e;
  }
  return 0;
}

int joy_test_memcpy_d2h(void *dst, const void *src, size_t bytes) {
  cudaError_t e = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
  if (e != cudaSuccess) {
    fprintf(stderr, "[joy_test] D2H failed: %s\n", cudaGetErrorString(e));
    return (int)e;
  }
  return 0;
}

int joy_test_memset_zero(void *dst, size_t bytes) {
  cudaError_t e = cudaMemset(dst, 0, bytes);
  return (e == cudaSuccess) ? 0 : (int)e;
}

int joy_test_device_synchronize(void) {
  cudaError_t e = cudaDeviceSynchronize();
  return (e == cudaSuccess) ? 0 : (int)e;
}

// ---- GpuContext (cublas/cudnn handles bound to a stream) ----
GpuContext *joy_test_create_context(void) {
  auto *ctx = new GpuContext();

  cudaStream_t stream = nullptr;
  if (cudaStreamCreate(&stream) != cudaSuccess) {
    fprintf(stderr, "[joy_test] cudaStreamCreate failed\n");
    delete ctx;
    return nullptr;
  }
  ctx->stream = (void *)stream;

  cublasHandle_t cublas = nullptr;
  if (cublasCreate(&cublas) != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "[joy_test] cublasCreate failed\n");
  } else {
    cublasSetStream(cublas, stream);
  }
  ctx->cublas = (void *)cublas;

  cudnnHandle_t cudnn = nullptr;
  if (cudnnCreate(&cudnn) != CUDNN_STATUS_SUCCESS) {
    fprintf(stderr, "[joy_test] cudnnCreate failed\n");
  } else {
    cudnnSetStream(cudnn, stream);
  }
  ctx->cudnn = (void *)cudnn;

  return ctx;
}

void joy_test_destroy_context(GpuContext *ctx) {
  if (!ctx) return;
  if (ctx->cudnn) cudnnDestroy((cudnnHandle_t)ctx->cudnn);
  if (ctx->cublas) cublasDestroy((cublasHandle_t)ctx->cublas);
  if (ctx->stream) cudaStreamDestroy((cudaStream_t)ctx->stream);
  delete ctx;
}

int joy_test_stream_synchronize(GpuContext *ctx) {
  if (!ctx || !ctx->stream) return -1;
  cudaError_t e = cudaStreamSynchronize((cudaStream_t)ctx->stream);
  return (e == cudaSuccess) ? 0 : (int)e;
}

// ---- A small build/identity probe so Python can verify the lib loaded ----
const char *joy_test_runtime_signature(void) {
  return "joy_gpu_runtime[cuda12-cublas-cudnn]";
}

int joy_test_cuda_runtime_version(void) {
  int v = 0;
  cudaRuntimeGetVersion(&v);
  return v;
}

int joy_test_cudnn_version(void) {
  return (int)cudnnGetVersion();
}

} // extern "C"
