//===- gpu_runner.h - GPU backend runtime interface -----------*- C++ -*-===//
//
// Joy Compiler - GPU backend operator interface
//
// This header defines the runtime types and base class for GPU operator
// dispatch.  Each joyh.custom_call maps to an extern "C" entry point
// (e.g. joy_gpu_embedding) that constructs an op object and calls compute().
//
//===----------------------------------------------------------------------===//

#ifndef JOY_BACKEND_GPU_GPU_RUNNER_H
#define JOY_BACKEND_GPU_GPU_RUNNER_H

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace joy {
namespace gpu {

/// Runtime descriptor for a single memref operand.
/// Runtime descriptor for a memref operand — carries rank, data pointer, shape, and dtype.
struct MemrefDesc {
  int64_t rank;
  void *data;
  const int64_t *shape;
  int32_t element_type; // 0=f16, 1=f32, 2=f64, 3=i32, 4=i64

  int64_t numElements() const {
    int64_t n = 1;
    for (int64_t i = 0; i < rank; ++i)
      n *= shape[i];
    return n;
  }

  size_t elementSize() const {
    switch (element_type) {
    case 0: return 2;  // f16
    case 1: return 4;  // f32
    case 2: return 8;  // f64
    case 3: return 4;  // i32
    case 4: return 8;  // i64
    default: return 0;
    }
  }

  size_t sizeInBytes() const { return numElements() * elementSize(); }

  std::string shapeStr() const {
    std::string s = "[";
    for (int64_t i = 0; i < rank; ++i) {
      if (i > 0) s += "x";
      s += std::to_string(shape[i]);
    }
    s += "]";
    return s;
  }
};

/// GPU execution context (stream, handles, etc.).
/// In a real implementation this would wrap cudaStream_t, cublasHandle_t,
/// cudnnHandle_t, etc.
struct GpuContext {
  void *stream = nullptr;     // cudaStream_t
  void *cublas = nullptr;     // cublasHandle_t
  void *cudnn = nullptr;      // cudnnHandle_t

  static GpuContext &getDefault() {
    static GpuContext ctx;
    return ctx;
  }
};

/// Base class for GPU operator implementations.
/// Base class for GPU operator implementations.
///
/// Subclasses override compute() to implement the actual GPU kernel.
/// The runtime entry point constructs the op with the appropriate
/// inputs/outputs and calls compute().
class GpuOperation {
public:
  GpuOperation(GpuContext *ctx, MemrefDesc *inputs, int64_t numInputs,
               MemrefDesc *outputs, int64_t numOutputs, const char *opName)
      : ctx_(ctx), inputs_(inputs), numInputs_(numInputs),
        outputs_(outputs), numOutputs_(numOutputs), opName_(opName) {}

  virtual ~GpuOperation() = default;

  virtual void compute() = 0;

  GpuContext *getContext() const { return ctx_; }

  MemrefDesc &getInput(int64_t idx) const { return inputs_[idx]; }
  MemrefDesc &getOutput(int64_t idx) const { return outputs_[idx]; }
  int64_t getNumInputs() const { return numInputs_; }
  int64_t getNumOutputs() const { return numOutputs_; }
  const char *getOpName() const { return opName_; }

protected:
  GpuContext *ctx_;
  MemrefDesc *inputs_;
  int64_t numInputs_;
  MemrefDesc *outputs_;
  int64_t numOutputs_;
  const char *opName_;
};

// -----------------------------------------------------------------------
// GPU operator declarations (one per Qwen3 op)
// -----------------------------------------------------------------------

class GpuEmbeddingOp : public GpuOperation {
public:
  using GpuOperation::GpuOperation;
  void compute() override;
};

class GpuRMSNormOp : public GpuOperation {
public:
  using GpuOperation::GpuOperation;
  float epsilon = 1e-6f;
  void compute() override;
};

class GpuLinearOp : public GpuOperation {
public:
  using GpuOperation::GpuOperation;
  void compute() override;
};

class GpuMatMulOp : public GpuOperation {
public:
  using GpuOperation::GpuOperation;
  void compute() override;
};

class GpuSoftmaxOp : public GpuOperation {
public:
  using GpuOperation::GpuOperation;
  int64_t axis = -1;
  void compute() override;
};

class GpuSiLUOp : public GpuOperation {
public:
  using GpuOperation::GpuOperation;
  void compute() override;
};

class GpuAddOp : public GpuOperation {
public:
  using GpuOperation::GpuOperation;
  void compute() override;
};

class GpuMulOp : public GpuOperation {
public:
  using GpuOperation::GpuOperation;
  void compute() override;
};

class GpuReshapeOp : public GpuOperation {
public:
  using GpuOperation::GpuOperation;
  void compute() override;
};

class GpuTransposeOp : public GpuOperation {
public:
  using GpuOperation::GpuOperation;
  std::vector<int64_t> permutation;
  void compute() override;
};

class GpuApplyRotaryEmbOp : public GpuOperation {
public:
  using GpuOperation::GpuOperation;
  void compute() override;
};

class GpuRepeatKVOp : public GpuOperation {
public:
  using GpuOperation::GpuOperation;
  int64_t n_rep = 1;
  void compute() override;
};

class GpuFuseAddRMSNormOp : public GpuOperation {
public:
  using GpuOperation::GpuOperation;
  float epsilon = 1e-6f;
  void compute() override;
};

} // namespace gpu
} // namespace joy

#endif // JOY_BACKEND_GPU_GPU_RUNNER_H
