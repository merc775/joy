#include "gpu_runner.h"
#include <cstdio>

using namespace joy::gpu;

extern "C" {

void joy_gpu_embedding(GpuContext *ctx, int64_t numInputs, int64_t numOutputs, MemrefDesc *operands) {
  GpuEmbeddingOp op(ctx, operands, numInputs, operands + numInputs, numOutputs, "embedding");
  op.compute();
}

void joy_gpu_rms_norm(GpuContext *ctx, int64_t numInputs, int64_t numOutputs, MemrefDesc *operands, float epsilon) {
  GpuRMSNormOp op(ctx, operands, numInputs, operands + numInputs, numOutputs, "rms_norm");
  op.epsilon = epsilon;
  op.compute();
}

void joy_gpu_linear(GpuContext *ctx, int64_t numInputs, int64_t numOutputs, MemrefDesc *operands) {
  GpuLinearOp op(ctx, operands, numInputs, operands + numInputs, numOutputs, "linear");
  op.compute();
}

void joy_gpu_matmul(GpuContext *ctx, int64_t numInputs, int64_t numOutputs, MemrefDesc *operands) {
  GpuMatMulOp op(ctx, operands, numInputs, operands + numInputs, numOutputs, "matmul");
  op.compute();
}

void joy_gpu_softmax(GpuContext *ctx, int64_t numInputs, int64_t numOutputs, MemrefDesc *operands, int64_t axis) {
  GpuSoftmaxOp op(ctx, operands, numInputs, operands + numInputs, numOutputs, "softmax");
  op.axis = axis;
  op.compute();
}

void joy_gpu_silu(GpuContext *ctx, int64_t numInputs, int64_t numOutputs, MemrefDesc *operands) {
  GpuSiLUOp op(ctx, operands, numInputs, operands + numInputs, numOutputs, "silu");
  op.compute();
}

void joy_gpu_add(GpuContext *ctx, int64_t numInputs, int64_t numOutputs, MemrefDesc *operands) {
  GpuAddOp op(ctx, operands, numInputs, operands + numInputs, numOutputs, "add");
  op.compute();
}

void joy_gpu_mul(GpuContext *ctx, int64_t numInputs, int64_t numOutputs, MemrefDesc *operands) {
  GpuMulOp op(ctx, operands, numInputs, operands + numInputs, numOutputs, "mul");
  op.compute();
}

void joy_gpu_reshape(GpuContext *ctx, int64_t numInputs, int64_t numOutputs, MemrefDesc *operands) {
  GpuReshapeOp op(ctx, operands, numInputs, operands + numInputs, numOutputs, "reshape");
  op.compute();
}

void joy_gpu_transpose(GpuContext *ctx, int64_t numInputs, int64_t numOutputs, MemrefDesc *operands, const int64_t *perm, int64_t permLen) {
  GpuTransposeOp op(ctx, operands, numInputs, operands + numInputs, numOutputs, "transpose");
  op.permutation.assign(perm, perm + permLen);
  op.compute();
}

void joy_gpu_apply_rotary_emb(GpuContext *ctx, int64_t numInputs, int64_t numOutputs, MemrefDesc *operands) {
  GpuApplyRotaryEmbOp op(ctx, operands, numInputs, operands + numInputs, numOutputs, "apply_rotary_emb");
  op.compute();
}

void joy_gpu_repeat_kv(GpuContext *ctx, int64_t numInputs, int64_t numOutputs, MemrefDesc *operands, int64_t n_rep) {
  GpuRepeatKVOp op(ctx, operands, numInputs, operands + numInputs, numOutputs, "repeat_kv");
  op.n_rep = n_rep;
  op.compute();
}

void joy_gpu_fuse_add_rmsnorm(GpuContext *ctx, int64_t numInputs, int64_t numOutputs, MemrefDesc *operands, float epsilon) {
  GpuFuseAddRMSNormOp op(ctx, operands, numInputs, operands + numInputs, numOutputs, "fuse_add_rmsnorm");
  op.epsilon = epsilon;
  op.compute();
}

} // extern "C"
