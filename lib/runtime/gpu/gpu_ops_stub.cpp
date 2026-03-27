//===- gpu_ops_stub.cpp - Print-only stubs (fallback when CUDA disabled) -===//
//
// Built only when the CMake option JOY_ENABLE_CUDA is OFF.  Mirrors the
// original behaviour (just logs the operator name) so the rest of the
// compiler/runtime can still link.
//
//===----------------------------------------------------------------------===//

#include "gpu_runner.h"
#include <cstdio>

namespace joy {
namespace gpu {

void GpuEmbeddingOp::compute()        { fprintf(stderr, "[GPU stub] embedding\n"); }
void GpuRMSNormOp::compute()          { fprintf(stderr, "[GPU stub] rms_norm(eps=%.2e)\n", epsilon); }
void GpuLinearOp::compute()           { fprintf(stderr, "[GPU stub] linear\n"); }
void GpuMatMulOp::compute()           { fprintf(stderr, "[GPU stub] matmul\n"); }
void GpuSoftmaxOp::compute()          { fprintf(stderr, "[GPU stub] softmax(axis=%ld)\n", (long)axis); }
void GpuSiLUOp::compute()             { fprintf(stderr, "[GPU stub] silu\n"); }
void GpuAddOp::compute()              { fprintf(stderr, "[GPU stub] add\n"); }
void GpuMulOp::compute()              { fprintf(stderr, "[GPU stub] mul\n"); }
void GpuReshapeOp::compute()          { fprintf(stderr, "[GPU stub] reshape\n"); }
void GpuTransposeOp::compute()        { fprintf(stderr, "[GPU stub] transpose\n"); }
void GpuApplyRotaryEmbOp::compute()   { fprintf(stderr, "[GPU stub] apply_rotary_emb\n"); }
void GpuRepeatKVOp::compute()         { fprintf(stderr, "[GPU stub] repeat_kv(n_rep=%ld)\n", (long)n_rep); }
void GpuFuseAddRMSNormOp::compute()   { fprintf(stderr, "[GPU stub] fuse_add_rmsnorm(eps=%.2e)\n", epsilon); }

} // namespace gpu
} // namespace joy
