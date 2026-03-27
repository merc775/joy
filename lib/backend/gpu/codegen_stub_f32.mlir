// Stub MLIR for joy-emit-cuda: triggers CodegenRMSNormPass to materialise
// @joy_rms_norm_kernel / @joy_fuse_add_rmsnorm_kernel funcs in f32.  The
// `stub` function is otherwise dead -- only the codegen'd kernels matter.
//
// Used by joy/scripts/regen_codegen_kernel.sh.

module {
  func.func @stub() {
    %x  = memref.alloc() : memref<4x16xf32>
    %s  = memref.alloc() : memref<16xf32>
    %y  = memref.alloc() : memref<4x16xf32>
    "joyl.rms_norm"(%x, %s, %y) {epsilon = 1.000000e-06 : f32}
        : (memref<4x16xf32>, memref<16xf32>, memref<4x16xf32>) -> ()

    %x2  = memref.alloc() : memref<4x16xf32>
    %r2  = memref.alloc() : memref<4x16xf32>
    %s2  = memref.alloc() : memref<16xf32>
    %ao  = memref.alloc() : memref<4x16xf32>
    %no  = memref.alloc() : memref<4x16xf32>
    "joyl.fuse_add_rmsnorm"(%x2, %r2, %s2, %ao, %no)
        {epsilon = 1.000000e-06 : f32}
        : (memref<4x16xf32>, memref<4x16xf32>, memref<16xf32>,
           memref<4x16xf32>, memref<4x16xf32>) -> ()
    return
  }
}
