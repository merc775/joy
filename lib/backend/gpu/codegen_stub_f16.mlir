// Stub MLIR for joy-emit-cuda: triggers CodegenRMSNormPass to materialise
// @joy_rms_norm_kernel / @joy_fuse_add_rmsnorm_kernel funcs in f16.

module {
  func.func @stub() {
    %x  = memref.alloc() : memref<4x16xf16>
    %s  = memref.alloc() : memref<16xf16>
    %y  = memref.alloc() : memref<4x16xf16>
    "joyl.rms_norm"(%x, %s, %y) {epsilon = 1.000000e-06 : f32}
        : (memref<4x16xf16>, memref<16xf16>, memref<4x16xf16>) -> ()

    %x2  = memref.alloc() : memref<4x16xf16>
    %r2  = memref.alloc() : memref<4x16xf16>
    %s2  = memref.alloc() : memref<16xf16>
    %ao  = memref.alloc() : memref<4x16xf16>
    %no  = memref.alloc() : memref<4x16xf16>
    "joyl.fuse_add_rmsnorm"(%x2, %r2, %s2, %ao, %no)
        {epsilon = 1.000000e-06 : f32}
        : (memref<4x16xf16>, memref<4x16xf16>, memref<16xf16>,
           memref<4x16xf16>, memref<4x16xf16>) -> ()
    return
  }
}
