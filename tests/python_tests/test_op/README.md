# Joy GPU 后端 单算子单元测试

本目录为 `joy/lib/backend/gpu/` 中每一个 GPU 算子提供独立的数值正确性单元测试。
所有测试通过 `ctypes` 加载由 JOY 工程编译出的共享库
`libjoy_gpu_runtime.so`，调用 `joy_gpu_*` extern "C" 入口点，
并把结果与 NumPy（部分用 PyTorch）参考实现做数值对比。

底层算子库映射：

| 算子                     | 实现入口 (lib/backend/gpu) | 调用的 NVIDIA 库     |
| ------------------------ | -------------------------- | -------------------- |
| `joy_gpu_matmul`         | `GpuMatMulOp`              | **cuBLAS** GemmEx / GemmStridedBatchedEx |
| `joy_gpu_linear`         | `GpuLinearOp`              | **cuBLAS** GemmEx (transB) |
| `joy_gpu_softmax`        | `GpuSoftmaxOp`             | **cuDNN** SoftmaxForward |
| `joy_gpu_rms_norm`       | `GpuRMSNormOp`             | 自定义 CUDA kernel   |
| `joy_gpu_silu`           | `GpuSiLUOp`                | 自定义 CUDA kernel   |
| `joy_gpu_add`            | `GpuAddOp`                 | 自定义 CUDA kernel   |
| `joy_gpu_mul`            | `GpuMulOp`                 | 自定义 CUDA kernel   |
| `joy_gpu_embedding`      | `GpuEmbeddingOp`           | 自定义 CUDA kernel   |
| `joy_gpu_reshape`        | `GpuReshapeOp`             | `cudaMemcpyAsync`    |
| `joy_gpu_transpose`      | `GpuTransposeOp`           | 自定义 CUDA kernel   |
| `joy_gpu_apply_rotary_emb` | `GpuApplyRotaryEmbOp`    | 自定义 CUDA kernel   |
| `joy_gpu_repeat_kv`      | `GpuRepeatKVOp`            | 自定义 CUDA kernel   |
| `joy_gpu_fuse_add_rmsnorm` | `GpuFuseAddRMSNormOp`    | 融合自定义 CUDA kernel |

## 1. 一次性构建（包含 GPU 后端）

```bash
cd /data/workspace/joy
./scripts/build.sh
```

构建产物：

* 静态库 `build/lib/libJOYGpuBackend.a` —— 提供给 JOY 编译器/runtime
* 共享库 `build/lib/libjoy_gpu_runtime.so` —— 提供给本目录下的 Python 测试

如果只想构建 GPU runtime：

```bash
cd /data/workspace/joy/build
cmake --build . --target joy_gpu_runtime
```

如果环境没有 CUDA / cuDNN，可降级为 stub：

```bash
cmake -DJOY_ENABLE_CUDA=OFF ...
```

## 2. 跑所有单算子测试

```bash
cd /data/workspace/joy/tests/python_tests/test_op
python3 run_all.py
```

期望最后一行输出：`13/13 tests passed`。

只跑某几项（按子串匹配模块名）：

```bash
python3 run_all.py matmul softmax linear
```

也可以单独运行某个测试：

```bash
python3 test_matmul.py
python3 test_softmax.py
```

## 3. 自定义共享库路径

若 `libjoy_gpu_runtime.so` 不在默认位置（`joy/build/lib/`），可通过环境变量指定：

```bash
JOY_GPU_RUNTIME=/abs/path/to/libjoy_gpu_runtime.so python3 run_all.py
```

## 4. 测试结构（每个 test_*.py）

```
1. 用 numpy 生成确定性随机输入（固定 seed）
2. 通过 _runtime.JoyGpuRuntime 上传到 GPU
3. 创建 GpuContext（含 cuBLAS / cuDNN handle 与 stream）
4. 调用对应的 joy_gpu_* C ABI 入口
5. 同步 stream，把结果拷贝回 host
6. 与 numpy/torch 参考结果对比，atol/rtol 内则 PASS
```

`_runtime.py` 封装了所有 ctypes 细节、`MemrefDesc` / `GpuContext` 结构体、
GPU 内存分配与拷贝，让每个测试文件可以保持精简。
