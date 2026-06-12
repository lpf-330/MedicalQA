# sglang-kernel (prior sgl-kernel)

[Kernel Library](https://github.com/sgl-project/sglang/tree/main/sgl-kernel) for LLM inference engines

<div align="center">

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://github.com/sgl-project/sglang/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/sglang-kernel)](https://pypi.org/project/sglang-kernel)

</div>

`sglang-kernel` provides optimized compute primitives for LLM inference engines, enabling efficient inference for large language models and vision-language models through custom kernel operations. The source tree remains under the `sgl-kernel/` directory and the Python import path remains `sgl_kernel`.

---

## 自定义修改说明（SM75 适配）

本项目使用的 sglang-kernel 是**基于上游源码修改后的自编译版本**，主要修改目标：**让 SGLang 能在 RTX 2080 Ti（Turing, SM75）上运行**。上游版本仅支持 SM80+（Ampere 及以上），在 SM75 上会因缺少编译产物而无法加载。

### 修改内容

所有修改以 `[SM75 PATCH]` 注释标记，核心变更如下：

#### 1. CMakeLists.txt — 编译目标与内核裁剪

| 修改项 | 原始 | 修改后 | 原因 |
|--------|------|--------|------|
| gencode 目标 | sm_80, sm_89, sm_90a, sm_100a 等 | 仅 sm_75 | RTX 2080 Ti 仅需 SM75 |
| BF16 内核 | 启用 | 禁用 | SM75 无 BF16 Tensor Core |
| FP8 内核 | 启用 | 禁用 | FP8 要求 SM90+ |
| NVFP4 内核 | 启用 | 禁用 | FP4 要求 SM100+（Blackwell） |
| FA3 (Flash Attention 3) | 启用 | 禁用 | FA3 要求 SM90+ |
| flashinfer norm | 启用 | 替换为纯 PyTorch 实现 | flashinfer 依赖 CuTe/CUTLASS，仅 SM80+ |
| flash-attention | 启用 | 禁用 | 使用 triton attention 后端替代 |
| rmsnorm | flashinfer 版本 | `csrc/elementwise/rmsnorm_sm75.cu` | 纯 CUDA 实现，兼容 SM75 |
| sm100a/120a/103a/90a gencode | 包含 | 跳过 | 非 SM75 目标，减少编译时间和内存 |

#### 2. python/sgl_kernel/load_utils.py — SM75 运行时加载

原始代码仅识别 SM90 和 SM100 两种架构变体：

```python
# 原始逻辑
if compute_capability == 90:
    ops_subdir = "sm90"
else:
    ops_subdir = "sm100"
```

编译后的 SM75 内核产物放在 `sgl_kernel/sm75/` 目录下，运行时由加载逻辑自动匹配当前 GPU 架构。当前代码中 SM75 不再走 sm100 分支（因为 sm75 目录存在对应 .so 文件），而是由 `_load_architecture_specific_ops()` 的架构目录匹配机制自动加载。

#### 3. python/sgl_kernel/load_utils.py — CUDA 运行时库预加载

增加了 `_preload_cuda_library()` 函数，解决 `libcudart.so.12 not found` 问题：
- 自动搜索 CUDA_HOME 下的 lib/lib64 目录
- 优先加载与当前 CUDA 版本匹配的运行时库
- 兼容 CUDA 12 和 CUDA 13 路径

### 编译环境

| 项 | 值 |
|----|-----|
| GPU | NVIDIA RTX 2080 Ti (SM75) |
| CUDA | 12.9 |
| PyTorch | 2.11.0 |
| Python | 3.11 |
| 编译工具 | scikit-build-core 0.12.2 |
| 产物版本 | sglang-kernel 0.4.3 |

### 版本差异说明

| 项 | 上游 PyPI | 本项目自编译 |
|----|-----------|-------------|
| 版本号 | 0.4.2.post2 | 0.4.3 |
| SM75 支持 | 无 | 有 |
| SM90+ 优化内核 | 完整 | 裁剪（仅保留 SM75） |

sglang 0.5.12.post1 依赖 sglang-kernel==0.4.2.post2，安装时需使用 `--no-deps` 跳过依赖检查。

### 重新编译

如需重新编译（例如升级 SGLang 或修改内核）：

```bash
conda activate MedicalQA
cd build/sgl-kernel

# 方式一：Makefile
make build

# 方式二：pip 安装到当前环境
pip install --no-build-isolation --no-deps .

# 限制编译资源（内存不足时）
make build MAX_JOBS=2 CMAKE_ARGS="-DSGL_KERNEL_COMPILE_THREADS=1"
```

### Docker 镜像中的打包方式

Docker 构建时不重新编译，而是从宿主机 site-packages 打包为 tar，直接解压到容器的 site-packages：

```bash
# 打包（宿主机执行）
SITE="/home/ai_env/miniforge3/envs/MedicalQA/lib/python3.11/site-packages"
tar -cf docker/wheels/sglang_kernel.tar -C "$SITE" sgl_kernel sglang_kernel-0.4.3.dist-info

# Dockerfile 中解压
# COPY docker/wheels/sglang_kernel.tar /tmp/sglang_kernel.tar
# RUN tar -xf /tmp/sglang_kernel.tar -C "${SITE_PKGS}"
```

这样做的好处：
- 避免在容器内重新编译（编译耗时 30+ 分钟且依赖 GPU）
- 确保使用的是当前环境已验证的编译产物
- 其他机器通过 `docker load` 导入镜像即可使用，无需编译

---

## Installation
Requires torch == 2.11.0

```bash
# Latest version
pip3 install sglang-kernel --upgrade
```

## Building from Source
Requires
- CMake ≥3.31,
- Python ≥3.10
- scikit-build-core
- ninja(optional)

### Use Makefile to build from the sgl-kernel source tree

```bash
make build
```

### Limit build resource usage (CPU / parallelism)

By default, `make build` uses all available CPU cores. You can override build parallelism and NVCC compile threads:

```bash
# Limit parallel jobs (controls both make and cmake parallelism)
make build MAX_JOBS=2

# Additionally limit NVCC internal threads (reduces CPU and peak memory)
make build MAX_JOBS=2 CMAKE_ARGS="-DSGL_KERNEL_COMPILE_THREADS=1"
```

## Contribution

### Steps to add a new kernel:

1. Implement the kernel in [csrc](https://github.com/sgl-project/sglang/tree/main/sgl-kernel/csrc)
2. Expose the interface in [include/sgl_kernel_ops.h](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/include/sgl_kernel_ops.h)
3. Create torch extension in [csrc/common_extension.cc](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/csrc/common_extension.cc)
4. Update [CMakeLists.txt](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/CMakeLists.txt) to include new CUDA source
5. Expose Python interface in [python](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/python/sgl_kernel)
6. Add test and benchmark

### Development Tips

1. When creating torch extensions, add the function definition with `m.def`, and device binding with `m.impl`:

- How to write schema: [Schema reference](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#func)

   ```cpp
   // We need def with schema here for torch.compile
   m.def(
    "bmm_fp8(Tensor A, Tensor B, Tensor! D, Tensor A_scale, Tensor B_scale, Tensor workspace_buffer, "
    "int cublas_handle) -> ()");
   m.impl("bmm_fp8", torch::kCUDA, &bmm_fp8);
   ```

### Adapting C++ Native Types for Torch Compatibility

Third-party C++ libraries often use int and float, but PyTorch bindings require int64_t and double due to Python's type mapping.

Use make_pytorch_shim from sgl_kernel_torch_shim.h to handle conversions automatically:

```cpp

// Add type conversion for int -> int64_t
template <>
struct pytorch_library_compatible_type<int> {
  using type = int64_t;
  static int convert_from_type(int64_t arg) {
    TORCH_CHECK(arg <= std::numeric_limits<int>::max(), "value too large");
    TORCH_CHECK(arg >= std::numeric_limits<int>::min(), "value too small");
    return arg;
  }
};
```
```cpp
// Wrap your function
m.impl("fwd", torch::kCUDA, make_pytorch_shim(&mha_fwd));
```

### Testing & Benchmarking

1. Add pytest tests in [tests/](https://github.com/sgl-project/sglang/tree/main/sgl-kernel/tests), if you need to skip some test, please use `@pytest.mark.skipif`

```python
@pytest.mark.skipif(
    skip_condition, reason="Nvfp4 Requires compute capability of 10 or above."
)
```

2. Add benchmarks using [triton benchmark](https://triton-lang.org/main/python-api/generated/triton.testing.Benchmark.html) in [benchmark/](https://github.com/sgl-project/sglang/tree/main/sgl-kernel/benchmark)

   **We recommend using `triton.testing.do_bench_cudagraph` for kernel benchmarking**:

   Compared to `triton.testing.do_bench`, `do_bench_cudagraph` provides:
   - Reduced CPU overhead impact for more accurate kernel performance measurements
   - Incorporation of PDL (Programmatic Dependent Launch) effects into individual kernel results
   - More realistic performance data on PDL-supported architectures (SM >= 90)

3. Run test suite

## Kernel Size Analysis

Analyze CUDA kernel sizes in compiled wheel files to identify oversized kernels and template-instantiation bloat:

This tool requires `cubloaty` (install with `pip install cubloaty`) to work.

```bash
# Install cubloaty
pip install cubloaty

# Analyze a wheel file
python analyze_whl_kernel_sizes.py path/to/sglang_kernel-*.whl

# Custom output file
python analyze_whl_kernel_sizes.py path/to/sglang_kernel-*.whl --output my_analysis.txt
```

The tool generates:
- A text report with:
  - Kernel groups (by name prefix)
  - Individual kernel sizes (sorted by size)

Use this to identify large kernels and potential template instantiation bloat.

## FAQ
- Q: Segmentation fault with CUDA 12.6
- A: Update ptxas to 12.8, reference: [segment fault error](https://github.com/Dao-AILab/flash-attention/issues/1453)
