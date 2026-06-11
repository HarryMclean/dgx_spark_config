# DGX Spark HPC-Oriented PyTorch Setup

This document describes the supported HPC-style PyTorch setup for this DGX Spark.

## What "HPC PyTorch" Means Here

For PyTorch on DGX Spark, the supported high-performance path is:

- build PyTorch from source
- use the system CUDA toolkit and runtime libraries already installed on the machine
- target Blackwell explicitly with `TORCH_CUDA_ARCH_LIST=12.0;12.1+PTX`
- enable cuDNN, NCCL, cuSPARSELt, cuFile, distributed, Flash Attention, and memory-efficient attention where detected

It does not mean building PyTorch with `nvc` or `nvc++` from the NVIDIA HPC SDK.

## Why NVHPC Compilers Are Not The Build Path

PyTorch's supported Linux build path uses:

- `gcc` or `g++` for host C and C++ compilation
- `nvcc` for CUDA compilation

The NVIDIA HPC SDK can still be relevant as a packaging and scientific-computing toolkit, but it is not the correct primary compiler path for a PyTorch source build.

On this machine, there is also no installed HPC SDK under `/opt/nvidia/hpc_sdk`, so the practical and supported route is the standard CUDA Toolkit plus GCC/G++ path.

## What Was Changed In This Repo

Two repo-local pieces now support this workflow:

- [grace_blackwell_pytorch_autosetup.sh](./grace_blackwell_pytorch_autosetup.sh)
- [spark_pytorch_hpc_setup.sh](./spark_pytorch_hpc_setup.sh)

### `grace_blackwell_pytorch_autosetup.sh`

This script now supports:

```bash
source ./grace_blackwell_pytorch_autosetup.sh --skip-system-installs
```

That mode skips all `sudo` / `apt` / `dpkg` steps and only discovers and exports the build environment from whatever is already installed.

This was validated successfully on this machine.

The helper also now exports values that are required for a real source build on this DGX Spark:

- `PYTORCH_BUILD_NUMBER=1`
- `USE_SYSTEM_LIBS=0`
- `CUDA_HOME=/usr/local/cuda-13.0`
- aarch64 CUDA include and library paths under `targets/sbsa-linux`

### `spark_pytorch_hpc_setup.sh`

This new wrapper script:

1. Creates a dedicated build virtual environment.
2. Sources the autosetup helper in no-root discovery mode.
3. Writes a reusable activation helper.
4. Optionally clones PyTorch source.
5. Optionally installs PyTorch build requirements.
6. Optionally applies the repo's local PyTorch and FlashAttention patches.

It also supports a clean reinstall mode:

```bash
./spark_pytorch_hpc_setup.sh --env-dir ~/mllib-hpc --fresh
```

That removes the selected environment root and recreates it from scratch.

## Quick Start

### Prepare only the build environment

```bash
cd ~/dgx_spark_config
chmod +x ./spark_pytorch_hpc_setup.sh
./spark_pytorch_hpc_setup.sh --env-dir ~/mllib-hpc --skip-clone
```

This creates:

- `~/mllib-hpc/.venv`
- `~/mllib-hpc/activate_hpc_pytorch.sh`

### Prepare the environment and clone PyTorch source

```bash
cd ~/dgx_spark_config
./spark_pytorch_hpc_setup.sh --env-dir ~/mllib-hpc
```

### Do a fresh reinstall

```bash
cd ~/dgx_spark_config
./spark_pytorch_hpc_setup.sh --env-dir ~/mllib-hpc --fresh
```

That deletes the existing `~/mllib-hpc` tree first, then recreates the venv, activation helper, cloned PyTorch source, and Python build requirements.

By default, that will:

- clone `pytorch` into `~/mllib-hpc/pytorch`
- check out `v2.9.1`
- install `requirements.txt` and `requirements-build.txt`
- apply the repo's local patch files when they match the checked-out tree

This workspace was prepared successfully on this machine at:

- `~/mllib-hpc/.venv`
- `~/mllib-hpc/activate_hpc_pytorch.sh`
- `~/mllib-hpc/pytorch`

## How To Use The Environment

Activate it with:

```bash
source ~/mllib-hpc/activate_hpc_pytorch.sh
```

That activation helper does two things:

1. activates the Python virtual environment
2. re-runs the autosetup helper in `--skip-system-installs` mode so the CUDA, cuDNN, NCCL, cuSPARSELt, and cuFile build variables are present in each fresh shell

## Validated Build Environment On This Machine

The no-root detection path was validated and exported:

- `CUDA_HOME=/usr/local/cuda-13.0`
- `CUDA_NVCC_EXECUTABLE=/usr/local/cuda-13.0/bin/nvcc`
- `TORCH_CUDA_ARCH_LIST=12.0;12.1+PTX`
- `USE_CUDA=1`
- `USE_CUDNN=1`
- `USE_CUSPARSELT=1`
- `USE_CUFILE=1`
- `USE_SYSTEM_NCCL=1`
- `USE_DISTRIBUTED=1`

The helper also exports the correct aarch64 CUDA paths, including:

- `.../targets/sbsa-linux/include`
- `.../targets/sbsa-linux/lib`

That matters on DGX Spark because the CUDA toolkit layout is not just `lib64`.

## Example Source Build

After running the setup script and activating the helper:

```bash
source ~/mllib-hpc/activate_hpc_pytorch.sh
cd ~/mllib-hpc/pytorch
python setup.py bdist_wheel > ~/mllib-hpc/torch_build.log 2>&1
```

Using `setup.py bdist_wheel` with the log redirected to a file is the more practical choice on this machine because the full compile produces a very large amount of build output.

To monitor the build while it runs:

```bash
tail -f ~/mllib-hpc/torch_build.log
```

On this machine, the build completed successfully and produced:

- `~/mllib-hpc/pytorch/dist/torch-2.9.1-cp312-cp312-linux_aarch64.whl`

Or install in-place during development:

```bash
source ~/mllib-hpc/activate_hpc_pytorch.sh
cd ~/mllib-hpc/pytorch
python setup.py develop
```

## Benchmarking

The existing prebuilt optimized environment was benchmarked first as a reference point while the source-build path was being prepared.

Command used:

```bash
source ~/mllib/activate_pytorch.sh
cd ~/dgx_spark_config/bench

python - <<'PY'
from bench_gemm import bench_gemm_loop
bench_gemm_loop(M=8192, N=8192, K=8192, target_seconds=10, warmup=10)
PY
```

Observed result on this machine:

- Total time: `22.53 s`
- Total iterations: `1856`
- Average per iteration: `12.14 ms`
- Effective FP16 throughput: `90.58 TFLOPs`

After building the wheel, install it into the HPC environment and benchmark it with the same workload:

```bash
source ~/mllib-hpc/activate_hpc_pytorch.sh
python -m pip install --force-reinstall ~/mllib-hpc/pytorch/dist/torch-2.9.1-cp312-cp312-linux_aarch64.whl

cd ~/dgx_spark_config/bench
python - <<'PY'
from bench_gemm import bench_gemm_loop
bench_gemm_loop(M=8192, N=8192, K=8192, target_seconds=10, warmup=10)
PY
```

Observed result from the completed source-built wheel on this machine:

- Torch: `2.9.1`
- CUDA build: `13.0`
- Total time: `22.14 s`
- Total iterations: `1885`
- Average per iteration: `11.75 ms`
- Effective FP16 throughput: `93.61 TFLOPs`

Compared with the validated prebuilt-wheel baseline in `~/mllib`, that is an uplift of about `3.34%` on this short run.

## When You Still Need Root

Use the original autosetup script without `--skip-system-installs` if the machine is missing any of these system components:

- CUDA toolkit
- cuDNN development headers
- NCCL headers and libraries
- cuSPARSELt packages
- cuFile / GPUDirect Storage packages
- host build tools from `apt`

That full path still requires `sudo`.

## Summary

The supported HPC-oriented PyTorch path for this repo is:

- standard GCC/G++ plus `nvcc`
- Blackwell-specific build flags
- source build with the repo's PyTorch patches
- user-local no-root discovery mode when the system libraries are already installed

That is the closest supported equivalent to an "HPC PyTorch" setup on this DGX Spark.