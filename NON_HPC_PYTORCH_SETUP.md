# DGX Spark Non-HPC PyTorch Setup

This document records the user-space PyTorch setup that was validated on this DGX Spark machine for non-HPC use.

## Goal

The goal was to get an optimized PyTorch environment working on an NVIDIA DGX Spark without taking the full source-build path from the repository README.

This machine already had the core NVIDIA runtime pieces needed for GPU execution, so the faster and lower-risk path was:

1. Keep everything in a user-local virtual environment.
2. Use the repository's prebuilt release wheels.
3. Add the missing NVSHMEM runtime required by the released torch wheel.
4. Validate the result with a real CUDA import, FP16 matmul, and GEMM benchmarks.

## Machine State That Drove The Setup Choice

The following checks were run first:

```bash
uname -m
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader
python3 --version
ldconfig -p | grep -Ei 'libcudnn|libcublas|libcusparseLt|libnccl|libcufile'
```

Observed state:

- Architecture: `aarch64`
- OS: Ubuntu 24.04.4 LTS
- GPU: `NVIDIA GB10`
- Compute capability: `12.1`
- Python: `3.12`
- PyTorch was not installed yet
- These runtime libraries were already available on the system:
  - `libcublas.so`
  - `libcudnn.so`
  - `libcusparseLt.so`
  - `libnccl.so`
  - `libcufile.so`

Because those runtime libraries were already present, a source rebuild was unnecessary for the initial non-HPC setup.

## What Was Added

The setup is driven by this script:

- [spark_pytorch_user_setup.sh](./spark_pytorch_user_setup.sh)

That script does the following:

1. Checks that the machine is `aarch64` and that `nvidia-smi` works.
2. Confirms key runtime libraries are present through `ldconfig`.
3. Creates a virtual environment under `~/mllib/.venv` by default.
4. Queries the latest release from `GuigsEvt/dgx_spark_config` through the GitHub release API.
5. Downloads the optimized `triton` and `torch` wheels from the release.
6. Optionally downloads and installs `torchvision`, `torchaudio`, `onnx`, and `flash_attn`.
7. Installs `nvidia-nvshmem-cu13`, which is required by the released torch wheel on this machine.
8. Writes a persistent activation helper at `~/mllib/activate_pytorch.sh` so the NVSHMEM library path is available in fresh shells.
9. Validates the install with:
   - `import torch`
   - `torch.cuda.is_available()`
   - GPU property lookup
   - one FP16 CUDA matmul

## How The Setup Was Performed

From the cloned repository:

```bash
cd ~/dgx_spark_config
chmod +x ./spark_pytorch_user_setup.sh
./spark_pytorch_user_setup.sh --env-dir ~/mllib
```

What that created:

- Virtual environment: `~/mllib/.venv`
- Wheel cache: `~/mllib/wheels/latest`
- Activation helper: `~/mllib/activate_pytorch.sh`

## How To Use The Environment

Open a new shell and activate the environment with the helper script:

```bash
source ~/mllib/activate_pytorch.sh
```

Do not use the plain `.venv/bin/activate` script unless you also export the NVSHMEM library path yourself. The helper exists specifically to make the environment work from a fresh shell.

After activation, verify the install:

```bash
python - <<'PY'
import torch
print('Torch version:', torch.__version__)
print('Torch CUDA:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
print(torch.cuda.get_device_properties(0))
PY
```

## Installing Optional Extras

If you also want the extra wheels published in the release, rerun the installer with `--with-extras`:

```bash
cd ~/dgx_spark_config
./spark_pytorch_user_setup.sh --env-dir ~/mllib --with-extras
```

This enables installation of release wheels when available for:

- `torchvision`
- `torchaudio`
- `onnx`
- `flash_attn`

## Refreshing Downloaded Wheels

The script reuses wheels already downloaded under `~/mllib/wheels/latest`.

To force a re-download from the latest release:

```bash
cd ~/dgx_spark_config
./spark_pytorch_user_setup.sh --env-dir ~/mllib --force-download
```

## Validation That Was Run

The final validation on this machine included:

```bash
source ~/mllib/activate_pytorch.sh

python - <<'PY'
import torch
print('Torch:', torch.__version__)
print('CUDA:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
props = torch.cuda.get_device_properties(0)
print('Device:', props.name)
print('Compute capability:', f'{props.major}.{props.minor}')
PY
```

Observed result:

- Torch: `2.9.1a0+gitd38164a`
- CUDA build: `13.0`
- CUDA available: `True`
- Device: `NVIDIA GB10`
- Compute capability: `12.1`

A short FP16 GEMM benchmark was also run from the repo benchmark script:

```bash
source ~/mllib/activate_pytorch.sh
cd ~/dgx_spark_config/bench

python - <<'PY'
from bench_gemm import bench_gemm_loop
bench_gemm_loop(M=4096, N=4096, K=4096, target_seconds=5, warmup=10)
PY
```

That completed successfully and produced an effective FP16 throughput of about `85.44 TFLOPs` on the short run.

A larger run was also recorded with the repository benchmark dimensions (`8192 x 8192 x 8192`) and a requested `10s` burn-in:

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

## Troubleshooting

If `import torch` fails with `libnvshmem_host.so.3: cannot open shared object file`, it usually means the environment was activated with `.venv/bin/activate` instead of `activate_pytorch.sh`.

Use:

```bash
source ~/mllib/activate_pytorch.sh
```

If you need to confirm the helper is exporting the right path:

```bash
echo "$LD_LIBRARY_PATH"
python - <<'PY'
import site, os
for base in site.getsitepackages():
    candidate = os.path.join(base, 'nvidia', 'nvshmem', 'lib')
    if os.path.isdir(candidate):
        print(candidate)
PY
```

## Summary

This non-HPC setup uses the repository's optimized release wheels instead of rebuilding PyTorch from source.

The important operational commands are:

```bash
cd ~/dgx_spark_config
./spark_pytorch_user_setup.sh --env-dir ~/mllib

source ~/mllib/activate_pytorch.sh
python
```

That is the path that was validated successfully on this DGX Spark.