#!/usr/bin/env bash

set -euo pipefail

RELEASE_API="https://api.github.com/repos/GuigsEvt/dgx_spark_config/releases/latest"
ENV_ROOT="${HOME}/mllib"
INSTALL_EXTRAS=0
FORCE_DOWNLOAD=0
PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
  cat <<'EOF'
Usage: spark_pytorch_user_setup.sh [--env-dir PATH] [--with-extras] [--force-download]

Installs the latest optimized DGX Spark PyTorch release into a user-local virtual
environment without running the source-build autosetup path.

Options:
  --env-dir PATH      Target directory that will contain .venv and downloaded wheels.
  --with-extras       Also install torchvision, torchaudio, onnx, and flash-attn if available.
  --force-download    Re-download wheel files even if they are already cached locally.
  -h, --help          Show this help message.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-dir)
      ENV_ROOT="$2"
      shift 2
      ;;
    --with-extras)
      INSTALL_EXTRAS=1
      shift
      ;;
    --force-download)
      FORCE_DOWNLOAD=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[ERROR] Missing required command: $1" >&2
    exit 1
  fi
}

find_registered_lib() {
  local lib_name="$1"
  ldconfig -p 2>/dev/null | awk -v name="$lib_name" '$1 == name {print $NF; exit}'
}

need_cmd "${PYTHON_BIN}"
need_cmd nvidia-smi

ARCH="$(uname -m)"
if [[ "${ARCH}" != "aarch64" ]]; then
  echo "[WARN] Detected ${ARCH}. This release is tuned for DGX Spark aarch64." >&2
fi

GPU_CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 || true)"
if [[ -n "${GPU_CC}" ]]; then
  echo "[INFO] GPU compute capability: ${GPU_CC}"
  if [[ "${GPU_CC}" != 12.* ]]; then
    echo "[WARN] Expected Blackwell SM 12.x, got ${GPU_CC}." >&2
  fi
fi

declare -A RUNTIME_LIBS=(
  [libcublas.so]="cuBLAS"
  [libcudnn.so]="cuDNN"
  [libcusparseLt.so]="cuSPARSELt"
  [libnccl.so]="NCCL"
)

missing_runtime=0
for lib_name in "${!RUNTIME_LIBS[@]}"; do
  lib_path="$(find_registered_lib "${lib_name}" || true)"
  if [[ -n "${lib_path}" ]]; then
    echo "[INFO] ${RUNTIME_LIBS[${lib_name}]}: ${lib_path}"
  else
    echo "[WARN] ${RUNTIME_LIBS[${lib_name}]} runtime (${lib_name}) is not registered with ldconfig." >&2
    missing_runtime=1
  fi
done

if [[ "${missing_runtime}" -eq 1 ]]; then
  echo "[WARN] Proceeding anyway because this machine may expose libraries through non-standard paths." >&2
fi

VENV_DIR="${ENV_ROOT}/.venv"
WHEEL_DIR="${ENV_ROOT}/wheels/latest"
mkdir -p "${WHEEL_DIR}"

echo "[INFO] Creating virtual environment at ${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install numpy

export INSTALL_EXTRAS

mapfile -t RELEASE_ASSETS < <(
  "${PYTHON_BIN}" - <<'PY'
import json
import os
import sys
import urllib.request

release_api = "https://api.github.com/repos/GuigsEvt/dgx_spark_config/releases/latest"
wanted = ["triton", "torch"]
if os.environ.get("INSTALL_EXTRAS") == "1":
    wanted.extend(["torchvision", "torchaudio", "onnx", "flash_attn"])

with urllib.request.urlopen(release_api) as response:
    payload = json.load(response)

assets = payload.get("assets", [])
for prefix in wanted:
    match = next((asset for asset in assets if asset["name"].startswith(f"{prefix}-")), None)
    if match is None:
        print(f"missing asset for {prefix}", file=sys.stderr)
        sys.exit(1)
    print(f"{match['name']}\t{match['browser_download_url']}")
PY
)

download_asset() {
  local url="$1"
  local destination="$2"

  if [[ -f "${destination}" && "${FORCE_DOWNLOAD}" -eq 0 ]]; then
    echo "[INFO] Reusing ${destination}"
    return
  fi

  echo "[INFO] Downloading $(basename "${destination}")"
  "${PYTHON_BIN}" - "${url}" "${destination}" <<'PY'
import sys
import urllib.request

url, destination = sys.argv[1:3]
urllib.request.urlretrieve(url, destination)
PY
}

for entry in "${RELEASE_ASSETS[@]}"; do
  asset_name="${entry%%$'\t'*}"
  asset_url="${entry#*$'\t'}"
  download_asset "${asset_url}" "${WHEEL_DIR}/${asset_name}"
done

python -m pip install "${WHEEL_DIR}"/triton-*.whl
python -m pip install "${WHEEL_DIR}"/torch-*.whl

echo "[INFO] Installing NVSHMEM runtime for the CUDA 13 wheel"
python -m pip install nvidia-nvshmem-cu13

NVSHMEM_LIB_DIR="$(${PYTHON_BIN} - <<'PY'
import os
import site

candidates = []
for base in site.getsitepackages():
    candidates.append(os.path.join(base, 'nvidia', 'nvshmem', 'lib'))

user_site = site.getusersitepackages()
if user_site:
    candidates.append(os.path.join(user_site, 'nvidia', 'nvshmem', 'lib'))

for candidate in candidates:
    if os.path.isdir(candidate):
        print(candidate)
        break
PY
)"

if [[ -n "${NVSHMEM_LIB_DIR}" ]]; then
  export LD_LIBRARY_PATH="${NVSHMEM_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  echo "[INFO] NVSHMEM runtime: ${NVSHMEM_LIB_DIR}"
else
  echo "[WARN] NVSHMEM wheel installed, but its library directory was not found automatically." >&2
fi

ACTIVATE_HELPER="${ENV_ROOT}/activate_pytorch.sh"
cat > "${ACTIVATE_HELPER}" <<EOF
#!/usr/bin/env bash

source "${VENV_DIR}/bin/activate"
EOF

if [[ -n "${CUDA_HOME:-}" ]]; then
  cat >> "${ACTIVATE_HELPER}" <<EOF
export CUDA_HOME="${CUDA_HOME}"
export PATH="\${CUDA_HOME}/bin:\${PATH}"
EOF
fi

if [[ -n "${NVSHMEM_LIB_DIR}" ]]; then
  cat >> "${ACTIVATE_HELPER}" <<EOF
export LD_LIBRARY_PATH="${NVSHMEM_LIB_DIR}\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"
EOF
fi

chmod +x "${ACTIVATE_HELPER}"

if [[ "${INSTALL_EXTRAS}" -eq 1 ]]; then
  for pattern in torchvision torchaudio onnx flash_attn; do
    if compgen -G "${WHEEL_DIR}/${pattern}-*.whl" >/dev/null; then
      python -m pip install "${WHEEL_DIR}"/${pattern}-*.whl
    fi
  done
fi

python - <<'PY'
import torch

print("Torch version:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in the installed environment")

props = torch.cuda.get_device_properties(0)
print("Device:", props.name)
print("Compute capability:", f"{props.major}.{props.minor}")

x = torch.randn((2048, 2048), device="cuda", dtype=torch.float16)
y = torch.randn((2048, 2048), device="cuda", dtype=torch.float16)
z = x @ y
torch.cuda.synchronize()
print("FP16 matmul ok:", tuple(z.shape))
PY

echo "[INFO] Environment ready. Activate it with: source ${ACTIVATE_HELPER}"