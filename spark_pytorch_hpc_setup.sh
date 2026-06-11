#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_ROOT="${HOME}/mllib-hpc"
PYTORCH_DIR=""
TORCH_VERSION="${TORCH_VERSION:-2.9.1}"
CLONE_PYTORCH=1
INSTALL_REQUIREMENTS=1
APPLY_REPO_PATCHES=1
FRESH_INSTALL=0

usage() {
  cat <<'EOF'
Usage: spark_pytorch_hpc_setup.sh [options]

Prepare a supported, HPC-oriented PyTorch source-build environment for DGX Spark.
This uses GCC/G++ plus nvcc and Blackwell build flags. It does not try to build
PyTorch with NVHPC compilers, because PyTorch does not support that path.

Options:
  --env-dir PATH           Root directory for the build environment. Default: ~/mllib-hpc
  --pytorch-dir PATH       PyTorch source directory. Default: <env-dir>/pytorch
  --torch-version VERSION  PyTorch tag to check out. Default: 2.9.1
  --fresh                  Remove the existing env-dir before recreating it.
  --skip-clone             Do not clone or update the PyTorch source tree.
  --skip-requirements      Do not install PyTorch Python build requirements.
  --skip-patches           Do not apply the repo's local PyTorch patches.
  -h, --help               Show this help.
EOF
}

canonicalize_path() {
  local path="$1"

  if command -v realpath >/dev/null 2>&1; then
    realpath -m "$path"
    return
  fi

  python3 -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "$path"
}

remove_env_root_if_requested() {
  local env_root="$1"
  local canonical_env_root=""
  local canonical_home=""

  if [[ "$FRESH_INSTALL" -ne 1 ]]; then
    return 0
  fi

  canonical_env_root="$(canonicalize_path "$env_root")"
  canonical_home="$(canonicalize_path "$HOME")"

  if [[ -z "$canonical_env_root" || "$canonical_env_root" == "/" || "$canonical_env_root" == "$canonical_home" ]]; then
    echo "[ERROR] Refusing to remove unsafe env-dir: $canonical_env_root" >&2
    exit 1
  fi

  if [[ -e "$canonical_env_root" ]]; then
    echo "[INFO] Removing existing environment root at $canonical_env_root"
    rm -rf -- "$canonical_env_root"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-dir)
      ENV_ROOT="$2"
      shift 2
      ;;
    --pytorch-dir)
      PYTORCH_DIR="$2"
      shift 2
      ;;
    --torch-version)
      TORCH_VERSION="$2"
      shift 2
      ;;
    --fresh)
      FRESH_INSTALL=1
      shift
      ;;
    --skip-clone)
      CLONE_PYTORCH=0
      shift
      ;;
    --skip-requirements)
      INSTALL_REQUIREMENTS=0
      shift
      ;;
    --skip-patches)
      APPLY_REPO_PATCHES=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[ERROR] Missing required command: $1" >&2
    exit 1
  fi
}

apply_repo_patch_if_needed() {
  local repo_dir="$1"
  local patch_file="$2"
  local label="$3"

  if [[ ! -f "$patch_file" ]]; then
    echo "[WARN] Patch file not found for ${label}: $patch_file"
    return 0
  fi

  if git -C "$repo_dir" apply --check "$patch_file" >/dev/null 2>&1; then
    echo "[INFO] Applying ${label} patch"
    git -C "$repo_dir" apply "$patch_file"
    return 0
  fi

  echo "[INFO] Skipping ${label} patch; it is already applied or not compatible with the current tree"
}

write_activate_helper() {
  local helper_path="$1"
  local venv_dir="$2"

  cat > "$helper_path" <<EOF
#!/usr/bin/env bash

source "$venv_dir/bin/activate"
source "$ROOT_DIR/grace_blackwell_pytorch_autosetup.sh" --skip-system-installs
EOF
  chmod +x "$helper_path"
}

need_cmd python3
need_cmd git

remove_env_root_if_requested "$ENV_ROOT"

mkdir -p "$ENV_ROOT"

if [[ -z "$PYTORCH_DIR" ]]; then
  PYTORCH_DIR="$ENV_ROOT/pytorch"
fi

VENV_DIR="$ENV_ROOT/.venv"
ACTIVATE_HELPER="$ENV_ROOT/activate_hpc_pytorch.sh"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[INFO] Creating build virtual environment at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "[INFO] Installing base Python tooling"
python -m pip install --upgrade pip wheel
python -m pip install 'setuptools>=70.1.0,<80.0' numpy

echo "[INFO] Discovering the DGX Spark CUDA build environment"
# shellcheck disable=SC1090
source "$ROOT_DIR/grace_blackwell_pytorch_autosetup.sh" --skip-system-installs

write_activate_helper "$ACTIVATE_HELPER" "$VENV_DIR"

echo "[INFO] This setup uses GCC/G++ plus nvcc. PyTorch does not support NVHPC compilers as its primary host compiler path."

if [[ "$CLONE_PYTORCH" -eq 1 ]]; then
  if [[ ! -d "$PYTORCH_DIR/.git" ]]; then
    echo "[INFO] Cloning PyTorch into $PYTORCH_DIR"
    git clone --recursive --branch "v$TORCH_VERSION" https://github.com/pytorch/pytorch.git "$PYTORCH_DIR"
  fi

  echo "[INFO] Checking out PyTorch v$TORCH_VERSION"
  git -C "$PYTORCH_DIR" fetch --tags origin "v$TORCH_VERSION" >/dev/null 2>&1 || true
  git -C "$PYTORCH_DIR" checkout "v$TORCH_VERSION"
  git -C "$PYTORCH_DIR" submodule update --init --recursive

  if [[ "$APPLY_REPO_PATCHES" -eq 1 ]]; then
    apply_repo_patch_if_needed "$PYTORCH_DIR" "$ROOT_DIR/pytorch/pytorch.patch" "PyTorch"

    if [[ -e "$PYTORCH_DIR/third_party/flash-attention/.git" ]]; then
      apply_repo_patch_if_needed \
        "$PYTORCH_DIR/third_party/flash-attention" \
        "$ROOT_DIR/pytorch/flash-attention.patch" \
        "FlashAttention"
    fi
  fi

  if [[ "$INSTALL_REQUIREMENTS" -eq 1 ]]; then
    echo "[INFO] Installing PyTorch build requirements"
    python -m pip install -r "$PYTORCH_DIR/requirements.txt" -r "$PYTORCH_DIR/requirements-build.txt"
  fi
fi

echo
echo "[INFO] HPC-style PyTorch build environment is ready"
echo "[INFO] Activate it with: source $ACTIVATE_HELPER"
if [[ "$CLONE_PYTORCH" -eq 1 ]]; then
  echo "[INFO] Source tree: $PYTORCH_DIR"
  echo "[INFO] Example build command:"
  echo "       source $ACTIVATE_HELPER && cd $PYTORCH_DIR && python -m pip wheel . -w dist --no-deps --verbose"
fi