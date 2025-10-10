#!/usr/bin/env bash
###
# Running on arch you need to install:
# - rocminfo (unless you're using cpuonly)
# - pyenv (optional; if missing, system python3 will be used)
###

set -e

# Load .env if present (allows overriding defaults below)
if [ -f ./.env ]; then
  set -a
  # shellcheck disable=SC1091
  source ./.env
  set +a
fi

export PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
#export ROCM_VERSION="release"
export ROCM_VERSION="${ROCM_VERSION:-nightly}"
export DOCKER_INSTANCE="${DOCKER_INSTANCE:-local-comfyui}"
export ROOT_DIR="${ROOT_DIR:-${PWD}/data/home-local}"
export COMFYUI_PORT="${COMFYUI_PORT:-31490}"

# Prevent accidental use of /root when running locally as non-root
if [ "$(id -u)" -ne 0 ] && [[ "$ROOT_DIR" == /root* ]]; then
  echo "Warning: ROOT_DIR is set to '$ROOT_DIR' (likely from .env) but you are not root. Using local data dir instead."
  ROOT_DIR="${PWD}/data/home-local"
  export ROOT_DIR
fi

if [ ! -d "${ROOT_DIR}" ]; then
  mkdir -p "${ROOT_DIR}"
fi

. conf/functions.sh
has_rocm
activate_venv

# Non-interactive mode support
NON_INTERACTIVE="${NON_INTERACTIVE:-0}"
REINSTALL_ROCM="${REINSTALL_ROCM:-}"
REINSTALL_FLASH="${REINSTALL_FLASH:-}"

if [ "$NON_INTERACTIVE" = "1" ]; then
  if [[ "${REINSTALL_ROCM,,}" =~ ^(y|yes|1|true)$ ]]; then
    install_rocm_torch
  else
    echo "Skipping ROCm torch reinstallation (non-interactive)."
  fi
else
  printf "Reinstall ROCm torch? (y/N): "
  read -r reinstall_rocm
  reinstall_rocm=${reinstall_rocm:-n}
  if [[ $reinstall_rocm =~ ^[Yy]$ ]]; then
    install_rocm_torch
  else
    echo "Skipping ROCm torch reinstallation."
  fi
fi

if [ "$NON_INTERACTIVE" = "1" ]; then
  if [[ "${REINSTALL_FLASH,,}" =~ ^(y|yes|1|true)$ ]]; then
    install_flash_attention
  else
    echo "Skipping Flash Attention reinstallation (non-interactive)."
  fi
else
  printf "Reinstall Flash Attention? (y/N): "
  read -r reinstall_flash
  reinstall_flash=${reinstall_flash:-n}
  if [[ $reinstall_flash =~ ^[Yy]$ ]]; then
    install_flash_attention
  else
    echo "Skipping Flash Attention reinstallation."
  fi
fi

setup_comfyui
has_cuda
launch_comfyui
