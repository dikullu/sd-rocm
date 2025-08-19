#!/bin/sh
export PYTHON_VERSION="3.12"
export ROCM_VERSION="nightly"
export DOCKER_INSTANCE="local-comfyui"
export ROOT_DIR="${PWD}/data/home-local"

if [ ! -d "${ROOT_DIR}" ]; then
  mkdir "${ROOT_DIR}"
fi

export COMFYUI_PORT=31490

. conf/functions.sh
activate_venv
