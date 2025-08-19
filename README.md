# Docker Compose for Stable Diffusion for ROCm

Simple docker compose for getting ComfyUI and sd-webui (Forge) up and running with minimal modification on the main system.

## Requirements

* docker-compose
* AMD GPU (PRs for other cards are welcome)
* checkpoints are saved into **data/checkpoints** other model files in their respective subfolder, for example
  **data/comfyui/models** or **data/sd-webui/models**

## Instructions

1. clone this repo
2. open repo directory in terminal
3. start up the docker container by typing:
   1. `docker-compose up`
   2. It will take a while to download all python libraries
   3. wait a while until you see the text:  `To see the GUI go to: http://0.0.0.0:31488`
   4. After getting that message start ComfyUI by open browser with the following link: http://localhost:31488
   5. or WebUI by going to http://localhost:31489

### Run locally (no Docker)

You can also run without containers using the helper script:

```
./run-local.sh
```

- On first run it will create a Python virtual environment and set up ComfyUI, ROCm PyTorch and Flash Attention.
- It creates a marker file in the data directory to skip repeated heavy setup on subsequent runs.
- By default it prompts whether to reinstall ROCm Torch and Flash Attention. To automate, use environment variables:
  - `NON_INTERACTIVE=1` to disable prompts
  - `REINSTALL_ROCM=yes` to reinstall ROCm Torch in non-interactive mode
  - `REINSTALL_FLASH=yes` to reinstall Flash Attention in non-interactive mode

You can customize behavior via a `.env` file at the repo root (auto-loaded) or by exporting variables before running:

- `PYTHON_VERSION` (default `3.12`)
- `ROCM_VERSION` (`nightly`, `release`, or `cpuonly`; default `nightly` for local run)
- `ROOT_DIR` (default `${PWD}/data/home-local`)
- `COMFYUI_PORT` (default `31490`)

Example non-interactive run on CPU-only:

```
ROCM_VERSION=cpuonly NON_INTERACTIVE=1 ./run-local.sh
```

Important: The provided .env in this repo is tailored for Docker and sets ROOT_DIR=/root. When running locally as a non-root user, run-local.sh will ignore ROOT_DIR=/root and use ${PWD}/data/home-local instead. To explicitly control this, set ROOT_DIR to a writable directory when invoking the script or in your own .env.local file.

Examples:
```
ROOT_DIR="$PWD/data/home-local" ./run-local.sh
# or
export ROOT_DIR="$PWD/data/home-local"; ./run-local.sh
```

## Hints

To make the docker created files accessable from your own user run these commands

```
sudo setfacl -d -m "u:${USER}:rwX" data/
sudo setfacl -R -m "u:${USER}:rwX" data/
```

You can run in CPU-only mode by setting `ROCM_VERSION=cpuonly` and remove the lines with `/dev/kfd` in **docker-compose.yml**


## Notes

Big thanks for all Open Source gang that have made this possible.

## Links

* ComfyUI: https://github.com/comfyanonymous/ComfyUI
* Automatic1111: https://github.com/AUTOMATIC1111/stable-diffusion-webui
* WebUI Forge: https://github.com/lllyasviel/stable-diffusion-webui-forge
