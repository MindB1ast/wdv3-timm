#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "$0")/.." && pwd)
comfy=${1:?Usage: bash audit/install_comfy_cpu.sh COMFY_ROOT}
revision=423cbd20511becb1f16122ab518e92e58bfa03d9 # ComfyUI v0.43.2

python -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
if [[ ! -d "$comfy/.git" ]]; then
    git clone --depth 1 --branch v0.43.2 https://github.com/Comfy-Org/ComfyUI.git "$comfy"
fi
test "$(git -C "$comfy" rev-parse HEAD)" = "$revision"

# Only this isolated environment replaces the pins incompatible with SAM and ComfyUI.
requirements=$(mktemp)
trap 'rm -f "$requirements"' EXIT
grep -Ev '^(huggingface_hub|safetensors|torch|torchvision|torchaudio)==' "$repo/requirements.txt" > "$requirements"
python -m pip install --index-url https://download.pytorch.org/whl/cpu \
    torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0
python -m pip install -r "$requirements" -r "$repo/requirements-sam-transformers.txt" \
    -r "$comfy/requirements.txt" timm==1.0.31 huggingface-hub==1.31.0 safetensors==0.8.0
python -m pip check
python -c 'import comfy_kitchen, torch; from transformers import Sam3Model, Sam3Processor; assert torch.version.cuda is None'
