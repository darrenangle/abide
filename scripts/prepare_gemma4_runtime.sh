#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

GEMMA4_VLLM_WHEEL_URL="${ABIDE_GEMMA4_VLLM_WHEEL_URL:-https://wheels.vllm.ai/66e86f1dbd565292a253e7d2d6851f65dc4f14ba/vllm-0.18.2rc1.dev73%2Bg66e86f1db-cp38-abi3-manylinux_2_31_x86_64.whl}"
GEMMA4_TRANSFORMERS_REF="${ABIDE_GEMMA4_TRANSFORMERS_REF:-git+https://github.com/huggingface/transformers.git@edaac7db98e34208209fd67d8c66781b8c2e4a53}"
EXPECTED_VLLM_VERSION="0.18.2rc1.dev73+g66e86f1db"
EXPECTED_FLASHINFER_VERSION="0.6.7"
EXPECTED_TRANSFORMERS_PREFIX="5.5.0.dev0"

if [ ! -d "${REPO_ROOT}/.venv" ]; then
    echo "ERROR: missing .venv"
    echo "Run: uv sync --extra training"
    exit 1
fi

if "${REPO_ROOT}/.venv/bin/python" - <<'PY'
import importlib
import sys

checks = {
    "flashinfer": "0.6.7",
    "transformers": "5.5.0.dev0",
    "vllm": "0.18.2rc1.dev73+g66e86f1db",
    "wandb": "0.",
}

for name, prefix in checks.items():
    try:
        module = importlib.import_module(name)
    except Exception:
        sys.exit(1)
    if not getattr(module, "__version__", "").startswith(prefix):
        sys.exit(1)

try:
    from transformers import AutoConfig
except Exception:
    sys.exit(1)

config = AutoConfig.from_pretrained("google/gemma-4-E4B-it")
if type(config).__name__ != "Gemma4Config":
    sys.exit(1)
PY
then
    echo "Gemma 4 runtime overlay already prepared."
    exit 0
fi

echo "Preparing Gemma 4 runtime overlay in .venv..."
# Do not chase the latest nightly blindly here. Gemma 4 serving was verified
# against this exact vLLM wheel; newer nightlies regressed into a torch ABI
# mismatch on this machine.
uv pip install --reinstall "$GEMMA4_VLLM_WHEEL_URL"
uv pip install --upgrade "$GEMMA4_TRANSFORMERS_REF"
uv pip install "wandb>=0.16.0"

"${REPO_ROOT}/.venv/bin/python" - <<'PY'
import flashinfer
import transformers
import vllm
import wandb
from transformers import AutoConfig

print("flashinfer", flashinfer.__version__)
print("transformers", transformers.__version__)
print("vllm", vllm.__version__)
print("wandb", wandb.__version__)
print(type(AutoConfig.from_pretrained("google/gemma-4-E4B-it")).__name__)
PY
