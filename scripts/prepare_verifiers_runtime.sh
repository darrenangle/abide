#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV_DIR="${ABIDE_VERIFIERS_VENV:-${REPO_ROOT}/.venv-verifiers}"
VERIFIERS_TAG="${ABIDE_VERIFIERS_TAG:-v0.1.11}"
VERIFIERS_RL_URL="git+https://github.com/PrimeIntellect-ai/verifiers@${VERIFIERS_TAG}#subdirectory=packages/verifiers-rl"
GEMMA4_VLLM_WHEEL_URL="${ABIDE_GEMMA4_VLLM_WHEEL_URL:-https://wheels.vllm.ai/66e86f1dbd565292a253e7d2d6851f65dc4f14ba/vllm-0.18.2rc1.dev73%2Bg66e86f1db-cp38-abi3-manylinux_2_31_x86_64.whl}"
GEMMA4_TRANSFORMERS_REF="${ABIDE_GEMMA4_TRANSFORMERS_REF:-git+https://github.com/huggingface/transformers.git@edaac7db98e34208209fd67d8c66781b8c2e4a53}"
EXPECTED_VLLM_VERSION="0.18.2rc1.dev73+g66e86f1db"
EXPECTED_TRANSFORMERS_PREFIX="5.5.0.dev0"

if [ ! -d "$VENV_DIR" ]; then
    uv venv "$VENV_DIR"
fi

if "${VENV_DIR}/bin/python" - <<'PY'
import importlib
from importlib.util import find_spec
import sys

checks = {
    "bitsandbytes": "",
    "verifiers": "0.1.11",
    "verifiers_rl": "",
    "vllm": "0.18.2rc1.dev73+g66e86f1db",
    "transformers": "5.5.0.dev0",
}

for name, prefix in checks.items():
    try:
        module = importlib.import_module(name)
    except Exception:
        sys.exit(1)
    if prefix and not getattr(module, "__version__", "").startswith(prefix):
        sys.exit(1)

try:
    import abide.verifiers_vllm_server  # noqa: F401
except Exception:
    sys.exit(1)

if find_spec("flash_attn") is not None:
    sys.exit(1)
PY
then
    echo "Legacy verifiers runtime already prepared at ${VENV_DIR}."
    exit 0
fi

echo "Preparing legacy verifiers runtime in ${VENV_DIR}..."
UV_PROJECT_ENVIRONMENT="$VENV_DIR" uv sync --no-dev --extra evals
uv pip install --python "${VENV_DIR}/bin/python" "${VERIFIERS_RL_URL}"
uv pip install --python "${VENV_DIR}/bin/python" --reinstall "${GEMMA4_VLLM_WHEEL_URL}"
uv pip install --python "${VENV_DIR}/bin/python" --upgrade "${GEMMA4_TRANSFORMERS_REF}"
uv pip install --python "${VENV_DIR}/bin/python" bitsandbytes
uv pip uninstall --python "${VENV_DIR}/bin/python" flash-attn

"${VENV_DIR}/bin/python" - <<'PY'
import verifiers
import verifiers_rl
import transformers
import vllm
from transformers import AutoConfig

print("verifiers", verifiers.__version__)
print("verifiers_rl", getattr(verifiers_rl, "__file__", "installed"))
print("transformers", transformers.__version__)
print("vllm", vllm.__version__)
print(type(AutoConfig.from_pretrained("google/gemma-4-E4B-it")).__name__)
PY
