#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV_DIR="${ABIDE_PRIME_RL_VENV:-${REPO_ROOT}/.venv-prime-rl}"
PRIME_RL_SRC="${ABIDE_PRIME_RL_SRC:-${REPO_ROOT}/.prime-rl-src}"
PRIME_RL_REPO="${ABIDE_PRIME_RL_REPO:-https://github.com/PrimeIntellect-ai/prime-rl.git}"
PRIME_RL_REF="${ABIDE_PRIME_RL_REF:-main}"
GEMMA4_VLLM_WHEEL_URL="${ABIDE_GEMMA4_VLLM_WHEEL_URL:-https://wheels.vllm.ai/66e86f1dbd565292a253e7d2d6851f65dc4f14ba/vllm-0.18.2rc1.dev73%2Bg66e86f1db-cp38-abi3-manylinux_2_31_x86_64.whl}"
TRANSFORMERS_REF="${ABIDE_PRIME_RL_TRANSFORMERS_REF:-git+https://github.com/huggingface/transformers.git@edaac7db98e34208209fd67d8c66781b8c2e4a53}"
MISTRAL_COMMON_VERSION="${ABIDE_PRIME_RL_MISTRAL_COMMON_VERSION:-1.11.0}"
EXPECTED_VLLM_VERSION="0.18.2rc1.dev73+g66e86f1db"
EXPECTED_TRANSFORMERS_PREFIX="5.5.0.dev0"

case "$VENV_DIR" in
    /*) ;;
    *) VENV_DIR="${REPO_ROOT}/${VENV_DIR}" ;;
esac

case "$PRIME_RL_SRC" in
    /*) ;;
    *) PRIME_RL_SRC="${REPO_ROOT}/${PRIME_RL_SRC}" ;;
esac

if [ -d "$VENV_DIR" ] && [ -d "$PRIME_RL_SRC/.git" ] && "${VENV_DIR}/bin/python" - <<'PY'
import importlib
import sys

checks = {
    "transformers": "5.5.0.dev0",
    "vllm": "0.18.2rc1.dev73+g66e86f1db",
}

for name, prefix in checks.items():
    try:
        module = importlib.import_module(name)
    except Exception:
        sys.exit(1)
    if not getattr(module, "__version__", "").startswith(prefix):
        sys.exit(1)

try:
    import abide_poetry_forms
    import prime_rl
    import verifiers
    from transformers import AutoConfig
except Exception:
    sys.exit(1)

config = AutoConfig.from_pretrained("google/gemma-4-E2B-it", trust_remote_code=True)
if type(config).__name__ != "Gemma4Config":
    sys.exit(1)
PY
then
    echo "Prime-rl Gemma 4 runtime already prepared."
    exit 0
fi

if [ ! -d "$PRIME_RL_SRC/.git" ]; then
    git clone "$PRIME_RL_REPO" "$PRIME_RL_SRC"
fi

git -C "$PRIME_RL_SRC" fetch --tags origin
if [ "$PRIME_RL_REF" = "main" ]; then
    git -C "$PRIME_RL_SRC" checkout main
    git -C "$PRIME_RL_SRC" pull --ff-only origin main
else
    git -C "$PRIME_RL_SRC" checkout --detach "$PRIME_RL_REF"
fi

if [ ! -d "$VENV_DIR" ]; then
    uv venv --python 3.12 "$VENV_DIR"
fi

echo "Preparing prime-rl runtime in ${VENV_DIR} from ${PRIME_RL_REF}..."
UV_PROJECT_ENVIRONMENT="$VENV_DIR" uv sync --project "$PRIME_RL_SRC" --extra flash-attn --frozen
uv pip install --python "${VENV_DIR}/bin/python" --reinstall "${GEMMA4_VLLM_WHEEL_URL}"
uv pip install --python "${VENV_DIR}/bin/python" --upgrade "${TRANSFORMERS_REF}"
uv pip install --python "${VENV_DIR}/bin/python" --upgrade "mistral-common==${MISTRAL_COMMON_VERSION}"
uv pip install --python "${VENV_DIR}/bin/python" --editable "$REPO_ROOT"

"${VENV_DIR}/bin/python" - <<'PY'
import importlib.metadata as md

import abide_poetry_forms
import prime_rl
import transformers
import verifiers
import vllm
from transformers import AutoConfig

print("prime-rl", md.version("prime-rl"))
print("vllm", vllm.__version__)
print("transformers", transformers.__version__)
print("verifiers", md.version("verifiers"))
print("mistral-common", md.version("mistral-common"))
print("abide-poetry-forms", abide_poetry_forms.DEFAULT_ENV_ID)
print(type(AutoConfig.from_pretrained("google/gemma-4-E2B-it", trust_remote_code=True)).__name__)
PY
