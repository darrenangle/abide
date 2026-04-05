#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV_DIR="${ABIDE_PRIME_RL_VENV:-${REPO_ROOT}/.venv-prime-rl}"
PRIME_RL_SRC="${ABIDE_PRIME_RL_SRC:-${REPO_ROOT}/.prime-rl-src}"
PRIME_RL_REPO="${ABIDE_PRIME_RL_REPO:-https://github.com/PrimeIntellect-ai/prime-rl.git}"
PRIME_RL_REF="${ABIDE_PRIME_RL_REF:-main}"
TRANSFORMERS_REF="${ABIDE_PRIME_RL_TRANSFORMERS_REF:-git+https://github.com/huggingface/transformers.git}"
MISTRAL_COMMON_VERSION="${ABIDE_PRIME_RL_MISTRAL_COMMON_VERSION:-1.11.0}"

case "$VENV_DIR" in
    /*) ;;
    *) VENV_DIR="${REPO_ROOT}/${VENV_DIR}" ;;
esac

case "$PRIME_RL_SRC" in
    /*) ;;
    *) PRIME_RL_SRC="${REPO_ROOT}/${PRIME_RL_SRC}" ;;
esac

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
uv pip install --python "${VENV_DIR}/bin/python" --upgrade "${TRANSFORMERS_REF}"
uv pip install --python "${VENV_DIR}/bin/python" --upgrade "mistral-common==${MISTRAL_COMMON_VERSION}"
uv pip install --python "${VENV_DIR}/bin/python" --editable "$REPO_ROOT"

"${VENV_DIR}/bin/python" - <<'PY'
import importlib.metadata as md

import abide_poetry_forms
import prime_rl
import verifiers
from transformers import AutoConfig

print("prime-rl", md.version("prime-rl"))
print("verifiers", md.version("verifiers"))
print("mistral-common", md.version("mistral-common"))
print("abide-poetry-forms", abide_poetry_forms.DEFAULT_ENV_ID)
print(type(AutoConfig.from_pretrained("google/gemma-4-E2B-it", trust_remote_code=True)).__name__)
PY
