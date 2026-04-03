#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV_DIR="${ABIDE_VERIFIERS_VENV:-${REPO_ROOT}/.venv-verifiers}"
VERIFIERS_TAG="${ABIDE_VERIFIERS_TAG:-v0.1.11}"
VERIFIERS_RL_URL="git+https://github.com/PrimeIntellect-ai/verifiers@${VERIFIERS_TAG}#subdirectory=packages/verifiers-rl"

if [ ! -d "$VENV_DIR" ]; then
    uv venv "$VENV_DIR"
fi

if "${VENV_DIR}/bin/python" - <<'PY'
import importlib
import sys

checks = {
    "verifiers": "0.1.11",
    "verifiers_rl": "",
}

for name, prefix in checks.items():
    try:
        module = importlib.import_module(name)
    except Exception:
        sys.exit(1)
    if prefix and not getattr(module, "__version__", "").startswith(prefix):
        sys.exit(1)
PY
then
    echo "Legacy verifiers runtime already prepared at ${VENV_DIR}."
    exit 0
fi

echo "Preparing legacy verifiers runtime in ${VENV_DIR}..."
UV_PROJECT_ENVIRONMENT="$VENV_DIR" uv sync --no-dev --extra evals
uv pip install --python "${VENV_DIR}/bin/python" "${VERIFIERS_RL_URL}"

"${VENV_DIR}/bin/python" - <<'PY'
import verifiers
import verifiers_rl

print("verifiers", verifiers.__version__)
print("verifiers_rl", getattr(verifiers_rl, "__file__", "installed"))
PY
