#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
TRL_BIN="${REPO_ROOT}/.venv/bin/trl"

# Gemma 4 E4B canary runner.
# This uses the current stable TRL path with exact-form routing and reward telemetry.
# Defaults are intentionally conservative for the first working vLLM run on a 2x4090 setup.

MODEL="${ABIDE_MODEL:-google/gemma-4-E4B-it}"
PORT="${ABIDE_PORT:-8000}"
NUM_PROMPTS="${ABIDE_NUM_PROMPTS:-4}"
BATCH_SIZE="${ABIDE_BATCH_SIZE:-1}"
NUM_GENERATIONS="${ABIDE_NUM_GENERATIONS:-2}"
BETA="${ABIDE_BETA:-0.02}"
LR="${ABIDE_LR:-3e-5}"
SAVE_STEPS="${ABIDE_SAVE_STEPS:-10}"
TELEMETRY_EVERY="${ABIDE_TELEMETRY_EVERY:-1}"
OUTPUT_DIR="${ABIDE_OUTPUT_DIR:-models/grpo_trl_gemma4_e4b}"
FORM_SET="${ABIDE_FORM_SET:-rl_default}"
GPU_MEMORY_UTILIZATION="${ABIDE_GPU_MEMORY_UTILIZATION:-0.80}"
MAX_MODEL_LEN="${ABIDE_MAX_MODEL_LEN:-1024}"
STARTUP_TIMEOUT="${ABIDE_STARTUP_TIMEOUT:-600}"
MAX_COMPLETION_LENGTH="${ABIDE_MAX_COMPLETION_LENGTH:-128}"
USE_VLLM="${ABIDE_USE_VLLM:-1}"
ENFORCE_EAGER="${ABIDE_ENFORCE_EAGER:-1}"
LORA_R="${ABIDE_LORA_R:-16}"
LORA_ALPHA="${ABIDE_LORA_ALPHA:-32}"
PREPARE_RUNTIME="${ABIDE_PREPARE_RUNTIME:-1}"
VLLM_PID=""

cleanup() {
    echo ""
    echo "Interrupted. Cleaning up..."
    if [ -n "$VLLM_PID" ]; then
        kill "$VLLM_PID" 2>/dev/null || true
    fi
    pkill -f "trl vllm-serve" 2>/dev/null || true
    exit 1
}
trap cleanup INT TERM

echo "============================================================"
echo "Gemma 4 E4B GRPO Canary"
echo "============================================================"
echo "Model: $MODEL"
echo "Prompts: $NUM_PROMPTS"
echo "Batch size: $BATCH_SIZE"
echo "Generations: $NUM_GENERATIONS"
echo "Beta (KL): $BETA"
echo "Learning rate: $LR"
echo "Form set: $FORM_SET"
echo "Use vLLM: $USE_VLLM"
echo "LoRA rank: $LORA_R"
echo "LoRA alpha: $LORA_ALPHA"
echo "Telemetry every: $TELEMETRY_EVERY"
echo "Output: $OUTPUT_DIR"
echo "============================================================"

mkdir -p logs "$OUTPUT_DIR"

if [ ! -x "$PYTHON_BIN" ] || [ ! -x "$TRL_BIN" ]; then
    echo "ERROR: missing virtualenv entrypoints."
    echo "Run: uv sync --extra training"
    exit 1
fi

if [ "$PREPARE_RUNTIME" = "1" ]; then
    "${REPO_ROOT}/scripts/prepare_gemma4_runtime.sh"
fi

if ! "$PYTHON_BIN" - <<PY
from transformers import AutoConfig

AutoConfig.from_pretrained("$MODEL")
PY
then
    echo "ERROR: current Transformers build does not support $MODEL."
    echo "Try: uv pip install --upgrade git+https://github.com/huggingface/transformers.git"
    exit 1
fi

if [ "$USE_VLLM" = "1" ]; then
    echo "Cleaning up old vLLM/TRL processes..."
    pkill -f "trl vllm-serve" || true
    pkill -f "vf-vllm" || true
    pkill -f "vllm.entrypoints" || true
    sleep 2

    echo "Starting TRL vLLM server on GPU 1..."
    SERVER_ARGS=(
        vllm-serve
        --model "$MODEL"
        --port "$PORT"
        --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION"
        --tensor_parallel_size 1
        --max_model_len "$MAX_MODEL_LEN"
    )
    if [ "$ENFORCE_EAGER" = "1" ]; then
        SERVER_ARGS+=(--enforce_eager)
    fi

    CUDA_VISIBLE_DEVICES=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    nohup "$TRL_BIN" "${SERVER_ARGS[@]}" \
        > logs/vllm_gemma4_e4b.log 2>&1 &

    VLLM_PID=$!
    echo "TRL vLLM server started with PID $VLLM_PID (ctrl-c to abort)"

    echo "Waiting for vLLM server to be ready..."
    timeout "$STARTUP_TIMEOUT" bash -c "until curl -s localhost:$PORT/health > /dev/null 2>&1; do sleep 5; done" || {
        echo "ERROR: vLLM failed to start. Check logs/vllm_gemma4_e4b.log"
        tail -100 logs/vllm_gemma4_e4b.log
        exit 1
    }
    echo "vLLM server is ready!"
else
    echo "Skipping vLLM server startup."
    echo "Gemma 4 E4B currently falls back to local TRL generation on this machine."
fi

echo ""
echo "Starting Gemma 4 E4B training on GPU 0..."
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ABIDE_FORM_SET="$FORM_SET"
export ABIDE_MODEL="$MODEL"

TRAIN_ARGS=(
    --model "$MODEL"
    --prompts "$NUM_PROMPTS"
    --batch-size "$BATCH_SIZE"
    --num-generations "$NUM_GENERATIONS"
    --beta "$BETA"
    --lr "$LR"
    --output "$OUTPUT_DIR"
    --save-steps "$SAVE_STEPS"
    --telemetry-every "$TELEMETRY_EVERY"
    --max-completion-length "$MAX_COMPLETION_LENGTH"
    --lora-r "$LORA_R"
    --lora-alpha "$LORA_ALPHA"
)

if [ "$USE_VLLM" = "1" ]; then
    TRAIN_ARGS+=(--use-vllm --port "$PORT")
else
    TRAIN_ARGS+=(--no-vllm)
fi

CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" scripts/train_grpo_trl.py "${TRAIN_ARGS[@]}"

echo ""
echo "Training finished. Cleaning up..."
kill "$VLLM_PID" 2>/dev/null || true

echo ""
echo "============================================================"
echo "GEMMA 4 E4B CANARY COMPLETE"
echo "============================================================"
echo "Inspect reward telemetry, pass rate, and zero-rate before scaling up."
echo "============================================================"
