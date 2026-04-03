#!/bin/bash
set -e

# GRPO Training for Gemma 3 4B-it with the legacy verifiers trainer.
# This runner focuses the prompt set on a smaller group of canonical forms.

# Configuration
MODEL="google/gemma-3-4b-it"
PORT=8000
VLLM_PID=""
NUM_PROMPTS="${ABIDE_NUM_PROMPTS:-100000}"
FORM_SET="${ABIDE_FORM_SET:-well_known}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERIFIERS_VENV="${ABIDE_VERIFIERS_VENV:-${REPO_ROOT}/.venv-verifiers}"
cd "$REPO_ROOT"

# Cleanup on ctrl-c
cleanup() {
    echo ""
    echo "Interrupted. Cleaning up..."
    [ -n "$VLLM_PID" ] && kill $VLLM_PID 2>/dev/null
    pkill -f "vf-vllm" 2>/dev/null || true
    exit 1
}
trap cleanup INT TERM

echo "============================================================"
echo "Abide GRPO Training - GEMMA 3 4B-it"
echo "============================================================"
echo "Model: $MODEL (4B params, text-only)"
echo "Prompts: $NUM_PROMPTS ($FORM_SET forms, balanced sampling)"
echo "vLLM: GPU 1, port $PORT"
echo "Training: GPU 0"
echo "============================================================"
echo "Runtime: ${VERIFIERS_VENV}"

# Cleanup old processes
echo "Cleaning up old vLLM processes..."
pkill -f "vf-vllm" || true
pkill -f "vllm.entrypoints" || true
sleep 2

"${REPO_ROOT}/scripts/prepare_verifiers_runtime.sh"

# Create log directory
mkdir -p logs

# Start vf-vllm on GPU 1
echo "Starting vf-vllm on GPU 1..."
CUDA_VISIBLE_DEVICES=1 nohup "${VERIFIERS_VENV}/bin/vf-vllm" \
    --model "$MODEL" \
    --port $PORT \
    --gpu-memory-utilization 0.92 \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --trust-remote-code \
    --disable-log-stats \
    --enforce-eager \
    --dtype bfloat16 \
    > logs/vllm_gemma3.log 2>&1 &

VLLM_PID=$!
echo "vLLM started with PID $VLLM_PID (ctrl-c to abort)"

# Wait for vLLM to be ready
echo "Waiting for vLLM to be ready..."
timeout 300 bash -c "until curl -s localhost:$PORT/health > /dev/null 2>&1; do sleep 2; done" || {
    echo "ERROR: vLLM failed to start. Check logs/vllm_gemma3.log"
    cat logs/vllm_gemma3.log | tail -50
    exit 1
}
echo "vLLM is ready!"

# Run training on GPU 0
echo ""
echo "Starting Gemma 3 training on GPU 0..."
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ABIDE_MODEL="$MODEL"
export ABIDE_FORM_SET="$FORM_SET"
CUDA_VISIBLE_DEVICES=0 "${VERIFIERS_VENV}/bin/python" scripts/train_grpo.py \
    --model "$MODEL" \
    --form-set "$FORM_SET" \
    --prompts $NUM_PROMPTS \
    --output models/abide_verifiers_gemma3_well_known

# Cleanup
echo ""
echo "Training finished. Cleaning up..."
kill $VLLM_PID 2>/dev/null || true
echo "Done."
