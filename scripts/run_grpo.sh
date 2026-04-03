#!/bin/bash
set -e

# Legacy verifiers GRPO training for Abide poetry forms.
# This script targets the smaller local verifiers RL trainer, not TRL.

# Configuration
MODEL="${ABIDE_MODEL:-google/gemma-3-4b-it}"
FORM_SET="${ABIDE_FORM_SET:-well_known}"
OUTPUT_DIR="${ABIDE_OUTPUT_DIR:-models/abide_verifiers_gemma_well_known}"
PORT=8000
VLLM_PID=""
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
echo "Abide Verifiers GRPO Training"
echo "============================================================"
echo "Model: $MODEL"
echo "Form set: $FORM_SET"
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
# Using V1 engine (patched scheduler assertion)
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
    > logs/vllm.log 2>&1 &

VLLM_PID=$!
echo "vLLM started with PID $VLLM_PID (ctrl-c to abort)"

# Wait for vLLM to be ready
echo "Waiting for vLLM to be ready..."
timeout 300 bash -c "until curl -s localhost:$PORT/health > /dev/null 2>&1; do sleep 2; done" || {
    echo "ERROR: vLLM failed to start. Check logs/vllm.log"
    cat logs/vllm.log | tail -50
    exit 1
}
echo "vLLM is ready!"

# Run training on GPU 0
echo ""
echo "Starting training on GPU 0..."
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=0 "${VERIFIERS_VENV}/bin/python" scripts/train_grpo.py \
    --model "$MODEL" \
    --form-set "$FORM_SET" \
    --output "$OUTPUT_DIR" \
    --port "$PORT"

# Cleanup
echo ""
echo "Training finished. Cleaning up..."
kill $VLLM_PID 2>/dev/null || true
echo "Done."
