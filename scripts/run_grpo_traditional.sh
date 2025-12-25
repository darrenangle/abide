#!/bin/bash
set -e

# GRPO Training for Traditional Poetry Forms Only
# Focuses on well-known forms with weighted sampling:
# - Tier 1 (sonnets, haiku, etc): 40% of training
# - Tier 2 (ghazal, sestina, etc): 39%
# - Tier 3 (less common): 21%

# Configuration
MODEL="${ABIDE_MODEL:-allenai/OLMo-3-7B-Instruct}"
PORT=8000
VLLM_PID=""
NUM_PROMPTS="${ABIDE_NUM_PROMPTS:-50000}"

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
echo "Abide GRPO Training - TRADITIONAL FORMS ONLY"
echo "============================================================"
echo "Model: $MODEL"
echo "Prompts: $NUM_PROMPTS (weighted by form popularity)"
echo "vLLM: GPU 1, port $PORT"
echo "Training: GPU 0"
echo "============================================================"

# Cleanup old processes
echo "Cleaning up old vLLM processes..."
pkill -f "vf-vllm" || true
pkill -f "vllm.entrypoints" || true
sleep 2

# Create log directory
mkdir -p logs

# Start vf-vllm on GPU 1
echo "Starting vf-vllm on GPU 1..."
CUDA_VISIBLE_DEVICES=1 nohup /home/darren/miniconda3/bin/vf-vllm \
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
echo "Starting traditional forms training on GPU 0..."
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ABIDE_TRADITIONAL=1  # Flag for train_grpo.py to use traditional forms
CUDA_VISIBLE_DEVICES=0 /home/darren/miniconda3/bin/torchrun --nproc_per_node=1 scripts/train_grpo.py \
    --prompts $NUM_PROMPTS \
    --output models/abide_grpo_traditional

# Cleanup
echo ""
echo "Training finished. Cleaning up..."
kill $VLLM_PID 2>/dev/null || true
echo "Done."
