#!/bin/bash
set -e

# GRPO Training for Gemma 3n E2B-it
# Gemma 3n uses MatFormer architecture with 2B effective parameters (6B raw)
# Should have better poetry exposure than OLMo from Google's training data
#
# Memory notes:
# - Gemma 3n E2B takes ~6-8GB for weights in bfloat16
# - Plenty of room for KV cache and training on 24GB GPU
# - E4B was too large (OOM during backward pass)

# Configuration
MODEL="google/gemma-3n-E2B-it"
PORT=8000
VLLM_PID=""
NUM_PROMPTS="${ABIDE_NUM_PROMPTS:-100000}"

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
echo "Abide GRPO Training - GEMMA 3n E2B-it"
echo "============================================================"
echo "Model: $MODEL (2B effective params, MatFormer architecture)"
echo "Prompts: $NUM_PROMPTS (traditional forms, weighted sampling)"
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
# E2B is smaller so we can use higher memory utilization
# enforce-eager for stability with new architecture
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
    > logs/vllm_gemma3n.log 2>&1 &

VLLM_PID=$!
echo "vLLM started with PID $VLLM_PID (ctrl-c to abort)"

# Wait for vLLM to be ready (Gemma 3n takes longer to load)
echo "Waiting for vLLM to be ready (Gemma 3n takes ~2-3 min to load)..."
timeout 600 bash -c "until curl -s localhost:$PORT/health > /dev/null 2>&1; do sleep 5; done" || {
    echo "ERROR: vLLM failed to start. Check logs/vllm_gemma3n.log"
    cat logs/vllm_gemma3n.log | tail -100
    exit 1
}
echo "vLLM is ready!"

# Run training on GPU 0
echo ""
echo "Starting Gemma 3n training on GPU 0..."
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ABIDE_TRADITIONAL=1  # Use traditional forms only
export ABIDE_MODEL="$MODEL"
CUDA_VISIBLE_DEVICES=0 /home/darren/miniconda3/bin/torchrun --nproc_per_node=1 scripts/train_grpo.py \
    --prompts $NUM_PROMPTS \
    --output models/abide_grpo_gemma3n

# Cleanup
echo ""
echo "Training finished. Cleaning up..."
kill $VLLM_PID 2>/dev/null || true
echo "Done."
