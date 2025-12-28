#!/bin/bash
set -e

# TRL GRPO Training with KL Regularization
#
# This uses TRL's GRPOTrainer instead of verifiers library.
# Key difference: includes beta (KL coefficient) to prevent policy collapse.
#
# TRL uses its own vLLM server integration via `trl vllm-serve`

# Configuration
MODEL="google/gemma-3-4b-it"
PORT=8000
VLLM_PID=""
NUM_PROMPTS=50000
BETA=0.04  # KL coefficient (DeepSeek uses 0.001, some papers use 0.04)

# Cleanup on ctrl-c
cleanup() {
    echo ""
    echo "Interrupted. Cleaning up..."
    [ -n "$VLLM_PID" ] && kill $VLLM_PID 2>/dev/null
    pkill -f "trl vllm-serve" 2>/dev/null || true
    exit 1
}
trap cleanup INT TERM

echo "============================================================"
echo "TRL GRPO Training with KL Regularization"
echo "============================================================"
echo "Model: $MODEL"
echo "Prompts: $NUM_PROMPTS"
echo "Beta (KL coef): $BETA"
echo ""
echo "Key advantage over verifiers library:"
echo "  - Includes beta parameter for KL divergence regularization"
echo "  - Prevents policy collapse (entropy collapse, KL explosion)"
echo "============================================================"
echo ""

# Cleanup old processes
echo "Cleaning up old vLLM/TRL processes..."
pkill -f "trl vllm-serve" || true
pkill -f "vf-vllm" || true
pkill -f "vllm.entrypoints" || true
sleep 2

# Create log directory
mkdir -p logs models/grpo_trl

# Start TRL vLLM server on GPU 1
echo "Starting TRL vLLM server on GPU 1..."
CUDA_VISIBLE_DEVICES=1 nohup /home/darren/miniconda3/bin/trl vllm-serve \
    --model "$MODEL" \
    --port $PORT \
    --gpu_memory_utilization 0.92 \
    --tensor_parallel_size 1 \
    --max_model_len 4096 \
    > logs/vllm_trl.log 2>&1 &

VLLM_PID=$!
echo "TRL vLLM server started with PID $VLLM_PID (ctrl-c to abort)"

# Wait for vLLM to be ready
echo "Waiting for vLLM server to be ready..."
timeout 300 bash -c "until curl -s localhost:$PORT/health > /dev/null 2>&1; do sleep 2; done" || {
    echo "ERROR: vLLM failed to start. Check logs/vllm_trl.log"
    cat logs/vllm_trl.log | tail -50
    exit 1
}
echo "vLLM server is ready!"

# Run TRL GRPO training on GPU 0
echo ""
echo "Starting TRL GRPO training on GPU 0..."
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ABIDE_LEARNABLE=1
export ABIDE_MODEL="$MODEL"

CUDA_VISIBLE_DEVICES=0 python scripts/train_grpo_trl.py \
    --prompts $NUM_PROMPTS \
    --batch-size 16 \
    --num-generations 16 \
    --beta $BETA \
    --lr 5e-5 \
    --output models/grpo_trl \
    --save-steps 50

# Cleanup
echo ""
echo "Training finished. Cleaning up..."
kill $VLLM_PID 2>/dev/null || true

echo ""
echo "============================================================"
echo "TRL GRPO TRAINING COMPLETE"
echo "============================================================"
echo "Check wandb for reward curves and KL metrics."
echo "With beta=$BETA, KL divergence should stay bounded."
echo "============================================================"
