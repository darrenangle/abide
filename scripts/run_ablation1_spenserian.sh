#!/bin/bash
set -e

# ABLATION 1: Single Form Training (Spenserian Sonnet Only)
#
# Hypothesis: Training on a single complex form will show if learning is possible
# at all with the verifiers library's GRPO implementation.
#
# From grpo_analysis.md:
#   "Simplify to see if learning is possible at all"
#   "Success Metric: Reward increase > 0.1 over 500 steps"
#
# Using Gemma 3 4B as the base model (same as interrupted run)

# Configuration
MODEL="google/gemma-3-4b-it"
PORT=8000
VLLM_PID=""
NUM_PROMPTS=5000  # 5000 prompts for single form (plenty of variety with topics)
MAX_STEPS=500    # Run for 500 steps as per ablation plan

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
echo "ABLATION 1: Single Form Training (Spenserian Sonnet)"
echo "============================================================"
echo "Model: $MODEL"
echo "Form: SpenserianSonnet ONLY"
echo "Prompts: $NUM_PROMPTS"
echo "Max Steps: $MAX_STEPS"
echo "Success Metric: Reward increase > 0.1"
echo "============================================================"
echo ""
echo "Hypothesis: If the model can learn ANY form with GRPO,"
echo "it should show improvement on a single focused task."
echo "============================================================"

# Cleanup old processes
echo "Cleaning up old vLLM processes..."
pkill -f "vf-vllm" || true
pkill -f "vllm.entrypoints" || true
sleep 2

# Create log directory
mkdir -p logs experiments

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
    > logs/vllm_ablation1.log 2>&1 &

VLLM_PID=$!
echo "vLLM started with PID $VLLM_PID (ctrl-c to abort)"

# Wait for vLLM to be ready
echo "Waiting for vLLM to be ready..."
timeout 300 bash -c "until curl -s localhost:$PORT/health > /dev/null 2>&1; do sleep 2; done" || {
    echo "ERROR: vLLM failed to start. Check logs/vllm_ablation1.log"
    cat logs/vllm_ablation1.log | tail -50
    exit 1
}
echo "vLLM is ready!"

# Run training on GPU 0 with single form mode
echo ""
echo "Starting Ablation 1 training on GPU 0..."
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ABIDE_SINGLE_FORM="SpenserianSonnet"  # Single form mode!
export ABIDE_MODEL="$MODEL"

# Calculate batch size to get ~500 steps: 5000 / batch_size = 500 => batch_size = 10
# But we want to keep batch_size=32 for stable gradients, so we'll run 156 steps
# Actually let's use smaller batch for more gradient updates
BATCH_SIZE=16  # 5000 / 16 = 312 steps, close enough for 500-step target

CUDA_VISIBLE_DEVICES=0 /home/darren/miniconda3/bin/torchrun --nproc_per_node=1 scripts/train_grpo.py \
    --prompts $NUM_PROMPTS \
    --batch-size $BATCH_SIZE \
    --output models/ablation1_spenserian \
    --save-steps 50

# Cleanup
echo ""
echo "Training finished. Cleaning up..."
kill $VLLM_PID 2>/dev/null || true

echo ""
echo "============================================================"
echo "ABLATION 1 COMPLETE"
echo "============================================================"
echo "Check wandb for reward curves and entropy metrics."
echo "Compare initial reward to final reward."
echo "Success if: reward increases by > 0.1"
echo "============================================================"
