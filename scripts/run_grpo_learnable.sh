#!/bin/bash
set -e

# GRPO Training on Top 10 Learnable Forms
#
# These forms have the highest within-rollout variance, making them ideal for GRPO:
# - Epigram, ThunderVerse, ColorSpectrum, CoprimeVerse, ElementalVerse
# - CharacterPalindromePoem, QuestionQuest, VowelPilgrimage, Mesostic, Terzanelle
#
# From find_learnable_forms.py analysis:
# - within_prompt_std > 0.1 (model shows variance on same prompt)
# - 0.2 < mean_reward < 0.8 (not too easy or too hard)
# - This creates meaningful advantage signal for GRPO learning

# Configuration
MODEL="google/gemma-3-4b-it"
PORT=8000
VLLM_PID=""
NUM_PROMPTS=50000   # 5000 prompts per form (10 forms), longer training
MAX_STEPS=3125      # 50000 / 16 batch_size = 3125 steps

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
echo "GRPO Training: Top 10 Learnable Forms"
echo "============================================================"
echo "Model: $MODEL"
echo "Prompts: $NUM_PROMPTS (5000 per form)"
echo "Max Steps: ~$MAX_STEPS"
echo ""
echo "Forms (sorted by GRPO signal):"
echo "  1. Epigram (signal=0.332)"
echo "  2. ThunderVerse (signal=0.280)"
echo "  3. ColorSpectrum (signal=0.250)"
echo "  4. CoprimeVerse (signal=0.238)"
echo "  5. ElementalVerse (signal=0.238)"
echo "  6. CharacterPalindromePoem (signal=0.228)"
echo "  7. QuestionQuest (signal=0.218)"
echo "  8. VowelPilgrimage (signal=0.208)"
echo "  9. Mesostic (signal=0.207)"
echo " 10. Terzanelle (signal=0.191)"
echo "============================================================"
echo ""
echo "Hypothesis: Training on forms with high within-rollout variance"
echo "should produce meaningful learning, unlike traditional forms where"
echo "the model either always succeeds or always fails."
echo "============================================================"

# Cleanup old processes
echo "Cleaning up old vLLM processes..."
pkill -f "vf-vllm" || true
pkill -f "vllm.entrypoints" || true
sleep 2

# Create log directory
mkdir -p logs models/grpo_learnable_lr5e5

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
    > logs/vllm_learnable.log 2>&1 &

VLLM_PID=$!
echo "vLLM started with PID $VLLM_PID (ctrl-c to abort)"

# Wait for vLLM to be ready
echo "Waiting for vLLM to be ready..."
timeout 300 bash -c "until curl -s localhost:$PORT/health > /dev/null 2>&1; do sleep 2; done" || {
    echo "ERROR: vLLM failed to start. Check logs/vllm_learnable.log"
    cat logs/vllm_learnable.log | tail -50
    exit 1
}
echo "vLLM is ready!"

# Run training on GPU 0 with learnable forms
echo ""
echo "Starting GRPO training on GPU 0..."
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ABIDE_LEARNABLE=1
export ABIDE_MODEL="$MODEL"

# Find latest checkpoint for resume
RESUME_ARG=""
LATEST_CKPT=$(ls -d models/grpo_learnable_lr5e5/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
if [ -n "$LATEST_CKPT" ]; then
    echo "Resuming from $LATEST_CKPT"
    RESUME_ARG="--resume $LATEST_CKPT"
fi

CUDA_VISIBLE_DEVICES=0 /home/darren/miniconda3/bin/torchrun --nproc_per_node=1 scripts/train_grpo.py \
    --prompts $NUM_PROMPTS \
    --batch-size 16 \
    --rollouts 16 \
    --output models/grpo_learnable_lr5e5 \
    --save-steps 50 \
    $RESUME_ARG

# Cleanup
echo ""
echo "Training finished. Cleaning up..."
kill $VLLM_PID 2>/dev/null || true

echo ""
echo "============================================================"
echo "LEARNABLE FORMS TRAINING COMPLETE"
echo "============================================================"
echo "Check wandb for reward curves and entropy metrics."
echo "Compare initial vs final reward to measure learning."
echo "============================================================"
