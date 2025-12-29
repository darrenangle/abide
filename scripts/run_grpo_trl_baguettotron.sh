#!/bin/bash
set -e

# TRL GRPO Training for Baguettotron
#
# Baguettotron is a reasoning model with <think>...</think> traces.
# Inference parameters were tuned via param_search.py (Dec 2025):
#   - max_tokens=2048 (reasoning traces need space)
#   - repetition_penalty=1.2 (prevents loops without killing output)
#   - temperature=0.6, top_p=0.95
#
# Training uses TRL's GRPOTrainer with:
#   - DAPO loss (reduces length bias)
#   - KL regularization (beta=0.04, prevents policy collapse)

# ========================================
# BAGUETTOTRON INFERENCE PARAMETERS
# (tuned via scripts/param_search.py)
# ========================================
MODEL="/home/darren/10k-poems/models/baguettotron_sft/final"
MAX_TOKENS=2048
REP_PENALTY=1.2
TEMPERATURE=0.6
TOP_P=0.95

# ========================================
# TRAINING PARAMETERS
# ========================================
PORT=8000
NUM_PROMPTS=10000  # Reduced - Gemma converged in ~400 steps
BETA=0.01  # Lowered from 0.04 to allow more divergence
BATCH_SIZE=16
NUM_GENERATIONS=16
LR=5e-5

# ========================================
# WORKFLOW FLAGS
# ========================================
RUN_PARAM_SEARCH=0
RUN_LEARNABLE_SEARCH=0
RUN_TRAINING=1

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --param-search)
            RUN_PARAM_SEARCH=1
            RUN_TRAINING=0
            shift
            ;;
        --find-learnable)
            RUN_LEARNABLE_SEARCH=1
            RUN_TRAINING=0
            shift
            ;;
        --train)
            RUN_TRAINING=1
            shift
            ;;
        --all)
            RUN_PARAM_SEARCH=1
            RUN_LEARNABLE_SEARCH=1
            RUN_TRAINING=1
            shift
            ;;
        *)
            echo "Usage: $0 [--param-search] [--find-learnable] [--train] [--all]"
            echo ""
            echo "Options:"
            echo "  --param-search    Tune max_tokens, rep_penalty (already done, see experiments/)"
            echo "  --find-learnable  Find forms with high within-rollout variance"
            echo "  --train           Run TRL GRPO training (default)"
            echo "  --all             Run all steps in sequence"
            exit 1
            ;;
    esac
done

VLLM_PID=""

# Cleanup on ctrl-c
cleanup() {
    echo ""
    echo "Interrupted. Cleaning up..."
    [ -n "$VLLM_PID" ] && kill $VLLM_PID 2>/dev/null
    pkill -f "trl vllm-serve" 2>/dev/null || true
    pkill -f "vf-vllm" 2>/dev/null || true
    exit 1
}
trap cleanup INT TERM

echo "============================================================"
echo "Baguettotron GRPO Pipeline"
echo "============================================================"
echo "Model: $MODEL"
echo ""
echo "Inference params (tuned):"
echo "  max_tokens=$MAX_TOKENS, rep_penalty=$REP_PENALTY"
echo "  temperature=$TEMPERATURE, top_p=$TOP_P"
echo ""
echo "Training params:"
echo "  beta=$BETA (KL), batch=$BATCH_SIZE, gens=$NUM_GENERATIONS"
echo ""
echo "Steps: param_search=$RUN_PARAM_SEARCH, learnable=$RUN_LEARNABLE_SEARCH, train=$RUN_TRAINING"
echo "============================================================"

# Create directories
mkdir -p logs models/grpo_trl_baguettotron experiments

# ========================================
# PHASE 1: Parameter Search (optional)
# ========================================
if [ "$RUN_PARAM_SEARCH" -eq 1 ]; then
    echo ""
    echo "============================================================"
    echo "PHASE 1: Inference Parameter Search"
    echo "============================================================"
    echo "Testing max_tokens and repetition_penalty combinations..."
    echo ""

    # Cleanup old processes
    pkill -f "vf-vllm" || true
    pkill -f "vllm.entrypoints" || true
    sleep 2

    # Start vLLM for param search (verifiers library)
    echo "Starting vf-vllm server..."
    CUDA_VISIBLE_DEVICES=1 nohup /home/darren/miniconda3/bin/vf-vllm \
        --model "$MODEL" \
        --port $PORT \
        --max-model-len 4096 \
        --enforce-eager \
        > logs/vllm_param_search.log 2>&1 &

    VLLM_PID=$!
    echo "vLLM PID: $VLLM_PID"

    # Wait for ready
    echo "Waiting for vLLM..."
    timeout 300 bash -c "until curl -s localhost:$PORT/health > /dev/null 2>&1; do sleep 2; done" || {
        echo "ERROR: vLLM failed. Check logs/vllm_param_search.log"
        exit 1
    }

    # Run parameter search
    for max_tokens in 1024 2048 3072; do
        for rep_penalty in 1.1 1.15 1.2 1.25 1.3; do
            echo ""
            echo ">>> Testing max_tokens=$max_tokens, rep_penalty=$rep_penalty"
            python scripts/param_search.py \
                --max-tokens $max_tokens \
                --rep-penalty $rep_penalty \
                --num-rollouts 128 \
                --output experiments/baguettotron_param_search.jsonl
        done
    done

    echo ""
    echo "Parameter search complete. Results in experiments/baguettotron_param_search.jsonl"

    # Cleanup
    kill $VLLM_PID 2>/dev/null || true
    VLLM_PID=""

    if [ "$RUN_LEARNABLE_SEARCH" -eq 0 ] && [ "$RUN_TRAINING" -eq 0 ]; then
        exit 0
    fi
fi

# ========================================
# PHASE 2: Find Learnable Forms (optional)
# ========================================
if [ "$RUN_LEARNABLE_SEARCH" -eq 1 ]; then
    echo ""
    echo "============================================================"
    echo "PHASE 2: Finding Learnable Forms"
    echo "============================================================"
    echo "Testing all 140+ forms to find ones with high GRPO signal..."
    echo ""

    # Cleanup and start fresh vLLM
    pkill -f "trl vllm-serve" || true
    pkill -f "vf-vllm" || true
    pkill -f "vllm.entrypoints" || true
    sleep 2

    # Use standard vLLM (OpenAI-compatible API) for learnable forms search
    # TRL's vLLM server has a different API that doesn't support /v1/completions
    echo "Starting standard vLLM server (OpenAI-compatible)..."
    CUDA_VISIBLE_DEVICES=1 nohup python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --port $PORT \
        --gpu-memory-utilization 0.92 \
        --max-model-len 4096 \
        --trust-remote-code \
        > logs/vllm_learnable_search.log 2>&1 &

    VLLM_PID=$!

    # Wait for server with retry loop
    echo "Waiting for vLLM server..."
    for i in {1..60}; do
        if curl -s "localhost:$PORT/v1/models" > /dev/null 2>&1; then
            echo "vLLM server ready!"
            break
        fi
        if [ $i -eq 60 ]; then
            echo "ERROR: vLLM failed to start. Check logs/vllm_learnable_search.log"
            cat logs/vllm_learnable_search.log | tail -50
            exit 1
        fi
        sleep 5
    done

    # Run learnable forms search
    python scripts/find_learnable_forms.py \
        --model "$MODEL" \
        --vllm-url "http://localhost:$PORT" \
        --prompts 5 \
        --rollouts 8 \
        --output experiments/baguettotron_form_variance.json

    echo ""
    echo "Learnable forms saved to experiments/baguettotron_form_variance.json"

    # Cleanup
    kill $VLLM_PID 2>/dev/null || true
    VLLM_PID=""

    if [ "$RUN_TRAINING" -eq 0 ]; then
        exit 0
    fi
fi

# ========================================
# PHASE 3: TRL GRPO Training
# ========================================
if [ "$RUN_TRAINING" -eq 1 ]; then
    echo ""
    echo "============================================================"
    echo "PHASE 3: TRL GRPO Training"
    echo "============================================================"

    # Cleanup old processes
    pkill -f "trl vllm-serve" || true
    pkill -f "vf-vllm" || true
    pkill -f "vllm.entrypoints" || true
    sleep 2

    # Start TRL vLLM server
    echo "Starting TRL vLLM server on GPU 1..."
    CUDA_VISIBLE_DEVICES=1 nohup /home/darren/miniconda3/bin/trl vllm-serve \
        --model "$MODEL" \
        --port $PORT \
        --gpu_memory_utilization 0.92 \
        --tensor_parallel_size 1 \
        --max_model_len 4096 \
        > logs/vllm_trl_baguettotron.log 2>&1 &

    VLLM_PID=$!
    echo "vLLM PID: $VLLM_PID"

    timeout 300 bash -c "until curl -s localhost:$PORT/health > /dev/null 2>&1; do sleep 2; done" || {
        echo "ERROR: vLLM failed. Check logs/vllm_trl_baguettotron.log"
        exit 1
    }
    echo "vLLM ready!"

    # Run training
    export OMP_NUM_THREADS=4
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export ABIDE_MODEL="$MODEL"

    echo ""
    echo "Starting TRL GRPO training on GPU 0..."
    CUDA_VISIBLE_DEVICES=0 python scripts/train_grpo_baguettotron.py \
        --prompts $NUM_PROMPTS \
        --batch-size $BATCH_SIZE \
        --num-generations $NUM_GENERATIONS \
        --beta $BETA \
        --lr $LR \
        --output models/grpo_trl_baguettotron \
        --save-steps 50 \
        --port $PORT

    # Cleanup
    kill $VLLM_PID 2>/dev/null || true

    echo ""
    echo "============================================================"
    echo "BAGUETTOTRON GRPO TRAINING COMPLETE"
    echo "============================================================"
    echo "Model: models/grpo_trl_baguettotron/final"
    echo "============================================================"
fi
