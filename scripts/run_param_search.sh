#!/bin/bash
set -e

# GRPO Parameter Search Runner
# Runs all experiments, finds best params, confirms, and starts training

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

MODEL="/home/darren/10k-poems/models/baguettotron_sft/final"
PORT=8000
PYTHON="/home/darren/miniconda3/bin/python"
LOG_FILE="experiments/grpo_param_search.md"
RESULTS_FILE="experiments/param_search_results.jsonl"

echo "============================================================"
echo "GRPO Parameter Search for Baguettotron"
echo "============================================================"
echo "Project: $PROJECT_DIR"
echo "Model: $MODEL"
echo "Started: $(date)"
echo "============================================================"

# Cleanup function
cleanup() {
    echo ""
    echo "Cleaning up..."
    pkill -f "vf-vllm.*$MODEL" 2>/dev/null || true
}
trap cleanup EXIT

# Kill any existing vLLM
echo "Cleaning up old vLLM processes..."
pkill -f "vf-vllm" 2>/dev/null || true
pkill -f "vllm.entrypoints" 2>/dev/null || true
sleep 3

# Clear old results
rm -f "$RESULTS_FILE"
mkdir -p experiments logs

# Start vLLM on GPU 1 with Baguettotron
echo "Starting vLLM on GPU 1..."
CUDA_VISIBLE_DEVICES=1 /home/darren/miniconda3/bin/vf-vllm \
    --model "$MODEL" \
    --port $PORT \
    --gpu-memory-utilization 0.7 \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --trust-remote-code \
    --disable-log-stats \
    --enforce-eager \
    > logs/vllm_param_search.log 2>&1 &

VLLM_PID=$!
echo "vLLM started with PID $VLLM_PID"

# Wait for vLLM to be ready
echo "Waiting for vLLM to be ready..."
for i in {1..60}; do
    if curl -s localhost:$PORT/health > /dev/null 2>&1; then
        echo "vLLM is ready! (${i}0s)"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "ERROR: vLLM failed to start. Check logs/vllm_param_search.log"
        tail -50 logs/vllm_param_search.log
        exit 1
    fi
    sleep 10
done

# Define experiments
declare -a EXPERIMENTS=(
    "1024 1.0"   # Exp 1: baseline
    "1024 1.15"  # Exp 2: moderate penalty
    "1024 1.3"   # Exp 3: high penalty
    "2048 1.0"   # Exp 4: more tokens, no penalty
    "2048 1.15"  # Exp 5: more tokens, moderate
    "2048 1.3"   # Exp 6: more tokens, high
    "512 1.15"   # Exp 7: fewer tokens
    "2048 1.2"   # Exp 8: more tokens, between moderate and high
)

echo ""
echo "============================================================"
echo "RUNNING ${#EXPERIMENTS[@]} EXPERIMENTS"
echo "============================================================"

EXP_NUM=1
for exp in "${EXPERIMENTS[@]}"; do
    read -r MAX_TOKENS REP_PENALTY <<< "$exp"
    echo ""
    echo ">>> Experiment $EXP_NUM: max_tokens=$MAX_TOKENS, rep_penalty=$REP_PENALTY"

    CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/param_search.py \
        --max-tokens "$MAX_TOKENS" \
        --rep-penalty "$REP_PENALTY" \
        --num-rollouts 128 \
        --output "$RESULTS_FILE" 2>&1 | tee -a logs/param_search_exp${EXP_NUM}.log

    EXP_NUM=$((EXP_NUM + 1))
done

echo ""
echo "============================================================"
echo "ANALYZING RESULTS"
echo "============================================================"

# Find best params using Python
BEST_PARAMS=$($PYTHON -c "
import json
results = []
with open('$RESULTS_FILE') as f:
    for line in f:
        results.append(json.loads(line))

# Sort by mean_reward descending
results.sort(key=lambda x: x['mean_reward'], reverse=True)

best = results[0]
print(f\"{best['max_tokens']} {best['repetition_penalty']} {best['mean_reward']:.4f}\")

print('\\nAll results (sorted by reward):')
for r in results:
    print(f\"  max_tokens={r['max_tokens']}, rep_penalty={r['repetition_penalty']}: reward={r['mean_reward']:.4f} ({r['nonzero_pct']:.1f}% nonzero)\")
")

read -r BEST_MAX_TOKENS BEST_REP_PENALTY BEST_REWARD <<< "$(echo "$BEST_PARAMS" | head -1)"

echo ""
echo "BEST PARAMS: max_tokens=$BEST_MAX_TOKENS, rep_penalty=$BEST_REP_PENALTY, reward=$BEST_REWARD"

echo ""
echo "============================================================"
echo "CONFIRMATION RUNS (3x with best params)"
echo "============================================================"

for i in 1 2 3; do
    echo ""
    echo ">>> Confirmation run $i/3: max_tokens=$BEST_MAX_TOKENS, rep_penalty=$BEST_REP_PENALTY"

    CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/param_search.py \
        --max-tokens "$BEST_MAX_TOKENS" \
        --rep-penalty "$BEST_REP_PENALTY" \
        --num-rollouts 128 \
        --output "experiments/confirmation_results.jsonl" 2>&1 | tee -a logs/confirmation_${i}.log
done

# Analyze confirmation runs
echo ""
echo "============================================================"
echo "CONFIRMATION ANALYSIS"
echo "============================================================"

$PYTHON -c "
import json
results = []
with open('experiments/confirmation_results.jsonl') as f:
    for line in f:
        results.append(json.loads(line))

rewards = [r['mean_reward'] for r in results]
mean = sum(rewards) / len(rewards)
std = (sum((r - mean)**2 for r in rewards) / len(rewards)) ** 0.5

print(f'Confirmation runs: {len(results)}')
print(f'Mean reward: {mean:.4f} +/- {std:.4f}')
print(f'Individual: {[f\"{r:.4f}\" for r in rewards]}')
print(f'Consistent: {\"YES\" if std < 0.05 else \"NO (high variance)\"}')
"

echo ""
echo "============================================================"
echo "UPDATING train_grpo.py WITH BEST PARAMS"
echo "============================================================"

# Update max_tokens in train_grpo.py
sed -i "s/max_tokens=[0-9]*/max_tokens=$BEST_MAX_TOKENS/" scripts/train_grpo.py
sed -i "s/repetition_penalty=[0-9.]*/repetition_penalty=$BEST_REP_PENALTY/" scripts/train_grpo.py

echo "Updated train_grpo.py:"
grep -E "(max_tokens|repetition_penalty)" scripts/train_grpo.py | head -4

# Update experiment log
echo ""
echo "============================================================"
echo "UPDATING EXPERIMENT LOG"
echo "============================================================"

# Append final results to log
cat >> "$LOG_FILE" << EOF

---

## Final Results ($(date))

### Best Parameters Found
- **max_tokens**: $BEST_MAX_TOKENS
- **repetition_penalty**: $BEST_REP_PENALTY
- **mean_reward**: $BEST_REWARD

### All Experiment Results
\`\`\`
$(cat "$RESULTS_FILE" | $PYTHON -c "
import json, sys
for line in sys.stdin:
    r = json.loads(line)
    print(f\"max_tokens={r['max_tokens']}, rep_penalty={r['repetition_penalty']}: reward={r['mean_reward']:.4f} ({r['nonzero_pct']:.1f}% nonzero)\")
")
\`\`\`

### Confirmation Results
\`\`\`
$(cat experiments/confirmation_results.jsonl 2>/dev/null | $PYTHON -c "
import json, sys
for i, line in enumerate(sys.stdin, 1):
    r = json.loads(line)
    print(f\"Run {i}: reward={r['mean_reward']:.4f} ({r['nonzero_pct']:.1f}% nonzero)\")
" 2>/dev/null || echo "No confirmation results")
\`\`\`

### Applied to train_grpo.py
Parameters have been updated in the training script.
EOF

echo "Experiment log updated: $LOG_FILE"

# Kill vLLM before training (will restart with correct params)
echo ""
echo "Stopping param search vLLM..."
kill $VLLM_PID 2>/dev/null || true
sleep 5

echo ""
echo "============================================================"
echo "STARTING TRAINING IN BACKGROUND"
echo "============================================================"

# Update run_grpo.sh max-model-len to match
sed -i "s/--max-model-len [0-9]*/--max-model-len $(($BEST_MAX_TOKENS + 512))/" scripts/run_grpo.sh

echo "Starting training via run_grpo.sh..."
nohup bash scripts/run_grpo.sh > logs/training_$(date +%Y%m%d_%H%M%S).log 2>&1 &
TRAIN_PID=$!

echo ""
echo "============================================================"
echo "TRAINING STARTED"
echo "============================================================"
echo "Training PID: $TRAIN_PID"
echo "Log file: logs/training_*.log"
echo "Monitor with: tail -f logs/training_*.log"
echo ""
echo "Parameter search complete!"
echo "Best params: max_tokens=$BEST_MAX_TOKENS, rep_penalty=$BEST_REP_PENALTY"
echo "============================================================"
