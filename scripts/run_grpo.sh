#!/bin/bash
set -e

# Legacy verifiers GRPO training for Abide poetry forms.
# This is the Gemma 4 E4B path for the smaller local verifiers RL trainer.

# Configuration
MODEL="${ABIDE_MODEL:-google/gemma-4-E4B-it}"
FORM_SET="${ABIDE_FORM_SET:-well_known}"
OUTPUT_DIR="${ABIDE_OUTPUT_DIR:-models/abide_verifiers_gemma4_e4b_well_known}"
PORT="${ABIDE_PORT:-8000}"
VLLM_PID=""
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERIFIERS_VENV="${ABIDE_VERIFIERS_VENV:-${REPO_ROOT}/.venv-verifiers}"
TRAIN_GPU="${ABIDE_TRAIN_GPU:-}"
VLLM_GPU="${ABIDE_VLLM_GPU:-}"
TRAIN_MIN_FREE_MIB="${ABIDE_TRAIN_MIN_FREE_MIB:-8192}"
VLLM_MIN_FREE_MIB="${ABIDE_VLLM_MIN_FREE_MIB:-18000}"
cd "$REPO_ROOT"

gpu_free_memory_mib() {
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | awk -F', ' -v gpu="$1" '$1 == gpu { print $2; exit }'
}

pick_default_gpus() {
    local -a candidates
    mapfile -t candidates < <(
        nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
            | sort -t, -k2 -nr
    )

    if [ "${#candidates[@]}" -lt 2 ]; then
        echo "ERROR: need at least two visible GPUs for the legacy Gemma 4 runner." >&2
        exit 1
    fi

    if [ -z "$TRAIN_GPU" ]; then
        TRAIN_GPU="$(echo "${candidates[0]}" | awk -F', ' '{print $1}')"
    fi

    if [ -z "$VLLM_GPU" ]; then
        for candidate in "${candidates[@]}"; do
            local gpu
            gpu="$(echo "$candidate" | awk -F', ' '{print $1}')"
            if [ "$gpu" != "$TRAIN_GPU" ]; then
                VLLM_GPU="$gpu"
                break
            fi
        done
    fi
}

ensure_gpu_ready() {
    local label="$1"
    local gpu="$2"
    local min_free_mib="$3"
    local free_mib

    free_mib="$(gpu_free_memory_mib "$gpu")"
    if [ -z "$free_mib" ]; then
        echo "ERROR: unable to determine free memory for GPU ${gpu}." >&2
        exit 1
    fi
    if [ "$free_mib" -lt "$min_free_mib" ]; then
        echo "ERROR: ${label} GPU ${gpu} only has ${free_mib} MiB free; need at least ${min_free_mib} MiB." >&2
        exit 1
    fi
}

# Cleanup on ctrl-c
cleanup() {
    echo ""
    echo "Interrupted. Cleaning up..."
    [ -n "$VLLM_PID" ] && kill $VLLM_PID 2>/dev/null
    pkill -f "vf-vllm" 2>/dev/null || true
    pkill -f "abide.verifiers_vllm_server" 2>/dev/null || true
    exit 1
}
trap cleanup INT TERM

pick_default_gpus

if [ "$TRAIN_GPU" = "$VLLM_GPU" ]; then
    echo "ERROR: training GPU and vLLM GPU must be different (both resolved to ${TRAIN_GPU})." >&2
    exit 1
fi

ensure_gpu_ready "Training" "$TRAIN_GPU" "$TRAIN_MIN_FREE_MIB"
ensure_gpu_ready "vLLM" "$VLLM_GPU" "$VLLM_MIN_FREE_MIB"

echo "============================================================"
echo "Abide Verifiers GRPO Training"
echo "============================================================"
echo "Model: $MODEL"
echo "Form set: $FORM_SET"
echo "vLLM: GPU $VLLM_GPU, port $PORT"
echo "Training: GPU $TRAIN_GPU"
echo "============================================================"
echo "Runtime: ${VERIFIERS_VENV}"

# Cleanup old processes
echo "Cleaning up old vLLM processes..."
pkill -f "vf-vllm" || true
pkill -f "vllm.entrypoints" || true
pkill -f "abide.verifiers_vllm_server" || true
sleep 2

"${REPO_ROOT}/scripts/prepare_verifiers_runtime.sh"

# Create log directory
mkdir -p logs

# Start Gemma 4-capable vLLM server
echo "Starting Gemma 4-capable vLLM server on GPU ${VLLM_GPU}..."
CUDA_VISIBLE_DEVICES="${VLLM_GPU}" nohup "${VERIFIERS_VENV}/bin/python" -m abide.verifiers_vllm_server \
    --model "$MODEL" \
    --port $PORT \
    --gpu-memory-utilization 0.80 \
    --tensor-parallel-size 1 \
    --max-model-len 1024 \
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

# Run training
echo ""
echo "Starting training on GPU ${TRAIN_GPU}..."
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" "${VERIFIERS_VENV}/bin/python" scripts/train_grpo.py \
    --model "$MODEL" \
    --form-set "$FORM_SET" \
    --output "$OUTPUT_DIR" \
    --port "$PORT"

# Cleanup
echo ""
echo "Training finished. Cleaning up..."
kill $VLLM_PID 2>/dev/null || true
echo "Done."
