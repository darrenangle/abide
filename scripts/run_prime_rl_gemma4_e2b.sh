#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL="${ABIDE_MODEL:-google/gemma-4-E2B-it}"
FORM_SET="${ABIDE_FORM_SET:-well_known}"
OUTPUT_DIR="${ABIDE_OUTPUT_DIR:-models/prime_rl_gemma4_e2b_well_known}"
RUN_PROFILE="${ABIDE_RUN_PROFILE:-smoke}"
PORT="${ABIDE_PORT:-8000}"
PRIME_RL_VENV="${ABIDE_PRIME_RL_VENV:-${REPO_ROOT}/.venv-prime-rl}"
TRAIN_GPU="${ABIDE_TRAIN_GPU:-}"
INFER_GPU="${ABIDE_INFER_GPU:-}"
TRAIN_MIN_FREE_MIB="${ABIDE_TRAIN_MIN_FREE_MIB:-12000}"
INFER_MIN_FREE_MIB="${ABIDE_INFER_MIN_FREE_MIB:-10000}"

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
        echo "ERROR: need at least two visible GPUs for prime-rl local RL." >&2
        exit 1
    fi

    if [ -z "$INFER_GPU" ]; then
        INFER_GPU="$(echo "${candidates[0]}" | awk -F', ' '{print $1}')"
    fi

    if [ -z "$TRAIN_GPU" ]; then
        for candidate in "${candidates[@]}"; do
            local gpu
            gpu="$(echo "$candidate" | awk -F', ' '{print $1}')"
            if [ "$gpu" != "$INFER_GPU" ]; then
                TRAIN_GPU="$gpu"
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

profile_args() {
    case "$RUN_PROFILE" in
        smoke)
            echo "--max-steps 1 --max-async-level 1 --num-prompts 16 --batch-size 4 --rollouts-per-example 2 --max-tokens 96 --seq-len 640"
            ;;
        canary)
            echo "--max-steps 8 --num-prompts 1024 --batch-size 64 --rollouts-per-example 4 --max-tokens 384 --seq-len 1280"
            ;;
        soak)
            echo "--max-steps 32 --num-prompts 4096 --batch-size 96 --rollouts-per-example 8 --max-tokens 512 --seq-len 1536"
            ;;
        *)
            echo "ERROR: unknown ABIDE_RUN_PROFILE=${RUN_PROFILE}. Expected smoke, canary, or soak." >&2
            exit 1
            ;;
    esac
}

pick_default_gpus

if [ "$TRAIN_GPU" = "$INFER_GPU" ]; then
    echo "ERROR: training GPU and inference GPU must differ (both resolved to ${TRAIN_GPU})." >&2
    exit 1
fi

ensure_gpu_ready "Inference" "$INFER_GPU" "$INFER_MIN_FREE_MIB"
ensure_gpu_ready "Training" "$TRAIN_GPU" "$TRAIN_MIN_FREE_MIB"

echo "============================================================"
echo "Abide prime-rl Gemma 4 E2B"
echo "============================================================"
echo "Model: $MODEL"
echo "Form set: $FORM_SET"
echo "Profile: $RUN_PROFILE"
echo "Inference GPU: $INFER_GPU"
echo "Training GPU: $TRAIN_GPU"
echo "Runtime: $PRIME_RL_VENV"
echo "============================================================"

read -r -a EXTRA_PROFILE_ARGS <<< "$(profile_args)"
VISIBLE_GPUS="${INFER_GPU},${TRAIN_GPU}"

CUDA_VISIBLE_DEVICES="$VISIBLE_GPUS" python scripts/train_prime_rl.py \
    --model "$MODEL" \
    --form-set "$FORM_SET" \
    --output "$OUTPUT_DIR" \
    --port "$PORT" \
    --runtime-venv "$PRIME_RL_VENV" \
    "${EXTRA_PROFILE_ARGS[@]}" \
    "$@"
