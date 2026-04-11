#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL="${ABIDE_MODEL:-google/gemma-4-E2B-it}"
FORM_SET_OVERRIDE="${ABIDE_FORM_SET:-}"
OUTPUT_DIR_OVERRIDE="${ABIDE_OUTPUT_DIR:-}"
RUN_PROFILE="${ABIDE_RUN_PROFILE:-smoke}"
PORT="${ABIDE_PORT:-8000}"
PRIME_RL_VENV="${ABIDE_PRIME_RL_VENV:-${REPO_ROOT}/.venv-prime-rl}"
TRAIN_GPU="${ABIDE_TRAIN_GPU:-}"
INFER_GPU="${ABIDE_INFER_GPU:-}"
TRAIN_MIN_FREE_MIB="${ABIDE_TRAIN_MIN_FREE_MIB:-12000}"
INFER_MIN_FREE_MIB="${ABIDE_INFER_MIN_FREE_MIB:-10000}"
MAX_STEPS_OVERRIDE="${ABIDE_MAX_STEPS:-}"
MAX_ASYNC_LEVEL_OVERRIDE="${ABIDE_MAX_ASYNC_LEVEL:-}"
NUM_PROMPTS_OVERRIDE="${ABIDE_NUM_PROMPTS:-}"
SEED_OVERRIDE="${ABIDE_SEED:-}"
BATCH_SIZE_OVERRIDE="${ABIDE_BATCH_SIZE:-}"
ROLLOUTS_PER_EXAMPLE_OVERRIDE="${ABIDE_ROLLOUTS_PER_EXAMPLE:-}"
MAX_TOKENS_OVERRIDE="${ABIDE_MAX_TOKENS:-}"
SEQ_LEN_OVERRIDE="${ABIDE_SEQ_LEN:-}"
LEARNING_RATE_OVERRIDE="${ABIDE_LEARNING_RATE:-}"

gpu_free_memory_mib() {
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | awk -F', ' -v gpu="$1" '$1 == gpu { print $2; exit }'
}

canonical_profile_name() {
    case "$RUN_PROFILE" in
        canary)
            echo "mixed-canary"
            ;;
        soak)
            echo "mixed-soak"
            ;;
        *)
            echo "$RUN_PROFILE"
            ;;
    esac
}

profile_default_form_set() {
    case "$(canonical_profile_name)" in
        smoke|short-canary)
            echo "well_known_short"
            ;;
        long-canary)
            echo "well_known_long"
            ;;
        mixed-canary|mixed-stable|mixed-soak)
            echo "well_known"
            ;;
        *)
            echo "ERROR: unknown ABIDE_RUN_PROFILE=${RUN_PROFILE}. Expected smoke, short-canary, long-canary, mixed-canary, mixed-stable, canary, mixed-soak, or soak." >&2
            exit 1
            ;;
    esac
}

default_output_dir() {
    local resolved_profile

    resolved_profile="$(canonical_profile_name | tr '-' '_')"
    echo "models/prime_rl_gemma4_e2b_${FORM_SET}_${resolved_profile}"
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
    case "$(canonical_profile_name)" in
        smoke)
            echo "--max-steps 1 --max-async-level 0 --num-prompts 4 --batch-size 1 --rollouts-per-example 1 --max-tokens 128 --seq-len 512"
            ;;
        short-canary)
            echo "--max-steps 4 --max-async-level 0 --num-prompts 24 --batch-size 2 --rollouts-per-example 2 --max-tokens 128 --seq-len 512"
            ;;
        long-canary)
            echo "--max-steps 4 --max-async-level 0 --num-prompts 16 --batch-size 1 --rollouts-per-example 1 --max-tokens 384 --seq-len 1024"
            ;;
        mixed-canary)
            echo "--max-steps 4 --max-async-level 0 --num-prompts 24 --batch-size 1 --rollouts-per-example 1 --max-tokens 384 --seq-len 1024"
            ;;
        mixed-stable)
            echo "--max-steps 8 --max-async-level 0 --num-prompts 48 --batch-size 1 --rollouts-per-example 1 --max-tokens 384 --seq-len 1280 --learning-rate 1e-05"
            ;;
        mixed-soak)
            echo "--max-steps 16 --max-async-level 1 --num-prompts 128 --batch-size 2 --rollouts-per-example 2 --max-tokens 384 --seq-len 1280"
            ;;
        *)
            echo "ERROR: unknown ABIDE_RUN_PROFILE=${RUN_PROFILE}. Expected smoke, short-canary, long-canary, mixed-canary, mixed-stable, canary, mixed-soak, or soak." >&2
            exit 1
            ;;
    esac
}

FORM_SET="${FORM_SET_OVERRIDE:-$(profile_default_form_set)}"
OUTPUT_DIR="${OUTPUT_DIR_OVERRIDE:-$(default_output_dir)}"

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
echo "Profile: $(canonical_profile_name)"
echo "Inference GPU: $INFER_GPU"
echo "Training GPU: $TRAIN_GPU"
echo "Output: $OUTPUT_DIR"
echo "Runtime: $PRIME_RL_VENV"
echo "============================================================"

read -r -a EXTRA_PROFILE_ARGS <<< "$(profile_args)"
if [ -n "$MAX_STEPS_OVERRIDE" ]; then
    EXTRA_PROFILE_ARGS+=("--max-steps" "$MAX_STEPS_OVERRIDE")
fi
if [ -n "$MAX_ASYNC_LEVEL_OVERRIDE" ]; then
    EXTRA_PROFILE_ARGS+=("--max-async-level" "$MAX_ASYNC_LEVEL_OVERRIDE")
fi
if [ -n "$NUM_PROMPTS_OVERRIDE" ]; then
    EXTRA_PROFILE_ARGS+=("--num-prompts" "$NUM_PROMPTS_OVERRIDE")
fi
if [ -n "$SEED_OVERRIDE" ]; then
    EXTRA_PROFILE_ARGS+=("--seed" "$SEED_OVERRIDE")
fi
if [ -n "$BATCH_SIZE_OVERRIDE" ]; then
    EXTRA_PROFILE_ARGS+=("--batch-size" "$BATCH_SIZE_OVERRIDE")
fi
if [ -n "$ROLLOUTS_PER_EXAMPLE_OVERRIDE" ]; then
    EXTRA_PROFILE_ARGS+=("--rollouts-per-example" "$ROLLOUTS_PER_EXAMPLE_OVERRIDE")
fi
if [ -n "$MAX_TOKENS_OVERRIDE" ]; then
    EXTRA_PROFILE_ARGS+=("--max-tokens" "$MAX_TOKENS_OVERRIDE")
fi
if [ -n "$SEQ_LEN_OVERRIDE" ]; then
    EXTRA_PROFILE_ARGS+=("--seq-len" "$SEQ_LEN_OVERRIDE")
fi
if [ -n "$LEARNING_RATE_OVERRIDE" ]; then
    EXTRA_PROFILE_ARGS+=("--learning-rate" "$LEARNING_RATE_OVERRIDE")
fi
VISIBLE_GPUS="${INFER_GPU},${TRAIN_GPU}"

CUDA_VISIBLE_DEVICES="$VISIBLE_GPUS" python scripts/train_prime_rl.py \
    --model "$MODEL" \
    --form-set "$FORM_SET" \
    --output "$OUTPUT_DIR" \
    --port "$PORT" \
    --runtime-venv "$PRIME_RL_VENV" \
    "${EXTRA_PROFILE_ARGS[@]}" \
    "$@"
