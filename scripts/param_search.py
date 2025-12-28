#!/usr/bin/env python3
"""
Quick parameter search for GRPO hyperparameters.

Runs a small number of rollouts and measures mean reward to compare
different parameter configurations before committing to a full training run.

Usage:
    python scripts/param_search.py --max-tokens 2048 --rep-penalty 1.15
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Add src and scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import requests

BAGUETTOTRON_PATH = "/home/darren/10k-poems/models/baguettotron_sft/final"
VLLM_PORT = 8000


@dataclass
class ExperimentConfig:
    max_tokens: int = 2048
    repetition_penalty: float = 1.15
    temperature: float = 0.6
    top_p: float = 0.95
    num_rollouts: int = 128
    num_prompts: int = 16  # 128 rollouts / 8 rollouts_per_example


def check_vllm_ready() -> bool:
    """Check if vLLM server is ready."""
    try:
        resp = requests.get(f"http://localhost:{VLLM_PORT}/health", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


def wait_for_vllm(timeout: int = 300) -> bool:
    """Wait for vLLM to be ready."""
    start = time.time()
    while time.time() - start < timeout:
        if check_vllm_ready():
            return True
        time.sleep(5)
    return False


def run_experiment(config: ExperimentConfig) -> dict:
    """Run a single experiment and return results."""
    from prompt_generator import generate_verifiers_dataset, get_forms

    import verifiers as vf
    from verifiers.rl.trainer import RLConfig, RLTrainer

    print(f"\n{'='*60}")
    print(f"EXPERIMENT: max_tokens={config.max_tokens}, rep_penalty={config.repetition_penalty}")
    print(f"{'='*60}")

    # Load forms
    forms = get_forms()
    print(f"Loaded {len(forms)} forms")

    # Generate small dataset
    print(f"Generating {config.num_prompts} prompts...")
    dataset = generate_verifiers_dataset(
        num_prompts=config.num_prompts,
        seed=42,
    )

    # Create reward function
    def get_completion_text(completion, require_think_close: bool = True) -> str:
        val = completion
        if isinstance(val, list):
            if len(val) > 0 and isinstance(val[0], dict):
                val = val[-1].get("content", "")
            elif len(val) > 0 and isinstance(val[0], str):
                val = "".join(val)
        if isinstance(val, list):
            val = "".join([str(x) for x in val])

        text = str(val)
        has_think_close = "</think>" in text

        if require_think_close:
            if not has_think_close:
                return ""
            text = text.split("</think>", 1)[-1]

        text = text.replace("<think>", "").replace("</think>", "")
        text = text.replace("<|im_end|>", "").replace("<|im_start|>", "")
        text = text.replace("<|end_of_text|>", "").replace("<|begin_of_text|>", "")

        return text.strip()

    rewards = []

    def reward_fn(completion, **kwargs) -> float:
        try:
            poem = get_completion_text(completion, require_think_close=True)
            if not poem or len(poem.strip()) < 10:
                rewards.append(0.0)
                return 0.0

            info = kwargs.get("info", {})
            form_name = info.get("form_name") if isinstance(info, dict) else None

            if not form_name:
                rewards.append(0.0)
                return 0.0

            form_instance = forms.get(form_name)
            if form_instance is None:
                rewards.append(0.0)
                return 0.0

            result = form_instance.verify(poem)
            rewards.append(result.score)
            return result.score
        except Exception as e:
            print(f"[reward error: {e}]")
            rewards.append(0.0)
            return 0.0

    rubric = vf.Rubric(funcs=[reward_fn], weights=[1.0])
    env = vf.SingleTurnEnv(dataset=dataset, rubric=rubric)

    # Create config
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rl_config = RLConfig(
        output_dir="models/param_search_temp",
        run_name=f"param-search-{int(time.time())}",
        learning_rate=1e-6,
        num_train_epochs=1,
        max_steps=1,  # Just one step to measure initial reward
        per_device_train_batch_size=2,
        batch_size=config.num_rollouts,
        micro_batch_size=2,
        rollouts_per_example=config.num_rollouts // config.num_prompts,
        max_seq_len=config.max_tokens + 512,  # Extra for prompt
        max_prompt_len=384,
        vllm_server_port=VLLM_PORT,
        temperature=config.temperature,
        top_p=config.top_p,
        repetition_penalty=config.repetition_penalty,
        max_tokens=config.max_tokens,
        mask_truncated_completions=True,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        save_steps=9999,  # Don't save
        report_to="none",
        remove_unused_columns=False,
        use_lora=True,
    )

    # Add stop tokens
    rl_config.sampling_args["stop"] = ["<|im_end|>"]

    # Load model
    print(f"Loading model: {BAGUETTOTRON_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        BAGUETTOTRON_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BAGUETTOTRON_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create trainer and run one step
    trainer = RLTrainer(
        model=model,
        env=env,
        args=rl_config,
        processing_class=tokenizer,
    )

    print("Running rollouts...")
    start_time = time.time()

    try:
        trainer.train()
    except Exception as e:
        # Training may error on step 2 since we only have 1 step, that's fine
        if "max_steps" not in str(e).lower():
            print(f"Training error (expected): {e}")

    elapsed = time.time() - start_time

    # Calculate statistics
    if rewards:
        mean_reward = sum(rewards) / len(rewards)
        max_reward = max(rewards)
        min_reward = min(rewards)
        nonzero = sum(1 for r in rewards if r > 0)
        nonzero_pct = nonzero / len(rewards) * 100
    else:
        mean_reward = max_reward = min_reward = 0.0
        nonzero = nonzero_pct = 0

    results = {
        "timestamp": datetime.now().isoformat(),
        "max_tokens": config.max_tokens,
        "repetition_penalty": config.repetition_penalty,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "num_rollouts": len(rewards),
        "mean_reward": mean_reward,
        "max_reward": max_reward,
        "min_reward": min_reward,
        "nonzero_count": nonzero,
        "nonzero_pct": nonzero_pct,
        "elapsed_seconds": elapsed,
    }

    print(f"\n{'='*60}")
    print("RESULTS:")
    print(f"  Mean Reward: {mean_reward:.4f}")
    print(f"  Max Reward:  {max_reward:.4f}")
    print(f"  Min Reward:  {min_reward:.4f}")
    print(f"  Non-zero:    {nonzero}/{len(rewards)} ({nonzero_pct:.1f}%)")
    print(f"  Time:        {elapsed:.1f}s")
    print(f"{'='*60}\n")

    # Cleanup
    del model, trainer
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="GRPO parameter search")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--rep-penalty", type=float, default=1.15)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--num-rollouts", type=int, default=128)
    parser.add_argument("--output", type=str, default="experiments/param_search_results.jsonl")
    args = parser.parse_args()

    # Check vLLM
    if not check_vllm_ready():
        print("vLLM not ready. Please start it first:")
        print(
            f"  CUDA_VISIBLE_DEVICES=1 vf-vllm --model {BAGUETTOTRON_PATH} --port {VLLM_PORT} --max-model-len 4096 --enforce-eager"
        )
        sys.exit(1)

    config = ExperimentConfig(
        max_tokens=args.max_tokens,
        repetition_penalty=args.rep_penalty,
        temperature=args.temperature,
        top_p=args.top_p,
        num_rollouts=args.num_rollouts,
        num_prompts=max(8, args.num_rollouts // 8),
    )

    results = run_experiment(config)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a") as f:
        f.write(json.dumps(results) + "\n")

    print(f"Results saved to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
