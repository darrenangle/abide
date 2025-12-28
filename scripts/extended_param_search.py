#!/usr/bin/env python3
"""Extended parameter search - higher repetition penalties."""

import json
import subprocess
import sys
from pathlib import Path

MODEL = "/home/darren/10k-poems/models/baguettotron_sft/final"
PORT = 8000
PYTHON = "/home/darren/miniconda3/bin/python"
OUTPUT = "experiments/extended_results.jsonl"

# Experiments: max_tokens=2048, higher rep_penalty
EXPERIMENTS = [
    (2048, 1.25),
    (2048, 1.35),
    (2048, 1.4),
    (2048, 1.5),
]

print("=" * 60)
print("EXTENDED PARAM SEARCH: Higher Repetition Penalties")
print("=" * 60)

# Clear output file
Path(OUTPUT).open("w").close()

for max_tokens, rep_penalty in EXPERIMENTS:
    print(f"\n>>> max_tokens={max_tokens}, rep_penalty={rep_penalty}")
    cmd = [
        PYTHON,
        "scripts/param_search.py",
        "--max-tokens",
        str(max_tokens),
        "--rep-penalty",
        str(rep_penalty),
        "--num-rollouts",
        "128",
        "--output",
        OUTPUT,
    ]
    result = subprocess.run(
        cmd, env={"CUDA_VISIBLE_DEVICES": "0", **dict(__import__("os").environ)}
    )
    if result.returncode != 0:
        print("ERROR: Experiment failed")
        sys.exit(1)

print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

# Read all results (old + new)
all_results = []
for fname in ["experiments/param_search_results.jsonl", OUTPUT]:
    try:
        with Path(fname).open() as f:
            for line in f:
                all_results.append(json.loads(line))
    except FileNotFoundError:
        pass

# Sort by reward
all_results.sort(key=lambda x: x["mean_reward"], reverse=True)

print("\nAll results (sorted by reward):")
for r in all_results:
    marker = "<<< BEST" if r == all_results[0] else ""
    print(
        f"  max_tokens={r['max_tokens']}, rep_penalty={r['repetition_penalty']}: "
        f"reward={r['mean_reward']:.4f} ({r['nonzero_pct']:.1f}% nonzero) {marker}"
    )

best = all_results[0]
print(
    f"\nBEST: max_tokens={best['max_tokens']}, rep_penalty={best['repetition_penalty']}, reward={best['mean_reward']:.4f}"
)
