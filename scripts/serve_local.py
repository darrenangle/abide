#!/usr/bin/env python3
"""
Launch a local vLLM server for Gemma 3n models.

This creates an OpenAI-compatible API server that can be used
with the verifiers framework for training and evaluation.

Usage:
    # Start server with Gemma 3n E4B (recommended for 2x4090)
    uv run python scripts/serve_local.py

    # Use the smaller E2B variant
    uv run python scripts/serve_local.py --model google/gemma-3n-E2B-it

    # Specify GPU memory utilization
    uv run python scripts/serve_local.py --gpu-memory 0.85

Prerequisites:
    uv sync --extra vllm
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch vLLM OpenAI-compatible server for local inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--model",
        default="google/gemma-3n-E4B-it",
        help="Model to serve (default: google/gemma-3n-E4B-it)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to serve on (default: 8000)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--gpu-memory",
        type=float,
        default=0.90,
        help="GPU memory utilization (default: 0.90)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum model context length (default: 4096)",
    )
    parser.add_argument(
        "--tensor-parallel",
        type=int,
        default=2,
        help="Number of GPUs for tensor parallelism (default: 2 for dual 4090)",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["auto", "float16", "bfloat16"],
        help="Data type for model weights (default: bfloat16)",
    )

    args = parser.parse_args()

    # Build vLLM serve command
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--gpu-memory-utilization",
        str(args.gpu_memory),
        "--max-model-len",
        str(args.max_model_len),
        "--tensor-parallel-size",
        str(args.tensor_parallel),
        "--dtype",
        args.dtype,
        "--trust-remote-code",
    ]

    print("=" * 60)
    print("Starting vLLM OpenAI-Compatible Server")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Server: http://{args.host}:{args.port}")
    print(f"GPU Memory: {args.gpu_memory:.0%}")
    print(f"Tensor Parallel: {args.tensor_parallel} GPUs")
    print(f"Max Context: {args.max_model_len}")
    print()
    print("Use with OpenAI client:")
    print('  client = OpenAI(base_url="http://localhost:8000/v1", api_key="local")')
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 60)

    try:
        subprocess.run(cmd, check=True)
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Error: vLLM server failed with exit code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\nServer stopped.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
