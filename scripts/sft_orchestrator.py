#!/usr/bin/env python3
"""
SFT Data Generation Orchestrator.

Coordinates parallel poem generation across:
1. Claude Code haiku/sonnet agents (via subprocess claude CLI)
2. OpenRouter API workers (Kimi K2, DeepSeek, etc.)

Generates verified poems with dual reasoning traces (SYNTH + natural).

Usage:
    # OpenRouter only
    python scripts/sft_orchestrator.py --forms Sonnet,Haiku --num 10 --backend openrouter

    # Claude Code agents only
    python scripts/sft_orchestrator.py --forms Sonnet,Haiku --num 10 --backend claude

    # Mixed mode (50/50)
    python scripts/sft_orchestrator.py --forms Sonnet,Haiku --num 10 --model-mix 0.5

    # From config file
    python scripts/sft_orchestrator.py --config config/sft_generation.yaml
"""

import argparse
import json
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Add parent and src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from scripts.openrouter_generator import OpenRouterGenerator, append_to_jsonl


@dataclass
class GenerationTask:
    """A single poem generation task."""

    form: str
    topic: str
    tone: str
    backend: str  # "openrouter" or "claude"
    model: str


@dataclass
class GenerationResult:
    """Result of a generation task."""

    success: bool
    form: str
    data: dict | None = None
    error: str | None = None


# Default configuration
DEFAULT_CONFIG = {
    "pipeline": {
        "forms": [
            "IrregularOde",
            "ConsonantCascade",
            "Sonnet",
            "CoprimeVerse",
            "Anaphora",
            "Mesostic",
            "Rubaiyat",
            "PetrarchanSonnet",
            "Etheree",
            "BroadBallad",
        ],
        "num_per_form": 100,
        "min_score": 0.8,
        "max_retries": 5,
    },
    "model_mix": {
        "claude_ratio": 0.5,
        "claude": {
            "model": "haiku",
            "parallel_agents": 4,
        },
        "openrouter": {
            "models": ["moonshotai/kimi-k2"],
            "parallel_workers": 4,
        },
    },
    "output": {
        "file": "data/sft_dataset.jsonl",
        "checkpoint_every": 10,
    },
}

# Default topics and tones
DEFAULT_TOPICS = [
    "autumn leaves falling",
    "ocean waves at sunset",
    "memory of childhood",
    "city lights at night",
    "mountain sunrise",
    "winter's first snow",
    "summer thunderstorm",
    "old photographs",
    "garden in spring",
    "starlight and silence",
    "morning coffee ritual",
    "forest after rain",
    "empty train platform",
    "grandmother's kitchen",
    "midnight library",
    "street musician's song",
    "abandoned lighthouse",
    "first day of spring",
    "paper boats on puddles",
    "wind through tall grass",
]

DEFAULT_TONES = [
    "melancholic",
    "joyful",
    "contemplative",
    "nostalgic",
    "serene",
    "passionate",
    "wistful",
    "hopeful",
    "mysterious",
    "tender",
]


class ClaudeAgentWorker:
    """Worker that uses Claude Code CLI to generate poems."""

    def __init__(self, model: str = "haiku"):
        self.model = model
        self.claude_path = self._find_claude()

    def _find_claude(self) -> str:
        """Find claude CLI executable."""
        # Try common locations
        for path in ["claude", "/usr/local/bin/claude", str(Path.home() / ".local/bin/claude")]:
            try:
                result = subprocess.run([path, "--version"], capture_output=True, text=True)
                if result.returncode == 0:
                    return path
            except FileNotFoundError:
                continue
        raise RuntimeError("claude CLI not found. Install Claude Code first.")

    def generate_poem(
        self, form: str, topic: str, tone: str, min_score: float = 0.8
    ) -> dict | None:
        """Generate a poem using Claude Code agent."""
        prompt = f"""You are generating training data for an SFT dataset.

Generate a {form} poem about "{topic}" in a {tone} tone.

Requirements:
1. First, output your reasoning in <think>...</think> tags
2. Then output the poem after </think>
3. The poem must score at least {min_score:.0%} on the {form} form constraints

Use abide to verify your poem:
```python
from abide.forms import {form}
from abide import verify
result = verify(poem, {form}())
print(f"Score: {{result.score}}")
```

Keep trying until you get a score >= {min_score}. When successful, output the final result as JSON:

```json
{{
  "form": "{form}",
  "prompt": "Write a {form} about {topic} in a {tone} tone",
  "synth_trace": "<the reasoning trace in SYNTH stenographic format>",
  "natural_trace": "<the reasoning trace in natural language>",
  "poem": "<the verified poem>",
  "score": <the verification score>
}}
```
"""
        try:
            # Run claude with the prompt
            result = subprocess.run(
                [self.claude_path, "-p", prompt, "--model", self.model, "--output-format", "text"],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            if result.returncode != 0:
                print(f"  Claude agent error: {result.stderr}")
                return None

            # Try to extract JSON from output
            output = result.stdout
            json_match = output.rfind("```json")
            if json_match != -1:
                json_end = output.find("```", json_match + 7)
                if json_end != -1:
                    json_str = output[json_match + 7 : json_end].strip()
                    data = json.loads(json_str)
                    data["model"] = f"claude-{self.model}"
                    data["timestamp"] = datetime.utcnow().isoformat() + "Z"
                    return data

            print("  Could not parse JSON from Claude output")
            return None

        except subprocess.TimeoutExpired:
            print("  Claude agent timed out")
            return None
        except json.JSONDecodeError as e:
            print(f"  JSON parse error: {e}")
            return None
        except Exception as e:
            print(f"  Claude agent error: {e}")
            return None


class SFTOrchestrator:
    """Orchestrates parallel SFT data generation."""

    def __init__(self, config: dict):
        self.config = config
        self.pipeline = config["pipeline"]
        self.model_mix = config["model_mix"]
        self.output_config = config["output"]

        # Stats
        self.stats = {
            "total": 0,
            "successes": 0,
            "failures": 0,
            "by_form": {},
            "by_backend": {"openrouter": 0, "claude": 0},
        }
        self.stats_lock = threading.Lock()

        # Load topics and tones
        self.topics = self._load_file("data/topics.txt", DEFAULT_TOPICS)
        self.tones = self._load_file("data/tones.txt", DEFAULT_TONES)

        # Workers initialized lazily in run()
        self.openrouter_generator = None
        self.claude_worker = None
        self._initialized = False

    def _init_workers(self):
        """Initialize workers (lazy initialization)."""
        if self._initialized:
            return

        if self.model_mix.get("claude_ratio", 0) < 1.0:
            # Need OpenRouter
            try:
                self.openrouter_generator = OpenRouterGenerator(
                    model=self.model_mix["openrouter"]["models"][0],
                    min_score=self.pipeline["min_score"],
                    max_retries=self.pipeline["max_retries"],
                )
            except ValueError as e:
                print(f"Warning: {e}. OpenRouter backend disabled.")
                self.model_mix["claude_ratio"] = 1.0

        if self.model_mix.get("claude_ratio", 0) > 0:
            # Need Claude
            try:
                self.claude_worker = ClaudeAgentWorker(model=self.model_mix["claude"]["model"])
            except RuntimeError as e:
                print(f"Warning: {e}. Claude backend disabled.")
                self.model_mix["claude_ratio"] = 0

        self._initialized = True

    def _load_file(self, filepath: str, defaults: list) -> list:
        """Load lines from file or return defaults."""
        path = Path(filepath)
        if path.exists():
            with path.open() as f:
                lines = [line.strip() for line in f if line.strip()]
                return lines if lines else defaults
        return defaults

    def _create_tasks(self) -> list[GenerationTask]:
        """Create all generation tasks."""
        tasks = []
        forms = self.pipeline["forms"]
        num_per_form = self.pipeline["num_per_form"]
        claude_ratio = self.model_mix.get("claude_ratio", 0.5)

        for form in forms:
            self.stats["by_form"][form] = {"target": num_per_form, "success": 0, "fail": 0}

            for _ in range(num_per_form):
                topic = random.choice(self.topics)
                tone = random.choice(self.tones)

                # Decide backend based on ratio
                if random.random() < claude_ratio and self.claude_worker:
                    backend = "claude"
                    model = self.model_mix["claude"]["model"]
                else:
                    backend = "openrouter"
                    models = self.model_mix["openrouter"]["models"]
                    model = random.choice(models)

                tasks.append(
                    GenerationTask(
                        form=form,
                        topic=topic,
                        tone=tone,
                        backend=backend,
                        model=model,
                    )
                )

        random.shuffle(tasks)  # Interleave forms
        return tasks

    def _process_task(self, task: GenerationTask) -> GenerationResult:
        """Process a single generation task."""
        try:
            if task.backend == "openrouter" and self.openrouter_generator:
                # Temporarily set model
                original_model = self.openrouter_generator.model
                self.openrouter_generator.model = task.model

                result = self.openrouter_generator.generate_poem(
                    form_name=task.form,
                    topic=task.topic,
                    tone=task.tone,
                )

                self.openrouter_generator.model = original_model

                if result:
                    return GenerationResult(success=True, form=task.form, data=result)
                else:
                    return GenerationResult(
                        success=False, form=task.form, error="Max retries exceeded"
                    )

            elif task.backend == "claude" and self.claude_worker:
                result = self.claude_worker.generate_poem(
                    form=task.form,
                    topic=task.topic,
                    tone=task.tone,
                    min_score=self.pipeline["min_score"],
                )

                if result:
                    return GenerationResult(success=True, form=task.form, data=result)
                else:
                    return GenerationResult(
                        success=False, form=task.form, error="Claude agent failed"
                    )

            else:
                return GenerationResult(
                    success=False, form=task.form, error=f"Backend {task.backend} not available"
                )

        except Exception as e:
            return GenerationResult(success=False, form=task.form, error=str(e))

    def _update_stats(self, result: GenerationResult):
        """Thread-safe stats update."""
        with self.stats_lock:
            self.stats["total"] += 1
            if result.success:
                self.stats["successes"] += 1
                self.stats["by_form"][result.form]["success"] += 1
                if result.data:
                    backend = "claude" if "claude" in result.data.get("model", "") else "openrouter"
                    self.stats["by_backend"][backend] += 1
            else:
                self.stats["failures"] += 1
                self.stats["by_form"][result.form]["fail"] += 1

    def run(self, dry_run: bool = False):
        """Run the orchestration."""
        print("=" * 60)
        print("SFT Data Generation Orchestrator")
        print("=" * 60)

        forms = self.pipeline["forms"]
        num_per_form = self.pipeline["num_per_form"]
        total = len(forms) * num_per_form

        print(f"Forms: {len(forms)}")
        print(f"Per form: {num_per_form}")
        print(f"Total target: {total}")
        print(f"Min score: {self.pipeline['min_score']}")
        print(f"Claude ratio: {self.model_mix.get('claude_ratio', 0.5):.0%}")
        print(f"Output: {self.output_config['file']}")
        print("=" * 60)

        if dry_run:
            print("\n[Dry run - no generation]")
            for form in forms:
                print(f"  - {form}: {num_per_form} poems")
            return

        # Initialize workers (lazy)
        self._init_workers()

        # Create tasks
        tasks = self._create_tasks()
        print(f"\nCreated {len(tasks)} tasks")

        # Ensure output directory exists
        output_path = Path(self.output_config["file"])
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Determine parallelism
        openrouter_workers = self.model_mix["openrouter"].get("parallel_workers", 4)
        claude_workers = self.model_mix["claude"].get("parallel_agents", 2)
        max_workers = max(openrouter_workers, claude_workers)

        print(f"Running with {max_workers} parallel workers...")
        print()

        # Process tasks
        start_time = time.time()
        checkpoint_counter = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._process_task, task): task for task in tasks}

            for future in as_completed(futures):
                _task = futures[future]
                result = future.result()

                self._update_stats(result)
                checkpoint_counter += 1

                if result.success and result.data:
                    append_to_jsonl(str(output_path), result.data)
                    print(
                        f"[{self.stats['successes']}/{total}] {result.form}: score={result.data['score']:.2f} ({result.data['model']})"
                    )
                else:
                    print(f"[FAIL] {result.form}: {result.error}")

                # Checkpoint
                if checkpoint_counter >= self.output_config.get("checkpoint_every", 10):
                    elapsed = time.time() - start_time
                    rate = self.stats["total"] / elapsed if elapsed > 0 else 0
                    print(f"  ... {self.stats['total']}/{total} processed ({rate:.1f}/min)")
                    checkpoint_counter = 0

        # Final stats
        elapsed = time.time() - start_time
        print()
        print("=" * 60)
        print("Generation Complete")
        print("=" * 60)
        print(f"Total: {self.stats['total']}")
        print(
            f"Successes: {self.stats['successes']} ({100*self.stats['successes']/max(1,self.stats['total']):.1f}%)"
        )
        print(f"Failures: {self.stats['failures']}")
        print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
        print(f"Rate: {self.stats['successes']/elapsed*60:.1f} poems/min")
        print()
        print("By backend:")
        for backend, count in self.stats["by_backend"].items():
            print(f"  {backend}: {count}")
        print()
        print("By form:")
        for form, data in self.stats["by_form"].items():
            success_rate = 100 * data["success"] / max(1, data["success"] + data["fail"])
            print(f"  {form}: {data['success']}/{data['target']} ({success_rate:.0f}%)")
        print()
        print(f"Output: {output_path}")


def load_config(config_path: str | None) -> dict:
    """Load configuration from YAML file or use defaults."""
    if config_path and Path(config_path).exists():
        if not HAS_YAML:
            print("Warning: PyYAML not installed, using defaults")
            return DEFAULT_CONFIG.copy()
        with Path(config_path).open() as f:
            return yaml.safe_load(f)
    return DEFAULT_CONFIG.copy()


def main():
    parser = argparse.ArgumentParser(description="SFT Data Generation Orchestrator")
    parser.add_argument("--config", help="YAML config file")
    parser.add_argument("--forms", help="Comma-separated list of forms")
    parser.add_argument("--num", type=int, help="Number per form")
    parser.add_argument(
        "--backend", choices=["openrouter", "claude", "mixed"], help="Backend to use"
    )
    parser.add_argument("--model", help="Model for OpenRouter")
    parser.add_argument("--model-mix", type=float, help="Claude ratio (0-1)")
    parser.add_argument("--min-score", type=float, help="Minimum score")
    parser.add_argument("--output", help="Output JSONL file")
    parser.add_argument("--dry-run", action="store_true", help="Show config without running")

    args = parser.parse_args()

    # Load base config
    config = load_config(args.config)

    # Override with CLI args
    if args.forms:
        config["pipeline"]["forms"] = args.forms.split(",")
    if args.num:
        config["pipeline"]["num_per_form"] = args.num
    if args.min_score:
        config["pipeline"]["min_score"] = args.min_score
    if args.output:
        config["output"]["file"] = args.output
    if args.model:
        config["model_mix"]["openrouter"]["models"] = [args.model]
    if args.model_mix is not None:
        config["model_mix"]["claude_ratio"] = args.model_mix
    if args.backend == "openrouter":
        config["model_mix"]["claude_ratio"] = 0.0
    elif args.backend == "claude":
        config["model_mix"]["claude_ratio"] = 1.0

    # Run
    orchestrator = SFTOrchestrator(config)
    orchestrator.run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
