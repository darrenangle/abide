# SFT Warmup Data Generation Pipeline

## Overview

A multi-agent pipeline for generating high-quality SFT training data with reasoning traces. Triggered via Claude Code slash command, it orchestrates haiku agents and OpenRouter models to produce verified poetry examples with two trace styles per poem.

## SYNTH-Style Reasoning Traces

Based on [Baguettotron's stenographic notation](https://huggingface.co/PleIAs/Baguettotron):

### Logical Markers
| Token | Meaning | Usage |
|-------|---------|-------|
| → | derivation/implication | Very short causal/logical flow |
| ↺ | iterative return/refinement | Backtracking, reconsidering |
| ? | uncertainty/questions | Appended to expressions |
| !/※ | insight/breakthrough | Knowledge discovery |
| ≈ | approximation/estimate | Preliminary hypothesis |
| ∴ | therefore/final step | Stable conclusions |

### Uncertainty Markers
| Token | Meaning | Usage |
|-------|---------|-------|
| ● | high confidence | Well-supported ground |
| ◐ | medium confidence | Incomplete data |
| ○ | low confidence | Speculation |
| ⚠ | bias/premise risk | Domain mismatch |
| ?maybe? | soft speculation | Tentative ideas |

### Verification Process
| Token | Meaning | Usage |
|-------|---------|-------|
| ☐ | unverified hypothesis | Raw claim |
| ☑ | intermediate verification | One source supports |
| ✓ | confirmed/validated | Multiple supports |

### Example SYNTH Trace for Sonnet
```
<think>
Sonnet requirements:
├─ lines: 14 ●
├─ meter: iambic pentameter ◐
├─ rhyme: ABAB CDCD EFEF GG ●
└─ topic: autumn melancholy

→ 14 lines structured as 3 quatrains + couplet
→ each line ≈10 syllables da-DUM pattern
?maybe? allow feminine endings?

Line 1 planning:
├─ end word needs A-rhyme partner
├─ syllable count: target 10
└─ iambic stress pattern

☐ "When autumn's breath turns leaves to gold"
→ syllables: 8 ⚠ too short
↺ revise: "When autumn's gentle breath turns leaves to gold"
→ syllables: 10 ✓
→ stress: da-DUM-da-DUM-da-DUM-da-DUM-da-DUM ●

⟨H≈0.5⟩ exploring rhyme partners for "gold"...
├─ cold, told, old, behold, unfold
└─ selecting: "behold" for line 3

∴ Line 1 complete: ● high confidence
</think>
```

### Example Non-SYNTH Trace (Natural Think)
```
<think>
I need to write a Sonnet about autumn with a melancholic tone.

A Sonnet has 14 lines with iambic pentameter - that's 10 syllables per line
in a da-DUM da-DUM pattern. The rhyme scheme is ABAB CDCD EFEF GG.

Let me start with the first line. I want something that evokes autumn's
fading beauty...

"When autumn's gentle breath turns leaves to gold"

Let me count the syllables: When-au-tumn's-gen-tle-breath-turns-leaves-to-gold
That's 10 syllables, good! And the stress pattern works for iambic pentameter.

For line 3, I need something that rhymes with "gold"... behold, cold, told, old...
I'll use "behold" - it has a nice formal quality for a sonnet.

Moving on to line 2, I need a B-rhyme...
</think>
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    /generate-sft Slash Command                          │
├─────────────────────────────────────────────────────────────────────────┤
│  Config: forms[], num_per_form, min_score, model_mix                    │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Orchestrator Agent                              │
│  - Loads form list and topics                                           │
│  - Spawns parallel form-specific agents                                 │
│  - Monitors progress, handles failures                                  │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
           ┌─────────────────────┼─────────────────────┐
           ▼                     ▼                     ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  Form Agent #1   │  │  Form Agent #2   │  │  Form Agent #N   │
│  (Sonnet)        │  │  (Haiku)         │  │  (Villanelle)    │
│                  │  │                  │  │                  │
│  Backend:        │  │  Backend:        │  │  Backend:        │
│  - Claude Haiku  │  │  - Kimi K2       │  │  - Claude Sonnet │
│  - via Task tool │  │  - via OpenRouter│  │  - via Task tool │
└────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
         │                     │                     │
         ▼                     ▼                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Generation Loop                                  │
│  1. Pick random topic + tone                                            │
│  2. Generate poem with reasoning trace                                  │
│  3. Extract poem, verify with abide                                     │
│  4. If score >= 0.8: synthesize both trace styles, write to log         │
│  5. If score < 0.8: retry with feedback (max 5 attempts)                │
│  6. Repeat until num_per_form reached                                   │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Append-Only JSONL Log                                │
│  data/sft_dataset.jsonl                                                 │
│                                                                         │
│  Each entry:                                                            │
│  {                                                                      │
│    "form": "Sonnet",                                                    │
│    "prompt": "Write a Sonnet about autumn in a melancholic tone",       │
│    "synth_trace": "<think>Sonnet requirements:\n├─ lines: 14 ●...",     │
│    "natural_trace": "<think>I need to write a Sonnet about...",         │
│    "poem": "When autumn's gentle breath turns leaves to gold...",       │
│    "score": 0.92,                                                       │
│    "rubric": {...},                                                     │
│    "model": "claude-3-5-haiku",                                         │
│    "timestamp": "2024-12-29T22:30:00Z"                                  │
│  }                                                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Slash Command (`/.claude/commands/generate-sft.md`)

```markdown
Generate SFT training data for poetry forms with reasoning traces.

Usage: /generate-sft [options]

Options:
  --forms: Comma-separated list of forms (default: top 10 learnable)
  --num: Number of examples per form (default: 100)
  --min-score: Minimum abide score to accept (default: 0.8)
  --model-mix: Ratio of Claude:OpenRouter (default: 0.5)
  --openrouter-model: Model for OpenRouter (default: moonshotai/kimi-k2)
  --output: Output file (default: data/sft_dataset.jsonl)

Run the SFT data generation pipeline using the configuration above.
Spawn parallel agents for each form, using a mix of Claude haiku agents
and OpenRouter models as specified.
```

### 2. Orchestrator (`scripts/sft_orchestrator.py`)

Main entry point that:
- Parses configuration
- Loads topics from `data/topics.txt`
- Distributes forms across agents based on model_mix
- Spawns parallel Task agents for Claude backends
- Spawns parallel OpenRouter workers for external models
- Monitors progress and aggregates results

### 3. Form Agent (Claude Code Task)

Each agent receives:
- Assigned form name
- Target count (e.g., 100 poems)
- Topic list
- Min score threshold
- Access to abide tools

Agent loop:
```python
while successes < target:
    topic = random.choice(topics)
    tone = random.choice(tones)
    prompt = f"Write a {form_name} about {topic} in a {tone} tone."

    for attempt in range(max_retries):
        # Generate poem
        poem = generate_with_reasoning(prompt, form_instruction)

        # Verify with abide
        result = verify(poem, form)

        if result.score >= min_score:
            # Generate both trace styles
            synth_trace = generate_synth_trace(poem, form, result)
            natural_trace = generate_natural_trace(poem, form, result)

            # Append to log
            write_to_log(form_name, prompt, synth_trace, natural_trace, poem, result)
            successes += 1
            break
        else:
            # Provide feedback for retry
            feedback = format_rubric_feedback(result)
```

### 4. OpenRouter Backend (`scripts/openrouter_generator.py`)

```python
class OpenRouterGenerator:
    def __init__(self, api_key: str, model: str = "moonshotai/kimi-k2"):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        self.model = model

    def generate_poem(self, prompt: str, form_instruction: str) -> tuple[str, str]:
        """Generate poem with natural reasoning trace."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": POET_SYSTEM_PROMPT},
                {"role": "user", "content": f"{form_instruction}\n\n{prompt}"}
            ],
            max_tokens=4096
        )
        full_response = response.choices[0].message.content
        # Parse out <think> trace and poem
        return parse_response(full_response)

    def generate_with_feedback(self, prompt: str, form_instruction: str,
                               previous_poem: str, rubric: dict) -> tuple[str, str]:
        """Retry generation with verification feedback."""
        feedback = format_rubric_feedback(rubric)
        messages = [
            {"role": "system", "content": POET_SYSTEM_PROMPT},
            {"role": "user", "content": f"{form_instruction}\n\n{prompt}"},
            {"role": "assistant", "content": previous_poem},
            {"role": "user", "content": f"That didn't quite meet the requirements:\n{feedback}\n\nPlease try again."}
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=4096
        )
        return parse_response(response.choices[0].message.content)
```

### 5. SYNTH Trace Synthesizer

Converts natural reasoning into SYNTH stenographic format:

```python
def synthesize_synth_trace(poem: str, form_name: str, result: VerifyResult,
                           natural_trace: str) -> str:
    """Convert natural reasoning to SYNTH stenographic style."""
    form = get_form_by_name(form_name)()

    trace = f"<think>\n{form_name} requirements:\n"

    # Build requirement tree
    for item in result.rubric:
        confidence = "●" if item.score > 0.9 else "◐" if item.score > 0.7 else "○"
        trace += f"├─ {item.criterion}: {confidence}\n"

    # Add planning section
    trace += f"└─ topic: {extract_topic(poem)}\n\n"

    # Convert natural reasoning to stenographic
    trace += convert_to_stenographic(natural_trace)

    # Add verification summary
    trace += f"\n∴ {form_name} complete: "
    trace += "● high confidence" if result.score > 0.9 else "◐ needs review"
    trace += "\n</think>\n"

    return trace


def convert_to_stenographic(natural_trace: str) -> str:
    """Convert verbose reasoning to dense stenographic form."""
    lines = []
    for sentence in natural_trace.split('. '):
        # Convert causal language
        if 'because' in sentence or 'so' in sentence:
            lines.append(f"→ {compress(sentence)}")
        # Convert uncertainty
        elif 'maybe' in sentence or 'might' in sentence:
            lines.append(f"?maybe? {compress(sentence)}")
        # Convert verification
        elif 'check' in sentence or 'count' in sentence:
            lines.append(f"☑ {compress(sentence)}")
        # Convert conclusions
        elif 'therefore' in sentence or 'final' in sentence:
            lines.append(f"∴ {compress(sentence)}")
        else:
            lines.append(compress(sentence))
    return '\n'.join(lines)
```

### 6. Dataset Writer

Thread-safe append-only JSONL writer:

```python
import fcntl
import json
from datetime import datetime

def append_to_dataset(
    output_file: str,
    form: str,
    prompt: str,
    synth_trace: str,
    natural_trace: str,
    poem: str,
    score: float,
    rubric: list,
    model: str
):
    """Thread-safe append to JSONL dataset."""
    entry = {
        "form": form,
        "prompt": prompt,
        "synth_trace": synth_trace,
        "natural_trace": natural_trace,
        "poem": poem,
        "score": score,
        "rubric": [{"criterion": r.criterion, "score": r.score} for r in rubric],
        "model": model,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

    with open(output_file, 'a') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.write(json.dumps(entry) + '\n')
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
```

## Configuration

```yaml
# config/sft_generation.yaml
pipeline:
  forms:
    - IrregularOde
    - ConsonantCascade
    - Sonnet
    - CoprimeVerse
    - Anaphora
    - Mesostic
    - Rubaiyat
    - PetrarchanSonnet
    - Etheree
    - BroadBallad

  num_per_form: 100
  min_score: 0.8
  max_retries: 5

model_mix:
  # Ratio of Claude Code agents vs OpenRouter
  claude_ratio: 0.5  # 50% Claude, 50% OpenRouter

  claude:
    model: haiku  # or sonnet
    parallel_agents: 4

  openrouter:
    api_key_env: OPENROUTER_API_KEY
    models:
      - moonshotai/kimi-k2
      - deepseek/deepseek-r1
      - anthropic/claude-sonnet-4
    parallel_workers: 4

output:
  file: data/sft_dataset.jsonl
  checkpoint_every: 10  # Save progress every N examples
```

## Usage

### Via Slash Command
```bash
# In Claude Code session
/generate-sft --forms Sonnet,Haiku --num 10 --model-mix 0.7

# Full run with defaults
/generate-sft
```

### Via Python Script
```bash
# Using OpenRouter only
OPENROUTER_API_KEY=your_key python scripts/sft_orchestrator.py \
  --forms Sonnet,Haiku \
  --num 100 \
  --backend openrouter \
  --model moonshotai/kimi-k2

# Using Claude Code agents only
python scripts/sft_orchestrator.py \
  --forms Sonnet,Haiku \
  --num 100 \
  --backend claude \
  --model haiku

# Mixed mode
python scripts/sft_orchestrator.py --config config/sft_generation.yaml
```

### Quick Test
```bash
# Generate 5 sonnets with dry run (no API calls)
python scripts/sft_orchestrator.py --forms Sonnet --num 5 --dry-run
```

## Output Format

Each line in `data/sft_dataset.jsonl`:
```json
{
  "form": "Sonnet",
  "prompt": "Write a Sonnet about autumn in a melancholic tone",
  "synth_trace": "<think>\nSonnet requirements:\n├─ lines: 14 ●\n├─ meter: iambic pentameter ◐\n...",
  "natural_trace": "<think>\nI need to write a Sonnet about autumn...\n</think>",
  "poem": "When autumn's gentle breath turns leaves to gold,\nAnd shadows...",
  "score": 0.92,
  "rubric": [
    {"criterion": "line_count", "score": 1.0},
    {"criterion": "syllables", "score": 0.85},
    {"criterion": "rhyme_scheme", "score": 0.91}
  ],
  "model": "claude-3-5-haiku",
  "timestamp": "2024-12-29T22:30:00Z"
}
```

## Implementation Plan

### Phase 1: POC (This PR)
- [ ] OpenRouter generator with Kimi K2
- [ ] Single-form generation loop with abide verification
- [ ] Basic SYNTH trace synthesis
- [ ] Append-only JSONL writer
- [ ] CLI interface

### Phase 2: Multi-Agent
- [ ] Claude Code slash command
- [ ] Task-based haiku/sonnet agents
- [ ] Parallel form processing
- [ ] Progress monitoring

### Phase 3: Production
- [ ] Configurable model mix
- [ ] Checkpoint/resume support
- [ ] Quality metrics dashboard
- [ ] Integration with SFT training pipeline
