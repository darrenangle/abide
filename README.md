# Abide

> **Note**: This is an experimental research library in active development. APIs may change.

**Automatic reward functions for training LLMs to write poetry.**

Abide is a composable constraint algebra that transforms poetic form specifications into differentiable reward signals. Define a form once, get verification, scoring, and natural language instructions automatically.

## The Problem

Training LLMs to write poetry is hard because:
- **Manual evaluation doesn't scale** - You can't hand-score thousands of poems during RL training
- **Binary pass/fail loses signal** - A poem with 13/14 correct lines shouldn't score the same as garbage
- **Form rules are complex** - Villanelles have refrains, sestinas rotate end-words, and some forms also involve meter
- **Verification and prompts diverge** - Your reward function checks one thing, your prompt says another

## The Solution

```python
from abide.forms import ShakespeareanSonnet

sonnet = ShakespeareanSonnet()

# Automatic reward signal (0-1 continuous score)
result = sonnet.verify(llm_output)
reward = result.score  # 0.73

# Automatic prompt generation
prompt = sonnet.instruction()
# "Shakespearean Sonnet: 14 lines of about 10 syllables,
#  ABAB CDCD EFEF GG rhyme scheme."

# Automatic rubric (explainable rewards)
for item in result.rubric:
    print(f"{item.criterion}: {item.score:.0%}")
# Line count: 100%
# Syllables per line: 85%
# Rhyme scheme: 67%
```

---

## Key Capabilities

### 1. Continuous Reward Signals

Every constraint produces smooth, differentiable scores—not binary pass/fail:

```python
from abide.constraints import SyllablesPerLine

constraint = SyllablesPerLine([5, 7, 5])  # Haiku (strict)

# Poem with 6-7-5 syllables (first line has 6 instead of 5)
result = constraint.verify("An old and silent pond\nA frog jumps into the pond\nSplash! Silence again")
print(result.score)  # 0.87 (partial credit for being close)
```

### 2. Metrical Analysis (Scansion)

Heuristically score meter using CMU Pronouncing Dictionary stress patterns:

```python
from abide.primitives import scan_line, MeterType, meter_score
from abide.constraints import Meter, FootLength

# Analyze a likely iambic-pentameter line
line = "Shall I compare thee to a summer's day"
result = scan_line(line)

print(result.binary_pattern)    # "1101110101"
print(result.foot_count)        # 5
print(result.dominant_meter)    # MeterType.IAMB
print(result.regularity)        # 0.84

# Score against expected meter
score = meter_score(line, MeterType.IAMB, expected_feet=5)
print(score)  # 0.92 (high heuristic match with substitutions)

# Use as a constraint
blank_verse = Meter(MeterType.IAMB, FootLength.PENTAMETER)
result = blank_verse.verify(poem)
```

### 3. Form Inference (Self-Consistent Spec Extraction)

Analyze a source poem and derive a structural/prosodic `FormSpec` that the
source poem itself satisfies with score `1.0`:

```python
from abide.inference import analyze_poem, infer_form

# Take a poem you want to model
poem = """The morning sun glows
Cherry blossoms gently fall
Spring has come at last"""

analysis = analyze_poem(poem)
print(analysis.structure.line_count)  # 3
print(analysis.syllable_pattern)      # [5, 7, 5]
print([c.id for c in analysis.constraints])
# ['line_count', 'syllables_exact']

# Generate a self-consistent FormSpec
spec = infer_form(poem, name="Three-Line Form")
assert spec.weighted_score(poem) == 1.0

# Now use it to score other poems against that inferred pattern
new_poem_score = spec.weighted_score(new_llm_output)
```

For longer structured poems, inference may also recover stanza counts,
repeated-line refrains, and rhyme-group constraints when those patterns are
reproducibly detectable. It is useful for extracting verifiable patterns from a
source poem, not for canonical literary-form classification.

### 4. Visual/Shape Poetry

Enforce line length patterns for concrete poetry:

```python
from abide.constraints import LineShape, ShapeType, MeasureMode

# Diamante: 1-2-3-4-3-2-1 word pattern
diamante = LineShape(
    shape_type=ShapeType.DIAMOND,
    num_lines=7,
    mode=MeasureMode.WORDS,
)

# Etheree: 1-2-3-4-5-6-7-8-9-10 syllables
etheree = LineShape(
    lengths=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    mode=MeasureMode.SYLLABLES,
)
```

### 5. Composable Constraint Algebra

Build complex forms from simple primitives:

```python
from abide.constraints import (
    LineCount, RhymeScheme, SyllablesPerLine,
    Refrain, Meter, And, Or, WeightedSum
)

# Villanelle: 19 lines, ABA rhyme, two rotating refrains
villanelle = And([
    LineCount(19),
    RhymeScheme("ABA ABA ABA ABA ABA ABAA"),
    Refrain(reference_line=0, repeat_at=[5, 11, 17]),   # First refrain
    Refrain(reference_line=2, repeat_at=[8, 14, 18]),   # Second refrain
])

# Weighted scoring (some constraints matter more)
sonnet = WeightedSum([
    (LineCount(14), 2.0),           # Structure is critical
    (SyllablesPerLine([10]*14), 1.5),  # Meter matters
    (RhymeScheme("ABABCDCDEFEFGG"), 2.0),  # Rhyme is important
], threshold=0.7)  # Minimum score to "pass"
```

---

## Catalog Coverage

Abide exports a broad catalog of historical, repeating, stanzaic, shape, and
experimental forms. Representative examples include:

- `Haiku`, `Tanka`, `ShakespeareanSonnet`, `Villanelle`, `Sestina`
- `Pantoum`, `TerzaRima`, `Rondeau`, `Triolet`, `Ghazal`
- `OttavaRima`, `RhymeRoyal`, `BalladStanza`, `BurnsStanza`, `Clerihew`
- `Diamante`, `Cinquain`, `Etheree`, plus many parameterized experimental forms

The long-term goal is full-catalog reliability. While that convergence work is
still in progress, the RL scripts use a curated rollout-default subset from
`abide.forms.catalog` so training runs do not accidentally depend on form
families that have not completed the same hardening pass yet.

Some intentionally generic families such as `Ode`, `IrregularOde`, and
`Epigram` are structural shells or proxies rather than thematic verifiers.

---

## For LLM Training

### Reward Function Integration

```python
from abide.forms import Villanelle
from abide.verifiers import PoeticFormReward

reward_fn = PoeticFormReward(Villanelle())

def compute_reward(generated_poem: str) -> float:
    return reward_fn(generated_poem).score
```

### Current RL Default

```python
from abide.forms.catalog import load_rl_default_form_instances

# Current curated default used by the RL scripts while
# the remaining form families are being brought up to the same bar.
forms = load_rl_default_form_instances()
```

### Prompt-Reward Alignment

The same specification generates both prompts and rewards:

```python
from abide.specs import villanelle_spec

spec = villanelle_spec()

# Generate training prompt
prompt = spec.full_instruction()

# Verify output with matching criteria
result = spec.verify(llm_output)
reward = spec.weighted_score(llm_output)
```

### Curriculum Learning

Test specific constraints in isolation:

```python
from abide.specs import shakespearean_sonnet_spec

spec = shakespearean_sonnet_spec()

# Stage 1: Just learn line count
stage1_reward = spec.verify_subset(poem, "line_count")

# Stage 2: Add syllables
stage2_reward = spec.verify_subset(poem, "line_count", "syllables")

# Stage 3: Full form
stage3_reward = spec.weighted_score(poem)
```

---

## Installation

```bash
# Package install
uv add abide
```

## Quick Start

```python
from abide.forms import Haiku
from abide import verify

poem = """An old silent pond
A frog jumps into the pond
Splash! Silence again"""

result = verify(poem, Haiku())
print(f"Score: {result.score:.0%}")  # Score: 100%

# See what matched and what didn't
for item in result.rubric:
    print(f"  {item.criterion}: {'PASS' if item.passed else 'FAIL'} ({item.score:.0%})")
```

## Development

```bash
git clone https://github.com/darrenangle/abide.git
cd abide
uv sync --all-extras
uv run pre-commit install
uv run pytest
```

## Architecture

```
abide/
├── primitives/     # NLP tools: syllables, phonetics, rhyme, meter
├── constraints/    # Composable constraint types
├── forms/          # Pre-built form templates and catalog helpers
├── specs/          # FormSpec for instruction generation
├── inference/      # Reverse-engineer forms from poems
└── verifiers/      # RL framework integration
```

## Experiments

We're actively running GRPO (Group Relative Policy Optimization) experiments to train language models on poetic form constraints.

### Training Scripts

| Script | Description |
|--------|-------------|
| [`scripts/train_grpo.py`](scripts/train_grpo.py) | Legacy verifiers RL trainer for local Gemma experiments on the well-known-form subset; upstream verifiers now recommends `prime-rl` for production training |
| [`scripts/train_grpo_trl.py`](scripts/train_grpo_trl.py) | TRL-based GRPO with KL regularization (beta parameter) |
| [`scripts/find_learnable_forms.py`](scripts/find_learnable_forms.py) | Identify forms with high within-rollout variance (best GRPO signal) |
| [`scripts/prompt_generator.py`](scripts/prompt_generator.py) | Generate training prompts from the form catalog |

### Run Scripts

| Script | Model | Forms | Notes |
|--------|-------|-------|-------|
| [`scripts/run_grpo_trl.sh`](scripts/run_grpo_trl.sh) | Gemma 3 4B | Top 10 learnable | TRL with beta=0.04 KL regularization |
| [`scripts/run_grpo_gemma4_e4b.sh`](scripts/run_grpo_gemma4_e4b.sh) | Gemma 4 E4B | RL-default | Resumable TRL runner with `smoke` / `canary` / `soak` profiles and persisted reward telemetry |
| [`scripts/run_grpo.sh`](scripts/run_grpo.sh) | Gemma 3 4B | Well-known | Legacy verifiers GRPO; auto-prepares `.venv-verifiers` via `scripts/prepare_verifiers_runtime.sh` |
| [`scripts/run_grpo_gemma3.sh`](scripts/run_grpo_gemma3.sh) | Gemma 3 4B | Well-known | Legacy verifiers GRPO runner focused on canonical forms |

### Key Findings

1. **Learnable forms matter**: Forms with high within-rollout variance (model sometimes succeeds, sometimes fails on same prompt) produce better GRPO learning signal than forms that are too easy or too hard.

2. **KL regularization helps stability**: Adding a beta parameter for KL divergence (e.g., beta=0.04) helps prevent policy collapse. Still experimenting with the right settings.

3. **Top 10 learnable forms** (by GRPO signal):
   - Epigram, ThunderVerse, ColorSpectrum, CoprimeVerse, ElementalVerse
   - CharacterPalindromePoem, QuestionQuest, VowelPilgrimage, Mesostic, Terzanelle

## License

MIT
