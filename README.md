# Abide

**Automatic reward functions for training LLMs to write poetry.**

Abide is a composable constraint algebra that transforms poetic form specifications into differentiable reward signals. Define a form once, get verification, scoring, and natural language instructions automatically.

## The Problem

Training LLMs to write poetry is hard because:
- **Manual evaluation doesn't scale** - You can't hand-score thousands of poems during RL training
- **Binary pass/fail loses signal** - A poem with 13/14 correct lines shouldn't score the same as garbage
- **Form rules are complex** - Villanelles have refrains, sestinas rotate end-words, sonnets need iambic pentameter
- **Verification and prompts diverge** - Your reward function checks one thing, your prompt says another

## The Solution

```python
from abide.forms import ShakespeareanSonnet
from abide.primitives import MeterType

sonnet = ShakespeareanSonnet()

# Automatic reward signal (0-1 continuous score)
result = sonnet.verify(llm_output)
reward = result.score  # 0.73

# Automatic prompt generation (matches your reward function exactly)
prompt = sonnet.instruction()
# "Write a sonnet with 14 lines of iambic pentameter,
#  following the rhyme scheme ABAB CDCD EFEF GG..."

# Automatic rubric (explainable rewards)
for item in result.rubric:
    print(f"{item.criterion}: {item.score:.0%}")
# Line count: 100%
# Syllables per line: 85%
# Rhyme scheme: 67%
```

**One specification. Three outputs. Perfect alignment.**

---

## Key Capabilities

### 1. Continuous Reward Signals

Every constraint produces smooth, differentiable scores—not binary pass/fail:

```python
from abide.constraints import SyllablesPerLine

constraint = SyllablesPerLine([5, 7, 5], tolerance=1)  # Haiku

# Poem with 5-8-5 syllables (one line off by 1)
result = constraint.verify("An old silent pond\nA frog jumps into the pond\nSplash! Silence again")
print(result.score)  # 0.89 (partial credit, not 0)
```

### 2. Metrical Analysis (Scansion)

Detect and enforce meter using CMU Pronouncing Dictionary stress patterns:

```python
from abide.primitives import scan_line, MeterType, meter_score
from abide.constraints import Meter, FootLength

# Analyze Shakespeare's iambic pentameter
line = "Shall I compare thee to a summer's day"
result = scan_line(line)

print(result.binary_pattern)    # "1101110101"
print(result.foot_count)        # 5
print(result.dominant_meter)    # MeterType.IAMB
print(result.regularity)        # 0.80

# Score against expected meter
score = meter_score(line, MeterType.IAMB, expected_feet=5)
print(score)  # 0.99 (accounts for natural substitutions)

# Use as a constraint
blank_verse = Meter(MeterType.IAMB, FootLength.PENTAMETER)
result = blank_verse.verify(poem)
```

### 3. Form Inference (Reverse Engineering)

Analyze any poem and extract a specification that it passes 100%:

```python
from abide.inference import analyze_poem, infer_form

# Take any poem
poem = """Do not go gentle into that good night,
Old age should burn and rave at close of day;
Rage, rage against the dying of the light..."""

# Infer its constraints
analysis = analyze_poem(poem)
print(analysis.rhyme_scheme)      # "ABABAB..."
print(analysis.syllable_pattern)  # [10, 10, 10, ...]
print(analysis.refrains)          # [(0, [5, 11, 17]), ...]

# Generate a FormSpec the poem passes with score 1.0
spec = infer_form(poem, name="Dylan Thomas Style")
assert spec.weighted_score(poem) == 1.0

# Now use it to train/evaluate other poems in the same style
new_poem_score = spec.weighted_score(new_llm_output)
```

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

## Supported Forms (50+)

### Classic Forms
| Form | Lines | Key Features |
|------|-------|--------------|
| **Haiku** | 3 | 5-7-5 syllables |
| **Tanka** | 5 | 5-7-5-7-7 syllables |
| **Sonnet** (3 variants) | 14 | Iambic pentameter + rhyme scheme |
| **Villanelle** | 19 | ABA rhyme + two refrains |
| **Sestina** | 39 | End-word rotation across 6 stanzas |
| **Pantoum** | Variable | Interlocking quatrains |
| **Ghazal** | Variable | Couplets with radif + qafiya |

### Stanza Forms
| Form | Lines/Stanza | Rhyme Scheme |
|------|--------------|--------------|
| **Ottava Rima** | 8 | ABABABCC |
| **Rhyme Royal** | 7 | ABABBCC |
| **Spenserian Stanza** | 9 | ABABBCBCC + alexandrine |
| **Ballad Stanza** | 4 | ABCB, 8-6-8-6 syllables |
| **Burns Stanza** | 6 | AAABAB |

### Shape/Visual Poetry
| Form | Pattern |
|------|---------|
| **Diamante** | 1-2-3-4-3-2-1 words |
| **Cinquain** | 2-4-6-8-2 syllables |
| **Etheree** | 1-2-3-4-5-6-7-8-9-10 syllables |

### And Many More
Quatrain, Couplet (heroic, short, elegiac), Blank Verse, Ode (Pindaric, Horatian, Irregular), Ballad, Kyrielle, Epigram, Tercet, Rubaiyat, Free Verse, Rondeau, Triolet, Ballade, Blues Poem, Clerihew, Limerick...

---

## For LLM Training

### Reward Function Integration

```python
from abide.forms import Villanelle
from abide.verifiers import RewardEnvironment

# Create reward environment for your RL framework
env = RewardEnvironment(
    form=Villanelle(),
    reward_scale=(0.0, 1.0),  # Continuous rewards
)

# In your training loop
def compute_reward(generated_poem: str) -> float:
    return env.score(generated_poem)
```

### Prompt-Reward Alignment

The same specification generates both prompts and rewards:

```python
from abide.specs import villanelle_spec

spec = villanelle_spec()

# Generate training prompt
prompt = spec.full_instruction()
# "Write a Villanelle with the following requirements:
#  - Write exactly 19 lines in 6 stanzas (5 tercets + 1 quatrain)
#  - Follow the rhyme scheme ABA ABA ABA ABA ABA ABAA
#  - Line 1 must be repeated exactly at lines 6, 12, 18
#  - Line 3 must be repeated exactly at lines 9, 15, 19"

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
# Using uv (recommended)
uv add abide

# Using pip
pip install abide
```

## Quick Start

```python
from abide.forms import Haiku
from abide import verify

poem = """An old silent pond
A frog jumps into the pond
Splash! Silence again"""

result = verify(poem, Haiku())
print(f"Score: {result.score:.2%}")  # Score: 89%

# See what matched and what didn't
for item in result.rubric:
    print(f"  {item.criterion}: {'PASS' if item.passed else 'FAIL'} ({item.score:.0%})")
```

## Development

```bash
git clone https://github.com/darrenangle/abide.git
cd abide
uv sync
uv run pytest  # 435 tests
```

## Architecture

```
abide/
├── primitives/     # NLP tools: syllables, phonetics, rhyme, meter
├── constraints/    # Composable constraint types
├── forms/          # Pre-built form templates (50+)
├── specs/          # FormSpec for instruction generation
├── inference/      # Reverse-engineer forms from poems
└── verifiers/      # RL framework integration
```

## License

MIT

---

**Abide**: Because poetry has rules, and rules can be rewards.
