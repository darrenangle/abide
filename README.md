# Abide

Composable constraint algebra for specifying and auto-verifying poetic forms.

## Overview

Abide provides a framework for defining poetic form constraints and verifying whether text adheres to those constraints. It produces spectrum rewards (0-1 scores with uncertainty) suitable for reinforcement learning, and includes auto-generated rubrics explaining verification results.

## Features

- **Composable Constraints**: Build complex forms from primitive constraints using logical operators (AND, OR, NOT, WeightedSum)
- **Spectrum Rewards**: Continuous scores with configurable strictness curves, not just pass/fail
- **Auto-Rubrics**: Human-readable explanations of what matched and what didn't
- **Plain English Instructions**: Each constraint maps to a natural language instruction for LLM evaluation
- **Real NLP**: Uses CMU Pronouncing Dictionary, phonetic encoders (Soundex, Metaphone), and fuzzy matching
- **Verifiers Compatible**: Designed for integration with the verifiers framework and ART RL

## Installation

```bash
# Using uv (recommended)
uv add abide

# Using pip
pip install abide
```

## Quick Start

```python
from abide.forms import Haiku, Villanelle, ShakespeareanSonnet
from abide import verify

poem = """An old silent pond
A frog jumps into the pond
Splash! Silence again"""

result = verify(poem, Haiku())
print(f"Score: {result.score:.2f}")
print(result.rubric)
```

## Supported Forms

- **Haiku**: 5-7-5 syllable structure
- **Tanka**: 5-7-5-7-7 syllable structure
- **Villanelle**: 19 lines with ABA rhyme and two refrains
- **Sestina**: 39 lines with end-word rotation
- **Sonnet**: Shakespearean (ABABCDCDEFEFGG), Petrarchan (ABBAABBA + sestet), Spenserian
- **Limerick**: AABBA rhyme scheme
- **Triolet**: 8 lines with ABaAabAB rhyme and refrains
- **Pantoum**: Interlocking quatrains with line repetition
- **Terza Rima**: ABA BCB CDC chain rhyme pattern
- **Ghazal**: Couplets with radif (refrain) and qafiya (rhyme)
- **Rondeau**: 15 lines with rentrement (opening phrase refrain)
- **Ballade**: 28 lines in 3 octaves plus envoi
- **Blues Poem**: AAB tercets with line repetition
- **Clerihew**: 4-line biographical humor with AABB rhyme

## Form Specifications with Plain English Instructions

Abide's `FormSpec` class maps programmatic constraints to plain English instructions, enabling:

- **Composable instructions**: Each constraint generates a natural language directive
- **Subset testing**: Test individual constraints in isolation
- **LLM evaluation**: Generate prompts that match your verification rubric

### Basic Usage

```python
from abide.specs import shakespearean_sonnet_spec, haiku_spec

# Get the full specification
spec = shakespearean_sonnet_spec()

# Generate plain English instructions for LLM prompts
print(spec.full_instruction())
```

Output:
```
Write a Shakespearean Sonnet with the following requirements:

- Write exactly 14 lines.
- Write in iambic pentameter (approximately 10 syllables per line).
- Follow the rhyme scheme ABAB CDCD EFEF GG: three quatrains with alternating rhyme, ending in a rhyming couplet.
```

### Testing Individual Constraints

```python
from abide.specs import shakespearean_sonnet_spec

spec = shakespearean_sonnet_spec()

# Verify all constraints
results = spec.verify(poem)
for item_id, result in results.items():
    print(f"{item_id}: {result.score:.2f}")

# Verify only specific constraints
line_results = spec.verify_subset(poem, "line_count")
rhyme_results = spec.verify_subset(poem, "rhyme_scheme")

# Get instructions for specific constraints only
print(spec.instruction_for("line_count", "rhyme_scheme"))
```

### Instructions by Category

```python
from abide.specs import villanelle_spec

spec = villanelle_spec()

# Get structural constraints only
print(spec.instructions_by_category("structural"))
# Output: Write exactly 19 lines.

# Get relational constraints (rhyme, refrains)
print(spec.instructions_by_category("relational"))
```

### Weighted Scoring

```python
from abide.specs import haiku_spec

spec = haiku_spec()

# Get weighted aggregate score
score = spec.weighted_score(poem)
print(f"Weighted score: {score:.2f}")
```

### Creating Custom Specifications

```python
from abide.specs import FormSpec
from abide.constraints import LineCount, RhymeScheme, SyllablesPerLine

# Build a custom form specification
spec = (
    FormSpec(name="Custom Quatrain", description="A 4-line rhyming verse")
    .add(
        "line_count",
        LineCount(4),
        category="structural",
    )
    .add(
        "rhyme_scheme",
        RhymeScheme("ABAB"),
        instruction="Lines 1 and 3 rhyme; lines 2 and 4 rhyme.",
        category="relational",
    )
    .add(
        "syllables",
        SyllablesPerLine([8, 8, 8, 8]),
        instruction="Each line should have 8 syllables.",
        category="prosodic",
    )
)

# Generate the full instruction
print(spec.full_instruction())

# Verify a poem
results = spec.verify(poem)
```

### Available Pre-built Specifications

```python
from abide.specs import (
    haiku_spec,
    tanka_spec,
    limerick_spec,
    shakespearean_sonnet_spec,
    petrarchan_sonnet_spec,
    villanelle_spec,
    sestina_spec,
    triolet_spec,
    clerihew_spec,
)

# Each returns a FormSpec with appropriate constraints and instructions
spec = villanelle_spec(rhyme_threshold=0.6, refrain_threshold=0.9)
```

## Constraint Instructions

Every constraint has an `instruction()` method that returns plain English:

```python
from abide.constraints import LineCount, RhymeScheme, SyllablesPerLine, Refrain

# Line count
LineCount(14).instruction()
# "Write exactly 14 lines."

# Rhyme scheme
RhymeScheme("ABABCDCDEFEFGG").instruction()
# "Follow the rhyme scheme ABABCDCDEFEFGG, where lines with the same letter must rhyme..."

# Syllables per line (uniform)
SyllablesPerLine([10] * 14).instruction()
# "Each line should have approximately 10 syllables."

# Syllables per line (pattern)
SyllablesPerLine([5, 7, 5]).instruction()
# "Follow the syllable pattern 5-7-5 (syllables per line in order)."

# Refrain
Refrain(reference_line=0, repeat_at=[5, 11, 17]).instruction()
# "Line 1 must be repeated exactly (as a refrain) at lines 6, 12, 18."
```

## Custom Forms

Build your own forms from primitives:

```python
from abide.constraints import (
    LineCount,
    StanzaCount,
    StanzaSizes,
    RhymeScheme,
    SyllablesPerLine,
)
from abide.constraints.operators import WeightedSum

# Define a custom form
my_form = WeightedSum([
    (LineCount(8), 0.2),
    (StanzaCount(2), 0.1),
    (StanzaSizes([4, 4]), 0.2),
    (RhymeScheme("ABABCDCD"), 0.3),
    (SyllablesPerLine([8] * 8), 0.2),
])

result = verify(poem, my_form)
```

## Development

```bash
# Clone and install
git clone https://github.com/darrenangle/abide.git
cd abide
uv sync

# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov=src/abide

# Type checking
uv run mypy src/

# Linting
uv run ruff check src/ tests/
```

## License

MIT
