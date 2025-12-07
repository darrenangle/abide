# Abide

Composable constraint algebra for specifying and auto-verifying poetic forms.

## Overview

Abide provides a framework for defining poetic form constraints and verifying whether text adheres to those constraints. It produces spectrum rewards (0-1 scores with uncertainty) suitable for reinforcement learning, and includes auto-generated rubrics explaining verification results.

## Features

- **Composable Constraints**: Build complex forms from primitive constraints using logical operators (AND, OR, NOT, WeightedSum)
- **Spectrum Rewards**: Continuous scores with configurable strictness curves, not just pass/fail
- **Auto-Rubrics**: Human-readable explanations of what matched and what didn't
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
- **Ghazal, Pantoum, Blues Poem**, and more

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
from abide.constraints.operators import AND, WeightedSum

# Define a custom form
my_form = WeightedSum([
    (LineCount(exact=8), 0.2),
    (StanzaCount(exact=2), 0.1),
    (StanzaSizes([4, 4]), 0.2),
    (RhymeScheme("ABABCDCD"), 0.3),
    (SyllablesPerLine([8] * 8), 0.2),
])

result = verify(poem, my_form)
```

## Development

```bash
# Clone and install
git clone https://github.com/darren/abide.git
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
