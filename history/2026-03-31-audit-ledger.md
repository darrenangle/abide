# Abide Audit Ledger

Date: 2026-03-31
Scope: static audit plus targeted reproductions for reward semantics, form verification, and training integration
Status: findings only, no code fixes applied in this pass

## Baseline

- `uv run pytest`: 435 passed, 1 skipped
- `uv run ruff check src/abide tests`: clean
- `uv run mypy src/abide`: clean

The important conclusion is that the current suite is proving happy-path behavior, not protecting the training stack from false positives or bad reward assumptions.

## Summary Table

| ID | Severity | Area | Short finding | Suggested tracker action |
| --- | --- | --- | --- | --- |
| AUDIT-001 | High | core scoring semantics | `passed` is often just `weighted_score >= 0.6`, so canonical forms can pass without their defining feature | `RF-001` |
| AUDIT-002 | High | prosody/form fidelity | Multiple forms claim meter or unrhymed behavior but only enforce syllable counts | `RF-003`, `RF-004`, `RF-008` |
| AUDIT-003 | High | ghazal verifier | Odd trailing lines are ignored because couplets are computed with floor division | `RF-005` |
| AUDIT-004 | High | TRL training path | TRL reward infers form from prompt substring and can score against the wrong form | `RF-006` |
| AUDIT-005 | Medium | packaging / training entrypoint | `train_grpo_trl.py` depends on undeclared packages and fails at import time in a normal env | `RF-007` |
| AUDIT-006 | Medium | regression coverage | Tests are mostly positive examples and do not lock down the false positives below | `RF-002` |

## Findings

### AUDIT-001: Composite form `passed` semantics are too lenient

- Severity: High
- Tracker ticket: `RF-001`
- Primary files:
  - `src/abide/constraints/operators.py`
  - `src/abide/forms/villanelle.py`
  - `src/abide/forms/sonnet.py`

Why this matters:

- RL reward is one thing; the boolean `passed` flag is another.
- Right now many form classes treat `passed` as "average weighted score cleared 0.6".
- That allows outputs to be labeled as valid examples of a form even when the defining property is missing.

Evidence from code:

- `WeightedSum.verify()` sets `overall_passed = overall_score >= self.threshold` and never checks child `passed` values.
- Reference:
  - `src/abide/constraints/operators.py:210-240`
- `Villanelle` defaults to `strict=False` and uses `WeightedSum(..., threshold=0.6)`.
- Reference:
  - `src/abide/forms/villanelle.py:105-129`
- `ShakespeareanSonnet` also defaults to `strict=False` with the same pattern.
- Reference:
  - `src/abide/forms/sonnet.py:138-155`

Concrete reproductions:

1. Villanelle with correct stanza pattern and rhyme endings, but no refrain repetition:

```python
from abide.forms import Villanelle

poem = """alpha cat
beta bat
gamma cat

delta cat
epsilon bat
zeta cat

eta cat
theta bat
iota cat

kappa cat
lambda bat
mu cat

nu cat
xi bat
omicron cat

pi cat
rho bat
sigma cat
tau cat"""

print(Villanelle().verify(poem).score, Villanelle().verify(poem).passed)
```

Observed result:

- `score = 0.62`
- `passed = True`

2. Repeated-line Shakespearean sonnet toy poem:

```python
from abide.forms import ShakespeareanSonnet

line_a = "The winter sunlight settles on the bay"
line_b = "A silver morning wanders through the rain"
poem = "\n".join([line_a, line_b] * 7)

print(ShakespeareanSonnet().verify(poem).score, ShakespeareanSonnet().verify(poem).passed)
```

Observed result:

- `score = 0.8077922077922078`
- `passed = True`

Acceptance criteria:

- Separate `reward_score` from canonical `passed`.
- Introduce hard gates for defining constraints on each form.
- Add adversarial tests proving the reproductions above fail.

Refactor note:

- This is the highest-leverage fix in the repository. If `passed` stays lenient, every downstream eval, filter, and training report will keep lying.

### AUDIT-002: Form descriptions overclaim what is actually verified

- Severity: High
- Tracker tickets: `RF-003`, `RF-004`, `RF-008`
- Primary files:
  - `src/abide/forms/sonnet.py`
  - `src/abide/forms/blank_verse.py`
  - `src/abide/forms/couplet.py`
  - `src/abide/forms/ottava_rima.py`

Why this matters:

- Several forms promise meter-sensitive verification but only count syllables.
- That trains models toward the wrong target while the descriptions claim literary accuracy.

Evidence from code:

- `Sonnet` docstring says "14 lines of iambic pentameter" but the implementation only uses `SyllablesPerLine([10] * 14)`.
- Reference:
  - `src/abide/forms/sonnet.py:26-88`
- `BlankVerse` says "unrhymed iambic pentameter", but the default path only checks line count plus 10 syllables per line. There is no anti-rhyme constraint at all unless the caller manually chooses a different setup, and even `strict_meter=True` still does not enforce "unrhymed".
- Reference:
  - `src/abide/forms/blank_verse.py:28-150`
- `HeroicCouplet`, `HeroicQuatrain`, `OttavaRima`, `RhymeRoyal`, and related classes repeat the same pattern: "iambic pentameter" in prose, syllable counts in code.
- References:
  - `src/abide/forms/couplet.py:92-132`
  - `src/abide/forms/quatrain.py:145-178`
  - `src/abide/forms/ottava_rima.py:26-109`

Concrete reproduction:

```python
from abide.forms import BlankVerse

poem = """The winter sunlight settles on the bay
The distant harbor lingers in the gray
The river answers slowly to the day"""

print(BlankVerse().verify(poem).score, BlankVerse().verify(poem).passed)
```

Observed result:

- `score = 1.0`
- `passed = True`

This poem is clearly rhymed and only demonstrates 10-syllable lines, not verified blank verse.

Acceptance criteria:

- Either enforce real meter/rhyme exclusions with explicit constraints, or rename descriptions and instructions to match what is actually checked.
- Add a dedicated "syllable proxy" vocabulary if meter enforcement is intentionally approximate.
- Add negative tests for rhymed blank verse and non-metrical 10-syllable sonnets.

### AUDIT-003: `Ghazal` ignores odd trailing lines

- Severity: High
- Tracker ticket: `RF-005`
- Primary file:
  - `src/abide/forms/ghazal.py`

Why this matters:

- A ghazal is defined by couplets.
- The current verifier silently drops an unmatched final line because it uses floor division to compute couplet count.

Evidence from code:

- `num_couplets = structure.line_count // 2`
- All later checks iterate over those inferred couplets only.
- No rubric item or penalty is added for an unmatched final line.
- Reference:
  - `src/abide/forms/ghazal.py:76-110`
  - `src/abide/forms/ghazal.py:164-254`

Concrete reproduction:

```python
from abide.forms import Ghazal

poem = """soft light moon
cold night moon

green sight moon
warm night moon

small flight moon
deep night moon

bright kite moon
still night moon

faint white moon
calm night moon

EXTRA DANGLING LINE"""

result = Ghazal().verify(poem)
print(result.score, result.passed, result.details)
```

Observed result:

- `score = 1.0`
- `passed = True`
- `details = {'couplet_count': 5, 'radif': 'moon'}`

The eleventh line is ignored.

Acceptance criteria:

- Fail or sharply penalize odd line counts.
- Prefer stanza-based couplet partitioning over `line_count // 2`.
- Add a regression test for a perfect 10-line ghazal plus one extra garbage line.

### AUDIT-004: TRL reward assignment can target the wrong form

- Severity: High
- Tracker ticket: `RF-006`
- Primary files:
  - `scripts/train_grpo_trl.py`
  - `src/abide/forms/__init__.py`

Why this matters:

- This contaminates reward during training.
- The completion may be scored against a different form than the prompt requested.

Evidence from code:

- `create_dataset()` only stores a `prompt` column and drops structured `form_name` metadata.
- Reference:
  - `scripts/train_grpo_trl.py:336-367`
- `create_reward_function()` then guesses the form by substring search over the prompt text.
- Reference:
  - `scripts/train_grpo_trl.py:298-304`
- `__all__` ordering puts `Ode` before `PindaricOde`.
- Reference:
  - `src/abide/forms/__init__.py:234-267`

Concrete reproduction of the matching logic:

```python
import abide.forms as forms_module

prompt = "Write a PindaricOde poem about winter. Requirements: ..."
matched = None
for name in forms_module.__all__:
    if name.lower() in prompt.lower() or name in prompt:
        matched = name
        break
print(matched)
```

Observed result:

- `matched = "Ode"`

That means a `PindaricOde` training prompt can be rewarded as plain `Ode`.

Acceptance criteria:

- Carry `form_name` through the TRL dataset as a dedicated column.
- Read that exact value inside the reward function.
- Ban prompt parsing for reward identity.
- Add a regression test covering `Ode` / `PindaricOde` and any other overlapping names.

### AUDIT-005: TRL entrypoint dependencies are not declared cleanly

- Severity: Medium
- Tracker ticket: `RF-007`
- Primary files:
  - `scripts/train_grpo_trl.py`
  - `pyproject.toml`

Why this matters:

- Clean setup is part of reliability.
- The script currently assumes a larger environment than the project metadata advertises.

Evidence from code:

- `train_grpo_trl.py` imports `torch`, `datasets`, `peft`, `transformers`, and `trl` at module import time.
- Reference:
  - `scripts/train_grpo_trl.py:34-43`
- The `training` extra only declares:
  - `verifiers[rl]`
  - `wandb`
- Reference:
  - `pyproject.toml:50-54`

Observed failure in the current environment:

```python
import scripts.train_grpo_trl
```

Raised:

- `ModuleNotFoundError: No module named 'transformers'`

Acceptance criteria:

- Add an explicit extra for the TRL path, or expand `training` so it actually installs the script's runtime dependencies.
- Move heavyweight imports into `main()` or the specific codepaths that need them so helper functions remain importable.
- Add one smoke test that imports the module in a training-enabled environment.

### AUDIT-006: The current tests do not protect against the false positives above

- Severity: Medium
- Tracker ticket: `RF-002`
- Primary files:
  - `tests/forms/test_forms.py`
  - `tests/fixtures/test_form_validation.py`

Why this matters:

- The suite is green while the reproduced issues above still exist.
- That means the tests are not measuring the failure modes you care about in training.

Evidence from tests:

- `tests/forms/test_forms.py` focuses on positive examples and gross length mismatches.
- Reference:
  - `tests/forms/test_forms.py:131-204`
- `tests/fixtures/test_form_validation.py` is heavily skewed toward synthetic perfect examples and permissive score thresholds.
- Reference:
  - `tests/fixtures/test_form_validation.py:134-277`

Missing regressions:

- Villanelle with no refrain repetition should fail.
- Shakespearean sonnet with repeated ABAB lines should fail.
- Rhymed blank verse should fail.
- Ghazal with an extra dangling line should fail.
- TRL reward mapping should use exact form metadata.

Acceptance criteria:

- Add one failing adversarial fixture for every reproduced bug in this ledger.
- Add an integration test for the TRL dataset/reward path.
- Require negative cases for every future form audit ticket before closing it.

## Refactor Path

### Phase 1: Fix semantics before adding more forms

1. Split `score` from canonical `passed`.
2. Introduce hard-gated defining constraints for each form class.
3. Add adversarial tests for the reproduced false positives.

### Phase 2: Make form claims truthful

1. Audit every class that claims meter, refrain, end-word rotation, or unrhymed behavior.
2. Either enforce the property directly or downgrade the description to the approximated property.
3. Standardize a vocabulary: `meter-checked`, `syllable-checked`, `rhyme-checked`, `structure-only`.

### Phase 3: Repair training plumbing

1. Stop inferring form identity from prompt text.
2. Pass structured metadata through TRL and verifiers paths.
3. Put every reward extraction function behind a small integration test with recorded examples.

### Phase 4: Reconcile issue tracking

Relevant current tracker tickets:

- `RF-001`: composite pass/fail semantics
- `RF-002`: adversarial regressions
- `RF-003`: sonnet-family claim alignment
- `RF-004`: blank-verse and metrical-form claim alignment
- `RF-005`: ghazal dangling-line fix
- `RF-006`: TRL form metadata
- `RF-007`: TRL dependency hygiene
- `RF-008`: follow-up claim audit
