# Generate SFT Training Data

Generate verified poetry SFT data with dual reasoning traces (SYNTH + natural).

## Arguments
$ARGUMENTS

Parse the arguments to extract:
- `--forms`: Comma-separated list of poetry forms (default: top 10 learnable)
- `--num`: Number of examples per form (default: 100)
- `--min-score`: Minimum abide score to accept (default: 0.8)
- `--model-mix`: Ratio of Claude agents vs OpenRouter (0-1, default: 0.5)
- `--openrouter-model`: Model for OpenRouter (default: moonshotai/kimi-k2)
- `--output`: Output JSONL file (default: data/sft_dataset.jsonl)
- `--backend`: Force backend: openrouter, claude, or mixed

## Task

You are orchestrating SFT data generation for poetry forms. For each form:

1. **Spawn parallel haiku agents** (if using Claude backend):
   - Each agent generates poems for its assigned form
   - Agent verifies with abide until score >= min_score
   - Agent produces both SYNTH and natural reasoning traces

2. **Run OpenRouter workers** (if using OpenRouter backend):
   - Use the openrouter_generator.py script
   - Models: Kimi K2, DeepSeek, etc.
   - Same verification loop with abide

3. **Output format** (append-only JSONL):
```json
{
  "form": "Sonnet",
  "prompt": "Write a Sonnet about autumn in a melancholic tone",
  "synth_trace": "<think>\nSonnet requirements:\n├─ lines: 14 ●\n...",
  "natural_trace": "<think>\nI need to write a Sonnet about autumn...",
  "poem": "When autumn's gentle breath...",
  "score": 0.92,
  "rubric": [...],
  "model": "claude-haiku",
  "timestamp": "2024-12-29T22:30:00Z"
}
```

## Execution

Run the orchestrator with parsed arguments:

```bash
python scripts/sft_orchestrator.py \
  --forms "$FORMS" \
  --num "$NUM" \
  --min-score "$MIN_SCORE" \
  --model-mix "$MODEL_MIX" \
  --output "$OUTPUT"
```

Or for specific backends:
- OpenRouter only: `--backend openrouter --model moonshotai/kimi-k2`
- Claude only: `--backend claude`

## SYNTH Trace Format

The SYNTH traces use Baguettotron's stenographic notation:

**Logical markers:**
- `→` derivation/implication
- `↺` iterative refinement
- `?` uncertainty
- `!/※` insight/breakthrough
- `≈` approximation
- `∴` conclusion

**Confidence markers:**
- `●` high confidence
- `◐` medium confidence
- `○` low confidence
- `⚠` warning/risk

**Verification:**
- `☐` unverified
- `☑` partial verification
- `✓` confirmed

**Structure:**
```
<think>
FormName requirements:
├─ constraint1: ●
├─ constraint2: ◐
└─ topic: theme

### 1. Task Analysis
→ planning steps...

### 2. Composition Strategy
⟨H≈0.5⟩ exploring options...

### 3. Verification
✓ line_count: 100%
☑ syllables: 85%

∴ FormName complete: ● high confidence
</think>
```

## Progress Tracking

Report progress periodically:
- `[N/total] FormName: score=X.XX (model)`
- Success/failure counts
- Rate (poems/minute)
- ETA for completion

## Example Usage

```
/generate-sft --forms Sonnet,Haiku --num 10 --model-mix 0.7
```

This generates 20 poems total (10 Sonnets + 10 Haikus), with 70% using Claude agents and 30% using OpenRouter.
