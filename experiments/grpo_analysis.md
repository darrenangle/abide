# GRPO Poetry Training Analysis & Ablation Plan

## Current Situation Summary

Training runs plateau or "go sideways" after ~330 steps with no meaningful improvement on poetry form verification.

## Research Findings

### 1. Reference GRPO Configurations

| Paper/System | Rollouts | Batch Size | Learning Rate | KL Coef (β) | Steps | Notes |
|--------------|----------|------------|---------------|-------------|-------|-------|
| DeepSeekMath | 64 | 1024 | 1e-6 | 0.04 | ? | Original GRPO paper |
| Mini-R1 (Phil Schmid) | 8-64 | 256 | 5e-7 | 0.001 | 500 | Had to reduce LR/β for stability |
| Nemotron-3 | 8→16 | ? | 1e-6 | varies | 350+ | Used curriculum, filtered dataset |
| **Our Current** | 16 | 32 | 1e-6 | **NONE** | 3125 | No KL regularization! |

### 2. Key Insights from Literature

#### A. The KL Problem
The verifiers library **has no KL coefficient/beta parameter**. This is a major deviation from standard GRPO.

From GTPO paper: "Standard GRPO without KL is highly unstable: entropy quickly spikes and then collapses"

From M-GRPO paper: "Existing methods suffer from a critical failure mode under long-horizon training: a 'policy collapse' where performance precipitously degrades."

#### B. Entropy Collapse
- Off-policy experiments show entropy collapse after ~150 steps
- The library logs `entropy` and `mismatch_kl` but doesn't use them for regularization
- We should monitor these closely

#### C. Dataset Filtering (Critical!)
From Nemotron/AceMath-RL:
> "With training set accuracy around 90%+, they found that **70% of questions were achieving 100% accuracy within 16 rollouts**, providing no advantage for GRPO training. They **filtered the dataset from 46K to 2.2K samples** based on the model's pass rate to focus on the most challenging problems."

This is huge: if most prompts in our mix are "easy" (getting high rewards), there's no meaningful signal for GRPO.

#### D. Poetry is Not Math
From Writing-Zero paper:
> "A significant gap remains for non-verifiable tasks, like creative writing and open-ended dialogue, where quality assessment is inherently subjective."

Our constraint verification IS verifiable, but the nature of poetry generation is different from math:
- Math: Many wrong paths, one right answer
- Poetry: Many right paths to the same form constraints

#### E. The "Exploration" Problem
From RLVR research:
> "RLVR does not make models more capable; it merely makes them more reliable. In the process of promoting reliability, RLVR limits the model's exploration."

If the model doesn't already know how to write sonnets reasonably well, GRPO won't teach it.

### 3. Hypothesis: Why Training Goes Sideways

**Primary Hypothesis**: The verifiers library's GRPO implementation lacks KL regularization, leading to:
1. Initial improvement as model learns obvious patterns
2. Entropy collapse as model becomes overconfident
3. Reward hacking / mode collapse on easy solutions
4. No further learning

**Secondary Hypothesis**: Our task mix is too diverse:
- 82 traditional forms with very different structures
- Model can't develop expertise in any one form
- Advantage signal gets diluted across too many objectives

**Tertiary Hypothesis**: The model lacks sufficient poetry grounding:
- Base model may not have enough sonnet examples in pretraining
- GRPO can only amplify existing capabilities, not create new ones

## Proposed Ablations

### Ablation 1: Single Form Training (Spenserian Sonnet Only)

**Rationale**: Simplify to see if learning is possible at all
**Changes**:
- Only train on Spenserian Sonnet prompts
- ~3000-5000 prompts (one form, many topics)
- Same hyperparameters

**Success Metric**: Reward increase > 0.1 over 500 steps

### Ablation 2: Lower Learning Rate

**Rationale**: Mini-R1 had to use 5e-7 instead of 1e-6 for stability
**Changes**:
- LR: 5e-7 (half current)
- Same mix

**Success Metric**: More stable reward curve, no collapse

### Ablation 3: Fewer Rollouts + Larger Batch

**Rationale**: DeepSeekMath uses 64 rollouts per example, batch 1024
**Changes**:
- 8 rollouts per example (instead of 16)
- Batch size 64 (instead of 32)
- More gradient updates per compute

**Success Metric**: Faster iteration, potentially more stable

### Ablation 4: Easy Forms Only

**Rationale**: Filter to forms the model CAN already do
**Changes**:
- Only include: Haiku, Limerick, Quatrain, Couplet
- Simple structures, well-represented in training data
- ~25000 prompts

**Success Metric**: Higher baseline reward, clearer learning signal

### Ablation 5: Pre-filter by Baseline Performance

**Rationale**: Like Nemotron filtering to challenging samples
**Changes**:
- Generate baseline completions for each prompt
- Only keep prompts where baseline score is 0.2-0.8
- Skip "too easy" (>0.8) and "too hard" (<0.2)

**Success Metric**: More consistent advantage signal

## Monitoring Plan

For each ablation, track:
1. `train/reward` - mean reward (should increase)
2. `reward/std` - within-batch variance (should stay moderate)
3. `entropy` - should stay stable, not collapse to 0
4. `mismatch_kl` - should stay low (<1.0)
5. `importance_ratio` - should stay near 1.0

## Ablation Results Log

### Ablation 1: Spenserian Sonnet Only (2025-12-27)

**Status**: Killed after analysis - hypothesis confirmed

**Step 10 Metrics Comparison**:

| Metric | Single Form (Spenserian) | Multi-Form (82 forms) |
|--------|--------------------------|----------------------|
| reward | 0.847 | 0.703 |
| reward/std | 0.044 | 0.208 |
| advantage/absmean | **0.038** | **0.039** |

**Critical Finding**: The advantage signal is nearly identical (~0.038) in both cases!

The multi-form run has higher reward/std (0.21) but this is **between-form variance**
(ChantRoyal=0.05, Sonnet=0.85), NOT within-rollout variance.

GRPO computes advantage **per prompt group**: all 16 rollouts for the same prompt
are compared to their own mean. So:
- ChantRoyal prompt: 16 rollouts all score ~0.05 → advantage ≈ 0
- Sonnet prompt: 16 rollouts all score ~0.85 → advantage ≈ 0

## Fundamental Conclusion

**GRPO is not suitable for teaching poetic forms** (without SFT warmup) because:

1. **GRPO needs within-rollout variance** - for the same prompt, some attempts succeed,
   some fail. That's where the advantage signal comes from.

2. **Poetry forms lack this variance** - the model either:
   - Knows the form → all 16 rollouts score ~0.85
   - Doesn't know the form → all 16 rollouts score ~0.05

3. **Math works because it has variance** - a model might solve a problem correctly
   in 3/16 rollouts, giving clear signal about what works.

4. **The RLVR insight applies**: "RLVR does not make models more capable; it merely
   makes them more reliable." GRPO can't teach ChantRoyal if the model has no idea
   how to write one.

## Alternative Approaches

1. **SFT first** - Teach forms with supervised examples, then GRPO for refinement
2. **Rejection sampling** - Generate N samples, pick best (no training)
3. **DPO with synthetic pairs** - Create good/bad examples for preference learning
4. **Weaker base model** - A model "on the edge" might have useful variance
5. **More granular rewards** - Score each constraint separately to create variance

---

## Next Steps

1. ~~Stop current run~~
2. ~~Run Ablation 1 (Single Form) for 500 steps~~ (Running)
3. Analyze results
4. Decide next ablation based on findings

## Key Questions to Answer

1. **Can the model learn ANY form well with GRPO?** (Ablation 1)
2. **Is the learning rate causing instability?** (Ablation 2)
3. **Would more focused training help?** (Ablation 4)
4. **Is the base model capable of writing sonnets at all?** (Test with prompting)

---

### Ablation: Learnable Forms Only (2025-12-27)

**Hypothesis**: Train only on forms with high within-rollout variance (found via `find_learnable_forms.py`)

**Setup**:
- 10 forms with highest GRPO signal (within_std > 0.1, 0.2 < mean < 0.8)
- Epigram, ThunderVerse, ColorSpectrum, CoprimeVerse, ElementalVerse,
  CharacterPalindromePoem, QuestionQuest, VowelPilgrimage, Mesostic, Terzanelle
- 10,000 prompts (1,000 per form), 16 rollouts, batch_size=16

**Wandb**: https://wandb.ai/darren-angle-tomorrow-computing-company/abide-grpo/runs/yraq1p6v

**Step 10 Metrics**:
- reward: 0.42
- reward/std: 0.24
- advantage/absmean: **0.21** (5x higher than traditional forms!)
- entropy: 0.35
- mismatch_kl: 0.29

**Step 20 Metrics**:
- reward: 0.57
- advantage/absmean: 0.20

**Observation**: These forms DO produce meaningful advantage signal (~0.20 vs ~0.04 for traditional forms).
However, rewards are still fluctuating without clear upward trend.

**Root Cause Finding**:
Examined verifiers library source code at `/home/darren/poetry-rm-darren/verifiers/verifiers/rl/trainer/trainer.py`:

```python
# Line 262 - THE LOSS FUNCTION
loss = (-importance_ratio * advantages)[keep_mask].sum()

# Line 264 - mismatch_kl computed but ONLY for logging
mismatch_kl = torch.exp(log_importance_ratio) - log_importance_ratio - 1
```

**Critical Finding**: The verifiers GRPO implementation has **NO KL regularization**!

Standard GRPO loss should be:
```python
loss = -importance_ratio * advantages + beta * mismatch_kl
```

But verifiers only computes the first term. This explains:
1. Why even forms with variance don't produce stable learning
2. Why entropy can drift without constraint
3. Why policy can diverge without bounds

**Possible Fixes**:
1. Fork verifiers, add `+ self.kl_coef * mismatch_kl.mean()` to loss
2. Try higher learning rate (default is 1e-5, we use 1e-6)
3. Switch to a GRPO implementation with proper KL (TRL, etc.)

---

## Sources

- [Nemotron 3 Nano Technical Report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf)
- [AceMath-RL Paper](https://research.nvidia.com/labs/adlr/acemath_rl/)
- [Mini-R1 by Phil Schmid](https://www.philschmid.de/mini-deepseek-r1)
- [GTPO: Stabilizing GRPO](https://arxiv.org/abs/2508.03772)
- [M-GRPO: Momentum-Anchored GRPO](https://arxiv.org/abs/2512.13070)
- [Writing-Zero](https://arxiv.org/abs/2506.00103)
- [GRPO Collapse in Search-R1](https://arxiv.org/abs/2512.04220)
