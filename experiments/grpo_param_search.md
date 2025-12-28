# GRPO Parameter Search for Baguettotron

## Hypothesis
Different combinations of `max_tokens` and `repetition_penalty` will yield different initial reward distributions. We hypothesize that:
1. Higher max_tokens allows more reasoning space but may enable more degenerate outputs
2. Repetition penalty helps break loops but too high may hurt valid poetic repetition
3. There exists an optimal combination that maximizes mean reward on 128 rollouts

## Method
- Model: Baguettotron SFT (`/home/darren/10k-poems/models/baguettotron_sft/final`)
- Rollouts per experiment: 128
- Fixed params: temperature=0.6, top_p=0.95, learning_rate=1e-6
- Variable params:
  - max_tokens: [512, 1024, 2048]
  - repetition_penalty: [1.0, 1.1, 1.15, 1.2, 1.3]
- Metric: Mean reward across 128 rollouts
- Confirmation: Best params repeated 3x to verify consistency

## Experiment Log

### Experiment 1
- **Timestamp**: (pending)
- **Params**: max_tokens=1024, repetition_penalty=1.0 (baseline)
- **Mean Reward**: (pending)
- **Notes**: (pending)

### Experiment 2
- **Timestamp**: (pending)
- **Params**: max_tokens=1024, repetition_penalty=1.15
- **Mean Reward**: (pending)
- **Notes**: (pending)

### Experiment 3
- **Timestamp**: (pending)
- **Params**: max_tokens=1024, repetition_penalty=1.3
- **Mean Reward**: (pending)
- **Notes**: (pending)

### Experiment 4
- **Timestamp**: (pending)
- **Params**: max_tokens=2048, repetition_penalty=1.0
- **Mean Reward**: (pending)
- **Notes**: (pending)

### Experiment 5
- **Timestamp**: (pending)
- **Params**: max_tokens=2048, repetition_penalty=1.15
- **Mean Reward**: (pending)
- **Notes**: (pending)

### Experiment 6
- **Timestamp**: (pending)
- **Params**: max_tokens=2048, repetition_penalty=1.3
- **Mean Reward**: (pending)
- **Notes**: (pending)

### Experiment 7
- **Timestamp**: (pending)
- **Params**: max_tokens=512, repetition_penalty=1.15
- **Mean Reward**: (pending)
- **Notes**: (pending)

### Experiment 8
- **Timestamp**: (pending)
- **Params**: max_tokens=2048, repetition_penalty=1.2
- **Mean Reward**: (pending)
- **Notes**: (pending)

## Confirmation Runs
(Best params repeated 3x)

### Confirmation 1
- **Params**: (pending)
- **Mean Reward**: (pending)

### Confirmation 2
- **Params**: (pending)
- **Mean Reward**: (pending)

### Confirmation 3
- **Params**: (pending)
- **Mean Reward**: (pending)

## Analysis
(To be filled after experiments)

## Conclusion
(To be filled after analysis)

## Final Parameters
(To be applied to train_grpo.py)

---

## Final Results (Tue Dec 23 21:12:05 CST 2025)

### Best Parameters Found
- **max_tokens**: 2048
- **repetition_penalty**: 1.2
- **mean_reward**: 0.1970

### All Experiment Results
```
max_tokens=1024, rep_penalty=1.0: reward=0.1147 (43.8% nonzero)
max_tokens=1024, rep_penalty=1.15: reward=0.1720 (65.2% nonzero)
max_tokens=1024, rep_penalty=1.3: reward=0.1721 (69.5% nonzero)
max_tokens=2048, rep_penalty=1.0: reward=0.1063 (43.8% nonzero)
max_tokens=2048, rep_penalty=1.15: reward=0.1955 (79.7% nonzero)
max_tokens=2048, rep_penalty=1.3: reward=0.1659 (71.5% nonzero)
max_tokens=512, rep_penalty=1.15: reward=0.0251 (10.9% nonzero)
max_tokens=2048, rep_penalty=1.2: reward=0.1970 (82.4% nonzero)
```

### Confirmation Results
```
Run 1: reward=0.2026 (84.4% nonzero)
Run 2: reward=0.2183 (80.1% nonzero)
Run 3: reward=0.2148 (78.8% nonzero)
```

### Applied to train_grpo.py
Parameters have been updated in the training script.
