# GitHub Copilot Instructions for Abide

## Project Overview

**Abide** is a framework for specification of poetic instructions that can be auto-verified in RL runs. It creates verifiable or spectrum rewards for poetry adherence to strict forms like sestinas and villanelles.

**Key Features:**
- Structural parsing for poetic forms
- End-word extraction and pattern matching (sestinas)
- Rhyme detection via CMU pronouncing dictionary
- Refrain verification (villanelles)
- Fuzzy matching with multiple similarity algorithms
- Spectrum rewards with uncertainty quantification

## Tech Stack

- **Language**: Python 3.10+
- **Core Algorithms**: Levenshtein, Jaro-Winkler, Soundex, Metaphone
- **Phonetic Data**: CMU Pronouncing Dictionary
- **Target Integration**: `verifiers` framework, ART RL framework

## Refactor Tracking

**CRITICAL**: Use the markdown tracker in `history/refactor/` for the current reliability and refactor work.

### Workflow

1. Check `history/refactor/BOARD.md`
2. Claim work by creating `history/refactor/locks/<TICKET-ID>--<agent>.md`
3. Update the matching file in `history/refactor/tickets/`
4. Append a short note to `history/refactor/WORKLOG.md`
5. Delete the lock file when the ticket is done

### Priorities

- `P1` - correctness, reward integrity, training reliability
- `P2` - important follow-up or support work
- `P3` - cleanup or lower-risk backlog

## Project Structure (Planned)

```
abide/
├── abide/
│   ├── __init__.py
│   ├── preprocessing.py     # Whitespace normalization
│   ├── structure.py         # Line/stanza parsing
│   ├── similarity.py        # String similarity algorithms
│   ├── phonetics.py         # Soundex, Metaphone, CMU dict
│   ├── forms/
│   │   ├── __init__.py
│   │   ├── sestina.py       # Sestina verification
│   │   └── villanelle.py    # Villanelle verification
│   ├── rewards.py           # Reward aggregation
│   └── environment.py       # Main reward environment
├── tests/
├── examples/
└── history/
    └── refactor/            # Markdown ticket tracker, locks, and worklog
```

## Important Rules

- Use `history/refactor/` for current refactor tracking
- Do NOT create markdown TODO lists

---

**For detailed workflows and advanced features, see [AGENTS.md](../AGENTS.md)**
