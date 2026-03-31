# Agent Instructions for Abide

Abide is a framework for specification of poetic instructions that can be auto-verified in RL runs. It provides verifiable or spectrum rewards for poetry adherence to strict forms like sestinas and villanelles.

## Project Overview

- **Purpose**: Create reward environments for RL training that verify poetry structure
- **Compatibility**: Designed to work with `verifiers` framework and ART RL framework
- **Core algorithms**: Levenshtein distance, Jaro-Winkler similarity, Soundex/Metaphone phonetics, CMU dictionary rhyme detection

## Refactor Tracking

**IMPORTANT**: Use the markdown tracker under `history/refactor/` for the current reliability and refactor work. Do not create parallel trackers elsewhere in the repo.

### Tracker Files

- `history/refactor/BOARD.md` - summary of active tickets
- `history/refactor/WORKLOG.md` - append-only ledger of work as it happens
- `history/refactor/tickets/` - one file per ticket
- `history/refactor/locks/` - one lock file per active claim

### Workflow for AI Agents

1. Check `history/refactor/BOARD.md` for available work
2. Claim a ticket by creating `history/refactor/locks/<TICKET-ID>--<agent>.md`
3. Update the matching ticket file and `BOARD.md`
4. Append a short entry to `history/refactor/WORKLOG.md`
5. When finished, delete the lock file and mark the ticket `done`

### Priority Labels

- `P1` - correctness, reward integrity, training reliability
- `P2` - important follow-up or support work
- `P3` - cleanup or lower-risk backlog

### Managing AI-Generated Planning Documents

AI assistants often create planning and design documents during development:
- PLAN.md, IMPLEMENTATION.md, ARCHITECTURE.md
- DESIGN.md, CODEBASE_SUMMARY.md, INTEGRATION_PLAN.md
- TESTING_GUIDE.md, TECHNICAL_DESIGN.md, and similar files

**Best Practice: Use a dedicated directory for these ephemeral files**

**Recommended approach:**
- Create a `history/` directory in the project root
- Store ALL AI-generated planning/design docs in `history/`
- Keep the repository root clean and focused on permanent project files
- Only access `history/` when explicitly asked to review past planning

**Example .gitignore entry (optional):**
```
# AI planning documents (ephemeral)
history/
```

**Benefits:**
- Clean repository root
- Clear separation between ephemeral and permanent documentation
- Easy to exclude from version control if desired
- Preserves planning history for archeological research
- Reduces noise when browsing the project

## Code Quality Checks

**IMPORTANT**: Before committing or pushing code, run linting and type checking:

```bash
# Run all checks (what pre-commit will run)
uv run ruff check src/abide/
uv run mypy src/abide/

# Run tests
uv run pytest
```

### Pre-commit Hooks

Pre-commit hooks are configured to run automatically:
- **ruff**: Linting and formatting
- **mypy**: Type checking
- **basic checks**: trailing whitespace, merge conflicts, etc.

Install hooks with:
```bash
uv run pre-commit install
```

After installation, checks run automatically on `git commit`. If a check fails, the commit is blocked until the issue is fixed.

### Important Rules

- **Run `uv run ruff check src/abide/` and `uv run mypy src/abide/` before committing**
- Use `history/refactor/` as the source of truth for active refactor work
- Store AI planning docs in `history/` directory
- Do NOT create markdown TODO lists
- Do NOT use external issue trackers
- Do NOT duplicate tracking systems
- Do NOT clutter repo root with planning documents

For more details, see README.md and QUICKSTART.md.
