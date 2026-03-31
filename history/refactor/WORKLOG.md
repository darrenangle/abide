# Refactor Worklog

## 2026-03-31

- Initialized markdown refactor tracker under `history/refactor/`.
- Broke the 2026-03-31 audit into concrete repair tickets `RF-001` through `RF-008`.
- Claimed `RF-001` for the first implementation pass.
- Removed the legacy issue tracker files and scrubbed repo instructions so later agents only see the markdown workflow.
- Implemented `WeightedSum.required_indices` to separate weighted reward from canonical `passed`.
- Migrated `Villanelle` to require all defining child constraints for `passed=True`.
- Added regression tests covering required child gating and a non-refrain villanelle false positive.
- Fixed `Ghazal.verify()` so odd dangling lines fail, lower score sharply, and appear in the rubric.
- Added a ghazal regression test for the dangling-line case.
- Verified the repo with `uv run pytest`, `uv run ruff check src/abide tests`, and `uv run mypy src/abide`.
- Refactored `scripts/train_grpo_trl.py` so helper imports no longer require TRL, Transformers, PEFT, or Torch at module import time.
- Preserved exact `form_name` metadata in the TRL dataset and routed reward scoring through that metadata only.
- Added integration coverage for TRL helper importability, dataset metadata preservation, and exact form routing.
- Tightened Shakespearean, Petrarchan, and Spenserian sonnet pass/fail to require their defining child constraints.
- Added a regression test for the repeated-line Shakespearean sonnet false positive from the audit.
