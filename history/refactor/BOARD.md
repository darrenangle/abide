# Refactor Board

Updated: 2026-04-01

| ID | Priority | Status | Lock | Summary | Depends On |
| --- | --- | --- | --- | --- | --- |
| RF-001 | P1 | done | | Add gated composite pass/fail semantics and migrate high-risk forms onto it | |
| RF-002 | P1 | done | | Add adversarial regression tests for false-positive passes | RF-001 |
| RF-003 | P1 | done | | Align sonnet-family descriptions and meter claims with actual verification | RF-001 |
| RF-004 | P1 | done | | Align blank-verse and related metrical form claims with actual verification | RF-001 |
| RF-005 | P1 | done | | Reject dangling lines in `Ghazal` verification | |
| RF-006 | P1 | done | | Carry exact `form_name` metadata through the TRL reward path | |
| RF-007 | P2 | done | | Declare and isolate TRL training dependencies | RF-006 |
| RF-008 | P2 | done | | Audit remaining form descriptions for verifier/claim mismatch after core fixes land | RF-003, RF-004 |
| RF-009 | P1 | done | | Refactor remaining manual fixed-form verifiers onto shared constraints | RF-001, RF-002 |
| RF-010 | P1 | done | | Add a training-safe form catalog and generated adversarial harness | RF-009 |
| RF-011 | P1 | done | | Harden `Ghazal` canonical pass/fail and reevaluate training-safe eligibility | RF-010 |
| RF-012 | P1 | done | | Add meter/scansion tests and align metrical claims with verified behavior | RF-011 |
| RF-013 | P1 | done | | Add inference regression coverage and narrow inference guarantees to what is proven | RF-012 |
| RF-014 | P1 | done | | Rewrite README and public docs around verified APIs and temporary rollout guardrails | RF-012, RF-013 |
| RF-015 | P1 | done | | Harden Japanese short forms and shape-poetry families against lenient false positives | RF-014 |
| RF-016 | P1 | done | | Harden Fibonacci and free-form structural verifiers and narrow their public claims | RF-015 |
| RF-017 | P1 | done | | Add constrained-form coverage and fix catalog instantiation for semantic constraint forms | RF-016 |
| RF-018 | P1 | done | | Harden mathematical forms against prefix-pass and token-count false positives | RF-017 |
| RF-019 | P1 | done | | Harden novel semantic-token forms and short-input handling | RF-018 |
| RF-020 | P1 | done | | Harden blended-score pass/fail in Naani and Skeltonic | RF-019 |
| RF-021 | P2 | done | | Align pre-commit Ruff hook with project Ruff version | RF-020 |
| RF-022 | P1 | done | | Harden exact-count constrained forms and FreeVerse word-bound semantics | RF-021 |
| RF-023 | P1 | done | | Harden Haiku and Tanka exact-structure pass semantics in lenient mode | RF-022 |
