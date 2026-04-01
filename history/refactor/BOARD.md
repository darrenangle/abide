# Refactor Board

Updated: 2026-03-31

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
| RF-013 | P1 | in_progress | `RF-013--codex-main.md` | Add inference regression coverage and narrow inference guarantees to what is proven | RF-012 |
| RF-014 | P1 | pending | | Rewrite README and public docs around support tiers and verified APIs | RF-012, RF-013 |
