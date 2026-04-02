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
| RF-024 | P2 | done | | Remove permanent trust-tier framing from the form catalog and RL defaults | RF-023 |
| RF-025 | P2 | done | | Narrow generic-form claims to the structure the code actually verifies | RF-024 |
| RF-026 | P2 | done | | Strip remaining unverified thematic claims from form docstrings and catalog summaries | RF-025 |
| RF-027 | P1 | done | | Harden Epigram so 3-line prose blocks no longer pass as perfect fits | RF-026 |
| RF-028 | P1 | done | | Stop Skeltonic from treating repeated identical end words as valid rhyme runs | RF-027 |
| RF-029 | P1 | done | | Fix malformed `PositionalPoem` catalog defaults and fail early on invalid positions | RF-028 |
| RF-030 | P1 | done | | Gate `LiteraryBallad` canonical passes on stanza count instead of blended score alone | RF-029 |
| RF-031 | P1 | done | | Gate generic `Sonnet` canonical passes on the syllable proxy instead of line count alone | RF-030 |
| RF-032 | P2 | done | | Make the test suite warning-clean under `pytest -W error` | RF-031 |
| RF-033 | P1 | done | | Remove the `pronouncing` dependency and read CMU data directly from `cmudict` | RF-032 |
| RF-034 | P1 | done | | Fix stale special-form kwargs so the catalog instantiates every exported form | RF-033 |
| RF-035 | P1 | done | | Add fixture-driven line-add/drop mutation regressions for core fixed forms | RF-034 |
| RF-036 | P1 | done | | Reject flattened single-block Pantoum layouts instead of chunking them back into quatrains | RF-035 |
| RF-037 | P2 | done | | Add fixture-driven stanza-flattening regressions for multi-stanza fixed forms | RF-036 |
| RF-038 | P2 | done | | Add exported-form surface smoke tests for instantiate/describe/verify stability | RF-037 |
| RF-039 | P2 | done | | Expand the fixed-form mutation harness to additional stable fixture families | RF-038 |
| RF-040 | P1 | done | | Reject flattened single-block Blues layouts instead of chunking them back into tercets | RF-039 |
| RF-041 | P1 | done | | Reject flattened single-block Terza Rima layouts instead of chunking them back into tercets | RF-040 |
| RF-042 | P2 | done | | Add shared mutation coverage for exact-count short-form families | RF-041 |
| RF-043 | P1 | done | | Stop Clerihew from treating bare sentence-start capitalization as a name | RF-042 |
| RF-044 | P1 | done | | Require whole-token prefix matches for explicit Anaphora phrases | RF-043 |
| RF-045 | P1 | done | | Remove the hidden 20-word gate from UniqueUtterance | RF-044 |
| RF-046 | P1 | done | | Reject unsupported PrimeVerse line counts instead of cycling the prime pattern | RF-045 |
| RF-047 | P1 | done | | Replace silent line-count clamps with early validation in bounded pattern forms | RF-046 |
| RF-048 | P1 | done | | Reject unsupported Rispetto variants instead of silently falling back to Tuscan | RF-047 |
| RF-049 | P1 | done | | Validate PalindromePoem levels and ModularVerse moduli explicitly | RF-048 |
| RF-050 | P1 | done | | Reject empty or non-alphabetic target sequences in acrostic-like forms | RF-049 |
| RF-051 | P1 | done | | Reject degenerate zero-valued hard-form configurations | RF-050 |
| RF-052 | P1 | done | | Reject empty explicit target strings in Anaphora and EchoEnd | RF-051 |
| RF-053 | P1 | done | | Reject degenerate shell-form bounds and empty generic-form constructors | RF-052 |
| RF-054 | P1 | done | | Harden public constraint constructors against empty patterns and invalid indices | RF-053 |
| RF-055 | P1 | done | | Harden shape and meter constraint constructors against degenerate configs | RF-054 |
| RF-056 | P1 | done | | Harden remaining lexical constraint constructors and 1-based position contracts | RF-055 |
| RF-057 | P1 | done | | Harden remaining character and sound lexical constraint constructors | RF-056 |
| RF-058 | P1 | done | | Eliminate high-score empty-input reward leaks in remaining public forms and constraints | RF-057 |
| RF-059 | P1 | done | | Penalize short-prefix leaks in non-uniform word-count patterns | RF-058 |
| RF-060 | P1 | done | | Penalize low-sample semantic reward leaks in uniqueness and monosyllabic checks | RF-059 |
| RF-061 | P1 | done | | Penalize underlength line-match reward leaks in manual novel forms | RF-060 |
| RF-062 | P1 | done | | Reject vacuous comparison passes and punctuation-only reward leaks | RF-061 |
