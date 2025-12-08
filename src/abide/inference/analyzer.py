"""
Poem analyzer for form inference.

Analyzes a poem's structure and derives constraints that it satisfies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from abide.constraints import (
    Constraint,
    LineCount,
    Refrain,
    RhymeScheme,
    StanzaCount,
    StanzaSizes,
    SyllablesPerLine,
)
from abide.primitives import (
    RhymeType,
    classify_rhyme,
    count_line_syllables,
    extract_end_words,
    parse_structure,
)
from abide.specs import FormSpec

if TYPE_CHECKING:
    from abide.primitives import PoemStructure


@dataclass
class InferredConstraint:
    """A constraint inferred from poem analysis."""

    id: str
    constraint: Constraint
    confidence: float  # How certain we are this is intentional (0-1)
    description: str
    category: str = "general"
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class FormAnalysis:
    """Complete analysis of a poem's formal properties."""

    poem: str
    structure: PoemStructure
    constraints: list[InferredConstraint]
    rhyme_scheme: str | None
    syllable_pattern: list[int]
    refrains: list[tuple[int, list[int]]]  # (reference_line, repeat_positions)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_form_spec(self, name: str = "Inferred Form") -> FormSpec:
        """
        Convert analysis to a FormSpec that the poem passes 100%.

        Args:
            name: Name for the form specification

        Returns:
            FormSpec with all inferred constraints
        """
        spec = FormSpec(
            name=name,
            description=f"Form inferred from poem with {self.structure.line_count} lines",
        )

        for ic in self.constraints:
            spec.add(
                id=ic.id,
                constraint=ic.constraint,
                category=ic.category,
                weight=ic.confidence,
            )

        return spec

    def verify_score(self) -> float:
        """Verify that the poem passes all inferred constraints."""
        spec = self.to_form_spec()
        return spec.weighted_score(self.poem)


def analyze_poem(poem: str) -> FormAnalysis:
    """
    Analyze a poem and extract its formal properties.

    Args:
        poem: The poem text to analyze

    Returns:
        FormAnalysis with all detected structural patterns
    """
    structure = parse_structure(poem)
    constraints: list[InferredConstraint] = []

    # 1. Structural analysis
    structural = _analyze_structure(structure)
    constraints.extend(structural)

    # 2. Syllable analysis
    syllables, syllable_constraints = _analyze_syllables(structure)
    constraints.extend(syllable_constraints)

    # 3. Rhyme analysis - try strict first, then lenient if needed
    rhyme_scheme, rhyme_constraints = _analyze_rhymes(structure, accept_near_rhymes=False)

    # If strict rhyme analysis produces poor results, try with near rhymes
    if rhyme_constraints:
        # Test the score quickly
        test_result = rhyme_constraints[0].constraint.verify(poem)
        if test_result.score < 0.95:
            # Try with near rhymes for historical poetry
            rhyme_scheme_lenient, rhyme_constraints_lenient = _analyze_rhymes(
                structure, accept_near_rhymes=True
            )
            if rhyme_constraints_lenient:
                test_result_lenient = rhyme_constraints_lenient[0].constraint.verify(poem)
                if test_result_lenient.score > test_result.score:
                    rhyme_scheme = rhyme_scheme_lenient
                    rhyme_constraints = rhyme_constraints_lenient

    constraints.extend(rhyme_constraints)

    # 4. Refrain analysis
    refrains, refrain_constraints = _analyze_refrains(structure)
    constraints.extend(refrain_constraints)

    return FormAnalysis(
        poem=poem,
        structure=structure,
        constraints=constraints,
        rhyme_scheme=rhyme_scheme,
        syllable_pattern=syllables,
        refrains=refrains,
    )


def infer_form(poem: str, name: str = "Custom Form") -> FormSpec:
    """
    Infer a FormSpec from a poem that the poem passes 100%.

    This is the main entry point for form inference.

    Args:
        poem: The poem to analyze
        name: Name for the inferred form

    Returns:
        FormSpec that the poem passes with score 1.0

    Example:
        >>> spec = infer_form(my_poem, name="My Custom Form")
        >>> result = spec.verify(my_poem)
        >>> assert result["line_count"].score == 1.0
    """
    analysis = analyze_poem(poem)
    return analysis.to_form_spec(name)


# =============================================================================
# Analysis Functions
# =============================================================================


def _analyze_structure(structure: PoemStructure) -> list[InferredConstraint]:
    """Analyze structural properties."""
    constraints = []

    # Line count (always infer, high confidence)
    constraints.append(
        InferredConstraint(
            id="line_count",
            constraint=LineCount(structure.line_count),
            confidence=1.0,
            description=f"Exactly {structure.line_count} lines",
            category="structural",
        )
    )

    # Stanza count
    if structure.stanza_count > 1:
        constraints.append(
            InferredConstraint(
                id="stanza_count",
                constraint=StanzaCount(structure.stanza_count),
                confidence=0.9,
                description=f"Exactly {structure.stanza_count} stanzas",
                category="structural",
            )
        )

        # Stanza sizes
        constraints.append(
            InferredConstraint(
                id="stanza_sizes",
                constraint=StanzaSizes(list(structure.stanza_sizes)),
                confidence=0.8,
                description=f"Stanza sizes: {list(structure.stanza_sizes)}",
                category="structural",
                details={"sizes": list(structure.stanza_sizes)},
            )
        )

    return constraints


def _analyze_syllables(
    structure: PoemStructure,
) -> tuple[list[int], list[InferredConstraint]]:
    """Analyze syllable patterns."""
    constraints = []
    syllables = [count_line_syllables(line) for line in structure.lines]

    # Check if uniform syllable count (with tolerance)
    if syllables:
        mean_syl = sum(syllables) / len(syllables)
        max_deviation = max(abs(s - mean_syl) for s in syllables)

        if max_deviation <= 1:
            # Nearly uniform - use tolerance to ensure 100% pass
            target = round(mean_syl)
            tolerance = int(max_deviation) + 1  # +1 to guarantee pass
            constraints.append(
                InferredConstraint(
                    id="syllables_uniform",
                    constraint=SyllablesPerLine([target] * len(syllables), tolerance=tolerance),
                    confidence=0.9 if max_deviation == 0 else 0.8,
                    description=f"~{target} syllables per line (±{tolerance})",
                    category="prosodic",
                    details={"target": target, "tolerance": tolerance, "actual": syllables},
                )
            )
        else:
            # Check if there's a repeating pattern
            pattern = _detect_repeating_pattern(syllables)
            if pattern and len(pattern) < len(syllables):
                constraints.append(
                    InferredConstraint(
                        id="syllables_pattern",
                        constraint=SyllablesPerLine(syllables, tolerance=1),
                        confidence=0.7,
                        description=f"Syllable pattern: {'-'.join(map(str, pattern))} (repeating)",
                        category="prosodic",
                        details={"pattern": pattern, "full": syllables},
                    )
                )
            else:
                # Record exact syllables with tolerance to guarantee pass
                constraints.append(
                    InferredConstraint(
                        id="syllables_exact",
                        constraint=SyllablesPerLine(syllables, tolerance=2),
                        confidence=0.5,
                        description=f"Syllable counts: {syllables}",
                        category="prosodic",
                        details={"syllables": syllables},
                    )
                )

    return syllables, constraints


def _detect_repeating_pattern(values: list[int]) -> list[int] | None:
    """Detect if a list contains a repeating pattern."""
    if len(values) < 2:
        return None

    for pattern_len in range(1, len(values) // 2 + 1):
        pattern = values[:pattern_len]
        matches = True
        for i in range(len(values)):
            if values[i] != pattern[i % pattern_len]:
                matches = False
                break
        if matches:
            return pattern

    return None


def _analyze_rhymes(
    structure: PoemStructure,
    accept_near_rhymes: bool = False,  # Default to strict for better grouping
) -> tuple[str | None, list[InferredConstraint]]:
    """
    Analyze rhyme scheme.

    Args:
        structure: Parsed poem structure
        accept_near_rhymes: If True, accept family/near rhymes (useful for
            historical poetry where pronunciation has changed)
    """
    constraints: list[InferredConstraint] = []

    if structure.line_count < 2:
        return None, constraints

    end_words = extract_end_words(structure)

    # Rhyme types we accept for grouping
    # Default to PERFECT only for more accurate grouping
    accepted_types = {RhymeType.PERFECT}
    if accept_near_rhymes:
        accepted_types.add(RhymeType.FAMILY)
        accepted_types.add(RhymeType.NEAR)

    # Group lines by rhyme - more conservative approach
    # Only group words that actually rhyme with each other
    groups: list[set[int]] = []
    for i in range(len(end_words)):
        found_group = None
        best_type = RhymeType.NONE
        best_score = 0.0

        for j, group in enumerate(groups):
            # Check if this word rhymes with any word in the group
            for member_idx in group:
                rhyme_type, confidence = classify_rhyme(end_words[i], end_words[member_idx])
                if rhyme_type in accepted_types and confidence > best_score:
                    # Prefer perfect rhymes over near rhymes
                    found_group = j
                    best_type = rhyme_type
                    best_score = confidence
                    if rhyme_type == RhymeType.PERFECT:
                        break
            if best_type == RhymeType.PERFECT:
                break

        # Only add to group if we found a real rhyme (not just "none" matching "none")
        if found_group is not None and best_score > 0:
            groups[found_group].add(i)
        else:
            groups.append({i})

    # Assign letters to groups
    group_for_line: dict[int, int] = {}
    for group_idx, group in enumerate(groups):
        for line_idx in group:
            group_for_line[line_idx] = group_idx

    # Build scheme string
    scheme_chars = []
    letter_for_group: dict[int, str] = {}
    next_letter = ord("A")

    for i in range(len(end_words)):
        group_idx = group_for_line[i]
        if group_idx not in letter_for_group:
            letter_for_group[group_idx] = chr(next_letter)
            next_letter += 1
        scheme_chars.append(letter_for_group[group_idx])

    scheme = "".join(scheme_chars)

    # Determine confidence based on how many rhymes we found
    num_rhyme_groups = sum(1 for g in groups if len(g) > 1)
    if num_rhyme_groups == 0:
        # No rhymes detected - might be free verse
        return scheme, constraints

    # Calculate rhyme density (portion of lines that rhyme with something)
    rhyming_lines = sum(len(g) for g in groups if len(g) > 1)
    rhyme_density = rhyming_lines / len(end_words)

    # Find minimum rhyme score among all detected rhyme pairs
    # This ensures we set threshold below the weakest rhyme to get 100% pass
    min_rhyme_score = 1.0
    has_imperfect_rhymes = False
    non_rhyming_pairs = 0
    total_pairs = 0

    for group in groups:
        if len(group) > 1:
            group_list = list(group)
            for i in range(len(group_list)):
                for j in range(i + 1, len(group_list)):
                    w1, w2 = end_words[group_list[i]], end_words[group_list[j]]
                    # Identical words count as rhyming (score 1.0 for our purposes)
                    if w1.lower() == w2.lower():
                        continue  # Skip identical words - they always "rhyme"
                    from abide.primitives import rhyme_score as rs

                    score = rs(w1, w2)
                    total_pairs += 1
                    if score > 0:  # Only consider actual rhymes
                        min_rhyme_score = min(min_rhyme_score, score)
                    else:
                        non_rhyming_pairs += 1
                    rtype, _ = classify_rhyme(w1, w2)
                    if rtype in (RhymeType.FAMILY, RhymeType.NEAR):
                        has_imperfect_rhymes = True

    # If any pairs in "rhyme groups" don't actually rhyme,
    # the grouping is likely incorrect - skip rhyme constraint
    if non_rhyming_pairs > 0:
        # False positives detected - don't add rhyme constraint
        return scheme, constraints

    # Set threshold just below the minimum rhyme score to guarantee 100% pass
    # Ensure all detected "rhyme" pairs will pass
    if min_rhyme_score <= 0.2:
        # Rhymes are too weak - skip constraint
        return scheme, constraints

    threshold = max(0.2, min_rhyme_score - 0.05)

    constraints.append(
        InferredConstraint(
            id="rhyme_scheme",
            constraint=RhymeScheme(
                scheme,
                threshold=threshold,
                allow_identical=True,
                binary_scoring=True,  # Use binary for 100% pass on threshold
            ),
            confidence=max(0.5, rhyme_density),
            description=f"Rhyme scheme: {scheme}"
            + (" (with near rhymes)" if has_imperfect_rhymes else ""),
            category="relational",
            details={
                "scheme": scheme,
                "rhyme_groups": [[i for i in g] for g in groups],
                "rhyme_density": rhyme_density,
                "has_imperfect_rhymes": has_imperfect_rhymes,
                "threshold": threshold,
            },
        )
    )

    return scheme, constraints


def _analyze_refrains(
    structure: PoemStructure,
) -> tuple[list[tuple[int, list[int]]], list[InferredConstraint]]:
    """Analyze refrain patterns (repeated lines)."""
    constraints: list[InferredConstraint] = []
    refrains: list[tuple[int, list[int]]] = []

    if structure.line_count < 4:
        return refrains, constraints

    # Find repeated lines
    line_occurrences: dict[str, list[int]] = {}
    for i, line in enumerate(structure.lines):
        normalized = _normalize_line(line)
        if normalized not in line_occurrences:
            line_occurrences[normalized] = []
        line_occurrences[normalized].append(i)

    # Create refrain constraints for repeated lines
    for _normalized, positions in line_occurrences.items():
        if len(positions) > 1:
            reference = positions[0]
            repeats = positions[1:]
            refrains.append((reference, repeats))

            constraints.append(
                InferredConstraint(
                    id=f"refrain_line_{reference + 1}",
                    constraint=Refrain(
                        reference_line=reference,
                        repeat_at=repeats,
                        threshold=0.95,
                    ),
                    confidence=0.95,
                    description=f"Line {reference + 1} repeats at lines {[r + 1 for r in repeats]}",
                    category="relational",
                    details={
                        "reference": reference,
                        "repeats": repeats,
                        "text": structure.lines[reference][:50],
                    },
                )
            )

    return refrains, constraints


def _normalize_line(line: str) -> str:
    """Normalize a line for comparison."""
    import re

    line = line.lower().strip()
    line = re.sub(r"[^\w\s]", "", line)
    return " ".join(line.split())
