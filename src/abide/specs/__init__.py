"""
Form specifications with composable instructions.

FormSpec provides a mapping between programmatic constraints and
plain English instructions for LLM evaluation prompts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

    from abide.constraints import Constraint, VerificationResult


@dataclass
class InstructionItem:
    """
    A single atomic instruction component.

    Attributes:
        id: Unique identifier for this instruction (e.g., "line_count", "rhyme_scheme")
        instruction: Plain English instruction text
        constraint: The underlying constraint (for verification)
        weight: Relative importance (for weighted scoring)
        category: Category of constraint (structural, relational, prosodic)
    """

    id: str
    instruction: str
    constraint: Constraint
    weight: float = 1.0
    category: str = "general"


@dataclass
class FormSpec:
    """
    A complete form specification with decomposable instructions.

    FormSpec maps a poetic form to:
    1. Atomic constraints (each verifiable independently)
    2. Plain English instructions (for LLM prompts)
    3. Full composed instructions (all requirements together)

    This enables:
    - Testing individual constraints in isolation
    - Testing subsets of constraints
    - Generating natural prompts for LLM evaluation
    - Mapping rubric items back to specific instructions

    Example:
        >>> spec = FormSpec.shakespearean_sonnet()
        >>> print(spec.full_instruction())
        Write a Shakespearean sonnet with the following requirements:
        - Write exactly 14 lines.
        - Each line should have approximately 10 syllables.
        - Follow the rhyme scheme ABABCDCDEFEFGG...

        >>> # Test individual constraints
        >>> for item in spec.items:
        ...     result = item.constraint.verify(poem)
        ...     print(f"{item.id}: {result.score}")
    """

    name: str
    description: str
    items: list[InstructionItem] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add(
        self,
        id: str,
        constraint: Constraint,
        instruction: str | None = None,
        weight: float = 1.0,
        category: str = "general",
    ) -> FormSpec:
        """
        Add a constraint to the specification.

        Args:
            id: Unique identifier for this instruction
            constraint: The constraint to add
            instruction: Custom instruction text (default: use constraint.instruction())
            weight: Relative importance
            category: Category of constraint

        Returns:
            Self for method chaining
        """
        instruction_text = instruction if instruction else constraint.instruction()
        self.items.append(
            InstructionItem(
                id=id,
                instruction=instruction_text,
                constraint=constraint,
                weight=weight,
                category=category,
            )
        )
        return self

    def full_instruction(self, include_intro: bool = True) -> str:
        """
        Generate complete plain English instruction for LLM prompts.

        Args:
            include_intro: Whether to include introductory sentence

        Returns:
            Complete instruction text with all requirements
        """
        lines = []
        if include_intro:
            lines.append(f"Write a {self.name} with the following requirements:")
            lines.append("")

        for item in self.items:
            lines.append(f"- {item.instruction}")

        return "\n".join(lines)

    def instruction_for(self, *ids: str) -> str:
        """
        Generate instruction for specific constraint IDs only.

        Args:
            *ids: Constraint IDs to include

        Returns:
            Instruction text for selected constraints
        """
        lines = ["Write a poem with the following requirements:", ""]
        for item in self.items:
            if item.id in ids:
                lines.append(f"- {item.instruction}")
        return "\n".join(lines)

    def instructions_by_category(self, category: str) -> str:
        """
        Generate instructions for a specific category.

        Args:
            category: Category to filter by (structural, relational, prosodic)

        Returns:
            Instruction text for that category
        """
        lines = [f"Write a poem with these {category} requirements:", ""]
        for item in self.items:
            if item.category == category:
                lines.append(f"- {item.instruction}")
        return "\n".join(lines)

    def verify(self, poem: str) -> dict[str, VerificationResult]:
        """
        Verify poem against all constraints.

        Args:
            poem: Poem text to verify

        Returns:
            Dict mapping constraint ID to verification result
        """
        results = {}
        for item in self.items:
            results[item.id] = item.constraint.verify(poem)
        return results

    def verify_subset(self, poem: str, *ids: str) -> dict[str, VerificationResult]:
        """
        Verify poem against specific constraints only.

        Args:
            poem: Poem text to verify
            *ids: Constraint IDs to verify

        Returns:
            Dict mapping constraint ID to verification result
        """
        results = {}
        for item in self.items:
            if item.id in ids:
                results[item.id] = item.constraint.verify(poem)
        return results

    def weighted_score(self, poem: str) -> float:
        """
        Compute weighted aggregate score for poem.

        Args:
            poem: Poem text to verify

        Returns:
            Weighted average score (0.0 to 1.0)
        """
        results = self.verify(poem)
        total_weight = sum(item.weight for item in self.items)
        if total_weight == 0:
            return 0.0
        weighted_sum = sum(
            results[item.id].score * item.weight for item in self.items
        )
        return weighted_sum / total_weight

    def to_dict(self) -> dict[str, Any]:
        """Serialize specification to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "items": [
                {
                    "id": item.id,
                    "instruction": item.instruction,
                    "weight": item.weight,
                    "category": item.category,
                    "constraint_type": item.constraint.__class__.__name__,
                }
                for item in self.items
            ],
            "metadata": self.metadata,
        }

    def __iter__(self) -> Iterator[InstructionItem]:
        """Iterate over instruction items."""
        return iter(self.items)

    def __len__(self) -> int:
        """Number of constraints in specification."""
        return len(self.items)


# =============================================================================
# Pre-built form specifications
# =============================================================================


def haiku_spec(syllable_tolerance: int = 0) -> FormSpec:
    """Create a Haiku form specification."""
    from abide.constraints import LineCount, SyllablesPerLine

    spec = FormSpec(
        name="Haiku",
        description="Japanese 3-line poem with 5-7-5 syllable pattern",
    )
    spec.add(
        "line_count",
        LineCount(3),
        category="structural",
    )
    spec.add(
        "syllables",
        SyllablesPerLine([5, 7, 5], tolerance=syllable_tolerance),
        instruction="Follow the syllable pattern 5-7-5: first line has 5 syllables, second has 7, third has 5.",
        category="prosodic",
    )
    return spec


def tanka_spec(syllable_tolerance: int = 0) -> FormSpec:
    """Create a Tanka form specification."""
    from abide.constraints import LineCount, SyllablesPerLine

    spec = FormSpec(
        name="Tanka",
        description="Japanese 5-line poem with 5-7-5-7-7 syllable pattern",
    )
    spec.add(
        "line_count",
        LineCount(5),
        category="structural",
    )
    spec.add(
        "syllables",
        SyllablesPerLine([5, 7, 5, 7, 7], tolerance=syllable_tolerance),
        instruction="Follow the syllable pattern 5-7-5-7-7: lines have 5, 7, 5, 7, 7 syllables respectively.",
        category="prosodic",
    )
    return spec


def limerick_spec(rhyme_threshold: float = 0.6) -> FormSpec:
    """Create a Limerick form specification."""
    from abide.constraints import LineCount, RhymeScheme

    spec = FormSpec(
        name="Limerick",
        description="5-line humorous poem with AABBA rhyme scheme",
    )
    spec.add(
        "line_count",
        LineCount(5),
        category="structural",
    )
    spec.add(
        "rhyme_scheme",
        RhymeScheme("AABBA", threshold=rhyme_threshold),
        instruction="Follow the rhyme scheme AABBA: lines 1, 2, and 5 rhyme with each other; lines 3 and 4 rhyme with each other.",
        category="relational",
    )
    return spec


def shakespearean_sonnet_spec(
    syllable_tolerance: int = 1,
    rhyme_threshold: float = 0.6,
) -> FormSpec:
    """Create a Shakespearean Sonnet form specification."""
    from abide.constraints import LineCount, RhymeScheme, SyllablesPerLine

    spec = FormSpec(
        name="Shakespearean Sonnet",
        description="14-line poem with ABAB CDCD EFEF GG rhyme scheme in iambic pentameter",
    )
    spec.add(
        "line_count",
        LineCount(14),
        category="structural",
    )
    spec.add(
        "syllables",
        SyllablesPerLine([10] * 14, tolerance=syllable_tolerance),
        instruction="Write in iambic pentameter (approximately 10 syllables per line).",
        category="prosodic",
    )
    spec.add(
        "rhyme_scheme",
        RhymeScheme("ABABCDCDEFEFGG", threshold=rhyme_threshold),
        instruction="Follow the rhyme scheme ABAB CDCD EFEF GG: three quatrains with alternating rhyme, ending in a rhyming couplet.",
        category="relational",
    )
    return spec


def petrarchan_sonnet_spec(
    syllable_tolerance: int = 1,
    rhyme_threshold: float = 0.6,
    sestet_scheme: str = "CDECDE",
) -> FormSpec:
    """Create a Petrarchan Sonnet form specification."""
    from abide.constraints import LineCount, RhymeScheme, SyllablesPerLine

    full_scheme = "ABBAABBA" + sestet_scheme

    spec = FormSpec(
        name="Petrarchan Sonnet",
        description=f"14-line poem with ABBAABBA {sestet_scheme} rhyme scheme in iambic pentameter",
    )
    spec.add(
        "line_count",
        LineCount(14),
        category="structural",
    )
    spec.add(
        "syllables",
        SyllablesPerLine([10] * 14, tolerance=syllable_tolerance),
        instruction="Write in iambic pentameter (approximately 10 syllables per line).",
        category="prosodic",
    )
    spec.add(
        "rhyme_scheme",
        RhymeScheme(full_scheme, threshold=rhyme_threshold),
        instruction=f"Follow the rhyme scheme ABBAABBA {sestet_scheme}: an octave with enclosed rhyme, followed by a sestet.",
        category="relational",
    )
    return spec


def villanelle_spec(
    rhyme_threshold: float = 0.6,
    refrain_threshold: float = 0.9,
) -> FormSpec:
    """Create a Villanelle form specification."""
    from abide.constraints import LineCount, Refrain, RhymeScheme

    spec = FormSpec(
        name="Villanelle",
        description="19-line poem with two refrains and ABA rhyme scheme",
    )
    spec.add(
        "line_count",
        LineCount(19),
        category="structural",
    )
    spec.add(
        "rhyme_scheme",
        RhymeScheme("ABAAABABABABABABABAA", threshold=rhyme_threshold),
        instruction="Follow the ABA rhyme scheme throughout, where all A lines rhyme and all B lines rhyme.",
        category="relational",
    )
    spec.add(
        "refrain_a",
        Refrain(reference_line=0, repeat_at=[5, 11, 17], threshold=refrain_threshold),
        instruction="The first line (refrain A1) must repeat exactly at lines 6, 12, and 18.",
        category="relational",
    )
    spec.add(
        "refrain_b",
        Refrain(reference_line=2, repeat_at=[8, 14, 18], threshold=refrain_threshold),
        instruction="The third line (refrain A2) must repeat exactly at lines 9, 15, and 19.",
        category="relational",
    )
    return spec


def sestina_spec(word_match_threshold: float = 0.8) -> FormSpec:
    """Create a Sestina form specification."""
    from abide.constraints import EndWordPattern, LineCount, StanzaSizes

    spec = FormSpec(
        name="Sestina",
        description="39-line poem with 6 sestets and a 3-line envoi, using end-word rotation",
    )
    spec.add(
        "line_count",
        LineCount(39),
        category="structural",
    )
    spec.add(
        "stanza_sizes",
        StanzaSizes([6, 6, 6, 6, 6, 6, 3]),
        instruction="Write 6 stanzas of 6 lines each, followed by a 3-line envoi (concluding stanza).",
        category="structural",
    )
    spec.add(
        "end_words",
        EndWordPattern(num_words=6, num_stanzas=6, threshold=word_match_threshold),
        instruction="Choose 6 end words for the first stanza. These same 6 words must appear at the end of lines in every stanza, rotating in the pattern: 6-1-5-2-4-3 (the last end word becomes the first, etc.).",
        category="relational",
    )
    return spec


def triolet_spec(
    rhyme_threshold: float = 0.6,
    refrain_threshold: float = 0.9,
) -> FormSpec:
    """Create a Triolet form specification."""
    from abide.constraints import LineCount, Refrain, RhymeScheme

    spec = FormSpec(
        name="Triolet",
        description="8-line poem with ABaAabAB rhyme scheme and refrains",
    )
    spec.add(
        "line_count",
        LineCount(8),
        category="structural",
    )
    spec.add(
        "rhyme_scheme",
        RhymeScheme("ABAAABAB", threshold=rhyme_threshold),
        instruction="Follow the rhyme scheme ABaAabAB (A and B indicate the rhyme sounds).",
        category="relational",
    )
    spec.add(
        "refrain_a",
        Refrain(reference_line=0, repeat_at=[3, 6], threshold=refrain_threshold),
        instruction="Line 1 must repeat exactly at lines 4 and 7.",
        category="relational",
    )
    spec.add(
        "refrain_b",
        Refrain(reference_line=1, repeat_at=[7], threshold=refrain_threshold),
        instruction="Line 2 must repeat exactly at line 8.",
        category="relational",
    )
    return spec


def clerihew_spec(rhyme_threshold: float = 0.5) -> FormSpec:
    """Create a Clerihew form specification."""
    from abide.constraints import LineCount, RhymeScheme

    spec = FormSpec(
        name="Clerihew",
        description="4-line biographical poem with AABB rhyme scheme",
    )
    spec.add(
        "line_count",
        LineCount(4),
        category="structural",
    )
    spec.add(
        "rhyme_scheme",
        RhymeScheme("AABB", threshold=rhyme_threshold),
        instruction="Follow the rhyme scheme AABB: lines 1 and 2 rhyme, lines 3 and 4 rhyme.",
        category="relational",
    )
    spec.add(
        "name_in_first_line",
        LineCount(4),  # Placeholder - actual check would need custom constraint
        instruction="The first line must contain a person's name (the subject of the poem).",
        category="content",
        weight=0.5,  # Lower weight since we can't verify this automatically
    )
    return spec


# Export all specs
__all__ = [
    "FormSpec",
    "InstructionItem",
    "clerihew_spec",
    "haiku_spec",
    "limerick_spec",
    "petrarchan_sonnet_spec",
    "sestina_spec",
    "shakespearean_sonnet_spec",
    "tanka_spec",
    "triolet_spec",
    "villanelle_spec",
]
