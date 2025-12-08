"""
Meter and scansion analysis for poetry.

Provides meter type detection, foot counting, and scansion scoring
using CMU pronouncing dictionary stress patterns.

Metrical Feet:
- Iamb: unstressed-stressed (u/)
- Trochee: stressed-unstressed (/u)
- Anapest: unstressed-unstressed-stressed (uu/)
- Dactyl: stressed-unstressed-unstressed (/uu)
- Spondee: stressed-stressed (//)
- Pyrrhic: unstressed-unstressed (uu)

Common Meters:
- Iambic Pentameter: 5 iambs per line (u/ u/ u/ u/ u/)
- Trochaic Tetrameter: 4 trochees per line (/u /u /u /u)
- Anapestic Trimeter: 3 anapests per line (uu/ uu/ uu/)
- Dactylic Hexameter: 6 dactyls per line (/uu /uu /uu /uu /uu /uu)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import TYPE_CHECKING

from abide.primitives.phonetics import get_line_stress_pattern, get_stress_pattern

if TYPE_CHECKING:
    pass


class MeterType(Enum):
    """Types of metrical feet."""

    IAMB = "iamb"  # u/ (unstressed-stressed)
    TROCHEE = "trochee"  # /u (stressed-unstressed)
    ANAPEST = "anapest"  # uu/ (unstressed-unstressed-stressed)
    DACTYL = "dactyl"  # /uu (stressed-unstressed-unstressed)
    SPONDEE = "spondee"  # // (stressed-stressed)
    PYRRHIC = "pyrrhic"  # uu (unstressed-unstressed)
    AMPHIBRACH = "amphibrach"  # u/u (unstressed-stressed-unstressed)
    FREE = "free"  # No consistent meter


class FootLength(Enum):
    """Number of feet per line (meter length)."""

    MONOMETER = 1
    DIMETER = 2
    TRIMETER = 3
    TETRAMETER = 4
    PENTAMETER = 5
    HEXAMETER = 6
    HEPTAMETER = 7
    OCTAMETER = 8


# Canonical stress patterns for each foot type
# 0 = unstressed, 1 = primary stress, 2 = secondary stress
# For matching, we treat 1 and 2 as stressed (/)
FOOT_PATTERNS: dict[MeterType, tuple[str, ...]] = {
    MeterType.IAMB: ("01", "02"),  # u/
    MeterType.TROCHEE: ("10", "20"),  # /u
    MeterType.ANAPEST: ("001", "002"),  # uu/
    MeterType.DACTYL: ("100", "200"),  # /uu
    MeterType.SPONDEE: ("11", "12", "21", "22"),  # //
    MeterType.PYRRHIC: ("00",),  # uu
    MeterType.AMPHIBRACH: ("010", "020"),  # u/u
}

# Foot sizes (number of syllables per foot)
FOOT_SIZES: dict[MeterType, int] = {
    MeterType.IAMB: 2,
    MeterType.TROCHEE: 2,
    MeterType.ANAPEST: 3,
    MeterType.DACTYL: 3,
    MeterType.SPONDEE: 2,
    MeterType.PYRRHIC: 2,
    MeterType.AMPHIBRACH: 3,
}


@dataclass
class FootMatch:
    """Result of matching a foot pattern."""

    meter_type: MeterType
    position: int  # Start position in stress pattern
    length: int  # Number of syllables matched
    exact: bool  # True if exact match, False if close match


@dataclass
class ScansionResult:
    """Result of scanning a line for meter."""

    stress_pattern: str  # Raw stress pattern (0s and 1s/2s)
    binary_pattern: str  # Normalized to 0s and 1s (/ and u)
    feet: list[FootMatch]  # Detected feet
    dominant_meter: MeterType | None  # Most common foot type
    foot_count: int  # Number of feet detected
    regularity: float  # How regular the meter is (0-1)
    syllable_count: int


def normalize_stress(pattern: str) -> str:
    """
    Normalize stress pattern to binary (0=unstressed, 1=stressed).

    CMU uses 0=no stress, 1=primary, 2=secondary.
    We treat both 1 and 2 as stressed.
    """
    return "".join("1" if c in "12" else "0" for c in pattern)


def stress_to_symbols(pattern: str) -> str:
    """
    Convert stress pattern to traditional scansion symbols.

    Args:
        pattern: Binary stress pattern (0s and 1s)

    Returns:
        String with u (unstressed) and / (stressed)

    Example:
        >>> stress_to_symbols("0101010101")
        'u/u/u/u/u/'
    """
    return "".join("/" if c == "1" else "u" for c in pattern)


@lru_cache(maxsize=1000)
def scan_line(line: str) -> ScansionResult:
    """
    Scan a line of poetry for its metrical pattern.

    Args:
        line: Line of text to scan

    Returns:
        ScansionResult with stress pattern, detected feet, and analysis

    Example:
        >>> result = scan_line("Shall I compare thee to a summer's day")
        >>> result.dominant_meter
        MeterType.IAMB
        >>> result.foot_count
        5
    """
    stress_pattern = get_line_stress_pattern(line)
    if not stress_pattern:
        return ScansionResult(
            stress_pattern="",
            binary_pattern="",
            feet=[],
            dominant_meter=None,
            foot_count=0,
            regularity=0.0,
            syllable_count=0,
        )

    binary = normalize_stress(stress_pattern)
    feet = _detect_feet(binary)

    # Count foot types
    type_counts: dict[MeterType, int] = {}
    for foot in feet:
        type_counts[foot.meter_type] = type_counts.get(foot.meter_type, 0) + 1

    # Find dominant meter
    dominant = None
    max_count = 0
    for meter_type, count in type_counts.items():
        if count > max_count:
            max_count = count
            dominant = meter_type

    # Calculate regularity (what portion of feet match dominant type)
    regularity = max_count / len(feet) if feet else 0.0

    return ScansionResult(
        stress_pattern=stress_pattern,
        binary_pattern=binary,
        feet=feet,
        dominant_meter=dominant,
        foot_count=len(feet),
        regularity=regularity,
        syllable_count=len(binary),
    )


def _detect_feet(binary_pattern: str) -> list[FootMatch]:
    """
    Detect metrical feet in a binary stress pattern.

    Uses greedy matching, preferring longer feet (anapest/dactyl)
    when they match, then falling back to shorter feet (iamb/trochee).
    """
    feet: list[FootMatch] = []
    pos = 0

    while pos < len(binary_pattern):
        remaining = binary_pattern[pos:]
        matched = False

        # Try 3-syllable feet first (anapest, dactyl, amphibrach)
        if len(remaining) >= 3:
            for meter_type in [MeterType.ANAPEST, MeterType.DACTYL, MeterType.AMPHIBRACH]:
                if _matches_foot(remaining[:3], meter_type):
                    feet.append(FootMatch(
                        meter_type=meter_type,
                        position=pos,
                        length=3,
                        exact=True,
                    ))
                    pos += 3
                    matched = True
                    break

        # Try 2-syllable feet (iamb, trochee, spondee, pyrrhic)
        if not matched and len(remaining) >= 2:
            for meter_type in [MeterType.IAMB, MeterType.TROCHEE, MeterType.SPONDEE, MeterType.PYRRHIC]:
                if _matches_foot(remaining[:2], meter_type):
                    feet.append(FootMatch(
                        meter_type=meter_type,
                        position=pos,
                        length=2,
                        exact=True,
                    ))
                    pos += 2
                    matched = True
                    break

        # If no exact match, make best guess for 2 syllables
        if not matched and len(remaining) >= 2:
            chunk = remaining[:2]
            # Guess based on pattern
            if chunk == "01":
                meter_type = MeterType.IAMB
            elif chunk == "10":
                meter_type = MeterType.TROCHEE
            elif chunk == "11":
                meter_type = MeterType.SPONDEE
            else:
                meter_type = MeterType.PYRRHIC

            feet.append(FootMatch(
                meter_type=meter_type,
                position=pos,
                length=2,
                exact=False,
            ))
            pos += 2
            matched = True

        # Handle single trailing syllable
        if not matched:
            # Single syllable at end - could be catalexis (incomplete foot)
            pos += 1

    return feet


def _matches_foot(pattern: str, meter_type: MeterType) -> bool:
    """Check if a pattern matches a foot type."""
    expected_patterns = FOOT_PATTERNS.get(meter_type, ())
    return pattern in expected_patterns


def detect_meter(line: str) -> tuple[MeterType | None, float]:
    """
    Detect the dominant meter of a line.

    Args:
        line: Line of text

    Returns:
        Tuple of (MeterType or None, confidence score 0-1)

    Example:
        >>> meter, conf = detect_meter("Shall I compare thee to a summer's day")
        >>> meter
        MeterType.IAMB
        >>> conf > 0.7
        True
    """
    result = scan_line(line)
    return result.dominant_meter, result.regularity


def meter_score(
    line: str,
    expected_meter: MeterType,
    expected_feet: int | None = None,
    tolerance: int = 0,
    allow_substitutions: bool = True,
) -> float:
    """
    Score how well a line matches expected meter.

    Real poetry uses metrical substitutions (e.g., a trochee or spondee
    substituting for an iamb). This function scores based on:
    1. Syllable count matching expected foot size * feet
    2. Overall stress pattern alignment with expected pattern
    3. Allowance for common substitutions

    Args:
        line: Line of text
        expected_meter: The meter type to check for
        expected_feet: Expected number of feet (e.g., 5 for pentameter)
        tolerance: Allowed deviation in foot count
        allow_substitutions: If True, allow common metrical substitutions

    Returns:
        Score from 0.0 to 1.0

    Example:
        >>> # Check for iambic pentameter
        >>> score = meter_score("Shall I compare thee to a summer's day", MeterType.IAMB, 5)
        >>> score > 0.7
        True
    """
    result = scan_line(line)

    if not result.binary_pattern:
        return 0.0

    foot_size = FOOT_SIZES.get(expected_meter, 2)
    expected_syllables = foot_size * expected_feet if expected_feet else None

    # 1. Score syllable count
    if expected_syllables is not None:
        syl_diff = abs(result.syllable_count - expected_syllables)
        if syl_diff <= tolerance:
            syllable_score = 1.0
        else:
            syllable_score = max(0.0, 1.0 - (syl_diff - tolerance) / expected_syllables)
    else:
        syllable_score = 1.0

    # 2. Score stress pattern alignment
    # Generate expected pattern and compare
    pattern_score = _score_pattern_alignment(
        result.binary_pattern,
        expected_meter,
        expected_feet,
        allow_substitutions,
    )

    # 3. Score foot count (more lenient than pattern)
    if expected_feet is not None:
        foot_diff = abs(result.foot_count - expected_feet)
        if foot_diff <= tolerance + 1:  # +1 for natural variation
            foot_score = 1.0
        else:
            foot_score = max(0.0, 1.0 - (foot_diff - tolerance - 1) / expected_feet)
    else:
        foot_score = 1.0

    # Combined score (pattern alignment most important)
    return pattern_score * 0.5 + syllable_score * 0.3 + foot_score * 0.2


def _score_pattern_alignment(
    actual: str,
    expected_meter: MeterType,
    expected_feet: int | None,
    allow_substitutions: bool,
) -> float:
    """
    Score how well an actual stress pattern aligns with expected meter.

    Uses a sliding window approach to find best alignment.
    """
    if not actual:
        return 0.0

    foot_size = FOOT_SIZES.get(expected_meter, 2)

    # Get canonical pattern for meter type
    canonical = _get_canonical_pattern(expected_meter)
    if not canonical:
        return 0.5  # Unknown meter

    # Generate expected pattern (repeating canonical for expected feet)
    if expected_feet:
        expected = canonical * expected_feet
    else:
        expected = canonical * (len(actual) // foot_size + 1)

    # Trim to match length
    expected = expected[: len(actual)]

    if len(expected) != len(actual):
        # Pad shorter one
        max_len = max(len(expected), len(actual))
        expected = expected.ljust(max_len, expected[-1] if expected else "0")
        actual = actual.ljust(max_len, "0")

    # Count matching positions
    matches = sum(1 for a, e in zip(actual, expected) if a == e)
    base_score = matches / len(actual) if actual else 0.0

    # Bonus for substitutions that are metrically acceptable
    if allow_substitutions:
        # Common substitutions don't reduce score as much
        substitution_bonus = _count_valid_substitutions(
            actual, expected, expected_meter
        )
        base_score = min(1.0, base_score + substitution_bonus * 0.1)

    return base_score


def _get_canonical_pattern(meter_type: MeterType) -> str:
    """Get the canonical binary stress pattern for a meter type."""
    patterns = {
        MeterType.IAMB: "01",  # unstressed-stressed
        MeterType.TROCHEE: "10",  # stressed-unstressed
        MeterType.ANAPEST: "001",  # unstressed-unstressed-stressed
        MeterType.DACTYL: "100",  # stressed-unstressed-unstressed
        MeterType.SPONDEE: "11",  # stressed-stressed
        MeterType.PYRRHIC: "00",  # unstressed-unstressed
        MeterType.AMPHIBRACH: "010",  # unstressed-stressed-unstressed
    }
    return patterns.get(meter_type, "")


def _count_valid_substitutions(
    actual: str, expected: str, expected_meter: MeterType
) -> int:
    """
    Count metrical substitutions that are acceptable.

    In poetry, certain substitutions are common and acceptable:
    - Trochaic substitution in iambic (especially line-initial)
    - Spondaic substitution for emphasis
    - Pyrrhic substitution for flow
    """
    if len(actual) < 2 or len(expected) < 2:
        return 0

    foot_size = FOOT_SIZES.get(expected_meter, 2)
    valid_subs = 0

    # Check each foot position
    for i in range(0, min(len(actual), len(expected)) - foot_size + 1, foot_size):
        actual_foot = actual[i : i + foot_size]
        expected_foot = expected[i : i + foot_size]

        if actual_foot == expected_foot:
            continue

        # Line-initial trochaic substitution for iambic
        if expected_meter == MeterType.IAMB and i == 0 and actual_foot == "10":
            valid_subs += 1

        # Spondaic substitution (emphasis)
        if actual_foot == "11":
            valid_subs += 1

        # Pyrrhic substitution (de-emphasis)
        if actual_foot == "00":
            valid_subs += 1

    return valid_subs


def get_expected_syllables(meter: MeterType, feet: int) -> int:
    """
    Calculate expected syllables for a given meter and foot count.

    Args:
        meter: The meter type
        feet: Number of feet

    Returns:
        Expected syllable count

    Example:
        >>> get_expected_syllables(MeterType.IAMB, 5)
        10
        >>> get_expected_syllables(MeterType.ANAPEST, 3)
        9
    """
    foot_size = FOOT_SIZES.get(meter, 2)
    return foot_size * feet


# Common meter presets for convenience
IAMBIC_PENTAMETER = (MeterType.IAMB, 5)
IAMBIC_TETRAMETER = (MeterType.IAMB, 4)
IAMBIC_TRIMETER = (MeterType.IAMB, 3)
TROCHAIC_TETRAMETER = (MeterType.TROCHEE, 4)
TROCHAIC_OCTAMETER = (MeterType.TROCHEE, 8)
ANAPESTIC_TETRAMETER = (MeterType.ANAPEST, 4)
DACTYLIC_HEXAMETER = (MeterType.DACTYL, 6)

# Common meter names
COMMON_METER = (MeterType.IAMB, 4, 3, 4, 3)  # 8-6-8-6 syllables, for ballads
LONG_METER = (MeterType.IAMB, 4, 4, 4, 4)  # 8-8-8-8 syllables
SHORT_METER = (MeterType.IAMB, 3, 3, 4, 3)  # 6-6-8-6 syllables
