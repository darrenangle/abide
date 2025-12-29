"""
SYNTH-style reasoning trace generation for poetry form adherence.

Based on Baguettotron/Pleias SYNTH stenographic notation:
- Logical: → ↺ ? !/※ ≈ ∴
- Confidence: ● ◐ ○ ⚠ ?maybe?
- Verification: ☐ ☑ ✓
- Entropy: ⟨H≈X.X⟩

See: https://huggingface.co/PleIAs/Baguettotron
"""

from dataclasses import dataclass
from typing import Any, cast


@dataclass
class RubricItem:
    """A single rubric item from verification."""

    criterion: str
    score: float
    passed: bool = True
    details: str = ""


def confidence_marker(score: float) -> str:
    """Return SYNTH confidence marker based on score."""
    if score >= 0.95:
        return "●"  # High confidence
    elif score >= 0.8:
        return "◐"  # Medium confidence
    elif score >= 0.5:
        return "○"  # Low confidence
    else:
        return "⚠"  # Warning/problem


def format_constraint_tree(form_name: str, rubric: list[RubricItem], topic: str) -> str:
    """Format constraints as SYNTH-style tree structure."""
    lines = [f"{form_name} requirements:"]

    for i, item in enumerate(rubric):
        marker = confidence_marker(item.score)
        prefix = "├─" if i < len(rubric) - 1 else "└─"
        lines.append(f"{prefix} {item.criterion}: {marker}")

    if topic:
        lines.append(f"└─ topic: {topic}")

    return "\n".join(lines)


def synthesize_planning_section(form_name: str, form_instruction: str) -> str:
    """Generate SYNTH-style planning section."""
    lines = []
    lines.append("### 1. Task Analysis")
    lines.append(f"→ Write {form_name} poem")

    # Parse instruction into key requirements
    if "line" in form_instruction.lower():
        lines.append("→ Line count constraint detected ●")
    if "syllable" in form_instruction.lower():
        lines.append("→ Syllable pattern required ◐")
    if "rhyme" in form_instruction.lower():
        lines.append("→ Rhyme scheme specified ●")
    if "meter" in form_instruction.lower():
        lines.append("→ Meter constraint active ○")

    return "\n".join(lines)


def synthesize_verification_section(rubric: list[RubricItem]) -> str:
    """Generate SYNTH-style verification section."""
    lines = ["### 3. Verification Process"]

    for item in rubric:
        if item.score >= 0.95:
            lines.append(f"✓ {item.criterion}: {item.score:.0%}")
        elif item.score >= 0.8:
            lines.append(f"☑ {item.criterion}: {item.score:.0%}")
        else:
            lines.append(f"☐ {item.criterion}: {item.score:.0%} ⚠")

    return "\n".join(lines)


def synthesize_conclusion(form_name: str, total_score: float) -> str:
    """Generate SYNTH-style conclusion."""
    marker = confidence_marker(total_score)

    if total_score >= 0.9:
        return f"∴ {form_name} complete: {marker} high confidence → all constraints satisfied"
    elif total_score >= 0.8:
        return f"∴ {form_name} acceptable: {marker} minor deviations within tolerance"
    else:
        return f"∴ {form_name} needs revision: {marker} constraint failures detected"


def generate_synth_trace(
    form_name: str,
    topic: str,
    tone: str,
    form_instruction: str,
    rubric: list[RubricItem] | list[dict[str, Any]],
    total_score: float,
    poem_lines: list[str] | None = None,
) -> str:
    """
    Generate a full SYNTH-style reasoning trace for poetry generation.

    Example output:
    ```
    <think>
    Sonnet requirements:
    ├─ lines: 14 ●
    ├─ meter: iambic pentameter ◐
    ├─ rhyme: ABAB CDCD EFEF GG ●
    └─ topic: autumn melancholy

    ### 1. Task Analysis
    → Write Sonnet poem
    → Line count constraint detected ●
    → Rhyme scheme specified ●
    → Meter constraint active ○

    ### 2. Composition Strategy
    ⟨H≈0.5⟩ exploring thematic elements...
    ├─ autumn imagery: leaves, cold, fading
    ├─ melancholic tone: loss, memory, time
    └─ structural approach: volta at line 9

    ?maybe? use extended metaphor?
    → autumn as life's twilight ●

    ### 3. Verification Process
    ✓ line_count: 100%
    ☑ syllables: 85%
    ✓ rhyme_scheme: 92%

    ∴ Sonnet complete: ● high confidence
    </think>
    ```
    """
    # Convert rubric to RubricItem if needed
    rubric_items: list[RubricItem]
    if rubric and isinstance(rubric[0], dict):
        dict_rubric = cast("list[dict[str, Any]]", rubric)
        rubric_items = [
            RubricItem(
                criterion=d.get("criterion", "unknown"),
                score=d.get("score", 0.0),
                passed=d.get("passed", d.get("score", 0) >= 0.5),
                details=d.get("details", ""),
            )
            for d in dict_rubric
        ]
    else:
        rubric_items = cast("list[RubricItem]", rubric)

    lines = ["<think>"]

    # 1. Requirements tree
    lines.append(format_constraint_tree(form_name, rubric_items, topic))
    lines.append("")

    # 2. Task analysis
    lines.append(synthesize_planning_section(form_name, form_instruction))
    lines.append("")

    # 3. Composition strategy
    lines.append("### 2. Composition Strategy")
    lines.append("⟨H≈0.5⟩ exploring thematic elements...")
    lines.append(f"├─ topic elements: {topic}")
    lines.append(f"├─ tonal approach: {tone}")
    lines.append("└─ structural planning: section-by-section")
    lines.append("")

    # Add line-by-line analysis if poem provided
    if poem_lines and len(poem_lines) > 0:
        lines.append("Line composition:")
        for i, line in enumerate(poem_lines[:3]):  # First 3 lines
            lines.append(
                f'├─ L{i + 1}: "{line[:40]}..." ●' if len(line) > 40 else f'├─ L{i + 1}: "{line}" ●'
            )
        if len(poem_lines) > 3:
            lines.append(f"└─ ... {len(poem_lines) - 3} more lines")
        lines.append("")

    # 4. Verification
    lines.append(synthesize_verification_section(rubric_items))
    lines.append("")

    # 5. Conclusion
    lines.append(synthesize_conclusion(form_name, total_score))
    lines.append("</think>")

    return "\n".join(lines)


def generate_natural_trace(
    form_name: str,
    topic: str,
    tone: str,
    form_instruction: str,
    rubric: list[RubricItem] | list[dict[str, Any]],
    total_score: float,
) -> str:
    """
    Generate a natural (non-SYNTH) reasoning trace.

    Uses plain English inside <think> tags, like a human thinking through
    the poetry composition process.

    Example output:
    ```
    <think>
    I need to write a Sonnet about autumn with a melancholic tone.

    A Sonnet has 14 lines with iambic pentameter - that's 10 syllables per line
    in a da-DUM da-DUM pattern. The rhyme scheme is ABAB CDCD EFEF GG.

    Let me start with the first line. I want something that evokes autumn's
    fading beauty...

    For line 3, I need something that rhymes with the first line ending...

    Let me verify my work:
    - Line count: 14 lines, good!
    - Syllables: mostly 10 per line
    - Rhyme scheme: checking A-A, B-B matches...

    I think this captures the melancholic tone well.
    </think>
    ```
    """
    # Convert rubric to RubricItem if needed
    rubric_items: list[RubricItem]
    if rubric and isinstance(rubric[0], dict):
        dict_rubric = cast("list[dict[str, Any]]", rubric)
        rubric_items = [
            RubricItem(
                criterion=d.get("criterion", "unknown"),
                score=d.get("score", 0.0),
                passed=d.get("passed", d.get("score", 0) >= 0.5),
            )
            for d in dict_rubric
        ]
    else:
        rubric_items = cast("list[RubricItem]", rubric)

    lines = ["<think>"]
    lines.append(f"I need to write a {form_name} about {topic} with a {tone} tone.")
    lines.append("")

    # Explain form requirements naturally
    lines.append(f"Let me recall the requirements for a {form_name}:")
    lines.append(form_instruction[:500] if len(form_instruction) > 500 else form_instruction)
    lines.append("")

    # Planning
    lines.append("My approach:")
    lines.append(f"- Theme: {topic}")
    lines.append(f"- Mood: {tone}")
    lines.append("- I'll work through the structure carefully, checking each constraint.")
    lines.append("")

    # Composition thoughts
    lines.append("As I write each line, I'm considering:")
    lines.append("- Does it fit the required pattern?")
    lines.append("- Does it maintain the tone?")
    lines.append("- Does it connect to the theme?")
    lines.append("")

    # Verification
    lines.append("Let me verify my work:")
    for item in rubric_items:
        status = (
            "good!" if item.score >= 0.8 else "needs attention" if item.score >= 0.5 else "problem"
        )
        lines.append(f"- {item.criterion}: {item.score:.0%} - {status}")
    lines.append("")

    # Conclusion
    if total_score >= 0.8:
        lines.append(
            f"I think this {form_name} successfully captures the {tone} tone while maintaining the form's requirements."
        )
    else:
        lines.append(
            f"This {form_name} has some issues with the form requirements that could be improved."
        )

    lines.append("</think>")

    return "\n".join(lines)


# SYNTH trace templates for specific form types
FORM_TRACE_TEMPLATES = {
    "Sonnet": """<think>
{form_name} requirements:
├─ lines: 14 ●
├─ meter: iambic pentameter ◐
├─ rhyme: ABAB CDCD EFEF GG ●
└─ topic: {topic}

### 1. Structural Planning
→ 14 lines: 3 quatrains + 1 couplet
→ volta expected around line 9 ◐
→ iambic pentameter: da-DUM x 5

### 2. Thematic Development
⟨H≈0.5⟩ {topic} imagery...
├─ concrete images to ground abstraction
├─ {tone} emotional register
└─ progression toward resolution

### 3. Verification
{verification}

∴ {conclusion}
</think>""",
    "Haiku": """<think>
{form_name} requirements:
├─ lines: 3 ●
├─ syllables: 5-7-5 ●
└─ topic: {topic}

### 1. Structure Check
→ L1: 5 syllables
→ L2: 7 syllables
→ L3: 5 syllables

### 2. Imagery
⟨H≈0.3⟩ {topic} essence...
├─ seasonal reference (kigo)
├─ cutting word (kireji) moment
└─ {tone} sensibility

### 3. Verification
{verification}

∴ {conclusion}
</think>""",
}


def generate_synth_trace_for_form(
    form_name: str,
    topic: str,
    tone: str,
    rubric: list[dict[str, Any]],
    total_score: float,
) -> str:
    """Generate a form-specific SYNTH trace using templates when available."""
    # Build verification section
    verif_lines = []
    for item in rubric:
        score = item.get("score", 0)
        criterion = item.get("criterion", "unknown")
        if score >= 0.95:
            verif_lines.append(f"✓ {criterion}: {score:.0%}")
        elif score >= 0.8:
            verif_lines.append(f"☑ {criterion}: {score:.0%}")
        else:
            verif_lines.append(f"☐ {criterion}: {score:.0%} ⚠")
    verification = "\n".join(verif_lines)

    # Build conclusion
    marker = confidence_marker(total_score)
    if total_score >= 0.9:
        conclusion = f"{form_name} complete: {marker} high confidence"
    elif total_score >= 0.8:
        conclusion = f"{form_name} acceptable: {marker} minor issues"
    else:
        conclusion = f"{form_name} needs work: {marker}"

    # Use template if available
    if form_name in FORM_TRACE_TEMPLATES:
        return FORM_TRACE_TEMPLATES[form_name].format(
            form_name=form_name,
            topic=topic,
            tone=tone,
            verification=verification,
            conclusion=conclusion,
        )

    # Fall back to generic trace
    return generate_synth_trace(
        form_name=form_name,
        topic=topic,
        tone=tone,
        form_instruction="",
        rubric=[
            RubricItem(criterion=r.get("criterion", ""), score=r.get("score", 0)) for r in rubric
        ],
        total_score=total_score,
    )
