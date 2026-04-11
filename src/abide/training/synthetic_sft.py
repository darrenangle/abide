"""Verifier-gated SFT warmup dataset builders for Abide forms."""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from abide.training.prime_rl_env import resolve_prime_rl_form_instances

SourceKind = Literal["synthetic", "public_domain"]
_SUPPORTED_SYNTHETIC_SFT_FORM_SETS = ("well_known", "rl_default")

_FORM_LABELS: dict[str, str] = {
    "Haiku": "haiku",
    "Tanka": "tanka",
    "Limerick": "limerick",
    "ShakespeareanSonnet": "Shakespearean sonnet",
    "PetrarchanSonnet": "Petrarchan sonnet",
    "Villanelle": "villanelle",
    "Ghazal": "ghazal",
    "Sestina": "sestina",
    "Triolet": "triolet",
    "Pantoum": "pantoum",
    "TerzaRima": "terza rima",
    "Rondeau": "rondeau",
    "Clerihew": "clerihew",
    "BluesPoem": "blues poem",
}

_PROMPT_TEMPLATES: tuple[str, ...] = (
    "Write a {form_label} about {topic}. Keep the tone {tone}. Return only the poem.",
    "Compose a {tone} {form_label} centered on {topic}. Return only the poem.",
    "Create a {form_label} on the theme of {topic}. Let it feel {tone}. Return only the poem.",
    "Write a {form_label}. Theme: {topic}. Tone: {tone}. Return only the poem.",
    "Write a {form_label} about {topic}. Follow this structural brief: {description}. Return only the poem.",
    "Compose a {tone} {form_label} about {topic}. Follow this brief: {description}. Return only the poem.",
)


@dataclass(frozen=True)
class SeedPoem:
    form_name: str
    source_id: str
    poem: str
    topic: str
    tone: str
    source_kind: SourceKind


@dataclass(frozen=True)
class VerifiedSeedPoem:
    seed: SeedPoem
    score: float
    passed: bool
    structural_brief: str


# Synthetic and public-domain seed poems mirror the validated fixtures used by the
# form-validation test suite, but are kept in src so the SFT exporter remains
# usable from the installable package instead of depending on tests/.
_SEED_POEMS: tuple[SeedPoem, ...] = (
    SeedPoem(
        form_name="Haiku",
        source_id="haiku_spring_synth",
        poem="""The morning sun glows\nCherry blossoms gently fall\nSpring has come at last""",
        topic="spring blossoms at dawn",
        tone="tender",
        source_kind="synthetic",
    ),
    SeedPoem(
        form_name="Tanka",
        source_id="tanka_moonlake_synth",
        poem="""The autumn moon shines\nCasting silver on the lake\nGentle ripples spread\nWhile the night birds call softly\nDreams drift on the quiet waves""",
        topic="moonlight over a quiet lake",
        tone="meditative",
        source_kind="synthetic",
    ),
    SeedPoem(
        form_name="Limerick",
        source_id="limerick_writer_synth",
        poem="""A writer who lived by the sea\nWrote verses with patience and glee\nHe crafted each line\nTo make the words shine\nAnd shared them for all who could see""",
        topic="a cheerful seaside writer",
        tone="playful",
        source_kind="synthetic",
    ),
    SeedPoem(
        form_name="ShakespeareanSonnet",
        source_id="shakespeare_sonnet_18",
        poem="""Shall I compare thee to a summer's day?\nThou art more lovely and more temperate:\nRough winds do shake the darling buds of May,\nAnd summer's lease hath all too short a date:\nSometime too hot the eye of heaven shines,\nAnd often is his gold complexion dimm'd;\nAnd every fair from fair sometime declines,\nBy chance, or nature's changing course untrimm'd;\nBut thy eternal summer shall not fade,\nNor lose possession of that fair thou ow'st;\nNor shall death brag thou wander'st in his shade,\nWhen in eternal lines to time thou grow'st:\nSo long as men can breathe, or eyes can see,\nSo long lives this, and this gives life to thee.""",
        topic="enduring beauty through time",
        tone="reverent",
        source_kind="public_domain",
    ),
    SeedPoem(
        form_name="PetrarchanSonnet",
        source_id="milton_blindness",
        poem="""When I consider how my light is spent,\nEre half my days, in this dark world and wide,\nAnd that one Talent which is death to hide\nLodged with me useless, though my Soul more bent\nTo serve therewith my Maker, and present\nMy true account, lest he returning chide;\nDoth God exact day-labour, light denied?\nI fondly ask. But patience, to prevent\nThat murmur, soon replies, God doth not need\nEither man's work or his own gifts; who best\nBear his mild yoke, they serve him best. His state\nIs Kingly. Thousands at his bidding speed\nAnd post o'er Land and Ocean without rest:\nThey also serve who only stand and wait.""",
        topic="patience under loss and duty",
        tone="restrained",
        source_kind="public_domain",
    ),
    SeedPoem(
        form_name="Villanelle",
        source_id="villanelle_stars_synth",
        poem="""The stars will shine when day is done\nAnd shadows fall across the land\nWe walk beneath the fading sun\n\nThe birds that sing have now begun\nTo nest where gentle breezes fanned\nThe stars will shine when day is done\n\nThe rivers flow and waters run\nThrough valleys by the master planned\nWe walk beneath the fading sun\n\nWhat once was lost cannot be won\nYet still we reach with open hand\nThe stars will shine when day is done\n\nOur journey here is never done\nThough time slips by like grains of sand\nWe walk beneath the fading sun\n\nWhen all our earthly work is spun\nAnd we have made our final stand\nThe stars will shine when day is done\nWe walk beneath the fading sun""",
        topic="endurance at the close of day",
        tone="melancholic",
        source_kind="synthetic",
    ),
    SeedPoem(
        form_name="Ghazal",
        source_id="ghazal_tonight_above_synth",
        poem="""The moon hangs low and bright tonight above\nThe stars shine clear and light tonight above\n\nI walk alone through empty streets at dusk\nMy path is lit by sight tonight above\n\nThe wind has gone to sleep within the trees\nThe world is calm and right tonight above\n\nWhat dreams may come to those who wait for dawn\nWhen stars take flight tonight above\n\nThe poet writes these words with all his might\nHis verses take their flight tonight above""",
        topic="moonlit solitude and ascent",
        tone="reverent",
        source_kind="synthetic",
    ),
    SeedPoem(
        form_name="Sestina",
        source_id="sestina_morning_walk_synth",
        poem="""I walked alone beneath the ancient trees\nAnd watched the light dance softly on the stream\nThe morning air was fresh with scent of flowers\nWhile birds sang out their songs upon the breeze\nI thought of days now faded into dreams\nAnd felt the turning of the endless hours\n\nI sat beside the bank for many hours\nAnd listened to the whispers of the trees\nThe water flowed like silver through my dreams\nA ribbon winding down the gentle stream\nI closed my eyes and felt the cooling breeze\nAnd breathed the fragrance of the morning flowers\n\nThe meadow there was carpeted with flowers\nI wandered through them counting up the hours\nThe petals danced upon the summer breeze\nCast shadows underneath the swaying trees\nI heard the distant murmur of the stream\nAnd drifted slowly into waking dreams\n\nHow often I had walked here in my dreams\nThrough fields of gold and crimson colored flowers\nAlong the mossy edges of the stream\nI passed away the long and languid hours\nBeneath the sheltering branches of the trees\nAnd felt upon my face the gentle breeze\n\nThere came a sudden stirring of the breeze\nThat woke me gently from my pleasant dreams\nI saw the sunlight slanting through the trees\nIt fell like gold upon the waiting flowers\nI knew that I had lingered here for hours\nBut still I sat beside the quiet stream\n\nHow beautiful and peaceful was that stream\nHow soft and sweet the ever-present breeze\nI wished that I could stay for endless hours\nAnd never wake from all my lovely dreams\nForever walking through the fragrant flowers\nForever resting underneath the trees\n\nThe stream flows on through dreams beneath the trees\nWhile flowers bloom and hours pass with the breeze\nAnd all my dreams drift softly through the hours""",
        topic="a daylong walk beside a stream",
        tone="meditative",
        source_kind="synthetic",
    ),
    SeedPoem(
        form_name="Triolet",
        source_id="triolet_roses_synth",
        poem="""The roses bloom in early spring\nAnd fill the air with sweet perfume\nThe morning birds begin to sing\nThe roses bloom in early spring\nWhat joy and beauty they do bring\nBefore the summer's heat and gloom\nThe roses bloom in early spring\nAnd fill the air with sweet perfume""",
        topic="roses in early spring",
        tone="joyful",
        source_kind="synthetic",
    ),
    SeedPoem(
        form_name="Pantoum",
        source_id="pantoum_rainroof_synth",
        poem="""The rain falls soft upon the roof tonight\nAnd shadows dance upon the window pane\nI sit alone and read by candlelight\nWhile thunder rumbles distantly again\n\nAnd shadows dance upon the window pane\nThe storm moves slowly over hills afar\nWhile thunder rumbles distantly again\nI watch and wait beneath the evening star\n\nThe storm moves slowly over hills afar\nThe wind begins to whisper through the eaves\nI watch and wait beneath the evening star\nAnd listen to the rustling of the leaves\n\nThe wind begins to whisper through the eaves\nThe rain falls soft upon the roof tonight\nAnd listen to the rustling of the leaves\nI sit alone and read by candlelight""",
        topic="reading through a rainstorm",
        tone="melancholic",
        source_kind="synthetic",
    ),
    SeedPoem(
        form_name="TerzaRima",
        source_id="terza_morning_hill_synth",
        poem="""The morning light breaks golden on the hill\nAnd wakes the sleeping valley down below\nThe birds begin their chorus sharp and shrill\n\nAcross the fields the gentle breezes blow\nAnd carry scents of flowers on the air\nThe river sparkles in the morning glow\n\nThe trees stand tall and green beyond compare\nTheir branches reaching upward to the sky\nThe world awakes and life is everywhere\n\nAnd so another day goes drifting by""",
        topic="morning in a waking valley",
        tone="reverent",
        source_kind="synthetic",
    ),
    SeedPoem(
        form_name="Rondeau",
        source_id="flanders_fields",
        poem="""In Flanders fields the poppies blow\nBetween the crosses, row on row,\nThat mark our place; and in the sky\nThe larks, still bravely singing, fly\nScarce heard amid the guns below.\n\nWe are the Dead. Short days ago\nWe lived, felt dawn, saw sunset glow,\nLoved and were loved, and now we lie\nIn Flanders fields.\n\nTake up our quarrel with the foe:\nTo you from failing hands we throw\nThe torch; be yours to hold it high.\nIf ye break faith with us who die\nWe shall not sleep, though poppies grow\nIn Flanders fields.""",
        topic="memory and sacrifice in war",
        tone="severe",
        source_kind="public_domain",
    ),
    SeedPoem(
        form_name="Clerihew",
        source_id="clerihew_poe_synth",
        poem="""Edgar Allan Poe\nWrote tales of woe and snow\nHis stories caused great fright\nWith ravens in the night""",
        topic="Edgar Allan Poe and his haunted reputation",
        tone="wry",
        source_kind="synthetic",
    ),
    SeedPoem(
        form_name="BluesPoem",
        source_id="blues_old_street_synth",
        poem="""Woke up this morning with the sun in my eyes\nWoke up this morning with the sun in my eyes\nLooked out my window and I watched the blue birds rise\n\nMy baby left me standing at the door\nMy baby left me standing at the door\nNow I don't know what I'm living for\n\nThe rain keeps falling down on this old street\nThe rain keeps falling down on this old street\nAnd I got nothing left but my tired feet""",
        topic="heartbreak and weather on an old street",
        tone="melancholic",
        source_kind="synthetic",
    ),
)


def humanize_form_name(form_name: str) -> str:
    """Convert exported class names into readable form labels."""
    if form_name in _FORM_LABELS:
        return _FORM_LABELS[form_name]
    return re.sub(r"(?<!^)(?=[A-Z])", " ", form_name).strip().lower()


def _parse_form_names(form_names: str | list[str] | tuple[str, ...] | None) -> list[str] | None:
    if form_names is None:
        return None
    if isinstance(form_names, str):
        parsed = [name.strip() for name in form_names.split(",") if name.strip()]
        return parsed or None
    parsed = [name.strip() for name in form_names if name.strip()]
    return parsed or None


def _select_seed_poems(
    *,
    selected_form_names: set[str],
    include_synthetic: bool,
    include_public_domain: bool,
) -> list[SeedPoem]:
    allowed_source_kinds: set[SourceKind] = set()
    if include_synthetic:
        allowed_source_kinds.add("synthetic")
    if include_public_domain:
        allowed_source_kinds.add("public_domain")
    if not allowed_source_kinds:
        raise ValueError("At least one of include_synthetic/include_public_domain must be true.")

    selected = [
        seed
        for seed in _SEED_POEMS
        if seed.form_name in selected_form_names and seed.source_kind in allowed_source_kinds
    ]
    covered_forms = {seed.form_name for seed in selected}
    missing = sorted(selected_form_names - covered_forms)
    if missing:
        raise ValueError("Missing seed poems for selected forms: " + ", ".join(missing))
    return selected


def _build_prompt(
    seed: SeedPoem,
    *,
    structural_brief: str,
    variant_index: int,
    rng: random.Random,
) -> str:
    template = _PROMPT_TEMPLATES[variant_index % len(_PROMPT_TEMPLATES)]
    prompt = template.format(
        form_label=humanize_form_name(seed.form_name),
        topic=seed.topic,
        tone=seed.tone,
        description=structural_brief,
    )
    if variant_index >= len(_PROMPT_TEMPLATES):
        suffix_options = (
            "Return only the poem.",
            "No title or explanation.",
            "Do not add commentary.",
        )
        suffix = suffix_options[rng.randrange(len(suffix_options))]
        prompt = f"{prompt}\n{suffix}"
    return prompt


def build_synthetic_sft_records(
    *,
    form_set: str = "rl_default",
    form_names: str | list[str] | tuple[str, ...] | None = None,
    prompt_variants_per_seed: int = 8,
    seed: int = 42,
    include_synthetic: bool = True,
    include_public_domain: bool = True,
    require_passed: bool = True,
    min_score: float = 0.0,
) -> list[dict[str, Any]]:
    """Build verifier-gated chat-SFT records for the selected forms."""
    if form_set not in _SUPPORTED_SYNTHETIC_SFT_FORM_SETS and form_names is None:
        raise ValueError(
            "Synthetic SFT builder currently supports form_set='well_known' or 'rl_default'; "
            f"got {form_set!r}."
        )
    if prompt_variants_per_seed < 1:
        raise ValueError("prompt_variants_per_seed must be at least 1.")

    explicit_form_names = _parse_form_names(form_names)
    forms = resolve_prime_rl_form_instances(form_set=form_set, form_names=explicit_form_names)
    selected_form_names = set(forms)
    seed_poems = _select_seed_poems(
        selected_form_names=selected_form_names,
        include_synthetic=include_synthetic,
        include_public_domain=include_public_domain,
    )

    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    seen_prompts: set[tuple[str, str]] = set()

    for seed_poem in seed_poems:
        form = forms[seed_poem.form_name]
        result = form.verify(seed_poem.poem)
        verified = VerifiedSeedPoem(
            seed=seed_poem,
            score=float(result.score),
            passed=bool(result.passed),
            structural_brief=form.describe(),
        )
        if require_passed and not verified.passed:
            raise ValueError(
                f"Seed poem {seed_poem.source_id!r} for {seed_poem.form_name} does not pass "
                f"the verifier (score={verified.score:.4f})."
            )
        if verified.score < min_score:
            raise ValueError(
                f"Seed poem {seed_poem.source_id!r} for {seed_poem.form_name} scored "
                f"{verified.score:.4f}, below min_score={min_score:.4f}."
            )

        for variant_index in range(prompt_variants_per_seed):
            prompt = _build_prompt(
                seed_poem,
                structural_brief=verified.structural_brief,
                variant_index=variant_index,
                rng=rng,
            )
            prompt_key = (seed_poem.source_id, prompt)
            if prompt_key in seen_prompts:
                continue
            seen_prompts.add(prompt_key)
            records.append(
                {
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": seed_poem.poem},
                    ],
                    "prompt": prompt,
                    "response": seed_poem.poem,
                    "form_name": seed_poem.form_name,
                    "form_label": humanize_form_name(seed_poem.form_name),
                    "source_id": seed_poem.source_id,
                    "source_kind": seed_poem.source_kind,
                    "topic": seed_poem.topic,
                    "tone": seed_poem.tone,
                    "structural_brief": verified.structural_brief,
                    "verifier_score": verified.score,
                    "verifier_passed": verified.passed,
                }
            )

    return records


def summarize_synthetic_sft_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize a synthetic SFT record set for logging or JSON export."""
    form_counts = Counter(record["form_name"] for record in records)
    source_counts = Counter(record["source_kind"] for record in records)
    return {
        "num_records": len(records),
        "form_counts": dict(sorted(form_counts.items())),
        "source_kind_counts": dict(sorted(source_counts.items())),
    }


def write_synthetic_sft_jsonl(records: list[dict[str, Any]], output_path: str | Path) -> Path:
    """Write chat-SFT records to JSONL."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
    return output


__all__ = [
    "SeedPoem",
    "build_synthetic_sft_records",
    "humanize_form_name",
    "summarize_synthetic_sft_records",
    "write_synthetic_sft_jsonl",
]
