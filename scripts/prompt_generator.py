#!/usr/bin/env python3
"""
Recombinant prompt generator for GRPO training.

Generates highly diverse prompts by combining:
- Forms (abide poetry forms)
- Topics (330+ base + 140+ niche + combinatorial mixing)
- Tones/moods
- Perspectives
- Temporal framings
- Sensory focuses

Creates a massive dataset for RL training with guaranteed uniqueness.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

# =============================================================================
# TOPICS - Ported from 10k-poems with niche/expert terminology
# =============================================================================

NATURE_TOPICS = [
    "winter",
    "spring",
    "summer",
    "autumn",
    "garden",
    "forest",
    "ocean",
    "mountain",
    "river",
    "desert",
    "sky",
    "rain",
    "snow",
    "wind",
    "light",
    "darkness",
    "dawn",
    "dusk",
    "night",
    "moon",
    "stars",
    "sun",
    "clouds",
    "storm",
    "frost",
    "fog",
    "mist",
    "thunder",
    "lightning",
    "tide",
    "waves",
    "trees",
    "flowers",
    "grass",
    "leaves",
    "seeds",
    "roots",
    "branches",
    "birds",
    "insects",
    "fish",
    "animals",
    "stones",
    "rocks",
    "sand",
    "soil",
    "earth",
    "ice",
    "fire",
    "water",
    "air",
]

HUMAN_EXPERIENCE = [
    "memory",
    "childhood",
    "aging",
    "family",
    "love",
    "loss",
    "grief",
    "joy",
    "loneliness",
    "belonging",
    "identity",
    "migration",
    "home",
    "exile",
    "birth",
    "death",
    "marriage",
    "divorce",
    "friendship",
    "betrayal",
    "forgiveness",
    "anger",
    "jealousy",
    "compassion",
    "guilt",
    "shame",
    "pride",
    "humility",
    "courage",
    "fear",
    "anxiety",
    "peace",
    "longing",
    "satisfaction",
    "emptiness",
    "fulfillment",
    "confusion",
    "clarity",
    "dreams",
    "nightmares",
    "awakening",
    "sleep",
]

ABSTRACT_CONCEPTS = [
    "time",
    "patience",
    "hope",
    "regret",
    "desire",
    "silence",
    "waiting",
    "change",
    "permanence",
    "truth",
    "beauty",
    "power",
    "freedom",
    "justice",
    "mercy",
    "fate",
    "chance",
    "destiny",
    "purpose",
    "meaning",
    "void",
    "infinity",
    "eternity",
    "mortality",
    "presence",
    "absence",
    "distance",
    "proximity",
    "balance",
    "chaos",
    "order",
    "symmetry",
    "asymmetry",
    "repetition",
    "variation",
    "stillness",
    "motion",
    "transformation",
    "stasis",
    "beginning",
    "ending",
]

EVERYDAY_LIFE = [
    "work",
    "objects",
    "routine",
    "meals",
    "walking",
    "conversation",
    "solitude",
    "cities",
    "streets",
    "buildings",
    "rooms",
    "windows",
    "doors",
    "walls",
    "floors",
    "stairs",
    "bridges",
    "roads",
    "markets",
    "shops",
    "cafes",
    "trains",
    "buses",
    "cars",
    "hands",
    "eyes",
    "voice",
    "breath",
    "touch",
    "taste",
    "smell",
    "sound",
    "noise",
    "quiet",
    "crowds",
    "strangers",
    "neighbors",
    "letters",
    "phones",
    "mirrors",
    "clocks",
    "keys",
]

ART_CULTURE = [
    "music",
    "painting",
    "books",
    "language",
    "translation",
    "history",
    "mythology",
    "religion",
    "science",
    "technology",
    "war",
    "politics",
    "poetry",
    "theater",
    "dance",
    "sculpture",
    "film",
    "photography",
    "architecture",
    "philosophy",
    "mathematics",
    "astronomy",
    "geography",
    "archaeology",
    "anthropology",
    "ritual",
    "ceremony",
    "tradition",
    "innovation",
    "revolution",
    "evolution",
    "progress",
    "decline",
]

# Out-of-distribution niche topics (expert/hobbyist knowledge)
# These create interesting, unexpected prompts that force diverse outputs
NICHE_TOPICS = [
    # Crafts & Making
    "warp and weft",
    "selvage",
    "mordant",
    "annealing",
    "burnishing",
    "patina",
    "flux",
    "slake",
    "temper",
    "slip",
    "grog",
    "raku",
    # Climbing & Outdoors
    "crux",
    "crimp",
    "belay",
    "rappel",
    "cairn",
    "col",
    "moraine",
    "scree",
    "talus",
    "couloir",
    "bergschrund",
    "serac",
    # Cooking & Fermentation
    "mise en place",
    "mother",
    "pellicle",
    "bloom",
    "crumb",
    "lamination",
    "fond",
    "maillard",
    "sous vide",
    "bain-marie",
    "diastatic",
    "poolish",
    # Music (technical)
    "sustain",
    "timbre",
    "tremolo",
    "legato",
    "staccato",
    "fermata",
    "cadence",
    "voicing",
    "temperament",
    "overtones",
    "drone",
    "ostinato",
    # Fiber Arts & Textiles
    "roving",
    "singles",
    "ply",
    "skein",
    "hank",
    "bobbin",
    "shuttle",
    "heddle",
    "reed",
    "treadle",
    "niddy-noddy",
    "swift",
    # Gardening & Horticulture
    "tilth",
    "hardening off",
    "bolting",
    "stratification",
    "scarification",
    "deadheading",
    "pinching",
    "vernalization",
    "damping off",
    "leggy",
    # Woodworking
    "kerf",
    "grain",
    "burl",
    "spalting",
    "dovetail",
    "mortise",
    "tenon",
    "rabbet",
    "dado",
    "chamfer",
    "bevel",
    "flush",
    "proud",
    # Bookbinding & Letterpress
    "signature",
    "gathering",
    "folio",
    "quire",
    "endpaper",
    "headband",
    "kettle stitch",
    "smyth sewing",
    "leading",
    "kerning",
    "furniture",
    # Astronomy & Navigation
    "azimuth",
    "zenith",
    "nadir",
    "meridian",
    "parallax",
    "precession",
    "libration",
    "syzygy",
    "occultation",
    "conjunction",
    "opposition",
    # Geology & Mineralogy
    "cleavage",
    "luster",
    "habit",
    "twinning",
    "striations",
    "conchoidal",
    "drusy",
    "botryoidal",
    "massive",
    "foliation",
    "unconformity",
    # Mycology
    "mycelium",
    "primordium",
    "volva",
    "annulus",
    "spore print",
    "hyphae",
    "fruiting body",
    "rhizomorph",
    "sclerotium",
    "fairy ring",
    # Maritime & Sailing
    "windward",
    "leeward",
    "halyard",
    "boom",
    "tack",
    "jibe",
    "heave to",
    "painter",
    "gunwale",
    "thwart",
    "cleat",
    "fairlead",
    "bitter end",
    # Falconry & Birds
    "mews",
    "jesses",
    "bewit",
    "creance",
    "mantling",
    "bating",
    "rousing",
    "casting",
    "hack",
    "eyass",
    "passager",
    "haggard",
    # Beekeeping
    "propolis",
    "brood",
    "queen cell",
    "drone comb",
    "festooning",
    "waggle dance",
    "piping",
    "absconding",
    "supersedure",
    "nasonov",
    # Printmaking
    "burr",
    "aquatint",
    "chine-collé",
    "ghost print",
    "viscosity",
    "reduction",
    "carborundum",
    "sugar lift",
    "spit bite",
    "drypoint",
    # Cheese & Dairy
    "affinage",
    "bloomy rind",
    "washed rind",
    "eyes",
    "paste",
    "tyrosine",
    "rennet",
    "curds",
    "whey",
    "cave",
    "terroir",
    # Blacksmithing & Metalwork
    "quench",
    "fuller",
    "swage",
    "pritchel",
    "hardy",
    "punching",
    "drifting",
    "upsetting",
    "drawing out",
    "forge welding",
    "scale",
    "clinker",
    # Archery
    "nock",
    "nocking point",
    "serving",
    "flemish twist",
    "tillering",
    "stacking",
    "hand shock",
    "paradox",
    "anchor point",
    "cant",
    # Microscopy & Biology
    "parfocal",
    "coverslip",
    "immersion",
    "phase contrast",
    "fixing",
    "mounting medium",
    "microtome",
    "staining",
    "serial sections",
    # Coffee & Tea
    "channeling",
    "extraction",
    "agitation",
    "degassing",
    "first crack",
    "second crack",
    "chaff",
    "silverskin",
    "tamp",
]

# =============================================================================
# STYLE DIMENSIONS (replacing poet names)
# =============================================================================

TONES = [
    "contemplative",
    "urgent",
    "wistful",
    "sardonic",
    "reverent",
    "melancholic",
    "jubilant",
    "restrained",
    "fierce",
    "tender",
    "ironic",
    "earnest",
    "detached",
    "intimate",
    "elegiac",
    "playful",
    "solemn",
    "defiant",
    "resigned",
    "wondering",
    "bitter",
    "hopeful",
    "ambivalent",
    "serene",
    "turbulent",
]

PERSPECTIVES = [
    "from a great distance",
    "up close",
    "from within",
    "as if for the first time",
    "as if for the last time",
    "through a child's eyes",
    "through an elder's memory",
    "from the threshold",
    "from the center",
    "as witness",
    "as participant",
    "as the thing itself",
    "looking back",
    "looking forward",
    "in the eternal present",
    "from above",
    "from below",
    "from the margins",
]

TEMPORAL_FRAMES = [
    "at the moment of",
    "in the aftermath of",
    "on the eve of",
    "during the long wait for",
    "in the middle of",
    "at the turning point of",
    "in the pause before",
    "as it unfolds",
    "in retrospect",
    "in anticipation",
    "at the boundary between",
    "in the cycle of",
    "at the end of",
    "at the beginning of",
]

SENSORY_FOCUSES = [
    "through texture and touch",
    "through sound and silence",
    "through light and shadow",
    "through scent and memory",
    "through taste and hunger",
    "through movement and stillness",
    "through weight and weightlessness",
    "through heat and cold",
    "through the body's knowledge",
    "through what remains unseen",
]

APPROACHES = [
    "as meditation",
    "as elegy",
    "as praise",
    "as lament",
    "as argument",
    "as question",
    "as confession",
    "as witness",
    "as inventory",
    "as instruction",
    "as prayer",
    "as curse",
    "as letter",
    "as farewell",
    "as return",
    "as discovery",
    "through accumulation",
    "through subtraction",
    "through juxtaposition",
]

# =============================================================================
# TOPIC COMBINATION LOGIC
# =============================================================================

ALL_BASE_TOPICS = (
    NATURE_TOPICS
    + HUMAN_EXPERIENCE
    + ABSTRACT_CONCEPTS
    + EVERYDAY_LIFE
    + ART_CULTURE
    + NICHE_TOPICS
)

# Category pairs for two-word combinations
CATEGORY_PAIRS = [
    (NATURE_TOPICS, ABSTRACT_CONCEPTS),
    (NATURE_TOPICS, HUMAN_EXPERIENCE),
    (EVERYDAY_LIFE, ABSTRACT_CONCEPTS),
    (EVERYDAY_LIFE, HUMAN_EXPERIENCE),
    (ART_CULTURE, NATURE_TOPICS),
    (ART_CULTURE, HUMAN_EXPERIENCE),
    (NICHE_TOPICS, ABSTRACT_CONCEPTS),
    (NICHE_TOPICS, HUMAN_EXPERIENCE),
    (NICHE_TOPICS, NATURE_TOPICS),
]


def generate_combined_topics(seed: int = 42) -> list[str]:
    """Generate combinatorial topic variations."""
    rng = random.Random(seed)
    combined = []

    # Two-word combinations from different categories
    for cat1, cat2 in CATEGORY_PAIRS:
        sample1 = rng.sample(cat1, min(20, len(cat1)))
        sample2 = rng.sample(cat2, min(20, len(cat2)))
        for t1 in sample1:
            for t2 in sample2:
                combined.append(f"{t1} and {t2}")

    # Three-word combinations (more selective)
    nature_sample = rng.sample(NATURE_TOPICS, 15)
    human_sample = rng.sample(HUMAN_EXPERIENCE, 15)
    abstract_sample = rng.sample(ABSTRACT_CONCEPTS, 10)

    for t1 in nature_sample:
        for t2 in human_sample:
            for t3 in abstract_sample:
                combined.append(f"{t1}, {t2}, and {t3}")

    return combined


def get_all_topics(seed: int = 42) -> list[str]:
    """Get all topics including base and combined."""
    combined = generate_combined_topics(seed)
    return ALL_BASE_TOPICS + combined


# =============================================================================
# FORM TIERS - For weighted sampling of traditional forms
# =============================================================================

# Tier 1: Classic forms everyone knows (weight: 5)
TIER_1_FORMS = [
    "Sonnet",
    "ShakespeareanSonnet",
    "PetrarchanSonnet",
    "Haiku",
    "Senryu",
    "Tanka",
    "Limerick",
    "Couplet",
    "Tercet",
    "Quatrain",
    "FreeVerse",
    "BlankVerse",
    "Ballad",
    "BalladStanza",
    "Ode",
    "Villanelle",
]

# Tier 2: Well-known traditional forms (weight: 3)
TIER_2_FORMS = [
    "Ghazal",
    "Sestina",
    "Rondeau",
    "Rondel",
    "Triolet",
    "Pantoum",
    "TerzaRima",
    "OttavaRima",
    "RhymeRoyal",
    "SpenserianStanza",
    "SpenserianSonnet",
    "CurtalSonnet",
    "CaudateSonnet",
    "HeroicCouplet",
    "HeroicQuatrain",
    "EnvelopeQuatrain",
    "Aubade",
    "Elegiac",
    "Epigram",
    "Cinquain",
    "Clerihew",
    "Rispetto",
    "Canzone",
    "Rubai",
    "Rubaiyat",
]

# Tier 3: Less common but legitimate historical forms (weight: 1)
TIER_3_FORMS = [
    "Diamante",
    "Etheree",
    "ReverseEtheree",
    "Ballade",
    "DoubleBallade",
    "ChantRoyal",
    "Kyrielle",
    "KyrielleSonnet",
    "BurnsStanza",
    "OneginStanza",
    "Lai",
    "Virelai",
    "Roundel",
    "Rondelet",
    "Rondine",
    "Tritina",
    "Quatina",
    "Quintina",
    "SandwichSonnet",
    "CrownOfSonnets",
    "Tanaga",
    "Katauta",
    "Sedoka",
    "Naani",
    "Seguidilla",
    "BluesPoem",
    "Bop",
    "ProsePoem",
    "DramaticVerse",
    "Monostich",
    "Distich",
    "Triplet",
    "Terzanelle",
    "SapphicStanza",
    "SapphicOde",
    "PindaricOde",
    "HoratianOde",
    "IrregularOde",
    "LiteraryBallad",
    "BroadBallad",
    "Skeltonic",
]

# All traditional forms (excludes constraint-based/mathematical forms)
TRADITIONAL_FORMS = set(TIER_1_FORMS + TIER_2_FORMS + TIER_3_FORMS)

# Tier weights for sampling
TIER_WEIGHTS = {1: 5, 2: 3, 3: 1}


def get_form_tier(form_name: str) -> int:
    """Get the tier of a form (1, 2, 3, or 0 for excluded)."""
    if form_name in TIER_1_FORMS:
        return 1
    elif form_name in TIER_2_FORMS:
        return 2
    elif form_name in TIER_3_FORMS:
        return 3
    return 0


# =============================================================================
# FORMS
# =============================================================================


def get_forms() -> dict[str, object]:
    """Load ALL training forms from abide.forms."""
    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    import abide.forms as forms_module

    all_forms = {}
    for name in forms_module.__all__:
        try:
            form_class = getattr(forms_module, name)
            # Try to instantiate with no args first
            try:
                all_forms[name] = form_class()
            except TypeError:
                # Some forms need specific params - use sensible defaults
                if name == "StaircasePoem" or name == "DescendingStaircasePoem":
                    all_forms[name] = form_class(num_words=7)
                elif name == "VowelBudgetPoem":
                    all_forms[name] = form_class(vowel_count=30)
                elif name == "PrecisionVerse":
                    all_forms[name] = form_class(chars_per_line=25)
                elif name == "ExactWordPoem":
                    all_forms[name] = form_class(word_count=20)
                elif name == "CharacterBudgetPoem":
                    all_forms[name] = form_class(character="e", count=10)
                elif name == "TotalCharacterPoem":
                    all_forms[name] = form_class(total_chars=100)
                elif name == "FibonacciVerse":
                    all_forms[name] = form_class(num_lines=5)
                elif name == "TriangularVerse":
                    all_forms[name] = form_class(num_lines=4)
                elif name == "PiKu":
                    all_forms[name] = form_class(num_lines=5)
                elif name == "PrecisionHaiku":
                    all_forms[name] = form_class(chars_per_line=17)
                elif name == "ArithmeticVerse":
                    all_forms[name] = form_class(start=2, diff=2, num_lines=5)
                elif name == "PositionalPoem":
                    all_forms[name] = form_class(positions=[1, 2, 3])
                elif name == "IsolatedCouplet":
                    all_forms[name] = form_class(position=3)
                elif name == "AlternatingIsolation":
                    all_forms[name] = form_class(num_lines=6)
                elif name == "DoubleAcrosticPoem":
                    all_forms[name] = form_class(word="POETRY")
                elif name == "CombinedChallenge":
                    all_forms[name] = form_class(num_lines=4)
                elif name == "Lipogram":
                    all_forms[name] = form_class(forbidden="e")
                elif name == "Univocalic":
                    all_forms[name] = form_class(vowel="a")
                elif name == "Mesostic":
                    all_forms[name] = form_class(spine="POEM")
                elif name == "Anaphora":
                    all_forms[name] = form_class(phrase="I am", num_lines=4)
                elif name == "ModularVerse":
                    all_forms[name] = form_class(modulus=3, num_lines=6)
                elif name == "CoprimeVerse":
                    all_forms[name] = form_class(base=6, num_lines=4)
                elif name == "SquareStanzas":
                    all_forms[name] = form_class(size=4)
                elif name == "SelfReferential":
                    all_forms[name] = form_class(num_lines=4)
                elif name == "GoldenRatioVerse":
                    all_forms[name] = form_class(num_lines=6)
                elif name == "PythagoreanTercet":
                    all_forms[name] = form_class(scale=2)
                else:
                    # Skip forms we can't instantiate
                    continue
        except Exception:
            continue

    return all_forms


# =============================================================================
# PROMPT GENERATION
# =============================================================================


def generate_prompt(
    form_name: str,
    form_instance: object,
    topic: str,
    tone: str | None = None,
    perspective: str | None = None,
    temporal: str | None = None,
    sensory: str | None = None,
    approach: str | None = None,
) -> str:
    """Generate a single training prompt with optional style dimensions."""
    description = form_instance.describe()

    # Build the prompt with selected style dimensions
    parts = [f"Write a {form_name} poem about {topic}"]

    style_parts = []
    if tone:
        style_parts.append(f"in a {tone} tone")
    if perspective:
        style_parts.append(perspective)
    if temporal:
        style_parts.append(temporal)
    if sensory:
        style_parts.append(sensory)
    if approach:
        style_parts.append(approach)

    if style_parts:
        parts[0] += ", " + ", ".join(style_parts)

    parts[0] += "."
    parts.append(f"\nRequirements: {description}")
    parts.append("\nOutput ONLY the poem, nothing else.")

    return " ".join(parts)


def generate_unique_prompt_key(
    form_name: str,
    topic: str,
    tone: str | None,
    perspective: str | None,
    temporal: str | None,
    sensory: str | None,
    approach: str | None,
) -> str:
    """Generate a unique key for deduplication."""
    key = f"{form_name}|{topic}|{tone}|{perspective}|{temporal}|{sensory}|{approach}"
    return hashlib.md5(key.encode()).hexdigest()


def generate_dataset(
    num_prompts: int = 100000,
    seed: int = 42,
) -> list[dict]:
    """Generate a large dataset of diverse, unique prompts."""
    rng = random.Random(seed)

    forms = get_forms()
    form_names = list(forms.keys())
    all_topics = get_all_topics(seed)

    print(f"  Forms: {len(form_names)}")
    print(f"  Topics: {len(all_topics)}")
    print(f"  Tones: {len(TONES)}")
    print(f"  Perspectives: {len(PERSPECTIVES)}")
    print(f"  Temporal frames: {len(TEMPORAL_FRAMES)}")
    print(f"  Sensory focuses: {len(SENSORY_FOCUSES)}")
    print(f"  Approaches: {len(APPROACHES)}")

    # Calculate theoretical max unique combinations
    # Each prompt uses: form + topic + (0-2 style dimensions)
    # We use probabilistic selection to ensure variety
    max_theoretical = len(form_names) * len(all_topics) * (len(TONES) + 1) * (len(PERSPECTIVES) + 1)
    print(f"  Theoretical max (formxtopicxtonexperspective): {max_theoretical:,}")

    dataset = []
    seen_keys = set()

    # Ensure balanced form distribution
    prompts_per_form = num_prompts // len(form_names)
    remainder = num_prompts % len(form_names)

    for form_idx, form_name in enumerate(form_names):
        form_instance = forms[form_name]
        count = prompts_per_form + (1 if form_idx < remainder else 0)

        generated = 0
        attempts = 0
        max_attempts = count * 10  # Prevent infinite loops

        while generated < count and attempts < max_attempts:
            attempts += 1

            # Select topic
            topic = rng.choice(all_topics)

            # Probabilistically select style dimensions (0-3 active)
            # This creates variety in prompt complexity
            num_styles = rng.choices([0, 1, 2, 3], weights=[0.1, 0.3, 0.4, 0.2])[0]

            tone = rng.choice(TONES) if num_styles >= 1 and rng.random() < 0.7 else None
            perspective = (
                rng.choice(PERSPECTIVES) if num_styles >= 2 and rng.random() < 0.5 else None
            )
            temporal = (
                rng.choice(TEMPORAL_FRAMES) if num_styles >= 2 and rng.random() < 0.3 else None
            )
            sensory = (
                rng.choice(SENSORY_FOCUSES) if num_styles >= 3 and rng.random() < 0.4 else None
            )
            approach = rng.choice(APPROACHES) if num_styles >= 3 and rng.random() < 0.3 else None

            # Check uniqueness
            key = generate_unique_prompt_key(
                form_name, topic, tone, perspective, temporal, sensory, approach
            )

            if key in seen_keys:
                continue

            seen_keys.add(key)

            prompt = generate_prompt(
                form_name=form_name,
                form_instance=form_instance,
                topic=topic,
                tone=tone,
                perspective=perspective,
                temporal=temporal,
                sensory=sensory,
                approach=approach,
            )

            dataset.append(
                {
                    "prompt": [{"role": "user", "content": prompt}],
                    "info": {
                        "form_name": form_name,
                        "topic": topic,
                        "tone": tone,
                        "perspective": perspective,
                    },
                }
            )

            generated += 1

        if generated < count:
            print(f"  Warning: Only generated {generated}/{count} for {form_name}")

    # Shuffle the dataset
    rng.shuffle(dataset)

    return dataset


def generate_verifiers_dataset(num_prompts: int = 100000, seed: int = 42):
    """Generate dataset in verifiers-compatible format."""
    from datasets import Dataset

    data = generate_dataset(num_prompts=num_prompts, seed=seed)
    return Dataset.from_list(data)


def get_traditional_forms() -> dict[str, object]:
    """Load only traditional forms (no constraint/mathematical forms)."""
    all_forms = get_forms()
    return {name: form for name, form in all_forms.items() if name in TRADITIONAL_FORMS}


def generate_traditional_dataset(
    num_prompts: int = 50000,
    seed: int = 42,
) -> list[dict]:
    """Generate dataset with weighted sampling of traditional forms.

    - Tier 1 (classic forms): 5x weight
    - Tier 2 (well-known): 3x weight
    - Tier 3 (less common): 1x weight
    """
    rng = random.Random(seed)

    forms = get_traditional_forms()
    all_topics = get_all_topics(seed)

    # Build weighted form list
    weighted_forms = []
    for name in forms:
        tier = get_form_tier(name)
        if tier > 0:
            weight = TIER_WEIGHTS[tier]
            weighted_forms.extend([name] * weight)

    print(f"  Traditional forms: {len(forms)}")
    print(f"  Tier 1 (weight 5): {len([f for f in forms if get_form_tier(f) == 1])}")
    print(f"  Tier 2 (weight 3): {len([f for f in forms if get_form_tier(f) == 2])}")
    print(f"  Tier 3 (weight 1): {len([f for f in forms if get_form_tier(f) == 3])}")
    print(f"  Topics: {len(all_topics)}")
    print(f"  Tones: {len(TONES)}")

    dataset = []
    seen_keys = set()
    form_counts = {}

    attempts = 0
    max_attempts = num_prompts * 3

    while len(dataset) < num_prompts and attempts < max_attempts:
        attempts += 1

        # Weighted form selection
        form_name = rng.choice(weighted_forms)
        form_instance = forms[form_name]

        # Select topic
        topic = rng.choice(all_topics)

        # Simpler style selection for traditional training
        tone = rng.choice(TONES) if rng.random() < 0.7 else None
        perspective = rng.choice(PERSPECTIVES) if rng.random() < 0.3 else None

        # Check uniqueness
        key = generate_unique_prompt_key(form_name, topic, tone, perspective, None, None, None)

        if key in seen_keys:
            continue

        seen_keys.add(key)

        prompt = generate_prompt(
            form_name=form_name,
            form_instance=form_instance,
            topic=topic,
            tone=tone,
            perspective=perspective,
        )

        dataset.append(
            {
                "prompt": [{"role": "user", "content": prompt}],
                "info": {
                    "form_name": form_name,
                    "topic": topic,
                    "tone": tone,
                    "perspective": perspective,
                },
            }
        )

        form_counts[form_name] = form_counts.get(form_name, 0) + 1

    # Log distribution
    tier_counts = {1: 0, 2: 0, 3: 0}
    for name, count in form_counts.items():
        tier = get_form_tier(name)
        tier_counts[tier] += count

    print(f"\nForm distribution ({len(dataset)} prompts):")
    for tier in [1, 2, 3]:
        pct = 100 * tier_counts[tier] / len(dataset)
        print(f"  Tier {tier}: {tier_counts[tier]} ({pct:.1f}%)")

    # Shuffle the dataset
    rng.shuffle(dataset)

    return dataset


def generate_traditional_verifiers_dataset(num_prompts: int = 50000, seed: int = 42):
    """Generate traditional dataset in verifiers-compatible format."""
    from datasets import Dataset

    data = generate_traditional_dataset(num_prompts=num_prompts, seed=seed)
    return Dataset.from_list(data)


def main():
    """Generate and save a training dataset."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate GRPO training prompts")
    parser.add_argument("--num", type=int, default=100000, help="Number of prompts")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="data/grpo_prompts.jsonl")
    parser.add_argument("--stats", action="store_true", help="Print statistics only")
    args = parser.parse_args()

    if args.stats:
        forms = get_forms()
        all_topics = get_all_topics(args.seed)
        print(f"Forms: {len(forms)}")
        print(f"Base topics: {len(ALL_BASE_TOPICS)}")
        print(f"Combined topics: {len(all_topics) - len(ALL_BASE_TOPICS)}")
        print(f"Total topics: {len(all_topics)}")
        print(f"Tones: {len(TONES)}")
        print(f"Perspectives: {len(PERSPECTIVES)}")
        print(f"Temporal frames: {len(TEMPORAL_FRAMES)}")
        print(f"Sensory focuses: {len(SENSORY_FOCUSES)}")
        print(f"Approaches: {len(APPROACHES)}")

        # Conservative estimate (form x topic x tone x perspective)
        conservative = len(forms) * len(all_topics) * len(TONES) * len(PERSPECTIVES)
        print(f"\nConservative unique combinations: {conservative:,}")
        return

    print(f"Generating {args.num:,} prompts with seed {args.seed}...")
    dataset = generate_dataset(num_prompts=args.num, seed=args.seed)

    # Count forms
    form_counts = {}
    for item in dataset:
        fn = item["info"]["form_name"]
        form_counts[fn] = form_counts.get(fn, 0) + 1

    print(f"\nGenerated {len(dataset):,} unique prompts")
    print(f"Forms represented: {len(form_counts)}")

    # Show distribution
    counts = list(form_counts.values())
    print(
        f"Prompts per form: min={min(counts)}, max={max(counts)}, avg={sum(counts)/len(counts):.1f}"
    )

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")

    print(f"\nSaved {len(dataset):,} prompts to {output_path}")


if __name__ == "__main__":
    main()
