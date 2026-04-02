import pytest

from abide.forms import (
    Cinquain,
    Diamante,
    DoubleFibonacci,
    Etheree,
    FibonacciPoem,
    Katauta,
    ReverseEtheree,
    ReverseFibonacci,
    Sedoka,
    Senryu,
    WordCinquain,
)
from abide.forms.world import Naani


def _mono_line(count: int, word: str = "sun") -> str:
    return " ".join([word] * count)


def _append_trailing_line(poem: str) -> str:
    return poem.rstrip() + "\nspare line"


def _drop_last_nonempty_line(poem: str) -> str:
    lines = poem.splitlines()
    for idx in range(len(lines) - 1, -1, -1):
        if lines[idx].strip():
            del lines[idx]
            break
    return "\n".join(lines)


EXACT_COUNT_CASES = [
    ("senryu", Senryu(strict=False), "\n".join([_mono_line(5), _mono_line(7), _mono_line(5)])),
    ("katauta", Katauta(strict=False), "\n".join([_mono_line(5), _mono_line(7), _mono_line(7)])),
    (
        "sedoka",
        Sedoka(strict=False),
        "\n".join(
            [
                _mono_line(5),
                _mono_line(7),
                _mono_line(7),
                _mono_line(5),
                _mono_line(7),
                _mono_line(7),
            ]
        ),
    ),
    (
        "diamante",
        Diamante(),
        "\n".join(
            [
                "stone",
                "calm bright",
                "drift glow sing",
                "earth wind fire rain",
                "turn move hum",
                "soft warm",
                "ember",
            ]
        ),
    ),
    (
        "cinquain",
        Cinquain(),
        "\n".join([_mono_line(2), _mono_line(4), _mono_line(6), _mono_line(8), _mono_line(2)]),
    ),
    (
        "word-cinquain",
        WordCinquain(),
        "\n".join(
            [
                "stone",
                "calm bright",
                "drift glow sing",
                "earth wind fire rain",
                "ember",
            ]
        ),
    ),
    ("etheree", Etheree(), "\n".join([_mono_line(count) for count in range(1, 11)])),
    (
        "reverse-etheree",
        ReverseEtheree(),
        "\n".join([_mono_line(count) for count in range(10, 0, -1)]),
    ),
    (
        "fibonacci",
        FibonacciPoem(strict=False),
        "\n".join([_mono_line(count) for count in [1, 1, 2, 3, 5, 8]]),
    ),
    (
        "reverse-fibonacci",
        ReverseFibonacci(strict=False),
        "\n".join([_mono_line(count) for count in [8, 5, 3, 2, 1, 1]]),
    ),
    (
        "double-fibonacci",
        DoubleFibonacci(strict=False),
        "\n".join([_mono_line(count) for count in [1, 1, 2, 3, 5, 8, 8, 5, 3, 2, 1, 1]]),
    ),
    ("naani", Naani(), "\n".join([_mono_line(5)] * 4)),
]


@pytest.mark.parametrize(
    ("name", "form", "poem"),
    EXACT_COUNT_CASES,
    ids=[name for name, _, _ in EXACT_COUNT_CASES],
)
def test_exact_structure_mutation_harness_rejects_added_line(name: str, form, poem: str) -> None:
    baseline = form.verify(poem)
    assert baseline.passed is True, f"{name} baseline should pass before mutation"

    mutated = form.verify(_append_trailing_line(poem))

    assert mutated.passed is False


@pytest.mark.parametrize(
    ("name", "form", "poem"),
    EXACT_COUNT_CASES,
    ids=[name for name, _, _ in EXACT_COUNT_CASES],
)
def test_exact_structure_mutation_harness_rejects_missing_line(name: str, form, poem: str) -> None:
    baseline = form.verify(poem)
    assert baseline.passed is True, f"{name} baseline should pass before mutation"

    mutated = form.verify(_drop_last_nonempty_line(poem))

    assert mutated.passed is False
