import pytest

from abide.forms.hard import PositionalPoem


def test_positional_poem_rejects_malformed_positions_early() -> None:
    with pytest.raises(ValueError, match="positions must be a list of"):
        PositionalPoem(positions=[(1, "T"), 2])  # type: ignore[list-item]
