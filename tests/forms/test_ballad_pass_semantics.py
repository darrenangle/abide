from abide.forms.ballad import LiteraryBallad


def test_literary_ballad_accepts_valid_multi_stanza_example() -> None:
    poem = "\n\n".join(
        [
            "\n".join(
                [
                    "First long line sings the weather",
                    "Short line answers after",
                    "Third long line carries farther",
                    "Short line closes after",
                ]
            ),
            "\n".join(
                [
                    "Next long line moves the river",
                    "Brief line trails forever",
                    "Third long line turns the silver",
                    "Brief line ends forever",
                ]
            ),
            "\n".join(
                [
                    "Last long line tells the harbor",
                    "Soft line falls to amber",
                    "Third long line dims the lantern",
                    "Soft line leaves the amber",
                ]
            ),
        ]
    )

    result = LiteraryBallad().verify(poem)

    assert result.passed is True


def test_literary_ballad_rejects_single_stanza_false_positive() -> None:
    poem = "\n".join(["the same line again"] * 14)

    result = LiteraryBallad().verify(poem)

    assert result.score >= 0.6
    assert result.passed is False
