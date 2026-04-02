from abide.forms.catalog import load_form_instances

DEGENERATE_CASES = {
    "empty": "",
    "one_word": "alpha",
    "one_question": "?",
    "one_exclamation": "!",
    "one_time_word": "today",
    "one_element": "gold",
    "one_long_line": "x" * 60,
    "three_word_line": "one two three",
    "two_lines": "alpha\nbravo",
    "punct_only": "!!!\n???\n...\n!!!\n???",
    "repeated_ten": "\n".join(["same line again"] * 10),
}


def test_exported_forms_do_not_leak_high_failed_scores_on_degenerate_inputs() -> None:
    forms = load_form_instances()

    for name, form in forms.items():
        for case_name, poem in DEGENERATE_CASES.items():
            result = form.verify(poem)
            if result.passed:
                continue
            assert result.score <= 0.75, (
                f"{name} leaked failed score {result.score:.3f} on degenerate case {case_name}"
            )
