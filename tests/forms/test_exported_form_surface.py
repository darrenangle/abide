from abide.forms.catalog import load_form_instances

CASES = {
    "empty": "",
    "plain_four": "alpha\nbravo\ncharlie\ndelta",
    "two_stanzas": "alpha\nbravo\n\ncharlie\ndelta",
    "repeated_ten": "\n".join(["same line again"] * 10),
}


def test_exported_forms_describe_and_verify_without_runtime_exceptions() -> None:
    forms = load_form_instances()

    for name, form in forms.items():
        description = form.describe()
        assert description.strip(), f"{name} should provide a non-empty description"

        for case_name, poem in CASES.items():
            result = form.verify(poem)
            assert 0.0 <= result.score <= 1.0, f"{name} returned invalid score for {case_name}"
            assert isinstance(result.passed, bool), (
                f"{name} returned non-bool passed flag for {case_name}"
            )
            assert result.constraint_name, f"{name} returned empty constraint_name for {case_name}"
