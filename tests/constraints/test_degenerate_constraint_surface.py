import inspect

import abide.constraints as constraints

DEGENERATE_CASES = {
    "empty": "",
    "one_word": "alpha",
    "punct_only": "!!!\n???\n...\n!!!\n???",
    "repeated_ten": "\n".join(["same line again"] * 10),
}


def _load_zero_arg_public_constraints() -> dict[str, constraints.Constraint]:
    loaded: dict[str, constraints.Constraint] = {}

    for name in constraints.__all__:
        obj = getattr(constraints, name)
        if not inspect.isclass(obj):
            continue
        if not issubclass(obj, constraints.Constraint):
            continue
        if inspect.isabstract(obj):
            continue

        signature = inspect.signature(obj)
        required = [
            param
            for param in signature.parameters.values()
            if param.default is inspect.Signature.empty
            and param.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        ]
        if required:
            continue

        try:
            loaded[name] = obj()
        except Exception:
            continue

    return loaded


def test_zero_arg_public_constraints_do_not_pass_empty_or_punctuation_only_inputs() -> None:
    for name, constraint in _load_zero_arg_public_constraints().items():
        for case_name in ("empty", "punct_only"):
            result = constraint.verify(DEGENERATE_CASES[case_name])
            assert result.passed is False, f"{name} unexpectedly passed degenerate case {case_name}"


def test_zero_arg_public_constraints_do_not_leak_high_failed_scores() -> None:
    for name, constraint in _load_zero_arg_public_constraints().items():
        for case_name, poem in DEGENERATE_CASES.items():
            result = constraint.verify(poem)
            if result.passed:
                continue
            assert result.score <= 0.5, (
                f"{name} leaked failed score {result.score:.3f} on degenerate case {case_name}"
            )
