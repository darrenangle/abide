from __future__ import annotations

import contextlib
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class _FormStats:
    count: int = 0
    score_sum: float = 0.0
    pass_count: int = 0
    zero_count: int = 0
    failure_reasons: Counter[str] = field(default_factory=Counter)

    def record(self, reward: float, *, passed: bool, failure_reason: str | None) -> None:
        self.count += 1
        self.score_sum += reward
        self.pass_count += int(passed)
        self.zero_count += int(reward == 0.0)
        if failure_reason:
            self.failure_reasons[failure_reason] += 1

    def serialize(self, *, max_failure_reasons: int) -> dict[str, Any]:
        mean_reward = self.score_sum / self.count if self.count else 0.0
        pass_rate = self.pass_count / self.count if self.count else 0.0
        zero_rate = self.zero_count / self.count if self.count else 0.0
        return {
            "count": self.count,
            "mean_reward": mean_reward,
            "pass_rate": pass_rate,
            "zero_rate": zero_rate,
            "failure_reasons": self.failure_reasons.most_common(max_failure_reasons),
        }


class RewardTelemetry:
    """Aggregate and emit reward telemetry during training."""

    def __init__(
        self,
        *,
        label: str,
        emit_every: int = 256,
        use_wandb: bool = True,
        max_logged_forms: int = 12,
        max_failure_reasons: int = 2,
        log_fn: Callable[[dict[str, float]], None] | None = None,
    ) -> None:
        if emit_every <= 0:
            raise ValueError("emit_every must be positive")
        self.label = label
        self.emit_every = emit_every
        self.use_wandb = use_wandb
        self.max_logged_forms = max_logged_forms
        self.max_failure_reasons = max_failure_reasons
        self.log_fn = log_fn
        self._samples_seen = 0
        self._window_total = _FormStats()
        self._window_forms: dict[str, _FormStats] = {}

    def record(
        self,
        form_name: str | None,
        *,
        reward: float,
        passed: bool,
        failure_reason: str | None = None,
    ) -> None:
        normalized_form_name = form_name or "(unknown)"
        self._samples_seen += 1
        self._window_total.record(reward, passed=passed, failure_reason=failure_reason)
        form_stats = self._window_forms.setdefault(normalized_form_name, _FormStats())
        form_stats.record(reward, passed=passed, failure_reason=failure_reason)

    def snapshot(self) -> dict[str, Any]:
        total = self._window_total.serialize(max_failure_reasons=self.max_failure_reasons)
        forms = {
            name: stats.serialize(max_failure_reasons=self.max_failure_reasons)
            for name, stats in sorted(
                self._window_forms.items(),
                key=lambda item: (-item[1].count, item[0]),
            )
        }
        return {
            "label": self.label,
            "samples_seen": self._samples_seen,
            "window_count": total["count"],
            "mean_reward": total["mean_reward"],
            "pass_rate": total["pass_rate"],
            "zero_rate": total["zero_rate"],
            "failure_reasons": total["failure_reasons"],
            "forms": forms,
        }

    def emit(self, *, force: bool = False) -> dict[str, Any] | None:
        if self._window_total.count == 0:
            return None
        if not force and self._window_total.count < self.emit_every:
            return None

        snapshot = self.snapshot()
        self._print_snapshot(snapshot)
        self._log_snapshot(snapshot)
        self._reset_window()
        return snapshot

    def _reset_window(self) -> None:
        self._window_total = _FormStats()
        self._window_forms = {}

    def _print_snapshot(self, snapshot: dict[str, Any]) -> None:
        print(
            "[reward telemetry]"
            f" {self.label}: n={snapshot['window_count']}"
            f" mean={snapshot['mean_reward']:.3f}"
            f" pass={snapshot['pass_rate']:.1%}"
            f" zero={snapshot['zero_rate']:.1%}",
            flush=True,
        )
        if snapshot["failure_reasons"]:
            failure_summary = "; ".join(
                f"{reason} x{count}" for reason, count in snapshot["failure_reasons"]
            )
            print(f"  failures: {failure_summary}", flush=True)

        for form_name, stats in list(snapshot["forms"].items())[: self.max_logged_forms]:
            failure_summary = ""
            reasons = stats["failure_reasons"]
            if reasons:
                failure_summary = " fail=" + "; ".join(
                    f"{reason} x{count}" for reason, count in reasons
                )
            print(
                f"  {form_name}: n={stats['count']}"
                f" mean={stats['mean_reward']:.3f}"
                f" pass={stats['pass_rate']:.1%}"
                f" zero={stats['zero_rate']:.1%}"
                f"{failure_summary}",
                flush=True,
            )

    def _log_snapshot(self, snapshot: dict[str, Any]) -> None:
        metrics: dict[str, float] = {
            f"reward_telemetry/{self.label}/window_count": float(snapshot["window_count"]),
            f"reward_telemetry/{self.label}/mean_reward": float(snapshot["mean_reward"]),
            f"reward_telemetry/{self.label}/pass_rate": float(snapshot["pass_rate"]),
            f"reward_telemetry/{self.label}/zero_rate": float(snapshot["zero_rate"]),
        }
        for reason, count in snapshot["failure_reasons"]:
            key = _sanitize_metric_key(reason)
            metrics[f"reward_telemetry/{self.label}/failures/{key}/count"] = float(count)
        for form_name, stats in list(snapshot["forms"].items())[: self.max_logged_forms]:
            key = _sanitize_metric_key(form_name)
            metrics[f"reward_telemetry/{self.label}/forms/{key}/count"] = float(stats["count"])
            metrics[f"reward_telemetry/{self.label}/forms/{key}/mean_reward"] = float(
                stats["mean_reward"]
            )
            metrics[f"reward_telemetry/{self.label}/forms/{key}/pass_rate"] = float(
                stats["pass_rate"]
            )
            metrics[f"reward_telemetry/{self.label}/forms/{key}/zero_rate"] = float(
                stats["zero_rate"]
            )

        if self.log_fn is not None:
            self.log_fn(metrics)
            return

        if not self.use_wandb:
            return

        with contextlib.suppress(Exception):
            import wandb

            if getattr(wandb, "run", None) is not None:
                wandb.log(metrics)


def extract_form_names_from_metadata(
    metadata: dict[str, Any],
    expected_count: int,
) -> list[str | None]:
    """Extract exact form names from trainer metadata."""

    def normalize_name(value: Any) -> str | None:
        if isinstance(value, str) and value:
            return value
        if isinstance(value, dict):
            form_name = value.get("form_name")
            if isinstance(form_name, str) and form_name:
                return form_name
        return None

    for key in ("form_name", "info"):
        values = metadata.get(key)
        if isinstance(values, list):
            names = [normalize_name(value) for value in values[:expected_count]]
            if len(names) < expected_count:
                names.extend([None] * (expected_count - len(names)))
            return names
        normalized = normalize_name(values)
        if normalized is not None:
            return [normalized] + [None] * (expected_count - 1)

    return [None] * expected_count


def extract_failure_reason(result: Any) -> str | None:
    """Best-effort failure reason extraction from a verification result."""
    if getattr(result, "passed", False):
        return None

    details = getattr(result, "details", None)
    if isinstance(details, dict):
        error = details.get("error")
        if isinstance(error, str) and error:
            return error

        issues = details.get("issues")
        if isinstance(issues, list):
            for issue in issues:
                if isinstance(issue, str) and issue:
                    return issue

        for key in ("word_bound_violations", "line_details"):
            value = details.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and item and "✓" not in item:
                        return item

        if details.get("canonical_requirements_passed") is False:
            return "canonical requirements failed"

    rubric = getattr(result, "rubric", None)
    if isinstance(rubric, list):
        for item in rubric:
            if getattr(item, "passed", True) is False:
                criterion = getattr(item, "criterion", None)
                if isinstance(criterion, str) and criterion:
                    return criterion

    return "verification failed"


def bind_reward_telemetry(
    func: Callable[..., Any], telemetry: RewardTelemetry
) -> Callable[..., Any]:
    """Attach telemetry to a reward function for later flushing."""
    func._reward_telemetry = telemetry  # type: ignore[attr-defined]
    return func


def flush_reward_telemetry(func: Any) -> None:
    """Flush any attached telemetry on a reward function."""
    telemetry = getattr(func, "_reward_telemetry", None)
    if isinstance(telemetry, RewardTelemetry):
        telemetry.emit(force=True)


def _sanitize_metric_key(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_") or "unknown"
