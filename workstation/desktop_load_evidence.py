"""Validate the versioned H013 Desktop/Browser load evidence report.

The Playwright harness owns execution and writes the report only after native
task cleanup. This module is the read-only release-side contract: it validates
that a candidate run actually exercised the configured scale and reached its
round or duration boundary without duplicating browser/runtime state.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class DesktopLoadRequirements:
    """Minimum H013 profile required by a release gate."""

    min_task_count: int = 16
    min_round_count: int = 300
    min_duration_ms: int = 120_000
    min_chat_turns: int = 8


@dataclass(frozen=True, slots=True)
class DesktopLoadValidation:
    """Bounded validation result suitable for logs and CI summaries."""

    accepted: bool
    reason: str
    task_count: int | None = None
    requested_rounds: int | None = None
    completed_rounds: int | None = None
    requested_duration_ms: int | None = None
    observed_duration_ms: int | None = None
    chat_turns: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_desktop_load_report(
    report: Mapping[str, Any],
    requirements: DesktopLoadRequirements = DesktopLoadRequirements(),
) -> DesktopLoadValidation:
    """Validate one H013 report without reading or mutating any external state."""

    if report.get("accepted") is not True:
        return DesktopLoadValidation(False, "report is not accepted")

    values: dict[str, int] = {}
    for field in (
        "task_count",
        "requested_rounds",
        "completed_rounds",
        "requested_duration_ms",
        "observed_duration_ms",
        "chat_turns",
    ):
        value = report.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            return DesktopLoadValidation(False, f"report field {field} must be an integer")
        values[field] = value

    for field in ("task_count", "requested_rounds", "chat_turns"):
        if values[field] <= 0:
            return DesktopLoadValidation(False, f"{field} must be positive", **values)
    for field in ("requested_duration_ms", "observed_duration_ms"):
        if values[field] < 0:
            return DesktopLoadValidation(False, f"{field} must be non-negative", **values)
    if values["task_count"] < requirements.min_task_count:
        return DesktopLoadValidation(False, f"task_count is below {requirements.min_task_count}", **values)
    if values["requested_rounds"] < requirements.min_round_count:
        return DesktopLoadValidation(False, f"requested_rounds is below {requirements.min_round_count}", **values)
    if values["requested_duration_ms"] < requirements.min_duration_ms:
        return DesktopLoadValidation(False, f"requested_duration_ms is below {requirements.min_duration_ms}", **values)
    if values["chat_turns"] < requirements.min_chat_turns:
        return DesktopLoadValidation(False, f"chat_turns is below {requirements.min_chat_turns}", **values)
    if values["completed_rounds"] <= 0:
        return DesktopLoadValidation(False, "completed_rounds must be positive", **values)

    boundary_reached = (
        values["completed_rounds"] >= values["requested_rounds"]
        or values["observed_duration_ms"] >= values["requested_duration_ms"]
    )
    if not boundary_reached:
        return DesktopLoadValidation(False, "round or duration boundary was not reached", **values)

    return DesktopLoadValidation(True, "H013 profile accepted", **values)


def validate_desktop_load_report_file(
    path: Path,
    requirements: DesktopLoadRequirements = DesktopLoadRequirements(),
) -> DesktopLoadValidation:
    """Read and validate a report file; never write to it."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return DesktopLoadValidation(False, f"could not read H013 report: {exc}")
    if not isinstance(raw, dict):
        return DesktopLoadValidation(False, "H013 report must be a JSON object")
    return validate_desktop_load_report(raw, requirements)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a Hermes H013 Desktop/Browser load report")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--min-tasks", type=int, default=DesktopLoadRequirements.min_task_count)
    parser.add_argument("--min-rounds", type=int, default=DesktopLoadRequirements.min_round_count)
    parser.add_argument("--min-duration-ms", type=int, default=DesktopLoadRequirements.min_duration_ms)
    parser.add_argument("--min-chat-turns", type=int, default=DesktopLoadRequirements.min_chat_turns)
    args = parser.parse_args(argv)

    requirements = DesktopLoadRequirements(
        min_task_count=args.min_tasks,
        min_round_count=args.min_rounds,
        min_duration_ms=args.min_duration_ms,
        min_chat_turns=args.min_chat_turns,
    )
    result = validate_desktop_load_report_file(args.report, requirements)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0 if result.accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
