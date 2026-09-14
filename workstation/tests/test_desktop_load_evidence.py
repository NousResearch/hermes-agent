from __future__ import annotations

import json

import pytest

from workstation.desktop_load_evidence import (
    DesktopLoadRequirements,
    validate_desktop_load_report,
    validate_desktop_load_report_file,
)


def _report(**overrides):
    report = {
        "accepted": True,
        "task_count": 16,
        "requested_rounds": 300,
        "completed_rounds": 150,
        "requested_duration_ms": 120_000,
        "observed_duration_ms": 120_075,
        "chat_turns": 8,
    }
    report.update(overrides)
    return report


def test_desktop_load_report_accepts_configured_profile():
    result = validate_desktop_load_report(_report())

    assert result.accepted is True
    assert result.reason == "H013 profile accepted"
    assert result.completed_rounds == 150


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("task_count", 15, "task_count"),
        ("requested_rounds", 299, "requested_rounds"),
        ("requested_duration_ms", 119_999, "requested_duration_ms"),
        ("chat_turns", 7, "chat_turns"),
        ("completed_rounds", 0, "completed_rounds"),
    ],
)
def test_desktop_load_report_rejects_under_sized_profile(field, value, reason):
    result = validate_desktop_load_report(_report(**{field: value}))

    assert result.accepted is False
    assert reason in result.reason


def test_desktop_load_report_requires_a_round_or_duration_boundary():
    result = validate_desktop_load_report(_report(observed_duration_ms=119_999))

    assert result.accepted is False
    assert "boundary" in result.reason


def test_desktop_load_report_rejects_negative_duration_even_at_round_boundary():
    result = validate_desktop_load_report(_report(completed_rounds=300, observed_duration_ms=-1))

    assert result.accepted is False
    assert "observed_duration_ms" in result.reason


def test_desktop_load_report_accepts_round_boundary_without_duration_elapsed():
    result = validate_desktop_load_report(_report(completed_rounds=300, observed_duration_ms=1))

    assert result.accepted is True


def test_desktop_load_report_rejects_malformed_integer_fields():
    result = validate_desktop_load_report(_report(chat_turns=True))

    assert result.accepted is False
    assert "chat_turns" in result.reason


def test_desktop_load_report_file_is_read_only(tmp_path):
    path = tmp_path / "h013.json"
    original = json.dumps(_report())
    path.write_text(original, encoding="utf-8")

    result = validate_desktop_load_report_file(path)

    assert result.accepted is True
    assert path.read_text(encoding="utf-8") == original


def test_desktop_load_report_accepts_explicit_larger_requirements():
    requirements = DesktopLoadRequirements(
        min_task_count=20,
        min_round_count=100,
        min_duration_ms=60_000,
        min_chat_turns=4,
    )
    result = validate_desktop_load_report(_report(task_count=24, chat_turns=6), requirements)

    assert result.accepted is True
