"""Tests for the load-time freshness re-check reminder on skill_view."""

import json
from datetime import date
from unittest.mock import patch

from tools.skills_tool import _FRESHNESS_REVIEW_DAYS, _freshness_note, skill_view

TODAY = date(2026, 11, 13)


def _make_skill(root, name, extra_frontmatter, body):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        f"description: Freshness fixture for {name}.\n"
        f"{extra_frontmatter}"
        "---\n\n"
        f"# {name}\n\n"
        f"{body}\n",
        encoding="utf-8",
    )


def test_old_verification_produces_a_dated_note():
    note = _freshness_note({"volatile": True, "verified_at": "2026-01-01"}, today=TODAY)
    age = (TODAY - date(2026, 1, 1)).days
    assert note is not None
    assert "2026-01-01" in note and str(age) in note


def test_recent_verification_has_no_note():
    fresh = TODAY.fromordinal(TODAY.toordinal() - _FRESHNESS_REVIEW_DAYS + 1)
    assert _freshness_note(
        {"volatile": True, "verified_at": fresh.isoformat()}, today=TODAY
    ) is None


def test_boundary_at_the_review_threshold_produces_a_note():
    boundary = TODAY.fromordinal(TODAY.toordinal() - _FRESHNESS_REVIEW_DAYS)
    assert _freshness_note(
        {"volatile": True, "verified_at": boundary.isoformat()}, today=TODAY
    ) is not None


def test_unmarked_or_incomplete_markers_have_no_note():
    cases = [
        {},
        {"volatile": True},
        {"verified_at": "2020-01-01"},
        {"volatile": False, "verified_at": "2020-01-01"},
        {"volatile": True, "verified_at": "not-a-date"},
        {"volatile": True, "verified_at": None},
    ]
    for fm in cases:
        assert _freshness_note(fm, today=TODAY) is None


def test_date_object_from_unquoted_yaml_is_accepted():
    """Unquoted YAML dates parse as datetime.date; the reminder accepts them too."""
    note = _freshness_note({"volatile": True, "verified_at": date(2020, 1, 1)}, today=TODAY)
    assert note is not None and "2020-01-01" in note


def test_skill_view_adds_a_structured_note_and_keeps_the_body(tmp_path):
    body = "Step 1: confirm the installed version before writing config.\n"
    _make_skill(tmp_path, "marked", "volatile: true\nverified_at: '2026-01-01'\n", body)
    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        result = json.loads(skill_view("marked", preprocess=False))
    assert result["success"] is True
    assert "freshness_note" in result
    assert "2026-01-01" in result["freshness_note"]
    assert "Step 1: confirm the installed version" in result["content"]
    assert "freshness_note" not in result["content"]


def test_unmarked_skill_view_has_no_note(tmp_path):
    _make_skill(tmp_path, "plain", "", "Step 1: do the thing.\n")
    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        result = json.loads(skill_view("plain", preprocess=False))
    assert result["success"] is True
    assert "freshness_note" not in result



def test_yaml_timestamp_is_normalized_to_a_date():
    """A YAML timestamp parses as datetime; the reminder reads its date part."""
    from datetime import datetime
    note = _freshness_note(
        {"volatile": True, "verified_at": datetime(2020, 1, 1, 10, 30)}, today=TODAY
    )
    assert note is not None and "2020-01-01" in note


def test_only_boolean_true_enables_the_marker():
    cases = [
        {"volatile": "false", "verified_at": "2020-01-01"},
        {"volatile": 1, "verified_at": "2020-01-01"},
        {"volatile": "true", "verified_at": "2020-01-01"},
    ]
    for fm in cases:
        assert _freshness_note(fm, today=TODAY) is None
