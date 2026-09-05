"""Tests for the daily Telegram upstream report renderer."""

from __future__ import annotations

from pathlib import Path

import scripts.hermes_upstream_report as report


def test_added_symbols_extracts_new_python_functions_and_classes(monkeypatch):
    diff = (
        "diff --git a/hermes_cli/new_feature.py b/hermes_cli/new_feature.py\n"
        "+++ b/hermes_cli/new_feature.py\n"
        "@@ -1,0 +2,8 @@\n"
        "+def first_feature(value):\n"
        "+    return value\n"
        "+\n"
        "+class FeatureRegistry:\n"
        "+    pass\n"
        "+async def second_feature():\n"
        "+    return None\n"
    )
    monkeypatch.setattr(report, "_git", lambda *_args: diff)

    assert report._added_symbols(Path("/repo"), "a" * 40, "b" * 40) == [
        "hermes_cli/new_feature.py: function first_feature",
        "hermes_cli/new_feature.py: class FeatureRegistry",
        "hermes_cli/new_feature.py: function second_feature",
    ]


def test_diff_stats_handles_binary_files(monkeypatch):
    monkeypatch.setattr(
        report,
        "_git",
        lambda *_args: "10\t2\tsrc/new.py\n-\t-\tassets/logo.png\n",
    )

    assert report._diff_stats(Path("/repo"), "a" * 40, "b" * 40) == (
        2,
        10,
        2,
        ["src/new.py", "assets/logo.png"],
    )


def test_render_report_contains_update_action_and_summary(monkeypatch):
    monkeypatch.setattr(report, "_diff_stats", lambda *_args: (2, 20, 4, ["src/new.py", "README.md"]))
    monkeypatch.setattr(report, "_added_symbols", lambda *_args: ["src/new.py: function new_feature"])
    monkeypatch.setattr(report, "_upstream_commits", lambda *_args: ["abc1234 add new feature"])

    rendered = report.render_report(
        Path("/repo"),
        {
            "run_id": "20260905-123422",
            "upstream_sha": "a" * 40,
            "source_sha": "b" * 40,
            "candidate_sha": "c" * 40,
            "checks": {"preflight_ok": True, "rebase_ok": True, "tests_ok": True},
        },
        checked_at="2026-09-05 20:00 CST",
    )

    assert "🔔 Hermes upstream 有新更新" in rendered
    assert "變更規模：2 files，+20/-4" in rendered
    assert "src/new.py: function new_feature" in rendered
    assert "abc1234 add new feature" in rendered
    assert "核准套用 upstream 更新 20260905-123422" in rendered


def test_render_noop_does_not_request_apply():
    rendered = report.render_report(
        Path("/repo"),
        {"upstream_sha": "a" * 40, "checks": {"noop": True}},
        checked_at="2026-09-05 20:00 CST",
    )

    assert "沒有新的 upstream 更新" in rendered
    assert "核准套用" not in rendered
