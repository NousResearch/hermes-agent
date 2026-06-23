"""Integration tests for the self_evolution tool wrapper.

Tests every action through the full tool → backend path,
using a temporary ledger for isolation.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

from tools.self_evolution_tool import self_evolution


def _call(action: str, **kwargs) -> dict:
    """Call the tool handler and parse the JSON result."""
    return json.loads(self_evolution({"action": action, **kwargs}))


# ── record ────────────────────────────────────────────────────────────

def test_record_creates_lesson(tmp_path: Path) -> None:
    ledger = tmp_path / "lessons.jsonl"
    with mock.patch("agent.self_evolution.default_ledger_path", return_value=ledger):
        result = _call("record",
                       mistake="Skipping tests before merge",
                       lesson="Always run tests before merging",
                       trigger="git merge or push",
                       fix="pytest tests/ -x",
                       tags=["tests", "git"],
                       evidence=["CI failure on main"],
                       severity="high",
                       confidence=0.9,
                       source="test_failure")

    assert result["success"] is True
    assert result["action"] == "created"
    lesson = result["lesson"]
    assert lesson["mistake"] == "Skipping tests before merge"
    assert lesson["lesson"] == "Always run tests before merging"
    assert lesson["trigger"] == "git merge or push"
    assert lesson["fix"] == "pytest tests/ -x"
    assert set(lesson["tags"]) == {"tests", "git"}
    assert lesson["severity"] == "high"
    assert lesson["confidence"] == 0.9
    assert lesson["source"] == "test_failure"
    assert lesson["occurrences"] == 1
    assert lesson["status"] == "active"
    assert "id" in lesson
    assert len(lesson["id"]) == 16  # fingerprint hash


def test_record_deduplicates_by_content(tmp_path: Path) -> None:
    ledger = tmp_path / "lessons.jsonl"
    with mock.patch("agent.self_evolution.default_ledger_path", return_value=ledger):
        first = _call("record",
                      mistake="Same mistake twice",
                      lesson="Same lesson twice",
                      tags=["a"])
        second = _call("record",
                       mistake="Same mistake twice",
                       lesson="Same lesson twice",
                       tags=["b"])

    assert first["action"] == "created"
    assert second["action"] == "updated"
    assert second["lesson"]["occurrences"] == 2
    assert first["lesson"]["id"] == second["lesson"]["id"]


def test_record_rejects_empty_inputs(tmp_path: Path) -> None:
    ledger = tmp_path / "lessons.jsonl"
    with mock.patch("agent.self_evolution.default_ledger_path", return_value=ledger):
        result = _call("record", mistake="", lesson="")

    assert result["success"] is False
    assert "required" in result.get("error", "")


# ── recall ────────────────────────────────────────────────────────────

def test_recall_finds_recorded_lesson(tmp_path: Path) -> None:
    ledger = tmp_path / "lessons.jsonl"
    with mock.patch("agent.self_evolution.default_ledger_path", return_value=ledger):
        _call("record",
              mistake="Deploying on Friday evening",
              lesson="Never deploy on Friday",
              tags=["deploy", "schedule"])

        result = _call("recall", query="deploy friday schedule")

    assert result["success"] is True
    assert result["count"] >= 1
    assert result["lessons"][0]["mistake"] == "Deploying on Friday evening"


def test_recall_respects_limit(tmp_path: Path) -> None:
    ledger = tmp_path / "lessons.jsonl"
    with mock.patch("agent.self_evolution.default_ledger_path", return_value=ledger):
        for i in range(5):
            _call("record",
                  mistake=f"Mistake {i} with searchable keyword",
                  lesson=f"Lesson {i}")

        result = _call("recall", query="searchable", limit=2)

    assert result["count"] == 2


def test_recall_excludes_resolved_by_default(tmp_path: Path) -> None:
    ledger = tmp_path / "lessons.jsonl"
    with mock.patch("agent.self_evolution.default_ledger_path", return_value=ledger):
        created = _call("record",
                        mistake="Bug from missing null check",
                        lesson="Always null-check inputs")
        _call("resolve",
              lesson_id=created["lesson"]["id"],
              outcome="Fixed in v2")

        result = _call("recall", query="null check")

    assert result["count"] == 0


# ── list ───────────────────────────────────────────────────────────────

def test_list_returns_active_lessons(tmp_path: Path) -> None:
    ledger = tmp_path / "lessons.jsonl"
    with mock.patch("agent.self_evolution.default_ledger_path", return_value=ledger):
        _call("record", mistake="A", lesson="Fix A")
        _call("record", mistake="B", lesson="Fix B")
        _call("record", mistake="C", lesson="Fix C")

        result = _call("list")

    assert result["success"] is True
    assert result["count"] >= 3  # may include pre-existing in default ledger


# ── resolve ────────────────────────────────────────────────────────────

def test_resolve_retires_lesson(tmp_path: Path) -> None:
    ledger = tmp_path / "lessons.jsonl"
    with mock.patch("agent.self_evolution.default_ledger_path", return_value=ledger):
        created = _call("record",
                        mistake="Hardcoding API URLs",
                        lesson="Use config for endpoints")
        lesson_id = created["lesson"]["id"]

        resolved = _call("resolve",
                         lesson_id=lesson_id,
                         outcome="Migrated to config system")

    assert resolved["success"] is True

    # Verify it no longer appears in recall
    recall = _call("recall", query="hardcoding api")
    ids = [l["id"] for l in recall.get("lessons", [])]
    assert lesson_id not in ids


def test_resolve_rejects_empty_id(tmp_path: Path) -> None:
    ledger = tmp_path / "lessons.jsonl"
    with mock.patch("agent.self_evolution.default_ledger_path", return_value=ledger):
        result = _call("resolve", lesson_id="", outcome="done")

    assert result["success"] is False


# ── export_context ─────────────────────────────────────────────────────

def test_export_context_produces_formatted_block(tmp_path: Path) -> None:
    ledger = tmp_path / "lessons.jsonl"
    with mock.patch("agent.self_evolution.default_ledger_path", return_value=ledger):
        _call("record",
              mistake="Not verifying generated HTML output",
              lesson="Open browser and scroll through every page",
              tags=["frontend", "verification"])

        result = _call("export_context", query="frontend HTML verification")

    assert result["success"] is True
    context = result.get("context", "")
    assert "Avoid:" in context or "prior mistakes" in context.lower()
    assert "frontend" in context.lower() or "HTML" in context or "verification" in context.lower()


# ── error handling ─────────────────────────────────────────────────────

def test_unknown_action_returns_error() -> None:
    result = _call("wat")

    assert result["success"] is False
    assert "unknown action" in result.get("error", "")


def test_missing_action_returns_error() -> None:
    result = json.loads(self_evolution({}))

    assert result["success"] is False
