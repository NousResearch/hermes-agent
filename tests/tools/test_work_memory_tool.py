"""Tests for profile-scoped structured work memory."""

import json
from pathlib import Path

import pytest

from tools import work_memory_tool
from tools.work_memory_tool import work_memory_handler


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    yield tmp_path


def _result(**kwargs):
    return json.loads(work_memory_handler(**kwargs))


def test_add_structured_decision_persists_profile_scoped_record(isolated_home):
    result = _result(
        action="add",
        kind="decision",
        title="Use phased rollout for Sports launch",
        content="Team agreed to ship Sports launch behind a rollout flag.",
        project="sports_product",
        source_channel="#sports_product",
        source_ts="1710000000.000001",
        confidence="high",
        tags=["launch", "rollout"],
    )

    assert result["ok"] is True
    item = result["item"]
    assert item["kind"] == "decision"
    assert item["project"] == "sports_product"
    assert item["source"]["channel"] == "#sports_product"
    assert item["confidence"] == "high"
    assert item["tags"] == ["launch", "rollout"]

    store_path = isolated_home / "work_memory" / "items.jsonl"
    assert store_path.exists()
    stored = [json.loads(line) for line in store_path.read_text().splitlines()]
    assert len(stored) == 1
    assert stored[0]["id"] == item["id"]


def test_add_same_source_updates_existing_record_not_duplicate():
    first = _result(
        action="add",
        kind="open_loop",
        title="Confirm odds copy owner",
        content="Need a named owner for odds copy approval.",
        project="sports_product",
        source_channel="#sports_product",
        source_ts="1710000000.000002",
        status="open",
    )
    second = _result(
        action="add",
        kind="open_loop",
        title="Confirm odds copy owner",
        content="Sarah is checking who owns odds copy approval.",
        project="sports_product",
        source_channel="#sports_product",
        source_ts="1710000000.000002",
        status="watching",
    )

    assert first["item"]["id"] == second["item"]["id"]
    listed = _result(action="list", kind="open_loop")
    assert listed["count"] == 1
    assert listed["items"][0]["status"] == "watching"
    assert "Sarah" in listed["items"][0]["content"]


def test_query_filters_by_project_kind_status_and_text():
    _result(action="add", kind="risk", title="Feed delay", content="Sports feed delay may slip launch", project="sports_product", status="open", tags=["feed"])
    _result(action="add", kind="risk", title="Casino copy", content="Casino copy has no owner", project="casino", status="open")
    _result(action="add", kind="decision", title="Feed vendor", content="Proceed with vendor A", project="sports_product", status="closed")

    result = _result(action="query", query="feed launch", project="sports_product", kind="risk", status="open")

    assert result["ok"] is True
    assert result["count"] == 1
    assert result["items"][0]["title"] == "Feed delay"


def test_update_changes_status_and_appends_note():
    added = _result(action="add", kind="open_loop", title="Confirm QA date", content="QA date needed", project="sports_product", status="open")

    updated = _result(action="update", id=added["item"]["id"], status="closed", note="QA date confirmed for Friday.")

    assert updated["ok"] is True
    assert updated["item"]["status"] == "closed"
    assert updated["item"]["notes"][-1]["text"] == "QA date confirmed for Friday."


def test_summary_groups_open_items_and_recent_decisions():
    _result(action="add", kind="decision", title="Rollout flag", content="Use rollout flag", project="sports_product", status="closed")
    _result(action="add", kind="risk", title="Feed delay", content="Feed risk", project="sports_product", status="open")
    _result(action="add", kind="open_loop", title="Copy owner", content="Need owner", project="sports_product", status="open")

    result = _result(action="summary", project="sports_product")

    assert result["ok"] is True
    assert result["project"] == "sports_product"
    assert result["counts_by_kind"]["decision"] == 1
    assert result["open_items_count"] == 2
    assert {item["kind"] for item in result["open_items"]} == {"risk", "open_loop"}


def test_export_markdown_contains_structured_sections():
    _result(action="add", kind="person", title="Maya", content="Maya owns Sports analytics", project="sports_product", tags=["analytics"])
    _result(action="add", kind="decision", title="Rollout flag", content="Use rollout flag", project="sports_product")

    result = _result(action="export_markdown", project="sports_product")

    assert result["ok"] is True
    assert "## Decisions" in result["markdown"]
    assert "## People" in result["markdown"]
    assert "Maya owns Sports analytics" in result["markdown"]
