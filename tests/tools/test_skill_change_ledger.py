"""Focused tests for the local skill change ledger."""

import json
from datetime import datetime
from pathlib import Path

from tools.skill_change_ledger import (
    compute_text_diff,
    get_skill_change,
    hash_skill_dir,
    list_skill_changes,
    mark_skill_change_reviewed,
    record_skill_change,
)


EXPECTED_EVENT_KEYS = {
    "event_id",
    "timestamp",
    "skill",
    "category",
    "action",
    "actor",
    "source",
    "session_id",
    "reason",
    "reason_kind",
    "before_hash",
    "after_hash",
    "changed_files",
    "diff_path",
    "metadata",
    "review_status",
    "reviewed_at",
    "review_note",
}


def _set_hermes_home(monkeypatch, tmp_path: Path) -> Path:
    hermes_home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    return hermes_home


def _ledger_lines(hermes_home: Path) -> list[str]:
    return (hermes_home / "skill_changes.jsonl").read_text(encoding="utf-8").splitlines()


def test_empty_ledger_returns_clean_results(monkeypatch, tmp_path):
    hermes_home = _set_hermes_home(monkeypatch, tmp_path)

    assert not (hermes_home / "skill_changes.jsonl").exists()
    assert list_skill_changes() == []
    assert get_skill_change("missing-event") is None
    assert mark_skill_change_reviewed("missing-event") is None


def test_hash_skill_dir_is_deterministic_and_handles_missing(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    (first / "references").mkdir(parents=True)
    (second / "references").mkdir(parents=True)

    # Create the same relative files in different orders; only relative paths and
    # bytes should drive the digest.
    (first / "SKILL.md").write_text("# Demo\n", encoding="utf-8")
    (first / "references" / "notes.txt").write_text("notes\n", encoding="utf-8")
    (second / "references" / "notes.txt").write_text("notes\n", encoding="utf-8")
    (second / "SKILL.md").write_text("# Demo\n", encoding="utf-8")

    digest = hash_skill_dir(first)
    assert digest is not None
    assert digest.startswith("sha256:")
    assert digest == hash_skill_dir(second)

    (second / "SKILL.md").write_text("# Demo v2\n", encoding="utf-8")
    assert digest != hash_skill_dir(second)
    assert hash_skill_dir(tmp_path / "does-not-exist") is None


def test_compute_text_diff_returns_unified_diff_for_changes_adds_and_deletes():
    diff = compute_text_diff(
        before={
            "SKILL.md": "old line\nsame\n",
            "deleted.md": "remove me\n",
        },
        after={
            "SKILL.md": "new line\nsame\n",
            "added.md": "add me\n",
        },
    )

    assert "--- a/SKILL.md" in diff
    assert "+++ b/SKILL.md" in diff
    assert "-old line" in diff
    assert "+new line" in diff
    assert "--- a/added.md" in diff
    assert "+add me" in diff
    assert "--- a/deleted.md" in diff
    assert "-remove me" in diff


def test_record_skill_change_writes_default_ledger_and_diff_artifact(monkeypatch, tmp_path):
    hermes_home = _set_hermes_home(monkeypatch, tmp_path)

    event = record_skill_change(
        skill="demo-skill",
        category="devops",
        action="patch",
        actor="hermes-agent",
        source="skill_manage",
        session_id="session-123",
        reason=None,
        before_hash="sha256:old",
        after_hash="sha256:new",
        changed_files=["SKILL.md"],
        before_text={"SKILL.md": "old\n"},
        after_text={"SKILL.md": "new\n"},
        metadata={"tool_call": "skill_manage.patch"},
    )

    assert set(event) == EXPECTED_EVENT_KEYS
    assert event["skill"] == "demo-skill"
    assert event["category"] == "devops"
    assert event["action"] == "patch"
    assert event["actor"] == "hermes-agent"
    assert event["source"] == "skill_manage"
    assert event["session_id"] == "session-123"
    assert event["reason"] is None
    assert event["reason_kind"] == "unattributed"
    assert event["before_hash"] == "sha256:old"
    assert event["after_hash"] == "sha256:new"
    assert event["changed_files"] == ["SKILL.md"]
    assert event["metadata"] == {"tool_call": "skill_manage.patch"}
    assert event["review_status"] == "unreviewed"
    assert event["reviewed_at"] is None
    assert event["review_note"] is None
    assert event["event_id"]
    datetime.fromisoformat(event["timestamp"])

    ledger_path = hermes_home / "skill_changes.jsonl"
    assert ledger_path.exists()
    lines = _ledger_lines(hermes_home)
    assert len(lines) == 1
    assert json.loads(lines[0]) == event

    diff_path = Path(event["diff_path"])
    assert diff_path == hermes_home / "skill-history" / "events" / event["event_id"] / "diff.patch"
    diff_text = diff_path.read_text(encoding="utf-8")
    assert "-old" in diff_text
    assert "+new" in diff_text


def test_get_skill_change_includes_diff_text_when_artifact_exists(monkeypatch, tmp_path):
    _set_hermes_home(monkeypatch, tmp_path)
    raw_diff = "--- a/SKILL.md\n+++ b/SKILL.md\n@@ -1 +1 @@\n-old\n+new\n"
    event = record_skill_change(
        skill="demo-skill",
        action="edit",
        actor="hermes-agent",
        source="unit-test",
        reason="User asked for clearer instructions.",
        before_hash=None,
        after_hash="sha256:new",
        changed_files=["SKILL.md"],
        diff_text=raw_diff,
    )

    detail = get_skill_change(event["event_id"])

    assert detail is not None
    assert detail["event_id"] == event["event_id"]
    assert detail["reason_kind"] == "explicit"
    assert detail["diff_text"] == raw_diff


def test_list_skill_changes_filters_by_skill_limit_and_review_state(monkeypatch, tmp_path):
    _set_hermes_home(monkeypatch, tmp_path)
    first = record_skill_change(
        skill="alpha",
        action="create",
        actor="hermes-agent",
        source="unit-test",
        reason="Create alpha.",
    )
    second = record_skill_change(
        skill="beta",
        action="patch",
        actor="hermes-agent",
        source="unit-test",
        reason="Patch beta.",
    )
    third = record_skill_change(
        skill="alpha",
        action="edit",
        actor="hermes-agent",
        source="unit-test",
        reason="Edit alpha.",
    )

    assert [event["event_id"] for event in list_skill_changes(limit=2)] == [
        third["event_id"],
        second["event_id"],
    ]
    assert [event["event_id"] for event in list_skill_changes(skill="alpha")] == [
        third["event_id"],
        first["event_id"],
    ]

    reviewed = mark_skill_change_reviewed(second["event_id"])
    assert reviewed is not None
    assert reviewed["review_status"] == "reviewed"

    assert [event["event_id"] for event in list_skill_changes(unreviewed=True)] == [
        third["event_id"],
        first["event_id"],
    ]
    assert [event["event_id"] for event in list_skill_changes(unreviewed=False)] == [
        second["event_id"],
    ]


def test_mark_skill_change_reviewed_appends_review_update(monkeypatch, tmp_path):
    hermes_home = _set_hermes_home(monkeypatch, tmp_path)
    event = record_skill_change(
        skill="demo-skill",
        action="edit",
        actor="hermes-agent",
        source="unit-test",
        reason_kind="system",
        reason="Bundled skill sync detected an update.",
    )

    updated = mark_skill_change_reviewed(
        event["event_id"],
        status="needs_followup",
        note="Confirm generated diff before accepting.",
    )

    assert updated is not None
    assert updated["event_id"] == event["event_id"]
    assert updated["review_status"] == "needs_followup"
    assert updated["review_note"] == "Confirm generated diff before accepting."
    assert updated["reviewed_at"] is not None
    datetime.fromisoformat(updated["reviewed_at"])

    lines = _ledger_lines(hermes_home)
    assert len(lines) == 2
    assert json.loads(lines[0])["review_status"] == "unreviewed"
    assert json.loads(lines[1])["review_status"] == "needs_followup"

    detail = get_skill_change(event["event_id"])
    assert detail is not None
    assert detail["review_status"] == "needs_followup"
    assert detail["review_note"] == "Confirm generated diff before accepting."
