from datetime import datetime, timezone, UTC
import json

from agent.skill_utils import TIER_LOCAL, get_skill_search_roots
from tools.skill_lifecycle import (
    LIFECYCLE_TIMEZONE, archive_review_candidates, create_proposal,
    deterministic_scan, freeze_inventory, lifecycle_job_specs, morning_digest,
    proposal_for_session, restrict_session, review_proposals, tool_allowed,
    write_batch_report,
)

SKILL = """---
name: safe-reader
description: Read local reports safely.
---
# Safe Reader

Use `read_file` to inspect a report.
"""
WHY = {"need": "Repeatable local report inspection.", "tools": ["read_file"],
       "example_run": "Load a report and summarize it."}


def test_proposal_is_session_bound_and_not_in_active_catalog(tmp_path):
    path = create_proposal("safe-reader", SKILL, session_id="creator", risk="read-only", why=WHY, home=tmp_path)
    assert path == tmp_path / "skills" / "proposed" / "safe-reader"
    assert (path / "SKILL.md").is_file()
    assert (path / "why.md").is_file()
    assert proposal_for_session("safe-reader", "creator", tmp_path) == path
    assert proposal_for_session("safe-reader", "other", tmp_path) is None
    assert not (tmp_path / "skills" / "active" / "safe-reader").exists()


def test_loading_proposal_restricts_creating_session_only():
    restrict_session("creator-policy-test")
    assert tool_allowed("creator-policy-test", "read_file")
    assert not tool_allowed("creator-policy-test", "write_file")
    assert not tool_allowed("creator-policy-test", "imessage")
    assert not tool_allowed("creator-policy-test", "memory")
    assert tool_allowed("other", "write_file")


def test_scan_rejects_all_required_danger_classes(tmp_path):
    path = create_proposal("safe-reader", SKILL, session_id="creator", risk="read-only", why=WHY, home=tmp_path)
    (path / "references").mkdir()
    (path / "references" / "bad.md").write_text(
        "hidden\u200b https://example.test webhook /send .env send_email write_file base64", encoding="utf-8")
    result = deterministic_scan(path)
    assert not result.passed
    assert {"hidden-unicode", "url-or-webhook", "slash-command", "secret-path",
            "send-call", "write-call"} <= set(result.findings)
    # Mentioning the encoding is prose, not itself an encoded payload.
    assert "encoded-payload" not in result.findings


def test_review_auto_promotes_read_only_and_requires_booth_for_widening(tmp_path):
    create_proposal("safe-reader", SKILL, session_id="one", risk="read-only", why=WHY, home=tmp_path)
    create_proposal("writer", SKILL.replace("safe-reader", "writer"), session_id="two", risk="write", why=WHY, home=tmp_path)
    outcome = review_proposals(tmp_path)
    assert [row["name"] for row in outcome["merged"]] == ["safe-reader"]
    assert [row["name"] for row in outcome["waiting_approval"]] == ["writer"]
    assert (tmp_path / "skills" / "active" / "safe-reader").is_dir()
    assert (tmp_path / "skills" / "proposed" / "writer").is_dir()
    approved = review_proposals(tmp_path, booth_approved={"writer"}, reviewer_session_id="reviewer")
    assert [row["name"] for row in approved["merged"]] == ["writer"]


def test_creator_cannot_self_approve_access_widening(tmp_path):
    create_proposal("writer", SKILL.replace("safe-reader", "writer"), session_id="creator", risk="write", why=WHY, home=tmp_path)
    outcome = review_proposals(tmp_path, booth_approved={"writer"}, reviewer_session_id="creator")
    assert not outcome["merged"]
    assert outcome["waiting_approval"]


def test_schedules_digest_archive_and_read_only_inventory(tmp_path):
    specs = lifecycle_job_specs()
    assert [(j["schedule"], j["timezone"]) for j in specs] == [
        ("0 1 * * *", LIFECYCLE_TIMEZONE), ("30 7 * * *", LIFECYCLE_TIMEZONE),
        ("0 8 1 * *", LIFECYCLE_TIMEZONE)]
    active = tmp_path / "skills" / "active" / "old"
    active.mkdir(parents=True)
    (active / "SKILL.md").write_text(SKILL, encoding="utf-8")
    (tmp_path / "skills" / ".usage.json").write_text(json.dumps({"old": {"last_used_at": "2020-01-01T00:00:00+00:00"}}), encoding="utf-8")
    assert archive_review_candidates(tmp_path, now=datetime(2026, 1, 1, tzinfo=UTC)) == ["old"]
    inv = freeze_inventory(tmp_path)
    assert inv["count"] == 1 and len(inv["skills"][0]["sha256"]) == 64
    report = {"merged": [{"name": "old"}], "waiting_approval": [], "rejected": []}
    write_batch_report(report, tmp_path)
    assert "Merged: 1" in morning_digest(tmp_path)


def test_legacy_root_remains_until_safe_bootstrap_marker(tmp_path, monkeypatch):
    from tools.skill_lifecycle import bootstrap_layout
    legacy = tmp_path / "skills"
    package = legacy / "legacy-reader"
    package.mkdir(parents=True)
    (package / "SKILL.md").write_text(SKILL, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    roots = get_skill_search_roots(local=legacy, include_project=False)
    assert (TIER_LOCAL, legacy) in roots
    (legacy / "active").mkdir()
    # An empty directory is not authority to hide legacy packages.
    assert (TIER_LOCAL, legacy) in get_skill_search_roots(local=legacy, include_project=False)
    result = bootstrap_layout(tmp_path)
    assert result["migrated"] == ["legacy-reader"]
    assert (legacy / "active" / "legacy-reader" / "SKILL.md").is_file()
    assert (TIER_LOCAL, legacy / "active") in get_skill_search_roots(local=legacy, include_project=False)


def test_inventory_hashes_every_package_file(tmp_path):
    package = tmp_path / "skills" / "active" / "reader"
    (package / "references").mkdir(parents=True)
    (package / "SKILL.md").write_text(SKILL, encoding="utf-8")
    (package / "references" / "case.md").write_text("case", encoding="utf-8")
    inv = freeze_inventory(tmp_path)
    assert inv["file_count"] == 2
    assert {row["path"] for row in inv["skills"][0]["files"]} == {
        "active/reader/SKILL.md", "active/reader/references/case.md"}


def test_reviewer_access_conflict_cannot_approve(tmp_path):
    from tools.skill_lifecycle import review_proposals
    create_proposal("writer", SKILL.replace("safe-reader", "writer"), session_id="creator",
                    creator_identity="agent-a", creator_capabilities={"write"},
                    risk="write", why=WHY, home=tmp_path)
    outcome = review_proposals(tmp_path, booth_approved={"writer"},
                               reviewer={"identity": "agent-b", "capabilities": ["write"]})
    assert outcome["waiting_approval"][0]["reason"] == "reviewer access conflict"


def test_job_specs_are_executable_and_digest_uses_delivery_surface():
    specs = lifecycle_job_specs()
    assert all(j["no_agent"] and j["script"] for j in specs)
    assert specs[1]["deliver"] == "origin"
