"""Tests for tools/skill_ledger.py — per-mutation audit ledger + rollback.

Covers tracker #79686 P3: ledger entries on patch/edit/delete/archive, blob
dedupe, single-entry rollback (incl. fail-closed safety capture), actor
tagging, and the skills.ledger config gate.

The first four tests are adapted from PR #50261 by @yu-xin-c (autonomous
skill history), reshaped for the all-actor JSONL ledger design.
"""

import json
from pathlib import Path

import pytest


VALID_SKILL_CONTENT = """---
name: my-skill
description: test skill
---

# My Skill

Original body.
"""


@pytest.fixture
def ledger_env(tmp_path, monkeypatch):
    """Isolated HERMES_HOME + skills dir for skill_manage and the ledger."""
    from agent import skill_utils
    from tools import skill_ledger, skill_manager_tool, skill_usage

    home = tmp_path / "home"
    skills_dir = home / "skills"
    skills_dir.mkdir(parents=True)

    monkeypatch.setattr(skill_ledger, "get_hermes_home", lambda: home)
    monkeypatch.setattr(skill_usage, "get_hermes_home", lambda: home)
    monkeypatch.setattr(skill_manager_tool, "SKILLS_DIR", skills_dir)
    monkeypatch.setattr(skill_utils, "get_all_skills_dirs", lambda: [skills_dir])
    return {"home": home, "skills": skills_dir}


def _create(name="my-skill", content=VALID_SKILL_CONTENT):
    from tools.skill_manager_tool import skill_manage

    return json.loads(skill_manage(action="create", name=name, content=content))


# ---------------------------------------------------------------------------
# Adapted from PR #50261 (@yu-xin-c)
# ---------------------------------------------------------------------------


def test_background_review_patch_ledgers_and_rolls_back(ledger_env, monkeypatch):
    """A curator-pass patch lands in the ledger tagged 'curator', and a
    single-entry rollback restores the exact pre-patch content."""
    from tools import skill_ledger
    from tools.skill_manager_tool import skill_manage
    from tools.skill_provenance import (
        BACKGROUND_REVIEW,
        reset_current_write_origin,
        set_current_write_origin,
    )
    from tools.skill_manager_tool import mark_background_review_skill_read

    token = set_current_write_origin(BACKGROUND_REVIEW)
    try:
        # Created under the review fork → marked created_by: agent, so the
        # curator pass is allowed to patch it (curator invariant unchanged).
        assert _create()["success"] is True
        skill_md = ledger_env["skills"] / "my-skill" / "SKILL.md"
        original = skill_md.read_text(encoding="utf-8")
        mark_background_review_skill_read(skill_md)
        patched = json.loads(
            skill_manage(
                action="patch",
                name="my-skill",
                old_string="Original body.",
                new_string="Updated body.",
            )
        )
    finally:
        reset_current_write_origin(token)

    assert patched["success"] is True
    assert "Updated body." in skill_md.read_text(encoding="utf-8")

    rows = skill_ledger.list_entries(skill="my-skill")
    patch_rows = [r for r in rows if r["action"] == "patch"]
    assert len(patch_rows) == 1
    entry = patch_rows[0]
    assert entry["actor"] == "curator"
    assert any(i["path"].endswith("SKILL.md") for i in entry["before"])

    ok, msg = skill_ledger.rollback_entry(entry["id"])
    assert ok is True, msg
    assert skill_md.read_text(encoding="utf-8") == original


def test_foreground_patch_is_ledgered_as_agent(ledger_env):
    """Foreground skill_manage patches are ledgered too (all-actor design —
    unlike #50261's autonomous-only history) and tagged 'agent'."""
    from tools import skill_ledger
    from tools.skill_manager_tool import skill_manage

    assert _create()["success"] is True
    patched = json.loads(
        skill_manage(
            action="patch",
            name="my-skill",
            old_string="Original body.",
            new_string="Updated body.",
        )
    )
    assert patched["success"] is True

    rows = [r for r in skill_ledger.list_entries(skill="my-skill") if r["action"] == "patch"]
    assert len(rows) == 1
    assert rows[0]["actor"] == "agent"


def test_rollback_refuses_paths_outside_hermes_home(ledger_env):
    """A hand-edited ledger entry pointing outside HERMES_HOME must not
    become a write-anywhere primitive."""
    from tools import skill_ledger

    entry_id = skill_ledger.append_entry(
        "patch",
        "evil",
        before=[{"path": "/etc/passwd", "sha256": "0" * 64}],
        after=[],
    )
    assert entry_id is not None
    ok, msg = skill_ledger.rollback_entry(entry_id)
    assert ok is False
    assert "outside" in msg


def test_missing_blob_aborts_rollback_before_any_change(ledger_env):
    from tools import skill_ledger

    assert _create()["success"] is True
    skill_md = ledger_env["skills"] / "my-skill" / "SKILL.md"
    entry_id = skill_ledger.append_entry(
        "patch",
        "my-skill",
        before=[{"path": str(skill_md), "sha256": "a" * 64}],
        after=[],
    )
    current = skill_md.read_bytes()
    ok, msg = skill_ledger.rollback_entry(entry_id)
    assert ok is False
    assert "missing blob" in msg
    assert skill_md.read_bytes() == current


# ---------------------------------------------------------------------------
# New-design coverage
# ---------------------------------------------------------------------------


def test_ledger_entry_on_edit_and_delete(ledger_env):
    from tools import skill_ledger
    from tools.skill_manager_tool import skill_manage

    assert _create()["success"] is True
    edited = json.loads(
        skill_manage(
            action="edit",
            name="my-skill",
            content=VALID_SKILL_CONTENT.replace("Original body.", "Edited body."),
        )
    )
    assert edited["success"] is True
    deleted = json.loads(
        skill_manage(action="delete", name="my-skill", absorbed_into="")
    )
    assert deleted["success"] is True

    actions = [r["action"] for r in skill_ledger.list_entries(skill="my-skill")]
    assert actions == ["delete", "edit", "create"]  # newest first

    delete_entry = skill_ledger.list_entries(skill="my-skill")[0]
    # Delete intent recorded: explicit prune (absorbed_into="") + hard delete.
    assert delete_entry["evidence"]["absorbed_into"] == ""
    assert delete_entry["evidence"]["archived"] is False
    # Before-state captured, after empty (skill gone).
    assert delete_entry["before"]
    assert delete_entry["after"] == []


def test_deleted_skill_recoverable_from_ledger(ledger_env):
    """A foreground hard delete stays a hard delete — but the ledger entry
    can restore the skill's files from blobs."""
    from tools import skill_ledger
    from tools.skill_manager_tool import skill_manage

    assert _create()["success"] is True
    skill_md = ledger_env["skills"] / "my-skill" / "SKILL.md"
    original = skill_md.read_bytes()

    assert json.loads(skill_manage(action="delete", name="my-skill"))["success"]
    assert not skill_md.exists()

    entry = skill_ledger.list_entries(skill="my-skill")[0]
    ok, msg = skill_ledger.rollback_entry(entry["id"])
    assert ok is True, msg
    assert skill_md.read_bytes() == original


def test_archive_lands_in_ledger_with_curator_actor(ledger_env, monkeypatch):
    from tools import skill_ledger, skill_usage

    assert _create()["success"] is True
    # Curator auto-transition path tags the actor explicitly.
    tok = skill_ledger.set_ledger_actor("curator")
    try:
        ok, msg = skill_usage.archive_skill("my-skill")
    finally:
        skill_ledger.reset_ledger_actor(tok)
    assert ok, msg

    rows = [r for r in skill_ledger.list_entries(skill="my-skill") if r["action"] == "archive"]
    assert len(rows) == 1
    assert rows[0]["actor"] == "curator"
    assert rows[0]["before"] and rows[0]["after"]

    # And restore is ledgered as well.
    ok, msg = skill_usage.restore_skill("my-skill")
    assert ok, msg
    assert any(
        r["action"] == "restore" for r in skill_ledger.list_entries(skill="my-skill")
    )


def test_blob_dedupe_same_content_one_blob(ledger_env):
    from tools import skill_ledger

    d = ledger_env["skills"] / "dedupe-src"
    d.mkdir()
    (d / "a.md").write_text("identical content", encoding="utf-8")
    (d / "b.md").write_text("identical content", encoding="utf-8")

    manifest = skill_ledger.snapshot_paths(d)
    assert len(manifest) == 2
    hashes = {m["sha256"] for m in manifest}
    assert len(hashes) == 1  # same content → same hash
    blobs = list(skill_ledger.blobs_dir().iterdir())
    assert len(blobs) == 1  # → one blob on disk


def test_rollback_fails_closed_when_safety_capture_fails(ledger_env, monkeypatch):
    """If the pre-rollback safety ledger entry can't be written, the rollback
    must abort with nothing changed (consistent with #63366)."""
    from tools import skill_ledger
    from tools.skill_manager_tool import skill_manage

    assert _create()["success"] is True
    skill_md = ledger_env["skills"] / "my-skill" / "SKILL.md"
    patched = json.loads(
        skill_manage(
            action="patch",
            name="my-skill",
            old_string="Original body.",
            new_string="Updated body.",
        )
    )
    assert patched["success"] is True
    entry = [r for r in skill_ledger.list_entries("my-skill") if r["action"] == "patch"][0]
    current = skill_md.read_bytes()

    monkeypatch.setattr(skill_ledger, "append_entry", lambda *a, **k: None)
    ok, msg = skill_ledger.rollback_entry(entry["id"])
    assert ok is False
    assert "safety capture failed" in msg
    assert skill_md.read_bytes() == current  # nothing changed


def test_rollback_removes_files_created_by_the_mutation(ledger_env):
    from tools import skill_ledger
    from tools.skill_manager_tool import skill_manage

    assert _create()["success"] is True
    wrote = json.loads(
        skill_manage(
            action="write_file",
            name="my-skill",
            file_path="references/extra.md",
            file_content="new supporting file",
        )
    )
    assert wrote["success"] is True
    extra = ledger_env["skills"] / "my-skill" / "references" / "extra.md"
    assert extra.exists()

    entry = [r for r in skill_ledger.list_entries("my-skill") if r["action"] == "write_file"][0]
    ok, msg = skill_ledger.rollback_entry(entry["id"])
    assert ok is True, msg
    assert not extra.exists()  # created by the mutation → removed on rollback


def test_config_gate_off_no_ledger_writes(ledger_env, monkeypatch):
    from tools import skill_ledger
    from tools.skill_manager_tool import skill_manage

    import hermes_cli.config as _cfg

    monkeypatch.setattr(_cfg, "load_config", lambda *a, **k: {"skills": {"ledger": False}})

    assert _create()["success"] is True
    patched = json.loads(
        skill_manage(
            action="patch",
            name="my-skill",
            old_string="Original body.",
            new_string="Updated body.",
        )
    )
    assert patched["success"] is True  # mutation unaffected
    assert not skill_ledger.ledger_path().exists()
    assert not skill_ledger.blobs_dir().exists()


def test_ledger_failure_never_blocks_the_mutation(ledger_env, monkeypatch):
    from tools import skill_ledger
    from tools.skill_manager_tool import skill_manage

    def _boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(skill_ledger, "snapshot_paths", _boom)

    assert _create()["success"] is True
    patched = json.loads(
        skill_manage(
            action="patch",
            name="my-skill",
            old_string="Original body.",
            new_string="Updated body.",
        )
    )
    assert patched["success"] is True


def test_list_entries_filtering_and_limit(ledger_env):
    from tools import skill_ledger

    for i in range(5):
        skill_ledger.append_entry("patch", f"skill-{i % 2}", before=[], after=[])
    assert len(skill_ledger.list_entries(limit=3)) == 3
    only_zero = skill_ledger.list_entries(skill="skill-0")
    assert len(only_zero) == 3
    assert all(r["skill"] == "skill-0" for r in only_zero)


def test_user_actor_override(ledger_env):
    from tools import skill_ledger

    tok = skill_ledger.set_ledger_actor("user")
    try:
        entry_id = skill_ledger.append_entry("archive", "some-skill")
    finally:
        skill_ledger.reset_ledger_actor(tok)
    entry = skill_ledger.get_entry(entry_id)
    assert entry["actor"] == "user"
