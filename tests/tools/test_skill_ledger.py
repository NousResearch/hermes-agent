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


# ---------------------------------------------------------------------------
# Package-completeness fill from the newest curator backup (issue #96962)
# ---------------------------------------------------------------------------


def _write_skills_tarball(home: Path, files: dict, stamp: str = "2026-08-01T00-00-00Z"):
    """Write a curator-shaped ``skills.tar.gz`` under *home* (arcnames are
    relative to skills/, exactly like agent.curator_backup.snapshot_skills)."""
    import io
    import tarfile

    snap = home / "skills" / ".curator_backups" / stamp
    snap.mkdir(parents=True, exist_ok=True)
    tar_path = snap / "skills.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        for rel, content in files.items():
            data = content.encode("utf-8") if isinstance(content, str) else content
            info = tarfile.TarInfo(name=rel)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return tar_path


def test_delete_after_rehome_ledgers_full_package_from_backup(ledger_env):
    """The incident shape (#96962): consolidation re-homes references/ out of
    the tree, then deletes. The delete entry must still capture the support
    file from the newest curator backup, and rollback must restore both."""
    from tools import skill_ledger
    from tools.skill_manager_tool import skill_manage

    assert _create()["success"] is True
    extra = ledger_env["skills"] / "my-skill" / "references" / "extra.md"
    wrote = json.loads(skill_manage(
        action="write_file",
        name="my-skill",
        file_path="references/extra.md",
        file_content="roadmap body",
    ))
    assert wrote["success"] is True

    # The pre-curator-run snapshot, taken while the package was whole.
    skill_md = ledger_env["skills"] / "my-skill" / "SKILL.md"
    _write_skills_tarball(
        ledger_env["home"],
        {
            "my-skill/SKILL.md": skill_md.read_text(encoding="utf-8"),
            "my-skill/references/extra.md": "roadmap body",
        },
    )

    # Re-home: the support file leaves the tree before the delete.
    extra.unlink()
    extra.parent.rmdir()

    deleted = json.loads(skill_manage(action="delete", name="my-skill"))
    assert deleted["success"] is True

    delete_entry = [
        r for r in skill_ledger.list_entries(skill="my-skill")
        if r["action"] == "delete"
    ][0]
    before_names = {Path(i["path"]).name for i in delete_entry["before"]}
    assert "SKILL.md" in before_names
    assert "extra.md" in before_names, (
        "delete ledger captured only SKILL.md after the support files were "
        "re-homed — rollback would restore a hollow skill (#96962)"
    )

    ok, msg = skill_ledger.rollback_entry(delete_entry["id"])
    assert ok is True, msg
    assert skill_md.is_file()
    assert extra.is_file()
    assert extra.read_text(encoding="utf-8") == "roadmap body"


def test_rollback_historical_hollow_entry_restores_full_package(ledger_env):
    """Entries recorded BEFORE this fix (files: 1) still restore the whole
    package: rollback-time fill from the newest curator backup."""
    from tools import skill_ledger

    skill_dir = ledger_env["skills"] / "my-skill"
    skill_dir.mkdir()
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(VALID_SKILL_CONTENT, encoding="utf-8")
    _write_skills_tarball(
        ledger_env["home"],
        {
            "my-skill/SKILL.md": VALID_SKILL_CONTENT,
            "my-skill/references/roadmap.md": "week 1",
        },
    )
    # The mutation that made the entry: package gone, only SKILL.md captured.
    skill_md.unlink()
    skill_dir.rmdir()

    entry_id = skill_ledger.append_entry(
        "delete",
        "my-skill",
        before=[{"path": str(skill_md), "sha256": skill_ledger._store_blob(
            VALID_SKILL_CONTENT.encode("utf-8")
        )}],
        after=[],
    )
    assert entry_id is not None

    ok, msg = skill_ledger.rollback_entry(entry_id)
    assert ok is True, msg
    roadmap = skill_dir / "references" / "roadmap.md"
    assert skill_md.is_file()
    assert roadmap.is_file(), "hollow rollback: support file not restored"
    assert roadmap.read_text(encoding="utf-8") == "week 1"


def test_delete_rollback_without_backup_still_works(ledger_env):
    """No curator backup present: the fill degrades to the old behavior and
    must not break the plain delete -> rollback round trip."""
    from tools import skill_ledger
    from tools.skill_manager_tool import skill_manage

    assert _create()["success"] is True
    skill_md = ledger_env["skills"] / "my-skill" / "SKILL.md"

    deleted = json.loads(skill_manage(action="delete", name="my-skill"))
    assert deleted["success"] is True
    delete_entry = [
        r for r in skill_ledger.list_entries(skill="my-skill")
        if r["action"] == "delete"
    ][0]
    assert {Path(i["path"]).name for i in delete_entry["before"]} == {"SKILL.md"}

    ok, msg = skill_ledger.rollback_entry(delete_entry["id"])
    assert ok is True, msg
    assert skill_md.read_text(encoding="utf-8") == VALID_SKILL_CONTENT


def test_backup_fill_does_not_clobber_disk_hash(ledger_env):
    """Disk state wins: a live SKILL.md that differs from the backup copy is
    captured with the LIVE hash; the backup only fills missing paths."""
    from tools import skill_ledger

    skill_dir = ledger_env["skills"] / "my-skill"
    skill_dir.mkdir()
    skill_md = skill_dir / "SKILL.md"
    live = VALID_SKILL_CONTENT.replace("Original body.", "Live body.")
    skill_md.write_text(live, encoding="utf-8")
    _write_skills_tarball(
        ledger_env["home"],
        {
            "my-skill/SKILL.md": VALID_SKILL_CONTENT,
            "my-skill/references/extra.md": "from tar",
        },
    )

    captured = skill_ledger.snapshot_paths(skill_dir, complete_package=True)
    by_name = {Path(i["path"]).name: i["sha256"] for i in captured}
    live_hash = skill_ledger._store_blob(live.encode("utf-8"))
    tar_hash = skill_ledger._store_blob(VALID_SKILL_CONTENT.encode("utf-8"))
    assert by_name["SKILL.md"] == live_hash, "disk hash must win over backup"
    assert by_name["SKILL.md"] != tar_hash
    assert by_name["extra.md"] == skill_ledger._store_blob(b"from tar")


def test_backup_fill_ignores_tar_path_traversal(ledger_env):
    """Fill runs AND malicious members are rejected: a legitimate missing
    file is restored while members escaping the package prefix (absolute,
    ..) are never filled. Both assertions matter — the positive one keeps
    this test honest (a silently inert fill would pass a negatives-only
    check), the negative one pins the traversal defense."""
    from tools import skill_ledger

    skill_dir = ledger_env["skills"] / "my-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(VALID_SKILL_CONTENT, encoding="utf-8")
    _write_skills_tarball(
        ledger_env["home"],
        {
            "my-skill/SKILL.md": VALID_SKILL_CONTENT,
            "my-skill/references/legit.md": "legit body",
            "../evil.md": "nope",
            "my-skill/../outside.md": "nope",
        },
    )

    captured = skill_ledger.snapshot_paths(skill_dir, complete_package=True)
    paths = [i["path"] for i in captured]
    # The legitimate missing file WAS filled — proof the fill is live.
    assert any(p.endswith("references/legit.md") for p in paths), (
        "package fill did not restore the missing support file"
    )
    # Malicious members are not.
    assert not any(p.endswith("evil.md") or p.endswith("outside.md") for p in paths)


# ---------------------------------------------------------------------------
# File-tool skill-tree writes (observability for write_file / patch bypass)
# ---------------------------------------------------------------------------


def test_classify_skill_tree_write_same_profile(ledger_env):
    from tools import skill_ledger

    skill_md = ledger_env["skills"] / "obs-gap" / "SKILL.md"
    skill_md.parent.mkdir(parents=True)
    skill_md.write_text(VALID_SKILL_CONTENT.replace("my-skill", "obs-gap"), encoding="utf-8")
    hit = skill_ledger.classify_skill_tree_write(str(skill_md))
    assert hit is not None
    assert hit["skill"] == "obs-gap"
    assert hit["path"] == str(skill_md.resolve())


def test_classify_skill_tree_write_skips_sidecars_and_outsiders(ledger_env, tmp_path):
    from tools import skill_ledger

    usage = ledger_env["skills"] / ".usage.json"
    usage.write_text("{}", encoding="utf-8")
    assert skill_ledger.classify_skill_tree_write(str(usage)) is None

    hub = ledger_env["skills"] / ".hub" / "lock.json"
    hub.parent.mkdir()
    hub.write_text("{}", encoding="utf-8")
    assert skill_ledger.classify_skill_tree_write(str(hub)) is None

    outsider = tmp_path / "notes.txt"
    outsider.write_text("hi", encoding="utf-8")
    assert skill_ledger.classify_skill_tree_write(str(outsider)) is None


def test_write_file_to_skills_is_ledgered(ledger_env, monkeypatch):
    """write_file landing in ~/.hermes/skills must leave an apply event
    even when skills.write_approval is on (the gate only covers skill_manage)."""
    from tools import skill_ledger, write_approval as wa
    from tools.file_tools import write_file_tool

    monkeypatch.setenv("HERMES_HOME", str(ledger_env["home"]))
    monkeypatch.setattr(wa, "write_approval_enabled", lambda subsystem: True)

    target = ledger_env["skills"] / "obs-gap" / "SKILL.md"
    content = VALID_SKILL_CONTENT.replace("name: my-skill", "name: obs-gap")
    result = json.loads(write_file_tool(str(target), content))
    assert not result.get("error"), result
    assert target.exists()
    assert wa.pending_count("skills") == 0

    rows = [r for r in skill_ledger.list_entries() if r.get("skill") == "obs-gap"]
    assert rows, "write_file to a skills tree must append a ledger apply event"
    entry = rows[0]
    assert entry["action"] == "write_file"
    assert entry["evidence"].get("source") == "write_file"
    assert entry["evidence"].get("path", "").endswith("SKILL.md")


def test_patch_to_skills_is_ledgered(ledger_env, monkeypatch):
    from tools import skill_ledger
    from tools.file_tools import patch_tool, write_file_tool

    monkeypatch.setenv("HERMES_HOME", str(ledger_env["home"]))
    target = ledger_env["skills"] / "obs-gap" / "SKILL.md"
    content = VALID_SKILL_CONTENT.replace("name: my-skill", "name: obs-gap")
    assert not json.loads(write_file_tool(str(target), content)).get("error")

    patched = json.loads(
        patch_tool(
            mode="replace",
            path=str(target),
            old_string="Original body.",
            new_string="Patched body.",
        )
    )
    assert not patched.get("error"), patched
    assert "Patched body." in target.read_text(encoding="utf-8")

    patch_rows = [
        r for r in skill_ledger.list_entries(skill="obs-gap") if r["action"] == "patch"
    ]
    assert patch_rows
    assert patch_rows[0]["evidence"].get("source") == "patch"


def test_write_file_outside_skills_is_not_ledgered(ledger_env, tmp_path, monkeypatch):
    from tools import skill_ledger
    from tools.file_tools import write_file_tool

    monkeypatch.setenv("HERMES_HOME", str(ledger_env["home"]))
    outsider = tmp_path / "notes.txt"
    result = json.loads(write_file_tool(str(outsider), "hello"))
    assert not result.get("error"), result
    assert outsider.exists()
    assert skill_ledger.list_entries() == []
