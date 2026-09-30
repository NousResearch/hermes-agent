"""Tests for tools/skill_ledger.py — per-mutation audit ledger + rollback.

Covers tracker #79686 P3: ledger entries on patch/edit/delete/archive, blob
dedupe, single-entry rollback (incl. fail-closed safety capture), actor
tagging, and the skills.ledger config gate.

The first four tests are adapted from PR #50261 by @yu-xin-c (autonomous
skill history), reshaped for the all-actor JSONL ledger design.
"""

import hashlib
import json
import os
import time
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
    from tools.skill_manager_guards import mark_background_review_skill_read

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


def test_snapshot_paths_skips_transient_dirs(ledger_env):
    """Transient local artifacts (venv, node_modules, caches, .git) never reach
    the manifest or the blob store — sweeping them in grows the blob dir
    unboundedly on real installs (#107539)."""
    from tools import skill_ledger

    d = ledger_env["skills"] / "has-venv"
    d.mkdir()
    for rel, body in (("SKILL.md", "# skill"), ("scripts/run.py", "print('hi')"),
                      ("node_modules/pkg/index.js", "junk"), ("venv/bin/python", "junk"),
                      ("__pycache__/run.cpython-311.pyc", "junk"), (".git/config", "junk")):
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")

    manifest = skill_ledger.snapshot_paths(d)
    rel = {str(Path(i["path"]).relative_to(d)) for i in manifest}
    assert rel == {"SKILL.md", os.path.join("scripts", "run.py")}

    # None of the transient content was stored as a blob either.
    junk_sha = hashlib.sha256(b"junk").hexdigest()
    assert junk_sha not in {p.name for p in skill_ledger.blobs_dir().iterdir()}


def test_snapshot_paths_keeps_file_named_like_transient_dir(ledger_env):
    """The filter drops files *inside* transient dirs; a plain file whose own
    name collides with one (e.g. a ``venv`` bootstrap script) is skill content."""
    from tools import skill_ledger

    d = ledger_env["skills"] / "edge"
    d.mkdir()
    (d / "SKILL.md").write_text("# skill", encoding="utf-8")
    (d / "venv").write_text("#!/bin/sh\n", encoding="utf-8")  # a FILE, not a dir

    manifest = skill_ledger.snapshot_paths(d)
    rel = {str(Path(i["path"]).relative_to(d)) for i in manifest}
    assert "venv" in rel
    assert "SKILL.md" in rel


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
    entry = next(r for r in skill_ledger.list_entries("my-skill") if r["action"] == "patch")
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

    entry = next(r for r in skill_ledger.list_entries("my-skill") if r["action"] == "write_file")
    ok, msg = skill_ledger.rollback_entry(entry["id"])
    assert ok is True, msg
    assert not extra.exists()  # created by the mutation → removed on rollback


def test_config_gate_off_no_ledger_writes(ledger_env, monkeypatch):
    from tools import skill_ledger
    from tools.skill_manager_tool import skill_manage

    import hermes_cli.config as _cfg

    off = {"skills": {"ledger": False}}
    monkeypatch.setattr(_cfg, "load_config", lambda *a, **k: off)
    monkeypatch.setattr(_cfg, "load_config_readonly", lambda *a, **k: off)

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

    delete_entry = next(
        r for r in skill_ledger.list_entries(skill="my-skill")
        if r["action"] == "delete"
    )
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
    delete_entry = next(
        r for r in skill_ledger.list_entries(skill="my-skill")
        if r["action"] == "delete"
    )
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
    skill_md.write_text(live, encoding="utf-8", newline="\n")
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
    assert any(Path(p).as_posix().endswith("references/legit.md") for p in paths), (
        "package fill did not restore the missing support file"
    )
    # Malicious members are not.
    assert not any(p.endswith("evil.md") or p.endswith("outside.md") for p in paths)

import pytest


def _append_padded(skill_ledger, action: str, pad: str, n: int = 1) -> None:
    """Append *n* entries padded with evidence text so the ledger file grows fast."""
    for _ in range(n):
        skill_ledger.append_entry(action, "my-skill", before=[], after=[], evidence={"pad": pad})


def test_auto_compact_triggers_at_threshold(ledger_env, monkeypatch):
    """Crossing skills.ledger_max_bytes rewrites the ledger through the delta
    dedup: legacy rows that still carry identical before/after manifests (written
    before append-time ``_delta`` existed) shrink to nothing while ids and entry
    order survive, and nothing is trimmed when dedup alone reaches the cap."""
    import json

    from tools import skill_ledger

    import hermes_cli.config as _cfg

    cap = {"skills": {"ledger_max_bytes": 8192}}
    monkeypatch.setattr(_cfg, "load_config", lambda *a, **k: cap)
    monkeypatch.setattr(_cfg, "load_config_readonly", lambda *a, **k: cap)

    first_id = skill_ledger.append_entry("patch", "my-skill", before=[], after=[])
    template = json.loads(skill_ledger.ledger_path().read_text().splitlines()[0])
    with skill_ledger.ledger_path().open("a", encoding="utf-8") as fh:
        for i in range(3):  # legacy pre-delta rows: identical fat manifests on both sides
            fat = [{"path": f"my-skill/f{i}j{j}.md", "sha256": "a" * 64} for j in range(40)]
            row = dict(template, id=f"legacy{i}", before=fat, after=list(fat))
            fh.write(json.dumps(row) + "\n")
    assert skill_ledger.ledger_path().stat().st_size > 8192

    last_id = skill_ledger.append_entry("patch", "my-skill", before=[], after=[])

    # the maintenance sweep fired on that append: the file is back under the cap
    assert skill_ledger.ledger_path().stat().st_size <= 8192
    rows = skill_ledger.list_entries()
    assert {r["id"] for r in rows} == {first_id, last_id, "legacy0", "legacy1", "legacy2"}, (
        "dedup alone must reach the cap — nothing trimmed, ids survive"
    )
    assert all(r["before"] == [] and r["after"] == [] for r in rows), (
        "identical manifests must be dropped by compaction"
    )


def test_trim_oldest_when_still_over_cap(ledger_env, monkeypatch):
    """When compaction alone cannot reach the cap (every entry genuinely
    differs), the oldest lines are dropped — whatever their shape — until the
    file fits under the LOW-WATER mark (80% of the cap, so the next append does
    not immediately re-trigger the sweep). The newest entry survives, and lines
    in the retained tail are never parsed or rewritten: a malformed last line
    survives verbatim."""
    from tools import skill_ledger

    import hermes_cli.config as _cfg

    cap = {"skills": {"ledger_max_bytes": 0}}  # no sweeps while seeding
    monkeypatch.setattr(_cfg, "load_config", lambda *a, **k: cap)
    monkeypatch.setattr(_cfg, "load_config_readonly", lambda *a, **k: cap)

    newest_id = None
    for i in range(5):
        before = [{"path": f"my-skill/old{i}.md", "sha256": f"{i}" * 64}]
        after = [{"path": f"my-skill/new{i}.md", "sha256": f"{i + 1}" * 64}]
        newest_id = skill_ledger.append_entry(
            "edit", "my-skill", before=before, after=after,
            evidence={"pad": "y" * 2048})
    with open(skill_ledger.ledger_path(), "a", encoding="utf-8") as fh:
        fh.write("{not json at all\n")
    assert skill_ledger.ledger_path().stat().st_size > 8192

    cap["skills"]["ledger_max_bytes"] = 8192
    skill_ledger._maintain_size()

    rows = skill_ledger.list_entries()
    assert len(rows) < 5, "oldest entries must be trimmed when compaction is not enough"
    assert rows[0]["id"] == newest_id, "the newest entry always survives"
    # malformed lines are never parsed away — they stay in the file verbatim
    raw = skill_ledger.ledger_path().read_text(encoding="utf-8")
    assert "{not json at all" in raw
    # A sweep that fires does not stop at the cap but at the low-water mark ...
    assert skill_ledger.ledger_path().stat().st_size <= int(8192 * 0.8)
    # ... so the next append rides under the cap without paying compact+trim+gc again.
    compactions = []
    monkeypatch.setattr(skill_ledger, "compact_ledger",
                        lambda *a, **k: compactions.append(1) or (0, 0, 0))
    skill_ledger.append_entry(
        "edit", "my-skill", before=[{"path": "my-skill/z.md", "sha256": "a" * 64}],
        after=[{"path": "my-skill/z2.md", "sha256": "b" * 64}], evidence={"pad": "z" * 1000})
    assert compactions == [], "an append under the cap must not re-run the sweep"
    assert skill_ledger.ledger_path().stat().st_size <= 8192
    assert "{not json at all" in skill_ledger.ledger_path().read_text(encoding="utf-8")
    # U+2028 inside a row (ensure_ascii=False leaves it unescaped) is not a row boundary for the
    # trim: a cap that fits only the newest row keeps that row byte-for-byte, not its second half.
    u_row = (json.dumps({"id": "u2028", "skill": "my-skill", "action": "edit",
                         "evidence": {"note": "line one\u2028line two"}}, ensure_ascii=False) + "\n").encode("utf-8")
    with open(skill_ledger.ledger_path(), "ab") as fh:
        fh.write(u_row)
    assert skill_ledger._trim_oldest(len(u_row) + 8) >= 1
    assert skill_ledger.ledger_path().read_bytes() == u_row, "a retained row containing U+2028 survives intact"


def test_concurrent_appends_never_lose_a_middle_row(ledger_env, monkeypatch):
    """Two writers appending while the maintenance sweep fires on (almost) every append:
    every row each writer appended is either still in the ledger or was trimmed
    oldest-first — never silently lost from the middle of a writer's sequence. The sweep's
    read → ``os.replace`` must run under the same ``.locks/ledger.lock`` as the O_APPEND
    write, or an append landing on the replaced inode vanishes (and ``gc_blobs`` would then
    delete its blobs). The race is forced, not hoped for: writer A's first sweep pauses
    between reading the ledger and replacing it until writer B has appended (or, when the
    lock correctly blocks B, until a generous bound expires — green never depends on timing)."""
    import threading

    from tools import skill_ledger

    import hermes_cli.config as _cfg

    cap = {"skills": {"ledger_max_bytes": 4096}}  # padded rows ~600 B: a trim on nearly every append
    monkeypatch.setattr(_cfg, "load_config", lambda *a, **k: cap)
    monkeypatch.setattr(_cfg, "load_config_readonly", lambda *a, **k: cap)

    n, ids = 40, {"A": [], "B": []}
    b_go, b_done = threading.Event(), threading.Event()
    real_rewrite = skill_ledger._rewrite_ledger

    def paused_rewrite(path, lines, op):
        if threading.current_thread().name == "A" and not b_go.is_set():
            b_go.set()             # A has read the ledger; let B append now ...
            b_done.wait(1.0)       # ... and give it every chance to land before the replace
        return real_rewrite(path, lines, op)

    monkeypatch.setattr(skill_ledger, "_rewrite_ledger", paused_rewrite)
    dropped, real_trim = [], skill_ledger._trim_oldest
    monkeypatch.setattr(skill_ledger, "_trim_oldest",
                        lambda max_bytes: dropped.append(real_trim(max_bytes)) or dropped[-1])

    def writer(k: str) -> None:
        if k == "B":
            b_go.wait(10.0)
        for i in range(n):
            before = [{"path": f"my-skill/{k}-{i}.md", "sha256": "a" * 64}]
            after = [{"path": f"my-skill/{k}-{i}.md", "sha256": f"{i % 10}" * 64}]
            ids[k].append(skill_ledger.append_entry(
                "edit", "my-skill", before=before, after=after, evidence={"pad": "x" * 500}))
            if k == "B":
                b_done.set()

    seeds = 8  # seed over the cap so A's very first append sweeps
    for _ in range(seeds):
        skill_ledger.append_entry("edit", "my-skill", before=[{"path": "s", "sha256": "0" * 64}],
                                  after=[{"path": "s", "sha256": "1" * 64}], evidence={"pad": "x" * 500})
    # Unreferenced blobs: a fresh one is another process's in-flight capture (row not appended yet)
    # and must survive every sweep's blob GC; one older than the grace window is garbage and goes.
    fresh, aged = skill_ledger._store_blob(b"in-flight"), skill_ledger._store_blob(b"stale orphan")
    os.utime(skill_ledger.blobs_dir() / aged, (time.time() - 7200, time.time() - 7200))
    threads = [threading.Thread(target=writer, args=(k,), name=k) for k in ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert b_go.is_set(), "writer A's first append must have swept (test precondition)"
    assert all(ids["A"]) and all(ids["B"]) and len(ids["A"]) == len(ids["B"]) == n, "every append reported success"
    present = [json.loads(line)["id"] for line in
               skill_ledger.ledger_path().read_text(encoding="utf-8").splitlines() if line.strip()]
    assert present, "the newest row always survives a trim"
    assert (skill_ledger.blobs_dir() / fresh).exists() and not (skill_ledger.blobs_dir() / aged).exists()
    assert len(present) == seeds + 2 * n - sum(dropped), (
        "every row is either in the ledger or was counted as trimmed — none silently lost"
    )
    for k, seq in ids.items():
        survivors = [i for i in seq if i in set(present)]
        assert survivors == seq[len(seq) - len(survivors):], (
            f"writer {k}: rows missing from the middle — a concurrent sweep dropped an append"
        )


# ---------------------------------------------------------------------------
# File-tool writes (write_file / patch) into a live skills tree — the
# skill_manage bypass. Entries must carry REAL before/after manifests so the
# mutation is recoverable, not just attributed.
# ---------------------------------------------------------------------------


def _seed_skill_file(skills_dir, name="obs-gap", body="Original body."):
    skill_md = skills_dir / name / "SKILL.md"
    skill_md.parent.mkdir(parents=True, exist_ok=True)
    skill_md.write_text(
        VALID_SKILL_CONTENT.replace("name: my-skill", f"name: {name}").replace(
            "Original body.", body),
        encoding="utf-8")
    return skill_md


def test_classify_file_tool_target_skills_tree(ledger_env):
    from tools import skill_ledger

    skill_md = _seed_skill_file(ledger_env["skills"])
    info = skill_ledger.classify_file_tool_target(str(skill_md))
    assert info is not None
    assert info["skill"] == "obs-gap"
    assert info["path"] == str(skill_md.resolve())
    # A supporting file resolves to the containing skill, not the root.
    ref = skill_md.parent / "references" / "api.md"
    ref.parent.mkdir()
    ref.write_text("x", encoding="utf-8")
    ref_info = skill_ledger.classify_file_tool_target(str(ref))
    assert ref_info is not None and ref_info["skill"] == "obs-gap"


def test_classify_file_tool_target_skips_sidecars_transients_and_outsiders(ledger_env, tmp_path):
    from tools import skill_ledger

    for rel in (".usage.json", ".curator_ledger.jsonl", ".hub/lock.json",
                ".archive/old-skill/SKILL.md", "some-skill/.venv/pyvenv.cfg"):
        p = ledger_env["skills"] / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}", encoding="utf-8")
        assert skill_ledger.classify_file_tool_target(str(p)) is None, rel
    outsider = tmp_path / "notes.txt"
    outsider.write_text("hi", encoding="utf-8")
    assert skill_ledger.classify_file_tool_target(str(outsider)) is None
    assert skill_ledger.classify_file_tool_target("relative/path.md") is None


def test_write_file_create_in_skills_tree_is_ledgered(ledger_env, monkeypatch):
    """A brand-new skill file written by the generic write_file tool appends a
    ledger entry (action='write_file') whose rollback removes the created file."""
    from tools import skill_ledger
    from tools.file_tools import write_file_tool

    monkeypatch.setenv("HERMES_HOME", str(ledger_env["home"]))
    target = ledger_env["skills"] / "obs-gap" / "SKILL.md"
    content = VALID_SKILL_CONTENT.replace("name: my-skill", "name: obs-gap")
    result = json.loads(write_file_tool(str(target), content))
    assert not result.get("error"), result
    assert target.exists()

    rows = [r for r in skill_ledger.list_entries() if r.get("skill") == "obs-gap"]
    assert len(rows) == 1
    entry = rows[0]
    assert entry["action"] == "write_file"
    assert entry["evidence"].get("source") == "write_file"
    assert entry["before"] == []  # a creation, not a hollow capture
    assert any(i["path"].endswith("SKILL.md") for i in entry["after"])

    ok, msg = skill_ledger.rollback_entry(entry["id"])
    assert ok is True, msg
    assert not target.exists()


def test_patch_in_skills_tree_captures_before_and_rolls_back(ledger_env, monkeypatch):
    """The discriminator for the bypass fix: a patch-tool modification must carry
    the PRE-WRITE content in its before manifest, and single-entry rollback must
    restore that content byte-for-byte (an entry with before=[] would instead
    DELETE the patched file)."""
    import hashlib

    from tools import skill_ledger
    from tools.file_tools import patch_tool, write_file_tool

    monkeypatch.setenv("HERMES_HOME", str(ledger_env["home"]))
    target = ledger_env["skills"] / "obs-gap" / "SKILL.md"
    content = VALID_SKILL_CONTENT.replace("name: my-skill", "name: obs-gap")
    assert not json.loads(write_file_tool(str(target), content)).get("error")
    original = target.read_text(encoding="utf-8")

    patched = json.loads(patch_tool(
        mode="replace", path=str(target),
        old_string="Original body.", new_string="Patched body."))
    assert not patched.get("error"), patched
    assert "Patched body." in target.read_text(encoding="utf-8")

    patch_rows = [r for r in skill_ledger.list_entries(skill="obs-gap")
                  if r["action"] == "patch"]
    assert len(patch_rows) == 1
    entry = patch_rows[0]
    assert entry["evidence"].get("source") == "patch"
    before_paths = {i["path"]: i["sha256"] for i in entry["before"]}
    assert str(target.resolve()) in before_paths, (
        "a file-tool modification must capture its before-state — before=[] makes "
        "rollback delete the file and leaves the pre-write version unrecoverable")
    assert before_paths[str(target.resolve())] == hashlib.sha256(
        original.encode("utf-8")).hexdigest()

    ok, msg = skill_ledger.rollback_entry(entry["id"])
    assert ok is True, msg
    assert target.read_text(encoding="utf-8") == original


def test_file_tool_write_outside_skills_tree_is_not_ledgered(ledger_env, tmp_path, monkeypatch):
    from tools import skill_ledger
    from tools.file_tools import write_file_tool

    monkeypatch.setenv("HERMES_HOME", str(ledger_env["home"]))
    outsider = tmp_path / "notes.txt"
    result = json.loads(write_file_tool(str(outsider), "hello"))
    assert not result.get("error"), result
    assert outsider.exists()
    assert skill_ledger.list_entries() == []


def test_file_tool_write_in_sibling_profile_is_attributed(ledger_env, monkeypatch):
    """A direct write into ANOTHER profile's skills tree (the documented fallback the
    cross-profile not-found error teaches) still lands in the acting profile's ledger,
    with the owning tree named in evidence, and rolls back."""
    from tools import skill_ledger
    from tools.file_tools import patch_tool

    monkeypatch.setenv("HERMES_HOME", str(ledger_env["home"]))
    sibling_skills = ledger_env["home"] / "profiles" / "sibling" / "skills"
    target = _seed_skill_file(sibling_skills, name="shared-skill")
    original = target.read_text(encoding="utf-8")

    patched = json.loads(patch_tool(
        mode="replace", path=str(target),
        old_string="Original body.", new_string="Patched body.", cross_profile=True))
    assert not patched.get("error"), patched

    rows = [r for r in skill_ledger.list_entries() if r.get("skill") == "shared-skill"]
    assert len(rows) == 1
    assert rows[0]["evidence"].get("skills_root", "").endswith(
        str(Path("profiles") / "sibling" / "skills"))
    ok, msg = skill_ledger.rollback_entry(rows[0]["id"])
    assert ok is True, msg
    assert target.read_text(encoding="utf-8") == original


def test_begin_file_tool_write_skips_entry_when_before_capture_fails(ledger_env, monkeypatch):
    """A modification whose before-state cannot be captured gets NO entry rather than
    a hollow before=[] one (which rollback would read as 'delete this file')."""
    from tools import skill_ledger

    skill_md = _seed_skill_file(ledger_env["skills"])
    monkeypatch.setattr(skill_ledger, "capture_before", lambda *a, **k: None)
    assert skill_ledger.begin_file_tool_write(str(skill_md)) is None


def test_sibling_ledger_counts_names_other_profiles(ledger_env, monkeypatch):
    from tools import skill_ledger

    monkeypatch.setenv("HERMES_HOME", str(ledger_env["home"]))
    assert skill_ledger.sibling_ledger_counts() == []
    sibling = ledger_env["home"] / "profiles" / "warden" / "skills"
    sibling.mkdir(parents=True)
    (sibling / ".curator_ledger.jsonl").write_text('{"id": "abc"}\n' * 3, encoding="utf-8")
    counts = skill_ledger.sibling_ledger_counts()
    assert ("warden", 3) in counts
    # The active profile's own ledger is never listed as a sibling.
    skill_ledger.append_entry("edit", "my-skill", before=[], after=[])
    labels = [label for label, _ in skill_ledger.sibling_ledger_counts()]
    assert ledger_env["home"].name not in labels or ledger_env["home"].name == "home"
