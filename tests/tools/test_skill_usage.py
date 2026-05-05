"""Tests for tools/skill_usage.py — sidecar telemetry + provenance filtering."""

import json
import os
from pathlib import Path

import pytest


@pytest.fixture
def skills_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with a clean skills/ dir for each test."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "skills").mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Force skill_usage module to re-resolve paths per test
    import importlib
    import tools.skill_usage as mod
    importlib.reload(mod)
    return home


def _write_skill(skills_dir: Path, name: str, category: str = ""):
    """Create a minimal SKILL.md with a name: frontmatter field."""
    if category:
        d = skills_dir / category / name
    else:
        d = skills_dir / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(
        f"""---
name: {name}
description: test skill
---

# body
""",
        encoding="utf-8",
    )
    return d


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------

def test_empty_usage_returns_empty_dict(skills_home):
    from tools.skill_usage import load_usage
    assert load_usage() == {}


def test_save_and_load_roundtrip(skills_home):
    from tools.skill_usage import load_usage, save_usage
    data = {"skill-a": {"use_count": 3, "state": "active"}}
    save_usage(data)
    loaded = load_usage()
    assert loaded["skill-a"]["use_count"] == 3
    assert loaded["skill-a"]["state"] == "active"


def test_save_is_atomic_no_partial_tmp_files(skills_home):
    from tools.skill_usage import save_usage, _usage_file
    save_usage({"x": {"use_count": 1}})
    skills_dir = _usage_file().parent
    # No leftover tempfile
    for p in skills_dir.iterdir():
        assert not p.name.startswith(".usage_"), f"leftover tmp: {p.name}"


def test_get_record_missing_returns_empty_record(skills_home):
    from tools.skill_usage import get_record
    rec = get_record("nonexistent")
    assert rec["use_count"] == 0
    assert rec["view_count"] == 0
    assert rec["state"] == "active"
    assert rec["pinned"] is False
    assert rec["archived_at"] is None


def test_get_record_backfills_missing_keys(skills_home):
    from tools.skill_usage import get_record, save_usage
    save_usage({"legacy": {"use_count": 5}})  # old-format record
    rec = get_record("legacy")
    assert rec["use_count"] == 5
    assert "view_count" in rec  # backfilled
    assert "state" in rec


def test_load_usage_handles_corrupt_file(skills_home):
    from tools.skill_usage import load_usage, _usage_file
    _usage_file().write_text("{ not json }", encoding="utf-8")
    assert load_usage() == {}


# ---------------------------------------------------------------------------
# Counter bumps
# ---------------------------------------------------------------------------

def test_bump_view_increments_and_timestamps(skills_home):
    from tools.skill_usage import bump_view, get_record
    bump_view("my-skill")
    bump_view("my-skill")
    rec = get_record("my-skill")
    assert rec["view_count"] == 2
    assert rec["last_viewed_at"] is not None


def test_bump_use_increments_and_timestamps(skills_home):
    from tools.skill_usage import bump_use, get_record
    bump_use("my-skill")
    rec = get_record("my-skill")
    assert rec["use_count"] == 1
    assert rec["last_used_at"] is not None


def test_bump_patch_increments_and_timestamps(skills_home):
    from tools.skill_usage import bump_patch, get_record
    bump_patch("my-skill")
    rec = get_record("my-skill")
    assert rec["patch_count"] == 1
    assert rec["last_patched_at"] is not None


def test_bump_on_empty_name_is_noop(skills_home):
    from tools.skill_usage import bump_view, load_usage
    bump_view("")
    assert load_usage() == {}


def test_bumps_do_not_corrupt_other_skills(skills_home):
    from tools.skill_usage import bump_view, bump_use, get_record
    bump_view("skill-a")
    bump_use("skill-b")
    bump_view("skill-a")
    assert get_record("skill-a")["view_count"] == 2
    assert get_record("skill-a")["use_count"] == 0
    assert get_record("skill-b")["use_count"] == 1


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------

def test_set_state_active(skills_home):
    from tools.skill_usage import set_state, get_record, STATE_ACTIVE
    set_state("x", STATE_ACTIVE)
    assert get_record("x")["state"] == "active"


def test_set_state_archived_records_timestamp(skills_home):
    from tools.skill_usage import set_state, get_record, STATE_ARCHIVED
    set_state("x", STATE_ARCHIVED)
    rec = get_record("x")
    assert rec["state"] == "archived"
    assert rec["archived_at"] is not None


def test_set_state_invalid_is_noop(skills_home):
    from tools.skill_usage import set_state, get_record
    set_state("x", "bogus")
    # No record created for invalid state
    rec = get_record("x")
    assert rec["state"] == "active"  # default


def test_restoring_from_archive_clears_timestamp(skills_home):
    from tools.skill_usage import set_state, get_record, STATE_ARCHIVED, STATE_ACTIVE
    set_state("x", STATE_ARCHIVED)
    assert get_record("x")["archived_at"] is not None
    set_state("x", STATE_ACTIVE)
    assert get_record("x")["archived_at"] is None


def test_set_pinned(skills_home):
    from tools.skill_usage import set_pinned, get_record
    set_pinned("x", True)
    assert get_record("x")["pinned"] is True
    set_pinned("x", False)
    assert get_record("x")["pinned"] is False


def test_forget_removes_record(skills_home):
    from tools.skill_usage import bump_view, forget, load_usage
    bump_view("x")
    assert "x" in load_usage()
    forget("x")
    assert "x" not in load_usage()


# ---------------------------------------------------------------------------
# Provenance filter — the load-bearing safety check
# ---------------------------------------------------------------------------

def test_agent_created_excludes_bundled(skills_home):
    from tools.skill_usage import list_agent_created_skill_names, mark_agent_created
    skills_dir = skills_home / "skills"
    _write_skill(skills_dir, "bundled-skill", category="github")
    _write_skill(skills_dir, "my-skill")
    mark_agent_created("my-skill")
    # Seed a bundled manifest marking bundled-skill as upstream
    (skills_dir / ".bundled_manifest").write_text(
        "bundled-skill:abc123\n", encoding="utf-8",
    )
    names = list_agent_created_skill_names()
    assert "my-skill" in names
    assert "bundled-skill" not in names


def test_agent_created_excludes_hub_installed(skills_home):
    from tools.skill_usage import list_agent_created_skill_names, mark_agent_created
    skills_dir = skills_home / "skills"
    _write_skill(skills_dir, "hub-skill")
    _write_skill(skills_dir, "my-skill")
    mark_agent_created("my-skill")
    hub_dir = skills_dir / ".hub"
    hub_dir.mkdir()
    (hub_dir / "lock.json").write_text(
        json.dumps({"version": 1, "installed": {"hub-skill": {"source": "taps/main"}}}),
        encoding="utf-8",
    )
    names = list_agent_created_skill_names()
    assert "my-skill" in names
    assert "hub-skill" not in names


def test_agent_created_excludes_hub_installed_frontmatter_name(skills_home):
    from tools.skill_usage import (
        is_agent_created,
        list_agent_created_skill_names,
        mark_agent_created,
    )

    skills_dir = skills_home / "skills"
    hub_skill = skills_dir / "productivity" / "getnote"
    hub_skill.mkdir(parents=True)
    (hub_skill / "SKILL.md").write_text(
        """---
name: Get笔记
description: test skill
---

# body
""",
        encoding="utf-8",
    )
    _write_skill(skills_dir, "my-skill")
    mark_agent_created("my-skill")
    hub_dir = skills_dir / ".hub"
    hub_dir.mkdir()
    (hub_dir / "lock.json").write_text(
        json.dumps(
            {
                "version": 1,
                "installed": {
                    "getnote": {
                        "source": "taps/main",
                        "install_path": "productivity/getnote",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    names = list_agent_created_skill_names()
    assert "my-skill" in names
    assert "Get笔记" not in names
    assert is_agent_created("Get笔记") is False
    assert is_agent_created("getnote") is False


def test_is_agent_created(skills_home):
    from tools.skill_usage import is_agent_created
    skills_dir = skills_home / "skills"
    (skills_dir / ".bundled_manifest").write_text("bundled:abc\n", encoding="utf-8")
    hub_dir = skills_dir / ".hub"
    hub_dir.mkdir()
    (hub_dir / "lock.json").write_text(
        json.dumps({"installed": {"hubbed": {}}}), encoding="utf-8",
    )
    assert is_agent_created("my-skill") is True
    assert is_agent_created("bundled") is False
    assert is_agent_created("hubbed") is False


def test_agent_created_skips_archive_and_hub_dirs(skills_home):
    from tools.skill_usage import list_agent_created_skill_names, mark_agent_created
    skills_dir = skills_home / "skills"
    _write_skill(skills_dir, "real-skill")
    mark_agent_created("real-skill")
    # Dot-prefixed dirs must be ignored even if they contain SKILL.md
    archive = skills_dir / ".archive" / "old-skill"
    archive.mkdir(parents=True)
    (archive / "SKILL.md").write_text(
        "---\nname: old-skill\n---\n", encoding="utf-8",
    )
    names = list_agent_created_skill_names()
    assert "real-skill" in names
    assert "old-skill" not in names


# ---------------------------------------------------------------------------
# Archive / restore
# ---------------------------------------------------------------------------

def test_archive_skill_moves_directory(skills_home):
    from tools.skill_usage import archive_skill, get_record, STATE_ARCHIVED
    skills_dir = skills_home / "skills"
    skill_dir = _write_skill(skills_dir, "old-skill")
    assert skill_dir.exists()

    ok, msg = archive_skill("old-skill")
    assert ok, msg
    assert not skill_dir.exists()
    assert (skills_dir / ".archive" / "old-skill" / "SKILL.md").exists()
    assert get_record("old-skill")["state"] == "archived"
    assert get_record("old-skill")["archived_at"] is not None


def test_archive_refuses_bundled_skill(skills_home):
    from tools.skill_usage import archive_skill
    skills_dir = skills_home / "skills"
    _write_skill(skills_dir, "bundled")
    (skills_dir / ".bundled_manifest").write_text("bundled:abc\n", encoding="utf-8")

    ok, msg = archive_skill("bundled")
    assert not ok
    assert "bundled" in msg.lower() or "hub" in msg.lower()


def test_archive_refuses_hub_skill(skills_home):
    from tools.skill_usage import archive_skill
    skills_dir = skills_home / "skills"
    _write_skill(skills_dir, "hub-skill")
    hub_dir = skills_dir / ".hub"
    hub_dir.mkdir()
    (hub_dir / "lock.json").write_text(
        json.dumps({"installed": {"hub-skill": {}}}), encoding="utf-8",
    )

    ok, msg = archive_skill("hub-skill")
    assert not ok


def test_archive_missing_skill_returns_error(skills_home):
    from tools.skill_usage import archive_skill
    ok, msg = archive_skill("nonexistent")
    assert not ok
    assert "not found" in msg.lower()


def test_restore_skill_moves_back(skills_home):
    from tools.skill_usage import archive_skill, restore_skill, get_record
    skills_dir = skills_home / "skills"
    _write_skill(skills_dir, "temp-skill")
    archive_skill("temp-skill")
    assert not (skills_dir / "temp-skill").exists()

    ok, msg = restore_skill("temp-skill")
    assert ok, msg
    assert (skills_dir / "temp-skill" / "SKILL.md").exists()
    assert get_record("temp-skill")["state"] == "active"


def test_restore_skill_finds_nested_archive_subdir(skills_home):
    """Skills archived under nested category subdirs (e.g.
    .archive/<category>/<skill>/) — left behind by older archive layouts or
    external imports — must still be restorable by name."""
    from tools.skill_usage import restore_skill, get_record
    skills_dir = skills_home / "skills"
    nested = skills_dir / ".archive" / "openclaw-imports" / "nested-skill"
    nested.mkdir(parents=True)
    (nested / "SKILL.md").write_text(
        "---\nname: nested-skill\ndescription: x\n---\n", encoding="utf-8",
    )

    ok, msg = restore_skill("nested-skill")
    assert ok, msg
    assert (skills_dir / "nested-skill" / "SKILL.md").exists()
    assert not nested.exists()
    assert get_record("nested-skill")["state"] == "active"


def test_restore_skill_finds_nested_timestamped_prefix(skills_home):
    """Prefix-match path (timestamped dupes) must also descend into nested
    archive subdirs, not just .archive/ top-level."""
    from tools.skill_usage import restore_skill
    skills_dir = skills_home / "skills"
    nested = skills_dir / ".archive" / "imports" / "dup-skill-20260101000000"
    nested.mkdir(parents=True)
    (nested / "SKILL.md").write_text(
        "---\nname: dup-skill\ndescription: x\n---\n", encoding="utf-8",
    )

    ok, msg = restore_skill("dup-skill")
    assert ok, msg
    assert (skills_dir / "dup-skill" / "SKILL.md").exists()


def test_archive_collision_gets_suffix(skills_home):
    from tools.skill_usage import archive_skill
    skills_dir = skills_home / "skills"
    _write_skill(skills_dir, "dup")
    archive_skill("dup")
    _write_skill(skills_dir, "dup")  # recreate
    ok, msg = archive_skill("dup")
    assert ok
    # Two entries under .archive/ — second should have a timestamp suffix
    archived = sorted(p.name for p in (skills_dir / ".archive").iterdir() if p.is_dir())
    assert "dup" in archived
    assert any(n.startswith("dup-") and n != "dup" for n in archived)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def test_agent_created_report_includes_marked_skills_with_defaults(skills_home):
    from tools.skill_usage import agent_created_report, bump_view, mark_agent_created
    skills_dir = skills_home / "skills"
    _write_skill(skills_dir, "a")
    _write_skill(skills_dir, "b")
    mark_agent_created("a")
    mark_agent_created("b")
    bump_view("a")
    rows = agent_created_report()
    by_name = {r["name"]: r for r in rows}
    assert "a" in by_name and "b" in by_name
    assert by_name["a"]["view_count"] == 1
    # b has only the provenance marker — activity fields still default.
    assert by_name["b"]["view_count"] == 0
    assert by_name["b"]["state"] == "active"


def test_manual_skill_with_usage_is_not_curator_managed(skills_home):
    from tools.skill_usage import agent_created_report, bump_view, list_agent_created_skill_names
    skills_dir = skills_home / "skills"
    _write_skill(skills_dir, "manual-skill")

    bump_view("manual-skill")

    assert "manual-skill" not in list_agent_created_skill_names()
    assert "manual-skill" not in {r["name"] for r in agent_created_report()}


def test_agent_created_report_excludes_bundled_and_hub(skills_home):
    from tools.skill_usage import agent_created_report, mark_agent_created
    skills_dir = skills_home / "skills"
    _write_skill(skills_dir, "mine")
    _write_skill(skills_dir, "bundled")
    _write_skill(skills_dir, "hubbed")
    mark_agent_created("mine")
    (skills_dir / ".bundled_manifest").write_text("bundled:abc\n", encoding="utf-8")
    hub = skills_dir / ".hub"
    hub.mkdir()
    (hub / "lock.json").write_text(
        json.dumps({"installed": {"hubbed": {}}}), encoding="utf-8",
    )
    names = {r["name"] for r in agent_created_report()}
    assert "mine" in names
    assert "bundled" not in names
    assert "hubbed" not in names


def test_agent_created_report_derives_activity_from_view_and_patch(skills_home, monkeypatch):
    import tools.skill_usage as skill_usage

    skills_dir = skills_home / "skills"
    _write_skill(skills_dir, "mine")
    timestamps = iter([
        "2026-04-30T10:00:00+00:00",
        "2026-04-30T11:00:00+00:00",
        "2026-04-30T12:00:00+00:00",
        "2026-04-30T13:00:00+00:00",
    ])
    monkeypatch.setattr(skill_usage, "_now_iso", lambda: next(timestamps))

    skill_usage.mark_agent_created("mine")
    skill_usage.bump_view("mine")
    skill_usage.bump_patch("mine")

    row = next(r for r in skill_usage.agent_created_report() if r["name"] == "mine")
    assert row["activity_count"] == 2
    assert row["last_activity_at"] == "2026-04-30T12:00:00+00:00"


# ---------------------------------------------------------------------------
# Provenance guard — telemetry must not leak records for bundled/hub skills
# ---------------------------------------------------------------------------

def test_bump_view_no_op_for_bundled_skill(skills_home):
    """Telemetry bumps on bundled skills are dropped — the sidecar must stay
    focused on agent-created skills only."""
    from tools.skill_usage import bump_view, load_usage
    skills_dir = skills_home / "skills"
    (skills_dir / ".bundled_manifest").write_text(
        "ship-bundled:abc\n", encoding="utf-8",
    )

    bump_view("ship-bundled")
    assert "ship-bundled" not in load_usage(), (
        "bundled skill leaked into .usage.json"
    )


def test_bump_patch_no_op_for_hub_skill(skills_home):
    from tools.skill_usage import bump_patch, load_usage
    skills_dir = skills_home / "skills"
    hub = skills_dir / ".hub"
    hub.mkdir()
    (hub / "lock.json").write_text(
        json.dumps({"installed": {"from-hub": {}}}), encoding="utf-8",
    )

    bump_patch("from-hub")
    assert "from-hub" not in load_usage()


def test_bump_use_no_op_for_hub_skill(skills_home):
    from tools.skill_usage import bump_use, load_usage
    skills_dir = skills_home / "skills"
    hub = skills_dir / ".hub"
    hub.mkdir()
    (hub / "lock.json").write_text(
        json.dumps({"installed": {"from-hub": {}}}), encoding="utf-8",
    )

    bump_use("from-hub")
    assert "from-hub" not in load_usage()


def test_set_state_no_op_for_bundled_skill(skills_home):
    """State transitions on bundled skills must not land in the sidecar."""
    from tools.skill_usage import set_state, load_usage, STATE_ARCHIVED
    skills_dir = skills_home / "skills"
    (skills_dir / ".bundled_manifest").write_text(
        "locked:abc\n", encoding="utf-8",
    )
    set_state("locked", STATE_ARCHIVED)
    assert "locked" not in load_usage()


def test_restore_refuses_to_shadow_bundled_skill(skills_home):
    """If a bundled skill now occupies the name, refuse to restore."""
    from tools.skill_usage import archive_skill, restore_skill
    skills_dir = skills_home / "skills"
    _write_skill(skills_dir, "shared-name")
    archive_skill("shared-name")

    # Now a bundled skill appears with the same name
    (skills_dir / ".bundled_manifest").write_text(
        "shared-name:abc\n", encoding="utf-8",
    )
    _write_skill(skills_dir, "shared-name")  # bundled install landed

    ok, msg = restore_skill("shared-name")
    assert not ok
    assert "bundled" in msg.lower() or "shadow" in msg.lower()


def test_end_to_end_no_code_path_mutates_bundled_skill(skills_home):
    """The combined guarantee: no curator code path can archive, mark stale,
    set-state, or persist telemetry for a bundled or hub-installed skill."""
    from tools.skill_usage import (
        bump_view, bump_use, bump_patch, set_state, set_pinned,
        archive_skill, load_usage, STATE_STALE, STATE_ARCHIVED,
    )
    skills_dir = skills_home / "skills"
    _write_skill(skills_dir, "bundled-one")
    _write_skill(skills_dir, "hub-one")
    _write_skill(skills_dir, "mine")

    (skills_dir / ".bundled_manifest").write_text(
        "bundled-one:abc\n", encoding="utf-8",
    )
    hub = skills_dir / ".hub"
    hub.mkdir()
    (hub / "lock.json").write_text(
        json.dumps({"installed": {"hub-one": {}}}), encoding="utf-8",
    )

    # Hammer every mutator at the bundled/hub names
    for name in ("bundled-one", "hub-one"):
        bump_view(name)
        bump_use(name)
        bump_patch(name)
        set_state(name, STATE_STALE)
        set_state(name, STATE_ARCHIVED)
        set_pinned(name, True)
        ok, _msg = archive_skill(name)
        assert not ok, f"archive_skill(\"{name}\") should refuse"

    # Sidecar must be clean of all three
    data = load_usage()
    assert "bundled-one" not in data
    assert "hub-one" not in data

    # Directories must still be in place on disk
    assert (skills_dir / "bundled-one" / "SKILL.md").exists()
    assert (skills_dir / "hub-one" / "SKILL.md").exists()

    # The agent-created skill can still be mutated normally
    bump_view("mine")
    assert load_usage()["mine"]["view_count"] == 1
