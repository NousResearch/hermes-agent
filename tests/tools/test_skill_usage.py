"""Tests for tools/skill_usage.py — sidecar telemetry + provenance filtering."""

import json
import multiprocessing as mp
import os
from pathlib import Path

import pytest


def _bump_view_many(hermes_home: str, skill_name: str, iterations: int) -> None:
    os.environ["HERMES_HOME"] = hermes_home
    from tools.skill_usage import bump_view

    for _ in range(iterations):
        bump_view(skill_name)


@pytest.fixture
def skills_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with a clean skills/ dir for each test.

    Pins ``curator.prune_builtins`` OFF so the bundled/hub-protection tests in
    this module exercise the off-path semantics regardless of the shipped
    default. Tests that want built-ins to be curation-eligible flip it back on
    explicitly via ``monkeypatch.setattr(mod, "_prune_builtins_enabled", ...)``.
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "skills").mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Force skill_usage module to re-resolve paths per test
    import importlib
    import tools.skill_usage as mod
    importlib.reload(mod)
    monkeypatch.setattr(mod, "_prune_builtins_enabled", lambda: False)
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


def test_get_record_missing_returns_empty_record(skills_home):
    from tools.skill_usage import get_record
    rec = get_record("nonexistent")
    assert rec["use_count"] == 0
    assert rec["view_count"] == 0
    assert rec["state"] == "active"
    assert rec["pinned"] is False
    assert rec["archived_at"] is None


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


def test_bumps_do_not_corrupt_other_skills(skills_home):
    from tools.skill_usage import bump_view, bump_use, get_record
    bump_view("skill-a")
    bump_use("skill-b")
    bump_view("skill-a")
    assert get_record("skill-a")["view_count"] == 2
    assert get_record("skill-a")["use_count"] == 0
    assert get_record("skill-b")["use_count"] == 1


def test_concurrent_bump_view_preserves_all_updates(skills_home):
    from tools.skill_usage import get_record

    process_count = 6
    iterations = 25
    ctx = mp.get_context("spawn")
    processes = [
        ctx.Process(
            target=_bump_view_many,
            args=(str(skills_home), "shared-skill", iterations),
        )
        for _ in range(process_count)
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=20)

    for process in processes:
        assert process.exitcode == 0
    assert get_record("shared-skill")["view_count"] == process_count * iterations


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------

def test_set_state_active(skills_home):
    from tools.skill_usage import set_state, get_record, STATE_ACTIVE
    set_state("x", STATE_ACTIVE)
    assert get_record("x")["state"] == "active"


def test_restoring_from_archive_clears_timestamp(skills_home):
    from tools.skill_usage import set_state, get_record, STATE_ARCHIVED, STATE_ACTIVE
    set_state("x", STATE_ARCHIVED)
    assert get_record("x")["archived_at"] is not None
    set_state("x", STATE_ACTIVE)
    assert get_record("x")["archived_at"] is None


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


# ---------------------------------------------------------------------------
# Archive / restore
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Telemetry vs curation — usage is tracked for ALL skills; curation is not
# ---------------------------------------------------------------------------


def test_end_to_end_telemetry_tracked_but_lifecycle_refused(skills_home):
    """The combined guarantee under decoupled telemetry/curation:

    - Usage telemetry (view/use/patch) IS recorded for bundled & hub skills.
    - Lifecycle mutations (set_state, set_pinned, archive) are REFUSED for them
      (with pruning off, the fixture default), so no state/pinned/archived flag
      lands and the directories stay on disk.
    """
    from tools.skill_usage import (
        bump_view, bump_use, bump_patch, set_state, set_pinned,
        archive_skill, load_usage, STATE_ACTIVE, STATE_STALE, STATE_ARCHIVED,
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

    for name in ("bundled-one", "hub-one"):
        bump_view(name)
        bump_use(name)
        bump_patch(name)
        set_state(name, STATE_STALE)
        set_state(name, STATE_ARCHIVED)
        set_pinned(name, True)
        ok, _msg = archive_skill(name)
        assert not ok, f"archive_skill(\"{name}\") should refuse"

    data = load_usage()
    # Telemetry landed for both.
    for name in ("bundled-one", "hub-one"):
        assert name in data, f"{name} telemetry should be recorded"
        assert data[name]["view_count"] == 1
        assert data[name]["use_count"] == 1
        assert data[name]["patch_count"] == 1
        # But lifecycle mutators were refused — state stays the default, never
        # archived/stale/pinned, and created_by is never agent.
        assert data[name]["state"] == STATE_ACTIVE
        assert data[name]["archived_at"] is None
        assert data[name]["pinned"] is False
        assert data[name].get("created_by") != "agent"

    # Directories must still be in place on disk.
    assert (skills_dir / "bundled-one" / "SKILL.md").exists()
    assert (skills_dir / "hub-one" / "SKILL.md").exists()

    # The agent-created skill can still be mutated normally.
    bump_view("mine")
    assert load_usage()["mine"]["view_count"] == 1


# ---------------------------------------------------------------------------
# Unmanaged enumeration + adoption
#
# A skill only becomes curator-managed when ``created_by: agent`` lands on its
# usage record, and that only happens for background-review creations. Records
# written before the marker existed carry no key at all, and every foreground
# `skill_manage(create)` leaves it unset — both are curation-eligible yet
# invisible to every automatic transition. These tests pin the contract that
# the blind spot is enumerable and that adoption is an explicit declaration:
# never inferred from telemetry, never silently reached by the curator.
# ---------------------------------------------------------------------------

def _seed_usage(skills_dir: Path, records: dict) -> None:
    (skills_dir / ".usage.json").write_text(
        json.dumps(records, indent=1), encoding="utf-8"
    )


def test_adopt_preserves_the_inactivity_clock(skills_home):
    """Adoption must not reset staleness — it hands over an EXISTING history.

    If adopting re-anchored the clock to now, every legacy skill would buy a
    fresh archive_after_days window, which is the opposite of what the user
    wants when they hand over a library they already stopped using.
    """
    from tools.skill_usage import adopt_skill, get_record, latest_activity_at

    skills_dir = skills_home / "skills"
    _write_skill(skills_dir, "legacy")
    _seed_usage(skills_dir, {
        "legacy": {
            "use_count": 5,
            "patch_count": 7,
            "last_used_at": "2026-04-29T00:00:00+00:00",
            "created_at": "2026-04-28T00:00:00+00:00",
        }
    })
    before = latest_activity_at(get_record("legacy"))

    ok, _msg = adopt_skill("legacy")
    assert ok is True
    rec = get_record("legacy")
    assert latest_activity_at(rec) == before
    assert rec["use_count"] == 5
    assert rec["patch_count"] == 7


@pytest.mark.parametrize("kind", ["bundled", "hub", "protected", "missing"])
def test_adopt_refuses_skills_the_user_does_not_own(skills_home, monkeypatch, kind):
    """Adoption writes a provenance claim, so it must refuse anything with an
    external owner rather than stamping a lie onto the record.

    ``prune_builtins`` is forced ON here — the shipped default — because that
    is the configuration in which a bundled skill is otherwise curation-
    eligible. With it off, ``mark_agent_created``'s own eligibility gate would
    block the write and this test would pass without exercising adopt's guard
    at all.
    """
    from tools import skill_usage
    from tools.skill_usage import adopt_skill, load_usage

    monkeypatch.setattr(skill_usage, "_prune_builtins_enabled", lambda: True)

    skills_dir = skills_home / "skills"
    if kind == "bundled":
        name = "bundled-one"
        _write_skill(skills_dir, name)
        (skills_dir / ".bundled_manifest").write_text(f"{name}:abc\n", encoding="utf-8")
    elif kind == "hub":
        name = "hub-one"
        _write_skill(skills_dir, name)
        hub = skills_dir / ".hub"
        hub.mkdir()
        (hub / "lock.json").write_text(
            json.dumps({"installed": {name: {}}}), encoding="utf-8",
        )
    elif kind == "protected":
        name = sorted(skill_usage.PROTECTED_BUILTIN_SKILLS)[0]
        _write_skill(skills_dir, name)
    else:
        name = "no-such-skill"

    ok, _msg = adopt_skill(name)
    assert ok is False
    assert load_usage().get(name, {}).get("created_by") != "agent"


def test_adopt_rejects_empty_name(skills_home):
    from tools.skill_usage import adopt_skill

    assert adopt_skill("")[0] is False

