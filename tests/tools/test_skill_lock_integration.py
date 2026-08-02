"""Cross-process contracts at the real skill-mutation entry points.

The primitive tests in ``tests/agent/test_skill_lock.py`` show the locks work.
These show the writers actually *take* them — the gap that let a bundled-skill
sync rmtree a directory while ``skill_manage`` was rewriting a file inside it.

Each test runs the second actor in a spawned process, because the locks are
advisory OS locks: two threads in one process would be short-circuited by the
in-process re-entrancy guards and prove nothing.
"""

from __future__ import annotations

import json
import multiprocessing
import os
from pathlib import Path

import pytest

SKILL_MD = """---
name: alpha
description: Use when testing locks. Does nothing.
---

# Alpha

Step 1. Do nothing.
"""


def _make_profile(home: Path) -> None:
    skill_dir = home / "skills" / "alpha"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")


def _child_env(home: Path) -> None:
    os.environ["HERMES_HOME"] = str(home)
    # Internal bridge (not a user-facing setting): keep a contended run
    # bounded without writing a config file into the temp profile.
    os.environ["HERMES_INTERNAL_SKILL_LOCK_TIMEOUT"] = "0.25"


# --- child entry points (module level: required by the spawn start method) ---


def _run_sync_skills(home: str, queue) -> None:
    _child_env(Path(home))
    from tools.skills_sync import sync_skills

    result = sync_skills(quiet=True)
    queue.put(bool(result.get("skipped_locked")))


def _run_skill_patch(home: str, queue) -> None:
    _child_env(Path(home))
    from tools.skill_manager_tool import skill_manage

    raw = skill_manage(
        action="patch",
        name="alpha",
        old_string="Step 1. Do nothing.",
        new_string="Step 1. Do nothing, carefully.",
    )
    queue.put(raw)


def _hold_namespace_lock(home: str, ready, release) -> None:
    """Hold the exclusive namespace lock, standing in for an in-flight sync."""
    _child_env(Path(home))
    from agent.skill_lock import skills_namespace_lock

    with skills_namespace_lock(exclusive=True, timeout=5.0):
        ready.put("held")
        release.get(timeout=10)


@pytest.fixture
def profile(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    _make_profile(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_INTERNAL_SKILL_LOCK_TIMEOUT", "0.25")
    return home


def test_sync_defers_while_a_skill_write_is_in_flight(profile):
    """The reported race: a structural sync must not run mid-``skill_manage``.

    The manager holds the namespace shared plus the per-skill lock for the
    whole lookup-to-write interval; ``sync_skills`` needs it exclusively, so it
    reports ``skipped_locked`` and retries on the next startup pass instead of
    moving the directory out from under the writer.
    """
    from agent.skill_lock import skill_write_lock, skills_namespace_lock

    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    with skills_namespace_lock(exclusive=False):
        with skill_write_lock(profile / "skills" / "alpha"):
            child = context.Process(target=_run_sync_skills, args=(str(profile), queue))
            child.start()
            assert queue.get(timeout=30) is True
            child.join(timeout=30)
    assert child.exitcode == 0


def test_skill_manage_reports_busy_while_a_structural_pass_holds_the_lock(profile):
    """The reverse direction: an agent write waits on a structural writer.

    It must surface as a clean tool error, not a partial write against a tree
    another process is restructuring.
    """
    context = multiprocessing.get_context("spawn")
    ready, release = context.Queue(), context.Queue()
    holder = context.Process(
        target=_hold_namespace_lock, args=(str(profile), ready, release)
    )
    holder.start()
    try:
        assert ready.get(timeout=30) == "held"

        from tools.skill_manager_tool import skill_manage

        raw = skill_manage(
            action="patch",
            name="alpha",
            old_string="Step 1. Do nothing.",
            new_string="Step 1. Do nothing, carefully.",
        )
        payload = json.loads(raw)
        assert payload.get("success") is False
        assert "busy" in payload.get("error", "").lower()
    finally:
        release.put("go")
        holder.join(timeout=30)

    # The contended write must not have landed.
    assert "carefully" not in (profile / "skills" / "alpha" / "SKILL.md").read_text()


def _run_curator_rollback(home: str, queue) -> None:
    _child_env(Path(home))
    from agent.curator_backup import rollback

    ok, msg, _ = rollback()
    queue.put((ok, msg))


def test_curator_rollback_refuses_while_a_skill_write_is_in_flight(profile):
    """The most destructive writer must not empty the tree mid-write.

    ``rollback`` moves every top-level entry out of ``skills/`` and extracts an
    archive over it. Without the lock it would do that while ``skill_manage``
    holds a skill open, silently discarding the write.
    """
    from agent.curator_backup import snapshot_skills
    from agent.skill_lock import skill_write_lock, skills_namespace_lock

    # A restorable snapshot must exist, or rollback bails out early on "no
    # matching backup" and the test would pass without the lock doing anything.
    assert snapshot_skills(reason="test") is not None
    skill_md = profile / "skills" / "alpha" / "SKILL.md"
    skill_md.write_text(SKILL_MD.replace("Do nothing.", "Do something."), encoding="utf-8")

    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    with skills_namespace_lock(exclusive=False):
        with skill_write_lock(profile / "skills" / "alpha"):
            child = context.Process(
                target=_run_curator_rollback, args=(str(profile), queue)
            )
            child.start()
            ok, msg = queue.get(timeout=30)
            child.join(timeout=30)

    assert ok is False
    assert "busy" in msg.lower()
    # Without the lock the rollback would restore the snapshot over the tree
    # and this post-snapshot edit would be gone.
    assert "Do something." in skill_md.read_text()


def test_skill_manage_patch_still_succeeds_uncontended(profile):
    """Guard the happy path: the locking wrapper must not break normal writes."""
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    child = context.Process(target=_run_skill_patch, args=(str(profile), queue))
    child.start()
    payload = json.loads(queue.get(timeout=30))
    child.join(timeout=30)

    assert payload.get("success") is True, payload
    assert "carefully" in (profile / "skills" / "alpha" / "SKILL.md").read_text()
