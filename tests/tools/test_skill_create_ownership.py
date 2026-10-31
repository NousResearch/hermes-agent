"""Ownership of what a ``skill_manage`` create/batch undoes: only what it PROVABLY published, nothing else.

``test_skill_create_cleanup_safety`` covers a rollback that sees another writer's file after the batch has
recorded its own. These tests pin the gaps left around that:

* the identity a rollback trusts comes from the actual publish, never from a later read (an editor that saves
  between "op returned" and "identity recorded" must not be mistaken for this batch's file);
* a no-hard-link fallback never deletes a file it cannot prove is its own, and the error names what it kept;
* a parent directory renamed after it was pinned is detected, not written into;
* a rollback restores only skills an executed op touched, never deletes a pre-existing ``*.rollback-broken``
  directory, and keeps (and names) any file in a restored skill that neither the snapshot nor the batch wrote;
* a rollback that could not finish never claims it did, nor calls the batch's own blocked file "someone else's";
* malformed ``file_path`` is a structured error, not an exception.

Every uncooperative writer is a separate OS process that takes no lock, every hook asserts it fired.
"""

from __future__ import annotations

import errno
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tools import skill_manager_tool as sm
from utils import atomic_create_text

NAME = "race-skill"
ROBOT = """\
---
name: race-skill
description: Use when the robot content must be published.
---

# Robot

Step 1: robot wrote this.
"""
OLD = ROBOT.replace("race-skill", "old-skill")
HUMAN = b"HUMAN SAVE - must survive\n"
BAD_OP = {"action": "write_file", "name": NAME, "file_path": "bad/nope.md", "file_content": "x"}  # always fails
WAIT = 20

_SAVE = ("import os, pathlib, sys; p = pathlib.Path(sys.argv[1]); t = p.with_name('editor.tmp'); "
         "t.write_bytes(sys.argv[2].encode()); os.replace(t, p)")  # an editor's atomic save
_WRITE_IN_PLACE = "import pathlib, sys; pathlib.Path(sys.argv[1]).write_bytes(sys.argv[2].encode())"
_MOVE_PARENT = "import pathlib, sys; p = pathlib.Path(sys.argv[1]); p.rename(sys.argv[2]); p.mkdir()"


@pytest.fixture
def skills(tmp_path, monkeypatch):
    home = tmp_path / "home"
    root = home / "skills"
    root.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(sm, "SKILLS_DIR", root)
    monkeypatch.setattr("agent.skill_utils.get_all_skills_dirs", lambda: [root])
    assert sm._skills_dir() == root and root.resolve().is_relative_to(tmp_path.resolve())
    return root


def _child(tmp_path: Path, script: str, *args) -> None:
    """Run ``script`` in an independent interpreter with its own isolated HOME/HERMES_HOME/TMPDIR."""
    env = {"PATH": os.defpath, "LC_ALL": "C.UTF-8"}
    for key in ("HOME", "HERMES_HOME", "TMPDIR"):
        directory = tmp_path / f"child-{key}"
        directory.mkdir(exist_ok=True)
        env[key] = str(directory)
    subprocess.run([sys.executable, "-I", "-c", script, *map(str, args)], env=env, check=True, timeout=WAIT)


def _batch(*ops: dict) -> dict:
    return json.loads(sm.apply_skill_pending({"operations": list(ops)}))


def _create_op(name: str = NAME, content: str = ROBOT) -> dict:
    return {"action": "create", "name": name, "content": content}


def _after_op(monkeypatch, action_name: str, hook, *, only_successful: bool = True, file_path=None) -> list:
    """Run ``hook()`` right after the first real ``skill_manage`` op of this action RETURNED."""
    fired: list = []
    real = sm._skill_manage_from

    def wrapper(payload, **kw):
        out = real(payload, **kw)
        if (payload.get("action") == action_name and not fired
                and (file_path is None or payload.get("file_path") == file_path)
                and (not only_successful or json.loads(out).get("success"))):
            hook()
            fired.append(True)
        return out

    monkeypatch.setattr(sm, "_skill_manage_from", wrapper)
    return fired


def _before_op(monkeypatch, action_name: str, hook, *, file_path=None) -> list:
    fired: list = []
    real = sm._skill_manage_from

    def wrapper(payload, **kw):
        if payload.get("action") == action_name and not fired and (file_path is None
                                                                  or payload.get("file_path") == file_path):
            hook()
            fired.append(True)
        return real(payload, **kw)

    monkeypatch.setattr(sm, "_skill_manage_from", wrapper)
    return fired


# --- 1. identity comes from the publish, not from a later read ----------------------------------------

@pytest.mark.platforms("posix")
def test_a_save_between_the_create_returning_and_the_batch_noting_it_is_not_deleted(skills, tmp_path, monkeypatch):
    skill_md = skills / NAME / "SKILL.md"
    fired = _after_op(monkeypatch, "create", lambda: _child(tmp_path, _SAVE, skill_md, HUMAN.decode()))
    result = _batch(_create_op(), BAD_OP)
    assert fired, "the editor never ran, this test proved nothing"
    assert result["success"] is False and result["failed_index"] == 1
    assert skill_md.read_bytes() == HUMAN  # the editor's file is not this batch's to delete
    assert "all touched skills rolled back" not in result["error"] and str(skill_md) in result["error"]


@pytest.mark.platforms("posix")
def test_a_save_after_a_supporting_file_write_returned_is_not_deleted_by_the_rollback(skills, tmp_path, monkeypatch):
    note = skills / NAME / "references" / "a.md"
    write = {"action": "write_file", "name": NAME, "file_path": "references/a.md", "file_content": "robot a"}
    fired = _after_op(monkeypatch, "write_file", lambda: _child(tmp_path, _SAVE, note, HUMAN.decode()),
                      file_path="references/a.md")
    result = _batch(_create_op(), write, BAD_OP)
    assert fired and result["success"] is False and result["failed_index"] == 2
    assert note.read_bytes() == HUMAN
    assert str(note) in result["error"]
    assert not (skills / NAME / "SKILL.md").exists()  # what the batch itself wrote is still undone


def test_rollback_still_removes_everything_the_batch_wrote_when_nobody_interferes(skills):
    write = {"action": "write_file", "name": NAME, "file_path": "references/a.md", "file_content": "robot a"}
    result = _batch(_create_op(), write, BAD_OP)
    assert result["success"] is False and result["error"].endswith("all touched skills rolled back.")
    assert not (skills / NAME).exists()


# --- 2. an error that cannot finish says so and owns its own blocked file ------------------------------

@pytest.mark.platforms("posix")
def test_a_batch_whose_blocked_file_cannot_be_removed_does_not_claim_a_rollback_or_blame_someone_else(skills,
                                                                                                    monkeypatch):
    skill_md = skills / NAME / "SKILL.md"
    monkeypatch.setattr(sm, "_security_scan_skill", lambda _d: "blocked")
    real_rename, denied = os.rename, []

    def deny(src, dst, *a, **kw):
        if os.path.basename(os.fspath(src)) == "SKILL.md":
            denied.append(True)
            raise PermissionError(errno.EACCES, "simulated cleanup denied")
        return real_rename(src, dst, *a, **kw)

    monkeypatch.setattr(os, "rename", deny)
    result = _batch(_create_op())
    assert denied, "no removal was attempted, this test proved nothing"
    assert skill_md.exists() and result["success"] is False and result["cleanup_incomplete"] is True
    error = result["error"]
    assert "ROLLBACK INCOMPLETE" in error and str(skill_md) in error
    assert "rolled back everything" not in error  # untrue: the blocked file is still there
    assert "did not write" not in error  # and it is this batch's own file, not another writer's


# --- 3. the no-hard-link fallback never deletes what it cannot prove is its own -----------------------

@pytest.fixture
def no_hard_links(monkeypatch):
    def unsupported(*a, **kw):
        raise OSError(errno.EPERM, "hard links not supported")

    monkeypatch.setattr(os, "link", unsupported)


def _fsync_fails_after(monkeypatch, target: Path, before_failure) -> list:
    real_fsync, fired = os.fsync, []

    def fsync(fd):
        try:
            hit = os.fstat(fd).st_ino == target.stat().st_ino
        except OSError:
            hit = False
        if hit and not fired:
            fired.append(True)
            before_failure()
            raise OSError(errno.EIO, "simulated fsync failure")
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fsync)
    return fired


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("saved", [ROBOT[:12], ROBOT[:-1], "HUMAN\n"])
def test_fallback_failure_keeps_a_file_an_independent_process_saved_in_place_whatever_its_bytes(
        tmp_path, no_hard_links, monkeypatch, saved):
    box = tmp_path / "box"
    box.mkdir()
    target = box / "SKILL.md"
    fired = _fsync_fails_after(monkeypatch, target, lambda: _child(tmp_path, _WRITE_IN_PLACE, target, saved))
    with pytest.raises(OSError) as raised:
        atomic_create_text(target, ROBOT)
    assert fired, "the in-place create never failed its fsync, this test proved nothing"
    kept = [p for p in box.iterdir() if p.read_bytes() == saved.encode()]
    assert len(kept) == 1 and len(list(box.iterdir())) == 1, [p.name for p in box.iterdir()]
    assert str(kept[0]) in "\n".join(getattr(raised.value, "__notes__", []))  # the error says where it is
    assert raised.value.preserved_at == str(kept[0])


@pytest.mark.platforms("posix")
def test_skill_create_error_names_the_file_the_fallback_kept(skills, tmp_path, no_hard_links, monkeypatch):
    target = skills / NAME / "SKILL.md"
    fired = _fsync_fails_after(monkeypatch, target, lambda: _child(tmp_path, _WRITE_IN_PLACE, target, ROBOT[:12]))
    result = sm._create_skill(NAME, ROBOT)
    assert fired and result["success"] is False
    (kept,) = [p for p in target.parent.iterdir() if p.read_bytes() == ROBOT[:12].encode()]
    assert str(kept) in result["error"] and result["preserved_at"] == str(kept)
    assert kept.name != "SKILL.md"  # a loader must not pick the unfinished text up as the skill
    assert sm._find_skill(NAME) is None


@pytest.mark.platforms("posix")
def test_fallback_failure_still_cleans_up_its_own_untouched_file(tmp_path, no_hard_links, monkeypatch):
    box = tmp_path / "box"
    box.mkdir()
    fired = _fsync_fails_after(monkeypatch, box / "SKILL.md", lambda: None)
    with pytest.raises(OSError):
        atomic_create_text(box / "SKILL.md", ROBOT)
    assert fired and list(box.iterdir()) == []  # byte-for-byte what we wrote: nothing to lose, nothing left


# --- 4. a parent directory renamed after it was pinned is detected -------------------------------------

@pytest.mark.platforms("posix")
def test_a_parent_renamed_just_before_publication_gets_no_file_and_the_error_says_so(tmp_path, monkeypatch):
    skill_dir, moved = tmp_path / "skill", tmp_path / "relocated"
    skill_dir.mkdir()
    real_link, fired = os.link, []

    def move_parent_then_link(src, dst, *a, **kw):
        if os.path.basename(os.fspath(dst)) == "SKILL.md" and not fired:
            fired.append(True)
            _child(tmp_path, _MOVE_PARENT, skill_dir, moved)
        return real_link(src, dst, *a, **kw)

    monkeypatch.setattr(os, "link", move_parent_then_link)
    with pytest.raises(OSError) as raised:
        atomic_create_text(skill_dir / "SKILL.md", ROBOT)
    assert fired, "the parent was never moved, this test proved nothing"
    assert list(moved.iterdir()) == [] and list(skill_dir.iterdir()) == []  # nothing published anywhere
    assert "moved or replaced" in str(raised.value)


@pytest.mark.platforms("posix")
def test_a_skill_dir_moved_during_a_blocked_create_leaves_no_loadable_skill_behind(skills, tmp_path, monkeypatch):
    moved = skills / "relocated"
    monkeypatch.setattr(sm, "_security_scan_skill", lambda _d: "blocked")
    real_link, fired = os.link, []

    def move_parent_then_link(src, dst, *a, **kw):
        if os.path.basename(os.fspath(dst)) == "SKILL.md" and not fired:
            fired.append(True)
            _child(tmp_path, _MOVE_PARENT, skills / NAME, moved)
        return real_link(src, dst, *a, **kw)

    monkeypatch.setattr(os, "link", move_parent_then_link)
    result = sm._create_skill(NAME, ROBOT)
    assert fired and result["success"] is False
    assert not (moved / "SKILL.md").exists() and sm._find_skill("relocated") is None
    assert "SKILL.md" in result["error"] and "moved or replaced" in result["error"]


# --- 5. snapshot restore: only executed ops' skills, no foreign deletions ------------------------------

def _old_skill(skills: Path) -> Path:
    old = skills / "old-skill"
    old.mkdir()
    (old / "SKILL.md").write_text(OLD, encoding="utf-8")
    return old


@pytest.mark.platforms("posix")
def test_rollback_does_not_restore_a_skill_whose_op_never_ran_nor_delete_an_unrelated_note(skills, tmp_path,
                                                                                         monkeypatch):
    old = _old_skill(skills)
    unrelated = skills / "old-skill.rollback-broken"
    unrelated.mkdir()
    (unrelated / "note.txt").write_text("UNRELATED EXISTING NOTE\n")
    fired = _before_op(monkeypatch, "write_file", lambda: _child(tmp_path, _SAVE, old / "SKILL.md", HUMAN.decode()),
                       file_path="bad/nope.md")
    untouched = {"action": "patch", "name": "old-skill", "old_string": "Step 1", "new_string": "Step 2"}
    result = _batch(_create_op(), BAD_OP, untouched)
    assert fired and result["success"] is False and result["failed_index"] == 1
    assert (old / "SKILL.md").read_bytes() == HUMAN  # its op never ran: nothing of it to undo
    assert (unrelated / "note.txt").read_text() == "UNRELATED EXISTING NOTE\n"
    assert not (skills / NAME).exists()


def test_restoring_a_touched_skill_never_deletes_a_pre_existing_rollback_broken_dir(skills):
    old = _old_skill(skills)
    unrelated = skills / "old-skill.rollback-broken"
    unrelated.mkdir()
    (unrelated / "note.txt").write_text("UNRELATED EXISTING NOTE\n")
    patch_old = {"action": "patch", "name": "old-skill", "old_string": "Step 1", "new_string": "Step 2"}
    bad_on_old = {**BAD_OP, "name": "old-skill"}
    result = _batch(patch_old, bad_on_old)
    assert result["success"] is False and result["error"].endswith("all touched skills rolled back.")
    assert (old / "SKILL.md").read_text(encoding="utf-8") == OLD  # restored from the snapshot
    assert (unrelated / "note.txt").read_text() == "UNRELATED EXISTING NOTE\n"
    assert sorted(p.name for p in skills.iterdir() if p.name.startswith("old-skill")) == [
        "old-skill", "old-skill.rollback-broken"]


@pytest.mark.platforms("posix")
def test_restore_keeps_and_names_a_file_nobody_in_the_batch_wrote(skills, tmp_path, monkeypatch):
    old = _old_skill(skills)
    patch_old = {"action": "patch", "name": "old-skill", "old_string": "Step 1", "new_string": "Step 2"}
    bad_on_old = {**BAD_OP, "name": "old-skill"}
    fired = _before_op(monkeypatch, "write_file", lambda: _child(tmp_path, _WRITE_IN_PLACE, old / "extra.md", "HUMAN"),
                       file_path="bad/nope.md")
    result = _batch(patch_old, bad_on_old)
    assert fired and result["success"] is False
    assert (old / "SKILL.md").read_text(encoding="utf-8") == OLD  # the batch's own patch is still undone
    assert "ROLLBACK INCOMPLETE" not in result["error"] and "extra.md" in result["error"]
    where = re.search(r"preserved at '([^']+)'", result["error"])
    assert where, result["error"]
    assert [p.read_text() for p in Path(where.group(1)).rglob("extra.md")] == ["HUMAN"]  # the copy is where it says


# --- 6. malformed input is a structured error ----------------------------------------------------------

def test_a_list_valued_file_path_is_rejected_as_an_error_not_an_exception(skills):
    result = _batch({"action": "write_file", "name": NAME, "file_path": ["references", "a"], "file_content": "x"})
    assert result["success"] is False and "file_path" in result["error"]
