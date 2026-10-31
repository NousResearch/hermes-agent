"""Cleanup/rollback safety of ``skill_manage(create)`` against an uncooperative writer.

``test_skill_create_exclusive`` pins that a SKILL.md another writer put in place is never overwritten.
These tests pin what happens AFTER that: the cleanup paths (batch rollback, scan-blocked removal, error
exits) must not delete the other writer's data either, must say so when they cannot finish, and a failed
publish must be a reported failure that rolls the batch back, not an exception that strands state.

Everything runs through the public entry points (``apply_skill_pending`` with ``operations=[...]`` and
``_create_skill``) so the real rollback / cleanup code is on the path. The uncooperative writer is a
thread or a separate OS process that never takes the per-skill lock. Each hook asserts it fired, so a
test cannot go green by never reaching the code it is about.
"""

from __future__ import annotations

import errno
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from tools import skill_manager_tool as sm
from tests.tools.test_skill_create_exclusive import (
    HUMAN_TEXT, NAME, ROBOT, WAIT, _AfterMkdirBarrier, _plant_at_publish,
)

NOTE_TEXT = b"HUMAN NOTE - a different file in the same directory\n"
DIRFD = hasattr(os, "O_NOFOLLOW") and os.link in os.supports_dir_fd and os.open in os.supports_dir_fd
BAD_OP = {"action": "write_file", "name": NAME, "file_path": "bad/nope.md", "file_content": "x"}  # always fails


@pytest.fixture
def skills(tmp_path, monkeypatch):
    """Skills dir + hermetic HERMES_HOME strictly inside this test's tmp_path."""
    home = tmp_path / "home"
    root = home / "skills"
    root.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(sm, "SKILLS_DIR", root)
    monkeypatch.setattr("agent.skill_utils.get_all_skills_dirs", lambda: [root])
    assert sm._skills_dir() == root and root.resolve().is_relative_to(tmp_path.resolve())
    return root


def _batch(*ops: dict) -> dict:
    """The public batch shape, replayed through the approve handler (gate bypassed, locks held)."""
    return json.loads(sm.apply_skill_pending({"operations": list(ops)}))


def _create_op(name: str = NAME, content: str = ROBOT, **extra) -> dict:
    return {"action": "create", "name": name, "content": content, **extra}


def _spy_on_rollback(monkeypatch) -> list:
    """Count calls of the real batch rollback, so a green test proves the rollback path actually ran."""
    from tools import skill_manager_batch as batch

    calls: list = []
    real = batch._rollback

    def spy(*a, **kw):
        calls.append(1)
        return real(*a, **kw)

    monkeypatch.setattr(batch, "_rollback", spy)
    return calls


def _plant_skill_and_note(skill_dir: Path):
    def plant():
        (skill_dir / "SKILL.md").write_bytes(HUMAN_TEXT)
        (skill_dir / "human-note.txt").write_bytes(NOTE_TEXT)
    return plant


def _names(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir())


# --- 1. operations=[create]: the REAL rollback must not delete an outsider's files -----------------

def test_batch_create_refusal_preserves_every_file_another_writer_put_in_the_skill_dir(skills, monkeypatch):
    skill_dir = skills / NAME
    rollbacks = _spy_on_rollback(monkeypatch)
    barrier = _AfterMkdirBarrier(monkeypatch, skill_dir, _plant_skill_and_note(skill_dir))
    result = barrier.run(lambda: _batch(_create_op()))
    assert rollbacks, "the batch rollback never ran, this test proved nothing"
    assert result["success"] is False and result["failed_index"] == 0
    assert "already exists" in result["error"]
    assert (skill_dir / "SKILL.md").read_bytes() == HUMAN_TEXT
    assert (skill_dir / "human-note.txt").read_bytes() == NOTE_TEXT
    assert _names(skill_dir) == ["SKILL.md", "human-note.txt"]
    assert "all touched skills rolled back" not in result["error"]  # false while files are kept
    assert "preserved" in result["error"].lower()


@pytest.mark.platforms("posix")
def test_batch_create_refusal_preserves_files_written_by_an_independent_process(skills, tmp_path, monkeypatch):
    skill_dir = skills / NAME
    env = {"HOME": str(tmp_path / "child-home"), "HERMES_HOME": str(tmp_path / "child-hermes"),
           "TMPDIR": str(tmp_path / "child-tmp"), "PATH": os.defpath, "LC_ALL": "C.UTF-8"}
    for key in ("HOME", "HERMES_HOME", "TMPDIR"):
        Path(env[key]).mkdir()
    script = ("import pathlib, sys; d = pathlib.Path(sys.argv[1]); "
              "(d / 'SKILL.md').write_bytes(sys.argv[2].encode()); (d / 'human-note.txt').write_bytes(sys.argv[3].encode())")

    def plant():
        subprocess.run([sys.executable, "-I", "-c", script, str(skill_dir), HUMAN_TEXT.decode(), NOTE_TEXT.decode()],
                       env=env, check=True, timeout=WAIT)

    rollbacks = _spy_on_rollback(monkeypatch)
    fired = _plant_at_publish(monkeypatch, plant)
    result = _batch(_create_op())
    assert fired and rollbacks, "neither the publish barrier nor the rollback ran, this test proved nothing"
    assert result["success"] is False
    assert (skill_dir / "SKILL.md").read_bytes() == HUMAN_TEXT
    assert (skill_dir / "human-note.txt").read_bytes() == NOTE_TEXT


def test_batch_rollback_removes_what_the_batch_wrote_and_nothing_else(skills, monkeypatch):
    """Compatibility: a late failure still removes a batch-created skill completely; others stay."""
    other = ROBOT.replace(NAME, "bystander")
    assert sm._create_skill("bystander", other)["success"] is True
    rollbacks = _spy_on_rollback(monkeypatch)
    result = _batch(_create_op(),
                    {"action": "write_file", "name": NAME, "file_path": "references/a.md", "file_content": "a"},
                    BAD_OP)
    assert rollbacks and result["success"] is False and result["failed_index"] == 2
    assert not (skills / NAME).exists()
    assert (skills / "bystander" / "SKILL.md").read_bytes() == other.encode("utf-8")
    assert "all touched skills rolled back" in result["error"]  # nothing foreign here: the claim holds


def test_batch_rollback_keeps_a_file_an_outsider_added_to_the_batch_created_dir(skills, monkeypatch):
    skill_dir = skills / NAME
    real_from = sm._skill_manage_from
    dropped: list = []

    def outsider_drops_a_note_then_op_runs(payload, **kw):
        if payload.get("action") == "write_file":
            dropped.append(True)
            (skill_dir / "human-note.txt").write_bytes(NOTE_TEXT)
        return real_from(payload, **kw)

    monkeypatch.setattr(sm, "_skill_manage_from", outsider_drops_a_note_then_op_runs)
    result = _batch(_create_op(), BAD_OP)
    assert dropped and result["success"] is False and result["failed_index"] == 1
    assert _names(skill_dir) == ["human-note.txt"]  # the batch's own SKILL.md went, the note stayed
    assert (skill_dir / "human-note.txt").read_bytes() == NOTE_TEXT
    assert "preserved" in result["error"].lower()


def test_batch_rollback_keeps_a_skill_md_an_outsider_replaced_after_the_create(skills, monkeypatch):
    skill_md = skills / NAME / "SKILL.md"
    real_from = sm._skill_manage_from
    replaced: list = []

    def outsider_saves_the_file_then_op_runs(payload, **kw):
        if payload.get("action") == "write_file":
            replaced.append(True)
            tmp = skill_md.with_name("human.tmp")
            tmp.write_bytes(HUMAN_TEXT)
            os.replace(tmp, skill_md)  # an editor's atomic save: new inode, new bytes
        return real_from(payload, **kw)

    monkeypatch.setattr(sm, "_skill_manage_from", outsider_saves_the_file_then_op_runs)
    result = _batch(_create_op(), BAD_OP)
    assert replaced and result["success"] is False and result["failed_index"] == 1
    assert skill_md.read_bytes() == HUMAN_TEXT
    assert _names(skill_md.parent) == ["SKILL.md"]


def test_batch_rollback_keeps_a_skill_md_edited_in_place_with_size_and_mtime_unchanged(skills, monkeypatch):
    skill_md = skills / NAME / "SKILL.md"
    real_from = sm._skill_manage_from
    edited: list = []

    def outsider_edits_in_place_then_op_runs(payload, **kw):
        if payload.get("action") == "write_file":
            edited.append(True)
            before = skill_md.stat()
            skill_md.write_bytes(b"H" * before.st_size)  # same inode, same size ...
            os.utime(skill_md, ns=(before.st_atime_ns, before.st_mtime_ns))  # ... same mtime
        return real_from(payload, **kw)

    monkeypatch.setattr(sm, "_skill_manage_from", outsider_edits_in_place_then_op_runs)
    result = _batch(_create_op(), BAD_OP)
    assert edited and result["success"] is False
    assert skill_md.exists() and skill_md.read_bytes().startswith(b"HHHH")


# --- 2. scan-blocked cleanup of the file this create published -------------------------------------

def _arm_after_scan(monkeypatch, on_scan=None) -> list:
    """Scan hook answering "blocked"; the returned list is non-empty once the scan has run."""
    armed: list = []

    def scan(_skill_dir):
        if on_scan is not None:
            on_scan()
        armed.append(True)
        return "blocked"

    monkeypatch.setattr(sm, "_security_scan_skill", scan)
    return armed


def _hook_removal_primitives(monkeypatch, armed: list, action) -> list:
    """Run ``action(primitive_name)`` before the first remove/rename/replace of a ``SKILL.md`` after the scan.

    Wraps every spelling of "take this file away" so the test does not depend on which one the cleanup uses.
    """
    fired: list = []
    for name in ("unlink", "remove", "rename", "replace"):
        def hook(src, *a, _real=getattr(os, name), _name=name, **kw):
            if armed and os.path.basename(os.fspath(src)) == "SKILL.md":
                fired.append(_name)
                action(_name)
            return _real(src, *a, **kw)
        monkeypatch.setattr(os, name, hook)
    return fired


@pytest.mark.platforms("posix")
def test_scan_block_does_not_delete_a_replacement_that_lands_between_check_and_removal(skills, monkeypatch):
    skill_md = skills / NAME / "SKILL.md"
    armed = _arm_after_scan(monkeypatch)
    swapped: list = []
    real_replace = os.replace

    def swap_once(_name):
        if not swapped:
            swapped.append(True)
            tmp = skill_md.with_name("human.tmp")
            tmp.write_bytes(HUMAN_TEXT)
            real_replace(tmp, skill_md)  # lands after the cleanup's own check, before its removal

    fired = _hook_removal_primitives(monkeypatch, armed, swap_once)
    result = sm._create_skill(NAME, ROBOT)
    assert armed and fired and swapped, "no removal primitive ran after the scan, this test proved nothing"
    assert result["success"] is False and "blocked" in result["error"]
    assert skill_md.read_bytes() == HUMAN_TEXT
    assert _names(skill_md.parent) == ["SKILL.md"]


def test_scan_block_does_not_delete_a_file_edited_in_place_with_size_and_mtime_unchanged(skills, monkeypatch):
    skill_md = skills / NAME / "SKILL.md"

    def edit_in_place():
        before = skill_md.stat()
        skill_md.write_bytes(b"H" * before.st_size)
        os.utime(skill_md, ns=(before.st_atime_ns, before.st_mtime_ns))

    armed = _arm_after_scan(monkeypatch, edit_in_place)
    result = sm._create_skill(NAME, ROBOT)
    assert armed and result["success"] is False
    assert skill_md.exists() and skill_md.read_bytes().startswith(b"HHHH")


def test_scan_block_whose_cleanup_fails_says_so_instead_of_just_blocked(skills, monkeypatch):
    skill_md = skills / NAME / "SKILL.md"
    armed = _arm_after_scan(monkeypatch)

    def deny(name):
        raise PermissionError(errno.EACCES, f"simulated: {name} denied")

    fired = _hook_removal_primitives(monkeypatch, armed, deny)
    result = sm._create_skill(NAME, ROBOT)
    assert armed and fired, "no removal was attempted, this test proved nothing"
    assert result["success"] is False and "blocked" in result["error"]
    assert skill_md.exists()  # the blocked file could not be removed ...
    assert result.get("cleanup_incomplete") is True  # ... and the caller is told, not left to assume
    assert str(skill_md) in result["error"] and "remov" in result["error"].lower()


def test_batch_scan_block_whose_cleanup_fails_is_reported_through_the_batch_error(skills, monkeypatch):
    armed = _arm_after_scan(monkeypatch)

    def deny(name):
        raise PermissionError(errno.EACCES, "simulated denial")

    fired = _hook_removal_primitives(monkeypatch, armed, deny)
    result = _batch(_create_op())
    assert armed and fired, "no removal was attempted, this test proved nothing"
    assert (skills / NAME / "SKILL.md").exists()
    assert result["success"] is False and "blocked" in result["error"]
    assert str(skills / NAME / "SKILL.md") in result["error"]
    assert "all touched skills rolled back" not in result["error"]  # the blocked file is still on disk


@pytest.mark.platforms("posix")
def test_scan_block_keeps_a_file_an_independent_process_saved_during_the_scan(skills, tmp_path, monkeypatch):
    skill_md = skills / NAME / "SKILL.md"
    env = {"HOME": str(tmp_path / "child-home"), "HERMES_HOME": str(tmp_path / "child-hermes"),
           "TMPDIR": str(tmp_path / "child-tmp"), "PATH": os.defpath, "LC_ALL": "C.UTF-8"}
    for key in ("HOME", "HERMES_HOME", "TMPDIR"):
        Path(env[key]).mkdir()
    script = ("import os, pathlib, sys; p = pathlib.Path(sys.argv[1]); t = p.with_name('editor.tmp'); "
              "t.write_bytes(sys.argv[2].encode()); os.replace(t, p)")  # an editor's atomic save

    def editor_saves():
        subprocess.run([sys.executable, "-I", "-c", script, str(skill_md), HUMAN_TEXT.decode()],
                       env=env, check=True, timeout=WAIT)

    armed = _arm_after_scan(monkeypatch, editor_saves)
    result = _batch(_create_op())
    assert armed and result["success"] is False and "blocked" in result["error"]
    assert skill_md.read_bytes() == HUMAN_TEXT
    assert _names(skill_md.parent) == ["SKILL.md"]


def test_batch_rollback_that_cannot_remove_its_own_file_says_incomplete_and_names_it(skills, monkeypatch):
    armed: list = []
    real_from = sm._skill_manage_from

    def arm_when_the_failing_op_starts(payload, **kw):
        if payload.get("action") == "write_file":
            armed.append(True)  # the rollback follows this op's failure
        return real_from(payload, **kw)

    monkeypatch.setattr(sm, "_skill_manage_from", arm_when_the_failing_op_starts)

    def deny(name):
        raise PermissionError(errno.EACCES, f"simulated: {name} denied")

    fired = _hook_removal_primitives(monkeypatch, armed, deny)
    result = _batch(_create_op(), BAD_OP)
    assert armed and fired, "the rollback never tried to remove the file, this test proved nothing"
    assert result["success"] is False and result["failed_index"] == 1
    assert (skills / NAME / "SKILL.md").exists()
    assert "ROLLBACK INCOMPLETE" in result["error"] and str(skills / NAME / "SKILL.md") in result["error"]
    assert "all touched skills rolled back" not in result["error"]


def test_a_skill_md_changed_during_the_scan_is_not_reported_as_a_successful_create(skills, monkeypatch):
    skill_md = skills / NAME / "SKILL.md"

    def writer_replaces_during_a_passing_scan(_skill_dir):
        tmp = skill_md.with_name("human.tmp")
        tmp.write_bytes(HUMAN_TEXT)
        os.replace(tmp, skill_md)
        return None  # the scan itself passes

    monkeypatch.setattr(sm, "_security_scan_skill", writer_replaces_during_a_passing_scan)
    result = sm._create_skill(NAME, ROBOT)
    assert skill_md.read_bytes() == HUMAN_TEXT  # theirs stays
    assert result["success"] is False  # what was validated is not what is on disk


# --- 3. exceptions are failures with a rollback, never an escape that strands partial state --------

def _fail_publish_of(monkeypatch, which: int, exc: BaseException) -> list:
    real = sm.atomic_create_text
    calls: list = []

    def create(path, *a, **kw):
        calls.append(path)
        if len(calls) == which:
            raise exc
        return real(path, *a, **kw)

    monkeypatch.setattr(sm, "atomic_create_text", create)
    return calls


def test_batch_publish_error_is_a_failed_batch_with_everything_rolled_back(skills, tmp_path, monkeypatch):
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    calls = _fail_publish_of(monkeypatch, 2, PermissionError(errno.EACCES, "simulated link denied"))
    rollbacks = _spy_on_rollback(monkeypatch)
    result = _batch(_create_op("one", ROBOT.replace(NAME, "one")), _create_op("two", ROBOT.replace(NAME, "two")))
    assert len(calls) == 2 and rollbacks, "the second publish never ran, this test proved nothing"
    assert result["success"] is False and result["failed_index"] == 1
    assert "simulated link denied" in result["error"]  # reported, not swallowed
    assert not (skills / "one").exists() and not (skills / "two").exists()
    assert not list(scratch.glob("skill_batch_*")), "the batch snapshot area was leaked"


def test_batch_op_that_raises_unexpectedly_rolls_back_the_earlier_ops(skills, monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("simulated handler crash")

    rollbacks = _spy_on_rollback(monkeypatch)
    monkeypatch.setattr(sm, "_write_file", boom)
    result = _batch(_create_op(),
                    {"action": "write_file", "name": NAME, "file_path": "references/a.md", "file_content": "a"})
    assert rollbacks and result["success"] is False and result["failed_index"] == 1
    assert "simulated handler crash" in result["error"]
    assert not (skills / NAME).exists()


def test_direct_create_publish_error_is_an_error_result_and_leaves_no_empty_dir(skills, monkeypatch):
    calls = _fail_publish_of(monkeypatch, 1, PermissionError(errno.EACCES, "simulated link denied"))
    result = sm._create_skill(NAME, ROBOT)
    assert calls and result["success"] is False and "simulated link denied" in result["error"]
    assert not (skills / NAME).exists()


# --- 4. the skill dir swapped for a symlink after our mkdir: nothing is written through it ---------

@pytest.mark.skipif(not DIRFD, reason="needs O_NOFOLLOW and dir_fd support (not Windows)")
@pytest.mark.parametrize("when", ["after-mkdir", "before-publish"])
@pytest.mark.parametrize("via", ["direct", "batch"])
def test_skill_dir_swapped_for_a_symlink_is_never_written_through(skills, tmp_path, monkeypatch, when, via):
    skill_dir = skills / NAME
    outside = tmp_path / "outside"
    outside.mkdir()
    swapped: list = []

    def swap():
        if not swapped:
            swapped.append(True)
            skill_dir.rmdir()
            skill_dir.symlink_to(outside, target_is_directory=True)

    if when == "after-mkdir":
        real_mkdir = Path.mkdir

        def mkdir(path_self, *a, **kw):
            result = real_mkdir(path_self, *a, **kw)
            if path_self == skill_dir:
                swap()
            return result

        monkeypatch.setattr(Path, "mkdir", mkdir)
    else:
        import hermes_constants

        real_helper = hermes_constants.mkdir_under_hermes_home

        def helper(path, *a, **kw):
            result = real_helper(path, *a, **kw)
            if Path(path) == skill_dir:  # the guarded write's own "parent exists" step, right before publishing
                swap()
            return result

        monkeypatch.setattr(hermes_constants, "mkdir_under_hermes_home", helper)

    result = sm._create_skill(NAME, ROBOT) if via == "direct" else _batch(_create_op())
    assert swapped, "the swap never fired, this test proved nothing"
    assert result["success"] is False
    assert list(outside.iterdir()) == [], "a file was written through the swapped directory"
    assert skill_dir.is_symlink()  # the other party's symlink is theirs: not followed, not removed
