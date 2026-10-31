"""A snapshot restore never deletes what an independent, non-cooperating writer put on disk.

``_restore_snapshot`` moves the state a failed batch left aside, puts the snapshot back and drops the displaced
copy. Two gaps let a writer that takes no skill lock lose data there, both reproduced with a real second process:

* the restore wrote straight into the live name; when someone created that directory in the meantime the copy
  failed with ``EEXIST`` and the error handler ``rmtree``'d the directory - the other writer's file with it;
* the displaced copy was inventoried and then ``rmtree``'d, so a file saved into it after the inventory went with it.

The restore is now staged privately and published with a rename that cannot replace a non-empty directory, and
the displaced copy is emptied file by file (only files proven ours) and left in place - named in the error - the
moment anything else is in it. Not claimed: POSIX has no compare-and-delete, so a process that already holds one
of OUR files (byte-identical to the snapshot) open can still write into it between the check and its unlink.

Every uncooperative writer is a separate ``python -I`` process with its own HOME/HERMES_HOME/TMPDIR that takes no
lock; every hook asserts it fired.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from tools import skill_manager_tool as sm

OLD = """\
---
name: old-skill
description: Use when the old content must stay.
---

# Old

Step 1: original.
"""
PATCH_OLD = {"action": "patch", "name": "old-skill", "old_string": "Step 1", "new_string": "Step 2"}
BAD_ON_OLD = {"action": "write_file", "name": "old-skill", "file_path": "bad/nope.md", "file_content": "x"}
HUMAN = "HUMAN DATA - must survive\n"
WAIT = 20

_MKDIR_AND_SAVE = ("import pathlib, sys; p = pathlib.Path(sys.argv[1]); p.mkdir(); "
                   "(p / 'human.txt').write_text(sys.argv[2])")
_SAVE_INTO = "import pathlib, sys; (pathlib.Path(sys.argv[1]) / 'human.txt').write_text(sys.argv[2])"


@pytest.fixture
def skills(tmp_path, monkeypatch):
    home = tmp_path / "home"
    root = home / "skills"
    root.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(sm, "SKILLS_DIR", root)
    monkeypatch.setattr("agent.skill_utils.get_all_skills_dirs", lambda: [root])
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))  # where preserved copies land
    (tmp_path / "tmp").mkdir()
    return root


def _child(tmp_path: Path, script: str, *args) -> None:
    """Run ``script`` in an independent interpreter with its own isolated HOME/HERMES_HOME/TMPDIR."""
    env = {"PATH": os.defpath, "LC_ALL": "C.UTF-8"}
    for key in ("HOME", "HERMES_HOME", "TMPDIR"):
        directory = tmp_path / f"child-{key}"
        directory.mkdir(exist_ok=True)
        env[key] = str(directory)
    subprocess.run([sys.executable, "-I", "-c", script, *map(str, args)], env=env, check=True, timeout=WAIT)


def _old_skill(skills: Path, *, with_reference: bool = False) -> Path:
    old = skills / "old-skill"
    old.mkdir()
    (old / "SKILL.md").write_text(OLD, encoding="utf-8")
    if with_reference:
        (old / "references").mkdir()
        (old / "references" / "a.md").write_text("reference a\n", encoding="utf-8")
    return old


def _batch(*ops: dict) -> dict:
    return json.loads(sm.apply_skill_pending({"operations": list(ops)}))


def _everywhere(tmp_path: Path, filename: str) -> list[str]:
    """Contents of every file with this name anywhere under the test's tmp area (skills root, temp dir, ...)."""
    return sorted(p.read_text() for p in tmp_path.rglob(filename)
                  if p.is_file() and not any(part.startswith("child-") for part in p.relative_to(tmp_path).parts))


def _fire_once(monkeypatch, target: str, name: str, hook, *, match) -> list:
    """Wrap ``<module>.<name>`` (e.g. os.unlink / shutil.rmtree / shutil.copytree) so ``hook(arg)`` runs once,
    right before the first call whose first argument satisfies ``match``."""
    module = {"os": os, "shutil": __import__("shutil")}[target]
    real = getattr(module, name)
    fired: list = []

    def wrapper(path, *args, **kw):
        if not fired and match(str(path)):
            fired.append(str(path))
            hook(Path(str(path)))
        return real(path, *args, **kw)

    monkeypatch.setattr(module, name, wrapper)
    return fired


@pytest.mark.platforms("posix")
def test_restore_never_deletes_a_directory_another_process_created_at_the_skill_name(skills, tmp_path, monkeypatch):
    old = _old_skill(skills)
    # The restore copies the snapshot (first argument: the snapshot dir, a skill_batch_* temp dir); a second
    # process creates the skill directory with its own file just before that copy lands.
    fired = _fire_once(monkeypatch, "shutil", "copytree", lambda _: _child(tmp_path, _MKDIR_AND_SAVE, old, HUMAN),
                       match=lambda src: "skill_batch_" in src and not old.exists())
    result = _batch(PATCH_OLD, BAD_ON_OLD)
    assert fired, "the writer never ran: this test would prove nothing"
    assert result["success"] is False
    assert (old / "human.txt").read_text() == HUMAN  # untouched where the other process put it
    error = result["error"]
    assert "ROLLBACK" in error and "rolled back everything" not in error and "all touched skills rolled back" not in error
    kept = re.search(r"batch state preserved at '([^']+)'", error)
    assert kept, error  # the state the batch left is named, not dropped
    assert "Step 2" in (Path(kept.group(1)) / "SKILL.md").read_text()
    assert not [p for p in skills.iterdir() if ".restore-" in p.name]  # no private staging copy left behind


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("where", ["top", "nested"])
def test_a_file_saved_into_the_displaced_copy_after_it_was_checked_is_kept_and_named(skills, tmp_path, monkeypatch,
                                                                                     where):
    old = _old_skill(skills, with_reference=True)

    def save(path: Path) -> None:
        directory = next(p for p in skills.iterdir() if ".rollback-broken-" in p.name)
        _child(tmp_path, _SAVE_INTO, directory if where == "top" else directory / "references", HUMAN)

    def displaced(path: str) -> bool:
        return ".rollback-broken-" in path

    # Whatever primitive the teardown uses to drop the displaced copy (a tree delete or per-file unlinks), the
    # second process saves into it right before the first deletion.
    fired = _fire_once(monkeypatch, "shutil", "rmtree", save, match=displaced)
    fired_unlink = _fire_once(monkeypatch, "os", "unlink", save, match=displaced)
    result = _batch(PATCH_OLD, BAD_ON_OLD)
    assert fired or fired_unlink, "the writer never ran: this test would prove nothing"
    assert result["success"] is False
    assert (old / "SKILL.md").read_text(encoding="utf-8") == OLD  # the batch's own patch is still undone
    assert _everywhere(tmp_path, "human.txt") == [HUMAN]  # not deleted, exactly one copy
    where_kept = re.search(r"preserved at '([^']+)'", result["error"])
    assert where_kept and "human.txt" in result["error"], result["error"]
    assert (Path(where_kept.group(1)) / ("human.txt" if where == "top" else "references/human.txt")).read_text() == HUMAN
    assert "all touched skills rolled back" not in result["error"]


def test_an_undisturbed_restore_leaves_nothing_behind_and_says_so(skills):
    old = _old_skill(skills, with_reference=True)
    result = _batch(PATCH_OLD, BAD_ON_OLD)
    assert result["success"] is False and result["error"].endswith("all touched skills rolled back.")
    assert (old / "SKILL.md").read_text(encoding="utf-8") == OLD
    assert (old / "references" / "a.md").read_text() == "reference a\n"
    assert [p.name for p in skills.iterdir() if "rollback" in p.name or "restore" in p.name] == []  # no aside/staging
