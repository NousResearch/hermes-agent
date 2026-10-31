"""``skill_manage(create)`` publishes SKILL.md exclusively: a file that another writer put there is never replaced.

``_create_skill`` makes the skill directory and only later writes SKILL.md. The per-skill lock
serialises ``skill_manage`` callers but not an independent writer (an editor, a shell, another
tool) that creates ``SKILL.md`` inside that window. These tests pin the contract: the other
writer's bytes survive, ``create`` reports a clear error instead of success, and nothing foreign
is removed on any cleanup path. The barrier is placed right after the real ``mkdir`` (and, for the
late variant, at the publish primitive) so the tests do not depend on how the write is spelled.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from tools import skill_manager_tool as sm

NAME = "race-skill"
ROBOT = """\
---
name: race-skill
description: Use when the robot content must be published.
---

# 로봇 교훈 🤖

Step 1: robot wrote this.
"""
HUMAN_TEXT = "HUMAN 작성 내용 ✓ — must survive\n".encode()
HUMAN_BINARY = b"\xff\xfe\x00not utf-8 \x80\x81\n"
WAIT = 20


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


def _payload(name: str = NAME, content: str = ROBOT, **extra) -> dict:
    return {"action": "create", "name": name, "content": content, **extra}


def _approve(payload: dict) -> dict:
    """The /skills approve replay path: full dispatcher, per-skill lock, gate bypassed."""
    return json.loads(sm.apply_skill_pending(payload))


class _AfterMkdirBarrier:
    """Pause the creating thread right after it made its skill dir; run ``plant`` from another thread.

    The planting thread is the uncooperative writer: it never takes the skill lock.
    """

    def __init__(self, monkeypatch, skill_dir: Path, plant):
        self.skill_dir, self.plant = skill_dir, plant
        self.reached, self.proceed = threading.Event(), threading.Event()
        self.errors: list[BaseException] = []
        real_mkdir = Path.mkdir

        def mkdir(path_self, *a, **kw):
            result = real_mkdir(path_self, *a, **kw)
            if path_self == skill_dir:
                self.reached.set()
                assert self.proceed.wait(WAIT), "writer never released the creator"
            return result

        monkeypatch.setattr(Path, "mkdir", mkdir)

    def _writer(self):
        try:
            assert self.reached.wait(WAIT), "create never reached its mkdir"
            assert self.skill_dir.is_dir() and not (self.skill_dir / "SKILL.md").exists()
            self.plant()
        except BaseException as exc:  # reported on the main thread
            self.errors.append(exc)
        finally:
            self.proceed.set()

    def run(self, create):
        out: list = []

        def create_in_thread():
            try:
                out.append(create())
            except BaseException as exc:  # surfaced on the main thread, never a silent empty result
                self.errors.append(exc)

        writer = threading.Thread(target=self._writer)
        creator = threading.Thread(target=create_in_thread)
        writer.start(); creator.start()
        creator.join(WAIT * 2); writer.join(WAIT)
        assert not creator.is_alive() and not writer.is_alive()
        assert not self.errors, self.errors
        return out[0]


def _plant_file(skill_dir: Path, data: bytes):
    return lambda: (skill_dir / "SKILL.md").write_bytes(data)


def _assert_refused_and_preserved(result: dict, skill_dir: Path, data: bytes) -> None:
    assert result["success"] is False
    assert "SKILL.md" in result["error"] and "already" in result["error"]
    assert (skill_dir / "SKILL.md").read_bytes() == data
    # nothing else of ours is left behind next to the writer's file
    assert sorted(p.name for p in skill_dir.iterdir()) == ["SKILL.md"]


@pytest.mark.parametrize("data", [HUMAN_TEXT, HUMAN_BINARY, b""], ids=["unicode", "non-utf8", "empty"])
def test_writer_after_mkdir_is_preserved_through_approval_path(skills, monkeypatch, data):
    skill_dir = skills / NAME
    barrier = _AfterMkdirBarrier(monkeypatch, skill_dir, _plant_file(skill_dir, data))
    result = barrier.run(lambda: _approve(_payload()))
    _assert_refused_and_preserved(result, skill_dir, data)


def test_writer_after_mkdir_is_preserved_by_direct_create(skills, monkeypatch):
    skill_dir = skills / NAME
    barrier = _AfterMkdirBarrier(monkeypatch, skill_dir, _plant_file(skill_dir, HUMAN_TEXT))
    result = barrier.run(lambda: sm._create_skill(NAME, ROBOT))
    _assert_refused_and_preserved(result, skill_dir, HUMAN_TEXT)


def test_writer_in_a_category_dir_and_in_a_configured_create_dir_is_preserved(tmp_path, skills, monkeypatch):
    create_dir = tmp_path / "shared-fleet"
    create_dir.mkdir()
    monkeypatch.setattr("agent.skill_utils.get_skill_create_dir", lambda: create_dir)
    monkeypatch.setattr("agent.skill_utils.get_all_skills_dirs", lambda: [skills, create_dir])
    skill_dir = create_dir / "devops" / NAME
    barrier = _AfterMkdirBarrier(monkeypatch, skill_dir, _plant_file(skill_dir, HUMAN_TEXT))
    result = barrier.run(lambda: _approve(_payload(category="devops")))
    _assert_refused_and_preserved(result, skill_dir, HUMAN_TEXT)
    assert not (skills / "devops").exists()


def _plant_at_publish(monkeypatch, plant):
    """Run ``plant`` once, immediately before whichever primitive publishes a SKILL.md."""
    fired: list[str] = []
    for fn_name in ("link", "replace", "rename"):
        real = getattr(os, fn_name)

        def hook(src, dst, *a, _real=real, _name=fn_name, **kw):
            if not fired and os.path.basename(os.fspath(dst)) == "SKILL.md":
                fired.append(_name)
                plant()
            return _real(src, dst, *a, **kw)

        monkeypatch.setattr(os, fn_name, hook)
    return fired


@pytest.mark.platforms("posix")
def test_writer_in_the_last_instant_before_publish_is_preserved(skills, monkeypatch):
    skill_dir = skills / NAME
    fired = _plant_at_publish(monkeypatch, _plant_file(skill_dir, HUMAN_TEXT))
    result = sm._create_skill(NAME, ROBOT)
    assert fired, "no publish primitive ran, the barrier proved nothing"
    _assert_refused_and_preserved(result, skill_dir, HUMAN_TEXT)


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("kind", ["live-symlink", "dangling-symlink", "directory"])
def test_planted_symlink_or_directory_is_never_followed_or_replaced(skills, tmp_path, monkeypatch, kind):
    outside = tmp_path / "outside"
    outside.mkdir()
    victim = outside / "victim.md"
    victim.write_bytes(b"OUTSIDE FILE\n")
    skill_md = skills / NAME / "SKILL.md"

    def plant():
        if kind == "live-symlink":
            skill_md.symlink_to(victim)
        elif kind == "dangling-symlink":
            skill_md.symlink_to(outside / "missing.md")
        else:
            skill_md.mkdir()
            (skill_md / "keep.txt").write_bytes(b"k")

    fired = _plant_at_publish(monkeypatch, plant)
    result = sm._create_skill(NAME, ROBOT)
    assert fired, "no publish primitive ran, the barrier proved nothing"
    assert result["success"] is False and "already" in result["error"]
    assert victim.read_bytes() == b"OUTSIDE FILE\n"
    assert not (outside / "missing.md").exists()
    if kind == "directory":
        assert skill_md.is_dir() and (skill_md / "keep.txt").read_bytes() == b"k"
    else:
        assert skill_md.is_symlink()
    assert [p.name for p in skill_md.parent.iterdir()] == ["SKILL.md"]


def test_independent_process_writer_is_preserved(skills, tmp_path, monkeypatch):
    """The uncooperative writer is a different OS process with its own synthetic environment."""
    skill_dir = skills / NAME
    synthetic = {"HOME": str(tmp_path / "child-home"), "HERMES_HOME": str(tmp_path / "child-hermes"),
                 "TMPDIR": str(tmp_path / "child-tmp"), "PATH": os.defpath, "LC_ALL": "C.UTF-8"}
    for key in ("HOME", "HERMES_HOME", "TMPDIR"):
        Path(synthetic[key]).mkdir()
    script = ("import sys, pathlib; pathlib.Path(sys.argv[1]).write_bytes(sys.argv[2].encode('utf-8'))")

    def plant():
        subprocess.run([sys.executable, "-I", "-c", script, str(skill_dir / "SKILL.md"), HUMAN_TEXT.decode("utf-8")],
                       env=synthetic, check=True, timeout=WAIT)

    result = _AfterMkdirBarrier(monkeypatch, skill_dir, plant).run(lambda: _approve(_payload()))
    _assert_refused_and_preserved(result, skill_dir, HUMAN_TEXT)


def test_unlocked_concurrent_creators_yield_exactly_one_success(skills, monkeypatch):
    """Direct ``_create_skill`` callers (the dashboard route) take no skill lock: one winner, no lost success."""
    count = 6
    skill_dir = skills / NAME
    barrier = threading.Barrier(count, timeout=WAIT)
    real_mkdir = Path.mkdir

    def mkdir(path_self, *a, **kw):
        try:
            return real_mkdir(path_self, *a, **kw)
        finally:
            if path_self == skill_dir:
                barrier.wait()  # everyone has seen the empty dir (made or adopted) before anyone publishes

    monkeypatch.setattr(Path, "mkdir", mkdir)
    contents = [ROBOT.replace("robot wrote this", f"writer {i} wrote this") for i in range(count)]
    results: list = [None] * count

    def run(i):
        results[i] = sm._create_skill(NAME, contents[i])

    threads = [threading.Thread(target=run, args=(i,)) for i in range(count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(WAIT * 2)
    assert not any(t.is_alive() for t in threads)
    winners = [i for i, r in enumerate(results) if r["success"]]
    assert len(winners) == 1, [r.get("error") or "ok" for r in results]
    assert (skill_dir / "SKILL.md").read_text(encoding="utf-8-sig") == contents[winners[0]]
    assert all("already" in r["error"] for i, r in enumerate(results) if i not in winners)
    assert sorted(p.name for p in skill_dir.iterdir()) == ["SKILL.md"]


# --- cleanup paths never remove something that is not ours -------------------------------------

def test_scan_block_still_removes_our_own_file_and_empty_dir(skills):
    with patch.object(sm, "_security_scan_skill", return_value="blocked"):
        result = sm._create_skill(NAME, ROBOT)
    assert result["success"] is False and "blocked" in result["error"]
    assert not (skills / NAME).exists()


def test_scan_block_does_not_delete_a_file_swapped_in_during_the_scan(skills):
    skill_md = skills / NAME / "SKILL.md"

    def scan_while_writer_replaces(_skill_dir):
        tmp = skill_md.with_name("human.tmp")
        tmp.write_bytes(HUMAN_TEXT)
        os.replace(tmp, skill_md)  # an editor's atomic save lands during the (slow) scan
        return "blocked"

    with patch.object(sm, "_security_scan_skill", side_effect=scan_while_writer_replaces):
        result = sm._create_skill(NAME, ROBOT)
    assert result["success"] is False and "blocked" in result["error"]
    assert skill_md.read_bytes() == HUMAN_TEXT
    assert [p.name for p in skill_md.parent.iterdir()] == ["SKILL.md"]


# --- ordinary create semantics are unchanged --------------------------------------------------

def test_ordinary_create_publishes_exact_bytes_and_leaves_nothing_else(skills):
    result = _approve(_payload(category="ops"))
    assert result["success"] is True and result["category"] == "ops"
    skill_dir = skills / "ops" / NAME
    assert (skill_dir / "SKILL.md").read_bytes() == ROBOT.encode("utf-8")
    assert [p.name for p in skill_dir.iterdir()] == ["SKILL.md"]
    assert sm._find_skill(NAME)["path"] == skill_dir


@pytest.mark.platforms("posix")
def test_new_skill_md_is_0644_even_under_a_restrictive_umask(skills):
    old = os.umask(0o077)
    try:
        assert sm._create_skill(NAME, ROBOT)["success"] is True
    finally:
        os.umask(old)
    assert stat.S_IMODE((skills / NAME / "SKILL.md").stat().st_mode) == 0o644


def test_empty_leftover_directory_is_still_adopted_and_foreign_content_is_still_refused(skills):
    (skills / NAME).mkdir()
    assert sm._create_skill(NAME, ROBOT)["success"] is True
    (skills / "busy").mkdir()
    (skills / "busy" / "notes.txt").write_bytes(b"n")
    refused = sm._create_skill("busy", ROBOT.replace(NAME, "busy"))
    assert refused["success"] is False and "Choose another name" in refused["error"]
    assert (skills / "busy" / "notes.txt").read_bytes() == b"n"


def test_duplicate_create_is_still_refused_and_edit_still_overwrites(skills):
    assert sm._create_skill(NAME, ROBOT)["success"] is True
    again = sm._create_skill(NAME, ROBOT.replace("robot wrote this", "second"))
    assert again["success"] is False and "already exists" in again["error"]
    assert "robot wrote this" in (skills / NAME / "SKILL.md").read_text(encoding="utf-8-sig")
    edited = sm._edit_skill(NAME, ROBOT.replace("robot wrote this", "edited"))
    assert edited["success"] is True
    assert "edited" in (skills / NAME / "SKILL.md").read_text(encoding="utf-8-sig")
