"""Tests for save_trajectory's cross-process advisory lock.

The JSONL append must be serialized between processes: two writers that
interleave produce torn/corrupt JSON lines. The lock fails closed — when
it cannot be acquired within the timeout, nothing is written.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from agent import trajectory


@pytest.fixture
def traj_file(tmp_path):
    return tmp_path / "trajectory_samples.jsonl"


def test_save_appends_valid_jsonl(traj_file):
    trajectory.save_trajectory(
        [{"from": "human", "value": "hi"}], "m", completed=True,
        filename=str(traj_file),
    )
    lines = traj_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["model"] == "m"


def test_lock_timeout_skips_write_fail_closed(traj_file, monkeypatch):
    """A lock that can never be acquired must NOT fall through to a write."""

    class _AlwaysBusy:
        def __enter__(self):
            raise TimeoutError("simulated contention")

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(trajectory, "_trajectory_lock", lambda f: _AlwaysBusy())
    trajectory.save_trajectory(
        [{"from": "human", "value": "hi"}], "m", completed=True,
        filename=str(traj_file),
    )
    assert not traj_file.exists(), "write happened despite lock failure"


def test_sidecar_lock_created_next_to_output(traj_file):
    trajectory.save_trajectory([], "m", completed=True, filename=str(traj_file))
    lock_path = Path(str(traj_file) + ".lock")
    assert lock_path.exists()


def _wait_for(path, timeout_steps=1000):
    for _ in range(timeout_steps):
        if path.exists():
            return True
        __import__("time").sleep(0.01)
    return False


def test_lock_excludes_another_process(tmp_path):
    """Real cross-process exclusion, modeled on the cron jobs.json lock test."""
    if trajectory.fcntl is None and trajectory.msvcrt is None:
        pytest.skip("no advisory locking primitive on this platform")

    traj_file = tmp_path / "trajectory_samples.jsonl"
    ready = tmp_path / "ready"
    release = tmp_path / "release"
    blocker_acquired = tmp_path / "blocker_acquired"
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(trajectory.__file__)))

    holder = tmp_path / "holder.py"
    holder.write_text(
        textwrap.dedent(
            f"""
            import os, sys, time
            sys.path.insert(0, {repo_root!r})
            from agent import trajectory

            with trajectory._trajectory_lock({str(traj_file)!r}):
                open({str(ready)!r}, "w").write("1")
                for _ in range(1000):
                    if os.path.exists({str(release)!r}):
                        break
                    time.sleep(0.01)
            """
        )
    )

    blocker = tmp_path / "blocker.py"
    blocker.write_text(
        textwrap.dedent(
            f"""
            import sys
            sys.path.insert(0, {repo_root!r})
            from agent import trajectory

            with trajectory._trajectory_lock({str(traj_file)!r}):
                open({str(blocker_acquired)!r}, "w").write("1")
            """
        )
    )

    child = subprocess.Popen([sys.executable, str(holder)])
    try:
        assert _wait_for(ready), "child never acquired the trajectory lock"
        assert child.poll() is None, "holder process exited early"

        # While the holder owns the sidecar, this process must be unable to
        # take the same kernel-level lock (proves it is not just in-process).
        lock_path = str(traj_file) + ".lock"
        if trajectory.fcntl is not None:
            fd = os.open(lock_path, os.O_RDWR | os.O_CREAT)
            try:
                with pytest.raises(OSError):
                    trajectory.fcntl.flock(fd, trajectory.fcntl.LOCK_EX | trajectory.fcntl.LOCK_NB)
            finally:
                os.close(fd)
        else:
            with open(lock_path, "r+", encoding="utf-8") as lf:
                lf.seek(0)
                with pytest.raises(OSError):
                    trajectory.msvcrt.locking(lf.fileno(), trajectory.msvcrt.LK_NBLCK, 1)

        # A second process must still be waiting while the lock is held.
        release.touch()  # keep held until we're ready below
        release.unlink()
        blocker_child = subprocess.Popen([sys.executable, str(blocker)])
        try:
            import time as t

            t.sleep(0.2)
            assert not blocker_acquired.exists(), (
                "second process entered the critical section while held"
            )
        finally:
            release.write_text("1")
            child.wait(timeout=15)
            blocker_child.wait(timeout=15)

        assert blocker_acquired.exists(), "second process never acquired after release"
    finally:
        release.write_text("1")


def test_sequential_saves_stay_wellformed(traj_file):
    for i in range(5):
        trajectory.save_trajectory(
            [{"from": "human", "value": f"msg-{i}"}], "m", completed=True,
            filename=str(traj_file),
        )
    entries = [json.loads(line) for line in traj_file.read_text(encoding="utf-8").splitlines()]
    assert [e["conversations"][0]["value"] for e in entries] == [
        f"msg-{i}" for i in range(5)
    ]
