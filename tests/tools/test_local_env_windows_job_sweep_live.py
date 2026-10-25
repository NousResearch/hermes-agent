"""LIVE Windows proof: the fallback kill sweep reaches a real Git Bash grep that
MSYS parent handoff orphaned (#73403 class).

Git Bash runs external commands through fork+exec stubs that may exit once the
new image runs, so the Windows parent chain from the bash wrapper to a grandchild
(grep) breaks. Neither psutil's ``children()`` snapshot nor ``taskkill /T`` —
both parent-pid walks — can ever reach such a process again. The spawn-time
membership-only Job Object (``_hermes_win32_job``) contains it regardless of
ancestry, and ``_kill_process_windows`` terminates the job as its final sweep.

These tests drive the REAL ``LocalEnvironment._run_bash`` (real Git Bash, real
grep.exe) on the live Windows process table — no mocks. Grep ownership is proven
by a private output-file handle (unique per test), never by argv.
"""

from __future__ import annotations

import contextlib
import subprocess
import time
import uuid
from pathlib import Path

import psutil
import pytest

from tools.environments import local as local_env
from tools.environments.local import LocalEnvironment, _windows_to_msys_path

pytestmark = pytest.mark.platforms("windows")  # live native Git Bash proof

# grep stays alive ~30s blocked on the pipeline's open read end; the first
# line-buffered marker write proves it is actively working before we kill.
_PIPELINE = (
    "{{ printf '%s\\n' '{marker}'; sleep 30; }} "
    "| grep --line-buffered -F {marker} > {outfile}"
)


def _owned_greps(outfile: Path) -> list:
    """grep.exe processes holding OUR unique output file open (handle identity)."""
    greps: list = []
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            if (proc.info["name"] or "").lower() != "grep.exe":
                continue
            if any(Path(f.path).resolve() == outfile for f in proc.open_files()):
                greps.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return greps


def _spawn_orphan_grep(env: LocalEnvironment, tmp_path: Path):
    """Run the real pipeline; return (wrapper proc, owned grep procs) once the
    marker line proves grep is live."""
    marker = f"e01_{uuid.uuid4().hex}"
    outfile = tmp_path / f"{marker}.out"
    proc = env._run_bash(_PIPELINE.format(
        marker=marker, outfile=_windows_to_msys_path(str(outfile))))
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if outfile.exists() and marker in outfile.read_text(errors="replace"):
            break
        if proc.poll() is not None:
            raise AssertionError(f"pipeline exited early rc={proc.returncode}")
        time.sleep(0.05)
    else:
        raise AssertionError("grep never wrote its marker line (never ran?)")
    greps: list = []
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not greps:
        greps = _owned_greps(outfile)
        time.sleep(0.1)  # process-table visibility lag
    assert greps, "probe-owned grep not found via its private output handle"
    return proc, greps


@pytest.fixture()
def env(tmp_path):
    e = LocalEnvironment(cwd=str(tmp_path), timeout=60)
    yield e
    e.cleanup()


def _assert_and_cleanup(proc, greps):
    try:
        _, alive = psutil.wait_procs(greps, timeout=5)
        assert not alive, f"kill sweep missed probe-owned Git Bash grep: {[p.pid for p in alive]}"
        assert proc.poll() is not None, "bash wrapper still running after the sweep"
    finally:
        # Finalizer touches ONLY the probe-owned greps and its own wrapper.
        for grep in greps:
            with contextlib.suppress(psutil.NoSuchProcess):
                grep.kill()
        if proc.poll() is None:
            proc.kill()
        with contextlib.suppress(Exception):
            proc.wait(timeout=5)


def test_job_sweep_kills_msys_orphan_grep_when_tree_kill_fails(env, tmp_path, monkeypatch):
    """The exact orphan shape: taskkill /T fails outright, the psutil snapshot is empty
    (parent handoff), and the job sweep is the only thing that can reach grep."""
    import gateway.status as gs

    proc, greps = _spawn_orphan_grep(env, tmp_path)
    children = psutil.Process(proc.pid).children(recursive=True)
    # Precondition: the gap is real on the host — grep is NOT reachable from the
    # wrapper by parent pid. If this ever fails, MSYS linkage changed and this
    # test no longer proves the orphan case.
    assert not ({g.pid for g in greps} & {c.pid for c in children}), (
        f"grep unexpectedly visible as a child: children={[(c.pid, c.name()) for c in children]}")

    def _boom(*args, **kwargs):
        raise subprocess.TimeoutExpired("taskkill", 10)

    monkeypatch.setattr(gs, "terminate_pid", _boom)
    local_env._kill_process_windows(proc)
    _assert_and_cleanup(proc, greps)


def test_job_sweep_kills_msys_orphan_grep_after_taskkill(env, tmp_path):
    """Realistic path: taskkill /T succeeds on the wrapper (identity guard intact)
    but cannot reach the MSYS-detached grep; the job sweep must still end it."""
    proc, greps = _spawn_orphan_grep(env, tmp_path)
    local_env._kill_process_windows(proc)
    _assert_and_cleanup(proc, greps)
