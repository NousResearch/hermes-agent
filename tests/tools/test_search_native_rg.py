"""Local POSIX content/file search runs rg as a direct argv subprocess.

The shell path pays two bash spawns per search (``test -e`` probe + ``set -o
pipefail; rg ... | head``). On a ``LocalEnvironment`` the same rg argv can run
natively with a bounded stdout read; the parser and every argument builder are
shared, so the two transports must agree on results.
"""

import json
import sys

import pytest

from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="native rg lane is POSIX-only")


@pytest.fixture
def tree(tmp_path):
    (tmp_path / "a.py").write_text("needle one\nplain\nneedle two\n")
    (tmp_path / "b.txt").write_text("needle three\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.py").write_text("needle four\n")
    return tmp_path


@pytest.fixture(scope="module")
def _local_env(tmp_path_factory):
    """One real LocalEnvironment per module (constructing one costs ~0.8 s)."""
    return LocalEnvironment(cwd=str(tmp_path_factory.mktemp("native-rg")))


@pytest.fixture
def ops_factory(_local_env):
    """``make(tree, spy)`` → ShellFileOperations over the shared env, every execute recorded in ``spy``."""
    real = type(_local_env).execute.__get__(_local_env, type(_local_env))

    def make(tree, spy):
        _local_env.cwd = str(tree)

        def recording(command, *a, **kw):
            spy.append(command)
            return real(command, *a, **kw)

        _local_env.execute = recording
        return ShellFileOperations(_local_env, cwd=str(tree))

    yield make
    _local_env.__dict__.pop("execute", None)


def _normalized(result):
    d = result.to_dict()
    for key in ("matches", "files"):
        if key in d:
            d[key] = sorted(json.dumps(item, sort_keys=True) for item in d[key])
    return d


def test_native_search_never_touches_the_shell_and_matches_shell_results(tree, ops_factory, monkeypatch):
    cases = [
        dict(pattern="needle", path=str(tree)),
        # (no offset/limit slicing here: rg's parallel walk orders files
        # nondeterministically, so a page differs run-to-run on either transport)
        dict(pattern="needle", path=str(tree), output_mode="count"),
        dict(pattern="needle", path=str(tree), output_mode="files_only", file_glob="*.py"),
        dict(pattern="NEEDLE_NOPE", path=str(tree)),  # zero-match probes
        dict(pattern="*.py", path=str(tree), target="files"),
        dict(pattern="needle", path=str(tree / "missing")),
    ]
    for case in cases:
        monkeypatch.setenv("HERMES_NATIVE_FILE_READ", "0")
        shell = _normalized(ops_factory(tree, []).search(**case))
        monkeypatch.setenv("HERMES_NATIVE_FILE_READ", "1")
        calls = []
        native = _normalized(ops_factory(tree, calls).search(**case))
        assert native == shell, case
        # rg resolution (``command -v rg``) still goes through the shell once; the
        # existence probe and the rg pipeline itself must not.
        assert not [c for c in calls if "pipefail" in c or c.startswith("test -e")], case


def test_native_runner_honours_deadline_and_interrupt_while_rg_is_silent(tree, ops_factory, monkeypatch):
    """A search producing no output must still stop at the deadline / on /stop
    (the shell path gets this from ``_wait_for_process``)."""
    import threading
    import time

    from tools import interrupt

    ops = ops_factory(tree, [])
    started = time.monotonic()
    result = ops._run_rg_native(["sh", "-c", "'sleep 30'"], 5, timeout=1)
    assert result.exit_code == 124 and time.monotonic() - started < 5

    tid = threading.get_ident()
    threading.Timer(0.3, lambda: interrupt.set_interrupt(True, tid)).start()
    try:
        started = time.monotonic()
        result = ops._run_rg_native(["sh", "-c", "'sleep 30'"], 5, timeout=30)
    finally:
        interrupt.set_interrupt(False, tid)
    assert result.exit_code == 130 and time.monotonic() - started < 5


def test_kill_switch_routes_search_back_to_the_shell(tree, ops_factory, monkeypatch):
    monkeypatch.setenv("HERMES_NATIVE_FILE_READ", "0")
    calls = []
    result = ops_factory(tree, calls).search(pattern="needle", path=str(tree))
    assert result.total_count == 4
    assert any(c.startswith("test -e") for c in calls)
    assert any("pipefail" in c and "rg" in c for c in calls)


def test_limit_hit_keeps_drained_matches_when_group_kill_is_refused(tree, ops_factory, monkeypatch):
    """Reaching ``limit`` takes the early-stop branch that TERMs rg's group. macOS answers
    ``killpg`` with EPERM (not ESRCH) for a zombie-only group; that must not surface as a tool
    error nor discard the matches already drained (#116855)."""
    import os

    monkeypatch.setenv("HERMES_NATIVE_FILE_READ", "1")
    ops = ops_factory(tree, [])
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: (_ for _ in ()).throw(PermissionError(1, "Operation not permitted")))
    result = ops.search(pattern="needle", path=str(tree), limit=2)
    assert not result.error and len(result.matches) == 2, result.to_dict()


@pytest.mark.parametrize("pattern,limit,expected_count", [
    ("needle", 2, 2),
    ("needle", 50, 4),
    ("absent-token", 50, 0),
    ("[", 50, None),
])
def test_exited_rg_cleanup_preserves_search_outcome(tree, ops_factory, monkeypatch,
                                                    pattern, limit, expected_count):
    """OBL-5929: exit between poll and getpgid must not replace real rg results."""
    import os
    import subprocess

    from tools import file_tools
    from tools.registry import registry

    monkeypatch.setenv("HERMES_NATIVE_FILE_READ", "1")
    ops = ops_factory(tree, [])
    executable = ops._resolve_command("rg")
    assert executable, "this regression requires real ripgrep"
    monkeypatch.setattr(file_tools, "_get_file_ops", lambda task_id: ops)
    real_popen = subprocess.Popen
    raced = []

    def popen_with_exit_at_poll(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        if os.path.basename(args[0][0]) == "rg":
            real_poll = proc.poll

            def exit_after_poll():
                # Report the live observation, then schedule rg's exit/reap before
                # the caller reaches getpgid. No fake rg output or kill helper.
                proc.poll = real_poll
                proc.wait(timeout=5)
                raced.append(proc.pid)
                return None

            proc.poll = exit_after_poll
        return proc

    monkeypatch.setattr(subprocess, "Popen", popen_with_exit_at_poll)
    result = json.loads(registry.dispatch("search_files", {
        "pattern": pattern, "path": str(tree), "limit": limit,
    }, task_id="exited-rg-regression"))
    assert raced
    if expected_count is None:
        assert "Search failed" in result.get("error", ""), result
    else:
        assert not result.get("error"), result
        assert len(result.get("matches", [])) == expected_count, result
        assert all("needle" in match["content"] for match in result.get("matches", []))


@pytest.mark.parametrize("failure", [PermissionError(1, "cleanup denied"), OSError(5, "cleanup I/O failure")])
def test_native_cleanup_does_not_hide_other_os_errors(tree, ops_factory, monkeypatch, failure):
    """Only an already-gone process is an idempotent cleanup outcome."""
    from tools.environments import local

    ops = ops_factory(tree, [])
    real_kill = local._kill_process_group_posix

    def fail_cleanup(proc):
        real_kill(proc)
        proc.wait()
        proc.stdout.close()
        raise failure

    monkeypatch.setattr(local, "_kill_process_group_posix", fail_cleanup)
    with pytest.raises(type(failure)) as caught:
        ops._run_rg_native(["sh", "-c", "'printf \"needle\\n\"; sleep 30'"], 1, timeout=5)
    assert caught.value is failure
