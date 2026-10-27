"""``_run_rg_native`` must not turn a finished bounded search into a tool error
when the group-kill helper fails on an already-exited rg.

Once ripgrep has exited but is not yet reaped, macOS raises ESRCH from
``os.getpgid`` and EPERM from ``os.killpg`` inside ``_kill_process_group_posix``
(Linux returns success for both, so upstream CI never sees it). A
``search_files`` call with more matches than the fetch limit stops reading
while rg is still exiting, so it can land in that window and return a tool
error (20/20 on an idle macOS host, about half the time under load).
The helper is monkeypatched to raise so the failure is deterministic on any
POSIX host; the child is a real process that is still alive when the bound is
reached, so the kill path is always taken.

Each test fails if ``_run_rg_native`` calls ``_kill_process_group_posix`` bare,
without catching OSError and falling back to ``proc.kill()`` (regressed in
2d7799275).
"""

import errno
import sys

import pytest

from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="native rg lane is POSIX-only")

# Emits more lines than the fetch limit, then stays alive (exec so the PID is
# the sleeper itself and a kill leaves no orphan holding the pipe open).
_CHILD_ARGV = ["sh", "-c", "'seq 1 20; exec sleep 30'"]
_FETCH_LIMIT = 5


@pytest.fixture(scope="module")
def _local_env(tmp_path_factory):
    """One real LocalEnvironment per module (constructing one costs ~0.8 s)."""
    return LocalEnvironment(cwd=str(tmp_path_factory.mktemp("native-rg-zombie")))


@pytest.fixture
def ops(_local_env, tmp_path):
    _local_env.cwd = str(tmp_path)
    return ShellFileOperations(_local_env, cwd=str(tmp_path))


@pytest.fixture
def raising_kill(monkeypatch, request):
    """Replace the group-kill helper with one that records the Popen and raises
    the OSError the test asks for; kills any child left behind afterwards."""
    seen = []

    def install(exc):
        def _kill(proc):
            seen.append(proc)
            raise exc

        monkeypatch.setattr("tools.environments.local._kill_process_group_posix", _kill)
        return seen

    yield install
    for proc in seen:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def _run_bounded(ops):
    try:
        return ops._run_rg_native(_CHILD_ARGV, _FETCH_LIMIT, timeout=10)
    except OSError as exc:
        pytest.fail(f"bounded native search raised {exc!r} instead of returning its results")


@pytest.mark.parametrize(
    "exc",
    [
        pytest.param(ProcessLookupError(errno.ESRCH, "No such process"), id="ESRCH"),
        pytest.param(PermissionError(errno.EPERM, "Operation not permitted"), id="EPERM"),
    ],
)
def test_bounded_search_returns_results_when_kill_helper_raises(ops, raising_kill, exc):
    seen = raising_kill(exc)
    result = _run_bounded(ops)
    assert seen, "kill path was not taken: the child exited before the bound, the test proved nothing"
    assert result.exit_code == 0
    assert result.stdout.splitlines() == ["1", "2", "3", "4", "5"]


def test_child_is_killed_and_reaped_when_kill_helper_raises(ops, raising_kill):
    seen = raising_kill(ProcessLookupError(errno.ESRCH, "No such process"))
    _run_bounded(ops)
    assert seen, "kill path was not taken: the child exited before the bound, the test proved nothing"
    proc = seen[0]
    # Reaped (wait() ran) and killed by a signal rather than left sleeping.
    assert proc.poll() is not None
    assert proc.returncode < 0
