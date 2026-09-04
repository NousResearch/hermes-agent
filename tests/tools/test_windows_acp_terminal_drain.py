"""Regression tests for the Windows/ACP terminal subprocess fixes.

Two Windows-specific hangs were observed when Hermes' terminal ran under an ACP
host (e.g. a Node server) that gave Hermes piped stdio: a bash grandchild could
inherit and hold a captured pipe's write end open, so

  * ``_bash_starts`` (bash discovery) hung in ``subprocess.run(capture_output=
    True, timeout=...)`` because the reader-thread join inside ``communicate()``
    is unbounded on Windows — the timeout only bounds the child-exit wait; and
  * ``_wait_for_process``'s Windows drain blocked in a plain ``os.read`` waiting
    for EOF that never came (and then ``proc.stdout.close()`` blocked on the
    in-flight read), because it lacked the POSIX ``select`` path's ``proc.poll()``
    early-exit.

The fixes (temp-file probe capture; PeekNamedPipe + proc.poll() drain) are
Windows-specific, but these tests are cross-platform on purpose: they exercise
the same code paths on POSIX (where the ``select`` drain has always handled the
grandchild case) so a future change that regresses either platform is caught,
and they prove the fixes did not change POSIX behavior. Everything here uses a
real bash; the module skips when no bash is available.
"""
import os
import time

import pytest

os.environ.setdefault("HERMES_HOME", os.path.expanduser("~/.hermes"))

from tools.environments.local import LocalEnvironment, _find_bash, _bash_starts  # noqa: E402
import tools.environments.local as _local  # noqa: E402


def _bash_available() -> bool:
    try:
        return bool(_find_bash())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _bash_available(), reason="no bash available")


@pytest.fixture
def env():
    e = LocalEnvironment(cwd=os.environ["HERMES_HOME"], timeout=60)
    yield e
    try:
        e.cleanup()
    except Exception:
        pass


def test_bash_probe_is_bounded():
    """``_bash_starts`` must return promptly (Fix 1: the Windows capture no
    longer joins reader threads that a grandchild can keep alive)."""
    bash = _find_bash()
    _local._bash_starts_cache.pop(bash, None)  # force a real probe
    t0 = time.time()
    ok = _bash_starts(bash)
    dt = time.time() - t0
    assert ok is True
    # The probe's own timeout is 15s; a hang would run to the grandchild's
    # lifetime / forever. Anything past ~20s means the bound is not enforced.
    assert dt < 20.0, f"bash probe took {dt:.1f}s (timeout not bounded)"


def test_drain_grandchild_holding_pipe_does_not_block(env):
    """Fix 2: a detached grandchild that inherits stdout and outlives bash must
    not keep the drain (or the follow-up close) blocked. The command backgrounds
    ``sleep 8`` inside a subshell so bash exits immediately while ``sleep`` holds
    the pipe."""
    proc = env._run_bash("printf 'HELLO_DRAIN\\n'; ( sleep 8 & )", login=False, timeout=60)
    t0 = time.time()
    res = env._wait_for_process(proc, timeout=60)
    dt = time.time() - t0
    assert "HELLO_DRAIN" in res.get("output", "")
    assert dt < 5.0, f"drain blocked {dt:.1f}s on a grandchild-held pipe"


def test_timeout_is_bounded(env):
    """A non-terminating child must be interrupted at the configured deadline."""
    proc = env._run_bash("sleep 300", login=False, timeout=3)
    t0 = time.time()
    res = env._wait_for_process(proc, timeout=3)
    dt = time.time() - t0
    assert res.get("returncode") == 124, f"expected timeout rc 124, got {res.get('returncode')}"
    assert dt < 10.0, f"timeout took {dt:.1f}s to fire"


def test_output_fidelity_preserved(env):
    """The poll/peek drain must not drop output: every line survives."""
    proc = env._run_bash("for i in $(seq 1 2000); do echo LINE_$i; done", login=False, timeout=60)
    res = env._wait_for_process(proc, timeout=60)
    out = res.get("output", "")
    assert out.count("LINE_") == 2000, f"expected 2000 lines, got {out.count('LINE_')}"
    assert "LINE_2000" in out
