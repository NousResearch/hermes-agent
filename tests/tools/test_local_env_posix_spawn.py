"""End-to-end tests for the real POSIX spawn path in ``LocalEnvironment._run_bash``.

``_run_bash`` starts every command with ``os.posix_spawn`` on POSIX (no ``fork`` of a
multi-GB heap, see ``tools/environments/local.py``), which means the two things
``subprocess.Popen`` used to provide for free are now this module's responsibility and need
their own coverage:

* session leadership — the child must lead its own process group, or
  ``_kill_process_group_posix`` stops reaping descendants;
* descriptor hygiene (``close_fds=True``) — a child must not inherit the parent's ~190
  open descriptors, which is also what ``_inherited_fd_close_actions`` needs its
  ``F_GETFD`` guard for: an unguarded ``POSIX_SPAWN_CLOSE`` list makes ``posix_spawn``
  abort with ``EBADF`` instead of starting a child at all.

These tests spawn real children (no fakes) so a regression in the spawn wiring itself is
visible; the faking style is left to ``test_local_env_cwd_recovery.py``, which covers the
cwd side of the same call.
"""

import os
import subprocess
import sys
from unittest.mock import patch

import pytest

from tools.environments.local import LocalEnvironment, _PosixSpawnProcess

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX spawn path only")


def _run_bash(tree_cwd: str, cmd: str, timeout: int = 30):
    """Run *cmd* through the real ``_run_bash`` path with a live session snapshot skipped."""
    with patch.object(LocalEnvironment, "init_session", autospec=True, return_value=None):
        env = LocalEnvironment(cwd=tree_cwd, timeout=timeout)
    proc = env._run_bash(cmd)
    result = env._wait_for_process(proc, timeout=timeout)
    return proc, result


class TestSpawnSessionLeader:
    """The child must be a session leader (``pgid == pid``) — the teardown contract."""

    def test_child_leads_its_own_process_group(self, tmp_path):
        proc, result = _run_bash(str(tmp_path), "echo $$; ps -o pgid= -p $$")

        # A fallback to the fork path would satisfy the assertions below vacuously
        # (``start_new_session=True`` does the same job), so pin the mechanism under test.
        assert isinstance(proc, _PosixSpawnProcess), (
            f"posix_spawn did not run; got {type(proc).__name__}: {result.get('output')!r}")

        reported = [int(tok) for tok in result["output"].split()]
        assert reported == [proc.pid, proc.pid], (
            f"child pid/pgid {reported} != spawned pid {proc.pid}: {result['output']!r}")
        # ... and that group is the child's own, not one inherited from this test process.
        assert proc.pid != os.getpgrp()


def _fd_probe_command() -> str:
    """Bash-only "which fds do I hold" probe.

    ``/dev/fd`` lists fd *numbers*; the listing's own directory handle is one of them and is
    closed again by the time the loop body runs, so ``-e`` (stat) both filters those stale
    numbers out and leaves only descriptors that are genuinely open. No forks: one ``fork``
    here would itself inherit the very descriptors under test.

    Identity (``-ef`` against a witness path) is deliberately NOT used: measured on macOS,
    ``/dev/fd/N`` does not stat as the file it points at, so that check can never fire. The
    caller compares the whole set against the ``Popen`` reference child instead.
    """
    return (
        "for f in /dev/fd/*; do "
        "  n=${f##*/}; "
        "  [ -e \"$f\" ] || continue; "
        "  echo OPEN $n; "
        "done"
    )


def _parse_fd_report(output: str) -> set:
    """The set of open fd numbers an ``_fd_probe_command`` run reported."""
    open_fds = set()
    for line in output.splitlines():
        parts = line.split()
        if len(parts) != 2 or not parts[1].isdigit():
            continue
        if parts[0] == "OPEN":
            open_fds.add(int(parts[1]))
    return open_fds


def _reference_child_fd_set(cmd: str, cwd: str) -> set:
    """The same probe through ``subprocess.Popen`` — the ``close_fds=True`` reference.

    ``_run_bash`` falls back to exactly this call, so the posix_spawn path is expected to
    produce the same child fd set; comparing against the reference keeps the assertion
    independent of how many transient descriptors the probe's own ``/dev/fd`` listing holds
    on a given platform."""
    proc = subprocess.Popen(
        ["/bin/bash", "-c", cmd], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, start_new_session=True, cwd=cwd)
    out = proc.communicate(timeout=30)[0]
    assert proc.returncode == 0, out
    return _parse_fd_report(out)


class TestSpawnChildDescriptors:
    """``close_fds=True`` semantics survive the switch away from ``Popen``."""

    def test_child_holds_stdio_and_nothing_of_ours(self, tmp_path):
        witness = tmp_path / "witness.txt"
        witness.write_text("witness")
        # CLOEXEC cleared on purpose: an inheritable descriptor is the case the close
        # actions must catch, and the case a CLOEXEC-only implementation lets through.
        witness_fd = os.open(str(witness), os.O_RDONLY)
        os.set_inheritable(witness_fd, True)
        cmd = _fd_probe_command()
        try:
            proc, result = _run_bash(str(tmp_path), cmd)
            assert isinstance(proc, _PosixSpawnProcess), (
                f"posix_spawn did not run; got {type(proc).__name__}")
            open_fds = _parse_fd_report(result["output"])
            reference = _reference_child_fd_set(cmd, str(tmp_path))
        finally:
            os.close(witness_fd)

        assert open_fds == reference, (
            f"child fds {sorted(open_fds)} != Popen(close_fds=True) reference "
            f"{sorted(reference)} (probe fd {witness_fd} was open in the parent)")
        # Names the leak when the number is not one the reference child reuses transiently.
        if witness_fd not in reference:
            assert witness_fd not in open_fds, (
                f"child inherited the probe descriptor {witness_fd} ({witness})")
        assert {0, 1, 2} <= open_fds, f"stdio missing from the child fd set {sorted(open_fds)}"
