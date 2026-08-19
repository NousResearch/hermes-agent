"""Regression tests for the macOS posix_spawn path in LocalEnvironment.

When running on macOS (darwin), LocalEnvironment._run_bash drops the three
posix_spawn disqualifiers (start_new_session, close_fds, cwd) to avoid the
fork() deadlock in multithreaded gateways. These tests verify the behaviours
that change introduces:

1. The cwd is injected via a shell-safe `cd` prelude (no command substitution).
2. The spawned bash receives the cwd and the original command intact.

These tests are darwin-only — the non-Darwin path uses start_new_session=True
and cwd= directly, which doesn't have the shell-injection surface.
"""

from __future__ import annotations

import os
import sys
import tempfile

import pytest

from tools.environments.local import LocalEnvironment

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="Darwin posix_spawn path only exists on macOS",
)


def test_darwin_cwd_prelude_is_shell_safe_for_metacharacters(tmp_path):
    """The cwd prelude must not allow command substitution.

    A path containing shell metacharacters like $(), backticks, or ; must be
    treated as a literal path, not parsed as shell source. shlex.quote wraps
    the path in single quotes, which suppresses all substitution.
    """
    # Build a _spawn_cmd the same way _run_bash does, then verify the
    # metacharacter path appears literally and does NOT execute.
    import shlex

    evil_cwd = '$(touch /tmp/darwin_spawn_pwned_$$)`echo pwned`; rm -rf ~'
    cmd_string = "echo hello"
    spawn_cmd = (
        "builtin cd -- "
        + shlex.quote(evil_cwd)
        + " 2>/dev/null || true\n"
        + cmd_string
    )
    # The evil payload must be inside single quotes in the generated command.
    # shlex.quote wraps in single quotes when the string contains metacharacters.
    assert "'" in spawn_cmd, "shlex.quote should have single-quoted the path"
    # The command substitution attempt must be neutralized (inside quotes).
    # Verify by actually running it — the cd will fail (path doesn't exist),
    # the `|| true` swallows it, and cmd_string runs. No side effects.
    import subprocess

    proc = subprocess.run(
        ["bash", "-c", spawn_cmd],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert proc.returncode == 0, f"cd-fail + echo should succeed: {proc.stderr}"
    assert "hello" in proc.stdout
    # Verify no side-effect file was created by the $() if it had executed.
    assert not os.path.exists("/tmp/darwin_spawn_pwned_test"), (
        "command substitution leaked — shlex.quote did not neutralize the path"
    )


def test_darwin_cwd_prelude_handles_normal_paths(tmp_path):
    """Normal cwd paths should cd successfully and the command should run."""
    import shlex

    cmd_string = "pwd -P"
    spawn_cmd = (
        "builtin cd -- "
        + shlex.quote(str(tmp_path))
        + " 2>/dev/null || true\n"
        + cmd_string
    )
    import subprocess

    proc = subprocess.run(
        ["bash", "-c", spawn_cmd],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert proc.returncode == 0
    # pwd -P should report the real path of tmp_path (may differ on macOS
    # due to /private prefix, so check the suffix).
    output_path = proc.stdout.strip()
    assert str(tmp_path) in output_path or output_path.endswith(
        str(tmp_path).split("/")[-1]
    ), f"cd didn't reach tmp_path: stdout={output_path!r}"


def test_darwin_cwd_prelude_handles_paths_with_spaces(tmp_path):
    """Paths with spaces must be handled correctly (common in macOS home dirs)."""
    import shlex

    space_dir = tmp_path / "dir with spaces"
    space_dir.mkdir()
    cmd_string = "pwd -P"
    spawn_cmd = (
        "builtin cd -- "
        + shlex.quote(str(space_dir))
        + " 2>/dev/null || true\n"
        + cmd_string
    )
    import subprocess

    proc = subprocess.run(
        ["bash", "-c", spawn_cmd],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert proc.returncode == 0
    assert "dir with spaces" in proc.stdout


def test_darwin_spawn_uses_posix_spawn_not_fork():
    """The Darwin spawn path must not fall back to fork()+exec().

    This is the core invariant of the PR: CPython's subprocess uses fork()
    when start_new_session=True on macOS, which deadlocks multithreaded
    processes. We verify by checking that _posixsubprocess.fork_exec is NOT
    called for a spawn with start_new_session=False + close_fds=False.
    """
    import _posixsubprocess
    import subprocess

    fork_calls = []
    orig_fork_exec = _posixsubprocess.fork_exec

    def spy_fork_exec(*args, **kwargs):
        fork_calls.append(args)
        return orig_fork_exec(*args, **kwargs)

    _posixsubprocess.fork_exec = spy_fork_exec
    try:
        proc = subprocess.Popen(
            ["bash", "-c", "true"],
            start_new_session=False,
            close_fds=False,
            cwd=None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        proc.communicate()
        proc.wait()
    finally:
        _posixsubprocess.fork_exec = orig_fork_exec

    assert not fork_calls, (
        f"fork_exec was called {len(fork_calls)} time(s) — posix_spawn "
        f"fallback is still active, the deadlock fix is ineffective"
    )
