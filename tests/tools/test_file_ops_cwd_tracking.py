"""Regression tests for cwd-staleness in ShellFileOperations.

The bug: ShellFileOperations captured the terminal env's cwd at __init__
time and used that stale value for every subsequent _exec() call.  When
a user ran ``cd`` via the terminal tool, ``env.cwd`` updated but
``ops.cwd`` did not.  Relative paths passed to patch/read/write/search
then targeted the wrong directory — typically the session's start dir
instead of the current working directory.

Observed symptom: patch_replace() returned ``success=True`` with a
plausible diff, but the user's ``git diff`` showed no change (because
the patch landed in a different directory's copy of the same file).

Fix: _exec() now prefers the LIVE ``env.cwd`` over the init-time
``self.cwd``.  Explicit ``cwd`` arg to _exec still wins over both.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from tools.file_operations import ShellFileOperations


class _FakeEnv:
    """Minimal terminal env that tracks cwd across execute() calls.

    Matches the real ``BaseEnvironment`` contract: ``cwd`` attribute plus
    an ``execute(command, cwd=...)`` method whose return dict carries
    ``output`` and ``returncode``.  Commands are executed in a real
    subdirectory so file system effects match production.
    """

    def __init__(self, start_cwd: str):
        self.cwd = start_cwd
        self.calls: list[dict] = []

    def execute(self, command: str, cwd: str = None, **kwargs) -> dict:
        import subprocess
        self.calls.append({"command": command, "cwd": cwd})
        # Simulate cd by updating self.cwd (the real env does the same
        # via _extract_cwd_from_output after a successful command)
        if command.strip().startswith("cd "):
            new = command.strip()[3:].strip()
            self.cwd = new
            return {"output": "", "returncode": 0}
        # Actually run the command — handle stdin via subprocess
        stdin_data = kwargs.get("stdin_data")
        proc = subprocess.run(
            ["bash", "-c", command],
            cwd=cwd or self.cwd,
            input=stdin_data,
            capture_output=True,
            text=True,
        )
        return {
            "output": proc.stdout + proc.stderr,
            "returncode": proc.returncode,
        }


class TestShellFileOpsCwdTracking:
    """_exec() must use live env.cwd, not the init-time cached cwd."""

    def test_exec_follows_env_cwd_after_cd(self, tmp_path):
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        (dir_a / "target.txt").write_text("content-a\n")
        (dir_b / "target.txt").write_text("content-b\n")

        env = _FakeEnv(start_cwd=str(dir_a))
        ops = ShellFileOperations(env, cwd=str(dir_a))
        assert ops.cwd == str(dir_a)  # init-time

        # Simulate the user running `cd b` in terminal
        env.execute(f"cd {dir_b}")
        assert env.cwd == str(dir_b)
        assert ops.cwd == str(dir_a), "ops.cwd is still init-time (fallback only)"

        # Reading a relative path must now hit dir_b, not dir_a
        result = ops._exec("cat target.txt")
        assert result.exit_code == 0
        assert "content-b" in result.stdout, (
            f"Expected dir_b content, got {result.stdout!r}. "
            "Stale ops.cwd leaked through — _exec must prefer env.cwd."
        )

    def test_patch_replace_targets_live_cwd_not_init_cwd(self, tmp_path):
        """The exact bug reported: patch lands in wrong dir after cd."""
        dir_a = tmp_path / "main"
        dir_b = tmp_path / "worktree"
        dir_a.mkdir()
        dir_b.mkdir()
        (dir_a / "t.txt").write_text("shared text\n")
        (dir_b / "t.txt").write_text("shared text\n")

        env = _FakeEnv(start_cwd=str(dir_a))
        ops = ShellFileOperations(env, cwd=str(dir_a))

        # Emulate user cd'ing into the worktree
        env.execute(f"cd {dir_b}")
        assert env.cwd == str(dir_b)

        # Patch with a RELATIVE path — must target the worktree, not main
        result = ops.patch_replace("t.txt", "shared text\n", "PATCHED\n")
        assert result.success is True

        assert (dir_b / "t.txt").read_text() == "PATCHED\n", (
            "patch must land in the live-cwd dir (worktree)"
        )
        assert (dir_a / "t.txt").read_text() == "shared text\n", (
            "patch must NOT land in the init-time dir (main)"
        )

    def test_explicit_cwd_arg_still_wins(self, tmp_path):
        """An explicit cwd= arg to _exec must override both env.cwd and self.cwd."""
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_c = tmp_path / "c"
        for d in (dir_a, dir_b, dir_c):
            d.mkdir()
        (dir_a / "target.txt").write_text("from-a\n")
        (dir_b / "target.txt").write_text("from-b\n")
        (dir_c / "target.txt").write_text("from-c\n")

        env = _FakeEnv(start_cwd=str(dir_a))
        ops = ShellFileOperations(env, cwd=str(dir_a))
        env.execute(f"cd {dir_b}")

        # Explicit cwd=dir_c should win over env.cwd (dir_b) and self.cwd (dir_a)
        result = ops._exec("cat target.txt", cwd=str(dir_c))
        assert "from-c" in result.stdout

    def test_env_without_cwd_attribute_falls_back_to_self_cwd(self, tmp_path):
        """Backends without a cwd attribute still work via init-time cwd."""
        dir_a = tmp_path / "fixed"
        dir_a.mkdir()
        (dir_a / "target.txt").write_text("fixed-content\n")

        class _NoCwdEnv:
            def execute(self, command, cwd=None, **kwargs):
                import subprocess
                proc = subprocess.run(["bash", "-c", command], cwd=cwd,
                                      capture_output=True, text=True)
                return {"output": proc.stdout, "returncode": proc.returncode}

        env = _NoCwdEnv()
        ops = ShellFileOperations(env, cwd=str(dir_a))
        result = ops._exec("cat target.txt")
        assert result.exit_code == 0
        assert "fixed-content" in result.stdout

    def test_patch_returns_success_only_when_file_actually_written(self, tmp_path):
        """Safety rail: patch_replace success must reflect the real file state.

        This test doesn't trigger the bug directly (it would require manual
        corruption of the write), but it pins the invariant: when
        patch_replace returns success=True, the file on disk matches the
        intended content.  If a future write_file change ever regresses,
        this test catches it.
        """
        target = tmp_path / "file.txt"
        target.write_text("old content\n")

        env = _FakeEnv(start_cwd=str(tmp_path))
        ops = ShellFileOperations(env, cwd=str(tmp_path))

        result = ops.patch_replace(str(target), "old content\n", "new content\n")
        assert result.success is True
        assert result.error is None
        assert target.read_text() == "new content\n", (
            "patch_replace claimed success but file wasn't written correctly"
        )
