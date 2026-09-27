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

import pytest

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


class _WrapperEnv:
    """Backend whose command wrapper does ``builtin cd -- <cwd> || exit 126`` (the
    real terminal backends' shape), so a bad cwd kills every command before it runs."""

    def __init__(self, cwd, env_type=None):
        self.cwd = cwd
        if env_type:
            self.env_type = env_type

    def execute(self, command, cwd=None, **kwargs):
        import shlex
        import subprocess
        script = f"builtin cd -- {shlex.quote(cwd)} || exit 126\n{command}"
        proc = subprocess.run(["bash", "-c", script], capture_output=True, text=True, encoding="utf-8",
                              input=kwargs.get("stdin_data"))
        return {"output": proc.stdout + proc.stderr, "returncode": proc.returncode}


class TestShellFileOpsCwdTracking:
    """_exec() must use live env.cwd, not the init-time cached cwd."""

    def test_exec_follows_env_cwd_after_cd(self, tmp_path):
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        (dir_a / "target.txt").write_text("content-a\n", encoding="utf-8")
        (dir_b / "target.txt").write_text("content-b\n", encoding="utf-8")

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


    def test_env_without_cwd_attribute_falls_back_to_self_cwd(self, tmp_path):
        """Backends without a cwd attribute still work via init-time cwd."""
        dir_a = tmp_path / "fixed"
        dir_a.mkdir()
        (dir_a / "target.txt").write_text("fixed-content\n", encoding="utf-8")

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

    @pytest.mark.platforms("linux")
    def test_patch_returns_success_only_when_file_actually_written(self, tmp_path):
        """Safety rail: patch_replace success must reflect the real file state.

        This test doesn't trigger the bug directly (it would require manual
        corruption of the write), but it pins the invariant: when
        patch_replace returns success=True, the file on disk matches the
        intended content.  If a future write_file change ever regresses,
        this test catches it.
        """
        target = tmp_path / "file.txt"
        target.write_text("old content\n", encoding="utf-8")

        env = _FakeEnv(start_cwd=str(tmp_path))
        ops = ShellFileOperations(env, cwd=str(tmp_path))

        result = ops.patch_replace(str(target), "old content\n", "new content\n")
        assert result.success is True
        assert result.error is None
        assert target.read_text(encoding="utf-8") == "new content\n", (
            "patch_replace claimed success but file wasn't written correctly"
        )

    def test_wrapper_cd_failure_names_the_invalid_working_directory(self, tmp_path):
        """When the backend's own ``builtin cd -- <cwd> || exit 126`` fails (a
        host ``terminal.cwd`` inside a container, #113894) the surfaced error
        must name the working directory / ``terminal.cwd`` problem, not just
        the raw ``cd:`` line that reads like a fault at the requested path."""
        host_cwd = r"C:\Users\rashi\OneDrive\Documents\ai_workspace"
        ops = ShellFileOperations(_WrapperEnv(host_cwd))
        result = ops.write_file(str(tmp_path / "probe.py"), "print('hi')\n")

        assert result.error is not None
        assert "terminal.cwd" in result.error and host_cwd in result.error
        assert "No such file or directory" in result.error  # the shell's own line is kept
        assert not (tmp_path / "probe.py").exists()

    @pytest.mark.parametrize("op", ["read_file", "read_file_raw", "read_file_bytes", "patch", "search"])
    def test_wrapper_cd_failure_is_reported_as_such_on_every_read_path(self, tmp_path, op):
        """#98723: a wrapper-level failure must be reported as what it is on the
        paths that do NOT embed stdout (stat-probe reads, patch pre-image, search),
        never as 'environment unavailable' / 'file not found' / 'rg or find missing';
        and no ``_has_command`` verdict may be cached from a probe that never ran."""
        target = tmp_path / "t.txt"
        target.write_text("hello\n", encoding="utf-8")
        ops = ShellFileOperations(_WrapperEnv("/Users/nobody/ws"))
        result = {
            "read_file": lambda: ops.read_file(str(target)),
            "read_file_raw": lambda: ops.read_file_raw(str(target)),
            "read_file_bytes": lambda: ops.read_file_bytes(str(target)),
            "patch": lambda: ops.patch_replace(str(target), "hello", "bye"),
            "search": lambda: ops.search("t*", str(tmp_path), target="files"),
        }[op]()
        assert "/Users/nobody/ws" in result.error and "No such file or directory" in result.error
        assert "unavailable" not in result.error and "requires" not in result.error
        assert ops._has_command("find") is False
        assert ops._command_cache == {}  # a later valid cwd must re-probe

    def test_container_hint_only_for_container_backends(self, tmp_path):
        """The in-container path hint misdirects on local/ssh backends (the cwd may
        be an explicit arg, a session ``cd`` or a deleted directory)."""
        local = ShellFileOperations(_WrapperEnv("/nope/local", env_type="local")).read_file_raw("/x")
        docker = ShellFileOperations(_WrapperEnv("/nope/docker", env_type="docker")).read_file_raw("/x")
        assert "/workspace" not in local.error
        assert "/workspace" in docker.error
