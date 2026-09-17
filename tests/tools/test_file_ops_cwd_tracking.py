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

    def test_exec_ignores_poisoned_host_cwd_on_container_backend(self, tmp_path):
        """A workspace override can set env.cwd to a raw host path (#113894).

        Desktop/TUI/gateway surfaces register a workspace cwd via
        ``register_task_env_overrides``/``record_session_cwd``, which (unlike
        environment CREATION) does not re-run the container-cwd sanity check.
        On a container backend (e.g. docker), a host-style override like
        ``C:\\Users\\rashi\\...`` lands directly on the live env. Every
        _exec() previously trusted that live cwd unconditionally, so the
        wrapper's own ``cd -- <host path>`` failed and the raw shell error
        leaked into write_file's response. _exec() must detect the
        unusable container cwd and fall back to the last known-good cwd
        instead of handing it to the backend.
        """
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "target.txt").write_text("workspace-content\n", encoding="utf-8")

        env = _FakeEnv(start_cwd=str(workspace))
        ops = ShellFileOperations(env, cwd=str(workspace), env_type="docker")

        # Simulate the poisoning: a workspace override lands the raw host
        # path directly on the live env, bypassing the creation-time guard.
        env.cwd = r"C:\Users\rashi\OneDrive\Documents\ai_workspace"

        result = ops._exec("cat target.txt")

        assert result.exit_code == 0, (
            f"exec failed with poisoned cwd instead of falling back: {result.stdout!r}"
        )
        assert "workspace-content" in result.stdout
        assert env.calls[-1]["cwd"] != r"C:\Users\rashi\OneDrive\Documents\ai_workspace", (
            "the unusable host cwd was handed to the backend instead of being discarded"
        )

    def test_construction_time_poisoned_cwd_falls_back_to_validated_default(
        self, tmp_path, monkeypatch
    ):
        """The production call site (``tools/file_tools.py`` ``_get_file_ops``)
        never passes a ``cwd=`` kwarg -- ``self.cwd`` is derived solely from
        ``env.cwd``. A workspace override (``register_task_env_overrides``)
        can poison ``env.cwd`` on an ALREADY-ACTIVE environment before the
        task's first file-tool call ever builds this wrapper (e.g. the env
        was created earlier via the ``terminal`` tool, an override lands on
        it, and only then does a file tool run for the first time). Without
        re-validating at construction time, that poisoned value gets baked
        into ``self.cwd`` permanently, so _exec()'s fallback would be exactly
        as broken as the live cwd it discards.
        """
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "target.txt").write_text("workspace-content\n", encoding="utf-8")

        env = _FakeEnv(start_cwd=str(workspace))
        # Poison BEFORE construction -- mirrors an override landing on an
        # already-active env ahead of the task's first _get_file_ops() call.
        env.cwd = r"C:\Users\rashi\OneDrive\Documents\ai_workspace"

        monkeypatch.setattr(
            "tools.terminal_tool._get_env_config",
            lambda: {"cwd": str(workspace)},
        )

        # Matches tools/file_tools.py:367 verbatim -- no cwd= kwarg.
        ops = ShellFileOperations(env, env_type="docker")

        assert ops.cwd == str(workspace), (
            f"poisoned env.cwd was baked into self.cwd at construction: {ops.cwd!r}"
        )

        result = ops._exec("cat target.txt")
        assert result.exit_code == 0
        assert "workspace-content" in result.stdout

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
