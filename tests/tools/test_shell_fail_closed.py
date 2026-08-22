"""Unimplemented ``terminal.shell: pwsh`` paths must fail closed.

Synchronous native-Windows foreground execution is implemented. Background
and PTY argv construction are not; those entry points must still raise instead
of silently running bash under a ``pwsh`` selection.

Host-independent tests assert the guard raises with a clear ``pwsh`` error;
``windows_only`` tests pin the exact exception type on the native Windows
host where a ``pwsh`` selection is a *valid* configuration reaching an
*unimplemented* execution path.
"""

import os
from unittest.mock import patch

import pytest

from tools.environments import shell_selection as ss


@pytest.fixture(autouse=True)
def _reset_active_shell_identity():
    ss._clear_active_shell_name_cache()
    try:
        yield
    finally:
        ss._clear_active_shell_name_cache()


class TestLocalForegroundSelection:
    def test_local_environment_default_shell_still_constructs(self, tmp_path):
        """No TERMINAL_SHELL → bash → construction unchanged."""
        from tools.environments.local import LocalEnvironment

        env = LocalEnvironment(cwd=str(tmp_path))
        try:
            assert env is not None
        finally:
            env.cleanup()


class TestBackgroundPtyFailsClosed:
    @pytest.mark.windows_only
    def test_frozen_pwsh_environment_rejects_after_process_env_drifts_to_bash(self, tmp_path):
        from tools.environments.local import LocalEnvironment
        from tools.process_registry import ProcessRegistry

        registry = ProcessRegistry()
        with patch.dict(
            os.environ,
            {"TERMINAL_SHELL": "pwsh", "TERMINAL_ENV": "local"},
        ):
            env = LocalEnvironment(cwd=str(tmp_path))
            os.environ["TERMINAL_SHELL"] = "bash"
            try:
                with pytest.raises(ss.ShellExecutionNotImplementedError):
                    registry.spawn_local(
                        "Write-Output drifted",
                        cwd=str(tmp_path),
                        shell_name=env.shell_name,
                    )
            finally:
                env.cleanup()
        assert len(registry._running) == 0

    @pytest.mark.parametrize("use_pty", [False, True])
    def test_spawn_local_fails_closed_for_pwsh(self, tmp_path, use_pty):
        from tools.process_registry import ProcessRegistry

        registry = ProcessRegistry()
        with patch.dict(
            os.environ,
            {"TERMINAL_SHELL": "pwsh", "TERMINAL_ENV": "local"},
        ):
            with pytest.raises(
                (ss.ShellSelectionError, ss.ShellExecutionNotImplementedError)
            ) as exc_info:
                registry.spawn_local("echo hi", cwd=str(tmp_path), use_pty=use_pty)
        assert "pwsh" in str(exc_info.value)

    @pytest.mark.windows_only
    @pytest.mark.parametrize("use_pty", [False, True])
    def test_windows_spawn_local_pwsh_never_registers_or_spawns(self, tmp_path, use_pty):
        from tools.process_registry import ProcessRegistry

        registry = ProcessRegistry()
        with patch.dict(
            os.environ,
            {"TERMINAL_SHELL": "pwsh", "TERMINAL_ENV": "local"},
        ):
            with pytest.raises(ss.ShellExecutionNotImplementedError):
                registry.spawn_local("echo hi", cwd=str(tmp_path), use_pty=use_pty)
        # The guard must fire before any session is created or process spawned.
        assert len(registry._running) == 0

    def test_spawn_local_default_shell_still_spawns(self, tmp_path):
        """No TERMINAL_SHELL → bash → existing behavior unchanged."""
        from tools.process_registry import ProcessRegistry

        registry = ProcessRegistry()
        session = registry.spawn_local("echo hi", cwd=str(tmp_path))
        try:
            assert session.pid > 0
        finally:
            registry.kill_process(session.id)
