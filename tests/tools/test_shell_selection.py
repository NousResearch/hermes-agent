"""Tests for the single ``terminal.shell`` selection semantic source.

``tools.environments.shell_selection`` is the unique place where the
configured shell dialect is validated and resolved.  Prompt building, the
local foreground environment, and the background/PTY process registry all
consume it, so the value policy and the host/backend constraints must be
pinned here:

- allowed values: ``bash`` (default) and ``pwsh``;
- normalization policy (fixed in this slice): leading/trailing whitespace
  is trimmed and case is ignored, then the value must match one of the two
  allowed words exactly;
- ``pwsh`` is only valid for the native Windows local backend (win32 host,
  not WSL, ``TERMINAL_ENV`` local);
- a valid ``pwsh`` selection reaching an execution path that this slice has
  not implemented must fail closed (never run bash, never silently fall
  back).
"""

import pytest
from unittest.mock import patch

from tools.environments import shell_selection as ss


class TestConstants:
    def test_default_and_env_var_contract(self):
        assert ss.DEFAULT_SHELL == "bash"
        assert ss.SHELL_BASH == "bash"
        assert ss.SHELL_PWSH == "pwsh"
        assert ss.ALLOWED_SHELLS == {"bash", "pwsh"}
        # Internal projection name must match the config bridge key.
        assert ss.SHELL_ENV_VAR == "TERMINAL_SHELL"


class TestActiveShellIdentity:
    @pytest.mark.windows_only
    def test_active_identity_is_frozen_for_process_lifetime(self):
        ss._clear_active_shell_name_cache()
        try:
            with patch.dict(
                ss.os.environ,
                {"TERMINAL_SHELL": "pwsh", "TERMINAL_ENV": "local"},
            ):
                assert ss.get_active_shell_name() == "pwsh"
                ss.os.environ["TERMINAL_SHELL"] = "bash"
                assert ss.get_active_shell_name() == "pwsh"
        finally:
            ss._clear_active_shell_name_cache()


class TestResolveShellNameValuePolicy:
    def test_defaults_to_bash_when_env_unset(self):
        assert ss.resolve_shell_name(env={}) == "bash"

    def test_defaults_to_bash_when_only_backend_present(self):
        assert ss.resolve_shell_name(env={"TERMINAL_ENV": "local"}) == "bash"

    def test_explicit_bash_is_accepted(self):
        assert ss.resolve_shell_name(env={"TERMINAL_SHELL": "bash"}) == "bash"

    def test_empty_value_is_treated_as_unset(self):
        assert ss.resolve_shell_name(env={"TERMINAL_SHELL": ""}) == "bash"

    def test_whitespace_is_trimmed(self):
        assert ss.resolve_shell_name(env={"TERMINAL_SHELL": "  bash  "}) == "bash"
        assert ss.resolve_shell_name(env={"TERMINAL_SHELL": "\t pwsh \n"}) == "pwsh"

    def test_case_is_ignored(self):
        assert ss.resolve_shell_name(env={"TERMINAL_SHELL": "Bash"}) == "bash"
        assert ss.resolve_shell_name(env={"TERMINAL_SHELL": "PWSH"}) == "pwsh"

    @pytest.mark.parametrize(
        "bad",
        [
            "fish",
            "zsh",
            "sh",
            "powershell",
            "PowerShell",
            "cmd",
            "cmd.exe",
            "pwsh.exe",
            "bash.exe",
            "pw sh",
            "ba sh",
        ],
    )
    def test_unsupported_values_are_rejected_with_clear_error(self, bad):
        with pytest.raises(ss.ShellSelectionError) as exc_info:
            ss.resolve_shell_name(env={"TERMINAL_SHELL": bad})
        message = str(exc_info.value)
        assert "terminal.shell" in message
        assert repr(bad) in message
        assert "'bash'" in message and "'pwsh'" in message


class TestResolveShellNamePwshConstraints:
    def test_pwsh_accepted_on_native_windows_local(self):
        assert (
            ss.resolve_shell_name(
                env={"TERMINAL_SHELL": "pwsh", "TERMINAL_ENV": "local"},
                platform_name="win32",
                wsl=False,
            )
            == "pwsh"
        )

    def test_pwsh_accepted_when_backend_defaults_to_local(self):
        assert (
            ss.resolve_shell_name(
                env={"TERMINAL_SHELL": "pwsh"},
                platform_name="win32",
                wsl=False,
            )
            == "pwsh"
        )

    def test_pwsh_backend_value_is_normalized(self):
        assert (
            ss.resolve_shell_name(
                env={"TERMINAL_SHELL": "pwsh", "TERMINAL_ENV": " LOCAL "},
                platform_name="win32",
                wsl=False,
            )
            == "pwsh"
        )

    def test_pwsh_requires_native_windows_host(self):
        with pytest.raises(ss.ShellSelectionError) as exc_info:
            ss.resolve_shell_name(
                env={"TERMINAL_SHELL": "pwsh"},
                platform_name="linux",
                wsl=False,
            )
        assert "Windows" in str(exc_info.value)

    def test_pwsh_rejected_under_wsl(self):
        with pytest.raises(ss.ShellSelectionError) as exc_info:
            ss.resolve_shell_name(
                env={"TERMINAL_SHELL": "pwsh"},
                platform_name="win32",
                wsl=True,
            )
        assert "WSL" in str(exc_info.value)

    @pytest.mark.parametrize(
        "backend",
        ["docker", "ssh", "modal", "singularity", "daytona", "vercel_sandbox", "managed_modal"],
    )
    def test_pwsh_rejected_for_remote_backends(self, backend):
        with pytest.raises(ss.ShellSelectionError) as exc_info:
            ss.resolve_shell_name(
                env={"TERMINAL_SHELL": "pwsh", "TERMINAL_ENV": backend},
                platform_name="win32",
                wsl=False,
            )
        assert "local" in str(exc_info.value)

    def test_pwsh_rejected_for_explicit_remote_backend_param(self):
        with pytest.raises(ss.ShellSelectionError) as exc_info:
            ss.resolve_shell_name(
                env={"TERMINAL_SHELL": "pwsh"},
                platform_name="win32",
                wsl=False,
                backend="docker",
            )
        assert "local" in str(exc_info.value)

    def test_bash_is_accepted_on_any_host_or_backend(self):
        for platform_name in ("win32", "linux", "darwin"):
            assert (
                ss.resolve_shell_name(
                    env={"TERMINAL_SHELL": "bash"},
                    platform_name=platform_name,
                    wsl=True,
                    backend="ssh",
                )
                == "bash"
            )


class TestFailClosedExecutionGuard:
    def test_bash_passes_the_guard(self):
        assert ss.reject_unimplemented_shell(env={"TERMINAL_SHELL": "bash"}) is None

    def test_pwsh_on_native_windows_local_fails_closed_as_not_implemented(self):
        with pytest.raises(ss.ShellExecutionNotImplementedError) as exc_info:
            ss.reject_unimplemented_shell(
                env={"TERMINAL_SHELL": "pwsh"},
                platform_name="win32",
                wsl=False,
                backend="local",
            )
        message = str(exc_info.value)
        assert "not implemented" in message
        assert "pwsh" in message

    def test_pwsh_off_windows_is_a_validation_error_before_the_execution_guard(self):
        # An invalid combination is a configuration error, not a
        # not-implemented error — the distinction keeps diagnostics honest.
        with pytest.raises(ss.ShellSelectionError):
            ss.reject_unimplemented_shell(
                env={"TERMINAL_SHELL": "pwsh"},
                platform_name="linux",
                wsl=False,
                backend="local",
            )

    def test_guard_never_falls_back_to_bash_for_pwsh(self):
        # The fail-closed contract: selecting pwsh must raise (either the
        # invalid-combination error or the not-implemented error), never
        # return a bash executable or silently proceed.
        with pytest.raises((ss.ShellSelectionError, ss.ShellExecutionNotImplementedError)):
            ss.reject_unimplemented_shell(
                env={"TERMINAL_SHELL": "pwsh"},
                platform_name="win32",
                wsl=False,
                backend="local",
            )
