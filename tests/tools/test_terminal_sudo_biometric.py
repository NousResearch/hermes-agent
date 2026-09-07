"""HERMES_SUDO_BIOMETRIC: interactive-local sudo runs through a PTY so the platform
PAM stack (pam_fprintd on Linux) can drive fingerprint auth instead of a text
password box.

Opt-in via HERMES_SUDO_BIOMETRIC=1. Only applies on a local, interactive,
non-delegated, non-gateway session with util-linux ``script`` present. Headless
contexts (delegate_task children, cron, gateway/messaging) keep failing gracefully
exactly as before — no hang, no prompt.
"""

import contextvars
import os
import threading

import pytest

from agent.delegation_context import delegated_child_context
from tools import terminal_tool as tt
from tools import terminal_tool_sudo as tts


@pytest.fixture(autouse=True)
def _clean_sudo_state(monkeypatch):
    """Isolate sudo-related process/thread state per test."""
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_SUDO_BIOMETRIC", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    # Host sudoers NOPASSWD must not short-circuit the path under test.
    monkeypatch.setattr(tts, "_sudo_nopasswd_works", lambda: False)
    # Deterministic script availability regardless of the test host.
    monkeypatch.setattr(tts, "_script_available", lambda: True)
    tts._reset_cached_sudo_passwords()
    tt.set_sudo_password_callback(None)
    yield
    tt.set_sudo_password_callback(None)
    tts._reset_cached_sudo_passwords()


def _transform_in_child(command: str):
    """Run _transform_sudo_command the way delegate_tool runs children:
    inside delegated_child_context(), through contextvars.copy_context(),
    on a separate worker thread."""
    result = {}

    def _parent_side():
        with delegated_child_context("child-session"):
            ctx = contextvars.copy_context()

            def _worker():
                result["value"] = ctx.run(tts._transform_sudo_command, command)

            t = threading.Thread(target=_worker)
            t.start()
            t.join(timeout=10)
            assert not t.is_alive(), "child transform blocked (prompt fired?)"

    _parent_side()
    return result["value"]


class TestBiometricSudoPath:
    def test_enabled_local_interactive_wraps_in_pty(self, monkeypatch):
        monkeypatch.setenv("HERMES_SUDO_BIOMETRIC", "1")
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")

        transformed, sudo_stdin = tts._transform_sudo_command("sudo apt-get update")

        assert sudo_stdin is None
        assert transformed is not None
        assert transformed.startswith("script -qec ")
        assert "sudo apt-get update" in transformed
        assert "sudo -S" not in transformed

    def test_banner_present_so_user_knows_to_touch(self, monkeypatch):
        monkeypatch.setenv("HERMES_SUDO_BIOMETRIC", "1")
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")

        transformed, sudo_stdin = tts._transform_sudo_command("sudo whoami")

        assert sudo_stdin is None
        assert transformed is not None
        assert "TOUCH YOUR FINGER NOW" in transformed
        # Banner must run before the sudo command inside the PTY.
        assert transformed.index("TOUCH YOUR FINGER NOW") < transformed.index("sudo whoami")

    def test_disabled_keeps_password_prompt(self, monkeypatch):
        # HERMES_SUDO_BIOMETRIC unset -> existing password-box path unchanged.
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        monkeypatch.setattr(
            tts, "_prompt_for_sudo_password", lambda timeout_seconds=45: "hunter2"
        )

        transformed, sudo_stdin = tts._transform_sudo_command("sudo whoami")

        assert sudo_stdin == "hunter2\n"
        assert transformed is not None
        assert "sudo -S -p ''" in transformed

    def test_configured_password_wins_over_biometric(self, monkeypatch):
        monkeypatch.setenv("SUDO_PASSWORD", "s3cret")
        monkeypatch.setenv("HERMES_SUDO_BIOMETRIC", "1")
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")

        transformed, sudo_stdin = tts._transform_sudo_command("sudo whoami")

        assert sudo_stdin == "s3cret\n"
        assert transformed is not None
        assert "sudo -S -p ''" in transformed

    def test_cached_password_wins_over_biometric(self, monkeypatch):
        monkeypatch.setenv("HERMES_SUDO_BIOMETRIC", "1")
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        tts._set_cached_sudo_password("cachedpw")
        try:
            transformed, sudo_stdin = tts._transform_sudo_command("sudo whoami")
        finally:
            tts._reset_cached_sudo_passwords()

        assert sudo_stdin == "cachedpw\n"
        assert transformed is not None
        assert "sudo -S -p ''" in transformed

    def test_never_in_delegated_child(self, monkeypatch):
        monkeypatch.setenv("HERMES_SUDO_BIOMETRIC", "1")
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")

        transformed, sudo_stdin = _transform_in_child("sudo apt-get update")

        assert sudo_stdin is None
        # Unchanged: child has no user on the other side, fails gracefully.
        assert transformed == "sudo apt-get update"
        assert "script -qec" not in (transformed or "")

    def test_never_in_gateway_session(self, monkeypatch):
        monkeypatch.setenv("HERMES_SUDO_BIOMETRIC", "1")
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")

        transformed, sudo_stdin = tts._transform_sudo_command("sudo whoami")

        # Gateway: no interactive prompt either -> unchanged, no password.
        assert sudo_stdin is None
        assert transformed == "sudo whoami"
        assert "script -qec" not in (transformed or "")

    def test_script_missing_falls_back_to_password(self, monkeypatch):
        monkeypatch.setenv("HERMES_SUDO_BIOMETRIC", "1")
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        monkeypatch.setattr(tts, "_script_available", lambda: False)
        monkeypatch.setattr(
            tts, "_prompt_for_sudo_password", lambda timeout_seconds=45: "hunter2"
        )

        transformed, sudo_stdin = tts._transform_sudo_command("sudo whoami")

        assert sudo_stdin == "hunter2\n"
        assert transformed is not None
        assert "sudo -S -p ''" in transformed

    def test_windows_never_biometric(self, monkeypatch):
        monkeypatch.setenv("HERMES_SUDO_BIOMETRIC", "1")
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        # Force Windows detection; the biometric path must not fire on Windows
        # (no PAM/fprintd; MSYS 'script' would be a false positive).
        monkeypatch.setattr(tts.platform, "system", lambda: "Windows")

        assert tts._sudo_biometric_path_available() is False

        transformed, sudo_stdin = tts._transform_sudo_command("sudo whoami")

        assert "script -qec" not in (transformed or "")
        assert sudo_stdin is None

    def test_non_local_backend_never_biometric(self, monkeypatch):
        monkeypatch.setenv("HERMES_SUDO_BIOMETRIC", "1")
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")

        # _sudo_biometric_path_available re-imports _tenv from tools.terminal_tool at
        # call time, so patching the module attribute controls the branch.
        import tools.terminal_tool as tt_mod

        monkeypatch.setattr(tt_mod, "_tenv", lambda name, default=None: "ssh")
        assert tts._sudo_biometric_path_available() is False

        monkeypatch.setattr(tt_mod, "_tenv", lambda name, default=None: "local")
        assert tts._sudo_biometric_path_available() is True
