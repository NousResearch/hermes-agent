"""delegate_task children must never trigger the interactive sudo prompt.

Subagents run on worker threads of the parent process and inherit
process-wide interactivity signals (``HERMES_INTERACTIVE=1`` set by the CLI
at startup). Before the fix, ``_transform_sudo_command`` passed the
interactive gate inside a child, found no thread-local sudo callback, and
fell through to the raw ``/dev/tty`` password prompt — printed mid-TUI from
a background thread and blocking the child for the full 45s timeout.

The delegated-child ContextVar (``agent.delegation_context``) is the
authoritative "this execution context has no user" signal: it is set around
every child run and propagates through ``contextvars.copy_context`` onto the
executor thread.
"""

import contextvars
import os
import threading

import pytest

from agent.delegation_context import delegated_child_context
from tools import terminal_tool as tt


@pytest.fixture(autouse=True)
def _clean_sudo_state(monkeypatch):
    """Isolate sudo-related process/thread state per test."""
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    # Host sudoers NOPASSWD must not short-circuit the path under test.
    monkeypatch.setattr(tt, "_sudo_nopasswd_works", lambda: False)
    tt._reset_cached_sudo_passwords()
    tt.set_sudo_password_callback(None)
    yield
    tt.set_sudo_password_callback(None)
    tt._reset_cached_sudo_passwords()


def _transform_in_child(command: str):
    """Run _transform_sudo_command the way delegate_tool runs children:

    inside delegated_child_context(), through contextvars.copy_context(),
    on a separate worker thread.
    """
    result = {}

    def _parent_side():
        with delegated_child_context("child-session"):
            ctx = contextvars.copy_context()

            def _worker():
                result["value"] = ctx.run(tt._transform_sudo_command, command)

            t = threading.Thread(target=_worker)
            t.start()
            t.join(timeout=10)
            assert not t.is_alive(), "child transform blocked (prompt fired?)"

    _parent_side()
    return result["value"]


class TestDelegatedChildNeverPrompts:
    def test_interactive_env_does_not_prompt_in_child(self, monkeypatch):
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        calls = []
        monkeypatch.setattr(
            tt,
            "_prompt_for_sudo_password",
            lambda timeout_seconds=45: calls.append(1) or "hunter2",
        )

        transformed, sudo_stdin = _transform_in_child("sudo apt-get update")

        assert calls == [], "child must not reach the interactive sudo prompt"
        assert sudo_stdin is None
        assert transformed == "sudo apt-get update"

    def test_stale_thread_callback_does_not_prompt_in_child(self, monkeypatch):
        # Precondition sanity: the interactive gate WOULD fire outside a child.
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        calls = []

        def _run_child_with_stale_callback():
            # Simulate a recycled worker thread that still holds a parent
            # CLI callback in its thread-local slot.
            tt.set_sudo_password_callback(lambda: calls.append(1) or "pw")
            try:
                with delegated_child_context("child-session"):
                    return tt._transform_sudo_command("sudo systemctl restart foo")
            finally:
                tt.set_sudo_password_callback(None)

        transformed, sudo_stdin = _run_child_with_stale_callback()
        assert calls == []
        assert sudo_stdin is None
        assert transformed == "sudo systemctl restart foo"

    def test_parent_context_still_prompts(self, monkeypatch):
        """The fix must not break interactive prompting outside children."""
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        monkeypatch.setattr(
            tt, "_prompt_for_sudo_password", lambda timeout_seconds=45: "hunter2"
        )

        transformed, sudo_stdin = tt._transform_sudo_command("sudo whoami")

        assert sudo_stdin == "hunter2\n"
        assert "sudo -S -p ''" in transformed

    def test_configured_password_still_works_in_child(self, monkeypatch):
        monkeypatch.setenv("SUDO_PASSWORD", "s3cret")
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")

        transformed, sudo_stdin = _transform_in_child("sudo whoami")

        assert sudo_stdin == "s3cret\n"
        assert "sudo -S -p ''" in transformed


class TestDelegatedChildFailureMessaging:
    def test_child_gets_headless_sudo_tip(self):
        with delegated_child_context("child-session"):
            out = tt._handle_sudo_failure(
                "sudo: a password is required", env_type="local"
            )
        assert "Subagents cannot prompt" in out
        assert "SUDO_PASSWORD" in out

    def test_parent_output_unchanged(self, monkeypatch):
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        out = tt._handle_sudo_failure("sudo: a password is required", env_type="local")
        assert out == "sudo: a password is required"

    def test_gateway_tip_preserved(self, monkeypatch):
        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
        out = tt._handle_sudo_failure("sudo: a password is required", env_type="local")
        assert "To enable sudo over messaging" in out
