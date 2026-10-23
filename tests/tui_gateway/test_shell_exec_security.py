"""Security-boundary regressions for the TUI ``shell.exec`` RPC."""

from __future__ import annotations

import subprocess

from agent import redact
from tui_gateway import server


def test_shell_exec_scrubs_child_env_and_force_redacts_rpc_output(monkeypatch):
    secret = "sk-shell-exec-secret-1234567890"
    captured = {}

    def fake_run(command, **kwargs):
        captured.update(kwargs)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=f"visible\n{secret}\n",
            stderr=f"diagnostic {secret}",
        )

    monkeypatch.setenv("OPENROUTER_API_KEY", secret)
    monkeypatch.setattr(redact, "_REDACT_ENABLED", False)
    monkeypatch.setattr(server.subprocess, "run", fake_run)

    response = server.handle_request(
        {"id": "security", "method": "shell.exec", "params": {"command": "echo safe"}}
    )

    assert "OPENROUTER_API_KEY" not in captured["env"]
    assert secret not in response["result"]["stdout"]
    assert secret not in response["result"]["stderr"]
    assert "visible" in response["result"]["stdout"]


def test_shell_exec_survives_multiplexed_passthrough_registration(monkeypatch):
    """A skill-registered passthrough name used to abort every ``shell.exec`` with
    ``UnscopedSecretError`` on a multiplexed backend: the handler built the child env with
    no profile secret scope bound, so resolving the passthrough failed closed (#134343)."""
    captured = {}

    def fake_run(command, **kwargs):
        captured.update(kwargs)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    from agent.secret_scope import reset_multiplex_context, set_multiplex_context
    from tui_gateway import launch_profile_policy
    from tools import env_passthrough as ep

    monkeypatch.setattr(server.subprocess, "run", fake_run)
    monkeypatch.setenv("OP_SERVICE_ACCOUNT_TOKEN", "dummy-not-real")
    # Freeze the launch env ourselves: the module keeps only the FIRST capture, which an
    # earlier test in the session may already have taken without our variable.
    monkeypatch.setattr(
        launch_profile_policy,
        "_snapshot",
        {"OP_SERVICE_ACCOUNT_TOKEN": "dummy-not-real"},
    )
    token = set_multiplex_context(True)
    ep._get_allowed().add("OP_SERVICE_ACCOUNT_TOKEN")
    try:
        response = server.handle_request({
            "id": "mux",
            "method": "shell.exec",
            "params": {"command": "true"},
        })
    finally:
        ep._get_allowed().discard("OP_SERVICE_ACCOUNT_TOKEN")
        reset_multiplex_context(token)

    assert response["result"]["code"] == 0
    # Resolved through the bound launch scope (the frozen snapshot's value), not by raising.
    assert captured["env"]["OP_SERVICE_ACCOUNT_TOKEN"] == "dummy-not-real"


def test_shell_exec_keeps_single_profile_passthrough_value(monkeypatch):
    """Single-profile passthrough keeps its overlay semantics: the value in the process env
    still reaches the child (the profile scope is an overlay there, not a blindfold)."""
    captured = {}

    def fake_run(command, **kwargs):
        captured.update(kwargs)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    from tools import env_passthrough as ep

    monkeypatch.setattr(server.subprocess, "run", fake_run)
    monkeypatch.setenv("HERMES_SHELL_EXEC_PASSTHRU", "single-profile-value")
    ep._get_allowed().add("HERMES_SHELL_EXEC_PASSTHRU")
    try:
        response = server.handle_request({
            "id": "single",
            "method": "shell.exec",
            "params": {"command": "true"},
        })
    finally:
        ep._get_allowed().discard("HERMES_SHELL_EXEC_PASSTHRU")

    assert response["result"]["code"] == 0
    assert captured["env"]["HERMES_SHELL_EXEC_PASSTHRU"] == "single-profile-value"
