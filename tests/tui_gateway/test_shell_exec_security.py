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


def test_shell_exec_raw_stdout_preserves_structured_token(monkeypatch):
    """raw_stdout=True opts out of redaction so plugin probes parsing structured
    output receive verbatim bytes (#149778)."""
    token = "1341a55f03ce51f23d46420e9bcd6226"
    payload = '{"snapshot_token":"' + token + '","snapshots":[]}'

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout=payload, stderr="")

    monkeypatch.setattr(server.subprocess, "run", fake_run)

    response = server.handle_request(
        {"id": "raw", "method": "shell.exec", "params": {"command": "probe.py", "raw_stdout": True}}
    )

    assert response["result"]["stdout"] == payload
    assert token in response["result"]["stdout"]


def test_shell_exec_default_redacts_structured_token(monkeypatch):
    """Without raw_stdout the default force-redaction still masks random-looking
    tokens, preserving the security boundary for user-facing !cmd output (#149778)."""
    token = "1341a55f03ce51f23d46420e9bcd6226"
    payload = '{"snapshot_token":"' + token + '","snapshots":[]}'

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout=payload, stderr="")

    monkeypatch.setattr(server.subprocess, "run", fake_run)

    response = server.handle_request(
        {"id": "default", "method": "shell.exec", "params": {"command": "probe.py"}}
    )

    assert token not in response["result"]["stdout"]
    assert "snapshot_token" in response["result"]["stdout"]


def test_shell_exec_string_raw_stdout_value_still_redacts(monkeypatch):
    """A non-bool ``raw_stdout`` (e.g. the string ``"false"``) must NOT opt out of
    redaction — only the strict boolean ``True`` activates the raw branch, so a
    client that sends ``raw_stdout: "false"`` still gets force-redacted output."""
    token = "1341a55f03ce51f23d46420e9bcd6226"
    payload = '{"snapshot_token":"' + token + '","snapshots":[]}'

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout=payload, stderr="")

    monkeypatch.setattr(server.subprocess, "run", fake_run)

    response = server.handle_request(
        {"id": "str-false", "method": "shell.exec", "params": {"command": "probe.py", "raw_stdout": "false"}}
    )

    assert token not in response["result"]["stdout"]
    assert "snapshot_token" in response["result"]["stdout"]
