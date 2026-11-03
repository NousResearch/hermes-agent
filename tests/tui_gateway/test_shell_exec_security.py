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
