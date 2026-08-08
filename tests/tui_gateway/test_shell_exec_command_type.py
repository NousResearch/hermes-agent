"""shell.exec must reject non-string ``command`` before approval checks.

``params.get("command", "")`` returns None when the key is present with JSON
null (caught by the empty check). A list/int is truthy, reaches
``detect_hardline_command``, and raises TypeError — which the ImportError-only
catch does not swallow.
"""

from __future__ import annotations

from tui_gateway import server


def test_shell_exec_rejects_null_command():
    resp = server.handle_request(
        {
            "id": "1",
            "method": "shell.exec",
            "params": {"command": None},
        }
    )
    assert resp["error"]["code"] == 4004


def test_shell_exec_rejects_list_command():
    resp = server.handle_request(
        {
            "id": "2",
            "method": "shell.exec",
            "params": {"command": ["echo", "hi"]},
        }
    )
    assert resp["error"]["code"] == 4004
    assert "string" in resp["error"]["message"].lower()


def test_shell_exec_rejects_empty_string():
    resp = server.handle_request(
        {
            "id": "3",
            "method": "shell.exec",
            "params": {"command": "   "},
        }
    )
    assert resp["error"]["code"] == 4004
    assert "empty" in resp["error"]["message"].lower()


def test_cli_exec_tolerates_non_int_timeout(monkeypatch):
    """Malformed timeout must not raise before the subprocess call."""
    calls = {}

    def fake_run(*args, **kwargs):
        calls["timeout"] = kwargs.get("timeout")

        class R:
            returncode = 0
            stdout = ""
            stderr = ""

        return R()

    monkeypatch.setattr(server.subprocess, "run", fake_run)
    monkeypatch.setattr(server, "_cli_exec_blocked", lambda _argv: None)

    resp = server.handle_request(
        {
            "id": "4",
            "method": "cli.exec",
            "params": {"argv": ["version"], "timeout": "nope"},
        }
    )
    assert "error" not in resp, resp
    assert calls["timeout"] == 240
