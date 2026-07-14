"""terminal_tool() must refuse commands referencing a denied secret path
before doing any environment setup — the guard sits ahead of
``_get_env_config()`` so it can't be skipped by an env-selection quirk, and
it is NOT gated by ``force=True`` (force only pre-confirms the separate
dangerous-command check; it isn't a secret-access override).
"""

from __future__ import annotations

import json

import tools.terminal_tool as terminal_tool


def test_env_file_read_is_blocked_before_env_setup(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("TELEGRAM_BOT_TOKEN=live-secret-value", encoding="utf-8")

    def _fail_if_called():
        raise AssertionError("_get_env_config must not run for a blocked command")

    monkeypatch.setattr(terminal_tool, "_get_env_config", _fail_if_called)

    out = json.loads(
        terminal_tool.terminal_tool(command=f"cat {env_file}", workdir=str(tmp_path))
    )
    assert out["status"] == "error"
    assert "secret-bearing environment file" in out["error"]
    assert "live-secret-value" not in json.dumps(out)


def test_force_does_not_bypass_secret_guard(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("TELEGRAM_BOT_TOKEN=live-secret-value", encoding="utf-8")

    def _fail_if_called():
        raise AssertionError("_get_env_config must not run for a blocked command")

    monkeypatch.setattr(terminal_tool, "_get_env_config", _fail_if_called)

    out = json.loads(
        terminal_tool.terminal_tool(
            command=f"cat {env_file}", workdir=str(tmp_path), force=True
        )
    )
    assert out["status"] == "error"
    assert "secret-bearing environment file" in out["error"]


def test_ordinary_command_reaches_env_setup(tmp_path, monkeypatch):
    """Sanity check: an unrelated command still reaches _get_env_config —
    proves the guard isn't accidentally blocking everything. terminal_tool()
    wraps its body in a broad except-Exception, so a raise from the patched
    helper surfaces as an error-status JSON rather than propagating."""
    calls = []

    def _record():
        calls.append(1)
        raise RuntimeError("stop here, we only care that we got this far")

    monkeypatch.setattr(terminal_tool, "_get_env_config", _record)

    out = json.loads(
        terminal_tool.terminal_tool(command="ls -la /tmp", workdir=str(tmp_path))
    )
    assert calls == [1]
    assert out["status"] == "error"
