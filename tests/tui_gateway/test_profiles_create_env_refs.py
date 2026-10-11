"""A bot created from the launch profile inherits templates without persisting secrets.

``${VAR}`` refs stay in the launch configuration.  The new profile stores only its delta, so
neither refs nor their resolved values are copied into its ``config.yaml``.
"""

from __future__ import annotations

from pathlib import Path

from tui_gateway import server
from hermes_cli.config import load_config, read_user_config_raw

_SECRETS = {"FAKE_TTS_KEY": "tts-FAKE-111", "FAKE_STT_KEY": "stt-FAKE-222", "FAKE_MCP_TOKEN": "mcp-FAKE-333"}


def _ok(method: str, params: dict) -> dict:
    resp = server._methods[method](1, params)
    assert "error" not in resp, resp.get("error")
    return resp["result"]


def test_new_bot_config_keeps_launch_env_refs_not_their_values(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    for name, value in _SECRETS.items():
        monkeypatch.setenv(name, value)
    (home / "config.yaml").write_text(
        "tts:\n  provider: elevenlabs\n  elevenlabs:\n    api_key: ${FAKE_TTS_KEY}\n"
        "stt:\n  provider: openai\n  openai:\n    api_key: ${FAKE_STT_KEY}\n"
        "mcp_servers:\n  tracker:\n    command: tracker-mcp\n    env:\n      TRACKER_TOKEN: ${FAKE_MCP_TOKEN}\n",
        encoding="utf-8")

    # The Desktop "New bot" flow: create, then enable a launch-catalog MCP server in the editor.
    _ok("profiles.create", {"name": "scout", "no_alias": True, "no_skills": True})
    _ok("profiles.configure", {"name": "scout", "enabled_mcp_servers": ["tracker"]})

    profile = home / "profiles" / "scout"
    written = (profile / "config.yaml").read_text(encoding="utf-8")
    with server._hermes_home_scope(profile):
        cfg = load_config()
        child_raw = read_user_config_raw() or {}
    assert [value for value in _SECRETS.values() if value in written] == []
    assert [f"${{{name}}}" for name in _SECRETS if f"${{{name}}}" in written] == []
    assert child_raw.get("mcp_servers", {}).get("tracker") == {"enabled": True}
    assert cfg["tts"]["elevenlabs"]["api_key"] == "${FAKE_TTS_KEY}"
    assert cfg["stt"]["openai"]["api_key"] == "${FAKE_STT_KEY}"
    assert cfg["mcp_servers"]["tracker"]["env"]["TRACKER_TOKEN"] == "${FAKE_MCP_TOKEN}"
