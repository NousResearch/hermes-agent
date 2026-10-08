"""A bot created from the launch profile gets its ``${VAR}`` refs, never their resolved values.

``${VAR}`` refs in ``config.yaml`` resolve against each profile's own ``.env``; copying the
launch profile's LOADED (expanded) config into a new profile wrote the launch secrets into the
new ``config.yaml`` as plaintext, where rotating the key in the bot's ``.env`` no longer applies.
"""

from __future__ import annotations

from pathlib import Path

import tui_gateway.server as server

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

    written = (home / "profiles" / "scout" / "config.yaml").read_text(encoding="utf-8")
    assert {ref: f"${{{ref}}}" in written for ref in _SECRETS} == dict.fromkeys(_SECRETS, True)
    assert [value for value in _SECRETS.values() if value in written] == []
