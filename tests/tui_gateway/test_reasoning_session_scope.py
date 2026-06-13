from types import SimpleNamespace

import pytest
import yaml


@pytest.fixture
def server(tmp_path, monkeypatch):
    import tui_gateway.server as mod

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "agent:\n  reasoning_effort: medium\ndisplay:\n  show_reasoning: false\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "_hermes_home", tmp_path)
    monkeypatch.setattr(mod, "_session_info", lambda agent, session: {"reasoning_effort": getattr(agent, "reasoning_config", {})})
    monkeypatch.setattr(mod, "_emit", lambda *args, **kwargs: None)
    mod._cfg_cache = None
    mod._cfg_mtime = 0
    mod._sessions.clear()
    yield mod, config_path
    mod._sessions.clear()
    mod._cfg_cache = None
    mod._cfg_mtime = 0


def test_tui_reasoning_set_is_session_scoped_by_default(server):
    mod, config_path = server
    sid = "s1"
    session = {
        "session_key": "k1",
        "agent": SimpleNamespace(reasoning_config={"enabled": True, "effort": "medium"}),
        "running": False,
    }
    mod._sessions[sid] = session

    result = mod.handle_request(
        {"jsonrpc": "2.0", "id": "1", "method": "config.set", "params": {"key": "reasoning", "value": "high", "session_id": sid}}
    )

    assert result["result"]["scope"] == "session"
    assert session["agent"].reasoning_config == {"enabled": True, "effort": "high"}
    assert session["reasoning_config_override"] == {"enabled": True, "effort": "high"}
    assert yaml.safe_load(config_path.read_text(encoding="utf-8"))["agent"]["reasoning_effort"] == "medium"

    got = mod.handle_request(
        {"jsonrpc": "2.0", "id": "2", "method": "config.get", "params": {"key": "reasoning", "session_id": sid}}
    )
    assert got["result"]["value"] == "high"


def test_tui_reasoning_global_flag_persists_and_clears_session_override(server):
    mod, config_path = server
    sid = "s1"
    session = {
        "session_key": "k1",
        "agent": SimpleNamespace(reasoning_config={"enabled": True, "effort": "high"}),
        "reasoning_config_override": {"enabled": True, "effort": "high"},
        "running": False,
    }
    mod._sessions[sid] = session

    result = mod.handle_request(
        {"jsonrpc": "2.0", "id": "1", "method": "config.set", "params": {"key": "reasoning", "value": "low --global", "session_id": sid}}
    )

    assert result["result"]["scope"] == "global"
    assert session["agent"].reasoning_config == {"enabled": True, "effort": "low"}
    assert session["reasoning_config_override"] is None
    assert yaml.safe_load(config_path.read_text(encoding="utf-8"))["agent"]["reasoning_effort"] == "low"
