"""``config.set model "X --reasoning <level>"`` on the TUI gateway: the effort rides with the pick.

Applied AFTER the live swap (``agent.switch_model`` re-resolves ``reasoning_config`` from
config.yaml) and scoped like the pick: a session pin by default, ``agent.reasoning_effort`` on
``--global``. The Ink TUI picker and the classic CLI both emit this exact shape.
"""

from types import SimpleNamespace

import pytest

import tui_gateway.server as server
from hermes_cli.model_switch import persist_model_selection as _real_persist_model_selection

_real_write_config_key = server._write_config_key


class _Agent:
    def __init__(self):
        self.model, self.provider, self.base_url, self.api_key, self.api_mode = "old", "nous", "", "", ""
        self.reasoning_config = {"enabled": True, "effort": "medium"}

    def switch_model(self, **_kw):
        self.reasoning_config = {"enabled": True, "effort": "medium"}  # re-resolved from config


@pytest.fixture
def _quiet_switch(monkeypatch):
    result = SimpleNamespace(
        success=True, new_model="new/model", target_provider="nous", base_url="", api_key="key",
        api_mode="chat_completions", warning_message="", model_info=None, error_message="",
        runtime_capabilities=None)
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **_kw: result)
    monkeypatch.setattr("hermes_cli.model_switch.persist_model_selection", lambda _r: None)
    monkeypatch.setattr("hermes_cli.model_cost_guard.expensive_model_warning", lambda *a, **k: None)
    for name in ("_restart_slash_worker", "_persist_live_session_runtime", "_persist_live_session_system_prompt",
                 "_append_model_switch_marker", "_emit_session_info"):
        monkeypatch.setattr(server, name, lambda *a, **k: None)
    written = {}
    monkeypatch.setattr(server, "_write_config_key", lambda k, v: written.__setitem__(k, v))
    return written


def test_reasoning_flag_survives_the_swap_and_pins_the_session(_quiet_switch):
    agent = _Agent()
    session = {"agent": agent}

    out = server._apply_model_switch("sid", session, "new/model --provider nous --reasoning high --session")

    assert out["value"] == "new/model"
    assert agent.reasoning_config == {"enabled": True, "effort": "high"}
    assert session["create_reasoning_override"] == {"enabled": True, "effort": "high"}
    assert "agent.reasoning_effort" not in _quiet_switch


def test_reasoning_flag_with_global_writes_config_and_drops_the_pin(_quiet_switch):
    agent = _Agent()
    session = {"agent": agent, "create_reasoning_override": {"enabled": True, "effort": "low"}}

    server._apply_model_switch("sid", session, "new/model --provider nous --reasoning none --global")

    assert agent.reasoning_config == {"enabled": False}
    assert _quiet_switch["agent.reasoning_effort"] == "none"
    assert "create_reasoning_override" not in session

    with pytest.raises(ValueError):
        server._apply_model_switch("sid", {"agent": _Agent()}, "new/model --reasoning turbo")


def test_persistence_failure_rolls_back_live_runtime_and_session_overrides(_quiet_switch, monkeypatch):
    class MutatingAgent(_Agent):
        def switch_model(self, **kw):
            self.model = kw["new_model"]
            self.provider = kw["new_provider"]
            self.base_url = kw["base_url"]
            self.api_key = kw["api_key"]
            self.api_mode = kw["api_mode"]
            self.reasoning_config = {"enabled": False}

    agent = MutatingAgent()
    original_override = {"model": "old-pin", "provider": "nous"}
    original_reasoning = {"enabled": True, "effort": "low"}
    session = {
        "agent": agent,
        "model_override": original_override.copy(),
        "create_reasoning_override": original_reasoning.copy(),
    }
    calls = 0

    def fail_first_persist(_session):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("state db unavailable")

    monkeypatch.setattr(server, "_persist_live_session_runtime", fail_first_persist)

    with pytest.raises(OSError, match="state db unavailable"):
        server._apply_model_switch(
            "sid", session, "new/model --provider nous --reasoning high --session")

    assert (agent.model, agent.provider, agent.base_url, agent.api_key, agent.api_mode) == (
        "old", "nous", "", "", "")
    assert agent.reasoning_config == {"enabled": True, "effort": "medium"}
    assert session["model_override"] == original_override
    assert session["create_reasoning_override"] == original_reasoning
    assert "one_turn_model_restore" not in session
    assert calls == 2  # failed switched-runtime write, then restored-runtime write


@pytest.mark.parametrize("config_exists", [True, False])
def test_global_switch_failure_restores_config_yaml(_quiet_switch, monkeypatch, tmp_path, config_exists):
    """``--global`` writes config.yaml (model keys, then agent.reasoning_effort) before the live
    session runtime is persisted. When that later step fails, the rollback must put config.yaml
    back too, or the next process starts on the model the user was told the switch failed to."""
    home = tmp_path / "home"
    home.mkdir()
    config = home / "config.yaml"
    original = (b"# user comment\nmodel:\n  default: old/model\n  provider: nous\n"
                b"agent:\n  reasoning_effort: low\n")
    if config_exists:
        config.write_bytes(original)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(server, "_hermes_home", str(home))
    # The real durable writers, so the test sees the bytes a --global switch actually writes.
    monkeypatch.setattr("hermes_cli.model_switch.persist_model_selection", _real_persist_model_selection)
    monkeypatch.setattr(server, "_write_config_key", _real_write_config_key)
    written_during_switch = []

    calls = 0

    def fail_reasoning_persist(_session):
        nonlocal calls
        calls += 1
        if calls == 2:  # 1: _commit_agent_switch, 2: _apply_switch_reasoning, 3: rollback
            written_during_switch.append(config.read_bytes())
            raise OSError("state db unavailable")

    monkeypatch.setattr(server, "_persist_live_session_runtime", fail_reasoning_persist)
    agent = _Agent()
    session = {"agent": agent}

    with pytest.raises(OSError, match="state db unavailable"):
        server._apply_model_switch("sid", session, "new/model --provider nous --reasoning high --global")

    # Both durable writes had landed when the failure hit...
    assert b"new/model" in written_during_switch[0]
    assert b"reasoning_effort: high" in written_during_switch[0]
    # ...and the rollback put the file back exactly (or removed the one it created).
    if config_exists:
        assert config.read_bytes() == original
    else:
        assert not config.exists()
    assert agent.model == "old"
    assert calls == 3
