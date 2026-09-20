"""``config.set model "X --reasoning <level>"`` on the TUI gateway: the effort rides with the pick.

Applied AFTER the live swap (``agent.switch_model`` re-resolves ``reasoning_config`` from
config.yaml) and scoped like the pick: a session pin by default, ``agent.reasoning_effort`` on
``--global``. The Ink TUI picker and the classic CLI both emit this exact shape.
"""

from types import SimpleNamespace

import pytest

import tui_gateway.server as server


class _Agent:
    def __init__(self):
        self.model, self.provider, self.base_url, self.api_key, self.api_mode = "old", "nous", "", "", ""
        self.reasoning_config = {"enabled": True, "effort": "medium"}
        self.switch_kwargs = {}

    def switch_model(self, **kwargs):
        self.switch_kwargs = kwargs
        self.reasoning_config = kwargs.get(
            "reasoning_config_override",
            {"enabled": True, "effort": "medium"},  # re-resolved from config
        )


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

    with pytest.raises(ValueError, match="--reasoning takes"):
        server._apply_model_switch("sid", {"agent": _Agent()}, "new/model --reasoning turbo")


def test_plain_model_switch_preserves_existing_session_reasoning_pin(_quiet_switch):
    agent = _Agent()
    explicit_disabled = {"enabled": False}
    session = {"agent": agent, "create_reasoning_override": explicit_disabled}

    out = getattr(server, "_apply_model_switch")(
        "sid", session, "new/model --provider nous --session"
    )

    assert out["value"] == "new/model"
    assert agent.reasoning_config == explicit_disabled
    assert agent.switch_kwargs["reasoning_config_override"] == explicit_disabled
