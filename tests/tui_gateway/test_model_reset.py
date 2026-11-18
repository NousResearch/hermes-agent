"""TUI gateway ``/model reset`` (``config.set model`` and Desktop's slash mirror both land here)."""

import json
from types import SimpleNamespace

from tui_gateway import server


class _Agent:
    def __init__(self):
        self.model, self.provider, self.base_url, self.api_key, self.api_mode = "pinned", "pin-prov", "", "", ""
        self.switched_to = []

    def switch_model(self, **kw):
        self.switched_to.append((kw["new_model"], kw["new_provider"]))
        self.model, self.provider = kw["new_model"], kw["new_provider"]


def test_reset_drops_the_pin_and_adopts_config_without_repinning(monkeypatch, tmp_path):
    (tmp_path / "config.yaml").write_text(json.dumps({"model": {"default": "cfg-model", "provider": "cfg-prov"}}))
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    resolved = []

    def _switch_model(**kw):  # provider resolution is the boundary; record what reset asks for
        resolved.append((kw["raw_input"], kw["explicit_provider"], kw["is_global"]))
        return SimpleNamespace(
            success=True, new_model=kw["raw_input"], target_provider=kw["explicit_provider"], base_url="",
            api_key="k", api_mode="chat_completions", warning_message="", model_info=None, error_message="",
            runtime_capabilities=None, request_overrides=None, usage_notice="")

    monkeypatch.setattr("hermes_cli.model_switch.switch_model", _switch_model)
    agent = _Agent()
    session = {"agent": agent, "model_override": {"model": "pinned", "provider": "pin-prov"},
               "one_turn_model_restore": {"model": "pinned"}, "config_model_seen": ("cfg-model", "cfg-prov")}

    out = server._apply_model_switch("sid", session, "reset")

    assert resolved == [("cfg-model", "cfg-prov", False)]  # never persisted to config.yaml
    assert agent.switched_to == [("cfg-model", "cfg-prov")]
    assert "model_override" not in session  # no fresh pin: later config edits reach this chat
    assert "one_turn_model_restore" not in session
    assert out["value"] == "cfg-model" and out["scope"] == "session"

    again = server._apply_model_switch("sid", session, "reset")
    assert again["warning"] != out["warning"] and agent.switched_to == [("cfg-model", "cfg-prov")]
