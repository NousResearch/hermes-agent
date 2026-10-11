"""Regression tests for deferring TUI /moa while a turn owns the live agent."""

from types import SimpleNamespace

from tui_gateway import server


class _MoaConfig:
    @staticmethod
    def moa_usage():
        return "usage: /moa <prompt>"

    @staticmethod
    def normalize_moa_config(_config):
        return {"default_preset": "default"}


def _session(*, running=True):
    return {
        "agent": SimpleNamespace(model="gpt-4", provider="openai"),
        "model_override": {"model": "standing", "provider": "openai"},
        "running": running,
    }


def test_moa_during_running_turn_does_not_touch_live_agent(monkeypatch):
    session = _session(running=True)
    switched = []
    monkeypatch.setattr(server, "_tools_mod", lambda _name: _MoaConfig)
    monkeypatch.setattr(server, "_load_cfg", lambda: {"moa": {}})
    monkeypatch.setattr(server, "_apply_model_switch", lambda *args, **kwargs: switched.append(args))

    result = server._cmd_moa("r1", {"session_id": "sid"}, session, "moa", "compare answers")

    assert result["result"]["type"] == "send"
    assert result["result"]["message"] == "compare answers"
    # Ink must queue (not steer/interrupt) this prompt, or a stale pending entry would remain.
    assert result["result"]["queued"] is True
    assert session["agent"].model == "gpt-4"
    assert switched == []
    assert session["pending_moa"] == [{
        "prompt": "compare answers",
        "preset": "default",
        "restore": {"override": {"model": "standing", "provider": "openai"},
                    "model": "gpt-4", "provider": "openai"},
    }]


def test_pending_moa_applies_to_matching_next_turn_and_restores(monkeypatch):
    session = _session(running=False)
    session["pending_moa"] = [{
        "prompt": "compare answers", "preset": "default",
        "restore": {"override": {"model": "standing", "provider": "openai"},
                    "model": "gpt-4", "provider": "openai"},
    }]
    calls = []

    def apply(_sid, _session, raw, **kwargs):
        calls.append(raw)
        if "--provider moa" in raw:
            _session["agent"].model = "default"
            _session["agent"].provider = "moa"
        else:
            _session["agent"].model = "gpt-4"
            _session["agent"].provider = "openai"
        return {}

    monkeypatch.setattr(server, "_apply_model_switch", apply)
    server._apply_pending_moa("sid", session, "unrelated prompt")
    assert session["agent"].model == "gpt-4"
    assert session["pending_moa"]

    server._apply_pending_moa("sid", session, "compare answers")
    assert calls == ["default --provider moa"]
    assert session["agent"].model == "default"
    assert session["moa_one_shot_restore"]["override"]["model"] == "standing"

    server._restore_moa_one_shot("sid", session)
    assert calls == ["default --provider moa", "gpt-4 --provider openai"]
    assert session["agent"].model == "gpt-4"
    assert session["model_override"] == {"model": "standing", "provider": "openai"}
