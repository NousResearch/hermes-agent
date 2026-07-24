"""The turn-time image mode decision must report downgrades, not hide them.

A user who forces ``agent.image_input_mode: native`` on a Codex app-server
session (or hits a decision failure) previously got a text description with
zero UI signal that the setting was overridden — the only evidence was a
stderr line in the gateway log (#66829).
"""

from types import SimpleNamespace

import agent.image_routing as image_routing
from tui_gateway.server import _decide_turn_image_mode


def _agent(api_mode: str = "chat_completions") -> SimpleNamespace:
    return SimpleNamespace(provider="openai", model="gpt-5.5", api_mode=api_mode)


class TestDecideTurnImageMode:
    def test_native_decision_is_honored_with_no_reason(self, monkeypatch):
        monkeypatch.setattr(
            image_routing, "decide_image_input_mode", lambda p, m, c, **kw: "native"
        )
        mode, reason = _decide_turn_image_mode(_agent())
        assert mode == "native"
        assert reason is None

    def test_codex_app_server_downgrades_native_with_reason(self, monkeypatch):
        monkeypatch.setattr(
            image_routing, "decide_image_input_mode", lambda p, m, c, **kw: "native"
        )
        mode, reason = _decide_turn_image_mode(_agent(api_mode="codex_app_server"))
        assert mode == "text"
        assert reason is not None and "Codex app-server" in reason

    def test_codex_app_server_text_decision_has_no_spurious_reason(self, monkeypatch):
        # When the decision is already text, nothing was overridden — the
        # status line must not cry wolf.
        monkeypatch.setattr(
            image_routing, "decide_image_input_mode", lambda p, m, c, **kw: "text"
        )
        mode, reason = _decide_turn_image_mode(_agent(api_mode="codex_app_server"))
        assert mode == "text"
        assert reason is None

    def test_decision_failure_falls_back_to_text_with_reason(self, monkeypatch):
        def _boom(p, m, c, **kw):
            raise RuntimeError("metadata backend unreachable")

        monkeypatch.setattr(image_routing, "decide_image_input_mode", _boom)
        mode, reason = _decide_turn_image_mode(_agent())
        assert mode == "text"
        assert reason is not None
        assert "RuntimeError" in reason and "metadata backend unreachable" in reason

    def test_trace_line_names_provider_model_and_both_modes(self, monkeypatch, capsys):
        monkeypatch.setattr(
            image_routing, "decide_image_input_mode", lambda p, m, c, **kw: "native"
        )
        _decide_turn_image_mode(_agent(api_mode="codex_app_server"))
        err = capsys.readouterr().err
        assert "[tui_gateway] image_routing:" in err
        assert "decided=native" in err
        assert "final=text" in err
