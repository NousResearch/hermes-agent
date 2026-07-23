"""TUI/Desktop reasoning-visibility coverage (PR review follow-up).

The messaging gateway and classic CLI resolve display.show_reasoning into
``reasoning_config["include_thoughts"]`` so Gemini/Vertex suppress thought
summaries at the request. The TUI backend must do the same at its
agent-construction sites, and `/reasoning show|hide` must synchronize the
live cached agent — otherwise a TUI Gemini/Vertex session with hidden
reasoning keeps requesting summaries until the agent is rebuilt.
"""

from types import SimpleNamespace

import tui_gateway.server as tui_server


def _cfg(show_reasoning: bool, effort: str = "medium") -> dict:
    return {
        "model": {"default": "google/gemini-3.1-pro-preview"},
        "agent": {"reasoning_effort": effort},
        "display": {"show_reasoning": show_reasoning},
    }


class TestAnnotateReasoningVisibility:
    def test_display_off_marks_thoughts_hidden(self, monkeypatch):
        monkeypatch.setattr(tui_server, "_load_cfg", lambda: _cfg(False))
        result = tui_server._annotate_reasoning_visibility(
            tui_server._load_reasoning_config()
        )
        assert result == {
            "enabled": True,
            "effort": "medium",
            "include_thoughts": False,
        }

    def test_display_on_requests_thoughts(self, monkeypatch):
        monkeypatch.setattr(tui_server, "_load_cfg", lambda: _cfg(True, "high"))
        result = tui_server._annotate_reasoning_visibility(
            tui_server._load_reasoning_config()
        )
        assert result == {
            "enabled": True,
            "effort": "high",
            "include_thoughts": True,
        }

    def test_none_stays_none(self, monkeypatch):
        monkeypatch.setattr(tui_server, "_load_cfg", lambda: _cfg(False))
        assert tui_server._annotate_reasoning_visibility(None) is None

    def test_session_override_not_mutated(self, monkeypatch):
        """Stored /reasoning overrides must not accumulate visibility state."""
        monkeypatch.setattr(tui_server, "_load_cfg", lambda: _cfg(False))
        override = {"enabled": True, "effort": "xhigh"}
        annotated = tui_server._annotate_reasoning_visibility(override)
        assert annotated["include_thoughts"] is False
        assert override == {"enabled": True, "effort": "xhigh"}

    def test_loader_stays_effort_only(self, monkeypatch):
        """Parity contract: the loader itself must not grow visibility keys
        (tests/tui_gateway/test_reasoning_config_per_model.py compares it
        against the gateway loader verbatim)."""
        monkeypatch.setattr(tui_server, "_load_cfg", lambda: _cfg(False))
        assert "include_thoughts" not in (tui_server._load_reasoning_config() or {})


class TestLiveToggleSync:
    def _session_with_agent(self, reasoning_config):
        agent = SimpleNamespace(reasoning_config=reasoning_config)
        return {"agent": agent, "show_reasoning": True}, agent

    def test_hide_updates_live_agent_without_touching_effort(self):
        session, agent = self._session_with_agent(
            {"enabled": True, "effort": "high", "include_thoughts": True}
        )
        tui_server._sync_live_agent_reasoning_visibility(session, False)
        assert agent.reasoning_config == {
            "enabled": True,
            "effort": "high",
            "include_thoughts": False,
        }

    def test_show_reenables_thoughts(self):
        session, agent = self._session_with_agent(
            {"enabled": True, "effort": "medium", "include_thoughts": False}
        )
        tui_server._sync_live_agent_reasoning_visibility(session, True)
        assert agent.reasoning_config["include_thoughts"] is True
        assert agent.reasoning_config["effort"] == "medium"

    def test_no_agent_is_a_noop(self):
        tui_server._sync_live_agent_reasoning_visibility({"agent": None}, False)

    def test_agent_without_reasoning_config_is_a_noop(self):
        session, agent = self._session_with_agent(None)
        tui_server._sync_live_agent_reasoning_visibility(session, False)
        assert agent.reasoning_config is None
