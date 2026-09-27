"""Fast-mode (service tier) session scoping in the TUI gateway (desktop backend).

Sibling of test_reasoning_session_scope.py — the ``reasoning`` key was made
session-scoped when a session is targeted, but ``fast`` kept writing the
global ``agent.service_tier`` to config.yaml on every call. The desktop's
per-model presets call ``config.set key=fast`` on every model selection, so
toggling fast in ONE session silently flipped the tier for every other
session, profile, CLI, and gateway build ("switch one session, switches
everywhere").

Contract under test:

1. ``config.set key=fast`` with a session must NOT write config.yaml; it pins
   ``create_service_tier_override`` ("priority" / "" for explicit normal) so
   lazily-built sessions and rebuilds keep the choice.
2. Without a session it persists globally, unchanged. A session_id the
   backend no longer holds is refused with 4001, not treated as "no session".
3. ``config.get key=fast`` must read a pre-build session's pin.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import tui_gateway.server as server

FAST_OVERRIDES = {"service_tier": "priority"}

def _agent(service_tier=None):
    return SimpleNamespace(
        reasoning_config=None,
        service_tier=service_tier,
        request_overrides={},
        model="gpt-6",
        provider="openai",
        session_id="sess-key",
    )

def _set(params: dict) -> dict:
    return server._methods["config.set"]("rid-1", params)

def _get(params: dict) -> dict:
    return server._methods["config.get"]("rid-1", params)

class TestConfigSetFastSessionScope:
    """Session-targeted fast changes must never touch global config."""

    def test_session_scoped_fast_skips_global_write(self) -> None:
        agent = _agent()
        session = {"session_key": "k1", "agent": agent}
        with patch.dict(server._sessions, {"s1": session}, clear=False), \
                patch.object(server, "_write_config_key") as write_key, \
                patch.object(server, "_persist_live_session_runtime"), \
                patch.object(server, "_emit"), \
                patch(
                    "hermes_cli.models.resolve_fast_mode_overrides",
                    return_value=FAST_OVERRIDES,
                ):
            resp = _set({"key": "fast", "session_id": "s1", "value": "fast"})
        assert resp["result"]["value"] == "fast"
        assert agent.service_tier == "priority"
        assert session["create_service_tier_override"] == "priority"
        write_key.assert_not_called()

    def test_lazy_session_pins_create_override(self) -> None:
        """A pre-build (agent=None) session must keep the change for the
        deferred agent build instead of dropping it."""
        session = {
            "session_key": "k3",
            "agent": None,
            "model_override": {"model": "gpt-6", "provider": "openai"},
        }
        with patch.dict(server._sessions, {"s3": session}, clear=False), \
                patch.object(server, "_write_config_key") as write_key, \
                patch(
                    "hermes_cli.models.resolve_fast_mode_overrides",
                    return_value=FAST_OVERRIDES,
                ):
            resp = _set({"key": "fast", "session_id": "s3", "value": "fast"})
        assert resp["result"]["value"] == "fast"
        assert session["create_service_tier_override"] == "priority"
        write_key.assert_not_called()

    def test_toggle_flips_prebuild_pin(self) -> None:
        """An empty value toggles from the session's pin, not the global."""
        session = {
            "session_key": "k5",
            "agent": None,
            "create_service_tier_override": "priority",
        }
        with patch.dict(server._sessions, {"s5": session}, clear=False), \
                patch.object(server, "_write_config_key") as write_key:
            resp = _set({"key": "fast", "session_id": "s5", "value": ""})
        assert resp["result"]["value"] == "normal"
        assert session["create_service_tier_override"] == ""
        write_key.assert_not_called()

    def test_no_session_persists_globally(self) -> None:
        with patch.object(server, "_write_config_key") as write_key:
            resp = _set({"key": "fast", "value": "normal"})
        assert resp["result"]["value"] == "normal"
        write_key.assert_called_once_with("agent.service_tier", "normal")

    def test_stale_session_id_is_refused_not_persisted_globally(self) -> None:
        """A runtime id the backend no longer holds (reaped / re-minted) is not "no session": it
        must 4001 so the client resumes, not rewrite the profile's tier for every surface."""
        with patch.dict(server._sessions, {}, clear=True), \
                patch.object(server, "_write_config_key") as write_key:
            resp = _set({"key": "fast", "session_id": "reaped-sid", "value": "normal"})
        assert resp.get("error", {}).get("code") == 4001, resp
        write_key.assert_not_called()

class TestConfigGetFastSessionScope:
    def test_reads_prebuild_pin(self) -> None:
        session = {
            "session_key": "k6",
            "agent": None,
            "create_service_tier_override": "priority",
        }
        with patch.dict(server._sessions, {"s6": session}, clear=False):
            resp = _get({"key": "fast", "session_id": "s6"})
        assert resp["result"]["value"] == "fast"


class TestSessionInfoFastFollowsTheRoute:
    """``session.info`` reports Fast only where the priority tier reaches the wire. A profile-wide
    ``service_tier: fast`` on a local model sends nothing, so the Fast switch and label stay hidden."""

    @staticmethod
    def _info(**agent_fields) -> dict:
        agent = SimpleNamespace(**{
            "reasoning_config": None, "service_tier": "priority", "request_overrides": {},
            "session_id": "sess-key", "api_mode": "chat_completions", **agent_fields,
        })
        return server._session_info(agent, {"session_key": "k7", "agent": agent})

    def test_first_party_route_reports_fast(self) -> None:
        info = self._info(model="gpt-5.4", provider="openai", base_url="https://api.openai.com/v1")
        assert (info["service_tier"], info["fast"]) == ("priority", True)

    def test_anthropic_route_reads_the_anthropic_base_url(self) -> None:
        info = self._info(model="claude-opus-5", provider="anthropic", api_mode="anthropic_messages",
                          base_url="", _anthropic_base_url="https://api.anthropic.com")
        assert info["fast"] is True

    def test_local_model_keeps_the_tier_but_reports_no_fast(self) -> None:
        info = self._info(model="Qwen3.8-27B-UD-Q4_K_M", provider="llamacpp", base_url="http://127.0.0.1:18434/v1")
        assert (info["service_tier"], info["fast"]) == ("priority", False)

    def test_fast_capable_model_behind_a_proxy_reports_no_fast(self) -> None:
        info = self._info(model="gpt-5.4", provider="openrouter", base_url="https://openrouter.ai/api/v1")
        assert info["fast"] is False

    def test_pending_switch_is_judged_by_the_new_route(self) -> None:
        """Mid-turn the agent still holds the old base URL; the pending pick decides."""
        agent = SimpleNamespace(
            reasoning_config=None, service_tier="priority", request_overrides={}, session_id="sess-key",
            api_mode="chat_completions", model="gpt-5.4", provider="openai", base_url="https://api.openai.com/v1")
        session = {"session_key": "k8", "agent": agent, "pending_model_switch": {
            "display_model": "Qwen3.8-27B-UD-Q4_K_M", "display_provider": "llamacpp"}}
        assert server._session_info(agent, session)["fast"] is False
        session["pending_model_switch"] = {"display_model": "claude-opus-5", "display_provider": "anthropic"}
        assert server._session_info(agent, session)["fast"] is True
