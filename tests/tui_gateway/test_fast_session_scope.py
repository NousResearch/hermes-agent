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
2. Without a session it persists globally, unchanged.
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


    def test_falls_back_to_global(self) -> None:
        with patch.object(server, "_load_service_tier", return_value="priority"):
            resp = _get({"key": "fast"})
        assert resp["result"]["value"] == "fast"


class TestFastAcceptedOnTrustedProxyRoute:
    """A live session must not be harder to toggle than a fresh one.

    The other tests here stub ``resolve_fast_mode_overrides``, so the real route
    gate never runs. It is exactly that gate which rejected an EXISTING chat
    (live agent supplies provider + base_url) while a NEW chat -- no agent, so
    nothing to check -- was accepted. Same model, same config, two answers.
    """

    PROXY_BASE_URL = "https://proxy.example.com/v1"

    def _proxy_agent(self):
        return SimpleNamespace(
            reasoning_config=None,
            service_tier=None,
            request_overrides={},
            model="gpt-5.4",
            provider="llm-proxy",
            base_url=self.PROXY_BASE_URL,
            session_id="sess-key",
        )

    def _set_fast_with_trust(self, session_id, session, trusted):
        with patch.dict(server._sessions, {session_id: session}, clear=False), \
                patch.object(server, "_write_config_key"), \
                patch.object(server, "_persist_live_session_runtime"), \
                patch.object(server, "_emit"), \
                patch("hermes_cli.config.load_config",
                      return_value={"agent": {"fast_mode_trusted_endpoints": trusted}}):
            return _set({"key": "fast", "session_id": session_id, "value": "fast"})

    def test_live_session_on_opted_in_endpoint_is_accepted(self) -> None:
        agent = self._proxy_agent()
        resp = self._set_fast_with_trust(
            "s-proxy", {"session_key": "k-proxy", "agent": agent},
            ["llm-proxy:proxy.example.com"])

        assert "error" not in resp, resp.get("error")
        assert resp["result"]["value"] == "fast"
        assert agent.service_tier == "priority"
        assert agent.request_overrides.get("service_tier") == "priority"

    def test_live_session_without_opt_in_is_still_refused(self) -> None:
        agent = self._proxy_agent()
        resp = self._set_fast_with_trust(
            "s-other", {"session_key": "k-other", "agent": agent}, [])

        assert "error" in resp
        assert agent.service_tier is None
