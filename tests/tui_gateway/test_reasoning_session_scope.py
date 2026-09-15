"""Reasoning-effort session scoping in the TUI gateway (desktop backend).

Covers the "desktop reverts thinking to medium after one turn" report:

1. ``_session_info`` must report ``reasoning_effort: "none"`` when reasoning
   is disabled — reporting ``""`` (indistinguishable from "unset") made the
   desktop adopt the empty value after the first turn, wiping its sticky
   "thinking off" pick so every later chat reverted to the default effort.

2. ``config.set key=reasoning`` with a live session must be session-scoped:
   it must NOT rewrite the global ``agent.reasoning_effort`` in config.yaml
   (the desktop model menu applies a per-model preset on every selection,
   which was silently clobbering the user's configured value), and it must
   land on ``create_reasoning_override`` so lazily-built sessions (agent not
   constructed until the first prompt) don't drop the change.

3. ``_load_reasoning_config`` must honor a YAML boolean False
   (``reasoning_effort: false`` / ``off`` / ``no``) as thinking-disabled.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import tui_gateway.server as server
from tui_gateway.server import _session_info


def _agent(reasoning_config):
    return SimpleNamespace(
        reasoning_config=reasoning_config,
        service_tier=None,
        model="glm-5",
        provider="zai",
        session_id="sess-key",
    )


class TestSessionInfoReasoningEffort:
    """Disabled reasoning must be reported as 'none', never ''."""

    def test_disabled_reports_none(self) -> None:
        info = _session_info(_agent({"enabled": False}))
        assert info["reasoning_effort"] == "none"

    def test_enabled_reports_effort(self) -> None:
        info = _session_info(_agent({"enabled": True, "effort": "high"}))
        assert info["reasoning_effort"] == "high"

    def test_unset_reports_empty(self) -> None:
        info = _session_info(_agent(None))
        assert info["reasoning_effort"] == ""


class TestConfigSetReasoningSessionScope:
    """Session-targeted reasoning changes must not touch global config."""

    def _dispatch(self, params: dict) -> dict:
        handler = server._methods["config.set"]
        return handler("rid-1", params)

    def test_session_scoped_set_skips_global_write(self) -> None:
        agent = _agent(None)
        session = {"session_key": "k1", "agent": agent}
        with patch.dict(server._sessions, {"s1": session}, clear=False), \
                patch.object(server, "_write_config_key") as write_key, \
                patch.object(server, "_persist_live_session_runtime"), \
                patch.object(server, "_emit"):
            resp = self._dispatch(
                {"key": "reasoning", "session_id": "s1", "value": "none"}
            )
        assert resp["result"]["value"] == "none"
        assert agent.reasoning_config == {"enabled": False}
        write_key.assert_not_called()


    def test_no_session_persists_globally(self) -> None:
        with patch.object(server, "_write_config_key") as write_key:
            resp = self._dispatch({"key": "reasoning", "value": "low"})
        assert resp["result"]["value"] == "low"
        write_key.assert_called_once_with("agent.reasoning_effort", "low")

    def test_unknown_value_rejected(self) -> None:
        resp = self._dispatch({"key": "reasoning", "value": "bogus"})
        assert "error" in resp


class TestLoadReasoningConfigYamlBoolean:
    """YAML `reasoning_effort: false` means disabled, not default."""

    def test_boolean_false_disables(self) -> None:
        with patch.object(
            server, "_load_cfg", return_value={"agent": {"reasoning_effort": False}}
        ):
            assert server._load_reasoning_config() == {"enabled": False}

    def test_string_false_disables(self) -> None:
        with patch.object(
            server, "_load_cfg", return_value={"agent": {"reasoning_effort": "false"}}
        ):
            assert server._load_reasoning_config() == {"enabled": False}


class TestSessionNoneReachesDeepSeekWire:
    """Desktop ``config.set value=none`` must disable DeepSeek V4 thinking.

    ``{effort: "none"}`` without ``enabled: False`` is what ``_session_info``
    already reports as Off; the profile used to ignore it and send enabled.
    """

    def test_session_info_effort_none_without_enabled_reports_none(self) -> None:
        info = _session_info(_agent({"effort": "none"}))
        assert info["reasoning_effort"] == "none"

    def test_config_set_none_on_lazy_session_pins_disabled_override(self) -> None:
        session = {"session_key": "k-lazy", "agent": None}
        with patch.dict(server._sessions, {"s-lazy": session}, clear=False), \
                patch.object(server, "_write_config_key") as write_key:
            resp = server._methods["config.set"](
                "rid-1", {"key": "reasoning", "session_id": "s-lazy", "value": "none"}
            )
        assert resp["result"]["value"] == "none"
        assert session["create_reasoning_override"] == {"enabled": False}
        write_key.assert_not_called()
        kw = server._deferred_build_agent_kwargs(session, session_db=None)
        assert kw["reasoning_config_override"] == {"enabled": False}

    def test_effort_none_override_emits_thinking_disabled(self) -> None:
        import model_tools  # noqa: F401
        import providers
        from agent.transports.chat_completions import ChatCompletionsTransport

        profile = providers.get_provider_profile("deepseek")
        kwargs = ChatCompletionsTransport().build_kwargs(
            model="deepseek-v4.1-flash-expires-on-0910",
            messages=[{"role": "user", "content": "ping"}],
            tools=None,
            provider_profile=profile,
            reasoning_config={"effort": "none"},
            base_url="https://api.deepseek.com/v1",
            provider_name="deepseek",
        )
        assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
        assert "reasoning_effort" not in kwargs

