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

import pytest

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
        assert info["reasoning_effort_wire"] == ""

    def test_wire_level_is_what_the_route_actually_sends(self) -> None:
        """`ultra` is a Hermes-internal step (#61634): the route clamps it, and the Desktop must be able to
        say so ("ultra sends max on this route") instead of presenting Ultra as a distinct wire level."""
        info = _session_info(_agent({"enabled": True, "effort": "ultra"}))
        assert info["reasoning_effort"] == "ultra"
        assert info["reasoning_effort_wire"] == "max"
        # Verbatim levels report themselves, so clients only annotate a real clamp.
        assert _session_info(_agent({"enabled": True, "effort": "high"}))["reasoning_effort_wire"] == "high"
        assert _session_info(_agent({"enabled": False}))["reasoning_effort_wire"] == ""

    def test_remote_agent_reports_session_overrides(self) -> None:
        info = _session_info(
            None,
            {
                "create_reasoning_override": {"enabled": True, "effort": "high"},
                "create_service_tier_override": "priority",
                "_compute_host_active": True,
                "_metadata_mirror": {"model": "gpt-5", "provider": "openai"},
            },
        )
        assert info["reasoning_effort"] == "high"
        assert info["service_tier"] == "priority"
        assert info["fast"] is True

    def test_remote_agent_preserves_override_precedence_and_sentinels(self) -> None:
        mirrored = _session_info(
            None,
            {
                "_metadata_mirror": {
                    "reasoning_effort": "medium",
                    "service_tier": "flex",
                },
                "create_reasoning_override": {"enabled": True, "effort": "high"},
                "create_service_tier_override": "priority",
            },
        )
        assert mirrored["reasoning_effort"] == "medium"
        assert mirrored["service_tier"] == "flex"
        assert mirrored["fast"] is False

        disabled = _session_info(
            None,
            {
                "create_reasoning_override": {"enabled": False},
                "create_service_tier_override": "",
            },
        )
        assert disabled["reasoning_effort"] == "none"
        assert disabled["service_tier"] == ""
        assert disabled["fast"] is False

        inherited = _session_info(None, {})
        assert inherited["reasoning_effort"] == ""
        assert inherited["service_tier"] == ""
        assert inherited["fast"] is False

        live_agent = _agent(None)
        live_agent.service_tier = ""
        normal = _session_info(live_agent, {"_metadata_mirror": {"service_tier": "priority"}})
        assert normal["service_tier"] == ""
        assert normal["fast"] is False


    @pytest.mark.parametrize("mirrored_tier,persisted_tier,expected", [
        ("", "priority", ""),
        (None, "priority", "priority"),
        (None, "", ""),
        (None, None, ""),
        ("flex", "priority", "flex"),
    ])
    def test_remote_tier_only_inherits_when_mirror_is_unset(self, mirrored_tier, persisted_tier, expected):
        info = _session_info(None, {
            "_metadata_mirror": {"model": "gpt-5", "provider": "openai", "service_tier": mirrored_tier},
            "create_service_tier_override": persisted_tier,
        })
        assert info["service_tier"] == expected
        assert info["fast"] is (expected == "priority")


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
