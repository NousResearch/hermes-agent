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

import json
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


class TestStoredReasoningProvenance:
    @pytest.mark.parametrize(
        ("model_config", "expected"),
        [
            ({"reasoning_config": {"enabled": True, "effort": "low"}}, {}),
            (
                {"reasoning_config_override": {"enabled": False}},
                {"reasoning_config_override": {"enabled": False}},
            ),
        ],
    )
    def test_only_explicit_override_markers_are_restored(self, model_config, expected) -> None:
        assert server._stored_session_runtime_overrides({"model_config": model_config}) == expected

    def test_runtime_persistence_writes_and_clears_only_the_explicit_marker(self) -> None:
        class _DB:
            def __init__(self):
                self.model_config = {"reasoning_config_override": {"enabled": True, "effort": "low"}}

            def get_session(self, _session_key):
                return {"model_config": json.dumps(self.model_config)}

            def update_session_meta(self, _session_key, model_config, _model):
                self.model_config = json.loads(model_config)

        db = _DB()
        agent = _agent({"enabled": False})
        agent._session_db = db
        session = {
            "agent": agent,
            "session_key": "sess-key",
            "create_reasoning_override": {"enabled": False},
        }

        server._persist_live_session_runtime(session)
        assert db.model_config["reasoning_config_override"] == {"enabled": False}

        session.pop("create_reasoning_override")
        server._persist_live_session_runtime(session)
        assert "reasoning_config_override" not in db.model_config
        assert db.model_config["reasoning_config"] == {"enabled": False}

