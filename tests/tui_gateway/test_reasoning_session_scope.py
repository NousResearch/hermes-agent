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
from unittest.mock import MagicMock, patch

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


class TestCompressionChildKeepsExplicitMarker:
    """save -> compress -> reopen: the explicit marker must survive the compression child.

    Compression publishes the child row from ``agent._session_init_model_config``. That dict was seeded with only
    the effective ``reasoning_config``, so the child looked like a legacy row and resume dropped the explicit pin.
    """

    EXPLICIT = {"enabled": True, "effort": "high"}

    @staticmethod
    def _agent(db, session_id, reasoning_config):
        return SimpleNamespace(
            _session_db=db, session_id=session_id, platform="tui", model="glm-5", provider="zai", base_url="",
            api_mode="", service_tier=None, reasoning_config=reasoning_config,
            _session_init_model_config={"max_iterations": 5, "reasoning_config": reasoning_config, "max_tokens": None},
            working_directory=None, _memory_manager=None, context_compressor=SimpleNamespace(),
            _flush_messages_to_session_db=lambda *a, **k: None, _persist_user_message_idx=None,
            _session_messages=None, _gateway_session_key=None, _cached_system_prompt="sys",
        )

    @staticmethod
    def _compress(agent):
        from agent import conversation_compression as cc

        cc._publish_rotated_compaction(
            agent, [{"role": "user", "content": "hello"}], [{"role": "user", "content": "[handoff]"}],
            new_system_prompt="sys", lease=SimpleNamespace(holder=None, ttl=60.0, watermark=None),
            old_session_id="parent", compressed_user_turn_outcome="none",
        )

    @pytest.fixture
    def db(self, tmp_path):
        from hermes_state import SessionDB

        database = SessionDB(tmp_path / "state.db")
        database.create_session("parent", source="tui", model="glm-5")
        database.append_message("parent", "user", "hello")
        try:
            yield database
        finally:
            database.close()

    def test_explicit_marker_survives_save_compress_reopen(self, db) -> None:
        agent = self._agent(db, "parent", self.EXPLICIT)
        session = {"agent": agent, "session_key": "parent", "create_reasoning_override": self.EXPLICIT}

        server._persist_live_session_runtime(session)
        assert json.loads(db.get_session("parent")["model_config"])["reasoning_config_override"] == self.EXPLICIT

        self._compress(agent)

        assert agent.session_id != "parent"
        child = db.get_session(agent.session_id)
        child_config = json.loads(child["model_config"])
        assert child_config["reasoning_config"] == self.EXPLICIT
        assert child_config["reasoning_config_override"] == self.EXPLICIT
        assert server._stored_session_runtime_overrides(child)["reasoning_config_override"] == self.EXPLICIT

    def test_cleared_override_is_not_carried_into_the_child(self, db) -> None:
        agent = self._agent(db, "parent", self.EXPLICIT)
        session = {"agent": agent, "session_key": "parent", "create_reasoning_override": self.EXPLICIT}
        server._persist_live_session_runtime(session)
        assert agent._session_init_model_config["reasoning_config_override"] == self.EXPLICIT

        # A global reasoning write clears the session pin; the child must not resurrect it.
        session.pop("create_reasoning_override")
        server._persist_live_session_runtime(session)
        assert "reasoning_config_override" not in agent._session_init_model_config

        self._compress(agent)

        child = db.get_session(agent.session_id)
        assert "reasoning_config_override" not in json.loads(child["model_config"])
        assert "reasoning_config_override" not in server._stored_session_runtime_overrides(child)

    @pytest.mark.parametrize("override", [EXPLICIT, None])
    def test_make_agent_seeds_marker_before_first_persist(self, override) -> None:
        """A resumed pin can hit compaction on its very first turn, before any runtime persist has run."""
        fake_agent = MagicMock()
        fake_agent._session_init_model_config = {"reasoning_config": {"enabled": True, "effort": "low"}}
        fake_cfg = {"model": {"default": "glm-5", "provider": "zai"}, "agent": {"system_prompt": "test"}}
        fake_runtime = {
            "provider": "zai", "base_url": "https://api.z.ai/v1", "api_key": "sk-test", "api_mode": "chat_completions",
            "command": None, "args": None, "credential_pool": None,
        }
        with (
            patch("tui_gateway.server._load_cfg", return_value=fake_cfg),
            patch("tui_gateway.server._get_db", return_value=MagicMock()),
            patch("tui_gateway.server._load_tool_progress_mode", return_value="compact"),
            patch("tui_gateway.server._load_reasoning_config", return_value={"enabled": True, "effort": "low"}),
            patch("tui_gateway.server._load_service_tier", return_value=None),
            patch("tui_gateway.server._load_enabled_toolsets", return_value=None),
            patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value=fake_runtime),
            patch("run_agent.AIAgent", return_value=fake_agent),
        ):
            agent = server._make_agent("sid-1", "key-1", reasoning_config_override=override)

        assert agent is fake_agent
        if override is None:
            assert "reasoning_config_override" not in fake_agent._session_init_model_config
        else:
            assert fake_agent._session_init_model_config["reasoning_config_override"] == override

