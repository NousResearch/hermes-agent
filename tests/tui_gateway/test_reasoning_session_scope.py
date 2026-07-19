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

    def test_session_scoped_set_updates_create_override_for_lazy_session(self) -> None:
        """A pre-build (agent=None) session must keep the change for the
        deferred agent build instead of dropping it."""
        session = {"session_key": "k2", "agent": None}
        with patch.dict(server._sessions, {"s2": session}, clear=False), \
                patch.object(server, "_write_config_key") as write_key:
            resp = self._dispatch(
                {"key": "reasoning", "session_id": "s2", "value": "high"}
            )
        assert resp["result"]["value"] == "high"
        assert session["create_reasoning_override"] == {
            "enabled": True,
            "effort": "high",
        }
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

    def test_unset_returns_default(self) -> None:
        with patch.object(server, "_load_cfg", return_value={"agent": {}}):
            assert server._load_reasoning_config() is None


class TestReasoningUpdateCallback:
    """The reasoning_effort tool's platform hook on the TUI/desktop surface.

    The agent's in-process ``reasoning_config`` mutation does not survive a
    session rebuild (resume / deferred build re-applies the resolved config),
    so the callback must land session scope on ``create_reasoning_override``
    and route ``persist=True`` to config.yaml — mirroring the session-scoped
    ``config.set key=reasoning`` handler above.
    """

    def _session(self, sid: str, agent=None) -> dict:
        return {"session_key": f"key-{sid}", "agent": agent}

    def test_session_scope_lands_on_create_override(self) -> None:
        session = self._session("s1")
        with patch.dict(server._sessions, {"s1": session}, clear=False), \
                patch.object(server, "_write_config_key") as write_key:
            persisted = server._make_reasoning_update_callback("s1")(
                "high", {"enabled": True, "effort": "high"}, False
            )
        assert persisted is False
        assert session["create_reasoning_override"] == {
            "enabled": True,
            "effort": "high",
        }
        write_key.assert_not_called()

    def test_persist_writes_config_and_clears_override(self) -> None:
        session = self._session("s2")
        session["create_reasoning_override"] = {"enabled": True, "effort": "low"}
        with patch.dict(server._sessions, {"s2": session}, clear=False), \
                patch.object(server, "_write_config_key") as write_key:
            persisted = server._make_reasoning_update_callback("s2")(
                "high", {"enabled": True, "effort": "high"}, True
            )
        assert persisted is True
        write_key.assert_called_once_with("agent.reasoning_effort", "high")
        assert "create_reasoning_override" not in session

    def test_persist_failure_falls_back_to_session_override(self) -> None:
        session = self._session("s3")
        with patch.dict(server._sessions, {"s3": session}, clear=False), \
                patch.object(
                    server, "_write_config_key", side_effect=OSError("disk full")
                ):
            persisted = server._make_reasoning_update_callback("s3")(
                "high", {"enabled": True, "effort": "high"}, True
            )
        assert persisted is False
        assert session["create_reasoning_override"] == {
            "enabled": True,
            "effort": "high",
        }

    def test_live_agent_gets_config_and_footer_update(self) -> None:
        agent = _agent(None)
        session = self._session("s4", agent=agent)
        with patch.dict(server._sessions, {"s4": session}, clear=False), \
                patch.object(server, "_persist_live_session_runtime") as persist_rt, \
                patch.object(server, "_emit") as emit:
            server._make_reasoning_update_callback("s4")(
                "xhigh", {"enabled": True, "effort": "xhigh"}, False
            )
        assert agent.reasoning_config == {"enabled": True, "effort": "xhigh"}
        persist_rt.assert_called_once_with(session)
        assert emit.call_args.args[0] == "session.info"

    def test_unknown_session_is_noop_but_persist_still_works(self) -> None:
        with patch.object(server, "_write_config_key") as write_key:
            persisted = server._make_reasoning_update_callback("gone")(
                "low", {"enabled": True, "effort": "low"}, True
            )
        assert persisted is True
        write_key.assert_called_once_with("agent.reasoning_effort", "low")


class TestPersistTrueEndToEnd:
    """persist=true from the reasoning_effort tool must reach config.yaml.

    Drives the REAL agent-side dispatch (``AIAgent._apply_reasoning_effort``)
    through the REAL TUI callback and the REAL ``_write_config_key`` /
    ``_save_cfg`` path against a temp HERMES home — the exact surface the
    #63316 review flagged as unwired.
    """

    def test_persist_true_saves_tui_default(self, tmp_path, monkeypatch) -> None:
        import yaml

        home = tmp_path / "hermes"
        home.mkdir()
        (home / "config.yaml").write_text(
            "agent:\n  reasoning_effort: medium\n", encoding="utf-8"
        )
        monkeypatch.setattr(server, "_hermes_home", home)
        # Invalidate the module-level config cache (keyed on path, but keep
        # this test independent of prior cache state).
        monkeypatch.setattr(server, "_cfg_cache", None)
        monkeypatch.setattr(server, "_cfg_mtime", None)
        monkeypatch.setattr(server, "_cfg_path", None)

        tool_defs = [
            {
                "type": "function",
                "function": {
                    "name": "reasoning_effort",
                    "description": "reasoning_effort tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        with (
            patch("run_agent.get_tool_definitions", return_value=tool_defs),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
        ):
            from run_agent import AIAgent

            agent = AIAgent(
                api_key="test-key-1234567890",
                base_url="https://openrouter.ai/api/v1",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                reasoning_config={"enabled": True, "effort": "medium"},
                reasoning_update_callback=server._make_reasoning_update_callback(
                    "sid-e2e"
                ),
            )

        session = {"session_key": "key-e2e", "agent": agent}
        session["create_reasoning_override"] = {"enabled": True, "effort": "medium"}
        with patch.dict(server._sessions, {"sid-e2e": session}, clear=False), \
                patch.object(server, "_persist_live_session_runtime"), \
                patch.object(server, "_emit"):
            import json as _json

            result = _json.loads(
                agent._apply_reasoning_effort({"level": "high", "persist": True})
            )

        assert result["success"] is True
        assert result["persisted"] is True
        assert agent.reasoning_config == {"enabled": True, "effort": "high"}
        # The durable TUI/desktop default actually landed on disk.
        saved = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
        assert saved["agent"]["reasoning_effort"] == "high"
        # Session override cleared — the global value now owns rebuilds.
        assert "create_reasoning_override" not in session


class TestMakeAgentWiresReasoningCallback:
    """_make_agent must construct agents with the reasoning_update_callback —
    the exact gap flagged in the #63316 review (toolsets.py thread)."""

    def test_make_agent_passes_callback(self) -> None:
        from unittest.mock import MagicMock

        fake_runtime = {
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "sk-test",
            "api_mode": "chat_completions",
            "command": None,
            "args": None,
            "credential_pool": None,
        }
        with (
            patch.object(server, "_load_cfg", return_value={"agent": {}}),
            patch.object(server, "_get_db", return_value=MagicMock()),
            patch.object(server, "_load_tool_progress_mode", return_value="compact"),
            patch.object(server, "_load_reasoning_config", return_value=None),
            patch.object(server, "_load_service_tier", return_value=None),
            patch.object(server, "_load_enabled_toolsets", return_value=None),
            patch(
                "hermes_cli.runtime_provider.resolve_runtime_provider",
                return_value=fake_runtime,
            ),
            patch("run_agent.AIAgent") as mock_agent,
        ):
            server._make_agent("sid-cb", "key-cb")

        cb = mock_agent.call_args.kwargs.get("reasoning_update_callback")
        assert callable(cb)
