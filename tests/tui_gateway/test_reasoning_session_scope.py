"""Reasoning-effort session scoping in the TUI gateway (desktop backend)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

import tui_gateway.server as server
from tui_gateway.server import _session_info


@pytest.fixture
def isolated_server(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "agent:\n  reasoning_effort: medium\ndisplay:\n  show_reasoning: false\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: None)
    server._cfg_cache = None
    server._cfg_mtime = 0
    server._sessions.clear()
    yield server, config_path
    server._sessions.clear()
    server._cfg_cache = None
    server._cfg_mtime = 0


def _agent(reasoning_config):
    return SimpleNamespace(
        reasoning_config=reasoning_config,
        service_tier=None,
        model="glm-5",
        provider="zai",
        session_id="sess-key",
    )


def _set_reasoning(params: dict) -> dict:
    return server._methods["config.set"]("rid-1", params)


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

    def test_session_scoped_set_skips_global_write(self, isolated_server) -> None:
        _mod, config_path = isolated_server
        agent = _agent({"enabled": True, "effort": "medium"})
        session = {"session_key": "k1", "agent": agent, "running": False}
        with patch.object(server, "_persist_live_session_runtime"):
            server._sessions["s1"] = session
            resp = _set_reasoning({"key": "reasoning", "session_id": "s1", "value": "high"})
            got = server._methods["config.get"]("rid-2", {"key": "reasoning", "session_id": "s1"})
        assert resp["result"] == {"key": "reasoning", "value": "high", "scope": "session"}
        assert agent.reasoning_config == {"enabled": True, "effort": "high"}
        assert session["create_reasoning_override"] == {"enabled": True, "effort": "high"}
        assert session["reasoning_config_override"] == {"enabled": True, "effort": "high"}
        assert yaml.safe_load(config_path.read_text(encoding="utf-8"))["agent"]["reasoning_effort"] == "medium"
        assert got["result"]["value"] == "high"

    def test_session_scoped_set_updates_create_override_for_lazy_session(self) -> None:
        """A pre-build session must keep the change for the deferred agent build."""
        session = {"session_key": "k2", "agent": None, "running": False}
        with patch.dict(server._sessions, {"s2": session}, clear=False), \
                patch.object(server, "_write_config_key") as write_key:
            resp = _set_reasoning({"key": "reasoning", "session_id": "s2", "value": "high"})
        assert resp["result"] == {"key": "reasoning", "value": "high", "scope": "session"}
        assert session["create_reasoning_override"] == {"enabled": True, "effort": "high"}
        assert session["reasoning_config_override"] == {"enabled": True, "effort": "high"}
        write_key.assert_not_called()

    def test_global_flag_persists_and_clears_session_override(self, isolated_server) -> None:
        _mod, config_path = isolated_server
        agent = _agent({"enabled": True, "effort": "high"})
        session = {
            "session_key": "k1",
            "agent": agent,
            "create_reasoning_override": {"enabled": True, "effort": "high"},
            "reasoning_config_override": {"enabled": True, "effort": "high"},
            "running": False,
        }
        server._sessions["s1"] = session

        with patch.object(server, "_persist_live_session_runtime"):
            resp = _set_reasoning({"key": "reasoning", "session_id": "s1", "value": "low --global"})

        assert resp["result"] == {"key": "reasoning", "value": "low", "scope": "global"}
        assert agent.reasoning_config == {"enabled": True, "effort": "low"}
        assert session["create_reasoning_override"] is None
        assert session["reasoning_config_override"] is None
        assert yaml.safe_load(config_path.read_text(encoding="utf-8"))["agent"]["reasoning_effort"] == "low"

    def test_no_session_persists_globally(self) -> None:
        with patch.object(server, "_write_config_key") as write_key:
            resp = _set_reasoning({"key": "reasoning", "value": "low"})
        assert resp["result"] == {"key": "reasoning", "value": "low", "scope": "global"}
        write_key.assert_called_once_with("agent.reasoning_effort", "low")

    def test_session_scoped_set_rejects_busy_session(self) -> None:
        session = {"session_key": "k1", "agent": _agent(None), "running": True}
        with patch.dict(server._sessions, {"s1": session}, clear=False), \
                patch.object(server, "_write_config_key") as write_key:
            resp = _set_reasoning({"key": "reasoning", "session_id": "s1", "value": "high"})
        assert resp["error"]["code"] == 4009
        write_key.assert_not_called()

    def test_unknown_value_rejected(self) -> None:
        resp = _set_reasoning({"key": "reasoning", "value": "bogus"})
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
