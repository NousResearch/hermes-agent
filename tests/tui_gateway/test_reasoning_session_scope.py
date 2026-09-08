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


@pytest.fixture
def reasoning_factory(tmp_path, monkeypatch):
    """Real profile loader, factory and DB; no provider request is needed to build."""
    import yaml
    from hermes_state import SessionDB

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_IGNORE_RULES", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:1/v1")
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({
        "model": {"default": "review-model", "provider": "custom:fixture",
                  "base_url": "http://127.0.0.1:1/v1"},
        "agent": {"reasoning_effort": "medium",
                  "reasoning_overrides": {"review-model": "high"},
                  "adaptive_reasoning": {"enabled": True, "min_effort": "low"}},
        "custom_providers": [{"name": "fixture", "base_url": "http://127.0.0.1:1/v1",
                              "api_key": "test-only", "model": "review-model"}],
        "toolsets": {"desktop": []},
    }))
    db = SessionDB(db_path=tmp_path / "fixture.db")
    session = {
        "agent": None, "session_key": "reasoning-fixture", "source": "desktop",
        "model_override": {"model": "review-model", "provider": "custom:fixture",
                           "base_url": "http://127.0.0.1:1/v1",
                           "api_key": "test-only", "api_mode": "chat_completions"},
    }
    yield session, db
    db.close()


@pytest.mark.parametrize("level,explicit", [("high", True), ("medium", True)])
def test_explicit_reasoning_survives_deferred_and_stored_build(reasoning_factory, monkeypatch, level, explicit):
    import io
    import threading
    import json
    from tui_gateway.compute_host import ComputeHost
    from agent.adaptive_reasoning import adaptive_reasoning_turn

    session, db = reasoning_factory
    sid = session["session_key"]
    response = server._set_reasoning("pin", {"scope": "session"}, "reasoning", level, session)
    assert "error" not in response
    agent = server._make_agent(sid, sid, **server._deferred_build_agent_kwargs(session, db))
    assert agent.reasoning_user_override is explicit
    with adaptive_reasoning_turn(agent, "thanks"):
        assert agent.reasoning_config["effort"] == level
    # Persist and read the actual row, then restore through the same resume factory seam.
    db.create_session(sid, source="desktop", model=agent.model)
    session["agent"] = agent
    server._persist_live_session_runtime(session)
    row = db.get_session(sid)
    assert json.loads(row["model_config"])["reasoning_user_override"] is explicit
    restored = server._stored_session_runtime_overrides(row)
    rebuilt = server._make_agent(sid, sid, session_db=db, **restored)
    assert rebuilt.reasoning_user_override is explicit
    with adaptive_reasoning_turn(rebuilt, "thanks"):
        assert rebuilt.reasoning_config["effort"] == level

    # Cold resume -> compute transport -> real factory must retain the same pin.
    session.update(agent=None, resume_runtime_overrides=restored, history_lock=threading.Lock())
    session.pop("reasoning_user_override")
    session.pop("create_reasoning_override")
    assert server._overrides_have_routable_provider(restored)
    frame = server._compute_host_turn_frame("turn", sid, session, "thanks")
    assert frame["reasoning_user_override"] is True
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_start_session_services", lambda *a: None)
    monkeypatch.setattr(server, "_schedule_mcp_late_refresh", lambda *a: None)
    monkeypatch.setattr(server, "_emit", lambda *a, **kw: None)
    host = ComputeHost(stdout=io.StringIO(), heartbeat_secs=0)
    try:
        hosted = host._build_server_session(server, frame, sid)
        assert hosted["agent"].reasoning_user_override is True
        replacement = server._rebuild_session_agent(sid, hosted)
        with adaptive_reasoning_turn(replacement, "thanks"):
            assert replacement.reasoning_config["effort"] == level
        # /new also discards the cold-resume snapshot so another deferred build
        # cannot resurrect the old pin after the live agent was reset.
        hosted["resume_runtime_overrides"] = restored.copy()
        server._reset_session_agent(sid, hosted)
        assert not hosted["agent"].reasoning_user_override
        assert "reasoning_user_override" not in hosted
        later = server._make_agent(sid, sid, **server._deferred_build_agent_kwargs(hosted, db))
        assert not later.reasoning_user_override
    finally:
        server._sessions.pop(sid, None)
        host.close()

    # A session pick after lazy resume beats the stored snapshot; --global clears both.
    server._set_reasoning("repin", {"scope": "session"}, "reasoning", "xhigh", session)
    fresh = server._make_agent(sid, sid, **server._deferred_build_agent_kwargs(session, db))
    assert fresh.reasoning_config["effort"] == "xhigh" and fresh.reasoning_user_override
    response = server._set_reasoning("global", {"scope": "global"}, "reasoning", "medium", session)
    assert "error" not in response
    fresh = server._make_agent(sid, sid, **server._deferred_build_agent_kwargs(session, db))
    assert not fresh.reasoning_user_override


@pytest.mark.parametrize("level,explicit", [("high", True), ("medium", False), (None, False)])
def test_desktop_factory_compares_composer_global_not_model_default(reasoning_factory, level, explicit):
    from hermes_constants import parse_reasoning_effort
    from agent.adaptive_reasoning import adaptive_reasoning_turn

    session, db = reasoning_factory
    if level:
        session["create_reasoning_override"] = parse_reasoning_effort(level)
    agent = server._make_agent("desktop", "desktop", **server._deferred_build_agent_kwargs(session, db))
    assert agent.reasoning_user_override is explicit
    if level:
        _, row_config = server._workdir_row_model_config(session)
        assert row_config["reasoning_user_override"] is explicit
    with adaptive_reasoning_turn(agent, "thanks"):
        assert agent.reasoning_config["effort"] == (level if explicit else "low")
    # False provenance must survive even though a model baseline differs from the global default.
    stored = server._runtime_model_config(agent)
    restored = server._stored_session_runtime_overrides({"model_config": stored, "model": agent.model})
    again = server._make_agent("desktop", "desktop", session_db=db, **restored)
    assert again.reasoning_user_override is explicit


@pytest.mark.parametrize("reasoning", [None, {}, {"enabled": False}, {"effort": "high"}])
@pytest.mark.parametrize("pinned", [False, True])
def test_runtime_reasoning_provenance_requires_a_current_config(reasoning, pinned):
    agent = _agent(reasoning)
    agent.reasoning_user_override = pinned
    stored = server._runtime_model_config(agent, {
        "reasoning_config": {"effort": "xhigh"}, "reasoning_user_override": True,
    })
    restored = server._stored_session_runtime_overrides({"model_config": stored})
    if reasoning is None:
        assert "reasoning_config" not in stored
        assert "reasoning_user_override" not in stored
        assert "reasoning_config_override" not in restored
        assert "reasoning_user_override" not in restored
    else:
        assert stored["reasoning_config"] == reasoning
        assert stored["reasoning_user_override"] is pinned
        assert restored["reasoning_config_override"] == reasoning
        assert restored["reasoning_user_override"] is pinned


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

