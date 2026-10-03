"""`hermes --tui --reasoning LEVEL` is honored for new TUI sessions (NousResearch/hermes-agent#107780).

The launcher exports the flag as ``HERMES_TUI_REASONING``; ``session.create`` uses it as the fallback
reasoning override when the request carries no ``reasoning_effort`` of its own. It is process-scoped:
never a config write, and it never overrides a resumed session's stored reasoning.
"""

from __future__ import annotations

import logging
import types

import pytest

from hermes_state import SessionDB
from tui_gateway import server

LAUNCH_ENV = "HERMES_TUI_REASONING"


@pytest.fixture
def create(monkeypatch, tmp_path):
    """``session.create`` without an agent build; sessions created by a test are dropped afterwards."""
    db = SessionDB(db_path=tmp_path / "state.db")
    monkeypatch.delenv(LAUNCH_ENV, raising=False)
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_schedule_agent_build", lambda _sid: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda _session: None)
    created = []

    def _create(**params):
        resp = server.handle_request(
            {"id": "create", "method": "session.create", "params": {"cols": 80, **params}})
        assert "result" in resp, resp
        created.append(resp["result"]["session_id"])
        return server._sessions[resp["result"]["session_id"]]

    yield _create
    for sid in created:
        server._sessions.pop(sid, None)
    db.close()


@pytest.mark.parametrize(
    ("launch_value", "expected"),
    [
        ("medium", {"enabled": True, "effort": "medium"}),
        (" High ", {"enabled": True, "effort": "high"}),
        ("none", {"enabled": False}),
    ],
)
def test_launch_value_becomes_the_new_session_override(create, monkeypatch, launch_value, expected):
    monkeypatch.setenv(LAUNCH_ENV, launch_value)
    assert create()["create_reasoning_override"] == expected


def test_unparseable_launch_value_warns_and_yields_no_override(create, monkeypatch, caplog):
    monkeypatch.setenv(LAUNCH_ENV, "hgih")
    with caplog.at_level(logging.WARNING):
        session = create()
    assert session["create_reasoning_override"] is None
    assert "hgih" in caplog.text


@pytest.mark.parametrize("launch_value", [None, "", "   "])
def test_no_launch_value_means_no_override(create, monkeypatch, launch_value):
    if launch_value is not None:
        monkeypatch.setenv(LAUNCH_ENV, launch_value)
    assert create()["create_reasoning_override"] is None


@pytest.mark.parametrize(
    ("request_effort", "expected"),
    [("low", {"enabled": True, "effort": "low"}), ("none", {"enabled": False})],
)
def test_session_create_reasoning_beats_the_launch_value(create, monkeypatch, request_effort, expected):
    monkeypatch.setenv(LAUNCH_ENV, "high")
    assert create(reasoning_effort=request_effort)["create_reasoning_override"] == expected


def test_launch_value_covers_every_new_session_in_the_process(create, monkeypatch):
    """`/new` is another bare session.create in the same TUI process: it keeps the launch reasoning,
    paired with the launch model/provider that the same process environment already carries."""
    monkeypatch.setenv(LAUNCH_ENV, "medium")
    first, second = create(), create()
    assert first["session_key"] != second["session_key"]
    assert first["create_reasoning_override"] == second["create_reasoning_override"] == {
        "enabled": True, "effort": "medium"}


def test_launch_value_reaches_the_agent_and_leaves_config_alone(create, monkeypatch, tmp_path):
    """launch env -> session.create -> deferred-build kwargs -> AIAgent(reasoning_config=...), beating the
    configured level, with config.yaml byte-identical afterwards."""
    from hermes_constants import get_hermes_home

    config_path = get_hermes_home() / "config.yaml"
    config_path.write_text("agent:\n  reasoning_effort: low\n", encoding="utf-8")
    before = config_path.read_bytes()
    monkeypatch.setenv(LAUNCH_ENV, "medium")
    writes = []
    monkeypatch.setattr(server, "_write_config_key", lambda *a, **k: writes.append(a))

    built = {}
    monkeypatch.setattr("run_agent.AIAgent", lambda **kwargs: built.update(kwargs) or types.SimpleNamespace(**kwargs))
    monkeypatch.setattr(server, "_resolve_agent_model_runtime", lambda *_a: ("test-model", {}))
    monkeypatch.setattr(server, "_load_enabled_toolsets", lambda *_a, **_kw: None)
    monkeypatch.setattr(server, "_agent_cbs", lambda _sid: {})

    session = create()
    server._make_agent("sid", "key", **server._deferred_build_agent_kwargs(session, None))

    assert built["reasoning_config"] == {"enabled": True, "effort": "medium"}
    assert writes == []
    assert config_path.read_bytes() == before


def test_resume_keeps_stored_reasoning_when_a_launch_value_is_set(monkeypatch):
    """session.resume restores the stored runtime; the launch value only backs session.create."""
    monkeypatch.setenv(LAUNCH_ENV, "low")
    captured = {}

    class FakeDB:
        def get_session(self, target):
            return {
                "id": target, "model": "gpt-5.4", "billing_provider": "openai-codex",
                "model_config": '{"reasoning_config":{"enabled":true,"effort":"high"}}',
            }

        def reopen_session(self, target):
            pass

        def get_resume_conversations(self, session_id):
            return ([{"role": "user", "content": "hello"}],) * 2

        def get_ancestor_display_prefix(self, _sid):
            return []

        def get_messages_as_conversation(self, target, **_kwargs):
            return [{"role": "user", "content": "hello"}]

    def fake_make_agent(sid, key, session_id=None, session_db=None, **kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(model="gpt-5.4", provider="openai-codex")

    monkeypatch.setattr(server, "_get_db", lambda: FakeDB())
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    monkeypatch.setattr(server, "_set_session_context", lambda target, cwd=None: [])
    monkeypatch.setattr(server, "_clear_session_context", lambda tokens: None)
    monkeypatch.setattr(server, "_make_agent", fake_make_agent)
    monkeypatch.setattr(server, "_session_info", lambda agent, *a: {})
    monkeypatch.setattr(
        server, "_init_session",
        lambda sid, key, agent, history, cols=80, **_kwargs: server._sessions.update(
            {sid: {"agent": agent, "session_key": key}}))

    resp = server.handle_request(
        {"id": "1", "method": "session.resume", "params": {"session_id": "stored-session", "eager_build": True}})
    try:
        assert captured["reasoning_config_override"] == {"enabled": True, "effort": "high"}
    finally:
        server._sessions.pop(resp["result"]["session_id"], None)
