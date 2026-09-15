"""Delegation keeps late-parent and orphaned transcripts out of session pickers."""

import json
import sqlite3

import pytest

from hermes_state import SessionDB
from run_agent import AIAgent
from tools import delegate_tool


@pytest.fixture
def make_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("agent.model_metadata.fetch_model_metadata", lambda *a, **kw: {})
    monkeypatch.setattr("agent.model_metadata.get_model_context_length", lambda *a, **kw: 32768)
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda *a, **kw: [])
    monkeypatch.setattr(delegate_tool, "_load_config", lambda: {})
    monkeypatch.setattr("tools.delegate_tool_config._load_config", lambda: {})
    agents = []
    with SessionDB(db_path=tmp_path / "state.db") as db:
        def build(session_id, **kwargs):
            agent = AIAgent(
                api_key="test-key", base_url="http://127.0.0.1:1/v1", provider="openai",
                api_mode="chat_completions", model="test-model", session_id=session_id,
                session_db=db, quiet_mode=True, skip_context_files=True, skip_memory=True,
                save_trajectories=False, enabled_toolsets=[], **kwargs,
            )
            agents.append(agent)
            return agent

        yield build
        for agent in reversed(agents):
            agent.close()


@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("parent_write_recovers", [False, True])
def test_dispatch_commits_parent_before_child_construction(
    make_agent, monkeypatch, caplog, background, parent_write_recovers,
):
    parent = make_agent("late-parent", platform="cli")
    db = parent._session_db
    create_session = db.create_session
    attempts = 0

    def delayed_create(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1 or not parent_write_recovers:
            raise sqlite3.OperationalError("database is locked")
        return create_session(*args, **kwargs)

    monkeypatch.setattr(db, "create_session", delayed_create)
    parent._ensure_db_session()
    assert db.get_session(parent.session_id) is None
    caplog.clear()
    children = []
    construction_attempts = []
    build_child = delegate_tool._build_child_agent

    def observe_child(**kwargs):
        construction_attempts.append(kwargs["task_index"])
        # A separate handle proves COMMIT visibility, not just an in-transaction insert.
        with SessionDB(db_path=db.db_path) as reader:
            assert reader.get_session(parent.session_id) is not None
        child = build_child(**kwargs)
        children.append(child)
        child._ensure_db_session()
        row = child._session_db.get_session(child.session_id)
        assert row["parent_session_id"] == parent.session_id
        assert json.loads(row["model_config"])["_delegate_from"] == parent.session_id
        return child

    monkeypatch.setattr(delegate_tool, "_build_child_agent", observe_child)
    # Stop at the batch boundary: construction and SQLite are real, model execution is not.
    monkeypatch.setattr(delegate_tool, "_run_batch", lambda batch, background: '{"results": []}')
    try:
        result = json.loads(delegate_tool.delegate_task(
            tasks=[{"goal": "Inspect the first module"}, {"goal": "Inspect the second module"}],
            parent_agent=parent, background=background,
        ))
        if parent_write_recovers:
            assert "error" not in result
            assert len(children) == 2
        else:
            assert "error" in result
            assert not construction_attempts
        assert "FOREIGN KEY constraint failed" not in caplog.text
    finally:
        for child in children:
            child.close()


@pytest.mark.parametrize("has_marker", [False, True])
@pytest.mark.parametrize("platform", ["cli", "subagent"])
def test_subagent_flush_recovers_orphan_without_sidebar_leak(make_agent, has_marker, platform):
    child = make_agent("orphan-child", platform=platform, parent_session_id="missing-parent")
    if has_marker:
        child._session_init_model_config["_delegate_from"] = "missing-parent"
    initial_config = dict(child._session_init_model_config)
    messages = [{"role": "user", "content": "scratchpad task"},
                {"role": "assistant", "content": "scratchpad result"}]

    db = child._session_db
    flushed = child._flush_messages_to_session_db(messages, [])
    if platform != "subagent":
        assert flushed is False
        assert db.get_session(child.session_id) is None
        return
    assert flushed is True
    row = db.get_session(child.session_id)
    assert row["parent_session_id"] is None
    assert row["source"] == "tool"
    assert json.loads(row["model_config"]) == {**initial_config, "_delegate_from": "missing-parent"}
    assert child._session_init_model_config == initial_config
    assert [(msg["role"], msg["content"]) for msg in db.get_messages(child.session_id)] == [
        ("user", "scratchpad task"), ("assistant", "scratchpad result"),
    ]
    db.create_session("visible-parent", source="cli")
    assert {row["id"] for row in db.list_sessions_rich()} == {"visible-parent"}
    assert child.session_id in {row["id"] for row in db.list_sessions_rich(include_children=True)}
    assert child._flush_messages_to_session_db(messages, []) is True
    assert len(db.get_messages(child.session_id)) == 2
