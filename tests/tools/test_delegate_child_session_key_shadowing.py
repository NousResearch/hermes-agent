"""A delegate child must not inherit the parent's ``gateway_session_key``.

The child's own ``subagent`` session row would otherwise be the newest row for that key and shadow
the parent chat in every session_key-keyed query (``list_gateway_sessions`` behind the ``hermes status``
active count and the channel directory, ``find_latest_gateway_session_for_peer`` stale-route
recovery). User/chat attribution is still inherited.

Builds a real gateway parent AIAgent and a real delegate child (``_build_child_agent``) against a
scratch SessionDB, persists both rows the way a turn would (``_ensure_db_session``), ends the child,
and checks the parent chat is still visible.
"""

import os
import tempfile
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

SESSION_KEY = "agent:main:discord:channel:c1"
PEER = dict(source="discord", user_id="222", session_key=SESSION_KEY, chat_id="c1", chat_type="channel", thread_id=None)


@pytest.fixture
def db():
    from hermes_state import SessionDB

    with tempfile.TemporaryDirectory() as tmpdir:
        session_db = SessionDB(db_path=Path(tmpdir) / "state.db")
        try:
            yield session_db
        finally:
            session_db.close()


def _make_parent(db):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        parent = AIAgent(
            api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test/model", quiet_mode=True,
            session_db=db, session_id="parent-sess", skip_context_files=True, skip_memory=True,
            platform="discord", user_id="222", user_name="wife", chat_id="c1", chat_name="general",
            chat_type="channel", gateway_session_key=SESSION_KEY,
        )
    parent._ensure_db_session()
    return parent


def _make_child(parent):
    from tools.delegate_tool import _build_child_agent

    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        child = _build_child_agent(
            task_index=0, goal="do it", context=None, toolsets=None, model=None,
            max_iterations=5, task_count=1, parent_agent=parent,
        )
    child._ensure_db_session()
    return child


def test_child_row_inherits_identity_but_not_session_key(db):
    parent = _make_parent(db)
    child = _make_child(parent)
    try:
        child_row = db.get_session(child.session_id)
        parent_row = db.get_session(parent.session_id)
        assert parent_row["session_key"] == SESSION_KEY
        assert child_row["source"] == "subagent"
        assert child_row["parent_session_id"] == parent.session_id
        # Identity attribution is inherited, the per-chat routing key is not.
        assert child_row["user_id"] == "222" and child_row["chat_id"] == "c1"
        assert child_row["session_key"] is None, child_row
    finally:
        child.close()


def test_parent_chat_survives_child_in_session_key_queries(db):
    parent = _make_parent(db)
    child = _make_child(parent)
    try:
        # While the child runs: the parent must still be the live row for its key.
        active_ids = {row["id"] for row in db.list_gateway_sessions(active_only=True)}
        assert parent.session_id in active_ids
        assert child.session_id not in active_ids
    finally:
        child.close()
    db.end_session(child.session_id, "completed")
    # After the child ends: the parent chat is still listed as active ...
    active_ids = {row["id"] for row in db.list_gateway_sessions(active_only=True)}
    assert parent.session_id in active_ids
    # ... and stale-route recovery finds the parent, not the ended subagent row.
    found = db.find_latest_gateway_session_for_peer(**PEER)
    assert found is not None and found["id"] == parent.session_id
