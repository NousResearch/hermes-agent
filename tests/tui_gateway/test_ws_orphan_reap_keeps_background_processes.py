"""A WS-orphan reap must not kill background work its session started (#41225, #120648).

Real chain, no mocked registry: a real ``sleep`` spawned through ``process_registry.spawn_local``
is owned by a real ``AIAgent`` whose desktop session is torn down exactly as the WS-orphan reaper
does it — ``_teardown_popped_session(session, end_reason="ws_orphan_reap")``. That reaper treats
the conversation as recoverable (the row is not ended), so the background job must still be running
afterwards. A deliberate close (``tui_close``) must keep killing it.
"""

from __future__ import annotations

import threading
import time

import pytest

from hermes_state import SessionDB
from tools.process_registry import process_registry
from tui_gateway import server


def _agent(db: SessionDB, session_id: str, owner: str):
    from unittest.mock import patch

    with patch("run_agent.AIAgent.__init__", return_value=None):
        from run_agent import AIAgent

        agent = AIAgent.__new__(AIAgent)
    agent.session_id = session_id
    agent._session_db = db
    agent._process_owner_task_ids = {owner}
    agent._active_children = []
    agent._active_children_lock = threading.Lock()
    agent.client = None
    agent._end_session_on_close = True
    return agent


def _session(agent, sid: str, session_id: str, source: str) -> dict:
    return {
        "_sid": sid, "session_key": session_id, "agent": agent, "history": [],
        "history_lock": threading.RLock(), "source": source,
    }


@pytest.fixture
def db(tmp_path, monkeypatch):
    handle = SessionDB(db_path=tmp_path / "state.db")
    # One real store for both halves: the gateway's finalize (via _session_db -> _get_db) and the agent's close.
    monkeypatch.setattr(server, "_get_db", lambda: handle)
    yield handle
    handle.close()


# desktop: automatic cleanup hands the row forward (_end_session_on_close=False) and never ends it.
# tui: the gateway ends the row itself with the automatic ``ws_orphan_reap`` stamp before agent.close().
@pytest.mark.parametrize("source", ["desktop", "tui"])
@pytest.mark.parametrize(("end_reason", "survives"), [("ws_orphan_reap", True), ("tui_close", False)])
def test_orphan_reap_keeps_background_work_but_deliberate_close_kills_it(db, source, end_reason, survives):
    session_id = f"reap-{source}-{end_reason}"
    owner = f"sa-{source}-{end_reason}"
    db.create_session(session_id, source=source, model="m")
    proc = process_registry.spawn_local("exec sleep 120", task_id=owner, owner_task_id=owner)
    try:
        agent = _agent(db, session_id, owner)
        server._teardown_popped_session(_session(agent, f"ui-{source}-{end_reason}", session_id, source), end_reason=end_reason)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and process_registry.get(proc.id).exited == survives:
            time.sleep(0.05)
        assert (not process_registry.get(proc.id).exited) is survives
    finally:
        process_registry.kill_process(proc.id, source="test_cleanup")
