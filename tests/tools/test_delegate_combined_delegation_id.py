"""The settled combined result names the call's delegation id (#136532).

The desktop delegate card joins live subagent state onto a call's rows. The
background dispatch handle already names ``delegation_id``; the settled
(sync/inline) combined payload must name it too, or a card whose goal string
repeats an earlier delegation's goal falls back to a fuzzy join and re-adopts
the other call's workers.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from tools.delegate_tool_dispatch import _Batch, _execute_and_aggregate


class _CompletingChild:
    """Minimal worker double: returns a finished result immediately."""

    def __init__(self, session_id: str) -> None:
        self.tool_progress_callback = None
        self._credential_pool = None
        self._delegate_saved_tool_names = []
        self._delegate_role = "leaf"
        self._delegate_depth = 1
        self._subagent_id = None
        self.session_id = session_id

    def run_conversation(self, **_kwargs):
        return {"final_response": "DONE", "completed": True, "api_calls": 1, "messages": []}

    def get_activity_summary(self):
        return {"api_call_count": 1, "current_tool": None, "last_activity_ts": 1000.0, "max_iterations": 50}

    def hard_interrupt(self, *_args, **_kwargs):
        pass

    def close(self):
        pass


def _parent():
    return SimpleNamespace(
        session_id="parent",
        _current_task_id=None,
        _active_children=[],
        _active_children_lock=threading.Lock(),
        _touch_activity=lambda _desc: None,
        _interrupt_requested=False,
    )


def _batch(live_deleg_id):
    return _Batch(
        task_list=[{"goal": "A"}],
        children=[(0, {"goal": "A"}, _CompletingChild("child"))],
        parent_agent=_parent(),
        creds={"model": "test"},
        context=None,
        top_role="leaf",
        max_children=1,
        live_deleg_id=live_deleg_id,
        live_writers=[None],
        live_paths=[],
        origin_wake_sid="",
        origin_ui_session_id="",
        origin_owner_transport=None,
        origin_owner_session_record=None,
        origin_session_history_delivery=False,
        overall_start=time.monotonic(),
    )


def test_combined_result_names_the_live_delegation_id(monkeypatch):
    monkeypatch.setattr("tools.delegate_tool._get_child_timeout", lambda: None)
    monkeypatch.setattr("tools.delegate_tool._get_worktree_isolation", lambda: False)

    combined = _execute_and_aggregate(_batch("deleg_join1"))

    assert combined["delegation_id"] == "deleg_join1"


def test_combined_result_omits_the_key_when_live_transcripts_were_unavailable(monkeypatch):
    monkeypatch.setattr("tools.delegate_tool._get_child_timeout", lambda: None)
    monkeypatch.setattr("tools.delegate_tool._get_worktree_isolation", lambda: False)

    combined = _execute_and_aggregate(_batch(None))

    assert "delegation_id" not in combined
