"""DelegateEvent.TASK_DIAGNOSTIC — visibility for child stale-kill / diagnostic chatter.

Three behaviour contracts:
1. Event kind round-trips construction + serialisation of the payload fields.
2. Tee (not redirect): a child that hits the non-stream stale-kill path twice delivers
   two TASK_DIAGNOSTIC events to the parent's progress callback (attempt 1 and 2).
3. A stale-kill retry is not progress: a child stuck in that loop trips the parent
   heartbeat idle threshold, not the giveup ceiling.
"""
from __future__ import annotations

import sys
import threading
import time
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

from agent.chat_completion_helpers import direct_api_call
from agent.status_output import StatusOutputMixin
from tools.delegate_tool_progress import DelegateEvent, _build_child_progress_callback
from tools import delegate_tool


# ---------------------------------------------------------------------------
# 1. Event kind round-trip
# ---------------------------------------------------------------------------

def test_task_diagnostic_event_round_trips_fields():
    """Constructing/serialising TASK_DIAGNOSTIC preserves the contract fields."""
    assert DelegateEvent.TASK_DIAGNOSTIC.value == "delegate.task_diagnostic"
    assert str(DelegateEvent.TASK_DIAGNOSTIC) != DelegateEvent.TASK_DIAGNOSTIC.value  # wire ≠ str(enum)

    payload = {
        "subagent_id": "sa-1",
        "task_index": 2,
        "text": "No response from provider for 12s",
        "api_call": 7,
        "attempt": 3,
        "giveup": 5,
    }
    # Serialise the way progress callbacks do: event wire string + kwargs.
    event = {"event_type": DelegateEvent.TASK_DIAGNOSTIC.value, **payload}
    restored = dict(event)
    assert restored["event_type"] == "delegate.task_diagnostic"
    for key in ("subagent_id", "task_index", "text", "api_call", "attempt", "giveup"):
        assert restored[key] == payload[key]

    # Enum reconstructs from the wire string (DelegateEvent is a str Enum).
    assert DelegateEvent(restored["event_type"]) is DelegateEvent.TASK_DIAGNOSTIC


# ---------------------------------------------------------------------------
# 2. Tee: two stale kills → two TASK_DIAGNOSTIC on the parent callback
# ---------------------------------------------------------------------------

def _stalling_child(*, parent_cb, subagent_id="sa-diag", api_call=4):
    """Child agent with real StatusOutputMixin + relay progress callback."""
    parent = SimpleNamespace(
        tool_progress_callback=parent_cb,
        status_callback=None,
        _delegate_spinner=None,
        platform="cli",
        _notification_platform="cli",
        _notification_config=None,
    )
    relay = _build_child_progress_callback(
        0, "slow goal", parent, 1, subagent_id=subagent_id,
    )

    class Child(StatusOutputMixin):
        def __init__(self):
            self.quiet_mode = True
            self.tool_progress_callback = relay
            self.status_callback = None
            self.platform = "subagent"
            self._notification_platform = "cli"
            self._notification_config = None
            self.log_prefix = ""
            self._print_fn = None
            self.suppress_status_output = True
            self._mute_notification_reply = False
            self._mute_post_response = False
            self._executing_tools = False
            self._retry_status_buffer = []
            self._pending_fallback_notice = None
            self._interrupt_requested = False
            self._consecutive_stale_streams = 0
            self.provider = "openrouter"
            self.api_mode = "chat_completions"
            self.model = "slow-model"
            self.api_call_count = api_call
            self._subagent_id = subagent_id
            self._delegate_depth = 1
            self._release = threading.Event()
            self._aborted = []
            self._last_activity_ts = time.time()

        def _has_stream_consumers(self):
            return False

        def _touch_activity(self, msg, provenance=None):
            self._last_activity_ts = time.time()

        def _create_request_openai_client(self, reason=None, api_kwargs=None):
            return self._client

        def _close_request_openai_client(self, client=None, reason=None):
            pass

        def _abort_request_openai_client(self, client, reason=None):
            self._aborted.append(reason)
            self._release.set()

        def _compute_non_stream_stale_timeout(self, api_payload):
            return 0.2

        def get_activity_summary(self):
            return {
                "api_call_count": self.api_call_count,
                "current_tool": None,
                "last_activity_ts": self._last_activity_ts,
                "max_iterations": 50,
            }

    child = Child()
    client = MagicMock()

    def stall(**_kwargs):
        if not child._release.wait(timeout=5.0):
            raise AssertionError("watchdog never aborted")
        # Reset release for a second attempt on the same child.
        child._release.clear()
        raise ConnectionError("socket shut down")

    client.chat.completions.create.side_effect = stall
    child._client = client
    return child


def test_stale_kill_tees_task_diagnostic_to_parent_progress(monkeypatch):
    """Two consecutive stale kills deliver attempt=1 and attempt=2 diagnostics."""
    monkeypatch.setenv("HERMES_STREAM_STALE_GIVEUP", "5")
    received = []

    def parent_cb(event_type, tool_name=None, preview=None, args=None, **kwargs):
        received.append({"event_type": event_type, "preview": preview, **kwargs})

    child = _stalling_child(parent_cb=parent_cb)

    for _ in range(2):
        with pytest.raises(TimeoutError):
            direct_api_call(child, {"model": "slow-model", "messages": [{"role": "user", "content": "hi"}]})

    diags = [e for e in received if e["event_type"] == DelegateEvent.TASK_DIAGNOSTIC.value]
    assert len(diags) == 2, f"expected 2 TASK_DIAGNOSTIC, got {received!r}"
    attempts = [d.get("attempt") for d in diags]
    assert attempts == [1, 2], attempts
    for d in diags:
        assert d.get("subagent_id") == "sa-diag"
        assert d.get("task_index") == 0
        assert "text" in d and d["text"]
        assert d.get("giveup") == 5
        assert d.get("api_call") == 4
    # Buffer still holds (tee, not redirect).
    assert child._retry_status_buffer, "retry buffer must still hold the diagnostics"


# ---------------------------------------------------------------------------
# 3. Stale-kill retry is not progress → idle threshold trips
# ---------------------------------------------------------------------------

def test_stale_kill_loop_trips_idle_heartbeat_not_giveup(monkeypatch):
    """A child whose activity clock freezes after the first stale kill goes stale
    at the idle threshold (2 fast cycles here), not after exhausting giveup."""
    from tools.delegate_tool import _run_single_child

    parent = SimpleNamespace(
        session_id="parent",
        _current_task_id=None,
        _active_children=[],
        _active_children_lock=threading.Lock(),
        _touch_activity=lambda _d: None,
        _interrupt_requested=False,
    )

    # Frozen activity clock: models a child whose post-kill retries no longer
    # refresh last_activity_ts (the production contract of direct_api_call on
    # stale_retry). api_call_count and current_tool stay fixed too.
    frozen_ts = 1000.0
    child = MagicMock()
    child.tool_progress_callback = None
    child._credential_pool = None
    child._delegate_saved_tool_names = []
    child._delegate_role = "leaf"
    child._delegate_depth = 1
    child._subagent_id = None
    child.session_id = "stale-loop-child"
    child.model = "m"
    child.get_activity_summary.return_value = {
        "api_call_count": 1,
        "current_tool": None,
        "last_activity_ts": frozen_ts,
        "max_iterations": 50,
        "last_activity_desc": "waiting for non-streaming API response",
    }
    settled = threading.Event()

    def slow_run(**_kwargs):
        # Stay alive until the heartbeat abandons us (or timeout).
        settled.wait(5.0)
        return {"final_response": "should not finish", "completed": True, "api_calls": 1}

    child.run_conversation.side_effect = slow_run
    child.hard_interrupt = lambda *_a, **_k: settled.set()
    child.close = lambda: None

    # 2 idle cycles × 0.05s interval = 0.1s idle threshold — far below any giveup ceiling.
    with (
        pytest.MonkeyPatch.context() as mp,
    ):
        mp.setattr(delegate_tool, "_HEARTBEAT_INTERVAL", 0.05)
        mp.setattr(delegate_tool, "_HEARTBEAT_STALE_CYCLES_IDLE", 2)
        mp.setattr(delegate_tool, "_HEARTBEAT_STALE_CYCLES_IN_TOOL", 40)
        mp.setattr(delegate_tool, "_get_child_timeout", lambda: None)
        mp.setattr(delegate_tool, "_get_worktree_isolation", lambda: False)
        t0 = time.time()
        entry = _run_single_child(
            task_index=0, goal="stale-kill loop", child=child, parent_agent=parent,
        )
        elapsed = time.time() - t0

    # Abandoned for heartbeat-stale / timeout shape, not a clean completion.
    assert entry.get("status") in {"timeout", "error", "failed", "interrupted"} or entry.get("exit_reason") in {
        "timeout", "error", "interrupted", "stale",
    }, entry
    # Must trip well under a multi-minute giveup ceiling (5 × 300s).
    assert elapsed < 3.0, f"idle threshold did not trip promptly ({elapsed:.2f}s): {entry}"
