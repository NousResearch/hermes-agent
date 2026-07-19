"""Ephemeral TUI agents must clear their per-task file-tool trackers.

`prompt.background` (bg_<uuid>) and `preview.restart` (preview_<uuid>) each spawn
a throwaway AIAgent under a unique task_id and used to clear only session
context. Ordinary turn cleanup frees just VM/browser state, so a file-tool call
inside either path left `_read_tracker[task_id]` et al. pinned for the process
lifetime — the exact leak clear_task_trackers() targets (PR #62934 review
follow-up).
"""
from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

import tools.file_tools as ft
from tui_gateway import server


class _SyncThread:
    """Run target() inline on start() so the finally block runs before the
    method returns — no daemon-thread join race in the test."""

    def __init__(self, target=None, daemon=None, **_kw):
        self._target = target

    def start(self):
        if self._target:
            self._target()


def _make_touching_agent(seen: dict):
    """Factory standing in for AIAgent: its run_conversation simulates a
    file-tool call by seeding every per-task tracker for the run's task_id,
    and records that the entry was present mid-run."""

    def factory(**_kwargs):
        agent = MagicMock()

        def _run(**kw):
            tid = kw["task_id"]
            with ft._read_tracker_lock:
                ft._read_tracker[tid] = {"touched": True}
            with ft._patch_failure_lock:
                ft._patch_failure_tracker[tid] = {"n": 1}
            with ft._file_ops_lock:
                ft._last_known_cwd[tid] = "/x"
                ft._file_ops_cache[tid] = {"op": 1}
            seen["tid"] = tid
            seen["present_during_run"] = tid in ft._read_tracker
            return {"final_response": "done"}

        agent.run_conversation.side_effect = _run
        return agent

    return factory


def _assert_all_trackers_absent(tid: str):
    assert tid not in ft._read_tracker
    assert tid not in ft._patch_failure_tracker
    assert tid not in ft._last_known_cwd
    assert tid not in ft._file_ops_cache


@pytest.fixture()
def registered_session(monkeypatch):
    sid = "sid-ephemeral"
    server._sessions[sid] = {
        "agent": MagicMock(),
        "history": [],
        "history_lock": threading.Lock(),
    }
    monkeypatch.setattr(server.threading, "Thread", _SyncThread)
    monkeypatch.setattr(server, "_emit", MagicMock())
    monkeypatch.setattr(server, "_session_cwd", lambda s: "")
    try:
        yield sid
    finally:
        server._sessions.pop(sid, None)


def test_prompt_background_clears_task_trackers(registered_session, monkeypatch):
    sid = registered_session
    seen: dict = {}
    monkeypatch.setattr(server, "_background_agent_kwargs", lambda agent, tid: {})
    monkeypatch.setattr("run_agent.AIAgent", _make_touching_agent(seen))

    resp = server._methods["prompt.background"](
        "r1", {"session_id": sid, "text": "do a thing"}
    )
    task_id = resp["result"]["task_id"]

    assert task_id.startswith("bg_")
    assert seen["present_during_run"] is True  # the run actually seeded trackers
    _assert_all_trackers_absent(task_id)  # …and the finally cleared them


def test_preview_restart_clears_task_trackers(registered_session, monkeypatch):
    sid = registered_session
    seen: dict = {}
    monkeypatch.setattr(server, "_ephemeral_preview_agent_kwargs", lambda agent, tid: {})
    monkeypatch.setattr(server, "_preview_restart_callbacks", lambda parent, tid: {})
    monkeypatch.setattr(server, "_preview_restart_history", lambda session: [])
    monkeypatch.setattr("run_agent.AIAgent", _make_touching_agent(seen))

    resp = server._methods["preview.restart"](
        "r2", {"session_id": sid, "url": "http://localhost:3000"}
    )
    task_id = resp["result"]["task_id"]

    assert task_id.startswith("preview_")
    assert seen["present_during_run"] is True
    _assert_all_trackers_absent(task_id)
