"""A detached idle session with a queued prompt survives the WS orphan reaper until the queue drains.

The turn that ended drains its queue at the idle boundary, after post-turn housekeeping. A sender that
submitted with ``queued: true`` and disconnected on the ack (``scripts/dashboard_dispatch.py``) has no
transport left, so reaping in that window silently dropped the prompt.
"""

from __future__ import annotations

import importlib
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def server(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    with patch.dict("sys.modules", {"hermes_cli.env_loader": MagicMock(), "hermes_cli.banner": MagicMock()}):
        mod = importlib.import_module("tui_gateway.server")
        yield mod
        mod._sessions.clear()
        mod._pending_ws_reaps.clear()


class _Timer:
    created: list = []

    def __init__(self, delay, callback):
        self.delay, self.callback = delay, callback
        _Timer.created.append(self)

    def start(self):
        pass

    def cancel(self):
        pass


@pytest.fixture()
def detached(server, monkeypatch):
    _Timer.created = []
    monkeypatch.setattr(server.threading, "Timer", _Timer)
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 20)
    monkeypatch.setattr(server, "_session_has_active_delegations", lambda *a: False)
    monkeypatch.setattr(server, "_session_owns_live_wakeup_schedule", lambda *a: False)
    popped = []
    monkeypatch.setattr(server, "_teardown_popped_session", lambda s, **kw: popped.append(s))
    sid = "sid-detached-queue"
    session = {
        "session_key": "queued-session",
        "transport": server._detached_ws_transport,
        "running": False,
        "history_lock": threading.Lock(),
        "last_active": 0.0,
        "created_at": 0.0,
    }
    server._sessions[sid] = session
    return sid, session, popped


def _fire(server, sid):
    if sid not in server._pending_ws_reaps:
        server._schedule_ws_orphan_reap(sid)
    _Timer.created[-1].callback()


def test_queued_prompt_pins_the_detached_idle_session(server, detached):
    sid, session, popped = detached
    session["queued_prompt"] = {"text": "mail from Home", "transport": None}

    _fire(server, sid)

    assert server._sessions.get(sid) is session and not popped
    assert server._pending_ws_reaps[sid] is _Timer.created[-1]  # re-armed, re-checked each grace


def test_session_is_reaped_once_the_queue_drains(server, detached):
    sid, session, popped = detached
    session["queued_prompt"] = {"text": "mail from Home", "transport": None}
    _fire(server, sid)
    session.pop("queued_prompt")  # drained: the turn ran and settled

    _fire(server, sid)

    assert sid not in server._sessions and popped == [session]


def test_a_queue_that_never_drains_is_reaped_after_the_budget(server, detached, monkeypatch):
    sid, session, popped = detached
    monkeypatch.setattr(server, "_WS_ORPHAN_QUEUED_MAX_POLLS", 3)
    session["queued_prompt"] = {"text": "stuck", "transport": None}

    for _ in range(3):
        _fire(server, sid)
        assert server._sessions.get(sid) is session
    _fire(server, sid)

    assert sid not in server._sessions and popped == [session]


def test_idle_session_without_a_queue_is_still_reaped(server, detached):
    sid, session, popped = detached

    _fire(server, sid)

    assert sid not in server._sessions and popped == [session]


def test_running_turn_resets_the_budget(server, detached):
    sid, session, _ = detached
    session["queued_prompt"] = {"text": "mail", "transport": None}
    _fire(server, sid)
    assert session["_ws_orphan_queued_polls"] == 1

    session["running"] = True  # the queue drained into a new turn (head still set by a later queuer)
    session["_last_turn_activity_at"] = None
    server._schedule_ws_orphan_reap(sid, _expected_timer=server._pending_ws_reaps.get(sid))
    _Timer.created[-1].callback()

    assert "_ws_orphan_queued_polls" not in session


def test_a_reattach_gives_the_next_detachment_the_whole_budget(server, detached, monkeypatch):
    """Reattaching ends the wait: the polls it spent must not shorten the budget of a later detachment."""
    sid, session, popped = detached
    monkeypatch.setattr(server, "_WS_ORPHAN_QUEUED_MAX_POLLS", 2)
    session["queued_prompt"] = {"text": "mail from Home", "transport": None}
    _fire(server, sid)
    _fire(server, sid)
    assert session["_ws_orphan_queued_polls"] == 2  # spent: the next poll would reap

    server._cancel_ws_orphan_reap(sid)  # the client comes back

    assert "_ws_orphan_queued_polls" not in session
    _fire(server, sid)  # detaches again with the same prompt still queued

    assert server._sessions.get(sid) is session and not popped
    assert session["_ws_orphan_queued_polls"] == 1


def test_a_short_turn_between_polls_does_not_inherit_a_spent_budget(server, detached, monkeypatch):
    """A drain, turn and re-queue that all happen between two polls must not reap the NEW prompt on the old budget."""
    sid, session, popped = detached
    monkeypatch.setattr(server, "_WS_ORPHAN_QUEUED_MAX_POLLS", 2)
    session["queued_prompt"] = {"text": "first", "transport": None}
    _fire(server, sid)
    _fire(server, sid)
    assert session["_ws_orphan_queued_polls"] == 2

    # The whole cycle the reaper never observes: the queue drains into a turn, the turn runs and settles, and a
    # follow-up is queued — no poll in between, which is what production timing actually allows.
    session.pop("queued_prompt")
    session["running"] = True
    session["running"] = False
    session["queued_prompt"] = {"text": "second", "transport": None}

    _fire(server, sid)

    assert server._sessions.get(sid) is session and not popped
    assert session["_ws_orphan_queued_polls"] == 1


def test_real_turn_queues_a_leftover_steer_before_releasing_running_and_drains_it_once(tmp_path, monkeypatch):
    """Drives the real ``_run_prompt_submit``: in the post-turn housekeeping window (after ``running`` is
    released, before the follow-ups) the steer the turn never consumed must already be queued."""
    import types

    from tui_gateway import server

    agent = types.SimpleNamespace(session_id="x", clear_interrupt=lambda: None,
                                  run_conversation=lambda *a, **k: {"final_response": "done",
                                                                    "pending_steer": "late mail"})
    releases = []  # the queue at the instant ``running`` flips False (the reaper reads both without the lock)

    class _Session(dict):
        def __setitem__(self, key, value):
            if key == "running" and value is False:
                releases.append((self.get("queued_prompt") or {}).get("text"))
            super().__setitem__(key, value)

    session = _Session({"agent": agent, "session_key": "k", "history": [], "profile_home": str(tmp_path),
                        "history_lock": threading.RLock(), "history_version": 0, "running": True,
                        "attached_images": [], "image_counter": 0, "cols": 80, "slash_worker": None,
                        "show_reasoning": False, "tool_progress_mode": "all", "inflight_turn": None})
    for name in ("_wire_callbacks", "_sync_agent_model_with_config", "_register_session_cwd",
                 "_sync_session_key_after_compress"):
        monkeypatch.setattr(server, name, lambda *a, **k: None)
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    monkeypatch.setattr(server, "_session_cwd", lambda s: str(tmp_path))
    monkeypatch.setattr(server, "_tts_stream_begin", lambda: None)
    monkeypatch.setattr(server, "_get_usage", lambda a: {})
    # The turn runs on a spawned context thread; run it in place so the ordering is observed here.
    monkeypatch.setattr(server, "_start_session_work",
                        lambda target, **kw: target() or object())
    seen, drained = [], []
    monkeypatch.setattr(server, "_emit_settled_session_info",  # runs in the housekeeping window
                        lambda *a: seen.append((session["running"], (session.get("queued_prompt") or {}).get("text"))))

    def drain(rid, sid, s):
        drained.append((s.get("queued_prompt") or {}).get("text"))
        s.pop("queued_prompt", None)
        s.pop("queued_prompts", None)
        return True

    monkeypatch.setattr(server, "_drain_queued_prompt", drain)
    monkeypatch.setitem(server._sessions, "s", session)

    server._run_prompt_submit("r", "s", session, "hello")

    assert releases == ["late mail"]
    assert seen == [(False, "late mail")]
    assert drained == ["late mail"]


def test_lru_and_ttl_reapers_spare_a_fresh_queue_but_not_a_stuck_one(server, detached, monkeypatch):
    import time as _time

    sid, session, _ = detached
    # session_reaper's bodies are rebound onto tui_gateway.server's globals (bind_module): patch that seam.
    monkeypatch.setattr(server, "_session_is_lru_evictable", lambda *a: True)
    session["last_active"] = _time.time()
    assert server._session_is_reapable(sid, session) is True
    session["queued_prompt"] = {"text": "mail", "transport": None}
    assert server._session_is_reapable(sid, session) is False  # drains at the idle boundary
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0)  # "park forever" WS setting: spare still holds
    assert server._session_is_reapable(sid, session) is False
    session["last_active"] = _time.time() - 2 * server._QUEUED_PROMPT_SPARE_S
    assert server._session_is_reapable(sid, session) is True  # stuck: no longer pinned
    assert session["_queued_prompt_stale_logged"] is True  # logged once, not on every pass
