"""Imported turns retain their receipt and cannot bypass the local FIFO."""
import threading
from types import SimpleNamespace

from tui_gateway.method_ctx import rebind
from tui_gateway import session_notifications, session_auto_continue
from tui_gateway.turn_marker import record_turn_start, read_turn_marker


def test_refused_input_commits_failed_mailbox_receipt(tmp_path):
    import contextlib
    import contextvars
    import logging
    import time
    from tui_gateway import prompt_turn
    from tools import bot_live_delivery as mailbox

    owner = dict(profile_home=str(tmp_path.resolve()), session_id="chat",
                 lease_id="lease", live_session_id="live")
    queued = dict(id='a' * 32, delivery_id='a' * 32, owner=owner, **owner,
                  status='claimed', message='refused input', created_at=1)
    with mailbox._locked(tmp_path) as root:
        mailbox._write(root / (queued['id'] + '.json'), queued)
    agent = SimpleNamespace(session_id="chat")
    session = dict(agent=agent, session_key="chat", history_lock=threading.RLock(), running=True)
    retired = []
    noop = lambda *args, **kwargs: None
    submit = rebind(prompt_turn._run_prompt_submit, {
        "threading": threading, "time": time, "logger": logging.getLogger(__name__),
        "_sessions_lock": threading.RLock(), "_sessions": {},
        "_admit_prompt_turn": lambda *args: ([], agent),
        "_emit": noop, "bind_transport": noop, "reset_transport": noop,
        "_current_runtime_session_record": contextvars.ContextVar("refused_turn"),
        "_TurnRun": prompt_turn._TurnRun,
        "_record_turn_marker": lambda *args, **kwargs: "marker",
        "_prepare_turn_input": lambda *args: None,
        "_finish_turn": noop, "_clear_inflight_turn": noop,
        "_retire_turn_marker": lambda *args: retired.append(args),
        "_emit_settled_session_info": noop,
        "_routing_provenance_db": lambda _session: contextlib.nullcontext(None),
        "_reopen_routed_session_row": noop,
    })
    def terminal(outcome):
        mailbox.complete_delivery(tmp_path, queued["id"], status=outcome["status"],
                                  error=outcome.get("error", ""))
    assert submit(None, "live", session, "refused input", terminal_callback=terminal)
    session["_run_thread"].join(timeout=5)
    assert not session["_run_thread"].is_alive()
    assert mailbox.read_delivery_result(tmp_path, queued["id"])["status"] == "failed"
    assert retired and session["running"] is False


def test_imported_crash_marker_never_autocontinues(tmp_path):
    record_turn_start(tmp_path, "chat", "imported", auto_continue=False)
    marker = read_turn_marker(tmp_path, "chat")
    assert marker["auto_continue"] is False
    schedule = rebind(session_auto_continue._maybe_schedule_auto_continue, {
        "_session_home": lambda session: tmp_path,
        "read_turn_marker": read_turn_marker,
    })
    assert schedule("live", {}, "chat") is None


def test_viewer_poller_never_discovers_or_claims_bot_execution(monkeypatch, tmp_path):
    import queue
    import time
    import tools.bot_live_delivery as mailbox
    from tools.process_registry import process_registry
    calls = []
    monkeypatch.setattr(mailbox, 'find_canonical_live_owner', lambda home: calls.append(home))
    stop = threading.Event()
    events = queue.Queue()
    events.put({'type': 'owned-completion'})
    monkeypatch.setattr(process_registry, 'completion_queue', events)
    delivered = []
    poll = rebind(session_notifications._notification_poller_loop, {
        'time': time, '_LOOP_POLL_SECONDS': 0, '_KANBAN_POLL_SECONDS': 0,
        '_poll_bot_live_delivery_once': lambda *a: calls.append('legacy-claim'),
        '_maybe_fire_tui_loop_tick': lambda *a: None,
        '_maybe_fire_tui_heartbeat_tick': lambda *a: None,
        '_notif_poll_kanban': lambda *a: None,
        '_session_home': lambda session: tmp_path,
        '_notif_handle_ready': lambda sid, session, ready, *a, **kw: (delivered.extend(ready), stop.set()),
    })
    poll(stop, 'live', {'agent': object(), 'history_lock': threading.RLock(), 'session_key': 'chat',
                      'active_session_lease': SimpleNamespace(lease_id='lease', released=False)})
    assert delivered == [{'type': 'owned-completion'}]
    assert calls == []
