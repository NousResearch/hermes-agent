"""Imported turns retain their receipt and cannot bypass the local FIFO."""
import threading
from types import SimpleNamespace

from tui_gateway.method_ctx import rebind
from tui_gateway import session_notifications, session_auto_continue
from tui_gateway.turn_marker import record_turn_start, read_turn_marker


def test_refused_input_commits_failed_mailbox_receipt(tmp_path):
    import contextvars
    import logging
    import time
    from tui_gateway import prompt_turn
    from tools import bot_live_delivery as mailbox

    owner = dict(profile_home=str(tmp_path.resolve()), session_id="chat",
                 lease_id="lease", live_session_id="live")
    queued = mailbox.deliver_to_live_owner(tmp_path, owner, "refused input")
    mailbox.claim_pending_delivery(tmp_path, owner)
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


def test_local_work_blocks_mailbox_claim_without_consuming_envelope(monkeypatch, tmp_path):
    import tools.bot_live_delivery as mailbox
    owner = {"lease_id": "lease", "live_session_id": "live", "session_id": "chat"}
    pending = [{"id": "receipt", "message": "imported"}]
    monkeypatch.setattr(mailbox, "find_canonical_live_owner", lambda home: owner)
    monkeypatch.setattr(mailbox, "claim_pending_delivery", lambda home, pinned: pending.pop(0))
    receipts = []
    monkeypatch.setattr(mailbox, "complete_delivery", lambda *args, **kwargs: receipts.append((args, kwargs)))
    submitted = []
    def submit(rid, sid, session, text, **kwargs):
        submitted.append(text)
        kwargs["terminal_callback"]({"status": "settled", "text": "reply"})
        return True
    poll = rebind(session_notifications._poll_bot_live_delivery_once, {
        "_session_home": lambda session: tmp_path,
        "_run_prompt_submit": submit,
        "_notif_release_turn": lambda session: session.update(running=False),
    })
    session = {"history_lock": threading.RLock(), "agent": object(), "session_key": "chat",
               "active_session_lease": SimpleNamespace(lease_id="lease", released=False)}
    for blocker in ("running", "queued_prompt", "queued_prompts", "_auto_continue_scheduled"):
        session[blocker] = True
        assert poll("live", session) is False
        assert pending and not submitted
        session.pop(blocker)
    assert poll("other-live", session) is False
    assert pending
    assert poll("live", session) is True
    assert submitted == ["imported"] and not pending
    assert receipts[0][0][1] == "receipt"
    assert receipts[0][1]["reply"] == "reply"
