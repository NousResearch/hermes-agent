"""A live Desktop/TUI session is a mailbox address.

``tools/bot_live_delivery`` is the durable out-of-process → live-session handoff, and
``tui_gateway/session_notifications._poll_bot_live_delivery_once`` is its consumer. The address is
the session id; resolving it through the "Bot Chat" title makes every other live Desktop/TUI
session undeliverable — a delivery addressed to it stays ``queued`` for ever.
"""
import contextlib
import threading
from types import SimpleNamespace

from tui_gateway import session_notifications
from tui_gateway.method_ctx import rebind


def _live_session(home, session_key, live_session_id, *, capable=True):
    """Real SessionDB row + the active-session lease a live Desktop/TUI session holds."""
    from hermes_cli.active_sessions import try_acquire_active_session
    from hermes_state import SessionDB

    db = SessionDB(db_path=home / "state.db")
    db.create_session(session_id=session_key, source="desktop")
    lease, refusal = try_acquire_active_session(
        session_id=session_key, surface="desktop", config={}, registry_home=home,
        metadata={"live_session_id": live_session_id, "bot_live_delivery_consumer": capable})
    assert refusal is None and lease is not None
    return db, lease


def _owner(home, session_key, lease, live_session_id):
    """The tuple a producer resolves from the active-session registry before admitting."""
    return {"profile_home": str(home.resolve()), "session_id": session_key,
            "lease_id": lease.lease_id, "live_session_id": live_session_id}


def _poll(submitted):
    """The real poller body, bound to the server globals it touches and nothing else."""
    @contextlib.contextmanager
    def admission(session):
        with session["history_lock"]:
            yield True

    def submit(_rid, _sid, _session, text, **kwargs):
        submitted.append(text)
        kwargs["terminal_callback"]({"status": "settled", "text": "ack"})
        return True

    return rebind(session_notifications._poll_bot_live_delivery_once, {
        "_session_home": lambda session: session["profile_home"],
        "_session_turn_admission": admission,
        "_run_prompt_submit": submit,
        "_notif_release_turn": lambda session: session.update(running=False),
    })


def test_session_claims_its_own_mailbox_and_nobody_elses(tmp_path):
    from tools import bot_live_delivery as mailbox

    db, lease = _live_session(tmp_path, "desk-1", "tab-7")
    other_db, other_lease = _live_session(tmp_path, "desk-2", "tab-8")
    submitted = []
    try:
        mine = mailbox.deliver_to_live_owner(tmp_path, _owner(tmp_path, "desk-1", lease, "tab-7"),
                                            "HERM-634: run started")
        theirs = mailbox.deliver_to_live_owner(tmp_path, _owner(tmp_path, "desk-2", other_lease, "tab-8"),
                                               "other session's mail")
        session = {"history_lock": threading.RLock(), "agent": object(), "session_key": "desk-1",
                   "profile_home": tmp_path, "running": False,
                   "active_session_lease": SimpleNamespace(lease_id=lease.lease_id, released=False)}
        assert _poll(submitted)("tab-7", session) is True
        assert submitted == ["HERM-634: run started"]
        settled = mailbox.read_delivery_result(tmp_path, mine["delivery_id"])
        queued = mailbox.read_delivery_result(tmp_path, theirs["delivery_id"])
        assert settled is not None and queued is not None
        assert settled["status"] == "settled"
        # The other live session's envelope is not this session's to take.
        assert queued["status"] == "queued"
    finally:
        lease.release()
        other_lease.release()
        db.close()
        other_db.close()
