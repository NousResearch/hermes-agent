"""A completed hosted writer must not deadlock or evict an attached viewer."""

import contextlib
import threading
from pathlib import Path

import pytest

from hermes_state import SessionDB
from tui_gateway import server
from tui_gateway.hosted_room_server_rpc import HostedRoomServerRPC
from tui_gateway.hosted_room_sessions import retire_hosted_session


class Viewer:
    _closed = False

    def write(self, _message):
        return True


def test_activation_and_retirement_do_not_invert_registry_and_history_locks(monkeypatch):
    activation_has_history = threading.Event()
    retirement_waits_for_history = threading.Event()
    errors = []

    class ObservedHistoryLock:
        def __init__(self):
            self.lock = threading.Lock()

        def __enter__(self):
            if threading.current_thread().name == "retirement":
                retirement_waits_for_history.set()
                # Bound cleanup on the broken base instead of hanging the suite.
                if not self.lock.acquire(timeout=2):
                    raise TimeoutError("retirement inverted activation's history/registry locks")
            else:
                self.lock.acquire()
            return self

        def __exit__(self, *_args):
            self.lock.release()

    record = {"history_lock": ObservedHistoryLock(), "source": "bot_room", "running": False}
    monkeypatch.setattr(server, "_sessions", {"live": record})
    original_cancel = server._cancel_ws_orphan_reap

    class ProbeComplete(Exception):
        pass

    def cancel(sid):
        activation_has_history.set()
        assert retirement_waits_for_history.wait(5)
        original_cancel(sid)
        raise ProbeComplete()

    monkeypatch.setattr(server, "_cancel_ws_orphan_reap", cancel)

    def activate():
        try:
            server._live_session_payload("live", record, transport=Viewer(), omit_messages=True)
        except ProbeComplete:
            pass
        except BaseException as exc:
            errors.append(exc)

    def retire():
        try:
            retire_hosted_session(server, "live", record)
        except BaseException as exc:
            errors.append(exc)

    activation = threading.Thread(target=activate, name="activation")
    retirement = threading.Thread(target=retire, name="retirement")
    activation.start()
    assert activation_has_history.wait(5)
    retirement.start()
    activation.join(8)
    retirement.join(8)
    assert not activation.is_alive() and not retirement.is_alive()
    assert not errors, [str(exc) for exc in errors]


@pytest.mark.parametrize("stdio_primary", [False, True])
def test_internal_resume_preserves_all_attached_viewers(tmp_path, monkeypatch, stdio_primary):
    home = tmp_path / "runtime-state"
    profile_home = home / "profiles" / "reviewer"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db = SessionDB(db_path=profile_home / "state.db")
    key, sid = "stored", "live"
    db.create_session(key, source="bot_room", profile_name="reviewer")
    db.set_session_title(key, "Group: review-room")
    db.set_session_hidden(key, True)
    monkeypatch.setattr(server, "_profile_home", lambda _profile: profile_home)
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_load_cfg", lambda: {})
    monkeypatch.setattr(server, "current_transport", lambda: None)
    lease, refusal = server._claim_active_session_slot(
        key, live_session_id=sid, surface="bot_room", profile_home=profile_home,
    )
    assert refusal is None and lease is not None
    viewer = Viewer()
    completed = threading.Thread(target=lambda: None)
    completed.start()
    completed.join()
    transport = server._stdio_transport if stdio_primary else viewer
    record = {
        "history_lock": threading.Lock(), "source": "bot_room", "running": False,
        "session_key": key, "profile_home": str(profile_home), "agent": None,
        "history": [], "transport": transport, "viewers": {viewer: 1},
        "active_session_lease": lease, "_run_thread": completed,
        "_hosted_retirement_pending": True,
    }
    monkeypatch.setattr(server, "_sessions", {sid: record})

    @contextlib.contextmanager
    def session_db(_record):
        yield db

    monkeypatch.setattr(server, "_session_db", session_db)
    try:
        assert not retire_hosted_session(server, sid, record)
        from hermes_state_registry import stats
        borrowed_before = stats()["total_refcounts"]
        result = HostedRoomServerRPC(server).resume(profile="reviewer", session_id=key, source="bot_room")
        assert result["session_id"] == sid
        assert stats()["total_refcounts"] == borrowed_before
        assert record["transport"] is transport
        assert viewer in record["viewers"] and not viewer._closed
        assert not retire_hosted_session(server, sid, record)
        assert server._sessions[sid] is record
        assert record["active_session_lease"] is lease and not lease.released
    finally:
        lease.release()
        db.close()
