"""Concurrent submits must admit one turn and queue the other for the same session."""

import threading
import types

from tui_gateway import server
from tui_gateway.transport import bind_transport, reset_transport


class _AdmissionWindowLock:
    """Pause the first idle observation after it releases the real lock."""

    def __init__(self, session: dict, first_release: threading.Event, release_window: threading.Event):
        self._session = session
        self._real = threading.Lock()
        self._first_release = first_release
        self._release_window = release_window
        self._parked = False

    def __enter__(self):
        self._real.acquire()
        return self

    def __exit__(self, *exc):
        park = not self._parked and not self._session.get("running")
        if park:
            self._parked = True
        # Release first: the parked thread must not hold the lock, or the second submit
        # could not run the admission path this probe is about.
        self._real.release()
        self._first_release.set()
        if park:
            assert self._release_window.wait(10.0)
        return False


def _session(**extra):
    return {
        "agent": types.SimpleNamespace(),
        "session_key": "session-key",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "transport": None,
        "attached_images": [],
        **extra,
    }


def _submit(sid: str, text: str, out: dict, key: str) -> None:
    token = bind_transport(None)
    try:
        out[key] = server.handle_request(
            {"id": key, "method": "prompt.submit", "params": {"session_id": sid, "text": text}}
        )
    finally:
        reset_transport(token)


def test_concurrent_same_session_submits_admit_one_turn(monkeypatch):
    first_release = threading.Event()
    release_window = threading.Event()
    session = _session()
    session["history_lock"] = _AdmissionWindowLock(session, first_release, release_window)
    sid = "admission-sid"
    server._sessions[sid] = session

    admissions: list = []
    real_start_inflight = server._start_inflight_turn

    def _record_inflight(target_session, text, display_kind=None):
        admissions.append(text)
        real_start_inflight(target_session, text, display_kind=display_kind)

    monkeypatch.setattr(server, "_ensure_active_session_slot", lambda _sid, _session: None)
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda *_a, **_k: False)
    monkeypatch.setattr(server, "_load_dashboard_process_isolation_config", lambda: {})
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "queue")
    monkeypatch.setattr(server, "_persist_session_row_for_submit", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_restart_completed_failed_agent_build", lambda *_a, **_k: True)
    monkeypatch.setattr(server, "_start_agent_build", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_start_inflight_turn", _record_inflight)
    monkeypatch.setattr(server, "_run_after_agent_ready", lambda *_a, **_k: None)

    results: dict = {}
    try:
        first = threading.Thread(target=_submit, args=(sid, "first prompt", results, "r1"), name="submit-1")
        first.start()
        assert first_release.wait(10.0)
        second = threading.Thread(target=_submit, args=(sid, "second prompt", results, "r2"), name="submit-2")
        second.start()
        second.join(10.0)
        release_window.set()
        first.join(10.0)
    finally:
        release_window.set()
        server._sessions.pop(sid, None)

    assert not first.is_alive() and not second.is_alive(), "a submit thread never returned"
    assert set(results) == {"r1", "r2"}, results
    outcome = {
        "admissions": admissions,
        "statuses": sorted((results[key].get("result") or {}).get("status") for key in results),
        "running": session["running"],
        "user": session["inflight_turn"]["user"],
    }
    assert outcome == {
        "admissions": ["first prompt"], "statuses": ["queued", "streaming"],
        "running": True, "user": "first prompt",
    }
