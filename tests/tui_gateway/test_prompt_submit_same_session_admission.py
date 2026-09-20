"""Two concurrent ``prompt.submit`` calls for ONE session must admit exactly one turn.

``prompt.submit`` observed ``running`` under ``history_lock``, left the lock (the
``break`` that ends the idle observation), and only re-acquired it later, inside the
turn-claim helper, to flip ``running`` and start the in-flight turn. A second submit
for the same session could land in that gap, observe the same idle state, and start a
SECOND turn: two turn runners for one session, i.e. two ``pre_llm_call`` hook runs for one
prompt and duplicate delivery of whatever those runs emit. The observation and the claim
now happen under ONE ``history_lock`` hold, and the submit that finds the session already
claimed falls through to the pre-existing ``_handle_busy_submit`` contract
(queued / steered / redirected).

The probe makes that old window deterministic instead of hoping for a scheduling
coincidence: ``history_lock`` is wrapped so the release which ends an idle observation
parks its thread, letting a second submit run the entire admission path first.
"""

import threading
import types

from tui_gateway import server
from tui_gateway.transport import bind_transport, reset_transport


class _AdmissionWindowLock:
    """``threading.Lock`` stand-in that parks the first thread which releases the lock while
    ``running`` is falsy — the exact admission window a second submit must not fit into."""

    def __init__(self, session: dict, window_entered: threading.Event, release_window: threading.Event):
        self._session = session
        self._real = threading.Lock()
        self._window_entered = window_entered
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
        if park:
            self._window_entered.set()
            self._release_window.wait(10.0)
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
    window_entered = threading.Event()
    release_window = threading.Event()
    session = _session()
    session["history_lock"] = _AdmissionWindowLock(session, window_entered, release_window)
    sid = "admission-sid"
    server._sessions[sid] = session

    admissions: list = []
    real_start_inflight = server._start_inflight_turn

    def _record_inflight(target_session, text, display_kind=None):
        admissions.append(text)
        real_start_inflight(target_session, text, display_kind=display_kind)

    # Slot claim / persistence / agent build / turn body are NOT the admission boundary
    # under test: they are stubbed so the probe observes the gate itself. The slot claim is
    # safe to stub out — its per-session exclusivity is re-entrant for the same writer
    # (pid + live_session_id), so it never serializes two submits of one live process
    # (hermes_cli/active_sessions.py:137-148, :500-506).
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
        # Pre-fix the first submit parks here with the session still idle, so this fires.
        observed_window = window_entered.wait(1.0)
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
    statuses = sorted((results[key].get("result") or {}).get("status") for key in results)
    # One prompt admitted (a turn runner + its pre_llm_call), the other handled by the busy
    # contract — never two admissions for one session.
    assert len(admissions) == 1, (observed_window, statuses, admissions)
    assert statuses == ["queued", "streaming"], (observed_window, statuses)
    assert session["running"] is True
    assert session["inflight_turn"]["user"] in ("first prompt", "second prompt")
