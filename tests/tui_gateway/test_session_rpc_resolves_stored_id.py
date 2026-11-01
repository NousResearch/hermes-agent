"""Session-scoped RPCs must accept the durable id a client already holds.

``session.resume`` registers the live runtime in ``_sessions`` under a short transient sid, while the
record keeps the durable stored id in ``session_key``/``resume_session_id``. Every non-CLI client —
Desktop, the web dashboard, third-party JSON-RPC clients — keeps sending that stored id on later RPCs
(a client has no reason to re-learn the runtime id after a resume it just performed). ``_sess_nowait``
consulted only ``_sessions[sid]``, so the first ``prompt.submit`` after a *successful* resume answered
4001 "session not found": the client resumed again, got the same runtime back, and looped —
"session not found" on a chat that was demonstrably live, with the app stacking reconnect attempts.

The durable-identity bridge already exists in this module (``_find_live_session_by_key``) and is trusted
for resume-reuse (``_claim_or_reuse_live``) and for notification routing
(``session_notifications._notif_session_matches``). It was wired into exactly one RPC path —
``approval.respond``, via ``_approval_respond_session_fallback``. This test pins the general case:
the resolver bridges a stored id to its live runtime, still rejects an id that belongs to nothing, and
never bridges a finalized runtime back into service.
"""

import threading
from types import SimpleNamespace

import pytest

import tui_gateway.server as srv


STORED = "20260831_090025_ec8d87"   # the durable id in the session store (what clients keep sending)
LIVE = "d1074d97"                   # the transient runtime id session.resume registered under


@pytest.fixture
def registry():
    """An empty ``_sessions`` for the test, restored afterwards."""
    with srv._sessions_lock:
        saved = dict(srv._sessions)
        srv._sessions.clear()
    try:
        yield srv._sessions
    finally:
        with srv._sessions_lock:
            srv._sessions.clear()
            srv._sessions.update(saved)


def _resumed_runtime(registry, *, agent=None, profile_home=None, finalized=False) -> dict:
    """A runtime shaped exactly as ``_claim_or_reuse_live`` registers a resumed one: keyed by the live
    sid, carrying the stored id in ``session_key`` (+ ``resume_session_id``)."""
    record = {
        "agent": agent,
        "session_key": STORED,
        "resume_session_id": STORED,
        "profile_home": profile_home,
        "running": False,
        "created_at": 0.0,
        "last_active": 0.0,
        "inflight_turn": None,
        "history": [],
        "history_lock": threading.Lock(),
    }
    if finalized:
        record["_finalized"] = True
    registry[LIVE] = record
    return record


def test_the_existing_helper_loses_the_stored_id_once_the_agent_is_built(registry):
    """Why the bridge cannot be ``_find_live_session_by_key``: it matches one key per record —
    ``agent.session_id or session_key or sid`` — and a built resumed agent ranks its fresh runtime id
    first. Pre-build the stored id resolves, post-build it does not: the same chat stops existing in the
    middle of its own resume, which is how a client ends up resuming the same session successfully and
    being told it is not found on the next request."""
    _resumed_runtime(registry)  # resume accepted, agent not built yet
    assert srv._find_live_session_by_key(STORED) is not None, "unbuilt resume record must be findable"

    _resumed_runtime(registry, agent=SimpleNamespace(session_id=LIVE))  # built: agent carries the fresh id
    assert srv._find_live_session_by_key(STORED) is None, "expected the shadowed-id behaviour this fixes"


def test_sess_nowait_resolves_the_stored_id_the_client_holds(registry):
    """The regression: resume succeeds, the next session-scoped RPC uses the id the client already has."""
    record = _resumed_runtime(registry)
    session, err = srv._sess_nowait({"session_id": STORED}, 1)
    assert err is None, f"stored id must reach the live runtime, got {err}"
    assert session is record


def test_sess_nowait_resolves_after_the_agent_is_built(registry):
    """The production shape, not the half-built one: a resumed session whose agent has finished building
    reports the fresh runtime id, which is the field the shared lookup ranks first — so the client still
    holding the stored id was rejected exactly when the session was most definitely alive and working."""
    record = _resumed_runtime(registry)
    record["agent"] = SimpleNamespace(session_id=LIVE)  # what a built resumed agent's agent/session look like
    session, err = srv._sess_nowait({"session_id": STORED}, 1)
    assert err is None, f"a built resumed runtime must stay reachable by its stored id, got {err}"
    assert session is record


def test_sess_nowait_still_rejects_an_id_that_belongs_to_nothing(registry):
    """Bridging durable ids must not turn into fuzzy matching: an unknown id keeps returning 4001 so the
    client's own resume path still sees a real not-found."""
    _resumed_runtime(registry)
    session, err = srv._sess_nowait({"session_id": "20260831_090025_nope00"}, 1)
    assert session is None
    assert err["error"]["code"] == 4001


def test_sess_nowait_does_not_revive_a_finalized_runtime(registry):
    """A torn-down runtime is not a live session; reviving it would hand a client an unbuildable slot."""
    _resumed_runtime(registry, finalized=True)
    session, err = srv._sess_nowait({"session_id": STORED}, 1)
    assert session is None
    assert err["error"]["code"] == 4001


def test_resolver_rekeys_any_identity_the_client_might_hold(registry):
    """Every id a record legitimately owns — registered sid, stored key, the resume key it was built from,
    and the id its built agent reports — resolves to one live slot."""
    for identity in (STORED, LIVE):
        record = _resumed_runtime(registry, agent=SimpleNamespace(session_id=identity))
        assert srv._find_live_session_by_client_id(STORED)[0] == LIVE
        assert srv._resolve_client_session_id(STORED) == LIVE
        assert srv._resolve_client_session_id(LIVE) == LIVE
        assert srv._resolve_client_session_id(STORED, record) == LIVE
        del record


def test_resolver_leaves_an_unknown_id_alone(registry):
    """Re-keying must not degrade into fuzzy matching: an id belonging to nothing is handed straight back
    so the caller's resume path still sees a real not-found."""
    _resumed_runtime(registry)
    missing = "20260831_090025_nope00"
    assert srv._find_live_session_by_client_id(missing) is None
    assert srv._resolve_client_session_id(missing) == missing


def test_scoped_lookup_does_not_cross_profiles(registry):
    """A stored id is a timestamp shape and the same one can be live in two profiles' stores (#100029).
    The unscoped resolver is what the RPC path needs today; the lookup underneath it still takes a profile
    so a profile-aware caller cannot be handed another profile's runtime."""
    _resumed_runtime(registry, profile_home="/home/a/.hermes")
    assert srv._find_live_session_by_client_id(STORED, profile_home="/home/a/.hermes")[0] == LIVE
    assert srv._find_live_session_by_client_id(STORED, profile_home="/home/b/.hermes") is None


def test_start_agent_build_accepts_the_stored_id(registry, monkeypatch):
    """The second half of the same class: holding the durable key, ``_start_agent_build`` looked itself
    up in ``_sessions`` by that key, saw nothing, marked the session ready and returned — the agent
    silently never built (image/file attach on a resumed session went nowhere, with no error)."""
    record = _resumed_runtime(registry)
    record["agent_ready"] = threading.Event()

    reached = []

    def _probe(sid, current):
        # Reached only when the build found its own record: with the stale key it is never called, and
        # the build settles "ready" without an agent. Returning False keeps the probe off the heavy path.
        reached.append((sid, current))
        return False

    monkeypatch.setattr(srv, "_await_resume_history", _probe)

    srv._start_agent_build(STORED, record)
    thread = record.get("_agent_build_thread")
    if thread is not None:
        thread.join(timeout=5)

    assert reached, (
        "_start_agent_build keyed its own lookup by the caller's stale id and returned without building"
    )
    assert reached[0][0] == LIVE, "the build must run against the live sid the record is registered under"
