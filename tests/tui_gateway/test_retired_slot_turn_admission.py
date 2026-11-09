"""A slot retired after the gate passed must not keep a durable turn lease.

The phone's slot check (``_ensure_active_session_slot``) and the durable claim
(``admit_durable_turn_lease``, inside ``run_conversation``) are not one decision.
Desktop resume drops an idle webui slot under the registry lock, then a phone
send that already passed the gate can still insert a ``session_turn_leases`` row.
The admission path rechecks the slot after the claim and releases it.

Removing that recheck (``lease_is_registered`` after a successful claim, forced
true) makes ``test_a_retired_slot_cannot_claim_a_turn_lease`` fail: the phone
is no longer refused ``SESSION_NOT_OWNED`` and the conversation loop runs.
"""
import threading

from hermes_cli.active_sessions import (
    SESSION_NOT_OWNED,
    YIELDABLE_SURFACES,
    active_session_liveness_guard,
    lease_is_registered,
    try_acquire_active_session,
)
from hermes_state import SessionDB


def _turn_lease_rows(db_path):
    """Rows another process would see. A same-connection read can miss an uncommitted insert."""
    other = SessionDB(db_path=db_path)
    try:
        with other._read_ctx() as conn:
            return conn.execute(
                "SELECT conversation_id, holder, expires_at FROM session_turn_leases"
            ).fetchall()
    finally:
        other.close()


def _phone(db, slot, session_id):
    """The slice of an agent ``run_conversation`` touches before the model loop."""
    from agent.turn_facade import TurnFacadeMixin

    class Phone(TurnFacadeMixin):
        pass

    agent = Phone()
    agent._session_db = db
    agent.session_id = session_id
    agent._turn_slot_lease = slot
    agent.platform = "webui"
    agent._persist_disabled = False
    agent._interrupt_requested = False
    agent._session_turn_lease_refresh_interval = 60.0
    agent._parent_session_id = None
    agent._execution_thread_id = None
    agent.statuses = []
    agent._emit_status = agent.statuses.append
    agent._emit_warning = agent.statuses.append
    agent._touch_activity = lambda *a, **k: None
    agent._liveness_activity_lock = lambda: threading.Lock()
    agent._reset_activity_labels_after_turn = lambda: None
    agent._conversation_root_id = lambda: session_id
    return agent


def _admit(agent, monkeypatch):
    """Phone turn admission. The loop is stubbed so a missing recheck returns success, not a model error."""
    entered = []

    def _loop(*args, **kwargs):
        entered.append(True)
        return {"final_response": "phone turn ran", "messages": [], "completed": True}

    monkeypatch.setattr("agent.conversation_loop.run_conversation", _loop)
    result = agent.run_conversation("from the phone", conversation_history=[], task_id="phone-turn")
    return result, entered


def test_a_retired_slot_cannot_claim_a_turn_lease(tmp_path, monkeypatch):
    sid = "20261109_230000_c3race"
    slot, refusal = try_acquire_active_session(
        session_id=sid, surface="webui", config={}, registry_home=tmp_path,
        metadata={"live_session_id": "phone-socket"})
    assert slot is not None and refusal is None, refusal
    assert lease_is_registered(slot) is True  # the slot gate would pass

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session(sid, source="webui")
        agent = _phone(db, slot, sid)

        # Desktop resume: probe sees no turn lease, drops the idle webui slot, commits.
        with active_session_liveness_guard(
                sid, registry_home=tmp_path, yield_surfaces=YIELDABLE_SURFACES,
                holder_is_idle=lambda entry: True) as active:
            assert active is False
        assert lease_is_registered(slot) is False

        result, entered = _admit(agent, monkeypatch)
    finally:
        db.close()

    assert result.get("failure_reason") == SESSION_NOT_OWNED
    assert result.get("error") == SESSION_NOT_OWNED
    assert entered == []  # the extra phone turn does not run
    assert _turn_lease_rows(tmp_path / "state.db") == []


def test_a_slot_retired_during_the_claim_is_released(tmp_path, monkeypatch):
    """The recheck is after the row exists: a drop inside the claim still leaves no row."""
    sid = "20261109_230200_c3mid"
    slot, refusal = try_acquire_active_session(
        session_id=sid, surface="webui", config={}, registry_home=tmp_path,
        metadata={"live_session_id": "phone-socket"})
    assert slot is not None and refusal is None, refusal
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session(sid, source="webui")
        agent = _phone(db, slot, sid)
        real = db.acquire_session_turn_lease

        def acquire(*args, **kwargs):
            ok = real(*args, **kwargs)
            with active_session_liveness_guard(
                    sid, registry_home=tmp_path, yield_surfaces=YIELDABLE_SURFACES,
                    holder_is_idle=lambda entry: True) as active:
                assert active is False
            return ok

        monkeypatch.setattr(db, "acquire_session_turn_lease", acquire)
        result, entered = _admit(agent, monkeypatch)
    finally:
        db.close()

    assert result.get("failure_reason") == SESSION_NOT_OWNED
    assert entered == []
    assert _turn_lease_rows(tmp_path / "state.db") == []


def test_a_live_slot_still_admits_the_turn(tmp_path, monkeypatch):
    """Control: the recheck refuses a retired slot, not one this runtime still holds."""
    sid = "20261109_230100_c3live"
    slot, refusal = try_acquire_active_session(
        session_id=sid, surface="webui", config={}, registry_home=tmp_path,
        metadata={"live_session_id": "phone-socket"})
    assert slot is not None and refusal is None, refusal
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session(sid, source="webui")
        agent = _phone(db, slot, sid)
        assert lease_is_registered(slot) is True
        result, entered = _admit(agent, monkeypatch)
    finally:
        db.close()
        slot.release()

    assert result.get("failure_reason") != SESSION_NOT_OWNED
    assert entered == [True]
    assert _turn_lease_rows(tmp_path / "state.db") == []
