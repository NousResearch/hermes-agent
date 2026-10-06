"""Conditional local operations: authority is compared again at use, never inferred from events."""
from __future__ import annotations

import contextlib

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method

# Opting in is per connection, and survives removal/replacement of its runtime.
# Unknown and future mutators fail closed; a consumer must use the typed envelope.
_BOUND_READ_METHODS = frozenset({
    "session.activate_bound", "session.invoke_bound", "session.create", "session.branch_stored",
    "session.history", "session.events.since", "session.events.stats", "session.active_list",
    "session.list", "client.capabilities", "gateway.ping",
})


def _bound_legacy_refusal(peer, name: str, rid):
    if getattr(peer, "_conditional_session_mode", False) and name not in _BOUND_READ_METHODS:
        return _err(rid, 4007, "Conditional connection requires session.invoke_bound")
    return None


@contextlib.contextmanager
def _bound_operation_cut(params: dict):
    from sqlite3 import Error as SQLiteError

    sid = params["session_id"]
    peer = current_transport()
    owner = _transport_auth_user_id(peer)
    requested_home = _profile_home(params.get("profile"))
    store_path = str((Path(requested_home or _hermes_home) / "state.db").resolve())
    with _sessions_lock:
        session = _sessions.get(sid)
    if session is None or not isinstance(session.get("history_lock"), type(threading.RLock())):
        yield None
        return
    # Reentrant gateway locks let the existing immediate handlers participate in
    # this cut. Only ready local engines are eligible: no build wait or host RPC
    # can occur while these locks are held. Try the leaf engine guard, never wait.
    with _session_resume_lock, session["history_lock"], _sessions_lock, _session_transport_lock:
        with _activation_engine_guard(session) as proven:
            receipt = _bound_activation_receipt(session, sid, owner, store_path, params["expected_binding"])
            if (not proven or session.get("agent") is None or _sessions.get(sid) is not session
                    or receipt is None or not _transport_is_live_peer(peer)
                    or not _session_transport_contains(session, peer)
                    or session.get("bound_subscribers", {}).get(peer) != receipt):
                yield None
                return
            try:
                if session["creation_engine"].database.resolve_resume_session_id(
                        receipt["stored_session_id"], strict=True) != receipt["stored_session_id"]:
                    yield None
                    return
            except (OSError, SQLiteError):
                yield None  # unreadable durable identity cannot authorize a write
                return
            yield session


@contextlib.contextmanager
def _bound_request_cut(session_id: str, operation: dict):
    from . import server_requests

    request_id = operation.get("request_id") or operation.get("id")
    if request_id is None:
        yield True
        return
    # Keep ownership lookup and settlement under the request registry's authority;
    # an id copied from another session cannot be answered by this envelope.
    with server_requests._lock:
        request = server_requests._open.get(request_id)
        yield request is not None and request.sid == session_id


def _bound_turn_runtime_current(session: dict, sid: str, witness) -> bool:
    origin = session["creation_binding"]
    return (_sessions.get(sid) is session and session.get("agent") is witness.engine
            and session.get("session_key") == origin.stored_session_id
            and session.get("auth_user_id") == origin.authenticated_owner
            and str((Path(session.get("profile_home") or _hermes_home) / "state.db").resolve()) == origin.store_path
            and not any(session.get(flag) for flag in (
                "_closing", "_finalized", "_turn_cancel_requested", "_compute_host_ever_owned",
                "_compute_host_active", "_compute_host_turn_id")))


@method("session.invoke_bound")
def _(rid, params: dict) -> dict:
    from pydantic import ValidationError
    from .contracts.sessions import SessionInvokeBoundParams
    from agent.session_identity import expected_turn_identity

    try:
        SessionInvokeBoundParams.model_validate(params)
    except ValidationError:
        return _err(rid, 4000, "Invalid conditional operation precondition")
    operation = dict(params["operation"])
    name = operation.pop("method")
    with _bound_operation_cut(params) as session:
        if session is None:
            return _err(rid, 4007, "Conditional operation refused")
        # Prompts are deliberate new idle turns. Busy steering/queueing, destructive
        # rewinds, attachments and synthesized turns have separate admission rules.
        if name == "prompt.submit" and (session.get("running") or session.get("attached_images")):
            return _err(rid, 4009, "Conditional prompt requires an idle session without staged attachments")
        with _bound_request_cut(params["session_id"], operation) as owned:
            if not owned:
                return _err(rid, 4007, "Conditional request refused")
            kwargs = {**operation, "session_id": params["session_id"]}
            if name in {"request.answer", "clarify.lock"}:
                kwargs.pop("session_id")  # legacy models address requests, not sessions
            # The worker carries this witness through its durable lease wait. It
            # may not adopt a different tip or replay a carried unadmitted input.
            witness = session["creation_engine"]
            identity_scope = (expected_turn_identity(
                witness.engine, witness.database, witness.revision,
                session["creation_binding"].stored_session_id,
                lambda: _bound_turn_runtime_current(session, params["session_id"], witness))
                if name == "prompt.submit" else contextlib.nullcontext())
            with identity_scope, _session_profile_runtime_scope(session, hydrate_secrets=False):
                response = _methods[name](rid, kwargs)
            if "error" in response:
                return response
            return _ok(rid, {"operation_result": response["result"]})


def register(server) -> None:
    bind_module(globals(), server, skip=("_",))
