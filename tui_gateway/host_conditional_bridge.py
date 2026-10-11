"""Parent half of child-held conditional reservations; never certify a mirror."""
from __future__ import annotations

import contextlib
import queue
import uuid

from .method_ctx import bind_module


class TentativePeer:
    """Suppress delivery before commit. Lost ranges remain explicit replay work."""
    _conditional_tentative = True

    def __init__(self, peer):
        self.peer = peer
        self.auth_identity = getattr(peer, "auth_identity", None)

    def write(self, _obj):
        return not self._closed

    @property
    def _closed(self):
        return bool(getattr(self.peer, "_closed", False))

    def close(self):
        pass  # owns no client resources


def _host_candidate(session):
    return session is not None and (session.get("_compute_host_ever_owned") or session.get("_compute_host_active"))


@contextlib.contextmanager
def _host_parent_cut(params, session, peer, *, admission=False):
    sid = params["session_id"]
    requested = _profile_home(params.get("profile"))
    store = str((Path(requested or _hermes_home) / "state.db").resolve())
    with _session_resume_lock, session["history_lock"], _sessions_lock, _session_transport_lock:
        receipt = _bound_activation_receipt(session, sid, _transport_auth_user_id(peer), store,
                                           params["expected_binding"], host=True)
        yield (receipt is not None and _sessions.get(sid) is session
               and session.get("agent") is None and (admission or _transport_is_live_peer(peer)))


def _host_existing_authority(session):
    supervisor = _compute_host_supervisor
    boot = session.get("_conditional_host_boot")
    if supervisor is None or not boot or supervisor.conditional_boot() != boot:
        return None
    return supervisor, boot


def _host_stage_tentative(session, tentative):
    # A delivery sink is provisional membership, never client liveness. Using
    # normal peer attach would let an abandoned prepare suppress orphan reaping.
    existing = session.get("transport")
    if isinstance(existing, FanoutTransport):
        existing.attach(tentative)
    else:
        session["transport"] = FanoutTransport(existing, tentative)


def _host_activate_bound(rid, params):
    with _sessions_lock:
        session = _sessions.get(params["session_id"])
    if not _host_candidate(session):
        return None
    peer = current_transport()
    with _host_parent_cut(params, session, peer) as valid:
        authority = _host_existing_authority(session) if valid else None
    if authority is None:
        return _err(rid, 4007, "Conditional host activation refused")
    # Concurrent retries on one connection must reuse the first accepted child
    # token. This peer-only lock never holds a gateway guard while waiting.
    with _session_transport_lock:
        gate = getattr(peer, "_host_activation_lock", None)
        if gate is None:
            gate = peer._host_activation_lock = threading.Lock()
    with gate:
        return _host_activate_reserved(rid, params, session, peer, authority)


def _host_activate_reserved(rid, params, session, peer, authority):
    supervisor, boot = authority
    with _host_parent_cut(params, session, peer) as valid:
        if not valid or supervisor.conditional_boot() != boot:
            return _err(rid, 4007, "Conditional host activation refused")
    # No parent locks while waiting on a pipe: preceding mirrored RPC frames may
    # require them before the stdout reader can deliver the reservation reply.
    retained = session.get("host_bound_subscribers", {}).get(peer)
    prepared = supervisor.conditional_exchange(boot, {"action": "prepare", "params": params,
        **({"subscription": retained["subscription"]} if retained else {})})
    if "reservation" not in prepared:
        return _err(rid, prepared.get("error", 4007), "Conditional host activation refused")
    tentative = TentativePeer(peer)
    committed = False
    try:
        with _host_parent_cut(params, session, peer) as valid:
            if not valid or supervisor.conditional_boot() != boot:
                return _err(rid, 4007, "Conditional host activation refused")
            _host_stage_tentative(session, tentative)
            ticket = supervisor.conditional_send(boot, {
                "action": "commit", "reservation": prepared["reservation"]})
        ack = supervisor.conditional_receive(ticket)
        response = ack.get("response", {})
        with _host_parent_cut(params, session, peer) as valid:
            if (not valid or supervisor.conditional_boot() != boot or "result" not in response
                    or response["result"].get("accepted_binding") != params["expected_binding"]):
                return _err(rid, 4007, "Conditional host activation refused")
            if not _attach_session_transport(session, peer):
                return _err(rid, 4007, "Conditional host activation refused")
            session.setdefault("bound_subscribers", {})[peer] = dict(params["expected_binding"])
            session.setdefault("host_bound_subscribers", {})[peer] = {
                "boot": boot, "subscription": prepared["subscription"], "supervisor": supervisor}
            peer._conditional_session_mode = True
            session.setdefault("viewers", {})[peer] = time.time()
            _cancel_ws_orphan_reap(params["session_id"])
            committed = True
            return _ok(rid, response["result"])
    finally:
        with _session_transport_lock:
            _detach_session_transport(session, tentative)
        if not committed:
            supervisor.conditional_abort(boot, prepared["reservation"])
            if retained is None:
                supervisor.conditional_release(boot, prepared["subscription"])


def _host_invoke_bound(rid, params):
    with _sessions_lock:
        session = _sessions.get(params["session_id"])
    if not _host_candidate(session):
        return None
    peer = current_transport()
    with _host_parent_cut(params, session, peer) as valid:
        member = session.get("host_bound_subscribers", {}).get(peer) if valid else None
        if (member is None or not _session_transport_contains(session, peer)
                or session.get("bound_subscribers", {}).get(peer) != params["expected_binding"]):
            return _err(rid, 4007, "Conditional host operation refused")
    supervisor, boot = member["supervisor"], member["boot"]
    turn_id = uuid.uuid4().hex
    prepared = supervisor.conditional_exchange(boot, {
        "action": "prepare", "params": params, "subscription": member["subscription"], "turn_id": turn_id})
    if "reservation" not in prepared:
        return _err(rid, prepared.get("error", 4007), "Conditional host operation refused")
    is_prompt = params["operation"]["method"] == "prompt.submit"
    sent = False
    try:
        with _host_parent_cut(params, session, peer) as valid:
            if (not valid or session.get("host_bound_subscribers", {}).get(peer) is not member
                    or not _session_transport_contains(session, peer)):
                return _err(rid, 4007, "Conditional host operation refused")
            if is_prompt:
                session.setdefault("_conditional_pending_turns", {})[turn_id] = {
                    "boot": boot, "params": params, "peer": peer, "accepted": False, "done": None}
                supervisor.conditional_track_turn(turn_id, params["session_id"],
                    lambda done: _host_bound_turn_done(rid, params["session_id"], session, turn_id, done))
            ticket = supervisor.conditional_send(boot, {"action": "commit", "reservation": prepared["reservation"]})
        sent = True
        ack = supervisor.conditional_receive(ticket)
    except (OSError, RuntimeError, queue.Empty):
        if is_prompt:
            _host_forget_pending_turn(supervisor, session, turn_id)
        if sent:
            return _err(rid, 5019, "Conditional host outcome unconfirmed; do not replay")
        raise
    finally:
        if not sent:
            supervisor.conditional_abort(boot, prepared["reservation"])
    response = ack.get("response")
    if not isinstance(response, dict) or "result" not in response:
        if is_prompt:
            _host_forget_pending_turn(supervisor, session, turn_id)
        return _err(rid, (response or {}).get("error", {}).get("code", ack.get("error", 5019)),
                    "Conditional host operation refused or unconfirmed; do not replay")
    completed = None
    if is_prompt:
        with session["history_lock"]:
            pending = session.get("_conditional_pending_turns", {}).get(turn_id)
            if pending is not None:
                pending["accepted"] = True
                if pending["done"] is None:
                    session["_compute_host_turn_id"] = turn_id
                    session["running"] = True
                    _start_inflight_turn(session, params["operation"]["text"])
                else:
                    completed = pending["done"]
        if completed is not None:
            _host_bound_turn_done(rid, params["session_id"], session, turn_id, completed)
    return _ok(rid, response["result"])


def _host_bound_turn_done(rid, sid, session, turn_id, done):
    with session["history_lock"], _sessions_lock:
        pending = session.get("_conditional_pending_turns", {}).get(turn_id)
        if pending is None:
            return
        if _sessions.get(sid) is not session:
            session["_conditional_pending_turns"].pop(turn_id, None)
            return
        if not pending["accepted"]:
            pending["done"] = done
            return
        session["_conditional_pending_turns"].pop(turn_id, None)
        if session.get("_compute_host_turn_id") == turn_id:
            session.pop("_compute_host_turn_id", None)
    _on_compute_host_turn_done(rid, sid, session, done)


def _host_conditional_call(handler, rid, params):
    try:
        return handler(rid, params)
    except (OSError, RuntimeError, queue.Empty):
        return _err(rid, 4007, "Conditional host unavailable; do not replay")


def register(server):
    bind_module(globals(), server)


def _host_invalidate_delivery_boot(sid, boot):
    session = _sessions.get(sid)
    if session is None or not boot:
        return
    with _session_transport_lock:
        members = session.get("host_bound_subscribers", {})
        for peer, member in list(members.items()):
            if member["boot"] != boot:
                _detach_session_transport(session, peer)


def _host_forget_pending_turn(supervisor, session, turn_id):
    with session["history_lock"]:
        session.get("_conditional_pending_turns", {}).pop(turn_id, None)
    supervisor.conditional_untrack_turn(turn_id)


def _host_admission_allowed(frame):
    with _sessions_lock:
        session = _sessions.get(frame.get("sid"))
        pending = (session or {}).get("_conditional_pending_turns", {}).get(frame.get("turn_id"))
    if session is None or pending is None or pending["boot"] != frame.get("boot_id"):
        return False
    with _host_parent_cut(pending["params"], session, pending["peer"], admission=True) as valid:
        return valid and not session.get("_turn_cancel_requested") and _host_existing_authority(session) is not None


def _host_membership_allowed(frame):
    """Renew only exact registered parent peers; no release tombstones are needed."""
    accepted = []
    with _sessions_lock, _session_transport_lock:
        for entry in frame.get("members", []):
            session = _sessions.get(entry.get("sid"))
            origin = (session or {}).get("creation_binding")
            if origin is None or origin.runtime_record is not session:
                continue
            for peer, member in session.get("host_bound_subscribers", {}).items():
                if (member["boot"] == frame.get("boot_id")
                        and member["subscription"] == entry.get("subscription")
                        and member["supervisor"].conditional_boot() == member["boot"]
                        and _transport_is_live_peer(peer) and _session_transport_contains(session, peer)
                        and session.get("bound_subscribers", {}).get(peer) == origin.fields_for(
                            _transport_auth_user_id(peer), origin.store_path).get("creation_binding")):
                    accepted.append(member["subscription"])
    return accepted
