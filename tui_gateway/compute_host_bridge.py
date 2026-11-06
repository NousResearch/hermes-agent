"""Compute-host (turn isolation) bridge: relay prompts/controls to the child process and
mirror its metadata/clarify/compress acks back into the session. Bodies are rebound onto
server.py's globals at install time (method_ctx.bind_module), so they use them bare."""

from __future__ import annotations

import contextlib
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
import threading

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()

_compute_host_supervisors: dict[str, object] = {}
_compute_host_supervisor_lock = threading.Lock()
# A target is marked while its last-session shutdown runs outside ``_sessions_lock``.  This
# prevents a newly-created session for the same profile from attaching to a supervisor whose
# process and registry are being torn down, while also keeping callbacks free to acquire the
# session lock during shutdown.
_compute_host_supervisor_retiring: set[str] = set()
_compute_host_target_context: ContextVar[object | None] = ContextVar(
    "compute_host_target_context", default=None)
# Cap on how long session.compress blocks its RPC on the compute host. Must stay
# below the desktop's SESSION_COMPRESS_TIMEOUT_MS (660s) so the client gets the
# `pending` answer, not its own timeout; the late-ack path covers anything slower.
# See #97948.
_COMPUTE_HOST_COMPRESS_WAIT_CAP_SECS = 630.0


def _turn_isolation_enabled(cfg: dict | None = None) -> bool:
    if os.environ.get("HERMES_COMPUTE_HOST_CHILD") == "1":
        return False
    return bool((cfg or _load_dashboard_process_isolation_config()).get("turn_isolation"))


def _session_uses_compute_host(session: dict, cfg: dict | None = None) -> bool:
    # Routes lazy sessions whose AIAgent was never built in-process; already-built
    # sessions keep the in-process path unless a prior isolated turn marked host ownership.
    return _turn_isolation_enabled(cfg) and (
        bool(session.get("_compute_host_active"))
        or (session.get("agent") is None and session.get("agent_ready") is not None))


@dataclass(frozen=True)
class _ComputeHostTarget:
    """The immutable parent-side binding for one compute-host operation."""

    key: str
    home: Path
    sid: str
    session: dict


class _ComputeHostTargetError(RuntimeError):
    """A host operation could not be bound to a live, authoritative profile."""


def _compute_host_home_key(session: dict) -> tuple[str, Path]:
    raw_home = session.get("profile_home") or _hermes_home
    try:
        home = Path(raw_home).expanduser().resolve()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise _ComputeHostTargetError("compute-host profile home is invalid") from exc
    if not home.is_dir():
        raise _ComputeHostTargetError("compute-host profile home is unavailable")
    return str(home), home


def _compute_host_target_key_matches(target_key: str | None, session: dict) -> bool:
    if target_key is None:
        return True
    try:
        return _compute_host_home_key(session)[0] == target_key
    except _ComputeHostTargetError:
        return False


def _compute_host_target(sid: str, session: dict | None = None) -> _ComputeHostTarget:
    """Resolve the live session's profile home without trusting RPC or ambient process state."""
    with _sessions_lock:
        live = _sessions.get(sid)
        if live is None or (session is not None and live is not session):
            raise _ComputeHostTargetError("compute-host session is unavailable")
        key, home = _compute_host_home_key(live)
        return _ComputeHostTarget(key=key, home=home, sid=str(sid), session=live)


def _compute_host_detached_target(sid: str, session: dict) -> _ComputeHostTarget:
    """Bind a completion callback to a session object after close detached it from _sessions."""
    key, home = _compute_host_home_key(session)
    return _ComputeHostTarget(key=key, home=home, sid=str(sid), session=session)


def _compute_host_target_matches(target: _ComputeHostTarget, sid: str, session: dict, *, allow_detached=False) -> bool:
    if str(sid) != target.sid or session is not target.session:
        return False
    with _sessions_lock:
        live = _sessions.get(sid)
        if live is not None and live is not session:
            # A callback from an old session must never touch a replacement using the same sid.
            return False
        if live is None and (not allow_detached or not session.get("_closing")):
            return False
    try:
        key, _home = _compute_host_home_key(session)
    except _ComputeHostTargetError:
        return False
    return key == target.key


def _get_compute_host_supervisor(cfg: dict | None = None):
    """Return the supervisor bound by the current operation's target context."""
    isolation_cfg = cfg or _load_dashboard_process_isolation_config()
    target = _compute_host_target_context.get()
    if target is None:
        raise _ComputeHostTargetError("compute-host target is required")
    assert isinstance(target, _ComputeHostTarget)
    with _compute_host_supervisor_lock:
        if target.key in _compute_host_supervisor_retiring:
            raise _ComputeHostTargetError("compute-host target is retiring")
        supervisor = _compute_host_supervisors.get(target.key)
        if supervisor is None:
            from tui_gateway.host_supervisor import HostSupervisor
            from tools.environments.local import served_profile_child_env
            env = served_profile_child_env(target_home=target.home, inherit_credentials=True)

            def _target_rpc_sink(message: dict, *, _key=target.key) -> bool:
                return _relay_compute_host_rpc(message, target_key=_key)

            supervisor = HostSupervisor(
                registry_path=target.home / "state" / "dashboard-compute-host.json",
                env=env,
                expected_hermes_home=str(target.home),
                rpc_sink=_target_rpc_sink,
                heartbeat_secs=int(isolation_cfg.get("compute_host_heartbeat_secs") or 15),
                respawn_max=int(isolation_cfg.get("compute_host_respawn_max") or 3))
            # If the last session detached while a host callback was running, the close path
            # could not retire the target yet.  The supervisor retries after its callback count
            # reaches zero, while preserving the per-home pool for any sibling session.
            supervisor._on_callbacks_idle = lambda _target=target: _retire_compute_host_target(
                _target.sid, _target.session)
            _compute_host_supervisors[target.key] = supervisor
        return supervisor


def _compute_host_target_and_supervisor(sid: str, session: dict | None = None, cfg: dict | None = None):
    target = _compute_host_target(sid, session)
    token = _compute_host_target_context.set(target)
    try:
        return target, _get_compute_host_supervisor(cfg)
    finally:
        _compute_host_target_context.reset(token)


def _compute_host_supervisor_for_session(sid: str, session: dict | None = None, cfg: dict | None = None):
    return _compute_host_target_and_supervisor(sid, session, cfg)[1]


def _capture_compute_host_close_owner(sid: str, session: dict) -> None:
    """Capture the existing target owner before close detaches a session from the live registry."""
    if not _session_uses_compute_host(session):
        return
    try:
        target = _compute_host_target(sid, session)
    except _ComputeHostTargetError:
        return
    with _compute_host_supervisor_lock:
        supervisor = _compute_host_supervisors.get(target.key)
    if supervisor is None:
        return
    session["_compute_host_close_target"] = target
    session["_compute_host_close_supervisor"] = supervisor


def _compute_host_close_owner(session: dict, sid: str):
    """Return the captured owner only while its detached session and pool binding still match."""
    target = session.get("_compute_host_close_target")
    supervisor = session.get("_compute_host_close_supervisor")
    if not isinstance(target, _ComputeHostTarget) or supervisor is None:
        return None
    if not _compute_host_target_matches(target, sid, session, allow_detached=True):
        return None
    with _compute_host_supervisor_lock:
        if (target.key in _compute_host_supervisor_retiring
                or _compute_host_supervisors.get(target.key) is not supervisor):
            return None
    return supervisor


def _retire_compute_host_target(sid: str, session: dict) -> None:
    """Shutdown one target only after its last session and all host callbacks are gone."""
    try:
        target = _compute_host_detached_target(sid, session)
    except _ComputeHostTargetError:
        return
    # The close path must interrupt this still-admitted turn before any supervisor shutdown.  The
    # callback will retry retirement after it clears the turn, so a coarse ``has_pending_work``
    # result cannot retire the exact owner in the interim.
    if session.get("_compute_host_turn_id"):
        return
    # Claim retirement while holding the locks, then release ``_sessions_lock`` before the
    # potentially blocking process shutdown.  The retiring marker closes the attach race for a
    # new session with the same profile home; callbacks can therefore reacquire the session lock
    # without deadlocking the close path.
    with _sessions_lock:
        if any(_compute_host_target_key_matches(target.key, live) for live in _sessions.values()):
            return
        with _compute_host_supervisor_lock:
            if target.key in _compute_host_supervisor_retiring:
                return
            supervisor = _compute_host_supervisors.get(target.key)
            if supervisor is None or supervisor.has_pending_work():
                return
            _compute_host_supervisor_retiring.add(target.key)
    shutdown_succeeded = False
    try:
        supervisor.shutdown()
        shutdown_succeeded = True
    except Exception:
        # Leave the target in the pool for a later retry; a failed close is not proof of cleanup.
        logger.warning("compute-host target shutdown failed home=%s", target.home, exc_info=True)
        return
    finally:
        with _compute_host_supervisor_lock:
            _compute_host_supervisor_retiring.discard(target.key)
            if shutdown_succeeded and _compute_host_supervisors.get(target.key) is supervisor:
                # A successful shutdown is the only point at which this target may be removed.
                # A failed close leaves its exact supervisor and registry available for retry.
                _compute_host_supervisors.pop(target.key, None)


def _compute_host_turn_frame(
    rid: str, sid: str, session: dict, text: Any, image_paths: list[str] | None = None,
    queued_prompt_generation: int | None = None, display_kind: str | None = None,
    display_metadata: dict | None = None) -> dict:
    with session["history_lock"]:
        history = list(session.get("history", []))
        history_version = int(session.get("history_version", 0))
        attached_images = list(image_paths if image_paths is not None else session.get("attached_images", []))
    return {
        "type": "turn.start", "sid": sid, "request_id": rid,
        "session_key": session.get("session_key") or sid, "text": text,
        **({"display_kind": display_kind} if display_kind else {}), "history": history,
        **({"display_metadata": display_metadata} if display_metadata else {}),
        "history_version": history_version, "cols": int(session.get("cols", 80) or 80),
        "cwd": _session_cwd(session),
        # Workspace provenance rides with the cwd: a sandbox backend registers an unpicked host cwd as nothing.
        "explicit_cwd": bool(session.get("explicit_cwd")),
        "context_cwd_is_launch_artifact": _context_cwd_is_launch_artifact(session),
        "profile_home": session.get("profile_home") or "",
        "model_override": session.get("model_override"),
        # A model switch queued while the session was busy (config.set model ->
        # pending_model_switch) must cross the process boundary — the live agent is in
        # the compute host, and the server-side _apply_pending_model_switch would
        # pop-and-drop it against agent=None. COPIED (not popped): the stash is
        # cleared only after a successful isolated turn (_on_compute_host_turn_done),
        # so the fail-open in-process path can still apply it if the host dispatch fails.
        "pending_model_switch": session.get("pending_model_switch"),
        "reasoning_config_override": session.get("create_reasoning_override"),
        "service_tier_override": session.get("create_service_tier_override"),
        "source": _session_source(session), "attached_images": attached_images,
        "auth_user_id": _session_auth_user_id(session),
        "queued_prompt_generation": queued_prompt_generation,
        # #101416: vouch that this process already holds the registry lease for this session, so
        # the child adopts it as an inert token instead of re-claiming and being fenced out by
        # our own entry ("Session ... already has a live owner"). No lease held = no vouch, and
        # the child keeps its legacy self-claim path (fail-closed refusal on conflict).
        "active_session_lease": _active_session_lease_vouch(session)}


def _active_session_lease_vouch(session: dict) -> dict | None:
    """``{lease_id, session_id}`` of the REAL registry lease this process holds for the session's
    current stored id, else None. Qualified, not a bare bool: after a compression rotation a lease
    still keyed on the old id must not let a (replacement) child borrow the continuation."""
    lease = session.get("active_session_lease")
    if lease is None or getattr(lease, "released", False) or not getattr(lease, "enabled", False):
        return None
    if str(lease.session_id) != str(session.get("session_key") or ""):
        return None
    return {"lease_id": str(lease.lease_id), "session_id": str(lease.session_id)}


def _metadata_mirror(session: dict | None) -> dict:
    mirror = (session or {}).get("_metadata_mirror")
    return mirror if isinstance(mirror, dict) else {}


def _compute_host_session_info(session: dict) -> dict:
    return _session_info(session.get("agent"), session)


def _compute_host_adopt_frame_meta(session: dict, frame: dict) -> None:
    """Adopt a host frame's session_key / history_version. Caller holds history_lock.

    A rotated ``session_key`` means the child compressed A->B. The child only holds an inert
    borrowed token (``_install_borrowed_lease``), so the REAL registry lease is re-anchored here,
    by its owner — never claimed from the child pid (authority stays singular, #103737 review)."""
    new_key = str(frame.get("session_key") or "")
    if new_key and new_key != str(session.get("session_key") or ""):
        if not _transfer_active_session_slot(str(frame.get("sid") or ""), session, new_session_id=new_key):
            logger.warning("Compression session lease did not re-anchor: sid=%s old_session_id=%s new_session_id=%s",
                           frame.get("sid"), session.get("session_key"), new_key)
        session["session_key"] = new_key
    if frame.get("history_version") is not None:
        with contextlib.suppress(Exception):
            session["history_version"] = max(int(session.get("history_version", 0)),
                                             int(frame.get("history_version") or 0))


def _write_target_compute_host_rpc(
    message: dict, target_key: str, session_id: str, session: dict,
) -> bool:
    """Capture target custody and mirror state before releasing the registry lock, then write to it."""
    from collections.abc import Mapping
    from tui_gateway.hosted_room_member_activity import _COORDINATE_FIELDS

    with _sessions_lock:
        if (_sessions.get(session_id) is not session
                or not _compute_host_target_key_matches(target_key, session)):
            return False
        transport = session.get("transport")
        if transport is None:
            return False
        params = message.get("params") if isinstance(message, dict) else None
        method = message.get("method") if isinstance(message, dict) else None
        if isinstance(message, dict) and isinstance(message.get("id"), str) and method not in (None, "event"):
            with _history_lock(session):
                session["_compute_host_open_request"] = {
                    "id": message["id"], "method": message["method"], "params": dict(params)}
        elif isinstance(params, dict) and params.get("type") == "request.cancel":
            payload = params.get("payload")
            if isinstance(payload, dict):
                with _history_lock(session):
                    if _open_request_matches(session, payload.get("id")):
                        session.pop("_compute_host_open_request", None)
        # Hosted-room writers update this task under the session history lock. Capture only the
        # six scalar coordinates needed by projection while that lock is held; passing the live
        # session outside the registry lock would let a concurrent turn/retirement mutate the
        # room target during plugin discovery or transport I/O.
        with _history_lock(session):
            hosted_task = session.get("_hosted_room_task")
            hosted_task_snapshot = (
                {field: hosted_task.get(field) for field in _COORDINATE_FIELDS}
                if isinstance(hosted_task, Mapping) else None
            )
        # Detach a minimal one-session projection mapping. The potentially blocking replay,
        # observer work, and transport I/O happen after the registry lock is released.
        projection_sessions = (
            {session_id: {"_hosted_room_task": hosted_task_snapshot}}
            if hosted_task_snapshot is not None else {}
        )
    from tui_gateway.event_replay import _stamp_event
    from tui_gateway.hosted_room_member_activity import project_room_member_activity
    _stamp_event(message)
    project_room_member_activity(message, projection_sessions)
    return transport.write(message)


def _broadcast_target_compute_host_global_event(
    target_key: str, event: str, payload: dict | None,
) -> bool:
    """Fan a child-global event only to transports captured from sessions in its profile home."""
    with _sessions_lock:
        targets = []
        seen: set[int] = set()
        for live in _sessions.values():
            if not _compute_host_target_key_matches(target_key, live):
                continue
            transport = live.get("transport")
            if transport is None or transport is _detached_ws_transport:
                continue
            for peer in (_session_live_transports(live)
                         if transport is not _stdio_transport else [transport]):
                if peer is not None and id(peer) not in seen:
                    seen.add(id(peer))
                    targets.append(peer)
    if not targets:
        return False
    frame = _event_frame(event, "", payload)
    for transport in targets:
        try:
            transport.write(frame)
        except Exception:
            logger.debug("targeted global-event broadcast failed type=%s", event, exc_info=True)
    return True


def _relay_compute_host_rpc(message: dict, *, target_key: str | None = None) -> bool:
    """Relay host frames to the client while mirroring the server→client request the host has open, so a
    reconnecting client gets it back through ``open_requests``."""
    params = message.get("params") if isinstance(message, dict) else None
    session_id = str(params.get("session_id") or "") if isinstance(params, dict) else ""
    session = _sessions.get(session_id) if session_id else None
    method = message.get("method") if isinstance(message, dict) else None
    is_global_event = method == "event" and isinstance(params, dict) and not session_id
    is_session_bound = bool(session_id) or (
        isinstance(message, dict)
        and isinstance(message.get("id"), str)
        and method not in (None, "event")
    ) or (isinstance(params, dict) and params.get("type") == "request.cancel")
    if target_key is not None and is_session_bound and not is_global_event:
        # A target-scoped child may only emit into a live session that resolves to its own
        # canonical home.  Do this before mirror mutation and before write_json so a stale or
        # cross-profile frame cannot reach another profile's transport.
        if session is None or not _compute_host_target_key_matches(target_key, session):
            return False
    elif target_key is not None and not is_global_event and not session_id:
        # Unknown target-scoped frames are fail-closed; only explicitly session-less events are
        # allowed to use the global broadcast path below.
        return False
    if isinstance(message, dict) and message.get("method") == "compute_host.activity":
        if session is not None:
            if target_key is not None:
                with _sessions_lock:
                    if (_sessions.get(session_id) is not session
                            or not _compute_host_target_key_matches(target_key, session)):
                        return False
                    with _history_lock(session):
                        if (session.get("running") and params.get("turn_id")
                                and session.get("_compute_host_turn_id") == params["turn_id"]):
                            session["_compute_host_activity_ns"] = params.get("activity_ns")
            elif _compute_host_target_key_matches(target_key, session):
                with _history_lock(session):
                    if (session.get("running") and params.get("turn_id")
                            and session.get("_compute_host_turn_id") == params["turn_id"]):
                        session["_compute_host_activity_ns"] = params.get("activity_ns")
        return True  # Internal observation, not a client event or replay entry.
    if is_global_event:
        # A target-scoped child only fans out to transports owned by its canonical profile home;
        # an unscoped local emitter retains the existing process-global broadcast behavior.
        if target_key is not None:
            return _broadcast_target_compute_host_global_event(
                target_key, str(params.get("type") or ""), params.get("payload"))
        _broadcast_global_event(str(params.get("type") or ""), params.get("payload"))
        return True
    if target_key is not None and is_session_bound:
        return _write_target_compute_host_rpc(message, target_key, session_id, session)
    if isinstance(message, dict) and isinstance(message.get("id"), str) and method not in (None, "event"):
        # A server request minted by the child: remember it against its session until it is answered/withdrawn.
        if session is not None and (target_key is None or _compute_host_target_key_matches(target_key, session)):
            with _history_lock(session):
                session["_compute_host_open_request"] = {
                    "id": message["id"], "method": message["method"], "params": dict(params)}
    elif isinstance(params, dict) and params.get("type") == "request.cancel":
        payload = params.get("payload")
        if (session is not None and (target_key is None or _compute_host_target_key_matches(target_key, session))
                and isinstance(payload, dict)):
            with _history_lock(session):
                if _open_request_matches(session, payload.get("id")):
                    session.pop("_compute_host_open_request", None)
    return write_json(message)


def _history_lock(session: dict):
    return session.get("history_lock", threading.Lock())


def _open_request_matches(session: dict, request_id) -> bool:
    """Whether ``session``'s mirrored open request is ``request_id``. Caller holds history_lock."""
    mirrored = session.get("_compute_host_open_request")
    return isinstance(mirrored, dict) and mirrored.get("id") == request_id


def _compute_host_request_session(request_id: str) -> tuple[str, dict] | None:
    """Find the parent mirror for one host-owned server request."""
    for sid, session in list(_sessions.items()) if request_id else ():
        with _history_lock(session):
            if _open_request_matches(session, request_id):
                return sid, session
    return None


def _relay_compute_host_response(frame: dict) -> bool:
    """Forward a client's response frame to the compute-host child that owns the request. False when no
    child owns that id."""
    located = _compute_host_request_session(str(frame.get("id") or ""))
    if located is None or not _session_uses_compute_host(located[1]):
        return False
    sid, session = located
    with _history_lock(session):
        session.pop("_compute_host_open_request", None)
    try:
        _compute_host_supervisor_for_session(sid, session).respond(sid, {"frame": dict(frame)})
    except Exception:
        logger.debug("compute-host response relay failed sid=%s", sid, exc_info=True)
    return True


def _cancel_session_requests(sid: str, reason: str, *, method: str) -> None:
    """Withdraw *sid*'s open *method* requests wherever they wait: in this process, or in the compute-host
    child that minted them (the parent holds only a mirror, so a local cancel never reaches that wait).
    The child's ``request.cancel`` comes back through ``_relay_compute_host_rpc``, which drops the mirror."""
    from tui_gateway import server_requests
    server_requests.cancel(sid, reason, method=method)
    session = _sessions.get(sid)
    if session is None:
        return
    with _history_lock(session):
        mirrored = session.get("_compute_host_open_request")
        hosted = isinstance(mirrored, dict) and mirrored.get("method") == method
    if not hosted or not _session_uses_compute_host(session):
        return
    try:
        _compute_host_supervisor_for_session(sid, session).respond(
            sid, {"cancel": {"method": method, "reason": reason}})
    except Exception:  # health: allow BLE001 -- the bridge raises transport-specific errors unknown here; the child's own deadline still ends the wait
        logger.warning("compute-host request cancel failed sid=%s method=%s", sid, method, exc_info=True)


def _lock_compute_host_clarify(rid: str, request_id: str, question_id: str, answer: str | None) -> dict | None:
    """Proxy a batch-clarify lock into the child that owns the request; keeps the parent mirror's locked
    answers current for reconnect snapshots. None when the request is not host-owned."""
    located = _compute_host_request_session(request_id)
    if located is None or not _session_uses_compute_host(located[1]):
        return None
    sid, session = located
    try:
        ack = _compute_host_supervisor_for_session(sid, session).respond(
            sid, {"lock": {"request_id": request_id, "question_id": question_id, "answer": answer}})
    except Exception as exc:
        return _err(rid, 5019, f"compute-host clarify lock failed: {exc}")
    if ack.get("type") == "respond.error":
        return _err(rid, 5019, str(ack.get("message") or "compute-host clarify lock failed"))
    response = ack.get("response")
    if not isinstance(response, dict):
        return _err(rid, 5019, "compute-host clarify lock returned an invalid response")
    if "error" in response:
        error = response["error"] if isinstance(response["error"], dict) else {}
        return _err(rid, int(error.get("code") or 5000), str(error.get("message") or "clarify lock failed"))
    result = response.get("result") if isinstance(response.get("result"), dict) else {}
    with _history_lock(session):
        if _open_request_matches(session, request_id):
            mirrored = session["_compute_host_open_request"]
            if result.get("status") == "expired" or result.get("remaining") == []:
                session.pop("_compute_host_open_request", None)
            else:
                mirrored["params"]["answers"] = {**(mirrored["params"].get("answers") or {}), question_id: answer}
    return _ok(rid, result)


def _apply_compute_host_metadata_mirror(session: dict, frame: dict | None) -> None:
    """Mirror host-owned session metadata: under turn isolation the host is the only
    writer of live agent/history state, and UI reads must not build a second agent."""
    if not isinstance(frame, dict):
        return
    with _history_lock(session):
        _compute_host_adopt_frame_meta(session, frame)
        if frame.get("message_count") is not None:
            with contextlib.suppress(Exception):
                session["_metadata_message_count"] = int(frame.get("message_count") or 0)
    info = frame.get("session_info")
    if isinstance(info, dict):
        session["_metadata_mirror"] = {**_metadata_mirror(session), **info}
        session["_metadata_mirror_updated_at"] = time.time()


def _on_compute_host_turn_done(rid: str, sid: str, session: dict, frame: dict) -> None:
    with session["history_lock"]:
        _compute_host_adopt_frame_meta(session, frame)
        session["running"] = False
        session["last_active"] = time.time()
        _clear_inflight_turn(session)
        session.pop("_compute_host_open_request", None)
    # The isolated turn carried the queued model switch to the compute host, whose
    # turn thread applied it. Clear the server-side stash so it isn't re-forwarded
    # (kept on error so the fail-open in-process path can still apply it).
    if frame.get("type") != "turn.error":
        session.pop("pending_model_switch", None)
    if frame.get("type") == "turn.error":
        message = str(frame.get("message") or "compute host turn failed")
        _emit("message.complete", sid, {"text": f"Error: {message}", "status": "error"})
    _apply_compute_host_metadata_mirror(session, frame)
    # Settlement of a turn whose session was closed mid-flight: the real lease was held for it.
    _release_deferred_active_session_lease(session)
    info = _compute_host_session_info(session)
    if not frame.get("session_info_emitted"):
        _emit("session.info", sid, info)
    _drain_queued_prompt(rid, sid, session)


def _submit_prompt_to_compute_host(
    rid: str, sid: str, session: dict, text: Any, image_paths: list[str] | None = None,
    queued_prompt_generation: int | None = None, display_kind: str | None = None,
    display_metadata: dict | None = None) -> dict:
    cfg = _load_dashboard_process_isolation_config()
    frame = _compute_host_turn_frame(rid, sid, session, text, image_paths=image_paths,
                                     queued_prompt_generation=queued_prompt_generation,
                                     display_kind=display_kind, display_metadata=display_metadata)
    # Caller JSON-RPC ids may repeat across sockets and turns. Use an opaque
    # dispatch lifetime token, installed before a fast child can send activity.
    turn_id = frame["turn_id"] = frame["request_id"] = uuid.uuid4().hex
    with session["history_lock"]:
        session["_compute_host_turn_id"] = turn_id
        session.pop("_compute_host_activity_ns", None)

    def _complete(done: dict) -> None:
        # submit_turn reports a synchronous pipe failure via the callback before re-raising;
        # leave the session untouched so prompt.submit can fail open to the in-process path.
        if done.get("reason") != "send_failed":
            with session["history_lock"]:
                if session.get("_compute_host_turn_id") != turn_id:
                    return
                session.pop("_compute_host_turn_id", None)
                session.pop("_compute_host_activity_ns", None)
            _on_compute_host_turn_done(rid, sid, session, done)
    try:
        target, supervisor = _compute_host_target_and_supervisor(sid, session, cfg)
        _complete_target = target

        def _target_complete(done: dict) -> None:
            if not _compute_host_target_matches(_complete_target, sid, session, allow_detached=True):
                return
            try:
                _complete(done)
            finally:
                # A close may detach the session while its last host turn is still producing a
                # terminal frame.  Retry target retirement after that callback drains, rather than
                # leaving an idle per-home child behind from the earlier close attempt.
                if session.get("_closing"):
                    _retire_compute_host_target(sid, session)

        supervisor.submit_turn(frame, on_complete=_target_complete)
    except Exception as exc:
        with session["history_lock"]:
            if session.get("_compute_host_turn_id") == turn_id:
                session.pop("_compute_host_turn_id", None)
                session.pop("_compute_host_activity_ns", None)
        return _err(rid, 5019, f"compute-host dispatch failed: {exc}")
    with session["history_lock"]:
        session["_compute_host_active"] = True
        if image_paths is None:
            session["attached_images"] = []
    return _ok(rid, {"status": "streaming", "turn_isolation": True})


def _send_compute_host_control(
    sid: str, *, route_name: str, command: str = "", payload: dict | None = None,
    wait: bool = True, timeout: float = 30.0, on_late_ack=None) -> dict:
    target, supervisor = _compute_host_target_and_supervisor(sid)
    frame = dict(payload or {})
    frame.setdefault("type", "control")
    frame.setdefault("command", command)
    if on_late_ack is not None:
        original_late_ack = on_late_ack

        def _target_late_ack(ack: dict) -> None:
            if _compute_host_target_matches(target, sid, target.session):
                original_late_ack(ack)

        on_late_ack = _target_late_ack
    return supervisor.control(sid, route_name=route_name, payload=frame, wait=wait, timeout=timeout,
                             on_late_ack=on_late_ack)


def _compute_host_ack_error(rid, ack: dict, code: int, default: str):
    """``_err`` for a ``control.error``/``error`` ack, else None."""
    if ack.get("type") in {"control.error", "error"}:
        return _err(rid, code, str(ack.get("message") or default))
    return None


def _compute_host_compress_wait_seconds(cfg: dict | None = None) -> float:
    """RPC wait budget for a compute-host compress control: the configured compression
    ceiling plus slack, capped below the desktop's RPC timeout (a fixed waiter reported
    false timeouts while the host kept working); slower acks land via the late-ack path.

    See #97948.
    """
    from agent.conversation_compression import resolve_context_compression_timeouts
    try:
        compression_cfg = (cfg if cfg is not None else _load_cfg()).get("compression", {})
    except Exception:
        compression_cfg = {}
    if not isinstance(compression_cfg, dict):
        compression_cfg = {}
    _idle, ceiling = resolve_context_compression_timeouts(compression_cfg)
    return float(min(max(ceiling + 30.0, 120.0), _COMPUTE_HOST_COMPRESS_WAIT_CAP_SECS))


def _adopt_late_compute_host_compress_ack(sid: str, session: dict, ack: dict, *, route_name: str) -> None:
    """Adopt a compress ack that arrived after its RPC waiter answered ``pending``: the only place
    the rotated session_key / history_version / mirror can land and the client's only signal
    (the same ``session.info`` + ``compacted`` edges the in-process /compress path emits). A late
    ``control.error`` goes out via ``error``."""
    with _sessions_lock:
        if _sessions.get(sid) is not session:
            return
    if not isinstance(ack, dict) or ack.get("type") in {"control.error", "error"}:
        message = str((ack or {}).get("message") or f"compute-host {route_name} failed")
        _emit("error", sid, {"message": f"compression failed: {message}"})
        _status_update(sid, "ready")
        return
    _apply_compute_host_metadata_mirror(session, ack)
    _emit("session.info", sid, _compute_host_session_info(session))
    _status_update(sid, "compacted", "✓ Context compression complete")


def register(server) -> None:
    """Publish this module's helpers + handlers onto ``server``, rebound to its globals."""
    bind_module(globals(), server, skip=("_",))
