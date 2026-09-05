"""Compute-host (turn isolation) bridge: relay prompts/controls to the child process and
mirror its metadata/clarify/compress acks back into the session. Bodies are rebound onto
server.py's globals at install time (method_ctx.bind_module), so they use them bare."""

from __future__ import annotations

import contextlib
import threading

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()

_compute_host_supervisor = None
_compute_host_supervisor_lock = threading.Lock()
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


def _get_compute_host_supervisor(cfg: dict | None = None):
    global _compute_host_supervisor
    isolation_cfg = cfg or _load_dashboard_process_isolation_config()
    with _compute_host_supervisor_lock:
        if _compute_host_supervisor is None:
            from tui_gateway.host_supervisor import HostSupervisor
            _compute_host_supervisor = HostSupervisor(
                rpc_sink=_relay_compute_host_rpc,
                heartbeat_secs=int(isolation_cfg.get("compute_host_heartbeat_secs") or 15),
                respawn_max=int(isolation_cfg.get("compute_host_respawn_max") or 3))
        return _compute_host_supervisor


def _compute_host_turn_frame(
    rid: str, sid: str, session: dict, text: Any, image_paths: list[str] | None = None,
    queued_prompt_generation: int | None = None, display_kind: str | None = None) -> dict:
    with session["history_lock"]:
        history = list(session.get("history", []))
        history_version = int(session.get("history_version", 0))
        attached_images = list(image_paths if image_paths is not None else session.get("attached_images", []))
    return {
        "type": "turn.start", "sid": sid, "request_id": rid,
        "session_key": session.get("session_key") or sid, "text": text,
        **({"display_kind": display_kind} if display_kind else {}), "history": history,
        "history_version": history_version, "cols": int(session.get("cols", 80) or 80),
        "cwd": _session_cwd(session),
        "context_cwd_is_launch_artifact": _context_cwd_is_launch_artifact(session),
        "profile_home": session.get("profile_home") or "",
        "model_override": session.get("model_override"),
        "reasoning_config_override": session.get("create_reasoning_override"),
        "service_tier_override": session.get("create_service_tier_override"),
        "source": _session_source(session), "attached_images": attached_images,
        "queued_prompt_generation": queued_prompt_generation,
        # #101416: vouch that this process already holds the registry lease for this session, so
        # the child adopts it as an inert token instead of re-claiming and being fenced out by
        # our own entry ("Session ... already has a live owner"). No lease held = no vouch, and
        # the child keeps its legacy self-claim path (fail-closed refusal on conflict).
        "parent_owns_active_session_lease": session.get("active_session_lease") is not None}


def _metadata_mirror(session: dict | None) -> dict:
    mirror = (session or {}).get("_metadata_mirror")
    return mirror if isinstance(mirror, dict) else {}


def _compute_host_session_info(session: dict) -> dict:
    return _session_info(session.get("agent"), session)


def _compute_host_adopt_frame_meta(session: dict, frame: dict) -> None:
    """Adopt a host frame's session_key / history_version. Caller holds history_lock."""
    if frame.get("session_key"):
        session["session_key"] = str(frame.get("session_key"))
    if frame.get("history_version") is not None:
        with contextlib.suppress(Exception):
            session["history_version"] = max(int(session.get("history_version", 0)),
                                             int(frame.get("history_version") or 0))


def _relay_compute_host_rpc(message: dict) -> bool:
    """Relay host events while retaining the clarify snapshot needed on resume."""
    params = message.get("params") if isinstance(message, dict) else None
    kind = params.get("type") if isinstance(params, dict) else None
    if kind in {"clarify.request", "clarify.expire"}:
        session = _sessions.get(str(params.get("session_id") or ""))
        payload = params.get("payload")
        request_id = payload.get("request_id") if isinstance(payload, dict) else None
        if session is not None and request_id:
            with _history_lock(session):
                if kind == "clarify.request":
                    session["_compute_host_pending_clarify"] = dict(payload)
                elif _pending_clarify_matches(session, request_id):
                    session.pop("_compute_host_pending_clarify", None)
    return write_json(message)


def _history_lock(session: dict):
    return session.get("history_lock", threading.Lock())


def _pending_clarify_matches(session: dict, request_id) -> bool:
    """Whether ``session``'s mirrored pending clarify is ``request_id``. Caller holds
    history_lock."""
    pending = session.get("_compute_host_pending_clarify")
    return isinstance(pending, dict) and pending.get("request_id") == request_id


def _compute_host_clarify_session(request_id: str) -> tuple[str, dict] | None:
    """Find the parent mirror for one host-owned clarify request."""
    for sid, session in list(_sessions.items()) if request_id else ():
        with _history_lock(session):
            if _pending_clarify_matches(session, request_id):
                return sid, session
    return None


def _update_compute_host_clarify_snapshot(sid: str, session: dict, params: dict, result: dict) -> None:
    """Keep reconnect snapshots accurate while a batch clarify is answered."""
    request_id = str(params.get("request_id") or "")
    question_id = str(params.get("question_id") or "")
    with _history_lock(session):
        if not _pending_clarify_matches(session, request_id):
            return
        pending = session["_compute_host_pending_clarify"]
        if result.get("status") == "expired" or not result.get("remaining") and not question_id:
            session.pop("_compute_host_pending_clarify", None)
        elif question_id and isinstance(result.get("remaining"), list):
            pending["answers"] = {**(pending.get("answers") or {}),
                                  question_id: str(params.get("answer") or "")}
            if not result["remaining"]:
                session.pop("_compute_host_pending_clarify", None)


def _respond_compute_host_clarify(rid: str, params: dict) -> dict | None:
    """Proxy a clarify answer into the process that owns its pending Event."""
    located = _compute_host_clarify_session(str(params.get("request_id") or ""))
    if located is None or not _session_uses_compute_host(located[1]):
        return None
    sid, session = located
    try:
        ack = _get_compute_host_supervisor().respond(sid, params)
    except Exception as exc:
        return _err(rid, 5019, f"compute-host clarify response failed: {exc}")
    if ack.get("type") == "respond.error":
        return _err(rid, 5019, str(ack.get("message") or "compute-host clarify response failed"))
    response = ack.get("response")
    if not isinstance(response, dict):
        return _err(rid, 5019, "compute-host clarify response returned an invalid response")
    if "error" in response:
        error = response["error"] if isinstance(response["error"], dict) else {}
        return _err(rid, int(error.get("code") or 5000),
                    str(error.get("message") or "clarify response failed"))
    result = response.get("result")
    if not isinstance(result, dict):
        return _err(rid, 5019, "compute-host clarify response returned an invalid result")
    _update_compute_host_clarify_snapshot(sid, session, params, result)
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
        session.pop("_compute_host_pending_clarify", None)
    if frame.get("type") == "turn.error":
        message = str(frame.get("message") or "compute host turn failed")
        _emit("message.complete", sid, {"text": f"Error: {message}", "status": "error"})
    _apply_compute_host_metadata_mirror(session, frame)
    info = _compute_host_session_info(session)
    if not frame.get("session_info_emitted"):
        _emit("session.info", sid, info)
    _drain_queued_prompt(rid, sid, session)


def _submit_prompt_to_compute_host(
    rid: str, sid: str, session: dict, text: Any, image_paths: list[str] | None = None,
    queued_prompt_generation: int | None = None, display_kind: str | None = None) -> dict:
    cfg = _load_dashboard_process_isolation_config()
    frame = _compute_host_turn_frame(rid, sid, session, text, image_paths=image_paths,
                                     queued_prompt_generation=queued_prompt_generation,
                                     display_kind=display_kind)

    def _complete(done: dict) -> None:
        # submit_turn reports a synchronous pipe failure via the callback before re-raising;
        # leave the session untouched so prompt.submit can fail open to the in-process path.
        if done.get("reason") != "send_failed":
            _on_compute_host_turn_done(rid, sid, session, done)
    try:
        _get_compute_host_supervisor(cfg).submit_turn(frame, on_complete=_complete)
    except Exception as exc:
        return _err(rid, 5019, f"compute-host dispatch failed: {exc}")
    with session["history_lock"]:
        session["_compute_host_active"] = True
        if image_paths is None:
            session["attached_images"] = []
    return _ok(rid, {"status": "streaming", "turn_isolation": True})


def _send_compute_host_control(
    sid: str, *, route_name: str, command: str = "", payload: dict | None = None,
    wait: bool = True, timeout: float = 30.0, on_late_ack=None) -> dict:
    frame = dict(payload or {})
    frame.setdefault("type", "control")
    frame.setdefault("command", command)
    return _get_compute_host_supervisor().control(
        sid, route_name=route_name, payload=frame, wait=wait, timeout=timeout,
        on_late_ack=on_late_ack)


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
