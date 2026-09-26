"""Server-side queued prompts as addressable items: ``session.queue.get`` / ``session.queue.update``.

A prompt submitted while a turn runs is held in ``queued_prompt`` + ``queued_prompts`` (head first,
``session_auto_continue._enqueue_prompt``). Each envelope carries a stable ``queue_id`` (client-chosen
via ``prompt.submit {queue_id}`` or server-minted) and the caller's opaque ``client_message_id``, so a
client that reconnects, or a second window, can list, edit, delete or steer a follow-up it did not hold
locally. ``revision`` advances whenever the visible item list changes; every change is published as a
``session.queue`` event to the session's clients plus a global ``session.queue.changed`` signal for
windows watching other sessions. Bodies are rebound onto server.py's globals (method_ctx.bind_module).
"""

from __future__ import annotations

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method

# Client identities are opaque; bound them so a queue item cannot carry a payload in its id.
_QUEUE_ID_MAX = 128
_CLIENT_MESSAGE_ID_MAX = 256
_QUEUE_ACTIONS = ("edit", "delete", "steer")


def _queued_envelopes(session: dict) -> list[dict]:
    """Every queued envelope in execution order (head first). Caller holds ``history_lock``."""
    head = session.get("queued_prompt")
    return [e for e in ([head] if head else []) + list(session.get("queued_prompts") or []) if isinstance(e, dict)]


def _queue_identity_param(params: dict, key: str, limit: int) -> tuple[str, str | None]:
    """``(value, error)`` for an optional opaque identity string."""
    raw = params.get(key)
    if raw is None:
        return "", None
    if not isinstance(raw, str) or len(raw) > limit:
        return "", f"{key} must be a string of at most {limit} characters"
    return raw.strip(), None


def _queue_envelope_id(envelope: dict) -> str:
    """The envelope's stable id, minted on first sight (envelopes queued by internal paths)."""
    queue_id = envelope.get("_queue_id")
    if not isinstance(queue_id, str) or not queue_id:
        queue_id = envelope["_queue_id"] = f"q-{uuid.uuid4().hex}"
    return queue_id


def _find_queued_envelope(session: dict, queue_id: str) -> dict | None:
    """Caller holds ``history_lock``."""
    return next((e for e in _queued_envelopes(session) if _queue_envelope_id(e) == queue_id), None)


def _queue_item(session: dict, envelope: dict) -> dict:
    raw = envelope.get("text")
    text = _inflight_text(raw)
    has_images = bool(envelope.get("image_paths"))
    # Only the sender's own plain text is editable: attachments are bound to the turn, and an
    # authored envelope (another sender, e.g. a relayed bot message) is not this client's to rewrite.
    plain = isinstance(raw, str) and bool(text.strip()) and not has_images and not envelope.get("turn_author")
    agent = session.get("agent")
    return {
        "id": _queue_envelope_id(envelope),
        "client_message_id": str(envelope.get("_client_message_id") or ""),
        "text": text,
        "has_images": has_images,
        "editable": plain,
        "steerable": plain and bool(session.get("running")) and agent is not None and hasattr(agent, "steer"),
    }


def _queue_snapshot_unlocked(session: dict) -> dict:
    """``{revision, items}``; the revision advances exactly when the visible list changed since the last
    snapshot, so every mutation path (enqueue, merge, drain, Stop, reset, scrub) is covered without each
    one having to remember to bump it. Caller holds ``history_lock``."""
    items = [_queue_item(session, e) for e in _queued_envelopes(session)]
    if items != session.get("_queue_items_seen", []):
        session["_queue_items_seen"] = items
        session["_queue_revision"] = int(session.get("_queue_revision", 0)) + 1
    return {"revision": int(session.get("_queue_revision", 0)), "items": [dict(i) for i in items]}


def _queue_snapshot(session: dict) -> dict:
    with session["history_lock"]:
        return _queue_snapshot_unlocked(session)


def _publish_queue(sid: str, session: dict | None) -> dict | None:
    """Emit ``session.queue`` (+ the global ``session.queue.changed``) when the queue moved since the last
    publish; a no-op otherwise, so callers may invoke it after any operation that might touch the queue.
    Never call under ``history_lock`` (emission writes to transports). Returns the current snapshot."""
    if not session or session.get("history_lock") is None:
        return None
    with session["history_lock"]:
        snapshot = _queue_snapshot_unlocked(session)
        changed = snapshot["revision"] != session.get("_queue_published_revision")
        session["_queue_published_revision"] = snapshot["revision"]
    if changed:
        # Best-effort like session.control.update: a dead transport must not fail the queue operation.
        try:
            _emit("session.queue", sid, snapshot)
            if session_key := str(session.get("session_key") or ""):
                _broadcast_global_event("session.queue.changed",
                                        {"session_key": session_key, "revision": snapshot["revision"]})
        except Exception:
            logger.debug("session.queue publish failed for %s", sid, exc_info=True)
    return snapshot


def _retire_queued_envelope_row(session: dict, envelope: dict) -> None:
    """Deactivate the accept-time user row of an envelope that will never run as its own turn (kept on
    disk, never deleted), mirroring the drain's re-placement. Caller holds ``history_lock``."""
    staged = envelope.get("_submit_user_row")
    if not (isinstance(staged, dict) and isinstance(staged.get("_row_id"), int)):
        return
    with _session_db(session) as db:
        if db is None:
            return
        try:
            db.deactivate_message(session.get("session_key"), staged["_row_id"])
        except Exception:
            logger.debug("queued-prompt row retire failed", exc_info=True)


def _remove_queued_envelope(session: dict, envelope: dict) -> int:
    """Drop *envelope* from the queue; returns its former index. Caller holds ``history_lock``."""
    entries = _queued_envelopes(session)
    index = next(i for i, e in enumerate(entries) if e is envelope)
    _ac_set_queue(session, entries[:index] + entries[index + 1:])
    return index


@method("session.queue.get")
def _(rid, params: dict) -> dict:
    """The authoritative queued prompts for a live session, in execution order."""
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    return _ok(rid, _publish_queue(str(params.get("session_id") or ""), session))


def _steer_queued(rid, sid: str, session: dict, queue_id: str) -> dict | None:
    """Promote one queued prompt into the running turn. Claim it first, then call ``steer`` outside
    ``history_lock`` (a provider call must never run under it); a refused steer puts it back in place."""
    agent = session.get("agent")
    if agent is None or not hasattr(agent, "steer"):
        return _err(rid, 4010, "agent does not support steer")
    # A correction that reaches the provider mid-compression aborts the compression (#61042).
    if _session_compression_in_flight(session):
        return _err(rid, 4009, "session is compressing; the queued prompt will run when it finishes")
    with session["history_lock"]:
        envelope = _find_queued_envelope(session, queue_id)
        if envelope is None:
            return _err(rid, 4041, "queued item not found")
        if not _queue_item(session, envelope)["steerable"]:
            return _err(rid, 4009, "queued item cannot be steered (no running turn, attachments, or another sender)")
        index = _remove_queued_envelope(session, envelope)
        text = _inflight_text(envelope.get("text")).strip()
    try:
        accepted = bool(agent.steer(text))
    except Exception as exc:
        logger.debug("queued-prompt steer failed", exc_info=True)
        accepted, failure = False, f"steer failed: {exc}"
    else:
        failure = "steer was rejected"
    with session["history_lock"]:
        if accepted:
            _record_inflight_correction(session, text)
            _retire_queued_envelope_row(session, envelope)
            session["last_active"] = time.time()
        else:
            entries = _queued_envelopes(session)
            entries.insert(min(index, len(entries)), envelope)
            _ac_set_queue(session, entries)
        idle = not session.get("running")
    if accepted:
        return None
    if idle:
        # The turn ended while the item was claimed; its drain may already have run past it.
        _drain_queued_prompt(rid, sid, session)
    return _err(rid, 4009, failure)


@method("session.queue.update")
def _(rid, params: dict) -> dict:
    """Edit, delete, or promote to a steer one queued prompt, addressed by ``queue_id``."""
    sid = str(params.get("session_id") or "")
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    queue_id = params.get("queue_id")
    if not isinstance(queue_id, str) or not queue_id.strip():
        return _err(rid, 4004, "queue_id required")
    queue_id = queue_id.strip()
    action = params.get("action")
    if action not in _QUEUE_ACTIONS:
        return _err(rid, 4004, f"action must be one of: {', '.join(_QUEUE_ACTIONS)}")
    if action == "steer":
        if (failure := _steer_queued(rid, sid, session, queue_id)) is not None:
            return failure
    else:
        text = params.get("text")
        if action == "edit" and (not isinstance(text, str) or not text.strip()):
            return _err(rid, 4002, "text is required")
        with session["history_lock"]:
            envelope = _find_queued_envelope(session, queue_id)
            if envelope is None:
                return _err(rid, 4041, "queued item not found")
            if action == "edit":
                if not _queue_item(session, envelope)["editable"]:
                    return _err(rid, 4004, "queued item cannot be edited (attachments or another sender)")
                envelope["text"] = text.strip()
                staged = envelope.get("_submit_user_row")
                if isinstance(staged, dict) and isinstance(staged.get("_row_id"), int):
                    # Merge-sync path: rewrites the accept-time durable row to the edited text.
                    _persist_queued_user_row(session, envelope, envelope.get("_queued_display_kind"))
            else:
                _remove_queued_envelope(session, envelope)
                _retire_queued_envelope_row(session, envelope)
            session["last_active"] = time.time()
    return _ok(rid, {"status": action, **_publish_queue(sid, session)})


def register(server) -> None:
    """Publish this module's helpers + handlers onto ``server``, rebound to its globals."""
    bind_module(globals(), server, skip=("_",))
