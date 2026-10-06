"""Durable prompt destination validation and retry claims, bound to the gateway."""

from __future__ import annotations

from .method_ctx import bind_module


def _prompt_submit_session_contract(
    params: dict, rid, session: dict
) -> dict | None:
    """Return a fail-closed durable-destination error, if any.

    ``session_id`` is only the live runtime handle. New desktop clients also
    send the stored conversation they selected, which must still identify this
    session before prompt.submit mutates transport, history, inflight state, or
    persistence. Omission remains compatible with older clients.

    Normal sessions accept a compression ancestor by resolving both sides
    through the session's profile-correct DB. An unupgraded lazy/watch session
    is exact-child only: its parent lineage is not an equivalent destination.
    """
    expected = str(
        params.get("expected_stored_session_id")
        or params.get("expected_session_key")
        or ""
    ).strip()
    runtime_id = str(params.get("session_id") or "").strip()
    live_lookup = _session_lookup_key(session, fallback=runtime_id).strip()
    current_ids = {
        str(session.get("session_key") or "").strip(),
        str(session.get("resume_session_id") or "").strip(),
        live_lookup,
    }
    current_ids.discard("")

    # No durable assertion means there is nothing to resolve. This preserves
    # the legacy fast path and avoids opening the profile session database for
    # lineage resolution solely because a submit carries a client request id.
    if not expected:
        return None

    # The normal desktop path names the live durable id exactly. Reserve DB
    # lineage resolution for compression ancestors that are not exact matches.
    if expected in current_ids:
        return None

    resolved_expected = expected
    resolved_current_ids = set(current_ids)
    lazy_watch = bool(session.get("lazy") and session.get("agent") is None)

    if not lazy_watch:
        try:
            with _session_db(session) as db:
                resolver = (
                    getattr(db, "resolve_resume_session_id", None)
                    if db is not None
                    else None
                )
                if callable(resolver):
                    if expected:
                        resolved_expected = str(
                            resolver(expected) or expected
                        ).strip()
                    resolved_current_ids = {
                        str(resolver(current) or current).strip()
                        for current in current_ids
                    }
                    resolved_current_ids.discard("")
        except Exception:
            logger.warning(
                "prompt.submit session contract resolution unavailable",
                exc_info=True,
            )
            return _err(
                rid,
                4022,
                "stored session lineage is temporarily unavailable; retry later",
            )

    matches = (
        not lazy_watch
        and bool(resolved_expected)
        and resolved_expected in resolved_current_ids
    )
    if matches:
        return None

    live = ", ".join(sorted(current_ids)) or "unknown"
    if resolved_current_ids != current_ids or resolved_expected != expected:
        resolved = ", ".join(sorted(resolved_current_ids)) or "unknown"
        live = f"{live} (resolved: {resolved})"
    return _err(
        rid,
        4019,
        f"stored session mismatch: expected {expected!r}; live session is {live}",
    )


def _claim_prompt_submit_intent(
    params: dict, rid, session: dict
) -> dict | None:
    """Atomically reserve a client prompt intent, or return its retry result."""
    global _prompt_intents

    client_request_id = str(params.get("client_request_id") or "").strip()
    if not client_request_id:
        return None

    try:
        ledger = _prompt_intents
        if ledger is None:
            with _prompt_intents_lock:
                ledger = _prompt_intents
                if ledger is None:
                    ledger = PromptIntentLedger(
                        db_path=_hermes_home / "state" / "prompt_intents.sqlite3",
                        session_ttl_s=_SESSION_TTL_S,
                    )
                    _prompt_intents = ledger
        claim = ledger.claim(
            profile_scope=str(session.get("profile_home") or get_hermes_home()),
            request_id=client_request_id,
            route_identity=(
                params.get("expected_stored_session_id")
                or params.get("expected_session_key")
                or _session_lookup_key(
                    session, fallback=str(params.get("session_id") or "")
                )
            ),
            text=params.get("text"),
            truncate_ordinal=params.get("truncate_before_user_ordinal"),
        )
    except (OSError, sqlite3.Error):
        logger.warning("prompt intent ledger unavailable", exc_info=True)
        return _err(
            rid,
            4022,
            "prompt idempotency is temporarily unavailable; retry later",
        )
    if claim is PromptIntentClaim.ACCEPTED:
        return None
    if claim is PromptIntentClaim.DUPLICATE:
        # A busy-path submit can land in the head slot or in the overflow
        # list, and a text merge can move it between them, so scan every
        # envelope rather than just the head.
        queued_match = False
        for envelope in (
            session.get("queued_prompt"),
            *(session.get("queued_prompts") or ()),
        ):
            if not isinstance(envelope, dict):
                continue
            if client_request_id in (envelope.get("client_request_ids") or ()):
                # The original queued acknowledgement may have been lost with
                # its websocket. A matching durable retry must move the
                # eventual drain to this request's freshly rebound transport.
                envelope["transport"] = session.get("transport")
                queued_match = True
                break
        status = (
            "queued"
            if queued_match
            else ("streaming" if session.get("running") else "complete")
        )
        payload = {"duplicate": True, "status": status}
        if status == "complete":
            # prompt.submit holds history_lock while claiming. Returning the
            # transcript here makes completion + hydration one atomic snapshot,
            # closing the race where an earlier session.resume saw the turn
            # streaming just before this duplicate observed it complete.
            payload["messages"] = list(session.get("history", []))
        return _ok(rid, payload)
    if claim is PromptIntentClaim.CONFLICT:
        return _err(
            rid,
            4020,
            "client_request_id was already used for a different prompt",
        )
    if claim is PromptIntentClaim.INVALID:
        return _err(rid, 4021, "client_request_id must be at most 256 characters")
    return _err(
        rid,
        4022,
        "prompt idempotency is temporarily unavailable; retry later",
    )


def _abort_prompt_submit_intent(params: dict, session: dict) -> None:
    """Release an accepted intent when setup failed before agent execution."""
    ledger = _prompt_intents
    if ledger is None:
        return
    ledger.abort(
        profile_scope=str(session.get("profile_home") or get_hermes_home()),
        request_id=str(params.get("client_request_id") or ""),
    )


def register(server) -> None:
    bind_module(globals(), server)
