"""Exact, inert stored-session attachment; deliberately separate from executing resume.

Legacy persistence has no terminal execution receipt. A cold attachment therefore
stays fenced even when its crash marker is absent. It is not permission to resume.
"""

import contextlib
from typing import TYPE_CHECKING

from .method_ctx import HandlerRegistry, bind_module

if TYPE_CHECKING:
    # Runtime binding is performed by method_ctx, like the other method siblings.
    from .server import (
        _canonical_profile_request,
        _db_unavailable_error,
        _default_session_cwd,
        _deferred_session_record,
        _err,
        _hermes_home,
        _live_profile_matches,
        _ok,
        _profile_db,
        _profile_home,
        _response_profile_name,
        _session_lookup_key,
        _session_resume_lock,
        _sessions,
        _sessions_lock,
        _stdio_transport,
        current_transport,
    )
    from .methods_session import _new_runtime_ids
    from .session_lifecycle import _reattach_refusal, _rebind_live_transport
    from .model_switch import _session_profile_runtime_scope
    from .session_reaper import _transport_is_dead

_registry = HandlerRegistry()
method = _registry.method


def _attachment_execution_error(rid, session):
    fence = session.get("attachment_fence")
    if fence is None:
        return None
    return _err(
        rid,
        4091,
        "attached prior work requires explicit disposition; start a new conversation",
        {"reason": "attachment_execution_fenced", "disposition": fence["disposition"]},
    )


def _attachment_descriptor(sid, session, target, profile, row, *, reused):
    """Caller holds resume + history locks: identity and local lifecycle are one cut.

    Other stores have no common revision barrier. Do not sample them piecemeal and
    present an apparently atomic request/history/replay recovery snapshot.
    """
    from .event_replay import replay_epoch
    from uuid import uuid4

    fence = session.get("attachment_fence")
    incarnation = session.setdefault("attachment_incarnation", uuid4().hex)
    disposition = (
        fence["disposition"]
        if fence
        else (
            "live"
            if session.get("running")
            or session.get("resume_hydrating")
            or session.get("_auto_continue_scheduled")
            else "idle"
        )
    )
    return {
        "requested_session_id": target,
        "stored_session_id": target,
        "session_id": sid,
        "profile": profile,
        "reused_runtime": reused,
        "runtime_incarnation": incarnation,
        "owner_incarnation": None if fence else f"{replay_epoch()}:{incarnation}",
        "parent_session_id": row.get("parent_session_id"),
        "disposition": disposition,
        "execution_fenced": fence is not None,
        "fence_reason": fence["reason"] if fence else None,
        "can_submit_prompt": fence is None,
        "recovery": {
            "complete": False,
            "history_loaded": False,
            "history_revision": None,
            "request_lifecycle": "unknown",
            "request_revision": None,
            "replay_epoch": replay_epoch(),
            "stream_incarnation": None,
            "replay_high_water": None,
            "events_complete": False,
            "execution_complete": False,
        },
    }


def _exact_attachment_live(target, home):
    with _sessions_lock:
        return [
            (sid, s)
            for sid, s in _sessions.items()
            if not s.get("_finalized")
            and _live_profile_matches(s, home)
            and (s.get("session_key") == target or _session_lookup_key(s) == target)
        ]


@method("session.attach")
def _(rid, params):
    from hermes_cli.profiles import normalize_profile_name

    target, profile = params.get("session_id"), params.get("profile")
    # The gateway's parameter validator currently checks unknown keys only;
    # identity must also fail closed when a required value is missing/malformed.
    if (
        not isinstance(target, str)
        or not 1 <= len(target) <= 512
        or not isinstance(profile, str)
        or not 1 <= len(profile) <= 64
    ):
        return _err(rid, 4000, "exact session and profile identity required")
    # Do not accept whitespace/title/profile aliases as alternate identities.
    if (
        target != target.strip()
        or profile != profile.strip()
        or normalize_profile_name(profile) != profile
        or _canonical_profile_request(profile) != profile
    ):
        return _err(rid, 4000, "exact session and profile identity required")
    home = _profile_home(profile)
    if _response_profile_name(profile) != profile:
        return _err(rid, 4064, "exact profile identity required")
    transport = current_transport() or _stdio_transport
    if _transport_is_dead(transport):
        return _err(rid, 4009, "attachment transport is closed")
    try:
        with _profile_db(params) as db, _session_resume_lock:
            if db is None:
                return _db_unavailable_error(rid, code=5000)
            row = db.get_session(
                target
            )  # exact primary key; never resolve_session_id / compression tip
            if not row or row.get("id") != target:
                return _err(rid, 4007, "exact stored session not found")
            candidates = _exact_attachment_live(target, home)
            if len(candidates) > 1:
                return _err(
                    rid,
                    4090,
                    "ambiguous runtime ownership",
                    {"reason": "session_owned"},
                )
            if candidates:
                sid, session = candidates[0]
                with session["history_lock"]:
                    # A compression rotation or closing watcher is not this exact segment.
                    if (
                        _session_lookup_key(session) != target
                        or session.get("_closing")
                        or session.get("_lease_taken_over")
                        or session.get("lazy")
                        and not session.get("attachment_fence")
                    ):
                        return _err(
                            rid,
                            4090,
                            "runtime cannot safely attach this exact segment",
                            {"reason": "session_owned"},
                        )
                    if (refusal := _reattach_refusal(rid, sid, session)) is not None:
                        return refusal
                    # A cold attachment has no lease. Recheck under the registry
                    # lock before reusing it; another process may have claimed the
                    # stored session since this inert record was created.
                    from hermes_cli.active_sessions import active_session_liveness_guard

                    ownership = (
                        active_session_liveness_guard(
                            target, registry_home=home or _hermes_home
                        )
                        if session.get("attachment_fence") is not None
                        and session.get("active_session_lease") is None
                        else contextlib.nullcontext(False)
                    )
                    with ownership as owned:
                        if owned:
                            return _err(
                                rid,
                                4090,
                                "session has a live owner",
                                {"reason": "session_owned"},
                            )
                        _rebind_live_transport(
                            sid, session, current_transport() or _stdio_transport
                        )
                        return _ok(
                            rid,
                            _attachment_descriptor(
                                sid, session, target, profile, row, reused=True
                            ),
                        )
            # This is observation, not an ownership claim. The existing execution
            # lease stays with its owner; the new cold record cannot execute at all.
            from hermes_cli.active_sessions import active_session_liveness_guard
            from .turn_marker import inspect_turn_recovery

            with active_session_liveness_guard(
                target, registry_home=home or _hermes_home
            ) as owned:
                if owned:
                    return _err(
                        rid,
                        4090,
                        "session has a live owner",
                        {"reason": "session_owned"},
                    )
                disposition, reason = inspect_turn_recovery(
                    home or _hermes_home, target
                )
                sid, source = _new_runtime_ids({})
                # No history repair/reopen, model construction, timer, services,
                # prompt hooks or auto-continuation. History stays in its paged store.
                with _session_profile_runtime_scope(
                    {"profile_home": home}, hydrate_secrets=False
                ):
                    record = _deferred_session_record(
                        target,
                        cols=80,
                        cwd=row.get("cwd") or _default_session_cwd(),
                        history=[],
                        lease=None,
                        source=source,
                        profile_home=home,
                        lazy=True,
                    )
                record["attachment_fence"] = {
                    "disposition": disposition,
                    "reason": reason,
                }
                with _sessions_lock:
                    _sessions[sid] = record
                with record["history_lock"]:
                    return _ok(
                        rid,
                        _attachment_descriptor(
                            sid, record, target, profile, row, reused=False
                        ),
                    )
    except Exception:
        # Never turn a failed store/liveness/recovery query into unowned/idle.
        return _err(
            rid,
            4090,
            "cannot establish safe exact attachment",
            {"reason": "recovery_unknown"},
        )


def register(server):
    bind_module(globals(), server, skip=("_", "TYPE_CHECKING"))
