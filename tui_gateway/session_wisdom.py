"""Wisdom session activity and commands, bound to the shared TUI server globals."""

from __future__ import annotations

import threading

from .method_ctx import bind_module

_WISDOM_POLL_SECONDS = 60.0
_WISDOM_NOTICE_KEY = "wisdom.activity"
_wisdom_poll_lock = threading.Lock()
_wisdom_profile_last_poll: dict[str, float] = {}


def _wisdom_profile_key(session: dict) -> str:
    home = session.get("profile_home") or _hermes_home
    return str(Path(home).expanduser().resolve())


def _collect_wisdom_activity_notice(session: dict) -> tuple[bool, str | None]:
    """Refresh one profile and project its pending Wisdom activity.

    The boolean distinguishes a successful empty read (which may clear the
    sticky notice) from a transient failure (which must retain it). No result
    is injected into chat history and this path never invokes a model.
    """

    if session.get("_finalized") or session.get("running"):
        return False, None
    session_key = str(session.get("session_key") or "").strip()
    if not session_key:
        return False, None

    try:
        from hermes_wisdom.entitlement import local_work_allowed
        from hermes_wisdom.service import WisdomService

        with _session_profile_runtime_scope(session):
            service = WisdomService()
            if not local_work_allowed(service.store):
                return True, None
            profile_key = _wisdom_profile_key(session)
            now = time.monotonic()
            with _wisdom_poll_lock:
                last_poll = _wisdom_profile_last_poll.get(profile_key, 0.0)
                if now - last_poll >= _WISDOM_POLL_SECONDS:
                    # Claim the cadence before the request. A failed refresh is
                    # briefly cached too, preventing multiple live sessions
                    # from turning an outage into a request storm.
                    _wisdom_profile_last_poll[profile_key] = now
                    service.check(apply_automatic=False)

            organization_events = service.notifications(mark_seen=False).get(
                "events", []
            )
            candidate_events = service.local_candidate_events(
                session_id=session_key
            )
    except Exception:
        logger.debug("Collective Wisdom TUI poll failed", exc_info=True)
        return False, None

    organization_count = len(organization_events)
    candidate_count = len(candidate_events)
    if not organization_count and not candidate_count:
        return True, None

    parts: list[str] = []
    commands: list[str] = []
    if organization_count:
        noun = "update" if organization_count == 1 else "updates"
        parts.append(f"{organization_count} team {noun}")
        commands.append("/wisdom notifications")
    if candidate_count:
        noun = "skill" if candidate_count == 1 else "skills"
        parts.append(f"{candidate_count} {noun} ready to review")
        commands.append("/wisdom candidates")
    return True, (
        f"Collective Wisdom: {' and '.join(parts)}. "
        f"Run {' or '.join(commands)} to manage them."
    )


def _sync_wisdom_activity_notice(sid: str, session: dict) -> None:
    try:
        with _session_profile_runtime_scope(session):
            from hermes_wisdom.mediation import delivery_mode

            mediated = delivery_mode() == "agent"
            session["_wisdom_activity_tracking"] = True
        if (session.get("_wisdom_user_activity") and not session.get("running")
                and not _transport_is_dead(session.get("transport"))):
            from tui_gateway.wisdom_mediation import poll

            if _ensure_active_session_slot(sid, session) is not None:
                return
            try:
                poll(session, emit=lambda method, payload: _emit(method, sid, payload),
                     profile_scope=_session_profile_runtime_scope,
                     connected=lambda: not _transport_is_dead(session.get("transport")))
            finally:
                # This is a real user prompt queued while the isolated read-only
                # assessment held the turn guard, not a synthetic Wisdom wake.
                _drain_queued_prompt("wisdom-queued-user", sid, session)
        if mediated:
            return
    except Exception:
        logger.debug("Wisdom mediation poll failed", exc_info=True)
        return
    success, text = _collect_wisdom_activity_notice(session)
    if not success:
        return
    previous = session.get("_wisdom_notice_text")
    if text:
        if text != previous:
            _emit(
                "notification.show",
                sid,
                {
                    "key": _WISDOM_NOTICE_KEY,
                    "id": _WISDOM_NOTICE_KEY,
                    "kind": "wisdom",
                    "level": "info",
                    "text": text,
                    "ttl_ms": None,
                },
            )
            session["_wisdom_notice_text"] = text
    elif previous:
        _emit("notification.clear", sid, {"key": _WISDOM_NOTICE_KEY})
        session.pop("_wisdom_notice_text", None)


def _format_live_wisdom_output(session: dict, name: str, arg: str) -> str:
    """Run the shared Wisdom controller in the gateway's profile scope."""
    from gateway.wisdom_command import (
        WisdomCommandContext,
        WisdomCommandController,
        command_error_text,
        render_local_view,
    )
    from hermes_wisdom.service import WisdomService

    raw_args = arg.strip()
    if name == "collective-wisdom-install":
        raw_args = f"install {raw_args}".strip()

    try:
        with _session_profile_runtime_scope(session):
            service = WisdomService()
            context = WisdomCommandContext(
                user_id="local-user",
                chat_id=f"local:{session.get('session_key', '')}",
                profile=str(get_hermes_home()),
                organization_id=service.store.active_org_id(),
                is_group=False,
            )
            view = WisdomCommandController().execute(raw_args, service, context)
            return render_local_view(view, context)
    except Exception as exc:
        return f"Collective Wisdom could not continue: {command_error_text(exc)}"


def register(server) -> None:
    """Publish Wisdom helpers using the same binding as other session modules."""
    bind_module(globals(), server)
