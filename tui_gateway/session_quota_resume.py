"""Wait out a subscription quota window, then finish the turn it killed.

A quota wall is the one failure that comes with a promise: the provider says when
it lifts. Everywhere else the honest answer is "retry and see", but here the
session can simply sleep until the window reopens and pick the work back up —
which is what Claude Code's own "continue automatically at usage limit" does.

Scheduling lives in the backend, not the renderer, because the backend owns the
session: it already arbitrates turn ownership across the Desktop app, the TUI and
the dashboard, and it survives a window reload. A renderer timer would fire twice
when two windows have the same session open, and not at all after a refresh.

The deadline itself is resolved by ``agent.quota_resume`` and is always
provider-reported — this module only decides *when to act on it* and *whether
acting is still wanted*. A pending resume is abandoned the moment the user takes
the turn back (new prompt, manual retry, cancel, model switch): they have moved
on, and firing anyway would talk over them.

Bodies are rebound onto server.py's globals at install time (method_ctx.bind_module),
so they reference server.py globals bare.
"""

from __future__ import annotations

import contextlib

from .method_ctx import bind_module

# Poll cadence for due resumes. Coarse on purpose: the deadline is minutes-to-hours
# away, and the grace period in agent.quota_resume already absorbs clock skew, so
# a few seconds of slop costs nothing.
_QUOTA_RESUME_POLL_SECONDS = 15.0

_QUOTA_RESUME_ENABLED_DEFAULT = True
# One re-arm covers the common "provider moved the goalposts" case (the window
# reopened later than first advertised) without letting a mis-reported deadline
# loop a session forever.
_QUOTA_RESUME_MAX_ATTEMPTS_DEFAULT = 2

_QUOTA_RESUME_NOTE_PREFIX = "[Resumed after a provider usage limit"


def _quota_resume_config() -> tuple[bool, int]:
    """``(enabled, max_attempts)`` from config.yaml (``desktop.quota_resume``)."""
    desktop = _load_cfg().get("desktop")
    cfg = desktop.get("quota_resume") if isinstance(desktop, dict) else None
    if not isinstance(cfg, dict):
        cfg = {}
    return (
        is_truthy_value(cfg.get("enabled"), default=_QUOTA_RESUME_ENABLED_DEFAULT),
        _coerce_int_config_value(
            cfg.get("max_attempts"), _QUOTA_RESUME_MAX_ATTEMPTS_DEFAULT, min_value=0
        ),
    )


def _quota_resume_note(prompt: str) -> str:
    """Continuation note for the resumed turn.

    Carries the original request because the failed turn produced no assistant
    message to continue from, and warns that earlier tool work may already be
    done — the turn died after the model call, so side effects can predate it.
    """
    return (
        f"{_QUOTA_RESUME_NOTE_PREFIX} — the provider's quota window has reset and "
        "the interrupted request is being retried automatically. Some of the work may "
        "already be complete; check the current state before redoing anything, then "
        "finish the task. The interrupted request was:]\n\n"
        f"{prompt}"
    )


def _clear_quota_resume(session: dict) -> None:
    """Drop any pending resume (the user took the turn back, or it fired)."""
    for key in ("_quota_resume_at", "_quota_resume_prompt", "_quota_resume_meta"):
        session.pop(key, None)


def _arm_quota_resume(sid: str, session: dict, plan: dict, prompt: str) -> bool:
    """Record a due-time for a failed turn whose provider named a reset.

    Called from the turn's terminal path. Returns True when a resume was armed.
    Refuses an ineligible plan, a disabled feature, an empty prompt (nothing to
    resubmit) and a session that has already used its attempts.
    """
    if not isinstance(plan, dict) or not plan.get("eligible"):
        return False
    resume_at = plan.get("resume_at")
    if not isinstance(resume_at, (int, float)):
        return False
    enabled, max_attempts = _quota_resume_config()
    if not enabled:
        return False
    # Hosted-room turns recover through their own durable lease state machine;
    # a generic resume would bypass its execution generation and duplicate work.
    if session.get("source") == "bot_room":
        return False
    if not (prompt or "").strip():
        return False
    if int(session.get("_quota_resume_attempt", 0) or 0) >= max_attempts:
        logger.info("quota-resume: attempts exhausted for %s; leaving manual retry", sid)
        return False
    session["_quota_resume_at"] = float(resume_at)
    session["_quota_resume_prompt"] = prompt
    session["_quota_resume_meta"] = {
        "source": plan.get("source", ""),
        "provider": plan.get("provider", ""),
        "resume_at": float(resume_at),
    }
    logger.info(
        "quota-resume armed for %s in %.0fs (source=%s provider=%s)",
        sid, max(0.0, float(resume_at) - time.time()),
        plan.get("source", ""), plan.get("provider", ""),
    )
    _emit("status.update", sid, {
        "kind": "process",
        "text": _quota_resume_status_text(float(resume_at), plan.get("provider", "")),
    })
    return True


def _quota_resume_status_text(resume_at: float, provider: str) -> str:
    """Human-readable 'waiting until…' line for the status stack."""
    delta = max(0, int(resume_at - time.time()))
    hours, minutes = divmod(delta // 60, 60)
    when = f"{hours}h {minutes}m" if hours else f"{minutes}m"
    label = f"{provider} " if provider else ""
    return f"{label}usage limit reached — resuming automatically in {when}"


def _maybe_fire_quota_resume(sid: str, session: dict) -> None:
    """Fire a due quota resume for an idle session (per-session poller).

    Claims the session under ``history_lock`` exactly as the /loop tick and
    crash auto-continue do, so a racing user prompt always wins. Ownership is
    re-checked against the shared HERMES_HOME before dispatch: a sibling backend
    may be mid-turn on the same session.
    """
    resume_at = session.get("_quota_resume_at")
    if not isinstance(resume_at, (int, float)) or time.time() < float(resume_at):
        return
    # A user prompt after the failure means they are driving again.
    if session.get("running") or session.get("_finalized") or session.get("_turn_cancel_requested"):
        _clear_quota_resume(session)
        return
    prompt = str(session.get("_quota_resume_prompt") or "")
    meta = session.get("_quota_resume_meta") or {}
    if not prompt.strip():
        _clear_quota_resume(session)
        return
    if not _notif_claim_turn(session):
        return  # busy — stays due, next poll retries
    # Past this point the resume either dispatches or is abandoned; never left pending.
    _clear_quota_resume(session)
    if _ensure_active_session_slot(sid, session) is not None:
        logger.info("quota-resume for %s refused: session has another live owner", sid)
        _notif_release_turn(session)
        return
    session["_quota_resume_attempt"] = int(session.get("_quota_resume_attempt", 0) or 0) + 1
    rid = f"__quota_resume__{int(time.time() * 1000)}"
    try:
        _emit("status.update", sid, {
            "kind": "process",
            "text": f"Usage limit reset{' on ' + meta['provider'] if meta.get('provider') else ''} — resuming…",
        })
        _emit("message.start", sid)
        _run_prompt_submit(
            rid, sid, session, _quota_resume_note(prompt), display_kind="auto_continue"
        )
    except Exception as exc:
        _notif_log_failure("quota-resume dispatch failed", exc)
        _notif_release_turn(session)


def register(server) -> None:
    bind_module(globals(), server)
