"""Cron job argument normalization, validation and result shaping (re-exported by
tools/cronjob_tools.py)."""

import logging
import math
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from cron.jobs import effective_job_state

# Logger parity with the origin module.
logger = logging.getLogger("tools.cronjob_tools")


def _origin_from_env() -> Optional[Dict[str, str]]:
    from gateway.session_context import get_session_env
    origin_platform = get_session_env("HERMES_SESSION_PLATFORM")
    origin_chat_id = get_session_env("HERMES_SESSION_CHAT_ID")
    if not (origin_platform and origin_chat_id):
        return None
    thread_id = get_session_env("HERMES_SESSION_THREAD_ID") or None
    # Slack stamps every TOP-LEVEL message's own id as the session thread (a per-message
    # KEY, not a location); persisting it would pin all future deliveries inside an
    # ephemeral thread, so thread == creating message id is synthetic and dropped.
    if thread_id and origin_platform == "slack":
        message_id = get_session_env("HERMES_SESSION_MESSAGE_ID") or None
        if message_id and str(thread_id) == str(message_id):
            logger.debug(
                "Cron origin: dropping synthetic per-message Slack "
                "thread_id=%s (== creation message id)", thread_id)
            thread_id = None
    if thread_id:
        logger.debug(
            "Cron origin captured thread_id=%s for %s:%s",
            thread_id, origin_platform, origin_chat_id)
    return {
        "platform": origin_platform, "chat_id": origin_chat_id,
        "chat_name": get_session_env("HERMES_SESSION_CHAT_NAME") or None, "thread_id": thread_id,
        # Lets a delivery mirror resolve the participant's session in per-user-isolated groups.
        "user_id": get_session_env("HERMES_SESSION_USER_ID") or None,
        # Workspace/server scope (Slack team, Discord guild...): Slack session keys embed it,
        # so a continuable cron seed built without it would never resolve a scoped reply.
        "scope_id": get_session_env("HERMES_SESSION_SCOPE_ID") or None,
    }


def _local_delivery_notice(job: Dict[str, Any], user_deliver: Optional[str]) -> Optional[str]:
    """Notice when a created job won't deliver anywhere: CLI/TUI sessions have no capturable
    origin, so deliver='origin' (or omitted) saves output but never delivers it. None when the
    user explicitly asked for ``local`` or the job resolves to a real target.

    TUI/CLI sessions cannot be captured as a cron ``origin`` (no ``HERMES_SESSION_PLATFORM``/``CHAT_ID`` is
    set for them), so a ``deliver="origin"`` request — or an omitted ``deliver`` that defaults to
    origin-or-local — produces a job that runs and saves output to ``last_output`` but is never delivered
    back into the session. This is by design (there is no live-delivery channel for local sessions), but
    silently dropping the user's "tell me when it runs" intent is the trap reported in 51568. Surface it at
    create time so the agent can relay it instead of promising a delivery that never happens. See #51568.
    """
    if (user_deliver or "").strip().lower() == "local":
        return None
    try:
        from cron.scheduler import _resolve_delivery_targets
        if _resolve_delivery_targets(job):
            return None
    except Exception:  # resolution unavailable — fall back to the origin signal
        if job.get("origin"):
            return None
    return (
        "This is a local-only cron job: its output is saved (view it with "
        "cronjob(action='list')) but will NOT be delivered back into this "
        "session — CLI/TUI sessions have no live-delivery channel. To be "
        "notified when it runs, recreate or update the job with deliver set to "
        "a gateway-connected platform, e.g. deliver='telegram' or deliver='all'.")


def _mode_guidance_notes(job: Dict[str, Any], user_deliver: Optional[str]) -> List[str]:
    """Mode guidance echoed once in the create/update response (not in the schema, which is
    paid for on every API call)."""
    notes: List[str] = []
    if job.get("monitor_script") or job.get("monitor_url"):
        notes.append(
            "Monitor mode: the source runs first each tick and its output is "
            "hashed as exact bytes — unchanged output suppresses the agent run "
            "(silent no_change tick), changed output injects a MONITOR CHANGE "
            "DETECTED diff into the prompt. The first tick always runs as "
            "baseline. The source must emit STABLE output (no timestamps, no "
            "random ordering) or every tick will look changed.")
    if job.get("no_agent"):
        notes.append(
            "no_agent mode: stdout is delivered verbatim; EMPTY stdout sends "
            "nothing at all (watchdog pattern — script should stay quiet when "
            "there is nothing to report). Non-zero exit or timeout sends an "
            "error alert. prompt/skills are ignored.")
    _deliver = (user_deliver or "").strip().lower()
    if _deliver:
        if "all" in _deliver.split(","):
            notes.append(
                "deliver='all' resolves at fire time and never includes "
                "bot-chat targets — channels connected later are picked up "
                "automatically.")
        if _deliver.startswith("bot-chat:"):
            notes.append("Targeting another profile's Bot Chat costs that bot an agent turn per run.")
        # platform:chat_id with no thread segment loses topic targeting.
        for target in _deliver.split(","):
            parts = target.strip().split(":")
            if (
                len(parts) == 2
                and parts[0] not in ("bot-chat", "sms")
                and parts[1]
                and not parts[1].startswith("#")):
                notes.append(
                    f"deliver target '{target.strip()}' has no :thread_id "
                    "segment — on thread/topic platforms the delivery lands in "
                    "the main chat, not a topic.")
                break
    return notes


def _split_monitor_arg(
    monitor: Optional[str],
    monitor_script: Optional[str],
    monitor_url: Optional[str]) -> tuple:
    """Resolve the model-facing ``monitor`` field into the stored ``(monitor_script,
    monitor_url)`` pair. http(s):// is a URL, anything else a script path (a legal script path
    never starts with a URL scheme). None = unchanged, '' = clear; setting one source clears
    the other so switching transports never trips mutual exclusion; an explicit ``monitor``
    wins over the legacy alias fields."""
    if monitor is None:
        return monitor_script, monitor_url
    value = monitor.strip()
    if not value:
        return "", ""
    if value.lower().startswith(("http://", "https://")):
        return "", value
    return value, ""


def _repeat_display(job: Dict[str, Any]) -> str:
    rep = job.get("repeat")
    if type(rep) is not dict:
        rep = {}
    times, completed = rep.get("times"), rep.get("completed", 0)
    if times is None:
        return "forever"
    if times == 1:
        return "once" if completed == 0 else "1/1"
    return f"{completed}/{times}" if completed else f"{times} times"


def _clean_str_list(items: Any) -> List[str]:
    """Stripped, non-empty ``str(item)`` values from a str-or-iterable (order kept)."""
    if items is None:
        return []
    if isinstance(items, str):
        items = [items]
    return [s for s in (str(i).strip() for i in items) if s]


def _canonical_skills(skill: Optional[str] = None, skills: Optional[Any] = None) -> List[str]:
    if skills is None:
        skills = [skill] if skill else []
    elif isinstance(skills, str):
        skills = [skills]
    # `item or ""`: a None entry must drop out, not stringify to "None".
    return list(dict.fromkeys(_clean_str_list(item or "" for item in skills)))


def _normalize_optional_job_value(value: Optional[Any], *, strip_trailing_slash: bool = False) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if strip_trailing_slash:
        text = text.rstrip("/")
    return text or None


def _normalize_deliver_param(value: Any) -> Optional[str]:
    """Canonical string form of ``deliver``; None for None/empty. MCP clients may pass a list
    (``["telegram"]``) which the scheduler's ``str(deliver).split(",")`` would mangle."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return ",".join(_clean_str_list(value)) or None
    return str(value).strip() or None


def _validate_bot_chat_deliver(deliver: Optional[str]) -> Optional[str]:
    """Validate ``bot-chat[:<profile>]`` deliver elements at create time: Bot Chat delivery is
    machine-local, so the profile must exist where the scheduler fires (Desktop rosters may
    show same-named profiles from other machines). Returns an error string or None."""
    if not deliver:
        return None
    try:
        from cron.scheduler_delivery import parse_bot_chat_deliver_token
        from hermes_cli.profiles import normalize_profile_name, profile_exists
    except Exception:
        return None  # best-effort; resolution re-checks at fire time
    for part in str(deliver).split(","):
        profile_arg = parse_bot_chat_deliver_token(part.strip())
        if not profile_arg:
            continue  # not a bot-chat token, or bare token (own profile)
        try:
            canon = normalize_profile_name(profile_arg)
        except Exception:
            return f"invalid bot-chat profile name '{profile_arg}'"
        if not profile_exists(canon):
            return (
                f"bot-chat delivery profile '{profile_arg}' not found on this "
                "gateway's machine. Bot Chat delivery is machine-local — use a "
                "profile that exists here (hermes profile list), or omit the "
                "name (deliver='bot-chat') for the job's own profile.")
    return None


def _resolve_cron_context_deliver(deliver: Optional[str]) -> Optional[str]:
    """Resolve ``origin`` to a concrete target for creates made FROM a cron run (the creating
    session is ephemeral, so by fire time there is no origin). Non-cron sessions: unchanged.
    Cron sessions: ``origin`` (or omitted) becomes the creating run's ``platform:chat_id[:thread]``
    from HERMES_CRON_AUTO_DELIVER_*, or ``local`` when it has no concrete target; other
    elements pass through. Otherwise the scheduler would guess a home channel."""
    from gateway.session_context import get_session_env
    from utils import is_truthy_value
    if not is_truthy_value(get_session_env("HERMES_CRON_SESSION", "")):
        return deliver

    def _creator_target() -> str:
        platform = get_session_env("HERMES_CRON_AUTO_DELIVER_PLATFORM", "").strip()
        chat_id = get_session_env("HERMES_CRON_AUTO_DELIVER_CHAT_ID", "").strip()
        if not platform or not chat_id:
            return "local"
        thread_id = get_session_env("HERMES_CRON_AUTO_DELIVER_THREAD_ID", "").strip()
        return f"{platform}:{chat_id}:{thread_id}" if thread_id else f"{platform}:{chat_id}"

    if deliver is None:
        return _creator_target()
    resolved = [_creator_target() if p.lower() == "origin" else p for p in _clean_str_list(str(deliver).split(","))]
    # Order-preserving de-dup: 'origin,local' with a local creator -> 'local'.
    return ",".join(dict.fromkeys(resolved)) or None


def _validate_cron_base_url(
    provider: Optional[Any], base_url: Optional[Any]) -> Optional[str]:
    """Reject pairing a named provider's stored credential with an off-host base_url (a
    prompt-injected job could exfil the key). Allowed: no override; bare 'custom' (pure BYOK,
    key derived from the base_url); an override whose host matches the named provider's own
    endpoint. Everything else fails closed. Returns an error string if blocked, else None."""
    bu = _normalize_optional_job_value(base_url, strip_trailing_slash=True)
    if not bu:
        return None
    prov = _normalize_optional_job_value(provider)
    if not prov:  # no provider inherits the default provider's stored key — same primitive
        return (
            "base_url override requires an explicit provider. Set provider to a "
            "configured custom provider to use a custom endpoint.")
    try:
        from hermes_cli.runtime_provider import (
            has_named_custom_provider,
            resolve_requested_provider,
            _get_named_custom_provider)
        from hermes_cli.auth import PROVIDER_REGISTRY
        from utils import base_url_host_matches, base_url_hostname
    except Exception:
        return f"Unable to validate base_url override for provider {prov!r}; refused."

    if prov.lower() == "custom":  # pure BYOK: key keyed by THIS base_url, never a stored secret
        return None
    if has_named_custom_provider(prov):
        # A NAMED custom provider's STORED key is still sent to an override base_url.
        try:
            cp = _get_named_custom_provider(prov)
        except Exception:
            cp = None
        cfg_host = base_url_hostname((cp or {}).get("base_url", "")) if cp else ""
        if cfg_host and base_url_host_matches(bu, cfg_host):
            return None
        return (
            f"base_url {bu!r} is not allowed for provider {prov!r}. A named "
            f"custom provider's stored credential may only be sent to its own "
            f"configured endpoint ({cfg_host or 'unknown'}).")
    try:
        resolved = resolve_requested_provider(prov)
    except Exception:
        resolved = prov
    pconfig = PROVIDER_REGISTRY.get(resolved) if isinstance(resolved, str) else None
    known_host = base_url_hostname(getattr(pconfig, "inference_base_url", "") if pconfig else "")
    if known_host and base_url_host_matches(bu, known_host):
        return None
    # Fail closed: named providers with stored credentials AND unknown names we cannot host-match.
    return (
        f"base_url {bu!r} is not allowed for provider {prov!r}. A named "
        f"provider's stored credential may only be sent to its own endpoint; "
        f'use a configured custom provider (provider="custom") for a custom base_url.')


def _validate_cron_script_path(script: Optional[str]) -> Optional[str]:
    """Scripts must be relative paths within HERMES_HOME/scripts/ (absolute / ~ / drive-letter
    rejected — prompt-injection guard). Error string if blocked, else None; empty = clear."""
    if not script or not script.strip():
        return None

    from hermes_constants import get_hermes_home
    raw = script.strip()
    if raw.startswith(("/", "~")) or (len(raw) >= 2 and raw[1] == ":"):
        return (
            f"Script path must be relative to ~/.hermes/scripts/. "
            f"Got absolute or home-relative path: {raw!r}. "
            f"Place scripts in ~/.hermes/scripts/ and use just the filename.")

    from tools.path_security import validate_within_dir
    scripts_dir = get_hermes_home() / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    if validate_within_dir(scripts_dir / raw, scripts_dir):
        return f"Script path escapes the scripts directory via traversal: {raw!r}"
    return None


def _apply_continuity(
    context_from: Optional[Union[str, List[str]]],
    continuity: bool) -> Optional[List[str]]:
    """continuity=True ensures "self" is in context_from; False removes it; others untouched."""
    refs = _clean_str_list(context_from)
    has_self = any(r.lower() == "self" for r in refs)
    if continuity and not has_self:
        refs.append("self")
    elif not continuity and has_self:
        refs = [r for r in refs if r.lower() != "self"]
    return refs or None


def _validate_context_from_refs(refs: List[Any]) -> Optional[str]:
    """Error string if any non-"self" ref names a missing job ("self" resolves to the job's
    own id at run time, so it can't be checked — the job doesn't exist yet at create)."""
    from cron.jobs import get_job as _get_job
    for ref_id in refs:
        if isinstance(ref_id, str) and ref_id.strip().lower() == "self":
            continue
        if not _get_job(ref_id):
            return (
                f"context_from job '{ref_id}' not found. "
                "Use cronjob(action='list') to see available jobs.")
    return None


# Optional fields echoed by _format_job only when truthy (order = JSON key order).
_FORMAT_JOB_OPTIONAL_KEYS = (
    "script", "reasoning_effort", "monitor_script", "monitor_url",
    "monitor_state", "no_agent", "enabled_toolsets", "workdir")


_PUBLIC_CRON_JOB_STATES = frozenset({"scheduled", "paused", "completed", "error"})
_PUBLIC_CRON_LAST_STATUSES = frozenset({"ok", "error", "delivery_failed", "delivery_queued", "blocked_config", "interrupted"})


def _format_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """Model-safe cron summary: no prompts, targets, paths, or raw failures."""
    if type(job) is not dict:
        raise TypeError("cron job projection requires an object")

    def timestamp(value: Any) -> Optional[str]:
        if type(value) is not str or len(value) > 64:
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return value if parsed.tzinfo is not None and parsed.utcoffset() is not None else None

    def category(value: Any) -> Optional[str]:
        return value if type(value) is str and re.fullmatch(r"[a-z][a-z0-9_]{0,31}", value) else None

    raw_id = job.get("id")
    job_id = raw_id if type(raw_id) is str and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", raw_id) else "unknown"
    name = job.get("name")
    safe_name = name if type(name) is str and name.isprintable() and len(name) <= 256 else job_id
    deliver = job.get("deliver")
    delivery_kind = (
        deliver if type(deliver) is str and deliver in {"local", "origin", "all"}
        else "external" if type(deliver) is str and deliver else "local"
    )
    last_dispatch = job.get("last_dispatch")
    dispatch = None
    if type(last_dispatch) is dict:
        lateness = last_dispatch.get("lateness_seconds")
        if isinstance(lateness, (int, float)) and not isinstance(lateness, bool):
            lateness_value = float(lateness)
            scheduled_at = timestamp(last_dispatch.get("scheduled_at"))
            dispatched_at = timestamp(last_dispatch.get("dispatched_at"))
            if (scheduled_at and dispatched_at
                    and last_dispatch.get("kind") in {"on_time", "catch_up", "late"}
                    and math.isfinite(lateness_value) and lateness_value >= 0):
                dispatch = {"scheduled_at": scheduled_at, "dispatched_at": dispatched_at,
                            "lateness_seconds": lateness_value, "kind": last_dispatch["kind"]}
    result = {
        "job_id": job_id, "name": safe_name,
        "schedule": job.get("schedule_display") if type(job.get("schedule_display")) is str and len(job["schedule_display"]) <= 256 else "?",
        "repeat": _repeat_display(job), "delivery_kind": delivery_kind,
        "mode": "monitor" if any(type(job.get(key)) is str and job.get(key) for key in ("monitor_script", "monitor_url")) else "script" if job.get("no_agent") is True else "agent",
        "next_run_at": timestamp(job.get("next_run_at")), "last_run_at": timestamp(job.get("last_run_at")),
        "last_dispatch": dispatch,
        "last_status": job["last_status"] if type(job.get("last_status")) is str and job["last_status"] in _PUBLIC_CRON_LAST_STATUSES else None,
        "last_error": "run_failed" if job.get("last_error") is not None else None,
        "last_delivery_error": "delivery_failed" if job.get("last_delivery_error") is not None else None,
        "last_delivery_unverified": True if isinstance(job.get("last_delivery_unverified"), list) and job["last_delivery_unverified"] else None,
        "last_fire_error": {"at": timestamp(job["last_fire_error"].get("at")), "error_kind": "fire_forward_failed"} if type(job.get("last_fire_error")) is dict else None,
        "enabled": job.get("enabled") if type(job.get("enabled")) is bool else True,
        "state": effective_job_state(job) if type(effective_job_state(job)) is str and effective_job_state(job) in _PUBLIC_CRON_JOB_STATES else None,
    }
    if isinstance(job.get("attach_to_session"), bool):
        result["attach_to_session"] = job["attach_to_session"]
    try:
        from cron.executions import latest_execution, receipt_summary
        execution = latest_execution(job_id)
        if execution is not None:
            result["last_execution"] = {"status": category(execution.get("status")), "receipt": receipt_summary(execution["id"])}
    except Exception:
        pass
    return result


def _gateway_liveness_notice(plural: bool = False) -> dict:
    """``gateway_running``/``warning`` payload via the shared CLI helper so CLI and tool agree
    on "scheduler active". False -> warning (no gateway process), None -> probe failed.

    Thin adapter over the shared CLI helper ``hermes_cli.cron._builtin_gateway_liveness`` (#87033) so the
    CLI and this tool can never disagree about what "scheduler active" means. ``plural`` rewords the warning
    for multi-job results (the ``list`` action).
    """
    try:
        from hermes_cli.cron import _builtin_gateway_liveness
        _gw = _builtin_gateway_liveness()
    except Exception:
        return {"gateway_running": None}
    if _gw is False:
        subject = "these jobs are saved" if plural else "this job is saved"
        return {
            "gateway_running": False,
            "warning": (
                f"The Hermes gateway is not running — {subject} "
                "but will NOT fire until the gateway is started "
                "(hermes gateway install / hermes gateway start). "
                "Tell the user the task is scheduled but not active yet."),
        }
    return {"gateway_running": None if _gw is None else True}
