"""
Cron job management tools for Hermes Agent.

Expose a single compressed action-oriented tool to avoid schema/context bloat.
Compatibility wrappers remain for direct Python callers and legacy tests.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from hermes_constants import display_hermes_home

logger = logging.getLogger(__name__)

# Import from cron module (will be available when properly installed)
sys.path.insert(0, str(Path(__file__).parent.parent))

from cron.jobs import (
    AmbiguousJobReference,
    claim_job_for_fire,
    create_job,
    get_job,
    list_jobs,
    mark_job_run,
    parse_schedule,
    pause_job,
    remove_job,
    resolve_job_ref,
    resume_job,
    update_job,
)


def _enforce_production_job_candidate(job: Dict[str, Any]) -> None:
    """Apply the process-latched Cloud route contract when active."""

    from cron.production_policy import enforce_production_cron_job

    enforce_production_cron_job(job)


def _pin_production_create_route(
    provider: Optional[str],
    model: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    """Apply the production create defaults before candidate validation."""

    from cron.production_policy import pin_production_cron_create_route

    return pin_production_cron_create_route(provider, model)


def _notify_provider_jobs_changed_safe() -> None:
    """Tell the active cron scheduler provider the job set changed (no-op for
    the built-in). Best-effort — never lets a provider error break the tool."""
    try:
        from cron.scheduler import _notify_provider_jobs_changed
        _notify_provider_jobs_changed()
    except Exception:
        pass


def _origin_from_env() -> Optional[Dict[str, str]]:
    from gateway.session_context import get_session_env
    origin_platform = get_session_env("HERMES_SESSION_PLATFORM")
    origin_chat_id = get_session_env("HERMES_SESSION_CHAT_ID")
    if origin_platform and origin_chat_id:
        thread_id = get_session_env("HERMES_SESSION_THREAD_ID") or None
        if thread_id:
            logger.debug(
                "Cron origin captured thread_id=%s for %s:%s",
                thread_id, origin_platform, origin_chat_id,
            )
        return {
            "platform": origin_platform,
            "chat_id": origin_chat_id,
            "chat_name": get_session_env("HERMES_SESSION_CHAT_NAME") or None,
            "thread_id": thread_id,
            # Captured so an opt-in delivery mirror (cron.mirror_delivery /
            # attach_to_session) can resolve the exact participant's session in
            # per-user-isolated group chats — parity with interactive
            # send_message, which passes HERMES_SESSION_USER_ID to
            # gateway.mirror.mirror_to_session. Harmless for DMs/shared sessions.
            "user_id": get_session_env("HERMES_SESSION_USER_ID") or None,
        }
    return None


def _local_delivery_notice(job: Dict[str, Any], user_deliver: Optional[str]) -> Optional[str]:
    """Return an informational notice when a created job won't deliver anywhere.

    TUI/CLI sessions cannot be captured as a cron ``origin`` (no
    ``HERMES_SESSION_PLATFORM``/``CHAT_ID`` is set for them), so a
    ``deliver="origin"`` request — or an omitted ``deliver`` that defaults to
    origin-or-local — produces a job that runs and saves output to
    ``last_output`` but is never delivered back into the session. This is by
    design (there is no live-delivery channel for local sessions), but silently
    dropping the user's "tell me when it runs" intent is the trap reported in
    #51568. Surface it at create time so the agent can relay it instead of
    promising a delivery that never happens.

    Returns ``None`` when the user explicitly asked for ``local`` (no surprise),
    or when the job resolves to a real delivery target.
    """
    # An explicit local request is exactly what the user asked for — no notice.
    if (user_deliver or "").strip().lower() == "local":
        return None
    try:
        from cron.scheduler import _resolve_delivery_targets

        if _resolve_delivery_targets(job):
            return None  # Will actually deliver somewhere — nothing to flag.
    except Exception:
        # If resolution can't be evaluated, fall back to the origin signal.
        if job.get("origin"):
            return None
    return (
        "This is a local-only cron job: its output is saved (view it with "
        "cronjob(action='list')) but will NOT be delivered back into this "
        "session — CLI/TUI sessions have no live-delivery channel. To be "
        "notified when it runs, recreate or update the job with deliver set to "
        "a gateway-connected platform, e.g. deliver='telegram' or deliver='all'."
    )


def _repeat_display(job: Dict[str, Any]) -> str:
    times = (job.get("repeat") or {}).get("times")
    completed = (job.get("repeat") or {}).get("completed", 0)
    if times is None:
        return "forever"
    if times == 1:
        return "once" if completed == 0 else "1/1"
    return f"{completed}/{times}" if completed else f"{times} times"


def _canonical_skills(skill: Optional[str] = None, skills: Optional[Any] = None) -> List[str]:
    if skills is None:
        raw_items = [skill] if skill else []
    elif isinstance(skills, str):
        raw_items = [skills]
    else:
        raw_items = list(skills)

    normalized: List[str] = []
    for item in raw_items:
        text = str(item or "").strip()
        if text and text not in normalized:
            normalized.append(text)
    return normalized




def _resolve_model_override(model_obj: Optional[Dict[str, Any]]) -> tuple:
    """Resolve a model override object into (provider, model) for job storage.

    If provider is omitted, pins the current main provider from config so the
    job doesn't drift when the user later changes their default via hermes model.

    Returns (provider_str_or_none, model_str_or_none).
    """
    if not model_obj or not isinstance(model_obj, dict):
        return (None, None)
    model_name = (model_obj.get("model") or "").strip() or None
    provider_name = (model_obj.get("provider") or "").strip() or None
    # Bare "custom" is usually an incomplete spec — the canonical form is
    # "custom:<name>" matching a custom_providers entry, and LLMs frequently
    # supply the bare type because the schema does not advertise the
    # ":<name>" suffix. It is only a problem when it can't resolve at runtime:
    # a user may literally name a ``providers.custom`` (or custom_providers
    # "custom") entry, in which case the job should keep ``provider="custom"``
    # and run against that endpoint. Only when no such entry exists do we treat
    # the bare value as "no provider supplied" and pin the current main
    # provider below — otherwise pinning to ``model.provider`` (e.g. codex)
    # silently hijacks a job that meant to use the configured custom endpoint.
    if provider_name == "custom":
        try:
            from hermes_cli.runtime_provider import has_named_custom_provider
            if not has_named_custom_provider("custom"):
                provider_name = None
        except Exception:
            provider_name = None
    if model_name and not provider_name:
        # Pin to the current main provider so the job is stable
        try:
            from hermes_cli.config import load_config
            cfg = load_config()
            model_cfg = cfg.get("model", {})
            if isinstance(model_cfg, dict):
                provider_name = model_cfg.get("provider") or None
        except Exception:
            pass  # Best-effort; provider stays None
    return (provider_name, model_name)


def _normalize_optional_job_value(value: Optional[Any], *, strip_trailing_slash: bool = False) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if strip_trailing_slash:
        text = text.rstrip("/")
    return text or None


def _normalize_deliver_param(value: Any) -> Optional[str]:
    """Normalize a user-supplied ``deliver`` value to the canonical string form.

    The cron schema documents ``deliver`` as a string (``"local"``, ``"origin"``,
    ``"telegram"``, ``"telegram:chat_id[:thread_id]"``, or comma-separated combos).
    Some callers — MCP clients passing arrays, scripts building the payload as a
    list — supply ``["telegram"]``.  ``create_job``/``update_job`` store it as-is,
    and the scheduler's ``str(deliver).split(",")`` then serializes the list to
    the literal ``"['telegram']"`` which is not a known platform.  Flatten lists
    / tuples at the API boundary so storage is always a string.  Returns ``None``
    for ``None``/empty so callers can treat it as "not supplied".
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        parts = [str(p).strip() for p in value if str(p).strip()]
        return ",".join(parts) if parts else None
    text = str(value).strip()
    return text or None


def _validate_cron_base_url(
    provider: Optional[Any], base_url: Optional[Any]
) -> Optional[str]:
    """Reject pairing a named provider's stored credential with an off-host base_url.

    The cron tool is model-callable, so a prompt-injected job could set a real
    provider plus an attacker ``base_url``; on fire the scheduler resolves that
    provider's stored API key and sends it to the URL, exfiltrating the
    credential (CWE-200/CWE-522). Allow a ``base_url`` override only when it
    cannot leak a stored secret: no override at all, a configured custom/byok
    provider that carries its own endpoint+key, or an override whose host
    matches the named provider's own endpoint.

    Returns an error string if blocked, else None (valid).
    """
    bu = _normalize_optional_job_value(base_url, strip_trailing_slash=True)
    if not bu:
        return None
    prov = _normalize_optional_job_value(provider)
    if not prov:
        # A base_url with no explicit provider inherits the default/session
        # provider's stored key — the same exfil primitive without naming a
        # provider. Require an explicit (custom) provider for custom endpoints.
        return (
            "base_url override requires an explicit provider. Set provider to a "
            "configured custom provider to use a custom endpoint."
        )
    try:
        from hermes_cli.runtime_provider import (
            has_named_custom_provider,
            resolve_requested_provider,
            _get_named_custom_provider,
        )
        from hermes_cli.auth import PROVIDER_REGISTRY
        from utils import base_url_host_matches, base_url_hostname
    except Exception:
        # Can't resolve provider metadata -> fail closed.
        return f"Unable to validate base_url override for provider {prov!r}; refused."

    if prov.lower() == "custom":
        # Bare/inline 'custom' (and aliases that resolve to it) is pure BYOK: the
        # runtime derives the key from a pool keyed by THIS base_url or from
        # host-gated env vars, never an arbitrary stored secret. Safe to allow.
        return None
    if has_named_custom_provider(prov):
        # A NAMED custom provider carries a STORED key, and
        # _resolve_named_custom_runtime prefers the override base_url while still
        # sending that stored key — so an off-host override exfiltrates it.
        # Require the override host to match the provider's CONFIGURED endpoint.
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
            f"configured endpoint ({cfg_host or 'unknown'})."
        )
    try:
        resolved = resolve_requested_provider(prov)
    except Exception:
        resolved = prov
    pconfig = PROVIDER_REGISTRY.get(resolved) if isinstance(resolved, str) else None
    known_host = base_url_hostname(getattr(pconfig, "inference_base_url", "") if pconfig else "")
    if known_host and base_url_host_matches(bu, known_host):
        return None
    # Fail closed: any non-custom provider we cannot host-match to its own
    # endpoint is refused. This covers named providers with a stored credential
    # AND aliases/unknown names we can't resolve to a known host (e.g. "openai",
    # "google"), which would otherwise pair a stored key with the override URL.
    return (
        f"base_url {bu!r} is not allowed for provider {prov!r}. A named "
        f"provider's stored credential may only be sent to its own endpoint; "
        f'use a configured custom provider (provider="custom") for a custom base_url.'
    )


def _validate_cron_script_path(script: Optional[str]) -> Optional[str]:
    """Validate a cron job script path at the API boundary.

    Scripts must be relative paths that resolve within HERMES_HOME/scripts/.
    Absolute paths and ~ expansion are rejected to keep cron execution inside
    the operator-managed scripts directory.

    Returns an error string if blocked, else None (valid).
    """
    if not script or not script.strip():
        return None  # empty/None = clearing the field, always OK

    from hermes_constants import get_hermes_home

    raw = script.strip()

    # Reject absolute paths and ~ expansion at the API boundary.
    # Only relative paths within ~/.hermes/scripts/ are allowed.
    if raw.startswith(("/", "~")) or (len(raw) >= 2 and raw[1] == ":"):
        return (
            f"Script path must be relative to ~/.hermes/scripts/. "
            f"Got absolute or home-relative path: {raw!r}. "
            f"Place scripts in ~/.hermes/scripts/ and use just the filename."
        )

    # Validate containment after resolution
    from tools.path_security import validate_within_dir

    scripts_dir = get_hermes_home() / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    containment_error = validate_within_dir(scripts_dir / raw, scripts_dir)
    if containment_error:
        return (
            f"Script path escapes the scripts directory via traversal: {raw!r}"
        )

    return None


def _format_job(job: Dict[str, Any]) -> Dict[str, Any]:
    prompt = str(job.get("prompt") or "")
    skills = _canonical_skills(job.get("skill"), job.get("skills"))
    job_id = str(job.get("id") or "unknown")
    name = str(job.get("name") or prompt[:50] or (skills[0] if skills else "") or job_id or "cron job")
    result = {
        "job_id": job_id,
        "name": name,
        "skill": skills[0] if skills else None,
        "skills": skills,
        "prompt_preview": prompt[:100] + "..." if len(prompt) > 100 else prompt,
        "model": job.get("model"),
        "provider": job.get("provider"),
        "base_url": job.get("base_url"),
        "schedule": job.get("schedule_display") or "?",
        "repeat": _repeat_display(job),
        "deliver": job.get("deliver", "local"),
        "next_run_at": job.get("next_run_at"),
        "last_run_at": job.get("last_run_at"),
        "last_status": job.get("last_status"),
        "last_delivery_error": job.get("last_delivery_error"),
        "last_delivery_status": job.get("last_delivery_status", "none"),
        "last_delivery_confirmed_at": job.get("last_delivery_confirmed_at"),
        "enabled": job.get("enabled", True),
        "state": job.get("state", "scheduled" if job.get("enabled", True) else "paused"),
        "paused_at": job.get("paused_at"),
        "paused_reason": job.get("paused_reason"),
    }
    if job.get("script"):
        result["script"] = job["script"]
    if job.get("no_agent"):
        result["no_agent"] = True
    if job.get("enabled_toolsets"):
        result["enabled_toolsets"] = job["enabled_toolsets"]
    if job.get("workdir"):
        result["workdir"] = job["workdir"]
    return result


def _execute_job_now(job: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a cron job immediately, outside the scheduler tick.

    Atomically claims the job first via ``claim_job_for_fire`` — the same
    at-most-once CAS the scheduler/external-provider fire path uses — so a
    concurrently-running gateway ticker cannot also fire it (the claim both
    blocks a duplicate fire and advances ``next_run_at`` for recurring jobs).
    If the claim is lost (another fire is in flight), this is a no-op.

    The actual firing is delegated to ``run_one_job`` — the single shared
    execute→save→deliver→mark body the ticker and external providers use — so
    failure delivery, structured delivery-outcome handling, and live-adapter delivery stay
    identical across paths and can't drift.

    Returns {"claimed": bool, "success": bool, "error": str|None}.
    """
    job_id = job["id"]
    try:
        from cron.scheduler import run_one_job

        # At-most-once claim: bail without running if a tick/other fire owns it.
        if not claim_job_for_fire(job_id):
            # claim_job_for_fire returns False for paused/disabled/missing
            # jobs too — don't mislabel those as "already being fired"
            # (#60703): that message sends the user chasing a phantom
            # in-flight run when the job simply isn't runnable.
            refreshed = get_job(job_id)
            if refreshed is None:
                reason = "Job no longer exists; nothing to run."
            elif not refreshed.get("enabled", True) or refreshed.get("state") == "paused":
                reason = "Job is paused/disabled; resume it before running."
            else:
                reason = "Job is already being fired by the scheduler; not run again."
            return {"claimed": False, "success": False, "error": reason}

        # run_one_job records last_run_at/last_status via mark_job_run (which
        # also clears the fire claim) and returns True iff it processed the job.
        processed = run_one_job(job)
        refreshed = get_job(job_id) or {}
        ok = refreshed.get("last_status") == "ok"
        return {
            "claimed": True,
            "success": bool(processed and ok),
            "error": refreshed.get("last_error"),
        }

    except Exception as e:
        logger.error("Failed to execute cron job %s immediately: %s", job_id, e)
        try:
            mark_job_run(job_id, False, str(e))
        except Exception:
            pass
        return {"claimed": True, "success": False, "error": str(e)}


def cronjob(
    action: str,
    job_id: Optional[str] = None,
    prompt: Optional[str] = None,
    schedule: Optional[str] = None,
    name: Optional[str] = None,
    repeat: Optional[int] = None,
    deliver: Optional[str] = None,
    include_disabled: bool = False,
    skill: Optional[str] = None,
    skills: Optional[List[str]] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    reason: Optional[str] = None,
    script: Optional[str] = None,
    context_from: Optional[Union[str, List[str]]] = None,
    enabled_toolsets: Optional[List[str]] = None,
    workdir: Optional[str] = None,
    no_agent: Optional[bool] = None,
    attach_to_session: Optional[bool] = None,
    task_id: str = None,
) -> str:
    """Unified cron job management tool."""
    del task_id  # unused but kept for handler signature compatibility

    try:
        normalized = (action or "").strip().lower()

        if normalized == "create":
            if not schedule:
                return tool_error("schedule is required for create", success=False)
            canonical_skills = _canonical_skills(skill, skills)
            _no_agent = bool(no_agent)
            # Job-shape validation differs by mode:
            #   - no_agent=True → script is the job; prompt/skills are optional
            #     (and irrelevant to execution).
            #   - no_agent=False (default) → at least one of prompt/skills must
            #     be set, same as before.
            if _no_agent:
                if not script:
                    return tool_error(
                        "create with no_agent=True requires a script — "
                        "the script is the job.",
                        success=False,
                    )
            elif not prompt and not canonical_skills:
                return tool_error("create requires either prompt or at least one skill", success=False)
            # Validate script path before storing
            if script:
                script_error = _validate_cron_script_path(script)
                if script_error:
                    return tool_error(script_error, success=False)

            # Reject a model-supplied base_url that would route a named
            # provider's stored credential to an attacker endpoint (F8).
            base_url_error = _validate_cron_base_url(provider, base_url)
            if base_url_error:
                return tool_error(base_url_error, success=False)

            normalized_model = _normalize_optional_job_value(model)
            normalized_provider = _normalize_optional_job_value(provider)
            normalized_provider, normalized_model = _pin_production_create_route(
                normalized_provider,
                normalized_model,
            )
            normalized_base_url = _normalize_optional_job_value(
                base_url, strip_trailing_slash=True
            )
            normalized_script = _normalize_optional_job_value(script)
            normalized_deliver = _normalize_deliver_param(deliver)
            normalized_origin = _origin_from_env()
            effective_deliver = normalized_deliver or (
                "origin" if normalized_origin else "local"
            )
            _enforce_production_job_candidate(
                {
                    "enabled": True,
                    "no_agent": _no_agent,
                    "script": normalized_script,
                    "prompt": prompt or "",
                    "provider": normalized_provider,
                    "model": normalized_model,
                    "base_url": normalized_base_url,
                    "enabled_toolsets": enabled_toolsets or None,
                    "workdir": _normalize_optional_job_value(workdir),
                    "deliver": effective_deliver,
                    "origin": normalized_origin,
                }
            )

            # Validate context_from references existing jobs
            if context_from:
                from cron.jobs import get_job as _get_job
                refs = [context_from] if isinstance(context_from, str) else context_from
                for ref_id in refs:
                    if not _get_job(ref_id):
                        return tool_error(
                            f"context_from job '{ref_id}' not found. "
                            "Use cronjob(action='list') to see available jobs.",
                            success=False,
                        )

            job = create_job(
                prompt=prompt or "",
                schedule=schedule,
                name=name,
                repeat=repeat,
                deliver=normalized_deliver,
                origin=normalized_origin,
                skills=canonical_skills,
                model=normalized_model,
                provider=normalized_provider,
                base_url=normalized_base_url,
                script=normalized_script,
                context_from=context_from,
                enabled_toolsets=enabled_toolsets or None,
                workdir=_normalize_optional_job_value(workdir),
                no_agent=_no_agent,
                attach_to_session=attach_to_session,
            )
            _notify_provider_jobs_changed_safe()
            _create_message = f"Cron job '{job['name']}' created."
            _local_notice = _local_delivery_notice(job, _normalize_deliver_param(deliver))
            if _local_notice:
                _create_message = f"{_create_message} {_local_notice}"
            return json.dumps(
                {
                    "success": True,
                    "job_id": job["id"],
                    "name": job["name"],
                    "skill": job.get("skill"),
                    "skills": job.get("skills", []),
                    "schedule": job["schedule_display"],
                    "repeat": _repeat_display(job),
                    "deliver": job.get("deliver", "local"),
                    "next_run_at": job["next_run_at"],
                    "job": _format_job(job),
                    "message": _create_message,
                },
                indent=2,
            )

        if normalized == "list":
            jobs = [_format_job(job) for job in list_jobs(include_disabled=include_disabled)]
            return json.dumps({"success": True, "count": len(jobs), "jobs": jobs}, indent=2)

        if not job_id:
            return tool_error(f"job_id is required for action '{normalized}'", success=False)

        try:
            job = resolve_job_ref(job_id)
        except AmbiguousJobReference as exc:
            return json.dumps(
                {
                    "success": False,
                    "error": str(exc),
                    "matches": [
                        {
                            "id": m["id"],
                            "name": m.get("name"),
                            "schedule": m.get("schedule_display"),
                            "next_run_at": m.get("next_run_at"),
                        }
                        for m in exc.matches
                    ],
                },
                indent=2,
            )
        if not job:
            return json.dumps(
                {"success": False, "error": f"Job with ID or name '{job_id}' not found. Use cronjob(action='list') to inspect jobs."},
                indent=2,
            )
        # Resolve to canonical ID (supports name-based lookup)
        job_id = job["id"]

        if normalized == "remove":
            removed = remove_job(job_id)
            if not removed:
                return tool_error(f"Failed to remove job '{job_id}'", success=False)
            _notify_provider_jobs_changed_safe()
            return json.dumps(
                {
                    "success": True,
                    "message": f"Cron job '{job['name']}' removed.",
                    "removed_job": {
                        "id": job_id,
                        "name": job["name"],
                        "schedule": job.get("schedule_display"),
                    },
                },
                indent=2,
            )

        if normalized == "pause":
            updated = pause_job(job_id, reason=reason)
            _notify_provider_jobs_changed_safe()
            return json.dumps({"success": True, "job": _format_job(updated)}, indent=2)

        if normalized == "resume":
            _enforce_production_job_candidate(
                {**job, "enabled": True, "state": "scheduled"}
            )
            updated = resume_job(job_id)
            _notify_provider_jobs_changed_safe()
            return json.dumps({"success": True, "job": _format_job(updated)}, indent=2)

        if normalized in {"run", "run_now", "trigger"}:
            # Execute the job immediately rather than only scheduling it for the
            # next scheduler tick — a manual `run` should actually run, even when
            # no gateway/ticker is active (the #41037 case). The claim inside
            # _execute_job_now advances next_run_at and blocks a concurrent tick
            # from double-firing.
            exec_result = _execute_job_now(job)
            # Re-read so the response reflects the post-run last_run_at/last_status.
            result = _format_job(get_job(job_id) or {"id": job_id})
            result["executed"] = exec_result.get("claimed", False)
            result["execution_success"] = exec_result.get("success", False)
            if not exec_result.get("claimed", False):
                result["execution_skipped"] = exec_result.get("error") or (
                    "Already being fired by the scheduler; not run again."
                )
            elif exec_result.get("error"):
                result["execution_error"] = exec_result["error"]
            return json.dumps({"success": True, "job": result}, indent=2)

        if normalized == "update":
            updates: Dict[str, Any] = {}
            if prompt is not None:
                updates["prompt"] = prompt
            if name is not None:
                updates["name"] = name
            if deliver is not None:
                updates["deliver"] = _normalize_deliver_param(deliver)
            if skills is not None or skill is not None:
                canonical_skills = _canonical_skills(skill, skills)
                updates["skills"] = canonical_skills
                updates["skill"] = canonical_skills[0] if canonical_skills else None
            if model is not None:
                updates["model"] = _normalize_optional_job_value(model)
            if provider is not None:
                updates["provider"] = _normalize_optional_job_value(provider)
            if base_url is not None:
                updates["base_url"] = _normalize_optional_job_value(base_url, strip_trailing_slash=True)
            # Re-validate the EFFECTIVE provider/base_url on EVERY update, not
            # only when this update supplies provider/base_url. A job persisted
            # before this guard (or written directly to the jobs store) may
            # already hold an unsafe named-provider + off-host base_url pair;
            # if we only checked when the update touches those axes, editing any
            # unrelated field (name, schedule, ...) would succeed and leave that
            # exfil-capable pair active and schedulable (F8). The effective pair
            # merges this update's normalized values over the stored job; an
            # operator can still remediate in the same update by clearing
            # base_url or pointing provider/base_url at a safe pair.
            eff_provider = (
                updates["provider"] if "provider" in updates else job.get("provider")
            )
            eff_base_url = (
                updates["base_url"] if "base_url" in updates else job.get("base_url")
            )
            base_url_error = _validate_cron_base_url(eff_provider, eff_base_url)
            if base_url_error:
                return tool_error(base_url_error, success=False)
            if script is not None:
                # Pass empty string to clear an existing script
                if script:
                    script_error = _validate_cron_script_path(script)
                    if script_error:
                        return tool_error(script_error, success=False)
                updates["script"] = _normalize_optional_job_value(script) if script else None
            if context_from is not None:
                # Empty string / empty list clears the field; otherwise validate
                # each referenced job exists before storing. Normalized to a list
                # (or None) to match the shape stored by create_job().
                if isinstance(context_from, str):
                    refs = [context_from.strip()] if context_from.strip() else []
                else:
                    refs = [str(j).strip() for j in context_from if str(j).strip()]
                if refs:
                    from cron.jobs import get_job as _get_job
                    for ref_id in refs:
                        if not _get_job(ref_id):
                            return tool_error(
                                f"context_from job '{ref_id}' not found. "
                                "Use cronjob(action='list') to see available jobs.",
                                success=False,
                            )
                updates["context_from"] = refs or None
            if enabled_toolsets is not None:
                updates["enabled_toolsets"] = enabled_toolsets or None
            if attach_to_session is not None:
                updates["attach_to_session"] = bool(attach_to_session)
            if workdir is not None:
                # Empty string clears the field (restores old behaviour);
                # otherwise pass raw — update_job() validates / normalizes.
                updates["workdir"] = _normalize_optional_job_value(workdir) or None
            if no_agent is not None:
                # Toggling no_agent on/off at update time. If flipping to True,
                # we need a script to already exist on the job (or be part of
                # the same update) — otherwise the next tick would error out.
                target_no_agent = bool(no_agent)
                if target_no_agent:
                    effective_script = updates.get("script") if "script" in updates else job.get("script")
                    if not effective_script:
                        return tool_error(
                            "Cannot set no_agent=True on a job without a script. "
                            "Set `script` in the same update, or on the job first.",
                            success=False,
                        )
                updates["no_agent"] = target_no_agent
            if repeat is not None:
                # Normalize: treat 0 or negative as None (infinite)
                normalized_repeat = None if repeat <= 0 else repeat
                repeat_state = dict(job.get("repeat") or {})
                repeat_state["times"] = normalized_repeat
                updates["repeat"] = repeat_state
            if schedule is not None:
                parsed_schedule = parse_schedule(schedule)
                updates["schedule"] = parsed_schedule
                updates["schedule_display"] = parsed_schedule.get("display", schedule)
                if job.get("state") != "paused":
                    updates["state"] = "scheduled"
                    updates["enabled"] = True
            if not updates:
                return tool_error("No updates provided.", success=False)
            _enforce_production_job_candidate({**job, **updates})
            updated = update_job(job_id, updates)
            _notify_provider_jobs_changed_safe()
            return json.dumps({"success": True, "job": _format_job(updated)}, indent=2)

        return tool_error(f"Unknown cron action '{action}'", success=False)

    except Exception as e:
        return tool_error(str(e), success=False)



CRONJOB_SCHEMA = {
    "name": "cronjob",
    "description": """Manage scheduled cron jobs with a single compressed tool.

Use action='create' to schedule a new job from a prompt or one or more skills.
Use action='list' to inspect jobs.
Use action='update', 'pause', 'resume', 'remove', or 'run' to manage an existing job.

To stop a job the user no longer wants: first action='list' to find the job_id, then action='remove' with that job_id. Never guess job IDs — always list first.

Jobs run in a fresh session with no current-chat context, so prompts must be self-contained.
If skills are provided on create, the future cron run loads those skills in order, then follows the prompt as the task instruction.
On update, passing skills=[] clears attached skills.

NOTE: The agent's final response is auto-delivered to the target. Put the primary
user-facing content in the final response. Cron jobs run autonomously with no user
present — they cannot ask questions or request clarification.

Important safety rule: cron-run sessions should not recursively schedule more cron jobs.""",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "One of: create, list, update, pause, resume, remove, run. When action=create, the 'schedule' and 'prompt' fields are REQUIRED."
            },
            "job_id": {
                "type": "string",
                "description": "Required for update/pause/resume/remove/run"
            },
            "prompt": {
                "type": "string",
                "description": "For create: the full self-contained prompt. If skills are also provided, this becomes the task instruction paired with those skills."
            },
            "schedule": {
                "type": "string",
                "description": "REQUIRED for action=create. For create/update: '30m', 'every 2h', '0 9 * * *', or ISO timestamp. Examples: '30m' (every 30 minutes), 'every 2h' (every 2 hours), '0 9 * * *' (daily at 9am), '2026-06-01T09:00:00' (one-shot). You MUST include this field when action=create."
            },
            "name": {
                "type": "string",
                "description": "Optional human-friendly name"
            },
            "repeat": {
                "type": "integer",
                "description": "Optional repeat count. Omit for defaults (once for one-shot, forever for recurring)."
            },
            "deliver": {
                "type": "string",
                "description": "Omit this parameter to auto-deliver back to the current chat and topic (recommended). Auto-detection preserves thread/topic context. Only set explicitly when the user asks to deliver somewhere OTHER than the current conversation. Values: 'origin' (same as omitting), 'local' (no delivery, save only), 'all' (fan out to every connected home channel), or platform:chat_id:thread_id for a specific destination. Combine with comma: 'origin,all' delivers to the origin plus every other connected channel. Examples: 'telegram:-1001234567890:17585', 'discord:#engineering', 'sms:+15551234567', 'all'. WARNING: 'platform:chat_id' without :thread_id loses topic targeting. 'all' resolves at fire time, so a job created before a channel was wired up will pick it up automatically once connected."
            },
            "skills": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional ordered list of skill names to load before executing the cron prompt. On update, pass an empty array to clear attached skills."
            },
            "model": {
                "type": "object",
                "description": "Optional per-job model override. If provider is omitted, the current main provider is pinned at creation time so the job stays stable.",
                "properties": {
                    "provider": {
                        "type": "string",
                        "description": "Provider name (e.g. 'openrouter', 'anthropic', or 'custom:<name>' for a provider defined in custom_providers config — always include the ':<name>' suffix, never pass the bare 'custom'). Omit to use and pin the current provider."
                    },
                    "model": {
                        "type": "string",
                        "description": "Model name (e.g. 'anthropic/claude-sonnet-4', 'claude-sonnet-4')"
                    }
                },
                "required": ["model"]
            },
            "script": {
                "type": "string",
                "description": f"Optional path to a script that runs each tick. In the default mode its stdout is injected into the agent's prompt as context (data-collection / change-detection pattern). With no_agent=True, the script IS the job and its stdout is delivered verbatim (classic watchdog pattern). Relative paths resolve under {display_hermes_home()}/scripts/. ``.sh``/``.bash`` extensions run via bash, everything else via Python. On update, pass empty string to clear."
            },
            "no_agent": {
                "type": "boolean",
                "default": False,
                "description": (
                    "Default: False (LLM-driven job — the agent runs the prompt each tick). "
                    "Set True to skip the LLM entirely: the scheduler just runs ``script`` on schedule and delivers its stdout verbatim. No tokens, no agent loop, no model override honoured. "
                    "\n\n"
                    "REQUIREMENTS when True: ``script`` MUST be set (``prompt`` and ``skills`` are ignored). "
                    "\n\n"
                    "DELIVERY SEMANTICS when True: "
                    "(a) non-empty stdout is sent verbatim as the message; "
                    "(b) EMPTY stdout means no delivery — nothing is sent to the user, so design your script to stay quiet when there's nothing to report (the watchdog pattern); "
                    "(c) non-zero exit / timeout sends an error alert so a broken watchdog can't fail silently. "
                    "\n\n"
                    "WHEN TO USE True: recurring script-only pings where the script itself produces the exact message text (memory/disk/GPU watchdogs, threshold alerts, heartbeats, CI notifications, API pollers with a fixed output shape). "
                    "WHEN TO USE False (default): anything that needs reasoning — summarize a feed, draft a daily briefing, pick interesting items, rephrase data for a human, follow conditional logic based on content."
                ),
            },
            "context_from": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional job ID or list of job IDs whose most recent completed output is "
                    "injected into the prompt as context before each run. "
                    "Use this to chain cron jobs: job A collects data, job B processes it. "
                    "Each entry must be a valid job ID (from cronjob action='list'). "
                    "Note: injects the most recent completed output — does not wait for "
                    "upstream jobs running in the same tick. "
                    "On update, pass an empty array to clear."
                ),
            },
            "enabled_toolsets": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of toolset names to restrict the job's agent to (e.g. [\"web\", \"terminal\", \"file\", \"delegation\"]). When set, only tools from these toolsets are loaded, significantly reducing input token overhead. When omitted, all default tools are loaded. Infer from the job's prompt — e.g. use \"web\" if it calls web_search, \"terminal\" if it runs scripts, \"file\" if it reads files, \"delegation\" if it calls delegate_task. On update, pass an empty array to clear."
            },
            "workdir": {
                "type": "string",
                "description": "Optional absolute path to run the job from. When set, AGENTS.md / CLAUDE.md / .cursorrules from that directory are injected into the system prompt, and the terminal/file/code_exec tools use it as their working directory — useful for running a job inside a specific project repo. Must be an absolute path that exists. When unset (default), preserves the original behaviour: no project context files, tools use the scheduler's cwd. On update, pass an empty string to clear. Jobs with workdir run sequentially (not parallel) to keep per-job directories isolated."
            },
            "attach_to_session": {
                "type": "boolean",
                "description": "When True, this job becomes CONTINUABLE: the user can reply to its delivery and the agent has the brief in context instead of asking 'what is that?'. On thread-capable platforms (Telegram topics, Discord/Slack threads) a dedicated thread is opened for the job and its replies; on DM-only platforms (WhatsApp/Signal) the brief is mirrored into the origin DM session. Use this for conversational recurring jobs the user will reply to — daily briefings, reminders that kick off follow-up work. Leave unset for fire-and-forget alerts/watchdogs. Overrides the global cron.mirror_delivery config for this one job. Only the origin chat is touched (never fan-out targets); no effect when deliver='local'."
            },
        },
        "required": ["action"]
    }
}


def check_cronjob_requirements() -> bool:
    """
    Check if cronjob tools can be used.

    Available in interactive CLI mode and gateway/messaging platforms.
    The cron system is internal (JSON file-based scheduler ticked by the gateway),
    so no external crontab executable is required.

    Legacy session env vars must hold an explicit truthy string (``1``,
    ``true``, ``yes``, ``on``).  Concurrent gateways use the task-local
    ``HERMES_SESSION_PLATFORM`` ContextVar instead of a process-global env
    marker; accept that exact bound session too.  A cron execution context is
    always excluded so an unattended job cannot acquire the management tool
    merely because it runs inside a gateway process.
    """
    from utils import env_var_enabled

    if env_var_enabled("HERMES_CRON_SESSION"):
        return False

    try:
        from gateway.session_context import get_session_env

        gateway_platform = get_session_env("HERMES_SESSION_PLATFORM", "")
    except Exception:
        gateway_platform = ""

    return bool(gateway_platform) or (
        env_var_enabled("HERMES_INTERACTIVE")
        or env_var_enabled("HERMES_GATEWAY_SESSION")
        or env_var_enabled("HERMES_EXEC_ASK")
    )


# --- Registry ---
from tools.registry import registry, tool_error

registry.register(
    name="cronjob",
    toolset="cronjob",
    schema=CRONJOB_SCHEMA,
    handler=lambda args, **kw: (lambda _mo=_resolve_model_override(args.get("model")): cronjob(
        action=args.get("action", ""),
        job_id=args.get("job_id"),
        prompt=args.get("prompt"),
        schedule=args.get("schedule"),
        name=args.get("name"),
        repeat=args.get("repeat"),
        deliver=args.get("deliver"),
        include_disabled=args.get("include_disabled", True),
        skill=args.get("skill"),
        skills=args.get("skills"),
        model=_mo[1],
        provider=_mo[0] or args.get("provider"),
        base_url=args.get("base_url"),
        reason=args.get("reason"),
        script=args.get("script"),
        context_from=args.get("context_from"),
        enabled_toolsets=args.get("enabled_toolsets"),
        workdir=args.get("workdir"),
        no_agent=args.get("no_agent"),
        task_id=kw.get("task_id"),
    ))(),
    check_fn=check_cronjob_requirements,
    emoji="⏰",
)
