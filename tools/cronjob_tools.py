"""
Cron job management tools for Hermes Agent.

Expose a single compressed action-oriented tool to avoid schema/context bloat.
Compatibility wrappers remain for direct Python callers and legacy tests.
"""

import json
import logging
import re
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


def _notify_provider_jobs_changed_safe() -> None:
    """Tell the active cron scheduler provider the job set changed (no-op for
    the built-in). Best-effort — never lets a provider error break the tool."""
    try:
        from cron.scheduler import _notify_provider_jobs_changed
        _notify_provider_jobs_changed()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Cron prompt scanning
# ---------------------------------------------------------------------------
#
# Two threat surfaces, two scanners:
#
#   1. User-supplied cron prompt (small, written as a directive).
#      Strict scanning is appropriate — a legit cron prompt has no business
#      saying "cat ~/.hermes/.env" or "rm -rf /". `_scan_cron_prompt()` runs
#      against this at create/update time and as a runtime defense-in-depth.
#
#   2. Assembled prompt that includes loaded skill content (large markdown
#      bodies, often security docs, postmortems, runbooks discussing attack
#      patterns in PROSE). Reusing the strict patterns here false-positives
#      every time a skill *describes* a command — see #3968 follow-up: the
#      `hermes-agent-dev` skill contains a security postmortem mentioning
#      `cat ~/.hermes/.env`, which tripped `read_secrets` and silently
#      killed all PR-scout jobs.
#
#      Skill bodies are user-curated and scanned at install time by
#      `skills_guard.py`. The runtime cron scan only needs to catch the
#      patterns whose phrasing does NOT survive normal English prose:
#      classic prompt-injection directives ("ignore previous instructions",
#      "disregard your rules"), deception directives, and invisible
#      unicode. `_scan_cron_skill_assembled()` runs against the assembled
#      prompt with this tighter pattern set.
#
# Both scanners share the invisible-unicode check and the GitHub Authorization
# header exemption.

# Strict patterns — applied to the user prompt only.
_CRON_THREAT_PATTERNS = [
    (r'ignore\s+(?:\w+\s+)*(?:previous|all|above|prior)\s+(?:\w+\s+)*instructions', "prompt_injection"),
    (r'do\s+not\s+tell\s+the\s+user', "deception_hide"),
    (r'system\s+prompt\s+override', "sys_prompt_override"),
    (r'disregard\s+(your|all|any)\s+(instructions|rules|guidelines)', "disregard_rules"),
    (r'cat\s+[^\n]*(\.env|credentials|\.netrc|\.pgpass)', "read_secrets"),
    (r'authorized_keys', "ssh_backdoor"),
    (r'/etc/sudoers|visudo', "sudoers_mod"),
    (r'rm\s+-rf\s+/', "destructive_root_rm"),
]

# Looser pattern set — applied to the assembled prompt when skills are
# attached. Only patterns whose phrasing is unambiguous in any context;
# command-shape patterns are dropped because they false-positive on prose
# in security docs / postmortems. Skill bodies are scanned at install time
# by `skills_guard.py`, so the runtime cron scan is purely a tripwire for
# obvious injection directives surviving a malicious skill that slipped
# through install.
_CRON_SKILL_ASSEMBLED_PATTERNS = [
    (r'ignore\s+(?:\w+\s+)*(?:previous|all|above|prior)\s+(?:\w+\s+)*instructions', "prompt_injection"),
    (r'do\s+not\s+tell\s+the\s+user', "deception_hide"),
    (r'system\s+prompt\s+override', "sys_prompt_override"),
    (r'disregard\s+(your|all|any)\s+(instructions|rules|guidelines)', "disregard_rules"),
]

_CRON_SECRET_VAR_RE = r'\$\{?\w*(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)\w*\}?'
_CRON_EXFIL_COMMAND_PATTERNS = [
    # Tighten exfil detection to obvious leak paths: embedding a secret
    # directly in the destination URL, sending it in POST/FORM payloads,
    # or shipping it via Authorization headers to arbitrary hosts. The
    # only intended allowlist exception today is the bundled GitHub skill
    # pattern that talks to api.github.com.
    (rf'curl\s+[^\n]*https?://[^\s"\'`]*{_CRON_SECRET_VAR_RE}', "exfil_curl_url"),
    (rf'wget\s+[^\n]*https?://[^\s"\'`]*{_CRON_SECRET_VAR_RE}', "exfil_wget_url"),
    (rf'curl\s+[^\n]*(?:--data(?:-raw|-binary|-urlencode)?|-d|--form|-F)\s+[^\n]*{_CRON_SECRET_VAR_RE}', "exfil_curl_data"),
    (rf'wget\s+[^\n]*--post-(?:data|file)=[^\n]*{_CRON_SECRET_VAR_RE}', "exfil_wget_post"),
    (rf'curl\s+[^\n]*(?:-H|--header)\s+["\']Authorization:\s*(?:Bearer|token)\s+{_CRON_SECRET_VAR_RE}["\']', "exfil_curl_auth_header"),
]

# Single source of truth, shared with the install-time scanner
# (threat_patterns.INVISIBLE_CHARS / skills_guard). Keeping a separate, narrower
# copy here let an obfuscated injection directive slip past this runtime cron
# tripwire while being caught at install time (or vice versa): U+2062-U+2064
# (invisible math operators) and U+2066-U+2069 (directional isolates) are real
# attack tools and were missing from the cron-local set. Importing the canonical
# set keeps the cron tripwire and the install scanner from drifting apart.
from tools.threat_patterns import INVISIBLE_CHARS as _CRON_INVISIBLE_CHARS

# U+200D Zero-Width Joiner is also a legitimate, required part of many
# Unicode emoji sequences (for example 👨‍👩‍👧, 🏳️‍🌈, ❤️‍🩹, 🧑‍💻).
# We should still block ZWJ when it is hiding between plain text characters,
# but not when it is clearly part of an emoji grapheme cluster.
_EMOJI_NEIGHBOUR_CP_RANGES = (
    (0x1F000, 0x1FFFF),
    (0x2600, 0x27BF),
    (0x2300, 0x23FF),
    (0x1F1E6, 0x1F1FF),
    (0x20E3, 0x20E3),
)
_VARIATION_SELECTOR_CP = 0xFE0F


def _is_emoji_cp(cp: int) -> bool:
    return any(lo <= cp <= hi for lo, hi in _EMOJI_NEIGHBOUR_CP_RANGES)


def _zwj_has_emoji_neighbour(text: str, idx: int) -> bool:
    """Return True when the ZWJ at text[idx] appears inside an emoji sequence."""
    left = idx - 1
    while left >= 0 and ord(text[left]) == _VARIATION_SELECTOR_CP:
        left -= 1
    right = idx + 1
    while right < len(text) and ord(text[right]) == _VARIATION_SELECTOR_CP:
        right += 1
    return (
        left >= 0 and right < len(text)
        and _is_emoji_cp(ord(text[left]))
        and _is_emoji_cp(ord(text[right]))
    )


def _strip_legitimate_emoji_zwj(prompt: str) -> str:
    if '\u200d' not in prompt:
        return prompt
    cleaned: list[str] = []
    for idx, ch in enumerate(prompt):
        if ch == '\u200d' and _zwj_has_emoji_neighbour(prompt, idx):
            continue
        cleaned.append(ch)
    return ''.join(cleaned)


def _strip_cron_safe_constructs(prompt: str) -> str:
    """Strip the GitHub `Authorization: token $GITHUB_TOKEN` auth-header
    pattern so it doesn't trip the broader curl-auth-header exfil rule.

    Allows the bundled GitHub skill fallback without opening a blanket
    exemption for arbitrary Authorization-header exfiltration.
    """
    github_auth_header = re.search(
        rf'curl\s+[^\n]*(?:-H|--header)\s+["\']Authorization:\s*token\s+{_CRON_SECRET_VAR_RE}["\']'
        r'\s+["\']?https://api\.github\.com(?:/|\b)',
        prompt,
        re.IGNORECASE,
    )
    if github_auth_header:
        return prompt.replace(github_auth_header.group(0), "curl https://api.github.com/user")
    return prompt


def _check_invisible_unicode(prompt: str) -> str:
    """Return an error string if the prompt contains invisible-unicode
    injection markers (ZWJ inside legitimate emoji sequences is allowed).
    """
    prompt_for_invisible_scan = _strip_legitimate_emoji_zwj(prompt)
    for char in _CRON_INVISIBLE_CHARS:
        if char in prompt_for_invisible_scan:
            return f"Blocked: prompt contains invisible unicode U+{ord(char):04X} (possible injection)."
    return ""


def _strip_invisible_unicode(prompt: str) -> tuple[str, list[str]]:
    """Strip invisible-unicode characters from *prompt*, preserving the ZWJ
    that lives inside legitimate emoji sequences.

    Returns ``(cleaned_prompt, removed_codepoints)`` where ``removed_codepoints``
    is the sorted list of ``U+XXXX`` labels that were stripped (empty when the
    prompt was already clean). Used by the skills-attached cron path, where the
    skill body is already vetted at install time by ``skills_guard.py`` — a
    stray zero-width space in a code example should be sanitized, not turned
    into a hard block that permanently kills the job.
    """
    if not prompt:
        return prompt, []
    # Keep emoji-ZWJ: temporarily remove the legitimate joiners, scan/strip the
    # rest, then the legitimate joiners survive because we operate on the
    # original string and only drop chars that are NOT part of an emoji cluster.
    removed: set[str] = set()
    cleaned: list[str] = []
    for idx, ch in enumerate(prompt):
        if ch in _CRON_INVISIBLE_CHARS:
            if ch == '\u200d' and _zwj_has_emoji_neighbour(prompt, idx):
                cleaned.append(ch)  # legitimate emoji joiner — keep
                continue
            removed.add(f"U+{ord(ch):04X}")
            continue
        cleaned.append(ch)
    return ''.join(cleaned), sorted(removed)


def _scan_cron_prompt(prompt: str) -> str:
    """Scan the USER-SUPPLIED cron prompt for critical threats.

    Strict pattern set — used at job create/update time and as a runtime
    defense-in-depth for prompts authored before the scanner existed.
    The user prompt is small and directive; bare `cat .env` or `rm -rf /`
    there is a smoking gun, not prose. Returns an error string when
    blocked, else empty string.
    """
    prompt_to_scan = _strip_cron_safe_constructs(prompt)
    invisible_err = _check_invisible_unicode(prompt_to_scan)
    if invisible_err:
        return invisible_err
    for pattern, pid in _CRON_THREAT_PATTERNS:
        if re.search(pattern, prompt_to_scan, re.IGNORECASE):
            return f"Blocked: prompt matches threat pattern '{pid}'. Cron prompts must not contain injection or exfiltration payloads."
    for pattern, pid in _CRON_EXFIL_COMMAND_PATTERNS:
        if re.search(pattern, prompt_to_scan, re.IGNORECASE):
            return f"Blocked: prompt matches threat pattern '{pid}'. Cron prompts must not contain injection or exfiltration payloads."
    return ""


def _scan_cron_skill_assembled(assembled: str) -> tuple[str, str]:
    """Scan an ASSEMBLED cron prompt that includes loaded skill content.

    Looser pattern set — only catches unambiguous prompt-injection
    directives. Drops command-shape patterns (cat .env, rm -rf /,
    authorized_keys, /etc/sudoers) because they false-positive on
    legitimate skill markdown that *describes* attack commands in
    security postmortems and runbooks.

    Invisible unicode is SANITIZED, not blocked. Skill bodies are
    user-curated and already scanned at install time by
    ``skills_guard.py``; a stray zero-width space in a code example
    (common in copy-pasted unicode docs) should not permanently kill the
    job. The offending codepoints are stripped and logged, the cleaned
    prompt is returned. The hard block remains for raw user prompts via
    ``_scan_cron_prompt`` — that path is the actual injection surface.

    Returns ``(cleaned_prompt, error)``; ``error`` is empty when the
    prompt passed (after sanitization).
    """
    cleaned, removed = _strip_invisible_unicode(assembled)
    if removed:
        logger.warning(
            "Cron skill-assembled prompt: stripped %d invisible-unicode "
            "char(s) (%s) from vetted skill content",
            len(removed), ", ".join(removed),
        )
    prompt_to_scan = _strip_cron_safe_constructs(cleaned)
    for pattern, pid in _CRON_SKILL_ASSEMBLED_PATTERNS:
        if re.search(pattern, prompt_to_scan, re.IGNORECASE):
            return cleaned, f"Blocked: prompt matches threat pattern '{pid}'. Cron prompts must not contain injection or exfiltration payloads."
    return cleaned, ""


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
    Absolute paths and ~ expansion are rejected to prevent arbitrary script
    execution via prompt injection.

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


def _manual_run_should_detach() -> bool:
    """True for persistent gateway sessions that can receive async delivery."""
    try:
        from gateway.session_context import async_delivery_supported, get_session_env

        platform = get_session_env("HERMES_SESSION_PLATFORM", "").strip()
        return bool(platform and async_delivery_supported())
    except Exception:
        return False


def _execute_job_now(job: Dict[str, Any], *, wait: bool = True) -> Dict[str, Any]:
    """Execute a cron job immediately, outside the scheduler tick.

    Atomically claims the job first via ``claim_job_for_fire`` — the same
    at-most-once CAS the scheduler/external-provider fire path uses — so a
    concurrently-running gateway ticker cannot also fire it (the claim both
    blocks a duplicate fire and advances ``next_run_at`` for recurring jobs).
    If the claim is lost (another fire is in flight), this is a no-op.

    The actual firing is delegated to ``run_one_job`` — the single shared
    execute→save→deliver→mark body the ticker and external providers use — so
    failure delivery, ``[SILENT]`` handling, and live-adapter delivery stay
    identical across paths and can't drift.

    When ``wait`` is false, dispatches through the scheduler's persistent pool
    and returns as soon as the job has started. CLI/standalone callers keep the
    historical synchronous behavior by using the default ``wait=True``.

    Returns {"claimed": bool, "started": bool, "success": bool|None,
    "error": str|None}.
    """
    job_id = job["id"]
    try:
        from cron.scheduler import run_one_job, submit_claimed_job

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
            return {"claimed": False, "started": False, "success": False, "error": reason}

        if not wait:
            future = submit_claimed_job(job)
            if future is None:
                error = "Cron scheduler could not start the claimed job."
                mark_job_run(job_id, False, error)
                return {"claimed": True, "started": False, "success": False, "error": error}
            return {"claimed": True, "started": True, "success": None, "error": None}

        # run_one_job records last_run_at/last_status via mark_job_run (which
        # also clears the fire claim) and returns True iff it processed the job.
        processed = run_one_job(job)
        refreshed = get_job(job_id) or {}
        ok = refreshed.get("last_status") == "ok"
        return {
            "claimed": True,
            "started": True,
            "success": bool(processed and ok),
            "error": refreshed.get("last_error"),
        }

    except Exception as e:
        logger.error("Failed to execute cron job %s immediately: %s", job_id, e)
        try:
            mark_job_run(job_id, False, str(e))
        except Exception:
            pass
        return {"claimed": True, "started": True, "success": False, "error": str(e)}


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
            if prompt:
                scan_error = _scan_cron_prompt(prompt)
                if scan_error:
                    return tool_error(scan_error, success=False)

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
                deliver=_normalize_deliver_param(deliver),
                origin=_origin_from_env(),
                skills=canonical_skills,
                model=_normalize_optional_job_value(model),
                provider=_normalize_optional_job_value(provider),
                base_url=_normalize_optional_job_value(base_url, strip_trailing_slash=True),
                script=_normalize_optional_job_value(script),
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
            updated = resume_job(job_id)
            _notify_provider_jobs_changed_safe()
            return json.dumps({"success": True, "job": _format_job(updated)}, indent=2)

        if normalized in {"run", "run_now", "trigger"}:
            # Execute the job immediately rather than only scheduling it for the
            # next scheduler tick — a manual `run` should actually run, even when
            # no gateway/ticker is active (the #41037 case). The claim inside
            # _execute_job_now advances next_run_at and blocks a concurrent tick
            # from double-firing.
            detached = _manual_run_should_detach()
            exec_result = _execute_job_now(job, wait=not detached)
            # Re-read so the response reflects the post-run last_run_at/last_status.
            result = _format_job(get_job(job_id) or {"id": job_id})
            result["executed"] = exec_result.get("claimed", False)
            result["execution_started"] = exec_result.get("started", False)
            if exec_result.get("success") is None:
                result["execution_pending"] = True
            else:
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
                scan_error = _scan_cron_prompt(prompt)
                if scan_error:
                    return tool_error(scan_error, success=False)
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
                    "(b) EMPTY stdout means SILENT — nothing is sent to the user and they won't see anything happened, so design your script to stay quiet when there's nothing to report (the watchdog pattern); "
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

    Session env vars must hold an explicit truthy string (``1``, ``true``,
    ``yes``, ``on``) — false-like values (``0``, ``false``, ``no``, ``off``)
    leave the tool disabled. Uses the shared ``env_var_enabled`` helper so
    every consumer of these flags agrees on the truthy set.
    """
    from utils import env_var_enabled

    return (
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
