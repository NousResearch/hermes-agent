"""Cron job management tool: one compressed action-oriented `cronjob_manage` tool
(schema/context bloat avoided); `cronjob()` stays callable for direct Python callers."""

import contextlib
import json
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import copy

from hermes_constants import display_hermes_home

logger = logging.getLogger(__name__)

# Heartbeat cadence keeping the caller's inactivity watchdog at bay while a manual
# `cronjob(action="run")` executes in-process (comfortably below HERMES_AGENT_TIMEOUT).
# Mirrors the 10s cadence of tools/environments/base.py::touch_activity_if_due (delegate_task's heartbeat
# uses 30s) — comfortably below the 1800s default HERMES_AGENT_TIMEOUT. See #76502.
_CRON_RUN_HEARTBEAT_INTERVAL = 10.0
# Hard ceiling: with HERMES_CRON_TIMEOUT=0 a truly hung run would otherwise mask the
# gateway watchdog forever; past this the heartbeat stops and the watchdog regains authority.
# The child cron run has its own inactivity watchdog (HERMES_CRON_TIMEOUT, default 600s) that bounds a
# wedged job, but with HERMES_CRON_TIMEOUT=0 (explicit "unlimited") a truly hung run_one_job would otherwise
# mask the gateway watchdog forever — pre-#76502 the parent was at least reaped at ~1800s.
_CRON_RUN_HEARTBEAT_CEILING = 6 * 3600.0

sys.path.insert(0, str(Path(__file__).parent.parent))

from cron.jobs import (
    AmbiguousJobReference,
    claim_job_for_fire,
    get_job,
    is_job_runnable,
    list_jobs,
    mark_job_run,
    parse_schedule,
    pause_job,
    remove_job,
    resolve_job_ref,
    resnapshot_all_unpinned,
    resnapshot_job,
    resume_job,
    update_job)
from tools.cronjob_prompt_scan import _scan_cron_prompt
from tools.cronjob_job_args import (
    _apply_continuity,
    _canonical_skills,
    _clean_str_list,
    _format_job,
    _gateway_liveness_notice,
    _local_delivery_notice,
    _mode_guidance_notes,
    _normalize_deliver_param,
    _normalize_optional_job_value,
    _origin_from_env,
    _repeat_display,
    _resolve_cron_context_deliver,
    _split_monitor_arg,
    _validate_bot_chat_deliver,
    _validate_context_from_refs,
    _validate_cron_base_url,
    _validate_cron_script_path)
from tools.registry import registry, tool_error


def _dumps(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, indent=2)


def _notify_provider_jobs_changed_safe() -> None:
    """Tell the active scheduler provider the job set changed; best-effort, never raises."""
    try:
        from cron.scheduler import _notify_provider_jobs_changed
        _notify_provider_jobs_changed()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Manual run execution (claim -> run_one_job -> report)
# ---------------------------------------------------------------------------

# Strict patterns — applied to the user prompt only.
_CRON_THREAT_PATTERNS = [
    (r'ignore\s+(?:\w+\s+)*(?:previous|all|above|prior)\s+(?:\w+\s+)*instructions', "prompt_injection"),
    (r'do\s+not\s+tell\s+the\s+user', "deception_hide"),
    (r'system\s+prompt\s+override', "sys_prompt_override"),
    (r'disregard\s+(your|all|any)\s+(instructions|rules|guidelines)', "disregard_rules"),
    (r'cat\s+[^\n]*(\.env|credentials|\.netrc|\.pgpass|id_rsa|id_ed25519|id_ecdsa)', "read_secrets"),
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

    Uses ``re.sub`` so EVERY occurrence is scrubbed, not just the first — a
    cron job that loads 2+ GitHub skills (e.g. github-issues +
    github-pr-workflow + github-code-review) contains several such blocks,
    and the old ``re.search`` + single ``str.replace`` left the rest to trip
    the exfil_curl_auth_header detector on every run. The trailing
    ``[^\\s;&|$`]*`` consumes only the URL path — never whitespace, command
    separators, or subshell openers — so a payload smuggled onto the same
    line (``;``, ``&&``, ``|``, ``$(...)``, backticks) survives the strip
    and is still scanned. The host must be exactly ``api.github.com``
    followed by ``/``, whitespace, quote, or end: lookalike authorities
    (``api.github.com.evil.com``, ``api.github.com@evil.com``) are not the
    trusted construct and fall through to the exfil detectors, while
    legitimately quoted bare-host URLs stay exempt.
    """
    return re.sub(
        rf'curl\s+[^\n;&|$`]*(?:-H|--header)\s+["\']Authorization:\s*token\s+{_CRON_SECRET_VAR_RE}["\']'
        r'\s+["\']?https://api\.github\.com(?::\d+)?(?:/|\s|$|["\'])[^\s;&|$`]*',
        'curl https://api.github.com/user',
        prompt,
        flags=re.IGNORECASE,
    )


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
        # Slack thread-per-message session keying (native parity: thread_ts =
        # event.thread_ts or ts) stamps every TOP-LEVEL message's own id as
        # the session thread. That stamp is a per-message session KEY, not a
        # durable conversation location — persisting it as origin routing
        # pins every future delivery inside the ephemeral thread spawned
        # around the creation message. Recognize it at the source: a Slack
        # thread id equal to the triggering message's own id is synthetic.
        # A genuine in-thread creation (thread == the parent's id != this
        # message's id) keeps its thread.
        if thread_id and origin_platform == "slack":
            message_id = get_session_env("HERMES_SESSION_MESSAGE_ID") or None
            if message_id and str(thread_id) == str(message_id):
                logger.debug(
                    "Cron origin: dropping synthetic per-message Slack "
                    "thread_id=%s (== creation message id)", thread_id,
                )
                thread_id = None
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
            # Workspace/server scope (Slack team, Discord guild, Matrix
            # server). build_session_key embeds it in every Slack session key
            # (dm/group/thread alike), so a continuable cron seed built
            # WITHOUT it creates a row no scoped reply ever resolves to —
            # the seeded key is agent:main:slack:dm:<chat>:<thread> while the
            # reply keys agent:main:slack:dm:<team>:<chat>:<thread>. Captured
            # here so the scheduler's seed helpers can reproduce the reply's
            # exact key. Same session-context var async_delegation already
            # snapshots; None for platforms without scope.
            "scope_id": get_session_env("HERMES_SESSION_SCOPE_ID") or None,
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
        targets = _resolve_delivery_targets(job) or []
    except Exception:
        return set()
    return {t.get("platform") for t in targets if t.get("platform")} & fronted


def _mode_guidance_notes(job: Dict[str, Any], user_deliver: Optional[str]) -> List[str]:
    """Mode-specific guidance echoed in the create/update response.

    The teaching that used to live in CRONJOB_SCHEMA parameter descriptions
    (paid for on every API call of every session) is delivered here instead —
    once, in the tool result, at the moment the model actually created a job
    in that mode. Keep each note short and actionable; only fire notes for
    modes the job actually uses.
    """
    notes: List[str] = []
    if job.get("monitor_script") or job.get("monitor_url"):
        notes.append(
            "Monitor mode: the source runs first each tick and its output is "
            "hashed as exact bytes — unchanged output suppresses the agent run "
            "(silent no_change tick), changed output injects a MONITOR CHANGE "
            "DETECTED diff into the prompt. The first tick always runs as "
            "baseline. The source must emit STABLE output (no timestamps, no "
            "random ordering) or every tick will look changed."
        )
    if job.get("no_agent"):
        notes.append(
            "no_agent mode: stdout is delivered verbatim; EMPTY stdout sends "
            "nothing at all (watchdog pattern — script should stay quiet when "
            "there is nothing to report). Non-zero exit or timeout sends an "
            "error alert. prompt/skills are ignored."
        )
    _deliver = (user_deliver or "").strip().lower()
    if _deliver:
        if "all" in _deliver.split(","):
            notes.append(
                "deliver='all' resolves at fire time and never includes "
                "bot-chat targets — channels connected later are picked up "
                "automatically."
            )
        if _deliver.startswith("bot-chat:"):
            notes.append(
                "Targeting another profile's Bot Chat costs that bot an agent "
                "turn per run."
            )
        # platform:chat_id with no thread segment loses topic targeting —
        # warn once here instead of carrying the warning in the schema.
        for target in _deliver.split(","):
            parts = target.strip().split(":")
            if (
                len(parts) == 2
                and parts[0] not in ("bot-chat", "sms")
                and parts[1]
                and not parts[1].startswith("#")
            ):
                notes.append(
                    f"deliver target '{target.strip()}' has no :thread_id "
                    "segment — on thread/topic platforms the delivery lands in "
                    "the main chat, not a topic."
                )
                break
    return notes


def _split_monitor_arg(
    monitor: Optional[str],
    monitor_script: Optional[str],
    monitor_url: Optional[str],
) -> tuple:
    """Resolve the model-facing ``monitor`` field into the stored pair.

    The schema advertises ONE ``monitor`` field; the value's shape decides the
    transport: ``http(s)://...`` is a URL source, anything else is a script
    path (a legal script path can never start with a URL scheme). Jobs keep
    storing ``monitor_script``/``monitor_url`` separately — this is an
    interface merge, not a storage migration — and the legacy field names are
    still accepted as aliases so older transcripts/replays keep working.

    Returns ``(monitor_script, monitor_url)`` with update semantics:
    ``None`` = leave unchanged, ``''`` = clear. Setting one source via
    ``monitor`` clears the other, so switching transports in one call never
    trips the mutual-exclusion invariant. An explicit ``monitor`` wins over
    the legacy aliases.
    """
    if monitor is None:
        return monitor_script, monitor_url
    value = monitor.strip()
    if not value:
        return "", ""  # clear both sources
    if value.lower().startswith(("http://", "https://")):
        return "", value
    return value, ""


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


def _validate_bot_chat_deliver(deliver: Optional[str]) -> Optional[str]:
    """Validate any ``bot-chat[:<profile>]`` deliver elements at create time.

    Bot Chat delivery is machine-local: the named profile must exist on THIS
    machine (the one whose scheduler will fire the job). Failing loudly here
    beats a per-run ``last_delivery_error`` at 3am — especially for Desktop
    clients whose merged multi-gateway rosters may show same-named profiles
    from other machines. Returns an error string or None.
    """
    if not deliver:
        return None
    try:
        from cron.scheduler import parse_bot_chat_deliver_token
        from hermes_cli.profiles import normalize_profile_name, profile_exists
    except Exception:
        return None  # validation is best-effort; resolution re-checks at fire time
    for part in str(deliver).split(","):
        profile_arg = parse_bot_chat_deliver_token(part.strip())
        if profile_arg is None or not profile_arg:
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
                "name (deliver='bot-chat') for the job's own profile."
            )
    return None


def _resolve_cron_context_deliver(deliver: Optional[str]) -> Optional[str]:
    """Resolve ``origin`` to a concrete target for cron-context creates.

    A job created FROM a cron run must never store the literal ``origin``:
    the creating session is ephemeral, so by fire time there is no origin to
    resolve and the scheduler would fall back to guessing a home channel.
    Resolve at create time instead, using the creating run's own concrete
    delivery target — the ``HERMES_CRON_AUTO_DELIVER_*`` contextvars that
    ``run_job`` publishes per run (already per-job-safe under the parallel
    pool). Rules:

    * Not a cron-context session → returned unchanged (chat/CLI creates keep
      today's fire-time ``origin`` semantics, byte-identical).
    * ``origin`` element (or an omitted value, which the scheduler treats as
      origin) → replaced with ``platform:chat_id[:thread_id]`` from the
      creating run's target; ``local`` when the creating run has no concrete
      target (e.g. its own deliver is ``local``).
    * Every other element (``local``, ``all``, explicit ``platform:...``)
      passes through verbatim, including inside comma lists.
    """
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
        if thread_id:
            return f"{platform}:{chat_id}:{thread_id}"
        return f"{platform}:{chat_id}"

    if deliver is None:
        return _creator_target()
    parts = [p.strip() for p in str(deliver).split(",") if p.strip()]
    resolved = [_creator_target() if p.lower() == "origin" else p for p in parts]
    # De-dup while preserving order: 'origin,local' with a local-target
    # creator would otherwise store 'local,local'.
    seen: set = set()
    unique = [p for p in resolved if not (p in seen or seen.add(p))]
    return ",".join(unique) if unique else None


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
        port = int(port_raw) if port_raw else 8642
    except ValueError:
        port = 8642
    try:
        from hermes_cli.config import cfg_get, load_config_readonly
        host = str(cfg_get(load_config_readonly(), "platforms", "api_server", "extra", "host", default="") or "").strip()
    except Exception:
        host = ""
    host = host or os.getenv("API_SERVER_HOST", "").strip()
    if not host or host in ("0.0.0.0", "::", "*"):
        host = "127.0.0.1"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"  # bare IPv6 literal
    return f"http://{host}:{port}"


def _forward_relay_fronted_run(job: Dict[str, Any], extra_prompt: Optional[str] = None) -> Optional[str]:
    """Forward a manual run to the gateway when it targets a relay-fronted platform: such delivery
    has no standalone sender — the gateway's live relay adapter is the only path, reached via
    ``POST /api/jobs/{id}/run`` (marks the job due; ``extra_prompt`` rides in the body). Returns a
    JSON result string when forwarding engages, else None (normal in-process run)."""
    if not _relay_fronted_delivery_platforms(job):
        return None
    from agent.secret_scope import get_secret
    key = get_secret("API_SERVER_KEY", "") or ""
    try:
        import httpx
        resp = httpx.post(
            f"{_api_server_base_url()}/api/jobs/{job['id']}/run", headers={"Authorization": f"Bearer {key}"},
            json=({"prompt": extra_prompt} if extra_prompt else {}), timeout=10.0)
    except Exception:
        resp = None
    if resp is not None and resp.status_code < 300:
        return _dumps({
            "success": True,
            "forwarded_to_gateway": True,
            "note": (
                "This job targets a relay-fronted platform; it was dispatched "
                "to the running gateway, whose live relay adapter owns that "
                "delivery."),
        })
    return _dumps({
        "success": False,
        "error": (
            "This job targets a relay-fronted platform, which has no "
            "standalone sender. Start the gateway — its ticker will "
            "deliver the job on schedule via the live relay adapter."),
    })


def _manual_run_delivery_note(deliver: str, refreshed: Dict[str, Any]) -> str:
    """Parenthetical delivery note for a manual run's summary; follows the refreshed record's
    ``last_delivery_error`` so the summary never claims success over a failed delivery.

    Follows the refreshed job record (#83993): ``run_one_job`` writes ``last_delivery_error`` via
    ``mark_job_run`` when the post-run delivery (telegram/discord/…) failed, and the summary must not claim
    success over that record — the calling agent relays this line to the user. Local jobs never deliver; an
    empty/missing error keeps the legacy wording byte-for-byte.
    """
    # Falsy deliver ("", stored JSON null) is normalized to "local" at fire time -> saved
    # locally. Whitespace-only values fall through so the fire-time "no target" error surfaces.
    if not deliver or deliver == "local":
        return " (output saved locally only)"
    err = str(refreshed.get("last_delivery_error") or "").strip()
    if not err:
        if refreshed.get("last_delivery_queued"):
            return " (output queued for Bot Chat; completion unverified, do not resend)"
        return " (output was delivered there by the job itself)"
    return f" (⚠ delivery FAILED: {err[:200]})"


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
        "last_delivery_unverified": job.get("last_delivery_unverified"),
        "last_fire_error": job.get("last_fire_error"),
        "enabled": job.get("enabled", True),
        # Derive from enabled so half-paused records never render as paused.
        "state": effective_job_state(job),
        "paused_at": job.get("paused_at"),
        "paused_reason": job.get("paused_reason"),
    }
    if job.get("script"):
        result["script"] = job["script"]
    if job.get("reasoning_effort"):
        result["reasoning_effort"] = job["reasoning_effort"]
    if job.get("monitor_script"):
        result["monitor_script"] = job["monitor_script"]
    if job.get("monitor_url"):
        result["monitor_url"] = job["monitor_url"]
    if job.get("monitor_state"):
        result["monitor_state"] = job["monitor_state"]
    if job.get("no_agent"):
        result["no_agent"] = True
    if job.get("enabled_toolsets"):
        result["enabled_toolsets"] = job["enabled_toolsets"]
    if job.get("workdir"):
        result["workdir"] = job["workdir"]
    stored_refs = job.get("context_from") or []
    if isinstance(stored_refs, str):
        stored_refs = [stored_refs]
    if any(str(r).strip().lower() == "self" or r == job.get("id") for r in stored_refs):
        result["continuity"] = True
    external_refs = [
        r for r in stored_refs
        if str(r).strip().lower() != "self" and r != job.get("id")
    ]
    if external_refs:
        result["context_from"] = external_refs
    if isinstance(job.get("attach_to_session"), bool):
        result["attach_to_session"] = job["attach_to_session"]
    return result


def _relay_fronted_delivery_platforms(job: Dict[str, Any]) -> set:
    """Delivery-platform names for this job that the relay connector fronts."""
    try:
        claimed_job = claim_job_for_fire(job_id, manual=True, return_job=True)
        if isinstance(claimed_job, dict):
            return claimed_job, None
        refreshed = get_job(job_id)
        if refreshed is None:
            reason = "Job no longer exists; nothing to run."
        elif not is_job_runnable(refreshed):
            reason = "Job is paused/disabled; resume it before running."
        else:
            reason = "Job is already being fired by the scheduler; not run again."
        return None, {"claimed": False, "success": False, "error": reason}
    except Exception as e:
        logger.error("Failed to claim cron job %s for %s: %s", job_id, log_label, e)
        with contextlib.suppress(Exception):
            mark_job_run(job_id, False, str(e))
        return None, {"claimed": True, "success": False, "error": str(e)}


def _execute_job_now(job: Dict[str, Any], extra_prompt: Optional[str] = None) -> Dict[str, Any]:
    """Run a job now, outside the scheduler tick: claim via ``claim_job_for_fire`` (the ticker's
    CAS, so a concurrent tick cannot double-fire and next_run_at advances), then fire through
    the shared ``run_one_job`` body. Returns {"claimed", "success", "error"}."""
    claimed_job, err = _claim_for_manual_run(job["id"], "immediate run")
    return err if err is not None else _run_claimed_job(claimed_job, extra_prompt=extra_prompt)


@contextlib.contextmanager
def _run_heartbeat(job_name: str):
    """Heartbeat into the caller's activity tracker while a manual run executes (minutes,
    synchronously on the caller's thread — without tool activity the gateway inactivity
    watchdog would kill the parent turn). Best-effort: no callback -> no thread."""
    stop = threading.Event()
    thread = None
    try:
        # run_one_job records last_run_at/last_status via mark_job_run (which also clears the fire claim)
        # and returns True iff it processed the job. ``job`` here is the exact claimed snapshot
        # (owner-bearing), so the shared body fences every terminal write by that owner. A manual `run`
        # executes the job synchronously on the caller's thread, and a cron job is itself a full agent run
        # that routinely takes minutes. The calling turn emits no tool activity for that entire window, so
        # the gateway inactivity watchdog concludes the agent is hung and kills the parent turn (#76502).
        # Fire a heartbeat into the caller's activity tracker (the same signal tool progress uses) while the
        # job runs, so the watchdog sees a working tool instead of a silent one — mirrors the delegate_task
        # heartbeat pattern. Best-effort: if no activity callback is registered (direct Python callers,
        # tests), behavior is unchanged.
        from tools.environments.base import get_activity_callback
        # Capture on THIS thread: the callback is thread-local (installed by the tool
        # executor), so a freshly spawned thread cannot read it.
        activity_cb = get_activity_callback()
    except Exception:
        activity_cb = None

    def _heartbeat_loop() -> None:
        started = time.monotonic()
        while not stop.wait(_CRON_RUN_HEARTBEAT_INTERVAL):
            elapsed = time.monotonic() - started
            if elapsed > _CRON_RUN_HEARTBEAT_CEILING:
                # A run this long with an unlimited child watchdog is likely wedged.
                logger.warning(
                    "cronjob run heartbeat ceiling reached for job "
                    "'%s' (%.0fs) — stopping heartbeat; gateway watchdog regains authority",
                    job_name, elapsed)
                return
            try:
                activity_cb(f"cronjob: running job '{job_name}' ({int(elapsed)}s elapsed)")
            except Exception:
                continue  # one transient callback error must not drop protection

    if activity_cb is not None:
        thread = threading.Thread(target=_heartbeat_loop, daemon=True, name="cronjob-run-heartbeat")
        thread.start()
    try:
        yield
    finally:
        stop.set()
        if thread is not None:
            thread.join(timeout=_CRON_RUN_HEARTBEAT_INTERVAL + 1)


def _run_claimed_job(job: Dict[str, Any], extra_prompt: Optional[str] = None) -> Dict[str, Any]:
    """Fire an already-claimed job through the shared ``run_one_job`` body (split from
    ``_execute_job_now`` so the background path can claim synchronously and hand the run
    to a worker). Returns {"claimed": True, "success": bool, "error": ...}."""
    job_id = job["id"]
    _registered = False
    fire_owner = None
    try:
        from cron.scheduler import release_running_job, run_one_job, try_register_running_job

        # In-flight dedupe: the fire claim's TTL is routinely outlived by real jobs, so
        # register in the scheduler's shared running set (same guard the ticker uses;
        # also visible to the gateway shutdown drain).
        # In-flight dedupe (idea from #53395 by @izumi0uu): the fire claim's TTL (300s) is routinely
        # outlived by real jobs, so it alone cannot stop a manual run from double-firing a job the ticker
        # (or another manual run) is still executing.
        if not try_register_running_job(job_id):
            return {"claimed": True, "success": False, "error": _ALREADY_RUNNING_ERROR}
        _registered = True

        claim = job.get("fire_claim")
        fire_owner = str(claim.get("by") or "") if isinstance(claim, dict) else None

        # Inside the gateway process deliver on the loop that owns clients such as
        # Matrix/aiohttp (a standalone asyncio.run() loop breaks them).
        runner_ref = getattr(sys.modules.get("gateway.run"), "_gateway_runner_ref", None)
        # Manual runs invoked from a gateway agent execute outside the scheduler ticker, but they still
        # share the process with the live platform adapters. Calling those clients from run_one_job's
        # standalone asyncio.run() loop raises errors like "Timeout context manager should be used inside a
        # task" and can break encrypted Matrix delivery (#61495 — salvaged from #63586 by @Fly-onlyone).
        runner = runner_ref() if callable(runner_ref) else None
        adapters = getattr(runner, "adapters", None) if runner is not None else None
        gateway_loop = getattr(runner, "_gateway_loop", None) if runner is not None else None
        try:
            # run_one_job records last_run_at/last_status via mark_job_run; `job` is the
            # owner-bearing claimed snapshot, so terminal writes stay fenced by that owner.
            with _run_heartbeat(str(job.get("name") or job_id)):
                processed = run_one_job(job, adapters=adapters, loop=gateway_loop, extra_prompt=extra_prompt)
        finally:
            _registered = False
            release_running_job(job_id)
        refreshed = get_job(job_id) or {}
        execution = None
        execution_id = job.get("execution_id")
        if execution_id:
            from cron.executions import get_execution

            execution = get_execution(str(execution_id))
        last_status = refreshed.get("last_status")
        # "delivery_failed" (#83993): the agent run itself succeeded but the
        # output never reached the user. That is NOT a success for the caller
        # — the calling agent relays this result — so report it as failed
        # and surface the delivery error, which lives in last_delivery_error
        # (last_error is None for these runs, and a bare success=False with
        # error=None reads as an unexplained failure).
        ok = last_status == "ok"
        run_error = refreshed.get("last_error")
        if last_status == "delivery_failed" and not run_error:
            run_error = refreshed.get("last_delivery_error")
        if execution is not None and execution.get("status") != "completed":
            ok = False
            run_error = (
                execution.get("error")
                or f"execution ended in {execution.get('status') or 'unknown'} state"
            )
        return {
            "claimed": True,
            "success": bool(processed and ok),
            "error": run_error,
        }

            execution = get_execution(str(execution_id))
        last_status = refreshed.get("last_status")
        # "delivery_failed": the run succeeded but output never reached the user — not a
        # success for the caller; surface last_delivery_error.
        run_error = refreshed.get("last_error")
        if last_status == "delivery_failed" and not run_error:
            run_error = refreshed.get("last_delivery_error")
        # That is NOT a success for the caller — the calling agent relays this result — so report it as
        # failed and surface the delivery error, which lives in last_delivery_error (last_error is None for
        # these runs, and a bare success=False with error=None reads as an unexplained failure). See #83993.
        ok = last_status in {"ok", "delivery_queued"}
        if execution is not None and execution.get("status") != "completed":
            ok = False
            run_error = execution.get("error") or f"execution ended in {execution.get('status') or 'unknown'} state"
        return {"claimed": True, "success": bool(processed and ok), "error": run_error}
    except Exception as e:
        logger.error("Failed to execute cron job %s immediately: %s", job_id, e)
        if _registered:
            # Raised before the run's own release (e.g. heartbeat setup): don't leave the
            # job marked in-flight. Only release registrations WE took — a bare discard
            # could erase a ticker-owned entry.
            with contextlib.suppress(Exception):
                release_running_job(job_id)
        with contextlib.suppress(Exception):
            mark_job_run(job_id, False, str(e), expected_fire_owner=fire_owner)
        return {"claimed": True, "success": False, "error": str(e)}


def execute_job_for_event(
    job_ref: str, extra_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """Fire an existing cron job in response to an external event.

    Public entry point for event-driven triggers (the webhook adapter's
    ``cron_job`` routes). Resolves ``job_ref`` (ID or name) and
    fires it through the exact same claimed-run body a manual
    ``cronjob(action='run')`` uses, so at-most-once claiming, in-flight
    dedupe, delivery, and ``[SILENT]`` handling stay identical across the
    scheduler / manual / event paths.

    ``extra_prompt`` is injected as transient per-run context (the job's
    stored prompt is never mutated), exactly like ``action='run'`` with a
    ``prompt`` argument.

    Returns the ``_execute_job_now`` result shape:
    ``{"claimed": bool, "success": bool, "error": str|None}``.
    """
    try:
        job = resolve_job_ref(job_ref)
    except AmbiguousJobReference as e:
        return {"claimed": False, "success": False, "error": str(e)}
    if job is None:
        return {
            "claimed": False,
            "success": False,
            "error": f"Cron job '{job_ref}' not found.",
        }
    return _execute_job_now(job, extra_prompt=extra_prompt)


def _latest_job_output_excerpt(job_id: str, max_chars: int = 2000) -> Optional[str]:
    """Excerpt of the job's most recent saved output file for the background completion
    block (parent sees what the job produced). Never raises."""
    try:
        from cron.jobs import get_cron_output_dir
        files = sorted((get_cron_output_dir() / job_id).glob("*.md"))
        text = files[-1].read_text(encoding="utf-8", errors="replace").strip() if files else ""
        if not text:
            return None
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n… (truncated; full output: {files[-1]})"
        return text
    except Exception:
        return None


def _reap_stale_executions(job_name: str) -> None:
    """Reap execution rows left 'claimed'/'running' by a provably-dead owner (e.g. a prior
    one-shot `hermes cron run` that died mid-run). The ticker does this at startup; one-shot
    invocations have no such moment, so a stale claim would block every later manual run.
    Best-effort self-heal: must not block dispatch."""
    try:
        # Reap any execution row this job (or any job) left stranded 'claimed'/ 'running' by a dead owner
        # process -- e.g. a PRIOR one-shot `hermes cron run` invocation whose dispatched runner died with
        # the exiting process before writing a terminal status (issue #86721). Safe and cheap: only
        # provably-dead owners (PID gone, or PID reused by a different process per its start time) are
        # reaped; a genuinely live owner's row is left untouched.
        from cron.executions import recover_interrupted_executions
        _reclaimed = recover_interrupted_executions()
        if _reclaimed:
            logger.warning(
                "Reclaimed %d stale cron execution(s) from dead owner(s) before dispatching job '%s'",
                _reclaimed, job_name)
    except Exception as _reap_exc:
        logger.debug("Stale execution reclaim failed: %s", _reap_exc)


def _background_session_key(session_id: Optional[str]) -> str:
    """Routing key for a detached completion, captured on THIS thread (contextvars don't
    cross the pool). Empty string = no durable consumer."""
    try:
        from tools.approval_context import get_current_session_key
        session_key = get_current_session_key(default="")
    except Exception:
        session_key = ""
    # CLI path: the approval contextvar is only bound during gateway/TUI turns; the CLI
    # drain filters completions by the durable session id, and an empty key would fail
    # closed (completion never claimable).
    return session_key or (str(session_id) if session_id else "")


def _manual_run_completion(
    res: Dict[str, Any], job_id: str, job_name: str, deliver: str, started_at: float) -> Dict[str, Any]:
    """Async-delegation completion block for a finished background manual run."""
    duration = round(time.time() - started_at, 2)
    refreshed = get_job(job_id) or {}
    lines = [
        f"Cron job '{job_name}' ({job_id}) finished its manual run.",
        f"Result: {'ok' if res.get('success') else 'FAILED'}"
        + (f" — {res.get('error')}" if res.get("error") else ""),
        f"Delivery target: {deliver}" + _manual_run_delivery_note(deliver, refreshed),
    ]
    if refreshed.get("next_run_at"):
        lines.append(f"Next scheduled run: {refreshed['next_run_at']}")
    excerpt = _latest_job_output_excerpt(job_id)
    if excerpt:
        lines += ["--- JOB OUTPUT ---", excerpt]
    return {
        "status": "completed" if res.get("success") else "error", "summary": "\n".join(lines),
        "error": res.get("error"), "api_calls": 0, "duration_seconds": duration,
    }


def _try_dispatch_background_run(
    job: Dict[str, Any], session_id: Optional[str] = None, extra_prompt: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Claim ``job`` now (SYNCHRONOUSLY, so unrunnable jobs report immediately), then fire it
    on the async-delegation executor like ``delegate_task``'s background mode: the tool returns
    a handle and a ``type="async_delegation"`` completion re-enters as a fresh turn (role
    alternation legal, prompt cache intact) instead of blocking the parent turn for minutes.
    Returns None when background delivery is unavailable (caller runs sync); ``{"claimed":
    False}`` on a lost claim; ``{"claimed": True, "dispatched": True, "delegation_id"}``; or
    ``{"claimed": True, "dispatched": False, ...}`` when the pool was full and it ran inline."""
    # Finite sessions cannot route a detached result back after the turn ends (delegate_task's gate).
    try:
        from gateway.session_context import async_delivery_supported
        if not async_delivery_supported():
            return None
    except Exception:
        pass

    job_id = job["id"]
    job_name = str(job.get("name") or job_id)
    _reap_stale_executions(job_name)

    # Routing capture BEFORE the claim: no routable session = no durable consumer for a detached
    # completion, so don't claim-and-dispatch (direct callers like `hermes cron run` exit right after).
    session_key = _background_session_key(session_id)
    # CLI path: the approval contextvar is only bound during gateway/TUI turns. The CLI drain filters
    # completions by the durable agent session id (#64240), so stamp it as the key — an empty key would fail
    # closed and the completion could never be claimed.
    if not session_key:
        return None

    # Early dedupe so a mid-run job reports in THIS response, not as a delayed error completion
    # (authoritative check: try_register_running_job).
    try:
        from cron.scheduler import get_running_job_ids
        if job_id in get_running_job_ids():
            return {"claimed": False, "success": False, "error": _ALREADY_RUNNING_ERROR}
    except Exception:
        pass

    claimed_job, err = _claim_for_manual_run(job_id, "background run")
    if err is not None:
        if err["claimed"]:
            err["dispatched"] = False
        return err

    origin_ui_session_id = ""
    try:
        from gateway.session_context import get_session_env
        origin_ui_session_id = get_session_env("HERMES_UI_SESSION_ID", "") or ""
    except Exception:
        pass

    try:
        from tools.async_delegation import _current_origin_session_id, dispatch_async_delegation
        origin_session_id = _current_origin_session_id()
    except Exception as e:
        logger.warning(
            "cronjob run: async delegation registry unavailable (%s); running job '%s' inline.", e, job_name)
        result = _run_claimed_job(claimed_job, extra_prompt=extra_prompt)
        result["dispatched"] = False
        return result

    try:
        from tools.delegate_tool import _get_max_async_children
        max_async = _get_max_async_children()
    except Exception:
        max_async = 3

    started_at = time.time()
    # Canonicalize with the scheduler's own normalizer so the summary states
    # the same target fire time will use: falsy ("", stored JSON null) reads
    # "local", legacy list-form deliver flattens to its comma string. Read
    # from the claimed snapshot — the owner-bearing record the run actually
    # executes — not the pre-claim `job` the tool loaded.
    from cron.scheduler import _normalize_deliver_value

    deliver = _normalize_deliver_value(claimed_job.get("deliver", "local"))

    def _runner() -> Dict[str, Any]:
        res = _run_claimed_job(claimed_job, extra_prompt=extra_prompt)
        duration = round(time.time() - started_at, 2)
        refreshed = get_job(job_id) or {}
        lines = [
            f"Cron job '{job_name}' ({job_id}) finished its manual run.",
            f"Result: {'ok' if res.get('success') else 'FAILED'}"
            + (f" — {res.get('error')}" if res.get("error") else ""),
            f"Delivery target: {deliver}"
            + _manual_run_delivery_note(deliver, refreshed),
        ]
        if refreshed.get("next_run_at"):
            lines.append(f"Next scheduled run: {refreshed['next_run_at']}")
        excerpt = _latest_job_output_excerpt(job_id)
        if excerpt:
            lines.append("--- JOB OUTPUT ---")
            lines.append(excerpt)
        return {
            "status": "completed" if res.get("success") else "error",
            "summary": "\n".join(lines),
            "error": res.get("error"),
            "api_calls": 0,
            "duration_seconds": duration,
        }

    dispatch = dispatch_async_delegation(
        goal=f"Manual run of cron job '{job_name}' ({job_id})",
        context=("Triggered via cronjob(action='run'). The job executed in its own "
                 "fresh cron session; this block reports its outcome."),
        toolsets=None, role="cron_run", model=job.get("model"), session_key=session_key,
        parent_session_id=str(session_id) if session_id else None, runner=_runner,
        origin_ui_session_id=origin_ui_session_id, origin_session_id=origin_session_id,
        max_async_children=max_async)
    if dispatch.get("status") == "dispatched":
        return {"claimed": True, "dispatched": True, "delegation_id": dispatch.get("delegation_id")}

    # Pool at capacity (or submit failure): the claim is already taken and must not be stranded.
    logger.info(
        "cronjob run: background pool unavailable (%s); running job '%s' inline.",
        dispatch.get("error", "rejected"), job_name)
    result = _run_claimed_job(job, extra_prompt=extra_prompt)
    result["dispatched"] = False
    return result


# ---------------------------------------------------------------------------
# Tool actions. Each takes the cronjob() argument dict `a` (and the resolved
# job record for job-bound actions) and returns the JSON result string.
# ---------------------------------------------------------------------------

def _with_guidance(result: Dict[str, Any], job: Dict[str, Any], deliver: Optional[str]) -> Dict[str, Any]:
    """Attach mode/delivery guidance (create and update echo the same notes)."""
    _notes = _mode_guidance_notes(job, deliver)
    if _notes:
        result["guidance"] = _notes
    return result


def _action_create(a: Dict[str, Any]) -> str:
    prompt, script = a["prompt"], a["script"]
    deliver = _normalize_deliver_param(a["deliver"])
    if not a["schedule"]:
        return tool_error("schedule is required for create", success=False)
    canonical_skills = _canonical_skills(a["skill"], a["skills"])
    _no_agent = bool(a["no_agent"])
    # no_agent=True -> the script IS the job (prompt/skills optional); else prompt or skills.
    if _no_agent:
        if not script:
            return tool_error(
                "create with no_agent=True requires a script — "
                "the script is the job. In no_agent mode the LLM is "
                "skipped entirely: prompt and skills are ignored, "
                "non-empty stdout is delivered verbatim, empty stdout "
                "sends nothing (watchdog pattern), and a non-zero exit or timeout sends an error alert.",
                success=False)
    elif not prompt and not canonical_skills:
        return tool_error("create requires either prompt or at least one skill", success=False)
    error = (
        (prompt and _scan_cron_prompt(prompt))
        or (script and _validate_cron_script_path(script))
        or (a["monitor_script"] and _validate_cron_script_path(a["monitor_script"]))
        # A model-supplied base_url must not route a named provider's stored credential
        # to an attacker endpoint.
        or _validate_cron_base_url(a["provider"], a["base_url"])
        # bot-chat targets are machine-local: fail the CREATE, not the run.
        or _validate_bot_chat_deliver(deliver)
        # failure_deliver shares deliver's grammar and validators.
        or _validate_bot_chat_deliver(_normalize_deliver_param(a["failure_deliver"]))
        or (a["context_from"] and _validate_context_from_refs(
            [a["context_from"]] if isinstance(a["context_from"], str) else a["context_from"])))
    if error:
        return tool_error(error, success=False)

    context_from = a["context_from"]
    if a["continuity"] is not None:
        context_from = _apply_continuity(context_from, a["continuity"])

    from cron.scheduler import CronSchedulerRegistrationError, create_job_with_scheduler_registration
    try:
        job = create_job_with_scheduler_registration(
            prompt=prompt or "", schedule=a["schedule"], name=a["name"], repeat=a["repeat"],
            deliver=_resolve_cron_context_deliver(deliver), origin=_origin_from_env(), skills=canonical_skills,
            model=_normalize_optional_job_value(a["model"]), provider=_normalize_optional_job_value(a["provider"]),
            base_url=_normalize_optional_job_value(a["base_url"], strip_trailing_slash=True),
            script=_normalize_optional_job_value(script), context_from=context_from,
            enabled_toolsets=a["enabled_toolsets"] or None, workdir=_normalize_optional_job_value(a["workdir"]),
            no_agent=_no_agent, attach_to_session=a["attach_to_session"],
            monitor_script=_normalize_optional_job_value(a["monitor_script"]),
            monitor_url=_normalize_optional_job_value(a["monitor_url"]),
            # CLI-only lane: absent from CRONJOB_SCHEMA and the model dispatch (models don't pick models).
            reasoning_effort=a["reasoning_effort"],
            failure_deliver=_resolve_cron_context_deliver(_normalize_deliver_param(a["failure_deliver"])),
            **({"paused": a["paused"], "paused_reason": a["paused_reason"]}
               if a["paused"] is not False or a["paused_reason"] is not None else {}))
    except CronSchedulerRegistrationError as exc:
        _partial = exc.to_dict()
        return tool_error(_partial.pop("error"), success=False, **_partial)
    _create_message = " ".join(filter(None, (f"Cron job '{job['name']}' created.",
        "Created PAUSED — resume to schedule, or explicitly run now." if not job.get("enabled", True) else None,
        _local_delivery_notice(job, deliver))))
    # The builtin ticker lives in the gateway process: with no gateway running the job is stored
    # but never fires — tell the model (the CLI already warns).
    _result = {
        "success": True, "job_id": job["id"], "name": job["name"], "skill": job.get("skill"),
        "skills": job.get("skills", []), "schedule": job["schedule_display"], "repeat": _repeat_display(job),
        "deliver": job.get("deliver", "local"), "next_run_at": job["next_run_at"], "job": _format_job(job),
        "message": _create_message, **_gateway_liveness_notice(),
    }
    return _dumps(_with_guidance(_result, job, deliver))


def _action_list(a: Dict[str, Any]) -> str:
    jobs = [_format_job(job) for job in list_jobs(include_disabled=a["include_disabled"])]
    _result = {"success": True, "count": len(jobs), "jobs": jobs}
    # Same inert-job class as create; an empty list has nothing inert.
    if jobs:
        # Same silent-inert-job class as create (#87033): an agent inspecting existing jobs in a
        # gateway-less environment must learn they are not firing, not just see a clean list.
        _result.update(_gateway_liveness_notice(plural=True))
    return _dumps(_result)


def _action_remove(job: Dict[str, Any], a: Dict[str, Any]) -> str:
    job_id = job["id"]
    if not remove_job(job_id):
        return tool_error(f"Failed to remove job '{job_id}'", success=False)
    _notify_provider_jobs_changed_safe()
    return _dumps({
        "success": True,
        "message": f"Cron job '{job['name']}' removed.",
        "removed_job": {"id": job_id, "name": job["name"], "schedule": job.get("schedule_display")},
    })


def _job_state_result(updated: Dict[str, Any]) -> str:
    _notify_provider_jobs_changed_safe()
    return _dumps({"success": True, "job": _format_job(updated)})


def _refreshed_job_view(job_id: str) -> Dict[str, Any]:
    """Re-read so the response reflects the post-run last_run_at/last_status."""
    return _format_job(get_job(job_id) or {"id": job_id})


def _action_run(job: Dict[str, Any], a: Dict[str, Any]) -> str:
    job_id = job["id"]
    # `prompt` on run is transient per-fire context appended to the stored prompt, never
    # persisted; same strict scan as stored prompts.
    extra_prompt = a["prompt"] or None
    # See #57331, #57342, #57360.
    if extra_prompt:
        scan_error = _scan_cron_prompt(extra_prompt)
        if scan_error:
            return tool_error(scan_error, success=False)
    # A manual run must actually run even with no ticker active. Preferred: background
    # dispatch (handle now, outcome as a completion event); inline fallback otherwise.
    bg = _try_dispatch_background_run(job, session_id=a["session_id"], extra_prompt=extra_prompt)
    if bg is not None and bg.get("dispatched"):
        _notify_provider_jobs_changed_safe()
        result = _refreshed_job_view(job_id)
        result["executed"] = True
        result["execution_mode"] = "background"
        result["delegation_id"] = bg.get("delegation_id")
        return _dumps({
            "success": True,
            "job": result,
            "note": (
                "The job is running in the background. You and the "
                "user can keep working; its outcome re-enters the "
                "conversation as a new message when it finishes. "
                "Do not wait or poll — just continue."),
        })
    if bg is not None:
        exec_result = bg  # terminal result: claim lost or inline fallback
    else:
        # Relay-fronted manual run: no live adapter here — forward to the running gateway.
        forwarded = _forward_relay_fronted_run(job, extra_prompt=extra_prompt)
        if forwarded is not None:
            return forwarded
        exec_result = _execute_job_now(job, extra_prompt=extra_prompt)
    # A claimed direct run advances next_run_at and may race an external provider's
    # one-shot for the same occurrence; a lost consumed fire cannot re-arm itself, so
    # reconcile after the run has persisted its final state.
    claimed = exec_result.get("claimed", False)
    if claimed:
        _notify_provider_jobs_changed_safe()
    result = _refreshed_job_view(job_id)
    result["executed"] = claimed
    result["execution_success"] = exec_result.get("success", False)
    if not claimed:
        result["execution_skipped"] = exec_result.get("error") or (
            "Already being fired by the scheduler; not run again.")
    elif exec_result.get("error"):
        result["execution_error"] = exec_result["error"]
    return _dumps({"success": True, "job": result})


def _pick(updates: Dict[str, Any], job: Dict[str, Any], key: str) -> Any:
    """Effective value of ``key`` after this update: pending update wins over the stored job."""
    return updates[key] if key in updates else job.get(key)


def _update_core_fields(job: Dict[str, Any], a: Dict[str, Any], updates: Dict[str, Any]) -> Optional[str]:
    """prompt / name / deliver / skills / model pins; returns an error string or None."""
    prompt, deliver, skill, skills = a["prompt"], a["deliver"], a["skill"], a["skills"]
    if prompt is not None:
        scan_error = _scan_cron_prompt(prompt)
        if scan_error:
            return scan_error
        updates["prompt"] = prompt
    if a["name"] is not None and a["name"].strip():
        # Blank name is a no-op, not a clear: a model re-sending the whole schema with
        # type-default empties must not wipe untouched fields.
        updates["name"] = a["name"]
    if deliver is not None:
        bot_chat_error = _validate_bot_chat_deliver(_normalize_deliver_param(deliver))
        if bot_chat_error:
            return bot_chat_error
        updates["deliver"] = _resolve_cron_context_deliver(_normalize_deliver_param(deliver))
    if a["failure_deliver"] is not None:
        # '' clears the override (failures fall back to deliver); non-empty values share
        # deliver's validation AND its cron-context origin resolution (a job created from
        # inside a cron run must never store literal 'origin').
        _norm_fd = _normalize_deliver_param(a["failure_deliver"])
        if _norm_fd:
            bot_chat_error = _validate_bot_chat_deliver(_norm_fd)
            if bot_chat_error:
                return bot_chat_error
            _norm_fd = _resolve_cron_context_deliver(_norm_fd)
        updates["failure_deliver"] = _norm_fd
    if skills is not None or skill is not None:
        canonical_skills = _canonical_skills(skill, skills)
        updates["skills"] = canonical_skills
        updates["skill"] = canonical_skills[0] if canonical_skills else None
    if a["model"] is not None:
        updates["model"] = _normalize_optional_job_value(a["model"])
    if a["provider"] is not None:
        updates["provider"] = _normalize_optional_job_value(a["provider"])
    if a["base_url"] is not None:
        updates["base_url"] = _normalize_optional_job_value(a["base_url"], strip_trailing_slash=True)
    if a["reasoning_effort"] is not None:
        # CLI-only lane; update_job validates, empty string clears the pin.
        updates["reasoning_effort"] = a["reasoning_effort"]
    # Re-validate the EFFECTIVE provider/base_url on EVERY update: a job persisted before
    # this guard may hold an unsafe pair, and editing an unrelated field must not leave it
    # schedulable. Merging this update over the stored job lets an operator remediate.
    return _validate_cron_base_url(_pick(updates, job, "provider"), _pick(updates, job, "base_url"))


def _update_script_fields(job: Dict[str, Any], a: Dict[str, Any], updates: Dict[str, Any]) -> Optional[str]:
    """script / monitor_script / monitor_url (empty string clears); returns an error string or None."""
    monitor_script, monitor_url = a["monitor_script"], a["monitor_url"]
    for field, value in (("script", a["script"]), ("monitor_script", monitor_script)):
        if value is not None:
            if value:
                path_error = _validate_cron_script_path(value)
                if path_error:
                    return path_error
            updates[field] = _normalize_optional_job_value(value) if value else None
    if monitor_url is not None:
        updates["monitor_url"] = _normalize_optional_job_value(monitor_url) if monitor_url else None
    if (monitor_script is not None or monitor_url is not None) and (
        _pick(updates, job, "monitor_script") and _pick(updates, job, "monitor_url")):
        return("monitor_script and monitor_url are mutually exclusive — clear one before setting the other.")
    return None


def _update_context_from(job: Dict[str, Any], a: Dict[str, Any], updates: Dict[str, Any]) -> Optional[str]:
    """context_from / continuity: empty string / list clears; otherwise every ref must
    exist. Stored as a list (or None) to match create_job()."""
    context_from, continuity = a["context_from"], a["continuity"]
    if context_from is None and continuity is None:
        return None
    if context_from is None:
        context_from = list(job.get("context_from") or [])  # continuity-only update
    refs = _clean_str_list(context_from)
    if continuity is not None:
        refs = _apply_continuity(refs, continuity) or []
    if refs:
        ref_error = _validate_context_from_refs(refs)
        if ref_error:
            return ref_error
    updates["context_from"] = refs or None
    return None


def _update_run_fields(job: Dict[str, Any], a: Dict[str, Any], updates: Dict[str, Any]) -> Optional[str]:
    """enabled_toolsets / attach_to_session / workdir / no_agent / repeat / schedule."""
    if a["enabled_toolsets"] is not None:
        updates["enabled_toolsets"] = a["enabled_toolsets"] or None
    if a["attach_to_session"] is not None:
        updates["attach_to_session"] = bool(a["attach_to_session"])
    if a["workdir"] is not None:
        # Empty string clears; otherwise update_job() validates/normalizes.
        updates["workdir"] = _normalize_optional_job_value(a["workdir"]) or None
    if a["no_agent"] is not None:
        # Flipping to True needs a script on the job or in this same update.
        target_no_agent = bool(a["no_agent"])
        if target_no_agent and not _pick(updates, job, "script"):
            return (
                "Cannot set no_agent=True on a job without a script. "
                "Set `script` in the same update, or on the job first.")
        updates["no_agent"] = target_no_agent
    if a["repeat"] is not None:
        # Shared chokepoint coerces string forms ('forever'/'once'/'3') and 0/negative.
        from cron.jobs import normalize_repeat_value
        repeat_state = dict(job.get("repeat") or {})
        repeat_state["times"] = normalize_repeat_value(a["repeat"])
        updates["repeat"] = repeat_state
    if a["schedule"] is not None:
        parsed_schedule = parse_schedule(a["schedule"])
        updates["schedule"] = parsed_schedule
        updates["schedule_display"] = parsed_schedule.get("display", a["schedule"])
        if job.get("state") != "paused":
            updates["state"] = "scheduled"
            updates["enabled"] = True
    return None


# Validation order is behavior (first failing field wins): keep this sequence.
_UPDATE_STEPS = (_update_core_fields, _update_script_fields, _update_context_from, _update_run_fields)


def _action_update(job: Dict[str, Any], a: Dict[str, Any]) -> str:
    updates: Dict[str, Any] = {}
    for step in _UPDATE_STEPS:
        error = step(job, a, updates)
        if error:
            return tool_error(error, success=False)
    if not updates:
        return tool_error("No updates provided.", success=False)
    updated = update_job(job["id"], updates)
    _notify_provider_jobs_changed_safe()
    # An update can switch modes or delivery — echo the same guidance as create.
    return _dumps(_with_guidance(
        {"success": True, "job": _format_job(updated)}, updated, _normalize_deliver_param(a["deliver"])))


def _action_resnap(a: Dict[str, Any]) -> str:
    """Adopt the current global inference resolution without pinning (#44585).

    Bulk (``all=true``) refreshes every unpinned job; single-job resolves
    ``job_id`` and refreshes just that job. Refuses to guess scope.
    """
    if bool(a["all"]):
        updated = resnapshot_all_unpinned()
        _notify_provider_jobs_changed_safe()
        return _dumps({
            "success": True,
            "message": (
                f"Refreshed inference snapshots on {len(updated)} unpinned "
                "job(s) to the current global resolution. Jobs remain "
                "unpinned and will track future global changes."),
            "updated_jobs": [_format_job(j) for j in updated],
        })
    job_id = a["job_id"]
    if not job_id:
        return tool_error(
            "resnap requires either `job_id=<id>` (single job) or `all=true` "
            "(refresh every unpinned job). Refusing to guess scope.",
            success=False,
        )
    job, error = _resolve_job_or_error(job_id)
    if error is not None:
        return error
    assert job is not None  # error is None ⇔ job resolved
    updated = resnapshot_job(job["id"])
    if not updated:
        return tool_error(f"Failed to resnap job '{job_id}'", success=False)
    _notify_provider_jobs_changed_safe()
    return _dumps({
        "success": True,
        "message": (
            f"Cron job '{updated['name']}' refreshed to the current "
            "global inference resolution. It remains unpinned and will "
            "track future global changes."),
        "job": _format_job(updated),
    })


# Actions that need no job_id, and job-bound actions (job resolved first).
_JOBLESS_ACTIONS = {"create": _action_create, "list": _action_list, "resnap": _action_resnap}
_JOB_ACTIONS = {
    "remove": _action_remove, "update": _action_update,
    "run": _action_run, "run_now": _action_run, "trigger": _action_run,
    "pause": lambda job, a: _job_state_result(pause_job(job["id"], reason=a["reason"])),
    "resume": lambda job, a: _job_state_result(resume_job(job["id"])),
}


def _resolve_job_or_error(job_id: str):
    """``(job, None)`` or ``(None, json_error)`` for a job_id/name reference."""
    try:
        job = resolve_job_ref(job_id)
    except AmbiguousJobReference as exc:
        return None, _dumps({
            "success": False,
            "error": str(exc),
            "matches": [
                {"id": m["id"], "name": m.get("name"), "schedule": m.get("schedule_display"), "next_run_at": m.get("next_run_at")}
                for m in exc.matches
            ],
        })
    if not job:
        return None, _dumps(
            {"success": False, "error": f"Job with ID or name '{job_id}' not found. Use cronjob(action='list') to inspect jobs."},
        )
    return job, None


def _gateway_liveness_notice(plural: bool = False) -> dict:
    """Build the ``gateway_running``/``warning`` payload for tool results.

    Thin adapter over the shared CLI helper ``hermes_cli.cron._builtin_gateway_liveness``
    (#87033) so the CLI and this tool can never disagree about what "scheduler
    active" means. Returns ``{"gateway_running": False, "warning": ...}`` when
    the builtin ticker has no gateway process to run it, ``{"gateway_running":
    None}`` when the probe failed, and ``{"gateway_running": True}`` when the
    scheduler is active. ``plural`` rewords the warning for multi-job results
    (the ``list`` action).
    """
    try:
        from hermes_cli.cron import _builtin_gateway_liveness

        _gw = _builtin_gateway_liveness()
    except Exception:
        return {"gateway_running": None}
    subject = "these jobs are saved" if plural else "this job is saved"
    if _gw is False:
        return {
            "gateway_running": False,
            "warning": (
                f"The Hermes gateway is not running — {subject} "
                "but will NOT fire until the gateway is started "
                "(hermes gateway install / hermes gateway start). "
                "Tell the user the task is scheduled but not active yet."
            ),
        }
    if _gw is None:
        return {"gateway_running": None}
    return {"gateway_running": True}


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
    continuity: Optional[bool] = None,
    enabled_toolsets: Optional[List[str]] = None,
    workdir: Optional[str] = None,
    no_agent: Optional[bool] = None,
    attach_to_session: Optional[bool] = None,
    monitor_script: Optional[str] = None,
    monitor_url: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    failure_deliver: Optional[Union[str, List[str]]] = None,
    all: Optional[bool] = None,
    task_id: str = None,
    session_id: Optional[str] = None,
    paused: bool = False,
    paused_reason: Optional[str] = None) -> str:
    """Unified cron job management tool."""
    a = dict(locals())
    del a["task_id"]  # unused but kept for handler signature compatibility
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
                        "the script is the job. In no_agent mode the LLM is "
                        "skipped entirely: prompt and skills are ignored, "
                        "non-empty stdout is delivered verbatim, empty stdout "
                        "sends nothing (watchdog pattern), and a non-zero "
                        "exit or timeout sends an error alert.",
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

            # Validate monitor source (same containment rules as script).
            if monitor_script:
                monitor_error = _validate_cron_script_path(monitor_script)
                if monitor_error:
                    return tool_error(monitor_error, success=False)

            # Reject a model-supplied base_url that would route a named
            # provider's stored credential to an attacker endpoint (F8).
            base_url_error = _validate_cron_base_url(provider, base_url)
            if base_url_error:
                return tool_error(base_url_error, success=False)

            # bot-chat deliver targets are machine-local: named profiles must
            # exist here, and a bad name should fail the CREATE, not the run.
            bot_chat_error = _validate_bot_chat_deliver(_normalize_deliver_param(deliver))
            if bot_chat_error:
                return tool_error(bot_chat_error, success=False)
            # failure_deliver shares deliver's grammar and validators (NS-788).
            bot_chat_error = _validate_bot_chat_deliver(
                _normalize_deliver_param(failure_deliver)
            )
            if bot_chat_error:
                return tool_error(bot_chat_error, success=False)

            # Validate context_from references existing jobs
            if context_from:
                from cron.jobs import get_job as _get_job
                refs = [context_from] if isinstance(context_from, str) else context_from
                for ref_id in refs:
                    # "self" is resolved to the job's own id at run time —
                    # it can't be validated against the store (the job does
                    # not exist yet at create time).
                    if isinstance(ref_id, str) and ref_id.strip().lower() == "self":
                        continue
                    if not _get_job(ref_id):
                        return tool_error(
                            f"context_from job '{ref_id}' not found. "
                            "Use cronjob(action='list') to see available jobs.",
                            success=False,
                        )

            # continuity=True is sugar for context_from including "self":
            # the job wakes up with its own previous run's output injected.
            if continuity is not None:
                context_from = _apply_continuity(context_from, continuity)

            from cron.scheduler import (
                CronSchedulerRegistrationError,
                create_job_with_scheduler_registration,
            )

            try:
                job = create_job_with_scheduler_registration(
                    prompt=prompt or "",
                    schedule=schedule,
                    name=name,
                    repeat=repeat,
                    deliver=_resolve_cron_context_deliver(
                        _normalize_deliver_param(deliver)
                    ),
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
                    monitor_script=_normalize_optional_job_value(monitor_script),
                    monitor_url=_normalize_optional_job_value(monitor_url),
                    # reasoning_effort reaches here from the CLI
                    # (hermes cron create --reasoning-effort) ONLY — it is
                    # deliberately absent from CRONJOB_SCHEMA and the model
                    # dispatch below: models do not make model-config
                    # decisions (standing policy).
                    reasoning_effort=reasoning_effort,
                    failure_deliver=_resolve_cron_context_deliver(
                        _normalize_deliver_param(failure_deliver)
                    ),
                )
            except CronSchedulerRegistrationError as exc:
                _partial = exc.to_dict()
                return tool_error(_partial.pop("error"), success=False, **_partial)
            _create_message = f"Cron job '{job['name']}' created."
            _local_notice = _local_delivery_notice(job, _normalize_deliver_param(deliver))
            if _local_notice:
                _create_message = f"{_create_message} {_local_notice}"
            # Gateway liveness surfacing (#87033): the builtin scheduler's
            # ticker lives in the gateway process, so a job created with no
            # gateway running is stored but will never fire. Tell the model
            # here — the CLI already warns, but the agent path saw only a
            # clean success and confidently told the user it was scheduled.
            _result = {
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
                **_gateway_liveness_notice(),
            }
            # Mode-specific guidance rides in the create response (once, when
            # relevant) instead of in the schema (every API call). See
            # _mode_guidance_notes.
            _notes = _mode_guidance_notes(job, _normalize_deliver_param(deliver))
            if _notes:
                _result["guidance"] = _notes
            return json.dumps(_result, indent=2)

        if normalized == "list":
            jobs = [_format_job(job) for job in list_jobs(include_disabled=include_disabled)]
            _result = {"success": True, "count": len(jobs), "jobs": jobs}
            # Same silent-inert-job class as create (#87033): an agent
            # inspecting existing jobs in a gateway-less environment must
            # learn they are not firing, not just see a clean list. An empty
            # list has nothing inert — stay quiet (and skip the probe).
            if jobs:
                _result.update(_gateway_liveness_notice(plural=True))
            return json.dumps(_result, indent=2)

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
            # Per-run context (#57331, salvaged from #57342/@liuhao1024 and
            # #57360/@ghedeselmabot): `prompt` on the run action is transient
            # context appended to the stored prompt for THIS fire only, never
            # persisted. It goes through the same strict injection scan as
            # stored prompts before firing.
            extra_prompt = prompt or None
            if extra_prompt:
                scan_error = _scan_cron_prompt(extra_prompt)
                if scan_error:
                    return tool_error(scan_error, success=False)
            # Execute the job immediately rather than only scheduling it for the
            # next scheduler tick — a manual `run` should actually run, even when
            # no gateway/ticker is active (the #41037 case). The claim (taken
            # inside both paths below) advances next_run_at and blocks a
            # concurrent tick from double-firing.
            #
            # Preferred path: dispatch the run to the background like
            # delegate_task — the tool returns a handle immediately and the
            # job's outcome re-enters the conversation as a completion event.
            # A cron job is a full agent run (minutes to hours); executing it
            # inline made the parent turn uninterruptible and serialized
            # batches of manual runs (#80xxx — the "stuck Telegram session"
            # incident). Falls back to inline execution when the session
            # runtime can't receive detached completions.
            bg = _try_dispatch_background_run(
                job, session_id=session_id, extra_prompt=extra_prompt
            )
            if bg is not None and bg.get("dispatched"):
                _notify_provider_jobs_changed_safe()
                result = _format_job(get_job(job_id) or {"id": job_id})
                result["executed"] = True
                result["execution_mode"] = "background"
                result["delegation_id"] = bg.get("delegation_id")
                return json.dumps(
                    {
                        "success": True,
                        "job": result,
                        "note": (
                            "The job is running in the background. You and the "
                            "user can keep working; its outcome re-enters the "
                            "conversation as a new message when it finishes. "
                            "Do not wait or poll — just continue."
                        ),
                    },
                    indent=2,
                )
            # bg carries a terminal result (claim lost, or inline fallback
            # after pool rejection); None means background delivery is
            # unsupported here — run synchronously as before.
            if bg is not None:
                exec_result = bg
            else:
                # Relay-fronted manual run: a standalone process has no live
                # relay adapter and no standalone sender, so forward to the
                # running gateway (its live adapter owns that delivery).
                forwarded = _forward_relay_fronted_run(job, extra_prompt=extra_prompt)
                if forwarded is not None:
                    return forwarded
                exec_result = _execute_job_now(job, extra_prompt=extra_prompt)
            # A claimed direct run advances next_run_at and may race the
            # external one-shot for the same occurrence. If Chronos loses that
            # claim, its consumed fire cannot re-arm itself; reconcile from the
            # winning direct path after the run has persisted its final state.
            if exec_result.get("claimed", False):
                _notify_provider_jobs_changed_safe()
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
                scan_error = _scan_cron_prompt(prompt)
                if scan_error:
                    return tool_error(scan_error, success=False)
                updates["prompt"] = prompt
            if name is not None and name.strip():
                # Blank name is a no-op, not a clear. The `is not None` sentinel
                # treats every supplied field as an explicit edit, and a model
                # that re-sends the whole schema with type-default empties ("", [], 0)
                # then wipes fields it never meant to touch.
                updates["name"] = name
            if deliver is not None:
                bot_chat_error = _validate_bot_chat_deliver(_normalize_deliver_param(deliver))
                if bot_chat_error:
                    return tool_error(bot_chat_error, success=False)
                updates["deliver"] = _resolve_cron_context_deliver(
                    _normalize_deliver_param(deliver)
                )
            if failure_deliver is not None:
                # '' clears the override (job falls back to deliver on
                # failures); non-empty values share deliver's validation
                # AND its cron-context origin resolution (a job created
                # from inside a cron run must never store literal
                # 'origin' — same rule as deliver).
                _norm_fd = _normalize_deliver_param(failure_deliver)
                if _norm_fd:
                    bot_chat_error = _validate_bot_chat_deliver(_norm_fd)
                    if bot_chat_error:
                        return tool_error(bot_chat_error, success=False)
                    _norm_fd = _resolve_cron_context_deliver(_norm_fd)
                updates["failure_deliver"] = _norm_fd
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
            if reasoning_effort is not None:
                # CLI-only lane (see create above): update_job validates
                # against the canonical grammar; empty string clears the pin.
                updates["reasoning_effort"] = reasoning_effort
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
            if monitor_script is not None:
                # Pass empty string to clear an existing monitor_script
                if monitor_script:
                    monitor_error = _validate_cron_script_path(monitor_script)
                    if monitor_error:
                        return tool_error(monitor_error, success=False)
                updates["monitor_script"] = (
                    _normalize_optional_job_value(monitor_script) if monitor_script else None
                )
            if monitor_url is not None:
                # Pass empty string to clear an existing monitor_url
                updates["monitor_url"] = (
                    _normalize_optional_job_value(monitor_url) if monitor_url else None
                )
            if monitor_script is not None or monitor_url is not None:
                eff_mon_script = (
                    updates["monitor_script"] if "monitor_script" in updates else job.get("monitor_script")
                )
                eff_mon_url = (
                    updates["monitor_url"] if "monitor_url" in updates else job.get("monitor_url")
                )
                if eff_mon_script and eff_mon_url:
                    return tool_error(
                        "monitor_script and monitor_url are mutually exclusive — "
                        "clear one before setting the other.",
                        success=False,
                    )
            if context_from is not None or continuity is not None:
                # Empty string / empty list clears the field; otherwise validate
                # each referenced job exists before storing. Normalized to a list
                # (or None) to match the shape stored by create_job().
                if context_from is None:
                    # continuity-only update: start from the job's stored refs.
                    existing = job.get("context_from") or []
                    refs = [str(j).strip() for j in existing if str(j).strip()]
                elif isinstance(context_from, str):
                    refs = [context_from.strip()] if context_from.strip() else []
                else:
                    refs = [str(j).strip() for j in context_from if str(j).strip()]
                if continuity is not None:
                    refs = _apply_continuity(refs, continuity) or []
                if refs:
                    from cron.jobs import get_job as _get_job
                    for ref_id in refs:
                        # "self" resolves to the job's own id at run time.
                        if ref_id.lower() == "self":
                            continue
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
                # Coerce string forms ('forever'/'once'/'3') and 0/negative
                # via the shared chokepoint — a bare `repeat <= 0` here
                # raised TypeError for string repeats on the UPDATE path
                # (create was fixed first; same class).
                from cron.jobs import normalize_repeat_value
                normalized_repeat = normalize_repeat_value(repeat)
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
            _upd_result: Dict[str, Any] = {"success": True, "job": _format_job(updated)}
            # An update can switch a job into monitor / no_agent mode or
            # change its delivery — echo the same mode guidance as create.
            _upd_notes = _mode_guidance_notes(updated, _normalize_deliver_param(deliver))
            if _upd_notes:
                _upd_result["guidance"] = _upd_notes
            return json.dumps(_upd_result, indent=2)

        return tool_error(f"Unknown cron action '{action}'", success=False)

    except Exception as e:
        return tool_error(str(e), success=False)


def _script_description(home: str) -> str:
    return (f"Optional script run each tick; stdout is injected into the agent's prompt as context (with no_agent=True "
            f"the script IS the job). Relative paths resolve under {home}/scripts/; .sh/.bash via bash, else Python. "
            "On update, '' clears.")


def _cronjob_schema_overrides() -> dict:
    """Rebuild the ``script`` path hint from the ACTIVE profile at every get_definitions(): the
    static schema is built once per process, but the multiplexed gateway serves every profile from
    that process, so a path baked in at import would name the launch profile's home (#95685)."""
    params = copy.deepcopy(CRONJOB_SCHEMA["parameters"])
    params["properties"]["script"]["description"] = _script_description(display_hermes_home())
    return {"parameters": params}


CRONJOB_SCHEMA = {
    "name": "cronjob_manage",
    "description": """Manage scheduled cron jobs: action='create' schedules a job from a prompt and/or skills; 'list' inspects jobs; 'update'/'pause'/'resume'/'remove' manage one by job_id (always list first — never guess job IDs); 'run' fires a job immediately in the BACKGROUND (returns a handle at once, outcome re-enters the conversation when done — do not wait or poll; optional 'prompt' adds transient context for that fire only).

'resnap' adopts the CURRENT global inference resolution for an unpinned job (job_id) or all unpinned jobs (all=true) WITHOUT pinning it, so it keeps tracking future global changes — use after deliberately changing the default model.

Jobs run in a fresh session with no current-chat context, so prompts must be self-contained, and the agent's FINAL RESPONSE is what gets delivered — cron runs are autonomous and cannot ask questions. Prefer updating an existing job over creating near-duplicates.""",
    "parameters": {
        "type": "object",
        "properties": {
            "paused": {"type": "boolean", "description": "Create only: persist disabled atomically. Resume to schedule; explicit run remains available. Default false."},
            "paused_reason": {"type": "string", "description": "Create only: auditable reason; requires paused=true."},
            "action": {
                "type": "string",
                "description": "One of: create, list, update, pause, resume, remove, run, resnap. When action=create, the 'schedule' and 'prompt' fields are REQUIRED. When action=resnap, pass either job_id (single job) or all=true (every unpinned job)."
            },
            "job_id": {
                "type": "string",
                "description": "Required for update/pause/resume/remove/run. For resnap: the job to adopt the current global inference resolution (omit if all=true)."
            },
            "all": {
                "type": "boolean",
                "description": "Only for action='resnap'. all=true refreshes the inference snapshot of EVERY unpinned agent job to the current global resolution (bulk 'make everything follow my new default'). Must be explicitly set to true — never implied. Omit (or false) to resnap a single job via job_id."
            },
            "prompt": {
                "type": "string",
                "description": "For create: the full self-contained prompt (paired with any skills as the task instruction). For run: optional transient context for that single fire (never persisted)."
            },
            "schedule": {
                "type": "string",
                "type": "string",
                "description": "REQUIRED for create. Schedule forms: (1) recurring interval — '30m', 'every 2h', 'every hour' (EVERY 30 minutes / 2 hours / hour, forever by default); (2) explicit one-shot by duration — 'in 30m', 'in 2h' (fires ONCE that far from now; use this for 'remind me in N minutes' — do NOT hand-compute an absolute timestamp); (3) natural day/time — 'every monday 9am', 'weekdays at 9am', 'every day at 9am' (recurring weekly/daily); (4) cron syntax — '0 9 * * *' (daily 9am); (5) absolute one-shot — ISO timestamp '2026-06-01T09:00:00'."
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
                "description": "Where the job's output is POSTED as a one-way message (the job itself always runs in a fresh session with no chat context). Omit to address the chat/topic this job was created from. Otherwise: 'local' (save only, no delivery), 'all' (every connected home channel, resolved at fire time), 'bot-chat' or 'bot-chat:<profile>' (inject into a Bot Chat as a real message), or platform:chat_id:thread_id (e.g. 'telegram:-1001234567890:17585'). Comma-combine like 'origin,all'."
            },
            "failure_deliver": {
                "type": "string",
                "description": "Optional override target for FAILURE notices only (same grammar as deliver). When set, engine failure/interruption notices go here instead of the deliver target; 'local' suppresses them entirely (state still recorded in cron list/run history). Use for jobs delivering into shared channels where failure noise is unwanted. Omit = failures follow deliver (default). On update, '' clears."
            },
            "skills": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional ordered skill names loaded before the cron prompt. On update, [] clears."
            },
            "script": {
                "type": "string",
                "description": _script_description("the profile HERMES_HOME")
            },
            "monitor": {
                "type": "string",
                "description": "Optional change-detector that gates the agent: an http(s) URL (fetched each tick) or a script path (same rules as `script`, run each tick) — cheap, no LLM. Output identical to the previous tick skips the agent run entirely; changed output wakes the agent with a diff injected into the prompt. First tick always runs (baseline). Output must be deterministic (no timestamps) or every tick looks changed. Incompatible with no_agent. On update, '' clears."
            },
            "no_agent": {
                "type": "boolean",
                "default": False,
                "description": "True = no LLM: the scheduler runs `script` (required) on schedule and delivers its stdout verbatim; empty stdout sends nothing (watchdog pattern). Use for script-only pings with fixed output; keep False for anything needing reasoning."
            },
            "context_from": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional job ID(s) whose most recent completed output is injected as context each run — chains jobs (A collects, B processes). For a job's OWN previous output prefer `continuity`. On update, [] clears."
            },
            "continuity": {
                "type": "boolean",
                "description": "True = each run sees the job's own previous output, so it can dedupe and continue where it left off (scouts, monitors, incremental digests). Default false. On update, false turns it off."
            },
            "enabled_toolsets": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional toolset names to restrict the job's agent to (e.g. [\"web\", \"terminal\"]) — cuts token overhead. Infer from the prompt. Omit for all default tools. On update, [] clears."
            },
            "workdir": {
                "type": "string",
                "description": "Optional absolute existing path to run the job from: injects that directory's AGENTS.md/context files and anchors terminal/file tools there. On update, '' clears."
            },
            "attach_to_session": {
                "type": "boolean",
                "description": "True = the job's delivery is CONTINUABLE — the user can reply and the agent has the brief in context (threads on thread-capable platforms, mirrored into the DM elsewhere). Use for conversational recurring jobs (briefings); leave unset for fire-and-forget alerts. Scope: the job's own conversation only — the origin chat, the home-channel fallback when deliver='origin' captured no origin (script-created jobs), a user-written bare platform target (deliver='slack' — that platform's home channel), or the job's single explicit platform:chat target (this flag is the only way to attach an explicit target). Broadcast targets are never attached; no effect when deliver='local'."
            },
        },
        "required": ["action"]
    }
}


def check_cronjob_requirements() -> bool:
    """Available in interactive CLI mode, gateway/messaging platforms, and cron runs (the
    scheduler is internal; no crontab needed). Flags must be explicitly truthy via
    ``env_var_enabled``. An external cron worker has the presence vars stripped from its env, so
    the cron session marker keeps ``cron.allow_agent_scheduling`` meaningful there."""
    from gateway.session_context import get_session_env
    from utils import env_var_enabled, is_truthy_value
    return (
        env_var_enabled("HERMES_INTERACTIVE")
        or env_var_enabled("HERMES_GATEWAY_SESSION")
        or env_var_enabled("HERMES_EXEC_ASK")
        or is_truthy_value(get_session_env("HERMES_CRON_SESSION", ""))
    )


# Agent-facing arguments forwarded verbatim to cronjob(). model / provider / base_url are
# intentionally NOT here: per-job inference pins are user-owned (dashboard, `hermes cron
# create/edit --model`, hand-edited jobs) — the agent must not point unattended spend at a
# different model. Programmatic callers of cronjob() itself retain the parameters.
_HANDLER_FORWARDED_ARGS = (
    "job_id", "prompt", "schedule", "name", "repeat", "deliver", "failure_deliver", "skill", "skills", "reason",
    "script", "context_from", "continuity", "enabled_toolsets", "workdir", "no_agent", "attach_to_session",
    "paused_reason", "all")


def _cronjob_handler(args, **kw):
    """Model-tool dispatch for ``cronjob``.

    Resolves the one model-facing ``monitor`` field into the stored
    ``monitor_script``/``monitor_url`` pair (legacy field names still accepted
    as aliases so older transcripts/replays keep working).
    """
    _mon_script, _mon_url = _split_monitor_arg(
        args.get("monitor"), args.get("monitor_script"), args.get("monitor_url")
    )
    return cronjob(
        action=args.get("action", ""),
        job_id=args.get("job_id"),
        prompt=args.get("prompt"),
        schedule=args.get("schedule"),
        name=args.get("name"),
        repeat=args.get("repeat"),
        deliver=args.get("deliver"),
        failure_deliver=args.get("failure_deliver"),
        include_disabled=args.get("include_disabled", True),
        skill=args.get("skill"),
        skills=args.get("skills"),
        # model / provider / base_url are intentionally NOT read from the
        # agent's arguments: per-job inference pins are user-owned (dashboard,
        # `hermes cron create/edit --model`, or hand-edited jobs). The agent
        # must not be able to point unattended spend at a different model.
        # Programmatic callers of cronjob() itself retain the parameters.
        reason=args.get("reason"),
        script=args.get("script"),
        context_from=args.get("context_from"),
        continuity=args.get("continuity"),
        enabled_toolsets=args.get("enabled_toolsets"),
        workdir=args.get("workdir"),
        no_agent=args.get("no_agent"),
        attach_to_session=args.get("attach_to_session"),
        monitor_script=_mon_script,
        monitor_url=_mon_url,
        task_id=kw.get("task_id"),
        session_id=kw.get("session_id"),
    )


registry.register(
    name="cronjob_manage",
    toolset="cronjob",
    schema=CRONJOB_SCHEMA,
    handler=_cronjob_handler,
    check_fn=check_cronjob_requirements,
    emoji="⏰",
    dynamic_schema_overrides=_cronjob_schema_overrides,
)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import re  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'effective_job_state': ('cron.jobs', 'effective_job_state'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
