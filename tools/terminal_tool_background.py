"""Background-process launch path: ``terminal(background=true)`` spawns a tracked
process via the process registry (Popen locally, ``env.execute`` in a sandbox),
stamps gateway routing metadata for completion / watch-pattern notifications and
returns the JSON result. Lazy ``tools.terminal_tool`` imports keep the origin's
monkeypatch points authoritative.
"""

import json
import logging
from typing import Any, List, Optional

logger = logging.getLogger("tools.terminal_tool")

# A silent background process (no notify_on_complete / watch_patterns) is right
# only for servers/watchers; for bounded tasks the agent almost always wanted a
# notification and forgot the flag, so nudge it (cheap false positive).
_SILENT_BACKGROUND_HINT = (
    'background=true without notify_on_complete=true means this process runs SILENTLY — you '
    'will not be told when it exits. If this is a bounded task (test suite, build, CI poller, '
    'deploy, anything with a defined end), you almost certainly wanted notify_on_complete=true '
    'so the system pings you on exit. Re-launch with notify_on_complete=true, or call '
    "process(action='poll') / process(action='wait') yourself to learn the outcome. Only "
    'ignore this hint for genuine long-lived processes that never exit (servers, watchers, '
    'daemons).'
)

# Homebrewed CI pollers built on `gh pr view --json statusCheckRollup` or
# `gh pr checks | jq` fail silently in known ways (block-buffered stdout never
# reaches capture, jq null-key edge cases exit the loop, conclusion-vs-status
# confusion declares all-green early, TTY-only banners never appear piped).
# Detector is deliberately narrow: the canonical column-2 awk poller is fine.
_HOMEBREW_CI_POLLER_HINT = (
    'This looks like a homebrewed CI poller built from `gh pr view --json statusCheckRollup` '
    'and/or `gh pr checks | jq`. That shape has burned us repeatedly in hermes-agent dev work '
    '(PRs #31329, #31448, #31695, #31709, #31745, #32264, #33131) — stdout buffering kills '
    'output capture, jq null-key edge cases silently exit the loop, conclusion-vs-status field '
    'confusion exits early with bogus all-green verdicts, TTY-only summary banners never '
    'appear when piped. Use the canonical snippets in the green-ci-policy skill instead: the '
    'exit-code-driven `gh pr checks $PR >/dev/null` (rc 0 = green, 8 = pending, else fail) for '
    'exit-on-first-fail behavior, or the column-2 awk-on-tabs poller (`awk -F"\\t" '
    '"$2==\\"pending\\""`) for sharded matrices. Load '
    "skill_view(name='github/hermes-agent-dev', file_path='references/green-ci-policy.md') for "
    'the verbatim snippets. If you must roll a custom loop with rich structured output, write '
    "each tick to a known file (`tee -a /tmp/ci.log`) and rely on `process(action='log')` to "
    'read THAT file — do not rely on background-process stdout capture for line-buffered shell '
    'loops.'
)

_ASYNC_UNSUPPORTED_NOTE = (
    'notify_on_complete / watch_patterns are not available in this session — it cannot receive '
    'an async completion after the turn ends (a one-shot runner such as `hermes -z`, a cron '
    'job, a Kanban worker, or a stateless HTTP endpoint). The process is running in the '
    "background; retrieve its result with process(action='poll') or process(action='wait')."
)

# proc_session attribute -> HERMES_SESSION_* env var carrying it.
_ROUTING_FIELDS = (
    ("watcher_chat_id", "HERMES_SESSION_CHAT_ID"),
    ("watcher_user_id", "HERMES_SESSION_USER_ID"),
    ("watcher_user_name", "HERMES_SESSION_USER_NAME"),
    ("watcher_thread_id", "HERMES_SESSION_THREAD_ID"),
    ("watcher_message_id", "HERMES_SESSION_MESSAGE_ID"),
    # The spawning conversation's session-db id lets the gateway's completion
    # pre-flight drop the notification if the user closed this session (/new)
    # before the process finished, instead of injecting it into the NEW one.
    ("parent_session_id", "HERMES_SESSION_ID"),
)


def _looks_like_homebrew_ci_poller(command: str) -> bool:
    has_gh = "gh pr view" in command or "gh pr checks" in command
    has_jq = " jq " in command or "| jq" in command or "$(jq" in command
    # `gh pr checks` doesn't emit JSON, so piping it to jq is confused intent.
    return "statusCheckRollup" in command or (has_gh and has_jq)


def _stamp_gateway_routing(proc_session, get_session_env) -> None:
    """Copy the spawning chat's routing metadata onto the process session so
    completion / watch notifications reach the right chat/thread."""
    platform = get_session_env("HERMES_SESSION_PLATFORM", "")
    if not platform:
        return
    proc_session.watcher_platform = platform
    for attr, var in _ROUTING_FIELDS:
        setattr(proc_session, attr, get_session_env(var, ""))


def _spawn(process_registry, *, env, env_type, command, cwd, effective_task_id, task_id,
           session_key, effective_pty, origin_ui_session_id):
    common = dict(command=command, cwd=cwd, task_id=effective_task_id,
                  owner_task_id=task_id or effective_task_id, session_key=session_key,
                  origin_ui_session_id=origin_ui_session_id)
    if env_type == "local":
        return process_registry.spawn_local(
            env_vars=env.env if hasattr(env, 'env') else None, use_pty=effective_pty, **common)
    return process_registry.spawn_via_env(env=env, **common)


def _apply_async_support(proc_session, result_data, notify_on_complete, watch_patterns):
    """Finite sessions (stateless HTTP, one-shot Kanban workers) can't route a
    completion back after the turn ends: drop the flags and tell the agent to
    poll. Otherwise stamp gateway routing. Returns (notify, watch_patterns)."""
    if not (notify_on_complete or watch_patterns):
        return notify_on_complete, watch_patterns
    from gateway.session_context import async_delivery_supported, get_session_env

    if async_delivery_supported():
        _stamp_gateway_routing(proc_session, get_session_env)
        return notify_on_complete, watch_patterns
    result_data["notify_on_complete"] = False
    result_data["notify_unsupported"] = _ASYNC_UNSUPPORTED_NOTE
    logger.info("background proc %s: async delivery unsupported on this "
                "session; notify_on_complete/watch_patterns disabled", proc_session.id)
    return False, None


def _register_completion_watcher(process_registry, proc_session, session_key) -> None:
    """Gateway mode: register a fast watcher so completion triggers a new
    agent turn (CLI mode uses the completion_queue directly)."""
    proc_session.watcher_interval = 5
    process_registry.pending_watchers.append({
        "session_id": proc_session.id, "check_interval": 5, "session_key": session_key,
        "platform": proc_session.watcher_platform,
        **{attr.removeprefix("watcher_"): getattr(proc_session, attr)
           for attr, _ in _ROUTING_FIELDS[:-1]},
        "notify_on_complete": True, "parent_session_id": proc_session.parent_session_id,
    })


def spawn_background_process(
    *, command: str, env: Any, env_type: str, effective_task_id: str, task_id: Optional[str],
    session_key: str, workdir: Optional[str], cwd: str, effective_pty: bool,
    notify_on_complete: bool, watch_patterns: Optional[List[str]], approval_note: Optional[str],
    pty_disabled_reason: Optional[str],
) -> str:
    """Spawn *command* as a tracked background process and return the JSON result.

    Never inline-polls ``is_interrupted()``: the spawn detaches and returns
    exit_code 0 immediately, so the stale-interrupt kill cannot occur here.
    """
    from tools.process_registry import process_registry
    from tools.terminal_tool import (
        _redact_terminal_error_text, _resolve_command_cwd, _resolve_notification_flag_conflict,
    )

    effective_cwd = _resolve_command_cwd(
        workdir=workdir, default_cwd=cwd, session_key=session_key, env_type=env_type,
    )
    try:
        try:
            from gateway.session_context import get_session_env
            origin_ui_session_id = get_session_env("HERMES_UI_SESSION_ID", "") or ""
        except Exception:
            origin_ui_session_id = ""
        proc_session = _spawn(
            process_registry, env=env, env_type=env_type, command=command, cwd=effective_cwd,
            effective_task_id=effective_task_id, task_id=task_id, session_key=session_key,
            effective_pty=effective_pty, origin_ui_session_id=origin_ui_session_id,
        )
        result_data = {"output": "Background process started", "session_id": proc_session.id,
                       "pid": proc_session.pid, "exit_code": 0, "error": None}
        if approval_note:
            result_data["approval"] = approval_note
        if pty_disabled_reason:
            result_data["pty_note"] = pty_disabled_reason
        if not notify_on_complete and not watch_patterns:
            result_data["hint"] = _SILENT_BACKGROUND_HINT
        if command and _looks_like_homebrew_ci_poller(command):
            existing = result_data.get("hint", "")
            result_data["hint"] = (existing + "\n\n" + _HOMEBREW_CI_POLLER_HINT if existing
                                   else _HOMEBREW_CI_POLLER_HINT)

        notify_on_complete, watch_patterns = _apply_async_support(
            proc_session, result_data, notify_on_complete, watch_patterns)
        watch_patterns, conflict_note = _resolve_notification_flag_conflict(
            notify_on_complete=bool(notify_on_complete), watch_patterns=watch_patterns, background=True,
        )
        if conflict_note:
            logger.warning("background proc %s: %s", proc_session.id, conflict_note)
            result_data["watch_patterns_ignored"] = conflict_note
        if notify_on_complete:
            proc_session.notify_on_complete = True
            result_data["notify_on_complete"] = True
            if proc_session.watcher_platform:
                _register_completion_watcher(process_registry, proc_session, session_key)
        if watch_patterns:
            proc_session.watch_patterns = list(watch_patterns)
            result_data["watch_patterns"] = proc_session.watch_patterns
        return json.dumps(result_data, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "output": "", "exit_code": -1,
            "error": _redact_terminal_error_text(f"Failed to start background process: {e}"),
        }, ensure_ascii=False)
