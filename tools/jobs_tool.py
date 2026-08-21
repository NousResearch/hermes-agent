#!/usr/bin/env python3
"""
Jobs Tool Module

The ``jobs`` tool: owner-scoped background jobs with asynchronous collection.

Spawns commands in the background through the existing process registry
(``tools.process_registry`` — the same infrastructure that backs
``terminal(background=true)``), then lets the agent list, read, and kill
them by job id. Key semantics:

- **Owner isolation.** A job is owned by the exact (session_key, task_id)
  pair that started it. ``list`` returns only the caller's own jobs and
  ``output``/``kill`` refuse foreign job ids with the same error as an
  unknown id (no existence leak). Two sessions never see each other's jobs.
- **No busy-polling.** Every job started with ``jobs start`` is spawned with
  ``notify_on_complete=True``, so its completion lands in the existing
  background-process notification channel (CLI after-turn drain / gateway
  watcher) exactly like ``terminal(background=true, notify_on_complete=true)``.
  The tool description tells the model to keep working and collect with
  ``output`` only when blocked.
- **wait with timeout is not an error.** ``output wait=true timeout_ms=N``
  blocks up to N ms; if the job is still running when the window elapses the
  response carries ``"status": "running"`` (plus the wait note), never an
  error. The job stays alive and the completion notification still arrives.
- **Byte-bounded output.** ``output`` truncates the captured output by bytes,
  keeping the head and tail separated by a ``[output truncated: showing head
  and tail]`` marker, so a 200 KB log cannot blow up the tool result.
- **Backend scope.** v1 runs jobs on the local backend only (the same
  ``spawn_local`` path the ``process`` tool uses under ``TERMINAL_ENV=local``).
  A non-local ``TERMINAL_ENV`` is refused with a clear error instead of
  silently executing on the host.

Footprint: one tool with an ``action`` enum (``start|list|output|kill``),
mirroring the existing ``process`` tool's subcommand pattern in
``tools/process_registry.py``.

Usage:
    from tools.jobs_tool import _handle_jobs
    result = _handle_jobs({"action": "start", "command": "pytest -q"}, task_id="t1")
"""

import json
import logging
import os
import time

from tools.registry import registry, tool_error
from tools.process_registry import process_registry, _redact_process_result

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults / config
# ---------------------------------------------------------------------------
# Mirror the TypeScript reference (@deepseek-ai/dsh-jobs-local):
#   maxConcurrentJobsPerOwner default = 10.
# Override via config.yaml:  jobs:
#                              max_concurrent_per_session: 10
#                              output_limit_bytes: 20000
DEFAULT_MAX_CONCURRENT_JOBS_PER_OWNER = 10
DEFAULT_OUTPUT_LIMIT_BYTES = 20_000
_MIN_OUTPUT_LIMIT_BYTES = 512

# Allowed actions — kept here so tests and docs reference one source of truth.
JOBS_ACTIONS = ("start", "list", "output", "kill")


def _jobs_config() -> dict:
    """Read the ``jobs`` section of config.yaml without importing the CLI.

    Mirrors tools/code_execution_tool._load_config: read the lightweight raw
    config (already cached by (mtime, size)); an absent key falls back to the
    module defaults above. Never raises.
    """
    try:
        from hermes_cli.config import read_raw_config

        cfg = read_raw_config().get("jobs", {})
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def _config_int(key: str, default: int) -> int:
    """Resolve one ``jobs.*`` config int with a safe fallback."""
    raw = _jobs_config().get(key, default)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def max_concurrent_jobs_per_session() -> int:
    """Max simultaneously-running jobs an owner may have (default 10)."""
    return _config_int("max_concurrent_per_session", DEFAULT_MAX_CONCURRENT_JOBS_PER_OWNER)


def output_limit_bytes() -> int:
    """Byte cap for ``output`` results (default 20 KB)."""
    return max(_MIN_OUTPUT_LIMIT_BYTES, _config_int("output_limit_bytes", DEFAULT_OUTPUT_LIMIT_BYTES))


# ---------------------------------------------------------------------------
# Owner identity & access control
# ---------------------------------------------------------------------------
def _caller_identity(kw: dict) -> tuple[str, str]:
    """Return the caller's (session_key, task_id) owner identity.

    ``task_id`` is injected into every tool handler by the executor. The
    session key prefers an explicit kwarg (tests, RPC callers) and otherwise
    resolves through the approval contextvars/env chain that the gateway and
    CLI populate — the same source the ``process`` tool uses. Both values are
    stable across turns within one session (CLI: task_id == session id;
    gateway: session_key == gateway session key, task_id == session id), so
    a job started in turn N is collectable in turn N+k by the same session.
    """
    task_id = str(kw.get("task_id") or "")
    session_key = str(kw.get("session_key") or "")
    if not session_key:
        try:
            from tools.approval import get_current_session_key

            session_key = get_current_session_key(default="") or ""
        except Exception:
            session_key = ""
    return session_key, task_id


def _owns_job(session, session_key: str, task_id: str) -> bool:
    """Strict ownership: BOTH identities must match (AND, not OR).

    The ``process`` tool's list intentionally surfaces cross-task processes
    that share a gateway session (session-scoped discovery, #29177); jobs are
    the inverse contract — a job is visible only to the exact session that
    started it.
    """
    return (session.task_id or "") == task_id and (session.session_key or "") == session_key


def _owned_sessions(session_key: str, task_id: str) -> list:
    """All running+finished ProcessSessions owned by (session_key, task_id).

    Iterates the registry's session tables directly: ``list_sessions()``
    returns display dicts that deliberately omit the ownership fields, and
    the ownership contract here is the AND of both identities.
    """
    with process_registry._lock:
        sessions = list(process_registry._running.values()) + list(
            process_registry._finished.values()
        )
    return [s for s in sessions if _owns_job(s, session_key, task_id)]


def _require_owned_job(job_id: str, session_key: str, task_id: str):
    """Return the owned session or None. Foreign and unknown ids are
    indistinguishable (both return None) so a caller cannot probe another
    session's job ids."""
    session = process_registry.get(job_id)
    if session is None or not _owns_job(session, session_key, task_id):
        return None
    return session


# ---------------------------------------------------------------------------
# Output truncation (byte-bounded head + tail)
# ---------------------------------------------------------------------------
_TRUNCATION_MARKER = "\n[output truncated: showing head and tail]\n"


def truncate_output_bytes(text: str, limit_bytes: int) -> tuple[str, bool]:
    """Truncate *text* to *limit_bytes* bytes, keeping head and tail.

    Returns (truncated_text, was_truncated). A cut that lands mid-multibyte
    drops the incomplete sequence at the boundary (``errors="ignore"``) so
    the result is always <= *limit_bytes* bytes; a final safety trim enforces
    the bound even against replacement-char edge cases.
    """
    if limit_bytes is None:
        return text, False
    data = text.encode("utf-8", errors="replace")
    if len(data) <= limit_bytes:
        return text, False
    marker = _TRUNCATION_MARKER.encode("utf-8")
    if len(marker) >= limit_bytes:
        return marker[:limit_bytes].decode("utf-8", errors="ignore"), True
    head_bytes = (limit_bytes - len(marker)) // 2
    tail_bytes = limit_bytes - len(marker) - head_bytes
    head = data[:head_bytes].decode("utf-8", errors="ignore")
    tail = data[-tail_bytes:].decode("utf-8", errors="ignore")
    truncated = head + _TRUNCATION_MARKER + tail
    encoded = truncated.encode("utf-8", errors="replace")
    if len(encoded) > limit_bytes:
        truncated = encoded[:limit_bytes].decode("utf-8", errors="ignore")
    return truncated, True


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------
def _action_start(args: dict, session_key: str, task_id: str) -> str:
    command = str(args.get("command") or "").strip()
    if not command:
        return tool_error("command is required for action='start'")

    # v1 backend scope: local only. Refuse non-local TERMINAL_ENV explicitly
    # instead of silently running on the host for users on docker/modal/etc.
    try:
        from tools.terminal_tool import _ensure_terminal_env_bridged
        _ensure_terminal_env_bridged()
    except Exception:
        pass
    env_type = os.getenv("TERMINAL_ENV", "local") or "local"
    if env_type != "local":
        return tool_error(
            f"jobs currently supports only the local backend (TERMINAL_ENV=local); "
            f"found TERMINAL_ENV={env_type!r}. Use terminal(background=true) for "
            f"the {env_type} backend."
        )

    # Per-owner concurrency cap (running jobs only). Mirrors the TS reference's
    # maxConcurrentJobsPerOwner admission check.
    limit = max_concurrent_jobs_per_session()
    running = [s for s in _owned_sessions(session_key, task_id) if not s.exited]
    if len(running) >= limit:
        return tool_error(
            f"background job limit reached for this session (limit: {limit}); "
            f"kill an unneeded job with jobs action='kill' or wait for one to "
            f"finish, then retry"
        )

    # Security: same pre-exec guards as terminal (tirith + dangerous-command
    # detection). jobs must not become an approval bypass for local execution.
    try:
        from tools.terminal_tool import _check_all_guards
    except Exception:  # pragma: no cover — terminal_tool is always importable
        _check_all_guards = None
    if _check_all_guards is not None:
        approval = _check_all_guards(command, "local", has_host_access=True)
        if not approval["approved"]:
            if approval.get("status") == "pending_approval":
                return json.dumps({
                    "output": "",
                    "exit_code": -1,
                    "error": "",
                    "status": "pending_approval",
                    "approval_pending": True,
                    "command": approval.get("command", command),
                    "description": approval.get("description", "command flagged"),
                    "pattern_key": approval.get("pattern_key", ""),
                    "smart_denied": approval.get("smart_denied", False),
                    "allow_permanent": approval.get("allow_permanent", True),
                }, ensure_ascii=False)
            desc = approval.get("description", "command flagged")
            return json.dumps({
                "output": "",
                "exit_code": -1,
                "error": approval.get("message", f"Command denied: {desc}. Use the approval prompt to allow it, or rephrase the command."),
                "status": "blocked",
            }, ensure_ascii=False)

    # Workdir resolution mirrors terminal: explicit workdir wins, then the
    # session's recorded cwd, then the process cwd.
    workdir = args.get("workdir")
    if workdir is not None:
        try:
            from tools.terminal_tool import _validate_workdir
            wd_error = _validate_workdir(str(workdir))
        except Exception:
            wd_error = None
        if wd_error:
            return tool_error(f"Invalid workdir: {wd_error}")
    try:
        from tools.terminal_tool import _resolve_command_cwd
        cwd = _resolve_command_cwd(
            workdir=workdir,
            default_cwd=os.getcwd(),
            session_key=session_key,
            env_type="local",
        )
    except Exception:
        cwd = os.getcwd()

    session = process_registry.spawn_local(
        command=command,
        cwd=cwd,
        task_id=task_id,
        session_key=session_key,
        env_vars=None,
        use_pty=False,
    )
    # Connect to the existing completion-notification channel: on exit the
    # registry enqueues a completion event (CLI drains it after the turn,
    # gateway watchers trigger a fresh turn) — identical wiring to
    # terminal(background=true, notify_on_complete=true).
    session.notify_on_complete = True

    result = {
        "job_id": session.id,
        "status": "started",
        "command": command,
        "pid": session.pid,
        "workdir": cwd,
        "note": (
            "Job started. You will be notified automatically when it finishes — "
            "do NOT busy-poll it; keep working on independent steps. Collect the "
            "result later with jobs action='output' (job_id, wait=true only when "
            "genuinely blocked) and stop jobs that stopped mattering with "
            "action='kill'."
        ),
    }
    return json.dumps(_redact_process_result(result), ensure_ascii=False)


def _action_list(session_key: str, task_id: str) -> str:
    jobs = []
    for s in _owned_sessions(session_key, task_id):
        entry = {
            "job_id": s.id,
            "command": s.command[:200],
            "status": "exited" if s.exited else "running",
            "pid": s.pid,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(s.started_at)),
            "uptime_seconds": int(time.time() - s.started_at),
        }
        if s.exited:
            entry["exit_code"] = s.exit_code
            entry["completion_reason"] = s.completion_reason
        jobs.append(_redact_process_result(entry))
    return json.dumps({"jobs": jobs}, ensure_ascii=False)


def _action_output(args: dict, session_key: str, task_id: str) -> str:
    job_id = str(args.get("job_id") or "")
    if not job_id:
        return tool_error("job_id is required for action='output'")
    session = _require_owned_job(job_id, session_key, task_id)
    if session is None:
        # Same error for unknown and foreign ids — no existence leak.
        return tool_error(f"No job with ID {job_id}")

    wait_note = None
    if args.get("wait"):
        timeout_ms = args.get("timeout_ms")
        timeout = None
        if timeout_ms is not None:
            try:
                timeout = float(timeout_ms) / 1000.0
            except (TypeError, ValueError):
                return tool_error(f"timeout_ms must be a number (got {timeout_ms!r})")
            if timeout <= 0:
                return tool_error(f"timeout_ms must be positive (got {timeout_ms})")
        wres = process_registry.wait(job_id, timeout=timeout)
        if wres.get("status") == "not_found":
            return tool_error(f"No job with ID {job_id}")
        # timeout/interrupted both mean "still running, wait window elapsed" —
        # this is a status, NOT an error (mirrors the process tool's
        # process_running: True contract, and the TS job_output semantics).
        if wres.get("status") in ("timeout", "interrupted"):
            status = "running"
            wait_note = wres.get("timeout_note") or wres.get("note")
        else:
            status = "exited"
            if wres.get("timeout_note"):
                wait_note = wres.get("timeout_note")
        session = process_registry.get(job_id)
    else:
        status = "exited" if session.exited else "running"

    # Full captured output; read_log(offset=0) marks the completion consumed
    # when the job has exited and the whole log was shown, so the async
    # completion notice does not duplicate a result the agent already holds.
    log = process_registry.read_log(job_id, offset=0, limit=10**9)
    output = log.get("output", "")

    limit_bytes = args.get("max_output_bytes")
    if limit_bytes is not None:
        try:
            limit_bytes = int(limit_bytes)
        except (TypeError, ValueError):
            return tool_error(f"max_output_bytes must be an integer (got {limit_bytes!r})")
        if limit_bytes < _MIN_OUTPUT_LIMIT_BYTES:
            limit_bytes = _MIN_OUTPUT_LIMIT_BYTES
    else:
        limit_bytes = output_limit_bytes()

    output, truncated = truncate_output_bytes(output, limit_bytes)

    result = {
        "job_id": job_id,
        "status": status,
        "command": session.command if session else log.get("command", ""),
        "output": output,
    }
    if session is not None:
        if session.exited:
            result["exit_code"] = session.exit_code
            result["completion_reason"] = session.completion_reason
        result["uptime_seconds"] = int(time.time() - session.started_at)
    if truncated:
        result["truncated"] = True
        result["output_limit_bytes"] = limit_bytes
    if wait_note:
        result["note"] = wait_note
    return json.dumps(_redact_process_result(result), ensure_ascii=False)


def _action_kill(args: dict, session_key: str, task_id: str) -> str:
    job_id = str(args.get("job_id") or "")
    if not job_id:
        return tool_error("job_id is required for action='kill'")
    if _require_owned_job(job_id, session_key, task_id) is None:
        # Same error for unknown and foreign ids — no existence leak.
        return tool_error(f"No job with ID {job_id}")

    kres = process_registry.kill_process(job_id, source="jobs.kill")
    if kres.get("status") == "not_found":
        return tool_error(f"No job with ID {job_id}")
    result = {
        "job_id": job_id,
        "status": kres.get("status"),  # killed | already_exited
        "command": kres.get("command", ""),
        "exit_code": kres.get("exit_code"),
        "output": kres.get("output", ""),
    }
    if kres.get("status") == "already_exited":
        result["note"] = (
            "Job had already finished; nothing was killed. Its final output is "
            "in the 'output' field; completion_reason: "
            f"{kres.get('completion_reason') or 'exited'}."
        )
    return json.dumps(_redact_process_result(result), ensure_ascii=False)


def _handle_jobs(args, **kw):
    session_key, task_id = _caller_identity(kw)
    action = str(args.get("action") or "").strip()

    if action == "start":
        return _action_start(args, session_key, task_id)
    if action == "list":
        return _action_list(session_key, task_id)
    if action == "output":
        return _action_output(args, session_key, task_id)
    if action == "kill":
        return _action_kill(args, session_key, task_id)
    return tool_error(f"Unknown jobs action: {action}. Use one of: {', '.join(JOBS_ACTIONS)}")


# ---------------------------------------------------------------------------
# Registry — the "jobs" tool schema + handler
# ---------------------------------------------------------------------------
JOBS_SCHEMA = {
    "name": "jobs",
    "description": (
        "Manage YOUR OWN background jobs — commands started with 'start' and "
        "collected asynchronously. Jobs are private to the session that started "
        "them; other sessions' job ids return 'No job with ID' like unknown ids. "
        "Actions: 'start' (launch a command in the background; runs on the local "
        "backend), 'list' (your running and finished jobs), 'output' (read "
        "captured output; optional blocking wait), 'kill' (terminate a job). "
        "You are notified automatically when a job finishes — do NOT busy-poll "
        "or sleep on jobs; keep working on independent steps. Use output with "
        "wait=true only when you are genuinely blocked on the result. A wait "
        "that times out returns status 'running' (not an error) and the job "
        "keeps running; its completion notification still arrives. Output is "
        "truncated by bytes with a head/tail marker to stay bounded."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": list(JOBS_ACTIONS),
                "description": "Action to perform: start, list, output, kill",
            },
            "command": {
                "type": "string",
                "description": "Shell command to run in the background (action='start' only; required for start)",
            },
            "workdir": {
                "type": "string",
                "description": "Working directory for the job (action='start' only; defaults to the session's current directory)",
            },
            "job_id": {
                "type": "string",
                "description": "Job id returned by 'start'. Required for 'output' and 'kill'.",
            },
            "wait": {
                "type": "boolean",
                "description": "Block until the job finishes or timeout_ms elapses (action='output' only). A timed-out wait returns status 'running' — not an error.",
            },
            "timeout_ms": {
                "type": "integer",
                "minimum": 1,
                "description": "Max wait in milliseconds (action='output' with wait=true; default 30000; capped by the terminal wait limit, TERMINAL_TIMEOUT).",
            },
            "max_output_bytes": {
                "type": "integer",
                "minimum": 512,
                "description": "Byte cap for the returned output, keeping head and tail with a truncation marker (action='output' only; default from config jobs.output_limit_bytes).",
            },
        },
        "required": ["action"],
    },
}


registry.register(
    name="jobs",
    toolset="terminal",
    schema=JOBS_SCHEMA,
    handler=_handle_jobs,
    emoji="🧵",
)
