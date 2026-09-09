"""dsh_task -- run agent tasks on the local DeepSeek Harness (dsh).

Single LLM-callable tool behind a service-gated toolset ``dsh`` that is OFF
by default (mirrors the Home Assistant toolset pattern), so existing profiles
and sessions are unaffected until a user enables it.

Two actions:

* ``run`` -- one-shot headless task via the dsh CLI
  (``dsh --profile headless "<goal>"``): a fresh agent works on the goal and
  the final assistant text is returned. Blocking, bounded by the ``timeout``
  argument (config ``dsh.timeout``, default 900s); the process tree is
  killed when the deadline passes. Only a summary comes back -- the full
  session persists on disk under ~/.dsh/sessions/<cwd>, inspectable with
  action=list (requires dsh web) or read directly.
* ``list`` -- session inventory from the dsh *web* JSON-RPC endpoint
  (config ``dsh.url``, default http://127.0.0.1:3080). The web profile is
  managed externally (start-harness.ps1 / scheduled task); this tool never
  launches it.

dsh executes tasks with its own default agent settings (settings.yaml:
agent-default-model, agent presets) and keeps its upstream model credentials
in ~/.dsh/.credentials.yaml -- Hermes never sees them. Tasks can be slow and
expensive; prefer this tool when work genuinely belongs in dsh (its own
tool/session/preset ecosystem or per-cwd session grouping).
"""

import asyncio
import json
import logging

logger = logging.getLogger(__name__)

from tools import dsh_client
from tools.dsh_client import (
    DshError,
    DshNotConfigured,
    DshUnreachable,
)

# ---------------------------------------------------------------------------
# Action handlers
# ---------------------------------------------------------------------------

async def _do_run(args: dict) -> str:
    goal = str(args.get("goal") or "").strip()
    if not goal:
        return _err(
            "action=run requires a non-empty 'goal' describing the task for "
            "the dsh agent.",
            code="DSH_BAD_ARGUMENT",
        )
    cwd = args.get("cwd")
    if cwd is not None and not str(cwd).strip():
        cwd = None
    timeout = args.get("timeout")
    if timeout is not None:
        try:
            timeout = float(timeout)
        except (TypeError, ValueError):
            return _err(
                f"timeout must be a number of seconds, got {timeout!r}",
                code="DSH_BAD_ARGUMENT",
            )
        if timeout <= 0:
            return _err("timeout must be > 0", code="DSH_BAD_ARGUMENT")

    summary = await dsh_client.run_headless(
        goal, cwd=str(cwd).strip() if cwd else None, timeout=timeout
    )
    return json.dumps(summary, ensure_ascii=False)


async def _do_list(args: dict) -> str:
    limit = args.get("limit")
    if limit is not None:
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            return _err(f"limit must be an integer, got {limit!r}", code="DSH_BAD_ARGUMENT")
        if limit <= 0 or limit > 100:
            return _err("limit must be between 1 and 100", code="DSH_BAD_ARGUMENT")
    payload = await dsh_client.list_sessions(limit=limit or 20)
    return json.dumps(payload, ensure_ascii=False)


def _err(message: str, **extra) -> str:
    """Format a tool error with a stable DSH_* code."""
    from tools.registry import tool_error

    return tool_error(message, code=extra.pop("code", "DSH_ERROR"), **extra)


async def _handle_dsh_task(args: dict, **kw) -> str:
    """Dispatcher for the dsh_task tool (async: headless runs can be long)."""
    action = str(args.get("action") or "run").strip().lower()
    try:
        if action == "run":
            return await _do_run(args)
        if action == "list":
            return await _do_list(args)
        return _err(
            f"Unknown action {action!r} -- expected 'run' or 'list'.",
            code="DSH_BAD_ARGUMENT",
        )
    except (DshNotConfigured, DshUnreachable) as exc:
        # Expected operational states: log at info, guidance goes to the model.
        logger.info("dsh_task %s not usable: %s", action, exc)
        return _err(exc.message, code=exc.code, **exc.extra)
    except DshError as exc:
        logger.warning("dsh_task %s failed: [%s] %s", action, exc.code, exc)
        return _err(exc.message, code=exc.code, **exc.extra)
    except Exception as exc:  # pragma: no cover - defensive boundary
        logger.exception("dsh_task %s unexpected error", action)
        return _err(f"dsh_task {action} failed: {type(exc).__name__}: {exc}", code="DSH_INTERNAL")


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def _check_dsh_available() -> bool:
    """Tool is only visible when the user configured the dsh: block.

    Config-presence gate (like HASS_TOKEN for homeassistant): the headless
    run path needs no live port, so we deliberately do NOT probe dsh web
    here -- an unavailable web profile is a runtime error for action=list,
    not a reason to hide action=run.
    """
    config = dsh_client.load_config()
    if not config:
        return False
    return bool(dsh_client._cfg_command(config)) or bool(dsh_client._cfg_str(config, "url"))


# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------

DSH_TASK_SCHEMA = {
    "name": "dsh_task",
    "description": (
        "Run an agent task on the local DeepSeek Harness (dsh) -- DeepSeek's "
        "own agent harness with its own tools, sessions and agent presets. "
        "action=run submits the goal to a fresh headless dsh agent and waits "
        "(default 900s, overridable via 'timeout'); dsh picks the working "
        "directory from 'cwd' (default: dsh.cwd / Hermes' own). Returns a "
        "summary; the full session is stored under ~/.dsh/sessions/<cwd>. "
        "action=list shows recent dsh sessions (requires the dsh web profile "
        "on dsh.url). Tasks run with dsh's configured agent model (often a "
        "high-reasoning model) and can be slow/expensive -- use for work that "
        "genuinely belongs in dsh, not for quick lookups."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["run", "list"],
                "description": (
                    "Default 'run'. 'run' = execute one headless dsh task "
                    "(blocking). 'list' = show recent dsh sessions (needs "
                    "the dsh web profile running)."
                ),
            },
            "goal": {
                "type": "string",
                "description": (
                    "For action=run: the full task description, submitted "
                    "verbatim as the user message to the dsh agent (e.g. "
                    "'run update_all.py in this repo and report today's row "
                    "counts')."
                ),
            },
            "cwd": {
                "type": "string",
                "description": (
                    "For action=run: working directory for the dsh agent. "
                    "dsh groups its persisted sessions by cwd, so pin this to "
                    "where the work should happen. Defaults to the dsh.cwd "
                    "config value or Hermes' own working directory."
                ),
            },
            "timeout": {
                "type": "integer",
                "description": (
                    "For action=run: max seconds to wait before the dsh "
                    "process tree is killed. Default: dsh.timeout config "
                    "(900s). Raise for long tasks; on timeout the error "
                    "carries a partial-output tail."
                ),
            },
            "limit": {
                "type": "integer",
                "description": (
                    "For action=list: max number of sessions to return "
                    "(default 20, max 100)."
                ),
            },
        },
        "required": [],
    },
}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

from tools.registry import registry, tool_error  # noqa: E402

registry.register(
    name="dsh_task",
    toolset="dsh",
    schema=DSH_TASK_SCHEMA,
    handler=_handle_dsh_task,
    check_fn=_check_dsh_available,
    is_async=True,
    emoji="🤖",
    max_result_size_chars=16_000,
)
