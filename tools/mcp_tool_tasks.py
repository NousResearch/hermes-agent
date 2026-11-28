"""MCP Tasks extension (``io.modelcontextprotocol/tasks``) on the client side.

A server that supports Tasks may answer a ``tools/call`` with a task handle
(``resultType: "task"``) instead of the result, then finish the work in the
background. Hermes declares the extension on modern (2026-07-28+) sessions,
polls ``tasks/get`` at the server's suggested interval and hands the agent the
final ``CallToolResult``, so a long-running tool reads like any other call.
A failed or cancelled task becomes an ``isError`` result; a caller that stops
waiting (tool timeout, interrupt) cancels the task on the server.

``input_required`` (a task asking for mid-flight input through ``tasks/update``)
is not answered yet: the task is cancelled and the agent is told why.

Spec: https://github.com/modelcontextprotocol/ext-tasks (schema v2). Inspired by
Claude Code's MCP task support (2.1.319; Tasks GA Nov 24, 2026).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal

logger = logging.getLogger("tools.mcp_tool")

EXTENSION = "io.modelcontextprotocol/tasks"
_DEFAULT_POLL_SECONDS = 1.0
_POLL_BOUNDS_SECONDS = (0.2, 10.0)
_TERMINAL = frozenset({"completed", "failed", "cancelled"})
_MAX_POLL_MISSES = 3


def tasks_enabled(config: dict | None) -> bool:
    """On unless ``mcp_servers.<name>.tasks: false``; servers without the extension never send a task."""
    from tools.mcp_tool_common import _parse_boolish
    return _parse_boolish((config or {}).get("tasks"), default=True)


def _claim_model():
    from mcp_types import Result

    class CreateTaskResult(Result):
        """The flat ``Result & Task`` handle a server returns in place of a ``CallToolResult``."""

        result_type: Literal["task"] = "task"
        task_id: str
        status: str
        status_message: str | None = None
        ttl_ms: int | None = None
        poll_interval_ms: int | None = None

    return CreateTaskResult


_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        _MODEL = _claim_model()
    return _MODEL


def session_kwargs() -> dict[str, Any]:
    """ClientSession kwargs that advertise Tasks and let ``tools/call`` parse a task handle."""
    from mcp.client.extension import ResultClaim
    claim = ResultClaim(result_type="task", model=_model(), resolve=lambda result, ctx: poll(ctx.session, result))
    return {"extensions": {EXTENSION: {}}, "result_claims": {EXTENSION: [claim]}}


def session_declares_tasks(session) -> bool:
    return EXTENSION in (getattr(session, "_result_claims", None) or {})


async def _send(session, method: str, params: dict) -> dict:
    from mcp import types
    from pydantic import TypeAdapter
    return await session.send_request(types.Request[dict, str](method=method, params=params), TypeAdapter(dict))


def _poll_seconds(value) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        return _DEFAULT_POLL_SECONDS
    low, high = _POLL_BOUNDS_SECONDS
    return min(max(value / 1000.0, low), high)


def _error_result(text: str):
    from mcp import types
    return types.CallToolResult(content=[types.TextContent(type="text", text=text)], is_error=True)


async def _cancel_quietly(session, task_id: str) -> None:
    try:
        await asyncio.wait_for(_send(session, "tasks/cancel", {"taskId": task_id}), timeout=5)
    except BaseException:  # health: allow BLE001 -- best-effort cancel on the way out; logged with traceback
        logger.debug("MCP tasks/cancel for %s failed", task_id, exc_info=True)


async def poll(session, handle):
    """Poll ``tasks/get`` until the task is terminal; return what the tool call would have returned."""
    from mcp import types
    task_id, interval = handle.task_id, _poll_seconds(handle.poll_interval_ms)
    misses = 0
    try:
        while True:
            await asyncio.sleep(interval)
            try:
                task = await _send(session, "tasks/get", {"taskId": task_id})
            except Exception as exc:  # health: allow BLE001 -- a started task must never reach the call-replay recoverers
                # Auth expiry, a dropped session, a server error: escaping here would let the reconnect and
                # re-auth recoverers replay the whole tools/call and start the job a second time.
                misses += 1
                logger.warning("MCP tasks/get for %s failed (%d/%d): %s", task_id, misses, _MAX_POLL_MISSES, exc)
                if misses >= _MAX_POLL_MISSES:
                    return _error_result(f"MCP task {task_id} could not be polled ({type(exc).__name__}: {exc}); "
                                         "it may still be running on the server, so do not start it again "
                                         "without checking")
                continue
            misses = 0
            status = task.get("status")
            interval = _poll_seconds(task.get("pollIntervalMs", handle.poll_interval_ms))
            if status == "completed":
                return types.CallToolResult.model_validate(task.get("result") or {})
            if status == "failed":
                error = task.get("error") or {}
                return _error_result(f"MCP task {task_id} failed: {error.get('message') or task.get('statusMessage') or error}")
            if status == "cancelled":
                return _error_result(f"MCP task {task_id} was cancelled by the server"
                                     + (f": {task['statusMessage']}" if task.get("statusMessage") else ""))
            if status == "input_required":
                await _cancel_quietly(session, task_id)
                return _error_result(f"MCP task {task_id} asked for input mid-task, which Hermes cannot answer "
                                     "yet (tasks/update); the task was cancelled")
            if status != "working":
                return _error_result(f"MCP task {task_id} reported an unknown status {status!r}")
    except asyncio.CancelledError:
        await asyncio.shield(_cancel_quietly(session, task_id))
        raise


def call_tool(session, tool_name: str, args: dict, **extra):
    """``session.call_tool`` (returned as-is) unless Tasks is declared, then a coroutine that also
    resolves a task handle into its final result."""
    if not session_declares_tasks(session):
        return session.call_tool(tool_name, arguments=args, **extra)
    return _call_tool_resolving_task(session, tool_name, args, **extra)


async def _call_tool_resolving_task(session, tool_name: str, args: dict, **extra):
    result = await session.call_tool(tool_name, arguments=args, allow_claimed=True, **extra)
    if isinstance(result, _model()):
        logger.info("MCP tool %s started task %s (status=%s)", tool_name, result.task_id, result.status)
        if result.status in _TERMINAL:  # a server may hand back an already-finished handle
            return await poll(session, result.model_copy(update={"poll_interval_ms": 1}))
        return await poll(session, result)
    return result
