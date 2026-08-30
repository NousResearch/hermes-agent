"""Host-side RPC servers for execute_code sandboxes.

Two transports share one request pipeline (token check → allow-list → call
budget → dispatch under output silence → log): ``_rpc_server_loop`` serves the
local UDS/TCP socket, ``_rpc_poll_loop`` polls a remote filesystem for request
files via ``env.execute()``.
"""

import base64
import json
import logging
import secrets
import shlex
import socket
import threading
import time

from agent.thread_scoped_output import thread_scoped_silence
from tools.registry import tool_error

# Logger name kept as the origin module's so existing log expectations hold.
logger = logging.getLogger("tools.code_execution_tool")
DEFAULT_MAX_MCP_TOOL_CALLS = 10

# Terminal parameters that must not be used from ephemeral sandbox scripts.
_TERMINAL_BLOCKED_PARAMS = {"background", "pty", "notify", "notify_on_complete", "watch_patterns"}


def _default_dispatch(task_id):
    from model_tools import handle_function_call
    return lambda tool_name, tool_args: handle_function_call(tool_name, tool_args, task_id=task_id)


def _rpc_token_ok(request: dict, rpc_token: str) -> bool:
    """Constant-time token check; an empty server token fails closed. Compared as bytes:
    compare_digest raises TypeError on a non-ASCII str, and the token is script-supplied JSON."""
    return bool(rpc_token) and secrets.compare_digest(
        str(request.get("token") or "").encode(), rpc_token.encode()
    )


def _handle_rpc_request(request: dict, *, allowed_tools: frozenset, tool_call_counter: list,
                        max_tool_calls: int, dispatch, tool_call_log: list, call_start: float,
                        where: str, mcp_tools: frozenset = frozenset(),
                        mcp_tool_call_counter: list | None = None,
                        max_mcp_tool_calls: int = DEFAULT_MAX_MCP_TOOL_CALLS) -> str:
    """Enforce allow-list + budget, then dispatch one authenticated request. Only a dispatched
    call consumes budget and is logged; refusals are free."""
    tool_name = request.get("tool", "")
    tool_args = request.get("args", {})
    if mcp_tool_call_counter is None:
        mcp_tool_call_counter = [0]
    if tool_name not in allowed_tools:
        return tool_error(f"Tool '{tool_name}' is not available in execute_code. "
                          f"Available: {', '.join(sorted(allowed_tools))}")
    is_mcp = tool_name.startswith("mcp__")
    if is_mcp and tool_name not in mcp_tools:
        return tool_error(f"MCP tool '{tool_name}' is not available in execute_code.")
    if is_mcp:
        # Revalidate live policy for every request. Persistent kernels can
        # outlive config changes or list_changed annotation downgrades.
        try:
            from tools.code_execution_tool import _sandbox_mcp_tools
            still_exposed = tool_name in _sandbox_mcp_tools(mcp_tools)
        except Exception:
            still_exposed = False
        if not still_exposed:
            return tool_error(
                f"MCP tool '{tool_name}' is no longer exposed in execute_code. "
                "It must remain enabled by code_execution.expose_mcp_tools and "
                "carry discovery-time readOnlyHint=true."
            )
        if mcp_tool_call_counter[0] >= max_mcp_tool_calls:
            return tool_error(
                f"MCP tool call limit reached ({max_mcp_tool_calls}) before "
                f"'{tool_name}'. No more MCP calls are allowed in this execution."
            )
    if tool_call_counter[0] >= max_tool_calls:
        subject = f"MCP tool '{tool_name}'" if is_mcp else f"Tool '{tool_name}'"
        return tool_error(f"{subject} was not called: total tool call limit reached "
                          f"({max_tool_calls}). No more tool calls are allowed in this execution.")
    if tool_name == "terminal" and isinstance(tool_args, dict):
        for param in _TERMINAL_BLOCKED_PARAMS:
            tool_args.pop(param, None)
    # Silence handler status prints so they don't leak into the CLI spinner.
    try:
        with thread_scoped_silence():
            result = dispatch(tool_name, tool_args)
    except Exception as exc:
        logger.error("Tool call failed in %s: %s", where, exc, exc_info=True)
        result = tool_error(str(exc))
    tool_call_counter[0] += 1
    if is_mcp:
        mcp_tool_call_counter[0] += 1
    tool_call_log.append({"tool": tool_name, "source": "mcp" if is_mcp else "built-in",
                          "args_preview": str(tool_args)[:80],
                          "duration": round(time.monotonic() - call_start, 2)})
    return result


def _rpc_server_loop(server_sock: socket.socket, task_id: str, tool_call_log: list,
                     tool_call_counter: list, max_tool_calls: int, allowed_tools: frozenset,
                     stop_event: threading.Event, rpc_token: str, dispatch=None, *,
                     mcp_tools: frozenset = frozenset(),
                     mcp_tool_call_counter: list | None = None,
                     max_mcp_tool_calls: int = DEFAULT_MAX_MCP_TOOL_CALLS):
    """Accept one client and serve newline-delimited JSON requests until it disconnects, idles
    300s, or the call limit is reached. ``tool_call_counter`` is a mutable ``[int]``. ``dispatch``
    overrides how an allowed, budgeted call runs: per-call sandboxes use the default (the thread
    carries the cell's context); session kernels rebind each call to the CURRENT cell's authority.
    """
    if dispatch is None:
        dispatch = _default_dispatch(task_id)
    if mcp_tool_call_counter is None:
        mcp_tool_call_counter = [0]
    conn = None
    try:
        server_sock.settimeout(0.05)
        while not stop_event.is_set():
            try:
                conn, _ = server_sock.accept()
                break
            except socket.timeout:
                continue
        if conn is None:
            return
        conn.settimeout(300)
        buf = b""
        while True:
            try:
                chunk = conn.recv(65536)
            except socket.timeout:
                break
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                call_start = time.monotonic()
                try:
                    request = json.loads(line.decode())
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    resp = tool_error(f"Invalid RPC request: {exc}")
                else:
                    resp = _handle_rpc_request(
                        request, allowed_tools=allowed_tools, tool_call_counter=tool_call_counter,
                        max_tool_calls=max_tool_calls, dispatch=dispatch, tool_call_log=tool_call_log,
                        call_start=call_start, where="sandbox", mcp_tools=mcp_tools,
                        mcp_tool_call_counter=mcp_tool_call_counter,
                        max_mcp_tool_calls=max_mcp_tool_calls,
                    ) if _rpc_token_ok(request, rpc_token) else tool_error("Unauthorized RPC request")
                conn.sendall((resp + "\n").encode())
    except socket.timeout:
        logger.debug("RPC listener socket timeout")
    except OSError as e:
        logger.debug("RPC listener socket error: %s", e, exc_info=True)
    finally:
        if conn:
            try:
                conn.close()
            except OSError as e:
                logger.debug("RPC conn close error: %s", e)


def _rpc_poll_loop(env, rpc_dir: str, task_id: str, tool_call_log: list, tool_call_counter: list,
                   max_tool_calls: int, allowed_tools: frozenset, stop_event: threading.Event,
                   rpc_token: str, *, mcp_tools: frozenset = frozenset(),
                   mcp_tool_call_counter: list | None = None,
                   max_mcp_tool_calls: int = DEFAULT_MAX_MCP_TOOL_CALLS):
    """Poll the remote filesystem for request files and answer them. Background thread; each
    ``env.execute()`` is an independent process, so this is safe alongside the script-execution
    thread. Malformed or unauthorized requests are removed without a response."""
    dispatch = _default_dispatch(task_id)
    if mcp_tool_call_counter is None:
        mcp_tool_call_counter = [0]
    poll_interval = 0.1
    quoted_rpc_dir = shlex.quote(rpc_dir)
    while not stop_event.is_set():
        try:
            ls_result = env.execute(f"ls -1 {quoted_rpc_dir}/req_* 2>/dev/null || true", cwd="/", timeout=10)
            output = ls_result.get("output", "").strip()
            if not output:
                stop_event.wait(poll_interval)
                continue
            req_files = sorted(f for f in (line.strip() for line in output.split("\n"))
                               if f and not f.endswith(".tmp") and "/req_" in f)
            for req_file in req_files:
                if stop_event.is_set():
                    break
                call_start = time.monotonic()
                quoted_req_file = shlex.quote(req_file)
                read_result = env.execute(f"cat {quoted_req_file}", cwd="/", timeout=10)
                try:
                    request = json.loads(read_result.get("output", ""))
                except (json.JSONDecodeError, ValueError):
                    logger.debug("Malformed RPC request in %s", req_file)
                    env.execute(f"rm -f {quoted_req_file}", cwd="/", timeout=5)
                    continue
                if not _rpc_token_ok(request, rpc_token):
                    logger.debug("Unauthorized RPC request in %s", req_file)
                    env.execute(f"rm -f {quoted_req_file}", cwd="/", timeout=5)
                    continue
                tool_result = _handle_rpc_request(
                    request, allowed_tools=allowed_tools, tool_call_counter=tool_call_counter,
                    max_tool_calls=max_tool_calls, dispatch=dispatch, tool_call_log=tool_call_log,
                    call_start=call_start, where="remote sandbox", mcp_tools=mcp_tools,
                    mcp_tool_call_counter=mcp_tool_call_counter,
                    max_mcp_tool_calls=max_mcp_tool_calls,
                )
                # Write the response atomically (tmp + rename) via echo piping —
                # Modal doesn't reliably deliver stdin_data to chained commands.
                quoted_res_file = shlex.quote(f"{rpc_dir}/res_{request.get('seq', 0):06d}")
                encoded_result = base64.b64encode(tool_result.encode("utf-8")).decode("ascii")
                env.execute(
                    f"echo '{encoded_result}' | base64 -d > {quoted_res_file}.tmp"
                    f" && mv {quoted_res_file}.tmp {quoted_res_file}",
                    cwd="/", timeout=60,
                )
                env.execute(f"rm -f {quoted_req_file}", cwd="/", timeout=5)
        except Exception as e:
            if not stop_event.is_set():
                logger.debug("RPC poll error: %s", e, exc_info=True)
        if not stop_event.is_set():
            stop_event.wait(poll_interval)
