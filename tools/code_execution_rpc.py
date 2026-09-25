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
from typing import Optional

from agent.thread_scoped_output import thread_scoped_silence
from tools.registry import tool_error

# Logger name kept as the origin module's so existing log expectations hold.
logger = logging.getLogger("tools.code_execution_tool")

# Terminal parameters that must not be used from ephemeral sandbox scripts.
_TERMINAL_BLOCKED_PARAMS = {"background", "pty", "notify", "notify_on_complete", "watch_patterns", "heartbeat"}

# Bound on one protocol message (a request line on the socket transport, a
# reply frame body on the kernel transport). On Windows the socket transport
# is loopback TCP reachable by any local process, and a peer that streams
# without a newline would otherwise grow the pre-auth read buffer without
# limit. Sized for the largest legit sandbox tool request (multi-MB
# write_file/patch payloads); same order as tools.mcp_tool_errors' inbound
# MCP body cap.
MAX_RPC_MESSAGE_BYTES = 10 * 1024 * 1024
_RPC_RECV_POLL_S = 1.0   # recv poll so stop_event teardown is prompt
_RPC_IDLE_S = 300.0      # no-data idle window per connection
_RPC_SEND_S = 60.0       # response send budget; the client reads eagerly
_RPC_DRAIN_S = 5.0       # bounded wait for an oversized line's newline


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
                        where: str) -> str:
    """Enforce allow-list + budget, then dispatch one authenticated request. Only a dispatched
    call consumes budget and is logged; refusals are free."""
    tool_name = request.get("tool", "")
    tool_args = request.get("args", {})
    if tool_name not in allowed_tools:
        return tool_error(f"Tool '{tool_name}' is not available in execute_code. "
                          f"Available: {', '.join(sorted(allowed_tools))}")
    if tool_call_counter[0] >= max_tool_calls:
        return tool_error(f"Tool call limit reached ({max_tool_calls}). "
                          "No more tool calls allowed in this execution.")
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
    tool_call_log.append({"tool": tool_name, "args_preview": str(tool_args)[:80],
                          "duration": round(time.monotonic() - call_start, 2)})
    return result


def _send_line(conn: socket.socket, resp: str) -> bool:
    """Send one response line under a bounded send budget. False when the peer
    could not be written to; the caller drops the connection."""
    conn.settimeout(_RPC_SEND_S)
    try:
        conn.sendall((resp + "\n").encode())
    except OSError:
        return False
    finally:
        conn.settimeout(_RPC_RECV_POLL_S)
    return True


def _drain_oversized_line(conn: socket.socket) -> Optional[bytes]:
    """Discard incoming bytes until the oversized line's newline so a client
    still streaming the request can read the error reply. Returns the bytes
    after the newline, or None when the line never terminated within the drain
    deadline or the peer went away."""
    end = time.monotonic() + _RPC_DRAIN_S
    while time.monotonic() < end:
        try:
            chunk = conn.recv(65536)
        except socket.timeout:
            continue
        except OSError:
            return None
        if not chunk:
            return None
        if b"\n" in chunk:
            return chunk.split(b"\n", 1)[1]
    return None


def _size_limit_error() -> str:
    limit = MAX_RPC_MESSAGE_BYTES // (1024 * 1024)
    return tool_error(f"RPC request exceeds the {limit} MiB size limit; shrink the arguments and retry")


def _rpc_server_loop(server_sock: socket.socket, task_id: str, tool_call_log: list,
                     tool_call_counter: list, max_tool_calls: int, allowed_tools: frozenset,
                     stop_event: threading.Event, rpc_token: str, dispatch=None):
    """Accept one client and serve newline-delimited JSON requests until it disconnects, idles
    _RPC_IDLE_S without data, or stop_event fires. ``tool_call_counter`` is a mutable ``[int]``.
    ``dispatch`` overrides how an allowed, budgeted call runs: session kernels rebind each call
    to the CURRENT cell's authority. Requests past MAX_RPC_MESSAGE_BYTES get a size-limit error;
    the connection is only dropped when the oversized line cannot be resynced.
    """
    if dispatch is None:
        dispatch = _default_dispatch(task_id)
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
        # Short recv poll so stop_event teardown is prompt. The idle contract
        # is 300s without received bytes: the deadline is armed once and only
        # resets on received bytes. Arming it on every loop iteration would
        # let the short recv polls re-arm it forever, so a silent peer would
        # hold the server past the contract.
        conn.settimeout(_RPC_RECV_POLL_S)
        buf = b""
        conn_dead = False
        idle_deadline = time.monotonic() + _RPC_IDLE_S
        while not (conn_dead or stop_event.is_set()):
            try:
                chunk = conn.recv(65536)
            except socket.timeout:
                if time.monotonic() >= idle_deadline:
                    break
                continue
            if not chunk:
                break
            idle_deadline = time.monotonic() + _RPC_IDLE_S
            # buf entering here has no newline (the inner loop consumes every
            # complete line), so only the appended bytes can hold the next one:
            # scanning from the append point keeps the dribble case linear.
            search_from = len(buf)
            buf += chunk
            newline = buf.find(b"\n", search_from)
            while newline >= 0 or len(buf) > MAX_RPC_MESSAGE_BYTES:
                if newline < 0:
                    # Partial line over cap: drain to its newline (bounded) so
                    # the error reply reaches a client still streaming it.
                    tail = _drain_oversized_line(conn)
                    if not _send_line(conn, _size_limit_error()) or tail is None:
                        conn_dead = True
                        break
                    buf = tail
                elif newline > MAX_RPC_MESSAGE_BYTES:
                    buf = buf[newline + 1:]
                    if not _send_line(conn, _size_limit_error()):
                        conn_dead = True
                        break
                else:
                    line, buf = buf[:newline].strip(), buf[newline + 1:]
                    if line:
                        call_start = time.monotonic()
                        try:
                            request = json.loads(line.decode())
                        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                            resp = tool_error(f"Invalid RPC request: {exc}")
                        else:
                            if not isinstance(request, dict):
                                resp = tool_error("Invalid RPC request")
                            elif not _rpc_token_ok(request, rpc_token):
                                resp = tool_error("Unauthorized RPC request")
                            else:
                                resp = _handle_rpc_request(
                                    request, allowed_tools=allowed_tools,
                                    tool_call_counter=tool_call_counter,
                                    max_tool_calls=max_tool_calls, dispatch=dispatch,
                                    tool_call_log=tool_call_log, call_start=call_start,
                                    where="sandbox",
                                )
                        if not _send_line(conn, resp):
                            conn_dead = True
                            break
                newline = buf.find(b"\n")
            # Serving the request consumed wall-clock time; a long dispatch
            # must not eat the client's idle window.
            idle_deadline = time.monotonic() + _RPC_IDLE_S
    except OSError as e:
        logger.debug("RPC connection socket error: %s", e, exc_info=True)
    finally:
        if conn:
            try:
                conn.close()
            except OSError as e:
                logger.debug("RPC conn close error: %s", e)


def _rpc_poll_loop(env, rpc_dir: str, task_id: str, tool_call_log: list, tool_call_counter: list,
                   max_tool_calls: int, allowed_tools: frozenset, stop_event: threading.Event,
                   rpc_token: str):
    """Poll the remote filesystem for request files and answer them. Background thread; each
    ``env.execute()`` is an independent process, so this is safe alongside the script-execution
    thread. Malformed or unauthorized requests are removed without a response."""
    dispatch = _default_dispatch(task_id)
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
                    call_start=call_start, where="remote sandbox",
                )
                # Write the response atomically (tmp + rename) via echo piping:
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
