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

# Terminal parameters that must not be used from ephemeral sandbox scripts.
_TERMINAL_BLOCKED_PARAMS = {"background", "pty", "notify", "notify_on_complete", "watch_patterns", "heartbeat"}


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


def _rpc_server_loop(server_sock: socket.socket, task_id: str, tool_call_log: list,
                     tool_call_counter: list, max_tool_calls: int, allowed_tools: frozenset,
                     stop_event: threading.Event, rpc_token: str, dispatch=None):
    """Accept one client and serve newline-delimited JSON requests until it disconnects, idles
    300s, or the call limit is reached. ``tool_call_counter`` is a mutable ``[int]``. ``dispatch``
    overrides how an allowed, budgeted call runs: per-call sandboxes use the default (the thread
    carries the cell's context); session kernels rebind each call to the CURRENT cell's authority.
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
                        call_start=call_start, where="sandbox",
                    ) if isinstance(request, dict) and _rpc_token_ok(request, rpc_token) \
                        else tool_error("Unauthorized RPC request")
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
                   rpc_token: str):
    """Poll the remote filesystem for request files and answer them. Background thread; each
    ``env.execute()`` is an independent process, so this is safe alongside the script-execution
    thread. Response names come from the req FILENAME seq (the body seq field is untrusted),
    and each request is consumed before dispatch so a call runs at most once. Malformed,
    unauthorized, or unanswerable requests are removed without a response."""
    dispatch = _default_dispatch(task_id)
    poll_interval = 0.1
    quoted_rpc_dir = shlex.quote(rpc_dir)
    dispatched = {}    # req path -> raw body already dispatched (at-most-once)
    rm_failures = {}   # req path -> consecutive rm failures; persistent losers are skipped
    pending = {}       # res path -> base64 payload whose write has not landed yet
    error_streak = 0

    def _rm(path: str) -> None:
        """rm -rf (a req_-named DIRECTORY also passes the name filter, and -f would leave
        it spinning the loop forever; rm never follows symlinks). A transport error raises
        to the poll loop; a silent rc!=0 is counted so an undeletable file stops costing
        remote round-trips every cycle."""
        res = env.execute(f"rm -rf {shlex.quote(path)}", cwd="/", timeout=5)
        if not int(res.get("returncode") or 0):
            rm_failures.pop(path, None)
        else:
            rm_failures[path] = rm_failures.get(path, 0) + 1
            if rm_failures[path] == 3:
                logger.debug("Giving up removing RPC request %s after 3 tries", path)

    def _read_body(req_file: str):
        """Regular-file gate + size cap: a FIFO would stall the read for the full timeout,
        and a huge or device-backed file would balloon host memory (RPC reads must keep
        bounded_capture off). Returns None when the file is unreadable or not regular."""
        quoted = shlex.quote(req_file)
        res = env.execute(f"test -f {quoted} && head -c 1048576 {quoted}", cwd="/", timeout=10)
        if int(res.get("returncode") or 0) != 0:
            return None
        return res.get("output", "")

    def _deliver(res_file: str, encoded: str) -> None:
        """Atomic write (tmp + rename) via echo piping; Modal doesn't reliably deliver
        stdin_data to chained commands. rm -rf the target first: a planted res_<digits>
        DIRECTORY would otherwise make `mv` move the tmp inside it with rc=0, so the
        requester sees the name exists and crashes opening it. A failed write is kept in
        ``pending`` and retried on later cycles; dispatch is already at-most-once, only
        delivery is outstanding."""
        quoted = shlex.quote(res_file)
        res = env.execute(
            f"rm -rf {quoted} {quoted}.tmp && echo '{encoded}' | base64 -d > {quoted}.tmp"
            f" && mv {quoted}.tmp {quoted}",
            cwd="/", timeout=60,
        )
        if int(res.get("returncode") or 0):
            if res_file not in pending:
                logger.debug("RPC response write to %s failed; will retry", res_file)
            pending[res_file] = encoded
        else:
            pending.pop(res_file, None)

    while not stop_event.is_set():
        try:
            for res_file, encoded in list(pending.items()):
                if stop_event.is_set():
                    break
                _deliver(res_file, encoded)
            # find -print0 so names round-trip byte-exact: ls line-splitting mints a
            # phantom req_<digits> spelling out of a name containing a newline (the
            # phantom passes the name filter but never exists, so it spins forever),
            # and a req_-named directory lists as itself. An unterminated final entry
            # is a truncated capture and gets dropped.
            ls_result = env.execute(
                f"find {quoted_rpc_dir} -mindepth 1 -maxdepth 1 -name 'req_*' -print0 2>/dev/null || true",
                cwd="/", timeout=10)
            if int(ls_result.get("returncode") or 0) != 0:
                raise RuntimeError(f"req listing failed (rc={ls_result.get('returncode')})")
            error_streak = 0
            entries = ls_result.get("output", "").split("\0")
            if entries and entries[-1] != "":
                entries.pop()
            req_files = sorted(f for f in entries if f and not f.endswith(".tmp") and "/req_" in f)
            listed = set(req_files)
            # A req gone from the listing was really removed; a same-name file appearing
            # later is a NEW request (a second stub process reuses seq numbers).
            for path in list(dispatched):
                if path not in listed:
                    del dispatched[path]
            for path in list(rm_failures):
                if path not in listed:
                    del rm_failures[path]
            for req_file in req_files:
                if stop_event.is_set():
                    break
                if not req_file.startswith(rpc_dir + "/"):
                    continue   # banner/noise line, not ours to remove
                if rm_failures.get(req_file, 0) >= 3:
                    continue
                req_name = req_file.rsplit("/", 1)[-1]
                file_seq = req_name[4:] if req_name.startswith("req_") else ""
                # Only req_<ascii digits> names can correlate to a waiting requester;
                # anything else is unanswerable litter, removed like any malformed
                # request. isascii because str.isdigit() also accepts non-ASCII digits.
                if not (file_seq.isascii() and file_seq.isdigit()):
                    _rm(req_file)
                    continue
                call_start = time.monotonic()
                body = _read_body(req_file)
                if body is None:
                    _rm(req_file)
                    continue
                if dispatched.get(req_file) == body:
                    # The consume-rm failed silently: same request re-sighted. Remove
                    # it again, but never dispatch it twice. A same-name file with a
                    # DIFFERENT body is a new request and falls through.
                    _rm(req_file)
                    continue
                try:
                    request = json.loads(body)
                except (json.JSONDecodeError, ValueError):
                    logger.debug("Malformed RPC request in %s", req_file)
                    _rm(req_file)
                    continue
                # json.loads also succeeds on non-dict bodies (null, 5, "x", [...]);
                # _rpc_token_ok would raise AttributeError and leave the file wedged.
                if not isinstance(request, dict) or not _rpc_token_ok(request, rpc_token):
                    logger.debug("Malformed or unauthorized RPC request in %s", req_file)
                    _rm(req_file)
                    continue
                # Consume BEFORE dispatch: if the response write fails, a surviving req
                # file would re-run the same side-effecting call every cycle. An rm
                # transport error raises and retries the still-undispatched request next
                # cycle (if the rm landed remotely anyway, the file is gone and the
                # request is dropped; unavoidable without an ack protocol); a silent
                # rm failure is caught by the dispatched map on the next sighting.
                _rm(req_file)
                dispatched[req_file] = body
                tool_result = _handle_rpc_request(
                    request, allowed_tools=allowed_tools, tool_call_counter=tool_call_counter,
                    max_tool_calls=max_tool_calls, dispatch=dispatch, tool_call_log=tool_call_log,
                    call_start=call_start, where="remote sandbox",
                )
                _deliver(f"{rpc_dir}/res_{file_seq}",
                         base64.b64encode(tool_result.encode("utf-8")).decode("ascii"))
        except Exception as e:
            error_streak += 1
            if not stop_event.is_set():
                if error_streak < 4 or error_streak % 50 == 0:
                    logger.debug("RPC poll error: %s", e, exc_info=True)
                stop_event.wait(min(poll_interval * 2 ** min(error_streak, 6), 5.0))
                continue
        if not stop_event.is_set():
            stop_event.wait(poll_interval)
