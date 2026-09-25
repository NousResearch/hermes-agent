"""Remote tool RPC over one backend process instead of per-request shell polling.

The remote process relays a private Unix socket to its stdin/stdout. Only the
host dispatches tools; script stdout and the existing cell protocol stay separate.
One request is in flight, preserving the file transport's dispatch ordering.
"""
from __future__ import annotations

import json
import logging
import shlex
import subprocess
import threading
import time
import uuid

from tools.code_execution_rpc import _default_dispatch, _handle_rpc_request, _rpc_token_ok
from tools.registry import tool_error
from tools.thread_context import propagate_context_to_thread

logger = logging.getLogger(__name__)
_READY = b"hermes-code-rpc/1\n"

# This runs INSIDE the existing terminal sandbox, with no host credentials or
# network listener. EOF on the backend channel reaps it even between requests.
RELAY_SOURCE = r'''
import atexit, os, socket, sys, threading

endpoint = sys.argv[1]
os.umask(0o077)
responses = sys.stdin.buffer
requests = sys.stdout.buffer

def remove_socket():
    try:
        os.unlink(endpoint)
    except FileNotFoundError:
        pass

atexit.register(remove_socket)

def parent_gone():
    # The main loop reads stdin only while awaiting a response. A separate
    # reader owns it throughout, so EOF also terminates idle socket accepts.
    while True:
        line = responses.readline()
        if not line or not line.endswith(b"\n"):
            remove_socket()
            os._exit(0)
        replies.put(line)

import queue
replies = queue.Queue(maxsize=1)
threading.Thread(target=parent_gone, daemon=True).start()
with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
    listener.bind(endpoint)
    listener.listen(1)
    requests.write(b"hermes-code-rpc/1\n")
    requests.flush()
    while True:
        conn, _ = listener.accept()
        with conn:
            conn.settimeout(300)
            try:
                with conn.makefile("rb") as reader:
                    line = reader.readline()
                if not line or not line.endswith(b"\n"):
                    continue
                requests.write(line)
                requests.flush()
                reply = replies.get(timeout=300)
                conn.sendall(reply)
            except (OSError, queue.Empty):
                # A request may already have run: never replay it on another channel.
                sys.exit(1)
'''

# Appended to the file stubs. An endpoint is selected before a cell starts;
# once selected, a disconnect raises instead of silently falling back/replaying.
REMOTE_CLIENT_SOURCE = r'''
import socket
_file_call = _call
_stream_call_lock = threading.Lock()

def _call(tool_name, args):
    endpoint = os.environ.get("HERMES_RPC_SOCKET")
    if not endpoint:
        return _file_call(tool_name, args)
    request = json.dumps({"tool": tool_name, "args": args,
                          "token": os.environ.get("HERMES_RPC_TOKEN", "")}).encode() + b"\n"
    with _stream_call_lock, socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.settimeout(300)
        conn.connect(endpoint)
        conn.sendall(request)
        with conn.makefile("rb") as reader:
            raw = reader.readline()
        if not raw or not raw.endswith(b"\n"):
            raise RuntimeError("Tool RPC disconnected; request was not retried")
    result = json.loads(raw)
    if isinstance(result, str):
        try:
            return json.loads(result)
        except (json.JSONDecodeError, TypeError):
            pass
    return result
'''


class RemoteRPCStream:
    """One cell's channel and authority. Closing it fences queued requests."""

    def __init__(self, process, endpoint: str, task_id: str, counter: list,
                 max_tool_calls: int, allowed_tools: frozenset, rpc_token: str):
        self.process, self.endpoint = process, endpoint
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.available = False
        self.thread = threading.Thread(
            target=propagate_context_to_thread(self._serve), daemon=True,
            args=(task_id, counter, max_tool_calls, allowed_tools, rpc_token))
        self.thread.start()

    def _serve(self, task_id, counter, max_tool_calls, allowed_tools, rpc_token):
        try:
            if self.process.stdout.readline(len(_READY) + 1) != _READY:
                return
            self.available = True
            self.ready.set()
            dispatch, log = _default_dispatch(task_id), []
            while not self.stop.is_set():
                line = self.process.stdout.readline()
                if not line or self.stop.is_set():
                    break
                if not line.endswith(b"\n"):
                    break
                started = time.monotonic()
                try:
                    request = json.loads(line)
                    if (not isinstance(request, dict) or not isinstance(request.get("tool"), str)
                            or not isinstance(request.get("args", {}), dict)):
                        raise ValueError("expected tool name and argument object")
                except (ValueError, UnicodeDecodeError):
                    result = tool_error("Invalid RPC request")
                else:
                    result = _handle_rpc_request(
                        request, allowed_tools=allowed_tools, tool_call_counter=counter,
                        max_tool_calls=max_tool_calls, dispatch=dispatch, tool_call_log=log,
                        call_start=started, where="remote stream",
                    ) if _rpc_token_ok(request, rpc_token) else tool_error("Unauthorized RPC request")
                if self.stop.is_set():
                    break
                encoded = result.encode() + b"\n"
                self.process.stdin.write(encoded)
                self.process.stdin.flush()
        except (OSError, ValueError):
            logger.debug("Remote tool RPC channel closed", exc_info=True)
        finally:
            self.ready.set()
            try:
                self.process.stdin.close()
            except OSError:
                logger.debug("Remote tool RPC input already disconnected")
            self.process.stdout.close()

    def close(self):
        self.stop.set()
        try:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=2)
        except (OSError, subprocess.TimeoutExpired):
            # Teardown must not escape into the caller's pre-execution fallback:
            # the cell has already run, so doing so could replay side effects.
            logger.warning("Could not reap remote tool RPC process", exc_info=True)
        self.thread.join(timeout=5)
        # A tool handler can outlive cancellation. It observes stop before writing.


def open_remote_rpc(env, sandbox_dir: str, task_id: str, counter: list,
                    max_tool_calls: int, allowed_tools: frozenset, rpc_token: str):
    """Negotiate before running user code; unsupported/unavailable backends use files."""
    opener = getattr(env, "open_code_rpc", None)
    if not callable(opener):
        return None
    from tools.code_execution_tool import _ship_file_to_remote
    stream = None
    try:
        relay = f"{sandbox_dir}/rpc_relay.py"
        endpoint = f"{sandbox_dir}/rpc-{uuid.uuid4().hex[:8]}.sock"
        _ship_file_to_remote(env, relay, RELAY_SOURCE)
        process = opener(f"exec python3 -u {shlex.quote(relay)} {shlex.quote(endpoint)}")
        stream = RemoteRPCStream(process, endpoint, task_id, counter, max_tool_calls, allowed_tools, rpc_token)
        if stream.ready.wait(timeout=10) and stream.available:
            return stream
    except Exception:
        logger.debug("Remote tool RPC stream unavailable before execution", exc_info=True)
    if stream is not None:
        stream.close()
    return None
