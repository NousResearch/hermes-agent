"""Session-persistent Python kernels for execute_code: one child per (owner, mode,
interpreter, cwd, tool-set), one code cell per call, state survives across calls.

Constraints, in order: (1) SAME security envelope as per-call (``_build_child_env``
scrubbing, ``_rpc_server_loop`` token + per-cell tool budget, ANSI strip + secret
redaction) — only lifetime widens. (2) A wedged kernel dies, never hangs the agent:
timeout/interrupt kills the process tree and drops the registry entry; state loss is
deliberate (a cell cannot be interrupted in place safely). (3) Env frozen at spawn:
later passthrough is invisible until ``reset=true`` (the result names the kernel).

Wire protocol: one JSON request per stdin line ``{"id", "code"}``; replies framed on
stdout as ``<SENTINEL> <byte-length>\\n<json>`` with a per-kernel random SENTINEL from
the env. Bytes outside frames are raw fd output attributed to the running cell (calls
are serialized per kernel). A forged frame can only fake its own cell result.
Also hosts what ``tools.code_kernel_remote`` shares: owner resolution, registry, cell core.
"""

from __future__ import annotations

import atexit
import glob
import io
import json
import logging
import os
import queue
import secrets
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_IS_WINDOWS = sys.platform == "win32"

# Runner-side cap on captured python-level output; the host re-applies its own MAX_STDOUT cap.
_RUNNER_CAPTURE_BYTES = 1_000_000
_INLINE_SPILL_MAX_CHARS = 5_000_000
_MAX_KERNEL_FRAME_BYTES = 64 * 1024 * 1024

# Shared by both generated runners (which define _CAPTURE_LIMIT first): exec one request in the
# persistent GLOBALS namespace, build the payload. `__name__` is `__main__` as on the per-call path.
RUNNER_CELL_SOURCE = '''\
GLOBALS = {"__name__": "__main__", "__builtins__": __builtins__}


def _clip(text):
    return (text, False) if len(text) <= _CAPTURE_LIMIT else (text[:_CAPTURE_LIMIT], True)


def run_cell(request, execution_count):
    """Exec one cell; returns (response payload, FULL stdout text)."""
    out, err = io.StringIO(), io.StringIO()
    status, trace = "ok", ""
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            exec(compile(request["code"], "<cell>", "exec"), GLOBALS)
    except SystemExit as exc:
        status, trace = "exit", "SystemExit: " + repr(exc.code)
    except BaseException:
        status, trace = "error", traceback.format_exc()
    stdout_text, stdout_clipped = _clip(out.getvalue())
    stderr_text, stderr_clipped = _clip(err.getvalue())
    return {
        "id": request.get("id", ""), "status": status,
        "stdout": stdout_text, "stderr": stderr_text,
        "stdout_clipped": stdout_clipped, "stderr_clipped": stderr_clipped,
        "traceback": trace, "execution_count": execution_count,
    }, out.getvalue()
'''

KERNEL_RUNNER_SOURCE = '''\
"""Auto-generated Hermes session-kernel runner. One exec cell per request."""
import contextlib
import importlib.util
import io
import json
import os
import sys
import threading
import traceback

_SENTINEL = os.environ["HERMES_KERNEL_SENTINEL"]
_CAPTURE_LIMIT = {capture_limit}
_SPILL_DIR = os.environ.get("HERMES_KERNEL_SPILL_DIR", "")
_INLINE_SPILL = os.environ.get("HERMES_KERNEL_INLINE_SPILL") == "1"
_SPILL_CAP = {spill_cap}
_PARENT_PROCESS_HANDLE = os.environ.pop("HERMES_KERNEL_PARENT_PROCESS_HANDLE", "")
_PARENT_DEATH_FD = os.environ.pop("HERMES_KERNEL_PARENT_DEATH_FD", "")


def _preload_hermes_tools():
    """Load the generated client by exact path without listing the staging directory."""
    module_path = os.environ.pop(
        "HERMES_KERNEL_TOOLS_PATH",
        os.path.join(os.path.dirname(__file__), "hermes_tools.py"),
    )
    spec = importlib.util.spec_from_file_location("hermes_tools", module_path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load generated hermes_tools module")
    module = importlib.util.module_from_spec(spec)
    sys.modules["hermes_tools"] = module
    spec.loader.exec_module(module)


def _start_parent_death_pipe_watchdog():
    """POSIX twin of the Windows handle watchdog: exit when the parent dies.

    The host holds the only write end of an inherited pipe; a blocking read
    returns EOF the instant the host exits by ANY means (SIGKILL, OOM, crash),
    exactly like the MCP death supervisor. Stdin EOF alone is not enough: the
    main loop only sees it between cells, so a kernel SIGKILLed mid-cell
    outlived its host. Not PR_SET_PDEATHSIG — that is bound to the spawning
    THREAD, and kernels are spawned from per-cell threads that exit.
    """
    global _PARENT_DEATH_FD
    raw_fd = _PARENT_DEATH_FD
    _PARENT_DEATH_FD = ""
    if sys.platform == "win32" or not raw_fd:
        return
    try:
        fd = int(raw_fd)
        os.set_inheritable(fd, False)
    except (OSError, ValueError):
        return

    def _wait():
        try:
            while os.read(fd, 1):
                pass
        except OSError:
            pass
        os._exit(0)

    threading.Thread(target=_wait, name="hermes-parent-watchdog", daemon=True).start()


def _start_parent_process_watchdog():
    """Exit when the exact Windows parent process object is signaled.

    The inherited SYNCHRONIZE handle names a process object, not a reusable
    PID. Missing or invalid handles fail open so watchdog setup can never kill
    an otherwise healthy kernel.
    """
    global _PARENT_PROCESS_HANDLE
    raw_handle = _PARENT_PROCESS_HANDLE
    _PARENT_PROCESS_HANDLE = ""
    if sys.platform != "win32" or not raw_handle:
        return
    try:
        import ctypes
        from ctypes import wintypes

        handle = int(raw_handle)
        if handle <= 0:
            return
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.WaitForSingleObject.restype = wintypes.DWORD
        kernel32.SetHandleInformation.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.DWORD,
        ]
        kernel32.SetHandleInformation.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        # This process needs the handle, but user code spawned by a cell must
        # not pass it any further. If Windows refuses to clear inheritance,
        # disable the watchdog rather than leak the handle into cell children.
        if not kernel32.SetHandleInformation(handle, 0x00000001, 0):
            kernel32.CloseHandle(handle)
            return
    except (ImportError, OSError, TypeError, ValueError):
        return

    def _wait():
        try:
            result = kernel32.WaitForSingleObject(handle, 0xFFFFFFFF)
        finally:
            kernel32.CloseHandle(handle)
        if result == 0x00000000:  # WAIT_OBJECT_0: the parent exited
            os._exit(0)

    threading.Thread(target=_wait, name="hermes-parent-watchdog", daemon=True).start()


_start_parent_process_watchdog()
_start_parent_death_pipe_watchdog()
_preload_hermes_tools()

_real_stdout = sys.stdout

{cell_source}

def _spill(text, spill_name):
    """Best-effort: write the FULL clipped stdout to disk, return its path or ""."""
    if not _SPILL_DIR:
        return ""
    try:
        spill_path = os.path.join(_SPILL_DIR, spill_name)
        with open(spill_path, "w", encoding="utf-8", errors="replace") as f:
            f.write(text[:_SPILL_CAP])
            if len(text) > _SPILL_CAP:
                f.write("\\n\\n[... spill capped ...]")
        return spill_path
    except Exception:
        return ""


def _reply(payload):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    _real_stdout.buffer.write(("\\n" + _SENTINEL + " " + str(len(body)) + "\\n").encode("utf-8"))
    _real_stdout.buffer.write(body)
    _real_stdout.buffer.flush()


def main():
    execution_count = 0
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except ValueError:
            continue
        execution_count += 1
        payload, full_stdout = run_cell(request, execution_count)
        if payload["stdout_clipped"] and _INLINE_SPILL:
            payload["stdout_spill_content"] = full_stdout[:_SPILL_CAP]
            if len(full_stdout) > _SPILL_CAP:
                payload["stdout_spill_content"] += "\\n\\n[... spill capped ...]"
            payload["stdout_spill_path"] = ""
        else:
            payload["stdout_spill_path"] = (
                _spill(full_stdout, "cell_%06d_stdout.txt" % execution_count)
                if payload["stdout_clipped"] else ""
            )
        _reply(payload)
        if payload["status"] == "exit":
            break


if __name__ == "__main__":
    main()
'''.format(
    cell_source=RUNNER_CELL_SOURCE,
    capture_limit=_RUNNER_CAPTURE_BYTES,
    spill_cap=_INLINE_SPILL_MAX_CHARS,
)


class CellAuthority:
    """The approval/context identity of exactly one execute_code cell.

    Interpreter state persists across cells; RPC authority must not. Each cell installs a
    fresh authority captured from the CALLING thread at cell start (what
    ``propagate_context_to_thread`` captures for a per-call RPC thread) and retires it when
    the cell settles, so a late tool call (leaked background thread, raced client write) is
    refused instead of running under a stale approval/session/turn identity.
    """

    def __init__(self, task_id: str):
        import contextvars
        self.task_id = task_id
        self.ctx = contextvars.copy_context()
        self.active = True
        # ((getter, setter), captured value) per thread-local prompt callback (approval, sudo, vault unlock…)
        self._callbacks: list = []
        try:
            from tools.thread_context import _callback_api
            self._callbacks = [(pair, pair[0]()) for pair in _callback_api()]
        except Exception:
            # Fail-closed like propagate_context_to_thread: no callbacks → dangerous approvals deny.
            self._callbacks = []

    def retire(self) -> None:
        self.active = False

    def dispatch(self, tool_name: str, tool_args: dict) -> str:
        """Run one tool call under THIS cell's context and callbacks."""
        from tools.registry import tool_error
        if not self.active:
            return tool_error("No active execute_code cell: the cell this kernel call "
                              "belonged to has settled, so its tool authority is retired.")
        return self.ctx.run(self._invoke, tool_name, tool_args)

    def _invoke(self, tool_name: str, tool_args: dict) -> str:
        from model_tools import handle_function_call
        previous = None
        if self._callbacks:
            try:
                previous = [(setter, getter()) for (getter, setter), _cb in self._callbacks]
                for (_getter, setter), cb in self._callbacks:
                    setter(cb)
            except Exception:
                previous = None
        try:
            return handle_function_call(tool_name, tool_args, task_id=self.task_id)
        finally:
            if previous is not None:
                try:
                    for setter, cb in previous:
                        setter(cb)
                except Exception:
                    pass


class _BoundedBuffer:
    """Byte chunks capped at a total size; ``drain`` returns text and resets."""

    def __init__(self):
        self.chunks: List[bytes] = []
        self.total = 0

    def append(self, data: bytes, cap: int) -> None:
        keep = data[: max(0, cap - self.total)]
        if keep:
            self.chunks.append(keep)
            self.total += len(keep)

    def drain(self) -> str:
        chunks, self.chunks, self.total = self.chunks, [], 0
        return b"".join(chunks).decode("utf-8", errors="replace")


class SessionKernel:
    """One live kernel process plus its RPC server and reader threads."""

    def __init__(self, key: Tuple):
        self.key, self.owner, self.lock = key, key[0], threading.Lock()
        self.proc: Optional[subprocess.Popen] = None
        self.tmpdir = self.rpc_token = self.sentinel = ""
        self.sock_path: Optional[str] = None
        self.server_sock: Optional[socket.socket] = None
        self.rpc_peer_uid: Optional[int] = None
        self.stop_event = threading.Event()
        self.death_pipe_w: Optional[int] = None
        self.broker_lease: Optional[socket.socket] = None
        self.tool_call_log: List = []
        self.tool_call_counter: List[int] = [0]
        # Cells currently attached (bumped under the registry lock on selection, dropped when the
        # cell settles). Reaping/cap-eviction skip attached kernels: tearing one down mid-spawn
        # rmtree'd the staging dir under the spawner and killed live cells.
        self.attached: int = 0
        # Owned by a live delegate_task child: exempt from LRU eviction (the child's teardown disposes it).
        self.pinned: bool = False
        self.response_q: "queue.Queue[dict]" = queue.Queue()
        self.raw, self.stderr = _BoundedBuffer(), _BoundedBuffer()
        self.execution_count, self.last_used = 0, time.monotonic()
        self.cell_authority: Optional[CellAuthority] = None

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def dead(self) -> bool:
        """True only once a spawned process has exited. ``proc is None`` is mid-spawn, not dead:
        parallel cells for one owner race the first ``_spawn``, and treating the pending kernel as
        dead made every racer replace it, orphaning the winner's process outside the registry."""
        return self.proc is not None and self.proc.poll() is not None

    def teardown(self) -> None:
        self.stop_event.set()
        if self.death_pipe_w is not None:
            try:
                os.close(self.death_pipe_w)
            except OSError:
                pass
            self.death_pipe_w = None
        if self.broker_lease is not None:
            from tools.local_exec_broker import BrokerError

            try:
                process_alive = self.alive()
            except BrokerError as exc:
                process_alive = False
                logger.warning(
                    "broker-owned kernel returned an invalid exit reply during teardown: %s",
                    exc,
                )
            if process_alive:
                self.proc.kill()
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning(
                        "broker-owned kernel %s did not exit within teardown grace",
                        self.proc.pid,
                    )
                    self.proc._close_owned_handles()
                except BrokerError as exc:
                    logger.warning(
                        "broker-owned kernel %s returned an invalid exit reply: %s",
                        self.proc.pid,
                        exc,
                    )
        elif self.alive():
            from tools.code_execution_tool import _kill_process_group

            _kill_process_group(self.proc, escalate=True)
        if self.broker_lease is not None:
            try:
                self.broker_lease.close()
            except OSError:
                pass
            self.broker_lease = None
        sock, self.server_sock = self.server_sock, None
        try:
            if sock is not None:
                sock.close()
            if self.sock_path:
                os.unlink(self.sock_path)
        except OSError:
            pass
        if self.tmpdir:
            import shutil
            shutil.rmtree(self.tmpdir, ignore_errors=True)


class KernelRegistry:
    """Key -> kernel map plus its lock (shared with the remote registry). Kernels are popped
    under the lock and torn down outside it — teardown may block on the child or the transport."""

    def __init__(self, teardown: Callable[[Any], None]):
        self.kernels: Dict[Tuple, Any] = {}
        self.lock, self._teardown = threading.Lock(), teardown

    def shutdown(self, owner: Optional[str] = None, *, owner_matches: Optional[Callable[[str], bool]] = None) -> None:
        """Tear down every kernel, every kernel one owner (key[0]) holds, or every kernel whose owner
        satisfies ``owner_matches``."""
        with self.lock:
            doomed = [self.kernels.pop(key) for key in list(self.kernels)
                      if (owner is None and owner_matches is None) or key[0] == owner
                      or (owner_matches is not None and owner_matches(key[0]))]
        for kernel in doomed:
            self._teardown(kernel)

    def discard(self, key: Tuple, kernel: Any) -> None:
        """Drop *kernel*'s registry entry (only if it is still the one registered under *key* —
        never a replacement) and tear the kernel down."""
        with self.lock:
            if self.kernels.get(key) is kernel:
                self.kernels.pop(key, None)
        self._teardown(kernel)


_REGISTRY = KernelRegistry(lambda kernel: kernel.teardown())
_KERNELS: Dict[Tuple, SessionKernel] = _REGISTRY.kernels

# Bounded lifecycle defaults (config: code_execution.max_session_kernels / kernel_idle_timeout).
# A long-lived gateway must never accumulate one live child per finished conversation:
# stable owner id, owner-teardown disposal, idle reaping, max-live bound.
# See #88637.
DEFAULT_MAX_SESSION_KERNELS = 4
DEFAULT_KERNEL_IDLE_TIMEOUT = 1800


def _lifecycle_limits() -> Tuple[int, int]:
    from tools.code_execution_tool import _load_config
    config = _load_config()
    def limit(key: str, default: int) -> int:
        try:
            return max(1, int(config.get(key, default)))
        except (TypeError, ValueError):
            return default
    return limit("max_session_kernels", DEFAULT_MAX_SESSION_KERNELS), limit("kernel_idle_timeout", DEFAULT_KERNEL_IDLE_TIMEOUT)


_CHILD_OWNER_QUALIFIER = "::child::"


def _resolve_owner(task_id: str) -> str:
    """The stable identity a session kernel belongs to: the conversation's approval session key
    (context-propagated, stable across turns, distinct per session). ``run_agent`` mints a fresh
    task id per turn, so a task-keyed kernel would neither survive the next turn nor be torn down
    with anything; the task id is only the last-resort owner (embeds/tests without a session).

    Delegated children INHERIT the parent's approval session key — without the ``::child::``
    qualifier a child's execute_code would attach to the parent's kernel and read its state
    (verified live, both directions). Children get their own kernels keyed by delegation session id.
    """
    try:
        from tools.approval_context import get_current_session_key
        session_key = get_current_session_key(default="")
    except Exception:
        session_key = ""
    owner = session_key or (task_id or "")
    try:
        from agent.delegation_context import is_delegated_child_context
        if is_delegated_child_context():
            from gateway.session_context import get_session_env
            child_id = get_session_env("HERMES_SESSION_ID", "") or (task_id or "")
            owner = f"{owner}{_CHILD_OWNER_QUALIFIER}{child_id}"
    except Exception:
        pass
    return owner


def shutdown_all_kernels() -> None:
    """Kill every session kernel. Registered via atexit; also used by tests."""
    _REGISTRY.shutdown()


def shutdown_kernels_for_owner(owner: str) -> None:
    """Dispose every kernel a session owns — wired into ``tools.approval.clear_session``
    so kernels die at the same boundary that clears approval/yolo state (/new, session close).

    See #88637.
    """
    if owner:
        _REGISTRY.shutdown(owner)


def delegated_child_owner_matcher(child_session_id: str) -> Callable[[str], bool]:
    """Predicate for the kernels a delegate_task child owns (``_resolve_owner`` qualifies a child's
    owner with its delegation session id). Shared with the remote registry."""
    suffix = f"{_CHILD_OWNER_QUALIFIER}{child_session_id}"
    return lambda owner: owner.endswith(suffix)


def shutdown_kernels_for_delegated_child(child_session_id: str) -> None:
    """Dispose a finished child's kernels (local and remote). A child's kernel lives exactly as long as the
    child: pinned against LRU eviction while it runs, torn down here — otherwise finished children's
    kernels squatted the process-wide cap for ``kernel_idle_timeout`` and evicted LIVE children's kernels,
    which then lost their state mid-task with no signal but ``reused: false``."""
    if not child_session_id:
        return
    matcher = delegated_child_owner_matcher(child_session_id)
    _REGISTRY.shutdown(owner_matches=matcher)
    from tools.code_kernel_remote import shutdown_remote_kernels_where
    shutdown_remote_kernels_where(matcher)


atexit.register(shutdown_all_kernels)


class _UidFilteringSocket:
    """Accept only Unix peers running as the configured broker uid."""

    def __init__(self, server_sock: socket.socket, expected_uid: int):
        self._server_sock = server_sock
        self._expected_uid = expected_uid

    def settimeout(self, timeout: float) -> None:
        self._server_sock.settimeout(timeout)

    def accept(self):
        while True:
            conn, address = self._server_sock.accept()
            try:
                _pid, uid, _gid = struct.unpack(
                    "3i", conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
                )
            except OSError:
                conn.close()
                continue
            if uid == self._expected_uid:
                return conn, address
            conn.close()


def _rpc_forever(kernel: SessionKernel, max_tool_calls: int,
                 sandbox_tools: frozenset) -> None:
    """Serve tool RPC for the kernel's whole life: ``_rpc_server_loop`` returns on disconnect or
    its 300s idle timeout, and a kernel idles longer between cells, so re-accept until teardown
    (the client stub reconnects: HERMES_RPC_PERSISTENT). The serving thread carries NO frozen
    authority — every dispatch routes through the CURRENT cell's ``CellAuthority``."""
    from tools.code_execution_rpc import _rpc_server_loop
    from tools.registry import tool_error
    def _dispatch(tool_name: str, tool_args: dict) -> str:
        authority = kernel.cell_authority
        if authority is None:
            return tool_error("No active execute_code cell: this kernel has no cell authority installed.")
        return authority.dispatch(tool_name, tool_args)
    while not kernel.stop_event.is_set():
        server_sock = kernel.server_sock
        if kernel.rpc_peer_uid is not None:
            server_sock = _UidFilteringSocket(server_sock, kernel.rpc_peer_uid)
        _rpc_server_loop(server_sock, "", kernel.tool_call_log, kernel.tool_call_counter,
                         max_tool_calls, sandbox_tools, kernel.stop_event, kernel.rpc_token,
                         dispatch=_dispatch)


def _materialize_inline_spill(kernel: SessionKernel, payload: Dict[str, Any]) -> None:
    """Persist a broker child's bounded spill payload into host-owned staging."""
    content = payload.pop("stdout_spill_content", None)
    if not payload.get("stdout_clipped") or not isinstance(content, str):
        return
    if len(content) > _INLINE_SPILL_MAX_CHARS:
        content = (
            content[:_INLINE_SPILL_MAX_CHARS]
            + "\n\n[... spill capped ...]"
        )
    try:
        count = int(payload.get("execution_count", 0))
    except (TypeError, ValueError):
        count = 0
    spill_path = os.path.join(
        kernel.tmpdir,
        f"cell_{max(0, count):06d}_{secrets.token_hex(8)}_stdout.txt",
    )
    fd = None
    try:
        fd = os.open(
            spill_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
            0o600,
        )
        with os.fdopen(fd, "w", encoding="utf-8", errors="replace") as spill_file:
            fd = None
            spill_file.write(content)
        payload["stdout_spill_path"] = spill_path
    except OSError:
        payload["stdout_spill_path"] = ""
    finally:
        if fd is not None:
            os.close(fd)


def _stdout_reader(kernel: SessionKernel) -> None:
    """Split the child's stdout into protocol frames and raw passthrough."""
    from tools.code_execution_tool import MAX_STDOUT_BYTES
    assert kernel.proc is not None and kernel.proc.stdout is not None
    stream = kernel.proc.stdout
    marker = ("\n" + kernel.sentinel + " ").encode("utf-8")
    def raw(data: bytes) -> None:
        kernel.raw.append(data, MAX_STDOUT_BYTES)
    buf = b""
    while True:
        # read1 returns as soon as any bytes arrive; a plain read(n) on a BufferedReader
        # blocks until n bytes or EOF and would sit on a complete small frame forever.
        chunk = stream.read1(4096)
        if not chunk:
            if buf:
                raw(buf)
            kernel.response_q.put({"status": "kernel-eof"})
            return
        buf += chunk
        while True:
            index = buf.find(marker)
            if index < 0:
                # Keep a marker-sized tail (marker may be split across reads); the rest is raw.
                spill = buf[: -len(marker)] if len(buf) > len(marker) else b""
                if spill:
                    raw(spill)
                    buf = buf[len(spill):]
                break
            if index:
                raw(buf[:index])
            rest = buf[index + len(marker):]
            newline = rest.find(b"\n")
            if newline < 0:
                buf = buf[index:]
                break
            try:
                length = int(rest[:newline])
            except ValueError:
                # Not a real frame header (user output containing the marker bytes): raw.
                raw(marker)
                buf = rest
                continue
            if length < 0 or length > _MAX_KERNEL_FRAME_BYTES:
                kernel.response_q.put({"status": "protocol-error"})
                return
            body = rest[newline + 1:]
            while len(body) < length:
                more = stream.read1(length - len(body))
                if not more:
                    kernel.response_q.put({"status": "kernel-eof"})
                    return
                body += more
            try:
                payload = json.loads(body[:length].decode("utf-8", errors="replace"))
                if not isinstance(payload, dict):
                    raise ValueError("kernel response is not an object")
                _materialize_inline_spill(kernel, payload)
                kernel.response_q.put(payload)
            except ValueError:
                kernel.response_q.put({"status": "protocol-error"})
            buf = body[length:]


def _stderr_reader(kernel: SessionKernel) -> None:
    from tools.code_execution_tool import MAX_STDERR_BYTES
    assert kernel.proc is not None and kernel.proc.stderr is not None
    while chunk := kernel.proc.stderr.read1(4096):
        kernel.stderr.append(chunk, MAX_STDERR_BYTES)


def _bind_rpc_socket(kernel: SessionKernel, *, cross_uid: bool = False) -> str:
    """Bind tool RPC, keeping cross-UID UDS reachability inside unlistable staging."""
    if _IS_WINDOWS:
        kernel.sock_path = None
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.bind(("127.0.0.1", 0))
        host, port = server_sock.getsockname()[:2]
        rpc_endpoint = f"tcp://{host}:{port}"
    else:
        if cross_uid:
            # The random 0711 staging directory is not listable by the broker uid or
            # unrelated local users. The broker child receives the exact path, while
            # the 0666 socket lets that known cross-UID child connect. The RPC token
            # remains the application-level authentication boundary.
            sock_tmpdir = kernel.tmpdir
            socket_mode = 0o666
        else:
            from hermes_constants import socket_safe_tmpdir

            sock_tmpdir = socket_safe_tmpdir()
            socket_mode = 0o600
        rpc_endpoint = kernel.sock_path = os.path.join(
            sock_tmpdir, f"hermes_rpc_{uuid.uuid4().hex}.sock"
        )
        server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server_sock.bind(kernel.sock_path)
        os.chmod(kernel.sock_path, socket_mode)
    server_sock.listen(1)
    kernel.server_sock = server_sock
    return rpc_endpoint


def _parent_process_handle(child_env: Dict[str, str]):
    """Windows: open an inheritable SYNCHRONIZE handle to this process for the kernel's parent-death
    watchdog. Returns (handle, CloseHandle, startupinfo) or (None, None, None); fails open."""
    handle = close = startupinfo = None
    try:
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcessId.argtypes = []
        kernel32.GetCurrentProcessId.restype = wintypes.DWORD
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        close = kernel32.CloseHandle
        # SYNCHRONIZE; inherited only by the explicitly allow-listed child.
        handle = kernel32.OpenProcess(0x00100000, True, kernel32.GetCurrentProcessId())
        if handle:
            child_env["HERMES_KERNEL_PARENT_PROCESS_HANDLE"] = str(int(handle))
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.lpAttributeList = {"handle_list": [int(handle)]}
    except (AttributeError, ImportError, OSError, TypeError, ValueError):
        if handle and close is not None:
            close(handle)
        child_env.pop("HERMES_KERNEL_PARENT_PROCESS_HANDLE", None)
        handle = close = startupinfo = None
    return handle, close, startupinfo


class _ReaderOwnedStream(io.BufferedReader):
    """Kernel output whose close yields to the reader thread parked inside it.

    ``BufferedReader.close()`` takes the buffer lock, and ``_stdout_reader`` holds that lock
    for as long as it sits in ``read1`` — that is, until EOF, which needs every write end of
    the pipe gone. The child's own copy dies with it, but a grandchild that inherited a cell's
    stdout keeps one open indefinitely, so a status check that closed the stream inline (the
    recycled-PID path through ``poll``) blocked teardown on a reader that was never going to
    return. Closing is therefore a request: a stream nobody is reading closes immediately —
    every teardown that is not racing a reader — and otherwise the reader inside it performs
    the close on its way out, which is the first moment the descriptor is safe to release.

    Reads after a close report EOF rather than raising, because the readers are abandoned
    daemon threads whose only job left is to notice the kernel is gone and unwind.
    """

    def __init__(self, raw):
        super().__init__(raw)
        self._readers = 0
        self._close_requested = False
        self._state = threading.Lock()

    def _guarded(self, read, *args):
        with self._state:
            if self._close_requested:
                return b""
            self._readers += 1
        try:
            return read(*args)
        finally:
            # Last reader out closes for the teardown that handed us the stream. The buffer
            # lock is free again here, so this cannot block; holding _state keeps a new read
            # from slipping in ahead of it.
            with self._state:
                self._readers -= 1
                if self._close_requested and not self._readers:
                    super().close()

    def read(self, *args):
        return self._guarded(super().read, *args)

    def read1(self, *args):
        return self._guarded(super().read1, *args)

    def readline(self, *args):
        return self._guarded(super().readline, *args)

    def peek(self, *args):
        return self._guarded(super().peek, *args)

    def close(self):
        with self._state:
            self._close_requested = True
            if self._readers:
                return
            super().close()


class _BrokerKernelProcess:
    """Binary Popen-like handle whose broker connection leases the kernel."""

    def __init__(
        self,
        conn,
        pid: int,
        start_time: int,
        stdin_fd: int,
        stdout_fd: int,
        stderr_fd: int,
        remainder: bytes,
    ):
        self._conn = conn
        self.pid = pid
        self._start_time = start_time
        self.stdin = os.fdopen(stdin_fd, "wb")
        # Not os.fdopen: teardown runs concurrently with the reader threads that own these two,
        # so their close has to be one a reader can finish (_ReaderOwnedStream).
        self.stdout = _ReaderOwnedStream(io.FileIO(stdout_fd, "r", closefd=True))
        self.stderr = _ReaderOwnedStream(io.FileIO(stderr_fd, "r", closefd=True))
        self.returncode: Optional[int] = None
        self._reply = remainder
        self._poll_lock = threading.Lock()
        self._lease_closed = False
        conn.setblocking(False)

    def _close_lease_and_input(self) -> None:
        # The broker connection is the kill signal. Close it before waiting on any
        # stream lock held by the reader thread, or teardown can deadlock while the
        # still-live child keeps stdout open.
        try:
            self._conn.close()
        except OSError:
            pass
        try:
            self.stdin.close()
        except (OSError, ValueError):
            pass

    def _close_owned_handles(self) -> None:
        self._close_lease_and_input()
        for stream in (self.stdout, self.stderr):
            try:
                stream.close()
            except (OSError, ValueError):
                pass

    def _finish(self, returncode: int) -> int:
        self.returncode = returncode
        self._close_owned_handles()
        return returncode

    def poll(self):
        with self._poll_lock:
            if self.returncode is not None:
                return self.returncode
            from tools.local_exec_broker import (
                BrokerError,
                MAX_REPLY_BYTES,
                _process_start_time,
            )

            if self._lease_closed:
                try:
                    current_start_time = _process_start_time(self.pid)
                except (OSError, ValueError):
                    return self._finish(-9)
                if current_start_time != self._start_time:
                    return self._finish(-9)
                return None
            if b"\n" not in self._reply:
                try:
                    chunk = self._conn.recv(4096)
                except BlockingIOError:
                    return None
                except OSError:
                    return self._finish(-1)
                if not chunk:
                    return self._finish(-1)
                self._reply += chunk
            if len(self._reply) > MAX_REPLY_BYTES:
                self._finish(-1)
                raise BrokerError("bad_reply", "broker returned an oversized exit reply")
            if b"\n" not in self._reply:
                return None
            try:
                reply = json.loads(self._reply.split(b"\n", 1)[0])
                returncode = reply["exit"]
                if not isinstance(returncode, int) or isinstance(returncode, bool):
                    raise ValueError("exit status is not an integer")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                self._finish(-1)
                raise BrokerError(
                    "bad_reply", f"broker returned an invalid exit reply: {exc}"
                ) from exc
            return self._finish(returncode)

    def wait(self, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.poll() is None:
            if deadline is not None and time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired("local execution broker", timeout)
            time.sleep(0.01)
        return self.returncode

    def kill(self):
        with self._poll_lock:
            if self.returncode is not None:
                return
            self._lease_closed = True
            self._close_lease_and_input()


def _local_exec_broker_config() -> Optional[Tuple[str, int]]:
    """Resolve and validate the opt-in broker endpoint for this profile."""
    if _IS_WINDOWS:
        return None
    from hermes_cli.config import load_config_readonly
    config = (load_config_readonly() or {}).get("terminal") or {}
    if "local_exec_broker" not in config:
        return None
    broker = config["local_exec_broker"]
    socket_path = broker.get("socket") if isinstance(broker, dict) else None
    uid = broker.get("uid") if isinstance(broker, dict) else None
    if not isinstance(socket_path, str) or not socket_path:
        raise RuntimeError(
            "terminal.local_exec_broker requires a non-empty string socket"
        )
    if not isinstance(uid, int) or isinstance(uid, bool) or uid < 0:
        raise RuntimeError(
            "terminal.local_exec_broker requires a non-negative integer uid"
        )
    return socket_path, uid


def _spawn_through_broker(
    runner_path: str,
    child_python: str,
    child_cwd: str,
    child_env: Dict[str, str],
    broker_config: Tuple[str, int],
) -> Tuple[_BrokerKernelProcess, socket.socket]:
    """Launch a kernel with explicit stdio; the returned connection is its lease."""
    from tools.local_exec_broker import BrokerError, request_launch

    stdin_r = stdin_w = stdout_r = stdout_w = stderr_r = stderr_w = runner_fd = None
    conn = None
    try:
        stdin_r, stdin_w = os.pipe()
        stdout_r, stdout_w = os.pipe()
        stderr_r, stderr_w = os.pipe()
        runner_fd = os.open(runner_path, os.O_RDONLY)
        conn, reply, remainder = request_launch(
            broker_config[0],
            expected_peer_uid=broker_config[1],
            runner_fd=runner_fd,
            runner_python=child_python,
            cwd=child_cwd or None,
            env=child_env,
            fds=[],
            stdin_fd=stdin_r,
            stdout_fd=stdout_w,
            stderr_fd=stderr_w,
            scratch=True,
        )
        pid = reply.get("pid")
        start_time = reply.get("start_time")
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
            raise BrokerError("bad_reply", "broker returned an invalid launch reply")
        if (
            not isinstance(start_time, int)
            or isinstance(start_time, bool)
            or start_time <= 0
        ):
            raise BrokerError("bad_reply", "broker returned an invalid process identity")
        os.close(stdin_r)
        stdin_r = None
        os.close(stdout_w)
        stdout_w = None
        os.close(stderr_w)
        stderr_w = None
        # From here the Popen-like handle owns the three parent pipe ends. Disclaim
        # them before construction so outer cleanup cannot re-close a recycled fd if
        # construction fails after taking ownership.
        handle_stdin_fd, stdin_w = stdin_w, None
        handle_stdout_fd, stdout_r = stdout_r, None
        handle_stderr_fd, stderr_r = stderr_r, None
        proc = _BrokerKernelProcess(
            conn,
            pid,
            start_time,
            handle_stdin_fd,
            handle_stdout_fd,
            handle_stderr_fd,
            remainder,
        )
        return proc, conn
    except BaseException:
        if conn is not None:
            conn.close()
        raise
    finally:
        for fd in (
            stdin_r,
            stdin_w,
            stdout_r,
            stdout_w,
            stderr_r,
            stderr_w,
            runner_fd,
        ):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass


def _spawn(kernel: SessionKernel, *, child_python: str, child_cwd: str,
           sandbox_tools: frozenset, max_tool_calls: int, task_id: str = "") -> None:
    from tools.code_execution_env import _build_child_env
    from tools.code_execution_tool import generate_hermes_tools_module
    broker_config = _local_exec_broker_config()
    kernel.rpc_peer_uid = broker_config[1] if broker_config is not None else None
    kernel.tmpdir = tempfile.mkdtemp(
        prefix="hermes_kernel_",
        # no-tmp: ok — a broker child under another uid must traverse the parent directory.
        dir="/tmp" if broker_config is not None else None,
    )
    if broker_config is not None:
        # The broker child runs as a different, deliberately less-privileged uid. It needs to
        # traverse the random staging directory to import hermes_tools.py. Keep directory
        # listing owner-only; writes stay in sticky work and spill subdirectories.
        os.chmod(kernel.tmpdir, 0o711)
    kernel.rpc_token = secrets.token_urlsafe(32)
    kernel.sentinel = "@@HERMES-KERNEL-" + secrets.token_urlsafe(16) + "@@"
    rpc_endpoint = _bind_rpc_socket(kernel, cross_uid=broker_config is not None)
    for name, src in (("hermes_tools.py", generate_hermes_tools_module(list(sandbox_tools))),
                      ("hermes_kernel_runner.py", KERNEL_RUNNER_SOURCE)):
        module_path = Path(kernel.tmpdir, name)
        module_path.write_text(src, encoding="utf-8")
        if broker_config is not None:
            module_path.chmod(0o644)
    child_env = _build_child_env(rpc_endpoint=rpc_endpoint, rpc_token=kernel.rpc_token,
                                 tmpdir=kernel.tmpdir, child_python=child_python)
    child_env["HERMES_KERNEL_SENTINEL"] = kernel.sentinel
    child_env["HERMES_KERNEL_TOOLS_PATH"] = os.path.join(
        kernel.tmpdir, "hermes_tools.py"
    )
    # Full clipped stdout must land in host-owned staging so the agent can page it with
    # read_file. A cross-UID broker child returns the bounded spill in its framed response;
    # the host reader materializes it without creating a world-writable rendezvous directory.
    spill_dir = kernel.tmpdir
    broker_cwd = child_cwd
    if broker_config is not None:
        child_env["HERMES_KERNEL_INLINE_SPILL"] = "1"
    else:
        child_env["HERMES_KERNEL_SPILL_DIR"] = spill_dir
    # Generated client reconnects after the RPC server's 300s idle timeout between cells.
    child_env["HERMES_RPC_PERSISTENT"] = "1"
    # Parent-death watchdog plumbing: Windows inherits a SYNCHRONIZE handle to this process; POSIX
    # inherits the read end of a pipe whose only write end we hold (EOF == host gone, any cause).
    parent_handle, close_handle, startupinfo = _parent_process_handle(child_env) if _IS_WINDOWS else (None, None, None)
    death_r: Optional[int] = None
    pass_fds: Tuple[int, ...] = ()
    if not _IS_WINDOWS and broker_config is None:
        death_r, kernel.death_pipe_w = os.pipe()
        child_env["HERMES_KERNEL_PARENT_DEATH_FD"] = str(death_r)
        pass_fds = (death_r,)
    try:
        runner_path = os.path.join(kernel.tmpdir, "hermes_kernel_runner.py")
        if broker_config is not None:
            kernel.proc, kernel.broker_lease = _spawn_through_broker(
                runner_path,
                child_python,
                broker_cwd,
                child_env,
                broker_config,
            )
        else:
            kernel.proc = subprocess.Popen(
                [child_python, runner_path],
                # Strict mode passes an empty cwd: the kernel's staging dir plays the per-call tmpdir's role.
                cwd=child_cwd or kernel.tmpdir, env=child_env, start_new_session=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if _IS_WINDOWS else 0,
                close_fds=True, pass_fds=pass_fds, startupinfo=startupinfo,
            )
    finally:
        if parent_handle and close_handle is not None:
            close_handle(parent_handle)
        if death_r is not None:
            os.close(death_r)
    # Deliberately NOT propagate_context_to_thread: that would freeze the spawning cell's
    # context/callbacks into the server thread for life. Authority is rebound per cell.
    for target, args in ((_rpc_forever, (kernel, max_tool_calls, sandbox_tools)),
                         (_stdout_reader, (kernel,)), (_stderr_reader, (kernel,))):
        threading.Thread(target=target, args=args, daemon=True).start()
    _ensure_background_reaper()


def _pop_idle_expired(now: float, idle_timeout: float) -> List[SessionKernel]:
    """Pop (caller holds ``_REGISTRY.lock``) every kernel idle past *idle_timeout*. Kernels with
    attached cells are skipped: the last cell out tears them down."""
    return [_KERNELS.pop(k) for k in list(_KERNELS)
            if _KERNELS[k].attached == 0 and now - _KERNELS[k].last_used > idle_timeout]


def _acquire_kernel(key: Tuple, reset: bool, *, pinned: bool = False) -> Tuple[SessionKernel, bool]:
    """Look up or register the kernel for *key*; returns (kernel, state_reset). Every entry also
    sweeps idle-expired kernels and enforces the process-wide LRU cap (doomed kernels are popped
    under the lock, torn down outside it), so a long-lived host stays bounded. ``pinned`` kernels
    (live delegate_task children) are exempt from the cap: their lifetime is the child's, ended by
    ``shutdown_kernels_for_delegated_child``, so the cap has nothing to bound for them."""
    cap, idle_timeout = _lifecycle_limits()
    with _REGISTRY.lock:
        expired = _pop_idle_expired(time.monotonic(), idle_timeout)
        kernel = _KERNELS.get(key)
        state_reset = kernel is not None and (reset or kernel.dead())
        if state_reset:
            dropped = _KERNELS.pop(key)
            if dropped.attached == 0:
                expired.append(dropped)
            kernel = None
        if kernel is None:
            kernel = _KERNELS[key] = SessionKernel(key)
            kernel.pinned = pinned
        kernel.last_used = time.monotonic()
        kernel.attached += 1
        unpinned = [k for k in _KERNELS if not _KERNELS[k].pinned]
        by_age = sorted((k for k in unpinned if k != key and _KERNELS[k].attached == 0),
                        key=lambda k: _KERNELS[k].last_used)
        expired.extend(_KERNELS.pop(k) for k in by_age[: max(0, len(unpinned) - cap)])
    for doomed in expired:
        doomed.teardown()
    return kernel, state_reset


# The acquire-path sweep above only fires on the NEXT kernel request. A host that stays
# alive but stops executing anything (a pids-exhausted container whose tool dispatch is
# fail-closed) never acquires again, so idle kernels and their thread pools survive
# indefinitely (#117169). One low-frequency daemon thread reapplies the same criteria on
# its own schedule, independent of tool traffic, and also sweeps staging dirs that
# outlived a host which died without cleanup (SIGKILL / container restart).
_STALE_STAGING_DIR_AGE = 7 * 86400
_REAPER_INTERVAL_FLOOR, _REAPER_INTERVAL_CEIL = 30.0, 300.0
_REAPER_STARTED = False


def _sweep_stale_staging_dirs(now: Optional[float] = None) -> int:
    """Remove ``hermes_kernel_*`` staging dirs untouched for over a week. A live host
    rmtrees each dir within one idle timeout of the kernel's last use, so a week-old
    dir belongs to a host that died before its cleanup could run; younger dirs are left
    alone because a concurrently running host's live kernel may own one. rmtree never
    follows symlinks, so a planted link is rejected rather than chased."""
    now = time.time() if now is None else now
    removed = 0
    for path in glob.glob(os.path.join(tempfile.gettempdir(), "hermes_kernel_*")):
        try:
            if now - os.path.getmtime(path) > _STALE_STAGING_DIR_AGE:
                # No ignore_errors: a rejected symlink (or a half-removed dir) must not
                # count as swept — it stays for the next pass instead.
                shutil.rmtree(path)
                removed += 1
        except OSError:
            continue
    return removed


def _reap_once() -> None:
    """One background pass: the acquire-path idle criteria, then the stale-dir sweep."""
    _, idle_timeout = _lifecycle_limits()
    with _REGISTRY.lock:
        expired = _pop_idle_expired(time.monotonic(), idle_timeout)
    for doomed in expired:
        doomed.teardown()
    _sweep_stale_staging_dirs()


def _ensure_background_reaper() -> None:
    """Start the reaper once per process (on the first kernel spawn)."""
    global _REAPER_STARTED
    with _REGISTRY.lock:
        if _REAPER_STARTED:
            return
        _REAPER_STARTED = True
    threading.Thread(target=_background_reaper, daemon=True,
                     name="hermes-kernel-idle-reaper").start()


def _background_reaper() -> None:
    while True:
        _, idle_timeout = _lifecycle_limits()
        time.sleep(min(_REAPER_INTERVAL_CEIL,
                       max(_REAPER_INTERVAL_FLOOR, idle_timeout / 6.0)))
        try:
            _reap_once()
        except Exception:
            logger.exception("kernel idle reaper pass failed; retrying next interval")


def _await_cell(kernel: SessionKernel, timeout: int, is_interrupted) -> Tuple[str, Dict[str, Any]]:
    """Wait for the cell's reply; returns (host status, payload)."""
    deadline = time.monotonic() + timeout if timeout else None
    while True:
        if is_interrupted():
            return "interrupted", {}
        if deadline is not None and time.monotonic() > deadline:
            return "timeout", {}
        try:
            payload = kernel.response_q.get(timeout=0.05)
        except queue.Empty:
            continue
        if payload.get("status") in ("kernel-eof", "protocol-error"):
            return "error", payload
        return "success", payload


def _with_stderr(stdout_text: str, stderr_text: str) -> str:
    return stdout_text + "\n--- stderr ---\n" + stderr_text


def _cell_result(kernel: SessionKernel, key: Tuple, status: str, payload: Dict[str, Any], *,
                 timeout: int, sandbox_tools: frozenset, reused: bool,
                 state_reset: bool, exec_start: float) -> Dict[str, Any]:
    """Assemble the tool result for one settled cell (disposing the kernel where the contract says so)."""
    from tools.code_execution_tool import _sandbox_failure_hint, _truncate_stdout_text
    from agent.redact import redact_sensitive_text
    from tools.ansi_strip import strip_ansi
    def clean(text: str) -> str:
        return redact_sensitive_text(strip_ansi(text), code_file=True)
    if status in ("timeout", "interrupted"):
        # No safe way to interrupt one cell in place: kill the kernel, report the loss, respawn next call.
        _REGISTRY.discard(key, kernel)
    duration = round(time.monotonic() - exec_start, 2)
    kernel.execution_count = int(payload.get("execution_count", kernel.execution_count + 1))
    stderr_raw = kernel.stderr.drain()
    stdout_text, stdout_metadata = _truncate_stdout_text(clean(str(payload.get("stdout", "")) + kernel.raw.drain()))
    cell_stderr = clean(str(payload.get("stderr", "")) + stderr_raw)
    cell_status = payload.get("status", "")
    result: Dict[str, Any] = {
        "status": status, "output": stdout_text, "exit_code": 0,
        "tool_calls_made": kernel.tool_call_counter[0], "duration_seconds": duration,
        "kernel": {"mode": "session", "reused": reused,
                   "execution_count": kernel.execution_count, "state_reset": state_reset},
    }
    result.update(stdout_metadata)
    # Cell-side spill (runner clipped before replying): same read_file recipe as the host-side spill.
    cell_spill = str(payload.get("stdout_spill_path", "") or "")
    if cell_spill and payload.get("stdout_clipped"):
        result["stdout_spill_path"] = cell_spill
        result["warning"] = (
            f"Cell stdout exceeded the inline cap; head shown. FULL output saved to {cell_spill} "
            f'— page it with read_file(path="{cell_spill}", offset=...) instead of re-running. '
            "(Kernel state persists: printing a narrower slice next call is often cheaper.)"
        )
    if status == "timeout":
        message = (f"Cell timed out after {timeout}s; the session kernel was killed and its "
                   "state was lost. The next execute_code call starts a fresh kernel.")
        result.update(exit_code=-1, error=message,
                      output=(stdout_text + "\n\n⏰ " + message) if stdout_text else ("⏰ " + message))
    elif status == "interrupted":
        from tools.code_execution_tool import _format_interrupted_output
        result.update(exit_code=-1, output=_format_interrupted_output(stdout_text),
                      error="Interrupted; the session kernel was killed and its state was lost.")
    elif cell_status == "error":
        trace = clean(str(payload.get("traceback", "")))
        result.update(status="error", exit_code=1, error=trace or "Cell raised an exception.",
                      output=_with_stderr(stdout_text, cell_stderr + trace) if (cell_stderr or trace) else stdout_text)
        hint = _sandbox_failure_hint(trace, enabled_tools=sandbox_tools)
        if hint:
            result["hint"] = hint
    elif cell_status == "exit":
        # The cell called sys.exit(): honor it as end-of-kernel.
        _REGISTRY.discard(key, kernel)
        result["kernel"]["ended"] = True
        if cell_stderr:
            result["output"] = _with_stderr(stdout_text, cell_stderr)
    elif status == "error":
        _REGISTRY.discard(key, kernel)
        result.update(exit_code=-1, error="The session kernel died while running the cell"
                      + (": " + stderr_raw.strip() if stderr_raw.strip() else "."))
    elif cell_stderr:
        result["output"] = _with_stderr(stdout_text, cell_stderr)
    return result


def execute_in_session_kernel(
    code: str, *, task_id: str, mode: str, child_python: str, child_cwd: str,
    sandbox_tools: frozenset, timeout: int, max_tool_calls: int, reset: bool, is_interrupted,
) -> str:
    """Run one cell in the (owner, mode, python, cwd, tools) session kernel. The owner is the
    session key (``_resolve_owner``), not the per-turn task id, so state survives across turns."""
    key = (_resolve_owner(task_id) or "", mode, child_python, child_cwd, tuple(sorted(sandbox_tools)))
    exec_start = time.monotonic()
    from agent.delegation_context import is_delegated_child_context
    kernel, state_reset = _acquire_kernel(key, reset, pinned=is_delegated_child_context())
    try:
        return _run_cell(kernel, key, code, task_id=task_id, child_python=child_python, child_cwd=child_cwd,
                         sandbox_tools=sandbox_tools, timeout=timeout, max_tool_calls=max_tool_calls,
                         is_interrupted=is_interrupted, exec_start=exec_start, state_reset=state_reset)
    finally:
        with _REGISTRY.lock:
            kernel.attached -= 1
            kernel.last_used = time.monotonic()
            # Dropped from the registry (reset/dead/reaped) while cells were still attached:
            # the last one out owns the teardown.
            orphaned = kernel.attached == 0 and _KERNELS.get(key) is not kernel
        if orphaned:
            kernel.teardown()


def _run_cell(kernel: SessionKernel, key: Tuple, code: str, *, task_id: str, child_python: str,
              child_cwd: str, sandbox_tools: frozenset, timeout: int, max_tool_calls: int,
              is_interrupted, exec_start: float, state_reset: bool) -> str:
    reused = kernel.proc is not None
    # Captured on the calling thread BEFORE the cell runs (the snapshot a per-call RPC thread
    # would get) and installed on the kernel so RPC dispatches under THIS cell's identity.
    authority = CellAuthority(task_id)
    with kernel.lock:
        try:
            if kernel.proc is None:
                _spawn(kernel, task_id=task_id, child_python=child_python, child_cwd=child_cwd,
                       sandbox_tools=sandbox_tools, max_tool_calls=max_tool_calls)
            assert kernel.proc is not None and kernel.proc.stdin is not None
            # Per-cell tool budget: the RPC loop enforces counter < max; reset without restarting.
            kernel.tool_call_counter[0] = 0
            kernel.raw.drain(), kernel.stderr.drain()  # raw output leaked between cells belongs to no cell
            kernel.cell_authority = authority
            kernel.proc.stdin.write((json.dumps({"id": uuid.uuid4().hex, "code": code}) + "\n").encode("utf-8"))
            kernel.proc.stdin.flush()
            status, payload = _await_cell(kernel, timeout, is_interrupted)
            result = _cell_result(
                kernel, key, status, payload,
                timeout=timeout, sandbox_tools=sandbox_tools, reused=reused,
                state_reset=state_reset, exec_start=exec_start,
            )
            return json.dumps(result, ensure_ascii=False)
        except Exception as exc:  # pragma: no cover - defensive parity with per-call
            from tools.code_execution_tool import _error_result
            logger.error("session kernel failed: %s: %s", type(exc).__name__, exc, exc_info=True)
            _REGISTRY.discard(key, kernel)
            return _error_result(str(exc), tool_calls_made=kernel.tool_call_counter[0],
                                 duration=round(time.monotonic() - exec_start, 2))
        finally:
            # The cell has settled on every path: its tool authority retires with it, so
            # nothing the cell left running can dispatch under it.
            authority.retire()
