"""Session-persistent Python kernels for execute_code.

With ``code_execution.kernel_mode: session``, execute_code keeps one Python
child process alive per (task, mode, interpreter, cwd, tool-set) and feeds it
one code cell per call, so variables, imports, and loaded data survive across
calls::

    execute_code(code="df = load_big_csv()")     # cell 1
    execute_code(code="print(df.describe())")    # cell 2 — df still exists

The default mode, ``per-call``, keeps today's behavior exactly: a fresh
process per call, no state carried over.

Design constraints, in order:

- **Same security envelope as per-call.** The child env is built by the same
  ``_build_child_env`` the per-call path uses (secret scrubbing, tool
  whitelist, PYTHONPATH rules); the RPC server is the same
  ``_rpc_server_loop`` with the same token and per-cell tool budget; output
  passes through the same ANSI strip + secret redaction. Nothing here widens
  what a script can reach — it only widens how long one interpreter lives.
- **A wedged kernel dies, never hangs the agent.** A cell that exceeds the
  timeout (or an interrupt) kills the whole kernel process tree and drops the
  registry entry; the next call spawns a fresh kernel. Losing kernel state on
  timeout is deliberate: there is no reliable way to interrupt one cell
  in-place without leaving the interpreter in an unknown state.
- **The env is frozen at spawn.** Skills that register env passthrough after
  the kernel started are not visible until ``reset=true`` (or the kernel is
  otherwise replaced). The result payload names the kernel so this is
  diagnosable.

Wire protocol (host <-> kernel child):

- Requests: one JSON object per line on the child's stdin:
  ``{"id": <str>, "code": <str>}``.
- Responses: framed on the child's stdout as
  ``<SENTINEL> <byte-length>\\n<json-payload>`` where SENTINEL carries a
  per-kernel random token from the environment. Bytes outside frames are
  raw fd-level output (subprocesses spawned by user code inherit the real
  stdout) and are attributed to the cell that was running when they arrived —
  calls are serialized per kernel, so attribution is unambiguous.
- Python-level stdout/stderr inside a cell are captured by the runner via
  ``contextlib.redirect_*`` and returned inside the JSON payload. A script
  that deliberately prints a forged frame can fake its own cell result; that
  is the same trust position as a per-call script printing a forged success
  message, and it gains nothing beyond lying to its own caller.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import secrets
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_IS_WINDOWS = sys.platform == "win32"

# Runner-side caps: bound captured python-level output before it ever reaches
# the host (the host applies its own MAX_STDOUT truncation again).
_RUNNER_CAPTURE_BYTES = 1_000_000

KERNEL_RUNNER_SOURCE = '''\
"""Auto-generated Hermes session-kernel runner. One exec cell per request."""
import contextlib
import io
import json
import os
import sys
import threading
import traceback

_SENTINEL = os.environ["HERMES_KERNEL_SENTINEL"]
_CAPTURE_LIMIT = {capture_limit}
_SPILL_DIR = os.environ.get("HERMES_KERNEL_SPILL_DIR", "")
_SPILL_CAP = {spill_cap}
_PARENT_PROCESS_HANDLE = os.environ.pop("HERMES_KERNEL_PARENT_PROCESS_HANDLE", "")
_PARENT_DEATH_FD = os.environ.pop("HERMES_KERNEL_PARENT_DEATH_FD", "")


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

# The persistent cell namespace. `__name__` is `__main__` so scripts behave
# like the per-call path; builtins resolve normally through exec.
GLOBALS = {{"__name__": "__main__", "__builtins__": __builtins__}}

_real_stdout = sys.stdout


def _bounded(text, spill_name=None):
    """Clip to the inline cap; spill the FULL text to disk when clipping.

    Returns (clipped_text, clipped?, spill_path_or_empty). Spill is
    best-effort — a failed write degrades to plain clipping.
    """
    if len(text) <= _CAPTURE_LIMIT:
        return text, False, ""
    spill_path = ""
    if _SPILL_DIR and spill_name:
        try:
            spill_path = os.path.join(_SPILL_DIR, spill_name)
            with open(spill_path, "w", encoding="utf-8", errors="replace") as f:
                f.write(text[:_SPILL_CAP])
                if len(text) > _SPILL_CAP:
                    f.write("\\n\\n[... spill capped ...]")
        except Exception:
            spill_path = ""
    return text[: _CAPTURE_LIMIT], True, spill_path


def _reply(payload):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    _real_stdout.buffer.write(
        ("\\n" + _SENTINEL + " " + str(len(body)) + "\\n").encode("utf-8")
    )
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
        out, err = io.StringIO(), io.StringIO()
        status = "ok"
        trace = ""
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                exec(compile(request["code"], "<cell>", "exec"), GLOBALS)
        except SystemExit as exc:
            status = "exit"
            trace = "SystemExit: " + repr(exc.code)
        except BaseException:
            status = "error"
            trace = traceback.format_exc()
        stdout_text, stdout_clipped, stdout_spill = _bounded(
            out.getvalue(), "cell_%06d_stdout.txt" % execution_count
        )
        stderr_text, stderr_clipped, _ = _bounded(err.getvalue())
        _reply(
            {{
                "id": request.get("id", ""),
                "status": status,
                "stdout": stdout_text,
                "stderr": stderr_text,
                "stdout_clipped": stdout_clipped,
                "stderr_clipped": stderr_clipped,
                "stdout_spill_path": stdout_spill,
                "traceback": trace,
                "execution_count": execution_count,
            }}
        )
        if status == "exit":
            break


if __name__ == "__main__":
    main()
'''.format(capture_limit=_RUNNER_CAPTURE_BYTES,
           spill_cap=5_000_000)


class CellAuthority:
    """The approval/context identity of exactly one execute_code cell.

    Interpreter state persists across cells; RPC authority must not. Each
    cell installs a fresh authority — captured from the CALLING thread at
    cell start, exactly what ``propagate_context_to_thread`` would have
    captured for a per-call RPC thread — and retires it when the cell
    settles, so a tool call arriving later (a background thread the cell
    left behind, a raced client write) is refused instead of running under
    a stale approval/session/turn identity.
    """

    def __init__(self, task_id: str):
        import contextvars

        self.task_id = task_id
        self.ctx = contextvars.copy_context()
        self.active = True
        self._approval_cb = None
        self._sudo_cb = None
        self._callback_setters = None
        try:
            from tools.thread_context import _callback_api

            get_approval, get_sudo, set_approval, set_sudo = _callback_api()
            self._approval_cb = get_approval()
            self._sudo_cb = get_sudo()
            self._callback_setters = (set_approval, set_sudo)
        except Exception:
            # Fail-closed, mirroring propagate_context_to_thread: with no
            # callbacks installed, dangerous approvals deny.
            self._callback_setters = None

    def retire(self) -> None:
        self.active = False

    def dispatch(self, tool_name: str, tool_args: dict) -> str:
        """Run one tool call under THIS cell's context and callbacks."""
        from tools.code_execution_tool import tool_error

        if not self.active:
            return tool_error(
                "No active execute_code cell: the cell this kernel call "
                "belonged to has settled, so its tool authority is retired."
            )
        return self.ctx.run(self._invoke, tool_name, tool_args)

    def _invoke(self, tool_name: str, tool_args: dict) -> str:
        from model_tools import handle_function_call

        previous = None
        if self._callback_setters is not None:
            try:
                from tools.thread_context import _callback_api

                get_approval, get_sudo, set_approval, set_sudo = _callback_api()
                previous = (get_approval(), get_sudo())
                set_approval(self._approval_cb)
                set_sudo(self._sudo_cb)
            except Exception:
                previous = None
        try:
            return handle_function_call(tool_name, tool_args, task_id=self.task_id)
        finally:
            if previous is not None and self._callback_setters is not None:
                set_approval, set_sudo = self._callback_setters
                try:
                    set_approval(previous[0])
                    set_sudo(previous[1])
                except Exception:
                    pass


class SessionKernel:
    """One live kernel process plus its RPC server and reader threads."""

    def __init__(self, key: Tuple):
        self.key = key
        self.owner: str = key[0]
        self.lock = threading.Lock()
        self.proc: Optional[subprocess.Popen] = None
        self.tmpdir: str = ""
        self.sock_path: Optional[str] = None
        self.server_sock: Optional[socket.socket] = None
        self.stop_event = threading.Event()
        self.rpc_token: str = ""
        self.sentinel: str = ""
        self.death_pipe_w: Optional[int] = None
        self.tool_call_log: List = []
        self.tool_call_counter: List[int] = [0]
        # Cells currently attached to this kernel (bumped under _KERNELS_LOCK
        # when a caller selects it, dropped when its cell settles). Reaping
        # and cap-eviction skip kernels with attached cells: tearing one down
        # mid-spawn rmtree'd the staging dir under the spawner
        # (FileNotFoundError) and killed live cells (Sep 2026).
        self.attached: int = 0
        self.response_q: "queue.Queue[dict]" = queue.Queue()
        self.raw_chunks: List[bytes] = []
        self.raw_bytes = [0]
        self.stderr_chunks: List[bytes] = []
        self.stderr_bytes = [0]
        self.execution_count = 0
        self.last_used: float = time.monotonic()
        self.cell_authority: Optional[CellAuthority] = None

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def dead(self) -> bool:
        """True only once a spawned process has exited.

        A kernel whose ``proc`` is still ``None`` is mid-spawn, not dead:
        parallel cells for one owner race the first cell's ``_spawn``, and
        treating the pending kernel as dead made every racer replace it,
        orphaning the winner's process outside the registry (110 live
        kernels under one 4-capped process, Sep 2026).
        """
        return self.proc is not None and self.proc.poll() is not None


_KERNELS: Dict[Tuple, SessionKernel] = {}
_KERNELS_LOCK = threading.Lock()

# Bounded lifecycle defaults (config: code_execution.max_session_kernels /
# code_execution.kernel_idle_timeout). A long-lived gateway must never
# accumulate one live child per finished conversation — the ownership,
# disposal, idle-reap, and cap shape here deliberately carries forward the
# lifecycle invariants of the earlier session-persistent implementation in
# hermes-agent#88637 by @z80dev (stable owner id, owner-teardown disposal,
# idle reaping, max-live bound).
DEFAULT_MAX_SESSION_KERNELS = 4
DEFAULT_KERNEL_IDLE_TIMEOUT = 1800


def _lifecycle_limits() -> Tuple[int, int]:
    from tools.code_execution_tool import _load_config

    config = _load_config()
    try:
        cap = int(config.get("max_session_kernels", DEFAULT_MAX_SESSION_KERNELS))
    except (TypeError, ValueError):
        cap = DEFAULT_MAX_SESSION_KERNELS
    try:
        idle = int(config.get("kernel_idle_timeout", DEFAULT_KERNEL_IDLE_TIMEOUT))
    except (TypeError, ValueError):
        idle = DEFAULT_KERNEL_IDLE_TIMEOUT
    return max(1, cap), max(1, idle)


def _resolve_owner(task_id: str) -> str:
    """The stable identity a session kernel belongs to.

    The conversation's approval session key — context-propagated, stable
    across turns of one conversation, and distinct per session. ``run_agent``
    mints a fresh task id per top-level turn, so a task-keyed kernel would
    neither survive the next user turn nor ever be torn down with anything;
    the task id is only the last-resort owner for embeds and tests that run
    with no session context at all.

    Delegated children run in a copy of the parent's context and therefore
    INHERIT the parent's approval session key — without the qualifier below,
    a child's execute_code would attach to the parent's kernel and read its
    in-memory state (verified live: parent-planted globals were readable
    from a delegated_child_context, both directions). Children get their own
    kernels, keyed by their delegation session id.
    """
    try:
        from tools.approval import get_current_session_key

        session_key = get_current_session_key(default="")
    except Exception:
        session_key = ""

    owner = session_key or (task_id or "")

    try:
        from agent.delegation_context import is_delegated_child_context

        if is_delegated_child_context():
            from gateway.session_context import get_session_env

            child_id = get_session_env("HERMES_SESSION_ID", "") or (task_id or "")
            owner = f"{owner}::child::{child_id}"
    except Exception:
        pass

    return owner


def _kernel_key(owner: str, mode: str, child_python: str, child_cwd: str,
                sandbox_tools: frozenset) -> Tuple:
    return (owner or "", mode, child_python, child_cwd, tuple(sorted(sandbox_tools)))


def shutdown_all_kernels() -> None:
    """Kill every session kernel. Registered via atexit; also used by tests."""
    with _KERNELS_LOCK:
        kernels = list(_KERNELS.values())
        _KERNELS.clear()
    for kernel in kernels:
        _teardown(kernel)


def shutdown_kernels_for_owner(owner: str) -> None:
    """Dispose every kernel a session owns.

    Wired into ``tools.approval.clear_session`` so kernels die at the same
    session boundary that clears the owner's approval and yolo state
    (the /new + session-close disposal shape from hermes-agent#88637).
    """
    if not owner:
        return
    with _KERNELS_LOCK:
        doomed = [key for key in _KERNELS if key[0] == owner]
        kernels = [_KERNELS.pop(key) for key in doomed]
    for kernel in kernels:
        _teardown(kernel)


def _reap_unlocked() -> List[SessionKernel]:
    """Pop idle-expired kernels; caller tears them down outside the lock."""
    _, idle_timeout = _lifecycle_limits()
    now = time.monotonic()
    doomed = [
        key
        for key, kernel in _KERNELS.items()
        if kernel.attached == 0 and now - kernel.last_used > idle_timeout
    ]
    return [_KERNELS.pop(key) for key in doomed]


def _evict_over_cap_unlocked(keep: Tuple) -> List[SessionKernel]:
    """Pop least-recently-used idle kernels beyond the process-wide cap."""
    cap, _ = _lifecycle_limits()
    if len(_KERNELS) <= cap:
        return []
    by_age = sorted(
        (key for key in _KERNELS if key != keep and _KERNELS[key].attached == 0),
        key=lambda key: _KERNELS[key].last_used,
    )
    doomed = by_age[: len(_KERNELS) - cap]
    return [_KERNELS.pop(key) for key in doomed]


atexit.register(shutdown_all_kernels)


def _teardown(kernel: SessionKernel) -> None:
    kernel.stop_event.set()
    if kernel.death_pipe_w is not None:
        try:
            os.close(kernel.death_pipe_w)
        except OSError:
            pass
        kernel.death_pipe_w = None
    if kernel.proc is not None and kernel.proc.poll() is None:
        from tools.code_execution_tool import _kill_process_group

        _kill_process_group(kernel.proc, escalate=True)
    if kernel.server_sock is not None:
        try:
            kernel.server_sock.close()
        except OSError:
            pass
        kernel.server_sock = None
    if kernel.sock_path:
        try:
            os.unlink(kernel.sock_path)
        except OSError:
            pass
    if kernel.tmpdir:
        import shutil

        shutil.rmtree(kernel.tmpdir, ignore_errors=True)


def _rpc_forever(kernel: SessionKernel, max_tool_calls: int,
                 sandbox_tools: frozenset) -> None:
    """Serve tool RPC for the kernel's whole life.

    ``_rpc_server_loop`` serves one connection and returns on disconnect or
    on its 300s idle timeout; a kernel legitimately sits idle longer than
    that between cells, so re-accept until the kernel is torn down. The
    client stub reconnects on its side (HERMES_RPC_PERSISTENT).

    The serving thread carries NO frozen authority of its own: every
    dispatch is routed through the CURRENT cell's ``CellAuthority``, so a
    later cell's tool calls run under that cell's approval/session/turn
    context instead of whatever the first cell happened to capture.
    Interpreter state persists; RPC authority does not.
    """
    from tools.code_execution_tool import _rpc_server_loop, tool_error

    def _dispatch(tool_name: str, tool_args: dict) -> str:
        authority = kernel.cell_authority
        if authority is None:
            return tool_error(
                "No active execute_code cell: this kernel has no cell "
                "authority installed."
            )
        return authority.dispatch(tool_name, tool_args)

    while not kernel.stop_event.is_set():
        _rpc_server_loop(
            kernel.server_sock,
            "",
            kernel.tool_call_log,
            kernel.tool_call_counter,
            max_tool_calls,
            sandbox_tools,
            kernel.stop_event,
            kernel.rpc_token,
            dispatch=_dispatch,
        )


def _append_bounded(chunks: List[bytes], total: List[int], data: bytes, cap: int) -> None:
    if total[0] >= cap:
        return
    keep = data[: cap - total[0]]
    chunks.append(keep)
    total[0] += len(keep)


def _stdout_reader(kernel: SessionKernel) -> None:
    """Split the child's stdout into protocol frames and raw passthrough."""
    from tools.code_execution_tool import MAX_STDOUT_BYTES

    assert kernel.proc is not None and kernel.proc.stdout is not None
    stream = kernel.proc.stdout
    marker = ("\n" + kernel.sentinel + " ").encode("utf-8")
    buf = b""
    while True:
        # read1: return as soon as any bytes arrive. A plain read(n) on a
        # BufferedReader blocks until n bytes or EOF, which would sit on a
        # complete frame smaller than the buffer forever.
        chunk = stream.read1(4096)
        if not chunk:
            if buf:
                _append_bounded(kernel.raw_chunks, kernel.raw_bytes, buf, MAX_STDOUT_BYTES)
            kernel.response_q.put({"status": "kernel-eof"})
            return
        buf += chunk
        while True:
            index = buf.find(marker)
            if index < 0:
                # Keep a marker-sized tail in case the marker is split
                # across reads; everything before it is raw output.
                spill = buf[: -len(marker)] if len(buf) > len(marker) else b""
                if spill:
                    _append_bounded(kernel.raw_chunks, kernel.raw_bytes, spill, MAX_STDOUT_BYTES)
                    buf = buf[len(spill):]
                break
            if index:
                _append_bounded(kernel.raw_chunks, kernel.raw_bytes, buf[:index], MAX_STDOUT_BYTES)
            rest = buf[index + len(marker):]
            newline = rest.find(b"\n")
            if newline < 0:
                buf = buf[index:]
                break
            try:
                length = int(rest[:newline])
            except ValueError:
                # Not a real frame header (user output that happens to
                # contain the marker bytes); treat the marker as raw.
                _append_bounded(kernel.raw_chunks, kernel.raw_bytes, marker, MAX_STDOUT_BYTES)
                buf = rest
                continue
            body = rest[newline + 1:]
            missing = length - len(body)
            while missing > 0:
                more = stream.read1(missing)
                if not more:
                    kernel.response_q.put({"status": "kernel-eof"})
                    return
                body += more
                missing -= len(more)
            try:
                kernel.response_q.put(json.loads(body[:length].decode("utf-8", errors="replace")))
            except ValueError:
                kernel.response_q.put({"status": "protocol-error"})
            buf = body[length:]


def _stderr_reader(kernel: SessionKernel) -> None:
    from tools.code_execution_tool import MAX_STDERR_BYTES

    assert kernel.proc is not None and kernel.proc.stderr is not None
    while True:
        chunk = kernel.proc.stderr.read1(4096)
        if not chunk:
            return
        _append_bounded(kernel.stderr_chunks, kernel.stderr_bytes, chunk, MAX_STDERR_BYTES)


def _spawn(kernel: SessionKernel, *, task_id: str, child_python: str,
           child_cwd: str, sandbox_tools: frozenset, max_tool_calls: int) -> None:
    from tools.code_execution_tool import (
        _build_child_env,
        generate_hermes_tools_module,
    )

    kernel.tmpdir = tempfile.mkdtemp(prefix="hermes_kernel_")
    _sock_tmpdir = "/tmp" if sys.platform == "darwin" else tempfile.gettempdir()

    kernel.rpc_token = secrets.token_urlsafe(32)
    kernel.sentinel = "@@HERMES-KERNEL-" + secrets.token_urlsafe(16) + "@@"

    if _IS_WINDOWS:
        kernel.sock_path = None
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.bind(("127.0.0.1", 0))
        host, port = server_sock.getsockname()[:2]
        rpc_endpoint = f"tcp://{host}:{port}"
    else:
        kernel.sock_path = os.path.join(_sock_tmpdir, f"hermes_rpc_{uuid.uuid4().hex}.sock")
        server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server_sock.bind(kernel.sock_path)
        os.chmod(kernel.sock_path, 0o600)
        rpc_endpoint = kernel.sock_path
    server_sock.listen(1)
    kernel.server_sock = server_sock

    tools_src = generate_hermes_tools_module(list(sandbox_tools))
    with open(os.path.join(kernel.tmpdir, "hermes_tools.py"), "w", encoding="utf-8") as f:
        f.write(tools_src)
    runner_path = os.path.join(kernel.tmpdir, "hermes_kernel_runner.py")
    with open(runner_path, "w", encoding="utf-8") as f:
        f.write(KERNEL_RUNNER_SOURCE)

    child_env = _build_child_env(
        rpc_endpoint=rpc_endpoint,
        rpc_token=kernel.rpc_token,
        tmpdir=kernel.tmpdir,
        child_python=child_python,
    )
    child_env["HERMES_KERNEL_SENTINEL"] = kernel.sentinel
    # Cells clip stdout to the inline cap; the full text spills to the
    # kernel's own tmpdir so the agent can read_file the middle instead of
    # re-running (host surfaces the path in the result).
    child_env["HERMES_KERNEL_SPILL_DIR"] = kernel.tmpdir
    # Tell the generated client to reconnect after the RPC server's idle
    # timeout — a kernel outlives the 300s window between cells.
    child_env["HERMES_RPC_PERSISTENT"] = "1"

    parent_process_handle = None
    close_parent_process_handle = None
    startupinfo = None
    if _IS_WINDOWS:
        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.GetCurrentProcessId.argtypes = []
            kernel32.GetCurrentProcessId.restype = wintypes.DWORD
            kernel32.OpenProcess.argtypes = [
                wintypes.DWORD,
                wintypes.BOOL,
                wintypes.DWORD,
            ]
            kernel32.OpenProcess.restype = wintypes.HANDLE
            kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel32.CloseHandle.restype = wintypes.BOOL
            close_parent_process_handle = kernel32.CloseHandle
            parent_process_handle = kernel32.OpenProcess(
                0x00100000,  # SYNCHRONIZE
                True,  # inherited only by the explicitly allow-listed child
                kernel32.GetCurrentProcessId(),
            )
            if parent_process_handle:
                child_env["HERMES_KERNEL_PARENT_PROCESS_HANDLE"] = str(
                    int(parent_process_handle)
                )
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.lpAttributeList = {
                    "handle_list": [int(parent_process_handle)]
                }
        except (AttributeError, ImportError, OSError, TypeError, ValueError):
            if parent_process_handle and close_parent_process_handle is not None:
                close_parent_process_handle(parent_process_handle)
            child_env.pop("HERMES_KERNEL_PARENT_PROCESS_HANDLE", None)
            parent_process_handle = None
            close_parent_process_handle = None
            startupinfo = None

    death_r: Optional[int] = None
    pass_fds: Tuple[int, ...] = ()
    if not _IS_WINDOWS:
        death_r, kernel.death_pipe_w = os.pipe()
        child_env["HERMES_KERNEL_PARENT_DEATH_FD"] = str(death_r)
        pass_fds = (death_r,)

    try:
        kernel.proc = subprocess.Popen(
            [child_python, runner_path],
            # Strict mode resolves an empty cwd: the kernel's own staging dir
            # then plays the per-call tmpdir's role.
            cwd=child_cwd or kernel.tmpdir,
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
            start_new_session=True,
            creationflags=subprocess.CREATE_NO_WINDOW if _IS_WINDOWS else 0,
            close_fds=True,
            pass_fds=pass_fds,
            startupinfo=startupinfo,
        )
    finally:
        if parent_process_handle and close_parent_process_handle is not None:
            close_parent_process_handle(parent_process_handle)
        if death_r is not None:
            os.close(death_r)

    # Deliberately NOT propagate_context_to_thread: that would freeze the
    # spawning cell's context/callbacks into the server thread for the
    # kernel's whole life. Authority is rebound per cell via CellAuthority.
    threading.Thread(
        target=_rpc_forever,
        args=(kernel, max_tool_calls, sandbox_tools),
        daemon=True,
    ).start()
    threading.Thread(target=_stdout_reader, args=(kernel,), daemon=True).start()
    threading.Thread(target=_stderr_reader, args=(kernel,), daemon=True).start()


def _drain_raw(kernel: SessionKernel) -> str:
    chunks, kernel.raw_chunks, kernel.raw_bytes = kernel.raw_chunks, [], [0]
    return b"".join(chunks).decode("utf-8", errors="replace")


def _drain_stderr(kernel: SessionKernel) -> str:
    chunks, kernel.stderr_chunks, kernel.stderr_bytes = kernel.stderr_chunks, [], [0]
    return b"".join(chunks).decode("utf-8", errors="replace")


def execute_in_session_kernel(
    code: str,
    *,
    task_id: str,
    mode: str,
    child_python: str,
    child_cwd: str,
    sandbox_tools: frozenset,
    timeout: int,
    max_tool_calls: int,
    reset: bool,
    is_interrupted,
) -> str:
    """Run one cell in the (owner, mode, python, cwd, tools) session kernel.

    The owner is the conversation's session key (``_resolve_owner``), not
    the per-turn task id, so state genuinely survives across user turns of
    one conversation and dies with the session. Every entry also sweeps
    idle-expired kernels and enforces the process-wide cap, so a long-lived
    host stays bounded even for owners that never toggle or reset.
    """
    from tools.code_execution_tool import (
        _sandbox_failure_hint,
        _truncate_stdout_text,
    )
    from agent.redact import redact_sensitive_text
    from tools.ansi_strip import strip_ansi

    owner = _resolve_owner(task_id)
    key = _kernel_key(owner, mode, child_python, child_cwd, sandbox_tools)
    exec_start = time.monotonic()
    state_reset = False

    with _KERNELS_LOCK:
        expired = _reap_unlocked()
        kernel = _KERNELS.get(key)
        if kernel is not None and (reset or kernel.dead()):
            _KERNELS.pop(key, None)
            if kernel.attached == 0:
                expired.append(kernel)
            kernel = None
            state_reset = True
        if kernel is None:
            kernel = SessionKernel(key)
            _KERNELS[key] = kernel
        kernel.last_used = time.monotonic()
        kernel.attached += 1
        expired.extend(_evict_over_cap_unlocked(keep=key))
    for doomed in expired:
        _teardown(doomed)
    try:
        return _run_cell(
            kernel, key, code, task_id=task_id, child_python=child_python,
            child_cwd=child_cwd, sandbox_tools=sandbox_tools, timeout=timeout,
            max_tool_calls=max_tool_calls, is_interrupted=is_interrupted,
            exec_start=exec_start, state_reset=state_reset,
        )
    finally:
        with _KERNELS_LOCK:
            kernel.attached -= 1
            kernel.last_used = time.monotonic()
            # Dropped from the registry (reset/dead/reaped) while cells were
            # still attached: the last one out owns the teardown.
            orphaned = kernel.attached == 0 and _KERNELS.get(key) is not kernel
        if orphaned:
            _teardown(kernel)


def _run_cell(
    kernel: SessionKernel,
    key: Tuple,
    code: str,
    *,
    task_id: str,
    child_python: str,
    child_cwd: str,
    sandbox_tools: frozenset,
    timeout: int,
    max_tool_calls: int,
    is_interrupted,
    exec_start: float,
    state_reset: bool,
) -> str:
    from tools.code_execution_tool import (
        _sandbox_failure_hint,
        _truncate_stdout_text,
    )
    from agent.redact import redact_sensitive_text
    from tools.ansi_strip import strip_ansi

    reused = kernel.proc is not None

    # Captured on the calling thread BEFORE the cell runs — the same
    # snapshot a per-call RPC thread would have received — and installed
    # atomically on the kernel so the serving thread dispatches this cell's
    # tool calls under this cell's approval/session/turn identity.
    authority = CellAuthority(task_id)

    with kernel.lock:
        try:
            if kernel.proc is None:
                _spawn(
                    kernel,
                    task_id=task_id,
                    child_python=child_python,
                    child_cwd=child_cwd,
                    sandbox_tools=sandbox_tools,
                    max_tool_calls=max_tool_calls,
                )
            assert kernel.proc is not None and kernel.proc.stdin is not None

            # Per-cell tool budget: the RPC loop enforces counter < max, so a
            # fresh cell starts from zero without restarting the server.
            kernel.tool_call_counter[0] = 0
            # Anything raw that leaked between cells belongs to no cell.
            _drain_raw(kernel)
            _drain_stderr(kernel)
            kernel.cell_authority = authority

            request = json.dumps({"id": uuid.uuid4().hex, "code": code}) + "\n"
            kernel.proc.stdin.write(request.encode("utf-8"))
            kernel.proc.stdin.flush()

            deadline = time.monotonic() + timeout if timeout else None
            status = "success"
            payload: Dict[str, Any] = {}
            while True:
                if is_interrupted():
                    status = "interrupted"
                    break
                if deadline is not None and time.monotonic() > deadline:
                    status = "timeout"
                    break
                try:
                    payload = kernel.response_q.get(timeout=0.05)
                except queue.Empty:
                    continue
                if payload.get("status") in ("kernel-eof", "protocol-error"):
                    status = "error"
                break

            if status in ("timeout", "interrupted"):
                # No safe way to interrupt one cell in place: kill the kernel,
                # report the state loss, let the next call respawn.
                with _KERNELS_LOCK:
                    if _KERNELS.get(key) is kernel:
                        _KERNELS.pop(key, None)
                _teardown(kernel)

            duration = round(time.monotonic() - exec_start, 2)
            kernel.execution_count = int(payload.get("execution_count", kernel.execution_count + 1))

            raw_text = _drain_raw(kernel)
            stderr_raw = _drain_stderr(kernel)
            stdout_text = str(payload.get("stdout", ""))
            if raw_text:
                stdout_text = stdout_text + raw_text
            cell_stderr = str(payload.get("stderr", ""))
            if stderr_raw:
                cell_stderr = cell_stderr + stderr_raw

            stdout_text = redact_sensitive_text(strip_ansi(stdout_text), code_file=True)
            cell_stderr = redact_sensitive_text(strip_ansi(cell_stderr), code_file=True)
            stdout_text, stdout_metadata = _truncate_stdout_text(stdout_text)

            cell_status = payload.get("status", "")
            result: Dict[str, Any] = {
                "status": status,
                "output": stdout_text,
                "exit_code": 0,
                "tool_calls_made": kernel.tool_call_counter[0],
                "duration_seconds": duration,
                "kernel": {
                    "mode": "session",
                    "reused": reused,
                    "execution_count": kernel.execution_count,
                    "state_reset": state_reset,
                },
            }
            result.update(stdout_metadata)

            # Cell-side spill (runner clipped before replying): surface the
            # full-output path with the same read_file recipe as the
            # host-side spill in _truncate_stdout_text.
            cell_spill = str(payload.get("stdout_spill_path", "") or "")
            if cell_spill and payload.get("stdout_clipped"):
                result["stdout_spill_path"] = cell_spill
                result["warning"] = (
                    "Cell stdout exceeded the inline cap; head shown. FULL "
                    f"output saved to {cell_spill} — page it with "
                    f'read_file(path="{cell_spill}", offset=...) instead of '
                    "re-running. (Kernel state persists: printing a narrower "
                    "slice next call is often cheaper.)"
                )

            if status == "timeout":
                message = (
                    f"Cell timed out after {timeout}s; the session kernel was "
                    "killed and its state was lost. The next execute_code call "
                    "starts a fresh kernel."
                )
                result["exit_code"] = -1
                result["error"] = message
                result["output"] = (stdout_text + "\n\n⏰ " + message) if stdout_text else ("⏰ " + message)
            elif status == "interrupted":
                from tools.code_execution_tool import _format_interrupted_output

                result["exit_code"] = -1
                result["output"] = _format_interrupted_output(stdout_text)
                result["error"] = "Interrupted; the session kernel was killed and its state was lost."
            elif cell_status == "error":
                trace = redact_sensitive_text(strip_ansi(str(payload.get("traceback", ""))), code_file=True)
                result["status"] = "error"
                result["exit_code"] = 1
                result["error"] = trace or "Cell raised an exception."
                joined = stdout_text
                if cell_stderr or trace:
                    joined = joined + "\n--- stderr ---\n" + cell_stderr + trace
                result["output"] = joined
                hint = _sandbox_failure_hint(trace, enabled_tools=sandbox_tools)
                if hint:
                    result["hint"] = hint
            elif cell_status == "exit":
                # The cell called sys.exit(): honor it as end-of-kernel.
                with _KERNELS_LOCK:
                    if _KERNELS.get(key) is kernel:
                        _KERNELS.pop(key, None)
                _teardown(kernel)
                result["kernel"]["ended"] = True
                if cell_stderr:
                    result["output"] = stdout_text + "\n--- stderr ---\n" + cell_stderr
            elif status == "error":
                result["exit_code"] = -1
                result["error"] = (
                    "The session kernel died while running the cell"
                    + (": " + stderr_raw.strip() if stderr_raw.strip() else ".")
                )
                with _KERNELS_LOCK:
                    if _KERNELS.get(key) is kernel:
                        _KERNELS.pop(key, None)
                _teardown(kernel)
            elif cell_stderr:
                result["output"] = stdout_text + "\n--- stderr ---\n" + cell_stderr

            return json.dumps(result, ensure_ascii=False)
        except Exception as exc:  # pragma: no cover - defensive parity with per-call
            logger.error("session kernel failed: %s: %s", type(exc).__name__, exc, exc_info=True)
            with _KERNELS_LOCK:
                if _KERNELS.get(key) is kernel:
                    _KERNELS.pop(key, None)
            _teardown(kernel)
            return json.dumps({
                "status": "error",
                "error": str(exc),
                "tool_calls_made": kernel.tool_call_counter[0],
                "duration_seconds": round(time.monotonic() - exec_start, 2),
            }, ensure_ascii=False)
        finally:
            # The cell has settled on every path (success, exception,
            # timeout, exit, kernel death): its tool authority retires with
            # it, so nothing the cell left running can dispatch under it.
            authority.retire()
