"""Supervisor for the dashboard compute-host child: with ``dashboard.turn_isolation``
agent turns run in one persistent ``python -m tui_gateway.compute_host`` child so heavy
agent threads do not contend with the serving process' event loop for the GIL."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home
from hermes_cli._startup_fast import project_root_str
from tools.environments.local import hermes_subprocess_env

logger = logging.getLogger(__name__)

MUTATOR_ROUTE_TABLE: dict[str, str] = {
    "prompt.submit": "turn-path", "session.interrupt": "turn-path", "reload.mcp": "run-concurrent",
    "session.save": "run-concurrent", "session.compress": "idle-gated",
    "prompt.submit.truncate": "idle-gated", "slash.model": "idle-gated",
    "slash.personality": "idle-gated", "slash.prompt": "idle-gated", "slash.compress": "idle-gated",
    "slash.refine": "idle-gated",
    "session.reset": "idle-gated", "session.history.reload": "idle-gated",
    "slash.retry": "idle-gated"}

_REGISTRY_NAME = "dashboard-compute-host.json"
_RESPAWN_WINDOW_SECS = 300.0
_SHUTDOWN_TIMEOUT_SECS = 10.0
# A cold host imports the agent stack before its hello (~1.3s idle, 5s+ on a loaded box).
_HELLO_TIMEOUT_SECS = 10.0
# An exited host's last frames (an ack written just before it exited) are read before its waiters
# fail; bounded because a grandchild can hold the stdout pipe open.
_EXIT_DRAIN_SECS = 2.0
# Late control-ack handlers: a compress that outlives its RPC waiter can run for the full
# compression ceiling plus a stall-fallback retry, so keep registrations past that — bounded.
# See #97948.
_LATE_CONTROL_TTL_SECS = 1800.0
_LATE_CONTROL_MAX = 64
# Host frames whose ``request_id`` resolves a pending/late control waiter.
_CONTROL_REPLY_TYPES = frozenset({
    "control.ack", "control.error", "respond.ack", "respond.error", "interrupt.ack",
    "reload_mcp.ack", "shutdown.ack"})


def append_log_record(path: str | Path, record: str) -> None:
    """Append one log record using O_APPEND and exactly one os.write call."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    text = record if record.endswith("\n") else f"{record}\n"
    if os.name == "nt":
        # CRT O_APPEND is seek+write, not an atomic append across handles.
        # FILE_APPEND_DATA without FILE_WRITE_DATA makes the kernel append.
        import win32con
        import win32file
        from ntsecuritycon import FILE_APPEND_DATA
        handle = win32file.CreateFile(
            str(path), FILE_APPEND_DATA,
            win32con.FILE_SHARE_READ | win32con.FILE_SHARE_WRITE | win32con.FILE_SHARE_DELETE,
            None, win32con.OPEN_ALWAYS, win32con.FILE_ATTRIBUTE_NORMAL, None,
        )
        try:
            win32file.WriteFile(handle, text.encode("utf-8", errors="replace"))
        finally:
            handle.Close()
        return
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, text.encode("utf-8", errors="replace"))
    finally:
        os.close(fd)


def _repo_root() -> Path:
    return Path(project_root_str())


def _check_output(argv: list[str], **kwargs: Any) -> str:
    """Stripped stdout of a short subprocess, or ``""`` on any failure."""
    with contextlib.suppress(Exception):
        return subprocess.check_output(
            argv, stdin=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace", stderr=subprocess.DEVNULL,
            timeout=2, **kwargs).strip()
    return ""


def _build_sha() -> str:
    """HEAD sha or ``"unknown"``; shared with ``compute_host`` so the hello handshake agrees."""
    return _check_output(["git", "rev-parse", "HEAD"], cwd=str(_repo_root())) or "unknown"


def _call_logged(cb: Callable[[dict], None], frame: dict, failure: str) -> None:
    """Invoke a host-frame callback; a raising callback is logged, never propagated."""
    try:
        cb(frame)
    except Exception:
        logger.exception(failure)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except Exception as exc:
        return isinstance(exc, PermissionError)


def _signal_pid(pid: int, sig: int, label: str) -> bool:
    """Send ``sig``; False when the pid is gone or the signal failed (logged)."""
    try:
        os.kill(pid, sig)
        return True
    except ProcessLookupError:
        return False
    except Exception:
        logger.debug("failed to %s compute host pid=%s", label, pid, exc_info=True)
        return False


def _pid_command(pid: int) -> str:
    if pid <= 0:
        return ""
    with contextlib.suppress(Exception):  # Linux fast path
        data = (Path("/proc") / str(pid) / "cmdline").read_bytes()
        if data:
            return data.replace(b"\x00", b" ").decode("utf-8", errors="replace")
    return _check_output(["ps", "-p", str(pid), "-o", "command="])


def is_compute_host_identity(pid: int) -> bool:
    return "tui_gateway.compute_host" in _pid_command(pid)


class HostSupervisor:
    """Own one persistent compute-host child and relay its frames."""

    def __init__(
        self, *, registry_path: str | Path | None = None, argv: list[str] | None = None,
        cwd: str | Path | None = None, env: dict[str, str] | None = None,
        rpc_sink: Callable[[dict], None] | None = None, respawn_max: int = 3,
        heartbeat_secs: int = 15, expected_build_sha: str | None = None,
        expected_hermes_home: str | None = None, autostart: bool = True) -> None:
        self.registry_path = (
            Path(registry_path) if registry_path is not None
            else get_hermes_home() / "state" / _REGISTRY_NAME)
        self.argv = argv or [sys.executable, "-m", "tui_gateway.compute_host"]
        self.cwd = Path(cwd) if cwd is not None else _repo_root()
        self.env = env
        self.rpc_sink = rpc_sink or (lambda _obj: None)
        self.respawn_max = max(0, int(respawn_max))
        self.heartbeat_secs = max(1, int(heartbeat_secs))
        self.expected_build_sha = _build_sha() if expected_build_sha is None else expected_build_sha
        self.expected_hermes_home = (
            str(get_hermes_home()) if expected_hermes_home is None else expected_hermes_home)
        self._lock = threading.RLock()
        self._proc: subprocess.Popen[str] | None = None
        self._hello_event = threading.Event()
        self._hello: dict[str, Any] = {}
        self._closing = False
        self._stopped_respawning = False
        self._restart_times: list[float] = []
        self._pending_turns: dict[str, tuple[str, Callable[[dict], None] | None]] = {}
        # request_id -> (reply queue, the error frame type that answers it if the host exits).
        self._pending_controls: dict[str, tuple[queue.Queue[dict], str]] = {}
        # request_id -> (registered_at, handler) for control waiters that timed out while their
        # host work still runs, so the eventual control.ack is not silently dropped.
        # The host emits its control.ack whenever it finishes; without this the ack matched no queue and was
        # silently dropped. See #97948.
        self._late_control_handlers: dict[str, tuple[float, Callable[[dict], None]]] = {}
        self._stderr_tail: list[str] = []
        self._last_progress_counter = 0
        if autostart:
            self.start()

    @property
    def pid(self) -> int:
        proc = self._proc
        return int(proc.pid or 0) if proc is not None else 0

    def is_running(self) -> bool:
        proc = self._proc
        return proc is not None and proc.poll() is None and not self._stopped_respawning

    def start(self) -> None:
        with self._lock:
            if self.is_running():
                return
            self._closing = False
            self.reconcile_startup_orphan()
            self._spawn_locked(reason="startup")

    def shutdown(self) -> None:
        with self._lock:
            self._closing = True
            proc = self._proc
        if proc is None:
            return
        try:
            if proc.poll() is None and proc.stdin is not None:
                self._send_frame({"type": "shutdown", "request_id": f"shutdown-{uuid.uuid4().hex}"})
                proc.wait(timeout=_SHUTDOWN_TIMEOUT_SECS)
        except Exception:
            self._terminate_process(proc)
        finally:
            self._remove_registry()

    def reconcile_startup_orphan(self) -> str:
        """Terminate a stale registered host, guarding against PID reuse."""
        try:
            data = json.loads(self.registry_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return "none"
        except Exception:
            data = None
        try:
            pid = int((data or {}).get("host_pid") or 0)
        except Exception:
            pid = 0
        if data is None:
            outcome = "invalid-registry"
        elif pid <= 0 or not _pid_alive(pid):
            outcome = "not-running"
        elif not self._pid_matches_compute_host(pid):
            outcome = "pid-reuse-ignored"  # PID reused by another process: never signal it
        else:
            self._terminate_pid(pid, timeout=_SHUTDOWN_TIMEOUT_SECS)
            outcome = "terminated"
        self._remove_registry()
        return outcome

    def submit_turn(self, frame: dict[str, Any], *, on_complete: Callable[[dict], None] | None = None) -> str:
        self.start()
        request_id = str(frame.get("request_id") or uuid.uuid4().hex)
        sid = str(frame.get("sid") or "")
        payload = {**frame, "type": "turn.start", "request_id": request_id}
        with self._lock:
            self._pending_turns[request_id] = (sid, on_complete)
        try:
            self._send_frame(payload)
        except Exception as exc:
            with self._lock:
                self._pending_turns.pop(request_id, None)
            if on_complete is not None:
                on_complete({"type": "turn.error", "sid": sid, "request_id": request_id,
                             "reason": "send_failed", "message": str(exc)})
            raise
        return request_id

    def interrupt(self, sid: str, *, request_id: str | None = None) -> None:
        self.start()
        self._send_frame(
            {"type": "interrupt", "sid": sid, "request_id": request_id or uuid.uuid4().hex})

    def has_pending_turn(self, sid: str) -> bool:
        """True when a host completion callback is still registered for ``sid``.

        Read by the serving process to tell "the host still owes this session a
        terminal frame" from "nothing will ever call back for it". Entries are
        added in :meth:`submit_turn` and removed by ``turn.end``/``turn.error``
        handling or :meth:`_fail_pending_turns`, all under ``self._lock``.
        """
        with self._lock:
            return any(s == sid for s, _cb in self._pending_turns.values())

    def _await_reply(self, frame: dict[str, Any], request_id: str, timeout: float,
                     on_late_ack: Callable[[dict], None] | None = None) -> dict:
        """Send ``frame`` and block for the host reply carrying ``request_id``. On timeout ``on_late_ack``
        takes over in the same locked step that retires the waiter's queue, so a reply landing at any moment
        reaches exactly one of them (a dropped ack would leak whatever the handler releases)."""
        q: queue.Queue[dict] = queue.Queue(maxsize=1)
        try:
            # One locked step with the send: a crash cleanup can't detach this waiter before its frame
            # reaches the host it was registered for.
            with self._lock:
                self._pending_controls[request_id] = (q, "respond.error" if frame["type"] == "respond" else "control.error")
                self._send_frame(frame)
            return q.get(timeout=timeout)
        except queue.Empty:
            dropped: list = []
            with self._lock:
                self._pending_controls.pop(request_id, None)
                if not q.empty():  # landed after the wait gave up, before the queue was retired
                    return q.get_nowait()
                if on_late_ack is not None:
                    dropped = self._register_late_control_handler_locked(request_id, on_late_ack)
            self._fire_dropped_late_handlers(dropped)
            raise
        finally:
            with self._lock:
                self._pending_controls.pop(request_id, None)

    def respond(self, sid: str, params: dict[str, Any], *, timeout: float = 15.0) -> dict:
        """Deliver an interactive prompt response to the host that owns it."""
        self.start()
        request_id = uuid.uuid4().hex
        frame = {"type": "respond", "sid": sid, "request_id": request_id, "params": dict(params)}
        return self._await_reply(frame, request_id, timeout)

    def reload_mcp(self, sid: str, *, request_id: str | None = None) -> dict:
        payload = {"type": "reload_mcp", "sid": sid, "request_id": request_id or uuid.uuid4().hex}
        return self.control(sid, route_name="reload.mcp", wait=True, payload=payload)

    def control(
        self, sid: str, *, route_name: str, payload: dict[str, Any] | None = None,
        wait: bool = True, timeout: float = 30.0, on_late_ack: Callable[[dict], None] | None = None,
    ) -> dict:
        """Send a control frame; with ``wait`` block up to ``timeout`` for its ack. ``on_late_ack``
        (only with ``wait``) keeps the request adoptable after the waiter gives up: the host's
        eventual ``control.ack``/``control.error``/``error`` fires it once (bounded by
        ``_LATE_CONTROL_TTL_SECS``/``_MAX``) instead of being dropped."""
        if route_name not in MUTATOR_ROUTE_TABLE:
            raise ValueError(f"unclassified host mutator route: {route_name}")
        self.start()
        payload = payload or {}
        request_id = str(payload.get("request_id") or uuid.uuid4().hex)
        frame = {"type": "control", **payload, "sid": sid, "route_name": route_name,
                 "request_id": request_id}
        if not wait:
            self._send_frame(frame)
            return {"status": "sent", "request_id": request_id}
        return self._await_reply(frame, request_id, timeout, on_late_ack)

    def _register_late_control_handler(self, request_id: str, handler: Callable[[dict], None]) -> None:
        with self._lock:
            dropped = self._register_late_control_handler_locked(request_id, handler)
        self._fire_dropped_late_handlers(dropped)

    def _register_late_control_handler_locked(self, request_id: str, handler: Callable[[dict], None]) -> list:
        """Register under ``self._lock``; returns the handlers the TTL or the cap dropped, for the caller to fire
        outside the lock (``_fire_dropped_late_handlers``)."""
        now = time.monotonic()
        dropped = []
        handlers = self._late_control_handlers
        for rid in [r for r, (at, _cb) in handlers.items() if now - at > _LATE_CONTROL_TTL_SECS]:
            dropped.append((rid, handlers.pop(rid)[1]))
        while len(handlers) >= _LATE_CONTROL_MAX:
            rid = min(handlers, key=lambda r: handlers[r][0])
            dropped.append((rid, handlers.pop(rid)[1]))
        handlers[request_id] = (now, handler)
        return dropped

    @staticmethod
    def _fire_dropped_late_handlers(dropped: list) -> None:
        """A dropped handler still fires once, with a ``control.error``: it may own a release (a transcript
        claim) that nothing else would run."""
        for rid, cb in dropped:
            frame = {"type": "control.error", "request_id": rid,
                     "message": "compute host reply abandoned: no answer within the late-ack window"}
            _call_logged(cb, frame, f"compute host late control drop handler failed (request_id={rid})")

    def _deliver_control_frame(self, request_id: str, frame: dict[str, Any]) -> None:
        with self._lock:
            q = self._pending_controls.get(request_id, (None, ""))[0]
            late = None if q is not None else self._late_control_handlers.pop(request_id, None)
        if q is not None:
            with contextlib.suppress(queue.Full):
                q.put_nowait(frame)
        elif late is not None:
            _call_logged(late[1], frame, f"compute host late control ack handler failed (request_id={request_id})")

    def _spawn_locked(self, *, reason: str) -> None:
        if self._stopped_respawning:
            raise RuntimeError("compute host respawn disabled after crash loop")
        self._hello_event.clear()
        self._hello = {}
        env = {**hermes_subprocess_env(inherit_credentials=True), **os.environ, **(self.env or {})}
        env["HERMES_COMPUTE_HOST_HEARTBEAT_SECS"] = str(self.heartbeat_secs)
        root = str(_repo_root())
        env.setdefault("PYTHONPATH", root)
        if root not in env["PYTHONPATH"].split(os.pathsep):
            env["PYTHONPATH"] = root + os.pathsep + env["PYTHONPATH"]
        # Lossy UTF-8 decode: a locale-mismatched byte must not raise inside the drain threads.
        proc = subprocess.Popen(
            self.argv, cwd=str(self.cwd), env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", bufsize=1,
            # detached-spawn: ok — compute host exits on its own parent-guard (compute_host._parent_guard_loop) and flushes sessions; a supervisor KILL could cut that flush short
            start_new_session=True)
        self._proc = proc
        stdout_drain = threading.Thread(target=self._drain_stdout, args=(proc,), name="compute-host-stdout", daemon=True)
        stdout_drain.start()
        for target, args, name in ((self._drain_stderr, (proc,), "compute-host-stderr"),
                                   (self._wait_for_exit, (proc, stdout_drain), "compute-host-wait")):
            threading.Thread(target=target, args=args, name=name, daemon=True).start()
        if not self._hello_event.wait(timeout=_HELLO_TIMEOUT_SECS):
            self._terminate_process(proc)
            raise RuntimeError(f"compute host did not send hello; stderr={self._stderr_tail[-5:]}")
        self._validate_hello()
        self._persist_registry()
        logger.info("compute host started pid=%s reason=%s", proc.pid, reason)

    def _validate_hello(self) -> None:
        hello = self._hello
        if not hello:
            raise RuntimeError("compute host missing hello")
        got_home = str(hello.get("hermes_home") or "")
        if got_home and got_home != self.expected_hermes_home:
            raise RuntimeError(
                f"compute host HERMES_HOME mismatch: {got_home} != {self.expected_hermes_home}")
        got_sha = str(hello.get("build_sha") or "")
        expected = self.expected_build_sha
        if expected != "unknown" and got_sha not in {"", "unknown", expected}:
            raise RuntimeError(f"compute host build mismatch: {got_sha} != {expected}")

    def _persist_registry(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.registry_path.with_suffix(self.registry_path.suffix + ".tmp")
        payload = {"host_pid": self.pid, "boot_id": self._hello.get("boot_id") or "",
                   "build_sha": self._hello.get("build_sha") or "", "started_at": time.time(),
                   "argv": self.argv}
        tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        tmp.replace(self.registry_path)

    def _remove_registry(self) -> None:
        try:
            self.registry_path.unlink(missing_ok=True)
        except Exception:
            logger.debug("failed to remove compute host registry", exc_info=True)

    def _send_frame(self, frame: dict[str, Any]) -> None:
        with self._lock:
            proc = self._proc
            if proc is None or proc.poll() is not None or proc.stdin is None:
                raise RuntimeError("compute host is not running")
            proc.stdin.write(json.dumps(frame, separators=(",", ":"), ensure_ascii=False) + "\n")
            proc.stdin.flush()

    def _drain_stdout(self, proc: subprocess.Popen[str]) -> None:
        assert proc.stdout is not None
        for raw in proc.stdout:
            try:
                frame = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("compute host emitted invalid json: %r", raw[:200])
                continue
            if isinstance(frame, dict):
                self._handle_host_frame(frame)

    def _drain_stderr(self, proc: subprocess.Popen[str]) -> None:
        assert proc.stderr is not None
        for raw in proc.stderr:
            if text := raw.rstrip("\n"):
                self._stderr_tail = (self._stderr_tail + [text])[-80:]
                logger.warning("compute host stderr: %s", text)

    def _handle_host_frame(self, frame: dict[str, Any]) -> None:
        ftype = str(frame.get("type") or "")
        request_id = str(frame.get("request_id") or "")
        if ftype in _CONTROL_REPLY_TYPES or (ftype == "error" and request_id):
            self._deliver_control_frame(request_id, frame)
        elif ftype == "hello":
            self._hello = dict(frame)
            self._hello_event.set()
        elif ftype == "hb":
            self._last_progress_counter = int(frame.get("progress_counter") or self._last_progress_counter)
            logger.debug("compute host heartbeat: %s", frame)
        elif ftype == "rpc":
            if isinstance(frame.get("message"), dict):
                self.rpc_sink(frame["message"])
        elif ftype in ("turn.end", "turn.error"):
            with self._lock:
                pending = self._pending_turns.pop(request_id, None)
            if pending is not None and pending[1] is not None:
                _call_logged(pending[1], frame, "compute host turn completion callback failed")

    def _wait_for_exit(self, proc: subprocess.Popen[str], stdout_drain: threading.Thread | None = None) -> None:
        code = proc.wait()
        if self._closing:
            return
        with self._lock:
            if self._proc is not proc:
                return
            self._proc = None
            # Owned by the exited process: anything registered after this belongs to a replacement host.
            owned = {*self._pending_turns, *self._pending_controls, *self._late_control_handlers}
        self._remove_registry()
        if stdout_drain is not None:
            # Frames the host wrote before it exited still answer their own waiters.
            stdout_drain.join(timeout=_EXIT_DRAIN_SECS)
        with self._lock:
            waiters = self._detach_waiters_locked(owned)
        self._fail_pending_turns(reason="crash", message=f"compute host exited with code {code}", waiters=waiters)
        self._maybe_respawn_after_crash()

    def _detach_waiters_locked(self, owned: set[str]) -> tuple[dict, dict, dict]:
        tables = (self._pending_turns, self._pending_controls, self._late_control_handlers)
        pending, waiting, late = ({rid: table.pop(rid) for rid in table.keys() & owned} for table in tables)
        return pending, waiting, late

    def _fail_pending_turns(self, *, reason: str, message: str, waiters: tuple[dict, dict, dict]) -> None:
        pending, waiting, late = waiters
        failure = {"reason": reason, "message": message}
        # A crashed host never answers controls still waiting or the late acks timed-out waiters
        # expect. Answer them before the turn callbacks, which can run long or start a new host:
        # a waiter that timed out meanwhile would register its late handler past this cleanup.
        for request_id, (q, error_type) in waiting.items():
            with contextlib.suppress(queue.Full):
                q.put_nowait({"type": error_type, "request_id": request_id, **failure})
        for request_id, (_registered_at, handler) in late.items():
            frame = {"type": "control.error", "request_id": request_id, **failure}
            _call_logged(handler, frame, "compute host late control error handler failed")
        for request_id, (sid, cb) in pending.items():
            self.rpc_sink({"jsonrpc": "2.0", "method": "event",
                           "params": {"type": "error", "session_id": sid, "payload": dict(failure)}})
            if cb is not None:
                frame = {"type": "turn.error", "sid": sid, "request_id": request_id, **failure}
                _call_logged(cb, frame, "compute host error callback failed")

    def _maybe_respawn_after_crash(self) -> None:
        now = time.monotonic()
        self._restart_times = [t for t in self._restart_times if now - t <= _RESPAWN_WINDOW_SECS]
        if len(self._restart_times) >= self.respawn_max:
            self._stopped_respawning = True
            logger.error(
                "compute host crash loop: max %s restarts per 5min reached; not respawning",
                self.respawn_max)
            return
        self._restart_times.append(now)
        # Small bounded backoff; tests and first recovery stay quick.
        delay = min(5.0, 0.25 * (2 ** max(0, len(self._restart_times) - 1)))

        def _respawn() -> None:
            time.sleep(delay)
            with self._lock:
                if self._closing or self._stopped_respawning or self._proc is not None:
                    return
                try:
                    self._spawn_locked(reason="crash")
                except Exception:
                    logger.exception("compute host respawn failed")
        threading.Thread(target=_respawn, name="compute-host-respawn", daemon=True).start()

    _pid_matches_compute_host = staticmethod(is_compute_host_identity)

    def _terminate_pid(self, pid: int, *, timeout: float = _SHUTDOWN_TIMEOUT_SECS) -> None:
        if not _signal_pid(pid, signal.SIGTERM, "SIGTERM"):
            return
        deadline = time.monotonic() + timeout
        while _pid_alive(pid):
            if time.monotonic() >= deadline:
                _signal_pid(pid, signal.SIGKILL, "SIGKILL")
                return
            time.sleep(0.05)

    def _terminate_process(self, proc: subprocess.Popen[str]) -> None:
        if proc.poll() is not None:
            return
        with contextlib.suppress(Exception):
            proc.terminate()
            proc.wait(timeout=_SHUTDOWN_TIMEOUT_SECS)
            return
        for step in (proc.kill, lambda: proc.wait(timeout=2)):
            with contextlib.suppress(Exception):
                step()


__all__ = ["MUTATOR_ROUTE_TABLE", "HostSupervisor", "append_log_record", "is_compute_host_identity"]
