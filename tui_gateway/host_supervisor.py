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
_OOM_RESPAWN_MAX = 1
_OOM_RESPAWN_MEMORY_FRACTION = 0.80
_OOM_RESPAWN_WAIT_SECS = 60.0
_OOM_RESPAWN_POLL_SECS = 5.0
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
    return Path(__file__).resolve().parents[1]


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


def _nonnegative_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _own_cgroup_v2_dir() -> Path | None:
    """Return this process' cgroup-v2 directory, or None outside Linux/cgroup-v2."""
    with contextlib.suppress(Exception):
        for line in Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines():
            parts = line.split(":", 2)
            if len(parts) == 3 and parts[0] == "0":
                relative = parts[2].lstrip("/")
                return Path("/sys/fs/cgroup") / relative if relative else Path("/sys/fs/cgroup")
    return None


def _read_cgroup_int(cgroup: Path, name: str) -> int | None:
    with contextlib.suppress(Exception):
        raw = (cgroup / name).read_text(encoding="utf-8").strip()
        if raw and raw != "max":
            return int(raw)
    return None


def _read_cgroup_events(cgroup: Path) -> dict[str, int]:
    events: dict[str, int] = {}
    with contextlib.suppress(Exception):
        for line in (cgroup / "memory.events").read_text(encoding="utf-8").splitlines():
            key, raw = line.split(maxsplit=1)
            events[key] = int(raw)
    return events


def _read_cgroup_memory_state() -> dict[str, Any]:
    """Best-effort cgroup-v2 memory counters for OOM post-mortems.

    Empty means unavailable; callers must never treat it as healthy memory.
    """
    cgroup = _own_cgroup_v2_dir()
    if cgroup is None:
        return {}
    events = _read_cgroup_events(cgroup)
    return {
        "path": str(cgroup),
        "oom_kill": events.get("oom_kill"),
        "current": _read_cgroup_int(cgroup, "memory.current"),
        "max": _read_cgroup_int(cgroup, "memory.max"),
    }


def _memory_usage_fraction(state: dict[str, Any]) -> float | None:
    current = _nonnegative_int(state.get("current"))
    maximum = _nonnegative_int(state.get("max"))
    if current is None or maximum is None or maximum <= 0:
        return None
    return current / maximum


def _memory_below_oom_respawn_threshold(state: dict[str, Any]) -> bool:
    fraction = _memory_usage_fraction(state)
    return fraction is not None and fraction < _OOM_RESPAWN_MEMORY_FRACTION


def _format_memory_state(state: dict[str, Any]) -> str:
    parts: list[str] = []
    if path := state.get("path"):
        parts.append(f"cgroup={path}")
    for label, key in (("memory.current", "current"), ("memory.max", "max"), ("oom_kill", "oom_kill")):
        if state.get(key) is not None:
            parts.append(f"{label}={state[key]}")
    if (fraction := _memory_usage_fraction(state)) is not None:
        parts.append(f"usage={fraction:.1%}")
    return "; ".join(parts) if parts else "cgroup memory state unavailable"


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
        self._oom_restart_times: list[float] = []
        self._oom_respawn_pending = False
        self._spawn_memory_state: dict[str, Any] = {}
        self._pending_turns: dict[str, tuple[str, Callable[[dict], None] | None]] = {}
        self._pending_controls: dict[str, queue.Queue[dict]] = {}
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
            if self._oom_respawn_pending:
                raise RuntimeError(
                    "compute host respawn is deferred after a cgroup OOM until memory recovers")
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

    def _await_reply(self, frame: dict[str, Any], request_id: str, timeout: float) -> dict:
        """Send ``frame`` and block for the host reply carrying ``request_id``."""
        q: queue.Queue[dict] = queue.Queue(maxsize=1)
        with self._lock:
            self._pending_controls[request_id] = q
        try:
            self._send_frame(frame)
            return q.get(timeout=timeout)
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
        try:
            return self._await_reply(frame, request_id, timeout)
        except queue.Empty:
            if on_late_ack is not None:
                self._register_late_control_handler(request_id, on_late_ack)
            raise

    def _register_late_control_handler(self, request_id: str, handler: Callable[[dict], None]) -> None:
        now = time.monotonic()
        with self._lock:
            handlers = self._late_control_handlers
            for rid in [r for r, (at, _cb) in handlers.items() if now - at > _LATE_CONTROL_TTL_SECS]:
                handlers.pop(rid, None)
            while len(handlers) >= _LATE_CONTROL_MAX:
                handlers.pop(min(handlers, key=lambda rid: handlers[rid][0]), None)
            handlers[request_id] = (now, handler)

    def _deliver_control_frame(self, request_id: str, frame: dict[str, Any]) -> None:
        with self._lock:
            q = self._pending_controls.get(request_id)
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
        self._spawn_memory_state = _read_cgroup_memory_state()
        proc = subprocess.Popen(
            self.argv, cwd=str(self.cwd), env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", bufsize=1,
            start_new_session=True)
        self._proc = proc
        for target, name in ((self._drain_stdout, "compute-host-stdout"),
                             (self._drain_stderr, "compute-host-stderr"),
                             (self._wait_for_exit, "compute-host-wait")):
            threading.Thread(target=target, args=(proc,), name=name, daemon=True).start()
        if not self._hello_event.wait(timeout=10.0):
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

    def _wait_for_exit(self, proc: subprocess.Popen[str]) -> None:
        code = proc.wait()
        if self._closing:
            return
        with self._lock:
            if self._proc is not proc:
                return
            self._proc = None
        self._remove_registry()
        exit_reason, message, exit_info = self._classify_exit(code)
        self._fail_pending_turns(reason=exit_reason, message=message)
        self._maybe_respawn_after_crash(exit_reason=exit_reason, exit_info=exit_info)

    def _classify_exit(self, code: int) -> tuple[str, str, dict[str, Any]]:
        info: dict[str, Any] = {"exit_code": code}
        if code == -signal.SIGKILL:
            before = self._spawn_memory_state
            after = _read_cgroup_memory_state()
            before_oom = _nonnegative_int(before.get("oom_kill"))
            after_oom = _nonnegative_int(after.get("oom_kill"))
            if before_oom is not None and after_oom is not None and after_oom > before_oom:
                delta = after_oom - before_oom
                info.update({"oom_kill_delta": delta, "memory": after})
                return (
                    "oom_kill",
                    f"compute host exited with code {code}; cgroup OOM kill detected "
                    f"(oom_kill delta={delta}; {_format_memory_state(after)})",
                    info,
                )
        return "crash", f"compute host exited with code {code}", info

    def _fail_pending_turns(self, *, reason: str, message: str) -> None:
        with self._lock:
            pending = self._pending_turns
            self._pending_turns = {}
        failure = {"reason": reason, "message": message}
        for request_id, (sid, cb) in pending.items():
            self.rpc_sink({"jsonrpc": "2.0", "method": "event",
                           "params": {"type": "error", "session_id": sid, "payload": dict(failure)}})
            if cb is not None:
                frame = {"type": "turn.error", "sid": sid, "request_id": request_id, **failure}
                _call_logged(cb, frame, "compute host error callback failed")
        # A crashed host never emits the late acks timed-out control waiters still expect; fail
        # them too so the client's "still running" notice can't hang.
        with self._lock:
            late = self._late_control_handlers
            self._late_control_handlers = {}
        for request_id, (_registered_at, handler) in late.items():
            frame = {"type": "control.error", "request_id": request_id, **failure}
            _call_logged(handler, frame, "compute host late control error handler failed")

    def _maybe_respawn_after_crash(
        self, *, exit_reason: str = "crash", exit_info: dict[str, Any] | None = None,
    ) -> None:
        if exit_reason == "oom_kill":
            self._maybe_respawn_after_oom(exit_info or {})
            return
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

    def _maybe_respawn_after_oom(self, exit_info: dict[str, Any]) -> None:
        now = time.monotonic()
        self._oom_restart_times = [t for t in self._oom_restart_times if now - t <= _RESPAWN_WINDOW_SECS]
        if len(self._oom_restart_times) >= _OOM_RESPAWN_MAX:
            self._stopped_respawning = True
            logger.error(
                "compute host OOM restart loop: max %s OOM restarts per 5min reached; "
                "not respawning; %s", _OOM_RESPAWN_MAX,
                _format_memory_state(exit_info.get("memory") or {}))
            return
        self._oom_restart_times.append(now)
        self._oom_respawn_pending = True
        logger.error(
            "compute host was OOM-killed; waiting for cgroup memory below %.0f%% before respawn; %s",
            _OOM_RESPAWN_MEMORY_FRACTION * 100,
            _format_memory_state(exit_info.get("memory") or {}))
        if not self._wait_until_oom_pressure_recedes(exit_info):
            self._oom_respawn_pending = False
            self._stopped_respawning = True
            logger.error(
                "compute host OOM respawn suppressed: cgroup memory did not fall below %.0f%%; %s",
                _OOM_RESPAWN_MEMORY_FRACTION * 100,
                _format_memory_state(_read_cgroup_memory_state()))
            return
        with self._lock:
            self._oom_respawn_pending = False
            if self._closing or self._stopped_respawning or self._proc is not None:
                return
            try:
                self._spawn_locked(reason="oom_recovery")
            except Exception:
                logger.exception("compute host OOM recovery respawn failed")

    def _wait_until_oom_pressure_recedes(self, exit_info: dict[str, Any]) -> bool:
        raw_state = exit_info.get("memory")
        state = raw_state if isinstance(raw_state, dict) else {}
        deadline = time.monotonic() + _OOM_RESPAWN_WAIT_SECS
        while True:
            if _memory_below_oom_respawn_threshold(state):
                return True
            # A confirmed OOM without a readable finite cap has no safe threshold to wait for.
            if _memory_usage_fraction(state) is None or time.monotonic() >= deadline:
                return False
            time.sleep(min(_OOM_RESPAWN_POLL_SECS, max(0.0, deadline - time.monotonic())))
            state = _read_cgroup_memory_state()

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
