"""Persistent local Python kernels used by ``execute_code``."""

from __future__ import annotations

import ast
import asyncio
import atexit
import base64
import builtins
import inspect
import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


_FRAME_PREFIX = b"\x1ehermes-python\t"


class KernelDiedError(RuntimeError):
    pass


class KernelStartupTimeout(KernelDiedError):
    pass


class KernelStartupInterrupted(KernelDiedError):
    pass


class _CapturedOutput:
    def __init__(self, limit: int):
        self._head_limit = int(limit * 0.4)
        self._tail_limit = limit - self._head_limit
        self._head = bytearray()
        self._tail: deque[bytes] = deque()
        self._tail_size = 0
        self.total = 0

    def append(self, data: bytes) -> None:
        self.total += len(data)
        if len(self._head) < self._head_limit:
            keep = min(len(data), self._head_limit - len(self._head))
            self._head.extend(data[:keep])
            data = data[keep:]
        if not data:
            return
        self._tail.append(data)
        self._tail_size += len(data)
        while self._tail_size > self._tail_limit and self._tail:
            overflow = self._tail_size - self._tail_limit
            first = self._tail[0]
            if len(first) <= overflow:
                self._tail.popleft()
                self._tail_size -= len(first)
            else:
                self._tail[0] = first[overflow:]
                self._tail_size -= overflow

    @property
    def head(self) -> bytes:
        return bytes(self._head)

    @property
    def tail(self) -> bytes:
        return b"".join(self._tail)


@dataclass
class KernelExecutionResult:
    status: str
    stdout_head: bytes
    stdout_tail: bytes
    stdout_total: int
    stderr: bytes
    error: str
    invalidate_kernel: bool


class PersistentPythonKernel:
    def __init__(
        self,
        python: str,
        cwd: str,
        env: dict[str, str],
        tools_source: str,
        *,
        deadline: Optional[float] = None,
        interrupted: Optional[Callable[[], bool]] = None,
    ):
        self.python = python
        self.staging_dir = tempfile.mkdtemp(prefix="hermes_kernel_")
        self.cwd = cwd or self.staging_dir
        self.env = dict(env)
        existing_path = self.env.get("PYTHONPATH", "")
        self.env["PYTHONPATH"] = os.pathsep.join(
            part for part in (self.staging_dir, existing_path) if part
        )
        self._lock = threading.Lock()
        self._frames: queue.Queue[dict] = queue.Queue()
        self._infra_stderr = bytearray()
        self._closed = False
        Path(self.staging_dir, "hermes_tools.py").write_text(
            tools_source, encoding="utf-8"
        )
        self._proc = subprocess.Popen(
            [python, str(Path(__file__).resolve()), "--runner"],
            cwd=self.cwd,
            env=self.env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            creationflags=(
                subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            ),
        )
        self._stdout_thread = threading.Thread(
            target=self._read_frames, daemon=True
        )
        self._stderr_thread = threading.Thread(
            target=self._read_infra_stderr, daemon=True
        )
        self._stdout_thread.start()
        self._stderr_thread.start()
        try:
            self._wait_until_ready(
                deadline=deadline,
                interrupted=interrupted or (lambda: False),
            )
        except Exception:
            self.close()
            raise

    @property
    def pid(self) -> int:
        return self._proc.pid

    @property
    def alive(self) -> bool:
        return not self._closed and self._proc.poll() is None

    def execute(
        self,
        code: str,
        *,
        cwd: str,
        env: dict[str, str],
        python_paths: list[str],
        deadline: float,
        interrupted: Callable[[], bool],
        stdout_limit: int,
        stderr_limit: int,
        activity: Optional[Callable[[], None]] = None,
    ) -> KernelExecutionResult:
        while True:
            if interrupted():
                return KernelExecutionResult(
                    status="interrupted",
                    stdout_head=b"",
                    stdout_tail=b"",
                    stdout_total=0,
                    stderr=b"",
                    error="",
                    invalidate_kernel=False,
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return KernelExecutionResult(
                    status="timeout",
                    stdout_head=b"",
                    stdout_tail=b"",
                    stdout_total=0,
                    stderr=b"",
                    error="",
                    invalidate_kernel=False,
                )
            if self._lock.acquire(timeout=min(0.1, remaining)):
                break
        try:
            if not self.alive:
                raise KernelDiedError(self._death_message())
            request_id = uuid.uuid4().hex
            payload = {
                "type": "execute",
                "id": request_id,
                "code": code,
                "cwd": cwd,
                "env": env,
                "python_paths": python_paths,
            }
            assert self._proc.stdin is not None
            try:
                self._proc.stdin.write((json.dumps(payload) + "\n").encode("utf-8"))
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                raise KernelDiedError(self._death_message()) from exc

            stdout = _CapturedOutput(stdout_limit)
            stderr = _CapturedOutput(stderr_limit)
            status = "success"
            error = ""
            invalidate_kernel = False
            while True:
                if interrupted():
                    status = "interrupted"
                    invalidate_kernel = True
                    self._interrupt_then_stop()
                    break
                if time.monotonic() >= deadline:
                    status = "timeout"
                    invalidate_kernel = True
                    self._interrupt_then_stop()
                    break
                if activity is not None:
                    activity()
                try:
                    frame = self._frames.get(timeout=0.1)
                except queue.Empty:
                    if self._proc.poll() is not None:
                        raise KernelDiedError(self._death_message())
                    continue
                if frame.get("type") == "eof":
                    raise KernelDiedError(self._death_message())
                if frame.get("id") not in (None, request_id):
                    continue
                kind = frame.get("type")
                if kind == "stdout":
                    stdout.append(base64.b64decode(frame.get("data", "")))
                elif kind == "stderr":
                    stderr.append(base64.b64decode(frame.get("data", "")))
                elif kind == "error":
                    status = "error"
                    error = str(frame.get("error") or "Python execution failed")
                elif kind == "done":
                    break

            return KernelExecutionResult(
                status=status,
                stdout_head=stdout.head,
                stdout_tail=stdout.tail,
                stdout_total=stdout.total,
                stderr=stderr.head + stderr.tail,
                error=error,
                invalidate_kernel=invalidate_kernel,
            )
        finally:
            self._lock.release()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._terminate()
        shutil.rmtree(self.staging_dir, ignore_errors=True)

    def _wait_until_ready(
        self,
        *,
        deadline: Optional[float],
        interrupted: Callable[[], bool],
    ) -> None:
        deadline = deadline if deadline is not None else time.monotonic() + 10
        while time.monotonic() < deadline:
            if interrupted():
                self._terminate()
                raise KernelStartupInterrupted("Python kernel startup was interrupted")
            try:
                frame = self._frames.get(
                    timeout=min(0.1, max(0.001, deadline - time.monotonic()))
                )
            except queue.Empty:
                if self._proc.poll() is not None:
                    break
                continue
            if frame.get("type") == "ready":
                return
            if frame.get("type") == "eof":
                break
        self._terminate()
        if time.monotonic() >= deadline:
            raise KernelStartupTimeout("Python kernel did not start before the deadline")
        raise KernelDiedError(self._death_message() or "Python kernel did not start")

    def _read_frames(self) -> None:
        assert self._proc.stdout is not None
        try:
            for line in iter(self._proc.stdout.readline, b""):
                if not line.startswith(_FRAME_PREFIX):
                    continue
                try:
                    self._frames.put(json.loads(line[len(_FRAME_PREFIX):]))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
        finally:
            self._frames.put({"type": "eof"})

    def _read_infra_stderr(self) -> None:
        assert self._proc.stderr is not None
        try:
            while len(self._infra_stderr) < 10_000:
                chunk = self._proc.stderr.read(4096)
                if not chunk:
                    break
                self._infra_stderr.extend(chunk[: 10_000 - len(self._infra_stderr)])
        except (OSError, ValueError):
            pass

    def _interrupt_then_stop(self) -> None:
        try:
            if os.name == "nt":
                self._proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(self._proc.pid, signal.SIGINT)  # windows-footgun: ok
            self._proc.wait(timeout=1)
        except Exception:
            pass
        self._terminate()

    def _terminate(self) -> None:
        if os.name != "nt":
            try:
                os.killpg(self._proc.pid, signal.SIGTERM)  # windows-footgun: ok
            except ProcessLookupError:
                return
            try:
                self._proc.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                pass
            time.sleep(0.05)
            try:
                os.killpg(self._proc.pid, 0)  # windows-footgun: ok
            except ProcessLookupError:
                pass
            else:
                try:
                    os.killpg(self._proc.pid, signal.SIGKILL)  # windows-footgun: ok
                except ProcessLookupError:
                    pass
            return
        if self._proc.poll() is not None:
            return
        try:
            import psutil

            parent = psutil.Process(self._proc.pid)
            processes = parent.children(recursive=True)
            processes.append(parent)
            for process in processes:
                try:
                    process.terminate()
                except psutil.NoSuchProcess:
                    pass
            _, alive = psutil.wait_procs(processes, timeout=1)
            for process in alive:
                try:
                    process.kill()
                except psutil.NoSuchProcess:
                    pass
            self._proc.wait(timeout=1)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass

    def _death_message(self) -> str:
        detail = bytes(self._infra_stderr).decode("utf-8", errors="replace").strip()
        if detail:
            return detail
        return f"Python kernel exited with code {self._proc.poll()}"


@dataclass
class _KernelEntry:
    kernel: PersistentPythonKernel
    scope_id: str
    profile_key: str
    last_used: float
    idle_seconds: float
    active: int = 0


@dataclass
class _PendingKernel:
    scope_id: str
    profile_key: str
    generation: int
    event: threading.Event
    error: Optional[BaseException] = None
    invalidated: bool = False


class KernelRegistry:
    def __init__(self):
        self._entries: dict[tuple, _KernelEntry] = {}
        self._pending: dict[tuple, _PendingKernel] = {}
        self._scope_generations: dict[tuple[str, str], int] = {}
        self._lock = threading.RLock()
        self._reaper_started = False
        self._idle_seconds = 1800.0

    def acquire(
        self,
        key: tuple,
        *,
        scope_id: str,
        profile_key: str,
        factory: Callable[[], PersistentPythonKernel],
        max_live: int,
        idle_seconds: float,
        deadline: Optional[float] = None,
        interrupted: Optional[Callable[[], bool]] = None,
    ) -> tuple[PersistentPythonKernel, bool]:
        interrupted = interrupted or (lambda: False)
        scope_key = (scope_id, profile_key)
        while True:
            victim = None
            starter = False
            with self._lock:
                entry_idle_seconds = max(1.0, float(idle_seconds))
                self._idle_seconds = min(self._idle_seconds, entry_idle_seconds)
                self._start_reaper_locked()
                entry = self._entries.get(key)
                if entry is not None and entry.kernel.alive:
                    entry.active += 1
                    entry.last_used = time.monotonic()
                    return entry.kernel, True
                if entry is not None:
                    victim = entry.kernel
                    self._entries.pop(key, None)
                pending = self._pending.get(key)
                if pending is None:
                    pending = _PendingKernel(
                        scope_id=scope_id,
                        profile_key=profile_key,
                        generation=self._scope_generations.get(scope_key, 0),
                        event=threading.Event(),
                    )
                    self._pending[key] = pending
                    starter = True
            if victim is not None:
                victim.close()
            if starter:
                break
            while not pending.event.wait(timeout=0.05):
                if interrupted():
                    raise KernelStartupInterrupted(
                        "Python kernel startup was interrupted"
                    )
                if deadline is not None and time.monotonic() >= deadline:
                    raise KernelStartupTimeout(
                        "Python kernel did not start before the deadline"
                    )
            if pending.error is not None:
                raise pending.error
            if pending.invalidated:
                raise KernelDiedError("Python kernel startup was invalidated")

        kernel = None
        victims: list[PersistentPythonKernel] = []
        try:
            if interrupted():
                raise KernelStartupInterrupted("Python kernel startup was interrupted")
            if deadline is not None and time.monotonic() >= deadline:
                raise KernelStartupTimeout(
                    "Python kernel did not start before the deadline"
                )
            kernel = factory()
            with self._lock:
                current = self._pending.get(key)
                generation = self._scope_generations.get(scope_key, 0)
                if current is not pending or pending.invalidated or generation != pending.generation:
                    raise KernelDiedError("Python kernel startup was invalidated")
                self._entries[key] = _KernelEntry(
                    kernel=kernel,
                    scope_id=scope_id,
                    profile_key=profile_key,
                    last_used=time.monotonic(),
                    idle_seconds=entry_idle_seconds,
                    active=1,
                )
                self._pending.pop(key, None)
                victims.extend(self._evict_lru_locked(max_live, preserve=key))
                pending.event.set()
            for victim in victims:
                victim.close()
            return kernel, False
        except BaseException as exc:
            if kernel is not None:
                kernel.close()
            with self._lock:
                if self._pending.get(key) is pending:
                    self._pending.pop(key, None)
                pending.error = exc
                pending.event.set()
            raise

    def release(self, key: tuple, kernel: Optional[PersistentPythonKernel] = None) -> None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and (kernel is None or entry.kernel is kernel):
                entry.active = max(0, entry.active - 1)
                entry.last_used = time.monotonic()

    def discard(
        self,
        key: tuple,
        expected_kernel: Optional[PersistentPythonKernel] = None,
    ) -> None:
        with self._lock:
            entry = self._entries.pop(key, None)
            if entry is not None and (
                expected_kernel is None or entry.kernel is expected_kernel
            ):
                removed = entry.kernel
            elif entry is not None:
                self._entries[key] = entry
                removed = None
            else:
                removed = None
        if removed is not None:
            removed.close()

    def dispose_scope(self, scope_id: str, profile_key: Optional[str] = None) -> None:
        victims = []
        with self._lock:
            profiles = {entry.profile_key for entry in self._entries.values() if entry.scope_id == scope_id}
            profiles.update(
                pending.profile_key
                for pending in self._pending.values()
                if pending.scope_id == scope_id
            )
            if profile_key is not None:
                profiles.add(profile_key)
            for profile in profiles:
                if profile_key is None or profile == profile_key:
                    scope_key = (scope_id, profile)
                    self._scope_generations[scope_key] = self._scope_generations.get(scope_key, 0) + 1
            for key, entry in list(self._entries.items()):
                if entry.scope_id != scope_id:
                    continue
                if profile_key is not None and entry.profile_key != profile_key:
                    continue
                victims.append(entry.kernel)
                self._entries.pop(key, None)
            for key, pending in list(self._pending.items()):
                if pending.scope_id != scope_id:
                    continue
                if profile_key is not None and pending.profile_key != profile_key:
                    continue
                pending.invalidated = True
                pending.event.set()
                self._pending.pop(key, None)
        for victim in victims:
            victim.close()

    def close_all(self) -> None:
        with self._lock:
            victims = [entry.kernel for entry in self._entries.values()]
            self._entries.clear()
            for pending in self._pending.values():
                pending.invalidated = True
                pending.event.set()
            self._pending.clear()
            self._scope_generations = {
                scope: generation + 1
                for scope, generation in self._scope_generations.items()
            }
        for victim in victims:
            victim.close()

    def size(self) -> int:
        with self._lock:
            return len(self._entries)

    def reap_idle(self, now: Optional[float] = None) -> int:
        current = time.monotonic() if now is None else now
        victims = []
        with self._lock:
            for key, entry in list(self._entries.items()):
                if (
                    entry.active == 0
                    and current - entry.last_used > entry.idle_seconds
                ):
                    victims.append(entry.kernel)
                    self._entries.pop(key, None)
        for victim in victims:
            victim.close()
        return len(victims)

    def _evict_lru_locked(self, max_live: int, preserve: tuple) -> list[PersistentPythonKernel]:
        victims = []
        limit = max(1, int(max_live))
        while len(self._entries) > limit:
            candidates = [
                (key, entry) for key, entry in self._entries.items()
                if key != preserve and entry.active == 0
            ]
            if not candidates:
                break
            key, entry = min(candidates, key=lambda item: item[1].last_used)
            self._entries.pop(key, None)
            victims.append(entry.kernel)
        return victims

    def _start_reaper_locked(self) -> None:
        if self._reaper_started:
            return
        self._reaper_started = True
        threading.Thread(target=self._reaper_loop, daemon=True).start()

    def _reaper_loop(self) -> None:
        while True:
            time.sleep(min(60.0, max(1.0, self._idle_seconds / 2)))
            self.reap_idle()


kernel_registry = KernelRegistry()
atexit.register(kernel_registry.close_all)


def _runner_send(protocol, frame: dict, lock: threading.Lock) -> None:
    data = _FRAME_PREFIX + json.dumps(frame, ensure_ascii=False).encode("utf-8") + b"\n"
    with lock:
        protocol.write(data)
        protocol.flush()


def _runner_output_drain(pipe, stream_name: str, current_id, protocol, lock) -> None:
    while True:
        data = os.read(pipe, 4096)
        if not data:
            return
        request_id = current_id[0]
        if request_id:
            _runner_send(
                protocol,
                {
                    "type": stream_name,
                    "id": request_id,
                    "data": base64.b64encode(data).decode("ascii"),
                },
                lock,
            )


def _runner_eval(code: str, namespace: dict) -> None:
    tree = ast.parse(code, filename="<cell>", mode="exec")
    body = list(tree.body)
    last_expr = body.pop() if body and isinstance(body[-1], ast.Expr) else None
    flags = ast.PyCF_ALLOW_TOP_LEVEL_AWAIT
    if body:
        prefix = ast.Module(body=body, type_ignores=[])
        ast.fix_missing_locations(prefix)
        value = eval(compile(prefix, "<cell>", "exec", flags=flags), namespace)
        if inspect.isawaitable(value):
            asyncio.run(value)
    if last_expr is not None:
        expression = ast.Expression(last_expr.value)
        ast.fix_missing_locations(expression)
        value = eval(compile(expression, "<cell>", "eval", flags=flags), namespace)
        if inspect.isawaitable(value):
            value = asyncio.run(value)
        namespace["_"] = value
        if value is not None:
            print(repr(value))


def _format_runner_exception(exc: BaseException) -> str:
    formatted = traceback.TracebackException.from_exception(exc)
    runner_path = os.path.realpath(__file__)
    formatted.stack = traceback.StackSummary.from_list(
        frame
        for frame in formatted.stack
        if os.path.realpath(frame.filename) != runner_path
    )
    return "".join(formatted.format())


def run_kernel() -> None:
    protocol = os.fdopen(os.dup(1), "wb", buffering=0)  # windows-footgun: ok
    send_lock = threading.Lock()
    current_id = [None]
    stdout_read, stdout_write = os.pipe()
    stderr_read, stderr_write = os.pipe()
    os.dup2(stdout_write, 1)
    os.dup2(stderr_write, 2)
    os.close(stdout_write)
    os.close(stderr_write)
    sys.stdout = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
    sys.stderr = os.fdopen(os.dup(2), "w", encoding="utf-8", buffering=1)
    threading.Thread(
        target=_runner_output_drain,
        args=(stdout_read, "stdout", current_id, protocol, send_lock),
        daemon=True,
    ).start()
    threading.Thread(
        target=_runner_output_drain,
        args=(stderr_read, "stderr", current_id, protocol, send_lock),
        daemon=True,
    ).start()

    def _no_input(*_args, **_kwargs):
        raise RuntimeError("input() is unavailable inside execute_code")

    namespace = {
        "__name__": "__main__",
        "__builtins__": dict(vars(builtins), input=_no_input),
    }
    managed_env: set[str] = set()
    managed_paths: list[str] = []
    _runner_send(protocol, {"type": "ready"}, send_lock)
    for raw in sys.stdin.buffer:
        try:
            request = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if request.get("type") == "shutdown":
            return
        if request.get("type") != "execute":
            continue
        request_id = str(request.get("id") or "")
        current_id[0] = request_id
        try:
            requested_env = request.get("env") or {}
            for key in managed_env - set(requested_env):
                os.environ.pop(key, None)
            for key, value in requested_env.items():
                os.environ[str(key)] = str(value)
            managed_env = set(requested_env)
            requested_paths = [str(p) for p in request.get("python_paths") or [] if p]
            for path in managed_paths:
                while path in sys.path:
                    sys.path.remove(path)
            for path in reversed(requested_paths):
                if path not in sys.path:
                    sys.path.insert(0, path)
            managed_paths = requested_paths
            os.chdir(str(request.get("cwd") or os.getcwd()))
            _runner_eval(str(request.get("code") or ""), namespace)
        except BaseException as exc:
            error = _format_runner_exception(exc)
            print(error, file=sys.stderr, end="")
            _runner_send(
                protocol,
                {"type": "error", "id": request_id, "error": error},
                send_lock,
            )
        finally:
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            except Exception:
                pass
            time.sleep(0.01)
            os.environ.pop("HERMES_RPC_SOCKET", None)
            os.environ.pop("HERMES_RPC_TOKEN", None)
            managed_env.discard("HERMES_RPC_SOCKET")
            managed_env.discard("HERMES_RPC_TOKEN")
            _runner_send(protocol, {"type": "done", "id": request_id}, send_lock)
            current_id[0] = None


if __name__ == "__main__" and "--runner" in sys.argv:
    run_kernel()
