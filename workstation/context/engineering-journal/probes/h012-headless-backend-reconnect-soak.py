"""Real headless Hermes backend multi-session/reconnect soak.

This probe starts the shipped ``hermes serve`` command as a real child
process, connects through the production WebSocket gateway, drives concurrent
desktop-shaped sessions, kills/restarts the backend, and resumes the durable
sessions from the same ``HERMES_HOME``.  The synthetic turn seam is deliberate:
it keeps the evidence deterministic and token-free while exercising the real
backend process, JSON-RPC transport, session persistence, worker dispatch and
streamed completion path.

It does not launch Electron, open a browser, or claim a real-provider/model
quality result.  Its accepted classification is a backend/load boundary that
complements the native Browser runtime probe.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable, Iterable


REPO_ROOT = Path(__file__).resolve().parents[4]
READY_RE = re.compile(r"HERMES_(?:BACKEND|DASHBOARD)_READY\s+port=(\d+)")
DEFAULT_SESSION_COUNT = 4
DEFAULT_MAX_CYCLES = 12
DEFAULT_RESTARTS = 2
DEFAULT_RESTART_EVERY = 3
DEFAULT_DURATION_MS = 120_000
RPC_TIMEOUT_SECONDS = 30.0


class ProbeFailure(RuntimeError):
    """A deterministic probe contract failure."""


@dataclass(slots=True)
class BackendSoakReport:
    scenario: str
    accepted: bool
    duration_ms: int
    elapsed_ms: int
    session_count: int
    completed_cycles: int
    turns: int
    reconnects: int
    backend_restarts: int
    required_restarts: int
    heartbeats: int
    streamed_events: int
    max_concurrent_turns: int
    backend_pids: list[int] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    retained_root: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _BackendProcess:
    """Own one real ``hermes serve`` process and its READY sentinel."""

    def __init__(self, root: Path, token: str) -> None:
        self.root = root
        self.token = token
        self.process: subprocess.Popen[str] | None = None
        self._stdout: queue.Queue[str] = queue.Queue()
        self._stderr: list[str] = []
        self._threads: list[threading.Thread] = []
        self.port: int | None = None

    def start(self) -> int:
        if self.process is not None and self.process.poll() is None:
            raise ProbeFailure("backend process is already running")

        home = self.root / "hermes-home"
        home.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update(
            {
                "HERMES_HOME": str(home),
                "HERMES_DASHBOARD_SESSION_TOKEN": self.token,
                "HERMES_ISO_CERTIFY_SYNTH_TURN": "1",
                "HERMES_TUI_WS_ORPHAN_REAP_GRACE_S": "1",
                "HERMES_SERVE_HEADLESS": "1",
                "PYTHONPATH": str(REPO_ROOT)
                + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
            }
        )
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "hermes_cli.main",
                "serve",
                "--host",
                "127.0.0.1",
                "--port",
                "0",
                "--skip-build",
            ],
            cwd=REPO_ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creationflags,
        )
        self._start_reader(self.process.stdout, self._stdout, self._threads)
        self._start_reader(self.process.stderr, None, self._threads)

        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise ProbeFailure(self._failure_message("backend exited before READY"))
            try:
                line = self._stdout.get(timeout=0.2)
            except queue.Empty:
                continue
            match = READY_RE.search(line)
            if match:
                self.port = int(match.group(1))
                return self.port
        raise ProbeFailure(self._failure_message("backend READY timeout"))

    def _start_reader(
        self,
        stream: Any,
        output: queue.Queue[str] | None,
        threads: list[threading.Thread],
    ) -> None:
        def read_lines() -> None:
            try:
                for raw in stream:
                    line = raw.rstrip("\r\n")
                    if output is not None:
                        output.put(line)
                    else:
                        self._stderr.append(line)
                        if len(self._stderr) > 80:
                            del self._stderr[:-80]
            except (OSError, ValueError):
                return

        thread = threading.Thread(target=read_lines, name="h012-backend-reader", daemon=True)
        thread.start()
        threads.append(thread)

    @property
    def ws_url(self) -> str:
        if self.port is None:
            raise ProbeFailure("backend port is not available")
        return f"ws://127.0.0.1:{self.port}/api/ws?token={self.token}"

    def stop(self) -> None:
        process = self.process
        self.process = None
        self.port = None
        if process is None or process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            try:
                process.kill()
                process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                pass

    def _failure_message(self, prefix: str) -> str:
        detail = "\n".join(self._stderr[-20:])
        return f"{prefix}; backend stderr:\n{detail}" if detail else prefix


class _RpcClient:
    """Small JSON-RPC client for the shipped WebSocket transport."""

    def __init__(self, url: str, label: str, report: BackendSoakReport) -> None:
        self.url = url
        self.label = label
        self.report = report
        self.websocket: Any = None
        self.reader: asyncio.Task[None] | None = None
        self.events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.pending: dict[int, asyncio.Future[Any]] = {}
        self.next_id = 0

    async def connect(self) -> None:
        try:
            from websockets.asyncio.client import connect
        except ImportError as exc:  # pragma: no cover - dependency is bundled
            raise ProbeFailure("websockets package is required for H012") from exc

        self.websocket = await connect(
            self.url,
            open_timeout=RPC_TIMEOUT_SECONDS,
            close_timeout=3,
            max_size=4 * 1024 * 1024,
            ping_interval=None,
        )
        self.reader = asyncio.create_task(self._read_frames())
        await self.wait_event(
            lambda frame: frame.get("method") == "event"
            and (frame.get("params") or {}).get("type") == "gateway.ready"
        )

    async def _read_frames(self) -> None:
        try:
            async for raw in self.websocket:
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", errors="replace")
                frame = json.loads(raw)
                if not isinstance(frame, dict):
                    continue
                request_id = frame.get("id")
                if isinstance(request_id, int) and request_id in self.pending:
                    future = self.pending.pop(request_id)
                    if "error" in frame:
                        error = frame.get("error") or {}
                        message = str(error.get("message") or "JSON-RPC error")
                        if not future.done():
                            future.set_exception(ProbeFailure(f"{self.label}: {message}"))
                    elif not future.done():
                        future.set_result(frame.get("result"))
                    continue
                params = frame.get("params") or {}
                if params.get("type") in {"message.delta", "reasoning.delta", "thinking.delta"}:
                    self.report.streamed_events += 1
                await self.events.put(frame)
        except Exception as exc:
            error = ProbeFailure(f"{self.label}: WebSocket reader stopped: {exc}")
            for future in list(self.pending.values()):
                if not future.done():
                    future.set_exception(error)
            self.pending.clear()

    async def request(self, method: str, params: dict[str, Any]) -> Any:
        if self.websocket is None:
            raise ProbeFailure(f"{self.label}: request before connect")
        self.next_id += 1
        request_id = self.next_id
        future = asyncio.get_running_loop().create_future()
        self.pending[request_id] = future
        await self.websocket.send(
            json.dumps({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        )
        return await asyncio.wait_for(future, timeout=RPC_TIMEOUT_SECONDS)

    async def wait_event(
        self,
        predicate: Callable[[dict[str, Any]], bool],
        *,
        timeout: float = RPC_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ProbeFailure(f"{self.label}: event timeout")
            frame = await asyncio.wait_for(self.events.get(), timeout=remaining)
            if predicate(frame):
                return frame

    async def close(self) -> None:
        if self.websocket is not None:
            try:
                await self.websocket.close()
            except Exception:
                pass
        if self.reader is not None:
            try:
                await asyncio.wait_for(self.reader, timeout=3)
            except (Exception, asyncio.CancelledError):
                self.reader.cancel()
        self.websocket = None
        self.reader = None


async def _drive_session(
    client: _RpcClient,
    *,
    session_index: int,
    stored_session_id: str | None,
    concurrency: dict[str, int],
) -> str:
    if stored_session_id:
        payload = await client.request(
            "session.resume",
            {
                "session_id": stored_session_id,
                "cols": 120,
                "source": "desktop",
            },
        )
        if not isinstance(payload, dict) or payload.get("resumed") != stored_session_id:
            raise ProbeFailure(f"{client.label}: session.resume did not restore {stored_session_id}")
        live_session_id = str(payload.get("session_id") or "")
    else:
        payload = await client.request(
            "session.create",
            {
                "cols": 120,
                "cwd": str(REPO_ROOT),
                "source": "desktop",
                "title": f"H012 session {session_index}",
            },
        )
        if not isinstance(payload, dict):
            raise ProbeFailure(f"{client.label}: session.create returned invalid payload")
        live_session_id = str(payload.get("session_id") or "")
        stored_session_id = str(payload.get("stored_session_id") or "")
    if not live_session_id or not stored_session_id:
        raise ProbeFailure(f"{client.label}: session identity was incomplete")

    await client.request("gateway.ping", {})
    client.report.heartbeats += 1
    spec = {
        "duration_s": 0.16,
        "chunk": 12_000,
        "delta_interval_s": 0.035,
        "tokens_per_delta": 32,
    }
    concurrency["active"] += 1
    concurrency["max"] = max(concurrency["max"], concurrency["active"])
    try:
        await client.request(
            "prompt.submit",
            {"session_id": live_session_id, "text": json.dumps(spec, separators=(",", ":"))},
        )
        await client.wait_event(
            lambda frame: (frame.get("params") or {}).get("type") == "message.complete"
            and (frame.get("params") or {}).get("session_id") == live_session_id,
            timeout=RPC_TIMEOUT_SECONDS,
        )
    finally:
        concurrency["active"] -= 1
    client.report.turns += 1
    return stored_session_id


async def _run_scenario(
    *,
    root: Path,
    duration_ms: int,
    session_count: int,
    max_cycles: int,
    required_restarts: int,
    restart_every: int,
) -> BackendSoakReport:
    started = time.monotonic()
    report = BackendSoakReport(
        scenario="real-headless-hermes-backend-multi-session-reconnect",
        accepted=False,
        duration_ms=duration_ms,
        elapsed_ms=0,
        session_count=session_count,
        completed_cycles=0,
        turns=0,
        reconnects=0,
        backend_restarts=0,
        required_restarts=required_restarts,
        heartbeats=0,
        streamed_events=0,
        max_concurrent_turns=0,
        retained_root=True,
    )
    token = f"h012-{os.getpid()}-{time.time_ns()}"
    backend = _BackendProcess(root, token)
    stored_ids: list[str] = []
    concurrency = {"active": 0, "max": 0}
    try:
        backend.start()
        report.backend_pids.append(int(backend.process.pid))
        deadline = started + (duration_ms / 1000.0)
        while (
            report.completed_cycles < max_cycles
            and time.monotonic() < deadline
        ):
            cycle = report.completed_cycles
            clients = [
                _RpcClient(backend.ws_url, f"cycle-{cycle}-session-{index}", report)
                for index in range(session_count)
            ]
            try:
                await asyncio.gather(*(client.connect() for client in clients))
                previous = list(stored_ids)
                results = await asyncio.gather(
                    *(
                        _drive_session(
                            client,
                            session_index=index,
                            stored_session_id=previous[index] if previous else None,
                            concurrency=concurrency,
                        )
                        for index, client in enumerate(clients)
                    )
                )
                stored_ids = [str(value) for value in results]
                if previous:
                    report.reconnects += len(stored_ids)
            finally:
                await asyncio.gather(*(client.close() for client in clients))

            report.completed_cycles += 1
            report.max_concurrent_turns = max(report.max_concurrent_turns, concurrency["max"])
            if (
                report.backend_restarts < required_restarts
                and report.completed_cycles % max(1, restart_every) == 0
            ):
                backend.stop()
                report.backend_restarts += 1
                backend = _BackendProcess(root, token)
                backend.start()
                report.backend_pids.append(int(backend.process.pid))
    except Exception as exc:
        report.errors.append(str(exc))
    finally:
        backend.stop()

    report.elapsed_ms = int((time.monotonic() - started) * 1000)
    report.accepted = bool(
        not report.errors
        and report.completed_cycles > 0
        and report.turns >= report.session_count
        and report.reconnects >= report.session_count
        and report.backend_restarts >= report.required_restarts
        and report.max_concurrent_turns >= report.session_count
    )
    return report


def _write_report(path: Path | None, report: BackendSoakReport) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the real Hermes headless backend reconnect soak")
    parser.add_argument("--duration-ms", type=int, default=int(os.environ.get("H012_DURATION_MS", DEFAULT_DURATION_MS)))
    parser.add_argument("--sessions", type=int, default=int(os.environ.get("H012_SESSION_COUNT", DEFAULT_SESSION_COUNT)))
    parser.add_argument("--cycles", type=int, default=int(os.environ.get("H012_MAX_CYCLES", DEFAULT_MAX_CYCLES)))
    parser.add_argument("--restarts", type=int, default=int(os.environ.get("H012_RESTARTS", DEFAULT_RESTARTS)))
    parser.add_argument("--restart-every", type=int, default=int(os.environ.get("H012_RESTART_EVERY", DEFAULT_RESTART_EVERY)))
    parser.add_argument("--root", type=Path, default=Path(os.environ["H012_WORK_ROOT"]) if os.environ.get("H012_WORK_ROOT") else None)
    parser.add_argument("--report", type=Path, default=Path(os.environ["H012_REPORT"]) if os.environ.get("H012_REPORT") else None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.duration_ms <= 0 or args.sessions <= 0 or args.cycles <= 0:
        parser.error("duration, sessions and cycles must be positive")
    if args.restarts < 0 or args.restart_every <= 0:
        parser.error("restarts must be non-negative and restart-every must be positive")
    if args.restarts > args.cycles:
        parser.error("restarts cannot exceed cycles")

    temporary_root: tempfile.TemporaryDirectory[str] | None = None
    if args.root is None:
        temporary_root = tempfile.TemporaryDirectory(prefix="hermes-h012-backend-")
        root = Path(temporary_root.name)
    else:
        root = args.root.resolve()
        root.mkdir(parents=True, exist_ok=True)

    try:
        report = asyncio.run(
            _run_scenario(
                root=root,
                duration_ms=args.duration_ms,
                session_count=args.sessions,
                max_cycles=args.cycles,
                required_restarts=args.restarts,
                restart_every=args.restart_every,
            )
        )
        _write_report(args.report, report)
        print(json.dumps(report.to_dict(), indent=2))
        if report.accepted:
            print("H012_HEADLESS_BACKEND_CLASSIFICATION=VALIDATED")
            return 0
        print("H012_HEADLESS_BACKEND_CLASSIFICATION=FAILED")
        return 1
    finally:
        if temporary_root is not None:
            try:
                temporary_root.cleanup()
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
