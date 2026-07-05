"""Minimal dashboard compute-host process for the Phase-0 isolation spike.

This module is intentionally small and line-JSON based, mirroring the existing
``_SlashWorker`` subprocess shape. It is not wired into runtime dispatch yet;
Phase 0 uses it to measure the process-boundary costs before Phase 1 moves the
real dashboard turn path behind the host.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable


def now_ns() -> int:
    return time.perf_counter_ns()


@dataclass
class SpikeAgent:
    """A deterministic AIAgent-shaped object for pipe/interrupt measurements.

    The spike needs stable transport numbers, so this agent exercises the same
    ``run_conversation(..., stream_callback=...)`` and ``interrupt()`` contract
    the dashboard turn path uses without making provider/network calls.
    """

    session_id: str
    history: list[dict[str, str]] = field(default_factory=list)
    _interrupt: threading.Event = field(default_factory=threading.Event)

    def clear_interrupt(self) -> None:
        self._interrupt.clear()

    def interrupt(self) -> None:
        self._interrupt.set()

    def run_conversation(
        self,
        prompt: str,
        *,
        conversation_history: list[dict[str, str]] | None = None,
        stream_callback: Callable[[str], None] | None = None,
        delta_count: int = 24,
        delay_s: float = 0.001,
    ) -> dict[str, Any]:
        base_history = list(conversation_history if conversation_history is not None else self.history)
        chunks: list[str] = []
        interrupted = False
        for index in range(max(0, int(delta_count))):
            if self._interrupt.is_set():
                interrupted = True
                break
            chunk = f"{self.session_id}:{prompt}:{index:04d} "
            chunks.append(chunk)
            if stream_callback is not None:
                stream_callback(chunk)
            if delay_s > 0:
                time.sleep(delay_s)
        if self._interrupt.is_set():
            interrupted = True
        final = "".join(chunks)
        if interrupted:
            final += "[interrupted]"
        messages = [
            *base_history,
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": final},
        ]
        self.history = messages
        return {
            "final_response": final,
            "messages": messages,
            "interrupted": interrupted,
        }


@dataclass
class HostSession:
    sid: str
    agent: SpikeAgent
    history_version: int = 0
    running: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


class ComputeHost:
    def __init__(self, *, stdout: Any = None, max_workers: int | None = None) -> None:
        self._stdout = stdout or sys.stdout
        self._write_lock = threading.Lock()
        self._sessions: dict[str, HostSession] = {}
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers or _default_workers(),
            thread_name_prefix="compute-host-turn",
        )
        self._closed = threading.Event()

    def emit(self, frame: dict[str, Any]) -> None:
        frame.setdefault("host_ns", now_ns())
        data = json.dumps(frame, separators=(",", ":"), ensure_ascii=False)
        with self._write_lock:
            print(data, file=self._stdout, flush=True)

    def close(self) -> None:
        self._closed.set()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def handle_frame(self, frame: dict[str, Any]) -> None:
        kind = str(frame.get("type") or "")
        if kind == "session.seed":
            self._handle_seed(frame)
        elif kind == "turn.start":
            self._handle_turn_start(frame)
        elif kind == "interrupt":
            self._handle_interrupt(frame)
        elif kind == "shutdown":
            self.emit({"type": "shutdown.ack", "request_id": frame.get("request_id")})
            self.close()
        else:
            self.emit(
                {
                    "type": "error",
                    "request_id": frame.get("request_id"),
                    "message": f"unknown frame type: {kind}",
                }
            )

    def _handle_seed(self, frame: dict[str, Any]) -> None:
        sid = str(frame.get("sid") or "")
        if not sid:
            self.emit({"type": "error", "request_id": frame.get("request_id"), "message": "sid required"})
            return
        history = frame.get("history")
        if not isinstance(history, list):
            history = []
        self._sessions[sid] = HostSession(sid=sid, agent=SpikeAgent(sid, list(history)))
        self.emit({"type": "session.seeded", "sid": sid, "request_id": frame.get("request_id")})

    def _handle_turn_start(self, frame: dict[str, Any]) -> None:
        sid = str(frame.get("sid") or "")
        session = self._sessions.get(sid)
        if session is None:
            self.emit({"type": "turn.error", "sid": sid, "request_id": frame.get("request_id"), "message": "unknown session"})
            return
        with session.lock:
            if session.running:
                self.emit({"type": "turn.error", "sid": sid, "request_id": frame.get("request_id"), "message": "session busy"})
                return
            session.running = True
        self._executor.submit(self._run_turn, session, dict(frame))

    def _handle_interrupt(self, frame: dict[str, Any]) -> None:
        sid = str(frame.get("sid") or "")
        session = self._sessions.get(sid)
        if session is None:
            self.emit({"type": "interrupt.ack", "sid": sid, "request_id": frame.get("request_id"), "applied": False})
            return
        session.agent.interrupt()
        self.emit(
            {
                "type": "interrupt.ack",
                "sid": sid,
                "request_id": frame.get("request_id"),
                "applied": True,
                "applied_ns": now_ns(),
            }
        )

    def _run_turn(self, session: HostSession, frame: dict[str, Any]) -> None:
        request_id = frame.get("request_id") or uuid.uuid4().hex
        prompt = str(frame.get("prompt") or "")
        try:
            delta_count = int(frame.get("delta_count", 24))
        except (TypeError, ValueError):
            delta_count = 24
        try:
            delay_s = float(frame.get("delay_s", 0.001))
        except (TypeError, ValueError):
            delay_s = 0.001
        with session.lock:
            history = list(session.agent.history)
        session.agent.clear_interrupt()
        self.emit({"type": "turn.started", "sid": session.sid, "request_id": request_id, "started_ns": now_ns()})

        def stream(delta: str) -> None:
            self.emit(
                {
                    "type": "delta",
                    "sid": session.sid,
                    "request_id": request_id,
                    "text": delta,
                    "emitted_ns": now_ns(),
                }
            )

        try:
            result = session.agent.run_conversation(
                prompt,
                conversation_history=history,
                stream_callback=stream,
                delta_count=delta_count,
                delay_s=delay_s,
            )
            with session.lock:
                session.history_version += 1
                session.running = False
                history_version = session.history_version
            self.emit(
                {
                    "type": "turn.end",
                    "sid": session.sid,
                    "request_id": request_id,
                    "history_version": history_version,
                    "message_count": len(result.get("messages") or []),
                    "interrupted": bool(result.get("interrupted")),
                    "ended_ns": now_ns(),
                }
            )
        except Exception as exc:  # pragma: no cover - defensive host boundary
            with session.lock:
                session.running = False
            self.emit(
                {
                    "type": "turn.error",
                    "sid": session.sid,
                    "request_id": request_id,
                    "message": str(exc),
                }
            )


def _default_workers() -> int:
    try:
        return max(2, int(os.environ.get("HERMES_COMPUTE_HOST_WORKERS") or "8"))
    except (TypeError, ValueError):
        return 8


def run_host() -> None:
    host = ComputeHost()
    host.emit(
        {
            "type": "hello",
            "host_pid": os.getpid(),
            "boot_id": uuid.uuid4().hex,
            "cwd": os.getcwd(),
            "hermes_home": os.environ.get("HERMES_HOME", ""),
        }
    )
    for raw in sys.stdin:
        if host._closed.is_set():
            break
        try:
            frame = json.loads(raw)
        except json.JSONDecodeError as exc:
            host.emit({"type": "error", "message": f"invalid json: {exc}"})
            continue
        if not isinstance(frame, dict):
            host.emit({"type": "error", "message": "frame must be an object"})
            continue
        host.handle_frame(frame)
        if host._closed.is_set():
            break
    host.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase-0 dashboard compute-host spike process")
    parser.parse_args(argv)
    run_host()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
