"""Persistent dashboard compute-host process.

Phase 0 used this module as a deterministic line-JSON spike.  Phase 1 keeps the
same transport and turns it into the long-lived child that owns live AIAgent
objects when ``dashboard.turn_isolation`` is enabled.
"""

from __future__ import annotations

import argparse
import copy
import concurrent.futures
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, cast


_INTERACTIVE_REQUEST_EVENTS = frozenset(
    {
        "approval.request",
        "clarify.request",
        "input.request",
        "secret.request",
        "sudo.request",
        "terminal.read.request",
    }
)
_INTERACTIVE_RESPONSE_METHODS = frozenset(
    {
        "approval.respond",
        "clarify.respond",
        "secret.respond",
        "sudo.respond",
        "terminal.read.respond",
    }
)
_INTERACTIVE_COMPLETE_EVENT = "_host.interactive.complete"


def now_ns() -> int:
    return time.perf_counter_ns()


@dataclass
class SpikeAgent:
    """A deterministic AIAgent-shaped object for pipe/interrupt measurements."""

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
        return {"final_response": final, "messages": messages, "interrupted": interrupted}


@dataclass
class HostSession:
    sid: str
    agent: SpikeAgent
    history_version: int = 0
    running: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


class _HostTransport:
    def __init__(self, emit: Callable[[dict[str, Any]], None]) -> None:
        self._emit = emit

    def write(self, obj: dict) -> bool:
        sid = ""
        event_type = ""
        payload: dict[str, Any] = {}
        try:
            if obj.get("method") == "event":
                event_params = obj.get("params") or {}
                sid = str(event_params.get("session_id") or "")
                event_type = str(event_params.get("type") or "")
                raw_payload = event_params.get("payload")
                payload = dict(raw_payload) if isinstance(raw_payload, dict) else {}
        except Exception:
            sid = ""
            event_type = ""
            payload = {}
        if event_type in _INTERACTIVE_REQUEST_EVENTS:
            self._emit(
                {
                    "type": "interactive.request",
                    "sid": sid,
                    "source_sid": sid,
                    "request_id": str(payload.get("request_id") or ""),
                    "event": event_type,
                    "payload": payload,
                }
            )
            return True
        if event_type == _INTERACTIVE_COMPLETE_EVENT:
            self._emit(
                {
                    "type": "interactive.complete",
                    "sid": sid,
                    "source_sid": sid,
                    "request_id": str(payload.get("request_id") or ""),
                    "event": str(payload.get("event") or ""),
                    "reason": str(payload.get("reason") or "complete"),
                }
            )
            return True
        self._emit({"type": "rpc", "sid": sid, "message": obj})
        return True

    def close(self) -> None:
        return None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _build_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_repo_root()),
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).strip()
    except Exception:
        return "unknown"


class ComputeHost:
    def __init__(
        self,
        *,
        stdout: Any = None,
        max_workers: int | None = None,
        heartbeat_secs: int | float | None = None,
    ) -> None:
        self._stdout = stdout or sys.stdout
        self._write_lock = threading.Lock()
        self._sessions: dict[str, HostSession] = {}
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers or _default_workers(),
            thread_name_prefix="compute-host-turn",
        )
        self._closed = threading.Event()
        self._parent_pid = os.getppid()
        self._boot_id = uuid.uuid4().hex
        self._progress_counter = 0
        self._progress_lock = threading.Lock()
        self._turn_futures: set[concurrent.futures.Future] = set()
        self._turn_futures_lock = threading.Lock()
        # Admission is decided synchronously in the stdin reader, before a turn
        # enters the shared executor. Otherwise executor scheduling can turn a
        # genuinely concurrent request into a later "successor" (or let a
        # control/poller overtake an accepted turn before its worker sets
        # session["running"]). Initial sessions do not yet have a history_lock,
        # so their reservation lives here until the worker creates and claims it.
        self._real_turn_admissions: dict[str, str] = {}
        self._real_turn_admissions_lock = threading.Lock()
        self._transport = _HostTransport(self.emit)
        self._heartbeat_secs = (
            float(heartbeat_secs)
            if heartbeat_secs is not None
            else float(os.environ.get("HERMES_COMPUTE_HOST_HEARTBEAT_SECS") or "15")
        )
        if self._heartbeat_secs > 0:
            threading.Thread(target=self._heartbeat_loop, name="compute-host-heartbeat", daemon=True).start()
            threading.Thread(target=self._parent_guard_loop, name="compute-host-ppid-guard", daemon=True).start()

    def emit(self, frame: dict[str, Any]) -> None:
        frame.setdefault("host_ns", now_ns())
        data = json.dumps(frame, separators=(",", ":"), ensure_ascii=False)
        with self._write_lock:
            print(data, file=self._stdout, flush=True)

    def close(self) -> None:
        self._closed.set()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def shutdown(self, *, reason: str = "shutdown", wait: float = 10.0) -> None:
        self._closed.set()
        self.flush_all_sessions(reason=reason)
        deadline = time.monotonic() + max(0.0, wait)
        while time.monotonic() < deadline:
            with self._turn_futures_lock:
                pending = [f for f in self._turn_futures if not f.done()]
            if not pending:
                break
            time.sleep(0.05)
        self._executor.shutdown(wait=False, cancel_futures=True)

    def flush_all_sessions(self, *, reason: str = "shutdown") -> None:
        try:
            from tui_gateway import server
        except Exception:
            return
        for sid, session in list(getattr(server, "_sessions", {}).items()):
            try:
                server._clear_pending(sid)
                server._drop_pending_approvals(sid)
            except Exception:
                pass
            try:
                server._finalize_session(session, end_reason=f"compute_host_{reason}")
            except Exception:
                pass

    def handle_frame(self, frame: dict[str, Any]) -> None:
        kind = str(frame.get("type") or "")
        if kind == "session.seed":
            self._handle_seed(frame)
        elif kind == "turn.start":
            self._handle_turn_start(frame)
        elif kind == "interrupt":
            self._handle_interrupt(frame)
        elif kind == "reload_mcp":
            self._handle_reload_mcp(frame)
        elif kind == "control":
            self._handle_control(frame)
        elif kind == "interactive.response":
            self._handle_interactive_response(frame)
        elif kind == "shutdown":
            self.emit({"type": "shutdown.ack", "request_id": frame.get("request_id")})
            # Explicit supervisor/test shutdown is a clean child-process close;
            # SIGTERM and orphan paths are the durability flush paths.
            self._closed.set()
            self._executor.shutdown(wait=False, cancel_futures=True)
        else:
            self.emit(
                {
                    "type": "error",
                    "request_id": frame.get("request_id"),
                    "message": f"unknown frame type: {kind}",
                }
            )

    # ── Phase-0 deterministic spike frames ─────────────────────────────

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
        if sid in self._sessions:
            self._handle_spike_turn_start(frame)
            return
        request_id = str(frame.get("request_id") or uuid.uuid4().hex)
        token = uuid.uuid4().hex
        payload = dict(frame)
        payload["request_id"] = request_id
        payload["_compute_host_admission_token"] = token
        admission_kind = "initial"
        payload["_compute_host_admission_kind"] = admission_kind
        try:
            from tui_gateway import server

            # The stdin reader is the protocol-order authority. Reserve the
            # request here, not later in an executor worker, so controls,
            # notification pollers, and subsequently received turns observe it.
            with self._real_turn_admissions_lock:
                if sid in self._real_turn_admissions:
                    self.emit({"type": "turn.error", "sid": sid, "request_id": request_id, "message": "session busy"})
                    return
                session = server._sessions.get(sid)
                if session is None:
                    self._real_turn_admissions[sid] = token
                else:
                    with session["history_lock"]:
                        terminal_owner = session.get("_compute_host_terminal_pending")
                        if terminal_owner:
                            # Once the terminal worker begins serializing its
                            # frame, the parent may legitimately send exactly one
                            # next turn (queued callback or fresh user input). A
                            # request handled earlier while the turn was running
                            # was already rejected synchronously, so executor
                            # scheduling cannot reclassify genuine overlap.
                            if (
                                session.get("_compute_host_terminal_accepting_successor")
                                != terminal_owner
                                or session.get("_compute_host_terminal_successor_pending")
                            ):
                                self.emit({"type": "turn.error", "sid": sid, "request_id": request_id, "message": "session busy"})
                                return
                            admission_kind = "terminal-successor"
                            session["_compute_host_terminal_successor_pending"] = token
                        elif (
                            session.get("running")
                            or session.get("_compute_host_turn_admission")
                            or session.get("_compute_host_terminal_successor_pending")
                            or session.get("_compute_host_control_pending")
                        ):
                            self.emit({"type": "turn.error", "sid": sid, "request_id": request_id, "message": "session busy"})
                            return
                        else:
                            admission_kind = "idle"
                            session["_compute_host_turn_admission"] = token
            payload["_compute_host_admission_kind"] = admission_kind
            future = self._executor.submit(self._run_real_turn, payload)
        except Exception as exc:
            self._release_real_turn_admission(payload)
            self.emit({"type": "turn.error", "sid": sid, "request_id": request_id, "reason": "dispatch_failed", "message": str(exc)})
            return
        with self._turn_futures_lock:
            self._turn_futures.add(future)

        def _done(done: concurrent.futures.Future) -> None:
            with self._turn_futures_lock:
                self._turn_futures.discard(done)
            # Cancellation or an exception before the worker's own finally must
            # not leave an admission that permanently blocks this session.
            self._release_real_turn_admission(payload)

        future.add_done_callback(_done)

    def _release_real_turn_admission(
        self, frame: dict[str, Any], session: dict[str, Any] | None = None
    ) -> None:
        """Release only the reservation owned by ``frame`` (idempotent)."""
        sid = str(frame.get("sid") or "")
        token = str(frame.get("_compute_host_admission_token") or "")
        kind = str(frame.get("_compute_host_admission_kind") or "")
        if not sid or not token:
            return
        if kind == "initial":
            if session is None:
                try:
                    from tui_gateway import server

                    session = server._sessions.get(sid)
                except Exception:
                    session = None
            with self._real_turn_admissions_lock:
                if self._real_turn_admissions.get(sid) == token:
                    self._real_turn_admissions.pop(sid, None)
                if session is not None:
                    with session.get("history_lock", threading.Lock()):
                        if session.get("_compute_host_turn_admission") == token:
                            session.pop("_compute_host_turn_admission", None)
            return
        if session is None:
            try:
                from tui_gateway import server

                session = server._sessions.get(sid)
            except Exception:
                session = None
        if session is None:
            return
        field = (
            "_compute_host_terminal_successor_pending"
            if kind == "terminal-successor"
            else "_compute_host_turn_admission"
        )
        with session.get("history_lock", threading.Lock()):
            if session.get(field) == token:
                session.pop(field, None)

    @staticmethod
    def _mark_real_turn_running(
        server: Any, session: dict, frame: dict[str, Any]
    ) -> int:
        """Claim a validated admission while ``history_lock`` is held."""
        base_history_version = int(session.get("history_version", 0))
        session["running"] = True
        session["_turn_cancel_requested"] = False
        session["last_active"] = time.time()
        server._start_inflight_turn(
            session, frame.get("text") if "text" in frame else frame.get("prompt")
        )
        return base_history_version

    def _handle_spike_turn_start(self, frame: dict[str, Any]) -> None:
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
        future = self._executor.submit(self._run_spike_turn, session, dict(frame))
        with self._turn_futures_lock:
            self._turn_futures.add(future)
        future.add_done_callback(self._turn_futures.discard)

    def _handle_interrupt(self, frame: dict[str, Any]) -> None:
        sid = str(frame.get("sid") or "")
        spike = self._sessions.get(sid)
        if spike is not None:
            spike.agent.interrupt()
            self.emit(
                {
                    "type": "interrupt.ack",
                    "sid": sid,
                    "request_id": frame.get("request_id"),
                    "applied": True,
                    "applied_ns": now_ns(),
                }
            )
            return
        initial_admission_cancelled = False
        with self._real_turn_admissions_lock:
            if sid in self._real_turn_admissions:
                self._real_turn_admissions.pop(sid, None)
                initial_admission_cancelled = True
        try:
            from tui_gateway import server

            session = server._sessions.get(sid)
            if session is None:
                self.emit(
                    {
                        "type": "interrupt.ack",
                        "sid": sid,
                        "request_id": frame.get("request_id"),
                        "applied": initial_admission_cancelled,
                    }
                )
                return
            with session.get("history_lock", threading.Lock()):
                # Cancel admitted-but-not-running work as well as a live turn.
                # Workers validate these request-owned tokens before claiming
                # running=True, so removing them makes queued/waiting futures
                # expire instead of starting after an acknowledged stop.
                session.pop("_compute_host_terminal_successor_pending", None)
                session.pop("_compute_host_turn_admission", None)
                session["_turn_cancel_requested"] = True
                session["queued_prompt"] = None
            agent = session.get("agent")
            if agent is not None and hasattr(agent, "interrupt"):
                agent.interrupt()
            server._clear_pending(sid)
            try:
                from tools.approval import resolve_gateway_approval

                session_key = str(session.get("session_key") or "")
                if session_key:
                    resolve_gateway_approval(session_key, "deny", resolve_all=True)
            except Exception:
                pass
            server._drop_pending_approvals(sid)
            self.emit({"type": "interrupt.ack", "sid": sid, "request_id": frame.get("request_id"), "applied": True, "applied_ns": now_ns()})
        except Exception as exc:
            self.emit({"type": "interrupt.ack", "sid": sid, "request_id": frame.get("request_id"), "applied": False, "message": str(exc)})

    def _run_spike_turn(self, session: HostSession, frame: dict[str, Any]) -> None:
        request_id = frame.get("request_id") or uuid.uuid4().hex
        prompt = str(frame.get("prompt") or frame.get("text") or "")
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
            self._bump_progress()
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
            self._bump_progress()
            self.emit(
                {
                    "type": "turn.end",
                    "sid": session.sid,
                    "request_id": request_id,
                    "history_version": history_version,
                    "base_history_version": max(0, history_version - 1),
                    "history": list(result.get("messages") or []),
                    "message_count": len(result.get("messages") or []),
                    "interrupted": bool(result.get("interrupted")),
                    "ended_ns": now_ns(),
                }
            )
        except Exception as exc:  # pragma: no cover - defensive host boundary
            with session.lock:
                session.running = False
            self.emit({"type": "turn.error", "sid": session.sid, "request_id": request_id, "message": str(exc)})

    # ── Real dashboard turn path ───────────────────────────────────────

    def _run_real_turn(self, frame: dict[str, Any]) -> None:
        sid = str(frame.get("sid") or "")
        request_id = str(frame.get("request_id") or uuid.uuid4().hex)
        session: dict[str, Any] | None = None
        turn_claimed = False
        terminal_metadata: dict[str, Any] | None = None
        if not sid:
            self.emit({"type": "turn.error", "sid": sid, "request_id": request_id, "message": "sid required"})
            return
        try:
            from tui_gateway import server

            session = self._ensure_server_session(server, frame)
            admission_token = str(frame.get("_compute_host_admission_token") or "")
            admission_kind = str(frame.get("_compute_host_admission_kind") or "")
            if admission_kind == "initial":
                # Keep the host-level reservation and the freshly initialized
                # session reservation under one lock order until running=True.
                with self._real_turn_admissions_lock:
                    with session["history_lock"]:
                        if (
                            self._real_turn_admissions.get(sid) != admission_token
                            or session.get("_compute_host_turn_admission")
                            != admission_token
                        ):
                            raise RuntimeError("turn admission expired")
                        self._real_turn_admissions.pop(sid, None)
                        session.pop("_compute_host_turn_admission", None)
                        base_history_version = self._mark_real_turn_running(
                            server, session, frame
                        )
                        turn_claimed = True
            elif admission_kind in {"idle", "terminal-successor"}:
                admission_field = (
                    "_compute_host_terminal_successor_pending"
                    if admission_kind == "terminal-successor"
                    else "_compute_host_turn_admission"
                )
                while True:
                    wait_for_terminal = False
                    with session["history_lock"]:
                        if session.get(admission_field) != admission_token:
                            raise RuntimeError("turn admission expired")
                        if (
                            admission_kind == "terminal-successor"
                            and session.get("_compute_host_terminal_pending")
                        ):
                            wait_for_terminal = True
                        else:
                            if (
                                session.get("running")
                                or session.get("_compute_host_control_pending")
                            ):
                                raise RuntimeError("session busy")
                            session.pop(admission_field, None)
                            base_history_version = self._mark_real_turn_running(
                                server, session, frame
                            )
                            turn_claimed = True
                            break
                    if wait_for_terminal and self._closed.wait(0.005):
                        raise RuntimeError("compute host shutting down")
            else:
                # Direct unit-level callers predate protocol-time admission.
                # Production turn.start frames always carry one of the kinds
                # above; preserve the narrow test seam without weakening wire
                # ordering semantics.
                while True:
                    with session["history_lock"]:
                        terminal_pending = bool(
                            session.get("_compute_host_terminal_pending")
                        )
                        if not terminal_pending:
                            if session.get("running"):
                                raise RuntimeError("session busy")
                            base_history_version = self._mark_real_turn_running(
                                server, session, frame
                            )
                            turn_claimed = True
                            break
                    if self._closed.wait(0.005):
                        raise RuntimeError("compute host shutting down")
            self.emit({"type": "turn.started", "sid": sid, "request_id": request_id, "started_ns": now_ns()})
            try:
                server._ensure_session_db_row(session)
            except Exception:
                pass
            try:
                import hermes_undo

                hermes_undo.on_user_message_appended(session["session_key"])
            except Exception:
                pass
            try:
                server._persist_branch_seed(session)
            except Exception:
                pass
            text = frame.get("text") if "text" in frame else frame.get("prompt", "")
            server._run_prompt_submit(request_id, sid, session, text)
            # server._run_prompt_submit runs the turn on session["_run_thread"]
            # and, in that thread's tail, can chain a SUCCESSOR turn — a queued
            # prompt, a /goal continuation, or a post-turn completion — by
            # calling _run_prompt_submit again, which installs a fresh
            # session["_run_thread"] and re-arms session["running"] before the
            # observed thread exits. Joining that first thread once and
            # snapshotting would race the live successor: we'd emit turn.end
            # with partial history and mark the source idle while a newer turn
            # is still writing. Wait for the whole chain to reach stable
            # quiescence, snapshotting under the lock that proves it.
            snapshot = self._await_turn_quiescence(session, request_id)
            history_version = snapshot["history_version"]
            history = snapshot["history"]
            message_count = snapshot["message_count"]
            interrupted = snapshot["interrupted"]
            session_key = snapshot["session_key"]
            session_info = server._session_info(session.get("agent"), session)
            terminal_metadata = {
                "history_version": history_version,
                "session_key": session_key,
                "base_history_version": base_history_version,
                "message_count": message_count,
                "history": history,
                "interrupted": interrupted,
                "session_info": session_info,
                "session_info_emitted": True,
            }
            self._bump_progress()
            with session["history_lock"]:
                if session.get("_compute_host_terminal_pending") == request_id:
                    session["_compute_host_terminal_accepting_successor"] = request_id
            self.emit(
                {
                    "type": "turn.end",
                    "sid": sid,
                    "request_id": request_id,
                    **terminal_metadata,
                    "ended_ns": now_ns(),
                }
            )
        except Exception as exc:
            try:
                from tui_gateway import server

                if session is not None and turn_claimed:
                    with session.get("history_lock", threading.Lock()):
                        session["running"] = False
                        if (
                            session.get("_compute_host_terminal_accepting_successor")
                            == request_id
                        ):
                            session.pop(
                                "_compute_host_terminal_accepting_successor", None
                            )
                        server._clear_inflight_turn(session)
            except Exception:
                pass
            error_frame = {
                "type": "turn.error",
                "sid": sid,
                "request_id": request_id,
                "reason": "exception",
                "message": str(exc),
            }
            if terminal_metadata is not None:
                # If only turn.end serialization failed, the completed child
                # history is still authoritative. Mirror it through the fallback
                # terminal frame so the parent can CAS-apply version N before it
                # dispatches a queued successor from that same request.
                error_frame.update(terminal_metadata)
                if session is not None:
                    with session.get("history_lock", threading.Lock()):
                        if (
                            session.get("_compute_host_terminal_pending")
                            == request_id
                        ):
                            session[
                                "_compute_host_terminal_accepting_successor"
                            ] = request_id
            self.emit(error_frame)
        finally:
            self._release_real_turn_admission(frame, session)
            if session is not None:
                # Keep the terminal barrier raised through either the successful
                # turn.end write or the replacement turn.error attempt.  Clearing
                # it earlier re-opens the snapshot/order race on emit failures.
                with session.get("history_lock", threading.Lock()):
                    if (
                        session.get("_compute_host_terminal_accepting_successor")
                        == request_id
                    ):
                        session.pop(
                            "_compute_host_terminal_accepting_successor", None
                        )
                    if session.get("_compute_host_terminal_pending") == request_id:
                        session.pop("_compute_host_terminal_pending", None)

    def _await_turn_quiescence(
        self, session: dict, terminal_owner: str
    ) -> dict[str, Any]:
        """Block until the session's turn-thread chain is *stably* settled,
        then snapshot history under the lock that proves it.

        A single ``prompt.submit`` can expand into a chain of turns: the
        dispatch thread, in its tail, drains a queued prompt / fires a /goal
        continuation / delivers a post-turn completion by calling
        ``_run_prompt_submit`` again, each time replacing
        ``session["_run_thread"]`` and re-setting ``session["running"]``
        *before* the thread we observed exits. So one observe-join-snapshot is
        not enough — we loop: observe the current thread, join it (never the
        current thread, and never while holding ``history_lock`` so the turn
        thread can still take it), then re-verify under the lock that the turn
        is no longer running AND that no newer thread replaced the one we
        joined. Only a state satisfying both is a real settle point. Before
        releasing that proof lock, we raise a request-owned terminal barrier;
        notification pollers defer and overlapping parent requests wait until the
        caller has synchronously serialized ``turn.end``. This closes
        the post-snapshot/pre-emission race without holding ``history_lock``
        across session-info work or pipe I/O. The loop cannot busy-spin: every
        iteration either blocks in ``join()`` until a thread finishes, advances
        to a newly installed successor thread, or briefly yields during the
        assignment/start handoff. Shutdown does not bypass this proof and emit
        an incomplete terminal frame; the host's existing shutdown timeout
        bounds how long its supervisor waits.
        """
        current = threading.current_thread()
        while True:
            run_thread = session.get("_run_thread")
            if run_thread is current:
                raise RuntimeError("compute host cannot wait on its own turn thread")

            can_join = run_thread is not None and hasattr(run_thread, "join")
            if can_join:
                try:
                    run_thread.join()
                except RuntimeError:
                    # server._run_prompt_submit assigns session["_run_thread"]
                    # a beat before it calls .start(). Observing a successor
                    # thread inside that window makes join() raise "cannot join
                    # thread before it is started" — yield briefly and
                    # re-observe once it is started (or replaced again).
                    time.sleep(0.005)
                    continue

            with session["history_lock"]:
                latest = session.get("_run_thread")
                running = bool(session.get("running"))
                control_pending = bool(
                    session.get("_compute_host_control_pending")
                )
                alive = (
                    run_thread is not None
                    and hasattr(run_thread, "is_alive")
                    and run_thread.is_alive()
                )
                if (
                    latest is run_thread
                    and not alive
                    and not running
                    and not control_pending
                ):
                    # Stable: the thread we observed is still the installed one,
                    # it has exited (or there was none), and the turn is no
                    # longer running — no successor took over. Raise a terminal
                    # barrier under this same lock before exposing the snapshot:
                    # otherwise the autonomous notification poller can claim the
                    # idle session in the snapshot→turn.end window.
                    session["_compute_host_terminal_pending"] = terminal_owner
                    return self._snapshot_turn_state(session)

            # A successor replaced the observed thread, or the dispatcher is in
            # the brief running-without-a-started-thread handoff window. Recheck
            # until the chain is genuinely idle. In particular, shutdown must
            # not manufacture an incomplete turn.end while a daemon successor
            # still owns history; the host's existing shutdown timeout controls
            # how long the supervisor waits for this worker future.
            if control_pending or not can_join:
                self._closed.wait(0.005)

    def _snapshot_turn_state(self, session: dict) -> dict[str, Any]:
        """Snapshot the fields ``turn.end`` needs.

        MUST be called while holding ``session["history_lock"]``.
        """
        history = list(session.get("history") or [])
        return {
            "history_version": int(session.get("history_version", 0)),
            "history": history,
            "message_count": len(history),
            "interrupted": bool(session.get("_turn_cancel_requested")),
            "session_key": str(session.get("session_key") or ""),
        }

    def _ensure_server_session(self, server: Any, frame: dict[str, Any]) -> dict:
        sid = str(frame.get("sid") or "")
        key = str(frame.get("session_key") or sid)
        session = server._sessions.get(sid)
        if session is not None:
            if frame.get("_compute_host_admission_kind") == "initial":
                admission_token = str(
                    frame.get("_compute_host_admission_token") or ""
                )
                with self._real_turn_admissions_lock:
                    if self._real_turn_admissions.get(sid) == admission_token:
                        with session["history_lock"]:
                            session["_compute_host_turn_admission"] = admission_token
            session["transport"] = self._transport
            if frame.get("cols") is not None:
                session["cols"] = int(frame.get("cols") or 80)
            if frame.get("cwd"):
                session["cwd"] = str(frame.get("cwd"))
            if frame.get("profile_home"):
                session["profile_home"] = str(frame.get("profile_home"))
            if isinstance(frame.get("attached_images"), list):
                session["attached_images"] = list(frame.get("attached_images") or [])
            incoming_history = frame.get("history")
            try:
                incoming_version = int(frame.get("history_version") or 0)
            except (TypeError, ValueError):
                incoming_version = 0
            with session.get("history_lock", threading.Lock()):
                current_version = int(session.get("history_version", 0))
                # A parent-side mutation that is provably newer (for example a
                # recovered/reloaded source) becomes the next turn's base. A
                # stale parent frame never rolls the authoritative child back.
                if (
                    not session.get("running")
                    and isinstance(incoming_history, list)
                    and incoming_version > current_version
                    and all(isinstance(message, dict) for message in incoming_history)
                ):
                    session["history"] = copy.deepcopy(incoming_history)
                    session["history_version"] = incoming_version
            return session

        raw_history = frame.get("history")
        history = cast(list[Any], raw_history) if isinstance(raw_history, list) else []
        profile_home = str(frame.get("profile_home") or "")
        initial_session_state = None
        if frame.get("_compute_host_admission_kind") == "initial":
            initial_session_state = {
                "_compute_host_turn_admission": str(
                    frame.get("_compute_host_admission_token") or ""
                )
            }
        session_db = None
        home_token = None
        try:
            if profile_home:
                from hermes_constants import set_hermes_home_override
                from hermes_state import SessionDB

                home_token = set_hermes_home_override(profile_home)
                session_db = SessionDB(db_path=Path(profile_home) / "state.db")
            agent = server._make_agent(
                sid,
                key,
                session_id=key,
                model_override=frame.get("model_override"),
                reasoning_config_override=frame.get("reasoning_config_override"),
                service_tier_override=frame.get("service_tier_override"),
                platform_override=frame.get("source"),
                session_db=session_db,
            )
            try:
                from tui_gateway.transport import bind_transport, reset_transport

                token = bind_transport(self._transport)
                try:
                    server._init_session(
                        sid,
                        key,
                        agent,
                        list(history),
                        cols=int(frame.get("cols") or 80),
                        cwd=str(frame.get("cwd") or "") or None,
                        session_db=session_db,
                        source=frame.get("source"),
                        profile_home=profile_home or None,
                        initial_session_state=initial_session_state,
                    )
                finally:
                    reset_transport(token)
            except Exception:
                # If _init_session's side machinery (slash worker, approval
                # notify) is unavailable, keep a minimal host-owned session
                # rather than failing the turn after the expensive agent build
                # succeeded. HERMES_HOME remains bound here so every fallback
                # config read belongs to the same remote profile.
                server._sessions[sid] = {
                    "agent": agent,
                    "session_key": key,
                    "history": list(history),
                    "history_lock": threading.Lock(),
                    "history_version": int(frame.get("history_version") or 0),
                    "inflight_turn": None,
                    "created_at": time.time(),
                    "last_active": time.time(),
                    "running": False,
                    "attached_images": [],
                    "image_counter": 0,
                    "cwd": str(frame.get("cwd") or os.getcwd()),
                    "cols": int(frame.get("cols") or 80),
                    "slash_worker": None,
                    "show_reasoning": server._load_show_reasoning(),
                    "tool_progress_mode": server._load_tool_progress_mode(),
                    "edit_snapshots": {},
                    "tool_started_at": {},
                    "model_override": frame.get("model_override"),
                    "source": server._resolve_session_source(frame.get("source")),
                    "transport": self._transport,
                }
                if profile_home:
                    server._sessions[sid]["profile_home"] = profile_home
                if initial_session_state:
                    server._sessions[sid].update(initial_session_state)
        finally:
            if home_token is not None:
                try:
                    from hermes_constants import reset_hermes_home_override

                    reset_hermes_home_override(home_token)
                except Exception:
                    pass
        session = server._sessions[sid]
        session["transport"] = self._transport
        with session.get("history_lock", threading.Lock()):
            session["history_version"] = max(
                int(session.get("history_version", 0)),
                int(frame.get("history_version") or 0),
            )
        session["profile_home"] = profile_home or session.get("profile_home")
        if isinstance(frame.get("attached_images"), list):
            session["attached_images"] = list(frame.get("attached_images") or [])
        if frame.get("model_override") is not None:
            session["model_override"] = frame.get("model_override")
        return session

    def _handle_interactive_response(self, frame: dict[str, Any]) -> None:
        """Apply a parent-owned UI response to the child-owned real waiter."""
        sid = str(frame.get("sid") or "")
        control_id = str(frame.get("request_id") or "")
        interactive_request_id = str(frame.get("interactive_request_id") or "")
        method = str(frame.get("method") or "")
        if not sid or not interactive_request_id or method not in _INTERACTIVE_RESPONSE_METHODS:
            self.emit(
                {
                    "type": "interactive.response.error",
                    "sid": sid,
                    "request_id": control_id,
                    "interactive_request_id": interactive_request_id,
                    "message": "invalid interactive response frame",
                }
            )
            return
        try:
            from tui_gateway import server

            params = dict(frame.get("params") or {})
            params["request_id"] = interactive_request_id
            params["session_id"] = sid
            response = server.handle_request(
                {
                    "id": control_id,
                    "method": method,
                    "params": params,
                }
            )
            if not isinstance(response, dict) or response.get("error"):
                error = (response or {}).get("error") if isinstance(response, dict) else None
                self.emit(
                    {
                        "type": "interactive.response.error",
                        "sid": sid,
                        "request_id": control_id,
                        "interactive_request_id": interactive_request_id,
                        "message": str((error or {}).get("message") or "interactive response failed"),
                        "response": response,
                    }
                )
                return
            self.emit(
                {
                    "type": "interactive.response.ack",
                    "sid": sid,
                    "request_id": control_id,
                    "interactive_request_id": interactive_request_id,
                    "response": response,
                }
            )
        except Exception as exc:
            self.emit(
                {
                    "type": "interactive.response.error",
                    "sid": sid,
                    "request_id": control_id,
                    "interactive_request_id": interactive_request_id,
                    "message": str(exc),
                }
            )

    def _handle_reload_mcp(self, frame: dict[str, Any]) -> None:
        sid = str(frame.get("sid") or "")
        request_id = frame.get("request_id")
        try:
            from tui_gateway import server

            resp = server.handle_request({"id": request_id, "method": "reload.mcp", "params": {"session_id": sid, "confirm": True}})
            self.emit({"type": "reload_mcp.ack", "sid": sid, "request_id": request_id, "response": resp})
        except Exception as exc:
            self.emit({"type": "control.error", "sid": sid, "request_id": request_id, "message": str(exc)})

    def _handle_control(self, frame: dict[str, Any]) -> None:
        sid = str(frame.get("sid") or "")
        request_id = frame.get("request_id")
        route_name = str(frame.get("route_name") or "")
        session: dict[str, Any] | None = None
        control_token: str | None = None
        try:
            from tui_gateway import server
            from tui_gateway.host_supervisor import MUTATOR_ROUTE_TABLE

            route = MUTATOR_ROUTE_TABLE.get(route_name)
            if route is None:
                self.emit({"type": "control.error", "sid": sid, "request_id": request_id, "message": f"unclassified route: {route_name}"})
                return
            session = server._sessions.get(sid)
            if session is None:
                self.emit({"type": "control.error", "sid": sid, "request_id": request_id, "message": "session not found"})
                return
            if route == "idle-gated":
                candidate = uuid.uuid4().hex
                with session["history_lock"]:
                    run_thread = session.get("_run_thread")
                    run_thread_alive = bool(
                        run_thread is not None
                        and hasattr(run_thread, "is_alive")
                        and run_thread.is_alive()
                    )
                    control_busy = bool(
                        session.get("running")
                        or run_thread_alive
                        or session.get("_compute_host_terminal_pending")
                        or session.get("_compute_host_terminal_successor_pending")
                        or session.get("_compute_host_turn_admission")
                        or session.get("_compute_host_control_pending")
                    )
                    if not control_busy:
                        # Reserve the whole mutation, not merely its pre-check.
                        # The stdin reader cannot admit another frame while this
                        # synchronous handler runs, and autonomous pollers/turn
                        # workers honor this field under the same history lock.
                        session["_compute_host_control_pending"] = candidate
                        control_token = candidate
                if control_busy:
                    self.emit({"type": "control.error", "sid": sid, "request_id": request_id, "message": "session busy"})
                    return
            if route_name == "reload.mcp":
                self._handle_reload_mcp({**frame, "type": "reload_mcp"})
                return
            command = str(frame.get("command") or "")
            output = ""
            if command:
                output = server._mirror_slash_side_effects(sid, session, command)
            with session["history_lock"]:
                history_version = int(session.get("history_version", 0))
                message_count = len(session.get("history") or [])
                session_key = str(session.get("session_key") or "")
            self.emit(
                {
                    "type": "control.ack",
                    "sid": sid,
                    "request_id": request_id,
                    "route_name": route_name,
                    "output": output,
                    "session_key": session_key,
                    "history_version": history_version,
                    "message_count": message_count,
                    "session_info": server._session_info(session.get("agent"), session),
                }
            )
        except Exception as exc:
            self.emit({"type": "control.error", "sid": sid, "request_id": request_id, "message": str(exc)})
        finally:
            if session is not None and control_token is not None:
                with session.get("history_lock", threading.Lock()):
                    if session.get("_compute_host_control_pending") == control_token:
                        session.pop("_compute_host_control_pending", None)

    def _bump_progress(self) -> None:
        with self._progress_lock:
            self._progress_counter += 1

    def _heartbeat_loop(self) -> None:
        while not self._closed.wait(self._heartbeat_secs):
            with self._turn_futures_lock:
                active_turns = sum(1 for f in self._turn_futures if not f.done())
            with self._progress_lock:
                counter = self._progress_counter
            self.emit(
                {
                    "type": "hb",
                    "active_turns": active_turns,
                    "progress_counter": counter,
                    "rss_mb": _rss_mb(os.getpid()),
                }
            )

    def _parent_guard_loop(self) -> None:
        while not self._closed.wait(1.0):
            ppid = os.getppid()
            if ppid in {0, 1} or (self._parent_pid and ppid != self._parent_pid):
                self.emit({"type": "orphan", "old_ppid": self._parent_pid, "ppid": ppid})
                self.shutdown(reason="orphan")
                os._exit(0)


def _rss_mb(pid: int) -> float:
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True, stdin=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2).strip()
        return int(out.splitlines()[-1].strip()) / 1024.0 if out else 0.0
    except Exception:
        return 0.0


def _default_workers() -> int:
    try:
        return max(2, int(os.environ.get("HERMES_TUI_RPC_POOL_WORKERS") or "8"))
    except (TypeError, ValueError):
        return 8


def run_host(stdin: Any = None, stdout: Any = None) -> None:
    os.environ["HERMES_COMPUTE_HOST_CHILD"] = "1"
    stdin = stdin or sys.stdin
    host = ComputeHost(stdout=stdout or sys.stdout)
    shutting_down = threading.Event()

    def _signal_handler(_signum, _frame) -> None:
        if shutting_down.is_set():
            return
        shutting_down.set()
        host.shutdown(reason="sigterm")
        raise SystemExit(0)

    try:
        signal.signal(signal.SIGTERM, _signal_handler)
        signal.signal(signal.SIGINT, _signal_handler)
    except Exception:
        pass

    host.emit(
        {
            "type": "hello",
            "host_pid": os.getpid(),
            "boot_id": host._boot_id,
            "build_sha": _build_sha(),
            "cwd": os.getcwd(),
            "hermes_home": os.environ.get("HERMES_HOME", ""),
        }
    )

    def _reader() -> None:
        for raw in stdin:
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
            if frame.get("type") == "shutdown":
                os._exit(0)
            if host._closed.is_set():
                break

    reader = threading.Thread(target=_reader, name="compute-host-control-reader", daemon=True)
    reader.start()
    try:
        while not host._closed.wait(0.2):
            if not reader.is_alive():
                break
    finally:
        host.shutdown(reason="stdin_closed", wait=2.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dashboard compute-host process")
    parser.parse_args(argv)
    run_host()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
