"""Persistent dashboard compute-host child: owns live AIAgent objects when
``dashboard.turn_isolation`` is enabled; frames are line-JSON over stdin/stdout."""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Collection

from tui_gateway.host_supervisor import MUTATOR_ROUTE_TABLE, _build_sha


def now_ns() -> int:
    return time.perf_counter_ns()


class _HostTransport:
    def __init__(self, emit: Callable[[dict[str, Any]], None]) -> None:
        self._emit = emit

    def write(self, obj: dict) -> bool:
        sid = ""
        with contextlib.suppress(Exception):
            if obj.get("method") == "event":
                sid = str(((obj.get("params") or {}).get("session_id")) or "")
        self._emit({"type": "rpc", "sid": sid, "message": obj})
        return True

    def close(self) -> None:
        return None


# Slice of ``ComputeHost.shutdown``'s budget held back for the post-drain finalize: the
# supervisor SIGKILLs the host 10s (= default ``wait``) after SIGTERM, so a drain that ate
# the whole budget would leave the flush racing that kill and persist nothing.
_FLUSH_RESERVE_SECS = 1.0

# Fallback control.error text when a routed server method returns an error without a message.
_CONTROL_FAILURES = {
    "session.save": "session save failed", "session.compress": "session compression failed"}


class ComputeHost:
    # frame ``type`` -> handler method name (resolved per call so monkeypatches take effect).
    _FRAME_HANDLERS: dict[str, str] = {
        "turn.start": "_handle_turn_start", "interrupt": "_handle_interrupt",
        "respond": "_handle_respond", "reload_mcp": "_handle_reload_mcp",
        "control": "_handle_control", "shutdown": "_handle_shutdown"}

    def __init__(
        self, *, stdout: Any = None, max_workers: int | None = None,
        heartbeat_secs: int | float | None = None) -> None:
        self._stdout = stdout or sys.stdout
        self._write_lock = threading.Lock()
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers or _default_workers(), thread_name_prefix="compute-host-turn")
        self._closed = threading.Event()
        self._parent_pid = os.getppid()
        self._boot_id = uuid.uuid4().hex
        self._progress_counter = 0
        self._progress_lock = threading.Lock()
        # Future -> the ``sid`` whose turn it runs; ``shutdown`` leaves live sids unfinalized.
        self._turn_futures: dict[concurrent.futures.Future, str] = {}
        self._turn_futures_lock = threading.Lock()
        self._transport = _HostTransport(self.emit)
        self._heartbeat_secs = (
            float(heartbeat_secs) if heartbeat_secs is not None
            else float(os.environ.get("HERMES_COMPUTE_HOST_HEARTBEAT_SECS") or "15"))
        if self._heartbeat_secs > 0:
            for target, name in (
                (self._heartbeat_loop, "compute-host-heartbeat"),
                (self._parent_guard_loop, "compute-host-ppid-guard")):
                threading.Thread(target=target, name=name, daemon=True).start()

    def emit(self, frame: dict[str, Any]) -> None:
        frame.setdefault("host_ns", now_ns())
        data = json.dumps(frame, separators=(",", ":"), ensure_ascii=False)
        with self._write_lock:
            print(data, file=self._stdout, flush=True)

    def _reply(self, kind: str, sid: str, request_id: Any, **extra: Any) -> None:
        """Emit a per-session frame keyed by the request it answers."""
        self.emit({"type": kind, "sid": sid, "request_id": request_id, **extra})

    def close(self) -> None:
        self._closed.set()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def shutdown(self, *, reason: str = "shutdown", wait: float = 10.0) -> None:
        """Drain in-flight turns, then finalize every session.

        ``_finalize_session`` is a one-shot latch, so finalizing before the drain would spend
        it mid-turn and release the lease. ``_FLUSH_RESERVE_SECS`` (at most half of ``wait``)
        is withheld from the drain so the flush still runs when turns outlast the window.
        Sessions still running at the deadline are skipped (unfinalized keeps them
        recoverable; atexit ``server._shutdown_sessions`` may re-finalize them).
        """
        self._closed.set()
        budget = max(0.0, wait)
        deadline = time.monotonic() + budget - min(_FLUSH_RESERVE_SECS, budget / 2.0)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self._live_turns():
                break
            # Bounded by ``remaining``: a flat sleep would eat the reserve it protects.
            time.sleep(min(0.05, remaining))
        with self._turn_futures_lock:
            live_sids = {sid for f, sid in self._turn_futures.items() if sid and not f.done()}
        self.flush_all_sessions(reason=reason, skip_sids=live_sids)
        self.close()

    def flush_all_sessions(
        self, *, reason: str = "shutdown", skip_sids: Collection[str] | None = None) -> None:
        """Finalize every server session except ``skip_sids`` (turn still live)."""
        try:
            from tui_gateway import server
        except Exception:
            return
        skip = set(skip_sids or ())
        for sid, session in list(server._sessions.items()):
            if sid in skip:
                continue
            with contextlib.suppress(Exception):
                server._finalize_session(session, end_reason=f"compute_host_{reason}")

    def handle_frame(self, frame: dict[str, Any]) -> None:
        kind = str(frame.get("type") or "")
        handler = self._FRAME_HANDLERS.get(kind)
        if handler is None:
            self.emit({
                "type": "error", "request_id": frame.get("request_id"),
                "message": f"unknown frame type: {kind}"})
        else:
            getattr(self, handler)(frame)

    def _handle_shutdown(self, frame: dict[str, Any]) -> None:
        self.emit({"type": "shutdown.ack", "request_id": frame.get("request_id")})
        # Explicit shutdown is a clean close; SIGTERM and orphan paths do the durability flush.
        self.close()

    def _track_turn_future(self, future: concurrent.futures.Future, sid: str) -> None:
        """Track an in-flight turn; the done callback pops it or the map grows forever."""
        with self._turn_futures_lock:
            self._turn_futures[future] = sid
        future.add_done_callback(self._untrack_turn_future)

    def _untrack_turn_future(self, future: concurrent.futures.Future) -> None:
        with self._turn_futures_lock:
            self._turn_futures.pop(future, None)

    def _handle_turn_start(self, frame: dict[str, Any]) -> None:
        future = self._executor.submit(self._run_real_turn, dict(frame))
        self._track_turn_future(future, str(frame.get("sid") or ""))

    def _guarded(
        self, frame: dict[str, Any], error_kind: str, body: Callable, *,
        on_error: Callable[[str], None] | None = None, **error_extra: Any) -> None:
        """Run ``body(server, sid, request_id)``; any exception becomes an ``error_kind`` reply."""
        sid = str(frame.get("sid") or "")
        request_id = frame.get("request_id")
        try:
            from tui_gateway import server
            body(server, sid, request_id)
        except Exception as exc:
            if on_error is not None:
                on_error(sid)
            self._reply(error_kind, sid, request_id, **error_extra, message=str(exc))

    def _handle_interrupt(self, frame: dict[str, Any]) -> None:
        def body(server: Any, sid: str, request_id: Any) -> None:
            session = server._sessions.get(sid)
            if session is None:
                self._reply("interrupt.ack", sid, request_id, applied=False)
                return
            # In the child the shared helper interrupts the local agent and releases this
            # process's pending clarify Event (the parent only has a metadata mirror).
            server._interrupt_session_turn(sid, session)
            self._reply("interrupt.ack", sid, request_id, applied=True, applied_ns=now_ns())
        self._guarded(frame, "interrupt.ack", body, applied=False)

    def _handle_respond(self, frame: dict[str, Any]) -> None:
        """Resolve an interactive request in the host-owned pending registry."""
        def body(server: Any, sid: str, request_id: Any) -> None:
            params = frame.get("params")
            error = ("session not found" if sid not in server._sessions
                     else None if isinstance(params, dict) else "response params must be an object")
            if error:
                self._reply("respond.error", sid, request_id, message=error)
                return
            response = server._methods["clarify.respond"](request_id, params)
            self._reply("respond.ack", sid, request_id, response=response)
        self._guarded(frame, "respond.error", body)

    def _run_real_turn(self, frame: dict[str, Any]) -> None:
        sid = str(frame.get("sid") or "")
        request_id = str(frame.get("request_id") or uuid.uuid4().hex)
        if not sid:
            self._reply("turn.error", sid, request_id, message="sid required")
            return
        try:
            from tui_gateway import server
            session = self._ensure_server_session(server, frame)
            text = frame["text"] if "text" in frame else frame.get("prompt", "")
            inflight = frame["text"] if "text" in frame else frame.get("prompt")
            with session["history_lock"]:
                queued_gen = frame.get("queued_prompt_generation")
                current_gen = int(session.get("_queued_prompt_generation", 0))
                if queued_gen is not None and current_gen != int(queued_gen):
                    self._reply("turn.end", sid, request_id, interrupted=True, ended_ns=now_ns())
                    return
                if session.get("running"):
                    self._reply("turn.error", sid, request_id, message="session busy")
                    return
                session.update(running=True, _turn_cancel_requested=False, last_active=time.time())
                server._start_inflight_turn(session, inflight)
            self._reply("turn.started", sid, request_id, started_ns=now_ns())
            with contextlib.suppress(Exception):
                server._ensure_session_db_row(session)
            with contextlib.suppress(Exception):
                import hermes_undo
                hermes_undo.on_user_message_appended(session["session_key"])
            with contextlib.suppress(Exception):
                server._persist_branch_seed(session)
            server._run_prompt_submit(
                request_id, sid, session, text, display_kind=frame.get("display_kind") or None)
            run_thread = session.get("_run_thread")
            if run_thread is not None and hasattr(run_thread, "join"):
                run_thread.join()
            with session["history_lock"]:
                meta = _history_meta(session)
                interrupted = bool(session.get("_turn_cancel_requested"))
            session_info = server._session_info(session.get("agent"), session)
            with self._progress_lock:
                self._progress_counter += 1
            self._reply(
                "turn.end", sid, request_id, **meta, interrupted=interrupted, ended_ns=now_ns(),
                session_info=session_info, session_info_emitted=True)
        except Exception as exc:
            with contextlib.suppress(Exception):
                from tui_gateway import server
                session = server._sessions.get(sid)
                if session is not None:
                    with session.get("history_lock", threading.Lock()):
                        session["running"] = False
                        server._clear_inflight_turn(session)
            self._reply("turn.error", sid, request_id, reason="exception", message=str(exc))

    def _ensure_server_session(self, server: Any, frame: dict[str, Any]) -> dict:
        sid = str(frame.get("sid") or "")
        session = server._sessions.get(sid)
        if session is not None:
            session["transport"] = self._transport
            if frame.get("cols") is not None:
                session["cols"] = int(frame.get("cols") or 80)
            for key in ("cwd", "profile_home"):
                if frame.get(key):
                    session[key] = str(frame[key])
        else:
            session = self._build_server_session(server, frame, sid)
        if isinstance(frame.get("attached_images"), list):
            session["attached_images"] = list(frame.get("attached_images") or [])
        return session

    def _build_server_session(self, server: Any, frame: dict[str, Any], sid: str) -> dict:
        """Build the agent under the frame's profile scope and register the session."""
        key = str(frame.get("session_key") or sid)
        history = frame.get("history") if isinstance(frame.get("history"), list) else []
        profile_home = str(frame.get("profile_home") or "")
        session_db = home_token = secret_token = None
        owns_db = False
        try:
            if profile_home:
                from hermes_constants import set_hermes_home_override
                from agent.secret_scope import build_profile_secret_scope, set_secret_scope
                home_token = set_hermes_home_override(profile_home)
                secret_token = set_secret_scope(build_profile_secret_scope(Path(profile_home)))
            # DEDICATED handle — ours only until _make_agent succeeds, then the agent owns
            # it. A RAISING _make_agent is the one path where nothing takes it (``owns_db``).
            # BOTH branches flow through this single owns_db/_transfer/finally bracket,
            # differing only in the path arg: a named profile binds <profile_home>/state.db,
            # a DEFAULT session binds the EXPLICIT default-root state.db (never the lazy
            # no-path open that would follow this reused worker's residual ambient home).
            # Lifting the open out of the if-guard keeps the refcount/close accounting single
            # and avoids the leak/double-close a second bracket would cause.
            from hermes_state_registry import acquire
            from hermes_state import default_root_db_path
            db_path = Path(profile_home) / "state.db" if profile_home else default_root_db_path()
            session_db = acquire(db_path)
            owns_db = True
            agent = server._make_agent(
                sid, key, session_id=key, model_override=frame.get("model_override"),
                reasoning_config_override=frame.get("reasoning_config_override"),
                service_tier_override=frame.get("service_tier_override"),
                platform_override=frame.get("source"),
                context_cwd_is_launch_artifact=bool(
                    frame.get("context_cwd_is_launch_artifact", False)),
                session_db=session_db)
            if server._transfer_db_to_agent(agent, session_db):
                owns_db = False
        finally:
            if owns_db and session_db is not None:
                with contextlib.suppress(Exception):
                    from hermes_state_registry import release_or_close
                    release_or_close(session_db)
            if home_token is not None:
                with contextlib.suppress(Exception):
                    from hermes_constants import reset_hermes_home_override
                    from agent.secret_scope import reset_secret_scope
                    reset_hermes_home_override(home_token)
                    reset_secret_scope(secret_token)
        try:
            from tui_gateway.transport import bind_transport, reset_transport
            token = bind_transport(self._transport)
            try:
                server._init_session(
                    sid, key, agent, list(history), cols=int(frame.get("cols") or 80),
                    cwd=str(frame.get("cwd") or "") or None, session_db=session_db,
                    source=frame.get("source"))
            finally:
                reset_transport(token)
        except Exception:
            # _init_session's side machinery (slash worker, approval notify) unavailable: keep a
            # minimal host-owned session rather than failing after the expensive agent build.
            server._sessions[sid] = {
                "agent": agent, "session_key": key, "history": list(history),
                "history_lock": threading.Lock(),
                "history_version": int(frame.get("history_version") or 0), "inflight_turn": None,
                "created_at": time.time(), "last_active": time.time(), "running": False,
                "attached_images": [], "image_counter": 0,
                "cwd": str(frame.get("cwd") or os.getcwd()), "cols": int(frame.get("cols") or 80),
                "slash_worker": None, "show_reasoning": server._load_show_reasoning(),
                "tool_progress_mode": server._load_tool_progress_mode(), "edit_snapshots": {},
                "tool_started_at": {}, "model_override": frame.get("model_override"),
                "source": server._sanitize_client_source(frame.get("source")),
                "transport": self._transport}
        session = server._sessions[sid]
        session["transport"] = self._transport
        session["profile_home"] = profile_home or session.get("profile_home")
        if frame.get("model_override") is not None:
            session["model_override"] = frame.get("model_override")
        return session

    def _handle_reload_mcp(self, frame: dict[str, Any]) -> None:
        def body(server: Any, sid: str, request_id: Any) -> None:
            resp = server.handle_request({
                "id": request_id, "method": "reload.mcp",
                "params": {"session_id": sid, "confirm": True}})
            self._reply("reload_mcp.ack", sid, request_id, response=resp)
        self._guarded(frame, "control.error", body)

    def _handle_control(self, frame: dict[str, Any]) -> None:
        route_name = str(frame.get("route_name") or "")

        def body(server: Any, sid: str, request_id: Any) -> None:
            route = MUTATOR_ROUTE_TABLE.get(route_name)
            session = server._sessions.get(sid)
            error = (f"unclassified route: {route_name}" if route is None
                     else "session not found" if session is None
                     else "session busy" if route == "idle-gated" and session.get("running")
                     else None)
            if error:
                self._reply("control.error", sid, request_id, message=error)
            elif route_name == "reload.mcp":
                self._handle_reload_mcp({**frame, "type": "reload_mcp"})
            else:
                ack = self._control_ack(server, frame, session)
                if "error" in ack:
                    self._reply("control.error", sid, request_id, message=ack["error"])
                else:
                    self._reply("control.ack", sid, request_id, route_name=route_name, **ack)

        def on_error(sid: str) -> None:
            if route_name in {"session.compress", "slash.compress"}:
                # The compress mirror defers the context-engine boundary notification until the
                # host commits; discard it so it can't fire against a rejected boundary later
                # (finalize is exactly-once, so a no-op if the mirror already emitted it).
                with contextlib.suppress(Exception):
                    from tui_gateway import server as _server
                    from agent.conversation_compression import (
                        finalize_context_engine_compression_notification as _finalize)
                    _agent = (_server._sessions.get(sid) or {}).get("agent")
                    if _agent is not None:
                        _finalize(_agent, committed=False)
        self._guarded(frame, "control.error", body, on_error=on_error)

    def _control_ack(self, server: Any, frame: dict[str, Any], session: dict) -> dict:
        """control.ack payload for one classified route, or ``{"error": message}``."""
        sid = str(frame.get("sid") or "")
        route_name = str(frame.get("route_name") or "")
        command = str(frame.get("command") or "")
        if route_name in {"session.save", "session.compress"}:
            params = {"session_id": sid}
            if route_name == "session.compress":
                focus_topic = command.removeprefix("/compress").strip()
                if focus_topic:
                    params["focus_topic"] = focus_topic
            response = server._methods[route_name](frame.get("request_id"), params)
            if "error" in response:
                failure = _CONTROL_FAILURES[route_name]
                return {"error": str(response["error"].get("message") or failure)}
            ack = {"result": response.get("result") or {}}
            if route_name == "session.save":
                return ack
            with session["history_lock"]:
                ack.update(_history_meta(session))
        else:
            output = server._mirror_slash_side_effects(sid, session, command) if command else ""
            with session["history_lock"]:
                messages = server._history_to_messages(list(session.get("history") or []))
                ack = {"output": output, **_history_meta(session), "messages": messages}
        ack["session_info"] = server._session_info(session.get("agent"), session)
        return ack

    def _live_turns(self) -> list[concurrent.futures.Future]:
        with self._turn_futures_lock:
            return [f for f in self._turn_futures if not f.done()]

    def _heartbeat_loop(self) -> None:
        while not self._closed.wait(self._heartbeat_secs):
            active_turns = len(self._live_turns())
            with self._progress_lock:
                counter = self._progress_counter
            self.emit({
                "type": "hb", "active_turns": active_turns, "progress_counter": counter,
                "rss_mb": _rss_mb(os.getpid())})

    def _parent_guard_loop(self) -> None:
        while not self._closed.wait(1.0):
            ppid = os.getppid()
            if ppid in {0, 1} or (self._parent_pid and ppid != self._parent_pid):
                self.emit({"type": "orphan", "old_ppid": self._parent_pid, "ppid": ppid})
                self.shutdown(reason="orphan")
                os._exit(0)


def _history_meta(session: dict) -> dict[str, Any]:
    """Transcript identity for turn.end / control.ack frames; caller holds history_lock."""
    return {
        "session_key": str(session.get("session_key") or ""),
        "history_version": int(session.get("history_version", 0)),
        "message_count": len(session.get("history") or [])}


def _rss_mb(pid: int) -> float:
    try:
        out = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(pid)], text=True, encoding="utf-8", errors="replace",
            stdin=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2).strip()
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
    with contextlib.suppress(Exception):
        signal.signal(signal.SIGTERM, _signal_handler)
        signal.signal(signal.SIGINT, _signal_handler)
    host.emit({
        "type": "hello", "host_pid": os.getpid(), "boot_id": host._boot_id,
        "build_sha": _build_sha(), "cwd": os.getcwd(),
        "hermes_home": os.environ.get("HERMES_HOME", "")})

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
    argparse.ArgumentParser(description="Dashboard compute-host process").parse_args(argv)
    run_host()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from dataclasses import field  # noqa: F401,E402
from dataclasses import dataclass  # noqa: F401,E402
from dataclasses import dataclass  # noqa: F401,E402
from dataclasses import field  # noqa: F401,E402

@dataclass
class SpikeAgent:
    """A deterministic AIAgent-shaped object for pipe/interrupt measurements."""

    session_id: str
    history: list[dict[str, str]] = field(default_factory=list)
    _interrupt: threading.Event = field(default_factory=threading.Event)

    def clear_interrupt(self) -> None:
        self._interrupt.clear()

    def interrupt(self, *, hard_cancel: bool = False) -> None:
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


_PLUGIN_COMPAT_LAZY = {
    'request_hard_interrupt': ('agent.interrupt_compat', 'request_hard_interrupt'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
