import atexit
import concurrent.futures
import contextlib
import contextvars
import copy
import inspect
import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from hermes_constants import (
    get_hermes_home,
    get_hermes_home_override,
    reset_hermes_home_override,
    set_hermes_home_override,
)
from hermes_cli.env_loader import load_hermes_dotenv
from utils import is_truthy_value
from tools.environments.local import hermes_subprocess_env
from agent.replay_cleanup import sanitize_replay_history
from tui_gateway import git_probe
from tui_gateway.transport import (
    StdioTransport,
    Transport,
    bind_transport,
    current_transport,
    reset_transport,
)

logger = logging.getLogger(__name__)

_hermes_home = get_hermes_home()
load_hermes_dotenv(
    hermes_home=_hermes_home, project_env=Path(__file__).parent.parent / ".env"
)


# ── Panic logger ─────────────────────────────────────────────────────
# Gateway crashes in a TUI session leave no forensics: stdout is the
# JSON-RPC pipe (TUI side parses it, doesn't log raw), the root logger
# only catches handled warnings, and the subprocess exits before stderr
# flushes through the stderr->gateway.stderr event pump. This hook
# appends every unhandled exception to ~/.hermes/logs/tui_gateway_crash.log
# AND re-emits a one-line summary to stderr so the TUI can surface it in
# Activity — exactly what was missing when the voice-mode turns started
# exiting the gateway mid-TTS.
_CRASH_LOG = os.path.join(_hermes_home, "logs", "tui_gateway_crash.log")


def _panic_hook(exc_type, exc_value, exc_tb):
    import traceback

    trace = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    try:
        os.makedirs(os.path.dirname(_CRASH_LOG), exist_ok=True)
        with open(_CRASH_LOG, "a", encoding="utf-8") as f:
            f.write(
                f"\n=== unhandled exception · {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
            )
            f.write(trace)
    except Exception:
        pass
    # Stderr goes through to the TUI as a gateway.stderr Activity line —
    # the first line here is what the user will see without opening any
    # log files.  Rest of the stack is still in the log for full context.
    first = (
        str(exc_value).strip().splitlines()[0]
        if str(exc_value).strip()
        else exc_type.__name__
    )
    print(f"[gateway-crash] {exc_type.__name__}: {first}", file=sys.stderr, flush=True)
    # Chain to the default hook so the process still terminates normally.
    sys.__excepthook__(exc_type, exc_value, exc_tb)


sys.excepthook = _panic_hook


def _thread_panic_hook(args):
    # threading.excepthook signature: SimpleNamespace(exc_type, exc_value, exc_traceback, thread)
    import traceback

    trace = "".join(
        traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)
    )
    try:
        os.makedirs(os.path.dirname(_CRASH_LOG), exist_ok=True)
        with open(_CRASH_LOG, "a", encoding="utf-8") as f:
            f.write(
                f"\n=== thread exception · {time.strftime('%Y-%m-%d %H:%M:%S')} "
                f"· thread={args.thread.name} ===\n"
            )
            f.write(trace)
    except Exception:
        pass
    first_line = (
        str(args.exc_value).strip().splitlines()[0]
        if str(args.exc_value).strip()
        else args.exc_type.__name__
    )
    print(
        f"[gateway-crash] thread {args.thread.name} raised {args.exc_type.__name__}: {first_line}",
        file=sys.stderr,
        flush=True,
    )


threading.excepthook = _thread_panic_hook

try:
    from hermes_cli.banner import prefetch_update_check

    prefetch_update_check()
except Exception:
    pass

from tui_gateway.render import make_stream_renderer, render_diff, render_message

_sessions: dict[str, dict] = {}
_methods: dict[str, callable] = {}
_pending: dict[str, tuple[str, threading.Event]] = {}
_pending_prompt_payloads: dict[str, tuple[str, dict]] = {}
_answers: dict[str, str] = {}
_db = None
_db_error: str | None = None
_stdout_lock = threading.Lock()
_cfg_lock = threading.Lock()
_sessions_lock = threading.RLock()  # reentrant: _close_session_by_id may run under callers that already hold it
_prompt_lock = threading.Lock()
_cfg_cache: dict | None = None
_cfg_mtime: float | None = None
_cfg_path = None
_session_resume_lock = threading.Lock()
try:
    _slash_timeout = float(os.environ.get("HERMES_TUI_SLASH_TIMEOUT_S") or "45")
except (ValueError, TypeError):
    _slash_timeout = 45.0
_SLASH_WORKER_TIMEOUT_S = max(5.0, _slash_timeout)

# When a WebSocket client (the dashboard's embedded-chat tab / desktop app)
# disconnects, ``tui_gateway.ws`` detaches the transport but intentionally
# leaves the session parked so a quick reconnect can reattach it (see ws.py).
# That park is unbounded, though: a browser refresh spins up a brand-new
# ``session.create`` (new sid + a fresh _SlashWorker via _deferred_build) and
# never reattaches the OLD sid, so the old session's slash-worker subprocess
# lingers forever — one leaked python process per refresh (#38591 fallout).
# After this grace window, an orphaned (transport-detached, not-running) WS
# session is reaped: its _SlashWorker is closed and the session finalized.
# Set to 0 to disable (park forever, pre-fix behaviour).
try:
    _ws_orphan_reap_grace = float(
        os.environ.get("HERMES_TUI_WS_ORPHAN_REAP_GRACE_S") or "20"
    )
except (ValueError, TypeError):
    _ws_orphan_reap_grace = 20.0
_WS_ORPHAN_REAP_GRACE_S = max(0.0, _ws_orphan_reap_grace)
_DETAIL_SECTION_NAMES = ("thinking", "tools", "subagents", "activity")
_DETAIL_MODES = frozenset({"hidden", "collapsed", "expanded"})

# ── Async RPC dispatch (#12546) ──────────────────────────────────────
# A handful of handlers block the dispatcher loop in entry.py for seconds
# to minutes (slash.exec, cli.exec, shell.exec, session.resume,
# session.branch, session.compress, skills.manage).  While they're running, inbound RPCs —
# notably approval.respond and session.interrupt — sit unread in the
# stdin pipe.  We route only those slow handlers onto a small thread pool;
# everything else stays on the main thread so ordering stays sane for the
# fast path.  write_json is already _stdout_lock-guarded, so concurrent
# response writes are safe.
_LONG_HANDLERS = frozenset(
    {
        "billing.step_up",
        "browser.manage",
        "cli.exec",
        # Completion RPCs run inline on the reader thread by default, but both
        # can block it for seconds: complete.path spawns `git ls-files` and
        # fuzzy-ranks the whole repo (slow on large repos / WSL2 mounts), and
        # complete.slash does first-call prompt_toolkit imports + a skill-dir
        # scan. While either runs inline, prompt.submit / session.interrupt sit
        # unread in the stdin pipe — the TUI appears frozen until the 120s RPC
        # timeout fires (#21123). Routing them to the pool keeps the fast path
        # responsive; completion is read-only and write_json is lock-guarded.
        "complete.path",
        "complete.slash",
        "llm.oneshot",
        # Pet RPCs hit the network (manifest fetch / spritesheet download) or do
        # per-frame PNG decode/encode (pet.cells): inline they serialize on the
        # reader thread, so picker previews trickle in one at a time and the
        # animation poll stutters. On the pool they run concurrently.
        "pet.cells",
        "pet.gallery",
        # Generation is the heaviest pet path by far — multiple image-model
        # round-trips per call — so it must never block the reader thread.
        "pet.generate",
        "pet.hatch",
        "pet.info",
        "pet.select",
        "pet.thumb",
        "learning.frames",
        "plugins.manage",
        "process.list",
        "projects.discover_repos",
        "projects.record_repos",
        "projects.for_cwd",
        "projects.tree",
        "projects.project_sessions",
        # Setup readiness RPCs are polled by the Desktop frontend on connect
        # and periodically (use-status-snapshot → evaluateRuntimeReadiness).
        # setup.runtime_check calls resolve_runtime_provider() which reads
        # config, checks auth state, and may probe the provider endpoint;
        # setup.status calls _has_any_provider_configured() which scans
        # provider config + credential files. Under GIL pressure from
        # concurrent agent turns, either can take seconds inline, blocking
        # the WS read loop and causing false "needs setup" (#50005 family).
        "setup.runtime_check",
        "setup.status",
        "session.branch",
        "session.compress",
        "session.list",
        "session.resume",
        "shell.exec",
        "skills.manage",
        "slash.exec",
    }
)

try:
    _rpc_pool_workers = max(
        2, int(os.environ.get("HERMES_TUI_RPC_POOL_WORKERS") or "8")
    )
except (ValueError, TypeError):
    _rpc_pool_workers = 4
_pool = concurrent.futures.ThreadPoolExecutor(
    max_workers=_rpc_pool_workers,
    thread_name_prefix="tui-rpc",
)
atexit.register(lambda: _pool.shutdown(wait=False, cancel_futures=True))

# Reserve real stdout for JSON-RPC only; redirect Python's stdout to stderr
# so stray print() from libraries/tools becomes harmless gateway.stderr instead
# of corrupting the JSON protocol.
_real_stdout = sys.stdout
sys.stdout = sys.stderr


class _DropTransport:
    """Detached WS sink: keep sessions resumable without writing stale frames."""

    def write(self, obj: dict) -> bool:
        return False

    def close(self) -> None:
        return None


# Module-level stdio transport — fallback sink when no transport is bound via
# contextvar or session. Stream resolved through a lambda so runtime monkey-
# patches of `_real_stdout` (used extensively in tests) still land correctly.
_stdio_transport = StdioTransport(lambda: _real_stdout, _stdout_lock)

# Detached websocket sessions use a drop sink instead of stdio. Desktop embeds
# the gateway in-process and captures stdout into logs, so stale JSON-RPC frames
# must not fall through there while the session waits for resume or reap.
_detached_ws_transport = _DropTransport()


class _SlashWorker:
    """Persistent HermesCLI subprocess for slash commands."""

    def __init__(self, session_key: str, model: str, profile_home: str | None = None):
        self._lock = threading.Lock()
        self._seq = 0
        self.stderr_tail: list[str] = []
        self.stdout_queue: queue.Queue[dict | None] = queue.Queue()

        argv = [
            sys.executable,
            "-m",
            "tui_gateway.slash_worker",
            "--session-key",
            session_key,
        ]
        if model:
            argv += ["--model", model]

        self._closed = False
        from hermes_cli._subprocess_compat import windows_hide_flags

        # slash_worker runs the Hermes agent → needs provider credentials.
        # Tier-1 secrets (gateway/GitHub/infra) are still stripped (#29157).
        env = hermes_subprocess_env(inherit_credentials=True)
        if profile_home:
            # Global-remote / multi-profile sessions: the worker must resolve
            # config/skills/state against the session's profile home, not the
            # gateway's launch HERMES_HOME (#40677).
            env["HERMES_HOME"] = str(profile_home)

        # start_new_session=True detaches the slash worker into its own
        # process group / session. Without this, the worker inherits the
        # gateway's pgid (= TUI parent PID). When mcp_tool's
        # _kill_orphaned_mcp_children races with slash_worker spawn and sweeps
        # the gateway's child set, it captures the worker PID, records the
        # inherited pgid, and killpg() then kills the TUI parent itself.
        # See agent/lsp/client.py for the symmetric LSP server fix and
        # tools/mcp_tool.py _filter_mcp_children for defense-in-depth.
        self.proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            cwd=os.getcwd(),
            env=env,
            creationflags=windows_hide_flags(),
            start_new_session=True,
        )
        threading.Thread(target=self._drain_stdout, daemon=True).start()
        threading.Thread(target=self._drain_stderr, daemon=True).start()

    def _drain_stdout(self):
        for line in self.proc.stdout or []:
            try:
                self.stdout_queue.put(json.loads(line))
            except json.JSONDecodeError:
                continue
        self.stdout_queue.put(None)

    def _drain_stderr(self):
        for line in self.proc.stderr or []:
            if text := line.rstrip("\n"):
                self.stderr_tail = (self.stderr_tail + [text])[-80:]

    def run(self, command: str) -> str:
        if self.proc.poll() is not None:
            raise RuntimeError("slash worker exited")

        with self._lock:
            self._seq += 1
            rid = self._seq
            self.proc.stdin.write(json.dumps({"id": rid, "command": command}) + "\n")
            self.proc.stdin.flush()

            while True:
                try:
                    msg = self.stdout_queue.get(timeout=_SLASH_WORKER_TIMEOUT_S)
                except queue.Empty:
                    raise RuntimeError("slash worker timed out")
                if msg is None:
                    break
                if msg.get("id") != rid:
                    continue
                if not msg.get("ok"):
                    raise RuntimeError(msg.get("error", "slash worker failed"))
                return str(msg.get("output", "")).rstrip()

            raise RuntimeError(
                f"slash worker closed pipe{': ' + chr(10).join(self.stderr_tail[-8:]) if self.stderr_tail else ''}"
            )

    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        proc = self.proc
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=1)
                except Exception:
                    proc.kill()
                    try:
                        proc.wait(timeout=1)  # reap the zombie SIGKILL leaves behind
                    except Exception:
                        pass
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=1)
            except Exception:
                pass
        finally:
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                try:
                    stream.close()
                except Exception:
                    pass


def _load_busy_input_mode() -> str:
    display = _load_cfg().get("display")
    if not isinstance(display, dict):
        display = {}
    raw = str(display.get("busy_input_mode", "") or "").strip().lower()
    return raw if raw in {"queue", "steer", "interrupt"} else "interrupt"


def _notify_session_boundary(
    event_type: str, session_id: str | None, platform: str | None = None
) -> None:
    """Fire session lifecycle hooks with CLI parity."""
    try:
        from hermes_cli.plugins import invoke_hook as _invoke_hook

        _invoke_hook(
            event_type,
            session_id=session_id,
            platform=_resolve_agent_platform(platform),
        )
    except Exception:
        pass


def _claim_active_session_slot(
    session_key: str,
    *,
    live_session_id: str,
    surface: str = "tui",
) -> tuple[Any, str | None]:
    try:
        from hermes_cli.active_sessions import try_acquire_active_session

        return try_acquire_active_session(
            session_id=session_key,
            surface=surface,
            config=_load_cfg(),
            metadata={"live_session_id": live_session_id},
        )
    except Exception as exc:
        logger.warning("Failed to claim active session slot: %s", exc)
        return None, None


def _release_active_session_slot(session: dict | None) -> None:
    if not session:
        return
    lease = session.pop("active_session_lease", None)
    if lease is None:
        return
    try:
        lease.release()
    except Exception:
        logger.debug("Failed to release active session slot", exc_info=True)


def _transfer_active_session_slot(
    sid: str,
    session: dict,
    *,
    new_session_id: str,
) -> bool:
    if not new_session_id:
        return False
    lease = session.get("active_session_lease")
    if lease is None:
        return True
    try:
        from hermes_cli.active_sessions import transfer_active_session

        if transfer_active_session(
            lease,
            session_id=new_session_id,
            metadata={"live_session_id": sid},
        ):
            return True
    except Exception:
        logger.debug("Failed to transfer active session slot", exc_info=True)

    # Fallback: the in-place transfer could not move the lease (entry pruned /
    # pid-check transiently failed). Reserve the new slot BEFORE releasing the
    # old one, so a concurrent gateway at the session cap cannot grab the freed
    # slot in a release-then-reacquire window and leave this session with no
    # lease at all (#49041 review). If the reserve fails, KEEP the old lease.
    new_lease, limit_message = _claim_active_session_slot(
        new_session_id,
        live_session_id=sid,
        surface=_session_source(session),
    )
    if new_lease is not None:
        old_lease = session.pop("active_session_lease", None)
        if old_lease is not None:
            try:
                old_lease.release()
            except Exception:
                logger.debug("Failed to release stale active session slot", exc_info=True)
        session["active_session_lease"] = new_lease
        return True
    # Reserve failed — retain the existing lease rather than dropping it.
    if limit_message:
        logger.warning(
            "Compression session lease re-anchor failed (kept old lease): "
            "sid=%s new_session_id=%s reason=%s",
            sid,
            new_session_id,
            limit_message,
        )
    return False


# Session sources the TUI/desktop backend must never end in state.db: the
# messaging gateway owns those sessions' lifecycle — the TUI is only a viewer
# (a resume of a Telegram/Discord/... session).  Ending one creates the
# #60609 Groundhog Day routing loop (see _finalize_session).  Sources the
# TUI backend itself creates ("tui", plus whatever a client passes as its
# own ``source``) and the CLI's own sessions are NOT gateway-owned.
_NON_GATEWAY_SOURCES = frozenset({
    "", "tui", "cli", "webui", "desktop", "cron", "subagent", "test",
    "local", "acp", "webhook", "api_server", "msgraph_webhook",
})


def _is_gateway_owned_source(source: str) -> bool:
    """True when ``source`` names a messaging-gateway platform whose session
    lifecycle belongs to the gateway, not to this TUI backend.

    Structural rather than a hardcoded platform list: any source that
    resolves to a known gateway ``Platform`` (built-in enum member OR a
    registered platform plugin, via ``Platform._missing_``) counts, so new
    platforms are covered automatically.  Local/self-owned sources are
    excluded explicitly — ``local``/``webhook``/``api_server`` are Platform
    members but their sessions are not owned by a remote chat surface that
    routes by session_key, so reaping them is safe and keeps /resume clean.
    """
    src = (source or "").strip().lower()
    if src in _NON_GATEWAY_SOURCES:
        return False
    try:
        from gateway.config import Platform

        Platform(src)  # raises ValueError for arbitrary non-platform strings
        return True
    except Exception:
        return False


def _finalize_session(session: dict | None, end_reason: str = "tui_close") -> None:
    """Best-effort finalize hook + memory commit for a session.

    Fires ``on_session_end`` plugin hook and attempts to persist any
    unflushed messages before closing the session.  This mirrors the
    CLI's exit-path behaviour and prevents data loss when the TUI is
    force-quit (double Ctrl‑C, terminal‑close, SIGHUP) while the agent
    is mid‑turn.
    """
    if not session or session.get("_finalized"):
        return
    session["_finalized"] = True
    _release_active_session_slot(session)
    stop_event = session.get("_notif_stop")
    if stop_event is not None:
        stop_event.set()

    agent = session.get("agent")
    lock = session.get("history_lock")
    if lock is not None:
        with lock:
            history = list(session.get("history", []))
    else:
        history = list(session.get("history", []))

    # ── Persist unflushed messages to SQLite ──────────────────────────
    # Flush ``agent._session_messages`` via ``_persist_session``'s marker-based
    # dedup (same contract as the gateway-shutdown flush, #13121). Do NOT pass
    # ``conversation_history``: ``session["history"]`` and ``_session_messages``
    # alias the SAME list once a turn completes, so passing it made
    # ``_flush_messages_to_session_db`` treat every message as already-durable
    # and skip it — a data-loss bug when finalize is the sole persist path after
    # a WS disconnect/restart (e.g. the in-turn flush hit a transient SQLite
    # failure). Markers persist the genuinely-unflushed tail without duplicating
    # durable rows (including a resumed-but-not-run session's already-in-DB
    # transcript, which stays in ``session["history"]`` only).
    if agent is not None and hasattr(agent, "_persist_session"):
        snapshot = getattr(agent, "_session_messages", None)
        if snapshot:
            try:
                agent._persist_session(snapshot)
            except Exception:
                pass

    # ── Plugin hook: on_session_end ────────────────────────────────────
    # Signals every plugin that the session is closing, with
    # interrupted=True so crash‑recovery plugins can flush buffers,
    # persist state, or close connections before the gateway exits.
    # Mirrors cli.py's atexit handler that fires the same hook when
    # the user Ctrl‑C's mid‑turn.
    if agent is not None:
        try:
            from hermes_cli.plugins import invoke_hook

            invoke_hook(
                "on_session_end",
                session_id=getattr(agent, "session_id", None)
                or session.get("session_key", ""),
                completed=False,
                interrupted=True,
                model=getattr(agent, "model", "unknown"),
                platform=getattr(agent, "platform", None) or "tui",
            )
        except Exception:
            pass

    if agent is not None and history and hasattr(agent, "commit_memory_session"):
        try:
            agent.commit_memory_session(history)
        except Exception:
            pass

    session_key = session.get("session_key")
    session_id = getattr(agent, "session_id", None) or session_key
    _notify_session_boundary("on_session_finalize", session_id, _session_source(session))

    # Mark session ended in DB so it doesn't linger as a ghost row in /resume.
    # Use session_id (from agent.session_id) not session_key — after compression,
    # session_key may be stale (the ended parent) while session_id is the live
    # continuation. Fix for #20001.
    _tui_owns_lifecycle = True
    if session_id:
        try:
            db = _get_db()
            if db is not None:
                # Don't end gateway-originated sessions — the gateway owns
                # their lifecycle.  The TUI is a viewer, not the owner.
                # Ending a gateway session in state.db triggers a Groundhog
                # Day routing loop: the gateway's #54878 self-heal detects
                # the stale entry, recovers to the parent session, context
                # compression splits back to the reaped child, and the cycle
                # repeats on every inbound message.  (#60609)
                row = db.get_session(session_id)
                source = (row or {}).get("source", "")
                _tui_owns_lifecycle = not _is_gateway_owned_source(source)
                if _tui_owns_lifecycle:
                    db.end_session(session_id, end_reason)
        except Exception:
            pass

    # A session's in-flight async delegations end WITH the session (#55578):
    # once nobody owns the return address, a still-running background subagent
    # can only burn tokens and park an orphaned completion on the shared
    # queue. Always interrupt delegations commissioned by THIS live UI session
    # (its sid); additionally interrupt by durable session_key, but only when
    # the TUI owns the lifecycle — closing a viewer tab on a live gateway
    # session must not kill the gateway's own background work.
    try:
        from tools.async_delegation import interrupt_for_session

        _own_sid = str(session.get("_sid") or "")
        if not _own_sid:
            try:
                with _sessions_lock:
                    for _cand_sid, _cand in _sessions.items():
                        if _cand is session:
                            _own_sid = _cand_sid
                            break
            except Exception:
                _own_sid = ""
        interrupt_for_session(
            session_key=str(session_key or "") if _tui_owns_lifecycle else "",
            origin_ui_session_id=_own_sid,
            reason=end_reason,
        )
    except Exception:
        pass

    # Close the slash-worker subprocess as part of finalize itself, not just
    # in the callers. Defense-in-depth: every session-end path goes through
    # _finalize_session (it's the single ``_finalized``-guarded chokepoint), so
    # folding worker cleanup in here means a future code path that calls
    # _finalize_session directly — without the surrounding _teardown_session /
    # _shutdown_sessions worker.close() — can't reintroduce the #38095 leak.
    # Idempotent: _SlashWorker.close() is poll()-guarded, so the explicit
    # close() still in those callers is harmless.
    try:
        worker = session.get("slash_worker")
        if worker:
            worker.close()
    except Exception:
        pass


def _teardown_session(session: dict | None, *, end_reason: str = "tui_close") -> None:
    """Fully tear down a session: finalize, unregister, close agent + worker.

    Shared by ``session.close`` and the orphaned-WS-session reaper. The
    slash-worker subprocess is closed inside ``_finalize_session`` (the single
    finalize chokepoint); this still unregisters the approval notifier and
    closes the in-process agent. Idempotent: the ``_finalized`` guard in
    ``_finalize_session`` and the ``poll()`` guard in ``_SlashWorker.close``
    make repeat calls harmless.
    """
    if not session:
        return
    _finalize_session(session, end_reason=end_reason)
    try:
        from tools.approval import unregister_gateway_notify

        if key := session.get("session_key"):
            unregister_gateway_notify(key)
    except Exception:
        pass
    try:
        agent = session.get("agent")
        if agent is not None and hasattr(agent, "close"):
            agent.close()
    except Exception:
        pass
    # NOTE: the slash-worker is closed inside _finalize_session (the single
    # _finalized-guarded chokepoint that main folded it into), exactly once.
    # We deliberately do NOT re-close it here — _teardown_session's job beyond
    # finalize is unregistering the notifier and closing the in-process agent.


def _attach_worker(sid: str, session: dict, worker) -> None:
    """Store worker on session iff sid still maps to it, else close it — a
    concurrent teardown already popped the session and would orphan the
    worker. Closes the create/close race at every slash-worker spawn site."""
    with _sessions_lock:
        if _sessions.get(sid) is session:
            session["slash_worker"] = worker
            return
    worker.close()


def _close_session_by_id(sid: str, *, end_reason: str = "tui_close") -> bool:
    """Single idempotent teardown for one session: pop it under the sessions
    lock, then finalize, unregister notify, close agent + slash worker via the
    shared ``_teardown_session`` path. Returns True iff it closed a live
    session. The ``_finalized`` / worker ``_closed`` guards make concurrent or
    repeat calls (e.g. session.close racing the WS-orphan reaper) harmless."""
    with _sessions_lock:
        session = _sessions.pop(sid, None)
    if session is None:
        return False
    # The session is already out of _sessions here, so downstream teardown
    # (e.g. _finalize_session's per-session async-delegation interrupt) can't
    # recover its live id by scanning the dict — stamp it on the record.
    session["_sid"] = sid
    _teardown_session(session, end_reason=end_reason)
    return True



def _ws_session_is_orphaned(session: dict | None) -> bool:
    """True if a WS session has no live transport and no in-flight turn.

    After ``handle_ws`` detaches a disconnected client it points the session at
    ``_detached_ws_transport``. A session left on that transport (and not
    mid-turn) is genuinely orphaned and safe to reap.
    """
    if not session or session.get("_finalized"):
        return False
    if session.get("running"):
        return False
    return session.get("transport") is _detached_ws_transport


def _schedule_ws_orphan_reap(sid: str) -> None:
    """After a grace window, reap session ``sid`` iff it's still orphaned.

    Called from the WS-disconnect path. The grace window lets a transient
    reconnect (or a ``session.resume`` that reattaches the transport) cancel
    the reap by re-binding a live transport. Disabled when the grace is 0.
    """
    if _WS_ORPHAN_REAP_GRACE_S <= 0:
        return

    def _reap() -> None:
        # Serialize the orphan re-check against session.resume (which re-binds a
        # live transport under _session_resume_lock and would make this session
        # non-orphaned). The actual pop + teardown then goes through the shared
        # _close_session_by_id funnel so the dict mutation happens under
        # _sessions_lock — consistent with every other _sessions mutator
        # (#39591: _reap previously popped under _session_resume_lock, giving no
        # mutual exclusion against _init_session / _close_session_by_id, which
        # guard with _sessions_lock). _sessions_lock is an RLock and the global
        # ordering is always resume_lock -> sessions_lock, so nesting is safe.
        with _session_resume_lock:
            if not _ws_session_is_orphaned(_sessions.get(sid)):
                return
            _close_session_by_id(sid, end_reason="ws_orphan_reap")

    timer = threading.Timer(_WS_ORPHAN_REAP_GRACE_S, _reap)
    timer.daemon = True
    timer.start()


def _close_sessions_for_transport(
    transport, *, end_reason: str = "ws_disconnect"
) -> tuple[int, int]:
    """On transport disconnect, reap the sessions that opted into
    close_on_disconnect (sidecar/dashboard) immediately via the unified
    ``_close_session_by_id`` path, and re-point the rest back to stdio so later
    emits don't hit a dead socket.

    Non-flagged detached sessions are handed to the grace-windowed WS-orphan
    reaper (``_schedule_ws_orphan_reap``): a quick reconnect / session.resume
    that re-binds a live transport cancels the reap, otherwise the orphan is
    torn down through the same idempotent ``_teardown_session`` path. This is
    the single WS-disconnect teardown entry point — there is no second
    independent reap loop in ``handle_ws``.

    Returns ``(reaped, detached)`` counts for disconnect-path observability."""
    with _sessions_lock:
        owned = [(sid, s) for sid, s in _sessions.items() if s.get("transport") is transport]
    reaped = 0
    detached = 0
    for sid, session in owned:
        if session.get("close_on_disconnect"):
            _close_session_by_id(sid, end_reason=end_reason)
            reaped += 1
        else:
            # Point detached sessions at the drop sentinel (NOT real stdio) so
            # _ws_session_is_orphaned recognizes them and the grace-reap can
            # actually fire; a standalone `hermes --tui` keeps real _stdio.
            session["transport"] = _detached_ws_transport
            detached += 1
            try:
                _schedule_ws_orphan_reap(sid)
            except Exception:
                pass
    return reaped, detached


def _shutdown_sessions() -> None:
    with _sessions_lock:
        sids = list(_sessions)
    for sid in sids:
        _close_session_by_id(sid, end_reason="tui_shutdown")


# Last-resort net for any disconnect path that slips past the WS finally. TTL is
# hours-scale because last_active freezes during a long turn and on passive
# viewing — running/pending/starting/live-transport are hard exemptions instead.
try:
    _SESSION_TTL_S = float(os.environ.get("HERMES_TUI_SESSION_TTL_S") or 6 * 3600)
except (TypeError, ValueError):
    _SESSION_TTL_S = float(6 * 3600)
_SESSION_TTL_S = max(0.0, _SESSION_TTL_S)
_REAPER_SCAN_S = 300.0


def _transport_is_dead(transport) -> bool:
    # _detached_ws_transport is the post-WS-disconnect drop sentinel; a session
    # parked on it has no live client. _stdio_transport is the REAL transport
    # for a standalone `hermes --tui`, so it must NOT count as dead here (doing
    # so let the idle reaper evict healthy standalone TUI sessions).
    if transport is _detached_ws_transport:
        return True
    return getattr(transport, "_closed", None) is True


def _session_is_evictable(sid: str, session: dict, now: float) -> bool:
    if session.get("running") or _session_pending_kind(sid):
        return False
    ready = session.get("agent_ready")
    # Lazy watch sessions (subagent spectator windows) never start a build,
    # so their forever-unset agent_ready must not make them immortal.
    if ready is not None and not ready.is_set() and not session.get("lazy"):
        return False
    if not _transport_is_dead(session.get("transport")):
        return False
    last_active = float(session.get("last_active") or 0.0)
    created_at = float(session.get("created_at") or 0.0)
    return (now - last_active) > _SESSION_TTL_S and (now - created_at) > _SESSION_TTL_S


def _reap_idle_sessions() -> None:
    now = time.time()
    with _sessions_lock:
        victims = [sid for sid, s in _sessions.items() if _session_is_evictable(sid, s, now)]
    for sid in victims:
        _close_session_by_id(sid, end_reason="idle_timeout")
    _enforce_session_cap()


# Soft LRU cap on in-memory sessions. The 6h TTL reaper above only frees
# sessions that have been idle for hours; a heavy user who reconnects often
# accumulates detached sessions (the report's ``detached_sessions=5``) whose
# agents sit resident for the full TTL. The cap evicts the least-recently-active
# DETACHED sessions sooner so live agents don't pile up under memory pressure.
# Default-on but provably safe: it only touches sessions with no live client
# (reopening re-resumes them from the DB) and never a running / pending /
# mid-build / live-transport one. 0/null disables.
def _max_live_sessions() -> int:
    try:
        from hermes_cli.active_sessions import coerce_max_concurrent_sessions

        cfg = _load_cfg() or {}
        raw = cfg.get("max_live_sessions")
        if raw is None:
            gateway_cfg = cfg.get("gateway")
            if isinstance(gateway_cfg, dict):
                raw = gateway_cfg.get("max_live_sessions")
        coerced = coerce_max_concurrent_sessions(raw, key="max_live_sessions")
        return int(coerced) if coerced else 0
    except Exception:
        return 0


def _session_is_lru_evictable(sid: str, session: dict) -> bool:
    # Same hard exemptions as the TTL reaper (never evict a session mid-turn,
    # awaiting input, or still building), but WITHOUT the hours-scale age gate:
    # a detached session is eligible the moment it loses its client.
    if session.get("running") or _session_pending_kind(sid):
        return False
    ready = session.get("agent_ready")
    if ready is not None and not ready.is_set() and not session.get("lazy"):
        return False
    return _transport_is_dead(session.get("transport"))


def _enforce_session_cap() -> None:
    cap = _max_live_sessions()
    if cap <= 0:
        return
    with _sessions_lock:
        total = len(_sessions)
        if total <= cap:
            return
        evictable = [
            (sid, s) for sid, s in _sessions.items() if _session_is_lru_evictable(sid, s)
        ]
    # Oldest-touched first; only evict down to the cap (live/focused sessions on
    # a live transport are never eligible, so we may stop short of the cap).
    evictable.sort(key=lambda kv: float(kv[1].get("last_active") or 0.0))
    overflow = total - cap
    for sid, _s in evictable[:overflow]:
        _close_session_by_id(sid, end_reason="lru_evict")


def _schedule_session_cap_enforcement() -> None:
    """Run the LRU sweep off the response path (eviction can call agent.close)."""

    def _run():
        try:
            _enforce_session_cap()
        except Exception:
            logger.debug("session cap enforcement failed", exc_info=True)

    timer = threading.Timer(0.1, _run)
    timer.daemon = True
    timer.start()


def _start_idle_reaper() -> None:
    def _loop():
        while True:
            time.sleep(_REAPER_SCAN_S)
            try:
                _reap_idle_sessions()
            except Exception:
                pass

    threading.Thread(target=_loop, daemon=True).start()


atexit.register(_shutdown_sessions)
_start_idle_reaper()


# ── Plumbing ──────────────────────────────────────────────────────────


def _get_db():
    global _db, _db_error
    if _db is None:
        from hermes_state import SessionDB

        try:
            _db = SessionDB()
            _db_error = None
        except Exception as exc:
            _db_error = str(exc)
            logger.warning(
                "TUI session store unavailable — continuing without state.db features: %s",
                exc,
            )
            return None
    return _db


def _db_unavailable_error(rid, *, code: int):
    detail = _db_error or "state.db unavailable"
    return _err(rid, code, f"state.db unavailable: {detail}")


# ── per-session profile scoping (global remote mode) ───────────────────────────
# One dashboard normally serves its launch profile. But the desktop's app-global
# remote mode points every profile at this single backend, so resume/prompt must
# be able to act on ANOTHER local profile's state.db + home. The desktop passes
# ``profile`` on those calls; we open that profile's db and bind its HERMES_HOME
# (a ContextVar override) for the duration of the call so config/skills/model and
# message persistence all resolve to the right profile. Omitted/own profile → the
# launch profile (unchanged for single-profile and per-profile-remote setups).
def _profile_home(profile: str | None) -> Path | None:
    """Resolve a named profile's home on THIS host, or None for the launch profile."""
    name = (profile or "").strip()
    if not name:
        return None
    try:
        from hermes_cli import profiles as profiles_mod

        home = Path(profiles_mod.get_profile_dir(name))
    except Exception:
        return None
    # Already the launch profile? No override needed.
    if home.resolve() == Path(_hermes_home).resolve():
        return None
    return home if (home / "state.db").exists() or home.exists() else None


def _profile_scoped(handler):
    """Bind ``params['profile']``'s HERMES_HOME around a pet RPC handler.

    Pets are per-profile: ``display.pet.*`` lives in the profile's config.yaml and
    sprites install under its ``pets/`` dir (both resolve via ``get_hermes_home``).
    The desktop sends ``profile`` on pet calls so config + pets dir resolve to the
    focused profile even in app-global remote mode, where one backend serves every
    profile. No-op for the launch profile (own-profile backends already resolve it).
    """

    def wrapper(rid, params):
        home = _profile_home(params.get("profile") if isinstance(params, dict) else None)
        if home is None:
            return handler(rid, params)
        token = set_hermes_home_override(home)
        try:
            return handler(rid, params)
        finally:
            reset_hermes_home_override(token)

    return wrapper


# Placeholder ``terminal.cwd`` values that don't name a real directory — the
# gateway resolves these to the home dir at runtime, so they must NOT be treated
# as an explicit workspace (mirrors gateway/run.py's config bridge).
_CWD_PLACEHOLDERS = {".", "auto", "cwd"}


def _configured_cwd_from_cfg(cfg: dict | None) -> str | None:
    """Return an absolute, existing ``terminal.cwd`` from a config mapping.

    Returns None for placeholders (``.``/``auto``/``cwd``), missing values, or
    paths that don't resolve to a real directory.
    """
    if not isinstance(cfg, dict):
        return None
    terminal_cfg = cfg.get("terminal")
    if not isinstance(terminal_cfg, dict):
        return None
    raw = str(terminal_cfg.get("cwd") or "").strip()
    if not raw or raw in _CWD_PLACEHOLDERS:
        return None
    resolved = os.path.abspath(os.path.expanduser(raw))
    return resolved if os.path.isdir(resolved) else None


def _profile_configured_cwd(profile_home: Path | None) -> str | None:
    """Resolve a non-launch profile's ``terminal.cwd`` from its own config.yaml.

    The desktop's app-global remote mode serves every profile from one backend,
    so the process-global ``TERMINAL_CWD`` belongs to the *launch* profile. A new
    session bound to another profile must take its workspace from THAT profile's
    config, not the stale env var (issue #40334). Returns an absolute, existing
    directory, or None for placeholders / missing / invalid paths.
    """
    if profile_home is None:
        return None
    try:
        import yaml

        p = Path(profile_home) / "config.yaml"
        if not p.exists():
            return None
        with open(p, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return _configured_cwd_from_cfg(data)
    except Exception:
        return None


def _launch_configured_cwd() -> str | None:
    """Resolve the launch profile's ``terminal.cwd`` from config.yaml.

    Dashboard ``/chat`` for the launch profile attaches to the dashboard
    process's in-memory TUI gateway. The Node PTY child receives a bridged
    ``TERMINAL_CWD`` env var, but this in-memory process does not — so reading
    the process env alone leaves a fresh chat starting in ``os.getcwd()``
    (wherever ``hermes dashboard`` was launched) instead of the configured
    ``terminal.cwd``. Read config directly so changing ``terminal.cwd`` affects
    new in-memory TUI sessions too.
    """
    try:
        return _configured_cwd_from_cfg(_load_cfg())
    except Exception:
        return None


def _default_session_cwd() -> str:
    """Fallback cwd for a session with no explicit / stored / profile cwd.

    Mirrors the launch-config-aware tail of :func:`_completion_cwd` so freshly
    created AND resumed sessions land in the configured ``terminal.cwd`` rather
    than ``os.getcwd()`` when the in-memory gateway's process env has no bridged
    ``TERMINAL_CWD``.
    """
    return _launch_configured_cwd() or os.getenv("TERMINAL_CWD") or os.getcwd()


def write_json(obj: dict) -> bool:
    """Emit one JSON frame. Routes via the most-specific transport available.

    Precedence:

    1. Event frames with a session id → the transport stored on that session,
       so async events land with the client that owns the session even if
       the emitting thread has no contextvar binding.
    2. Otherwise the transport bound on the current context (set by
       :func:`dispatch` for the lifetime of a request).
    3. Otherwise the module-level stdio transport, matching the historical
       behaviour and keeping tests that monkey-patch ``_real_stdout`` green.
    """
    if obj.get("method") == "event":
        sid = ((obj.get("params") or {}).get("session_id")) or ""
        if sid and (t := (_sessions.get(sid) or {}).get("transport")) is not None:
            return t.write(obj)

    return (current_transport() or _stdio_transport).write(obj)


def _emit(event: str, sid: str, payload: dict | None = None):
    params = {"type": event, "session_id": sid}
    if payload is not None:
        params["payload"] = payload
    write_json({"jsonrpc": "2.0", "method": "event", "params": params})


def _emit_approval_request(sid: str, data: dict | None) -> None:
    """Emit an ``approval.request`` event to the TUI client with the command
    redacted. The approval payload is built from the RAW command string, so a
    credential-shaped value Tirith flagged would otherwise be echoed verbatim
    to the TUI client (#48456 — third egress transport alongside the chat
    platforms and the SSE/API stream fixed in #50767). Reuse the shared gateway
    seam so all approval transports redact consistently."""
    payload = dict(data or {})
    if "command" in payload:
        from gateway.run import _redact_approval_command

        payload["command"] = _redact_approval_command(payload.get("command"))
    _emit("approval.request", sid, payload)


def _status_update(sid: str, kind: str, text: str | None = None):
    body = (text if text is not None else kind).strip()
    if not body:
        return
    out_kind = kind if text is not None else "status"
    # Auto-compaction reaches us as a generic "lifecycle" status. Re-tag it so
    # drivers (desktop app) can show an explicit "Summarizing…" indicator —
    # otherwise a mid-turn compaction looks like the transcript reset itself.
    if out_kind == "lifecycle":
        from agent.conversation_compression import COMPACTION_STATUS_MARKER

        if COMPACTION_STATUS_MARKER in body:
            out_kind = "compacting"
    _emit("status.update", sid, {"kind": out_kind, "text": body})


def _estimate_image_tokens(width: int, height: int) -> int:
    """Very rough UI estimate for image prompt cost.

    Uses 512px tiles at ~85 tokens/tile as a lightweight cross-provider hint.
    This is intentionally approximate and only used for attachment display.
    """
    if width <= 0 or height <= 0:
        return 0
    return max(1, (width + 511) // 512) * max(1, (height + 511) // 512) * 85


def _image_meta(path: Path) -> dict:
    meta = {"name": path.name}
    try:
        from PIL import Image

        with Image.open(path) as img:
            width, height = img.size
        meta["width"] = int(width)
        meta["height"] = int(height)
        meta["token_estimate"] = _estimate_image_tokens(int(width), int(height))
    except Exception:
        pass
    return meta


def _ok(rid, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": rid, "result": result}


def _err(rid, code: int, msg: str) -> dict:
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": msg}}


def method(name: str):
    def dec(fn):
        _methods[name] = fn
        return fn

    return dec


def _normalize_request(req: Any) -> tuple[Any, str, dict] | dict:
    """Validate a JSON-RPC request enough for safe local dispatch."""
    if not isinstance(req, dict):
        return _err(None, -32600, "invalid request: expected an object")

    rid = req.get("id")
    method = req.get("method")
    if not isinstance(method, str) or not method:
        return _err(rid, -32600, "invalid request: method must be a non-empty string")

    params = req.get("params", {})
    if params is None:
        params = {}
    elif not isinstance(params, dict):
        return _err(rid, -32602, "invalid params: expected an object")

    return rid, method, params


def handle_request(req: dict) -> dict | None:
    normalized = _normalize_request(req)
    if isinstance(normalized, dict):
        return normalized

    rid, method, params = normalized
    fn = _methods.get(method)
    if not fn:
        return _err(rid, -32601, f"unknown method: {method}")
    return fn(rid, params)


def dispatch(req: dict, transport: Optional[Transport] = None) -> dict | None:
    """Route inbound RPCs — long handlers to the pool, everything else inline.

    Returns a response dict when handled inline. Returns None when the
    handler was scheduled on the pool; the worker writes its own response
    via the bound transport when done.

    *transport* (optional): pins every write produced by this request —
    including any events emitted by the handler — to the given transport.
    Omitting it falls back to the module-level stdio transport, preserving
    the original behaviour for ``tui_gateway.entry``.
    """
    t = transport or _stdio_transport
    token = bind_transport(t)
    try:
        normalized = _normalize_request(req)
        if isinstance(normalized, dict):
            return normalized

        _rid, method, _params = normalized
        if method not in _LONG_HANDLERS:
            return handle_request(req)

        # Snapshot the context so the pool worker sees the bound transport.
        ctx = contextvars.copy_context()

        def run():
            try:
                resp = handle_request(req)
            except Exception as exc:
                resp = _err(req.get("id"), -32000, f"handler error: {exc}")
            if resp is not None:
                t.write(resp)

        _pool.submit(lambda: ctx.run(run))

        return None
    finally:
        reset_transport(token)


def _wait_agent(session: dict, rid: str, timeout: float = 30.0) -> dict | None:
    ready = session.get("agent_ready")
    if ready is not None and not ready.wait(timeout=timeout):
        return _err(rid, 5032, "agent initialization timed out")
    err = session.get("agent_error")
    return _err(rid, 5032, err) if err else None


def _start_agent_build(sid: str, session: dict) -> None:
    """Start building the real AIAgent for a TUI session, once.

    Classic `hermes` shows the prompt before constructing AIAgent; the TUI used
    to eagerly build it during session.create, making startup feel blocked on
    tool discovery/model metadata even though the composer was visible.  Keep
    the shell responsive by deferring this work until the first prompt (or any
    command that actually needs the agent), while retaining the same ready/error
    event contract for the frontend.
    """
    ready = session.get("agent_ready")
    if ready is None:
        return
    # A lazy watch session spectating an in-flight child must stay lazy so the
    # subagent live-mirror keeps flowing. Incidental RPCs (session.info, model
    # metadata, etc.) resolve through _sess(), which would otherwise upgrade it
    # to a full agent mid-stream and silently kill the mirror (the mirror bails
    # once agent is set). Once the child completes, the guard lifts and the next
    # prompt/RPC builds the agent normally so the user can talk to the session.
    if session.get("lazy") and _child_run_active(str(session.get("session_key") or "")):
        return
    lock = session.setdefault("agent_build_lock", threading.Lock())
    with lock:
        if ready.is_set() or session.get("agent_build_started"):
            return
        session["agent_build_started"] = True
        # An upgrading lazy session is now genuinely mid-construction — restore
        # its "still starting" eviction exemption.
        session.pop("lazy", None)
    key = session["session_key"]

    def _build() -> None:
        with _sessions_lock:
            current = _sessions.get(sid)
        if current is None:
            ready.set()
            return

        worker = None
        notify_registered = False
        home_token = None
        profile_home = current.get("profile_home")
        try:
            tokens = _set_session_context(key)
            # Build against the session's profile (global-remote): bind its
            # HERMES_HOME so config/skills/model resolve to it, and hand the
            # agent that profile's db so turns persist to the right state.db.
            session_db = None
            if profile_home:
                home_token = set_hermes_home_override(profile_home)
                try:
                    from hermes_state import SessionDB

                    session_db = SessionDB(db_path=Path(profile_home) / "state.db")
                except Exception:
                    session_db = None
            try:
                # Lazy-resumed (watch) sessions carry the stored conversation
                # id — pass it through so the upgrade continues that session
                # instead of starting a fresh one under the same key.
                kw = {"session_db": session_db}
                if resume_sid := current.get("resume_session_id"):
                    kw["session_id"] = resume_sid
                kw["platform_override"] = _session_source(current)
                resume_overrides = current.get("resume_runtime_overrides")
                if isinstance(resume_overrides, dict) and resume_overrides:
                    # Cold deferred resume: restore the full persisted runtime
                    # identity (model/provider/base_url/api_mode/reasoning/tier)
                    # exactly as the eager resume path's _stored_session_runtime_
                    # overrides splat did, so a deferred build can't drop the
                    # provider and fail with "No LLM provider configured".
                    kw.update(resume_overrides)
                else:
                    # Model/effort/fast the desktop picked for a brand-new chat
                    # ride in as per-session overrides so the first build uses
                    # them directly (no global config, no build-then-switch).
                    if override := current.get("model_override"):
                        kw["model_override"] = override
                    if (reasoning := current.get("create_reasoning_override")) is not None:
                        kw["reasoning_config_override"] = reasoning
                    if (tier := current.get("create_service_tier_override")) is not None:
                        kw["service_tier_override"] = tier
                agent = _make_agent(sid, key, **kw)
            finally:
                _clear_session_context(tokens)

            # Session DB row deferred to first run_conversation() call.
            # pending_title applied post-first-message (see cli.exec handler).
            current["agent"] = agent
            # Baseline for the per-turn config sync; the profile home
            # override is still active here.
            current["config_model_seen"] = _config_model_target()

            try:
                worker = _SlashWorker(
                    key,
                    getattr(agent, "model", _resolve_model()),
                    profile_home=current.get("profile_home"),
                )
                _attach_worker(sid, current, worker)
            except Exception:
                pass

            try:
                from tools.approval import (
                    register_gateway_notify,
                    load_permanent_allowlist,
                )

                register_gateway_notify(
                    key, lambda data: _emit_approval_request(sid, data)
                )
                notify_registered = True
                load_permanent_allowlist()
            except Exception:
                pass

            _wire_callbacks(sid)
            # Surface the self-improvement review's "💾 …" summary as an event
            # the TUI/desktop render in-transcript, honoring
            # display.memory_notifications. _init_session wires this for the
            # eager/branch paths; deferred-built sessions (session.create and the
            # default cold resume) build through here, so without this their
            # review summaries would leak to stdout instead of the chat.
            try:
                agent.background_review_callback = lambda message, _sid=sid: _emit(
                    "review.summary", _sid, {"text": str(message)}
                )
                agent.memory_notifications = _load_memory_notifications()
            except Exception:
                pass
            # Hydrate credits notices at session OPEN (not just on the first
            # message), so depletion / usage-band warnings show at "ready". Runs
            # off the build thread, after the notice_callback is wired. Fail-open.
            try:
                from agent.credits_tracker import seed_credits_at_session_start

                seed_credits_at_session_start(agent)
            except Exception:
                pass
            with _sessions_lock:
                if sid in _sessions:
                    _sessions[sid]["_notif_stop"] = _start_notification_poller(sid, _sessions[sid])
            _notify_session_boundary("on_session_reset", key, _session_source(current))

            info = _session_info(agent, current)
            cfg_warn = _probe_config_health(_load_cfg())
            if cfg_warn:
                info["config_warning"] = cfg_warn
                logger.warning(cfg_warn)
            _emit("session.info", sid, info)
            # If MCP discovery is still in flight (a server slower than the
            # bounded wait_for_mcp_discovery join in _make_agent), the agent
            # was built without those tools. Catch up once they land — see
            # _schedule_mcp_late_refresh. Cache-safe (pre-first-turn only).
            _schedule_mcp_late_refresh(sid, agent)
        except Exception as e:
            current["agent_error"] = str(e)
            _emit("error", sid, {"message": f"agent init failed: {e}"})
        finally:
            if home_token is not None:
                reset_hermes_home_override(home_token)
            # _attach_worker already closed the worker if this session was
            # reaped mid-build; only the late notify registration can still
            # leak (session.close unregistered before _build registered it).
            with _sessions_lock:
                replaced = _sessions.get(sid) is not current
            if replaced and notify_registered:
                try:
                    from tools.approval import unregister_gateway_notify

                    unregister_gateway_notify(key)
                except Exception:
                    pass
            ready.set()

    threading.Thread(target=_build, daemon=True).start()


def _sess_nowait(params, rid):
    s = _sessions.get(params.get("session_id") or "")
    return (s, None) if s else (None, _err(rid, 4001, "session not found"))


def _sess(params, rid):
    s, err = _sess_nowait(params, rid)
    if err:
        return (None, err)
    _start_agent_build(params.get("session_id") or "", s)
    return (s, _wait_agent(s, rid))


def _normalize_completion_path(path_part: str) -> str:
    expanded = os.path.expanduser(path_part)
    if os.name != "nt":
        normalized = expanded.replace("\\", "/")
        if (
            len(normalized) >= 3
            and normalized[1] == ":"
            and normalized[2] == "/"
            and normalized[0].isalpha()
        ):
            return f"/mnt/{normalized[0].lower()}/{normalized[3:]}"
    return expanded


def _completion_cwd(params: dict | None = None) -> str:
    params = params or {}
    raw = (
        params.get("cwd")
        or _sessions.get(params.get("session_id") or "", {}).get("cwd")
        # A session bound to another profile resolves its workspace from THAT
        # profile's config before falling back to the launch profile's env var.
        or _profile_configured_cwd(_profile_home(params.get("profile")))
        # The launch profile's dashboard /chat attaches to the dashboard's
        # in-memory gateway, which does NOT inherit the PTY child's bridged
        # TERMINAL_CWD. Read the launch profile's config.yaml directly so a
        # configured terminal.cwd wins over a stale process env / launch dir.
        or _launch_configured_cwd()
        or os.environ.get("TERMINAL_CWD")
        or os.getcwd()
    )
    try:
        resolved = os.path.abspath(os.path.expanduser(str(raw)))
        if os.path.isdir(resolved):
            return resolved
    except Exception:
        pass
    return os.getcwd()


def _terminal_task_cwd(session: dict | None) -> str:
    """Return the cwd that terminal_tool should use for this TUI session.

    ``_completion_cwd`` validates paths on the host so file completion does not
    point at nonsense.  Non-local terminal backends are different: their cwd is
    inside the target environment, so an SSH path like /home/user/workspace may
    not exist on the local macOS host but is still the correct execution cwd.
    """
    backend = (os.environ.get("TERMINAL_ENV") or "").strip().lower()
    if backend and backend != "local":
        raw = os.environ.get("TERMINAL_CWD", "").strip()
        if not raw:
            try:
                terminal_cfg = _load_cfg().get("terminal", {})
                if isinstance(terminal_cfg, dict):
                    raw = str(terminal_cfg.get("cwd") or "").strip()
            except Exception:
                raw = ""
        if raw and raw not in {".", "auto", "cwd"}:
            return raw

    return _session_cwd(session)


# Git working-tree probing (run git, resolve roots, fold worktrees) lives in a
# focused, single-flight-cached module; these stay as the in-server names every
# call site already uses.
_git = git_probe.run_git
_git_branch_for_cwd = git_probe.branch
_git_repo_root_for_cwd = git_probe.repo_root
_git_common_repo_root_for_cwd = git_probe.common_repo_root
_resolve_cwd_git = git_probe.resolve


def _session_cwd(session: dict | None) -> str:
    if session and session.get("cwd"):
        return str(session["cwd"])
    return _completion_cwd()


def _heal_dead_cwd(cwd: str) -> str:
    """Resolve a session cwd that points at a now-deleted directory.

    A session anchored to a linked worktree (``<repo>/.worktrees/<name>``) keeps
    that path after the worktree is removed (branch merged, `git worktree
    remove`, etc). The literal dir is gone, so a probe of it returns nothing and
    the composer shows no branch — while the sidebar still folds the path up to
    the repo's main lane. Heal the mismatch: walk up to the first existing
    ancestor, then resolve its common git root, so a dead-worktree cwd collapses
    to the live repo root (and its real current branch).

    Only meaningful for local backends; a remote/SSH cwd may legitimately not
    exist on the host, so callers must skip healing there.
    """
    raw = (cwd or "").strip()
    if not raw or os.path.isdir(raw):
        return raw

    probe = raw
    # Climb to the first ancestor that still exists on disk.
    for _ in range(64):
        parent = os.path.dirname(probe)
        if not parent or parent == probe:
            break
        probe = parent
        if os.path.isdir(probe):
            break

    if not os.path.isdir(probe):
        return raw

    try:
        root = _git_common_repo_root_for_cwd(probe) or _git_repo_root_for_cwd(probe)
    except Exception:
        root = ""

    return root or probe


def _is_local_terminal_backend() -> bool:
    backend = (os.environ.get("TERMINAL_ENV") or "").strip().lower()
    return not backend or backend == "local"


def _display_session_cwd(session: dict | None) -> str:
    """Session cwd for display/probe surfaces, healed past deleted worktrees.

    Persists the healed value back to the session row (best-effort, local only)
    so the next load is already coherent and the sidebar lane stops showing a
    session pinned to a vanished path.
    """
    cwd = _session_cwd(session)
    if not _is_local_terminal_backend():
        return cwd

    healed = _heal_dead_cwd(cwd)
    if healed and healed != cwd and session is not None:
        session["cwd"] = healed
        try:
            with _session_db(session) as db:
                if db is not None:
                    db.update_session_cwd(session.get("session_key", ""), healed)
        except Exception:
            logger.debug("failed to persist healed session cwd", exc_info=True)
        _persist_session_git_meta(session, healed)

    return healed


def _session_source(session: dict | None) -> str:
    if session:
        source = str(session.get("source") or "").strip()
        if source:
            return source
    return _resolve_session_platform()


def _register_session_cwd(session: dict | None) -> None:
    if not session:
        return
    try:
        from tools.terminal_tool import register_task_env_overrides

        register_task_env_overrides(
            session["session_key"], {"cwd": _terminal_task_cwd(session)}
        )
    except Exception:
        pass


def _ensure_session_db_row(session: dict) -> None:
    """Idempotently persist the session's DB row on first real activity.

    Called from prompt.submit so a row only exists once the user actually sends
    a message — abandoned drafts never leave an empty "Untitled" session behind.
    Uses INSERT OR IGNORE under the hood, so re-calls (and the AIAgent's own
    lazy create) are no-ops.

    Only an *explicitly chosen* workspace is persisted as the session's cwd.
    The agent still runs in the auto-detected directory (session["cwd"]), but
    we don't stamp that onto the row — otherwise every session the user never
    picked a folder for gets grouped under whatever directory the desktop
    happened to launch in (e.g. "desktop"). Leaving it null groups them under
    "No workspace", which is the desired default.
    """
    key = session.get("session_key")
    if not key:
        return
    # Persist into the session's own profile db (global remote mode), not the
    # launch profile's — otherwise the row lands in the wrong state.db, the
    # unified list mis-tags it, and resume 404s ("session not found").
    profile_home = session.get("profile_home")
    if profile_home:
        from hermes_state import SessionDB

        try:
            db = SessionDB(db_path=Path(profile_home) / "state.db")
        except Exception:
            logger.debug("failed to open profile db for session row", exc_info=True)
            return
        close_db = True
    else:
        db = _get_db()
        close_db = False
    if db is None:
        return
    # The session's own model/effort/fast pick — the composer override shipped on
    # session.create, or a restored /model switch — must own the row's model +
    # model_config. The agent isn't built yet at first prompt.submit, so derive
    # the row from the live override dict; fall back to the global resolved model
    # only when this chat made no explicit pick. Writing the global default here
    # used to win the INSERT-OR-IGNORE race against the agent's own correct
    # lazy-create, so a reconnect/resume rebuilt from the global model and
    # silently reverted the chat (e.g. picked gpt-5.5, reconnect snapped back to
    # the profile default). model_config carries provider/reasoning/service_tier
    # so resume restores effort + fast too, not just the model name.
    override = session.get("model_override")
    override = override if isinstance(override, dict) else {}
    row_model = str(override.get("model") or "").strip() or _resolve_model()
    model_config: dict = {}
    for src_key, cfg_key in (
        ("model", "model"),
        ("provider", "provider"),
        ("base_url", "base_url"),
        ("api_mode", "api_mode"),
    ):
        if val := override.get(src_key):
            model_config[cfg_key] = str(val)
    # The composer override may carry the RESOLVED provider "custom" for a named
    # ``providers:`` / ``custom_providers:`` entry. Persisting bare "custom" here
    # (the very first DB write for a fresh desktop session, before the agent is
    # built) is the origin of the recurring "No LLM provider configured" rows:
    # on the next resume bare "custom" routes to OpenRouter with no key. Recover
    # the durable ``custom:<name>`` identity from the override's base_url, else
    # the configured provider, so a routable identity is persisted from the
    # start (matches _runtime_model_config's normalization).
    if str(model_config.get("provider") or "").strip().lower() == "custom":
        try:
            from hermes_cli.runtime_provider import canonical_custom_identity

            healed = canonical_custom_identity(
                base_url=model_config.get("base_url") or None
            )
            if healed:
                model_config["provider"] = healed
        except Exception:
            logger.debug(
                "custom provider identity recovery failed (db row)", exc_info=True
            )
    if (reasoning := session.get("create_reasoning_override")) is not None:
        model_config["reasoning_config"] = reasoning
    if tier := session.get("create_service_tier_override"):
        model_config["service_tier"] = tier
    # Branch lineage: stamp the same ``_branched_from`` marker the TUI /branch
    # uses so list_sessions_rich keeps the branch listed and the desktop sidebar
    # can nest it under its parent.
    parent_session_id = session.get("parent_session_id") or None
    if parent_session_id:
        model_config["_branched_from"] = parent_session_id
    try:
        db.create_session(
            key,
            source=_session_source(session),
            model=row_model,
            model_config=model_config or None,
            parent_session_id=parent_session_id,
            cwd=_session_cwd(session) if session.get("explicit_cwd") else None,
        )
    except Exception:
        logger.debug("failed to persist desktop session row", exc_info=True)
    finally:
        if close_db:
            try:
                db.close()
            except Exception:
                pass


def _persist_branch_seed(session: dict) -> None:
    """First-turn persist of a branch's copied transcript.

    A branch is a draft until its first submit: the parent's messages live only
    in ``session["history"]`` (they ride into the agent as ``conversation_history``,
    which ``_flush_messages_to_session_db`` skips by identity). Without this the
    branch row would resume missing its pre-branch context. Runs once; the row +
    parent link are written by ``_ensure_session_db_row`` just before this.
    """
    if not session.get("parent_session_id") or session.get("_branch_seed_persisted"):
        return
    key = session.get("session_key")
    if not key:
        return
    with session["history_lock"]:
        seed = [dict(msg) for msg in (session.get("history") or [])]
    if not seed:
        return
    with _session_db(session) as db:
        if db is None:
            return
        try:
            for msg in seed:
                db.append_message(session_id=key, role=msg.get("role", "user"), content=msg.get("content"))
            session["_branch_seed_persisted"] = True
        except Exception:
            logger.debug("branch seed persist failed", exc_info=True)


@contextlib.contextmanager
def _session_db(session: dict):
    """Yield the SessionDB that owns this session's row (profile-aware).

    Mirrors :func:`_ensure_session_db_row`: a remote/profile session persists
    into its own profile's ``state.db`` (a fresh handle we close on exit);
    everything else borrows the shared ``_get_db()`` handle (left open). Yields
    None when the db is unavailable.
    """
    db, close_db = None, False
    profile_home = session.get("profile_home")
    if profile_home:
        from hermes_state import SessionDB

        try:
            db, close_db = SessionDB(db_path=Path(profile_home) / "state.db"), True
        except Exception:
            logger.debug("failed to open profile db for session", exc_info=True)
    else:
        db = _get_db()
    try:
        yield db
    finally:
        if close_db and db is not None:
            with contextlib.suppress(Exception):
                db.close()


def _session_db_for_immediate_use(session: dict):
    """Return a SessionDB handle for immediate (non-context-manager) use.

    Profile sessions open a dedicated ``state.db`` that the caller must close
    in a ``finally`` block. Default sessions borrow the shared ``_get_db()``.
    """
    profile_home = session.get("profile_home")
    if profile_home:
        from hermes_state import SessionDB

        try:
            return SessionDB(db_path=Path(profile_home) / "state.db")
        except Exception:
            logger.debug("failed to open profile db for session", exc_info=True)
            return None
    return _get_db()


def _persist_session_git_meta(session: dict, cwd: str) -> None:
    """Resolve + persist a session's git branch / repo root WITHOUT blocking.

    Branch and root come from ``git`` subprocess probes; running them inline on
    the session-init / cwd-set path would stall startup whenever ``cwd`` is slow
    or on an unreachable mount. Run them on a short-lived daemon thread instead
    and persist via the same profile-aware db the caller writes ``cwd`` to.

    Best-effort: ``cwd`` itself is persisted synchronously by the caller, so a
    probe failure just leaves these enrichment columns unset (the project tree
    falls back to its live resolver / lazy backfill). Daemon, so a mid-flight
    probe never delays gateway shutdown.
    """
    session_key = session.get("session_key", "")
    if not session_key or not cwd:
        return
    # Snapshot the routing fields now; the live session dict may be gone by the
    # time the thread runs. `_session_db` reopens the profile-correct db inside.
    db_session = {"session_key": session_key, "profile_home": session.get("profile_home")}

    def _run() -> None:
        try:
            branch = _git_branch_for_cwd(cwd)
            root = _git_common_repo_root_for_cwd(cwd)
            if not (branch or root):
                return
            with _session_db(db_session) as db:
                if db is not None:
                    db.update_session_cwd(session_key, cwd, branch, root)
        except Exception:
            logger.debug("failed to persist session git metadata", exc_info=True)

    threading.Thread(target=_run, name="git-meta", daemon=True).start()


def _set_session_cwd(session: dict, cwd: str) -> str:
    resolved = os.path.abspath(os.path.expanduser(str(cwd)))
    if not os.path.isdir(resolved):
        raise ValueError(f"working directory does not exist: {cwd}")
    session["cwd"] = resolved
    # An explicit user choice — persist it as the workspace (and let a later
    # lazy row creation persist it too, not the launch-dir fallback).
    session["explicit_cwd"] = True
    _register_session_cwd(session)
    with _session_db(session) as db:
        if db is not None:
            try:
                db.update_session_cwd(session.get("session_key", ""), resolved)
            except Exception:
                logger.debug("failed to persist session cwd", exc_info=True)
    # Branch/repo-root probes are git subprocesses — capture them off the hot path.
    _persist_session_git_meta(session, resolved)
    try:
        from tools.terminal_tool import cleanup_vm

        cleanup_vm(session["session_key"])
    except Exception:
        pass
    return resolved


# ── Config I/O ────────────────────────────────────────────────────────


# Keep aligned with `INDICATOR_STYLES` / `DEFAULT_INDICATOR_STYLE` in
# ``ui-tui/src/app/interfaces.ts`` — both ends validate against the
# same shape so `config.get indicator` and the live TUI render agree.
_INDICATOR_STYLES: tuple[str, ...] = ("ascii", "emoji", "kaomoji", "unicode")
_INDICATOR_DEFAULT = "kaomoji"


def _load_cfg() -> dict:
    global _cfg_cache, _cfg_mtime, _cfg_path
    try:
        import yaml

        # Honor a per-session profile override (see session.resume) so a resumed
        # remote profile loads ITS config (model, skills, prompt); otherwise the
        # launch profile's _hermes_home. Cache is keyed on the resolved path, so
        # profiles don't clobber each other.
        override = get_hermes_home_override()
        home = override if isinstance(override, str) and override else _hermes_home
        p = Path(home) / "config.yaml"
        mtime = p.stat().st_mtime if p.exists() else None
        with _cfg_lock:
            if _cfg_cache is not None and _cfg_mtime == mtime and _cfg_path == p:
                return _apply_managed(copy.deepcopy(_cfg_cache))
        if p.exists():
            with open(p, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        else:
            data = {}
        with _cfg_lock:
            # Cache the RAW user config (no managed overlay) so _save_cfg, which
            # writes _cfg_cache back to disk, never persists managed values into
            # the user's file. The managed overlay is applied on every return
            # path instead (read-side only).
            _cfg_cache = copy.deepcopy(data)
            _cfg_mtime = mtime
            _cfg_path = p
        return _apply_managed(data)
    except Exception:
        pass
    return {}


def _apply_managed(cfg: dict) -> dict:
    """Overlay administrator-pinned managed-scope values on a config dict.

    The TUI/desktop backend builds config independently of
    hermes_cli.config.load_config, so without this a managed skin / reasoning_effort
    / service_tier / provider_routing would be silently ignored here. Read-side
    only — the raw user config is what gets cached and saved. Fail-open.
    """
    try:
        from hermes_cli import managed_scope

        return managed_scope.apply_managed_overlay(cfg if isinstance(cfg, dict) else {})
    except Exception:
        return cfg


def _save_cfg(cfg: dict):
    global _cfg_cache, _cfg_mtime, _cfg_path

    from hermes_cli.config import atomic_config_write

    path = _hermes_home / "config.yaml"
    atomic_config_write(path, cfg)
    with _cfg_lock:
        _cfg_cache = copy.deepcopy(cfg)
        _cfg_path = path
        try:
            _cfg_mtime = path.stat().st_mtime
        except Exception:
            _cfg_mtime = None


def _cwd_for_session_key(session_key: str) -> str:
    """Reverse-map session_key to the session's logical cwd.

    Snapshots ``_sessions`` first: concurrent RPC handlers mutate it from the
    thread pool, so iterating the live view risks ``RuntimeError: dictionary
    changed size during iteration``.
    """
    if not session_key:
        return ""
    with _sessions_lock:
        for sess in list(_sessions.values()):
            if sess.get("session_key") == session_key:
                return str(sess.get("cwd") or "")
    return ""


def _set_session_context(
    session_key: str,
    cwd: str | None = None,
    *,
    ui_session_id: str = "",
) -> list:
    try:
        from gateway.session_context import set_session_vars

        # Ephemeral task IDs (background, preview) aren't in `_sessions`, so the
        # reverse-map returns "" and would clear the cwd override. Callers that
        # know the parent workspace pass it explicitly so spawned agents inherit
        # it instead of falling back to the gateway launch dir.
        resolved = cwd if cwd is not None else _cwd_for_session_key(session_key)
        source = _resolve_session_platform()
        with _sessions_lock:
            for sess in list(_sessions.values()):
                if sess.get("session_key") == session_key:
                    source = _session_source(sess)
                    break
        return set_session_vars(
            session_key=session_key,
            source=source,
            cwd=resolved,
            ui_session_id=ui_session_id,
        )
    except Exception:
        return []


def _clear_session_context(tokens: list) -> None:
    if not tokens:
        return
    try:
        from gateway.session_context import clear_session_vars

        clear_session_vars(tokens)
    except Exception:
        pass


def _enable_gateway_prompts() -> None:
    """Route approvals through gateway callbacks instead of CLI input()."""
    os.environ["HERMES_GATEWAY_SESSION"] = "1"
    os.environ["HERMES_EXEC_ASK"] = "1"
    os.environ["HERMES_INTERACTIVE"] = "1"


# ── Blocking prompt factory ──────────────────────────────────────────


def _block(event: str, sid: str, payload: dict, timeout: int = 300) -> str:
    rid = uuid.uuid4().hex[:8]
    ev = threading.Event()
    with _prompt_lock:
        _pending[rid] = (sid, ev)
        payload["request_id"] = rid
        _pending_prompt_payloads[rid] = (event, dict(payload))
    try:
        _emit(event, sid, payload)
        ev.wait(timeout=timeout)
    finally:
        with _prompt_lock:
            _pending.pop(rid, None)
            _pending_prompt_payloads.pop(rid, None)
    with _prompt_lock:
        return _answers.pop(rid, "")


def _clear_pending(sid: str | None = None) -> None:
    """Release pending prompts with an empty answer.

    When *sid* is provided, only prompts owned by that session are
    released — critical for session.interrupt, which must not
    collaterally cancel clarify/sudo/secret prompts on unrelated
    sessions sharing the same tui_gateway process.  When *sid* is
    None, every pending prompt is released (used during shutdown).
    """
    with _prompt_lock:
        for rid, (owner_sid, ev) in list(_pending.items()):
            if sid is None or owner_sid == sid:
                _answers[rid] = ""
                ev.set()


# ── Agent factory ────────────────────────────────────────────────────


def resolve_skin() -> dict:
    try:
        from hermes_cli.skin_engine import init_skin_from_config, get_active_skin

        init_skin_from_config(_load_cfg())
        skin = get_active_skin()
        return {
            "name": skin.name,
            "colors": skin.colors,
            "branding": skin.branding,
            "banner_logo": skin.banner_logo,
            "banner_hero": skin.banner_hero,
            "tool_prefix": skin.tool_prefix,
            "help_header": (skin.branding or {}).get("help_header", ""),
        }
    except Exception:
        return {}


def _resolve_model() -> str:
    env = (
        os.environ.get("HERMES_MODEL", "")
        or os.environ.get("HERMES_INFERENCE_MODEL", "")
    ).strip()
    if env:
        return env
    m = _load_cfg().get("model", "")
    if isinstance(m, dict):
        return str(m.get("default", "") or "").strip()
    if isinstance(m, str) and m:
        return m.strip()
    return "anthropic/claude-sonnet-4"


def _resolve_session_platform() -> str:
    """Resolve the platform tag for a tui_gateway-routed session.

    The desktop app's chat panel and the standalone TUI both speak to this
    gateway; without a branch they all get stamped ``platform="tui"``,
    which makes the agent think it's talking to a terminal user. That
    mis-tag is the root cause of the desktop chat agent suggesting
    TUI-only slash commands (``/reload-mcp``, …) to chat-panel users.

    Resolution:
      * ``HERMES_DESKTOP=1`` and ``HERMES_DESKTOP_TERMINAL`` unset → "desktop"
        (the chat-panel backend — a graphical React surface, not a terminal).
      * ``HERMES_DESKTOP_TERMINAL=1`` → "tui"
        (``hermes --tui`` running in the desktop's embedded terminal pane;
        it IS a TUI, just embedded. The clarifier attached to the tui hint
        in system_prompt.py tells the agent about the embedding.)
      * neither set → "tui"
        (standalone ``hermes --tui``.)
    """
    if is_truthy_value(os.environ.get("HERMES_DESKTOP")) and not is_truthy_value(
        os.environ.get("HERMES_DESKTOP_TERMINAL")
    ):
        return "desktop"
    return "tui"


def _resolve_session_source(explicit: str | None) -> str:
    """Default the session DB ``source`` field from the resolved platform.

    A caller that explicitly passes ``source`` (e.g. a plugin session tagged
    ``"telegram"``) keeps its value. Only an empty/None ``source`` falls back
    to the env-resolved platform — so env-driven resolution never silently
    rewrites a caller's intent.
    """
    if explicit:
        return explicit
    return _resolve_session_platform()


def _resolve_agent_platform(source: str | None) -> str:
    return _resolve_session_source(source)


def _config_model_target() -> tuple[str, str]:
    """(model, provider) currently selected by config.yaml — and ONLY config.

    Unlike `_resolve_model()`, this never reads HERMES_MODEL /
    HERMES_INFERENCE_MODEL. Those env vars are a launch-scoped seed
    (`hermes --tui -m <model>`, hosted-instance provisioning); if they
    fed the per-turn sync, the seed would be replayed as a /model switch
    and persisted globally, or would pin the session so dashboard/CLI
    model changes never reach an open chat.
    """
    cfg_model = _load_cfg().get("model")
    model = ""
    provider = ""
    if isinstance(cfg_model, dict):
        model = str(cfg_model.get("default", "") or "").strip()
        provider = str(cfg_model.get("provider") or "").strip()
        if provider.lower() == "auto":
            provider = ""
    elif isinstance(cfg_model, str):
        model = cfg_model.strip()
    # No fallback to _resolve_model() here: that reads HERMES_MODEL /
    # HERMES_INFERENCE_MODEL, which `hermes --tui -m <model>` sets as a
    # session-scoped seed for THIS launch. When config.yaml has no
    # model.default (custom-provider-only setups), falling back to the env
    # seed made the per-turn sync treat the -m flag as "the configured
    # model" and replay it as a /model switch — which then persisted the
    # one-shot flag into config.yaml globally (#-m leak). An empty model
    # simply means "config expresses no preference": the sync is a no-op
    # and the agent keeps whatever it was built with.
    return model, provider


def _resolve_startup_runtime() -> tuple[str, str | None]:
    model = _resolve_model()
    explicit_provider = os.environ.get("HERMES_TUI_PROVIDER", "").strip()
    if explicit_provider:
        return model, explicit_provider

    explicit_model = (
        os.environ.get("HERMES_MODEL", "")
        or os.environ.get("HERMES_INFERENCE_MODEL", "")
    ).strip()
    if not explicit_model:
        return model, None

    try:
        from hermes_cli.models import detect_static_provider_for_model

        cfg = _load_cfg().get("model") or {}
        current_provider = (
            (
                str(cfg.get("provider") or "").strip().lower()
                if isinstance(cfg, dict)
                else ""
            )
            or os.environ.get("HERMES_INFERENCE_PROVIDER", "").strip().lower()
            or "auto"
        )
        detected = detect_static_provider_for_model(explicit_model, current_provider)
        if detected:
            provider, detected_model = detected
            return detected_model, provider
    except Exception:
        pass
    return model, None


# Bare billing buckets are not routable provider identities (kept in parity with the
# provider gate in agent_init). Restoring one as a session provider override breaks resume.
_BARE_BILLING_PROVIDERS = {"auto", "openrouter", "custom"}


def _stored_session_runtime_overrides(row: dict | None) -> dict:
    """Return runtime fields persisted with a stored session.

    ``session.resume`` is a session-scoped operation: reopening an older chat
    must restore the model/provider/reasoning state that chat actually used,
    not whatever global model the user most recently selected in another chat.
    The durable session row stores the model directly, the billing provider in
    ``billing_provider``, and richer runtime knobs in JSON ``model_config``.
    """
    if not row:
        return {}

    raw_config = row.get("model_config")
    model_config: dict = {}
    if isinstance(raw_config, dict):
        model_config = raw_config
    elif isinstance(raw_config, str) and raw_config.strip():
        try:
            parsed = json.loads(raw_config)
            if isinstance(parsed, dict):
                model_config = parsed
        except Exception:
            logger.debug("failed to parse stored session model_config", exc_info=True)

    overrides: dict = {}
    model = str(row.get("model") or model_config.get("model") or "").strip()
    # ``billing_provider`` is only the billing bucket — for a custom endpoint it is the
    # bare class ``"custom"``, which agent_init treats as non-routable, so restoring it as
    # the provider override makes ``session.resume`` fail with "No LLM provider configured".
    # Only restore an explicit provider; otherwise leave it unset so resume falls back to
    # the configured default, matching the working CLI path.
    explicit_provider = str(model_config.get("provider") or "").strip()
    billing_provider = str(
        model_config.get("billing_provider") or row.get("billing_provider") or ""
    ).strip()
    provider = explicit_provider
    if not provider and billing_provider.lower() not in _BARE_BILLING_PROVIDERS:
        provider = billing_provider
    base_url = str(model_config.get("base_url") or "").strip()
    api_mode = str(model_config.get("api_mode") or "").strip()
    reasoning_config = model_config.get("reasoning_config")
    service_tier = str(model_config.get("service_tier") or "").strip()

    # Heal a bare ``"custom"`` provider stored by an older build (or any leak
    # site that bypassed _runtime_model_config's normalization). Bare custom is
    # the resolved billing class, not a routable identity — restoring it as the
    # session's provider override routes the resume to the OpenRouter default
    # URL with no api_key, surfacing as "No LLM provider configured". Recover
    # the durable ``custom:<name>`` menu key from the stored base_url, falling
    # back to the configured provider when the row has no base_url (the
    # recurring Desktop/TUI regression vector). If neither names a real entry,
    # drop the bare provider entirely so resume falls back to the configured
    # default rather than the broken OpenRouter route.
    if provider.strip().lower() == "custom":
        healed = None
        try:
            from hermes_cli.runtime_provider import canonical_custom_identity

            healed = canonical_custom_identity(base_url=base_url or None)
        except Exception:
            logger.debug(
                "custom provider identity recovery failed", exc_info=True
            )
        provider = healed or ("" if not base_url else provider)

    if model:
        # Use the same dict-shaped override that live /model switches use so a
        # DB-restored session can preserve custom endpoint metadata across both
        # initial resume and later rebuilds (/new). Deliberately do not persist
        # or restore raw api_key here; endpoint credentials should continue to
        # come from config/env/provider resolution rather than the session DB.
        overrides["model_override"] = {
            "model": model,
            "provider": provider or None,
            "base_url": base_url or None,
            "api_mode": api_mode or None,
        }
    if provider:
        overrides["provider_override"] = provider
    if isinstance(reasoning_config, dict):
        overrides["reasoning_config_override"] = reasoning_config
    if service_tier:
        overrides["service_tier_override"] = service_tier

    return overrides


def _runtime_model_config(agent, existing: dict | None = None) -> dict:
    config = dict(existing or {})
    model = str(getattr(agent, "model", "") or "").strip()
    provider = str(getattr(agent, "provider", "") or "").strip()
    base_url = str(getattr(agent, "base_url", "") or "").strip()
    api_mode = str(getattr(agent, "api_mode", "") or "").strip()
    reasoning_config = getattr(agent, "reasoning_config", None)
    service_tier = getattr(agent, "service_tier", None)

    if model:
        config["model"] = model
    if provider:
        if provider.strip().lower() == "custom":
            # ``agent.provider`` is the RESOLVED provider, and for any named
            # ``providers:`` / ``custom_providers:`` entry that is the literal
            # string "custom" — persisting it loses the entry identity, so a
            # later resume/rebuild cannot re-resolve the entry's credentials
            # (the api_key is deliberately never persisted; see
            # _stored_session_runtime_overrides). Recover the canonical
            # ``custom:<name>`` menu key from the endpoint URL when present,
            # else from the configured provider — this second fallback is the
            # fix for sessions built WITHOUT a base_url on the override (the
            # recurring Desktop/TUI "No LLM provider configured" regression:
            # bare "custom" with no base_url was persisted verbatim and routed
            # to OpenRouter with no key on the next resume).
            try:
                from hermes_cli.runtime_provider import (
                    canonical_custom_identity,
                )

                provider = (
                    canonical_custom_identity(base_url=base_url) or provider
                )
            except Exception:
                logger.debug(
                    "custom provider identity lookup failed", exc_info=True
                )
        config["provider"] = provider
    if base_url:
        config["base_url"] = base_url
    else:
        config.pop("base_url", None)
    if api_mode:
        config["api_mode"] = api_mode
    else:
        config.pop("api_mode", None)
    if isinstance(reasoning_config, dict):
        config["reasoning_config"] = reasoning_config
    else:
        config.pop("reasoning_config", None)
    if service_tier:
        config["service_tier"] = service_tier
    else:
        config.pop("service_tier", None)

    return config


def _persist_live_session_runtime(session: dict | None) -> None:
    """Persist active session runtime so future resumes restore the same footer."""
    if not session:
        return
    agent = session.get("agent")
    session_key = str(session.get("session_key") or "").strip()
    if agent is None or not session_key:
        return

    db = getattr(agent, "_session_db", None) or _get_db()
    if db is None:
        return

    try:
        row = db.get_session(session_key) or {}
        raw_config = row.get("model_config")
        existing_config = {}
        if isinstance(raw_config, dict):
            existing_config = raw_config
        elif isinstance(raw_config, str) and raw_config.strip():
            parsed = json.loads(raw_config)
            if isinstance(parsed, dict):
                existing_config = parsed
        model_config = _runtime_model_config(agent, existing_config)
        model = str(getattr(agent, "model", "") or "").strip()
        if hasattr(db, "update_session_meta"):
            db.update_session_meta(session_key, json.dumps(model_config), model or None)
        elif model and hasattr(db, "update_session_model"):
            db.update_session_model(session_key, model)
    except Exception:
        logger.debug("failed to persist live session runtime", exc_info=True)


def _persist_live_session_system_prompt(session: dict | None) -> None:
    """Refresh the stored system prompt after a live runtime identity change."""
    if not session:
        return
    agent = session.get("agent")
    session_key = str(session.get("session_key") or "").strip()
    if agent is None or not session_key or not hasattr(agent, "_build_system_prompt"):
        return

    db = getattr(agent, "_session_db", None) or _get_db()
    if db is None or not hasattr(db, "update_system_prompt"):
        return

    try:
        prompt = agent._build_system_prompt(None)
        agent._cached_system_prompt = prompt
        db.update_system_prompt(getattr(agent, "session_id", None) or session_key, prompt)
    except Exception:
        logger.debug("failed to persist live session system prompt", exc_info=True)


def _append_model_switch_marker(session: dict | None, *, model: str, provider: str) -> None:
    """Record a real system-history pivot after a live model switch."""
    if not session:
        return
    session_key = str(session.get("session_key") or "").strip()
    if not session_key:
        return

    provider_part = f" via provider {provider}" if provider else ""
    marker = (
        "[System: The active model for this chat has changed to "
        f"{model}{provider_part}. From this point forward, use this runtime "
        "metadata when answering questions about what model/provider is active.]"
    )
    # Persist as a user message, not a system message.  The gateway appends
    # this marker after prior conversation turns, and strict OpenAI-compatible
    # providers (vLLM, Qwen) reject system messages that are not at the
    # beginning of the API message list (#48338).
    entry = {"role": "user", "content": marker}

    lock = session.get("history_lock")
    if lock is not None:
        with lock:
            session.setdefault("history", []).append(entry)
            session["history_version"] = int(session.get("history_version", 0)) + 1
    else:
        session.setdefault("history", []).append(entry)
        session["history_version"] = int(session.get("history_version", 0)) + 1

    try:
        agent = session.get("agent")
        db = getattr(agent, "_session_db", None) if agent is not None else None
        if db is not None:
            db.append_message(session_id=session_key, role="user", content=marker)
            return

        _ensure_session_db_row(session)
        with _session_db(session) as scoped_db:
            if scoped_db is not None:
                scoped_db.append_message(
                    session_id=session_key, role="user", content=marker
                )
    except Exception:
        logger.debug("failed to persist model switch marker", exc_info=True)


def _write_config_key(key_path: str, value):
    cfg = _load_cfg()
    current = cfg
    keys = key_path.split(".")
    for key in keys[:-1]:
        if key not in current or not isinstance(current.get(key), dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value
    _save_cfg(cfg)


_STATUSBAR_MODES = frozenset({"off", "top", "bottom"})


def _coerce_statusbar(raw) -> str:
    if raw is False:
        return "off"
    if isinstance(raw, str) and (s := raw.strip().lower()) in _STATUSBAR_MODES:
        return s
    return "top"


_MOUSE_TRACKING_ALIASES = {
    "0": "off",
    "1": "all",
    "all": "all",
    "any": "all",
    "button": "buttons",
    "buttons": "buttons",
    "click": "buttons",
    "false": "off",
    "full": "all",
    "no": "off",
    "off": "off",
    "on": "all",
    "scroll": "wheel",
    "true": "all",
    "wheel": "wheel",
    "yes": "all",
}


def _display_mouse_tracking(display: dict) -> str:
    """Resolve display.mouse_tracking to one of ``off|wheel|buttons|all``.

    Boolean values keep their legacy meaning (``True`` → ``all``, ``False`` →
    ``off``). The ``wheel`` preset (DEC 1000+1006) is the tmux-friendly
    subset — wheel + click only, no hover events to trigger prompt-row
    clipboard probes. Legacy ``tui_mouse`` is honored only when
    ``mouse_tracking`` is absent.
    """
    if not isinstance(display, dict):
        return "all"
    if "mouse_tracking" in display:
        raw = display.get("mouse_tracking")
    else:
        raw = display.get("tui_mouse", True)
    if raw is False or raw == 0:
        return "off"
    if raw is True or raw is None:
        return "all"
    if isinstance(raw, (int, float)):
        return "all"
    if isinstance(raw, str):
        return _MOUSE_TRACKING_ALIASES.get(raw.strip().lower(), "all")
    return "all"


def _load_reasoning_config() -> dict | None:
    from hermes_constants import parse_reasoning_effort

    # Pass the raw value through — ``or ""`` would coerce a YAML boolean
    # False (``reasoning_effort: false``/``off``/``no``) to "", silently
    # re-enabling thinking for users who explicitly turned it off.
    return parse_reasoning_effort(
        (_load_cfg().get("agent") or {}).get("reasoning_effort", "")
    )


def _load_service_tier() -> str | None:
    raw = (
        str((_load_cfg().get("agent") or {}).get("service_tier", "") or "")
        .strip()
        .lower()
    )
    if not raw or raw in {"normal", "default", "standard", "off", "none"}:
        return None
    if raw in {"fast", "priority", "on"}:
        return "priority"
    return None


def _load_provider_routing() -> dict:
    """OpenRouter provider-routing prefs from config.yaml (``provider_routing``).

    Parity with the messaging gateway (``gateway/run.py::_load_provider_routing``)
    and the classic CLI: without this the desktop/TUI backend builds agents with
    no routing prefs, so OpenRouter falls back to its default (effectively random)
    provider selection even when the user configured ``provider_routing``.
    """
    try:
        return _load_cfg().get("provider_routing", {}) or {}
    except Exception:
        return {}


def _load_show_reasoning() -> bool:
    # Fallback True — keep in sync with DEFAULT_CONFIG display.show_reasoning
    # (this loader reads the raw user YAML without the DEFAULT_CONFIG merge).
    return bool((_load_cfg().get("display") or {}).get("show_reasoning", True))


def _load_memory_notifications() -> str:
    """Self-improvement review notification mode from config.yaml.

    Parity with the messaging gateway (``gateway/run.py``) and the classic CLI:
    ``display.memory_notifications`` controls whether the background review's
    "💾 Self-improvement review: …" summary is surfaced. Without this the
    TUI/desktop backend always behaved as ``"on"`` and silently ignored a user
    who set ``off``. Accepts ``off`` / ``on`` (default) / ``verbose``; a bool is
    normalized for back-compat.
    """
    raw = (_load_cfg().get("display") or {}).get("memory_notifications")
    if isinstance(raw, bool):
        return "on" if raw else "off"
    return str(raw).lower() if raw else "on"


def _load_tool_progress_mode() -> str:
    env = os.environ.get("HERMES_TUI_TOOL_PROGRESS", "").strip().lower()
    if env in {"off", "new", "all", "verbose"}:
        return env
    raw = (_load_cfg().get("display") or {}).get("tool_progress", "all")
    if raw is False:
        return "off"
    if raw is True:
        return "all"
    mode = str(raw or "all").strip().lower()
    return mode if mode in {"off", "new", "all", "verbose"} else "all"


def _load_enabled_toolsets() -> list[str] | None:
    explicit = [
        item.strip()
        for item in os.environ.get("HERMES_TUI_TOOLSETS", "").split(",")
        if item.strip()
    ]
    cfg = None
    fallback_notice = None

    # Coding posture (base Hermes): with no explicit pin, collapse to the
    # coding toolset (+ enabled MCP servers) when sitting in a code workspace.
    # The desktop app and `hermes --tui` both land here. See
    # agent/coding_context.py. No config is loaded yet at this point, so we let
    # coding_selection() load it lazily (cli.py passes its already-resolved
    # CLI_CONFIG instead, purely to avoid a redundant read).
    if not explicit:
        try:
            from agent.coding_context import coding_selection

            selection = coding_selection(platform=_resolve_session_platform())
            if selection is not None:
                # Fold in `project` here too: this is a GUI-only resolver, and
                # the focus-mode coding posture returns before the fallback path
                # that normally adds it — without this the desktop loses the
                # project tools exactly when sitting in a repo (see below).
                return sorted({*selection, "project"})
        except Exception:
            pass

    try:
        from toolsets import validate_toolset
    except Exception:
        validate_toolset = None

    if explicit and validate_toolset is not None:
        built_in = [name for name in explicit if validate_toolset(name)]
        unresolved = [name for name in explicit if name not in built_in]

        if unresolved:
            try:
                from hermes_cli.plugins import discover_plugins

                discover_plugins()
                plugin_valid = [name for name in unresolved if validate_toolset(name)]
            except Exception:
                plugin_valid = []

            if plugin_valid:
                built_in.extend(plugin_valid)
                unresolved = [name for name in unresolved if name not in plugin_valid]

        if any(name in {"all", "*"} for name in built_in):
            ignored = [name for name in explicit if name not in {"all", "*"}]
            if ignored:
                print(
                    "[tui] HERMES_TUI_TOOLSETS=all enables every toolset; "
                    f"ignoring additional entries: {', '.join(ignored)}",
                    file=sys.stderr,
                    flush=True,
                )
            return None

        if not unresolved:
            return built_in

        mcp_names: set[str] = set()
        mcp_disabled: set[str] = set()
        try:
            from hermes_cli.config import read_raw_config
            from hermes_cli.tools_config import _parse_enabled_flag

            raw_cfg = read_raw_config()
            mcp_servers = (
                raw_cfg.get("mcp_servers")
                if isinstance(raw_cfg.get("mcp_servers"), dict)
                else {}
            )
            for name, server_cfg in mcp_servers.items():
                if not isinstance(server_cfg, dict):
                    continue
                if _parse_enabled_flag(server_cfg.get("enabled", True), default=True):
                    mcp_names.add(str(name))
                else:
                    mcp_disabled.add(str(name))
        except Exception:
            mcp_names = set()
            mcp_disabled = set()

        mcp_valid = [name for name in unresolved if name in mcp_names]
        disabled = [name for name in unresolved if name in mcp_disabled]
        unknown = [
            name
            for name in unresolved
            if name not in mcp_names and name not in mcp_disabled
        ]
        valid = built_in + mcp_valid

        if unknown:
            print(
                f"[tui] ignoring unknown HERMES_TUI_TOOLSETS entries: {', '.join(unknown)}",
                file=sys.stderr,
                flush=True,
            )
        if disabled:
            print(
                "[tui] ignoring disabled MCP servers in HERMES_TUI_TOOLSETS "
                "(set enabled: true in config.yaml to use): "
                f"{', '.join(disabled)}",
                file=sys.stderr,
                flush=True,
            )

        if valid:
            return valid

        fallback_notice = (
            "[tui] no valid HERMES_TUI_TOOLSETS entries; using configured CLI toolsets"
        )

    try:
        from hermes_cli.config import load_config
        from hermes_cli.tools_config import _get_platform_tools

        cfg = cfg if cfg is not None else load_config()

        # Runtime toolset resolution must include default MCP servers so the
        # agent can actually call them. Passing ``False`` here is the
        # config-editing variant — used when we need to persist a toolset
        # list without baking in implicit MCP defaults. Using the wrong
        # variant at agent creation time makes MCP tools silently missing
        # from the TUI. See PR #3252 for the original design split.
        enabled = _get_platform_tools(cfg, "cli", include_default_mcp_servers=True)
        if fallback_notice is not None:
            print(fallback_notice, file=sys.stderr, flush=True)
        if not enabled:
            return None
        # The desktop Project tools are off _HERMES_CORE_TOOLS (every other
        # platform would carry their schema for nothing), so the platform
        # recovery above — which keys off hermes-cli's tool universe — can't
        # surface them. This resolver runs ONLY in the desktop/TUI gateway, so
        # folding in the `project` toolset here is the gate that exposes them on
        # exactly the surface that can follow a project move.
        return sorted(enabled | {"project"})
    except Exception:
        if fallback_notice is not None:
            print(
                "[tui] no valid HERMES_TUI_TOOLSETS entries and configured CLI toolsets could not be loaded; enabling all toolsets",
                file=sys.stderr,
                flush=True,
            )
        return None


def _session_tool_progress_mode(sid: str) -> str:
    return str(_sessions.get(sid, {}).get("tool_progress_mode", "all") or "all")


def _session_verbose(sid: str) -> bool:
    return _session_tool_progress_mode(sid) == "verbose"


def _tool_progress_enabled(sid: str) -> bool:
    return _session_tool_progress_mode(sid) != "off"


def _restart_slash_worker(sid: str, session: dict):
    worker = session.get("slash_worker")
    if worker:
        try:
            worker.close()
        except Exception:
            pass
    try:
        new_worker = _SlashWorker(
            session["session_key"],
            getattr(session.get("agent"), "model", _resolve_model()),
            profile_home=session.get("profile_home"),
        )
    except Exception:
        session["slash_worker"] = None
        return
    # Route through the same store-iff-still-mapped guard as the spawn sites:
    # the post-turn restart runs as `running` flips false, exactly when a
    # close_on_disconnect reap can pop this session — a bare store would orphan
    # the fresh worker (it self-heals only on gateway exit via the watchdog).
    _attach_worker(sid, session, new_worker)


def _persist_model_switch(result) -> None:
    # Use targeted, atomic key writes (comment/ordering-preserving) instead of
    # rewriting the whole `model:` block. A full-block rewrite via save_config()
    # destroys sibling keys the user set under `model:` — `model_slots`,
    # `model_fallback`, etc. — when switching models from the TUI (#48305).
    from cli import save_config_value

    save_config_value("model.default", result.new_model)
    save_config_value("model.provider", result.target_provider)
    if result.base_url:
        save_config_value("model.base_url", result.base_url)
    else:
        # Clear any stale base_url when switching to a provider that doesn't use
        # one (e.g. custom endpoint -> native provider). Reads coalesce null to
        # absent (`model_cfg.get("base_url") or ""`), so a null is equivalent to
        # removal without needing a key-delete. Leaving the old value would
        # route the new model at the previous custom host (#48305).
        save_config_value("model.base_url", None)


def _apply_model_switch(
    sid: str,
    session: dict,
    raw_input: str,
    *,
    confirm_expensive_model: bool = False,
    pin_session_override: bool = True,
    parsed_flags: tuple[str, str, bool, bool, bool] | None = None,
    persist_override: bool | None = None,
) -> dict:
    from hermes_cli.model_switch import (
        parse_model_flags,
        resolve_persist_behavior,
        switch_model,
    )
    from hermes_cli.runtime_provider import resolve_runtime_provider

    if parsed_flags is None:
        parsed_flags = parse_model_flags(raw_input)
    (
        model_input,
        explicit_provider,
        is_global_flag,
        _force_refresh,
        is_session,
    ) = parsed_flags
    persist_global = (
        persist_override
        if persist_override is not None
        else resolve_persist_behavior(is_global_flag, is_session)
    )
    if not model_input:
        raise ValueError("model value required")

    agent = session.get("agent")
    if agent:
        current_provider = getattr(agent, "provider", "") or ""
        current_model = getattr(agent, "model", "") or ""
        current_base_url = getattr(agent, "base_url", "") or ""
        current_api_key = getattr(agent, "api_key", "") or ""
    else:
        current_model = _resolve_model()
        current_provider = explicit_provider.strip()
        current_base_url = ""
        current_api_key = ""
        if not explicit_provider:
            runtime = resolve_runtime_provider(requested=None)
            current_provider = str(runtime.get("provider", "") or "")
            current_base_url = str(runtime.get("base_url", "") or "")
            # Preserve a callable api_key (Azure Foundry Entra ID bearer
            # provider) unchanged — ``str(...)`` would produce
            # ``"<function ...>"`` and poison downstream switch_model
            # validation. Match the agent-present branch's behavior at the
            # top of this block.
            _runtime_key = runtime.get("api_key", "")
            if callable(_runtime_key) and not isinstance(_runtime_key, str):
                current_api_key = _runtime_key
            else:
                current_api_key = str(_runtime_key or "")

    # Load user-defined providers so switch_model can resolve named custom
    # endpoints (e.g. "ollama-launch") and validate against saved model lists.
    user_provs = None
    custom_provs = None
    try:
        from hermes_cli.config import get_compatible_custom_providers, load_config

        cfg = load_config()
        user_provs = cfg.get("providers")
        custom_provs = get_compatible_custom_providers(cfg)
    except Exception:
        pass

    result = switch_model(
        raw_input=model_input,
        current_provider=current_provider,
        current_model=current_model,
        current_base_url=current_base_url,
        current_api_key=current_api_key,
        is_global=persist_global,
        explicit_provider=explicit_provider,
        user_providers=user_provs,
        custom_providers=custom_provs,
    )
    if not result.success:
        raise ValueError(result.error_message or "model switch failed")

    if agent:
        try:
            from hermes_cli.context_switch_guard import merge_preflight_compression_warning

            _cfg_ctx = None
            if isinstance(cfg, dict):
                _mc = cfg.get("model", {})
                if isinstance(_mc, dict) and _mc.get("context_length") is not None:
                    _cfg_ctx = int(_mc["context_length"])
            merge_preflight_compression_warning(
                result,
                agent=agent,
                messages=list(session.get("history", [])),
                custom_providers=custom_provs,
                config_context_length=_cfg_ctx,
            )
        except Exception as exc:
            logger.debug("preflight-compression switch warning failed: %s", exc)

    if not confirm_expensive_model:
        try:
            from hermes_cli.model_cost_guard import expensive_model_warning

            warning = expensive_model_warning(
                result.new_model,
                provider=result.target_provider,
                base_url=result.base_url or current_base_url,
                api_key=result.api_key or current_api_key,
                model_info=result.model_info,
            )
        except Exception:
            warning = None
        if warning is not None:
            confirm_msg = warning.message
            if result.warning_message:
                confirm_msg = f"{confirm_msg}\n\n{result.warning_message}"
            return {
                "value": result.new_model,
                "warning": confirm_msg,
                "confirm_required": True,
                "confirm_message": confirm_msg,
            }

    if agent:
        try:
            agent.switch_model(
                new_model=result.new_model,
                new_provider=result.target_provider,
                api_key=result.api_key,
                base_url=result.base_url,
                api_mode=result.api_mode,
            )
        except Exception as exc:
            # The in-place swap rolled the agent back to the old working
            # model/client and re-raised.  Abort the commit: do NOT restart the
            # slash worker, persist runtime, append the switch marker, set a
            # session model_override, or persist to config — all of which would
            # otherwise leave the session pinned to a broken model and kill the
            # conversation on the next turn (#50163).  A failed switch is a
            # no-op; surface a clean error to the client.
            logger.warning("In-place model switch failed for TUI agent: %s", exc)
            raise ValueError(
                f"Model switch to {result.new_model} failed ({exc}); "
                f"staying on {getattr(agent, 'model', current_model)}."
            ) from exc
        _restart_slash_worker(sid, session)
        _persist_live_session_runtime(session)
        _persist_live_session_system_prompt(session)
        _append_model_switch_marker(
            session, model=result.new_model, provider=result.target_provider
        )
        _emit("session.info", sid, _session_info(agent, session))

    # Record the switch as a PER-SESSION override so a later rebuild of THIS
    # session (e.g. /new via _reset_session_agent, or resume) re-derives the
    # user's chosen model/provider instead of falling back to global config.
    #
    # We deliberately do NOT write process-global env vars (HERMES_MODEL /
    # HERMES_INFERENCE_MODEL / HERMES_TUI_PROVIDER / HERMES_INFERENCE_PROVIDER)
    # here. The desktop backend hosts every same-profile session in ONE process,
    # so mutating os.environ on a /model switch leaked the new model/provider
    # into every OTHER live session's next agent rebuild — switching the model
    # in one session silently changed it in the others (the cross-session
    # contamination bug). agent.switch_model() above already mutated the right
    # agent in place; the override dict makes that choice survive a rebuild
    # without touching shared process state.
    if pin_session_override and isinstance(session, dict):
        session["model_override"] = {
            "model": result.new_model,
            "provider": result.target_provider,
            "base_url": result.base_url,
            "api_key": result.api_key,
            "api_mode": result.api_mode,
        }
    if persist_global:
        _persist_model_switch(result)
    return {
        "value": result.new_model,
        "warning": result.warning_message or "",
        "confirm_required": False,
    }


def _sync_agent_model_with_config(sid: str, session: dict) -> None:
    """Adopt a config.yaml model change at turn start, like gateways do per
    message. Sessions pinned with /model keep their choice; a failed switch
    keeps the current model and never blocks the turn.
    """
    agent = session.get("agent")
    if agent is None or session.get("model_override"):
        return
    target = _config_model_target()
    if not target[0]:
        return
    seen = session.get("config_model_seen")
    # Record first so a broken config gets one attempt per edit, not per turn.
    session["config_model_seen"] = target
    if target == seen:
        return
    model, provider = target
    # Already running the configured model (branched/resumed session before
    # its first sync, or a config revert after a failed switch): adopt the
    # baseline without a redundant switch.
    if model == getattr(agent, "model", "") and (
        not provider or provider == getattr(agent, "provider", "")
    ):
        return
    raw = f"{model} --provider {provider}" if provider else model
    try:
        _apply_model_switch(
            sid,
            session,
            raw,
            confirm_expensive_model=True,
            pin_session_override=False,
            # This sync ADOPTS a config.yaml change into the live session; it
            # must never write config back. Without this, the flag/config
            # default (persist_switch_by_default=True) re-persisted whatever
            # target the sync computed — the path that leaked `hermes --tui -m`
            # into config.yaml as the permanent global model.
            persist_override=False,
        )
    except Exception as e:
        _emit(
            "error",
            sid,
            {"message": f"Could not switch to configured model {model}: {e}"},
        )


def _compress_session_history(
    session: dict,
    focus_topic: str | None = None,
    approx_tokens: int | None = None,
    before_messages: list | None = None,
    history_version: int | None = None,
) -> tuple[int, dict]:
    from agent.model_metadata import estimate_request_tokens_rough

    agent = session["agent"]
    # Snapshot history under the lock so the LLM-bound compression call
    # below does NOT hold history_lock for the duration of the request —
    # otherwise other handlers acquiring the lock (prompt.submit etc.)
    # block on the dispatcher loop while compaction runs.
    if before_messages is None or history_version is None:
        with session["history_lock"]:
            before_messages = list(session.get("history", []))
            history_version = int(session.get("history_version", 0))
    history = before_messages
    if len(history) < 4:
        usage = _get_usage(agent)
        return 0, usage
    if approx_tokens is None:
        # Include system prompt + tool schemas so the figure reflects real
        # request pressure, not a transcript-only underestimate (#6217).
        _sys_prompt = getattr(agent, "_cached_system_prompt", "") or ""
        _tools = getattr(agent, "tools", None) or None
        approx_tokens = estimate_request_tokens_rough(
            history, system_prompt=_sys_prompt, tools=_tools
        )
    # Pass system_message=None so AIAgent._compress_context rebuilds the
    # system prompt cleanly via _build_system_prompt(None). Passing the
    # cached prompt (which already contains the agent identity block)
    # makes the rebuild append the identity a second time. Mirrors the
    # CLI's _manual_compress fix for issue #15281.
    compressed, _ = agent._compress_context(
        history,
        None,
        approx_tokens=approx_tokens,
        focus_topic=focus_topic or None,
    )
    with session["history_lock"]:
        if int(session.get("history_version", 0)) != history_version:
            # External mutation during compaction — drop the compressed
            # result so we don't clobber concurrent edits.
            usage = _get_usage(agent)
            return 0, usage
        session["history"] = compressed
        session["history_version"] = history_version + 1
    usage = _get_usage(agent)
    return len(history) - len(compressed), usage


def _sync_session_key_after_compress(
    sid: str,
    session: dict,
    *,
    clear_pending_title: bool = True,
    restart_slash_worker: bool = True,
) -> None:
    """Re-anchor session_key when AIAgent._compress_context rotates session_id.

    AIAgent._compress_context ends the current SessionDB session and creates
    a new continuation session, rotating ``agent.session_id``.  The TUI
    gateway keeps the gateway-side ``session_key`` separate (used for
    approval routing, slash worker init, DB title/history lookups, yolo
    state).  Without this sync, those operations would target the ended
    parent session while the agent writes to the new continuation session.

    Policy flags:
        clear_pending_title: True for manual /compress (title belongs to old
            session). False for post-turn auto-compression (preserve user
            intent so pending_title can be applied to the continuation).
        restart_slash_worker: True for manual /compress and post-turn
            auto-compression (worker holds stale session key). False only
            if the caller manages the worker lifecycle separately.
    """
    agent = session.get("agent")
    new_session_id = getattr(agent, "session_id", None) or ""
    old_key = session.get("session_key", "") or ""
    if not new_session_id or new_session_id == old_key:
        return

    lease_reanchored = _transfer_active_session_slot(
        sid,
        session,
        new_session_id=new_session_id,
    )
    if not lease_reanchored:
        logger.warning(
            "Compression session lease did not re-anchor: sid=%s old_session_id=%s new_session_id=%s",
            sid,
            old_key,
            new_session_id,
        )

    try:
        from tools.approval import (
            disable_session_yolo,
            enable_session_yolo,
            is_session_yolo_enabled,
            register_gateway_notify,
            unregister_gateway_notify,
        )

        try:
            unregister_gateway_notify(old_key)
        except Exception:
            pass
        session["session_key"] = new_session_id
        try:
            yolo_was_on = is_session_yolo_enabled(old_key)
        except Exception:
            yolo_was_on = False
        if yolo_was_on:
            try:
                enable_session_yolo(new_session_id)
                disable_session_yolo(old_key)
            except Exception:
                pass
        try:
            register_gateway_notify(
                new_session_id,
                lambda data: _emit_approval_request(sid, data),
            )
        except Exception:
            pass
    except Exception:
        # Even if the approval module fails to import, still anchor the
        # session_key on the new continuation id so downstream lookups
        # don't keep targeting the ended row.
        session["session_key"] = new_session_id

    if clear_pending_title:
        session["pending_title"] = None
    if restart_slash_worker:
        try:
            _restart_slash_worker(sid, session)
        except Exception:
            pass


def _get_usage(agent) -> dict:
    g = lambda k, fb=None: getattr(agent, k, 0) or (getattr(agent, fb, 0) if fb else 0)
    usage = {
        "model": getattr(agent, "model", "") or "",
        "input": g("session_input_tokens", "session_prompt_tokens"),
        "output": g("session_output_tokens", "session_completion_tokens"),
        "reasoning": g("session_reasoning_tokens"),
        "prompt": g("session_prompt_tokens"),
        "completion": g("session_completion_tokens"),
        "total": g("session_total_tokens"),
        "calls": g("session_api_calls"),
    }
    comp = getattr(agent, "context_compressor", None)
    if comp:
        # context_used is the *current-window* occupancy. Do NOT fall back to
        # usage["total"] (cumulative lifetime session_total_tokens): for an
        # external context engine that doesn't report last_prompt_tokens that
        # substitution showed lifetime totals as the live context fill, yielding
        # impossible readings such as 1.9m/120k clamped to 100% (#50421).
        #
        # Per the issue, populate context_used/percent only from a *real*
        # current-occupancy value and "leave it unknown otherwise" — so a falsy
        # last_prompt_tokens (0 or missing, i.e. an engine that doesn't track
        # per-window occupancy) intentionally emits no gauge rather than a
        # fabricated 0% or the old cumulative reading. The built-in compressor
        # always reports a real last_prompt_tokens once a turn runs, so it is
        # unaffected.
        # Clamp the -1 "compression just ran, awaiting real usage" sentinel
        # (conversation_compression.py) to 0 so the transitional turn reads as
        # unknown (no gauge) instead of leaking context_used=-1. Matches the
        # CLI status-bar path (cli.py _get_status_bar_snapshot).
        last_prompt = getattr(comp, "last_prompt_tokens", 0) or 0
        if last_prompt < 0:
            last_prompt = 0
        ctx_max = getattr(comp, "context_length", 0) or 0
        if ctx_max and last_prompt:
            usage["context_used"] = last_prompt
            usage["context_max"] = ctx_max
            usage["context_percent"] = max(0, min(100, round(last_prompt / ctx_max * 100)))
        usage["compressions"] = getattr(comp, "compression_count", 0) or 0
    # Live count of background/async subagents still running (delegate_task
    # batches + background single delegations). Mirrors the classic CLI status
    # bar's ⛓ indicator; sourced from the same async_delegation registry.
    try:
        from tools.async_delegation import active_count as _async_active_count
        usage["active_subagents"] = _async_active_count()
    except Exception:
        pass
    # Dev-only live credits-spent readout (L0 usage-aware-credits). Gated on
    # HERMES_DEV_CREDITS so the payload stays clean when the flag is off.
    if is_truthy_value(os.environ.get("HERMES_DEV_CREDITS")):
        try:
            spent = agent.get_credits_spent_micros()
            if spent is not None:
                usage["dev_credits_spent_micros"] = int(spent)
        except Exception:
            pass
    return usage


def _probe_credentials(agent) -> str:
    """Light credential check at session creation — returns warning or ''."""
    try:
        key = getattr(agent, "api_key", "") or ""
        provider = getattr(agent, "provider", "") or ""
        if not key or key == "no-key-required":
            return f"No API key configured for provider '{provider}'. First message will fail."
    except Exception:
        pass
    return ""


def _probe_config_health(cfg: dict) -> str:
    """Flag bare YAML keys (`agent:` with no value → None) that silently
    drop nested settings. Returns warning or ''."""
    if not isinstance(cfg, dict):
        return ""
    warnings: list[str] = []
    null_keys = sorted(k for k, v in cfg.items() if v is None)
    if not null_keys:
        pass
    else:
        keys = ", ".join(f"`{k}`" for k in null_keys)
        warnings.append(
            f"config.yaml has empty section(s): {keys}. "
            f"Remove the line(s) or set them to `{{}}` — "
            f"empty sections silently drop nested settings."
        )
    display_cfg = cfg.get("display")
    agent_cfg = cfg.get("agent")
    if isinstance(display_cfg, dict):
        personality = str(display_cfg.get("personality", "") or "").strip().lower()
        if (
            personality
            and personality not in {"default", "none", "neutral"}
            and isinstance(agent_cfg, dict)
            and agent_cfg.get("personalities") is None
        ):
            warnings.append(
                "`display.personality` is set but `agent.personalities` is empty/null; "
                "personality overlay will be skipped."
            )
    return " ".join(warnings).strip()


def _current_profile_name() -> str:
    try:
        from hermes_cli.profiles import get_active_profile_name

        return get_active_profile_name() or "default"
    except Exception:
        return "default"


# Monotonic GUI<->backend contract version. The desktop app refuses to drive a
# backend reporting less than its required value (or none at all — a pre-GUI
# checkout), surfacing a one-click "update to align" prompt instead of failing
# cryptically downstream. Bump whenever the desktop's backend contract changes.
# v2: adds the file.attach RPC (remote-gateway non-image file upload).
DESKTOP_BACKEND_CONTRACT = 2


def _session_info(agent, session: dict | None = None) -> dict:
    if session is None:
        for candidate in _sessions.values():
            if candidate.get("agent") is agent:
                session = candidate
                break
    cwd = _display_session_cwd(session)
    session_key = str(
        (session or {}).get("session_key") or getattr(agent, "session_id", "") or ""
    )
    cfg_personality = ((_load_cfg().get("display") or {}).get("personality") or "")
    personality = (session or {}).get("personality", cfg_personality)
    reasoning_config = getattr(agent, "reasoning_config", None)
    reasoning_effort = ""
    if isinstance(reasoning_config, dict):
        if reasoning_config.get("enabled") is False:
            # Disabled must be distinguishable from unset ("" = provider
            # default). Reporting "" here made the desktop adopt the empty
            # value after the first turn, wiping its sticky "thinking off"
            # pick and re-creating every later chat at the default effort.
            reasoning_effort = "none"
        else:
            reasoning_effort = str(reasoning_config.get("effort", "") or "")
    service_tier = getattr(agent, "service_tier", None) or ""
    # Effective approval-bypass state — the same three sources that
    # check_all_command_guards() ORs together: persistent config
    # (approvals.mode=off), the process-scoped --yolo env, and the
    # per-session flag. Reporting only the per-session flag here would lie to
    # the desktop status bar (it would show YOLO "off" while approvals.mode=off
    # silently auto-approves every dangerous command).
    yolo = False
    try:
        from tools.approval import (
            _YOLO_MODE_FROZEN,
            _get_approval_mode,
            is_session_yolo_enabled,
        )

        session_yolo = (
            bool(is_session_yolo_enabled(session_key)) if session_key else False
        )
        yolo = bool(_YOLO_MODE_FROZEN) or session_yolo or _get_approval_mode() == "off"
    except Exception:
        yolo = False
    info: dict = {
        "model": getattr(agent, "model", ""),
        "provider": getattr(agent, "provider", ""),
        "reasoning_effort": reasoning_effort,
        "service_tier": service_tier,
        "fast": service_tier == "priority",
        "yolo": yolo,
        "tools": {},
        "skills": {},
        "cwd": cwd,
        "branch": _git_branch_for_cwd(cwd),
        "personality": str(personality or ""),
        "running": bool((session or {}).get("running")),
        "title": _session_live_title(session or {}, session_key) if session_key else "",
        "desktop_contract": DESKTOP_BACKEND_CONTRACT,
        "version": "",
        "release_date": "",
        "update_behind": None,
        "update_command": "",
        "usage": _get_usage(agent),
        "profile_name": _current_profile_name(),
    }
    try:
        from hermes_cli.config import (
            detect_install_method,
            format_unsupported_install_warning,
            is_unsupported_install_method,
        )

        _install_method = detect_install_method()
        if is_unsupported_install_method(_install_method):
            info["install_warning"] = format_unsupported_install_warning(_install_method)
    except Exception:
        pass
    try:
        from hermes_cli import __version__, __release_date__

        info["version"] = __version__
        info["release_date"] = __release_date__
    except Exception:
        pass
    try:
        from model_tools import get_toolset_for_tool

        for t in getattr(agent, "tools", []) or []:
            name = t["function"]["name"]
            info["tools"].setdefault(get_toolset_for_tool(name) or "other", []).append(
                name
            )
    except Exception:
        pass
    try:
        from hermes_cli.banner import get_available_skills

        info["skills"] = get_available_skills()
    except Exception:
        pass
    try:
        from tools.mcp_tool import get_mcp_status

        info["mcp_servers"] = get_mcp_status()
    except Exception:
        info["mcp_servers"] = []
    try:
        info["system_prompt"] = getattr(agent, "_cached_system_prompt", "") or ""
    except Exception:
        pass
    try:
        from hermes_cli.banner import get_update_result
        from hermes_cli.config import recommended_update_command

        info["update_behind"] = get_update_result(timeout=0.5)
        info["update_command"] = recommended_update_command()
    except Exception:
        pass
    warn = _probe_credentials(agent)
    if warn:
        info["credential_warning"] = warn
    return info


def _tool_ctx(name: str, args: dict) -> str:
    try:
        from agent.display import build_tool_label

        return build_tool_label(name, args, max_len=80) or ""
    except Exception:
        return ""


def _emit_session_info_for_session(sid: str, session: dict) -> None:
    agent = session.get("agent")
    if agent is None:
        return
    try:
        _emit("session.info", sid, _session_info(agent, session))
    except Exception:
        pass


# Tool Args/Result text shipped to the TUI for the verbose trail line. The TUI
# renders only a small persisted preview (ui-tui VERBOSE_TRAIL_MAX_CHARS), kept
# all session and expanded by default — so shipping more than that is pure pipe
# waste AND feeds the Ink render-tree blowup that silently OOM-killed the TUI
# parent (#34095). Cap here to match the render budget (a hair more, so the
# "[omitted …]" label is still informative when output is genuinely large).
# Full output stays in the agent context and the SQLite session, untouched.
_TUI_VERBOSE_TEXT_MAX_CHARS = 1_000
_TUI_VERBOSE_TEXT_MAX_LINES = 16


def _cap_tui_verbose_text(text: str) -> str:
    if (
        len(text) <= _TUI_VERBOSE_TEXT_MAX_CHARS
        and text.count("\n") < _TUI_VERBOSE_TEXT_MAX_LINES
    ):
        return text

    idx = len(text)
    start = 0
    for _ in range(_TUI_VERBOSE_TEXT_MAX_LINES):
        idx = text.rfind("\n", 0, idx)
        if idx < 0:
            start = 0
            break
        start = idx + 1

    line_start = start
    start = max(line_start, len(text) - _TUI_VERBOSE_TEXT_MAX_CHARS)
    if start > line_start:
        next_break = text.find("\n", start)
        if 0 <= next_break < len(text) - 1:
            start = next_break + 1

    tail = text[start:].lstrip()
    omitted_chars = max(0, len(text) - len(tail))
    omitted_lines = text[:start].count("\n")
    if omitted_lines:
        label = (
            "[showing verbose tail; omitted "
            f"{omitted_lines} lines / {omitted_chars} chars]\n"
        )
    else:
        label = f"[showing verbose tail; omitted {omitted_chars} chars]\n"
    return f"{label}{tail}"


def _redact_tui_verbose_text(text: str) -> str:
    try:
        from agent.redact import redact_sensitive_text

        redacted = redact_sensitive_text(str(text), force=True)
    except Exception:
        return ""
    return _cap_tui_verbose_text(redacted)


def _tool_args_text(args: dict) -> str:
    try:
        raw = json.dumps(args or {}, indent=2, ensure_ascii=False, default=str)
    except Exception:
        raw = str(args or {})
    return _redact_tui_verbose_text(raw)


def _tool_result_text(result: object) -> str:
    try:
        from agent.tool_dispatch_helpers import _multimodal_text_summary

        raw = _multimodal_text_summary(result)
    except Exception:
        raw = str(result)
    return _redact_tui_verbose_text(raw)


def _fmt_tool_duration(seconds: float | None) -> str:
    if seconds is None:
        return ""
    if seconds < 10:
        return f"{seconds:.1f}s"
    if seconds < 60:
        return f"{round(seconds)}s"
    mins, secs = divmod(int(round(seconds)), 60)
    return f"{mins}m {secs}s" if secs else f"{mins}m"


def _count_list(obj: object, *path: str) -> int | None:
    cur = obj
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return len(cur) if isinstance(cur, list) else None


def _tool_summary(name: str, result: str, duration_s: float | None) -> str | None:
    try:
        data = json.loads(result)
    except Exception:
        data = None

    dur = _fmt_tool_duration(duration_s)
    suffix = f" in {dur}" if dur else ""
    text = None

    if name == "web_search" and isinstance(data, dict):
        n = _count_list(data, "data", "web")
        if n is not None:
            text = f"Did {n} {'search' if n == 1 else 'searches'}"

    elif name == "web_extract" and isinstance(data, dict):
        n = _count_list(data, "results") or _count_list(data, "data", "results")
        if n is not None:
            text = f"Extracted {n} {'page' if n == 1 else 'pages'}"

    if isinstance(data, dict) and data.get("fallback_warning"):
        warning = str(data.get("fallback_warning") or "").strip()
        if warning:
            return f"{warning}{suffix}"

    return f"{text}{suffix}" if text else None


def _on_tool_start(sid: str, tool_call_id: str, name: str, args: dict):
    session = _sessions.get(sid)
    if session is not None:
        try:
            from agent.display import capture_local_edit_snapshot

            snapshot = capture_local_edit_snapshot(name, args)
            if snapshot is not None:
                session.setdefault("edit_snapshots", {})[tool_call_id] = snapshot
        except Exception:
            pass
        session.setdefault("tool_started_at", {})[tool_call_id] = time.time()
    if _tool_progress_enabled(sid):
        payload = {
            "tool_id": tool_call_id,
            "name": name,
            "context": _tool_ctx(name, args),
        }
        if _session_verbose(sid):
            args_text = _tool_args_text(args)
            if args_text:
                payload["args_text"] = args_text
        # tool.complete is the source of truth for todos (full list from the
        # tool result). args.todos here may be a partial merge update.
        _emit("tool.start", sid, payload)


def _on_tool_complete(sid: str, tool_call_id: str, name: str, args: dict, result: str):
    payload = {"tool_id": tool_call_id, "name": name, "args": args}
    session = _sessions.get(sid)
    snapshot = None
    started_at = None
    if session is not None:
        snapshot = session.setdefault("edit_snapshots", {}).pop(tool_call_id, None)
        started_at = session.setdefault("tool_started_at", {}).pop(tool_call_id, None)
    duration_s = time.time() - started_at if started_at else None
    if duration_s is not None:
        payload["duration_s"] = duration_s
    try:
        payload["result"] = json.loads(result)
    except Exception:
        payload["result"] = result
    summary = _tool_summary(name, result, duration_s)
    if summary:
        payload["summary"] = summary
    if _session_verbose(sid):
        result_text = _tool_result_text(result)
        if result_text:
            payload["result_text"] = result_text
    if name == "todo":
        try:
            data = json.loads(result)
            if isinstance(data, dict) and isinstance(data.get("todos"), list):
                payload["todos"] = data.get("todos")
        except Exception:
            pass
    try:
        from agent.display import render_edit_diff_with_delta

        rendered: list[str] = []
        if render_edit_diff_with_delta(
            name,
            result,
            function_args=args,
            snapshot=snapshot,
            print_fn=rendered.append,
        ):
            payload["inline_diff"] = "\n".join(rendered)
    except Exception:
        pass
    if _tool_progress_enabled(sid) or payload.get("inline_diff"):
        _emit("tool.complete", sid, payload)


def _on_tool_progress(
    sid: str,
    event_type: str,
    name: str | None = None,
    preview: str | None = None,
    _args: dict | None = None,
    **_kwargs,
):
    if not _tool_progress_enabled(sid):
        return
    if event_type == "tool.started" and name:
        # `_on_tool_start` already emits the authoritative `tool.start` with
        # the stable tool id and args. Emitting another id-less progress row
        # here makes the desktop live view diverge from hydrated history.
        return
    if event_type == "tool.output_risk" and name:
        metadata = _kwargs.get("risk_metadata")
        if not isinstance(metadata, dict):
            return
        payload: dict[str, object] = {
            "tool_id": str(_kwargs.get("tool_call_id") or ""),
            "name": str(name),
            "risk": str(metadata.get("risk") or "low"),
            "findings": [str(item) for item in metadata.get("findings", [])],
            "redacted": bool(metadata.get("redacted", False)),
        }
        _emit("tool.output_risk", sid, payload)
        return
    if event_type == "reasoning.available" and preview:
        payload: dict[str, object] = {"text": str(preview)}
        if _session_verbose(sid):
            payload["verbose"] = True
        _emit("reasoning.available", sid, payload)
        return
    if event_type == "moa.reference" and name:
        # MoA reference-model output — relay as a labelled block the Ink/desktop
        # client renders before the aggregator's response (like a thinking
        # block, tagged with the source model). `name` is the slot label,
        # `preview` is the reference text.
        ref_payload: dict[str, object] = {
            "label": str(name),
            "text": str(preview or ""),
        }
        if _kwargs.get("moa_index") is not None:
            ref_payload["index"] = _kwargs.get("moa_index")
        if _kwargs.get("moa_count") is not None:
            ref_payload["count"] = _kwargs.get("moa_count")
        _emit("moa.reference", sid, ref_payload)
        return
    if event_type == "moa.aggregating":
        _emit("moa.aggregating", sid, {"aggregator": str(name or "")})
        return
    if event_type.startswith("subagent."):
        payload = {
            "goal": str(_kwargs.get("goal") or ""),
            "task_count": int(_kwargs.get("task_count") or 1),
            "task_index": int(_kwargs.get("task_index") or 0),
        }
        # Identity fields for the TUI spawn tree.  All optional — older
        # emitters that omit them fall back to flat rendering client-side.
        if _kwargs.get("subagent_id"):
            payload["subagent_id"] = str(_kwargs["subagent_id"])
        if _kwargs.get("parent_id"):
            payload["parent_id"] = str(_kwargs["parent_id"])
        if _kwargs.get("child_session_id"):
            payload["child_session_id"] = str(_kwargs["child_session_id"])
        if _kwargs.get("depth") is not None:
            payload["depth"] = int(_kwargs["depth"])
        if _kwargs.get("model"):
            payload["model"] = str(_kwargs["model"])
        if _kwargs.get("tool_count") is not None:
            payload["tool_count"] = int(_kwargs["tool_count"])
        if _kwargs.get("toolsets"):
            payload["toolsets"] = [str(t) for t in _kwargs["toolsets"]]
        # Per-branch rollups emitted on subagent.complete (features 1+2+4).
        for int_key in (
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "api_calls",
        ):
            val = _kwargs.get(int_key)
            if val is not None:
                try:
                    payload[int_key] = int(val)
                except (TypeError, ValueError):
                    pass
        if _kwargs.get("files_read"):
            payload["files_read"] = [str(p) for p in _kwargs["files_read"]]
        if _kwargs.get("files_written"):
            payload["files_written"] = [str(p) for p in _kwargs["files_written"]]
        if _kwargs.get("output_tail"):
            payload["output_tail"] = list(_kwargs["output_tail"])  # list of dicts
        if name:
            payload["tool_name"] = str(name)
        if preview:
            payload["text"] = str(preview)
        if _kwargs.get("status"):
            payload["status"] = str(_kwargs["status"])
        if _kwargs.get("summary"):
            payload["summary"] = str(_kwargs["summary"])
        if _kwargs.get("duration_seconds") is not None:
            payload["duration_seconds"] = float(_kwargs["duration_seconds"])
        if preview and event_type == "subagent.tool":
            payload["tool_preview"] = str(preview)
            payload["text"] = str(preview)
        # subagent.text is the child's per-token reply, relayed solely to feed a
        # watch window's live mirror. It is meaningless on the parent session
        # (which shows the child via the spawn tree, not its reply body), so
        # skip the parent emit — sending hundreds of ignored token frames there
        # is wasted traffic and a trap for any future parent-side subagent
        # catch-all. The mirror keys off the child sid and is unaffected.
        if event_type != "subagent.text":
            _emit(event_type, sid, payload)
        _mirror_subagent_to_child(event_type, payload)


# ── Child-session live mirror ────────────────────────────────────────
# A delegated child is not a live gateway session — it runs synchronously
# inside the parent's turn, and its activity reaches the gateway only as
# relayed ``subagent.*`` events on the PARENT sid. When a UI opens the child's
# own session (session.resume on ``child_session_id``, e.g. the desktop's
# open-in-new-window), that window would otherwise sit silent until the run
# persists. Translate the relayed events into the native stream events the
# window already renders — emitted on the CHILD sid, routed to its transport
# by write_json — so the window shows a real midstream turn.
_child_mirrors: dict[str, dict] = {}
_child_mirrors_lock = threading.Lock()
# Stored child session ids with a delegation run currently in flight (refreshed
# on every relayed subagent.* event, popped on subagent.complete). Lets a lazy
# watch resume report running=true so the window shows a busy indicator even
# while the child is silent inside a long tool call (no events for 25s+).
_active_child_runs: dict[str, float] = {}
# Staleness bound for the registry: entries refresh on every relayed event, so
# anything this quiet means the completion event was lost (callback raised,
# parent crashed) — don't let a leaked entry pin "running" forever.
_CHILD_RUN_STALE_S = 3600.0


def _child_run_active(child_key: str) -> bool:
    ts = _active_child_runs.get(child_key)
    return ts is not None and (time.time() - ts) < _CHILD_RUN_STALE_S


def _mirror_subagent_to_child(event_type: str, payload: dict) -> None:
    child_key = str(payload.get("child_session_id") or "")
    if not child_key:
        return
    # Liveness registry first — it must be accurate even when no window is
    # open, so a window opened mid-run can immediately know the child is busy.
    if event_type == "subagent.complete":
        _active_child_runs.pop(child_key, None)
    else:
        _active_child_runs[child_key] = time.time()
    # Mirror only into a live watch session (keyed by session_key; its live sid
    # differs from the stored id) that has NOT been upgraded to a full agent.
    # No window / closed → nothing to mirror; an upgraded session owns a real
    # native stream and mirroring on top would interleave two turns on one sid.
    # Either way drop state so a reopened window starts a fresh synthetic turn.
    live = _find_live_session_by_key(child_key)
    if live is None or live[1].get("agent") is not None:
        with _child_mirrors_lock:
            _child_mirrors.pop(child_key, None)
        return
    csid = live[0]
    with _child_mirrors_lock:
        st = _child_mirrors.setdefault(child_key, {"seq": 0, "open_tool": None, "started": False})
        if not st["started"]:
            st["started"] = True
            _emit("message.start", csid)
        if event_type == "subagent.thinking":
            if text := str(payload.get("text") or ""):
                _emit("reasoning.delta", csid, {"text": text})
        elif event_type == "subagent.text":
            # The child's streamed reply text — the actual "agent talking".
            # Relayed token-by-token from the child's run_conversation
            # stream_callback, so the watch window streams the reply live.
            if text := str(payload.get("text") or ""):
                _emit("message.delta", csid, {"text": text})
        elif event_type == "subagent.start":
            # One-time header line (the child's goal) so a freshly opened window
            # shows immediate context before the first reply token streams.
            if text := str(payload.get("text") or ""):
                _emit("message.delta", csid, {"text": f"{text}\n"})
        elif event_type == "subagent.tool":
            if st["open_tool"]:
                _emit("tool.complete", csid, st["open_tool"])
            st["seq"] += 1
            tool = {
                "name": str(payload.get("tool_name") or "tool"),
                "tool_id": f"submirror:{child_key}:{st['seq']}",
                "args": {},
            }
            if preview := str(payload.get("tool_preview") or payload.get("text") or ""):
                tool["preview"] = preview
            st["open_tool"] = tool
            _emit("tool.start", csid, tool)
        elif event_type == "subagent.complete":
            if st["open_tool"]:
                _emit("tool.complete", csid, st["open_tool"])
            summary = str(payload.get("summary") or payload.get("text") or "")
            _emit("message.complete", csid, {"text": summary})
            _child_mirrors.pop(child_key, None)


def _agent_cbs(sid: str) -> dict:
    return {
        "tool_start_callback": lambda tc_id, name, args: _on_tool_start(
            sid, tc_id, name, args
        ),
        "tool_complete_callback": lambda tc_id, name, args, result: _on_tool_complete(
            sid, tc_id, name, args, result
        ),
        "tool_progress_callback": lambda event_type, name=None, preview=None, args=None, **kwargs: (
            _on_tool_progress(sid, event_type, name, preview, args, **kwargs)
        ),
        "tool_gen_callback": lambda name: (
            _tool_progress_enabled(sid)
            and _emit("tool.generating", sid, {"name": name})
        ),
        "thinking_callback": lambda text: _emit("thinking.delta", sid, {"text": text}),
        "reasoning_callback": lambda text: _emit(
            "reasoning.delta",
            sid,
            {"text": text, **({"verbose": True} if _session_verbose(sid) else {})},
        ),
        "status_callback": lambda kind, text=None: _status_update(
            sid, str(kind), None if text is None else str(text)
        ),
        # Credits/notice spine (L1): an AgentNotice fired by the agent becomes a
        # notification.show WS event; a recovery clear becomes notification.clear.
        # Snake_case payload to match the existing gateway-event convention.
        "notice_callback": lambda n: _emit(
            "notification.show",
            sid,
            {
                "text": n.text,
                "level": n.level,
                "kind": n.kind,
                "ttl_ms": n.ttl_ms,
                "key": n.key,
                "id": n.id,
            },
        ),
        "notice_clear_callback": lambda key: _emit(
            "notification.clear", sid, {"key": key}
        ),
        "clarify_callback": lambda q, c: _block(
            "clarify.request", sid, {"question": q, "choices": c}
        ),
        # read_terminal tool (desktop GUI): same blocking bridge as clarify — the
        # renderer answers terminal.read.respond with the serialized buffer.
        "read_terminal_callback": lambda start=None, count=None: _block(
            "terminal.read.request",
            sid,
            {k: v for k, v in (("start", start), ("count", count)) if v is not None},
            timeout=30,
        ),
    }


def _apply_project_workspace(task_id: str, path: str, _name: str = "") -> None:
    """Intentional workspace move from the project_* tools: re-anchor the live
    session's cwd to the chosen project's folder and push session.info so the
    desktop follows (refresh tree + scope into the project). This is the ONLY
    auto-cwd path — driven by an explicit tool call, never a terminal `cd`."""
    if not path:
        return

    # The tool's task_id is the durable session_key, but _sessions is keyed by a
    # short sid uuid (and the desktop routes events by that sid). Resolve it.
    key = str(task_id or "")
    sid = ""
    session = None
    with _sessions_lock:
        if key in _sessions:
            sid, session = key, _sessions[key]
        else:
            for cand_sid, cand in _sessions.items():
                if cand.get("session_key") == key or getattr(cand.get("agent"), "session_id", None) == key:
                    sid, session = cand_sid, cand
                    break

    if session is None:
        return

    resolved = os.path.abspath(os.path.expanduser(str(path)))
    if not os.path.isdir(resolved):
        return

    session["cwd"] = resolved
    session["explicit_cwd"] = True
    _register_session_cwd(session)

    with _session_db(session) as db:
        if db is not None:
            try:
                db.update_session_cwd(session.get("session_key", ""), resolved)
            except Exception:
                logger.debug("failed to persist project workspace cwd", exc_info=True)

    _persist_session_git_meta(session, resolved)

    try:
        agent = session.get("agent")
        info = (
            _session_info(agent, session)
            if agent is not None
            else {"cwd": resolved, "branch": _git_branch_for_cwd(resolved), "lazy": True}
        )
        _emit("session.info", sid, info)
    except Exception:
        logger.debug("failed to emit session.info after project workspace move", exc_info=True)


def _wire_callbacks(sid: str):
    from tools.terminal_tool import set_sudo_password_callback
    from tools.skills_tool import set_secret_capture_callback
    from tools.project_tools import set_project_workspace_callback

    set_sudo_password_callback(lambda: _block("sudo.request", sid, {}, timeout=120))
    set_project_workspace_callback(_apply_project_workspace)

    def secret_cb(env_var, prompt, metadata=None):
        pl = {"prompt": prompt, "env_var": env_var}
        if metadata:
            pl["metadata"] = metadata
        val = _block("secret.request", sid, pl)
        if not val:
            return {
                "success": True,
                "stored_as": env_var,
                "validated": False,
                "skipped": True,
                "message": "skipped",
            }
        from hermes_cli.config import save_env_value_secure

        return {
            **save_env_value_secure(env_var, val),
            "skipped": False,
            "message": "ok",
        }

    set_secret_capture_callback(secret_cb)


def _render_personality_prompt(value) -> str:
    if isinstance(value, dict):
        parts = [value.get("system_prompt", "")]
        if value.get("tone"):
            parts.append(f"Tone: {value['tone']}")
        if value.get("style"):
            parts.append(f"Style: {value['style']}")
        return "\n".join(p for p in parts if p)
    return str(value)


def _available_personalities(cfg: dict | None = None) -> dict:
    try:
        from cli import load_cli_config

        return (load_cli_config().get("agent") or {}).get("personalities", {}) or {}
    except Exception:
        try:
            from hermes_cli.config import load_config as _load_full_cfg

            return (_load_full_cfg().get("agent") or {}).get("personalities", {}) or {}
        except Exception:
            cfg = cfg or _load_cfg()
            return (cfg.get("agent") or {}).get("personalities", {}) or {}


def _validate_personality(value: str, cfg: dict | None = None) -> tuple[str, str]:
    raw = str(value or "").strip()
    name = raw.lower()
    if not name or name in {"none", "default", "neutral"}:
        return "", ""

    personalities = _available_personalities(cfg)
    if name not in personalities:
        names = sorted(personalities)
        available = ", ".join(f"`{n}`" for n in names)
        base = f"Unknown personality: `{raw}`."
        if available:
            base += f"\n\nAvailable: `none`, {available}"
        else:
            base += "\n\nNo personalities configured."
        raise ValueError(base)

    return name, _render_personality_prompt(personalities[name])


def _prompt_text(value) -> str:
    """Normalize config prompt values from YAML before handing them to AIAgent."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def _apply_personality_to_session(
    sid: str, session: dict, new_prompt: str, personality: str = ""
) -> tuple[bool, dict | None]:
    """Apply a personality change to an existing session without resetting history.

    Updates the agent's ephemeral system prompt in-place so the new personality
    takes effect on the next turn.  The cached base system prompt is left intact
    (ephemeral_system_prompt is appended at API-call time, not baked into the
    cache), which preserves prompt-cache hits.

    Also injects a system-role marker into the conversation history so the model
    knows to pivot its style from this point forward (without this, LLMs tend to
    continue the tone established by earlier messages in the transcript).

    Returns (history_reset, info) — history_reset is always False since we
    preserve the conversation.
    """
    if not session:
        return False, None
    session["personality"] = personality

    agent = session.get("agent")
    if agent:
        agent.ephemeral_system_prompt = new_prompt or None
        # Inject a pivot marker into history so the model sees the change point.
        # This prevents it from pattern-matching its prior style.
        if new_prompt:
            marker = (
                "[System: The user has changed the assistant's personality. "
                "From this point forward, adopt the following persona and respond "
                f"accordingly: {new_prompt}]"
            )
        else:
            marker = (
                "[System: The user has cleared the personality overlay. "
                "From this point forward, respond in your normal default style.]"
            )
        with session["history_lock"]:
            session["history"].append({"role": "user", "content": marker})
            session["history_version"] = int(session.get("history_version", 0)) + 1
        info = _session_info(agent)
        _emit("session.info", sid, info)
        return False, info
    return False, None


def _cfg_max_turns(cfg: dict, default: int) -> int:
    try:
        env_max = int(os.environ.get("HERMES_TUI_MAX_TURNS", "") or 0)
        if env_max > 0:
            return env_max
    except (TypeError, ValueError):
        pass
    agent_cfg = cfg.get("agent") or {}
    return int(agent_cfg.get("max_turns") or cfg.get("max_turns") or default)


def _parse_tui_skills_env() -> list[str]:
    raw = os.environ.get("HERMES_TUI_SKILLS", "")
    skills: list[str] = []
    seen: set[str] = set()
    for part in raw.replace("\n", ",").split(","):
        item = part.strip()
        if item and item not in seen:
            seen.add(item)
            skills.append(item)
    return skills


def _load_fallback_model():
    """Return the configured fallback chain for TUI-created agents.

    Delegates to the shared ``resolve_fallback_chain`` helper so the TUI path
    stays in parity with ``HermesCLI.__init__`` and ``gateway/run.py``:
    ``fallback_providers`` is the primary source of truth and keeps its
    order, with legacy ``fallback_model`` entries merged in afterwards
    (deduped on provider/model/base_url), then NVIDIA/Nous dynamic entries
    expanded for runtime failover.
    """
    from hermes_cli.fallback_config import resolve_fallback_chain

    return resolve_fallback_chain(_load_cfg())


def _agent_fallback_model(agent):
    """Return an agent's fallback chain without rehydrating deliberately empty chains."""
    if hasattr(agent, "_fallback_chain"):
        return getattr(agent, "_fallback_chain") or []
    if hasattr(agent, "_fallback_model"):
        return getattr(agent, "_fallback_model", None)
    return _load_fallback_model()


def _background_agent_kwargs(agent, task_id: str) -> dict:
    cfg = _load_cfg()

    return {
        "base_url": getattr(agent, "base_url", None) or None,
        "api_key": getattr(agent, "api_key", None) or None,
        "provider": getattr(agent, "provider", None) or None,
        "api_mode": getattr(agent, "api_mode", None) or None,
        "acp_command": getattr(agent, "acp_command", None) or None,
        "acp_args": getattr(agent, "acp_args", None) or None,
        "model": getattr(agent, "model", None) or _resolve_model(),
        "max_iterations": _cfg_max_turns(cfg, 25),
        "enabled_toolsets": getattr(agent, "enabled_toolsets", None)
        or _load_enabled_toolsets(),
        "quiet_mode": True,
        "verbose_logging": False,
        "ephemeral_system_prompt": getattr(agent, "ephemeral_system_prompt", None)
        or None,
        "providers_allowed": getattr(agent, "providers_allowed", None),
        "providers_ignored": getattr(agent, "providers_ignored", None),
        "providers_order": getattr(agent, "providers_order", None),
        "provider_sort": getattr(agent, "provider_sort", None),
        "provider_require_parameters": getattr(
            agent, "provider_require_parameters", False
        ),
        "provider_data_collection": getattr(agent, "provider_data_collection", None),
        "openrouter_min_coding_score": getattr(
            agent, "openrouter_min_coding_score", None
        ),
        "session_id": task_id,
        "reasoning_config": getattr(agent, "reasoning_config", None)
        or _load_reasoning_config(),
        "service_tier": getattr(agent, "service_tier", None) or _load_service_tier(),
        "request_overrides": dict(getattr(agent, "request_overrides", {}) or {}),
        "platform": "tui",
        "session_db": _get_db(),
        "fallback_model": _agent_fallback_model(agent),
    }


def _ephemeral_preview_agent_kwargs(agent, task_id: str) -> dict:
    kwargs = _background_agent_kwargs(agent, task_id)
    kwargs.update({
        "enabled_toolsets": ["terminal", "file"],
        "session_db": None,
        "skip_memory": True,
    })
    return kwargs


def _preview_restart_history(
    session: dict, max_messages: int = 24, max_tool_chars: int = 1200
) -> list[dict]:
    """Distill the parent session's recent history into a context the
    ephemeral preview-restart agent can actually use.

    The restart agent has no idea what app the user was building, what
    server they ran, what cwd was active, or which port belongs to which
    project. Without this, it would take the bare URL + console logs and
    guess — usually starting the wrong thing.

    We keep the last ``max_messages`` messages from the parent session so
    the restart agent sees recent user prompts, assistant replies, and
    most importantly any terminal/tool calls. Tool result payloads are
    truncated so we don't blow the context window with file dumps.
    """
    try:
        with session["history_lock"]:
            history = list(session.get("history", []) or [])
    except Exception:
        history = list(session.get("history", []) or [])

    if not history:
        return []

    # Anchor on the last user turn so we always include at least the most
    # recent request and the assistant/tool work that followed it. Then
    # extend backwards up to max_messages so we capture the prior context.
    last_user_idx = None
    for idx in range(len(history) - 1, -1, -1):
        if history[idx].get("role") == "user":
            last_user_idx = idx
            break

    start = max(0, len(history) - max_messages)
    if last_user_idx is not None:
        start = min(start, last_user_idx)

    trimmed: list[dict] = []
    for msg in history[start:]:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role not in ("user", "assistant", "tool", "system"):
            continue

        copy = {k: v for k, v in msg.items() if k != "reasoning"}
        # Truncate heavy tool outputs so a single 50KB file read doesn't
        # crowd out the rest of the context.
        if role == "tool":
            content = copy.get("content")
            if isinstance(content, str) and len(content) > max_tool_chars:
                copy["content"] = (
                    content[:max_tool_chars]
                    + f"\n... (truncated, original {len(content)} chars)"
                )
        trimmed.append(copy)

    return trimmed


def _preview_tool_result_preview(name: str, result: str) -> str:
    try:
        data = json.loads(result)
    except Exception:
        return ""

    if not isinstance(data, dict):
        return ""

    if name == "terminal":
        output = str(data.get("output") or "").strip()
        exit_code = data.get("exit_code")
        if output:
            return output[-1200:]
        if data.get("session_id"):
            return f"Background process started: {data.get('session_id')}"
        if exit_code is not None:
            return f"terminal exited with code {exit_code}"

    return str(data.get("error") or "").strip()[:1200]


def _preview_restart_callbacks(parent: str, task_id: str) -> dict:
    started_at: dict[str, float] = {}

    def progress(message: str, level: str = "info") -> None:
        text = str(message or "").strip()
        if text:
            _emit(
                "preview.restart.progress",
                parent,
                {"task_id": task_id, "level": level, "text": text},
            )

    def tool_start(tool_call_id: str, name: str, args: dict) -> None:
        started_at[tool_call_id] = time.time()
        ctx = _tool_ctx(name, args)
        progress(f"Running {name}{f': {ctx}' if ctx else ''}")

    def tool_complete(tool_call_id: str, name: str, _args: dict, result: str) -> None:
        duration_s = time.time() - started_at.get(tool_call_id, time.time())
        summary = (
            _tool_summary(name, result, duration_s)
            or f"Finished {name}{f' in {_fmt_tool_duration(duration_s)}' if duration_s else ''}"
        )
        output = _preview_tool_result_preview(name, result)
        progress(summary + (f"\n{output}" if output else ""))

    def tool_progress(
        event_type: str, name: str | None = None, preview: str | None = None, **_kwargs
    ) -> None:
        if preview:
            progress(str(preview))
        elif name:
            progress(f"{event_type.replace('.', ' ')}: {name}")

    return {
        "tool_start_callback": tool_start,
        "tool_complete_callback": tool_complete,
        "tool_progress_callback": tool_progress,
        "tool_gen_callback": lambda name: progress(f"Preparing {name}"),
        "status_callback": lambda kind, text=None: progress(
            text if text is not None else kind
        ),
    }


def _reset_session_agent(sid: str, session: dict) -> dict:
    tokens = _set_session_context(session["session_key"])
    try:
        # Preserve this session's chosen model AND reasoning across /new so a
        # reset doesn't silently revert to global config (or to a model
        # another session set). See the cross-session-contamination note in
        # _apply_model_switch.
        reset_kw = {"model_override": session.get("model_override")}
        old_reasoning = getattr(session.get("agent"), "reasoning_config", None)
        if old_reasoning is None:
            old_reasoning = session.get("create_reasoning_override")
        if isinstance(old_reasoning, dict):
            reset_kw["reasoning_config_override"] = old_reasoning
        new_agent = _make_agent(
            sid,
            session["session_key"],
            session_id=session["session_key"],
            platform_override=_session_source(session),
            **reset_kw,
        )
    finally:
        _clear_session_context(tokens)
    session["agent"] = new_agent
    session["config_model_seen"] = _config_model_target()
    session["attached_images"] = []
    session["edit_snapshots"] = {}
    session["image_counter"] = 0
    session["running"] = False
    session["show_reasoning"] = _load_show_reasoning()
    session["tool_progress_mode"] = _load_tool_progress_mode()
    session["tool_started_at"] = {}
    with session["history_lock"]:
        session["history"] = []
        session["history_version"] = int(session.get("history_version", 0)) + 1
    info = _session_info(new_agent, session)
    _emit("session.info", sid, info)
    _restart_slash_worker(sid, session)
    return info


def _schedule_mcp_late_refresh(sid: str, agent) -> None:
    """Refresh a session's tool snapshot when MCP discovery lands late.

    The agent snapshots ``agent.tools`` once at build time and never re-reads
    the registry (run_agent/agent_init). ``_make_agent`` briefly joins the
    background MCP discovery thread (``wait_for_mcp_discovery``, bounded by the
    ``mcp_discovery_timeout`` config value, default 1.5s) so
    already-spawning servers land in that snapshot — but a server that takes
    longer than the bound to connect (common for an HTTP MCP server on first
    connect) lands *after* the agent is built. Its tools are then absent from
    both the agent and the banner for the whole session, even though the
    classic CLI shows them (the CLI re-derives ``get_tool_definitions`` at
    banner render time, which re-waits, so it picks them up).

    This schedules an off-critical-path daemon that waits for discovery to
    finish, then rebuilds the snapshot and re-emits ``session.info`` so both
    the agent's callable tools and the banner count catch up — the same
    rebuild ``/reload-mcp`` performs, but automatic.

    Cache safety: the rebuild only runs while the session is still pre-first-
    turn (no API call made yet → nothing cached to invalidate). If the user
    has already sent a message, we leave the snapshot frozen rather than
    invalidate the prompt cache mid-conversation — those late tools then
    require an explicit ``/reload-mcp`` (which gates on user consent), exactly
    as today. No-op when discovery already finished before the agent build.
    """
    try:
        from tui_gateway.entry import mcp_discovery_in_flight, join_mcp_discovery
    except Exception:
        return
    if not mcp_discovery_in_flight():
        return

    def _wait_then_refresh() -> None:
        # Bounded but generous — a server still not connected after this is
        # genuinely slow/dead; the user can /reload-mcp once it recovers.
        if not join_mcp_discovery(timeout=30.0):
            return
        with _sessions_lock:
            session = _sessions.get(sid)
            # Session may have been closed/reset while we waited.
            if session is None or session.get("agent") is not agent:
                return
            # Cache safety: never rebuild the tool list once the conversation
            # has started — that would invalidate the cached prompt prefix.
            if (
                int(getattr(agent, "_user_turn_count", 0) or 0) > 0
                or int(getattr(agent, "_api_call_count", 0) or 0) > 0
            ):
                return
            try:
                from tools.mcp_tool import refresh_agent_mcp_tools

                added = refresh_agent_mcp_tools(agent, quiet_mode=True)
            except Exception as exc:
                logger.warning(
                    "Late MCP refresh: tool snapshot rebuild failed for %s: %s",
                    sid,
                    exc,
                )
                return
            # No new tools landed (discovery added nothing) → don't churn the client.
            if not added:
                return
            info = _session_info(agent, session)
        # Emit outside the lock — write_json must not block under _sessions_lock.
        _emit("session.info", sid, info)
    threading.Thread(
        target=_wait_then_refresh,
        name=f"tui-mcp-late-refresh-{sid}",
        daemon=True,
    ).start()


def _resolve_runtime_with_fallback(
    resolve_kwargs: dict | None = None,
) -> dict:
    """Resolve runtime provider with init-time fallback on auth failure.

    Mirrors the fallback pattern in ``cron/scheduler.py`` and
    ``hermes_cli/cli_agent_setup_mixin.py``: when the primary provider
    raises ``AuthError``, walk the configured ``fallback_providers`` /
    ``fallback_model`` chain before giving up.
    """
    from hermes_cli.auth import AuthError
    from hermes_cli.runtime_provider import resolve_runtime_provider

    kwargs = resolve_kwargs or {}
    try:
        return resolve_runtime_provider(**kwargs)
    except AuthError as primary_exc:
        fb_chain = _load_fallback_model() or []
        for entry in fb_chain:
            if not isinstance(entry, dict):
                continue
            fb_provider = (entry.get("provider") or "").strip()
            if not fb_provider:
                continue
            try:
                fb_kwargs: dict = {"requested": fb_provider}
                if entry.get("base_url"):
                    fb_kwargs["explicit_base_url"] = entry["base_url"]
                if entry.get("api_key"):
                    fb_kwargs["explicit_api_key"] = entry["api_key"]
                runtime = resolve_runtime_provider(**fb_kwargs)
                import logging

                logging.getLogger(__name__).warning(
                    "Primary auth failed (%s), falling back to %s",
                    primary_exc,
                    fb_provider,
                )
                return runtime
            except Exception:
                continue
        raise


def _make_agent(
    sid: str,
    key: str,
    session_id: str | None = None,
    session_db=None,
    model_override: dict | str | None = None,
    provider_override: str | None = None,
    reasoning_config_override: dict | None = None,
    service_tier_override: str | None = None,
    platform_override: str | None = None,
):
    from run_agent import AIAgent

    # MCP tool discovery runs in a background daemon thread at startup so a
    # dead server can't freeze the shell.  The agent snapshots its tool list
    # once here and never re-reads it, so briefly wait for in-flight discovery
    # to land before building — bounded, so a slow/dead server still can't
    # block. Dashboard /api/ws uses hermes_cli.mcp_startup; TUI stdio keeps
    # its existing tui_gateway.entry-owned thread.
    try:
        from hermes_cli.mcp_startup import wait_for_mcp_discovery

        wait_for_mcp_discovery()
    except Exception:
        pass
    try:
        from tui_gateway.entry import wait_for_mcp_discovery

        wait_for_mcp_discovery()
    except Exception:
        pass

    cfg = _load_cfg()
    agent_cfg = cfg.get("agent") or {}
    system_prompt = _prompt_text(agent_cfg.get("system_prompt", ""))
    startup_skills = _parse_tui_skills_env()
    if startup_skills:
        from agent.skill_commands import build_preloaded_skills_prompt

        skills_prompt, loaded_skills, missing_skills = build_preloaded_skills_prompt(
            startup_skills,
            task_id=session_id or key,
        )
        if missing_skills:
            missing_display = ", ".join(missing_skills)
            # Degrade gracefully when some skills loaded; only hard-fail when
            # every requested skill is missing. Mirrors cli.py — a typo'd skill
            # name should not crash the worker and auto-block the Kanban task.
            if loaded_skills:
                logger.warning(
                    "Unknown skill(s) requested, skipping: %s. "
                    "Continuing with: %s. "
                    "List available skills with `hermes skills list`.",
                    missing_display,
                    ", ".join(loaded_skills),
                )
            else:
                raise ValueError(f"Unknown skill(s): {missing_display}")
        if skills_prompt:
            system_prompt = "\n\n".join(
                part for part in (system_prompt, skills_prompt) if part
            ).strip()
    # Prefer a per-session model override (set by a prior in-session /model
    # switch) over global config/env resolution. Resume-time stored sessions may
    # also pass scalar model/provider/runtime knobs from the persisted DB row.
    if isinstance(model_override, dict) and model_override.get("model"):
        model = str(model_override.get("model") or "")
        requested_provider = model_override.get("provider") or provider_override or None
        override_base_url = model_override.get("base_url")
        override_api_key = model_override.get("api_key")
        override_api_mode = model_override.get("api_mode")
        resolve_kwargs = {}
        if str(requested_provider or "").strip().lower() == "custom":
            # Session rows persisted before the custom-provider identity fix
            # (see _runtime_model_config) stored the resolved provider
            # "custom", which _get_named_custom_provider cannot match back to
            # a named ``providers:`` / ``custom_providers:`` entry — the
            # rebuild then either raised auth_unavailable, silently resolved
            # placeholder credentials against the patched-back base_url, or
            # (when no base_url was stored) routed to the OpenRouter default
            # with no key, surfacing as "No LLM provider configured". Recover
            # the entry identity from the persisted base_url, falling back to
            # the configured provider when the override carries no base_url
            # (the recurring Desktop/TUI regression vector).
            from hermes_cli.runtime_provider import canonical_custom_identity

            recovered = canonical_custom_identity(base_url=override_base_url or None)
            if recovered:
                requested_provider = recovered
            if override_base_url:
                # Failing identity recovery, still hand the base_url to the
                # direct-alias branch so pool/env credentials resolve for it.
                resolve_kwargs["explicit_base_url"] = override_base_url
        resolve_kwargs["requested"] = requested_provider
        resolve_kwargs["target_model"] = model or None
        runtime = _resolve_runtime_with_fallback(resolve_kwargs)
        # The switch already resolved concrete credentials/endpoint; honor them
        # so a custom/named endpoint survives the rebuild even if global
        # resolution would pick a different one.
        if override_base_url:
            runtime["base_url"] = override_base_url
        if override_api_key:
            runtime["api_key"] = override_api_key
        if override_api_mode:
            runtime["api_mode"] = override_api_mode
    else:
        model, requested_provider = _resolve_startup_runtime()
        if isinstance(model_override, str) and model_override:
            model = model_override
        if provider_override:
            requested_provider = provider_override
        runtime = _resolve_runtime_with_fallback({
            "requested": requested_provider,
            "target_model": model or None,
        })
    _pr = _load_provider_routing()
    return AIAgent(
        model=model,
        max_iterations=_cfg_max_turns(cfg, 90),
        provider=runtime.get("provider"),
        base_url=runtime.get("base_url"),
        api_key=runtime.get("api_key"),
        api_mode=runtime.get("api_mode"),
        acp_command=runtime.get("command"),
        acp_args=runtime.get("args"),
        credential_pool=runtime.get("credential_pool"),
        quiet_mode=True,
        # verbose_logging controls DEBUG-level agent logging; it is intentionally
        # independent of tool_progress_mode (which only controls per-tool
        # display detail).  See cli.py PR (decoupling fix) for the matching
        # change on the classic CLI side.
        verbose_logging=False,
        reasoning_config=(
            reasoning_config_override
            if reasoning_config_override is not None
            else _load_reasoning_config()
        ),
        service_tier=(
            service_tier_override
            if service_tier_override is not None
            else _load_service_tier()
        ),
        enabled_toolsets=_load_enabled_toolsets(),
        # OpenRouter provider-routing prefs (config.yaml `provider_routing`).
        # Mirrors the messaging gateway + CLI so the desktop/TUI honors the same
        # routing instead of letting OpenRouter pick providers at random.
        providers_allowed=_pr.get("only"),
        providers_ignored=_pr.get("ignore"),
        providers_order=_pr.get("order"),
        provider_sort=_pr.get("sort"),
        provider_require_parameters=_pr.get("require_parameters", False),
        provider_data_collection=_pr.get("data_collection"),
        platform=_resolve_agent_platform(platform_override),
        session_id=session_id or key,
        session_db=session_db if session_db is not None else _get_db(),
        ephemeral_system_prompt=system_prompt or None,
        checkpoints_enabled=is_truthy_value(os.environ.get("HERMES_TUI_CHECKPOINTS")),
        pass_session_id=is_truthy_value(os.environ.get("HERMES_TUI_PASS_SESSION_ID")),
        skip_context_files=is_truthy_value(os.environ.get("HERMES_IGNORE_RULES")),
        skip_memory=is_truthy_value(os.environ.get("HERMES_IGNORE_RULES")),
        fallback_model=_load_fallback_model(),
        **_agent_cbs(sid),
    )


def _init_session(
    sid: str,
    key: str,
    agent,
    history: list,
    cols: int = 80,
    cwd: str | None = None,
    session_db=None,
    source: str | None = None,
):
    now = time.time()
    with _sessions_lock:
        _sessions[sid] = {
            "agent": agent,
            "session_key": key,
            "history": history,
            "history_lock": threading.Lock(),
            "history_version": 0,
            "inflight_turn": None,
            "created_at": now,
            "last_active": now,
            "running": False,
            "attached_images": [],
            "image_counter": 0,
            "cwd": cwd or _completion_cwd(),
            "cols": cols,
            "slash_worker": None,
            "show_reasoning": _load_show_reasoning(),
            "source": _resolve_session_source(source),
            "tool_progress_mode": _load_tool_progress_mode(),
            "edit_snapshots": {},
            "tool_started_at": {},
            # Per-session model override set by an in-session /model switch.
            # Honored on rebuild (/new, resume) so a switch in THIS session
            # never leaks into siblings via process-global env vars.
            "model_override": None,
            # Pin async event emissions to whichever transport created the
            # session (stdio for Ink, JSON-RPC WS for the dashboard sidebar).
            "transport": current_transport() or _stdio_transport,
        }
    db = session_db if session_db is not None else _get_db()
    if db is not None:
        row = db.get_session(key)
        if row and row.get("cwd"):
            with _sessions_lock:
                if sid in _sessions:
                    _sessions[sid]["cwd"] = row["cwd"]
        else:
            try:
                _cwd = _sessions[sid]["cwd"]
                db.update_session_cwd(key, _cwd)
                # git branch/root probes run off the hot path (see _set_session_cwd).
                _persist_session_git_meta(_sessions[sid], _cwd)
            except Exception:
                logger.debug("failed to persist resumed session cwd", exc_info=True)
    _register_session_cwd(_sessions[sid])
    try:
        _attach_worker(
            sid,
            _sessions[sid],
            _SlashWorker(
                key,
                getattr(agent, "model", _resolve_model()),
                profile_home=_sessions[sid].get("profile_home"),
            ),
        )
    except Exception:
        # Defer hard-failure to slash.exec; chat still works without slash worker.
        _sessions[sid]["slash_worker"] = None
    try:
        from tools.approval import register_gateway_notify, load_permanent_allowlist

        register_gateway_notify(key, lambda data: _emit_approval_request(sid, data))
        load_permanent_allowlist()
    except Exception:
        pass
    # Surface the self-improvement background review's "💾 …" summary as a
    # review.summary event so Ink can render it as a persistent system line
    # in the transcript. In the CLI path this message is printed via
    # prompt_toolkit; the TUI has no equivalent print surface, so without
    # this callback the review would write the skill/memory change silently.
    try:
        agent.background_review_callback = lambda message, _sid=sid: _emit(
            "review.summary", _sid, {"text": str(message)}
        )
        # Honor display.memory_notifications (off | on | verbose) like the
        # messaging gateway and CLI do — otherwise the review always behaved as
        # "on" on the TUI/desktop and a user who set "off" was ignored.
        agent.memory_notifications = _load_memory_notifications()
    except Exception:
        # Bare AIAgents that don't expose the attribute (unlikely, but keep
        # session startup resilient).
        pass
    _wire_callbacks(sid)
    with _sessions_lock:
        if sid in _sessions:
            _sessions[sid]["_notif_stop"] = _start_notification_poller(sid, _sessions[sid])
    _notify_session_boundary("on_session_reset", key, _session_source(_sessions.get(sid, {})))
    _emit("session.info", sid, _session_info(agent, _sessions.get(sid, {})))
    _schedule_mcp_late_refresh(sid, agent)


def _new_session_key() -> str:
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


def _with_checkpoints(session, fn):
    return fn(session["agent"]._checkpoint_mgr, _session_cwd(session))


def _resolve_checkpoint_hash(mgr, cwd: str, ref: str) -> str:
    try:
        checkpoints = mgr.list_checkpoints(cwd)
        idx = int(ref) - 1
    except ValueError:
        return ref
    if 0 <= idx < len(checkpoints):
        return checkpoints[idx].get("hash", ref)
    raise ValueError(f"Invalid checkpoint number. Use 1-{len(checkpoints)}.")


def _enrich_with_attached_images(user_text: str, image_paths: list[str]) -> str:
    """Pre-analyze attached images via vision and prepend descriptions to user text."""
    import asyncio, json as _json
    from tools.vision_tools import vision_analyze_tool

    prompt = (
        "Describe everything visible in this image in thorough detail. "
        "Include any text, code, data, objects, people, layout, colors, "
        "and any other notable visual information."
    )

    parts: list[str] = []
    for path in image_paths:
        p = Path(path)
        if not p.exists():
            continue
        hint = f"[You can examine it with vision_analyze using image_url: {p}]"
        try:
            r = _json.loads(
                asyncio.run(vision_analyze_tool(image_url=str(p), user_prompt=prompt))
            )
            desc = r.get("analysis", "") if r.get("success") else None
            parts.append(
                f"[The user attached an image:\n{desc}]\n{hint}"
                if desc
                else f"[The user attached an image but analysis failed.]\n{hint}"
            )
        except Exception:
            parts.append(f"[The user attached an image but analysis failed.]\n{hint}")

    text = user_text or ""
    prefix = "\n\n".join(parts)
    if prefix:
        return f"{prefix}\n\n{text}" if text else prefix
    return text or "What do you see in this image?"


def _content_display_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (int, float)):
        return str(content)
    if isinstance(content, list):
        parts = []
        for part in content:
            text = _content_display_text(part).strip()
            if text:
                parts.append(text)
        return "\n".join(parts)
    if isinstance(content, dict):
        kind = content.get("type")
        if kind in {"text", "input_text", "output_text"}:
            return str(content.get("text") or content.get("content") or "")
        if kind in {"image_url", "input_image", "image"}:
            return "[image]"
        if kind in {"input_audio", "audio"}:
            return "[audio]"
        if kind:
            return f"[{kind}]"
        if "text" in content:
            return str(content.get("text") or "")
        return "[structured content]"
    return str(content)


def _coerce_message_text(content: Any) -> str:
    """Render ``message['content']`` as a plain string for transport.

    Provider-side, ``content`` may be a string (most common), a list of
    multimodal parts (e.g. ``[{"type": "text", "text": "..."},
    {"type": "image_url", "image_url": {...}}]``), or a single structured
    dict. Calling ``.strip()`` on a list raises ``'list' object has no
    attribute 'strip'`` and breaks session resume entirely.

    Image parts (``image_url``) are preserved by appending the underlying
    URL (data: or http:) into the text. The desktop renderer pulls these
    back out via ``extractEmbeddedImages`` so the user sees the image
    instead of the URL — and it stops the resume payload from disagreeing
    with the cached message (which would otherwise cause the inline image
    to flash, then disappear when the resume payload overwrites the cache).

    Other structured dict shapes (audio, unknown types) fall back to a
    bracketed placeholder so resume doesn't drop the message entirely.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (int, float)):
        return str(content)
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, str):
                chunks.append(part)
                continue
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str):
                chunks.append(text)
                continue
            kind = part.get("type")
            if kind in {"text", "input_text", "output_text"}:
                t = part.get("text") or part.get("content") or ""
                if t:
                    chunks.append(str(t))
                continue
            if kind in {"image_url", "input_image", "image"}:
                image_url = part.get("image_url")
                url = ""
                if isinstance(image_url, dict):
                    candidate = image_url.get("url")
                    if isinstance(candidate, str):
                        url = candidate
                elif isinstance(image_url, str):
                    url = image_url
                if url:
                    chunks.append(f"\n{url}")
                else:
                    chunks.append("\n[image]")
                continue
            if kind in {"input_audio", "audio"}:
                chunks.append("\n[audio]")
                continue
            if kind:
                chunks.append(f"\n[{kind}]")
        return "".join(chunks)
    if isinstance(content, dict):
        kind = content.get("type")
        if kind in {"text", "input_text", "output_text"}:
            return str(content.get("text") or content.get("content") or "")
        if kind in {"image_url", "input_image", "image"}:
            image_url = content.get("image_url")
            url = ""
            if isinstance(image_url, dict):
                candidate = image_url.get("url")
                if isinstance(candidate, str):
                    url = candidate
            elif isinstance(image_url, str):
                url = image_url
            return url or "[image]"
        if kind in {"input_audio", "audio"}:
            return "[audio]"
        if kind:
            return f"[{kind}]"
        if "text" in content:
            return str(content.get("text") or "")
        return "[structured content]"
    return str(content)


def _history_to_messages(history: list[dict]) -> list[dict]:
    messages = []
    tool_call_args = {}

    for m in history:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role not in {"user", "assistant", "tool", "system"}:
            continue
        content_text = _coerce_message_text(m.get("content"))
        if role == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                tc_id = tc.get("id", "")
                if tc_id and fn.get("name"):
                    try:
                        args = json.loads(fn.get("arguments", "{}"))
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    tool_call_args[tc_id] = (fn["name"], args)
            if not content_text.strip():
                continue
        if role == "tool":
            tc_id = m.get("tool_call_id", "")
            tc_info = tool_call_args.get(tc_id) if tc_id else None
            name = (tc_info[0] if tc_info else None) or m.get("tool_name") or "tool"
            args = (tc_info[1] if tc_info else None) or {}
            messages.append({
                "role": "tool",
                "name": name,
                "context": _tool_ctx(name, args),
            })
            continue
        # An assistant turn may carry only reasoning/thinking content with no
        # visible text (extended-thinking turns, thinking-only recovery
        # responses). Such a turn is persisted with its reasoning fields and is
        # recallable from the transcript, but dropping it here as "empty" makes
        # it vanish from the resumed/reloaded session view while the desktop's
        # reasoning disclosure has nothing to render. Keep it when it carries
        # reasoning so the "Thinking…" block still shows. (#44022)
        reasoning_keys = (
            "reasoning",
            "reasoning_content",
            "reasoning_details",
            "codex_reasoning_items",
        )
        has_reasoning = role == "assistant" and any(
            m.get(key) for key in reasoning_keys
        )
        if not content_text.strip() and not has_reasoning:
            continue
        msg = {"role": role, "text": content_text}
        if role == "assistant":
            for key in reasoning_keys:
                if key in m and m.get(key) is not None:
                    msg[key] = m.get(key)
        messages.append(msg)

    return messages


def _coerce_seed_history(value: Any) -> list[dict]:
    if not isinstance(value, list):
        return []

    history = []
    for item in value:
        if not isinstance(item, dict):
            continue

        role = item.get("role")
        if role not in ("user", "assistant", "system"):
            continue

        content = item.get("content")
        if content is None:
            content = item.get("text")
        if not isinstance(content, str) or not content.strip():
            continue

        history.append({"role": role, "content": content})

    return history


def _content_display_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (int, float)):
        return str(content)
    if isinstance(content, list):
        parts = []
        for part in content:
            text = _content_display_text(part).strip()
            if text:
                parts.append(text)
        return "\n".join(parts)
    if isinstance(content, dict):
        kind = content.get("type")
        if kind in {"text", "input_text", "output_text"}:
            return str(content.get("text") or content.get("content") or "")
        if kind in {"image_url", "input_image", "image"}:
            return "[image]"
        if kind in {"input_audio", "audio"}:
            return "[audio]"
        if kind:
            return f"[{kind}]"
        if "text" in content:
            return str(content.get("text") or "")
        return "[structured content]"
    return str(content)


def _inflight_text(value: Any) -> str:
    return _content_display_text(value).strip()


def _start_inflight_turn(session: dict, text: Any) -> None:
    now = time.time()
    session["inflight_turn"] = {
        "assistant": "",
        "started_at": now,
        "streaming": True,
        "updated_at": now,
        "user": _inflight_text(text),
    }


def _append_inflight_delta(session: dict, delta: Any) -> None:
    text = "" if delta is None else str(delta)
    if not text:
        return
    turn = session.get("inflight_turn")
    if not isinstance(turn, dict):
        turn = {"assistant": "", "streaming": True, "user": ""}
    turn["assistant"] = f"{turn.get('assistant') or ''}{text}"
    turn["streaming"] = True
    turn["updated_at"] = time.time()
    session["inflight_turn"] = turn


def _clear_inflight_turn(session: dict) -> None:
    session["inflight_turn"] = None


def _enqueue_prompt(session: dict, text: Any, transport: Any) -> None:
    """Stash a message to run as the very next turn once the live one ends.

    Used when a prompt arrives mid-turn (see ``_handle_busy_submit``). A single
    slot is kept; a second arrival is merged (lossless, mirroring the
    consecutive-user merge in ``repair_message_sequence``) so nothing the user
    typed is dropped. ``transport`` is pinned so the drained turn streams back to
    the client that sent it even if the session transport is rebound meanwhile.
    """
    existing = session.get("queued_prompt")
    if (
        existing
        and isinstance(existing.get("text"), str)
        and isinstance(text, str)
    ):
        prev = existing["text"]
        text = f"{prev}\n\n{text}" if prev and text else (prev or text)
    session["queued_prompt"] = {"text": text, "transport": transport}


def _handle_busy_submit(rid, sid: str, session: dict, text: Any, transport: Any) -> dict:
    """Apply the ``display.busy_input_mode`` policy to a prompt that lands while
    a turn is in flight, instead of rejecting it with ``session busy``.

    The old rejection forced clients into a deadline-bounded busy-retry that
    silently dropped the send when turn teardown outlived the deadline (e.g. a
    slow, non-interruptible tool like ``web_search`` running when the user hits
    stop). The message is instead queued to run as the next turn — and, for the
    default ``interrupt`` policy, the live turn is interrupted so it winds down
    promptly. Drained in ``run``'s tail (see ``_run_prompt_submit``).

    Modes: ``interrupt`` (default) → interrupt + queue; ``queue`` → queue
    without interrupting; ``steer`` → inject into the live turn if accepted,
    else queue.
    """
    mode = _load_busy_input_mode()
    agent = session.get("agent")
    if mode == "steer" and agent is not None and hasattr(agent, "steer"):
        try:
            if agent.steer(text):
                session["last_active"] = time.time()
                return _ok(rid, {"status": "steered"})
        except Exception:
            pass  # fall through to queue
    if mode != "queue" and agent is not None and hasattr(agent, "interrupt"):
        try:
            agent.interrupt()
        except Exception:
            pass
    _enqueue_prompt(session, text, transport)
    session["last_active"] = time.time()
    return _ok(rid, {"status": "queued"})


def _drain_queued_prompt(rid, sid: str, session: dict) -> bool:
    """Fire a queued next-turn prompt if one is waiting and the session is idle.

    Returns True if a queued prompt was dispatched (the caller should then skip
    lower-priority follow-ups this cycle — the user's message wins). Mirrors the
    claim-under-lock pattern used by the goal-continuation re-fire.
    """
    with session["history_lock"]:
        queued = session.get("queued_prompt")
        if not queued or session.get("running"):
            return False
        session["queued_prompt"] = None
        session["running"] = True
        if queued.get("transport") is not None:
            session["transport"] = queued["transport"]
    try:
        _run_prompt_submit(rid, sid, session, queued["text"])
    except Exception as exc:
        print(
            f"[tui_gateway] queued prompt dispatch failed: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        with session["history_lock"]:
            session["running"] = False
    return True


def _inflight_snapshot(session: dict) -> dict | None:
    turn = session.get("inflight_turn")
    if not isinstance(turn, dict):
        return None
    user = str(turn.get("user") or "").strip()
    assistant = str(turn.get("assistant") or "")
    streaming = bool(turn.get("streaming"))
    if not user and not assistant and not streaming:
        return None
    return {
        "assistant": assistant,
        "streaming": streaming,
        "user": user,
    }


# ── Methods: session ─────────────────────────────────────────────────


@method("session.create")
def _(rid, params: dict) -> dict:
    sid = uuid.uuid4().hex[:8]
    key = _new_session_key()
    cols = int(params.get("cols", 80))
    history = _coerce_seed_history(params.get("messages"))
    title = str(params.get("title") or "").strip()
    # When set, this is a branch: the new chat copies an existing conversation's
    # history and links back to it so list_sessions_rich keeps it visible and the
    # sidebar can nest it under its parent. Mirrors the TUI /branch marker.
    parent_session_id = str(params.get("parent_session_id") or "").strip() or None
    # Did the client pick a workspace, or are we falling back to the gateway's
    # launch directory? Only an explicit choice is persisted as the session's
    # workspace (see _ensure_session_db_row); otherwise it lands in "No
    # workspace" instead of whatever folder the desktop launched in.
    raw_cwd = str(params.get("cwd") or "").strip()
    try:
        explicit_cwd = bool(raw_cwd) and os.path.isdir(
            os.path.abspath(os.path.expanduser(raw_cwd))
        )
    except Exception:
        explicit_cwd = False
    resolved_cwd = _completion_cwd(params)
    source = _resolve_session_source(str(params.get("source") or "").strip() or None)
    _enable_gateway_prompts()

    # ``profile`` (app-global remote mode): a new chat started under a non-launch
    # profile must build its agent + persist against THAT profile's home/state.db,
    # not the dashboard's launch profile. Stored on the session so _start_agent_build
    # and each turn re-bind HERMES_HOME. None/own profile → launch (unchanged).
    profile = (params.get("profile") or "").strip() or None
    profile_home = _profile_home(profile)

    # The desktop composer owns its model/effort/fast as plain UI state and ships
    # it on every session.create. Honor each as a PER-SESSION override (built into
    # the agent below) — never a global config write, so picking a model/effort
    # for a new chat can't mutate the profile default. provider is optional
    # (resolved at build).
    create_model = str(params.get("model") or "").strip()
    session_model_override = (
        {"model": create_model, "provider": str(params.get("provider") or "").strip() or None}
        if create_model
        else None
    )
    create_reasoning_override = None
    if effort := str(params.get("reasoning_effort") or "").strip():
        try:
            from hermes_constants import parse_reasoning_effort

            create_reasoning_override = parse_reasoning_effort(effort)
        except Exception:
            create_reasoning_override = None
    # Only pin "fast" when explicitly requested; leaving it None lets the build
    # fall back to the profile default service tier rather than forcing normal.
    create_service_tier_override = "priority" if params.get("fast") else None

    ready = threading.Event()
    now = time.time()
    lease, limit_message = _claim_active_session_slot(
        key, live_session_id=sid, surface=source
    )
    if limit_message is not None:
        return _err(rid, 4090, limit_message)

    with _sessions_lock:
        _sessions[sid] = {
            "agent": None,
            "agent_error": None,
            "agent_ready": ready,
            "attached_images": [],
            "close_on_disconnect": is_truthy_value(params.get("close_on_disconnect", False)),
            "active_session_lease": lease,
            "cols": cols,
            "created_at": now,
            "edit_snapshots": {},
            "explicit_cwd": explicit_cwd,
            "history": history,
            "history_lock": threading.Lock(),
            "history_version": 0,
            "image_counter": 0,
            "cwd": resolved_cwd,
            "inflight_turn": None,
            "last_active": now,
            "model_override": session_model_override,
            "create_reasoning_override": create_reasoning_override,
            "create_service_tier_override": create_service_tier_override,
            "parent_session_id": parent_session_id,
            "pending_title": title or None,
            "profile_home": str(profile_home) if profile_home is not None else None,
            "running": False,
            "session_key": key,
            "show_reasoning": _load_show_reasoning(),
            "source": source,
            "slash_worker": None,
            "tool_progress_mode": _load_tool_progress_mode(),
            "tool_started_at": {},
            "transport": current_transport() or _stdio_transport,
        }
        _register_session_cwd(_sessions[sid])

    # NOTE: we intentionally do NOT persist a DB row here. Every TUI/desktop
    # launch (and every "New agent" / draft) opens a session here just to paint
    # the composer, so eagerly creating a row left an "Untitled" empty session
    # behind for every launch the user never typed into. The row is now created
    # lazily on the first prompt (see _ensure_session_db_row + prompt.submit),
    # and the AIAgent's own INSERT-OR-IGNORE persists it on the first turn too.

    # Return the lightweight session immediately so Ink can paint the composer
    # + skeleton panel, then build the real AIAgent just after this response is
    # flushed.  This keeps startup responsive while still hydrating tools/skills
    # without requiring the user to submit a first prompt.
    _schedule_agent_build(sid)
    _schedule_session_cap_enforcement()  # trim detached idle sessions over the cap

    return _ok(
        rid,
        {
            "session_id": sid,
            "stored_session_id": key,
            "message_count": len(history),
            "messages": _history_to_messages(history),
            "info": {
                # Reflect the per-session model override (desktop composer pick)
                # in the immediate response so the client doesn't briefly clobber
                # its sticky pick with the global default before the deferred
                # build's session.info lands.
                "model": (
                    session_model_override.get("model")
                    if session_model_override
                    else _resolve_model()
                ),
                **(
                    {"provider": session_model_override["provider"]}
                    if session_model_override and session_model_override.get("provider")
                    else {}
                ),
                "tools": {},
                "skills": {},
                "cwd": _sessions[sid]["cwd"],
                "branch": _git_branch_for_cwd(_sessions[sid]["cwd"]),
                "lazy": True,
                "desktop_contract": DESKTOP_BACKEND_CONTRACT,
                "profile_name": _current_profile_name(),
            },
        },
    )


@method("session.list")
def _(rid, params: dict) -> dict:
    db = _get_db()
    if db is None:
        return _db_unavailable_error(rid, code=5006)
    try:
        # Resume picker should surface human conversation sessions from every
        # user-facing surface — CLI, TUI, all gateway platforms (including new
        # ones not enumerated here), ACP adapter clients, webhook sessions,
        # custom `HERMES_SESSION_SOURCE` values, and older installs with
        # different source labels. We deny-list only the noisy internal
        # sources (``tool`` sub-agent runs) rather than allow-listing a
        # fixed set of platform names that goes stale whenever a new
        # platform is added or a user names their own source.
        deny = frozenset({"tool"})

        limit = int(params.get("limit", 200) or 200)
        # Over-fetch modestly so per-source filtering doesn't leave us
        # short; the compression-tip projection in ``list_sessions_rich``
        # can also merge rows.
        fetch_limit = max(limit * 2, 200)
        rows = [
            s
            for s in db.list_sessions_rich(source=None, limit=fetch_limit, order_by_last_active=True, compact_rows=True)
            if (s.get("source") or "").strip().lower() not in deny
        ][:limit]
        return _ok(
            rid,
            {
                "sessions": [
                    {
                        "id": s["id"],
                        "title": s.get("title") or "",
                        "preview": s.get("preview") or "",
                        "started_at": s.get("started_at") or 0,
                        "message_count": s.get("message_count") or 0,
                        "source": s.get("source") or "",
                    }
                    for s in rows
                ]
            },
        )
    except Exception as e:
        return _err(rid, 5006, str(e))


@method("session.most_recent")
def _(rid, params: dict) -> dict:
    """Return the most recent human-facing session id, or ``None``.

    Mirrors ``session.list``'s deny-list behaviour (drops ``tool``
    sub-agent rows).  Used by TUI auto-resume when
    ``display.tui_auto_resume_recent`` is on; the field is also handy
    for any CLI tooling that wants "latest session" without paginating
    the full list.

    Contract: a ``{"session_id": null}`` result means "no eligible
    session found right now".  Errors are also folded into that
    null-result shape (and logged) so callers don't have to special-
    case JSON-RPC error envelopes for what is a normal "no answer".
    """
    db = _get_db()
    if db is None:
        return _ok(rid, {"session_id": None})
    try:
        deny = frozenset({"tool"})
        # Over-fetch by a generous bounded amount so heavy sub-agent
        # users (lots of recent ``tool`` rows) don't get a false
        # "no eligible session" answer.  ``session.list`` uses a
        # similar over-fetch strategy.
        rows = db.list_sessions_rich(source=None, limit=200, order_by_last_active=True, compact_rows=True)
        for row in rows:
            src = (row.get("source") or "").strip().lower()
            if src in deny:
                continue
            return _ok(
                rid,
                {
                    "session_id": row.get("id"),
                    "title": row.get("title") or "",
                    "started_at": row.get("started_at") or 0,
                    "source": row.get("source") or "",
                },
            )
        return _ok(rid, {"session_id": None})
    except Exception:
        logger.exception("session.most_recent failed")
        return _ok(rid, {"session_id": None})


@method("project.facts")
def _(rid, params: dict) -> dict:
    """Structured project facts for a cwd — manifests, package manager, the
    exact verify commands, and context files.

    The same detection the coding-context posture (#43316) bakes into the system
    prompt, exposed so UIs (the desktop verify surface) consume it instead of
    re-sniffing. ``{"facts": null}`` means the cwd isn't a code workspace.
    """
    try:
        from agent.coding_context import project_facts_for

        return _ok(rid, {"facts": project_facts_for(params.get("cwd"))})
    except Exception:
        logger.exception("project.facts failed")
        return _ok(rid, {"facts": None})


@method("verification.status")
def _(rid, params: dict) -> dict:
    """Best known coding verification evidence for a cwd/session.

    Read-only consumer of the core ledger. It never runs checks and never
    upgrades targeted evidence into a repository-wide guarantee.
    """
    try:
        from agent.verification_evidence import verification_status

        return _ok(
            rid,
            {
                "verification": verification_status(
                    session_id=params.get("session_id") or params.get("session_key"),
                    cwd=params.get("cwd"),
                )
            },
        )
    except Exception:
        logger.exception("verification.status failed")
        return _ok(rid, {"verification": {"status": "unknown", "evidence": None}})


def _lazy_resume_info(cwd: str, *, model: str = "", provider: str = "") -> dict:
    """session.info for a not-yet-built session (the shape session.create
    returns). tools/skills land later when the deferred build emits session.info."""
    info = {
        "cwd": cwd,
        "branch": _git_branch_for_cwd(cwd),
        "model": model or _resolve_model(),
        "tools": {},
        "skills": {},
        "lazy": True,
        "desktop_contract": DESKTOP_BACKEND_CONTRACT,
        "profile_name": _current_profile_name(),
    }
    if provider:
        info["provider"] = provider
    return info


def _deferred_session_record(
    session_key: str,
    *,
    cols: int,
    cwd: str,
    history: list,
    lease,
    source: str = "tui",
    close_on_disconnect: bool = False,
    display_history_prefix: list | None = None,
    profile_home: Path | None = None,
    lazy: bool = False,
    model_override=None,
    resume_runtime_overrides: dict | None = None,
) -> dict:
    """A live-session record whose AIAgent is built later (lazy watch / cold
    resume) — _init_session's shape minus the agent."""
    now = time.time()
    return {
        "agent": None,
        "agent_error": None,
        "agent_ready": threading.Event(),
        "attached_images": [],
        "close_on_disconnect": close_on_disconnect,
        "active_session_lease": lease,
        "cols": cols,
        "created_at": now,
        "cwd": cwd,
        "display_history_prefix": display_history_prefix or [],
        "edit_snapshots": {},
        "explicit_cwd": False,
        "history": history,
        "history_lock": threading.Lock(),
        "history_version": 0,
        "image_counter": 0,
        "inflight_turn": None,
        "last_active": now,
        "lazy": lazy,
        "model_override": model_override,
        "pending_title": None,
        "profile_home": str(profile_home) if profile_home is not None else None,
        "resume_runtime_overrides": resume_runtime_overrides,
        "resume_session_id": session_key,
        "running": False,
        "session_key": session_key,
        "show_reasoning": _load_show_reasoning(),
        "slash_worker": None,
        "source": source,
        "tool_progress_mode": _load_tool_progress_mode(),
        "tool_started_at": {},
        "transport": current_transport() or _stdio_transport,
    }


def _claim_or_reuse_live(
    sid: str, session_key: str, record: dict, lease
) -> tuple[str, dict] | None:
    """Register ``record`` as the live session for ``session_key`` under the
    resume lock, or — if a concurrent resume already won — release ``lease`` and
    return the winner for the caller to reuse."""
    with _session_resume_lock:
        live = _find_live_session_by_key(session_key)
        if live is not None:
            if lease is not None:
                lease.release()
            return live
        with _sessions_lock:
            _sessions[sid] = record
            _register_session_cwd(_sessions[sid])
    return None


def _schedule_agent_build(sid: str, delay: float = 0.05) -> None:
    """Pre-warm a deferred session's agent off the response path (session.create
    and cold resume both build through here; _sess() also builds on demand)."""

    def _run():
        session = _sessions.get(sid)
        if session is not None:
            _start_agent_build(sid, session)

    timer = threading.Timer(delay, _run)
    timer.daemon = True
    timer.start()


@method("session.resume")
def _(rid, params: dict) -> dict:
    target = params.get("session_id", "")
    if not target:
        return _err(rid, 4006, "session_id required")
    try:
        cols = int(params.get("cols", 80))
    except (TypeError, ValueError):
        cols = 80
    # ``profile`` (app-global remote mode): resume a session that lives in another
    # local profile's state.db. None/own profile → the launch profile (unchanged).
    profile = (params.get("profile") or "").strip() or None
    profile_home = _profile_home(profile)

    # In a profile scope, the agent OWNS a long-lived db handle bound to that
    # profile (do NOT auto-close it here). Otherwise reuse the shared launch db.
    if profile_home is not None:
        from hermes_state import SessionDB

        db = SessionDB(db_path=profile_home / "state.db")
    else:
        db = _get_db()
    if db is None:
        return _db_unavailable_error(rid, code=5000)

    found = db.get_session(target)
    if not found:
        found = db.get_session_by_title(target)
        if found:
            target = found["id"]
        elif is_truthy_value(params.get("lazy", False)) and _child_run_active(target):
            # Race: a watch window opened on a freshly-spawned subagent. The
            # child relays `subagent.start` (which carries child_session_id and
            # triggers the window) BEFORE its first run_conversation() flushes
            # the DB row via _ensure_db_session, so db.get_session(target) is
            # momentarily empty. On slower hosts (notably WSL2, where SQLite +
            # process scheduling widen the gap) the window's resume consistently
            # lands inside this window and used to hard-fail "session not found"
            # — the frontend then 404'd on the REST messages fallback and the
            # window spun forever. The child is provably live (_child_run_active),
            # so proceed into the lazy branch with empty history; the live mirror
            # streams the whole turn anyway and the row exists by upgrade time.
            found = {}
        else:
            return _err(rid, 4007, "session not found")

    # Follow the compression-continuation chain to the live tip so a resume on
    # a rotated-out parent id binds to the descendant that actually holds the
    # post-compression turns. Auto-compression ends the session and forks a
    # continuation child; without this, resuming the original id (the desktop's
    # routed id when the chat was opened before it rotated) reloads the parent
    # transcript and the response generated after compression is missing — the
    # "I came back and the reply isn't there" bug on large sessions. Resolving
    # here also re-anchors the fast path below so a still-live rotated session
    # is reused (by its new key) instead of rebuilding a duplicate agent on the
    # stale parent. Skipped for lazy watch windows, which intentionally attach
    # to the exact child branch they were opened on.
    if found and not is_truthy_value(params.get("lazy", False)):
        try:
            tip = db.resolve_resume_session_id(target)
        except Exception:
            tip = target
        if tip and tip != target:
            target = tip
            found = db.get_session(target) or found

    profile_resume_cwd = str(found.get("cwd") or "").strip() or _profile_configured_cwd(
        profile_home
    )

    def _reuse_live_payload(sid: str, session: dict) -> dict:
        payload = _live_session_payload(
            sid,
            session,
            cols=cols,
            touch=True,
            transport=current_transport() or _stdio_transport,
        )
        payload["resumed"] = target
        # A lazy watch session never owns a run loop, so its payload's running
        # flag is always False — overlay the child-run registry so a reconnecting
        # watch window keeps its busy indicator while the child is still mid-run.
        if session.get("agent") is None and _child_run_active(target):
            payload["running"] = True
            payload["status"] = "streaming"
        return payload

    # Fast path: if the session is already live, reuse it under the lock.
    with _session_resume_lock:
        live = _find_live_session_by_key(target)
        if live is not None:
            return _ok(rid, _reuse_live_payload(*live))

    # Lazy/watch resume: register the live session WITHOUT building an agent.
    # Used by the desktop's subagent windows — the child runs inside the
    # parent's turn, so its window only needs the stored history plus a
    # transport for the child-mirror's live events. Skipping _make_agent here
    # is what keeps the window cheap while the backend is busy running the
    # delegation. A later prompt.submit upgrades it via _start_agent_build
    # (resume_session_id keeps the upgrade on the stored conversation).
    if is_truthy_value(params.get("lazy", False)):
        sid = uuid.uuid4().hex[:8]
        source = _resolve_session_source(str(params.get("source") or "").strip() or None)
        lease, limit_message = _claim_active_session_slot(
            target, live_session_id=sid, surface=source
        )
        if limit_message is not None:
            return _err(rid, 4090, limit_message)
        try:
            db.reopen_session(target)
            # The child's OWN conversation only — include_ancestors would prepend
            # the parent's transcript onto the subagent's branch.
            history = db.get_messages_as_conversation(target)
        except Exception as e:
            if lease is not None:
                lease.release()
            return _err(rid, 5000, f"resume failed: {e}")
        cwd = profile_resume_cwd or _default_session_cwd()
        record = _deferred_session_record(
            target,
            cols=cols,
            cwd=cwd,
            history=history,
            lease=lease,
            source=source,
            close_on_disconnect=is_truthy_value(params.get("close_on_disconnect", False)),
            profile_home=profile_home,
            lazy=True,
        )
        if (live := _claim_or_reuse_live(sid, target, record, lease)) is not None:
            return _ok(rid, _reuse_live_payload(*live))
        # A delegated child mid-run emits no session events of its own — report
        # its liveness from the relay registry so the window shows a busy turn.
        child_running = _child_run_active(target)
        messages = _history_to_messages(history)
        return _ok(
            rid,
            {
                "session_id": sid,
                "resumed": target,
                "message_count": len(messages),
                "messages": messages,
                "info": _lazy_resume_info(cwd),
                "inflight": None,
                "running": child_running,
                "session_key": target,
                "started_at": record["created_at"],
                "status": "streaming" if child_running else "idle",
            },
        )

    # Cold resume default: register the live session and read its stored
    # transcript, but build the agent OFF the response path. _make_agent can
    # block for seconds (MCP discovery, prompt/skill build, AIAgent
    # construction), and every resume caller (desktop + Ink TUI) awaits this RPC
    # before it paints — so building eagerly is the bulk of the multi-second
    # "switching sessions is frozen" latency. Return the full display transcript
    # immediately and pre-warm the agent on a short timer (the same deferred-
    # build contract session.create uses); _sess() also builds on demand if the
    # first prompt beats the timer. A caller that needs the agent built
    # synchronously (e.g. tests of the build race) passes ``eager_build: true``
    # to fall through to the eager path below. Distinct from the lazy/watch
    # branch above: a normal resume restores the full ancestor history and the
    # session's persisted runtime identity, and is a real (upgradable) session.
    if not is_truthy_value(params.get("eager_build", False)):
        sid = uuid.uuid4().hex[:8]
        source = _resolve_session_source(str(params.get("source") or "").strip() or None)
        lease, limit_message = _claim_active_session_slot(
            target, live_session_id=sid, surface=source
        )
        if limit_message is not None:
            return _err(rid, 4090, limit_message)
        # Interactive resume routes approvals/clarify through gateway prompts;
        # the deferred build wires the remaining per-session callbacks.
        _enable_gateway_prompts()
        try:
            db.reopen_session(target)
            raw_history = db.get_messages_as_conversation(target)
            display_history = db.get_messages_as_conversation(target, include_ancestors=True)
        except Exception as e:
            if lease is not None:
                lease.release()
            return _err(rid, 5000, f"resume failed: {e}")
        # Display keeps the full transcript; the model-fed history drops a
        # dangling/interrupted tool-call tail so a session killed mid-loop does
        # not replay the unanswered call forever (#29086).
        prefix = display_history[: max(0, len(display_history) - len(raw_history))]
        history = sanitize_replay_history(raw_history)
        # Restore the model/provider/reasoning/tier this chat last used so the
        # deferred build (and the info below) match the eager path — without them
        # the build drops the provider ("No LLM provider configured").
        overrides = _stored_session_runtime_overrides(found) or {}
        model_override = overrides.get("model_override") or {}
        cwd = profile_resume_cwd or _default_session_cwd()
        record = _deferred_session_record(
            target,
            cols=cols,
            cwd=cwd,
            history=history,
            lease=lease,
            source=source,
            close_on_disconnect=is_truthy_value(params.get("close_on_disconnect", False)),
            display_history_prefix=prefix,
            profile_home=profile_home,
            model_override=overrides.get("model_override"),
            resume_runtime_overrides=overrides or None,
        )
        if (live := _claim_or_reuse_live(sid, target, record, lease)) is not None:
            return _ok(rid, _reuse_live_payload(*live))

        _schedule_agent_build(sid)
        _schedule_session_cap_enforcement()  # trim detached idle sessions over the cap

        messages = _history_to_messages(display_history)
        return _ok(
            rid,
            {
                "session_id": sid,
                "resumed": target,
                "message_count": len(messages),
                "messages": messages,
                "info": _lazy_resume_info(
                    cwd,
                    model=model_override.get("model") or "",
                    provider=overrides.get("provider_override") or "",
                ),
                "inflight": None,
                "running": False,
                "session_key": target,
                "started_at": record["created_at"],
                "status": "idle",
            },
        )

    # Build the agent OUTSIDE the lock — _make_agent can block for seconds
    # (MCP discovery, prompt/skill build, AIAgent construction). Holding
    # _session_resume_lock across it would stall session.close on the main
    # dispatch thread (it's not a _LONG_HANDLER), blocking fast-path RPCs.
    sid = uuid.uuid4().hex[:8]
    source = _resolve_session_source(str(params.get("source") or "").strip() or None)
    lease, limit_message = _claim_active_session_slot(
        target, live_session_id=sid, surface=source
    )
    if limit_message is not None:
        return _err(rid, 4090, limit_message)
    _enable_gateway_prompts()
    home_token = (
        set_hermes_home_override(str(profile_home))
        if profile_home is not None
        else None
    )
    try:
        db.reopen_session(target)
        raw_history = db.get_messages_as_conversation(target)
        display_history = db.get_messages_as_conversation(
            target, include_ancestors=True
        )
        # The display transcript keeps every row so the user still sees their
        # full history.  The model-fed history is sanitized: a session whose
        # last turn died mid-tool-loop persists a dangling assistant(tool_calls)
        # (or interrupted assistant→tool) tail; replaying it makes the model
        # re-issue the unanswered call forever — the permanent-"thinking" stuck
        # session in #29086.  The messaging gateway already strips this; this is
        # the WebUI/TUI resume path picking up the same cleanup.
        display_history_prefix = display_history[
            : max(0, len(display_history) - len(raw_history))
        ]
        history = sanitize_replay_history(raw_history)
        messages = _history_to_messages(display_history)
        tokens = _set_session_context(target)
        try:
            # Pass the profile's db so the agent persists turns to the right
            # state.db; home override is active here so config/skills/model
            # resolve to the profile too. Runtime identity is restored from the
            # stored session row so switching chats does not inherit whatever
            # global model another chat last selected.
            stored_runtime_overrides = _stored_session_runtime_overrides(found)
            agent = _make_agent(
                sid,
                target,
                session_id=target,
                session_db=db,
                platform_override=source,
                **stored_runtime_overrides,
            )
        finally:
            _clear_session_context(tokens)
    except Exception as e:
        if lease is not None:
            lease.release()
        return _err(rid, 5000, f"resume failed: {e}")
    finally:
        if home_token is not None:
            reset_hermes_home_override(home_token)

    # Double-checked locking: another concurrent resume may have created the
    # live session while we were building. Re-check under the lock; if it won,
    # discard our just-built agent and reuse theirs (no worker/poller wired yet).
    with _session_resume_lock:
        live = _find_live_session_by_key(target)
        if live is not None:
            try:
                if hasattr(agent, "close"):
                    agent.close()
            except Exception:
                pass
            if lease is not None:
                lease.release()
            other_sid, other_session = live
            payload = _live_session_payload(
                other_sid,
                other_session,
                cols=cols,
                touch=True,
                transport=current_transport() or _stdio_transport,
            )
            payload["resumed"] = target
            return _ok(rid, payload)
        try:
            init_home_token = (
                set_hermes_home_override(str(profile_home))
                if profile_home is not None
                else None
            )
            try:
                _init_session(
                    sid,
                    target,
                    agent,
                    history,
                    cols=cols,
                    cwd=profile_resume_cwd,
                    session_db=db,
                    source=source,
                )
            finally:
                if init_home_token is not None:
                    reset_hermes_home_override(init_home_token)
            if sid in _sessions:
                if stored_runtime_overrides.get("model_override") is not None:
                    _sessions[sid]["model_override"] = stored_runtime_overrides[
                        "model_override"
                    ]
                _sessions[sid]["display_history_prefix"] = display_history_prefix
                # Remember the profile home so each turn re-binds HERMES_HOME (the
                # agent persists to its own db, but mid-turn home reads — memory,
                # skills — must resolve to the resumed profile too).
                if profile_home is not None:
                    _sessions[sid]["profile_home"] = str(profile_home)
                _sessions[sid]["active_session_lease"] = lease
        except Exception as e:
            if lease is not None:
                lease.release()
            return _err(rid, 5000, f"resume failed: {e}")
        session = _sessions.get(sid) or {}
    return _ok(
        rid,
        {
            "session_id": sid,
            "resumed": target,
            "message_count": len(messages),
            "messages": messages,
            "info": _session_info(agent, session),
            "inflight": None,
            "running": False,
            "session_key": target,
            "started_at": float(session.get("created_at") or time.time()),
            "status": "idle",
        },
    )


@method("session.cwd.set")
def _(rid, params: dict) -> dict:
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    if session.get("running"):
        return _err(rid, 4009, "session busy")
    raw = str(params.get("cwd", "") or "").strip()
    if not raw:
        return _err(rid, 4016, "cwd required")
    try:
        cwd = _set_session_cwd(session, raw)
    except ValueError as e:
        return _err(rid, 4017, str(e))
    agent = session.get("agent")
    info = (
        _session_info(agent, session)
        if agent is not None
        else {
            "cwd": cwd,
            "branch": _git_branch_for_cwd(cwd),
            "lazy": True,
        }
    )
    _emit("session.info", params.get("session_id", ""), info)
    return _ok(rid, info)


def _session_pending_kind(sid: str) -> str:
    for rid, (owner_sid, _ev) in list(_pending.items()):
        if owner_sid != sid:
            continue
        event, _payload = _pending_prompt_payloads.get(rid, ("input.request", {}))
        return str(event).removesuffix(".request")
    return ""


def _session_live_status(sid: str, session: dict) -> str:
    if _session_pending_kind(sid):
        return "waiting"
    ready = session.get("agent_ready")
    # Unset + build never started = a lazy watch session sitting idle, not a
    # session stuck mid-construction.
    if ready is not None and not ready.is_set() and session.get("agent_build_started"):
        return "starting"
    if session.get("running"):
        return "working"
    return "idle"


def _message_preview(history: list) -> str:
    for msg in reversed(history or []):
        text = _content_display_text(msg.get("content", msg.get("text", ""))).strip()
        if text:
            return " ".join(text.split())[:160]
    return ""


def _session_live_title(session: dict, key: str) -> str:
    title = str(session.get("pending_title") or "").strip()
    db = _get_db()
    if db is not None:
        try:
            title = str(db.get_session_title(key) or title or "").strip()
        except Exception:
            pass
    return title


def _session_live_item(sid: str, session: dict, current_sid: str = "") -> dict:
    key = _session_lookup_key(session, fallback=sid)
    agent = session.get("agent")
    history = list(session.get("history") or [])
    status = _session_live_status(sid, session)
    inflight = _inflight_snapshot(session)
    preview = _message_preview(history)
    if inflight:
        preview = inflight.get("assistant") or inflight.get("user") or preview
        preview = " ".join(str(preview).split())[:160]
    now = time.time()
    return {
        "current": sid == current_sid,
        "id": sid,
        "last_active": float(
            session.get("last_active") or session.get("created_at") or now
        ),
        "message_count": len(history),
        "model": str(getattr(agent, "model", "") or _resolve_model()),
        "preview": preview,
        "session_key": key,
        "started_at": float(session.get("created_at") or now),
        "status": status,
        "title": _session_live_title(session, key),
    }


def _session_lookup_key(session: dict, *, fallback: str = "") -> str:
    agent = session.get("agent")
    return str(
        getattr(agent, "session_id", None)
        or session.get("session_key")
        or fallback
        or ""
    )


def _find_live_session_by_key(session_key: str) -> tuple[str, dict] | None:
    for sid, session in list(_sessions.items()):
        if session.get("_finalized"):
            continue
        if _session_lookup_key(session, fallback=sid) == session_key:
            return sid, session
    return None


def _fallback_session_info(session: dict) -> dict:
    agent = session.get("agent")
    if agent is not None:
        return _session_info(agent)
    return {
        "cwd": _default_session_cwd(),
        "lazy": True,
        "model": _resolve_model(),
        "skills": {},
        "tools": {},
    }


def _live_session_payload(
    sid: str,
    session: dict,
    *,
    cols: int | None = None,
    touch: bool = False,
    transport: Transport | None = None,
) -> dict:
    with session["history_lock"]:
        if cols is not None:
            session["cols"] = cols
        if transport is not None:
            session["transport"] = transport
        if touch:
            session["last_active"] = time.time()
        history = list(session.get("display_history_prefix") or []) + list(
            session.get("history") or []
        )
        inflight = _inflight_snapshot(session)
        running = bool(session.get("running"))
    payload = {
        "info": _fallback_session_info(session),
        "message_count": len(history),
        "messages": _history_to_messages(history),
        "running": running,
        "session_id": sid,
        "session_key": _session_lookup_key(session, fallback=sid),
        "started_at": float(session.get("created_at") or time.time()),
        "status": _session_live_status(sid, session),
    }
    if inflight:
        payload["inflight"] = inflight
    return payload


@method("session.active_list")
def _(rid, params: dict) -> dict:
    """Return live TUI sessions in this gateway process.

    Unlike ``session.list`` this is not a historical DB browser: it reports only
    sessions with in-memory agents/workers that the current TUI can switch to
    without closing siblings.
    """
    current = str(params.get("current_session_id") or "")
    try:
        with _sessions_lock:
            snapshot = list(_sessions.items())
    except Exception as e:
        return _err(rid, 5036, f"could not enumerate active sessions: {e}")

    # Liveness filter (#38950): a session whose teardown has begun (``_finalized``)
    # is dead — its agent/worker are being released and it is no longer
    # attachable — but it can briefly remain in ``_sessions`` until the reaper
    # pops it (the WS grace-reap and idle reaper both set ``_finalized`` inside
    # ``_teardown_session`` before the pop). Counting these inflated the footer's
    # "N sessions" count, which only ever went up until a gateway restart. Drop
    # them here so the count reflects genuinely attachable sessions. We do NOT
    # filter on ``transport is _detached_ws_transport`` (the WS-detached drop
    # sentinel): a detached session is still attachable via a quick reconnect /
    # session.resume until the grace-reap finalizes it, and a standalone
    # ``hermes --tui`` session legitimately rides the real stdio transport and
    # must stay visible.
    # Keep the natural creation/insertion order from ``_sessions``.  The
    # frontend marks the focused session with ``current``; it should not jump to
    # the top just because the user switched to it.
    rows = [
        _session_live_item(sid, session, current)
        for sid, session in snapshot
        if not session.get("_finalized")
    ]
    return _ok(rid, {"sessions": rows})


@method("session.activate")
def _(rid, params: dict) -> dict:
    """Attach the frontend to an already-live TUI session.

    This intentionally does not close the previously focused session; it merely
    returns enough state for Ink to redraw around another live session id.
    """
    sid = str(params.get("session_id") or "")
    session, err = _sess_nowait({"session_id": sid}, rid)
    if err:
        return err
    assert session is not None

    return _ok(
        rid,
        _live_session_payload(
            sid,
            session,
            touch=True,
            transport=current_transport() or _stdio_transport,
        ),
    )


@method("session.delete")
def _(rid, params: dict) -> dict:
    """Delete a stored session and its on-disk transcript files.

    Used by the TUI resume picker (``d`` key) so users can prune old
    sessions without dropping to the CLI.  Refuses to delete a session
    that is currently active in this gateway process — those rows are
    still being written to and removing them out from under the live
    agent corrupts message ordering and trips FK constraints when the
    next message append flushes.
    """
    target = params.get("session_id", "")
    if not target:
        return _err(rid, 4006, "session_id required")
    db = _get_db()
    if db is None:
        return _db_unavailable_error(rid, code=5036)
    # Block deletion of any session currently bound to a live TUI session
    # in this process.  The picker hides the active session anyway, but a
    # racing caller could still target it.  Snapshot via ``list(...)``
    # because ``_sessions`` is mutated by concurrent RPCs on the thread
    # pool — iterating the dict directly can raise ``RuntimeError:
    # dictionary changed size during iteration``.  If even the snapshot
    # raises, fail closed (refuse the delete) rather than fail open.
    try:
        with _sessions_lock:
            snapshot = list(_sessions.values())
    except Exception as e:
        return _err(rid, 5036, f"could not enumerate active sessions: {e}")
    active = {s.get("session_key") for s in snapshot if s.get("session_key")}
    if target in active:
        return _err(rid, 4023, "cannot delete an active session")
    sessions_dir = get_hermes_home() / "sessions"
    try:
        deleted = db.delete_session(target, sessions_dir=sessions_dir)
    except Exception as e:
        return _err(rid, 5036, f"delete failed: {e}")
    if not deleted:
        return _err(rid, 4007, "session not found")
    return _ok(rid, {"deleted": target})


@method("session.title")
def _(rid, params: dict) -> dict:
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    key = session["session_key"]
    if "title" not in params:
        fallback = session.get("pending_title") or ""
        with _session_db(session) as db:
            if db is None:
                return _db_unavailable_error(rid, code=5007)
            try:
                resolved_title = db.get_session_title(key) or ""
                if fallback:
                    if db.set_session_title(key, fallback):
                        session["pending_title"] = None
                        resolved_title = fallback
                    else:
                        existing_row = db.get_session(key)
                        existing_title = ((existing_row or {}).get("title") or "").strip()
                        if existing_title == fallback:
                            session["pending_title"] = None
                            resolved_title = fallback
                        elif not resolved_title:
                            resolved_title = fallback
                elif resolved_title:
                    session["pending_title"] = None
            except Exception:
                resolved_title = fallback
        _emit_session_info_for_session(params.get("session_id", ""), session)
        return _ok(
            rid,
            {
                "title": resolved_title,
                "session_key": key,
            },
        )
    title = (params.get("title", "") or "").strip()
    if not title:
        return _err(rid, 4021, "title required")
    db = _session_db_for_immediate_use(session)
    if db is None:
        return _db_unavailable_error(rid, code=5007)
    try:
        if db.set_session_title(key, title):
            session["pending_title"] = None
            _emit_session_info_for_session(params.get("session_id", ""), session)
            return _ok(rid, {"pending": False, "title": title})
        # rowcount == 0 can mean "same value" as well as "missing row".
        existing_row = db.get_session(key)
        if existing_row:
            session["pending_title"] = None
            _emit_session_info_for_session(params.get("session_id", ""), session)
            return _ok(
                rid,
                {
                    "pending": False,
                    "session_key": key,
                    "title": (existing_row.get("title") or title),
                },
            )
        # No row yet (the DB write is deferred to the first prompt so empty
        # drafts don't litter the sidebar). An explicit /title is clear user
        # intent, not an abandoned draft — so persist the row NOW and set the
        # title, mirroring the messaging gateway's _handle_title_command. The
        # old behavior only queued pending_title and relied on the post-turn
        # apply block; if that turn never landed under this session_key the
        # title was silently lost and the sidebar fell back to the message
        # preview. Creating the row up front removes that race entirely. The
        # min-messages sidebar filter keeps a titled 0-message row hidden, so
        # a /title'd-but-never-used draft still doesn't clutter the list.
        _ensure_session_db_row(session)
        with _session_db(session) as scoped_db:
            if scoped_db is not None and scoped_db.set_session_title(key, title):
                session["pending_title"] = None
                _emit_session_info_for_session(params.get("session_id", ""), session)
                return _ok(rid, {"pending": False, "title": title})
        # Row creation didn't take (DB unavailable, or a concurrent writer) —
        # fall back to queuing so the post-turn apply block can still recover.
        session["pending_title"] = title
        _emit_session_info_for_session(params.get("session_id", ""), session)
        return _ok(rid, {"pending": True, "title": title})
    except ValueError as e:
        return _err(rid, 4022, str(e))
    except Exception as e:
        return _err(rid, 5007, str(e))
    finally:
        if session.get("profile_home"):
            with contextlib.suppress(Exception):
                db.close()


def _main_runtime_from_agent(agent) -> dict | None:
    """Build an aux-client main_runtime override from a live agent.

    Lets a one-shot inherit the session's provider/model/credentials so its
    output matches the model the user is actually coding with, instead of
    falling back to the cheapest auto-detected backend.
    """
    if agent is None:
        return None
    runtime: dict = {}
    for field in ("provider", "model", "base_url", "api_key", "api_mode", "auth_mode"):
        value = getattr(agent, field, None)
        if isinstance(value, str) and value.strip():
            runtime[field] = value.strip()
        elif field == "api_key" and callable(value):
            runtime[field] = value
    return runtime or None


@method("llm.oneshot")
def _(rid, params: dict) -> dict:
    """Run a single stateless LLM request outside any conversation.

    Generic helper for small generative chores (e.g. a commit message from a
    diff). Accepts either a named ``template`` + ``variables`` or an explicit
    ``instructions`` / ``input`` pair. When ``session_id`` resolves to a live
    session the call inherits that agent's model; otherwise it uses the
    configured auxiliary ``task`` backend. Never mutates session history, so
    prompt caching is untouched.
    """
    template = (params.get("template") or "").strip() or None
    instructions = params.get("instructions") or ""
    user_input = params.get("input") or ""
    variables = params.get("variables") if isinstance(params.get("variables"), dict) else {}
    task = (params.get("task") or "title_generation").strip() or "title_generation"

    try:
        max_tokens = int(params.get("max_tokens") or 1024)
    except (TypeError, ValueError):
        max_tokens = 1024
    temperature = params.get("temperature")
    if temperature is not None:
        try:
            temperature = float(temperature)
        except (TypeError, ValueError):
            temperature = None

    if not template and not str(instructions).strip() and not str(user_input).strip():
        return _err(rid, 4030, "llm.oneshot requires a template or instructions/input")

    # Optional: inherit the live session's model (no error if absent).
    session = _sessions.get(params.get("session_id") or "")
    main_runtime = _main_runtime_from_agent(session.get("agent")) if session else None

    try:
        from agent.oneshot import run_oneshot

        text = run_oneshot(
            instructions=instructions,
            user_input=user_input,
            template=template,
            variables=variables,
            task=task,
            max_tokens=max_tokens,
            temperature=temperature if temperature is not None else 0.3,
            main_runtime=main_runtime,
        )
    except KeyError as e:
        return _err(rid, 4031, str(e))
    except ValueError as e:
        return _err(rid, 4032, str(e))
    except Exception as e:
        logger.warning("llm.oneshot failed: %s", e)
        return _err(rid, 5030, f"one-shot generation failed: {e}")

    return _ok(rid, {"text": text})


@method("handoff.request")
def _(rid, params: dict) -> dict:
    """Queue a handoff of this session to a messaging platform.

    Desktop parity with the CLI ``/handoff`` command: we only write
    ``handoff_state='pending'`` onto the persisted session row. The actual
    transfer is performed by the separate ``hermes gateway`` process, whose
    ``_handoff_watcher`` claims the row, re-binds the session to the platform's
    home channel, and forges a synthetic turn. The desktop then polls
    ``handoff.state`` for the terminal result.
    """
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    if session.get("running"):
        return _err(
            rid,
            4009,
            "session busy — wait for the current turn to finish, then retry the handoff",
        )

    platform_name = (params.get("platform", "") or "").strip().lower()
    if not platform_name:
        return _err(rid, 4023, "platform required")

    # Validate against the live gateway config — an unconfigured platform or a
    # missing home channel would leave the handoff pending forever, so reject
    # up front with a clear, actionable message (mirrors cli.py).
    try:
        from gateway.config import Platform, load_gateway_config
    except Exception as e:  # pragma: no cover — gateway pkg always ships
        return _err(rid, 5021, f"could not load gateway config: {e}")
    try:
        platform = Platform(platform_name)
    except (ValueError, KeyError):
        return _err(rid, 4024, f"unknown platform '{platform_name}'")
    try:
        gw_config = load_gateway_config()
    except Exception as e:
        return _err(rid, 5021, f"could not load gateway config: {e}")
    pcfg = gw_config.platforms.get(platform)
    if not pcfg or not pcfg.enabled:
        return _err(
            rid,
            4025,
            f"platform '{platform_name}' is not configured/enabled in the gateway",
        )
    home = gw_config.get_home_channel(platform)
    if not home or not home.chat_id:
        return _err(
            rid,
            4026,
            f"no home channel configured for {platform_name} — set one with "
            "/sethome on the destination chat first",
        )

    # The watcher transfers a persisted DB row, so make sure one exists even
    # for a brand-new empty chat (mirrors the CLI's set_session_title stub).
    _ensure_session_db_row(session)

    with _session_db(session) as db:
        if db is None:
            return _db_unavailable_error(rid, code=5007)
        key = session["session_key"]
        try:
            if not db.get_session(key):
                db.set_session_title(key, f"handoff-{key[:8]}")
            ok = db.request_handoff(key, platform_name)
        except Exception as e:
            return _err(rid, 5007, str(e))

    if not ok:
        return _err(
            rid,
            4027,
            "session is already in flight for handoff — wait for it to settle, then retry",
        )
    return _ok(
        rid,
        {
            "queued": True,
            "session_key": key,
            "platform": platform_name,
            "home_name": home.name,
        },
    )


@method("handoff.state")
def _(rid, params: dict) -> dict:
    """Poll the handoff state for a session.

    Returns ``{state, platform, error}`` where ``state`` is one of
    ``pending|running|completed|failed`` (or empty when no handoff record
    exists). Desktop polls this after ``handoff.request``.
    """
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    with _session_db(session) as db:
        if db is None:
            return _db_unavailable_error(rid, code=5007)
        record = db.get_handoff_state(session["session_key"])

    record = record or {}
    return _ok(
        rid,
        {
            "state": record.get("state") or "",
            "platform": record.get("platform") or "",
            "error": record.get("error") or "",
        },
    )


@method("handoff.fail")
def _(rid, params: dict) -> dict:
    """Mark an in-flight handoff as failed so the user can retry.

    Desktop calls this when its bounded poll times out. Only pending/running
    rows are changed so a late success from the gateway watcher is not clobbered.
    """
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    reason = str(params.get("error") or "handoff failed").strip()[:500]
    with _session_db(session) as db:
        if db is None:
            return _db_unavailable_error(rid, code=5007)
        key = session["session_key"]
        record = db.get_handoff_state(key) or {}
        state = record.get("state") or ""
        if state in {"pending", "running"}:
            db.fail_handoff(key, reason)
            return _ok(rid, {"failed": True, "state": "failed"})

    return _ok(rid, {"failed": False, "state": state})


@method("session.usage")
def _(rid, params: dict) -> dict:
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    agent = session.get("agent")
    usage: dict = (
        _get_usage(agent)
        if agent is not None
        else {"calls": 0, "input": 0, "output": 0, "total": 0}
    )
    # Nous credits block — agent-independent (a portal fetch), so it shows even
    # with zero API calls or on a resumed session. The TUI /usage panel renders
    # these lines regardless of `calls`. Fail-open: [] when not logged into Nous
    # or on any portal hiccup.
    try:
        from agent.account_usage import nous_credits_lines

        credits = nous_credits_lines()
        if credits:
            usage["credits_lines"] = credits
    except Exception:
        pass
    return _ok(rid, usage)


@method("session.context_breakdown")
def _(rid, params: dict) -> dict:
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    agent = session.get("agent")
    if agent is None:
        usage = _get_usage(None)
        return _ok(
            rid,
            {
                "categories": [],
                "context_max": usage.get("context_max", 0) or 0,
                "context_percent": usage.get("context_percent", 0) or 0,
                "context_used": usage.get("context_used", 0) or 0,
                "estimated_total": 0,
                "model": "",
            },
        )
    with session["history_lock"]:
        history = list(session.get("history", []))
    try:
        from agent.context_breakdown import compute_session_context_breakdown

        payload = compute_session_context_breakdown(agent, history)
    except Exception as exc:
        return _err(rid, 5000, f"Could not compute context breakdown: {exc}")
    return _ok(rid, payload)


def _pet_frame_counts(spritesheet) -> dict:
    """Real (padding-trimmed) frame count per state, for the desktop canvas.

    Fail-open: a decode hiccup returns ``{}`` and the canvas falls back to its
    static ``framesPerState`` rather than breaking the (cosmetic) pet.
    """
    try:
        from agent.pet import render

        return render.state_frame_counts(str(spritesheet))
    except Exception:  # noqa: BLE001 - cosmetic, never break the surface
        return {}


_pet_payload_cache_lock = threading.Lock()
_pet_payload_cache: dict[tuple, dict] = {}


def _pet_sheet_revision(spritesheet) -> str:
    """Stable revision id for one spritesheet file."""
    try:
        stat = spritesheet.stat()
        return f"{stat.st_mtime_ns}:{stat.st_size}"
    except Exception:  # noqa: BLE001 - cosmetic, never break the surface
        return "0:0"


def _pet_payload_cache_key(pet, *, scale: float) -> tuple | None:
    """Cache key for the expensive sprite payload build."""
    try:
        stat = pet.spritesheet.stat()
    except Exception:  # noqa: BLE001
        return None
    return (
        str(pet.spritesheet),
        stat.st_mtime_ns,
        stat.st_size,
        pet.slug,
        pet.display_name,
        round(scale, 4),
    )


def _clone_pet_payload(payload: dict) -> dict:
    """Shallow-clone cached payloads so callers can't mutate shared state."""
    out = dict(payload)
    if isinstance(payload.get("framesByState"), dict):
        out["framesByState"] = dict(payload["framesByState"])
    if isinstance(payload.get("framesByRow"), dict):
        out["framesByRow"] = dict(payload["framesByRow"])
    if isinstance(payload.get("stateRows"), list):
        out["stateRows"] = list(payload["stateRows"])
    return out


def _pet_row_frame_counts(spritesheet) -> dict:
    """Real frame count per concrete spritesheet row name."""
    try:
        from PIL import Image

        from agent.pet import constants, render

        with Image.open(spritesheet) as opened:
            image = opened.convert("RGBA")
        cols = max(1, image.width // constants.FRAME_W)
        row_count = max(1, image.height // constants.FRAME_H)
        rows = constants.state_rows_for_grid(row_count)
        out: dict[str, int] = {}
        for row_idx, name in enumerate(rows[:row_count]):
            top = row_idx * constants.FRAME_H
            count = 0
            for col in range(cols):
                left = col * constants.FRAME_W
                frame = image.crop((left, top, left + constants.FRAME_W, top + constants.FRAME_H))
                if render._frame_is_blank(frame):
                    break
                count += 1
            out[name] = count
        return out
    except Exception:  # noqa: BLE001 - cosmetic, never break the surface
        return {}


def _pet_config_scale() -> float:
    """Configured ``display.pet.scale`` (or the engine default), never raises."""
    from agent.pet import constants

    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        display = cfg.get("display", {}) if isinstance(cfg.get("display"), dict) else {}
        pet_cfg = display.get("pet", {}) if isinstance(display.get("pet"), dict) else {}
        return float(pet_cfg.get("scale", constants.DEFAULT_SCALE) or constants.DEFAULT_SCALE)
    except Exception:  # noqa: BLE001
        return constants.DEFAULT_SCALE


def _pet_sprite_payload(pet, *, scale: float) -> dict:
    """Build the renderer payload (spritesheet bytes + geometry) for *pet*.

    Shared by ``pet.info`` (the active mascot) and ``pet.hatch`` (the unadopted
    preview) so both feed the desktop canvas / TUI from one shape.
    """
    import base64

    from agent.pet import constants

    cache_key = _pet_payload_cache_key(pet, scale=scale)
    if cache_key is not None:
        with _pet_payload_cache_lock:
            cached = _pet_payload_cache.get(cache_key)
        if cached is not None:
            return _clone_pet_payload(cached)

    raw = pet.spritesheet.read_bytes()
    suffix = pet.spritesheet.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/webp"
    payload = {
        "slug": pet.slug,
        "displayName": pet.display_name,
        "mime": mime,
        "spritesheetBase64": base64.standard_b64encode(raw).decode("ascii"),
        "spritesheetRevision": _pet_sheet_revision(pet.spritesheet),
        "frameW": constants.FRAME_W,
        "frameH": constants.FRAME_H,
        "framesPerState": constants.FRAMES_PER_STATE,
        "framesByState": _pet_frame_counts(pet.spritesheet),
        "framesByRow": _pet_row_frame_counts(pet.spritesheet),
        "loopMs": constants.LOOP_MS,
        "scale": scale,
        "stateRows": _pet_state_rows(pet.spritesheet),
    }
    if cache_key is not None:
        with _pet_payload_cache_lock:
            _pet_payload_cache[cache_key] = payload
            while len(_pet_payload_cache) > 8:
                _pet_payload_cache.pop(next(iter(_pet_payload_cache)))
    return _clone_pet_payload(payload)


def _pet_active_selection():
    """Resolve configured active pet + scale from config."""
    from agent.pet import constants, store

    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        display = cfg.get("display", {}) if isinstance(cfg.get("display"), dict) else {}
        pet_cfg = display.get("pet", {}) if isinstance(display.get("pet"), dict) else {}
    except Exception:
        pet_cfg = {}

    enabled = bool(pet_cfg.get("enabled"))
    configured_slug = str(pet_cfg.get("slug", "") or "")
    pet = store.resolve_active_pet(configured_slug) if enabled else None
    scale = float(pet_cfg.get("scale", constants.DEFAULT_SCALE) or constants.DEFAULT_SCALE)
    return enabled, pet, scale


def _pet_state_rows(spritesheet) -> list[str]:
    """Row taxonomy for the concrete active pet sheet.

    Hermes has to support both the legacy 8-row petdex atlas and the current
    Codex/petdex 9-row atlas. The desktop canvas gets this list and indexes it
    with the same `PetState` names the Python renderer uses.
    """
    try:
        from PIL import Image

        from agent.pet import constants

        with Image.open(spritesheet) as image:
            row_count = max(1, image.height // constants.FRAME_H)
        return list(constants.state_rows_for_grid(row_count))
    except Exception:  # noqa: BLE001 - cosmetic, never break the surface
        from agent.pet import constants

        return list(constants.STATE_ROWS)


@method("pet.info")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Return the active petdex pet for surfaces that render sprites.

    Shared by the desktop (canvas) and the TUI (half-block). Carries the
    spritesheet bytes (base64) plus the engine's frame geometry + state-row
    taxonomy so the renderer is a thin, framework-native consumer. The
    activity→state decision is mirrored from ``agent.pet.state`` client-side.

    Agent-independent (reads config + disk), so it works on any session and
    before the agent finishes building. Fail-open: returns ``enabled=False``
    on any error rather than erroring the surface.
    """
    try:
        enabled, pet, scale = _pet_active_selection()

        if not enabled or pet is None or not pet.exists:
            return _ok(rid, {"enabled": False})

        return _ok(rid, {"enabled": True, **_pet_sprite_payload(pet, scale=scale)})
    except Exception as exc:  # noqa: BLE001 - cosmetic, never break the surface
        logger.debug("pet.info failed: %s", exc)
        return _ok(rid, {"enabled": False})


@method("pet.info.meta")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Cheap active-pet metadata used to avoid full payload refreshes."""
    try:
        enabled, pet, scale = _pet_active_selection()
        if not enabled or pet is None or not pet.exists:
            return _ok(rid, {"enabled": False})
        return _ok(
            rid,
            {
                "enabled": True,
                "slug": pet.slug,
                "displayName": pet.display_name,
                "scale": scale,
                "spritesheetRevision": _pet_sheet_revision(pet.spritesheet),
            },
        )
    except Exception as exc:  # noqa: BLE001 - cosmetic, never break the surface
        logger.debug("pet.info.meta failed: %s", exc)
        return _ok(rid, {"enabled": False})


@method("pet.cells")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Return half-block cell frames for one pet state (TUI renderer).

    The TUI can't draw a canvas, so the engine downsamples the spritesheet to
    a grid of half-block cells and the Ink side paints them with native color
    props. Each cell is ``[tr,tg,tb,ta, br,bg,bb,ba]`` (top + bottom pixel).

    Params: ``state`` (idle/run/review/failed/wave/jump), ``cols`` (width).
    Fail-open: ``enabled=False`` on any problem.
    """
    try:
        from agent.pet import constants, render, store
        from agent.pet.render import PetRenderer

        try:
            from hermes_cli.config import load_config

            cfg = load_config()
            display = cfg.get("display", {}) if isinstance(cfg.get("display"), dict) else {}
            pet_cfg = display.get("pet", {}) if isinstance(display.get("pet"), dict) else {}
        except Exception:
            pet_cfg = {}

        if not bool(pet_cfg.get("enabled")):
            return _ok(rid, {"enabled": False})

        pet = store.resolve_active_pet(str(pet_cfg.get("slug", "") or ""))
        if pet is None or not pet.exists:
            return _ok(rid, {"enabled": False})

        state = str(params.get("state") or constants.PetState.IDLE.value)
        scale = float(pet_cfg.get("scale", constants.DEFAULT_SCALE) or constants.DEFAULT_SCALE)
        cols = int(params.get("cols") or 0) or constants.resolve_cols(scale, pet_cfg.get("unicode_cols", 0))

        # Graphics path: when the TUI is attached to a real TTY (``graphics``)
        # and the terminal speaks the kitty protocol, return a Unicode-
        # placeholder payload for a crisp image instead of half-blocks. Env
        # detection (KITTY_WINDOW_ID / TERM / TERM_PROGRAM) is shared with the
        # Ink process since it spawns us; the dashboard PTY (xterm.js) has no
        # such env, so it falls through to half-blocks automatically. Only
        # kitty is grid-safe in Ink — iTerm/sixel stay on the fallback.
        if params.get("graphics"):
            configured = str(pet_cfg.get("render_mode", "auto") or "auto").lower()
            gmode = render.detect_terminal_graphics() if configured in ("", "auto") else configured
            if gmode == "kitty":
                image_id = render.kitty_image_id(pet.slug)
                # kitty sizes from scaled pixels (_cell_box), so unicode_cols is moot here.
                payload = PetRenderer(
                    str(pet.spritesheet), mode="kitty", scale=scale
                ).kitty_payload(state, image_id=image_id)
                if payload:
                    kcount = len(payload["frames"]) or 1
                    return _ok(
                        rid,
                        {
                            "enabled": True,
                            "slug": pet.slug,
                            "displayName": pet.display_name,
                            "state": state,
                            "graphics": "kitty",
                            "imageId": image_id,
                            "color": render.kitty_color_hex(image_id),
                            "cols": payload["cols"],
                            "rows": payload["rows"],
                            "placeholder": payload["placeholder"],
                            "frames": payload["frames"],
                            "frameMs": constants.LOOP_MS / max(1, kcount),
                            "scale": scale,
                        },
                    )

        renderer = PetRenderer(
            str(pet.spritesheet),
            mode="unicode",
            scale=scale,
            unicode_cols=cols,
        )
        count = renderer.frame_count(state) or 1
        frames = []
        for i in range(count):
            grid = renderer.cells(state, i, cols=cols)
            frames.append(
                [[[*top, *bottom] for (top, bottom) in row] for row in grid]
            )

        return _ok(
            rid,
            {
                "enabled": True,
                "slug": pet.slug,
                "displayName": pet.display_name,
                "state": state,
                "cols": cols,
                "frameMs": constants.LOOP_MS / max(1, count),
                "frames": frames,
                "scale": scale,
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("pet.cells failed: %s", exc)
        return _ok(rid, {"enabled": False})


@method("pet.gallery")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """List adoptable pets for the desktop appearance picker.

    Returns the petdex gallery merged with local install state plus the
    current config (active slug + enabled). Agent-independent. Fail-open:
    returns whatever is installed locally if the gallery can't be reached, so
    the picker still works offline.

    Param ``localOnly`` (bool): skip the remote petdex manifest fetch and return
    only locally-installed pets. The desktop loads this first so the user's own
    pets render instantly instead of waiting on the (possibly slow) manifest.
    """
    local_only = bool(params.get("localOnly"))
    try:
        from agent.pet import store

        try:
            from hermes_cli.config import load_config

            cfg = load_config()
            display = cfg.get("display", {}) if isinstance(cfg.get("display"), dict) else {}
            pet_cfg = display.get("pet", {}) if isinstance(display.get("pet"), dict) else {}
        except Exception:
            pet_cfg = {}

        installed = {p.slug: p for p in store.installed_pets()}

        gallery: list[dict] = []
        seen: set[str] = set()
        try:
            from agent.pet.manifest import fetch_manifest, prefetch

            # Local-only: skip the network entirely, but kick off a background
            # warm so the follow-up full request usually hits a cached manifest.
            if local_only:
                prefetch()

            for entry in [] if local_only else fetch_manifest():
                seen.add(entry.slug)
                gallery.append(
                    {
                        "slug": entry.slug,
                        "displayName": entry.display_name,
                        "installed": entry.slug in installed,
                        "spritesheetUrl": entry.spritesheet_url,
                        # petdex exposes no popularity metric; "curated" (its
                        # hand-picked/official set, identified by the asset path)
                        # is the closest signal, so the picker can surface it first.
                        "curated": "/curated/" in entry.spritesheet_url,
                        "generated": entry.slug in installed and installed[entry.slug].generated,
                    }
                )
        except Exception as exc:  # noqa: BLE001 - offline: fall back to installed
            logger.debug("pet.gallery manifest fetch failed: %s", exc)

        # Always include locally-installed pets even if the gallery is unreachable.
        for slug, pet in installed.items():
            if slug not in seen:
                gallery.append(
                    {
                        "slug": slug,
                        "displayName": pet.display_name,
                        "installed": True,
                        "spritesheetUrl": "",
                        "generated": pet.generated,
                    }
                )

        return _ok(
            rid,
            {
                "enabled": bool(pet_cfg.get("enabled")),
                "active": str(pet_cfg.get("slug", "") or ""),
                "pets": gallery,
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("pet.gallery failed: %s", exc)
        return _ok(rid, {"enabled": False, "active": "", "pets": []})


@method("pet.select")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Adopt a pet from the desktop picker: install (if needed) + activate.

    Params: ``slug`` (required). Writes ``display.pet.*`` to config and returns
    ``{ok, slug, displayName}``. The surface re-pulls ``pet.info`` to render it.
    """
    slug = str(params.get("slug") or "").strip()
    if not slug:
        return _err(rid, 4004, "missing slug")
    try:
        from agent.pet import store
        from agent.pet.manifest import ManifestError
        from hermes_cli.pets import _set_active

        try:
            pet = store.install_pet(slug)
        except (store.PetStoreError, ManifestError) as exc:
            return _err(rid, 5031, f"could not adopt '{slug}': {exc}")
        _set_active(slug)
        return _ok(rid, {"ok": True, "slug": slug, "displayName": pet.display_name})
    except Exception as exc:  # noqa: BLE001
        logger.debug("pet.select failed: %s", exc)
        return _err(rid, 5031, f"pet.select failed: {exc}")


@method("pet.remove")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Uninstall a pet from the desktop picker (delete its on-disk directory).

    Params: ``slug`` (required). If the removed pet was the active one, the
    display is turned off so nothing tries to render a now-missing sprite.
    Returns ``{ok, slug}`` where ``ok`` reflects whether a directory was deleted.
    """
    slug = str(params.get("slug") or "").strip()
    if not slug:
        return _err(rid, 4004, "missing slug")
    try:
        from agent.pet import store
        from hermes_cli.pets import _clear_active_if

        removed = store.remove_pet(slug)

        # If that was the active pet, stop surfaces pointing at a deleted sprite.
        try:
            _clear_active_if(slug)
        except Exception as exc:  # noqa: BLE001 - removal already succeeded
            logger.debug("pet.remove config update failed: %s", exc)

        return _ok(rid, {"ok": removed, "slug": slug})
    except Exception as exc:  # noqa: BLE001
        logger.debug("pet.remove failed: %s", exc)
        return _err(rid, 5031, f"pet.remove failed: {exc}")


@method("pet.export")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Export an installed pet as a re-importable ``.zip`` (pet.json + sprite).

    Params: ``slug`` (required). Returns ``{ok, filename, zipBase64}`` — the
    client decodes the base64 and saves it. Heavy-ish (reads + zips files) but
    small; runs inline.
    """
    slug = str(params.get("slug") or "").strip()
    if not slug:
        return _err(rid, 4004, "missing slug")
    try:
        import base64

        from agent.pet import store

        filename, data = store.export_pet(slug)
        return _ok(
            rid,
            {"ok": True, "filename": filename, "zipBase64": base64.standard_b64encode(data).decode("ascii")},
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("pet.export failed: %s", exc)
        return _err(rid, 5031, f"pet.export failed: {exc}")


@method("pet.rename")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Rename an installed pet's display name + realign its slug/dir.

    Params: ``slug`` + ``name`` (both required). Lets the generate flow hatch
    with a provisional name and apply the user's chosen name at adopt time.
    Returns ``{ok, slug, displayName}`` with the (possibly new) slug.
    """
    slug = str(params.get("slug") or "").strip()
    name = str(params.get("name") or "").strip()
    if not slug:
        return _err(rid, 4004, "missing slug")
    if not name:
        return _err(rid, 4004, "missing name")
    try:
        from agent.pet import store

        new_slug = store.rename_pet(slug, name)
        if not new_slug:
            return _err(rid, 5031, "pet.rename failed")

        # The dir may have moved; if the renamed pet was active, follow the slug
        # in config so surfaces don't point at the old (now-missing) directory.
        if new_slug != slug:
            try:
                from hermes_cli.pets import _rename_active_if

                _rename_active_if(slug, new_slug)
            except Exception as exc:  # noqa: BLE001 - rename already succeeded
                logger.debug("pet.rename config update failed: %s", exc)

        return _ok(rid, {"ok": True, "slug": new_slug, "displayName": name})
    except Exception as exc:  # noqa: BLE001
        logger.debug("pet.rename failed: %s", exc)
        return _err(rid, 5031, f"pet.rename failed: {exc}")


@method("pet.thumb")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Return a small idle-frame PNG (data URI) for one pet — the picker preview.

    Cropped + cached server-side so the renderer gets a same-origin data URL
    instead of a CDN ``<img>`` (which the desktop CSP / R2 hotlink rules break).
    Params: ``slug`` (required), ``url`` (optional petdex spritesheet URL used
    only for not-yet-installed pets). Fail-open: ``{ok: false}`` with no error.
    """
    slug = str(params.get("slug") or "").strip()
    if not slug:
        return _err(rid, 4004, "missing slug")
    try:
        import base64

        from agent.pet import store

        data = store.thumbnail_png(slug, source_url=str(params.get("url") or ""))
        if not data:
            return _ok(rid, {"ok": False, "slug": slug})

        return _ok(
            rid,
            {
                "ok": True,
                "slug": slug,
                "dataUri": "data:image/png;base64," + base64.standard_b64encode(data).decode("ascii"),
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("pet.thumb failed: %s", exc)
        return _ok(rid, {"ok": False, "slug": slug})


@method("pet.disable")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Turn the pet off from the desktop picker (``display.pet.enabled=false``)."""
    try:
        from hermes_cli.pets import _set_enabled

        _set_enabled(False)
        return _ok(rid, {"ok": True})
    except Exception as exc:  # noqa: BLE001
        logger.debug("pet.disable failed: %s", exc)
        return _err(rid, 5031, f"pet.disable failed: {exc}")


@method("pet.scale")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Persist ``display.pet.scale`` from the desktop slider. Params: ``scale``.

    Clamped to the engine bounds. The renderer updates its own ``$petInfo`` for
    instant feedback; this just makes the change durable + visible to the other
    terminal surfaces on their next read.
    """
    try:
        from hermes_cli.pets import set_pet_scale

        scale, err = set_pet_scale(params.get("scale"))
        if err:
            return _err(rid, 4004, err)
        return _ok(rid, {"ok": True, "scale": scale})
    except Exception as exc:  # noqa: BLE001
        logger.debug("pet.scale failed: %s", exc)
        return _err(rid, 5031, f"pet.scale failed: {exc}")


def _pet_gen_root():
    """Profile-scoped staging dir for in-progress generation drafts."""
    from hermes_constants import get_hermes_home

    root = get_hermes_home() / "cache" / "pet-gen"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _pet_gen_sweep(root, *, max_age_s: float = 3600.0) -> None:
    """Drop stale draft staging dirs so cache never grows unbounded."""
    import shutil
    import time

    try:
        now = time.time()
        for child in root.iterdir():
            if child.is_dir() and now - child.stat().st_mtime > max_age_s:
                shutil.rmtree(child, ignore_errors=True)
    except Exception as exc:  # noqa: BLE001 - cleanup is best-effort
        logger.debug("pet-gen sweep failed: %s", exc)


def _pet_png_data_uri(path, *, max_px: int = 160) -> str:
    """Downscaled PNG data URI for a draft image (small preview payload)."""
    import base64
    import io

    from PIL import Image

    with Image.open(path) as opened:
        img = opened.convert("RGBA")
    img.thumbnail((max_px, max_px), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.standard_b64encode(buf.getvalue()).decode("ascii")


# Cooperative cancellation for the heavy pet generation paths. The client's Stop
# aborts its RPC immediately, but the worker-pool generation keeps running unless
# told to stop — pet.cancel flips a token's flag, which generate_base_drafts /
# hatch_pet poll between provider calls to skip work they haven't started.
_pet_cancel_lock = threading.Lock()
_pet_cancelled: set[str] = set()
_PET_REFERENCE_MIME_EXT = {
    "png": "png",
    "jpeg": "jpg",
    "jpg": "jpg",
    "webp": "webp",
    "gif": "gif",
}
try:
    _PET_REFERENCE_MAX_BYTES = max(
        1,
        int(os.environ.get("HERMES_PET_REFERENCE_MAX_BYTES") or str(16 * 1024 * 1024)),
    )
except (TypeError, ValueError):
    _PET_REFERENCE_MAX_BYTES = 16 * 1024 * 1024


def _pet_reference_images_from_data_url(ref_raw: str, stage) -> list:
    """Decode + validate a reference-image data URL into the stage dir."""
    import base64
    import binascii
    import re as _re

    match = _re.match(r"^data:image/([a-zA-Z0-9.+-]+);base64,(.*)$", ref_raw, _re.DOTALL)
    if not match:
        raise ValueError("invalid reference image format")

    mime = match.group(1).lower()
    ext = _PET_REFERENCE_MIME_EXT.get(mime)
    if ext is None:
        raise ValueError("unsupported reference image type")

    payload = "".join(match.group(2).split())
    approx = (len(payload) * 3) // 4
    if approx > _PET_REFERENCE_MAX_BYTES:
        raise ValueError("reference image too large")

    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("invalid reference image data") from exc

    if len(raw) > _PET_REFERENCE_MAX_BYTES:
        raise ValueError("reference image too large")

    ref_path = stage / f"reference.{ext}"
    ref_path.write_bytes(raw)
    return [ref_path]


def _pet_cancel_arm(token: str) -> None:
    """Clear a stale cancel flag at the start of a generate/hatch run."""
    with _pet_cancel_lock:
        _pet_cancelled.discard(token)


def _pet_cancel_request(token: str) -> None:
    with _pet_cancel_lock:
        _pet_cancelled.add(token)


def _pet_is_cancelled(token: str) -> bool:
    with _pet_cancel_lock:
        return token in _pet_cancelled


def _pet_cancel_release(token: str) -> None:
    with _pet_cancel_lock:
        _pet_cancelled.discard(token)


@method("pet.cancel")
def _(rid, params: dict) -> dict:
    """Signal an in-flight ``pet.generate``/``pet.hatch`` (by token) to stop.

    Best-effort + idempotent: cancelling an unknown/finished token is a no-op.
    Stays off the worker pool so it lands while a heavy generation is occupying
    it. Returns ``{ok: True}``.
    """
    token = str(params.get("token") or "").strip()
    if token:
        _pet_cancel_request(token)
    return _ok(rid, {"ok": True})


@method("pet.generate.status")
def _(rid, params: dict) -> dict:
    """Whether pet generation is possible right now.

    True only when a reference-capable image backend (Nous Portal / OpenRouter /
    OpenAI gpt-image) is configured — the desktop checks this on open so it can
    offer setup instead of a dead prompt. Cheap (config + plugin discovery).
    """
    try:
        from agent.pet.generate.imagegen import (
            GenerationError,
            list_sprite_providers,
            resolve_provider,
        )

        try:
            resolve_provider(require_references=True)
            available = True
        except GenerationError:
            available = False
        try:
            providers = list_sprite_providers()
        except Exception as exc:  # noqa: BLE001 - picker is best-effort
            logger.debug("pet provider list failed: %s", exc)
            providers = []
        return _ok(rid, {"available": available, "providers": providers})
    except Exception as exc:  # noqa: BLE001 - never break the surface
        logger.debug("pet.generate.status failed: %s", exc)
        return _ok(rid, {"available": False, "providers": []})


@method("pet.generate")
def _(rid, params: dict) -> dict:
    """Generate candidate base looks for a new pet (the draft/variant step).

    Params: ``prompt`` (required unless ``referenceImage`` is given), ``count``
    (default 4), ``style`` (default ``auto``), ``referenceImage`` (optional data
    URL — a user photo/reference every draft is grounded on, e.g. to make *their*
    pet). Returns ``{ok, token, drafts:[{index, dataUri}]}`` — the token keys the
    staged base images for a later ``pet.hatch``. Heavy (network): worker pool.
    """
    prompt = str(params.get("prompt") or "").strip()
    ref_raw = str(params.get("referenceImage") or "").strip()
    if not prompt and not ref_raw:
        return _err(rid, 4004, "missing prompt")
    try:
        count = max(1, min(4, int(params.get("count") or 4)))
    except (TypeError, ValueError):
        count = 4
    style = str(params.get("style") or "auto").strip() or "auto"

    try:
        import shutil
        import uuid

        from agent.pet.generate import generate_base_drafts
        from agent.pet.generate.imagegen import GenerationError, resolve_provider

        root = _pet_gen_root()
        _pet_gen_sweep(root)

        # Token up front so each draft can be staged + streamed the moment it
        # lands, instead of the user staring at a blank grid until all N finish.
        token = uuid.uuid4().hex[:12]
        _pet_cancel_arm(token)
        stage = root / token
        stage.mkdir(parents=True, exist_ok=True)

        reference_images = None
        if ref_raw:
            try:
                reference_images = _pet_reference_images_from_data_url(ref_raw, stage)
            except ValueError as exc:
                _pet_cancel_release(token)
                return _err(rid, 4004, str(exc))

        # Optional desktop picker override: resolve the chosen provider up front so
        # a bad/uncredentialed pick fails fast instead of mid-fan-out.
        provider_name = str(params.get("provider") or "").strip()
        sprite = None
        if provider_name:
            try:
                sprite = resolve_provider(require_references=bool(reference_images), prefer=provider_name)
            except GenerationError as exc:
                _pet_cancel_release(token)
                return _err(rid, 5031, str(exc))

        concept = prompt or "a pet based on the reference image"
        out: list[dict] = []

        # Hand the token to the client up front (token-only init event) so a Stop
        # fired before the first draft lands can still target this run.
        try:
            _emit("pet.generate.progress", "", {"token": token, "count": count})
        except Exception as exc:  # noqa: BLE001 - streaming is best-effort
            logger.debug("pet.generate init emit failed: %s", exc)

        def _on_draft(index: int, src) -> None:
            dest = stage / f"draft-{index}.png"
            try:
                shutil.copyfile(src, dest)
                data_uri = _pet_png_data_uri(dest)
            except Exception as exc:  # noqa: BLE001 - skip a bad draft, keep the rest
                logger.debug("pet.generate draft %d failed: %s", index, exc)
                return
            out.append({"index": index, "dataUri": data_uri})
            # Stream this draft to the client so the grid fills in live. Best-
            # effort: a transport hiccup must not abort the generation itself.
            try:
                _emit(
                    "pet.generate.progress",
                    "",
                    {"token": token, "index": index, "dataUri": data_uri, "count": count},
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("pet.generate progress emit failed: %s", exc)

        try:
            generate_base_drafts(
                concept,
                n=count,
                style=style,
                reference_images=reference_images,
                provider=sprite,
                on_draft=_on_draft,
                is_cancelled=lambda: _pet_is_cancelled(token),
            )
        except GenerationError as exc:
            _pet_cancel_release(token)
            return _err(rid, 5031, str(exc))

        cancelled = _pet_is_cancelled(token)
        _pet_cancel_release(token)
        if cancelled:
            return _err(rid, 5031, "generation cancelled")
        if not out:
            return _err(rid, 5031, "generation produced no usable drafts")
        out.sort(key=lambda d: d["index"])
        return _ok(rid, {"ok": True, "token": token, "drafts": out})
    except Exception as exc:  # noqa: BLE001
        logger.debug("pet.generate failed: %s", exc)
        return _err(rid, 5031, f"pet.generate failed: {exc}")


@method("pet.hatch")
def _(rid, params: dict) -> dict:
    """Turn a chosen base draft into a full pet — installed but NOT yet active.

    Generation is expensive and the result varies, so hatch produces a *preview*
    the surface plays (all frames) before the user commits: the pet is written to
    the store (so it can be rendered + later activated) but the active pet is left
    untouched. Adopt with ``pet.select`` or throw it away with ``pet.remove``.

    Params: ``token`` + ``index`` (from ``pet.generate``), ``name`` (required),
    ``description`` (optional), ``prompt`` (optional concept for row prompts),
    ``style`` (optional). Returns ``{ok, slug, displayName, warnings, pet}`` where
    ``pet`` is the renderer payload. Heavy (network + raster): worker pool.
    """
    token = str(params.get("token") or "").strip()
    # Hatch cancellation rides its own key, not the generation token: hatching a
    # draft mid-generation means pet.generate is still releasing `token`, which
    # would otherwise wipe the arm we set here. Falls back to `token` for clients
    # that don't send one.
    cancel_token = str(params.get("cancelToken") or "").strip() or token
    index = params.get("index", 0)
    name = str(params.get("name") or "").strip()
    if not token:
        return _err(rid, 4004, "missing token")
    if not name:
        return _err(rid, 4004, "missing name")
    try:
        index = int(index)
    except (TypeError, ValueError):
        index = 0

    try:
        from agent.pet import store
        from agent.pet.generate import hatch_pet
        from agent.pet.generate.imagegen import GenerationError, resolve_provider

        base = _pet_gen_root() / token / f"draft-{index}.png"
        if not base.is_file():
            return _err(rid, 4004, "draft expired — generate again")

        # Optional desktop picker override (rows always need reference grounding).
        provider_name = str(params.get("provider") or "").strip()
        sprite = None
        if provider_name:
            try:
                sprite = resolve_provider(require_references=True, prefer=provider_name)
            except GenerationError as exc:
                return _err(rid, 5031, str(exc))

        _pet_cancel_arm(cancel_token)
        slug = store.unique_slug(name)

        def _on_progress(event: str, detail: str) -> None:
            # Row progress is encoded as "<state>:<done>:<total>" so the egg
            # screen can show "Drawing <state>… (n/total)"; other phases
            # (compose, save) pass through as-is. Best-effort streaming.
            payload: dict = {"event": event, "detail": detail}
            if event == "row" and detail.count(":") == 2:
                state, done, total = detail.split(":")
                payload = {"event": "row", "state": state, "done": done, "total": total}
            try:
                _emit("pet.hatch.progress", "", payload)
            except Exception as exc:  # noqa: BLE001
                logger.debug("pet.hatch progress emit failed: %s", exc)

        try:
            result = hatch_pet(
                base_image=base,
                slug=slug,
                display_name=name,
                description=str(params.get("description") or ""),
                concept=str(params.get("prompt") or name),
                style=str(params.get("style") or "auto").strip() or "auto",
                provider=sprite,
                on_progress=_on_progress,
                is_cancelled=lambda: _pet_is_cancelled(cancel_token),
            )
        except GenerationError as exc:
            return _err(rid, 5031, str(exc))
        finally:
            _pet_cancel_release(cancel_token)

        pet = store.load_pet(result.slug)
        payload = _pet_sprite_payload(pet, scale=_pet_config_scale()) if pet else {}
        return _ok(
            rid,
            {
                "ok": True,
                "slug": result.slug,
                "displayName": result.display_name,
                "warnings": result.validation.get("warnings", []),
                "pet": payload,
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("pet.hatch failed: %s", exc)
        return _err(rid, 5031, f"pet.hatch failed: {exc}")


@method("credits.view")
def _(rid, params: dict) -> dict:
    """Structured Nous credit view for the TUI /credits command.

    Account-independent (a portal fetch gated on "a Nous account is logged in"),
    so it works with no live agent / on a resumed session — same as the /usage
    credits block. Returns the surface-agnostic CreditsView fields so the TUI can
    render a clickable top-up <Link>. Fail-open: a portal hiccup or logged-out
    account yields {logged_in: false}, never an error the user has to parse.
    """
    try:
        from agent.account_usage import build_credits_view

        view = build_credits_view()
        return _ok(
            rid,
            {
                "logged_in": bool(view.logged_in),
                "balance_lines": [
                    line for line in view.balance_lines if not line.lstrip().startswith("📈")
                ],
                "identity_line": view.identity_line,
                "topup_url": view.topup_url,
                "depleted": bool(view.depleted),
            },
        )
    except Exception:
        # Fail-open: TUI treats this as "not logged in" and shows the prompt.
        return _ok(rid, {"logged_in": False, "balance_lines": [], "identity_line": None, "topup_url": None, "depleted": False})


# ===========================================================================
# Phase 2b terminal billing RPC methods
# ===========================================================================
#
# These return STRUCTURED success envelopes (result.ok / result.error) rather
# than JSON-RPC-level errors, so the TUI's rpc() promise always resolves and the
# Ink side can branch on the typed billing error code (insufficient_scope,
# rate_limited, no_payment_method, …) to render the right affordance instead of
# landing in a generic catch. The data-building lives in the shared core
# (agent/billing_view.py + hermes_cli/nous_billing.py) — same as /credits.


def _serialize_billing_error(exc) -> dict:
    """Map a BillingError into the result.error envelope the TUI branches on."""
    from hermes_cli.nous_billing import (
        BillingRateLimited,
        BillingScopeRequired,
    )

    kind = "error"
    if isinstance(exc, BillingScopeRequired):
        kind = "insufficient_scope"
    elif isinstance(exc, BillingRateLimited):
        kind = "rate_limited"
    elif getattr(exc, "error", None):
        kind = str(exc.error)
    return {
        "ok": False,
        "error": kind,
        "message": str(exc),
        "portal_url": getattr(exc, "portal_url", None),
        "retry_after": getattr(exc, "retry_after", None),
        "payload": getattr(exc, "payload", {}) or {},
    }


def _serialize_billing_state(state) -> dict:
    """Serialize a BillingState for the wire (Decimals → strings, money-safe)."""
    from agent.billing_view import format_money

    def _s(value):
        return None if value is None else str(value)

    card = None
    if state.card is not None:
        card = {"brand": state.card.brand, "last4": state.card.last4, "masked": state.card.masked}
    monthly_cap = None
    if state.monthly_cap is not None:
        mc = state.monthly_cap
        monthly_cap = {
            "limit_usd": _s(mc.limit_usd),
            "limit_display": format_money(mc.limit_usd),
            "spent_this_month_usd": _s(mc.spent_this_month_usd),
            "spent_display": format_money(mc.spent_this_month_usd),
            "is_default_ceiling": mc.is_default_ceiling,
        }
    auto_reload = None
    if state.auto_reload is not None:
        ar = state.auto_reload
        auto_reload = {
            "enabled": ar.enabled,
            "threshold_usd": _s(ar.threshold_usd),
            "threshold_display": format_money(ar.threshold_usd),
            "reload_to_usd": _s(ar.reload_to_usd),
            "reload_to_display": format_money(ar.reload_to_usd),
        }
    return {
        "ok": True,
        "logged_in": state.logged_in,
        "org_name": state.org_name,
        "org_slug": state.org_slug,
        "role": state.role,
        "is_admin": state.is_admin,
        "can_charge": state.can_charge,
        "balance_usd": _s(state.balance_usd),
        "balance_display": format_money(state.balance_usd),
        "cli_billing_enabled": state.cli_billing_enabled,
        "charge_presets": [_s(p) for p in state.charge_presets],
        "charge_presets_display": [format_money(p) for p in state.charge_presets],
        "min_usd": _s(state.min_usd),
        "max_usd": _s(state.max_usd),
        "card": card,
        "monthly_cap": monthly_cap,
        "auto_reload": auto_reload,
        "portal_url": state.portal_url,
        "error": state.error,
    }


@method("billing.state")
def _(rid, params: dict) -> dict:
    """GET /api/billing/state → serialized BillingState (Screen 1 + 5).

    Fail-open like credits.view: a logged-out / unreachable portal yields
    {ok:true, logged_in:false}. No scope required for this endpoint.
    """
    try:
        from agent.billing_view import build_billing_state

        state = build_billing_state()
        return _ok(rid, _serialize_billing_state(state))
    except Exception:
        return _ok(rid, {"ok": True, "logged_in": False, "error": "could not load billing state"})


@method("billing.charge")
def _(rid, params: dict) -> dict:
    """POST /api/billing/charge → {ok, chargeId} or a typed error envelope.

    params: {amount_usd: str|number, idempotency_key?: str}. If no key is
    supplied, the server-side core mints a fresh one and returns it so the TUI can
    reuse it on retry of the SAME purchase.
    """
    from hermes_cli.nous_billing import BillingError, post_charge
    from agent.billing_view import new_idempotency_key

    amount = params.get("amount_usd")
    if amount is None:
        return _ok(rid, {"ok": False, "error": "invalid_request", "message": "amount_usd is required"})
    key = params.get("idempotency_key") or new_idempotency_key()
    try:
        result = post_charge(amount_usd=amount, idempotency_key=key)
        return _ok(rid, {"ok": True, "charge_id": result.get("chargeId"), "idempotency_key": key})
    except BillingError as exc:
        env = _serialize_billing_error(exc)
        env["idempotency_key"] = key  # so the TUI can reuse on retry
        return _ok(rid, env)
    except Exception as exc:
        return _ok(rid, {"ok": False, "error": "error", "message": str(exc), "idempotency_key": key})


@method("billing.charge_status")
def _(rid, params: dict) -> dict:
    """GET /api/billing/charge/{id} → {ok, status, ...} or typed error.

    The poll. Caller drives the 2s/5-min cadence; this is a single status read.
    """
    from hermes_cli.nous_billing import BillingError, get_charge_status

    charge_id = params.get("charge_id")
    if not charge_id:
        return _ok(rid, {"ok": False, "error": "invalid_charge_id", "message": "charge_id is required"})
    try:
        result = get_charge_status(charge_id)
        return _ok(
            rid,
            {
                "ok": True,
                "status": result.get("status"),
                "amount_usd": result.get("amountUsd"),
                "settled_at": result.get("settledAt"),
                "reason": result.get("reason"),
            },
        )
    except BillingError as exc:
        return _ok(rid, _serialize_billing_error(exc))
    except Exception as exc:
        return _ok(rid, {"ok": False, "error": "error", "message": str(exc)})


@method("billing.auto_reload")
def _(rid, params: dict) -> dict:
    """PATCH /api/billing/auto-top-up → {ok:true} or typed error (Screen 2).

    params: {enabled: bool, threshold: number, top_up_amount: number}.
    """
    from hermes_cli.nous_billing import BillingError, patch_auto_top_up

    try:
        enabled = bool(params.get("enabled"))
        threshold = params.get("threshold")
        top_up_amount = params.get("top_up_amount")
        if threshold is None or top_up_amount is None:
            return _ok(rid, {"ok": False, "error": "invalid_request", "message": "threshold and top_up_amount are required"})
        patch_auto_top_up(enabled=enabled, threshold=threshold, top_up_amount=top_up_amount)
        return _ok(rid, {"ok": True})
    except BillingError as exc:
        return _ok(rid, _serialize_billing_error(exc))
    except Exception as exc:
        return _ok(rid, {"ok": False, "error": "error", "message": str(exc)})


@method("billing.step_up")
def _(rid, params: dict) -> dict:
    """Run the lazy billing:manage step-up device flow → {ok, granted}.

    Triggered by the TUI after a billing call returns error=insufficient_scope.
    Returns granted:false when the server silently downscopes (non-admin / unticked).

    Runs on the thread pool (in _LONG_HANDLERS): the device flow blocks for the
    whole device-code lifetime (minutes), so it must not stall the main stdin loop.
    The verification URL/code reach the TUI via an out-of-band ``billing.step_up.
    verification`` event (a plain print would be dropped by the JSON-RPC stdout
    pipe), and the browser is opened TUI-side via openExternalUrl — never with the
    gateway's headless webbrowser.open (hence open_browser=False).
    """
    sid = params.get("session_id") or ""
    try:
        from hermes_cli.auth import step_up_nous_billing_scope

        def _on_verification(url: str, code: str) -> None:
            _emit(
                "billing.step_up.verification",
                sid,
                {"verification_url": url, "user_code": code},
            )

        granted = step_up_nous_billing_scope(
            open_browser=False, on_verification=_on_verification
        )
        return _ok(rid, {"ok": True, "granted": bool(granted)})
    except Exception as exc:
        return _ok(rid, {"ok": False, "error": "error", "message": str(exc), "granted": False})


@method("session.status")
def _(rid, params: dict) -> dict:
    session, err = _sess_nowait(params, rid)
    if err:
        return err

    from hermes_constants import display_hermes_home

    key = session.get("session_key") or params.get("session_id") or ""
    agent = session.get("agent")
    meta = {}
    db = _get_db()
    if db and key:
        try:
            meta = db.get_session(key) or {}
        except Exception:
            meta = {}

    def _dt(value, fallback: datetime | None = None) -> datetime:
        if value:
            try:
                return datetime.fromtimestamp(float(value))
            except Exception:
                pass
        return fallback or datetime.now()

    created = _dt(meta.get("started_at"))
    updated = created
    for field in ("updated_at", "last_updated_at", "last_activity_at"):
        if meta.get(field):
            updated = _dt(meta.get(field), created)
            break

    usage = _get_usage(agent) if agent is not None else {}
    provider = getattr(agent, "provider", None) or "unknown"
    model = getattr(agent, "model", None) or "(unknown)"
    lines = [
        "Hermes TUI Status",
        "",
        f"Session ID: {key}",
        f"Path: {display_hermes_home()}",
    ]
    title = (meta.get("title") or "").strip()
    if title:
        lines.append(f"Title: {title}")
    lines.extend([
        f"Model: {model} ({provider})",
        f"Created: {created.strftime('%Y-%m-%d %H:%M')}",
        f"Last Activity: {updated.strftime('%Y-%m-%d %H:%M')}",
        f"Tokens: {int(usage.get('total') or 0):,}",
        f"Agent Running: {'Yes' if session.get('running') else 'No'}",
    ])
    return _ok(rid, {"output": "\n".join(lines)})


@method("session.history")
def _(rid, params: dict) -> dict:
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    history = list(session.get("history", []))
    db = _get_db()
    if db is not None and session.get("session_key"):
        try:
            history = db.get_messages_as_conversation(
                session["session_key"], include_ancestors=True
            )
        except Exception:
            pass
    return _ok(
        rid,
        {
            "count": len(history),
            "messages": _history_to_messages(history),
        },
    )


@method("session.undo")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    # Reject during an in-flight turn.  If we mutated history while
    # the agent thread is running, prompt.submit's post-run history
    # write would either clobber the undo (version matches) or
    # silently drop the agent's output (version mismatch, see below).
    # Neither is what the user wants — make them /interrupt first.
    if session.get("running"):
        return _err(
            rid, 4009, "session busy — /interrupt the current turn before /undo"
        )
    removed = 0
    with session["history_lock"]:
        history = session.get("history", [])
        while history and history[-1].get("role") in {"assistant", "tool"}:
            history.pop()
            removed += 1
        if history and history[-1].get("role") == "user":
            history.pop()
            removed += 1
        if removed:
            session["history_version"] = int(session.get("history_version", 0)) + 1
    return _ok(rid, {"removed": removed})


@method("session.compress")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    if session.get("running"):
        return _err(
            rid, 4009, "session busy — /interrupt the current turn before /compress"
        )
    sid = params.get("session_id", "")
    focus_topic = str(params.get("focus_topic", "") or "").strip()
    try:
        from agent.manual_compression_feedback import summarize_manual_compression
        from agent.model_metadata import estimate_request_tokens_rough

        with session["history_lock"]:
            before_messages = list(session.get("history", []))
            history_version = int(session.get("history_version", 0))
        before_count = len(before_messages)
        _agent = session["agent"]
        _sys_prompt = getattr(_agent, "_cached_system_prompt", "") or ""
        _tools = getattr(_agent, "tools", None) or None
        before_tokens = (
            estimate_request_tokens_rough(
                before_messages, system_prompt=_sys_prompt, tools=_tools
            )
            if before_count
            else 0
        )

        if before_count >= 4:
            focus_suffix = f', focus: "{focus_topic}"' if focus_topic else ""
            _status_update(
                sid,
                "compressing",
                f"⠋ compressing {before_count} messages "
                f"(~{before_tokens:,} tok){focus_suffix}…",
            )

        try:
            removed, usage = _compress_session_history(
                session,
                focus_topic,
                approx_tokens=before_tokens,
                before_messages=before_messages,
                history_version=history_version,
            )
            with session["history_lock"]:
                messages = list(session.get("history", []))
            after_count = len(messages)
            # Re-read system prompt + tools after compression — _compress_context
            # may have rebuilt the system prompt (_cached_system_prompt=None).
            _sys_prompt_after = (
                getattr(_agent, "_cached_system_prompt", "") or _sys_prompt
            )
            _tools_after = getattr(_agent, "tools", None) or _tools
            after_tokens = (
                estimate_request_tokens_rough(
                    messages,
                    system_prompt=_sys_prompt_after,
                    tools=_tools_after,
                )
                if after_count
                else 0
            )
            agent = session["agent"]
            _sync_session_key_after_compress(sid, session)
            summary = summarize_manual_compression(
                before_messages, messages, before_tokens, after_tokens
            )
            info = _session_info(agent, session)
            _emit("session.info", sid, info)
            return _ok(
                rid,
                {
                    "status": "compressed",
                    "removed": removed,
                    "before_messages": before_count,
                    "after_messages": after_count,
                    "before_tokens": before_tokens,
                    "after_tokens": after_tokens,
                    "summary": summary,
                    "usage": usage,
                    "info": info,
                    "messages": messages,
                },
            )
        finally:
            # Always clear the pinned compressing status so the bar
            # reverts to neutral whether compaction succeeded, was a
            # no-op, or raised.
            _status_update(sid, "ready")
    except Exception as e:
        return _err(rid, 5005, str(e))


@method("session.save")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err

    agent = session["agent"]
    # Mirror the classic CLI /save: snapshot under the Hermes profile home
    # (~/.hermes/sessions/saved/) rather than the project/workspace CWD, and
    # include the system prompt so the export matches the dashboard save.
    saved_dir = get_hermes_home() / "sessions" / "saved"
    try:
        saved_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return _err(rid, 5011, f"failed to create save directory {saved_dir}: {e}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = saved_dir / f"hermes_conversation_{timestamp}.json"

    with session["history_lock"]:
        messages = list(session.get("history", []))

    session_id = getattr(agent, "session_id", None) or session.get("session_key") or ""
    # Prefer the agent's session_start datetime (matches the classic CLI export);
    # fall back to the gateway session's created_at timestamp.
    agent_start = getattr(agent, "session_start", None)
    if isinstance(agent_start, datetime):
        session_start = agent_start.isoformat()
    else:
        created_at = session.get("created_at")
        session_start = (
            datetime.fromtimestamp(created_at).isoformat()
            if isinstance(created_at, (int, float))
            else ""
        )

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": getattr(agent, "model", ""),
                    "session_id": session_id,
                    "session_start": session_start,
                    "system_prompt": getattr(agent, "_cached_system_prompt", "") or "",
                    "messages": messages,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        return _ok(rid, {"file": str(path)})
    except Exception as e:
        return _err(rid, 5011, str(e))


@method("session.close")
def _(rid, params: dict) -> dict:
    sid = params.get("session_id", "")
    # Serialize against the WS-orphan reaper (which also pops under
    # _session_resume_lock) so a disconnect-reap and an explicit close can't
    # both tear the same session down. _close_session_by_id is the single
    # idempotent teardown path (pop + _teardown_session) and returns False
    # when the session is already gone.
    with _session_resume_lock:
        return _ok(rid, {"closed": _close_session_by_id(sid, end_reason="tui_close")})


@method("session.branch")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    db = _get_db()
    if db is None:
        return _db_unavailable_error(rid, code=5008)
    old_key = session["session_key"]
    with session["history_lock"]:
        history = [dict(msg) for msg in session.get("history", [])]
    if not history:
        return _err(rid, 4008, "nothing to branch — send a message first")
    new_key = _new_session_key()
    new_sid = uuid.uuid4().hex[:8]
    source = _session_source(session)
    lease, limit_message = _claim_active_session_slot(
        new_key, live_session_id=new_sid, surface=source
    )
    if limit_message is not None:
        return _err(rid, 4090, limit_message)
    branch_name = params.get("name", "")
    try:
        if branch_name:
            title = branch_name
        else:
            current = db.get_session_title(old_key) or "branch"
            title = (
                db.get_next_title_in_lineage(current)
                if hasattr(db, "get_next_title_in_lineage")
                else f"{current} (branch)"
            )
        db.create_session(
            new_key,
            source=source,
            model=_resolve_model(),
            # Stable _branched_from marker so list_sessions_rich() keeps the
            # branch visible in /resume and /sessions. The TUI branch leaves
            # the parent live (no end_reason='branched'), so the legacy
            # end_reason heuristic never matches it — the marker is the only
            # thing that surfaces TUI branches. See issue #20856.
            model_config={"_branched_from": old_key},
            parent_session_id=old_key,
            cwd=_session_cwd(session),
        )
        for msg in history:
            db.append_message(
                session_id=new_key,
                role=msg.get("role", "user"),
                content=msg.get("content"),
            )
        db.set_session_title(new_key, title)
    except Exception as e:
        if lease is not None:
            lease.release()
        return _err(rid, 5008, f"branch failed: {e}")
    try:
        tokens = _set_session_context(new_key)
        try:
            agent = _make_agent(
                new_sid,
                new_key,
                session_id=new_key,
                platform_override=source,
            )
        finally:
            _clear_session_context(tokens)
        _init_session(
            new_sid,
            new_key,
            agent,
            list(history),
            cols=session.get("cols", 80),
            source=source,
        )
        if new_sid in _sessions:
            _sessions[new_sid]["active_session_lease"] = lease
    except Exception as e:
        if lease is not None:
            lease.release()
        return _err(rid, 5000, f"agent init failed on branch: {e}")
    return _ok(rid, {"session_id": new_sid, "title": title, "parent": old_key})


@method("session.interrupt")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    # Safety net: if the turn's run thread is already gone but `running` stayed
    # stuck (a crash/desync that skipped the run loop's `finally`), force-clear it
    # so the session can't be permanently bricked at 4009 "session busy" — every
    # send/restore/resume would otherwise reject until a full backend restart.
    # Always tell the agent to interrupt when the session claims a run is active:
    # stale flags are cleared below, and fresh turns clear the interrupt flag at
    # entry. This keeps a stale/missing thread handle from making Stop a no-op.
    run_thread = session.get("_run_thread")
    run_thread_alive = run_thread is not None and run_thread.is_alive()
    should_interrupt = bool(session.get("running"))
    if should_interrupt and hasattr(session["agent"], "interrupt"):
        session["agent"].interrupt()
    with session["history_lock"]:
        session["_turn_cancel_requested"] = True
        session["queued_prompt"] = None
    if not run_thread_alive:
        with session["history_lock"]:
            if session.get("running"):
                session["running"] = False
                _clear_inflight_turn(session)

    # Stop = stop the TURN (cooperative interrupt above also kills the in-flight
    # foreground subprocess). Background processes the agent started (dev servers,
    # watchers) are intentionally left running — kill those individually with the
    # "x" on the task row (process.kill). Don't reap them here.
    # Scope the pending-prompt release to THIS session.  A global
    # _clear_pending() would collaterally cancel clarify/sudo/secret
    # prompts on unrelated sessions sharing the same tui_gateway
    # process, silently resolving them to empty strings.
    _clear_pending(params.get("session_id", ""))
    try:
        from tools.approval import resolve_gateway_approval

        resolve_gateway_approval(session["session_key"], "deny", resolve_all=True)
    except Exception:
        pass
    return _ok(rid, {"status": "interrupted"})


# ── Delegation: subagent tree observability + controls ───────────────
# Powers the TUI's /agents overlay (see ui-tui/src/components/agentsOverlay).
# The registry lives in tools/delegate_tool — these handlers are thin
# translators between JSON-RPC and the Python API.


@method("delegation.status")
def _(rid, params: dict) -> dict:
    from tools.delegate_tool import (
        is_spawn_paused,
        list_active_subagents,
        _get_max_concurrent_children,
        _get_max_spawn_depth,
    )

    return _ok(
        rid,
        {
            "active": list_active_subagents(),
            "paused": is_spawn_paused(),
            "max_spawn_depth": _get_max_spawn_depth(),
            "max_concurrent_children": _get_max_concurrent_children(),
        },
    )


@method("delegation.pause")
def _(rid, params: dict) -> dict:
    from tools.delegate_tool import set_spawn_paused

    paused = bool(params.get("paused", True))
    return _ok(rid, {"paused": set_spawn_paused(paused)})


@method("subagent.interrupt")
def _(rid, params: dict) -> dict:
    from tools.delegate_tool import interrupt_subagent

    subagent_id = str(params.get("subagent_id") or "").strip()
    if not subagent_id:
        return _err(rid, 4000, "subagent_id required")
    ok = interrupt_subagent(subagent_id)
    return _ok(rid, {"found": ok, "subagent_id": subagent_id})


# ── Spawn-tree snapshots: TUI-written, disk-persisted ────────────────
# The TUI is the source of truth for subagent state (it assembles payloads
# from the event stream).  On turn-complete it posts the final tree here;
# /replay and /replay-diff fetch past snapshots by session_id + filename.
#
# Layout:  $HERMES_HOME/spawn-trees/<session_id>/<timestamp>.json
# Each file contains { session_id, started_at, finished_at, subagents: [...] }.


def _spawn_trees_root():
    from hermes_constants import get_hermes_home

    root = get_hermes_home() / "spawn-trees"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _spawn_tree_session_dir(session_id: str):
    safe = (
        "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id) or "unknown"
    )
    d = _spawn_trees_root() / safe
    d.mkdir(parents=True, exist_ok=True)
    return d


# Per-session append-only index of lightweight snapshot metadata.  Read by
# `spawn_tree.list` so scanning doesn't require reading every full snapshot
# file (Copilot review on #14045).  One JSON object per line.
_SPAWN_TREE_INDEX = "_index.jsonl"


def _append_spawn_tree_index(session_dir, entry: dict) -> None:
    try:
        with (session_dir / _SPAWN_TREE_INDEX).open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as exc:
        # Index is a cache — losing a line just means list() falls back
        # to a directory scan for that entry.  Never block the save.
        logger.debug("spawn_tree index append failed: %s", exc)


def _read_spawn_tree_index(session_dir) -> list[dict]:
    index_path = session_dir / _SPAWN_TREE_INDEX
    if not index_path.exists():
        return []
    out: list[dict] = []
    try:
        with index_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return out


@method("spawn_tree.save")
def _(rid, params: dict) -> dict:
    session_id = str(params.get("session_id") or "").strip()
    subagents = params.get("subagents") or []
    if not isinstance(subagents, list) or not subagents:
        return _err(rid, 4000, "subagents list required")

    from datetime import datetime

    started_at = params.get("started_at")
    finished_at = params.get("finished_at") or time.time()
    label = str(params.get("label") or "")
    ts = datetime.utcfromtimestamp(float(finished_at)).strftime("%Y%m%dT%H%M%S")
    fname = f"{ts}.json"
    d = _spawn_tree_session_dir(session_id or "default")
    path = d / fname
    try:
        payload = {
            "session_id": session_id,
            "started_at": float(started_at) if started_at else None,
            "finished_at": float(finished_at),
            "label": label,
            "subagents": subagents,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        return _err(rid, 5000, f"spawn_tree.save failed: {exc}")

    _append_spawn_tree_index(
        d,
        {
            "path": str(path),
            "session_id": session_id,
            "started_at": payload["started_at"],
            "finished_at": payload["finished_at"],
            "label": label,
            "count": len(subagents),
        },
    )

    return _ok(rid, {"path": str(path), "session_id": session_id})


@method("spawn_tree.list")
def _(rid, params: dict) -> dict:
    session_id = str(params.get("session_id") or "").strip()
    limit = int(params.get("limit") or 50)
    cross_session = bool(params.get("cross_session"))

    if cross_session:
        root = _spawn_trees_root()
        roots = [p for p in root.iterdir() if p.is_dir()]
    else:
        roots = [_spawn_tree_session_dir(session_id or "default")]

    entries: list[dict] = []
    for d in roots:
        indexed = _read_spawn_tree_index(d)
        if indexed:
            # Skip index entries whose snapshot file was manually deleted.
            entries.extend(
                e for e in indexed if (p := e.get("path")) and Path(p).exists()
            )
            continue

        # Fallback for legacy (pre-index) sessions: full scan.  O(N) reads
        # but only runs once per session until the next save writes the index.
        for p in d.glob("*.json"):
            if p.name == _SPAWN_TREE_INDEX:
                continue
            try:
                stat = p.stat()
                try:
                    raw = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    raw = {}
                subagents = raw.get("subagents") or []
                entries.append({
                    "path": str(p),
                    "session_id": raw.get("session_id") or d.name,
                    "finished_at": raw.get("finished_at") or stat.st_mtime,
                    "started_at": raw.get("started_at"),
                    "label": raw.get("label") or "",
                    "count": len(subagents) if isinstance(subagents, list) else 0,
                })
            except OSError:
                continue

    entries.sort(key=lambda e: e.get("finished_at") or 0, reverse=True)
    return _ok(rid, {"entries": entries[:limit]})


@method("spawn_tree.load")
def _(rid, params: dict) -> dict:
    from pathlib import Path

    raw_path = str(params.get("path") or "").strip()
    if not raw_path:
        return _err(rid, 4000, "path required")

    # Reject paths escaping the spawn-trees root.
    root = _spawn_trees_root().resolve()
    try:
        resolved = Path(raw_path).resolve()
        resolved.relative_to(root)
    except (ValueError, OSError) as exc:
        return _err(rid, 4030, f"path outside spawn-trees root: {exc}")

    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _err(rid, 5000, f"spawn_tree.load failed: {exc}")

    return _ok(rid, payload)


@method("session.steer")
def _(rid, params: dict) -> dict:
    """Inject a user message into the next tool result without interrupting.

    Mirrors AIAgent.steer(). Safe to call while a turn is running — the text
    lands on the last tool result of the next tool batch and the model sees
    it on its next iteration. No interrupt, no new user turn, no role
    alternation violation.
    """
    text = (params.get("text") or "").strip()
    if not text:
        return _err(rid, 4002, "text is required")
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    agent = session.get("agent")
    if agent is None or not hasattr(agent, "steer"):
        return _err(rid, 4010, "agent does not support steer")
    try:
        accepted = agent.steer(text)
    except Exception as exc:
        return _err(rid, 5000, f"steer failed: {exc}")
    return _ok(rid, {"status": "queued" if accepted else "rejected", "text": text})


@method("terminal.resize")
def _(rid, params: dict) -> dict:
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    session["cols"] = int(params.get("cols", 80))
    return _ok(rid, {"cols": session["cols"]})


# ── Methods: prompt ──────────────────────────────────────────────────


@method("prompt.submit")
def _(rid, params: dict) -> dict:
    sid, text = params.get("session_id", ""), params.get("text", "")
    truncate_user_ordinal = params.get("truncate_before_user_ordinal")
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    # Re-bind to the current client transport for this request. This keeps
    # streaming events on the active websocket even if an earlier disconnect
    # or fallback moved the session transport to stdio.
    if (t := current_transport()) is not None:
        session["transport"] = t
    with session["history_lock"]:
        if session.get("running"):
            # Don't reject a mid-turn prompt — queue it (and, by default,
            # interrupt the live turn) so it runs as the next turn. See
            # _handle_busy_submit for why the old "session busy" rejection
            # dropped messages when teardown outlived the client's retry window.
            return _handle_busy_submit(rid, sid, session, text, t or session.get("transport"))
        # A watch session's run lives in the PARENT turn, so its own running
        # flag is False — without this, typing mid-run builds a second agent
        # racing the in-flight child on the same stored session (interleaved
        # transcript, stale fork). After the run completes, submitting is fine:
        # the upgrade resumes the child's transcript as a normal conversation.
        if session.get("lazy") and _child_run_active(str(session.get("session_key") or "")):
            return _err(rid, 4009, "subagent still running — wait for it to finish")
        if truncate_user_ordinal is not None:
            try:
                ordinal = int(truncate_user_ordinal)
            except (TypeError, ValueError):
                return _err(
                    rid, 4004, "truncate_before_user_ordinal must be an integer"
                )
            history = session.get("history", [])
            user_indices = [i for i, m in enumerate(history) if m.get("role") == "user"]
            # Reject out-of-range ordinals on BOTH ends. A negative value would
            # otherwise sail past the upper-bound check and hit Python's negative
            # indexing below (user_indices[-1] -> the LAST user turn), silently
            # truncating history to everything before it and persisting that loss
            # via replace_messages — an unrecoverable overwrite of the session DB.
            if ordinal < 0 or ordinal >= len(user_indices):
                return _err(rid, 4018, "target user message is no longer in session history")
            truncated = history[: user_indices[ordinal]]
            session["history"] = truncated
            session["history_version"] = int(session.get("history_version", 0)) + 1
            if (db := _get_db()) is not None:
                try:
                    db.replace_messages(session["session_key"], truncated)
                except Exception as exc:
                    print(
                        f"[tui_gateway] prompt.submit: replace_messages failed: {exc}",
                        file=sys.stderr,
                    )
        session["running"] = True
        session["_turn_cancel_requested"] = False
        session["last_active"] = time.time()
        _start_inflight_turn(session, text)

    # Persist the DB row lazily, now that the user has actually sent a message.
    _ensure_session_db_row(session)
    # A branch becomes real here: copy its parent's transcript into the row so it
    # resumes with full context (the agent won't persist the seed itself).
    _persist_branch_seed(session)
    _start_agent_build(sid, session)

    def run_after_agent_ready() -> None:
        err = _wait_agent(session, rid)
        if err:
            _emit(
                "error",
                sid,
                {
                    "message": err.get("error", {}).get(
                        "message", "agent initialization failed"
                    )
                },
            )
            with session["history_lock"]:
                session["running"] = False
                _clear_inflight_turn(session)
            return
        with session["history_lock"]:
            if session.get("_turn_cancel_requested") or not session.get("running"):
                session["running"] = False
                _clear_inflight_turn(session)
                return
        _run_prompt_submit(rid, sid, session, text)

    run_thread = threading.Thread(target=run_after_agent_ready, daemon=True)
    # Keep a handle so session.interrupt can tell a live turn from a stuck
    # `running` flag (a turn that died without clearing it) and recover the latter.
    session["_run_thread"] = run_thread
    run_thread.start()
    return _ok(rid, {"status": "streaming"})


def _notification_event_belongs_elsewhere(sid: str, session: dict, evt: dict) -> bool:
    """True if ``evt`` is owned by a *different* live session.

    Background completions carry the ``session_key`` of the session that started
    the work. Async delegation completions from the desktop also carry
    ``origin_ui_session_id``: the live TUI tab/window that commissioned them.
    Since all desktop sessions share one process-wide completion queue, each
    poller must skip events it doesn't own so a detached result surfaces in the
    launching session, not whichever poller happened to dequeue first.
    """
    evt_ui_sid = str(evt.get("origin_ui_session_id") or "")
    if evt_ui_sid:
        if evt_ui_sid == str(sid or "") and not session.get("_finalized"):
            return False
        try:
            with _sessions_lock:
                owner_live = evt_ui_sid in _sessions and not _sessions[evt_ui_sid].get("_finalized")
        except Exception:
            owner_live = False
        if owner_live:
            return True
        # If the exact UI tab is gone, fall through to durable session_key
        # routing. That avoids wrong-session delivery while still allowing a
        # resumed continuation with the same durable key/lineage to claim it.

    evt_key = str(evt.get("session_key") or "")
    if not evt_key:
        return False

    current_keys = {
        str(session.get("session_key") or ""),
        _session_lookup_key(session, fallback=sid),
    }

    # Compression can rotate AIAgent.session_id while the detached child is
    # still running. Resolve the event's original key to its continuation tip so
    # an event captured before or after compression still maps to the same live
    # desktop session instead of becoming an orphan that any poller may consume.
    resolved_key = evt_key
    try:
        db = _get_db()
        if db is not None:
            resolved_key = db.resolve_resume_session_id(evt_key) or evt_key
    except Exception:
        resolved_key = evt_key

    # If the key has a live continuation, prefer that continuation over the
    # compressed parent. Otherwise a stale parent tab could consume the event
    # before the real current conversation sees it.
    if resolved_key != evt_key:
        if resolved_key in current_keys:
            return False
        try:
            with _sessions_lock:
                continuation_live = any(
                    not s.get("_finalized")
                    and (
                        str(s.get("session_key") or "") == resolved_key
                        or _session_lookup_key(s, fallback="") == resolved_key
                    )
                    for s in _sessions.values()
                )
        except Exception:
            continuation_live = False
        if continuation_live:
            return True

    if evt_key in current_keys:
        return False

    try:
        with _sessions_lock:
            snapshot = list(_sessions.values())
    except Exception:
        # If we can't safely enumerate live sessions, fail open so we don't
        # crash the poller thread or drop the event.
        return False

    return any(
        s is not session
        and not s.get("_finalized")
        and (
            str(s.get("session_key") or "") in {evt_key, resolved_key}
            or _session_lookup_key(s, fallback="") in {evt_key, resolved_key}
        )
        for s in snapshot
    )


def _session_owns_notification_event(sid: str, session: dict, evt: dict) -> bool:
    """True iff *this* session PROVABLY owns ``evt``.

    Positive ownership — the mirror of ``_notification_event_belongs_elsewhere``
    minus its orphan-adoption fallback. An event owns-matches when its
    ``origin_ui_session_id`` is this live session, or its ``session_key``
    (raw or resolved through the compression chain) matches this session's
    key/lineage. Used as a fail-closed gate for async-delegation payloads:
    "not provably elsewhere" is NOT good enough to inject a conversation
    payload into this chat (#55578).
    """
    if session.get("_finalized"):
        return False
    if str(evt.get("origin_ui_session_id") or "") == str(sid or ""):
        return True
    evt_key = str(evt.get("session_key") or "")
    if not evt_key:
        return False
    current_keys = {
        str(session.get("session_key") or ""),
        _session_lookup_key(session, fallback=sid),
    }
    if evt_key in current_keys:
        return True
    try:
        db = _get_db()
        resolved_key = (
            db.resolve_resume_session_id(evt_key) if db is not None else evt_key
        ) or evt_key
    except Exception:
        resolved_key = evt_key
    return resolved_key in current_keys


def _notification_event_dedup_key(evt: dict) -> tuple:
    """Return the UI-emission identity for a process notification event.

    Completion events are terminal notifications for a background process, so
    they remain one-shot per process session. Watch-match events are not
    terminal: a single background process can legitimately match the same or
    different patterns many times, so include event-specific content to avoid
    suppressing later distinct matches from the same process.
    """
    evt_type = evt.get("type", "completion")
    evt_sid = evt.get("session_id", "")
    if evt_type == "watch_match":
        return (
            evt_sid,
            evt_type,
            evt.get("command", ""),
            evt.get("pattern", ""),
            evt.get("output", ""),
            evt.get("suppressed", 0),
        )
    if evt_type.startswith("watch_overflow_") or evt_type == "watch_disabled":
        return (
            evt_sid,
            evt_type,
            evt.get("command", ""),
            evt.get("message", ""),
            evt.get("suppressed", 0),
        )
    if evt_type == "async_delegation":
        # Async-delegation completions have no process session_id; without
        # this the fallthrough keys every one as ("", "async_delegation")
        # and the second completion's status update is suppressed forever.
        return (evt.get("delegation_id", ""), evt_type)
    return (evt_sid, evt_type)


def _notification_poller_loop(
    stop_event: threading.Event, sid: str, session: dict
) -> None:
    """Poll completion_queue and dispatch notifications autonomously.

    Runs in a daemon thread started by _init_session(). Emits a
    status.update (kind=process) for user visibility, then chains an
    agent turn via _run_prompt_submit if the session is idle.

    NOTE: The completion_queue is global (one per process). If multiple
    TUI sessions coexist, whichever poller wakes first grabs the event,
    even if the process was started by a different session. This matches
    CLI/gateway behavior (single session per process).
    """
    from tools.process_registry import process_registry, format_process_notification

    _emitted = set()  # dedup re-queued events so same completion isn't emitted 50 times while session is busy
    while not stop_event.is_set() and not session.get("_finalized"):
        try:
            evt = process_registry.completion_queue.get(timeout=0.5)
        except Exception:
            continue

        # Multiple desktop sessions share this one process-wide queue. Only
        # consume events that belong to *this* session — otherwise a background
        # process started in session A would surface its completion in whichever
        # session's poller happened to wake first (Ben's "reported in a
        # different session" bug). Leave foreign events for their owner.
        if _notification_event_belongs_elsewhere(sid, session, evt):
            process_registry.completion_queue.put(evt)
            time.sleep(0.1)
            continue

        # Fail closed for async-delegation results (#55578): these carry a
        # conversation payload, and injecting one into any chat other than the
        # one that commissioned it is a hard cross-session leak. The
        # belongs-elsewhere check above already re-queued events owned by
        # another LIVE session; what reaches here is either ours or an
        # orphan whose owner is gone. Orphaned delegation payloads are
        # DROPPED, not adopted — the subagent's summary is already persisted
        # in the delegation records/output store, so nothing is lost, whereas
        # a wrong-chat injection is unrecoverable. Non-delegation events
        # (background process completions etc.) keep the historical
        # adopt-orphans behavior.
        if evt.get("type") == "async_delegation" and not _session_owns_notification_event(
            sid, session, evt
        ):
            logger.warning(
                "async-delegation completion %s has no live owner "
                "(origin=%r key=%r); dropping from injection instead of "
                "delivering to session %s (#55578 fail-closed; result "
                "remains in the delegation records)",
                evt.get("delegation_id", "?"),
                str(evt.get("origin_ui_session_id") or ""),
                str(evt.get("session_key") or ""),
                sid,
            )
            continue

        _evt_sid = evt.get("session_id", "")
        if evt.get("type") == "completion" and process_registry.is_completion_consumed(
            _evt_sid
        ):
            continue

        text = format_process_notification(evt)
        if not text:
            continue

        # Only emit the same notification identity to TUI once — re-queued
        # completions get re-emitted every 0.5s otherwise when session is busy,
        # while distinct watch_match events from the same process must remain
        # visible independently.
        _dedup_key = _notification_event_dedup_key(evt)
        if _dedup_key not in _emitted:
            _emit("status.update", sid, {"kind": "process", "text": text})
            _emitted.add(_dedup_key)

        _requeued = False
        with session["history_lock"]:
            if session.get("running"):
                process_registry.completion_queue.put(evt)
                _requeued = True
            else:
                session["running"] = True
        if _requeued:
            # Back off before re-polling: the re-queued event keeps the queue
            # non-empty, so without a sleep this loop spins at full speed
            # (100% CPU, GIL churn) for as long as the session stays busy.
            time.sleep(0.25)
            continue

        rid = f"__notif__{int(time.time() * 1000)}"
        try:
            _emit("message.start", sid)
            _run_prompt_submit(rid, sid, session, text)
        except Exception as exc:
            print(
                f"[tui_gateway] notification poller dispatch failed: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            with session["history_lock"]:
                session["running"] = False

    # Drain any remaining events after stop signal (process all pending
    # before exiting so nothing is lost on shutdown). Events owned by other
    # live sessions are set aside and re-queued so their poller still sees them.
    deferred: list = []
    while not process_registry.completion_queue.empty():
        try:
            evt = process_registry.completion_queue.get_nowait()
        except Exception:
            break
        if _notification_event_belongs_elsewhere(sid, session, evt):
            deferred.append(evt)
            continue
        # Same fail-closed rule as the live loop: an orphaned async-delegation
        # payload is never adopted by a foreign session — defer it (a later
        # resume of the owner's lineage can still claim it) rather than
        # injecting another chat's conversation here (#55578).
        if evt.get("type") == "async_delegation" and not _session_owns_notification_event(
            sid, session, evt
        ):
            deferred.append(evt)
            continue
        _evt_sid = evt.get("session_id", "")
        if evt.get("type") == "completion" and process_registry.is_completion_consumed(
            _evt_sid
        ):
            continue
        text = format_process_notification(evt)
        if not text:
            continue

        _dedup_key = _notification_event_dedup_key(evt)
        if _dedup_key not in _emitted:
            _emit("status.update", sid, {"kind": "process", "text": text})
            _emitted.add(_dedup_key)

        with session["history_lock"]:
            if session.get("running"):
                process_registry.completion_queue.put(evt)
                break
            session["running"] = True

        rid = f"__notif__{int(time.time() * 1000)}"
        try:
            _emit("message.start", sid)
            _run_prompt_submit(rid, sid, session, text)
        except Exception as exc:
            print(
                f"[tui_gateway] notification poller dispatch failed: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            with session["history_lock"]:
                session["running"] = False

    # Hand any other sessions' events back to the shared queue.
    for evt in deferred:
        process_registry.completion_queue.put(evt)


def _wire_agent_terminal_output() -> None:
    """Idempotently route background-process output (and tab-close requests) to
    the desktop, keyed by process id. Read-only agent terminal tabs stream
    `agent.terminal.output` chunks live instead of polling the output tail, and
    `process_registry.request_close_terminal` emits `terminal.close` so the agent
    can drop a tab without killing the process. Events are routed to the window
    that owns the process (its gateway session); `_emit`/`write_json` is
    `_stdout_lock`-guarded, so calling it from the registry's reader threads is
    safe."""
    from tools.process_registry import process_registry

    has_output_sink = getattr(process_registry, "on_output", None) is not None
    has_close_sink = getattr(process_registry, "on_close", None) is not None
    if has_output_sink and has_close_sink:
        return

    def _owner_sid_for_process(session) -> str:
        session_key = str(getattr(session, "session_key", "") or "")
        if not session_key:
            return ""
        with _sessions_lock:
            for sid, tui_session in _sessions.items():
                if str(tui_session.get("session_key") or "") == session_key:
                    return sid
        return ""

    def _emit_agent_terminal_output(session, chunk):
        _emit(
            "agent.terminal.output",
            _owner_sid_for_process(session),
            {"process_id": session.id, "chunk": chunk},
        )

    def _emit_agent_terminal_close(session, process_id):
        # session may be None (process already finished/pruned) — the tab can
        # still linger and be closed; route to the owning window when we can.
        sid = _owner_sid_for_process(session) if session is not None else ""
        _emit("terminal.close", sid, {"process_id": process_id})

    if not has_output_sink:
        process_registry.on_output = _emit_agent_terminal_output
    if not has_close_sink:
        process_registry.on_close = _emit_agent_terminal_close


def _start_notification_poller(sid: str, session: dict) -> threading.Event:
    """Start the background notification poller for a TUI session."""
    _wire_agent_terminal_output()
    stop = threading.Event()
    t = threading.Thread(
        target=_notification_poller_loop,
        args=(stop, sid, session),
        daemon=True,
    )
    t.start()
    return stop


def _run_prompt_submit(rid, sid: str, session: dict, text: Any) -> None:
    with session["history_lock"]:
        history = list(session["history"])
        history_version = int(session.get("history_version", 0))
        images = list(session.get("attached_images", []))
        session["attached_images"] = []
        if not isinstance(session.get("inflight_turn"), dict):
            _start_inflight_turn(session, text)
    agent = session["agent"]
    if hasattr(agent, "clear_interrupt"):
        try:
            agent.clear_interrupt()
        except Exception:
            pass
    _emit("message.start", sid)

    def run():
        approval_token = None
        session_tokens = []
        home_token = None  # per-turn HERMES_HOME override for a resumed remote profile
        goal_followup = None  # set by the post-turn goal hook below
        try:
            from tools.approval import (
                reset_current_session_key,
                set_current_session_key,
            )

            approval_token = set_current_session_key(session["session_key"])
            session_tokens = _set_session_context(
                session["session_key"],
                ui_session_id=sid,
            )
            _profile_home_str = session.get("profile_home")
            if _profile_home_str:
                home_token = set_hermes_home_override(_profile_home_str)
            # The sudo password callback is thread-local (tools.terminal_tool
            # _callback_tls), so wiring it on the build thread doesn't reach this
            # turn thread — terminal sudo prompts would fall through to /dev/tty
            # and hang the headless gateway. Re-wire here so the prompt routes to
            # the sudo.request overlay. (secret capture is a module global, so
            # re-running is a harmless no-op.)
            _wire_callbacks(sid)
            _sync_agent_model_with_config(sid, session)
            cwd = _session_cwd(session)
            _register_session_cwd(session)
            cols = session.get("cols", 80)
            streamer = make_stream_renderer(cols)
            prompt = text

            if isinstance(prompt, str) and "@" in prompt:
                from agent.context_references import preprocess_context_references
                from agent.model_metadata import get_model_context_length

                ctx_len = get_model_context_length(
                    getattr(agent, "model", "") or _resolve_model(),
                    base_url=getattr(agent, "base_url", "") or "",
                    api_key=getattr(agent, "api_key", "") or "",
                    provider=getattr(agent, "provider", "") or "",
                    config_context_length=getattr(
                        agent, "_config_context_length", None
                    ),
                )
                ctx = preprocess_context_references(
                    prompt,
                    cwd=cwd,
                    allowed_root=cwd,
                    context_length=ctx_len,
                )
                if ctx.blocked:
                    _emit(
                        "error",
                        sid,
                        {
                            "message": "\n".join(ctx.warnings)
                            or "Context injection refused."
                        },
                    )
                    return
                prompt = ctx.message

            # Decide image routing per-turn based on active provider/model.
            # "native" → pass pixels to the main model as OpenAI-style content
            # parts (adapters translate for Anthropic/Gemini/Bedrock/etc.).
            # "text"   → pre-analyze with vision_analyze and prepend the text.
            # See agent/image_routing.py for the full decision table.
            run_message: Any = prompt
            if images:
                try:
                    from agent.image_routing import (
                        decide_image_input_mode,
                        build_native_content_parts,
                    )
                    from agent.auxiliary_client import (
                        _read_main_model,
                        _read_main_provider,
                    )
                    from hermes_cli.config import load_config as _tui_load_config

                    _cfg = _tui_load_config()
                    _mode = decide_image_input_mode(
                        _read_main_provider(),
                        _read_main_model(),
                        _cfg,
                    )
                    if getattr(agent, "api_mode", "") == "codex_app_server":
                        _mode = "text"
                except Exception as _img_exc:
                    print(
                        f"[tui_gateway] image_routing decision failed, defaulting to text: {_img_exc}",
                        file=sys.stderr,
                    )
                    _mode = "text"

                if _mode == "native":
                    try:
                        _parts, _skipped = build_native_content_parts(
                            prompt,
                            images,
                        )
                        if _skipped:
                            print(
                                f"[tui_gateway] native image attachment skipped {len(_skipped)} unreadable path(s)",
                                file=sys.stderr,
                            )
                        if any(p.get("type") == "image_url" for p in _parts):
                            run_message = _parts
                        else:
                            run_message = _enrich_with_attached_images(prompt, images)
                    except Exception as _img_exc:
                        print(
                            f"[tui_gateway] native attach failed, falling back to text: {_img_exc}",
                            file=sys.stderr,
                        )
                        run_message = _enrich_with_attached_images(prompt, images)
                else:
                    run_message = _enrich_with_attached_images(prompt, images)

            def _stream(delta):
                with session["history_lock"]:
                    _append_inflight_delta(session, delta)
                payload = {"text": delta}
                if streamer and (r := streamer.feed(delta)) is not None:
                    payload["rendered"] = r
                _emit("message.delta", sid, payload)

            run_kwargs = {
                "conversation_history": list(history),
                "stream_callback": _stream,
            }
            try:
                if "task_id" in inspect.signature(agent.run_conversation).parameters:
                    run_kwargs["task_id"] = session["session_key"]
            except (TypeError, ValueError):
                pass
            result = agent.run_conversation(run_message, **run_kwargs)
            if "moa_one_shot_restore" in session:
                _restore = session.pop("moa_one_shot_restore", None)
                # Restore the model the user was on before the /moa one-shot.
                # The one-shot did a real in-place agent.switch_model() to MoA
                # (#53444), so undoing it must go back through the switch path —
                # resetting session["model_override"] alone would leave the live
                # agent's client pinned to MoA for the next turn.
                if isinstance(_restore, dict):
                    _prev_override = _restore.get("override")
                    _prev_model = _restore.get("model")
                    _prev_provider = _restore.get("provider")
                    if _prev_override is None:
                        session.pop("model_override", None)
                    else:
                        session["model_override"] = _prev_override
                    if _prev_model:
                        _raw = (
                            f"{_prev_model} --provider {_prev_provider}"
                            if _prev_provider
                            else _prev_model
                        )
                        try:
                            _apply_model_switch(
                                sid,
                                session,
                                _raw,
                                confirm_expensive_model=False,
                                pin_session_override=bool(_prev_override),
                                # Session-internal restore after the /moa
                                # one-shot — never persist to config.yaml.
                                persist_override=False,
                            )
                        except Exception as _moa_restore_exc:
                            logger.warning(
                                "MoA one-shot model restore failed: %s",
                                _moa_restore_exc,
                            )
                elif _restore is None:
                    session.pop("model_override", None)
                else:
                    session["model_override"] = _restore

            last_reasoning = None
            status_note = None
            if isinstance(result, dict):
                if isinstance(result.get("messages"), list):
                    with session["history_lock"]:
                        current_version = int(session.get("history_version", 0))
                        if current_version == history_version:
                            session["history"] = result["messages"]
                            session["history_version"] = history_version + 1
                        else:
                            # History mutated externally during the turn
                            # (undo/compress/retry/rollback now guard on
                            # session.running, but this is the defensive
                            # backstop for any path that slips past).
                            # Surface the desync rather than silently
                            # dropping the agent's output — the UI can
                            # show the response and warn that it was
                            # not persisted.
                            print(
                                f"[tui_gateway] prompt.submit: history_version mismatch "
                                f"(expected={history_version} current={current_version}) — "
                                f"agent output NOT written to session history",
                                file=sys.stderr,
                            )
                            status_note = (
                                "History changed during this turn — the response above is visible "
                                "but was not saved to session history."
                            )

                # If auto-compression fired inside run_conversation(), agent.session_id
                # may have rotated. Sync session_key before downstream title/goal/finalize
                # handling uses it. Preserve pending_title (user intent) so it can be
                # applied to the continuation. Restart slash worker so subsequent
                # worker-backed commands (/title etc.) target the live session.
                # Fix for #20001.
                _sync_session_key_after_compress(
                    sid,
                    session,
                    clear_pending_title=False,
                    restart_slash_worker=True,
                )

                raw = result.get("final_response", "")
                status = (
                    "interrupted"
                    if result.get("interrupted")
                    else "error"
                    if result.get("error")
                    else "complete"
                )
                # When the backend produced no visible response AND reported a
                # real error (e.g. invalid model slug → provider 4xx), surface
                # that error as the visible text instead of shipping an empty
                # turn to Ink. Mirrors classic CLI behavior at cli.py where
                # (failed|partial) + no final_response → "Error: <detail>".
                # Leaves the None-with-no-error path untouched: an empty
                # successful turn still renders as empty, and the existing
                # "(empty)" sentinel handling stays in its own lane.
                if (
                    (not raw)
                    and result.get("error")
                    and (result.get("failed") or result.get("partial"))
                ):
                    raw = f"Error: {result.get('error')}"
                lr = result.get("last_reasoning")
                if isinstance(lr, str) and lr.strip():
                    last_reasoning = lr.strip()
            else:
                raw = str(result)
                status = "complete"

            payload = {"text": raw, "usage": _get_usage(agent), "status": status}
            if last_reasoning:
                payload["reasoning"] = last_reasoning
            if status_note:
                payload["warning"] = status_note
            rendered = render_message(raw, cols)
            if rendered:
                payload["rendered"] = rendered
            with session["history_lock"]:
                _clear_inflight_turn(session)
            _emit("message.complete", sid, payload)

            # ── /goal continuation (Ralph-style loop) ─────────────────
            # After every TUI turn, if a /goal is active, ask the judge
            # whether the goal is done and — if not and we're still under
            # budget — queue a continuation prompt to run after this
            # thread releases session["running"]. The verdict message
            # ("✓ Goal achieved" / "⏸ budget exhausted") is surfaced as
            # a system line so the user sees progress regardless of
            # outcome. Mirrors gateway/run._post_turn_goal_continuation.
            if status == "complete" and isinstance(raw, str) and raw.strip():
                try:
                    from hermes_cli.goals import GoalManager

                    sid_key = session.get("session_key") or ""
                    if sid_key:
                        try:
                            goals_cfg = _load_cfg().get("goals") or {}
                            goal_max_turns = int(goals_cfg.get("max_turns", 20) or 20)
                        except Exception:
                            goal_max_turns = 20
                        goal_mgr = GoalManager(
                            session_id=sid_key,
                            default_max_turns=goal_max_turns,
                        )
                        if goal_mgr.is_active():
                            try:
                                from hermes_cli.goals import gather_background_processes as _gather_bg
                                _bg_procs = _gather_bg()
                            except Exception:
                                _bg_procs = None
                            decision = goal_mgr.evaluate_after_turn(
                                raw,
                                user_initiated=True,
                                background_processes=_bg_procs,
                            )
                            verdict_msg = decision.get("message") or ""
                            if verdict_msg:
                                _emit(
                                    "status.update",
                                    sid,
                                    {"kind": "goal", "text": verdict_msg},
                                )
                            if decision.get("should_continue"):
                                cont_prompt = decision.get("continuation_prompt") or ""
                                if cont_prompt:
                                    goal_followup = cont_prompt
                except Exception as _goal_exc:
                    print(
                        f"[tui_gateway] goal continuation hook failed: "
                        f"{type(_goal_exc).__name__}: {_goal_exc}",
                        file=sys.stderr,
                    )

            # Apply pending_title now that the DB row exists.
            _pending = session.get("pending_title")
            if _pending and status == "complete":
                _pdb = _get_db()
                if _pdb:
                    _session_key = session.get("session_key") or sid
                    try:
                        if _pdb.set_session_title(_session_key, _pending):
                            session["pending_title"] = None
                    except ValueError as exc:
                        # Invalid/duplicate title — non-retryable, drop it.
                        # Auto-title will take over. Fix for #19029.
                        session["pending_title"] = None
                        logger.info(
                            "Dropping pending title for session %s: %s",
                            _session_key,
                            exc,
                        )
                    except Exception:
                        # Transient DB failure — keep pending_title for retry.
                        pass

            if (
                status == "complete"
                and isinstance(raw, str)
                and raw.strip()
                and isinstance(text, str)
                and text.strip()
            ):
                try:
                    from agent.title_generator import maybe_auto_title

                    _title_key = session.get("session_key") or sid
                    maybe_auto_title(
                        _get_db(),
                        _title_key,
                        text,
                        raw,
                        session.get("history", []),
                        # Push the generated title live so the sidebar renames
                        # without waiting for the next list refresh (the titler
                        # runs async, after this turn's refresh already fired).
                        title_callback=lambda t, _k=_title_key: _emit(
                            "session.title", sid, {"session_id": _k, "title": t}
                        ),
                    )
                except Exception:
                    pass

            # CLI parity: when voice-mode TTS is on, speak the agent reply
            # (cli.py:_voice_speak_response).  Only the final text — tool
            # calls / reasoning already stream separately and would be
            # noisy to read aloud.
            if (
                status == "complete"
                and isinstance(raw, str)
                and raw.strip()
                and _voice_tts_enabled()
            ):
                try:
                    from hermes_cli.voice import speak_text

                    spoken = raw
                    threading.Thread(
                        target=speak_text, args=(spoken,), daemon=True
                    ).start()
                except ImportError:
                    logger.warning("voice TTS skipped: hermes_cli.voice unavailable")
                except Exception as e:
                    logger.warning("voice TTS dispatch failed: %s", e)
        except Exception as e:
            import traceback

            trace = traceback.format_exc()
            try:
                os.makedirs(os.path.dirname(_CRASH_LOG), exist_ok=True)
                with open(_CRASH_LOG, "a", encoding="utf-8") as f:
                    f.write(
                        f"\n=== turn-dispatcher exception · "
                        f"{time.strftime('%Y-%m-%d %H:%M:%S')} · sid={sid} ===\n"
                    )
                    f.write(trace)
            except Exception:
                pass
            print(
                f"[gateway-turn] {type(e).__name__}: {e}", file=sys.stderr, flush=True
            )
            _emit("error", sid, {"message": str(e)})
        finally:
            try:
                if approval_token is not None:
                    reset_current_session_key(approval_token)
            except Exception:
                pass
            if home_token is not None:
                reset_hermes_home_override(home_token)
            _clear_session_context(session_tokens)
            with session["history_lock"]:
                session["running"] = False
                session["last_active"] = time.time()
                _clear_inflight_turn(session)
            _emit("session.info", sid, _session_info(agent, session))

        # A user prompt that arrived mid-turn (interrupt + queue) wins over
        # every auto follow-up below — drain it first and skip them this cycle;
        # the goal judge / notifications re-evaluate at the end of that turn.
        if _drain_queued_prompt(rid, sid, session):
            return

        # Chain a goal-continuation turn if the judge said so. We do
        # this AFTER the finally releases session["running"], so the
        # nested _run_prompt_submit doesn't deadlock on the busy
        # guard. A real user prompt that races us wins because
        # prompt.submit sets running=True under the history_lock and
        # we check that guard before re-firing.
        if goal_followup:
            with session["history_lock"]:
                if session.get("running"):
                    # User already sent something — their turn wins,
                    # the judge will re-run on the next turn anyway.
                    return
                session["running"] = True
            try:
                _emit("message.start", sid)
                _run_prompt_submit(rid, sid, session, goal_followup)
            except Exception as _cont_exc:
                print(
                    f"[tui_gateway] goal continuation dispatch failed: "
                    f"{type(_cont_exc).__name__}: {_cont_exc}",
                    file=sys.stderr,
                )
                with session["history_lock"]:
                    session["running"] = False

        # Drain completion notifications that arrived during this turn.
        # The background poller handles between-turn delivery; this is
        # the safety net for events that arrived mid-turn.
        try:
            from tools.process_registry import process_registry

            # Positive-proof ownership (compression-chain aware) — the same
            # fail-closed gate the poller uses, so the post-turn drain can't
            # adopt another session's (or an orphan's) delegation payload,
            # while a post-compression session still claims its own
            # pre-compression dispatches (#55578).
            for _evt, synth in process_registry.drain_notifications(
                session_key=session.get("session_key", ""),
                owns_event=lambda e: _session_owns_notification_event(sid, session, e),
            ):
                with session["history_lock"]:
                    if session.get("running"):
                        process_registry.completion_queue.put(_evt)
                        break
                    session["running"] = True
                try:
                    _emit("message.start", sid)
                    _run_prompt_submit(rid, sid, session, synth)
                except Exception as _n_exc:
                    print(
                        f"[tui_gateway] completion notification dispatch failed: "
                        f"{type(_n_exc).__name__}: {_n_exc}",
                        file=sys.stderr,
                    )
                    with session["history_lock"]:
                        session["running"] = False
        except Exception as _drain_exc:
            print(
                f"[tui_gateway] completion queue drain failed: "
                f"{type(_drain_exc).__name__}: {_drain_exc}",
                file=sys.stderr,
            )

    run_thread = threading.Thread(target=run, daemon=True)
    session["_run_thread"] = run_thread
    run_thread.start()


@method("clipboard.paste")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    try:
        from hermes_cli.clipboard import has_clipboard_image, save_clipboard_image
    except Exception as e:
        return _err(rid, 5027, f"clipboard unavailable: {e}")

    session["image_counter"] = session.get("image_counter", 0) + 1
    img_dir = _hermes_home / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    img_path = (
        img_dir
        / f"clip_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{session['image_counter']}.png"
    )

    # Save-first: mirrors CLI keybinding path; more robust than has_image() precheck
    if not save_clipboard_image(img_path):
        session["image_counter"] = max(0, session["image_counter"] - 1)
        msg = (
            "Clipboard has image but extraction failed"
            if has_clipboard_image()
            else "No image found in clipboard"
        )
        return _ok(rid, {"attached": False, "message": msg})

    session.setdefault("attached_images", []).append(str(img_path))
    return _ok(
        rid,
        {
            "attached": True,
            "path": str(img_path),
            "count": len(session["attached_images"]),
            **_image_meta(img_path),
        },
    )


@method("image.attach")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    raw = str(params.get("path", "") or "").strip()
    if not raw:
        return _err(rid, 4015, "path required")
    try:
        from cli import (
            _IMAGE_EXTENSIONS,
            _detect_file_drop,
            _resolve_attachment_path,
            _split_path_input,
        )

        dropped = _detect_file_drop(raw)
        if dropped:
            image_path = dropped["path"]
            remainder = dropped["remainder"]
        else:
            path_token, remainder = _split_path_input(raw)
            image_path = _resolve_attachment_path(path_token)
            if image_path is None:
                return _err(rid, 4016, f"image not found: {path_token}")
        if image_path.suffix.lower() not in _IMAGE_EXTENSIONS:
            return _err(rid, 4016, f"unsupported image: {image_path.name}")
        session.setdefault("attached_images", []).append(str(image_path))
        return _ok(
            rid,
            {
                "attached": True,
                "path": str(image_path),
                "count": len(session["attached_images"]),
                "remainder": remainder,
                "text": remainder or f"[User attached image: {image_path.name}]",
                **_image_meta(image_path),
            },
        )
    except Exception as e:
        return _err(rid, 5027, str(e))


# Byte-upload attach caps. 25 MB matches Anthropic's per-image limit; 50 MB / 25
# pages bounds a single PDF drop so it can't blow the context budget.
_ATTACH_BYTES_MAX_BYTES = 25 * 1024 * 1024
_PDF_ATTACH_MAX_BYTES = 50 * 1024 * 1024
_PDF_ATTACH_MAX_PAGES = 25

# Leading magic bytes → file extension, for filename-less uploads.
_IMAGE_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", ".png"),
    (b"\xff\xd8\xff", ".jpg"),
    (b"GIF87a", ".gif"),
    (b"GIF89a", ".gif"),
    (b"BM", ".bmp"),
)


def _decode_attach_base64(raw: str, *, mime_prefix: str) -> bytes | None:
    """Decode a base64 (optionally data-URL-wrapped) payload.

    Accepts ``data:<mime_prefix>...;base64,<b64>`` plus embedded whitespace.
    Returns the decoded bytes, or ``None`` when the input isn't valid base64.
    """
    import base64 as _base64
    import re as _re

    cleaned = raw.strip()
    m = _re.match(
        rf"^data:{_re.escape(mime_prefix)}[a-zA-Z0-9.+-]*;base64,(.*)$",
        cleaned,
        _re.DOTALL,
    )
    if m:
        cleaned = m.group(1)
    cleaned = _re.sub(r"\s+", "", cleaned)
    try:
        return _base64.b64decode(cleaned, validate=True)
    except Exception:
        return None


def _sniff_image_ext(img_bytes: bytes, filename: str = "") -> str:
    """Resolve an image extension from a filename hint, else magic bytes.

    Falls back to ``.png``. WebP needs the RIFF/WEBP container check, handled
    before the generic table.
    """
    if filename:
        suffix = Path(filename).suffix.lower()
        if suffix:
            return suffix
    head = img_bytes[:16]
    if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
        return ".webp"
    for sig, ext in _IMAGE_MAGIC:
        if head.startswith(sig):
            return ext
    return ".png"


def _allowed_image_extensions() -> frozenset[str]:
    try:
        from cli import _IMAGE_EXTENSIONS

        return frozenset(_IMAGE_EXTENSIONS)
    except Exception:
        return frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"})


def _queue_attached_image(
    session: dict, img_bytes: bytes, ext: str, *, prefix: str
) -> Path:
    """Write image bytes into the gateway's images dir and queue them.

    Mirrors what ``image.attach`` does for a local path: appends to
    ``session["attached_images"]`` so the next ``prompt.submit`` picks it up via
    the existing native-image-attach pipeline. Returns the written path.
    """
    session["image_counter"] = session.get("image_counter", 0) + 1
    img_dir = _hermes_home / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_path = img_dir / f"{prefix}_{ts}_{session['image_counter']}{ext}"
    try:
        img_path.write_bytes(img_bytes)
    except Exception:
        session["image_counter"] = max(0, session["image_counter"] - 1)
        raise
    session.setdefault("attached_images", []).append(str(img_path))
    return img_path


@method("image.attach_bytes")
def _(rid, params: dict) -> dict:
    """Attach an image to the session from base64 bytes (remote-client path).

    A desktop app or web dashboard running on a DIFFERENT machine than the
    gateway can't hand us a local path — that file only exists on the client's
    disk. So it uploads the raw image bytes (base64) and we write them into the
    gateway's own images dir. The response shape mirrors ``image.attach`` so the
    client treats both identically.

    Params:
      content_base64 / data (str, required): base64 image bytes. Accepts a
        ``data:image/...;base64,`` prefix and embedded whitespace. ``data`` is
        an accepted alias for older desktop builds.
      filename / ext (str, optional): extension hint. Without it, magic bytes
        identify PNG/JPEG/GIF/WebP/BMP, falling back to ``.png``.
    """
    session, err = _sess(params, rid)
    if err:
        return err

    raw_b64 = str(params.get("content_base64") or params.get("data") or "").strip()
    if not raw_b64:
        return _err(rid, 4015, "content_base64 required")

    img_bytes = _decode_attach_base64(raw_b64, mime_prefix="image/")
    if img_bytes is None:
        return _err(rid, 4017, "data is not valid base64")
    if not img_bytes:
        return _err(rid, 4017, "image is empty")
    if len(img_bytes) > _ATTACH_BYTES_MAX_BYTES:
        mb = _ATTACH_BYTES_MAX_BYTES // (1024 * 1024)
        return _err(
            rid, 4018, f"image too large ({len(img_bytes)} bytes; cap is {mb} MB)"
        )

    filename = str(params.get("filename", "") or "")
    ext_hint = str(params.get("ext", "") or "").strip().lower()
    if ext_hint and not ext_hint.startswith("."):
        ext_hint = "." + ext_hint
    ext = _sniff_image_ext(img_bytes, filename or (f"x{ext_hint}" if ext_hint else ""))
    if ext not in _allowed_image_extensions():
        return _err(rid, 4016, f"unsupported image extension: {ext}")

    try:
        img_path = _queue_attached_image(session, img_bytes, ext, prefix="upload")
    except Exception as e:
        return _err(rid, 5027, f"write failed: {e}")

    return _ok(
        rid,
        {
            "attached": True,
            "path": str(img_path),
            "count": len(session["attached_images"]),
            "remainder": "",
            "text": f"[User attached image: {img_path.name}]",
            "bytes": len(img_bytes),
            **_image_meta(img_path),
        },
    )


@method("pdf.attach")
def _(rid, params: dict) -> dict:
    """Attach a PDF by rendering each page to PNG and queuing the pages.

    Anthropic's vision pipeline accepts images, not PDFs, so this runs
    ``pdftoppm`` (poppler-utils) at 150 DPI per page and queues each rendered
    page as an attached image. Accepts either a host ``path`` (local mode) or
    base64 ``content_base64`` (remote upload). Caps at 50 MB / 25 pages per call.

    Requires ``pdftoppm`` on $PATH (``apt install poppler-utils``); returns 5028
    if missing.
    """
    import shutil
    import subprocess
    import tempfile

    session, err = _sess(params, rid)
    if err:
        return err

    if shutil.which("pdftoppm") is None:
        return _err(
            rid, 5028, "pdftoppm not installed (poppler-utils package required)"
        )

    raw_path = str(params.get("path", "") or "").strip()
    raw_b64 = str(params.get("content_base64") or params.get("data") or "").strip()
    if not raw_path and not raw_b64:
        return _err(rid, 4015, "path or content_base64 required")

    with tempfile.TemporaryDirectory(prefix="pdf_attach_") as td:
        td_path = Path(td)
        if raw_b64:
            pdf_bytes = _decode_attach_base64(raw_b64, mime_prefix="application/pdf")
            if pdf_bytes is None:
                return _err(rid, 4017, "data is not valid base64")
            if not pdf_bytes:
                return _err(rid, 4017, "decoded PDF is empty")
            if len(pdf_bytes) > _PDF_ATTACH_MAX_BYTES:
                mb = _PDF_ATTACH_MAX_BYTES // (1024 * 1024)
                return _err(
                    rid, 4018, f"PDF too large ({len(pdf_bytes)} bytes; cap is {mb} MB)"
                )
            if pdf_bytes[:5] != b"%PDF-":
                return _err(
                    rid, 4017, "payload is not a PDF (missing %PDF- magic bytes)"
                )
            pdf_path = td_path / "input.pdf"
            pdf_path.write_bytes(pdf_bytes)
            display_name = str(params.get("filename", "") or "uploaded.pdf")
        else:
            try:
                from cli import _resolve_attachment_path

                resolved = _resolve_attachment_path(raw_path)
            except Exception:
                resolved = None
            if resolved is None or not Path(resolved).is_file():
                return _err(rid, 4016, f"PDF not found: {raw_path}")
            if Path(resolved).suffix.lower() != ".pdf":
                return _err(rid, 4016, f"not a PDF: {Path(resolved).name}")
            if Path(resolved).stat().st_size > _PDF_ATTACH_MAX_BYTES:
                mb = _PDF_ATTACH_MAX_BYTES // (1024 * 1024)
                return _err(rid, 4018, f"PDF too large; cap is {mb} MB")
            pdf_path = Path(resolved)
            display_name = pdf_path.name

        try:
            first_page = int(params.get("first_page") or 1)
            last_page_param = params.get("last_page")
            last_page = int(last_page_param) if last_page_param is not None else None
        except (TypeError, ValueError):
            return _err(rid, 4015, "first_page/last_page must be integers")

        if first_page < 1:
            return _err(rid, 4015, "first_page must be >= 1")
        if last_page is None:
            last_page = first_page + _PDF_ATTACH_MAX_PAGES - 1
        if last_page < first_page:
            return _err(rid, 4015, "last_page must be >= first_page")
        if last_page - first_page + 1 > _PDF_ATTACH_MAX_PAGES:
            return _err(
                rid,
                4019,
                f"page range exceeds cap of {_PDF_ATTACH_MAX_PAGES} pages per attach call",
            )

        out_prefix = td_path / "page"
        argv = [
            "pdftoppm",
            "-png",
            "-r",
            "150",
            "-f",
            str(first_page),
            "-l",
            str(last_page),
            str(pdf_path),
            str(out_prefix),
        ]
        from hermes_cli._subprocess_compat import windows_hide_flags

        try:
            res = subprocess.run(
                argv, capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL,
                creationflags=windows_hide_flags(),
            )
        except subprocess.TimeoutExpired:
            return _err(rid, 5028, "pdftoppm timed out (>120s)")
        if res.returncode != 0:
            tail = (res.stderr or res.stdout or "").strip().splitlines()[-3:]
            return _err(rid, 5028, "pdftoppm failed: " + " | ".join(tail))

        rendered = sorted(td_path.glob("page-*.png"))
        if not rendered:
            return _err(rid, 5028, "pdftoppm produced no pages (corrupt PDF?)")

        attached_pages = []
        for src in rendered:
            page_num = src.stem.split("-", 1)[-1]
            try:
                page_int = int(page_num)
            except ValueError:
                page_int = first_page + len(attached_pages)
            dst = _queue_attached_image(
                session, src.read_bytes(), ".png", prefix=f"pdf_p{page_num}"
            )
            attached_pages.append({
                "path": str(dst),
                "page": page_int,
                **_image_meta(dst),
            })

        return _ok(
            rid,
            {
                "attached": True,
                "filename": display_name,
                "pages_attached": len(attached_pages),
                "pages": attached_pages,
                "count": len(session["attached_images"]),
                "text": f"[User attached PDF: {display_name} ({len(attached_pages)} page(s))]",
            },
        )


_ATTACHMENT_REF_NEEDS_QUOTING_RE = None


def _format_ref_value(value: str) -> str:
    """Quote a context-ref value when it contains whitespace or bracket chars.

    Mirrors the desktop ``formatRefValue`` so the staged ``@file:`` ref round-trips
    through ``agent.context_references`` cleanly.
    """
    import re as _re

    global _ATTACHMENT_REF_NEEDS_QUOTING_RE
    if _ATTACHMENT_REF_NEEDS_QUOTING_RE is None:
        _ATTACHMENT_REF_NEEDS_QUOTING_RE = _re.compile(r"""[\s()\[\]{}<>"'`]""")
    if not value or not _ATTACHMENT_REF_NEEDS_QUOTING_RE.search(value):
        return value
    if "`" not in value:
        return f"`{value}`"
    if '"' not in value:
        return f'"{value}"'
    if "'" not in value:
        return f"'{value}'"
    return value


def _attachment_ref_path(session: dict, target: Path) -> str:
    """Workspace-relative path for an attachment, or the absolute path if outside."""
    workspace = Path(_session_cwd(session)).resolve()
    try:
        rel = target.resolve().relative_to(workspace)
        return str(rel).replace(os.sep, "/")
    except ValueError:
        return str(target.resolve())


def _desktop_attachment_dir(session: dict) -> Path:
    root = Path(_session_cwd(session)).resolve() / ".hermes" / "desktop-attachments"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _sanitize_attachment_name(name: str) -> str:
    import re as _re

    candidate = Path(str(name or "").strip()).name
    candidate = _re.sub(r"[\x00-\x1f]+", "_", candidate)
    candidate = candidate.strip().strip(".")
    return candidate or "attachment"


def _unique_attachment_path(root: Path, filename: str) -> Path:
    candidate = root / filename
    if not candidate.exists():
        return candidate
    stem = Path(filename).stem or "attachment"
    suffix = Path(filename).suffix
    counter = 2
    while True:
        next_candidate = root / f"{stem}-{counter}{suffix}"
        if not next_candidate.exists():
            return next_candidate
        counter += 1


def _resolve_gateway_attachment_path(raw: str) -> Path | None:
    """Resolve a raw path token to a gateway-visible file, or None."""
    if not raw:
        return None
    try:
        from cli import _detect_file_drop, _resolve_attachment_path, _split_path_input
    except Exception:
        return None

    dropped = _detect_file_drop(raw)
    if dropped:
        return Path(dropped["path"]).resolve()
    path_token, _remainder = _split_path_input(raw)
    resolved = _resolve_attachment_path(path_token)
    return Path(resolved).resolve() if resolved is not None else None


def _decode_attachment_data_url(data_url: str) -> bytes:
    """Decode a ``data:<any-mime>;base64,<b64>`` payload to bytes.

    Unlike ``_decode_attach_base64`` (image-mime-specific), this accepts any
    media type — text/csv, application/pdf, etc. — so non-image file uploads
    round-trip. Also tolerates a bare base64 string with no data-URL prefix.
    """
    import base64 as _base64
    import binascii as _binascii
    import re as _re

    cleaned = (data_url or "").strip()
    m = _re.match(r"^data:[^;,]*(?:;[^;,=]+=[^;,]+)*;base64,(.*)$", cleaned, _re.DOTALL | _re.I)
    if m:
        cleaned = m.group(1)
    cleaned = _re.sub(r"\s+", "", cleaned)
    try:
        return _base64.b64decode(cleaned, validate=True)
    except (ValueError, _binascii.Error) as exc:
        raise ValueError("invalid data_url payload") from exc


def _stage_session_file_attachment(
    session: dict,
    *,
    raw_path: str,
    data_url: str,
    name: str,
) -> tuple[Path, bool]:
    """Make a desktop file attachment available to the remote gateway agent.

    Three cases:
      1. The path resolves to a file already INSIDE the session workspace — use
         it as-is (no copy, ``uploaded=False``).
      2. The path resolves to a gateway-visible file OUTSIDE the workspace — copy
         it into ``.hermes/desktop-attachments/`` so the ``@file:`` ref resolves.
      3. The path doesn't exist on the gateway (the common remote case: it's a
         path on the CLIENT's disk) — decode the uploaded ``data_url`` bytes and
         write them into ``.hermes/desktop-attachments/``.

    Returns ``(stored_path, uploaded)``.
    """
    workspace = Path(_session_cwd(session)).resolve()
    resolved = _resolve_gateway_attachment_path(raw_path)
    if resolved is not None:
        try:
            resolved.relative_to(workspace)
            return resolved, False
        except ValueError:
            payload = resolved.read_bytes()
            filename = resolved.name
    else:
        if not data_url:
            raise ValueError("file not found on gateway and no data_url provided")
        payload = _decode_attachment_data_url(data_url)
        filename = _sanitize_attachment_name(name or Path(str(raw_path or "")).name)

    upload_dir = _desktop_attachment_dir(session)
    target = _unique_attachment_path(upload_dir, _sanitize_attachment_name(filename))
    target.write_bytes(payload)
    return target.resolve(), True


@method("file.attach")
def _(rid, params: dict) -> dict:
    """Stage a non-image file attachment into the session workspace.

    The image/PDF path renders to vision tiles; this one keeps the file as a
    readable artifact and returns a workspace-relative ``@file:`` ref so the
    agent's file tools (and ``agent.context_references``) can read it. Solves the
    remote-gateway case where the desktop passes a path that only exists on the
    CLIENT's disk: the client uploads ``data_url`` bytes and we materialize the
    file on the gateway.

    Params:
      session_id (str, required)
      path (str): client/host path of the file (used for naming + local-mode
        gateway-visible resolution).
      data_url (str): ``data:<mime>;base64,<b64>`` upload of the file bytes,
        required when the path isn't visible to the gateway.
      name (str, optional): preferred filename.
    """
    session, err = _sess(params, rid)
    if err:
        return err
    raw = str(params.get("path", "") or "").strip()
    data_url = str(params.get("data_url", "") or "").strip()
    name = str(params.get("name", "") or "").strip()
    if not raw and not data_url:
        return _err(rid, 4015, "path or data_url required")
    try:
        stored_path, uploaded = _stage_session_file_attachment(
            session, raw_path=raw, data_url=data_url, name=name
        )
        ref_path = _attachment_ref_path(session, stored_path)
        return _ok(
            rid,
            {
                "attached": True,
                "name": stored_path.name,
                "path": str(stored_path),
                "ref_path": ref_path,
                "ref_text": f"@file:{_format_ref_value(ref_path)}",
                "uploaded": uploaded,
            },
        )
    except Exception as e:
        return _err(rid, 5028, str(e))


@method("image.detach")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    raw = str(params.get("path", "") or "").strip()
    if not raw:
        return _err(rid, 4015, "path required")
    images = session.setdefault("attached_images", [])
    before = len(images)
    session["attached_images"] = [path for path in images if path != raw]
    return _ok(
        rid,
        {
            "detached": len(session["attached_images"]) != before,
            "count": len(session["attached_images"]),
        },
    )


@method("input.detect_drop")
def _(rid, params: dict) -> dict:
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    try:
        from cli import _detect_file_drop

        raw = str(params.get("text", "") or "")
        dropped = _detect_file_drop(raw)
        if not dropped:
            return _ok(rid, {"matched": False})

        drop_path = dropped["path"]
        remainder = dropped["remainder"]
        if dropped["is_image"]:
            session.setdefault("attached_images", []).append(str(drop_path))
            text = remainder or f"[User attached image: {drop_path.name}]"
            return _ok(
                rid,
                {
                    "matched": True,
                    "is_image": True,
                    "path": str(drop_path),
                    "count": len(session["attached_images"]),
                    "text": text,
                    **_image_meta(drop_path),
                },
            )

        text = f"[User attached file: {drop_path}]" + (
            f"\n{remainder}" if remainder else ""
        )
        return _ok(
            rid,
            {
                "matched": True,
                "is_image": False,
                "path": str(drop_path),
                "name": drop_path.name,
                "text": text,
            },
        )
    except Exception as e:
        return _err(rid, 5027, str(e))


@method("prompt.background")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    text, parent = params.get("text", ""), params.get("session_id", "")
    if not text:
        return _err(rid, 4012, "text required")
    task_id = f"bg_{uuid.uuid4().hex[:6]}"

    def run():
        session_tokens = _set_session_context(task_id, cwd=_session_cwd(session))
        try:
            from run_agent import AIAgent

            result = AIAgent(
                **_background_agent_kwargs(session["agent"], task_id)
            ).run_conversation(
                user_message=text,
                task_id=task_id,
            )
            _emit(
                "background.complete",
                parent,
                {
                    "task_id": task_id,
                    "text": (
                        result.get("final_response", str(result))
                        if isinstance(result, dict)
                        else str(result)
                    ),
                },
            )
        except Exception as e:
            _emit(
                "background.complete",
                parent,
                {"task_id": task_id, "text": f"error: {e}"},
            )
        finally:
            _clear_session_context(session_tokens)

    threading.Thread(target=run, daemon=True).start()
    return _ok(rid, {"task_id": task_id})


@method("preview.restart")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err

    url = str(params.get("url") or "").strip()
    cwd = str(params.get("cwd") or "").strip()
    context = str(params.get("context") or "").strip()

    if not url:
        return _err(rid, 4012, "url required")

    task_id = f"preview_{uuid.uuid4().hex[:6]}"
    parent = params.get("session_id", "")
    parent_history = _preview_restart_history(session)
    has_history = bool(parent_history)
    prompt = "\n".join(
        line
        for line in [
            "The desktop preview pane cannot load a local server URL.",
            "",
            f"Preview URL: {url}",
            f"Current working directory: {cwd or '(unknown)'}",
            "",
            f"Preview console:\n{context}" if context else "",
            "" if context else "",
            (
                "The conversation history above is from the user's main session — including the commands you (the assistant) previously ran to start servers, edit files, or check ports. Use it to figure out exactly which server should be running at this Preview URL. The user did not start a brand new task; recover what they had working."
                if has_history
                else None
            ),
            "Restart exactly the app intended for the Preview URL, not Hermes Desktop itself.",
            "The Preview URL and port are the target. Preserve that target unless you conclude it is impossible.",
            "If the prior conversation shows a specific command that bound this URL/port, prefer re-running THAT exact command (in the same cwd) over guessing a new one.",
            "First inspect what process, if any, owns the Preview URL port. If a stale server exists, inspect its cwd and prefer that cwd over the Hermes/Desktop process cwd.",
            "The Current working directory is only a hint. Do not assume it is the preview app root when the port owner or files indicate another root.",
            "If the console shows a module-script MIME error for src/main.tsx or similar, a static server is serving source files. Do not restart python -m http.server or any dumb static server for that app.",
            "For module-script MIME failures, inspect package.json/vite config in the candidate app root and start the real dev server/bundler (for example npm/pnpm/yarn dev) so module transforms happen.",
            "Before declaring success, verify the Preview URL responds with the intended app, not Hermes Desktop. If it serves Hermes/Desktop UI or another unrelated app, stop that process and report failure.",
            "Do not modify files. Do not ask the user unless blocked.",
            "Prefer existing project scripts or commands when they are clear.",
            "If a stale process owns the needed port, handle it safely.",
            "Start long-running servers detached/in the background, then return immediately.",
            "Do not run a foreground dev server command that blocks this background task.",
            "Keep the final response short: what command/server was started, or why it could not be restarted.",
        ]
        if line
    )

    # Normalize defensively: a malformed client path (embedded NUL, etc.) must
    # not blow up the whole restart — treat it as "no validated cwd".
    try:
        preview_cwd = os.path.abspath(os.path.expanduser(cwd)) if cwd else ""
        if preview_cwd and not os.path.isdir(preview_cwd):
            preview_cwd = ""
    except Exception:
        preview_cwd = ""

    def run():
        # Pin the validated preview cwd, else the parent workspace — never an
        # invalid client path, which would silently fall back to the launch dir.
        session_tokens = _set_session_context(
            task_id, cwd=(preview_cwd or _session_cwd(session))
        )
        try:
            from run_agent import AIAgent
            from tools.terminal_tool import register_task_env_overrides

            if preview_cwd:
                register_task_env_overrides(task_id, {"cwd": preview_cwd})

            history_note = (
                f" (with {len(parent_history)} parent-session messages of context)"
                if parent_history
                else ""
            )
            _emit(
                "preview.restart.progress",
                parent,
                {
                    "task_id": task_id,
                    "text": f"Starting hidden restart agent{history_note}",
                },
            )
            result = AIAgent(
                **_ephemeral_preview_agent_kwargs(session["agent"], task_id),
                **_preview_restart_callbacks(parent, task_id),
            ).run_conversation(
                user_message=prompt,
                task_id=task_id,
                conversation_history=parent_history or None,
            )
            text = (
                result.get("final_response", str(result))
                if isinstance(result, dict)
                else str(result)
            )
            _emit(
                "preview.restart.complete", parent, {"task_id": task_id, "text": text}
            )
        except Exception as e:
            _emit(
                "preview.restart.complete",
                parent,
                {"task_id": task_id, "text": f"error: {e}"},
            )
        finally:
            try:
                from tools.terminal_tool import clear_task_env_overrides

                clear_task_env_overrides(task_id)
            except Exception:
                pass
            _clear_session_context(session_tokens)

    threading.Thread(target=run, daemon=True).start()
    return _ok(rid, {"task_id": task_id})


# ── Methods: respond ─────────────────────────────────────────────────


def _respond(rid, params, key):
    r = params.get("request_id", "")
    with _prompt_lock:
        entry = _pending.get(r)
        if not entry:
            return _err(rid, 4009, f"no pending {key} request")
        _, ev = entry
        _answers[r] = params.get(key, "")
        ev.set()
    return _ok(rid, {"status": "ok"})


@method("clarify.respond")
def _(rid, params: dict) -> dict:
    return _respond(rid, params, "answer")


@method("terminal.read.respond")
def _(rid, params: dict) -> dict:
    # `text` is a JSON string of the serialized terminal buffer + line metadata.
    return _respond(rid, params, "text")


@method("sudo.respond")
def _(rid, params: dict) -> dict:
    return _respond(rid, params, "password")


@method("secret.respond")
def _(rid, params: dict) -> dict:
    return _respond(rid, params, "value")


@method("approval.respond")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    try:
        from tools.approval import resolve_gateway_approval

        return _ok(
            rid,
            {
                "resolved": resolve_gateway_approval(
                    session["session_key"],
                    params.get("choice", "deny"),
                    resolve_all=params.get("all", False),
                )
            },
        )
    except Exception as e:
        return _err(rid, 5004, str(e))


# ── Methods: config ──────────────────────────────────────────────────


@method("config.set")
def _(rid, params: dict) -> dict:
    key, value = params.get("key", ""), params.get("value", "")
    session = _sessions.get(params.get("session_id", ""))

    if key == "model":
        try:
            if not value:
                return _err(rid, 4002, "model value required")
            if session:
                # Reject during an in-flight turn.  agent.switch_model()
                # mutates self.model / self.provider / self.base_url /
                # self.client in place; the worker thread running
                # agent.run_conversation is reading those on every
                # iteration.  A mid-turn swap can send an HTTP request
                # with the new base_url but old model (or vice versa),
                # producing 400/404s the user never asked for.  Parity
                # with the gateway's running-agent /model guard.
                if session.get("running"):
                    return _err(
                        rid,
                        4009,
                        "session busy — /interrupt the current turn before switching models",
                    )
                from hermes_cli.model_switch import parse_model_flags

                parsed_flags = parse_model_flags(value)
                _model_input, explicit_provider, _persist_global, _force_refresh, _is_session = parsed_flags
                if session.get("agent") is None and not explicit_provider.strip():
                    session_id = params.get("session_id", "")
                    _start_agent_build(session_id, session)
                    init_err = _wait_agent(session, rid)
                    if init_err:
                        return init_err
                    if session.get("agent") is None:
                        return _err(rid, 5032, "agent initialization failed")
                result = _apply_model_switch(
                    params.get("session_id", ""),
                    session,
                    value,
                    confirm_expensive_model=bool(
                        params.get("confirm_expensive_model", False)
                    ),
                    parsed_flags=parsed_flags,
                )
            else:
                result = _apply_model_switch(
                    "",
                    {"agent": None},
                    value,
                    confirm_expensive_model=bool(
                        params.get("confirm_expensive_model", False)
                    ),
                )
            return _ok(
                rid,
                {
                    "key": key,
                    "value": result["value"],
                    "warning": result["warning"],
                    "confirm_required": result.get("confirm_required", False),
                    "confirm_message": result.get("confirm_message", ""),
                },
            )
        except Exception as e:
            return _err(rid, 5001, str(e))

    if key == "fast":
        raw = str(value or "").strip().lower()
        agent = session.get("agent") if session else None
        if agent is not None:
            current_fast = getattr(agent, "service_tier", None) == "priority"
        else:
            current_fast = _load_service_tier() == "priority"

        if raw in {"status"}:
            return _ok(
                rid,
                {"key": key, "value": "fast" if current_fast else "normal"},
            )

        if raw in {"", "toggle"}:
            nv = "normal" if current_fast else "fast"
        elif raw in {"fast", "on"}:
            nv = "fast"
        elif raw in {"normal", "off"}:
            nv = "normal"
        else:
            return _err(rid, 4002, f"unknown fast mode: {value}")

        overrides = None
        if nv == "fast":
            from hermes_cli.models import resolve_fast_mode_overrides

            target_model = (
                getattr(agent, "model", None) if agent is not None else _resolve_model()
            )
            if not target_model:
                return _err(
                    rid,
                    4002,
                    "fast mode is not available without a selected model",
                )
            overrides = resolve_fast_mode_overrides(target_model)
            if overrides is None:
                return _err(
                    rid,
                    4002,
                    "fast mode is not available for this model",
                )

        _write_config_key("agent.service_tier", nv)
        if agent is not None:
            agent.service_tier = "priority" if nv == "fast" else None
            current_overrides = dict(getattr(agent, "request_overrides", {}) or {})
            current_overrides.pop("service_tier", None)
            current_overrides.pop("speed", None)
            if nv == "fast":
                current_overrides.update(overrides)
            agent.request_overrides = current_overrides
            _persist_live_session_runtime(session)
            _emit(
                "session.info",
                params.get("session_id", ""),
                _session_info(agent, session),
            )
        return _ok(rid, {"key": key, "value": nv})

    if key == "busy":
        raw = str(value or "").strip().lower()
        if raw in {"", "status"}:
            return _ok(rid, {"key": key, "value": _load_busy_input_mode()})
        if raw not in {"queue", "steer", "interrupt"}:
            return _err(rid, 4002, f"unknown busy mode: {value}")
        _write_config_key("display.busy_input_mode", raw)
        return _ok(rid, {"key": key, "value": raw})

    if key == "verbose":
        cycle = ["off", "new", "all", "verbose"]
        cur = (
            session.get("tool_progress_mode", _load_tool_progress_mode())
            if session
            else _load_tool_progress_mode()
        )
        if value and value != "cycle":
            nv = str(value).strip().lower()
            if nv not in cycle:
                return _err(rid, 4002, f"unknown verbose mode: {value}")
        else:
            try:
                idx = cycle.index(cur)
            except ValueError:
                idx = 2
            nv = cycle[(idx + 1) % len(cycle)]
        _write_config_key("display.tool_progress", nv)
        if session:
            session["tool_progress_mode"] = nv
            agent = session.get("agent")
            if agent is not None:
                agent.verbose_logging = nv == "verbose"
        return _ok(rid, {"key": key, "value": nv})

    if key == "yolo":
        # Approval bypass. Two scopes:
        #   scope="session" (default) — same as the TUI's Shift+Tab. Toggles
        #     ONLY this session's _session_yolo flag; never touches global
        #     config, so CLI / TUI / cron behavior is unaffected.
        #   scope="global" (Shift+click the zap) — flips the persistent global
        #     approvals.mode in config.yaml between "off" (bypass on) and
        #     "manual" (bypass off). This DOES affect every session, the CLI,
        #     the TUI, and cron, and survives restarts.
        scope = str(params.get("scope") or "session").strip().lower()
        try:
            from tools.approval import (
                disable_session_yolo,
                enable_session_yolo,
                is_session_yolo_enabled,
            )

            raw = str(value or "").strip().lower()

            def _resolve_toggle(current: bool) -> bool:
                if raw in {"1", "on", "true", "yes"}:
                    return True
                if raw in {"0", "off", "false", "no"}:
                    return False
                return not current

            if scope == "global":
                from tools.approval import _normalize_approval_mode

                cfg = _load_cfg()
                appr = cfg.get("approvals") if isinstance(cfg, dict) else None
                if not isinstance(appr, dict):
                    appr = {}
                current = _normalize_approval_mode(appr.get("mode", "manual")) == "off"
                enable = _resolve_toggle(current)
                # Toggle between full bypass and the default manual gate. We do
                # not try to restore a prior "smart"/custom mode — the zap is a
                # binary on/off affordance; users with bespoke modes set them in
                # config.yaml.
                _write_config_key("approvals.mode", "off" if enable else "manual")
                nv = "1" if enable else "0"
                # Reflect the global flip in every live session's indicator.
                for sid, sess in list(_sessions.items()):
                    agent = sess.get("agent")
                    if agent is not None:
                        _emit("session.info", sid, _session_info(agent, sess))
                return _ok(rid, {"key": key, "value": nv, "scope": "global"})

            if session:
                current = is_session_yolo_enabled(session["session_key"])
                enable = _resolve_toggle(current)
                if enable:
                    enable_session_yolo(session["session_key"])
                    nv = "1"
                else:
                    disable_session_yolo(session["session_key"])
                    nv = "0"
                agent = session.get("agent")
                if agent is not None:
                    _emit(
                        "session.info",
                        params.get("session_id", ""),
                        _session_info(agent, session),
                    )
            else:
                current = is_truthy_value(os.environ.get("HERMES_YOLO_MODE"))
                enable = _resolve_toggle(current)
                if enable:
                    os.environ["HERMES_YOLO_MODE"] = "1"
                    nv = "1"
                else:
                    os.environ.pop("HERMES_YOLO_MODE", None)
                    nv = "0"
            return _ok(rid, {"key": key, "value": nv, "scope": "session"})
        except Exception as e:
            return _err(rid, 5001, str(e))

    if key == "reasoning":
        try:
            from hermes_constants import parse_reasoning_effort

            arg = str(value or "").strip().lower()
            if arg in {"show", "on"}:
                cfg = _load_cfg()
                display = (
                    cfg.get("display") if isinstance(cfg.get("display"), dict) else {}
                )
                sections = (
                    display.get("sections")
                    if isinstance(display.get("sections"), dict)
                    else {}
                )
                display["show_reasoning"] = True
                sections["thinking"] = "expanded"
                display["sections"] = sections
                cfg["display"] = display
                _save_cfg(cfg)
                if session:
                    session["show_reasoning"] = True
                return _ok(rid, {"key": key, "value": "show"})
            if arg in {"hide", "off"}:
                cfg = _load_cfg()
                display = (
                    cfg.get("display") if isinstance(cfg.get("display"), dict) else {}
                )
                sections = (
                    display.get("sections")
                    if isinstance(display.get("sections"), dict)
                    else {}
                )
                display["show_reasoning"] = False
                sections["thinking"] = "hidden"
                display["sections"] = sections
                cfg["display"] = display
                _save_cfg(cfg)
                if session:
                    session["show_reasoning"] = False
                return _ok(rid, {"key": key, "value": "hide"})

            # /reasoning full | clamp — parity with the classic CLI's
            # reasoning_full toggle. The TUI renders thinking as an
            # expand/collapse section rather than a fixed 10-line recap, so
            # full maps to sections.thinking=expanded and clamp to collapsed.
            # display.reasoning_full is persisted too so the config key stays
            # consistent across the CLI and TUI surfaces.
            if arg in {"full", "all"}:
                cfg = _load_cfg()
                display = (
                    cfg.get("display") if isinstance(cfg.get("display"), dict) else {}
                )
                sections = (
                    display.get("sections")
                    if isinstance(display.get("sections"), dict)
                    else {}
                )
                display["reasoning_full"] = True
                sections["thinking"] = "expanded"
                display["sections"] = sections
                cfg["display"] = display
                _save_cfg(cfg)
                return _ok(rid, {"key": key, "value": "full"})
            if arg in {"clamp", "collapse", "short"}:
                cfg = _load_cfg()
                display = (
                    cfg.get("display") if isinstance(cfg.get("display"), dict) else {}
                )
                sections = (
                    display.get("sections")
                    if isinstance(display.get("sections"), dict)
                    else {}
                )
                display["reasoning_full"] = False
                sections["thinking"] = "collapsed"
                display["sections"] = sections
                cfg["display"] = display
                _save_cfg(cfg)
                return _ok(rid, {"key": key, "value": "clamp"})

            parsed = parse_reasoning_effort(arg)
            if parsed is None:
                return _err(rid, 4002, f"unknown reasoning value: {value}")
            if session is not None:
                # Session-scoped, like the messaging gateway's `/reasoning
                # <level>` (global persistence is `--global` / Settings →
                # Model territory). Writing config.yaml here let every
                # desktop model-menu selection rewrite the user's global
                # agent.reasoning_effort to the preset default.
                session["create_reasoning_override"] = parsed
                if session.get("agent") is not None:
                    session["agent"].reasoning_config = parsed
                    _persist_live_session_runtime(session)
                    _emit(
                        "session.info",
                        params.get("session_id", ""),
                        _session_info(session["agent"], session),
                    )
            else:
                _write_config_key("agent.reasoning_effort", arg)
            return _ok(rid, {"key": key, "value": arg})
        except Exception as e:
            return _err(rid, 5001, str(e))

    if key == "details_mode":
        nv = str(value or "").strip().lower()
        if nv not in _DETAIL_MODES:
            return _err(rid, 4002, f"unknown details_mode: {value}")
        cfg = _load_cfg()
        display = cfg.get("display") if isinstance(cfg.get("display"), dict) else {}
        sections = (
            display.get("sections") if isinstance(display.get("sections"), dict) else {}
        )
        display["details_mode"] = nv
        for section in _DETAIL_SECTION_NAMES:
            sections[section] = nv
        display["sections"] = sections
        cfg["display"] = display
        _save_cfg(cfg)
        return _ok(rid, {"key": key, "value": nv})

    if key.startswith("details_mode."):
        # Per-section override: `details_mode.<section>` writes to
        # `display.sections.<section>`. Empty value clears the explicit
        # override and lets frontend resolution apply built-in section defaults
        # before the global details_mode.
        section = key.split(".", 1)[1]
        if section not in _DETAIL_SECTION_NAMES:
            return _err(rid, 4002, f"unknown section: {section}")

        cfg = _load_cfg()
        display = cfg.get("display") if isinstance(cfg.get("display"), dict) else {}
        sections_cfg = (
            display.get("sections") if isinstance(display.get("sections"), dict) else {}
        )

        nv = str(value or "").strip().lower()
        if not nv:
            sections_cfg.pop(section, None)
            display["sections"] = sections_cfg
            cfg["display"] = display
            _save_cfg(cfg)
            return _ok(rid, {"key": key, "value": ""})

        if nv not in _DETAIL_MODES:
            return _err(rid, 4002, f"unknown details_mode: {value}")

        sections_cfg[section] = nv
        display["sections"] = sections_cfg
        cfg["display"] = display
        _save_cfg(cfg)
        return _ok(rid, {"key": key, "value": nv})

    if key == "thinking_mode":
        nv = str(value or "").strip().lower()
        allowed_tm = frozenset({"collapsed", "truncated", "full"})
        if nv not in allowed_tm:
            return _err(rid, 4002, f"unknown thinking_mode: {value}")
        _write_config_key("display.thinking_mode", nv)
        # Backward compatibility bridge: keep details_mode aligned.
        _write_config_key(
            "display.details_mode", "expanded" if nv == "full" else "collapsed"
        )
        return _ok(rid, {"key": key, "value": nv})

    if key == "compact":
        raw = str(value or "").strip().lower()
        cfg0 = _load_cfg()
        d0 = cfg0.get("display") if isinstance(cfg0.get("display"), dict) else {}
        cur_b = bool(d0.get("tui_compact", False))
        if raw in {"", "toggle"}:
            nv_b = not cur_b
        elif raw == "on":
            nv_b = True
        elif raw == "off":
            nv_b = False
        else:
            return _err(rid, 4002, f"unknown compact value: {value}")
        _write_config_key("display.tui_compact", nv_b)
        return _ok(rid, {"key": key, "value": "on" if nv_b else "off"})

    if key == "statusbar":
        raw = str(value or "").strip().lower()
        display = _load_cfg().get("display")
        d0 = display if isinstance(display, dict) else {}
        current = _coerce_statusbar(d0.get("tui_statusbar", "top"))

        if raw in {"", "toggle"}:
            nv = "top" if current == "off" else "off"
        elif raw == "on":
            nv = "top"
        elif raw in _STATUSBAR_MODES:
            nv = raw
        else:
            return _err(rid, 4002, f"unknown statusbar value: {value}")

        _write_config_key("display.tui_statusbar", nv)
        return _ok(rid, {"key": key, "value": nv})

    if key == "mouse":
        # Explicit None check rather than `value or ""` so falsy non-string
        # inputs (0, False) reach the alias map as themselves — both map to
        # 'off' via _MOUSE_TRACKING_ALIASES — instead of being collapsed to
        # '' and triggering the toggle path. The slash command always passes
        # a string, but programmatic JSON-RPC callers may send booleans.
        raw = ("" if value is None else str(value)).strip().lower()
        cfg = _load_cfg()
        display = cfg.get("display") if isinstance(cfg.get("display"), dict) else {}
        current = _display_mouse_tracking(display)

        if raw in {"", "toggle"}:
            nv = "all" if current == "off" else "off"
        elif raw in _MOUSE_TRACKING_ALIASES:
            nv = _MOUSE_TRACKING_ALIASES[raw]
        else:
            return _err(rid, 4002, f"unknown mouse value: {value}")

        _write_config_key("display.mouse_tracking", nv)
        return _ok(rid, {"key": key, "value": nv})

    if key == "indicator":
        # Use an explicit None check rather than `value or ""` so falsy
        # non-string inputs (0, False, []) still surface as themselves
        # in the error message instead of looking like a blank value.
        raw = ("" if value is None else str(value)).strip().lower()
        if raw not in _INDICATOR_STYLES:
            return _err(
                rid,
                4002,
                f"unknown indicator: {raw!r}; pick one of {'|'.join(_INDICATOR_STYLES)}",
            )
        _write_config_key("display.tui_status_indicator", raw)
        return _ok(rid, {"key": key, "value": raw})

    if key in {"cwd", "terminal.cwd", "workdir"}:
        raw = str(value or "").strip()
        if not raw:
            return _err(rid, 4002, "cwd required")
        cwd = os.path.abspath(os.path.expanduser(raw))
        if not os.path.isdir(cwd):
            return _err(rid, 4002, f"working directory does not exist: {raw}")
        _write_config_key("terminal.cwd", cwd)
        os.environ["TERMINAL_CWD"] = cwd
        return _ok(
            rid,
            {
                "key": "terminal.cwd",
                "value": cwd,
                "cwd": cwd,
                "branch": _git_branch_for_cwd(cwd),
            },
        )

    if key in {"prompt", "personality", "skin"}:
        try:
            cfg = _load_cfg()
            if key == "prompt":
                if value == "clear":
                    cfg.pop("custom_prompt", None)
                    nv = ""
                else:
                    cfg["custom_prompt"] = value
                    nv = value
                _save_cfg(cfg)
            elif key == "personality":
                sid_key = params.get("session_id", "")
                pname, new_prompt = _validate_personality(str(value or ""), cfg)
                _write_config_key("display.personality", pname)
                _write_config_key("agent.system_prompt", new_prompt)
                nv = str(value or "none")
                history_reset, info = _apply_personality_to_session(
                    sid_key, session, new_prompt, pname
                )
            else:
                _write_config_key(f"display.{key}", value)
                nv = value
                if key == "skin":
                    _emit("skin.changed", "", resolve_skin())
            resp = {"key": key, "value": nv}
            if key == "personality":
                resp["history_reset"] = history_reset
                if info is not None:
                    resp["info"] = info
            return _ok(rid, resp)
        except Exception as e:
            return _err(rid, 5001, str(e))

    return _err(rid, 4002, f"unknown config key: {key}")


# ---------------------------------------------------------------------------
# Projects — first-class, per-profile, multi-folder workspaces
# ---------------------------------------------------------------------------


# JSON-RPC error codes for the projects surface.
_E_PROJECTS = 5061  # generic failure
_E_NO_PROJECT = 5062  # id resolved to nothing
_E_PROJECT_ARG = 5063  # invalid argument (e.g. bad name/slug)


class _NoProject(Exception):
    """Raised inside a projects handler when ``params['id']`` resolves to None."""


def _projects_payload(conn) -> dict:
    from hermes_cli import projects_db as pdb

    return {
        "projects": [p.to_dict() for p in pdb.list_projects(conn, include_archived=True)],
        "active_id": pdb.get_active_id(conn),
    }


def _projects_method(name: str):
    """Register a projects RPC, injecting (pdb, conn) and unifying error mapping.

    Every project CRUD handler opened the per-profile DB, mapped a missing id to
    5062, bad args to 5063, and everything else to 5061. This collapses that
    boilerplate so each handler is just its one meaningful operation.
    """

    def decorator(fn):
        @method(name)
        def handler(rid, params: dict) -> dict:
            try:
                from hermes_cli import projects_db as pdb

                with pdb.connect_closing() as conn:
                    return fn(rid, params, pdb, conn)
            except _NoProject:
                return _err(rid, _E_NO_PROJECT, "no such project")
            except ValueError as e:
                return _err(rid, _E_PROJECT_ARG, str(e))
            except Exception as e:
                return _err(rid, _E_PROJECTS, str(e))

        return handler

    return decorator


def _require_project(pdb, conn, params: dict):
    """The project named by ``params['id']`` (or raise ``_NoProject``)."""
    proj = pdb.get_project(conn, str(params.get("id") or ""))
    if proj is None:
        raise _NoProject
    return proj


@_projects_method("projects.list")
def _(rid, params, pdb, conn) -> dict:
    return _ok(rid, _projects_payload(conn))


@_projects_method("projects.get")
def _(rid, params, pdb, conn) -> dict:
    return _ok(rid, {"project": _require_project(pdb, conn, params).to_dict()})


@_projects_method("projects.create")
def _(rid, params, pdb, conn) -> dict:
    pid = pdb.create_project(
        conn,
        name=str(params.get("name") or ""),
        slug=params.get("slug"),
        folders=params.get("folders") or [],
        primary_path=params.get("primary_path"),
        description=params.get("description"),
        icon=params.get("icon"),
        color=params.get("color"),
        board_slug=params.get("board_slug"),
    )
    if params.get("use"):
        pdb.set_active(conn, pid)
    proj = pdb.get_project(conn, pid)
    return _ok(rid, {"project": proj.to_dict() if proj else None})


@_projects_method("projects.update")
def _(rid, params, pdb, conn) -> dict:
    proj = _require_project(pdb, conn, params)
    pdb.update_project(
        conn,
        proj.id,
        name=params.get("name"),
        description=params.get("description"),
        icon=params.get("icon"),
        color=params.get("color"),
        board_slug=params.get("board_slug"),
    )
    return _ok(rid, {"project": pdb.get_project(conn, proj.id).to_dict()})


@_projects_method("projects.add_folder")
def _(rid, params, pdb, conn) -> dict:
    proj = _require_project(pdb, conn, params)
    pdb.add_folder(
        conn,
        proj.id,
        str(params.get("path") or ""),
        label=params.get("label"),
        is_primary=bool(params.get("is_primary")),
    )
    return _ok(rid, {"project": pdb.get_project(conn, proj.id).to_dict()})


@_projects_method("projects.remove_folder")
def _(rid, params, pdb, conn) -> dict:
    proj = _require_project(pdb, conn, params)
    pdb.remove_folder(conn, proj.id, str(params.get("path") or ""))
    return _ok(rid, {"project": pdb.get_project(conn, proj.id).to_dict()})


@_projects_method("projects.set_primary")
def _(rid, params, pdb, conn) -> dict:
    proj = _require_project(pdb, conn, params)
    pdb.set_primary(conn, proj.id, str(params.get("path") or ""))
    return _ok(rid, {"project": pdb.get_project(conn, proj.id).to_dict()})


@_projects_method("projects.archive")
def _(rid, params, pdb, conn) -> dict:
    proj = _require_project(pdb, conn, params)
    (pdb.restore_project if params.get("restore") else pdb.archive_project)(conn, proj.id)
    return _ok(rid, _projects_payload(conn))


@_projects_method("projects.delete")
def _(rid, params, pdb, conn) -> dict:
    proj = _require_project(pdb, conn, params)
    pdb.delete_project(conn, proj.id)
    return _ok(rid, _projects_payload(conn))


@_projects_method("projects.set_active")
def _(rid, params, pdb, conn) -> dict:
    pdb.set_active(conn, _require_project(pdb, conn, params).id if params.get("id") else None)
    return _ok(rid, {"active_id": pdb.get_active_id(conn)})


@_projects_method("projects.for_cwd")
def _(rid, params, pdb, conn) -> dict:
    cwd = _completion_cwd({"cwd": str(params.get("cwd") or "").strip()} if params.get("cwd") else {})
    proj = pdb.project_for_path(conn, cwd)
    return _ok(rid, {"project": proj.to_dict() if proj else None, "cwd": cwd, "branch": _git_branch_for_cwd(cwd)})


def _is_repo_junk(root: str) -> bool:
    """A git root we never auto-surface as a project: the bare home dir or
    anything under HERMES_HOME (~/.hermes by default) — config/sessions/skills,
    not a workspace. User-created projects pointing there are still honored."""
    if not root:
        return True

    from hermes_constants import get_hermes_home

    real = os.path.realpath(root)
    home = os.path.realpath(os.path.expanduser("~"))
    hermes_home = os.path.realpath(str(get_hermes_home()))

    return real == home or real == hermes_home or real.startswith(hermes_home + os.sep)


def _discover_repos_payload(db, *, conn=None, backfill: bool = True) -> list[dict]:
    """Merge filesystem-scanned repos (cached) with session-derived repo roots.

    Repo-first: the disk scan (persisted by `projects.record_repos`) surfaces
    repos even with zero hermes sessions. Session-derived roots cover repos
    outside the scan roots. Both are junk-filtered (hermes home subtree + bare
    home) and carry their session totals for the overview.

    ``conn`` reuses an already-open projects.db connection (the tree path holds
    one); ``backfill`` persists resolved roots back onto session rows — kept off
    the per-turn tree path (grouping uses the live git resolver regardless) and
    done only on the explicit discover/record refresh.
    """
    _is_junk = _is_repo_junk
    repos: dict[str, dict] = {}

    def _agg(root: str) -> dict:
        return repos.setdefault(root, {"root": root, "label": "", "sessions": 0, "last_active": 0.0})

    # Session-derived roots (common repo root, folding worktrees; cached) +
    # backfill the column so persisted git_repo_root matches the tree grouping.
    cwd_rows = list(db.distinct_session_cwds())
    # Warm the per-cwd git probes in parallel so a cold first paint doesn't
    # serialize one subprocess per distinct cwd before this loop reads the cache.
    git_probe.warm_roots(str(r.get("cwd") or "") for r in cwd_rows)
    cwd_to_root: dict[str, str] = {}
    for row in cwd_rows:
        cwd = str(row.get("cwd") or "")
        root = _git_common_repo_root_for_cwd(cwd)
        if not root:
            continue
        cwd_to_root[cwd] = root
        if _is_junk(root):
            continue
        agg = _agg(root)
        agg["sessions"] += int(row.get("sessions") or 0)
        agg["last_active"] = max(agg["last_active"], float(row.get("last_active") or 0))

    if backfill:
        try:
            db.backfill_repo_roots(cwd_to_root)
        except Exception:
            logger.debug("failed to backfill repo roots", exc_info=True)

    # Filesystem-scanned roots from the cache (may have zero sessions). Reuse the
    # caller's projects.db connection when given, else open a short-lived one.
    try:
        from hermes_cli import projects_db as pdb

        def _read(c) -> None:
            for entry in pdb.list_discovered_repos(c):
                root = str(entry.get("root") or "")
                if not root or _is_junk(root):
                    continue
                agg = _agg(root)
                if entry.get("label"):
                    agg["label"] = entry["label"]
                agg["last_active"] = max(agg["last_active"], float(entry.get("last_seen") or 0))

        if conn is not None:
            _read(conn)
        else:
            with pdb.connect_closing() as own:
                _read(own)
    except Exception:
        logger.debug("failed to read discovered repo cache", exc_info=True)

    out = sorted(repos.values(), key=lambda r: r["last_active"], reverse=True)
    for r in out:
        r["label"] = r["label"] or os.path.basename(r["root"].rstrip("/\\")) or r["root"]
    return out


@method("projects.discover_repos")
def _(rid, params: dict) -> dict:
    """Repos for the desktop overview: scanned-from-disk (cached) ∪ session-derived."""
    try:
        db = _get_db()
        if db is None:
            return _ok(rid, {"repos": []})
        return _ok(rid, {"repos": _discover_repos_payload(db)})
    except Exception as e:
        return _err(rid, 5061, str(e))


@method("projects.record_repos")
def _(rid, params: dict) -> dict:
    """Persist git repo roots found by the client's filesystem scan, then return
    the merged repo list. The native crawl runs on the desktop (local fs); this
    caches the result so later reads are instant instead of re-walking disk."""
    try:
        from hermes_cli import projects_db as pdb

        pairs: list[tuple[str, str | None]] = []
        for item in params.get("repos") or []:
            if isinstance(item, str):
                pairs.append((item, None))
            elif isinstance(item, dict) and item.get("root"):
                pairs.append((str(item["root"]), item.get("label")))

        with pdb.connect_closing() as conn:
            pdb.record_discovered_repos(conn, pairs, replace=True)

        db = _get_db()
        return _ok(rid, {"repos": _discover_repos_payload(db) if db is not None else []})
    except Exception as e:
        return _err(rid, 5061, str(e))


# Sources excluded from the project tree: cron runs and tool/subagent children
# are not user conversations. Subagent/compression children are already dropped
# by list_sessions_rich(include_children=False); cron has its own section.
_PROJECT_TREE_EXCLUDED_SOURCES = ["cron"]


def _project_tree_row(r: dict) -> dict:
    """Project a SessionDB row to the minimal shape the sidebar renders.

    Keeps the fields the grouping needs (cwd / git_branch / git_repo_root) plus
    everything ``SidebarSessionRow`` reads, and drops the heavy columns
    (system_prompt, model_config, ...) so the tree payload stays lean.
    """
    return {
        "id": r.get("id"),
        "_lineage_root_id": r.get("_lineage_root_id"),
        # The sidebar nests branch/fork sessions under their parent
        # (flattenSessionsWithBranches keys on this); without it, lane rows can't
        # draw the └─ connector the flat Recents list shows.
        "parent_session_id": r.get("parent_session_id"),
        "title": r.get("title"),
        "preview": r.get("preview"),
        "started_at": r.get("started_at") or 0,
        "ended_at": r.get("ended_at"),
        "last_active": r.get("last_active") or r.get("started_at") or 0,
        "source": r.get("source"),
        "archived": bool(r.get("archived")),
        "message_count": r.get("message_count") or 0,
        "tool_call_count": r.get("tool_call_count") or 0,
        "input_tokens": r.get("input_tokens") or 0,
        "output_tokens": r.get("output_tokens") or 0,
        "model": r.get("model"),
        "is_active": False,
        "cwd": r.get("cwd"),
        "git_branch": r.get("git_branch"),
        "git_repo_root": r.get("git_repo_root"),
    }


def _project_tree_inputs(
    db, session_limit: int, *, include_discovered: bool
) -> tuple[list[dict], list[dict], list[dict], str | None]:
    """Gather (sessions, projects, discovered_repos, active_id) for build_tree.

    ``include_discovered`` is the zero-session-repo overview tier; the entered
    view (drill-in) skips it entirely — it only needs the project it's showing,
    which already has sessions — avoiding the distinct-cwd scan + git probes on
    that per-turn path. One projects.db connection serves both reads.
    """
    rows = db.list_sessions_rich(
        limit=session_limit,
        offset=0,
        order_by_last_active=True,
        min_message_count=1,
        include_children=False,
        exclude_sources=_PROJECT_TREE_EXCLUDED_SOURCES,
        include_archived=False,
    )
    sessions = [_project_tree_row(r) for r in rows]
    # Parallel-warm the git cache so build_tree's resolver reads it instead of
    # cold-probing each cwd in sequence (matters on the drill-in path, which
    # skips the discovery warm-up below).
    git_probe.warm_roots(s["cwd"] for s in sessions if s.get("cwd"))

    from hermes_cli import projects_db as pdb

    with pdb.connect_closing() as conn:
        projects = [p.to_dict() for p in pdb.list_projects(conn)]
        active_id = pdb.get_active_id(conn)
        # backfill stays off the hot tree path — grouping uses the live resolver.
        discovered = _discover_repos_payload(db, conn=conn, backfill=False) if include_discovered else []

    return sessions, projects, discovered, active_id


def _build_project_tree(
    db, *, preview_limit: int, hydrate: bool, session_limit: int, include_discovered: bool
) -> tuple[dict, str | None]:
    """Gather inputs and run the one authoritative builder. Returns (tree, active_id)."""
    from tui_gateway import project_tree

    sessions, projects, discovered, active_id = _project_tree_inputs(
        db, session_limit, include_discovered=include_discovered
    )
    tree = project_tree.build_tree(
        projects,
        sessions,
        discovered,
        _resolve_cwd_git,
        preview_limit=preview_limit,
        hydrate=hydrate,
        is_junk_root=_is_repo_junk,
    )
    return tree, active_id


@method("projects.tree")
def _(rid, params: dict) -> dict:
    """Authoritative project overview: project -> repo -> lane structure with
    counts + a few preview sessions per project, plus the flat set of session
    ids claimed by any project (so the desktop excludes them from flat Recents).
    Lanes carry no session rows here; drill-in uses ``projects.project_sessions``.
    """
    try:
        db = _get_db()
        if db is None:
            return _ok(rid, {"projects": [], "active_id": None, "scoped_session_ids": []})

        tree, active_id = _build_project_tree(
            db,
            preview_limit=int(params.get("preview_limit") or 3),
            hydrate=False,
            session_limit=int(params.get("session_limit") or 2000),
            include_discovered=True,
        )
        return _ok(
            rid,
            {"projects": tree["projects"], "active_id": active_id, "scoped_session_ids": tree["scoped_session_ids"]},
        )
    except Exception as e:
        return _err(rid, 5061, str(e))


@method("projects.project_sessions")
def _(rid, params: dict) -> dict:
    """Fully hydrated lanes (repo -> lane -> session rows) for one project,
    built from the same authoritative grouping as ``projects.tree`` so ids and
    membership match exactly. Used when the user enters a project."""
    try:
        project_id = str(params.get("project_id") or "")
        if not project_id:
            return _err(rid, 5063, "project_id required")

        db = _get_db()
        if db is None:
            return _ok(rid, {"project": None})

        # Drill-in only needs the entered project (which has sessions), so skip
        # the zero-session discovery tier entirely.
        tree, _active = _build_project_tree(
            db, preview_limit=0, hydrate=True, session_limit=int(params.get("session_limit") or 5000),
            include_discovered=False,
        )
        proj = next((p for p in tree["projects"] if p["id"] == project_id), None)
        return _ok(rid, {"project": proj})
    except Exception as e:
        return _err(rid, 5061, str(e))


@method("config.get")
def _(rid, params: dict) -> dict:
    key = params.get("key", "")
    if key == "provider":
        try:
            from hermes_cli.models import list_available_providers, normalize_provider

            model = _resolve_model()
            parts = model.split("/", 1)
            return _ok(
                rid,
                {
                    "model": model,
                    "provider": (
                        normalize_provider(parts[0]) if len(parts) > 1 else "unknown"
                    ),
                    "providers": list_available_providers(),
                },
            )
        except Exception as e:
            return _err(rid, 5013, str(e))
    if key == "profile":
        from hermes_constants import display_hermes_home

        return _ok(rid, {"home": str(_hermes_home), "display": display_hermes_home()})
    if key == "project":
        cfg_terminal = _load_cfg().get("terminal") or {}
        raw = str(params.get("cwd", "") or cfg_terminal.get("cwd", "") or "").strip()
        cwd = _completion_cwd({"cwd": raw} if raw else {})
        return _ok(rid, {"cwd": cwd, "branch": _git_branch_for_cwd(cwd)})
    if key == "full":
        return _ok(rid, {"config": _load_cfg()})
    if key == "prompt":
        return _ok(rid, {"prompt": _load_cfg().get("custom_prompt", "")})
    if key == "skin":
        return _ok(
            rid, {"value": (_load_cfg().get("display") or {}).get("skin", "default")}
        )
    if key == "indicator":
        # Normalize so a hand-edited config.yaml with stray casing or
        # an unknown value reads back the SAME value the TUI actually
        # rendered (frontend's `normalizeIndicatorStyle` falls back to
        # `_INDICATOR_DEFAULT` for the same inputs).  Otherwise
        # `/indicator` would print one thing while the UI shows another.
        raw = (_load_cfg().get("display") or {}).get("tui_status_indicator", "")
        norm = str(raw).strip().lower()
        return _ok(
            rid,
            {"value": norm if norm in _INDICATOR_STYLES else _INDICATOR_DEFAULT},
        )
    if key == "personality":
        return _ok(
            rid,
            {"value": (_load_cfg().get("display") or {}).get("personality") or "none"},
        )
    if key == "reasoning":
        cfg = _load_cfg()
        effort = ""
        # Prefer the session's live value — `config.set reasoning` is
        # session-scoped, so the global key may not reflect this chat.
        session = _sessions.get(params.get("session_id", ""))
        live = getattr((session or {}).get("agent"), "reasoning_config", None)
        if live is None and session is not None:
            live = session.get("create_reasoning_override")
        if isinstance(live, dict):
            if live.get("enabled") is False:
                effort = "none"
            else:
                effort = str(live.get("effort", "") or "")
        if not effort:
            raw_effort = (cfg.get("agent") or {}).get("reasoning_effort", "")
            if raw_effort is False:
                # YAML `reasoning_effort: false`/`off`/`no` — thinking
                # disabled, not "unset, show the medium default".
                effort = "none"
            else:
                effort = str(raw_effort or "medium")
        display = (
            "show"
            if bool((cfg.get("display") or {}).get("show_reasoning", True))
            else "hide"
        )
        return _ok(rid, {"value": effort, "display": display})
    if key == "fast":
        return _ok(
            rid,
            {
                "value": (
                    "fast"
                    if (session := _sessions.get(params.get("session_id", "")))
                    and getattr(session.get("agent"), "service_tier", None)
                    == "priority"
                    else ("fast" if _load_service_tier() == "priority" else "normal")
                ),
            },
        )
    if key == "busy":
        return _ok(rid, {"value": _load_busy_input_mode()})
    if key == "details_mode":
        allowed_dm = frozenset({"hidden", "collapsed", "expanded"})
        raw = (
            str(
                (_load_cfg().get("display") or {}).get("details_mode", "collapsed")
                or "collapsed"
            )
            .strip()
            .lower()
        )
        nv = raw if raw in allowed_dm else "collapsed"
        return _ok(rid, {"value": nv})
    if key == "thinking_mode":
        allowed_tm = frozenset({"collapsed", "truncated", "full"})
        cfg = _load_cfg()
        raw = (
            str((cfg.get("display") or {}).get("thinking_mode", "") or "")
            .strip()
            .lower()
        )
        if raw in allowed_tm:
            nv = raw
        else:
            dm = (
                str(
                    (cfg.get("display") or {}).get("details_mode", "collapsed")
                    or "collapsed"
                )
                .strip()
                .lower()
            )
            nv = "full" if dm == "expanded" else "collapsed"
        return _ok(rid, {"value": nv})
    if key == "compact":
        on = bool((_load_cfg().get("display") or {}).get("tui_compact", False))
        return _ok(rid, {"value": "on" if on else "off"})
    if key == "statusbar":
        display = _load_cfg().get("display")
        raw = (
            display.get("tui_statusbar", "top") if isinstance(display, dict) else "top"
        )
        return _ok(rid, {"value": _coerce_statusbar(raw)})
    if key == "mouse":
        display = _load_cfg().get("display")
        return _ok(rid, {"value": _display_mouse_tracking(display)})
    if key == "mtime":
        cfg_path = _hermes_home / "config.yaml"
        try:
            return _ok(
                rid, {"mtime": cfg_path.stat().st_mtime if cfg_path.exists() else 0}
            )
        except Exception:
            return _ok(rid, {"mtime": 0})
    return _err(rid, 4002, f"unknown config key: {key}")


@method("setup.status")
def _(rid, params: dict) -> dict:
    try:
        from hermes_cli.main import _has_any_provider_configured

        return _ok(rid, {"provider_configured": bool(_has_any_provider_configured())})
    except Exception as e:
        return _err(rid, 5016, str(e))


@method("setup.runtime_check")
def _(rid, params: dict) -> dict:
    """Strict provider check: does the configured/default model actually resolve to a usable runtime?

    Unlike setup.status (which returns True if ANY provider auth state is
    discoverable, including indirect fallbacks like ``gh auth token`` for
    Copilot), this runs the same resolve_runtime_provider() call the agent
    uses on session creation. It returns ok=False with the auth error message
    when the user's configured model cannot actually be served, so UIs can
    surface onboarding before the user submits a doomed prompt.
    """
    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider
        from hermes_cli.auth import has_usable_secret
        from hermes_cli.main import _has_any_provider_configured

        requested = str(params.get("provider") or "").strip() or None
        runtime = resolve_runtime_provider(requested=requested)
        provider_configured = bool(_has_any_provider_configured())
        provider = runtime.get("provider") or "provider"
        source = str(runtime.get("source") or "")
        if (
            not provider_configured
            and provider == "bedrock"
            and source
            in {
                "iam-role",
                "aws-sdk-default-chain",
            }
        ):
            return _ok(
                rid,
                {
                    "ok": False,
                    "provider": provider,
                    "model": runtime.get("model"),
                    "source": source,
                    "error": "No Hermes provider is configured.",
                },
            )

        api_key = runtime.get("api_key")
        api_key_text = "" if callable(api_key) else str(api_key or "").strip()
        credential_ok = (
            callable(api_key)
            or api_key_text in {"aws-sdk", "no-key-required"}
            or has_usable_secret(api_key_text)
            or bool(runtime.get("command"))
        )

        if not credential_ok:
            return _ok(
                rid,
                {
                    "ok": False,
                    "provider": provider,
                    "model": runtime.get("model"),
                    "source": runtime.get("source"),
                    "error": f"No usable credentials found for {provider}.",
                },
            )

        return _ok(
            rid,
            {
                "ok": True,
                "provider": runtime.get("provider"),
                "model": runtime.get("model"),
                "source": runtime.get("source"),
            },
        )
    except Exception as e:
        return _ok(rid, {"ok": False, "error": str(e)})


# ── Methods: tools & system ──────────────────────────────────────────


@method("process.stop")
def _(rid, params: dict) -> dict:
    try:
        from tools.process_registry import process_registry

        return _ok(rid, {"killed": process_registry.kill_all()})
    except Exception as e:
        return _err(rid, 5010, str(e))


def _session_processes(session: dict) -> list:
    """Background processes owned by this session (registry session_key match)."""
    from tools.process_registry import process_registry

    key = str(session.get("session_key") or "")
    owned = []
    for entry in process_registry.list_sessions():
        proc = process_registry.get(entry["session_id"])
        if proc is None or str(getattr(proc, "session_key", "") or "") != key:
            continue
        # The 200-char list preview is too thin for the desktop's inline
        # terminal viewer — ship a real tail alongside it.
        entry["output_tail"] = (proc.output_buffer or "")[-4000:]
        owned.append(entry)
    return owned


@method("process.list")
def _(rid, params: dict) -> dict:
    """Session-scoped view of the background process registry (desktop status stack)."""
    session, err = _sess(params, rid)
    if err:
        return err
    try:
        return _ok(rid, {"processes": _session_processes(session)})
    except Exception as e:
        return _err(rid, 5010, str(e))


@method("process.kill")
def _(rid, params: dict) -> dict:
    """Kill ONE background process — scoped to the caller's session so one
    window can't reap another session's work (unlike process.stop's kill_all)."""
    session, err = _sess(params, rid)
    if err:
        return err
    proc_id = str(params.get("process_id") or "")
    if not proc_id:
        return _err(rid, 4012, "process_id required")
    try:
        from tools.process_registry import process_registry

        proc = process_registry.get(proc_id)
        if proc is None or str(getattr(proc, "session_key", "") or "") != str(
            session.get("session_key") or ""
        ):
            return _err(rid, 4044, f"no such process: {proc_id}")
        return _ok(rid, process_registry.kill_process(proc_id))
    except Exception as e:
        return _err(rid, 5010, str(e))


@method("reload.mcp")
def _(rid, params: dict) -> dict:
    session = _sessions.get(params.get("session_id", ""))
    try:
        # Gate: /reload-mcp invalidates the prompt cache for this session.
        # Respect the ``approvals.mcp_reload_confirm`` config toggle — if
        # set (default true) AND the caller did not pass ``confirm=true``
        # in params, surface a warning to the transcript instead of just
        # reloading silently.  Users pass confirm=true either by
        # re-invoking after reading the warning, or by setting the
        # config key to false permanently.
        user_confirm = bool(params.get("confirm", False))
        if not user_confirm:
            try:
                from hermes_cli.config import load_config as _load_config

                _cfg = _load_config()
                _approvals = _cfg.get("approvals") if isinstance(_cfg, dict) else None
                _confirm_required = True
                if isinstance(_approvals, dict):
                    _confirm_required = bool(_approvals.get("mcp_reload_confirm", True))
            except Exception:
                _confirm_required = True
            if _confirm_required:
                # Return a structured response the Ink client can surface
                # as a warning/confirmation without actually reloading yet.
                # Ink's ops.ts reads ``status`` and prints ``message`` to
                # the transcript; a follow-up invocation with confirm=true
                # (or an `always` choice that flips the config) proceeds.
                return _ok(
                    rid,
                    {
                        "status": "confirm_required",
                        "message": (
                            "⚠️  /reload-mcp invalidates the prompt cache (next "
                            "message re-sends full input tokens). Reply `/reload-mcp "
                            "now` to proceed, or `/reload-mcp always` to proceed and "
                            "silence this prompt permanently."
                        ),
                    },
                )

        from tools.mcp_tool import shutdown_mcp_servers, discover_mcp_tools

        shutdown_mcp_servers()
        discover_mcp_tools()
        if session:
            agent = session["agent"]
            # Rebuild the cached agent's tool snapshot so the current session
            # picks up added/removed MCP tools without `/new` (which discards
            # history).  The agent snapshots tools once at build and never
            # re-reads the registry, so an explicit rebuild is required here.
            # The user already consented to the prompt-cache invalidation via
            # the confirm gate above.  Mirrors gateway/run.py::_execute_mcp_reload.
            try:
                from tools.mcp_tool import refresh_agent_mcp_tools

                # Explicit reload: re-resolve enabled toolsets so a server the
                # user just enabled in config this session is picked up.
                refresh_agent_mcp_tools(
                    agent,
                    enabled_override=_load_enabled_toolsets(),
                    quiet_mode=True,
                )
            except Exception as _exc:
                logger.warning(
                    "Failed to refresh cached agent tools after /reload-mcp: %s",
                    _exc,
                )
            _emit(
                "session.info",
                params.get("session_id", ""),
                _session_info(agent, session),
            )

        # Honor `always=true` by persisting the opt-out to config.
        if bool(params.get("always", False)):
            try:
                from cli import save_config_value as _save_cfg

                _save_cfg("approvals.mcp_reload_confirm", False)
            except Exception as _exc:
                logger.warning("Failed to persist mcp_reload_confirm=false: %s", _exc)

        return _ok(rid, {"status": "reloaded"})
    except Exception as e:
        return _err(rid, 5015, str(e))


@method("reload.env")
def _(rid, params: dict) -> dict:
    """Re-read ``~/.hermes/.env`` into the gateway process via
    ``hermes_cli.config.reload_env``, matching classic CLI's ``/reload``
    handler.  Newly added API keys take effect on the next agent call
    without restarting the TUI.

    The credential pool / provider routing for any *already-constructed*
    agent does not auto-rebuild — that's the same behaviour as classic
    CLI's ``/reload``.  Users who want a brand-new credential resolution
    should follow with ``/new``.
    """
    try:
        from hermes_cli.config import reload_env

        count = reload_env()
        return _ok(rid, {"updated": int(count)})
    except Exception as e:
        return _err(rid, 5015, str(e))


_TUI_HIDDEN: frozenset[str] = frozenset({
    "sethome",
    "set-home",
    "commands",
    "approve",
    "deny",
})

_TUI_EXTRA: list[tuple[str, str, str]] = [
    ("/compact", "Toggle compact display mode", "TUI"),
    ("/logs", "Show recent gateway log lines", "TUI"),
    (
        "/mouse",
        "Set mouse tracking preset [on|off|toggle|wheel|buttons|all]",
        "TUI",
    ),
    ("/sessions", "Switch between live TUI sessions", "TUI"),
]

# Commands that queue messages onto _pending_input in the CLI.
# In the TUI the slash worker subprocess has no reader for that queue,
# so slash.exec routes them to command.dispatch internally (which handles
# them and returns a structured payload) instead of erroring out and
# relying on a client-side fallback. See #48848.
_PENDING_INPUT_COMMANDS: frozenset[str] = frozenset(
    {
        "retry",
        "queue",
        "q",
        "steer",
        "plan",
        "goal",
        "moa",
        "undo",
        "learn",
        "compress",
        "compact",
    }
)

_WORKER_BLOCKED_COMMANDS: frozenset[str] = frozenset({"snapshot", "snap"})


@method("commands.catalog")
def _(rid, params: dict) -> dict:
    """Registry-backed slash metadata for the TUI — categorized, no aliases."""
    try:
        from hermes_cli.commands import (
            COMMAND_REGISTRY,
            SUBCOMMANDS,
            _build_description,
        )

        all_pairs: list[list[str]] = []
        canon: dict[str, str] = {}
        categories: list[dict] = []
        cat_map: dict[str, list[list[str]]] = {}
        cat_order: list[str] = []

        for cmd in COMMAND_REGISTRY:
            if cmd.name in _TUI_HIDDEN or cmd.gateway_only:
                continue

            c = f"/{cmd.name}"
            canon[c.lower()] = c
            for a in cmd.aliases:
                canon[f"/{a}".lower()] = c

            desc = _build_description(cmd)
            all_pairs.append([c, desc])

            cat = cmd.category
            if cat not in cat_map:
                cat_map[cat] = []
                cat_order.append(cat)
            cat_map[cat].append([c, desc])

        for name, desc, cat in _TUI_EXTRA:
            all_pairs.append([name, desc])
            if cat not in cat_map:
                cat_map[cat] = []
                cat_order.append(cat)
            cat_map[cat].append([name, desc])

        warning = ""
        try:
            qcmds = _load_cfg().get("quick_commands", {}) or {}
            if isinstance(qcmds, dict) and qcmds:
                bucket = "User commands"
                if bucket not in cat_map:
                    cat_map[bucket] = []
                    cat_order.append(bucket)
                for qname, qc in sorted(qcmds.items()):
                    if not isinstance(qc, dict):
                        continue
                    key = f"/{qname}"
                    canon[key.lower()] = key
                    qtype = qc.get("type", "")
                    if qtype == "exec":
                        default_desc = f"exec: {qc.get('command', '')}"
                    elif qtype == "alias":
                        default_desc = f"alias → {qc.get('target', '')}"
                    else:
                        default_desc = qtype or "quick command"
                    qdesc = str(qc.get("description") or default_desc)
                    qdesc = qdesc[:120] + ("…" if len(qdesc) > 120 else "")
                    all_pairs.append([key, qdesc])
                    cat_map[bucket].append([key, qdesc])
        except Exception as e:
            if not warning:
                warning = f"quick_commands discovery unavailable: {e}"

        skill_count = 0
        try:
            from agent.skill_commands import scan_skill_commands

            for k, info in sorted(scan_skill_commands().items()):
                d = str(info.get("description", "Skill"))
                all_pairs.append([k, d[:120] + ("…" if len(d) > 120 else "")])
                skill_count += 1
        except Exception as e:
            warning = f"skill discovery unavailable: {e}"

        for cat in cat_order:
            categories.append({"name": cat, "pairs": cat_map[cat]})

        sub = {k: v[:] for k, v in SUBCOMMANDS.items()}
        return _ok(
            rid,
            {
                "pairs": all_pairs,
                "sub": sub,
                "canon": canon,
                "categories": categories,
                "skill_count": skill_count,
                "warning": warning,
            },
        )
    except Exception as e:
        return _err(rid, 5020, str(e))


def _cli_exec_blocked(argv: list[str]) -> str | None:
    """Return user hint if this argv must not run headless in the gateway process."""
    if not argv:
        return "bare `hermes` is interactive — use `/hermes chat -q …` or run `hermes` in another terminal"
    a0 = argv[0].lower()
    if a0 == "setup":
        return "`hermes setup` needs a full terminal — run it outside the TUI"
    if a0 == "gateway":
        return "`hermes gateway` is long-running — run it in another terminal"
    if a0 == "sessions" and len(argv) > 1 and argv[1].lower() == "browse":
        return "`hermes sessions browse` is interactive — use /resume here, or run browse in another terminal"
    if a0 == "config" and len(argv) > 1 and argv[1].lower() == "edit":
        return "`hermes config edit` needs $EDITOR in a real terminal"
    return None


@method("cli.exec")
def _(rid, params: dict) -> dict:
    """Run `python -m hermes_cli.main` with argv; capture stdout/stderr (non-interactive only)."""
    argv = params.get("argv", [])
    if not isinstance(argv, list) or not all(isinstance(x, str) for x in argv):
        return _err(rid, 4003, "argv must be list[str]")
    hint = _cli_exec_blocked(argv)
    if hint:
        return _ok(rid, {"blocked": True, "hint": hint, "code": -1, "output": ""})
    try:
        r = subprocess.run(
            [sys.executable, "-m", "hermes_cli.main", *argv],
            capture_output=True,
            text=True,
            timeout=min(int(params.get("timeout", 240)), 600),
            cwd=os.getcwd(),
            # cli.exec runs `python -m hermes_cli.main` (can drive the agent) →
            # needs provider credentials. Tier-1 secrets still stripped (#29157).
            env=hermes_subprocess_env(inherit_credentials=True),
            stdin=subprocess.DEVNULL,
        )
        parts = [r.stdout or "", r.stderr or ""]
        out = "\n".join(p for p in parts if p).strip() or "(no output)"
        return _ok(
            rid, {"blocked": False, "code": r.returncode, "output": out[:48_000]}
        )
    except subprocess.TimeoutExpired:
        return _err(rid, 5016, "cli.exec: timeout")
    except Exception as e:
        return _err(rid, 5017, str(e))


@method("command.resolve")
def _(rid, params: dict) -> dict:
    try:
        from hermes_cli.commands import resolve_command

        r = resolve_command(params.get("name", ""))
        if r:
            return _ok(
                rid,
                {
                    "canonical": r.name,
                    "description": r.description,
                    "category": r.category,
                },
            )
        return _err(rid, 4011, f"unknown command: {params.get('name')}")
    except Exception as e:
        return _err(rid, 5012, str(e))


def _resolve_name(name: str) -> str:
    try:
        from hermes_cli.commands import resolve_command

        r = resolve_command(name)
        return r.name if r else name
    except Exception:
        return name


@method("command.dispatch")
def _(rid, params: dict) -> dict:
    name, arg = params.get("name", "").lstrip("/"), params.get("arg", "")
    resolved = _resolve_name(name)
    if resolved != name:
        name = resolved
    session = _sessions.get(params.get("session_id", ""))

    qcmds = _load_cfg().get("quick_commands", {})
    if name in qcmds:
        qc = qcmds[name]
        if qc.get("type") == "exec":
            # Sanitize env to prevent credential leakage —
            # quick commands run in the TUI server process which
            # has all API keys in os.environ.
            from tools.environments.local import _sanitize_subprocess_env
            sanitized_env = _sanitize_subprocess_env(os.environ.copy())
            r = subprocess.run(
                qc.get("command", ""),
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
                stdin=subprocess.DEVNULL,
                env=sanitized_env,
            )
            output = (
                (r.stdout or "")
                + ("\n" if r.stdout and r.stderr else "")
                + (r.stderr or "")
            ).strip()[:4000]
            if output:
                from agent.redact import redact_sensitive_text
                output = redact_sensitive_text(output)
            if r.returncode != 0:
                return _err(
                    rid,
                    4018,
                    output or f"quick command failed with exit code {r.returncode}",
                )
            return _ok(rid, {"type": "exec", "output": output})
        if qc.get("type") == "alias":
            return _ok(rid, {"type": "alias", "target": qc.get("target", "")})

    try:
        from hermes_cli.plugins import (
            get_plugin_command_handler,
            resolve_plugin_command_result,
        )

        handler = get_plugin_command_handler(name)
        if handler:
            result = resolve_plugin_command_result(handler(arg))
            return _ok(rid, {"type": "plugin", "output": str(result or "")})
    except Exception:
        pass

    try:
        from agent.skill_commands import (
            scan_skill_commands,
            build_skill_invocation_message,
        )

        cmds = scan_skill_commands()
        key = f"/{name}"
        if key in cmds:
            msg = build_skill_invocation_message(
                key, arg, task_id=session.get("session_key", "") if session else ""
            )
            if msg:
                return _ok(
                    rid,
                    {
                        "type": "skill",
                        "message": msg,
                        "name": cmds[key].get("name", name),
                    },
                )
    except Exception:
        pass

    # ── Commands that queue messages onto _pending_input in the CLI ───
    # In the TUI the slash worker subprocess has no reader for that queue,
    # so we handle them here and return a structured payload.

    if name in {"queue", "q"}:
        if not arg:
            return _err(rid, 4004, "usage: /queue <prompt>")
        return _ok(rid, {"type": "send", "message": arg})

    if name == "learn":
        # Open-ended: build the standards-guided prompt and submit it as a
        # normal agent turn. The live agent gathers whatever the user
        # described (dirs, URLs, this conversation, pasted text) with its own
        # tools and authors the skill via skill_manage. Works on any backend.
        from agent.learn_prompt import build_learn_prompt

        return _ok(rid, {"type": "send", "message": build_learn_prompt(arg)})
    if name == "moa":
        # /moa is one-shot sugar only: run a single prompt through the default
        # MoA preset, then restore the prior model. To *switch* to a MoA preset
        # for the rest of the session, pick it from the model picker (MoA
        # presets surface as a virtual "Mixture of Agents" provider).
        try:
            from hermes_cli.moa_config import moa_usage, normalize_moa_config

            if not arg:
                return _err(rid, 4004, moa_usage())
            if not session:
                return _err(rid, 4001, "no active session")
            sid = params.get("session_id", "")
            moa_cfg = normalize_moa_config(_load_cfg().get("moa") or {})
            preset = moa_cfg["default_preset"]
            # Record the live model identity so it can be restored after the
            # one-shot turn, then swap the agent's client in place (#53444:
            # setting session["model_override"] alone never switched the
            # already-built agent, so the turn silently ran on the old model).
            agent = session.get("agent")
            session["moa_one_shot_restore"] = {
                "override": session.get("model_override"),
                "model": getattr(agent, "model", None) if agent else None,
                "provider": getattr(agent, "provider", None) if agent else None,
            }
            if agent is not None:
                # Live agent: swap its client in place so THIS turn runs MoA.
                try:
                    _apply_model_switch(
                        sid,
                        session,
                        f"{preset} --provider moa",
                        confirm_expensive_model=False,
                        pin_session_override=True,
                        # One-shot turn-scoped swap — never persist the MoA
                        # virtual provider to config.yaml.
                        persist_override=False,
                    )
                except Exception as exc:
                    session.pop("moa_one_shot_restore", None)
                    return _err(rid, 5030, f"moa unavailable: {exc}")
            else:
                # No agent built yet (lazy/fresh session): the override is
                # consumed by the first build, so the turn runs MoA without an
                # in-place switch.
                session["model_override"] = {
                    "provider": "moa",
                    "model": preset,
                    "base_url": "moa://local",
                    "api_key": "moa-virtual-provider",
                    "api_mode": "chat_completions",
                }
            return _ok(
                rid,
                {
                    "type": "send",
                    "notice": f"MoA one-shot queued with preset {preset}; previous model will be restored after this turn.",
                    "message": arg,
                },
            )
        except Exception as exc:
            return _err(rid, 5030, f"moa unavailable: {exc}")

    if name == "retry":
        if not session:
            return _err(rid, 4001, "no active session to retry")
        if session.get("running"):
            return _err(
                rid, 4009, "session busy — /interrupt the current turn before /retry"
            )
        history = session.get("history", [])
        if not history:
            return _err(rid, 4018, "no previous user message to retry")
        # Walk backwards to find the last user message
        last_user_idx = None
        for i in range(len(history) - 1, -1, -1):
            if history[i].get("role") == "user":
                last_user_idx = i
                break
        if last_user_idx is None:
            return _err(rid, 4018, "no previous user message to retry")
        content = history[last_user_idx].get("content", "")
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
        if not content:
            return _err(rid, 4018, "last user message is empty")
        # Truncate history: remove everything from the last user message onward
        # (mirrors CLI retry_last() which strips the failed exchange)
        with session["history_lock"]:
            session["history"] = history[:last_user_idx]
            session["history_version"] = int(session.get("history_version", 0)) + 1
        return _ok(rid, {"type": "send", "message": content})

    if name == "steer":
        if not arg:
            return _err(rid, 4004, "usage: /steer <prompt>")
        agent = session.get("agent") if session else None
        if agent and hasattr(agent, "steer"):
            try:
                accepted = agent.steer(arg)
                if accepted:
                    return _ok(
                        rid,
                        {
                            "type": "exec",
                            "output": f"⏩ Steer queued — arrives after the next tool call: {arg[:80]}{'...' if len(arg) > 80 else ''}",
                        },
                    )
            except Exception:
                pass
        # Fallback: no active run, treat as next-turn message
        return _ok(rid, {"type": "send", "message": arg})

    if name == "goal":
        if not session:
            return _err(rid, 4001, "no active session")
        try:
            from hermes_cli.goals import GoalManager
        except Exception as exc:
            return _err(rid, 5030, f"goals unavailable: {exc}")

        sid_key = session.get("session_key") or ""
        if not sid_key:
            return _err(rid, 4001, "no session key")

        try:
            goals_cfg = _load_cfg().get("goals") or {}
            max_turns = int(goals_cfg.get("max_turns", 20) or 20)
        except Exception:
            max_turns = 20
        mgr = GoalManager(session_id=sid_key, default_max_turns=max_turns)

        lower = arg.strip().lower()
        if not arg.strip() or lower == "status":
            return _ok(rid, {"type": "exec", "output": mgr.status_line()})
        if lower == "pause":
            state = mgr.pause(reason="user-paused")
            out = "No goal set." if state is None else f"⏸ Goal paused: {state.goal}"
            return _ok(rid, {"type": "exec", "output": out})
        if lower == "resume":
            state = mgr.resume()
            if state is None:
                return _ok(rid, {"type": "exec", "output": "No goal to resume."})
            return _ok(
                rid,
                {
                    "type": "exec",
                    "output": (
                        f"▶ Goal resumed: {state.goal}\n"
                        "Send any message to continue, or wait — I'll take the next step on the next turn."
                    ),
                },
            )
        if lower in {"clear", "stop", "done"}:
            had = mgr.has_goal()
            mgr.clear()
            return _ok(
                rid,
                {
                    "type": "exec",
                    "output": "✓ Goal cleared." if had else "No active goal.",
                },
            )

        # Otherwise — treat the remaining text as the new goal.
        try:
            state = mgr.set(arg)
        except ValueError as exc:
            return _err(rid, 4004, f"invalid goal: {exc}")

        notice = (
            f"⊙ Goal set ({state.max_turns}-turn budget): {state.goal}\n"
            "I'll keep working until the goal is done, you pause/clear it, or the budget is exhausted.\n"
            "Controls: /goal status · /goal pause · /goal resume · /goal clear"
        )
        # Send the goal text as the kickoff prompt. The TUI client sees
        # {type: send, notice, message} → renders `notice` as a sys line,
        # then submits `message` as a user turn. The post-turn judge
        # wired in _run_prompt_submit takes over from there.
        return _ok(
            rid,
            {"type": "send", "notice": notice, "message": state.goal},
        )

    if name == "undo":
        # /undo [N]: back up N user turns (default 1), soft-delete the
        # truncated rows on disk, and prefill the composer with the text
        # of the user message we backed up to so it can be edited and
        # resubmitted. N=1 is the Claude-Code-style single-step undo;
        # /undo 3 backs up three user turns at once. See issue #21910.
        if not session:
            return _err(rid, 4001, "no active session to undo")
        if session.get("running"):
            return _err(
                rid, 4009, "session busy — /interrupt the current turn before /undo"
            )
        db = _get_db()
        if db is None:
            return _db_unavailable_error(rid, code=5008)
        session_key = session.get("session_key", "")
        if not session_key:
            return _err(rid, 4001, "no session key for undo")
        # Parse the optional count argument (e.g. "/undo 3" → 3).
        n = 1
        arg_str = (arg or "").strip()
        if arg_str:
            try:
                n = int(arg_str.split()[0])
            except (ValueError, IndexError):
                return _err(
                    rid, 4004, f"undo: invalid count {arg_str!r} — use /undo or /undo N"
                )
        if n < 1:
            n = 1
        try:
            recents = db.list_recent_user_messages(session_key, limit=max(n, 10))
        except Exception as e:
            return _err(rid, 5008, f"undo: failed to load history: {e}")
        if not recents:
            return _err(rid, 4018, "no user messages to undo")
        # recents[0] is the most-recent user turn; pick the Nth-from-last.
        # If N exceeds the number of user turns, back up to the oldest.
        target_idx = min(n - 1, len(recents) - 1)
        target_id = recents[target_idx]["id"]
        try:
            result = db.rewind_to_message(session_key, target_id)
        except ValueError as e:
            return _err(rid, 4004, f"undo: {e}")
        except Exception as e:
            return _err(rid, 5008, f"undo: {e}")
        # Reload the active-only transcript into the in-memory session
        # history so subsequent turns see the truncated view.
        try:
            active = db.get_messages_as_conversation(session_key)
        except Exception:
            active = []
        with session["history_lock"]:
            session["history"] = list(active)
            session["history_version"] = int(session.get("history_version", 0)) + 1
        # Notify memory providers — same hook /branch fires, plus the
        # rewound flag so providers caching per-turn document state
        # know to invalidate. See #6672 + #21910.
        agent = session.get("agent")
        if agent is not None:
            mm = getattr(agent, "_memory_manager", None)
            if mm is not None:
                try:
                    mm.on_session_switch(
                        session_key,
                        parent_session_id="",
                        reset=False,
                        rewound=True,
                    )
                except Exception:
                    pass
            if hasattr(agent, "_invalidate_system_prompt"):
                try:
                    agent._invalidate_system_prompt()
                except Exception:
                    pass
            if hasattr(agent, "_last_flushed_db_idx"):
                try:
                    agent._last_flushed_db_idx = len(active)
                except Exception:
                    pass
        target_msg = result.get("target_message") or {}
        target_text = target_msg.get("content") or ""
        if isinstance(target_text, list):
            parts = [
                p.get("text", "")
                for p in target_text
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            target_text = "\n".join(t for t in parts if t)
        if not isinstance(target_text, str):
            target_text = ""
        rewound_count = result.get("rewound_count", 0)
        turns_undone = target_idx + 1
        turn_word = "turn" if turns_undone == 1 else "turns"
        notice = (
            f"↶ Undid {turns_undone} {turn_word} ({rewound_count} message(s)). "
            "Edit and resubmit, or send a new message."
        )
        return _ok(
            rid,
            {"type": "prefill", "message": target_text, "notice": notice},
        )

    if name in {"snapshot", "snap"}:
        subcommand = arg.split(maxsplit=1)[0].lower() if arg else ""
        if subcommand in {"restore", "rewind"}:
            return _ok(
                rid,
                {
                    "type": "exec",
                    "output": (
                        "/snapshot restore is blocked in the TUI because it changes "
                        "config/state on disk while the live agent has cached settings. "
                        "Run it in the classic CLI, then restart the TUI."
                    ),
                },
            )

    if name in {"compress", "compact"}:
        if not session:
            return _err(rid, 4001, "no active session to compress")
        if session.get("running"):
            return _err(
                rid, 4009, "session busy — /interrupt the current turn before /compress"
            )
        sid = params.get("session_id", "")
        try:
            from agent.manual_compression_feedback import summarize_manual_compression
            from agent.model_metadata import estimate_request_tokens_rough

            with session["history_lock"]:
                before_messages = list(session.get("history", []))
                history_version = int(session.get("history_version", 0))
            before_count = len(before_messages)
            _agent = session["agent"]
            _sys_prompt = getattr(_agent, "_cached_system_prompt", "") or ""
            _tools = getattr(_agent, "tools", None) or None
            before_tokens = (
                estimate_request_tokens_rough(
                    before_messages, system_prompt=_sys_prompt, tools=_tools
                )
                if before_count
                else 0
            )
            removed, usage = _compress_session_history(
                session,
                arg.strip() or None,
                approx_tokens=before_tokens,
                before_messages=before_messages,
                history_version=history_version,
            )
            with session["history_lock"]:
                after_messages = list(session.get("history", []))
            after_count = len(after_messages)
            _sys_prompt_after = (
                getattr(_agent, "_cached_system_prompt", "") or _sys_prompt
            )
            _tools_after = getattr(_agent, "tools", None) or _tools
            after_tokens = (
                estimate_request_tokens_rough(
                    after_messages,
                    system_prompt=_sys_prompt_after,
                    tools=_tools_after,
                )
                if after_count
                else 0
            )
            _sync_session_key_after_compress(sid, session)
            summary = summarize_manual_compression(
                before_messages, after_messages, before_tokens, after_tokens
            )
            _emit("session.info", sid, _session_info(session.get("agent"), session))
            return _ok(
                rid,
                {
                    "type": "exec",
                    "output": "\n".join(
                        filter(None, [summary["headline"], summary["token_line"], summary.get("note")])
                    ),
                },
            )
        except Exception as exc:
            return _err(rid, 5009, f"compress failed: {exc}")

    return _err(rid, 4018, f"not a quick/plugin/skill command: {name}")


# ── Methods: paste ────────────────────────────────────────────────────

_paste_counter = 0


@method("paste.collapse")
def _(rid, params: dict) -> dict:
    global _paste_counter
    text = params.get("text", "")
    if not text:
        return _err(rid, 4004, "empty paste")

    _paste_counter += 1
    line_count = text.count("\n") + 1
    paste_dir = _hermes_home / "pastes"
    paste_dir.mkdir(parents=True, exist_ok=True)

    from datetime import datetime

    paste_file = (
        paste_dir / f"paste_{_paste_counter}_{datetime.now().strftime('%H%M%S')}.txt"
    )
    paste_file.write_text(text, encoding="utf-8")

    placeholder = (
        f"[Pasted text #{_paste_counter}: {line_count} lines \u2192 {paste_file}]"
    )
    return _ok(
        rid, {"placeholder": placeholder, "path": str(paste_file), "lines": line_count}
    )


# ── Methods: complete ─────────────────────────────────────────────────

_FUZZY_CACHE_TTL_S = 5.0
_FUZZY_CACHE_MAX_FILES = 20000
_FUZZY_FALLBACK_EXCLUDES = frozenset({
    ".git",
    ".hg",
    ".svn",
    ".next",
    ".cache",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    "dist",
    "build",
    "target",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
})
_fuzzy_cache_lock = threading.Lock()
_fuzzy_cache: dict[str, tuple[float, list[str]]] = {}


def _list_repo_files(root: str) -> list[str]:
    """Return file paths relative to ``root``.

    Uses ``git ls-files`` from the repo top (resolved via
    ``rev-parse --show-toplevel``) so the listing covers tracked + untracked
    files anywhere in the repo, then converts each path back to be relative
    to ``root``. Files outside ``root`` (parent directories of cwd, sibling
    subtrees) are excluded so the picker stays scoped to what's reachable
    from the gateway's cwd. Falls back to a bounded ``os.walk(root)`` when
    ``root`` isn't inside a git repo. Result cached per-root for
    ``_FUZZY_CACHE_TTL_S`` so rapid keystrokes don't respawn git processes.
    """
    now = time.monotonic()
    with _fuzzy_cache_lock:
        cached = _fuzzy_cache.get(root)
        if cached and now - cached[0] < _FUZZY_CACHE_TTL_S:
            return cached[1]

    files: list[str] = []
    from hermes_cli._subprocess_compat import windows_hide_flags

    _creationflags = windows_hide_flags()
    try:
        top_result = subprocess.run(
            ["git", "-C", root, "rev-parse", "--show-toplevel"],
            capture_output=True,
            timeout=2.0,
            check=False,
            stdin=subprocess.DEVNULL,
            creationflags=_creationflags,
        )
        if top_result.returncode == 0:
            top = top_result.stdout.decode("utf-8", "replace").strip()
            list_result = subprocess.run(
                [
                    "git",
                    "-C",
                    top,
                    "ls-files",
                    "-z",
                    "--cached",
                    "--others",
                    "--exclude-standard",
                ],
                capture_output=True,
                timeout=2.0,
                check=False,
                stdin=subprocess.DEVNULL,
                creationflags=_creationflags,
            )
            if list_result.returncode == 0:
                for p in list_result.stdout.decode("utf-8", "replace").split("\0"):
                    if not p:
                        continue
                    rel = os.path.relpath(os.path.join(top, p), root).replace(
                        os.sep, "/"
                    )
                    # Skip parents/siblings of cwd — keep the picker scoped
                    # to root-and-below, matching Cmd-P workspace semantics.
                    if rel.startswith("../"):
                        continue
                    files.append(rel)
                    if len(files) >= _FUZZY_CACHE_MAX_FILES:
                        break
    except (OSError, subprocess.TimeoutExpired):
        pass

    if not files:
        # Fallback walk: skip vendor/build dirs + dot-dirs so the walk stays
        # tractable. Dotfiles themselves survive — the ranker decides based
        # on whether the query starts with `.`.
        try:
            for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
                dirnames[:] = [
                    d
                    for d in dirnames
                    if d not in _FUZZY_FALLBACK_EXCLUDES and not d.startswith(".")
                ]
                rel_dir = os.path.relpath(dirpath, root)
                for f in filenames:
                    rel = f if rel_dir == "." else f"{rel_dir}/{f}"
                    files.append(rel.replace(os.sep, "/"))
                    if len(files) >= _FUZZY_CACHE_MAX_FILES:
                        break
                if len(files) >= _FUZZY_CACHE_MAX_FILES:
                    break
        except OSError:
            pass

    with _fuzzy_cache_lock:
        _fuzzy_cache[root] = (now, files)

    return files


def _fuzzy_basename_rank(name: str, query: str) -> tuple[int, int] | None:
    """Rank ``name`` against ``query``; lower is better. Returns None to reject.

    Tiers (kind):
      0 — exact basename
      1 — basename prefix (e.g. `app` → `appChrome.tsx`)
      2 — word-boundary / camelCase hit (e.g. `chrome` → `appChrome.tsx`)
      3 — substring anywhere in basename
      4 — subsequence match (every query char appears in order)

    Secondary key is `len(name)` so shorter names win ties.
    """
    if not query:
        return (3, len(name))

    nl = name.lower()
    ql = query.lower()

    if nl == ql:
        return (0, len(name))

    if nl.startswith(ql):
        return (1, len(name))

    # Word-boundary split: `foo-bar_baz.qux` → ["foo","bar","baz","qux"].
    # camelCase split: `appChrome` → ["app","Chrome"]. Cheap approximation;
    # falls through to substring/subsequence if it misses.
    parts: list[str] = []
    buf = ""
    for ch in name:
        if ch in "-_." or (ch.isupper() and buf and not buf[-1].isupper()):
            if buf:
                parts.append(buf)
            buf = ch if ch not in "-_." else ""
        else:
            buf += ch
    if buf:
        parts.append(buf)
    for p in parts:
        if p.lower().startswith(ql):
            return (2, len(name))

    if ql in nl:
        return (3, len(name))

    i = 0
    for ch in nl:
        if ch == ql[i]:
            i += 1
            if i == len(ql):
                return (4, len(name))

    return None


@method("complete.path")
def _(rid, params: dict) -> dict:
    word = params.get("word", "")
    if not word:
        return _ok(rid, {"items": []})

    items: list[dict] = []
    try:
        root = _completion_cwd(params)
        is_context = word.startswith("@")
        query = word[1:] if is_context else word

        if is_context and not query:
            items = [
                {"text": "@diff", "display": "@diff", "meta": "git diff"},
                {"text": "@staged", "display": "@staged", "meta": "staged diff"},
                {"text": "@file:", "display": "@file:", "meta": "attach file"},
                {"text": "@folder:", "display": "@folder:", "meta": "attach folder"},
                {"text": "@url:", "display": "@url:", "meta": "fetch url"},
                {"text": "@git:", "display": "@git:", "meta": "git log"},
            ]
            return _ok(rid, {"items": items})

        # Accept both `@folder:path` and the bare `@folder` form so the user
        # sees directory listings as soon as they finish typing the keyword,
        # without first accepting the static `@folder:` hint.
        if is_context and query in {"file", "folder"}:
            prefix_tag, path_part = query, ""
        elif is_context and query.startswith(("file:", "folder:")):
            prefix_tag, _, tail = query.partition(":")
            path_part = tail
        else:
            prefix_tag = ""
            path_part = query if is_context else query

        # Fuzzy basename search across the repo when the user types a bare
        # name with no path separator — `@appChrome` surfaces every file
        # whose basename matches, regardless of directory depth. Matches what
        # editors like Cursor / VS Code do for Cmd-P. Path-ish queries (with
        # `/`, `./`, `~/`, `/abs`) fall through to the directory-listing
        # path so explicit navigation intent is preserved.
        if (
            is_context
            and path_part
            and len(path_part.strip()) >= 2
            and "/" not in path_part
            and prefix_tag != "folder"
        ):
            ranked: list[tuple[tuple[int, int], str, str]] = []
            for rel in _list_repo_files(root):
                basename = os.path.basename(rel)
                if basename.startswith(".") and not path_part.startswith("."):
                    continue
                rank = _fuzzy_basename_rank(basename, path_part)
                if rank is None:
                    continue
                ranked.append((rank, rel, basename))

            ranked.sort(key=lambda r: (r[0], len(r[1]), r[1]))
            tag = prefix_tag or "file"
            for _, rel, basename in ranked[:30]:
                items.append({
                    "text": f"@{tag}:{rel}",
                    "display": basename,
                    "meta": os.path.dirname(rel),
                })

            return _ok(rid, {"items": items})

        expanded = _normalize_completion_path(path_part) if path_part else "."
        if expanded == "." or not expanded:
            search_dir, match = ".", ""
        elif expanded.endswith("/"):
            search_dir, match = expanded, ""
        else:
            search_dir = os.path.dirname(expanded) or "."
            match = os.path.basename(expanded)

        search_dir = (
            search_dir if os.path.isabs(search_dir) else os.path.join(root, search_dir)
        )
        if not os.path.isdir(search_dir):
            return _ok(rid, {"items": []})

        want_dir = prefix_tag == "folder"
        match_lower = match.lower()
        for entry in sorted(os.listdir(search_dir)):
            if match and not entry.lower().startswith(match_lower):
                continue
            if is_context and entry in _FUZZY_FALLBACK_EXCLUDES:
                continue
            if is_context and not prefix_tag and entry.startswith("."):
                continue
            full = os.path.join(search_dir, entry)
            is_dir = os.path.isdir(full)
            # Explicit `@folder:` / `@file:` — honour the user's filter.  Skip
            # the opposite kind instead of auto-rewriting the completion tag,
            # which used to defeat the prefix and let `@folder:` list files.
            if prefix_tag and want_dir != is_dir:
                continue
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            suffix = "/" if is_dir else ""

            if is_context and prefix_tag:
                text = f"@{prefix_tag}:{rel}{suffix}"
            elif is_context:
                kind = "folder" if is_dir else "file"
                text = f"@{kind}:{rel}{suffix}"
            elif word.startswith("~"):
                text = "~/" + os.path.relpath(full, os.path.expanduser("~")) + suffix
            elif word.startswith("./"):
                text = "./" + rel + suffix
            else:
                text = rel + suffix

            items.append({
                "text": text,
                "display": entry + suffix,
                "meta": "dir" if is_dir else "",
            })
            if len(items) >= 30:
                break
    except Exception as e:
        return _err(rid, 5021, str(e))

    return _ok(rid, {"items": items})


def _details_completion_item(value: str, meta: str = "") -> dict:
    return {"text": value, "display": value, "meta": meta}


def _details_root_completion_item(
    value: str, meta: str, needs_leading_space: bool
) -> dict:
    return _details_completion_item(
        f" {value}" if needs_leading_space else value,
        meta,
    )


def _details_completions(text: str) -> list[dict] | None:
    if not text.lower().startswith("/details"):
        return None

    stripped = text.strip()
    if stripped and not "/details".startswith(stripped.lower().split()[0]):
        return None

    body = text[len("/details") :]
    if body.startswith(" "):
        body = body[1:]
    parts = body.split()
    has_trailing_space = text.endswith(" ")
    sections = ("thinking", "tools", "subagents", "activity")
    modes = ("hidden", "collapsed", "expanded")

    if not body or (len(parts) == 0 and has_trailing_space):
        return [
            *[
                _details_root_completion_item(
                    mode, "global mode", not has_trailing_space
                )
                for mode in modes
            ],
            _details_root_completion_item(
                "cycle", "cycle global mode", not has_trailing_space
            ),
            *[
                _details_root_completion_item(
                    section, "section override", not has_trailing_space
                )
                for section in sections
            ],
        ]

    if len(parts) == 1 and not has_trailing_space:
        prefix = parts[0].lower()
        candidates = [*modes, "cycle", *sections]
        return [
            _details_completion_item(
                candidate,
                (
                    "section override"
                    if candidate in sections
                    else "cycle global mode"
                    if candidate == "cycle"
                    else "global mode"
                ),
            )
            for candidate in candidates
            if candidate.startswith(prefix) and candidate != prefix
        ]

    if len(parts) == 1 and has_trailing_space and parts[0].lower() in sections:
        return [
            *[
                _details_completion_item(mode, f"set {parts[0].lower()}")
                for mode in modes
            ],
            _details_completion_item("reset", f"clear {parts[0].lower()} override"),
        ]

    if len(parts) == 2 and not has_trailing_space and parts[0].lower() in sections:
        prefix = parts[1].lower()
        return [
            _details_completion_item(
                candidate,
                (
                    f"clear {parts[0].lower()} override"
                    if candidate == "reset"
                    else f"set {parts[0].lower()}"
                ),
            )
            for candidate in (*modes, "reset")
            if candidate.startswith(prefix) and candidate != prefix
        ]

    return []


@method("complete.slash")
def _(rid, params: dict) -> dict:
    text = params.get("text", "")
    if not text.startswith("/"):
        return _ok(rid, {"items": []})

    try:
        from hermes_cli.commands import SlashCommandCompleter
        from prompt_toolkit.document import Document
        from prompt_toolkit.formatted_text import to_plain_text

        from agent.skill_commands import get_skill_commands
        from agent.skill_bundles import get_skill_bundles

        completer = SlashCommandCompleter(
            skill_commands_provider=lambda: get_skill_commands(),
            skill_bundles_provider=lambda: get_skill_bundles(),
        )
        doc = Document(text, len(text))
        items = [
            {
                "text": c.text,
                # prompt_toolkit gives us FormattedText (a list of (style,
                # text) tuples) for display/display_meta. Serialize both as
                # plain strings — the TUI's CompletionItem.display contract
                # is a string, and sending the raw list trips Ink's row
                # layout into 1-char truncation of the next column.
                "display": to_plain_text(c.display) if c.display else c.text,
                "meta": to_plain_text(c.display_meta) if c.display_meta else "",
            }
            for c in completer.get_completions(doc, None)
        ][:30]
        text_lower = text.lower()
        extras = [
            {
                "text": "/compact",
                "display": "/compact",
                "meta": "Toggle compact display mode",
            },
            {
                "text": "/details",
                "display": "/details",
                "meta": "Control agent detail visibility",
            },
            {
                "text": "/logs",
                "display": "/logs",
                "meta": "Show recent gateway log lines",
            },
            {
                "text": "/mouse",
                "display": "/mouse",
                "meta": "Set mouse tracking preset [on|off|toggle|wheel|buttons|all]",
            },
        ]
        for extra in extras:
            if extra["text"].startswith(text_lower) and not any(
                item["text"] == extra["text"] for item in items
            ):
                items.append(extra)

        details_items = _details_completions(text)
        if details_items is not None:
            return _ok(
                rid,
                {
                    "items": details_items,
                    "replace_from": text.rfind(" ") + 1 if " " in text else len(text),
                },
            )

        return _ok(
            rid,
            {"items": items, "replace_from": text.rfind(" ") + 1 if " " in text else 1},
        )
    except Exception as e:
        return _err(rid, 5020, str(e))


@method("model.options")
def _(rid, params: dict) -> dict:
    try:
        from hermes_cli.inventory import build_models_payload, load_picker_context

        session = _sessions.get(params.get("session_id", ""))
        agent = session.get("agent") if session else None
        # Layer agent-session state on top of disk config — once an agent
        # is spawned, IT owns the live provider/model/base_url. Empty
        # agent attributes must NOT clobber disk config (with_overrides
        # is truthy-only).
        ctx = load_picker_context().with_overrides(
            current_provider=getattr(agent, "provider", "") if agent else "",
            current_model=(
                (getattr(agent, "model", "") if agent else "") or _resolve_model()
            ),
            current_base_url=getattr(agent, "base_url", "") if agent else "",
        )
        # picker_hints + canonical_order produce the TUI/desktop picker shape:
        # `authenticated`/`auth_type`/`key_env`/`warning` per row, in
        # CANONICAL_PROVIDERS declaration order. Desktop pickers default to the
        # configured subset; callers that need setup affordances can pass
        # include_unconfigured=true explicitly.
        # Curated model lists are preserved — list_authenticated_providers
        # populates `models` from the curated catalog, not provider_model_ids
        # (which would pull non-agentic models like TTS/embeddings/etc.).
        payload = build_models_payload(
            ctx,
            explicit_only=bool(params.get("explicit_only")),
            include_unconfigured=bool(params.get("include_unconfigured")),
            picker_hints=True,
            canonical_order=True,
            pricing=True,
            capabilities=True,
            refresh=bool(params.get("refresh")),
            probe_custom_providers=bool(params.get("refresh")),
            probe_current_custom_provider=not bool(params.get("refresh")),
        )
        return _ok(rid, payload)
    except Exception as e:
        return _err(rid, 5033, str(e))


@method("model.save_key")
def _(rid, params: dict) -> dict:
    """Save an API key for a provider, then return its refreshed model list.

    Params:
        slug: provider slug (e.g. "deepseek", "xai")
        api_key: the key value to save

    Returns the provider dict with models populated (same shape as
    model.options entries) on success.
    """
    try:
        from hermes_cli.auth import PROVIDER_REGISTRY
        from hermes_cli.config import is_managed, save_env_value
        from hermes_cli.inventory import build_models_payload, load_picker_context

        slug = (params.get("slug") or "").strip()
        api_key = (params.get("api_key") or "").strip()
        if not slug or not api_key:
            return _err(rid, 4001, "slug and api_key are required")

        if is_managed():
            return _err(rid, 4006, "managed install — credentials are read-only")

        pconfig = PROVIDER_REGISTRY.get(slug)
        if not pconfig:
            return _err(rid, 4002, f"unknown provider: {slug}")
        if pconfig.auth_type != "api_key":
            return _err(
                rid,
                4003,
                f"{pconfig.name} uses {pconfig.auth_type} auth — "
                f"run `hermes model` to configure",
            )
        if not pconfig.api_key_env_vars:
            return _err(rid, 4004, f"no env var defined for {pconfig.name}")

        # Save the key to ~/.hermes/.env
        env_var = pconfig.api_key_env_vars[0]
        save_env_value(env_var, api_key)
        # Also set in current process so the refreshed inventory sees it.
        import os

        os.environ[env_var] = api_key

        # Refresh provider data via the shared inventory builder so this
        # surface stays in lock-step with model.options + dashboard
        # /api/model/options. picker_hints=True ensures the returned row
        # carries `authenticated` for the TUI frontend.
        session = _sessions.get(params.get("session_id", ""))
        agent = session.get("agent") if session else None
        ctx = load_picker_context().with_overrides(
            current_provider=getattr(agent, "provider", "") if agent else "",
            current_model=(
                (getattr(agent, "model", "") if agent else "") or _resolve_model()
            ),
            current_base_url=getattr(agent, "base_url", "") if agent else "",
        )
        payload = build_models_payload(
            ctx,
            picker_hints=True,
            max_models=50,
        )
        provider_data = next(
            (p for p in payload["providers"] if p["slug"] == slug), None
        )
        if provider_data is None:
            # Key was saved but provider didn't appear — still return success.
            provider_data = {
                "slug": slug,
                "name": pconfig.name,
                "is_current": False,
                "models": [],
                "total_models": 0,
                "authenticated": True,
            }
        # picker_hints sets `authenticated` from the row state, but the
        # synthetic fallback above doesn't go through that path.
        provider_data["authenticated"] = True
        return _ok(rid, {"provider": provider_data})
    except Exception as e:
        return _err(rid, 5034, str(e))


@method("model.disconnect")
def _(rid, params: dict) -> dict:
    """Remove credentials for a provider.

    Params:
        slug: provider slug (e.g. "deepseek", "xai")

    Returns success status and the provider's slug.
    """
    try:
        from hermes_cli.auth import PROVIDER_REGISTRY, clear_provider_auth
        from hermes_cli.config import remove_env_value

        slug = (params.get("slug") or "").strip()
        if not slug:
            return _err(rid, 4001, "slug is required")

        pconfig = PROVIDER_REGISTRY.get(slug)
        cleared_env = False
        cleared_auth = False

        # Remove API key env vars from .env and process
        if pconfig and pconfig.api_key_env_vars:
            for ev in pconfig.api_key_env_vars:
                if remove_env_value(ev):
                    cleared_env = True

        # Clear OAuth / credential pool state
        cleared_auth = clear_provider_auth(slug)

        if not cleared_env and not cleared_auth:
            return _err(rid, 4005, f"no credentials found for {slug}")

        provider_name = pconfig.name if pconfig else slug
        return _ok(
            rid,
            {
                "slug": slug,
                "name": provider_name,
                "disconnected": True,
            },
        )
    except Exception as e:
        return _err(rid, 5035, str(e))


# ── Methods: slash.exec ──────────────────────────────────────────────


def _mirror_slash_side_effects(sid: str, session: dict, command: str) -> str:
    """Apply side effects that must also hit the gateway's live agent."""
    parts = command.lstrip("/").split(None, 1)
    if not parts:
        return ""
    name, arg, agent = (
        parts[0],
        (parts[1].strip() if len(parts) > 1 else ""),
        session.get("agent"),
    )

    # Reject agent-mutating commands during an in-flight turn.  These
    # all do read-then-mutate on live agent/session state that the
    # worker thread running agent.run_conversation is using.  Parity
    # with the session.compress / session.undo guards and the gateway
    # runner's running-agent /model guard.
    _MUTATES_WHILE_RUNNING = {"model", "personality", "prompt", "compress"}
    if name in _MUTATES_WHILE_RUNNING and session.get("running"):
        return f"session busy — /interrupt the current turn before running /{name}"

    try:
        if name == "model" and arg and agent:
            result = _apply_model_switch(sid, session, arg)
            return result.get("warning", "")
        elif name == "personality" and arg and agent:
            pname, new_prompt = _validate_personality(arg, _load_cfg())
            _apply_personality_to_session(sid, session, new_prompt, pname)
        elif name == "prompt" and agent:
            cfg = _load_cfg()
            new_prompt = _prompt_text((cfg.get("agent") or {}).get("system_prompt", ""))
            agent.ephemeral_system_prompt = new_prompt or None
            agent._cached_system_prompt = None
        elif name == "compress" and agent:
            # Mirror the session.compress RPC: build a before/after summary so
            # the user gets feedback (#46686). The slash path previously just
            # compressed + emitted session.info and returned "", so the TUI
            # showed no "compressed N → M messages / ~X → ~Y tokens" stats
            # while CLI and gateway both did.
            from agent.manual_compression_feedback import summarize_manual_compression
            from agent.model_metadata import estimate_request_tokens_rough

            with session["history_lock"]:
                _before_messages = list(session.get("history", []))
            _before_count = len(_before_messages)
            _sys_prompt = getattr(agent, "_cached_system_prompt", "") or ""
            _tools = getattr(agent, "tools", None) or None
            _before_tokens = (
                estimate_request_tokens_rough(
                    _before_messages, system_prompt=_sys_prompt, tools=_tools
                )
                if _before_count
                else 0
            )

            _compress_session_history(session, arg)
            _sync_session_key_after_compress(sid, session)

            with session["history_lock"]:
                _after_messages = list(session.get("history", []))
            _sys_prompt_after = getattr(agent, "_cached_system_prompt", "") or _sys_prompt
            _tools_after = getattr(agent, "tools", None) or _tools
            _after_tokens = (
                estimate_request_tokens_rough(
                    _after_messages, system_prompt=_sys_prompt_after, tools=_tools_after
                )
                if _after_messages
                else 0
            )
            _emit("session.info", sid, _session_info(agent, session))
            _fb = summarize_manual_compression(
                _before_messages, _after_messages, _before_tokens, _after_tokens
            )
            _lines = [_fb["headline"], _fb["token_line"]]
            if _fb.get("note"):
                _lines.append(_fb["note"])
            return "\n".join(_lines)
        elif name == "fast" and agent:
            mode = arg.lower()
            if mode in {"fast", "on"}:
                agent.service_tier = "priority"
            elif mode in {"normal", "off"}:
                agent.service_tier = None
            _emit("session.info", sid, _session_info(agent, session))
        elif name == "reload-mcp" and agent and hasattr(agent, "reload_mcp_tools"):
            agent.reload_mcp_tools()
        elif name == "stop":
            from tools.process_registry import process_registry

            process_registry.kill_all()
    except Exception as e:
        return f"live session sync failed: {e}"
    return ""


@method("slash.exec")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err

    cmd = params.get("command", "").strip()
    if not cmd:
        return _err(rid, 4004, "empty command")

    # Skill slash commands and _pending_input commands must NOT go through the
    # slash worker — see _PENDING_INPUT_COMMANDS definition above. Plugin
    # commands must also avoid the worker, but unlike skills/pending-input they
    # still return normal slash.exec output so the TUI keeps the pager path.
    _cmd_text = cmd.lstrip("/") if cmd.startswith("/") else cmd
    _cmd_parts = _cmd_text.split(maxsplit=1)
    _cmd_base = (_cmd_parts[0] if _cmd_parts else "").lower()
    _cmd_arg = _cmd_parts[1] if len(_cmd_parts) > 1 else ""

    if _cmd_base in _PENDING_INPUT_COMMANDS:
        # Route directly to command.dispatch instead of returning an error
        # that requires the frontend to retry.  Some TUI clients fail the
        # fallback, leaving the command empty and showing "empty command".
        return _methods["command.dispatch"](
            rid,
            {
                "name": _cmd_base,
                "arg": _cmd_arg,
                "session_id": params.get("session_id", ""),
            },
        )

    if _cmd_base in _WORKER_BLOCKED_COMMANDS:
        subcommand = _cmd_arg.split(maxsplit=1)[0].lower() if _cmd_arg else ""
        if subcommand in {"restore", "rewind"}:
            return _err(
                rid,
                4018,
                "snapshot restore mutates live config/state; use command.dispatch for /snapshot restore",
            )

    try:
        from agent.skill_commands import get_skill_commands

        _cmd_key = f"/{_cmd_base}"
        if _cmd_key in get_skill_commands():
            return _err(
                rid, 4018, f"skill command: use command.dispatch for {_cmd_key}"
            )
    except Exception:
        pass

    plugin_handler = None
    resolve_plugin_command_result = None
    if _cmd_base:
        try:
            from hermes_cli.plugins import (
                get_plugin_command_handler,
                resolve_plugin_command_result,
            )

            plugin_handler = get_plugin_command_handler(_cmd_base)
        except Exception:
            plugin_handler = None
            resolve_plugin_command_result = None

    if plugin_handler and resolve_plugin_command_result:
        try:
            result = resolve_plugin_command_result(plugin_handler(_cmd_arg))
            return _ok(rid, {"output": str(result or "(no output)")})
        except Exception as e:
            return _ok(rid, {"output": f"Plugin command error: {e}"})

    worker = session.get("slash_worker")
    if not worker:
        try:
            worker = _SlashWorker(
                session["session_key"],
                getattr(session.get("agent"), "model", _resolve_model()),
                profile_home=session.get("profile_home"),
            )
            _attach_worker(params.get("session_id", ""), session, worker)
        except Exception as e:
            return _err(rid, 5030, f"slash worker start failed: {e}")

    try:
        output = worker.run(cmd)
        warning = _mirror_slash_side_effects(params.get("session_id", ""), session, cmd)
        payload = {"output": output or "(no output)"}
        if warning:
            payload["warning"] = warning
        return _ok(rid, payload)
    except Exception as e:
        try:
            worker.close()
        except Exception:
            pass
        session["slash_worker"] = None
        return _err(rid, 5030, str(e))


# ── Methods: voice ───────────────────────────────────────────────────


_voice_sid_lock = threading.Lock()
_voice_event_sid: str = ""


def _voice_emit(event: str, payload: dict | None = None) -> None:
    """Emit a voice event toward the session that most recently turned the
    mode on. Voice is process-global (one microphone), so there's only ever
    one sid to target; the TUI handler treats an empty sid as "active
    session". Kept separate from _emit to make the lack of per-call sid
    argument explicit."""
    with _voice_sid_lock:
        sid = _voice_event_sid
    _emit(event, sid, payload)


def _voice_mode_enabled() -> bool:
    """Current voice-mode flag (runtime-only, CLI parity).

    cli.py initialises ``_voice_mode = False`` at startup and only flips
    it via ``/voice on``; it never reads a persisted enable bit from
    config.yaml.  We match that: no config lookup, env var only.  This
    avoids the TUI auto-starting in REC the next time the user opens it
    just because they happened to enable voice in a prior session.
    """
    return os.environ.get("HERMES_VOICE", "").strip() == "1"


def _voice_tts_enabled() -> bool:
    """Whether agent replies should be spoken back via TTS (runtime only)."""
    return os.environ.get("HERMES_VOICE_TTS", "").strip() == "1"


def _voice_cfg_dict() -> dict:
    """Shape-safe accessor for the ``voice:`` block in config.yaml.

    ``_load_cfg()`` returns raw ``yaml.safe_load()`` output, so both the
    root AND ``voice`` may be any YAML scalar / list / None. A hand-edit
    like ``voice: true`` or a malformed top-level config that parses to
    a scalar would otherwise break ``.get("…")`` and take every
    ``voice.*`` branch down with it (Copilot round-3..7 review on
    #19835). Coerce through ``isinstance`` at every level so malformed
    config falls back to an empty dict instead of crashing /voice.
    """
    cfg = _load_cfg()
    voice_cfg = cfg.get("voice") if isinstance(cfg, dict) else None

    return voice_cfg if isinstance(voice_cfg, dict) else {}


def _voice_record_key() -> str:
    """Current ``voice.record_key`` value, documented default on error."""
    record_key = _voice_cfg_dict().get("record_key")

    return str(record_key) if isinstance(record_key, str) and record_key else "ctrl+b"


@method("voice.toggle")
def _(rid, params: dict) -> dict:
    """CLI parity for the ``/voice`` slash command.

    Subcommands:

    * ``status`` — report mode + TTS flags (default when action is unknown).
    * ``on`` / ``off`` — flip voice *mode* (the umbrella bit). Turning it
      off also tears down any active continuous recording loop. Does NOT
      start recording on its own; recording is driven by ``voice.record``
      (Ctrl+B) after mode is on, matching cli.py's enable/Ctrl+B split.
    * ``tts`` — toggle speech-output of agent replies. Requires mode on
      (mirrors CLI's _toggle_voice_tts guard).
    """
    action = params.get("action", "status")

    if action == "status":
        # Mirror CLI's _show_voice_status: include STT/TTS provider
        # availability so the user can tell at a glance *why* voice mode
        # isn't working ("STT provider: MISSING ..." is the common case).
        # ``record_key`` mirrors the configured ``voice.record_key`` so the
        # TUI can both bind it (frontend ``isVoiceToggleKey``) and display
        # it in /voice status — previously the TUI hardcoded Ctrl+B and
        # ignored the config (#18994).
        payload: dict = {
            "enabled": _voice_mode_enabled(),
            "record_key": _voice_record_key(),
            "tts": _voice_tts_enabled(),
        }
        try:
            from tools.voice_mode import check_voice_requirements

            reqs = check_voice_requirements()
            payload["available"] = bool(reqs.get("available"))
            payload["audio_available"] = bool(reqs.get("audio_available"))
            payload["stt_available"] = bool(reqs.get("stt_available"))
            payload["details"] = reqs.get("details") or ""
        except Exception as e:
            # check_voice_requirements pulls optional transcription deps —
            # swallow so /voice status always returns something useful.
            logger.warning("voice.toggle status: requirements probe failed: %s", e)

        return _ok(rid, payload)

    if action in {"on", "off"}:
        enabled = action == "on"
        # Runtime-only flag (CLI parity) — no _write_config_key, so the
        # next TUI launch starts with voice OFF instead of auto-REC from a
        # persisted stale toggle.
        os.environ["HERMES_VOICE"] = "1" if enabled else "0"

        if not enabled:
            # Disabling the mode must tear the continuous loop down; the
            # loop holds the microphone and would otherwise keep running.
            try:
                from hermes_cli.voice import stop_continuous

                stop_continuous()
            except ImportError:
                pass
            except Exception as e:
                logger.warning("voice: stop_continuous failed during toggle off: %s", e)

            # Clear TTS so it can be toggled independently after voice is off.
            os.environ["HERMES_VOICE_TTS"] = "0"

        return _ok(
            rid,
            {
                "enabled": enabled,
                "record_key": _voice_record_key(),
                "tts": _voice_tts_enabled(),
            },
        )

    if action == "tts":
        if not _voice_mode_enabled():
            return _err(rid, 4014, "enable voice mode first: /voice on")
        new_value = not _voice_tts_enabled()
        # Runtime-only flag (CLI parity) — see voice.toggle on/off above.
        os.environ["HERMES_VOICE_TTS"] = "1" if new_value else "0"
        # Include ``record_key`` on every branch so a /voice tts toggle
        # doesn't reset the TUI's cached shortcut to the default when a
        # user has a custom binding configured (Copilot review, round 2
        # on #19835). Keeps parity with the status/on/off branches above.
        return _ok(
            rid,
            {
                "enabled": True,
                "record_key": _voice_record_key(),
                "tts": new_value,
            },
        )

    return _err(rid, 4013, f"unknown voice action: {action}")


@method("voice.record")
def _(rid, params: dict) -> dict:
    """VAD-bounded push-to-talk capture, CLI-parity.

    ``start`` begins one VAD-bounded capture and emits ``voice.transcript``
    after silence stops the recorder. ``stop`` forces transcription of the
    active buffer, matching classic CLI push-to-talk. The voice wrapper retains
    no-speech counts across single-shot starts, so three consecutive silent
    captures emit ``voice.transcript`` with ``no_speech_limit=True``.
    """
    action = params.get("action", "start")

    if action not in {"start", "stop"}:
        return _err(rid, 4019, f"unknown voice action: {action}")

    try:
        if action == "start":
            if not _voice_mode_enabled():
                return _err(rid, 4015, "voice mode is off — enable with /voice on")

            with _voice_sid_lock:
                global _voice_event_sid
                _voice_event_sid = params.get("session_id") or _voice_event_sid

            from hermes_cli.voice import start_continuous

            # Shape-safe lookups: malformed ``voice:`` YAML (bool/scalar/list)
            # must not crash /voice with a 5025 — fall back to VAD defaults.
            #
            # Exclude ``bool`` from the numeric check since Python's bool is
            # a subclass of int — a hand-edit like ``silence_threshold: true``
            # would otherwise forward as ``1`` instead of falling back to
            # the documented 200 / 3.0 defaults (Copilot round-12 on #19835).
            voice_cfg = _voice_cfg_dict()
            threshold = voice_cfg.get("silence_threshold")
            duration = voice_cfg.get("silence_duration")
            safe_threshold = (
                threshold
                if isinstance(threshold, (int, float))
                and not isinstance(threshold, bool)
                else 200
            )
            safe_duration = (
                duration
                if isinstance(duration, (int, float)) and not isinstance(duration, bool)
                else 3.0
            )
            started = start_continuous(
                on_transcript=lambda t: _voice_emit("voice.transcript", {"text": t}),
                on_status=lambda s: _voice_emit("voice.status", {"state": s}),
                on_silent_limit=lambda: _voice_emit(
                    "voice.transcript", {"no_speech_limit": True}
                ),
                silence_threshold=safe_threshold,
                silence_duration=safe_duration,
                auto_restart=False,
            )
            if started is False:
                return _ok(rid, {"status": "busy"})
            return _ok(rid, {"status": "recording"})

        # action == "stop"
        with _voice_sid_lock:
            _voice_event_sid = params.get("session_id") or _voice_event_sid

        from hermes_cli.voice import stop_continuous

        stop_continuous(force_transcribe=True)
        return _ok(rid, {"status": "stopped"})
    except ImportError:
        return _err(
            rid, 5025, "voice module not available — install audio dependencies"
        )
    except Exception as e:
        return _err(rid, 5025, str(e))


@method("voice.tts")
def _(rid, params: dict) -> dict:
    text = params.get("text", "")
    if not text:
        return _err(rid, 4020, "text required")
    try:
        from hermes_cli.voice import speak_text

        threading.Thread(target=speak_text, args=(text,), daemon=True).start()
        return _ok(rid, {"status": "speaking"})
    except ImportError:
        return _err(rid, 5026, "voice module not available")
    except Exception as e:
        return _err(rid, 5026, str(e))


# ── Methods: insights ────────────────────────────────────────────────


@method("insights.get")
def _(rid, params: dict) -> dict:
    days = params.get("days", 30)
    db = _get_db()
    if db is None:
        return _db_unavailable_error(rid, code=5017)
    try:
        cutoff = time.time() - days * 86400
        rows = [
            s
            for s in db.list_sessions_rich(limit=500, compact_rows=True)
            if (s.get("started_at") or 0) >= cutoff
        ]
        return _ok(
            rid,
            {
                "days": days,
                "sessions": len(rows),
                "messages": sum(s.get("message_count", 0) for s in rows),
            },
        )
    except Exception as e:
        return _err(rid, 5017, str(e))


# ── Methods: rollback ────────────────────────────────────────────────


@method("rollback.list")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    try:

        def go(mgr, cwd):
            if not mgr.enabled:
                return _ok(rid, {"enabled": False, "checkpoints": []})
            return _ok(
                rid,
                {
                    "enabled": True,
                    "checkpoints": [
                        {
                            "hash": c.get("hash", ""),
                            "timestamp": c.get("timestamp", ""),
                            "message": c.get("message", ""),
                        }
                        for c in mgr.list_checkpoints(cwd)
                    ],
                },
            )

        return _with_checkpoints(session, go)
    except Exception as e:
        return _err(rid, 5020, str(e))


@method("rollback.restore")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    target = params.get("hash", "")
    file_path = params.get("file_path", "")
    if not target:
        return _err(rid, 4014, "hash required")
    # Full-history rollback mutates session history.  Rejecting during
    # an in-flight turn prevents prompt.submit from silently dropping
    # the agent's output (version mismatch path) or clobbering the
    # rollback (version-matches path).  A file-scoped rollback only
    # touches disk, so we allow it.
    if not file_path and session.get("running"):
        return _err(
            rid,
            4009,
            "session busy — /interrupt the current turn before full rollback.restore",
        )
    try:

        def go(mgr, cwd):
            resolved = _resolve_checkpoint_hash(mgr, cwd, target)
            result = mgr.restore(cwd, resolved, file_path=file_path or None)
            if result.get("success") and not file_path:
                removed = 0
                with session["history_lock"]:
                    history = session.get("history", [])
                    while history and history[-1].get("role") in {"assistant", "tool"}:
                        history.pop()
                        removed += 1
                    if history and history[-1].get("role") == "user":
                        history.pop()
                        removed += 1
                    if removed:
                        session["history_version"] = (
                            int(session.get("history_version", 0)) + 1
                        )
                result["history_removed"] = removed
            return result

        return _ok(rid, _with_checkpoints(session, go))
    except Exception as e:
        return _err(rid, 5021, str(e))


@method("rollback.diff")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    target = params.get("hash", "")
    if not target:
        return _err(rid, 4014, "hash required")
    try:
        r = _with_checkpoints(
            session,
            lambda mgr, cwd: mgr.diff(cwd, _resolve_checkpoint_hash(mgr, cwd, target)),
        )
        raw = r.get("diff", "")[:4000]
        payload = {"stat": r.get("stat", ""), "diff": raw}
        rendered = render_diff(raw, session.get("cols", 80))
        if rendered:
            payload["rendered"] = rendered
        return _ok(rid, payload)
    except Exception as e:
        return _err(rid, 5022, str(e))


# ── Methods: browser / plugins / cron / skills ───────────────────────


def _resolve_browser_cdp_url() -> str:
    """Return the configured browser CDP override without network I/O.

    ``/browser status`` must be fast — calling
    ``tools.browser_tool._get_cdp_override`` would invoke
    ``_resolve_cdp_override``, which performs an HTTP probe to
    ``.../json/version`` for discovery-style URLs.  That probe has
    a multi-second timeout and would block the TUI on a slow or
    unreachable host even though status only needs to report whether
    an override is set.

    Mirrors the env/config precedence of ``_get_cdp_override`` (env
    var first, then ``browser.cdp_url`` from config.yaml) without the
    websocket-resolution step, so the answer reflects user intent
    even when the configured host is not currently reachable.  The
    actual WS normalization happens in ``browser_navigate`` on the
    next tool call.
    """
    env_url = os.environ.get("BROWSER_CDP_URL", "").strip()
    if env_url:
        return env_url
    try:
        from hermes_cli.config import read_raw_config

        cfg = read_raw_config()
        browser_cfg = cfg.get("browser", {}) if isinstance(cfg, dict) else {}
        if isinstance(browser_cfg, dict):
            return str(browser_cfg.get("cdp_url", "") or "").strip()
    except Exception:
        pass
    return ""


def _is_default_local_cdp(parsed) -> bool:
    """Match the discovery-style local default; never the concrete WS form.

    A user-supplied ``ws://127.0.0.1:9222/devtools/browser/<id>`` is a
    real, connectable endpoint — collapsing it to bare ``http://...:9222``
    would strip the path and break the connect.
    """
    try:
        port = parsed.port or 80
    except ValueError:
        return False

    discovery_path = parsed.path in {"", "/", "/json", "/json/version"}
    return (
        parsed.scheme in {"http", "ws"}
        and parsed.hostname in {"127.0.0.1", "localhost"}
        and port == 9222
        and discovery_path
    )


def _http_ok(url: str, timeout: float) -> bool:
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= getattr(resp, "status", 200) < 300
    except Exception:
        return False


def _probe_urls(parsed) -> list[str]:
    scheme = {"ws": "http", "wss": "https"}.get(parsed.scheme, parsed.scheme)
    root = f"{scheme}://{parsed.netloc}".rstrip("/")
    return [f"{root}/json/version", f"{root}/json"]


def _normalize_cdp_url(parsed) -> str:
    # Concrete ``/devtools/browser/<id>`` endpoints (Browserbase et al.)
    # are connectable as-is. Discovery-style inputs collapse to bare
    # ``scheme://host:port`` so ``_resolve_cdp_override`` can append
    # ``/json/version`` later without doubling the path.
    if parsed.path.startswith("/devtools/browser/"):
        return parsed.geturl()
    return parsed._replace(path="", params="", query="", fragment="").geturl()


def _failure_messages(url: str, port: int, system: str) -> list[str]:
    from hermes_cli.browser_connect import manual_chrome_debug_command

    command = manual_chrome_debug_command(port, system)
    hint = (
        [
            "Start a Chromium-family browser with remote debugging, then retry /browser connect:",
            command,
        ]
        if command
        else [
            "No supported Chromium-family browser executable was found in this environment.",
            f"Install one or start a Chromium-family browser with --remote-debugging-port={port}, then retry /browser connect.",
        ]
    )
    return [
        f"Browser CDP is not reachable at {url}.",
        *hint,
        "Browser not connected — start a Chromium-family browser with remote debugging and retry /browser connect",
    ]


@method("browser.manage")
def _(rid, params: dict) -> dict:
    action = params.get("action", "status")

    if action == "status":
        url = _resolve_browser_cdp_url()
        return _ok(rid, {"connected": bool(url), "url": url})

    if action == "disconnect":
        return _browser_disconnect(rid)

    if action != "connect":
        return _err(rid, 4015, f"unknown action: {action}")

    return _browser_connect(rid, params)


def _browser_connect(rid, params: dict) -> dict:
    import platform

    from hermes_cli.browser_connect import DEFAULT_BROWSER_CDP_URL
    from tools.browser_tool import cleanup_all_browsers
    from urllib.parse import urlparse

    raw_url = params.get("url")
    if raw_url is not None and not isinstance(raw_url, str):
        return _err(
            rid, 4015, f"browser url must be a string, got {type(raw_url).__name__}"
        )
    url = (raw_url or "").strip() or DEFAULT_BROWSER_CDP_URL

    sid = params.get("session_id") or ""
    system = platform.system()
    messages: list[str] = []

    def announce(message: str, *, level: str = "info") -> None:
        messages.append(message)
        # Without a session id the TUI prints `messages` from the
        # response; emitting an event would double-render. Only stream
        # progress when there's a real session to scope it to.
        if sid:
            _emit("browser.progress", sid, {"message": message, "level": level})

    parsed = urlparse(url if "://" in url else f"http://{url}")
    if parsed.scheme not in {"http", "https", "ws", "wss"}:
        return _err(rid, 4015, f"unsupported browser url: {url}")
    if not parsed.hostname:
        return _err(rid, 4015, f"missing host in browser url: {url}")
    try:
        port = parsed.port or (443 if parsed.scheme in {"https", "wss"} else 80)
    except ValueError:
        return _err(rid, 4015, f"invalid port in browser url: {url}")

    # Always normalize default-local to 127.0.0.1:9222 so downstream
    # comparisons + messaging match what we'll actually persist.
    if _is_default_local_cdp(parsed):
        url = DEFAULT_BROWSER_CDP_URL
        parsed = urlparse(url)
        port = parsed.port or 9222

    try:
        # ws[s]://.../devtools/browser/<id> endpoints (hosted CDP
        # providers) don't serve the HTTP discovery path; just check
        # TCP-level reachability and let browser_navigate handshake.
        if parsed.scheme in {"ws", "wss"} and parsed.path.startswith(
            "/devtools/browser/"
        ):
            import socket

            try:
                with socket.create_connection((parsed.hostname, port), timeout=2.0):
                    pass
            except OSError as e:
                return _err(rid, 5031, f"could not reach browser CDP at {url}: {e}")
        else:
            probes = _probe_urls(parsed)
            ok = any(_http_ok(p, timeout=2.0) for p in probes)

            if not ok and _is_default_local_cdp(parsed):
                from hermes_cli.browser_connect import launch_chrome_debug

                announce(
                    "Chromium-family browser isn't running with remote debugging — attempting to launch..."
                )

                launch = launch_chrome_debug(port, system)
                if launch.launched:
                    for _ in range(20):
                        time.sleep(0.5)
                        if any(_http_ok(p, timeout=1.0) for p in probes):
                            ok = True
                            break

                if ok:
                    announce(
                        f"Chromium-family browser launched and listening on port {port}"
                    )
                else:
                    hint = launch.hint
                    if hint:
                        announce(hint, level="error")
                    for line in _failure_messages(url, port, system)[1:]:
                        announce(line, level="error")
                    return _ok(
                        rid, {"connected": False, "url": url, "messages": messages}
                    )
            elif not ok:
                return _err(rid, 5031, f"could not reach browser CDP at {url}")
            elif _is_default_local_cdp(parsed):
                announce(f"Chromium-family browser is already listening on port {port}")

        normalized = _normalize_cdp_url(parsed)

        # Order matters: reap sessions BEFORE publishing the new env
        # so an in-flight tool call sees the old supervisor closed,
        # then again AFTER so the default task's cached supervisor
        # is drained against the new URL.
        cleanup_all_browsers()
        os.environ["BROWSER_CDP_URL"] = normalized
        cleanup_all_browsers()
    except Exception as e:
        return _err(rid, 5031, str(e))

    payload: dict[str, object] = {"connected": True, "url": normalized}
    if messages:
        payload["messages"] = messages
    return _ok(rid, payload)


def _browser_disconnect(rid) -> dict:
    # Reap, drop the env override, reap again — closes the same swap
    # window covered by ``_browser_connect``.
    def reap() -> None:
        try:
            from tools.browser_tool import cleanup_all_browsers

            cleanup_all_browsers()
        except Exception:
            pass

    reap()
    os.environ.pop("BROWSER_CDP_URL", None)
    reap()
    return _ok(rid, {"connected": False})


@method("plugins.list")
def _(rid, params: dict) -> dict:
    try:
        from hermes_cli.plugins import get_plugin_manager

        return _ok(
            rid,
            {
                "plugins": [
                    {
                        "name": n,
                        "version": getattr(i, "version", "?"),
                        "enabled": getattr(i, "enabled", True),
                    }
                    for n, i in get_plugin_manager()._plugins.items()
                ]
            },
        )
    except Exception as e:
        return _err(rid, 5032, str(e))


@method("config.show")
def _(rid, params: dict) -> dict:
    try:
        cfg = _load_cfg()
        model = _resolve_model()
        api_key = os.environ.get("HERMES_API_KEY", "") or cfg.get("api_key", "")
        masked = f"****{api_key[-4:]}" if len(api_key) > 4 else "(not set)"
        base_url = os.environ.get("HERMES_BASE_URL", "") or cfg.get("base_url", "")

        sections = [
            {
                "title": "Model",
                "rows": [
                    ["Model", model],
                    ["Base URL", base_url or "(default)"],
                    ["API Key", masked],
                ],
            },
            {
                "title": "Agent",
                "rows": [
                    ["Max Turns", str(_cfg_max_turns(cfg, 90))],
                    ["Toolsets", ", ".join(cfg.get("enabled_toolsets", [])) or "all"],
                    ["Verbose", str(cfg.get("verbose", False))],
                ],
            },
            {
                "title": "Environment",
                "rows": [
                    ["Working Dir", os.getcwd()],
                    ["Config File", str(_hermes_home / "config.yaml")],
                ],
            },
        ]
        return _ok(rid, {"sections": sections})
    except Exception as e:
        return _err(rid, 5030, str(e))


@method("tools.list")
def _(rid, params: dict) -> dict:
    try:
        from toolsets import get_all_toolsets, get_toolset_info

        session = _sessions.get(params.get("session_id", ""))
        enabled = (
            set(getattr(session["agent"], "enabled_toolsets", []) or [])
            if session
            else set(_load_enabled_toolsets() or [])
        )

        items = []
        for name in sorted(get_all_toolsets().keys()):
            info = get_toolset_info(name)
            if not info:
                continue
            items.append({
                "name": name,
                "description": info["description"],
                "tool_count": info["tool_count"],
                "enabled": name in enabled if enabled else True,
                "tools": info["resolved_tools"],
            })
        return _ok(rid, {"toolsets": items})
    except Exception as e:
        return _err(rid, 5031, str(e))


@method("tools.show")
def _(rid, params: dict) -> dict:
    try:
        from model_tools import get_toolset_for_tool, get_tool_definitions

        session = _sessions.get(params.get("session_id", ""))
        enabled = (
            getattr(session["agent"], "enabled_toolsets", None)
            if session
            else _load_enabled_toolsets()
        )
        tools = get_tool_definitions(enabled_toolsets=enabled, quiet_mode=True)
        sections = {}

        for tool in sorted(tools, key=lambda t: t["function"]["name"]):
            name = tool["function"]["name"]
            desc = str(tool["function"].get("description", "") or "").split("\n")[0]
            if ". " in desc:
                desc = desc[: desc.index(". ") + 1]
            sections.setdefault(get_toolset_for_tool(name) or "unknown", []).append({
                "name": name,
                "description": desc,
            })

        return _ok(
            rid,
            {
                "sections": [
                    {"name": name, "tools": rows}
                    for name, rows in sorted(sections.items())
                ],
                "total": len(tools),
            },
        )
    except Exception as e:
        return _err(rid, 5034, str(e))


@method("tools.configure")
def _(rid, params: dict) -> dict:
    action = str(params.get("action", "") or "").strip().lower()
    targets = [
        str(name).strip() for name in params.get("names", []) or [] if str(name).strip()
    ]
    if action not in {"disable", "enable"}:
        return _err(rid, 4017, f"unknown tools action: {action}")
    if not targets:
        return _err(rid, 4018, "names required")

    try:
        from hermes_cli.config import load_config, save_config
        from hermes_cli.tools_config import (
            CONFIGURABLE_TOOLSETS,
            _apply_mcp_change,
            _apply_toolset_change,
            _get_platform_tools,
            _get_plugin_toolset_keys,
        )

        cfg = load_config()
        valid_toolsets = {
            ts_key for ts_key, _, _ in CONFIGURABLE_TOOLSETS
        } | _get_plugin_toolset_keys()
        toolset_targets = [name for name in targets if ":" not in name]
        mcp_targets = [name for name in targets if ":" in name]
        unknown = [name for name in toolset_targets if name not in valid_toolsets]
        toolset_targets = [name for name in toolset_targets if name in valid_toolsets]

        if toolset_targets:
            _apply_toolset_change(cfg, "cli", toolset_targets, action)

        missing_servers = (
            _apply_mcp_change(cfg, mcp_targets, action) if mcp_targets else set()
        )
        save_config(cfg)

        session = _sessions.get(params.get("session_id", ""))
        info = (
            _reset_session_agent(params.get("session_id", ""), session)
            if session
            else None
        )
        enabled = sorted(
            _get_platform_tools(load_config(), "cli", include_default_mcp_servers=False)
        )
        changed = [
            name
            for name in targets
            if name not in unknown
            and (":" not in name or name.split(":", 1)[0] not in missing_servers)
        ]

        return _ok(
            rid,
            {
                "changed": changed,
                "enabled_toolsets": enabled,
                "info": info,
                "missing_servers": sorted(missing_servers),
                "reset": bool(session),
                "unknown": unknown,
            },
        )
    except Exception as e:
        return _err(rid, 5035, str(e))


@method("toolsets.list")
def _(rid, params: dict) -> dict:
    try:
        from toolsets import get_all_toolsets, get_toolset_info

        session = _sessions.get(params.get("session_id", ""))
        enabled = (
            set(getattr(session["agent"], "enabled_toolsets", []) or [])
            if session
            else set(_load_enabled_toolsets() or [])
        )

        items = []
        for name in sorted(get_all_toolsets().keys()):
            info = get_toolset_info(name)
            if not info:
                continue
            items.append({
                "name": name,
                "description": info["description"],
                "tool_count": info["tool_count"],
                "enabled": name in enabled if enabled else True,
            })
        return _ok(rid, {"toolsets": items})
    except Exception as e:
        return _err(rid, 5032, str(e))


@method("agents.list")
def _(rid, params: dict) -> dict:
    try:
        from tools.process_registry import process_registry

        procs = process_registry.list_sessions()
        return _ok(
            rid,
            {
                "processes": [
                    {
                        "session_id": p["session_id"],
                        "command": p["command"][:80],
                        "status": p["status"],
                        "uptime": p["uptime_seconds"],
                    }
                    for p in procs
                ]
            },
        )
    except Exception as e:
        return _err(rid, 5033, str(e))


@method("cron.manage")
def _(rid, params: dict) -> dict:
    action, jid = params.get("action", "list"), params.get("name", "")
    try:
        from tools.cronjob_tools import cronjob

        if action == "list":
            return _ok(rid, json.loads(cronjob(action="list")))
        if action == "add":
            return _ok(
                rid,
                json.loads(
                    cronjob(
                        action="create",
                        name=jid,
                        schedule=params.get("schedule", ""),
                        prompt=params.get("prompt", ""),
                    )
                ),
            )
        if action in {"remove", "pause", "resume"}:
            return _ok(rid, json.loads(cronjob(action=action, job_id=jid)))
        return _err(rid, 4016, f"unknown cron action: {action}")
    except Exception as e:
        return _err(rid, 5023, str(e))


@method("learning.frames")
def _(rid, params: dict) -> dict:
    """Pre-render the learning timeline for the TUI ``/journey`` overlay.

    Returns ``frames`` (reveal 0→1) plus static legend/summary/bucket metadata,
    so Ink can render and walk the tree locally without round-tripping the
    gateway. Shares its renderer with the ``hermes journey`` CLI.
    """
    try:
        cols = int(params.get("cols", 80) or 80)
        rows = int(params.get("rows", 24) or 24)
        frames = int(params.get("frames", 48) or 48)
    except (TypeError, ValueError):
        cols, rows, frames = 80, 24, 48
    try:
        from agent.learning_graph import build_learning_graph
        from agent.learning_graph_render import render_frames

        payload = build_learning_graph()
        return _ok(rid, render_frames(payload, cols=max(20, cols), rows=max(10, rows), frames=frames))
    except Exception as exc:  # noqa: BLE001
        return _err(rid, 5000, f"learning.frames failed: {exc}")


@method("learning.detail")
def _(rid, params: dict) -> dict:
    """Current content of a journey node, for an edit prefill."""
    try:
        from agent.learning_mutations import node_detail

        return _ok(rid, node_detail(str(params.get("id", ""))))
    except Exception as exc:  # noqa: BLE001
        return _err(rid, 5000, f"learning.detail failed: {exc}")


@method("learning.delete")
def _(rid, params: dict) -> dict:
    """Delete a journey node — skills are archived (restorable), memories removed."""
    try:
        from agent.learning_mutations import delete_node

        return _ok(rid, delete_node(str(params.get("id", ""))))
    except Exception as exc:  # noqa: BLE001
        return _err(rid, 5000, f"learning.delete failed: {exc}")


@method("learning.edit")
def _(rid, params: dict) -> dict:
    """Rewrite a journey node's content (SKILL.md or memory chunk)."""
    try:
        from agent.learning_mutations import edit_node

        return _ok(rid, edit_node(str(params.get("id", "")), str(params.get("content", ""))))
    except Exception as exc:  # noqa: BLE001
        return _err(rid, 5000, f"learning.edit failed: {exc}")


@method("skills.manage")
def _(rid, params: dict) -> dict:
    action, query = params.get("action", "list"), params.get("query", "")
    try:
        if action == "list":
            from hermes_cli.banner import get_available_skills

            return _ok(rid, {"skills": get_available_skills()})
        if action == "search":
            from tools.skills_hub import (
                GitHubAuth,
                create_source_router,
                unified_search,
            )

            raw = (
                unified_search(
                    query,
                    create_source_router(GitHubAuth()),
                    source_filter="all",
                    limit=20,
                )
                or []
            )
            return _ok(
                rid,
                {
                    "results": [
                        {"name": r.name, "description": r.description} for r in raw
                    ]
                },
            )
        if action == "install":
            from hermes_cli.skills_hub import do_install

            class _Q:
                def print(self, *a, **k):
                    pass

            do_install(query, skip_confirm=True, console=_Q())
            return _ok(rid, {"installed": True, "name": query})
        if action == "browse":
            from hermes_cli.skills_hub import browse_skills

            pg = int(params.get("page", 0) or 0) or (
                int(query) if query.isdigit() else 1
            )
            return _ok(
                rid, browse_skills(page=pg, page_size=int(params.get("page_size", 20)))
            )
        if action == "inspect":
            from hermes_cli.skills_hub import inspect_skill

            return _ok(rid, {"info": inspect_skill(query) or {}})
        return _err(rid, 4017, f"unknown skills action: {action}")
    except Exception as e:
        return _err(rid, 5024, str(e))


@method("skills.reload")
def _(rid, params: dict) -> dict:
    try:
        from agent.skill_commands import reload_skills

        result = reload_skills()
        added = result.get("added") or []
        removed = result.get("removed") or []
        total = int(result.get("total") or 0)

        lines = ["Reloading skills..."]
        if not added and not removed:
            lines.append("No new skills detected.")
        if added:
            lines.append("Added skills:")
            lines.extend(f"  - {item.get('name', '')}" for item in added)
        if removed:
            lines.append("Removed skills:")
            lines.extend(f"  - {item.get('name', '')}" for item in removed)
        lines.append(f"{total} skill(s) available")
        return _ok(rid, {"output": "\n".join(lines), "result": result})
    except Exception as e:
        return _err(rid, 5025, str(e))


@method("plugins.manage")
def _(rid, params: dict) -> dict:
    """List installed plugins with activation state, or toggle one on/off.

    Backs the TUI Plugins Hub. Uses the same disk-discovery + enable/disable
    primitives as ``hermes plugins`` / the dashboard, so the three surfaces
    agree on what's installed and what's enabled.

    Actions:
      - ``list``   → {"plugins": [{name, version, description, source,
                       status}], "user_count": N, "bundled_count": M}
      - ``toggle`` → flip ``name`` based on ``enable`` (bool). Returns the
                       refreshed row plus {"ok", "unchanged"}.
    """
    action = params.get("action", "list")
    try:
        from hermes_cli.plugins_cmd import (
            _discover_all_plugins,
            _get_disabled_set,
            _get_enabled_set,
            _plugin_status,
        )

        def _rows():
            enabled = _get_enabled_set()
            disabled = _get_disabled_set()
            out = []
            for name, version, desc, source, _dir, key in sorted(
                _discover_all_plugins()
            ):
                out.append(
                    {
                        "name": name,
                        "version": str(version or ""),
                        "description": desc or "",
                        "source": source,
                        "status": _plugin_status(name, enabled, disabled, key=key),
                    }
                )
            return out

        if action == "list":
            rows = _rows()
            user_count = sum(1 for r in rows if r["source"] != "bundled")
            return _ok(
                rid,
                {
                    "plugins": rows,
                    "user_count": user_count,
                    "bundled_count": len(rows) - user_count,
                },
            )

        if action == "toggle":
            from hermes_cli.plugins_cmd import dashboard_set_agent_plugin_enabled

            name = (params.get("name") or "").strip()
            if not name:
                return _err(rid, 4019, "plugins.toggle requires a 'name'")
            enable = bool(params.get("enable"))
            result = dashboard_set_agent_plugin_enabled(name, enabled=enable)
            if not result.get("ok"):
                return _err(rid, 5026, result.get("error") or "toggle failed")
            row = next((r for r in _rows() if r["name"] == name), None)
            return _ok(
                rid,
                {
                    "ok": True,
                    "unchanged": bool(result.get("unchanged")),
                    "name": name,
                    "plugin": row,
                },
            )

        return _err(rid, 4017, f"unknown plugins action: {action}")
    except Exception as e:
        return _err(rid, 5026, str(e))


@method("shell.exec")
def _(rid, params: dict) -> dict:
    cmd = params.get("command", "")
    if not cmd:
        return _err(rid, 4004, "empty command")
    try:
        from tools.approval import detect_dangerous_command, detect_hardline_command

        is_hardline, hardline_desc = detect_hardline_command(cmd)
        if is_hardline:
            return _err(
                rid,
                4005,
                f"blocked (hardline): {hardline_desc}. Use the agent for dangerous commands.",
            )
        is_dangerous, _, desc = detect_dangerous_command(cmd)
        if is_dangerous:
            return _err(
                rid, 4005, f"blocked: {desc}. Use the agent for dangerous commands."
            )
    except ImportError:
        return _err(
            rid, 5001, "shell.exec unavailable: approval safety module not importable"
        )
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30, cwd=os.getcwd(),
            stdin=subprocess.DEVNULL,
        )
        return _ok(
            rid,
            {
                "stdout": r.stdout[-4000:],
                "stderr": r.stderr[-2000:],
                "code": r.returncode,
            },
        )
    except subprocess.TimeoutExpired:
        return _err(rid, 5002, "command timed out (30s)")
    except Exception as e:
        return _err(rid, 5003, str(e))
