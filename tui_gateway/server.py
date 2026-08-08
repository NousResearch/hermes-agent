import atexit
import concurrent.futures
import contextlib
import contextvars
import copy
import hashlib
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
from typing import Any, Callable, NamedTuple, Optional

from agent.secret_scope import (
    build_profile_secret_scope,
    reset_secret_scope,
    set_secret_scope,
)
from hermes_constants import (
    DEFAULT_INDICATOR_STYLE,
    INDICATOR_STYLES,
    get_hermes_home,
    get_hermes_home_override,
    reset_hermes_home_override,
    set_hermes_home_override,
)
from hermes_cli.env_loader import load_hermes_dotenv
from utils import is_truthy_value
from tools.environments.local import hermes_subprocess_env
from agent.replay_cleanup import sanitize_replay_history
from agent.skill_commands import describe_skill_invocation
from agent.conversation_loop import INTERRUPT_WAITING_FOR_MODEL_PREFIX
from tui_gateway import git_probe
from tui_gateway.turn_marker import (
    clear_turn_marker,
    read_turn_marker,
    record_turn_start,
)
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
        # Billing/usage reads each do a blocking portal HTTP fetch (state + usage
        # is two serial round-trips); keep them off the main stdin loop so a slow
        # portal can't stall approval.respond / session.interrupt / other RPCs.
        "billing.state",
        "subscription.state",
        # Subscription change (V3): preview + the pending-change mutations + upgrade
        # each do a blocking portal round-trip (preview + upgrade also hit Stripe,
        # which can take seconds) — keep them off the main stdin loop.
        "subscription.preview",
        "subscription.change",
        "subscription.resume",
        "subscription.upgrade",
        "usage.bars",
        "session.usage",
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
        # model.options builds the full picker payload — per-provider credential
        # pool checks, pricing fetch, Nous tier check, optional custom-provider
        # probe — measured seconds inline. While it runs on the reader thread,
        # prompt.submit / session.interrupt sit unread (same class as #21123),
        # and the Desktop model pill / picker block on it every open.
        "model.options",
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
        # reload.mcp shuts down and rediscovers every MCP server — with a
        # flapping server (retry loops, connect timeouts up to 120s) that can
        # block for minutes. Inline it froze the reader thread: config.set,
        # complete.slash, prompt.submit all sat unread and the TUI appeared
        # dead after a few skin switches. The handler serializes concurrent
        # reloads via _mcp_reload_lock.
        "reload.mcp",
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
        # Desktop also polls the in-memory live-session registry every 15s.
        # The handler is normally cheap, but under heavy agent GIL pressure it
        # can still stall for tens of seconds. Keep it off the WS reader thread
        # so a delayed status rehydrate cannot block runtime readiness, prompt
        # submission, or interrupts queued behind it on the same socket.
        "session.active_list",
        "session.branch",
        "session.compress",
        "session.list",
        "session.resume",
        # Workspace re-home runs git branch/root subprocess probes against an
        # arbitrary folder — inline they'd stall the reader on a slow mount.
        "session.workspace.move",
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
    _rpc_pool_workers = 8
_pool = concurrent.futures.ThreadPoolExecutor(
    max_workers=_rpc_pool_workers,
    thread_name_prefix="tui-rpc",
)
atexit.register(lambda: _pool.shutdown(wait=False, cancel_futures=True))

# Exact in-memory session generation executing on the current turn thread.
# Unlike a public session id, this object identity cannot be supplied by RPC.
_current_runtime_session_record: contextvars.ContextVar[dict | None] = (
    contextvars.ContextVar("hermes_gateway_runtime_session_record", default=None)
)

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
        # Global-remote / multi-profile sessions: the worker must resolve
        # config/skills/state against the session's profile home, not the
        # gateway's launch HERMES_HOME (#40677). The override goes through the
        # build_subprocess_env factory's `extra` (applied last, always wins)
        # instead of a hand-rolled env["HERMES_HOME"] assignment.
        from tools.environments.local import build_subprocess_env
        env = build_subprocess_env(
            hermes_subprocess_env(inherit_credentials=True),
            scrub_secrets=False,
            inherit_profile_home=False,  # base already carries the HOME contract
            extra={"HERMES_HOME": str(profile_home)} if profile_home else None,
        )

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
            # Force UTF-8 with lossy decoding so child output containing bytes
            # that are invalid in the system locale (e.g. GBK on Chinese
            # Windows) can't raise UnicodeDecodeError inside the drain threads
            # and crash the gateway. See #53137.
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


def _load_interim_assistant_messages() -> bool:
    """Return whether interim assistant commentary should be surfaced to UIs.

    Honors ``display.interim_assistant_messages`` (default true). When false,
    the tui_gateway does not install ``interim_assistant_callback``, so
    interim text from tool-call turns and verify-on-stop candidates is never
    emitted as ``message.interim`` — mirroring the messaging gateway's gating.
    """
    display = _load_cfg().get("display")
    if not isinstance(display, dict):
        return True
    return is_truthy_value(display.get("interim_assistant_messages", True))


def _notify_session_boundary(
    event_type: str, session_id: str | None, platform: str | None = None
) -> None:
    """Fire session lifecycle hooks with CLI parity."""
    try:
        from hermes_cli.lifecycle import finalize_session, invoke_hook

        if event_type == "on_session_finalize":
            finalize_session(
                session_id=session_id,
                platform=_resolve_agent_platform(platform),
            )
        else:
            invoke_hook(
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


def _ensure_active_session_slot(sid: str, session: dict) -> str | None:
    """Claim this session's cap slot on its first real turn; None when ok.

    session.create / session.resume deliberately do NOT claim one. Every
    desktop tile paint, background reconnect-resume and abandoned draft opens a
    session just to paint a composer, and a slot held by one of those is
    invisible everywhere: an unprompted draft has no DB row, and the sidebar
    filters it out with min_messages=1. Idle desktop tabs therefore silently
    starved the messaging gateway, which shares this cap — five parked tabs on
    a websocket-flappy host locked a Discord bot out of a 5-slot cap while
    running no agents at all. Claiming on the first turn mirrors the lazy
    contract _ensure_session_db_row already uses for the row itself, and keeps
    the invariant that anything holding a slot is something the user can see.
    """
    if session.get("active_session_lease") is not None:
        return None
    lease, limit_message = _claim_active_session_slot(
        str(session.get("session_key") or ""),
        live_session_id=sid,
        surface=_session_source(session),
    )
    if limit_message is not None:
        return limit_message
    session["active_session_lease"] = lease
    return None


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
    "", "tui", "cli", "webui", "desktop", "cron", "kanban", "subagent", "test",
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
            from hermes_cli.lifecycle import invoke_hook

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
            # End the row in the *session's* profile state.db (app-global
            # remote mode), not the launch profile's shared handle.
            with _session_db(session) as db:
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


# End reasons where the BACKEND reclaimed a session the client never asked to
# close: the idle-TTL reaper, the LRU cap, and the WS-orphan reap. A client
# holding that live session id gets no signal today — its next prompt fails
# against an id the backend has already forgotten, which reads as the session
# silently vanishing rather than being reclaimed. ``tui_close`` and friends are
# deliberately absent: the client initiated those and already knows.
_RECLAIM_END_REASONS = frozenset({"idle_timeout", "lru_evict", "ws_orphan_reap"})


def _announce_session_reclaimed(session: dict, end_reason: str) -> None:
    """Tell connected clients a session was reclaimed out from under them.

    Broadcast rather than session-targeted: the reap paths run on background
    timer threads with no contextvar binding, and the WS-orphan case has by
    definition lost its own transport — ``_emit`` would bottom out on stdio and
    the peer that owns the session would never see it. Best-effort; a failed
    notify must never break teardown.
    """
    if end_reason not in _RECLAIM_END_REASONS:
        return
    try:
        _broadcast_global_event(
            "session.reclaimed",
            {
                "session_id": str(session.get("_sid") or ""),
                "stored_session_id": str(session.get("session_key") or ""),
                "reason": end_reason,
            },
        )
    except Exception:
        logger.debug("session.reclaimed broadcast failed", exc_info=True)


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
    _announce_session_reclaimed(session, end_reason)
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


def _pop_session_by_id(sid: str) -> dict | None:
    """Atomically detach one live session from the registry.

    Detaching is the ownership claim for teardown: once the record is no
    longer in ``_sessions``, a concurrent close/reaper becomes a no-op.  Keep
    this operation separate from ``_teardown_session`` because finalization can
    flush SQLite state, invoke plugins, commit memory, interrupt delegations,
    and close agents/workers.  None of that slow external work belongs under
    the global ``_session_resume_lock``.
    """
    with _sessions_lock:
        session = _sessions.pop(sid, None)
    if session is None:
        return None
    # The session is already out of _sessions here, so downstream teardown
    # (e.g. _finalize_session's per-session async-delegation interrupt) can't
    # recover its live id by scanning the dict — stamp it on the record.
    session["_sid"] = sid
    return session


def _teardown_popped_session(
    session: dict | None, *, end_reason: str = "tui_close"
) -> bool:
    """Finish a close after the caller has atomically detached the session."""
    if session is None:
        return False
    _teardown_session(session, end_reason=end_reason)
    return True


def _close_session_by_id(
    sid: str,
    *,
    end_reason: str = "tui_close",
    predicate: Callable[[dict], bool] | None = None,
) -> bool:
    """Single idempotent teardown funnel for callers needing no resume race.

    Resume-sensitive callers first pop under ``_session_resume_lock`` and then
    call ``_teardown_popped_session`` after releasing it.  Other reapers can use
    this convenience wrapper directly.  The pop remains the single atomic
    ownership claim, so concurrent/repeat close attempts stay harmless.

    Automatic reapers can pass ``predicate`` to revalidate under
    ``_sessions_lock`` immediately before the ownership claim. This prevents a
    stale scan result from closing a session that reattached or gained active
    delegated work before teardown.
    """
    if predicate is None:
        session = _pop_session_by_id(sid)
    else:
        with _sessions_lock:
            current = _sessions.get(sid)
            if current is None or not predicate(current):
                return False
            session = _pop_session_by_id(sid)
    return _teardown_popped_session(session, end_reason=end_reason)


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


def _session_owns_durable_lifecycle(session_id: str | None) -> bool:
    """Whether this TUI/desktop session may end its durable DB row by key."""
    if not session_id:
        return True
    try:
        db = _get_db()
        if db is None:
            return True
        # Don't end gateway-originated sessions — the gateway owns their
        # lifecycle. The TUI is only a viewer there (#60609).
        row = db.get_session(session_id)
        source = (row or {}).get("source", "")
        return not _is_gateway_owned_source(source)
    except Exception:
        return True


def _session_async_delegation_selectors(
    session: dict | None, *, sid_hint: str = ""
) -> tuple[str, str]:
    """Ownership selectors for async background work tied to one UI session."""
    if not session:
        return "", ""
    own_sid = str(sid_hint or session.get("_sid") or "")
    if not own_sid:
        try:
            with _sessions_lock:
                for _cand_sid, _cand in _sessions.items():
                    if _cand is session:
                        own_sid = _cand_sid
                        break
        except Exception:
            own_sid = ""
    agent = session.get("agent")
    session_key = str(session.get("session_key") or "")
    session_id = getattr(agent, "session_id", None) or session_key
    owned_session_key = session_key if _session_owns_durable_lifecycle(session_id) else ""
    return own_sid, owned_session_key


def _session_has_active_delegations(sid: str, session: dict | None = None) -> bool:
    """True when UI session ``sid`` still owns live background work.

    Matches by the live UI sid AND — when the TUI owns the durable lifecycle
    (never for gateway-viewer tabs, #60609) — by the durable session_key, so a
    delegation dispatched from an earlier tab of the same resumed session still
    keeps it alive.
    """
    if session is None:
        with _sessions_lock:
            session = _sessions.get(sid)
    own_sid, owned_session_key = _session_async_delegation_selectors(
        session, sid_hint=sid
    )
    if not own_sid and not owned_session_key:
        return False
    try:
        from tools.async_delegation import has_live_for_session

        return has_live_for_session(
            session_key=owned_session_key,
            origin_ui_session_id=own_sid,
        )
    except Exception:
        logger.debug(
            "Failed to query active delegations for UI session %s",
            sid,
            exc_info=True,
        )
        # A transient registry/import failure must not turn into destructive
        # cleanup. Conservatively keep the detached session and let the next
        # orphan timer retry the lookup.
        return True


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
        # non-orphaned). Claim teardown by popping under both lifecycle locks,
        # then release the global resume lock before the slow finalization work.
        # The dict mutation still happens under _sessions_lock — consistent
        # with every other _sessions mutator
        # (#39591: _reap previously popped under _session_resume_lock, giving no
        # mutual exclusion against _init_session / _close_session_by_id, which
        # guard with _sessions_lock). _sessions_lock is an RLock and the global
        # ordering is always resume_lock -> sessions_lock, so nesting is safe.
        reschedule = False
        session = None
        with _session_resume_lock:
            current = _sessions.get(sid)
            if not _ws_session_is_orphaned(current):
                return
            if _session_has_active_delegations(sid, current):
                reschedule = True
            else:
                session = _pop_session_by_id(sid)
        if reschedule:
            _schedule_ws_orphan_reap(sid)
            return
        _teardown_popped_session(session, end_reason="ws_orphan_reap")

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
    try:
        _release_gateway_wake_owner()
    except Exception:
        pass
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
    if _session_has_active_delegations(sid, session):
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
        _close_session_by_id(
            sid,
            end_reason="idle_timeout",
            predicate=lambda session, victim_sid=sid: _session_is_evictable(
                victim_sid, session, time.time()
            ),
        )
    _enforce_session_cap()
    _reclaim_orphaned_leases()
    # Periodic heap release for long-lived gateway processes.  Even when no
    # session is reaped, Python's generational GC rarely runs gen2 collection
    # under steady-state allocation, and glibc retains freed pages as RSS.
    # Calling trim_memory here ensures every reaper scan (default every 5 min)
    # returns releasable pages, preventing unbounded RSS growth over days/weeks.
    try:
        from hermes_cli.mem_trim import trim_memory

        trim_memory(reason="idle reaper periodic trim")
    except Exception as exc:
        # debug, not warning — persistent failure would repeat every reaper
        # scan (300s) forever; sibling failure branches log at debug.
        logger.debug(
            "idle reaper memory trim failed: %s: %s", type(exc).__name__, exc
        )


def _reclaim_orphaned_leases() -> None:
    """Hand the registry the lease ids we still own so it can drop the rest."""
    try:
        from hermes_cli.active_sessions import release_orphaned_leases

        with _sessions_lock:
            live = {
                lease.lease_id
                for session in _sessions.values()
                if (lease := session.get("active_session_lease")) is not None
            }
        if dropped := release_orphaned_leases(live):
            logger.info("Reclaimed %d orphaned active-session lease(s)", dropped)
    except Exception:
        logger.debug("orphaned lease reclaim failed", exc_info=True)


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
    # awaiting input, still building, or owning active delegated work), but
    # WITHOUT the hours-scale age gate: a detached session is eligible the
    # moment it loses its client.
    if session.get("running") or _session_pending_kind(sid):
        return False
    if _session_has_active_delegations(sid, session):
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
    for sid, _s in evictable:
        with _sessions_lock:
            if len(_sessions) <= cap:
                break
        _close_session_by_id(
            sid,
            end_reason="lru_evict",
            predicate=lambda session, victim_sid=sid: _session_is_lru_evictable(
                victim_sid, session
            ),
        )


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


def _db_for_profile(profile: str | None = None):
    """Return SessionDB for ``params.profile`` when it differs from launch.

    App-global remote mode passes ``profile`` on session.* RPCs so history/list/
    create operate on that profile's ``state.db``. Launch/own profile → shared
    ``_get_db()`` handle (left open). Non-launch profile → a dedicated handle
    the caller should ``close()`` (see :func:`_profile_db` contextmanager).

    Returns (db, owns_handle). ``db`` is None when unavailable.
    """
    profile_home = _profile_home(profile)
    if profile_home is None:
        return _get_db(), False
    try:
        from hermes_state import SessionDB

        return SessionDB(db_path=Path(profile_home) / "state.db"), True
    except Exception as exc:
        logger.warning(
            "TUI profile session store unavailable for %s: %s",
            profile,
            exc,
        )
        return None, False


def _transfer_db_to_agent(agent, db) -> bool:
    """Hand a DEDICATED profile handle to *agent*, which closes it on teardown.

    The build sites open a per-profile ``state.db`` handle, pass it to
    ``_make_agent``, and own it until the built agent is the one that will be
    torn down. This marks that transfer: from here ``AIAgent.close()`` (reached
    via :func:`_teardown_session`) releases the handle, so the caller must stop
    closing it.

    Returns True only when the transfer actually happened. It is refused when
    *agent* is not holding *this* handle — the build failed before
    ``_make_agent``, or the agent was given a different db — because a False
    return is what tells the caller the handle is still its own to close.
    Never called for the shared launch handle: that one is opened by
    ``_get_db()``, outlives every agent, and stays at ``_owns_session_db``
    False.
    """
    if agent is None or db is None:
        return False
    try:
        if getattr(agent, "_session_db", None) is not db:
            return False
        agent._owns_session_db = True
        return True
    except Exception:
        return False


@contextlib.contextmanager
def _profile_db(params: dict | None = None):
    """Yield the SessionDB for ``params['profile']`` (app-global remote mode).

    Closes dedicated profile handles; leaves the launch-profile shared handle open.
    Yields None when the db is unavailable.
    """
    profile = None
    if isinstance(params, dict):
        profile = (params.get("profile") or "").strip() or None
    db, owns = _db_for_profile(profile)
    try:
        yield db
    finally:
        if owns and db is not None:
            with contextlib.suppress(Exception):
                db.close()


def _response_profile_name(profile: str | None = None) -> str:
    """Profile name to report on session.* payloads.

    Prefer the RPC's requested profile when it is a real non-launch profile;
    otherwise the process launch profile.
    """
    name = (profile or "").strip()
    if name and _profile_home(name) is not None:
        return name
    return _current_profile_name()


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
        from hermes_cli.config import _expand_env_vars, read_user_config_raw

        p = Path(profile_home) / "config.yaml"
        if not p.exists():
            return None
        # Behavioral read of a NON-launch profile's config: load_config()
        # would resolve the ACTIVE profile's path, so read this profile's
        # file directly, then apply the same read-side pipeline as
        # _load_cfg (managed overlay + ${VAR} expansion). Fail-open.
        data = _apply_managed(read_user_config_raw(p))
        expanded = _expand_env_vars(data)
        if isinstance(expanded, dict):
            data = expanded
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


def _event_frame(event: str, sid: str, payload: dict | None = None) -> dict:
    params: dict = {"type": event, "session_id": sid}
    if payload is not None:
        params["payload"] = payload
    return {"jsonrpc": "2.0", "method": "event", "params": params}


def _emit(event: str, sid: str, payload: dict | None = None):
    write_json(_event_frame(event, sid, payload))


# Live client transports, one per connected WS peer (maintained by tui_gateway.ws).
# A session-less event from a background thread has neither a session transport
# nor a contextvar binding, so write_json would drop it on stdio — this registry
# is how such events reach WS clients at all. See _broadcast_global_event.
_live_transports: set[Transport] = set()
_live_transports_lock = threading.Lock()


def register_live_transport(transport: Transport | None) -> None:
    """Track a connected client transport for global broadcasts. Idempotent."""
    if transport is None:
        return
    with _live_transports_lock:
        _live_transports.add(transport)


def unregister_live_transport(transport: Transport | None) -> None:
    """Stop tracking a transport (call on disconnect). Idempotent."""
    with _live_transports_lock:
        _live_transports.discard(transport)


def _broadcast_global_event(event: str, payload: dict | None = None) -> None:
    """Fan a session-less, surface-global event (``skin.changed``) to every
    connected client. Emitters like the skin watcher run on background threads
    where ``write_json``'s ladder bottoms out at stdio and WS peers never see
    the frame. No registered transports (stdio TUI, tests) → plain ``_emit``,
    which that path already tees where it needs to go.
    """
    with _live_transports_lock:
        targets = list(_live_transports)

    if not targets:
        _emit(event, "", payload)
        return

    frame = _event_frame(event, "", payload)
    for transport in targets:
        try:
            transport.write(frame)
        except Exception:
            # One wedged peer must not stall the rest; disconnect teardown
            # unregisters it.
            logger.debug("global-event broadcast write failed type=%s", event, exc_info=True)


_compute_host_supervisor = None
_compute_host_supervisor_lock = threading.Lock()


def _inside_compute_host_child() -> bool:
    return os.environ.get("HERMES_COMPUTE_HOST_CHILD") == "1"


def _turn_isolation_enabled(cfg: dict | None = None) -> bool:
    if _inside_compute_host_child():
        return False
    isolation_cfg = cfg or _load_dashboard_process_isolation_config()
    return bool(isolation_cfg.get("turn_isolation"))


def _session_uses_compute_host(session: dict, cfg: dict | None = None) -> bool:
    if not _turn_isolation_enabled(cfg):
        return False
    # Phase 1 routes lazy/dashboard sessions whose live AIAgent has not been
    # built inside the serving process. Already-built in-process sessions keep
    # the historical path unless a prior isolated turn marked host ownership.
    return bool(session.get("_compute_host_active")) or (
        session.get("agent") is None and session.get("agent_ready") is not None
    )


def _get_compute_host_supervisor(cfg: dict | None = None):
    global _compute_host_supervisor
    isolation_cfg = cfg or _load_dashboard_process_isolation_config()
    with _compute_host_supervisor_lock:
        if _compute_host_supervisor is None:
            from tui_gateway.host_supervisor import HostSupervisor

            _compute_host_supervisor = HostSupervisor(
                rpc_sink=write_json,
                heartbeat_secs=int(isolation_cfg.get("compute_host_heartbeat_secs") or 15),
                respawn_max=int(isolation_cfg.get("compute_host_respawn_max") or 3),
            )
        return _compute_host_supervisor


def _compute_host_turn_frame(
    rid: str,
    sid: str,
    session: dict,
    text: Any,
    image_paths: list[str] | None = None,
    queued_prompt_generation: int | None = None,
) -> dict:
    with session["history_lock"]:
        history = list(session.get("history", []))
        history_version = int(session.get("history_version", 0))
        attached_images = (
            list(image_paths)
            if image_paths is not None
            else list(session.get("attached_images", []))
        )
    return {
        "type": "turn.start",
        "sid": sid,
        "request_id": rid,
        "session_key": session.get("session_key") or sid,
        "text": text,
        "history": history,
        "history_version": history_version,
        "cols": int(session.get("cols", 80) or 80),
        "cwd": _session_cwd(session),
        "profile_home": session.get("profile_home") or "",
        "model_override": session.get("model_override"),
        "reasoning_config_override": session.get("create_reasoning_override"),
        "service_tier_override": session.get("create_service_tier_override"),
        "source": _session_source(session),
        "attached_images": attached_images,
        "queued_prompt_generation": queued_prompt_generation,
    }


def _metadata_mirror(session: dict | None) -> dict:
    mirror = (session or {}).get("_metadata_mirror")
    return mirror if isinstance(mirror, dict) else {}


def _apply_compute_host_metadata_mirror(session: dict, frame: dict | None) -> None:
    """Mirror host-owned session metadata in the serving process.

    The compute host is the only writer of live agent/history state while turn
    isolation is active. The serving process keeps read metadata from the last
    host frame so UI reads do not construct a second in-process agent.
    """
    if not isinstance(frame, dict):
        return
    with session.get("history_lock", threading.Lock()):
        if frame.get("session_key"):
            session["session_key"] = str(frame.get("session_key"))
        if frame.get("history_version") is not None:
            try:
                session["history_version"] = max(
                    int(session.get("history_version", 0)),
                    int(frame.get("history_version") or 0),
                )
            except Exception:
                pass
        if frame.get("message_count") is not None:
            try:
                session["_metadata_message_count"] = int(frame.get("message_count") or 0)
            except Exception:
                pass
    info = frame.get("session_info")
    if isinstance(info, dict):
        mirror = dict(_metadata_mirror(session))
        mirror.update(info)
        session["_metadata_mirror"] = mirror
        session["_metadata_mirror_updated_at"] = time.time()


def _on_compute_host_turn_done(rid: str, sid: str, session: dict, frame: dict) -> None:
    is_error = frame.get("type") == "turn.error"
    with session["history_lock"]:
        if frame.get("session_key"):
            session["session_key"] = str(frame.get("session_key"))
        if frame.get("history_version") is not None:
            try:
                session["history_version"] = max(
                    int(session.get("history_version", 0)),
                    int(frame.get("history_version") or 0),
                )
            except Exception:
                pass
        session["running"] = False
        session["last_active"] = time.time()
        _clear_inflight_turn(session)
    if is_error:
        message = str(frame.get("message") or "compute host turn failed")
        _emit("message.complete", sid, {"text": f"Error: {message}", "status": "error"})
    _apply_compute_host_metadata_mirror(session, frame)
    try:
        info = _session_info(session.get("agent"), session)
    except TypeError:
        info = _session_info(session.get("agent"))
    if not frame.get("session_info_emitted"):
        _emit("session.info", sid, info)
    _drain_queued_prompt(rid, sid, session)


def _submit_prompt_to_compute_host(
    rid: str,
    sid: str,
    session: dict,
    text: Any,
    image_paths: list[str] | None = None,
    queued_prompt_generation: int | None = None,
) -> dict:
    cfg = _load_dashboard_process_isolation_config()
    frame = _compute_host_turn_frame(
        rid,
        sid,
        session,
        text,
        image_paths=image_paths,
        queued_prompt_generation=queued_prompt_generation,
    )

    def _complete(done: dict) -> None:
        # submit_turn reports a synchronous pipe failure through the callback
        # before re-raising. Leave the parent session untouched so prompt.submit
        # can fail open to the historical in-process path without emitting a
        # duplicate terminal error.
        if done.get("reason") == "send_failed":
            return
        _on_compute_host_turn_done(rid, sid, session, done)

    try:
        _get_compute_host_supervisor(cfg).submit_turn(frame, on_complete=_complete)
    except Exception as exc:
        return _err(rid, 5019, f"compute-host dispatch failed: {exc}")
    with session["history_lock"]:
        session["_compute_host_active"] = True
        if image_paths is None:
            session["attached_images"] = []
    return _ok(rid, {"status": "streaming", "turn_isolation": True})


def _send_compute_host_control(
    sid: str,
    *,
    route_name: str,
    command: str = "",
    payload: dict | None = None,
    wait: bool = True,
    timeout: float = 30.0,
) -> dict:
    frame = dict(payload or {})
    frame.setdefault("type", "control")
    frame.setdefault("command", command)
    return _get_compute_host_supervisor().control(
        sid,
        route_name=route_name,
        payload=frame,
        wait=wait,
        timeout=timeout,
    )


def _emit_approval_request(sid: str, data: dict | None) -> None:
    """Emit an ``approval.request`` event to the TUI client with the command
    redacted. The approval payload is built from the RAW command string, so a
    credential-shaped value Tirith flagged would otherwise be echoed verbatim
    to the TUI client (#48456 — third egress transport alongside the chat
    platforms and the SSE/API stream fixed in #50767). Reuse the shared gateway
    seam so all approval transports redact consistently."""
    payload = dict(data or {})
    if "choices" not in payload:
        if payload.get("smart_denied"):
            payload["choices"] = ["once", "deny"]
        elif payload.get("allow_permanent") is False:
            payload["choices"] = ["once", "session", "deny"]
        elif "allow_permanent" in payload:
            payload["choices"] = ["once", "session", "always", "deny"]
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


def _current_session_steer_authority(
    session_id: str,
) -> tuple[Transport | None, dict | None]:
    """Resolve unforgeable steering authority for this exact RPC context.

    The public session id is only a lookup hint. Authority is the identity of
    both the request's ContextVar-bound transport and the live in-memory
    session record currently stored under that id. Session transport rebinding,
    removal, or id reuse therefore invalidates an earlier generation.
    """
    transport = current_transport()
    if transport is None or not session_id:
        return None, None
    expected_session = _current_runtime_session_record.get()
    with _sessions_lock:
        session = _sessions.get(session_id)
        if (
            session is None
            or (expected_session is not None and session is not expected_session)
            or session.get("transport") is not transport
        ):
            return None, None
        return transport, session


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


# The deferred prompt path waits in short slices so a cancel is honored
# promptly and a slow build can be reported to the client exactly once.
_AGENT_BUILD_WAIT_SLICE = 5.0
_AGENT_BUILD_SLOW_NOTICE_AFTER = 30.0
_AGENT_BUILD_SLOW_NOTICE_KEY = "agent-build-slow"


def _agent_build_wait_cap() -> float:
    """Upper bound (seconds) a submitted prompt waits for the deferred agent
    build before failing permanently. ``agent.build_wait_timeout`` in
    config.yaml overrides the 600s default (raise it for deployments with
    many slow/unreachable MCP servers or high-latency provider metadata)."""
    try:
        agent_cfg = _load_cfg().get("agent") or {}
        raw = agent_cfg.get("build_wait_timeout")
        if raw is not None:
            value = float(raw)
            if value > 0:
                return value
    except Exception:
        pass
    return 600.0


def _wait_agent_for_prompt(session: dict, rid: str, sid: str) -> dict | None:
    """Patient variant of ``_wait_agent`` for the deferred prompt.submit path.

    The flat 30s ``_wait_agent`` ceiling was a message-eating cliff (#63078):
    ``prompt.submit`` has already returned ``{"status": "streaming"}``, the
    user's first message IS the turn in flight, and the deferred agent build
    (MCP discovery with per-server retry backoff, synchronous model-metadata
    HTTP, skills scanning) routinely outlives 30 seconds on cold starts. On
    timeout the old path emitted an error EVENT and returned without ever
    calling ``_run_prompt_submit`` — the first message was permanently
    discarded while the build finished successfully in the background, leaving
    the blank first session.

    This wait instead:
      - keeps the pending prompt attached to this (already off-RPC) thread and
        delivers it the moment the still-running build completes;
      - waits in short slices so a cancel (session.interrupt / session churn)
        is honored promptly instead of after the full timeout;
      - tells the client once, via a keyed notice, when the build outlives
        ``_AGENT_BUILD_SLOW_NOTICE_AFTER`` — the wait is patient but never
        silent;
      - fails permanently only when the build itself fails: the build thread
        died without signalling ready, or the bounded cap
        (``agent.build_wait_timeout``, default 600s — no infinite waits)
        expired on a genuinely hung build.

    Returns ``None`` on success OR when the turn was cancelled mid-wait (the
    caller's cancel branch owns that messaging), an ``_err`` dict otherwise.
    """
    ready = session.get("agent_ready")
    if ready is None:
        return None
    start = time.monotonic()
    cap = _agent_build_wait_cap()
    notified_slow = False
    while not ready.wait(timeout=_AGENT_BUILD_WAIT_SLICE):
        with session["history_lock"]:
            cancelled = session.get("_turn_cancel_requested") or not session.get(
                "running"
            )
        if cancelled:
            # The caller's cancel/not-running branch emits the user-visible
            # event for this — bail without an error of our own.
            return None
        waited = time.monotonic() - start
        if waited >= cap:
            return _err(
                rid,
                5032,
                f"agent initialization timed out after {int(waited)}s — "
                "your message was not sent; retry once the session is ready",
            )
        build_thread = session.get("_agent_build_thread")
        if (
            build_thread is not None
            and not build_thread.is_alive()
            and not ready.is_set()
        ):
            # _build's ``finally`` guarantees ready.set(); a dead thread with
            # ready still unset means the build died hard (interpreter-level
            # kill) — don't wait on a corpse for the rest of the cap.
            return _err(
                rid,
                5032,
                session.get("agent_error")
                or "agent initialization failed before completing",
            )
        if not notified_slow and waited >= _AGENT_BUILD_SLOW_NOTICE_AFTER:
            # One keyed, replace-in-place notice: the desktop shows it as a
            # toast, the TUI in its status bar. Without this the extended wait
            # would be exactly the silent hang this function exists to fix.
            notified_slow = True
            _emit(
                "notification.show",
                sid,
                {
                    "text": (
                        "Still starting the agent (tool discovery / model "
                        "setup) — your message will be sent as soon as it's "
                        "ready."
                    ),
                    "level": "info",
                    "kind": "agent",
                    "ttl_ms": None,
                    "key": _AGENT_BUILD_SLOW_NOTICE_KEY,
                    "id": _AGENT_BUILD_SLOW_NOTICE_KEY,
                },
            )
    if notified_slow:
        _emit("notification.clear", sid, {"key": _AGENT_BUILD_SLOW_NOTICE_KEY})
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

        notify_registered = False
        home_token = None
        secret_token = None
        session_db = None
        owns_db = False
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
                    from agent.secret_scope import build_profile_secret_scope, set_secret_scope

                    secret_token = set_secret_scope(build_profile_secret_scope(Path(profile_home)))
                except Exception:
                    pass
                try:
                    from hermes_state import SessionDB

                    # DEDICATED handle — ours until _transfer_db_to_agent hands
                    # it to the built agent in the finally below. Every path
                    # that leaves this build without that transfer (the except
                    # below, and a session reaped mid-build) must close it.
                    session_db = SessionDB(db_path=Path(profile_home) / "state.db")
                    owns_db = True
                except Exception:
                    session_db = None

            try:
                from tui_gateway.entry import ensure_mcp_discovery_started

                ensure_mcp_discovery_started()
            except Exception:
                logger.warning("MCP discovery startup failed", exc_info=True)

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

            # No eager slash-worker pre-warm: slash.exec spawns one on demand
            # (its error path already relies on that respawn to recover from a
            # dead worker). Each worker child runs its own MCP discovery
            # (#61891), so pre-warming one per session forks the full stdio
            # MCP fleet — ~20 OS processes per retained session on a config
            # with a few stdio servers — even for sessions that never run a
            # worker-routed command. Sessions held by a live transport are
            # never reaped, so with the desktop app open for days those
            # fleets accumulate until the OS refuses new process spawns.

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
            if secret_token is not None:
                try:
                    from agent.secret_scope import reset_secret_scope

                    reset_secret_scope(secret_token)
                except Exception:
                    pass
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
            # Dedicated profile handle: hand it to the agent that will actually
            # be torn down, or close it here when no such agent exists. Both
            # non-transfer cases are real: the except above (build raised, so
            # nothing holds the handle) and `replaced` (the session was reaped
            # mid-build, so this agent is discarded and _teardown_session will
            # never reach it). Transferring to a discarded agent would leak the
            # handle exactly as before.
            if owns_db and session_db is not None:
                built = None if replaced else current.get("agent")
                if not _transfer_db_to_agent(built, session_db):
                    with contextlib.suppress(Exception):
                        session_db.close()
            ready.set()

    build_thread = threading.Thread(target=_build, daemon=True)
    # Handle for _wait_agent_for_prompt: a dead build thread with agent_ready
    # still unset means the build died hard — waiters must not sit out the
    # full cap on a corpse.
    session["_agent_build_thread"] = build_thread
    build_thread.start()


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

    When ``TERMINAL_ENV`` is unset (dashboard/TUI process) the config's
    ``terminal.backend`` is consulted as a fallback so the non-local cwd
    resolution path is taken even when the dashboard entrypoint did not call
    ``apply_terminal_config_to_env`` on its own ``os.environ``.
    """
    backend = (os.environ.get("TERMINAL_ENV") or "").strip().lower()
    if not backend or backend == "local":
        # Fall back to config when TERMINAL_ENV is unset (dashboard/TUI process
        # never calls apply_terminal_config_to_env on os.environ).
        try:
            terminal_cfg = _load_cfg().get("terminal", {})
            if isinstance(terminal_cfg, dict):
                cfg_backend = str(terminal_cfg.get("backend") or "").strip().lower()
                if cfg_backend and cfg_backend != "local":
                    backend = cfg_backend
        except Exception:
            pass

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
        if backend == "ssh":
            return "~"

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


# Sources whose launch directory is an artifact of how the app was started, not
# a workspace the user picked. Everything else is terminal-started: the process
# runs in a directory the user deliberately cd'd into.
_LAUNCH_CWD_NOT_A_WORKSPACE = {"desktop"}


def _persisted_session_cwd(session: dict) -> str | None:
    """The cwd to stamp on the session's DB row, or None to leave it unset.

    See :func:`_ensure_session_db_row` for why the launch directory counts as a
    workspace for terminal sessions but not for the desktop.
    """
    if session.get("explicit_cwd"):
        return _session_cwd(session)
    if _session_source(session) in _LAUNCH_CWD_NOT_A_WORKSPACE:
        return None
    # Only the session's OWN directory. `_session_cwd` falls back to the
    # gateway-wide completion cwd, which belongs to no session in particular —
    # stamping that would invent a workspace for a session that never had one.
    return str(session.get("cwd") or "") or None


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


def _reconcile_session_cwd_from_terminal(session: dict | None) -> bool:
    """Re-anchor a session that SETTLED in another git checkout. Returns moved.

    An agent told to work in a fresh worktree does exactly that — `git worktree
    add`, `cd` into it, and every later command runs there — but the session
    stayed pinned to wherever it started, so the desktop kept labelling the chat
    with the primary checkout's branch while all the work landed elsewhere.

    A plain `cd` is deliberately NOT a workspace move (see
    ``_apply_project_workspace``): browsing to /tmp to read a log must not
    re-home the chat. What we adopt here is narrower — the session's recorded
    cwd is in a DIFFERENT working tree of the SAME repository (the shape
    ``git worktree add`` produces). Everything else — a non-git workspace
    stepping into a repo, or a git workspace visiting an unrelated repo — is
    a browsing visit, and a user's explicitly chosen workspace is never
    overridden at all.

    Local backends only: a remote/SSH cwd names a path on the host, which this
    gateway can neither stat nor probe with git.
    """
    if not session or not _is_local_terminal_backend():
        return False

    # A workspace the user (or GUI) explicitly chose is never overridden by
    # where the agent's terminal happened to settle — only another explicit
    # action (`_set_session_cwd`, a project switch) moves it. A cwd this very
    # function adopted is marked `cwd_from_settle` so a session can keep
    # following the agent through successive worktrees.
    if session.get("explicit_cwd") and not session.get("cwd_from_settle"):
        return False

    try:
        from tools.terminal_tool import get_session_cwd

        recorded = get_session_cwd(session.get("session_key") or "")
    except Exception:
        return False

    if not recorded:
        return False

    resolved = os.path.abspath(os.path.expanduser(str(recorded)))
    current = os.path.abspath(os.path.expanduser(_session_cwd(session)))
    if resolved == current or not os.path.isdir(resolved):
        return False

    # The worktree ROOT, not the common repo root: folding worktrees together
    # here is exactly what hides the move we're looking for.
    landed = _git_repo_root_for_cwd(resolved)
    current_root = _git_repo_root_for_cwd(current)
    # A relocation is a move between two DIFFERENT git working trees. When the
    # session's own workspace is not in a git repo, the agent stepping into one
    # to read a file or run a command is a browsing visit, not a re-home:
    # adopting it would hijack a non-git workspace onto whatever repo a tool
    # call touched first (e.g. a home-directory session pinned to the checkout
    # it read a file from).
    if not landed or not current_root or landed == current_root:
        return False

    # And only between checkouts of the SAME repository — the shape a real
    # `git worktree add` produces (linked worktrees share the common .git
    # dir). Settling in an UNRELATED repo (`cd ~/other-project && git log`)
    # is likewise a visit: adopting it would re-home the chat onto whatever
    # foreign repo the terminal last touched.
    landed_common = _git_common_repo_root_for_cwd(resolved)
    current_common = _git_common_repo_root_for_cwd(current)
    if not landed_common or landed_common != current_common:
        return False

    session["cwd"] = resolved
    # The session works here now, so this is its workspace — a desktop chat
    # whose cwd was an unpersisted launch artifact earns a real row. The
    # settle marker keeps this adoption overridable by the NEXT settle while
    # still yielding to a user's explicit choice (see the guard above).
    session["explicit_cwd"] = True
    session["cwd_from_settle"] = True
    _register_session_cwd(session)

    with _session_db(session) as db:
        if db is not None:
            try:
                db.update_session_cwd(session.get("session_key", ""), resolved)
            except Exception:
                logger.debug("failed to persist settled session cwd", exc_info=True)

    _persist_session_git_meta(session, resolved)
    return True


def _emit_settled_session_info(sid: str, session: dict, agent) -> None:
    """Emit end-of-turn ``session.info``, reconciling a settled cwd first.

    The turn is over, so the agent has stopped moving: this is the one moment
    where its recorded cwd is a stable answer to "where does this session
    work". Reconciling before building the payload means the same event that
    already tells the desktop the turn ended also carries the new cwd/branch —
    the client follows it with no new event type and no extra round trip.
    """
    try:
        _reconcile_session_cwd_from_terminal(session)
    except Exception:
        logger.debug("failed to reconcile settled session cwd", exc_info=True)
    _emit("session.info", sid, _session_info(agent, session))


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

    A cwd the user *chose* is always persisted. When they made no explicit
    choice the launch directory stands in, and whether that is meaningful
    depends on how the session was started:

    * The desktop launches from wherever the app bundle was opened (often ``/``
      or the user's home), so stamping that would file every unpicked chat under
      a folder the user never chose. Those stay null and group under "No
      workspace", which is the desired default.
    * A terminal session (``hermes`` / ``hermes --tui`` / CLI) is started from a
      directory the user deliberately ``cd``'d into — that IS the workspace, and
      it is also where the agent's terminal actually runs. Dropping it stranded
      the session with no cwd AND no git_repo_root, so the sidebar could never
      place it under its project.
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
                base_url=model_config.get("base_url") or None,
                model=model_config.get("model") or row_model or None,
            )
            if healed:
                model_config["provider"] = healed
        except Exception:
            logger.debug(
                "custom provider identity recovery failed (db row)", exc_info=True
            )
    if (reasoning := session.get("create_reasoning_override")) is not None:
        model_config["reasoning_config"] = reasoning
    create_service_tier_override = session.get("create_service_tier_override")
    if create_service_tier_override is not None:
        # Empty string is the in-memory sentinel for an explicit normal tier:
        # it bypasses _make_agent's profile fallback without sending a bogus
        # service_tier value to the provider. Persist a durable marker so resume
        # can distinguish that choice from an omitted/inherited tier.
        model_config["service_tier"] = create_service_tier_override or "normal"
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
            cwd=_persisted_session_cwd(session),
            # Self-describing rows: aggregators that merge multiple profile DBs
            # into one list can't rely on which file a row came from alone. NULL
            # means the launch/default profile (matches run_agent's convention).
            profile_name=Path(profile_home).name if profile_home else None,
        )
    except Exception as exc:
        # Disk-full is not a soft failure: if we swallow it here, prompt.submit
        # returns {"status":"streaming"} and the user's message vanishes with
        # no toast. Re-raise so the submit handler can return a real RPC error.
        from hermes_state import is_disk_full_error

        if is_disk_full_error(exc):
            raise
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
            # Bounded-chunk transactions (see #23254): a branch seed can be
            # hundreds of rows; chunking keeps each BEGIN IMMEDIATE short so
            # concurrent writers aren't starved. Recovery semantics match the
            # old per-row loop (mid-copy failure leaves a partial seed with
            # _branch_seed_persisted unset).
            db.append_messages_batch(
                key,
                [
                    {
                        "role": msg.get("role", "user"),
                        "content": msg.get("content"),
                        # Preserve the parent's original message timestamps —
                        # append_message would otherwise stamp time.time() and the
                        # branch's copied history would all appear authored "now".
                        "timestamp": msg.get("timestamp"),
                    }
                    for msg in seed
                ],
                chunk_rows=500,
            )
            session["_branch_seed_persisted"] = True
        except Exception as exc:
            from hermes_state import is_disk_full_error

            if is_disk_full_error(exc):
                raise
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
    from hermes_constants import translate_cwd_for_wsl_backend

    cwd = translate_cwd_for_wsl_backend(str(cwd))
    resolved = os.path.abspath(os.path.expanduser(cwd))
    if not os.path.isdir(resolved):
        raise ValueError(f"working directory does not exist: {cwd}")
    session["cwd"] = resolved
    # An explicit user choice — persist it as the workspace (and let a later
    # lazy row creation persist it too, not the launch-dir fallback).
    session["explicit_cwd"] = True
    # A user's choice supersedes any earlier settle-adopted cwd: from here on
    # the terminal wandering must not move the workspace again.
    session["cwd_from_settle"] = False
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


_DASHBOARD_TURN_ISOLATION_DEFAULT = False
_DASHBOARD_COMPUTE_HOST_HEARTBEAT_SECS_DEFAULT = 15
_DASHBOARD_COMPUTE_HOST_RESPAWN_MAX_DEFAULT = 3


def _coerce_int_config_value(value: Any, default: int, *, min_value: int) -> int:
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return default
    return coerced if coerced >= min_value else default


def _load_dashboard_process_isolation_config(cfg: dict | None = None) -> dict[str, Any]:
    """Return dashboard process-isolation config with read-site defaults.

    ``_load_cfg()`` intentionally returns the user ``config.yaml`` plus the
    managed overlay and ``${VAR}`` expansion; it does not deep-merge
    ``hermes_cli.config.DEFAULT_CONFIG``. Keep
    the Phase-0 defaults here so dashboard runtime and the REST editor's
    DEFAULT_CONFIG-backed schema cannot drift.
    """
    root = _load_cfg() if cfg is None else cfg
    dashboard = root.get("dashboard") if isinstance(root, dict) else {}
    if not isinstance(dashboard, dict):
        dashboard = {}
    return {
        "turn_isolation": is_truthy_value(
            dashboard.get("turn_isolation"),
            default=_DASHBOARD_TURN_ISOLATION_DEFAULT,
        ),
        "compute_host_heartbeat_secs": _coerce_int_config_value(
            dashboard.get("compute_host_heartbeat_secs"),
            _DASHBOARD_COMPUTE_HOST_HEARTBEAT_SECS_DEFAULT,
            min_value=1,
        ),
        "compute_host_respawn_max": _coerce_int_config_value(
            dashboard.get("compute_host_respawn_max"),
            _DASHBOARD_COMPUTE_HOST_RESPAWN_MAX_DEFAULT,
            min_value=0,
        ),
    }


def _load_cfg_raw() -> dict:
    """Read the active profile's config.yaml EXACTLY as written (write-back primitive).

    ONLY legal for read→mutate→``_save_cfg`` round-trips (and raw-file
    inspection): merging defaults, the managed overlay, or ``${VAR}``
    expansion here would be persisted into the user's file on the next
    save. Behavioral reads must use :func:`_load_cfg`, which layers the
    managed overlay + env expansion on top of this raw read.
    """
    global _cfg_cache, _cfg_mtime, _cfg_path
    try:
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
                return copy.deepcopy(_cfg_cache)
        if p.exists():
            from hermes_cli.config import read_user_config_raw
            data = read_user_config_raw(p)
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
        return data
    except Exception:
        pass
    return {}


def _load_cfg() -> dict:
    """Behavioral config read: raw user file + managed overlay + ${VAR} expansion.

    Delegates the disk read to :func:`_load_cfg_raw` (shared cache), then
    applies the same read-side pipeline as the canonical
    ``hermes_cli.config.load_config_readonly`` — managed-scope overlay and
    ``${ENV_VAR}`` expansion — minus the DEFAULT_CONFIG merge (callers here
    treat a missing key as "unset" and apply their own defaults; merging
    would also break ``_load_cfg() == {}`` sentinels). Do NOT pass the
    result to ``_save_cfg``: use ``_load_cfg_raw()`` for write-back
    round-trips or expanded/overlaid values get persisted into the user's
    file.
    """
    cfg = _apply_managed(_load_cfg_raw())
    try:
        from hermes_cli.config import _expand_env_vars

        expanded = _expand_env_vars(cfg)
        if isinstance(expanded, dict):
            cfg = expanded
    except Exception:
        pass
    return cfg


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
        # Derive the live conversation id so terminal/execute_code subprocesses
        # can read HERMES_SESSION_ID. Without this, set_session_vars leaves the
        # session-id contextvar as "" (explicitly empty), and the subprocess-env
        # bridge treats that as authoritative — NOT falling back to os.environ —
        # so every command in a dashboard/TUI/web session saw an empty
        # HERMES_SESSION_ID even though agent_init set it via
        # set_current_session_id(). Prefer the agent's durable session_id, then
        # fall back to the session_key (matching the id derivation used at
        # session-finalize), so an identified session is never left blank.
        session_id = session_key
        with _sessions_lock:
            for sess in list(_sessions.values()):
                if sess.get("session_key") == session_key:
                    source = _session_source(sess)
                    session_id = (
                        getattr(sess.get("agent"), "session_id", None) or session_key
                    )
                    break
        return set_session_vars(
            session_key=session_key,
            session_id=session_id,
            source=source,
            cwd=resolved,
            ui_session_id=ui_session_id,
            cron_session="",
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


def _block(event: str, sid: str, payload: dict, timeout: float | None = 300) -> str:
    rid = uuid.uuid4().hex[:8]
    ev = threading.Event()
    with _prompt_lock:
        _pending[rid] = (sid, ev)
        payload["request_id"] = rid
        _pending_prompt_payloads[rid] = (event, dict(payload))
    answered = False
    answer = ""
    answer_present = False
    try:
        _emit(event, sid, payload)
        # Natural Event semantics: None → wait forever (clarify configured with
        # clarify_timeout <= 0, released only by a real answer or
        # session.interrupt), 0 → return immediately, > 0 → bounded wait.
        answered = ev.wait(timeout)
    finally:
        with _prompt_lock:
            _pending.pop(rid, None)
            _pending_prompt_payloads.pop(rid, None)
            answer_present = rid in _answers
            answer = _answers.pop(rid, "")

    # Emit an `.expire` notification on timeout for every blocking request type
    # whose `*.respond` handler tolerates a late reply (allow_expired=True).
    # All four blocking bridges — secret, sudo, clarify, terminal.read — share
    # the same lifecycle: the tool gives up on timeout and returns empty, but a
    # slow renderer (or a reconnect that dropped tool.complete) can still answer
    # afterward. Without this the late `*.respond` would hit the generic 4009
    # "no pending request" error and clients would surface a raw JSON-RPC string.
    if not answered and not answer_present and event in {
        "secret.request",
        "sudo.request",
        "clarify.request",
        "terminal.read.request",
        "preview.read.request",
    }:
        _emit(
            f"{event.removesuffix('.request')}.expire",
            sid,
            {"request_id": rid},
        )
    return answer


def _clarify_timeout_seconds() -> float | None:
    """Clarify wait (seconds) for the TUI/desktop bridge, from the same
    canonical config the messaging gateway and CLI use. Falls back to the
    historical 300s _block default if config can't be read. ``<= 0`` in config
    means unlimited and is returned as ``None`` (never auto-skip)."""
    try:
        from tools.clarify_gateway import get_clarify_timeout
        timeout = get_clarify_timeout()
        return timeout if timeout > 0 else None
    except Exception:
        return 300


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
            # Paired palettes: the TUI detects the terminal's polarity and
            # prefers the matching hand-tuned block over adapting `colors`.
            "light_colors": skin.light_colors,
            "dark_colors": skin.dark_colors,
            "branding": skin.branding,
            "banner_logo": skin.banner_logo,
            "banner_hero": skin.banner_hero,
            "tool_prefix": skin.tool_prefix,
            "help_header": (skin.branding or {}).get("help_header", ""),
        }
    except Exception:
        return {}


# Signature of the last skin broadcast: (name, active user-file mtime). Lets the
# per-tool reconcile fire ``skin.changed`` on any real move — a name switch OR a
# live color edit to the active skin — and nothing else.
_last_skin_sig: tuple[str, float | None] | None = None


def _skin_sig() -> tuple[str, float | None]:
    """(active skin name, its user-file mtime). Built-ins have no file, so only
    their name moves; a user skin's mtime lets an in-place color edit repaint too."""
    name = str((_load_cfg().get("display") or {}).get("skin") or "default")
    override = get_hermes_home_override()
    home = override if isinstance(override, str) and override else _hermes_home
    try:
        mtime: float | None = (Path(home) / "skins" / f"{name}.yaml").stat().st_mtime
    except OSError:
        mtime = None
    return name, mtime


def _note_skin_broadcast() -> None:
    """Sync the reconcile baseline after the /skin RPC emits, so the per-tool
    check doesn't re-broadcast the skin /skin just applied."""
    global _last_skin_sig
    try:
        _last_skin_sig = _skin_sig()
    except Exception:
        pass


def _broadcast_skin_if_changed() -> None:
    """Emit ``skin.changed`` when the active skin moved — the agent switched it
    (``hermes config set display.skin``) OR edited the active skin's colors in
    place ("I don't like that coral" → tweak the YAML).

    Routes through the SAME live path as ``/skin`` so every surface (TUI + desktop)
    repaints, no slash command. The signature check is a dict lookup + one stat,
    so polling it is ~free.
    """
    global _last_skin_sig
    try:
        sig = _skin_sig()
    except Exception:
        return
    if sig == _last_skin_sig:
        return
    _last_skin_sig = sig
    try:
        _broadcast_global_event("skin.changed", resolve_skin())
    except Exception:
        pass


def _watcher_home() -> Path:
    """Active profile home for the change watcher's signature probes."""
    override = get_hermes_home_override()
    return Path(override if isinstance(override, str) and override else _hermes_home)


def _pet_sig() -> tuple:
    """(slug, spritesheet revision, scale) of the active pet — ("off",) when none.

    Cheap by construction: config comes from the mtime-cached ``_load_cfg`` and
    the sheet revision is one stat. Moves when ``/pet`` (de)activates a pet, the
    hatch flow rebuilds a sheet, or the scale changes."""
    display = _load_cfg().get("display") or {}
    pet_cfg = display.get("pet") if isinstance(display.get("pet"), dict) else {}
    if not pet_cfg or not pet_cfg.get("enabled"):
        return ("off",)
    try:
        enabled, pet, scale = _pet_active_selection()
        if not enabled or pet is None or not pet.exists:
            return ("off",)
        return (pet.slug, _pet_sheet_revision(pet.spritesheet), scale)
    except Exception:  # noqa: BLE001 - cosmetic, never break the watcher
        return ("off",)


def _pet_changed_payload() -> dict:
    """``pet.info.meta``-shaped payload for ``pet.changed`` — enough for the
    renderer to decide whether the heavy sprite payload needs a refetch."""
    try:
        enabled, pet, scale = _pet_active_selection()
        if not enabled or pet is None or not pet.exists:
            return {"enabled": False}
        return {
            "enabled": True,
            "slug": pet.slug,
            "displayName": pet.display_name,
            "scale": scale,
            "spritesheetRevision": _pet_sheet_revision(pet.spritesheet),
        }
    except Exception:  # noqa: BLE001 - cosmetic, never break the watcher
        return {"enabled": False}


def _cron_sig():
    """mtime of the profile's cron/jobs.json — moves on create/edit/pause/
    remove AND on scheduler tick bookkeeping (last_run/next_run)."""
    try:
        return (_watcher_home() / "cron" / "jobs.json").stat().st_mtime_ns
    except OSError:
        return None


def _sessions_sig():
    """Newest mtime across state.db and its WAL — the cross-process change
    signal. Messaging-gateway turns and cron runs are written by OTHER
    processes that never touch this gateway's transports; the shared SQLite
    file is the one thing they all move (#58671)."""
    home = _watcher_home()
    sig = None
    for name in ("state.db", "state.db-wal"):
        try:
            mtime = (home / name).stat().st_mtime_ns
        except OSError:
            continue
        sig = mtime if sig is None else max(sig, mtime)
    return sig


def _platforms_sig():
    """mtime of gateway_state.json — the messaging gateway process persists
    platform connect/disconnect/health there, so its movement is the
    "connection status changed" signal for the Messaging page."""
    try:
        return (_watcher_home() / "gateway_state.json").stat().st_mtime_ns
    except OSError:
        return None


def _pairing_sig():
    """Newest mtime across every profile's pairing store.

    An unknown DMer's pending code is written by the messaging gateway — a
    DIFFERENT process that never touches this gateway's transports — so the
    files are the only shared signal. ``platforms.changed`` cannot stand in
    for this: it tracks connect/disconnect/health, and a pairing request
    moves nothing in gateway_state.json.
    """
    home = _watcher_home()
    sig = None
    # Global store (legacy `pairing/` and consolidated `platforms/pairing/`)
    # plus every named profile's own — the Messaging page can be scoped to any
    # of them, and a request landing in a profile store must still tick.
    roots = [home / "pairing", home / "platforms" / "pairing"]
    try:
        for profile_dir in (home / "profiles").iterdir():
            roots.append(profile_dir / "pairing")
            roots.append(profile_dir / "platforms" / "pairing")
    except OSError:
        pass

    for root in roots:
        try:
            entries = list(root.iterdir())
        except OSError:
            continue
        for entry in entries:
            # Only the pending/approved ledgers — _rate_limits.json moves on
            # every unauthorized DM, including ones that produce no new row.
            if not entry.name.endswith(("-pending.json", "-approved.json")):
                continue
            try:
                mtime = entry.stat().st_mtime_ns
            except OSError:
                continue
            sig = mtime if sig is None else max(sig, mtime)
    return sig


# Watched change signals: event → (check interval, signature fn, payload fn).
# Signatures are stat/dict-lookup cheap, same bar as the skin watcher; the
# check interval keeps the pricier probes (pet resolves the active sheet off
# disk) off the 0.5s tick.
_CHANGE_WATCHES: dict[str, tuple[float, Any, Any]] = {
    "pet.changed": (2.0, _pet_sig, _pet_changed_payload),
    "cron.changed": (1.0, _cron_sig, lambda: {}),
    "sessions.changed": (0.5, _sessions_sig, lambda: {}),
    "platforms.changed": (2.0, _platforms_sig, lambda: {}),
    "pairing.changed": (2.0, _pairing_sig, lambda: {}),
}

# state.db moves on every message append during a streaming turn, and the
# gateway rewrites gateway_state.json for in-flight-count bookkeeping; the
# floor coalesces those bursts to one broadcast per window (trailing edge
# included — a floored change keeps its old signature and re-fires next tick).
_CHANGE_BROADCAST_FLOOR_S = {"sessions.changed": 2.0, "platforms.changed": 5.0}

_change_sigs: dict[str, Any] = {}
_change_checked_at: dict[str, float] = {}
_change_broadcast_at: dict[str, float] = {}


def _broadcast_watched_changes(now: float | None = None) -> None:
    """One pass over ``_CHANGE_WATCHES``: recompute due signatures, broadcast
    the events whose signature moved. First sighting seeds silently so a
    gateway boot never fires a spurious refresh storm."""
    now = time.monotonic() if now is None else now
    for event, (interval, sig_fn, payload_fn) in _CHANGE_WATCHES.items():
        if now - _change_checked_at.get(event, -interval) < interval:
            continue
        _change_checked_at[event] = now
        try:
            sig = sig_fn()
        except Exception:  # noqa: BLE001 - a broken probe must not kill the loop
            continue
        if event not in _change_sigs:
            _change_sigs[event] = sig
            continue
        if sig == _change_sigs[event]:
            continue
        floor = _CHANGE_BROADCAST_FLOOR_S.get(event, 0.0)
        if floor and now - _change_broadcast_at.get(event, -floor) < floor:
            # Floored: leave the old signature in place so the change re-fires
            # once the window opens (the trailing edge of the burst).
            continue
        _change_sigs[event] = sig
        _change_broadcast_at[event] = now
        try:
            _broadcast_global_event(event, payload_fn())
        except Exception:  # noqa: BLE001
            pass


_skin_watcher_started = False


def _ensure_skin_watcher() -> None:
    """Watch cheap on-disk signatures and broadcast change events — so a skin
    Hermes activates, a pet ``/pet`` adopts, a cron the scheduler fires, or a
    messaging turn another process writes goes live on every surface within a
    couple seconds, on its own, with no client-side poll in the loop.
    Idempotent; started at gateway.ready. (Named for its original skin-only
    duty; it is the process's one change watcher.)"""
    global _skin_watcher_started
    if _skin_watcher_started:
        return
    _skin_watcher_started = True
    _note_skin_broadcast()  # seed the baseline so only a real change repaints

    def _loop() -> None:
        while True:
            time.sleep(0.5)
            _broadcast_skin_if_changed()
            _broadcast_watched_changes()

    threading.Thread(target=_loop, name="hermes-change-watcher", daemon=True).start()


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
    # No env seed and no config preference: fall back to the cost-safe silent
    # default (catalog-labeled, cache-only read), never an expensive Anthropic
    # flagship the user didn't pick.
    try:
        from hermes_cli.models import get_preferred_silent_default_model

        return get_preferred_silent_default_model()
    except Exception:
        return "z-ai/glm-5.2"


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
    # the durable ``custom:<name>`` menu key from the stored base_url, then
    # from the entry that serves the stored model, falling back to the
    # configured provider when the row has neither (the recurring Desktop/TUI
    # regression vector). If none names a real entry,
    # drop the bare provider entirely so resume falls back to the configured
    # default rather than the broken OpenRouter route.
    if provider.strip().lower() == "custom":
        healed = None
        try:
            from hermes_cli.runtime_provider import canonical_custom_identity

            healed = canonical_custom_identity(
                base_url=base_url or None, model=model or None
            )
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
    if service_tier.lower() == "normal":
        # None means "inherit the profile" at _make_agent. Empty string is a
        # real override that means "do not request a priority service tier".
        overrides["service_tier_override"] = ""
    elif service_tier:
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
                    canonical_custom_identity(
                        base_url=base_url, model=model or None
                    )
                    or provider
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
        create_service_tier_override = session.get("create_service_tier_override")
        if create_service_tier_override is not None:
            # _runtime_model_config sees agent.service_tier=None for explicit
            # normal and would otherwise erase the distinction on every live
            # metadata persist.
            model_config["service_tier"] = create_service_tier_override or "normal"
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


# Stable leading text of the model-switch marker, shared by the builder and the
# dedup below. Only the newest marker is meaningful (it names the *currently*
# active model); older ones are stale and would otherwise be re-sent to the
# provider on every turn (#65891).
_MODEL_SWITCH_MARKER_PREFIX = "[System: The active model for this chat has changed to "


def _is_model_switch_marker(entry: Any) -> bool:
    """Whether a history entry is a (self-replacing) model-switch marker."""
    if not isinstance(entry, dict):
        return False
    content = entry.get("content")
    return isinstance(content, str) and content.startswith(_MODEL_SWITCH_MARKER_PREFIX)


def _append_model_switch_marker(session: dict | None, *, model: str, provider: str) -> None:
    """Record a real system-history pivot after a live model switch.

    Only the most recent marker is kept: each new switch first strips any
    prior model-switch markers from the live history, so N switches leave one
    marker (naming the active model), not N stale ones accumulating tokens on
    every subsequent API call (#65891). The in-memory history is the payload
    re-sent each turn; the dedup is self-healing across resumes because the
    next switch collapses whatever markers a reload brought back.
    """
    if not session:
        return
    session_key = str(session.get("session_key") or "").strip()
    if not session_key:
        return

    provider_part = f" via provider {provider}" if provider else ""
    marker = (
        f"{_MODEL_SWITCH_MARKER_PREFIX}"
        f"{model}{provider_part}. From this point forward, use this runtime "
        "metadata when answering questions about what model/provider is active.]"
    )
    # Persist as a user message, not a system message.  The gateway appends
    # this marker after prior conversation turns, and strict OpenAI-compatible
    # providers (vLLM, Qwen) reject system messages that are not at the
    # beginning of the API message list (#48338).
    entry = {"role": "user", "content": marker, "display_kind": "model_switch"}

    def _replace_markers() -> None:
        history = session.setdefault("history", [])
        # Drop any earlier markers in place before appending the new one.
        history[:] = [h for h in history if not _is_model_switch_marker(h)]
        history.append(entry)
        session["history_version"] = int(session.get("history_version", 0)) + 1

    lock = session.get("history_lock")
    if lock is not None:
        with lock:
            _replace_markers()
    else:
        _replace_markers()

    try:
        agent = session.get("agent")
        db = getattr(agent, "_session_db", None) if agent is not None else None
        if db is not None:
            db.append_message(
                session_id=session_key,
                role="user",
                content=marker,
                display_kind="model_switch",
            )
            return

        _ensure_session_db_row(session)
        with _session_db(session) as scoped_db:
            if scoped_db is not None:
                scoped_db.append_message(
                    session_id=session_key,
                    role="user",
                    content=marker,
                    display_kind="model_switch",
                )
    except Exception:
        logger.debug("failed to persist model switch marker", exc_info=True)


def _write_config_key(key_path: str, value):
    # Write-back round-trip: raw read is mandatory — saving the managed-
    # overlaid / env-expanded view would persist those values into the file.
    cfg = _load_cfg_raw()
    current = cfg
    keys = key_path.split(".")
    for key in keys[:-1]:
        if key not in current or not isinstance(current.get(key), dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value
    _save_cfg(cfg)


_STATUSBAR_MODES = frozenset({"off", "top", "bottom"})
_APPROVAL_MODES = frozenset({"manual", "smart", "off"})


def _load_approval_mode() -> str:
    """Resolve the effective ``approvals.mode`` for the TUI surface.

    Delegates to the canonical resolver in ``tools.approval``
    (``_get_approval_mode``) so mode resolution cannot drift per surface —
    the same normalization, defaults, and config precedence the approval
    gate itself uses (see ``tools/approval.py``).

    Previously this re-read the config raw via ``_load_cfg`` +
    ``_deep_merge(DEFAULT_CONFIG, ...)`` and normalized locally, which
    could disagree with the gate's own view of the mode (e.g. the
    canonical ``hermes_cli.config.load_config`` path applies managed-scope
    overlays and ``${VAR}`` env expansion that the TUI's raw YAML read did
    not fully mirror).
    """
    from tools.approval import _get_approval_mode

    mode = _get_approval_mode()
    return mode if mode in _APPROVAL_MODES else "manual"


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


def _load_reasoning_config(model: str = "") -> dict | None:
    """Load reasoning effort from config.yaml, respecting per-model overrides.

    Thin wrapper over the shared chokepoint
    :func:`hermes_constants.resolve_reasoning_config` (per-model override >
    global ``agent.reasoning_effort``; YAML boolean False = disabled).
    Closes #21256.
    """
    from hermes_constants import resolve_reasoning_config

    return resolve_reasoning_config(_load_cfg(), model)


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


def _gui_surface_toolsets(platform: str) -> set[str]:
    """Toolsets that exist because of the CLIENT on the other end, not the host.

    Both entries are deliberately off ``_HERMES_CORE_TOOLS`` — every other
    platform would carry their schema for nothing — so this resolver is the one
    gate that exposes them.

    ``platform`` is the SESSION's source (``session.create``'s ``source``
    field), never a process env var. The desktop app is a client: it can be
    driving a local, SSH, URL, or cloud backend, and only the local/SSH spawn
    paths run with ``HERMES_DESKTOP=1``. Keying GUI capability off that env var
    silently stripped every pane/browser tool from URL and cloud gateways while
    the same backend told the model it was "chatting inside the Hermes desktop
    app". See the surface-capability rule in AGENTS.md.
    """
    surfaces = {"project"}
    if platform == "desktop":
        surfaces.add("desktop_ui")
    return surfaces


def _load_enabled_toolsets(platform: str | None = None) -> list[str] | None:
    session_platform = platform or _resolve_session_platform()
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

            selection = coding_selection(platform=session_platform)
            if selection is not None:
                # Fold in the client-surface toolsets here too: the focus-mode
                # coding posture returns before the fallback path that normally
                # adds them — without this the desktop loses its pane/project
                # tools exactly when sitting in a repo (see below).
                return sorted({*selection, *_gui_surface_toolsets(session_platform)})
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
        # The client-surface toolsets are off _HERMES_CORE_TOOLS (every other
        # platform would carry their schema for nothing), so the platform
        # recovery above — which keys off hermes-cli's tool universe — can't
        # surface them. This resolver runs ONLY in the desktop/TUI gateway, so
        # folding them in here is the gate that exposes them on exactly the
        # surface that can answer them.
        return sorted(enabled | _gui_surface_toolsets(session_platform))
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


def _tool_lifecycle_required_for_ui(name: str) -> bool:
    """Return True for tool events that are interactive UI, not optional chrome."""
    # Desktop renders the clarify choices/question from the tool-call part, then
    # wires request_id from clarify.request. If tool progress is off, suppressing
    # clarify's lifecycle events leaves only the sidebar attention dot visible.
    return name == "clarify"


def _restart_slash_worker(sid: str, session: dict):
    worker = session.get("slash_worker")
    # A session that never spawned a worker has nothing stale to replace —
    # the next slash.exec builds one with the current session key/model.
    # Spawning here would fork the per-worker stdio MCP fleet for sessions
    # that never use worker-routed commands.
    if worker is None:
        return
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


def _snapshot_agent_model_runtime(agent) -> dict:
    """Capture the current agent model runtime for a one-turn restore."""
    return {
        "model": getattr(agent, "model", ""),
        "provider": getattr(agent, "provider", ""),
        "api_key": getattr(agent, "api_key", ""),
        "base_url": getattr(agent, "base_url", ""),
        "api_mode": getattr(agent, "api_mode", ""),
        "primary_runtime": copy.deepcopy(getattr(agent, "_primary_runtime", None)),
    }


def _restore_agent_model_runtime(agent, snapshot: dict | None) -> None:
    """Restore an agent model runtime captured before a one-turn override."""
    if not snapshot or agent is None:
        return
    primary = snapshot.get("primary_runtime")
    if primary and hasattr(agent, "_restore_primary_runtime"):
        try:
            agent._primary_runtime = copy.deepcopy(primary)
            agent._fallback_activated = True
            agent._rate_limited_until = 0
            if agent._restore_primary_runtime():
                return
        except Exception:
            logger.debug("TUI one-turn model restore via primary runtime failed", exc_info=True)
    if hasattr(agent, "switch_model"):
        agent.switch_model(
            new_model=snapshot.get("model", ""),
            new_provider=snapshot.get("provider", ""),
            api_key=snapshot.get("api_key", ""),
            base_url=snapshot.get("base_url", ""),
            api_mode=snapshot.get("api_mode", ""),
        )


def _apply_model_switch(
    sid: str,
    session: dict,
    raw_input: str,
    *,
    confirm_expensive_model: bool = False,
    pin_session_override: bool = True,
    parsed_flags: Any | None = None,
    persist_override: bool | None = None,
) -> dict:
    from hermes_cli.model_switch import (
        parse_model_switch_args,
        resolve_persist_behavior,
        switch_model,
        MODEL_SWITCH_ERR_ONCE_WITH_GLOBAL,
        MODEL_SWITCH_ERROR_TEXT,
    )
    from hermes_cli.runtime_provider import resolve_runtime_provider

    if parsed_flags is None:
        parsed_flags = parse_model_switch_args(raw_input)
    if hasattr(parsed_flags, "model_input"):
        model_input = parsed_flags.model_input
        explicit_provider = parsed_flags.explicit_provider
        is_global_flag = parsed_flags.is_global
        is_session = parsed_flags.is_session
        one_turn = parsed_flags.is_once
    else:
        model_input, explicit_provider, is_global_flag, _force_refresh, is_session = parsed_flags
        one_turn = False
    # Conflict validation delegates to the shared single-owner parser; the
    # TUI surfaces it as a raised ValueError (its historical behavior)
    # using the canonical error copy.
    if is_global_flag and one_turn:
        raise ValueError(MODEL_SWITCH_ERROR_TEXT[MODEL_SWITCH_ERR_ONCE_WITH_GLOBAL])
    persist_global = (
        persist_override
        if persist_override is not None
        else resolve_persist_behavior(
            is_global_flag,
            is_session,
            is_once=one_turn,
            explicit_provider=explicit_provider,
        )
    )
    if not model_input:
        raise ValueError("model value required")

    agent = session.get("agent")
    if one_turn and not agent:
        raise ValueError("/model --once requires a live session")
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
    cfg = None
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

    restore_snapshot = _snapshot_agent_model_runtime(agent) if (one_turn and agent) else None

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
        if one_turn:
            session["one_turn_model_restore"] = restore_snapshot
        else:
            session.pop("one_turn_model_restore", None)

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
    if pin_session_override and isinstance(session, dict) and not one_turn:
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
        "scope": "once" if one_turn else ("global" if persist_global else "session"),
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


def _apply_pending_model_switch(sid: str, session: dict) -> None:
    """Apply a model switch queued while a turn was running.

    ``config.set model`` on a busy session doesn't mutate the live agent (the
    worker thread is reading model/client mid-request); it stashes the pick in
    ``session["pending_model_switch"]``.  This runs on the TURN thread at turn
    start — before the first model call, nothing in flight — so the in-place
    swap (client rebuild, the slow part) is safe here.  A failed switch keeps
    the current model and never blocks the turn, matching
    ``_sync_agent_model_with_config``.
    """
    pending = session.pop("pending_model_switch", None)
    if not pending or session.get("agent") is None:
        return
    try:
        result = _apply_model_switch(
            sid,
            session,
            pending["raw"],
            confirm_expensive_model=bool(pending.get("confirm_expensive_model")),
        )
        # A queued pick is a deliberate user action; honour the expensive-model
        # confirm by NOT applying it silently — surface the warning and drop the
        # switch rather than spend on a pricey model the user never confirmed.
        if result.get("confirm_required"):
            _emit(
                "error",
                sid,
                {"message": result.get("confirm_message") or result.get("warning") or ""},
            )
    except Exception as e:
        _emit(
            "error",
            sid,
            {"message": f"Could not switch model: {e}"},
        )


class CompressionLockHeld(Exception):
    """Raised by _compress_session_history when compression skipped due
    to a concurrent lock on the session's compression_locks row."""
    def __init__(self, holder: str | None = None):
        self.holder = holder
        super().__init__(f"Compression lock held: {holder or 'unknown'}")


def _compress_session_history(
    session: dict,
    focus_topic: str | None = None,
    approx_tokens: int | None = None,
    before_messages: list | None = None,
    history_version: int | None = None,
) -> tuple[int, dict]:
    """Compress a session's history — the single choke point shared by all
    three manual-compress routes (session.compress RPC, command.dispatch
    /compress|/compact, and the slash-exec mirror).

    ``focus_topic`` is the RAW argument string after ``/compress``. It is
    parsed here with :func:`parse_partial_compress_args` so boundary-aware
    forms (``here [N]``, ``up to here``, ``--keep N``) trigger a partial
    compress — head summarized, most recent ``keep_last`` exchanges kept
    verbatim — on EVERY route, mirroring cli.py's ``_manual_compress`` and
    gateway/slash_commands.py (PR #35252). Parsing at the choke point (not
    per-route) is what fixes #35533: previously "/compress here 3" reached
    this helper unparsed and ran a FULL compress focused on the literal
    text "here 3".
    """
    from agent.conversation_compression import (
        finalize_context_engine_compression_notification,
    )
    from agent.model_metadata import estimate_request_tokens_rough
    from hermes_cli.partial_compress import (
        parse_partial_compress_args,
        rejoin_compressed_head_and_tail,
        split_history_for_partial_compress,
    )

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
    partial, keep_last, focus_topic = parse_partial_compress_args(focus_topic or "")
    # Boundary-aware split: only the head is summarized; the most recent
    # `keep_last` exchanges ride along verbatim. A degenerate split (empty
    # tail — everything would be kept, or no head left to compress) falls
    # back to full compression so the user still gets an action.
    tail: list = []
    head = history
    if partial:
        head, tail = split_history_for_partial_compress(history, keep_last)
        if not tail:
            partial = False
            head = history
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
    # force=True: every caller of this helper is a manual /compress path
    # (session.compress RPC, slash compress/compact, slash-worker mirror) —
    # auto-compaction runs inside the agent loop, not here. Manual
    # compaction bypasses the summary-failure cooldown, matching the CLI
    # and gateway handlers.
    try:
        compressed, _ = agent._compress_context(
            head,
            None,
            approx_tokens=approx_tokens,
            # Partial compress has no focus topic (the modes are exclusive;
            # parse_partial_compress_args returns focus_topic=None for the
            # boundary-aware forms).
            focus_topic=focus_topic or None,
            force=True,
            defer_context_engine_notification=True,
        )
    except Exception:
        finalize_context_engine_compression_notification(
            agent,
            committed=False,
        )
        raise
    # If _compress_context returned unchanged because a concurrent
    # compression lock is held, raise so callers can surface a clear
    # message instead of the misleading "No changes from compression" text.
    # Type-pinned (is True / str): real values are None/True/holder-string;
    # bare truthiness is fooled by MagicMock auto-attrs on test doubles.
    _lock_skipped = getattr(agent, "_compression_skipped_due_to_lock", None)
    if _lock_skipped is True or isinstance(_lock_skipped, str):
        agent._compression_skipped_due_to_lock = None
        # No boundary was committed on a lock-skip; discard any pending
        # deferred context-engine notification (exactly-once, no-op safe).
        finalize_context_engine_compression_notification(
            agent,
            committed=False,
        )
        raise CompressionLockHeld(
            _lock_skipped if isinstance(_lock_skipped, str) else None
        )

    if partial and tail:
        compressed = rejoin_compressed_head_and_tail(compressed, tail)
    with session["history_lock"]:
        if int(session.get("history_version", 0)) != history_version:
            # External mutation during compaction — drop the compressed
            # result so we don't clobber concurrent edits.
            finalize_context_engine_compression_notification(
                agent,
                committed=False,
            )
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
    """Light credential check at session creation — returns warning or ''.

    ``no-key-required`` is a valid sentinel for keyless custom providers; only
    warn when the key is genuinely missing.
    """
    try:
        key = getattr(agent, "api_key", "") or ""
        provider = getattr(agent, "provider", "") or ""
        if not key:
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
# v3: adds approvals.mode config RPCs and session.info reconciliation.
# v4: session.create fast=false is an explicit per-session normal-tier override.
# v5: uvicorn ws_max_size raised for one-shot base64 file.attach frames (>16 MiB).
DESKTOP_BACKEND_CONTRACT = 5


def _session_usage_snapshot(session: dict | None) -> dict:
    agent = (session or {}).get("agent")
    mirror_usage = _metadata_mirror(session).get("usage")
    if (session or {}).get("_compute_host_active") and isinstance(mirror_usage, dict):
        return dict(mirror_usage)
    if agent is not None:
        return _get_usage(agent)
    return dict(mirror_usage) if isinstance(mirror_usage, dict) else {}


def _project_info_for_cwd(cwd: str) -> dict | None:
    """Return the first-class Project owning ``cwd`` for UI status surfaces.

    Backed by the per-profile projects.db (the same store the desktop's project
    tree caches), so the TUI status label, the desktop status bar, and ``/status``
    all name the session's workspace identically. Only explicit, named projects
    resolve here — an auto-discovered repo root has no projects.db row, so it
    falls back to the cwd leaf on every surface.
    """
    if not str(cwd or "").strip():
        return None
    try:
        from hermes_cli import projects_db as pdb

        with pdb.connect_closing() as conn:
            project = pdb.project_for_path(conn, cwd)
        if project is None:
            return None
        return {
            "id": project.id,
            "slug": project.slug,
            "name": project.name,
            "primary_path": project.primary_path,
        }
    except Exception:
        logger.debug("failed to resolve project for cwd", exc_info=True)
        return None


def _session_info(agent, session: dict | None = None) -> dict:
    if session is None:
        for candidate in _sessions.values():
            if candidate.get("agent") is agent:
                session = candidate
                break
    mirror = _metadata_mirror(session)
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
    service_tier = getattr(agent, "service_tier", None) or mirror.get("service_tier") or ""
    # Effective approval-bypass state — the same three sources that
    # check_all_command_guards() ORs together: persistent config
    # (approvals.mode=off), the process-scoped --yolo env, and the
    # per-session flag. Reporting only the per-session flag here would lie to
    # the desktop status bar (it would show YOLO "off" while approvals.mode=off
    # silently auto-approves every dangerous command).
    yolo = False
    approval_mode = "manual"
    try:
        from tools.approval import _YOLO_MODE_FROZEN, is_session_yolo_enabled

        session_yolo = (
            bool(is_session_yolo_enabled(session_key)) if session_key else False
        )
        approval_mode = _load_approval_mode()
        yolo = bool(_YOLO_MODE_FROZEN) or session_yolo or approval_mode == "off"
    except Exception:
        yolo = False
    # A model switch queued mid-turn (pending_model_switch) applies at the next
    # turn start, so agent.model still reads the OLD model until then. Report the
    # pending pick instead — it's the model the next turn will run, and it stops
    # the end-of-turn settle from blipping the UI back to the old model before
    # the switch lands. Cleared once _apply_pending_model_switch consumes it.
    pending_switch = (session or {}).get("pending_model_switch") or {}
    pending_model = str(pending_switch.get("display_model") or "").strip()
    pending_provider = str(pending_switch.get("display_provider") or "").strip()
    info: dict = {
        "model": pending_model or mirror.get("model", getattr(agent, "model", "")),
        "provider": pending_provider
        or mirror.get("provider", getattr(agent, "provider", "")),
        "reasoning_effort": reasoning_effort,
        "service_tier": service_tier,
        "fast": service_tier == "priority",
        "yolo": yolo,
        "approval_mode": approval_mode,
        "tools": dict(mirror.get("tools") or {}) if isinstance(mirror.get("tools"), dict) else {},
        "skills": dict(mirror.get("skills") or {}) if isinstance(mirror.get("skills"), dict) else {},
        "cwd": cwd,
        "branch": _git_branch_for_cwd(cwd),
        "project": _project_info_for_cwd(cwd),
        "personality": str(personality or ""),
        "running": bool((session or {}).get("running")),
        "title": _session_live_title(session or {}, session_key) if session_key else "",
        "stored_session_id": session_key or "",
        "desktop_contract": DESKTOP_BACKEND_CONTRACT,
        "version": "",
        "release_date": "",
        "update_behind": None,
        "update_command": "",
        "usage": _session_usage_snapshot(session),
        "profile_name": _response_profile_name(
            Path(session["profile_home"]).name
            if isinstance(session, dict) and session.get("profile_home")
            else None
        )
        if isinstance(session, dict) and session.get("profile_home")
        else _current_profile_name(),
    }
    try:
        from hermes_cli import __version__, __release_date__

        info["version"] = __version__
        info["release_date"] = __release_date__
    except Exception:
        pass
    if agent is not None and not (session or {}).get("_compute_host_active"):
        try:
            from model_tools import get_toolset_for_tool

            info["tools"] = {}
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
        info["system_prompt"] = (
            mirror.get("system_prompt")
            if "system_prompt" in mirror
            else getattr(agent, "_cached_system_prompt", "") or ""
        )
    except Exception:
        pass
    try:
        from hermes_cli.banner import get_update_result
        from hermes_cli.config import recommended_update_command

        info["update_behind"] = get_update_result(timeout=0.5)
        info["update_command"] = recommended_update_command()
    except Exception:
        pass
    if agent is not None and not (session or {}).get("_compute_host_active"):
        warn = _probe_credentials(agent)
        if warn:
            info["credential_warning"] = warn
    return info


def _tool_ctx(name: str, args: dict) -> str:
    """Argument preview for a tool row — never a phrased label.

    Clients own their own phrasing: the TUI wraps this as ``Terminal("...")``
    and the desktop prepends its own localized verb ("Running"/"Ran"). Sending
    ``build_tool_label`` here instead of the raw preview stutters the verb on
    both surfaces ("Running Running sleep 70 + 2 commands") and leaks a display
    label into the desktop's ``args.context``, where it stands in for the real
    command. The friendly labels belong on the CLI spinner, which builds them
    from ``build_tool_label`` at its own call sites.
    """
    try:
        from agent.display import build_tool_preview

        return build_tool_preview(name, args, max_len=80) or ""
    except Exception:
        return ""


def _emit_session_info_for_session(sid: str, session: dict) -> None:
    agent = session.get("agent")
    if agent is None and not _metadata_mirror(session):
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
    if _tool_progress_enabled(sid) or _tool_lifecycle_required_for_ui(name):
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
    if _tool_progress_enabled(sid) or payload.get("inline_diff") or _tool_lifecycle_required_for_ui(name):
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
    if event_type == "moa.progress":
        # Per-reference completion — drives the status-bar progress indicator
        # (`MOA: 2/3 refs done`) requested in issue #59546. Only emitted when
        # both counters are present so the client can render deterministically.
        refs_done = _kwargs.get("moa_refs_done")
        refs_total = _kwargs.get("moa_refs_total")
        if refs_done is None or refs_total is None:
            return
        _emit(
            "moa.progress",
            sid,
            {
                "label": str(name or ""),
                "refs_done": int(refs_done),
                "refs_total": int(refs_total),
            },
        )
        return
    if event_type == "moa.phase":
        # Phase transition — currently only ``phase="aggregator"`` fires once
        # the fan-out completes and the aggregator is about to act. Tells the
        # client which phase of the MoA pipeline is currently running so it
        # can swap status-bar copy accordingly.
        phase = _kwargs.get("moa_phase")
        if not phase:
            return
        phase_payload: dict[str, object] = {"phase": str(phase)}
        refs_done = _kwargs.get("moa_refs_done")
        refs_total = _kwargs.get("moa_refs_total")
        if refs_done is not None:
            phase_payload["refs_done"] = int(refs_done)
        if refs_total is not None:
            phase_payload["refs_total"] = int(refs_total)
        if name:
            phase_payload["aggregator"] = str(name)
        _emit("moa.phase", sid, phase_payload)
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
    callbacks = {
        "tool_start_callback": lambda tc_id, name, args: _on_tool_start(
            sid, tc_id, name, args
        ),
        "tool_complete_callback": lambda tc_id, name, args, result: _on_tool_complete(
            sid, tc_id, name, args, result
        ),
        "tool_progress_callback": lambda event_type, name=None, preview=None, args=None, **kwargs: _on_tool_progress(
            sid, event_type, name, preview, args, **kwargs
        ),
        "tool_gen_callback": lambda name: _tool_progress_enabled(sid)
        and _emit("tool.generating", sid, {"name": name}),
        "thinking_callback": lambda text: _emit("thinking.delta", sid, {"text": text}),
        # Affection reaction (ily / <3 / good bot) → hearts. Core-detected, so
        # the TUI heart and desktop floating hearts share one signal.
        "reaction_callback": lambda kind: _emit("reaction", sid, {"kind": kind}),
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
        "clarify_callback": lambda q, c, multi_select=False: _block(
            "clarify.request",
            sid,
            # multi_select is a pass-through hint: renderers with checkbox
            # support can honor it; older renderers ignore the extra field
            # and stay single-select (a single answer still parses as a
            # one-element list on the tool side). Only emitted when True so
            # single-select payloads keep the exact pre-multi-select shape.
            (
                {"question": q, "choices": c, "multi_select": True}
                if multi_select
                else {"question": q, "choices": c}
            ),
            timeout=_clarify_timeout_seconds(),
        ),
        # read_terminal tool (desktop GUI): same blocking bridge as clarify — the
        # renderer answers terminal.read.respond with the serialized buffer.
        "read_terminal_callback": lambda start=None, count=None: _block(
            "terminal.read.request",
            sid,
            {k: v for k, v in (("start", start), ("count", count)) if v is not None},
            timeout=30,
        ),
        # read_preview tool (desktop GUI): the renderer serializes the active
        # preview tab (a Browser webview's readable text, a file's identity)
        # and answers preview.read.respond. Longer timeout than the terminal
        # read — a URL tab extracts text from a live page.
        "read_preview_callback": lambda start=None, count=None: _block(
            "preview.read.request",
            sid,
            {k: v for k, v in (("start", start), ("count", count)) if v is not None},
            timeout=45,
        ),
    }

    # Interim assistant commentary (text alongside tool calls, or the attempted
    # final answer before a verify-on-stop nudge). Gated on
    # display.interim_assistant_messages (default true). Also set per-turn in
    # _run_prompt_submit as defense-in-depth — the per-turn set overwrites
    # this, and the finally block clears it so a stale closure can't fire.
    if _load_interim_assistant_messages():
        callbacks["interim_assistant_callback"] = (
            lambda text, *, already_streamed=False: _emit(
                "message.interim",
                sid,
                {"text": str(text), "already_streamed": bool(already_streamed)},
            )
        )

    return callbacks


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
    # An explicit project switch supersedes any earlier settle-adopted cwd.
    session["cwd_from_settle"] = False
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
            else {
                "cwd": resolved,
                "branch": _git_branch_for_cwd(resolved),
                "project": _project_info_for_cwd(resolved),
                "lazy": True,
            }
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
            parts.append(f'Tone: {value["tone"]}')
        if value.get("style"):
            parts.append(f'Style: {value["style"]}')
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

    Delegates to the shared ``get_fallback_chain`` helper so the TUI path
    stays in parity with ``HermesCLI.__init__`` and ``gateway/run.py``:
    ``fallback_providers`` is the primary source of truth and keeps its
    order, with legacy ``fallback_model`` entries merged in afterwards
    (deduped on provider/model/base_url).
    """
    from hermes_cli.fallback_config import get_fallback_chain

    return get_fallback_chain(_load_cfg())


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
        # Detached background tasks declare platform="tui" below: they have no
        # UI session id, so a renderer-routed event has nowhere to land. Resolve
        # their toolsets against that same platform rather than the gateway
        # process's, so they never carry GUI schema they cannot use.
        or _load_enabled_toolsets("tui"),
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
        "openrouter_min_coding_score": getattr(agent, "openrouter_min_coding_score", None),
        "session_id": task_id,
        "reasoning_config": getattr(agent, "reasoning_config", None)
        or _load_reasoning_config(str(getattr(agent, "model", "") or "")),
        "service_tier": getattr(agent, "service_tier", None) or _load_service_tier(),
        "request_overrides": dict(getattr(agent, "request_overrides", {}) or {}),
        "platform": "tui",
        "session_db": _get_db(),
        "fallback_model": _agent_fallback_model(agent),
    }


def _ephemeral_preview_agent_kwargs(agent, task_id: str) -> dict:
    kwargs = _background_agent_kwargs(agent, task_id)
    kwargs.update(
        {
            "enabled_toolsets": ["terminal", "file"],
            "session_db": None,
            "skip_memory": True,
        }
    )
    return kwargs


def _preview_restart_history(session: dict, max_messages: int = 24, max_tool_chars: int = 1200) -> list[dict]:
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
            _emit("preview.restart.progress", parent, {"task_id": task_id, "level": level, "text": text})

    def tool_start(tool_call_id: str, name: str, args: dict) -> None:
        started_at[tool_call_id] = time.time()
        ctx = _tool_ctx(name, args)
        progress(f"Running {name}{f': {ctx}' if ctx else ''}")

    def tool_complete(tool_call_id: str, name: str, _args: dict, result: str) -> None:
        duration_s = time.time() - started_at.get(tool_call_id, time.time())
        summary = _tool_summary(name, result, duration_s) or f"Finished {name}{f' in {_fmt_tool_duration(duration_s)}' if duration_s else ''}"
        output = _preview_tool_result_preview(name, result)
        progress(summary + (f"\n{output}" if output else ""))

    def tool_progress(event_type: str, name: str | None = None, preview: str | None = None, **_kwargs) -> None:
        if preview:
            progress(str(preview))
        elif name:
            progress(f"{event_type.replace('.', ' ')}: {name}")

    return {
        "tool_start_callback": tool_start,
        "tool_complete_callback": tool_complete,
        "tool_progress_callback": tool_progress,
        "tool_gen_callback": lambda name: progress(f"Preparing {name}"),
        "status_callback": lambda kind, text=None: progress(text if text is not None else kind),
    }


def _reset_session_agent(sid: str, session: dict) -> dict:
    tokens = _set_session_context(session["session_key"])
    try:
        # /new is a full conversation boundary: session-scoped runtime
        # overrides (/model, /reasoning, /fast) do NOT carry forward — the
        # fresh agent re-derives model/provider, reasoning, and service tier
        # from config.yaml (#48055, #23131). Session pins are cleared below so
        # a rebuild can't resurrect them. (Global process state is still never
        # touched — see the cross-session-contamination note in
        # _apply_model_switch.)
        session.pop("model_override", None)
        session.pop("create_reasoning_override", None)
        session.pop("create_service_tier_override", None)
        session.pop("one_turn_model_restore", None)
        new_agent = _make_agent(
            sid,
            session["session_key"],
            session_id=session["session_key"],
            platform_override=_session_source(session),
        )
    finally:
        _clear_session_context(tokens)
    session["agent"] = new_agent
    session["config_model_seen"] = _config_model_target()
    session["attached_images"] = []
    session["queued_prompt"] = None
    session.pop("queued_prompts", None)
    session["_queued_prompt_generation"] = int(session.get("_queued_prompt_generation", 0)) + 1
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


class _RuntimeFallbackResolution(NamedTuple):
    runtime: dict
    selected_model: str | None
    used_fallback: bool


def _resolve_runtime_with_fallback(
    resolve_kwargs: dict | None = None,
) -> _RuntimeFallbackResolution:
    """Resolve the primary runtime or one complete provider/model fallback.

    Setup-time auth fallback only accepts entries with both fields. Provider-
    only entries are skipped so the unavailable primary model can never leak
    into a different runtime. ``used_fallback`` remains explicit rather than
    overloading a nullable model as control flow.
    """
    from hermes_cli.auth import AuthError
    from hermes_cli.runtime_provider import resolve_runtime_provider

    kwargs = resolve_kwargs or {}
    try:
        return _RuntimeFallbackResolution(
            resolve_runtime_provider(**kwargs),
            None,
            False,
        )
    except AuthError as primary_exc:
        fb_chain = _load_fallback_model() or []
        for entry in fb_chain:
            if not isinstance(entry, dict):
                continue
            fb_provider = str(entry.get("provider") or "").strip()
            fb_model = str(entry.get("model") or "").strip()
            if not fb_provider or not fb_model:
                continue
            try:
                from hermes_cli.fallback_config import resolve_entry_api_key

                fb_kwargs: dict = {
                    "requested": fb_provider,
                    "target_model": fb_model,
                }
                if entry.get("base_url"):
                    fb_kwargs["explicit_base_url"] = entry["base_url"]
                fb_api_key = resolve_entry_api_key(entry)
                if fb_api_key:
                    fb_kwargs["explicit_api_key"] = fb_api_key
                runtime = resolve_runtime_provider(**fb_kwargs)
                import logging

                logging.getLogger(__name__).warning(
                    "Primary auth failed (%s), falling back to %s model %s",
                    primary_exc,
                    fb_provider,
                    fb_model,
                )
                return _RuntimeFallbackResolution(runtime, fb_model, True)
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
    # AC-4 test seam: dead unless explicitly armed by the isolated certify
    # harness. Both inline and compute-host paths construct through _make_agent,
    # leaving the process boundary as the only experimental variable.
    from tui_gateway.synthetic_turn import maybe_build_synthetic_agent

    synthetic = maybe_build_synthetic_agent(session_id or key, model_override)
    if synthetic is not None:
        return synthetic

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

            recovered = canonical_custom_identity(
                base_url=override_base_url or None, model=model or None
            )
            if recovered:
                requested_provider = recovered
            if override_base_url:
                # Failing identity recovery, still hand the base_url to the
                # direct-alias branch so pool/env credentials resolve for it.
                resolve_kwargs["explicit_base_url"] = override_base_url
        resolve_kwargs["requested"] = requested_provider
        resolve_kwargs["target_model"] = model or None
        resolution = _resolve_runtime_with_fallback(resolve_kwargs)
        runtime = resolution.runtime
        if resolution.used_fallback:
            if not resolution.selected_model:
                raise RuntimeError("Auth fallback resolved without a model")
            model = resolution.selected_model
        else:
            # The switch already resolved concrete credentials/endpoint; honor
            # persisted overrides only while using that original runtime. They
            # must not leak into a different fallback provider/model pair.
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
        resolution = _resolve_runtime_with_fallback({
            "requested": requested_provider,
            "target_model": model or None,
        })
        runtime = resolution.runtime
        if resolution.used_fallback:
            if not resolution.selected_model:
                raise RuntimeError("Auth fallback resolved without a model")
            model = resolution.selected_model
    _pr = _load_provider_routing()
    return AIAgent(
        model=model,
        max_iterations=_cfg_max_turns(cfg, 500),
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
            else _load_reasoning_config(str(model or ""))
        ),
        service_tier=(
            service_tier_override
            if service_tier_override is not None
            else _load_service_tier()
        ),
        enabled_toolsets=_load_enabled_toolsets(_resolve_agent_platform(platform_override)),
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
    profile_home: str | None = None,
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
            # Profile-scoped HERMES_HOME for app-global remote mode; None =
            # launch profile. SessionBranch copies the parent's value so the
            # child stays on the same state.db.
            "profile_home": profile_home,
            # Per-session model override set by an in-session /model switch.
            # Honored on rebuild (/new, resume) so a switch in THIS session
            # never leaks into siblings via process-global env vars.
            "model_override": None,
            # Pin async event emissions to whichever transport created the
            # session (stdio for Ink, JSON-RPC WS for the dashboard sidebar).
            "transport": current_transport() or _stdio_transport,
        }
    _init_owns_db = False
    if session_db is not None:
        db = session_db
    elif profile_home:
        try:
            from hermes_state import SessionDB

            db = SessionDB(db_path=Path(profile_home) / "state.db")
            _init_owns_db = True
        except Exception:
            db = _get_db()
    else:
        db = _get_db()
    try:
        if db is not None:
            row = db.get_session(key) if hasattr(db, "get_session") else None
            if row and row.get("cwd"):
                with _sessions_lock:
                    if sid in _sessions:
                        _sessions[sid]["cwd"] = row["cwd"]
            else:
                try:
                    _cwd = _sessions[sid]["cwd"]
                    if hasattr(db, "update_session_cwd"):
                        db.update_session_cwd(key, _cwd)
                    # git branch/root probes run off the hot path (see _set_session_cwd).
                    _persist_session_git_meta(_sessions[sid], _cwd)
                except Exception:
                    logger.debug(
                        "failed to persist resumed session cwd", exc_info=True
                    )
    finally:
        if _init_owns_db and db is not None:
            try:
                db.close()
            except Exception:
                pass
    _register_session_cwd(_sessions[sid])
    # No eager slash-worker pre-warm — the session dict already carries
    # slash_worker=None and slash.exec builds one on demand. See the
    # deferred-build path in _start_agent_build for the full rationale
    # (per-worker MCP fleets accumulating across retained sessions).
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


def _active_image_routing_identity(agent: Any) -> tuple[str, str]:
    """Return the live provider/model, falling back before agent startup."""
    from agent.auxiliary_client import _read_main_model, _read_main_provider

    return (
        getattr(agent, "provider", "") or _read_main_provider(),
        getattr(agent, "model", "") or _read_main_model(),
    )


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


def _build_persist_message_with_image_refs(user_text: str, image_paths: list[str]) -> str:
    """Build the clean, UI-recognizable version of the user's message for
    persisting to session history. Uses ``@image:<path>`` directives — the
    format the desktop client (directive-text.tsx / HERMES_DIRECTIVE_RE)
    actually parses and renders as an image — unlike
    ``_enrich_with_attached_images``, which embeds a vision description and
    an ``image_url:`` hint meant only for the model and must never be
    persisted as-is (it silently breaks image rendering after a full
    restart, and reorders image/text on live session-switch reconciliation).

    The caption leads and the directives trail: session previews are the first
    60 characters of the first user message (``list_sessions_rich``), so a
    leading directive would label the session with a truncated file path in the
    sidebar, switcher, and command palette. Clients lift the refs out of the
    body by line, so their position does not affect how the turn renders.
    """
    from agent.context_references import format_reference_value

    text = user_text or ""
    refs = "\n".join(f"@image:{format_reference_value(p)}" for p in image_paths if Path(p).exists())
    if not refs:
        return text
    return f"{text}\n{refs}" if text else refs


def _build_persist_user_message(user_text: str, image_paths: list[str], run_message: Any) -> Any:
    """Shape the persisted user turn to match what was sent to the model.

    Native-vision turns send ``content`` as a parts list, and
    ``_flush_messages_to_session_db`` deliberately ignores a plain-string
    override for a list payload (a text override must not erase a turn's
    image/audio summary). So mirror the shape: replace only the text part with
    the ``@image:`` ref form and keep the image parts, so the model still has
    the pixels for the rest of the session. Any API-only text part (the
    barge-in note) is dropped along the way, which is the point of the override.
    """
    persist_text = _build_persist_message_with_image_refs(user_text, image_paths)
    if not isinstance(run_message, list):
        return persist_text
    image_parts = [p for p in run_message if not (isinstance(p, dict) and p.get("type") == "text")]
    return [{"type": "text", "text": persist_text}, *image_parts]


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


_TEXT_ONLY_BUSY_PART_KINDS = frozenset({"text", "input_text", "output_text"})


def _is_text_only_busy_payload(content: Any) -> bool:
    """True when a busy submit carries only plain text, not attachments/media."""
    if content is None:
        return False
    if isinstance(content, (str, int, float)):
        return True
    if isinstance(content, list):
        if not content:
            return False
        for part in content:
            if isinstance(part, str):
                continue
            if not isinstance(part, dict):
                return False
            kind = part.get("type")
            if kind in _TEXT_ONLY_BUSY_PART_KINDS:
                continue
            if kind is None and isinstance(part.get("text"), str):
                continue
            return False
        return True
    if isinstance(content, dict):
        kind = content.get("type")
        if kind in _TEXT_ONLY_BUSY_PART_KINDS:
            return True
        return kind is None and isinstance(content.get("text"), str)
    return False


def _is_display_hidden_marker(role: str | None, text: str) -> bool:
    """Gateway bookkeeping notices (model-switch, personality) are persisted as
    role=user ``[System: …]`` rows so strict providers accept them mid-history.
    They are model-facing runtime metadata, not user turns, and must never
    render as a user bubble in ANY client transcript (desktop, TUI, CLI, web).

    Filtering here — the single display projection every surface reads — hides
    them everywhere while the raw marker stays in ``session["history"]`` for the
    model. It also removes the stored marker from the payload the desktop
    reconciles against, so it can no longer shift user-message ordinals and
    duplicate the optimistic prompt (#67603)."""
    return role == "user" and text.lstrip().startswith("[System:")


def _skill_scaffold_projection(content_text: str) -> str:
    """Return the invocation a slash-skill-expanded turn came from, else "".

    A ``/skill`` invocation expands into a model-facing message that embeds the
    whole skill body. That payload belongs to the agent — every UI renders the
    invocation (``/work fix the leak``) instead, so no surface can leak the
    body into a chat bubble.
    """
    return describe_skill_invocation(content_text, separator=" ") or ""


def _expand_skill_invocation_for_replay(text: str, task_id: str) -> str:
    """Re-expand a projected `/skill` invocation before re-running that turn.

    The inverse of :func:`_skill_scaffold_projection`. Because a skill turn is
    displayed as its invocation, a rewind/regenerate hands us back
    ``/work fix the leak`` rather than the body the agent originally saw —
    re-running that verbatim would drop the skill. Re-expanding here keeps the
    body server-side (no client ever holds it) and makes the replayed turn
    identical to the original.

    Returns *text* unchanged when it isn't a resolvable skill invocation.
    """
    head, _, arg = (text or "").strip().partition(" ")
    if not head.startswith("/"):
        return text

    try:
        from agent.skill_commands import (
            build_skill_invocation_message,
            resolve_skill_command_key,
        )

        cmd_key = resolve_skill_command_key(head.lstrip("/"))
        if cmd_key is None:
            return text

        return build_skill_invocation_message(cmd_key, arg.strip(), task_id=task_id) or text
    except Exception:
        # A skill that no longer resolves (renamed, disabled, external dir
        # gone) must not break the rewind — replay the text as typed.
        logger.debug("skill re-expansion failed for replay", exc_info=True)
        return text


# Opening of the crash-recovery note synthesized by _auto_continue_note.
# Matched (not just built) so a row persisted before the display type was
# stamped at turn start still reads as a timeline event, and to recognize the
# messaging gateway's twin note.
_AUTO_CONTINUE_NOTE_PREFIX = "[System note: Your previous turn was interrupted mid-run"


def _legacy_display_kind(role: str, text: str) -> str | None:
    """Infer the display type of a synthetic row persisted without one.

    Turn-start typing (see ``persist_user_display_kind``) covers everything
    written from here on. Sessions already on disk carry untyped rows — and a
    turn killed mid-run never reached the post-turn stamp at all, which is
    exactly the auto-continue case — so the raw recovery note would paint as a
    user bubble forever. Sniffing the one fixed synthetic prefix is the
    migration for those rows; it is not how new rows get typed.
    """
    if role == "user" and text.lstrip().startswith(_AUTO_CONTINUE_NOTE_PREFIX):
        return "auto_continue"
    return None


def _history_to_messages(history: list[dict]) -> list[dict]:
    messages = []
    tool_call_args = {}

    for m in history:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role not in {"user", "assistant", "tool", "system"}:
            continue
        # An explicit display_kind="hidden" row is model-facing scaffolding
        # (compaction references, interrupted-turn checkpoints). The string
        # sniff below only catches the "[System:" convention; honor the
        # declared field too, or scaffolding reaches every surface that reads
        # this projection.
        if m.get("display_kind") == "hidden":
            continue
        content_text = _coerce_message_text(m.get("content"))
        if _is_display_hidden_marker(role, content_text):
            continue
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
            messages.append(
                {"role": "tool", "name": name, "context": _tool_ctx(name, args)}
            )
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
        # Durable row identity, stamped by _rows_to_conversation. The renderer's
        # own message ids are ephemeral (timestamp+index derived, and a
        # different shape for live vs rehydrated vs optimistic rows), so
        # anything that addresses a specific persisted message later — message
        # reactions — needs this instead.
        if m.get("_row_id") is not None:
            msg["row_id"] = m["_row_id"]
        if role == "user":
            invocation = _skill_scaffold_projection(content_text)
            if invocation:
                # Show the invocation, never the expanded skill body. The raw
                # payload stays server-side: a rewind/regenerate re-sends the
                # turn by ordinal, so no client needs it.
                msg["text"] = invocation
                msg["display_kind"] = "skill_invocation"
        if role == "assistant":
            for key in reasoning_keys:
                if key in m and m.get(key) is not None:
                    msg[key] = m.get(key)
        # Forward display-only timeline metadata so the TUI can render
        # model switches and delegation completions as events instead of
        # opaque user messages, and hide compaction handoffs entirely.
        display_kind = m.get("display_kind") or _legacy_display_kind(role, content_text)
        if display_kind:
            msg["display_kind"] = display_kind
        if m.get("display_metadata"):
            msg["display_metadata"] = m["display_metadata"]
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


def _record_inflight_correction(session: dict, text: Any) -> None:
    """Record an accepted mid-turn correction on the live turn.

    The correction is appended, never written over ``user``: a resuming client
    must be able to rebuild BOTH bubbles. Overwriting the slot erased the
    prompt that started the turn from the only snapshot resume can read, so a
    reconnect (or a dev hot-reload that wipes the renderer cache) repainted the
    thread with the user's original message missing.
    """
    correction = _inflight_text(text)
    if not correction:
        return
    turn = session.get("inflight_turn")
    if not isinstance(turn, dict):
        return
    turn = dict(turn)
    corrections = list(turn.get("corrections") or [])
    corrections.append(correction)
    turn["corrections"] = corrections
    turn["updated_at"] = time.time()
    session["inflight_turn"] = turn


def _clear_inflight_turn(session: dict) -> None:
    session["inflight_turn"] = None


def _fail_inflight_turn(session: dict, error: Any) -> None:
    """Mark the in-flight turn terminal-error but keep it replayable.

    Normal completion clears ``inflight_turn`` because the response is now in
    canonical history. Failures are different: the terminal frame can be lost
    on a WS disconnect, and the failed turn may never have been committed.
    Retaining a compact error snapshot lets ``session.resume`` replay the
    user's prompt, any partial assistant text, and the error itself instead of
    leaving the client stranded on a spinner or hydrating from stale DB state.
    The snapshot lives until the next turn starts (``_start_inflight_turn``
    overwrites it) or the session closes.

    Caller must hold ``session["history_lock"]``.
    """
    message = str(error) if not isinstance(error, BaseException) else (str(error) or type(error).__name__)
    now = time.time()
    turn = session.get("inflight_turn")
    if not isinstance(turn, dict):
        turn = {"assistant": "", "user": "", "started_at": now}
    turn["assistant"] = str(turn.get("assistant") or "")
    turn["user"] = str(turn.get("user") or "")
    turn["error"] = message or "turn failed"
    turn["status"] = "error"
    turn["recoverable"] = True
    turn["streaming"] = False
    turn["updated_at"] = now
    session["inflight_turn"] = turn


# ── Auto-continue: resume a turn killed by a process/machine death ────
#
# A turn that concludes — success, handled error, interrupt — clears its
# durable marker (see tui_gateway/turn_marker.py) in _run_prompt_submit's
# finally. Only a process death leaves the marker behind, so a marker found
# at session.resume time is positive proof the turn never finished AND the
# client never saw a terminal frame. If the interruption is fresh, re-submit
# the interrupted prompt automatically (the messaging gateway has done this
# for restart-interrupted sessions since #27856); if it's stale, clear the
# marker and let the recovered partial transcript speak for itself — the
# user can ask to continue manually.

_AUTO_CONTINUE_ENABLED_DEFAULT = True
_AUTO_CONTINUE_FRESHNESS_MINUTES_DEFAULT = 15
_AUTO_CONTINUE_MAX_ATTEMPTS_DEFAULT = 2


def _auto_continue_config() -> tuple[bool, float, int]:
    """(enabled, freshness window in seconds, max attempts) from config.yaml."""
    desktop = _load_cfg().get("desktop")
    cfg = desktop.get("auto_continue") if isinstance(desktop, dict) else None
    if not isinstance(cfg, dict):
        cfg = {}
    try:
        minutes = float(cfg.get("freshness_minutes", _AUTO_CONTINUE_FRESHNESS_MINUTES_DEFAULT))
    except (TypeError, ValueError):
        minutes = float(_AUTO_CONTINUE_FRESHNESS_MINUTES_DEFAULT)
    return (
        is_truthy_value(cfg.get("enabled"), default=_AUTO_CONTINUE_ENABLED_DEFAULT),
        max(0.0, minutes) * 60.0,
        _coerce_int_config_value(
            cfg.get("max_attempts"), _AUTO_CONTINUE_MAX_ATTEMPTS_DEFAULT, min_value=0
        ),
    )


def _session_home(session: dict) -> Path:
    """The HERMES_HOME the session's durable state lives in (profile-aware)."""
    profile_home = session.get("profile_home")
    return Path(profile_home) if profile_home else Path(_hermes_home)


def _retire_turn_marker(session: dict, *keys: str) -> None:
    """Drop the crash marker for a turn whose outcome is about to reach the client.

    Called immediately before the terminal frame rather than at the end of the
    turn thread: post-turn work (titles, memory sync, goal hooks) runs for a
    second or more after the client has its answer, and quitting inside that
    window would leave a marker that looks like a crash — re-running a finished
    turn on the next launch. Extra ``keys`` cover a session_key that
    compression rotated mid-turn.
    """
    home = _session_home(session)
    for key in dict.fromkeys((*keys, str(session.get("session_key") or ""))):
        if key:
            clear_turn_marker(home, key)


def _auto_continue_note(prompt: str) -> str:
    # Same opening as the messaging gateway's recovery notes so transcript
    # tooling recognizes both. The original prompt is embedded because a hard
    # crash persists nothing of the interrupted turn to the session DB — this
    # note is the only copy the model will see.
    return (
        f"{_AUTO_CONTINUE_NOTE_PREFIX} — the app or its backend process "
        "stopped before the turn could finish. Some of the work may already "
        "be complete; check the current state before redoing anything, then "
        "finish the task. The interrupted request was:]\n\n"
        f"{prompt}"
    )


def _maybe_schedule_auto_continue(sid: str, session: dict, session_key: str) -> dict | None:
    """Kick off a continuation turn for a crash-interrupted session.

    Called from session.resume's cold paths after the live record is
    registered. Returns a small descriptor for the resume payload when a
    continuation was scheduled, else None. The turn itself runs on a
    background thread after the (deferred) agent build finishes, through the
    same _run_prompt_submit machinery as every other synthesized turn — so
    the client that just resumed streams it live.
    """
    home = _session_home(session)
    marker = read_turn_marker(home, session_key)
    if marker is None:
        return None
    enabled, freshness_secs, max_attempts = _auto_continue_config()
    age = time.time() - marker["started_at"]
    if not enabled or age > freshness_secs or marker["attempts"] >= max_attempts:
        # Stale, disabled, or crash-looping: stop trying. The journal/partial
        # transcript still shows what happened; a manual message continues it.
        clear_turn_marker(home, session_key)
        return None
    if session.get("_auto_continue_scheduled"):
        return None
    session["_auto_continue_scheduled"] = True
    attempt = marker["attempts"] + 1
    text = _auto_continue_note(marker["prompt"])

    def kickoff() -> None:
        rid = f"__auto_continue__{int(time.time() * 1000)}"
        try:
            _start_agent_build(sid, session)
            err = _wait_agent(session, rid, timeout=120.0)
        except Exception:
            logger.warning("auto-continue agent build failed for %s", sid, exc_info=True)
            err = {"error": {"message": "agent build failed"}}
        if err:
            # Leave the marker: the next resume retries (bounded by attempts).
            session["_auto_continue_scheduled"] = False
            return
        with session["history_lock"]:
            if session.get("running") or session.get("_turn_cancel_requested") or session.get("_finalized"):
                # A real user prompt beat us to it — their turn wins, and its
                # own conclusion clears the marker.
                session["_auto_continue_scheduled"] = False
                return
            session["running"] = True
            session["last_active"] = time.time()
            # Hand this turn its own marker inputs (read back by
            # _run_prompt_submit): count the attempt so a crash during the
            # continuation trips the breaker, and re-record the ORIGINAL
            # prompt so a second crash doesn't nest note inside note. Set
            # here, not at schedule time, so a bail above leaves nothing
            # behind for a racing user turn to inherit.
            session["_auto_continue_attempt"] = attempt
            session["_auto_continue_prompt"] = marker["prompt"]
        try:
            _emit(
                "status.update",
                sid,
                {"kind": "process", "text": "Resuming interrupted turn…"},
            )
            _emit("message.start", sid)
            _run_prompt_submit(rid, sid, session, text, display_kind="auto_continue")
        except Exception as exc:
            print(
                f"[tui_gateway] auto-continue dispatch failed: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            with session["history_lock"]:
                session["running"] = False

    threading.Thread(target=kickoff, daemon=True).start()
    logger.info(
        "auto-continue scheduled for session %s (attempt %d, interrupted %.0fs ago)",
        session_key,
        attempt,
        age,
    )
    return {"attempt": attempt, "interrupted_at": marker["started_at"]}


def _enqueue_prompt(
    session: dict,
    text: Any,
    transport: Any,
    image_paths: list[str] | None = None,
) -> None:
    """Stash a message to run as the very next turn once the live one ends.

    Used when a prompt arrives mid-turn (see ``_handle_busy_submit``). Text-only
    arrivals share a slot and merge losslessly (mirroring the consecutive-user
    merge in ``repair_message_sequence``). Image-bearing submissions stay as
    separate envelopes, so their attachment ownership and chronology survive.
    ``transport`` is pinned so the drained turn streams back to the client that
    sent it even if the session transport is rebound meanwhile.
    """
    image_paths = list(image_paths or [])
    queued = {"text": text, "transport": transport}
    if image_paths:
        queued["image_paths"] = image_paths
    existing = session.get("queued_prompt")
    if (
        existing
        and isinstance(existing.get("text"), str)
        and isinstance(text, str)
        and not existing.get("image_paths")
        and not image_paths
        and not session.get("queued_prompts")
    ):
        prev = existing["text"]
        existing["text"] = f"{prev}\n\n{text}" if prev and text else (prev or text)
        return
    if existing:
        session.setdefault("queued_prompts", []).append(queued)
        return
    session["queued_prompt"] = queued


def _interrupt_busy_session(sid: str, session: dict, agent: Any) -> None:
    """Interrupt a busy turn without blocking the RPC reader or session lock.

    Some providers cannot apply ``interrupt()`` until a synchronous tool or
    network call returns. Running that call inline used to leave
    ``prompt.submit`` holding ``history_lock`` for the whole wait, which in turn
    blocked ``session.resume`` and delayed the queued prompt itself. Keep at
    most one interrupt worker per session so repeated steering cannot leak an
    unbounded number of blocked threads.
    """
    use_agent = agent is not None and hasattr(agent, "interrupt")
    use_compute_host = not use_agent and _session_uses_compute_host(session)
    if not use_agent and not use_compute_host:
        return

    with session["history_lock"]:
        if session.get("_busy_interrupt_pending"):
            return
        session["_busy_interrupt_pending"] = True

    def interrupt() -> None:
        try:
            if use_agent:
                agent.interrupt()
            else:
                _get_compute_host_supervisor().interrupt(sid)
        except Exception:
            pass
        finally:
            with session["history_lock"]:
                session["_busy_interrupt_pending"] = False

    threading.Thread(target=interrupt, daemon=True, name=f"busy-interrupt-{sid}").start()


def _handle_busy_submit(
    rid, sid: str, session: dict, text: Any, transport: Any, queued: bool = False
) -> dict | None:
    """Apply the ``display.busy_input_mode`` policy to a prompt that lands while
    a turn is in flight, instead of rejecting it with ``session busy``.

    The old rejection forced clients into a deadline-bounded busy-retry that
    silently dropped the send when turn teardown outlived the deadline. The
    default policy now redirects a capable core agent in place; older agents
    retain the proven interrupt-and-queue path drained from ``run``'s tail.

    Modes: ``interrupt`` (default) → redirect the live turn, falling back to
    hard interrupt + queue for older agents; ``queue`` → queue without
    interrupting; ``steer`` → inject after the current atomic action.

    ``queued=True`` (client's queue drain, ``prompt.submit`` param) overrides
    the mode entirely: the message was explicitly queued as "run after", so it
    must NEVER become a live-turn correction or interrupt. Without this, a
    drain that loses the settle race (client observed idle, server still
    unwinding the turn) redirected the live turn with next-turn text — queue
    semantics betrayed by a millisecond race the user can't see.
    """
    mode = "queue" if queued else _load_busy_input_mode()
    agent = session.get("agent")
    with session["history_lock"]:
        if not session.get("running"):
            # The turn ended between prompt.submit's first busy check and this
            # helper. Let the caller retry and claim the now-idle session.
            return None
    with session["history_lock"]:
        if not session.get("running"):
            return None
        image_paths = list(session.get("attached_images", []))
        if image_paths:
            # Claim at submission time. A later paste must not be consumed by
            # this prompt after the active turn finally yields.
            session["attached_images"] = []
    text_only = not image_paths and _is_text_only_busy_payload(text)
    plain_text = _coerce_message_text(text).strip() if text_only else ""
    if mode == "steer" and text_only and plain_text and agent is not None and hasattr(agent, "steer"):
        try:
            if agent.steer(plain_text):
                with session["history_lock"]:
                    session["last_active"] = time.time()
                return _ok(rid, {"status": "steered"})
        except Exception:
            pass  # fall through to queue
    # Text-only corrections redirect the live turn in place when the runtime
    # supports it; media/attachment payloads and older agents fall through to
    # the proven interrupt + queue path below.
    if (
        mode == "interrupt"
        and text_only
        and plain_text
        and agent is not None
        and getattr(agent, "_supports_active_turn_redirect", False) is True
        and hasattr(agent, "redirect")
    ):
        try:
            if agent.redirect(plain_text):
                with session["history_lock"]:
                    _record_inflight_correction(session, plain_text)
                    session["last_active"] = time.time()
                return _ok(rid, {"status": "redirected"})
        except Exception:
            pass  # preserve the proven interrupt + queue fallback below
    # Queue before asking the live turn to stop. In particular, never call a
    # provider or compute-host method while holding history_lock: an interrupt
    # can wait behind the very operation it is trying to cancel.
    with session["history_lock"]:
        if not session.get("running"):
            if image_paths:
                session["attached_images"] = image_paths + list(session.get("attached_images", []))
            return None
        _enqueue_prompt(session, text, transport, image_paths=image_paths)
        session["last_active"] = time.time()

    # Attachments need a separate model invocation. Queue them without
    # cancelling the active turn so the user gets both results in order.
    if mode != "queue" and not image_paths:
        _interrupt_busy_session(sid, session, agent)
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
        queue_generation = int(session.get("_queued_prompt_generation", 0))
        queued_prompts = session.get("queued_prompts") or []
        session["queued_prompt"] = queued_prompts.pop(0) if queued_prompts else None
        if not queued_prompts:
            session.pop("queued_prompts", None)
        session["running"] = True
        if queued.get("transport") is not None:
            session["transport"] = queued["transport"]
    use_compute_host = _session_uses_compute_host(session)
    with session["history_lock"]:
        if int(session.get("_queued_prompt_generation", 0)) != queue_generation:
            session["running"] = False
            return True
    dispatch_failed = False
    try:
        if use_compute_host:
            if queued.get("image_paths"):
                resp = _submit_prompt_to_compute_host(
                    rid,
                    sid,
                    session,
                    queued["text"],
                    image_paths=queued["image_paths"],
                    queued_prompt_generation=queue_generation,
                )
            else:
                resp = _submit_prompt_to_compute_host(
                    rid, sid, session, queued["text"], queued_prompt_generation=queue_generation
                )
            if resp.get("error"):
                message = str(((resp.get("error") or {}).get("message")) or "queued prompt failed")
                with session["history_lock"]:
                    session["running"] = False
                    _clear_inflight_turn(session)
                _emit("error", sid, {"message": message})
                dispatch_failed = True
        else:
            if queued.get("image_paths"):
                _run_prompt_submit(
                    rid,
                    sid,
                    session,
                    queued["text"],
                    image_paths=queued["image_paths"],
                    queued_prompt_generation=queue_generation,
                )
            else:
                _run_prompt_submit(
                    rid,
                    sid,
                    session,
                    queued["text"],
                    queued_prompt_generation=queue_generation,
                )
    except Exception as exc:
        print(
            f"[tui_gateway] queued prompt dispatch failed: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        with session["history_lock"]:
            session["running"] = False
        dispatch_failed = True
    if dispatch_failed:
        with session["history_lock"]:
            drain_next = bool(session.get("queued_prompt")) and not session.get(
                "_turn_cancel_requested"
            )
        if drain_next:
            _drain_queued_prompt(rid, sid, session)
    return True


def _inflight_snapshot(session: dict) -> dict | None:
    turn = session.get("inflight_turn")
    if not isinstance(turn, dict):
        return None
    user = str(turn.get("user") or "").strip()
    assistant = str(turn.get("assistant") or "")
    streaming = bool(turn.get("streaming"))
    error = str(turn.get("error") or "").strip()
    if not user and not assistant and not streaming and not error:
        return None
    snapshot = {
        "assistant": assistant,
        "streaming": streaming,
        "user": user,
    }
    corrections = [c for c in (turn.get("corrections") or []) if str(c).strip()]
    if corrections:
        # Mid-turn redirects. Carried alongside the original prompt (not over
        # it) so resume can rebuild every user bubble the turn produced.
        snapshot["corrections"] = [str(c) for c in corrections]
    if error:
        # Retained failed turn (see _fail_inflight_turn): carry the error
        # semantics so a resuming client can rebuild the failed-turn bubble
        # instead of rendering the partial text as a healthy reply.
        snapshot["error"] = error
        snapshot["status"] = str(turn.get("status") or "error")
        snapshot["recoverable"] = bool(turn.get("recoverable"))
    return snapshot


def _emit_terminal_turn_error(sid: str, session: dict, error: Any) -> None:
    """Close a failed turn with a terminal ``message.complete`` frame.

    Emits the same ``status: "error"`` frame shape the returned-error path in
    ``_run_prompt_submit`` already produces (so TUI/desktop handling is
    uniform), and retains the failed turn via ``_fail_inflight_turn`` so a
    client that missed this frame (disconnect window) can recover it from
    ``session.resume``'s ``inflight`` payload.
    """
    with session["history_lock"]:
        _fail_inflight_turn(session, error)
        turn = session.get("inflight_turn") or {}
        message = str(turn.get("error") or "turn failed")
        partial = str(turn.get("assistant") or "")
        cols = int(session.get("cols", 80))
    text = partial or f"Error: {message}"
    agent = session.get("agent")
    payload = {
        "text": text,
        "usage": _get_usage(agent) if agent is not None else {},
        "status": "error",
        "error": message,
        "recoverable": True,
    }
    if partial:
        payload["partial"] = True
    try:
        rendered = render_message(text, cols)
    except Exception:
        rendered = ""
    if rendered:
        payload["rendered"] = rendered
    _retire_turn_marker(session)
    _emit("message.complete", sid, payload)


def _restore_agent_history_after_turn_error(session: dict, agent) -> bool:
    """Keep a failed turn's working transcript in the gateway session.

    ``AIAgent`` persists its working messages independently of the gateway's
    history snapshot. If the turn raises after that persistence, the next
    prompt must see the working transcript instead of the pre-turn snapshot.
    """
    agent_messages = getattr(agent, "_session_messages", None)
    if not isinstance(agent_messages, list):
        return False
    with session["history_lock"]:
        session["history"] = list(agent_messages)
        session["history_version"] = int(session.get("history_version", 0)) + 1
    return True


def _queued_prompt_snapshot(session: dict) -> dict | None:
    """Return the accepted next-turn prompt without its transport handle.

    A busy ``prompt.submit`` lives only in ``session["queued_prompt"]`` until
    the current turn winds down. Desktop may reconnect or restart during that
    window, so the live-session projection must carry the user-visible text;
    otherwise the accepted prompt disappears until it finally drains.
    """
    queued = session.get("queued_prompt")
    if not isinstance(queued, dict):
        return None
    user = _inflight_text(queued.get("text"))
    return {"user": user} if user else None


# ── Methods: session ─────────────────────────────────────────────────


def _lazy_resume_info(
    cwd: str,
    *,
    model: str = "",
    provider: str = "",
    profile: str | None = None,
) -> dict:
    """session.info for a not-yet-built session (the shape session.create
    returns). tools/skills land later when the deferred build emits session.info."""
    info = {
        "cwd": cwd,
        "branch": _git_branch_for_cwd(cwd),
        "project": _project_info_for_cwd(cwd),
        "model": model or _resolve_model(),
        "tools": {},
        "skills": {},
        "lazy": True,
        "desktop_contract": DESKTOP_BACKEND_CONTRACT,
        "profile_name": _response_profile_name(profile),
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
    try:
        with _session_db(session) as db:
            if db is not None:
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
    queued = _queued_prompt_snapshot(session)
    preview = _message_preview(history)
    if queued:
        preview = queued.get("user") or preview
        preview = " ".join(str(preview).split())[:160]
    elif inflight:
        preview = inflight.get("assistant") or inflight.get("user") or preview
        preview = " ".join(str(preview).split())[:160]
    now = time.time()
    return {
        "current": sid == current_sid,
        "id": sid,
        "last_active": float(session.get("last_active") or session.get("created_at") or now),
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
    # The SESSION's own workspace, not the gateway's launch directory. Reporting
    # `_default_session_cwd()` here told a lazily-resumed session's client that
    # its workspace was wherever the gateway process happened to start, so the
    # desktop Files pane painted the wrong project even after the renderer
    # rebound correctly (#71254). `branch` is always emitted ("" outside a git
    # repo) so a client can clear a stale label instead of retaining it — the
    # same contract `_lazy_session_info` above already follows.
    cwd = _session_cwd(session)
    return {
        "cwd": cwd,
        "branch": _git_branch_for_cwd(cwd),
        "project": _project_info_for_cwd(cwd),
        "lazy": True,
        "model": _resolve_model(),
        "skills": {},
        "tools": {},
    }


def _reconcile_display_with_live(
    db_display: list[dict], in_memory: list[dict]
) -> list[dict]:
    """Merge the persisted DISPLAY lineage with the in-memory live history.

    Two projections of the same session that each hold something the other
    lacks:

    - ``db_display`` — the verbatim persisted lineage. It includes
      *model-invisible* rows (verification candidates, finish_reason
      ``verification_required`` / ``verify_hook_continue``) that the in-memory
      model history collapses out via ``repair_message_sequence`` (#65919), but
      it can lag the newest turn by a flush.
    - ``in_memory`` — ``display_history_prefix + session["history"]``. It is the
      freshest recency authority (a just-appended turn may not be flushed yet)
      but it is the collapsed *model* projection, so it is missing candidates.

    The merge keeps the DB display (candidate-inclusive) as the base and appends
    only the in-memory tail that the DB does not yet cover, anchored on the last
    DB row's ``(role, text)``. This satisfies BOTH invariants at once: the
    substantive verification answer survives a warm/live switch (matching the
    eager resume + REST payloads), and a not-yet-flushed live turn is not
    dropped.
    """
    if not db_display:
        return in_memory
    if not in_memory:
        return db_display

    def _key(msg: dict) -> tuple:
        return (msg.get("role"), _coerce_message_text(msg.get("content")))

    anchor = _key(db_display[-1])
    last_shared = -1
    for idx, msg in enumerate(in_memory):
        if isinstance(msg, dict) and _key(msg) == anchor:
            last_shared = idx
    if last_shared == -1:
        # The DB tail isn't present in memory (DB is ahead, or the histories
        # diverged) — trust the persisted display rather than risk duplicating.
        return db_display
    return list(db_display) + list(in_memory[last_shared + 1 :])


def _live_visible_history(session: dict, db, in_memory_fallback: list[dict]) -> list[dict]:
    """Return the user-visible DISPLAY projection for a live/warm session.

    Serving the raw in-memory *model* history for the user-visible payload
    dropped model-invisible rows (verification candidates persisted by #65919)
    whenever a warm/live session was reused, while the eager ``session.resume``
    path (which reads the verbatim display lineage) still showed them — the two
    payloads disagreed about the same session, which is the cross-session
    "substantive answer vanishes on switch" class of bug.

    This reconciles the persisted display lineage (candidate-inclusive, via
    ``get_messages_as_conversation(..., include_ancestors=True)`` — the same
    read the eager resume and REST paths use) with the fresh in-memory tail, so
    all surfaces agree while a not-yet-flushed turn is still shown. Falls back to
    the in-memory history when the DB/session_key is unavailable or the DB read
    fails.
    """
    key = session.get("session_key")
    if db is not None and key:
        try:
            display = db.get_messages_as_conversation(key, include_ancestors=True, include_row_ids=True)
            return _reconcile_display_with_live(display, in_memory_fallback)
        except Exception:
            logger.debug("live display projection read failed", exc_info=True)
    return in_memory_fallback


def _live_session_payload(
    sid: str,
    session: dict,
    *,
    cols: int | None = None,
    touch: bool = False,
    transport: Transport | None = None,
    omit_messages: bool = False,
) -> dict:
    with session["history_lock"]:
        if cols is not None:
            session["cols"] = cols
        if transport is not None:
            session["transport"] = transport
        if touch:
            session["last_active"] = time.time()
        in_memory_history = list(session.get("display_history_prefix") or []) + list(
            session.get("history") or []
        )
        inflight = _inflight_snapshot(session)
        queued = _queued_prompt_snapshot(session)
        running = bool(session.get("running"))
    # Prefer the persisted display lineage (candidate-inclusive) so this payload
    # matches the eager session.resume + REST transcript; the DB has its own
    # lock, so read it outside the session history lock.
    history = (
        in_memory_history
        if omit_messages
        else _live_visible_history(session, _get_db(), in_memory_history)
    )
    payload = {
        "info": _fallback_session_info(session),
        "message_count": len(history),
        "messages": [] if omit_messages else _history_to_messages(history),
        "messages_omitted": omit_messages,
        "running": running,
        "session_id": sid,
        "session_key": _session_lookup_key(session, fallback=sid),
        "started_at": float(session.get("created_at") or time.time()),
        "status": _session_live_status(sid, session),
    }
    if inflight:
        payload["inflight"] = inflight
    if queued:
        payload["queued"] = queued
    return payload


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


# ===========================================================================
# Phase 2b Remote Spending RPC methods
# ===========================================================================
#
# These return STRUCTURED success envelopes (result.ok / result.error) rather
# than JSON-RPC-level errors, so the TUI's rpc() promise always resolves and the
# Ink side can branch on the typed billing error code (insufficient_scope,
# rate_limited, no_payment_method, …) to render the right affordance instead of
# landing in a generic catch. The data-building lives in the shared core
# (agent/billing_view.py + hermes_cli/nous_billing.py) — same as /topup.


def _serialize_billing_error(exc) -> dict:
    """Map a BillingError into the result.error envelope the TUI branches on."""
    from hermes_cli.nous_billing import (
        BillingRemoteSpendingRevoked,
        BillingScopeRequired,
        BillingSessionRevoked,
        BillingTransient,
    )

    kind = "error"
    if isinstance(exc, BillingRemoteSpendingRevoked):
        kind = "remote_spending_revoked"
    elif isinstance(exc, BillingSessionRevoked):
        kind = "session_revoked"
    elif isinstance(exc, BillingScopeRequired):
        kind = "insufficient_scope"
    elif isinstance(exc, BillingTransient):
        kind = str(exc.error) if getattr(exc, "error", None) else "rate_limited"
    elif getattr(exc, "error", None):
        kind = str(exc.error)
    return {
        "ok": False,
        "error": kind,
        "message": str(exc),
        "portal_url": getattr(exc, "portal_url", None),
        "retry_after": getattr(exc, "retry_after", None),
        "payload": getattr(exc, "payload", {}) or {},
        # Remote-Spending contract extras (threaded so the TUI can render
        # actor-aware copy + route recovery without re-parsing the message).
        "actor": getattr(exc, "actor", None),
        "code": getattr(exc, "code", None),
        "recovery": getattr(exc, "recovery", None),
    }


def _serialize_billing_state(state) -> dict:
    """Serialize a BillingState for the wire (Decimals → strings, money-safe)."""
    from agent.billing_view import format_money

    def _s(value):
        return None if value is None else str(value)

    card = None
    if state.card is not None:
        card = {
            "brand": state.card.brand,
            "last4": state.card.last4,
            "masked": state.card.masked,
            # Post-card-resolver fields (None/False on older NAS payloads):
            # display = "Visa ····4242 — the card on your subscription";
            # resolved_via = the raw resolution rung, for rung-gated surfaces
            # (the /subscription confirm only shows the card when the rung
            # matches what a subscription charge would use).
            "display": state.card.display,
            "resolved_via": state.card.resolved_via,
        }
    payment_method = None
    if state.payment_method is not None:
        pm = state.payment_method
        # Each kind sends only its own fields. Emitting every key with nulls
        # would contradict the shared type — a client checking `'brand' in pm`
        # would read every Link method as a card.
        if pm.kind == "card":
            payment_method = {
                "kind": "card",
                "brand": pm.brand,
                "last4": pm.last4,
                "wallet": pm.wallet,
                "resolved_via": pm.resolved_via,
            }
        elif pm.kind == "link":
            payment_method = {
                "kind": "link",
                "email": pm.email,
                "resolved_via": pm.resolved_via,
            }
        else:
            payment_method = {
                "kind": "unknown",
                "raw_kind": pm.raw_kind,
                "resolved_via": pm.resolved_via,
            }
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
        card_out = None
        if ar.card is not None:
            if ar.card.kind == "distinct":
                card_out = {
                    "kind": "distinct",
                    "payment_method_id": ar.card.payment_method_id,
                    "brand": ar.card.brand,
                    "last4": ar.card.last4,
                }
            else:
                card_out = {"kind": ar.card.kind}
        auto_reload = {
            "enabled": ar.enabled,
            "threshold_usd": _s(ar.threshold_usd),
            "threshold_display": format_money(ar.threshold_usd),
            "reload_to_usd": _s(ar.reload_to_usd),
            "reload_to_display": format_money(ar.reload_to_usd),
            "card": card_out,
        }
    return {
        "ok": True,
        "logged_in": state.logged_in,
        "org_name": state.org_name,
        "org_slug": state.org_slug,
        "role": state.role,
        "is_admin": state.is_admin,
        "can_change_plan": state.can_change_plan,
        "can_charge": state.can_charge,
        "balance_usd": _s(state.balance_usd),
        "balance_display": format_money(state.balance_usd),
        "cli_billing_enabled": state.cli_billing_enabled,
        "charge_presets": [_s(p) for p in state.charge_presets],
        "charge_presets_display": [format_money(p) for p in state.charge_presets],
        "min_usd": _s(state.min_usd),
        "max_usd": _s(state.max_usd),
        "card": card,
        "payment_method": payment_method,
        "monthly_cap": monthly_cap,
        "auto_reload": auto_reload,
        "portal_url": state.portal_url,
        "error": state.error,
        # Shared dollar usage model (two-bar view) embedded so /topup renders the
        # same plan + top-up bars as /usage and /subscription from its single
        # fetch. Built from the separate account-info path; fail-open when logged
        # out or the portal is down.
        "usage": _usage_payload(state),
    }


def _usage_payload(state) -> dict:
    """Best-effort shared usage model for the /topup + /subscription overlay bars.

    Only fetched when logged in; fail-open to {available:false} so the overview
    still renders if the account-info path is down.
    """
    if not getattr(state, "logged_in", False):
        return {"available": False}
    try:
        from agent.billing_usage import build_usage_model

        return _serialize_usage_model(build_usage_model())
    except Exception:
        return {"available": False}


def _serialize_usage_bar(bar) -> Optional[dict]:
    """Serialize a UsageBar (dollar magnitudes → display strings + fractions)."""
    if bar is None:
        return None
    from agent.billing_usage import _fmt_usd

    return {
        "kind": bar.kind,
        "remaining_display": _fmt_usd(bar.remaining_usd),
        "total_display": _fmt_usd(bar.total_usd),
        "spent_display": _fmt_usd(bar.spent_usd),
        "pct_used": bar.pct_used,
        "fill_fraction": bar.fill_fraction,
    }


def _serialize_usage_model(model) -> dict:
    """Serialize a UsageModel for the wire — the shared two-bar dollar view.

    Dollars-only (no 'credits'); fail-open shape mirrors the other billing RPCs
    ({ok, available:false} when logged out / unreachable).
    """
    from agent.billing_usage import _fmt_usd, format_renews

    if model is None or not getattr(model, "available", False):
        return {"ok": True, "available": False}

    return {
        "ok": True,
        "available": True,
        "status": model.status,
        "plan_name": model.plan_name,
        "renews_at": model.renews_at,
        "renews_display": getattr(model, "renews_display", None) or format_renews(model.renews_at),
        "subscription_remaining_display": (
            None if model.subscription_remaining_usd is None else _fmt_usd(model.subscription_remaining_usd)
        ),
        "topup_remaining_display": (
            None if model.topup_remaining_usd is None else _fmt_usd(model.topup_remaining_usd)
        ),
        "total_spendable_display": (
            None if model.total_spendable_usd is None else _fmt_usd(model.total_spendable_usd)
        ),
        "has_topup": model.has_topup,
        "plan_bar": _serialize_usage_bar(model.plan_bar),
        "topup_bar": _serialize_usage_bar(model.topup_bar),
    }


def _serialize_subscription_state(state) -> dict:
    """Serialize a SubscriptionState for the wire (Decimals → strings)."""
    from agent.billing_usage import format_renews
    from agent.billing_view import format_money

    def _s(value):
        return None if value is None else str(value)

    current = None
    if state.current is not None:
        c = state.current
        current = {
            "tier_id": c.tier_id,
            "tier_name": c.tier_name,
            "monthly_credits": _s(c.monthly_credits),
            "credits_remaining": _s(c.credits_remaining),
            "cycle_ends_at": c.cycle_ends_at,
            "pending_downgrade_tier_name": c.pending_downgrade_tier_name,
            "pending_downgrade_at": c.pending_downgrade_at,
            "pending_downgrade_display": format_renews(c.pending_downgrade_at),
            "cancel_at_period_end": c.cancel_at_period_end,
            "cancellation_effective_at": c.cancellation_effective_at,
            "cancellation_effective_display": format_renews(c.cancellation_effective_at),
        }
    # Selectable catalog for the in-terminal tier picker; price is pre-formatted
    # ($X / $X.YY) so the TUI renders it directly.
    tiers = [
        {
            "tier_id": t.tier_id,
            "name": t.name,
            "tier_order": t.tier_order,
            "dollars_per_month_display": format_money(t.dollars_per_month),
            "monthly_credits": _s(t.monthly_credits),
            "is_current": t.is_current,
            "is_enabled": t.is_enabled,
        }
        for t in state.tiers
    ]
    return {
        "ok": True,
        "logged_in": state.logged_in,
        "is_admin": state.is_admin,
        "can_change_plan": state.can_change_plan,
        "org_name": state.org_name,
        "org_id": state.org_id,
        "role": state.role,
        "context": state.context,
        "current": current,
        "tiers": tiers,
        "portal_url": state.portal_url,
        "error": state.error,
        # Shared dollar usage model (two-bar view) embedded so /subscription
        # renders the same bars as /usage from its single fetch. Built from the
        # separate account-info path (the only source with top-up dollars);
        # fail-open → {available:false}. Computed lazily so a logged-out state
        # adds no cost.
        "usage": _usage_payload(state),
    }


def _serialize_subscription_preview(p) -> dict:
    """Serialize a SubscriptionChangePreview for the wire (Decimal → string)."""
    return {
        "ok": True,
        "effect": p.effect,
        "reason": p.reason,
        "current_tier_id": p.current_tier_id,
        "current_tier_name": p.current_tier_name,
        "target_tier_id": p.target_tier_id,
        "target_tier_name": p.target_tier_name,
        "monthly_credits_delta": (
            None if p.monthly_credits_delta is None else str(p.monthly_credits_delta)
        ),
        "amount_due_now_cents": p.amount_due_now_cents,
        "effective_at": p.effective_at,
    }


# ── Delegation: subagent tree observability + controls ───────────────
# Powers the TUI's /agents overlay (see ui-tui/src/components/agentsOverlay).
# The registry lives in tools/delegate_tool — these handlers are thin
# translators between JSON-RPC and the Python API.


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


# ── Methods: prompt ──────────────────────────────────────────────────


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
    key/lineage. Used as the fail-closed gate for every addressed notification:
    "not provably elsewhere" is NOT good enough to inject a payload into this
    chat (#55578).
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


def _notification_event_requires_owner(evt: dict) -> bool:
    """Whether ``evt`` must be positively claimed before TUI delivery."""
    return evt.get("type") == "async_delegation" or bool(
        str(evt.get("origin_ui_session_id") or "")
        or str(evt.get("session_key") or "")
    )


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
            evt.get("message_id", ""),
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


# Mirror gateway/kanban_watchers.py TERMINAL_KINDS: claim silent kinds too so
# the cursor advances past them and they can't wedge a later completed/blocked
# event behind an unclaimed row.
_KANBAN_NOTIFY_KINDS = (
    "completed", "blocked", "gave_up", "crashed", "timed_out",
    "status", "archived", "unblocked",
)
_KANBAN_SILENT_KINDS = frozenset({"archived", "unblocked"})
_KANBAN_POLL_SECONDS = 5.0


def _format_kanban_event_text(sub: dict, task, ev, board_slug: str) -> Optional[str]:
    """Single-line notification text for one kanban event.

    Wording mirrors the gateway notifier (gateway/kanban_watchers.py) so a
    task completion reads the same in the TUI as it does on Telegram.
    Returns None for kinds that are claimed but intentionally silent.
    """
    kind = getattr(ev, "kind", "")
    if not kind or kind in _KANBAN_SILENT_KINDS:
        return None
    task_id = sub.get("task_id", "")
    title = (getattr(task, "title", None) or task_id)[:120]
    board_tag = f"[{board_slug}] " if board_slug else ""
    who = getattr(task, "assignee", None) or ""
    tag = f"@{who} " if who else ""
    payload = getattr(ev, "payload", None) or {}
    if kind == "completed":
        handoff = ""
        summary = payload.get("summary")
        if summary:
            lines = str(summary).strip().splitlines()
            handoff = f"\n{lines[0][:200]}" if lines else ""
        elif getattr(task, "result", None):
            lines = str(task.result).strip().splitlines()
            handoff = f"\n{lines[0][:160]}" if lines else ""
        return f"✔ {board_tag}{tag}Kanban {task_id} done — {title}{handoff}"
    if kind == "blocked":
        reason = f": {str(payload.get('reason'))[:160]}" if payload.get("reason") else ""
        return f"⏸ {board_tag}{tag}Kanban {task_id} blocked{reason}"
    if kind == "gave_up":
        err = f"\n{str(payload.get('error'))[:200]}" if payload.get("error") else ""
        return f"✖ {board_tag}{tag}Kanban {task_id} gave up after repeated spawn failures{err}"
    if kind == "crashed":
        return f"✖ {board_tag}{tag}Kanban {task_id} worker crashed (pid gone); dispatcher will retry"
    if kind == "timed_out":
        limit = 0
        try:
            limit = int(payload.get("limit_seconds") or 0)
        except (TypeError, ValueError):
            pass
        return f"⏱ {board_tag}{tag}Kanban {task_id} timed out (max_runtime={limit}s); will retry"
    if kind == "status":
        return f"🔄 {board_tag}{tag}Kanban {task_id} → {payload.get('status') or ''}"
    return None


def _collect_kanban_notifications(session: dict) -> list:
    """Claim unseen terminal kanban events for this TUI session's subscriptions.

    ``kanban_create`` auto-subscribes TUI/desktop sessions with
    ``platform="tui"`` and ``chat_id=HERMES_SESSION_KEY`` (see
    tools/kanban_tools.py ``_maybe_auto_subscribe``). The gateway notifier
    can't deliver those — there is no "tui" messaging adapter — so this
    poller is the delivery path for them (issue #59890). Uses the same
    atomic cursor-claim (``claim_unseen_events_for_sub``) as the gateway
    notifier, so a subscription is delivered exactly once even if a gateway
    and a TUI poll the same board DB.

    Returns the list of formatted notification texts (may be empty).
    """
    session_key = str(session.get("session_key") or "")
    if not session_key or session.get("_finalized"):
        return []
    try:
        from hermes_cli import kanban_db as _kb
    except Exception:
        return []
    texts: list = []
    try:
        boards = _kb.list_boards(include_archived=False)
    except Exception:
        try:
            boards = [_kb.read_board_metadata(_kb.DEFAULT_BOARD)]
        except Exception:
            return []
    # Poll each resolved DB path once — multiple slugs can point at the same
    # DB when HERMES_KANBAN_DB pins the board path (same guard as the gateway
    # notifier).
    seen_db_paths: set = set()
    for board_meta in boards:
        slug = (board_meta or {}).get("slug") or _kb.DEFAULT_BOARD
        db_path = (board_meta or {}).get("db_path")
        try:
            resolved = (
                str(Path(db_path).expanduser().resolve())
                if db_path else str(_kb.kanban_db_path(slug).resolve())
            )
        except Exception:
            resolved = f"slug:{slug}"
        if resolved in seen_db_paths:
            continue
        seen_db_paths.add(resolved)
        # A poller runs per live TUI/Desktop session. Avoid opening this board
        # writable unless it has a subscription owned by this exact session;
        # subscriptions for gateways or other sessions are not actionable here.
        try:
            if _kb.count_notify_subs(
                board=slug,
                platform="tui",
                chat_id=session_key,
            ) == 0:
                continue
        except Exception:
            # Preserve delivery if the read-only probe cannot inspect a
            # locked, corrupt, or otherwise unusual database.
            pass
        try:
            conn = _kb.connect(board=slug)
        except Exception:
            continue
        try:
            try:
                subs = _kb.list_notify_subs(conn)
            except Exception:
                continue
            for sub in subs:
                if (sub.get("platform") or "").lower() != "tui":
                    continue
                if sub.get("chat_id") != session_key:
                    continue
                _old, _new, events = _kb.claim_unseen_events_for_sub(
                    conn,
                    task_id=sub["task_id"],
                    platform=sub["platform"],
                    chat_id=sub["chat_id"],
                    thread_id=sub.get("thread_id") or "",
                    kinds=_KANBAN_NOTIFY_KINDS,
                )
                if not events:
                    continue
                task = _kb.get_task(conn, sub["task_id"])
                for ev in events:
                    text = _format_kanban_event_text(sub, task, ev, slug)
                    if text:
                        texts.append(text)
                # Unsubscribe only at a truly final status (done/archived);
                # blocked/crashed subs stay live so a respawned task's next
                # terminal event still reaches the user (same rule as the
                # gateway notifier).
                if task and getattr(task, "status", "") in {"done", "archived"}:
                    try:
                        _kb.remove_notify_sub(
                            conn,
                            task_id=sub["task_id"],
                            platform=sub["platform"],
                            chat_id=sub["chat_id"],
                            thread_id=sub.get("thread_id") or "",
                        )
                    except Exception:
                        pass
        finally:
            conn.close()
    return texts


def _notification_poller_loop(
    stop_event: threading.Event, sid: str, session: dict
) -> None:
    """Poll completion_queue and dispatch notifications autonomously.

    Runs in a daemon thread started by _init_session(). Emits a
    status.update (kind=process) for user visibility, then chains an
    agent turn via _run_prompt_submit if the session is idle.

    The completion_queue is process-global. In multi-session Desktop each
    poller requeues events owned by another live session and drops addressed
    events whose owner is gone; ownerless legacy notifications remain global.

    Also polls ``kanban_notify_subs`` every ``_KANBAN_POLL_SECONDS`` for this
    session's TUI kanban subscriptions and delivers terminal task events the
    same way (status.update + agent turn) — the delivery path
    tools/kanban_tools.py documents for platform="tui" rows (issue #59890).
    """
    from tools.process_registry import process_registry, format_process_notification

    _emitted = set()  # dedup re-queued events so same completion isn't emitted 50 times while session is busy
    _last_kanban_poll = 0.0
    while not stop_event.is_set() and not session.get("_finalized"):
        _now = time.monotonic()
        if _now - _last_kanban_poll >= _KANBAN_POLL_SECONDS:
            _last_kanban_poll = _now
            try:
                _kanban_texts = _collect_kanban_notifications(session)
            except Exception as _kb_exc:
                print(
                    f"[tui_gateway] kanban notification poll failed: "
                    f"{type(_kb_exc).__name__}: {_kb_exc}",
                    file=sys.stderr,
                )
                _kanban_texts = []
            if _kanban_texts:
                for _kb_text in _kanban_texts:
                    _emit("status.update", sid, {"kind": "process", "text": _kb_text})
                # Events are cursor-claimed (never re-queued), so buffer them
                # until the session is idle instead of dropping the agent turn.
                session.setdefault("_kanban_pending", []).extend(_kanban_texts)
            _pending = session.get("_kanban_pending") or []
            if _pending:
                _batch: list = []
                with session["history_lock"]:
                    if not session.get("running"):
                        session["running"] = True
                        _batch = list(_pending)
                        session["_kanban_pending"] = []
                if _batch:
                    rid = f"__notif__{int(time.time() * 1000)}"
                    try:
                        _emit("message.start", sid)
                        _run_prompt_submit(rid, sid, session, "\n".join(_batch))
                    except Exception as exc:
                        print(
                            f"[tui_gateway] kanban notification dispatch failed: "
                            f"{type(exc).__name__}: {exc}",
                            file=sys.stderr,
                        )
                        with session["history_lock"]:
                            session["running"] = False
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

        # What reaches here is not owned by another LIVE session. Addressed
        # events still require positive proof before injection: exact UI origin,
        # direct durable key, or compression lineage. If none proves ownership,
        # the event is orphaned and must not be adopted by this chat. Truly
        # ownerless ordinary notifications retain legacy global delivery.
        requires_owner = _notification_event_requires_owner(evt)
        if requires_owner and not _session_owns_notification_event(sid, session, evt):
            log = (
                logger.warning
                if evt.get("type") == "async_delegation"
                else logger.debug
            )
            log(
                "Dropping unowned %s notification (origin=%r key=%r) instead "
                "of delivering to session %s",
                evt.get("type", "completion"),
                str(evt.get("origin_ui_session_id") or ""),
                str(evt.get("session_key") or ""),
                sid,
            )
            continue

        _evt_sid = evt.get("session_id", "")
        if evt.get("type") == "completion" and process_registry.is_completion_consumed(_evt_sid):
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
        from tools.async_delegation import (
            claim_event_delivery, complete_event_delivery, release_event_delivery,
        )
        _claim = claim_event_delivery(evt, "tui-poller")
        if _claim is None:
            continue
        try:
            _emit("message.start", sid)
            if evt.get("type") == "async_delegation":
                _run_prompt_submit(
                    rid,
                    sid,
                    session,
                    text,
                    display_kind="async_delegation_complete",
                    display_metadata=_async_delegation_display_metadata(evt),
                )
            else:
                _run_prompt_submit(rid, sid, session, text)
            complete_event_delivery(evt, _claim)
        except Exception as exc:
            release_event_delivery(evt, _claim)
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
    # Orphaned events (owner gone) are dropped — same guard as the main loop.
    deferred: list = []
    while not process_registry.completion_queue.empty():
        try:
            evt = process_registry.completion_queue.get_nowait()
        except Exception:
            break
        if _notification_event_belongs_elsewhere(sid, session, evt):
            deferred.append(evt)
            continue
        # Same positive-proof rule as the live loop. Preserve the existing
        # shutdown behavior for orphaned delegation payloads by deferring them
        # for a later resume; ordinary addressed orphans are dropped.
        requires_owner = _notification_event_requires_owner(evt)
        if requires_owner and not _session_owns_notification_event(sid, session, evt):
            if evt.get("type") == "async_delegation":
                deferred.append(evt)
            else:
                logger.debug(
                    "Dropping unowned %s notification during shutdown drain "
                    "(origin=%r key=%r)",
                    evt.get("type", "completion"),
                    str(evt.get("origin_ui_session_id") or ""),
                    str(evt.get("session_key") or ""),
                )
            continue
        _evt_sid = evt.get("session_id", "")
        if evt.get("type") == "completion" and process_registry.is_completion_consumed(_evt_sid):
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
        from tools.async_delegation import (
            claim_event_delivery, complete_event_delivery, release_event_delivery,
        )
        _claim = claim_event_delivery(evt, "tui-poller")
        if _claim is None:
            continue
        try:
            _emit("message.start", sid)
            if evt.get("type") == "async_delegation":
                _run_prompt_submit(
                    rid,
                    sid,
                    session,
                    text,
                    display_kind="async_delegation_complete",
                    display_metadata=_async_delegation_display_metadata(evt),
                )
            else:
                _run_prompt_submit(rid, sid, session, text)
            complete_event_delivery(evt, _claim)
        except Exception as exc:
            release_event_delivery(evt, _claim)
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


def _async_delegation_display_metadata(evt: dict) -> dict:
    """Build display-only metadata before the completion event is formatted."""
    raw_results = evt.get("results")
    results: list[dict] = [
        result for result in raw_results if isinstance(result, dict)
    ] if isinstance(raw_results, list) else []
    task_count = len(results) or 1
    completed_count = sum(
        1 for result in results
        if result.get("status") in {"completed", "success"}
    )
    failed_count = sum(
        1 for result in results
        if result.get("status") in {"failed", "error"}
    )
    metadata = {
        "delegation_id": str(evt.get("delegation_id") or ""),
        "task_count": task_count,
        "completed_count": completed_count or task_count - failed_count,
        "failed_count": failed_count,
    }
    duration = evt.get("total_duration_seconds") or evt.get("duration_seconds")
    if isinstance(duration, (int, float)):
        metadata["duration_seconds"] = duration
    return metadata


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


_desktop_ui_wired = False


def _wire_desktop_ui() -> None:
    """Bridge desktop-only tools (open_preview, focus_pane) to renderer events.

    Idempotent. The tool hands back the turn's ``HERMES_UI_SESSION_ID`` as
    ``sid`` so the event routes to the window that asked (``_emit`` /
    ``write_json`` is ``_stdout_lock``-guarded, so calling it from the tool's
    thread is safe)."""
    global _desktop_ui_wired
    if _desktop_ui_wired:
        return
    try:
        from tools import desktop_ui
    except Exception:
        return

    desktop_ui.set_emitter(lambda sid, event, payload: _emit(event, sid, payload))
    _desktop_ui_wired = True


def _start_notification_poller(sid: str, session: dict) -> threading.Event:
    """Start the background notification poller for a TUI session."""
    _wire_agent_terminal_output()
    _wire_desktop_ui()
    stop = threading.Event()
    t = threading.Thread(
        target=_notification_poller_loop,
        args=(stop, sid, session),
        daemon=True,
    )
    t.start()
    return stop


def _run_prompt_submit(
    rid,
    sid: str,
    session: dict,
    text: Any,
    *,
    display_kind: str | None = None,
    display_metadata: dict | None = None,
    image_paths: list[str] | None = None,
    queued_prompt_generation: int | None = None,
) -> None:
    with session["history_lock"]:
        if (
            queued_prompt_generation is not None
            and int(session.get("_queued_prompt_generation", 0)) != queued_prompt_generation
        ):
            session["running"] = False
            return
        if image_paths is None:
            images = list(session.get("attached_images", []))
            session["attached_images"] = []
        else:
            images = list(image_paths)
        inflight = session.get("inflight_turn")
        # A retained failed turn (see _fail_inflight_turn) is a stale leftover
        # by the time a new turn starts — replace it, never append onto it.
        if not isinstance(inflight, dict) or inflight.get("status") == "error":
            _start_inflight_turn(session, text)
        agent = session["agent"]
        if hasattr(agent, "clear_interrupt"):
            try:
                agent.clear_interrupt()
            except Exception:
                pass
    _emit("message.start", sid)

    def run():
        # The conversation runs on a fresh thread, so ContextVars from the RPC
        # dispatcher do not follow automatically. Rebind the exact transport
        # stored on this session generation before any tool can commission a
        # child; delegate_task then captures it as non-serializable authority.
        transport_token = bind_transport(session.get("transport"))
        runtime_session_token = _current_runtime_session_record.set(session)
        approval_token = None
        session_tokens = []
        home_token = None  # per-turn HERMES_HOME override for a resumed remote profile
        secret_token = None
        goal_followup = None  # set by the post-turn goal hook below
        result = None  # turn outcome; read after the finally for leftover /steer
        tts_queue = None  # streaming-TTS feed for this turn (voice mode)
        thinking_started = False  # ambient thinking sound armed for this turn
        one_turn_restore = session.pop("one_turn_model_restore", None)
        # True once a failed turn's snapshot was retained for resume replay —
        # tells the finally below to skip the normal inflight clear.
        turn_error_retained = False
        # Durable crash marker: written before the turn runs, retired the
        # moment its outcome reaches the client (see _retire_turn_marker).
        # Any concluded turn — success, handled error, interrupt — retires
        # it, so a marker that survives means the process died mid-turn;
        # session.resume auto-continues from it. Compression can rotate
        # session_key mid-turn, so remember the key we wrote under.
        marker_home = _session_home(session)
        marker_key = str(session.get("session_key") or "")
        marker_attempt = int(session.pop("_auto_continue_attempt", 0) or 0)
        marker_text = session.pop("_auto_continue_prompt", None) or text
        if isinstance(marker_text, str) and marker_text.strip():
            record_turn_start(marker_home, marker_key, marker_text, attempts=marker_attempt)
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
                secret_token = set_secret_scope(build_profile_secret_scope(Path(_profile_home_str)))
            # The sudo password callback is thread-local (tools.terminal_tool
            # _callback_tls), so wiring it on the build thread doesn't reach this
            # turn thread — terminal sudo prompts would fall through to /dev/tty
            # and hang the headless gateway. Re-wire here so the prompt routes to
            # the sudo.request overlay. (secret capture is a module global, so
            # re-running is a harmless no-op.)
            _wire_callbacks(sid)
            # Skip the config-model sync while a /model --once override is
            # active: the once-model is intentionally not pinned as a session
            # model_override (it must not persist), so without this guard the
            # sync would see "agent model != config model" and clobber the
            # once-override back to the config model before the turn runs
            # (#29923 review defect). Any config.yaml change is adopted on
            # the NEXT turn, after the finally-restore below.
            if not one_turn_restore:
                # A model picked mid-turn was queued (not applied in-place) —
                # apply it now, on the turn thread before the first model call,
                # so this turn runs on the model the user chose. Runs before the
                # config sync so an explicit pick wins over a config.yaml change.
                _apply_pending_model_switch(sid, session)
                _sync_agent_model_with_config(sid, session)
            # Snapshot after turn-start model sync. A deferred switch mutates
            # history and its version; that mutation belongs to this turn.
            with session["history_lock"]:
                history = list(session["history"])
                history_version = int(session.get("history_version", 0))
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
                    from hermes_cli.config import load_config as _tui_load_config

                    _cfg = _tui_load_config()
                    _provider, _model = _active_image_routing_identity(agent)
                    _mode = decide_image_input_mode(
                        _provider,
                        _model,
                        _cfg,
                        requested_provider=getattr(
                            agent, "requested_provider", ""
                        ),
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

            # Streaming TTS: voice-mode replies are spoken sentence-by-sentence
            # as tokens arrive (CLI parity) instead of after the full turn.
            # begin() first — it cuts any still-speaking previous turn, and
            # that cut IS this turn's barge-in, so it must latch before we
            # consume the latch below.
            tts_queue = _tts_stream_begin()

            # Full-duplex agent-turn listener: armed at utterance-submit so
            # the user can interject DURING generation, not just during
            # playback. _tts_stream_begin arms it too when a pipeline
            # starts; this covers voice mode without working TTS.
            if _voice_mode_enabled() and _voice_cfg_dict().get("barge_in", True):
                _arm_full_duplex_listener()

            # Ambient "thinking" sound (voice mode only): calm bubble blips
            # while the agent works with no audio flowing, so long
            # thinking/tool stretches don't read as a dead session. Per-blip
            # gate skips while real TTS audio flows or the mic is capturing;
            # stopped in the finally the instant the turn ends.
            # voice.thinking_sound config-gates it; macOS TCC handled inside.
            thinking_started = False
            if _voice_mode_enabled():
                try:
                    from tools.voice_mode import (
                        is_audio_output_active,
                        start_thinking_sound,
                    )

                    def _thinking_should_play() -> bool:
                        if is_audio_output_active():
                            return False
                        try:
                            from hermes_cli.voice import is_continuous_active

                            return not is_continuous_active()
                        except Exception:
                            return True

                    thinking_started = start_thinking_sound(
                        should_play=_thinking_should_play
                    )
                except Exception:
                    thinking_started = False

            # Barged mid-speech? Tell the model (API-message note, same
            # enrichment channel as attached images) so it can react
            # ("rude!") instead of being oblivious to its own interruption.
            from tools.tts_streaming import SPEECH_INTERRUPTED_NOTE, take_speech_interrupted

            if take_speech_interrupted():
                if isinstance(run_message, str):
                    run_message = f"{SPEECH_INTERRUPTED_NOTE}\n\n{run_message}"
                elif isinstance(run_message, list):
                    run_message = [{"type": "text", "text": SPEECH_INTERRUPTED_NOTE}, *run_message]

            # Reactions the user added since the last turn ride the MODEL INPUT
            # only (same enrichment channel as the speech-interrupted note);
            # persist_user_message below stays the clean prompt, so no
            # scaffolding reaches the transcript. Cache-safe: annotating the
            # NEW turn never rewrites an already-sent message.
            if reaction_notes := _pending_reaction_notes(session):
                if isinstance(run_message, str):
                    run_message = f"{reaction_notes}\n\n{run_message}"
                elif isinstance(run_message, list):
                    run_message = [{"type": "text", "text": reaction_notes}, *run_message]

            def _stream(delta):
                with session["history_lock"]:
                    _append_inflight_delta(session, delta)
                payload = {"text": delta}
                if streamer and (r := streamer.feed(delta)) is not None:
                    payload["rendered"] = r
                if tts_queue is not None and isinstance(delta, str):
                    tts_queue.put(delta)
                _emit("message.delta", sid, payload)

            # Surface interim assistant text (commentary emitted alongside
            # tool calls, or the attempted final answer before a verify-on-stop
            # nudge) so the desktop can seal it as its own segment instead of
            # losing it when message.complete replaces the streaming buffer.
            # Gated on display.interim_assistant_messages (default true).
            if _load_interim_assistant_messages():
                def _interim_assistant_cb(text: str, *, already_streamed: bool = False) -> None:
                    _emit("message.interim", sid, {
                        "text": text,
                        "already_streamed": already_streamed,
                    })

                agent.interim_assistant_callback = _interim_assistant_cb
            else:
                agent.interim_assistant_callback = None

            run_kwargs = {
                "conversation_history": list(history),
                "stream_callback": _stream,
                "persist_user_message": (
                    _build_persist_user_message(prompt, images, run_message) if images else prompt
                ),
            }
            # Type a synthesized turn at turn START so the crash persist writes
            # its row as a timeline event, instead of leaving a raw user bubble
            # until the turn ends — and forever if it never does, which is
            # exactly the auto-continue case. The post-turn stamp below is the
            # fallback for an older agent without the parameter; re-stamping
            # the same value is a no-op.
            try:
                _run_params = inspect.signature(agent.run_conversation).parameters
            except (TypeError, ValueError):
                _run_params = {}
            if "task_id" in _run_params:
                run_kwargs["task_id"] = session["session_key"]
            if display_kind and "persist_user_display_kind" in _run_params:
                run_kwargs["persist_user_display_kind"] = display_kind
                run_kwargs["persist_user_display_metadata"] = display_metadata
            result = agent.run_conversation(run_message, **run_kwargs)
            if display_kind and isinstance(text, str):
                db = getattr(agent, "_session_db", None)
                current_session_id = getattr(agent, "session_id", None) or session.get("session_key")
                if db is not None:
                    try:
                        db.set_latest_matching_message_display_kind(
                            current_session_id,
                            role="user",
                            content=text,
                            display_kind=display_kind,
                            display_metadata=display_metadata,
                        )
                    except Exception:
                        logger.debug("failed to stamp synthetic display kind", exc_info=True)
                if isinstance(result, dict) and isinstance(result.get("messages"), list):
                    for message in reversed(result["messages"]):
                        if message.get("role") == "user" and message.get("content") == text:
                            message["display_kind"] = display_kind
                            if display_metadata:
                                message["display_metadata"] = display_metadata
                            break
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
                            # History mutated externally during the turn.
                            # Check if the only mutation was a model-switch
                            # marker inserted mid-turn (#76870).  If so the
                            # agent output is still valid — merge it into the
                            # current history that now contains the marker.
                            #
                            # _append_model_switch_marker strips prior markers
                            # in-place then appends a new one, so the delta
                            # is NOT a simple tail-slice — we must compare
                            # content, not indices.
                            current_history = list(session["history"])
                            history_no_markers = [
                                e for e in history if not _is_model_switch_marker(e)
                            ]
                            current_no_markers = [
                                e for e in current_history if not _is_model_switch_marker(e)
                            ]
                            model_switch_only = (
                                current_no_markers == history_no_markers
                                and any(
                                    _is_model_switch_marker(e)
                                    for e in current_history
                                )
                            )
                            if model_switch_only:
                                # The agent's new messages start after the
                                # turn-start history.  Guard against
                                # auto-compression making result["messages"]
                                # shorter than history (#77274 review).
                                if len(result["messages"]) > len(history):
                                    new_messages = result["messages"][len(history):]
                                else:
                                    # Compression rebound the messages list —
                                    # use the full result as the base.
                                    new_messages = list(result["messages"])
                                session["history"] = current_history + new_messages
                                session["history_version"] = current_version + 1
                            else:
                                # Genuine desync (undo/compress/retry/rollback).
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
                    sid, session, clear_pending_title=False, restart_slash_worker=True,
                )

                raw = result.get("final_response", "")
                status = (
                    "interrupted"
                    if result.get("interrupted")
                    else "error" if result.get("error") else "complete"
                )
                # When the backend produced no visible response AND reported a
                # real error (e.g. invalid model slug → provider 4xx), surface
                # that error as the visible text instead of shipping an empty
                # turn to Ink. Mirrors classic CLI behavior at cli.py where
                # (failed|partial) + no final_response → "Error: <detail>".
                # Leaves the None-with-no-error path untouched: an empty
                # successful turn still renders as empty, and the existing
                # "(empty)" sentinel handling stays in its own lane.
                if (not raw) and result.get("error") and (
                    result.get("failed") or result.get("partial")
                ):
                    raw = f"Error: {result.get('error')}"
                # "Operation interrupted: waiting for model response (…)" is
                # cancellation metadata, not assistant prose. gateway/run.py
                # and the ACP adapter already suppress this sentinel; without
                # this the desktop paints it as the agent's reply whenever a
                # stop/steer lands mid-request (#7921).
                if status == "interrupted" and isinstance(raw, str) and raw.strip().startswith(
                    INTERRUPT_WAITING_FOR_MODEL_PREFIX
                ):
                    raw = ""
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
            if result.get("response_previewed"):
                payload["response_previewed"] = True
            # Forward the structured billing-wall descriptor (provider,
            # billing_url, is_nous, message) so the TUI/desktop render a
            # billing-specific recovery surface instead of re-parsing text.
            _billing_block = result.get("billing_block") if isinstance(result, dict) else None
            if _billing_block:
                payload["billing"] = _billing_block
                payload["failure_reason"] = result.get("failure_reason")
            rendered = render_message(raw, cols)
            if rendered:
                payload["rendered"] = rendered
            with session["history_lock"]:
                if status == "error":
                    # Returned-error result (provider 4xx, budget, etc.): retain
                    # the failed turn for resume replay instead of clearing it.
                    # If this terminal frame is lost to a disconnect, resume's
                    # inflight payload is the only carrier of the failure.
                    _fail_inflight_turn(
                        session,
                        result.get("error") if isinstance(result, dict) else raw,
                    )
                    turn_error_retained = True
                else:
                    _clear_inflight_turn(session)
            if status == "error":
                payload["error"] = str(
                    (result.get("error") if isinstance(result, dict) else "") or raw
                )
                payload["recoverable"] = True
            _retire_turn_marker(session, marker_key)
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

            # Apply pending_title now that the DB row exists — in the
            # session-owned profile store (not the launch profile).
            _pending = session.get("pending_title")
            if _pending and status == "complete":
                _session_key = session.get("session_key") or sid
                try:
                    with _session_db(session) as _pdb:
                        if _pdb and _pdb.set_session_title(_session_key, _pending):
                            session["pending_title"] = None
                except ValueError as exc:
                    # Invalid/duplicate title — non-retryable, drop it.
                    # Auto-title will take over. Fix for #19029.
                    session["pending_title"] = None
                    logger.info(
                        "Dropping pending title for session %s: %s",
                        _session_key, exc,
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
                    # Snapshot the runtime identity; the validator lets the
                    # background titler skip its LLM call if the session's
                    # model changed before it fires (#19027).
                    _title_model = getattr(agent, "model", None)
                    _title_provider = getattr(agent, "provider", None)
                    maybe_auto_title(
                        _get_db(),
                        _title_key,
                        text,
                        raw,
                        session.get("history", []),
                        # Keep auxiliary auto-detection aligned with the active
                        # Desktop/Webapp session. Without this, providers that
                        # rely on runtime auth (for example OpenAI Codex OAuth)
                        # are skipped and the new session remains untitled.
                        main_runtime={
                            "model": getattr(agent, "model", None),
                            "provider": getattr(agent, "provider", None),
                            "base_url": getattr(agent, "base_url", None),
                            "api_key": getattr(agent, "api_key", None),
                            "api_mode": getattr(agent, "api_mode", None),
                        },
                        runtime_validator=lambda: (
                            getattr(agent, "model", None) == _title_model
                            and getattr(agent, "provider", None) == _title_provider
                        ),
                        # Push the generated title live so the sidebar renames
                        # without waiting for the next list refresh (the titler
                        # runs async, after this turn's refresh already fired).
                        title_callback=lambda t, _k=_title_key: _emit(
                            "session.title", sid, {"session_id": _k, "title": t}
                        ),
                    )
                except Exception:
                    pass

            # Voice TTS fallback: when the streaming pipeline couldn't start
            # (no provider / missing deps probed at turn start), speak the
            # final text whole (cli.py:_voice_speak_response parity). The
            # streaming path already spoke everything via tts_queue.
            if (
                status == "complete"
                and tts_queue is None
                and isinstance(raw, str)
                and raw.strip()
                and _voice_tts_enabled()
            ):
                try:
                    spoken = raw
                    # Barge-aware: spoken interruptions must cut this
                    # fallback playback too, not just the streaming path.
                    threading.Thread(
                        target=_speak_text_with_barge, args=(spoken,), daemon=True
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
            # The agent persists its working transcript on normal finalization,
            # but an exception in that finalizer can otherwise leave the
            # gateway's separate in-memory history at the turn-start snapshot.
            # Keep the partial turn available to the next prompt; the durable
            # inflight record still carries the recoverable error state.
            _restore_agent_history_after_turn_error(session, agent)
            try:
                # Close the turn with the same terminal error frame shape as
                # the returned-error path (uniform client handling), retaining
                # the failed turn for resume replay.
                _emit_terminal_turn_error(sid, session, e)
                turn_error_retained = True
            except Exception as emit_exc:
                print(
                    f"[gateway-turn] terminal error emit failed: "
                    f"{type(emit_exc).__name__}: {emit_exc}",
                    file=sys.stderr,
                    flush=True,
                )
                _emit("error", sid, {"message": str(e)})
        finally:
            # Drop both local snapshots of the pre-turn history before asking
            # glibc to return pages. session["history"] already points at the
            # new/pruned result; retaining either list defeats this trim.
            history.clear()
            local_run_kwargs = locals().get("run_kwargs")
            if isinstance(local_run_kwargs, dict):
                local_run_kwargs.clear()

            # Run while any profile-specific HERMES_HOME override is still active
            # so context.memory_trim is resolved from the session's own config.
            try:
                from hermes_cli.mem_trim import trim_memory

                trim_memory(reason="tui turn completion")
            except Exception:
                logger.debug("post-turn memory trim failed", exc_info=True)

            if thinking_started:
                # Kill the ambient thinking sound the moment the turn ends —
                # error and success paths both land here.
                try:
                    from tools.voice_mode import stop_thinking_sound

                    stop_thinking_sound()
                except Exception:
                    pass
            if tts_queue is not None:
                tts_queue.put(None)  # end-of-text sentinel — flush + finish speaking
            if one_turn_restore:
                try:
                    _restore_agent_model_runtime(agent, one_turn_restore)
                    _restart_slash_worker(sid, session)
                    _persist_live_session_runtime(session)
                    _persist_live_session_system_prompt(session)
                except Exception:
                    logger.debug("TUI one-turn model restore failed", exc_info=True)
            try:
                if approval_token is not None:
                    reset_current_session_key(approval_token)
            except Exception:
                pass
            if home_token is not None:
                reset_hermes_home_override(home_token)
            if secret_token is not None:
                reset_secret_scope(secret_token)
            _clear_session_context(session_tokens)
            _current_runtime_session_record.reset(runtime_session_token)
            reset_transport(transport_token)
            # Clear the per-turn interim callback so a stale closure from
            # this turn can't fire during a later turn on the same agent.
            agent.interim_assistant_callback = None
            with session["history_lock"]:
                session["running"] = False
                session["last_active"] = time.time()
                if not turn_error_retained:
                    _clear_inflight_turn(session)
            # Backstop for turns that never reached a terminal frame (the
            # frame paths retire the marker as they emit).
            _retire_turn_marker(session, marker_key)
            session.pop("_auto_continue_scheduled", None)
            _emit_settled_session_info(sid, session, agent)

        # A user prompt that arrived mid-turn (interrupt + queue) wins over
        # every auto follow-up below — drain it first and skip them this cycle;
        # the goal judge / notifications re-evaluate at the end of that turn.
        # Leftover /steer: the steer arrived after the last tool batch (e.g.
        # during the final API call), so the agent couldn't inject it and
        # returned it in result["pending_steer"]. Requeue it as the next turn
        # so it isn't silently dropped — same rule as cli.py and gateway/run.py.
        # A real queued prompt still wins: the merge in _enqueue_prompt keeps
        # both texts.
        _leftover_steer = result.get("pending_steer") if isinstance(result, dict) else None
        if isinstance(_leftover_steer, str) and _leftover_steer.strip():
            with session["history_lock"]:
                _enqueue_prompt(session, _leftover_steer, session.get("transport"))
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
        #
        # Ownership filter (#42674, #35652): a turn finishing in session B
        # must not consume an event that belongs to session A. The registry
        # requeues every addressed event this session cannot positively claim;
        # the poller then delivers it to a live owner or drops an orphan.
        try:
            from tools.process_registry import process_registry

            # Positive-proof ownership (compression-chain aware) — the same
            # fail-closed gate the poller uses, so the post-turn drain can't
            # adopt another session's addressed notification while a
            # post-compression session still claims its own pre-compression
            # dispatches (#55578).
            drained = process_registry.drain_notifications(
                session_key=session.get("session_key", ""),
                owns_event=lambda e: _session_owns_notification_event(sid, session, e),
                skip_poll_observed=False,
            )
            for index, (_evt, synth) in enumerate(drained):
                with session["history_lock"]:
                    if session.get("running"):
                        for pending_evt, _pending_synth in drained[index:]:
                            process_registry.completion_queue.put(pending_evt)
                        break
                    session["running"] = True
                from tools.async_delegation import (
                    claim_event_delivery, complete_event_delivery, release_event_delivery,
                )
                _claim = claim_event_delivery(_evt, "tui-post-turn")
                if _claim is None:
                    continue
                try:
                    _emit("message.start", sid)
                    _run_prompt_submit(rid, sid, session, synth)
                    complete_event_delivery(_evt, _claim)
                except Exception as _n_exc:
                    release_event_delivery(_evt, _claim)
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


def _session_images_dir(session: dict) -> Path:
    """Resolve the uploads ``images/`` dir against the session's effective home.

    Attach RPCs (``image.attach_bytes``, ``clipboard.paste``, ``pdf.attach``)
    run BEFORE ``prompt.submit`` installs the session's profile HERMES_HOME
    override, so ``get_hermes_home()`` here would return the gateway's launch
    home. In a multi-profile / root-gateway deployment that writes the upload to
    the launch home's ``images/`` while the sandbox mount and the vision host-
    read allowlist both resolve the *session profile's* ``images/`` at run time
    — so the file the agent tries to read is never the file we wrote (#69575).

    Anchor the write on the session's stored ``profile_home`` when present
    (matching the mount/read scope), else fall back to the launch home. Keeps
    per-profile isolation: a profile's uploads stay under that profile's home.
    """
    profile_home = session.get("profile_home")
    base = Path(profile_home) if profile_home else _hermes_home
    return base / "images"


def _queue_attached_image(session: dict, img_bytes: bytes, ext: str, *, prefix: str) -> Path:
    """Write image bytes into the gateway's images dir and queue them.

    Mirrors what ``image.attach`` does for a local path: appends to
    ``session["attached_images"]`` so the next ``prompt.submit`` picks it up via
    the existing native-image-attach pipeline. Returns the written path.
    """
    session["image_counter"] = session.get("image_counter", 0) + 1
    img_dir = _session_images_dir(session)
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


# ── Methods: respond ─────────────────────────────────────────────────


def _respond(rid, params, key, *, allow_expired=False):
    r = params.get("request_id", "")
    with _prompt_lock:
        entry = _pending.get(r)
        if not entry:
            if allow_expired and r:
                return _ok(rid, {"status": "expired"})
            return _err(rid, 4009, f"no pending {key} request")
        _, ev = entry
        _answers[r] = params.get(key, "")
        ev.set()
    return _ok(rid, {"status": "ok"})


# ── Methods: config ──────────────────────────────────────────────────


# NOTE: config.set intentionally stays in server.py for now — the in-flight
# opt/model-resolution-core PR touches its body; move it to methods_config.py
# in a follow-up once that PR lands.
@method("config.set")
def _(rid, params: dict) -> dict:
    key, value = params.get("key", ""), params.get("value", "")
    session = _sessions.get(params.get("session_id", ""))

    if key == "model":
        try:
            if not value:
                return _err(rid, 4002, "model value required")
            if session:
                from hermes_cli.model_switch import parse_model_switch_args

                # A live swap can't run in-place while a turn streams:
                # agent.switch_model() mutates self.model / self.provider /
                # self.base_url / self.client, and the worker thread running
                # agent.run_conversation reads those every iteration — a
                # mid-turn swap can fire an HTTP request with the new base_url
                # but old model (400/404s).  So instead of rejecting the pick
                # (the old 4009), stash it and apply it at the NEXT turn start
                # (_apply_pending_model_switch), where nothing is in flight.
                # The user gets to pick, keep typing, and send the next turn on
                # the new model without waiting for the swap or interrupting.
                if session.get("running"):
                    parsed = parse_model_switch_args(value)
                    try:
                        pending_model = parsed.model_input
                    except Exception:
                        pending_model = str(value)
                    session["pending_model_switch"] = {
                        "raw": value,
                        "confirm_expensive_model": bool(
                            params.get("confirm_expensive_model", False)
                        ),
                        # The resolved model/provider the next turn will run on.
                        # _session_info reports these while the switch is pending
                        # so the end-of-turn settle keeps showing the user's pick
                        # instead of blipping back to the still-live old model.
                        "display_model": pending_model,
                        "display_provider": (
                            getattr(parsed, "explicit_provider", "") or ""
                        ).strip(),
                    }
                    return _ok(
                        rid,
                        {
                            "key": key,
                            "value": pending_model,
                            "warning": "",
                            "confirm_required": False,
                            "confirm_message": "",
                            "scope": "session",
                            "deferred": True,
                        },
                    )
                parsed_flags = parse_model_switch_args(value)
                explicit_provider = parsed_flags.explicit_provider
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
                    "scope": result.get("scope", "session"),
                },
            )
        except Exception as e:
            return _err(rid, 5001, str(e))

    if key == "fast":
        raw = str(value or "").strip().lower()
        agent = session.get("agent") if session else None
        if agent is not None:
            current_fast = getattr(agent, "service_tier", None) == "priority"
        elif session is not None and session.get("create_service_tier_override") is not None:
            # Pre-build session with a pinned tier (desktop draft pick or an
            # earlier session-scoped toggle) — report/toggle from the pin, not
            # the global default.
            current_fast = session["create_service_tier_override"] == "priority"
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

            if agent is not None:
                target_model = getattr(agent, "model", None)
            else:
                # A pre-build session may already have a picked model riding in
                # model_override (desktop draft) — validate fast support against
                # THAT model, not the global default it will never use.
                session_override = (session or {}).get("model_override") or {}
                target_model = (
                    session_override.get("model")
                    if isinstance(session_override, dict)
                    else None
                ) or _resolve_model()
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

        if session is not None:
            # Session-scoped, like `reasoning` below (global persistence is
            # `--global` / Settings → Model territory). Writing config.yaml
            # here let every desktop model-menu selection (per-model fast
            # preset) rewrite the user's global agent.service_tier — flipping
            # fast mode for every OTHER session, profile, CLI, and gateway
            # build ("switch one session, switches everywhere"). Pin the
            # create override so lazily-built sessions and rebuilds (/new,
            # deferred resume) keep the choice; "" pins normal explicitly.
            session["create_service_tier_override"] = (
                "priority" if nv == "fast" else ""
            )
        else:
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

    if key == "focus":
        # Focus view — display-only reduced-output mode (/focus). Composes with
        # the tool_progress machinery rather than duplicating it: enabling it
        # pins tool_progress to "off" (the same value /verbose off uses) after
        # stashing the configured mode, and disabling it restores that mode.
        # Nothing about the request payload changes.
        from hermes_cli.focus_view import (
            FOCUS_TOOL_PROGRESS_MODE,
            normalize_tool_progress_mode,
            resolve_focus_arg,
        )

        cfg_f = _load_cfg()
        _display_f = cfg_f.get("display")
        d_f: dict = _display_f if isinstance(_display_f, dict) else {}
        cur_focus = bool(d_f.get("focus_view", False))
        action, target = resolve_focus_arg(str(value or ""), cur_focus)
        if action == "usage":
            return _err(rid, 4002, f"unknown focus value: {value} (use on|off|status)")
        if action == "status" or target is None:
            return _ok(
                rid,
                {
                    "key": key,
                    "value": "on" if cur_focus else "off",
                    "tool_progress": _load_tool_progress_mode(),
                },
            )

        if target:
            saved = normalize_tool_progress_mode(
                (d_f.get("focus_saved_tool_progress") or _load_tool_progress_mode())
                if cur_focus
                else _load_tool_progress_mode()
            )
            _write_config_key("display.focus_saved_tool_progress", saved)
            _write_config_key("display.tool_progress", FOCUS_TOOL_PROGRESS_MODE)
            effective = FOCUS_TOOL_PROGRESS_MODE
        else:
            saved = normalize_tool_progress_mode(
                d_f.get("focus_saved_tool_progress") or "all"
            )
            _write_config_key("display.tool_progress", saved)
            effective = saved
        _write_config_key("display.focus_view", bool(target))

        if session:
            session["focus_view"] = bool(target)
            session["tool_progress_mode"] = effective
            agent_f = session.get("agent")
            if agent_f is not None:
                try:
                    agent_f.tool_progress_mode = effective
                except Exception:
                    pass
        return _ok(
            rid,
            {
                "key": key,
                "value": "on" if target else "off",
                "tool_progress": effective,
            },
        )

    if key in {"approval_mode", "approvals.mode"}:
        raw = str(value or "").strip().lower()
        if raw not in _APPROVAL_MODES:
            return _err(
                rid,
                4002,
                f"unknown approval mode: {value}; pick one of manual|smart|off",
            )

        _write_config_key("approvals.mode", raw)
        for sid, sess in list(_sessions.items()):
            agent = sess.get("agent")
            if agent is not None:
                _emit("session.info", sid, _session_info(agent, sess))
        return _ok(rid, {"key": "approvals.mode", "value": raw})

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
            scope = str(params.get("scope") or "").strip().lower()
            global_scope = scope == "global"
            if arg in {"show", "on"}:
                cfg = _load_cfg_raw()  # write-back round-trip
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
                cfg = _load_cfg_raw()  # write-back round-trip
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
                cfg = _load_cfg_raw()  # write-back round-trip
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
                cfg = _load_cfg_raw()  # write-back round-trip
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
            if global_scope or session is None:
                _write_config_key("agent.reasoning_effort", arg)
                if session is not None:
                    session.pop("create_reasoning_override", None)
            else:
                # Session-scoped, like the messaging gateway's `/reasoning
                # <level>` (global persistence is `--global` / Settings →
                # Model territory). Writing config.yaml here let every
                # desktop model-menu selection rewrite the user's global
                # agent.reasoning_effort to the preset default.
                session["create_reasoning_override"] = parsed
            if session and session.get("agent") is not None:
                session["agent"].reasoning_config = parsed
                _persist_live_session_runtime(session)
                _emit(
                    "session.info",
                    params.get("session_id", ""),
                    _session_info(session["agent"], session),
                )
            return _ok(rid, {"key": key, "value": arg})
        except Exception as e:
            return _err(rid, 5001, str(e))

    if key == "details_mode":
        nv = str(value or "").strip().lower()
        if nv not in _DETAIL_MODES:
            return _err(rid, 4002, f"unknown details_mode: {value}")
        cfg = _load_cfg_raw()  # write-back round-trip
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

        cfg = _load_cfg_raw()  # write-back round-trip
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

    if key == "density":
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
            return _err(rid, 4002, f"unknown density value: {value}")
        _write_config_key("display.tui_compact", nv_b)
        return _ok(rid, {"key": key, "value": "on" if nv_b else "off"})

    if key == "battery":
        raw = str(value or "").strip().lower()
        cfg0 = _load_cfg()
        d0 = cfg0.get("display") if isinstance(cfg0.get("display"), dict) else {}
        cur_b = bool(d0.get("battery", False))
        if raw in {"", "toggle"}:
            nv_b = not cur_b
        elif raw in {"on", "true", "yes"}:
            nv_b = True
        elif raw in {"off", "false", "no"}:
            nv_b = False
        else:
            return _err(rid, 4002, f"unknown battery value: {value}")
        _write_config_key("display.battery", nv_b)
        return _ok(rid, {"key": key, "value": "on" if nv_b else "off"})

    if key == "theme":
        # TUI light/dark mode pin: 'light'/'dark' beat background
        # auto-detection (xterm.js hosts misreport OSC 11); 'auto' trusts it.
        raw = str(value or "").strip().lower()
        if raw not in {"auto", "light", "dark"}:
            return _err(rid, 4002, f"unknown theme value: {value} (use auto|light|dark)")
        _write_config_key("display.tui_theme", raw)
        return _ok(rid, {"key": key, "value": raw})

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
        if raw not in INDICATOR_STYLES:
            return _err(
                rid,
                4002,
                f"unknown indicator: {raw!r}; pick one of {'|'.join(INDICATOR_STYLES)}",
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
            {"key": "terminal.cwd", "value": cwd, "cwd": cwd, "branch": _git_branch_for_cwd(cwd)},
        )

    if key in {"prompt", "personality", "skin"}:
        try:
            cfg = _load_cfg_raw()  # write-back round-trip ("prompt" saves cfg)
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
                    # Every connected surface repaints, not just the RPC's
                    # client; then sync the watcher baseline so the poll loop
                    # doesn't re-broadcast the skin this RPC just applied.
                    _broadcast_global_event("skin.changed", resolve_skin())
                    _note_skin_broadcast()
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


def _is_session_cwd_junk(cwd: str) -> bool:
    """A non-git cwd that should stay in flat Recents rather than auto-group.

    Unlike discovered git roots, an explicitly selected descendant of
    HERMES_HOME may be an intentional prose/data workspace. The pre-Projects
    desktop surfaced every such cwd, so exclude only the two broad defaults
    that would create catch-all projects.
    """
    if not cwd:
        return True

    from hermes_constants import get_hermes_home

    real = os.path.normcase(os.path.realpath(cwd))
    home = os.path.normcase(os.path.realpath(os.path.expanduser("~")))
    hermes_home = os.path.normcase(os.path.realpath(str(get_hermes_home())))
    return real == home or real == hermes_home


def _repo_discovery_policy(raw: dict | None = None) -> dict:
    """Return the effective, profile-local Desktop repository scan policy."""
    from hermes_cli.config import DEFAULT_CONFIG

    defaults = DEFAULT_CONFIG["desktop"]
    source = raw if isinstance(raw, dict) else (_load_cfg().get("desktop") or {})
    if not isinstance(source, dict):
        source = {}

    enabled = source.get("enabled", source.get("repo_scan_enabled", defaults["repo_scan_enabled"]))
    roots = source.get("roots", source.get("repo_scan_roots", defaults["repo_scan_roots"]))
    excludes = source.get(
        "exclude_paths",
        source.get("repo_scan_exclude_paths", defaults["repo_scan_exclude_paths"]),
    )

    return {
        "enabled": enabled if isinstance(enabled, bool) else defaults["repo_scan_enabled"],
        "roots": [value.strip() for value in roots if isinstance(value, str) and value.strip()]
        if isinstance(roots, list)
        else list(defaults["repo_scan_roots"]),
        "exclude_paths": [
            value.strip()
            for value in excludes
            if isinstance(value, str) and value.strip()
        ]
        if isinstance(excludes, list)
        else list(defaults["repo_scan_exclude_paths"]),
    }


def _repo_discovery_policy_key(policy: dict) -> str:
    def _paths(values: list[str]) -> list[str]:
        normalized = set()
        home = os.path.expanduser("~")
        for value in values:
            expanded = os.path.expanduser(value)
            if not os.path.isabs(expanded):
                expanded = os.path.join(home, expanded)
            normalized.add(os.path.normcase(os.path.abspath(expanded)))
        return sorted(normalized)

    canonical = {
        "enabled": bool(policy["enabled"]),
        "roots": _paths(policy["roots"]),
        "exclude_paths": _paths(policy["exclude_paths"]),
    }
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"))


def _repo_discovery_policy_is_default(policy: dict) -> bool:
    from hermes_cli.config import DEFAULT_CONFIG

    return _repo_discovery_policy_key(policy) == _repo_discovery_policy_key(
        _repo_discovery_policy(DEFAULT_CONFIG["desktop"])
    )


def _discover_repos_payload(
    db, *, conn=None, backfill: bool = True, include_cached: bool = True
) -> list[dict]:
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

    if not include_cached:
        out = sorted(repos.values(), key=lambda repo: repo["last_active"], reverse=True)
        for repo in out:
            repo["label"] = (
                repo["label"]
                or os.path.basename(repo["root"].rstrip("/\\"))
                or repo["root"]
            )
        return out

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
                # NOTE: `last_seen` is when the disk scan last saw the directory,
                # not when the user last worked in it. Folding it into
                # `last_active` stamped every scanned repo with the scan time —
                # i.e. "just now" — so a git checkout with zero Hermes sessions
                # outranked the repos the user actually works in. Activity stays
                # session-derived; a repo with no sessions has no activity.

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


# Sources excluded from the project tree: cron runs, and kanban dispatcher
# workers, are not user conversations. Subagent/compression children are
# already dropped by list_sessions_rich(include_children=False); cron has its
# own section, and kanban runs are read on the board.
_PROJECT_TREE_EXCLUDED_SOURCES = ["cron", "kanban"]


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

    policy = _repo_discovery_policy()
    policy_key = _repo_discovery_policy_key(policy)
    with pdb.connect_closing() as conn:
        if include_discovered:
            pdb.reconcile_discovered_repos_policy(
                conn,
                policy_key,
                preserve_unversioned=_repo_discovery_policy_is_default(policy),
            )
        projects = [p.to_dict() for p in pdb.list_projects(conn)]
        active_id = pdb.get_active_id(conn)
        # backfill stays off the hot tree path — grouping uses the live resolver.
        discovered = (
            _discover_repos_payload(
                db,
                conn=conn,
                backfill=False,
                include_cached=policy["enabled"],
            )
            if include_discovered
            else []
        )

    return sessions, projects, discovered, active_id


# Per-build memo for `_dir_exists_cached`. Cleared at the top of every
# `_build_project_tree`, so a dir created or deleted between sidebar refreshes
# is seen on the next one.
_DIR_EXISTS_CACHE: dict[str, bool] = {}


def _dir_exists_cached(path: str) -> bool:
    """``os.path.isdir`` for the project tree, memoized per build.

    ``build_tree`` asks per SESSION, not per distinct path, so a power user with
    hundreds of sessions across a handful of dirs would otherwise fire hundreds
    of redundant stats on every sidebar open. The memo is per build, so a dir
    created or deleted between refreshes is picked up on the next one.
    """
    hit = _DIR_EXISTS_CACHE.get(path)
    if hit is None:
        hit = os.path.isdir(path)
        _DIR_EXISTS_CACHE[path] = hit
    return hit


def _build_project_tree(
    db, *, preview_limit: int, hydrate: bool, session_limit: int, include_discovered: bool
) -> tuple[dict, str | None]:
    """Gather inputs and run the one authoritative builder. Returns (tree, active_id)."""
    from tui_gateway import project_tree

    _DIR_EXISTS_CACHE.clear()
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
        is_junk_cwd=_is_session_cwd_junk,
        exists=_dir_exists_cached,
    )
    return tree, active_id


# ── Methods: tools & system ──────────────────────────────────────────


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


# reload.mcp runs on the RPC pool (see _LONG_HANDLERS) so a slow/flapping MCP
# server can't freeze the reader thread. Serialize reloads: overlapping
# shutdown+discover pairs from stacked config-change polls would interleave
# and leave the registry half-built.
_mcp_reload_lock = threading.Lock()
# Bumped once per SUCCESSFUL shutdown+discover. A follower that waited on the
# lock only skips the redundant reload if this advanced while it waited — i.e.
# the leader actually completed. If the leader threw (flapping server), the
# follower sees no advance and re-runs the full reload itself.
_mcp_reload_gen = 0
# The mcp_rev hash that the last successful reload actually LOADED (config
# re-hashed after discovery, so it reflects what discover_mcp_tools read —
# not what the caller hoped for). A follower coalesces only when the
# revision it was asked to load matches this; otherwise the config changed
# under the leader (rev A loaded, rev B requested) and the follower must
# re-run the full reload itself instead of acking B against A's registry.
_mcp_reload_loaded_rev = ""
# Bounded convergence for a config edit racing a slow reload: the leader
# re-hashes after discovery and repeats until the hash is stable.
_MCP_RELOAD_MAX_PASSES = 3


def _compute_mcp_rev() -> str:
    """Hash of the MCP-relevant config sections (server definitions,
    settings, toolset enables). ``config.get mtime`` ships it to the TUI so
    cosmetic writes don't trigger reloads; ``reload.mcp`` uses it for
    revision-aware coalescing. Empty string = unknown (fail open)."""
    try:
        cfg = _load_cfg()
        # mcp_servers holds the server DEFINITIONS the classic CLI watches
        # for auto-reload (cli.py::_check_config_mcp_changes) — omitting it
        # meant editing a server bumped mtime but not mcp_rev, so the TUI
        # skipped reload.mcp and new servers never connected until a manual
        # /reload-mcp. `mcp` (settings) and `tools` (enable/disable) round
        # out the MCP-relevant surface.
        rev_src = json.dumps(
            {"mcp": cfg.get("mcp"), "mcp_servers": cfg.get("mcp_servers"), "tools": cfg.get("tools")},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha1(rev_src.encode()).hexdigest()[:12]
    except Exception:
        return ""


def _finish_reload(rid, params: dict, *, coalesced: bool) -> dict:
    """Shared tail for both reload paths: honor ``always`` (persist the
    confirm opt-out) and return the ok payload."""
    if bool(params.get("always", False)):
        try:
            from cli import save_config_value as _save_cfg

            _save_cfg("approvals.mcp_reload_confirm", False)
        except Exception as _exc:
            logger.warning("Failed to persist mcp_reload_confirm=false: %s", _exc)

    payload = {"status": "reloaded", "loaded_rev": _mcp_reload_loaded_rev}
    if coalesced:
        payload["coalesced"] = True

    return _ok(rid, payload)


_TUI_HIDDEN: frozenset[str] = frozenset(
    {
        "sethome",
        "set-home",
        "commands",
        "approve",
        "deny",
    }
)

_TUI_EXTRA: list[tuple[str, str, str]] = [
    ("/density", "Toggle compact display mode", "TUI"),
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
        "init",
        "compress",
        "compact",
    }
)

_WORKER_BLOCKED_COMMANDS: frozenset[str] = frozenset({"snapshot", "snap"})


def _skill_usage_lookup():
    """Build ``(usage, origin)`` callables for the skill-command catalog.

    ``usage(name)`` is the skill's observed activity count (use + view +
    patch); ``origin(name)`` is ``"hub"``, ``"bundled"``, or ``"local"`` — the
    same classification ``/api/skills`` reports as ``provenance`` (where
    "local" is spelled "agent"). Both read sidecar files that are cheap and
    already parsed once per catalog build. Any failure degrades to zero usage
    and ``"local"`` so a missing/corrupt sidecar can never break the catalog.
    """
    try:
        from tools.skill_usage import (
            _read_bundled_manifest_names,
            _read_hub_installed_names,
            activity_count,
            load_usage,
        )

        records = load_usage()
        bundled = _read_bundled_manifest_names()
        hub = _read_hub_installed_names()
    except Exception as e:
        logger.debug("skill usage lookup unavailable: %s", e)
        return (lambda _name: 0), (lambda _name: "local")

    def usage(name: str) -> int:
        try:
            return activity_count(records.get(name) or {})
        except Exception:
            return 0

    def origin(name: str) -> str:
        if name in hub:
            return "hub"
        if name in bundled:
            return "bundled"
        return "local"

    return usage, origin


_SLASH_COMPLETION_LIMIT = 30


def _rank_slash_completions(
    items: list[dict],
    usage,
    origin_of,
    *,
    browsing: bool,
) -> list[dict]:
    """Rank and bound slash completions the way the menu should read.

    ``usage``/``origin_of`` are the callables :func:`_skill_usage_lookup`
    returns. Registry commands keep their existing order — only the skill
    block is reordered, most-used first and A-Z within a tie, so the handful
    of skills someone invokes daily lead the ones that shipped with Hermes
    and were never opened.

    The limit is spent PER KIND rather than on one flat truncation. A flat
    cut is positional, not editorial: the completer emits every registry
    command before the first skill, so on a 230-skill install a bare ``/``
    hit the cap while still inside the command block and offered no skill at
    all, and ``/p`` dropped ``/proving-a-fix-works`` (471 uses) while keeping
    ``/pretext`` (2).

    ``browsing`` separates the two things a slash means. A bare ``/`` is
    BROWSING, so bundled skills with no recorded activity are dropped as
    noise. A typed query is SEARCHING, and a search that hides a match is
    broken — there nothing is pruned, the ranking only reorders.
    """

    def name_of(item: dict) -> str:
        return str(item.get("text", "")).strip().lstrip("/").lower()

    commands = [item for item in items if item.get("kind") != "skill"]
    skills = [item for item in items if item.get("kind") == "skill"]

    if browsing:
        skills = [
            item
            for item in skills
            if origin_of(name_of(item)) != "bundled" or usage(name_of(item)) > 0
        ]

    skills.sort(key=lambda item: (-usage(name_of(item)), name_of(item)))

    return commands[:_SLASH_COMPLETION_LIMIT] + skills[:_SLASH_COMPLETION_LIMIT]


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


def _resolve_name(name: str) -> str:
    try:
        from hermes_cli.commands import resolve_command

        r = resolve_command(name)
        return r.name if r else name
    except Exception:
        return name


# ── Methods: paste ────────────────────────────────────────────────────

_paste_counter = 0


# ── Methods: complete ─────────────────────────────────────────────────

_FUZZY_CACHE_TTL_S = 5.0
_FUZZY_CACHE_MAX_FILES = 20000
_FUZZY_FALLBACK_EXCLUDES = frozenset(
    {
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
    }
)
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


def _abs_completion_prefix_exists(path_part: str) -> bool:
    """True when ``path_part`` reads sensibly as an absolute path.

    A leading `/` is only meant literally if something is actually there:
    the parent directory has to exist, and a partially-typed final segment
    has to match at least one of its entries. Used to decide whether
    `@/foo` is the absolute `/foo` or shorthand for `foo` under the cwd.
    """
    expanded = _normalize_completion_path(path_part)
    parent = os.path.dirname(expanded.rstrip("/")) or "/"
    tail = os.path.basename(expanded.rstrip("/"))

    if not os.path.isdir(parent):
        return False

    if not tail or expanded.endswith("/"):
        return os.path.isdir(expanded) or expanded == "/"

    try:
        tail_lower = tail.lower()
        return any(e.lower().startswith(tail_lower) for e in os.listdir(parent))
    except OSError:
        return False


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
                    else "cycle global mode" if candidate == "cycle" else "global mode"
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


def _model_picker_context(agent):
    """Layer live session state onto config without losing custom identity."""
    from hermes_cli.inventory import load_picker_context

    ctx = load_picker_context()
    provider = getattr(agent, "provider", "") if agent else ""
    base_url = getattr(agent, "base_url", "") if agent else ""
    if str(provider or "").strip().lower() == "custom":
        try:
            from hermes_cli.runtime_provider import canonical_custom_identity

            provider = (
                canonical_custom_identity(
                    base_url=base_url or None,
                    config_provider=ctx.current_provider,
                    model=(getattr(agent, "model", "") if agent else "")
                    or None,
                )
                or provider
            )
        except Exception:
            logger.debug(
                "custom provider identity recovery failed (model picker)",
                exc_info=True,
            )

    return ctx.with_overrides(
        current_provider=provider,
        current_model=(getattr(agent, "model", "") if agent else "")
        or _resolve_model(),
        current_base_url=base_url,
    )


# ── Methods: slash.exec ──────────────────────────────────────────────


_LIVE_SESSION_DIRECT_COMMANDS = frozenset(
    {
        "clear",
        "compress",
        "effort",
        "history",
        "models",
        "prompt",
        "rename",
        "status",
        "usage",
    }
)

_ISOLATED_SESSION_READ_COMMANDS = frozenset({"context", "tools", "help"})


def _format_live_usage_output(session: dict) -> str:
    agent = session.get("agent")
    usage = _session_usage_snapshot(session)
    if agent is None and not usage:
        return "(._.) No active agent -- send a message first."
    if session.get("_metadata_message_count") is not None:
        message_count = int(session.get("_metadata_message_count") or 0)
    else:
        with session["history_lock"]:
            message_count = len(session.get("history", []))
    lines = [
        "Session Token Usage",
        "────────────────────────────────────────",
        f"Model: {usage.get('model') or _metadata_mirror(session).get('model') or getattr(agent, 'model', '') or '(unknown)'}",
        f"Input tokens:                 {int(usage.get('input') or 0):,}",
        f"Output tokens:                {int(usage.get('output') or 0):,}",
    ]
    reasoning = int(usage.get("reasoning") or 0)
    if reasoning:
        lines.append(f"Reasoning tokens:             {reasoning:,}")
    lines.extend(
        [
            f"Prompt tokens:                {int(usage.get('prompt') or 0):,}",
            f"Completion tokens:            {int(usage.get('completion') or 0):,}",
            f"Total tokens:                 {int(usage.get('total') or 0):,}",
            f"API calls:                    {int(usage.get('calls') or 0):,}",
        ]
    )
    if usage.get("context_max"):
        lines.append(
            "Current context:              "
            f"{int(usage.get('context_used') or 0):,} / "
            f"{int(usage.get('context_max') or 0):,} "
            f"({int(usage.get('context_percent') or 0)}%)"
        )
    lines.extend(
        [
            f"Messages:                     {message_count:,}",
            f"Compressions:                 {int(usage.get('compressions') or 0):,}",
        ]
    )
    return "\n".join(lines)


def _format_live_history_output(session: dict) -> str:
    with session["history_lock"]:
        history = list(session.get("history", []))
    db = _get_db()
    if db is not None and session.get("session_key"):
        try:
            history = db.get_messages_as_conversation(
                session["session_key"], include_ancestors=True, include_row_ids=True
            )
        except Exception:
            pass
    messages = _history_to_messages(history)
    if not messages:
        return "No conversation history yet."
    lines = ["Conversation History", "────────────────────────────────────────"]
    for idx, message in enumerate(messages, start=1):
        role = str(message.get("role") or "unknown")
        label = "You" if role == "user" else "Hermes" if role == "assistant" else role.title()
        text = str(message.get("text") or message.get("context") or "").strip()
        if len(text) > 400:
            text = f"{text[:400]}..."
        lines.append(f"[{label} #{idx}] {text or '(no text)'}")
    return "\n".join(lines)


def _format_live_prompt_output(session: dict) -> str:
    agent = session.get("agent")
    mirror = _metadata_mirror(session)
    if agent is None and "system_prompt" not in mirror:
        return "No active agent -- send a message first."
    prompt = (
        mirror.get("system_prompt")
        or getattr(agent, "ephemeral_system_prompt", None)
        or getattr(agent, "_cached_system_prompt", None)
        or ""
    )
    if not prompt:
        return "Current system prompt is not built yet; send a message first."
    return f"Current system prompt:\n{prompt}"


def _format_live_context_output(session: dict) -> str:
    messages = []
    db = _get_db()
    if db is not None and session.get("session_key"):
        try:
            messages = _history_to_messages(
                db.get_messages_as_conversation(
                    session["session_key"], include_ancestors=True, include_row_ids=True
                )
            )
        except Exception:
            messages = []
    if not messages:
        with session["history_lock"]:
            messages = _history_to_messages(list(session.get("history", [])))
    usage = _session_usage_snapshot(session)
    mirror = _metadata_mirror(session)
    lines = [
        f"Conversation: {len(messages)} messages" if messages else "Conversation is empty (no messages yet)."
    ]
    roles: dict[str, int] = {}
    for msg in messages:
        role = str(msg.get("role") or "unknown")
        roles[role] = roles.get(role, 0) + 1
    lines.append(
        f"  user: {roles.get('user', 0)}, assistant: {roles.get('assistant', 0)}, "
        f"tool: {roles.get('tool', 0)}, system: {roles.get('system', 0)}"
    )
    model = mirror.get("model") or usage.get("model") or ""
    provider = mirror.get("provider") or "auto"
    if model:
        lines.append(f"Model: {model}")
    lines.append(f"Provider: {provider}")
    context_used = int(usage.get("context_used") or usage.get("total") or 0)
    context_max = int(usage.get("context_max") or 0)
    if context_used:
        if context_max:
            usage_pct = (context_used / context_max) * 100
            lines.append(
                f"Context usage: ~{context_used:,} / {context_max:,} tokens ({usage_pct:.1f}%)"
            )
        else:
            lines.append(f"Context usage: ~{context_used:,} tokens")
    if usage.get("compressions"):
        lines.append(f"Compressions: {int(usage.get('compressions') or 0):,}")
    return "\n".join(lines)


def _format_live_tools_output(session: dict) -> str:
    info = _session_info(session.get("agent"), session)
    groups = info.get("tools") if isinstance(info, dict) else {}
    if not isinstance(groups, dict) or not groups:
        return "No tools available."
    names: list[str] = []
    for group_names in groups.values():
        if isinstance(group_names, list):
            names.extend(str(name) for name in group_names)
    names = sorted(set(names))
    if not names:
        return "No tools available."
    return "Available tools ({}):\n{}".format(
        len(names), "\n".join(f"  {name}" for name in names)
    )


def _format_live_help_output() -> str:
    try:
        from hermes_cli.commands import COMMANDS_BY_CATEGORY

        lines = ["Available commands:", ""]
        for category, commands in COMMANDS_BY_CATEGORY.items():
            lines.append(f"{category}:")
            for cmd, desc in commands.items():
                lines.append(f"  {cmd:<15} {desc}")
        return "\n".join(lines)
    except Exception as exc:
        return f"help unavailable: {exc}"


def _format_live_model_output(session: dict) -> str:
    agent = session.get("agent")
    model = getattr(agent, "model", "") if agent is not None else ""
    provider = getattr(agent, "provider", "") if agent is not None else ""
    if model and provider:
        return f"Current model: {model} ({provider})"
    if model:
        return f"Current model: {model}"
    return "Current model: (unknown)"


def _live_slash_command_output(sid: str, session: Optional[dict], name: str, arg: str) -> Optional[str]:
    name = (name or "").lstrip("/").lower()
    arg = arg or ""
    if name == "model" and not arg.strip():
        return _format_live_model_output(session or {})
    if name not in _LIVE_SESSION_DIRECT_COMMANDS:
        if not (
            name in _ISOLATED_SESSION_READ_COMMANDS
            and session is not None
            and _session_uses_compute_host(session)
        ):
            return None

    if name in _ISOLATED_SESSION_READ_COMMANDS and not (
        session is not None and _session_uses_compute_host(session)
    ):
        return None
    if name == "compress":
        if session is None:
            return "no active session for /compress"
        return _mirror_slash_side_effects(sid, session, f"/compress {arg}".strip())
    if name == "usage":
        if session is None:
            return "(._.) No active agent -- send a message first."
        return _format_live_usage_output(session)
    if name == "history":
        if session is None:
            return "No conversation history yet."
        return _format_live_history_output(session)
    if name == "prompt":
        if session is None:
            return "No active agent -- send a message first."
        return _format_live_prompt_output(session)
    if name == "status":
        response = _methods["session.status"]("status", {"session_id": sid})
        if response.get("error"):
            return str(response["error"].get("message") or "status unavailable")
        return str(response.get("result", {}).get("output") or "")
    if name == "context":
        if session is None:
            return "Conversation is empty (no messages yet)."
        return _format_live_context_output(session)
    if name == "tools":
        if session is None:
            return "No tools available."
        return _format_live_tools_output(session)
    if name == "help":
        return _format_live_help_output()
    if name == "clear":
        return "Screen clear is terminal-only; desktop/TUI chat left unchanged."
    if name == "models":
        return "Use /model to view or switch the current model; desktop users can also open the model picker."
    if name == "rename":
        return "Use /title <name> to rename this session."
    if name == "effort":
        return "Use /reasoning <effort> to change reasoning effort."
    return None



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
    if name == "compact":
        # /compact is an alias of /compress in every host. The compute-host
        # slash.compress control forwards the user's raw alias verbatim, so
        # without normalizing here the child mirror silently no-ops — the
        # session never compresses and the deferred context-engine
        # notification wiring below is never exercised for that route.
        name = "compress"

    # Reject agent-mutating commands during an in-flight turn.  These
    # all do read-then-mutate on live agent/session state that the
    # worker thread running agent.run_conversation is using.  Parity
    # with the session.compress / session.undo guards and the gateway
    # runner's running-agent /model guard.
    _MUTATES_WHILE_RUNNING = {"model", "personality", "prompt", "compress"}
    if _session_uses_compute_host(session) and name in _MUTATES_WHILE_RUNNING:
        route_name = f"slash.{name}"
        try:
            ack = _send_compute_host_control(
                sid,
                route_name=route_name,
                command=command,
                wait=True,
            )
        except Exception as exc:
            return f"compute-host {route_name} failed: {exc}"
        if ack.get("type") in {"control.error", "error"}:
            return str(ack.get("message") or f"compute-host {route_name} failed")
        _apply_compute_host_metadata_mirror(session, ack)
        return str(ack.get("output") or "")
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
            from agent.conversation_compression import (
                finalize_context_engine_compression_notification,
            )

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

            # The raw argument goes through unparsed: _compress_session_history
            # (the choke point shared by all three manual-compress routes)
            # parses the boundary-aware forms (here [N], up to here, --keep N)
            # and does the partial head/tail split there (#35533).
            try:
                _compress_session_history(session, arg)
            except CompressionLockHeld as e:
                from agent.manual_compression_feedback import (
                    describe_compression_lock_skip,
                )
                return describe_compression_lock_skip(e.holder)
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
                _before_messages,
                _after_messages,
                _before_tokens,
                _after_tokens,
                compression_state=getattr(agent, "context_compressor", None),
            )
            _lines = [_fb["headline"], _fb["token_line"]]
            if _fb.get("note"):
                _lines.append(_fb["note"])
            finalize_context_engine_compression_notification(
                agent,
                committed=True,
            )
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
        if name == "compress" and agent:
            from agent.conversation_compression import (
                finalize_context_engine_compression_notification,
            )

            finalize_context_engine_compression_notification(
                agent,
                committed=False,
            )
        return f"live session sync failed: {e}"
    return ""


# ── Methods: voice ───────────────────────────────────────────────────


_voice_sid_lock = threading.Lock()
_voice_event_sid: str = ""
_voice_wake_owner: "Optional[Transport]" = None


def _voice_emit(event: str, payload: dict | None = None) -> None:
    """Emit a voice event toward the session that most recently turned the
    mode on. Voice is process-global (one microphone), so there's only ever
    one sid to target; the TUI handler treats an empty sid as "active
    session". Kept separate from _emit to make the lack of per-call sid
    argument explicit."""
    with _voice_sid_lock:
        sid = _voice_event_sid
    _emit(event, sid, payload)


def _resume_voice_wake() -> None:
    global _voice_wake_owner
    with _voice_sid_lock:
        owner, _voice_wake_owner = _voice_wake_owner, None
    if owner is not None:
        _wake_resume_if_owner(owner)


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


def _any_session_running() -> bool:
    """True while any session's agent turn is in flight.

    Registered as the voice busy-probe (``hermes_cli.voice.set_voice_busy_probe``)
    so silent capture cycles during a long agent turn don't count toward the
    no-speech limit — the user is correctly quiet while the agent works.
    Voice is process-global (one microphone), so any running session holds.
    """
    try:
        with _sessions_lock:
            return any(s.get("running") for s in _sessions.values())
    except Exception:
        return False


# ── Streaming TTS (one active pipeline per process — one speaker) ──────────
# Token deltas from the running turn feed a sentence-buffering consumer
# (tools.tts_tool.stream_tts_to_speaker) so speech starts on the first
# sentence instead of after the full reply. Voice is process-global, so a
# single slot suffices; starting a new turn's pipeline barges in on the
# previous one.

_tts_stream_lock = threading.Lock()
_tts_stream_state: Optional[dict] = None


def _tts_stream_begin() -> Optional[queue.Queue]:
    """Start a per-turn streaming TTS consumer; None when TTS can't stream."""
    if not _voice_tts_enabled():
        return None
    try:
        from tools.tts_tool import check_tts_requirements, stream_tts_to_speaker

        if not check_tts_requirements():
            return None
    except Exception:
        return None

    _tts_stream_stop()
    text_queue: queue.Queue = queue.Queue()
    stop = threading.Event()
    done = threading.Event()
    threading.Thread(
        target=stream_tts_to_speaker, args=(text_queue, stop, done), daemon=True
    ).start()

    global _tts_stream_state
    with _tts_stream_lock:
        _tts_stream_state = {"stop": stop, "done": done}

    if _voice_mode_enabled() and _voice_cfg_dict().get("barge_in", True):
        _arm_full_duplex_listener()

    return text_queue


def _tts_stream_stop(user_barge: bool = True) -> None:
    """Cut any in-flight streaming TTS (new turn, interrupt, /voice off).

    *user_barge* latches the interruption for the next turn's model note
    (``mark_speech_interrupted``) — pass ``False`` for mode changes like
    ``/voice off`` where the user isn't talking over the reply.
    """
    global _tts_stream_state
    with _tts_stream_lock:
        state, _tts_stream_state = _tts_stream_state, None
    if state is None:
        return
    if user_barge and not state["done"].is_set():
        import traceback as _tb
        logger.debug(
            "TTS CUT: _tts_stream_stop(user_barge=True) — new turn or "
            "interrupt cutting in-flight TTS\n%s",
            "".join(_tb.format_stack()),
        )
        from tools.tts_streaming import mark_speech_interrupted

        mark_speech_interrupted()
    state["stop"].set()
    try:
        from tools.voice_mode import stop_playback

        stop_playback()
    except Exception:
        pass


def _tts_stream_barge_in_monitor(stop: threading.Event, done: threading.Event) -> None:
    """Deprecated shim — playback-only monitor replaced by the full-duplex
    agent-turn listener (see ``_full_duplex_listener``). Kept as a name so
    stray callers arm the new listener instead of a per-playback mic."""
    _arm_full_duplex_listener()


# ── Full-duplex agent-turn listener (one mic, whole turn) ──────────────────
# Replaces the per-playback barge monitors: those only opened the mic once
# TTS playback started (deaf during LLM generation) and calibrated the VAD
# floor against active speaker bleed (deaf during playback too, in practice).
# This listener arms at utterance-submit, spans generation AND playback, and
# disarms when no session is running, no TTS is pending, and no audio flows.

_fd_listener_lock = threading.Lock()
_fd_listener_active = False
# (stop, done) pairs for fallback whole-reply speak paths currently active —
# the listener must cut THEIR private stop events too, and must keep
# listening while any of them is still speaking.
_fd_speak_pipelines: "set[tuple[threading.Event, threading.Event]]" = set()


def _arm_full_duplex_listener() -> None:
    """Arm the process-global full-duplex listener (idempotent — one mic)."""
    global _fd_listener_active
    with _fd_listener_lock:
        if _fd_listener_active:
            return
        _fd_listener_active = True
    threading.Thread(
        target=_full_duplex_listener, daemon=True, name="voice-full-duplex"
    ).start()


def _fd_tts_pending() -> bool:
    """True while any TTS (streaming pipeline or fallback speak) is unfinished."""
    with _tts_stream_lock:
        state = _tts_stream_state
    if state is not None and not state["done"].is_set():
        return True
    with _fd_listener_lock:
        pipelines = list(_fd_speak_pipelines)
    return any(not done.is_set() for _stop, done in pipelines)


def _full_duplex_listener() -> None:
    """Mic live from utterance-submit to turn-complete; phase-aware trip.

    * generation phase (no TTS audio flowing): user speech interrupts every
      running session's agent turn — the same ``agent.interrupt()`` seam
      ``session.interrupt`` uses — and cuts any pending TTS pipeline so the
      stale reply never plays. The captured utterance is transcribed and
      emitted as ``voice.transcript`` (the TUI submits it as the next turn).
    * playback phase: cuts TTS (streaming pipeline + fallback speak paths +
      file player) and submits the captured interruption.

    Stop phrase is honored in both phases: mid-generation it interrupts the
    turn AND ends the voice chat ("stop everything").
    """
    global _fd_listener_active
    try:
        from tools.tts_streaming import mark_speech_interrupted
        from tools.voice_mode import (
            full_duplex_listen,
            is_audio_output_active,
            stop_playback,
            transcribe_recording,
        )

        cfg = _voice_cfg_dict()
        try:
            _mult = float(cfg.get("barge_in_threshold_multiplier", 0) or 0)
        except (TypeError, ValueError):
            _mult = 0.0
        try:
            _grace_ms = int(float(cfg.get("barge_in_grace_seconds", 0.5)) * 1000)
        except (TypeError, ValueError):
            _grace_ms = 500

        def _should_stop() -> bool:
            if not _voice_mode_enabled():
                return True
            if _any_session_running():
                return False
            if _fd_tts_pending():
                return False
            return not is_audio_output_active()

        tripped = threading.Event()

        def _cut_all_tts() -> None:
            # Streaming pipeline (private stop event + player).
            _tts_stream_stop(user_barge=True)
            # Fallback whole-reply speak paths (their own stop events).
            with _fd_listener_lock:
                pipelines = list(_fd_speak_pipelines)
            for _stop, _done in pipelines:
                _stop.set()
            stop_playback()

        def _on_trigger(phase: str) -> None:
            tripped.set()
            mark_speech_interrupted()
            if phase == "playback":
                logger.debug(
                    "TTS CUT: full-duplex listener tripped during playback"
                )
                _cut_all_tts()
            else:
                logger.debug(
                    "full-duplex listener tripped during generation — "
                    "interrupting running turn(s)"
                )
                # Cut pending TTS FIRST so the stale reply can never speak.
                _cut_all_tts()
                # Interrupt every running session's turn — voice is
                # process-global, and the same seam session.interrupt uses.
                try:
                    with _sessions_lock:
                        running = [
                            s for s in _sessions.values() if s.get("running")
                        ]
                    for s in running:
                        agent = s.get("agent")
                        if agent is not None and hasattr(agent, "interrupt"):
                            try:
                                agent.interrupt()
                            except Exception:
                                pass
                except Exception as e:
                    logger.debug("voice interjection interrupt failed: %s", e)
            _voice_emit("voice.interrupted")

        wav_path = full_duplex_listen(
            _should_stop,
            is_playing=is_audio_output_active,
            on_trigger=_on_trigger,
            multiplier=_mult or None,
            grace_ms=max(0, _grace_ms),
        )
        if not (wav_path and tripped.is_set()):
            return
        try:
            result = transcribe_recording(wav_path)
            text = (result.get("transcript") or "").strip() if result.get("success") else ""
            if text:
                # Stop-check must never break transcript delivery — if the
                # helper is unavailable (stubbed voice_mode in tests, partial
                # installs), treat as not-a-stop-phrase.
                try:
                    from tools.voice_mode import is_voice_stop_phrase
                    _is_stop = is_voice_stop_phrase(text)
                except Exception:
                    _is_stop = False

                if _is_stop:
                    # Bare stop phrase — in EITHER phase the user means
                    # "stop everything": the turn was already interrupted /
                    # TTS cut at trip time; now end the voice chat.
                    os.environ["HERMES_VOICE"] = "0"
                    os.environ["HERMES_VOICE_TTS"] = "0"
                    try:
                        from hermes_cli.voice import stop_continuous

                        stop_continuous()
                    except Exception:
                        pass
                    _voice_emit("voice.transcript", {"stop_phrase": True, "text": text})
                else:
                    _voice_emit("voice.transcript", {"text": text})
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass
    except Exception as e:
        logger.debug("full-duplex listener failed: %s", e)
    finally:
        with _fd_listener_lock:
            _fd_listener_active = False


def _speak_text_with_barge(text: str) -> None:
    """Speak *text* via hermes_cli.voice.speak_text with spoken barge-in.

    The fallback whole-reply path (streaming couldn't start) and the
    ``voice.tts`` RPC previously called ``speak_text`` bare — speech over
    those paths was UNINTERRUPTIBLE by voice. The full-duplex agent-turn
    listener covers this path too: the (stop, done) pair is registered in
    ``_fd_speak_pipelines`` so the listener can cut the private stop event
    on a playback trip and keeps listening while this speak is pending.
    """
    from hermes_cli.voice import speak_text

    stop = threading.Event()
    done = threading.Event()
    with _fd_listener_lock:
        _fd_speak_pipelines.add((stop, done))

    def _speak():
        try:
            speak_text(text, stop)
        except TypeError:
            # Older wrapper without the stop_event parameter.
            speak_text(text)
        finally:
            done.set()
            with _fd_listener_lock:
                _fd_speak_pipelines.discard((stop, done))

    threading.Thread(target=_speak, daemon=True).start()
    if _voice_mode_enabled() and _voice_cfg_dict().get("barge_in", True):
        _arm_full_duplex_listener()


def _voice_cfg_dict() -> dict:
    """Shape-safe accessor for the ``voice:`` block in config.yaml.

    ``_load_cfg()`` does not deep-merge DEFAULT_CONFIG, so both the
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


# ── Wake word ("Hey Hermes") ──────────────────────────────────────────────
# The detector is process-global (one mic), like voice. The first eligible
# transport to call wake.start owns it until stop, disconnect, or stream failure.
# On detection we emit wake.detected; the client opens a new session and starts
# its own voice capture. The detector yields the mic to gateway voice.record
# (pause/resume below) and to the desktop's browser mic (wake.pause/resume RPCs).
_wake_lock = threading.Lock()
_wake_owner_transport: "Optional[Transport]" = None
_wake_owner_surface = ""


def _wake_owner_snapshot():
    with _wake_lock:
        return _wake_owner_transport, _wake_owner_surface


def _release_wake_for_transport(transport: "Transport") -> bool:
    """Release the wake lease iff ``transport`` is the current gateway owner."""
    global _wake_owner_transport, _wake_owner_surface
    with _wake_lock:
        if _wake_owner_transport is not transport:
            return False
        _wake_owner_transport = None
        _wake_owner_surface = ""
    try:
        from tools.wake_word import stop_listening

        stop_listening(owner=transport)
    except Exception as e:
        logger.debug("wake stop failed: %s", e)
    return True


def _release_gateway_wake_owner() -> bool:
    owner, _surface = _wake_owner_snapshot()
    return owner is not None and _release_wake_for_transport(owner)


_wake_resume_retry_lock = threading.Lock()
_wake_resume_retry_active = False


def _wake_resume_if_owner(owner: "Transport", *, retry_seconds: float = 15.0,
                          retry_interval: float = 1.0) -> bool:
    """Resume the wake detector for ``owner``; self-heal a busy microphone.

    Reopening the mic right after a voice turn can fail while the capture
    device is still being released (browser WebRTC tracks release async).
    The CLI covers this with its idle watchdog; the gateway had nothing, so
    one failed resume left the listener silently dead until the user toggled
    it by hand — despite ``wake_word.enabled: true``. On an exception (mic
    open failure) we retry in a background thread until it sticks, the lease
    changes hands, or ``retry_seconds`` elapses. ``False`` from
    ``resume_listening`` (lease gone / different owner) is final — never
    retried, so this can't steal another surface's mic.
    """
    from tools.wake_word import resume_listening

    try:
        return resume_listening(owner=owner)
    except Exception as e:
        logger.debug("wake resume failed (will retry): %s", e)

    global _wake_resume_retry_active
    with _wake_resume_retry_lock:
        if _wake_resume_retry_active:
            return False
        _wake_resume_retry_active = True

    def _retry() -> None:
        global _wake_resume_retry_active
        deadline = time.monotonic() + retry_seconds
        try:
            while time.monotonic() < deadline:
                time.sleep(retry_interval)
                try:
                    if resume_listening(owner=owner):
                        logger.info("wake: detector resumed after retry")
                        return
                except Exception:
                    continue
                # False — detector gone or lease moved: stop, don't fight it.
                return
            logger.warning(
                "wake: could not resume detector after voice turn "
                "(microphone still busy?) — toggle the wake word to re-arm"
            )
        finally:
            with _wake_resume_retry_lock:
                _wake_resume_retry_active = False

    threading.Thread(target=_retry, daemon=True, name="wake-resume-retry").start()
    return False


def _persist_wake_enabled(enabled: bool) -> bool:
    """Write ``wake_word.enabled`` to config.yaml.

    Only called for explicit user gestures (the desktop ear toggle, ``/wake
    on|off``) — never from passive auto-arm paths, so a mic can't become
    persistently enabled without a deliberate click.
    """
    try:
        from cli import save_config_value

        return bool(save_config_value("wake_word.enabled", enabled))
    except Exception as e:
        logger.warning("wake: failed to persist wake_word.enabled=%s: %s", enabled, e)
        return False


@method("wake.start")
def _(rid, params: dict) -> dict:
    """Arm the wake-word listener for the calling surface ("tui" | "gui").

    Idempotent and gated: returns ``{started: False, reason}`` when the wake
    word is disabled, scoped to another surface, or its deps/mic aren't ready.

    ``persist: true`` marks an explicit user gesture (toggle click, /wake on):
    when the feature is disabled in config, it flips ``wake_word.enabled`` on
    and saves it before arming, so the choice sticks for future sessions.
    Passive auto-arm callers omit it and keep getting the config-gated refusal.
    """
    surface = str(params.get("surface") or "auto").strip().lower()
    persist = bool(params.get("persist"))
    transport = current_transport() or _stdio_transport
    try:
        from tools.wake_word import (
            WakeWordInUse,
            check_wake_word_requirements,
            detector_frame_info,
            load_wake_word_config,
            owns_listener,
            resolve_capture_mode,
            start_listening,
            wake_phrase,
            wake_surface_enabled,
        )
    except Exception as e:
        return _err(rid, 5026, f"wake module unavailable: {e}")

    cfg = load_wake_word_config()
    # Desktop remote (gui) prefers client capture: Mac mic → wake.feed PCM,
    # while the engine still runs on the backend. CLI/TUI stay local.
    prefer_client = surface in ("gui", "desktop") or bool(params.get("client_capture"))
    capture_mode = resolve_capture_mode(cfg, prefer_client=prefer_client)
    external_audio = capture_mode == "client"
    # Requirements first: a gesture on an unarmed-able setup (no STT/TTS, no
    # mic, missing key) must refuse WITHOUT flipping wake_word.enabled — else
    # config says on while nothing can ever arm, and auto-arm paths churn.
    # Temporarily stamp capture so the probe matches the arm mode.
    probe_cfg = dict(cfg)
    probe_cfg["capture"] = capture_mode
    reqs = check_wake_word_requirements(probe_cfg)
    if not reqs["available"]:
        logger.warning("wake.start(%s): not available — %s", surface, reqs.get("hint"))
        return _ok(rid, {
            "started": False,
            "reason": "unavailable",
            "hint": reqs.get("hint") or "",
            "capture": capture_mode,
        })
    enabled_persisted = False
    if persist and not cfg.get("enabled"):
        enabled_persisted = _persist_wake_enabled(True)
        if enabled_persisted:
            cfg = dict(cfg)
            cfg["enabled"] = True
    if not wake_surface_enabled(surface, cfg):
        # Distinguish "feature off in config" (reason: disabled — a persist:true
        # retry can turn it on) from "scoped to a different surface" (reason:
        # disabled_for_surface — respects an explicit wake_word.surface choice,
        # which persist does NOT override).
        reason = "disabled" if not cfg.get("enabled") else "disabled_for_surface"
        logger.info("wake.start(%s): %s (enabled=%s, surface=%s)",
                    surface, reason, cfg.get("enabled"), cfg.get("surface"))
        return _ok(rid, {"started": False, "reason": reason})

    existing_owner, existing_surface = _wake_owner_snapshot()
    if existing_owner is not None and (
        _transport_is_dead(existing_owner) or not owns_listener(existing_owner)
    ):
        _release_wake_for_transport(existing_owner)
        existing_owner = None
        existing_surface = ""
    if existing_owner is not None and existing_owner is not transport:
        return _ok(rid, {
            "started": False,
            "reason": "owned",
            "owner_surface": existing_surface,
        })

    sid = str(params.get("session_id") or "")
    phrase = wake_phrase(cfg)
    new_session = bool(cfg.get("start_new_session", True))

    def _on_detect() -> None:
        from tools.wake_word import get_last_match, owns_listener, pause_listening

        if not pause_listening(owner=transport):
            return
        if not owns_listener(transport):
            return
        if _transport_is_dead(transport):
            _release_wake_for_transport(transport)
            return
        # Multi-phrase engines report WHICH phrase fired and the profile it
        # belongs to, so one listener can wake any enrolled profile. Falls
        # back to the owner's configured phrase / no profile for
        # single-phrase engines.
        matched_phrase, matched_profile = get_last_match() or (phrase, "")
        logger.info("wake.detected: emitting to sid=%r (transport=%s, profile=%r)",
                    sid, type(transport).__name__, matched_profile)
        token = bind_transport(transport)
        try:
            _emit("wake.detected", sid, {
                "phrase": matched_phrase or phrase,
                "profile": matched_profile or None,
                "start_new_session": new_session,
            })
        finally:
            reset_transport(token)

    try:
        start_listening(
            _on_detect,
            owner=transport,
            config=cfg,
            external_audio=external_audio,
        )
    except WakeWordInUse:
        return _ok(rid, {
            "started": False,
            "reason": "owned",
            "owner_surface": existing_surface or None,
        })
    except Exception as e:
        logger.warning("wake.start(%s): failed to start listener: %s", surface, e)
        return _err(rid, 5026, str(e))
    global _wake_owner_transport, _wake_owner_surface
    with _wake_lock:
        _wake_owner_transport = transport
        _wake_owner_surface = surface
    frame = detector_frame_info()
    logger.info(
        "wake.start(%s): listening for %r (%s) capture=%s frame=%s",
        surface, reqs["phrase"], reqs["provider"], capture_mode, frame.get("frame_length"),
    )
    return _ok(rid, {
        "started": True,
        "phrase": reqs["phrase"],
        "provider": reqs["provider"],
        "owner_surface": surface,
        "enabled_persisted": enabled_persisted,
        "capture": capture_mode,
        "sample_rate": frame.get("sample_rate", 16000),
        "frame_length": frame.get("frame_length", 1280),
    })


@method("wake.stop")
def _(rid, params: dict) -> dict:
    """Stop this surface's listener.

    ``persist: true`` (explicit user gesture) also writes
    ``wake_word.enabled: false`` to config.yaml so auto-arm stays off in
    future sessions — the toggle is the config, not just the live listener.
    """
    transport = current_transport() or _stdio_transport
    stopped = _release_wake_for_transport(transport)
    disabled_persisted = False
    if bool(params.get("persist")):
        try:
            from tools.wake_word import load_wake_word_config

            currently_enabled = bool(load_wake_word_config().get("enabled"))
        except Exception:
            currently_enabled = True
        if currently_enabled:
            disabled_persisted = _persist_wake_enabled(False)
    return _ok(rid, {
        "stopped": stopped,
        "reason": None if stopped else "not_owner",
        "disabled_persisted": disabled_persisted,
    })


@method("wake.pause")
def _(rid, params: dict) -> dict:
    """Release the mic (e.g. while the desktop's browser captures audio)."""
    transport = current_transport() or _stdio_transport
    try:
        from tools.wake_word import pause_listening

        paused = pause_listening(owner=transport)
        logger.info("wake.pause: detector paused=%s", paused)
    except Exception as e:
        logger.debug("wake.pause failed: %s", e)
        paused = False
    return _ok(rid, {
        "paused": paused,
        "reason": None if paused else "not_owner",
    })


@method("wake.resume")
def _(rid, params: dict) -> dict:
    """Reclaim the mic after a pause; no-op if the listener isn't armed."""
    transport = current_transport() or _stdio_transport
    resumed = _wake_resume_if_owner(transport)
    logger.info("wake.resume: detector resumed=%s", resumed)
    return _ok(rid, {
        "resumed": resumed,
        "reason": None if resumed else "not_owner",
    })


@method("wake.status")
def _(rid, params: dict) -> dict:
    try:
        from tools.wake_word import (
            audio_is_silent,
            check_wake_word_requirements,
            detector_frame_info,
            get_input_device_status,
            is_listening,
            load_wake_word_config,
            owns_listener,
            resolve_capture_mode,
            silent_audio_hint,
        )
        cfg = load_wake_word_config()
        # Prefer client when the GUI asks (desktop remote re-arm / status).
        prefer_client = bool(params.get("client_capture")) or str(
            params.get("surface") or ""
        ).strip().lower() in ("gui", "desktop")
        probe_cfg = dict(cfg)
        probe_cfg["capture"] = resolve_capture_mode(cfg, prefer_client=prefer_client)
        reqs = check_wake_word_requirements(probe_cfg)
        transport = current_transport() or _stdio_transport
        owner, owner_surface = _wake_owner_snapshot()
        owned_by_caller = owns_listener(transport)
        listening = owned_by_caller and is_listening()
        silent = listening and audio_is_silent()
        input_device = get_input_device_status(cfg)
        hint = reqs.get("hint", "")
        if input_device.get("error") and not hint:
            hint = f"Wake-word input device could not be resolved: {input_device['error']}"
        if silent and not hint:
            hint = silent_audio_hint(input_device)
        # Effective capture: prefer the *armed* detector over config/auto.
        # With capture:auto the GUI arms client mode, but a bare status probe
        # would otherwise report "local" and the desktop would not reattach
        # the PCM feeder after wake.detected.
        frame = detector_frame_info()
        if owned_by_caller and frame.get("external_audio"):
            capture = "client"
        elif owned_by_caller and listening:
            capture = "local"
        else:
            capture = probe_cfg.get("capture") or reqs.get("capture") or str(
                cfg.get("capture") or "auto"
            )
        return _ok(rid, {
            "listening": listening,
            "owned_by_caller": owned_by_caller,
            "owner_surface": owner_surface if owner is not None else None,
            "phrase": reqs["phrase"],
            "provider": reqs["provider"],
            "configured_surface": str(cfg.get("surface") or "auto"),
            "input_device": input_device,
            "available": reqs["available"],
            "hint": hint,
            # Config truth: clients use this to re-arm after a voice turn
            # ("permanent on") without guessing from runtime listener state.
            "enabled": bool(cfg.get("enabled")),
            # Armed but deaf despite an open stream; see platform-specific hint.
            "audio_silent": silent,
            "capture": capture,
            "local_input_available": bool(reqs.get("local_input_available")),
            "sample_rate": frame.get("sample_rate", 16000),
            "frame_length": frame.get("frame_length", 1280),
        })
    except Exception as e:
        return _err(rid, 5026, str(e))


@method("wake.feed")
def _(rid, params: dict) -> dict:
    """Push client-captured PCM into the armed wake detector.

    Params:
      pcm: base64-encoded int16 mono little-endian samples (preferred), OR
      pcm_b64: alias of pcm
    Optional:
      sample_rate: must be 16000 (ignored if missing; mismatched rates rejected)

    Used when ``wake.start`` returned ``capture: "client"`` so remote backends
    without a microphone can still run openWakeWord on Mac/desktop audio.
    """
    transport = current_transport() or _stdio_transport
    raw_b64 = params.get("pcm") or params.get("pcm_b64") or ""
    if not isinstance(raw_b64, str) or not raw_b64.strip():
        return _err(rid, 4001, "wake.feed requires base64 pcm")
    try:
        import base64
        pcm = base64.b64decode(raw_b64, validate=False)
    except Exception as e:
        return _err(rid, 4001, f"invalid base64 pcm: {e}")
    if not pcm:
        return _ok(rid, {"fed": False, "reason": "empty"})
    # Soft size cap: 64000 bytes = 2s of 16 kHz int16 mono
    if len(pcm) > 64000:
        return _err(rid, 4001, "pcm frame too large")
    sr = params.get("sample_rate")
    if sr is not None and int(sr) not in (0, 16000):
        return _err(rid, 4001, "wake.feed only accepts 16 kHz PCM")
    try:
        from tools.wake_word import feed_audio
        ok = feed_audio(owner=transport, pcm_int16=pcm)
    except Exception as e:
        logger.debug("wake.feed failed: %s", e)
        return _err(rid, 5026, str(e))
    return _ok(rid, {"fed": bool(ok), "reason": None if ok else "not_owner"})


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

        stop_hint = ""
        if enabled:
            # Spoken-stop hint for the client to render on voice-mode start.
            # Sourced from voice.stop_phrases (custom phrases render
            # correctly); empty when the feature is disabled.
            try:
                from tools.voice_mode import voice_stop_hint

                stop_hint = voice_stop_hint()
            except Exception:
                stop_hint = ""

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

            # Clear TTS so it can be toggled independently after voice is off,
            # and silence any in-flight streaming speech.
            os.environ["HERMES_VOICE_TTS"] = "0"
            _tts_stream_stop(user_barge=False)

        return _ok(
            rid,
            {
                "enabled": enabled,
                "record_key": _voice_record_key(),
                "tts": _voice_tts_enabled(),
                "stop_hint": stop_hint,
            },
        )

    if action == "tts":
        if not _voice_mode_enabled():
            return _err(rid, 4014, "enable voice mode first: /voice on")
        new_value = not _voice_tts_enabled()
        # Runtime-only flag (CLI parity) — see voice.toggle on/off above.
        os.environ["HERMES_VOICE_TTS"] = "1" if new_value else "0"
        if not new_value:
            _tts_stream_stop(user_barge=False)
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
    wake_paused = False

    if action not in {"start", "stop"}:
        return _err(rid, 4019, f"unknown voice action: {action}")

    transport = current_transport() or _stdio_transport
    wake_owner, _surface = _wake_owner_snapshot()
    if wake_owner is not None and wake_owner is not transport:
        return _ok(rid, {"status": "busy", "reason": "wake_owned"})

    try:
        if action == "start":
            if not _voice_mode_enabled():
                return _err(rid, 4015, "voice mode is off — enable with /voice on")

            with _voice_sid_lock:
                global _voice_event_sid, _voice_wake_owner
                _voice_event_sid = params.get("session_id") or _voice_event_sid

            from hermes_cli.voice import start_continuous

            # Register the agent-busy probe so the shared voice wrapper can
            # hold the no-speech counter during long agent turns (item:
            # silence must not end the chat while the agent works). Safe to
            # re-register on every start; older wrappers without the setter
            # are tolerated.
            try:
                from hermes_cli.voice import set_voice_busy_probe

                set_voice_busy_probe(_any_session_running)
            except Exception:
                pass

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
            # Hand the mic to STT if the wake-word detector holds it; resume
            # once a terminal capture event fires (one-shot transcript / silence
            # limit), so wake-triggered and manual captures both coexist.
            try:
                from tools.wake_word import pause_listening

                wake_paused = pause_listening(owner=transport)
            except Exception:
                wake_paused = False
            if wake_paused:
                with _voice_sid_lock:
                    _voice_wake_owner = transport

            def _on_transcript(t):
                _voice_emit("voice.transcript", {"text": t})
                _resume_voice_wake()

            def _on_silent():
                _voice_emit("voice.transcript", {"no_speech_limit": True})
                _resume_voice_wake()

            def _on_stop_phrase(t):
                # Explicit user intent: the user SAID a bare stop phrase
                # ("stop"). End the voice chat exactly like a manual
                # /voice off — flip the mode flags and silence any live
                # streaming TTS — and emit a distinct signal so clients
                # (TUI, desktop) end the conversation instead of treating
                # it as a no-speech timeout. The continuous loop has
                # already halted before this callback fires.
                os.environ["HERMES_VOICE"] = "0"
                os.environ["HERMES_VOICE_TTS"] = "0"
                try:
                    _tts_stream_stop(user_barge=False)
                except Exception:
                    pass
                _voice_emit("voice.transcript", {"stop_phrase": True, "text": t})
                _resume_voice_wake()

            def _on_status(state):
                _voice_emit("voice.status", {"state": state})
                if state == "idle":
                    _resume_voice_wake()

            # voice.max_recording_seconds — hard cap on a single recording's
            # length. Same guard as the silence params: non-numeric / bool /
            # missing falls back to the documented 120 default, while an
            # explicit numeric value <= 0 disables the cap (0.0).
            max_rec = voice_cfg.get("max_recording_seconds")
            safe_max_rec = (
                (max_rec if max_rec > 0 else 0.0)
                if isinstance(max_rec, (int, float)) and not isinstance(max_rec, bool)
                else 120.0
            )
            started = start_continuous(
                on_transcript=_on_transcript,
                on_status=_on_status,
                on_silent_limit=_on_silent,
                silence_threshold=safe_threshold,
                silence_duration=safe_duration,
                auto_restart=False,
                max_recording_seconds=safe_max_rec,
                on_stop_phrase=_on_stop_phrase,
            )
            if started is False:
                _resume_voice_wake()
                return _ok(rid, {"status": "busy"})
            return _ok(rid, {"status": "recording"})

        # action == "stop"
        with _voice_sid_lock:
            _voice_event_sid = params.get("session_id") or _voice_event_sid

        from hermes_cli.voice import stop_continuous

        stop_continuous(force_transcribe=True)
        _resume_voice_wake()
        return _ok(rid, {"status": "stopped"})
    except ImportError:
        if wake_paused or action == "stop":
            _resume_voice_wake()
        return _err(
            rid, 5025, "voice module not available — install audio dependencies"
        )
    except Exception as e:
        if wake_paused or action == "stop":
            _resume_voice_wake()
        return _err(rid, 5025, str(e))


@method("voice.tts")
def _(rid, params: dict) -> dict:
    text = params.get("text", "")
    if not text:
        return _err(rid, 4020, "text required")
    try:
        # Import check up front so a missing voice module still returns the
        # documented 5026 instead of failing silently in the thread.
        import hermes_cli.voice  # noqa: F401

        threading.Thread(
            target=_speak_text_with_barge, args=(text,), daemon=True
        ).start()
        return _ok(rid, {"status": "speaking"})
    except ImportError:
        return _err(rid, 5026, "voice module not available")
    except Exception as e:
        return _err(rid, 5026, str(e))


# ── Methods: insights ────────────────────────────────────────────────


# ── Methods: rollback ────────────────────────────────────────────────


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
        ["Start a Chromium-family browser with remote debugging, then retry /browser connect:", command]
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
        elif _is_default_local_cdp(parsed):
            from hermes_cli.browser_connect import (
                discover_local_cdp_url,
                find_free_debug_port,
                launch_chrome_debug,
                local_port_in_use,
            )

            # Dual-stack discovery: when another app (an IDE debugger,
            # a dev server) squats the IPv4 loopback on the debug port,
            # a browser asked to bind that port comes up on [::1] only.
            # An IPv4-only probe misses it AND hangs against squatters
            # that accept TCP but never answer HTTP — the historic
            # cause of `browser.manage` RPC timeouts.
            discovered = discover_local_cdp_url(port, timeout=2.0)
            launch_port = port

            if discovered is None:
                if local_port_in_use(port):
                    launch_port = find_free_debug_port(port)
                    announce(
                        f"Port {port} is occupied by another application that "
                        "isn't a CDP browser (an IDE debugger or dev server may "
                        f"be using it) — launching a debug browser on port "
                        f"{launch_port} instead..."
                    )
                else:
                    announce(
                        "Chromium-family browser isn't running with remote debugging — attempting to launch..."
                    )

                launch = launch_chrome_debug(launch_port, system)
                if launch.launched:
                    # Bounded wait: the whole connect must finish well
                    # inside the client RPC timeout.
                    deadline = time.monotonic() + 10.0
                    while time.monotonic() < deadline:
                        discovered = discover_local_cdp_url(launch_port, timeout=1.0)
                        if discovered:
                            break
                        time.sleep(0.5)

                if discovered:
                    announce(
                        f"Chromium-family browser launched and listening on port {launch_port}"
                    )
                else:
                    hint = launch.hint
                    if hint:
                        announce(hint, level="error")
                    for line in _failure_messages(url, launch_port, system)[1:]:
                        announce(line, level="error")
                    return _ok(
                        rid, {"connected": False, "url": url, "messages": messages}
                    )
            else:
                announce(f"Chromium-family browser is already listening at {discovered}")

            # Adopt whatever loopback/port actually answered (may be
            # [::1] and/or an alternate port when 9222 was squatted).
            url = discovered
            parsed = urlparse(url)
        else:
            probes = _probe_urls(parsed)
            ok = any(_http_ok(p, timeout=2.0) for p in probes)
            if not ok:
                return _err(rid, 5031, f"could not reach browser CDP at {url}")

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




# ── Split @method handler modules (see method_ctx.py) ────────────────
# Imported at the end of this module so every global the handlers close
# over already exists; register() rebinds them onto this namespace.
from . import (  # noqa: E402
    methods_complete as _methods_complete,
    methods_config as _methods_config,
    methods_prompt as _methods_prompt,
    methods_session as _methods_session,
    methods_tools as _methods_tools,
)

for _m in (
    _methods_session,
    _methods_prompt,
    _methods_config,
    _methods_complete,
    _methods_tools,
):
    _m.register(sys.modules[__name__])
del _m
