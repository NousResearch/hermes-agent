import atexit
import concurrent.futures
import contextlib
import contextvars
import copy
import hashlib
import importlib
import inspect  # noqa: F401  (split modules)
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
from typing import Any, Callable, NamedTuple, Optional  # noqa: F401  (Callable: split modules)

# Several of these look unused here but are resolved BARE by split-module bodies rebound onto this
# namespace (method_ctx.bind_module) — deleting one breaks a handler at call time, not import time.
from agent.secret_scope import build_profile_secret_scope, reset_secret_scope, set_secret_scope  # noqa: F401
from hermes_constants import (
    get_hermes_home, get_hermes_home_override, profile_name_for_home,
    reset_hermes_home_override, set_hermes_home_override)
from hermes_cli.env_loader import load_hermes_dotenv
from utils import is_truthy_value
from tools.environments.local import hermes_subprocess_env
from agent.replay_cleanup import sanitize_replay_history
from agent.compaction_display import project_compaction_message_for_display  # noqa: F401
from agent.skill_commands import describe_skill_invocation  # noqa: F401
from agent.conversation_loop import INTERRUPT_WAITING_FOR_MODEL_PREFIX  # noqa: F401
from tui_gateway import git_probe
from tui_gateway._env import env_float, env_int
from tui_gateway.turn_marker import clear_turn_marker, read_turn_marker, record_turn_start  # noqa: F401
from tui_gateway.transport import (FanoutTransport, StdioTransport, Transport, bind_transport,
                                   current_transport, reset_transport)

logger = logging.getLogger(__name__)

_hermes_home = get_hermes_home()
load_hermes_dotenv(hermes_home=_hermes_home, project_env=Path(__file__).parent.parent / ".env")


# ── Panic logger: crashes otherwise leave no forensics (stdout is the JSON-RPC pipe, stderr doesn't
# flush before exit) → append every unhandled exception to the crash log + one-line stderr summary.
_CRASH_LOG = os.path.join(_hermes_home, "logs", "tui_gateway_crash.log")


def _record_crash(kind: str, exc_type, exc_value, exc_tb, *, thread_name: str | None = None) -> None:
    import traceback
    trace = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    suffix = f" · thread={thread_name}" if thread_name is not None else ""
    with contextlib.suppress(Exception):
        os.makedirs(os.path.dirname(_CRASH_LOG), exist_ok=True)
        with open(_CRASH_LOG, "a", encoding="utf-8") as f:
            f.write(f"\n=== {kind} · {time.strftime('%Y-%m-%d %H:%M:%S')}{suffix} ===\n")
            f.write(trace)
    # The first line is what the user sees (gateway.stderr Activity line); the rest stays in the log.
    first = str(exc_value).strip().splitlines()[0] if str(exc_value).strip() else exc_type.__name__
    who = f"thread {thread_name} raised " if thread_name is not None else ""
    print(f"[gateway-crash] {who}{exc_type.__name__}: {first}", file=sys.stderr, flush=True)


def _panic_hook(exc_type, exc_value, exc_tb):
    _record_crash("unhandled exception", exc_type, exc_value, exc_tb)
    sys.__excepthook__(exc_type, exc_value, exc_tb)  # chain so the process still terminates normally


sys.excepthook = _panic_hook
threading.excepthook = lambda args: _record_crash(
    "thread exception", args.exc_type, args.exc_value, args.exc_traceback, thread_name=args.thread.name)

with contextlib.suppress(Exception):
    from hermes_cli.banner import prefetch_update_check

    prefetch_update_check()

from tui_gateway.render import make_stream_renderer, render_diff, render_message  # noqa: F401

_sessions: dict[str, dict] = {}
_methods: dict[str, callable] = {}
_pending: dict[str, tuple[str, threading.Event]] = {}
_pending_prompt_payloads: dict[str, tuple[str, dict]] = {}
_answers: dict[str, str] = {}
# Batch clarify accumulators: rid → {"qids": [...], "answers": {qid: answer}}. Written by
# clarify.respond (per-question lock, update-in-place), read out by _block on resolution/timeout
# so locked answers survive the deadline.
_batch_clarify: dict[str, dict] = {}
_db = None
_db_error: str | None = None
_stdout_lock = threading.Lock()
_cfg_lock = threading.Lock()
# Shared profile UI metadata is updated concurrently by Desktop, mobile and pool RPCs; its
# compare/check/write transaction needs its own lock, not the unrelated config cache lock.
_profile_ui_meta_lock = threading.Lock()
_sessions_lock = threading.RLock()  # reentrant: _close_session_by_id may run under callers that already hold it
_prompt_lock = threading.Lock()
_cfg_cache: dict | None = None
_cfg_mtime: float | None = None
_cfg_path = None
_session_resume_lock = threading.Lock()
_SLASH_WORKER_TIMEOUT_S = max(5.0, env_float("HERMES_TUI_SLASH_TIMEOUT_S", 45.0))

def _ws_orphan_setting(env_var: str, cfg_key: str, default: float) -> float:
    """``dashboard.<cfg_key>`` seconds; the env var is an internal override that wins when set."""
    raw = os.environ.get(env_var)
    if raw is None or not str(raw).strip():
        raw = None
        with contextlib.suppress(Exception):
            from hermes_cli.config import load_config
            raw = (load_config().get("dashboard") or {}).get(cfg_key)
    with contextlib.suppress(ValueError, TypeError):
        return max(0.0, float(raw) if raw is not None else default)
    return max(0.0, default)


# When a WebSocket client (the dashboard's embedded-chat tab / desktop app) disconnects, ``tui_gateway.ws``
# detaches the transport but intentionally leaves the session parked so a quick reconnect can reattach it
# (see ws.py). That park is unbounded, though: a browser refresh spins up a brand-new ``session.create``
# (new sid + a fresh _SlashWorker via _deferred_build) and never reattaches the OLD sid, so the old
# session's slash-worker subprocess lingers forever — one leaked python process per refresh (#38591
# fallout). After this grace window, an orphaned WS session is interrupted if it is still running, then
# reaped once the normal turn-finalization path settles. Set to 0 to disable (park forever, pre-fix
# behaviour).
def _resolve_ws_orphan_reap_grace() -> float:
    """Grace before an orphaned WS session is interrupted/reaped (0 = park forever): ws.py parks a
    disconnected session for a quick reattach, but a browser refresh mints a NEW sid and never
    reattaches the old one (leaking its slash worker)."""
    return _ws_orphan_setting("HERMES_TUI_WS_ORPHAN_REAP_GRACE_S", "ws_orphan_reap_grace_s", 20.0)


_WS_ORPHAN_REAP_GRACE_S = _resolve_ws_orphan_reap_grace()
# A detached RUNNING turn is interrupted only once its activity clock (API waits, stream tokens, tool
# heartbeats) idled this long; 600s = the turn-liveness watchdog so "wedged" means the same. 0 disables.
_WS_ORPHAN_ACTIVITY_STALE_S = _ws_orphan_setting("HERMES_TUI_WS_ORPHAN_ACTIVITY_STALE_S", "ws_orphan_activity_stale_s", 600.0)
_WS_ORPHAN_INTERRUPT_REAP_POLL_S = 1.0
# Interrupt-then-reap poll budget: a turn that never settles (thread hung in a syscall) would
# reschedule the 1s poll forever; after this many polls, log loudly and force-reap.
# If an interrupted turn never settles (agent thread hung in a syscall, supervisor lost), each 1s poll would
# otherwise reschedule forever — trading the old leak-one-worker bug for leak-one-session-plus-timer-chain
# (review finding, PR #90373). After this many polls we log loudly and force-reap, mirroring the
# pre-existing stuck-`running` safety net's role of breaking the deadlock.
_WS_ORPHAN_INTERRUPT_REAP_MAX_POLLS = 60
_TURN_SETTLE_BEFORE_CLOSE_SECONDS = 5.0
_DETAIL_SECTION_NAMES = ("thinking", "tools", "subagents", "activity")
_DETAIL_MODES = frozenset({"hidden", "collapsed", "expanded"})

# ── Async RPC dispatch: slow handlers (seconds to minutes) would leave approval.respond and
# session.interrupt unread in the stdin pipe, so only THESE go to a small thread pool; everything else
# stays inline so fast-path ordering stays sane (write_json is _stdout_lock-guarded). Why each is slow:
# billing/subscription/usage = blocking portal (+Stripe) round-trips; complete.* = git ls-files /
# prompt_toolkit import + skill scan; model.options = credential pool + pricing + provider probe;
# pet.* = network or PNG decode (generate = several image-model round-trips); reload.mcp /
# mcp.servers.* = rediscovery, cold npx spawn, ~30s OAuth wait; profiles.* = skill-tree walk + state.db
# open; bot_relay.* = a FULL one-turn agent conversation (600s); setup.* / session.active_list =
# Desktop-polled and under GIL pressure block the WS read loop (false "needs setup", stalled
# interrupts); voice.*/wake.* = SYNCHRONOUS faster-whisper install (300s); session.workspace.move =
# git subprocess probes on an arbitrary (maybe slow) mount.
_LONG_HANDLERS = frozenset({
    "session.foreign.list", "session.foreign.preview", "session.foreign.import",
    "billing.state", "subscription.state", "subscription.preview", "subscription.change",
    "subscription.resume", "subscription.upgrade", "usage.bars", "session.usage", "billing.step_up",
    "browser.manage", "cli.exec", "complete.path", "complete.slash", "llm.oneshot", "model.options",
    "pet.cells", "pet.gallery", "pet.generate", "pet.hatch", "pet.info", "pet.select", "pet.thumb",
    "learning.frames", "plugins.manage", "reload.mcp", "mcp.servers.test", "mcp.servers.oauth.start",
    "process.list", "profiles.configure", "profiles.create", "profiles.describe", "profiles.get_asset",
    "profiles.list", "profiles.set_asset", "bot_relay.roster.sync", "bot_relay.outbox.drain",
    "bot_relay.deliver", "bot_relay.reply", "image.generate", "projects.discover_repos",
    "projects.record_repos", "projects.for_cwd", "projects.tree", "projects.project_sessions",
    "setup.runtime_check", "setup.status", "voice.toggle", "voice.record", "voice.tts", "wake.start",
    "wake.status", "session.active_list", "session.branch", "session.compress", "session.list",
    "session.resume", "session.workspace.move", "shell.exec", "skills.manage", "slash.exec",
    "command.dispatch",  # /goal draft invokes the auxiliary model; never block the RPC reader
})

_rpc_pool_workers = max(2, env_int("HERMES_TUI_RPC_POOL_WORKERS", 8))
_pool = concurrent.futures.ThreadPoolExecutor(max_workers=_rpc_pool_workers, thread_name_prefix="tui-rpc")
atexit.register(lambda: _pool.shutdown(wait=False, cancel_futures=True))

# Exact in-memory session record executing on the current turn thread — unlike a public session id,
# this object identity cannot be supplied by RPC.
_current_runtime_session_record: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "hermes_gateway_runtime_session_record", default=None)
# JSON-RPC method being dispatched on this thread/task. Diagnostic only (names WHICH client
# poll is looping in the 4001 warning); never authorization — the method string is client-supplied.
_current_rpc_method: contextvars.ContextVar[str] = contextvars.ContextVar("hermes_gateway_rpc_method", default="")

# Reserve real stdout for JSON-RPC only; redirect Python's stdout to stderr so stray print() from
# libraries/tools becomes harmless gateway.stderr instead of corrupting the JSON protocol.
_real_stdout = sys.stdout
sys.stdout = sys.stderr


class _DropTransport:
    """Detached WS sink: keep sessions resumable without writing stale frames."""

    def write(self, obj: dict) -> bool:
        return False

    def close(self) -> None:
        pass


# Module-level stdio transport — fallback sink when no transport is bound via contextvar or session.
# Stream resolved through a lambda so test monkey-patches of `_real_stdout` still land.
_stdio_transport = StdioTransport(lambda: _real_stdout, _stdout_lock)

# Detached websocket sessions use a drop sink instead of stdio: Desktop embeds the gateway in-process
# and captures stdout into logs, so stale frames must not fall through while a session awaits resume/reap.
_detached_ws_transport = _DropTransport()


def _prepend_tool_paths(env: dict[str, str]) -> dict[str, str]:
    """Prepend managed bin (first: managed-first policy for the Browser Use CLI), venv bin and
    ~/.local/bin to PATH so slash_worker children resolve Hermes-managed CLIs under the Desktop's minimal PATH."""
    managed_bin = ""
    with contextlib.suppress(Exception):
        managed_bin = str(Path(get_hermes_home()) / "bin")
    venv_bin = str(Path(sys.executable).parent)  # <venv>/bin (POSIX) or <venv>/Scripts (Windows)
    parts = [p for p in (managed_bin, venv_bin, str(Path.home() / ".local" / "bin"), env.get("PATH") or "") if p]
    env["PATH"] = os.pathsep.join(parts)
    return env


class _SlashWorker:
    """Persistent HermesCLI subprocess for slash commands."""

    def __init__(self, session_key: str, model: str, profile_home: str | None = None):
        self._lock = threading.Lock()
        self._seq = 0
        self.stderr_tail: list[str] = []
        self.stdout_queue: queue.Queue[dict | None] = queue.Queue()
        argv = [sys.executable, "-m", "tui_gateway.slash_worker", "--session-key", session_key] + (["--model", model] if model else [])
        self._closed = False
        from hermes_cli._subprocess_compat import windows_hide_flags
        # slash_worker runs the Hermes agent → needs provider credentials. Tier-1 secrets
        # (gateway/GitHub/infra) are still stripped (#29157). Global-remote / multi-profile sessions: the
        # worker must resolve config/skills/state against the session's profile home, not the gateway's
        # launch HERMES_HOME (#40677). The override goes through the build_subprocess_env factory's `extra`
        # (applied last, always wins) instead of a hand-rolled env["HERMES_HOME"] assignment.
        from tools.environments.local import build_subprocess_env

        # The worker runs the agent → needs provider credentials; tier-1 secrets (gateway/GitHub/
        # infra) are still stripped. Multi-profile sessions resolve against the session's profile
        # home via `extra` (applied last, always wins); the base already carries the HOME contract.
        env = _prepend_tool_paths(build_subprocess_env(
            hermes_subprocess_env(inherit_credentials=True), scrub_secrets=False,
            inherit_profile_home=False, extra={"HERMES_HOME": str(profile_home)} if profile_home else None))
        # Internal slash workers must import the same checkout as their parent.
        module_root = str(Path(__file__).resolve().parent.parent)
        env["PYTHONPATH"] = os.pathsep.join(
            part for part in (module_root, env.get("PYTHONPATH", "")) if part
        )
        # start_new_session: otherwise the worker inherits the gateway's pgid and mcp_tool's orphan
        # sweep, racing the spawn, killpg()s the TUI parent itself. errors="replace": bytes invalid
        # in the system locale (GBK Windows) must not raise UnicodeDecodeError in the drain threads.
        # Prepend the Hermes venv bin dir and the user-local bin dir to PATH so slash_worker child processes
        # can resolve Hermes-managed CLIs (browser-use, uvx) even when the parent gateway was launched with
        # a minimal PATH (e.g. by the Desktop/Dashboard app). See #83845.
        self.proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            encoding="utf-8", errors="replace", bufsize=1, cwd=os.getcwd(), env=env,
            creationflags=windows_hide_flags(), start_new_session=True)
        threading.Thread(target=self._drain_stdout, daemon=True).start()
        threading.Thread(target=self._drain_stderr, daemon=True).start()

    def _drain_stdout(self):
        for line in self.proc.stdout or []:
            with contextlib.suppress(json.JSONDecodeError):
                self.stdout_queue.put(json.loads(line))
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
                f"slash worker closed pipe{': ' + chr(10).join(self.stderr_tail[-8:]) if self.stderr_tail else ''}")

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
                    with contextlib.suppress(Exception):
                        proc.wait(timeout=1)  # reap the zombie SIGKILL leaves behind
        except Exception:
            with contextlib.suppress(Exception):
                proc.kill()
                proc.wait(timeout=1)
        finally:
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                with contextlib.suppress(Exception):
                    stream.close()


def _display_cfg() -> dict:
    """``display`` section of the behavioral config, or ``{}`` when absent/malformed."""
    display = _load_cfg().get("display")
    return display if isinstance(display, dict) else {}


def _load_busy_input_mode() -> str:
    raw = str(_display_cfg().get("busy_input_mode", "") or "").strip().lower()
    return raw if raw in {"queue", "steer", "interrupt"} else "interrupt"


def _load_interim_assistant_messages() -> bool:
    """``display.interim_assistant_messages`` (default true); when false no ``interim_assistant_callback``
    is installed, so tool-call/verify-on-stop interim text never becomes ``message.interim`` (gateway parity)."""
    return is_truthy_value(_display_cfg().get("interim_assistant_messages", True))


def _shutdown_sessions() -> None:
    # Durable-first: flush transcripts (bounded budget) BEFORE the slow teardown so a supervisor SIGKILL can't lose them.
    for step in (_flush_sessions_before_exit, _release_gateway_wake_owner):
        with contextlib.suppress(Exception):
            step()
    with _sessions_lock:
        sids = list(_sessions)
    for sid in sids:
        _close_session_by_id(sid, end_reason="tui_shutdown")


# Session reaping / flushing knobs (session_reaper.py). TTL is the last-resort net for disconnect paths that
# slip past the WS finally; hours-scale because last_active freezes during a long turn and on passive
# viewing — running/pending/starting/live-transport are hard exemptions.
_SESSION_TTL_S = max(0.0, env_float("HERMES_TUI_SESSION_TTL_S", float(6 * 3600)))
_REAPER_SCAN_S = 300.0
# Flush-on-kill budget + periodic incremental flush (piggybacks the reaper scan): a SIGTERM/SIGKILL
# mid-update loses at most one flush interval of session state.
_EXIT_FLUSH_BUDGET_S = max(0.0, env_float("HERMES_TUI_EXIT_FLUSH_BUDGET_S", 5.0))
_INCREMENTAL_FLUSH_INTERVAL_S = max(0.0, env_float("HERMES_TUI_SESSION_FLUSH_INTERVAL_S", _REAPER_SCAN_S))


def _start_idle_reaper() -> None:
    def _loop():
        while True:
            time.sleep(_REAPER_SCAN_S)
            with contextlib.suppress(Exception):
                _reap_idle_sessions()
    threading.Thread(target=_loop, daemon=True).start()


atexit.register(_shutdown_sessions)
_start_idle_reaper()


# ── Plumbing ──────────────────────────────────────────────────────────


def _get_db():
    global _db, _db_error
    if _db is None:
        from hermes_state_registry import acquire
        try:
            # Pin to import-time launch home (#102526). A bare acquire() follows
            # get_hermes_home(), which the desktop multiplex cron ticker temporarily
            # overrides per profile at startup — first touch inside a foreign window
            # permanently binds this process-wide handle to the wrong state.db.
            _db, _db_error = acquire(Path(_hermes_home) / "state.db"), None
        except Exception as exc:
            _db_error = str(exc)
            logger.warning("TUI session store unavailable — continuing without state.db features: %s", exc)
            return None
    return _db


def _transfer_db_to_agent(agent, db) -> bool:
    """Hand a DEDICATED profile ``state.db`` handle to *agent* (``AIAgent.close()`` then releases it).
    False = agent not holding *this* handle (build failed before ``_make_agent`` or got a different db):
    the caller still owns it. The shared launch handle never transfers — it outlives every agent, and
    ownership would let session.close() tear down the process-wide database."""
    with contextlib.suppress(Exception):
        if agent is None or db is None or getattr(agent, "_session_db", None) is not db:
            return False
        # Defense in depth (#91610): the shared launch handle must never transfer. Identity alone passes for
        # it — a launch-profile agent IS holding that handle — and ownership would make session.close() tear
        # down the process-wide database every other session shares. Refuse it explicitly even if a caller
        # invokes the transfer incorrectly; the caller's own `owns_db` gate is the first line of defense.
        if db is _get_db():
            logger.warning("Refused transfer of the shared launch SessionDB to a session "
                           "agent — the caller's owns_db gate should have prevented this.")
            return False
        agent._owns_session_db = True
        return True
    return False


def _open_profile_session_db(profile_home):
    """Open a DEDICATED handle on ``profile_home``'s ``state.db`` — FAIL CLOSED: a silent fallback to the
    launch ``state.db`` would bleed rows into the wrong profile's store exactly when the profile store is
    briefly unopenable (locked, mid-restore); callers let the error abort the build (→ ``agent_error``)."""
    from hermes_state_registry import acquire
    db_path = Path(profile_home) / "state.db"
    try:
        return acquire(db_path)
    except Exception as exc:
        raise RuntimeError(f"profile session store unavailable: {db_path}: {exc}") from exc


@contextlib.contextmanager
def _profile_db(params: dict | None = None):
    """Yield the SessionDB for ``params['profile']`` (None when unavailable); closes dedicated
    profile handles, leaves the launch-profile shared handle open."""
    profile = (params.get("profile") or "").strip() or None if isinstance(params, dict) else None
    # Launch/own profile → the shared _get_db() handle (left open); another profile → a dedicated
    # handle closed below (app-global remote mode). db is None when unavailable.
    if (profile_home := _profile_home(profile)) is None:
        db, owns = _get_db(), False
    else:
        try:
            from hermes_state_registry import acquire
            db, owns = acquire(Path(profile_home) / "state.db"), True
        except Exception as exc:
            logger.warning("TUI profile session store unavailable for %s: %s", profile, exc)
            db, owns = None, False
    try:
        yield db
    finally:
        if owns and db is not None:
            with contextlib.suppress(Exception):
                db.close()


def _canonical_profile_request(name: str) -> str:
    """Canonicalize profile basenames emitted by older session-info payloads.

    ``Path(default_home).name`` was historically sent as a profile id. Those basenames are
    installation details — unless a real named profile of that name exists (``hermes`` is a legal
    id), in which case it wins; other unknown names keep failing closed in ``_profile_home``.
    """
    if name.casefold() in {".hermes", "hermes"}:
        from hermes_cli import profiles as profiles_mod
        if not Path(profiles_mod.get_profile_dir(name)).is_dir():
            return "default"
    return name


def _response_profile_name(profile: str | None = None) -> str:
    """Profile name for session.* payloads: the requested real non-launch profile, else the launch one."""
    name = _canonical_profile_request((profile or "").strip())
    return name if name and _profile_home(name) is not None else _current_profile_name()


def _db_unavailable_error(rid, *, code: int):
    return _err(rid, code, f"state.db unavailable: {_db_error or 'state.db unavailable'}")


# ── Per-session profile scoping: the desktop's app-global remote mode points every profile at this
# backend, so calls carry ``profile`` → open that profile's db and bind its HERMES_HOME (ContextVar
# override) so config/skills/model/persistence resolve to it. Omitted/own profile → launch profile.
def _profile_home(profile: str | None) -> Path | None:
    """Resolve a named profile's home on THIS host, or None for the launch profile."""
    if not (name := _canonical_profile_request((profile or "").strip())):
        return None
    from hermes_cli import profiles as profiles_mod
    home = Path(profiles_mod.get_profile_dir(name))
    if not home.is_dir():
        raise FileNotFoundError(f"Profile '{name}' does not exist.")
    if home.resolve() == Path(_hermes_home).resolve():
        return None  # already the launch profile (no override needed)
    _served_profile_homes.add(home)  # the change watcher must stat every served sibling store too
    return home


# Profile homes served besides the launch home — the only extra stores the sessions watcher
# probes. Empty on single-profile installs, so their watcher stays byte-identical.
_served_profile_homes: set[Path] = set()


def _profile_scoped(handler):
    """Bind ``params['profile']``'s HERMES_HOME around a handler (pets/projects resolve via
    ``get_hermes_home``, so app-global remote mode still hits the focused profile). No-op for launch.

    Secondary-profile adapters are constructed inside ``_profile_runtime_scope`` (secret scope installed +
    multiplex active) — the same discriminator the Buzz/SimpleX adapters use for this bug class (#98738).
    Once multiplexing is active, launch-profile *turns* bind their own terminal scope
    (``prompt_turn._prepare_turn_input``) so they never depend on ambient ``os.environ``
    that a secondary context might have poisoned (#107422). Single-profile processes stay
    unscoped and keep legacy ``os.environ`` precedence.
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


# Placeholder ``terminal.cwd`` values (resolved to the home dir at runtime) — never an explicit
# workspace (mirrors gateway/run.py's config bridge).
_CWD_PLACEHOLDERS = {".", "auto", "cwd"}


def _configured_cwd_from_cfg(cfg: dict | None) -> str | None:
    """Absolute, existing ``terminal.cwd`` from a config mapping; None for placeholders/missing/invalid."""
    terminal_cfg = cfg.get("terminal") if isinstance(cfg, dict) else None
    raw = str(terminal_cfg.get("cwd") or "").strip() if isinstance(terminal_cfg, dict) else ""
    if not raw or raw in _CWD_PLACEHOLDERS:
        return None
    resolved = os.path.abspath(os.path.expanduser(raw))
    return resolved if os.path.isdir(resolved) else None


def _profile_configured_cwd(profile_home: Path | None) -> str | None:
    """A non-launch profile's ``terminal.cwd`` from ITS config.yaml (fail-open → None): the process-global
    ``TERMINAL_CWD`` belongs to the *launch* profile, and load_config() resolves the ACTIVE profile, so
    read the file directly through the _load_cfg pipeline.

    A new session bound to another profile must take its workspace from THAT profile's config, not the stale
    env var (issue #40334). Returns an absolute, existing directory, or None for placeholders / missing /
    invalid paths.
    """
    if profile_home is None:
        return None
    with contextlib.suppress(Exception):
        from hermes_cli.config import read_user_config_raw
        p = Path(profile_home) / "config.yaml"
        return _configured_cwd_from_cfg(_expand_cfg(_apply_managed(read_user_config_raw(p)))) if p.exists() else None
    return None


def _launch_configured_cwd() -> str | None:
    """Launch profile's ``terminal.cwd`` from config.yaml: the dashboard's in-memory gateway gets no bridged
    ``TERMINAL_CWD`` env (only the Node PTY child does), so a fresh /chat would otherwise start in ``os.getcwd()``."""
    with contextlib.suppress(Exception):
        return _configured_cwd_from_cfg(_load_cfg())
    return None


def _default_session_cwd() -> str:
    """Fallback cwd when no explicit / stored / profile cwd (mirrors :func:`_completion_cwd`'s tail so created
    AND resumed sessions land in the configured ``terminal.cwd``)."""
    return _launch_configured_cwd() or os.getenv("TERMINAL_CWD") or os.getcwd()


def write_json(obj: dict) -> bool:
    """Emit one JSON frame via the most-specific transport: (1) event frames with a session id → that
    session's transport (async events reach the owner even from threads with no contextvar binding);
    (2) the context-bound transport (:func:`dispatch`); (3) module stdio (tests monkey-patch ``_real_stdout``).
    Every event frame gets a per-session monotonic ``seq`` + replay-ring entry so ``session.events.since`` can resume."""
    from tui_gateway.event_replay import _stamp_event
    from tui_gateway.hosted_room_member_activity import project_room_member_activity
    _stamp_event(obj)
    if obj.get("method") == "event":
        # A room member's hidden session has no transport: its frames would die at stdio below.
        project_room_member_activity(obj, _sessions)
        params = obj.get("params")
        sid = ((params or {}).get("session_id")) if isinstance(params, dict) else ""
        if sid and (t := (_sessions.get(sid) or {}).get("transport")) is not None:
            return t.write(obj)
    return (current_transport() or _stdio_transport).write(obj)


def _event_frame(event: str, sid: str, payload: dict | None = None) -> dict:
    params: dict = {"type": event, "session_id": sid, **({"payload": payload} if payload is not None else {})}
    return {"jsonrpc": "2.0", "method": "event", "params": params}


def _emit(event: str, sid: str, payload: dict | None = None) -> bool:
    return write_json(_event_frame(event, sid, payload))


# Live WS peer transports (maintained by tui_gateway.ws): the only route for session-less background
# events, which write_json would otherwise drop on stdio (see _broadcast_global_event).
_live_transports: set[Transport] = set()
_live_transports_lock = threading.Lock()


def register_live_transport(transport: Transport | None) -> None:
    """Track a connected client transport for global broadcasts. Idempotent."""
    if transport is not None:
        with _live_transports_lock:
            _live_transports.add(transport)


def unregister_live_transport(transport: Transport | None) -> None:
    """Stop tracking a transport (call on disconnect). Idempotent."""
    with _live_transports_lock:
        _live_transports.discard(transport)


def _broadcast_global_event(event: str, payload: dict | None = None) -> None:
    """Fan a session-less, surface-global event (``skin.changed``) to every connected client — background
    emitters bottom out at stdio in ``write_json``'s ladder. No registered transports (stdio TUI, tests) → ``_emit``."""
    with _live_transports_lock:
        targets = list(_live_transports)
    if not targets:
        return _emit(event, "", payload)
    frame = _event_frame(event, "", payload)
    for transport in targets:
        try:
            transport.write(frame)
        except Exception:  # one wedged peer must not stall the rest; disconnect teardown unregisters it
            logger.debug("global-event broadcast write failed type=%s", event, exc_info=True)


def _approval_request_payload(data: dict | None) -> dict:
    """Build the client-safe representation of a pending approval."""
    payload = dict(data or {})
    if "choices" not in payload:
        choices = ["once"]
        if not payload.get("smart_denied") and payload.get("allow_session") is not False:
            choices.append("session")
            if payload.get("allow_permanent") is not False:
                choices.append("always")
        payload["choices"] = choices + ["deny"]
    if "command" in payload:
        from gateway.run import _redact_approval_command
        payload["command"] = _redact_approval_command(payload.get("command"))
    return payload


def _pending_clarify_request_payload(sid: str) -> dict | None:
    """Read-only snapshot of the clarify prompt still blocking a session: a client detached when
    `clarify.request` was emitted would otherwise never see it (agent parked until timeout). Same replay
    contract as `pending_approval`: the registry stays authoritative; `clarify.respond` resolves by request_id."""
    with _prompt_lock:
        for rid, (owner_sid, _ev) in _pending.items():
            event, prompt_payload = _pending_prompt_payloads.get(rid, ("", {}))
            if owner_sid != sid or event != "clarify.request":
                continue
            snapshot = dict(prompt_payload)
            # Batch clarify: replay the answers locked so far so a reconnecting client restores its ✓ state.
            if (batch := _batch_clarify.get(rid)) is not None and batch["answers"]:
                snapshot["answers"] = dict(batch["answers"])
            return snapshot
    if (session := _sessions.get(sid)) is not None:
        with session.get("history_lock", threading.Lock()):
            pending = session.get("_compute_host_pending_clarify")
            return dict(pending) if isinstance(pending, dict) else None
    return None


def _pending_approval_request_payload(session_key: str) -> dict | None:
    """Read the oldest unresolved approval in a session, if there is one."""
    try:
        from tools.approval import get_pending_gateway_approval
        approval = get_pending_gateway_approval(session_key)
    except Exception:
        logger.debug("failed to read pending approval for %s", session_key, exc_info=True)
        return None
    return _approval_request_payload(approval) if approval else None


def _emit_approval_request(sid: str, data: dict | None) -> None:
    """Emit ``approval.request`` with the command redacted: a credential-shaped value Tirith flagged would
    otherwise echo verbatim to the TUI (third egress alongside chat platforms and the SSE/API stream).

    Reuse the shared gateway See #48456, #50767.
    """
    _emit("approval.request", sid, _approval_request_payload(data))


def _status_update(sid: str, kind: str, text: str | None = None):
    if not (body := (text if text is not None else kind).strip()):
        return
    out_kind = kind if text is not None else "status"
    # Auto-compaction arrives as a generic "lifecycle" status; re-tag so drivers can show a
    # summarizing indicator — otherwise idle/preflight compaction looks like a hung turn.
    # See #97239.
    if out_kind == "lifecycle":
        from agent.conversation_compression import is_compaction_progress_status
        if is_compaction_progress_status(body):
            out_kind = "compacting"
    _emit("status.update", sid, {"kind": out_kind, "text": body})


def _image_meta(path: Path) -> dict:
    meta = {"name": path.name}
    with contextlib.suppress(Exception):
        from PIL import Image
        with Image.open(path) as img:
            width, height = (int(v) for v in img.size)
        # Rough attachment-display token estimate: 512px tiles at ~85 tokens/tile (cross-provider hint).
        tiles = max(1, (width + 511) // 512) * max(1, (height + 511) // 512) if width > 0 and height > 0 else 0
        meta.update(width=width, height=height, token_estimate=tiles * 85)
    return meta


def _ok(rid, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": rid, "result": result}


def _err(rid, code: int, msg: str, data=None) -> dict:
    error = {"code": code, "message": msg, **({"data": data} if data is not None else {})}
    return {"jsonrpc": "2.0", "id": rid, "error": error}


def method(name: str):
    def dec(fn):
        _methods[name] = fn
        return fn
    return dec


def _normalize_request(req: Any) -> tuple[Any, str, dict] | dict:
    """Validate a JSON-RPC request enough for safe local dispatch."""
    if not isinstance(req, dict):
        return _err(None, -32600, "invalid request: expected an object")
    rid, method = req.get("id"), req.get("method")
    if not isinstance(method, str) or not method:
        return _err(rid, -32600, "invalid request: method must be a non-empty string")
    params = req.get("params", {})
    if params is not None and not isinstance(params, dict):
        return _err(rid, -32602, "invalid params: expected an object")
    return rid, method, params if params is not None else {}


def handle_request(req: dict) -> dict | None:
    normalized = _normalize_request(req)
    if isinstance(normalized, dict):
        return normalized
    rid, method, params = normalized
    if not (fn := _methods.get(method)):
        return _err(rid, -32601, f"unknown method: {method}")
    token = _current_rpc_method.set(method)
    try:
        return fn(rid, params)
    finally:
        _current_rpc_method.reset(token)


def _current_session_steer_authority(session_id: str) -> tuple[Transport | None, dict | None]:
    """Unforgeable steering authority for this RPC context: the public session id is only a lookup
    hint; authority requires the ContextVar-bound transport to be ATTACHED to the live in-memory record
    under that id, so transport detachment, session removal or id reuse invalidates an earlier generation."""
    transport = current_transport()
    if transport is None or not session_id:
        return None, None
    expected_session = _current_runtime_session_record.get()
    with _sessions_lock:
        session = _sessions.get(session_id)
        # Membership, not slot identity: a mirrored session stores a FanoutTransport in the slot, so slot
        # identity alone would reject every client, the peer that commissioned the subagent included.
        # Authority is membership in the slot, which also grants it to any client attached to mirror the
        # session (see tests/tui_gateway/test_multi_client_fanout.py).
        if (session is None or (expected_session is not None and session is not expected_session)
                or not _session_transport_contains(session, transport)):
            return None, None
        return transport, session


def dispatch(req: dict, transport: Optional[Transport] = None) -> dict | None:
    """Route inbound RPCs — long handlers to the pool (returns None; the worker writes its own
    response via the bound transport), everything else inline (returns the response dict).
    *transport* pins every write of this request — events included — to that transport;
    omitted → the module stdio transport (``tui_gateway.entry`` behaviour)."""
    t = transport or _stdio_transport
    token = bind_transport(t)
    try:
        normalized = _normalize_request(req)
        if isinstance(normalized, dict):
            return normalized
        if normalized[1] not in _LONG_HANDLERS:
            return handle_request(req)
        ctx = contextvars.copy_context()  # the pool worker must see the bound transport
        if normalized[1] in _CONNECTOR_RPC_METHODS:
            ctx.run(_capture_connector_rpc_owner, normalized[2])

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
    return _err(rid, 5032, err) if (err := session.get("agent_error")) else None


# The deferred prompt path waits in short slices so a cancel is honored promptly and a slow
# build is reported to the client exactly once.
_AGENT_BUILD_WAIT_SLICE = 5.0
_AGENT_BUILD_SLOW_NOTICE_AFTER = 30.0
_AGENT_BUILD_SLOW_NOTICE_KEY = "agent-build-slow"


def _agent_build_wait_cap() -> float:
    """Seconds a submitted prompt waits for the deferred build before failing; ``agent.build_wait_timeout``
    overrides the 600s default (raise it for many slow MCP servers / high-latency provider metadata)."""
    with contextlib.suppress(Exception):
        raw = (_load_cfg().get("agent") or {}).get("build_wait_timeout")
        if raw is not None and float(raw) > 0:
            return float(raw)
    return 600.0


def _wait_agent_for_prompt(session: dict, rid: str, sid: str) -> dict | None:
    """Patient ``_wait_agent`` for deferred prompt.submit: the client already got ``streaming`` and the
    first message IS the turn, while a cold build routinely outlives the flat 30s ceiling (timing out
    silently discarded it). Waits in short slices (cancel honored promptly), notifies once (keyed) past
    ``_AGENT_BUILD_SLOW_NOTICE_AFTER``, fails only on a dead build thread or the bounded cap.
    Returns None on success OR cancel mid-wait (the caller's cancel branch owns that messaging).

    The flat 30s ``_wait_agent`` ceiling was a message-eating cliff (#63078): ``prompt.submit`` has already
    returned ``{"status": "streaming"}``, the user's first message IS the turn in flight, and the deferred
    agent build (MCP discovery with per-server retry backoff, synchronous model-metadata HTTP, skills
    scanning) routinely outlives 30 seconds on cold starts. On timeout the old path emitted an error EVENT
    and returned without ever calling ``_run_prompt_submit`` — the first message was permanently discarded
    while the build finished successfully in the background, leaving the blank first session.
    """
    ready = session.get("agent_ready")
    if ready is None:
        return None
    start, cap, notified_slow = time.monotonic(), _agent_build_wait_cap(), False
    while not ready.wait(timeout=_AGENT_BUILD_WAIT_SLICE):
        with session["history_lock"]:
            cancelled = session.get("_turn_cancel_requested") or not session.get("running")
        if cancelled:
            return None
        waited = time.monotonic() - start
        if waited >= cap:
            return _err(rid, 5032, f"agent initialization timed out after {int(waited)}s — "
                        "your message was not sent; retry once the session is ready")
        build_thread = session.get("_agent_build_thread")
        if build_thread is not None and not build_thread.is_alive() and not ready.is_set():
            # _build's finally guarantees ready.set(); dead thread + unset ready = died hard.
            return _err(rid, 5032, session.get("agent_error") or "agent initialization failed before completing")
        if not notified_slow and waited >= _AGENT_BUILD_SLOW_NOTICE_AFTER:
            notified_slow = True  # one keyed, replace-in-place notice (toast / status bar)
            _emit("notification.show", sid, {
                "text": "Still starting the agent (tool discovery / model setup) — your message will be sent as soon as it's ready.",
                "level": "info", "kind": "agent", "ttl_ms": None,
                "key": _AGENT_BUILD_SLOW_NOTICE_KEY, "id": _AGENT_BUILD_SLOW_NOTICE_KEY})
    if notified_slow:
        _emit("notification.clear", sid, {"key": _AGENT_BUILD_SLOW_NOTICE_KEY})
    return _err(rid, 5032, err) if (err := session.get("agent_error")) else None


def _bind_build_profile_scopes(profile_home: str) -> "_TurnScopes":
    """Bind a session profile's HERMES_HOME / secret / terminal scopes for an agent build. Fail-open per
    scope (the build must not die on a scope helper); the terminal installer itself fails closed (malformed
    policy → refusal scope) so _make_agent's terminal probing / cwd hints resolve the routed profile."""
    scopes = _TurnScopes()
    scopes.home = set_hermes_home_override(profile_home)
    with contextlib.suppress(Exception):
        scopes.secret = set_secret_scope(build_profile_secret_scope(Path(profile_home)))
    scopes.terminal = None
    with contextlib.suppress(Exception):
        from tools.terminal_scope import install_profile_terminal_scope
        scopes.terminal = install_profile_terminal_scope(Path(profile_home))
    return scopes


def _release_build_profile_scopes(scopes: "_TurnScopes") -> None:
    if scopes.home is not None:
        reset_hermes_home_override(scopes.home)
    if scopes.secret is not None:
        with contextlib.suppress(Exception):
            reset_secret_scope(scopes.secret)
    if scopes.terminal is not None:
        with contextlib.suppress(Exception):
            from tools.terminal_scope import reset_terminal_scope
            reset_terminal_scope(scopes.terminal)


def _deferred_build_agent_kwargs(current: dict, session_db) -> dict:
    """_make_agent kwargs for a deferred (first-prompt) build. A lazy-resumed (watch) session carries the
    stored conversation id so the upgrade continues it; a cold deferred resume restores the full persisted
    runtime identity (like the eager resume's overrides splat) so the build can't drop the provider. No
    stored runtime, or an unroutable provider → this session's picked model/effort/tier, else the default."""
    kw = {"session_db": session_db, "context_cwd_is_launch_artifact": _context_cwd_is_launch_artifact(current),
          "platform_override": _session_source(current)}
    if resume_sid := current.get("resume_session_id"):
        kw["session_id"] = resume_sid
    resume_overrides = current.get("resume_runtime_overrides")
    if isinstance(resume_overrides, dict) and resume_overrides and _overrides_have_routable_provider(resume_overrides):
        kw.update(resume_overrides)
    else:
        if override := current.get("model_override"):
            kw["model_override"] = override
        kw.update({k: v for k, v in (("reasoning_config_override", current.get("create_reasoning_override")),
                                     ("service_tier_override", current.get("create_service_tier_override")))
                   if v is not None})
    return kw


def _wire_session_agent(sid: str, key: str, agent) -> bool:
    """Post-build wiring; returns whether the approval notify got registered. Approval prompts route to the
    client; the self-improvement "💾 …" summary is emitted as review.summary (no print surface), honoring
    display.memory_notifications."""
    notify_registered = False
    with contextlib.suppress(Exception):
        from tools.approval import load_permanent_allowlist, register_gateway_notify
        register_gateway_notify(key, lambda data: _emit_approval_request(sid, data))
        notify_registered = True
        load_permanent_allowlist()
    _wire_callbacks(sid)
    with contextlib.suppress(Exception):  # bare agents without the attribute must not break startup
        agent.background_review_callback = lambda message, _sid=sid: _emit("review.summary", _sid, {"text": str(message)})
        agent.memory_notifications = _load_memory_notifications()
    return notify_registered


def _start_session_services(sid: str, key: str, current: dict) -> None:
    """Start the notification poller and fire the session-reset boundary hook."""
    with _sessions_lock:
        if (rec := _sessions.get(sid)) is not None:
            rec["_notif_stop"] = _start_notification_poller(sid, rec)
    _notify_session_boundary("on_session_reset", key, _session_source(current))


def _await_resume_history(sid: str, current: dict) -> bool:
    """Block on a cold resume's transcript hydration; False when this record was replaced meanwhile."""
    history_ready = current.get("resume_history_ready")
    if history_ready is None:
        return True
    if not history_ready.wait(timeout=300.0):
        raise TimeoutError("session history hydration timed out")
    if history_error := current.get("resume_history_error"):
        raise RuntimeError(str(history_error))
    with _sessions_lock:
        return _sessions.get(sid) is current


def _attach_built_agent(current: dict, agent) -> None:
    """Attach a freshly built agent to its live record (session DB row deferred to first run_conversation())."""
    # Bot Mode gate hint: the DB title lands post-first-turn but the system prompt builds at turn START.
    if _title_hint := str(current.get("pending_title") or "").strip():
        agent._session_title_hint = _title_hint
    current["agent"] = agent
    _session_todo_state(current)
    # Baseline for the per-turn config sync (profile home override still active).
    current["config_model_seen"] = _config_model_target()


def _announce_built_agent(sid: str, key: str, current: dict, agent) -> None:
    """Post-wiring tail of a build: credits seed, session services, session.info, late MCP catch-up."""
    # Credits notices at session OPEN (notice_callback already wired) so depletion warnings show at "ready".
    with contextlib.suppress(Exception):
        from agent.credits_tracker import seed_credits_at_session_start
        seed_credits_at_session_start(agent)
    _start_session_services(sid, key, current)
    info = _session_info(agent, current)
    if cfg_warn := _probe_config_health(_load_cfg()):
        info["config_warning"] = cfg_warn
        logger.warning(cfg_warn)
    _emit("session.info", sid, info)
    _schedule_mcp_late_refresh(sid, agent)  # servers slower than the bounded discovery wait land here


def _finish_agent_build(sid: str, key: str, current: dict, *, notify_registered: bool, scopes, session_db) -> None:
    """Release build scopes and settle ownership of the late notify registration + dedicated db handle."""
    if scopes is not None:
        _release_build_profile_scopes(scopes)
    # Reaped mid-build: _attach_worker closed the worker; only a late notify registration can still
    # leak (session.close unregistered before _build registered).
    with _sessions_lock:
        replaced = _sessions.get(sid) is not current
    if replaced and notify_registered:
        with contextlib.suppress(Exception):
            from tools.approval import unregister_gateway_notify
            unregister_gateway_notify(key)
    # Dedicated profile handle: hand it to the agent that will be torn down, else close it (build
    # failed, or `replaced`: this agent is discarded and _teardown_session never reaches it).
    if session_db is not None and not _transfer_db_to_agent(None if replaced else current.get("agent"), session_db):
        with contextlib.suppress(Exception):
            session_db.close()


def _start_agent_build(sid: str, session: dict) -> None:
    """Start building the real AIAgent for a TUI session, once. Deferred until the first prompt (or any
    command needing the agent) so the composer isn't blocked on tool discovery / model metadata;
    the ready/error event contract is unchanged."""
    ready = session.get("agent_ready")
    if ready is None:
        return
    # A lazy watch session spectating an in-flight child must stay lazy so the subagent live-mirror keeps
    # flowing (it bails once agent is set); incidental RPCs via _sess() would upgrade it mid-stream.
    if session.get("lazy") and _child_run_active(str(session.get("session_key") or "")):
        return
    with session.setdefault("agent_build_lock", threading.Lock()):
        if ready.is_set() or session.get("agent_build_started"):
            return
        session["agent_build_started"] = True
        session.pop("lazy", None)  # now genuinely mid-construction: restore the "still starting" eviction exemption
    key = session["session_key"]

    def _build() -> None:
        with _sessions_lock:
            current = _sessions.get(sid)
        if current is None:
            ready.set()
            return
        notify_registered, scopes, session_db = False, None, None
        profile_home = current.get("profile_home")
        try:
            if not _await_resume_history(sid, current):
                return
            tokens = _set_session_context(key)
            # Global-remote: bind the session profile's HERMES_HOME and hand the agent that profile's db —
            # DEDICATED and ours until _transfer_db_to_agent in the finally; FAIL CLOSED rather than
            # binding the launch DB and bleeding rows into the wrong state.db.
            if profile_home:
                scopes = _bind_build_profile_scopes(profile_home)
                session_db = _open_profile_session_db(profile_home)
            try:
                from tui_gateway.entry import ensure_mcp_discovery_started
                ensure_mcp_discovery_started()
            except Exception:
                logger.warning("MCP discovery startup failed", exc_info=True)
            try:
                agent = _make_agent(sid, key, **_deferred_build_agent_kwargs(current, session_db))
            finally:
                _clear_session_context(tokens)
            _attach_built_agent(current, agent)
            # No eager slash-worker pre-warm (slash.exec spawns on demand): each worker forks the full stdio
            # MCP fleet, and live-transport sessions are never reaped, so fleets would accumulate.
            notify_registered = _wire_session_agent(sid, key, agent)
            _announce_built_agent(sid, key, current, agent)
        except Exception as e:
            current["agent_error"] = str(e)
            _emit("error", sid, {"message": f"agent init failed: {e}"})
        finally:
            _finish_agent_build(
                sid, key, current, notify_registered=notify_registered, scopes=scopes, session_db=session_db)
            ready.set()

    build_thread = threading.Thread(target=_build, daemon=True)
    # _wait_agent_for_prompt handle: dead thread + unset agent_ready = died hard; waiters must not sit out the cap.
    session["_agent_build_thread"] = build_thread
    build_thread.start()


def _sess_nowait(params, rid):
    sid = params.get("session_id") or ""
    s = _sessions.get(sid)
    if s:
        return (s, None)
    # Stale runtime id (reaped/evicted/TTL): the client should session.resume the STORED id. Logged so
    # "message vanished" reads as "arrived and was rejected".
    logger.warning("session-scoped RPC rejected: method=%s session_id=%r not in memory "
                   # A session-scoped RPC hit a runtime id the gateway no longer holds (detached on WS
                   # disconnect and orphan-reaped, LRU-evicted, or torn down after an idle TTL). The client
                   # is expected to recover via session.resume on the STORED session id, but a plain
                   # stale-id send leaves no trace anywhere when the resume never fires — every RPC in this
                   # class returned a silent 4001. Log it so a "message vanished" report is diagnosable as
                   # "request arrived and was rejected" instead of "request never arrived" (see #90428).
                   "(detached/reaped runtime; client should resume the stored session), rid=%r",
                   _current_rpc_method.get() or "?", sid, rid)
    return (None, _err(rid, 4001, "session not found"))


def _sess(params, rid):
    s, err = _sess_building(params, rid)
    return (None, err) if err else (s, _wait_agent(s, rid))


def _sess_building(params, rid):
    """Resolve a session and warm its agent build WITHOUT waiting — for the attach RPCs (image/file/pdf,
    clipboard.paste, image.detach), which only touch creation-time fields and run inline on the socket
    reader thread, where waiting on a cold build stalled every RPC behind it ("text is instant, images hang")."""
    s, err = _sess_nowait(params, rid)
    if not err:
        _start_agent_build(params.get("session_id") or "", s)
    return (None, err) if err else (s, None)


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
    """Dashboard process-isolation config with read-site defaults: ``_load_cfg()`` does not
    deep-merge DEFAULT_CONFIG, so the Phase-0 defaults live here to stay in step with the REST editor."""
    root = _load_cfg() if cfg is None else cfg
    dash = root.get("dashboard") if isinstance(root, dict) else {}
    dash = dash if isinstance(dash, dict) else {}
    return {
        "turn_isolation": is_truthy_value(dash.get("turn_isolation"), default=_DASHBOARD_TURN_ISOLATION_DEFAULT),
        "compute_host_heartbeat_secs": _coerce_int_config_value(
            dash.get("compute_host_heartbeat_secs"), _DASHBOARD_COMPUTE_HOST_HEARTBEAT_SECS_DEFAULT, min_value=1),
        "compute_host_respawn_max": _coerce_int_config_value(
            dash.get("compute_host_respawn_max"), _DASHBOARD_COMPUTE_HOST_RESPAWN_MAX_DEFAULT, min_value=0),
    }


def _active_config_path() -> Path:
    """config.yaml of the per-session profile override (session.resume) when bound, else the launch home."""
    override = get_hermes_home_override()
    return Path(override if isinstance(override, str) and override else _hermes_home) / "config.yaml"


def _load_cfg_raw() -> dict:
    """The active profile's config.yaml EXACTLY as written — the write-back primitive, ONLY for
    read→mutate→``_save_cfg`` round-trips and raw inspection (defaults / managed overlay / ``${VAR}``
    expansion applied here would be persisted on the next save). Behavioral reads use :func:`_load_cfg`.
    Cache keyed on the resolved path so profiles don't clobber."""
    global _cfg_cache, _cfg_mtime, _cfg_path
    with contextlib.suppress(Exception):
        p = _active_config_path()
        mtime = p.stat().st_mtime if p.exists() else None
        with _cfg_lock:
            if _cfg_cache is not None and _cfg_mtime == mtime and _cfg_path == p:
                return copy.deepcopy(_cfg_cache)
        from hermes_cli.config import read_user_config_raw
        data = read_user_config_raw(p) if p.exists() else {}
        with _cfg_lock:  # cache the RAW config: _save_cfg writes _cfg_cache back to disk
            _cfg_cache, _cfg_mtime, _cfg_path = copy.deepcopy(data), mtime, p
        return data
    return {}


def _expand_cfg(cfg: dict) -> dict:
    """``${ENV_VAR}`` expansion (same as ``load_config_readonly``); non-dict results keep the input."""
    from hermes_cli.config import _expand_env_vars
    expanded = _expand_env_vars(cfg)
    return expanded if isinstance(expanded, dict) else cfg


def _load_cfg() -> dict:
    """Behavioral config read: raw user file + managed overlay + ${VAR} expansion — ``load_config_readonly``
    minus the DEFAULT_CONFIG merge (callers treat a missing key as "unset"; merging would break
    ``_load_cfg() == {}`` sentinels). Never pass the result to ``_save_cfg`` (use ``_load_cfg_raw()``)."""
    cfg = _apply_managed(_load_cfg_raw())
    with contextlib.suppress(Exception):
        cfg = _expand_cfg(cfg)
    return cfg


def _apply_managed(cfg: dict) -> dict:
    """Overlay administrator-pinned managed-scope values (read-side only, fail-open): this backend builds
    config independently of load_config, so managed skin/reasoning_effort/service_tier/provider_routing
    would otherwise be silently ignored."""
    with contextlib.suppress(Exception):
        from hermes_cli import managed_scope
        return managed_scope.apply_managed_overlay(cfg if isinstance(cfg, dict) else {})
    return cfg


def _save_cfg(cfg: dict):
    global _cfg_cache, _cfg_mtime, _cfg_path
    from utils import atomic_roundtrip_yaml_save
    path = _active_config_path()
    # Comment-, ordering- and Unicode-preserving write (a plain safe_dump clobbered hand-written configs);
    # fails closed on an unreadable existing config.yaml like atomic_config_write.
    atomic_roundtrip_yaml_save(path, cfg)
    with _cfg_lock:
        _cfg_cache, _cfg_path = copy.deepcopy(cfg), path
        try:
            _cfg_mtime = path.stat().st_mtime
        except Exception:
            _cfg_mtime = None


def _session_for_key(session_key: str) -> dict | None:
    """First live record with this session_key (snapshot under the lock: pool handlers mutate ``_sessions``)."""
    with _sessions_lock:
        return next((s for s in list(_sessions.values()) if s.get("session_key") == session_key), None)


def _set_session_context(session_key: str, cwd: str | None = None, *, ui_session_id: str = "") -> list:
    with contextlib.suppress(Exception):
        from gateway.session_context import set_session_vars
        sess = _session_for_key(session_key) if session_key else None
        # Ephemeral task ids aren't in `_sessions` (reverse-map → "" would clear the cwd override);
        # callers that know the workspace pass it.
        resolved = cwd if cwd is not None else (str(sess.get("cwd") or "") if sess is not None else "")
        source = _resolve_session_platform()
        browser_control_principal = browser_control_transport_family = ""
        # Live conversation id for subprocess HERMES_SESSION_ID: an explicitly empty contextvar is authoritative
        # (no os.environ fallback), so never leave it "" — agent's durable session_id, then session_key.
        session_id = session_key
        if sess is not None:
            source = _session_source(sess)
            session_id = getattr(sess.get("agent"), "session_id", None) or session_key
            identity = getattr(sess.get("transport"), "auth_identity", None)
            if _methods_browser_control._is_authenticated_identity(identity):
                browser_control_principal = _methods_browser_control._principal_digest(identity)
                browser_control_transport_family = _methods_browser_control._CLOUD_TRANSPORT_FAMILY
        return set_session_vars(
            session_key=session_key, session_id=session_id, source=source,
            browser_control_principal=browser_control_principal,
            browser_control_transport_family=browser_control_transport_family, cwd=resolved,
            ui_session_id=ui_session_id, cron_session="")
    return []


def _clear_session_context(tokens: list) -> None:
    if tokens:
        with contextlib.suppress(Exception):
            from gateway.session_context import clear_session_vars
            clear_session_vars(tokens)


def _enable_gateway_prompts() -> None:
    """Route approvals through gateway callbacks instead of CLI input()."""
    os.environ.update(HERMES_GATEWAY_SESSION="1", HERMES_EXEC_ASK="1", HERMES_INTERACTIVE="1")


# ── Blocking prompt factory ──────────────────────────────────────────


# Blocking bridges whose `*.respond` tolerates a late reply (allow_expired=True): on timeout the tool
# returns empty, but a slow renderer could still answer and hit a raw 4009 — `.expire` tears the card down.
_EXPIRING_REQUESTS = frozenset({
    "secret.request", "sudo.request", "vault.unlock.request", "vault.save_login.request", "vault.code.request", "clarify.request",
    "terminal.read.request",
    "preview.read.request", "preview.act.request", "window.read.request", "mcp.setup.request",
    "tour.request",
})


def _block(event: str, sid: str, payload: dict, timeout: float | None = 300, batch_qids: list[str] | None = None) -> str:
    rid = uuid.uuid4().hex[:8]
    ev = threading.Event()
    with _prompt_lock:
        _pending[rid] = (sid, ev)
        payload["request_id"] = rid
        _pending_prompt_payloads[rid] = (event, dict(payload))
        if batch_qids:
            # Multi-question clarify: per-question answers accumulate here (update-in-place until every
            # qid is locked); locked answers survive a timeout — see the batch read-out below.
            _batch_clarify[rid] = {"qids": list(batch_qids), "answers": {}}
    answered, batch_answers = False, None
    try:
        _emit(event, sid, payload)
        # Event semantics: None → wait forever (clarify_timeout <= 0; released only by a real answer or
        # session.interrupt), 0 → return immediately, > 0 → bounded wait.
        answered = ev.wait(timeout)
    finally:
        with _prompt_lock:
            _pending.pop(rid, None)
            _pending_prompt_payloads.pop(rid, None)
            answer_present = rid in _answers
            answer = _answers.pop(rid, "")
            if (batch_state := _batch_clarify.pop(rid, None)) is not None:
                batch_answers = dict(batch_state["answers"])
    expire = lambda: _emit(f"{event.removesuffix('.request')}.expire", sid, {"request_id": rid})
    if batch_qids is not None:
        # Cancel-all (respond with no question_id) resolves via _answers with "" — a plain cancel, not a partial result.
        if answer_present:
            return answer
        result: dict[str, object] = {"answers": batch_answers or {}}
        if not answered:
            # Deadline hit: keep what was locked, report the rest as absences (not skips), still expire live cards.
            result["timed_out"] = True
            expire()
        return json.dumps(result, ensure_ascii=False)
    if not answered and not answer_present and event in _EXPIRING_REQUESTS:
        expire()
    return answer


def _clarify_timeout_seconds() -> float | None:
    """Clarify wait for the TUI/desktop bridge from the canonical config (gateway/CLI parity); 300s
    historical default if config can't be read; ``<= 0`` = unlimited → None (never auto-skip)."""
    with contextlib.suppress(Exception):
        from tools.clarify_gateway import get_clarify_timeout
        timeout = get_clarify_timeout()
        return timeout if timeout > 0 else None
    return 300


def _clarify_block(sid: str, q, c, multi_select=False, questions=None) -> str:
    """Bridge the clarify tool callback onto _block. Single-question payloads keep their historical shape
    (``multi_select`` only when True — older renderers never see a new field); batch calls emit one
    clarify.request with only the wire fields (the tool-side entries carry result-assembly keys too)."""
    if questions:
        wire = [{"qid": e["qid"], "question": e["question"], "choices": e["choices"], "multi_select": bool(e["multi_select"])}
                for e in questions]
        return _block("clarify.request", sid, {"questions": wire}, timeout=_clarify_timeout_seconds(),
                      batch_qids=[e["qid"] for e in questions])
    payload = {"question": q, "choices": c, "multi_select": True} if multi_select else {"question": q, "choices": c}
    return _block("clarify.request", sid, payload, timeout=_clarify_timeout_seconds())


# A tour action is a DOM op the renderer answers in ms; the generous deadline exists only because a
# preview tour's first action injects the engine into a live page.
_TOUR_TIMEOUT_S = 45
# Until a session's client has proven it answers at all, hold it to a deadline a working renderer cannot miss.
_TOUR_PROBE_TIMEOUT_S = 10

_TOUR_BRIDGE_UNAVAILABLE = json.dumps({
    "success": False,
    "error": ("No Hermes Desktop window answered the tour request. The tour is driven by the desktop app's "
              "renderer, which updates separately from this backend, so an app build older than the tour tool "
              "has nothing listening. Update the Hermes Desktop app and start a new session. Do not retry tour "
              "in this session.")})


def _tour_request(sid: str, payload: dict) -> str:
    """Bridge the tour tool callback onto _block without paying for a client that cannot answer: against
    an older app nobody calls ``tour.respond`` and each action would block the full deadline, stacking per
    turn. First action per session gets the short probe deadline; unanswered → bridge marked unavailable
    for that session; once answered, the full deadline. Verdict lives on the record, so a new session re-probes.

    The renderer's ``tour.request`` handler ships in the desktop bundle, but the tool is offered by this
    backend — and the two update on different clocks. The model then does what the schema tells it to and
    tries the next action, so a single "give me a tour" turn stacks those waits (the timeouts reported
    against #89620).
    """
    session = _sessions.get(sid)
    if session is None:  # detached caller: throwaway record, plain bridge, unprobed ({} is falsy but a REAL record)
        session = {}
    state = session.get("tour_bridge")
    if state == "unanswered":
        return _TOUR_BRIDGE_UNAVAILABLE
    answer = _block("tour.request", sid, dict(payload),
                    timeout=_TOUR_TIMEOUT_S if state == "answered" else _TOUR_PROBE_TIMEOUT_S)
    if answer:
        session["tour_bridge"] = "answered"
    elif state != "answered":
        session["tour_bridge"] = "unanswered"
    return answer or _TOUR_BRIDGE_UNAVAILABLE


def _clear_pending(sid: str | None = None) -> None:
    """Release pending prompts with an empty answer: only *sid*'s (session.interrupt must not cancel other
    sessions' prompts), or every one when *sid* is None (shutdown)."""
    with _prompt_lock:
        for rid, (owner_sid, ev) in list(_pending.items()):
            if sid is None or owner_sid == sid:
                _answers[rid] = ""
                ev.set()


# ── Agent factory ────────────────────────────────────────────────────


def _env_model_seed() -> str:
    """The launch-scoped model seed (``hermes --tui -m``, hosted provisioning); "" when unset."""
    return (os.environ.get("HERMES_MODEL", "") or os.environ.get("HERMES_INFERENCE_MODEL", "")).strip()


def _resolve_model() -> str:
    if env := _env_model_seed():
        return env
    m = _load_cfg().get("model", "")
    if isinstance(m, dict):
        return str(m.get("default", "") or "").strip()
    if isinstance(m, str) and m:
        return m.strip()
    # No env seed / config preference: the cost-safe silent default (cache-only read), never an unpicked flagship.
    with contextlib.suppress(Exception):
        from hermes_cli.models import get_preferred_silent_default_model
        return get_preferred_silent_default_model()
    return "z-ai/glm-5.2"


def _resolve_session_platform() -> str:
    """``HERMES_DESKTOP=1`` without ``HERMES_DESKTOP_TERMINAL`` → "desktop" (chat panel; the agent then
    suggests TUI-only slash commands), else "tui" (embedded terminal pane or standalone ``hermes --tui``)."""
    desktop = is_truthy_value(os.environ.get("HERMES_DESKTOP"))
    return "desktop" if desktop and not is_truthy_value(os.environ.get("HERMES_DESKTOP_TERMINAL")) else "tui"


def _resolve_session_source(explicit: str | None) -> str:
    """Session DB ``source``: an explicit caller value (plugin session tagged ``"telegram"``) is never
    rewritten; only empty/None falls back to the env-resolved platform."""
    return explicit or _resolve_session_platform()


def _resolve_agent_platform(source: str | None) -> str:
    return _resolve_session_source(source)


def _config_model_target() -> tuple[str, str]:
    """(model, provider) selected by config.yaml — and ONLY config: the HERMES_MODEL launch seed fed into
    the per-turn sync would be replayed as a /model switch and persisted globally, or pin the session so
    dashboard/CLI model changes never reach an open chat. Empty model = "no preference" → no-op sync."""
    cfg_model = _load_cfg().get("model")
    if isinstance(cfg_model, dict):
        provider = str(cfg_model.get("provider") or "").strip()
        return str(cfg_model.get("default", "") or "").strip(), "" if provider.lower() == "auto" else provider
    return (cfg_model.strip() if isinstance(cfg_model, str) else ""), ""


def _resolve_startup_runtime() -> tuple[str, str | None]:
    model = _resolve_model()
    if explicit_provider := os.environ.get("HERMES_TUI_PROVIDER", "").strip():
        return model, explicit_provider
    if not (explicit_model := _env_model_seed()):
        return model, None
    with contextlib.suppress(Exception):
        from hermes_cli.models import detect_static_provider_for_model
        cfg = _load_cfg().get("model") or {}
        current_provider = ((str(cfg.get("provider") or "").strip().lower() if isinstance(cfg, dict) else "")
                            or os.environ.get("HERMES_INFERENCE_PROVIDER", "").strip().lower() or "auto")
        if detected := detect_static_provider_for_model(explicit_model, current_provider):
            provider, detected_model = detected
            return detected_model, provider
    return model, None


# Bare billing buckets are not routable provider identities; restoring one as a session provider override
# breaks resume. ``openrouter`` is deliberately NOT in this set (fully routable; agent_init's gate is a different set).
# (agent_init's fail-fast gate is a DIFFERENT set that also skips "openrouter" — there it means "default
# route, don't fail fast", not "unroutable".) ``openrouter`` is deliberately excluded here — it is a fully
# routable provider with its own API key and base_url. Sessions that used OpenRouter store
# ``billing_provider="openrouter"``; dropping it forces resume to the current global model (e.g. a custom
# endpoint), which is the wrong provider for the stored model. See #57588.
from hermes_state import _BARE_BILLING_PROVIDERS


def _is_routable_provider(provider: str) -> bool:
    with contextlib.suppress(Exception):
        from hermes_cli.runtime_provider import is_routable_provider
        return is_routable_provider(provider)
    return False


def _overrides_have_routable_provider(overrides: dict) -> bool:
    """Whether persisted runtime overrides still name a routable provider (renamed/removed → "Unknown
    provider" at agent init). Empty = NOT routable, so the caller falls back to the session's picked model."""
    provider = str(overrides.get("provider_override") or "").strip()
    if not provider:
        provider = str((overrides.get("model_override") or {}).get("provider") or "").strip()
    return bool(provider) and _is_routable_provider(provider)


def _parse_model_config(raw, *, quiet: bool = False) -> dict:
    """A row's ``model_config`` (dict or JSON text) as a dict; ``{}`` when absent/invalid."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            if not quiet:
                raise
            logger.debug("failed to parse stored session model_config", exc_info=True)
    return {}


def _stored_session_runtime_overrides(row: dict | None) -> dict:
    """Runtime fields persisted with a stored session (model column, ``billing_provider``, JSON ``model_config``):
    resume restores the model/provider/reasoning THAT chat used, not the global pick. Plugin-owned Bot-Mode
    sessions are exempt and rebuild from the member profile's CURRENT config (a stale provider pin left
    room bots "out of Nous credits" after a profile switch); signals: ``room_plumbing`` /
    ``follow_profile_config`` markers, the legacy hidden + "Group:" title, the title exactly "Bot Chat"."""
    if not row:
        return {}
    model_config = _parse_model_config(row.get("model_config"), quiet=True)
    _row_title = str(row.get("title") or "").strip()
    if (model_config.get("room_plumbing") or (row.get("hidden") and _row_title.startswith("Group:"))
            or model_config.get("follow_profile_config") or _row_title == "Bot Chat"):
        return {}
    overrides: dict = {}
    field = lambda k: str(model_config.get(k) or "").strip()
    model = str(row.get("model") or model_config.get("model") or "").strip()
    # ``billing_provider`` is only the billing bucket — for a custom endpoint the bare class "custom", which
    # agent_init treats as non-routable. Only restore an explicit provider; else resume uses the configured default.
    provider = field("provider")
    billing_provider = str(model_config.get("billing_provider") or row.get("billing_provider") or "").strip()
    if not provider and billing_provider.lower() not in _BARE_BILLING_PROVIDERS:
        provider = billing_provider
    base_url, api_mode, service_tier = field("base_url"), field("api_mode"), field("service_tier")
    reasoning_config = model_config.get("reasoning_config")
    # Heal a stale provider persisted by an older build (renamed/removed custom provider → "Unknown provider"):
    # recover ``custom:<name>`` from the stored base_url, then from the entry serving the model; else drop it.
    if provider and not _is_routable_provider(provider):
        healed = None
        try:
            from hermes_cli.runtime_provider import canonical_custom_identity
            healed = canonical_custom_identity(base_url=base_url or None, model=model or None)
        except Exception:
            logger.debug("custom provider identity recovery failed", exc_info=True)
        if healed:
            logger.info("healed stale session provider %r to %r", provider, healed)
            provider = healed
            base_url = ""  # the healed identity owns a registered endpoint; the snapshot URL must not override it
        else:
            provider = ""
    if model:
        # Same dict-shaped override live /model switches use, so a DB-restored session keeps custom endpoint
        # metadata across resume and rebuilds (/new). Raw api_key is never persisted/restored.
        overrides["model_override"] = {
            "model": model, "provider": provider or None, "base_url": base_url or None, "api_mode": api_mode or None}
    if provider:
        overrides["provider_override"] = provider
    if isinstance(reasoning_config, dict):
        overrides["reasoning_config_override"] = reasoning_config
    if service_tier:  # None = "inherit the profile" at _make_agent; "" = real override "no priority tier"
        overrides["service_tier_override"] = "" if service_tier.lower() == "normal" else service_tier
    return overrides


def _runtime_model_config(agent, existing: dict | None = None) -> dict:
    """Merge the agent's CURRENT runtime identity onto the row's persisted ``model_config``. Falsy agent
    attributes DELETE the key rather than skip the write: resume reads provider/endpoint from this JSON
    (model column written separately), so a stale provider would route the resumed chat to the wrong endpoint."""
    config = dict(existing or {})
    attr = lambda k: str(getattr(agent, k, "") or "").strip()
    model, provider, base_url = attr("model"), attr("provider"), attr("base_url")
    if provider.lower() == "custom":
        # ``agent.provider`` resolves every named custom entry to the literal "custom", losing the entry
        # identity (api_key is never persisted): recover ``custom:<name>`` from the endpoint URL.
        try:
            from hermes_cli.runtime_provider import canonical_custom_identity
            provider = canonical_custom_identity(base_url=base_url, model=model or None) or provider
        except Exception:
            logger.debug("custom provider identity lookup failed", exc_info=True)
    reasoning_config = getattr(agent, "reasoning_config", None)
    live = {
        "model": model, "provider": provider, "base_url": base_url, "api_mode": attr("api_mode"),
        # An empty dict is still a real (present) reasoning config.
        "reasoning_config": reasoning_config if isinstance(reasoning_config, dict) else None,
        "service_tier": getattr(agent, "service_tier", None),
    }
    for key, value in live.items():
        if value or isinstance(value, dict):
            config[key] = value
        else:
            config.pop(key, None)
    return config


def _persist_live_session_runtime(session: dict | None) -> None:
    """Persist active session runtime so future resumes restore the same footer."""
    live = _live_session_agent_db(session)
    if live is None:
        return
    agent, session_key, db = live
    try:
        row = db.get_session(session_key) or {}
        model_config = _runtime_model_config(agent, _parse_model_config(row.get("model_config")))
        if (tier_override := session.get("create_service_tier_override")) is not None:
            # agent.service_tier is None for explicit normal; without this the distinction is erased on every persist.
            model_config["service_tier"] = tier_override or "normal"
        model = str(getattr(agent, "model", "") or "").strip()
        if hasattr(db, "update_session_meta"):
            db.update_session_meta(session_key, json.dumps(model_config), model or None)
        elif model and hasattr(db, "update_session_model"):
            db.update_session_model(session_key, model)
    except Exception:
        logger.debug("failed to persist live session runtime", exc_info=True)


def _live_session_agent_db(session: dict | None):
    """(agent, session_key, db) for a live record, or None when any of them is missing."""
    agent = (session or {}).get("agent")
    session_key = str((session or {}).get("session_key") or "").strip()
    if agent is None or not session_key:
        return None
    db = getattr(agent, "_session_db", None) or _get_db()
    return None if db is None else (agent, session_key, db)


def _persist_live_session_system_prompt(session: dict | None) -> None:
    """Refresh the stored system prompt after a live runtime identity change."""
    live = _live_session_agent_db(session)
    if live is None or not hasattr(live[0], "_build_system_prompt") or not hasattr(live[2], "update_system_prompt"):
        return
    agent, session_key, db = live
    # Re-bind the session's profile HERMES_HOME (the build's finally reset it → root profile's SOUL.md/skills)
    # and session context (on the RPC thread _SESSION_CWD is unset → the process TERMINAL_CWD would persist).
    # Without this, _start_agent_build's finally block has already reset the override and the rebuilt prompt
    # silently uses the root profile's SOUL.md and skills. See issue #50233.
    profile_home = session.get("profile_home")
    home_token = set_hermes_home_override(profile_home) if profile_home else None
    session_tokens = _set_session_context(session_key, cwd=_session_cwd(session))
    try:
        prompt = agent._cached_system_prompt = agent._build_system_prompt(None)
        db.update_system_prompt(getattr(agent, "session_id", None) or session_key, prompt)
    except Exception:
        logger.warning("failed to persist live session system prompt for session %s", session_key, exc_info=True)
    finally:
        _clear_session_context(session_tokens)
        if home_token is not None:
            reset_hermes_home_override(home_token)


# Stable leading text of the model-switch marker (builder + dedup); only the newest marker is meaningful.
# Only the newest marker is meaningful (it names the *currently* active model); older ones are stale and
# would otherwise be re-sent to the provider on every turn (#65891).
_MODEL_SWITCH_MARKER_PREFIX = "[System: The active model for this chat has changed to "


def _is_model_switch_marker(entry: Any) -> bool:
    """Whether a history entry is a (self-replacing) model-switch marker."""
    if not isinstance(entry, dict):
        return False
    content = entry.get("content")
    return isinstance(content, str) and content.startswith(_MODEL_SWITCH_MARKER_PREFIX)


def _is_pivot_marker(entry: Any) -> bool:
    """A ``role=user`` pivot the gateway splices in mid-turn (model switch or personality change) — either can
    be the sole reason turn-start and current history differ. Only the model-switch marker is self-replacing."""
    return _is_model_switch_marker(entry) or (isinstance(entry, dict) and entry.get("display_kind") == "personality_switch")


def _append_model_switch_marker(session: dict | None, *, model: str, provider: str) -> None:
    """Record a real system-history pivot after a live model switch. Only the newest marker is kept (each
    switch strips prior ones, so N switches leave one marker, not N re-sent every API call; self-healing
    across resumes because the next switch collapses whatever a reload brought back).

    See #65891.
    """
    session_key = str((session or {}).get("session_key") or "").strip()
    if not session_key:
        return
    provider_part = f" via provider {provider}" if provider else ""
    marker = (
        f"{_MODEL_SWITCH_MARKER_PREFIX}{model}{provider_part}. From this point forward, use this runtime "
        "metadata when answering questions about what model/provider is active.]")
    # A user message, not system: strict OpenAI-compatible providers (vLLM, Qwen) reject non-leading system messages.
    # See #48338.
    entry = {"role": "user", "content": marker, "display_kind": "model_switch"}
    with session.get("history_lock") or contextlib.nullcontext():
        history = session.setdefault("history", [])
        history[:] = [h for h in history if not _is_model_switch_marker(h)]
        history.append(entry)
        session["history_version"] = int(session.get("history_version", 0)) + 1
    try:
        agent = session.get("agent")
        db = getattr(agent, "_session_db", None) if agent is not None else None
        if db is None:
            _ensure_session_db_row(session)
        with (contextlib.nullcontext(db) if db is not None else _session_db(session)) as db:
            if db is not None:
                db.append_message(session_id=session_key, role="user", content=marker, display_kind="model_switch")
    except Exception:
        logger.debug("failed to persist model switch marker", exc_info=True)


def _write_config_key(key_path: str, value):
    # Write-back round-trip: raw read is mandatory — saving the overlaid/expanded view would persist it.
    cfg = current = _load_cfg_raw()
    *parents, leaf = key_path.split(".")
    for key in parents:
        if not isinstance(current.get(key), dict):
            current[key] = {}
        current = current[key]
    current[leaf] = value
    _save_cfg(cfg)


_STATUSBAR_MODES = frozenset({"off", "top", "bottom"})
_APPROVAL_MODES = frozenset({"manual", "smart", "off"})

# Appearance switches the renderer owns but the AGENT must see (each gates a tool's `check_fn`). `config.set`
# answers 4002 for unlisted keys — a mirrored switch missing here writes nothing and its tool stays dark.
_DISPLAY_TOGGLE_KEYS = frozenset({"display.message_reactions", "display.in_app_tips", "display.in_app_tours"})
_BOOL_WORDS = {
    "1": True, "on": True, "true": True, "yes": True, "0": False, "off": False, "false": False, "no": False,
}


def _load_approval_mode() -> str:
    """Effective ``approvals.mode`` via the gate's own ``_get_approval_mode`` (a raw re-read missed the
    managed overlay and ``${VAR}`` expansion)."""
    from tools.approval_context import _get_approval_mode
    mode = _get_approval_mode()
    return mode if mode in _APPROVAL_MODES else "manual"


def _coerce_statusbar(raw) -> str:
    if raw is False:
        return "off"
    return s if isinstance(raw, str) and (s := raw.strip().lower()) in _STATUSBAR_MODES else "top"


_MOUSE_TRACKING_ALIASES = {
    "0": "off", "1": "all", "all": "all", "any": "all", "button": "buttons", "buttons": "buttons",
    "click": "buttons", "false": "off", "full": "all", "no": "off", "off": "off", "on": "all",
    "scroll": "wheel", "true": "all", "wheel": "wheel", "yes": "all",
}


def _display_mouse_tracking(display: dict) -> str:
    """display.mouse_tracking → ``off|wheel|buttons|all`` (bools: True → all, False → off); ``wheel`` (DEC
    1000+1006) is the tmux-friendly subset without hover events. Legacy ``tui_mouse`` only when ``mouse_tracking`` is absent."""
    if not isinstance(display, dict):
        return "all"
    raw = display.get("mouse_tracking") if "mouse_tracking" in display else display.get("tui_mouse", True)
    if isinstance(raw, str):
        return _MOUSE_TRACKING_ALIASES.get(raw.strip().lower(), "all")
    return "off" if raw is False or raw == 0 else "all"


def _load_reasoning_config(model: str = "") -> dict | None:
    """Via the shared chokepoint :func:`hermes_constants.resolve_reasoning_config` (per-model override >
    global ``agent.reasoning_effort``; YAML False = disabled).

    Closes #21256.
    """
    from hermes_constants import resolve_reasoning_config
    return resolve_reasoning_config(_load_cfg(), model)


_SERVICE_TIER_ALIASES = {"fast": "priority", "priority": "priority", "on": "priority", "auto": "auto", "cold": "cold"}


def _load_service_tier() -> str | None:
    raw = str((_load_cfg().get("agent") or {}).get("service_tier", "") or "").strip().lower()
    return _SERVICE_TIER_ALIASES.get(raw)


def _load_provider_routing() -> dict:
    """OpenRouter ``provider_routing`` prefs (gateway/CLI parity — without them OpenRouter picks an effectively random provider)."""
    with contextlib.suppress(Exception):
        return _load_cfg().get("provider_routing", {}) or {}
    return {}


def _load_show_reasoning() -> bool:
    # Fallback True — keep in sync with DEFAULT_CONFIG display.show_reasoning (no DEFAULT_CONFIG merge here).
    return bool(_display_cfg().get("show_reasoning", True))


def _load_memory_notifications() -> str:
    """``display.memory_notifications`` (``off`` / ``on`` default / ``verbose``; bool normalized) — gates the
    "💾 Self-improvement review" summary (gateway/CLI parity)."""
    raw = _display_cfg().get("memory_notifications")
    if isinstance(raw, bool):
        return "on" if raw else "off"
    return str(raw).lower() if raw else "on"


_TOOL_PROGRESS_MODES = frozenset({"off", "new", "all", "verbose"})


def _load_tool_progress_mode() -> str:
    env = os.environ.get("HERMES_TUI_TOOL_PROGRESS", "").strip().lower()
    if env in _TOOL_PROGRESS_MODES:
        return env
    raw = _display_cfg().get("tool_progress", "all")
    if isinstance(raw, bool):
        return "all" if raw else "off"
    mode = str(raw or "all").strip().lower()
    return mode if mode in _TOOL_PROGRESS_MODES else "all"


def _gui_surface_toolsets(platform: str) -> set[str]:
    """Toolsets that exist because of the CLIENT (both off ``_HERMES_CORE_TOOLS``; this is the one gate).
    ``platform`` is the SESSION's source, never a process env var: the desktop may drive a URL/cloud
    backend where ``HERMES_DESKTOP`` is unset (AGENTS.md surface rule)."""
    return {"project", "desktop_ui"} if platform == "desktop" else {"project"}


def _tui_notice(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


def _resolve_explicit_toolsets(explicit: list[str], validate_toolset) -> list[str] | None | bool:
    """Resolve a HERMES_TUI_TOOLSETS pin: list, None for "all", False when nothing was valid."""
    built_in = [name for name in explicit if validate_toolset(name)]
    unresolved = [name for name in explicit if name not in built_in]
    if unresolved:
        try:
            from hermes_cli.plugins import discover_plugins
            discover_plugins()
            plugin_valid = [name for name in unresolved if validate_toolset(name)]
        except Exception:
            plugin_valid = []
        built_in.extend(plugin_valid)
        unresolved = [name for name in unresolved if name not in plugin_valid]
    if any(name in {"all", "*"} for name in built_in):
        if ignored := [name for name in explicit if name not in {"all", "*"}]:
            _tui_notice(f"[tui] HERMES_TUI_TOOLSETS=all enables every toolset; ignoring additional entries: {', '.join(ignored)}")
        return None
    if not unresolved:
        return built_in
    try:  # (enabled, disabled) MCP server names from raw config; both empty on any failure
        from hermes_cli.config import read_raw_config
        from hermes_cli.tools_config import _parse_enabled_flag
        raw_cfg = read_raw_config()
        mcp_servers = raw_cfg.get("mcp_servers") if isinstance(raw_cfg.get("mcp_servers"), dict) else {}
        mcp_names, mcp_disabled = set(), set()
        for name, server_cfg in mcp_servers.items():
            if isinstance(server_cfg, dict):
                on = _parse_enabled_flag(server_cfg.get("enabled", True), default=True)
                (mcp_names if on else mcp_disabled).add(str(name))
    except Exception:
        mcp_names, mcp_disabled = set(), set()
    mcp_valid = [name for name in unresolved if name in mcp_names]
    disabled = [name for name in unresolved if name in mcp_disabled]
    unknown = [name for name in unresolved if name not in mcp_names and name not in mcp_disabled]
    if unknown:
        _tui_notice(f"[tui] ignoring unknown HERMES_TUI_TOOLSETS entries: {', '.join(unknown)}")
    if disabled:
        _tui_notice("[tui] ignoring disabled MCP servers in HERMES_TUI_TOOLSETS "
                    f"(set enabled: true in config.yaml to use): {', '.join(disabled)}")
    return (built_in + mcp_valid) or False


def _load_enabled_toolsets(platform: str | None = None) -> list[str] | None:
    """The agent's toolsets for this session (None = all): an explicit HERMES_TUI_TOOLSETS pin; else the
    coding posture (coding_context collapses to coding toolset + enabled MCP servers in a code workspace);
    else the configured CLI toolsets. Client-surface toolsets fold in here — only this surface can answer them."""
    session_platform = platform or _resolve_session_platform()
    explicit = [item.strip() for item in os.environ.get("HERMES_TUI_TOOLSETS", "").split(",") if item.strip()]
    fallback_notice = None
    if not explicit:
        with contextlib.suppress(Exception):
            from agent.coding_context import coding_selection
            selection = coding_selection(platform=session_platform)
            if selection is not None:
                return sorted({*selection, *_gui_surface_toolsets(session_platform)})
    try:
        from toolsets import validate_toolset
    except Exception:
        validate_toolset = None
    if explicit and validate_toolset is not None:
        resolved = _resolve_explicit_toolsets(explicit, validate_toolset)
        if resolved is not False:
            return resolved
        fallback_notice = "[tui] no valid HERMES_TUI_TOOLSETS entries; using configured CLI toolsets"
    try:
        from hermes_cli.config import load_config
        from hermes_cli.tools_config import _get_platform_tools
        cfg = load_config()
        # include_default_mcp_servers=True is the runtime variant (the agent must be able to call
        # default MCP servers); the config-editing variant would silently drop MCP tools from the TUI.
        # Passing ``False`` here is the config-editing variant — used when we need to persist a toolset list
        # without baking in implicit MCP defaults. Using the wrong variant at agent creation time makes MCP
        # tools silently missing from the TUI. See PR #3252 for the original design split.
        enabled = _get_platform_tools(cfg, "cli", include_default_mcp_servers=True)
        if fallback_notice is not None:
            _tui_notice(fallback_notice)
        return sorted(enabled | _gui_surface_toolsets(session_platform)) if enabled else None
    except Exception:
        if fallback_notice is not None:
            _tui_notice("[tui] no valid HERMES_TUI_TOOLSETS entries and configured CLI toolsets could not be loaded; enabling all toolsets")
        return None


def _session_tool_progress_mode(sid: str) -> str:
    return str(_sessions.get(sid, {}).get("tool_progress_mode", "all") or "all")


def _session_verbose(sid: str) -> bool:
    return _session_tool_progress_mode(sid) == "verbose"


def _tool_progress_enabled(sid: str) -> bool:
    return _session_tool_progress_mode(sid) != "off"


def _tool_lifecycle_required_for_ui(name: str) -> bool:
    """Interactive UI, not optional chrome: Desktop renders clarify / setup_mcp cards from the tool-call part."""
    return name in ("clarify", "setup_mcp")


def _restart_slash_worker(sid: str, session: dict):
    # Close the slash-worker subprocess as part of finalize itself, not just in the callers.
    # Defense-in-depth: every session-end path goes through _finalize_session (it's the single
    # ``_finalized``-guarded chokepoint), so folding worker cleanup in here means a future code path that
    # calls _finalize_session directly — without the surrounding _teardown_session / _shutdown_sessions
    # worker.close() — can't reintroduce the #38095 leak. Idempotent: _SlashWorker.close() is
    # poll()-guarded, so the explicit close() still in those callers is harmless.
    worker = session.get("slash_worker")
    if worker is None:
        return  # never spawned one; spawning here would fork the per-worker MCP fleet for nothing
    with contextlib.suppress(Exception):
        worker.close()
    try:
        new_worker = _SlashWorker(session["session_key"], getattr(session.get("agent"), "model", _resolve_model()),
                                  profile_home=session.get("profile_home"))
    except Exception:
        session["slash_worker"] = None
        return
    # Store-iff-still-mapped: the post-turn restart races a close_on_disconnect reap (a bare store would orphan it).
    _attach_worker(sid, session, new_worker)


def _get_usage(agent) -> dict:
    g = lambda k, fb=None: getattr(agent, k, 0) or (getattr(agent, fb, 0) if fb else 0)
    usage = {
        "model": getattr(agent, "model", "") or "",
        "input": g("session_input_tokens", "session_prompt_tokens"),
        "output": g("session_output_tokens", "session_completion_tokens"),
        "reasoning": g("session_reasoning_tokens"), "prompt": g("session_prompt_tokens"),
        "completion": g("session_completion_tokens"), "total": g("session_total_tokens"),
        "calls": g("session_api_calls"),
    }
    comp = getattr(agent, "context_compressor", None)
    if comp:
        from agent.context_breakdown import context_usage_fields
        usage.update(context_usage_fields(comp))
        usage["compressions"] = getattr(comp, "compression_count", 0) or 0
    # Cache-hit ratio + rolling latency/tps (CLI status-bar parity). Omitted, not fabricated, when there is no
    # data (Codex reports no latency; zero cache reads shows no hit% rather than an alarming 0).
    with contextlib.suppress(Exception):
        # Mirrors the classic CLI bar (cli.py _get_status_bar_snapshot / PR #98250): hit =
        # session_cache_read_tokens / session_prompt_tokens (CanonicalUsage.prompt_tokens = input +
        # cache_read + cache_write) latency/tps read the deque(maxlen=10) history maintained per API call in
        # agent/conversation_loop.py.
        _prompt_total = int(getattr(agent, "session_prompt_tokens", 0) or 0)
        _cache_read = int(getattr(agent, "session_cache_read_tokens", 0) or 0)
        if _prompt_total > 0 and _cache_read > 0:
            usage["cache_hit_pct"] = max(0, min(100, round(_cache_read / _prompt_total * 100)))
    with contextlib.suppress(Exception):  # a status-bar readout must never break usage reporting
        _lhist = list(getattr(agent, "_api_latency_history", []) or [])
        _ohist = list(getattr(agent, "_api_output_history", []) or [])
        if _n := min(len(_lhist), len(_ohist)):
            _total_lat = sum(_lhist[-_n:])
            _avg_vel = (sum(_ohist[-_n:]) / _total_lat) if _total_lat > 0 else None
            for _key, _val in (("avg_latency_s", _total_lat / _n), ("avg_tps", _avg_vel)):
                if _val is not None and _val == _val and 0 < _val < 1e6:  # guard NaN/negative/absurd provider timings
                    usage[_key] = round(float(_val), 1)
    # Live count of background/async subagents (CLI status bar ⛓ parity, same async_delegation registry).
    with contextlib.suppress(Exception):
        from tools.async_delegation import active_count as _async_active_count
        usage["active_subagents"] = _async_active_count()
    # Dev-only live credits-spent readout, gated on HERMES_DEV_CREDITS so the payload stays clean otherwise.
    if is_truthy_value(os.environ.get("HERMES_DEV_CREDITS")):
        with contextlib.suppress(Exception):
            spent = agent.get_credits_spent_micros()
            if spent is not None:
                usage["dev_credits_spent_micros"] = int(spent)
    return usage


def _probe_credentials(agent) -> str:
    """Warning or '' (``no-key-required`` is a valid sentinel for keyless custom providers)."""
    with contextlib.suppress(Exception):
        if not (getattr(agent, "api_key", "") or ""):
            provider = getattr(agent, "provider", "") or ""
            return f"No API key configured for provider '{provider}'. First message will fail."
    return ""


def _probe_config_health(cfg: dict) -> str:
    """Warn on bare YAML keys (`agent:` → None, silently dropping nested settings) and an unknown ``display.personality``."""
    if not isinstance(cfg, dict):
        return ""
    warnings: list[str] = []
    if null_keys := sorted(k for k, v in cfg.items() if v is None):
        keys = ", ".join(f"`{k}`" for k in null_keys)
        warnings.append(f"config.yaml has empty section(s): {keys}. Remove the line(s) or set them to `{{}}` — "
                        f"empty sections silently drop nested settings.")
    display_cfg = cfg.get("display")
    if isinstance(display_cfg, dict):
        personality = str(display_cfg.get("personality", "") or "").strip().lower()
        if personality and personality not in {"default", "none", "neutral"}:
            with contextlib.suppress(Exception):
                from hermes_cli.personality import available_personalities
                if personality not in available_personalities(cfg):
                    warnings.append(f"`display.personality: {personality}` does not match any built-in or "
                                    "`agent.personalities` entry; personality overlay will be skipped.")
    return " ".join(warnings).strip()


def _current_profile_name() -> str:
    with contextlib.suppress(Exception):
        from hermes_cli.profiles import get_active_profile_name
        return get_active_profile_name() or "default"
    return "default"


# Monotonic GUI<->backend contract version: the desktop refuses a backend reporting less (or none) with a
# one-click "update to align" prompt; bump whenever the desktop's backend contract changes. v2 file.attach;
# v3 approvals.mode RPCs + session.info reconciliation; v4 session.create fast=false = explicit normal tier;
# v5 ws_max_size >16 MiB file.attach frames; v6 plugins.manage rows carry the canonical registry key.
DESKTOP_BACKEND_CONTRACT = 6


def _session_usage_snapshot(session: dict | None) -> dict:
    sess = session or {}
    mirror_usage = _metadata_mirror(session).get("usage")
    if sess.get("agent") is not None and not (sess.get("_compute_host_active") and isinstance(mirror_usage, dict)):
        return _get_usage(sess["agent"])
    return dict(mirror_usage) if isinstance(mirror_usage, dict) else {}


def _project_info_for_cwd(cwd: str) -> dict | None:
    """The first-class Project owning ``cwd`` (per-profile projects.db) so TUI status, desktop status bar and
    ``/status`` name the workspace identically. Only explicit named projects resolve."""
    if not str(cwd or "").strip():
        return None
    try:
        from hermes_cli import projects_db as pdb
        with pdb.connect_closing() as conn:
            project = pdb.project_for_path(conn, cwd)
        return None if project is None else {
            "id": project.id, "slug": project.slug, "name": project.name, "primary_path": project.primary_path}
    except Exception:
        logger.debug("failed to resolve project for cwd", exc_info=True)
        return None


def _turn_started_at(session: dict | None) -> float | None:
    """Epoch seconds the current turn started, or None when idle (desktop keeps the elapsed timer across switches)."""
    inflight = (session or {}).get("inflight_turn")
    return float(inflight["started_at"]) if isinstance(inflight, dict) and inflight.get("started_at") else None


def _session_info(agent, session: dict | None = None) -> dict:
    if session is None:
        session = next((c for c in _sessions.values() if c.get("agent") is agent), None)
    sess = session or {}
    mirror = _metadata_mirror(session)
    cwd = _display_session_cwd(session)
    session_key = str(sess.get("session_key") or getattr(agent, "session_id", "") or "")
    personality = sess.get("personality", _display_cfg().get("personality") or "")
    reasoning_config = getattr(agent, "reasoning_config", None)
    reasoning_effort = ""
    if isinstance(reasoning_config, dict):
        # Disabled must differ from unset ("" = provider default) or the desktop loses "thinking off" after turn 1.
        reasoning_effort = "none" if reasoning_config.get("enabled") is False else str(reasoning_config.get("effort", "") or "")
    service_tier = getattr(agent, "service_tier", None) or mirror.get("service_tier") or ""
    # yolo ORs the same three sources check_all_command_guards() does (approvals.mode=off, the process
    # --yolo env, the per-session flag): the session flag alone would show "off" while config auto-approves.
    try:
        from tools.approval import _YOLO_MODE_FROZEN, is_session_yolo_enabled
        session_yolo = bool(is_session_yolo_enabled(session_key)) if session_key else False
        approval_mode = _load_approval_mode()
        yolo = bool(_YOLO_MODE_FROZEN) or session_yolo or approval_mode == "off"
    except Exception:
        yolo, approval_mode = False, "manual"
    # A switch queued mid-turn applies at next turn start (agent.model still reads the OLD model); report the
    # pending pick so the end-of-turn settle doesn't blip the UI back first.
    pending_switch = sess.get("pending_model_switch") or {}
    pending_model = str(pending_switch.get("display_model") or "").strip()
    pending_provider = str(pending_switch.get("display_provider") or "").strip()
    provider = mirror.get("provider", getattr(agent, "provider", ""))
    if provider == "custom" and "provider" not in mirror and agent is not None:
        # Clients reuse this identity for new chats without carrying the endpoint or key.
        # Broadcast/resume callers need not be bound to this session's profile.
        with _profile_build_scope(sess.get("profile_home") or _hermes_home):
            provider = _runtime_model_config(agent).get("provider", provider)
    info: dict = {
        "model": pending_model or mirror.get("model", getattr(agent, "model", "")),
        "provider": pending_provider or provider,
        "reasoning_effort": reasoning_effort, "service_tier": service_tier, "fast": service_tier == "priority",
        "yolo": yolo, "approval_mode": approval_mode,
        "tools": dict(mirror.get("tools") or {}) if isinstance(mirror.get("tools"), dict) else {},
        "skills": dict(mirror.get("skills") or {}) if isinstance(mirror.get("skills"), dict) else {},
        "cwd": cwd, "branch": git_probe.branch(cwd), "project": _project_info_for_cwd(cwd),
        "terminal_backend": _effective_terminal_backend(), "personality": str(personality or ""),
        "running": bool(sess.get("running")), "turn_started_at": _turn_started_at(session),
        "title": _session_live_title(sess, session_key) if session_key else "",
        "stored_session_id": session_key or "", "desktop_contract": DESKTOP_BACKEND_CONTRACT,
        "version": "", "release_date": "", "update_behind": None, "update_command": "",
        "usage": _session_usage_snapshot(session),
        "profile_name": profile_name_for_home(sess.get("profile_home")) or _current_profile_name(),
    }
    with contextlib.suppress(Exception):
        from hermes_cli import __version__, __release_date__
        info.update(version=__version__, release_date=__release_date__)
    live_agent = agent is not None and not sess.get("_compute_host_active")
    if live_agent:
        with contextlib.suppress(Exception):
            from model_tools import get_toolset_for_tool
            info["tools"] = {}
            for t in getattr(agent, "tools", []) or []:
                name = t["function"]["name"]
                info["tools"].setdefault(get_toolset_for_tool(name) or "other", []).append(name)
        with contextlib.suppress(Exception):
            from hermes_cli.banner import get_available_skills
            info["skills"] = get_available_skills()
    info["mcp_servers"] = []
    with contextlib.suppress(Exception):
        from tools.mcp_tool_discovery import get_mcp_status
        info["mcp_servers"] = get_mcp_status()
    with contextlib.suppress(Exception):
        info["system_prompt"] = (
            mirror.get("system_prompt") if "system_prompt" in mirror else getattr(agent, "_cached_system_prompt", "") or "")
    with contextlib.suppress(Exception):
        from hermes_cli.banner import get_update_result
        from hermes_cli.config import recommended_update_command
        # Two assignments (not one info.update): if recommended_update_command() raises,
        # update_behind must still be reported, as on main.
        info["update_behind"] = get_update_result(timeout=0.5)
        info["update_command"] = recommended_update_command()
    if live_agent and (warn := _probe_credentials(agent)):
        info["credential_warning"] = warn
    return info


def _tool_ctx(name: str, args: dict) -> str:
    """Argument preview for a tool row — never a phrased label: clients own their phrasing, so
    ``build_tool_label`` here would stutter ("Running Running …") and leak into the desktop's ``args.context``."""
    with contextlib.suppress(Exception):
        from agent.display import build_tool_preview
        return build_tool_preview(name, args, max_len=80) or ""
    return ""


def _emit_session_info_for_session(sid: str, session: dict) -> None:
    agent = session.get("agent")
    if agent is not None or _metadata_mirror(session):
        with contextlib.suppress(Exception):
            _emit("session.info", sid, _session_info(agent, session))


def broadcast_session_info() -> None:
    """Re-emit ``session.info`` to every live session — for approvals-config writers that bypass the
    self-re-emitting ``config.set`` RPC. Only THIS process; a spawned child gateway has its own ``_sessions``."""
    with _sessions_lock:
        sessions = list(_sessions.items())
    for sid, sess in sessions:
        _emit_session_info_for_session(sid, sess)


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
        payload: dict[str, object] = {
            "tool_id": tool_call_id,
            "name": name,
            "context": _tool_ctx(name, args),
        }
        # The desktop renders the expanded tool row (the `$` transcript) from
        # the args of the part, and `context` is an 80-char display preview.
        # tool.complete already ships full args to every client. When
        # tool.start ships them too, the expanded row is complete while the
        # tool runs, at the cost of one duplicate transient payload per call.
        if args:
            payload["args"] = args
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
        "clarify_callback": lambda q, c, multi_select=False, questions=None: (
            _clarify_block(sid, q, c, multi_select=multi_select, questions=questions)
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
        # drive_preview tool (desktop GUI): the renderer injects the interaction
        # engine into the preview pane's webview (or drives the pane's history)
        # and answers preview.act.respond with the outcome plus a refreshed
        # element inventory. Same budget as the preview read, which it ends
        # with — a click on a slow page pays for the settle and the re-scan.
        # annotate_preview rides this same callback: it resolves a target
        # through the same engine and differs only in the verb it sends, so it
        # needs a tool of its own but not a channel of its own.
        "drive_preview_callback": lambda payload: _block(
            "preview.act.request",
            sid,
            dict(payload),
            timeout=45,
        ),
        # read_window_below tool (desktop GUI): the renderer asks its main
        # process (which owns native window enumeration) which OS window sits
        # directly underneath the Hermes window, and answers
        # window.read.respond with the serialized metadata.
        "read_window_below_callback": lambda: _block(
            "window.read.request",
            sid,
            {},
            timeout=30,
        ),
        # setup_mcp tool (desktop GUI): the renderer shows an inline consent
        # card and walks the user through install/enable/OAuth via the REST
        # endpoints, then answers mcp.setup.respond with the JSON outcome.
        # Long timeout on purpose — the flow can include typing an API key or
        # a browser OAuth round-trip. Same lifecycle as clarify: on timeout
        # the tool returns "unanswered" and a late answer is tolerated.
        "setup_mcp_callback": lambda server, action, reason: _block(
            "mcp.setup.request",
            sid,
            {"server": server, "action": action, "reason": reason},
            timeout=600,
        ),
        # tour tool (desktop GUI): the renderer drives driver.js — highlighting
        # elements in the app's own DOM or injecting the engine into the
        # preview pane's webview — and answers tour.respond with the outcome
        # (did the selector match, which step is active).
        "tour_callback": lambda payload: _tour_request(sid, payload),
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

    _persist_session_cwd_and_schedule_git_meta(session, resolved)

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
    """Delegates to hermes_cli.personality (single owner of rendering)."""
    from hermes_cli.personality import render_personality_prompt

    return render_personality_prompt(value)


def _available_personalities(cfg: dict | None = None) -> dict:
    """Built-ins + user overrides, via hermes_cli.personality (single owner)."""
    from hermes_cli.personality import available_personalities

    if cfg is None:
        cfg = _load_cfg()
    return available_personalities(cfg)


def _validate_personality(value: str, cfg: dict | None = None) -> tuple[str, str]:
    """Resolve a requested personality against _available_personalities.

    Same contract as hermes_cli.personality.resolve_personality — (name,
    prompt) or ValueError — but resolves through the module-level
    _available_personalities so tests (and future gateway-side overrides)
    keep a single patch point.
    """
    from hermes_cli.personality import normalize_personality_name

    name = normalize_personality_name(value)
    if not name:
        return "", ""
    personalities = _available_personalities(cfg)
    if name not in personalities:
        names = ", ".join(f"`{n}`" for n in sorted(personalities))
        raise ValueError(
            f"Unknown personality: `{str(value).strip()}`.\n\nAvailable: `none`, {names}"
        )
    return name, _render_personality_prompt(personalities[name])


def _prompt_text(value) -> str:
    """Normalize config prompt values from YAML before handing them to AIAgent.

    Delegates to hermes_cli.personality (single owner).
    """
    from hermes_cli.personality import prompt_text

    return prompt_text(value)


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
        # Tagged like the model-switch marker (`_append_model_switch_marker`):
        # the marker rides as role=user so strict OpenAI-compatible providers
        # accept it mid-conversation, but `display_kind` keeps it out of the
        # `truncate_before_user_ordinal` addressing space. Untagged, it counts
        # as a real user turn on the gateway side while no client counts it, so
        # every later rewind resolves one turn too early and `replace_messages`
        # hard-deletes the difference (#82756).
        with session["history_lock"]:
            session["history"].append(
                {"role": "user", "content": marker, "display_kind": "personality_switch"}
            )
            session["history_version"] = int(session.get("history_version", 0)) + 1
        info = _session_info(agent)
        _emit("session.info", sid, info)
        return False, info
    return False, None


def _cfg_max_turns(
    cfg: dict,
    default: int,
    *,
    invalid_default: int | None = None,
) -> int:
    from hermes_cli.config import resolve_turn_limit as _resolve_turn_limit
    explicit_default = default if invalid_default is None else invalid_default
    # Env var override (highest priority)
    env_val = os.environ.get("HERMES_TUI_MAX_TURNS")
    if env_val:
        return _resolve_turn_limit(env_val, default=explicit_default)
    # Config file value — route through resolve_turn_limit so that
    # "none"/"unlimited"/0 are first-class spellings, not int() crashes.
    agent_cfg = cfg.get("agent") or {}
    raw = agent_cfg.get("max_turns")
    if raw is None:
        raw = cfg.get("max_turns")
    if raw is not None:
        return _resolve_turn_limit(raw, default=explicit_default)
    return default


def _sync_agent_turn_limit_with_config(session: dict) -> None:
    """Adopt config max-turn edits for an already-built Desktop/TUI agent.

    ``HERMES_TUI_MAX_TURNS`` remains the highest-priority input. An absent
    value preserves the active limit (important during a transient empty
    config read), while a malformed explicit value falls back to the same
    500-turn baseline used to build an interactive gateway agent.
    """
    agent = session.get("agent")
    if agent is None:
        return
    current = int(getattr(agent, "max_iterations", 500) or 500)
    agent.max_iterations = _cfg_max_turns(
        _load_cfg(),
        current,
        invalid_default=500,
    )


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
    """Refresh a session's tool snapshot when MCP discovery lands late (``_make_agent`` waits only a bounded
    ``mcp_discovery_timeout``, so a slow server's tools would be missing all session): a daemon joins discovery,
    rebuilds like ``/reload-mcp`` and re-emits ``session.info``. Only pre-first-turn (nothing cached to
    invalidate); afterwards late tools need an explicit, consent-gated ``/reload-mcp``."""
    try:
        from tui_gateway.entry import mcp_discovery_in_flight, join_mcp_discovery
    except Exception:
        return
    if not mcp_discovery_in_flight():
        return

    def _wait_then_refresh() -> None:
        if not join_mcp_discovery(timeout=30.0):  # a server still not connected after this is genuinely slow/dead
            return
        with _sessions_lock:
            session = _sessions.get(sid)
            if session is None or session.get("agent") is not agent:
                return  # closed/reset while we waited
            if int(getattr(agent, "_user_turn_count", 0) or 0) > 0 or int(getattr(agent, "_api_call_count", 0) or 0) > 0:
                return  # conversation started: a rebuild would invalidate the cached prompt prefix
            try:
                from tools.mcp_tool_agent import refresh_agent_mcp_tools
                added = refresh_agent_mcp_tools(agent, quiet_mode=True)
            except Exception as exc:
                logger.warning("Late MCP refresh: tool snapshot rebuild failed for %s: %s", sid, exc)
                return
            if not added:
                return  # discovery added nothing → don't churn the client
            info = _session_info(agent, session)
        _emit("session.info", sid, info)  # outside the lock — write_json must not block under _sessions_lock
    threading.Thread(target=_wait_then_refresh, name=f"tui-mcp-late-refresh-{sid}", daemon=True).start()


class _RuntimeFallbackResolution(NamedTuple):
    runtime: dict
    selected_model: str | None
    used_fallback: bool


def _resolve_runtime_with_fallback(resolve_kwargs: dict | None = None) -> _RuntimeFallbackResolution:
    """Resolve the primary runtime or one complete provider/model fallback. Provider-only fallback entries
    are skipped so the unavailable primary model can never leak into a different runtime."""
    from hermes_cli.auth import AuthError
    from hermes_cli.runtime_provider import resolve_runtime_provider
    try:
        return _RuntimeFallbackResolution(resolve_runtime_provider(**(resolve_kwargs or {})), None, False)
    except AuthError as primary_exc:
        for entry in _load_fallback_model() or []:
            fb_provider = str(entry.get("provider") or "").strip() if isinstance(entry, dict) else ""
            fb_model = str(entry.get("model") or "").strip() if isinstance(entry, dict) else ""
            if not fb_provider or not fb_model:
                continue
            try:
                from hermes_cli.fallback_config import resolve_entry_api_key
                fb_kwargs: dict = {"requested": fb_provider, "target_model": fb_model,
                                   **({"explicit_base_url": entry["base_url"]} if entry.get("base_url") else {})}
                if fb_api_key := resolve_entry_api_key(entry):
                    fb_kwargs["explicit_api_key"] = fb_api_key
                runtime = resolve_runtime_provider(**fb_kwargs)
                logging.getLogger(__name__).warning(
                    "Primary auth failed (%s), falling back to %s model %s", primary_exc, fb_provider, fb_model)
                return _RuntimeFallbackResolution(runtime, fb_model, True)
            except Exception:
                continue
        raise


def _resolve_agent_model_runtime(model_override, provider_override) -> tuple[str, dict]:
    """(model, runtime) for a new agent; a per-session override (/model switch or a resumed row's persisted
    runtime) wins over global config/env. Older rows stored the resolved provider "custom" (no named entry
    matches) — recover the identity from the persisted base_url or the rebuild fails "No LLM provider
    configured". Persisted base_url/api_key/api_mode are honored only for the original runtime, never a fallback."""
    if isinstance(model_override, dict) and model_override.get("model"):
        model = str(model_override.get("model") or "")
        requested_provider = model_override.get("provider") or provider_override or None
        override_base_url = model_override.get("base_url")
        resolve_kwargs = {}
        if str(requested_provider or "").strip().lower() == "custom":
            from hermes_cli.runtime_provider import canonical_custom_identity
            if recovered := canonical_custom_identity(base_url=override_base_url or None, model=model or None):
                requested_provider = recovered
            if override_base_url:
                # Failing identity recovery, still hand base_url to the direct-alias branch so pool/env credentials resolve.
                resolve_kwargs["explicit_base_url"] = override_base_url
        resolve_kwargs.update(requested=requested_provider, target_model=model or None)
        overrides = {k: model_override.get(k) for k in ("base_url", "api_key", "api_mode")}
    else:
        model, requested_provider = _resolve_startup_runtime()
        if isinstance(model_override, str) and model_override:
            model = model_override
        if provider_override:
            requested_provider = provider_override
        resolve_kwargs = {"requested": requested_provider, "target_model": model or None}
        overrides = {}
    resolution = _resolve_runtime_with_fallback(resolve_kwargs)
    if resolution.used_fallback:
        if not resolution.selected_model:
            raise RuntimeError("Auth fallback resolved without a model")
        return resolution.selected_model, resolution.runtime
    resolution.runtime.update({k: v for k, v in overrides.items() if v})
    return model, resolution.runtime


def _startup_system_prompt(cfg: dict, task_id: str) -> str:
    """Config ephemeral system prompt + HERMES_TUI_SKILLS preload block. Hard-fails only when EVERY requested
    skill is missing (cli.py parity): a typo'd name must not auto-block the Kanban task."""
    from hermes_cli.config import resolve_ephemeral_system_prompt_from_config
    system_prompt = resolve_ephemeral_system_prompt_from_config(cfg)
    startup_skills = _parse_tui_skills_env()
    if not startup_skills:
        return system_prompt
    from agent.skill_commands import build_preloaded_skills_prompt
    skills_prompt, loaded_skills, missing_skills = build_preloaded_skills_prompt(startup_skills, task_id=task_id)
    if missing_skills:
        missing_display = ", ".join(missing_skills)
        if not loaded_skills:
            raise ValueError(f"Unknown skill(s): {missing_display}")
        logger.warning("Unknown skill(s) requested, skipping: %s. Continuing with: %s. "
                       "List available skills with `hermes skills list`.", missing_display, ", ".join(loaded_skills))
    if skills_prompt:
        system_prompt = "\n\n".join(part for part in (system_prompt, skills_prompt) if part).strip()
    return system_prompt


def _make_agent(
    sid: str, key: str, session_id: str | None = None, session_db=None,
    model_override: dict | str | None = None, provider_override: str | None = None,
    reasoning_config_override: dict | None = None, service_tier_override: str | None = None,
    platform_override: str | None = None, context_cwd_is_launch_artifact: bool | None = None):
    # AC-4 test seam: dead unless armed by the isolated certify harness.
    from tui_gateway.synthetic_turn import maybe_build_synthetic_agent
    synthetic = maybe_build_synthetic_agent(session_id or key, model_override)
    if synthetic is not None:
        return synthetic
    from run_agent import AIAgent
    # MCP discovery runs in a daemon thread (a dead server can't freeze the shell); the agent snapshots its tool
    # list once, so briefly wait for in-flight discovery. Dashboard /api/ws uses mcp_startup; TUI stdio uses entry.
    for _mod in ("hermes_cli.mcp_startup", "tui_gateway.entry"):
        with contextlib.suppress(Exception):
            importlib.import_module(_mod).wait_for_mcp_discovery()
    cfg = _load_cfg()
    # Load hooks alongside the same profile config used to construct this agent.
    from agent.shell_hooks import register_from_config
    register_from_config(cfg)
    system_prompt = _startup_system_prompt(cfg, session_id or key)
    model, runtime = _resolve_agent_model_runtime(model_override, provider_override)
    _pr = _load_provider_routing()
    platform = _resolve_agent_platform(platform_override)
    ignore_rules = is_truthy_value(os.environ.get("HERMES_IGNORE_RULES"))
    agent = AIAgent(
        model=model, max_iterations=_cfg_max_turns(cfg, 500), provider=runtime.get("provider"),
        base_url=runtime.get("base_url"), api_key=runtime.get("api_key"), api_mode=runtime.get("api_mode"),
        acp_command=runtime.get("command"), acp_args=runtime.get("args"),
        credential_pool=runtime.get("credential_pool"), quiet_mode=True,
        verbose_logging=False,  # DEBUG agent logging; independent of tool_progress_mode
        reasoning_config=(
            reasoning_config_override if reasoning_config_override is not None else _load_reasoning_config(str(model or ""))),
        service_tier=service_tier_override if service_tier_override is not None else _load_service_tier(),
        enabled_toolsets=_load_enabled_toolsets(platform),
        # OpenRouter provider_routing prefs (gateway + CLI parity).
        providers_allowed=_pr.get("only"), providers_ignored=_pr.get("ignore"), providers_order=_pr.get("order"),
        provider_sort=_pr.get("sort"), provider_require_parameters=_pr.get("require_parameters", False),
        provider_data_collection=_pr.get("data_collection"), platform=platform, session_id=session_id or key,
        session_db=session_db if session_db is not None else _get_db(), ephemeral_system_prompt=system_prompt or None,
        checkpoints_enabled=is_truthy_value(os.environ.get("HERMES_TUI_CHECKPOINTS")),
        pass_session_id=is_truthy_value(os.environ.get("HERMES_TUI_PASS_SESSION_ID")),
        skip_context_files=ignore_rules, skip_memory=ignore_rules, fallback_model=_load_fallback_model(),
        **_agent_cbs(sid))
    if context_cwd_is_launch_artifact is None:
        with _sessions_lock:
            context_cwd_is_launch_artifact = _context_cwd_is_launch_artifact(_sessions.get(sid))
    agent._context_cwd_is_launch_artifact = bool(context_cwd_is_launch_artifact)
    return agent


def _hydrate_session_cwd(sid: str, key: str, session_db, profile_home: str | None) -> None:
    """Adopt the stored row's cwd, or persist the fresh session's cwd (+ schedule git meta) when the row has none."""
    owns_db, db = False, session_db
    if db is None and not profile_home:
        db = _get_db()
    elif db is None:
        try:
            db = _open_profile_session_db(profile_home)
            owns_db = True
        except Exception:
            # FAIL CLOSED (as the deferred-build bind): a named-profile session must never touch the launch
            # state.db — skip hydration (the row lands on the agent's own lazy-create once the store recovers).
            logger.warning("profile session store unavailable for %s — skipping cwd hydration instead of "
                           "touching the launch state.db", profile_home, exc_info=True)
    try:
        if db is not None:
            row = db.get_session(key) if hasattr(db, "get_session") else None
            if row and row.get("cwd"):
                with _sessions_lock:
                    if sid in _sessions:
                        _sessions[sid]["cwd"] = row["cwd"]
            elif hasattr(db, "update_session_cwd"):
                try:
                    _persist_session_cwd_and_schedule_git_meta(_sessions[sid], _sessions[sid]["cwd"], db=db)
                except Exception:
                    logger.debug("failed to persist resumed session cwd", exc_info=True)
    finally:
        if owns_db and db is not None:
            with contextlib.suppress(Exception):
                db.close()


def _init_session(
    sid: str, key: str, agent, history: list, cols: int = 80, cwd: str | None = None,
    session_db=None, source: str | None = None, profile_home: str | None = None,
    explicit_cwd: bool = False):
    now = time.time()
    with _sessions_lock:
        _sessions[sid] = {
            "agent": agent, "session_key": key, "history": history, "history_lock": threading.Lock(),
            "history_version": 0, "inflight_turn": None, "created_at": now, "last_active": now,
            "running": False, "attached_images": [], "image_counter": 0, "cwd": cwd or _completion_cwd(),
            "explicit_cwd": bool(explicit_cwd), "cols": cols, "slash_worker": None,
            "show_reasoning": _load_show_reasoning(), "source": _resolve_session_source(source),
            "tool_progress_mode": _load_tool_progress_mode(), "edit_snapshots": {}, "tool_started_at": {},
            # Profile-scoped HERMES_HOME (None = launch); SessionBranch copies the parent's (same state.db).
            "profile_home": profile_home,
            # In-session /model switch, honored on rebuild (/new, resume) — never leaks to siblings via env vars.
            "model_override": None,
            # Async events go to the transport that created the session (stdio for Ink, WS for the dashboard).
            "transport": current_transport() or _stdio_transport,
        }
        _session_todo_state(_sessions[sid])
    _hydrate_session_cwd(sid, key, session_db, profile_home)
    _register_session_cwd(_sessions[sid])
    _wire_session_agent(sid, key, agent)  # no eager slash-worker pre-warm (see _start_agent_build)
    _start_session_services(sid, key, _sessions.get(sid, {}))
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


# ── Methods: session ─────────────────────────────────────────────────


def _lazy_resume_info(cwd: str, *, model: str = "", provider: str = "", profile: str | None = None) -> dict:
    """session.info for a not-yet-built session (session.create's shape); tools/skills land with the deferred build."""
    return {
        "cwd": cwd, "branch": git_probe.branch(cwd), "project": _project_info_for_cwd(cwd),
        "model": model or _resolve_model(), "tools": {}, "skills": {}, "lazy": True,
        "desktop_contract": DESKTOP_BACKEND_CONTRACT, "profile_name": _response_profile_name(profile),
        **({"provider": provider} if provider else {}),
    }


def _deferred_session_record(
    session_key: str, *, cols: int, cwd: str, history: list, lease, source: str = "tui",
    close_on_disconnect: bool = False, display_history_prefix: list | None = None,
    profile_home: Path | None = None, lazy: bool = False, model_override=None,
    resume_runtime_overrides: dict | None = None, todo_state: dict | None = None,
    explicit_cwd: bool = False) -> dict:
    """A live-session record whose AIAgent is built later (lazy watch / cold resume) — _init_session's shape minus the agent."""
    now = time.time()
    return {
        "agent": None, "agent_error": None, "agent_ready": threading.Event(), "attached_images": [],
        "close_on_disconnect": close_on_disconnect, "active_session_lease": lease, "cols": cols,
        "created_at": now, "cwd": cwd, "display_history_prefix": display_history_prefix or [],
        "edit_snapshots": {}, "explicit_cwd": bool(explicit_cwd), "history": history,
        "history_lock": threading.Lock(), "history_version": 0, "image_counter": 0,
        "inflight_turn": None, "last_active": now, "lazy": lazy, "model_override": model_override,
        "pending_title": None,
        "profile_home": str(profile_home) if profile_home is not None else None,
        "resume_runtime_overrides": resume_runtime_overrides, "resume_session_id": session_key,
        "running": False, "session_key": session_key, "show_reasoning": _load_show_reasoning(),
        "slash_worker": None, "source": source, "tool_progress_mode": _load_tool_progress_mode(),
        "tool_started_at": {}, "todo_state": todo_state,
        "transport": current_transport() or _stdio_transport,
    }


_ANY_PROFILE = object()  # default: match a live session regardless of profile


def _live_profile_matches(session: dict, profile_home) -> bool:
    """True when ``session`` belongs to ``profile_home`` (None = launch profile; a record with no
    ``profile_home`` is the launch profile's). ``_ANY_PROFILE`` disables the check."""
    if profile_home is _ANY_PROFILE:
        return True
    return (session.get("profile_home") or None) == (str(profile_home) if profile_home else None)


def _claim_or_reuse_live(sid: str, session_key: str, record: dict, lease) -> tuple[str, dict] | None:
    """Register ``record`` as the live session for ``session_key`` under the resume lock, or — if a
    concurrent resume already won — release ``lease`` and return the winner for the caller to reuse."""
    # A live runtime of the same stored id under ANOTHER profile is not a winner to reuse.
    # See #100029.
    profile_home = record.get("profile_home")
    with _session_resume_lock:
        live = _find_live_session_by_key(session_key, profile_home)
        if live is not None:
            if lease is not None:
                lease.release()
            # The reap is cancelled by the guarded reuse (_reattach_refusal), not here: a rejected
            # reattach must leave an in-flight orphan interrupt polling.
            return live
        with _sessions_lock:
            _sessions[sid] = record
            _register_session_cwd(_sessions[sid])
        # A PRIOR runtime for this stored id may still be sentinel-parked with a reap Timer armed; cancel +
        # finalize it quietly so the reap doesn't broadcast session.reclaimed (storm).
        _cancel_ws_orphan_reap(sid)
        stale = _claim_parked_runtimes(session_key, keep_sid=sid, profile_home=profile_home)
    _finalize_superseded_runtimes(stale)  # slow finalization stays OUTSIDE _session_resume_lock
    return None


def _claim_parked_runtimes(session_key: str, *, keep_sid: str, profile_home=_ANY_PROFILE) -> list[tuple[str, dict]]:
    """Claim sentinel-parked stale runtimes of ``session_key`` for supersession: cancel their orphan-reap
    Timer and pop them here (under the caller's _session_resume_lock); the caller finalizes after release."""
    stale: list[tuple[str, dict]] = []
    with _sessions_lock:
        candidates = [
            (old_sid, old) for old_sid, old in list(_sessions.items())
            if old_sid != keep_sid and not old.get("_finalized")
            and _session_lookup_key(old, fallback=old_sid) == session_key
            and _live_profile_matches(old, profile_home) and old.get("transport") is _detached_ws_transport]
    for old_sid, _old in candidates:
        _cancel_ws_orphan_reap(old_sid)
        if (popped := _pop_session_by_id(old_sid)) is not None:
            stale.append((old_sid, popped))
    return stale


def _finalize_superseded_runtimes(stale: list[tuple[str, dict]]) -> None:
    """end_reason ``superseded_by_resume`` is deliberately NOT in _RECLAIM_END_REASONS (no ``session.reclaimed``
    broadcast → no reap->broadcast->resume loop) but IN _RECOVERABLE_END_REASONS (Bot Chat resurrection applies)."""
    for old_sid, popped in stale:
        try:
            _teardown_popped_session(popped, end_reason="superseded_by_resume")
        except Exception:
            logger.exception("superseded runtime teardown failed sid=%s", old_sid)


def _schedule_agent_build(sid: str, delay: float = 0.05) -> None:
    """Pre-warm a deferred session's agent off the response path (session.create + cold resume; _sess() also builds on demand)."""

    def _run():
        if (session := _sessions.get(sid)) is not None:
            _start_agent_build(sid, session)
    timer = threading.Timer(delay, _run)
    timer.daemon = True
    timer.start()


def _load_resume_transcript(db, stored_id: str) -> tuple[list, list, list]:
    """(raw_history, display_history, ancestor_prefix) for a cold resume. The full lineage is materialized
    only while it fits sessions.max_resume_messages (the transcript is REST-paginated), else the tip alone."""
    from hermes_state import SessionResumeTooLargeError
    prefix_fits = True
    guard = getattr(db, "assert_resume_safe", None)
    if callable(guard):
        try:
            guard(stored_id)
        except SessionResumeTooLargeError as exc:
            prefix_fits = False
            logger.info("resume %s: compression lineage exceeds the resume limit (%s); hydrating the tip segment only",
                        stored_id, exc)
        except Exception:
            logger.debug("resume lineage guard failed; loading full lineage", exc_info=True)
    if prefix_fits:
        raw_history, display_history = db.get_resume_conversations(stored_id)
        return raw_history, display_history, db.get_ancestor_display_prefix(stored_id)
    raw_history = db.get_messages_as_conversation(stored_id, repair_alternation=True, include_row_ids=True)
    return raw_history, raw_history, []


def _schedule_resume_hydration(sid: str, stored_id: str, db, *, close_db: bool = False) -> None:
    """Load a cold resume's transcript off the JSON-RPC response path."""

    def _run() -> None:
        session = _sessions.get(sid)
        try:
            if session is None:
                return
            _emit("session.resume_progress", sid, {"phase": "history", "status": "loading"})
            db.reopen_session(stored_id)
            raw_history, display_history, prefix = _load_resume_transcript(db, stored_id)
            # Display keeps the full transcript; the model-fed history drops a dangling/interrupted
            # tool-call tail so a session killed mid-loop does not replay the unanswered call forever
            # (#29086).
            history = sanitize_replay_history(raw_history)
            if _sessions.get(sid) is not session:
                return
            with session["history_lock"]:
                session.update(history=history, display_history_prefix=prefix, resume_hydrating=False,
                               resume_message_count=len(display_history))
            # Deferred resumes answered before the transcript existed; cache the derived todo snapshot now.
            todo_state = _todo_state_from_history(history)
            if todo_state is not None and session.get("todo_state") is None:
                session["todo_state"] = todo_state
            session["resume_history_ready"].set()
            _emit("session.resume_progress", sid,
                  {"message_count": len(display_history), "phase": "history", "status": "complete"})
            _maybe_schedule_auto_continue(sid, session, stored_id)
            _start_agent_build(sid, session)
        except Exception as exc:
            if _sessions.get(sid) is not session:
                return
            message = f"resume failed: {exc}"
            session.update(resume_hydrating=False, resume_history_error=message, agent_error=message)
            session["resume_history_ready"].set()
            session["agent_ready"].set()
            _emit("session.resume_progress", sid, {"message": message, "phase": "history", "status": "failed"})
            _emit("error", sid, {"message": message})
            with _sessions_lock:
                discarded = _sessions.pop(sid, None) if _sessions.get(sid) is session else None
            if (lease := (discarded or {}).get("active_session_lease")) is not None:
                lease.release()
        finally:
            if close_db and hasattr(db, "close"):
                try:
                    db.close()
                except Exception:
                    logger.debug("failed to close resume db for %s", sid, exc_info=True)
    threading.Thread(target=_run, daemon=True).start()


def _session_pending_kind(sid: str) -> str:
    return next((str(_pending_prompt_payloads.get(rid, ("input.request", {}))[0]).removesuffix(".request")
                 for rid, (owner_sid, _ev) in list(_pending.items()) if owner_sid == sid), "")


def _session_live_status(sid: str, session: dict) -> str:
    if _session_pending_kind(sid):
        return "waiting"
    ready = session.get("agent_ready")
    # Unset + build never started = a lazy watch session idling, not one stuck mid-construction.
    if ready is not None and not ready.is_set() and session.get("agent_build_started"):
        return "starting"
    return "working" if session.get("running") else "idle"


def _session_live_title(session: dict, key: str) -> str:
    title = str(session.get("pending_title") or "").strip()
    with contextlib.suppress(Exception), _session_db(session) as db:
        title = str(db.get_session_title(key) or title or "").strip() if db is not None else title
    return title


def _session_live_item(sid: str, session: dict, current_sid: str = "") -> dict:
    key = _session_lookup_key(session, fallback=sid)
    agent = session.get("agent")
    history = list(session.get("history") or [])
    status = _session_live_status(sid, session)
    inflight = _inflight_snapshot(session)
    queued = _queued_prompt_snapshot(session)
    preview = next((" ".join(text.split())[:160] for msg in reversed(history)
                    if (text := _content_display_text(msg.get("content", msg.get("text", ""))).strip())), "")
    if queued:
        preview = " ".join(str(queued.get("user") or preview).split())[:160]
    elif inflight:
        preview = " ".join(str(inflight.get("assistant") or inflight.get("user") or preview).split())[:160]
    now = time.time()
    return {
        "current": sid == current_sid, "id": sid,
        "last_active": float(session.get("last_active") or session.get("created_at") or now),
        "message_count": len(history),
        "model": str(getattr(agent, "model", "") or _resolve_model()), "preview": preview,
        "session_key": key, "started_at": float(session.get("created_at") or now), "status": status,
        "title": _session_live_title(session, key),
    }


def _session_lookup_key(session: dict, *, fallback: str = "") -> str:
    return str(getattr(session.get("agent"), "session_id", None) or session.get("session_key") or fallback or "")


def _find_live_session_by_key(session_key: str, profile_home=_ANY_PROFILE) -> tuple[str, dict] | None:
    # Timestamp-based stored ids can exist in several profiles' stores; a bare-id match would hand
    # profile B's resume profile A's runtime, so profile-aware callers match on (profile_home, key).
    # Profile-aware callers pass the home they resolved; the match must then be on (profile_home,
    # session_key). See #100029.
    for sid, session in list(_sessions.items()):
        if (not session.get("_finalized") and _session_lookup_key(session, fallback=sid) == session_key
                and _live_profile_matches(session, profile_home)):
            return sid, session
    return None


def _fallback_session_info(session: dict) -> dict:
    agent = session.get("agent")
    if agent is not None:
        return _session_info(agent)
    # The SESSION's own workspace, not the launch dir (wrong project in the desktop Files pane). `branch` is
    # always emitted ("" outside git) so a stale label clears; `desktop_contract` missing reads as "out of date".
    # Reporting `_default_session_cwd()` here told a lazily-resumed session's client that its workspace was
    # wherever the gateway process happened to start, so the desktop Files pane painted the wrong project
    # even after the renderer rebound correctly (#71254). `branch` is always emitted ("" outside a git repo)
    # so a client can clear a stale label instead of retaining it — the same contract `_lazy_session_info`
    # above already follows.
    cwd = _session_cwd(session)
    return {
        "cwd": cwd, "branch": git_probe.branch(cwd), "project": _project_info_for_cwd(cwd), "lazy": True,
        "model": _resolve_model(), "skills": {}, "tools": {}, "desktop_contract": DESKTOP_BACKEND_CONTRACT,
    }


def _reconcile_display_with_live(db_display: list[dict], in_memory: list[dict]) -> list[dict]:
    """Merge the persisted DISPLAY lineage with the in-memory live history: ``db_display`` is verbatim and
    candidate-inclusive (verification rows the model history collapses out) but can lag by a flush;
    ``in_memory`` is the recency authority but the collapsed *model* projection. Keep the DB display as base,
    append only the in-memory tail past the last DB row's ``(role, text)`` anchor — the verification answer
    survives a warm switch AND a not-yet-flushed live turn is kept."""
    if not db_display:
        return in_memory
    if not in_memory:
        return db_display

    def _key(msg: dict) -> tuple:
        return (msg.get("role"), _coerce_message_text(msg.get("content")))
    anchor = _key(db_display[-1])
    last_shared = max((idx for idx, msg in enumerate(in_memory) if isinstance(msg, dict) and _key(msg) == anchor), default=-1)
    if last_shared == -1:
        return db_display  # DB tail not in memory (DB ahead, or diverged) — trust it over duplicating
    return list(db_display) + list(in_memory[last_shared + 1 :])


def _live_visible_history(session: dict, db, in_memory_fallback: list[dict]) -> list[dict]:
    """User-visible DISPLAY projection for a live/warm session: the persisted display lineage (same read as
    resume/REST so the payloads agree) reconciled with the in-memory tail; in-memory when the DB is unavailable."""
    key = session.get("session_key")
    if db is not None and key:
        try:
            # include_compacted: a compacted session's archived turns are still the user's
            # conversation; without them a warm switch repainted the chat as summary + tail only.
            display = db.get_messages_as_conversation(
                key, include_ancestors=True, include_row_ids=True, include_compacted=True)
            # See #92080.
            return _reconcile_display_with_live(display, in_memory_fallback)
        except Exception:
            logger.debug("live display projection read failed", exc_info=True)
    return in_memory_fallback


def _live_session_payload(
    sid: str, session: dict, *, cols: int | None = None, touch: bool = False,
    transport: Transport | None = None, omit_messages: bool = False) -> dict:
    with session["history_lock"]:
        if cols is not None:
            session["cols"] = cols
        if transport is not None:
            _rebind_live_transport(sid, session, transport)
        if touch:
            # #84417: do not re-fire the live turn's original user text from a stale server-queue
            # self-duplicate after settle.
            session["last_active"] = time.time()
        in_memory_history = list(session.get("display_history_prefix") or []) + list(session.get("history") or [])
        inflight, queued = _inflight_snapshot(session), _queued_prompt_snapshot(session)
        running, turn_started_at = bool(session.get("running")), _turn_started_at(session)
    # Persisted display lineage via the session's profile-aware DB (not the launch ``_get_db()``), read
    # outside the history lock (the DB has its own). ``omit_messages`` skips the read (fast path).
    if omit_messages:
        history = in_memory_history
    else:
        with _session_db(session) as db:
            history = _live_visible_history(session, db, in_memory_history)
    # message_count follows _resume_response: the stored size when messages are omitted, else the wire count
    # (a hidden seed row is in ``history`` but never on the wire).
    messages = [] if omit_messages else _history_to_messages(history)
    payload = {
        "info": _fallback_session_info(session), "message_count": len(history) if omit_messages else len(messages),
        "messages": messages,
        "messages_omitted": omit_messages, "running": running, "turn_started_at": turn_started_at,
        "session_id": sid, "session_key": _session_lookup_key(session, fallback=sid),
        "started_at": float(session.get("created_at") or time.time()),
        "status": _session_live_status(sid, session),
    }
    for key, value in (("inflight", inflight), ("queued", queued),
                       ("pending_approval", _pending_approval_request_payload(str(session.get("session_key") or ""))),
                       ("pending_clarify", _pending_clarify_request_payload(sid))):
        if value:
            payload[key] = value
    return _attach_todo_state(payload, session)


def _main_runtime_from_agent(agent) -> dict | None:
    """Aux-client main_runtime override from a live agent, so a one-shot inherits the session's runtime."""
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


# Pet helpers are fail-open throughout: a decode hiccup degrades to a static fallback rather than
# breaking the (cosmetic) pet surface.
_pet_payload_cache_lock = threading.Lock()
_pet_payload_cache: dict[tuple, dict] = {}


def _pet_sheet_revision(spritesheet) -> str:
    """Stable revision id for one spritesheet file."""
    with contextlib.suppress(Exception):
        stat = spritesheet.stat()
        return f"{stat.st_mtime_ns}:{stat.st_size}"
    return "0:0"


def _clone_pet_payload(payload: dict) -> dict:
    """Shallow-clone cached payloads so callers can't mutate shared state."""
    out = dict(payload)
    for key, kind in (("framesByState", dict), ("framesByRow", dict), ("stateRows", list)):
        if isinstance(payload.get(key), kind):
            out[key] = kind(payload[key])
    return out


def _pet_row_frame_counts(spritesheet) -> dict:
    """Real frame count per concrete spritesheet row name."""
    with contextlib.suppress(Exception):
        from PIL import Image
        from agent.pet import constants, render
        with Image.open(spritesheet) as opened:
            image = opened.convert("RGBA")
        W, H = constants.FRAME_W, constants.FRAME_H
        cols = max(1, image.width // W)
        row_count = max(1, image.height // H)
        rows = constants.state_rows_for_grid(row_count)
        out: dict[str, int] = {}
        for row_idx, name in enumerate(rows[:row_count]):
            top = row_idx * H
            blank = lambda col: render._frame_is_blank(image.crop((col * W, top, col * W + W, top + H)))
            out[name] = next((col for col in range(cols) if blank(col)), cols)  # frames before the first blank cell
        return out
    return {}


def _pet_cfg() -> dict:
    """``display.pet`` from the canonical config ({} on any failure)."""
    with contextlib.suppress(Exception):
        from hermes_cli.config import load_config
        display = load_config().get("display")
        pet = display.get("pet") if isinstance(display, dict) else None
        return pet if isinstance(pet, dict) else {}
    return {}


def _pet_config_scale() -> float:
    """Configured ``display.pet.scale`` (or the engine default), never raises."""
    from agent.pet import constants
    with contextlib.suppress(Exception):
        return float(_pet_cfg().get("scale", constants.DEFAULT_SCALE) or constants.DEFAULT_SCALE)
    return constants.DEFAULT_SCALE


def _pet_sprite_payload(pet, *, scale: float) -> dict:
    """Renderer payload (spritesheet bytes + geometry) for *pet* — one shape for ``pet.info`` (active
    mascot) and ``pet.hatch`` (unadopted preview)."""
    import base64
    from agent.pet import constants
    try:
        stat = pet.spritesheet.stat()
        cache_key = (str(pet.spritesheet), stat.st_mtime_ns, stat.st_size, pet.slug, pet.display_name, round(scale, 4))
    except Exception:  # noqa: BLE001
        cache_key = None
    if cache_key is not None:
        with _pet_payload_cache_lock:
            cached = _pet_payload_cache.get(cache_key)
        if cached is not None:
            return _clone_pet_payload(cached)
    try:  # real (padding-trimmed) frame count per state; {} → the canvas uses the static framesPerState
        from agent.pet import render
        frames_by_state = render.state_frame_counts(str(pet.spritesheet))
    except Exception:  # noqa: BLE001
        frames_by_state = {}
    raw = pet.spritesheet.read_bytes()
    mime = "image/png" if pet.spritesheet.suffix.lower() == ".png" else "image/webp"
    payload = {
        "slug": pet.slug, "displayName": pet.display_name, "mime": mime,
        "spritesheetBase64": base64.standard_b64encode(raw).decode("ascii"),
        "spritesheetRevision": _pet_sheet_revision(pet.spritesheet), "frameW": constants.FRAME_W,
        "frameH": constants.FRAME_H, "framesPerState": constants.FRAMES_PER_STATE,
        "framesByState": frames_by_state,
        "framesByRow": _pet_row_frame_counts(pet.spritesheet), "loopMs": constants.LOOP_MS,
        "scale": scale, "stateRows": _pet_state_rows(pet.spritesheet),
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
    pet_cfg = _pet_cfg()
    enabled = is_truthy_value(pet_cfg.get("enabled"), default=False)
    pet = store.resolve_active_pet(str(pet_cfg.get("slug", "") or "")) if enabled else None
    return enabled, pet, float(pet_cfg.get("scale", constants.DEFAULT_SCALE) or constants.DEFAULT_SCALE)


def _pet_state_rows(spritesheet) -> list[str]:
    """Row taxonomy for the concrete sheet (legacy 8-row or current 9-row atlas), in the renderer's `PetState` names."""
    from agent.pet import constants
    with contextlib.suppress(Exception):
        from PIL import Image
        with Image.open(spritesheet) as image:
            row_count = max(1, image.height // constants.FRAME_H)
        return list(constants.state_rows_for_grid(row_count))
    return list(constants.STATE_ROWS)


def _pet_gen_root():
    """Profile-scoped staging dir for in-progress generation drafts."""
    root = get_hermes_home() / "cache" / "pet-gen"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _pet_gen_sweep(root, *, max_age_s: float = 3600.0) -> None:
    """Drop stale draft staging dirs so cache never grows unbounded."""
    import shutil
    try:
        now = time.time()
        for child in (c for c in root.iterdir() if c.is_dir() and now - c.stat().st_mtime > max_age_s):
            shutil.rmtree(child, ignore_errors=True)
    except Exception as exc:  # noqa: BLE001 - cleanup is best-effort
        logger.debug("pet-gen sweep failed: %s", exc)


def _pet_png_data_uri(path, *, max_px: int = 160) -> str:
    """Downscaled PNG data URI for a draft image (small preview payload)."""
    import base64, io
    from PIL import Image
    with Image.open(path) as opened:
        img = opened.convert("RGBA")
    img.thumbnail((max_px, max_px), Image.LANCZOS)
    img.save(buf := io.BytesIO(), format="PNG")
    return "data:image/png;base64," + base64.standard_b64encode(buf.getvalue()).decode("ascii")


# Cooperative cancellation for pet generation: Stop aborts the RPC, but the pool job keeps running unless
# pet.cancel flips its token (polled between provider calls).
_pet_cancel_lock = threading.Lock()
_pet_cancelled: set[str] = set()
_PET_REFERENCE_MIME_EXT = {"png": "png", "jpeg": "jpg", "jpg": "jpg", "webp": "webp", "gif": "gif"}
try:
    _PET_REFERENCE_MAX_BYTES = max(1, int(os.environ.get("HERMES_PET_REFERENCE_MAX_BYTES") or str(16 * 1024 * 1024)))
except (TypeError, ValueError):
    _PET_REFERENCE_MAX_BYTES = 16 * 1024 * 1024


def _pet_reference_images_from_data_url(ref_raw: str, stage) -> list:
    """Decode + validate a reference-image data URL into the stage dir."""
    import base64, binascii
    import re as _re
    match = _re.match(r"^data:image/([a-zA-Z0-9.+-]+);base64,(.*)$", ref_raw, _re.DOTALL)
    if not match:
        raise ValueError("invalid reference image format")
    if (ext := _PET_REFERENCE_MIME_EXT.get(match.group(1).lower())) is None:
        raise ValueError("unsupported reference image type")
    payload = "".join(match.group(2).split())
    if (len(payload) * 3) // 4 > _PET_REFERENCE_MAX_BYTES:
        raise ValueError("reference image too large")
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("invalid reference image data") from exc
    if len(raw) > _PET_REFERENCE_MAX_BYTES:
        raise ValueError("reference image too large")
    (ref_path := stage / f"reference.{ext}").write_bytes(raw)
    return [ref_path]


def _pet_cancel_arm(token: str) -> None:
    """Clear a stale cancel flag at the start of a generate/hatch run."""
    with _pet_cancel_lock:
        _pet_cancelled.discard(token)


_pet_cancel_release = _pet_cancel_arm


def _pet_cancel_request(token: str) -> None:
    with _pet_cancel_lock:
        _pet_cancelled.add(token)


def _pet_is_cancelled(token: str) -> bool:
    with _pet_cancel_lock:
        return token in _pet_cancelled


# ── Spawn-tree snapshots: the TUI owns subagent state (/agents overlay; registry in tools/delegate_tool), posts
# the final tree on turn-complete and /replay fetches by session_id + filename. Layout: spawn-trees/<sid>/<ts>.json


def _spawn_trees_root():
    root = get_hermes_home() / "spawn-trees"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _spawn_tree_session_dir(session_id: str):
    d = _spawn_trees_root() / ("".join(c if c.isalnum() or c in "-_" else "_" for c in session_id) or "unknown")
    d.mkdir(parents=True, exist_ok=True)
    return d


# Per-session append-only JSONL index so `spawn_tree.list` needn't read every snapshot; a cache — a lost
# line just means list() falls back to a directory scan.
# Read by `spawn_tree.list` so scanning doesn't require reading every full snapshot file (Copilot review on
# #14045). One JSON object per line.
_SPAWN_TREE_INDEX = "_index.jsonl"


def _append_spawn_tree_index(session_dir, entry: dict) -> None:
    try:
        with (session_dir / _SPAWN_TREE_INDEX).open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.debug("spawn_tree index append failed: %s", exc)  # never block the save


def _read_spawn_tree_index(session_dir) -> list[dict]:
    out: list[dict] = []
    try:
        with (session_dir / _SPAWN_TREE_INDEX).open("r", encoding="utf-8") as f:
            for line in f:
                if line := line.strip():
                    with contextlib.suppress(json.JSONDecodeError):
                        out.append(json.loads(line))
    except OSError:
        return []
    return out


# ── Methods: prompt ──────────────────────────────────────────────────


_GOAL_COMPRESSION_RECOVERY_ATTEMPTS = "_goal_compression_recovery_attempts"
_GOAL_COMPRESSION_RECOVERY_LIMIT = 1

# Captured at import time: tests monkeypatch threading.Thread with a synchronous stub, and the ticker only
# exits once `stop` is set AFTER run_conversation returns — inline it would spin forever.
_RealThread = threading.Thread


def _start_usage_ticker(sid: str, agent, interval: float = 1.0) -> tuple[threading.Event, threading.Thread]:
    """Push live ``session.usage`` snapshots every ``interval`` s while a turn runs. The caller must set the
    Event AND join the thread before ``message.complete``: a late tick would roll the final usage back."""
    stop = threading.Event()
    # Dedup baseline sampled BEFORE the thread starts (the client has the turn-start values); a late-scheduled
    # thread would otherwise absorb the first counter growth and never emit it.
    baseline: dict | None = None
    with contextlib.suppress(Exception):
        baseline = _get_usage(agent)

    def _loop() -> None:
        last = baseline
        while not stop.wait(interval):
            with contextlib.suppress(Exception):
                usage = _get_usage(agent)
                if usage == last:
                    continue  # counters frozen (one long API call in flight): don't re-render the status bar
                last = usage
                if stop.is_set():
                    break  # turn ended while snapshotting; message.complete carries the authoritative usage
                _emit("session.usage", sid, {"usage": usage})
    thread = _RealThread(target=_loop, daemon=True)
    thread.start()
    return stop, thread


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
) -> bool:
    with session["history_lock"]:
        if session.get("_closing"):
            session["running"] = False
            return False
        if (
            queued_prompt_generation is not None
            and int(session.get("_queued_prompt_generation", 0)) != queued_prompt_generation
        ):
            session["running"] = False
            return False
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
    # Desktop/TUI observability (#86647): this is the ONE INFO record proving
    # a Desktop/TUI prompt was accepted by THIS process, and it ties together
    # every id a rotation-mute trace needs — the UI session id, the gateway
    # session_key, and the agent's live session_id (which compression rotates
    # independently of the other two). Before this line a Desktop request left
    # no trace in agent.log at all ("0 platform=desktop" — see #86647), so a
    # muted window was structurally indistinguishable from a request that
    # never arrived. No prompt content is logged.
    _turn_started_monotonic = time.monotonic()
    logger.info(
        "tui prompt accepted: ui_session=%s session_key=%s agent_session_id=%s "
        "kind=%s chars=%s images=%d",
        sid,
        session.get("session_key") or "",
        getattr(agent, "session_id", "") or "",
        display_kind or "user",
        len(text) if isinstance(text, str) else "-",
        len(images),
    )
    _emit("message.start", sid)

    def run():
        # The conversation runs on a fresh thread, so ContextVars from the RPC
        # dispatcher do not follow automatically. Rebind the exact transport
        # stored on this session generation before any tool can commission a
        # child; delegate_task then captures it as non-serializable authority.
        transport_token = bind_transport(session.get("transport"))
        runtime_session_token = _current_runtime_session_record.set(session)
        # Bound eagerly so the except/finally paths below always have an agent
        # even if turn setup throws; re-read after _sync_bot_capabilities,
        # which may swap in a rebuilt agent for Bot Chat sessions.
        agent = session["agent"]
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
                _sync_agent_compression_with_config(sid, session)
            _sync_agent_turn_limit_with_config(session)
            # Bot Chat capability sync — adopt Settings→Capabilities edits
            # (skills/toolsets/MCP/SOUL) into the eternal bot session before
            # the turn runs. No-op for every other session shape.
            _sync_bot_capabilities(sid, session)
            agent = session["agent"]
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
            # "text"   → reference the image paths in the message so the agent
            #            analyzes them in-loop with vision_analyze (never
            #            blocking the submit path on vision calls — #83291).
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
                            run_message = _build_image_ref_message(prompt, images)
                    except Exception as _img_exc:
                        print(
                            f"[tui_gateway] native attach failed, falling back to text: {_img_exc}",
                            file=sys.stderr,
                        )
                        run_message = _build_image_ref_message(prompt, images)
                else:
                    run_message = _build_image_ref_message(prompt, images)

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
                run_message = _prepend_note(run_message, SPEECH_INTERRUPTED_NOTE)

            # Reactions the user added since the last turn.
            run_message = _prepend_note(run_message, _pending_reaction_notes(session))

            # Which window the message was typed into. HUD mode is per-turn
            # state, so it cannot live in the (byte-stable) system prompt.
            run_message = _prepend_note(run_message, _hud_surface_note(session))

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
            # Auto-titling now fires inside the turn prologue (shared by every
            # surface). Hand the agent this session's live-rename hook so the
            # sidebar repaints the moment a title lands, rather than waiting
            # for the next list refresh.
            _title_key = session.get("session_key") or sid
            agent._on_session_title = lambda t, _src, _k=_title_key: _emit(
                "session.title", sid, {"session_id": _k, "title": t}
            )
            _usage_stop, _usage_thread = _start_usage_ticker(sid, agent)
            try:
                result = agent.run_conversation(run_message, **run_kwargs)
            finally:
                # Stop AND join before anything below emits: an in-flight tick
                # surviving past message.complete would roll the client's final
                # usage back to a stale mid-turn snapshot. The join is
                # deliberately unbounded — once stop is set it only ever waits
                # out one in-flight _get_usage/_emit, and the worst case there
                # (a stalled transport write, up to _WS_WRITE_TIMEOUT_S) would
                # stall the message.complete emit below just the same. A
                # timed-out join would abandon the tick to land after
                # message.complete.
                _usage_stop.set()
                _usage_thread.join()
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
                            # Check if the only mutation was a pivot marker
                            # the gateway itself inserted mid-turn (#76870).
                            # If so the agent output is still valid — merge it
                            # into the current history that now contains the
                            # marker. A personality change counts here too:
                            # unlike a model switch it has no pending queue, so
                            # `/personality` during a running turn lands
                            # immediately and used to read as a genuine desync,
                            # dropping the finished turn (#82756).
                            #
                            # _append_model_switch_marker strips prior markers
                            # in-place then appends a new one, so the delta
                            # is NOT a simple tail-slice — we must compare
                            # content, not indices.
                            current_history = list(session["history"])
                            history_no_markers = [
                                e for e in history if not _is_pivot_marker(e)
                            ]
                            current_no_markers = [
                                e for e in current_history if not _is_pivot_marker(e)
                            ]
                            pivot_only = (
                                current_no_markers == history_no_markers
                                and any(
                                    _is_pivot_marker(e)
                                    for e in current_history
                                )
                            )
                            if pivot_only:
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
            # Structured layer descriptor ({layer, code, retryable}) so
            # clients can name WHICH part of the stack failed (provider /
            # streaming / auth / gateway / …) and offer layer-appropriate
            # recovery actions instead of sniffing the message string.
            # Advisory: older clients ignore it, absence falls back to
            # string heuristics on newer clients. Computed before the retain
            # below so resume replay carries the same descriptor.
            _error_surface = None
            if status == "error":
                try:
                    from agent.error_surface import build_error_surface_from_result

                    _error_surface = build_error_surface_from_result(
                        result,
                        provider=str(getattr(agent, "provider", "") or ""),
                        model=str(getattr(agent, "model", "") or ""),
                    )
                except Exception:
                    _error_surface = None
            with session["history_lock"]:
                if status == "error":
                    # Returned-error result (provider 4xx, budget, etc.): retain
                    # the failed turn for resume replay instead of clearing it.
                    # If this terminal frame is lost to a disconnect, resume's
                    # inflight payload is the only carrier of the failure.
                    _fail_inflight_turn(
                        session,
                        result.get("error") if isinstance(result, dict) else raw,
                        error_surface=_error_surface,
                    )
                    turn_error_retained = True
                else:
                    _clear_inflight_turn(session)
            if status == "error":
                payload["error"] = str(
                    (result.get("error") if isinstance(result, dict) else "") or raw
                )
                payload["recoverable"] = True
                if _error_surface:
                    payload["error_surface"] = _error_surface
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
            compression_exhausted = bool(
                isinstance(result, dict) and result.get("compression_exhausted")
            )
            try:
                recovery_prompt, recovery_notice = _plan_goal_compression_recovery(
                    session,
                    result,
                    status=status,
                    raw=raw,
                )
                if recovery_notice:
                    _emit(
                        "status.update",
                        sid,
                        {"kind": "goal", "text": recovery_notice},
                    )
                if recovery_prompt:
                    goal_followup = recovery_prompt
            except Exception as _goal_recovery_exc:
                print(
                    f"[tui_gateway] goal compression recovery failed: "
                    f"{type(_goal_recovery_exc).__name__}: {_goal_recovery_exc}",
                    file=sys.stderr,
                )

            # Compression failures are never judge input: the error text is
            # not work toward the goal, and evaluating it would spend a turn.
            if not compression_exhausted and _is_successful_goal_turn(
                result, status, raw
            ):
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

            # ── /loop tick completion ──────────────────────────────────
            # If the turn that just finished was a /loop wakeup (fired by
            # the notification poller), evaluate it: LOOP_COMPLETE marker,
            # --until judge, --times / max_ticks caps, next-tick schedule.
            if status == "complete":
                try:
                    from hermes_cli.loops import LoopManager

                    loop_sid_key = session.get("session_key") or ""
                    if loop_sid_key:
                        loop_mgr = LoopManager(session_id=loop_sid_key)
                        loop_state = loop_mgr.state
                        if loop_state is not None and loop_state.awaiting_response:
                            loop_decision = loop_mgr.complete_tick(
                                raw if isinstance(raw, str) else ""
                            )
                            loop_msg = loop_decision.get("message") or ""
                            if loop_msg:
                                _emit(
                                    "status.update",
                                    sid,
                                    {"kind": "loop", "text": loop_msg},
                                )
                except Exception as _loop_exc:
                    print(
                        f"[tui_gateway] loop completion hook failed: "
                        f"{type(_loop_exc).__name__}: {_loop_exc}",
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
            # Closing bookend of the "tui prompt accepted" record above —
            # fires on every path (success, returned error, exception,
            # interrupt), so one accepted prompt always produces exactly one
            # finished record. agent.session_id is re-read here because
            # compression may have rotated it mid-turn: an accepted/finished
            # pair whose agent_session_id changed IS a rotation trace
            # (#86647). A missing finished record means the turn thread died
            # without reaching this finally.
            logger.info(
                "tui turn finished: ui_session=%s session_key=%s "
                "agent_session_id=%s status=%s error_retained=%s duration=%.1fs",
                sid,
                session.get("session_key") or "",
                getattr(agent, "session_id", "") or "",
                (
                    result.get("interrupted")
                    and "interrupted"
                    or result.get("error")
                    and "error"
                    or "complete"
                )
                if isinstance(result, dict)
                else ("error" if turn_error_retained else "complete"),
                turn_error_retained,
                time.monotonic() - _turn_started_monotonic,
            )
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
    with _sessions_lock:
        registered = _sessions.get(sid)
        can_start = (
            not session.get("_closing")
            and (registered is None or registered is session)
        )
        if can_start:
            session["_run_thread"] = run_thread
            run_thread.start()
    if not can_start:
        with session["history_lock"]:
            session["running"] = False
    return can_start


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
    """Resolve the file-attachment staging dir against the session's effective home.

    Anchored on the session profile's ``attachments/`` dir (same rule as
    ``_session_images_dir``): ``file.attach`` runs BEFORE ``prompt.submit``
    installs the session's profile HERMES_HOME override, while the docker/ssh
    sandbox mounts are resolved against the *session profile's* home at run
    time — so the staged file must land where the bind mount points, or the
    container can never see it (#76577). ``attachments/`` is registered in
    ``tools.credential_files._CACHE_DIRS`` and auto-mounted into containers.
    """
    profile_home = session.get("profile_home")
    base = Path(profile_home) if profile_home else _hermes_home
    root = base / "attachments"
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
         it into the session home's ``attachments/`` dir (bind-mounted into
         container backends) so the ``@file:`` ref resolves inside the sandbox.
      3. The path doesn't exist on the gateway (the common remote case: it's a
         path on the CLIENT's disk) — decode the uploaded ``data_url`` bytes and
         write them into the session home's ``attachments/`` dir.

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
    question_id = str(params.get("question_id") or "")
    with _prompt_lock:
        entry = _pending.get(r)
        if not entry:
            return _ok(rid, {"status": "expired"}) if allow_expired and r else _err(rid, 4009, f"no pending {key} request")
        _, ev = entry
        batch = _batch_clarify.get(r)
        if batch is not None and question_id:
            # Per-question lock; update-in-place so an answer stays editable until every qid is locked (Confirm).
            if question_id not in batch["qids"]:
                return _err(rid, 4002, f"unknown question_id {question_id!r}")
            batch["answers"][question_id] = params.get(key, "")
            if not (remaining := [qid for qid in batch["qids"] if qid not in batch["answers"]]):
                ev.set()
            return _ok(rid, {"status": "ok", "remaining": remaining})
        _answers[r] = params.get(key, "")
        ev.set()
    return _ok(rid, {"status": "ok"})


# ── Methods: tools & system ──────────────────────────────────────────


def _session_processes(session: dict) -> list:
    """Background processes owned by this session (registry session_key match)."""
    # Drain completion notifications that arrived during this turn. The background poller handles
    # between-turn delivery; this is the safety net for events that arrived mid-turn. Ownership filter
    # (#42674, #35652): a turn finishing in session B must not consume an event that belongs to session A.
    # The registry requeues every addressed event this session cannot positively claim; the poller then
    # delivers it to a live owner or drops an orphan.
    from tools.process_registry import process_registry
    key = str(session.get("session_key") or "")
    owned = []
    for entry in process_registry.list_sessions():
        proc = process_registry.get(entry["session_id"])
        if proc is not None and str(getattr(proc, "session_key", "") or "") == key:
            entry["output_tail"] = (proc.output_buffer or "")[-4000:]  # the 200-char list preview is too thin for the viewer
            owned.append(entry)
    return owned


# Serialize reload.mcp (runs on the pool): overlapping shutdown+discover pairs would leave the registry half-built.
_mcp_reload_lock = threading.Lock()
# Bumped per SUCCESSFUL reload; a follower skips only if it advanced while it waited (a leader that threw
# leaves it unchanged → the follower reloads itself).
_mcp_reload_gen = 0
# The mcp_rev the last successful reload actually LOADED (re-hashed after discovery); a follower coalesces
# only when its requested rev matches, otherwise the config changed under the leader.
_mcp_reload_loaded_rev = ""
# Bounded convergence for a config edit racing a slow reload: the leader re-hashes until the hash is stable.
_MCP_RELOAD_MAX_PASSES = 3


def _compute_mcp_rev() -> str:
    """Hash of mcp_servers (definitions — omitting it meant an edited server never bumped the rev) + mcp +
    tools. ``config.get mtime`` ships it so cosmetic writes don't reload; ``reload.mcp`` coalesces on it. "" = unknown."""
    with contextlib.suppress(Exception):
        cfg = _load_cfg()
        rev_src = json.dumps({k: cfg.get(k) for k in ("mcp", "mcp_servers", "tools")}, sort_keys=True, default=str)
        return hashlib.sha1(rev_src.encode()).hexdigest()[:12]
    return ""


def _finish_reload(rid, params: dict, *, coalesced: bool) -> dict:
    """Shared tail for both reload paths: honor ``always`` (persist the confirm opt-out) and return the ok payload."""
    if bool(params.get("always", False)):
        try:
            from cli import save_config_value
            save_config_value("approvals.mcp_reload_confirm", False)
        except Exception as _exc:
            logger.warning("Failed to persist mcp_reload_confirm=false: %s", _exc)
    return _ok(rid, {"status": "reloaded", "loaded_rev": _mcp_reload_loaded_rev, **({"coalesced": True} if coalesced else {})})


_TUI_HIDDEN: frozenset[str] = frozenset({"sethome", "set-home", "commands", "approve", "deny"})

_TUI_EXTRA: list[tuple[str, str, str]] = [
    ("/density", "Toggle compact display mode", "TUI"),
    ("/logs", "Show recent gateway log lines", "TUI"),
    ("/mouse", "Set mouse tracking preset [on|off|toggle|wheel|buttons|all]", "TUI"),
    ("/sessions", "Switch between live TUI sessions", "TUI"),
]

# Commands that queue onto _pending_input in the CLI; the slash worker has no reader for that queue, so
# slash.exec routes them to command.dispatch instead.
_PENDING_INPUT_COMMANDS: frozenset[str] = frozenset({
    "retry", "queue", "q", "steer", "plan", "goal", "loop", "proactive", "moa", "undo", "learn",
    "init", "compress", "compact",
})

_WORKER_BLOCKED_COMMANDS: frozenset[str] = frozenset({"snapshot", "snap"})


def _skill_usage_lookup():
    """``(usage, origin)`` callables for the skill catalog: activity count (use + view + patch) and
    "hub" / "bundled" / "local" (``/api/skills`` ``provenance``, "local" spelled "agent"). Failure → 0 / "local"."""
    try:
        from tools.skill_usage import (
            _read_bundled_manifest_names, _read_hub_installed_names, activity_count, load_usage)
        records, bundled, hub = load_usage(), _read_bundled_manifest_names(), _read_hub_installed_names()
    except Exception as e:
        logger.debug("skill usage lookup unavailable: %s", e)
        return (lambda _name: 0), (lambda _name: "local")

    def usage(name: str) -> int:
        with contextlib.suppress(Exception):
            return activity_count(records.get(name) or {})
        return 0

    def origin(name: str) -> str:
        return "hub" if name in hub else "bundled" if name in bundled else "local"
    return usage, origin


_SLASH_COMPLETION_LIMIT = 30


def _rank_slash_completions(items: list[dict], usage, origin_of, *, browsing: bool, score_of=None) -> list[dict]:
    """Registry commands keep their order; only skills reorder: fuzzy ``score_of`` first, then most-used, then
    A-Z. The limit is spent PER KIND (a flat cut on a large install offered no skill at all). ``browsing``
    (bare ``/``) drops never-used bundled skills as noise; a typed query is SEARCHING — nothing pruned, only reordered."""
    def name_of(item: dict) -> str:
        return str(item.get("text", "")).strip().lstrip("/").lower()
    commands = [item for item in items if item.get("kind") != "skill"]
    skills = [item for item in items if item.get("kind") == "skill"]
    if browsing:
        skills = [item for item in skills if origin_of(name_of(item)) != "bundled" or usage(name_of(item)) > 0]
    skills.sort(key=lambda item: (
        *(() if score_of is None else (score_of(item),)), -usage(name_of(item)), name_of(item)))
    return commands[:_SLASH_COMPLETION_LIMIT] + skills[:_SLASH_COMPLETION_LIMIT]


# argv shapes that must not run headless in the gateway process → user hint.
_CLI_EXEC_BLOCKED = {
    ("setup",): "`hermes setup` needs a full terminal — run it outside the TUI",
    ("gateway",): "`hermes gateway` is long-running — run it in another terminal",
    ("sessions", "browse"): "`hermes sessions browse` is interactive — use /resume here, or run browse in another terminal",
    ("config", "edit"): "`hermes config edit` needs $EDITOR in a real terminal",
}


def _cli_exec_blocked(argv: list[str]) -> str | None:
    """Return user hint if this argv must not run headless in the gateway process."""
    if not argv:
        return "bare `hermes` is interactive — use `/hermes chat -q …` or run `hermes` in another terminal"
    head = tuple(a.lower() for a in argv[:2])
    return _CLI_EXEC_BLOCKED.get(head[:1]) or _CLI_EXEC_BLOCKED.get(head)


def _resolve_name(name: str) -> str:
    with contextlib.suppress(Exception):
        from hermes_cli.commands import resolve_command
        return r.name if (r := resolve_command(name)) else name
    return name


_paste_counter = 0


# mcp.servers.* handlers (methods_tools) resolve these BARE through this namespace.
from .mcp_rpc_helpers import (  # noqa: E402, F401
    reset_profile as _mcp_reset_profile,
    summarize_server as _mcp_summarize_server)


# ── Split @method handler modules (see method_ctx.py): imported last so every global the handlers close
# over exists; register() rebinds them onto this namespace.
from . import (  # noqa: E402
    methods_voice as _methods_voice, methods_browser as _methods_browser, methods_slash as _methods_slash,
    methods_complete_helpers as _methods_complete_helpers, session_auto_continue as _session_auto_continue,
    agent_callbacks as _agent_callbacks, session_history as _session_history,
    prompt_attachments as _prompt_attachments, session_notifications as _session_notifications,
    tool_progress as _tool_progress, change_watcher as _change_watcher,
    session_compression as _session_compression, model_switch as _model_switch,
    compute_host_bridge as _compute_host_bridge, session_workdir as _session_workdir,
    session_lifecycle as _session_lifecycle, session_reaper as _session_reaper,
    session_transports as _session_transports,
    methods_browser_control as _methods_browser_control, methods_bot_relay as _methods_bot_relay,
    methods_complete as _methods_complete, methods_config as _methods_config,
    methods_config_set as _methods_config_set, methods_images as _methods_images,
    methods_profiles as _methods_profiles, methods_prompt as _methods_prompt, methods_session as _methods_session,
    methods_tools as _methods_tools, prompt_turn as _prompt_turn, billing_view as _billing_view,
    methods_projects as _methods_projects, methods_session_foreign as _methods_session_foreign,
    methods_session_control as _methods_session_control, methods_subagents as _methods_subagents,
    methods_vault as _methods_vault, methods_free_tier as _methods_free_tier,
    methods_connectors as _methods_connectors)

for _m in (
    _session_transports, _session_reaper, _session_lifecycle, _session_workdir, _compute_host_bridge, _model_switch,
    _session_compression, _change_watcher, _tool_progress, _session_notifications,
    _prompt_attachments, _session_history, _agent_callbacks, _session_auto_continue,
    _methods_complete_helpers, _methods_slash, _methods_voice, _methods_browser,
    _methods_browser_control, _methods_session, _methods_prompt, _methods_config,
    _methods_config_set, _methods_complete, _methods_tools, _methods_profiles, _methods_images,
    _methods_bot_relay, _prompt_turn, _billing_view, _methods_projects, _methods_session_foreign,
    _methods_session_control, _methods_subagents, _methods_vault, _methods_free_tier, _methods_connectors):
    _m.register(sys.modules[__name__])
del _m
