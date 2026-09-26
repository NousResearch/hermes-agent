from profiles import names as profile_names
from profiles import paths as profile_paths
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
    get_hermes_home, get_hermes_home_override, get_process_hermes_home, profile_name_for_home,
    reset_hermes_home_override, set_hermes_home_override)
from hermes_cli.env_loader import load_hermes_dotenv
from utils import file_signature, is_truthy_value
from hermes_state_ids import new_session_id
from tools.environments.local import hermes_subprocess_env
from agent.replay_cleanup import canonicalize_replay_history
from agent.reasoning_effort import clamp_effort, route_supported_efforts
from agent.compaction_display import project_compaction_message_for_display  # noqa: F401
from agent.skill_commands import describe_skill_invocation  # noqa: F401
from agent.conversation_loop import INTERRUPT_WAITING_FOR_MODEL_PREFIX  # noqa: F401
from tui_gateway import git_probe
from tui_gateway._env import env_float, env_int
from tui_gateway.turn_marker import clear_turn_marker, read_turn_marker, record_turn_start  # noqa: F401
from tui_gateway.contracts import registry as _contracts
# User-facing copy shared with the split method modules (they close over this namespace).
from tui_gateway.user_messages import (  # noqa: F401
    AGENT_BUILD_ABANDONED, AGENT_MISSING_FOR_TURN, AGENT_STILL_STARTING, agent_init_failed_message, busy_message,
    resume_failed_message, turn_error_text)
from tui_gateway.transport import (FanoutTransport, StdioTransport, Transport, bind_transport,
                                   current_transport, reset_transport)

logger = logging.getLogger(__name__)

_hermes_home = _HERMES_HOME_AT_IMPORT = get_hermes_home()
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
_db = None
_db_error: str | None = None
_stdout_lock = threading.Lock()
_cfg_lock = threading.Lock()
# Shared profile UI metadata is updated concurrently by Desktop, mobile and pool RPCs; its
# compare/check/write transaction needs its own lock, not the unrelated config cache lock.
_profile_ui_meta_lock = threading.Lock()
_sessions_lock = threading.RLock()  # reentrant: _close_session_by_id may run under callers that already hold it
_cfg_cache: dict | None = None
_cfg_sig: tuple | None = None
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
    "setup.runtime_check", "setup.status", "free_tier.provision", "voice.toggle", "voice.record", "voice.tts", "wake.start",
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
        from runtime.subprocess_compat import windows_hide_flags
        # slash_worker runs the Hermes agent → needs provider credentials. Tier-1 secrets
        # (gateway/GitHub/infra) are still stripped (#29157). Global-remote / multi-profile sessions: the
        # worker must resolve config/skills/state against the session's profile home, not the gateway's
        # launch HERMES_HOME (#40677).
        from tools.environments.local import served_profile_child_env
        from agent.secret_scope import is_multiplex_active

        # The worker runs the agent → needs provider credentials; tier-1 secrets (gateway/GitHub/
        # infra) are still stripped. A served profile's worker gets THAT profile's home + secrets and
        # none of the launch profile's .env / TERMINAL_* residue, exactly what a standalone
        # `hermes -p X` would load itself. The launch profile is a profile too: once the process hosts
        # a second home (multiplex flipped), its worker must name its own home or the fail-closed
        # no-target/no-scope path raises UnscopedSecretError (#115427).
        env = _prepend_tool_paths(served_profile_child_env(
            target_home=profile_home or (_hermes_home if is_multiplex_active() else None),
            inherit_credentials=True))
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
    for step in (_flush_sessions_before_exit, _release_gateway_wake_owner, _stop_turns_before_exit):
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


def _launch_state_db_path() -> Path:
    """Launch profile's ``state.db`` at call time: the patched ``_hermes_home`` when a test changed
    it, else the live process home — resolved through :func:`get_process_hermes_home`, which honours
    ``HERMES_HOME`` but ignores the context-local override. The desktop multiplex cron ticker sets
    that override per profile at startup, and a first touch inside a foreign window would bind this
    process-wide handle to another profile's ``state.db`` (#102526). Resolving here rather than at
    import time lets a harness that redirects ``HERMES_HOME`` after import be honoured (#112692)."""
    home = _hermes_home if _hermes_home != _HERMES_HOME_AT_IMPORT else get_process_hermes_home()
    return Path(home) / "state.db"


def _get_db():
    global _db, _db_error
    if _db is None:
        from hermes_state_registry import acquire
        try:
            # Launch home, never the context-local override (#102526); resolved at first
            # use, not import time (#112692). See _launch_state_db_path.
            _db, _db_error = acquire(_launch_state_db_path()), None
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


def _bootstrap_served_profile_store(db_path: Path) -> None:
    """A served profile whose first RPC is a read (Bot Chat registry lookup on a fresh bot) has no
    ``state.db`` yet. Browsing never creates a store, but THIS process is the profile's session
    owner (``_open_profile_session_db`` acquires the writer for every agent build), so it seeds the
    store it is about to own instead of answering ``storage_unavailable`` until a turn happens."""
    try:
        if db_path.stat().st_size > 0:
            return
    except FileNotFoundError:
        pass
    from hermes_state_registry import acquire, release_or_close
    release_or_close(acquire(db_path))


@contextlib.contextmanager
def _profile_db(params: dict | None = None, *, writer: bool = False):
    """Yield the SessionDB for ``params['profile']`` (None when unavailable); closes dedicated
    profile handles, leaves the launch-profile shared handle open.

    Foreign-profile handles are read-only unless ``writer=True``: that store belongs to ITS
    gateway/dashboard, and a writer here would take its write lock per RPC. Mirrors
    hermes_cli.web_routers.profiles._read_profile_db."""
    profile = (params.get("profile") or "").strip() or None if isinstance(params, dict) else None
    # Launch/own profile → the shared _get_db() handle (left open); another profile → a dedicated
    # handle closed below (app-global remote mode). db is None when unavailable.
    if (profile_home := _profile_home(profile)) is None:
        db, owns = _get_db(), False
    else:
        try:
            if writer:
                from hermes_state_registry import acquire
                db = acquire(Path(profile_home) / "state.db")
            else:
                from hermes_cli.web_server_sessions import _open_session_db_at_path
                _bootstrap_served_profile_store(Path(profile_home) / "state.db")
                db = _open_session_db_at_path(Path(profile_home) / "state.db", read_only=True)
            owns = True
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
        # Check the profiles root directly: get_profile_dir rejects "hermes" as a
        # reserved name, but a pre-reserved-list install may still carry that dir.
        if not (profile_paths._get_profiles_root() / profile_names.normalize_profile_name(name)).is_dir():
            return "default"
    return name


def _response_profile_name(profile: str | None = None) -> str:
    """Profile name for session.* payloads: the requested real non-launch profile, else the launch one."""
    name = _canonical_profile_request((profile or "").strip())
    if not name:
        return _current_profile_name()
    try:
        return name if _profile_home(name) is not None else _current_profile_name()
    except ProfileUnavailableError:
        return _current_profile_name()


def _db_unavailable_error(rid, *, code: int):
    from hermes_state_user_copy import describe_storage_failure, storage_failure_details
    failure = describe_storage_failure(_db_error)
    return _err(
        rid, code,
        f"Session storage is unavailable: {failure.gloss}. {failure.action}",
        data={"code": failure.code, "cause": failure.cause, "details": storage_failure_details(_db_error)})


# ── Per-session profile scoping: the desktop's app-global remote mode points every profile at this
# backend, so calls carry ``profile`` → open that profile's db and bind its HERMES_HOME (ContextVar
# override) so config/skills/model/persistence resolve to it. Omitted/own profile → launch profile.
class ProfileUnavailableError(FileNotFoundError):
    """An explicit ``profile`` param names no live profile on this host. Raised out of the method
    (never a silent fall-back to the launch profile); ``handle_request`` turns it into JSON-RPC 4064
    so a client holding a deleted profile gets a typed error instead of a ws dispatch crash (#107829)."""


def _profile_home(profile: str | None) -> Path | None:
    """Resolve a named profile's home on THIS host, or None for the launch profile."""
    if not (name := _canonical_profile_request((profile or "").strip())):
        return None
    try:
        home = Path(profile_paths.get_profile_dir(name))
    except ValueError:
        home = None
    if home is None or not home.is_dir():
        raise ProfileUnavailableError(f"Profile '{name}' does not exist.")
    if home.resolve() == Path(_hermes_home).resolve():
        return None  # already the launch profile (no override needed)
    if home not in _served_profile_homes:
        # This process now hosts a second profile home: freeze the launch env as the launch
        # profile's own and flip get_secret() to fail closed, so an unscoped read for a
        # secondary raises instead of returning the launch profile's os.environ value
        # (tui_gateway/launch_profile_policy.py). Must run before any secondary code.
        from tui_gateway.launch_profile_policy import activate_multi_profile_hosting
        activate_multi_profile_hosting()
    _served_profile_homes.add(home)  # the change watcher must stat every served sibling store too
    return home


# Profile homes served besides the launch home — the only extra stores the sessions watcher
# probes. Empty on single-profile installs, so their watcher stays byte-identical.
_served_profile_homes: set[Path] = set()


def _profile_scoped(handler):
    """Bind ``params['profile']``'s full runtime scope (HERMES_HOME + secrets + terminal policy) around a
    handler, so config.yaml ``${VAR}`` refs, provider credential checks and ``.env`` writes resolve to
    THAT profile (app-global remote mode hits the focused profile). Home alone left ``get_secret`` on the
    launch process's ``os.environ``: ``config.get full`` for a secondary shipped the default profile's
    expanded secrets and ``config.set`` published a secondary's ``.env`` edit into the shared process env.

    Launch profile: unscoped while this is a single-profile process (legacy ``os.environ`` precedence,
    systemd / ``op run`` injection); once multiplexing is active it binds its own scope from the env
    frozen at activation (``_session_profile_runtime_scope``), never ambient state a secondary context
    might have poisoned (#107422).

    No ``profile`` param but a ``session_id`` naming a live session: that session's ``profile_home``
    is the scope. The TUI and the Desktop's ambient dispatcher send session-bound RPCs with the
    session id alone, so a ``config.set`` from a focused worker session otherwise persisted into the
    LAUNCH profile's config.yaml while the worker's stayed unchanged (#85669).
    """
    def wrapper(rid, params):
        p = params if isinstance(params, dict) else {}
        if str(p.get("profile") or "").strip():
            home = _profile_home(p.get("profile"))
            profile_home = str(home) if home else None
        else:
            session = _sessions.get(str(p.get("session_id") or ""))
            profile_home = session.get("profile_home") if isinstance(session, dict) else None
        with _session_profile_runtime_scope({"profile_home": profile_home or None}):
            return handler(rid, params)
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
    """A profile's ``terminal.cwd`` from ITS config.yaml (fail-open → None): the process-global
    ``TERMINAL_CWD`` belongs to the *launch* profile, and load_config() resolves the ACTIVE profile, so
    read the requested file through the same effective-config pipeline as ``_load_cfg``.

    A new session bound to another profile must take its workspace from THAT profile's config, not the stale
    env var (issue #40334). Returns an absolute, existing directory, or None for placeholders / missing /
    invalid paths.
    """
    if profile_home is None:
        return None
    with contextlib.suppress(Exception):
        from hermes_cli.config_effective import load_user_config_effective
        p = Path(profile_home) / "config.yaml"
        return _configured_cwd_from_cfg(load_user_config_effective(p)) if p.exists() else None
    return None


def _launch_configured_cwd() -> str | None:
    """Launch profile's ``terminal.cwd`` from config.yaml: the dashboard's in-memory gateway gets no bridged
    ``TERMINAL_CWD`` env (only the Node PTY child does), so a fresh /chat would otherwise start in ``os.getcwd()``."""
    # Read the launch file by path. ``_load_cfg`` follows the active HERMES_HOME
    # override, which may belong to a different profile-scoped RPC.
    return _profile_configured_cwd(Path(_hermes_home))


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
    params = obj.get("params")
    if obj.get("method") == "event" or (isinstance(obj.get("id"), str) and "method" in obj):
        # Event notifications AND server→client requests carry ``params.session_id``; both route to the
        # owning session's transport. A room member's hidden session has no transport: its frames would
        # die at stdio below.
        project_room_member_activity(obj, _sessions)
        sid = ((params or {}).get("session_id")) if isinstance(params, dict) else ""
        if sid and (t := (_sessions.get(sid) or {}).get("transport")) is not None:
            return t.write(obj)
    return (current_transport() or _stdio_transport).write(obj)


def _event_frame(event: str, sid: str, payload: dict | None = None) -> dict:
    _contracts.check_payload(event, payload)
    params: dict = {"type": event, "session_id": sid, **({"payload": payload} if payload is not None else {})}
    return {"jsonrpc": "2.0", "method": "event", "params": params}


def _emit(event: str, sid: str, payload: dict | None = None) -> bool:
    from agent.notification_presentation import event_presentation_muted
    if event_presentation_muted(event, sid):
        return False
    return write_json(_event_frame(event, sid, payload))


from tui_gateway import server_requests as _server_requests  # noqa: E402

_server_requests.bind_sinks(lambda frame: write_json(frame), lambda event, sid, payload: _emit(event, sid, payload),
                            lambda sid: _session_client_answers_requests(sid))


# Live WS peer transports (maintained by tui_gateway.ws): the only route for session-less background
# events, which write_json would otherwise drop on stdio (see _broadcast_global_event).
_live_transports: set[Transport] = set()
_live_transports_lock = threading.Lock()
# True only when real stdout IS the JSON-RPC client channel (``tui_gateway.entry.main``, the stdio TUI).
# `hermes serve` / dashboard processes speak JSON-RPC over WS only: their stdout is captured into
# desktop.log, so a peer-less global broadcast (the change watcher keeps ticking after the last WS client
# leaves) must be dropped there, not printed.
_stdio_is_rpc_channel = False


def register_live_transport(transport: Transport | None) -> None:
    """Track a connected client transport for global broadcasts. Idempotent."""
    if transport is not None:
        with _live_transports_lock:
            _live_transports.add(transport)


def unregister_live_transport(transport: Transport | None) -> None:
    """Stop tracking a transport (call on disconnect). Idempotent."""
    with _live_transports_lock:
        _live_transports.discard(transport)
    _server_requests.forget(transport)


def _broadcast_global_event(event: str, payload: dict | None = None) -> None:
    """Fan a session-less, surface-global event (``skin.changed``) to every connected client — background
    emitters bottom out at stdio in ``write_json``'s ladder. No registered transports → ``_emit`` when stdout is the
    stdio TUI's JSON-RPC channel, else dropped (nobody is listening; stdout is a log sink)."""
    with _live_transports_lock:
        targets = list(_live_transports)
    if not targets:
        if _stdio_is_rpc_channel:
            return _emit(event, "", payload)
        logger.debug("global-event broadcast dropped (no connected client) type=%s", event)
        return None
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


def _open_requests(sid: str) -> list[dict]:
    """Server→client requests still waiting on *sid*'s renderer, for reconnect snapshots (``session.resume`` /
    ``session.activate`` / ``session.events.since``). A client detached when the request frame was written would
    otherwise never see it (agent parked until timeout). Under turn isolation the compute-host child owns the
    request; the parent mirrors it from the relayed frame (compute_host_bridge)."""
    from tui_gateway import server_requests
    reqs = server_requests.open_requests(sid)
    if reqs:
        return reqs
    if (session := _sessions.get(sid)) is not None:
        with session.get("history_lock", threading.Lock()):
            mirrored = session.get("_compute_host_open_request")
            return [dict(mirrored)] if isinstance(mirrored, dict) else []
    return []


def _pending_connection_request_payload(sid: str) -> dict | None:
    """The open connection operation on *sid* as its ``connection.request`` payload, so a client
    that missed the event (or restarted) restores the card with the server's deadline."""
    from tools.connectors import live

    session = _sessions.get(sid)
    operation = (live.current(str(session.get("session_key") or ""), profile_home=session.get("profile_home"))
                 if session else None)
    return operation.request_payload() if operation is not None else None


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
    """Send an ``approval`` server request with the command redacted: a credential-shaped value Tirith flagged
    would otherwise echo verbatim to the TUI (third egress alongside chat platforms and the SSE/API stream).
    See #48456, #50767.

    The wait is owned by ``tools.approval``'s queue (its own timeout, ``/approve all``, coalescing), so the request
    is queue-backed: the response resolves the queue entry, and the entry's own resolution (any surface, timeout,
    interrupt) withdraws the request with ``request.cancel``."""
    from tui_gateway import server_requests
    from tools import approval as _approval
    payload = _approval_request_payload(data)
    request_id = str(payload.get("request_id") or "")
    session_key = str((_sessions.get(sid) or {}).get("session_key") or "")

    def on_result(result: dict | None) -> None:
        if result is None:
            # No client can answer this prompt: the request was never sent (the only attached client predates
            # server→client requests) or the client answered -32601 (no handler). Without withdrawing the
            # queue entry the agent would idle for the whole approvals.timeout with no prompt anywhere
            # (#112548). A withdrawal, not a deny: nobody refused the command.
            if request_id:
                _approval.withdraw_gateway_approval(session_key, request_id,
                                                    "the attached client cannot answer approval requests "
                                                    "(update the Hermes app)")
            return
        choice = str(result.get("choice") or "deny")
        _approval.resolve_gateway_approval(session_key, choice, resolve_all=bool(result.get("all")),
                                           request_id=request_id or None)

    settle = server_requests.send_async("approval", sid, payload, on_result)
    # The wait can end between the frame going out and the hook attaching (the client answered by RPC, another
    # surface resolved it): withdraw now, or the request stays in ``open_requests`` for every later resume.
    if request_id and not _approval.register_gateway_settle(session_key, request_id, settle):
        settle("resolved")


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


def register_method(name: str, fn) -> None:
    """The ONE registration seam (``@method`` here and ``HandlerRegistry.install`` for the split
    modules). ``tests/tui_gateway/contracts/test_generated.py::test_every_method_has_a_contract`` and the
    generator's ``assert_complete`` fail when a registered name has no contract."""
    _methods[name] = fn


def method(name: str):
    def dec(fn):
        register_method(name, fn)
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
    global _cfg_cache, _cfg_sig, _cfg_path
    from hermes_cli.config import read_user_config_raw
    from hermes_cli.config_read_errors import FailedConfigRead
    try:
        p = _active_config_path()
        sig = file_signature(p.stat()) if p.exists() else None
        with _cfg_lock:
            if _cfg_cache is not None and _cfg_sig == sig and _cfg_path == p:
                return copy.deepcopy(_cfg_cache)
        data = read_user_config_raw(p) if p.exists() else {}
    except Exception as exc:
        return FailedConfigRead(error=exc)  # readable as {}, refused by _save_cfg
    with _cfg_lock:  # cache the RAW config: _save_cfg writes _cfg_cache back to disk
        _cfg_cache, _cfg_sig, _cfg_path = copy.deepcopy(data), sig, p
    return data


def _load_cfg() -> dict:
    """Behavioral config read: the effective USER config (managed overlay, ``${VAR}`` expansion, model-key
    canon) minus the DEFAULT_CONFIG merge — callers treat a missing key as "unset", so merging would break
    ``_load_cfg() == {}`` sentinels. Fail-open to ``{}``. Never pass the result to ``_save_cfg`` (use
    ``_load_cfg_raw()``)."""
    with contextlib.suppress(Exception):
        from hermes_cli.config_effective import load_user_config_effective
        return load_user_config_effective(_active_config_path())
    return {}


def _save_cfg(cfg: dict):
    global _cfg_cache, _cfg_sig, _cfg_path
    from hermes_cli.config import atomic_config_write
    path = _active_config_path()
    atomic_config_write(path, cfg)
    with _cfg_lock:
        _cfg_cache, _cfg_path = copy.deepcopy(cfg), path
        try:
            _cfg_sig = file_signature(path.stat())
        except Exception:
            _cfg_sig = None


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
        profile = _current_profile_name()
        browser_control_principal = browser_control_transport_family = ""
        # Live conversation id for subprocess HERMES_SESSION_ID: an explicitly empty contextvar is authoritative
        # (no os.environ fallback), so never leave it "" — agent's durable session_id, then session_key.
        session_id = session_key
        if sess is not None:
            source = _session_source(sess)
            # App-global backends multiplex profiles: prefer the live session's own home.
            profile = profile_name_for_home(sess.get("profile_home")) or profile
            session_id = getattr(sess.get("agent"), "session_id", None) or session_key
            identity = getattr(sess.get("transport"), "auth_identity", None)
            if _methods_browser_control._is_authenticated_identity(identity):
                browser_control_principal = _methods_browser_control._principal_digest(identity)
                browser_control_transport_family = _methods_browser_control._CLOUD_TRANSPORT_FAMILY
        return set_session_vars(
            session_key=session_key, session_id=session_id, source=source,
            profile=profile,
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


def _ask(method: str, sid: str, params: dict, timeout: float | None = 300) -> str:
    """Server→client request whose answer is one string under ``value`` (sudo, secret, vault prompts, GUI reads,
    MCP setup). Empty string when the renderer skipped, timed out, or was cancelled."""
    from tui_gateway import server_requests
    result = server_requests.send(method, sid, params, timeout=timeout)
    value = (result or {}).get("value", "")
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)



def _clarify_timeout_seconds() -> float | None:
    """Clarify wait for the TUI/desktop bridge from the canonical config (gateway/CLI parity); 300s
    historical default if config can't be read; ``<= 0`` = unlimited → None (never auto-skip)."""
    with contextlib.suppress(Exception):
        from tools.clarify_gateway import get_clarify_timeout
        timeout = get_clarify_timeout()
        return timeout if timeout > 0 else None
    return 300


def _clarify_block(sid: str, q, c, multi_select=False, questions=None) -> str:
    """Bridge the clarify tool callback onto a ``clarify`` server request. Single question: the response is
    ``{"answer"}`` ("" = skip). Batch: one request with only the wire fields (tool-side entries carry
    result-assembly keys too); answers lock one at a time through ``clarify.lock`` and the tool gets
    ``{"answers", "timed_out"?}`` as JSON — a response with no ``answers`` is a cancel-all."""
    from tui_gateway import server_requests
    if questions:
        wire = [{"qid": e["qid"], "question": e["question"], "choices": e["choices"], "multi_select": bool(e["multi_select"])}
                for e in questions]
        result = server_requests.send("clarify", sid, {"questions": wire}, timeout=_clarify_timeout_seconds(),
                                      qids=[e["qid"] for e in questions])
        if not result or "answers" not in result:
            return ""
        return json.dumps(result, ensure_ascii=False)
    params = {"question": q, "choices": c, "multi_select": True} if multi_select else {"question": q, "choices": c}
    result = server_requests.send("clarify", sid, params, timeout=_clarify_timeout_seconds())
    answer = (result or {}).get("answer", "")
    return answer if isinstance(answer, str) else ""


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
    """Bridge the tour tool callback onto a ``tour`` server request without paying for a client that cannot answer: against
    an older app nobody answers ``tour`` and each action would block the full deadline, stacking per
    turn. First action per session gets the short probe deadline; unanswered → bridge marked unavailable
    for that session; once answered, the full deadline. Verdict lives on the record, so a new session re-probes.

    The renderer's ``tour`` handler ships in the desktop bundle, but the tool is offered by this
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
    answer = _ask("tour", sid, dict(payload),
                  timeout=_TOUR_TIMEOUT_S if state == "answered" else _TOUR_PROBE_TIMEOUT_S)
    if answer:
        session["tour_bridge"] = "answered"
    elif state != "answered":
        session["tour_bridge"] = "unanswered"
    return answer or _TOUR_BRIDGE_UNAVAILABLE


def _clear_pending(sid: str | None = None) -> None:
    """Withdraw open server→client requests: only *sid*'s (session.interrupt must not cancel other sessions'
    prompts), or every one when *sid* is None (process exit). Each one gets a ``request.cancel``."""
    from tui_gateway import server_requests
    server_requests.cancel(sid, reason="interrupted" if sid else "shutdown")


# ── Agent factory ────────────────────────────────────────────────────




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
    # Re-bind the session's profile runtime scope (the build's finally reset it → root profile's SOUL.md/skills,
    # #50233) and session context (on the RPC thread _SESSION_CWD is unset → the process TERMINAL_CWD would
    # persist). The full scope, not HERMES_HOME alone: the external memory provider's system_prompt_block()
    # reads its credential through get_secret, which fails closed once this process multiplexes (#112927).
    session_tokens = _set_session_context(session_key, cwd=_session_cwd(session))
    try:
        with _session_profile_runtime_scope(session):
            prompt = agent._cached_system_prompt = agent._build_system_prompt(None)
        db.update_system_prompt(getattr(agent, "session_id", None) or session_key, prompt)
    except Exception:
        logger.warning("failed to persist live session system prompt for session %s", session_key, exc_info=True)
    finally:
        _clear_session_context(session_tokens)


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
    entry: dict[str, Any] = {"role": "user", "content": marker, "display_kind": "model_switch"}
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
                from agent.context_compressor import _DB_PERSISTED_MARKER
                entry["_row_id"] = db.append_message(
                    session_id=session_key, role="user", content=marker, display_kind="model_switch")
                entry[_DB_PERSISTED_MARKER] = True
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
        from profiles.current import get_active_profile_name
        return get_active_profile_name() or "default"
    return "default"


# Monotonic GUI<->backend contract version: the desktop refuses a backend reporting less (or none) with a
# one-click "update to align" prompt; bump whenever the desktop's backend contract changes. v2 file.attach;
# v3 approvals.mode RPCs + session.info reconciliation; v4 session.create fast=false = explicit normal tier;
# v5 ws_max_size >16 MiB file.attach frames; v6 plugins.manage rows carry the canonical registry key;
# v7 blocking prompts are JSON-RPC server->client requests (`srq-<n>` frames, `open_requests` replay) — a v6
# backend still emits `<kind>.request` notifications the renderer no longer listens for.
DESKTOP_BACKEND_CONTRACT = 8


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


def _live_session_identity(session: dict) -> tuple[str, str]:
    """``(model, provider)`` the live session actually runs — the same precedence ``_session_info`` reports:
    a switch queued mid-turn, the metadata mirror, the built agent, the composer override a deferred record
    carries. The profile default is the LAST resort, never the answer for a chat that made its own pick."""
    pending = session.get("pending_model_switch") or {}
    mirror = _metadata_mirror(session)
    agent = session.get("agent")
    override = session.get("model_override") or {}
    model = (str(pending.get("display_model") or "").strip() or mirror.get("model")
             or getattr(agent, "model", "") or override.get("model") or _session_default_model(session))
    provider = (str(pending.get("display_provider") or "").strip() or mirror.get("provider")
                or getattr(agent, "provider", "") or override.get("provider") or "")
    return str(model), str(provider or "")


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
    model = pending_model or mirror.get("model", getattr(agent, "model", ""))
    # The level the route's entry clamp actually sends (== reasoning_effort when verbatim), so the
    # Desktop can say "ultra sends max on this route" like `/reasoning` does instead of presenting a
    # Hermes-internal step (#61634) as a wire level the route does not have.
    reasoning_effort_wire = ""
    if reasoning_effort and reasoning_effort != "none":
        reasoning_effort_wire = str(clamp_effort(reasoning_effort, route_supported_efforts(pending_provider or provider, model)) or "")
    info: dict = {
        "model": model,
        "provider": pending_provider or provider,
        "reasoning_effort": reasoning_effort, "reasoning_effort_wire": reasoning_effort_wire,
        "service_tier": service_tier, "fast": service_tier == "priority",
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


# Commands that queue onto _pending_input in the CLI; the slash worker has no reader for that queue, so
# slash.exec routes them to command.dispatch instead.
_PENDING_INPUT_COMMANDS: frozenset[str] = frozenset({
    "retry", "queue", "q", "steer", "plan", "goal", "loop", "proactive", "moa", "undo", "learn",
    "init", "compress", "compact",
})

_WORKER_BLOCKED_COMMANDS: frozenset[str] = frozenset({"snapshot", "snap"})


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


# mcp.servers.* handlers (methods_tools) resolve this BARE through this namespace.
from .mcp_rpc_helpers import summarize_server as _mcp_summarize_server  # noqa: E402, F401


# ── Split @method handler modules (see method_ctx.py): imported last so every global the handlers close
# over exists; register() rebinds them onto this namespace.
from hermes_state import _BARE_BILLING_PROVIDERS  # noqa: E402, F401

from . import (  # noqa: E402
    session_registry as _session_registry, agent_factory as _agent_factory,
    methods_voice as _methods_voice, methods_browser as _methods_browser, methods_slash as _methods_slash,
    methods_complete_helpers as _methods_complete_helpers, session_auto_continue as _session_auto_continue,
    plugin_inject as _plugin_inject,
    rpc_dispatch as _rpc_dispatch,
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
    methods_connectors as _methods_connectors, methods_connectors_account as _methods_connectors_account,
    methods_display as _methods_display, methods_display_watch as _methods_display_watch,
    methods_onboarding as _methods_onboarding)

for _m in (
    _session_registry, _agent_factory,
    _session_transports, _session_reaper, _session_lifecycle, _session_workdir, _compute_host_bridge, _model_switch,
    _session_compression, _change_watcher, _tool_progress, _session_notifications,
    _prompt_attachments, _session_history, _agent_callbacks, _session_auto_continue, _plugin_inject, _rpc_dispatch,
    _methods_complete_helpers, _methods_slash, _methods_voice, _methods_browser,
    _methods_browser_control, _methods_session, _methods_prompt, _methods_config,
    _methods_config_set, _methods_complete, _methods_tools, _methods_profiles, _methods_images,
    _methods_bot_relay, _prompt_turn, _billing_view, _methods_projects, _methods_session_foreign,
    _methods_session_control, _methods_subagents, _methods_vault, _methods_free_tier, _methods_connectors,
    _methods_connectors_account, _methods_display, _methods_display_watch, _methods_onboarding):
    _m.register(sys.modules[__name__])
del _m
