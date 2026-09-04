"""Use the Browser Use CLI 3.0 (https://browser-use.com) for browser automation

When browser.backend is "browser-use", the model gets ``browser_exec`` tool
instead of default browser tools
"""

import hashlib
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils import is_truthy_value

logger = logging.getLogger(__name__)

_BACKEND_KEY = "browser-use"
BACKEND_DISABLED = "off"

# Cloud daemon names become the BU_NAME env var
_SESSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

# Internal marker set by _resolve_backend_cdp on the env dict when the
# resolved browser is EXCLUSIVE to this named session (per-name provider
# browser, or a named Browser Use cloud browser). Popped before the
# subprocess launches — never exported to the CLI.
_PRIVATE_BROWSER_SENTINEL = "_HERMES_BU_PRIVATE_BROWSER"

# Logical sessions on a shared physical daemon must reselect their own tab on
# every call. Another logical session may have changed the daemon's current tab.
_OWN_TAB_PREAMBLE = """\
# hermes: select this logical session's tab on the shared CDP transport
def _hermes_ensure_own_tab():
    import hashlib as _hashlib, json as _json, os as _os, tempfile as _tf
    _name = _os.environ.get("HERMES_BU_LOGICAL_SESSION") or _os.environ.get("BU_NAME", "default")
    _runtime = _os.environ.get("BH_RUNTIME_DIR") or _tf.gettempdir()
    _slot = _os.path.join(_runtime, "hermes-tab-%s.json" % _hashlib.sha256(_name.encode()).hexdigest()[:20])
    _tid = None
    try:
        with open(_slot, "r", encoding="utf-8") as _fh:
            _tid = _json.load(_fh).get("target_id")
    except Exception:
        pass
    try:
        _live = {t.get("targetId") for t in cdp("Target.getTargets").get("targetInfos", []) if t.get("type") == "page"}
    except Exception:
        _live = set()
    if _tid not in _live:
        try:
            _tid = cdp("Target.createTarget", url="about:blank").get("targetId")
        except Exception:
            _tid = None
        if _tid:
            try:
                _os.makedirs(_runtime, mode=0o700, exist_ok=True)
                _fd, _tmp = _tf.mkstemp(prefix=".hermes-tab-", dir=_runtime)
                with _os.fdopen(_fd, "w", encoding="utf-8") as _fh:
                    _json.dump({"target_id": _tid}, _fh)
                _os.replace(_tmp, _slot)
            except OSError:
                pass
    if _tid:
        try:
            switch_tab(_tid)
        except Exception:
            pass
_hermes_ensure_own_tab()
del _hermes_ensure_own_tab
"""

_DEFAULT_TIMEOUT_S = 300
_MIN_TIMEOUT_S = 5
_MAX_TIMEOUT_S = 1800
_MAX_LEASE_MINUTES = 120
_STDERR_CAP_CHARS = 4000

# Filesystem-safe task ids for per-task workspace dirs.
_TASK_ID_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")

# Screenshot paths printed by capture_screenshot() in the exec output.
# Two alternatives: POSIX absolute (/tmp/shot.png) and Windows drive-letter
# absolute (C:\Users\...\shot.png or C:/Users/.../shot.png). Browser Use on
# Windows prints native paths — the POSIX-only pattern silently dropped them
# and screenshot_path / the multimodal attach never fired (#83884).
_IMAGE_PATH_RE = re.compile(
    r"((?:[A-Za-z]:[\\/]|/)[^\s\"']+?\.(?:png|jpe?g|webp))", re.IGNORECASE
)

# http(s) URL literals in exec code checked against browser_navigate's policy
_URL_RE = re.compile(r"https?://[^\s'\"\\)]+", re.IGNORECASE)

try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - Windows
    _fcntl = None

try:
    import msvcrt as _msvcrt
except ImportError:  # pragma: no cover - POSIX
    _msvcrt = None


@dataclass(frozen=True)
class _SessionTransport:
    key: str
    runtime_dir: Path
    tmp_dir: Path
    thread_lock: threading.RLock


_TRANSPORT_LOCK_GUARD = threading.Lock()
_TRANSPORT_THREAD_LOCKS: Dict[str, threading.RLock] = {}


def _private_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    try:
        path.chmod(0o700)
    except OSError:
        pass
    return path


def _configure_session_transport(
    env: dict, *, conversation_id: str, logical_session: str, private_browser: bool
) -> Optional[_SessionTransport]:
    """One browser-harness daemon per profile, conversation lineage, endpoint."""
    scope = str(conversation_id or "").strip()
    if not scope or private_browser:
        return None
    try:
        from hermes_constants import get_hermes_home

        home = Path(get_hermes_home()).expanduser().resolve()
    except Exception:
        return None
    endpoint = str(env.get("BU_CDP_URL") or env.get("BU_CDP_WS") or "local-auto")
    key = hashlib.sha256(f"{home}\0{scope}\0{endpoint}".encode()).hexdigest()[:24]
    tmp_dir = _private_dir(home / "cache" / "bu" / key / "tmp")
    if os.name == "nt":
        runtime_dir = _private_dir(home / "cache" / "bu" / key / "run")
    else:
        uid = os.getuid() if hasattr(os, "getuid") else 0
        runtime_dir = _private_dir(Path(tempfile.gettempdir()) / f"hermes-bu-{uid}" / key)
        if len(os.fsencode(str(runtime_dir / "bu.sock"))) >= 100:
            runtime_dir = _private_dir(Path("/tmp") / f"hermes-bu-{uid}" / key)
    env["BH_RUNTIME_DIR"] = str(runtime_dir)
    env["BH_TMP_DIR"] = str(tmp_dir)
    env["HERMES_BU_LOGICAL_SESSION"] = str(logical_session or "default")
    with _TRANSPORT_LOCK_GUARD:
        lock = _TRANSPORT_THREAD_LOCKS.setdefault(key, threading.RLock())
    return _SessionTransport(key, runtime_dir, tmp_dir, lock)


class _TransportExecutionLock:
    """Thread + cross-process lock for one daemon's mutable current target."""

    def __init__(self, transport: _SessionTransport, timeout_s: float):
        self.transport = transport
        self.timeout_s = max(0.1, float(timeout_s))
        self.handle = None
        self.thread_held = False

    def acquire(self) -> None:
        deadline = time.monotonic() + self.timeout_s
        if not self.transport.thread_lock.acquire(timeout=self.timeout_s):
            raise TimeoutError("another browser_exec call owns this CDP transport")
        self.thread_held = True
        try:
            path = self.transport.runtime_dir / "exec.lock"
            self.handle = open(path, "a+b")
            try:
                os.chmod(path, 0o600)
            except OSError:
                pass
            if _fcntl is not None:
                while True:
                    try:
                        _fcntl.flock(
                            self.handle.fileno(), _fcntl.LOCK_EX | _fcntl.LOCK_NB
                        )
                        return
                    except BlockingIOError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                "another process owns this browser CDP transport"
                            )
                        time.sleep(0.05)
            if _msvcrt is not None:  # pragma: no cover - Windows
                self.handle.seek(0)
                if self.handle.read(1) == b"":
                    self.handle.write(b"0")
                    self.handle.flush()
                while True:
                    try:
                        self.handle.seek(0)
                        getattr(_msvcrt, "locking")(
                            self.handle.fileno(), getattr(_msvcrt, "LK_NBLCK"), 1
                        )
                        return
                    except OSError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                "another process owns this browser CDP transport"
                            )
                        time.sleep(0.05)
            return
        except Exception:
            self.release()
            raise

    def release(self) -> None:
        if self.handle is not None:
            if _fcntl is not None:
                try:
                    _fcntl.flock(self.handle.fileno(), _fcntl.LOCK_UN)
                except OSError:
                    pass
            elif _msvcrt is not None:  # pragma: no cover - Windows
                try:
                    self.handle.seek(0)
                    getattr(_msvcrt, "locking")(
                        self.handle.fileno(), getattr(_msvcrt, "LK_UNLCK"), 1
                    )
                except OSError:
                    pass
            try:
                self.handle.close()
            except OSError:
                pass
            self.handle = None
        if self.thread_held:
            self.thread_held = False
            self.transport.thread_lock.release()


def _probe_daemon_pid(runtime_dir: Path) -> Optional[int]:
    """Return the live harness daemon PID after an authenticated IPC ping."""
    connection: Optional[socket.socket] = None
    try:
        request: Dict[str, Any] = {"meta": "ping"}
        if os.name == "nt":
            payload = json.loads(
                (runtime_dir / "bu.port").read_text(encoding="utf-8")
            )
            port = int(payload["port"])
            token = str(payload["token"])
            if not token:
                return None
            request["token"] = token
            connection = socket.create_connection(("127.0.0.1", port), timeout=1.0)
        else:
            connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            connection.settimeout(1.0)
            connection.connect(str(runtime_dir / "bu.sock"))
        connection.sendall((json.dumps(request) + "\n").encode("utf-8"))
        raw = b""
        while not raw.endswith(b"\n") and len(raw) <= 65536:
            chunk = connection.recv(65536)
            if not chunk:
                break
            raw += chunk
        response = json.loads(raw or b"{}")
        if not isinstance(response, dict) or response.get("pong") is not True:
            return None
        pid = response.get("pid")
        return pid if type(pid) is int and 0 < pid < (1 << 31) else None
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None
    finally:
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass


def _daemon_identity(runtime_dir: Path) -> Optional[str]:
    try:
        path = runtime_dir / "bu.pid"
        recorded_pid = int(path.read_text(encoding="utf-8").strip())
        live_pid = _probe_daemon_pid(runtime_dir)
        if live_pid is None or live_pid != recorded_pid:
            return None
        return hashlib.sha256(
            f"{live_pid}:{path.stat().st_mtime_ns}".encode()
        ).hexdigest()[:20]
    except (OSError, ValueError):
        return None


def _record_transport_outcome(
    transport: _SessionTransport,
    before: Optional[str],
    after: Optional[str],
    logical_session: str,
    success: bool,
    error_text: str,
) -> Dict[str, Any]:
    """Local content-free attach telemetry; never records endpoint or raw IDs."""
    path = transport.tmp_dir / "transport-telemetry.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema_version") != 1:
            raise ValueError("old schema")
    except (OSError, ValueError, AttributeError):
        data = {
            "schema_version": 1,
            "attach_attempts": 0,
            "attach_successes": 0,
            "reuses": 0,
            "drops": 0,
            "events": [],
        }
    attached = False
    reused = False
    event = "unknown"
    reason = ""
    lowered_error = error_text.lower()
    permission_failed = (
        "permission-blocked" in lowered_error
        or "allow remote debugging" in lowered_error
    )
    if before and after == before:
        data["reuses"] += 1
        reused = True
        event = "reuse"
    elif after and after != before:
        data["attach_attempts"] += 1
        data["attach_successes"] += 1
        attached = True
        event = "attach_success"
        reason = "cold_start" if before is None else "daemon_replaced"
        if before is not None:
            data["drops"] += 1
    elif permission_failed:
        data["attach_attempts"] += 1
        attached = True
        event = "attach_failed"
        reason = "permission_blocked"
        if before is not None:
            data["drops"] += 1
    elif before and after is None:
        data["drops"] += 1
        event = "drop_detected"
        reason = "daemon_exited"
    events = data.get("events", [])
    events.append(
        {
            "at_unix_ms": int(time.time() * 1000),
            "event": event,
            "reason": reason,
            "outcome": "success" if success else "error",
        }
    )
    data["events"] = events[-100:]
    data["logical_session_hash"] = hashlib.sha256(
        (logical_session or "default").encode()
    ).hexdigest()[:16]
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temp.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
        os.replace(temp, path)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass
    return {
        "scope": "conversation_lineage",
        "transport_key": transport.key,
        "attach_count": data["attach_attempts"],
        "attach_success_count": data["attach_successes"],
        "attached_this_call": 1 if attached else 0,
        "reuse_count": data["reuses"],
        "reused": reused,
        "drop_count": data["drops"],
    }


def _blocked_url_in_code(code: str) -> Optional[str]:
    """Return an error if a URL literal fails the built-in navigation checks."""
    from tools.browser_tool import evaluate_url_safety

    for url in _URL_RE.findall(code or ""):
        err = evaluate_url_safety(url)
        if err:
            return err.get("error", "Blocked: unsafe URL")
    return None


def _base_subprocess_env() -> dict:
    from tools.browser_tool import _build_browser_env

    env = _build_browser_env()
    # The browser-use CLI runs under its own Python (uv tool / uvx), which
    # may differ from Hermes's venv Python. PYTHONPATH/PYTHONHOME inherited
    # from the agent process point at Hermes's venv site-packages, and a
    # child interpreter honors them ahead of its own site-packages — so the
    # CLI imports compiled C-extensions (e.g. pydantic_core) built for the
    # wrong interpreter and crashes on ABI mismatch (#83427, #84841, #86006,
    # #86104). Strip both — the CLI manages its own environment and never
    # needs Hermes's import path.
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    # Same class of hazard, PATH flavor: profile-spawned workers (kanban
    # bots, cron jobs) can hand down a PATH of only version-manager dirs,
    # which kills the uv trampoline before the CLI's Python starts. Floor
    # the PATH so coreutils are always reachable (see below).
    env["PATH"] = _floor_subprocess_path(env.get("PATH", ""))
    env.setdefault("ANONYMIZED_TELEMETRY", "false")
    return env


def _floor_subprocess_path(path: str) -> str:
    """Guarantee core system dirs survive onto the CLI subprocess PATH.

    Profile workers can inherit a PATH holding only version-manager dirs
    (observed: the nvm node dir repeated 7x, nothing else). That is fatal
    for the uv-installed browser-use binary: its POSIX sh trampoline
    resolves ``dirname``/``realpath`` through PATH, so without /usr/bin it
    dies with ``realpath: not found … exec: /python: not found`` (exit
    127) before its own Python ever starts. Reuses browser_tool's
    ``_merge_browser_path`` floor — same hazard, same sane-dir list — and
    falls back to appending FHS bin dirs if that import is unavailable.
    Windows .cmd shims don't trampoline through PATH, so no-op there.
    """
    if os.name == "nt":
        return path
    try:
        from tools.browser_tool import _merge_browser_path

        return _merge_browser_path(path or "")
    except Exception:
        pass
    parts = [p for p in (path or "").split(os.pathsep) if p]
    existing = set(parts)
    for directory in (
        "/usr/local/sbin",
        "/usr/local/bin",
        "/usr/sbin",
        "/usr/bin",
        "/sbin",
        "/bin",
    ):
        if directory not in existing and os.path.isdir(directory):
            parts.append(directory)
    return os.pathsep.join(parts)


def _read_browser_cfg() -> dict:
    """Return the ``browser:`` config section, or {} on any failure."""
    try:
        from hermes_cli.config import cfg_get, read_raw_config

        cfg = cfg_get(read_raw_config(), "browser", default={})
        return cfg if isinstance(cfg, dict) else {}
    except Exception as e:
        logger.debug("Could not read browser config section: %s", e)
        return {}


def _resource_hygiene_enabled() -> bool:
    """Compatibility shim; browser_tool owns tab-lifecycle configuration."""
    try:
        from tools.browser_tool import _tab_lifecycle_enabled

        return _tab_lifecycle_enabled()
    except Exception:
        return False


def get_browser_backend() -> str:
    """Return the configured browser backend key ("" = unset → default).

    YAML 1.1 parses an unquoted ``off`` as boolean False — a hand-edited
    ``backend: off`` must mean BACKEND_DISABLED, not "unset". (True has no
    sensible backend meaning; normalize it to unset.)
    """
    raw = _read_browser_cfg().get("backend")
    if raw is False:
        return BACKEND_DISABLED
    if raw is True:
        return ""
    return str(raw or "").strip().lower()


def is_legacy_browser_use_cloud_config(browser_cfg: dict) -> bool:
    """True for pre-CLI direct-API Browser Use cloud configs"""
    if not isinstance(browser_cfg, dict):
        return False
    if browser_cfg.get("backend"):
        return False  # an explicit backend choice wins
    provider = str(browser_cfg.get("cloud_provider") or "").strip().lower()
    if provider not in {"browser-use", ""}:
        return False  # explicit local/Browserbase/… choices win
    if is_truthy_value(browser_cfg.get("use_gateway"), default=False):
        return False
    # Camofox is selected via env var, not cloud_provider — a Camofox user
    # with a stray BROWSER_USE_API_KEY must keep their explicit choice.
    try:
        from tools.browser_camofox import is_camofox_mode

        if is_camofox_mode():
            return False
    except Exception as e:
        logger.debug("Camofox activity check failed during migration: %s", e)
    return bool(os.getenv("BROWSER_USE_API_KEY"))


def is_browser_use_cli_mode() -> bool:
    """True when the Browser Use CLI replaces the built-in browser stack.

    Browser Use mode is the DEFAULT: an unset ``browser.backend`` ("") enables
    it whenever the browser-use CLI is runnable (installed binary or uvx).
    Set ``browser.backend: off`` (or ``/browser use off``) for the built-in
    browser_* tools.

    Camofox always falls back to the built-in tools regardless of
    ``browser.backend`` — it is Firefox-based with a custom HTTP API and no
    CDP surface, so the CDP-only browser-use harness cannot drive it.
    """
    try:
        from tools.browser_camofox import is_camofox_mode

        if is_camofox_mode():
            return False
    except Exception as e:
        logger.debug("Camofox activity check failed: %s", e)
    backend = get_browser_backend()
    if backend:
        return backend == _BACKEND_KEY
    if is_legacy_browser_use_cloud_config(_read_browser_cfg()):
        return True
    # Default (backend unset): Browser Use mode when the CLI can run at all;
    # otherwise keep the built-in tools so browsing never silently breaks.
    return _find_cli() is not None


_NOTICE_STAMP_NAME = ".browser_use_default_notice"
_NOTICE_INTERVAL_S = 24 * 3600


def default_downgrade_notice() -> Optional[str]:
    """One-line notice when the default Browser Use backend silently downgraded.

    Returns the notice string when ``browser.backend`` is unset (Browser Use
    would be the default) but the CLI is not runnable, so the session fell
    back to the built-in browser tools. Rate-limited to once per 24h via a
    stamp file so it nudges without nagging. Returns ``None`` otherwise.
    """
    try:
        if get_browser_backend():
            return None  # explicit choice — nothing downgraded
        try:
            from tools.browser_camofox import is_camofox_mode

            if is_camofox_mode():
                return None
        except Exception:
            pass
        if _find_cli() is not None:
            return None

        from hermes_constants import get_hermes_home

        stamp = Path(get_hermes_home()) / "cache" / _NOTICE_STAMP_NAME
        try:
            if 0 <= time.time() - stamp.stat().st_mtime < _NOTICE_INTERVAL_S:
                return None
        except OSError:
            pass
        try:
            stamp.parent.mkdir(parents=True, exist_ok=True)
            stamp.touch()
        except OSError:
            pass
        return (
            "Browser Use CLI not found — using the built-in browser tools. "
            "Run `hermes tools` (Browser Automation → Browser Use) to install it, "
            "or `browser.backend: off` in config.yaml to silence this."
        )
    except Exception as e:  # pragma: no cover — a notice must never break startup
        logger.debug("browser-use downgrade notice failed: %s", e)
        return None


def _managed_bin_dir() -> Optional[str]:
    """Hermes' own bin dir ($HERMES_HOME/bin) — where install.sh puts uv/uvx
    and where install_cli() links the browser-use binary."""
    try:
        from hermes_constants import get_hermes_home

        return str(Path(get_hermes_home()) / "bin")
    except Exception as e:  # pragma: no cover — defensive
        logger.debug("Could not resolve managed bin dir: %s", e)
        return None


def _user_local_bin_dir() -> Optional[str]:
    """The standard user-level tool dir (~/.local/bin on POSIX; uv's default
    tool bin dir on Windows). Desktop/TUI workers may start with a minimal
    PATH that omits it even when `uv tool install browser-use` put the
    binary there."""
    try:
        if os.name == "nt":
            base = os.environ.get("APPDATA")
            if base:
                return str(Path(base) / "uv" / "bin")
            return None
        return str(Path(os.path.expanduser("~")) / ".local" / "bin")
    except Exception as e:  # pragma: no cover — defensive
        logger.debug("Could not resolve user-local bin dir: %s", e)
        return None


def _find_cli() -> Optional[List[str]]:
    """Locate the browser-use CLI, or None when it can't be run.

    MANAGED-FIRST resolution: Hermes' own ``$HERMES_HOME/bin`` copy — the
    one every browser backend selection installs and updates via
    ``install_cli()`` — always wins, so all sessions drive one canonical,
    Hermes-controlled binary. PATH and the user-level tool dir
    (~/.local/bin / %APPDATA%\\uv\\bin, where a manual ``uv tool install``
    links binaries) are fallbacks for setups that never ran our install,
    and cover Desktop/TUI workers that spawn with a minimal PATH. The uvx
    zero-install path (same probe order) is the final fallback.
    """
    probe_paths = (_managed_bin_dir(), None, _user_local_bin_dir())
    for probe_path in probe_paths:
        if probe_path is None or probe_path:
            direct = shutil.which("browser-use", path=probe_path)
            if direct:
                return [direct]
    for probe_path in probe_paths:
        if probe_path is None or probe_path:
            uvx = shutil.which("uvx", path=probe_path)
            if uvx:
                return [uvx, "browser-use"]
    return None


def install_cli(timeout_s: int = 600) -> Tuple[bool, str]:
    """Install the browser-use CLI persistently via ``uv tool install``.

    Resolution order for uv: Hermes' managed uv (bootstrapped on demand via
    ``hermes_cli.managed_uv.ensure_uv``) → uv on PATH. The binary is linked
    into ``$HERMES_HOME/bin`` (``UV_TOOL_BIN_DIR``) so ``_find_cli()``
    resolves it for every profile without touching the user's PATH.

    Returns ``(ok, message)`` — never raises.
    """
    # MANAGED-FIRST: only the managed copy short-circuits the install. A
    # browser-use found on PATH is a user-level side install — it must NOT
    # prevent provisioning the canonical Hermes-managed copy, or resolution
    # stays pinned to a binary we don't control (version drift, no updates
    # through hermes tools).
    bin_dir = _managed_bin_dir()
    if bin_dir:
        managed = shutil.which("browser-use", path=bin_dir)
        if managed:
            return True, f"browser-use CLI already installed ({managed})"

    uv_bin: Optional[str] = None
    try:
        from hermes_cli.managed_uv import ensure_uv

        uv_bin = str(ensure_uv() or "") or None
    except Exception as e:
        logger.debug("Managed uv bootstrap unavailable: %s", e)
    if not uv_bin:
        uv_bin = shutil.which("uv")
    if not uv_bin:
        return False, (
            "uv is not available and could not be bootstrapped. Install uv "
            "(https://docs.astral.sh/uv/) and run `uv tool install browser-use`."
        )

    env = dict(os.environ)
    env["UV_NO_CONFIG"] = "1"
    if bin_dir:
        try:
            Path(bin_dir).mkdir(parents=True, exist_ok=True)
            env["UV_TOOL_BIN_DIR"] = bin_dir
        except OSError as e:
            logger.debug("Could not prepare %s: %s", bin_dir, e)

    try:
        result = subprocess.run(
            [uv_bin, "tool", "install", "browser-use"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return False, f"`uv tool install browser-use` timed out after {timeout_s}s"
    except Exception as e:
        return False, f"Failed to run `uv tool install browser-use`: {e}"

    if result.returncode != 0:
        tail = "\n".join(
            (result.stderr or result.stdout or "").strip().splitlines()[-3:]
        )
        return False, f"`uv tool install browser-use` failed:\n{tail}"

    found = _find_cli()
    if not found or len(found) != 1:
        return False, (
            "install reported success but the browser-use binary is still "
            "not resolvable — run `uv tool install browser-use` manually"
        )
    return True, f"browser-use CLI installed ({found[0]})"


def _workspace_dir(task_id: Optional[str]) -> Optional[str]:
    """Stable per-task scratch dir that persists across browser_exec calls"""
    existing = os.environ.get("BH_AGENT_WORKSPACE")
    if existing:
        return existing
    try:
        from pathlib import Path

        from hermes_constants import get_hermes_home

        safe = _TASK_ID_SAFE_RE.sub("_", str(task_id or "default"))[:80] or "default"
        path = Path(get_hermes_home()) / "cache" / "browser-use" / "workspace" / safe
        path.mkdir(parents=True, exist_ok=True)
        return str(path)
    except Exception as e:
        logger.debug("browser_exec workspace unavailable: %s", e)
        return None


def _find_screenshot(stdout: str, since: float) -> Optional[str]:
    """Return the last screenshot path printed during this exec, or None.

    Only accepts files that exist and were written after the exec started
    """
    for path in reversed(_IMAGE_PATH_RE.findall(stdout or "")):
        try:
            if os.path.isfile(path) and os.path.getmtime(path) >= since - 1:
                return path
        except OSError:
            continue
    return None


def _native_screenshot_result(result: Dict[str, Any], path: str) -> Optional[Dict[str, Any]]:
    """Build a multimodal tool result attaching path for vision models"""
    try:
        from pathlib import Path

        from tools.vision_tools import (
            _EMBED_MAX_DIMENSION,
            _EMBED_TARGET_BYTES,
            _resize_image_for_vision,
            _should_use_native_vision_fast_path,
        )

        if not _should_use_native_vision_fast_path():
            return None
        # History-reuse cap (#92699): this data URL bakes into the tool
        # result and is re-sent on every later turn — same policy as the
        # vision_analyze / browser_vision native embeds (256 KB / 1568 px,
        # JPEG quality ladder instead of PNG dimension-halving).
        data_url = _resize_image_for_vision(
            Path(path),
            mime_type="image/png",
            max_base64_bytes=_EMBED_TARGET_BYTES,
            max_dimension=_EMBED_MAX_DIMENSION,
            force_jpeg=True,
        )
        text = json.dumps(result, ensure_ascii=False)
        return {
            "_multimodal": True,
            "content": [
                {
                    "type": "text",
                    "text": (
                        text
                        + "\n\nThe screenshot from this call is attached — "
                        "inspect it with your native vision."
                    ),
                },
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
            "text_summary": text,
            "meta": {"screenshot_path": path, "native_vision": True},
        }
    except Exception as e:
        logger.debug("Native screenshot attach failed (falling back to text): %s", e)
        return None


def _backend_cache_key(task_id: Optional[str], session_name: str = "") -> str:
    """Session-cache key for a backend browser: named sessions get their own."""
    return f"bu-named-{session_name}" if session_name else (task_id or "browser-exec-default")


def _resolve_lightpanda_cdp(
    env: dict, task_id: Optional[str], session_name: str = ""
) -> Optional[str]:
    """Point the harness at a Hermes-spawned ``lightpanda serve``.

    Only when ``browser.engine`` is ``lightpanda`` and nothing with higher
    precedence (BU_CDP_* env, a CDP override, a cloud provider) claimed the
    session. Each cache key gets its own Lightpanda process via the legacy
    stack's ``_get_session_info()`` (cache, inactivity reaper, atexit), so
    the browser is private to this session and the own-tab preamble is
    skipped. Returns an error string when Lightpanda cannot start.
    """
    try:
        from tools.browser_tool import _get_session_info, _using_lightpanda_engine
    except Exception as e:  # pragma: no cover — stubbed browser_tool in tests
        logger.debug("browser_tool lightpanda resolution unavailable: %s", e)
        return None
    try:
        if not _using_lightpanda_engine():
            return None
    except Exception as e:
        logger.debug("browser engine lookup failed: %s", e)
        return None
    try:
        session_info = _get_session_info(_backend_cache_key(task_id, session_name))
    except Exception as e:
        return (
            f"Lightpanda could not be started: {e} Set browser.engine to auto "
            "to use local Chrome, or switch backends via `hermes tools` → "
            "Browser Automation."
        )
    cdp = str((session_info or {}).get("cdp_url") or "")
    if not cdp:
        return (
            "Lightpanda session returned no CDP endpoint. Set browser.engine "
            "to auto to use local Chrome."
        )
    env["BU_CDP_URL" if cdp.startswith(("http://", "https://")) else "BU_CDP_WS"] = cdp
    env[_PRIVATE_BROWSER_SENTINEL] = "1"
    return None


def _resolve_backend_cdp(
    env: dict, task_id: Optional[str], session_name: str = ""
) -> Optional[str]:
    """Point the harness at the configured browser backend's CDP endpoint.

    Resolution order (first hit wins):

    1. ``BU_CDP_WS`` / ``BU_CDP_URL`` already in the environment — explicit
       user/operator override, passed through untouched.
    2. ``BROWSER_CDP_URL`` env / ``browser.cdp_url`` config override — the
       ``/browser connect`` path, same precedence the built-in tools honor.
    3. A configured cloud browser provider (Browserbase, Firecrawl, Nous
       gateway/Browser Use cloud, …): reuse the legacy stack's
       ``_get_session_info()`` so browser_exec shares the SAME provider
       session machinery — per-task session cache, expiry replacement,
       inactivity reaper, and atexit cleanup — instead of duplicating it.
    4. ``browser.engine: lightpanda``: a Hermes-spawned ``lightpanda serve``
       per session key, through the same ``_get_session_info()`` machinery
       (see :func:`_resolve_lightpanda_cdp`).
    5. Nothing configured: return None; the harness attaches to local
       Chrome (or Browser Use cloud via BU_AUTOSPAWN for legacy configs).

    ``session_name`` (the tool's ``session`` argument / BU_NAME) keys the
    provider session cache when set, so every distinct name gets its OWN
    cloud browser and the same name reuses one — that is what makes named
    sessions actually concurrent-safe on provider backends instead of all
    names sharing a single per-task browser.

    Returns an error string on provider failure, None on success.
    """
    if env.get("BU_CDP_WS") or env.get("BU_CDP_URL"):
        return None

    try:
        from tools.browser_tool import (
            _get_cdp_override,
            _get_cloud_provider,
            _get_session_info,
        )
    except Exception as e:  # pragma: no cover — stubbed browser_tool in tests
        logger.debug("browser_tool backend resolution unavailable: %s", e)
        return None

    try:
        override = _get_cdp_override()
    except Exception:
        override = ""
    if override:
        env["BU_CDP_URL" if override.startswith(("http://", "https://")) else "BU_CDP_WS"] = override
        return None

    try:
        provider = _get_cloud_provider()
    except Exception as e:
        logger.debug("Cloud provider lookup failed: %s", e)
        provider = None
    if provider is None:
        return _resolve_lightpanda_cdp(env, task_id, session_name)

    # Browser Use direct-API configs: the CLI talks to Browser Use cloud
    # natively (BU_AUTOSPAWN / auth login) — routing through the legacy
    # provider here would just create a second, redundant session. The
    # Nous-gateway variant (use_gateway: true) DOES resolve through the
    # provider: the gateway provisions the cloud browser server-side and
    # returns its CDP URL, giving subscribers CLI mode with no raw key.
    provider_key = str(getattr(provider, "name", "") or "").strip().lower()
    if provider_key == _BACKEND_KEY and not is_truthy_value(
        _read_browser_cfg().get("use_gateway"), default=False
    ):
        # Named BU cloud browsers are exclusive to their daemon — no shared
        # tab to isolate from.
        env[_PRIVATE_BROWSER_SENTINEL] = "1"
        return None

    try:
        # Named sessions get their OWN provider browser, keyed by name so the
        # same name reuses one browser across calls and tasks, and different
        # names never collide. Unnamed calls keep the per-task key.
        cache_key = _backend_cache_key(task_id, session_name)
        session_info = _get_session_info(cache_key)
    except Exception as e:
        return (
            f"Cloud browser provider {type(provider).__name__} failed to "
            f"provide a session: {e}. Fix the provider configuration or "
            "switch backends via `hermes tools` → Browser Automation."
        )
    cdp = str((session_info or {}).get("cdp_url") or "")
    if not cdp:
        return (
            f"Cloud browser provider {type(provider).__name__} returned no "
            "CDP endpoint, so Browser Use mode cannot drive it. Switch to "
            "the built-in browser tools for this provider."
        )
    env["BU_CDP_URL" if cdp.startswith(("http://", "https://")) else "BU_CDP_WS"] = cdp
    # A provider browser keyed bu-named-<name> is exclusive to this session —
    # the own-tab preamble is unnecessary there (it would just leak a blank
    # tab into a browser nobody else touches).
    if session_name:
        env[_PRIVATE_BROWSER_SENTINEL] = "1"
    return None


def _real_profile_consented() -> bool:
    """Whether the user opted in to real-profile local browsing (config read)."""
    try:
        from tools.browser_tool import _use_real_profile

        return _use_real_profile()
    except Exception as e:  # pragma: no cover — stubbed browser_tool in tests
        logger.debug("real-profile consent lookup failed: %s", e)
        return False


def _resolve_real_profile_cdp(env: dict, force_local: bool) -> Optional[str]:
    """Point the harness at the user's real-profile copy-browser when consented.

    With ``browser.use_real_profile`` on, local browsing must mean the user's
    default Chromium with their logins — a browser Hermes launches on a
    SNAPSHOT of their real profile (see hermes_cli.browser_connect). Two ways
    in:

    - the effective backend is already local (no cloud provider, no CDP
      override, no legacy Browser Use cloud config): every local attach
      upgrades to the real profile, silently — this is requirement one; or
    - ``force_local`` (the consent-gated ``local`` tool arg): the model was
      asked to drive the user's actual browser even though a cloud backend
      is configured. The cloud backend keeps serving everything else.

    Explicit operator overrides (BU_CDP_WS/BU_CDP_URL env, /browser connect,
    ``browser.cdp_url``) own the session either way, matching the built-in
    lane's precedence.

    Sets BU_CDP_URL/BU_CDP_WS on success. Returns an error string when the
    real-profile launch fails (fail closed — a consented user is never
    silently downgraded to a throwaway browser), else None.
    """
    if not _real_profile_consented():
        return None
    if env.get("BU_CDP_WS") or env.get("BU_CDP_URL"):
        return None

    try:
        from tools.browser_tool import (
            _get_cdp_override_raw,
            _get_cloud_provider,
            _real_profile_cdp,
        )
    except Exception as e:  # pragma: no cover — stubbed browser_tool in tests
        logger.debug("real-profile backend resolution unavailable: %s", e)
        return None

    try:
        if _get_cdp_override_raw():
            return None
    except Exception:
        pass

    if not force_local:
        # Only auto-upgrade genuinely-local attaches; any cloud path (provider
        # or legacy Browser Use cloud config) stays on its backend unless the
        # model passes local=true.
        try:
            if _get_cloud_provider() is not None:
                return None
        except Exception:
            return None
        if is_legacy_browser_use_cloud_config(_read_browser_cfg()):
            return None

    cdp, err = _real_profile_cdp()
    if err:
        return err
    if cdp:
        env["BU_CDP_URL" if cdp.startswith(("http://", "https://")) else "BU_CDP_WS"] = cdp
    return None


def browser_exec(
    code: str,
    session: str = "",
    timeout_s: int = _DEFAULT_TIMEOUT_S,
    lease_minutes: int = 0,
    lease_reason: str = "",
    task_id: Optional[str] = None,
    local: bool = False,
    turn_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
):
    """Run Python code through the browser-use CLI, and return its output"""
    from tools.registry import tool_error, tool_result

    if not code or not code.strip():
        return tool_error("No code provided. Pass Python that uses the pre-imported helpers, e.g. new_tab(\"https://example.com\") then print(page_info()).")

    blocked = _blocked_url_in_code(code)
    if blocked:
        return tool_error(blocked)

    try:
        lease = int(lease_minutes or 0)
    except (TypeError, ValueError):
        return tool_error("lease_minutes must be an integer from 0 to 120.")
    if lease < 0 or lease > _MAX_LEASE_MINUTES:
        return tool_error(
            f"lease_minutes must be between 0 and {_MAX_LEASE_MINUTES}."
        )
    reason = str(lease_reason or "").strip()
    if lease and not reason:
        return tool_error(
            "lease_reason is required when lease_minutes is non-zero. "
            "Use a short task-specific reason."
        )

    cmd = _find_cli()
    if not cmd:
        return tool_error(
            "browser-use CLI not found on PATH, and uvx is unavailable for a "
            "zero-install run. Install it with `uv tool install browser-use` "
            "(or `pipx install browser-use`), then run `browser-use --doctor` "
            "to verify the setup."
        )

    env = _base_subprocess_env()
    if session:
        if not _SESSION_RE.match(session):
            return tool_error(
                f"Invalid session name {session!r}: use 1-64 letters, digits, "
                "dashes, or underscores (e.g. 'r7k2')."
            )
        env["BU_NAME"] = session
    # Real-profile consent: on a local backend this upgrades the attach to
    # the user's default browser (profile snapshot, logins included); with
    # local=True it forces that even under a cloud backend. Runs BEFORE
    # provider resolution so a real-profile hit short-circuits the cloud
    # path via the BU_CDP_* env contract.
    rp_err = _resolve_real_profile_cdp(env, force_local=bool(local))
    if rp_err:
        return tool_error(rp_err)
    if local and not (env.get("BU_CDP_URL") or env.get("BU_CDP_WS")):
        # local=True is only served by the real-profile route; anything else
        # (consent off — schema normally hidden, but be explicit; or an
        # operator CDP override owning the session) must not pretend.
        if not _real_profile_consented():
            return tool_error(
                "local=true was requested but browser.use_real_profile is off. "
                "Enable it in config.yaml (browser.use_real_profile: true) or "
                "the desktop Settings → Browser section, then retry."
            )
    # Route through the configured browser backend (Browserbase, Firecrawl,
    # Nous gateway, CDP override, local Chrome, …). Named sessions compose
    # with the backend: BU_NAME namespaces the harness daemon (its IPC
    # socket, log, and pid), and on provider backends the name additionally
    # keys its own cloud browser — so concurrent sessions stop clobbering
    # each other's daemon (#86894). Browser Use direct-API cloud configs
    # are the one exception: the CLI manages named cloud browsers natively,
    # and _resolve_backend_cdp skips provider resolution for them.
    backend_err = _resolve_backend_cdp(env, task_id, session_name=session)
    if backend_err:
        return tool_error(backend_err)

    # On a SHARED browser (local Chrome / CDP override) a fresh named daemon
    # attaches to the first existing page — the same page a sibling daemon
    # may hold. Pin each named session to a tab it created before running
    # the model's code. Private per-name browsers (provider-keyed or BU
    # cloud) skip this: no one to collide with, and the extra tab would leak.
    private_browser = env.pop(_PRIVATE_BROWSER_SENTINEL, None)
    transport = _configure_session_transport(
        env,
        conversation_id=str(conversation_id or ""),
        logical_session=session or "default",
        private_browser=bool(private_browser),
    )

    workspace = _workspace_dir(task_id)
    if workspace:
        env["BH_AGENT_WORKSPACE"] = workspace

    # BU_AUTOSPAWN makes the CLI start a Browser Use cloud browser when no
    # local Chrome/CDP endpoint is reachable (their API key authenticates it)
    if "BU_AUTOSPAWN" not in env and is_legacy_browser_use_cloud_config(_read_browser_cfg()):
        env["BU_AUTOSPAWN"] = "1"

    try:
        timeout = max(_MIN_TIMEOUT_S, min(int(timeout_s), _MAX_TIMEOUT_S))
    except (TypeError, ValueError):
        timeout = _DEFAULT_TIMEOUT_S

    hygiene_guard = None
    hygiene_start_error = None
    if _resource_hygiene_enabled() and not private_browser:
        from tools.browser_tool import prepare_browser_tab_lifecycle

        hygiene_guard, hygiene_start_error = prepare_browser_tab_lifecycle(
            session_name=session,
            owner_key=turn_id or task_id or "browser-exec-default",
            lease_minutes=lease,
            lease_reason=reason,
            lock_timeout_s=min(60, timeout),
            cdp_url=str(env.get("BU_CDP_URL") or env.get("BU_CDP_WS") or ""),
            private_browser=bool(private_browser),
        )
        if hygiene_start_error:
            return tool_error(
                "Browser tab lifecycle blocked this call before navigation: "
                f"{hygiene_start_error}. No browser code was executed."
            )

    if hygiene_guard is not None and hygiene_guard.target_id:
        code = f"switch_tab({json.dumps(hygiene_guard.target_id)})\n" + code
    elif not private_browser and (session or transport is not None):
        code = _OWN_TAB_PREAMBLE + code

    # Windows: hide the console the .cmd shim would flash (as browser_tool does)
    popen_extra: dict = {}
    if os.name == "nt":
        try:
            from hermes_cli._subprocess_compat import windows_hide_flags

            popen_extra["creationflags"] = windows_hide_flags()
            _si = subprocess.STARTUPINFO()
            _si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            popen_extra["startupinfo"] = _si
        except Exception as e:
            logger.debug("Windows hide-flags unavailable: %s", e)

    started = time.time()
    proc = None
    execution_error = None
    hygiene_report = None
    transport_telemetry = None
    before_daemon = None
    execution_lock = (
        _TransportExecutionLock(transport, min(60, timeout))
        if transport is not None
        else None
    )
    lock_held = False
    try:
        if execution_lock is not None and transport is not None:
            execution_lock.acquire()
            lock_held = True
            before_daemon = _daemon_identity(transport.runtime_dir)
        proc = subprocess.run(
            cmd,
            input=code,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            **popen_extra,
        )
    except subprocess.TimeoutExpired:
        execution_error = (
            f"browser-use exec timed out after {timeout}s. The daemon may "
            "still be working; retry with a larger timeout_s (max "
            f"{_MAX_TIMEOUT_S}), or split the work into several calls that "
            "append to workspace files — anything already written to the "
            "workspace is preserved."
        )
    except TimeoutError as e:
        execution_error = f"Browser CDP transport is busy: {e}"
    except OSError as e:
        execution_error = f"Failed to launch browser-use CLI: {e}"
    finally:
        try:
            hygiene_report = (
                hygiene_guard.finish() if hygiene_guard is not None else None
            )
        finally:
            if transport is not None and lock_held:
                try:
                    transport_telemetry = _record_transport_outcome(
                        transport,
                        before_daemon,
                        _daemon_identity(transport.runtime_dir),
                        session or "default",
                        execution_error is None
                        and proc is not None
                        and proc.returncode == 0,
                        execution_error
                        or ((proc.stderr or "") if proc is not None else ""),
                    )
                except Exception:
                    logger.debug("browser CDP telemetry failed", exc_info=True)
            if lock_held and execution_lock is not None:
                execution_lock.release()

    if execution_error:
        if hygiene_report and not hygiene_report.get("ok", True):
            execution_error += " Cleanup also failed: " + "; ".join(
                hygiene_report.get("errors") or ["unknown hygiene error"]
            )
        return tool_error(execution_error)

    assert proc is not None

    result = {
        "success": proc.returncode == 0,
        "exit_code": proc.returncode,
        "output": proc.stdout,
    }
    if workspace:
        result["workspace"] = workspace
    if session:
        result["session"] = session
    if transport_telemetry is not None:
        result["cdp_transport"] = transport_telemetry
    if hygiene_report is not None:
        result["hygiene"] = hygiene_report
        if not hygiene_report.get("ok", False):
            result["success"] = False
    stderr = (proc.stderr or "").strip()
    if stderr:
        if len(stderr) > _STDERR_CAP_CHARS:
            stderr = stderr[:_STDERR_CAP_CHARS] + "\n… (stderr truncated)"
        result["stderr"] = stderr

    screenshot = _find_screenshot(proc.stdout, started)
    if screenshot:
        result["screenshot_path"] = screenshot
        native = _native_screenshot_result(result, screenshot)
        if native is not None:
            return native
    return tool_result(result)


# The tool description is the CLI's skill, fetched from browser-use skill
_HEADER_BASE = (
    "Drive a real web browser via the Browser Use CLI: `code` runs as full "
    "Python (stdlib available) with pre-imported browser helpers; stdout "
    "comes back in the result. Start `code` with a one-line comment "
    "describing the step for the user in plain language, max 60 chars "
    "(e.g. `# Searching Amazon for paper towels`) — the UI shows it as the "
    "step label.\n\n"
    "STATE: the browser session and workspace persist across calls; Python "
    "variables do NOT (fresh interpreter each call). The workspace dir is "
    "$BH_AGENT_WORKSPACE (also `workspace` in every result); functions "
    "defined in agent_helpers.py there are auto-imported into every call. "
    "For multi-item tasks ('all N products / every entry'), append each "
    "batch to a JSON/CSV file in the workspace, then read it back and "
    "aggregate in code — dedupe/count/sort with Python, not in your head — "
    "and verify the collected count against what was asked before "
    "answering.\n\n"
    "Batch each sub-procedure (navigate, wait, extract, act) into one call "
    "— do not spend a call per action — but for long extractions prefer "
    "several medium calls that append to workspace files over one giant "
    "call, so progress survives timeouts."
)

_HEADER_VISION = (
    " Screenshots are attached to your context automatically: when the exec "
    "output contains a capture_screenshot() path, the image arrives with "
    "this tool's result and you inspect it directly with your own vision — "
    "never send browser screenshots to a separate vision tool."
)

_HEADER_TEXT_ONLY = (
    " Your model cannot view images, so work text-first: page_info() for "
    "state, js() for reading/extracting DOM text, fill_input(selector, "
    "text) for inputs, and js(\"document.querySelector('…').click()\") for "
    "clicks — skip the screenshot-driven workflow described below."
)

# Appended when the local engine is Lightpanda (browser.engine). Lightpanda
# has no graphical renderer, and one CDP connection holds one page: a second
# Target.createTarget fails with TargetAlreadyLoaded
# (lightpanda-io/browser#1962) — drop the new_tab() sentence once that lands.
_HEADER_LIGHTPANDA = (
    " The local engine is Lightpanda (no graphical renderer, one page per "
    "session): capture_screenshot() is unavailable, so work text-first; "
    "navigate with new_tab(url) exactly once, then goto_url(url) for every "
    "later navigation — a second new_tab() fails with TargetAlreadyLoaded."
)

_DESCRIPTION_HEADER = _HEADER_BASE  # back-compat alias for external imports

# NOTE: browser_exec is additionally gated at tool-definition time — sessions
# whose resolved toolsets do not include ``terminal`` never see it (see
# model_tools._compute_tool_definitions). The check_fn registered below only
# answers "is Browser Use mode configured"; surface policy lives with the
# session, not in the process-wide TTL-cached check_fn.


def _description_header() -> str:
    """Header tailored to the active engine, vision, and tab lifecycle."""
    if _lightpanda_engine_in_use():
        # Lightpanda sessions are private and one-page-only, so shared-CDP tab
        # lifecycle instructions do not apply even when the feature is enabled.
        return _HEADER_BASE + _HEADER_TEXT_ONLY + _HEADER_LIGHTPANDA

    hygiene = ""
    if _resource_hygiene_enabled():
        hygiene = (
            "\n\nTAB LIFECYCLE: Local tabs created or claimed from a blank "
            "baseline close automatically after each browser_exec call, including "
            "errors and timeouts. The tool verifies closure and leaves one blank "
            "baseline page. When the NEXT browser_exec call genuinely must reuse "
            "the current tab, set lease_minutes (1-120) and a task-specific "
            "lease_reason. On the final call, omit the lease so owned tabs close. "
            "Never lease a tab merely because it may be useful later."
        )
    try:
        from tools.vision_tools import _should_use_native_vision_fast_path

        if _should_use_native_vision_fast_path():
            return _HEADER_BASE + hygiene + _HEADER_VISION
    except Exception:
        pass
    return _HEADER_BASE + hygiene + _HEADER_TEXT_ONLY


def _lightpanda_engine_in_use() -> bool:
    try:
        from tools.browser_tool import lightpanda_engine_status

        return lightpanda_engine_status()[0]
    except Exception as e:
        logger.debug("lightpanda engine status unavailable: %s", e)
        return False

_skill_text_cache: Optional[str] = None
_skill_text_fetched = False

# Pinned quick-reference for the CLI's pre-imported helpers. Replaces the
# live ``browser-use skill`` fetch: embedding whatever text the installed CLI
# version prints would ship uncontrolled third-party content into every
# session's system-side schema (version drift across machines, supply-chain
# exposure, and a byte-unstable prompt). A/B benchmarked Aug 2026 (108 runs,
# opus-4.8 + kimi-k3, 6 multi-step tasks x 3 reps): header-only schema went
# 36/36 vs 36/36 for the full skill dump at ~equal tokens (-60% vs the
# legacy browser_* toolset either way). The pinned digest below keeps the
# first-call reliability of the helper names without the 7.7KB dump.
_HELPERS_DIGEST = (
    "\n\nHELPERS (pre-imported): new_tab(url) opens/navigates (use for the "
    "FIRST navigation), goto_url(url) navigates the current tab, "
    "wait_for_load() after navigation, page_info() summarizes the current "
    "page state, js(expr) evaluates a JS expression and returns its value "
    "(js('document.title'); wrap function bodies as js('(() => {...})()') — "
    "a bare '() => {...}' returns the function itself, uncalled), "
    "fill_input(selector, text) types into inputs, click_at_xy(x, y) clicks "
    "viewport coordinates, capture_screenshot() saves and prints a "
    "screenshot path, cdp('Domain.method', **kwargs) is raw CDP — "
    "cdp('Accessibility.getFullAXTree')['nodes'] lists every element's "
    "role/name/backendDOMNodeId (filter in Python before printing; it is "
    "thousands of nodes), then cdp('DOM.getBoxModel', backendNodeId=n) gives "
    "click coordinates. ensure_real_tab() recovers from a stale/internal "
    "tab. Login walls: stop and ask the user; never guess credentials."
)


def _cli_skill_text() -> str:
    """Deprecated: always returns "" — the schema uses the pinned header.

    Kept so tests and any external callers keep importing a stable symbol;
    see _HELPERS_DIGEST for the rationale (benchmark-backed removal of the
    live ``browser-use skill`` fetch).
    """
    return _skill_text_cache or ""


def _dynamic_schema_overrides() -> dict:
    overrides: dict = {"description": _description_header() + _HELPERS_DIGEST}
    # The ``local`` argument exists ONLY when the user consented to
    # real-profile browsing — everyone else's schema carries zero extra
    # surface. get_definitions() applies this at schema-build time, and the
    # caller memoizes on config.yaml mtime, so toggling consent changes the
    # schema on the next session rather than mid-conversation.
    if _real_profile_consented():
        props = dict(BROWSER_EXEC_SCHEMA["parameters"]["properties"])
        props["local"] = {
            "type": "boolean",
            "description": (
                "Drive the user's own local browser (a Hermes-managed copy of "
                "their real default-Chromium profile, logins/cookies included) "
                "instead of the configured cloud browser backend. Use when the "
                "user asks to act as themselves — their accounts, their "
                "sessions. No-op when the backend is already local. Default "
                "false."
            ),
            "default": False,
        }
        overrides["parameters"] = {**BROWSER_EXEC_SCHEMA["parameters"], "properties": props}
    return overrides


BROWSER_EXEC_SCHEMA = {
    "name": "browser_exec",
    # Static fallback, used only when the CLI (and uvx) is unavailable
    "description": (
        _HEADER_BASE
        + _HELPERS_DIGEST
        + "\n\n(The browser-use CLI is not installed yet. Install it with "
        "`uv tool install browser-use`.)"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute using the pre-imported browser helpers. Use print(...) for any data you need back.",
            },
            "session": {
                "type": "string",
                "description": (
                    "Named logical browser session. Related calls reuse its tab "
                    "namespace; cloud backends may also allocate a browser. Shared "
                    "local/CDP calls multiplex one persistent physical connection "
                    "per Hermes conversation lineage, avoiding Chrome re-prompts. "
                    "Reuse the same name for related calls; omit for default."
                ),
            },
            "timeout_s": {
                "type": "integer",
                "description": f"Max seconds to wait for the code to finish (default {_DEFAULT_TIMEOUT_S}, max {_MAX_TIMEOUT_S}).",
                "default": _DEFAULT_TIMEOUT_S,
            },
            "lease_minutes": {
                "type": "integer",
                "description": (
                    "Keep local task-owned tabs for the next call for 1-120 "
                    "minutes. Default 0 closes them after this call. Requires "
                    "lease_reason; omit on the final call."
                ),
                "minimum": 0,
                "maximum": _MAX_LEASE_MINUTES,
                "default": 0,
            },
            "lease_reason": {
                "type": "string",
                "description": (
                    "Short task-specific reason for a non-zero lease. Do not "
                    "include URLs, credentials, or private page content."
                ),
                "maxLength": 500,
            },
        },
        "required": ["code"],
    },
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
from tools.registry import registry

registry.register(
    name="browser_exec",
    toolset="browser-use",
    schema=BROWSER_EXEC_SCHEMA,
    handler=lambda args, **kw: browser_exec(
        code=args.get("code", ""),
        session=args.get("session", "") or "",
        timeout_s=args.get("timeout_s", _DEFAULT_TIMEOUT_S),
        lease_minutes=args.get("lease_minutes", 0),
        lease_reason=args.get("lease_reason", ""),
        task_id=kw.get("task_id"),
        local=bool(args.get("local", False)),
        turn_id=kw.get("turn_id"),
        conversation_id=kw.get("conversation_id") or kw.get("session_id"),
    ),
    check_fn=is_browser_use_cli_mode,
    dynamic_schema_overrides=_dynamic_schema_overrides,
    emoji="🌐",
)
