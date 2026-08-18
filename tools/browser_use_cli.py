"""Use the Browser Use CLI 3.0 (https://browser-use.com) for browser automation

When browser.backend is "browser-use", the model gets ``browser_exec`` tool
instead of default browser tools
"""

import json
import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils import is_truthy_value

logger = logging.getLogger(__name__)

_BACKEND_KEY = "browser-use"
BACKEND_DISABLED = "off"

# Cloud daemon names become the BU_NAME env var
_SESSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

# Internal marker set by _ensure_exec_cdp_endpoint on the env dict when the
# resolved browser is EXCLUSIVE to this named session (per-name provider
# browser). Popped before the subprocess launches — never exported to the CLI.
_PRIVATE_BROWSER_SENTINEL = "_HERMES_BU_PRIVATE_BROWSER"

# Preamble prepended to the model's code for named sessions on SHARED
# browsers (local Chrome / CDP override). The harness daemon attaches to the
# first existing page at startup, so two fresh named daemons can land on the
# SAME tab; steering this daemon onto a tab it created keeps concurrent named
# sessions from clobbering each other before their first new_tab(). Runs
# once per daemon (marker file keyed by BU_NAME under the harness runtime
# state), costs one IPC round-trip on later calls.
_OWN_TAB_PREAMBLE = """\
# hermes: pin this named session to its own tab (once per daemon process)
def _hermes_ensure_own_tab():
    import os as _os, tempfile as _tf
    _name = _os.environ.get("BU_NAME", "default")
    try:
        # Key the marker by the daemon's pid so a daemon restart (which
        # re-attaches to the first shared page) re-pins automatically,
        # while agent-driven tab switches mid-session are left alone.
        from browser_harness import _ipc as _bipc
        _dpid = _bipc.pid_path(_name).read_text().strip() or "0"
    except Exception:
        _dpid = "0"
    _uid = _os.getuid() if hasattr(_os, "getuid") else 0
    _marker = _os.path.join(
        _tf.gettempdir(), "hermes-bu-owntab-%s-%s-%s" % (_uid, _name, _dpid)
    )
    if _os.path.exists(_marker):
        return
    try:
        # Force a fresh target: new_tab() would REUSE a blank current tab,
        # which is exactly the tab a sibling daemon may also hold.
        _tid = cdp("Target.createTarget", url="about:blank").get("targetId")
        if _tid:
            switch_tab(_tid)
    except Exception:
        pass  # best-effort: worst case is pre-fix behavior
    try:
        open(_marker, "w").close()
    except OSError:
        pass
_hermes_ensure_own_tab()
del _hermes_ensure_own_tab
"""

_DEFAULT_TIMEOUT_S = 300
_MIN_TIMEOUT_S = 5
_MAX_TIMEOUT_S = 1800
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


def _blocked_url_in_code(code: str) -> Optional[str]:
    """Return an error if a URL literal fails the built-in navigation checks.

    Runs in ``strict`` mode (Region D): the private-address gate is
    unconditional w.r.t. the backend posture and fails closed on DNS errors.
    """
    from tools.browser_tool import evaluate_url_safety

    for url in _URL_RE.findall(code or ""):
        err = evaluate_url_safety(url, strict=True)
        if err:
            return err.get("error", "Blocked: unsafe URL")
    return None


# Marker the recheck trailer prints the landed URL behind. Long and unlikely
# to occur in page text, so a page that echoes it cannot forge a "safe" URL:
# the last occurrence wins (see _landed_url), and the trailer always runs
# after any page output.
_LANDED_URL_MARKER = "__HERMES_BROWSER_EXEC_LANDED_URL__:"

# Env keys the model can control via the first (untrusted) invocation and
# that may trigger harness/workspace bootstrap code auto-loading on a NEW
# subprocess (ref. the harness's $BH_AGENT_WORKSPACE/agent_helpers.py seam).
# The trusted probe must not inherit these, or the model could poison its
# js/print helpers the same way it can the main script.
_MODEL_CONTROLLED_PROBE_ENV = frozenset({
    "BH_AGENT_WORKSPACE",
    "BU_WORKSPACE",
})

# Matches ONLY the trailer's landing-report line: the marker immediately
# followed by an absolute http(s) URL. Used to strip exactly that line from
# output without touching page content that happens to echo the marker.
_URL_RECHECK_REPORT_RE = re.compile(
    r"^" + re.escape(_LANDED_URL_MARKER) + r"\s*https?://\S+$"
)

# Appended to exec code so the *executed* navigation target can be checked,
# not just the literals the pre-check could see in the source. Mirrors
# browser_tool's _current_page_private_url, which reads window.location.href
# after an eval for exactly this reason. Wrapped in try/except so a session
# with no page open (or a helper that raises) degrades to "no marker" rather
# than turning a working exec into an error.
# The marker literal is split across a concatenation so the trailer's own
# source text never contains it: the code is echoed back by some CLI modes,
# and a source echo must not read as a landing report.
_URL_RECHECK_TRAILER = (
    "\ntry:\n"
    "    print({head!r} + {tail!r} + str(js('window.location.href')))\n"
    "except Exception:\n"
    "    pass\n"
).format(head=_LANDED_URL_MARKER[:16], tail=_LANDED_URL_MARKER[16:])


# ---------------------------------------------------------------
# Trusted-boundary landed-URL observation (P1 review).
#
# The appended in-script trailer runs in the SAME namespace as the
# model's untrusted code, so the model can rebind js/print/str and forge
# a safe landing marker. The authoritative observation must therefore be
# produced OUTSIDE that namespace: run a SECOND Browser Use subprocess
# whose only input is Hermes-authored probe code (no model code), against
# the same session/CDP. That fresh namespace cannot be rebound, so its
# reported window.location.href is a trusted attestation of browser state.
# ---------------------------------------------------------------
def _trusted_landing_probe_code() -> str:
    """Hermes-authored probe reading window.location.href in a fresh CLI.

    No caller code is present, so js/print/str are the CLI's own builtins
    and cannot have been rebound. The marker is produced by trusted code.
    """
    head = _LANDED_URL_MARKER[:16]
    tail = _LANDED_URL_MARKER[16:]
    return (
        f"print({head!r} + {tail!r} + str(js('window.location.href')))"
    )


def _trusted_landed_url(cmd, env, popen_extra, timeout) -> Optional[str]:
    """Run a fresh, trusted CLI probe and return the observed landed URL.

    P1 hardening: the probe MUST NOT inherit any model-controlled workspace
    or bootstrap state. ``BH_AGENT_WORKSPACE`` (and the harness's auto-loaded
    ``agent_helpers.py`` seam) carry code the model can write in the first
    invocation — loading it into the probe would let the model poison the
    ``js``/``print`` helpers again. So this strips those env keys before the
    probe runs, leaving only the trusted, Hermes-authored probe code with the
    CLI's own builtins. Returns None if the probe could not run or reported
    nothing trustworthy (fail-open: no result means no claim).
    """
    probe_env = dict(env)
    for key in _MODEL_CONTROLLED_PROBE_ENV:
        probe_env.pop(key, None)
    probe = _trusted_landing_probe_code()
    try:
        p = subprocess.run(
            cmd, input=probe, capture_output=True,
            text=True, timeout=timeout, env=probe_env, **popen_extra,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    return _landed_url(p.stdout)


def _with_url_recheck(code: str) -> str:
    """Append the landed-URL probe when doing so cannot break the caller's code.

    ``ast.parse`` succeeding means the code is a syntactically complete
    module, so appending another top-level statement is guaranteed to keep it
    parseable. Code that does not parse is returned unchanged: it cannot
    navigate anywhere, and mangling it would replace the CLI's own syntax
    error with a confusing one.
    """
    import ast

    try:
        ast.parse(code)
    except SyntaxError:
        return code
    return code + _URL_RECHECK_TRAILER


def _landed_url(stdout: str) -> Optional[str]:
    """Return the URL the browser actually ended on, or None if unknown.

    Only an absolute http(s) URL counts. Anything else — a helper that
    returned None, a truncated line, page text that happens to carry the
    marker — is treated as "probe produced nothing", which fails open rather
    than blocking on a value that was never a navigation target.
    """
    landed = None
    for line in (stdout or "").splitlines():
        idx = line.rfind(_LANDED_URL_MARKER)
        if idx == -1:
            continue
        candidate = line[idx + len(_LANDED_URL_MARKER):].strip()
        if candidate.lower().startswith(("http://", "https://")):
            landed = candidate
    return landed


def _strip_landed_url_marker(stdout: str) -> str:
    """Drop the probe's landing-report line so it stays invisible.

    Only the line that carried the *landed URL* (the trailer's actual
    output) is removed. A content line that merely happens to contain the
    marker string is preserved — it is page data, not the probe report,
    and dropping it would lose legitimate output (Point 3 review).
    """
    if _LANDED_URL_MARKER not in (stdout or ""):
        return stdout
    kept = [ln for ln in stdout.splitlines()
            if not (_URL_RECHECK_REPORT_RE.match(ln))]
    stripped = "\n".join(kept)
    return stripped + "\n" if stdout.endswith("\n") and stripped else stripped


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
    env.setdefault("ANONYMIZED_TELEMETRY", "false")
    return env


def _read_browser_cfg() -> dict:
    """Return the ``browser:`` config section, or {} on any failure."""
    try:
        from hermes_cli.config import cfg_get, read_raw_config

        cfg = cfg_get(read_raw_config(), "browser", default={})
        return cfg if isinstance(cfg, dict) else {}
    except Exception as e:
        logger.debug("Could not read browser config section: %s", e)
        return {}


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
            _resize_image_for_vision,
            _should_use_native_vision_fast_path,
        )

        if not _should_use_native_vision_fast_path():
            return None
        data_url = _resize_image_for_vision(Path(path))
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


def _ensure_exec_cdp_endpoint(env: dict, task_id: Optional[str], session: Optional[str]) -> Optional[str]:
    """Guarantee a monitorable CDP endpoint for ``browser_exec`` (Region A C1).

    Called in BOTH branches of ``browser_exec`` (plain and ``session=``/
    ``BU_NAME``): every exec resolves an endpoint the Hermes-side network
    monitor can attach to — or fails before the CLI spawns. Resolution order
    (first hit wins):

    1. ``BU_CDP_WS`` / ``BU_CDP_URL`` already in the environment — explicit
       user/operator override, passed through untouched.
    2. ``BROWSER_CDP_URL`` env / ``browser.cdp_url`` config override — the
       ``/browser connect`` path, same precedence the built-in tools honor.
    3. A configured cloud browser provider (Browserbase, Firecrawl, Nous
       gateway/Browser Use cloud, …): reuse the legacy stack's
       ``_get_session_info()`` so browser_exec shares the SAME provider
       session machinery — per-task session cache, expiry replacement,
       inactivity reaper, and atexit cleanup. Direct-API Browser Use cloud
       configs (``provider_key == "browser-use"`` and no gateway) are
       REFUSED: their autospawned browser is unknown to Hermes, so it cannot
       be monitored.
    4. Nothing configured: spawn a Hermes-supervised local headless Chrome
       (``spawn_supervised_chrome``) and point the CLI at it. The no-endpoint
       case no longer exists as a silent "let the harness auto-spawn" path.

    ``session`` (the tool's ``session`` argument / BU_NAME) keys the
    provider session cache when set, so every distinct name gets its OWN
    cloud browser and the same name reuses one — that is what makes named
    sessions actually concurrent-safe on provider backends instead of all
    names sharing a single per-task browser. It also labels the
    supervised-Chrome cache key (per (task_id, session) reuse).

    Returns an error string on any failure, None on success.
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
    if provider is not None:
        # Browser Use direct-API configs: the CLI talks to Browser Use cloud
        # natively (BU_AUTOSPAWN / auth login) — that autospawned browser is
        # by definition unknown to Hermes, so it cannot be CDP-monitored.
        # The operator must pick a monitorable backend. (The Nous-gateway
        # variant, use_gateway: true, DOES resolve through the provider: the
        # gateway provisions the cloud browser server-side and returns its
        # CDP URL.)
        provider_key = str(getattr(provider, "name", "") or "").strip().lower()
        if provider_key == _BACKEND_KEY and not is_truthy_value(
            _read_browser_cfg().get("use_gateway"), default=False
        ):
            return (
                "direct-API Browser Use cloud config cannot be CDP-monitored; "
                "set `browser.cdp_url`/`BROWSER_CDP_URL` or "
                "`browser.use_gateway: true`"
            )
        try:
            # Named sessions get their OWN provider browser, keyed by name so
            # the same name reuses one browser across calls and tasks, and
            # different names never collide. Unnamed calls keep the per-task
            # key.
            cache_key = f"bu-named-{session}" if session else (task_id or "browser-exec-default")
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
        if session:
            env[_PRIVATE_BROWSER_SENTINEL] = "1"
        return None

    # 4. Hermes-supervised local Chrome fallback — the exec runs monitored
    #    or it errors; there is no unmonitorable auto-spawn path anymore.
    try:
        from tools.browser_exec_monitor import spawn_supervised_chrome

        tag = f"{task_id or 'default'}:{session or 'default'}"
        ws = spawn_supervised_chrome(tag)
    except Exception as e:
        return (
            f"browser_exec requires a Hermes-resolvable CDP endpoint for its "
            f"network monitor, and no local Chrome could be started: {e}. "
            "Set `browser.cdp_url`/`BROWSER_CDP_URL` or configure a cloud "
            "provider."
        )
    env["BU_CDP_WS"] = ws
    return None


def _resolve_backend_cdp(env: dict, task_id: Optional[str]) -> Optional[str]:
    """Back-compat wrapper around :func:`_ensure_exec_cdp_endpoint`.

    Kept for existing callers/tests; ``browser_exec`` itself calls
    ``_ensure_exec_cdp_endpoint`` directly (the session= path included).
    """
    return _ensure_exec_cdp_endpoint(env, task_id, None)


def _exec_monitor_config() -> dict:
    """Region A monitor config knobs from the ``browser:`` section."""
    cfg = _read_browser_cfg()
    return {
        "enabled": not (
            str(cfg.get("exec_network_monitor") or "").strip().lower() == "off"
            or cfg.get("exec_network_monitor") is False
        ),
        "fail_open": is_truthy_value(cfg.get("exec_monitor_fail_open"), default=False),
        "grace_s": float(cfg.get("exec_monitor_grace_s") or 1.0),
        "attach_timeout_s": float(cfg.get("exec_monitor_attach_timeout_s") or 15.0),
    }


def _start_exec_network_monitor(env: dict, task_id: Optional[str]):
    """Start the Region A CDP network monitor for one exec window.

    Returns the ``NetworkExecMonitor`` instance, or None when the monitor is
    disabled by config (``browser.exec_network_monitor: off`` — the
    operator's explicit opt-out; the result is annotated ``"monitor":
    "disabled"``). Reads the endpoint from the exact dict handed to the CLI
    (``BU_CDP_WS`` / ``BU_CDP_URL`` — single source of truth), normalized
    via ``_resolve_cdp_override``. Never raises: any start failure leaves the
    monitor in ``attach_failed`` state, which the caller's decision rule
    treats as withhold.
    """
    cfg = _exec_monitor_config()
    if not cfg["enabled"]:
        return None
    cdp_url = env.get("BU_CDP_WS") or env.get("BU_CDP_URL") or ""
    if not cdp_url:
        return None
    try:
        from tools.browser_tool import _resolve_cdp_override

        cdp_url = _resolve_cdp_override(cdp_url) or cdp_url
    except Exception as e:
        logger.debug("CDP endpoint normalization failed (%s); using raw URL", e)
    try:
        from tools.browser_exec_monitor import NetworkExecMonitor

        monitor = NetworkExecMonitor(cdp_url, task_id=str(task_id or "default"))
        monitor.start(timeout=cfg["attach_timeout_s"])
        return monitor
    except Exception as e:
        logger.debug("browser_exec network monitor failed to start: %s", e)
        return None


def _exec_network_violation_error(violation: dict) -> dict:
    """Build the withhold error for a latched monitor violation (Region A H3)."""
    url = str(violation.get("url") or "")
    policy = str(violation.get("policy") or "private")
    event = str(violation.get("event") or "Network.requestWillBeSent")
    return {
        "success": False,
        "error": (
            f"Blocked: during execution the browser requested a URL the "
            f"navigation policy rejects ({url} — {policy}); all output was "
            f"withheld. Intermediate requests are monitored via CDP Network "
            f"events ({event}); the final landing check alone cannot detect "
            f"this."
        ),
    }


def _monitor_unverified_error() -> dict:
    return {
        "success": False,
        "error": (
            "browser network monitoring could not be verified attached and "
            "active; output withheld"
        ),
    }


def browser_exec(
    code: str,
    session: str = "",
    timeout_s: int = _DEFAULT_TIMEOUT_S,
    task_id: Optional[str] = None,
):
    """Run Python code through the browser-use CLI, and return its output

    Security posture (network-boundary class closure, PR #84999):
    - ``_blocked_url_in_code`` checks URL *literals* in the source up front
      (strict mode: unconditional w.r.t. backend, fail-closed on DNS).
    - ``_with_url_recheck`` appends a trailer that reports the *final* landed
      URL (window.location.href) after execution; the trusted boundary probe
      re-observes it from a SECOND, workspace-stripped CLI subprocess.
    - A Hermes-side CDP network monitor (Region A) attaches to the SAME
      endpoint the CLI drives (guaranteed for every path incl. ``session=``)
      and validates EVERY intermediate request — navigation, redirect hop,
      subresource, iframe, new tab, cache-served, WS upgrade — with a
      strict, ungated, fail-closed predicate; any violation withholds ALL
      output.
    - A socket egress interposer (Region B) blocks private-external direct
      connects from the CLI subprocess (loopback + public allowed; IMDS
      floor unconditional), with tamper-evident markers.
    - A CDP Fetch page guard (Region C) enforces per-request at the network
      boundary (pre-connect gate + browser-side remote-IP gate).
    - The end-state landing recheck (Region D) consumes the shared
      ``resolve_and_check_url`` helper: ``error:dns`` blocks even when proxy
      env vars are set.
    - The final verdict applies the Region E agreement truth table
      (coverage precondition + violation/last-known/probe/tri-state rows).
      Output is released only when the guard was verified attached and
      active, no violation was observed, and the landing is safe.
    """
    from tools.registry import tool_error, tool_result

    if not code or not code.strip():
        return tool_error("No code provided. Pass Python that uses the pre-imported helpers, e.g. new_tab(\"https://example.com\") then print(page_info()).")

    blocked = _blocked_url_in_code(code)
    if blocked:
        return tool_error(blocked)

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
    # Region A C1: BOTH branches resolve a monitorable CDP endpoint. BU_NAME
    # is the daemon label; BU_CDP_WS/BU_CDP_URL are the endpoint the daemon
    # drives — orthogonal, both set. No endpoint → error before spawn.
    backend_err = _ensure_exec_cdp_endpoint(env, task_id, session or None)
    if backend_err:
        return tool_error(backend_err)

    # On a SHARED browser (local Chrome / CDP override) a fresh named daemon
    # attaches to the first existing page — the same page a sibling daemon
    # may hold. Pin each named session to a tab it created before running
    # the model's code. Private per-name browsers (provider-keyed) skip
    # this: no one to collide with, and the extra tab would leak.
    private_browser = env.pop(_PRIVATE_BROWSER_SENTINEL, None)
    if session and not private_browser:
        code = _OWN_TAB_PREAMBLE + code

    workspace = _workspace_dir(task_id)
    if workspace:
        env["BH_AGENT_WORKSPACE"] = workspace

    try:
        timeout = max(_MIN_TIMEOUT_S, min(int(timeout_s), _MAX_TIMEOUT_S))
    except (TypeError, ValueError):
        timeout = _DEFAULT_TIMEOUT_S

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

    # Region E H1 — arm the guard stack (A listener + C Fetch guard) BEFORE
    # the CLI spawns. Any arm failure fails the exec closed. The module is
    # imported once and called via its attributes so tests can patch the
    # module-level hooks.
    import tools.browser_use_guard as _exec_guard

    guard_ctx = _exec_guard._prepare_guard(env, task_id, session or None, popen_extra=popen_extra)
    if guard_ctx.get("error"):
        if guard_ctx.get("monitor") is not None:
            guard_ctx["monitor"].stop()
        _exec_guard._teardown_ssrf_guard(guard_ctx.get("ssrf_guard"))
        return tool_error(guard_ctx["error"])
    guard_enabled = bool(guard_ctx.get("enabled"))

    guard_env = env
    if guard_enabled:
        try:
            guard_env = _exec_guard._guard_env(env, guard_ctx)
        except Exception as e:
            if guard_ctx.get("monitor") is not None:
                guard_ctx["monitor"].stop()
            _exec_guard._teardown_ssrf_guard(guard_ctx.get("ssrf_guard"))
            return tool_error(f"browser_exec guard environment failed: {e}")
        # Region E H3 — self-test BEFORE spawn; failure = withhold.
        self_test_err = _exec_guard._guard_self_test(guard_ctx, guard_env)
        if self_test_err:
            if guard_ctx.get("monitor") is not None:
                guard_ctx["monitor"].stop()
            _exec_guard._teardown_ssrf_guard(guard_ctx.get("ssrf_guard"))
            return tool_error(self_test_err)

    exec_started = time.monotonic()
    guard_ctx["exec_started"] = exec_started

    started = time.time()
    try:
        if guard_enabled:
            run = _exec_guard._run_guarded_cli(
                cmd, guard_env, _with_url_recheck(code),
                popen_extra, timeout, guard_ctx,
            )
            if run.get("launch_error"):
                return tool_error(f"Failed to launch browser-use CLI: {run['launch_error']}")
            if run.get("timed_out"):
                return tool_error(
                    f"browser-use exec timed out after {timeout}s. The daemon may "
                    "still be working; retry with a larger timeout_s (max "
                    f"{_MAX_TIMEOUT_S}), or split the work into several calls that "
                    "append to workspace files — anything already written to the "
                    "workspace is preserved."
                )
        else:
            try:
                proc = subprocess.run(
                    cmd,
                    input=_with_url_recheck(code),
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env=env,
                    **popen_extra,
                )
            except subprocess.TimeoutExpired:
                return tool_error(
                    f"browser-use exec timed out after {timeout}s. The daemon may "
                    "still be working; retry with a larger timeout_s (max "
                    f"{_MAX_TIMEOUT_S}), or split the work into several calls that "
                    "append to workspace files — anything already written to the "
                    "workspace is preserved."
                )
            except OSError as e:
                return tool_error(f"Failed to launch browser-use CLI: {e}")
            run = {
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "timed_out": False,
                "guard_blocked": False,
                "guard_died": False,
                "markers": {"armed": None, "announce": None},
                "egress_reason": None,
            }

        # Post-navigation recheck. The pre-check can only see URL literals in the
        # source; this observes where the browser actually ended up. Per the P1
        # review, the observation MUST come from outside the model's namespace:
        # the in-script trailer can be forged by rebinding js/print/str, and the
        # model could poison the workspace agent_helpers.py the probe would
        # otherwise auto-load, so the authoritative URL is taken from a SECOND,
        # trusted CLI subprocess that runs only Hermes-authored probe code and
        # INHERITS NO model-controlled workspace/bootstrap env (so a planted
        # helper cannot rewrite it). Output is withheld rather than annotated.
        landed = _trusted_landed_url(cmd, guard_env, popen_extra, timeout)
        monitor = guard_ctx.get("monitor") if guard_enabled else None
        if landed and monitor is not None:
            # The probe connected through the SAME env → same endpoint →
            # same browser while the monitor was still attached: that is
            # exec-window activity for navigation-free execs.
            monitor.mark_probe_success()

        if guard_enabled:
            verdict = _exec_guard._guard_endstate_verdict(guard_ctx, landed, run)
            if verdict["verdict"] == "withhold":
                return tool_error(verdict["reason"])
            monitor_note = verdict.get("note") or "armed"
        else:
            monitor_note = "disabled"
            # Defense-in-depth landing recheck (Region D site 6) still runs
            # when the operator disabled the monitor: the shared helper
            # fails closed on DNS regardless of proxy env.
            if landed:
                from tools.browser_tool import _resolve_and_check_url

                v = _resolve_and_check_url(landed)
                if not v.ok:
                    if v.reason in (
                        "blocked:metadata-host", "blocked:metadata-ip",
                        "blocked:link-local", "blocked:ipv4-compatible",
                    ):
                        return tool_error(
                            "Blocked: URL targets a cloud metadata endpoint — "
                            "the browser ended on this address after the code "
                            "ran, so the page output was withheld."
                        )
                    if v.reason in ("error:dns", "error:internal"):
                        return tool_error(
                            "Blocked: the destination could not be safely "
                            "verified (DNS resolution failed) — page output "
                            "was withheld."
                        )
                    return tool_error(
                        "Blocked: URL targets a private or internal address — "
                        "the browser ended on this address after the code "
                        "ran, so the page output was withheld."
                    )

        result = {
            "success": run["returncode"] == 0,
            "exit_code": run["returncode"],
            "output": _exec_guard._strip_guard_markers(
                _strip_landed_url_marker(run["stdout"])
            ),
        }
        result["monitor"] = monitor_note
        if workspace:
            result["workspace"] = workspace
        if session:
            result["session"] = session
        stderr = _exec_guard._strip_guard_markers(run["stderr"] or "").strip()
        if stderr:
            if len(stderr) > _STDERR_CAP_CHARS:
                stderr = stderr[:_STDERR_CAP_CHARS] + "\n… (stderr truncated)"
            result["stderr"] = stderr

        screenshot = _find_screenshot(run["stdout"], started)
        if screenshot:
            result["screenshot_path"] = screenshot
            native = _native_screenshot_result(result, screenshot)
            if native is not None:
                return native
        return tool_result(result)
    finally:
        if guard_ctx.get("monitor") is not None:
            guard_ctx["monitor"].stop()
        _exec_guard._teardown_ssrf_guard(guard_ctx.get("ssrf_guard"))


# The tool description is the CLI's skill, fetched from browser-use skill
_HEADER_BASE = (
    "Drive a real web browser via the Browser Use CLI. The `code` argument "
    "is piped verbatim to the `browser-use` CLI on stdin and executed as "
    "full Python (standard library available) with the CLI's pre-imported "
    "browser helpers; stdout comes back in the result. Start `code` with a "
    "one-line comment describing the step for the user in plain, "
    "non-technical language, max 60 chars (e.g. `# Searching Amazon for "
    "paper towels`) — the UI displays it as the step label.\n\n"
    "STATE: the browser session and the workspace persist across calls; "
    "Python variables do NOT (each call is a fresh interpreter). The "
    "workspace is a stable directory — path in $BH_AGENT_WORKSPACE and "
    "returned as `workspace` in every result. For multi-item tasks "
    "('collect all N products / every entry / the full table'), append each "
    "batch to a JSON/CSV file in the workspace as you go, then read it back "
    "to assemble the final answer; define reusable functions in "
    "agent_helpers.py there — the harness auto-imports it into every call. "
    "Do aggregation in code, not in your head: dedupe, count, sort, and "
    "format with Python inside the exec. Before giving a final answer on a "
    "multi-item task, verify the collected count against what was asked "
    "and go back for anything missing.\n\n"
    "Batch each sub-procedure (navigate, wait, extract, act) into one call "
    "— do not spend a call per action — but for long extractions prefer "
    "several medium calls that append to workspace files over one giant "
    "call, so progress survives timeouts. For an isolated concurrent "
    "browser session (parallel tasks that must not share tabs), pass "
    "session=<name> (never BU_NAME env syntax) and reuse the same name on "
    "every related call."
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

_DESCRIPTION_HEADER = _HEADER_BASE  # back-compat alias for external imports

# NOTE: browser_exec is additionally gated at tool-definition time — sessions
# whose resolved toolsets do not include ``terminal`` never see it (see
# model_tools._compute_tool_definitions). The check_fn registered below only
# answers "is Browser Use mode configured"; surface policy lives with the
# session, not in the process-wide TTL-cached check_fn.


def _description_header() -> str:
    """Header tailored to whether the active model can see images natively"""
    try:
        from tools.vision_tools import _should_use_native_vision_fast_path

        if _should_use_native_vision_fast_path():
            return _HEADER_BASE + _HEADER_VISION
    except Exception:
        pass
    return _HEADER_BASE + _HEADER_TEXT_ONLY

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
    return {"description": _description_header() + _HELPERS_DIGEST}


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
                "description": "Named isolated browser session (sets BU_NAME): each name gets its own harness daemon — and on cloud backends its own browser — so concurrent tasks don't clobber each other. Omit for the shared default session. Reuse the same name across calls to keep working in that session (and the name passed to start_remote_daemon(), if used).",
            },
            "timeout_s": {
                "type": "integer",
                "description": f"Max seconds to wait for the code to finish (default {_DEFAULT_TIMEOUT_S}, max {_MAX_TIMEOUT_S}).",
                "default": _DEFAULT_TIMEOUT_S,
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
        task_id=kw.get("task_id"),
    ),
    check_fn=is_browser_use_cli_mode,
    dynamic_schema_overrides=_dynamic_schema_overrides,
    emoji="🌐",
)
