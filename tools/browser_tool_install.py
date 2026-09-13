"""PM-owned agent-browser / Chromium discovery, acquisition and readiness.

Split out of ``tools/browser_tool.py``. Facade-owned state is read through ``_bt`` (``tools.browser_tool``, resolved per call) — no import cycle."""

import functools
import os
import shutil

from hermes_constants import agent_browser_runnable, is_termux as _is_termux_environment
from tools.browser_tool_origin import origin_module as _origin
from tools import browser_tool_cdp as _cdp
from tools import browser_tool_cloud as _cloud
from tools import browser_tool_lightpanda_fallback as _lp


@functools.lru_cache(maxsize=1)
def _discover_homebrew_node_dirs() -> tuple[str, ...]:
    """Homebrew versioned Node bin dirs (node@20, ...) that ``brew`` may not link into /opt/homebrew/bin."""
    homebrew_opt = "/opt/homebrew/opt"
    try:
        entries = os.listdir(homebrew_opt) if os.path.isdir(homebrew_opt) else []
    except OSError:
        entries = []
    return tuple(
        bin_dir
        for entry in entries
        if entry.startswith("node") and entry != "node"
        if os.path.isdir(bin_dir := os.path.join(homebrew_opt, entry, "bin"))
    )


def _browser_candidate_path_dirs() -> list[str]:
    """System PATH fallbacks for externally owned browser helpers."""
    _bt = _origin()
    return [*_discover_homebrew_node_dirs(), *_bt._SANE_PATH_DIRS]


def _merge_browser_path(existing_path: str = "") -> str:
    """Prepend browser-specific PATH fallbacks without reordering existing entries."""
    path_parts = [p for p in (existing_path or "").split(os.pathsep) if p]
    prefix_parts: list[str] = []
    for part in _browser_candidate_path_dirs():
        if part and part not in path_parts and part not in prefix_parts and os.path.isdir(part):
            prefix_parts.append(part)
    return os.pathsep.join(prefix_parts + path_parts)


def _browser_install_hint() -> str:
    if _is_termux_environment():
        return "npm install -g agent-browser && agent-browser install"
    return "hermes pm install agent-browser (system libraries: npx playwright install-deps chromium)"


def _agent_browser_candidate_present(path: str | None) -> bool:
    if not path:
        return False
    return os.path.isfile(path) and (os.name == "nt" or os.access(path, os.X_OK))


def _find_agent_browser(*, validate: bool = True) -> str:
    """Select PM's exact binary, then an external PATH/Homebrew installation.

    Termux owns its browser installation. Elsewhere PM may acquire a missing
    CLI at execution time, subject to its lazy-install policy. Readiness checks
    (``validate=False``) never execute or install anything. Selection is not
    cached: a new PM fact or profile must be visible immediately.
    """
    import pm

    termux = _is_termux_environment()
    if not termux:
        installed = pm.installed_package("agent-browser")
        if installed and installed.binary is not None:
            return str(installed.binary)
    usable = agent_browser_runnable if validate else _agent_browser_candidate_present
    for search_path in (None, _merge_browser_path("")):
        if search_path == "":
            continue
        candidate = shutil.which("agent-browser", path=search_path)
        if candidate and usable(candidate):
            return candidate
    hint = f"agent-browser CLI not found. Install it with: {_browser_install_hint()}"
    if validate and not termux:
        try:
            pm.ensure("agent-browser")
        except (pm.InstallError, OSError) as exc:
            raise FileNotFoundError(f"{hint}\n{exc}") from exc
        installed = pm.installed_package("agent-browser")
        if installed and installed.binary is not None:
            return str(installed.binary)
    raise FileNotFoundError(hint)


def warm_agent_browser_npx_cache(timeout: float = 60.0) -> bool:
    """Historical updater export: no npx work is performed; relaunch instead."""
    return False


def _chromium_installed() -> bool:
    """An explicit browser executable or PM's selected full Chromium exists."""
    from hermes_cli.browser_runtime import chromium_executable

    ab_path = chromium_executable()
    return bool(ab_path and (os.path.isfile(ab_path) or shutil.which(ab_path)))


def _maybe_autoinstall_chromium() -> bool:
    """Install only PM's pinned full Chromium, never the upstream browser pair.

    Docker supplies the binary. Other installs require lazy-install consent.
    """
    _bt = _origin()
    if _bt._chromium_autoinstall_attempted:
        return _chromium_installed()
    _bt._chromium_autoinstall_attempted = True
    if _running_in_docker() or _is_termux_environment() or os.environ.get("AGENT_BROWSER_EXECUTABLE_PATH"):
        return False
    from pm import InstallError, ensure, lazy_installs_allowed
    if not lazy_installs_allowed():
        return False
    _bt.logger.info("browser: installing PM's pinned Chromium")
    try:
        ensure("chromium")
    except (InstallError, OSError) as exc:
        _bt.logger.warning("browser: Chromium auto-install failed: %s", exc)
        return False
    return _chromium_installed()


def _running_in_docker() -> bool:
    """Best-effort detection of whether we're inside a Docker container."""
    if os.path.exists("/.dockerenv"):
        return True
    try:
        with open("/proc/1/cgroup", "rt", encoding="utf-8-sig") as fp:
            return "docker" in fp.read()
    except OSError:
        return False


def check_browser_requirements() -> bool:
    """Whether the browser tools should be advertised.

    Local mode needs the ``agent-browser`` CLI plus a Chromium build (except Lightpanda-only text workflows);
    cloud mode needs the CLI plus provider credentials (the provider hosts its own Chromium).
    """
    _bt = _origin()
    # Browser Use CLI backend: browser_exec replaces the whole browser_* surface (incl. browser_cdp/browser_dialog check_fns).
    if _bt._is_browser_use_cli_mode():
        return False
    # Camofox only needs the server URL, no agent-browser CLI.
    if _bt._is_camofox_mode():
        return True
    # CDP override needs no local binary. Raw (no-I/O) check: this runs during schema build, where a stale endpoint must not cost a blocking probe.
    if _cdp._get_cdp_override_raw():
        return True
    # Do not exec ``agent-browser --version`` here: Windows .cmd shims flash a console during Desktop startup. Execution paths still validate.
    try:
        _find_agent_browser(validate=False)
    except FileNotFoundError:
        return False

    # Cloud mode also requires provider credentials; no local Chromium needed.
    provider = _cloud._get_cloud_provider()
    if provider is not None:
        return provider.is_available()
    # Lightpanda provides text/navigation tools without Chromium; screenshots/vision still return install errors.
    if _lp._using_lightpanda_engine():
        return True
    # Local Chrome mode needs Chromium on disk or the CLI hangs until the command timeout.
    return _chromium_installed()


def check_browser_vision_requirements() -> bool:
    """Advertise ``browser_vision`` only with BOTH a working browser AND a vision backend.

    Without the vision check, the tool stays in the model's tool list even when no vision provider is
    configured, then fails at call time with a cryptic provider-side error like ``unknown variant
    `image_url`, expected `text``` (issue #31179).
    """
    if not check_browser_requirements():
        return False
    try:
        from tools.vision_tools import check_vision_requirements
    except ImportError:
        return False
    return check_vision_requirements()
