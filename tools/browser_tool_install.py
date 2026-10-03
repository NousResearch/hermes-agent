"""PM-owned agent-browser / Chromium discovery, acquisition and readiness.

Split out of ``tools/browser_tool.py``. Facade-owned state is read through ``_bt`` (``tools.browser_tool``, resolved per call) — no import cycle."""

import functools
import os
import re
import subprocess
from typing import Optional

from hermes_cli._subprocess_compat import windows_hide_flags
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

class AgentBrowserCapabilityError(RuntimeError):
    """The selected CLI/runtime cannot provide strict shared-CDP target pinning."""

_SEMVER_TRIPLE_RE = re.compile(r"(?<!\d)(\d+)\.(\d+)\.(\d+)(?![\d-])")

def _version_probe_env() -> dict[str, str]:
    from pm import env_for
    env = _origin()._build_browser_env()
    env["PATH"] = _merge_browser_path(env.get("PATH", ""))
    return env_for("agent-browser", base_env=env)

def _probe_agent_browser_version(path: str) -> Optional[tuple[int, int, int]]:
    """Run a concrete CLI's real ``--version`` entrypoint and parse semver."""
    if not os.path.exists(path) or (os.name != "nt" and not os.access(path, os.X_OK)):
        return None
    try:
        result = subprocess.run(
            [path, "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            env=_version_probe_env(),
            creationflags=windows_hide_flags(),
            check=False,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    if result.returncode != 0:
        return None
    match = _SEMVER_TRIPLE_RE.search(f"{result.stdout}\n{result.stderr}")
    return tuple(int(part) for part in match.groups()) if match is not None else None

def _pin_tab_candidate_status(path: str) -> bool:
    """Verify the concrete command supports the pinning protocol before page access.

    PM selects the native Rust CLI/daemon; it does not require Node. External
    npm wrappers retain their own published engine requirements.
    """
    version = _probe_agent_browser_version(path)
    if version is None or version < _origin().AGENT_BROWSER_PIN_TAB_MIN_VERSION:
        return False
    try:
        result = subprocess.run([path, "--help"], capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=10,
                                env=_version_probe_env(), creationflags=windows_hide_flags(), check=False)
    except (OSError, subprocess.SubprocessError, ValueError):
        return False
    help_text = f"{result.stdout}\n{result.stderr}"
    return result.returncode == 0 and all(flag in help_text for flag in ("--pin-tab", "--session", "--cdp"))

def _pin_tab_capability_error() -> str:
    return ("Shared-CDP page isolation requires agent-browser >=0.34.0 with --pin-tab support. "
            "Install the pinned native runtime with 'hermes pm install agent-browser', then retry. "
            "An external compatible agent-browser on PATH is also supported. No unpinned package is downloaded.")

def _find_agent_browser(*, validate: bool = True, require_pin_tab: bool = False) -> str:
    """Use PM's selected executable, then an external PATH/Homebrew installation.

    Shared CDP requires a verified pin-tab-capable CLI/runtime before any browser
    command is dispatched. PM remains the only lazy-install owner; an older pin
    fails closed instead of introducing an alternate download path. Readiness
    checks without pinning never execute or acquire a package. Selection is not
    cached, so updated PM facts and profile changes are visible immediately.
    """
    import pm

    termux = _is_termux_environment()
    checked = set()

    def usable(candidate):
        if candidate in checked:
            return False
        checked.add(candidate)
        if require_pin_tab:
            return _pin_tab_candidate_status(candidate)
        return (agent_browser_runnable if validate else _agent_browser_candidate_present)(candidate)

    if not termux:
        installed = pm.installed_package("agent-browser")
        if installed and installed.binary is not None:
            candidate = str(installed.binary)
            if not require_pin_tab or usable(candidate):
                return candidate
    for search_path in (None, _merge_browser_path("")):
        if search_path == "":
            continue
        candidate = shutil.which("agent-browser", path=search_path)
        if candidate and usable(candidate):
            return candidate
    if require_pin_tab and (termux or pm.installed_package("agent-browser") is not None):
        raise AgentBrowserCapabilityError(_pin_tab_capability_error())
    hint = f"agent-browser CLI not found. Install it with: {_browser_install_hint()}"
    if validate and not termux:
        try:
            pm.ensure("agent-browser")
        except (pm.InstallError, OSError) as exc:
            raise FileNotFoundError(f"{hint}\n{exc}") from exc
        installed = pm.installed_package("agent-browser")
        if installed and installed.binary is not None:
            candidate = str(installed.binary)
            if not require_pin_tab or _pin_tab_candidate_status(candidate):
                return candidate
    if require_pin_tab:
        raise AgentBrowserCapabilityError(_pin_tab_capability_error())
    raise FileNotFoundError(hint)

def warm_agent_browser_npx_cache(timeout: float = 60.0) -> bool:
    """Frozen old-updater surface names this module too (the extraction-era home); tools.browser_tool
    carries the permanent definition. No npx work is performed; relaunch instead."""
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
        with open("/proc/1/cgroup", "rt", encoding="utf-8") as fp:
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
    from tools.vision_tools import check_vision_requirements
    return check_vision_requirements()
