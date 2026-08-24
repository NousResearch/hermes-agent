"""agent-browser / Chromium discovery and install: PATH merging, npx resolution, candidate binaries, Chromium detection + auto-install, requirement checks.

Split out of ``tools/browser_tool.py``. Facade-owned state is read through ``_bt`` (``tools.browser_tool``, resolved per call) — no import cycle."""

import contextlib
import functools
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from hermes_cli._subprocess_compat import windows_hide_flags
from hermes_constants import agent_browser_runnable, get_hermes_home, is_termux as _is_termux_environment, node_tool_runnable
from tools.browser_tool_origin import origin_module as _origin
from tools import browser_tool_cdp as _cdp
from tools import browser_tool_cloud as _cloud
from tools import browser_tool_lifecycle as _lifecycle
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
    """Return ordered browser CLI PATH candidates shared by discovery and execution."""
    _bt = _origin()
    home = get_hermes_home()
    managed = (home / "node" / "bin", home / "node", home / "node_modules" / ".bin")
    return [*map(str, managed), *_discover_homebrew_node_dirs(), *_bt._SANE_PATH_DIRS]


def _merge_browser_path(existing_path: str = "") -> str:
    """Prepend browser-specific PATH fallbacks without reordering existing entries."""
    path_parts = [p for p in (existing_path or "").split(os.pathsep) if p]
    prefix_parts: list[str] = []
    for part in _browser_candidate_path_dirs():
        if part and part not in path_parts and part not in prefix_parts and os.path.isdir(part):
            prefix_parts.append(part)
    return os.pathsep.join(prefix_parts + path_parts)


def _browser_install_hint() -> str:
    return "npm install -g agent-browser && agent-browser install" + ("" if _is_termux_environment() else " --with-deps")


def _is_npx_agent_browser_sentinel(browser_cmd: str) -> bool:
    return browser_cmd.strip() == _origin().NPX_AGENT_BROWSER_SENTINEL


def _requires_real_termux_browser_install(browser_cmd: str) -> bool:
    return _is_termux_environment() and _cloud._is_local_mode() and _is_npx_agent_browser_sentinel(browser_cmd)


def _termux_browser_install_error() -> str:
    return f"Local browser automation on Termux cannot rely on the bare npx fallback. Install agent-browser explicitly first: {_browser_install_hint()}"


def _agent_browser_candidate_present(path: str | None) -> bool:
    if not path:
        return False
    if " " in path and path.split()[0].endswith("npx"):
        return True
    return os.path.exists(path) and (os.name == "nt" or os.access(path, os.X_OK))


def _resolve_npx_bin() -> Optional[str]:
    """Resolve a runnable npx, extended (Hermes-managed/Homebrew) PATH first.

    Bare PATH first would let a broken system npx shadow a healthy managed one,
    so every candidate is validated with ``node_tool_runnable`` before use.
    """
    extended_path = _merge_browser_path("")
    for path in ([extended_path] if extended_path else []) + [None]:
        npx = shutil.which("npx", path=path)
        if npx and node_tool_runnable(npx):
            return npx
    return None


def _agent_browser_candidates(extended_path: str):
    """Yield agent-browser lookup candidates lazily: ambient PATH → extended PATH → repo-local node_modules/.bin.

    The local lookup uses ``shutil.which`` with an explicit path so Windows resolves the ``.cmd`` shim
    (CreateProcess cannot run npm's extensionless POSIX shim — WinError 193).
    """
    yield shutil.which("agent-browser")
    if extended_path:
        yield shutil.which("agent-browser", path=extended_path)
    local_bin_dir = Path(__file__).parent.parent / "node_modules" / ".bin"
    if local_bin_dir.is_dir():
        yield shutil.which("agent-browser", path=str(local_bin_dir))


def _find_agent_browser(*, validate: bool = True) -> str:
    """Find the agent-browser CLI: PATH, Homebrew/managed dirs, local node_modules/.bin, npx fallback, lazy install.

    A bare ``shutil.which`` hit is NOT trusted: agent-browser's npm postinstall re-points a global symlink at our
    local node_modules binary, which vanishes on the next ``hermes update`` and leaves a dangling link ``which``
    still reports (exec fails with 127). Candidates are validated with ``agent_browser_runnable`` before caching
    so a dead one falls through. ``validate=False`` (schema-time check_fn) only tests presence and never caches.
    Raises FileNotFoundError when agent-browser is not installed.
    """
    _bt = _origin()

    def _not_found(cached: bool) -> FileNotFoundError:
        return FileNotFoundError(f"agent-browser CLI not found{' (cached)' if cached else ''}. Install it with: "
                                 f"{_browser_install_hint()}\nOr ensure npx is available in your PATH.")

    def _accept(candidate: str) -> str:
        # Set resolved at each accept site (not before the search) so a concurrent reader never sees
        # resolved=True with a None cache.
        if validate:
            _bt._cached_agent_browser = candidate
            _bt._agent_browser_resolved = True
        return candidate

    if _bt._agent_browser_resolved:
        if _bt._cached_agent_browser is None:
            raise _not_found(cached=True)
        return _bt._cached_agent_browser
    ok = agent_browser_runnable if validate else _agent_browser_candidate_present
    extended_path = _merge_browser_path("")
    for candidate in _agent_browser_candidates(extended_path):
        if candidate and ok(candidate):
            return _accept(candidate)
    # npx fallback (also searches the extended PATH)
    if _resolve_npx_bin():
        return _accept(_bt.NPX_AGENT_BROWSER_SENTINEL)
    if not validate:
        raise FileNotFoundError("agent-browser CLI not found")
    try:  # Nothing found — try lazy installation before giving up.
        from hermes_cli.dep_ensure import ensure_dependency
        if ensure_dependency("browser"):
            home = get_hermes_home()
            managed = (home / "node_modules" / ".bin", home / "node" / "bin", home / "node")
            for path in (None, *([extended_path] if extended_path else []), *map(str, managed)):
                recheck = shutil.which("agent-browser", path=path)
                if recheck and agent_browser_runnable(recheck):
                    return _accept(recheck)
    except Exception:
        pass
    _bt._agent_browser_resolved = True
    raise _not_found(cached=False)


def warm_agent_browser_npx_cache(timeout: float = 60.0) -> bool:
    """Best-effort pre-fetch of the agent-browser npm package via npx (``hermes update`` / ``doctor --fix``).

    Runs with the credential-scrubbed env every other agent-browser spawn uses (registry-fetched npm code must
    never see the operator keyring), in its own process group, and tree-kills on timeout so a surviving
    descendant cannot hold the capture pipe open. Never raises; True only when npx exited 0.

    agent-browser is no longer a root package.json dependency (#43564) — it resolves lazily via ``npx
    agent-browser`` instead, which keeps it out of the npm workspace install graph entirely (nothing to
    prune it anymore) but means the first real invocation in a session would otherwise pay npx's
    registry-lookup/fetch cost. Calling this during ``hermes update`` (or ``hermes doctor --fix``) warms
    npx's own cache ahead of time, restoring the "available before any session starts" property
    agent-browser had while it was an eager root dependency — without re-entangling it with the workspace
    graph.
    """
    _bt = _origin()
    npx_bin = _resolve_npx_bin()
    if not npx_bin:
        return False
    env = _bt._build_browser_env()
    env["PATH"] = _merge_browser_path(env.get("PATH", ""))
    popen_kwargs: dict = {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE, "text": True, "env": env}
    if os.name == "posix":
        popen_kwargs.update(creationflags=windows_hide_flags(), start_new_session=True)
    else:
        popen_kwargs["creationflags"] = windows_hide_flags() | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    # --ignore-scripts: AGENT_BROWSER_NPX_SPEC is a floating range; a compromised future patch must not run
    # install-time lifecycle scripts here. --prefer-offline: once cached, repeat runs must not re-hit the registry.
    cmd = [npx_bin, "--ignore-scripts", "--prefer-offline", "-y", _bt.AGENT_BROWSER_NPX_SPEC, "--version"]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, **popen_kwargs)
    except Exception:
        return False
    try:
        proc.communicate(timeout=timeout)
        return proc.returncode == 0
    except Exception as exc:
        _lifecycle._kill_process_tree(proc)
        if isinstance(exc, subprocess.TimeoutExpired):
            with contextlib.suppress(Exception):
                proc.communicate(timeout=5)
        return False


def _chromium_search_roots() -> List[str]:
    """Chromium / headless-shell scan roots in agent-browser/Playwright probe order: ``PLAYWRIGHT_BROWSERS_PATH``, then the per-OS default cache."""
    env_path = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "").strip()
    home = os.path.expanduser("~")
    roots: List[str] = [env_path] if env_path and env_path != "0" else []
    roots.append(os.path.join(home, ".cache", "ms-playwright"))
    if sys.platform == "darwin":
        roots.append(os.path.join(home, "Library", "Caches", "ms-playwright"))
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA") or os.path.join(home, "AppData", "Local")
        roots.append(os.path.join(local, "ms-playwright"))
    return roots


def _has_chromium_build(root: str) -> bool:
    """True when ``root`` holds a Playwright ``chromium-*`` / ``chromium_headless_shell-*`` dir (agent-browser accepts either)."""
    try:
        return any(e.startswith(("chromium-", "chromium_headless_shell-")) for e in os.listdir(root))
    except OSError:
        return False


def _chromium_installed() -> bool:
    """True when a usable Chromium (or headless-shell) build is on disk; cached.

    Checks ``AGENT_BROWSER_EXECUTABLE_PATH``, then system Chrome/Chromium on PATH, then Playwright's cache.
    Without a binary the CLI hangs on first use until the command timeout fires, so the tool must not be advertised.
    """
    _bt = _origin()
    if _bt._cached_chromium_installed is not None:
        return _bt._cached_chromium_installed
    ab_path = os.environ.get("AGENT_BROWSER_EXECUTABLE_PATH", "").strip()
    _bt._cached_chromium_installed = bool(
        (ab_path and (os.path.isfile(ab_path) or shutil.which(ab_path)))
        or any(shutil.which(name) for name in ("google-chrome", "chromium", "chromium-browser", "chrome"))
        or any(root and os.path.isdir(root) and _has_chromium_build(root) for root in _chromium_search_roots())
    )
    return _bt._cached_chromium_installed


def _maybe_autoinstall_chromium() -> bool:
    """Best-effort, gated download of the Chromium *binary* on local cold start.

    Binary only (``agent-browser install``), never ``--with-deps`` — that shells ``apt`` and needs root. Gated by
    ``security.allow_lazy_installs``, skipped in Docker (Chromium ships in the image), attempted once per process.
    """
    _bt = _origin()
    if _bt._chromium_autoinstall_attempted:
        return _chromium_installed()
    _bt._chromium_autoinstall_attempted = True
    if _running_in_docker():
        return False
    from tools.lazy_deps import _allow_lazy_installs
    if not _allow_lazy_installs():
        return False
    try:
        browser_cmd = _find_agent_browser()
    except FileNotFoundError:
        return False
    install_cmd = [browser_cmd, "install"]
    if _is_npx_agent_browser_sentinel(browser_cmd):
        install_cmd = [_resolve_npx_bin() or "npx", "--ignore-scripts", "-y", _bt.AGENT_BROWSER_NPX_SPEC, "install"]

    _bt.logger.info("browser: Chromium missing — auto-installing the browser binary (one-time ~170MB; disable via security.allow_lazy_installs)")
    try:
        proc = subprocess.run(install_cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=600,
                              env=_bt._build_browser_env(), stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError) as e:
        _bt.logger.warning("browser: Chromium auto-install failed to start: %s", e)
        return False
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-300:]
        _bt.logger.warning("browser: Chromium auto-install exited %s: %s", proc.returncode, tail)
        return False
    _bt._cached_chromium_installed = None
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
        browser_cmd = _find_agent_browser(validate=False)
    except FileNotFoundError:
        return False
    # Termux: the bare npx fallback is too fragile to advertise as a satisfied local dependency.
    if _requires_real_termux_browser_install(browser_cmd):
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
