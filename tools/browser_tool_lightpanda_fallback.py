"""Lightpanda engine status and the automatic Chrome fallback for browser commands
that Lightpanda cannot serve (screenshots, empty snapshots, failed commands).

Facade-owned state is read through ``_bt`` (``tools.browser_tool``, resolved per call) — no import cycle.
"""

import json
import os
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Tuple
from tools.browser_tool_origin import origin_module as _origin
from tools import browser_tool_cdp as _cdp
from tools import browser_tool_cloud as _cloud
from tools import browser_tool_install as _install
from tools import browser_tool_session as _session

# Commands where Chrome can meaningfully produce a different result. Session-management
# commands (close, record) are tied to the engine's daemon and can't be retried elsewhere.
_FALLBACK_ELIGIBLE = frozenset({"open", "snapshot", "screenshot", "eval", "click",
                                "fill", "scroll", "back", "press", "console", "errors"})


def _using_lightpanda_engine() -> bool:
    """Return True when local browser commands are configured for Lightpanda."""
    return _cloud._get_browser_engine() == "lightpanda"


def lightpanda_engine_status() -> Tuple[bool, str]:
    """Whether ``browser.engine: lightpanda`` is actually in effect, and why.

    ``(False, "")`` when the engine isn't lightpanda; else the reason names the setting shadowing
    it or the driver running it. Mirrors ``_should_inject_engine`` / ``_resolve_backend_cdp``
    precedence with config-only gates (no network I/O) for ``/browser status`` / ``hermes doctor``.
    """
    _bt = _origin()
    if not _using_lightpanda_engine():
        return False, ""
    if _cdp._get_cdp_override_raw():
        return False, "a CDP override is active (/browser connect or browser.cdp_url)"
    if _bt._is_camofox_mode():
        return False, "Camofox is the selected browser (CAMOFOX_URL)"
    # Real-profile before cloud provider: browser_exec resolves real-profile before
    # the backend, so with both set the real-profile toggle claims the session.
    if _cloud._use_real_profile():
        return False, "browser.use_real_profile is on (Lightpanda cannot load a Chromium profile)"
    try:
        provider = _cloud._get_cloud_provider()
    except Exception:
        provider = None
    if provider is not None:
        try:
            name = provider.display_name
        except Exception:
            name = type(provider).__name__
        return False, f"cloud provider {name} is selected (browser.cloud_provider, or auto-detected from credentials)"
    if not _bt._is_browser_use_cli_mode():
        return True, "built-in browser tools: agent-browser --engine lightpanda"
    try:
        from tools.browser_use_cli import _read_browser_cfg, is_legacy_browser_use_cloud_config
        if is_legacy_browser_use_cloud_config(_read_browser_cfg()):
            return False, "Browser Use cloud (BROWSER_USE_API_KEY) is selected"
    except Exception as e:
        _bt.logger.debug("legacy Browser Use cloud check failed: %s", e)
    return True, "Browser Use mode: Hermes spawns `lightpanda serve` per session"


def _lightpanda_fallback_reason(engine: str, command: str, result: Dict[str, Any]) -> Optional[str]:
    """User-visible reason a Lightpanda result needs the Chrome fallback (copied into the result), or None."""
    if engine != "lightpanda" or command not in _FALLBACK_ELIGIBLE:
        return None
    if not result.get("success"):
        return f"Lightpanda {command!r} failed ({str(result.get('error') or 'command failed').strip()}); retried with Chrome."
    data = result.get("data", {})
    if command == "snapshot" and len((data.get("snapshot", "") or "").strip()) < 20:  # couldn't render
        return "Lightpanda returned an empty/too-short snapshot; retried with Chrome."
    if command == "screenshot" and data.get("path", ""):
        # Lightpanda returns a ~17 KB placeholder PNG (panda logo at 1920x1080); real Chromium is 100 KB+.
        try:
            size = os.path.getsize(data["path"])
        except OSError:
            return "Lightpanda screenshot file was missing/unreadable; retried with Chrome."
        if size < 20480:
            _origin().logger.debug("Lightpanda screenshot is suspiciously small (%d bytes), "
                                   "triggering Chrome fallback", size)
            return f"Lightpanda screenshot was suspiciously small ({size} bytes); " "retried with Chrome."
    return None


def _needs_lightpanda_fallback(engine: str, command: str, result: Dict[str, Any]) -> bool:
    """Check if a Lightpanda result should trigger an automatic Chrome fallback."""
    return _lightpanda_fallback_reason(engine, command, result) is not None


def _annotate_lightpanda_fallback(result: Dict[str, Any], reason: str) -> Dict[str, Any]:
    """Add a user-visible Chrome fallback warning to a browser command result."""
    warning = "⚠ Lightpanda fallback: Chrome was used for this browser action. " f"{reason}"
    fields = {"fallback_warning": warning, "browser_engine": "chrome",
              "browser_engine_fallback": {"from": "lightpanda", "to": "chrome", "reason": reason}}
    annotated = {**result, **fields}
    data = annotated.get("data")
    if isinstance(data, dict):
        annotated["data"] = data = dict(data)
        for key, value in fields.items():
            data.setdefault(key, dict(value) if isinstance(value, dict) else value)
    return annotated


def _copy_fallback_warning(target: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    """Copy browser fallback metadata from an internal result into a tool response."""
    if result.get("fallback_warning"):
        target["fallback_warning"] = result["fallback_warning"]
        target["browser_engine"] = result.get("browser_engine")
        target["browser_engine_fallback"] = result.get("browser_engine_fallback")
    return target


def _run_chrome_fallback_command(task_id: str, command: str, args: List[str], timeout: int) -> Dict[str, Any]:
    """Run a browser command in a temporary Chrome session at the current URL.

    agent-browser locks the engine when a named daemon starts, so ``--engine chrome`` on the
    Lightpanda session is ignored: fresh temp Chrome session -> same URL -> ``command`` -> tear down.
    """
    _bt = _origin()
    import uuid
    # 1. Current URL from the Lightpanda session. ``get url`` is not fallback-eligible,
    # so this can't recurse; the explicit override strips Chromium-only env flags.
    url_result = _session._run_browser_command(task_id, "get", ["url"], timeout=10, _engine_override="lightpanda")
    current_url = str(url_result.get("data", {}).get("url", "")).strip() if url_result.get("success") else None
    if not current_url:
        _bt.logger.warning("Chrome fallback: could not determine current URL from LP session")
        return {"success": False, "error": "Chrome fallback failed: could not determine current URL"}

    # 2. Temporary Chrome session (bypasses _get_session_info's cache).
    tmp_session = f"h_cfb_{uuid.uuid4().hex[:8]}"
    try:
        browser_cmd = _install._find_agent_browser()
    except FileNotFoundError as e:
        return {"success": False, "error": str(e)}

    if not _install._chromium_installed():
        if _install._running_in_docker():
            hint = ("Chrome fallback requires Chromium, but it is missing. You're running in Docker — "
                    "pull the latest image: docker pull ghcr.io/nousresearch/hermes-agent:latest")
        else:
            hint = ("Chrome fallback requires Chromium, but it is missing. Install it with: "
                    "npx agent-browser install --with-deps (or: npx playwright install --with-deps chromium)")
        return {"success": False, "error": hint}

    base_args = _session._agent_browser_argv(browser_cmd) + ["--engine", "chrome", "--session", tmp_session, "--json"]
    task_socket_dir = _session._prepare_session_socket_dir(tmp_session)
    # Bypasses _run_browser_command, so apply the same Chromium sandbox policy explicitly.
    browser_env = _session._agent_browser_command_env(task_socket_dir)
    _session._apply_chromium_sandbox_args(browser_env)

    def _run_tmp(cmd: str, cmd_args: List[str]) -> Dict[str, Any]:
        proc = _session._popen_agent_browser(base_args + [cmd] + cmd_args, browser_env, task_socket_dir, cmd)
        stdout_path = os.path.join(task_socket_dir, f"_stdout_{cmd}")
        stderr_path = os.path.join(task_socket_dir, f"_stderr_{cmd}")
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            return {"success": False, "error": f"Chrome fallback '{cmd}' timed out"}
        try:
            with open(stdout_path, encoding="utf-8") as f:
                stdout = f.read().strip()
            if stdout:
                return json.loads(stdout.split("\n")[-1])
        except Exception as exc:
            _bt.logger.debug("Chrome fallback tmp cmd '%s' error: %s", cmd, exc)
        finally:
            _session._unlink_command_output_files(stdout_path, stderr_path)
        return {"success": False, "error": f"Chrome fallback '{cmd}' failed"}

    try:
        # 3. Navigate Chrome to the same URL, then 4. run the requested command.
        nav = _run_tmp("open", [current_url])
        if not nav.get("success"):
            _bt.logger.warning("Chrome fallback: navigate failed: %s", nav.get("error"))
            return {"success": False, "error": f"Chrome fallback navigate failed: {nav.get('error')}"}
        return _run_tmp(command, args)
    finally:
        # 5. Tear down the temporary Chrome session and its socket directory.
        try:
            _run_tmp("close", [])
        except Exception:
            pass
        shutil.rmtree(task_socket_dir, ignore_errors=True)


def _chrome_fallback_screenshot(task_id: str, args: List[str], timeout: int) -> Dict[str, Any]:
    """Take a screenshot using a temporary Chrome session."""
    return _run_chrome_fallback_command(task_id, "screenshot", args, timeout)
