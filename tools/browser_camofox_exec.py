"""Camofox exec backend for ``browser_exec`` (script-per-call mode).

When ``browser.backend`` is ``"camofox"``, the ``browser_exec`` tool runs
the model's Python through the Camofox REST API instead of the Browser Use
CLI 3.0 harness. The surface is the same — one script per call, pre-imported
helpers — but every helper maps 1:1 onto the Camofox server
(``tools/browser_camofox_runtime.py`` runs in a fresh subprocess and speaks
plain HTTP to the Camofox server, keeping the anti-detection Camoufox engine,
profile persistence and VNC live view that the CDP-only browser-use harness
cannot drive).

Key differences vs the CLI backend:

* No browser-use CLI install needed — the subprocess is the Hermes Python
  interpreter (``sys.executable``) running a stdlib-only runtime.
* Sessions are resolved in-process via ``tools.browser_camofox._get_session``
  so managed persistence, identity overrides and tab adoption behave exactly
  like the built-in Camofox tools.
* Tabs survive across calls on the server; the runtime prints
  ``CAMOFOX_TAB_ID=<id>`` lines that we parse back into the in-memory
  session cache so the next call starts from the live tab.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Imported lazily to avoid a circular import (browser_use_cli imports this
# module's browser_exec only at call time).
from tools.browser_use_cli import (  # noqa: E402
    _DEFAULT_TIMEOUT_S,
    _MAX_TIMEOUT_S,
    _MIN_TIMEOUT_S,
    _base_subprocess_env,
    _find_screenshot,
    _native_screenshot_result,
    _workspace_dir,
)
from tools.registry import tool_error, tool_result  # noqa: E402

_STDERR_CAP_CHARS = 4000
_TAB_ID_RE = None  # compiled lazily below
_SCREENSHOT_RE = None  # compiled lazily below


def _tab_id_re():
    global _TAB_ID_RE
    if _TAB_ID_RE is None:
        import re

        _TAB_ID_RE = re.compile(r"^CAMOFOX_TAB_ID=(\S+)$", re.MULTILINE)
    return _TAB_ID_RE


def _screenshot_re():
    global _SCREENSHOT_RE
    if _SCREENSHOT_RE is None:
        import re

        _SCREENSHOT_RE = re.compile(r"^CAMOFOX_SCREENSHOT=(.+?)\s*$", re.MULTILINE)
    return _SCREENSHOT_RE


def _marked_screenshot(stdout: str, since: float) -> Optional[str]:
    """Last ``CAMOFOX_SCREENSHOT=`` path written during this exec, or None.

    The runtime emits this marker with the *native* path so Windows drive
    letters survive: ``_find_screenshot``'s regex requires a leading "/",
    so a bare ``C:/…/x.png`` print only matches from ``/…`` onward — which
    resolves against the process's current drive and silently misses the
    file whenever the workspace lives on another drive.
    """
    for path in reversed(_screenshot_re().findall(stdout or "")):
        try:
            if os.path.isfile(path) and os.path.getmtime(path) >= since - 1:
                return path
        except OSError:
            continue
    return None


def _runtime_wrapper() -> str:
    """Python -c wrapper that imports the runtime helpers and execs stdin."""
    runtime_dir = Path(__file__).resolve().parent.as_posix()
    return (
        "import sys\n"
        f"sys.path.insert(0, {runtime_dir!r})\n"
        "from browser_camofox_runtime import *\n"
        "exec(compile(sys.stdin.read(), '<browser-exec>', 'exec'))\n"
    )


def _resolve_session_env(task_id: Optional[str]) -> Optional[Dict[str, str]]:
    """Resolve the Camofox session in-process and build the subprocess env."""
    from agent.secret_scope import get_secret
    from tools.browser_camofox import _get_session, get_camofox_url

    base_url = get_camofox_url()
    if not base_url:
        return None
    session = _get_session(task_id)
    return {
        "CAMOFOX_URL": base_url,
        "CAMOFOX_API_KEY": (get_secret("CAMOFOX_API_KEY", "") or ""),
        "BH_USER_ID": str(session.get("user_id") or ""),
        "BH_SESSION_KEY": str(session.get("session_key") or ""),
        "BH_TAB_ID": str(session.get("tab_id") or ""),
    }


def _update_session_tab(task_id: Optional[str], tab_id: str) -> None:
    """Write the tab id the runtime resolved back into the session cache."""
    try:
        from tools.browser_camofox import _get_session

        session = _get_session(task_id)
        if session:
            session["tab_id"] = tab_id
    except Exception as exc:  # pragma: no cover — best-effort bookkeeping
        logger.debug("Could not persist Camofox tab id %s: %s", tab_id, exc)


def browser_exec(
    code: str,
    session: str = "",
    timeout_s: int = _DEFAULT_TIMEOUT_S,
    task_id: Optional[str] = None,
):
    """Run Python code through the Camofox exec runtime, return its output.

    Mirrors ``tools.browser_use_cli.browser_exec`` — same result shape, same
    screenshot detection, same workspace semantics. The ``session`` argument
    is accepted for schema parity; Camofox sessions are keyed by the resolved
    task identity, not by a daemon name.
    """
    if not code or not code.strip():
        return tool_error(
            "No code provided. Pass Python that uses the pre-imported helpers, "
            "e.g. new_tab(\"https://example.com\") then print(page_info())."
        )

    from tools.browser_use_cli import _blocked_url_in_code

    blocked = _blocked_url_in_code(code)
    if blocked:
        return tool_error(blocked)

    env = _resolve_session_env(task_id)
    if env is None:
        return tool_error(
            "Camofox exec mode requires a configured Camofox server. Set "
            "CAMOFOX_URL (e.g. http://localhost:9377) in ~/.hermes/.env and "
            "start the server, or switch browser backends via `hermes tools`."
        )

    workspace = _workspace_dir(task_id)
    if workspace:
        env["BH_AGENT_WORKSPACE"] = workspace

    try:
        timeout = max(_MIN_TIMEOUT_S, min(int(timeout_s), _MAX_TIMEOUT_S))
    except (TypeError, ValueError):
        timeout = _DEFAULT_TIMEOUT_S

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

    # Credential-scrubbed env, exactly like the CLI backend: the model's code
    # runs inside this subprocess, so inheriting os.environ wholesale would
    # hand it every provider key / gateway token in the parent process.
    proc_env = _base_subprocess_env()
    proc_env.update(env)

    started = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _runtime_wrapper()],
            input=code,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=proc_env,
            **popen_extra,
        )
    except subprocess.TimeoutExpired:
        return tool_error(
            f"Camofox exec timed out after {timeout}s. The browser may still "
            "be working; retry with a larger timeout_s (max "
            f"{_MAX_TIMEOUT_S}), or split the work into several calls that "
            "append to workspace files — anything already written to the "
            "workspace is preserved."
        )
    except OSError as e:
        return tool_error(f"Failed to launch Camofox exec runtime: {e}")

    result: Dict[str, Any] = {
        "success": proc.returncode == 0,
        "exit_code": proc.returncode,
        "output": proc.stdout,
    }
    if workspace:
        result["workspace"] = workspace

    stderr = (proc.stderr or "").strip()
    if stderr:
        if len(stderr) > _STDERR_CAP_CHARS:
            stderr = stderr[:_STDERR_CAP_CHARS] + "\n… (stderr truncated)"
        result["stderr"] = stderr

    # Persist the tab id the runtime resolved (fresh tab on 404, a second
    # new_tab() in the same script, …). The LAST marker is the live tab: the
    # runtime prints one every time it resolves a tab, so taking the first
    # would pin the cache to a tab the script has already navigated away from.
    tab_ids = _tab_id_re().findall(proc.stdout or "")
    if tab_ids:
        _update_session_tab(task_id, tab_ids[-1])

    screenshot = _marked_screenshot(proc.stdout or "", started) or _find_screenshot(
        proc.stdout or "", started
    )
    if screenshot:
        result["screenshot_path"] = screenshot
        native = _native_screenshot_result(result, screenshot)
        if native is not None:
            return native
    return tool_result(result)


def is_camofox_exec_mode() -> bool:
    """True when ``browser.backend: camofox`` is configured."""
    from tools.browser_use_cli import get_browser_backend

    return get_browser_backend() == "camofox"


# ---------------------------------------------------------------------------
# Helpers reused by browser_use_cli for schema/description building
# ---------------------------------------------------------------------------

CAMOFOX_HELPERS_DIGEST = (
    "\n\nHELPERS (pre-imported, Camofox backend): new_tab(url) opens a URL in "
    "a fresh tab (use for the FIRST navigation), goto_url(url) navigates the "
    "current tab, wait_for_load() after navigation, page_info() returns the "
    "page URL plus the accessibility snapshot with element refs "
    "(e1, e2, …), js(expr) evaluates JavaScript and returns its value "
    "(js('document.title'); wrap function bodies as js('(() => {...})()') — "
    "a bare '() => {...}' returns the function itself, uncalled), "
    "fill_input(selector, text) types into inputs by CSS selector, "
    "type_into_ref(ref, text) types into a snapshot ref, click_ref(ref) "
    "clicks a snapshot ref (PREFERRED over coordinates — refs are stable), "
    "click_selector(selector) clicks by CSS selector, click_at_xy(x, y) "
    "clicks viewport coordinates (fallback), press_key(key) sends a "
    "keyboard key, scroll_page(direction, amount) scrolls, "
    "capture_screenshot() saves a PNG and prints its path (the image is "
    "attached to the tool result for vision models), get_links(limit) lists "
    "page links, cdp(method, **kwargs) is a limited compatibility shim "
    "(Page.navigate, Runtime.evaluate, Accessibility.getFullAXTree, "
    "DOM.getBoxModel — prefer the typed helpers). ensure_real_tab() "
    "recovers from a stale tab. Login walls: stop and ask the user; never "
    "guess credentials."
)

CAMOFOX_DESCRIPTION_HEADER = (
    "Drive a real web browser (Camoufox anti-detection engine via the "
    "Camofox server) using pre-imported Python helpers. The `code` argument "
    "is piped verbatim to a fresh Python runtime and executed with the "
    "helpers pre-imported; stdout comes back in the result. Start `code` "
    "with a one-line comment describing the step for the user in plain, "
    "non-technical language, max 60 chars (e.g. `# Searching Amazon for "
    "paper towels`) — the UI displays it as the step label.\n\n"
    "STATE: the browser tab and the workspace persist across calls; Python "
    "variables do NOT (each call is a fresh interpreter). The workspace is a "
    "stable directory — path in $BH_AGENT_WORKSPACE and returned as "
    "`workspace` in every result. For multi-item tasks ('collect all N "
    "products / every entry / the full table'), append each batch to a "
    "JSON/CSV file in the workspace as you go, then read it back to assemble "
    "the final answer; define reusable functions in agent_helpers.py there — "
    "the harness auto-imports it into every call. Do aggregation in code, "
    "not in your head: dedupe, count, sort, and format with Python inside "
    "the exec. Before giving a final answer on a multi-item task, verify the "
    "collected count against what was asked and go back for anything "
    "missing.\n\n"
    "Batch each sub-procedure (navigate, wait, extract, act) into one call "
    "— do not spend a call per action — but for long extractions prefer "
    "several medium calls that append to workspace files over one giant "
    "call, so progress survives timeouts. Tabs opened by this backend share "
    "the Camofox profile (cookies persist per configured identity). The "
    "`session` argument is accepted for schema parity and ignored in "
    "Camofox mode — tabs are keyed to the task identity."
)
