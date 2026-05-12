#!/usr/bin/env python3
"""
Raw Chrome DevTools Protocol (CDP) passthrough tool.

Exposes a single tool, ``browser_cdp``, that sends arbitrary CDP commands to
the browser's DevTools WebSocket endpoint.  Works when a CDP URL is
configured — either via ``/browser connect`` (sets ``BROWSER_CDP_URL``) or
``browser.cdp_url`` in ``config.yaml`` — or when a CDP-backed cloud provider
session is active.

This is the escape hatch for browser operations not covered by the main
browser tool surface (``browser_navigate``, ``browser_click``,
``browser_console``, etc.) — handling native dialogs, iframe-scoped
evaluation, cookie/network control, low-level tab management, etc.

Method reference: https://chromedevtools.github.io/devtools-protocol/
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from tools.mac_cdp_config_builder import DEFAULT_SHARED_ROOT, build_config, write_config
from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

CDP_DOCS_URL = "https://chromedevtools.github.io/devtools-protocol/"

# ``websockets`` is a transitive dependency of hermes-agent (via fal_client
# and firecrawl-py) and is already imported by gateway/platforms/feishu.py.
# Wrap the import so a clean error surfaces if the package is ever absent.
try:
    import websockets
    from websockets.exceptions import WebSocketException

    _WS_AVAILABLE = True
except ImportError:
    websockets = None  # type: ignore[assignment]
    WebSocketException = Exception  # type: ignore[assignment,misc]
    _WS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Async-from-sync bridge (matches the pattern in homeassistant_tool.py)
# ---------------------------------------------------------------------------


def _run_async(coro):
    """Run an async coroutine from a sync handler, safe inside or outside a loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Endpoint resolution
# ---------------------------------------------------------------------------


def _resolve_cdp_endpoint() -> str:
    """Return the normalized CDP WebSocket URL, or empty string if unavailable.

    Delegates to ``tools.browser_tool._get_cdp_override`` so precedence stays
    consistent with the rest of the browser tool surface:

    1. ``BROWSER_CDP_URL`` env var (live override from ``/browser connect``)
    2. ``browser.cdp_url`` in ``config.yaml``
    """
    try:
        from tools.browser_tool import _get_cdp_override  # type: ignore[import-not-found]

        return (_get_cdp_override() or "").strip()
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("browser_cdp: failed to resolve CDP endpoint: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Core CDP call
# ---------------------------------------------------------------------------


async def _cdp_call(
    ws_url: str,
    method: str,
    params: Dict[str, Any],
    target_id: Optional[str],
    timeout: float,
) -> Dict[str, Any]:
    """Make a single CDP call, optionally attaching to a target first.

    When ``target_id`` is provided, we call ``Target.attachToTarget`` with
    ``flatten=True`` to multiplex a page-level session over the same
    browser-level WebSocket, then send ``method`` with that ``sessionId``.
    When ``target_id`` is None, ``method`` is sent at browser level — which
    works for ``Target.*``, ``Browser.*``, ``Storage.*`` and a few other
    globally-scoped domains.
    """
    assert websockets is not None  # guarded by _WS_AVAILABLE at call-site

    async with websockets.connect(
        ws_url,
        max_size=None,  # CDP responses (e.g. DOM.getDocument) can be large
        open_timeout=timeout,
        close_timeout=5,
        ping_interval=None,  # CDP server doesn't expect pings
    ) as ws:
        next_id = 1
        session_id: Optional[str] = None

        # --- Step 1: attach to target if requested ---
        if target_id:
            attach_id = next_id
            next_id += 1
            await ws.send(
                json.dumps(
                    {
                        "id": attach_id,
                        "method": "Target.attachToTarget",
                        "params": {"targetId": target_id, "flatten": True},
                    }
                )
            )
            deadline = asyncio.get_running_loop().time() + timeout
            while True:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Timed out attaching to target {target_id}"
                    )
                raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
                msg = json.loads(raw)
                if msg.get("id") == attach_id:
                    if "error" in msg:
                        raise RuntimeError(
                            f"Target.attachToTarget failed: {msg['error']}"
                        )
                    session_id = msg.get("result", {}).get("sessionId")
                    if not session_id:
                        raise RuntimeError(
                            "Target.attachToTarget did not return a sessionId"
                        )
                    break
                # Ignore events (messages without "id") while waiting

        # --- Step 2: dispatch the real method ---
        call_id = next_id
        next_id += 1
        req: Dict[str, Any] = {
            "id": call_id,
            "method": method,
            "params": params or {},
        }
        if session_id:
            req["sessionId"] = session_id
        await ws.send(json.dumps(req))

        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError(
                    f"Timed out waiting for response to {method}"
                )
            raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
            msg = json.loads(raw)
            if msg.get("id") == call_id:
                if "error" in msg:
                    raise RuntimeError(f"CDP error: {msg['error']}")
                return msg.get("result", {})
            # Ignore events / out-of-order responses


# ---------------------------------------------------------------------------
# Public tool function
# ---------------------------------------------------------------------------


def _browser_cdp_via_supervisor(
    task_id: str,
    frame_id: str,
    method: str,
    params: Optional[Dict[str, Any]],
    timeout: float,
) -> str:
    """Route a CDP call through the live supervisor session for an OOPIF frame.

    Looks up the frame in the supervisor's snapshot, extracts its child
    ``cdp_session_id``, and dispatches ``method`` with that sessionId via
    the supervisor's already-connected WebSocket (using
    ``asyncio.run_coroutine_threadsafe`` onto the supervisor loop).
    """
    try:
        from tools.browser_supervisor import SUPERVISOR_REGISTRY  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover — defensive
        return tool_error(
            f"CDP supervisor is not available: {exc}. frame_id routing requires "
            f"a running supervisor attached via /browser connect or an active "
            f"Browserbase session."
        )

    supervisor = SUPERVISOR_REGISTRY.get(task_id)
    if supervisor is None:
        return tool_error(
            f"No CDP supervisor is attached for task={task_id!r}. Call "
            f"browser_navigate or /browser connect first so the supervisor "
            f"can attach. Once attached, browser_snapshot will populate "
            f"frame_tree with frame_ids you can pass here."
        )

    snap = supervisor.snapshot()
    # Search both the top frame and the children for the requested id.
    top = snap.frame_tree.get("top")
    frame_info: Optional[Dict[str, Any]] = None
    if top and top.get("frame_id") == frame_id:
        frame_info = top
    else:
        for child in snap.frame_tree.get("children", []) or []:
            if child.get("frame_id") == frame_id:
                frame_info = child
                break
    if frame_info is None:
        # Check the raw frames dict too (frame_tree is capped at 30 entries)
        with supervisor._state_lock:  # type: ignore[attr-defined]
            raw = supervisor._frames.get(frame_id)  # type: ignore[attr-defined]
        if raw is not None:
            frame_info = raw.to_dict()

    if frame_info is None:
        return tool_error(
            f"frame_id {frame_id!r} not found in supervisor state. "
            f"Call browser_snapshot to see current frame_tree."
        )

    child_sid = frame_info.get("session_id")
    if not child_sid:
        # Not an OOPIF — fall back to top-level session (evaluating at page
        # scope).  Same-origin iframes don't get their own sessionId; the
        # agent can still use contentWindow/contentDocument from the parent.
        return tool_error(
            f"frame_id {frame_id!r} is not an out-of-process iframe (no "
            f"dedicated CDP session). For same-origin iframes, use "
            f"`browser_cdp(method='Runtime.evaluate', params={{'expression': "
            f"\"document.querySelector('iframe').contentDocument.title\"}})` "
            f"at the top-level page instead."
        )

    # Dispatch onto the supervisor's loop.
    import asyncio as _asyncio
    loop = supervisor._loop  # type: ignore[attr-defined]
    if loop is None or not loop.is_running():
        return tool_error(
            "CDP supervisor loop is not running. Try reconnecting with "
            "/browser connect."
        )

    async def _do_cdp():
        return await supervisor._cdp(  # type: ignore[attr-defined]
            method,
            params or {},
            session_id=child_sid,
            timeout=timeout,
        )

    try:
        fut = _asyncio.run_coroutine_threadsafe(_do_cdp(), loop)
        result_msg = fut.result(timeout=timeout + 2)
    except Exception as exc:
        return tool_error(
            f"CDP call via supervisor failed: {type(exc).__name__}: {exc}",
            cdp_docs=CDP_DOCS_URL,
        )

    payload: Dict[str, Any] = {
        "success": True,
        "method": method,
        "frame_id": frame_id,
        "session_id": child_sid,
        "result": result_msg.get("result", {}),
    }
    return json.dumps(payload, ensure_ascii=False)


def browser_cdp(
    method: str,
    params: Optional[Dict[str, Any]] = None,
    target_id: Optional[str] = None,
    frame_id: Optional[str] = None,
    timeout: float = 30.0,
    task_id: Optional[str] = None,
) -> str:
    """Send a raw CDP command.  See ``CDP_DOCS_URL`` for method documentation.

    Args:
        method: CDP method name, e.g. ``"Target.getTargets"``.
        params: Method-specific parameters; defaults to ``{}``.
        target_id: Optional target/tab ID for page-level methods.  When set,
            we first attach to the target (``flatten=True``) and send
            ``method`` with the resulting ``sessionId``.  Uses a fresh
            stateless CDP connection.
        frame_id: Optional cross-origin (OOPIF) iframe ``frame_id`` from
            ``browser_snapshot.frame_tree.children[]``.  When set (and the
            frame is an OOPIF with a live session tracked by the CDP
            supervisor), routes the call through the supervisor's existing
            WebSocket — which is how you Runtime.evaluate *inside* an
            iframe on backends where per-call fresh CDP connections would
            hit signed-URL expiry (Browserbase) or expensive reattach.
        timeout: Seconds to wait for the call to complete.
        task_id: Task identifier for supervisor lookup.  When ``frame_id``
            is set, this identifies which task's supervisor to use; the
            handler will default to ``"default"`` otherwise.

    Returns:
        JSON string ``{"success": True, "method": ..., "result": {...}}`` on
        success, or ``{"error": "..."}`` on failure.
    """
    # --- Route iframe-scoped calls through the supervisor ---------------
    if frame_id:
        return _browser_cdp_via_supervisor(
            task_id=task_id or "default",
            frame_id=frame_id,
            method=method,
            params=params,
            timeout=timeout,
        )
    del task_id  # stateless path below

    if not method or not isinstance(method, str):
        return tool_error(
            "'method' is required (e.g. 'Target.getTargets')",
            cdp_docs=CDP_DOCS_URL,
        )

    if not _WS_AVAILABLE:
        return tool_error(
            "The 'websockets' Python package is required but not installed. "
            "Install it with: pip install websockets"
        )

    endpoint = _resolve_cdp_endpoint()
    if not endpoint:
        return tool_error(
            "No CDP endpoint is available. Run '/browser connect' to attach "
            "to a running Chrome, or set 'browser.cdp_url' in config.yaml. "
            "The Camofox backend is REST-only and does not expose CDP.",
            cdp_docs=CDP_DOCS_URL,
        )

    if not endpoint.startswith(("ws://", "wss://")):
        return tool_error(
            f"CDP endpoint is not a WebSocket URL: {endpoint!r}. "
            "Expected ws://... or wss://... — the /browser connect "
            "resolver should have rewritten this. Check that Chrome is "
            "actually listening on the debug port."
        )

    call_params: Dict[str, Any] = params or {}
    if not isinstance(call_params, dict):
        return tool_error(
            f"'params' must be an object/dict, got {type(call_params).__name__}"
        )

    try:
        safe_timeout = float(timeout) if timeout else 30.0
    except (TypeError, ValueError):
        safe_timeout = 30.0
    safe_timeout = max(1.0, min(safe_timeout, 300.0))

    try:
        result = _run_async(
            _cdp_call(endpoint, method, call_params, target_id, safe_timeout)
        )
    except asyncio.TimeoutError as exc:
        return tool_error(
            f"CDP call timed out after {safe_timeout}s: {exc}",
            method=method,
        )
    except TimeoutError as exc:
        return tool_error(str(exc), method=method)
    except RuntimeError as exc:
        return tool_error(str(exc), method=method)
    except WebSocketException as exc:
        return tool_error(
            f"WebSocket error talking to CDP at {endpoint}: {exc}. The "
            "browser may have disconnected — try '/browser connect' again.",
            method=method,
        )
    except Exception as exc:  # pragma: no cover — unexpected
        logger.exception("browser_cdp unexpected error")
        return tool_error(
            f"Unexpected error: {type(exc).__name__}: {exc}",
            method=method,
        )

    payload: Dict[str, Any] = {
        "success": True,
        "method": method,
        "result": result,
    }
    if target_id:
        payload["target_id"] = target_id
    return json.dumps(payload, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Guarded Mac local-worker sidecar config entry points
# ---------------------------------------------------------------------------


MAC_CDP_INVENTORY_RUNNER = str(DEFAULT_SHARED_ROOT / "mac-cdp-readonly-dom-inventory.py")
MAC_CDP_FILL_RUNNER = str(DEFAULT_SHARED_ROOT / "mac-cdp-form-fill-sidecar.py")


def _mac_sidecar_config_payload(
    spec: Dict[str, Any],
    *,
    shared_root: str = str(DEFAULT_SHARED_ROOT),
    config_path: Optional[str] = None,
    runner_path: str,
    write_local_config: bool = False,
) -> str:
    """Build a guarded Mac CDP sidecar config.

    Browser tools run on the current Hermes host, not on the Mac.  Therefore
    the safe default is to return config JSON plus the intended Mac path; the
    caller installs it with mac_write_file before running mac_run_shared_python.
    Tests and local harnesses can opt into writing with write_local_config=True.
    """
    try:
        cfg = build_config(spec, shared_root=shared_root)
        mode = "inventory" if cfg.get("readOnly") is True else "fill"
        if config_path:
            intended_path = config_path
        else:
            name = "cdp-readonly-inventory-config.json" if mode == "inventory" else "cdp-form-fill-config.json"
            intended_path = str(DEFAULT_SHARED_ROOT / name) if shared_root == str(DEFAULT_SHARED_ROOT) else str(Path(shared_root) / name)
        if write_local_config:
            path = write_config(spec, shared_root=shared_root, out_path=intended_path)
            intended_path = str(path)
    except Exception as exc:
        return tool_error(f"Mac CDP sidecar config rejected: {exc}")

    payload: Dict[str, Any] = {
        "success": True,
        "mode": mode,
        "config_path": intended_path,
        "config_json": json.dumps(cfg, ensure_ascii=False, indent=2) + "\n",
        "runner_path": runner_path,
        "result_path": cfg.get("outputPath"),
        "screenshot_path": cfg.get("screenshotPath"),
        "allowSubmit": cfg.get("allowSubmit"),
        "sideEffectPolicy": cfg.get("sideEffectPolicy"),
        "url": cfg.get("url"),
        "next_step": (
            "Before running, state sideEffectPolicy to the user and obtain "
            "explicit approval for this URL and field set; then install "
            "config_json to config_path via mac_write_file and run runner_path "
            "via mac_run_shared_python."
            if mode == "fill"
            else "Install config_json to config_path via mac_write_file, then "
            "run runner_path via mac_run_shared_python for the read-only pass."
        ),
    }
    if cfg.get("readOnly") is True:
        payload["readOnly"] = True
    else:
        payload["fields_count"] = len(cfg.get("fields", []))
    return json.dumps(payload, ensure_ascii=False)


def browser_mac_cdp_inventory_config(
    url: str,
    allowed_domains: list[str],
    session_id: Optional[str] = None,
    output_prefix: Optional[str] = None,
    shared_root: str = str(DEFAULT_SHARED_ROOT),
    config_path: Optional[str] = None,
    write_local_config: bool = False,
) -> str:
    """Create a read-only DOM inventory config for the Mac CDP sidecar."""
    spec: Dict[str, Any] = {
        "mode": "inventory",
        "url": url,
        "allowedDomains": allowed_domains,
    }
    if session_id:
        spec["sessionId"] = session_id
    if output_prefix:
        spec["outputPrefix"] = output_prefix
    return _mac_sidecar_config_payload(
        spec,
        shared_root=shared_root,
        config_path=config_path,
        runner_path=MAC_CDP_INVENTORY_RUNNER,
        write_local_config=write_local_config,
    )


def browser_mac_cdp_fill_config(
    url: str,
    allowed_domains: list[str],
    fields: list[dict[str, Any]],
    session_id: str,
    output_prefix: Optional[str] = None,
    validation_expression: Optional[str] = None,
    shared_root: str = str(DEFAULT_SHARED_ROOT),
    config_path: Optional[str] = None,
    write_local_config: bool = False,
) -> str:
    """Create a guarded, non-submitting form-fill config for the Mac sidecar."""
    spec: Dict[str, Any] = {
        "mode": "fill",
        "url": url,
        "allowedDomains": allowed_domains,
        "fields": fields,
        "sessionId": session_id,
    }
    if output_prefix:
        spec["outputPrefix"] = output_prefix
    if validation_expression:
        spec["validationExpression"] = validation_expression
    return _mac_sidecar_config_payload(
        spec,
        shared_root=shared_root,
        config_path=config_path,
        runner_path=MAC_CDP_FILL_RUNNER,
        write_local_config=write_local_config,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


BROWSER_CDP_SCHEMA: Dict[str, Any] = {
    "name": "browser_cdp",
    "description": (
        "Send a raw Chrome DevTools Protocol (CDP) command. Escape hatch for "
        "browser operations not covered by browser_navigate, browser_click, "
        "browser_console, etc.\n\n"
        "**Requires a reachable CDP endpoint.** Available when the user has "
        "run '/browser connect' to attach to a running Chrome, or when "
        "'browser.cdp_url' is set in config.yaml. Not currently wired up for "
        "cloud backends (Browserbase, Browser Use, Firecrawl) — those expose "
        "CDP per session but live-session routing is a follow-up. Camofox is "
        "REST-only and will never support CDP. If the tool is in your toolset "
        "at all, a CDP endpoint is already reachable.\n\n"
        f"**CDP method reference:** {CDP_DOCS_URL} — use web_extract on a "
        "method's URL (e.g. '/tot/Page/#method-handleJavaScriptDialog') "
        "to look up parameters and return shape.\n\n"
        "**Common patterns:**\n"
        "- List tabs: method='Target.getTargets', params={}\n"
        "- Handle a native JS dialog: method='Page.handleJavaScriptDialog', "
        "params={'accept': true, 'promptText': ''}, target_id=<tabId>\n"
        "- Get all cookies: method='Network.getAllCookies', params={}\n"
        "- Eval in a specific tab: method='Runtime.evaluate', "
        "params={'expression': '...', 'returnByValue': true}, "
        "target_id=<tabId>\n"
        "- Set viewport for a tab: method='Emulation.setDeviceMetricsOverride', "
        "params={'width': 1280, 'height': 720, 'deviceScaleFactor': 1, "
        "'mobile': false}, target_id=<tabId>\n\n"
        "**Usage rules:**\n"
        "- Browser-level methods (Target.*, Browser.*, Storage.*): omit "
        "target_id and frame_id.\n"
        "- Page-level methods (Page.*, Runtime.*, DOM.*, Emulation.*, "
        "Network.* scoped to a tab): pass target_id from Target.getTargets.\n"
        "- **Cross-origin iframe scope** (Runtime.evaluate inside an OOPIF, "
        "Page.* targeting a frame target, etc.): pass frame_id from the "
        "browser_snapshot frame_tree output. This routes through the CDP "
        "supervisor's live connection — the only reliable way on "
        "Browserbase where stateless CDP calls hit signed-URL expiry.\n"
        "- Each stateless call (without frame_id) is independent — sessions "
        "and event subscriptions do not persist between calls. For stateful "
        "workflows, prefer the dedicated browser tools or use frame_id "
        "routing."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "method": {
                "type": "string",
                "description": (
                    "CDP method name, e.g. 'Target.getTargets', "
                    "'Runtime.evaluate', 'Page.handleJavaScriptDialog'."
                ),
            },
            "params": {
                "type": "object",
                "description": (
                    "Method-specific parameters as a JSON object. Omit or "
                    "pass {} for methods that take no parameters."
                ),
                "properties": {},
                "additionalProperties": True,
            },
            "target_id": {
                "type": "string",
                "description": (
                    "Optional. Target/tab ID from Target.getTargets result "
                    "(each entry's 'targetId'). Use for page-level methods "
                    "at the top-level tab scope. Mutually exclusive with "
                    "frame_id."
                ),
            },
            "frame_id": {
                "type": "string",
                "description": (
                    "Optional. Out-of-process iframe (OOPIF) frame_id from "
                    "browser_snapshot.frame_tree.children[] where "
                    "is_oopif=true. When set, routes the call through the "
                    "CDP supervisor's live session for that iframe. "
                    "Essential for Runtime.evaluate inside cross-origin "
                    "iframes, especially on Browserbase where fresh "
                    "per-call CDP connections can't keep up with signed "
                    "URL rotation. For same-origin iframes, use parent "
                    "contentWindow/contentDocument from Runtime.evaluate "
                    "at the top-level page instead."
                ),
            },
            "timeout": {
                "type": "number",
                "description": (
                    "Timeout in seconds (default 30, max 300)."
                ),
                "default": 30,
            },
        },
        "required": ["method"],
    },
}


def _browser_cdp_check() -> bool:
    """Availability check for browser_cdp.

    The tool is only offered when the Python side can actually reach a CDP
    endpoint right now — meaning a static URL is set via ``/browser connect``
    (``BROWSER_CDP_URL``) or ``browser.cdp_url`` in ``config.yaml``.

    Backends that do *not* currently expose CDP to us — Camofox (REST-only),
    the default local agent-browser mode (Playwright hides its internal CDP
    port), and cloud providers whose per-session ``cdp_url`` is not yet
    surfaced — are gated out so the model doesn't see a tool that would
    reliably fail.  Cloud-provider CDP routing is a follow-up.

    Kept in a thin wrapper so the registration statement stays at module top
    level (the tool-discovery AST scan only picks up top-level
    ``registry.register(...)`` calls).
    """
    try:
        from tools.browser_tool import (  # type: ignore[import-not-found]
            _get_cdp_override,
            check_browser_requirements,
        )
    except ImportError as exc:  # pragma: no cover — defensive
        logger.debug("browser_cdp check: browser_tool import failed: %s", exc)
        return False
    if not check_browser_requirements():
        return False
    return bool(_get_cdp_override())


registry.register(
    name="browser_cdp",
    toolset="browser-cdp",
    schema=BROWSER_CDP_SCHEMA,
    handler=lambda args, **kw: browser_cdp(
        method=args.get("method", ""),
        params=args.get("params"),
        target_id=args.get("target_id"),
        frame_id=args.get("frame_id"),
        timeout=args.get("timeout", 30.0),
        task_id=kw.get("task_id"),
    ),
    check_fn=_browser_cdp_check,
    emoji="🧪",
)


MAC_CDP_INVENTORY_SCHEMA: Dict[str, Any] = {
    "name": "browser_mac_cdp_inventory_config",
    "description": (
        "Build a guarded read-only DOM inventory config for Kagura's Mac "
        "local-worker CDP sidecar. This returns config JSON plus the intended "
        "Mac config path; it does not steal foreground focus and does not "
        "execute browser actions. Install config_json with mac_write_file, then "
        "run runner_path via mac_run_shared_python to perform the read-only pass."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Absolute http/https URL to inspect."},
            "allowed_domains": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Exact hostnames allowed for this operation; subdomains are not implied.",
            },
            "session_id": {"type": "string", "description": "Optional correlation ID appended as sessionId query parameter."},
            "output_prefix": {"type": "string", "description": "Optional safe filename prefix for result/screenshot artifacts."},
            "shared_root": {"type": "string", "description": "Mac shared-worker root. Defaults to Kagura local-worker shared root."},
            "config_path": {"type": "string", "description": "Optional intended output config JSON path under shared_root."},
            "write_local_config": {"type": "boolean", "description": "Advanced/testing only: also write config_path on the current Hermes host."},
        },
        "required": ["url", "allowed_domains"],
    },
}


MAC_CDP_FILL_SCHEMA: Dict[str, Any] = {
    "name": "browser_mac_cdp_fill_config",
    "description": (
        "Build a guarded non-submitting form-fill config for Kagura's Mac "
        "local-worker CDP sidecar. Requires a session_id from the prior "
        "inventory pass, rejects likely secrets, exact-domain mismatches, and "
        "always enforces allowSubmit=false. Returns config JSON plus the intended "
        "Mac config path; install config_json with mac_write_file and execute "
        "the returned runner_path via mac_run_shared_python after review."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Absolute http/https URL to fill."},
            "allowed_domains": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Exact hostnames allowed for this operation; subdomains are not implied.",
            },
            "fields": {
                "type": "array",
                "items": {"type": "object", "additionalProperties": True},
                "description": "Field specs: selector, optional kind=value/textContent/innerText, and value.",
            },
            "session_id": {"type": "string", "description": "Required correlation ID from a prior read-only inventory pass."},
            "output_prefix": {"type": "string", "description": "Optional safe filename prefix for result/screenshot artifacts."},
            "validation_expression": {
                "type": "string",
                "description": "Deprecated/blocked for guarded fill: arbitrary JS validation is rejected because it can cause side effects.",
            },
            "shared_root": {"type": "string", "description": "Mac shared-worker root. Defaults to Kagura local-worker shared root."},
            "config_path": {"type": "string", "description": "Optional intended output config JSON path under shared_root."},
            "write_local_config": {"type": "boolean", "description": "Advanced/testing only: also write config_path on the current Hermes host."},
        },
        "required": ["url", "allowed_domains", "fields", "session_id"],
    },
}


registry.register(
    name="browser_mac_cdp_inventory_config",
    toolset="browser-cdp",
    schema=MAC_CDP_INVENTORY_SCHEMA,
    handler=lambda args, **kw: browser_mac_cdp_inventory_config(
        url=args.get("url", ""),
        allowed_domains=args.get("allowed_domains") or [],
        session_id=args.get("session_id"),
        output_prefix=args.get("output_prefix"),
        shared_root=args.get("shared_root", str(DEFAULT_SHARED_ROOT)),
        config_path=args.get("config_path"),
        write_local_config=bool(args.get("write_local_config", False)),
    ),
    emoji="🛡️",
)

registry.register(
    name="browser_mac_cdp_fill_config",
    toolset="browser-cdp",
    schema=MAC_CDP_FILL_SCHEMA,
    handler=lambda args, **kw: browser_mac_cdp_fill_config(
        url=args.get("url", ""),
        allowed_domains=args.get("allowed_domains") or [],
        fields=args.get("fields") or [],
        session_id=args.get("session_id", ""),
        output_prefix=args.get("output_prefix"),
        validation_expression=args.get("validation_expression"),
        shared_root=args.get("shared_root", str(DEFAULT_SHARED_ROOT)),
        config_path=args.get("config_path"),
        write_local_config=bool(args.get("write_local_config", False)),
    ),
    emoji="🛡️",
)
