#!/usr/bin/env python3
"""Raw Chrome DevTools Protocol (CDP) passthrough tool ``browser_cdp``.

Sends arbitrary CDP commands to the browser's DevTools WebSocket when a CDP URL is
configured (``/browser connect`` → ``BROWSER_CDP_URL``, ``browser.cdp_url``, or a
CDP-backed cloud session). Escape hatch for operations the main browser tools don't
cover. Method reference: https://chromedevtools.github.io/devtools-protocol/
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Optional

from tools.registry import registry, tool_error
from tools.browser_extension_router import routed_browser_handler

logger = logging.getLogger(__name__)

CDP_DOCS_URL = "https://chromedevtools.github.io/devtools-protocol/"

# Browser/target inspection that never reads page body/cookies/DOM/storage — stays
# usable so the model can list tabs or navigate away from a blocked page.
_CDP_PRIVATE_PAGE_ALLOWED_METHODS = {
    "Browser.getVersion", "Target.getTargets", "Target.attachToTarget", "Target.detachFromTarget",
    "Page.navigate", "Page.reload", "Page.stopLoading",
}

# method → result paths that are ALWAYS opaque base64 (protocol-declared binary).
# redact_sensitive_text's Fernet pattern ("gAAAA" + base64 alphabet) can match arbitrary
# spans inside such payloads and corrupt the decoded bytes; the payload is not free text
# the model reads, so redaction protects no secret there.
_CDP_ALWAYS_BINARY_PATHS: Dict[str, tuple] = {
    "Page.captureScreenshot": (("data",),), "Page.printToPDF": (("data",),),
    "Network.streamResourceContent": (("bufferedData",),), "HeadlessExperimental.beginFrame": (("screenshotData",),),
    "CacheStorage.requestCachedResponse": (("response", "body"),),
}

# method → result paths that are opaque base64 ONLY when the carrying dict has a
# ``base64Encoded`` sibling that is exactly ``True``; otherwise text → redacted.
_CDP_FLAGGED_BINARY_PATHS: Dict[str, tuple] = {
    "Network.getResponseBody": (("body",),), "Fetch.getResponseBody": (("body",),),
    "IO.read": (("data",),), "Network.getRequestPostData": (("postData",),),
}


def _redact_cdp_output(value: Any, *, always_paths: tuple = (), flagged_paths: tuple = ()) -> Any:
    """Redact browser-originated CDP result text; opaque bytes stay byte-identical.

    Exemptions come ONLY from the calling method's spec as exact result paths. Path
    suffixes propagate only into the matching subtree, so ``base64Encoded`` is honored
    solely as a sibling on the trusted carrier object — never as ambient trust a
    ``Runtime.evaluate`` by-value object could spoof.

    See #94138, #94142.
    """
    from agent.redact import redact_sensitive_text
    if isinstance(value, str):
        return redact_sensitive_text(value, force=True)
    if isinstance(value, (list, tuple)):
        return type(value)(_redact_cdp_output(item) for item in value)
    if not isinstance(value, dict):
        return value
    base64_flagged = value.get("base64Encoded") is True
    def leaf(paths: tuple, key: str) -> bool:
        return any(len(p) == 1 and p[0] == key for p in paths)
    def descend(paths: tuple, key: str) -> tuple:
        return tuple(p[1:] for p in paths if len(p) > 1 and p[0] == key)
    redacted: Dict[str, Any] = {}
    for key, item in value.items():
        opaque = leaf(always_paths, key) or (leaf(flagged_paths, key) and base64_flagged)
        redacted[key] = item if isinstance(item, str) and opaque else _redact_cdp_output(
            item, always_paths=descend(always_paths, key), flagged_paths=descend(flagged_paths, key))
    return redacted


# ``websockets`` is a direct dependency; wrap so a stale env yields a clean error.
try:
    import websockets
    from websockets.exceptions import WebSocketException

    _WS_AVAILABLE = True
except ImportError:
    websockets = None  # type: ignore[assignment]
    WebSocketException = Exception  # type: ignore[assignment,misc]
    _WS_AVAILABLE = False


def _run_async(coro):
    """Run an async coroutine from a sync handler, safe inside or outside a loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


def _resolve_cdp_endpoint() -> str:
    """Normalized CDP WebSocket URL via ``browser_tool_cdp._get_cdp_override``, or ""."""
    try:
        from tools.browser_tool_cdp import _get_cdp_override  # type: ignore[import-not-found]
        return (_get_cdp_override() or "").strip()
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("browser_cdp: failed to resolve CDP endpoint: %s", exc)
        return ""


def _blocked(message: str, method: str) -> str:
    return tool_error(message, method=method, cdp_docs=CDP_DOCS_URL)


def _expression_private_target(expression: str) -> Optional[str]:
    from tools.browser_tool_eval_policy import _expression_targets_private_url
    return _expression_targets_private_url(expression)


def _navigate_private_target(bt: Any, params: Dict[str, Any]) -> Optional[str]:
    """Blocked URL literal for ``Page.navigate`` params, else ``None``."""
    from tools.browser_tool_eval_policy import _url_blocked
    target_url = str(params.get("url") or "").strip()
    return target_url if target_url and _url_blocked(bt, target_url) else None


# method → (probe(bt, params) -> blocked literal | None, error template)
_METHOD_PARAM_GUARDS = {
    "Page.navigate": (_navigate_private_target,
                      "Blocked: CDP Page.navigate target is a private or internal address ({})."),
    "Runtime.evaluate": (lambda bt, params: _expression_private_target(str(params.get("expression") or "")),
                         "Blocked: CDP Runtime.evaluate expression targets a private or internal address ({})."),
}


def _browser_cdp_private_guard(*, task_id: str, method: str, params: Dict[str, Any]) -> Optional[str]:
    """Apply the browser SSRF/private-page guard to raw CDP calls.

    Raw CDP shares the cloud/private-network boundary of ``browser_snapshot`` /
    ``browser_console`` / ``browser_eval`` and must not become their bypass. Probes are
    best-effort; a probe failure never breaks local/custom CDP workflows.
    """
    try:
        from tools import browser_tool as bt  # type: ignore[import-not-found]
        from tools import browser_tool_eval_policy as policy
        if not policy._eval_ssrf_guard_active(task_id):
            return None
        guard = _METHOD_PARAM_GUARDS.get(method)
        if guard is not None:
            probe, template = guard
            literal = probe(bt, params or {})
            if literal:
                return _blocked(template.format(literal), method)
        if method not in _CDP_PRIVATE_PAGE_ALLOWED_METHODS:
            blocked_url = policy._current_page_private_url(task_id)
            if blocked_url:
                return _blocked(f"Blocked: page URL targets a private or internal address ({blocked_url}). "
                                f"Raw CDP method {method!r} could expose private page content or state.", method)
    except Exception as exc:  # noqa: BLE001
        logger.debug("browser_cdp: private-page guard probe failed: %s", exc)
    return None


async def _cdp_call(ws_url: str, method: str, params: Dict[str, Any], target_id: Optional[str],
                    timeout: float) -> Dict[str, Any]:
    """Make a single CDP call. With ``target_id``, ``Target.attachToTarget(flatten=True)`` multiplexes a
    page-level session over the browser-level WebSocket; without it ``method`` runs at browser level."""
    assert websockets is not None  # guarded by _WS_AVAILABLE at call-site
    # max_size=None: CDP responses (e.g. DOM.getDocument) can be large; ping_interval=None: CDP
    # servers don't expect pings.
    async with websockets.connect(ws_url, max_size=None, open_timeout=timeout, close_timeout=5,
                                  ping_interval=None) as ws:
        next_id = 1

        async def _send(req: Dict[str, Any], what: str) -> Dict[str, Any]:
            nonlocal next_id
            call_id, next_id = next_id, next_id + 1
            await ws.send(json.dumps({"id": call_id, **req}))
            deadline = asyncio.get_running_loop().time() + timeout
            while True:  # ignore events / out-of-order responses
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise TimeoutError(f"Timed out {what}")
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=remaining))
                if msg.get("id") == call_id:
                    return msg

        req: Dict[str, Any] = {"method": method, "params": params or {}}
        if target_id:
            msg = await _send({"method": "Target.attachToTarget", "params": {"targetId": target_id, "flatten": True}},
                              f"attaching to target {target_id}")
            if "error" in msg:
                raise RuntimeError(f"Target.attachToTarget failed: {msg['error']}")
            session_id = msg.get("result", {}).get("sessionId")
            if not session_id:
                raise RuntimeError("Target.attachToTarget did not return a sessionId")
            req["sessionId"] = session_id

        msg = await _send(req, f"waiting for response to {method}")
        if "error" in msg:
            raise RuntimeError(f"CDP error: {msg['error']}")
        return msg.get("result", {})


def _browser_cdp_via_supervisor(task_id: str, frame_id: str, method: str, params: Optional[Dict[str, Any]],
                                timeout: float) -> str:
    """Route a CDP call through the live supervisor session for an OOPIF frame."""
    try:
        from tools.browser_supervisor import SUPERVISOR_REGISTRY  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover — defensive
        return tool_error(f"CDP supervisor is not available: {exc}. frame_id routing requires a running "
                          "supervisor attached via /browser connect or an active Browserbase session.")

    supervisor = SUPERVISOR_REGISTRY.get(task_id)
    if supervisor is None:
        return tool_error(f"No CDP supervisor is attached for task={task_id!r}. Call browser_navigate or "
                          "/browser connect first so the supervisor can attach. Once attached, browser_snapshot "
                          "will populate frame_tree with frame_ids you can pass here.")

    tree = supervisor.snapshot().frame_tree
    frame_info: Optional[Dict[str, Any]] = next(
        (f for f in [tree.get("top"), *(tree.get("children") or [])] if f and f.get("frame_id") == frame_id), None)
    if frame_info is None:  # frame_tree is capped at 30 entries — check the raw frames dict too.
        with supervisor._state_lock:  # type: ignore[attr-defined]
            raw = supervisor._frames.get(frame_id)  # type: ignore[attr-defined]
        frame_info = raw.to_dict() if raw is not None else None
    if frame_info is None:
        return tool_error(f"frame_id {frame_id!r} not found in supervisor state. "
                          "Call browser_snapshot to see current frame_tree.")

    child_sid = frame_info.get("session_id")
    if not child_sid:  # same-origin iframes have no dedicated session; reach them via contentWindow/contentDocument
        return tool_error(f"frame_id {frame_id!r} is not an out-of-process iframe (no dedicated CDP session). "
                          "For same-origin iframes, use `browser_cdp(method='Runtime.evaluate', params={'expression': "
                          "\"document.querySelector('iframe').contentDocument.title\"})` at the top-level page instead.")

    loop = supervisor._loop  # type: ignore[attr-defined]
    if loop is None or not loop.is_running():
        return tool_error("CDP supervisor loop is not running. Try reconnecting with /browser connect.")

    try:
        from agent.async_utils import safe_schedule_threadsafe
        fut = safe_schedule_threadsafe(
            supervisor._cdp(method, params or {}, session_id=child_sid, timeout=timeout), loop)  # type: ignore[attr-defined]
        if fut is None:
            return tool_error("CDP call via supervisor failed: loop unavailable", cdp_docs=CDP_DOCS_URL)
        result_msg = fut.result(timeout=timeout + 2)
    except Exception as exc:
        return tool_error(f"CDP call via supervisor failed: {type(exc).__name__}: {exc}", cdp_docs=CDP_DOCS_URL)

    return json.dumps({"success": True, "method": method, "frame_id": frame_id, "session_id": child_sid,
                       "result": result_msg.get("result", {})}, ensure_ascii=False)


def browser_cdp(method: str, params: Optional[Dict[str, Any]] = None, target_id: Optional[str] = None,
                frame_id: Optional[str] = None, timeout: float = 30.0, task_id: Optional[str] = None) -> str:
    """Send a raw CDP command (see ``CDP_DOCS_URL``). ``target_id`` attaches a fresh stateless connection
    to a tab; ``frame_id`` (OOPIF from ``browser_snapshot.frame_tree``) routes through the supervisor's live
    WebSocket instead — the only reliable way to evaluate inside an iframe where fresh per-call connections
    hit signed-URL expiry (Browserbase). Both paths share the same private-page/SSRF guard. Returns JSON
    ``{"success": True, "method", "result"}`` or ``{"error": ...}``."""
    effective_task_id = task_id or "default"

    if frame_id:
        blocked = _browser_cdp_private_guard(task_id=effective_task_id, method=method, params=params or {})
        if blocked:
            return blocked
        return _browser_cdp_via_supervisor(task_id=effective_task_id, frame_id=frame_id, method=method,
                                           params=params, timeout=timeout)

    if not method or not isinstance(method, str):
        return tool_error("'method' is required (e.g. 'Target.getTargets')", cdp_docs=CDP_DOCS_URL)
    if not _WS_AVAILABLE:
        return tool_error("The 'websockets' Python package is required but not installed. "
                          "Install it with: pip install websockets")
    endpoint = _resolve_cdp_endpoint()
    if not endpoint:
        return tool_error("No CDP endpoint is available. Run '/browser connect' to attach to a running Chrome, "
                          "Brave, Chromium, or Edge browser, or set 'browser.cdp_url' in config.yaml. The Camofox "
                          "backend is REST-only and does not expose CDP.", cdp_docs=CDP_DOCS_URL)
    if not endpoint.startswith(("ws://", "wss://")):
        return tool_error(f"CDP endpoint is not a WebSocket URL: {endpoint!r}. Expected ws://... or wss://... — "
                          "the /browser connect resolver should have rewritten this. Check that a Chromium-family "
                          "browser is actually listening on the debug port.")
    call_params: Dict[str, Any] = params or {}
    if not isinstance(call_params, dict):
        return tool_error(f"'params' must be an object/dict, got {type(call_params).__name__}")

    blocked = _browser_cdp_private_guard(task_id=effective_task_id, method=method, params=call_params)
    if blocked:
        return blocked

    try:
        safe_timeout = float(timeout) if timeout else 30.0
    except (TypeError, ValueError):
        safe_timeout = 30.0
    safe_timeout = max(1.0, min(safe_timeout, 300.0))
    try:
        result = _run_async(_cdp_call(endpoint, method, call_params, target_id, safe_timeout))
    except asyncio.TimeoutError as exc:
        return tool_error(f"CDP call timed out after {safe_timeout}s: {exc}", method=method)
    except (TimeoutError, RuntimeError) as exc:
        return tool_error(str(exc), method=method)
    except WebSocketException as exc:
        return tool_error(f"WebSocket error talking to CDP at {endpoint}: {exc}. The browser may have "
                          "disconnected — try '/browser connect' again.", method=method)
    except Exception as exc:  # pragma: no cover — unexpected
        logger.exception("browser_cdp unexpected error")
        return tool_error(f"Unexpected error: {type(exc).__name__}: {exc}", method=method)

    payload: Dict[str, Any] = {"success": True, "method": method, "result": _redact_cdp_output(
        result, always_paths=_CDP_ALWAYS_BINARY_PATHS.get(method, ()),
        flagged_paths=_CDP_FLAGGED_BINARY_PATHS.get(method, ()))}
    if target_id:
        payload["target_id"] = target_id
    return json.dumps(payload, ensure_ascii=False)


BROWSER_CDP_SCHEMA: Dict[str, Any] = {
    "name": "browser_cdp",
    "description": (
        "Send a raw Chrome DevTools Protocol (CDP) command. Escape hatch for browser operations not covered "
        "by browser_navigate, browser_click, browser_console, etc.\n\n"
        "**Requires a reachable CDP endpoint.** Available when the user has run '/browser connect' to attach "
        "to a running Chrome, Brave, Chromium, or Edge browser, or when 'browser.cdp_url' is set in "
        "config.yaml. Not currently wired up for cloud backends (Browserbase, Browser Use, Firecrawl) — "
        "those expose CDP per session but live-session routing is a follow-up. Camofox is REST-only and "
        "will never support CDP. If the tool is in your toolset at all, a CDP endpoint is already reachable.\n\n"
        f"**CDP method reference:** {CDP_DOCS_URL} — use web_extract on a method's URL "
        "(e.g. '/tot/Page/#method-handleJavaScriptDialog') to look up parameters and return shape.\n\n"
        "**Common patterns:**\n"
        "- List tabs: method='Target.getTargets', params={}\n"
        "- Handle a native JS dialog: method='Page.handleJavaScriptDialog', "
        "params={'accept': true, 'promptText': ''}, target_id=<tabId>\n"
        "- Get all cookies: method='Network.getAllCookies', params={}\n"
        "- Eval in a specific tab: method='Runtime.evaluate', params={'expression': '...', 'returnByValue': true}, "
        "target_id=<tabId>\n"
        "- Set viewport for a tab: method='Emulation.setDeviceMetricsOverride', "
        "params={'width': 1280, 'height': 720, 'deviceScaleFactor': 1, 'mobile': false}, target_id=<tabId>\n\n"
        "**Usage rules:**\n"
        "- Browser-level methods (Target.*, Browser.*, Storage.*): omit target_id and frame_id.\n"
        "- Page-level methods (Page.*, Runtime.*, DOM.*, Emulation.*, Network.* scoped to a tab): pass "
        "target_id from Target.getTargets.\n"
        "- **Cross-origin iframe scope** (Runtime.evaluate inside an OOPIF, Page.* targeting a frame target, "
        "etc.): pass frame_id from the browser_snapshot frame_tree output. This routes through the CDP "
        "supervisor's live connection — the only reliable way on Browserbase where stateless CDP calls hit "
        "signed-URL expiry.\n"
        "- Each stateless call (without frame_id) is independent — sessions and event subscriptions do not "
        "persist between calls. For stateful workflows, prefer the dedicated browser tools or use frame_id "
        "routing."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "method": {"type": "string", "description": (
                "CDP method name, e.g. 'Target.getTargets', 'Runtime.evaluate', 'Page.handleJavaScriptDialog'.")},
            "params": {"type": "object", "properties": {}, "additionalProperties": True, "description": (
                "Method-specific parameters as a JSON object. Omit or pass {} for methods that take no parameters.")},
            "target_id": {"type": "string", "description": (
                "Optional. Target/tab ID from Target.getTargets result (each entry's 'targetId'). Use for "
                "page-level methods at the top-level tab scope. Mutually exclusive with frame_id.")},
            "frame_id": {"type": "string", "description": (
                "Optional. Out-of-process iframe (OOPIF) frame_id from browser_snapshot.frame_tree.children[] "
                "where is_oopif=true. When set, routes the call through the CDP supervisor's live session for "
                "that iframe. Essential for Runtime.evaluate inside cross-origin iframes, especially on "
                "Browserbase where fresh per-call CDP connections can't keep up with signed URL rotation. For "
                "same-origin iframes, use parent contentWindow/contentDocument from Runtime.evaluate at the "
                "top-level page instead.")},
            "timeout": {"type": "number", "default": 30, "description": "Timeout in seconds (default 30, max 300)."},
        },
        "required": ["method"],
    },
}


def _browser_cdp_check() -> bool:
    """Availability check: offered only when a static CDP URL is set (Camofox is REST-only;
    the default local agent-browser hides its CDP port; cloud per-session ``cdp_url`` isn't
    surfaced). Raw (no-I/O) gate: check_fns run at every startup, and resolving the endpoint
    over HTTP here would block launch on a stale endpoint."""
    try:
        from tools.browser_tool_cdp import _get_cdp_override_raw
        from tools.browser_tool_install import check_browser_requirements  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover — defensive
        logger.debug("browser_cdp check: browser_tool import failed: %s", exc)
        return False
    return bool(check_browser_requirements() and _get_cdp_override_raw())


registry.register(
    name="browser_cdp",
    toolset="browser-cdp",
    schema=BROWSER_CDP_SCHEMA,
    handler=lambda args, **kw: routed_browser_handler(
        "browser_cdp", args,
        fallback=lambda: browser_cdp(
            method=args.get("method", ""), params=args.get("params"), target_id=args.get("target_id"),
            frame_id=args.get("frame_id"), timeout=args.get("timeout", 30.0), task_id=kw.get("task_id"),
        ),
        task_id=kw.get("task_id"), session_id=kw.get("session_id"),
    ),
    check_fn=_browser_cdp_check,
    emoji="🧪",
)
