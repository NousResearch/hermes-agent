#!/usr/bin/env python3
"""Read-only Chrome DevTools Protocol (CDP) inspection tool ``browser_cdp``."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Optional

from tools.registry import registry, tool_error
from tools.browser_extension_router import routed_browser_handler

logger = logging.getLogger(__name__)

CDP_DOCS_URL = "https://chromedevtools.github.io/devtools-protocol/"

# This is a capability allowlist, not a denylist: each method is browser-level,
# parameterless, read-only, and cannot access page content or create state.
_CDP_READ_ONLY_METHODS = frozenset({"Browser.getVersion", "Target.getTargets"})
CDP_CAPABILITY_ERROR = (
    "Blocked: browser_cdp only permits read-only inspection methods "
    "(Browser.getVersion, Target.getTargets)."
)

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
        out_key = redact_sensitive_text(key, force=True) if isinstance(key, str) else key  # by-value objects can carry a secret as a KEY
        redacted[out_key] = item if isinstance(item, str) and opaque else _redact_cdp_output(
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


def _validate_cdp_capability(method: Any, params: Any, target_id: Optional[str], frame_id: Optional[str]) -> Optional[str]:
    """Return the stable refusal unless this request is one of the safe capabilities."""
    if method not in _CDP_READ_ONLY_METHODS or params not in (None, {}) or target_id or frame_id:
        return CDP_CAPABILITY_ERROR
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
    """Run one safe browser-level CDP inspection command and return JSON."""
    if not method or not isinstance(method, str):
        return tool_error("'method' is required (e.g. 'Target.getTargets')", cdp_docs=CDP_DOCS_URL)
    if capability_error := _validate_cdp_capability(method, params, target_id, frame_id):
        return tool_error(capability_error)
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
    call_params: Dict[str, Any] = {}

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
        "Read-only Chrome DevTools Protocol (CDP) browser inspection. Only Browser.getVersion and "
        "Target.getTargets are supported; arbitrary CDP commands, page targeting, navigation, network, DOM "
        "changes, scripting, and JavaScript evaluation are blocked. Use the dedicated browser tools for "
        "navigation, snapshots, vision, and actions.\n\n"
        "**Requires an explicit CDP override.** Available when the user has run '/browser connect' or set "
        "'browser.cdp_url' in config.yaml. The endpoint may be local or cloud-hosted. A cloud provider's "
        "managed per-session CDP URL is not automatically surfaced to this tool. Camofox is REST-only. "
        "If the tool is in your toolset at all, an explicit CDP endpoint is configured.\n\n"
        "**Available patterns:**\n"
        "- List tabs: method='Target.getTargets', params={}\n"
        "- Inspect browser version: method='Browser.getVersion', params={}"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "method": {"type": "string", "enum": sorted(_CDP_READ_ONLY_METHODS), "description": "Browser.getVersion or Target.getTargets."},
            "params": {"type": "object", "properties": {}, "additionalProperties": False, "description": "Omit or pass {}."},
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


def _browser_cdp_handler(args: Dict[str, Any], **kw: Any) -> str:
    """Apply the raw-CDP capability floor before any controller can route it."""
    method = args.get("method", "")
    params = args.get("params")
    target_id = args.get("target_id")
    frame_id = args.get("frame_id")
    if capability_error := _validate_cdp_capability(method, params, target_id, frame_id):
        return tool_error(capability_error)
    return routed_browser_handler(
        "browser_cdp", args,
        fallback=lambda: browser_cdp(
            method=method, params=params, target_id=target_id, frame_id=frame_id,
            timeout=args.get("timeout", 30.0), task_id=kw.get("task_id"),
        ),
        task_id=kw.get("task_id"), session_id=kw.get("session_id"),
    )


registry.register(
    name="browser_cdp",
    toolset="browser-cdp",
    schema=BROWSER_CDP_SCHEMA,
    handler=_browser_cdp_handler,
    check_fn=_browser_cdp_check,
    emoji="🧪",
)
