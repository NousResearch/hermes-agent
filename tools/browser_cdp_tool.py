#!/usr/bin/env python3
"""Raw Chrome DevTools Protocol (CDP) passthrough tool ``browser_cdp``.

Sends arbitrary CDP commands to the browser's DevTools WebSocket when a CDP URL is
configured (``/browser connect`` → ``BROWSER_CDP_URL``, ``browser.cdp_url``). Escape hatch
for operations ``browser_exec``'s helpers don't cover. Method reference:
https://chromedevtools.github.io/devtools-protocol/
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Dict, Optional

from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

CDP_DOCS_URL = "https://chromedevtools.github.io/devtools-protocol/"

# Browser/target inspection and tab management that never reads page body/cookies/DOM/storage — stays
# usable so the model can list, open, switch and close tabs or navigate away from a blocked page. Methods
# that take a url (Page.navigate, Target.createTarget) are still held to _METHOD_PARAM_GUARDS.
_CDP_PRIVATE_PAGE_ALLOWED_METHODS = {
    "Browser.getVersion", "Target.getTargets", "Target.attachToTarget", "Target.detachFromTarget",
    "Target.createTarget", "Target.activateTarget", "Target.closeTarget",
    "Page.navigate", "Page.reload", "Page.stopLoading",
}

# method → result paths that are ALWAYS opaque base64 (protocol-declared binary).
# redact_sensitive_text's Fernet pattern ("gAAAA" + base64 alphabet) can match arbitrary
# spans inside such payloads and corrupt the decoded bytes; the payload is not free text
# the model reads, so redaction protects no secret there.
_CDP_ALWAYS_BINARY_PATHS: dict[str, tuple] = {
    "Page.captureScreenshot": (("data",),), "Page.printToPDF": (("data",),),
    "Network.streamResourceContent": (("bufferedData",),), "HeadlessExperimental.beginFrame": (("screenshotData",),),
    "CacheStorage.requestCachedResponse": (("response", "body"),),
}

# method → result paths that are opaque base64 ONLY when the carrying dict has a
# ``base64Encoded`` sibling that is exactly ``True``; otherwise text → redacted.
_CDP_FLAGGED_BINARY_PATHS: dict[str, tuple] = {
    "Network.getResponseBody": (("body",),), "Fetch.getResponseBody": (("body",),),
    "IO.read": (("data",),), "Network.getRequestPostData": (("postData",),),
}


def _redact_cdp_output(
    value: Any,
    *,
    always_paths: tuple = (),
    flagged_paths: tuple = (),
    _depth: int = 0,
) -> Any:
    """Redact browser-originated CDP result text; opaque bytes stay byte-identical.

    Exemptions come ONLY from the calling method's spec as exact result paths. Path
    suffixes propagate only into the matching subtree, so ``base64Encoded`` is honored
    solely as a sibling on the trusted carrier object — never as ambient trust a
    ``Runtime.evaluate`` by-value object could spoof.

    ``_depth`` stops a page-controlled tree from raising RecursionError. The
    unread remainder is replaced, not returned raw.

    See #94138, #94142.
    """
    from agent.redact import (
        REDACT_WALK_DEPTH_MARKER,
        REDACT_WALK_MAX_DEPTH,
        redact_sensitive_text,
    )
    if _depth >= REDACT_WALK_MAX_DEPTH:
        return REDACT_WALK_DEPTH_MARKER
    if isinstance(value, str):
        return redact_sensitive_text(value, force=True)
    if isinstance(value, (list, tuple)):
        return type(value)(
            _redact_cdp_output(item, _depth=_depth + 1) for item in value
        )
    if not isinstance(value, dict):
        return value
    base64_flagged = value.get("base64Encoded") is True
    def leaf(paths: tuple, key: str) -> bool:
        return any(len(p) == 1 and p[0] == key for p in paths)
    def descend(paths: tuple, key: str) -> tuple:
        return tuple(p[1:] for p in paths if len(p) > 1 and p[0] == key)
    redacted: dict[str, Any] = {}
    for key, item in value.items():
        opaque = leaf(always_paths, key) or (leaf(flagged_paths, key) and base64_flagged)
        out_key = redact_sensitive_text(key, force=True) if isinstance(key, str) else key  # by-value objects can carry a secret as a KEY
        redacted[out_key] = item if isinstance(item, str) and opaque else _redact_cdp_output(
            item,
            always_paths=descend(always_paths, key),
            flagged_paths=descend(flagged_paths, key),
            _depth=_depth + 1,
        )
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
        import contextvars
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(contextvars.copy_context().run, asyncio.run, coro).result()
    return asyncio.run(coro)


def _resolve_cdp_endpoint() -> str:
    """Normalized CDP WebSocket URL via ``browser_session_cdp.get_cdp_override``, or ""."""
    from tools.browser_session_cdp import get_cdp_override
    return (get_cdp_override() or "").strip()


def _blocked(message: str, method: str) -> str:
    return tool_error(message, method=method, cdp_docs=CDP_DOCS_URL)


def _expression_private_target(params: dict[str, Any], blocked: Callable[[str], bool]) -> Optional[str]:
    from tools.browser_url_policy import expression_targets_private_url
    return expression_targets_private_url(str(params.get("expression") or ""), blocked=blocked)


def _navigate_private_target(params: dict[str, Any], blocked: Callable[[str], bool]) -> Optional[str]:
    """Blocked URL literal for ``Page.navigate`` params, else ``None``."""
    target_url = str(params.get("url") or "").strip()
    return target_url if target_url and blocked(target_url) else None


def _new_tab_private_target(params: dict[str, Any], blocked: Callable[[str], bool]) -> Optional[str]:
    """``Target.createTarget``: a blank tab (``about:blank``, Chrome's default for no url) reaches nothing, so
    it opens before scripts are installed or a page is chosen; any other url is held to ``Page.navigate``'s
    rule (``url_blocked`` refuses every other non-http scheme)."""
    if str(params.get("url") or "").strip().lower() == "about:blank":
        return None
    return _navigate_private_target(params, blocked)


# method → (probe(params, blocked) -> blocked literal | None, error template)
_METHOD_PARAM_GUARDS = {
    "Page.navigate": (_navigate_private_target,
                      "Blocked: CDP Page.navigate target is a private or internal address ({})."),
    "Target.createTarget": (_new_tab_private_target,
                            "Blocked: CDP Target.createTarget url is a private or internal address ({})."),
    "Runtime.evaluate": (_expression_private_target,
                         "Blocked: CDP Runtime.evaluate expression targets a private or internal address ({})."),
}


# _current_page_private_url's verdict when the target list cannot be read: refuse, never assume public.
_PAGE_URL_UNVERIFIED = "<unverified>"
# Browser-internal pages carry no web origin a worker or browser-level call could borrow, so an open new tab
# must not lock every target-less call (url_blocked rejects their schemes as unsupported).
_ORIGINLESS_PAGE_PREFIXES = ("about:", "chrome:", "chrome-untrusted:", "chrome-search:", "devtools:", "edge:", "brave:")


def _current_page_private_url(endpoint: str, target_id: Optional[str]) -> Optional[str]:
    """The private/internal URL a non-allow-listed call would reach (or ``_PAGE_URL_UNVERIFIED``), else None.

    Fail-closed, read from ``Target.getTargets`` on the same endpoint. Only a ``page`` target whose own URL
    is known passes on that URL alone. A browser-level call (no target) or a worker / iframe / unknown
    target runs with some page's origin (a SharedWorker's ``fetch('/')`` reads its private page;
    ``Storage.*`` spans every origin), so it is refused while ANY page target is private."""
    from tools.browser_url_policy import url_blocked
    if not endpoint or not _WS_AVAILABLE:
        return _PAGE_URL_UNVERIFIED
    try:
        infos = _run_async(_cdp_call(endpoint, "Target.getTargets", {}, None, 5.0)).get("targetInfos") or []
    except (TimeoutError, RuntimeError, OSError, ValueError, WebSocketException) as exc:
        logger.debug("browser_cdp: page-URL probe failed: %s", exc)
        return _PAGE_URL_UNVERIFIED
    acted = next((t for t in infos if t.get("targetId") == target_id), None) if target_id else None
    url = str((acted or {}).get("url") or "")
    if url and url_blocked(url):
        return url
    if url and acted.get("type") == "page":
        return None
    page_urls = (str(t.get("url") or "") for t in infos if t.get("type") == "page")
    return next((u for u in page_urls if u and not u.startswith(_ORIGINLESS_PAGE_PREFIXES) and url_blocked(u)), None)


# Input.* only types text or dispatches key/mouse events; a URL in its text is never fetched by the call.
_TEXT_INPUT_METHOD_PREFIX = "Input."
# Nor are these methods' URL params: blocklist / interception match patterns (wildcard hosts, often
# naming the very hosts being blocked), dialog answer text, cookie scope URLs.
_CDP_NON_FETCHING_METHODS = frozenset({
    "Network.setBlockedURLs", "Network.setRequestInterception", "Fetch.enable",
    "Page.handleJavaScriptDialog",
    "Network.setCookie", "Network.setCookies", "Network.deleteCookies", "Storage.setCookies",
})


def _param_url_literals(value: Any, key: str = "") -> list[str]:
    """Every URL a CDP call's params name: each ``url`` value whole, plus the ``http(s)://`` and
    protocol-relative ``//host`` literals in every string at any depth (``Network.loadNetworkResource``,
    ``Runtime.callFunctionOn`` bodies, ``Page.addScriptToEvaluateOnNewDocument`` sources, injected HTML)."""
    from tools.browser_url_policy import url_literals
    if isinstance(value, str):
        return ([value.strip()] if key == "url" and value.strip() else []) + url_literals(value, protocol_relative=True)
    if isinstance(value, dict):
        return [u for k, v in value.items() for u in _param_url_literals(v, str(k))]
    if isinstance(value, (list, tuple)):
        return [u for v in value for u in _param_url_literals(v, key)]
    return []


def _fetched_param_urls(method: str, params: dict[str, Any]) -> list[str]:
    """The distinct URLs a call's params could make the browser fetch (none for text input or the
    non-fetching methods above)."""
    if method.startswith(_TEXT_INPUT_METHOD_PREFIX) or method in _CDP_NON_FETCHING_METHODS:
        return []
    return list(dict.fromkeys(_param_url_literals(params)))


def _browser_cdp_private_guard(*, method: str, params: dict[str, Any], endpoint: str = "",
                               target_id: Optional[str] = None) -> Optional[str]:
    """Apply the browser SSRF/private-page guard to raw CDP calls.

    Raw CDP shares the cloud/private-network boundary of ``browser_exec``'s URL policy and must not become
    its bypass, so a page whose URL cannot be verified as public refuses page-reading methods.
    """
    from tools.browser_url_policy import eval_ssrf_guard_active, literal_targets_metadata, url_blocked
    guard_active = eval_ssrf_guard_active()
    guard = _METHOD_PARAM_GUARDS.get(method)
    if guard_active and guard is not None:
        probe, template = guard
        literal = probe(params or {}, url_blocked)
        if literal:
            return _blocked(template.format(literal), method)
    # The cloud-metadata floor applies with the private guard on or off, sidecar and local Chromium included,
    # as in browser_exec's _url_policy_error (#16234), to every URL the call could fetch.
    literals = _fetched_param_urls(method, params or {})
    metadata = next((u for u in literals if literal_targets_metadata(u)), None)
    if metadata:
        return _blocked(f"Blocked: CDP {method} targets a cloud metadata endpoint ({metadata}).", method)
    if not guard_active:
        return None
    if guard is None:
        # A public page does not make an unlisted method's own URL params safe: Network.loadNetworkResource
        # fetches its ``url`` from the browser's network position whatever page is open.
        literal = next((u for u in literals if u.lower().startswith(("http://", "https://")) and url_blocked(u)), None)
        if literal:
            return _blocked(f"Blocked: CDP {method} params target a private or internal address ({literal}).",
                            method)
    if method not in _CDP_PRIVATE_PAGE_ALLOWED_METHODS:
        blocked_url = _current_page_private_url(endpoint, target_id)
        if blocked_url == _PAGE_URL_UNVERIFIED:
            return _blocked(f"Blocked: could not read the browser's targets to verify that raw CDP method "
                            f"{method!r} cannot reach a private or internal page.", method)
        if blocked_url:
            return _blocked(f"Blocked: page URL targets a private or internal address ({blocked_url}). "
                            f"Raw CDP method {method!r} could expose private page content or state.", method)
    return None


async def _cdp_call(ws_url: str, method: str, params: dict[str, Any], target_id: Optional[str],
                    timeout: float) -> dict[str, Any]:
    """Make a single CDP call. With ``target_id``, ``Target.attachToTarget(flatten=True)`` multiplexes a
    page-level session over the browser-level WebSocket; without it ``method`` runs at browser level."""
    assert websockets is not None  # guarded by _WS_AVAILABLE at call-site
    from agent.proxy_bypass import loopback_connect_kwargs
    # max_size=None: CDP responses (e.g. DOM.getDocument) can be large; ping_interval=None: CDP
    # servers don't expect pings.
    async with websockets.connect(ws_url, max_size=None, open_timeout=timeout, close_timeout=5,
                                  ping_interval=None, **loopback_connect_kwargs(ws_url)) as ws:
        next_id = 1

        async def _send(req: dict[str, Any], what: str) -> dict[str, Any]:
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

        req: dict[str, Any] = {"method": method, "params": params or {}}
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


def _browser_cdp_via_supervisor(task_id: str, frame_id: str, method: str, params: Optional[dict[str, Any]],
                                timeout: float) -> str:
    """Route a CDP call through the live supervisor session for an OOPIF frame."""
    try:
        from tools.browser_supervisor import SUPERVISOR_REGISTRY  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover — defensive
        return tool_error(f"CDP supervisor is not available: {exc}. frame_id routing requires a running "
                          "supervisor attached via /browser connect or an active Browserbase session.")

    supervisor = SUPERVISOR_REGISTRY.get(task_id)
    if supervisor is None:
        return tool_error(f"No CDP supervisor is attached for task={task_id!r}. Call browser_exec or "
                          "/browser connect first so the supervisor can attach; Page.getFrameTree lists the "
                          "frame ids you can pass here.")

    tree = supervisor.snapshot().frame_tree
    frame_info: Optional[dict[str, Any]] = next(
        (f for f in [tree.get("top"), *(tree.get("children") or [])] if f and f.get("frame_id") == frame_id), None)
    if frame_info is None:  # frame_tree is capped at 30 entries — check the raw frames dict too.
        with supervisor._state_lock:  # type: ignore[attr-defined]
            raw = supervisor._frames.get(frame_id)  # type: ignore[attr-defined]
        frame_info = raw.to_dict() if raw is not None else None
    if frame_info is None:
        return tool_error(f"frame_id {frame_id!r} not found in supervisor state. "
                          "Call Page.getFrameTree to see the current frames.")

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


def browser_cdp(method: str, params: Optional[dict[str, Any]] = None, target_id: Optional[str] = None,
                frame_id: Optional[str] = None, timeout: float = 30.0, task_id: Optional[str] = None) -> str:
    """Send a raw CDP command (see ``CDP_DOCS_URL``). ``target_id`` attaches a fresh stateless connection
    to a tab; ``frame_id`` (an OOPIF from ``Page.getFrameTree``) routes through the supervisor's live
    WebSocket instead — the only reliable way to evaluate inside an iframe where fresh per-call connections
    hit signed-URL expiry (Browserbase). Both paths share the same private-page/SSRF guard. Returns JSON
    ``{"success": True, "method", "result"}`` or ``{"error": ...}``."""
    effective_task_id = task_id or "default"

    if frame_id:
        # An OOPIF's frame id is its iframe target id, so the guard judges that frame (or fails closed).
        blocked = _browser_cdp_private_guard(method=method, params=params or {}, target_id=frame_id,
                                             endpoint=_resolve_cdp_endpoint() if _WS_AVAILABLE else "")
        if blocked:
            return blocked
        return _browser_cdp_via_supervisor(task_id=effective_task_id, frame_id=frame_id, method=method,
                                           params=params, timeout=timeout)

    if not method or not isinstance(method, str):
        return tool_error("'method' is required (e.g. 'Target.getTargets')", cdp_docs=CDP_DOCS_URL)
    if not _WS_AVAILABLE:
        return tool_error("The 'websockets' Python package is required but not installed. "
                          "Run: hermes pm repair")
    endpoint = _resolve_cdp_endpoint()
    if not endpoint:
        return tool_error("No CDP endpoint is available. Run '/browser connect' to attach to a running Chrome, "
                          "Brave, Chromium, or Edge browser, or set 'browser.cdp_url' in config.yaml.", cdp_docs=CDP_DOCS_URL)
    if not endpoint.startswith(("ws://", "wss://")):
        return tool_error(f"CDP endpoint is not a WebSocket URL: {endpoint!r}. Expected ws://... or wss://... — "
                          "the /browser connect resolver should have rewritten this. Check that a Chromium-family "
                          "browser is actually listening on the debug port.")
    call_params: dict[str, Any] = params or {}
    if not isinstance(call_params, dict):
        return tool_error(f"'params' must be an object/dict, got {type(call_params).__name__}")

    blocked = _browser_cdp_private_guard(method=method, params=call_params, endpoint=endpoint, target_id=target_id)
    if blocked:
        return blocked

    try:
        safe_timeout = float(timeout) if timeout else 30.0
    except (TypeError, ValueError):
        safe_timeout = 30.0
    safe_timeout = max(1.0, min(safe_timeout, 300.0))
    try:
        result = _run_async(_cdp_call(endpoint, method, call_params, target_id, safe_timeout))
    except TimeoutError as exc:
        return tool_error(f"CDP call timed out after {safe_timeout}s: {exc}", method=method)
    except RuntimeError as exc:
        return tool_error(str(exc), method=method)
    except WebSocketException as exc:
        return tool_error(f"WebSocket error talking to CDP at {endpoint}: {exc}. The browser may have "
                          "disconnected — try '/browser connect' again.", method=method)
    except Exception as exc:  # pragma: no cover — unexpected
        logger.exception("browser_cdp unexpected error")
        return tool_error(f"Unexpected error: {type(exc).__name__}: {exc}", method=method)

    payload: dict[str, Any] = {"success": True, "method": method, "result": _redact_cdp_output(
        result, always_paths=_CDP_ALWAYS_BINARY_PATHS.get(method, ()),
        flagged_paths=_CDP_FLAGGED_BINARY_PATHS.get(method, ()))}
    if target_id:
        payload["target_id"] = target_id
    return json.dumps(payload, ensure_ascii=False)


BROWSER_CDP_SCHEMA: dict[str, Any] = {
    "name": "browser_cdp",
    "description": (
        "Send a raw Chrome DevTools Protocol (CDP) command. Escape hatch for browser operations the other "
        "browser tools don't cover.\n\n"
        "**Requires a reachable CDP endpoint.** Available when the user has run '/browser connect' to attach "
        "to a running Chrome, Brave, Chromium, or Edge browser, or when 'browser.cdp_url' is set in "
        "config.yaml. Not currently wired up for cloud backends (Browserbase, Browser Use, Firecrawl) — "
        "those expose CDP per session but live-session routing is a follow-up. If the tool is in your toolset "
        "at all, a CDP endpoint is already reachable.\n\n"
        f"**CDP method reference:** {CDP_DOCS_URL} — use an available documentation lookup or extraction "
        "tool on a method's URL (e.g. '/tot/Page/#method-handleJavaScriptDialog') to look up parameters and "
        "return shape.\n\n"
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
        "etc.): pass a frame_id from Page.getFrameTree. This routes through the CDP "
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
                "Optional. Out-of-process iframe (OOPIF) frame id from Page.getFrameTree. "
                "When set, routes the call through the CDP supervisor's live session for "
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
    """Availability check: offered only when a static CDP URL is set (a launched or cloud per-session
    ``cdp_url`` isn't surfaced) and the browser surface exists (browser-harness importable). Raw (no-I/O)
    gate: check_fns run at every startup, and resolving the endpoint over HTTP here would block launch on
    a stale endpoint."""
    from tools.browser_session_cdp import get_cdp_override_raw
    from tools.browser_use_cli import browser_exec_available
    return bool(browser_exec_available() and get_cdp_override_raw())


registry.register(
    name="browser_cdp",
    toolset="browser-cdp",
    schema=BROWSER_CDP_SCHEMA,
    handler=lambda args, **kw: browser_cdp(
        method=args.get("method", ""), params=args.get("params"), target_id=args.get("target_id"),
        frame_id=args.get("frame_id"), timeout=args.get("timeout", 30.0), task_id=kw.get("task_id"),
    ),
    check_fn=_browser_cdp_check,
    emoji="🧪",
)
