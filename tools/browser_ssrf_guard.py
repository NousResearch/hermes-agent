"""CDP Fetch-domain page guard for ``browser_exec`` SSRF enforcement (Region C).

Two cooperating layers:

* **Layer 1 — page-JS interceptor** (``_SSRF_GUARD_JS``): main-world,
  non-configurable/non-writable wrappers over ``fetch``/``XMLHttpRequest``/
  ``WebSocket``/``EventSource``/``navigator.sendBeacon``/``window.open`` and
  best-effort ``location`` mutators. Blocks by URL shape (literal private
  IPs, localhost, ``*.local``/``*.lan``/``*.internal``, metadata hostnames,
  non-http(s) schemes, unresolvable → block). Fail-fast UX + survival layer;
  never the sole control (element-based requests bypass every JS wrapper —
  accepted by design).

* **Layer 2 — CDP Fetch guard (authoritative, remote-IP-gated)**:
  ``BrowserSsrfGuard`` is a trusted Hermes-side process holding its own CDP
  session per page target (existing + new + OOPIF + worker family),
  enforcing per-request at the network boundary with two gates:

  1. **Pre-connect gate (Request stage)** — literal-IP URLs and guard-side
     DNS resolution, fail-closed on DNS failure in ALL modes (no
     proxy-delegation allowance at the guard layer). The request is failed
     (``Fetch.failRequest``) before any byte leaves the browser.
  2. **Async authoritative gate (browser-observed remote IP)** —
     ``Fetch.enable`` arms REQUEST-stage interception ONLY (a Response-stage
     pause would suppress ``Network.responseReceived`` for intercepted
     requests and deadlock every legitimate public request against real
     Chrome). ``Network.responseReceived`` flows unimpeded and is correlated
     per ``requestId``; when the browser-observed ``remoteIPAddress`` for a
     request is private/IMDS (split-horizon DNS / DNS-rebinding where the
     browser actually connected somewhere private), the guard emits the
     block marker on the report channel and the parent kills the CLI and
     withholds all output. No response-stage pausing: the marker is the
     enforcement, and the request-stage gate still fails closed on any
     private URL.

The guard is fail-closed in every mode: arm failure → ``browser_exec``
errors before spawn; a mid-exec block or guard death → exec terminates with
output withheld. Block markers are written to the guard's dedicated report
channel (``__HERMES_BROWSER_EXEC_SSRF_BLOCK__:<url>``).
"""

import asyncio
import ipaddress
import json
import logging
import os
import re
import socket
import sys
import time
from typing import Any, Dict, Optional
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

# Marker emitted to the report channel when any request is blocked.
BLOCK_MARKER = "__HERMES_BROWSER_EXEC_SSRF_BLOCK__:"

# TTL + cap for the guard's hostname verdict cache (bounded fan-out; a
# verdict is never cached across the private/public boundary, only reused
# within a short window for the same hostname).
_CACHE_TTL_S = 30.0
_CACHE_MAX = 512

# URL shapes that are blocked without any DNS work (parser-divergence and
# obvious local/metadata names).
_GUARD_SHAPE_RE = re.compile(
    r"(?i)(^|\\.)(localhost|localhost\\.localdomain)$"
    r"|(\\.local|\\.lan|\\.internal)$"
    r"|(^|\\.)(metadata\\.google\\.internal|metadata\\.goog)$"
    r"|(^|[^0-9])(169\\.254\\.|10\\.|192\\.168\\.|172\\.(1[6-9]|2[0-9]|3[01])\\.|100\\.(6[4-9]|[7-9][0-9])\\."
    r"|fd00:|fe80:|::1$|127\\.|0\\.0\\.0\\.0$)"
)


# Target types the Fetch guard arms. Worker-family targets (dedicated /
# service / shared workers), worklets (auction / interest-group / shared
# storage), and fenced frames cannot always carry the page-JS Layer 1
# interceptor (no DOM for worklets) but still get the authoritative Fetch +
# Network + remote-IP gates: a worker or fenced-frame fetch to
# 169.254.169.254 is just as much SSRF as a page fetch, and OOPIF iframes /
# fenced frames are separate CDP sessions that must be armed too.
_GUARD_ARMED_TARGET_TYPES = frozenset({
    "page", "iframe", "worker", "service_worker", "shared_worker",
    "background_page", "webview", "fencedframe",
    "auction_worklet", "interest_group_worklet", "shared_storage_worklet",
})

# Target types that have a DOM and therefore get the Layer 1 page-JS
# interceptor injected (worker/worklet sessions would reject Page.* anyway;
# fenced frames are document-bearing and get the JS interceptor too).
_GUARD_JS_TARGET_TYPES = frozenset({
    "page", "iframe", "webview", "background_page", "fencedframe",
})


def _hostname_of(url: str) -> str:
    try:
        parsed = urlsplit(url or "")
        return (parsed.hostname or "").strip().lower().rstrip(".")
    except ValueError:
        return ""


def _scheme_of(url: str) -> str:
    try:
        return (urlsplit(url or "").scheme or "").lower()
    except ValueError:
        return ""


def _normalize_ws_url(url: str) -> str:
    """ws:/wss: URLs normalize to http:/https: for the oracles."""
    lowered = (url or "").strip().lower()
    if lowered.startswith("ws://"):
        return "http://" + url.strip()[5:]
    if lowered.startswith("wss://"):
        return "https://" + url.strip()[6:]
    return url


def _guard_is_safe_url(url: str) -> bool:
    """``is_safe_url`` with the proxy-delegation branch suppressed.

    DNS failure on a hostname NEVER delegates to the proxy at the guard
    layer — the guard must positively confirm a public resolution, else fail
    closed. Everything else (trusted-host bypass, per-IP loop, toggle
    semantics) is the shared ``resolve_and_check_url`` behavior.
    """
    from tools.url_safety import resolve_and_check_url

    v = resolve_and_check_url(url)
    return v.ok


def browser_exec_blocked(url: str) -> bool:
    """Region C decision composition (pre-connect gate).

    1. Shape pre-check — fires regardless of DNS (localhost, *.local,
       *.lan, *.internal, metadata hostnames, raw IP shapes).
    2. Ungated oracle floor — ``is_always_blocked_url`` (IMDS set).
    3. Fail-closed safety check — ``_guard_is_safe_url`` (no proxy
       delegation; ``error:dns`` blocks).
    """
    url = _normalize_ws_url(url or "")
    hostname = _hostname_of(url)
    if not hostname:
        return True  # unparseable → block
    if _GUARD_SHAPE_RE.search(hostname):
        return True
    from tools.url_safety import is_always_blocked_url

    if is_always_blocked_url(url):
        return True
    if not _guard_is_safe_url(url):
        return True
    return False


def remote_ip_blocked(remote_ip: str, url: str) -> bool:
    """Authoritative browser-side remote-IP gate (Region C §3).

    The decision uses the IP the BROWSER actually connected to — valid in
    cloud, local, and CDP-override modes, and closes split-horizon DNS
    rebinding. Honors ``_TRUSTED_PRIVATE_IP_HOSTS`` consistently (https
    only).

    Chrome reports IPv6 ``remoteIPAddress`` values bracketed (``[2606:4700:
    ...]``); the brackets are stripped before parsing so a public IPv6 peer
    is never misclassified as unparseable (fail-closed false positive).
    """
    raw = str(remote_ip or "").strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    try:
        ip = ipaddress.ip_address(raw)
    except ValueError:
        return True  # missing/unparseable remote IP → fail closed
    from tools.url_safety import (
        _ALWAYS_BLOCKED_IPS,
        _ALWAYS_BLOCKED_NETWORKS,
        _allows_private_ip_resolution,
        _is_blocked_ip,
    )

    if ip in _ALWAYS_BLOCKED_IPS or any(ip in net for net in _ALWAYS_BLOCKED_NETWORKS):
        return True
    if _allows_private_ip_resolution(_hostname_of(url), _scheme_of(url)):
        return False
    if _is_blocked_ip(ip):
        return True
    return False


# ── Layer 1: page-JS interceptor source ────────────────────────────────────
# Idempotent via Symbol.for('__hermesSsrfGuard'); non-configurable/
# non-writable descriptors; closure-held originals. Unresolvable URLs are
# blocked (fail-closed at Layer 1); public-looking names pass to Layer 2.
_SSRF_GUARD_JS = r"""
(() => {
  'use strict';
  const KEY = Symbol.for('__hermesSsrfGuard');
  if (window[KEY]) { return; }
  const guard = { blocked: 0, blockedUrls: [] };
  Object.defineProperty(window, KEY, { value: guard, configurable: false, writable: false });

  const IPV4_RE = /^(?:\d{1,3}\.){3}\d{1,3}$/;
  const IPV6_RE = /^\[[0-9a-fA-F:]+\]$/;
  const LOCAL_RE = /(^|\.)(localhost|localhost\.localdomain)$|(\.local|\.lan|\.internal)$|(^|\.)(metadata\.google\.internal|metadata\.goog)$/;

  function octetInRange(ip) {
    return ip.split('.').every(p => { const n = Number(p); return n >= 0 && n <= 255; });
  }
  function isPrivateLiteral(url) {
    try {
      const u = new URL(url, window.location.href);
      const host = u.hostname.replace(/^\[|\]$/g, '');
      if (LOCAL_RE.test(host)) { return true; }
      if (IPV4_RE.test(host)) {
        if (!octetInRange(host)) { return true; }
        const parts = host.split('.').map(Number);
        if (parts[0] === 10) { return true; }
        if (parts[0] === 127) { return true; }
        if (parts[0] === 0) { return true; }
        if (parts[0] === 169 && parts[1] === 254) { return true; }
        if (parts[0] === 172 && parts[1] >= 16 && parts[1] <= 31) { return true; }
        if (parts[0] === 192 && parts[1] === 168) { return true; }
        if (parts[0] === 100 && parts[1] >= 64 && parts[1] <= 127) { return true; }
        if (parts[0] === 198 && (parts[1] === 18 || parts[1] === 19)) { return true; }
        return false;
      }
      if (IPV6_RE.test('[' + host + ']') || host.includes(':')) {
        const h = host.toLowerCase();
        if (h === '::1' || h.startsWith('fe80:') || h.startsWith('fc') || h.startsWith('fd')) { return true; }
        if (h.startsWith('::ffff:')) {
          return isPrivateLiteral('http://' + h.slice(7) + '/');
        }
        return false;
      }
      return false;
    } catch (e) {
      return true; // unresolvable URL → fail closed at Layer 1
    }
  }
  function blocked(url) {
    try {
      const u = new URL(url, window.location.href);
      if (!/^https?:$/.test(u.protocol)) { return true; }
    } catch (e) {
      return true;
    }
    if (isPrivateLiteral(url)) { return true; }
    return false;
  }
  function record(url) {
    guard.blocked += 1;
    if (guard.blockedUrls.length < 64) { guard.blockedUrls.push(String(url)); }
  }

  const _fetch = window.fetch;
  const _XHR = window.XMLHttpRequest;
  const _WS = window.WebSocket;
  const _ES = window.EventSource;
  const _sendBeacon = navigator.sendBeacon ? navigator.sendBeacon.bind(navigator) : null;
  const _open = window.open;

  const fetchWrapper = function (input, init) {
    const url = (typeof input === 'string') ? input : (input && input.url) || '';
    if (blocked(url)) { record(url); return Promise.reject(new TypeError('fetch blocked by Hermes SSRF guard')); }
    return _fetch.call(this, input, init);
  };
  Object.defineProperty(window, 'fetch', { value: fetchWrapper, configurable: false, writable: false });

  const xhrOpen = _XHR.prototype.open;
  const xhrSend = _XHR.prototype.send;
  const xhrOpenWrapper = function (method, url, async, user, password) {
    if (blocked(String(url))) { record(url); throw new TypeError('XHR blocked by Hermes SSRF guard'); }
    return xhrOpen.call(this, method, url, async, user, password);
  };
  Object.defineProperty(_XHR.prototype, 'open', { value: xhrOpenWrapper, configurable: false, writable: false });
  // send() re-checks so a late-bound URL is still covered.
  const xhrSendWrapper = function (body) {
    try {
      if (blocked(String(this.__hermes_ssrf_url || ''))) { record(this.__hermes_ssrf_url); throw new TypeError('XHR blocked by Hermes SSRF guard'); }
    } catch (e) { throw e; }
    return xhrSend.call(this, body);
  };
  Object.defineProperty(_XHR.prototype, 'send', { value: xhrSendWrapper, configurable: false, writable: false });

  const WSWrapper = function (url, protocols) {
    if (blocked(String(url))) { record(url); throw new TypeError('WebSocket blocked by Hermes SSRF guard'); }
    return protocols === undefined ? new _WS(url) : new _WS(url, protocols);
  };
  Object.defineProperty(window, 'WebSocket', { value: WSWrapper, configurable: false, writable: false });

  const ESWrapper = function (url, options) {
    if (blocked(String(url))) { record(url); throw new TypeError('EventSource blocked by Hermes SSRF guard'); }
    return options === undefined ? new _ES(url) : new _ES(url, options);
  };
  Object.defineProperty(window, 'EventSource', { value: ESWrapper, configurable: false, writable: false });

  if (_sendBeacon) {
    const beaconWrapper = function (url, data) {
      if (blocked(String(url))) { record(url); return false; }
      return _sendBeacon(url, data);
    };
    Object.defineProperty(navigator, 'sendBeacon', { value: beaconWrapper, configurable: false, writable: false });
  }

  const openWrapper = function (url, target, features) {
    if (blocked(String(url))) { record(url); return null; }
    return _open.call(window, url, target, features);
  };
  Object.defineProperty(window, 'open', { value: openWrapper, configurable: false, writable: false });

  // Best-effort location mutators (assignment is configurable at the
  // property level; wrap the mutating methods that pages actually use).
  try {
    const loc = window.location;
    const desc = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(loc), 'href');
    if (desc && desc.set) {
      Object.defineProperty(loc, 'href', {
        set(v) {
          if (blocked(String(v))) { record(v); return; }
          desc.set.call(loc, v);
        },
        get() { return desc.get ? desc.get.call(loc) : loc.toString(); },
        configurable: false,
      });
    }
  } catch (e) { /* best-effort */ }
})();
"""


# ── Layer 2: CDP Fetch guard ────────────────────────────────────────────────

class _ReportChannel:
    """Report channel for block markers (inherited fd or connected socket)."""

    def __init__(self, target: Any) -> None:
        self._target = target  # int fd | socket.socket | None
        self._file = None

    def emit(self, line: str) -> None:
        data = (line + "\n").encode("utf-8", "replace")
        try:
            if isinstance(self._target, int):
                os.write(self._target, data)
            elif isinstance(self._target, socket.socket):
                self._target.sendall(data)
        except Exception:
            pass


class BrowserSsrfGuard:
    """Trusted per-exec CDP Fetch/Network guard for one browser endpoint.

    All CDP I/O runs on this process's asyncio loop. ``report_fd`` accepts an
    inherited OS fd (POSIX) or a connected ``socket.socket`` (cross-platform
    subprocess channel); block markers go there and never into the model's
    namespace.
    """

    def __init__(self, cdp_url: str, report_fd: Any = None, *, task_id: str = "") -> None:
        self.cdp_url = cdp_url
        self.task_id = task_id
        self.report = _ReportChannel(report_fd)
        self._ws = None
        self._next_call_id = 1
        self._pending: Dict[int, asyncio.Future] = {}
        self._verdict_cache: Dict[str, tuple[float, bool]] = {}
        self._sessions_armed: set[str] = set()
        self._reader: Optional[asyncio.Task] = None

    # ── Report ────────────────────────────────────────────────────────────

    def _emit_block(self, url: str, stage: str) -> None:
        logger.warning("browser ssrf guard blocked %s at %s stage", url[:160], stage)
        self.report.emit(BLOCK_MARKER + url)

    # ── CDP plumbing ──────────────────────────────────────────────────────

    async def _cdp(self, method: str, params: Optional[Dict[str, Any]] = None,
                   *, session_id: Optional[str] = None, timeout: float = 10.0) -> Dict[str, Any]:
        if self._ws is None:
            raise RuntimeError("ssrf guard WebSocket is not connected")
        call_id = self._next_call_id
        self._next_call_id += 1
        payload: Dict[str, Any] = {"id": call_id, "method": method}
        if params:
            payload["params"] = params
        if session_id:
            payload["sessionId"] = session_id
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[call_id] = fut
        await self._ws.send(json.dumps(payload))
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        finally:
            self._pending.pop(call_id, None)

    async def _read_loop(self) -> None:
        assert self._ws is not None
        async for raw in self._ws:
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            if "id" in msg:
                fut = self._pending.pop(msg["id"], None)
                if fut is not None and not fut.done():
                    if "error" in msg:
                        fut.set_exception(RuntimeError(f"CDP error: {msg['error']}"))
                    else:
                        fut.set_result(msg)
            elif "method" in msg:
                method = msg["method"]
                params = msg.get("params", {})
                session_id = msg.get("sessionId")
                if method == "Fetch.requestPaused":
                    asyncio.create_task(self.on_request_paused(params, session_id))
                elif method == "Network.responseReceived":
                    asyncio.create_task(self.on_response_received(params, session_id))
                elif method == "Target.attachedToTarget":
                    asyncio.create_task(self._on_target_attached(params, session_id))

    # ── Arming ────────────────────────────────────────────────────────────

    async def arm(self) -> None:
        """Connect, arm every current target, and auto-attach for new targets.

        Completes once every current target is armed (new/OOPIF/worker
        targets are paused via ``waitForDebuggerOnStart`` and resumed only
        AFTER the session is fully armed — witness C3). The reader loop keeps
        running in the background for the guard's lifetime; ``teardown()``
        stops it. Raises on any arm failure — the caller fails the exec
        closed.
        """
        import websockets

        self._ws = await asyncio.wait_for(
            websockets.connect(self.cdp_url, max_size=50 * 1024 * 1024),
            timeout=15.0,
        )
        reader = asyncio.create_task(self._read_loop(), name="ssrf-guard-reader")
        self._reader = reader
        try:
            resp = await self._cdp("Target.getTargets")
            targets = resp.get("result", {}).get("targetInfos", [])
            for target in targets:
                if target.get("type") in _GUARD_ARMED_TARGET_TYPES:
                    await self._arm_existing_target(
                        target.get("targetId"), target.get("type")
                    )
            await self._cdp(
                "Target.setAutoAttach",
                {"autoAttach": True, "waitForDebuggerOnStart": True, "flatten": True},
            )
        except BaseException:
            if not reader.done():
                reader.cancel()
                try:
                    await reader
                except (asyncio.CancelledError, Exception):
                    pass
            self._reader = None
            raise

    async def _arm_existing_target(
        self, target_id: Optional[str], target_type: Optional[str] = None
    ) -> None:
        if not target_id:
            return
        attach = await self._cdp("Target.attachToTarget", {"targetId": target_id, "flatten": True})
        session_id = attach.get("result", {}).get("sessionId")
        if session_id:
            await self.arm_session(session_id, target_type)

    async def _on_target_attached(self, params: Dict[str, Any], root_session: Optional[str]) -> None:
        target_info = params.get("targetInfo") or {}
        session_id = params.get("sessionId")
        waiting = bool(params.get("waitingForDebugger"))
        if target_info.get("type") in _GUARD_ARMED_TARGET_TYPES and session_id:
            # Arm FIRST, then resume — the target stays paused until the
            # guard is fully installed on it (any type: page, OOPIF iframe,
            # worker family).
            await self.arm_session(session_id, target_info.get("type"))
            if waiting:
                try:
                    await self._cdp("Runtime.runIfWaitingForDebugger", session_id=session_id)
                except Exception:
                    pass

    async def arm_session(
        self, session_id: str, target_type: Optional[str] = None
    ) -> None:
        """Install Layer 1 + Layer 2 on one session (any auto-attached type).

        Worker-family sessions (worker/service_worker/shared_worker) get the
        authoritative Fetch + Network + remote-IP gates (Layer 2); the
        page-JS Layer 1 interceptor only applies to DOM-having targets.
        """
        try:
            await self._cdp("Runtime.enable", session_id=session_id)
        except Exception:
            pass
        if target_type is None or target_type in _GUARD_JS_TARGET_TYPES:
            try:
                await self._cdp("Page.enable", session_id=session_id)
            except Exception:
                pass
            try:
                await self._cdp(
                    "Page.addScriptToEvaluateOnNewDocument",
                    {"source": _SSRF_GUARD_JS, "runImmediately": True},
                    session_id=session_id, timeout=5.0,
                )
            except Exception:
                pass
            try:
                await self._cdp(
                    "Runtime.evaluate",
                    {"expression": _SSRF_GUARD_JS, "returnByValue": True},
                    session_id=session_id, timeout=3.0,
                )
            except Exception:
                pass
        # REQUEST-stage interception ONLY. A Response-stage pause would
        # suppress Network.responseReceived for intercepted requests, so the
        # remote-IP gate would wait on an event that never arrives and fail
        # every legitimate public request (deadlock fix); the async gate runs
        # off Network.responseReceived, which flows unimpeded with
        # Request-stage-only Fetch.enable.
        await self._cdp(
            "Fetch.enable",
            {
                "patterns": [
                    {"urlPattern": "*", "requestStage": "Request"},
                ],
                "handleAuthRequests": False,
            },
            session_id=session_id,
        )
        await self._cdp("Network.enable", session_id=session_id)
        self._sessions_armed.add(session_id)

    # ── Decision handlers ─────────────────────────────────────────────────

    async def on_request_paused(self, params: Dict[str, Any], session_id: Optional[str]) -> None:
        request = params.get("request") or {}
        url = str(request.get("url") or "")
        request_id = str(params.get("requestId") or "")

        # Pre-connect gate (Request stage only — see arm_session: Fetch.enable
        # arms Request-stage interception exclusively, so every pause is a
        # pre-connect decision; the remote-IP gate is async off
        # Network.responseReceived and never pauses a response).
        scheme = _scheme_of(url)
        if scheme not in ("http", "https", "ws", "wss"):
            # Non-http(s) schemes cannot reach private TCP endpoints.
            await self._continue(params, request_id, session_id)
            return
        if await asyncio.to_thread(self._cached_blocked, url):
            await self._fail(params, request_id, session_id)
            self._emit_block(url, "request")
            return
        await self._continue(params, request_id, session_id)

    async def on_response_received(self, params: Dict[str, Any], session_id: Optional[str]) -> None:
        response = params.get("response") or {}
        remote_ip = str(response.get("remoteIPAddress") or "").strip()
        url = str(response.get("url") or "")
        # Async authoritative remote-IP gate: the browser-observed remote IP
        # of ANY request (Document navigation, XHR/fetch, WebSocket
        # handshake, worker/fenced-frame fetch, …) is checked WITHOUT
        # pausing the request (Response-stage Fetch pauses are gone — see
        # arm_session). A private/IMDS remote IP emits the block marker on
        # the report channel; the parent kills the CLI and withholds all
        # output. This catches split-horizon DNS / DNS-rebinding cases where
        # the request-stage resolution looked public but the browser
        # actually connected somewhere private.
        if remote_ip and url:
            if await asyncio.to_thread(remote_ip_blocked, remote_ip, url):
                self._emit_block(url, f"response:{remote_ip}")

    async def _continue(self, params: Dict[str, Any], request_id: str,
                        session_id: Optional[str]) -> None:
        try:
            await self._cdp(
                "Fetch.continueRequest", {"requestId": request_id},
                session_id=session_id,
            )
        except Exception:
            pass

    async def _fail(self, params: Dict[str, Any], request_id: str,
                    session_id: Optional[str]) -> None:
        try:
            await self._cdp(
                "Fetch.failRequest",
                {"requestId": request_id, "errorReason": "BlockedByClient"},
                session_id=session_id,
            )
        except Exception:
            pass

    def _cached_blocked(self, url: str) -> bool:
        """Bounded, short-TTL verdict cache (hostname-keyed)."""
        host = _hostname_of(url)
        now = time.monotonic()
        cached = self._verdict_cache.get(host)
        if cached and now - cached[0] < _CACHE_TTL_S:
            return cached[1]
        verdict = browser_exec_blocked(url)
        if len(self._verdict_cache) >= _CACHE_MAX:
            self._verdict_cache.clear()
        self._verdict_cache[host] = (now, verdict)
        return verdict

    async def teardown(self) -> None:
        if self._reader is not None and not self._reader.done():
            self._reader.cancel()
            try:
                await self._reader
            except (asyncio.CancelledError, Exception):
                pass
            self._reader = None
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None


async def _guard_main(cdp_url: str, report_port: Optional[int], report_token: str) -> int:
    report_sock: Optional[socket.socket] = None
    if report_port:
        report_sock = socket.create_connection(("127.0.0.1", report_port), timeout=10)
    guard = BrowserSsrfGuard(cdp_url, report_sock or None)
    try:
        await asyncio.wait_for(guard.arm(), timeout=30.0)
    except Exception:
        if report_sock is not None:
            try:
                report_sock.sendall(b"__HERMES_SSRF_GUARD_ARM_FAILED__\n")
            except Exception:
                pass
        return 1
    # Emit the ARMED/READY marker ONLY after arm() completed on every
    # current target. The parent treats a missing READY as an arm failure,
    # so emitting it before arming would let the exec start with no Fetch
    # interception / no browser-side remote-IP gate (defect-2 fix).
    if report_sock is not None:
        try:
            report_sock.sendall(
                f"__HERMES_SSRF_GUARD_READY__:{report_token}\n".encode("utf-8")
            )
        except Exception:
            # Report channel gone — nobody is listening; stop the guard.
            await guard.teardown()
            return 1
    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass
    finally:
        await guard.teardown()
        if report_sock is not None:
            try:
                report_sock.close()
            except Exception:
                pass
    return 0


def main(argv: Optional[list] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cdp_url = ""
    report_port = None
    report_token = ""
    i = 0
    while i < len(argv):
        if argv[i] == "--cdp-url" and i + 1 < len(argv):
            cdp_url = argv[i + 1]
            i += 2
        elif argv[i] == "--report-port" and i + 1 < len(argv):
            try:
                report_port = int(argv[i + 1])
            except ValueError:
                report_port = None
            i += 2
        elif argv[i] == "--report-token" and i + 1 < len(argv):
            report_token = argv[i + 1]
            i += 2
        else:
            i += 1
    if not cdp_url:
        print("usage: python -m tools.browser_ssrf_guard --cdp-url WS --report-port PORT --report-token TOKEN",
              file=sys.stderr)
        return 2
    try:
        return asyncio.run(_guard_main(cdp_url, report_port, report_token))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
