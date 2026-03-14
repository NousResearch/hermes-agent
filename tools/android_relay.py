"""
android_relay — WebSocket relay that bridges HTTP tool calls to a phone over WS.

The relay runs an aiohttp server exposing:
  - /ws          WebSocket endpoint the phone connects to (?token=CODE for auth)
  - /ping, /screen, /tap, /tap_text, /type, /swipe, /open_app, /press_key,
    /screenshot, /scroll, /wait, /apps, /current_app   HTTP endpoints matching
    the bridge API consumed by android_tool.py

Flow:
  1. Phone connects via WebSocket with ?token=<pairing_code>
  2. Python tool makes an HTTP request to e.g. /screen
  3. Relay wraps request as JSON command, sends over WS to phone
  4. Phone executes command, sends JSON response back over WS
  5. Relay returns the phone's response to the HTTP caller

Command JSON format:
  Relay -> Phone:  {"request_id": "uuid", "method": "GET|POST", "path": "/screen",
                    "params": {...}, "body": {...}}
  Phone -> Relay:  {"request_id": "uuid", "result": {...}, "status": 200}
"""

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from typing import Optional

import aiohttp
from aiohttp import web

logger = logging.getLogger("android_relay")

# ── Module-level state ────────────────────────────────────────────────────────

_relay_lock = threading.Lock()
_relay_instance: Optional["_RelayState"] = None


class _RelayState:
    """Holds all mutable state for one running relay instance."""

    def __init__(self, pairing_code: str, port: int):
        self.pairing_code: str = pairing_code
        self.port: int = port

        # asyncio loop running in the background thread
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None

        # aiohttp plumbing
        self.app: Optional[web.Application] = None
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None

        # The single connected phone WebSocket (or None)
        self.phone_ws: Optional[web.WebSocketResponse] = None
        self.phone_ws_lock = asyncio.Lock()  # created lazily in the event loop

        # Pending requests: request_id -> asyncio.Future
        self.pending: dict[str, asyncio.Future] = {}
        self.pending_lock: Optional[asyncio.Lock] = None  # created lazily

        # Shutdown event
        self.shutdown_event: Optional[asyncio.Event] = None


# ── Public API (called from sync code) ────────────────────────────────────────

def start_relay(pairing_code: str, port: int = 0) -> None:
    """Start the relay in a background thread.  No-op if already running."""
    global _relay_instance
    if port == 0:
        port = int(os.getenv("ANDROID_RELAY_PORT", "8766"))

    with _relay_lock:
        if _relay_instance is not None and _relay_instance.thread is not None and _relay_instance.thread.is_alive():
            logger.info("Relay already running on port %d", _relay_instance.port)
            return

        state = _RelayState(pairing_code, port)
        _relay_instance = state

        ready = threading.Event()
        t = threading.Thread(target=_run_loop, args=(state, ready), daemon=True, name="android-relay")
        state.thread = t
        t.start()
        # Wait until the server is actually listening (up to 10 s)
        if not ready.wait(timeout=10):
            logger.error("Relay failed to start within 10 seconds")
            raise RuntimeError("Relay failed to start")
        logger.info("Relay started on port %d", port)


def stop_relay() -> None:
    """Gracefully stop the relay."""
    global _relay_instance
    with _relay_lock:
        state = _relay_instance
        if state is None:
            return
        _relay_instance = None

    # Signal the event loop to shut down
    if state.loop is not None and state.shutdown_event is not None:
        state.loop.call_soon_threadsafe(state.shutdown_event.set)
    if state.thread is not None:
        state.thread.join(timeout=5)
    logger.info("Relay stopped")


def is_relay_running() -> bool:
    with _relay_lock:
        s = _relay_instance
        return s is not None and s.thread is not None and s.thread.is_alive()


def is_phone_connected() -> bool:
    with _relay_lock:
        s = _relay_instance
        if s is None:
            return False
        ws = s.phone_ws
        return ws is not None and not ws.closed


def get_relay_url() -> str:
    with _relay_lock:
        s = _relay_instance
        port = s.port if s else int(os.getenv("ANDROID_RELAY_PORT", "8766"))
    return f"http://localhost:{port}"


def set_pairing_code(code: str) -> None:
    """Update the pairing code (e.g. when user reconnects with new code)."""
    with _relay_lock:
        s = _relay_instance
        if s is not None:
            s.pairing_code = code
            logger.info("Pairing code updated")


# ── Background event-loop entry point ─────────────────────────────────────────

def _run_loop(state: _RelayState, ready: threading.Event) -> None:
    """Runs in the background thread — creates an event loop and serves."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    state.loop = loop
    state.phone_ws_lock = asyncio.Lock()
    state.pending_lock = asyncio.Lock()
    state.shutdown_event = asyncio.Event()

    try:
        loop.run_until_complete(_serve(state, ready))
    except Exception:
        logger.exception("Relay event loop crashed")
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()


async def _serve(state: _RelayState, ready: threading.Event) -> None:
    """Build the aiohttp app, start the site, and block until shutdown."""
    app = web.Application()
    state.app = app

    # WebSocket endpoint
    app.router.add_get("/ws", lambda req: _handle_ws(req, state))

    # HTTP bridge endpoints (GET)
    for path in ("/ping", "/screen", "/screenshot", "/apps", "/current_app"):
        app.router.add_get(path, lambda req, p=path: _handle_http(req, state, p))

    # HTTP bridge endpoints (POST)
    for path in ("/tap", "/tap_text", "/type", "/swipe", "/open_app", "/press_key", "/scroll", "/wait"):
        app.router.add_post(path, lambda req, p=path: _handle_http(req, state, p))

    runner = web.AppRunner(app)
    state.runner = runner
    await runner.setup()

    site = web.TCPSite(runner, "0.0.0.0", state.port)
    state.site = site
    await site.start()

    logger.info("Relay listening on 0.0.0.0:%d", state.port)
    ready.set()

    # Block until shutdown is signalled
    await state.shutdown_event.wait()

    # Cleanup
    await _cleanup_phone(state, reason="relay shutdown")
    await runner.cleanup()
    logger.info("Relay server cleaned up")


# ── Rate limiting for WebSocket auth ─────────────────────────────────────────

_AUTH_MAX_ATTEMPTS = 5        # max failed attempts before blocking
_AUTH_WINDOW_SECONDS = 60     # sliding window for counting failures
_AUTH_BLOCK_SECONDS = 300     # how long to block an IP (5 minutes)
_AUTH_CLEANUP_INTERVAL = 120  # seconds between cleanup sweeps

# {ip: [timestamp, timestamp, ...]} — tracks failed auth attempt times per IP
_auth_failures: dict[str, list[float]] = {}
# {ip: unblock_timestamp} — IPs currently blocked
_auth_blocked: dict[str, float] = {}
_auth_lock = threading.Lock()
_auth_last_cleanup: float = 0.0


def _auth_cleanup() -> None:
    """Remove expired entries from the failure and block dicts."""
    global _auth_last_cleanup
    now = time.monotonic()
    if now - _auth_last_cleanup < _AUTH_CLEANUP_INTERVAL:
        return
    _auth_last_cleanup = now

    # Remove expired blocks
    expired_blocks = [ip for ip, until in _auth_blocked.items() if now >= until]
    for ip in expired_blocks:
        del _auth_blocked[ip]

    # Remove stale failure windows
    cutoff = now - _AUTH_WINDOW_SECONDS
    stale = []
    for ip, timestamps in _auth_failures.items():
        _auth_failures[ip] = [t for t in timestamps if t > cutoff]
        if not _auth_failures[ip]:
            stale.append(ip)
    for ip in stale:
        del _auth_failures[ip]


def _auth_is_blocked(ip: str) -> bool:
    """Check whether *ip* is currently blocked. Also triggers periodic cleanup."""
    now = time.monotonic()
    with _auth_lock:
        _auth_cleanup()
        until = _auth_blocked.get(ip)
        if until is not None:
            if now < until:
                return True
            # Block expired — remove it
            del _auth_blocked[ip]
    return False


def _auth_record_failure(ip: str) -> None:
    """Record a failed auth attempt for *ip*; block if threshold reached."""
    now = time.monotonic()
    with _auth_lock:
        timestamps = _auth_failures.setdefault(ip, [])
        timestamps.append(now)
        # Prune timestamps outside the window
        cutoff = now - _AUTH_WINDOW_SECONDS
        _auth_failures[ip] = [t for t in timestamps if t > cutoff]

        if len(_auth_failures[ip]) >= _AUTH_MAX_ATTEMPTS:
            _auth_blocked[ip] = now + _AUTH_BLOCK_SECONDS
            _auth_failures.pop(ip, None)
            logger.warning(
                "IP %s blocked for %ds after %d failed auth attempts",
                ip, _AUTH_BLOCK_SECONDS, _AUTH_MAX_ATTEMPTS,
            )


# ── WebSocket handler (phone side) ───────────────────────────────────────────

async def _handle_ws(request: web.Request, state: _RelayState) -> web.WebSocketResponse:
    # Rate limiting — check before token validation
    remote_ip = request.remote or "unknown"
    if _auth_is_blocked(remote_ip):
        logger.warning("Auth attempt from blocked IP %s — returning 429", remote_ip)
        raise web.HTTPTooManyRequests(text="Too many failed authentication attempts. Try again later.")

    token = request.query.get("token", "")
    if token.upper() != state.pairing_code.upper():
        _auth_record_failure(remote_ip)
        logger.warning("Phone WS rejected — bad token (got %s) from %s", token, remote_ip)
        raise web.HTTPForbidden(text="Invalid pairing code")

    ws = web.WebSocketResponse(heartbeat=15.0)
    await ws.prepare(request)

    # Only one phone at a time — kick previous if any
    async with state.phone_ws_lock:
        if state.phone_ws is not None and not state.phone_ws.closed:
            logger.info("Replacing previous phone connection")
            await state.phone_ws.close(code=aiohttp.WSCloseCode.GOING_AWAY, message=b"replaced")
        state.phone_ws = ws

    logger.info("Phone connected from %s", request.remote)

    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                await _on_phone_message(state, msg.data)
            elif msg.type == aiohttp.WSMsgType.ERROR:
                logger.error("Phone WS error: %s", ws.exception())
                break
    finally:
        await _cleanup_phone(state, reason="phone disconnected")

    return ws


async def _on_phone_message(state: _RelayState, raw: str) -> None:
    """Route an incoming message from the phone to the matching pending future."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Non-JSON message from phone: %s", raw[:200])
        return

    request_id = data.get("request_id")
    if not request_id:
        logger.warning("Phone message missing request_id: %s", raw[:200])
        return

    async with state.pending_lock:
        future = state.pending.pop(request_id, None)

    if future is None:
        logger.debug("No pending future for request_id=%s (possibly timed out)", request_id)
        return

    if not future.done():
        future.set_result(data)


async def _cleanup_phone(state: _RelayState, reason: str = "") -> None:
    """Clean up phone connection and cancel all pending requests."""
    async with state.phone_ws_lock:
        ws = state.phone_ws
        state.phone_ws = None

    if ws is not None and not ws.closed:
        await ws.close()

    # Fail all pending futures
    async with state.pending_lock:
        pending = dict(state.pending)
        state.pending.clear()

    for rid, fut in pending.items():
        if not fut.done():
            fut.set_exception(ConnectionError(f"Phone disconnected ({reason})"))

    if pending:
        logger.info("Cancelled %d pending requests (%s)", len(pending), reason)


# ── HTTP handler (tool side) ─────────────────────────────────────────────────

_RESPONSE_TIMEOUT = 30  # seconds

async def _handle_http(request: web.Request, state: _RelayState, path: str) -> web.Response:
    """Forward an HTTP request from a tool to the phone over WebSocket."""
    ws = state.phone_ws
    if ws is None or ws.closed:
        return web.json_response(
            {"error": "No phone connected. Open the Hermes app on your phone and connect."},
            status=503,
        )

    # Build the command envelope
    request_id = str(uuid.uuid4())
    method = request.method  # GET or POST
    params = dict(request.query)

    body = {}
    if method == "POST":
        try:
            body = await request.json()
        except Exception:
            body = {}

    command = {
        "request_id": request_id,
        "method": method,
        "path": path,
        "params": params,
        "body": body,
    }
    logger.info(">>> %s %s body=%s", method, path, json.dumps(body) if body else "{}")

    # Register a future *before* sending so we never miss the reply
    future = state.loop.create_future()
    async with state.pending_lock:
        state.pending[request_id] = future

    try:
        await ws.send_json(command)
    except Exception as exc:
        async with state.pending_lock:
            state.pending.pop(request_id, None)
        logger.error("Failed to send command to phone: %s", exc)
        return web.json_response(
            {"error": f"Failed to send command to phone: {exc}"},
            status=502,
        )

    # Wait for the phone's response
    try:
        response_data = await asyncio.wait_for(future, timeout=_RESPONSE_TIMEOUT)
    except asyncio.TimeoutError:
        async with state.pending_lock:
            state.pending.pop(request_id, None)
        logger.warning("Phone did not respond within %ds for %s %s", _RESPONSE_TIMEOUT, method, path)
        return web.json_response(
            {"error": f"Phone did not respond within {_RESPONSE_TIMEOUT}s"},
            status=504,
        )
    except ConnectionError as exc:
        return web.json_response({"error": str(exc)}, status=502)

    # Return the phone's result
    status = response_data.get("status", 200)
    result = response_data.get("result", {})
    return web.json_response(result, status=status)
