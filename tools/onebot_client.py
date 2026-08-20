"""Shared OneBot v11 client for QQ integrations (HTTP or WebSocket).

Several Hermes tools talk to a running OneBot v11 implementation
(NapCat / Lagrange.Core): the ``qzone`` tool borrows the logged-in QQ
session to publish 说说, and the ``qq_voice`` tool sends synthesized
speech as a native QQ voice message. Both need the same plumbing, so it
lives here in one place rather than being duplicated per tool.

Transport is chosen automatically from the configured endpoint's URL
scheme:

* ``http://`` / ``https://`` — the OneBot v11 HTTP API (one POST per action).
* ``ws://`` / ``wss://``     — the OneBot v11 WebSocket API. Many NapCat /
  Lagrange deployments expose *only* a WebSocket server; each call opens a
  short-lived connection, sends one action, and reads the matching reply
  (skipping any pushed events).

Configuration (environment variables):
- ``ONEBOT_HTTP_URL``     -- OneBot endpoint, ``http(s)://`` or ``ws(s)://``.
- ``ONEBOT_WS_URL``       -- used as a fallback when ``ONEBOT_HTTP_URL`` is
                             unset (lets a WS-only deployment configure just
                             the one URL it has).
- ``ONEBOT_ACCESS_TOKEN`` -- optional bearer / access token.

This is a plain helper module — it registers no tools, so the registry's
module scan never imports it as a tool. Tools import the functions here.
"""

import asyncio
import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request

logger = logging.getLogger(__name__)

# Default per-request timeout (seconds). Callers doing heavier work (e.g.
# uploading media) may pass a larger value.
ONEBOT_TIMEOUT = 15


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def onebot_base_url() -> str:
    """Return the configured OneBot endpoint URL (no trailing slash).

    Prefers ``ONEBOT_HTTP_URL``; falls back to ``ONEBOT_WS_URL`` so a
    WebSocket-only deployment can configure just the one URL it has.
    """
    url = os.getenv("ONEBOT_HTTP_URL", "").strip()
    if not url:
        url = os.getenv("ONEBOT_WS_URL", "").strip()
    return url.rstrip("/")


def onebot_access_token() -> str:
    """Return the optional OneBot access token."""
    return os.getenv("ONEBOT_ACCESS_TOKEN", "").strip()


def onebot_configured() -> bool:
    """Return True when a OneBot endpoint is configured.

    Used as the ``check_fn`` for OneBot-backed tools so they are gated out
    of the model's schema entirely when no OneBot connection is set up.
    """
    return bool(onebot_base_url())


def _is_ws_url(url: str) -> bool:
    """Return True when *url* is a WebSocket endpoint."""
    return url.lower().startswith(("ws://", "wss://"))


# ---------------------------------------------------------------------------
# Shared response handling
# ---------------------------------------------------------------------------

def _check_onebot_payload(payload: dict, action: str) -> dict:
    """Validate a OneBot response envelope and return its ``data`` object.

    Raises ``RuntimeError`` on a ``failed`` status or a missing ``data``
    field so callers can surface one clear message to the model.
    """
    if not isinstance(payload, dict):
        raise RuntimeError(f"OneBot action '{action}' returned a non-object response.")
    if payload.get("status") == "failed":
        msg = payload.get("message") or payload.get("wording") or "unknown error"
        raise RuntimeError(
            f"OneBot action '{action}' failed: {msg} "
            f"(retcode={payload.get('retcode')})"
        )
    data = payload.get("data")
    if data is None:
        raise RuntimeError(f"OneBot action '{action}' returned no data.")
    return data


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def onebot_call(
    action: str,
    params: dict | None = None,
    *,
    timeout: int = ONEBOT_TIMEOUT,
) -> dict:
    """Invoke a OneBot v11 action and return its ``data`` object.

    Picks the HTTP or WebSocket transport from the configured endpoint's
    URL scheme. Raises ``RuntimeError`` on transport errors, a ``failed``
    status, or a missing ``data`` field.
    """
    base = onebot_base_url()
    if not base:
        raise RuntimeError("ONEBOT_HTTP_URL is not configured.")
    if _is_ws_url(base):
        return _onebot_call_ws(base, action, params or {}, timeout)
    return _onebot_call_http(base, action, params or {}, timeout)


# ---------------------------------------------------------------------------
# HTTP transport
# ---------------------------------------------------------------------------

def _onebot_call_http(base: str, action: str, params: dict, timeout: int) -> dict:
    """Invoke a OneBot action over the HTTP API."""
    url = f"{base}/{action}"
    body = json.dumps(params).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    token = onebot_access_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", errors="replace")[:200]
        except Exception:  # noqa: BLE001 — best-effort detail only
            pass
        raise RuntimeError(
            f"OneBot HTTP {e.code} for action '{action}'. {detail}".strip()
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Cannot reach OneBot at {base} — is NapCat/Lagrange running? ({e.reason})"
        ) from e

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"OneBot action '{action}' returned non-JSON: {raw[:200]}"
        ) from e

    return _check_onebot_payload(payload, action)


# ---------------------------------------------------------------------------
# WebSocket transport
# ---------------------------------------------------------------------------

def _onebot_call_ws(base: str, action: str, params: dict, timeout: int) -> dict:
    """Invoke a OneBot action over the WebSocket API (sync wrapper).

    Each call opens its own short-lived connection — these tools are
    low-frequency, so a fresh connect per action keeps the sync API simple
    and stateless.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No event loop in this thread — safe to drive one directly.
        return asyncio.run(_ws_roundtrip(base, action, params, timeout))

    # Already inside a running loop — isolate the round-trip in a worker
    # thread so we never block or nest event loops.
    import concurrent.futures  # noqa: PLC0415 — only needed on this path

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            lambda: asyncio.run(_ws_roundtrip(base, action, params, timeout))
        )
        return future.result(timeout=timeout + 15)


async def _ws_roundtrip(base: str, action: str, params: dict, timeout: int) -> dict:
    """Open a WS connection, send one action, and return its ``data``.

    NapCat/Lagrange push unsolicited events on the same socket; this reads
    until the reply carrying our ``echo`` arrives, skipping everything else.
    """
    try:
        import websockets  # noqa: PLC0415 — optional dep, imported on demand
    except ImportError as e:
        raise RuntimeError(
            "OneBot WebSocket transport needs the 'websockets' package — "
            "run `pip install websockets`, or point ONEBOT_HTTP_URL at an "
            "http:// endpoint instead."
        ) from e

    uri = base
    token = onebot_access_token()
    if token:
        sep = "&" if "?" in uri else "?"
        uri = f"{uri}{sep}access_token={urllib.parse.quote(token)}"

    echo = f"hermes-{action}-{os.urandom(4).hex()}"
    request = json.dumps({"action": action, "params": params, "echo": echo})

    try:
        async with websockets.connect(uri, max_size=None, open_timeout=timeout) as ws:
            await ws.send(request)
            # Skip pushed events; stop when our echo comes back.
            for _ in range(500):
                raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                try:
                    msg = json.loads(raw)
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
                if isinstance(msg, dict) and msg.get("echo") == echo:
                    return _check_onebot_payload(msg, action)
    except RuntimeError:
        raise
    except (OSError, asyncio.TimeoutError) as e:
        raise RuntimeError(
            f"Cannot reach OneBot at {base} — is NapCat/Lagrange running? ({e})"
        ) from e
    except Exception as e:  # noqa: BLE001 — surface one clear message
        raise RuntimeError(
            f"OneBot WebSocket action '{action}' failed: {e}"
        ) from e

    raise RuntimeError(
        f"OneBot action '{action}' got no matching reply (echo timeout)."
    )
