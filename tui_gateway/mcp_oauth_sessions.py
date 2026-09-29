"""Session-backed MCP OAuth flows for gateway RPC and connection-card callback relays.

The worker and callback receiver selection live in ``tools/connectors/mcp_oauth.py``. This module
owns flow registration, profile checks, polling, cancellation, and relayed callback delivery.
"""

from __future__ import annotations

import secrets
import threading
import time
from contextlib import suppress
from typing import Any, Dict, Optional

from tools.connectors.mcp_oauth import (
    _validate_client_redirect_uri,
    choose_callback_receiver,
    run_worker,
)

# session_id -> record wrapping the shared DashboardOAuthFlow bridge plus bookkeeping.
_sessions: Dict[str, Dict[str, Any]] = {}
_sessions_lock = threading.Lock()

_SESSION_TTL_SECONDS = 900  # completed/abandoned session lingers this long before GC
_MAX_PENDING = 12  # cap in-flight flows so a runaway client can't exhaust ports/threads


def _shutdown_listener(rec: Dict[str, Any]) -> None:
    server = rec.get("httpd")
    if server is None:
        return
    for stop in (server.shutdown, server.server_close):
        with suppress(Exception):
            stop()
    rec["httpd"] = None


def register_flow(flow, *, httpd=None) -> Dict[str, Any]:
    """Register a callback-relay flow so both RPC and card starts share ownership checks."""
    rec = {
        "session_id": flow.flow_id,
        "server_name": flow.server_name,
        "hermes_home": flow.hermes_home,
        "flow": flow,
        "httpd": httpd,
        "created_at": time.time(),
    }
    with _sessions_lock:
        _sessions[flow.flow_id] = rec
    return rec


def finish_flow(session_id: str) -> None:
    """Release a finished flow's backend listener without removing relay-visible outcome state."""
    with _sessions_lock:
        rec = _sessions.get(session_id)
    if rec is not None:
        _shutdown_listener(rec)


def start_flow(
    hermes_home: str, server_name: str, cfg: dict, *, reconnect_live: bool = False,
    url_timeout: float = 30.0, client_redirect_uri: Optional[str] = None) -> Dict[str, Any]:
    """Begin an MCP OAuth flow and return ``{session_id, auth_url, flow}``; blocks up to
    ``url_timeout`` for the authorization URL. With ``client_redirect_uri`` (invalid values
    raise ``ValueError``) no gateway-side listener is bound."""
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow
    if client_redirect_uri is not None:
        client_redirect_uri = _validate_client_redirect_uri(client_redirect_uri)
    cutoff = time.time() - _SESSION_TTL_SECONDS  # opportunistic GC of expired sessions
    with _sessions_lock:
        for sid in [sid for sid, rec in _sessions.items() if rec["created_at"] < cutoff]:
            _shutdown_listener(_sessions.pop(sid))
    with _sessions_lock:
        active = [r for r in _sessions.values() if not r["flow"].worker_done]
        if len(active) >= _MAX_PENDING:
            raise RuntimeError("Too many MCP OAuth flows are already in progress")
        if any(r["server_name"] == server_name and r["hermes_home"] == hermes_home for r in active):
            raise RuntimeError(f"MCP OAuth for '{server_name}' is already in progress")

    session_id = secrets.token_urlsafe(24)
    flow = DashboardOAuthFlow(
        flow_id=session_id, server_name=server_name, profile=None, hermes_home=hermes_home,
        redirect_uri="", reconnect_live=reconnect_live)
    httpd = choose_callback_receiver(flow, cfg, client_redirect_uri)
    rec = register_flow(flow, httpd=httpd)
    threading.Thread(
        target=run_worker, args=(hermes_home, server_name, dict(cfg), reconnect_live),
        kwargs={"flow": flow, "on_done": lambda: _shutdown_listener(rec)},
        daemon=True, name=f"mcp-oauth-{server_name}").start()
    try:
        auth_url = None
        # wait_for_authorization_url is async; run its wait synchronously.
        deadline = time.time() + url_timeout
        while time.time() < deadline:
            snap = flow.snapshot()
            if auth_url := snap.get("authorization_url"):
                break
            if snap.get("status") == "error":
                raise RuntimeError(
                    snap.get("error") or "MCP OAuth flow failed before authorization")
            time.sleep(0.1)
        if not auth_url:
            raise TimeoutError("Timed out waiting for MCP authorization URL")
    except Exception as exc:
        from tools.mcp_dashboard_oauth import exception_message
        flow.mark_error(exception_message(exc))  # no-op when the worker already recorded the cause
        _shutdown_listener(rec)
        raise
    # ``flow`` mirrors the provider-OAuth discriminator: open a URL then poll (no user_code).
    return {"session_id": session_id, "auth_url": auth_url, "flow": "pkce"}


def _lookup(
    session_id: str, server_name: str, hermes_home: Optional[str] = None,
) -> "tuple[Dict[str, Any] | None, str | None]":
    """Find a session belonging to the caller's resolved profile."""
    from hermes_constants import hermes_home_key
    with _sessions_lock:
        rec = _sessions.get(session_id)
    if rec is None:
        return None, "OAuth session not found or expired"
    if rec["server_name"] != server_name:
        return None, "server name mismatch for session"
    if hermes_home_key(rec["hermes_home"]) != hermes_home_key(hermes_home):
        return None, "profile mismatch for session"
    return rec, None


def poll_flow(session_id: str, server_name: str) -> Dict[str, Any]:
    """Poll a session → ``{status, error_message?, auth_url?, tools?}``; ``status`` is
    ``pending`` | ``approved`` | ``error`` (the bridge's ``authorization_required`` maps to
    ``pending``)."""
    rec, err = _lookup(session_id, server_name)
    if rec is None:
        return {"status": "error", "error_message": err}
    flow = rec["flow"]
    snap = flow.snapshot()
    raw = snap.get("status")
    status = raw if raw in ("approved", "error") else "pending"
    out: Dict[str, Any] = {
        "session_id": session_id, "status": status, "error_message": snap.get("error"),
        "auth_url": snap.get("authorization_url")}
    if status == "approved":
        out["tools"] = list(getattr(flow, "tools", []) or [])
    return out


def cancel_flow(session_id: str, server_name: str, hermes_home: str) -> Dict[str, Any]:
    """Cancel only the owning profile's flow and release its callback waiter."""
    rec, err = _lookup(session_id, server_name, hermes_home)
    if rec is None:
        return {"ok": False, "error_message": err}
    flow = rec["flow"]
    flow.mark_error("OAuth cancelled by user", cancelled=True)
    _shutdown_listener(rec)
    return {"ok": True, "status": flow.snapshot()["status"]}


def deliver_callback_flow(
    session_id: str, server_name: str, *, code: Optional[str], state: Optional[str],
    error: Optional[str] = None, iss: Optional[str] = None) -> Dict[str, Any]:
    """Relay a client-captured OAuth redirect into a session's flow (remote-backend companion
    to ``start_flow(client_redirect_uri=...)``); ``deliver_callback`` still verifies ``state``
    and rejects replays. Returns ``{ok: true}`` or ``{ok: false, error_message}``."""
    rec, err = _lookup(session_id, server_name)
    if rec is None:
        return {"ok": False, "error_message": err}
    try:
        rec["flow"].deliver_callback(code=code, state=state, error=error, iss=iss)
    except ValueError as exc:
        return {"ok": False, "error_message": str(exc)}
    return {"ok": True, "session_id": session_id}
