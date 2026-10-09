"""Session-backed MCP OAuth flows for gateway RPC and connection-card callback relays.

The worker and callback receiver selection live in ``tools/connectors/mcp_oauth.py``. This module
owns flow registration, profile checks, polling, cancellation, and relayed callback delivery.
"""

from __future__ import annotations

import secrets
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, Dict, Optional

from tools.connectors.mcp_oauth import (
    _validate_client_redirect_uri,
    choose_callback_receiver,
)

# session_id -> record wrapping the shared DashboardOAuthFlow bridge plus bookkeeping.
_sessions: dict[str, dict[str, Any]] = {}
_sessions_lock = threading.Lock()

_SESSION_TTL_SECONDS = 900  # completed/abandoned session lingers this long before GC
_MAX_PENDING = 12  # cap in-flight flows so a runaway client can't exhaust ports/threads


def _shutdown_listener(rec: dict[str, Any]) -> None:
    server = rec.get("httpd")
    if server is None:
        return
    for stop in (server.shutdown, server.server_close):
        with suppress(Exception):
            stop()
    rec["httpd"] = None


def register_flow(flow, *, httpd=None) -> dict[str, Any]:
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


def _probe_with_rollback(
    server_name: str, cfg: dict, hermes_home: str, flow, reconnect_live: bool,
) -> None:
    """Commit new OAuth tokens only after a complete authorization probe."""
    from hermes_cli.mcp_config import _probe_single_server, _save_mcp_server
    from tools.mcp_oauth import HermesTokenStorage, login_connect_timeout, oauth_reauth_staging
    from tools.mcp_oauth_manager import get_manager

    manager = get_manager()
    storage = HermesTokenStorage(server_name, hermes_home=hermes_home)
    original_snapshot = storage.snapshot()
    with oauth_reauth_staging(server_name, hermes_home=hermes_home) as staged_storage:
        previous_entry = manager.evict(server_name, hermes_home=hermes_home)
        manager.set_entry_persistence_suspended(previous_entry, True)
        try:
            tools = _probe_single_server(
                server_name, cfg, connect_timeout=login_connect_timeout(cfg)
            )
            if not staged_storage.has_cached_tokens():
                raise RuntimeError(
                    "The server responded, but no OAuth token was obtained — "
                    "this provider may require a manually-registered OAuth client.")
            storage.restore(staged_storage.snapshot())
            manager.evict(server_name, hermes_home=hermes_home)
            _save_mcp_server(server_name, cfg)
            if flow is not None:
                flow.tools = [{"name": tool, "description": description} for tool, description in tools]
                flow.mark_approved()
            if reconnect_live:
                from tools.mcp_tool_loop import reconnect_mcp_server
                reconnect_mcp_server(server_name)
        except Exception:
            storage.restore(original_snapshot)
            manager.evict(server_name, hermes_home=hermes_home)
            manager.set_entry_persistence_suspended(previous_entry, False)
            manager.restore_entry(server_name, previous_entry, hermes_home=hermes_home)
            raise


def _worker(
    session_id: str, hermes_home: str, server_name: str, cfg: dict, reconnect_live: bool,
) -> None:
    """Run TUI OAuth under the owning profile and staged-token transaction."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    rec = _sessions.get(session_id)
    flow = rec["flow"] if rec else None
    home_token = secret_token = None
    try:
        from agent.secret_scope import (
            build_profile_secret_scope, reset_secret_scope, set_secret_scope,
        )
        from tools.mcp_dashboard_oauth import dashboard_oauth_flow
        from tools.mcp_oauth import (
            force_interactive_oauth, oauth_reauth_transaction,
        )

        home_token = set_hermes_home_override(hermes_home)
        secret_token = set_secret_scope(
            build_profile_secret_scope(Path(hermes_home)), profile_home=hermes_home,
        )
        with oauth_reauth_transaction(server_name, hermes_home=hermes_home), \
                force_interactive_oauth(), dashboard_oauth_flow(flow):
            _probe_with_rollback(server_name, cfg, hermes_home, flow, reconnect_live)
    except Exception as exc:
        from tools.mcp_dashboard_oauth import exception_message
        msg = exception_message(exc)
        with suppress(Exception):
            from tools.mcp_oauth import humanize_oauth_registration_error
            msg = humanize_oauth_registration_error(
                server_name, exc, server_url=cfg.get("url") if isinstance(cfg, dict) else None,
            ) or msg
        if flow is not None:
            flow.mark_error(msg)
    finally:
        if secret_token is not None:
            reset_secret_scope(secret_token)
        if home_token is not None:
            reset_hermes_home_override(home_token)
        if flow is not None:
            flow.mark_worker_done()
        if rec is not None:
            _shutdown_listener(rec)


def finish_flow(session_id: str) -> None:
    """Release a finished flow's backend listener without removing relay-visible outcome state."""
    with _sessions_lock:
        rec = _sessions.get(session_id)
    if rec is not None:
        _shutdown_listener(rec)


def start_flow(
    hermes_home: str, server_name: str, cfg: dict, *, reconnect_live: bool = False,
    url_timeout: float = 30.0, client_redirect_uri: Optional[str] = None) -> dict[str, Any]:
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
        target=_worker, args=(session_id, hermes_home, server_name, dict(cfg), reconnect_live),
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
) -> "tuple[dict[str, Any] | None, str | None]":
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


def poll_flow(session_id: str, server_name: str) -> dict[str, Any]:
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
    out: dict[str, Any] = {
        "session_id": session_id, "status": status, "error_message": snap.get("error"),
        "auth_url": snap.get("authorization_url")}
    if status == "approved":
        out["tools"] = list(getattr(flow, "tools", []) or [])
    return out


def cancel_flow(session_id: str, server_name: str, hermes_home: str) -> dict[str, Any]:
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
    error: Optional[str] = None, iss: Optional[str] = None) -> dict[str, Any]:
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
