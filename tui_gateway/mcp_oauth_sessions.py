"""Session-backed MCP OAuth flows for the gateway (mcp.servers.oauth.*): ``start`` spawns a
worker and returns ``{session_id, auth_url, flow}``; ``poll`` reports ``{status}`` until tokens
land. Reuses ``hermes mcp login``'s probe under ``force_interactive_oauth`` plus
``DashboardOAuthFlow``; the only new piece is a loopback listener feeding ``deliver_callback``.
Remote backends host the listener (``client_redirect_uri``) and relay via
``deliver_callback_flow``."""

from __future__ import annotations

import http.server
import secrets
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

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


def _validate_client_redirect_uri(uri: str) -> str:
    """Accept only plain-http loopback URLs (RFC 8252) so the gateway can't pin an
    attacker-controlled redirect into a DCR registration."""
    parsed = urlparse(str(uri or "").strip())
    host = (parsed.hostname or "").lower()
    if (parsed.scheme != "http" or host not in ("127.0.0.1", "localhost", "::1") or not parsed.port
            or parsed.username is not None or parsed.password is not None):
        raise ValueError(
            "client_redirect_uri must be a loopback http URL like http://127.0.0.1:<port>/callback")
    return f"http://{'[' + host + ']' if ':' in host else host}:{parsed.port}{parsed.path or '/callback'}"


def _start_loopback_listener(flow) -> "http.server.HTTPServer":
    """Bind a loopback callback listener feeding ``flow.deliver_callback``; returns the
    HTTPServer already serving on a daemon thread (caller pins ``flow.redirect_uri`` from it)."""
    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 — stdlib naming
            parsed = urlparse(self.path)
            if parsed.path.rstrip("/") not in ("/callback", ""):
                self.send_response(404)
                self.end_headers()
                return
            qs = parse_qs(parsed.query)
            body = b"<h1>Authorization received</h1><p>You can close this tab and return to Hermes.</p>"
            status = 200
            try:
                flow.deliver_callback(
                    **{k: (qs.get(k) or [None])[0] for k in ("code", "state", "error")})
            except Exception:
                body = b"<h1>OAuth callback rejected</h1><p>The callback was invalid or already used.</p>"
                status = 400
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            with suppress(Exception):
                self.wfile.write(body)

        def log_message(self, *_a):  # silence stdlib request logging
            return

    httpd = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(
        target=httpd.serve_forever, kwargs={"poll_interval": 0.5}, daemon=True,
        name=f"mcp-oauth-cb-{flow.server_name}").start()
    return httpd


def _probe_with_rollback(
    server_name: str, cfg: dict, hermes_home: str, flow, reconnect_live: bool) -> None:
    """Run the OAuth probe; on ANY failure restore the prior token file + manager entry."""
    from hermes_cli.mcp_config import _oauth_tokens_present, _probe_single_server, _save_mcp_server
    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_manager import get_manager
    manager = get_manager()
    storage = HermesTokenStorage(server_name)
    backup = storage.snapshot()
    previous_entry = None
    try:
        previous_entry = manager.remove(server_name, hermes_home=hermes_home)
        timeout = max(float(cfg.get("connect_timeout", 0) or 0), 315)
        tools = _probe_single_server(server_name, cfg, connect_timeout=timeout)
        if not _oauth_tokens_present(server_name):
            raise RuntimeError(
                "The server responded, but no OAuth token was obtained — "
                "this provider may require a manually-registered OAuth client.")
        _save_mcp_server(server_name, cfg)
        if flow is not None:
            flow.tools = [{"name": t, "description": d} for t, d in tools]
            flow.mark_approved()
        if reconnect_live:
            from tools.mcp_tool_loop import reconnect_mcp_server
            reconnect_mcp_server(server_name)
    except Exception:
        storage.restore(backup, only_if_absent=True)
        manager.restore_entry(server_name, previous_entry, hermes_home=hermes_home)
        raise


def _worker(
        session_id: str, hermes_home: str, server_name: str, cfg: dict, reconnect_live: bool) -> None:
    """Drive the interactive MCP OAuth probe under the shared dashboard bridge (same wrapping
    as ``web_server._run_dashboard_mcp_oauth``), keyed to our session record."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    rec = _sessions.get(session_id)
    flow = rec["flow"] if rec else None
    try:
        from agent.secret_scope import (
            build_profile_secret_scope, reset_secret_scope, set_secret_scope)
        from tools.mcp_dashboard_oauth import dashboard_oauth_flow
        from tools.mcp_oauth import force_interactive_oauth
        home_token = set_hermes_home_override(hermes_home)
        secret_token = set_secret_scope(build_profile_secret_scope(Path(hermes_home)))
        try:
            with force_interactive_oauth(), dashboard_oauth_flow(flow):
                _probe_with_rollback(server_name, cfg, hermes_home, flow, reconnect_live)
        finally:
            reset_secret_scope(secret_token)
            reset_hermes_home_override(home_token)
    except Exception as exc:
        msg = str(exc)
        with suppress(Exception):
            from tools.mcp_oauth import humanize_oauth_registration_error
            msg = humanize_oauth_registration_error(
                server_name, exc, server_url=cfg.get("url") if isinstance(cfg, dict) else None
            ) or msg
        if flow is not None:
            flow.mark_error(msg)
    finally:
        if flow is not None:
            flow.mark_worker_done()
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
        redirect_uri="",  # set below once the loopback port is known
        reconnect_live=reconnect_live)
    # Client-hosted listener: a 127.0.0.1 port here would be unreachable from the browser.
    httpd = None if client_redirect_uri else _start_loopback_listener(flow)
    flow.redirect_uri = (
        client_redirect_uri or f"http://127.0.0.1:{httpd.server_address[1]}/callback")
    rec = {
        "session_id": session_id, "server_name": server_name, "hermes_home": hermes_home,
        "flow": flow, "httpd": httpd, "created_at": time.time()}
    with _sessions_lock:
        _sessions[session_id] = rec
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
    except Exception:
        flow.mark_error("Timed out waiting for MCP authorization URL")
        _shutdown_listener(rec)
        raise
    # ``flow`` mirrors the provider-OAuth discriminator: open a URL then poll (no user_code).
    return {"session_id": session_id, "auth_url": auth_url, "flow": "pkce"}


def _lookup(session_id: str, server_name: str) -> "tuple[Dict[str, Any] | None, str | None]":
    """Find a session record; returns ``(rec, None)`` or ``(None, error_message)``."""
    with _sessions_lock:
        rec = _sessions.get(session_id)
    if rec is None:
        return None, "OAuth session not found or expired"
    if rec["server_name"] != server_name:
        return None, "server name mismatch for session"
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


def deliver_callback_flow(
    session_id: str, server_name: str, *, code: Optional[str], state: Optional[str],
    error: Optional[str] = None) -> Dict[str, Any]:
    """Relay a client-captured OAuth redirect into a session's flow (remote-backend companion
    to ``start_flow(client_redirect_uri=...)``); ``deliver_callback`` still verifies ``state``
    and rejects replays. Returns ``{ok: true}`` or ``{ok: false, error_message}``."""
    rec, err = _lookup(session_id, server_name)
    if rec is None:
        return {"ok": False, "error_message": err}
    try:
        rec["flow"].deliver_callback(code=code, state=state, error=error)
    except ValueError as exc:
        return {"ok": False, "error_message": str(exc)}
    return {"ok": True, "session_id": session_id}
