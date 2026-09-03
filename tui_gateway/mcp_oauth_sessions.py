"""Session-backed MCP OAuth flows for the gateway (mcp.servers.oauth.*).

This mirrors the *provider* OAuth model used by the dashboard
(``/api/providers/oauth/{id}/start`` + ``/poll/{session_id}``) rather than the
FastAPI-request-coupled MCP dashboard flow: a ``start`` primitive kicks off a
background worker and returns ``{session_id, auth_url, flow}``; a ``poll``
primitive reports ``{status: pending|approved|error}`` until the tokens land on
disk for that server in that profile.

The underlying token machinery is the *same* one the CLI ``hermes mcp login``
uses — ``hermes_cli.mcp_config._probe_single_server`` under
``tools.mcp_oauth.force_interactive_oauth`` — so no OAuth logic is reimplemented
here. The only new piece is decoupling the two browser callbacks (authorization
URL out, ``code``/``state`` back in) from a FastAPI ``Request``:

* ``tools.mcp_dashboard_oauth.DashboardOAuthFlow`` already provides the two
  thread-safe rendezvous points (``publish_authorization_url`` /
  ``deliver_callback``). We reuse it verbatim as the bridge object.
* Instead of routing the browser redirect through a FastAPI callback route, we
  run a tiny loopback HTTP listener on ``127.0.0.1:<port>/callback`` and set the
  flow's ``redirect_uri`` to it. When the provider redirects the user's browser
  there, the listener calls ``flow.deliver_callback(...)``. This is the same
  loopback strategy the CLI uses by default, just wired to the shared bridge.

Client contract (what the desktop plugin does):
  1. call ``mcp.servers.oauth.start(profile, name)`` → ``{session_id, auth_url}``
  2. open ``auth_url`` in the native browser (``openExternal``)
  3. poll ``mcp.servers.oauth.poll(profile, name, session_id)`` until
     ``status == "approved"`` (tokens persisted) or ``"error"``.

Remote-backend variant (client-side callback): when the desktop app runs on a
DIFFERENT machine than the gateway (SSH/Tailscale remote backend), the
gateway-side ``127.0.0.1`` listener is unreachable from the user's browser —
the redirect lands on the user's machine where nothing is listening, and the
flow times out. For that topology the client binds its OWN loopback listener
(same pattern as the desktop's native gateway login), passes its
``redirect_uri`` to ``start`` (``client_redirect_uri``), and relays the
provider redirect back via ``deliver_callback_flow`` /
``mcp.servers.oauth.callback``. State verification stays server-side in
``DashboardOAuthFlow.deliver_callback`` — a relayed code with the wrong
``state`` is rejected exactly like a forged loopback hit.
"""

from __future__ import annotations

import http.server
import secrets
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

# Session registry: session_id -> record. A record wraps the shared
# DashboardOAuthFlow bridge plus a bit of gateway bookkeeping.
_sessions: Dict[str, Dict[str, Any]] = {}
_sessions_lock = threading.Lock()

# How long a completed/abandoned session lingers before GC (seconds).
_SESSION_TTL_SECONDS = 900
# Cap concurrent in-flight flows so a runaway client can't exhaust ports/threads.
_MAX_PENDING = 12


def _gc_sessions() -> None:
    """Drop expired sessions. Called opportunistically on start."""
    cutoff = time.time() - _SESSION_TTL_SECONDS
    with _sessions_lock:
        stale = [sid for sid, rec in _sessions.items() if rec["created_at"] < cutoff]
        for sid in stale:
            rec = _sessions.pop(sid, None)
            if rec is not None:
                _shutdown_listener(rec)


def _shutdown_listener(rec: Dict[str, Any]) -> None:
    server = rec.get("httpd")
    if server is not None:
        try:
            server.shutdown()
        except Exception:
            pass
        try:
            server.server_close()
        except Exception:
            pass
        rec["httpd"] = None


def _validate_client_redirect_uri(uri: str) -> str:
    """Validate a client-supplied loopback redirect URI.

    Only plain-http loopback URLs are accepted (``http://127.0.0.1:<port>/...``
    or ``http://localhost:<port>/...``), mirroring RFC 8252 native-app rules —
    the client hosts a one-shot listener on ITS machine, so anything else
    (public hosts, https proxies, schemes) is rejected to keep the gateway from
    pinning an attacker-controlled redirect into a DCR registration.
    """
    parsed = urlparse(str(uri or "").strip())
    host = (parsed.hostname or "").lower()
    if (
        parsed.scheme != "http"
        or host not in ("127.0.0.1", "localhost", "::1")
        or not parsed.port
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError(
            "client_redirect_uri must be a loopback http URL like "
            "http://127.0.0.1:<port>/callback"
        )
    return f"http://{'[' + host + ']' if ':' in host else host}:{parsed.port}{parsed.path or '/callback'}"


def _start_loopback_listener(flow) -> "http.server.HTTPServer":
    """Bind a loopback callback listener that feeds the flow's deliver_callback.

    Returns the running HTTPServer (already serving on a daemon thread). The
    bound port is read back off ``server.server_address`` so the caller can set
    ``flow.redirect_uri`` to the matching ``/callback`` URL BEFORE the worker
    starts the OAuth flow (the redirect URI must be pinned at authorization).
    """

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 — stdlib naming
            parsed = urlparse(self.path)
            if parsed.path.rstrip("/") not in ("/callback", ""):
                self.send_response(404)
                self.end_headers()
                return
            qs = parse_qs(parsed.query)
            code = (qs.get("code") or [None])[0]
            state = (qs.get("state") or [None])[0]
            error = (qs.get("error") or [None])[0]
            body = b"<h1>Authorization received</h1><p>You can close this tab and return to Hermes.</p>"
            status = 200
            try:
                flow.deliver_callback(code=code, state=state, error=error)
            except Exception:
                body = b"<h1>OAuth callback rejected</h1><p>The callback was invalid or already used.</p>"
                status = 400
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            try:
                self.wfile.write(body)
            except Exception:
                pass

        def log_message(self, *_a):  # silence stdlib request logging
            return

    httpd = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(
        target=httpd.serve_forever,
        kwargs={"poll_interval": 0.5},
        daemon=True,
        name=f"mcp-oauth-cb-{flow.server_name}",
    ).start()
    return httpd


def _worker(session_id: str, hermes_home: str, server_name: str, cfg: dict, reconnect_live: bool) -> None:
    """Drive the interactive MCP OAuth probe under the shared dashboard bridge.

    Structurally identical to ``web_server._run_dashboard_mcp_oauth`` — the same
    HERMES_HOME override + secret-scope + force_interactive_oauth +
    dashboard_oauth_flow wrapping around ``_probe_single_server`` — but keyed to
    our session record instead of a FastAPI request. On success the token file
    exists on disk (verified via ``_oauth_tokens_present``) and the server config
    is (re)saved into the profile's config.yaml.
    """
    from hermes_cli.mcp_config import (
        _oauth_tokens_present,
        _probe_single_server,
        _save_mcp_server,
    )
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    rec = _sessions.get(session_id)
    flow = rec["flow"] if rec else None
    try:
        from agent.secret_scope import (
            build_profile_secret_scope,
            reset_secret_scope,
            set_secret_scope,
        )
        from tools.mcp_dashboard_oauth import dashboard_oauth_flow
        from tools.mcp_oauth import force_interactive_oauth
        from tools.mcp_oauth_manager import get_manager

        home_token = set_hermes_home_override(hermes_home)
        secret_token = set_secret_scope(build_profile_secret_scope(Path(hermes_home)))
        try:
            with force_interactive_oauth(), dashboard_oauth_flow(flow):
                from tools.mcp_oauth import HermesTokenStorage

                manager = get_manager()
                storage = HermesTokenStorage(server_name)
                backup = storage.snapshot()
                previous_entry = None
                try:
                    previous_entry = manager.remove(server_name, hermes_home=hermes_home)
                    tools = _probe_single_server(
                        server_name,
                        cfg,
                        connect_timeout=max(float(cfg.get("connect_timeout", 0) or 0), 315),
                    )
                    if not _oauth_tokens_present(server_name):
                        raise RuntimeError(
                            "The server responded, but no OAuth token was obtained — "
                            "this provider may require a manually-registered OAuth client."
                        )
                    _save_mcp_server(server_name, cfg)
                    if flow is not None:
                        flow.tools = [{"name": t, "description": d} for t, d in tools]
                        flow.mark_approved()
                    if reconnect_live:
                        from tools.mcp_tool import reconnect_mcp_server

                        reconnect_mcp_server(server_name)
                except Exception:
                    storage.restore(backup, only_if_absent=True)
                    manager.restore_entry(server_name, previous_entry, hermes_home=hermes_home)
                    raise
        finally:
            reset_secret_scope(secret_token)
            reset_hermes_home_override(home_token)
    except Exception as exc:
        msg = str(exc)
        try:
            from tools.mcp_oauth import humanize_oauth_registration_error

            humanized = humanize_oauth_registration_error(
                server_name, exc, server_url=cfg.get("url") if isinstance(cfg, dict) else None
            )
            if humanized:
                msg = humanized
        except Exception:
            pass
        if flow is not None:
            flow.mark_error(msg)
    finally:
        if flow is not None:
            flow.mark_worker_done()
        if rec is not None:
            _shutdown_listener(rec)


def start_flow(
    hermes_home: str,
    server_name: str,
    cfg: dict,
    *,
    reconnect_live: bool = False,
    url_timeout: float = 30.0,
    client_redirect_uri: Optional[str] = None,
) -> Dict[str, Any]:
    """Begin an MCP OAuth flow and return ``{session_id, auth_url, flow}``.

    ``cfg`` is the server's resolved config dict (must have ``url`` and be
    OAuth-capable). ``hermes_home`` is the already-resolved profile home dir
    string. Blocks up to ``url_timeout`` for the worker to publish the browser
    authorization URL, then returns it.

    ``client_redirect_uri`` (remote-backend variant): a loopback callback URL
    the CLIENT hosts on its own machine. When set (and valid), no gateway-side
    listener is bound — the OAuth ``redirect_uri`` is pinned to the client's
    listener, and the client relays the redirect's ``code``/``state`` back via
    ``deliver_callback_flow``. Invalid values raise ``ValueError``.
    """
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow

    if client_redirect_uri is not None:
        client_redirect_uri = _validate_client_redirect_uri(client_redirect_uri)

    _gc_sessions()

    with _sessions_lock:
        pending = sum(
            1
            for r in _sessions.values()
            if not r["flow"].worker_done
        )
        if pending >= _MAX_PENDING:
            raise RuntimeError("Too many MCP OAuth flows are already in progress")
        if any(
            r["server_name"] == server_name
            and r["hermes_home"] == hermes_home
            and not r["flow"].worker_done
            for r in _sessions.values()
        ):
            raise RuntimeError(f"MCP OAuth for '{server_name}' is already in progress")

    session_id = secrets.token_urlsafe(24)
    flow = DashboardOAuthFlow(
        flow_id=session_id,
        server_name=server_name,
        profile=None,
        hermes_home=hermes_home,
        redirect_uri="",  # set below once the loopback port is known
        reconnect_live=reconnect_live,
    )
    if client_redirect_uri:
        # Remote-backend variant: the CLIENT hosts the callback listener on its
        # own machine and relays the code via deliver_callback_flow(). No
        # gateway-side listener is bound — a 127.0.0.1 port here would be
        # unreachable from the user's browser anyway.
        httpd = None
        flow.redirect_uri = client_redirect_uri
    else:
        httpd = _start_loopback_listener(flow)
        port = httpd.server_address[1]
        flow.redirect_uri = f"http://127.0.0.1:{port}/callback"

    rec = {
        "session_id": session_id,
        "server_name": server_name,
        "hermes_home": hermes_home,
        "flow": flow,
        "httpd": httpd,
        "created_at": time.time(),
    }
    with _sessions_lock:
        _sessions[session_id] = rec

    threading.Thread(
        target=_worker,
        args=(session_id, hermes_home, server_name, dict(cfg), reconnect_live),
        daemon=True,
        name=f"mcp-oauth-{server_name}",
    ).start()

    try:
        auth_url = None
        # wait_for_authorization_url is async; run its wait synchronously.
        deadline = time.time() + url_timeout
        while time.time() < deadline:
            snap = flow.snapshot()
            if snap.get("authorization_url"):
                auth_url = snap["authorization_url"]
                break
            if snap.get("status") == "error":
                raise RuntimeError(snap.get("error") or "MCP OAuth flow failed before authorization")
            time.sleep(0.1)
        if not auth_url:
            raise TimeoutError("Timed out waiting for MCP authorization URL")
    except Exception:
        flow.mark_error("Timed out waiting for MCP authorization URL")
        _shutdown_listener(rec)
        raise

    return {
        "session_id": session_id,
        "auth_url": auth_url,
        # "pkce" mirrors the provider-OAuth ``flow`` discriminator: the client
        # opens a URL then polls (no user_code to type, unlike device_code).
        "flow": "pkce",
    }


def poll_flow(session_id: str, server_name: str) -> Dict[str, Any]:
    """Poll a session's status → ``{status, error_message?, auth_url?, tools?}``.

    ``status`` is one of ``pending`` | ``approved`` | ``error`` — the same
    vocabulary as the provider poll endpoint (``authorization_required`` from
    the underlying bridge maps to ``pending`` since the client only needs to
    know whether to keep waiting).
    """
    with _sessions_lock:
        rec = _sessions.get(session_id)
    if rec is None:
        return {"status": "error", "error_message": "OAuth session not found or expired"}
    if rec["server_name"] != server_name:
        return {"status": "error", "error_message": "server name mismatch for session"}

    flow = rec["flow"]
    snap = flow.snapshot()
    raw = snap.get("status")
    if raw == "approved":
        status = "approved"
    elif raw == "error":
        status = "error"
    else:
        status = "pending"
    out: Dict[str, Any] = {
        "session_id": session_id,
        "status": status,
        "error_message": snap.get("error"),
        "auth_url": snap.get("authorization_url"),
    }
    if status == "approved":
        out["tools"] = list(getattr(flow, "tools", []) or [])
    return out


def deliver_callback_flow(
    session_id: str,
    server_name: str,
    *,
    code: Optional[str],
    state: Optional[str],
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """Relay a client-captured OAuth redirect into a session's flow.

    Remote-backend companion to ``start_flow(client_redirect_uri=...)``: the
    desktop app's loopback listener caught the provider redirect on the USER'S
    machine and forwards ``code``/``state`` (or ``error``) here. Security
    properties are unchanged from the gateway-listener path — the underlying
    ``DashboardOAuthFlow.deliver_callback`` verifies ``state`` against the
    pinned authorization request (constant-time compare) and rejects replays,
    so a forged or replayed relay fails identically to a forged loopback hit.

    Returns ``{ok: true}`` on acceptance or ``{ok: false, error_message}``.
    """
    with _sessions_lock:
        rec = _sessions.get(session_id)
    if rec is None:
        return {"ok": False, "error_message": "OAuth session not found or expired"}
    if rec["server_name"] != server_name:
        return {"ok": False, "error_message": "server name mismatch for session"}

    flow = rec["flow"]
    try:
        flow.deliver_callback(code=code, state=state, error=error)
    except ValueError as exc:
        return {"ok": False, "error_message": str(exc)}
    return {"ok": True, "session_id": session_id}
