"""MCP OAuth worker and callback receivers used by connection cards and gateway RPC flows."""

from __future__ import annotations

import http.server
import logging
import os
import secrets
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

URL_TIMEOUT_SECONDS = 30.0


def probe_with_rollback(
        server_name: str, cfg: dict, hermes_home: str, flow, reconnect_live: bool, *,
        on_commit: Optional[Callable[[], None]] = None) -> None:
    """Roll back failures through initialize; commit authorization before tool discovery.

    ``on_commit`` runs right after the configuration is saved: a card install persists its setup
    values there, so they land with the authorization and never before it."""
    from hermes_cli.mcp_config import _oauth_tokens_present, _probe_single_server
    from tools.mcp_dashboard_oauth import exception_message
    from tools.mcp_oauth import HermesTokenStorage, login_connect_timeout
    from tools.mcp_oauth_manager import get_manager
    manager = get_manager()
    storage = HermesTokenStorage(server_name)
    # An attempt that replaced a still-running one starts from that one's half-written files, so
    # it carries the older attempt's snapshot: the state from before either of them.
    backup = getattr(flow, "inherited_backup", None) or storage.snapshot()
    if flow is not None:
        flow.backup = backup
    previous_entry = None
    details: Dict[str, Any] = {}
    tools: list = []
    discovery_error = ""

    def undo() -> None:
        # ``manager.remove`` cleared the pre-attempt tokens, so anything on disk now is this
        # attempt's grant, and it is not being kept: put the snapshot back. A newer attempt for
        # the same server owns the token files; an older one must not write over it.
        if flow is not None and _ACTIVE.get((hermes_home, server_name)) not in (None, flow):
            return
        storage.restore(backup)
        manager.restore_entry(server_name, previous_entry, hermes_home=hermes_home)

    try:
        previous_entry = manager.remove(server_name, hermes_home=hermes_home)
        tools = _probe_single_server(
            server_name, cfg, connect_timeout=login_connect_timeout(cfg), details=details)
        if not _oauth_tokens_present(server_name):
            details["initialized"] = False
            raise RuntimeError(
                "The server responded, but no OAuth token was obtained — "
                "this provider may require a manually-registered OAuth client.")
    except Exception as exc:
        if not details.get("initialized"):
            undo()
            raise
        tools, discovery_error = [], exception_message(exc)
    try:
        _commit(server_name, cfg, on_commit, flow)
    except Exception:
        undo()
        raise
    if flow is not None:
        flow.tools = [{"name": t, "description": d} for t, d in tools]
        flow.discovery_error = discovery_error
        flow.mark_approved()
    if discovery_error:
        return
    if reconnect_live:
        from tools.mcp_tool_loop import reconnect_mcp_server
        reconnect_mcp_server(server_name)


class AttemptCanceled(RuntimeError):
    """The user canceled this attempt before it committed; nothing of it is kept."""


# One lock orders a cancel against the commit, so an attempt is either canceled with nothing kept or
# committed with everything kept. A worker parked inside the token request cannot be interrupted;
# it is stopped here, at the one point where its result would be adopted.
_COMMIT_GUARD = threading.Lock()
# (hermes home, server) -> the newest card attempt. A retry or a new operation replaces an attempt
# whose worker is still waiting on the browser; the older one is canceled so it cannot commit later.
_ACTIVE: Dict[tuple, Any] = {}


def cancel_attempt(flow) -> bool:
    """Cancel a card attempt. True when it had already committed: the authorization stays."""
    with _COMMIT_GUARD:
        if getattr(flow, "committed", False):
            return True
        flow.cancelled = True
    # Wakes a worker that is still waiting for the browser; a cancelled flow is never re-minted.
    flow.mark_error("canceled", cancelled=True)
    return False


def _commit(server_name: str, cfg: dict, on_commit: Optional[Callable[[], None]], flow=None) -> None:
    from hermes_cli.mcp_config import _mcp_server_entry, _restore_mcp_server_entry, _save_mcp_server

    with _COMMIT_GUARD:
        if flow is not None and getattr(flow, "cancelled", False):
            raise AttemptCanceled("canceled")
        previous = _mcp_server_entry(server_name)
        if not _save_mcp_server(server_name, cfg):
            raise RuntimeError(f"'{server_name}' was rejected: suspicious command/args configuration")
        try:
            if on_commit is not None:
                on_commit()
        except Exception:
            _restore_mcp_server_entry(server_name, previous)
            raise
        if flow is not None:
            flow.committed = True


def _reuse_saved_authorization(
        server_name: str, cfg: dict, flow, on_commit: Optional[Callable[[], None]]) -> bool:
    """Connect with the tokens already on disk, with no browser step and no consent.

    Retrying discovery for a server that is authorized must not ask the user to sign in again, and
    must not delete the working grant first. Any failure here falls through to the interactive
    flow, which replaces the grant."""
    from hermes_cli.mcp_config import _oauth_tokens_present, _probe_single_server
    from tools.mcp_oauth import suppress_interactive_oauth

    if not _oauth_tokens_present(server_name):
        return False
    try:
        with suppress_interactive_oauth():
            tools = _probe_single_server(server_name, cfg, connect_timeout=30)
        _commit(server_name, cfg, on_commit, flow)
    except Exception as exc:
        logger.debug("saved authorization for %s was not usable: %s", server_name, exc)
        return False
    if flow is not None:
        flow.tools = [{"name": t, "description": d} for t, d in tools]
        flow.discovery_error = ""
        flow.mark_approved()
    return True


def run_worker(
        hermes_home: str, server_name: str, cfg: dict, reconnect_live: bool, *,
        flow, on_done: Optional[Callable[[], None]] = None,
        env: Optional[Dict[str, str]] = None, on_commit: Optional[Callable[[], None]] = None,
        reuse_saved: bool = False) -> None:
    """Drive the interactive MCP OAuth probe under the shared callback bridge.

    ``env`` holds a card install's setup values. They join the secret scope for this attempt only,
    so the configuration can reference them before anything is saved. ``reuse_saved`` is the
    card's rule: a server whose saved tokens still work connects with no consent step. The RPC
    session surface keeps it off, because its caller waits for an authorization URL."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    try:
        from agent.secret_scope import (
            build_profile_secret_scope, reset_secret_scope, set_secret_scope)
        from tools.mcp_dashboard_oauth import dashboard_oauth_flow
        from tools.mcp_oauth import force_interactive_oauth
        home_token = set_hermes_home_override(hermes_home)
        secret_token = set_secret_scope({**build_profile_secret_scope(Path(hermes_home)), **(env or {})})
        try:
            if not (reuse_saved and flow is not None
                    and _reuse_saved_authorization(server_name, cfg, flow, on_commit)):
                with force_interactive_oauth(), dashboard_oauth_flow(flow):
                    probe_with_rollback(
                        server_name, cfg, hermes_home, flow, reconnect_live, on_commit=on_commit)
        finally:
            reset_secret_scope(secret_token)
            reset_hermes_home_override(home_token)
    except Exception as exc:
        from tools.mcp_dashboard_oauth import exception_message
        msg = exception_message(exc)
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
            with _COMMIT_GUARD:
                if _ACTIVE.get((hermes_home, server_name)) is flow:
                    _ACTIVE.pop((hermes_home, server_name), None)
        if on_done is not None:
            on_done()


def _validate_client_redirect_uri(uri: str) -> str:
    """Accept only plain-http loopback URLs (RFC 8252)."""
    parsed = urlparse(str(uri or "").strip())
    host = (parsed.hostname or "").lower()
    if (parsed.scheme != "http" or host not in ("127.0.0.1", "localhost", "::1") or not parsed.port
            or parsed.username is not None or parsed.password is not None):
        raise ValueError(
            "client_redirect_uri must be a loopback http URL like http://127.0.0.1:<port>/callback")
    return f"http://{'[' + host + ']' if ':' in host else host}:{parsed.port}{parsed.path or '/callback'}"


def _start_loopback_receiver(flow) -> "http.server.HTTPServer":
    """Bind the single backend-hosted one-shot receiver and feed its callback into ``flow``."""
    from tools.mcp_oauth import _parse_redirect_query

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path.rstrip("/") not in ("/callback", ""):
                self.send_response(404)
                self.end_headers()
                return
            body = b"<h1>Authorization received</h1><p>You can close this tab and return to Hermes.</p>"
            status = 200
            try:
                flow.deliver_callback(**_parse_redirect_query(parsed.query))
            except Exception:
                body = b"<h1>OAuth callback rejected</h1><p>The callback was invalid or already used.</p>"
                status = 400
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            with suppress(Exception):
                self.wfile.write(body)

        def log_message(self, format, *args):
            return

    httpd = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(
        target=httpd.serve_forever, kwargs={"poll_interval": 0.5}, daemon=True,
        name=f"mcp-oauth-cb-{flow.server_name}").start()
    return httpd


def _pinned_loopback(cfg: dict) -> bool:
    oauth = cfg.get("oauth") or {}
    return bool(oauth.get("client_id") and oauth.get("redirect_port"))


def _pinned_redirect_uri(cfg: dict) -> str:
    oauth = cfg.get("oauth") or {}
    host = oauth.get("redirect_host") or "127.0.0.1"
    return f"http://{host}:{oauth['redirect_port']}/callback"


def choose_callback_receiver(flow, cfg: dict, client_redirect_uri: Optional[str] = None):
    """Set the redirect consumed by the SDK and return the optional backend HTTP receiver."""
    if _pinned_loopback(cfg):
        flow.redirect_uri = _pinned_redirect_uri(cfg)
        return None
    if client_redirect_uri is not None:
        flow.redirect_uri = _validate_client_redirect_uri(client_redirect_uri)
        return None
    httpd = _start_loopback_receiver(flow)
    flow.redirect_uri = f"http://127.0.0.1:{httpd.server_address[1]}/callback"
    return httpd


def _ssh_detail(redirect_uri: str) -> str:
    if not (os.environ.get("SSH_CLIENT") or os.environ.get("SSH_TTY")) or not redirect_uri:
        return ""
    from tools.mcp_oauth import _SSH_HINT_LOOPBACK
    parsed = urlparse(redirect_uri)
    return _SSH_HINT_LOOPBACK.format(host=parsed.hostname or "127.0.0.1", port=parsed.port or 0).strip()


@dataclass
class OAuthAttempt:
    """One flow in flight: its URL, callback bridge, and outcome."""

    auth_url: str
    flow: Any
    detail: str = ""

    def poll(self) -> Dict[str, Any]:
        snapshot = self.flow.snapshot()
        raw = snapshot.get("status")
        status = raw if raw in ("approved", "error") else "pending"
        tools = [str(t.get("name") or "") for t in (getattr(self.flow, "tools", None) or [])]
        return {"status": status, "error": snapshot.get("error") or "",
                "tools": [t for t in tools if t] if status == "approved" else [],
                "discovery_error": (getattr(self.flow, "discovery_error", "") or "")
                if status == "approved" else ""}


def start(
    server_name: str, *, url_timeout: float = URL_TIMEOUT_SECONDS,
    client_redirect_uri: Optional[str] = None, cfg: Optional[dict] = None,
    env: Optional[Dict[str, str]] = None, on_commit: Optional[Callable[[], None]] = None,
) -> OAuthAttempt:
    """Start a card OAuth flow and wait until its authorization URL is published.

    ``cfg`` is an install's in-memory configuration; without it the saved one is authorized."""
    from hermes_cli.mcp_config import _get_mcp_servers
    from hermes_constants import get_hermes_home
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow
    from tui_gateway import mcp_oauth_sessions

    cfg = dict(cfg if cfg is not None else _get_mcp_servers().get(server_name) or {})
    if not cfg:
        raise RuntimeError(f"'{server_name}' is not a configured MCP server")
    if not cfg.get("url"):
        raise RuntimeError(f"'{server_name}' is a stdio server: it takes env keys, not OAuth")
    cfg["auth"] = "oauth"
    hermes_home = str(get_hermes_home().expanduser().resolve(strict=False))
    flow = DashboardOAuthFlow(
        flow_id=secrets.token_urlsafe(24), server_name=server_name, profile=None,
        hermes_home=hermes_home, redirect_uri="", reconnect_live=False)
    with _COMMIT_GUARD:
        older = _ACTIVE.get((hermes_home, server_name))
        _ACTIVE[(hermes_home, server_name)] = flow
    if older is not None and not older.worker_done:
        flow.inherited_backup = getattr(older, "backup", None)
        cancel_attempt(older)
    httpd = choose_callback_receiver(flow, cfg, client_redirect_uri)
    mcp_oauth_sessions.register_flow(flow, httpd=httpd)
    threading.Thread(
        target=run_worker, args=(hermes_home, server_name, cfg, False),
        kwargs={"flow": flow, "on_done": lambda: mcp_oauth_sessions.finish_flow(flow.flow_id),
                "reuse_saved": True,
                **({"env": env, "on_commit": on_commit} if env or on_commit else {})},
        daemon=True, name=f"mcp-oauth-{server_name}").start()
    deadline = time.time() + url_timeout
    while time.time() < deadline:
        snapshot = flow.snapshot()
        if snapshot.get("status") == "approved":
            return OAuthAttempt(auth_url="", flow=flow)  # the saved authorization still works
        if snapshot.get("authorization_url"):
            return OAuthAttempt(
                auth_url=snapshot["authorization_url"], flow=flow,
                detail=_ssh_detail(flow.redirect_uri) if client_redirect_uri is None else "")
        if snapshot.get("status") == "error":
            raise RuntimeError(snapshot.get("error") or "the OAuth flow failed before authorization")
        time.sleep(0.05)
    flow.mark_error("Timed out waiting for MCP authorization URL")
    raise TimeoutError(f"timed out waiting for the authorization URL for '{server_name}'")
