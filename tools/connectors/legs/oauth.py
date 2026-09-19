"""The MCP OAuth worker and the in-process flow ``manage_connections`` starts for an ``authorize``
target.

The worker moved here from ``tui_gateway/mcp_oauth_sessions.py``, which still owns the RPC session
table the Capabilities tab polls and now calls this module. A connection operation needs the same
probe without that session bookkeeping, and it sends the browser to the backend's own callback
route (``/api/mcp/oauth/callback/{server}``) so no renderer has to host a listener.
"""

from __future__ import annotations

import logging
import secrets
import sys
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

URL_TIMEOUT_SECONDS = 30.0
# Loopback hosts a bind may report that a browser cannot dial back.
_WILDCARD_HOSTS = frozenset({"", "0.0.0.0", "::"})


def probe_with_rollback(
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


def run_worker(
        hermes_home: str, server_name: str, cfg: dict, reconnect_live: bool, *,
        flow, on_done: Optional[Callable[[], None]] = None) -> None:
    """Drive the interactive MCP OAuth probe under the shared dashboard bridge (same wrapping
    as ``web_server._run_dashboard_mcp_oauth``), reporting into ``flow``."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    try:
        from agent.secret_scope import (
            build_profile_secret_scope, reset_secret_scope, set_secret_scope)
        from tools.mcp_dashboard_oauth import dashboard_oauth_flow
        from tools.mcp_oauth import force_interactive_oauth
        home_token = set_hermes_home_override(hermes_home)
        secret_token = set_secret_scope(build_profile_secret_scope(Path(hermes_home)))
        try:
            with force_interactive_oauth(), dashboard_oauth_flow(flow):
                probe_with_rollback(server_name, cfg, hermes_home, flow, reconnect_live)
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
        if on_done is not None:
            on_done()


def callback_url(server_name: str) -> str:
    """The externally reachable ``/api/mcp/oauth/callback/{server}`` URL, built from the operator's
    public URL or from the address this process bound. Same rule as the dashboard route, which has
    a Request to read and we do not."""
    from urllib.parse import quote

    from hermes_cli.dashboard_auth.prefix import resolve_public_url

    suffix = f"/api/mcp/oauth/callback/{quote(server_name, safe='')}"
    public_url = resolve_public_url()
    if public_url:
        return f"{public_url}{suffix}"
    state = getattr(sys.modules.get("hermes_cli.web_server"), "app", None)
    state = getattr(state, "state", None)
    port = getattr(state, "bound_port", None)
    if not port:
        raise RuntimeError(
            "this Hermes process is not serving the OAuth callback route; authorize the server "
            f"from a desktop session or run `hermes mcp login {server_name}`")
    host = str(getattr(state, "bound_host", "") or "")
    host = "127.0.0.1" if host in _WILDCARD_HOSTS else host
    netloc = f"[{host}]:{port}" if ":" in host else f"{host}:{port}"
    return f"http://{netloc}{suffix}"


def _register_for_callback(flow) -> Callable[[], None]:
    """Put ``flow`` in the table the callback route matches on (server name + state), and return
    the function that drops it again."""
    module = sys.modules.get("hermes_cli.web_routers.mcp")
    if module is None:
        raise RuntimeError(
            "this Hermes process is not serving the OAuth callback route; authorize the server "
            f"from a desktop session or run `hermes mcp login {flow.server_name}`")
    with module._mcp_oauth_flows_lock:
        module._mcp_oauth_flows[flow.flow_id] = flow

    def drop() -> None:
        with module._mcp_oauth_flows_lock:
            module._mcp_oauth_flows.pop(flow.flow_id, None)

    return drop


@dataclass
class OAuthAttempt:
    """One flow in flight: the URL the user opens, and the worker's outcome as the watcher reads it."""

    auth_url: str
    flow: Any

    def poll(self) -> Dict[str, Any]:
        """``{status: pending|approved|error, error, tools}``; the bridge's intermediate states
        are all ``pending`` to the watcher."""
        snapshot = self.flow.snapshot()
        raw = snapshot.get("status")
        status = raw if raw in ("approved", "error") else "pending"
        tools = [str(t.get("name") or "") for t in (getattr(self.flow, "tools", None) or [])]
        return {"status": status, "error": snapshot.get("error") or "",
                "tools": [t for t in tools if t] if status == "approved" else []}


def start(server_name: str, *, url_timeout: float = URL_TIMEOUT_SECONDS) -> OAuthAttempt:
    """Start an OAuth flow for a configured server and block until it publishes its authorization
    URL. Raises when the flow fails or times out before the URL exists."""
    from hermes_cli.mcp_config import _get_mcp_servers
    from hermes_constants import get_hermes_home
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow

    cfg = dict(_get_mcp_servers().get(server_name) or {})
    if not cfg:
        raise RuntimeError(f"'{server_name}' is not a configured MCP server")
    if not cfg.get("url"):
        raise RuntimeError(f"'{server_name}' is a stdio server: it takes env keys, not OAuth")
    cfg["auth"] = "oauth"
    hermes_home = str(get_hermes_home().expanduser().resolve(strict=False))
    flow = DashboardOAuthFlow(
        flow_id=secrets.token_urlsafe(24), server_name=server_name, profile=None,
        hermes_home=hermes_home, redirect_uri=callback_url(server_name),
        # A live reconnect would swap the session's toolset mid-conversation; the tools land on the
        # next turn instead.
        reconnect_live=False)
    drop = _register_for_callback(flow)
    threading.Thread(
        target=run_worker, args=(hermes_home, server_name, cfg, False),
        kwargs={"flow": flow, "on_done": drop},
        daemon=True, name=f"mcp-oauth-{server_name}").start()
    deadline = time.time() + url_timeout
    while time.time() < deadline:
        snapshot = flow.snapshot()
        if snapshot.get("authorization_url"):
            return OAuthAttempt(auth_url=snapshot["authorization_url"], flow=flow)
        if snapshot.get("status") == "error":
            raise RuntimeError(snapshot.get("error") or "the OAuth flow failed before authorization")
        time.sleep(0.05)
    flow.mark_error("Timed out waiting for MCP authorization URL")
    raise TimeoutError(f"timed out waiting for the authorization URL for '{server_name}'")
