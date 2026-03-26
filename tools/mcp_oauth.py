"""Thin OAuth adapter for MCP HTTP servers.

Wraps the MCP SDK's built-in ``OAuthClientProvider`` (which implements
``httpx.Auth``) with Hermes-specific token storage and browser-based
authorization.  The SDK handles all of the heavy lifting: PKCE generation,
metadata discovery, dynamic client registration, token exchange, and refresh.

Usage in mcp_tool.py::

    from tools.mcp_oauth import build_oauth_auth
    auth = build_oauth_auth(server_name, server_url)
    # pass ``auth`` as the httpx auth parameter
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)

_TOKEN_DIR_NAME = "mcp-tokens"


# ---------------------------------------------------------------------------
# Token storage — persists tokens + client info to ~/.hermes/mcp-tokens/
# ---------------------------------------------------------------------------

def _sanitize_server_name(name: str) -> str:
    """Sanitize server name for safe use as a filename."""
    import re
    clean = re.sub(r"[^\w\-]", "-", name.strip().lower())
    clean = re.sub(r"-+", "-", clean).strip("-")
    return clean[:60] or "unnamed"


class HermesTokenStorage:
    """File-backed token storage implementing the MCP SDK's TokenStorage protocol."""

    def __init__(self, server_name: str):
        self._server_name = _sanitize_server_name(server_name)

    def _base_dir(self) -> Path:
        home = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
        d = home / _TOKEN_DIR_NAME
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _tokens_path(self) -> Path:
        return self._base_dir() / f"{self._server_name}.json"

    def _client_path(self) -> Path:
        return self._base_dir() / f"{self._server_name}.client.json"

    # -- TokenStorage protocol (async) --

    async def get_tokens(self):
        data = self._read_json(self._tokens_path())
        if not data:
            return None
        try:
            from mcp.shared.auth import OAuthToken
            return OAuthToken(**data)
        except Exception:
            return None

    async def set_tokens(self, tokens) -> None:
        self._write_json(self._tokens_path(), tokens.model_dump(exclude_none=True))

    async def get_client_info(self):
        data = self._read_json(self._client_path())
        if not data:
            return None
        try:
            from mcp.shared.auth import OAuthClientInformationFull
            return OAuthClientInformationFull(**data)
        except Exception:
            return None

    async def set_client_info(self, client_info) -> None:
        self._write_json(self._client_path(), client_info.model_dump(exclude_none=True))

    # -- helpers --

    @staticmethod
    def _read_json(path: Path) -> dict | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    @staticmethod
    def _write_json(path: Path, data: dict) -> None:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        try:
            path.chmod(0o600)
        except OSError:
            pass

    def remove(self) -> None:
        """Delete stored tokens and client info for this server."""
        for p in (self._tokens_path(), self._client_path()):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Browser-based callback handler
# ---------------------------------------------------------------------------

def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_callback_handler():
    """Create a callback handler class with instance-scoped result storage."""
    result = {"auth_code": None, "state": None}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            qs = parse_qs(urlparse(self.path).query)
            result["auth_code"] = (qs.get("code") or [None])[0]
            result["state"] = (qs.get("state") or [None])[0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body><h3>Authorization complete. You can close this tab.</h3></body></html>")

        def log_message(self, *_args: Any) -> None:
            pass

    return Handler, result


# Port chosen at build time and shared with the callback handler via closure.
_oauth_port: int | None = None


async def _redirect_to_browser(auth_url: str) -> None:
    """Open the authorization URL in the user's browser."""
    try:
        if _can_open_browser():
            webbrowser.open(auth_url)
            print("  Opened browser for authorization...")
        else:
            print(f"\n  Open this URL to authorize:\n  {auth_url}\n")
    except Exception:
        print(f"\n  Open this URL to authorize:\n  {auth_url}\n")


async def _wait_for_callback() -> tuple[str, str | None]:
    """Start a local HTTP server on the pre-registered port and wait for the OAuth redirect."""
    global _oauth_port
    port = _oauth_port or _find_free_port()
    HandlerClass, result = _make_callback_handler()
    server = HTTPServer(("127.0.0.1", port), HandlerClass)

    def _serve():
        server.timeout = 120
        server.handle_request()

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()

    for _ in range(1200):  # 120 seconds
        await asyncio.sleep(0.1)
        if result["auth_code"] is not None:
            break

    server.server_close()
    code = result["auth_code"] or ""
    state = result["state"]
    if not code:
        print("  Browser callback timed out. Paste the authorization code manually:")
        code = input("  Code: ").strip()
    return code, state


def _can_open_browser() -> bool:
    if os.environ.get("SSH_CLIENT") or os.environ.get("SSH_TTY"):
        return False
    if not os.environ.get("DISPLAY") and os.name != "nt" and "darwin" not in os.uname().sysname.lower():
        return False
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_oauth_auth(server_name: str, server_url: str):
    """Build an ``httpx.Auth`` handler for the given MCP server using OAuth 2.1 PKCE.

    Uses the MCP SDK's ``OAuthClientProvider`` which handles discovery,
    registration, PKCE, token exchange, and refresh automatically.

    Returns an ``OAuthClientProvider`` instance (implements ``httpx.Auth``),
    or ``None`` if the MCP SDK auth module is not available.
    """
    try:
        from mcp.client.auth import OAuthClientProvider
        from mcp.shared.auth import OAuthClientMetadata
    except ImportError:
        logger.warning("MCP SDK auth module not available — OAuth disabled")
        return None

    global _oauth_port
    _oauth_port = _find_free_port()
    redirect_uri = f"http://127.0.0.1:{_oauth_port}/callback"

    client_metadata = OAuthClientMetadata(
        client_name="Hermes Agent",
        redirect_uris=[redirect_uri],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        scope="openid profile email offline_access",
        token_endpoint_auth_method="none",
    )

    storage = HermesTokenStorage(server_name)

    return OAuthClientProvider(
        server_url=server_url,
        client_metadata=client_metadata,
        storage=storage,
        redirect_handler=_redirect_to_browser,
        callback_handler=_wait_for_callback,
        timeout=120.0,
    )


def remove_oauth_tokens(server_name: str) -> None:
    """Delete stored OAuth tokens and client info for a server."""
    HermesTokenStorage(server_name).remove()
