#!/usr/bin/env python3
"""
MCP OAuth 2.1 PKCE Authentication

Implements the OAuth 2.1 Authorization Code + PKCE flow for MCP HTTP servers.
Handles discovery, dynamic client registration, browser-based authorization,
token storage, and automatic refresh.

Token files are stored per-server in ``~/.hermes/mcp-tokens/<server_name>.json``
with ``0o600`` permissions.

This module has **no external dependencies** — it uses only the Python stdlib
(``urllib.request``, ``hashlib``, ``secrets``, ``base64``, ``http.server``,
``webbrowser``).

Usage from ``mcp_tool.py``::

    from tools.mcp_oauth import get_auth_headers
    headers = get_auth_headers("my-server", "https://example.com/mcp")
    # → {"Authorization": "Bearer <access_token>"}
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import secrets
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────

_HERMES_HOME = Path(os.getenv("HERMES_HOME", str(Path.home() / ".hermes")))
_TOKEN_DIR = _HERMES_HOME / "mcp-tokens"

# ── Constants ────────────────────────────────────────────────────────────────

_CALLBACK_HOST = "127.0.0.1"
_CALLBACK_PORT_RANGE = (18400, 18500)  # try ports in this range
_TOKEN_EXPIRY_BUFFER_SECS = 60  # refresh 60s before actual expiry


# ── PKCE ─────────────────────────────────────────────────────────────────────

def generate_pkce() -> tuple[str, str]:
    """Generate PKCE ``code_verifier`` and ``code_challenge`` (S256).

    Returns (verifier, challenge) tuple.
    """
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode()
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()
    ).rstrip(b"=").decode()
    return verifier, challenge


# ── OAuth Metadata Discovery ────────────────────────────────────────────────

def discover_oauth_metadata(server_url: str, timeout: float = 10.0) -> Optional[Dict[str, Any]]:
    """Fetch OAuth 2.0 Authorization Server metadata.

    Tries ``/.well-known/oauth-authorization-server`` first, then
    ``/.well-known/openid-configuration`` as a fallback.

    Returns the metadata dict, or ``None`` if discovery fails.
    """
    parsed = urllib.parse.urlparse(server_url)
    base = f"{parsed.scheme}://{parsed.netloc}"

    well_known_paths = [
        "/.well-known/oauth-authorization-server",
        "/.well-known/openid-configuration",
    ]

    for path in well_known_paths:
        url = base + path
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    if isinstance(data, dict) and "authorization_endpoint" in data:
                        logger.debug("OAuth metadata discovered at %s", url)
                        return data
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError,
                OSError, ValueError) as e:
            logger.debug("OAuth discovery failed for %s: %s", url, e)
            continue

    return None


# ── Dynamic Client Registration (RFC 7591) ──────────────────────────────────

def register_client(
    metadata: Dict[str, Any],
    redirect_uri: str,
    server_name: str,
    timeout: float = 10.0,
) -> Optional[Dict[str, Any]]:
    """Dynamically register an OAuth client with the authorization server.

    Returns dict with ``client_id`` (and optionally ``client_secret``),
    or ``None`` if registration is not supported or fails.
    """
    reg_endpoint = metadata.get("registration_endpoint")
    if not reg_endpoint:
        logger.debug("Server does not advertise registration_endpoint — skipping dynamic registration")
        return None

    reg_data = json.dumps({
        "client_name": f"Hermes Agent ({server_name})",
        "redirect_uris": [redirect_uri],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",  # public client
    }).encode("utf-8")

    req = urllib.request.Request(
        reg_endpoint,
        data=reg_data,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            client_id = result.get("client_id")
            if client_id:
                logger.debug("Dynamic client registration successful: client_id=%s", client_id[:8] + "...")
                return {
                    "client_id": client_id,
                    "client_secret": result.get("client_secret", ""),
                }
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError,
            OSError, ValueError) as e:
        logger.debug("Dynamic client registration failed: %s", e)

    return None


# ── Token Storage ────────────────────────────────────────────────────────────

def _token_path(server_name: str) -> Path:
    """Return the token file path for a given server."""
    safe_name = server_name.replace("/", "_").replace("\\", "_").replace("..", "_")
    return _TOKEN_DIR / f"{safe_name}.json"


def load_tokens(server_name: str) -> Optional[Dict[str, Any]]:
    """Load cached OAuth tokens for a server.

    Returns token dict or ``None`` if no tokens are cached.
    """
    path = _token_path(server_name)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("access_token"):
            return data
    except (json.JSONDecodeError, OSError, IOError) as e:
        logger.debug("Failed to read tokens for %s: %s", server_name, e)
    return None


def save_tokens(server_name: str, tokens: Dict[str, Any]) -> None:
    """Save OAuth tokens for a server securely.

    Creates ``~/.hermes/mcp-tokens/`` directory if it doesn't exist.
    Sets file permissions to ``0o600`` (owner read/write only).
    """
    path = _token_path(server_name)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(tokens, indent=2), encoding="utf-8")
        path.chmod(0o600)
        logger.debug("Saved tokens for MCP server '%s'", server_name)
    except (OSError, IOError) as e:
        logger.debug("Failed to save tokens for %s: %s", server_name, e)


def _is_token_valid(tokens: Dict[str, Any]) -> bool:
    """Check if the access token is still valid (not expired)."""
    expires_at = tokens.get("expires_at", 0)
    if not expires_at:
        # No expiry — assume valid if token present
        return bool(tokens.get("access_token"))
    return time.time() < (expires_at - _TOKEN_EXPIRY_BUFFER_SECS)


# ── Token Refresh ────────────────────────────────────────────────────────────

def refresh_access_token(
    server_name: str,
    tokens: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Refresh an expired access token using the stored refresh token.

    Returns the new access token, or ``None`` if refresh fails.
    """
    tokens = tokens or load_tokens(server_name)
    if not tokens:
        return None

    refresh_token = tokens.get("refresh_token")
    token_endpoint = tokens.get("token_endpoint")
    client_id = tokens.get("client_id")

    if not refresh_token or not token_endpoint or not client_id:
        logger.debug("Missing refresh_token, token_endpoint, or client_id — cannot refresh")
        return None

    form_data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": client_id,
    }
    client_secret = tokens.get("client_secret", "")
    if client_secret:
        form_data["client_secret"] = client_secret

    data = urllib.parse.urlencode(form_data).encode("utf-8")
    req = urllib.request.Request(
        token_endpoint,
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode("utf-8"))

        new_access = result.get("access_token", "")
        if not new_access:
            logger.debug("Token refresh response missing access_token")
            return None

        # Update stored tokens
        tokens["access_token"] = new_access
        tokens["refresh_token"] = result.get("refresh_token", refresh_token)
        tokens["token_type"] = result.get("token_type", "Bearer")
        expires_in = result.get("expires_in")
        if expires_in:
            tokens["expires_at"] = time.time() + int(expires_in)

        save_tokens(server_name, tokens)
        logger.debug("Successfully refreshed token for MCP server '%s'", server_name)
        return new_access

    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError,
            OSError, ValueError) as e:
        logger.debug("Token refresh failed for %s: %s", server_name, e)
        return None


# ── Local Callback Server ───────────────────────────────────────────────────

class _OAuthCallbackHandler(BaseHTTPRequestHandler):
    """HTTP handler that captures the OAuth authorization code from the redirect."""

    auth_code: Optional[str] = None
    state: Optional[str] = None
    error: Optional[str] = None

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)

        _OAuthCallbackHandler.error = params.get("error", [None])[0]

        if "code" in params:
            _OAuthCallbackHandler.auth_code = params["code"][0]
            _OAuthCallbackHandler.state = params.get("state", [None])[0]

            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<html><body><h2>Authorization successful!</h2>"
                b"<p>You can close this tab and return to Hermes.</p>"
                b"</body></html>"
            )
        else:
            error_desc = params.get("error_description", ["Unknown error"])[0]
            self.send_response(400)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                f"<html><body><h2>Authorization failed</h2>"
                f"<p>{error_desc}</p></body></html>".encode("utf-8")
            )

    def log_message(self, format, *args):
        """Suppress default server log output."""
        pass


def _find_free_port() -> int:
    """Find a free port in the callback port range."""
    for port in range(_CALLBACK_PORT_RANGE[0], _CALLBACK_PORT_RANGE[1]):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((_CALLBACK_HOST, port))
                return port
        except OSError:
            continue
    raise RuntimeError(
        f"No free port found in range {_CALLBACK_PORT_RANGE[0]}-{_CALLBACK_PORT_RANGE[1]}"
    )


def _run_callback_server(port: int, timeout: float = 120.0) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Start a temporary HTTP server to capture the OAuth callback.

    Returns ``(code, state, error)`` tuple.
    Blocks until the callback is received or timeout expires.
    """
    # Reset class-level state
    _OAuthCallbackHandler.auth_code = None
    _OAuthCallbackHandler.state = None
    _OAuthCallbackHandler.error = None

    server = HTTPServer((_CALLBACK_HOST, port), _OAuthCallbackHandler)
    server.timeout = timeout

    # Handle one request (the OAuth callback)
    server.handle_request()
    server.server_close()

    return (
        _OAuthCallbackHandler.auth_code,
        _OAuthCallbackHandler.state,
        _OAuthCallbackHandler.error,
    )


# ── Interactive Auth Flow ───────────────────────────────────────────────────

def _prompt_manual_code(auth_url: str) -> Optional[str]:
    """Prompt the user to manually visit the auth URL and paste the code.

    Used in headless environments (SSH, gateway/messaging platforms).
    """
    print()
    print("╭─ MCP Server Authorization ─────────────────────────╮")
    print("│                                                     │")
    print("│  Open this link in your browser to authorize:       │")
    print("╰─────────────────────────────────────────────────────╯")
    print()
    print(f"  {auth_url}")
    print()
    print("After authorizing, paste the authorization code below.")
    print()
    try:
        code = input("Authorization code: ").strip()
        return code if code else None
    except (KeyboardInterrupt, EOFError):
        return None


def _can_open_browser() -> bool:
    """Check if we're in an environment that can open a browser."""
    # Gateway/messaging platforms set the platform env var
    platform = os.getenv("HERMES_PLATFORM", "")
    if platform and platform != "cli":
        return False
    # Check for display on Linux
    if os.name == "posix" and not os.getenv("DISPLAY") and not os.getenv("WAYLAND_DISPLAY"):
        # macOS doesn't need DISPLAY
        import sys
        if sys.platform != "darwin":
            return False
    return True


def start_auth_flow(
    server_name: str,
    server_url: str,
    metadata: Optional[Dict[str, Any]] = None,
    callback: Any = None,
) -> Optional[Dict[str, Any]]:
    """Run the full OAuth 2.1 PKCE authorization flow interactively.

    Steps:
      1. Discover OAuth metadata (if not provided)
      2. Dynamically register a client (if supported)
      3. Generate PKCE challenge
      4. Open browser to authorization URL
      5. Capture callback with authorization code
      6. Exchange code for tokens
      7. Save and return tokens

    *callback* is an optional callable for gateway platforms
    (``callback(auth_url) -> code``) to present the URL to the user.

    Returns the full token dict, or ``None`` on failure.
    """
    # Step 1: Discover metadata
    if metadata is None:
        metadata = discover_oauth_metadata(server_url)
    if not metadata:
        logger.warning("Could not discover OAuth metadata for %s", server_url)
        return None

    auth_endpoint = metadata.get("authorization_endpoint")
    token_endpoint = metadata.get("token_endpoint")
    if not auth_endpoint or not token_endpoint:
        logger.warning("OAuth metadata missing authorization_endpoint or token_endpoint")
        return None

    # Step 2: Determine redirect URI and try dynamic registration
    use_browser = _can_open_browser()

    if use_browser:
        try:
            port = _find_free_port()
            redirect_uri = f"http://{_CALLBACK_HOST}:{port}/callback"
        except RuntimeError:
            use_browser = False
            redirect_uri = "urn:ietf:wg:oauth:2.0:oob"
    else:
        redirect_uri = "urn:ietf:wg:oauth:2.0:oob"

    # Try dynamic client registration
    reg = register_client(metadata, redirect_uri, server_name)
    client_id = reg["client_id"] if reg else metadata.get("client_id", "hermes-agent")
    client_secret = reg.get("client_secret", "") if reg else ""

    # Step 3: Generate PKCE
    verifier, challenge = generate_pkce()
    state = secrets.token_urlsafe(16)

    # Step 4: Build authorization URL
    scopes = metadata.get("scopes_supported", [])
    scope_str = " ".join(scopes) if scopes else ""

    auth_params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
    }
    if scope_str:
        auth_params["scope"] = scope_str

    auth_url = f"{auth_endpoint}?{urllib.parse.urlencode(auth_params)}"

    # Step 5: Open browser or prompt for manual code
    auth_code = None

    if use_browser:
        import webbrowser

        print()
        print(f"Authorizing MCP server '{server_name}'...")
        print(f"Opening browser for authorization...")
        print()

        # Start callback server in background thread
        result_holder: Dict[str, Any] = {}

        def _serve():
            code, returned_state, error = _run_callback_server(port)
            result_holder["code"] = code
            result_holder["state"] = returned_state
            result_holder["error"] = error

        server_thread = threading.Thread(target=_serve, daemon=True)
        server_thread.start()

        try:
            webbrowser.open(auth_url)
        except Exception:
            print(f"Could not open browser automatically. Please visit:")
            print(f"  {auth_url}")

        # Wait for callback
        server_thread.join(timeout=120.0)

        auth_code = result_holder.get("code")
        returned_state = result_holder.get("state")
        error = result_holder.get("error")

        if error:
            logger.warning("OAuth authorization error: %s", error)
            return None

        if returned_state and returned_state != state:
            logger.warning("OAuth state mismatch — possible CSRF attack")
            return None

    elif callback is not None:
        # Gateway/platform callback
        auth_code = callback(auth_url)
    else:
        # Manual code entry (headless)
        auth_code = _prompt_manual_code(auth_url)

    if not auth_code:
        logger.warning("No authorization code received")
        return None

    # Step 6: Exchange code for tokens
    exchange_data = urllib.parse.urlencode({
        "grant_type": "authorization_code",
        "code": auth_code,
        "redirect_uri": redirect_uri,
        "client_id": client_id,
        "code_verifier": verifier,
    })
    if client_secret:
        exchange_data += f"&client_secret={urllib.parse.quote(client_secret)}"

    req = urllib.request.Request(
        token_endpoint,
        data=exchange_data.encode("utf-8"),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError,
            OSError, ValueError) as e:
        logger.warning("Token exchange failed for %s: %s", server_name, e)
        return None

    access_token = result.get("access_token", "")
    if not access_token:
        logger.warning("Token exchange response missing access_token")
        return None

    # Step 7: Build and save token data
    expires_in = result.get("expires_in")
    tokens = {
        "access_token": access_token,
        "refresh_token": result.get("refresh_token", ""),
        "token_type": result.get("token_type", "Bearer"),
        "expires_at": (time.time() + int(expires_in)) if expires_in else 0,
        "client_id": client_id,
        "client_secret": client_secret,
        "token_endpoint": token_endpoint,
        "server_url": server_url,
    }

    save_tokens(server_name, tokens)
    print(f"MCP server '{server_name}' authorized successfully!")
    return tokens


# ── Main Entry Point ─────────────────────────────────────────────────────────

def get_auth_headers(
    server_name: str,
    server_url: str,
    callback: Any = None,
) -> Dict[str, str]:
    """Get OAuth authorization headers for an MCP server.

    This is the main entry point called by ``mcp_tool.py``.

    1. Check for cached valid tokens → return header
    2. If expired, try refresh → return header
    3. If no tokens, run interactive auth flow → return header
    4. If all fails, return empty dict (connection will proceed without auth)

    *callback* is an optional callable for gateway platforms to present
    the authorization URL to the user.
    """
    # Try cached tokens
    tokens = load_tokens(server_name)

    if tokens and _is_token_valid(tokens):
        token_type = tokens.get("token_type", "Bearer")
        return {"Authorization": f"{token_type} {tokens['access_token']}"}

    # Try refresh
    if tokens and tokens.get("refresh_token"):
        logger.debug("Access token expired for '%s' — attempting refresh", server_name)
        new_access = refresh_access_token(server_name, tokens)
        if new_access:
            return {"Authorization": f"Bearer {new_access}"}

    # Run full auth flow
    logger.debug("No valid tokens for '%s' — starting OAuth flow", server_name)
    tokens = start_auth_flow(server_name, server_url, callback=callback)
    if tokens:
        token_type = tokens.get("token_type", "Bearer")
        return {"Authorization": f"{token_type} {tokens['access_token']}"}

    logger.warning("OAuth authentication failed for MCP server '%s'", server_name)
    return {}
