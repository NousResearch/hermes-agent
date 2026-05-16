"""Feishu User OAuth Tool -- per-user token management via PKCE.

Provides tools:
  - feishu_oauth_authorize  : Generate OAuth authorization URL for the current user
  - feishu_oauth_callback   : Exchange authorization code for stored tokens
  - feishu_oauth_status    : Check current user's OAuth authorization status

Token storage: ~/.hermes/feishu-user-tokens/<open_id>.json
PKCE verifier storage:  ~/.hermes/feishu-user-tokens/<open_id>.pending.json (10-min TTL)
"""

import base64
import hashlib
import json
import logging
import os
import secrets
import time
import urllib.parse
from pathlib import Path
from typing import Optional

from tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _get_hermes_home() -> Path:
    val = os.environ.get("HERMES_HOME", "").strip()
    return Path(val) if val else Path.home() / ".hermes"


class FeishuOAuthConfig:
    """Read OAuth / app credentials from environment."""

    def __init__(self):
        self.app_id: str = os.environ["FEISHU_APP_ID"]
        self.app_secret: str = os.environ["FEISHU_APP_SECRET"]
        self.redirect_uri: str = os.environ.get(
            "FEISHU_OAUTH_REDIRECT_URI", "http://localhost:8765/callback"
        )
        self.scopes: str = os.environ.get(
            "FEISHU_OAUTH_SCOPES",
            # Calendar (confirmed working with user token)
            "calendar:calendar:read "
            "calendar:calendar.event:read "
            "calendar:calendar.event:create "
            "calendar:calendar.event:update "
            # Task
            "task:task:read "
            "task:task:write "
            "task:tasklist:read "
            "task:tasklist:write "
        )


# ---------------------------------------------------------------------------
# Bot-level lark client (for token exchange / refresh — no user token needed)
# ---------------------------------------------------------------------------

_bot_client = None


def _get_bot_client():
    """Return a bot-level lark client with enable_set_token=True."""
    global _bot_client
    if _bot_client is None:
        import lark_oapi as lark

        cfg = FeishuOAuthConfig()
        _bot_client = (
            lark.Client.builder()
            .app_id(cfg.app_id)
            .app_secret(cfg.app_secret)
            .enable_set_token(True)
            .log_level(lark.LogLevel.WARNING)
            .build()
        )
    return _bot_client


# ---------------------------------------------------------------------------
# Token Store
# ---------------------------------------------------------------------------

_TOKEN_DIR: Optional[Path] = None


def _token_dir() -> Path:
    global _TOKEN_DIR
    if _TOKEN_DIR is None:
        _TOKEN_DIR = _get_hermes_home() / "feishu-user-tokens"
        _TOKEN_DIR.mkdir(parents=True, exist_ok=True)
    return _TOKEN_DIR


def _safe_filename(open_id: str) -> str:
    """Sanitize open_id for use as a filename."""
    return open_id.replace("/", "_").replace("\\", "_")


# ---------------------------------------------------------------------------
# Disk I/O helpers
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.chmod(tmp, 0o600)
        tmp.rename(path)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# FeishuUserTokenStore
# ---------------------------------------------------------------------------


class FeishuUserTokenStore:
    """Manage per-user Feishu OAuth tokens with PKCE and auto-refresh."""

    # ---------------------------------------------------------------------------
    # PKCE helpers
    # ---------------------------------------------------------------------------

    @staticmethod
    def _pkce_pair() -> tuple[str, str]:
        """Generate (code_verifier, code_challenge) for PKCE S256."""
        verifier = secrets.token_urlsafe(32)
        digest = hashlib.sha256(verifier.encode()).digest()
        challenge = (
            base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
        )
        return verifier, challenge

    # ---------------------------------------------------------------------------
    # File paths
    # ---------------------------------------------------------------------------

    def _token_path(self, open_id: str) -> Path:
        return _token_dir() / f"{_safe_filename(open_id)}.json"

    def _pending_path(self, open_id: str) -> Path:
        return _token_dir() / f"{_safe_filename(open_id)}.pending.json"

    # ---------------------------------------------------------------------------
    # Pending PKCE session
    # ---------------------------------------------------------------------------

    def _read_pending(self, open_id: str) -> Optional[dict]:
        p = self._pending_path(open_id)
        data = _read_json(p)
        if data and (time.time() - data.get("created_at", 0)) > 600:
            p.unlink(missing_ok=True)
            return None
        return data

    def _write_pending(self, open_id: str, data: dict) -> None:
        _write_json(self._pending_path(open_id), data)

    def _delete_pending(self, open_id: str) -> None:
        self._pending_path(open_id).unlink(missing_ok=True)

    # ---------------------------------------------------------------------------
    # Token exchange & refresh
    # ---------------------------------------------------------------------------

    def get_authorization_url(self, open_id: str) -> tuple[str, str]:
        """Build OAuth URL and persist PKCE verifier.

        Returns (url, state).  Caller displays the URL; user copies the
        ?code=... query param from the redirect and passes it to exchange_code().
        """
        cfg = FeishuOAuthConfig()
        verifier, challenge = self._pkce_pair()
        state = secrets.token_urlsafe(16)

        params = {
            "app_id": cfg.app_id,
            "redirect_uri": cfg.redirect_uri,
            "state": state,
            "scope": cfg.scopes,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
        url = (
            "https://open.feishu.cn/open-apis/authen/v1/authorize?"
            + urllib.parse.urlencode(params)
        )

        self._write_pending(open_id, {
            "verifier": verifier,
            "state": state,
            "created_at": time.time(),
        })

        logger.info("OAuth URL generated for open_id=%s", open_id)
        return url, state

    def exchange_code(self, open_id: str, code: str) -> dict:
        """Exchange authorization code for tokens using form-urlencoded (Feishu requirement)."""
        import urllib.request

        pending = self._read_pending(open_id)
        if not pending:
            raise RuntimeError("No PKCE pending session found for open_id. Authorization may have expired.")
        verifier = pending.get("verifier", "")
        self._delete_pending(open_id)

        cfg = FeishuOAuthConfig()
        data = urllib.parse.urlencode({
            "grant_type": "authorization_code",
            "code": code,
            "code_verifier": verifier,
            "app_id": cfg.app_id,
            "app_secret": cfg.app_secret,
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://open.feishu.cn/open-apis/authen/v1/oidc/access_token",
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())

        code_attr = result.get("code")
        if code_attr != 0:
            raise RuntimeError(f"Token exchange failed: code={code_attr} msg={result.get('msg')} resp={result}")

        data = result.get("data", {})
        now = time.time()
        record = {
            "access_token": data.get("access_token", ""),
            "refresh_token": data.get("refresh_token", ""),
            "expires_at": now + float(data.get("expires_in", 7200)),
            "refresh_expires_at": now + float(data.get("refresh_expires_in", 2592000)),
            "open_id": open_id,
        }

        _write_json(self._token_path(open_id), record)
        logger.info("OAuth tokens stored for open_id=%s", open_id)
        return record

    def refresh_token(self, open_id: str) -> Optional[dict]:
        """Refresh access_token using stored refresh_token. Returns updated record."""
        import urllib.request

        record = _read_json(self._token_path(open_id))
        if not record:
            return None
        refresh = record.get("refresh_token", "")
        if not refresh:
            return None

        cfg = FeishuOAuthConfig()
        # Get app_access_token for authentication
        app_token_req = urllib.request.Request(
            "https://open.feishu.cn/open-apis/auth/v3/app_access_token/internal",
            data=json.dumps({"app_id": cfg.app_id, "app_secret": cfg.app_secret}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(app_token_req, timeout=30) as r:
            app_token_result = json.loads(r.read())
        app_token = app_token_result.get("app_access_token", "")

        data = urllib.parse.urlencode({
            "grant_type": "refresh_token",
            "refresh_token": refresh,
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://open.feishu.cn/open-apis/authen/v1/oidc/refresh_access_token",
            data=data,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Authorization": f"Bearer {app_token}",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())
        except Exception as e:
            logger.warning("Token refresh failed for open_id=%s: %s", open_id, e)
            return None

        code_attr = result.get("code")
        if code_attr != 0:
            logger.warning("Token refresh failed for open_id=%s: %s", open_id, result.get("msg"))
            return None

        data = result.get("data", {})
        now = time.time()
        record.update({
            "access_token": data.get("access_token", record["access_token"]),
            "refresh_token": data.get("refresh_token", refresh),
            "expires_at": now + float(data.get("expires_in", 7200)),
            "refresh_expires_at": now + float(data.get("refresh_expires_in", 2592000)),
        })

        _write_json(self._token_path(open_id), record)
        logger.info("Token refreshed for open_id=%s", open_id)
        return record

    # ---------------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------------

    def get_user_token(self, open_id: str) -> Optional[str]:
        """Return a valid access_token, auto-refreshing if expired or near expiry."""
        if not open_id:
            return None
        record = _read_json(self._token_path(open_id))
        if not record:
            return None
        # Refresh if expired or expires within 5 minutes
        if time.time() >= record.get("expires_at", 0) - 300:
            record = self.refresh_token(open_id)
            if not record:
                return None
        return record.get("access_token")

    def get_request_option(self, open_id: str) -> Optional["RequestOption"]:
        """Return a RequestOption with user_access_token set, or None."""
        from lark_oapi import RequestOption

        token = self.get_user_token(open_id)
        if not token:
            return None
        return RequestOption.builder().user_access_token(token).build()

    def get_auth_status(self, open_id: str) -> dict:
        """Return OAuth status for the given open_id."""
        record = _read_json(self._token_path(open_id))
        if not record:
            return {"status": "not_authorized", "open_id": open_id}
        expires_at = record.get("expires_at", 0)
        now = time.time()
        if now >= expires_at:
            status = "expired"
        elif now >= expires_at - 300:
            status = "expiring_soon"
        else:
            status = "authorized"
        return {
            "status": status,
            "open_id": open_id,
            "expires_at": expires_at,
            "refresh_expires_at": record.get("refresh_expires_at"),
        }

    def revoke(self, open_id: str) -> None:
        """Delete stored tokens and pending PKCE data."""
        self._token_path(open_id).unlink(missing_ok=True)
        self._pending_path(open_id).unlink(missing_ok=True)
        logger.info("OAuth tokens revoked for open_id=%s", open_id)


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

FEISHU_OAUTH_AUTHORIZE_SCHEMA = {
    "name": "feishu_oauth_authorize",
    "description": (
        "Generate a Feishu OAuth 2.0 authorization URL for the current user. "
        "Visit the URL in a browser, log in, and copy the 'code' parameter from the redirect URL. "
        "Then call feishu_oauth_callback with that code to complete the flow.\n\n"
        "Required OAuth scopes: calendar:calendar:read, calendar:calendar.event:write, "
        "task:task:read, task:task:write.\n\n"
        "Returns: { authorization_url, state }"
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

FEISHU_OAUTH_CALLBACK_SCHEMA = {
    "name": "feishu_oauth_callback",
    "description": (
        "Complete the Feishu OAuth flow by exchanging the authorization code for tokens. "
        "Call this after the user visits the URL from feishu_oauth_authorize and provides "
        "the 'code' query parameter from the redirect URL.\n\n"
        "Args:\n"
        "  code (str, required): The 'code' query parameter from the OAuth redirect URL.\n\n"
        "Returns: { success, open_id, expires_at }"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": (
                    "The 'code' query parameter from the OAuth redirect URL. "
                    "Example: If redirect is 'http://localhost:8765/callback?code=abc123', "
                    "pass code='abc123'."
                ),
            },
        },
        "required": ["code"],
    },
}

FEISHU_OAUTH_STATUS_SCHEMA = {
    "name": "feishu_oauth_status",
    "description": (
        "Check the current user's Feishu OAuth authorization status.\n\n"
        "Statuses:\n"
        "  - not_authorized: User has not completed OAuth flow\n"
        "  - authorized:    User token is valid\n"
        "  - expiring_soon: Token valid but expires within 5 minutes — refresh recommended\n"
        "  - expired:       Token expired; user needs to re-authenticate\n\n"
        "Returns: { status, open_id, expires_at, refresh_expires_at }"
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


def _open_id_from_context() -> str:
    """Read current user's open_id from session context."""
    try:
        from gateway.session_context import get_session_env

        return get_session_env("HERMES_SESSION_USER_ID", "")
    except Exception:
        return ""


def _handle_authorize(args: dict, **kw) -> str:
    open_id = _open_id_from_context()
    if not open_id:
        return tool_error(
            "Cannot determine user identity. Run this command in an active Feishu session."
        )
    try:
        store = FeishuUserTokenStore()
        url, state = store.get_authorization_url(open_id)
        return tool_result(authorization_url=url, state=state)
    except Exception as e:
        logger.exception("feishu_oauth_authorize failed")
        return tool_error(f"Failed to build authorization URL: {e}")


def _handle_callback(args: dict, **kw) -> str:
    open_id = _open_id_from_context()
    if not open_id:
        return tool_error("Cannot determine user identity.")
    code = args.get("code", "").strip()
    if not code:
        return tool_error("'code' parameter is required.")
    try:
        store = FeishuUserTokenStore()
        result = store.exchange_code(open_id, code)
        return tool_result(
            success=True,
            open_id=open_id,
            expires_at=result.get("expires_at"),
            refresh_expires_at=result.get("refresh_expires_at"),
        )
    except Exception as e:
        logger.exception("feishu_oauth_callback failed")
        return tool_error(f"OAuth exchange failed: {e}")


def _handle_status(args: dict, **kw) -> str:
    open_id = _open_id_from_context()
    if not open_id:
        return tool_error("Cannot determine user identity.")
    try:
        store = FeishuUserTokenStore()
        status = store.get_auth_status(open_id)
        return tool_result(**status)
    except Exception as e:
        logger.exception("feishu_oauth_status failed")
        return tool_error(f"Status check failed: {e}")


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------


def _check_feishu():
    try:
        import lark_oapi  # noqa: F401
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

for _name, _schema, _handler in [
    ("feishu_oauth_authorize", FEISHU_OAUTH_AUTHORIZE_SCHEMA, _handle_authorize),
    ("feishu_oauth_callback", FEISHU_OAUTH_CALLBACK_SCHEMA, _handle_callback),
    ("feishu_oauth_status", FEISHU_OAUTH_STATUS_SCHEMA, _handle_status),
]:
    registry.register(
        name=_name,
        toolset="feishu_oauth",
        schema=_schema,
        handler=_handler,
        check_fn=_check_feishu,
        is_async=False,
        emoji="🔐",
    )
