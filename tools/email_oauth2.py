"""Email OAuth2 (XOAUTH2) helpers for Microsoft 365 / Exchange Online.

Provides token acquisition via client_credentials flow and SASL XOAUTH2
string generation for IMAP and SMTP authentication.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from dataclasses import replace as dataclass_replace
from hermes_constants import get_hermes_home

from tools.microsoft_graph_auth import (
    GraphCredentials,
    MicrosoftGraphTokenError,
    fetch_access_token_sync,
)

logger = logging.getLogger(__name__)

# Exchange Online scope for IMAP + SMTP access
EMAIL_OAUTH_SCOPE = "https://outlook.office.com/.default"
def _get_token_cache_path():
    """Return the OAuth token cache path, respecting EMAIL_OAUTH_TOKEN_PATH env."""
    from pathlib import Path  # noqa: F811
    env_path = os.getenv("EMAIL_OAUTH_TOKEN_PATH", "").strip()
    if env_path:
        return Path(env_path)
    return get_hermes_home() / "oauth_tokens.json"


OAUTH_TOKEN_CACHE_PATH = _get_token_cache_path()
TOKEN_SKEW_SECONDS = 120  # Refresh when < 120s from expiry


def get_xoauth2_string(address: str, access_token: str) -> bytes:
    """Build the SASL XOAUTH2 string for IMAP/SMTP authentication.

    Format: ``base64("user=<email>\\x01auth=Bearer <token>\\x01\\x01")``
    """
    auth_string = f"user={address}\x01auth=Bearer {access_token}\x01\x01"
    return base64.b64encode(auth_string.encode("utf-8"))


def load_disk_cache() -> dict | None:
    """Load the OAuth token cache from disk."""
    try:
        if OAUTH_TOKEN_CACHE_PATH.exists():
            data = json.loads(OAUTH_TOKEN_CACHE_PATH.read_text(encoding="utf-8"))
            return data.get("email_oauth2")
    except (json.JSONDecodeError, OSError) as e:
        logger.debug("[EmailOAuth2] Cache read error: %s", e)
    return None


def save_disk_cache(access_token: str, expires_at: float) -> None:
    """Persist the OAuth token to disk atomically with restricted permissions."""
    OAUTH_TOKEN_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {"email_oauth2": {"access_token": access_token, "expires_at": expires_at}}

    # Atomic write: create temp file with strict permissions, then rename
    tmp = OAUTH_TOKEN_CACHE_PATH.with_name(
        f"oauth_tokens.{os.getpid()}.tmp"
    )
    try:
        fd = os.open(tmp, os.O_CREAT | os.O_WRONLY | os.O_TRUNC | os.O_EXCL, 0o600)
        try:
            os.write(fd, json.dumps(data, indent=2).encode("utf-8"))
        except OSError:
            tmp.unlink(missing_ok=True)
            raise
        finally:
            os.close(fd)
        os.replace(tmp, OAUTH_TOKEN_CACHE_PATH)
    except OSError:
        # Fallback for uncommon platforms without os.open flags
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.chmod(0o600)
        tmp.rename(OAUTH_TOKEN_CACHE_PATH)

    logger.debug("[EmailOAuth2] Token cached to %s", OAUTH_TOKEN_CACHE_PATH)


def invalidate_disk_cache() -> None:
    """Remove the cached token so the next call fetches a fresh one."""
    try:
        OAUTH_TOKEN_CACHE_PATH.unlink(missing_ok=True)
        logger.debug("[EmailOAuth2] Cache invalidated")
    except OSError as e:
        logger.warning("[EmailOAuth2] Failed to invalidate cache: %s", e)


def _get_cached_token() -> str | None:
    """Return a valid cached token, or None if missing/expired."""
    cached = load_disk_cache()
    if cached is None:
        return None
    expires_at = cached.get("expires_at", 0)
    if time.time() + TOKEN_SKEW_SECONDS >= expires_at:
        logger.debug("[EmailOAuth2] Cached token expired (expired at %.0f)", expires_at)
        return None
    return cached.get("access_token")


def get_credentials_with_email_scope(
    environ: dict[str, str] | None = None,
) -> GraphCredentials:
    """Build GraphCredentials with the Exchange Online scope.

    Uses ``dataclasses.replace()`` to keep the dataclass frozen.
    """
    env = environ if environ is not None else os.environ
    creds = GraphCredentials.from_env(environ=env, required=True)
    return dataclass_replace(creds, scope=EMAIL_OAUTH_SCOPE)


def get_access_token(
    credentials: GraphCredentials | None = None,
    environ: dict[str, str] | None = None,
    *,
    force_refresh: bool = False,
) -> str:
    """Obtain an OAuth2 access token for Exchange Online.

    Tries disk cache first (unless ``force_refresh=True``), then acquires
    via ``fetch_access_token_sync``. The token is cached before being returned.
    """
    # 1. Try cache (unless force_refresh)
    if not force_refresh:
        cached = _get_cached_token()
        if cached is not None:
            logger.debug("[EmailOAuth2] Using cached token")
            return cached

    # 2. Acquire fresh token
    if credentials is None:
        credentials = get_credentials_with_email_scope(environ=environ)

    access_token, expires_at = fetch_access_token_sync(credentials)
    expires_in = int(expires_at - time.time())

    # 3. Cache and return
    save_disk_cache(access_token, expires_at)
    logger.info("[EmailOAuth2] Acquired fresh token (expires in %ds)", expires_in)
    return access_token
