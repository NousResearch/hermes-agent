"""Service-account credential for remote MCP servers (OAuth2 ``client_credentials``).

An MCP server may need *some* credential to accept a connection at all — before any user has
signed in, and for requests that are not made on a user's behalf (``initialize``, ``tools/list``,
the keepalive ``ping``). That credential represents **Hermes the client**, not a person: the
Kingdee server answers it with a ``caller`` role and no ``kd_username``, which is exactly the
right scope for "prove you are allowed to talk to me".

It is deliberately a *different* credential from the user's access token
(:mod:`agent.turn_token`). Two consequences worth stating, because they are the reason
this module exists rather than a static ``headers:`` entry:

* The token expires (Keycloak mints these per access-token lifespan), so a long-lived connection
  would start answering 401 mid-conversation. This mints on demand and refreshes ahead of expiry.
* The secret never reaches ``config.yaml``: ``config.yaml`` names an environment variable
  (``client_secret_env``) and the value is read through the profile secret scope, so a config
  file can be copied between machines without carrying a credential.

Config (``mcp_servers.<name>.service_credentials``)::

    service_credentials:
      token_url: "https://idp/realms/r/protocol/openid-connect/token"
      client_id: "hermes-mcp-client"
      client_secret_env: HERMES_MCP_CLIENT_SECRET   # the secret, resolved from .env
      scopes: "openid"                              # optional

Failure policy: a mint that fails yields ``None`` and the request goes out without the header, so
the server's own 401/403 is what the operator sees. Nothing falls back to another credential and
nothing is cached on failure.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger("tools.mcp_tool")

#: Refresh this many seconds before expiry: a request that starts just under the wire would
#: otherwise be rejected by the resource server after a slow mint.
EXPIRY_SKEW_SECONDS = 60
#: Bound on one mint. The hook runs inside a request, so a hung IdP must not hang the call
#: forever; the server's 401 is a better error than a stalled turn.
MINT_TIMEOUT_SECONDS = 10.0

#: What a mint can fail with, precisely: the HTTP/URL layer (DNS, TLS, timeout, a non-2xx),
#: JSON decoding, and our own "no access_token" verdict. Named rather than ``Exception`` so the
#: boundary states what it absorbs.
_MINT_FAILURES = (OSError, ValueError)


@dataclass(frozen=True)
class ServiceCredentialConfig:
    """Validated ``service_credentials`` block."""

    server_name: str
    token_url: str
    client_id: str
    client_secret_env: str = ""
    scopes: str = ""


def parse_service_credentials(server_name: str, config: dict) -> Optional[ServiceCredentialConfig]:
    """The validated block, or ``None`` for "not configured".

    Raises ``ValueError`` for a block that names itself but cannot work (no ``token_url`` or
    ``client_id``): connecting anyway would send no credential and surface as an opaque 401, so
    the operator gets the misconfiguration instead.
    """
    raw = config.get("service_credentials")
    if not isinstance(raw, dict) or not raw:
        return None

    def field(key: str) -> str:
        return str(raw.get(key) or "").strip()

    token_url, client_id = field("token_url"), field("client_id")
    if not token_url or not client_id:
        raise ValueError(
            f"MCP server '{server_name}': service_credentials needs both token_url and client_id")
    if not urllib.parse.urlparse(token_url).scheme:
        raise ValueError(
            f"MCP server '{server_name}': service_credentials.token_url must be an absolute URL")
    return ServiceCredentialConfig(
        server_name=server_name, token_url=token_url, client_id=client_id,
        client_secret_env=field("client_secret_env"), scopes=field("scopes") or field("scope"))


def _read_secret(env_name: str) -> str:
    """The client secret, resolved the way every other Hermes credential is.

    ``get_env_value`` is the canonical reader: ``os.environ`` through the profile secret scope
    (so a multiplexed gateway never borrows another profile's value) *and* a fallback to the
    profile's ``.env``. Reading only the bound scope would miss a secret that lives solely in
    ``.env`` — which is where ``config.yaml`` is supposed to keep it.
    """
    if not env_name:
        return ""
    try:
        from hermes_cli.config import get_env_value

        return (get_env_value(env_name) or "").strip()
    except Exception:  # health: allow BLE001 -- a config read must not stop a mint
        logger.warning("service_credentials: cannot read %s", env_name, exc_info=True)
        return ""


class ServiceCredentialMinter:
    """Mints and caches one server's service-account access token.

    Shared by every connection of the process (one per server entry), because the token is the
    *client's* credential: re-minting per connection would multiply IdP traffic for no benefit.

    ``async_token`` is the only entry point: the HTTP auth hook must not block the event loop,
    and a synchronous twin would invite a caller to do exactly that.
    """

    def __init__(self, spec: ServiceCredentialConfig) -> None:
        self._spec = spec
        self._lock = threading.Lock()
        self._token = ""
        self._expires_at = 0.0

    def cached_token(self, *, now: Optional[float] = None) -> Optional[str]:
        """The live cached token, or ``None`` when absent/expired. Never mints."""
        with self._lock:
            token, expires_at = self._token, self._expires_at
        if not token or expires_at - EXPIRY_SKEW_SECONDS <= (time.time() if now is None else now):
            return None
        return token

    async def async_token(self) -> Optional[str]:
        cached = self.cached_token()
        if cached is not None:
            return cached
        try:
            return await asyncio.to_thread(self._mint)
        except _MINT_FAILURES as exc:
            # A failed mint must never break the connection or the call: the request goes out
            # without the header and the server's own 401 is the error the operator sees.
            logger.warning("MCP server '%s': service credential mint failed: %s",
                           self._spec.server_name, exc)
            return None

    def _mint(self) -> Optional[str]:
        """POST the client_credentials grant; cache and return the token.

        Double-checked so a burst of concurrent requests mints once: the lock is held across the
        network call deliberately (the alternative is an IdP stampede on every expiry).
        """
        with self._lock:
            if self._token and self._expires_at - EXPIRY_SKEW_SECONDS > time.time():
                return self._token
            secret = _read_secret(self._spec.client_secret_env)
            if self._spec.client_secret_env and not secret:
                # Name the variable instead of sending an empty secret: the IdP answers
                # `unauthorized_client`, which reads like a permissions problem rather than the
                # missing .env entry it actually is.
                logger.warning("MCP server '%s': %s is set in service_credentials but has no "
                               "value in this profile's .env or secret source; the service "
                               "credential will likely be refused",
                               self._spec.server_name, self._spec.client_secret_env)
            data = {"grant_type": "client_credentials", "client_id": self._spec.client_id}
            if secret:
                data["client_secret"] = secret
            if self._spec.scopes:
                data["scope"] = self._spec.scopes
            payload = _post_form(self._spec.token_url, data)
            token = str(payload.get("access_token") or "").strip()
            if not token:
                raise ValueError("token endpoint returned no access_token")
            try:
                expires_in = int(payload.get("expires_in") or 0)
            except (TypeError, ValueError):
                expires_in = 0
            # No expires_in: assume the IdP's typical short life rather than caching forever.
            self._token = token
            self._expires_at = time.time() + (expires_in if expires_in > 0 else 300)
            logger.info("MCP server '%s': minted service credential for %s (expires in %ss)",
                        self._spec.server_name, self._spec.client_id, expires_in or 300)
            return self._token


def _post_form(url: str, data: dict[str, str]) -> dict[str, Any]:
    """``application/x-www-form-urlencoded`` POST returning a JSON object.

    ``urllib`` (not httpx) on purpose: this runs on a worker thread inside a request, and the
    module must not depend on which httpx generation the installed MCP SDK uses.
    """
    body = urllib.parse.urlencode(data).encode()
    request = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded",
                 "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=MINT_TIMEOUT_SECONDS) as response:
        payload = json.loads(response.read() or b"{}")
    return payload if isinstance(payload, dict) else {}


_minter_lock = threading.Lock()
_minters: dict[tuple, ServiceCredentialMinter] = {}


def minter_for(spec: ServiceCredentialConfig) -> ServiceCredentialMinter:
    """The process-wide minter for ``spec`` (shared across reconnects)."""
    key = (spec.server_name, spec.token_url, spec.client_id, spec.client_secret_env, spec.scopes)
    with _minter_lock:
        minter = _minters.get(key)
        if minter is None:
            minter = _minters[key] = ServiceCredentialMinter(spec)
        return minter


def clear_minters() -> None:
    """Test-only: drop every cached minter (and its token)."""
    with _minter_lock:
        _minters.clear()