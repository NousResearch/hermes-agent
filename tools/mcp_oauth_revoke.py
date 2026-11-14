"""Sign-out for a removed MCP server: delete its stored OAuth state, then revoke the grant (RFC 7009).

Removing a server must not leave a live refresh token behind at the authorization server, nor be
held hostage by it: local state is deleted first, then the refresh token and the access token are
revoked at the advertised ``revocation_endpoint`` (best effort, one shared time budget, through
the same endpoint rules as every other OAuth request). Without an advertised endpoint the local
delete is all that can be done.
"""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import contextvars
import logging
from pathlib import Path
from urllib.parse import quote, urlsplit

logger = logging.getLogger(__name__)

REVOKE_BUDGET_SECONDS = 5.0


def _revocation_plan(storage) -> tuple[str, list[tuple[str, str]], dict, dict] | None:
    """``(endpoint, [(token, hint)...], form auth fields, headers)`` from the stored state, or None."""
    from tools.mcp_oauth import _read_json
    meta = _read_json(storage._meta_path()) or {}
    tokens = _read_json(storage._tokens_path()) or {}
    endpoint = meta.get("revocation_endpoint") if isinstance(meta, dict) else None
    if not endpoint or not isinstance(tokens, dict):
        return None
    # Refresh first: revoking it usually ends the whole grant (RFC 7009 §2.1).
    pending = [(tokens[k], k) for k in ("refresh_token", "access_token") if isinstance(tokens.get(k), str) and tokens[k]]
    if not pending:
        return None
    client = _read_json(storage._client_info_path()) or {}
    client = client if isinstance(client, dict) else {}
    storage._coerce_secret_auth_method(client)
    client_id, secret = client.get("client_id") or "", client.get("client_secret") or ""
    fields, headers = {}, {}
    if client.get("token_endpoint_auth_method") == "client_secret_basic" and secret:
        pair = f"{quote(client_id, safe='')}:{quote(secret, safe='')}".encode()
        headers["Authorization"] = "Basic " + base64.b64encode(pair).decode()
    elif secret:
        fields = {"client_id": client_id, "client_secret": secret}
    elif client_id:
        fields = {"client_id": client_id}
    return str(endpoint), pending, fields, headers


async def _revoke(plan, server_url: str | None) -> str:
    from tools.mcp_oauth_endpoints import OAuthEndpointRefused, oauth_client_event_hooks
    from tools.mcp_tool import sdk_httpx
    httpx = sdk_httpx()
    if httpx is None:  # pragma: no cover — the MCP SDK (and its httpx) is required to have tokens at all
        return "MCP SDK unavailable; local tokens deleted"
    endpoint, pending, fields, headers = plan
    host = urlsplit(endpoint).netloc
    revoked, failures = 0, []
    try:
        async with asyncio.timeout(REVOKE_BUDGET_SECONDS):
            async with httpx.AsyncClient(timeout=REVOKE_BUDGET_SECONDS, follow_redirects=False,
                                         event_hooks=oauth_client_event_hooks(httpx, server_url)) as client:
                for token, hint in pending:
                    response = await client.post(endpoint, data={"token": token, "token_type_hint": hint, **fields},
                                                 headers=headers)
                    if response.status_code == 200:
                        revoked += 1
                    else:
                        failures.append(f"{hint} HTTP {response.status_code}")
    except TimeoutError:
        failures.append(f"timed out after {REVOKE_BUDGET_SECONDS:g}s")
    except OAuthEndpointRefused as exc:
        failures.append(str(exc))
    except httpx.HTTPError as exc:
        failures.append(type(exc).__name__)  # never the message: it can carry request details
    if failures:
        return f"token revocation at {host} incomplete ({'; '.join(failures)}); local tokens deleted"
    return f"revoked {revoked} token{'s' if revoked != 1 else ''} at {host}"


def forget_server_oauth(server_name: str, server_url: str | None, *,
                        hermes_home: str | Path | None = None) -> str | None:
    """Delete *server_name*'s stored OAuth state and revoke its tokens; a one-line outcome, or None
    when nothing was stored. Never raises for a network failure and returns within the budget."""
    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_manager import get_manager
    storage = HermesTokenStorage(server_name, hermes_home=hermes_home)
    had_tokens = storage.has_cached_tokens()
    plan = _revocation_plan(storage) if had_tokens else None
    get_manager().remove(server_name, hermes_home=hermes_home)  # evict the cached provider + delete files
    if not had_tokens:
        return None
    if plan is None:
        return "server advertises no token revocation endpoint; local tokens deleted"
    # A private loop on a worker thread: callers range from the CLI to async dashboard/RPC handlers.
    # The copied context keeps the caller's profile scope for the endpoint check's config reads.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        outcome = pool.submit(contextvars.copy_context().run, asyncio.run, _revoke(plan, server_url)).result()
    logger.info("MCP OAuth '%s': %s", server_name, outcome)
    return outcome
