"""Trust rules for the OAuth endpoints an MCP server advertises.

A server's discovery documents name the URLs Hermes then sends client secrets, authorization codes
and refresh tokens to, and opens in the user's browser. Every OAuth request Hermes makes for an MCP
server (SDK discovery/registration/token/refresh, pre-flight discovery, device login, revocation)
and the authorization URL it opens pass :func:`check_oauth_endpoint`:

- only ``http``/``https``; never the cloud-metadata floor;
- a public (or unresolvable) MCP server may not point Hermes at a private, loopback or link-local address
  (``security.allow_private_urls`` / ``fake_ip_ranges`` opt-outs apply, as for every other fetch);
- credentials need ``https``; plain ``http`` is accepted only for a loopback endpoint, or for a
  local-network endpoint of a server that is itself on the local network (that server already
  receives its bearer token over the same link).

Responses are capped at :data:`OAUTH_MAX_RESPONSE_BYTES` and :data:`OAUTH_FETCH_TIMEOUT` seconds.
"""

from __future__ import annotations

import functools
import time
from typing import Any
from urllib.parse import urlsplit

OAUTH_MAX_RESPONSE_BYTES = 1024 * 1024
OAUTH_FETCH_TIMEOUT = 20.0


class OAuthEndpointRefused(RuntimeError):
    """An advertised OAuth endpoint, or its response, failed the trust rules above."""


def _refuse(url: str, reason: str) -> OAuthEndpointRefused:
    parts = urlsplit(url)
    shown = f"{parts.scheme}://{parts.netloc}{parts.path}"  # never echo query strings (codes, state)
    return OAuthEndpointRefused(f"MCP OAuth refused {shown}: {reason}")


def check_oauth_endpoint(url: str, server_url: str | None) -> None:
    """Raise :class:`OAuthEndpointRefused` unless *url* may receive OAuth traffic for the MCP server
    at *server_url* (unknown server = public). Resolves DNS: call off the event loop."""
    from tools.url_safety import host_address_class, is_always_blocked_url, is_safe_url
    try:
        parts = urlsplit(url)
        scheme, host = parts.scheme.lower(), parts.hostname or ""
    except ValueError:
        raise OAuthEndpointRefused("MCP OAuth refused a malformed endpoint URL") from None
    if scheme not in ("http", "https") or not host:
        raise _refuse(url, "OAuth endpoints must be http(s) URLs")
    if is_always_blocked_url(url):
        raise _refuse(url, "it points at a cloud metadata address")
    server_host = urlsplit(server_url).hostname if server_url else None
    server_public = host_address_class(server_host) in ("public", "unresolved") if server_host else True
    endpoint_class = host_address_class(host)
    # A name that does not resolve cannot be dialed either; the request fails on its own.
    if server_public and endpoint_class != "unresolved" and not is_safe_url(url):
        raise _refuse(url, "the MCP server is public but this OAuth endpoint is on a private, loopback or "
                           "link-local address")
    if scheme == "http":
        if not (endpoint_class == "loopback" or (endpoint_class == "internal" and not server_public)):
            raise _refuse(url, "OAuth credentials are only sent over https (plain http is accepted for "
                               "loopback, or a local-network server's own local endpoints)")


async def check_oauth_endpoint_async(url: str, server_url: str | None) -> None:
    import anyio
    await anyio.to_thread.run_sync(check_oauth_endpoint, url, server_url)


# Request extension marking OAuth traffic: ``(server_url, url already checked)``. httpx copies a
# request's extensions onto every redirect hop it builds, so the MCP transport recognises those hops.
OAUTH_REQUEST_EXTENSION = "hermes_mcp_oauth"


async def guard_oauth_request(request: Any, server_url: str | None) -> None:
    """Validate an outgoing OAuth request, mark it as OAuth traffic and bound each network phase."""
    url = str(request.url)
    await check_oauth_endpoint_async(url, server_url)
    request.extensions = {**request.extensions, OAUTH_REQUEST_EXTENSION: (server_url, url),
                          "timeout": dict.fromkeys(("connect", "read", "write", "pool"), OAUTH_FETCH_TIMEOUT)}


async def guard_oauth_hop(request: Any) -> bool:
    """At the transport: True when *request* is marked OAuth traffic. A hop whose URL the auth flow did
    not check (a redirect the client follows on its own) passes the endpoint rules here first."""
    mark = request.extensions.get(OAUTH_REQUEST_EXTENSION)
    if mark is None:
        return False
    server_url, checked_url = mark
    if str(request.url) != checked_url:
        await check_oauth_endpoint_async(str(request.url), server_url)
    return True


def cap_oauth_response(httpx_mod: Any, response: Any, url: str) -> None:
    """Bound the not-yet-read body of an OAuth *response* by size and wall-clock time; reading past
    either raises :class:`OAuthEndpointRefused`. ``httpx_mod`` is the httpx flavour that built the
    response (httpx or the SDK's httpx2: each checks ``isinstance`` against its own stream base)."""
    declared = response.headers.get("content-length", "")
    # An oversized declaration is refused on the first read, inside the reader's own flow.
    oversized = declared.isdigit() and int(declared) > OAUTH_MAX_RESPONSE_BYTES
    response.stream = _oauth_stream_cls(httpx_mod)(
        response.stream, url, time.monotonic() + OAUTH_FETCH_TIMEOUT, oversized)


@functools.cache
def _oauth_stream_cls(httpx_mod: Any) -> type:
    class _CappedOAuthStream(httpx_mod.AsyncByteStream):
        def __init__(self, inner, url: str, deadline: float, oversized: bool):
            self._inner, self._url, self._deadline, self._oversized = inner, url, deadline, oversized

        async def __aiter__(self):
            import anyio
            if self._oversized:
                raise _refuse(self._url, f"its response exceeds {OAUTH_MAX_RESPONSE_BYTES} bytes")
            total, chunks = 0, self._inner.__aiter__()
            while True:
                try:
                    with anyio.fail_after(max(self._deadline - time.monotonic(), 0)):
                        chunk = await chunks.__anext__()
                except StopAsyncIteration:
                    return
                except TimeoutError:
                    raise _refuse(self._url, f"its response took longer than {OAUTH_FETCH_TIMEOUT:g}s") from None
                total += len(chunk)
                if total > OAUTH_MAX_RESPONSE_BYTES:
                    raise _refuse(self._url, f"its response exceeds {OAUTH_MAX_RESPONSE_BYTES} bytes")
                yield chunk

        async def aclose(self):
            await self._inner.aclose()

    return _CappedOAuthStream


def oauth_client_event_hooks(httpx_mod: Any, server_url: str | None) -> dict[str, list]:
    """``event_hooks=`` for an *httpx_mod* client Hermes owns for OAuth traffic: every request (redirect
    hops included: httpx runs the hooks per hop) is checked and every response capped before any read."""
    async def _on_request(request):
        await guard_oauth_request(request, server_url)

    async def _on_response(response):
        cap_oauth_response(httpx_mod, response, str(response.request.url))

    return {"request": [_on_request], "response": [_on_response]}
