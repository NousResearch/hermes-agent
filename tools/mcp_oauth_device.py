#!/usr/bin/env python3
"""RFC 8628 device-code flow for MCP OAuth servers (device authorization grant).

The SDK's ``OAuthClientProvider`` implements browser authorization-code PKCE only, so MCP
servers that require the ``device_code`` grant can never complete login from Hermes (#104742).
This module implements the device flow alongside the SDK rather than through it: dynamic client
registration (RFC 7591) requesting the device grant, device authorization + user-code display
(RFC 8628 §3.1), polling the token endpoint (§3.5), and persistence through
``HermesTokenStorage`` so refresh and cold-load reuse the existing machinery.

Opt-in per server so the browser path stays byte-identical unless asked::

    mcp_servers:
      <name>:
        url: https://...
        auth: oauth
        oauth:
          flow: device        # or: hermes mcp login <name> --flow device

HTTP is stdlib ``urllib`` behind an injectable ``http_post`` seam (no new dependencies, no
network in tests). Secrets never hit logs: errors name the endpoint and the RFC error code,
never the device code, user code, or tokens.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)

#: The grant type this module exists for (RFC 8628 §3.5).
DEVICE_CODE_GRANT = "urn:ietf:params:oauth:grant-type:device_code"

#: RFC 8628 §3.2: poll no faster than this when the server omits ``interval``.
DEFAULT_POLL_INTERVAL = 5.0

#: RFC 8628 §3.5: ``slow_down`` adds five seconds between polls.
SLOW_DOWN_INCREMENT = 5.0

#: Failure modes that end polling immediately (RFC 8628 §3.5).
_TERMINAL_TOKEN_ERRORS = frozenset({"access_denied", "expired_token"})

#: Retryable polling responses (RFC 8628 §3.5).
_PENDING_TOKEN_ERRORS = frozenset({"authorization_pending", "slow_down"})


class DeviceFlowError(RuntimeError):
    """Device flow failed (registration, authorization, polling, or persistence)."""


def _redacted_error(payload: dict) -> str:
    """``error`` + ``error_description`` from a payload; never echoes codes or tokens."""
    error = payload.get("error", "unknown_error")
    description = payload.get("error_description", "")
    return f"{error}: {description}" if description else str(error)


def _default_http_post(url: str, data: dict[str, str], *, timeout: float = 30.0) -> dict:
    """POST form-encoded *data* to *url*, returning the decoded JSON body."""
    body = urllib.parse.urlencode(data).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except OSError as exc:
        raise DeviceFlowError(f"device flow request to {url} failed: {exc}") from exc
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise DeviceFlowError(f"device flow endpoint {url} returned non-JSON") from exc
    if not isinstance(parsed, dict):
        raise DeviceFlowError(f"device flow endpoint {url} returned a non-object")
    return parsed


@dataclass
class DeviceAuthorization:
    """A pending device authorization (RFC 8628 §3.2 response)."""

    device_code: str
    user_code: str
    verification_uri: str
    verification_uri_complete: str | None = None
    expires_in: int = 1800
    interval: float = DEFAULT_POLL_INTERVAL


def request_device_authorization(
    device_authorization_endpoint: str,
    client_id: str,
    *,
    scope: str | None = None,
    resource: str | None = None,
    http_post: Callable[..., dict] | None = None,
) -> DeviceAuthorization:
    """Start a device authorization (RFC 8628 §3.1); raises ``DeviceFlowError`` on refusal."""
    payload: dict[str, str] = {"client_id": client_id}
    if scope:
        payload["scope"] = scope
    if resource:
        payload["resource"] = resource
    response = (http_post or _default_http_post)(device_authorization_endpoint, payload)
    if "device_code" not in response or "user_code" not in response or "verification_uri" not in response:
        raise DeviceFlowError(
            "device authorization refused: " + _redacted_error(response)
            if response.get("error")
            else "device authorization response is missing device_code, user_code, or verification_uri"
        )
    try:
        interval = float(response.get("interval", DEFAULT_POLL_INTERVAL))
        expires_in = int(response.get("expires_in", 1800))
    except (TypeError, ValueError) as exc:
        raise DeviceFlowError("device authorization response has non-numeric interval/expires_in") from exc
    return DeviceAuthorization(
        device_code=str(response["device_code"]),
        user_code=str(response["user_code"]),
        verification_uri=str(response["verification_uri"]),
        verification_uri_complete=response.get("verification_uri_complete"),
        expires_in=max(expires_in, 1),
        interval=max(interval, 1.0),
    )


def poll_device_token(
    token_endpoint: str,
    client_id: str,
    device_code: str,
    *,
    interval: float = DEFAULT_POLL_INTERVAL,
    expires_in: int = 1800,
    timeout: float | None = None,
    http_post: Callable[..., dict] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> dict:
    """Poll the token endpoint until approval, denial, or expiry (RFC 8628 §3.5).

    Returns the token payload (access_token + optional refresh_token/expires_in).
    ``sleep``/``http_post`` are injectable so tests never wait on a clock or a socket.
    """
    post = http_post or _default_http_post
    do_sleep = sleep or time.sleep
    deadline = time.monotonic() + min(expires_in, timeout if timeout is not None else expires_in)
    current_interval = max(interval, 1.0)
    while True:
        if time.monotonic() >= deadline:
            raise DeviceFlowError("device authorization expired before approval")
        response = post(token_endpoint, {
            "grant_type": DEVICE_CODE_GRANT,
            "device_code": device_code,
            "client_id": client_id,
        })
        if response.get("access_token"):
            return response
        error = str(response.get("error", ""))
        if error in _TERMINAL_TOKEN_ERRORS:
            raise DeviceFlowError("device authorization failed: " + _redacted_error(response))
        if error in _PENDING_TOKEN_ERRORS:
            if error == "slow_down":
                current_interval += SLOW_DOWN_INCREMENT
            do_sleep(current_interval)
            continue
        raise DeviceFlowError(
            "device token poll failed: " + _redacted_error(response) if error
            else "device token endpoint returned neither tokens nor a recognized error"
        )


def register_device_client(
    registration_endpoint: str,
    *,
    client_name: str = "Hermes Agent",
    scope: str | None = None,
    redirect_uris: list[str] | None = None,
    http_post: Callable[..., dict] | None = None,
) -> dict:
    """Dynamic client registration requesting the device grant (RFC 7591).

    Returns the raw client-info dict (client_id plus whatever the server echoes). Device-first
    servers often need no redirect URIs at all; they are sent only when provided.
    """
    payload: dict[str, Any] = {
        "client_name": client_name,
        "grant_types": ["authorization_code", "refresh_token", DEVICE_CODE_GRANT],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",
        "application_type": "native",
    }
    if scope:
        payload["scope"] = scope
    if redirect_uris:
        payload["redirect_uris"] = redirect_uris
    post = http_post or _default_http_post
    # Registration speaks JSON while the device/token endpoints speak form-encoded.
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        registration_endpoint, data=body, method="POST",
        headers={"Content-Type": "application/json"},
    ) if http_post is None else None
    if request is not None:
        try:
            with urllib.request.urlopen(request, timeout=30.0) as response:
                parsed = json.loads(response.read().decode("utf-8"))
        except OSError as exc:
            raise DeviceFlowError(f"client registration at {registration_endpoint} failed: {exc}") from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise DeviceFlowError(f"registration endpoint {registration_endpoint} returned non-JSON") from exc
    else:
        parsed = post(registration_endpoint, payload)
    if not isinstance(parsed, dict) or not parsed.get("client_id"):
        raise DeviceFlowError(
            "client registration failed: " + _redacted_error(parsed)
            if isinstance(parsed, dict) and parsed.get("error")
            else f"registration endpoint {registration_endpoint} returned no client_id"
        )
    return parsed


def server_supports_device_flow(oauth_metadata: Any) -> bool:
    """True when authorization-server metadata advertises a device endpoint.

    Accepts the SDK's ``OAuthMetadata`` model or a plain dict (ASM JSON).
    """
    endpoint = None
    if isinstance(oauth_metadata, dict):
        endpoint = oauth_metadata.get("device_authorization_endpoint")
    else:
        endpoint = getattr(oauth_metadata, "device_authorization_endpoint", None)
    return bool(endpoint)


def announce_device_authorization(authorization: DeviceAuthorization) -> None:
    """Show the user code + verification link (stderr, like the browser-flow announcer).

    A GUI-driven flow publishes the link to the dashboard instead of printing; the poll loop
    runs either way because approval happens on another device.
    """
    try:
        from tools.mcp_dashboard_oauth import get_dashboard_oauth_flow
        dashboard_flow = get_dashboard_oauth_flow()
    except (ImportError, AttributeError, ValueError):
        dashboard_flow = None
    link = authorization.verification_uri_complete or authorization.verification_uri
    if dashboard_flow is not None:
        import asyncio as _asyncio

        async def _publish() -> None:
            await dashboard_flow.publish_authorization_url(link)

        try:
            _asyncio.run(_publish())
            return
        except (RuntimeError, OSError, ValueError) as exc:
            logger.debug("dashboard device-code publish failed, falling back to stderr: %s", exc)
    print(
        "\n  MCP OAuth (device flow): open this URL on any device and enter the code:\n"
        f"\n    {link}\n"
        f"\n  Code: {authorization.user_code}\n"
        "\n  Waiting for approval (this times out when the code expires)...\n",
        file=sys.stderr,
    )


async def _persist_device_state(
    server_name: str,
    client_info: dict,
    token_payload: dict,
    *,
    hermes_home: Any = None,
) -> None:
    """Persist client registration + tokens through ``HermesTokenStorage``.

    Reuses the SDK token model so refresh, expiry seeding, and cold-load all work unchanged;
    a payload the model rejects raises ``DeviceFlowError`` naming fields, never values.
    """
    from tools.mcp_oauth import HermesTokenStorage, _model_json, _sdk_class, _write_json

    storage = HermesTokenStorage(server_name, hermes_home=hermes_home)
    token_cls = _sdk_class("OAuthToken")
    client_cls = _sdk_class("OAuthClientInformationFull")
    if token_cls is None or client_cls is None:
        raise DeviceFlowError("MCP OAuth SDK types are unavailable — cannot persist device-flow state")
    try:
        tokens = token_cls.model_validate(token_payload)
    except (ValueError, TypeError) as exc:
        detail = "validation failed"
        if hasattr(exc, "errors"):
            try:
                detail = "validation failed for " + ", ".join(
                    ".".join(map(str, e.get("loc", ()))) for e in exc.errors(include_input=False))
            except (TypeError, ValueError, AttributeError):
                pass
        raise DeviceFlowError(f"device token payload rejected: {detail}") from exc
    await storage.set_tokens(tokens)
    try:
        client_model = client_cls.model_validate(client_info)
    except (ValueError, TypeError) as exc:
        raise DeviceFlowError("device client registration rejected by the token model") from exc
    _write_json(storage._client_info_path(), _model_json(client_model))
    logger.debug("device-flow OAuth state saved for %s", server_name)


def persist_device_state(
    server_name: str,
    client_info: dict,
    token_payload: dict,
    *,
    hermes_home: Any = None,
) -> None:
    """Sync wrapper for login/CLI contexts (the storage API is async)."""
    asyncio.run(_persist_device_state(
        server_name, client_info, token_payload, hermes_home=hermes_home))


def _device_endpoints_from(payload: dict, asm: Any) -> dict[str, str] | None:
    """``(device, token, registration)`` endpoints from raw ASM + SDK model, or None.

    Split out for tests: the pinned SDK's ``OAuthMetadata`` drops ``device_authorization_endpoint``,
    so the device endpoint MUST come from the raw document — this function is where that invariant lives.
    """
    device_endpoint = str(
        payload.get("device_authorization_endpoint")
        or getattr(asm, "device_authorization_endpoint", None) or "")
    token_endpoint = str(getattr(asm, "token_endpoint", None) or "")
    registration_endpoint = str(getattr(asm, "registration_endpoint", None) or "")
    if not device_endpoint or not token_endpoint or not registration_endpoint:
        return None
    return {
        "device_authorization_endpoint": device_endpoint,
        "token_endpoint": token_endpoint,
        "registration_endpoint": registration_endpoint,
    }


def discover_device_endpoints(server_url: str, *, timeout: float = 10.0) -> dict[str, str]:
    """PRM + ASM discovery returning device/token/registration endpoints and resource.

    Uses the SDK's own URL builders and response handlers so Hermes tracks whatever the pinned
    SDK expects; raises ``DeviceFlowError`` when the server advertises no device endpoint.
    """
    from mcp.client.auth.utils import (
        build_oauth_authorization_server_metadata_discovery_urls,
        build_protected_resource_metadata_discovery_urls,
        handle_auth_metadata_response,
        handle_protected_resource_response,
    )
    from tools.mcp_tool import sdk_httpx

    httpx = sdk_httpx()
    if httpx is None:
        raise DeviceFlowError("MCP OAuth SDK HTTP layer is unavailable")

    def _get(url: str) -> Any:
        request = urllib.request.Request(url, method="GET",
                                         headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8")), response.status
        except OSError as exc:
            raise DeviceFlowError(f"OAuth discovery to {url} failed: {exc}") from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise DeviceFlowError(f"OAuth discovery at {url} returned non-JSON") from exc

    resource: str | None = None
    auth_server_url: str | None = None
    for url in build_protected_resource_metadata_discovery_urls(None, server_url):
        try:
            payload, status = _get(url)
        except DeviceFlowError:
            continue
        prm = _handle_prm(payload, status, handle_protected_resource_response)
        if prm is None:
            continue
        resource = getattr(prm, "resource", None) and str(prm.resource) or server_url
        servers = getattr(prm, "authorization_servers", None) or []
        if servers:
            auth_server_url = str(servers[0])
        break
    for url in build_oauth_authorization_server_metadata_discovery_urls(auth_server_url, server_url):
        try:
            payload, status = _get(url)
        except DeviceFlowError:
            continue
        ok, asm = _handle_asm(payload, status, handle_auth_metadata_response)
        if not ok:
            break
        if asm is None:
            continue
        endpoints = _device_endpoints_from(payload, asm)
        if endpoints is None:
            raise DeviceFlowError(
                "server metadata is missing device, token, or registration endpoints")
        endpoints["resource"] = resource or server_url
        return endpoints
    raise DeviceFlowError("OAuth metadata discovery failed: no authorization server found")


def _handle_prm(payload: dict, status: int, handler: Callable) -> Any:
    """Adapt a stdlib-fetched PRM document to the SDK's async response handler."""
    return asyncio.run(_adapt_metadata_response(payload, status, handler))


def _handle_asm(payload: dict, status: int, handler: Callable) -> Any:
    """Adapt a stdlib-fetched ASM document to the SDK's async response handler."""
    return asyncio.run(_adapt_metadata_response(payload, status, handler))


async def _adapt_metadata_response(payload: dict, status: int, handler: Callable) -> Any:
    """Run the SDK's metadata handler against a minimal response shim (no extra deps).

    The shim speaks the SDK handler's wire contract: ``status_code`` + ``aread()`` bytes.
    """

    class _Shim:
        status_code = status

        async def aread(self) -> bytes:
            return json.dumps(payload).encode("utf-8")

    try:
        return await handler(_Shim())
    except (TypeError, ValueError, AttributeError, KeyError) as exc:
        logger.debug("OAuth metadata handler rejected a document: %s", exc)
        return None
