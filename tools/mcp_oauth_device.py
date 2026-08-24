"""Explicit RFC 8628 MCP login, sharing SDK discovery, client auth and token storage.

The SDK still owns runtime requests and refresh. Device authorization is only
started by `hermes mcp login/reauth`, never a background reconnect.
"""
from __future__ import annotations

import asyncio
import math
import sys
import time

from mcp.shared.auth import OAuthMetadata
from pydantic import AnyHttpUrl

DEVICE_GRANT = "urn:ietf:params:oauth:grant-type:device_code"


class DeviceOAuthMetadata(OAuthMetadata):
    # RFC 8414 makes authorization_endpoint optional for grants not using it.
    authorization_endpoint: AnyHttpUrl | None = None
    device_authorization_endpoint: AnyHttpUrl


async def _discover(client, provider):
    from mcp.client.auth.utils import (
        build_oauth_authorization_server_metadata_discovery_urls,
        build_protected_resource_metadata_discovery_urls,
        extract_resource_metadata_from_www_auth,
        handle_protected_resource_response,
        validate_metadata_issuer,
    )
    context = provider.context
    response = await client.get(context.server_url)
    challenge = extract_resource_metadata_from_www_auth(response)
    for url in build_protected_resource_metadata_discovery_urls(challenge, context.server_url):
        response = await client.get(url)
        prm = await handle_protected_resource_response(response)
        if prm:
            await provider._validate_resource_match(prm)
            context.protected_resource_metadata = prm
            context.auth_server_url = str(prm.authorization_servers[0])
            break
    for url in build_oauth_authorization_server_metadata_discovery_urls(context.auth_server_url, context.server_url):
        response = await client.get(url)
        if response.status_code == 404:
            continue
        data = _payload(response, "OAuth metadata")
        if not data.get("device_authorization_endpoint"):
            raise RuntimeError("Server does not advertise device authorization; use --flow browser if supported")
        metadata = DeviceOAuthMetadata.model_validate(data)
        if context.auth_server_url:
            validate_metadata_issuer(metadata, context.auth_server_url)
        grants = metadata.grant_types_supported
        if grants is not None and DEVICE_GRANT not in grants:
            raise RuntimeError("Server does not advertise the device_code grant")
        context.oauth_metadata = metadata
        return
    raise RuntimeError("No OAuth authorization server metadata found")


def _payload(response, label):
    try:
        data = response.json()
    except ValueError:
        raise RuntimeError(f"{label}: invalid JSON response") from None
    if not isinstance(data, dict):
        raise RuntimeError(f"{label}: expected a JSON object")
    if not 200 <= response.status_code < 300:
        # Descriptions and arbitrary error values may contain credentials.
        raise RuntimeError(f"{label} failed (HTTP {response.status_code})")
    return data


async def _register(client, provider, cfg):
    from mcp.shared.auth import OAuthClientInformationFull
    from mcp.client.auth.oauth2 import OAuthRegistrationError, check_registration_usable

    context = provider.context
    metadata = context.client_metadata.model_dump(mode="json", exclude_none=True)
    metadata.update(grant_types=[DEVICE_GRANT, "refresh_token"], response_types=[])
    if cfg.get("client_id"):
        data = {**metadata, "client_id": cfg["client_id"]}
        if cfg.get("client_secret"):
            data["client_secret"] = cfg["client_secret"]
    else:
        endpoint = context.oauth_metadata.registration_endpoint
        if not endpoint:
            raise RuntimeError("Server has no registration endpoint; configure oauth.client_id (and client_secret if required)")
        response = await client.post(str(endpoint), json=metadata)
        data = _payload(response, "Client registration")
    data["issuer"] = str(context.oauth_metadata.issuer)
    context.client_info = OAuthClientInformationFull.model_validate(data)
    provider._coerce_client_secret_post()
    try:
        check_registration_usable(context.client_info)
    except OAuthRegistrationError:
        raise RuntimeError("Device OAuth client has unsupported or incomplete token endpoint authentication") from None


def _positive_seconds(value, label):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError(f"Device authorization has invalid {label}")
    return value


async def _authorize(client, provider, cfg):
    from tools.mcp_tool import sdk_httpx

    context = provider.context
    resource = context.get_resource_url()
    data = {"client_id": context.client_info.client_id, "resource": resource}
    if context.client_metadata.scope:
        data["scope"] = context.client_metadata.scope
    data, headers = context.prepare_token_auth(data, {})
    response = await client.post(str(context.oauth_metadata.device_authorization_endpoint), data=data, headers=headers)
    authorization = _payload(response, "Device authorization")
    for key in ("device_code", "user_code", "verification_uri"):
        if not isinstance(authorization.get(key), str) or not authorization[key]:
            raise RuntimeError(f"Device authorization is missing {key}")
    verification = AnyHttpUrl(authorization["verification_uri"])
    interval = _positive_seconds(authorization.get("interval", 5), "interval")
    deadline = time.monotonic() + min(_positive_seconds(authorization["expires_in"], "expires_in"),
                                     _positive_seconds(cfg.get("timeout", 300), "timeout"))
    print(f"\n  MCP OAuth: open {verification} on any device.\n  Code: {authorization['user_code']}\n"
          "  Waiting for approval...\n", file=sys.stderr, flush=True)
    token_data = {"client_id": context.client_info.client_id, "device_code": authorization["device_code"],
                  "grant_type": DEVICE_GRANT, "resource": resource}
    token_data, headers = context.prepare_token_auth(token_data, {})
    httpx = sdk_httpx()
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= interval:
            raise RuntimeError("Device authorization expired before approval; run login again")
        await asyncio.sleep(interval)
        request = provider._prepare_token_request(httpx.Request("POST", str(context.oauth_metadata.token_endpoint),
                                                               data=token_data, headers=headers))
        try:
            response = await asyncio.wait_for(client.send(request), timeout=deadline - time.monotonic())
        except (TimeoutError, httpx.TimeoutException):
            # RFC 8628 requires reducing polling frequency after connection timeouts.
            interval *= 2
            continue
        if 200 <= response.status_code < 300:
            from mcp.shared.auth import OAuthToken
            tokens = OAuthToken.model_validate(_payload(response, "Device token"))
            if not tokens.access_token:
                raise RuntimeError("Device token response has no access token")
            if tokens.scope is None:
                tokens.scope = context.client_metadata.scope
            return tokens
        try:
            error = response.json().get("error")
        except (ValueError, AttributeError):
            error = None
        if error == "authorization_pending":
            continue
        if error == "slow_down":
            interval += 5
            continue
        safe_error = error if error in {"access_denied", "expired_token"} else f"HTTP {response.status_code}"
        raise RuntimeError(f"Device authorization failed: {safe_error}")


async def login_device(name, server_url, oauth_config):
    """Authorize then commit state in the active profile; failed grants preserve old state."""
    from tools.mcp_oauth import _build_client_metadata
    from tools.mcp_oauth_manager import HermesMCPOAuthProvider, get_manager
    from tools.mcp_oauth_provider import prepare_oauth_config
    from tools.mcp_tool import sdk_httpx

    cfg, storage = prepare_oauth_config(name, server_url, oauth_config)
    # Device flow never binds a callback socket or uses the hosted browser CIMD.
    cfg["_resolved_port"] = cfg.get("redirect_port", 8420)
    provider = HermesMCPOAuthProvider(server_url=server_url, server_name=name, storage=storage,
                                     client_metadata=_build_client_metadata(cfg),
                                     token_user_agent=cfg.get("user_agent"))
    httpx = sdk_httpx()
    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=False) as client:
            await _discover(client, provider)
            await _register(client, provider, cfg)
            tokens = await _authorize(client, provider, cfg)
    except (ValueError, TypeError, KeyError):
        raise RuntimeError("Device OAuth response has invalid fields") from None
    except httpx.HTTPError:
        raise RuntimeError("Device OAuth network request failed") from None
    # Validate the entire grant before touching disk; reuse the existing scoped store.
    previous = storage.snapshot()
    try:
        await storage.set_client_info(provider.context.client_info)
        storage.save_oauth_metadata(provider.context.oauth_metadata)
        await storage.set_tokens(tokens)
    except OSError:
        storage.restore(previous)
        raise
    get_manager().evict(name)
