"""Hosted artifacts: the Nous-managed ``hermes-artifacts`` MCP server.

A profile signed in to Nous whose ``/api/oauth/account`` body says ``artifacts_access: true`` and
``paid_service_access.paid_access: true`` gets one more MCP server, added to the configured
``mcp_servers`` when they are loaded (``tools.mcp_tool_config._load_mcp_config``). It connects with
``auth: nous``: every request carries a freshly resolved Nous access token, never a stored header.

The server joins a conversation only at its start. MCP discovery can finish after a session has
built its tool list, and the between-turns MCP refresh would then append the new tools mid-session;
:func:`late_attach_drops` latches, per session, whether the server was registered at the first tool
load and keeps its tools out of that session for good when it was not.

Every failure (config off, no base URL, signed out, account unreadable, not entitled) leaves the
server out; the gateway stays authoritative for entitlement on every call.

``HOSTED_ARTIFACTS_CLIENT_VERSION`` is the stable probe external end-to-end harnesses use to detect
that this client supports hosted artifacts.
"""

from __future__ import annotations

import enum
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Optional
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

# Probe for cross-repository end-to-end tests: present (and >= 1) when this client can attach the
# hosted artifacts MCP server. Bumped only when the client's side of the wire contract changes.
HOSTED_ARTIFACTS_CLIENT_VERSION = 1

SERVER_NAME = "hermes-artifacts"
MCP_PATH = "/v1/artifacts/mcp"
VIEWER_RESOURCE_URI = "ui://hermes/artifact-viewer"
AUTH_TYPE = "nous"
TOOLSET_NAME = f"mcp-{SERVER_NAME}"
CONTRACT_TOOLS = ("artifact_create", "artifact_update", "artifact_get", "artifact_list",
                  "artifact_share", "artifact_show")

# One account read per profile per window: the MCP config is loaded by discovery, reconcile and
# status reads alike, and none of them may turn into a Portal round trip each.
DECISION_TTL_SECONDS = 300.0

_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_FALSE_STRINGS = frozenset({"false", "0", "no", "off"})


def contract_tool_names() -> frozenset[str]:
    """Registry names of the contract's tools (``mcp__hermes_artifacts__artifact_show`` ...)."""
    from tools.mcp_tool_schema import mcp_prefixed_tool_name

    return frozenset(mcp_prefixed_tool_name(SERVER_NAME, tool) for tool in CONTRACT_TOOLS)


class Reason(str, enum.Enum):
    attached = "attached"
    disabled = "disabled"
    no_base_url = "no_base_url"
    insecure_base_url = "insecure_base_url"
    not_logged_in = "not_logged_in"
    account_unreadable = "account_unreadable"
    not_entitled = "not_entitled"
    not_enabled = "not_enabled"


@dataclass(frozen=True)
class AttachDecision:
    reason: Reason
    url: Optional[str] = None

    @property
    def attach(self) -> bool:
        return self.reason is Reason.attached


@dataclass(frozen=True)
class ArtifactsConfig:
    """The ``artifacts`` config block: ``enabled`` (default on) and ``base_url`` (default: the relay
    connector origin when one is configured, else unset, which leaves the server out)."""

    enabled: bool = True
    base_url: str = ""

    @classmethod
    def from_raw(cls, raw: Any) -> ArtifactsConfig:
        if not isinstance(raw, dict):
            return cls()
        enabled = raw.get("enabled", True)
        if isinstance(enabled, str):
            enabled = enabled.strip().lower() not in _FALSE_STRINGS
        base_url = raw.get("base_url")
        return cls(enabled=enabled is not False and bool(enabled),
                   base_url=base_url.strip() if isinstance(base_url, str) else "")


def _relay_origin() -> str:
    """The relay connector's ``http(s)`` origin when this profile has one configured, else ``""``."""
    from gateway.relay import relay_url
    from gateway.relay.media import media_base_url

    dial = relay_url()
    return media_base_url(dial) if dial else ""


def _origin(url: str) -> Optional[str]:
    """``scheme://host[:port]`` of *url* when it may carry a bearer: https anywhere, http only to a
    loopback or ``*.localhost`` host. None otherwise."""
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return None
    host = (parts.hostname or "").lower()
    if not host or parts.scheme not in ("http", "https"):
        return None
    if parts.scheme == "http" and host not in _LOCAL_HOSTS and not host.endswith(".localhost"):
        return None
    return f"{parts.scheme}://{parts.netloc}"


def resolve_mcp_url(config: ArtifactsConfig, relay_origin: Optional[Callable[[], str]] = None,
                    ) -> tuple[Optional[str], Reason]:
    raw = config.base_url or (relay_origin or _relay_origin)()
    if not raw:
        return None, Reason.no_base_url
    origin = _origin(raw)
    if origin is None:
        return None, Reason.insecure_base_url
    return origin + MCP_PATH, Reason.attached


def _read_account() -> Any:
    """The fresh ``/api/oauth/account`` read (a JWT snapshot carries no ``artifacts_access``)."""
    from hermes_cli.nous_account import get_nous_portal_account_info

    return get_nous_portal_account_info(force_fresh=True)


def entitlement(account: Any) -> Reason:
    """``attached`` only for a fresh account body with both flags literally true."""
    if account is None or not getattr(account, "logged_in", False):
        return Reason.not_logged_in
    if getattr(account, "source", None) != "account_api":
        return Reason.account_unreadable
    raw = getattr(account, "raw_account", None) or {}
    access = getattr(account, "paid_service_access_info", None)
    if getattr(access, "paid_access", None) is not True:
        return Reason.not_entitled
    if raw.get("artifacts_access") is not True:
        return Reason.not_enabled
    return Reason.attached


def decide(config: ArtifactsConfig, *, account_reader: Optional[Callable[[], Any]] = None,
           relay_origin: Optional[Callable[[], str]] = None) -> AttachDecision:
    """Whether this profile attaches the server, and why not. Never raises: any failure is a refusal."""
    if not config.enabled:
        return AttachDecision(Reason.disabled)
    try:
        url, reason = resolve_mcp_url(config, relay_origin)
        if url is None:
            return AttachDecision(reason)
        reason = entitlement((account_reader or _read_account)())
    except Exception as exc:  # health: allow BLE001 -- the account read spans auth refresh, HTTP and JSON; any failure is a refusal
        logger.debug("Hosted artifacts attach check failed: %s", exc)
        return AttachDecision(Reason.account_unreadable)
    return AttachDecision(reason, url if reason is Reason.attached else None)


_decisions: dict[str, tuple[float, AttachDecision]] = {}
_decisions_lock = threading.Lock()

# (profile home, session id) -> whether the server's tools were in that session's first tool load.
# Survives an agent rebuild for the same conversation in this process. Bounded, oldest first.
_session_latch: OrderedDict[tuple[str, str], bool] = OrderedDict()
_SESSION_LATCH_MAX = 4096


def reset_decisions() -> None:
    """Forget every remembered attach decision and session latch."""
    with _decisions_lock:
        _decisions.clear()
        _session_latch.clear()


def cached_decision(config: ArtifactsConfig, *, now: Callable[[], float] = time.monotonic,
                    **kwargs: Any) -> AttachDecision:
    """:func:`decide`, remembered per profile home and config for :data:`DECISION_TTL_SECONDS`."""
    from hermes_constants import get_hermes_home

    key = f"{get_hermes_home()}|{config.enabled}|{config.base_url}"
    with _decisions_lock:
        hit = _decisions.get(key)
    if hit is not None and now() - hit[0] < DECISION_TTL_SECONDS:
        return hit[1]
    decision = decide(config, **kwargs)
    with _decisions_lock:
        _decisions[key] = (now(), decision)
    if not decision.attach:
        logger.debug("Hosted artifacts MCP server not attached: %s", decision.reason.value)
    return decision


def attached_server_names(root_config: dict) -> frozenset[str]:
    """``{SERVER_NAME}`` when this profile's remembered decision attached the server, else empty. Never
    reads the account: the decision is the one MCP config loading made (a toolset resolver must not
    turn into a Portal round trip), so before that load the server is simply not enabled yet."""
    from hermes_constants import get_hermes_home

    config = ArtifactsConfig.from_raw((root_config or {}).get("artifacts"))
    key = f"{get_hermes_home()}|{config.enabled}|{config.base_url}"
    with _decisions_lock:
        hit = _decisions.get(key)
    return frozenset({SERVER_NAME}) if hit is not None and hit[1].attach else frozenset()


def server_entry(url: str) -> dict[str, Any]:
    """The ``mcp_servers`` entry for the hosted server."""
    return {"url": url, "auth": AUTH_TYPE}


def _profile_config() -> ArtifactsConfig:
    from hermes_cli.config import load_config

    return ArtifactsConfig.from_raw(load_config().get("artifacts"))


def attach_to(servers: dict[str, dict], root_config: dict) -> None:
    """Add the hosted server to *servers* (the loaded ``mcp_servers``) when this profile is entitled.
    It replaces any same-named entry, so a plugin or config entry cannot stand in for it."""
    decision = cached_decision(ArtifactsConfig.from_raw(root_config.get("artifacts")))
    if not (decision.attach and decision.url):
        return
    if SERVER_NAME in servers:
        logger.warning("MCP server '%s' is managed by Hermes; the configured entry of that name is ignored",
                       SERVER_NAME)
    servers[SERVER_NAME] = server_entry(decision.url)


def bearer_allowed(url: str) -> bool:
    """``auth: nous`` hands out the Nous access token, so only the attached hosted server URL gets it."""
    decision = cached_decision(_profile_config())
    return decision.attach and decision.url == url


def nous_bearer_auth(url: str):
    """An ``httpx.Auth`` that sets ``Authorization: Bearer <Nous access token>`` on each request,
    resolving (and refreshing) the token off the event loop. Refuses any URL but the attached
    hosted server's, so another MCP entry declaring ``auth: nous`` cannot receive the token."""
    import asyncio

    from hermes_cli.auth import resolve_nous_access_token
    from tools.mcp_tool import sdk_httpx

    # The SDK's httpx flavour (httpx2 on mcp >= 2.0): its AsyncClient rejects any other Auth class.
    httpx = sdk_httpx()

    if not bearer_allowed(url):
        raise ValueError(f"auth: {AUTH_TYPE} is reserved for the managed {SERVER_NAME} server")

    class NousBearerAuth(httpx.Auth):
        def sync_auth_flow(self, request):
            request.headers["Authorization"] = f"Bearer {resolve_nous_access_token()}"
            yield request

        async def async_auth_flow(self, request):
            token = await asyncio.to_thread(resolve_nous_access_token)
            request.headers["Authorization"] = f"Bearer {token}"
            yield request

    return NousBearerAuth()



def _registered_tool_names() -> frozenset[str]:
    from tools.registry import registry

    return frozenset(registry.get_tool_names_for_toolset(TOOLSET_NAME) or ())


def _latch(agent: Any) -> bool:
    latched = getattr(agent, "_hosted_artifacts_at_start", None)
    if isinstance(latched, bool):
        return latched
    from hermes_constants import get_hermes_home

    session_id = getattr(agent, "session_id", None)
    key = (str(get_hermes_home()), session_id) if isinstance(session_id, str) and session_id else None
    with _decisions_lock:
        if key is not None and key in _session_latch:
            latched = _session_latch[key]
        else:
            latched = bool(_registered_tool_names())
            if key is not None:
                _session_latch[key] = latched
                while len(_session_latch) > _SESSION_LATCH_MAX:
                    _session_latch.popitem(last=False)
    try:
        agent._hosted_artifacts_at_start = latched
    except AttributeError:
        pass
    return latched


def late_attach_drops(agent: Any) -> frozenset[str]:
    """The hosted server's tool names when *agent*'s session started without them, else empty."""
    if agent is None or _latch(agent):
        return frozenset()
    return contract_tool_names() | _registered_tool_names()


def session_has_artifacts(tool_names: Any) -> bool:
    """True when a session's tool names include the hosted artifacts tools."""
    names = set(tool_names or ())
    return bool(names & contract_tool_names())
