"""Hosted artifacts: the Nous-managed ``hermes-artifacts`` MCP server.

A profile signed in to Nous whose ``/api/oauth/account`` body says ``artifacts_access: true`` and
``paid_service_access.paid_access: true`` gets one more MCP server, added to the configured
``mcp_servers`` when they are loaded (``tools.mcp_tool_config._load_mcp_config``). It connects with
``auth: nous``: every request carries a freshly resolved Nous access token, never a stored header.

The server joins a conversation only at its start. MCP discovery can finish after a session has
built its tool list, and the between-turns MCP refresh would then append the new tools mid-session;
:func:`latch_session_surface` records, per session, whether the server's tools were in that
session's first tool load (its definitions, or the Tool Search catalog of its own toolset selection),
and :func:`late_attach_drops` keeps them out of that session for good when they were not.

Every failure (config off, an insecure base URL, signed out, account unreadable, not entitled) leaves the
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
GATEWAY_VENDOR = "gateway"  # default origin: the managed gateway named like every vendor gateway
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
    """The ``artifacts`` config block: ``enabled`` (default on) and ``base_url`` (empty: the managed
    gateway, :func:`default_base_url`)."""

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


def default_base_url() -> str:
    """The managed artifacts gateway: ``{scheme}://gateway-gateway.{domain}``, built by the same helper
    and environment (``TOOL_GATEWAY_SCHEME``, ``TOOL_GATEWAY_DOMAIN``) as the managed tool gateways, so
    a non-production domain follows them. ``artifacts.base_url`` overrides it."""
    from tools.managed_tool_gateway import build_vendor_gateway_url

    return build_vendor_gateway_url(GATEWAY_VENDOR)


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


def resolve_mcp_url(config: ArtifactsConfig) -> tuple[Optional[str], Reason]:
    try:
        raw = config.base_url or default_base_url()
    except ValueError:  # TOOL_GATEWAY_SCHEME is neither http nor https
        return None, Reason.insecure_base_url
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


def decide(config: ArtifactsConfig, *, account_reader: Optional[Callable[[], Any]] = None) -> AttachDecision:
    """Whether this profile attaches the server, and why not. Never raises: any failure is a refusal."""
    if not config.enabled:
        return AttachDecision(Reason.disabled)
    try:
        url, reason = resolve_mcp_url(config)
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


def _endpoint(url: str) -> Optional[tuple[str, str, int, str]]:
    try:
        parts = urlsplit(url)
        port = parts.port or {"http": 80, "https": 443}.get(parts.scheme)
    except ValueError:
        return None
    host = (parts.hostname or "").lower()
    if not host or port is None:
        return None
    return parts.scheme, host, port, parts.path or "/"


def same_endpoint(request_url: str, permitted_url: str) -> bool:
    """True when *request_url* names exactly *permitted_url*: scheme, host, port and path (the query
    is ignored). A redirect elsewhere on the same origin is not the endpoint."""
    request = _endpoint(request_url)
    return request is not None and request == _endpoint(permitted_url)


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
        # The client follows redirects through this flow, so each request is checked again: only the
        # attached endpoint itself (same scheme, host, port and path) ever carries the token.
        def sync_auth_flow(self, request):
            if same_endpoint(str(request.url), url):
                request.headers["Authorization"] = f"Bearer {resolve_nous_access_token()}"
            else:
                request.headers.pop("Authorization", None)
            yield request

        async def async_auth_flow(self, request):
            if same_endpoint(str(request.url), url):
                token = await asyncio.to_thread(resolve_nous_access_token)
                request.headers["Authorization"] = f"Bearer {token}"
            else:
                request.headers.pop("Authorization", None)
            yield request

    return NousBearerAuth()



def _without_nulls(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _without_nulls(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_without_nulls(item) for item in value]
    return value


def call_arguments(server_name: str, args: Any) -> Any:
    """*args* as sent to *server_name*. The hosted server's input schemas are non-nullable and refuse
    a null, so for it a null a model wrote for an optional field (``"version": null``) means "not
    given" and is left out, at any depth. Every other server gets *args* unchanged."""
    if server_name != SERVER_NAME or not isinstance(args, dict):
        return args
    return _without_nulls(args)


def _registered_tool_names() -> frozenset[str]:
    from tools.registry import registry

    return frozenset(registry.get_tool_names_for_toolset(TOOLSET_NAME) or ())


def _artifact_names() -> frozenset[str]:
    return contract_tool_names() | _registered_tool_names()


def _catalog_names(agent: Any) -> frozenset[str]:
    """Every tool *agent*'s toolset selection offers before Tool Search collapses it into the bridge.
    Empty when that cannot be read (the latch then fails closed)."""
    try:
        import model_tools

        return frozenset((tool.get("function") or {}).get("name") for tool in model_tools.get_tool_definitions(
            enabled_toolsets=getattr(agent, "enabled_toolsets", None),
            disabled_toolsets=getattr(agent, "disabled_toolsets", None),
            quiet_mode=True, skip_tool_search_assembly=True) or [])
    except Exception as exc:  # health: allow BLE001 -- a catalog failure must only ever narrow the session
        logger.debug("Hosted artifacts: tool catalog unreadable at session start: %s", exc)
        return frozenset()


def _surface_names(agent: Any, tools: list[dict[str, Any]]) -> frozenset[str]:
    """Every tool name *tools* (a session's first tool load) offers: its definitions and, when the
    Tool Search bridge stands in for deferred tools, the deferred catalog of the session's selection."""
    from tools.tool_search_catalog import BRIDGE_TOOL_NAMES

    names = frozenset((tool.get("function") or {}).get("name") for tool in tools or ())
    return names | _catalog_names(agent) if names & BRIDGE_TOOL_NAMES else names


def _session_key(agent: Any) -> Optional[tuple[str, str]]:
    from hermes_constants import get_hermes_home

    session_id = getattr(agent, "session_id", None)
    return (str(get_hermes_home()), session_id) if isinstance(session_id, str) and session_id else None


def _store_latch(agent: Any, latched: bool) -> None:
    try:
        agent._hosted_artifacts_at_start = latched
    except AttributeError:
        pass


def _recorded_latch(agent: Any) -> Optional[bool]:
    latched = getattr(agent, "_hosted_artifacts_at_start", None)
    if isinstance(latched, bool):
        return latched
    key = _session_key(agent)
    with _decisions_lock:
        latched = _session_latch.get(key) if key is not None else None
    if latched is not None:
        _store_latch(agent, latched)
    return latched


def latch_session_surface(agent: Any, tools: list[dict[str, Any]]) -> None:
    """Record, once per session, whether *tools* (the session's first tool load, before any drop)
    carried the hosted artifacts tools. Later loads (a rebuilt agent, a toggled toolset,
    ``/reload-mcp``, a model switch) find the record and never widen it."""
    if agent is None or _recorded_latch(agent) is not None:
        return
    latched = bool(_surface_names(agent, tools) & _artifact_names())
    key = _session_key(agent)
    with _decisions_lock:
        if key is not None:
            latched = _session_latch.setdefault(key, latched)
            while len(_session_latch) > _SESSION_LATCH_MAX:
                _session_latch.popitem(last=False)
    _store_latch(agent, latched)


def late_attach_drops(agent: Any) -> frozenset[str]:
    """The hosted server's tool names unless *agent*'s session started with them. A session whose
    first tool load was never recorded (:func:`latch_session_surface`) gets none of them."""
    if agent is None or _recorded_latch(agent) is True:
        return frozenset()
    return _artifact_names()


def session_has_artifacts(tool_names: Any) -> bool:
    """True when a session's tool names include the hosted artifacts tools."""
    names = set(tool_names or ())
    return bool(names & contract_tool_names())



# ── Viewer links for surfaces that cannot render artifacts inline ─────────────────────────────

# Tools whose successful result names one artifact the user can open.
LINK_TOOLS = ("artifact_create", "artifact_update", "artifact_show", "artifact_get")
_TITLE_MAX_CHARS = 120


@dataclass(frozen=True)
class ViewerLink:
    """The shareable viewer page of one artifact, and its title when the result carried one."""

    url: str
    title: Optional[str] = None

    def as_payload(self) -> dict[str, str]:
        return {"url": self.url, **({"title": self.title} if self.title else {})}

    def text(self) -> str:
        """One plain line naming the artifact and its viewer page."""
        return f"Artifact \u201c{self.title}\u201d: {self.url}" if self.title else f"Artifact: {self.url}"


def _printable(value: str) -> bool:
    return all(ch.isprintable() and not ch.isspace() for ch in value)


def _clean_title(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    title = " ".join("".join(ch if ch.isprintable() else " " for ch in value).split())
    if len(title) > _TITLE_MAX_CHARS:
        title = title[:_TITLE_MAX_CHARS - 1].rstrip() + "\u2026"
    return title or None


def _result_bodies(result: Any) -> list[Any]:
    """The JSON objects a tool result may carry the artifact in: the handler envelope itself, its
    ``result`` (JSON text or an object) and its ``structuredContent``."""
    import json

    def _load(value: Any) -> Any:
        if isinstance(value, str):
            try:
                return json.loads(value)
            except ValueError:
                return None
        return value

    envelope = _load(result)
    if not isinstance(envelope, dict):
        return []
    return [envelope, _load(envelope.get("result")), envelope.get("structuredContent")]


def viewer_link(tool_name: Any, result: Any) -> Optional[ViewerLink]:
    """The viewer link of a successful create/update/show/get result, else None.

    Only ``viewer_url`` is ever returned: ``content_url`` is a short-lived ticket for an embedded
    frame and must not be printed or sent. A result that reports an error, a URL that is not a
    plain http(s) link (or that equals the ticket URL) and anything unparsable yield None."""
    from tools.mcp_tool_schema import mcp_prefixed_tool_name

    if not isinstance(tool_name, str) or tool_name not in {
            mcp_prefixed_tool_name(SERVER_NAME, tool) for tool in LINK_TOOLS}:
        return None
    for body in _result_bodies(result):
        if not isinstance(body, dict):
            continue
        if "error" in body:
            return None
        url = body.get("viewer_url")
        if (isinstance(url, str) and _printable(url) and _origin(url) is not None
                and url != body.get("content_url")):
            return ViewerLink(url, _clean_title(body.get("title")))
    return None
