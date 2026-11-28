"""Which credential an MCP request carries: the service account, or the signed-in user's token.

One MCP connection outlives every chat, and each request on it falls into one of two groups:

* **The connection's own traffic** — ``initialize``, ``tools/list``, the keepalive ``ping``, a
  schema refresh, and anything issued with no login driving it (CLI, cron, messaging). None of
  it is made on anyone's behalf; the server only needs to know *Hermes* is allowed to talk to
  it. It carries the **service account** (``tools.mcp_service_credentials``, an OAuth2
  ``client_credentials`` token), which the server scopes to a caller role with no ``kd_username``.
* **Work on a signed-in user's behalf** — a ``tools/call`` (or a ``resources/read`` /
  ``prompts/get``) issued while that user's turn is running. The server must authorise it as the
  person who asked, so it carries **that turn's own access token** (``agent.turn_token``), the
  one Keycloak minted for this user, with their ``kd_username``.

The distinction is made at *request* time by asking which turn is running, never by inspecting
the URL, the connection, or the RPC method: a connection is opened once and then used by
everybody, and the same method can be either kind of traffic. Replies depend only on the context
the request was issued in, and contexts are copied per turn, so two people in two tabs cannot
see each other's token. The serve task itself is created turn-less
(:meth:`tools.mcp_tool_server_run.MCPServerRunMixin.start`), so a lazily-connected server's
keepalives never inherit the credential of whoever triggered the connect.

Failure policy — the part that must not be softened (``auth_scope: user``):

* A user turn whose token is missing or expired **fails the call**. It never falls back to the
  service account: silently running a person's Kingdee query as the integration account would
  return data they are not entitled to, which is worse than an error. The tool handler refuses
  the call up front with a re-login message; if the token expires *during* the turn, after that
  check, the request goes out **bare** and the server's own 401/403 fails it loudly.
* That refusal happens in :meth:`CredentialPolicy.refusal`, which the tool handler calls *before*
  any transport work (:mod:`tools.mcp_tool_handlers`). It deliberately does NOT raise from the
  auth hook: the hook runs inside the SDK's transport TaskGroup, so raising there tears the whole
  connection down and reconnects — a much worse failure than one refused call, and one that would
  also interrupt every other turn on that connection. For the same reason the hook answers
  ``None`` (no credential) rather than raising on the late-expiry path.
* ``auth_scope: service`` never sends a user token, even inside a turn: the entry asked to act
  as the integration account, so that is the only credential it ever gets.
* Outside a turn there is no refusal: the service credential is the correct answer, and if it
  too is absent the request goes out bare so the server's own 401 is what the operator sees.

## One policy object, two entry points

:class:`CredentialPolicy` holds the three collaborators (config parser, turn-token reader,
ledger reader) so a second deployment can substitute any of them instead of forking this file.
The module-level functions are the singleton instance's bound behaviour, which is what the rest of
the tree calls.

Config::

    mcp_servers:
      kingdee:
        auth_scope: user            # run tools/call as the signed-in user
        service_credentials:        # connection / plumbing credential
          token_url: https://idp/realms/r/protocol/openid-connect/token
          client_id: hermes-mcp-client
          client_secret_env: HERMES_MCP_CLIENT_SECRET
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

from tools import mcp_service_credentials as _service
from tools.mcp_tool_common import _core

logger = logging.getLogger("tools.mcp_tool")

#: Which credential a server entry will accept on a user's behalf.
SCOPE_USER = "user"
SCOPE_SERVICE = "service"
SCOPE_UNSET = ""

#: What reading a server's config out of the connection ledger can fail with. These are plain
#: dict/module reads, so the only shapes are an unresolved server key (``LookupError``) and a
#: partially torn-down registry (``AttributeError``/``RuntimeError``). Named so the refusal path
#: states exactly what it tolerates instead of swallowing ``Exception``.
_LEDGER_READ_FAILURES = (LookupError, AttributeError, RuntimeError)

_DEFAULT_HEADER = "Authorization"
_BEARER = "Bearer "


@dataclass(frozen=True)
class CredentialPlan:
    """What a server entry asked for, resolved once per connection."""

    server_name: str
    auth_scope: str
    user_scope: bool
    service: Optional[_service.ServiceCredentialMinter]

    def describe(self) -> str:
        """One-line diagnostic (never includes a token)."""
        return (f"server={self.server_name} scope={self.auth_scope or 'unset'} "
                f"service={'yes' if self.service else 'no'}")


def _auth_scope_of(config: dict) -> str:
    """``"user"`` / ``"service"`` / ``""`` for a server entry."""
    raw = str(config.get("auth_scope") or "").strip().lower()
    if raw in (SCOPE_USER, SCOPE_SERVICE):
        return raw
    if raw:
        logger.warning("auth_scope %r is not 'user' or 'service'; ignoring it", raw)
    return SCOPE_UNSET


def _statically_configured(config: dict) -> bool:
    """True when the entry already owns the credential slot itself.

    An explicit ``headers.Authorization``, an ``identity_header`` named Authorization, or
    ``auth: oauth`` is a deliberate single credential for the connection; layering a second one
    on top would send two competing bearers and change which principal the server authorises.

    The ``identity_header`` form matters because it is merged into the outgoing headers *after*
    this decision (:func:`tools.mcp_tool_errors._apply_identity_header`) — a router that claimed
    the slot would overwrite it on every request, and the "send nothing when the user token is
    gone" guarantee would silently degrade to sending the deployment's static bearer.
    """
    names = {str(key).strip().lower() for key in (config.get("headers") or {})}
    if str(config.get("auth") or "").strip().lower() == "oauth":
        return True
    identity = config.get("identity_header")
    if isinstance(identity, dict) and str(identity.get("name") or "").strip().lower() == (
            _DEFAULT_HEADER.lower()):
        return True
    return _DEFAULT_HEADER.lower() in names


def _format_bearer_token(token: str) -> str:
    """The ``Authorization`` header value for a bearer *token*."""
    return f"{_BEARER}{token}"


class CredentialPolicy:
    """The routing decision, with its collaborators injectable.

    Everything is a plain callable so a test substitutes one behaviour without rebuilding a
    provider stack, and a deployment that keeps its users' tokens somewhere else overrides
    :meth:`user_token` alone.
    """

    def __init__(self, *, parse_credentials: Callable = _service.parse_service_credentials,
                 minter_for: Callable = _service.minter_for,
                 user_token: Optional[Callable[[], Optional[str]]] = None,
                 in_user_turn: Optional[Callable[[], bool]] = None,
                 server_config: Optional[Callable[[str], dict]] = None) -> None:
        self._parse_credentials = parse_credentials
        self._minter_for = minter_for
        self._user_token = user_token or _turn_token
        self._in_user_turn = in_user_turn or _in_turn
        self._server_config = server_config or _ledger_config

    # ---- the plan --------------------------------------------------------

    def plan_for(self, server_name: str, config: dict) -> Optional[CredentialPlan]:
        """Resolve the entry's credential plan, or ``None`` when Hermes should not manage one.

        ``None`` covers: nothing configured, and a statically configured credential (which wins).
        A malformed ``service_credentials`` block raises from the parser, so the operator sees the
        misconfiguration instead of an unexplained 401.
        """
        scope = _auth_scope_of(config)
        service_spec = self._parse_credentials(server_name, config)
        if (scope == SCOPE_UNSET and service_spec is None) or _statically_configured(config):
            if scope != SCOPE_UNSET or service_spec is not None:
                logger.debug("MCP server '%s': a credential is configured statically — Hermes "
                             "will not manage one", server_name)
            return None
        return CredentialPlan(
            server_name=server_name,
            auth_scope=scope,
            user_scope=scope == SCOPE_USER,
            service=self._minter_for(service_spec) if service_spec is not None else None)

    # ---- the httpx hook --------------------------------------------------

    def router(self, server_name: str, config: dict) -> Optional[Any]:
        """An ``httpx.Auth`` that routes each request to the right credential, or ``None``.

        ``None`` = leave the request alone (nothing configured, or the operator configured a
        static credential). Everything the hook needs is resolved once here — it holds the
        *plan*, never a token, so one instance is safe to share across reconnects and turns.
        """
        try:
            plan = self.plan_for(server_name, config)
        except ValueError as exc:
            # A misconfigured service_credentials block: refuse to connect with no credential at
            # all, which would read downstream as an unexplained 401.
            raise RuntimeError(str(exc)) from exc
        if plan is None:
            return None
        from tools.mcp_tool import sdk_httpx

        httpx_mod = sdk_httpx()
        if httpx_mod is None:  # pragma: no cover - an old mcp build with no resolvable httpx
            logger.warning("MCP server '%s': no SDK httpx module resolved; credentials will NOT "
                           "be attached", server_name)
            return None

        policy = self

        class _RoutingAuth(httpx_mod.Auth):  # type: ignore[misc,valid-type]
            """Attaches the service or the user credential, per request.

            Reads the *current turn* rather than capturing a token, because this object outlives
            every turn on the connection. Only the async flow is overridden: every client that
            mounts this hook is an ``AsyncClient``, and the base ``auth_flow`` would be
            unreachable — it is also the wrong shape here, since a mint is a network call that
            must not run on the event loop.
            """

            async def async_auth_flow(self, request):
                token = await policy._resolve_async(plan)
                if token is not None:
                    request.headers[_DEFAULT_HEADER] = _format_bearer_token(token)
                yield request

        logger.info("MCP server '%s': credential routing active (%s)", server_name, plan.describe())
        return _RoutingAuth()

    # ---- resolution ------------------------------------------------------

    async def async_token_for(self, plan: CredentialPlan) -> Optional[str]:
        """The credential for one request made OUTSIDE a turn."""
        return await plan.service.async_token() if plan.service is not None else None

    async def _resolve_async(self, plan: CredentialPlan) -> Optional[str]:
        # ``plan.user_scope`` gates the user branch as well as the turn check: an
        # ``auth_scope: service`` entry asked never to act as a person, so a turn running
        # must not put its token on that server's requests.
        if plan.user_scope and self._in_user_turn():
            return self._user_or_service(plan)
        return await self.async_token_for(plan)

    def _user_or_service(self, plan: CredentialPlan) -> Optional[str]:
        """The turn's own token, or ``None`` — a user-scoped call never downgrades.

        Only reached for ``auth_scope: user`` inside a login-driven turn (``_resolve_async``
        guarantees both). A missing token here means the access token expired *between* the
        handler's :meth:`refusal` check and this request, so there is deliberately no service
        fallback: sending the integration account would run the person's query as the shared
        caller and silently return data they are not entitled to. ``None`` sends the request
        bare, and the server's own 401/403 fails the call loudly instead.

        Never raises: this runs inside the SDK's transport TaskGroup, where an exception tears
        down the shared connection for every turn on it.
        """
        token = self._user_token()
        if token is not None:
            return token
        logger.warning("MCP server '%s': the sign-in token for this turn expired before the "
                       "request went out; sending it unauthenticated rather than running a "
                       "user-scoped call as the service account", plan.server_name)
        return None

    # ---- the pre-dispatch refusal ---------------------------------------

    def refusal(self, server_name: str, config: dict) -> Optional[str]:
        """A model-facing refusal when a user-scoped server would be called without a user token.

        ``None`` means "proceed": either the server is not user-scoped, or this turn carries a
        usable token. Returning a message instead of raising keeps the call path uniform with the
        other pre-dispatch guards and gives the model something actionable.

        Runs before any transport work on purpose — see the module docstring for why the refusal
        cannot live in the HTTP auth hook.
        """
        try:
            plan = self.plan_for(server_name, config)
        except (ValueError, RuntimeError):
            return None  # a config error is reported by the connect path, not here
        if plan is None or not plan.user_scope or not self._in_user_turn():
            return None
        if self._user_token() is not None:
            return None
        return (
            f"MCP server '{server_name}' runs user-scoped calls (auth_scope: user), but this turn "
            "has no usable sign-in token — it expired or this session was never signed in. Ask the "
            "user to sign in again and retry. Hermes will not run this call as the service "
            "account.")

    def refusal_for(self, server_name: str) -> Optional[str]:
        """Same refusal for a caller that has only a server name.

        The tool handler does not hold the server's config — it would have to resolve the
        connection to get one — so this pairs the ledger read with the policy decision and keeps
        both on this side of the boundary.
        """
        return self.refusal(server_name, self._server_config(server_name))


# ---- the default collaborators -------------------------------------------


def _turn_token() -> Optional[str]:
    """The running turn's access token, or ``None`` when there is none usable.

    ``None`` covers two different situations the caller must treat the same way — outside a turn,
    and a turn whose login's token is missing or expired. Both mean "no user credential to
    send"; what differs is whether :func:`_in_turn` says there was an identity to act as, which
    is what turns the second into a refusal instead of a service-account fallback.
    """
    from agent.turn_token import current_turn_token

    credential = current_turn_token()
    if credential is None or not credential.access_token:
        return None
    return credential.access_token if credential.is_fresh() else None


def _in_turn() -> bool:
    """True when a login is driving this turn — not when its token is still valid.

    The distinction is load-bearing. ``agent.turn_token.bind_session_token`` binds a placeholder
    for a login whose token expired, so this answers True for a user turn whose credential is
    stale; :func:`_turn_token` then answers ``None`` and the refusal fires. Treating "token
    expired" as "no turn" here is exactly what used to route the call to the service account.
    """
    from agent.turn_token import current_turn_token

    return current_turn_token() is not None


def _ledger_config(server_name: str) -> dict:
    """The server entry a credential decision is made from, or ``{}`` when not resolvable.

    Reads the connection ledger directly and NEVER connects: the refusal this feeds must happen
    before any transport work, and the lazy-start entry point (``discovery._get_connected_server_
    for_call``) spawns the child as a side effect of being asked. A refusal that connected first
    would do the transport work it exists to prevent — unapproved, for an untrusted server.

    The connected task's ``_config`` wins because it is the post-render entry the transport
    actually uses; ``_lazy_server_configs`` is the fallback for a server that has not connected
    yet, so a schema-cache-registered lazy server still reports its ``auth_scope``.

    Best-effort by design: this is a safety net, not a gate. An unreadable ledger yields ``{}``
    and leaves the call to the server's own 401 rather than blocking a legitimate one.
    """
    from tools.mcp_tool_scope import _resolve_server_key
    try:
        key = _resolve_server_key(server_name)
        with _core._lock:
            server = _core._servers.get(key)
            config = getattr(server, "_config", None) if server is not None else None
            if not (isinstance(config, dict) and config):
                config = _core._lazy_server_configs.get(key)
    except _LEDGER_READ_FAILURES as exc:
        logger.debug("credential precheck: no config for server '%s' (%s); skipping the check",
                     server_name, exc)
        return {}
    return config if isinstance(config, dict) else {}


#: The process-wide policy every production call site uses.
policy = CredentialPolicy()


# ---- module-level API (the singleton's behaviour) -------------------------
# Only what a production call site actually needs. ``_auth_scope_of`` stays private: it is a
# one-line pure helper whose behaviour ``plan_for`` already covers.

def credential_router(server_name: str, config: dict) -> Optional[Any]:
    """See :meth:`CredentialPolicy.router`."""
    return policy.router(server_name, config)


__all__ = ["SCOPE_SERVICE", "SCOPE_UNSET", "SCOPE_USER", "CredentialPlan", "CredentialPolicy",
           "credential_router", "policy"]