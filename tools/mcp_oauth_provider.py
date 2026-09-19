"""Shared ``OAuthClientProvider`` customizations for Hermes MCP OAuth.

Two code paths build an SDK provider — ``tools.mcp_oauth.build_oauth_auth`` (legacy public
API) and ``tools.mcp_oauth_manager.MCPOAuthManager`` — and both need the same real-world
fixes and config → constructor-kwargs plumbing. This module holds that core once; the origin
modules keep their own subclass (logger name, disk-watch hooks) on top of it.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tools.mcp_oauth import HermesTokenStorage
logger = logging.getLogger(__name__)

# Authorization servers that advertise ``authorization_response_iss_parameter_supported`` and then
# omit ``iss`` from the redirect (#111135). Exact issuer match, nothing else is relaxed.
_ISS_OMITTING_ISSUERS = frozenset({"https://api.figma.com"})

# Authorization-server metadata documents the SDK tries in its 401 branch (RFC 8414 / OIDC discovery).
_ASM_DISCOVERY_PATHS = ("/.well-known/oauth-authorization-server", "/.well-known/openid-configuration")
_DISCOVERY_CONTEXT_LEAD = "Could not read authorization-server metadata"


def _default_auth_request_user_agent() -> str:
    """``Hermes-Agent/<version>`` for SDK-built OAuth requests that would otherwise carry no User-Agent at
    all; versioned so an operator debugging a WAF block can tell which client they are looking at."""
    from hermes_cli import __version__
    return f"Hermes-Agent/{__version__}"


DEFAULT_AUTH_REQUEST_USER_AGENT = _default_auth_request_user_agent()


def stamp_default_user_agent(request):
    """Give an SDK-built OAuth request (discovery, registration, token) a User-Agent if it has none.

    The SDK builds those as bare ``httpx.Request`` objects and sends them through ``client.send()``,
    which never merges the client's default headers, so they leave with NO ``User-Agent`` at all.
    WAF-fronted authorization servers (www.tradingview.com, coda.io) answer 403 to header-less
    requests while curl and a plain ``client.get()`` get 200 — the metadata document is then "not
    found", the SDK falls back to guessing ``/register`` and ``/authorize`` on the MCP host, and the
    user sees ``Registration failed: 404`` (#113771). Only an absent header is filled, so a
    configured ``oauth.user_agent`` (token requests) still wins."""
    if "user-agent" not in request.headers:
        request.headers["User-Agent"] = DEFAULT_AUTH_REQUEST_USER_AGENT
    return request


def _asm_discovery_failure(response) -> str | None:
    """``"<status> from <url>"`` when *response* is a failed authorization-server metadata fetch."""
    req = getattr(response, "request", None)
    status = getattr(response, "status_code", None)
    if req is None or status is None or 200 <= status < 300:
        return None
    url = str(req.url)
    return f"{status} from {url}" if any(p in url for p in _ASM_DISCOVERY_PATHS) else None


def _with_discovery_context(exc: Exception, failures: list[str]):
    """Re-shape a registration error raised after every metadata fetch failed: lead with the discovery
    failure, since the 404 on the guessed ``/register`` URL is only its consequence (#113771)."""
    return type(exc)(f"{_DISCOVERY_CONTEXT_LEAD} ({'; '.join(failures)}); dynamic client registration "
                     f"then fell back to a guessed endpoint on the MCP host and failed: {exc}")



class _RefreshCompletedByPeer(Exception):
    """Restart the SDK auth flow: a peer rotated the grant we were about to present."""

class HermesProviderMixin:
    """Token-endpoint fixes layered over the SDK's ``OAuthClientProvider`` (must precede it in
    the MRO; subclasses set ``_hermes_logger`` to keep their own logger name).

    - Supabase-style dynamic registration returns a ``client_secret`` but omits
      ``token_endpoint_auth_method``; the SDK then treats the client as public and the token
      endpoint rejects the exchange (looping the browser page) — coerce ``client_secret_post``.
    - ``token_user_agent`` (``oauth.user_agent``) is stamped onto token-endpoint requests only
      (some authorization servers/WAFs reject httpx's default).
    - Any 2xx token/refresh response is accepted; token bodies never leak into errors/logs."""

    _hermes_logger: logging.Logger = logger

    def __init__(self, *args: Any, token_user_agent: str | None = None, oauth_flow: str = "browser", **kwargs: Any):
        super().__init__(*args, **kwargs)
        # The MCP SDK replaces client_metadata.scope with the provider metadata
        # scopes after a 401. Keep an explicitly configured scope as the
        # authorization ceiling so providers cannot silently broaden consent.
        self._hermes_configured_scope = self.context.client_metadata.scope
        self._hermes_oauth_flow = oauth_flow
        # oauth.user_agent — stamped onto token-endpoint requests only; some authorization servers/WAFs
        # reject httpx's default (#75576).
        self._hermes_token_user_agent = token_user_agent

    def _restore_configured_scope(self) -> None:
        """Restore the explicit config scope after SDK metadata discovery."""
        scope = getattr(self, "_hermes_configured_scope", None)
        if scope:
            self.context.client_metadata.scope = scope

    async def _perform_authorization(self):
        self._restore_configured_scope()
        info = self.context.client_info
        grants = getattr(info, "grant_types", None) or []
        if (getattr(self, "_hermes_oauth_flow", "browser") == "device"
                or ("urn:ietf:params:oauth:grant-type:device_code" in grants and "authorization_code" not in grants)):
            from tools.mcp_oauth import OAuthNonInteractiveError
            raise OAuthNonInteractiveError(
                "MCP device authorization requires `hermes mcp login <server> --flow device`; "
                "background reconnects cannot start a device login")
        self._tolerate_missing_iss_for_known_server()
        return await super()._perform_authorization()

    def _tolerate_missing_iss_for_known_server(self) -> None:
        """Figma advertises ``authorization_response_iss_parameter_supported`` and then omits ``iss``
        from the redirect, so the SDK's RFC 9207 check rejects every valid code (#111135). For that
        one issuer only, fill a missing ``iss`` with the discovered issuer and warn; a present-but-
        different ``iss`` still fails the SDK check, and every other server keeps the strict rule."""
        issuer = _metadata_issuer(self.context)
        if issuer not in _ISS_OMITTING_ISSUERS:
            return
        inner = self.context.callback_handler

        async def _fill_iss():
            result = await inner()
            if getattr(result, "iss", None) is None and getattr(result, "code", None):
                self._hermes_logger.warning(
                    "MCP OAuth: %s omitted the iss parameter it advertises; accepting the redirect for that issuer only", issuer)
                result = result.model_copy(update={"iss": str(self.context.oauth_metadata.issuer)})
            return result

        self.context.callback_handler = _fill_iss

    def _prepare_token_request(self, request):
        """Stamp the configured User-Agent onto a token/refresh request."""
        ua = getattr(self, "_hermes_token_user_agent", None)  # tests build via __new__
        if ua:
            request.headers["User-Agent"] = ua
        return request

    def _coerce_client_secret_post(self) -> None:
        """Same rule as ``HermesTokenStorage._coerce_secret_auth_method``, applied to the
        in-memory client info BEFORE the SDK builds a token-endpoint request from it."""
        info = self.context.client_info
        if not info:
            return
        from mcp.shared.auth import OAuthClientInformationFull
        from tools.mcp_oauth import HermesTokenStorage
        data = info.model_dump(mode="json", exclude_none=True)
        if HermesTokenStorage._coerce_secret_auth_method(data):
            self.context.client_info = OAuthClientInformationFull.model_validate(data)

    async def _exchange_token_authorization_code(self, *args: Any, **kwargs: Any):
        self._coerce_client_secret_post()
        return self._prepare_token_request(await super()._exchange_token_authorization_code(*args, **kwargs))

    # Locked descriptor while this provider owns the refresh fence; cleared by
    # _hermes_release_refresh_fence. Never shared across instances.
    _hermes_fence: int | None = None

    async def async_auth_flow(self, request):
        """Guarantee fence release even if the auth generator is abandoned.

        The SDK drives the refresh as a generator: ``_refresh_token`` yields a
        request and ``_handle_refresh_response`` consumes the response. If the
        caller cancels in between (timeout, task cancellation, transport
        teardown), GeneratorExit/CancelledError is raised at the yield and
        ``_handle_refresh_response`` never runs. Without this wrapper the fence
        file would stay locked for the life of the process and every later
        refresh would fail closed -- trading a race for a deadlock.

        When a peer already rotated the grant while we waited for the fence,
        ``_refresh_token`` adopts it and raises ``_RefreshCompletedByPeer``;
        the SDK flow is then restarted so it re-evaluates validity and sends
        the original request with the winner's access token, never POSTing
        the burned refresh token.
        """
        while True:
            inner = super().async_auth_flow(request)
            discovery_failures: list[str] = []
            try:
                sent, thrown = None, None
                while True:
                    try:
                        if thrown is not None:
                            exc, thrown = thrown, None
                            out = await inner.athrow(exc)
                        else:
                            out = await inner.asend(sent)
                    except StopAsyncIteration:
                        return
                    except _RefreshCompletedByPeer:
                        break
                    except Exception as exc:
                        from mcp.client.auth.oauth2 import OAuthRegistrationError
                        if (isinstance(exc, OAuthRegistrationError) and discovery_failures
                                and self.context.oauth_metadata is None):
                            raise _with_discovery_context(exc, discovery_failures) from exc
                        raise
                    if out is not request:
                        stamp_default_user_agent(out)
                    # Full bidirectional delegation: the SDK drives this flow with
                    # asend(response), so `async for` would swallow the response and
                    # feed the inner generator None. Async generators have no
                    # `yield from`, hence the manual pump.
                    try:
                        sent = yield out
                    except GeneratorExit:
                        await inner.aclose()
                        raise
                    except BaseException as exc:
                        sent, thrown = None, exc
                    else:
                        failure = _asm_discovery_failure(sent)
                        if failure:
                            discovery_failures.append(failure)
            finally:
                await self._hermes_release_refresh_fence()

    async def _refresh_token(self):
        """Take the refresh fence, then build the request from the token we own.

        The fence is acquired BEFORE the final read of the refresh token and is
        released only after _handle_refresh_response has persisted the
        replacement, so one refresh generation is consumed by exactly one
        process. See tools.mcp_oauth.acquire_refresh_fence for the interleaving this
        closes.

        Holding a lock across ``yield`` in the SDK's generator-based auth flow
        is safe here because the SDK already serializes the whole flow under
        ``self.context.lock``: the generator is always driven to completion (or
        aborted) by one task, and async_auth_flow cannot interleave two
        refreshes inside one process.
        """
        self._coerce_client_secret_post()
        await self._hermes_acquire_refresh_fence()
        try:
            # Re-read under the fence: a peer may have rotated while we waited
            # for it, in which case the token we were about to POST is dead.
            # If disk already holds a different refresh token, the peer that held
            # the fence before us won this generation; its value is the only one
            # the provider will still accept, so install it even when its access
            # token has already expired (the POST we build needs the new grant).
            candidate = await self._hermes_rotated_candidate()
            if candidate is not None:
                if not self._hermes_install_disk_pair(candidate):
                    # Issuer binding stripped the peer's refresh token: nothing
                    # left to POST. Restart the flow so the SDK lands in 401 ->
                    # full authorization instead of raising over a dead grant.
                    raise _RefreshCompletedByPeer
                if self._hermes_live_ttl() and self.context.is_token_valid():
                    # The peer's access token is live: presenting our copy of the
                    # refresh token would only burn a generation on a single-use
                    # provider. Skip the POST and let the flow restart.
                    raise _RefreshCompletedByPeer
            return self._prepare_token_request(await super()._refresh_token())
        except BaseException:
            # Never hold the fence when no POST will follow.
            await self._hermes_release_refresh_fence()
            raise

    def _hermes_live_ttl(self) -> bool:
        """True when the installed token is not known to be past due.

        Storage clamps a past-due token to ``expires_in == 0`` on read (see
        HermesTokenStorage.get_tokens); the SDK's is_token_valid() compares
        ``time.time() <= expiry`` and still reports True for that boundary,
        which would make us adopt a token the server rejects immediately.
        ``expires_in`` is optional in RFC 6749: None means no expiry was
        issued, which the SDK treats as valid, so it counts as live here too.
        """
        if self.context.current_tokens is None:
            return False
        exp = getattr(self.context.current_tokens, "expires_in", None)
        return exp is None or int(exp) > 0

    async def _hermes_acquire_refresh_fence(self) -> None:
        """Enter the fence, or let RefreshFenceTimeout abort this attempt.

        Fails closed on purpose: a refresh we are not certain we own must not
        be POSTed. A stale fence from an aborted attempt is replaced rather
        than stacked, so a crashed generator cannot leak ownership.
        """
        from tools.mcp_oauth import acquire_refresh_fence

        await self._hermes_release_refresh_fence()
        storage = self.context.storage
        tokens_path = getattr(storage, "_tokens_path", None)
        if tokens_path is None:  # pragma: no cover - non-Hermes storage
            return
        self._hermes_fence = await acquire_refresh_fence(tokens_path())

    async def _hermes_release_refresh_fence(self) -> None:
        """Release the fence if held. Idempotent and never raises."""
        from tools.mcp_oauth import release_refresh_fence

        fd, self._hermes_fence = self._hermes_fence, None
        if fd is not None:
            release_refresh_fence(fd)

    async def _hermes_rotated_candidate(self):
        """The on-disk pair, if a peer rotated it past the one we hold.

        A candidate must carry a refresh token different from ours (same
        token: disk has nothing newer) and a non-empty access token. A disk
        entry with no refresh token is never adopted: its access token may
        still be inside its TTL, but taking it trades an explicit reauth now
        for a silent one at expiry with no way to refresh in between.
        ``get_tokens`` already returns None for absent or corrupt files.
        """
        stored = await self.context.storage.get_tokens()
        if stored is None:
            return None
        current = self.context.current_tokens
        stored_refresh = getattr(stored, "refresh_token", None)
        if not stored_refresh or not getattr(stored, "access_token", None):
            return None
        if stored_refresh == getattr(current, "refresh_token", None):
            return None
        return stored

    def _hermes_install_disk_pair(self, tokens) -> bool:
        """Publish a disk pair to the context and re-run issuer binding on it.

        Returns False when the enforcer strips the refresh token (the pair was
        minted by a different issuer): there is nothing left to refresh with,
        and each caller decides what that means for its own flow.
        """
        self.context.current_tokens = tokens
        self.context.update_token_expiry(tokens)
        enforce_refresh_token_issuer(self.context)
        return bool(getattr(self.context.current_tokens, "refresh_token", None))

    async def _initialize(self) -> None:
        """Load stored state, restore persisted server metadata when the SDK has none (so the issuer
        check and any refresh see the discovered ``issuer``/``token_endpoint`` instead of SDK guesses),
        then enforce refresh-token issuer binding."""
        await super()._initialize()
        storage = self.context.storage
        from tools.mcp_oauth import HermesTokenStorage
        if isinstance(storage, HermesTokenStorage) and self.context.oauth_metadata is None:
            meta = storage.load_oauth_metadata()
            if meta is not None:
                self.context.oauth_metadata = meta
        enforce_refresh_token_issuer(self.context)

    async def _store_tokens(self, token_response) -> None:
        self.context.current_tokens = token_response
        self.context.update_token_expiry(token_response)
        bind_issuer_from_context(self.context)
        await self.context.storage.set_tokens(token_response)

    async def _handle_token_response(self, response):
        """Accept any 2xx token response; never echo the body into errors."""
        from mcp.client.auth.oauth2 import OAuthTokenError
        if not (200 <= response.status_code < 300):
            raise OAuthTokenError(f"Token exchange failed ({response.status_code})")
        from httpx import HTTPError
        from mcp.client.auth.utils import handle_token_response_scopes
        try:
            token_response = await handle_token_response_scopes(response)
        except (HTTPError, OAuthTokenError):
            raise OAuthTokenError("Invalid token response") from None
        await self._store_tokens(token_response)

    async def _handle_refresh_response(self, response) -> bool:
        """Accept any 2xx refresh response; never log the body.

        Always releases the refresh fence: this is the single exit point of the
        fenced section, whatever the outcome.
        """
        try:
            return await self._hermes_handle_refresh_response(response)
        finally:
            await self._hermes_release_refresh_fence()

    async def _hermes_handle_refresh_response(self, response) -> bool:
        if not (200 <= response.status_code < 300):
            self._hermes_logger.warning("Token refresh failed: %s", response.status_code)
            # A writer outside the fence (interactive `hermes mcp login`, or a
            # pre-fence Hermes sharing this HERMES_HOME) may have rotated the
            # grant and persisted the replacement. Providers issuing single-use
            # refresh tokens reject our stale copy with a 400. Re-read disk
            # before destroying the session.
            if await self._hermes_reload_tokens_after_refresh_failure():
                self._hermes_logger.info(
                    "Recovered a peer-rotated refresh token instead of clearing the session"
                )
                return True
            self.context.clear_tokens()
            return False
        from httpx import HTTPError
        from mcp.shared.auth import OAuthToken
        from pydantic import ValidationError
        try:
            token_response = OAuthToken.model_validate_json(await response.aread())
        except (HTTPError, ValidationError):
            self._hermes_logger.warning("Invalid refresh response: %s", response.status_code)
            self.context.clear_tokens()
            return False
        # RFC 6749 §6: a refresh response may omit refresh_token (AS does not rotate) and scope
        # (unchanged). The SDK's own _handle_refresh_response carries both forward; this override
        # must too, or every non-rotating refresh erases the stored refresh_token and the server
        # dies at the NEXT expiry with a forced browser re-auth (#62333).
        prior = self.context.current_tokens
        if prior is not None:
            if token_response.refresh_token is None:
                token_response.refresh_token = prior.refresh_token
            if token_response.scope is None:
                token_response.scope = prior.scope
        await self._store_tokens(token_response)
        return True

    async def _hermes_reload_tokens_after_refresh_failure(self) -> bool:
        """Re-read tokens from disk after a rejected refresh.

        Returns True only when disk holds a pair that is BOTH different from
        the one we just failed with AND still live. That is the signature of
        a writer outside the fence (an interactive ``hermes mcp login`` or a
        pre-fence Hermes) having rotated the grant between our read and our
        POST -- a recoverable race, not a dead credential.

        Returns False for the genuinely-expired case (nobody wrote a newer
        pair), so the caller still clears state and surfaces the reauth
        prompt.
        """
        candidate = await self._hermes_rotated_candidate()
        if candidate is None:
            return False
        # Publish, then restore on rejection. is_token_valid() reads the
        # context rather than taking a token argument, so the candidate has
        # to be installed to be tested; a losing probe must leave the context
        # exactly as it found it.
        previous_tokens = self.context.current_tokens
        if (
            self._hermes_install_disk_pair(candidate)
            and self._hermes_live_ttl()
            and self.context.is_token_valid()
        ):
            return True
        self.context.current_tokens = previous_tokens
        self.context.update_token_expiry(previous_tokens)
        return False


def _metadata_issuer(context: Any) -> str | None:
    """Discovered authorization-server issuer from the SDK auth context, without trailing slash."""
    meta = getattr(context, "oauth_metadata", None)
    issuer = getattr(meta, "issuer", None) if meta is not None else None
    return (str(issuer).rstrip("/") or None) if issuer else None


def bind_issuer_from_context(context: Any) -> None:
    """Record the discovered issuer so the next ``storage.set_tokens`` (exchange or refresh) carries
    it. No-op when metadata is not discovered yet or storage is not Hermes'."""
    from tools.mcp_oauth import HermesTokenStorage
    storage = getattr(context, "storage", None)
    issuer = _metadata_issuer(context)
    if isinstance(storage, HermesTokenStorage) and issuer:
        storage.bind_issuer(issuer)


def enforce_refresh_token_issuer(context: Any) -> None:
    """Refuse to reuse a refresh token minted by a different issuer.

    The authorization server discovered for an MCP server can change (DNS takeover, protected-resource
    metadata edit, server migration); sending the stored refresh token to the new issuer hands it a
    long-lived credential. On mismatch the refresh token is stripped (memory + disk) while an unexpired
    access token stays usable; full re-authorization happens at expiry. Token files predating the field
    adopt the current issuer once rather than forcing a re-login. Runs after ``_initialize`` restored
    tokens + metadata, before the SDK's ``can_refresh_token()`` decision."""
    from tools.mcp_oauth import HermesTokenStorage
    storage = getattr(context, "storage", None)
    tokens = getattr(context, "current_tokens", None)
    if not isinstance(storage, HermesTokenStorage) or tokens is None or not getattr(tokens, "refresh_token", None):
        return
    current = _metadata_issuer(context)
    if current is None:  # not discovered yet; the SDK's 401-branch discovery + _store_tokens stamp it later
        return
    stored = (storage.loaded_issuer or "").rstrip("/") or None
    if stored is None:
        storage.stamp_issuer(current)
        return
    if stored != current:
        logger.warning("MCP OAuth: authorization server issuer changed (%s -> %s); dropping the stored "
                       "refresh token rather than sending it to a different issuer", stored, current)
        storage.strip_refresh_token()
        tokens.refresh_token = None


def prepare_oauth_config(server_name: str, server_url: str, oauth_config: dict | None) -> tuple[dict, "HermesTokenStorage"]:
    """Copy the ``oauth:`` block, apply provider defaults, open its token storage. The copy
    matters: later steps record ``_resolved_port`` / ``_cimd_url`` in the dict, which must
    never leak back into the caller's config."""
    from tools import mcp_oauth as mo
    cfg = dict(oauth_config or {})
    mo.apply_oauth_provider_defaults(cfg, server_name=server_name, server_url=server_url)
    return cfg, mo.HermesTokenStorage(server_name)


def build_provider_kwargs(cfg: dict, storage: "HermesTokenStorage", *, ssh_proxy_hint: bool) -> dict[str, Any]:
    """Resolve the callback port and return the shared provider constructor kwargs. Order
    matters: metadata needs the resolved port, pre-registration needs the metadata.
    ``ssh_proxy_hint`` lets the redirect handler tailor its remote-session hint to a configured
    proxy ``redirect_uri``. Helpers are looked up on ``tools.mcp_oauth`` so tests can patch them."""
    from tools import mcp_oauth as mo
    port = mo._configure_callback_port(cfg, storage)
    client_metadata = mo._build_client_metadata(cfg)
    mo._maybe_preregister_client(storage, cfg, client_metadata)
    redirect_uri = (cfg.get("redirect_uri") or None) if ssh_proxy_hint else None
    return {
        "client_metadata": client_metadata,
        "storage": storage,
        "redirect_handler": mo._make_redirect_handler(port, redirect_uri=redirect_uri, redirect_host=cfg.get("redirect_host")),
        # mcp 2.0 dropped OAuthClientProvider's own `timeout`; the configured
        # `oauth.timeout` bounds the callback waiter's poll loop instead.
        "callback_handler": mo._make_callback_waiter(port, cfg.get("_cimd_url"), timeout=float(cfg.get("timeout", 300))),
        "token_user_agent": mo.token_request_user_agent(cfg),
        "oauth_flow": cfg.get("flow", "browser"),
        **mo.cimd_provider_kwargs(cfg)}
