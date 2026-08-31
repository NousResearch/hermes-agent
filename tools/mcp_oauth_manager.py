"""Central manager for per-server MCP OAuth state (one instance per process): per-server providers, cross-process
token reload (mtime-based disk watch so tokens refreshed by cron/another CLI are picked up without a restart), 401
deduplication (N concurrent 401s with the same access_token trigger one recovery) and reconnect signalling
(``MCPServerTask`` drives the reconnect; the manager decides when). The ONLY place that instantiates the SDK's
``OAuthClientProvider`` for runtime use; refresh stays lazy in the SDK — one ``stat()`` per tool call is cheaper
than an await + refresh round-trip."""

from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from tools.mcp_oauth_provider import HermesProviderMixin

logger = logging.getLogger(__name__)

try:
    from mcp.client.auth.oauth2 import OAuthClientProvider as _SDKOAuthClientProvider
    _SDK_BASES: tuple = (_SDKOAuthClientProvider,)
except ImportError:  # pragma: no cover — SDK required in CI; module must still import
    _SDK_BASES = ()


@dataclass
class _ProviderEntry:
    """Per-server OAuth state. ``last_mtime_ns``: last-seen tokens-file mtime (0 = never read)
    for external-refresh detection; ``lock`` binds to the first asyncio loop awaiting it (the MCP
    loop); ``pending_401`` dedupes thundering-herd 401s by failed access_token."""

    server_url: str
    oauth_config: Optional[dict]
    provider: Optional[Any] = None
    last_mtime_ns: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    pending_401: dict[str, "asyncio.Future[bool]"] = field(default_factory=dict)


class HermesMCPOAuthProvider(HermesProviderMixin, *_SDK_BASES):
    """OAuthClientProvider with pre-flow disk-mtime reload (external refreshes become visible to
    a running session), expiry seeding on cold load, pre-flight metadata discovery, dead-client
    registration detection and the bidirectional ``async_auth_flow`` bridge. Token-endpoint
    fixes come from ``HermesProviderMixin``. Only usable when the SDK's OAuth module imported.

    Reference: Claude Code's ``invalidateOAuthCacheIfDiskChanged`` (``src/utils/auth.ts:1320``, CC-1096 /
    GH#24317).
    """

    _hermes_logger = logger

    def __init__(self, *args: Any, server_name: str = "", preregistered: bool = False, **kwargs: Any):
        super().__init__(*args, **kwargs)
        # mcp 2.0 uses a task-owned anyio.Lock held across the yielded resource request (a session-long GET blocks
        # every POST; HTTPX may close the generator from another task). A binary semaphore drops task ownership.
        import anyio
        self.context.lock = anyio.Semaphore(1, max_value=1)
        self._hermes_server_name = server_name
        self._hermes_home = ""
        # A config-supplied client_id rejected as invalid_client means the *config* is wrong — only DCR clients auto-heal.
        self._hermes_preregistered = preregistered

    def _hermes_storage(self):
        """The context storage when it is a ``HermesTokenStorage``, else None."""
        from tools.mcp_oauth import HermesTokenStorage
        return self.context.storage if isinstance(self.context.storage, HermesTokenStorage) else None

    def _log_nonfatal(self, what: str, exc: BaseException) -> None:
        logger.debug("MCP OAuth '%s': %s failed (non-fatal): %s", self._hermes_server_name, what, exc)

        def __init__(
            self,
            *args: Any,
            server_name: str = "",
            preregistered: bool = False,
            token_user_agent: "str | None" = None,
            **kwargs: Any,
        ):
            super().__init__(*args, **kwargs)
            # mcp 2.0.0 uses a task-owned anyio.Lock and holds it across the
            # yielded resource request.  A session-long GET therefore blocks
            # every concurrent POST, and HTTPX may later close the auth-flow
            # generator from a different task than the lock owner.  A binary
            # semaphore preserves mutual exclusion without task ownership;
            # async_auth_flow below narrows its scope around resource I/O.
            import anyio

            self.context.lock = anyio.Semaphore(1, max_value=1)
            self._hermes_server_name = server_name
            self._hermes_home = ""
            # When the client_id comes from config.yaml (pre-registered), an
            # invalid_client rejection means the *config* is wrong — deleting
            # client.json would just be re-seeded from config and re-running
            # registration can't help. Only auto-heal dynamically-registered
            # clients. See _maybe_flag_poisoned_client.
            self._hermes_preregistered = preregistered
            # oauth.user_agent — stamped onto token-endpoint requests only;
            # some authorization servers/WAFs reject httpx's default (#75576).
            self._hermes_token_user_agent = token_user_agent

        def _stamp_token_user_agent(self, request):
            ua = getattr(self, "_hermes_token_user_agent", None)
            if ua:
                request.headers["User-Agent"] = ua
            return request

        def _coerce_client_secret_post(self) -> None:
            """Use client_secret_post when dynamic registration returned a secret.

            Some MCP OAuth providers, notably Supabase, return a
            ``client_secret`` from dynamic client registration but omit
            ``token_endpoint_auth_method``. The MCP SDK treats the missing
            value as public-client auth (``none``), so token exchange omits the
            secret and Supabase rejects it with ``Required parameter:
            client_secret``. Coerce the in-memory client info before token and
            refresh requests.
            """
            info = getattr(self.context, "client_info", None)
            if not info or not getattr(info, "client_secret", None):
                return
            method = getattr(info, "token_endpoint_auth_method", None)
            if method not in (None, "none", ""):
                return
            from mcp.shared.auth import OAuthClientInformationFull

            data = info.model_dump(mode="json", exclude_none=True)
            data["token_endpoint_auth_method"] = "client_secret_post"
            self.context.client_info = OAuthClientInformationFull.model_validate(data)

        async def _exchange_token_authorization_code(self, *args: Any, **kwargs: Any):
            self._coerce_client_secret_post()
            request = await super()._exchange_token_authorization_code(*args, **kwargs)
            return self._stamp_token_user_agent(request)

        async def _refresh_token(self):
            self._coerce_client_secret_post()
            request = await super()._refresh_token()
            return self._stamp_token_user_agent(request)

        async def _handle_token_response(self, response):
            """Accept any 2xx token response and avoid leaking token bodies in errors."""
            if 200 <= response.status_code < 300:
                from mcp.client.auth.utils import handle_token_response_scopes
                from mcp.client.auth.oauth2 import OAuthTokenError
                from httpx import HTTPError

                try:
                    token_response = await handle_token_response_scopes(response)
                except (HTTPError, OAuthTokenError):
                    raise OAuthTokenError("Invalid token response") from None
                self.context.current_tokens = token_response
                self.context.update_token_expiry(token_response)
                await self.context.storage.set_tokens(token_response)
                return

            from mcp.client.auth.oauth2 import OAuthTokenError

            raise OAuthTokenError(f"Token exchange failed ({response.status_code})")

        async def _handle_refresh_response(self, response) -> bool:
            """Accept any 2xx refresh response and avoid logging token bodies."""
            if not (200 <= response.status_code < 300):
                logger.warning("Token refresh failed: %s", response.status_code)
                self.context.clear_tokens()
                return False

            from mcp.shared.auth import OAuthToken
            from httpx import HTTPError
            from pydantic import ValidationError

            try:
                content = await response.aread()
                token_response = OAuthToken.model_validate_json(content)
                self.context.current_tokens = token_response
                self.context.update_token_expiry(token_response)
                await self.context.storage.set_tokens(token_response)
                return True
            except (HTTPError, ValidationError):
                logger.warning("Invalid refresh response: %s", response.status_code)
                self.context.clear_tokens()
                return False

        async def _initialize(self) -> None:
            """Load stored tokens + client info AND seed token_expiry_time.

            Also eagerly fetches OAuth authorization-server metadata (PRM +
            ASM) when we have stored tokens but no cached metadata, so the
            SDK's ``_refresh_token`` can build the correct token_endpoint
            URL on the preemptive-refresh path. Without this, the SDK
            falls back to ``{mcp_server_url}/token`` (wrong for providers
            whose AS is a different origin — BetterStack's MCP lives at
            ``https://mcp.betterstack.com`` but its token endpoint is at
            ``https://betterstack.com/oauth/token``), the refresh 404s, and
            we drop through to full browser reauth.

            The SDK's base ``_initialize`` populates ``current_tokens`` but
            does NOT call ``update_token_expiry``, so ``token_expiry_time``
            stays ``None`` and ``is_token_valid()`` returns True for any
            loaded token regardless of actual age. After a process restart
            this ships stale Bearer tokens to the server; some providers
            return HTTP 401 (caught by the 401 handler), others return 200
            with an app-level auth error (invisible to the transport layer,
            e.g. BetterStack returning "No teams found. Please check your
            authentication.").

            Seeding ``token_expiry_time`` from the reloaded token fixes that:
            ``is_token_valid()`` correctly reports False for expired tokens,
            ``async_auth_flow`` takes the ``can_refresh_token()`` branch,
            and the SDK quietly refreshes before the first real request.

            Paired with :class:`HermesTokenStorage` persisting an absolute
            ``expires_at`` timestamp (``mcp_oauth.py:set_tokens``) so the
            remaining TTL we compute here reflects real wall-clock age.
            """
            await super()._initialize()
            tokens = self.context.current_tokens
            if tokens is not None and tokens.expires_in is not None:
                self.context.update_token_expiry(tokens)
        if tokens is not None and self.context.oauth_metadata is None:
            try:
                await self._prefetch_oauth_metadata()
            except Exception as exc:  # pragma: no cover — the SDK's 401-branch discovery runs next request
                self._log_nonfatal("pre-flight metadata discovery", exc)
            else:
                from tools.mcp_oauth_provider import enforce_refresh_token_issuer
                enforce_refresh_token_issuer(self.context)  # metadata (issuer) only just became known

                storage = self.context.storage
                from tools.mcp_oauth import HermesTokenStorage

                # When the rejected client_id was our Client ID Metadata
                # Document URL, re-presenting it next flow would loop: the
                # server has already fetched that document and refused it.
                # Dropping the URL sends the retry down the DCR branch
                # instead, and the marker on disk keeps the next process from
                # walking back into the same refusal. `hermes mcp login`
                # clears the marker, so a fixed document gets another chance.
                cimd_url = getattr(self.context, "client_metadata_url", None)
                rejected_id = getattr(self.context.client_info, "client_id", None)
                if cimd_url and rejected_id == cimd_url:
                    logger.warning(
                        "MCP OAuth '%s': authorization server rejected our "
                        "Client ID Metadata Document (%s) with invalid_client "
                        "— falling back to dynamic client registration.",
                        self._hermes_server_name, cimd_url,
                    )
                    self.context.client_metadata_url = None
                    if isinstance(storage, HermesTokenStorage):
                        storage.mark_cimd_rejected()

                if isinstance(storage, HermesTokenStorage):
                    storage.poison_client_registration()
                # Drop the in-memory client so the SDK re-registers next flow.
                self.context.client_info = None
                self._initialized = False
            except Exception as exc:  # pragma: no cover — defensive, must not throw
                logger.debug(
                    "MCP OAuth '%s': invalid_client detection failed (non-fatal): %s",
                    self._hermes_server_name, exc,
                )

        async def _send(client, url: str, label: str):
            try:
                return await client.send(create_oauth_metadata_request(url))
            except httpx.HTTPError as exc:
                logger.debug("MCP OAuth '%s': %s discovery to %s failed: %s", self._hermes_server_name, label, url, exc)
                return None
        async with httpx.AsyncClient(timeout=10.0) as client:
            # PRM discovery to learn the authorization_server URL.
            for url in build_protected_resource_metadata_discovery_urls(None, server_url):
                resp = await _send(client, url, "PRM")
                prm = await handle_protected_resource_response(resp) if resp is not None else None
                if prm:
                    self.context.protected_resource_metadata = prm
                    if prm.authorization_servers:
                        self.context.auth_server_url = str(prm.authorization_servers[0])
                    break
            # ASM discovery against auth_server_url (server_url fallback for legacy providers).
            for url in build_oauth_authorization_server_metadata_discovery_urls(self.context.auth_server_url, server_url):
                resp = await _send(client, url, "ASM")
                if resp is None:
                    continue
                ok, asm = await handle_auth_metadata_response(resp)
                if not ok:
                    break
                if asm:
                    self.context.oauth_metadata = asm
                    storage = self._hermes_storage()  # persist now so a later cold-load skips discovery
                    if storage is not None:
                        storage.save_oauth_metadata(asm)
                    logger.debug("MCP OAuth '%s': pre-flight ASM discovered token_endpoint=%s",
                                 self._hermes_server_name, asm.token_endpoint)
                    break

            # Manually bridge the bidirectional generator protocol. httpx's
            # auth_flow driver (httpx._client._send_handling_auth) calls
            # ``auth_flow.asend(response)`` to feed HTTP responses back into
            # the generator. A naive wrapper using ``async for item in inner:
            # yield item`` DISCARDS those .asend(response) values and resumes
            # the inner generator with None, so the SDK's
            # ``response = yield request`` branch in
            # mcp/client/auth/oauth2.py sees response=None and crashes at
            # ``if response.status_code == 401`` with AttributeError.
            #
            # The bridge below forwards each .asend() value into the inner
            # generator via inner.asend(incoming), preserving the bidirectional
            # contract. Regression from PR #11383 caught by
            # tests/tools/test_mcp_oauth_bidirectional.py.
            inner = super().async_auth_flow(request)
            resource_lock_released = False
            sent_access_token = None
            retry_after_concurrent_auth = False
            try:
                outgoing = await inner.__anext__()
                while True:
                    # The SDK holds context.lock for its entire generator,
                    # including while HTTPX waits on the actual MCP request.
                    # Release it only for that request.  OAuth discovery,
                    # refresh, registration, and token exchange remain
                    # serialized exactly as the SDK implements them.
                    if outgoing is request:
                        tokens = self.context.current_tokens
                        sent_access_token = (
                            tokens.access_token if tokens is not None else None
                        )
                        self.context.lock.release()
                        resource_lock_released = True
                    incoming = yield outgoing
                    if resource_lock_released:
                        await self.context.lock.acquire()
                        resource_lock_released = False
                    # A different request may have completed refresh or full
                    # authorization while this resource request was in
                    # flight.  Retry with that token instead of starting a
                    # duplicate OAuth transition from the stale 401/403.
                    tokens = self.context.current_tokens
                    if (
                        getattr(incoming, "status_code", None) in (401, 403)
                        and self.context.is_token_valid()
                        and tokens is not None
                        and tokens.access_token != sent_access_token
                    ):
                        self._add_auth_header(request)
                        await inner.aclose()
                        retry_after_concurrent_auth = True
                        break
                    # Sniff the response for a dead-client-registration signal
                    # before handing it back to the SDK (best-effort, GH#36767).
                    await self._maybe_flag_poisoned_client(incoming)
                    outgoing = await inner.asend(incoming)
            except StopAsyncIteration:
                # Persist any metadata the SDK discovered lazily during the
                # 401 branch so a subsequent cold-load skips discovery.
                self._persist_oauth_metadata_if_changed()
                return
            finally:
                if resource_lock_released:
                    # Balance the SDK's surrounding ``async with`` even when
                    # HTTPX cancels or closes the flow while the resource
                    # request is still in flight.  Shield only this local
                    # bookkeeping; general inner-generator teardown remains
                    # the separate concern tracked by the cleanup PR.
                    import anyio

                    with anyio.CancelScope(shield=True):
                        await self.context.lock.acquire()

            if retry_after_concurrent_auth:
                yield request
                self._persist_oauth_metadata_if_changed()
                return

    async def async_auth_flow(self, request):  # type: ignore[override]
        try:  # pre-flow hook: reload from disk if it changed (non-fatal on error)
            await get_manager().invalidate_if_disk_changed(self._hermes_server_name, hermes_home=self._hermes_home)
        except Exception as exc:  # pragma: no cover — defensive
            self._log_nonfatal("pre-flow disk-watch", exc)
        # Bridge the bidirectional generator by hand: a naive ``async for item in inner: yield
        # item`` DISCARDS the responses httpx sends back via ``asend``, and the SDK crashes on None.
        # Manually bridge the bidirectional generator protocol. httpx's auth_flow driver
        # (httpx._client._send_handling_auth) calls ``auth_flow.asend(response)`` to feed HTTP responses
        # back into the generator. A naive wrapper using ``async for item in inner: yield item`` DISCARDS
        # those .asend(response) values and resumes the inner generator with None, so the SDK's ``response =
        # yield request`` branch in mcp/client/auth/oauth2.py sees response=None and crashes at ``if
        # response.status_code == 401`` with AttributeError. The bridge below forwards each .asend() value
        # into the inner generator via inner.asend(incoming), preserving the bidirectional contract.
        # Regression from PR #11383 caught by tests/tools/test_mcp_oauth_bidirectional.py.
        inner = super().async_auth_flow(request)
        resource_lock_released = retry_after_concurrent_auth = False
        sent_access_token = None
        try:
            outgoing = await inner.__anext__()
            while True:
                # The SDK holds context.lock for its whole generator, even while HTTPX waits on
                # the MCP request. Release it for that request only; OAuth transitions stay serialized.
                if outgoing is request:
                    tokens = self.context.current_tokens
                    sent_access_token = tokens.access_token if tokens is not None else None
                    self.context.lock.release()
                    resource_lock_released = True
                incoming = yield outgoing
                if resource_lock_released:
                    await self.context.lock.acquire()
                    resource_lock_released = False
                # Another request may have refreshed/authorized while this one was in flight:
                # retry with that token instead of a duplicate OAuth transition from a stale 401/403.
                tokens = self.context.current_tokens
                if (getattr(incoming, "status_code", None) in (401, 403) and self.context.is_token_valid()
                        and tokens is not None and tokens.access_token != sent_access_token):
                    self._add_auth_header(request)
                    await inner.aclose()
                    retry_after_concurrent_auth = True
                    break
                # Sniff the response for a dead-client-registration signal before handing it back to the SDK
                # (best-effort, GH#36767).
                await self._maybe_flag_poisoned_client(incoming)
                outgoing = await inner.asend(incoming)
        except StopAsyncIteration:
            self._persist_oauth_metadata_if_changed()  # metadata discovered lazily in the 401 branch
        finally:
            if resource_lock_released:
                # Balance the SDK's surrounding ``async with`` even when HTTPX cancels/closes the
                # flow mid-request; shield only this local bookkeeping.
                import anyio
                with anyio.CancelScope(shield=True):
                    await self.context.lock.acquire()
        if retry_after_concurrent_auth:
            yield request
            self._persist_oauth_metadata_if_changed()


# Cached at import time; None when the SDK's OAuth module is unavailable.
_HERMES_PROVIDER_CLS: Optional[type] = HermesMCPOAuthProvider if _SDK_BASES else None


class MCPOAuthManager:
    """Single source of truth for per-server MCP OAuth state. ``_entries`` is guarded by
    ``_entries_lock`` (get-or-create); per-entry state by the entry's ``asyncio.Lock``."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], _ProviderEntry] = {}
        self._entries_lock = threading.Lock()
        # Strong refs to in-flight 401 tasks so the loop's weak bookkeeping cannot GC them mid-run.
        self._inflight_tasks: set[asyncio.Task] = set()

    def get_or_build_provider(self, server_name: str, server_url: str, oauth_config: Optional[dict]) -> Optional[Any]:
        """Cached OAuth provider for ``server_name``, built on first use (rebuilt when ``server_url`` changes);
        None if the MCP SDK's OAuth support is unavailable."""
        key = self._key(server_name)
        with self._entries_lock:
            entry = self._entries.get(key)
            if entry is not None and entry.server_url != server_url:
                logger.info("MCP OAuth '%s': URL changed from %s to %s, discarding cache", server_name, entry.server_url, server_url)
                entry = None
            if entry is None:
                entry = self._entries[key] = _ProviderEntry(server_url=server_url, oauth_config=oauth_config)
            if entry.provider is None:
                entry.provider = self._build_provider(server_name, entry)
                if entry.provider is not None:
                    entry.provider._hermes_home = key[0]
            return entry.provider

    @staticmethod
    def _key(server_name: str, hermes_home: str | Path | None = None) -> tuple[str, str]:
        from hermes_constants import get_hermes_home
        home = Path(hermes_home) if hermes_home is not None else get_hermes_home()
        return (str(home.expanduser().resolve(strict=False)), server_name)

    def _build_provider(self, server_name: str, entry: _ProviderEntry) -> Optional[Any]:
        """Build a ``HermesMCPOAuthProvider``; None if the SDK's OAuth support is unavailable."""
        if _HERMES_PROVIDER_CLS is None:
            logger.warning("MCP OAuth '%s': SDK auth module unavailable", server_name)
            return None

        # Local imports avoid circular deps at module import time.
        from tools.mcp_oauth import (
            HermesTokenStorage,
            OAuthNonInteractiveError,
            _OAUTH_AVAILABLE,
            _build_client_metadata,
            _configure_callback_port,
            _is_interactive,
            _maybe_preregister_client,
            _make_callback_waiter,
            _make_redirect_handler,
            cimd_provider_kwargs,
            token_request_user_agent,
        )

        if not _OAUTH_AVAILABLE:
            return None
        cfg, storage = prepare_oauth_config(server_name, entry.server_url, entry.oauth_config)
        if get_dashboard_oauth_flow() is None and not _is_interactive() and not storage.has_cached_tokens():
            raise OAuthNonInteractiveError(
                "MCP OAuth for "
                f"'{server_name}': non-interactive environment and no "
                "cached tokens found. Run `hermes mcp login "
                f"{server_name}` interactively first to complete initial "
                "authorization."
            )

        _configure_callback_port(cfg, storage)
        client_metadata = _build_client_metadata(cfg)
        _maybe_preregister_client(storage, cfg, client_metadata)

        resolved_port = cfg.get("_resolved_port", 0)
        redirect_handler = _make_redirect_handler(resolved_port)
        # mcp 2.0 removed OAuthClientProvider's `timeout` argument, so the
        # configured `oauth.timeout` now bounds the callback waiter's own poll
        # loop instead — that is where the browser round-trip is awaited.
        callback_handler = _make_callback_waiter(
            resolved_port, cfg.get("_cimd_url"), timeout=float(cfg.get("timeout", 300))
        )

        return _HERMES_PROVIDER_CLS(
            server_name=server_name,
            preregistered=bool(cfg.get("client_id")),
            server_url=entry.server_url,
            client_metadata=client_metadata,
            storage=storage,
            redirect_handler=redirect_handler,
            callback_handler=callback_handler,
            token_user_agent=token_request_user_agent(cfg),
            **cimd_provider_kwargs(cfg),
        )

    def remove(
        self,
        server_name: str,
        *,
        hermes_home: str | Path | None = None,
    ) -> _ProviderEntry | None:
        """Evict the provider from cache AND delete tokens from disk.

        Called by ``hermes mcp remove <name>`` and (indirectly) by
        ``hermes mcp login <name>`` during forced re-auth.
        """
        with self._entries_lock:
            entry = self._entries.pop(self._key(server_name, hermes_home), None)

    def remove(self, server_name: str, *, hermes_home: str | Path | None = None) -> _ProviderEntry | None:
        """Evict the provider from cache AND delete tokens from disk (``hermes mcp remove`` / forced re-auth)."""
        entry = self.evict(server_name, hermes_home=hermes_home)
        from tools.mcp_oauth import remove_oauth_tokens
        remove_oauth_tokens(server_name, hermes_home=hermes_home)
        logger.info("MCP OAuth '%s': evicted from cache and removed from disk", server_name)
        return entry

    def restore_entry(self, server_name: str, entry: _ProviderEntry | None, *, hermes_home: str | Path | None = None) -> None:
        """Restore a provider entry removed for a failed reauthorization."""
        if entry is None:
            return
        with self._entries_lock:
            self._entries.setdefault(self._key(server_name, hermes_home), entry)

    def evict(self, server_name: str, *, hermes_home: str | Path | None = None) -> _ProviderEntry | None:
        """Drop only the in-process provider, preserving persisted OAuth state."""
        with self._entries_lock:
            return self._entries.pop(self._key(server_name, hermes_home), None)

    async def invalidate_if_disk_changed(self, server_name: str, *, hermes_home: str | Path | None = None) -> bool:
        """Force the SDK provider to reload when the tokens file mtime changed (e.g. a cron refresh); True if so."""
        from tools.mcp_oauth import _get_token_dir, _safe_filename
        entry = self._entries.get(self._key(server_name, hermes_home))
        if entry is None or entry.provider is None:
            return False
        async with entry.lock:
            try:
                mtime_ns = (_get_token_dir(hermes_home) / f"{_safe_filename(server_name)}.json").stat().st_mtime_ns
            except OSError:
                return False
            if mtime_ns == entry.last_mtime_ns:
                return False
            old, entry.last_mtime_ns = entry.last_mtime_ns, mtime_ns
            # `_initialized` is private SDK API but stable across the pinned versions (>=1.26.0).
            if hasattr(entry.provider, "_initialized"):
                entry.provider._initialized = False  # noqa: SLF001
            logger.info("MCP OAuth '%s': tokens file changed (mtime %d -> %d), forcing reload", server_name, old, mtime_ns)
            return True

    async def _recover_401(self, server_name: str, entry: _ProviderEntry, key: str, pending: asyncio.Future) -> None:
        """Single recovery attempt behind *pending*; always clears the dedup slot."""
        try:
            # Disk changed (external refresh)? Else: if the SDK can refresh in place, let the caller retry.
            can_refresh = await self.invalidate_if_disk_changed(server_name)
            if not can_refresh:
                try:
                    can_refresh = bool(entry.provider.context.can_refresh_token())
                except Exception:  # no context / not callable / probe failed
                    can_refresh = False
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("MCP OAuth '%s': 401 handler failed: %s", server_name, exc)
            can_refresh = False
        finally:
            entry.pending_401.pop(key, None)
        if not pending.done():
            pending.set_result(can_refresh)

    async def handle_401(self, server_name: str, failed_access_token: Optional[str] = None) -> bool:
        """Handle a 401 from a tool call. True: a (possibly new) token is available — reconnect and retry. False: no
        recovery path — surface ``needs_reauth`` so the model stops hallucinating manual refreshes. Concurrent 401s
        with the same ``failed_access_token`` fire one recovery attempt; the rest await its future."""
        entry = self._entries.get(self._key(server_name))
        if entry is None or entry.provider is None:
            return False
        key = failed_access_token or "<unknown>"
        async with entry.lock:
            pending = entry.pending_401.get(key)
            if pending is None:
                pending = entry.pending_401[key] = asyncio.get_running_loop().create_future()
                task = asyncio.create_task(self._recover_401(server_name, entry, key, pending))
                self._inflight_tasks.add(task)
                task.add_done_callback(self._inflight_tasks.discard)
        try:
            return await pending
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("MCP OAuth '%s': awaiting 401 handler failed: %s", server_name, exc)
            return False


_MANAGER: Optional[MCPOAuthManager] = None
_MANAGER_LOCK = threading.Lock()


def get_manager() -> MCPOAuthManager:
    """Return the process-wide :class:`MCPOAuthManager` singleton."""
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = MCPOAuthManager()
        return _MANAGER


def reset_manager_for_tests() -> None:
    """Test-only helper: drop the singleton so fixtures start clean."""
    global _MANAGER
    with _MANAGER_LOCK:
        _MANAGER = None
