"""Central manager for per-server MCP OAuth state (one instance per process): per-server providers, cross-process
token reload (mtime-based disk watch so tokens refreshed by cron/another CLI are picked up without a restart), 401
deduplication (N concurrent 401s with the same access_token trigger one recovery) and reconnect signalling
(``MCPServerTask`` drives the reconnect; the manager decides when). The ONLY place that instantiates the SDK's
``OAuthClientProvider`` for runtime use; refresh stays lazy in the SDK — one ``stat()`` per tool call is cheaper
than an await + refresh round-trip."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import random
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


# Processes sharing one tokens file (gateway, dashboard, desktop app backend) all reached expiry
# in the same second, each POSTed a refresh, and the losers sent a rotated-out refresh token
# (400, then 429) and parked the server. Each process now refreshes early by its own random
# margin, refreshes under a per-server file lock after re-reading the tokens file, and adopts
# tokens another process wrote when its refresh fails.
_REFRESH_SKEW_FRACTION = (0.05, 0.15)  # of the token's remaining lifetime
_REFRESH_SKEW_CAP_S = 600.0
_REFRESH_LOCK_WAIT_S = 15.0
_REFRESH_RETRY_STATUSES = (400, 401, 429)


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
        self._hermes_refresh_skew = random.uniform(*_REFRESH_SKEW_FRACTION)
        self._hermes_protocol_version: Optional[str] = None
        self._hermes_prm_attempted = False
        self._hermes_true_expiry: Optional[float] = None

    def _hermes_storage(self):
        """The context storage when it is a ``HermesTokenStorage``, else None."""
        from tools.mcp_oauth import HermesTokenStorage
        return self.context.storage if isinstance(self.context.storage, HermesTokenStorage) else None

    def _log_nonfatal(self, what: str, exc: BaseException) -> None:
        logger.debug("MCP OAuth '%s': %s failed (non-fatal): %s", self._hermes_server_name, what, exc)

    async def _initialize(self) -> None:
        """Load stored state, seed ``token_expiry_time``, restore/prefetch metadata. The SDK's
        ``_initialize`` never calls ``update_token_expiry``, so a restarted process would ship stale
        Bearer tokens as "valid"; seeding the expiry (``HermesTokenStorage`` persists absolute
        ``expires_at``) makes the SDK refresh first. Metadata is restored from disk, else discovered
        pre-flight when we hold tokens but no metadata: otherwise ``_refresh_token`` guesses
        ``{server_url}/token`` (wrong for split-origin providers), 404s, and we fall to browser reauth."""
        await super()._initialize()  # HermesProviderMixin: restores metadata from disk, enforces issuer binding
        tokens = self.context.current_tokens
        if tokens is not None and tokens.expires_in is not None:
            self.context.update_token_expiry(tokens)
            self._hermes_apply_refresh_skew(tokens)
        if tokens is not None and self.context.oauth_metadata is None:
            try:
                await self._prefetch_oauth_metadata()
            except Exception as exc:  # pragma: no cover — the SDK's 401-branch discovery runs next request
                self._log_nonfatal("pre-flight metadata discovery", exc)
            else:
                from tools.mcp_oauth_provider import enforce_refresh_token_issuer
                enforce_refresh_token_issuer(self.context)  # metadata (issuer) only just became known

    def _hermes_apply_refresh_skew(self, tokens: Any) -> None:
        """Treat the token as due a per-process random margin before it really expires, so the
        processes sharing its file stop refreshing in the same second."""
        lifetime = getattr(tokens, "expires_in", None)
        self._hermes_true_expiry = self.context.token_expiry_time
        if self.context.token_expiry_time and lifetime:
            margin = min(_REFRESH_SKEW_CAP_S, self._hermes_refresh_skew * float(lifetime))
            self.context.token_expiry_time -= margin

    async def _store_tokens(self, token_response) -> None:
        await super()._store_tokens(token_response)
        self._hermes_apply_refresh_skew(token_response)

    async def _refresh_token(self):
        """The SDK sends RFC 8707 ``resource`` on a refresh only when the triggering request carries
        MCP-Protocol-Version, and a new session's ``initialize`` carries none: that refresh left
        ``resource`` out and Cloudflare answered 400 while the same token refreshed from any other
        request. Use this provider's last negotiated version (else the SDK's latest) so every
        refresh of a grant is built the same way."""
        if not self.context.protocol_version:
            from mcp.types import LATEST_PROTOCOL_VERSION
            self.context.protocol_version = self._hermes_protocol_version or LATEST_PROTOCOL_VERSION
        await self._hermes_ensure_protected_resource_metadata()
        return await super()._refresh_token()

    async def _hermes_ensure_protected_resource_metadata(self) -> None:
        """A refresh must name the resource the grant was issued for (PRM ``resource``). Without PRM
        the SDK derives it from the server URL, query string included, so Cloudflare's
        ``/mcp?codemode=false`` refreshed as that and got 400 against a grant for ``/mcp``, until the
        401 branch discovered PRM. Cold-loaded providers restore ASM but not PRM: fetch PRM once
        before the first refresh."""
        if self.context.protected_resource_metadata is not None or self._hermes_prm_attempted:
            return
        self._hermes_prm_attempted = True
        from tools.mcp_tool import sdk_httpx
        httpx = sdk_httpx()
        if httpx is None:  # pragma: no cover — SDK import would have failed
            return
        from mcp.client.auth.utils import (
            build_protected_resource_metadata_discovery_urls, create_oauth_metadata_request,
            handle_protected_resource_response)
        async with httpx.AsyncClient(timeout=10.0) as client:
            for url in build_protected_resource_metadata_discovery_urls(None, self.context.server_url):
                try:
                    response = await client.send(create_oauth_metadata_request(url))
                except httpx.HTTPError as exc:
                    logger.debug("MCP OAuth '%s': PRM discovery to %s failed: %s", self._hermes_server_name, url, exc)
                    continue
                prm = await handle_protected_resource_response(response)
                if prm:
                    self.context.protected_resource_metadata = prm
                    return

    async def _hermes_adopt_disk_tokens(self) -> bool:
        """Load tokens another process wrote; True when they differ from ours and are valid."""
        storage = self._hermes_storage()
        disk = await storage.get_tokens() if storage is not None else None
        current = self.context.current_tokens
        if disk is None or not disk.access_token or disk.access_token == getattr(current, "access_token", None):
            return False
        self.context.current_tokens = disk
        self.context.update_token_expiry(disk)
        self._hermes_apply_refresh_skew(disk)
        return self.context.is_token_valid()

    async def _handle_refresh_response(self, response) -> bool:
        """A failed refresh usually means another process rotated the refresh token first: adopt
        what it wrote (re-reading once after a short wait) before clearing and re-authorizing."""
        if 200 <= response.status_code < 300:
            return await super()._handle_refresh_response(response)
        for attempt in range(2):
            if await self._hermes_adopt_disk_tokens():
                logger.info("MCP OAuth '%s': refresh returned %s after another process refreshed; using its tokens",
                            self._hermes_server_name, response.status_code)
                return True
            if attempt or response.status_code not in _REFRESH_RETRY_STATUSES:
                break
            import anyio
            await anyio.sleep(random.uniform(1.0, 3.0))
        error = await self._hermes_refresh_error_code(response)
        true_expiry = self._hermes_true_expiry
        tokens = self.context.current_tokens
        if (response.status_code == 400 and true_expiry and time.time() < true_expiry
                and tokens is not None and tokens.access_token):
            # Some servers (Cloudflare) refuse a refresh while the access token is still valid, which
            # turned the early-refresh margin into an hourly 400 from every process. Keep the valid
            # token and refresh at its real expiry from now on; the refresh lock still serializes it.
            self._hermes_refresh_skew = 0.0
            self.context.token_expiry_time = true_expiry
            logger.info("MCP OAuth '%s': early refresh refused (400 %s) while the token is valid for %ds more; "
                        "keeping it and refreshing at expiry from now on", self._hermes_server_name, error or "?",
                        int(true_expiry - time.time()))
            return True
        if error:
            logger.warning("MCP OAuth '%s': token refresh refused: %s %s", self._hermes_server_name,
                           response.status_code, error)
        return await super()._handle_refresh_response(response)

    @staticmethod
    async def _hermes_refresh_error_code(response) -> Optional[str]:
        """The OAuth ``error`` code of a refused refresh (RFC 6749 §5.2), e.g. invalid_grant; never
        the body or description, which a provider could fill with anything."""
        try:
            body = json.loads(await response.aread())
            code = str(body.get("error") or "")
        except Exception:
            return None
        return code if re.fullmatch(r"[A-Za-z0-9_.-]{1,40}", code) else None

    async def _hermes_refresh_lock_if_due(self):
        """When a refresh is due, take the server's cross-process refresh lock and re-read the
        tokens file; return the held lock file, or None when no refresh is needed (another process
        refreshed while we waited) or the lock could not be had in time."""
        storage = self._hermes_storage()
        if storage is None:
            return None
        async with self.context.lock:
            if not self._initialized:
                await self._initialize()
            if self.context.is_token_valid() or not self.context.can_refresh_token():
                return None
        try:
            import fcntl
        except ImportError:  # pragma: no cover - no advisory file locks on Windows; skew and adoption still apply
            return None
        import anyio
        path = storage._tokens_path().with_name(storage._tokens_path().name + ".refresh.lock")
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = open(path, "a+")
        deadline = time.monotonic() + _REFRESH_LOCK_WAIT_S
        while True:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() > deadline:
                    lock_file.close()
                    logger.warning("MCP OAuth '%s': refresh lock busy for %.0fs; refreshing without it",
                                   self._hermes_server_name, _REFRESH_LOCK_WAIT_S)
                    return None
                await anyio.sleep(0.1)
        async with self.context.lock:
            if await get_manager().invalidate_if_disk_changed(self._hermes_server_name, hermes_home=self._hermes_home):
                await self._initialize()
            if self.context.is_token_valid():
                self._hermes_release_refresh_lock(lock_file)
                return None
        return lock_file

    @staticmethod
    def _hermes_release_refresh_lock(lock_file) -> None:
        if lock_file is None:
            return
        with contextlib.suppress(ImportError, OSError, ValueError):
            import fcntl
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        with contextlib.suppress(OSError):
            lock_file.close()

    async def _prefetch_oauth_metadata(self) -> None:
        """Fetch PRM + ASM from the well-known endpoints before the first request, via the SDK's own URL
        builders/response handlers so we track whatever the pinned SDK expects."""
        # The SDK's httpx flavour, not Hermes': `create_oauth_metadata_request` returns *its* (httpx2) Request objects.
        from tools.mcp_tool import sdk_httpx
        httpx = sdk_httpx()
        if httpx is None:  # pragma: no cover — SDK import would have failed
            return
        from mcp.client.auth.utils import (
            build_oauth_authorization_server_metadata_discovery_urls,
            build_protected_resource_metadata_discovery_urls, create_oauth_metadata_request,
            handle_auth_metadata_response, handle_protected_resource_response)
        server_url = self.context.server_url

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

    def _persist_oauth_metadata_if_changed(self) -> None:
        """Save metadata the SDK discovered lazily (401 branch); no-op when absent/unchanged."""
        meta = self.context.oauth_metadata
        storage = self._hermes_storage()
        if meta is None or storage is None:
            return
        existing = storage.load_oauth_metadata()
        if existing is None or str(existing.token_endpoint) != str(meta.token_endpoint):
            storage.save_oauth_metadata(meta)

    async def _is_invalid_client_at_token_endpoint(self, response: Any) -> bool:
        """True when *response* is the token endpoint (same scheme/host/path, query ignored)
        rejecting our client_id with ``invalid_client`` — whole word, so RFC 7591's
        ``invalid_client_metadata`` does not trip it. The body is read only after the endpoint matches."""
        from urllib.parse import urlsplit
        token_endpoint = getattr(getattr(self.context, "oauth_metadata", None), "token_endpoint", None)
        req = getattr(response, "request", None)
        if not token_endpoint or req is None:
            return False
        try:
            pa, pb = urlsplit(str(req.url)), urlsplit(str(token_endpoint))
        except ValueError:  # pragma: no cover — malformed URL
            return False
        if (pa.scheme, pa.netloc.lower(), pa.path.rstrip("/")) != (pb.scheme, pb.netloc.lower(), pb.path.rstrip("/")):
            return False
        return re.search(rb"\binvalid_client\b", (await response.aread()).lower()) is not None

    async def _maybe_flag_poisoned_client(self, response: Any) -> None:
        """An ``invalid_client`` rejection of our ``client_id`` at the token endpoint proves the cached registration
        is dead server-side: delete ``client.json`` (+ stale metadata) so the SDK re-runs DCR next flow.
        Conservative: acts ONLY on 400/401 at the discovered ``token_endpoint`` (the only request carrying our
        ``client_id``) with ``invalid_client`` in the body; pre-registered clients are never poisoned; any failure
        is swallowed. The browser-side "Redirect URI Mismatch" case has no HTTP signal (``hermes mcp reauth``).

        See #36767.
        """
        try:
            if (self._hermes_preregistered or getattr(response, "status_code", None) not in (400, 401)
                    or not await self._is_invalid_client_at_token_endpoint(response)):
                return
            storage = self._hermes_storage()
            # A rejected CIMD URL would loop if re-presented (the server already fetched and refused
            # it): drop it so the retry takes DCR, and mark it on disk so the next process doesn't walk
            # back into the same refusal (`hermes mcp login` clears the marker).
            cimd_url = getattr(self.context, "client_metadata_url", None)
            if cimd_url and getattr(self.context.client_info, "client_id", None) == cimd_url:
                logger.warning("MCP OAuth '%s': authorization server rejected our Client ID Metadata Document (%s) "
                               "with invalid_client — falling back to dynamic client registration.",
                               self._hermes_server_name, cimd_url)
                self.context.client_metadata_url = None
                if storage is not None:
                    storage.mark_cimd_rejected()
            if storage is not None:
                storage.poison_client_registration()
            # Drop the in-memory client so the SDK re-registers next flow.
            self.context.client_info = None
            self._initialized = False
        except Exception as exc:  # pragma: no cover — must not throw
            self._log_nonfatal("invalid_client detection", exc)

    async def async_auth_flow(self, request):  # type: ignore[override]
        try:  # pre-flow hook: reload from disk if it changed (non-fatal on error)
            await get_manager().invalidate_if_disk_changed(self._hermes_server_name, hermes_home=self._hermes_home)
        except Exception as exc:  # pragma: no cover — defensive
            self._log_nonfatal("pre-flow disk-watch", exc)
        headers = getattr(request, "headers", None)
        negotiated = headers.get("mcp-protocol-version") if headers is not None else None
        if negotiated:
            self._hermes_protocol_version = negotiated
        refresh_lock = None
        try:
            refresh_lock = await self._hermes_refresh_lock_if_due()
        except Exception as exc:  # pragma: no cover — the SDK's own refresh path still runs
            self._log_nonfatal("refresh lock", exc)
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
                    # Any refresh is done and saved by now: let the other processes read it.
                    self._hermes_release_refresh_lock(refresh_lock)
                    refresh_lock = None
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
            self._hermes_release_refresh_lock(refresh_lock)
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
        from tools.mcp_dashboard_oauth import get_dashboard_oauth_flow  # lazy: circular at import time
        from tools.mcp_oauth import _OAUTH_AVAILABLE, OAuthNonInteractiveError, _is_interactive
        from tools.mcp_oauth_provider import build_provider_kwargs, prepare_oauth_config
        if not _OAUTH_AVAILABLE:
            return None
        cfg, storage = prepare_oauth_config(server_name, entry.server_url, entry.oauth_config)
        if get_dashboard_oauth_flow() is None and not _is_interactive() and not storage.has_cached_tokens():
            raise OAuthNonInteractiveError(
                f"MCP OAuth for '{server_name}': non-interactive environment and no cached tokens found. "
                f"Run `hermes mcp login {server_name}` interactively first to complete initial authorization.")
        return _HERMES_PROVIDER_CLS(
            server_name=server_name, preregistered=bool(cfg.get("client_id")), server_url=entry.server_url,
            **build_provider_kwargs(cfg, storage, ssh_proxy_hint=False))

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
