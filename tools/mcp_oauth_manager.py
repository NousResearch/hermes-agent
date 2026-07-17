#!/usr/bin/env python3
"""Central manager for per-server MCP OAuth state.

One instance shared across the process. Holds per-server OAuth provider
instances and coordinates:

- **Cross-process token reload** via mtime-based disk watch. When an external
  process (e.g. a user cron job) refreshes tokens on disk, the next auth flow
  picks them up without requiring a process restart.
- **401 deduplication** via in-flight futures. When N concurrent tool calls
  all hit 401 with the same access_token, only one recovery attempt fires;
  the rest await the same result.
- **Reconnect signalling** for long-lived MCP sessions. The manager itself
  does not drive reconnection — the `MCPServerTask` in `mcp_tool.py` does —
  but the manager is the single source of truth that decides when reconnect
  is warranted.

Replaces what used to be scattered across eight call sites in `mcp_oauth.py`,
`mcp_tool.py`, and `hermes_cli/mcp_config.py`. This module is the ONLY place
that instantiates the MCP SDK's `OAuthClientProvider` — all other code paths
go through `get_manager()`.

Design reference:

- Claude Code's ``invalidateOAuthCacheIfDiskChanged``
  (``claude-code/src/utils/auth.ts:1320``, CC-1096 / GH#24317). Identical
  external-refresh staleness bug class.
- Codex's ``refresh_oauth_if_needed`` / ``persist_if_needed``
  (``codex-rs/rmcp-client/src/rmcp_client.rs:805``). We lean on the MCP SDK's
  lazy refresh rather than calling refresh before every op, because one
  ``stat()`` per tool call is cheaper than an ``await`` + potential refresh
  round-trip, and the SDK's in-memory expiry path is already correct.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _same_endpoint(a: str, b: str) -> bool:
    """Return True if two URLs target the same endpoint (ignoring query/fragment).

    Compares scheme, host (case-insensitive), and path. Used to confirm a
    rejected response actually came from the OAuth token endpoint before we
    act on an ``invalid_client`` body.
    """
    from urllib.parse import urlsplit

    try:
        pa, pb = urlsplit(a), urlsplit(b)
    except ValueError:  # pragma: no cover — malformed URL
        return False
    return (
        pa.scheme == pb.scheme
        and pa.netloc.lower() == pb.netloc.lower()
        and pa.path.rstrip("/") == pb.path.rstrip("/")
    )


# ---------------------------------------------------------------------------
# Per-server entry
# ---------------------------------------------------------------------------


@dataclass
class _ProviderEntry:
    """Per-server OAuth state tracked by the manager.

    Fields:
        server_url: The MCP server URL used to build the provider. Tracked
            so we can discard a cached provider if the URL changes.
        oauth_config: Optional dict from ``mcp_servers.<name>.oauth``.
        profile_token_dir: Resolved token directory captured for a
            ``client_credentials`` provider. A different profile or credential
            identity must never reuse the cached source-bound provider.
        provider: The ``httpx.Auth``-compatible provider wrapping the MCP
            SDK. None until first use.
        last_mtime_ns: Last-seen ``st_mtime_ns`` of the on-disk tokens file.
            Zero if never read. Used by :meth:`MCPOAuthManager.invalidate_if_disk_changed`
            to detect external refreshes.
        lock: Serialises concurrent access to this entry's state. Bound to
            whichever asyncio loop first awaits it (the MCP event loop).
        pending_401: In-flight 401-handler futures keyed by the failed
            access_token, for deduplicating thundering-herd 401s. Mirrors
            Claude Code's ``pending401Handlers`` map.
    """

    server_url: str
    oauth_config: Optional[dict]
    profile_token_dir: Optional[str] = None
    provider: Optional[Any] = None
    last_mtime_ns: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    pending_401: dict[str, "asyncio.Future[bool]"] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# HermesMCPOAuthProvider — OAuthClientProvider subclass with disk-watch
# ---------------------------------------------------------------------------


def _make_hermes_provider_class() -> Optional[type]:
    """Lazy-import the SDK base class and return our subclass.

    Wrapped in a function so this module imports cleanly even when the
    MCP SDK's OAuth module is unavailable (e.g. older mcp versions).
    """
    try:
        from mcp.client.auth.oauth2 import OAuthClientProvider
    except ImportError:  # pragma: no cover — SDK required in CI
        return None

    class HermesMCPOAuthProvider(OAuthClientProvider):
        """OAuthClientProvider with pre-flow disk-mtime reload.

        Before every ``async_auth_flow`` invocation, asks the manager to
        check whether the tokens file on disk has been modified externally.
        If so, the manager resets ``_initialized`` so the next flow
        re-reads from storage.

        This makes external-process refreshes (cron, another CLI instance)
        visible to the running MCP session without requiring a restart.

        Reference: Claude Code's ``invalidateOAuthCacheIfDiskChanged``
        (``src/utils/auth.ts:1320``, CC-1096 / GH#24317).
        """

        def __init__(
            self,
            *args: Any,
            server_name: str = "",
            preregistered: bool = False,
            **kwargs: Any,
        ):
            super().__init__(*args, **kwargs)
            self._hermes_server_name = server_name
            self._hermes_home = ""
            # When the client_id comes from config.yaml (pre-registered), an
            # invalid_client rejection means the *config* is wrong — deleting
            # client.json would just be re-seeded from config and re-running
            # registration can't help. Only auto-heal dynamically-registered
            # clients. See _maybe_flag_poisoned_client.
            self._hermes_preregistered = preregistered

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

            # Cold-load: restore OAuth server metadata from disk before any
            # refresh attempt. Without this, a restarted process with cached
            # tokens but no in-memory metadata would fall back to the SDK's
            # guessed ``{server_url}/token`` path (returns 404 on most real
            # providers) and require a full browser re-authorization.
            storage = self.context.storage
            from tools.mcp_oauth import HermesTokenStorage
            if (
                isinstance(storage, HermesTokenStorage)
                and self.context.oauth_metadata is None
            ):
                meta = storage.load_oauth_metadata()
                if meta is not None:
                    self.context.oauth_metadata = meta
                    logger.debug(
                        "MCP OAuth '%s': restored metadata from disk "
                        "(token_endpoint=%s)",
                        self._hermes_server_name,
                        meta.token_endpoint,
                    )

            # Pre-flight OAuth AS discovery so ``_refresh_token`` has a
            # correct ``token_endpoint`` before the first refresh attempt.
            # Only runs when we have tokens on cold-load but no cached
            # metadata — i.e. the exact scenario where the SDK's built-in
            # 401-branch discovery hasn't had a chance to run yet.
            if (
                tokens is not None
                and self.context.oauth_metadata is None
            ):
                try:
                    await self._prefetch_oauth_metadata()
                except Exception as exc:  # pragma: no cover — defensive
                    # Non-fatal: if discovery fails, the SDK's normal 401-
                    # branch discovery will run on the next request.
                    logger.debug(
                        "MCP OAuth '%s': pre-flight metadata discovery "
                        "failed (non-fatal): %s",
                        self._hermes_server_name, exc,
                    )

        async def _prefetch_oauth_metadata(self) -> None:
            """Fetch PRM + ASM from the well-known endpoints, cache on context.

            Mirrors the SDK's 401-branch discovery (oauth2.py ~line 511-551)
            but runs synchronously before the first request instead of
            inside the httpx auth_flow generator. Uses the SDK's own URL
            builders and response handlers so we track whatever the SDK
            version we're pinned to expects.
            """
            import httpx  # local import: httpx is an MCP SDK dependency
            from mcp.client.auth.utils import (
                build_oauth_authorization_server_metadata_discovery_urls,
                build_protected_resource_metadata_discovery_urls,
                create_oauth_metadata_request,
                handle_auth_metadata_response,
                handle_protected_resource_response,
            )

            server_url = self.context.server_url
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Step 1: PRM discovery to learn the authorization_server URL.
                for url in build_protected_resource_metadata_discovery_urls(
                    None, server_url
                ):
                    req = create_oauth_metadata_request(url)
                    try:
                        resp = await client.send(req)
                    except httpx.HTTPError as exc:
                        logger.debug(
                            "MCP OAuth '%s': PRM discovery to %s failed: %s",
                            self._hermes_server_name, url, exc,
                        )
                        continue
                    prm = await handle_protected_resource_response(resp)
                    if prm:
                        self.context.protected_resource_metadata = prm
                        if prm.authorization_servers:
                            self.context.auth_server_url = str(
                                prm.authorization_servers[0]
                            )
                        break

                # Step 2: ASM discovery against the auth_server_url (or
                # server_url fallback for legacy providers).
                for url in build_oauth_authorization_server_metadata_discovery_urls(
                    self.context.auth_server_url, server_url
                ):
                    req = create_oauth_metadata_request(url)
                    try:
                        resp = await client.send(req)
                    except httpx.HTTPError as exc:
                        logger.debug(
                            "MCP OAuth '%s': ASM discovery to %s failed: %s",
                            self._hermes_server_name, url, exc,
                        )
                        continue
                    ok, asm = await handle_auth_metadata_response(resp)
                    if not ok:
                        break
                    if asm:
                        self.context.oauth_metadata = asm
                        # Persist immediately so a subsequent cold-load can
                        # skip discovery entirely.
                        storage = self.context.storage
                        from tools.mcp_oauth import HermesTokenStorage
                        if isinstance(storage, HermesTokenStorage):
                            storage.save_oauth_metadata(asm)
                        logger.debug(
                            "MCP OAuth '%s': pre-flight ASM discovered "
                            "token_endpoint=%s",
                            self._hermes_server_name, asm.token_endpoint,
                        )
                        break

        def _persist_oauth_metadata_if_changed(self) -> None:
            """Persist discovered OAuth metadata for future process restarts.

            Called after the SDK's normal 401-branch auth flow completes so
            metadata discovered via the lazy path (not pre-flight) is also
            saved. No-op when nothing to persist or metadata hasn't changed.
            """
            meta = self.context.oauth_metadata
            if meta is None:
                return
            storage = self.context.storage
            from tools.mcp_oauth import HermesTokenStorage
            if not isinstance(storage, HermesTokenStorage):
                return
            existing = storage.load_oauth_metadata()
            if (
                existing is None
                or str(existing.token_endpoint) != str(meta.token_endpoint)
            ):
                storage.save_oauth_metadata(meta)

        async def _maybe_flag_poisoned_client(self, response: Any) -> None:
            """Detect a dead client registration and force re-registration.

            When the IdP rejects our ``client_id`` with ``invalid_client`` on
            the token endpoint (token exchange or refresh), the cached client
            registration is provably dead server-side. We delete ``client.json``
            (+ stale metadata) so the SDK's next ``async_auth_flow`` takes the
            ``if not client_info`` branch and re-runs RFC 7591 dynamic client
            registration. This addresses the recurring manual-reset ritual in
            GH#36767 for the auto-detectable subset (token-endpoint rejection);
            the browser-side "Redirect URI Mismatch" case has no HTTP signal
            and is handled by ``hermes mcp reauth``.

            Conservative by construction — acts ONLY when all hold:
              * status is 400/401,
              * the request hit the discovered ``token_endpoint`` (the only
                request carrying our ``client_id``), and
              * the body carries the ``invalid_client`` error code
                (word-boundary match, so RFC 7591's ``invalid_client_metadata``
                registration error does not trip it).
            Pre-registered (config-supplied) clients are never poisoned.
            Fully best-effort: any failure here is swallowed so a detection
            miss never breaks the live auth flow.

            Covers both the authorization-code token exchange and the
            preemptive refresh — but only when ``token_endpoint`` was
            discovered (``_initialize`` prefetches it on cold-load). If that
            discovery was skipped, the guard returns early and the user falls
            back to ``hermes mcp reauth``.
            """
            try:
                if self._hermes_preregistered:
                    return
                status = getattr(response, "status_code", None)
                if status not in (400, 401):
                    return
                meta = getattr(self.context, "oauth_metadata", None)
                token_endpoint = (
                    str(meta.token_endpoint)
                    if meta is not None and getattr(meta, "token_endpoint", None)
                    else None
                )
                req = getattr(response, "request", None)
                req_url = str(req.url) if req is not None else None
                if not token_endpoint or not req_url:
                    return
                if not _same_endpoint(req_url, token_endpoint):
                    return
                body = await response.aread()
                # Word-boundary match: matches `"error":"invalid_client"` but
                # not the RFC 7591 registration error `invalid_client_metadata`
                # (the trailing `_metadata` removes the right-hand boundary).
                if not re.search(rb"\binvalid_client\b", body.lower()):
                    return

                storage = self.context.storage
                from tools.mcp_oauth import HermesTokenStorage
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

        async def async_auth_flow(self, request):  # type: ignore[override]
            # Pre-flow hook: ask the manager to refresh from disk if needed.
            # Any failure here is non-fatal — we just log and proceed with
            # whatever state the SDK already has.
            try:
                await get_manager().invalidate_if_disk_changed(
                    self._hermes_server_name,
                    hermes_home=self._hermes_home,
                )
            except Exception as exc:  # pragma: no cover — defensive
                logger.debug(
                    "MCP OAuth '%s': pre-flow disk-watch failed (non-fatal): %s",
                    self._hermes_server_name, exc,
                )

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
            try:
                outgoing = await inner.__anext__()
                while True:
                    incoming = yield outgoing
                    # Sniff the response for a dead-client-registration signal
                    # before handing it back to the SDK (best-effort, GH#36767).
                    await self._maybe_flag_poisoned_client(incoming)
                    outgoing = await inner.asend(incoming)
            except StopAsyncIteration:
                # Persist any metadata the SDK discovered lazily during the
                # 401 branch so a subsequent cold-load skips discovery.
                self._persist_oauth_metadata_if_changed()
                return

    return HermesMCPOAuthProvider


# Cached at import time. Tested and used by :class:`MCPOAuthManager`.
_HERMES_PROVIDER_CLS: Optional[type] = _make_hermes_provider_class()


# ---------------------------------------------------------------------------
# HermesClientCredentialsProvider -- non-interactive machine OAuth
# ---------------------------------------------------------------------------


@dataclass
class _ClientCredentialsContext:
    """Minimal context exposed to the manager's generic 401 recovery path."""

    storage: Any
    current_tokens: Optional[Any] = None
    token_expiry_time: Optional[float] = None

    def can_refresh_token(self) -> bool:
        # RFC 6749 client_credentials tokens have no refresh token. The client
        # can always mint a replacement with its registered credentials.
        return True


def _make_client_credentials_provider_class() -> Optional[type]:
    """Build an httpx.Auth provider without relying on a newer MCP SDK API."""
    try:
        import httpx
        from mcp.shared.auth import OAuthToken
    except ImportError:  # pragma: no cover -- dependencies required in CI
        return None

    class HermesClientCredentialsProvider(httpx.Auth):
        """OAuth client_credentials with profile-local token persistence.

        The client secret remains only in the resolved in-memory config. Token
        responses use HermesTokenStorage, which writes mode-0600 files below
        the active HERMES_HOME. Token minting uses a dedicated non-redirecting,
        proxy-free client so a 307/308 cannot replay the form-encoded secret to
        another origin. A 401 mints once and retries once.
        """

        def __init__(
            self,
            *,
            server_name: str,
            token_url: str,
            client_id: str,
            client_secret: str,
            scope: Optional[str],
            resource: Optional[str],
            storage: Any,
        ) -> None:
            self._hermes_server_name = server_name
            self._token_url = token_url
            self._client_id = client_id
            self._client_secret = client_secret
            self._scope = scope
            self._resource = resource
            self.context = _ClientCredentialsContext(storage=storage)
            self._initialized = False
            self._lock = asyncio.Lock()

        async def _initialize(self) -> None:
            tokens = await self.context.storage.get_tokens()
            self.context.current_tokens = tokens
            self.context.token_expiry_time = self._expiry_from(tokens)
            self._initialized = True

        @staticmethod
        def _expiry_from(tokens: Any) -> Optional[float]:
            if tokens is None or tokens.expires_in is None:
                return None
            try:
                lifetime = int(tokens.expires_in)
            except (TypeError, ValueError):
                return None
            if lifetime <= 0:
                return None
            return time.time() + lifetime

        def _token_is_valid(self) -> bool:
            tokens = self.context.current_tokens
            expiry = self.context.token_expiry_time
            return bool(
                tokens is not None
                and tokens.access_token
                and expiry is not None
                and expiry > time.time() + 30
            )

        def _token_request(self) -> "httpx.Request":
            data = {
                "grant_type": "client_credentials",
                "client_id": self._client_id,
                "client_secret": self._client_secret,
            }
            if self._scope:
                data["scope"] = self._scope
            if self._resource:
                data["resource"] = self._resource
            return httpx.Request(
                "POST",
                self._token_url,
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

        async def _accept_token_response(self, response: "httpx.Response") -> None:
            if response.status_code != 200:
                raise RuntimeError(
                    f"MCP OAuth client_credentials token request failed with HTTP {response.status_code}"
                )
            try:
                payload = await response.aread()
                tokens = OAuthToken.model_validate_json(payload)
            except Exception as exc:
                raise RuntimeError(
                    "MCP OAuth client_credentials token response was invalid"
                ) from exc
            expiry = self._expiry_from(tokens)
            if not tokens.access_token or expiry is None:
                raise RuntimeError(
                    "MCP OAuth client_credentials token response lacked a usable expiry"
                )
            self.context.current_tokens = tokens
            self.context.token_expiry_time = expiry
            await self.context.storage.set_tokens(tokens)

        async def _mint_token(self) -> None:
            """Mint once without allowing redirects or environment proxies.

            The outer MCP client may follow redirects for normal tool calls.
            Yielding the token request into that client would allow a 307/308
            response to replay the form body, including ``client_secret``, to
            another origin before this provider can inspect the response.
            Keep credential exchange on a dedicated fail-closed client.
            """
            async with httpx.AsyncClient(
                follow_redirects=False,
                trust_env=False,
                timeout=10.0,
            ) as client:
                response = await client.send(
                    self._token_request(),
                    follow_redirects=False,
                )
            if response.is_redirect:
                raise RuntimeError(
                    "MCP OAuth client_credentials token endpoint redirect refused"
                )
            await self._accept_token_response(response)

        def _authorize(self, request: "httpx.Request") -> None:
            tokens = self.context.current_tokens
            if tokens is None or not tokens.access_token:
                raise RuntimeError("MCP OAuth client_credentials token unavailable")
            request.headers["Authorization"] = f"Bearer {tokens.access_token}"

        async def async_auth_flow(self, request):  # type: ignore[override]
            async with self._lock:
                if not self._initialized:
                    await self._initialize()
                if not self._token_is_valid():
                    await self._mint_token()

                self._authorize(request)
                response = yield request
                if response.status_code != 401:
                    return

                # One bounded replacement attempt. Never enter browser auth.
                self.context.current_tokens = None
                self.context.token_expiry_time = None
                await self._mint_token()
                self._authorize(request)
                yield request

    return HermesClientCredentialsProvider


_CLIENT_CREDENTIALS_PROVIDER_CLS: Optional[type] = (
    _make_client_credentials_provider_class()
)


def _validated_client_credentials_config(
    server_url: str,
    cfg: dict,
) -> tuple[str, str, str, Optional[str], Optional[str]]:
    """Validate non-interactive OAuth config without disclosing credentials."""
    from urllib.parse import urlsplit

    required: dict[str, str] = {}
    for key in ("token_url", "client_id", "client_secret"):
        value = cfg.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"MCP OAuth client_credentials requires non-empty oauth.{key}"
            )
        required[key] = value.strip()

    token = urlsplit(required["token_url"])
    if (
        not token.hostname
        or token.username is not None
        or token.password is not None
        or token.fragment
    ):
        raise ValueError("MCP OAuth client_credentials token_url is unsafe")
    loopback = token.hostname in {"127.0.0.1", "localhost", "::1"}
    if token.scheme != "https" and not (token.scheme == "http" and loopback):
        raise ValueError(
            "MCP OAuth client_credentials token_url must use HTTPS or loopback HTTP"
        )

    server = urlsplit(server_url)
    if (
        not server.hostname
        or server.username is not None
        or server.password is not None
        or server.fragment
    ):
        raise ValueError("MCP OAuth client_credentials MCP URL is unsafe")
    server_loopback = server.hostname in {"127.0.0.1", "localhost", "::1"}
    if server.scheme != "https" and not (
        server.scheme == "http" and server_loopback
    ):
        raise ValueError(
            "MCP OAuth client_credentials MCP URL must use HTTPS or loopback HTTP"
        )

    scope = cfg.get("scope")
    resource = cfg.get("resource")
    if scope is not None and (not isinstance(scope, str) or not scope.strip()):
        raise ValueError("MCP OAuth client_credentials oauth.scope must be a non-empty string")
    if resource is not None and (not isinstance(resource, str) or not resource.strip()):
        raise ValueError("MCP OAuth client_credentials oauth.resource must be a non-empty string")
    return (
        required["token_url"],
        required["client_id"],
        required["client_secret"],
        scope.strip() if isinstance(scope, str) else None,
        resource.strip() if isinstance(resource, str) else None,
    )


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class MCPOAuthManager:
    """Single source of truth for per-server MCP OAuth state.

    Thread-safe: the ``_entries`` dict is guarded by ``_entries_lock`` for
    get-or-create semantics. Per-entry state is guarded by the entry's own
    ``asyncio.Lock`` (used from the MCP event loop thread).
    """

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], _ProviderEntry] = {}
        self._entries_lock = threading.Lock()
        # Holds strong references to in-flight 401 handler tasks so the
        # event loop's weak-reference bookkeeping cannot GC them mid-run
        # and leave `await pending` waiters hanging forever.
        self._inflight_tasks: set[asyncio.Task] = set()

    # -- Provider construction / caching -------------------------------------

    @staticmethod
    def _client_credentials_token_dir(oauth_config: Optional[dict]) -> Optional[str]:
        cfg = dict(oauth_config or {})
        if cfg.get("grant_type", "authorization_code") != "client_credentials":
            return None
        from tools.mcp_oauth import _get_token_dir

        return str(_get_token_dir().expanduser().resolve())

    def get_or_build_provider(
        self,
        server_name: str,
        server_url: str,
        oauth_config: Optional[dict],
    ) -> Optional[Any]:
        """Return a cached OAuth provider for ``server_name`` or build one.

        Idempotent: repeat calls with the same name return the same instance.
        If ``server_url`` changes for a normal authorization-code entry, the
        cached entry is discarded and a fresh provider is built. A
        ``client_credentials`` entry is source-bound and fails closed if the
        same manager/server name is reused with another URL, token directory,
        or credential config; callers must use profile-scoped managers.

        Returns None if the MCP SDK's OAuth support is unavailable.
        """
        key = self._key(server_name)
        normalized_config = (
            copy.deepcopy(oauth_config) if oauth_config is not None else None
        )
        profile_token_dir = self._client_credentials_token_dir(normalized_config)
        with self._entries_lock:
            entry = self._entries.get(key)
            if entry is not None and (
                entry.profile_token_dir is not None or profile_token_dir is not None
            ) and (
                entry.server_url != server_url
                or entry.oauth_config != normalized_config
                or entry.profile_token_dir != profile_token_dir
            ):
                raise RuntimeError(
                    f"MCP OAuth '{server_name}': client_credentials profile or "
                    "credential identity changed inside one manager; use a "
                    "profile-scoped MCP manager"
                )
            if entry is not None and entry.server_url != server_url:
                logger.info(
                    "MCP OAuth '%s': URL changed from %s to %s, discarding cache",
                    server_name, entry.server_url, server_url,
                )
                entry = None

            if entry is None:
                entry = _ProviderEntry(
                    server_url=server_url,
                    oauth_config=normalized_config,
                    profile_token_dir=profile_token_dir,
                )
                self._entries[key] = entry

            if entry.provider is None:
                entry.provider = self._build_provider(server_name, entry)
                if entry.provider is not None:
                    entry.provider._hermes_home = key[0]

            return entry.provider

    @staticmethod
    def _key(
        server_name: str,
        hermes_home: str | Path | None = None,
    ) -> tuple[str, str]:
        from hermes_constants import get_hermes_home

        home = Path(hermes_home) if hermes_home is not None else get_hermes_home()
        return (str(home.expanduser().resolve(strict=False)), server_name)

    def _build_provider(
        self,
        server_name: str,
        entry: _ProviderEntry,
    ) -> Optional[Any]:
        """Build the underlying OAuth provider.

        Constructs :class:`HermesMCPOAuthProvider` directly using the helpers
        extracted from ``tools.mcp_oauth``. The subclass injects a pre-flow
        disk-watch hook so external token refreshes (cron, other CLI
        instances) are visible to running MCP sessions.

        Returns None if the MCP SDK's OAuth support is unavailable.
        """
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
        )

        cfg = dict(entry.oauth_config or {})
        grant_type = cfg.get("grant_type", "authorization_code")

        if not _OAUTH_AVAILABLE:
            if grant_type == "client_credentials":
                raise RuntimeError(
                    f"MCP OAuth '{server_name}': client_credentials dependencies unavailable"
                )
            return None

        if grant_type == "client_credentials":
            if _CLIENT_CREDENTIALS_PROVIDER_CLS is None:
                raise RuntimeError(
                    f"MCP OAuth '{server_name}': client_credentials dependencies unavailable"
                )
            token_url, client_id, client_secret, scope, resource = (
                _validated_client_credentials_config(entry.server_url, cfg)
            )
            # A legacy/auth-code token for the same MCP server name must never
            # be reused under a newly source-scoped machine identity. Namespace
            # by a one-way client-id digest; the secret is neither hashed nor
            # persisted. Rotation to a new client id necessarily starts clean.
            client_namespace = hashlib.sha256(client_id.encode()).hexdigest()[:12]
            storage = HermesTokenStorage(
                f"{server_name}-client-credentials-{client_namespace}"
            )
            return _CLIENT_CREDENTIALS_PROVIDER_CLS(
                server_name=server_name,
                token_url=token_url,
                client_id=client_id,
                client_secret=client_secret,
                scope=scope,
                resource=resource,
                storage=storage,
            )

        if grant_type != "authorization_code":
            raise ValueError(
                f"Unsupported MCP OAuth grant_type for '{server_name}': {grant_type}"
            )

        if _HERMES_PROVIDER_CLS is None:
            logger.warning(
                "MCP OAuth '%s': SDK auth module unavailable", server_name,
            )
            return None

        storage = HermesTokenStorage(server_name)

        from tools.mcp_dashboard_oauth import get_dashboard_oauth_flow

        if (
            get_dashboard_oauth_flow() is None
            and not _is_interactive()
            and not storage.has_cached_tokens()
        ):
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
        callback_handler = _make_callback_waiter(resolved_port)

        return _HERMES_PROVIDER_CLS(
            server_name=server_name,
            preregistered=bool(cfg.get("client_id")),
            server_url=entry.server_url,
            client_metadata=client_metadata,
            storage=storage,
            redirect_handler=redirect_handler,
            callback_handler=callback_handler,
            timeout=float(cfg.get("timeout", 300)),
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

        if entry is not None and entry.provider is not None:
            context = getattr(entry.provider, "context", None)
            storage = getattr(context, "storage", None)
            remove_storage = getattr(storage, "remove", None)
            if callable(remove_storage):
                remove_storage()

        from tools.mcp_oauth import remove_oauth_tokens
        remove_oauth_tokens(server_name, hermes_home=hermes_home)
        logger.info(
            "MCP OAuth '%s': evicted from cache and removed from disk",
            server_name,
        )
        return entry

    def restore_entry(
        self,
        server_name: str,
        entry: _ProviderEntry | None,
        *,
        hermes_home: str | Path | None = None,
    ) -> None:
        """Restore a provider entry removed for a failed reauthorization."""
        if entry is None:
            return
        with self._entries_lock:
            self._entries.setdefault(self._key(server_name, hermes_home), entry)

    def evict(
        self,
        server_name: str,
        *,
        hermes_home: str | Path | None = None,
    ) -> None:
        """Drop only the in-process provider, preserving persisted OAuth state."""
        with self._entries_lock:
            self._entries.pop(self._key(server_name, hermes_home), None)

    # -- Disk watch ----------------------------------------------------------

    async def invalidate_if_disk_changed(
        self,
        server_name: str,
        *,
        hermes_home: str | Path | None = None,
    ) -> bool:
        """If the tokens file on disk has a newer mtime than last-seen, force
        the MCP SDK provider to reload its in-memory state.

        Returns True if the cache was invalidated (mtime differed). This is
        the core fix for the external-refresh workflow: a cron job writes
        fresh tokens to disk, and on the next tool call the running MCP
        session picks them up without a restart.
        """
        entry = self._entries.get(self._key(server_name, hermes_home))
        if entry is None or entry.provider is None:
            return False

        async with entry.lock:
            context = getattr(entry.provider, "context", None)
            storage = getattr(context, "storage", None)
            tokens_path_fn = getattr(storage, "_tokens_path", None)
            if not callable(tokens_path_fn):
                return False
            tokens_path = tokens_path_fn()
            try:
                mtime_ns = tokens_path.stat().st_mtime_ns
            except (FileNotFoundError, OSError):
                return False

            if mtime_ns != entry.last_mtime_ns:
                old = entry.last_mtime_ns
                entry.last_mtime_ns = mtime_ns
                # Force the SDK's OAuthClientProvider to reload from storage
                # on its next auth flow. `_initialized` is private API but
                # stable across the MCP SDK versions we pin (>=1.26.0).
                if hasattr(entry.provider, "_initialized"):
                    entry.provider._initialized = False  # noqa: SLF001
                logger.info(
                    "MCP OAuth '%s': tokens file changed (mtime %d -> %d), "
                    "forcing reload",
                    server_name, old, mtime_ns,
                )
                return True
            return False

    # -- 401 handler (dedup'd) -----------------------------------------------

    async def handle_401(
        self,
        server_name: str,
        failed_access_token: Optional[str] = None,
    ) -> bool:
        """Handle a 401 from a tool call, deduplicated across concurrent callers.

        Returns:
            True  if a (possibly new) access token is now available — caller
                  should trigger a reconnect and retry the operation.
            False if no recovery path exists — caller should surface a
                  ``needs_reauth`` error to the model so it stops hallucinating
                  manual refresh attempts.

        Thundering-herd protection: if N concurrent tool calls hit 401 with
        the same ``failed_access_token``, only one recovery attempt fires.
        Others await the same future.
        """
        entry = self._entries.get(self._key(server_name))
        if entry is None or entry.provider is None:
            return False

        key = failed_access_token or "<unknown>"
        loop = asyncio.get_running_loop()

        async with entry.lock:
            pending = entry.pending_401.get(key)
            if pending is None:
                pending = loop.create_future()
                entry.pending_401[key] = pending

                async def _do_handle() -> None:
                    try:
                        # Step 1: Did disk change? Picks up external refresh.
                        disk_changed = await self.invalidate_if_disk_changed(
                            server_name
                        )
                        if disk_changed:
                            if not pending.done():
                                pending.set_result(True)
                            return

                        # Step 2: No disk change — if the SDK can refresh
                        # in-place, let the caller retry. The SDK's httpx.Auth
                        # flow will issue the refresh on the next request.
                        provider = entry.provider
                        ctx = getattr(provider, "context", None)
                        can_refresh = False
                        if ctx is not None:
                            can_refresh_fn = getattr(ctx, "can_refresh_token", None)
                            if callable(can_refresh_fn):
                                try:
                                    can_refresh = bool(can_refresh_fn())
                                except Exception:
                                    can_refresh = False
                        if not pending.done():
                            pending.set_result(can_refresh)
                    except Exception as exc:  # pragma: no cover — defensive
                        logger.warning(
                            "MCP OAuth '%s': 401 handler failed: %s",
                            server_name, exc,
                        )
                        if not pending.done():
                            pending.set_result(False)
                    finally:
                        entry.pending_401.pop(key, None)

                task = asyncio.create_task(_do_handle())
                self._inflight_tasks.add(task)
                task.add_done_callback(self._inflight_tasks.discard)

        try:
            return await pending
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning(
                "MCP OAuth '%s': awaiting 401 handler failed: %s",
                server_name, exc,
            )
            return False


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------


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
