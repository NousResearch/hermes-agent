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
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


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

        def __init__(self, *args: Any, server_name: str = "", **kwargs: Any):
            super().__init__(*args, **kwargs)
            self._hermes_server_name = server_name

        async def async_auth_flow(self, request):  # type: ignore[override]
            # Pre-flow hook: ask the manager to refresh from disk if needed.
            # Any failure here is non-fatal — we just log and proceed with
            # whatever state the SDK already has.
            try:
                await get_manager().invalidate_if_disk_changed(
                    self._hermes_server_name
                )
            except Exception as exc:  # pragma: no cover — defensive
                logger.debug(
                    "MCP OAuth '%s': pre-flow disk-watch failed (non-fatal): %s",
                    self._hermes_server_name, exc,
                )

            # Delegate to the SDK's auth flow
            async for item in super().async_auth_flow(request):
                yield item

    return HermesMCPOAuthProvider


# Cached at import time. Tested and used by :class:`MCPOAuthManager`.
_HERMES_PROVIDER_CLS: Optional[type] = _make_hermes_provider_class()


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
        self._entries: dict[str, _ProviderEntry] = {}
        self._entries_lock = threading.Lock()

    # -- Provider construction / caching -------------------------------------

    def get_or_build_provider(
        self,
        server_name: str,
        server_url: str,
        oauth_config: Optional[dict],
    ) -> Optional[Any]:
        """Return a cached OAuth provider for ``server_name`` or build one.

        Idempotent: repeat calls with the same name return the same instance.
        If ``server_url`` changes for a given name, the cached entry is
        discarded and a fresh provider is built.

        Returns None if the MCP SDK's OAuth support is unavailable.
        """
        with self._entries_lock:
            entry = self._entries.get(server_name)
            if entry is not None and entry.server_url != server_url:
                logger.info(
                    "MCP OAuth '%s': URL changed from %s to %s, discarding cache",
                    server_name, entry.server_url, server_url,
                )
                entry = None

            if entry is None:
                entry = _ProviderEntry(
                    server_url=server_url,
                    oauth_config=oauth_config,
                )
                self._entries[server_name] = entry

            if entry.provider is None:
                entry.provider = self._build_provider(server_name, entry)

            return entry.provider

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
        if _HERMES_PROVIDER_CLS is None:
            logger.warning(
                "MCP OAuth '%s': SDK auth module unavailable", server_name,
            )
            return None

        # Local imports avoid circular deps at module import time.
        from tools.mcp_oauth import (
            HermesTokenStorage,
            _OAUTH_AVAILABLE,
            _build_client_metadata,
            _configure_callback_port,
            _is_interactive,
            _maybe_preregister_client,
            _parse_base_url,
            _redirect_handler,
            _wait_for_callback,
        )

        if not _OAUTH_AVAILABLE:
            return None

        cfg = dict(entry.oauth_config or {})
        storage = HermesTokenStorage(server_name)

        if not _is_interactive() and not storage.has_cached_tokens():
            logger.warning(
                "MCP OAuth for '%s': non-interactive environment and no "
                "cached tokens found. Run interactively first to complete "
                "initial authorization.",
                server_name,
            )

        _configure_callback_port(cfg)
        client_metadata = _build_client_metadata(cfg)
        _maybe_preregister_client(storage, cfg, client_metadata)

        return _HERMES_PROVIDER_CLS(
            server_name=server_name,
            server_url=_parse_base_url(entry.server_url),
            client_metadata=client_metadata,
            storage=storage,
            redirect_handler=_redirect_handler,
            callback_handler=_wait_for_callback,
            timeout=float(cfg.get("timeout", 300)),
        )

    def remove(self, server_name: str) -> None:
        """Evict the provider from cache AND delete tokens from disk.

        Called by ``hermes mcp remove <name>`` and (indirectly) by
        ``hermes mcp login <name>`` during forced re-auth.
        """
        with self._entries_lock:
            self._entries.pop(server_name, None)

        from tools.mcp_oauth import remove_oauth_tokens
        remove_oauth_tokens(server_name)
        logger.info(
            "MCP OAuth '%s': evicted from cache and removed from disk",
            server_name,
        )

    # -- Disk watch ----------------------------------------------------------

    async def invalidate_if_disk_changed(self, server_name: str) -> bool:
        """If the tokens file on disk has a newer mtime than last-seen, force
        the MCP SDK provider to reload its in-memory state.

        Returns True if the cache was invalidated (mtime differed). This is
        the core fix for the external-refresh workflow: a cron job writes
        fresh tokens to disk, and on the next tool call the running MCP
        session picks them up without a restart.
        """
        from tools.mcp_oauth import _get_token_dir, _safe_filename

        entry = self._entries.get(server_name)
        if entry is None or entry.provider is None:
            return False

        async with entry.lock:
            tokens_path = _get_token_dir() / f"{_safe_filename(server_name)}.json"
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
        entry = self._entries.get(server_name)
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

                asyncio.create_task(_do_handle())

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
