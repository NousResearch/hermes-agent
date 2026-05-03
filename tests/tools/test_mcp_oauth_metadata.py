"""Tests for OAuth server metadata persistence in tools/mcp_oauth.py.

Covers the ``.meta.json`` roundtrip on ``HermesTokenStorage`` and the
``_HermesOAuthProvider`` subclass behavior (restore on init, persist on
auth flow completion).
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp.shared.auth import OAuthMetadata

from tools.mcp_oauth import HermesTokenStorage, _HermesOAuthProvider
from tools.mcp_oauth_manager import _HERMES_PROVIDER_CLS


def _make_metadata(token_endpoint: str = "https://auth.example.com/oauth/token") -> OAuthMetadata:
    return OAuthMetadata.model_validate(
        {
            "issuer": "https://auth.example.com",
            "authorization_endpoint": "https://auth.example.com/oauth/authorize",
            "token_endpoint": token_endpoint,
            "response_types_supported": ["code"],
        }
    )


# ---------------------------------------------------------------------------
# HermesTokenStorage metadata roundtrip
# ---------------------------------------------------------------------------


class TestMetadataStorage:
    def test_save_and_load_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        storage = HermesTokenStorage("example-server")

        meta = _make_metadata()
        storage.save_oauth_metadata(meta)

        meta_path = tmp_path / "mcp-tokens" / "example-server.meta.json"
        assert meta_path.exists()

        loaded = storage.load_oauth_metadata()
        assert loaded is not None
        assert str(loaded.token_endpoint) == "https://auth.example.com/oauth/token"
        assert str(loaded.issuer).rstrip("/") == "https://auth.example.com"

    def test_load_missing_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        storage = HermesTokenStorage("nonexistent")
        assert storage.load_oauth_metadata() is None

    def test_load_corrupt_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        storage = HermesTokenStorage("corrupt-server")

        # Ensure dir exists and write garbage that is not a valid OAuthMetadata
        meta_path = storage._meta_path()
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps({"issuer": "not-a-url", "wrong_field": 123}))

        assert storage.load_oauth_metadata() is None

    def test_remove_deletes_meta_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        storage = HermesTokenStorage("cleanup-server")

        storage.save_oauth_metadata(_make_metadata())
        assert storage._meta_path().exists()

        storage.remove()
        assert not storage._meta_path().exists()


# ---------------------------------------------------------------------------
# _HermesOAuthProvider subclass behavior
# ---------------------------------------------------------------------------


def _provider_with_context(storage: HermesTokenStorage, **context_attrs) -> _HermesOAuthProvider:
    """Build an uninitialized _HermesOAuthProvider with a mocked context.

    Bypasses the full OAuthClientProvider init (which wants a working
    OAuth config) so we can test override logic in isolation.
    """
    provider = _HermesOAuthProvider.__new__(_HermesOAuthProvider)
    context = MagicMock()
    context.storage = storage
    context.oauth_metadata = context_attrs.get("oauth_metadata")
    context.current_tokens = context_attrs.get("current_tokens")
    context.server_url = context_attrs.get("server_url", "https://example.com")
    provider.context = context
    return provider


class TestHermesOAuthProviderInitialize:
    def test_restores_metadata_from_disk(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        storage = HermesTokenStorage("srv")
        meta = _make_metadata("https://auth.example.com/restored/token")
        storage.save_oauth_metadata(meta)

        provider = _provider_with_context(storage, oauth_metadata=None)

        with patch.object(
            _HermesOAuthProvider.__bases__[0], "_initialize", new=AsyncMock()
        ):
            asyncio.run(provider._initialize())

        assert provider.context.oauth_metadata is not None
        assert str(provider.context.oauth_metadata.token_endpoint) == \
            "https://auth.example.com/restored/token"

    def test_skips_restore_when_metadata_already_set(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        storage = HermesTokenStorage("srv2")
        # Disk has older metadata; in-memory context already has a value
        storage.save_oauth_metadata(_make_metadata("https://disk.example.com/token"))
        in_memory = _make_metadata("https://memory.example.com/token")

        provider = _provider_with_context(storage, oauth_metadata=in_memory)

        with patch.object(
            _HermesOAuthProvider.__bases__[0], "_initialize", new=AsyncMock()
        ):
            asyncio.run(provider._initialize())

        # In-memory value should not be overwritten
        assert str(provider.context.oauth_metadata.token_endpoint) == \
            "https://memory.example.com/token"

    def test_skips_when_storage_is_not_hermes(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        provider = _provider_with_context(
            storage=MagicMock(),  # not a HermesTokenStorage
            oauth_metadata=None,
        )

        with patch.object(
            _HermesOAuthProvider.__bases__[0], "_initialize", new=AsyncMock()
        ):
            # Should complete without raising and without touching metadata
            asyncio.run(provider._initialize())

        assert provider.context.oauth_metadata is None


class TestHermesOAuthProviderAuthFlow:
    def test_async_auth_flow_persists_new_metadata(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        storage = HermesTokenStorage("flow-srv")
        assert storage.load_oauth_metadata() is None

        discovered = _make_metadata("https://discovered.example.com/token")
        provider = _provider_with_context(storage, oauth_metadata=discovered)

        # Parent async_auth_flow is an async generator; simulate it completing
        # immediately with zero yields.
        async def fake_parent_flow(self, request):
            if False:
                yield  # pragma: no cover  -- make this an async generator
            return

        with patch.object(
            _HermesOAuthProvider.__bases__[0],
            "async_auth_flow",
            new=fake_parent_flow,
        ):
            async def drive():
                gen = provider.async_auth_flow(MagicMock())
                async for _ in gen:
                    pass

            asyncio.run(drive())

        # Metadata should now be persisted
        loaded = storage.load_oauth_metadata()
        assert loaded is not None
        assert str(loaded.token_endpoint) == "https://discovered.example.com/token"

    def test_async_auth_flow_noop_when_metadata_unchanged(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        storage = HermesTokenStorage("noop-srv")
        meta = _make_metadata("https://same.example.com/token")
        storage.save_oauth_metadata(meta)

        provider = _provider_with_context(storage, oauth_metadata=meta)

        async def fake_parent_flow(self, request):
            if False:
                yield  # pragma: no cover
            return

        with patch.object(
            _HermesOAuthProvider.__bases__[0],
            "async_auth_flow",
            new=fake_parent_flow,
        ), patch.object(
            HermesTokenStorage, "save_oauth_metadata"
        ) as save_spy:
            async def drive():
                gen = provider.async_auth_flow(MagicMock())
                async for _ in gen:
                    pass

            asyncio.run(drive())

            # Should not re-save unchanged metadata
            save_spy.assert_not_called()


# ---------------------------------------------------------------------------
# tools.mcp_oauth_manager production provider path
# ---------------------------------------------------------------------------


def _manager_provider_with_context(storage: HermesTokenStorage, **context_attrs):
    """Build an uninitialized manager provider with a mocked context."""
    if _HERMES_PROVIDER_CLS is None:
        pytest.skip("MCP SDK auth not available")
    provider = _HERMES_PROVIDER_CLS.__new__(_HERMES_PROVIDER_CLS)
    provider._hermes_server_name = context_attrs.get("server_name", "srv")
    context = MagicMock()
    context.storage = storage
    context.oauth_metadata = context_attrs.get("oauth_metadata")
    context.current_tokens = context_attrs.get("current_tokens")
    context.server_url = context_attrs.get("server_url", "https://example.com")
    context.update_token_expiry = MagicMock()
    provider.context = context
    return provider


class TestManagerOAuthProviderMetadata:
    def test_initialize_restores_metadata_from_disk(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        storage = HermesTokenStorage("mgr-srv")
        storage.save_oauth_metadata(_make_metadata("https://mgr.example.com/token"))
        provider = _manager_provider_with_context(storage, oauth_metadata=None)

        with patch.object(
            _HERMES_PROVIDER_CLS.__bases__[0], "_initialize", new=AsyncMock()
        ):
            asyncio.run(provider._initialize())

        assert provider.context.oauth_metadata is not None
        assert str(provider.context.oauth_metadata.token_endpoint) == \
            "https://mgr.example.com/token"

    def test_async_auth_flow_persists_metadata_from_production_provider(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        storage = HermesTokenStorage("mgr-flow")
        provider = _manager_provider_with_context(
            storage,
            oauth_metadata=_make_metadata("https://mgr-flow.example.com/token"),
            server_name="mgr-flow",
        )

        async def fake_parent_flow(self, request):
            if False:
                yield  # pragma: no cover
            return

        manager = MagicMock()
        manager.invalidate_if_disk_changed = AsyncMock(return_value=False)

        with patch.object(
            _HERMES_PROVIDER_CLS.__bases__[0],
            "async_auth_flow",
            new=fake_parent_flow,
        ), patch("tools.mcp_oauth_manager.get_manager", return_value=manager):
            async def drive():
                gen = provider.async_auth_flow(MagicMock())
                async for _ in gen:
                    pass

            asyncio.run(drive())

        loaded = storage.load_oauth_metadata()
        assert loaded is not None
        assert str(loaded.token_endpoint) == "https://mgr-flow.example.com/token"
