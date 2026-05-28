"""Tests for Exa credential-pool integration.

Covers pool-first credential resolution in :func:`plugins.web.exa.provider._get_exa_client`
and the rotate-and-retry behaviour of :func:`_call_with_rotation` on 401/403/429.
"""

from __future__ import annotations

import os
import sys
import pytest
from unittest.mock import MagicMock, patch


class _FakePoolEntry:
    def __init__(self, key: str) -> None:
        self.runtime_api_key = key
        self.access_token = key


class _FakePool:
    """Drop-in pool stub with controllable select / rotate."""

    def __init__(self, entries):
        self._entries = list(entries)
        self._current = 0
        self.rotate_calls = []

    def has_credentials(self) -> bool:
        return bool(self._entries)

    def select(self):
        if not self._entries:
            return None
        return self._entries[self._current]

    def mark_exhausted_and_rotate(self, status_code=None, **kwargs):  # noqa: ARG002
        self.rotate_calls.append(status_code)
        if self._current + 1 < len(self._entries):
            self._current += 1
            return self._entries[self._current]
        return None


class _SDKError(Exception):
    """Stand-in for an Exa SDK exception that carries an HTTP status."""

    def __init__(self, status_code: int, msg: str = "exa error"):
        super().__init__(msg)
        self.status_code = status_code


@pytest.fixture(autouse=True)
def _reset_exa_client_cache():
    """Drop the cached Exa client around every test so resolution re-runs."""
    import tools.web_tools as _wt

    _wt._exa_client = None
    yield
    _wt._exa_client = None


@pytest.fixture
def fake_exa_module(monkeypatch):
    """Inject a fake ``exa_py`` module whose ``Exa`` class records constructions."""
    constructed = []

    class _FakeExa:
        def __init__(self, api_key: str):
            self.api_key = api_key
            self.headers = {}
            constructed.append(api_key)

    fake_module = MagicMock()
    fake_module.Exa = _FakeExa
    monkeypatch.setitem(sys.modules, "exa_py", fake_module)
    return constructed


class TestExaCredentialPool:
    """Verify pool-first resolution + 401/403/429 rotation for Exa."""

    def test_pool_entry_preferred_over_env(self, fake_exa_module):
        pool = _FakePool([_FakePoolEntry("pool-key-1")])
        with patch.dict(os.environ, {"EXA_API_KEY": "env-shadowed"}):
            with patch("plugins.web.exa.provider.load_pool", return_value=pool):
                from plugins.web.exa.provider import _get_exa_client

                client = _get_exa_client()
                assert client.api_key == "pool-key-1"
                assert fake_exa_module == ["pool-key-1"]

    def test_env_used_when_pool_empty(self, fake_exa_module):
        pool = _FakePool([])
        with patch.dict(os.environ, {"EXA_API_KEY": "env-key"}):
            with patch("plugins.web.exa.provider.load_pool", return_value=pool):
                from plugins.web.exa.provider import _get_exa_client

                client = _get_exa_client()
                assert client.api_key == "env-key"
                assert fake_exa_module == ["env-key"]

    def test_raises_when_no_credential(self):
        pool = _FakePool([])
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("EXA_API_KEY", None)
            with patch("plugins.web.exa.provider.load_pool", return_value=pool):
                from plugins.web.exa.provider import _get_exa_client

                with pytest.raises(ValueError, match="EXA_API_KEY"):
                    _get_exa_client()

    def test_429_rotates_and_retries(self, fake_exa_module):
        pool = _FakePool([_FakePoolEntry("key-A"), _FakePoolEntry("key-B")])
        calls = []

        def fn(client):
            calls.append(client.api_key)
            if len(calls) == 1:
                raise _SDKError(429, "rate limited")
            return {"ok": True}

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("EXA_API_KEY", None)
            with patch("plugins.web.exa.provider.load_pool", return_value=pool):
                from plugins.web.exa.provider import _call_with_rotation

                result = _call_with_rotation(fn)
                assert result == {"ok": True}
                assert calls == ["key-A", "key-B"]
                assert pool.rotate_calls == [429]
                # Cache should hold the rotated client.
                assert fake_exa_module == ["key-A", "key-B"]

    def test_500_does_not_rotate(self, fake_exa_module):
        pool = _FakePool([_FakePoolEntry("key-A"), _FakePoolEntry("key-B")])

        def fn(client):
            raise _SDKError(500, "server error")

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("EXA_API_KEY", None)
            with patch("plugins.web.exa.provider.load_pool", return_value=pool):
                from plugins.web.exa.provider import _call_with_rotation

                with pytest.raises(_SDKError):
                    _call_with_rotation(fn)
                assert pool.rotate_calls == []
                # Only the initial client was built.
                assert fake_exa_module == ["key-A"]

    def test_429_with_env_only_does_not_rotate(self, fake_exa_module):
        pool = _FakePool([])
        calls = []

        def fn(client):
            calls.append(client.api_key)
            raise _SDKError(429, "rate limited")

        with patch.dict(os.environ, {"EXA_API_KEY": "env-only"}):
            with patch("plugins.web.exa.provider.load_pool", return_value=pool):
                from plugins.web.exa.provider import _call_with_rotation

                with pytest.raises(_SDKError):
                    _call_with_rotation(fn)
                assert calls == ["env-only"]
                assert pool.rotate_calls == []

    def test_status_extracted_from_message_when_attr_missing(self, fake_exa_module):
        """If the SDK exception lacks status_code, fall back to regex on str(exc)."""
        pool = _FakePool([_FakePoolEntry("key-A"), _FakePoolEntry("key-B")])
        calls = []

        class _StringOnly(Exception):
            pass

        def fn(client):
            calls.append(client.api_key)
            if len(calls) == 1:
                raise _StringOnly("HTTP 401 Unauthorized from exa")
            return {"ok": True}

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("EXA_API_KEY", None)
            with patch("plugins.web.exa.provider.load_pool", return_value=pool):
                from plugins.web.exa.provider import _call_with_rotation

                result = _call_with_rotation(fn)
                assert result == {"ok": True}
                assert pool.rotate_calls == [401]
                assert calls == ["key-A", "key-B"]

    def test_is_available_via_pool_only(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("EXA_API_KEY", None)
            with patch("hermes_cli.auth.read_credential_pool", return_value=[{"id": "k1"}]):
                from plugins.web.exa.provider import ExaWebSearchProvider

                assert ExaWebSearchProvider().is_available() is True

    def test_is_available_false_when_no_env_and_empty_pool(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("EXA_API_KEY", None)
            with patch("hermes_cli.auth.read_credential_pool", return_value=[]):
                from plugins.web.exa.provider import ExaWebSearchProvider

                assert ExaWebSearchProvider().is_available() is False
