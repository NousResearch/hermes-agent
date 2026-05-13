"""Tests for tools/openrouter_client.py — TOCTOU race fix for get_async_client()."""

import threading
from unittest.mock import patch, MagicMock

import pytest

from tools import openrouter_client


class TestGetAsyncClientTOCTOU:
    """Verify get_async_client() uses double-checked locking to prevent TOCTOU races.

    Bug: Two threads calling get_async_client() concurrently could both observe
    _client is None, both call resolve_provider_client(), and race to write
    _client. The loser's orphaned httpx.AsyncClient would leak.

    Fix: Uses _client_cache_lock (same lock used by all other cached clients in
    auxiliary_client.py) with double-checked locking pattern — outer None check
    is lock-free fast path, inner None check inside the lock prevents the race.
    """

    def test_concurrent_calls_create_only_one_client(self, monkeypatch):
        """Simulates the exact race condition: two threads hit get_async_client()
        at the same time. Only one client should be created — the other thread
        should block on the lock, then reuse the already-created client.
        """
        # Reset module state for this test
        monkeypatch.setattr(openrouter_client, "_client", None)

        client_instance_count = 0
        first_thread_done = threading.Event()
        second_thread_can_enter = threading.Event()
        call_order = []

        def mock_resolve_provider_client(provider, async_mode=False):
            nonlocal client_instance_count
            call_order.append(f"resolve-{threading.current_thread().name}")
            if threading.current_thread().name == "Thread-1":
                # Thread-1: signal thread-2 to enter, wait for it
                second_thread_can_enter.set()
                first_thread_done.wait(timeout=5)
            else:
                # Thread-2: wait for thread-1's signal, then proceed
                second_thread_can_enter.wait(timeout=5)
            client_instance_count += 1
            mock_client = MagicMock(name=f"client-{client_instance_count}")
            return mock_client, "openrouter-model"

        with patch.object(
            openrouter_client,
            "resolve_provider_client",
            side_effect=mock_resolve_provider_client
        ):
            results = []

            def thread_target(name):
                client = openrouter_client.get_async_client()
                results.append((name, client))
                if name == "Thread-1":
                    first_thread_done.set()

            t1 = threading.Thread(target=thread_target, name="Thread-1", args=("Thread-1",))
            t2 = threading.Thread(target=thread_target, name="Thread-2", args=("Thread-2",))

            t1.start()
            t2.start()

            t1.join(timeout=10)
            t2.join(timeout=10)

        # Only one client instance should have been created
        assert client_instance_count == 1, (
            f"TOCTOU race occurred: {client_instance_count} clients created, expected 1. "
            f"call_order={call_order}"
        )
        # Both threads should have received the same client object
        assert results[0][1] is results[1][1], (
            "Threads received different client instances — race condition not fixed"
        )

    def test_get_async_client_returns_client_object(self, monkeypatch):
        """Verify get_async_client() returns the cached client, not the tuple."""
        monkeypatch.setattr(openrouter_client, "_client", None)

        mock_client = MagicMock()

        with patch.object(
            openrouter_client,
            "resolve_provider_client",
            return_value=(mock_client, "openrouter/test-model")
        ):
            result = openrouter_client.get_async_client()
            # The function returns _client (the cached client object),
            # not the (client, model) tuple from resolve_provider_client.
            assert result is mock_client, (
                f"Expected the cached client object, got {type(result).__name__}"
            )

    def test_raises_value_error_when_api_key_missing(self, monkeypatch):
        """If OPENROUTER_API_KEY is missing, resolve_provider_client returns None."""
        monkeypatch.setattr(openrouter_client, "_client", None)

        with patch.object(
            openrouter_client,
            "resolve_provider_client",
            return_value=(None, None)
        ):
            with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
                openrouter_client.get_async_client()

    def test_client_reused_on_subsequent_calls(self, monkeypatch):
        """After first call, subsequent calls should not hit the lock or resolver."""
        monkeypatch.setattr(openrouter_client, "_client", None)

        mock_client = MagicMock()
        resolve_call_count = 0

        def mock_resolve(provider, async_mode=False):
            nonlocal resolve_call_count
            resolve_call_count += 1
            return mock_client, "openrouter/model"

        with patch.object(
            openrouter_client,
            "resolve_provider_client",
            side_effect=mock_resolve
        ):
            # First call — triggers creation
            client1 = openrouter_client.get_async_client()
            # Second call — reuses existing client, no lock contention
            client2 = openrouter_client.get_async_client()
            # Third call — still reused
            client3 = openrouter_client.get_async_client()

        assert resolve_call_count == 1, (
            f"resolve_provider_client called {resolve_call_count} times, expected 1. "
            "Client not being reused correctly."
        )
        assert client1 is client2 is client3

    def test_check_api_key_returns_bool(self):
        """Sanity check that check_api_key still works after the refactor."""
        result = openrouter_client.check_api_key()
        assert isinstance(result, bool)