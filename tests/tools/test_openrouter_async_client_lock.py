"""TOCTOU guard for tools.openrouter_client.get_async_client (#24731)."""

from __future__ import annotations

import threading
import time
from unittest.mock import patch

import tools.openrouter_client as orc


def test_get_async_client_creates_once_under_contention(monkeypatch):
    """Two concurrent first-callers must construct only one client."""
    monkeypatch.setattr(orc, "_client", None)

    creates = []
    creates_lock = threading.Lock()

    def _fake_resolve(provider, async_mode=False):
        # Hold long enough that a second unguarded caller would also enter.
        time.sleep(0.05)
        client = object()
        with creates_lock:
            creates.append(client)
        return client, "model"

    with patch(
        "agent.auxiliary_client.resolve_provider_client",
        side_effect=_fake_resolve,
    ):
        results = [None, None]

        def _call(idx):
            results[idx] = orc.get_async_client()

        threads = [
            threading.Thread(target=_call, args=(0,)),
            threading.Thread(target=_call, args=(1,)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

    assert results[0] is results[1]
    assert len(creates) == 1, f"expected 1 construct, got {len(creates)}"
    assert orc._client is results[0]
