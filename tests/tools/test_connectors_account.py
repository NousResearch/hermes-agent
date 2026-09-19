"""Account-owned managed connector operations."""

from __future__ import annotations

import threading

from tests.fakes.connectors_managed import FakeManagedClient
from tools.connectors import live


def test_account_connect_joins_the_open_operation_without_a_second_mint(monkeypatch):
    from tools.connectors import account, managed

    client = FakeManagedClient()
    settled = threading.Event()
    live.reset_for_tests()
    monkeypatch.setattr(managed, "managed_client", lambda: client)
    original_close = live.close

    def close(operation):
        original_close(operation)
        settled.set()

    monkeypatch.setattr(live, "close", close)
    try:
        first = account.find_or_start_operation(["gmail"], action="connect", profile_home=None)
        assert account.wait_for_prepare(first)
        second = account.find_or_start_operation(["gmail"], action="connect", profile_home=None)
        assert second.started is False and second.operation is first.operation and client.mints == 1
        client.connected.set()
        assert settled.wait(2)
    finally:
        live.reset_for_tests()
