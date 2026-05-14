"""Tests for tools/_running_adapters.py — the running-adapter registry.

The registry exists so that *outbound* code paths in webhook-receive
platforms (Teams, and any future Bot-Framework-style adapter) can reach
the live adapter instance held by the running gateway. Stateless REST
adapters (Telegram, Discord, Feishu, ...) instantiate fresh per call;
webhook-receive adapters cannot, because they hold per-process state
(``_pending_uploads``, ``_conv_refs``) that is the rendezvous point
between an outbound action and a later inbound webhook.

See ``hermes-agent-pilot`` skill, reference
``outbound-media-wiring-by-send-model.md`` for the full architecture
discussion.
"""

import pytest


def test_registry_returns_none_when_no_adapter_registered():
    """An unset platform yields ``None`` rather than raising."""
    from tools._running_adapters import get_running_adapter, clear_running_adapters

    clear_running_adapters()
    assert get_running_adapter("teams") is None


def test_registry_round_trips_a_set_adapter():
    """``set`` + ``get`` returns the same instance."""
    from tools._running_adapters import (
        clear_running_adapters,
        get_running_adapter,
        set_running_adapter,
    )

    clear_running_adapters()
    sentinel = object()
    set_running_adapter("teams", sentinel)
    assert get_running_adapter("teams") is sentinel


def test_registry_isolates_platforms():
    """Setting one platform does not leak into another."""
    from tools._running_adapters import (
        clear_running_adapters,
        get_running_adapter,
        set_running_adapter,
    )

    clear_running_adapters()
    teams = object()
    set_running_adapter("teams", teams)
    assert get_running_adapter("matrix") is None
    assert get_running_adapter("teams") is teams


def test_registry_overwrites_on_reconnect():
    """If the gateway reconnects an adapter, the new instance replaces the old."""
    from tools._running_adapters import (
        clear_running_adapters,
        get_running_adapter,
        set_running_adapter,
    )

    clear_running_adapters()
    first = object()
    second = object()
    set_running_adapter("teams", first)
    set_running_adapter("teams", second)
    assert get_running_adapter("teams") is second


def test_registry_clear_removes_a_specific_platform():
    """``clear_running_adapter(platform)`` drops only that entry."""
    from tools._running_adapters import (
        clear_running_adapter,
        clear_running_adapters,
        get_running_adapter,
        set_running_adapter,
    )

    clear_running_adapters()
    set_running_adapter("teams", object())
    set_running_adapter("matrix", object())

    clear_running_adapter("teams")

    assert get_running_adapter("teams") is None
    assert get_running_adapter("matrix") is not None
