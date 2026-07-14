"""Regression tests for the CDP supervisor reconnect budget."""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock

import pytest

from tools import browser_supervisor as bs


class _FakeWebSocket:
    def __init__(self) -> None:
        self.close = AsyncMock()


async def _fast_sleep(_delay: float) -> None:
    return None


def _ready_supervisor(*, max_attempts: int) -> bs.CDPSupervisor:
    supervisor = bs.CDPSupervisor(
        "reconnect-test",
        "wss://user:password@example.test/devtools?token=endpoint-secret",
        max_reconnect_attempts=max_attempts,
    )
    # The reconnect budget applies after at least one successful attachment.
    supervisor._ready_event.set()
    return supervisor


@pytest.mark.asyncio
async def test_attach_failures_consume_full_reconnect_budget(monkeypatch, caplog):
    """A connected socket with a failed CDP attach is a failed cycle."""
    supervisor = _ready_supervisor(max_attempts=3)
    connect_calls = 0

    async def _connect(*_args, **_kwargs):
        nonlocal connect_calls
        connect_calls += 1
        return _FakeWebSocket()

    async def _parked_reader():
        await asyncio.Event().wait()

    attach_mock = AsyncMock(
        side_effect=RuntimeError(
            "attach failed for "
            "wss://admin:supersecret@example.test/devtools?token=topsecret"
        )
    )
    monkeypatch.setattr(bs.websockets, "connect", _connect)
    monkeypatch.setattr(bs.asyncio, "sleep", _fast_sleep)
    monkeypatch.setattr(supervisor, "_read_loop", _parked_reader)
    monkeypatch.setattr(supervisor, "_attach_initial_page", attach_mock)

    with caplog.at_level(logging.WARNING):
        await supervisor._run()

    assert connect_calls == 3
    assert attach_mock.await_count == 3
    assert "reconnect budget exhausted (3/3)" in caplog.text
    assert "supersecret" not in caplog.text
    assert "topsecret" not in caplog.text


@pytest.mark.asyncio
async def test_quick_session_drops_are_bounded(monkeypatch):
    """Repeated post-attach reader failures consume the same budget."""
    supervisor = _ready_supervisor(max_attempts=2)
    connect = AsyncMock(side_effect=[_FakeWebSocket(), _FakeWebSocket()])

    async def _drop_reader():
        raise RuntimeError("reader dropped")

    attach_mock = AsyncMock(return_value=None)
    monkeypatch.setattr(bs.websockets, "connect", connect)
    monkeypatch.setattr(bs.asyncio, "sleep", _fast_sleep)
    monkeypatch.setattr(supervisor, "_attach_initial_page", attach_mock)
    monkeypatch.setattr(supervisor, "_read_loop", _drop_reader)

    await supervisor._run()

    assert connect.await_count == 2
    assert attach_mock.await_count == 2


@pytest.mark.asyncio
async def test_only_stable_session_resets_failure_streak(monkeypatch):
    """A stable attachment resets failures; a successful connect does not."""
    supervisor = _ready_supervisor(max_attempts=2)
    connect = AsyncMock(
        side_effect=[_FakeWebSocket(), _FakeWebSocket(), _FakeWebSocket()]
    )
    real_sleep = asyncio.sleep
    reader_calls = 0

    async def _reader():
        nonlocal reader_calls
        reader_calls += 1
        if reader_calls == 2:
            await real_sleep(0.02)
        raise RuntimeError(f"drop {reader_calls}")

    attach_mock = AsyncMock(return_value=None)
    monkeypatch.setattr(bs, "RECONNECT_STABLE_RESET_SECONDS", 0.01)
    monkeypatch.setattr(bs.websockets, "connect", connect)
    monkeypatch.setattr(bs.asyncio, "sleep", _fast_sleep)
    monkeypatch.setattr(supervisor, "_attach_initial_page", attach_mock)
    monkeypatch.setattr(supervisor, "_read_loop", _reader)

    await supervisor._run()

    # Quick failure 1; stable cycle resets then drops as failure 1; the third
    # quick failure reaches 2 and exhausts the fresh streak.
    assert connect.await_count == 3
    assert attach_mock.await_count == 3


def test_reconnect_budget_must_be_positive():
    with pytest.raises(ValueError, match="at least 1"):
        bs.CDPSupervisor("test", "ws://example.test", max_reconnect_attempts=0)
