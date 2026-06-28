"""Tests for tui_gateway.loop_noise — the WS peer-hangup teardown filter (#50005)."""

from __future__ import annotations

import asyncio

import pytest

from tui_gateway.loop_noise import (
    _is_benign_teardown,
    install_loop_noise_filter,
)


class _FakeConnectionLostCallback:
    """Stand-in whose repr matches asyncio's ``_call_connection_lost`` flood."""

    def __repr__(self) -> str:
        return "<Handle _ProactorBasePipeTransport._call_connection_lost(None)>"


def test_benign_teardown_matches_reset_in_connection_lost():
    ctx = {
        "exception": ConnectionResetError(10054, "forcibly closed"),
        "handle": _FakeConnectionLostCallback(),
    }
    assert _is_benign_teardown(ctx) is True


def test_benign_teardown_matches_aborted_and_broken_pipe():
    for exc in (
        ConnectionAbortedError(10053, "aborted"),
        BrokenPipeError("epipe"),
    ):
        ctx = {"exception": exc, "callback": _FakeConnectionLostCallback()}
        assert _is_benign_teardown(ctx) is True


def test_reset_outside_connection_lost_is_not_suppressed():
    # Same error type, but NOT from the connection-lost teardown path — must
    # fall through to the default handler.
    ctx = {
        "exception": ConnectionResetError("reset in a real handler"),
        "handle": "<Handle some_other_handler()>",
    }
    assert _is_benign_teardown(ctx) is False


def test_unrelated_exception_is_not_suppressed():
    ctx = {
        "exception": ValueError("boom"),
        "handle": _FakeConnectionLostCallback(),
    }
    assert _is_benign_teardown(ctx) is False


def test_no_exception_is_not_suppressed():
    assert _is_benign_teardown({"message": "loop warning, no exc"}) is False


def test_install_suppresses_flood_and_forwards_real_errors():
    loop = asyncio.new_event_loop()
    try:
        forwarded: list[dict] = []
        loop.set_exception_handler(lambda _loop, ctx: forwarded.append(ctx))

        install_loop_noise_filter(loop)

        # Benign teardown flood → swallowed, not forwarded.
        loop.call_exception_handler(
            {
                "exception": ConnectionResetError(10054, "forcibly closed"),
                "handle": _FakeConnectionLostCallback(),
            }
        )
        assert forwarded == []

        # Genuine loop error → forwarded to the previous handler unchanged.
        real_ctx = {"exception": RuntimeError("genuine loop bug")}
        loop.call_exception_handler(real_ctx)
        assert len(forwarded) == 1
        assert forwarded[0] is real_ctx
    finally:
        loop.close()


def test_install_is_idempotent():
    loop = asyncio.new_event_loop()
    try:
        install_loop_noise_filter(loop)
        first = loop.get_exception_handler()
        install_loop_noise_filter(loop)
        # Second install must NOT wrap again — same handler object.
        assert loop.get_exception_handler() is first
    finally:
        loop.close()


def test_install_falls_back_to_default_handler_when_none_set():
    loop = asyncio.new_event_loop()
    try:
        # No previous handler installed; benign flood still swallowed, and a
        # real error must not raise out of the filter.
        install_loop_noise_filter(loop)
        loop.call_exception_handler(
            {
                "exception": ConnectionResetError(10054, "reset"),
                "handle": _FakeConnectionLostCallback(),
            }
        )
        # A genuine error routes to default_exception_handler — should not raise.
        loop.call_exception_handler({"message": "some loop warning"})
    finally:
        loop.close()
