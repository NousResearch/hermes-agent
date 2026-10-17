"""Tests for send_message action='react'/'unreact' dispatch.

Kept separate from ``test_send_message_tool.py`` because that module skips
wholesale when optional Telegram dependencies are not installed.
"""

import asyncio
import json
import threading
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import tools.send_message_tool as smt


class _FakePhotonAdapter:
    """Adapter exposing add_reaction/remove_reaction coroutines."""

    def __init__(self):
        self.calls = []

    async def add_reaction(self, chat_id, emoji, message_id=None):
        self.calls.append(("add", chat_id, emoji, message_id))
        return {"success": True, "emoji": emoji}

    async def remove_reaction(self, chat_id, message_id=None):
        self.calls.append(("remove", chat_id, message_id))
        return {"success": True}


class _NoReactionAdapter:
    """Adapter with no reaction support at all."""


class _LoopRecordingAdapter:
    """Adapter whose reaction call records the loop it ran on.

    An adapter reached through the live gateway keeps loop-bound state (an
    ``httpx.AsyncClient``, queues, locks created on the gateway loop), so a
    reaction awaited on any other loop raises
    ``RuntimeError: ... is bound to a different event loop`` (#139314).
    """

    def __init__(self):
        self.loops = []

    async def add_reaction(self, chat_id, emoji, message_id=None):
        self.loops.append(id(asyncio.get_running_loop()))
        return {"success": True, "emoji": emoji}


@contextmanager
def _running_gateway_loop():
    """A real gateway loop, running in its own thread."""
    loop = asyncio.new_event_loop()
    started = threading.Event()

    def _run():
        asyncio.set_event_loop(loop)
        started.set()
        loop.run_forever()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    started.wait(timeout=2)
    try:
        yield loop
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)
        loop.close()


def _runner_with(adapter):
    from gateway.config import Platform

    return SimpleNamespace(adapters={Platform("photon"): adapter})


def _call(args):
    return json.loads(smt.send_message_tool(args))


def test_react_dispatches_to_add_reaction():
    adapter = _FakePhotonAdapter()
    with patch("gateway.run._gateway_runner_ref", lambda: _runner_with(adapter)):
        result = _call(
            {"action": "react", "target": "photon:+15551234567", "emoji": "❤️"}
        )
    assert result["success"] is True
    assert adapter.calls == [("add", "+15551234567", "❤️", None)]


def test_react_without_live_gateway():
    with patch("gateway.run._gateway_runner_ref", lambda: None):
        result = _call(
            {"action": "react", "target": "photon:+15551234567", "emoji": "👍"}
        )
    assert result.get("success") is not True
    assert "live" in json.dumps(result)


def test_react_runs_on_the_gateway_loop():
    """The reaction is scheduled on the gateway loop, never on the private loop
    ``_run_async`` spins up for the caller (the send path already dispatches
    through ``_dispatch_on_gateway_loop`` for the same reason)."""
    from gateway.config import Platform

    adapter = _LoopRecordingAdapter()
    with _running_gateway_loop() as gateway_loop:
        runner = SimpleNamespace(
            adapters={Platform("photon"): adapter},
            _gateway_loop=gateway_loop,
        )
        with patch("gateway.run._gateway_runner_ref", lambda: runner):
            result = _call(
                {"action": "react", "target": "photon:+155****4567", "emoji": "👍"}
            )
    assert result["success"] is True
    assert adapter.loops == [id(gateway_loop)]


def test_react_without_a_gateway_loop_still_reacts():
    """Standalone/CLI callers keep the direct await: no ``_gateway_loop`` on the
    runner, so there is no foreign loop to dispatch onto."""
    adapter = _LoopRecordingAdapter()
    with patch("gateway.run._gateway_runner_ref", lambda: _runner_with(adapter)):
        result = _call(
            {"action": "react", "target": "photon:+155****4567", "emoji": "👍"}
        )
    assert result["success"] is True
    assert len(adapter.loops) == 1
