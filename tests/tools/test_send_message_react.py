"""Tests for send_message action='react'/'unreact' dispatch.

Kept separate from ``test_send_message_tool.py`` because that module skips
wholesale when optional Telegram dependencies are not installed.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import patch

import tools.send_message_tool as smt
import pytest


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


@pytest.mark.asyncio
async def test_matrix_reactions_use_the_live_adapters_owner_loop():
    from gateway.config import Platform

    owner_loop = asyncio.get_running_loop()
    calls = []

    class MatrixAdapter:
        async def add_reaction(self, chat_id, emoji, message_id=None):
            calls.append(("react", asyncio.get_running_loop() is owner_loop, chat_id, emoji, message_id))
            if asyncio.get_running_loop() is not owner_loop:
                raise RuntimeError("Matrix client used from another loop")
            return {"success": True, "message_id": message_id}

        async def remove_reaction(self, chat_id, message_id=None):
            calls.append(("unreact", asyncio.get_running_loop() is owner_loop, chat_id, message_id))
            if asyncio.get_running_loop() is not owner_loop:
                raise RuntimeError("Matrix client used from another loop")
            return {"success": True, "message_id": message_id}

    runner = SimpleNamespace(_gateway_loop=owner_loop, adapters={Platform.MATRIX: MatrixAdapter()})
    with patch("gateway.run._gateway_runner_ref", lambda: runner):
        react = json.loads(await asyncio.to_thread(
            smt.send_message_tool,
            {"action": "react", "target": "matrix:!room:server", "emoji": "👍", "message_id": "$one"},
        ))
        unreact = json.loads(await asyncio.to_thread(
            smt.send_message_tool,
            {"action": "unreact", "target": "matrix:!room:server", "message_id": "$one"},
        ))

    assert (react, unreact, calls) == (
        {"success": True, "message_id": "$one"},
        {"success": True, "message_id": "$one"},
        [
            ("react", True, "!room:server", "👍", "$one"),
            ("unreact", True, "!room:server", "$one"),
        ],
    )
