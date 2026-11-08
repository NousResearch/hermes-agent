"""``send_message`` react/unreact must reach the platform on the GATEWAY loop.

Regression for the reaction-dispatch class of bug: ``send`` has always gone through
``_dispatch_on_gateway_loop`` (adapter coroutines touch state bound to the gateway's loop — the
Telegram ``Bot`` session, the WhatsApp bridge's aiohttp session), but ``_handle_react`` awaited the
adapter coroutine on the tool worker's own loop. Reacting from a turn therefore either blocked the
worker or raced a connection owned by a loop it did not run on.

Also pins the agent-facing contract end-to-end against the real Telegram adapter: the emoji and the
``None`` clear reach the Bot API, and this works while the *automatic* lifecycle reactions
(``TELEGRAM_REACTIONS``) are off — those gates must not leak into the explicit contract.
"""

import asyncio
import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

import tools.send_message_tool as smt
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult


class _RecordingBot:
    """Mock Bot transport: records every ``set_message_reaction`` call and the loop it ran on."""

    def __init__(self):
        self.calls = []
        self.loops = []

    async def set_message_reaction(self, **kwargs):
        self.calls.append(kwargs)
        self.loops.append(asyncio.get_running_loop())


def _telegram_adapter(**extra):
    """A real TelegramAdapter built without __init__ (no network, no token validation)."""
    pytest.importorskip("telegram", reason="python-telegram-bot not installed")
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="fake-token", extra=dict(extra))
    adapter._bot = _RecordingBot()
    return adapter


@pytest.fixture
def gateway_loop():
    """A running loop in ANOTHER thread — the gateway's loop, as seen by the tool worker."""
    loop = asyncio.new_event_loop()
    started = threading.Event()
    # Queued before run_forever so the callback fires on the FIRST iteration: the loop is provably
    # running before a test dispatches onto it, instead of racing the thread's startup.
    loop.call_soon(started.set)
    thread = threading.Thread(target=loop.run_forever, name="test-gateway-loop", daemon=True)
    thread.start()
    assert started.wait(timeout=5), "gateway loop thread never started"
    try:
        yield loop
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=10)
        assert not thread.is_alive(), "gateway loop thread did not stop"
        loop.close()


def _runner(adapter, loop):
    return SimpleNamespace(adapters={Platform.TELEGRAM: adapter}, _gateway_loop=loop)


def _call(args):
    return json.loads(smt.send_message_tool(args))


def _react_target(chat_id="123"):
    return f"telegram:{chat_id}"


def test_react_and_unreact_reach_bot_api_on_the_gateway_loop(gateway_loop, monkeypatch):
    """👍 then None (the documented clear) must reach ``set_message_reaction`` — on the gateway loop.

    ``reactions: false`` reproduces the stock config every install materializes: the automatic 👀/✅
    lifecycle indicators are off, and the explicit ``send_message`` action still works.
    """
    monkeypatch.delenv("TELEGRAM_REACTIONS", raising=False)
    adapter = _telegram_adapter(reactions=False)
    assert adapter._reactions_enabled() is False

    with patch("gateway.run._gateway_runner_ref", lambda: _runner(adapter, gateway_loop)):
        reacted = _call({"action": "react", "target": _react_target(), "emoji": "👍", "message_id": "456"})
        unreacted = _call({"action": "unreact", "target": _react_target(), "message_id": "456"})

    assert reacted["success"] is True, reacted
    assert unreacted["success"] is True, unreacted
    assert adapter._bot.calls == [
        {"chat_id": 123, "message_id": 456, "reaction": "👍"},
        {"chat_id": 123, "message_id": 456, "reaction": None},
    ]
    # The Bot session lives on the gateway loop; a reaction awaited anywhere else races it.
    assert adapter._bot.loops == [gateway_loop, gateway_loop]


def test_reaction_dispatch_uses_gateway_loop_not_the_tool_loop(gateway_loop):
    """Independent of Telegram: the reaction coroutine itself must be handed to the gateway loop."""

    class _Adapter:
        SUPPORTS_REACTIONS = True

        def __init__(self):
            self.loops = []

        async def add_reaction(self, chat_id, emoji, message_id=None):
            self.loops.append(asyncio.get_running_loop())
            return {"success": True}

        async def remove_reaction(self, chat_id, message_id=None):
            self.loops.append(asyncio.get_running_loop())
            return {"success": True}

    adapter = _Adapter()
    with patch("gateway.run._gateway_runner_ref", lambda: _runner(adapter, gateway_loop)):
        react = _call({"action": "react", "target": _react_target(), "emoji": "👀", "message_id": "m1"})
        unreact = _call({"action": "unreact", "target": _react_target(), "message_id": "m1"})

    assert react["success"] is True and unreact["success"] is True
    assert adapter.loops == [gateway_loop, gateway_loop]


def test_react_without_message_id_fails_cleanly_without_calling_the_bot(gateway_loop):
    """Telegram has no "react to the latest" affordance: refuse explicitly, never guess."""
    adapter = _telegram_adapter(reactions=False)

    with patch("gateway.run._gateway_runner_ref", lambda: _runner(adapter, gateway_loop)):
        result = _call({"action": "react", "target": _react_target(), "emoji": "👍"})

    assert result["success"] is False
    assert "message_id" in result["error"]
    assert adapter._bot.calls == []


class _ConcreteAdapter(BasePlatformAdapter):
    """Minimal CONCRETE adapter: the abstract surface every real platform implements.

    ``object.__new__`` is not used here — ``BasePlatformAdapter`` owns no I/O in ``__init__``, and
    inheriting the real base is what makes the reaction-capability resolution under test the
    production one (base mixin stubs + flag), not a hand-rolled duck type.
    """

    def __init__(self):
        self.platform = Platform.TELEGRAM
        self.sent = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append((chat_id, content))
        return SendResult(success=True, message_id="1")

    async def get_chat_info(self, chat_id):
        return {"name": "test", "type": "dm"}


class _LegacyReactAdapter(_ConcreteAdapter):
    """Pre-flag shape: overrides ``add_reaction`` but never declares ``SUPPORTS_REACTIONS``.

    Method presence alone must advertise the capability — flag-less overrides predate the flag.
    """

    def __init__(self):
        super().__init__()
        self.reactions = []

    async def add_reaction(self, chat_id, emoji, message_id=None):
        self.reactions.append({"chat_id": chat_id, "emoji": emoji, "message_id": message_id})
        return {"success": True}


class _OptOutReactAdapter(_LegacyReactAdapter):
    """Same override, but the subclass explicitly declares it cannot react."""

    SUPPORTS_REACTIONS = False


def _capability_adapter(mode):
    return {
        "legacy": _LegacyReactAdapter,
        "base_only": _ConcreteAdapter,
        "optout": _OptOutReactAdapter,
    }[mode]()


@pytest.mark.parametrize("mode", ["legacy", "base_only", "optout"])
def test_public_react_honors_the_adapter_declared_capability(gateway_loop, mode):
    """The agent-facing contract, end to end: a legacy override reacts, a declared opt-out does not.

    An adapter that only inherits the base stubs must answer "not supported" from the TOOL, never
    by awaiting a stub and reporting an empty success.
    """
    adapter = _capability_adapter(mode)

    with patch("gateway.run._gateway_runner_ref", lambda: _runner(adapter, gateway_loop)):
        result = _call({"action": "react", "target": _react_target(), "emoji": "👍", "message_id": "1"})

    if mode == "legacy":
        assert result["success"] is True, result
        assert adapter.reactions == [{"chat_id": "123", "emoji": "👍", "message_id": "1"}]
        return

    # Refused by the tool before any dispatch (the base stub's own message names the adapter).
    assert result.get("success") is not True, result
    assert result["error"] == "Platform 'telegram' does not support message reactions."
    assert getattr(adapter, "reactions", []) == []


@pytest.mark.parametrize("mode", ["bad_request", "invalid_id", "disconnected"])
def test_telegram_react_failures_surface_through_the_public_tool(gateway_loop, mode):
    """The real adapter's three failure shapes reach the caller as ``success: False`` + an error.

    Telegram rejects the reaction (``BadRequest: Reaction_invalid``), the id is not numeric, or the
    Bot session is gone — none of them may look like a successful reaction.
    """
    adapter = _telegram_adapter(reactions=False)
    message_id = "456"
    if mode == "bad_request":
        import telegram

        adapter._bot.set_message_reaction = AsyncMock(
            side_effect=telegram.error.BadRequest("Reaction_invalid")
        )
    elif mode == "invalid_id":
        message_id = "not-an-id"
    else:
        adapter._bot = None

    with patch("gateway.run._gateway_runner_ref", lambda: _runner(adapter, gateway_loop)):
        result = _call({"action": "react", "target": _react_target(), "emoji": "👍", "message_id": message_id})

    assert result["success"] is False, result
    assert result["error"]
    if mode == "bad_request":
        assert adapter._bot.set_message_reaction.await_count == 1
