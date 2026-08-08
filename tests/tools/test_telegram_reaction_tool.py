"""Tests for the Telegram-only current-message reaction tool."""

import asyncio
import json
import subprocess
import sys
import threading

from gateway.config import Platform
from gateway.session import SessionSource
from hermes_cli.tools_config import _get_platform_tools
from toolsets import resolve_toolset


def test_telegram_reaction_tool_is_scoped_and_uses_current_source(monkeypatch):
    from tools import telegram_reaction_tool as module

    values = {
        "HERMES_SESSION_PLATFORM": "telegram",
        "HERMES_SESSION_CHAT_ID": "-100",
        "HERMES_SESSION_MESSAGE_ID": "900",
        "HERMES_SESSION_KEY": "secondary-session",
    }
    monkeypatch.setattr(
        module,
        "get_session_env",
        lambda name, default="": values.get(name, default),
    )
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-100",
        chat_type="group",
        user_id="42",
        profile="secondary",
    )
    gateway_loop = asyncio.new_event_loop()
    loop_ready = threading.Event()
    executed = {}

    def run_gateway_loop():
        asyncio.set_event_loop(gateway_loop)
        loop_ready.set()
        gateway_loop.run_forever()

    loop_thread = threading.Thread(target=run_gateway_loop)
    loop_thread.start()
    loop_ready.wait(timeout=2)

    async def add_reaction(**kwargs):
        executed["thread_id"] = threading.get_ident()
        executed["loop"] = asyncio.get_running_loop()
        executed["kwargs"] = kwargs
        return True

    adapter = type("Adapter", (), {"platform": Platform.TELEGRAM})()
    adapter.add_reaction = add_reaction
    runner = type(
        "Runner",
        (),
        {
            "_gateway_loop": gateway_loop,
            "_get_cached_session_source": lambda self, key: source,
            "_adapter_for_source": lambda self, current_source: (
                executed.__setitem__("source", current_source) or adapter
            ),
        },
    )()

    import gateway.run

    monkeypatch.setattr(gateway.run, "_gateway_runner_ref", lambda: runner)
    monkeypatch.setenv("TELEGRAM_REACTIONS", "false")

    try:
        result = json.loads(module.telegram_reaction_tool("❤️"))
    finally:
        gateway_loop.call_soon_threadsafe(gateway_loop.stop)
        loop_thread.join(timeout=2)
        gateway_loop.close()

    assert result == {"success": True}
    assert executed["source"] is source
    assert executed["source"].profile == "secondary"
    assert executed["loop"] is gateway_loop
    assert executed["thread_id"] == loop_thread.ident
    assert executed["kwargs"] == {
        "chat_id": "-100",
        "emoji": "❤",
        "message_id": "900",
    }
    assert set(module.TELEGRAM_REACTION_SCHEMA["parameters"]["properties"]) == {"emoji"}


def test_telegram_reaction_tool_normalizes_variation_selectors():
    from tools import telegram_reaction_tool as module

    assert module._canonical_standard_emoji("❤️") == "❤"
    assert module._canonical_standard_emoji("👍") == "👍"


def test_telegram_reaction_tool_rejects_cached_source_chat_mismatch(monkeypatch):
    from tools import telegram_reaction_tool as module

    values = {
        "HERMES_SESSION_PLATFORM": "telegram",
        "HERMES_SESSION_CHAT_ID": "-999",
        "HERMES_SESSION_MESSAGE_ID": "900",
        "HERMES_SESSION_KEY": "session",
    }
    monkeypatch.setattr(
        module,
        "get_session_env",
        lambda name, default="": values.get(name, default),
    )
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-100",
        chat_type="group",
        user_id="42",
    )
    runner = type(
        "Runner",
        (),
        {
            "_get_cached_session_source": lambda self, key: source,
            "_adapter_for_source": lambda self, current_source: (_ for _ in ()).throw(
                AssertionError("adapter resolution must not run for mismatched context")
            ),
        },
    )()

    import gateway.run

    monkeypatch.setattr(gateway.run, "_gateway_runner_ref", lambda: runner)

    result = json.loads(module.telegram_reaction_tool("👍"))

    assert result == {"error": "The current Telegram message context is unavailable."}


def test_telegram_reaction_tool_rejects_unsupported_standard_emoji(monkeypatch):
    from tools import telegram_reaction_tool as module

    monkeypatch.setattr(
        module,
        "get_session_env",
        lambda name, default="": "telegram" if name == "HERMES_SESSION_PLATFORM" else default,
    )

    result = json.loads(module.telegram_reaction_tool("🧠"))

    assert result == {"error": "Telegram does not support that standard reaction emoji."}


def test_every_installed_standard_reaction_and_display_alias_is_canonicalized():
    # Some adapter tests install optional-dependency mocks in sys.modules at
    # collection time. Use a fresh interpreter to inspect the real PTB enum.
    script = r'''
from telegram.constants import ReactionEmoji
from tools.telegram_reaction_tool import _canonical_standard_emoji

reactions = [str(getattr(item, "value", item)) for item in ReactionEmoji]
assert len(reactions) >= 70
for emoji in reactions:
    assert _canonical_standard_emoji(emoji) == emoji
    if "\u200d" not in emoji and not emoji.endswith(("\ufe0e", "\ufe0f")):
        assert _canonical_standard_emoji(f"{emoji}\ufe0f") == emoji
    if "\u200d" in emoji and "\ufe0f" in emoji:
        assert _canonical_standard_emoji(emoji.replace("\ufe0f", "")) == emoji
print(len(reactions))
'''
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert int(result.stdout.strip()) >= 70


def test_telegram_reaction_tool_is_telegram_only_and_eager():
    from tools import tool_search

    assert "telegram_react" in resolve_toolset("hermes-telegram")
    assert "telegram_react" in resolve_toolset("telegram_reactions")
    assert "telegram_react" not in resolve_toolset("hermes-cli")
    assert "telegram_react" not in resolve_toolset("hermes-discord")
    assert "telegram_reactions" in _get_platform_tools({}, "telegram")
    assert "telegram_reactions" in _get_platform_tools(
        {"platform_toolsets": {"telegram": ["web"]}},
        "telegram",
    )
    assert "telegram_reactions" not in _get_platform_tools({}, "discord")
    assert tool_search.is_deferrable_tool_name("telegram_react") is False


def test_telegram_reaction_tool_rejects_non_telegram_context(monkeypatch):
    from tools import telegram_reaction_tool as module

    values = {
        "HERMES_SESSION_PLATFORM": "discord",
        "HERMES_SESSION_CHAT_ID": "channel",
        "HERMES_SESSION_MESSAGE_ID": "900",
        "HERMES_SESSION_KEY": "session",
    }
    monkeypatch.setattr(
        module,
        "get_session_env",
        lambda name, default="": values.get(name, default),
    )

    result = json.loads(module.telegram_reaction_tool("👍"))

    assert "Telegram session" in result["error"]
