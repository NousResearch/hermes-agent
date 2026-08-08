"""Intentional reactions on the Telegram message that started this turn."""

import asyncio
import concurrent.futures
import json

from gateway.session_context import get_session_env
from tools.registry import registry, tool_error


_GATEWAY_REACTION_TIMEOUT = 10.0


def _canonical_standard_emoji(emoji: str) -> str | None:
    """Return PTB's canonical spelling for a standard reaction emoji.

    Telegram's display form may add a variation selector (for example ``❤️``),
    while ``python-telegram-bot``'s ``ReactionEmoji`` enum uses ``❤``. Passing
    the display form makes PTB misclassify it as a custom-emoji ID. Match
    variation-selector-insensitively, but keep ZWJ structure otherwise intact.
    """
    # Telegram's standard list canonically omits the presentation selector on
    # simple one-base reactions such as ❤. Do this without PTB so collection
    # environments that mock optional Telegram modules behave identically.
    if "\u200d" not in emoji and emoji.endswith(("\ufe0e", "\ufe0f")):
        emoji = emoji[:-1]

    try:
        from telegram.constants import ReactionEmoji
    except ImportError:
        # Preserve compatibility with minimal/older Telegram installations;
        # the live adapter/API remains the final validator there.
        return emoji

    allowed = {str(getattr(item, "value", item)) for item in ReactionEmoji}
    if not allowed:
        return emoji
    if emoji in allowed:
        return emoji
    key = emoji.replace("\ufe0e", "").replace("\ufe0f", "")
    matches = {
        item
        for item in allowed
        if item.replace("\ufe0e", "").replace("\ufe0f", "") == key
    }
    return next(iter(matches)) if len(matches) == 1 else None


TELEGRAM_REACTION_SCHEMA = {
    "name": "telegram_react",
    "description": (
        "React to the current inbound Telegram message with one standard emoji. "
        "This is an intentional reaction, not a processing-status signal. "
        "The chat and message are taken from the current Telegram session; no "
        "other target or message can be selected, and this tool cannot send text. "
        "Do not narrate the reaction."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "emoji": {
                "type": "string",
                "description": "A single Telegram-supported standard emoji, such as 👍, ❤, 😂, or 👀.",
            },
        },
        "required": ["emoji"],
        "additionalProperties": False,
    },
}


def telegram_reaction_tool(emoji: str) -> str:
    """React through the live Telegram adapter using task-local session ids."""
    emoji = (emoji or "").strip()
    if not emoji:
        return tool_error("An emoji is required.")
    platform = str(get_session_env("HERMES_SESSION_PLATFORM", "") or "").lower()
    if platform != "telegram":
        return tool_error("This tool is only available in a Telegram session.")

    canonical_emoji = _canonical_standard_emoji(emoji)
    if canonical_emoji is None:
        return tool_error("Telegram does not support that standard reaction emoji.")
    emoji = canonical_emoji

    chat_id = get_session_env("HERMES_SESSION_CHAT_ID", "")
    message_id = get_session_env("HERMES_SESSION_MESSAGE_ID", "")
    session_key = get_session_env("HERMES_SESSION_KEY", "")
    if not chat_id or not message_id or not session_key:
        return tool_error("The current Telegram message context is unavailable.")

    try:
        from gateway.config import Platform
        from gateway.run import _gateway_runner_ref

        runner = _gateway_runner_ref()
        if runner is None:
            return tool_error("The Telegram gateway is not connected.")

        get_source = getattr(runner, "_get_cached_session_source", None)
        resolve_adapter = getattr(runner, "_adapter_for_source", None)
        source = get_source(session_key) if callable(get_source) else None
        if source is None or not callable(resolve_adapter):
            return tool_error("The current Telegram session is unavailable.")

        source_platform = getattr(
            getattr(source, "platform", None),
            "value",
            getattr(source, "platform", None),
        )
        if str(source_platform or "").lower() != Platform.TELEGRAM.value:
            return tool_error("The current session is not a Telegram session.")

        adapter = resolve_adapter(source)
        adapter_platform = getattr(
            getattr(adapter, "platform", None),
            "value",
            getattr(adapter, "platform", None),
        )
        if adapter is None or str(adapter_platform or "").lower() != Platform.TELEGRAM.value:
            return tool_error("The Telegram adapter is not connected.")
        react = getattr(adapter, "add_reaction", None)
        if not callable(react):
            return tool_error("The Telegram adapter is not connected.")

        gateway_loop = getattr(runner, "_gateway_loop", None)
        if (
            gateway_loop is None
            or gateway_loop.is_closed()
            or not gateway_loop.is_running()
        ):
            return tool_error("The Telegram gateway loop is not running.")

        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        if current_loop is gateway_loop:
            # A synchronous tool handler cannot wait on the loop that must run
            # its coroutine. Do not create an orphan coroutine or deadlock the
            # gateway; the caller receives a normal tool error instead.
            return tool_error("Telegram reaction is unavailable on the gateway loop.")

        coroutine = react(chat_id=chat_id, emoji=emoji, message_id=message_id)
        try:
            future = asyncio.run_coroutine_threadsafe(coroutine, gateway_loop)
        except Exception:
            coroutine.close()
            return tool_error("Telegram reaction failed.")
        try:
            result = future.result(timeout=_GATEWAY_REACTION_TIMEOUT)
        except concurrent.futures.TimeoutError:
            future.cancel()
            return tool_error("Telegram reaction timed out.")
        except Exception:
            future.cancel()
            return tool_error("Telegram reaction failed.")
    except Exception:
        return tool_error("Telegram reaction failed.")

    if isinstance(result, dict):
        return json.dumps(result, ensure_ascii=False)
    return json.dumps({"success": bool(result)}, ensure_ascii=False)


registry.register(
    name="telegram_react",
    toolset="telegram_reactions",
    schema=TELEGRAM_REACTION_SCHEMA,
    handler=lambda args, **kw: telegram_reaction_tool(args.get("emoji", "")),
    emoji="💛",
)
