"""Outbound Discord reaction tools.

Lets LLM-based skills programmatically add or remove emoji reactions on
existing Discord messages. The unified trigger framework already routes
INBOUND reactions to skills (Track 1 PR commits 1-8); these companion
tools close the asymmetry by giving skills an OUTBOUND path so the bot
can pre-attach a reaction (e.g. ``✅``) to its own message and the user
can complete the loop with a single tap.

Two tools are exposed:

* ``discord_add_reaction``    — bot adds an emoji reaction to a message.
* ``discord_remove_reaction`` — bot removes its own emoji reaction
  (used for cleanup when a recommendation expires or is reversed).

Only included in the ``discord`` toolset (same availability check as
``discord_send_button_message``: requires a live Discord gateway adapter
in the process, not just a bot token).

Both tools are **async** because they call into discord.py's async
``message.add_reaction`` / ``remove_reaction``. The registry routes
async handlers via ``_run_async`` automatically when ``is_async=True``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Adapter resolution (mirrors discord_button_tool._get_discord_adapter)
# ---------------------------------------------------------------------------

def _get_discord_adapter():
    """Resolve the running DiscordAdapter from hermes_state.

    Returns None (rather than raising) when the adapter is not initialized
    — callers return a structured error instead of crashing.
    """
    try:
        import hermes_state  # type: ignore
        state = hermes_state.get_state()
        adapter = getattr(state, "adapter", None) or getattr(state, "gateway", None)
        if adapter is None and hasattr(state, "adapters"):
            adapters = state.adapters or {}
            adapter = adapters.get("discord") or adapters.get("DISCORD")
        if adapter is not None and hasattr(adapter, "adapters"):
            inner = adapter.adapters or {}
            adapter = inner.get("discord") or inner.get("DISCORD") or adapter
        return adapter
    except Exception as exc:
        logger.debug("_get_discord_adapter: could not resolve adapter: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Shared lookup: adapter → client → channel → message
# ---------------------------------------------------------------------------

async def _resolve_message(channel_id: str, message_id: str):
    """Resolve a discord.Message from string ids.

    Returns ``(message, client, error_json)``. On failure, ``message`` and
    ``client`` are None and ``error_json`` is a tool_error JSON string.
    """
    adapter = _get_discord_adapter()
    if adapter is None:
        return None, None, tool_error(
            "Discord adapter not initialized. "
            "This tool requires Hermes to be running with a Discord gateway."
        )

    client = getattr(adapter, "_client", None)
    if client is None:
        return None, None, tool_error("Discord client not connected.")

    try:
        channel_id_int = int(channel_id)
    except (ValueError, TypeError):
        return None, None, tool_error(
            f"Invalid channel_id: {channel_id!r} (must be a numeric snowflake)."
        )

    try:
        message_id_int = int(message_id)
    except (ValueError, TypeError):
        return None, None, tool_error(
            f"Invalid message_id: {message_id!r} (must be a numeric snowflake)."
        )

    channel = client.get_channel(channel_id_int)
    if channel is None:
        try:
            channel = await client.fetch_channel(channel_id_int)
        except Exception as exc:
            try:
                import discord as _discord  # type: ignore
                if isinstance(exc, _discord.Forbidden):
                    return None, None, tool_error(
                        f"Bot lacks VIEW_CHANNEL permission for channel {channel_id}."
                    )
                if isinstance(exc, _discord.NotFound):
                    return None, None, tool_error(
                        f"Channel {channel_id} does not exist or bot is not in its guild."
                    )
            except ImportError:
                pass
            return None, None, tool_error(
                f"Channel {channel_id} not found or bot cannot access it: {exc}"
            )

    try:
        message = await channel.fetch_message(message_id_int)
    except Exception as exc:
        try:
            import discord as _discord  # type: ignore
            if isinstance(exc, _discord.NotFound):
                return None, None, tool_error(
                    f"Message {message_id} not found in channel {channel_id}."
                )
            if isinstance(exc, _discord.Forbidden):
                return None, None, tool_error(
                    f"Bot lacks READ_MESSAGE_HISTORY permission in channel {channel_id}."
                )
        except ImportError:
            pass
        return None, None, tool_error(
            f"Could not fetch message {message_id} from channel {channel_id}: {exc}"
        )

    return message, client, None


# ---------------------------------------------------------------------------
# add_reaction
# ---------------------------------------------------------------------------

async def _add_reaction(channel_id: str, message_id: str, emoji: str) -> str:
    """Bot adds ``emoji`` to ``message_id`` in ``channel_id``."""
    message, _client, err = await _resolve_message(channel_id, message_id)
    if err is not None:
        return err

    try:
        import discord as _discord  # type: ignore
    except ImportError:
        return tool_error("discord.py not installed; cannot add reaction.")

    try:
        await message.add_reaction(emoji)
    except _discord.Forbidden:
        return tool_error(
            f"Bot lacks ADD_REACTIONS permission in channel {channel_id}."
        )
    except _discord.NotFound:
        return tool_error(
            f"Message {message_id} or emoji {emoji!r} not found / unusable."
        )
    except _discord.HTTPException as exc:
        return tool_error(f"Discord HTTP error while adding reaction: {exc}")
    except Exception as exc:
        logger.exception(
            "discord_add_reaction: unexpected error on channel=%s message=%s emoji=%r",
            channel_id, message_id, emoji,
        )
        return tool_error(f"Unexpected error: {exc}")

    return tool_result(
        message_id=str(message.id),
        channel_id=str(message.channel.id),
        emoji=emoji,
    )


# ---------------------------------------------------------------------------
# remove_reaction
# ---------------------------------------------------------------------------

async def _remove_reaction(channel_id: str, message_id: str, emoji: str) -> str:
    """Bot removes its own ``emoji`` reaction from ``message_id``."""
    message, client, err = await _resolve_message(channel_id, message_id)
    if err is not None:
        return err

    try:
        import discord as _discord  # type: ignore
    except ImportError:
        return tool_error("discord.py not installed; cannot remove reaction.")

    bot_user = getattr(client, "user", None)
    if bot_user is None:
        return tool_error("Discord client user not available; bot identity unknown.")

    try:
        await message.remove_reaction(emoji, bot_user)
    except _discord.Forbidden:
        return tool_error(
            f"Bot lacks MANAGE_MESSAGES or ADD_REACTIONS permission in channel {channel_id}."
        )
    except _discord.NotFound:
        return tool_error(
            f"Reaction {emoji!r} by bot not found on message {message_id}."
        )
    except _discord.HTTPException as exc:
        return tool_error(f"Discord HTTP error while removing reaction: {exc}")
    except Exception as exc:
        logger.exception(
            "discord_remove_reaction: unexpected error on channel=%s message=%s emoji=%r",
            channel_id, message_id, emoji,
        )
        return tool_error(f"Unexpected error: {exc}")

    return tool_result(
        message_id=str(message.id),
        channel_id=str(message.channel.id),
        emoji=emoji,
    )


# ---------------------------------------------------------------------------
# Sync wrappers for registry dispatch
# ---------------------------------------------------------------------------

def _validate_common(args: Dict[str, Any]) -> Optional[str]:
    """Returns tool_error JSON if a required field is missing/empty, else None."""
    channel_id = str(args.get("channel_id", "")).strip()
    message_id = str(args.get("message_id", "")).strip()
    emoji = str(args.get("emoji", "")).strip()
    if not channel_id:
        return tool_error("'channel_id' is required.")
    if not message_id:
        return tool_error("'message_id' is required.")
    if not emoji:
        return tool_error("'emoji' is required.")
    return None


async def _handler_add_async(args: Dict[str, Any]) -> str:
    err = _validate_common(args)
    if err is not None:
        return err
    return await _add_reaction(
        channel_id=str(args["channel_id"]).strip(),
        message_id=str(args["message_id"]).strip(),
        emoji=str(args["emoji"]).strip(),
    )


async def _handler_remove_async(args: Dict[str, Any]) -> str:
    err = _validate_common(args)
    if err is not None:
        return err
    return await _remove_reaction(
        channel_id=str(args["channel_id"]).strip(),
        message_id=str(args["message_id"]).strip(),
        emoji=str(args["emoji"]).strip(),
    )


# ---------------------------------------------------------------------------
# Availability check (shared)
# ---------------------------------------------------------------------------

def _check_available() -> bool:
    """Available when discord.py is importable AND a bot token is configured."""
    try:
        import discord  # type: ignore  # noqa: F401
    except ImportError:
        return False
    from tools.discord_tool import check_discord_tool_requirements
    return check_discord_tool_requirements()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

_PARAMS: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "channel_id": {
            "type": "string",
            "description": "Discord channel ID (numeric snowflake) where the message lives.",
        },
        "message_id": {
            "type": "string",
            "description": "Discord message ID (numeric snowflake) to react on.",
        },
        "emoji": {
            "type": "string",
            "description": (
                "Emoji to add or remove. Unicode emoji (e.g. '✅') or "
                "a custom guild emoji string ('<:name:id>')."
            ),
        },
    },
    "required": ["channel_id", "message_id", "emoji"],
}

_ADD_SCHEMA: Dict[str, Any] = {
    "name": "discord_add_reaction",
    "description": (
        "Add an emoji reaction to an existing Discord message as the bot. "
        "Pairs with the unified trigger framework's inbound reaction routing: "
        "after sending a message, call this tool to pre-attach ✅ so users "
        "can complete the action with a single tap. Returns "
        "{message_id, channel_id, emoji} on success."
    ),
    "parameters": _PARAMS,
}

_REMOVE_SCHEMA: Dict[str, Any] = {
    "name": "discord_remove_reaction",
    "description": (
        "Remove the bot's own emoji reaction from a Discord message. "
        "Use for cleanup when a recommendation expires, an action is "
        "reversed, or a button no longer applies. Returns "
        "{message_id, channel_id, emoji} on success."
    ),
    "parameters": _PARAMS,
}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

registry.register(
    name="discord_add_reaction",
    toolset="discord",
    schema=_ADD_SCHEMA,
    handler=lambda args, **_kw: _handler_add_async(args),
    check_fn=_check_available,
    requires_env=["DISCORD_BOT_TOKEN"],
    is_async=True,
    description=_ADD_SCHEMA["description"],
    emoji="✅",
)

registry.register(
    name="discord_remove_reaction",
    toolset="discord",
    schema=_REMOVE_SCHEMA,
    handler=lambda args, **_kw: _handler_remove_async(args),
    check_fn=_check_available,
    requires_env=["DISCORD_BOT_TOKEN"],
    is_async=True,
    description=_REMOVE_SCHEMA["description"],
    emoji="🧹",
)
