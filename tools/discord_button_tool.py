"""Outbound Discord button message tool.

Lets LLM-based skills emit a Discord message with a :class:`SkillButtonView`
attached. The LLM specifies a list of buttons; this tool builds the view,
sends the message, and returns structured metadata so the skill can track
which buttons were created.

Only included in the ``discord`` toolset (same availability check as the
existing ``discord`` + ``discord_admin`` tools — requires a live Discord
gateway adapter in the process, not just a bot token).

The tool is **async** because it calls into discord.py's async channel.send.
The registry routes async handlers via ``_run_async`` automatically when
``is_async=True``.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Button style mapping
# ---------------------------------------------------------------------------

_STYLE_MAP: Dict[str, Any] = {}  # populated lazily after discord import


def _resolve_style(style_name: Optional[str]) -> Any:
    """Resolve a style name to a discord.ButtonStyle value.

    Falls back to primary for unknown or absent styles. Lazily imports
    discord so the tool remains importable even when discord.py is absent
    (the check_fn will return False in that case).
    """
    try:
        import discord  # type: ignore
    except ImportError:
        return None

    if not _STYLE_MAP:
        _STYLE_MAP.update({
            "primary": discord.ButtonStyle.primary,
            "secondary": discord.ButtonStyle.secondary,
            "success": discord.ButtonStyle.success,
            "danger": discord.ButtonStyle.danger,
        })

    return _STYLE_MAP.get(style_name or "primary", discord.ButtonStyle.primary)


# ---------------------------------------------------------------------------
# Adapter resolution
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
        # Some hermes_state builds expose adapters via a platform map.
        if adapter is None and hasattr(state, "adapters"):
            adapters = state.adapters or {}
            adapter = adapters.get("discord") or adapters.get("DISCORD")
        # Gateway object may wrap an adapter dict.
        if adapter is not None and hasattr(adapter, "adapters"):
            inner = adapter.adapters or {}
            adapter = inner.get("discord") or inner.get("DISCORD") or adapter
        return adapter
    except Exception as exc:
        logger.debug("_get_discord_adapter: could not resolve adapter: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Core async handler
# ---------------------------------------------------------------------------

async def _send_button_message(
    channel_id: str,
    content: str,
    skill_name: str,
    buttons: List[Dict[str, Any]],
    timeout_seconds: float,
) -> str:
    """Build a SkillButtonView and send the message. Returns JSON string."""
    # --- resolve adapter ---
    adapter = _get_discord_adapter()
    if adapter is None:
        return tool_error(
            "Discord adapter not initialized. "
            "This tool requires Hermes to be running with a Discord gateway."
        )

    handler = getattr(adapter, "_interactions", None)
    if handler is None:
        return tool_error(
            "DiscordInteractionsHandler not found on adapter. "
            "Ensure the adapter was started with discord_interactions support."
        )

    client = getattr(adapter, "_client", None)
    if client is None:
        return tool_error("Discord client not connected.")

    # --- validate channel ---
    try:
        channel_id_int = int(channel_id)
    except (ValueError, TypeError):
        return tool_error(f"Invalid channel_id: {channel_id!r} (must be a numeric snowflake).")

    channel = client.get_channel(channel_id_int)
    if channel is None:
        # get_channel only searches the cache; fall back to fetch_channel.
        try:
            channel = await client.fetch_channel(channel_id_int)
        except Exception as exc:
            try:
                import discord as _discord  # type: ignore
                if isinstance(exc, _discord.Forbidden):
                    return tool_error(
                        f"Bot lacks VIEW_CHANNEL permission for channel {channel_id}."
                    )
                if isinstance(exc, _discord.NotFound):
                    return tool_error(
                        f"Channel {channel_id} does not exist or bot is not in its guild."
                    )
            except ImportError:
                pass
            return tool_error(f"Channel {channel_id} not found or bot cannot access it: {exc}")

    # --- build view ---
    from gateway.platforms.discord_interactions import SkillButtonView, make_skill_custom_id  # noqa: WPS433

    # Build actions dict: {label: action} and per-label styles.
    try:
        import discord as _discord  # type: ignore
    except ImportError:
        return tool_error("discord.py not installed; cannot build button view.")

    actions: Dict[str, str] = {}
    button_styles: Dict[str, Any] = {}  # label → discord.ButtonStyle
    for btn in buttons:
        label = str(btn.get("label", "")).strip()
        action = str(btn.get("action", "")).strip()
        if not label or not action:
            return tool_error(
                f"Each button must have non-empty 'label' and 'action'. Got: {btn!r}"
            )
        actions[label] = action
        button_styles[label] = _resolve_style(btn.get("style"))

    if not actions:
        return tool_error("'buttons' list must not be empty.")

    # Build the view.  SkillButtonView supports per-button styles via an
    # extended constructor — pass them in so the view renders correctly.
    view = SkillButtonView(
        handler=handler,
        skill_name=skill_name,
        actions=actions,
        button_styles=button_styles,
        timeout=float(timeout_seconds),
    )

    # --- send ---
    try:
        message = await channel.send(content=content, view=view)
    except _discord.Forbidden:
        return tool_error(
            f"Bot lacks SEND_MESSAGES (or USE_EXTERNAL_COMPONENTS) permission in channel {channel_id}."
        )
    except _discord.HTTPException as exc:
        return tool_error(f"Discord HTTP error while sending message: {exc}")
    except Exception as exc:
        logger.exception("discord_send_button_message: unexpected error sending to %s", channel_id)
        return tool_error(f"Unexpected error: {exc}")

    # --- collect custom_ids from the built view children ---
    custom_ids = [
        child.custom_id
        for child in view.children
        if hasattr(child, "custom_id")
    ]

    return tool_result(
        message_id=str(message.id),
        channel_id=str(message.channel.id),
        view_id=str(id(view)),
        custom_ids=custom_ids,
    )


# ---------------------------------------------------------------------------
# Sync wrapper for registry dispatch
# ---------------------------------------------------------------------------

async def _handler_async(args: Dict[str, Any]) -> str:
    channel_id = str(args.get("channel_id", "")).strip()
    content = str(args.get("content", "")).strip()
    skill_name = str(args.get("skill_name", "")).strip()
    buttons = args.get("buttons") or []
    timeout_seconds = float(args.get("timeout_seconds") or 180)

    if not channel_id:
        return tool_error("'channel_id' is required.")
    if not content:
        return tool_error("'content' is required.")
    if not skill_name:
        return tool_error("'skill_name' is required.")
    if not isinstance(buttons, list):
        return tool_error("'buttons' must be an array.")

    return await _send_button_message(
        channel_id=channel_id,
        content=content,
        skill_name=skill_name,
        buttons=buttons,
        timeout_seconds=timeout_seconds,
    )


# ---------------------------------------------------------------------------
# Availability check
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
# Schema
# ---------------------------------------------------------------------------

_SCHEMA: Dict[str, Any] = {
    "name": "discord_send_button_message",
    "description": (
        "Send a Discord message with skill-routed buttons attached. "
        "Buttons fire skill triggers.button events when clicked. "
        "Each button must have a label and an action; the tool builds the "
        "canonical custom_id 'skill_<skill_name>_<action>' automatically. "
        "Use this from skills that need 1-tap UX (approvals, polls, menus)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "channel_id": {
                "type": "string",
                "description": "Discord channel ID (numeric snowflake, e.g. '1496609306995458048').",
            },
            "content": {
                "type": "string",
                "description": "Message text body shown above the buttons.",
            },
            "skill_name": {
                "type": "string",
                "description": (
                    "The skill that should receive button click events. "
                    "Used to build custom_ids and validate routing. "
                    "Must match the skill's triggers.button.custom_id_pattern prefix."
                ),
            },
            "buttons": {
                "type": "array",
                "description": "Ordered list of button specs.",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {
                            "type": "string",
                            "description": "Button display text (max 80 chars).",
                        },
                        "action": {
                            "type": "string",
                            "description": (
                                "Action key appended to the custom_id: "
                                "'skill_<skill_name>_<action>'."
                            ),
                        },
                        "style": {
                            "type": "string",
                            "enum": ["primary", "secondary", "success", "danger"],
                            "description": "Button color style (default: primary).",
                        },
                    },
                    "required": ["label", "action"],
                },
                "minItems": 1,
                "maxItems": 25,
            },
            "timeout_seconds": {
                "type": "number",
                "description": "How long the view listens for clicks, in seconds (default 180).",
                "default": 180,
            },
        },
        "required": ["channel_id", "content", "skill_name", "buttons"],
    },
}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

registry.register(
    name="discord_send_button_message",
    toolset="discord",
    schema=_SCHEMA,
    handler=lambda args, **_kw: _handler_async(args),
    check_fn=_check_available,
    requires_env=["DISCORD_BOT_TOKEN"],
    is_async=True,
    description=_SCHEMA["description"],
    emoji="🔘",
)
