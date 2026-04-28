"""Tests for the discord_send_button_message outbound tool.

Covers:
1. Happy path — correct SkillButtonView built + channel.send called
2. Multiple buttons — each gets custom_id of shape skill_<name>_<action>
3. Default timeout (180 s)
4. Custom timeout
5. Adapter not initialized — returns error JSON
6. Channel not found — returns error JSON
7. Missing required field — returns error JSON
8. Empty buttons list — returns error JSON
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# discord mock (mirrors _ensure_discord_mock from test_discord_reactions.py)
# ---------------------------------------------------------------------------

def _ensure_discord_mock() -> None:
    """Ensure sys.modules['discord'] has real Exception classes for NotFound/Forbidden/HTTPException.

    Idempotent and resilient to other test files setting up bare-MagicMock discord modules
    earlier in collection (xdist test order is non-deterministic). Real discord.py is left
    alone if it's importable.
    """
    real_discord = sys.modules.get("discord")
    if real_discord is not None and hasattr(real_discord, "__file__"):
        return  # real discord.py installed — leave it alone

    discord_mod = real_discord if real_discord is not None else MagicMock()

    def _is_real_exception(obj: Any) -> bool:
        return isinstance(obj, type) and issubclass(obj, BaseException)

    if not _is_real_exception(getattr(discord_mod, "Forbidden", None)):
        discord_mod.Forbidden = type("Forbidden", (Exception,), {})
    if not _is_real_exception(getattr(discord_mod, "HTTPException", None)):
        discord_mod.HTTPException = type("HTTPException", (Exception,), {})
    if not _is_real_exception(getattr(discord_mod, "NotFound", None)):
        discord_mod.NotFound = type("NotFound", (Exception,), {})

    # NOTE: bare `MagicMock()` returns truthy auto-attrs for any name, so
    # `hasattr(mm, "ui")` is always True. Use stricter type checks to detect
    # whether a previous test (or the real discord.py) actually set the field.
    if not isinstance(getattr(discord_mod, "DMChannel", None), type):
        discord_mod.DMChannel = type("DMChannel", (), {})
    if not isinstance(getattr(discord_mod, "Thread", None), type):
        discord_mod.Thread = type("Thread", (), {})
    if not isinstance(getattr(discord_mod, "ForumChannel", None), type):
        discord_mod.ForumChannel = type("ForumChannel", (), {})
    if getattr(discord_mod, "Interaction", None) is not object:
        discord_mod.Interaction = object
    if not isinstance(getattr(discord_mod, "ButtonStyle", None), SimpleNamespace):
        discord_mod.ButtonStyle = SimpleNamespace(
            primary="primary",
            secondary="secondary",
            success="success",
            danger="danger",
        )
    # Intents.default() must return something; MagicMock auto-attrs do this
    # implicitly so a bare-MagicMock discord_mod is fine without explicit setup.

    ui_mod = getattr(discord_mod, "ui", None)
    if not (isinstance(ui_mod, SimpleNamespace) and isinstance(getattr(ui_mod, "View", None), type)):
        class _FakeView:
            def __init__(self, *, timeout: float = 180.0) -> None:
                self.timeout = timeout
                self.children: List[Any] = []

            def add_item(self, item: Any) -> None:
                self.children.append(item)

        class _FakeButton:
            def __init__(self, *, label: str, custom_id: str, style: Any = "primary") -> None:
                self.label = label
                self.custom_id = custom_id
                self.style = style
                self.callback: Any = None

        discord_mod.ui = SimpleNamespace(View=_FakeView, Button=_FakeButton)

    sys.modules["discord"] = discord_mod
    if "discord.ext" not in sys.modules:
        ext_mod = MagicMock()
        commands_mod = MagicMock()
        commands_mod.Bot = MagicMock
        ext_mod.commands = commands_mod
        sys.modules["discord.ext"] = ext_mod
        sys.modules["discord.ext.commands"] = commands_mod


_ensure_discord_mock()

# Import after mock is in place
from gateway.platforms.discord_interactions import (  # noqa: E402
    SkillButtonView,
    make_skill_custom_id,
)
from tools.discord_button_tool import _handler_async  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter(channel: Any = None, fetch_raises: bool = False) -> MagicMock:
    """Build a minimal fake DiscordAdapter with ._interactions and ._client."""
    adapter = MagicMock()
    handler = MagicMock()
    handler.handle_skill_button_interaction = AsyncMock()
    adapter._interactions = handler

    fake_channel = channel or _make_channel()
    client = MagicMock()
    client.get_channel.return_value = fake_channel
    if fetch_raises:
        client.fetch_channel = AsyncMock(side_effect=Exception("channel not found"))
    else:
        client.fetch_channel = AsyncMock(return_value=fake_channel)
    adapter._client = client
    return adapter


def _make_channel(message_id: str = "9876543210") -> MagicMock:
    """Fake discord.TextChannel with .send() that returns a fake message."""
    channel = MagicMock()
    fake_message = SimpleNamespace(
        id=int(message_id),
        channel=SimpleNamespace(id=111222333),
    )
    channel.send = AsyncMock(return_value=fake_message)
    return channel


def _patch_adapter(adapter: Any):
    """Context-manager patch for _get_discord_adapter."""
    return patch("tools.discord_button_tool._get_discord_adapter", return_value=adapter)


# ---------------------------------------------------------------------------
# Test 1: Happy path — correct view built + channel.send called
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_happy_path_sends_message_and_returns_metadata() -> None:
    channel = _make_channel(message_id="111000111")
    adapter = _make_adapter(channel=channel)

    with _patch_adapter(adapter):
        result_str = await _handler_async({
            "channel_id": "123456789",
            "content": "Approve this request?",
            "skill_name": "approver",
            "buttons": [{"label": "Approve", "action": "approve"}],
        })

    result = json.loads(result_str)
    assert "error" not in result
    assert result["message_id"] == "111000111"
    assert "view_id" in result
    assert result["custom_ids"] == ["skill_approver_approve"]
    channel.send.assert_called_once()
    call_kwargs = channel.send.call_args[1]
    assert call_kwargs["content"] == "Approve this request?"
    assert "view" in call_kwargs


# ---------------------------------------------------------------------------
# Test 2: Multiple buttons — correct custom_ids
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_multiple_buttons_get_correct_custom_ids() -> None:
    channel = _make_channel()
    adapter = _make_adapter(channel=channel)

    with _patch_adapter(adapter):
        result_str = await _handler_async({
            "channel_id": "123456789",
            "content": "Pick one:",
            "skill_name": "voter",
            "buttons": [
                {"label": "Yes", "action": "yes"},
                {"label": "No", "action": "no"},
                {"label": "Maybe", "action": "maybe"},
            ],
        })

    result = json.loads(result_str)
    assert "error" not in result
    assert sorted(result["custom_ids"]) == sorted([
        "skill_voter_yes",
        "skill_voter_no",
        "skill_voter_maybe",
    ])


# ---------------------------------------------------------------------------
# Test 3: Default timeout is 180 seconds
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_default_timeout_is_180() -> None:
    channel = _make_channel()
    adapter = _make_adapter(channel=channel)

    sent_views: List[Any] = []

    async def _capture_send(**kwargs: Any) -> Any:
        sent_views.append(kwargs.get("view"))
        return SimpleNamespace(id=999, channel=SimpleNamespace(id=1))

    channel.send = _capture_send

    with _patch_adapter(adapter):
        result_str = await _handler_async({
            "channel_id": "1",
            "content": "test",
            "skill_name": "sk",
            "buttons": [{"label": "Go", "action": "go"}],
            # timeout_seconds NOT provided → should default to 180
        })

    result = json.loads(result_str)
    assert "error" not in result
    assert len(sent_views) == 1
    assert sent_views[0].timeout == 180.0


# ---------------------------------------------------------------------------
# Test 4: Custom timeout
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_custom_timeout_is_respected() -> None:
    channel = _make_channel()
    adapter = _make_adapter(channel=channel)

    sent_views: List[Any] = []

    async def _capture_send(**kwargs: Any) -> Any:
        sent_views.append(kwargs.get("view"))
        return SimpleNamespace(id=999, channel=SimpleNamespace(id=1))

    channel.send = _capture_send

    with _patch_adapter(adapter):
        result_str = await _handler_async({
            "channel_id": "1",
            "content": "test",
            "skill_name": "sk",
            "buttons": [{"label": "Go", "action": "go"}],
            "timeout_seconds": 60,
        })

    result = json.loads(result_str)
    assert "error" not in result
    assert sent_views[0].timeout == 60.0


# ---------------------------------------------------------------------------
# Test 5: Adapter not initialized — returns error JSON, does not crash
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_adapter_not_initialized_returns_error() -> None:
    with _patch_adapter(None):
        result_str = await _handler_async({
            "channel_id": "1",
            "content": "x",
            "skill_name": "sk",
            "buttons": [{"label": "A", "action": "a"}],
        })

    result = json.loads(result_str)
    assert "error" in result
    assert "adapter" in result["error"].lower() or "not initialized" in result["error"].lower()


# ---------------------------------------------------------------------------
# Test 6: Channel not found — returns error JSON
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_channel_forbidden_returns_perm_error() -> None:
    """fetch_channel raising discord.Forbidden surfaces a permission error,
    distinguishing 'bot lacks VIEW_CHANNEL' from 'channel does not exist'.
    Addresses architect's commit-9 next-pass note #1.
    """
    import discord as _discord_mod
    adapter = _make_adapter(fetch_raises=True)
    adapter._client.get_channel.return_value = None
    adapter._client.fetch_channel = AsyncMock(side_effect=_discord_mod.Forbidden("no view"))

    with _patch_adapter(adapter):
        result_str = await _handler_async({
            "channel_id": "1",
            "content": "x",
            "skill_name": "sk",
            "buttons": [{"label": "A", "action": "a"}],
        })
    result = json.loads(result_str)
    assert "error" in result
    assert "permission" in result["error"].lower() or "VIEW_CHANNEL" in result["error"]


@pytest.mark.asyncio
async def test_channel_not_found_returns_error() -> None:
    adapter = _make_adapter(fetch_raises=True)
    # Make get_channel also return None so fetch_channel is tried
    adapter._client.get_channel.return_value = None

    with _patch_adapter(adapter):
        result_str = await _handler_async({
            "channel_id": "000000000",
            "content": "x",
            "skill_name": "sk",
            "buttons": [{"label": "A", "action": "a"}],
        })

    result = json.loads(result_str)
    assert "error" in result
    assert "not found" in result["error"].lower() or "channel" in result["error"].lower()


# ---------------------------------------------------------------------------
# Test 7: Missing required field — returns error JSON
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_missing_channel_id_returns_error() -> None:
    result_str = await _handler_async({
        "content": "x",
        "skill_name": "sk",
        "buttons": [{"label": "A", "action": "a"}],
    })
    result = json.loads(result_str)
    assert "error" in result
    assert "channel_id" in result["error"]


@pytest.mark.asyncio
async def test_missing_skill_name_returns_error() -> None:
    result_str = await _handler_async({
        "channel_id": "1",
        "content": "x",
        "buttons": [{"label": "A", "action": "a"}],
    })
    result = json.loads(result_str)
    assert "error" in result
    assert "skill_name" in result["error"]


# ---------------------------------------------------------------------------
# Test 8: Empty buttons list — returns error JSON
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_buttons_list_returns_error() -> None:
    result_str = await _handler_async({
        "channel_id": "1",
        "content": "x",
        "skill_name": "sk",
        "buttons": [],
    })
    result = json.loads(result_str)
    assert "error" in result


# ---------------------------------------------------------------------------
# Test 9: Per-button style propagated to view children
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_per_button_style_propagated() -> None:
    channel = _make_channel()
    adapter = _make_adapter(channel=channel)

    sent_views: List[Any] = []

    async def _capture_send(**kwargs: Any) -> Any:
        sent_views.append(kwargs.get("view"))
        return SimpleNamespace(id=1, channel=SimpleNamespace(id=1))

    channel.send = _capture_send

    with _patch_adapter(adapter):
        result_str = await _handler_async({
            "channel_id": "1",
            "content": "Confirm?",
            "skill_name": "confirmer",
            "buttons": [
                {"label": "Yes", "action": "yes", "style": "success"},
                {"label": "No", "action": "no", "style": "danger"},
            ],
        })

    result = json.loads(result_str)
    assert "error" not in result
    view = sent_views[0]
    style_by_label = {child.label: child.style for child in view.children}
    # Styles default to primary; non-primary buttons must differ from primary.
    import discord as _discord_mod
    primary = _discord_mod.ButtonStyle.primary
    assert style_by_label["Yes"] != primary, "success style should differ from primary"
    assert style_by_label["No"] != primary, "danger style should differ from primary"
    # Yes and No should also have different styles from each other.
    assert style_by_label["Yes"] != style_by_label["No"]
