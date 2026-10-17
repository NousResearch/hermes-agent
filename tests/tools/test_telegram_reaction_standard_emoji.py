"""Telegram reactions: only the emoji the SDK's ``ReactionEmoji`` allowlist actually supports.

PTB 22.8 does not refuse an unsupported reaction string — ``Bot.set_message_reaction`` maps every
string outside ``telegram.constants.ReactionEmoji`` to ``ReactionTypeCustomEmoji(custom_emoji_id=...)``
and ships it, so ``🐴``, ``"12345"`` and arbitrary text fail server-side, where the only thing the
adapter could report is its generic "set_message_reaction failed". The adapter therefore validates
against the allowlist first, normalizing the heart clients send (``❤️``) to the allowlist spelling
``❤``.

These tests drive the REAL ``Bot.set_message_reaction`` — only the transport ``Bot._post`` is mocked —
and inspect the ``ReactionType.to_dict()`` payloads PTB would have sent, so they pin the emitted
reaction *type*, not just the adapter's return value.
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("telegram", reason="python-telegram-bot not installed")

from telegram import Bot
from telegram.constants import ReactionEmoji

from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType, ProcessingOutcome
from gateway.session import SessionSource
from hermes_cli.platform_actions import _telegram_add_reaction
from plugins.platforms.telegram.adapter import TelegramAdapter

HEART = "❤"  # U+2764 — the allowlist spelling
HEART_VS16 = "❤\ufe0f"  # what clients/emoji pickers send
DOVE = "🕊"
DOVE_VS16 = "🕊\ufe0f"
HEART_ON_FIRE = "❤\ufe0f\u200d🔥"  # the composed allowlist member (VS16 inside)
HEART_ON_FIRE_NO_VS16 = "❤\u200d🔥"  # not a member: composition with the VS16 removed
UNICORN = "🦄"  # IS a ReactionEmoji member (must be accepted)
HORSE = "🐴"  # is not


@pytest.fixture
def reactions(monkeypatch):
    """A real ``TelegramAdapter`` over a real ``Bot`` whose transport records the outgoing payload.

    ``Bot._post`` is the only mocked layer, so everything in between — the SDK's own reaction
    classification and ``ReactionType.to_dict()`` serialization — is the real thing, and ``sent``
    holds exactly what would have hit the wire (``reaction=None`` for the documented clear).
    """
    sent = []

    async def _post(self, endpoint, data, **kwargs):
        assert endpoint == "setMessageReaction"
        sent.append({"chat_id": data["chat_id"], "message_id": data["message_id"], "reaction": data["reaction"]})
        return True

    monkeypatch.setattr(Bot, "_post", _post)
    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="fake-token")
    adapter._bot = Bot("123:fake-token")
    return SimpleNamespace(adapter=adapter, sent=sent)


def _payload(record):
    """The wire shape of a recorded reaction: ``None`` for a clear, else the ``to_dict()`` list."""
    reaction = record["reaction"]
    return None if reaction is None else [entry.to_dict() for entry in reaction]


def _event(chat_id="123", message_id="456"):
    return MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM, chat_id=chat_id, chat_type="private", user_id="42", user_name="TestUser"),
        message_id=message_id,
    )


# ── every standard reaction reaches the API as an emoji ───────────────

STANDARD_REACTIONS = [pytest.param(member.value, id=member.name) for member in sorted(ReactionEmoji)]


@pytest.mark.asyncio
@pytest.mark.parametrize("emoji", STANDARD_REACTIONS)
async def test_every_standard_reaction_is_sent_as_a_standard_emoji(reactions, emoji):
    """No member of ``set(ReactionEmoji)`` may be dropped, rewritten, or sent as a custom emoji."""
    assert await reactions.adapter._set_reaction("123", "456", emoji) is True
    assert [_payload(record) for record in reactions.sent] == [[{"type": "emoji", "emoji": emoji}]]


@pytest.mark.asyncio
async def test_unicorn_is_accepted_while_the_horse_is_refused(reactions):
    """``🦄`` is a standard reaction (PTB serializes it as an emoji); ``🐴`` is not a reaction at all."""
    assert await reactions.adapter._set_reaction("123", "456", UNICORN) is True
    assert await reactions.adapter._set_reaction("123", "456", HORSE) is False
    assert [_payload(record) for record in reactions.sent] == [[{"type": "emoji", "emoji": UNICORN}]]


# ── variation selectors ──────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("sent_emoji", "expected"),
    [
        pytest.param(HEART, HEART, id="heart"),
        pytest.param(HEART_VS16, HEART, id="heart-vs16"),
        pytest.param(HEART_VS16 + "\ufe0f", HEART, id="heart-double-vs16"),
        pytest.param(DOVE, DOVE, id="dove"),
        pytest.param(DOVE_VS16, DOVE, id="dove-vs16"),
    ],
)
async def test_stray_vs16_codes_are_stripped_to_the_allowlist_spelling(reactions, sent_emoji, expected):
    """``❤️``/``❤️️``/``🕊️`` are only accepted because the VS16-free form is itself a member."""
    assert await reactions.adapter._set_reaction("123", "456", sent_emoji) is True
    assert [_payload(record) for record in reactions.sent] == [[{"type": "emoji", "emoji": expected}]]


@pytest.mark.asyncio
async def test_composed_heart_on_fire_keeps_its_exact_allowlist_payload(reactions):
    """The composed member carries a VS16 of its own: it must be sent verbatim, never stripped."""
    assert await reactions.adapter._set_reaction("123", "456", HEART_ON_FIRE) is True
    assert [_payload(record) for record in reactions.sent] == [[{"type": "emoji", "emoji": HEART_ON_FIRE}]]


@pytest.mark.asyncio
async def test_composed_heart_on_fire_without_the_vs16_is_rejected(reactions):
    """``❤‍🔥`` is nobody's allowlist member, so VS16 stripping must not "repair" it into one."""
    assert await reactions.adapter._set_reaction("123", "456", HEART_ON_FIRE_NO_VS16) is False
    result = await reactions.adapter.add_reaction("123", HEART_ON_FIRE_NO_VS16, message_id="456")
    assert result["success"] is False
    assert reactions.sent == []


@pytest.mark.asyncio
async def test_the_raw_sdk_would_have_shipped_the_heart_as_a_custom_emoji_id(reactions):
    """Root cause, pinned: PTB alone turns ``❤️`` into ``custom_emoji_id="❤️"`` — the adapter
    normalizes it to ``❤`` instead of refusing it."""
    await reactions.adapter._bot.set_message_reaction(chat_id=123, message_id=456, reaction=HEART_VS16)
    assert [_payload(record) for record in reactions.sent] == [[{"type": "custom_emoji", "custom_emoji_id": HEART_VS16}]]


# ── invalid input is refused before the API call ─────────────────────

INVALID_REACTIONS = [
    pytest.param("12345", id="numeric-string"),
    pytest.param(12345, id="numeric-id"),
    pytest.param("5368324170671202286", id="custom-emoji-id"),
    pytest.param(HORSE, id="horse"),
    pytest.param("thumbs up", id="text"),
    pytest.param("👍👎", id="two-emoji"),
    pytest.param("👍🏽", id="skin-tone-modifier"),
    pytest.param("\ufe0f", id="bare-vs16"),
    pytest.param("", id="empty"),
]
# ``None`` is the documented clear for ``_set_reaction``, so it is only invalid agent-facing.
INVALID_AGENT_REACTIONS = INVALID_REACTIONS + [pytest.param(None, id="none")]


@pytest.mark.asyncio
@pytest.mark.parametrize("emoji", INVALID_REACTIONS)
async def test_unsupported_reactions_never_reach_the_transport(reactions, emoji):
    """``_set_reaction`` keeps its bool contract: False, decided before any Bot call."""
    assert await reactions.adapter._set_reaction("123", "456", emoji) is False
    assert reactions.sent == []


@pytest.mark.asyncio
@pytest.mark.parametrize("emoji", INVALID_AGENT_REACTIONS)
async def test_the_agent_gets_an_actionable_error_for_unsupported_reactions(reactions, emoji):
    """The agent-facing contract explains the rule instead of reporting a bare transport failure.

    ``None`` included: the agent-facing path must never fall through to ``_set_reaction(None)``,
    which means "clear all reactions" rather than "react with nothing".
    """
    result = await reactions.adapter.add_reaction("123", emoji, message_id="456")

    assert result["success"] is False
    assert "standard emoji" in result["error"]
    assert "custom emoji IDs are not supported" in result["error"]
    assert result["error"] != "telegram set_message_reaction failed"
    assert reactions.sent == []


# ── the documented clear still works ─────────────────────────────────


@pytest.mark.asyncio
async def test_none_and_remove_reaction_still_clear_the_bot_reactions(reactions):
    """``None`` stays the Bot API clear for ``_set_reaction``/``remove_reaction`` (lifecycle CANCELLED)."""
    assert await reactions.adapter._set_reaction("123", "456", None) is True
    assert (await reactions.adapter.remove_reaction("123", "456"))["success"] is True
    assert [_payload(record) for record in reactions.sent] == [None, None]


# ── lifecycle hooks and the plugin facade share the same normalization ─


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        pytest.param(ProcessingOutcome.SUCCESS, [{"type": "emoji", "emoji": "👍"}], id="success"),
        pytest.param(ProcessingOutcome.FAILURE, [{"type": "emoji", "emoji": "👎"}], id="failure"),
        pytest.param(ProcessingOutcome.CANCELLED, None, id="cancelled"),
    ],
)
async def test_processing_lifecycle_reactions_are_standard_emoji(reactions, monkeypatch, outcome, expected):
    """The automatic 👀/👍/👎/clear lifecycle keeps working and emits the same payload shape."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")

    await reactions.adapter.on_processing_start(_event())
    await reactions.adapter.on_processing_complete(_event(), outcome)

    payloads = [_payload(record) for record in reactions.sent]
    assert payloads == [[{"type": "emoji", "emoji": "👀"}], expected]


@pytest.mark.asyncio
async def test_plugin_platform_actions_facade_shares_the_normalization(reactions):
    """``hermes_cli.platform_actions`` goes through ``_set_reaction``, so it inherits the allowlist."""
    added = await _telegram_add_reaction(reactions.adapter, "123", "456", HEART_VS16)
    refused = await _telegram_add_reaction(reactions.adapter, "123", "456", HORSE)

    assert added["ok"] is True
    assert refused["ok"] is False
    assert [_payload(record) for record in reactions.sent] == [[{"type": "emoji", "emoji": HEART}]]


# ── real transport failures keep their existing contract ─────────────


@pytest.mark.asyncio
async def test_transport_failure_still_reports_the_generic_error(reactions, monkeypatch):
    """A rejected/failed API call is not a validation refusal: bool False, no actionable-emoji claim."""
    import telegram

    async def _boom(self, endpoint, data, **kwargs):
        raise telegram.error.BadRequest("Reaction_invalid")

    monkeypatch.setattr(Bot, "_post", _boom)

    assert await reactions.adapter._set_reaction("123", "456", HEART) is False
    result = await reactions.adapter.add_reaction("123", HEART, message_id="456")
    assert result == {"success": False, "error": "telegram set_message_reaction failed"}
