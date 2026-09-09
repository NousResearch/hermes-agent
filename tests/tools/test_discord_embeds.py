"""Tests for the typed Discord embed builder (tools/discord_api/embeds.py).

Feature M4 of the Discord Omniscience campaign (EPIC #79564).
"""

import pytest

from tools.discord_api.embeds import (
    EMBED_LIMITS,
    Embed,
    EmbedAuthor,
    EmbedField,
    EmbedFooter,
    EmbedValidationError,
    contains_mention,
    embed_to_plain_text,
    validate_embeds,
)


def test_minimal_embed_payload():
    e = Embed(title="Hi", description="body")
    assert e.to_payload() == {"title": "Hi", "description": "body"}


def test_full_embed_payload_roundtrip():
    e = Embed(
        title="T",
        description="D",
        url="https://example.com",
        color=0xFF0000,
        timestamp="2026-08-14T10:00:00Z",
        author=EmbedAuthor("A", url="https://example.com/a", icon_url="https://x/i.png"),
        footer=EmbedFooter("F", icon_url="https://x/f.png"),
        fields=[EmbedField("N", "V", inline=True)],
        image_url="https://x/img.png",
        thumbnail_url="https://x/th.png",
    )
    payload = e.to_payload()
    assert payload["title"] == "T"
    assert payload["color"] == 0xFF0000
    assert payload["author"] == {
        "name": "A", "url": "https://example.com/a", "icon_url": "https://x/i.png"
    }
    assert payload["footer"] == {"text": "F", "icon_url": "https://x/f.png"}
    assert payload["fields"] == [{"name": "N", "value": "V", "inline": True}]
    assert payload["image"] == {"url": "https://x/img.png"}
    assert payload["thumbnail"] == {"url": "https://x/th.png"}


# ── Limit enforcement ────────────────────────────────────────────────────────
def test_title_too_long_rejected():
    with pytest.raises(EmbedValidationError):
        Embed(title="x" * (EMBED_LIMITS["title"] + 1))


def test_description_too_long_rejected():
    with pytest.raises(EmbedValidationError):
        Embed(description="x" * (EMBED_LIMITS["description"] + 1))


def test_field_limits_enforced():
    with pytest.raises(EmbedValidationError):
        EmbedField("x" * (EMBED_LIMITS["field_name"] + 1), "v")
    with pytest.raises(EmbedValidationError):
        EmbedField("n", "x" * (EMBED_LIMITS["field_value"] + 1))


def test_field_count_capped_at_25():
    fields = [EmbedField(f"n{i}", "v") for i in range(EMBED_LIMITS["fields"] + 1)]
    with pytest.raises(EmbedValidationError):
        Embed(fields=fields)


def test_total_character_budget_enforced():
    # Each component stays within its individual limit, but the combined total
    # exceeds the 6000-character aggregate budget. This exercises aggregate
    # validation rather than per-component rejection.
    title = "x" * 256  # within title limit (256)
    desc = "y" * 4096  # within description limit (4096)
    field = EmbedField("z" * 256, "w" * 1024)  # within field limits
    # total = 256 + 4096 + 256 + 1024 = 5632... need to push over 6000
    # Add a second field to exceed the aggregate:
    field2 = EmbedField("a" * 256, "b" * 1024)
    with pytest.raises(EmbedValidationError):
        Embed(title=title, description=desc, fields=[field, field2])


def test_budget_ok_under_limit():
    # title 200 (<256) + description 3000 (<4096) = 3200 total (<6000).
    Embed(title="x" * 200, description="y" * 3000)


# ── Field immutability ────────────────────────────────────────────────────────
def test_fields_converted_to_tuple():
    e = Embed(fields=[EmbedField("n", "v")])
    assert isinstance(e.fields, tuple)
    # Frozen dataclass: cannot mutate
    with pytest.raises((AttributeError, TypeError)):
        e.fields[0] = EmbedField("x", "y")


# ── URL validation ───────────────────────────────────────────────────────────
def test_url_must_be_http():
    with pytest.raises(EmbedValidationError):
        Embed(url="ftp://example.com")
    with pytest.raises(EmbedValidationError):
        Embed(title="t", image_url="javascript:alert(1)")
    with pytest.raises(EmbedValidationError):
        EmbedAuthor("a", icon_url="not-a-url")
    # Missing hostname
    with pytest.raises(EmbedValidationError):
        Embed(url="https://")


def test_url_http_ok():
    Embed(url="https://example.com", image_url="http://example.com/i.png")


def test_url_control_chars_rejected():
    with pytest.raises(EmbedValidationError):
        Embed(url="https://example.com\x00")


# ── color / timestamp validation ─────────────────────────────────────────────
def test_color_must_be_24bit_int():
    with pytest.raises(EmbedValidationError):
        Embed(color=0x1000000)
    with pytest.raises(EmbedValidationError):
        Embed(color=-1)
    with pytest.raises(EmbedValidationError):
        Embed(color="red")
    Embed(color=0xFFFFFF)


def test_timestamp_must_be_iso8601():
    with pytest.raises(EmbedValidationError):
        Embed(timestamp="yesterday")
    Embed(timestamp="2026-08-14T10:00:00Z")
    Embed(timestamp="2026-08-14T10:00:00.123+00:00")


def test_timestamp_invalid_calendar_date_rejected():
    # Matches ISO-8601 pattern but is not a valid calendar date.
    with pytest.raises(EmbedValidationError):
        Embed(timestamp="2026-02-30T00:00:00Z")


# ── Message-level batch validation ───────────────────────────────────────────
def test_validate_embeds_count_limit():
    embeds = [Embed(title=f"t{i}") for i in range(EMBED_LIMITS["per_message"])]
    validate_embeds(embeds)  # exactly 10 → OK
    embeds.append(Embed(title="t10"))
    with pytest.raises(EmbedValidationError):
        validate_embeds(embeds)


def test_validate_embeds_aggregate_budget_exceeded():
    # Each embed is individually valid (within its own 6000-char budget), but
    # the combined total across all embeds in the message exceeds 6000 chars.
    # This exercises the message-level aggregate-budget path, not per-embed
    # rejection.
    embeds = [Embed(description="x" * 3000) for _ in range(3)]
    # 3 embeds × 3000 chars = 9000 > 6000, but each embed is 3000 ≤ 6000
    with pytest.raises(EmbedValidationError):
        validate_embeds(embeds)


def test_validate_embeds_aggregate_budget_ok():
    # 10 embeds at 500 chars each (via description, 4096 limit) = 5000 ≤ 6000 → OK
    embeds = [Embed(description="x" * 500) for _ in range(10)]
    validate_embeds(embeds)


# ── mention policy ───────────────────────────────────────────────────────────
def test_mention_detection():
    assert contains_mention("ping @everyone")
    assert contains_mention("hi @here")
    assert contains_mention("<@123456>")
    assert contains_mention("<@!987654>")
    assert not contains_mention("just plain text")
    assert not contains_mention("email@example.com")


# ── plain-text fallback ──────────────────────────────────────────────────────
def test_plain_text_fallback_preserves_payload():
    e = Embed(
        author=EmbedAuthor("Author"),
        title="Title",
        description="Body line",
        fields=[EmbedField("Field", "Value")],
        footer=EmbedFooter("Foot"),
    )
    text = embed_to_plain_text(e)
    assert "**Author**" in text
    assert "# Title" in text
    assert "Body line" in text
    assert "**Field:** Value" in text
    assert "_Foot_" in text


def test_plain_text_empty_embed():
    assert embed_to_plain_text(Embed()) == ""
