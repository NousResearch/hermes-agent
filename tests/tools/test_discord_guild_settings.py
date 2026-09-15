"""Tests for the Discord guild-settings request contract."""

import pytest

from tools.discord_api.guild_settings import GuildSettingsError, edit_guild_request

GUILD_ID = "123456789012345678"
CHANNEL_ID = "987654321098765432"


def test_allowed_scalar_edits_full_payload():
    request = edit_guild_request(
        GUILD_ID,
        name="Hermes HQ",
        description="A fine place",
        verification_level=4,
        default_message_notifications=1,
        explicit_content_filter=2,
        premium_progress_bar_enabled=True,
        afk_channel_id=CHANNEL_ID,
        system_channel_id=CHANNEL_ID,
        rules_channel_id=CHANNEL_ID,
        public_updates_channel_id=CHANNEL_ID,
        safety_alerts_channel_id=CHANNEL_ID,
        afk_timeout=3600,
    )

    assert request == {
        "method": "PATCH",
        "path": f"/guilds/{GUILD_ID}",
        "json": {
            "name": "Hermes HQ",
            "description": "A fine place",
            "verification_level": 4,
            "default_message_notifications": 1,
            "explicit_content_filter": 2,
            "premium_progress_bar_enabled": True,
            "afk_channel_id": CHANNEL_ID,
            "system_channel_id": CHANNEL_ID,
            "rules_channel_id": CHANNEL_ID,
            "public_updates_channel_id": CHANNEL_ID,
            "safety_alerts_channel_id": CHANNEL_ID,
            "afk_timeout": 3600,
        },
    }


def test_nullable_fields_and_falsey_values_are_preserved():
    request = edit_guild_request(
        GUILD_ID,
        verification_level=None,
        default_message_notifications=0,
        explicit_content_filter=None,
        premium_progress_bar_enabled=False,
        description=None,
        afk_channel_id=None,
    )

    assert request["json"] == {
        "verification_level": None,
        "default_message_notifications": 0,
        "explicit_content_filter": None,
        "premium_progress_bar_enabled": False,
        "description": None,
        "afk_channel_id": None,
    }


@pytest.mark.parametrize(
    "bad_key",
    ["nsfw_level", "widget_enabled", "system_channel_flags", "bogus_field"],
)
def test_disallowed_key_rejected(bad_key):
    with pytest.raises(GuildSettingsError, match="unsupported guild setting"):
        edit_guild_request(GUILD_ID, **{bad_key: True})


def test_name_contract():
    assert edit_guild_request(GUILD_ID, name="xx")["json"]["name"] == "xx"
    assert edit_guild_request(GUILD_ID, name="x" * 100)["json"]["name"] == "x" * 100

    for bad in ("x", " x", "x ", "x" * 101, 123):
        with pytest.raises(GuildSettingsError):
            edit_guild_request(GUILD_ID, name=bad)


def test_description_max_contract():
    description = edit_guild_request(GUILD_ID, description="x" * 1024)["json"][
        "description"
    ]
    assert len(description) == 1024
    with pytest.raises(GuildSettingsError, match="exceeds 1024"):
        edit_guild_request(GUILD_ID, description="x" * 1025)


@pytest.mark.parametrize("level", [-1, 5, 100])
def test_verification_level_out_of_range(level):
    with pytest.raises(GuildSettingsError, match="must be one of"):
        edit_guild_request(GUILD_ID, verification_level=level)


@pytest.mark.parametrize("bad", [True, "3", 3.5])
def test_verification_level_wrong_type(bad):
    with pytest.raises(GuildSettingsError, match="must be an integer"):
        edit_guild_request(GUILD_ID, verification_level=bad)


@pytest.mark.parametrize(
    "field",
    [
        "afk_channel_id",
        "system_channel_id",
        "rules_channel_id",
        "public_updates_channel_id",
        "safety_alerts_channel_id",
    ],
)
@pytest.mark.parametrize(
    "bad",
    ["not-a-snowflake", "123abc", -5, 0, 2**64, 1.5, True],
)
def test_channel_id_invalid_rejected(field, bad):
    with pytest.raises(GuildSettingsError):
        edit_guild_request(GUILD_ID, **{field: bad})


def test_snowflakes_are_canonical_decimal_strings():
    request = edit_guild_request(
        "000123456789012345678",
        rules_channel_id=987654321098765432,
        system_channel_id="000987654321098765432",
    )

    assert request["path"] == "/guilds/123456789012345678"
    assert request["json"]["rules_channel_id"] == "987654321098765432"
    assert request["json"]["system_channel_id"] == "987654321098765432"


@pytest.mark.parametrize("guild_id", ["guild-abc", "", "0000", 0, -1, 2**64, True])
def test_invalid_guild_id_rejected(guild_id):
    with pytest.raises(GuildSettingsError):
        edit_guild_request(guild_id, name="Hermes")


@pytest.mark.parametrize("timeout", [60, 300, 900, 1800, 3600])
def test_afk_timeout_discrete_values_allowed(timeout):
    assert edit_guild_request(GUILD_ID, afk_timeout=timeout)["json"] == {
        "afk_timeout": timeout
    }


@pytest.mark.parametrize("timeout", [59, 61, 299, 301, 3601, 0, -1])
def test_afk_timeout_non_enum_values_rejected(timeout):
    with pytest.raises(GuildSettingsError, match="60, 300, 900, 1800, 3600"):
        edit_guild_request(GUILD_ID, afk_timeout=timeout)


def test_only_provided_fields_in_payload():
    assert edit_guild_request(GUILD_ID, name="Renamed")["json"] == {
        "name": "Renamed"
    }


def test_empty_patch_rejected():
    with pytest.raises(GuildSettingsError, match="no guild settings provided"):
        edit_guild_request(GUILD_ID)
