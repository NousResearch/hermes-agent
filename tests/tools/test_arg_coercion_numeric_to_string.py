"""Numeric args emitted where the schema declares ``string`` must be coerced, not rejected.

Reproduces a Discord read failure: ``discord`` declares every ID param
(``channel_id``/``guild_id``/``user_id``/``message_id``/``role_id``) as ``{"type": "string"}``,
but a model reading a snowflake ID out of the conversation context emits it as a JSON int
because the ID is all digits. ``tools/tool_search_validation.py`` coerces then validates with
jsonschema, so an unrepaired int fails validation and the tool is NEVER dispatched — the
agent silently loses channel reads.

The defect is the early return in ``coerce_tool_args`` for non-string values, which skipped
every numeric value regardless of what the schema asked for.
"""

import pytest

from tools.arg_coercion import coerce_tool_args
from tools.registry import registry


_DISCORD_SCHEMA = {
    "name": "discord",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["fetch_messages", "list_channels"]},
            "channel_id": {"type": "string", "description": "Discord channel ID."},
            "guild_id": {"type": "string", "description": "Discord server (guild) ID."},
            "user_id": {"type": "string", "description": "Discord user ID."},
            "limit": {"type": "integer", "minimum": 1, "maximum": 100},
            "query": {"type": "string", "description": "Member name prefix."},
        },
        "required": ["action"],
    },
}


@pytest.fixture
def discord_schema(monkeypatch):
    """Register a Discord-shaped schema so coercion is schema-guided, not a no-op."""
    schema = dict(_DISCORD_SCHEMA)
    monkeypatch.setattr(registry, "get_schema", lambda name: schema if name == "discord" else None)
    return schema


@pytest.mark.parametrize("channel_id", [1234567890123456789, 1111111111111111111])
def test_int_snowflake_is_stringified_for_string_schema(discord_schema, channel_id):
    """THE regression: an int channel_id must come out as its exact decimal string."""
    out = coerce_tool_args("discord", {"action": "fetch_messages", "channel_id": channel_id})
    assert out["channel_id"] == str(channel_id)
    assert isinstance(out["channel_id"], str)
    # No precision loss — these snowflakes are past 2**53, so a float round-trip would corrupt them.
    assert out["channel_id"] in {"1234567890123456789", "1111111111111111111"}
    assert int(out["channel_id"]) == channel_id


def test_int_guild_id_is_stringified(discord_schema):
    out = coerce_tool_args("discord", {"action": "list_channels", "guild_id": 2222222222222222222})
    assert out["guild_id"] == "2222222222222222222"


def test_string_id_is_left_untouched(discord_schema):
    """The already-correct form must not be double-processed."""
    out = coerce_tool_args("discord", {"action": "fetch_messages", "channel_id": "1234567890123456789"})
    assert out["channel_id"] == "1234567890123456789"


def test_integer_schema_still_coerces_string_to_int(discord_schema):
    """The pre-existing string->int direction must keep working (no regression)."""
    out = coerce_tool_args("discord", {"action": "fetch_messages", "channel_id": "1", "limit": "5"})
    assert out["limit"] == 5
    assert isinstance(out["limit"], int)


def test_int_for_integer_schema_is_unchanged(discord_schema):
    out = coerce_tool_args("discord", {"action": "fetch_messages", "channel_id": "1", "limit": 5})
    assert out["limit"] == 5
    assert isinstance(out["limit"], int)


def test_bool_is_not_stringified(discord_schema):
    """A real bool in a string field is a different mistake; coercing would hide it."""
    out = coerce_tool_args("discord", {"action": "fetch_messages", "channel_id": True})
    assert out["channel_id"] is True, "bool must stay a bool so the schema error still surfaces"


def test_float_is_stringified_without_precision_loss(discord_schema):
    out = coerce_tool_args("discord", {"action": "fetch_messages", "channel_id": 1.5})
    assert out["channel_id"] == "1.5"
    assert isinstance(out["channel_id"], str)


def test_validates_end_to_end_through_jsonschema(discord_schema):
    """The real failure path: coerce, then run the same jsonschema pass the validator uses."""
    pytest.importorskip("jsonschema")
    from jsonschema.exceptions import best_match
    from jsonschema.validators import validator_for

    schema = discord_schema["parameters"]
    validator_cls = validator_for(schema)
    validator_cls.check_schema(schema)

    coerced = coerce_tool_args("discord", {"action": "fetch_messages", "channel_id": 1234567890123456789})
    errors = list(validator_cls(schema).iter_errors(coerced))
    assert best_match(errors) is None, f"int channel_id still fails validation: {coerced}"