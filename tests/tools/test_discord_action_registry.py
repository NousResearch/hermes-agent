"""Invariant tests for the canonical Discord action-extension seam."""

import pytest

from tools.discord_api import action_registry
from tools.discord_api.action_registry import DiscordAction, register_discord_action


def _handler(**_kwargs):
    return "{}"


def test_action_rejects_missing_required_schema_property():
    with pytest.raises(ValueError, match="required properties missing"):
        DiscordAction(
            name="example_action",
            surface="admin",
            signature="(payload)",
            description="example",
            handler=_handler,
            required=("payload",),
        )


def test_action_rejects_canonical_action_field_override():
    with pytest.raises(ValueError, match="canonical 'action' field"):
        DiscordAction(
            name="example_action",
            surface="admin",
            signature="()",
            description="example",
            handler=_handler,
            properties={"action": {"type": "string"}},
        )


def test_action_rejects_invalid_property_schema():
    with pytest.raises(TypeError, match="must be a mapping"):
        DiscordAction(
            name="example_action",
            surface="admin",
            signature="(payload)",
            description="example",
            handler=_handler,
            properties={"payload": "not-a-schema"},
        )


def test_registration_rejects_duplicate_action_ownership(monkeypatch):
    monkeypatch.setattr(action_registry, "_ACTIONS", {})
    first = DiscordAction(
        name="example_action",
        surface="admin",
        signature="()",
        description="first owner",
        handler=_handler,
    )
    second = DiscordAction(
        name="example_action",
        surface="admin",
        signature="()",
        description="second owner",
        handler=_handler,
    )

    register_discord_action(first)
    with pytest.raises(ValueError, match="already registered"):
        register_discord_action(second)
