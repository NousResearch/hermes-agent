"""Wire-contract regressions for hosted Bot Mode room results."""

import pytest
from pydantic import ValidationError

from tui_gateway.contracts.groups_bot_relay import RoomActor


def test_room_actor_accepts_member_fields_and_rejects_unknown_fields():
    """Regression for #138908: replayed member actors carry their routing identity."""
    actor = RoomActor(
        kind="member",
        id="alice",
        profile="profile-a",
        display_name="Alice",
        connection_id="peer-a",
    )

    assert actor.model_dump() == {
        "kind": "member",
        "id": "alice",
        "profile": "profile-a",
        "display_name": "Alice",
        "connection_id": "peer-a",
    }
    with pytest.raises(ValidationError, match="extra_forbidden"):
        RoomActor(kind="member", id="alice", unexpected="value")
