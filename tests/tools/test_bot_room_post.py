"""Tests for tools/bot_room_post.py — the Bot-Chat-only ``room_post`` tool.

Two contracts matter here. Containment, like ``message_agent``: the schema exists
ONLY in a canonical Bot Chat session on a Bot-Mode-managed install. And the
request shape: a post the Desktop could never deliver (no room, no text, over the
cap) is refused HERE, with a reason the model can act on, instead of being
harvested into a room as a broken line.
"""

import json
import textwrap
from pathlib import Path

import pytest

from tools import bot_mode_probe, bot_room_post


@pytest.fixture(autouse=True)
def _fresh_probe_cache():
    bot_mode_probe._reset_cache_for_tests()
    yield
    bot_mode_probe._reset_cache_for_tests()


def _managed_home(tmp_path) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir(exist_ok=True)
    d = home / "profiles" / "researcher"
    d.mkdir(parents=True, exist_ok=True)
    (d / "profile.yaml").write_text(
        textwrap.dedent(
            """\
            description: teammate for tests
            ui_meta:
              hermes-bots:
                shape: cloud
            """
        ),
        encoding="utf-8",
    )
    return home


class _FakeDB:
    def __init__(self, home: Path, title: str):
        self.db_path = str(home / "state.db")
        self._title = title

    def get_session_title(self, _sid):
        return self._title


class _FakeAgent:
    def __init__(self, home: Path, title: str = "Bot Chat"):
        self._session_db = _FakeDB(home, title)
        self.session_id = "sess-1"
        self._session_title_hint = None
        self._bot_mode_protocol = True
        self.tools: list = []
        self.valid_tool_names: set = set()


# ── the request shape ────────────────────────────────────────────────────────


def test_schema_advertises_room_text_and_optional_mentions():
    schema = bot_room_post.room_post_tool_schema()["function"]

    assert schema["name"] == bot_room_post.ROOM_POST_TOOL_NAME
    assert sorted(schema["parameters"]["required"]) == ["room", "text"]
    assert schema["parameters"]["properties"]["mentions"]["type"] == "array"
    # A room is a conversation, not a status page: the description has to say so,
    # or the tool becomes a second way to dump reports into a room.
    assert "conversation" in schema["description"]
    assert "ASYNCHRONOUS" in schema["description"]


def test_accepts_a_well_formed_post():
    ack = json.loads(bot_room_post.room_post_tool("Spawn", "  shipped the payout client  ", ["user", "hotel-dev"]))

    assert ack["success"] is True
    assert ack["queued"] is True
    assert ack["room"] == "Spawn"
    assert ack["mentions"] == ["user", "hotel-dev"]
    assert "Desktop" in ack["note"]


def test_refuses_a_post_with_no_room_or_no_text():
    assert json.loads(bot_room_post.room_post_tool("", "hello"))["success"] is False
    assert json.loads(bot_room_post.room_post_tool("Spawn", "   "))["success"] is False


def test_refuses_an_over_long_post():
    ack = json.loads(bot_room_post.room_post_tool("Spawn", "x" * (bot_room_post.ROOM_POST_MAX_CHARS + 1)))

    assert ack["success"] is False
    assert str(bot_room_post.ROOM_POST_MAX_CHARS) in ack["error"]


def test_refuses_more_mentions_than_a_room_can_hold():
    ack = json.loads(
        bot_room_post.room_post_tool(
            "Spawn", "hi", [f"member-{index}" for index in range(bot_room_post.ROOM_POST_MAX_MENTIONS + 1)]
        )
    )

    assert ack["success"] is False
    assert "mentions" in ack["error"]


def test_ignores_blank_mentions_and_extra_arguments():
    ack = json.loads(bot_room_post.room_post_tool("Spawn", "hi", ["", "  ", "ops"], unexpected="x"))

    assert ack["success"] is True
    assert ack["mentions"] == ["ops"]


# ── injection gate (leak containment) ────────────────────────────────────────


def test_injects_only_into_bot_chat_on_managed_install(tmp_path):
    agent = _FakeAgent(_managed_home(tmp_path), title="Bot Chat")

    assert bot_room_post.ensure_room_post_tool(agent) is True
    assert [tool["function"]["name"] for tool in agent.tools] == [bot_room_post.ROOM_POST_TOOL_NAME]
    # Success means both halves: an advertised-but-nondispatchable tool is a bug
    # (#96105), so the executor allowlist has to carry it too.
    assert bot_room_post.ROOM_POST_TOOL_NAME in agent.valid_tool_names

    # Idempotent: the tool list stays byte-identical across turns.
    assert bot_room_post.ensure_room_post_tool(agent) is True
    assert len(agent.tools) == 1


def test_restores_the_allowlist_when_the_schema_survives_a_surface_rebuild(tmp_path):
    agent = _FakeAgent(_managed_home(tmp_path), title="Bot Chat")
    bot_room_post.ensure_room_post_tool(agent)
    agent.valid_tool_names = set()

    assert bot_room_post.ensure_room_post_tool(agent) is True
    assert bot_room_post.ROOM_POST_TOOL_NAME in agent.valid_tool_names


def test_injects_into_the_rooms_own_member_session(tmp_path):
    """The room drives a member in a hidden `Group: …` session and mirrors THAT
    transcript. Injecting only into the canonical Bot Chat would authorize a post
    the room never reads — the producer and the consumer have to share a session."""
    agent = _FakeAgent(_managed_home(tmp_path), title="Group: Room · thread-1")

    assert bot_room_post.ensure_room_post_tool(agent) is True
    assert [tool["function"]["name"] for tool in agent.tools] == [bot_room_post.ROOM_POST_TOOL_NAME]


def test_does_not_inject_outside_a_bot_chat_or_a_room(tmp_path):
    agent = _FakeAgent(_managed_home(tmp_path), title="Refactor the payout client")

    assert bot_room_post.ensure_room_post_tool(agent) is False
    assert agent.tools == []


def test_does_not_inject_a_group_lookalike_title_on_an_unmanaged_install(tmp_path):
    home = tmp_path / "not-hermes"
    home.mkdir()
    agent = _FakeAgent(home, title="Group: Room")

    assert bot_room_post.ensure_room_post_tool(agent) is False
    assert agent.tools == []


def test_does_not_inject_when_the_protocol_is_off(tmp_path):
    agent = _FakeAgent(_managed_home(tmp_path), title="Bot Chat")
    agent._bot_mode_protocol = False

    assert bot_room_post.ensure_room_post_tool(agent) is False
    assert agent.tools == []
