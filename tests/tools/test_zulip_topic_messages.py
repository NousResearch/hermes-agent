"""Tests for the agent-callable Zulip topic send tool."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig


class _FakeZulipClient:
    """Minimal Zulip SDK fake shared by lookup and standalone sending."""

    requests: list[dict] = []
    credentials: list[tuple[str, str, str]] = []

    def __init__(self, *, site: str, email: str, api_key: str, **_kwargs):
        self.credentials.append((site, email, api_key))

    def get_stream_id(self, stream: str) -> dict:
        if stream == "projects":
            return {"result": "success", "stream_id": 29361}
        return {"result": "error", "msg": "Stream does not exist"}

    def send_message(self, request: dict) -> dict:
        self.requests.append(request)
        return {"result": "success", "id": 4242}


@pytest.fixture(autouse=True)
def _isolated_zulip_tool_env(monkeypatch, tmp_path):
    """Keep persistent session routing and optional Zulip SDK isolated."""
    import hermes_state

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    monkeypatch.setitem(sys.modules, "zulip", SimpleNamespace(Client=_FakeZulipClient))
    _FakeZulipClient.requests = []
    _FakeZulipClient.credentials = []
    for name in ("ZULIP_SITE_URL", "ZULIP_BOT_EMAIL", "ZULIP_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    config = GatewayConfig(
        sessions_dir=tmp_path / "sessions",
        platforms={
            Platform("zulip"): PlatformConfig(
                enabled=True,
                token="zulip-api-key",
                extra={
                    "site_url": "https://chat.example.test",
                    "bot_email": "bot@example.test",
                },
            )
        },
    )
    with patch("gateway.config.load_gateway_config", return_value=config):
        yield config


def test_send_topic_message_creates_session_and_seeds_transcript(_isolated_zulip_tool_env):
    from gateway.session import SessionSource, SessionStore
    from plugins.platforms.zulip.topic_tool import zulip_send_topic_message

    result = json.loads(
        zulip_send_topic_message(
            stream="projects",
            topic="new work lane",
            message="Seed message for the new work lane.",
            session_user_email="user@example.test",
        )
    )

    assert result["success"] is True
    assert result["chat_id"] == "29361:new work lane"
    assert result["message_id"] == "4242"
    assert result["session_seeded"] is True
    assert _FakeZulipClient.requests == [{
        "type": "stream",
        "to": "29361",
        "topic": "new work lane",
        "content": "Seed message for the new work lane.",
    }]

    store = SessionStore(
        sessions_dir=_isolated_zulip_tool_env.sessions_dir,
        config=_isolated_zulip_tool_env,
    )
    source = SessionSource(
        platform=Platform("zulip"),
        chat_id="29361:new work lane",
        chat_name="projects",
        chat_type="stream",
        user_id="user@example.test",
        user_name="user@example.test",
        chat_topic="new work lane",
    )
    # A newly seeded topic is intentionally discovered through SessionStore's
    # DB recovery path on the first real inbound reply; no temporary store
    # rewrites the live gateway's whole routing index.
    session_id = store.get_or_create_session(source).session_id
    assert session_id == result["session_id"]
    messages = store.load_transcript(session_id)
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] == "[Hermes to Zulip]\nSeed message for the new work lane."
    assert messages[-1]["message_id"] == "4242"


def test_zulip_origin_infers_user_and_preserves_profile(_isolated_zulip_tool_env):
    from gateway.session import SessionSource, SessionStore
    from gateway.session_context import clear_session_vars, set_session_vars
    from plugins.platforms.zulip.topic_tool import zulip_send_topic_message

    _isolated_zulip_tool_env.multiplex_profiles = True
    tokens = set_session_vars(
        platform="zulip",
        user_id="user@example.test",
        user_name="User",
        profile="research",
    )
    try:
        result = json.loads(
            zulip_send_topic_message(
                stream="projects",
                topic="profile-aware lane",
                message="Continue this investigation separately.",
            )
        )
    finally:
        clear_session_vars(tokens)

    assert result["success"] is True
    assert result["session_user_email"] == "user@example.test"
    store = SessionStore(
        sessions_dir=_isolated_zulip_tool_env.sessions_dir,
        config=_isolated_zulip_tool_env,
    )
    source = SessionSource(
        platform=Platform("zulip"),
        chat_id="29361:profile-aware lane",
        chat_name="projects",
        chat_type="stream",
        user_id="user@example.test",
        user_name="User",
        chat_topic="profile-aware lane",
        profile="research",
    )
    assert store.get_or_create_session(source).session_id == result["session_id"]


def test_non_zulip_origin_requires_session_owner_before_sending(_isolated_zulip_tool_env):
    from plugins.platforms.zulip.topic_tool import zulip_send_topic_message

    result = json.loads(
        zulip_send_topic_message(
            stream="projects", topic="new work lane", message="Do not send this."
        )
    )

    assert "session_user_email" in result["error"]
    assert _FakeZulipClient.requests == []


def test_session_seed_failure_does_not_misreport_successful_send(
    _isolated_zulip_tool_env,
):
    from plugins.platforms.zulip.topic_tool import zulip_send_topic_message

    with patch(
        "plugins.platforms.zulip.topic_tool._seed_topic_session",
        side_effect=RuntimeError("state database unavailable"),
    ):
        result = json.loads(
            zulip_send_topic_message(
                stream="projects",
                topic="seed outage",
                message="The Zulip send still succeeds.",
                session_user_email="user@example.test",
            )
        )

    assert result["success"] is True
    assert result["session_seeded"] is False
    assert "state database unavailable" in result["session_error"]
    assert _FakeZulipClient.requests[0]["topic"] == "seed outage"


def test_topic_seed_does_not_replace_live_gateway_routing_index(_isolated_zulip_tool_env):
    """Seeding a topic must not overwrite unrelated live gateway routing."""
    from gateway.session import SessionSource, SessionStore
    from plugins.platforms.zulip.topic_tool import zulip_send_topic_message

    live_store = SessionStore(
        sessions_dir=_isolated_zulip_tool_env.sessions_dir,
        config=_isolated_zulip_tool_env,
    )
    existing_source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_type="dm",
        user_id="user-123",
    )
    existing = live_store.get_or_create_session(existing_source)
    existing_key = live_store._generate_session_key(existing_source)

    result = json.loads(
        zulip_send_topic_message(
            stream="projects",
            topic="new work lane",
            message="Keep the other routing entries intact.",
            session_user_email="user@example.test",
        )
    )

    assert result["session_seeded"] is True
    fresh_store = SessionStore(
        sessions_dir=_isolated_zulip_tool_env.sessions_dir,
        config=_isolated_zulip_tool_env,
    )
    assert fresh_store.peek_session_id(existing_key) == existing.session_id


def test_topic_tool_uses_one_resolved_credential_set_for_lookup_and_send(
    _isolated_zulip_tool_env, monkeypatch
):
    from plugins.platforms.zulip.topic_tool import zulip_send_topic_message

    monkeypatch.setenv("ZULIP_SITE_URL", "https://env-chat.example.test/")
    monkeypatch.setenv("ZULIP_BOT_EMAIL", "env-bot@example.test")
    monkeypatch.setenv("ZULIP_API_KEY", "env-zulip-api-key")

    result = json.loads(
        zulip_send_topic_message(
            stream="projects",
            topic="consistent credentials",
            message="Lookup and send must use one Zulip account.",
            session_user_email="user@example.test",
        )
    )

    assert result["success"] is True
    assert _FakeZulipClient.credentials == [
        ("https://env-chat.example.test", "env-bot@example.test", "env-zulip-api-key"),
        ("https://env-chat.example.test", "env-bot@example.test", "env-zulip-api-key"),
    ]
