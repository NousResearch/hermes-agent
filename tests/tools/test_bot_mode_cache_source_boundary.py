"""Same-session cross-source cache-boundary regressions (review round 3).

A trusted CLI canonical "Bot Chat" classification must never be reused by an
API-server or A2A agent for the same persisted session id, and a denied
source must not poison a later trusted Bot Chat classification for that
session. Covers the state helper, prompt injection, and dispatch.
"""

import json

import pytest

from gateway.session_context import clear_session_vars, set_session_vars
from tools import bot_mode_dm, bot_mode_probe


def _persisted_session(tmp_path, *, sid="persisted-1"):
    """Managed install with one yuki profile and a persisted session id."""
    home = bot_mode_probe and _make_home(tmp_path)
    yuki_home = home / "profiles" / "yuki"

    class _DB:
        def __init__(self, home):
            self.db_path = str(home / "state.db")

        def get_session_title(self, _sid):
            return "Bot Chat"

    class _Agent:
        def __init__(self, platform, title):
            self._session_db = _DB(yuki_home)
            self.session_id = sid
            self._session_title_hint = None
            self._bot_mode_protocol = True
            self.platform = platform
            self.tools = []
            self.valid_tool_names = set()

    return yuki_home, _Agent


def _make_home(tmp_path):
    from tests.tools.test_bot_mode_gateway_sessions import _make_hermes_home

    return _make_hermes_home(tmp_path, managed_profiles=("yuki",))


def _reset():
    bot_mode_probe._reset_cache_for_tests()


@pytest.fixture(autouse=True)
def _fresh():
    _reset()
    yield
    _reset()


def test_cli_bot_chat_then_api_server_same_session_denied(tmp_path, monkeypatch):
    """Trusted CLI Bot Chat cache entry must not leak to an API-server agent."""
    yuki_home, _Agent = _persisted_session(tmp_path)

    cli_agent = _Agent("cli", "Bot Chat")
    assert bot_mode_probe.bot_mode_session_state(cli_agent) == {
        "managed": True,
        "session_kind": "bot_chat",
    }
    assert bot_mode_dm.ensure_message_agent_tool(cli_agent) is True

    # Same persisted session, same home, different (API-like) source.
    api_agent = _Agent("api_server", "Bot Chat")
    assert bot_mode_probe.bot_mode_session_state(api_agent)["session_kind"] is None
    assert bot_mode_dm.ensure_message_agent_tool(api_agent) is False
    assert api_agent.tools == []

    dispatched = False

    def fake_spawn(*args, **kwargs):
        nonlocal dispatched
        dispatched = True
        return json.dumps({"status": "sent"})

    monkeypatch.setattr(bot_mode_dm, "_spawn_delivery", fake_spawn)
    result = json.loads(
        bot_mode_dm.message_agent_tool(target="coder", message="blocked", agent=api_agent)
    )
    assert "error" in result
    assert dispatched is False

    a2a_agent = _Agent("a2a", "Bot Chat")
    assert bot_mode_probe.bot_mode_session_state(a2a_agent)["session_kind"] is None
    assert bot_mode_dm.ensure_message_agent_tool(a2a_agent) is False


def test_denied_source_does_not_poison_later_trusted_bot_chat(tmp_path, monkeypatch):
    """A denied-source first classification must not suppress trusted routing."""
    yuki_home, _Agent = _persisted_session(tmp_path)

    api_agent = _Agent("api_server", "Bot Chat")
    assert bot_mode_probe.bot_mode_session_state(api_agent)["session_kind"] is None
    assert bot_mode_dm.ensure_message_agent_tool(api_agent) is False

    cli_agent = _Agent("cli", "Bot Chat")
    assert bot_mode_probe.bot_mode_session_state(cli_agent) == {
        "managed": True,
        "session_kind": "bot_chat",
    }
    assert bot_mode_dm.ensure_message_agent_tool(cli_agent) is True

    captured = {}

    def fake_spawn(command, label, *, dm_file=None, task_id=None, agent=None):
        captured["label"] = label
        return json.dumps({"status": "sent", "to": label})

    monkeypatch.setattr(bot_mode_dm, "_spawn_delivery", fake_spawn)
    result = json.loads(
        bot_mode_dm.message_agent_tool(target="coder", message="route", agent=cli_agent)
    )
    assert result["status"] == "sent"
    assert captured["label"] == "@coder"


def test_gateway_state_also_bound_to_source(tmp_path):
    """Gateway classification is source-scoped too, both directions."""
    yuki_home, _Agent = _persisted_session(tmp_path)

    discord_agent = _Agent("discord", "Group: 1")
    assert bot_mode_probe.bot_mode_session_state(discord_agent) == {
        "managed": True,
        "session_kind": "gateway",
    }
    api_agent = _Agent("api_server", "Group: 1")
    assert bot_mode_probe.bot_mode_session_state(api_agent)["session_kind"] is None
    assert bot_mode_dm.ensure_message_agent_tool(api_agent) is False

    # Reverse order: API first must not suppress the trusted gateway route.
    reverse_root = tmp_path / "reverse"
    reverse_root.mkdir()
    home2 = _make_home(reverse_root)
    yuki2 = home2 / "profiles" / "yuki"
    class _DB2:
        def __init__(self, home):
            self.db_path = str(home / "state.db")

        def get_session_title(self, _sid):
            return "Group: 1"

    class _Agent2:
        def __init__(self, platform):
            self._session_db = _DB2(yuki2)
            self.session_id = "persisted-2"
            self._session_title_hint = None
            self._bot_mode_protocol = True
            self.platform = platform
            self.tools = []
            self.valid_tool_names = set()

    denied = _Agent2("api_server")
    assert bot_mode_probe.bot_mode_session_state(denied)["session_kind"] is None
    trusted = _Agent2("discord")
    assert bot_mode_probe.bot_mode_session_state(trusted)["session_kind"] == "gateway"
    assert bot_mode_dm.ensure_message_agent_tool(trusted) is True


def test_authoritative_task_source_overrides_cli_platform(tmp_path, monkeypatch):
    """A machine task source cannot borrow canonical Bot Chat CLI trust."""
    _yuki_home, _Agent = _persisted_session(tmp_path)
    agent = _Agent("cli", "Bot Chat")
    dispatched = False

    def fake_spawn(*args, **kwargs):
        nonlocal dispatched
        dispatched = True
        return json.dumps({"status": "sent"})

    monkeypatch.setattr(bot_mode_dm, "_spawn_delivery", fake_spawn)
    tokens = set_session_vars(source="tool")
    try:
        assert bot_mode_probe.bot_mode_session_state(agent)["session_kind"] is None
        assert bot_mode_dm.ensure_message_agent_tool(agent) is False
        result = json.loads(
            bot_mode_dm.message_agent_tool(target="coder", message="blocked", agent=agent)
        )
    finally:
        clear_session_vars(tokens)

    assert "error" in result
    assert dispatched is False


def test_same_agent_source_change_denies_dispatch_without_schema_churn(
    tmp_path, monkeypatch
):
    """Source trust may tighten, but a live tool list stays byte-stable."""
    _yuki_home, _Agent = _persisted_session(tmp_path)
    agent = _Agent("cli", "Bot Chat")
    assert bot_mode_dm.ensure_message_agent_tool(agent) is True
    schema = json.dumps(agent.tools, sort_keys=True)
    dispatched = False

    def fake_spawn(*args, **kwargs):
        nonlocal dispatched
        dispatched = True
        return json.dumps({"status": "sent"})

    monkeypatch.setattr(bot_mode_dm, "_spawn_delivery", fake_spawn)
    tokens = set_session_vars(source="tool")
    try:
        assert bot_mode_dm.ensure_message_agent_tool(agent) is False
        assert json.dumps(agent.tools, sort_keys=True) == schema
        result = json.loads(
            bot_mode_dm.message_agent_tool(target="coder", message="blocked", agent=agent)
        )
    finally:
        clear_session_vars(tokens)

    assert "error" in result
    assert dispatched is False


def test_dispatch_rechecks_live_bot_mode_authorization(tmp_path, monkeypatch):
    """A cached schema grant cannot deliver after Bot Mode is revoked."""
    yuki_home, _Agent = _persisted_session(tmp_path)
    agent = _Agent("cli", "Bot Chat")
    assert bot_mode_dm.ensure_message_agent_tool(agent) is True
    (yuki_home / "profile.yaml").unlink()
    dispatched = False

    def fake_spawn(*args, **kwargs):
        nonlocal dispatched
        dispatched = True
        return json.dumps({"status": "sent"})

    monkeypatch.setattr(bot_mode_dm, "_spawn_delivery", fake_spawn)
    result = json.loads(
        bot_mode_dm.message_agent_tool(target="coder", message="blocked", agent=agent)
    )
    assert "error" in result
    assert dispatched is False


def test_process_session_state_cache_evicts_oldest_entry(tmp_path, monkeypatch):
    """Traffic-controlled session IDs cannot grow the process cache unbounded."""
    _yuki_home, _Agent = _persisted_session(tmp_path)
    monkeypatch.setattr(bot_mode_probe, "_SESSION_STATE_CACHE_MAX", 3)
    for index in range(4):
        agent = _Agent("discord", "Group: 1")
        agent.session_id = f"traffic-{index}"
        assert bot_mode_probe.bot_mode_session_state(agent)["session_kind"] == "gateway"
    assert len(bot_mode_probe._session_state_cache) == 3
    keys = list(bot_mode_probe._session_state_cache)
    assert all(key[1] != "traffic-0" for key in keys)
    assert any(key[1] == "traffic-3" for key in keys)
