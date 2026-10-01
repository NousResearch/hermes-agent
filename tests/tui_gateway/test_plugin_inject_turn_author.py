"""A plugin-injected turn belongs to the plugin, not to the human.

``PluginContext.inject_message`` queues the plugin's own words — usually an instruction TO the
agent. Both injectors used to drop the plugin identity, so memory providers that derive facts
from user messages stored the plugin's text as durable facts about the user.
"""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import hermes_yaml as yaml

from agent.turn_author import plugin_author_for_event, plugin_turn_author
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from tui_gateway import server


def _write_plugin_config(tmp_path, monkeypatch, entry: dict) -> None:
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"entries": {"notify-plugin": entry}}})
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))


def _context() -> tuple[PluginContext, PluginManager]:
    manager = PluginManager()
    manifest = PluginManifest(name="notify-plugin", key="notify-plugin", source="user")
    return PluginContext(manifest, manager), manager


def _session(session_key: str, **extra) -> dict:
    return {
        "agent": SimpleNamespace(),
        "session_key": session_key,
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": True,  # a busy session only queues, so the turn is inspectable in place
        "transport": None,
        "attached_images": [],
        "last_active": 1.0,
        **extra,
    }


def test_plugin_author_is_bot_authored_and_never_empty():
    assert plugin_turn_author("room-wake") == {
        "id": "plugin:room-wake", "name": "room-wake", "is_bot": True,
    }
    # No plugin named: the turn stays unattributed, exactly as before the fix.
    assert plugin_turn_author("") is None
    assert plugin_turn_author("   ") is None


def test_plugin_author_id_does_not_collide_with_a_bot_dm():
    """A plugin and a relayed bot DM are different senders and must not share an a2a session."""
    assert plugin_turn_author("room-wake")["id"] == "plugin:room-wake"
    assert not plugin_turn_author("room-wake")["id"].startswith("bot:")


def test_tui_injected_turn_carries_the_plugin_as_its_author(tmp_path, monkeypatch):
    _write_plugin_config(tmp_path, monkeypatch, {"allow_gateway_injection": True})
    context, manager = _context()
    live = _session("ses_live")
    monkeypatch.setattr(server, "_sessions", {"ui-live": live})
    server.install_tui_message_injector(manager)
    try:
        assert context.inject_message(
            "Read the room and call tool_x after you read it.", session_key="ses_live",
        ) is True
    finally:
        server.clear_tui_message_injector(manager)

    queued = live["queued_prompt"]
    assert queued["text"] == "Read the room and call tool_x after you read it."
    # The regression: without this the envelope has no author, and a memory provider reads
    # the plugin's instruction as something the user said.
    assert queued["turn_author"] == {
        "id": "plugin:notify-plugin", "name": "notify-plugin", "is_bot": True,
    }


def test_the_injector_call_signature_is_unchanged_without_an_explicit_author(tmp_path, monkeypatch):
    """A host registered against the older injector signature must keep working.

    ``author`` is only passed when a plugin actually names one, so the kwargs an injector
    receives are byte-for-byte what they were before this fix.
    """
    _write_plugin_config(tmp_path, monkeypatch, {"allow_gateway_injection": True})
    context, manager = _context()
    gateway = MagicMock(return_value=True)
    manager.set_gateway_message_injector(object(), gateway)
    monkeypatch.setattr(server, "_sessions", {})

    assert context.inject_message("wake telegram", session_key="agent:main:telegram:dm:42") is True

    gateway.assert_called_once_with(
        session_key="agent:main:telegram:dm:42",
        content="wake telegram",
        plugin_id="notify-plugin",
    )


def test_explicit_human_author_wins_over_the_plugin_default(tmp_path, monkeypatch):
    """A chat bridge relaying a real person keeps that person's attribution."""
    _write_plugin_config(tmp_path, monkeypatch, {"allow_gateway_injection": True})
    context, manager = _context()
    live = _session("ses_live")
    monkeypatch.setattr(server, "_sessions", {"ui-live": live})
    server.install_tui_message_injector(manager)
    try:
        assert context.inject_message(
            "ship it", session_key="ses_live",
            author={"id": "42", "name": "tester", "is_bot": False},
        ) is True
    finally:
        server.clear_tui_message_injector(manager)

    assert live["queued_prompt"]["turn_author"] == {"id": "42", "name": "tester", "is_bot": False}


def test_authored_plugin_turn_does_not_merge_into_a_queued_human_prompt(tmp_path, monkeypatch):
    """Two plugins injecting in a row stay two turns, each with its own author."""
    _write_plugin_config(tmp_path, monkeypatch, {"allow_gateway_injection": True})
    context, manager = _context()
    live = _session("ses_live")
    monkeypatch.setattr(server, "_sessions", {"ui-live": live})
    server.install_tui_message_injector(manager)
    try:
        assert context.inject_message("first", session_key="ses_live") is True
        assert context.inject_message("second", session_key="ses_live") is True
    finally:
        server.clear_tui_message_injector(manager)

    head, second = live["queued_prompt"], live["queued_prompts"][0]
    assert (head["text"], second["text"]) == ("first", "second")
    assert head["turn_author"]["id"] == second["turn_author"]["id"] == "plugin:notify-plugin"


def test_gateway_event_yields_the_plugin_author():
    event = SimpleNamespace(metadata={
        "hermes_plugin_id": "room-wake", "hermes_plugin_injection": True,
    })
    assert plugin_author_for_event(event) == {
        "id": "plugin:room-wake", "name": "room-wake", "is_bot": True,
    }


def test_gateway_event_honours_an_explicit_plugin_author():
    event = SimpleNamespace(metadata={
        "hermes_plugin_id": "chat-bridge", "hermes_plugin_injection": True,
        "hermes_plugin_author": {"id": "42", "name": "tester", "is_bot": False},
    })
    assert plugin_author_for_event(event) == {"id": "42", "name": "tester", "is_bot": False}


def test_a_normal_turn_stays_the_humans():
    """No injection marker: the restored source is authoritative and nothing is re-attributed."""
    assert plugin_author_for_event(SimpleNamespace(metadata={})) is None
    assert plugin_author_for_event(SimpleNamespace(metadata=None)) is None
    assert plugin_author_for_event(SimpleNamespace()) is None
    # A plugin merely NAMED in metadata is not a plugin-authored turn.
    assert plugin_author_for_event(SimpleNamespace(metadata={"hermes_plugin_id": "x"})) is None