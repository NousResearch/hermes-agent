from __future__ import annotations

import threading
import types
from pathlib import Path

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_cli import plugins as plugins_mod
from hermes_cli.plugin_cards import PluginCard, PluginCardAction
from hermes_cli.plugins import PluginContext, PluginManifest
from tui_gateway import server


def _session(profile_home: str):
    return {
        "agent": types.SimpleNamespace(),
        "session_key": "session-key",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "profile_home": profile_home,
    }


def _write_profile_plugin(home: Path) -> None:
    plugin = home / "plugins" / "scope-probe"
    plugin.mkdir(parents=True)
    (home / "config.yaml").write_text("plugins:\n  enabled: [scope-probe]\n", encoding="utf-8")
    (plugin / "plugin.yaml").write_text("name: scope-probe\nversion: 0.1.0\n", encoding="utf-8")
    (plugin / "__init__.py").write_text(
        "from hermes_constants import get_hermes_home\n"
        "def register(ctx):\n"
        "    ctx.register_command('scope-act', lambda args: f'{get_hermes_home().name}:{args}')\n",
        encoding="utf-8",
    )


def test_plugin_card_action_dispatches_exact_owning_command_and_args(tmp_path):
    home = tmp_path / "profile-a"
    home.mkdir()
    token = set_hermes_home_override(home)
    plugins_mod._reset_plugin_managers_for_tests()
    seen = []
    try:
        manager = plugins_mod.get_plugin_manager()
        manager._discovered = True
        ctx = PluginContext(PluginManifest(name="release-tools"), manager)
        ctx.register_command("release-open", lambda args: seen.append(args) or "opened")
        server._sessions["sid-a"] = _session(str(home))

        response = server.handle_request(
            {
                "id": "action-1",
                "method": "plugin.card.action",
                "params": {
                    "session_id": "sid-a",
                    "plugin_id": "release-tools",
                    "command": "release-open",
                    "args": "candidate-7",
                },
            }
        )

        assert response["result"] == {"kind": "text", "text": "opened"}
        assert seen == ["candidate-7"]

        manager._plugin_commands.pop("release-open")
        unavailable = server.handle_request(
            {
                "id": "action-2",
                "method": "plugin.card.action",
                "params": {
                    "session_id": "sid-a",
                    "plugin_id": "release-tools",
                    "command": "release-open",
                    "args": "candidate-7",
                },
            }
        )
        assert unavailable["error"]["code"] == 4046
    finally:
        server._sessions.pop("sid-a", None)
        plugins_mod._reset_plugin_managers_for_tests()
        reset_hermes_home_override(token)


def test_plugin_command_result_becomes_attributed_card(tmp_path, monkeypatch):
    home = tmp_path / "profile-a"
    home.mkdir()
    token = set_hermes_home_override(home)
    plugins_mod._reset_plugin_managers_for_tests()
    try:
        manager = plugins_mod.get_plugin_manager()
        manager._discovered = True
        ctx = PluginContext(PluginManifest(name="release-tools"), manager)
        ctx.register_command(
            "release-review",
            lambda _args: PluginCard(
                title="Candidate ready",
                body="Review before opening.",
                actions=(PluginCardAction("Open", "release-open", "candidate-7"),),
            ),
        )
        ctx.register_command(
            "release-open",
            lambda _args: PluginCard(
                title="Exact review",
                body="Full review body.",
                actions=(PluginCardAction("Confirm", "release-confirm", "candidate-7"),),
            ),
        )
        server._sessions["sid-a"] = _session(str(home))
        monkeypatch.setattr(server, "_plugin_cards_supported", lambda _sid: True)
        emitted = []
        monkeypatch.setattr(
            server,
            "_publish_plugin_card",
            lambda sid, payload: emitted.append((sid, payload)) or True,
        )

        response = server.handle_request(
            {
                "id": "command-1",
                "method": "command.dispatch",
                "params": {"session_id": "sid-a", "name": "release-review", "arg": ""},
            }
        )

        assert response["result"]["type"] == "plugin_card"
        assert response["result"]["output"] == "Candidate ready\n\nReview before opening."
        assert response["result"]["card"] == {
            "plugin_id": "release-tools",
            "plugin_name": "release-tools",
            "title": "Candidate ready",
            "body": "Review before opening.",
            "actions": [
                {"label": "Open", "command": "release-open", "args": "candidate-7"}
            ],
        }

        slash_response = server.handle_request(
            {
                "id": "slash-1",
                "method": "slash.exec",
                "params": {"session_id": "sid-a", "command": "release-review"},
            }
        )
        assert slash_response["result"]["type"] == "plugin_card"
        assert slash_response["result"]["card"] == response["result"]["card"]

        action_response = server.handle_request(
            {
                "id": "action-1",
                "method": "plugin.card.action",
                "params": {
                    "session_id": "sid-a",
                    "plugin_id": "release-tools",
                    "command": "release-open",
                    "args": "candidate-7",
                },
            }
        )
        assert action_response["result"]["card"]["title"] == "Exact review"
        assert emitted == []

        monkeypatch.setattr(server, "_plugin_cards_supported", lambda _sid: False)
        fallback = server.handle_request(
            {
                "id": "command-text",
                "method": "command.dispatch",
                "params": {"session_id": "sid-a", "name": "release-review", "arg": ""},
            }
        )
        assert fallback["result"] == {
            "type": "plugin",
            "output": "Candidate ready\n\nReview before opening.",
        }
    finally:
        server._sessions.pop("sid-a", None)
        plugins_mod._reset_plugin_managers_for_tests()
        reset_hermes_home_override(token)


def test_plugin_command_can_publish_card_to_current_session(tmp_path, monkeypatch):
    home = tmp_path / "profile-a"
    home.mkdir()
    token = set_hermes_home_override(home)
    plugins_mod._reset_plugin_managers_for_tests()
    emitted = []
    try:
        manager = plugins_mod.get_plugin_manager()
        manager._discovered = True
        ctx = PluginContext(PluginManifest(name="release-tools"), manager)
        card = PluginCard(
            title="Build ready", body="Checks passed",
            actions=(PluginCardAction("Share", "release-publish", "details"),),
        )
        ctx.register_command("release-publish", lambda _args: str(ctx.publish_card(card)))
        server._sessions["sid-a"] = _session(str(home))
        monkeypatch.setattr(
            server,
            "_publish_plugin_card",
            lambda sid, payload: emitted.append(("plugin.card.show", sid, payload)) or True,
        )

        response = server.handle_request(
            {
                "id": "command-2",
                "method": "command.dispatch",
                "params": {"session_id": "sid-a", "name": "release-publish", "arg": ""},
            }
        )

        assert response["result"] == {"type": "plugin", "output": "True"}
        assert emitted == [
            (
                "plugin.card.show",
                "sid-a",
                {
                    "plugin_id": "release-tools",
                    "plugin_name": "release-tools",
                    "title": "Build ready",
                    "body": "Checks passed",
                    "actions": [
                        {"label": "Share", "command": "release-publish", "args": "details"}
                    ],
                },
            )
        ]
    finally:
        server._sessions.pop("sid-a", None)
        plugins_mod._reset_plugin_managers_for_tests()
        reset_hermes_home_override(token)


def test_plugin_card_actions_follow_session_profile_a_b_a_with_real_discovery(tmp_path, monkeypatch):
    launch_home = tmp_path / "launch"
    home_a = tmp_path / "profile-a"
    home_b = tmp_path / "profile-b"
    for home in (launch_home, home_a, home_b):
        home.mkdir()
    _write_profile_plugin(home_a)
    _write_profile_plugin(home_b)
    empty_bundled = tmp_path / "bundled"
    empty_bundled.mkdir()
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(empty_bundled))
    monkeypatch.setenv("HERMES_ENABLE_PROJECT_PLUGINS", "0")
    token = set_hermes_home_override(launch_home)
    plugins_mod._reset_plugin_managers_for_tests()
    server._sessions.update({"sid-a": _session(str(home_a)), "sid-b": _session(str(home_b))})

    def dispatch(sid: str) -> str:
        response = server.handle_request(
            {
                "id": sid,
                "method": "plugin.card.action",
                "params": {
                    "session_id": sid,
                    "plugin_id": "scope-probe",
                    "command": "scope-act",
                    "args": "go",
                },
            }
        )
        return response["result"]["text"]

    try:
        assert [dispatch("sid-a"), dispatch("sid-b"), dispatch("sid-a")] == [
            "profile-a:go",
            "profile-b:go",
            "profile-a:go",
        ]
    finally:
        server._sessions.pop("sid-a", None)
        server._sessions.pop("sid-b", None)
        plugins_mod._reset_plugin_managers_for_tests()
        reset_hermes_home_override(token)


def test_plugin_execution_during_turn_can_publish_to_current_session(tmp_path, monkeypatch):
    home = tmp_path / "profile-a"
    home.mkdir()
    token = set_hermes_home_override(home)
    plugins_mod._reset_plugin_managers_for_tests()
    emitted = []

    class Stop:
        def set(self):
            pass

    class Thread:
        def join(self):
            pass

    try:
        manager = plugins_mod.get_plugin_manager()
        manager._discovered = True
        ctx = PluginContext(PluginManifest(name="release-tools"), manager)
        card = PluginCard(
            title="Turn update", body="Background check completed",
            actions=(PluginCardAction("Acknowledge", "release-ack", "turn"),),
        )

        class Agent:
            _mute_notification_reply = False

            def run_conversation(self, _message, **_kwargs):
                assert ctx.publish_card(card) is True
                return {"messages": [], "final_response": "done"}

        agent = Agent()
        session = _session(str(home))
        session["agent"] = agent
        state = server._TurnRun(
            agent=agent,
            one_turn_restore=None,
            terminal_callback=None,
            receipt_committed=False,
        )
        monkeypatch.setattr(server, "_start_usage_ticker", lambda *_args: (Stop(), Thread()))
        monkeypatch.setattr(server, "_adopt_submit_user_row", lambda *_args: None)
        monkeypatch.setattr(
            server,
            "_publish_plugin_card",
            lambda sid, payload: emitted.append(("plugin.card.show", sid, payload)) or True,
        )

        server._invoke_agent(
            "sid-a", session, state, "hello", "hello", None, [], None, None, text="hello"
        )

        assert emitted == [
            (
                "plugin.card.show",
                "sid-a",
                {
                    "plugin_id": "release-tools",
                    "plugin_name": "release-tools",
                    "title": "Turn update",
                    "body": "Background check completed",
                    "actions": [
                        {"label": "Acknowledge", "command": "release-ack", "args": "turn"}
                    ],
                },
            )
        ]
    finally:
        plugins_mod._reset_plugin_managers_for_tests()
        reset_hermes_home_override(token)


def test_plugin_card_publication_requires_capable_attached_transport(monkeypatch):
    class Transport:
        def write(self, _frame: dict) -> bool:
            return True

        def close(self) -> None:
            pass

    transport = Transport()
    server._sessions["sid-a"] = {"transport": transport}
    emitted = []
    monkeypatch.setattr(
        server, "_emit", lambda event, sid, payload=None: emitted.append((event, sid, payload)) or True
    )
    try:
        assert server._publish_plugin_card("sid-a", {"title": "Card"}) is False
        server._advertise_plugin_cards(transport, True)
        assert server._publish_plugin_card("sid-a", {"title": "Card"}) is True
        assert emitted == [("plugin.card.show", "sid-a", {"title": "Card"})]
    finally:
        server._advertise_plugin_cards(transport, False)
        server._sessions.pop("sid-a", None)
