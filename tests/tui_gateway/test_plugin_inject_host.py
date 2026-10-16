"""Plugin inject_message reaches the Ink TUI / desktop session it names.

The messaging gateway and the TUI must not share ``set_gateway_message_injector``.
A reported ``session_key`` is the routing key — not the ephemeral UI session id —
and the text lands on that session's prompt queue.
"""

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import hermes_yaml as yaml

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
        "running": False,
        "transport": None,
        "attached_images": [],
        "last_active": 1.0,
        **extra,
    }


def _wait_until(predicate, timeout=1.0) -> bool:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.01)
    return predicate()


def test_tui_injector_slot_does_not_replace_the_gateway_slot():
    manager = PluginManager()
    gateway = MagicMock(return_value=True)
    tui = MagicMock(return_value=False)
    gateway_owner, tui_owner = object(), object()

    manager.set_gateway_message_injector(gateway_owner, gateway)
    manager.set_tui_message_injector(tui_owner, tui)

    assert manager.has_gateway_message_injector is True
    assert manager.has_tui_message_injector is True
    assert manager.inject_gateway_message(session_key="agent:main:telegram:dm:1") is True
    assert manager.inject_tui_message(session_key="ses_tui") is False
    gateway.assert_called_once_with(session_key="agent:main:telegram:dm:1")
    tui.assert_called_once_with(session_key="ses_tui")

    manager.clear_tui_message_injector(tui_owner)
    assert manager.has_tui_message_injector is False
    assert manager.has_gateway_message_injector is True
    assert manager.inject_gateway_message(session_key="kept") is True


def test_reported_session_key_is_queued_not_the_ui_session_id(tmp_path, monkeypatch):
    """The durable session_key, not the ephemeral UI sid, selects the prompt queue."""
    _write_plugin_config(tmp_path, monkeypatch, {"allow_gateway_injection": True})
    context, manager = _context()
    gateway = MagicMock(return_value=True)
    manager.set_gateway_message_injector(object(), gateway)

    origin = _session("ses_origin", running=True, last_active=10.0)
    chatty = _session("ses_chatty", running=True, last_active=200.0)
    # UI sids deliberately differ from the durable keys a plugin reports.
    monkeypatch.setattr(server, "_sessions", {"ui-origin": origin, "ui-chatty": chatty})
    server.install_tui_message_injector(manager)
    try:
        assert context.inject_message("turn complete", session_key="ses_origin") is True
    finally:
        server.clear_tui_message_injector(manager)

    assert origin["queued_prompt"]["text"] == "turn complete"
    assert chatty.get("queued_prompt") is None
    gateway.assert_not_called()


def test_unknown_session_key_falls_through_to_the_gateway_slot(tmp_path, monkeypatch):
    _write_plugin_config(tmp_path, monkeypatch, {"allow_gateway_injection": True})
    context, manager = _context()
    gateway = MagicMock(return_value=True)
    manager.set_gateway_message_injector(object(), gateway)
    live = _session("ses_live", running=True)
    monkeypatch.setattr(server, "_sessions", {"ui-live": live})
    server.install_tui_message_injector(manager)
    try:
        assert context.inject_message(
            "wake telegram", session_key="agent:main:telegram:dm:42",
        ) is True
    finally:
        server.clear_tui_message_injector(manager)

    gateway.assert_called_once_with(
        session_key="agent:main:telegram:dm:42",
        content="wake telegram",
        plugin_id="notify-plugin",
    )
    assert live.get("queued_prompt") is None


def test_missing_session_is_not_rerouted_to_another_tui_session(tmp_path, monkeypatch):
    _write_plugin_config(tmp_path, monkeypatch, {"allow_gateway_injection": True})
    context, manager = _context()
    other = _session("ses_other", running=True, last_active=200.0)
    monkeypatch.setattr(server, "_sessions", {"ui-other": other})
    server.install_tui_message_injector(manager)
    try:
        assert context.inject_message("report", session_key="ses_gone") is False
    finally:
        server.clear_tui_message_injector(manager)

    assert other.get("queued_prompt") is None


def test_idle_session_key_drains_onto_that_prompt_queue(tmp_path, monkeypatch):
    _write_plugin_config(tmp_path, monkeypatch, {"allow_gateway_injection": True})
    context, manager = _context()
    fired = {}

    def fake_run_prompt_submit(rid, sid, session, text, **kwargs):
        fired["sid"] = sid
        fired["text"] = text

    monkeypatch.setattr(server, "_run_prompt_submit", fake_run_prompt_submit)
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda session: False)
    origin = _session("ses_idle", running=False)
    monkeypatch.setattr(server, "_sessions", {"ui-idle": origin})
    server.install_tui_message_injector(manager)
    try:
        assert context.inject_message("run finished", session_key="ses_idle") is True
        assert _wait_until(lambda: fired.get("text") == "run finished")
    finally:
        server.clear_tui_message_injector(manager)

    assert fired["sid"] == "ui-idle"


def test_cli_injection_still_bypasses_the_tui_host(tmp_path, monkeypatch):
    _write_plugin_config(tmp_path, monkeypatch, {"allow_gateway_injection": True})
    context, manager = _context()
    pending = []
    context._manager._cli_ref = SimpleNamespace(
        _agent_running=False,
        _pending_input=SimpleNamespace(put=pending.append),
        _interrupt_queue=SimpleNamespace(put=lambda item: None),
    )
    origin = _session("ses_origin", running=True)
    monkeypatch.setattr(server, "_sessions", {"ui-origin": origin})
    server.install_tui_message_injector(manager)
    try:
        assert context.inject_message("typed", session_key="ses_origin") is True
    finally:
        server.clear_tui_message_injector(manager)

    assert pending == ["typed"]
    assert origin.get("queued_prompt") is None
