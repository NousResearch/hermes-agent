"""Background terminal events reach their commissioning live UI tab."""

import types

from tui_gateway import server


# ---------------------------------------------------------------------------
# Background-process ownership: exact tab, not first key match
# ---------------------------------------------------------------------------


def _wire_terminal_sinks(monkeypatch):
    """Install the terminal output/close sinks and capture what they emit."""
    from tools.process_registry import process_registry

    emitted: list[tuple[str, str, dict]] = []
    monkeypatch.setattr(
        server,
        "_emit",
        lambda event, sid, payload: emitted.append((event, sid, payload)),
    )
    monkeypatch.setattr(process_registry, "on_output", None, raising=False)
    monkeypatch.setattr(process_registry, "on_close", None, raising=False)
    server._wire_desktop_sinks()
    return process_registry, emitted


def test_background_output_routes_to_the_tab_that_started_it(monkeypatch):
    """Two tabs share a durable session_key; output must reach the origin tab.

    ``session_key`` is the durable conversation key, so several live windows
    can carry the same one. Matching on it alone hands background output to
    whichever tab happens to come first in iteration order — the other tab
    then sees output from a command it never ran.
    """
    process_registry, emitted = _wire_terminal_sinks(monkeypatch)

    monkeypatch.setitem(server._sessions, "tab_first", {"session_key": "shared-key"})
    monkeypatch.setitem(server._sessions, "tab_origin", {"session_key": "shared-key"})

    proc = types.SimpleNamespace(
        id="proc_owned",
        session_key="shared-key",
        origin_ui_session_id="tab_origin",
    )
    process_registry.on_output(proc, "hello")

    assert emitted == [
        ("agent.terminal.output", "tab_origin", {"process_id": "proc_owned", "chunk": "hello"})
    ]


def test_background_output_falls_back_to_session_key_without_an_origin(monkeypatch):
    """Processes spawned before origins were tracked keep legacy key routing."""
    process_registry, emitted = _wire_terminal_sinks(monkeypatch)

    monkeypatch.setitem(server._sessions, "tab_only", {"session_key": "shared-key"})

    proc = types.SimpleNamespace(
        id="proc_legacy",
        session_key="shared-key",
        origin_ui_session_id="",
    )
    process_registry.on_output(proc, "legacy")

    assert emitted == [
        ("agent.terminal.output", "tab_only", {"process_id": "proc_legacy", "chunk": "legacy"})
    ]


def test_background_output_falls_back_to_most_recent_live_tab(monkeypatch):
    """A closed origin routes to the newest live continuation, not insertion order."""
    process_registry, emitted = _wire_terminal_sinks(monkeypatch)

    monkeypatch.setitem(
        server._sessions, "tab_origin", {"session_key": "shared-key", "_finalized": True}
    )
    monkeypatch.setitem(
        server._sessions,
        "tab_recent",
        {"session_key": "shared-key", "last_active": 30.0},
    )
    monkeypatch.setitem(
        server._sessions,
        "tab_oldest",
        {"session_key": "shared-key", "last_active": 10.0},
    )
    monkeypatch.setitem(
        server._sessions,
        "tab_middle",
        {"session_key": "shared-key", "last_active": 20.0},
    )

    proc = types.SimpleNamespace(
        id="proc_orphan",
        session_key="shared-key",
        origin_ui_session_id="tab_origin",
    )
    process_registry.on_output(proc, "orphaned")

    assert emitted == [
        (
            "agent.terminal.output",
            "tab_recent",
            {"process_id": "proc_orphan", "chunk": "orphaned"},
        )
    ]


def test_terminal_close_routes_to_the_origin_tab(monkeypatch):
    """Tab-close requests follow the same exact-owner rule as output."""
    process_registry, emitted = _wire_terminal_sinks(monkeypatch)

    monkeypatch.setitem(server._sessions, "tab_first", {"session_key": "shared-key"})
    monkeypatch.setitem(server._sessions, "tab_origin", {"session_key": "shared-key"})

    proc = types.SimpleNamespace(
        id="proc_closing",
        session_key="shared-key",
        origin_ui_session_id="tab_origin",
    )
    process_registry.on_close(proc, "proc_closing")

    assert emitted == [("terminal.close", "tab_origin", {"process_id": "proc_closing"})]


def test_post_turn_drain_cannot_claim_another_live_tabs_completion(monkeypatch):
    origin = {"session_key": "shared-key"}
    other = {"session_key": "shared-key"}
    monkeypatch.setitem(server._sessions, "tab_origin", origin)
    monkeypatch.setitem(server._sessions, "tab_other", other)
    event = {"type": "completion", "session_key": "shared-key", "origin_ui_session_id": "tab_origin"}

    assert server._session_owns_notification_event("tab_origin", origin, event)
    assert not server._session_owns_notification_event("tab_other", other, event)

    # A closed origin still allows a live continuation to receive the result.
    origin["_finalized"] = True
    assert server._session_owns_notification_event("tab_other", other, event)
