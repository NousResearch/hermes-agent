"""Profile scoping of the child-session live mirror (#120212).

The delegated child runs under its PARENT's profile, so both the liveness
registry and the live-session lookup during mirroring are keyed on that
profile home. A watch window on another profile that happens to share the
child's stored id must not bind to the run — neither by mirroring the relaid
``subagent.*`` events into its transport nor by reporting the run active.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def server():
    # Mocks are scoped to the initial import only (see
    # tests/tui_gateway/test_protocol.py for the rationale).
    with patch.dict(
        "sys.modules",
        {
            "hermes_constants": MagicMock(
                get_hermes_home=MagicMock(return_value="/tmp/hermes_test_mirror_scope")
            ),
            "hermes_cli.env_loader": MagicMock(),
            "hermes_cli.banner": MagicMock(),
            "hermes_state": MagicMock(),
        },
    ):
        import importlib

        mod = importlib.import_module("tui_gateway.server")

    yield mod
    mod._sessions.clear()
    __import__("tui_gateway.server_requests", fromlist=["x"]).reset_for_tests()
    mod._child_mirrors.clear()
    mod._active_child_runs.clear()


@pytest.fixture()
def emits(server, monkeypatch):
    captured: list = []
    monkeypatch.setattr(
        server,
        "_emit",
        lambda event, sid, payload=None: captured.append((event, sid, payload)),
    )
    monkeypatch.setattr(server, "_tool_progress_enabled", lambda sid: True)
    return captured


HOME_A = "/profiles/a"
HOME_B = "/profiles/b"


def _relay(server, event_type, **payload):
    """Drive _on_tool_progress the way the delegate relay does."""
    server._on_tool_progress(
        "parent-sid",
        event_type,
        payload.pop("tool_name", None),
        payload.pop("preview", None),
        None,
        goal="research X",
        task_count=1,
        task_index=0,
        **payload,
    )


def test_foreign_profile_watch_window_is_not_bound(server, emits):
    # The parent (and its child) run under profile A; a live watch session with
    # the same stored key exists under profile B.
    server._sessions["parent-sid"] = {"session_key": "parent", "profile_home": HOME_A}
    server._sessions["live-b"] = {"session_key": "child-1", "profile_home": HOME_B, "agent": None}

    _relay(server, "subagent.tool", tool_name="terminal", preview="ls", child_session_id="child-1")

    # Only the parent-sid relay event: nothing mirrored into B's window...
    assert [(e, s) for e, s, _ in emits] == [("subagent.tool", "parent-sid")]
    assert server._child_mirrors == {}
    # ...and the liveness registry is keyed under A, not B nor the launch profile.
    assert (HOME_A, "child-1") in server._active_child_runs
    assert not server._child_run_active("child-1", HOME_B)
    assert not server._child_run_active("child-1")


def test_owning_profile_watch_window_still_mirrors(server, emits):
    server._sessions["parent-sid"] = {"session_key": "parent", "profile_home": HOME_A}
    server._sessions["live-a"] = {"session_key": "child-1", "profile_home": HOME_A, "agent": None}

    _relay(server, "subagent.tool", tool_name="terminal", preview="ls", child_session_id="child-1")

    child = [(e, p) for e, s, p in emits if s == "live-a"]
    assert [e for e, _ in child] == ["message.start", "tool.start"]
    assert server._child_run_active("child-1", HOME_A)


def test_launch_profile_run_not_reported_active_for_named_profile(server, emits):
    # A parent on the launch profile (no profile_home) with no window open.
    server._sessions["parent-sid"] = {"session_key": "parent"}

    _relay(server, "subagent.tool", tool_name="terminal", child_session_id="child-2")

    assert server._child_run_active("child-2")
    assert not server._child_run_active("child-2", HOME_A)


def test_complete_clears_only_the_owning_profiles_entry(server, emits):
    server._sessions["parent-sid"] = {"session_key": "parent", "profile_home": HOME_A}
    server._sessions["live-b"] = {"session_key": "child-1", "profile_home": HOME_B, "agent": None}

    _relay(server, "subagent.tool", tool_name="terminal", child_session_id="child-1")
    # A second relay from a DIFFERENT profile's parent with the same child id
    # registers (HOME_B, "child-1") without disturbing A's entry.
    server._sessions["parent-sid"] = {"session_key": "parent", "profile_home": HOME_B}
    _relay(server, "subagent.tool", tool_name="terminal", child_session_id="child-1")
    assert server._child_run_active("child-1", HOME_A)
    assert server._child_run_active("child-1", HOME_B)

    server._sessions["parent-sid"] = {"session_key": "parent", "profile_home": HOME_B}
    _relay(server, "subagent.complete", child_session_id="child-1", status="completed", summary="done")

    assert server._child_run_active("child-1", HOME_A)
    assert not server._child_run_active("child-1", HOME_B)
