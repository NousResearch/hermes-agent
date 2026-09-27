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


def _relay(server, event_type, sid="parent-sid", **payload):
    """Drive _on_tool_progress the way the delegate relay does."""
    server._on_tool_progress(
        sid,
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


def test_mirror_state_dict_is_scoped_per_profile(server, emits):
    # The mirror's per-child stream state (started flag, open tool) is keyed on
    # (home, key) too: with a bare key, B's relay found A's entry "already started"
    # and completed A's open tool verbatim into B's window — no message.start, and
    # A's tool payload (preview included) eavesdropped into another profile.
    server._sessions["parent-a"] = {"session_key": "parent", "profile_home": HOME_A}
    server._sessions["parent-b"] = {"session_key": "parent", "profile_home": HOME_B}
    server._sessions["live-a"] = {"session_key": "child-1", "profile_home": HOME_A, "agent": None}
    server._sessions["live-b"] = {"session_key": "child-1", "profile_home": HOME_B, "agent": None}

    _relay(server, "subagent.tool", sid="parent-a", tool_name="terminal",
           preview="SECRET-A: cat ~/.aws/credentials", child_session_id="child-1")
    _relay(server, "subagent.tool", sid="parent-b", tool_name="web_search",
           preview="b query", child_session_id="child-1")

    a = [(e, p) for e, s, p in emits if s == "live-a"]
    b = [(e, p) for e, s, p in emits if s == "live-b"]
    # Each window got its OWN synthetic turn: message.start then its parent's tool.
    assert [e for e, _ in a] == ["message.start", "tool.start"]
    assert [e for e, _ in b] == ["message.start", "tool.start"]
    assert a[1][1]["preview"] == "SECRET-A: cat ~/.aws/credentials"
    assert b[1][1]["name"] == "web_search"
    # A's open tool stays open under A's dimension — not completed by B's relay.
    assert server._child_mirrors[(HOME_A, "child-1")]["open_tool"]["name"] == "terminal"


def test_gone_parent_record_fails_closed(server, emits):
    # Mid-run parent teardown (session.close pops unconditionally; the WS orphan reaper
    # force-pops mid-turn): the relaying sid is gone from _sessions, so the event cannot
    # be attributed to a profile. None is NOT a stand-in here — it is the launch profile,
    # a real dimension — so the unresolved event must bind nothing: no liveness row under
    # the launch dimension, no mirror into a same-key launch-profile window. The complete
    # still drains the row the child registered under its real home before the teardown.
    server._sessions["parent-a"] = {"session_key": "parent", "profile_home": HOME_A}
    server._sessions["launch-live"] = {"session_key": "child-1", "agent": None}
    _relay(server, "subagent.tool", sid="parent-a", tool_name="terminal",
           preview="ls", child_session_id="child-1")
    assert server._child_run_active("child-1", HOME_A)

    server._sessions.pop("parent-a")
    _relay(server, "subagent.tool", sid="parent-a", tool_name="terminal",
           preview="more", child_session_id="child-1")

    assert (None, "child-1") not in server._active_child_runs
    assert [(e, s) for e, s, _ in emits if s == "launch-live"] == []
    assert server._child_mirrors == {}

    _relay(server, "subagent.complete", sid="parent-a",
           child_session_id="child-1", status="completed", summary="done")
    assert server._active_child_runs == {}


def test_lazy_build_guard_holds_only_the_owning_profile_lazy(server):
    # _start_agent_build's spectate guard (server.py): a lazy watch window stays lazy
    # only while the run is in flight under ITS OWN profile — a same-key run under
    # another profile must not pin this window lazy and starve its agent build.
    import threading

    server._mirror_subagent_to_child("subagent.tool", {"child_session_id": "child-1"}, HOME_A)

    lazy_b = {"session_key": "child-1", "profile_home": HOME_B, "lazy": True,
              "agent_ready": threading.Event()}
    server._start_agent_build("live-b", lazy_b)
    lazy_b["_agent_build_thread"].join(timeout=5)  # no live-b in _sessions: the build abandons quietly
    assert lazy_b.get("agent_build_started") is True
    assert "lazy" not in lazy_b

    lazy_a = {"session_key": "child-1", "profile_home": HOME_A, "lazy": True,
              "agent_ready": threading.Event()}
    server._start_agent_build("live-a", lazy_a)
    assert lazy_a.get("agent_build_started") is None
    assert lazy_a.get("lazy") is True


def test_submit_turn_guard_holds_only_the_owning_profile(server):
    # _lock_in_submit_turn's busy guard (methods_prompt.py): typing into a lazy watch
    # window is refused only while the in-flight child runs under the window's OWN
    # profile — another profile's same-key run does not fence this turn.
    import threading

    server._mirror_subagent_to_child("subagent.tool", {"child_session_id": "child-1"}, HOME_A)

    live_b = {"agent": None, "history_lock": threading.Lock(), "lazy": True,
              "running": False, "session_key": "child-1", "profile_home": HOME_B}
    err, _fields = server._lock_in_submit_turn("rid-b", "live-b", live_b, "hi", {}, False, [], None, None)
    assert err is None  # admitted: B's window is not fenced by A's run

    live_a = {"agent": None, "history_lock": threading.Lock(), "lazy": True,
              "running": False, "session_key": "child-1", "profile_home": HOME_A}
    err_a, _fields = server._lock_in_submit_turn("rid-a", "live-a", live_a, "hi", {}, False, [], None, None)
    assert err_a["error"]["code"] == 4009
