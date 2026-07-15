import threading
import sqlite3
import time

import pytest

from tui_gateway import server
from tui_gateway.group_chat import GroupChatCoordinator, GroupChatStore, route_mentions


def test_room_context_settings_and_summary_snapshot_persist_across_restart(tmp_path):
    db_path = tmp_path / "groups.sqlite3"
    store = GroupChatStore(db_path)
    room = store.create_room("Long room", [{"profile": "default"}], {
        "trigger_tokens": 1200,
        "max_history_tokens": 800,
        "tail_message_count": 3,
    })
    store.save_summary(room["id"], "## 已完成\n- 初始化", through_seq=17)
    store.close()

    loaded = GroupChatStore(db_path).get_room(room["id"])

    assert loaded["trigger_tokens"] == 1200
    assert loaded["max_history_tokens"] == 800
    assert loaded["tail_message_count"] == 3
    assert loaded["summary"] == "## 已完成\n- 初始化"
    assert loaded["summary_through_seq"] == 17


def test_room_create_rpc_accepts_context_limits(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_group_store", GroupChatStore(tmp_path / "rpc.sqlite3"))
    monkeypatch.setattr(server, "_group_coordinator", None)
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name == "default")

    room = server._methods["group.room.create"]("create", {
        "name": "Bounded", "members": [{"profile": "default"}],
        "trigger_tokens": 1000, "max_history_tokens": 700, "tail_message_count": 5,
    })["result"]["room"]

    assert (room["trigger_tokens"], room["max_history_tokens"], room["tail_message_count"]) == (
        1000, 700, 5
    )


def test_interleaved_agent_messages_paginate_without_loss(tmp_path):
    store = GroupChatStore(tmp_path / "groups.sqlite3")
    room = store.create_room("Concurrent", [{"profile": "a"}, {"profile": "b"}])
    store.append_event(room["id"], "message.start", {"message_id": "A"}, "a")
    store.append_event(room["id"], "message.start", {"message_id": "B"}, "b")
    store.append_event(room["id"], "message.complete", {"message_id": "B", "text": "B"}, "b")
    store.append_event(room["id"], "message.complete", {"message_id": "A", "text": "A"}, "a")

    newest = store.message_page(room["id"], limit=1)
    older = store.message_page(room["id"], before_seq=newest["before_seq"], limit=1)

    assert [item["id"] for item in newest["messages"]] == ["A"]
    assert [item["id"] for item in older["messages"]] == ["B"]
    assert all(isinstance(item["seq"], int) for item in newest["messages"] + older["messages"])


def test_canonical_message_pagination_is_stable_and_keeps_tool_group_whole(tmp_path):
    store = GroupChatStore(tmp_path / "groups.sqlite3")
    room = store.create_room("History", [{"profile": "worker"}])
    first = store.append_event(room["id"], "user.message", {"text": "one"})
    store.append_event(room["id"], "message.start", {"message_id": "m1"}, "worker")
    store.append_event(room["id"], "tool.start", {
        "message_id": "m1", "tool_id": "t1", "name": "terminal"
    }, "worker")
    store.append_event(room["id"], "tool.complete", {
        "message_id": "m1", "tool_id": "t1", "result": "ok"
    }, "worker")
    store.append_event(room["id"], "message.complete", {
        "message_id": "m1", "text": "done"
    }, "worker")
    store.append_event(room["id"], "user.message", {"text": "two"})

    newest = store.message_page(room["id"], limit=2)
    older = store.message_page(room["id"], before_seq=newest["before_seq"], limit=2)

    assert [(item["role"], item["content"]) for item in newest["messages"]] == [
        ("assistant", "done"), ("user", "two")
    ]
    assert newest["messages"][0]["tools"] == [{
        "tool_id": "t1", "name": "terminal", "status": "complete", "result": "ok"
    }]
    assert newest["has_more"] is True
    assert older["messages"] == [{
        "id": f"group-{first['seq']}", "seq": 1, "role": "user", "content": "one",
        "status": "complete", "created_at": first["created_at"],
    }]
    assert older["has_more"] is False
    assert store.message_page(room["id"], before_seq=999999, limit=2) == newest


def test_group_timeline_rpc_exposes_canonical_cursor_page(monkeypatch, tmp_path):
    store = GroupChatStore(tmp_path / "rpc.sqlite3")
    room = store.create_room("RPC history", [{"profile": "worker"}])
    store.append_event(room["id"], "user.message", {"text": "one"})
    store.append_event(room["id"], "message.complete", {
        "message_id": "m1", "text": "done"
    }, "worker")
    store.append_event(room["id"], "user.message", {"text": "two"})
    monkeypatch.setattr(server, "_group_store", store)

    response = server._methods["group.timeline"]("page", {
        "room_id": room["id"], "limit": 2,
    })["result"]

    assert [message["content"] for message in response["messages"]] == ["done", "two"]
    assert response["has_more"] is True
    assert isinstance(response["before_seq"], int)


def test_concurrent_compression_never_regresses_summary_cursor(tmp_path):
    store = GroupChatStore(tmp_path / "groups.sqlite3")
    room = store.create_room("Race", [{"profile": "a"}], {
        "trigger_tokens": 1, "max_history_tokens": 20, "tail_message_count": 0,
    })
    store.append_event(room["id"], "user.message", {"text": "first"})
    entered, release = threading.Event(), threading.Event()
    calls = 0

    def summarizer(previous, messages):
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            release.wait(timeout=2)
        parts = [previous] if previous else []
        parts.extend(message["content"] for message in messages)
        return "|".join(parts)

    coordinator = GroupChatCoordinator(
        store, lambda _: {}, lambda *_: None, lambda _: True, lambda *_: True,
        summarizer=summarizer, token_counter=len,
    )
    first = threading.Thread(target=lambda: coordinator.compress_room(room["id"]))
    first.start()
    assert entered.wait(timeout=2)
    store.append_event(room["id"], "user.message", {"text": "second"})
    second = threading.Thread(target=lambda: coordinator.compress_room(room["id"]))
    second.start()
    release.set()
    first.join(timeout=2)
    second.join(timeout=2)

    latest = store.get_room(room["id"])
    assert latest["summary_through_seq"] == 2
    assert "first" in latest["summary"] and "second" in latest["summary"]


def test_compression_repeats_without_skipping_more_than_200_messages(tmp_path):
    store = GroupChatStore(tmp_path / "groups.sqlite3")
    room = store.create_room("Long", [{"profile": "a", "session_id": "sid"}], {
        "trigger_tokens": 1, "max_history_tokens": 30, "tail_message_count": 2,
    })
    for index in range(205):
        store.append_event(room["id"], "user.message", {"text": f"message-{index}-xxxxxxxx"})
    coordinator = GroupChatCoordinator(
        store, lambda _: "unused", lambda *_: None, lambda _: True, lambda *_: True,
        token_counter=len,
    )

    assert coordinator.compress_room(room["id"]) is True
    first_cursor = store.get_room(room["id"])["summary_through_seq"]
    store.append_event(room["id"], "user.message", {"text": "new-message-xxxxxxxx"})
    assert coordinator.compress_room(room["id"]) is True
    latest = store.get_room(room["id"])

    assert "message-0-" in latest["summary"]
    assert latest["summary_through_seq"] > first_cursor
    tail = [message for message in latest["messages"] if message["seq"] > latest["summary_through_seq"]]
    assert sum(len(message["content"]) for message in tail) <= latest["max_history_tokens"]


def test_compression_summarizes_old_completed_messages_and_projection_uses_snapshot(tmp_path):
    store = GroupChatStore(tmp_path / "groups.sqlite3")
    room = store.create_room("Compress", [{"profile": "a", "session_id": "sid"}], {
        "trigger_tokens": 12, "max_history_tokens": 100, "tail_message_count": 1,
    })
    store.append_event(room["id"], "user.message", {"text": "old request xxxxxxxxxxxx"})
    old = store.append_event(room["id"], "message.complete", {
        "message_id": "old", "text": "old peer answer yyyyyyyyyyyy"
    }, "b")
    store.append_event(room["id"], "message.complete", {
        "message_id": "tail", "text": "tail peer answer"
    }, "b")
    captured = []
    coordinator = GroupChatCoordinator(
        store, lambda _params: "unused", lambda _sid, prompt, _emit: captured.append(prompt),
        lambda _sid: True, lambda *_: True,
        summarizer=lambda previous, messages: (
            "## 房间摘要\n" + previous + "\n" + "\n".join(m["content"] for m in messages)
        ),
        token_counter=lambda text: len(text),
    )

    assert coordinator.compress_room(room["id"]) is True
    snapshot = store.get_room(room["id"])
    assert "old request" in snapshot["summary"]
    assert "old peer answer" in snapshot["summary"]
    assert "tail peer answer" not in snapshot["summary"]
    assert snapshot["summary_through_seq"] == old["seq"]

    coordinator.send(room["id"], "next")
    deadline = time.time() + 2
    while not captured and time.time() < deadline:
        time.sleep(0.01)
    assert captured[0].startswith("Room summary through canonical cursor")
    assert "## 房间摘要" in captured[0]
    assert "old peer answer" in captured[0]
    assert "tail peer answer" in captured[0]
    assert captured[0].count("next") == 1


def test_deleted_room_rejects_late_mention_dispatch_claim(tmp_path):
    store = GroupChatStore(tmp_path / "groups.sqlite3")
    room = store.create_room("Gone", [{"profile": "a"}])
    assert store.delete_room(room["id"]) is True

    assert store.claim_mention_dispatch(room["id"], "late") is False
    assert store._db.execute(
        "SELECT COUNT(*) FROM group_mention_dispatches WHERE room_id=?", (room["id"],)
    ).fetchone()[0] == 0


def test_append_event_rejects_missing_room(tmp_path):
    store = GroupChatStore(tmp_path / "groups.sqlite3")

    with pytest.raises(KeyError, match="missing"):
        store.append_event("missing", "message.delta", {"text": "late"})

    assert store._db.execute("SELECT COUNT(*) FROM group_timeline").fetchone()[0] == 0


def test_delete_room_releases_waiting_group_turn(monkeypatch, tmp_path):
    store = GroupChatStore(tmp_path / "groups.sqlite3")
    room = store.create_room("Ops", [{
        "profile": "default", "runtime_session_id": "sid-a",
        "stored_session_id": "stored-a",
    }])
    waiting = threading.Event()
    released = threading.Event()

    class Coordinator:
        def stop(self, _room_id, _profile):
            waiting.set()
            return True

    done = threading.Event()
    server._group_turn_done["sid-a"] = done
    server._group_projectors["sid-a"] = lambda *_args: {}
    monkeypatch.setattr(server, "_group_store", store)
    monkeypatch.setattr(server, "_group_coordinator", Coordinator())
    monkeypatch.setattr(server, "_group_session_targets", {
        "sid-a": (room["id"], "default")
    })
    monkeypatch.setattr(server, "_group_stream_ids", {"sid-a": "stream-a"})

    waiter = threading.Thread(target=lambda: (done.wait(), released.set()), daemon=True)
    waiter.start()
    response = server._methods["group.room.delete"]("delete", {"room_id": room["id"]})

    assert response["result"]["deleted"] is True
    assert waiting.is_set()
    assert released.wait(timeout=1)
    assert "sid-a" not in server._group_turn_done
    assert "sid-a" not in server._group_projectors


def test_delete_room_stops_members_and_drops_late_session_events(monkeypatch, tmp_path):
    store = GroupChatStore(tmp_path / "groups.sqlite3")
    room = store.create_room("Ops", [
        {"profile": "default", "runtime_session_id": "sid-a"},
        {"profile": "reviewer", "runtime_session_id": "sid-b"},
    ])
    stopped = []

    class Coordinator:
        def stop(self, room_id, profile):
            stopped.append((room_id, profile))
            return True

    monkeypatch.setattr(server, "_group_store", store)
    monkeypatch.setattr(server, "_group_coordinator", Coordinator())
    monkeypatch.setattr(server, "_group_session_targets", {
        "sid-a": (room["id"], "default"), "sid-b": (room["id"], "reviewer")
    })
    monkeypatch.setattr(server, "_group_stream_ids", {"sid-a": "stream-a", "sid-b": "stream-b"})
    subscriber = object()
    monkeypatch.setattr(server, "_group_subscribers", {room["id"]: {subscriber}})

    response = server._methods["group.room.delete"]("delete", {"room_id": room["id"]})

    assert response["result"]["deleted"] is True
    assert stopped == [(room["id"], "default"), (room["id"], "reviewer")]
    assert server._group_session_targets == {}
    assert server._group_stream_ids == {}
    assert server._group_subscribers == {}
    server._emit("message.delta", "sid-a", {"text": "too late"})
    assert store._db.execute("SELECT COUNT(*) FROM group_timeline").fetchone()[0] == 0


def test_emit_ignores_target_whose_room_was_already_deleted(monkeypatch, tmp_path):
    store = GroupChatStore(tmp_path / "groups.sqlite3")
    room = store.create_room("Race", [{"profile": "default"}])
    store.delete_room(room["id"])
    monkeypatch.setattr(server, "_group_store", store)
    monkeypatch.setattr(server, "_group_session_targets", {
        "late-sid": (room["id"], "default")
    })
    monkeypatch.setattr(server, "_group_stream_ids", {"late-sid": "stream"})

    server._emit("message.delta", "late-sid", {"text": "late"})

    assert "late-sid" not in server._group_session_targets
    assert "late-sid" not in server._group_stream_ids


@pytest.mark.parametrize("members", [
    "default",
    [],
    ["default"],
    [{}],
    [{"profile": 7}],
    [{"profile": ""}],
    [{"profile": "default", "name": 7}],
    [{"profile": "default"}, {"profile": "default"}],
    [{"profile": "x" * 65}],
])
def test_room_create_rejects_invalid_members(monkeypatch, tmp_path, members):
    monkeypatch.setattr(server, "_group_store", GroupChatStore(tmp_path / "rpc.sqlite3"))
    monkeypatch.setattr(server, "_group_coordinator", None)

    response = server._methods["group.room.create"]("create", {
        "name": "Team", "members": members,
    })

    assert response["error"]["code"] == 4000


def test_room_create_rejects_unknown_profile(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_group_store", GroupChatStore(tmp_path / "rpc.sqlite3"))
    monkeypatch.setattr(server, "_group_coordinator", None)
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name == "default")

    response = server._methods["group.room.create"]("create", {
        "name": "Team", "members": [{"profile": "unknown", "name": "Ghost"}],
    })

    assert response["error"]["code"] == 4000
    assert "unknown" in response["error"]["message"]
    assert server._group_store.list_rooms() == []


def test_subscribe_broadcasts_group_events_and_unsubscribe_stops_them(monkeypatch, tmp_path):
    store = GroupChatStore(tmp_path / "rpc.sqlite3")
    room = store.create_room("Live", [{"profile": "default"}])

    class Transport:
        def __init__(self):
            self.frames = []

        def write(self, frame):
            self.frames.append(frame)
            return True

        def close(self):
            pass

    subscriber = Transport()
    monkeypatch.setattr(server, "_group_store", store)
    monkeypatch.setattr(server, "_group_subscribers", {})
    monkeypatch.setattr(server, "_group_session_targets", {
        "sid": (room["id"], "default")
    })
    monkeypatch.setattr(server, "_group_stream_ids", {})

    response = server.dispatch({
        "jsonrpc": "2.0", "id": "sub", "method": "group.subscribe",
        "params": {"room_id": room["id"]},
    }, subscriber)
    assert response["result"]["subscribed"] is True
    server._emit("message.delta", "sid", {"text": "hello"})
    assert any(frame.get("params", {}).get("type") == "group.event" for frame in subscriber.frames)

    server.dispatch({
        "jsonrpc": "2.0", "id": "unsub", "method": "group.unsubscribe",
        "params": {"room_id": room["id"]},
    }, subscriber)
    count = len(subscriber.frames)
    server._emit("message.delta", "sid", {"text": "gone"})
    assert len(subscriber.frames) == count


def test_transport_disconnect_cleans_group_subscriptions(monkeypatch, tmp_path):
    store = GroupChatStore(tmp_path / "rpc.sqlite3")
    room = store.create_room("Live", [{"profile": "default"}])
    transport = object()
    monkeypatch.setattr(server, "_group_store", store)
    monkeypatch.setattr(server, "_group_subscribers", {room["id"]: {transport}})

    server._close_sessions_for_transport(transport)

    assert server._group_subscribers == {}


def test_legacy_member_session_column_migrates_to_distinct_runtime_and_stored_ids(tmp_path):
    db_path = tmp_path / "groups.sqlite3"
    db = sqlite3.connect(db_path)
    db.executescript(
        """
        CREATE TABLE group_rooms (id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at REAL NOT NULL);
        CREATE TABLE group_members (
            room_id TEXT NOT NULL, profile TEXT NOT NULL, name TEXT NOT NULL,
            session_id TEXT, ordinal INTEGER NOT NULL, PRIMARY KEY (room_id, profile)
        );
        CREATE TABLE group_timeline (
            seq INTEGER PRIMARY KEY AUTOINCREMENT, room_id TEXT NOT NULL,
            event_type TEXT NOT NULL, member_profile TEXT,
            payload_json TEXT NOT NULL, created_at REAL NOT NULL
        );
        CREATE TABLE group_projection_cursors (
            room_id TEXT NOT NULL, profile TEXT NOT NULL, seq INTEGER NOT NULL,
            PRIMARY KEY (room_id, profile)
        );
        INSERT INTO group_rooms VALUES ('room', 'Legacy', 1);
        INSERT INTO group_members VALUES ('room', 'worker', 'Worker', 'old-runtime', 0);
        """
    )
    db.commit()
    db.close()

    store = GroupChatStore(db_path)

    assert store.get_room("room")["members"] == [{
        "profile": "worker", "name": "Worker",
        "runtime_session_id": "old-runtime",
    }]
    columns = {row[1] for row in store._db.execute("PRAGMA table_info(group_members)")}
    assert {"runtime_session_id", "stored_session_id"} <= columns


def test_room_workspace_persists_and_is_used_for_member_session(tmp_path):
    workspace = tmp_path / "repo"
    workspace.mkdir()
    store = GroupChatStore(tmp_path / "groups.sqlite3")
    room = store.create_room(
        "Workspace Team",
        [{"profile": "default", "name": "Hermes"}],
        workspace=str(workspace),
    )
    created = []
    submitted = threading.Event()

    def create_session(params):
        created.append(params)
        return {
            "runtime_session_id": "runtime-default",
            "stored_session_id": "stored-default",
        }

    coordinator = GroupChatCoordinator(
        store,
        create_session,
        lambda *_args: submitted.set(),
        lambda _sid: True,
        lambda *_args: True,
    )
    coordinator.send(room["id"], "inspect workspace")

    assert submitted.wait(timeout=2)
    assert store.get_room(room["id"])["workspace"] == str(workspace)
    assert created == [{
        "profile": "default", "room_id": room["id"], "workspace": str(workspace)
    }]


def test_room_members_and_timeline_persist_across_store_instances(tmp_path):
    db_path = tmp_path / "groups.sqlite3"
    store = GroupChatStore(db_path)
    room = store.create_room("Release", [{"profile": "default", "name": "Hermes"}])
    store.append_event(room["id"], "user.message", {"text": "ship it"})
    store.close()

    reopened = GroupChatStore(db_path)
    loaded = reopened.get_room(room["id"])

    assert loaded["name"] == "Release"
    assert loaded["members"] == [{"profile": "default", "name": "Hermes"}]
    assert [event["payload"]["text"] for event in reopened.timeline(room["id"])] == ["ship it"]


def test_mentions_route_by_name_profile_and_all():
    members = [
        {"profile": "default", "name": "Hermes"},
        {"profile": "reviewer", "name": "Review Bot"},
    ]

    assert [m["profile"] for m in route_mentions("@Hermes implement", members)] == ["default"]
    assert [m["profile"] for m in route_mentions("@reviewer inspect", members)] == ["reviewer"]
    assert [m["profile"] for m in route_mentions("@all vote", members)] == ["default", "reviewer"]
    assert [m["profile"] for m in route_mentions("discuss", members)] == ["default", "reviewer"]
    assert route_mentions("@HermesExtra do not misroute", members) == []


def test_send_creates_independent_sessions_and_projects_streams(tmp_path):
    store = GroupChatStore(tmp_path / "groups.sqlite3")
    room = store.create_room("Pair", [
        {"profile": "default", "name": "Builder"},
        {"profile": "reviewer", "name": "Reviewer"},
    ])
    both_running = threading.Barrier(2)
    finished = threading.Event()
    calls = []

    def create_session(params):
        return {
            "runtime_session_id": f"runtime-{params['profile']}",
            "stored_session_id": f"stored-{params['profile']}",
        }

    def submit(session_id, text, emit):
        calls.append((session_id, text))
        both_running.wait(timeout=2)
        emit("message.delta", {"text": session_id[-1]})
        emit("message.complete", {"text": "done"})
        if len(calls) == 2:
            finished.set()

    coordinator = GroupChatCoordinator(store, create_session, submit, lambda _sid: True, lambda *_args: True)
    result = coordinator.send(room["id"], "review this")

    assert result["targets"] == ["default", "reviewer"]
    assert finished.wait(timeout=2)
    assert {call[0] for call in calls} == {"runtime-default", "runtime-reviewer"}
    events = store.timeline(room["id"])
    assert {event["member_profile"] for event in events if event["type"] == "message.delta"} == {"default", "reviewer"}
    assert store.get_room(room["id"])["members"] == [
        {"profile": "default", "name": "Builder", "runtime_session_id": "runtime-default", "stored_session_id": "stored-default"},
        {"profile": "reviewer", "name": "Reviewer", "runtime_session_id": "runtime-reviewer", "stored_session_id": "stored-reviewer"},
    ]


def test_agent_reply_mention_recursively_enqueues_the_canonical_reply(tmp_path):
    store = GroupChatStore(tmp_path / "groups.sqlite3")
    room = store.create_room("Relay", [
        {"profile": "builder", "name": "Builder", "session_id": "sid-builder"},
        {"profile": "reviewer", "name": "Review Bot", "session_id": "sid-reviewer"},
    ])
    calls = []
    finished = threading.Event()

    def submit(session_id, text, emit):
        calls.append((session_id, text))
        if session_id == "sid-builder":
            emit("message.complete", {"message_id": "builder-reply", "text": "@Review Bot please inspect"})
        else:
            emit("message.complete", {"message_id": "reviewer-reply", "text": "looks good"})
            finished.set()

    coordinator = GroupChatCoordinator(
        store, lambda _params: "unused", submit, lambda _sid: True, lambda *_: True
    )
    coordinator.send(room["id"], "@Builder implement")

    assert finished.wait(timeout=2)
    assert calls == [
        ("sid-builder", "@Builder implement"),
        ("sid-reviewer", "@Review Bot please inspect"),
    ]
    completions = [event for event in store.timeline(room["id"]) if event["type"] == "message.complete"]
    assert [(event["member_profile"], event["payload"]["mention_depth"]) for event in completions] == [
        ("builder", 0), ("reviewer", 1),
    ]


def test_mentions_require_exact_boundaries_and_support_special_names():
    members = [
        {"profile": "quality", "name": "QA (EU)"},
        {"profile": "review", "name": "Review.Bot+1"},
    ]

    assert [m["profile"] for m in route_mentions("(@QA (EU)), go", members)] == ["quality"]
    assert [m["profile"] for m in route_mentions("@Review.Bot+1：go", members)] == ["review"]
    assert route_mentions("x@QA (EU) no", members) == []
    assert route_mentions("@QA (EU)_extra no", members) == []
    assert route_mentions("@unknown no", members) == []


def test_duplicate_complete_is_idempotent_for_recursive_trigger(tmp_path):
    store = GroupChatStore(tmp_path / "groups.sqlite3")
    room = store.create_room("Once", [
        {"profile": "a", "name": "A", "session_id": "sid-a"},
        {"profile": "b", "name": "B", "session_id": "sid-b"},
    ])
    calls = []
    finished = threading.Event()

    def submit(session_id, text, emit):
        calls.append((session_id, text))
        if session_id == "sid-a":
            payload = {"message_id": "same-reply", "text": "@B inspect"}
            emit("message.complete", payload)
            emit("message.complete", payload)
        else:
            emit("message.complete", {"message_id": "b-reply", "text": "done"})
            finished.set()

    coordinator = GroupChatCoordinator(
        store, lambda _params: "unused", submit, lambda _sid: True, lambda *_: True
    )
    coordinator.send(room["id"], "@A start")

    assert finished.wait(timeout=2)
    time.sleep(0.05)
    assert [call for call in calls if call[0] == "sid-b"] == [("sid-b", "@B inspect")]


def test_recursive_mention_creates_target_session_on_first_use(tmp_path):
    store = GroupChatStore(tmp_path / "groups.sqlite3")
    room = store.create_room("Team", [
        {"profile": "a", "session_id": "sid-a"}, {"profile": "b"},
    ])
    created, called = [], []
    done = threading.Event()

    def create_session(params):
        created.append(params)
        return {"runtime_session_id": "sid-b", "stored_session_id": "stored-b"}

    def submit(sid, _prompt, emit):
        called.append(sid)
        if sid == "sid-a":
            emit("message.complete", {"message_id": "a1", "text": "@b inspect"})
        else:
            emit("message.complete", {"message_id": "b1", "text": "done"})
            done.set()

    coordinator = GroupChatCoordinator(store, create_session, submit, lambda _sid: True, lambda *_: True)
    coordinator.send(room["id"], "@a start")

    assert done.wait(timeout=2)
    assert called == ["sid-a", "sid-b"]
    assert created[0]["profile"] == "b"


def test_parallel_first_sends_share_one_member_session(tmp_path):
    store = GroupChatStore(tmp_path / "groups.sqlite3")
    room = store.create_room("Team", [{"profile": "a"}])
    created, submitted = [], []
    done = threading.Event()

    def create_session(_params):
        created.append(True)
        return {"runtime_session_id": "sid-a", "stored_session_id": "stored-a"}

    def submit(sid, _prompt, _emit):
        submitted.append(sid)
        if len(submitted) == 2:
            done.set()

    coordinator = GroupChatCoordinator(store, create_session, submit, lambda _sid: True, lambda *_: True)
    threads = [threading.Thread(target=lambda: coordinator.send(room["id"], "hello")) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert done.wait(timeout=2)
    assert len(created) == 1
    assert submitted == ["sid-a", "sid-a"]


def test_recursive_mentions_stop_at_room_max_depth(tmp_path):
    store = GroupChatStore(tmp_path / "groups.sqlite3")
    room = store.create_room("Bounded", [
        {"profile": "a", "name": "A", "session_id": "sid-a"},
        {"profile": "b", "name": "B", "session_id": "sid-b"},
    ], max_mention_depth=1)
    depths = []
    stopped = threading.Event()

    def submit(session_id, _text, emit):
        target = "B" if session_id == "sid-a" else "A"
        event = emit("message.complete", {"text": f"@{target} continue"})
        depths.append(event["payload"]["mention_depth"])
        if event["payload"]["mention_depth"] == 1:
            stopped.set()

    coordinator = GroupChatCoordinator(
        store, lambda _params: "unused", submit, lambda _sid: True, lambda *_: True
    )
    coordinator.send(room["id"], "@A start")

    assert stopped.wait(timeout=2)
    time.sleep(0.05)
    assert sorted(depths) == [0, 1]
    assert store.get_room(room["id"])["max_mention_depth"] == 1


def test_room_mention_depth_defaults_to_four_and_clamps_to_ten(tmp_path):
    store = GroupChatStore(tmp_path / "groups.sqlite3")
    default = store.create_room("Default", [{"profile": "a", "name": "A"}])
    capped = store.create_room(
        "Capped", [{"profile": "a", "name": "A"}], max_mention_depth=99
    )

    assert default["max_mention_depth"] == 4
    assert capped["max_mention_depth"] == 10


def test_same_profile_in_different_rooms_gets_independent_sessions(tmp_path):
    store = GroupChatStore(tmp_path / "groups.sqlite3")
    first = store.create_room("One", [{"profile": "worker", "name": "Worker"}])
    second = store.create_room("Two", [{"profile": "worker", "name": "Worker"}])
    created = []
    submitted = []
    done = threading.Event()

    def create_session(params):
        index = len(created) + 1
        created.append(params)
        return {
            "runtime_session_id": f"runtime-{index}",
            "stored_session_id": f"stored-{index}",
        }

    def submit(session_id, _text, _emit):
        submitted.append(session_id)
        if len(submitted) == 2:
            done.set()

    coordinator = GroupChatCoordinator(
        store, create_session, submit, lambda _sid: True, lambda *_: True
    )
    coordinator.send(first["id"], "hello")
    coordinator.send(second["id"], "hello")

    assert done.wait(timeout=2)
    assert set(submitted) == {"runtime-1", "runtime-2"}
    assert [params["room_id"] for params in created] == [first["id"], second["id"]]


def test_group_rpc_methods_are_registered():
    expected = {
        "group.room.create", "group.room.list", "group.room.get", "group.room.delete",
        "group.timeline", "group.send", "group.message.send", "group.stop", "group.run.interrupt",
        "group.subscribe", "group.unsubscribe", "group.approval.respond",
    }
    assert expected <= set(server._methods)


def test_group_room_create_rpc_persists_clamped_max_mention_depth(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_group_store", GroupChatStore(tmp_path / "rpc.sqlite3"))
    monkeypatch.setattr(server, "_group_coordinator", None)
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name == "default")

    response = server._methods["group.room.create"]("create", {
        "name": "Team",
        "members": [{"profile": "default", "name": "Hermes"}],
        "max_mention_depth": 42,
    })

    assert response["result"]["room"]["max_mention_depth"] == 10


def test_group_runtime_initializes_without_locking_itself(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_group_store", GroupChatStore(tmp_path / "rpc.sqlite3"))
    monkeypatch.setattr(server, "_group_coordinator", None)
    completed = threading.Event()
    thread = threading.Thread(target=lambda: (server._group_runtime(), completed.set()), daemon=True)
    thread.start()
    assert completed.wait(timeout=0.2)


def test_group_runtime_resumes_durable_member_lazily_on_first_send(monkeypatch, tmp_path):
    store = GroupChatStore(tmp_path / "rpc.sqlite3")
    room = store.create_room("Persistent", [{
        "profile": "worker", "name": "Worker",
        "runtime_session_id": "runtime-before-restart",
        "stored_session_id": "stored-worker",
    }])
    calls = []

    def resume(_rid, params):
        calls.append(params)
        server._sessions["runtime-after-restart"] = {}
        return {"result": {"session_id": "runtime-after-restart", "resumed": "stored-worker"}}

    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setattr(server, "_group_store", store)
    monkeypatch.setattr(server, "_group_coordinator", None)
    monkeypatch.setattr(server, "_group_session_targets", {})
    monkeypatch.setitem(server._methods, "session.resume", resume)

    coordinator = server._group_runtime()

    assert calls == []
    def submit(_rid, params):
        server._emit("message.complete", params["session_id"], {"text": "ok"})
        return {"result": {"status": "streaming"}}
    monkeypatch.setitem(server._methods, "prompt.submit", submit)
    coordinator.send(room["id"], "hello")
    deadline = time.time() + 2
    while not calls and time.time() < deadline:
        time.sleep(0.01)

    assert calls == [{
        "session_id": "stored-worker", "profile": "worker", "source": "desktop"
    }]
    assert store.get_room(room["id"])["members"][0] == {
        "profile": "worker", "name": "Worker",
        "runtime_session_id": "runtime-after-restart",
        "stored_session_id": "stored-worker",
    }
    assert server._group_session_targets == {
        "runtime-after-restart": (room["id"], "worker")
    }


def test_group_runtime_clears_stale_runtime_when_durable_resume_is_missing(monkeypatch, tmp_path):
    store = GroupChatStore(tmp_path / "rpc.sqlite3")
    room = store.create_room("Missing", [{
        "profile": "worker", "name": "Worker",
        "runtime_session_id": "runtime-before-restart",
        "stored_session_id": "missing-stored-worker",
    }])

    monkeypatch.setattr(server, "_group_store", store)
    monkeypatch.setattr(server, "_group_coordinator", None)
    monkeypatch.setattr(server, "_group_session_targets", {})
    monkeypatch.setitem(
        server._methods,
        "session.resume",
        lambda rid, _params: server._err(rid, 4007, "session not found"),
    )

    coordinator = server._group_runtime()
    assert store.get_room(room["id"])["members"][0]["runtime_session_id"] == "runtime-before-restart"

    coordinator.send(room["id"], "hello")
    deadline = time.time() + 2
    while store.get_room(room["id"])["members"][0].get("runtime_session_id") and time.time() < deadline:
        time.sleep(0.01)

    assert store.get_room(room["id"])["members"][0] == {
        "profile": "worker", "name": "Worker",
        "stored_session_id": "missing-stored-worker",
    }


def test_stop_and_approval_target_only_the_selected_member(tmp_path):
    store = GroupChatStore(tmp_path / "groups.sqlite3")
    room = store.create_room("Ops", [
        {"profile": "default", "name": "A", "session_id": "sid-a"},
        {"profile": "reviewer", "name": "B", "session_id": "sid-b"},
    ])
    interrupted = []
    approvals = []
    coordinator = GroupChatCoordinator(
        store, lambda _params: "unused", lambda *_args: None,
        lambda sid: interrupted.append(sid) or True,
        lambda sid, choice, all_: approvals.append((sid, choice, all_)) or 1,
    )

    assert coordinator.stop(room["id"], "reviewer") is True
    assert coordinator.respond_approval(room["id"], "default", "allow", True) == 1
    assert interrupted == ["sid-b"]
    assert approvals == [("sid-a", "allow", True)]


def test_later_turn_projects_only_new_group_context_into_one_user_prompt(tmp_path):
    store = GroupChatStore(tmp_path / "groups.sqlite3")
    room = store.create_room("Team", [{"profile": "default", "name": "A", "session_id": "sid-a"}])
    prompts = []
    done = threading.Event()

    def submit(_sid, text, emit):
        prompts.append(text)
        emit("message.complete", {"text": f"answer-{len(prompts)}"})
        done.set()

    coordinator = GroupChatCoordinator(store, lambda _params: "unused", submit, lambda _sid: True, lambda *_: True)
    coordinator.send(room["id"], "first")
    assert done.wait(timeout=2)
    store.append_event(room["id"], "message.complete", {"text": "peer-answer"}, "reviewer")
    done.clear()
    coordinator.send(room["id"], "second")
    assert done.wait(timeout=2)

    assert prompts[0] == "first"
    assert prompts[1].count("second") == 1
    assert "peer-answer" in prompts[1]
    assert "answer-1" not in prompts[1]
    assert "first" not in prompts[1]

    reopened_prompts = []
    reopened_done = threading.Event()
    def reopened_submit(_sid, text, _emit):
        reopened_prompts.append(text)
        reopened_done.set()
    reopened = GroupChatCoordinator(store, lambda _params: "unused", reopened_submit, lambda _sid: True, lambda *_: True)
    reopened.send(room["id"], "third")
    assert reopened_done.wait(timeout=2)
    assert "peer-answer" not in reopened_prompts[0]


def test_rapid_sends_to_one_member_run_fifo_with_distinct_message_ids(tmp_path):
    store = GroupChatStore(tmp_path / "groups.sqlite3")
    room = store.create_room("Queue", [{"profile": "worker", "name": "Worker", "session_id": "sid"}])
    first_started = threading.Event()
    release_first = threading.Event()
    finished = threading.Event()
    active = 0
    calls = []
    lock = threading.Lock()

    def submit(_sid, text, emit):
        nonlocal active
        with lock:
            active += 1
            calls.append((text, active))
        emit("message.start", {"message_id": f"message-{text}"})
        if text == "first":
            first_started.set()
            assert release_first.wait(timeout=2)
        emit("message.complete", {"message_id": f"message-{text}", "text": text})
        with lock:
            active -= 1
            if len(calls) == 2:
                finished.set()

    coordinator = GroupChatCoordinator(store, lambda _params: "unused", submit, lambda _sid: True, lambda *_: True)
    coordinator.send(room["id"], "first")
    assert first_started.wait(timeout=2)
    coordinator.send(room["id"], "second")
    time.sleep(0.05)
    assert calls == [("first", 1)]
    release_first.set()
    assert finished.wait(timeout=2)

    assert calls == [("first", 1), ("second", 1)]
    assistants = [message for message in store.messages(room["id"]) if message["role"] == "assistant"]
    assert [(message["id"], message["content"]) for message in assistants] == [
        ("message-first", "first"), ("message-second", "second")
    ]


def test_cursor_advances_to_turn_start_boundary_not_own_later_completion(tmp_path):
    store = GroupChatStore(tmp_path / "groups.sqlite3")
    room = store.create_room("Cursor", [{"profile": "a", "name": "A", "session_id": "sid-a"}])
    started = threading.Event()
    release = threading.Event()
    done = threading.Event()
    prompts = []

    def submit(_sid, text, emit):
        prompts.append(text)
        if len(prompts) == 1:
            started.set()
            assert release.wait(timeout=2)
            emit("message.complete", {"text": "a-first"})
        else:
            emit("message.complete", {"text": "a-second"})
            done.set()

    coordinator = GroupChatCoordinator(store, lambda _params: "unused", submit, lambda _sid: True, lambda *_: True)
    coordinator.send(room["id"], "first")
    assert started.wait(timeout=2)
    peer = store.append_event(room["id"], "message.complete", {"text": "peer-first"}, "b")
    release.set()
    deadline = time.time() + 2
    while store.projection_cursor(room["id"], "a") == 0 and time.time() < deadline:
        time.sleep(0.01)

    assert store.projection_cursor(room["id"], "a") < peer["seq"]
    coordinator.send(room["id"], "second")
    assert done.wait(timeout=2)
    assert "peer-first" in prompts[1]


def test_agent_error_and_tool_lifecycle_reconstruct_after_reopen(tmp_path):
    db_path = tmp_path / "groups.sqlite3"
    store = GroupChatStore(db_path)
    room = store.create_room("Durable", [{"profile": "worker", "name": "Worker"}])
    store.append_event(room["id"], "message.start", {"message_id": "m1"}, "worker")
    store.append_event(room["id"], "tool.start", {
        "message_id": "m1", "tool_id": "t1", "name": "terminal", "context": "pytest"
    }, "worker")
    store.append_event(room["id"], "tool.complete", {
        "message_id": "m1", "tool_id": "t1", "name": "terminal", "result": {"ok": True}
    }, "worker")
    store.append_event(room["id"], "agent.error", {
        "message_id": "m1", "message": "provider failed"
    }, "worker")
    store.append_event(room["id"], "message.start", {"message_id": "m2"}, "worker")
    store.append_event(room["id"], "approval.request", {
        "message_id": "m2", "session_id": "runtime-worker",
        "command": "rm file", "description": "Delete file",
        "choices": ["once", "deny"], "allow_permanent": False,
        "smart_denied": True,
    }, "worker")
    live = store.messages(room["id"])
    store.close()

    reopened = GroupChatStore(db_path).messages(room["id"])
    assert reopened == live
    error_message = reopened[-2]
    assert error_message["status"] == "error"
    assert error_message["content"] == "provider failed"
    assert error_message["tools"] == [{
        "tool_id": "t1", "name": "terminal", "context": "pytest",
        "status": "complete", "result": {"ok": True},
    }]
    approval_message = reopened[-1]
    assert approval_message["status"] == "approval"
    assert approval_message["runtime_session_id"] == "runtime-worker"
    assert approval_message["approval"] == {
        "command": "rm file", "description": "Delete file",
        "choices": ["once", "deny"], "allow_permanent": False,
        "smart_denied": True,
    }
