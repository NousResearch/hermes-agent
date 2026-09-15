"""session.branch full mode must keep tool_calls / tool results (issue #106501)."""

from __future__ import annotations

import threading

from tui_gateway import server


TOOL_HISTORY = [
    {"role": "user", "content": "run ls"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "terminal", "arguments": '{"command":"ls"}'},
            }
        ],
    },
    {
        "role": "tool",
        "content": "file.txt",
        "tool_call_id": "call-1",
        "tool_name": "terminal",
    },
    {"role": "assistant", "content": "done"},
]


def _roles(messages):
    return [message.get("role") for message in messages]


def _assistant_tool_calls(messages):
    return [
        message.get("tool_calls")
        for message in messages
        if message.get("role") == "assistant" and message.get("tool_calls")
    ]


def _branch(monkeypatch, tmp_path, *, params, history=None):
    profile_home = tmp_path / "profiles" / "mlperf"
    profile_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    seen: dict = {"msgs": [], "live_history": None}

    class LaunchDB:
        def get_session_title(self, _key):
            return "launch"

    class ProfileDB:
        def __init__(self, db_path=None):
            seen["db_path"] = db_path

        def get_session_title(self, _key):
            return "parent"

        def get_next_title_in_lineage(self, current):
            return f"{current} (branch)"

        def create_session(self, _new_key, **_kwargs):
            return None

        def append_message(self, **kwargs):
            seen["msgs"].append(kwargs)

        def append_messages_batch(self, session_id, messages, **kwargs):
            for message in messages:
                seen["msgs"].append(dict(message, session_id=session_id))
            return list(range(1, len(messages) + 1))

        def set_session_title(self, _key, _title):
            return True

        def get_session(self, key):
            return {"id": key, "cwd": str(tmp_path)}

        def update_session_cwd(self, *args, **kwargs):
            return None

        def close(self):
            return None

    class FakeAgent:
        model = "test-model"
        session_id = None

    def fake_init(_sid, _key, _agent, seeded, **_kwargs):
        seen["live_history"] = [dict(message) for message in seeded]

    parent = {
        "session_key": "parent-key",
        "history": list(history if history is not None else TOOL_HISTORY),
        "history_lock": threading.Lock(),
        "running": False,
        "cols": 80,
        "profile_home": str(profile_home),
        "source": "tui",
        "agent": FakeAgent(),
        "created_at": 1.0,
        "last_active": 1.0,
        "cwd": str(tmp_path),
    }
    server._sessions["parent"] = parent
    monkeypatch.setattr(server, "_get_db", lambda: LaunchDB())
    monkeypatch.setattr("hermes_state_registry.acquire", ProfileDB)
    monkeypatch.setattr(server, "_claim_active_session_slot", lambda *args, **kwargs: (None, None))
    monkeypatch.setattr(server, "_make_agent", lambda *args, **kwargs: FakeAgent())
    monkeypatch.setattr(server, "_set_session_context", lambda *args, **kwargs: {})
    monkeypatch.setattr(server, "_clear_session_context", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "_resolve_model", lambda: "test-model")
    monkeypatch.setattr(server, "_session_cwd", lambda _session: str(tmp_path))
    monkeypatch.setattr(server, "_register_session_cwd", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "_attach_worker", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "_init_session", fake_init)
    monkeypatch.setattr(server, "_wait_agent", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_start_agent_build", lambda *_a, **_k: None)

    try:
        response = server.handle_request(
            {
                "id": "1",
                "method": "session.branch",
                "params": {"session_id": "parent", **params},
            }
        )
        assert "result" in response, response
        return response["result"], seen
    finally:
        for key in list(server._sessions):
            server._sessions.pop(key, None)


def test_session_branch_full_mode_preserves_tool_rows(monkeypatch, tmp_path):
    result, seen = _branch(monkeypatch, tmp_path, params={"branch_mode": "full"})

    persisted_roles = _roles(seen["msgs"])
    live_roles = _roles(seen["live_history"] or [])
    response_roles = [message.get("role") for message in result["messages"]]

    assert "tool" in persisted_roles
    assert "tool" in live_roles
    assert "tool" in response_roles
    assert _assistant_tool_calls(seen["msgs"])
    assert _assistant_tool_calls(seen["live_history"] or [])
    assert any(message.get("tool_call_id") == "call-1" for message in seen["msgs"] if message.get("role") == "tool")

    # CONTROL: spine (legacy) still strips tool rows.
    spine_result, spine_seen = _branch(monkeypatch, tmp_path, params={"branch_mode": "spine"})
    assert "tool" not in _roles(spine_seen["msgs"])
    assert "tool" not in _roles(spine_seen["live_history"] or [])
    assert "tool" not in [message.get("role") for message in spine_result["messages"]]
    assert not _assistant_tool_calls(spine_seen["msgs"])


def test_session_branch_full_mode_from_config(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"session": {"branch_mode": "full"}},
    )
    result, seen = _branch(monkeypatch, tmp_path, params={})
    assert "tool" in _roles(seen["msgs"])
    assert "tool" in [message.get("role") for message in result["messages"]]
    assert _assistant_tool_calls(seen["live_history"] or [])


def test_session_branch_unknown_mode_fail_open_spine(monkeypatch, tmp_path):
    result, seen = _branch(monkeypatch, tmp_path, params={"branch_mode": "banana"})
    assert "tool" not in _roles(seen["msgs"])
    assert "tool" not in [message.get("role") for message in result["messages"]]


def test_session_branch_config_error_fail_open_spine(monkeypatch, tmp_path):
    def boom():
        raise RuntimeError("config unreadable")

    monkeypatch.setattr("hermes_cli.config.load_config", boom)
    _result, seen = _branch(monkeypatch, tmp_path, params={})
    assert "tool" not in _roles(seen["msgs"])


def test_session_branch_full_mode_honors_count(monkeypatch, tmp_path):
    # count=2 would cut after assistant-with-tool_calls; shrink to a complete prefix.
    _result, seen = _branch(monkeypatch, tmp_path, params={"branch_mode": "full", "count": 2})
    assert _roles(seen["msgs"]) == ["user"]
    assert not _assistant_tool_calls(seen["msgs"])


def test_session_branch_full_mode_count_keeps_complete_tool_pair(monkeypatch, tmp_path):
    _result, seen = _branch(monkeypatch, tmp_path, params={"branch_mode": "full", "count": 3})
    assert _roles(seen["msgs"]) == ["user", "assistant", "tool"]
    assert seen["msgs"][1].get("tool_calls")
    assert seen["msgs"][2].get("tool_call_id") == "call-1"


def test_session_branch_spine_strips_tool_calls_from_text_assistant(monkeypatch, tmp_path):
    history = [
        {"role": "user", "content": "run ls"},
        {
            "role": "assistant",
            "content": "running",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": '{"command":"ls"}'},
                }
            ],
        },
        {
            "role": "tool",
            "content": "file.txt",
            "tool_call_id": "call-1",
            "tool_name": "terminal",
        },
        {"role": "assistant", "content": "done"},
    ]
    _result, seen = _branch(monkeypatch, tmp_path, params={"branch_mode": "spine"}, history=history)
    assert _roles(seen["msgs"]) == ["user", "assistant", "assistant"]
    assert not _assistant_tool_calls(seen["msgs"])
    assert not _assistant_tool_calls(seen["live_history"] or [])
    assert all(not message.get("tool_calls") for message in seen["msgs"])


def test_seed_branch_row_persists_tool_bindings(monkeypatch, tmp_path):
    seen: list = []

    class SeedDB:
        def create_session(self, *_args, **_kwargs):
            return None

        def get_session_title(self, _key):
            return "parent"

        def get_next_title_in_lineage(self, current):
            return f"{current} (branch)"

        def append_messages_batch(self, _session_id, messages, **_kwargs):
            seen.extend(messages)
            return list(range(1, len(messages) + 1))

        def set_session_title(self, *_args, **_kwargs):
            return True

    class _DbCtx:
        def __enter__(self):
            return SeedDB()

        def __exit__(self, *_exc):
            return False

    monkeypatch.setattr(server, "_session_db", lambda _record: _DbCtx())
    monkeypatch.setattr(server, "_resolve_model", lambda: "test-model")
    monkeypatch.setattr(server, "_current_profile_name", lambda: "default")
    record = {"cwd": str(tmp_path), "pending_title": "branch"}
    server._seed_branch_row(record, "child-key", "parent-key", list(TOOL_HISTORY), "desktop", tmp_path)
    assert _roles(seen) == ["user", "assistant", "tool", "assistant"]
    assert seen[1].get("tool_calls")
    assert seen[2].get("tool_call_id") == "call-1"
    assert seen[2].get("tool_name") == "terminal"
    assert record.get("_branch_seed_persisted") is True


def test_truncate_full_branch_history_stops_before_dangling_tool_calls():
    history = [
        {"role": "user", "content": "run"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "a", "function": {"name": "terminal"}},
                {"id": "b", "function": {"name": "terminal"}},
            ],
        },
        {"role": "tool", "content": "a-out", "tool_call_id": "a"},
        {"role": "tool", "content": "b-out", "tool_call_id": "b"},
        {"role": "assistant", "content": "done"},
    ]
    assert [m["role"] for m in server._truncate_full_branch_history(history, 3)] == ["user"]
    assert [m["role"] for m in server._truncate_full_branch_history(history, 4)] == [
        "user",
        "assistant",
        "tool",
        "tool",
    ]
    assert len(server._truncate_full_branch_history(history, 5)) == 5


def test_coerce_seed_history_keeps_tool_bindings():
    history = server._coerce_seed_history(
        [
            {"role": "user", "content": "run ls"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call-1", "function": {"name": "terminal"}}],
            },
            {"role": "tool", "content": "ok", "tool_call_id": "call-1", "tool_name": "terminal"},
            {"role": "system", "content": "   "},
        ]
    )
    assert _roles(history) == ["user", "assistant", "tool"]
    assert history[1]["tool_calls"][0]["id"] == "call-1"
    assert history[2]["tool_call_id"] == "call-1"
