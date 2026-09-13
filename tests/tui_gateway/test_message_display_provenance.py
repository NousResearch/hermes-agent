"""Display provenance agrees with canonical turns after real DB replay."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import threading

import pytest


def _examples():
    from agent.context_compressor import (
        HISTORICAL_TASK_HEADING,
        SUMMARY_PREFIX,
        _SUMMARY_END_MARKER,
    )
    from tools.todo_tool import TODO_INJECTION_HEADER

    handoff = (
        f"{SUMMARY_PREFIX}\n{HISTORICAL_TASK_HEADING}\nold task\n\n"
        f"{_SUMMARY_END_MARKER}"
    )
    todo = f"{TODO_INJECTION_HEADER}\n- [>] current task"
    return [
        {"role": "user", "content": "repeat this"},
        {"role": "assistant", "content": "finished answer"},
        {
            "role": "user", "content": "runtime wake with no sentinel",
            "display_kind": "internal_notification",
        },
        {"role": "user", "content": todo},
        {
            "role": "user",
            "content": "[IMPORTANT: Background process proc_1 completed normally.]",
        },
        {"role": "user", "content": f"Keep working\n\n{todo}"},
        {
            "role": "user", "content": f"{handoff}\n\nREAL ASK",
            "display_kind": "hidden",
        },
        {"role": "user", "content": handoff, "display_kind": "hidden"},
        # The canonical legacy detector requires a newline after this header.
        # Quoting it in ordinary prose is human input, despite the UI prefix.
        {"role": "user", "content": f"{TODO_INJECTION_HEADER} explain this phrase"},
        {"role": "user", "content": "repeat this"},
        {"role": "assistant", "content": "second finished answer"},
    ]


def test_db_rest_and_rpc_keep_canonical_provenance_without_changing_history(
    tmp_path, monkeypatch,
):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    from agent.context_compressor import user_originated_turn_view
    from fastapi.testclient import TestClient
    from hermes_cli import web_server
    import hermes_state
    from tui_gateway import server

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", home / "state.db")
    db = hermes_state.SessionDB(db_path=home / "state.db")
    key = "display-provenance"
    try:
        db.create_session(key, source="desktop")
        source = _examples()
        original_source = copy.deepcopy(source)
        for index, message in enumerate(source, 1):
            db.append_message(
                key, message["role"], message["content"], timestamp=float(index),
                display_kind=message.get("display_kind"),
            )
        history = db.get_messages_as_conversation(key, include_row_ids=True)
        before_history = copy.deepcopy(history)
        before_rows = db.get_messages(key)
        monkeypatch.setitem(server._sessions, key, {
            "session_key": key,
            "history": history,
            "history_lock": threading.Lock(),
            "profile_home": str(home),
            "profile_incarnation": server._capture_profile_incarnation(home),
        })

        client = TestClient(web_server.app)
        client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
        response = client.get(f"/api/sessions/{key}/messages")
        assert response.status_code == 200
        rest_rows = response.json()["messages"]
        rpc = server._methods["session.history"]("provenance", {"session_id": key})
        assert "error" not in rpc
        rpc_rows = rpc["result"]["messages"]

        expected = {
            row["_row_id"]: user_originated_turn_view(row) is not None
            for row in history if row["role"] == "user"
        }
        # Typed and legacy notices remain visible, but cannot become human
        # ordinals. Pure hidden handoffs stay hidden on their existing path.
        visible_rest = [row for row in rest_rows if row.get("display_kind") != "hidden"]
        rest_origins = {
            row["id"]: row["user_originated"]
            for row in visible_rest if row["role"] == "user"
        }
        rpc_origins = {
            row["row_id"]: row["user_originated"]
            for row in rpc_rows if row["role"] == "user"
        }
        assert rest_origins == rpc_origins
        assert rest_origins == {row_id: expected[row_id] for row_id in rest_origins}
        assert {row_id for row_id, human in expected.items() if human} <= rest_origins.keys()
        assert sum(row.get("content") == "repeat this" for row in rest_rows) == 2
        assert any(row.get("text") == "runtime wake with no sentinel" for row in rpc_rows)
        carrier = next(row for row in rest_rows if "REAL ASK" in str(row.get("content")))
        assert carrier["display_content"] == "REAL ASK"
        assert carrier["user_originated"] is True
        assert carrier["content"] == source[6]["content"]

        assert source == original_source
        assert history == before_history
        assert db.get_messages(key) == before_rows
        assert all("user_originated" not in row for row in history)
        # A throwaway transport artifact lets the same real responses be
        # consumed by the focused Desktop hydration integration check.
        (tmp_path / "display-provenance.json").write_text(
            json.dumps({"rest": rest_rows, "rpc": rpc_rows}),
            encoding="utf-8",
        )
    finally:
        db.close()


@pytest.mark.parametrize("index", [0, 2, 3, 5, 6, 7, 8])
def test_inflight_uses_history_provenance_and_keeps_original_user_text(index):
    from agent.compaction_display import project_compaction_message_for_display
    from tui_gateway import server

    message = _examples()[index]
    original = copy.deepcopy(message)
    display = project_compaction_message_for_display(message)
    session = {}
    server._start_inflight_turn(
        session, message["content"], display_kind=message.get("display_kind"),
    )
    server._append_inflight_delta(session, "partial answer")
    snapshot = server._inflight_snapshot(session)

    assert snapshot["user"] == message["content"]
    assert snapshot["user_originated"] is (display is not None and display["user_originated"])
    assert snapshot["assistant"] == "partial answer"
    assert message == original
    if display is None:
        assert snapshot["display_kind"] == "hidden"


def test_legacy_inflight_omits_provenance_instead_of_claiming_human_origin():
    from tui_gateway import server

    snapshot = server._inflight_snapshot({
        "inflight_turn": {"user": "old prompt", "assistant": "", "streaming": True},
    })
    assert "user_originated" not in snapshot
