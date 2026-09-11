"""Regression contracts for TUI rewind of live compaction carriers."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from agent.context_compressor import (
    HISTORICAL_TASK_HEADING,
    SUMMARY_PREFIX,
    _SUMMARY_END_MARKER,
)
from hermes_state import SessionDB
from tui_gateway import server


def _composite_carrier() -> dict:
    return {
        "role": "user",
        "content": (
            f"{SUMMARY_PREFIX}\n{HISTORICAL_TASK_HEADING}\nold task\n\n"
            f"{_SUMMARY_END_MARKER}\n\nREAL ASK"
        ),
    }


@pytest.fixture()
def carrier_session(tmp_path):
    old_db = server._db
    db = SessionDB(db_path=tmp_path / "state.db")
    installed_ids: list[str] = []

    def install(history: list[dict]):
        sid = f"carrier-sid-{len(installed_ids)}"
        session_key = f"carrier-session-{len(installed_ids)}"
        installed_ids.append(sid)
        db.create_session(session_key, source="tui")
        for message in history:
            db.append_message(
                session_key,
                message["role"],
                message.get("content"),
            )
        durable = db.get_messages_as_conversation(session_key)
        agent = SimpleNamespace(
            _session_messages=list(durable),
            _last_flushed_db_idx=len(durable),
            _db_flush_scan_prefix=list(durable),
        )
        session = {
            "agent": agent,
            "attached_images": [],
            "history": list(durable),
            "history_lock": threading.Lock(),
            "history_version": 0,
            "running": False,
            "session_key": session_key,
        }
        server._sessions[sid] = session
        return sid, session_key, session

    server._db = db
    yield db, install
    for sid in installed_ids:
        server._sessions.pop(sid, None)
    server._db = old_db
    db.close()


def _dispatch(sid: str, name: str) -> dict:
    return server._methods["command.dispatch"](
        "request-id",
        {"session_id": sid, "name": name, "arg": ""},
    )


def _session_undo(sid: str) -> dict:
    return server._methods["session.undo"](
        "request-id",
        {"session_id": sid},
    )


def _assert_scaffold_preserved(
    db: SessionDB,
    session_key: str,
    session: dict,
    *,
    prefix_len: int = 0,
) -> None:
    active = db.get_messages_as_conversation(session_key, include_row_ids=True)
    assert len(active) == prefix_len + 1
    scaffold = active[prefix_len]
    assert scaffold["role"] == "user"
    assert scaffold["display_kind"] == "hidden"
    assert SUMMARY_PREFIX in scaffold["content"]
    assert "REAL ASK" not in scaffold["content"]
    assert session["history"][prefix_len]["content"] == scaffold["content"]
    assert session["history"][prefix_len]["display_kind"] == "hidden"


def test_retry_selects_the_live_ask_inside_a_force_user_leading_carrier(
    carrier_session,
):
    db, install = carrier_session
    sid, session_key, session = install(
        [_composite_carrier(), {"role": "assistant", "content": "failed"}]
    )

    response = _dispatch(sid, "retry")

    assert response["result"] == {"type": "send", "message": "REAL ASK"}
    _assert_scaffold_preserved(db, session_key, session)


@pytest.mark.parametrize("command", ["retry", "undo"])
def test_rewind_matches_cold_sanitized_carrier_to_unchanged_warm_ask(
    carrier_session, command
):
    db, install = carrier_session
    carrier = _composite_carrier()
    carrier["content"] = carrier["content"].replace(
        "REAL ASK",
        "  REAL ASK\n\n<memory-context>\nprivate\n</memory-context>  ",
    )
    sid, session_key, session = install(
        [carrier, {"role": "assistant", "content": "failed"}]
    )
    db._conn.execute(
        "UPDATE messages SET api_content = ? "
        "WHERE session_id = ? AND role = 'user'",
        (carrier["content"], session_key),
    )
    db._conn.commit()
    # A live agent still has the raw wire form while a cold DB projection has
    # already applied the role-aware sanitize_context(...).strip() rule and
    # retained the raw provider wire in api_content.
    session["history"][0] = carrier.copy()
    session["agent"]._session_messages = list(session["history"])

    response = _dispatch(sid, command)

    assert response["result"]["message"] == "REAL ASK"
    _assert_scaffold_preserved(db, session_key, session)


def test_retry_fails_closed_when_transcript_changes_after_snapshot(
    carrier_session, monkeypatch
):
    db, install = carrier_session
    sid, session_key, session = install(
        [_composite_carrier(), {"role": "assistant", "content": "failed"}]
    )
    sibling = SessionDB(db_path=db.db_path)
    original_rewind = db.rewind_to_message

    def _append_then_rewind(*args, **kwargs):
        sibling.append_message(session_key, "assistant", "concurrent tail")
        return original_rewind(*args, **kwargs)

    monkeypatch.setattr(db, "rewind_to_message", _append_then_rewind)
    before_history = [dict(message) for message in session["history"]]

    response = _dispatch(sid, "retry")

    assert response["error"]["code"] == 5008
    assert "active transcript changed" in response["error"]["message"]
    assert session["history"] == before_history
    rows = db._conn.execute(
        "SELECT content, active, display_kind FROM messages "
        "WHERE session_id = ? ORDER BY id",
        (session_key,),
    ).fetchall()
    assert [tuple(row) for row in rows] == [
        (_composite_carrier()["content"], 1, None),
        ("failed", 1, None),
        ("concurrent tail", 1, None),
    ]
    sibling.close()


@pytest.mark.parametrize("command", ["retry", "undo"])
def test_rewind_allows_database_only_reaction_metadata_change(
    carrier_session, command
):
    db, install = carrier_session
    sid, session_key, session = install(
        [
            {"role": "user", "content": "OLDER ASK"},
            {"role": "assistant", "content": "older answer"},
            _composite_carrier(),
            {"role": "assistant", "content": "failed"},
        ]
    )
    older_answer = next(
        row
        for row in db.get_messages(session_key)
        if row["role"] == "assistant" and row["content"] == "older answer"
    )
    assert db.set_message_reaction(
        session_key, older_answer["id"], "👍", author="user"
    )

    response = _dispatch(sid, command)

    assert response["result"]["message"] == "REAL ASK"
    _assert_scaffold_preserved(db, session_key, session, prefix_len=2)


def test_retry_ignores_buried_ephemeral_scaffolding_missing_from_db(
    carrier_session,
):
    db, install = carrier_session
    sid, session_key, session = install(
        [
            {"role": "user", "content": "OLDER ASK"},
            {"role": "assistant", "content": "older answer"},
            _composite_carrier(),
            {"role": "assistant", "content": "failed"},
        ]
    )
    session["history"].insert(
        2,
        {
            "role": "user",
            "content": "internal recovery nudge",
            "_dropped_toolcall_nudge": True,
        },
    )
    session["agent"]._session_messages = list(session["history"])

    response = _dispatch(sid, "retry")

    assert response["result"] == {"type": "send", "message": "REAL ASK"}
    _assert_scaffold_preserved(db, session_key, session, prefix_len=2)


def test_retry_drops_buried_ephemeral_scaffolding_from_the_warm_prefix(
    carrier_session,
):
    db, install = carrier_session
    sid, session_key, session = install(
        [
            {"role": "user", "content": "OLDER ASK"},
            {"role": "assistant", "content": "candidate answer"},
            {"role": "assistant", "content": "verified answer"},
            _composite_carrier(),
            {"role": "assistant", "content": "failed"},
        ]
    )
    session["history"].insert(
        2,
        {
            "role": "user",
            "content": "[System: verify before stopping]",
            "_verification_stop_synthetic": True,
        },
    )
    session["agent"]._session_messages = list(session["history"])

    response = _dispatch(sid, "retry")

    assert response["result"] == {"type": "send", "message": "REAL ASK"}
    assert [message.get("content") for message in session["history"][:3]] == [
        "OLDER ASK",
        "candidate answer",
        "verified answer",
    ]
    active = db.get_messages_as_conversation(session_key, include_row_ids=True)
    # Alternation repair is a model/memory projection; the two durable source
    # rows remain independently recoverable ahead of the inserted scaffold.
    assert [message.get("content") for message in active[:3]] == [
        "OLDER ASK",
        "candidate answer",
        "verified answer",
    ]
    assert active[3]["display_kind"] == "hidden"
    assert "REAL ASK" not in active[3]["content"]
    assert session["history"][3]["display_kind"] == "hidden"
    assert "REAL ASK" not in session["history"][3]["content"]


def test_retry_preserves_older_warm_media_while_targeting_plain_ask(
    carrier_session,
):
    db, install = carrier_session
    sid, session_key, session = install(
        [
            {"role": "user", "content": "look\n[screenshot]"},
            {"role": "assistant", "content": "seen"},
            _composite_carrier(),
            {"role": "assistant", "content": "failed"},
        ]
    )
    session["history"][0]["content"] = [
        {"type": "text", "text": "look"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}},
    ]
    session["agent"]._session_messages = list(session["history"])

    response = _dispatch(sid, "retry")

    assert response["result"] == {"type": "send", "message": "REAL ASK"}
    assert isinstance(session["history"][0]["content"], list)
    _assert_scaffold_preserved(db, session_key, session, prefix_len=2)


def test_undo_targets_the_composite_carrier_not_an_older_user_turn(
    carrier_session,
):
    db, install = carrier_session
    sid, session_key, session = install(
        [
            {"role": "user", "content": "OLDER ASK"},
            {"role": "assistant", "content": "older answer"},
            _composite_carrier(),
            {"role": "assistant", "content": "failed"},
        ]
    )

    response = _dispatch(sid, "undo")

    assert response["result"]["type"] == "prefill"
    assert response["result"]["message"] == "REAL ASK"
    active = db.get_messages_as_conversation(session_key, include_row_ids=True)
    assert [message.get("content") for message in active[:2]] == [
        "OLDER ASK",
        "older answer",
    ]
    _assert_scaffold_preserved(db, session_key, session, prefix_len=2)


def test_undo_rewinds_media_placeholder_without_treating_it_as_retry(
    carrier_session,
):
    db, install = carrier_session
    carrier = _composite_carrier()
    carrier["content"] = carrier["content"].replace(
        "REAL ASK", "look\n[screenshot]"
    )
    sid, session_key, session = install(
        [carrier, {"role": "assistant", "content": "seen"}]
    )

    response = _dispatch(sid, "undo")

    assert response["result"]["type"] == "prefill"
    assert response["result"]["message"] == "look\n[screenshot]"
    active = db.get_messages_as_conversation(session_key)
    assert len(active) == 1
    assert active[0]["display_kind"] == "hidden"
    assert "look\n[screenshot]" not in active[0]["content"]
    assert len(session["history"]) == 1
    assert session["history"][0]["content"] == active[0]["content"]
    assert session["history"][0]["display_kind"] == "hidden"


def test_session_undo_preserves_the_composite_carriers_scaffold(carrier_session):
    db, install = carrier_session
    sid, session_key, session = install(
        [_composite_carrier(), {"role": "assistant", "content": "answer"}]
    )

    response = _session_undo(sid)

    assert response["result"]["removed"] == 2
    _assert_scaffold_preserved(db, session_key, session)


def test_history_projection_unwraps_composite_and_hides_sole_handoff():
    composite = {**_composite_carrier(), "_row_id": 7}
    sole_handoff = {
        **composite,
        "content": composite["content"].split("\n\nREAL ASK", 1)[0],
    }

    assert server._history_to_messages(
        [composite, {"role": "user", "content": "newer ask", "_row_id": 9}]
    ) == [
        {"role": "user", "text": "REAL ASK", "row_id": 7},
        {"role": "user", "text": "newer ask", "row_id": 9},
    ]
    assert server._history_to_messages([sole_handoff]) == []


def test_retry_preserves_literal_media_like_text(carrier_session):
    db, install = carrier_session
    carrier = _composite_carrier()
    carrier["content"] = carrier["content"].replace(
        "REAL ASK", "inspect [image|ybres:RID]"
    )
    sid, session_key, session = install(
        [carrier, {"role": "assistant", "content": "failed"}]
    )
    response = _dispatch(sid, "retry")

    assert response["result"] == {
        "type": "send",
        "message": "inspect [image|ybres:RID]",
    }
    _assert_scaffold_preserved(db, session_key, session)


def test_retry_rejects_durable_media_before_rewind_when_warm_view_is_text(
    carrier_session,
):
    db, install = carrier_session
    carrier = _composite_carrier()
    handoff = carrier["content"].rsplit("\n\nREAL ASK", 1)[0]
    durable_carrier = carrier.copy()
    durable_carrier["content"] = [
        {"type": "text", "text": handoff},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}},
    ]
    sid, session_key, session = install(
        [durable_carrier, {"role": "assistant", "content": "failed"}]
    )
    # The warm projection can be a degraded text-only view that compares equal
    # to the durable media payload. Durable retryability must still be checked
    # before the physical carrier and tail are archived.
    warm_carrier = carrier.copy()
    warm_carrier["content"] = handoff + "\n\n[screenshot]"
    session["history"][0] = warm_carrier
    session["agent"]._session_messages = list(session["history"])
    before_history = [message.copy() for message in session["history"]]

    response = _dispatch(sid, "retry")

    assert response["error"]["code"] == 4018
    assert session["history"] == before_history
    assert len(db.get_messages_as_conversation(session_key)) == 2


def test_retry_rejects_pending_attachments_before_mutating_history(carrier_session):
    db, install = carrier_session
    sid, session_key, session = install(
        [_composite_carrier(), {"role": "assistant", "content": "failed"}]
    )
    session["attached_images"] = ["/tmp/pending.png"]
    before_memory = list(session["history"])

    response = _dispatch(sid, "retry")

    assert response["error"]["code"] == 4018
    assert session["history"] == before_memory
    assert len(db.get_messages_as_conversation(session_key)) == 2

def test_prompt_row_id_rewind_preserves_scaffold_before_regeneration(
    carrier_session, monkeypatch
):
    db, install = carrier_session
    sid, session_key, session = install(
        [_composite_carrier(), {"role": "assistant", "content": "failed"}]
    )
    target_row_id = db.get_messages_as_conversation(
        session_key, include_row_ids=True
    )[0]["_row_id"]
    seen = {}

    class _Agent:
        _session_messages = list(session["history"])
        _last_flushed_db_idx = len(_session_messages)
        _db_flush_scan_prefix = list(_session_messages)

        def run_conversation(
            self, prompt, conversation_history=None, stream_callback=None, **_kwargs
        ):
            seen["prompt"] = prompt
            seen["history"] = list(conversation_history or [])
            return {
                "final_response": "regenerated",
                "messages": [
                    *(conversation_history or []),
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "regenerated"},
                ],
            }

    class _ImmediateThread:
        def __init__(self, target=None, daemon=None):
            self._target = target

        def start(self):
            self._target()

    session["agent"] = _Agent()
    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(server, "_get_usage", lambda _agent: {})
    monkeypatch.setattr(server, "render_message", lambda *_args: "")
    monkeypatch.setattr(server, "_emit", lambda *_args: None)

    response = server._methods["prompt.submit"](
        "request-id",
        {
            "session_id": sid,
            "text": "EDITED ASK",
            "truncate_before_row_id": target_row_id,
            "truncate_before_user_ordinal": 0,
            "confirm_truncate": True,
        },
    )

    assert response["result"]["status"] == "streaming"
    assert seen["prompt"] == "EDITED ASK"
    assert len(seen["history"]) == 1
    assert seen["history"][0]["display_kind"] == "hidden"
    assert "REAL ASK" not in seen["history"][0]["content"]
    active = db.get_messages_as_conversation(session_key, include_row_ids=True)
    assert len(active) == 1
    assert active[0]["display_kind"] == "hidden"
