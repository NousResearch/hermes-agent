"""Session-scoped transcript ops must resolve against the session's own DB.

App-global remote mode gives a session its own profile (``profile_home`` on
the session dict, see ``session.create``), and that profile keeps its own
``state.db``.  ``_get_db()`` is the *launch* profile's handle, so any
session-scoped read or write that reaches for it operates on the wrong
database: writes land in a foreign profile under this session's id, and reads
come back empty because the row simply is not there.

``_session_db(session)`` is the profile-aware resolver that already exists for
exactly this (``tui_gateway/server.py``): the profile's ``state.db`` when
``session['profile_home']`` is set, otherwise the shared launch handle.

Every test here drives the real JSON-RPC entry point
(``server.handle_request``).  Handler bodies live in ``tui_gateway/methods_*``
but are rebound onto ``server.py``'s globals by
``method_ctx.HandlerRegistry.install()``, so calling a handler function
directly would bypass the path the gateway actually executes.
"""

from __future__ import annotations

import importlib
import threading
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_state import SessionDB

SESSION_ID = "sid-profile"
SESSION_KEY = "tui-profile-1"


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    yield home


@pytest.fixture()
def server(hermes_home):
    # Mocks are scoped to the initial import only (see
    # tests/tui_gateway/test_protocol.py for the rationale).
    with patch.dict(
        "sys.modules",
        {
            "hermes_cli.env_loader": MagicMock(),
            "hermes_cli.banner": MagicMock(),
        },
    ):
        mod = importlib.import_module("tui_gateway.server")

    methods = dict(mod._methods)
    yield mod
    # Restore in place instead of clear+reload: importlib.reload re-registers
    # atexit hooks and re-captures module-level paths against this test's
    # soon-deleted tmpdir (see tests/tui_gateway/test_undo_command.py).
    mod._methods.clear()
    mod._methods.update(methods)
    mod._sessions.clear()
    mod._pending.clear()
    mod._answers.clear()
    mod._db = None


@pytest.fixture()
def launch_db(server, hermes_home):
    """The launch profile's state.db, wired in as the ``_get_db()`` handle."""
    db = SessionDB(db_path=hermes_home / "state.db")
    server._db = db
    return db


@pytest.fixture()
def profile_db(tmp_path):
    """A second, non-launch profile's state.db."""
    profile_home = tmp_path / "profiles" / "work"
    profile_home.mkdir(parents=True)
    return profile_home, SessionDB(db_path=profile_home / "state.db")


def _seed(db, turns=3, *, _capture_user_row_ids=None):
    db.create_session(SESSION_KEY, source="tui")
    for i in range(1, turns + 1):
        uid = db.append_message(SESSION_KEY, "user", f"question {i}")
        if _capture_user_row_ids is not None:
            _capture_user_row_ids.append(uid)
        db.append_message(SESSION_KEY, "assistant", f"answer {i}")
    return db.get_messages_as_conversation(SESSION_KEY)


def _register(server, history, *, profile_home=None):
    # SimpleNamespace, not MagicMock: the usage snapshot compares attributes
    # numerically, and auto-created mock attributes are not orderable.
    agent = types.SimpleNamespace(
        _memory_manager=MagicMock(),
        _last_flushed_db_idx=len(history),
        model="test-model",
    )
    session = {
        "session_key": SESSION_KEY,
        "history": list(history),
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "agent": agent,
        "attached_images": [],
        "image_counter": 0,
        "cols": 120,
        # The cap slot is claimed on the first real turn; pre-claim it so the
        # test exercises the transcript path rather than the lease allocator.
        "active_session_lease": object(),
    }
    if profile_home is not None:
        session["profile_home"] = str(profile_home)
    server._sessions[SESSION_ID] = session
    return session


def _rpc(server, method, params):
    return server.handle_request({"id": "1", "method": method, "params": params})


def _texts(rows):
    out = []
    for row in rows:
        content = row.get("content")
        if isinstance(content, list):
            content = "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        out.append(str(content or ""))
    return out


# ---------------------------------------------------------------------------
# /undo — command.dispatch
# ---------------------------------------------------------------------------


def test_undo_rewinds_the_profile_transcript(server, launch_db, profile_db):
    """/undo on a profile session must read and rewind that profile's db.

    ``list_recent_user_messages`` is session-id scoped, so against the launch
    handle it finds nothing and /undo fails closed with 4018 — the command is
    unusable for the entire session in app-global remote mode.
    """
    profile_home, pdb = profile_db
    history = _seed(pdb)
    _register(server, history, profile_home=profile_home)

    resp = _rpc(
        server,
        "command.dispatch",
        {"session_id": SESSION_ID, "name": "undo", "arg": ""},
    )

    assert not resp.get("error"), f"/undo failed: {resp.get('error')}"
    result = resp["result"]
    assert result["type"] == "prefill"
    assert result["message"] == "question 3"
    # The rewind is durable in the profile's own db, not the launch one.
    assert _texts(pdb.get_messages_as_conversation(SESSION_KEY)) == [
        "question 1",
        "answer 1",
        "question 2",
        "answer 2",
    ]
    assert launch_db.get_messages_as_conversation(SESSION_KEY) == []


def test_undo_still_uses_the_shared_handle_without_a_profile(server, launch_db):
    """A launch-profile session keeps borrowing the shared ``_get_db()`` handle."""
    history = _seed(launch_db)
    _register(server, history)

    resp = _rpc(
        server,
        "command.dispatch",
        {"session_id": SESSION_ID, "name": "undo", "arg": ""},
    )

    assert not resp.get("error"), f"/undo failed: {resp.get('error')}"
    assert resp["result"]["message"] == "question 3"
    assert _texts(launch_db.get_messages_as_conversation(SESSION_KEY)) == [
        "question 1",
        "answer 1",
        "question 2",
        "answer 2",
    ]


# ---------------------------------------------------------------------------
# edit/resend truncation — prompt.submit
# ---------------------------------------------------------------------------


def _stop_after_truncate(server, monkeypatch):
    """Return the RPC right after the truncate branch, before the agent turn.

    Isolated turns hand the prompt to the compute host and return, which is
    the natural exit closest to the code under test; stubbing the handoff keeps
    the test on the transcript-persistence path instead of running a model.
    """
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda *a, **k: True)
    monkeypatch.setattr(
        server,
        "_submit_prompt_to_compute_host",
        lambda rid, sid, session, text, **_kwargs: server._ok(
            rid, {"status": "streaming"}
        ),
    )


def test_truncation_persists_to_the_profile_db(server, launch_db, profile_db, monkeypatch):
    """An edit/resend must truncate the profile's transcript, not the launch one."""
    profile_home, pdb = profile_db
    user_row_ids: list = []
    history = _seed(pdb, _capture_user_row_ids=user_row_ids)
    _register(server, history, profile_home=profile_home)
    _stop_after_truncate(server, monkeypatch)

    # Row-id addressed, not ordinal-only: current main refuses a bare ordinal
    # for a durable session (truncate_before_row_id required). Target the 2nd
    # user turn's durable row id; keep the matching ordinal as a cross-check.
    resp = _rpc(
        server,
        "prompt.submit",
        {
            "session_id": SESSION_ID,
            "text": "edited question 2",
            "truncate_before_row_id": user_row_ids[1],
            "truncate_before_user_ordinal": 1,
            "confirm_truncate": True,
        },
    )

    assert not resp.get("error"), f"prompt.submit failed: {resp.get('error')}"
    # The undone turns are gone from the profile's own db, so session.resume
    # (which opens the profile db correctly) cannot resurrect them.
    assert _texts(pdb.get_messages_as_conversation(SESSION_KEY)) == [
        "question 1",
        "answer 1",
    ]
    # ...and nothing was copied into a foreign profile under this session id.
    assert launch_db.get_messages_as_conversation(SESSION_KEY) == []


def test_truncation_does_not_copy_rows_into_the_launch_profile(
    server, launch_db, profile_db, monkeypatch
):
    """The launch profile must not receive a copy of a profile session's turns.

    When the launch db happens to hold a row under the same session id, the
    write through the launch handle succeeds instead of failing the foreign-key
    check, so the truncated transcript is inserted into a profile the session
    does not belong to.
    """
    profile_home, pdb = profile_db
    user_row_ids: list = []
    history = _seed(pdb, _capture_user_row_ids=user_row_ids)
    launch_db.create_session(SESSION_KEY, source="unknown")
    _register(server, history, profile_home=profile_home)
    _stop_after_truncate(server, monkeypatch)

    resp = _rpc(
        server,
        "prompt.submit",
        {
            "session_id": SESSION_ID,
            "text": "edited question 2",
            "truncate_before_row_id": user_row_ids[1],
            "truncate_before_user_ordinal": 1,
            "confirm_truncate": True,
        },
    )

    assert not resp.get("error"), f"prompt.submit failed: {resp.get('error')}"
    assert launch_db.get_messages_as_conversation(SESSION_KEY) == []
    assert _texts(pdb.get_messages_as_conversation(SESSION_KEY)) == [
        "question 1",
        "answer 1",
    ]


def test_truncation_surfaces_the_profile_dbs_new_row_ids(
    server, launch_db, profile_db, monkeypatch
):
    """``survivor_user_row_ids`` must carry the profile db's post-rewrite ids.

    ``replace_messages`` re-inserts the surviving prefix as NEW rows and the
    client rebinds its cached stamps from this payload, so the ids have to come
    from the db that actually did the rewrite. Ids minted anywhere else address
    nothing in the profile's transcript, and the next rewind is refused 4018.
    """
    profile_home, pdb = profile_db
    user_row_ids: list = []
    history = _seed(pdb, _capture_user_row_ids=user_row_ids)
    _register(server, history, profile_home=profile_home)
    _stop_after_truncate(server, monkeypatch)

    resp = _rpc(
        server,
        "prompt.submit",
        {
            "session_id": SESSION_ID,
            "text": "edited question 2",
            "truncate_before_row_id": user_row_ids[1],
            "truncate_before_user_ordinal": 1,
            "confirm_truncate": True,
        },
    )

    assert not resp.get("error"), f"prompt.submit failed: {resp.get('error')}"
    surviving = [
        row["_row_id"]
        for row in pdb.get_messages_as_conversation(SESSION_KEY, include_row_ids=True)
        if row["role"] == "user"
    ]
    assert resp["result"]["survivor_user_row_ids"] == surviving
    # Fresh rows, not the pre-rewind ids the client sent in.
    assert user_row_ids[0] not in surviving


def test_truncation_without_a_profile_uses_the_shared_handle(server, launch_db, monkeypatch):
    user_row_ids: list = []
    history = _seed(launch_db, _capture_user_row_ids=user_row_ids)
    _register(server, history)
    _stop_after_truncate(server, monkeypatch)

    resp = _rpc(
        server,
        "prompt.submit",
        {
            "session_id": SESSION_ID,
            "text": "edited question 2",
            "truncate_before_row_id": user_row_ids[1],
            "truncate_before_user_ordinal": 1,
            "confirm_truncate": True,
        },
    )

    assert not resp.get("error"), f"prompt.submit failed: {resp.get('error')}"
    assert _texts(launch_db.get_messages_as_conversation(SESSION_KEY)) == [
        "question 1",
        "answer 1",
    ]


# ---------------------------------------------------------------------------
# /history and /context — slash.exec
# ---------------------------------------------------------------------------


def test_history_reads_the_profile_transcript(server, launch_db, profile_db):
    """/history must render the profile session's own transcript."""
    profile_home, pdb = profile_db
    _seed(pdb)
    # In-memory history is deliberately empty: the point of the db read is to
    # rebuild the transcript for a session the process did not run itself.
    _register(server, [], profile_home=profile_home)

    resp = _rpc(server, "slash.exec", {"session_id": SESSION_ID, "command": "/history"})

    assert not resp.get("error"), f"/history failed: {resp.get('error')}"
    output = resp["result"]["output"]
    assert "question 3" in output
    assert "answer 3" in output


def test_context_reads_the_profile_transcript(server, launch_db, profile_db, monkeypatch):
    """/context must count the profile session's own messages."""
    profile_home, pdb = profile_db
    _seed(pdb)
    _register(server, [], profile_home=profile_home)
    # /context is an isolated-session read command, gated on the compute host.
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda *a, **k: True)

    resp = _rpc(server, "slash.exec", {"session_id": SESSION_ID, "command": "/context"})

    assert not resp.get("error"), f"/context failed: {resp.get('error')}"
    output = resp["result"]["output"]
    assert "Conversation: 6 messages" in output
    assert "user: 3, assistant: 3" in output
