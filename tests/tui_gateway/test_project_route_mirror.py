import sqlite3
from types import SimpleNamespace

from gateway.project_routes import bind_inbound_session
from hermes_cli import projects_db
from tui_gateway import server


def test_desktop_surface_mirrors_telegram_source_session(tmp_path, monkeypatch):
    conn = projects_db.connect(tmp_path / "projects.db")
    bind_inbound_session(
        conn,
        session_id="telegram-session",
        origin_kind="telegram",
        origin_key="-1001:44",
        telegram_chat_id="-1001",
        telegram_thread_id="44",
    )
    conn.close()
    session = {
        "source": "telegram",
        "session_key": "telegram-session",
        "agent": SimpleNamespace(session_id="telegram-session"),
        "profile_home": str(tmp_path),
    }
    calls = []
    monkeypatch.setattr(server, "_resolve_session_platform", lambda: "desktop")
    monkeypatch.setattr(
        server, "_persisted_turn_row_ids", lambda *args, **kwargs: (10, 11)
    )
    monkeypatch.setattr(
        "gateway.project_routes.mirror_desktop_turn",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    server._mirror_desktop_turn_after_persist(
        session,
        prior_session_id="telegram-session",
        user_text="question",
        assistant_text="answer",
        status="complete",
        status_note=None,
        display_kind=None,
        turn_start_row_id=9,
    )

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[1] == "telegram-session"
    assert kwargs["user_message_key"] == "row:10"
    assert kwargs["assistant_message_key"] == "row:11"


def test_non_desktop_surface_never_mirrors(monkeypatch):
    calls = []
    monkeypatch.setattr(server, "_resolve_session_platform", lambda: "tui")
    monkeypatch.setattr(
        "gateway.project_routes.mirror_desktop_turn",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    server._mirror_desktop_turn_after_persist(
        {"source": "telegram", "session_key": "s", "agent": SimpleNamespace(session_id="s")},
        prior_session_id="s",
        user_text="question",
        assistant_text="answer",
        status="complete",
        status_note=None,
        display_kind=None,
        turn_start_row_id=0,
    )
    assert calls == []


def test_persisted_turn_rows_must_be_new_exact_and_ordered(tmp_path):
    db = sqlite3.connect(tmp_path / "state.db")
    db.execute(
        "create table messages(id integer primary key, session_id text, role text, content text)"
    )
    db.executemany(
        "insert into messages(id,session_id,role,content) values(?,?,?,?)",
        [
            (1, "s", "user", "question"),
            (2, "s", "assistant", "answer"),
            (10, "s", "user", "question"),
            (11, "s", "assistant", "answer"),
        ],
    )
    db.commit()
    db.close()
    assert server._persisted_turn_row_ids(
        tmp_path,
        prior_session_id="s",
        current_session_id="s",
        after_row_id=2,
        user_text="question",
        assistant_text="answer",
    ) == (10, 11)
    assert (
        server._persisted_turn_row_ids(
            tmp_path,
            prior_session_id="s",
            current_session_id="s",
            after_row_id=11,
            user_text="question",
            assistant_text="answer",
        )
        is None
    )


def test_persisted_turn_rows_reject_interleaving_and_wrong_rotation_lineage(tmp_path):
    db = sqlite3.connect(tmp_path / "state.db")
    db.execute(
        "create table messages(id integer primary key, session_id text, role text, content text)"
    )
    db.executemany(
        "insert into messages(id,session_id,role,content) values(?,?,?,?)",
        [
            (20, "prior", "user", "question"),
            (21, "prior", "user", "other"),
            (22, "current", "assistant", "answer"),
            (30, "current", "user", "question"),
            (31, "prior", "assistant", "answer"),
        ],
    )
    db.commit()
    db.close()
    assert (
        server._persisted_turn_row_ids(
            tmp_path,
            prior_session_id="prior",
            current_session_id="current",
            after_row_id=19,
            user_text="question",
            assistant_text="answer",
        )
        is None
    )


def test_persisted_turn_rows_allow_tools_and_foreign_session_interleaving(tmp_path):
    db = sqlite3.connect(tmp_path / "state.db")
    db.execute(
        "create table messages(id integer primary key, session_id text, role text, content text)"
    )
    db.executemany(
        "insert into messages(id,session_id,role,content) values(?,?,?,?)",
        [
            (10, "s", "user", "question"),
            (11, "s", "assistant", None),
            (12, "foreign", "assistant", "unrelated"),
            (13, "s", "tool", "tool output"),
            (14, "s", "assistant", "answer"),
        ],
    )
    db.commit()
    db.close()
    assert server._persisted_turn_row_ids(
        tmp_path,
        prior_session_id="s",
        current_session_id="s",
        after_row_id=9,
        user_text="question",
        assistant_text="answer",
    ) == (10, 14)


def test_desynchronized_turn_never_mirrors(monkeypatch):
    calls = []
    monkeypatch.setattr(server, "_resolve_session_platform", lambda: "desktop")
    monkeypatch.setattr(
        server, "_persisted_turn_row_ids", lambda *args, **kwargs: (10, 11)
    )
    monkeypatch.setattr(
        "gateway.project_routes.mirror_desktop_turn",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    server._mirror_desktop_turn_after_persist(
        {"source": "telegram", "session_key": "s", "agent": SimpleNamespace(session_id="s")},
        prior_session_id="s",
        user_text="question",
        assistant_text="answer",
        status="complete",
        status_note="not persisted",
        display_kind=None,
        turn_start_row_id=9,
    )
    assert calls == []
