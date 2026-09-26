"""#121062: desktop branch children must never stay untitled.

Two creation paths mint child sessions (``session.create`` with
``parent_session_id`` + seeded messages, ``session.branch``); both route the
row + lineage title through ``_persist_branch``. A lineage-title collision
(``ValueError`` from the taken-name guard) used to escape ``_persist_branch``,
and with ``compensate=True`` the just-created row — transcript included — was
rolled back, leaving a row the lazy first-prompt path re-created WITHOUT a
title. Fan-out batches hit this constantly (concurrent children compute the
same next lineage title), which is why whole walls of sidebar rows stayed
NULL-titled forever.
"""
from __future__ import annotations


def _wire_create_harness(monkeypatch, server, db):
    class _FakeAgent:
        def __init__(self):
            self.model = "test-model"

    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_make_agent", lambda sid, key, session_db=None, **_kw: _FakeAgent())
    monkeypatch.setattr(server, "_SlashWorker", lambda *a, **k: None)
    monkeypatch.setattr(server, "_session_info", lambda _a, *a2: {"model": "x"})
    monkeypatch.setattr(server, "_probe_credentials", lambda _a: None)
    monkeypatch.setattr(server, "_wire_callbacks", lambda _sid: None)
    monkeypatch.setattr(server, "_emit", lambda *a, **kw: None)
    import tools.approval as _approval
    monkeypatch.setattr(_approval, "register_gateway_notify", lambda key, cb: None)
    monkeypatch.setattr(_approval, "load_permanent_allowlist", lambda: None)


def _create_child(server, **params):
    base = {"cols": 96, "source": "desktop",
            "parent_session_id": "parent-key",
            "messages": [{"role": "user", "content": "hello from parent"}]}
    base.update(params)
    return server.handle_request({"id": "1", "method": "session.create", "params": base})


def test_seeded_branch_child_lands_lineage_title_real_db(monkeypatch, tmp_path):
    """Seed path (real store): parent titled -> child 'Parent #2', derived."""
    from hermes_state import SessionDB
    from tui_gateway import server

    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("parent-key", source="desktop", model="m")
        db.set_session_title("parent-key", "My Parent Session")
        _wire_create_harness(monkeypatch, server, db)
        resp = _create_child(server)
        assert "result" in resp, resp
        key = resp["result"]["stored_session_id"]
        assert db.get_session_title(key) == "My Parent Session #2"
        assert db.get_session_title_source(key) == "derived"
        assert len(db.get_messages(key)) == 1
        server._sessions.pop(resp["result"]["session_id"], None)


def test_branch_child_of_untitled_parent_falls_back_to_branch(monkeypatch, tmp_path):
    """Untitled parent -> child 'branch', not NULL."""
    from hermes_state import SessionDB
    from tui_gateway import server

    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("parent-key", source="desktop", model="m")
        _wire_create_harness(monkeypatch, server, db)
        resp = _create_child(server)
        assert "result" in resp, resp
        key = resp["result"]["stored_session_id"]
        assert db.get_session_title(key) == "branch"
        assert db.get_session_title_source(key) == "derived"
        server._sessions.pop(resp["result"]["session_id"], None)


def test_branch_title_collision_keeps_child_titled_with_transcript(tmp_path):
    """#121062 RED: a taken lineage title must dedupe, never raise + roll back.

    Fan-out batches compute the same next title concurrently; the loser used
    to eat ValueError -> compensate deleted its row AND transcript -> the lazy
    fallback re-created it WITHOUT a title (the permanent-NULL wall).
    """
    from hermes_state import SessionDB
    from tui_gateway import server

    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("parent-key", source="desktop", model="m")
        db.set_session_title("parent-key", "Work")
        db.create_session("squatter", source="desktop", model="m")
        db.set_session_title("squatter", "Work #2")  # race loser lands here
        # Both children computed "Work #2" before either wrote (fan-out race).
        server._persist_branch(db, "child-a", "parent-key", "Work #2",
                               [{"role": "user", "content": "task a"}],
                               source="desktop", cwd=str(tmp_path), profile_name="default",
                               model="m", compensate=True, title_source="derived")
        assert db.get_session_title("child-a") == "Work #3"
        assert db.get_session_title_source("child-a") == "derived"
        assert len(db.get_messages("child-a")) == 1


def test_branch_children_of_same_parent_get_unique_titles(tmp_path):
    """Sequential fan-out over one parent: P #2, P #3, ... all titled."""
    from hermes_state import SessionDB
    from tui_gateway import server

    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("parent-key", source="desktop", model="m")
        db.set_session_title("parent-key", "Work")
        for i, cid in enumerate(["c1", "c2", "c3"]):
            server._persist_branch(db, cid, "parent-key",
                                   server._branch_title(db, "parent-key"),
                                   [{"role": "user", "content": f"task {i}"}],
                                   source="desktop", cwd=str(tmp_path), profile_name="default",
                                   model="m", compensate=True, title_source="derived")
        assert [db.get_session_title(c) for c in ["c1", "c2", "c3"]] == [
            "Work #2", "Work #3", "Work #4"]
        for c in ["c1", "c2", "c3"]:
            assert db.get_session_title_source(c) == "derived"
            assert len(db.get_messages(c)) == 1


def test_lazy_branch_child_gets_instant_title_on_first_turn(monkeypatch, tmp_path):
    """Parented but unseeded child (lazy row) is titled by the turn titler."""
    from hermes_state import SessionDB
    from agent import title_generator as tg

    monkeypatch.setattr(tg, "generate_title", lambda *a, **k: (None, "llm"))
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("parent-key", source="desktop", model="m")
        db.create_session("child-key", source="desktop", model="m",
                          parent_session_id="parent-key")
        tg.maybe_auto_title(db, "child-key", "investigate the auth flake",
                            [{"role": "user", "content": "investigate the auth flake"}],
                            main_runtime=None)
        tg.wait_for_title_upgrades(timeout=10)
        title = db.get_session_title("child-key")
        assert title and "auth" in title.lower(), title
