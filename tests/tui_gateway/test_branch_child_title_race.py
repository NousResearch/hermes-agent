"""Seeded branch children must not lose their row/title to a lineage-name race (#121062).

Fan-out siblings each run the same check-then-write: ``_branch_title`` ->
``get_next_title_in_lineage`` reads the same free number, then ``set_auto_title`` rejects the
losers with "Title already in use". That ValueError used to reach ``_persist_branch``'s
compensation guard: the child row was deleted, the lazy first-prompt rebuild re-created it with
the copied transcript but no title, and the sidebar filled with anonymous branch rows whose
titles never landed (neither the seed write nor a turn of their own ever names them)."""

import subprocess

import pytest

from hermes_state import SessionDB

HISTORY = [{"role": "user", "content": "Synthetic parent input"},
           {"role": "assistant", "content": "Synthetic parent response"}]


def _gateway(monkeypatch, tmp_path):
    from tui_gateway import server

    project = tmp_path / "proj"
    project.mkdir()
    subprocess.run(["git", "init", "-q", str(project)], check=True)
    db = SessionDB(tmp_path / "state.db")
    monkeypatch.setattr("hermes_cli.banner.prefetch_update_check", lambda: None)
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setattr(server, "_load_cfg", lambda: {})
    monkeypatch.setattr(server, "_profile_home", lambda *a: None)
    monkeypatch.setattr(server, "_resolve_model", lambda: "test-model")
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    monkeypatch.setattr(server, "_schedule_agent_build", lambda *a: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda *a: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda *a: None)
    monkeypatch.setattr(server, "_project_info_for_cwd", lambda *a: None)
    return server, db, project


def _occupy(db, cwd, *titles):
    """Sessions holding every lineage name a racing child would try, as a previous fan-out's
    winners would."""
    for i, title in enumerate(titles):
        holder = f"holder-{i}"
        db.create_session(holder, source="desktop", cwd=str(cwd))
        assert db.set_auto_title(holder, title, source="derived"), title


def test_seeded_branch_keeps_row_and_continues_past_occupied_lineage(monkeypatch, tmp_path):
    """session.create(parent, messages): names a previous fan-out's winners still hold; the
    next batch's child must land the next free lineage number with its row and transcript."""
    server, db, project = _gateway(monkeypatch, tmp_path)
    try:
        db.create_session("parent", source="desktop", cwd=str(project))
        _occupy(db, project, "branch", "branch #2")

        response = server._methods["session.create"]("create", {
            "source": "desktop", "cwd": str(project),
            "parent_session_id": "parent", "messages": HISTORY,
        })
        assert "error" not in response, response
        child = response["result"]["stored_session_id"]
        assert db.get_session(child) is not None
        assert db.get_session_title(child) == "branch #3"
        assert db.get_session_title_source(child) == SessionDB.TITLE_SOURCE_DERIVED
        assert db.message_count(child) == len(HISTORY)
    finally:
        db.close()


def test_persist_branch_derived_title_retry_advances_one_number_at_a_time(tmp_path):
    """Unit: a ValueError on the derived title (a racing sibling wrote the same name between
    _branch_title's read and this write) advances to the next lineage number and retries
    instead of reaching the compensation guard — which deletes the row, and the lazy
    first-prompt rebuild keeps the transcript with a NULL title forever (#121062)."""
    from tui_gateway.methods_session import _persist_branch

    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("parent", source="desktop", cwd=str(tmp_path))
        _occupy(db, tmp_path, "branch", "branch #2")

        # The racing value: what a sibling wrote between _branch_title's read and this write.
        _persist_branch(db, "child", "parent", "branch", HISTORY, source="desktop",
                        cwd=str(tmp_path), profile_name="default", model="test-model",
                        compensate=True, title_source="derived")
        # The row survived the compensation guard and the title advanced past both holders.
        assert db.get_session("child") is not None, "compensation deleted the racing child row"
        assert db.get_session_title("child") == "branch #3"
        assert db.get_session_title_source("child") == SessionDB.TITLE_SOURCE_DERIVED
        assert db.message_count("child") == len(HISTORY)
    finally:
        db.close()


def test_persist_branch_user_title_collision_still_raises(tmp_path):
    """A user-chosen name keeps the loud failure: silently renumbering a name the user typed is
    a rename they never asked for."""
    from tui_gateway.methods_session import _persist_branch

    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("parent", source="desktop", cwd=str(tmp_path))
        _occupy(db, tmp_path, "User chosen title")

        with pytest.raises(ValueError, match="already in use"):
            _persist_branch(db, "child", "parent", "User chosen title", HISTORY, source="desktop",
                            cwd=str(tmp_path), profile_name="default", model="test-model",
                            title_source="user")
        # The failed user-name branch never compensates (compensate defaults to False): the row
        # stays, session.branch reports the error, the caller decides what to show.
        assert db.get_session("child") is not None
    finally:
        db.close()
