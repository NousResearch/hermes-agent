"""A TUI/Desktop branch child sends the bytes its parent already sends.

Without a stored prompt the child's first turn rebuilds (a fresh workspace probe), so the warm
cache the copied transcript buys is lost at byte 0 whenever the repo moved since session start.
"""

from __future__ import annotations

from pathlib import Path


def test_persist_branch_copies_the_parent_system_prompt(tmp_path, monkeypatch):
    from hermes_state import SessionDB
    from tui_gateway import server

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    with SessionDB(home / "state.db") as db:
        db.create_session("parent", source="desktop", model="test-model")
        db.update_system_prompt("parent", "PARENT PROMPT\n\nWorkspace snapshot: session start")
        server._persist_branch(db, "child", "parent", "Branch", [{"role": "user", "content": "hello"}],
                               source="desktop", cwd=str(tmp_path), profile_name="default",
                               model="test-model")
        # A parent with no stored prompt yields a child with none — never a phantom empty string.
        db.create_session("bare", source="desktop", model="test-model")
        server._persist_branch(db, "bare-child", "bare", "Branch 2", [{"role": "user", "content": "hi"}],
                               source="desktop", cwd=str(tmp_path), profile_name="default",
                               model="test-model")

    with SessionDB(home / "state.db") as db:
        assert db.get_session("child")["system_prompt"] == "PARENT PROMPT\n\nWorkspace snapshot: session start"
        assert db.get_session("bare-child")["system_prompt"] is None
