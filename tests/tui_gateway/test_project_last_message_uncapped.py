"""Isolated regression fixtures for the native strict conversation-message clock."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_state import SessionDB
from tui_gateway import project_tree, server
from tui_gateway.contracts.projects_pets import ProjectTreeNode


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    home = tmp_path / "profile"
    home.mkdir()
    token = set_hermes_home_override(home)
    previous = (server._db, server._db_error, server._hermes_home)
    db = SessionDB(db_path=home / "state.db")
    server._db, server._db_error, server._hermes_home = db, None, home
    monkeypatch.setattr(server, "_load_cfg", lambda: {"desktop": {
        "repo_scan_enabled": False, "repo_scan_roots": [], "repo_scan_exclude_paths": []}})
    from tui_gateway import git_probe
    monkeypatch.setattr(git_probe, "warm_roots", lambda *_: None)
    monkeypatch.setattr(git_probe, "resolve", lambda *_: None)
    try:
        yield db, tmp_path
    finally:
        db.close()
        server._db, server._db_error, server._hermes_home = previous
        reset_hermes_home_override(token)


def add_session(db, sid, cwd, started, messages=(), *, source="cli", parent=None,
                config=None, archived=False, hidden=False, heartbeat=0):
    db.create_session(sid, source, cwd=str(cwd) if cwd else None, parent_session_id=parent,
                      model_config=config or {})
    for role, timestamp, active, compacted in messages:
        db.append_message(sid, role, f"{sid}-{role}-{timestamp}")
        db._write_sql("UPDATE messages SET timestamp=?, active=?, compacted=? "
                      "WHERE id=(SELECT MAX(id) FROM messages WHERE session_id=?)",
                      (timestamp, active, compacted, sid))
    db._write_sql("UPDATE sessions SET started_at=?, last_activity_at=?, archived=?, hidden=? WHERE id=?",
                  (started, heartbeat, int(archived), int(hidden), sid))


def tree_rpc():
    response = server._methods["projects.tree"](1, {})
    assert "error" not in response, response
    return {p["id"]: p for p in response["result"]["projects"]}


def test_rpc_strict_message_clock_beats_heartbeat_tool_creation_and_move(isolated):
    db, tmp = isolated
    old, newer, empty = [tmp / name for name in ("old", "newer", "empty")]
    for path in (old, newer, empty):
        path.mkdir()
    from hermes_cli import projects_db as pdb
    with pdb.connect_closing() as conn:
        ids = {path.name: pdb.create_project(conn, name=path.name, folders=[str(path)])
                 for path in (old, newer, empty)}
        # Project edits / folder moves must not advance the conversation clock.
        pdb.update_project(conn, ids["old"], name="Moved yesterday")
    add_session(db, "old", old, 990, [("user", 100, 1, 0), ("assistant", 110, 1, 0)], heartbeat=1200)
    add_session(db, "newer", newer, 10, [("user", 700, 1, 0), ("assistant", 800, 1, 0),
                                           ("tool", 950, 1, 0), ("user", 980, 0, 0)], heartbeat=0)
    # A child-less empty explicit project has no message clock, even if newly created.
    nodes = tree_rpc()
    assert nodes[ids["old"]]["lastActive"] == 1200  # legacy heartbeat semantics unchanged
    assert nodes[ids["newer"]]["lastActive"] == 980  # inactive rows also affect the legacy clock
    assert nodes[ids["old"]]["lastActive"] > nodes[ids["newer"]]["lastActive"]
    assert nodes[ids["old"]]["lastMessageAt"] == 110
    assert nodes[ids["newer"]]["lastMessageAt"] == 800
    assert nodes[ids["empty"]]["lastMessageAt"] == 0
    assert sorted((ids["old"], ids["newer"]), key=lambda pid: -nodes[pid]["lastMessageAt"]) == [
        ids["newer"], ids["old"]]
    assert ProjectTreeNode.model_validate(nodes[ids["newer"]]).lastMessageAt == 800
    drill = server._methods["projects.project_sessions"](2, {"project_id": ids["newer"]})
    assert drill["result"]["project"]["lastMessageAt"] == 800


def test_lineage_branch_delegate_exclusions_and_home(isolated):
    db, tmp = isolated
    root_path, other = tmp / "root", tmp / "other"
    root_path.mkdir(); other.mkdir()
    add_session(db, "root", root_path, 20, [("assistant", 760, 0, 1), ("tool", 990, 1, 0)])
    db._write_sql("UPDATE sessions SET end_reason='compression', ended_at=21 WHERE id='root'")
    add_session(db, "mid", root_path, 21, [("user", 250, 0, 1)], parent="root")
    db._write_sql("UPDATE sessions SET end_reason='compression', ended_at=22 WHERE id='mid'")
    add_session(db, "tip", root_path, 22, [("assistant", 300, 1, 0)], parent="mid")
    # Distinct branch is its own conversation; still aggregates into its owning project.
    add_session(db, "branch", other, 23, [("user", 740, 1, 0)], parent="root",
                config={"_branched_from": "root"})
    add_session(db, "delegate", root_path, 24, [("assistant", 999, 1, 0)], parent="root",
                config={"_delegate_from": "root"})
    add_session(db, "cron", root_path, 25, [("user", 998, 1, 0)], source="cron")
    add_session(db, "archived", root_path, 26, [("user", 997, 1, 0)], archived=True)
    add_session(db, "hidden", root_path, 27, [("user", 996, 1, 0)], hidden=True)
    add_session(db, "home", None, 28, [("user", 700, 1, 0)])
    nodes = tree_rpc()
    assert nodes[str(root_path)]["lastMessageAt"] == 760  # compacted root survives tip projection
    assert nodes[str(other)]["lastMessageAt"] == 740
    assert nodes[project_tree.NO_PROJECT_ID]["lastMessageAt"] == 700
    assert nodes[str(root_path)]["sessionCount"] == 1
    assert nodes[str(other)]["sessionCount"] == 1


def test_user_message_can_be_newest_and_missing_messages_are_zero(isolated):
    db, tmp = isolated
    a, b = tmp / "a", tmp / "b"
    a.mkdir(); b.mkdir()
    add_session(db, "a", a, 999, [("assistant", 200, 1, 0), ("user", 400, 1, 0),
                                    ("assistant", 990, 0, 0), ("assistant", 1e200, 1, 0)])
    add_session(db, "b", b, 1000, [("tool", 900, 1, 0)])
    # Seed count for a legacy/tool-only row to exercise the tree even without a qualifying message.
    nodes = tree_rpc()
    assert nodes[str(a)]["lastMessageAt"] == 400
    assert nodes[str(b)]["lastMessageAt"] == 0
    assert nodes[str(a)]["lastActive"] == 990


def test_message_lookup_batches_selected_ids_without_per_row_queries(isolated, monkeypatch):
    db, tmp = isolated
    add_session(db, "real", tmp, 10, [("assistant", 123, 1, 0)])
    rows = [{"id": f"absent-{i}"} for i in range(900)] + [{"id": "real"}]
    original = db._read_all
    queries = []

    def counted(sql, params=()):
        if "SELECT m.session_id, MAX(m.timestamp)" in sql:
            queries.append(len(params))
        return original(sql, params)

    monkeypatch.setattr(db, "_read_all", counted)
    times = db.last_user_assistant_message_times(rows)
    assert queries == [900, 1]
    assert times["real"] == 123
    assert times["absent-0"] == 0


def test_other_profile_state_cannot_supply_a_project_message(isolated, tmp_path):
    db, tmp = isolated
    cwd = tmp / "shared"
    cwd.mkdir()
    add_session(db, "launch", cwd, 10, [("user", 120, 1, 0)])
    launch = tree_rpc()[str(cwd)]
    other_home = tmp_path / "other-profile"
    other_home.mkdir()
    other_db = SessionDB(db_path=other_home / "state.db")
    previous_db, previous_home = server._db, server._hermes_home
    token = set_hermes_home_override(other_home)
    try:
        server._db, server._hermes_home = other_db, other_home
        add_session(other_db, "other", cwd, 20, [("assistant", 880, 1, 0)])
        other = tree_rpc()[str(cwd)]
    finally:
        server._db, server._hermes_home = previous_db, previous_home
        reset_hermes_home_override(token)
        other_db.close()
    assert launch["lastMessageAt"] == 120
    assert other["lastMessageAt"] == 880
    assert tree_rpc()[str(cwd)]["lastMessageAt"] == 120


def test_uncapped_overview_clock_for_project_outside_2000_selected_rows(isolated, monkeypatch):
    db, tmp = isolated
    old_dir, busy_dir = tmp / 'old-project', tmp / 'busy-project'
    old_dir.mkdir(); busy_dir.mkdir()
    from hermes_cli import projects_db as pdb
    with pdb.connect_closing() as conn:
        old_id = pdb.create_project(conn, name='Old', folders=[str(old_dir)])
        busy_id = pdb.create_project(conn, name='Busy', folders=[str(busy_dir)])
    # The old project has a retained compacted root message and a newer tip;
    # both segments are invisible to the bounded activity-ranked overview.
    add_session(db, 'old-root', old_dir, 5, [('assistant', 680, 0, 1)])
    db._write_sql("UPDATE sessions SET end_reason='compression', ended_at=6 WHERE id='old-root'")
    add_session(db, 'old-tip', old_dir, 6, [('user', 710, 1, 0)], parent='old-root')
    add_session(db, 'template', busy_dir, 900, [('tool', 2, 1, 0)], heartbeat=900)
    # Bulk fixtures avoid 2,000 transaction-per-row writes while retaining
    # a real tool-only message for every qualifying native session.
    db._write_sql(
        'INSERT INTO sessions (id, source, cwd, started_at, last_activity_at, message_count) '
        'VALUES (?, ?, ?, ?, ?, ?)',
        [(f'busy-{i}', 'cli', str(busy_dir), 800+i, 800+i, 1)
         for i in range(2001)], many=True)
    db._write_sql(
        'INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)',
        [(f'busy-{i}', 'tool', 'tool-only', 800+i) for i in range(2001)], many=True)
    original = db._read_all
    clock_queries = []
    def counted(sql, params=()):
        if 'SELECT m.session_id, MAX(m.timestamp)' in sql:
            clock_queries.append(len(params))
        return original(sql, params)
    monkeypatch.setattr(db, '_read_all', counted)
    response = server._methods['projects.tree'](7, {'session_limit': 2000})
    assert 'error' not in response, response
    nodes = {p['id']: p for p in response['result']['projects']}
    assert nodes[old_id]['sessionCount'] == 0  # existing bounded membership unchanged
    assert nodes[old_id]['lastMessageAt'] == 710  # full lineage, outside payload
    assert nodes[busy_id]['lastMessageAt'] == 0
    assert 'old-tip' not in response['result']['scoped_session_ids']
    assert max(clock_queries) <= 900
    assert len(clock_queries) < 10  # batched, not a query per conversation


def test_uncapped_clock_does_not_import_hidden_archived_delegate_or_wrong_source(isolated):
    db, tmp = isolated
    cwd = tmp / 'project'
    cwd.mkdir()
    add_session(db, 'visible', cwd, 1, [('user', 50, 1, 0)])
    add_session(db, 'archived', cwd, 1, [('user', 901, 1, 0)], archived=True)
    add_session(db, 'hidden', cwd, 1, [('assistant', 902, 1, 0)], hidden=True)
    add_session(db, 'cron', cwd, 1, [('user', 903, 1, 0)], source='cron')
    add_session(db, 'delegate', cwd, 1, [('user', 904, 1, 0)], parent='visible',
                config={'_delegate_from': 'visible'})
    assert tree_rpc()[str(cwd)]['lastMessageAt'] == 50


def test_all_profiles_merge_keeps_max_strict_clock_independent_of_last_active():
    from hermes_cli.web_routers.profiles import _merge_profile_tree
    merged = {}
    a = {"id": "p_a", "path": "/shared", "repos": [], "previewSessions": [],
         "sessionCount": 1, "lastActive": 900, "lastMessageAt": 100}
    b = {"id": "p_b", "path": "/shared", "repos": [], "previewSessions": [],
         "sessionCount": 1, "lastActive": 800, "lastMessageAt": 700}
    _merge_profile_tree(merged, [a], "default", 3)
    _merge_profile_tree(merged, [b], "other", 3)
    result = next(iter(merged.values()))
    assert result["lastMessageAt"] == 700
    assert result["lastActive"] == 900
