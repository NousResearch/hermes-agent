"""``projects.state.*`` JSON-RPC methods: the per-project handover record over the wire.

Calls go through ``server.handle_request`` so the declared contract (unknown keys → 4000, result
shape checked strictly under test isolation) is exercised along with the handler.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import pytest

import tui_gateway.server as server
from hermes_cli import projects_db as pdb
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def _rpc(method: str, params: dict) -> dict:
    return server.handle_request({"id": 1, "method": method, "params": params})


def _ok(method: str, params: dict) -> dict:
    resp = _rpc(method, params)
    assert "error" not in resp, resp.get("error")
    return resp["result"]


@pytest.fixture
def project_id(tmp_path):
    return _ok("projects.create", {"name": "Demo", "folders": [str(tmp_path)]})["project"]["id"]


def test_get_set_history_round_trip(project_id):
    assert _ok("projects.state.get", {"id": project_id}) == {"state": None}
    assert _ok("projects.state.history", {"id": project_id}) == {"history": []}

    first = _ok("projects.state.set", {"id": project_id, "goal": "Ship v1", "now": "tests"})["state"]
    assert (first["goal"], first["now"], first["next"], first["updated_by"]) == (
        "Ship v1", "tests", None, "user")

    # Addressed by slug; a partial write keeps the goal.
    second = _ok("projects.state.set", {"id": "demo", "now": "implementation"})["state"]
    assert (second["goal"], second["now"]) == ("Ship v1", "implementation")
    assert _ok("projects.state.get", {"id": project_id})["state"] == second

    history = _ok("projects.state.history", {"id": project_id})["history"]
    assert [h["now"] for h in history] == ["implementation", "tests"]
    limited = _ok("projects.state.history", {"id": project_id, "limit": 1})["history"]
    assert [h["now"] for h in limited] == ["implementation"]


def test_history_limit_beyond_sqlite_integer_range_is_clamped(project_id):
    _ok("projects.state.set", {"id": project_id, "goal": "g"})

    resp = _rpc("projects.state.history", {"id": project_id, "limit": 2**63})

    assert "error" not in resp, resp.get("error")
    assert [h["goal"] for h in resp["result"]["history"]] == ["g"]


@pytest.mark.parametrize("method,extra", [
    ("projects.state.get", {}),
    ("projects.state.set", {"goal": "g"}),
    ("projects.state.history", {}),
])
def test_unknown_project_is_5062(method, extra):
    assert _rpc(method, {"id": "nope", **extra})["error"]["code"] == 5062


def test_refused_writes_are_5063(project_id):
    assert _rpc("projects.state.set", {"id": project_id})["error"]["code"] == 5063
    too_long = "x" * (pdb.STATE_FIELD_MAX_CHARS + 1)
    assert _rpc("projects.state.set", {"id": project_id, "blockers": too_long})["error"]["code"] == 5063
    assert _ok("projects.archive", {"id": project_id})
    assert _rpc("projects.state.set", {"id": project_id, "goal": "g"})["error"]["code"] == 5063
    assert _ok("projects.state.get", {"id": project_id})["state"] is None


def test_set_always_records_user(project_id):
    # The handler ignores a smuggled author...
    resp = server._methods["projects.state.set"](1, {"id": project_id, "goal": "g", "updated_by": "agent"})
    assert resp["result"]["state"]["updated_by"] == "user"
    # ...and the wire contract rejects the key outright, writing nothing.
    wire = _rpc("projects.state.set", {"id": project_id, "now": "n", "updated_by": "agent"})
    assert wire["error"]["code"] == 4000
    state = _ok("projects.state.get", {"id": project_id})["state"]
    assert (state["now"], state["updated_by"]) == (None, "user")


@contextlib.contextmanager
def _serving_launch_profile(launch_home: Path):
    """Run handlers as a backend launched under ``launch_home`` (see test_projects_rpc.py)."""
    from hermes_state import SessionDB

    token = set_hermes_home_override(launch_home)
    prev_db, prev_error, prev_home = server._db, server._db_error, server._hermes_home
    server._hermes_home = launch_home
    server._db = SessionDB(db_path=launch_home / "state.db")
    server._db_error = None
    try:
        yield
    finally:
        server._db.close()
        server._db, server._db_error, server._hermes_home = prev_db, prev_error, prev_home
        reset_hermes_home_override(token)


def test_state_rpc_is_scoped_to_the_requested_profile(monkeypatch, tmp_path):
    launch_home = tmp_path / "homes" / "launch"
    coder_home = tmp_path / "homes" / "coder"
    for home in (launch_home, coder_home):
        home.mkdir(parents=True)
    homes = {"default": launch_home, "coder": coder_home}
    monkeypatch.setattr(
        "hermes_cli.profiles.get_profile_dir",
        lambda name: homes.get(name, tmp_path / "homes" / "missing" / name))

    with _serving_launch_profile(launch_home):
        coder_pid = _ok("projects.create", {
            "profile": "coder", "name": "Coder", "folders": [str(tmp_path / "repo")]})["project"]["id"]
        _ok("projects.state.set", {"profile": "coder", "id": coder_pid, "goal": "coder goal"})
        from_launch = _rpc("projects.state.get", {"id": coder_pid})
        from_coder = _ok("projects.state.get", {"profile": "coder", "id": coder_pid})

    assert from_launch["error"]["code"] == 5062
    assert from_coder["state"]["goal"] == "coder goal"
    with pdb.connect_closing(coder_home / "projects.db") as conn:
        assert pdb.get_project_state(conn, coder_pid)["goal"] == "coder goal"
