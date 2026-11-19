"""A branch seed whose transcript copy fails discards the child row it created.

A row nobody was admitted against is removed without retirement and without any post-delete work. A row that
already holds an admission goes through the fenced delete, so ``on_session_delete`` fires for it with
``reason="delete"``, as after any delete.
"""

from pathlib import Path

import pytest

from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_seed import seed_session
from tui_gateway import server


@pytest.fixture
def db(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    with open_persistence(profile_scope(home=home)) as handle:
        seed_session(handle, "parent", "desktop", model="test-model")
        yield handle


@pytest.fixture
def hooks(monkeypatch):
    calls = []
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", lambda name, **kw: calls.append((name, kw)) or [])
    return calls


def _admit(db, session_id: str) -> None:
    raw_holder(db)._execute_write(lambda conn: conn.execute(
        "INSERT INTO session_admissions (admission_id, request_id, principal_id, target_session_id, lineage_json,"
        " payload_json, payload_digest, intent, status, outcome)"
        " VALUES ('adm-1', 'req-1', 'principal', ?, '[]', '{}', 'digest', 'queue', 'terminal', 'completed')",
        (session_id,)))


def _fail_copy(monkeypatch, before=lambda: None):
    def fail(_row):
        before()
        raise RuntimeError("copy failed")
    monkeypatch.setattr(server, "_new_message", fail)


def _seed_child(db) -> None:
    with pytest.raises(RuntimeError, match="copy failed"):
        server._persist_branch(db, "child", "parent", "Branch", [{"role": "user", "content": "hello"}],
                               source="desktop", cwd=None, profile_name="default", model="test-model",
                               compensate=True)


def _delete_hooks(calls) -> list[tuple[str, str]]:
    return [(kw["session_id"], kw["reason"]) for name, kw in calls if name == "on_session_delete"]


def test_admitted_child_discard_fires_the_delete_hook(db, hooks, monkeypatch):
    _fail_copy(monkeypatch, before=lambda: _admit(db, "child"))

    _seed_child(db)

    assert db.sessions.get("child") is None
    assert db.runtime.retired_session("child")
    assert _delete_hooks(hooks) == [("child", "delete")]


def test_unadmitted_child_discard_fires_nothing(db, hooks, monkeypatch):
    _fail_copy(monkeypatch)

    _seed_child(db)

    assert db.sessions.get("child") is None
    assert not db.runtime.retired_session("child")
    assert _delete_hooks(hooks) == []
