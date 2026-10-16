"""Sessions in a named profile linked outside the Hermes root can open their store.

A multiplexed server (the Webapp, the dashboard) opens a linked profile's store through its
``profiles/<name>`` path, where the agent build and every token-bound writer RPC admit the
session's incarnation token.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hermes_state import SessionDB
from tui_gateway.profile_lifecycle import ProfileLifecycleFence


def link_directory(link: Path, target: Path) -> None:
    """The host's native directory link: a junction on Windows, a symlink elsewhere."""
    if os.name == "nt":
        import _winapi

        _winapi.CreateJunction(str(target), str(link))
    else:
        link.symlink_to(target, target_is_directory=True)


@pytest.fixture
def server(tmp_path, monkeypatch):
    from agent import secret_scope
    import hermes_state_registry as registry
    import tui_gateway.launch_profile_policy as launch_policy
    from tui_gateway import server

    root = tmp_path / "isolated-hermes"
    (root / "profiles").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(server, "_hermes_home", root)
    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setattr(server, "_served_profile_homes", set())
    monkeypatch.setattr(server, "_db", None)
    monkeypatch.setattr(server, "_profile_lifecycle", ProfileLifecycleFence())
    monkeypatch.setattr(launch_policy, "_snapshot", None)
    monkeypatch.setattr(server, "_schedule_agent_build", lambda *a, **k: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda *a, **k: None)
    secret_scope.set_multiplex_active(False)
    storage = tmp_path / "other-disk" / "linked"
    storage.mkdir(parents=True)
    (storage / "config.yaml").write_text("{}\n", encoding="utf-8")
    link_directory(root / "profiles" / "linked", storage)
    try:
        yield server
    finally:
        registry.close_all()
        secret_scope.set_multiplex_active(False)


def _rpc(server, method, params):
    return server.handle_request({"jsonrpc": "2.0", "id": method, "method": method, "params": params})


def test_linked_profile_session_opens_its_store_for_the_agent_build(server):
    created = _rpc(server, "session.create", {"profile": "linked", "source": "desktop"})
    assert "result" in created, created
    record = server._sessions[created["result"]["session_id"]]
    assert record["profile_incarnation"]

    # The same open the deferred agent build performs for a named-profile session.
    db = server._open_profile_session_db(
        record["profile_home"], expected_profile_incarnation=record["profile_incarnation"])
    try:
        db.create_session("linked-turn", "desktop")
        assert db.get_session("linked-turn")["id"] == "linked-turn"
    finally:
        import hermes_state_registry as registry

        registry.release(db)


def test_linked_profile_token_bound_writer_rpc_succeeds(server):
    home = Path(os.environ["HERMES_HOME"]) / "profiles" / "linked"
    with SessionDB(db_path=home / "state.db") as db:
        db.create_session("doomed", "tui")

    deleted = _rpc(server, "session.delete", {"profile": "linked", "session_id": "doomed"})

    assert deleted.get("result") == {"deleted": "doomed"}, deleted
    with SessionDB(db_path=home / "state.db") as db:
        assert db.get_session("doomed") is None


def test_message_to_a_retired_linked_profile_names_the_profile(server):
    from hermes_cli.profile_lifecycle import mark_profile_deleting, profile_lifecycle_lease

    created = _rpc(server, "session.create", {"profile": "linked", "source": "desktop"})
    sid = created["result"]["session_id"]
    record = server._sessions[sid]
    home = Path(record["profile_home"])
    with profile_lifecycle_lease(home):
        mark_profile_deleting(home, record["profile_incarnation"])

    sent = _rpc(server, "prompt.submit", {"session_id": sid, "text": "hello"})

    assert sent["error"]["code"] == 4041, sent
    assert "profile was deleted or recreated" in sent["error"]["message"]
    assert "doctor" not in sent["error"]["message"]
