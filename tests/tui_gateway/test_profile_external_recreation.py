"""A surviving RPC server admits external recreation without reviving stale owners."""

import multiprocessing
from pathlib import Path

import pytest

from hermes_cli import profiles
from hermes_cli.profile_incarnation import read_profile_incarnation
from hermes_cli.profile_lifecycle import (
    mark_profile_deleting,
    profile_lifecycle_lease,
    rollback_profile_retirement,
)
from hermes_state import SessionDB
from tui_gateway.profile_lifecycle import ProfileLifecycleFence


def _create_elsewhere(name):
    profiles._maybe_register_gateway_service = lambda *a, **k: None
    profiles._notify_multiplexer = lambda *a, **k: None
    profiles.create_profile(name, no_alias=True, no_skills=True)


def _external_create(name):
    child = multiprocessing.get_context("spawn").Process(target=_create_elsewhere, args=(name,))
    child.start()
    child.join(30)
    if child.is_alive():
        child.terminate()
        child.join(5)
    assert child.exitcode == 0


def _create_session(server, name):
    response = server.handle_request({
        "jsonrpc": "2.0", "id": name, "method": "session.create",
        "params": {"profile": name, "source": "desktop"},
    })
    assert "result" in response, response
    return server._sessions[response["result"]["session_id"]]


@pytest.mark.parametrize("external", [True, False], ids=["external", "local-control"])
def test_registered_rpc_admits_new_generation_and_fences_old(tmp_path, monkeypatch, external):
    import agent.secret_scope as secret_scope
    import tui_gateway.launch_profile_policy as launch_policy
    import tui_gateway.server as server

    root = tmp_path / "isolated-hermes"
    root.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    for name in ("_cleanup_gateway_service", "_maybe_unregister_gateway_service",
                 "_maybe_register_gateway_service", "_stop_bot_desktop", "_notify_multiplexer"):
        monkeypatch.setattr(profiles, name, lambda *a, **k: None)
    monkeypatch.setattr(profiles, "_profile_bound_backend_pids", lambda *a, **k: [])
    monkeypatch.setattr(server, "_hermes_home", root)
    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setattr(server, "_served_profile_homes", set())
    monkeypatch.setattr(server, "_db", None)
    monkeypatch.setattr(server, "_profile_lifecycle", ProfileLifecycleFence())
    monkeypatch.setattr(launch_policy, "_snapshot", None)
    monkeypatch.setattr(server, "_schedule_agent_build", lambda *a, **k: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda *a, **k: None)
    secret_scope.set_multiplex_active(False)
    try:
        home = profiles.create_profile("worker", no_alias=True, no_skills=True)
        _create_session(server, "default")
        old_record = _create_session(server, "worker")
        _create_session(server, "default")
        old = old_record["profile_incarnation"]
        # Independent fence copies prove each entrypoint admits the successor
        # from disk state alone, with no capture in their process.
        check_fence, lease_fence = ProfileLifecycleFence(), ProfileLifecycleFence()
        check_fence.retire(home, old)
        lease_fence.retire(home, old)
        assert profiles.delete_profile("worker", yes=True) == home
        assert not home.exists()
        assert old_record not in server._sessions.values()
        if external:
            _external_create("worker")
        else:
            profiles.create_profile("worker", no_alias=True, no_skills=True)
        new = read_profile_incarnation(home)
        assert new and new != old

        # Real registered RPC A -> B -> A on the original surviving server.
        _create_session(server, "default")
        fresh_record = _create_session(server, "worker")
        _create_session(server, "default")
        assert fresh_record["profile_incarnation"] == new
        assert not check_fence.rejected(home, new, require_incarnation=True)
        with lease_fence.lease(home, new):
            assert read_profile_incarnation(home) == new
        for fence in (server._profile_lifecycle, check_fence, lease_fence):
            assert fence.rejected(home, old, require_incarnation=True)
            with pytest.raises(FileNotFoundError):
                with fence.lease(home, old):
                    pytest.fail("stale callback entered the successor")
            assert (fence.key(home), old) in fence.retired_incarnations
        path = home / "state.db"
        assert not path.exists()
        with pytest.raises(FileNotFoundError):
            SessionDB(db_path=path, expected_profile_incarnation=old)
        assert not path.exists(), "stale DB admission created the successor's store"
        with SessionDB(db_path=path, expected_profile_incarnation=new) as db:
            db.create_session("fresh-row", source="cli")
            assert db.get_session("fresh-row")["id"] == "fresh-row"

        # A local retirement of the CURRENT token is not stale cache data.
        with profile_lifecycle_lease(home):
            server._profile_lifecycle.retire(home, new)
        response = server.handle_request({
            "jsonrpc": "2.0", "id": "retired", "method": "session.create",
            "params": {"profile": "worker", "source": "desktop"},
        })
        assert response["error"]["code"] == 4064
        marker = home / ".profile-incarnation"
        saved_marker = marker.read_bytes()
        marker.unlink()
        # A tokenless retirement (legacy tombstone) is fenced by the tombstone itself.
        mark_profile_deleting(home)
        with pytest.raises(FileNotFoundError):
            server._capture_profile_incarnation(home)
        assert not marker.exists(), "lazy marker backfill must not revive a retired generation"
        marker.write_bytes(saved_marker)
        assert check_fence.rejected(home, new, require_incarnation=True)
        rollback_profile_retirement(home, new)  # explicit rollback admits unchanged generation
        assert not check_fence.rejected(home, new, require_incarnation=True)
        assert _create_session(server, "worker")["profile_incarnation"] == new
    finally:
        for sid in list(server._sessions):
            server._close_session_by_id(sid)
        secret_scope.set_multiplex_active(False)
