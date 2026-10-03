"""Profile deletion must drain a shared serve backend's live sessions for that home (#89729).

A remote Bot Mode delete lands on the same process that hosts the profile's sessions: their
agents hold its state.db, run threads and log handles, so an unwrapped delete raced its own
writers. ``profile_deletion_scope`` pops + tears those sessions down (clients see
``session.reclaimed`` reason ``profile_delete``) and refuses any new registration for the
home until rmtree has finished.
"""

import contextlib

import pytest

from tui_gateway import server


class _Agent:
    def __init__(self, name):
        self.name = name
        self.closed = False

    def close(self):
        self.closed = True


def _session(agent, home):
    return {
        "agent": agent,
        "history": [],
        "history_lock": __import__("threading").Lock(),
        "profile_home": str(home),
        "session_key": f"{agent.name}-stored",
        "source": "desktop",
    }


@pytest.fixture
def two_homes(tmp_path, monkeypatch):
    target_home = tmp_path / "profiles" / "worker"
    other_home = tmp_path / "profiles" / "other"
    target_home.mkdir(parents=True)
    other_home.mkdir(parents=True)
    monkeypatch.setattr(server, "_finalize_session", lambda *a, **k: None)
    broadcasts = []
    monkeypatch.setattr(
        server, "_broadcast_global_event", lambda method, payload: broadcasts.append((method, payload)))
    server._sessions["target"] = _session(_Agent("target"), target_home)
    server._sessions["other"] = _session(_Agent("other"), other_home)
    try:
        yield target_home, other_home, broadcasts
    finally:
        for sid in ("target", "other"):
            server._sessions.pop(sid, None)


def test_deletion_scope_drains_only_the_deleted_home(two_homes):
    target_home, other_home, broadcasts = two_homes
    before = {sid: s for sid, s in server._sessions.items()}

    with server.profile_deletion_scope(target_home) as drained:
        assert drained == 1
        # The deleted home's session is gone and its agent closed; the sibling survives.
        assert "target" not in server._sessions
        assert before["target"]["agent"].closed
        assert "other" in server._sessions
        assert not before["other"]["agent"].closed
        # The drained client learns via the session.reclaimed broadcast (reason profile_delete).
        assert broadcasts == [
            ("session.reclaimed", {
                "reason": "profile_delete",
                "session_id": "target",
                "stored_session_id": "target-stored",
            }),
        ]

    # Teardown is a no-op when the scope exits: nothing resurrected, sibling untouched.
    assert set(server._sessions) == {"other"}


def test_deletion_scope_blocks_new_registration_until_exit(two_homes, monkeypatch):
    target_home, _other_home, _broadcasts = two_homes
    # session.create resolves the profile home first; it must stay resolvable (the
    # directory is only rmtree'd after the scope enters) so the registration guard,
    # not 4064, is what fails the race.
    monkeypatch.setattr(server, "_profile_home", lambda profile: target_home)
    monkeypatch.setattr(server, "_schedule_agent_build", lambda *a, **k: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)

    with server.profile_deletion_scope(target_home):
        assert server._profile_deletion_blocks(target_home)
        response = server.handle_request({
            "id": "delete-race", "method": "session.create", "params": {"profile": "worker"},
        })
        assert response["error"]["code"] == 5037
        assert "being deleted" in response["error"]["message"]
        assert set(server._sessions) == {"other"}  # refused, not registered

    # After the delete finishes the home accepts work again.
    assert not server._profile_deletion_blocks(target_home)
    response = server.handle_request({
        "id": "after-delete", "method": "session.create", "params": {"profile": "worker"},
    })
    assert response["result"]["session_id"]
    server._sessions.pop(response["result"]["session_id"], None)


def test_deletion_scope_blocks_deferred_resume_registration(two_homes):
    target_home, _other_home, _broadcasts = two_homes
    record = {
        "history": [], "profile_home": str(target_home), "session_key": "stored-worker",
        "history_lock": __import__("threading").Lock(),
    }

    with server.profile_deletion_scope(target_home):
        # A resume that raced the delete cannot register after the drain.
        with pytest.raises(RuntimeError, match="being deleted"):
            server._claim_or_reuse_live("late-resume", "stored-worker", record, None)

    assert set(server._sessions) == {"other"}


def test_deletion_scope_always_unblocks_on_teardown_failure(two_homes, monkeypatch):
    """A crashing teardown must not leave the home permanently refused."""
    target_home, _other_home, _broadcasts = two_homes

    def _boom(*_a, **_k):
        raise RuntimeError("teardown exploded")

    monkeypatch.setattr(server, "_teardown_popped_session", _boom)
    with pytest.raises(RuntimeError, match="teardown exploded"):
        with server.profile_deletion_scope(target_home):
            pass
    assert not server._profile_deletion_blocks(target_home)


def test_deletion_scope_nests_without_premature_unblock(two_homes):
    """Nested scopes for one home stay blocking until the outermost exits."""
    target_home, _other_home, _broadcasts = two_homes
    with server.profile_deletion_scope(target_home):
        with server.profile_deletion_scope(target_home):
            assert server._profile_deletion_blocks(target_home)
        assert server._profile_deletion_blocks(target_home)
    assert not server._profile_deletion_blocks(target_home)


def test_rest_delete_wraps_cli_delete_in_deletion_scope(monkeypatch, tmp_path):
    """The REST delete must drain the in-process gateway's sessions before rmtree."""
    import asyncio

    from hermes_cli import profiles as profiles_mod
    from hermes_cli.web_routers import profiles as router

    profile_home = tmp_path / "profiles" / "worker"
    profile_home.mkdir(parents=True)
    events = []

    @contextlib.contextmanager
    def deletion_scope(home):
        events.append(("enter", str(home)))
        yield 1
        events.append(("exit", str(home)))

    def delete_profile(name, *, yes):
        events.append(("delete", name, yes))
        return profile_home

    monkeypatch.setattr(server, "profile_deletion_scope", deletion_scope)
    monkeypatch.setattr(router, "_resolve_profile_dir", lambda _name: profile_home)
    monkeypatch.setattr(profiles_mod, "delete_profile", delete_profile)

    result = asyncio.run(router.delete_profile_endpoint("worker"))

    assert result == {"ok": True, "path": str(profile_home)}
    assert events == [
        ("enter", str(profile_home)),
        ("delete", "worker", True),
        ("exit", str(profile_home)),
    ]
