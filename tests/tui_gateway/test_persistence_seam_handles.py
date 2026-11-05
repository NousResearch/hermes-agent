"""The TUI gateway's storage interface handles sit beside its stores, open nothing until a verb runs, and reach the
agents and cross-area helpers those stores reach."""
import sqlite3
import types

import pytest

from persistence_assembly import close_all_stores, open_persistence, profile_scope
from tui_gateway import server


@pytest.fixture
def connects(monkeypatch):
    calls = []
    real = sqlite3.connect

    def counting(*args, **kwargs):
        calls.append(args[0] if args else kwargs.get("database"))
        return real(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", counting)
    return calls


@pytest.fixture
def launch(monkeypatch, tmp_path):
    home = tmp_path / "launch"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(server, "_hermes_home", str(home))
    monkeypatch.setattr(server, "_db", None)
    monkeypatch.setattr(server, "_db_error", None)
    monkeypatch.setattr(server, "_launch_beside", (None, None))
    try:
        yield home
    finally:
        close_all_stores()


def test_launch_handle_is_kept_with_the_launch_store_and_opens_nothing(launch, connects):
    assert server._launch_persistence() is None  # no launch store held yet
    db = server._get_db()
    before = len(connects)
    handle = server._launch_persistence()
    assert connects[before:] == []
    assert handle.scope.profile_dir == launch.resolve()
    assert server._launch_persistence() is handle
    assert server._persistence_beside(db) is handle
    server._release_persistence(handle)
    handle.ready()  # still open: the launch handle lives as long as the launch store


def test_dedicated_handle_opens_nothing_until_used_and_releases(launch, tmp_path, connects):
    other = tmp_path / "other"
    other.mkdir()
    store = types.SimpleNamespace(db_path=other / "state.db")
    handle = server._persistence_beside(store)
    assert handle.scope.profile_dir == other.resolve()
    assert connects == [] and not (other / "state.db").exists()
    handle.ready()
    assert (other / "state.db").exists()  # the check above would see an open
    server._release_persistence(handle)
    with pytest.raises(RuntimeError):
        handle.ready()
    assert server._persistence_beside(None) is None
    assert server._persistence_beside(types.SimpleNamespace()) is None


def test_per_call_handle_follows_the_store_mode(launch, tmp_path, connects):
    other = tmp_path / "other"
    other.mkdir()
    with open_persistence(profile_scope(home=other)) as created:
        created.ready()
    reader = types.SimpleNamespace(db_path=other / "state.db", read_only=True)
    before = len(connects)
    with server._persistence_for(reader) as handle:
        assert connects[before:] == []
        assert handle.read_only is True
        assert handle.scope.profile_dir == other.resolve()
    with server._persistence_for(types.SimpleNamespace(db_path=other / "state.db")) as handle:
        assert handle.read_only is False
    with pytest.raises(RuntimeError):
        handle.ready()
    launch_db = server._get_db()
    with server._persistence_for(launch_db) as handle:
        assert handle is server._launch_persistence()
    handle.ready()


def test_agent_builds_get_the_handle_beside_their_store(launch, monkeypatch):
    captured = []
    monkeypatch.setattr("run_agent.AIAgent", lambda **kwargs: captured.append(kwargs) or types.SimpleNamespace())
    monkeypatch.setattr(server, "_load_enabled_toolsets", lambda platform=None: ["file"])
    monkeypatch.setattr(server, "_load_cfg", lambda: {"model": {"default": "m", "provider": "openai-codex"}})
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", lambda **kw: {
        "provider": "openai-codex", "base_url": "http://127.0.0.1:9/v1", "api_key": "k", "api_mode": "chat_completions",
        "credential_pool": None})
    server._make_agent("sid", "launch-key")
    assert captured[-1]["session_db"] is server._db
    assert captured[-1]["persistence"] is server._launch_persistence()
    reads = []
    dedicated = types.SimpleNamespace(get_session_model_config_value=lambda *a: reads.append(("store", *a)))
    handle = types.SimpleNamespace(read_only=False, sessions=types.SimpleNamespace(
        get_model_config_value=lambda *a: reads.append(("handle", *a))))
    server._make_agent("sid", "profile-key", session_db=dedicated, persistence=handle)
    assert captured[-1]["session_db"] is dedicated and captured[-1]["persistence"] is handle
    assert reads == [("handle", "profile-key", "onboarding_run")]  # the onboarding read goes through the writer handle


def test_rpc_helpers_hand_the_handle_to_cross_area_callees(launch, tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr("hermes_cli.foreign_sessions_browser.preview_foreign_session",
                        lambda handle, db, *, persistence=None: seen.append((db, persistence)) or {"ok": True})
    db = server._get_db()
    reply = server._methods["session.foreign.preview"](1, {"id": "x"})
    assert reply["result"] == {"ok": True}
    assert seen == [(db, server._launch_persistence())]

    refusals = []
    monkeypatch.setattr("hermes_cli.kanban_resume_guard.kanban_resume_refusal",
                        lambda db, target, *, persistence=None: refusals.append(persistence) or "refused")
    ctx = server._Resume(1, {}, "stored")
    ctx.db, ctx.persistence = db, server._launch_persistence()
    assert "error" in server._resume_guard(ctx)
    assert refusals == [ctx.persistence]
