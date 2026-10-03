"""Retry fast paths must not revive a runtime from a replaced profile generation."""

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("operation", ["create", "branch"])
def test_idempotency_reuses_only_a_live_profile_generation(tmp_path, monkeypatch, operation):
    from hermes_cli.profile_incarnation import write_fresh_profile_incarnation
    from hermes_cli.profile_lifecycle import profile_lifecycle_lease
    from hermes_state import SessionDB
    from tui_gateway import server
    from tui_gateway.profile_lifecycle import ProfileLifecycleFence

    home = tmp_path / ".hermes"
    profile = home / "profiles" / "worker"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text("model:\n  default: test-model\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(server, "_hermes_home", str(home))
    monkeypatch.setattr(server, "_profile_lifecycle", ProfileLifecycleFence())
    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setattr(server, "_idempotency_keys", {})
    monkeypatch.setattr(server, "_get_db", lambda: None)
    for name in ("_schedule_agent_build", "_schedule_session_cap_enforcement", "_register_session_cwd",
                 "_enable_gateway_prompts"):
        monkeypatch.setattr(server, name, lambda *args: None)
    monkeypatch.setattr(server, "_session_default_model", lambda _session: "test-model")
    monkeypatch.setattr(server, "_session_info", lambda *_args: {})
    monkeypatch.setattr(server, "_fallback_session_info", lambda *_args: {})
    monkeypatch.setattr(server, "_project_info_for_cwd", lambda *_args: None)
    monkeypatch.setattr(server.git_probe, "branch", lambda *_args: None)

    incarnation = server._capture_profile_incarnation(profile)
    old = {"session_key": "old-child", "profile_home": str(profile), "profile_incarnation": incarnation,
           "cwd": str(tmp_path), "history": [{"role": "user", "content": "old generation"}],
           "history_lock": threading.Lock(), "branch_title": "old branch", "parent_session_id": "parent"}
    parent = {**old, "session_key": "parent", "history": [{"role": "user", "content": "current parent"}]}
    server._sessions["old-runtime"] = old
    server._idempotency_keys["retry"] = ("old-runtime", time.time())
    params = {"profile": "worker", "idempotency_key": "retry", "cwd": str(tmp_path)}

    def request():
        if operation == "create":
            return server._methods["session.create"]("retry", params)
        return server._branch_live("retry", {"idempotency_key": "retry", "name": "fresh branch"}, parent)

    # An ordinary retry still reuses its original runtime before replacement.
    assert request()["result"]["session_id"] == "old-runtime"

    # Another process replaces the pathname before this process reaps its old runtime.
    # The disk incarnation is authoritative even without an in-process retire call.
    with profile_lifecycle_lease(profile):
        profile.rename(home / "retired-worker")
        profile.mkdir()
        (profile / "config.yaml").write_text("model:\n  default: test-model\n")
        fresh = write_fresh_profile_incarnation(profile)
    assert fresh != incarnation
    assert server._profile_home_rejected(profile, incarnation, require_incarnation=True)
    parent["profile_incarnation"] = fresh

    def build_branch(_parent, sid, key, history, source):
        record = {**parent, "session_key": key, "history": history, "source": source}
        server._sessions[sid] = record
        return SimpleNamespace()

    monkeypatch.setattr(server, "_build_branch_agent", build_branch)
    db = SessionDB(db_path=profile / "state.db", expected_profile_incarnation=fresh)
    try:
        db.create_session("parent", source="desktop")
        db.append_message("parent", "user", "current parent")
        response = request()
        assert "error" not in response, response
        sid = response["result"]["session_id"]
        assert sid != "old-runtime"
        assert server._sessions[sid]["profile_incarnation"] == fresh
        assert "old generation" not in str(response)
        assert request()["result"]["session_id"] == sid
        assert old["profile_incarnation"] == incarnation
        assert not (home / "state.db").exists()
        if operation == "branch":
            child = db.get_session(response["result"]["stored_session_id"])
            assert child["parent_session_id"] == "parent"
    finally:
        db.close()
