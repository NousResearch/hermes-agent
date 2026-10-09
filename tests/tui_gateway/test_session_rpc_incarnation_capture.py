"""One session RPC captures its profile's incarnation once, and binds only that generation.

A capture takes the profile's cross-process lease and reads its incarnation marker. ``session.resume``
(lazy) and ``session.create`` for a named profile resolved the same home several times per call: the
home resolver captured and validated a token then dropped it, the handler captured again, the live
record captured a third time, and the response's route and display name re-ran the whole resolver.
Each call now captures once and every later step reuses that token, so a profile deleted and
recreated between two steps of one RPC is refused as stale instead of half-bound to its successor.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli.profile_incarnation import write_fresh_profile_incarnation
from hermes_cli.profile_lifecycle import profile_lifecycle_lease
from hermes_state import SessionDB
from tui_gateway import server
from tui_gateway.profile_lifecycle import ProfileLifecycleFence

_STORED = "20261008_000000_abc123"
_OPS_CONFIG = "model:\n  default: ops-model\n  provider: openrouter\n"


class _CountingFence(ProfileLifecycleFence):
    def __init__(self) -> None:
        super().__init__()
        self.captures = 0

    def capture(self, profile_home):
        self.captures += 1
        return super().capture(profile_home)


@pytest.fixture
def ops(tmp_path, monkeypatch):
    """A named ``ops`` profile beside the launch home, holding one stored session; yields its home."""
    root = tmp_path / ".hermes"
    home = root / "profiles" / "ops"
    home.mkdir(parents=True)
    (home / "config.yaml").write_text(_OPS_CONFIG, encoding="utf-8")
    (root / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(server, "_hermes_home", str(root))
    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setattr(server, "_idempotency_keys", {})
    monkeypatch.setattr(server, "_served_profile_homes", set())
    monkeypatch.setattr(server, "_profile_lifecycle", _CountingFence())
    for name in ("_schedule_agent_build", "_schedule_session_cap_enforcement", "_enable_gateway_prompts"):
        monkeypatch.setattr(server, name, lambda *a, **k: None)
    monkeypatch.setattr(server, "_default_session_cwd", lambda *a, **k: str(tmp_path))
    db = SessionDB(db_path=home / "state.db", expected_profile_incarnation=server._capture_profile_incarnation(home))
    try:
        db.create_session(_STORED, "desktop")
        db.append_message(_STORED, "user", "hi")
    finally:
        db.close()
    return home


def _rpc(method: str, params: dict) -> dict:
    server._profile_lifecycle.captures = 0
    return server.handle_request({"id": "1", "method": method, "params": params})


_RESUME_LAZY = {"session_id": _STORED, "profile": "ops", "lazy": True, "omit_messages": True}


@pytest.mark.parametrize(("method", "params", "captures"), [
    ("session.resume", _RESUME_LAZY, 1),
    ("session.create", {"profile": "ops"}, 1),
    ("session.branch_stored", {"profile": "ops", "parent_session_id": _STORED}, 1),
    ("session.list", {"profile": "ops"}, 1),
    ("session.delete", {"profile": "ops", "session_id": _STORED}, 1),
    ("session.workspace.move", {"profile": "ops", "session_key": _STORED, "cwd": "."}, 1),
    # ``@_profile_scoped`` resolves once to bind the runtime scope, the handler once for its db and stamped name.
    ("projects.tree", {"profile": "ops"}, 2),
    ("projects.project_sessions", {"profile": "ops", "project_id": "none"}, 2),
], ids=["lazy-resume", "create", "branch-stored", "list", "delete", "workspace-move", "projects-tree",
        "project-sessions"])
def test_profile_rpcs_capture_the_incarnation_once_per_resolution(ops, monkeypatch, tmp_path, method, params,
                                                                   captures):
    monkeypatch.chdir(tmp_path)  # workspace.move's "." target

    out = _rpc(method, params)

    assert "error" not in out, out
    assert server._profile_lifecycle.captures == captures


@pytest.mark.parametrize(("method", "params"), [("session.resume", _RESUME_LAZY), ("session.create", {"profile": "ops"})],
                         ids=["lazy-resume", "create"])
def test_session_info_reports_the_resolved_profile(ops, method, params):
    out = _rpc(method, params)

    assert "error" not in out, out
    assert out["result"]["info"]["profile_name"] == "ops"
    assert out["result"]["info"]["model"] == "ops-model"


def _replace_generation(home: Path) -> None:
    """Another process deletes and recreates the profile under its lease: same pathname, fresh incarnation."""
    with profile_lifecycle_lease(home):
        home.rename(home.with_name("ops-retired"))
        home.mkdir()
        (home / "config.yaml").write_text(_OPS_CONFIG, encoding="utf-8")
        write_fresh_profile_incarnation(home)


@pytest.mark.parametrize(("method", "params", "hook"), [
    # The hook runs after the RPC resolved the profile and before it registers the live session.
    ("session.create", {"profile": "ops"}, "_enable_gateway_prompts"),
    ("session.resume", {"session_id": _STORED, "profile": "ops", "lazy": True}, "_todo_state_from_history"),
], ids=["create", "lazy-resume"])
def test_a_profile_replaced_mid_rpc_is_refused_as_stale(ops, monkeypatch, method, params, hook):
    monkeypatch.setattr(server, hook, lambda *a, **k: _replace_generation(ops))

    with pytest.raises(FileNotFoundError, match="missing or being deleted"):
        _rpc(method, params)

    assert server._sessions == {}
