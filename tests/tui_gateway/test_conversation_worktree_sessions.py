"""Conversation-worktree ownership at desktop/TUI session boundaries."""

from __future__ import annotations

import contextlib
import threading
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import tui_gateway.server as server


@dataclass(frozen=True)
class _Binding:
    root_session_id: str
    path: Path
    branch: str
    base_commit: str
    repo_common_dir: Path


@pytest.fixture(autouse=True)
def _clean_sessions(monkeypatch):
    with server._sessions_lock:
        server._sessions.clear()
    monkeypatch.setattr(server, "_register_session_cwd", lambda _session: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(server, "_resolve_model", lambda: "test/model")
    monkeypatch.setattr(server.git_probe, "branch", lambda cwd: "stable")
    monkeypatch.setattr(server, "_project_info_for_cwd", lambda cwd: {})
    monkeypatch.setattr(server, "_load_show_reasoning", lambda: False)
    monkeypatch.setattr(server, "_load_tool_progress_mode", lambda: "compact")
    monkeypatch.setattr(
        server,
        "_acquire_conversation_root_lease",
        lambda _binding, *, surface: MagicMock(surface=surface),
    )
    yield
    with server._sessions_lock:
        server._sessions.clear()


def _binding(root: str) -> _Binding:
    return _Binding(
        root_session_id=root,
        path=Path("/repo/.worktrees") / root,
        branch=f"hermes/session/{root}",
        base_commit="a" * 40,
        repo_common_dir=Path("/repo/.git"),
    )


def test_session_project_repo_gets_its_own_worktree_policy(monkeypatch, tmp_path):
    from agent.conversation_worktree_policy import ConversationWorktreePolicy

    configured = tmp_path / "lunabot"
    selected = tmp_path / "hermes-agent"
    selected_common = tmp_path / "hermes-common"
    configured.mkdir()
    selected.mkdir()
    policy = ConversationWorktreePolicy(
        enabled=True,
        source_worktree=configured,
        worktree_root=tmp_path / "conversations",
    )

    monkeypatch.setattr(server.git_probe, "repo_root", lambda _cwd: str(selected))
    monkeypatch.setattr(
        server.git_probe,
        "common_repo_root",
        lambda cwd: str(tmp_path / ("lunabot-common" if str(cwd) == str(configured) else "hermes-common")),
    )

    routed = server._conversation_worktree_policy_for_session(policy, str(selected))

    assert routed.source_worktree == selected
    assert routed.worktree_root.parent == policy.worktree_root
    assert routed.worktree_root != policy.worktree_root
    assert routed.worktree_root.name.startswith("hermes-common-")


def test_alternate_project_root_stays_outside_both_repositories(monkeypatch, tmp_path):
    from agent.conversation_worktree_policy import ConversationWorktreePolicy

    configured = tmp_path / "lunabot"
    selected = tmp_path / "hermes-agent"
    configured.mkdir()
    selected.mkdir()
    policy = ConversationWorktreePolicy(
        enabled=True,
        source_worktree=configured,
        worktree_root=configured / ".worktrees",
    )

    monkeypatch.setattr(server.git_probe, "repo_root", lambda _cwd: str(selected))
    monkeypatch.setattr(
        server.git_probe,
        "common_repo_root",
        lambda cwd: str(configured if str(cwd) == str(configured) else selected),
    )
    monkeypatch.setattr(server, "get_hermes_home", lambda: tmp_path / ".hermes")

    routed = server._conversation_worktree_policy_for_session(policy, str(selected))

    assert routed.worktree_root is not None
    assert routed.worktree_root.is_relative_to(tmp_path / ".hermes")
    assert not routed.worktree_root.is_relative_to(configured)
    assert not routed.worktree_root.is_relative_to(selected)


def test_selected_linked_checkout_preserves_its_source_path(monkeypatch, tmp_path):
    from agent.conversation_worktree_policy import ConversationWorktreePolicy

    configured = tmp_path / "hermes-agent"
    selected = configured / ".worktrees" / "feature"
    configured.mkdir()
    selected.mkdir(parents=True)
    policy = ConversationWorktreePolicy(
        enabled=True,
        source_worktree=configured,
        worktree_root=tmp_path / "conversations",
    )

    monkeypatch.setattr(
        server.git_probe,
        "repo_root",
        lambda cwd: str(configured if str(cwd) == str(configured) else selected),
    )
    monkeypatch.setattr(server.git_probe, "common_repo_root", lambda _cwd: str(configured))
    monkeypatch.setattr(server, "get_hermes_home", lambda: tmp_path / ".hermes")

    routed = server._conversation_worktree_policy_for_session(policy, str(selected))

    assert routed.source_worktree == selected
    assert routed.worktree_root is not None
    assert routed.worktree_root.parent == policy.worktree_root


def test_alternate_project_namespace_uses_common_repository_identity(monkeypatch, tmp_path):
    from agent.conversation_worktree_policy import ConversationWorktreePolicy

    configured = tmp_path / "lunabot"
    selected_common = tmp_path / "hermes-agent"
    selected_checkout = selected_common / ".worktrees" / "conversation-old"
    configured.mkdir()
    selected_checkout.mkdir(parents=True)
    policy = ConversationWorktreePolicy(
        enabled=True,
        source_worktree=configured,
        worktree_root=tmp_path / "conversations",
    )

    monkeypatch.setattr(server.git_probe, "repo_root", lambda _cwd: str(selected_checkout))
    monkeypatch.setattr(
        server.git_probe,
        "common_repo_root",
        lambda cwd: str(configured if str(cwd) == str(configured) else selected_common),
    )

    routed = server._conversation_worktree_policy_for_session(policy, str(selected_checkout))

    assert routed.worktree_root is not None
    assert routed.source_worktree == selected_checkout
    assert routed.worktree_root.parent.name == "conversations"
    assert routed.worktree_root.name.startswith("hermes-agent-")


def test_alternate_project_root_inside_selected_repository_moves_outside_both(
    monkeypatch, tmp_path
):
    from agent.conversation_worktree_policy import ConversationWorktreePolicy

    configured = tmp_path / "lunabot"
    selected = tmp_path / "hermes-agent"
    configured.mkdir()
    selected.mkdir()
    policy = ConversationWorktreePolicy(
        enabled=True,
        source_worktree=configured,
        worktree_root=selected / ".worktrees",
    )

    monkeypatch.setattr(server.git_probe, "repo_root", lambda _cwd: str(selected))
    monkeypatch.setattr(
        server.git_probe,
        "common_repo_root",
        lambda cwd: str(configured if str(cwd) == str(configured) else selected),
    )
    monkeypatch.setattr(server, "get_hermes_home", lambda: tmp_path / ".hermes")

    routed = server._conversation_worktree_policy_for_session(policy, str(selected))

    assert routed.source_worktree == selected
    assert routed.worktree_root is not None
    assert routed.worktree_root.is_relative_to(tmp_path / ".hermes")
    assert not routed.worktree_root.is_relative_to(configured)
    assert not routed.worktree_root.is_relative_to(selected)


def test_alternate_linked_checkout_root_moves_outside_repository(monkeypatch, tmp_path):
    from agent.conversation_worktree_policy import ConversationWorktreePolicy

    configured = tmp_path / "lunabot"
    selected_common = tmp_path / "hermes-agent"
    selected_checkout = selected_common / ".worktrees" / "linked"
    configured.mkdir()
    selected_checkout.mkdir(parents=True)
    policy = ConversationWorktreePolicy(
        enabled=True,
        source_worktree=configured,
        worktree_root=selected_checkout / ".worktrees",
    )

    monkeypatch.setattr(server.git_probe, "repo_root", lambda _cwd: str(selected_checkout))
    monkeypatch.setattr(
        server.git_probe,
        "common_repo_root",
        lambda cwd: str(configured if str(cwd) == str(configured) else selected_common),
    )
    monkeypatch.setattr(
        server.git_probe,
        "run_git",
        lambda cwd, *_args: (
            f"worktree {selected_common}\nworktree {selected_checkout}\n"
            if str(cwd) == str(selected_common) else ""
        ),
    )
    monkeypatch.setattr(server, "get_hermes_home", lambda: tmp_path / ".hermes")

    routed = server._conversation_worktree_policy_for_session(policy, str(selected_checkout))

    assert routed.worktree_root is not None
    assert routed.worktree_root.is_relative_to(tmp_path / ".hermes")
    assert not routed.worktree_root.is_relative_to(selected_common)


def test_selected_git_repository_probe_failure_fails_closed(monkeypatch, tmp_path):
    from agent.conversation_worktree import ConversationWorktreeError
    from agent.conversation_worktree_policy import ConversationWorktreePolicy

    configured = tmp_path / "lunabot"
    selected = tmp_path / "hermes-agent"
    configured.mkdir()
    selected.mkdir()
    (selected / ".git").write_text("gitdir: ../hermes-agent.git\n", encoding="utf-8")
    policy = ConversationWorktreePolicy(
        enabled=True,
        source_worktree=configured,
        worktree_root=tmp_path / "conversations",
    )
    monkeypatch.setattr(server.git_probe, "repo_root", lambda _cwd: "")

    with pytest.raises(ConversationWorktreeError, match="could not be identified"):
        server._conversation_worktree_policy_for_session(policy, str(selected))


def test_session_create_defers_worktree_until_first_prompt(monkeypatch):
    calls: list[tuple[str, str]] = []
    scheduled: list[tuple[str, str]] = []

    def bind(root_session_id: str, *, profile_home=None, db=None):
        calls.append((root_session_id, "interactive"))
        return _binding(root_session_id)

    monkeypatch.setattr(server, "_bind_new_interactive_conversation_worktree", bind)
    monkeypatch.setattr(
        server,
        "_schedule_agent_build",
        lambda sid: scheduled.append((sid, server._sessions[sid]["cwd"])),
    )

    response = server._methods["session.create"](
        "create", {"source": "desktop", "cwd": "/stable"}
    )

    assert "error" not in response
    result = response["result"]
    root = result["stored_session_id"]
    assert calls == []
    assert result["info"]["cwd"] == server._completion_cwd({"cwd": "/stable"})
    assert scheduled == [(result["session_id"], server._completion_cwd({"cwd": "/stable"}))]
    record = server._sessions[result["session_id"]]
    assert record["cwd"] == server._completion_cwd({"cwd": "/stable"})
    assert record["explicit_cwd"] is False
    assert record["conversation_worktree"] == {}


def test_isolated_session_defers_agent_prewarm_until_worktree_binding(monkeypatch, tmp_path):
    started: list[str] = []

    class _Timer:
        def __init__(self, _delay, target):
            self.target = target

        def start(self):
            started.append("timer")
            self.target()

    monkeypatch.setattr(server, "_load_cfg", lambda: {
        "conversation_worktree": {
            "enabled": True,
            "source_worktree": str(tmp_path),
            "worktree_root": str(tmp_path / "worktrees"),
        },
    })
    monkeypatch.setattr(server.threading, "Timer", _Timer)
    monkeypatch.setattr(server, "_start_agent_build", lambda sid, _session: started.append(sid))

    session = {
        "source": "desktop",
        "conversation_worktree": {},
    }
    server._sessions["isolated"] = session
    server._schedule_agent_build("isolated")
    assert started == []

    session["conversation_worktree"] = {"path": str(tmp_path / "worktrees" / "isolated")}
    server._schedule_agent_build("isolated")
    assert started == ["timer", "isolated"]


def test_desktop_draft_has_no_root_lease(monkeypatch):
    lease = MagicMock()
    monkeypatch.setattr(
        server,
        "_acquire_conversation_root_lease",
        lambda _binding, *, surface: lease,
    )
    monkeypatch.setattr(
        server,
        "_bind_new_interactive_conversation_worktree",
        lambda root_session_id, **_kwargs: _binding(root_session_id),
    )
    monkeypatch.setattr(server, "_schedule_agent_build", lambda _sid: None)

    response = server._methods["session.create"](
        "create-lease", {"source": "desktop", "cwd": "/stable"}
    )
    sid = response["result"]["session_id"]
    session = server._sessions[sid]

    assert session["conversation_root_lease"] is None
    lease.release.assert_not_called()

    server._finalize_session(session)

    lease.release.assert_not_called()


def test_session_create_does_not_bind_worktree_for_a_draft(monkeypatch):
    scheduled: list[str] = []

    def fail(_root_session_id: str, *, profile_home=None, db=None):
        raise RuntimeError("bootstrap failed")

    monkeypatch.setattr(server, "_bind_new_interactive_conversation_worktree", fail)
    monkeypatch.setattr(server, "_schedule_agent_build", lambda sid: scheduled.append(sid))

    response = server._methods["session.create"]("create", {"source": "desktop"})

    assert "error" not in response
    assert scheduled == [response["result"]["session_id"]]
    assert server._sessions


def test_historical_resume_marks_unmanaged_and_never_binds_on_first_submit(monkeypatch):
    from tui_gateway.methods_session import _Resume

    ctx = _Resume("resume", {"source": "desktop"}, "legacy-session")
    ctx.conversation_worktree_historical = True
    record = ctx.record("desktop", "/legacy-workspace", [])

    calls = []
    monkeypatch.setattr(server, "_bind_conversation_worktree_for_new_root", lambda *a, **k: calls.append(a))
    session = {
        "source": "desktop", "session_key": "legacy-session",
        "conversation_worktree": {},
        "conversation_worktree_historical": record["conversation_worktree_historical"],
    }

    server._bind_conversation_worktree_on_submit(session)

    assert record["conversation_worktree_historical"] is True
    assert calls == []
    assert session["conversation_worktree"] == {}


def test_failed_seeded_binding_removes_ready_worktree_before_error(monkeypatch, tmp_path):
    binding = _binding("seeded-failure")
    lease = MagicMock()
    removed: list[tuple[str, bool]] = []

    class _DB:
        def update_session_cwd(self, *_args, **_kwargs):
            raise RuntimeError("metadata write failed")

    class _Manager:
        def remove_after_explicit_request(self, root, *, active_session_bound, retain_for_retry):
            removed.append((root, active_session_bound, retain_for_retry))
            return MagicMock(removed=True)

    session = {
        "source": "desktop", "session_key": "seeded-failure", "profile_home": None,
        "conversation_worktree": {}, "conversation_root_lease": None,
        "cwd": str(tmp_path), "explicit_cwd": False,
    }
    monkeypatch.setattr(server, "_session_db", lambda _session: contextlib.nullcontext(_DB()))
    monkeypatch.setattr(server, "_bind_conversation_worktree_for_new_root", lambda *a, **k: binding)
    monkeypatch.setattr(server, "_acquire_conversation_root_lease", lambda *a, **k: lease)
    monkeypatch.setattr(server, "_conversation_worktree_manager", lambda **_k: (_Manager(), None, False))

    with pytest.raises(RuntimeError, match="metadata write failed"):
        server._bind_conversation_worktree_on_submit(session)

    assert removed == [("seeded-failure", False, True)]
    lease.release.assert_called_once_with()


def test_enabled_isolation_does_not_treat_profile_cwd_as_historical(monkeypatch, tmp_path):
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / "historical-fallback.db")
    try:
        db.create_session("legacy-no-cwd", source="desktop")
        manager = MagicMock()
        manager.resolve_existing_session.return_value = None
        monkeypatch.setattr(server, "_get_db", lambda: db)
        monkeypatch.setattr(
            server, "_conversation_worktree_manager", lambda **_kw: (manager, db, False)
        )
        monkeypatch.setattr(server, "_profile_configured_cwd", lambda _home: str(tmp_path))

        response = server._methods["session.resume"]("resume", {
            "session_id": "legacy-no-cwd", "source": "desktop",
        })

        assert response["error"]["code"] == 5000
        assert "refusing profile checkout fallback" in response["error"]["message"]
    finally:
        db.close()


def test_resume_resolves_existing_binding_without_creation(monkeypatch):
    root = "root-existing"
    continuation = "compressed-tip"
    resolve_calls: list[str] = []
    create_calls: list[str] = []

    class _LineageDB:
        def is_explicit_fork_child(self, session_id):
            return False

        def get_session(self, session_id):
            return {"parent_session_id": root} if session_id == continuation else {}

    monkeypatch.setattr(
        server,
        "_resolve_existing_conversation_worktree",
        lambda root_session_id, *, profile_home=None, db=None: (
            resolve_calls.append(root_session_id)
            or (_binding(root_session_id) if root_session_id == root else None)
        ),
    )
    monkeypatch.setattr(
        server,
        "_bind_new_interactive_conversation_worktree",
        lambda root_session_id, *, profile_home=None, db=None: create_calls.append(root_session_id),
    )

    assert server._resolve_conversation_worktree_for_resume(
        continuation, profile_home=None, db=_LineageDB()
    ) == _binding(root)
    assert resolve_calls == [continuation, root]
    assert create_calls == []


def test_branch_binds_a_distinct_root_before_agent_construction(monkeypatch):
    calls: list[str] = []

    monkeypatch.setattr(
        server,
        "_bind_new_interactive_conversation_worktree",
        lambda root_session_id, *, profile_home=None, db=None: calls.append(root_session_id)
        or _binding(root_session_id),
    )

    binding = server._bind_conversation_worktree_for_new_root("branch-root", profile_home=None)

    assert binding.path == Path("/repo/.worktrees/branch-root")
    assert calls == ["branch-root"]


@pytest.mark.parametrize("mode", ["cold", "defer_history", "lazy", "eager_build"])
def test_resume_rpc_owns_branch_root_across_compression(monkeypatch, tmp_path, mode):
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / "resume.db")
    try:
        db.create_session("parent", source="desktop", cwd=str(tmp_path))
        db.create_session("branch", source="desktop", parent_session_id="parent",
                          model_config={"_branched_from": "parent"}, cwd=str(tmp_path))
        db.create_session("tip", source="desktop", parent_session_id="branch",
                          model_config={"_branched_from": "parent"}, cwd=str(tmp_path))
        manager = MagicMock()
        manager.resolve_existing_session.side_effect = lambda root: _binding(root) if root in {"branch", "parent"} else None
        monkeypatch.setattr(server, "_get_db", lambda: db)
        monkeypatch.setattr(server, "_conversation_worktree_manager", lambda **kw: (manager, db, False))
        monkeypatch.setattr(server, "_schedule_agent_build", lambda sid: None)
        monkeypatch.setattr(server, "_schedule_resume_hydration", lambda *a, **kw: None)
        monkeypatch.setattr(server, "_maybe_schedule_auto_continue", lambda *a: None)
        monkeypatch.setattr(server, "_make_agent_in_context", lambda *a, **kw: MagicMock())
        monkeypatch.setattr(server, "_wire_session_agent", lambda *a: None)
        monkeypatch.setattr(server, "_start_session_services", lambda *a: None)
        monkeypatch.setattr(server, "_schedule_mcp_late_refresh", lambda *a: None)
        monkeypatch.setattr(server, "_emit", lambda *a: None)
        monkeypatch.setattr(server, "_session_info", lambda agent, session=None: {"cwd": session["cwd"]})
        response = server._methods["session.resume"]("resume", {
            "session_id": "tip", "source": "desktop", **({mode: True} if mode != "cold" else {})})
        assert "error" not in response, response
        record = server._sessions[response["result"]["session_id"]]
        assert record["conversation_worktree"]["root_session_id"] == "branch"
        assert record["cwd"] == str(_binding("branch").path)
        record["conversation_root_lease"].release.assert_not_called()
        assert db.get_conversation_root("tip") == "parent"
        server._finalize_session(record)
        assert "conversation_root_lease" not in record
    finally:
        db.close()


@pytest.mark.parametrize("failure", ["missing", "lease", "history", "init"])
def test_resume_failure_releases_candidate_without_registering(monkeypatch, tmp_path, failure):
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / "failure.db")
    lease = MagicMock()
    try:
        db.create_session("parent", source="desktop")
        db.create_session("branch", source="desktop", parent_session_id="parent",
                          model_config={"_branched_from": "parent"}, cwd=str(tmp_path))
        manager = MagicMock()
        manager.resolve_existing_session.side_effect = lambda root: (
            None if failure == "missing" and root == "branch" else _binding(root))
        monkeypatch.setattr(server, "_get_db", lambda: db)
        monkeypatch.setattr(server, "_conversation_worktree_manager", lambda **kw: (manager, db, False))
        acquire = MagicMock(return_value=lease, side_effect=RuntimeError("lease unavailable") if failure == "lease" else None)
        monkeypatch.setattr(server, "_acquire_conversation_root_lease", acquire)
        if failure == "history":
            monkeypatch.setattr(db, "get_resume_conversations", MagicMock(side_effect=RuntimeError("read failed")))
        if failure == "init":
            monkeypatch.setattr(server, "_make_agent_in_context", lambda *a, **kw: MagicMock())

            def fail_init(sid, *args, **kwargs):
                server._sessions[sid] = {"conversation_root_lease": lease}
                raise RuntimeError("service initialization failed")

            monkeypatch.setattr(server, "_init_session", fail_init)
        response = server._methods["session.resume"]("resume-fail", {
            "session_id": "branch", "source": "desktop", "eager_build": failure == "init"})
        if failure == "missing":
            # Historical rows predate isolation; resume preserves their recorded
            # session instead of manufacturing a new binding or failing closed.
            assert "error" not in response, response
            record = server._sessions[response["result"]["session_id"]]
            assert record["conversation_worktree"] == {}
            acquire.assert_not_called()
            assert [call.args[0] for call in manager.resolve_existing_session.call_args_list] == ["branch"]
            return
        assert "error" in response
        assert server._sessions == {}
        if failure in {"history", "init"}:
            lease.release.assert_called_once_with()
    finally:
        db.close()


@pytest.mark.parametrize("persist_fails", [False, True])
def test_branch_rpc_stages_distinct_root_and_preserves_parent(monkeypatch, tmp_path, persist_fails):
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / "branch.db")
    lease = MagicMock()
    try:
        db.create_session("parent", source="desktop", cwd=str(tmp_path))
        parent = {"session_key": "parent", "source": "desktop", "cwd": str(tmp_path),
                  "explicit_cwd": True, "history_lock": threading.Lock(),
                  "history": [{"role": "user", "content": "work to fork"}]}
        monkeypatch.setattr(server, "_sess", lambda params, rid: (parent, None))
        monkeypatch.setattr(server, "_get_db", lambda: db)
        monkeypatch.setattr(server, "_bind_new_interactive_conversation_worktree", lambda root, **kw: _binding(root))
        monkeypatch.setattr(server, "_acquire_conversation_root_lease", lambda *a, **kw: lease)
        built = []

        def build(session, sid, key, history, source, **kwargs):
            assert kwargs["conversation_worktree"]["path"] != parent["cwd"]
            assert db.get_session(key)["cwd"] == kwargs["conversation_worktree"]["path"]
            built.append(key)
            server._sessions[sid] = {"conversation_root_lease": kwargs["conversation_root_lease"]}
            return MagicMock()

        monkeypatch.setattr(server, "_build_branch_agent", build)
        monkeypatch.setattr(server, "_session_info", lambda *a: {})
        if persist_fails:
            monkeypatch.setattr(db, "create_session", MagicMock(side_effect=RuntimeError("write failed")))
        response = server._methods["session.branch"]("branch", {"session_id": "live-parent"})
        assert parent["cwd"] == str(tmp_path)
        assert db.get_session("parent")["ended_at"] is None
        if persist_fails:
            assert "error" in response
            assert built == []
            lease.release.assert_called_once_with()
        else:
            assert "error" not in response, response
            assert built == [response["result"]["stored_session_id"]]
            lease.release.assert_not_called()
    finally:
        db.close()
