from __future__ import annotations

import threading
import types

import pytest

from hermes_state import SessionDB
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tui_gateway import server
from tui_gateway.transport import TeeTransport


class _WebSocketTransport:
    def write(self, _obj: dict) -> bool:
        return True

    def close(self) -> None:
        return None


@pytest.fixture()
def gateway_db(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    token = set_hermes_home_override(home)
    db = SessionDB(db_path=home / "state.db")
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_resolve_model", lambda: "test/model")
    monkeypatch.setattr(server, "_schedule_agent_build", lambda *_args, **_kwargs: None)
    for session in list(server._sessions.values()):
        server._teardown_session(session)
    server._sessions.clear()
    try:
        yield db
    finally:
        for session in list(server._sessions.values()):
            server._teardown_session(session)
        server._sessions.clear()
        db.close()
        reset_hermes_home_override(token)


def test_stdio_session_persists_launch_cwd_on_first_prompt(gateway_db, monkeypatch, tmp_path):
    launch_cwd = tmp_path / "launch"
    launch_cwd.mkdir()
    monkeypatch.chdir(launch_cwd)
    monkeypatch.delenv("HERMES_CWD", raising=False)
    monkeypatch.setattr(server, "current_transport", lambda: server._stdio_transport)

    response = server._methods["session.create"]("create", {})
    sid = response["result"]["session_id"]
    session_key = response["result"]["stored_session_id"]

    assert gateway_db.get_session(session_key) is None

    server._ensure_session_db_row(server._sessions[sid])

    row = gateway_db.get_session(session_key)
    assert row is not None
    assert row["cwd"] == str(launch_cwd)


def test_stdio_session_uses_hermes_cwd_over_configured_terminal_cwd(
    gateway_db, monkeypatch, tmp_path
):
    configured_cwd = tmp_path / "configured"
    launch_cwd = tmp_path / "launch"
    configured_cwd.mkdir()
    launch_cwd.mkdir()
    monkeypatch.chdir(launch_cwd)
    monkeypatch.setenv("HERMES_CWD", str(launch_cwd))
    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {"terminal": {"cwd": str(configured_cwd)}},
    )
    monkeypatch.setattr(server, "current_transport", lambda: server._stdio_transport)

    response = server._methods["session.create"]("create", {})
    sid = response["result"]["session_id"]
    session_key = response["result"]["stored_session_id"]

    assert response["result"]["info"]["cwd"] == str(launch_cwd)

    server._ensure_session_db_row(server._sessions[sid])

    row = gateway_db.get_session(session_key)
    assert row is not None
    assert row["cwd"] == str(launch_cwd)


def test_tee_wrapped_stdio_session_uses_hermes_cwd_over_configured_terminal_cwd(
    gateway_db, monkeypatch, tmp_path
):
    configured_cwd = tmp_path / "configured"
    launch_cwd = tmp_path / "launch"
    configured_cwd.mkdir()
    launch_cwd.mkdir()
    monkeypatch.chdir(configured_cwd)
    monkeypatch.setenv("HERMES_CWD", str(launch_cwd))
    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {"terminal": {"cwd": str(configured_cwd)}},
    )
    tee = TeeTransport(server._stdio_transport, _WebSocketTransport())
    monkeypatch.setattr(server, "current_transport", lambda: tee)

    response = server._methods["session.create"]("create-tee", {})
    sid = response["result"]["session_id"]
    session_key = response["result"]["stored_session_id"]

    assert response["result"]["info"]["cwd"] == str(launch_cwd)

    server._ensure_session_db_row(server._sessions[sid])

    row = gateway_db.get_session(session_key)
    assert row is not None
    assert row["cwd"] == str(launch_cwd)


def test_websocket_session_without_cwd_stays_unbound_but_explicit_cwd_persists(
    gateway_db, monkeypatch, tmp_path
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    transport = _WebSocketTransport()
    monkeypatch.setattr(server, "current_transport", lambda: transport)

    unbound = server._methods["session.create"]("create-unbound", {})
    unbound_sid = unbound["result"]["session_id"]
    unbound_key = unbound["result"]["stored_session_id"]
    server._ensure_session_db_row(server._sessions[unbound_sid])

    explicit = server._methods["session.create"](
        "create-explicit", {"cwd": str(workspace)}
    )
    explicit_sid = explicit["result"]["session_id"]
    explicit_key = explicit["result"]["stored_session_id"]
    server._ensure_session_db_row(server._sessions[explicit_sid])

    assert gateway_db.get_session(unbound_key)["cwd"] is None
    assert gateway_db.get_session(explicit_key)["cwd"] == str(workspace)


def test_lazy_row_git_enrichment_runs_after_row_creation(gateway_db, monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(server, "current_transport", lambda: server._stdio_transport)

    started = threading.Event()
    completed = threading.Event()
    session_key = ""

    original_update = gateway_db.update_session_git_meta_if_cwd_matches

    def update_and_signal(*args, **kwargs):
        result = original_update(*args, **kwargs)
        completed.set()
        return result

    monkeypatch.setattr(
        gateway_db,
        "update_session_git_meta_if_cwd_matches",
        update_and_signal,
    )

    def branch(_cwd):
        if session_key:
            assert gateway_db.get_session(session_key) is not None
            started.set()
        return "task-02"

    monkeypatch.setattr(server, "_git_branch_for_cwd", branch)
    monkeypatch.setattr(server, "_git_common_repo_root_for_cwd", lambda _cwd: str(repo))

    response = server._methods["session.create"]("create", {"cwd": str(repo)})
    sid = response["result"]["session_id"]
    session_key = response["result"]["stored_session_id"]

    server._ensure_session_db_row(server._sessions[sid])

    assert started.wait(timeout=2)
    assert completed.wait(timeout=2)
    row = gateway_db.get_session(session_key)
    assert row is not None
    assert row["cwd"] == str(repo)
    assert row["git_branch"] == "task-02"
    assert row["git_repo_root"] == str(repo)


def test_async_git_enrichment_rejects_stale_cwd_after_session_move(
    gateway_db, monkeypatch, tmp_path
):
    cwd_a = tmp_path / "a"
    cwd_b = tmp_path / "b"
    cwd_a.mkdir()
    cwd_b.mkdir()
    monkeypatch.setattr(server, "current_transport", lambda: server._stdio_transport)

    a_started = threading.Event()
    release_a = threading.Event()
    a_update_attempted = threading.Event()
    b_update_completed = threading.Event()

    def branch(cwd):
        if cwd == str(cwd_a):
            a_started.set()
            assert release_a.wait(timeout=2)
            return "branch-a"
        assert cwd == str(cwd_b)
        return "branch-b"

    monkeypatch.setattr(server, "_git_branch_for_cwd", branch)
    monkeypatch.setattr(
        server,
        "_git_common_repo_root_for_cwd",
        lambda cwd: str(tmp_path / cwd.split("/")[-1]),
    )

    original_update = gateway_db.update_session_git_meta_if_cwd_matches

    def update_and_signal(session_key, cwd, *args, **kwargs):
        result = original_update(session_key, cwd, *args, **kwargs)
        if cwd == str(cwd_a):
            a_update_attempted.set()
        elif cwd == str(cwd_b):
            b_update_completed.set()
        return result

    monkeypatch.setattr(
        gateway_db,
        "update_session_git_meta_if_cwd_matches",
        update_and_signal,
    )

    response = server._methods["session.create"]("create", {"cwd": str(cwd_a)})
    sid = response["result"]["session_id"]
    session_key = response["result"]["stored_session_id"]
    session = server._sessions[sid]

    server._ensure_session_db_row(session)
    assert a_started.wait(timeout=2)

    try:
        server._set_session_cwd(session, str(cwd_b))
        assert b_update_completed.wait(timeout=2)
        row = gateway_db.get_session(session_key)
        assert row["cwd"] == str(cwd_b)
        assert row["git_branch"] == "branch-b"
        assert row["git_repo_root"] == str(cwd_b)

        release_a.set()
        assert a_update_attempted.wait(timeout=2)
        row = gateway_db.get_session(session_key)
        assert row["cwd"] == str(cwd_b)
        assert row["git_branch"] == "branch-b"
        assert row["git_repo_root"] == str(cwd_b)
    finally:
        release_a.set()


def test_resume_preserves_stored_cwd(gateway_db, monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session_key = "resume-cwd"
    gateway_db.create_session(session_key, source="cli", cwd=str(workspace))
    gateway_db.append_message(session_key, "user", "hello")
    monkeypatch.setattr(server, "current_transport", lambda: server._stdio_transport)

    response = server._methods["session.resume"](
        "resume", {"session_id": session_key}
    )

    sid = response["result"]["session_id"]
    assert server._sessions[sid]["cwd"] == str(workspace)
    assert response["result"]["info"]["cwd"] == str(workspace)


def test_session_cwd_set_lazy_response_does_not_probe_git_synchronously(
    gateway_db, monkeypatch, tmp_path
):
    repo = tmp_path / "repo"
    repo.mkdir()
    transport = _WebSocketTransport()
    monkeypatch.setattr(server, "current_transport", lambda: transport)

    created = server._methods["session.create"]("create", {})
    sid = created["result"]["session_id"]
    session_key = created["result"]["stored_session_id"]
    server._ensure_session_db_row(server._sessions[sid])

    rpc_thread = threading.get_ident()
    probe_started = threading.Event()
    enrichment_completed = threading.Event()

    def branch(cwd):
        assert threading.get_ident() != rpc_thread
        assert cwd == str(repo)
        probe_started.set()
        return "task-02"

    def root(cwd):
        assert threading.get_ident() != rpc_thread
        assert cwd == str(repo)
        return str(repo)

    monkeypatch.setattr(server, "_git_branch_for_cwd", branch)
    monkeypatch.setattr(server, "_git_common_repo_root_for_cwd", root)
    original_update = gateway_db.update_session_git_meta_if_cwd_matches

    def update_and_signal(*args, **kwargs):
        result = original_update(*args, **kwargs)
        enrichment_completed.set()
        return result

    monkeypatch.setattr(
        gateway_db,
        "update_session_git_meta_if_cwd_matches",
        update_and_signal,
    )

    response = server._methods["session.cwd.set"](
        "set-cwd", {"session_id": sid, "cwd": str(repo)}
    )

    assert response["result"]["cwd"] == str(repo)
    assert response["result"]["branch"] is None
    assert probe_started.wait(timeout=2)
    assert enrichment_completed.wait(timeout=2)
    row = gateway_db.get_session(session_key)
    assert row["cwd"] == str(repo)
    assert row["git_branch"] == "task-02"
    assert row["git_repo_root"] == str(repo)


def test_session_cwd_set_initialized_agent_does_not_probe_git_on_rpc_thread(
    gateway_db, monkeypatch, tmp_path
):
    repo = tmp_path / "repo"
    repo.mkdir()
    transport = _WebSocketTransport()
    monkeypatch.setattr(server, "current_transport", lambda: transport)

    created = server._methods["session.create"]("create", {})
    sid = created["result"]["session_id"]
    session_key = created["result"]["stored_session_id"]
    session = server._sessions[sid]
    server._ensure_session_db_row(session)
    session["agent"] = types.SimpleNamespace(
        model="test/model", provider="test", tools=[], reasoning_config={}
    )

    rpc_thread = threading.get_ident()
    probe_started = threading.Event()
    enrichment_completed = threading.Event()

    def branch(cwd):
        assert threading.get_ident() != rpc_thread
        assert cwd == str(repo)
        probe_started.set()
        return "task-02"

    def root(cwd):
        assert threading.get_ident() != rpc_thread
        assert cwd == str(repo)
        return str(repo)

    monkeypatch.setattr(server, "_git_branch_for_cwd", branch)
    monkeypatch.setattr(server, "_git_common_repo_root_for_cwd", root)
    original_update = gateway_db.update_session_git_meta_if_cwd_matches

    def update_and_signal(*args, **kwargs):
        result = original_update(*args, **kwargs)
        enrichment_completed.set()
        return result

    monkeypatch.setattr(
        gateway_db,
        "update_session_git_meta_if_cwd_matches",
        update_and_signal,
    )

    response = server._methods["session.cwd.set"](
        "set-cwd", {"session_id": sid, "cwd": str(repo)}
    )

    assert response["result"]["cwd"] == str(repo)
    assert response["result"]["branch"] is None
    assert probe_started.wait(timeout=2)
    assert enrichment_completed.wait(timeout=2)
    row = gateway_db.get_session(session_key)
    assert row["git_branch"] == "task-02"
    assert row["git_repo_root"] == str(repo)


def _run_project_workspace_callback_test(
    gateway_db, monkeypatch, tmp_path, *, initialized_agent
):
    workspace = tmp_path / ("agent-workspace" if initialized_agent else "lazy-workspace")
    workspace.mkdir()
    transport = _WebSocketTransport()
    monkeypatch.setattr(server, "current_transport", lambda: transport)

    created = server._methods["session.create"]("create", {})
    sid = created["result"]["session_id"]
    session_key = created["result"]["stored_session_id"]
    session = server._sessions[sid]
    server._ensure_session_db_row(session)
    if initialized_agent:
        session["agent"] = types.SimpleNamespace(
            model="test/model", provider="test", tools=[], reasoning_config={}
        )

    rpc_thread = threading.get_ident()
    probe_started = threading.Event()
    enrichment_completed = threading.Event()
    emitted = []
    monkeypatch.setattr(server, "_emit", lambda *args: emitted.append(args))

    def branch(cwd):
        assert threading.get_ident() != rpc_thread
        assert cwd == str(workspace)
        probe_started.set()
        return "project-branch"

    def root(cwd):
        assert threading.get_ident() != rpc_thread
        assert cwd == str(workspace)
        return str(workspace)

    monkeypatch.setattr(server, "_git_branch_for_cwd", branch)
    monkeypatch.setattr(server, "_git_common_repo_root_for_cwd", root)
    original_update = gateway_db.update_session_git_meta_if_cwd_matches

    def update_and_signal(*args, **kwargs):
        result = original_update(*args, **kwargs)
        enrichment_completed.set()
        return result

    monkeypatch.setattr(
        gateway_db,
        "update_session_git_meta_if_cwd_matches",
        update_and_signal,
    )

    server._apply_project_workspace(session_key, str(workspace))

    assert emitted
    info = emitted[-1][2]
    assert info["cwd"] == str(workspace)
    assert info["branch"] is None
    assert probe_started.wait(timeout=2)
    assert enrichment_completed.wait(timeout=2)


def test_project_workspace_callback_initialized_agent_does_not_probe_git_synchronously(
    gateway_db, monkeypatch, tmp_path
):
    _run_project_workspace_callback_test(
        gateway_db, monkeypatch, tmp_path, initialized_agent=True
    )


def test_project_workspace_callback_lazy_session_does_not_probe_git_synchronously(
    gateway_db, monkeypatch, tmp_path
):
    _run_project_workspace_callback_test(
        gateway_db, monkeypatch, tmp_path, initialized_agent=False
    )
