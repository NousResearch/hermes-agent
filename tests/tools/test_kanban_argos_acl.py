"""Argos' dispatcher-worker grant is narrower than the generic Kanban toolset.

Real registry/SQLite and profile-home switching; no external DNS or HTTP calls.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.fixture
def board(tmp_path, monkeypatch):
    root = tmp_path / "home"
    root.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "board.db"))
    monkeypatch.setenv("HERMES_KANBAN_ATTACHMENTS_ROOT", str(tmp_path / "attachments"))
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACES_ROOT", str(tmp_path / "workspaces"))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")
    monkeypatch.setenv("HERMES_PROFILE", "kratos")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    for name in ("argos", "atena"):
        (root / "profiles" / name).mkdir(parents=True)
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    kb._INITIALIZED_PATHS.clear()
    with kbc.connect_closing() as conn:
        root_id = kb.create_task(conn, title="direct handoff", assignee="argos", created_by="kratos")
        kb.claim_task(conn, root_id)
        run_id = kb._current_run_id(conn, root_id)
    return root, root_id, run_id


def _call(name, args):
    from tools.registry import registry
    return json.loads(registry.dispatch(name, args))


def test_argos_create_acl_and_direct_handoff_across_profiles(board, monkeypatch, tmp_path):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    root, root_id, run_id = board
    # A parent project must never turn Argos' children into worktrees.
    with kbc.connect_closing() as conn:
        conn.execute("UPDATE tasks SET project_id = ? WHERE id = ?", ("parent-project", root_id))
        collision = kb.create_task(conn, title="existing unrelated key", assignee="kratos",
                                   idempotency_key="once")
    monkeypatch.setenv("HERMES_KANBAN_TASK", root_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    monkeypatch.setenv("HERMES_PROFILE", "kratos")  # multiplex launch profile must not win
    token = set_hermes_home_override(root / "profiles" / "argos")
    try:
        from model_tools import get_tool_definitions
        names = {s["function"]["name"] for s in get_tool_definitions(
            ["kanban"], quiet_mode=True, skip_tool_search_assembly=True)}
        assert {"kanban_create", "kanban_attach_url", "kanban_link"} <= names
        assert not {"kanban_list", "kanban_unblock"} & names
        assert "error" in _call("kanban_list", {})
        assert "error" in _call("kanban_unblock", {"task_id": root_id})
        bad = (
            {"assignee": "kratos"}, {"assignee": "third-party"},
            {"workspace_kind": "dir", "workspace_path": str(tmp_path)},
            {"workspace_kind": "worktree", "workspace_path": str(tmp_path)},
            {"workspace_path": str(tmp_path)}, {"board": "foreign"}, {"board": "default"},
            {"project": "other"}, {"project_id": "other"},
            {"model": "alternate"}, {"provider": "alternate", "model": "alternate"},
            {"skills": ["web"]}, {"goal_mode": True}, {"triage": True},
            {"completion_contract": "owner/repo"}, {"session_id": "foreign"},
            {"tenant": "foreign"},
        )
        with kbc.connect_closing() as conn:
            before = len(kb.list_tasks(conn))
        for override in bad:
            out = _call("kanban_create", {"title": "forbidden", "assignee": "hefesto", **override})
            assert "error" in out, (override, out)
        # A direct handler call must enforce exactly the same boundary as registry dispatch.
        from tools.kanban_tools import _handle_create
        assert "error" in json.loads(_handle_create({"title": "forbidden", "assignee": "kratos"}))
        with kbc.connect_closing() as conn:
            assert len(kb.list_tasks(conn)) == before
        child_args = {"title": "implement", "assignee": "hefesto", "idempotency_key": "once"}
        child = _call("kanban_create", child_args)
        assert child["ok"] and child["task_id"] != collision, child
        assert _call("kanban_create", child_args)["task_id"] == child["task_id"]
        reviewer = _call("kanban_create", {"title": "review", "assignee": "atena",
                                           "parents": [child["task_id"]]})
        assert reviewer["ok"] and reviewer["gated"]
        assert _call("kanban_link", {"parent_id": reviewer["task_id"], "child_id": root_id})["ok"]
        assert _call("kanban_block", {"reason": "await review", "kind": "dependency"})["ok"]
        with kbc.connect_closing() as conn:
            implementation = kb.get_task(conn, child["task_id"])
            review = kb.get_task(conn, reviewer["task_id"])
            assert (implementation.assignee, review.assignee) == ("hefesto", "atena")
            assert all(task.workspace_kind == "scratch" and task.project_id is None
                       for task in (implementation, review))
            assert review.status == "todo" and kb.get_task(conn, root_id).status == "todo"
            kb.claim_task(conn, child["task_id"])
            kb.complete_task(conn, child["task_id"], summary="implemented",
                             expected_run_id=kb._current_run_id(conn, child["task_id"]))
            assert kb.get_task(conn, reviewer["task_id"]).status == "ready"
            kb.claim_task(conn, reviewer["task_id"])
            kb.complete_task(conn, reviewer["task_id"], summary="reviewed",
                             expected_run_id=kb._current_run_id(conn, reviewer["task_id"]))
            assert kb.get_task(conn, root_id).status == "ready"
    finally:
        reset_hermes_home_override(token)

    # A→B→A in one process: another worker retains its generic tools, but the
    # Argos grant remains narrow on return (no process-global profile cache).
    with kbc.connect_closing() as conn:
        b_id = kb.create_task(conn, title="other", assignee="atena")
        kb.claim_task(conn, b_id)
    monkeypatch.setenv("HERMES_KANBAN_TASK", b_id)
    token = set_hermes_home_override(root / "profiles" / "atena")
    try:
        other = _call("kanban_create", {"title": "unrestricted worker", "assignee": "kratos",
                                        "workspace_kind": "dir", "workspace_path": str(tmp_path)})
        assert other["ok"], other
    finally:
        reset_hermes_home_override(token)
    monkeypatch.setenv("HERMES_KANBAN_TASK", root_id)
    token = set_hermes_home_override(root / "profiles" / "argos")
    try:
        assert "error" in _call("kanban_create", {"title": "still forbidden", "assignee": "kratos"})
    finally:
        reset_hermes_home_override(token)
    monkeypatch.delenv("HERMES_KANBAN_TASK")
    monkeypatch.setenv("HERMES_PROFILE", "kratos")
    assert "kanban_list" in {s["function"]["name"] for s in get_tool_definitions(
        ["kanban"], quiet_mode=True, skip_tool_search_assembly=True)}
    assert isinstance(_call("kanban_list", {}).get("tasks"), list)


def test_argos_public_attachment_rejects_private_redirect_and_token_without_network(board, monkeypatch):
    from tools import kanban_tools as kt, url_safety
    root, task_id, run_id = board
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    token = set_hermes_home_override(root / "profiles" / "argos")
    calls = []

    def dns(host, *args, **kwargs):
        address = {"files.example": "93.184.216.34", "private.example": "127.0.0.1"}.get(host, host)
        return [(2, 1, 6, "", (address, 443))]

    redirect = [True]

    class Response:
        headers = {"location": "http://private.example/file", "content-type": "text/plain"}

        @property
        def is_redirect(self): return redirect[0]
        def raise_for_status(self): pass
        def iter_bytes(self, size): yield b"public bytes"
        def __enter__(self): return self
        def __exit__(self, *args): return False

    class Client:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def stream(self, method, url, **kwargs):
            calls.append(url)
            return Response()

    monkeypatch.setattr(url_safety, "_getaddrinfo", dns)
    # Exercise the real httpx transport wiring without sending a request.
    with url_safety.create_ssrf_safe_client(public_only=True, trust_env=False) as guarded:
        assert guarded._transport._pool._network_backend._public_only is True
    monkeypatch.setattr(url_safety, "create_ssrf_safe_client", lambda **kw: Client())
    # The downloader binds the factory by import at invocation, no real HTTP.
    try:
        monkeypatch.setenv("HERMES_ALLOW_PRIVATE_URLS", "true")
        for url in ("http://127.0.0.1/file", "http://private.example/file",
                    "https://files.example/file?token=hidden", "https://user:secret@files.example/file"):
            out = _call("kanban_attach_url", {"url": url})
            assert "error" in out, out
            assert calls == []
        out = _call("kanban_attach_url", {"url": "https://files.example/file"})
        assert "error" in out and "SSRF" in out["error"]
        assert calls == ["https://files.example/file"]
        assert not url_safety.is_public_url("http://private.example/file")
        assert url_safety.is_public_url("https://files.example/file")
        with pytest.raises(url_safety.SSRFConnectionBlocked):
            url_safety._resolved_http_connect_ips("private.example", 80, "http", public_only=True)
        from hermes_cli import kanban_db_connect as kbc, kanban_db as kb
        with kbc.connect_closing() as conn:
            assert kb.list_attachments(conn, task_id) == []
        redirect[0] = False
        out = _call("kanban_attach_url", {"url": "https://files.example/file"})
        assert out.get("ok") and out["size"] == len(b"public bytes"), out
        with kbc.connect_closing() as conn:
            assert len(kb.list_attachments(conn, task_id)) == 1
        with pytest.raises(ValueError, match="exceeds"):
            kt._download_url_with_cap("https://files.example/file", 2, public_only=True)
    finally:
        reset_hermes_home_override(token)
