"""Regression tests for delegate_task isolation from parent Kanban workers."""
from __future__ import annotations

import json


def _make_running_kanban_task(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    attachments_root = tmp_path / "attachments"
    workspace = tmp_path / "parent-workspace"
    workspace.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "parent-worker")
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(workspace))
    monkeypatch.setenv("HERMES_KANBAN_ATTACHMENTS_ROOT", str(attachments_root))

    from hermes_cli import kanban_db as kb

    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="parent",
            assignee="parent-worker",
            workspace_kind="scratch",
            workspace_path=str(workspace),
        )
        claim = kb.claim_task(conn, tid)
        assert claim is not None
        run_id = claim.id
    finally:
        conn.close()

    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    return kb, tid, workspace, attachments_root


def test_delegated_child_context_suppresses_env_gated_kanban_tools(monkeypatch, tmp_path):
    """A delegate_task child must not inherit the parent's Kanban tool schema.

    The parent process may be a dispatcher worker with HERMES_KANBAN_TASK set;
    the child is only a subagent, not the run owner.
    """
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_parent")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "123")
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    import tools.kanban_tools  # noqa: F401 - ensure registered
    from agent.delegation_context import delegated_child_context
    from model_tools import _clear_tool_defs_cache, get_tool_definitions
    from tools.registry import invalidate_check_fn_cache

    invalidate_check_fn_cache()
    _clear_tool_defs_cache()
    with delegated_child_context():
        schema = get_tool_definitions(enabled_toolsets=["terminal"], quiet_mode=True)

    names = {s["function"].get("name") for s in schema if "function" in s}
    assert "terminal" in names
    assert {n for n in names if n and n.startswith("kanban_")} == set()


def test_build_child_agent_strips_kanban_toolset_even_when_parent_is_worker(monkeypatch):
    """Child construction must fail closed even if the parent exposes kanban."""
    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.valid_tool_names = {"terminal"}
            self.session_id = "child-session"

    import run_agent
    from tools import delegate_tool

    monkeypatch.setattr(run_agent, "AIAgent", FakeAgent)
    monkeypatch.setattr(delegate_tool, "_load_config", lambda: {})

    class Parent:
        enabled_toolsets = ["terminal", "kanban"]
        valid_tool_names = {"terminal", "kanban_complete", "kanban_comment"}
        model = "test-model"
        provider = "test-provider"
        base_url = "http://example.invalid"
        api_mode = "chat_completions"
        platform = "cli"
        session_id = "parent-session"

    child = delegate_tool._build_child_agent(
        task_index=0,
        goal="review only",
        context=None,
        toolsets=None,
        model=None,
        max_iterations=3,
        task_count=1,
        parent_agent=Parent(),
    )

    assert child.valid_tool_names == {"terminal"}
    assert "kanban" not in captured["enabled_toolsets"]
    assert "kanban" in captured["disabled_toolsets"]


def test_delegate_child_terminal_env_scrubs_parent_kanban_keys(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_parent")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "123")
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", "/tmp/parent-workspace")
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", "lock")

    from agent.delegation_context import delegated_child_context
    from tools.environments.local import _sanitize_subprocess_env

    with delegated_child_context():
        env = _sanitize_subprocess_env({
            "HERMES_KANBAN_TASK": "t_parent",
            "HERMES_KANBAN_RUN_ID": "123",
            "HERMES_KANBAN_WORKSPACE": "/tmp/parent-workspace",
            "HERMES_KANBAN_CLAIM_LOCK": "lock",
            "PATH": "/usr/bin",
        })

    assert env["PATH"] == "/usr/bin"
    assert "HERMES_KANBAN_TASK" not in env
    assert "HERMES_KANBAN_RUN_ID" not in env
    assert "HERMES_KANBAN_WORKSPACE" not in env
    assert "HERMES_KANBAN_CLAIM_LOCK" not in env


def test_delegate_child_kanban_mutator_guard_rejects_explicit_task_id(monkeypatch):
    """Defense in depth: direct handler access still cannot mutate a board."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_parent")
    from agent.delegation_context import delegated_child_context
    from tools import kanban_tools

    with delegated_child_context():
        raw = kanban_tools._handle_complete({
            "task_id": "t_parent",
            "summary": "should not complete",
        })

    payload = json.loads(raw)
    assert payload["error"]
    assert "delegate_task child" in payload["error"]


def test_delegate_child_attach_guard_leaves_no_row_or_file(monkeypatch, tmp_path):
    kb, tid, _workspace, attachments_root = _make_running_kanban_task(monkeypatch, tmp_path)

    from agent.delegation_context import delegated_child_context
    from tools import kanban_tools

    with delegated_child_context():
        raw = kanban_tools._handle_attach({
            "task_id": tid,
            "filename": "leak.txt",
            "content_base64": "bGVhay1ieXRlcw==",
            "content_type": "text/plain",
        })

    payload = json.loads(raw)
    assert payload["error"]
    assert "delegate_task child" in payload["error"]

    conn = kb.connect()
    try:
        assert kb.list_attachments(conn, tid) == []
    finally:
        conn.close()
    task_dir = attachments_root / tid
    assert not task_dir.exists() or list(task_dir.iterdir()) == []


def test_delegate_child_attach_url_guard_leaves_no_row_or_file(monkeypatch, tmp_path):
    kb, tid, _workspace, attachments_root = _make_running_kanban_task(monkeypatch, tmp_path)

    from agent.delegation_context import delegated_child_context
    from tools import kanban_tools

    def forbidden_download(*_args, **_kwargs):
        raise AssertionError("delegated child guard must run before URL download")

    monkeypatch.setattr(kanban_tools, "_download_url_with_cap", forbidden_download)

    with delegated_child_context():
        raw = kanban_tools._handle_attach_url({
            "task_id": tid,
            "url": "https://example.com/leak.txt",
        })

    payload = json.loads(raw)
    assert payload["error"]
    assert "delegate_task child" in payload["error"]

    conn = kb.connect()
    try:
        assert kb.list_attachments(conn, tid) == []
    finally:
        conn.close()
    task_dir = attachments_root / tid
    assert not task_dir.exists() or list(task_dir.iterdir()) == []


def test_child_attempting_default_complete_does_not_finish_parent_or_delete_workspace(
    monkeypatch,
    tmp_path,
):
    """Deterministic E2E: a delegated child cannot complete its parent task."""
    kb, tid, workspace, _attachments_root = _make_running_kanban_task(monkeypatch, tmp_path)
    from tools import delegate_tool
    from tools import kanban_tools

    class Parent:
        _current_task_id = tid

        def _touch_activity(self, _desc):
            return None

    class Child:
        tool_progress_callback = None
        _delegate_saved_tool_names = []
        _credential_pool = None
        _subagent_id = "sa-test"
        _delegate_depth = 1
        _parent_subagent_id = None
        model = "test-model"
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_estimated_cost_usd = 0.0
        session_reasoning_tokens = 0

        def get_activity_summary(self):
            return {"api_call_count": 0, "max_iterations": 1, "current_tool": None}

        def run_conversation(self, user_message, task_id, **_kwargs):
            attempted = kanban_tools._handle_complete({"summary": "wrong child completion"})
            return {
                "final_response": attempted,
                "completed": True,
                "api_calls": 0,
                "messages": [],
            }

        def close(self):
            return None

    result = delegate_tool._run_single_child(0, "try to complete parent", Child(), Parent())

    conn = kb.connect()
    try:
        task = kb.get_task(conn, tid)
        run = kb.latest_run(conn, tid)
    finally:
        conn.close()

    assert result["status"] == "completed"
    assert "delegate_task child" in result["summary"]
    assert task.status == "running"
    assert run.status == "running"
    assert workspace.is_dir()
