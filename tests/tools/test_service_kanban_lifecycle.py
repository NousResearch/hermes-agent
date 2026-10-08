"""Managed workers must be able to finish, or be refused before spawn."""

import json
from pathlib import Path

import pytest


@pytest.fixture
def worker_profile(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "worker"
    profile.mkdir(parents=True)
    (root / "config.yaml").write_text("{}")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_PROFILE", "worker")
    for key in (
        "HERMES_ALLOWED_TOOLSETS", "HERMES_KANBAN_TASK", "HERMES_KANBAN_DB",
        "HERMES_KANBAN_RUN_ID", "HERMES_KANBAN_CLAIM_LOCK", "HERMES_SESSION_ID",
    ):
        monkeypatch.delenv(key, raising=False)
    return root, profile


@pytest.mark.parametrize("allowed", [[], ["terminal"]])
@pytest.mark.parametrize("source", ["config", "environment"])
def test_dispatch_resolution_refuses_missing_lifecycle(worker_profile, monkeypatch, allowed, source):
    root, profile = worker_profile
    config = {"platform_toolsets": {"cli": ["terminal"]}}
    if source == "config":
        config["agent"] = {"allowed_toolsets": allowed}
    else:
        monkeypatch.setenv("HERMES_ALLOWED_TOOLSETS", ",".join(allowed))
    (profile / "config.yaml").write_text(json.dumps(config))

    from hermes_cli import kanban_db as kb
    from hermes_constants import get_hermes_home

    with pytest.raises(RuntimeError, match="kanban.*lifecycle"):
        kb._resolve_worker_cli_toolsets(str(profile))
    assert get_hermes_home() == root
    assert json.loads((profile / "config.yaml").read_text()) == config


def test_dispatch_resolution_does_not_fallback_on_bad_policy(worker_profile):
    root, profile = worker_profile
    (profile / "config.yaml").write_text('agent:\n  allowed_toolsets: {terminal: true}\n')
    from hermes_cli import kanban_db as kb
    from hermes_constants import get_hermes_home

    with pytest.raises(RuntimeError, match="refusing.*kanban"):
        kb._resolve_worker_cli_toolsets(str(profile))
    assert get_hermes_home() == root


def test_schema_refuses_worker_without_lifecycle(worker_profile, monkeypatch):
    _, profile = worker_profile
    (profile / "config.yaml").write_text('{}')
    monkeypatch.setenv("HERMES_HOME", str(profile))
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_worker")
    monkeypatch.setenv("HERMES_ALLOWED_TOOLSETS", "terminal")
    from model_tools import get_tool_definitions

    with pytest.raises(RuntimeError, match="kanban.*lifecycle"):
        get_tool_definitions(["terminal"], quiet_mode=True, skip_tool_search_assembly=True)


def test_schema_refuses_disabled_worker_lifecycle(worker_profile, monkeypatch):
    _, profile = worker_profile
    (profile / "config.yaml").write_text('{}')
    monkeypatch.setenv("HERMES_HOME", str(profile))
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_worker")
    from model_tools import get_tool_definitions

    with pytest.raises(RuntimeError, match="kanban.*lifecycle"):
        get_tool_definitions(
            ["terminal"], disabled_toolsets=["kanban"], quiet_mode=True,
            skip_tool_search_assembly=True,
        )


def test_default_spawn_refuses_policy_before_process_boundary(worker_profile, monkeypatch):
    root, profile = worker_profile
    (profile / "config.yaml").write_text(json.dumps({"agent": {"allowed_toolsets": ["terminal"]}}))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(root / "board.db"))
    from hermes_cli import kanban_db as kb

    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="denied-worker", assignee="worker")
        task = kb.claim_task(conn, task_id)

    # Only the external process boundary is replaced; profile/config resolution
    # and the real task record must refuse admission before even building argv.
    def process_boundary():
        pytest.fail("denied worker reached process construction")

    monkeypatch.setattr(kb, "_resolve_hermes_argv", process_boundary)
    with pytest.raises(RuntimeError, match="kanban.*lifecycle"):
        kb._default_spawn(task, str(profile))


@pytest.mark.parametrize("allowed", [["terminal", "kanban"], ["terminal", "hermes-cli"]])
def test_admitted_worker_can_finish_real_task(worker_profile, monkeypatch, allowed):
    root, profile = worker_profile
    (profile / "config.yaml").write_text(json.dumps({
        "agent": {"allowed_toolsets": allowed},
        "platform_toolsets": {"cli": ["terminal"]},
    }))
    from hermes_cli import kanban_db as kb
    from tools.registry import invalidate_check_fn_cache, registry
    from model_tools import _clear_tool_defs_cache, get_tool_definitions
    from toolsets import TOOLSETS, resolve_toolset

    resolved = kb._resolve_worker_cli_toolsets(str(profile))
    assert "terminal" in resolved
    monkeypatch.setenv("HERMES_HOME", str(profile))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(root / "board.db"))
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="managed-worker", assignee="worker")
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(claimed.current_run_id))
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", claimed.claim_lock)
    invalidate_check_fn_cache()
    _clear_tool_defs_cache()
    try:
        tools = get_tool_definitions(resolved, quiet_mode=True, skip_tool_search_assembly=True)
        assert {"kanban_complete", "kanban_block", "kanban_heartbeat"} <= {
            tool["function"]["name"] for tool in tools
        }
        assert {tool["function"]["name"] for tool in tools} <= {
            name for toolset in allowed for name in resolve_toolset(toolset)
        }
        result = json.loads(registry.dispatch("kanban_complete", {"summary": "finished"}))
        assert result["ok"] is True
        with kb.connect_closing() as conn:
            assert kb.get_task(conn, task_id).status == "done"
            assert kb.latest_run(conn, task_id).outcome == "completed"

        # Narrowing a permitted composite keeps the worker's terminal
        # lifecycle but must not reuse schemas for broader kanban effects.
        lifecycle = {"kanban_complete", "kanban_block", "kanban_heartbeat"}
        monkeypatch.setitem(TOOLSETS, "test-worker-lifecycle", {
            "tools": sorted(lifecycle), "includes": [],
        })
        monkeypatch.setenv("HERMES_ALLOWED_TOOLSETS", "terminal,test-worker-lifecycle")
        narrowed = get_tool_definitions(resolved, quiet_mode=True, skip_tool_search_assembly=True)
        assert lifecycle <= {tool["function"]["name"] for tool in narrowed}
        assert {tool["function"]["name"] for tool in narrowed} <= (
            lifecycle | set(resolve_toolset("terminal"))
        )

        # A cached schema must not authorize a later denied worker policy.
        monkeypatch.setenv("HERMES_ALLOWED_TOOLSETS", "terminal")
        with pytest.raises(RuntimeError, match="kanban.*lifecycle"):
            get_tool_definitions(resolved, quiet_mode=True, skip_tool_search_assembly=True)
    finally:
        invalidate_check_fn_cache()
        _clear_tool_defs_cache()
