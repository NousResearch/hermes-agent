"""Approval-gate regressions for detached Kanban card-owner workers.

The dispatcher runs workers in a separate process.  They must use the durable
Kanban broker instead of the gateway's process-local callback queue, and an
ambient parent ``--yolo`` marker must not become worker authority.
"""

from __future__ import annotations

import json
import subprocess
from unittest.mock import patch

import pytest

from tools import approval


@pytest.fixture(autouse=True)
def _kanban_owner(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_SESSION", "1")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-1")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "7")
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", "host:claim")
    monkeypatch.setenv("HERMES_PROFILE", "worker")
    monkeypatch.setenv("HERMES_SESSION_ID", "session-1")
    monkeypatch.delenv("HERMES_KANBAN_APPROVAL_ID", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_APPROVAL_NONCE", raising=False)
    monkeypatch.setattr(approval, "_is_kanban_owner_context", lambda: True)
    monkeypatch.setattr(approval, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(approval, "_get_kanban_approval_mode", lambda: "ask")


def test_dangerous_command_parks_worker_in_durable_broker(monkeypatch):
    parked = {
        "id": "apr_123",
        "display_target": "rm -rf /tmp/stuff",
    }
    park = patch.object(approval, "_request_kanban_approval", return_value=parked)
    gateway_wait = patch.object(
        approval,
        "_await_gateway_decision",
        side_effect=AssertionError("detached worker used process-local queue"),
    )
    with park as mock_park, gateway_wait:
        result = approval.check_all_command_guards("rm -rf /tmp/stuff", "local")

    assert result["approved"] is False
    assert result["status"] == "kanban_approval_pending"
    assert result["kanban_approval_pending"] is True
    assert result["request_id"] == "apr_123"
    mock_park.assert_called_once()


def test_global_smart_approval_cannot_bypass_kanban_ask_policy(monkeypatch):
    monkeypatch.setattr(approval, "_get_approval_mode", lambda: "smart")
    smart = patch.object(approval, "_smart_approve", return_value="approve")
    park = patch.object(
        approval,
        "_request_kanban_approval",
        return_value={"id": "apr_smart_terminal"},
    )

    with smart as mock_smart, park as mock_park:
        result = approval.check_all_command_guards(
            "rm -rf /tmp/smart-owner",
            "local",
        )

    assert result["approved"] is False
    assert result["status"] == "kanban_approval_pending"
    assert result["request_id"] == "apr_smart_terminal"
    mock_smart.assert_not_called()
    mock_park.assert_called_once()


def test_inherited_process_yolo_cannot_authorize_card_owner(monkeypatch):
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", True)
    monkeypatch.setattr(
        approval,
        "_request_kanban_approval",
        lambda **_kwargs: {"id": "apr_yolo"},
    )

    result = approval.check_all_command_guards("rm -rf /tmp/stuff", "local")

    assert result["approved"] is False
    assert result["status"] == "kanban_approval_pending"


def test_inherited_permanent_allowlist_cannot_authorize_card_owner(monkeypatch):
    monkeypatch.setattr(
        approval, "_command_matches_permanent_allowlist", lambda _command: True
    )
    monkeypatch.setattr(
        approval,
        "_request_kanban_approval",
        lambda **_kwargs: {"id": "apr_allowlist"},
    )

    result = approval.check_all_command_guards("rm -rf /tmp/stuff", "local")

    assert result["approved"] is False
    assert result["status"] == "kanban_approval_pending"


@pytest.mark.parametrize("approval_scope", ["session", "permanent"])
def test_stored_pattern_approval_cannot_authorize_card_owner(
    monkeypatch,
    approval_scope,
):
    command = "rm -rf /tmp/stored-pattern"
    dangerous, pattern_key, _description = approval.detect_dangerous_command(command)
    assert dangerous and pattern_key
    monkeypatch.setattr(approval, "_permanent_approved", set())
    monkeypatch.setattr(approval, "_session_approved", {})
    if approval_scope == "permanent":
        approval.approve_permanent(pattern_key)
    else:
        approval.approve_session(approval.get_current_session_key(), pattern_key)
    monkeypatch.setattr(
        approval,
        "_request_kanban_approval",
        lambda **_kwargs: {"id": f"apr_{approval_scope}"},
    )

    result = approval.check_all_command_guards(command, "local")

    assert result["approved"] is False
    assert result["status"] == "kanban_approval_pending"
    assert result["request_id"] == f"apr_{approval_scope}"


@pytest.mark.parametrize("mode, approved", [("approve", True), ("deny", False)])
def test_explicit_kanban_policy_controls_unattended_warning(
    monkeypatch, mode, approved
):
    monkeypatch.setattr(approval, "_get_kanban_approval_mode", lambda: mode)
    request = patch.object(
        approval,
        "_request_kanban_approval",
        side_effect=AssertionError("explicit policy should not create a request"),
    )
    with request:
        result = approval.check_all_command_guards("rm -rf /tmp/stuff", "local")

    assert result["approved"] is approved
    if not approved:
        assert result["outcome"] == "denied"


def test_safe_command_does_not_create_approval_under_deny_policy(monkeypatch):
    monkeypatch.setattr(approval, "_get_kanban_approval_mode", lambda: "deny")
    request = patch.object(
        approval,
        "_request_kanban_approval",
        side_effect=AssertionError("safe action should not create a request"),
    )

    with request:
        result = approval.check_all_command_guards("printf 'hello\\n'", "local")

    assert result == {"approved": True, "message": None}


def test_bound_exact_grant_is_consumed_once(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_APPROVAL_ID", "apr_bound")
    monkeypatch.setenv("HERMES_KANBAN_APPROVAL_NONCE", "nonce_bound")
    consume = patch.object(approval, "_consume_kanban_approval", return_value=True)
    request = patch.object(
        approval,
        "_request_kanban_approval",
        side_effect=AssertionError("bound grant should not create a new request"),
    )

    with consume as mock_consume, request:
        result = approval.check_all_command_guards(
            "rm -rf /tmp/stuff", "local", workdir="/approved/worktree"
        )

    assert result["approved"] is True
    assert result["user_approved"] is True
    mock_consume.assert_called_once_with(
        action_kind="terminal",
        raw_action="rm -rf /tmp/stuff",
        env_type="local",
        has_host_access=False,
        workdir="/approved/worktree",
        execution_context=None,
    )


def test_exact_action_digest_binds_resolved_workdir():
    common = {
        "action_kind": "terminal",
        "raw_action": "rm -rf /tmp/stuff",
        "env_type": "local",
    }

    approved = approval._kanban_action_digest(
        **common,
        workdir="/approved/worktree",
    )
    changed = approval._kanban_action_digest(
        **common,
        workdir="/different/worktree",
    )

    assert approved != changed


def test_exact_action_digest_binds_backend_and_invocation_mode():
    common = {
        "action_kind": "terminal",
        "raw_action": "rm -rf /tmp/stuff",
        "env_type": "ssh",
        "workdir": "/srv/app",
    }
    host_a = {
        "background": False,
        "pty": False,
        "timeout": 60,
        "target": {"host": "host-a", "user": "deploy", "port": 22},
    }
    host_b = {
        **host_a,
        "target": {"host": "host-b", "user": "deploy", "port": 22},
    }
    background = {**host_a, "background": True}

    approved = approval._kanban_action_digest(
        **common,
        execution_context=host_a,
    )

    assert approved != approval._kanban_action_digest(
        **common,
        execution_context=host_b,
    )
    assert approved != approval._kanban_action_digest(
        **common,
        execution_context=background,
    )


def test_docker_approval_context_persists_only_resolved_environment_hash(
    monkeypatch,
):
    from tools import env_passthrough
    from tools.terminal_tool import _terminal_approval_execution_context

    explicit_secret = "explicit-value-must-not-persist"
    forwarded_secret = "forwarded-value-must-not-persist"
    implicit_secret = "implicit-value-must-not-persist"
    monkeypatch.setenv("FORWARDED_SECRET", forwarded_secret)
    monkeypatch.setenv("IMPLICIT_SECRET", implicit_secret)
    monkeypatch.setattr(
        env_passthrough,
        "get_all_passthrough",
        lambda: frozenset({"IMPLICIT_SECRET"}),
    )
    monkeypatch.setattr(
        "tools.environments.docker._load_hermes_env_vars",
        lambda: {},
    )

    context = _terminal_approval_execution_context(
        config={
            "env_type": "docker",
            "docker_env": {"EXPLICIT_SECRET": explicit_secret},
            "docker_forward_env": ["FORWARDED_SECRET"],
        },
        image="example/image:latest",
        resolved_workdir="/workspace",
        effective_timeout=60,
        background=False,
        effective_pty=False,
        notify_on_complete=False,
        watch_patterns=None,
    )
    serialized = json.dumps(context, sort_keys=True)

    assert context["target"]["init_env_digest"].startswith("sha256:")
    assert "environment" not in context["target"]
    assert "forward_env" not in context["target"]
    assert explicit_secret not in serialized
    assert forwarded_secret not in serialized
    assert implicit_secret not in serialized


def test_changed_action_cannot_use_bound_grant(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_APPROVAL_ID", "apr_bound")
    monkeypatch.setenv("HERMES_KANBAN_APPROVAL_NONCE", "nonce_bound")
    monkeypatch.setattr(approval, "_consume_kanban_approval", lambda **_kwargs: False)
    request = patch.object(
        approval,
        "_request_kanban_approval",
        return_value={"id": "apr_replacement"},
    )

    with request as mock_request:
        result = approval.check_all_command_guards("rm -rf /tmp/changed", "local")

    assert result["approved"] is False
    assert result["status"] == "kanban_approval_pending"
    assert result["request_id"] == "apr_replacement"
    assert result["grant_mismatch"] is True
    mock_request.assert_called_once()


def test_hardline_block_wins_over_bound_grant(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_APPROVAL_ID", "apr_bound")
    monkeypatch.setenv("HERMES_KANBAN_APPROVAL_NONCE", "nonce_bound")
    consume = patch.object(
        approval,
        "_consume_kanban_approval",
        side_effect=AssertionError("hardline command must not consume grant"),
    )

    with consume:
        result = approval.check_all_command_guards("rm -rf /", "local")

    assert result["approved"] is False
    assert result["hardline"] is True


def test_configured_deny_rule_wins_over_bound_grant(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_APPROVAL_ID", "apr_bound")
    monkeypatch.setenv("HERMES_KANBAN_APPROVAL_NONCE", "nonce_bound")
    monkeypatch.setattr(
        approval, "_match_user_deny_rule", lambda _command: "git push --force*"
    )
    consume = patch.object(
        approval,
        "_consume_kanban_approval",
        side_effect=AssertionError("configured deny must not consume grant"),
    )

    with consume:
        result = approval.check_all_command_guards(
            "git push --force origin main", "local"
        )

    assert result["approved"] is False
    assert result["user_deny"] is True
    assert "git push --force*" in result["message"]


def test_execute_code_uses_same_durable_pause(monkeypatch):
    monkeypatch.setattr(
        approval,
        "_request_kanban_approval",
        lambda **_kwargs: {"id": "apr_code"},
    )

    result = approval.check_execute_code_guard("print('hello')", "local")

    assert result["approved"] is False
    assert result["status"] == "kanban_approval_pending"
    assert result["request_id"] == "apr_code"


def test_global_smart_approval_cannot_bypass_execute_code_ask_policy(monkeypatch):
    monkeypatch.setattr(approval, "_get_approval_mode", lambda: "smart")
    smart = patch.object(approval, "_smart_approve", return_value="approve")
    park = patch.object(
        approval,
        "_request_kanban_approval",
        return_value={"id": "apr_smart_code"},
    )

    with smart as mock_smart, park as mock_park:
        result = approval.check_execute_code_guard(
            "print('smart owner')",
            "local",
        )

    assert result["approved"] is False
    assert result["status"] == "kanban_approval_pending"
    assert result["request_id"] == "apr_smart_code"
    mock_smart.assert_not_called()
    mock_park.assert_called_once()


def test_execute_code_bound_grant_mismatch_fails_closed(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_APPROVAL_ID", "apr_code")
    monkeypatch.setenv("HERMES_KANBAN_APPROVAL_NONCE", "nonce")
    monkeypatch.setattr(approval, "_consume_kanban_approval", lambda **_kwargs: False)
    monkeypatch.setattr(
        approval,
        "_request_kanban_approval",
        lambda **_kwargs: {"id": "apr_code_replacement"},
    )

    result = approval.check_execute_code_guard("print('changed')", "local")

    assert result["approved"] is False
    assert result["status"] == "kanban_approval_pending"
    assert result["request_id"] == "apr_code_replacement"
    assert result["grant_mismatch"] is True


def test_real_db_request_approve_resume_and_consume_once(monkeypatch, tmp_path):
    """Exercise the real process-boundary state chain without executing a command."""
    from hermes_cli import kanban_db as kb

    db_path = tmp_path / "kanban.db"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(workspace))
    monkeypatch.setenv("TERMINAL_CWD", str(workspace))

    conn = kb.connect()
    task_id = kb.create_task(
        conn,
        title="approval e2e",
        assignee="worker",
        _trusted_gateway_origin={
            "platform": "telegram",
            "chat_id": "chat-1",
            "thread_id": "",
            "user_id": "user-1",
            "notifier_profile": "default",
        },
    )
    claimed = kb.claim_task(conn, task_id, claimer="host:claim")
    assert claimed is not None and claimed.current_run_id is not None
    first_run_id = claimed.current_run_id
    conn.close()

    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(first_run_id))
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", "host:claim")
    monkeypatch.setenv("HERMES_PROFILE", "worker")
    monkeypatch.setenv("HERMES_SESSION_ID", "worker-session")
    secret = "sk-proj-abc123xyz4567890abcdef"
    raw_command = f"rm -rf /tmp/approval-e2e # token {secret}"

    with patch(
        "tools.tirith_security.check_command_security",
        return_value={"action": "allow", "findings": [], "summary": ""},
    ):
        pending = approval.check_all_command_guards(
            raw_command, "local"
        )
    assert pending["status"] == "kanban_approval_pending"

    conn = kb.connect()
    request = kb.get_task_approval(conn, pending["request_id"])
    parked = kb.get_task(conn, task_id)
    first_run = kb.get_run(conn, first_run_id)
    assert request is not None and request["status"] == "pending"
    assert secret not in request["display_target"]
    assert secret not in str(kb.list_events(conn, task_id)[-1].payload)
    assert parked is not None and parked.status == "blocked"
    assert parked.current_run_id is None and parked.claim_lock is None
    assert first_run is not None and first_run.outcome == "approval_pending"

    decided = kb.decide_task_approval(
        conn,
        pending["request_id"],
        "approve",
        platform="telegram",
        chat_id="chat-1",
        thread_id="",
        user_id="user-1",
        notifier_profile="default",
    )
    assert decided is not None and decided["status"] == "approved"

    resumed = kb.claim_task(conn, task_id, claimer="host:resume")
    assert resumed is not None and resumed.current_run_id is not None
    assert resumed.approval_request_id == pending["request_id"]
    assert resumed.approval_worker_session_id == "worker-session"
    assert resumed.approval_grant_nonce
    second_run_id = resumed.current_run_id
    grant_nonce = resumed.approval_grant_nonce
    conn.close()

    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(second_run_id))
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", "host:resume")
    monkeypatch.setenv("HERMES_KANBAN_APPROVAL_ID", pending["request_id"])
    monkeypatch.setenv("HERMES_KANBAN_APPROVAL_NONCE", grant_nonce)
    with patch(
        "tools.tirith_security.check_command_security",
        return_value={"action": "allow", "findings": [], "summary": ""},
    ):
        approved = approval.check_all_command_guards(
            raw_command, "local"
        )
    assert approved["approved"] is True
    assert approved["kanban_approval_consumed"] is True

    conn = kb.connect()
    consumed = kb.get_task_approval(conn, pending["request_id"])
    assert consumed is not None and consumed["status"] == "consumed"
    assert not kb.consume_task_approval(
        conn,
        request_id=pending["request_id"],
        grant_nonce=grant_nonce,
        task_id=task_id,
        resume_run_id=second_run_id,
        profile="worker",
        action_digest=consumed["action_digest"],
    )
    conn.close()


@pytest.mark.parametrize(
    "choice, approved",
    [("deny", False), ("once", True)],
)
def test_kanban_delegate_uses_explicit_callback_policy(
    monkeypatch, choice, approved
):
    seen = []
    monkeypatch.setattr(approval, "_is_kanban_owner_context", lambda: False)
    monkeypatch.setattr(approval, "_is_kanban_delegate_context", lambda: True)
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", True)

    def callback(command, description, **_kwargs):
        seen.append((command, description))
        return choice

    monkeypatch.setattr(
        approval,
        "_resolve_explicit_approval_callback",
        lambda _callback=None: callback,
    )
    with patch(
        "tools.tirith_security.check_command_security",
        return_value={"action": "allow", "findings": [], "summary": ""},
    ):
        result = approval.check_all_command_guards(
            "rm -rf /tmp/delegate", "local"
        )

    assert result["approved"] is approved
    assert len(seen) == 1


def test_kanban_delegate_without_callback_fails_closed(monkeypatch):
    monkeypatch.setattr(approval, "_is_kanban_owner_context", lambda: False)
    monkeypatch.setattr(approval, "_is_kanban_delegate_context", lambda: True)
    monkeypatch.setattr(
        approval,
        "_resolve_explicit_approval_callback",
        lambda _callback=None: None,
    )

    result = approval.check_all_command_guards("rm -rf /tmp/delegate", "local")

    assert result["approved"] is False
    assert result["outcome"] == "approval_unavailable"


@pytest.mark.parametrize(
    "choice, approved",
    [("deny", False), ("once", True)],
)
def test_kanban_delegate_execute_code_uses_explicit_callback(
    monkeypatch, choice, approved,
):
    seen = []
    monkeypatch.setattr(approval, "_is_kanban_owner_context", lambda: False)
    monkeypatch.setattr(approval, "_is_kanban_delegate_context", lambda: True)
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", True)

    def callback(command, description, **_kwargs):
        seen.append((command, description))
        return choice

    monkeypatch.setattr(
        approval,
        "_resolve_explicit_approval_callback",
        lambda _callback=None: callback,
    )

    result = approval.check_execute_code_guard("print('delegate')", "local")

    assert result["approved"] is approved
    assert len(seen) == 1


def test_kanban_delegate_execute_code_without_callback_fails_closed(monkeypatch):
    monkeypatch.setattr(approval, "_is_kanban_owner_context", lambda: False)
    monkeypatch.setattr(approval, "_is_kanban_delegate_context", lambda: True)
    monkeypatch.setattr(
        approval,
        "_resolve_explicit_approval_callback",
        lambda _callback=None: None,
    )

    result = approval.check_execute_code_guard("print('delegate')", "local")

    assert result["approved"] is False
    assert result["outcome"] == "approval_unavailable"


def test_kanban_delegate_plugin_policy_ignores_inherited_session_approval(
    monkeypatch,
):
    seen = []
    monkeypatch.setattr(approval, "_is_kanban_owner_context", lambda: False)
    monkeypatch.setattr(approval, "_is_kanban_delegate_context", lambda: True)
    monkeypatch.setattr(approval, "is_approved", lambda *_args: True)

    def deny(command, description, **_kwargs):
        seen.append((command, description))
        return "deny"

    monkeypatch.setattr(
        approval,
        "_resolve_explicit_approval_callback",
        lambda _callback=None: deny,
    )

    result = approval.request_tool_approval(
        "write_file",
        "sensitive path",
        rule_key="ssh",
        tool_args={"path": "/tmp/file"},
    )

    assert result["approved"] is False
    assert len(seen) == 1


def test_nested_execute_code_approval_pause_reaches_outer_agent(
    monkeypatch, tmp_path,
):
    from agent.execution_context import (
        ExecutionRole,
        bind_agent_execution_context,
        issue_kanban_approval_pause_token,
        kanban_approval_pending_metadata,
        reset_agent_execution_context,
    )
    from tools.code_execution_tool import execute_code

    request_id = "ka_1234567890abcdef12345678"
    display_target = "rm -rf /tmp/nested"
    description = "nested command requires approval"
    inner = {
        "approved": False,
        "status": "kanban_approval_pending",
        "kanban_approval_pending": True,
        "request_id": request_id,
        "display_target": display_target,
        "description": description,
        "outcome": "approval_pending",
        "error": "",
    }
    inner["_hermes_kanban_pause_token"] = issue_kanban_approval_pause_token(
        request_id=request_id,
        task_id="task-1",
        run_id="7",
        profile="worker",
        display_target=display_target,
        description=description,
    )
    after_pause = tmp_path / "should-not-run"
    code = (
        "from hermes_tools import terminal\n"
        "try:\n"
        "    terminal('rm -rf /tmp/nested')\n"
        "except BaseException:\n"
        f"    open({str(after_pause)!r}, 'w').close()\n"
    )
    owner = type(
        "Agent",
        (),
        {"_execution_role": ExecutionRole.KANBAN_OWNER, "_delegate_depth": 0},
    )()

    with (
        patch(
            "tools.approval.check_execute_code_guard",
            return_value={"approved": True, "message": None},
        ),
        patch("model_tools.handle_function_call", return_value=json.dumps(inner)),
    ):
        token = bind_agent_execution_context(owner)
        try:
            raw = execute_code(
                code,
                task_id="task-1",
                enabled_tools=["terminal"],
            )
            verified = kanban_approval_pending_metadata(raw)
        finally:
            reset_agent_execution_context(token)

    result = json.loads(raw)
    if (
        result.get("status") == "error"
        and "Operation not permitted" in str(result.get("error"))
    ):
        pytest.skip("sandbox forbids local execute_code RPC sockets")
    assert result["status"] == "kanban_approval_pending"
    assert result["request_id"] == request_id
    assert verified is not None
    assert not after_pause.exists()


def test_remote_rpc_approval_pause_reaches_outer_execute_code(
    monkeypatch, tmp_path,
):
    from agent.execution_context import (
        ExecutionRole,
        bind_agent_execution_context,
        issue_kanban_approval_pause_token,
        kanban_approval_pending_metadata,
        reset_agent_execution_context,
    )
    from tools.code_execution_tool import _execute_remote

    class LocalFileRpcEnv:
        def get_temp_dir(self):
            return str(tmp_path)

        def execute(self, command, cwd=None, timeout=None, **_kwargs):
            try:
                completed = subprocess.run(
                    command,
                    shell=True,
                    cwd=cwd or "/",
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                return {
                    "output": completed.stdout,
                    "returncode": completed.returncode,
                }
            except subprocess.TimeoutExpired as exc:
                return {"output": exc.stdout or "", "returncode": 124}

    request_id = "ka_abcdef1234567890abcdef12"
    display_target = "rm -rf /tmp/remote-nested"
    description = "remote nested command requires approval"
    inner = {
        "approved": False,
        "status": "kanban_approval_pending",
        "kanban_approval_pending": True,
        "request_id": request_id,
        "display_target": display_target,
        "description": description,
        "outcome": "approval_pending",
        "error": "",
    }
    inner["_hermes_kanban_pause_token"] = issue_kanban_approval_pause_token(
        request_id=request_id,
        task_id="task-1",
        run_id="7",
        profile="worker",
        display_target=display_target,
        description=description,
    )
    after_pause = tmp_path / "remote-should-not-run"
    code = (
        "from hermes_tools import terminal\n"
        "try:\n"
        "    terminal('rm -rf /tmp/remote-nested')\n"
        "except BaseException:\n"
        f"    open({str(after_pause)!r}, 'w').close()\n"
    )
    owner = type(
        "Agent",
        (),
        {"_execution_role": ExecutionRole.KANBAN_OWNER, "_delegate_depth": 0},
    )()

    with (
        patch(
            "tools.code_execution_tool._get_or_create_env",
            return_value=(LocalFileRpcEnv(), "ssh"),
        ),
        patch(
            "tools.code_execution_tool._load_config",
            return_value={"timeout": 20, "max_tool_calls": 5},
        ),
        patch("model_tools.handle_function_call", return_value=json.dumps(inner)),
    ):
        token = bind_agent_execution_context(owner)
        try:
            raw = _execute_remote(code, "task-1", ["terminal"])
            verified = kanban_approval_pending_metadata(raw)
        finally:
            reset_agent_execution_context(token)

    result = json.loads(raw)
    assert result["status"] == "kanban_approval_pending"
    assert result["request_id"] == request_id
    assert verified is not None
    assert not after_pause.exists()


def test_registry_pause_cannot_be_leaked_or_erased_by_result_hooks(monkeypatch):
    from agent.execution_context import (
        ExecutionRole,
        bind_agent_execution_context,
        issue_kanban_approval_pause_token,
        kanban_approval_pending_metadata,
        reset_agent_execution_context,
    )
    import model_tools

    request_id = "ka_0123456789abcdef01234567"
    display_target = "rm -rf /tmp/hook-test"
    description = "hook test approval"
    marker = {
        "status": "kanban_approval_pending",
        "kanban_approval_pending": True,
        "request_id": request_id,
        "display_target": display_target,
        "description": description,
        "error": "",
    }
    marker["_hermes_kanban_pause_token"] = issue_kanban_approval_pause_token(
        request_id=request_id,
        task_id="task-1",
        run_id="7",
        profile="worker",
        display_target=display_target,
        description=description,
    )
    observed = []
    transformed = []
    middleware_observed = []

    def execution_middleware(*, next_call, **_kwargs):
        downstream = next_call()
        middleware_observed.append(downstream)
        return '{"status":"success"}'

    owner = type(
        "Agent",
        (),
        {"_execution_role": ExecutionRole.KANBAN_OWNER, "_delegate_depth": 0},
    )()

    with (
        patch.object(model_tools.registry, "dispatch", return_value=json.dumps(marker)),
        patch(
            "model_tools._emit_post_tool_call_hook",
            side_effect=lambda **kwargs: observed.append(kwargs["result"]),
        ),
        patch(
            "hermes_cli.plugins.has_hook",
            side_effect=lambda name: name == "transform_tool_result",
        ),
        patch(
            "hermes_cli.plugins.invoke_hook",
            side_effect=lambda *args, **kwargs: transformed.append((args, kwargs))
            or ['{"status":"success"}'],
        ),
        patch(
            "hermes_cli.middleware._get_middleware_callbacks",
            return_value=[execution_middleware],
        ),
    ):
        token = bind_agent_execution_context(owner)
        try:
            result = model_tools.handle_function_call(
                "terminal",
                {"command": "rm -rf /tmp/hook-test"},
                task_id="task-1",
                skip_pre_tool_call_hook=True,
            )
            verified = kanban_approval_pending_metadata(result)
        finally:
            reset_agent_execution_context(token)

    assert verified is not None
    assert transformed == []
    assert len(middleware_observed) == 1
    assert "_hermes_kanban_pause_token" not in middleware_observed[0]
    assert len(observed) == 1
    assert "_hermes_kanban_pause_token" not in observed[0]
    assert "_hermes_kanban_pause_token" in result


@pytest.mark.parametrize(
    "outcome",
    [
        "approval_pending",
        "approval_unavailable",
        "approval_persistence_failed",
    ],
)
def test_terminal_wrapper_preserves_signed_pause_until_executor(
    monkeypatch, outcome
):
    from agent.execution_context import (
        ExecutionRole,
        bind_agent_execution_context,
        kanban_approval_pending_metadata,
        reset_agent_execution_context,
    )
    from tools import terminal_tool

    class FakeEnv:
        env = {}
        cwd = "/tmp"

        def execute(self, *_args, **_kwargs):
            raise AssertionError("parked command executed")

    description = "signed terminal pause"
    request = None
    if outcome != "approval_persistence_failed":
        request = {
            "id": "ka_111111111111111111111111",
            "display_target": "rm -rf /tmp/signed",
            "description": description,
            "approval_unavailable": outcome == "approval_unavailable",
        }
    pending = approval._kanban_pending_result(request, description)
    expected_request_id = (
        request["id"] if request is not None else pending["request_id"]
    )
    monkeypatch.setattr(terminal_tool, "_active_environments", {"task-1": FakeEnv()})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
    monkeypatch.setattr(terminal_tool, "_resolve_container_task_id", lambda value: value)
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        terminal_tool,
        "_get_env_config",
        lambda: {
            "env_type": "local",
            "cwd": "/tmp",
            "timeout": 60,
            "lifetime_seconds": 3600,
        },
    )
    monkeypatch.setattr(terminal_tool, "_check_all_guards", lambda *_a, **_k: pending)
    owner = type(
        "Agent",
        (),
        {"_execution_role": ExecutionRole.KANBAN_OWNER, "_delegate_depth": 0},
    )()

    token = bind_agent_execution_context(owner)
    try:
        raw = terminal_tool.terminal_tool(
            command="rm -rf /tmp/signed",
            task_id="task-1",
        )
        verified = kanban_approval_pending_metadata(raw)
    finally:
        reset_agent_execution_context(token)

    assert verified is not None
    assert verified["request_id"] == expected_request_id
    if outcome == "approval_persistence_failed":
        assert expected_request_id.startswith("kaf_")
    assert verified["outcome"] == outcome
    assert json.loads(raw)["outcome"] == outcome


@pytest.mark.parametrize(
    "outcome",
    [
        "approval_pending",
        "approval_unavailable",
        "approval_persistence_failed",
    ],
)
def test_execute_code_wrapper_preserves_signed_pause_until_executor(
    monkeypatch, outcome
):
    from agent.execution_context import (
        ExecutionRole,
        bind_agent_execution_context,
        kanban_approval_pending_metadata,
        reset_agent_execution_context,
    )
    from tools import code_execution_tool

    description = "signed code pause"
    request = None
    if outcome != "approval_persistence_failed":
        request = {
            "id": "ka_222222222222222222222222",
            "display_target": "execute_code (redacted)",
            "description": description,
            "approval_unavailable": outcome == "approval_unavailable",
        }
    pending = approval._kanban_pending_result(request, description)
    expected_request_id = (
        request["id"] if request is not None else pending["request_id"]
    )
    monkeypatch.setattr(code_execution_tool, "SANDBOX_AVAILABLE", True)
    monkeypatch.setattr(
        "tools.terminal_tool._get_env_config",
        lambda: {"env_type": "local"},
    )
    monkeypatch.setattr(
        "tools.terminal_tool._docker_has_host_access",
        lambda _config: False,
    )
    monkeypatch.setattr(
        "tools.approval.check_execute_code_guard",
        lambda *_a, **_k: pending,
    )
    owner = type(
        "Agent",
        (),
        {"_execution_role": ExecutionRole.KANBAN_OWNER, "_delegate_depth": 0},
    )()

    token = bind_agent_execution_context(owner)
    try:
        raw = code_execution_tool.execute_code("print('blocked')")
        verified = kanban_approval_pending_metadata(raw)
    finally:
        reset_agent_execution_context(token)

    assert verified is not None
    assert verified["request_id"] == expected_request_id
    if outcome == "approval_persistence_failed":
        assert expected_request_id.startswith("kaf_")
    assert verified["outcome"] == outcome
    assert json.loads(raw)["outcome"] == outcome


@pytest.mark.parametrize(
    "outcome",
    ["approval_unavailable", "approval_persistence_failed"],
)
def test_nested_execute_code_resign_preserves_control_outcome(outcome):
    from agent.execution_context import (
        ExecutionRole,
        bind_agent_execution_context,
        kanban_approval_pending_metadata,
        reset_agent_execution_context,
    )
    from tools.code_execution_tool import _serialize_kanban_approval_pause

    pause = {
        "approved": False,
        "status": "kanban_approval_pending",
        "kanban_approval_pending": True,
        "request_id": (
            "kaf_333333333333333333333333"
            if outcome == "approval_persistence_failed"
            else "ka_333333333333333333333333"
        ),
        "display_target": "nested approval control halt",
        "description": "nested approval control halt",
        "outcome": outcome,
        "error": "",
    }
    owner = type(
        "Agent",
        (),
        {"_execution_role": ExecutionRole.KANBAN_OWNER, "_delegate_depth": 0},
    )()

    token = bind_agent_execution_context(owner)
    try:
        raw = _serialize_kanban_approval_pause(
            pause,
            tool_calls_made=1,
            duration_seconds=0.25,
        )
        verified = kanban_approval_pending_metadata(raw)
    finally:
        reset_agent_execution_context(token)

    assert verified is not None
    assert verified["outcome"] == outcome
    assert json.loads(raw)["outcome"] == outcome
