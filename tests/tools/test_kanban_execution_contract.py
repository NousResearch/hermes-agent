"""Kanban ArgoCD execution-contract admission.

A Docker backend normally bypasses dangerous-command prompts.  When a profile
opts into task-bound admission, a Kanban worker may mutate ArgoCD only through
the immutable contract on its assigned task.
"""
from __future__ import annotations

import pytest


def test_contract_allows_only_its_named_argocd_target():
    from tools.approval_task_contract import check_task_execution_contract

    contract = {
        "kind": "argocd",
        "targets": [{
            "server": "argocd-system-diiastage-3dc.diia.digital",
            "application": "example-app",
            "operation": "sync",
        }],
    }

    allowed = check_task_execution_contract(
        "argocd --grpc-web --server argocd-system-diiastage-3dc.diia.digital "
        "app sync example-app",
        task_id="t_example",
        contract=contract,
        required=True,
    )
    wrong_app = check_task_execution_contract(
        "argocd --server argocd-system-diiastage-3dc.diia.digital "
        "app sync another-app --grpc-web",
        task_id="t_example",
        contract=contract,
        required=True,
    )
    wrong_server = check_task_execution_contract(
        "argocd --server argocd-system-diiaprod-3dc.diia.digital "
        "app sync example-app --grpc-web",
        task_id="t_example",
        contract=contract,
        required=True,
    )

    assert allowed is None
    assert wrong_app and wrong_app["approved"] is False
    assert "another-app" in wrong_app["message"]
    assert wrong_server and wrong_server["approved"] is False
    assert "diiaprod" in wrong_server["message"]


def test_contract_blocks_argocd_write_when_card_has_no_contract():
    from tools.approval_task_contract import check_task_execution_contract

    result = check_task_execution_contract(
        "argocd --server argocd-system-diiastage-3dc.diia.digital app sync example-app",
        task_id="t_example",
        contract=None,
        required=True,
    )

    assert result and result["approved"] is False
    assert "execution_contract" in result["message"]


@pytest.mark.parametrize("command", [
    "argocd app get example-app",
    "argocd --server argocd-system-diiastage-3dc.diia.digital app diff example-app",
])
def test_contract_does_not_restrict_argocd_reads(command):
    from tools.approval_task_contract import check_task_execution_contract

    assert check_task_execution_contract(command, task_id="t_example", contract=None, required=True) is None


@pytest.mark.parametrize("command", [
    "printf 'argocd app sync example-app'",
    'busybox sh -c "argocd --server stage app sync another-app"',
])
def test_contract_rejects_non_direct_argocd_text(command):
    from tools.approval_task_contract import check_task_execution_contract

    result = check_task_execution_contract(
        command,
        task_id="t_example",
        contract={"kind": "argocd", "targets": [{
            "server": "stage", "application": "example-app", "operation": "sync",
        }]},
        required=True,
    )

    assert result and result["approved"] is False


def test_contract_rejects_shell_carrier_for_argocd_write():
    from tools.approval_task_contract import check_task_execution_contract

    result = check_task_execution_contract(
        "bash -lc 'argocd --server argocd-system-diiastage-3dc.diia.digital app sync example-app'",
        task_id="t_example",
        contract={
            "kind": "argocd",
            "targets": [{
                "server": "argocd-system-diiastage-3dc.diia.digital",
                "application": "example-app",
                "operation": "sync",
            }],
        },
        required=True,
    )

    assert result and result["approved"] is False
    assert "direct argocd invocation" in result["message"]


@pytest.mark.parametrize("command", [
    'bash -c "argocd --server stage app sync another-app"',
    "sh -c 'argocd --server stage app sync another-app'",
    "eval 'argocd --server stage app sync another-app'",
])
def test_contract_rejects_quoted_shell_carrier(command):
    from tools.approval_task_contract import check_task_execution_contract

    result = check_task_execution_contract(
        command,
        task_id="t_example",
        contract={"kind": "argocd", "targets": [{
            "server": "stage", "application": "example-app", "operation": "sync",
        }]},
        required=True,
    )

    assert result and result["approved"] is False
    assert "direct argocd invocation" in result["message"]


@pytest.mark.parametrize("command", [
    "argocd --server argocd-system-diiastage-3dc.diia.digital app sync example-app --prune",
    "argocd --server argocd-system-diiastage-3dc.diia.digital app sync example-app --prune=true",
    "argocd --server argocd-system-diiastage-3dc.diia.digital app sync example-app; argocd app sync another-app",
    "echo YXJnb2NkIGFwcCBkZWxldGUgYW5vdGhlci1hcHA= | base64 -d | sh",
    "X=$(echo YXJnb2NkIGFwcCBkZWxldGUgYW5vdGhlci1hcHA= | base64 -d); $X",
])
def test_contract_never_authorizes_prune_or_shell_composition(command):
    from tools.approval_task_contract import check_task_execution_contract

    result = check_task_execution_contract(
        command,
        task_id="t_example",
        contract={"kind": "argocd", "targets": [{
            "server": "argocd-system-diiastage-3dc.diia.digital",
            "application": "example-app",
            "operation": "sync",
        }]},
        required=True,
    )

    assert result and result["approved"] is False


@pytest.mark.parametrize("command", [
    "argocd --server stage app sync -l app.kubernetes.io/instance=another-app --prune",
    "argocd --server stage app delete -l app.kubernetes.io/instance=another-app",
    "argocd --server stage app sync example-app another-app",
    "argocd --server stage app delete-resource example-app --group apps --kind Deployment",
    "argocd --server stage app patch-resource example-app --kind Deployment --patch '{}'",
    "argocd --server stage app confirm-deletion example-app",
])
def test_contract_denies_unaddressable_argocd_app_writes(command):
    from tools.approval_task_contract import check_task_execution_contract

    result = check_task_execution_contract(
        command,
        task_id="t_example",
        contract={"kind": "argocd", "targets": [{
            "server": "stage", "application": "example-app", "operation": "sync",
        }]},
        required=True,
    )

    assert result and result["approved"] is False


@pytest.mark.parametrize("command", [
    "argocd proj create rogue-project",
    "argocd cluster add rogue-context",
    "argocd repo add https://evil.example.invalid/repo",
    "argocd account update-password",
])
def test_contract_denies_non_application_argocd_commands(command):
    from tools.approval_task_contract import check_task_execution_contract

    result = check_task_execution_contract(
        command,
        task_id="t_example",
        contract={"kind": "argocd", "targets": [{
            "server": "stage", "application": "example-app", "operation": "sync",
        }]},
        required=True,
    )

    assert result and result["approved"] is False


def test_contract_blocks_execute_code_before_docker_fast_path(monkeypatch):
    from tools import approval

    monkeypatch.setattr(
        "tools.approval_task_contract.check_current_task_execution_contract_for_code",
        lambda: {"approved": False, "message": "Blocked: task contract."},
    )

    result = approval.check_execute_code_guard(
        'subprocess.run(["argocd", "app", "sync", "another-app"])', "docker",
    )

    assert result["approved"] is False
    assert result["message"] == "Blocked: task contract."


def test_current_task_contract_blocks_execute_code_from_pinned_board(tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from tools import approval
    from tools import approval_context

    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "kanban.db"))
    kb._INITIALIZED_PATHS.clear()
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="deploy", execution_contract={
            "kind": "argocd", "targets": [{
                "server": "stage", "application": "example-app", "operation": "sync",
            }],
        })
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setattr(
        approval_context, "_get_approval_config",
        lambda: {"require_task_execution_contract": True},
    )

    result = approval.check_execute_code_guard(
        'subprocess.run(["argocd", "app", "sync", "another-app"])', "docker",
    )

    assert result["approved"] is False
    assert result["task_execution_contract"] is True


def test_delegated_child_without_task_id_cannot_write(monkeypatch):
    from tools import approval_context
    from tools.approval_task_contract import check_current_task_execution_contract

    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setenv("HERMES_DELEGATED_CHILD_CONTEXT", "1")
    monkeypatch.setattr(
        approval_context, "_get_approval_config",
        lambda: {"require_task_execution_contract": True},
    )

    result = check_current_task_execution_contract(
        "argocd --server argocd-system-diiastage-3dc.diia.digital app sync example-app",
    )

    assert result and result["approved"] is False
    assert "delegated child" in result["message"]


def test_in_process_delegated_child_cannot_inherit_task_authority(tmp_path, monkeypatch):
    from agent.delegation_context import delegated_child_context
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from tools import approval_context
    from tools.approval_task_contract import check_current_task_execution_contract

    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "kanban.db"))
    kb._INITIALIZED_PATHS.clear()
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="deploy", execution_contract={
            "kind": "argocd", "targets": [{
                "server": "stage", "application": "example-app", "operation": "sync",
            }],
        })
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setattr(
        approval_context, "_get_approval_config",
        lambda: {"require_task_execution_contract": True},
    )

    with delegated_child_context():
        result = check_current_task_execution_contract(
            "argocd --server stage app sync example-app",
        )

    assert result and result["approved"] is False
    assert "delegated child" in result["message"]


def test_contract_is_an_opt_in_config_default():
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["approvals"]["require_task_execution_contract"] is False


def test_execution_contract_precedes_docker_fast_path(monkeypatch):
    """A card-bound write remains blocked even though Docker skips prompts."""
    from tools import approval
    from tools import approval_context

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_example")
    monkeypatch.setattr(
        approval_context,
        "_get_approval_config",
        lambda: {"require_task_execution_contract": True},
    )
    monkeypatch.setattr(
        "tools.approval_task_contract.task_contract_from_environment",
        lambda: ("t_example", None, True),
    )

    result = approval.check_all_command_guards(
        "argocd --server argocd-system-diiastage-3dc.diia.digital app sync example-app",
        "docker",
        has_host_access=False,
    )

    assert result["approved"] is False
    assert result["task_execution_contract"] is True


def test_execution_contract_cannot_be_bypassed_with_force(monkeypatch):
    from tools import terminal_tool

    monkeypatch.setattr(
        "tools.approval_task_contract.check_current_task_execution_contract",
        lambda _command: {
            "approved": False,
            "message": "Blocked: task contract.",
            "description": "task execution contract",
            "task_execution_contract": True,
        },
    )

    with pytest.raises(terminal_tool._Rejected):
        terminal_tool._run_approval_guards(
            "argocd --server stage app sync example-app", "docker", {}, force=True,
        )


def test_current_task_contract_is_loaded_from_the_pinned_board(tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from tools import approval_context
    from tools.approval_task_contract import check_current_task_execution_contract

    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "kanban.db"))
    kb._INITIALIZED_PATHS.clear()
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="deploy", execution_contract={
            "kind": "argocd", "targets": [{
                "server": "stage", "application": "example-app", "operation": "sync",
            }],
        })
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setattr(
        approval_context, "_get_approval_config",
        lambda: {"require_task_execution_contract": True},
    )

    assert check_current_task_execution_contract(
        "argocd --server stage app sync example-app",
    ) is None
