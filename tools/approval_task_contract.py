"""Task-bound terminal admission for Kanban workers.

The normal Docker approval fast path is intentionally broad: a container without
host mounts can run commands without a human prompt.  A worker may nevertheless
have a narrower, card-specific authority.  This module enforces that authority
before the backend-specific fast path and is deliberately independent of any
container implementation.
"""
from __future__ import annotations

import os
import re
import shlex
from typing import Any


_CONTRACT_OPERATIONS = frozenset({"sync", "rollback", "terminate-op"})
_ARGOCD_READS = frozenset({"diff", "get", "history", "list", "logs", "manifests", "resources", "wait"})
_SHELL_ARGOCD_APP = re.compile(
    r"\b(?:[\w./-]+/)?argocd\b[^\n]*(?:\sapp(?:\s|$))",
)
_SHELL_CONTROL = re.compile(r"[;&|<>\n`$]")
_SHELL_CARRIERS = frozenset({
    "ash", "bash", "dash", "env", "eval", "fish", "ksh", "perl", "python",
    "python3", "sh", "zsh",
})



def _blocked(message: str) -> dict[str, Any]:
    return {
        "approved": False,
        "message": message,
        "description": "task execution contract",
        "task_execution_contract": True,
    }


def _argv(command: str) -> list[str] | None:
    try:
        return shlex.split(command, posix=True)
    except ValueError:
        return None


def _argocd_write(argv: list[str]) -> tuple[str, str] | None:
    """Return one contract-addressable write, or a fail-closed sentinel.

    Contract authority permits only an exact, single-app direct invocation. A selector,
    post-operation option, multi-app form, action, or unknown subcommand is outside the initial
    schema and must be denied rather than guessed.
    """
    if not argv or argv[0].rsplit("/", 1)[-1] != "argocd":
        return None
    try:
        app_index = argv.index("app")
    except ValueError:
        # An opted-in worker's ArgoCD authority is limited to application reads and the
        # contract-addressable application writes below. Project/repository/cluster/account
        # commands are never safe to infer from an application contract.
        return "", ""
    if app_index + 1 >= len(argv):
        return None
    operation = argv[app_index + 1]
    if operation in _ARGOCD_READS:
        return None
    targets = argv[app_index + 2:]
    if (
        operation not in _CONTRACT_OPERATIONS
        or not targets
        or targets[0].startswith("-")
        or any(token != "--grpc-web" for token in targets[1:])
    ):
        return operation, ""
    return operation, targets[0]


def _server(argv: list[str]) -> str | None:
    for index, token in enumerate(argv):
        if token == "--server" and index + 1 < len(argv):
            return argv[index + 1]
        if token.startswith("--server="):
            return token.split("=", 1)[1]
    return None


def _matches_target(contract: dict[str, Any], *, server: str | None, application: str, operation: str) -> bool:
    if contract.get("kind") != "argocd" or not isinstance(contract.get("targets"), list):
        return False
    return any(
        isinstance(target, dict)
        and target.get("server") == server
        and target.get("application") == application
        and target.get("operation") == operation
        for target in contract["targets"]
    )


def check_task_execution_contract(
    command: str,
    *,
    task_id: str,
    contract: dict[str, Any] | None,
    required: bool,
) -> dict[str, Any] | None:
    """Return a blocking approval result when a worker write exceeds its card.

    The caller resolves the immutable task contract.  ``required=False`` is the
    backwards-compatible default for profiles that do not opt into this policy.
    """
    if not required:
        return None
    argv = _argv(command)
    if argv is None:
        return _blocked("Blocked: task-bound admission requires a parseable direct argocd invocation.")
    # The terminal runs a shell string. Substitution, a pipeline, redirection, or a second command
    # can decode or construct an untrusted ArgoCD invocation after this guard has inspected it.
    # An opted-in task therefore admits only a literal argv command, never shell evaluation.
    if _SHELL_CONTROL.search(command):
        return _blocked("Blocked: task-bound admission requires one direct argocd invocation, not shell composition.")
    write = _argocd_write(argv)
    if write is None:
        # An interpreter or environment carrier can keep a whole command program in one argv
        # element (for example ``bash -c 'argocd …'``). Under an application-only authority we
        # cannot prove its eventual process tree is within the contract, so direct invocation is
        # the only admissible shape.
        executable = argv[0].rsplit("/", 1)[-1] if argv else ""
        if executable in _SHELL_CARRIERS:
            return _blocked("Blocked: task-bound admission requires a direct argocd invocation, not a shell carrier.")
        # A shell carrier can hide a mutation from argv parsing.  Refuse that
        # shape rather than treating the card's textual contract as authority.
        if argv and argv[0].rsplit("/", 1)[-1] != "argocd" and _SHELL_ARGOCD_APP.search(command):
            return _blocked("Blocked: task-bound admission requires a direct argocd invocation, not a shell carrier.")
        return None
    operation, application = write
    if any(token == "--prune" or token.startswith("--prune=") for token in argv):
        return _blocked("Blocked: ArgoCD --prune is never authorized by a task execution contract.")
    if not task_id:
        return _blocked("Blocked: a delegated child has no Kanban task authority for an ArgoCD write.")
    if not application:
        return _blocked(
            "Blocked: task-bound admission permits only one named application with sync, rollback, or terminate-op."
        )
    if not contract:
        return _blocked(
            f"Blocked: Kanban task {task_id} has no execution_contract authorizing argocd app {operation}."
        )
    server = _server(argv)
    if _matches_target(contract, server=server, application=application, operation=operation):
        return None
    return _blocked(
        "Blocked: argocd write is outside this task's execution_contract "
        f"(server={server or '(missing)'}, application={application or '(action)'}, operation={operation})."
    )


def task_contract_from_environment() -> tuple[str, dict[str, Any] | None, bool]:
    """Resolve this worker's persisted contract without trusting prompt text.

    This runs in the Hermes process, before a terminal command reaches Docker.
    Board resolution comes from the dispatcher-pinned environment rather than a
    worker-controlled filesystem path.
    """
    from tools import approval_context

    config = approval_context._get_approval_config()
    required = bool(config.get("require_task_execution_contract", False)) if isinstance(config, dict) else False
    task_id = os.environ.get("HERMES_KANBAN_TASK", "").strip()
    from agent.delegation_context import is_dispatcher_owned_worker_context
    if task_id and not is_dispatcher_owned_worker_context():
        return "", None, bool(required)
    if not task_id:
        # Delegate children deliberately lose the parent's task identity. Keep the admission
        # fence on that lineage: absence is not permission to issue a write as an orchestrator.
        from agent.delegation_context import is_delegated_child_process_context
        return "", None, bool(required and is_delegated_child_process_context())
    if not required:
        return task_id, None, False
    try:
        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kbc

        with kbc.connect_closing() as conn:
            task = kb.get_task(conn, task_id)
        return task_id, task.execution_contract if task else None, True
    except Exception:
        # An opted-in profile must fail closed when its claimed task cannot be
        # read.  ``None`` means no contract and produces the stable denial.
        return task_id, None, True


def check_current_task_execution_contract(command: str) -> dict[str, Any] | None:
    """Evaluate the dispatcher-pinned task contract for one terminal command."""
    task_id, contract, required = task_contract_from_environment()
    return check_task_execution_contract(
        command, task_id=task_id, contract=contract, required=required,
    )


def check_current_task_execution_contract_for_code() -> dict[str, Any] | None:
    """Refuse arbitrary Python for an opted-in worker or its delegated descendants.

    ``execute_code`` can spawn an ArgoCD subprocess without presenting a shell command to the
    terminal guard, so static inspection of Python is not an authority boundary.
    """
    task_id, _contract, required = task_contract_from_environment()
    if not required:
        return None
    owner = f"Kanban task {task_id}" if task_id else "a delegated child"
    return _blocked(f"Blocked: {owner} cannot use execute_code while task-bound admission is enabled.")
