#!/usr/bin/env python3
"""
Git Branch Tool - Git branch operations

Provides Git branch management: list, create, delete, and switch branches.
"""

import json
import os
import subprocess
from typing import Dict, List, Optional


def _run_git(args: List[str], cwd: Optional[str] = None) -> Dict[str, any]:
    """Run a git command and return the result."""
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd or os.getcwd(),
            capture_output=True,
            text=True,
            timeout=30,
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "returncode": result.returncode,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def git_branch(
    operation: str,
    branch_name: Optional[str] = None,
    start_point: Optional[str] = None,
    cwd: Optional[str] = None,
    force: bool = False,
    task_id: Optional[str] = None,
) -> str:  # noqa: D205
    """
    Perform Git branch operations.

    Args:
        operation: Operation to perform (list, create, delete, switch)
        branch_name: Name of branch for create/delete/switch
        start_point: Starting commit/branch for create (default: current HEAD)
        cwd: Working directory for git command
        force: Force delete unmerged branch (use with delete operation)

    Returns:
        JSON string with operation results
    """
    repo_root = cwd or os.getcwd()

    if operation == "list":
        result = _run_git(["branch", "-a"], repo_root)
        if result["success"]:
            branches = [b.strip() for b in result["stdout"].split("\n") if b.strip()]
            current = None
            for b in branches:
                if b.startswith("*"):
                    current = b[1:].strip()
                    break
            return json.dumps({
                "success": True,
                "operation": "list",
                "current_branch": current,
                "branches": branches,
            })
        return json.dumps({
            "success": False,
            "error": result.get("stderr", "Failed to list branches"),
        })

    elif operation == "create":
        if not branch_name:
            return json.dumps({
                "success": False,
                "error": "branch_name required for create operation",
            })
        args = ["checkout", "-b", branch_name]
        if start_point:
            args.append(start_point)
        result = _run_git(args, repo_root)
        if result["success"]:
            return json.dumps({
                "success": True,
                "operation": "create",
                "branch": branch_name,
                "message": f"Created and switched to branch '{branch_name}'",
            })
        return json.dumps({
            "success": False,
            "error": result.get("stderr", "Failed to create branch"),
        })

    elif operation == "delete":
        if not branch_name:
            return json.dumps({
                "success": False,
                "error": "branch_name required for delete operation",
            })

        # Gate force-delete through the dangerous-command detector so the
        # terminal approval path can intercept it.  `git branch -D` is
        # classified as dangerous by tools/approval.py:detect_dangerous_command.
        if force:
            from tools.approval import detect_dangerous_command
            cmd_str = f"git branch -D {branch_name}"
            dangerous, _, desc = detect_dangerous_command(cmd_str)
            if dangerous:
                return json.dumps({
                    "success": False,
                    "error": (
                        f"Blocked: {desc}. Use the terminal tool to run this "
                        "command so the approval gate can review it."
                    ),
                    "dangerous": True,
                })

        flag = "-D" if force else "-d"
        result = _run_git(["branch", flag, branch_name], repo_root)
        if result["success"]:
            return json.dumps({
                "success": True,
                "operation": "delete",
                "branch": branch_name,
                "force": force,
                "message": f"Deleted branch '{branch_name}' (forced: {force})",
            })
        return json.dumps({
            "success": False,
            "error": result.get("stderr", "Failed to delete branch"),
        })

    elif operation == "switch":
        if not branch_name:
            return json.dumps({
                "success": False,
                "error": "branch_name required for switch operation",
            })
        result = _run_git(["checkout", branch_name], repo_root)
        if result["success"]:
            return json.dumps({
                "success": True,
                "operation": "switch",
                "branch": branch_name,
                "message": f"Switched to branch '{branch_name}'",
            })
        return json.dumps({
            "success": False,
            "error": result.get("stderr", "Failed to switch branch"),
        })

    else:
        return json.dumps({
            "success": False,
            "error": f"Unknown operation: {operation}",
        })


def check_git_branch_requirements() -> bool:
    """Git branch tool requires git to be installed."""
    try:
        result = subprocess.run(
            ["git", "--version"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


GIT_BRANCH_SCHEMA = {
    "name": "git_branch",
    "description": (
        "Perform Git branch operations: list, create, delete, and switch branches.\n\n"
        "Operations:\n"
        "- list: List all local and remote branches\n"
        "- create: Create a new branch and switch to it\n"
        "- delete: Delete a local branch\n"
        "- switch: Switch to an existing branch"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "description": "Operation to perform",
                "enum": ["list", "create", "delete", "switch"],
            },
            "branch_name": {
                "type": "string",
                "description": "Name of branch for create/delete/switch",
            },
            "start_point": {
                "type": "string",
                "description": "Starting commit/branch for create (default: current HEAD)",
            },
            "cwd": {
                "type": "string",
                "description": "Working directory for git operation",
            },
            "force": {
                "type": "boolean",
                "description": "Force delete unmerged branch (use with delete operation)",
                "default": False,
            },
            "task_id": {
                "type": "string",
                "description": "Optional task ID for tracking",
            },
        },
        "required": ["operation"],
    },
}


from tools.registry import registry

registry.register(
    name="git_branch",
    toolset="git",
    schema=GIT_BRANCH_SCHEMA,
    handler=lambda args, **kw: git_branch(
        operation=args.get("operation", "list"),
        branch_name=args.get("branch_name"),
        start_point=args.get("start_point"),
        cwd=args.get("cwd"),
        force=args.get("force", False),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_git_branch_requirements,
    emoji="🌿",
)
