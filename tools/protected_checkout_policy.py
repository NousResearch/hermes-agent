"""Protected checkout policy module.

Provides deterministic allow/block decisions for path mutation attempts.
Identifies protected canonical checkouts, allowed worktree prefixes,
and returns stable reason-coded decisions before file/terminal guards use them.

Config/registry driven: protected_roots and allowed_worktree_prefixes
are explicit lists that callers (or config) populate. No hardcoded broad
global blocking beyond the configured DailyChingu canonical root.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List


# ---------------------------------------------------------------------------
# Registry — explicit; extend via config or programmatic registration
# ---------------------------------------------------------------------------

PROTECTED_CANONICAL_ROOTS: List[str] = [
    "/home/ubuntu/repos/dailychingu",
]

ALLOWED_WORKTREE_PREFIXES: List[str] = [
    "/home/ubuntu/.hermes/hermes-agent/.worktrees",
]

def _string_list(value: Any) -> list[str]:
    """Return a cleaned list of strings from a config value."""
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _load_configured_registry() -> tuple[list[str], list[str]]:
    """Load protected checkout roots/prefixes from config.yaml.

    Supported shape:

    protected_checkouts:
      canonical_roots:
        - /home/ubuntu/repos/dailychingu
      allowed_worktree_prefixes:
        - /home/ubuntu/.hermes/hermes-agent/.worktrees

    Missing or malformed config falls back to module defaults. That avoids
    globally breaking file/terminal tools while keeping one executable SSOT:
    callers ask this module for the effective registry instead of relying on
    SOUL, skills, docs, or comments.
    """
    roots = list(PROTECTED_CANONICAL_ROOTS)
    prefixes = list(ALLOWED_WORKTREE_PREFIXES)
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly() or {}
    except Exception:
        return roots, prefixes

    section = cfg.get("protected_checkouts")
    if not isinstance(section, dict):
        return roots, prefixes

    configured_roots = _string_list(section.get("canonical_roots"))
    configured_prefixes = _string_list(section.get("allowed_worktree_prefixes"))
    return configured_roots or roots, configured_prefixes or prefixes


def effective_protected_checkout_registry() -> dict[str, list[str]]:
    """Return the executable protected-checkout registry used by guards."""
    roots, prefixes = _load_configured_registry()
    return {
        "canonical_roots": list(roots),
        "allowed_worktree_prefixes": list(prefixes),
    }


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProtectedCheckoutDecision:
    """Result of a protected-checkout policy evaluation.

    Attributes:
        allowed: Whether the mutation is permitted.
        reason_code: Stable short code for logging/auditing.
            Values: ALLOWED_NON_PROTECTED, ALLOWED_WORKTREE, ALLOWED_TASK_WORKTREE,
                    BLOCKED_PROTECTED_CANONICAL, BLOCKED_BRANCH_LOOKUP_FAILED.
        reason_detail: Human-readable explanation.
    """

    allowed: bool
    reason_code: str
    reason_detail: str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_path_mutation(target_path: str) -> ProtectedCheckoutDecision:
    """Determine whether a mutation at *target_path* is allowed.

    Decision logic (fail-closed):

    1. Resolve the path, compare with ``PROTECTED_CANONICAL_ROOTS``.
    2. If matched → run ``git branch --show-current``:
       - ``autopilot/*`` / ``wt/*`` branches → ALLOWED_TASK_WORKTREE
       - any other branch → BLOCKED_PROTECTED_CANONICAL
       - `git` failure / empty output → BLOCKED_BRANCH_LOOKUP_FAILED
    3. If not matched → compare with ``ALLOWED_WORKTREE_PREFIXES``:
       - matched → ALLOWED_WORKTREE
       - not matched → ALLOWED_NON_PROTECTED
    """
    resolved = Path(target_path).resolve()
    resolved_str = str(resolved)

    protected_roots, allowed_prefixes = _load_configured_registry()

    # 1. Check against protected canonical roots
    for root in protected_roots:
        root_path = Path(root).resolve()
        root_str = str(root_path)
        if _is_under_or_equal(resolved_str, root_str):
            return _check_protected_root(target_path, root_str)

    # 2. Check allowed worktree prefixes
    for prefix in allowed_prefixes:
        prefix_path = Path(prefix).resolve()
        prefix_str = str(prefix_path)
        if _is_under(resolved_str, prefix_str):
            return ProtectedCheckoutDecision(
                allowed=True,
                reason_code="ALLOWED_WORKTREE",
                reason_detail="Path {} is under allowed worktree prefix {}".format(
                    resolved_str, prefix_str
                ),
            )

    # 3. Non-protected repo
    return ProtectedCheckoutDecision(
        allowed=True,
        reason_code="ALLOWED_NON_PROTECTED",
        reason_detail="Path {} is not under any protected root".format(resolved_str),
    )


def _is_under_or_equal(child: str, parent: str) -> bool:
    """True when *child* equals *parent* or is a descendant directory."""
    return child == parent or _is_under(child, parent)


def _is_under(child: str, parent: str) -> bool:
    """True when *child* strictly starts with *parent* + os.sep."""
    return child.startswith(parent + os.sep)


def _check_protected_root(
    target_path: str, root_str: str
) -> ProtectedCheckoutDecision:
    """Check a path inside a protected canonical root.

    We run ``git branch --show-current`` in the target directory. If we can't
    get a branch (or the branch isn't recognisably a task worktree), we block.
    """

    # Resolve to a real directory for the cwd of the git command.
    cwd = target_path
    if os.path.isfile(target_path):
        cwd = str(Path(target_path).parent)

    reason = "BLOCKED_BRANCH_LOOKUP_FAILED"

    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            detail = "git branch failed (rc={}): {}".format(
                result.returncode, result.stderr.strip() or "(no stderr)"
            )
            return ProtectedCheckoutDecision(
                allowed=False, reason_code=reason, reason_detail=detail
            )

        branch = result.stdout.strip()
        if not branch:
            detail = "git branch returned empty output"
            return ProtectedCheckoutDecision(
                allowed=False, reason_code=reason, reason_detail=detail
            )

        # Recognised task-worktree branches
        if branch.startswith("autopilot/") or branch.startswith("wt/"):
            return ProtectedCheckoutDecision(
                allowed=True,
                reason_code="ALLOWED_TASK_WORKTREE",
                reason_detail=(
                    "Protected root worktree with task branch '{}'".format(branch)
                ),
            )

        # Any other branch on the protected canonical → blocked
        return ProtectedCheckoutDecision(
            allowed=False,
            reason_code="BLOCKED_PROTECTED_CANONICAL",
            reason_detail=(
                "Protected canonical checkout {} on non-task branch '{}'".format(
                    target_path, branch
                )
            ),
        )

    except (subprocess.TimeoutExpired, OSError, FileNotFoundError):
        detail = "Cannot determine branch for {}".format(target_path)
        return ProtectedCheckoutDecision(
            allowed=False, reason_code=reason, reason_detail=detail
        )
