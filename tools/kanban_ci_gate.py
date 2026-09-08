"""CI gate for ``kanban_complete`` — done requires a GREEN CHECK, not a local run.

STAGED. Installs to ``hermes-agent/tools/kanban_ci_gate.py``.

WHY THIS EXISTS
---------------
The fleet's definition of "green" has been *a worker running pytest in its own
worktree*. That is the most expensive assumption on this board:

  * 5-6 Sep — 28 pre-review-gate bounces against a red baseline the cards did
    not cause;
  * 9-04 — four cards built, Rodge-approved and marked ``done`` with the work
    uncommitted, existing in no branch and no reflog;
  * 9-07 — Steve-o found the deliverable existed only as two DIVERGED branches
    (UI ``32f2496``, backend ``1687fec``) which he had to hand-stage to test at
    all, so nothing checkoutable was the thing he verified.

Every one of those is a card asserting its own correctness. This moves the
verdict off the machine that wrote the code and onto a check run GitHub
produced from a clean checkout.

Richie chose the fleet-side gate over GitHub branch protection (09-07) — the
repo stays private and there is no subscription. The trade is explicit: GitHub
would enforce this even if our code were wrong, whereas this can fail open if
*we* are wrong. So it is written to the standard that trade demands.

FAIL CLOSED — deliberately the OPPOSITE of the uncommitted-work guard
--------------------------------------------------------------------
That guard fails open because uncertainty there costs a false refusal on work
that is safely on disk. Here uncertainty means *"I cannot see a green check"*,
and treating that as green is the precise failure this exists to prevent. An
API error, a rate limit, a missing PR, an unresolvable SHA, a pending run: all
BLOCK, each naming which one it was.

"I cannot tell" and "it is fine" are not the same answer. Reading them as the
same is what let root serve stale code for 7.5 hours behind a GREEN pre-flight.

SCOPED, so fail-closed cannot strand the whole board
-----------------------------------------------------
The gate applies ONLY to a card whose tenant declares ``ci_gate`` in
``~/.hermes/kanban-tenants.json``. A tenant with no CI is out of scope and
passes straight through — fail-closed within scope, absent outside it. Adding a
repo to the gate is a one-line config change, and removing it is the rollback.

    "backupbrain": {
      ...
      "ci_gate": {"check_name": "test-gate", "repo": "meatacular/backupbrain"}
    }

It also never crashes the run. Closing the run is how work gets lost; a repeat
refusal points the worker at ``kanban_block``, which the escalator routes.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Optional

HERMES_HOME_ROOT = Path(__file__).resolve().parent.parent.parent
TENANTS = HERMES_HOME_ROOT / "kanban-tenants.json"

# A check that has not finished is NOT a pass. Naming the states explicitly
# rather than testing `!= "failure"` keeps a new GitHub conclusion (or a typo)
# from silently reading as success.
GREEN = {"success"}
NEUTRAL = {"neutral", "skipped"}  # reported, never treated as green


class GateResult:
    """Why the gate decided what it decided. `reason` is shown to the worker."""

    def __init__(self, blocked: bool, reason: str = "", detail: str = ""):
        self.blocked = blocked
        self.reason = reason
        self.detail = detail


def _tenant_ci(tenant: Optional[str]) -> Optional[dict]:
    """The tenant's ci_gate config, or None when the tenant is out of scope."""
    if not tenant:
        return None
    try:
        data = json.loads(TENANTS.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — a missing/broken map means "no tenant opted in"
        return None
    cfg = (data.get(tenant) or {}).get("ci_gate")
    if not isinstance(cfg, dict) or not cfg.get("check_name") or not cfg.get("repo"):
        return None
    return cfg


def _gh(args: list[str], cwd: str, timeout: int = 45) -> tuple[int, str, str]:
    """Run `gh` with the EXISTING credential (Richie, 09-07: no new PAT).

    Never raises: a missing binary or a timeout is a status code like any other,
    because the caller must be able to say *which* uncertainty it hit.
    """
    try:
        p = subprocess.run(["gh", *args], cwd=cwd, capture_output=True,
                           text=True, timeout=timeout)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except FileNotFoundError:
        return 127, "", "gh not found on PATH"
    except subprocess.TimeoutExpired:
        return 124, "", f"gh timed out after {timeout}s"
    except Exception as exc:  # noqa: BLE001
        return 1, "", f"gh failed: {exc}"


def evaluate(workspace_path: str, tenant: Optional[str], branch: Optional[str] = None) -> GateResult:
    """Decide whether this card may report done.

    Every `return GateResult(True, ...)` below is a distinct, nameable reason.
    A single generic "not green" would make the gate impossible to debug and
    therefore impossible to trust — the reason IS the product here.
    """
    cfg = _tenant_ci(tenant)
    if cfg is None:
        return GateResult(False, detail="tenant not in scope for the CI gate")

    check_name = cfg["check_name"]
    repo = cfg["repo"]
    ws = workspace_path or ""
    if not ws or not os.path.isdir(ws):
        return GateResult(True, reason=f"workspace {ws!r} is not a directory, so no PR can be resolved")

    if not branch:
        rc, out, err = _gh(["rev-parse"], ws)  # cheap probe that gh works at all
        rc2, branch_out, _ = _gh(["pr", "view", "--json", "headRefName", "-q", ".headRefName"], ws)
        if rc2 != 0 or not branch_out:
            return GateResult(True, reason=(
                "no open pull request found for this workspace. The CI gate needs a PR so "
                "GitHub can produce a check run from a clean checkout. Push your branch and "
                "open one: `gh pr create --fill`."), detail=err or branch_out)
        branch = branch_out

    rc, out, err = _gh([
        "pr", "view", "--json", "number,headRefOid,state,statusCheckRollup",
    ], ws)
    if rc != 0 or not out:
        return GateResult(True, reason=(
            f"could not read the pull request from GitHub ({err or 'no output'}). "
            "The gate fails CLOSED: an unreadable check is not a green one."), detail=err)

    try:
        pr = json.loads(out)
    except Exception as exc:  # noqa: BLE001
        return GateResult(True, reason=f"GitHub returned unparseable PR JSON ({exc})")

    if (pr.get("state") or "").upper() != "OPEN":
        return GateResult(True, reason=f"the pull request is {pr.get('state')!r}, not OPEN")

    sha = pr.get("headRefOid") or ""
    rollup = pr.get("statusCheckRollup") or []
    named = [c for c in rollup if (c.get("name") or c.get("context")) == check_name]

    if not named:
        seen = sorted({(c.get("name") or c.get("context") or "?") for c in rollup})
        return GateResult(True, reason=(
            f"no check named {check_name!r} on head {sha[:8]}. Checks present: "
            f"{', '.join(seen) or 'none'}. Either CI has not started, or the workflow "
            f"is not installed on this branch."))

    check = named[0]
    conclusion = (check.get("conclusion") or "").lower()
    status = (check.get("status") or "").lower()

    if status and status != "completed":
        return GateResult(True, reason=(
            f"{check_name} is {status!r} on head {sha[:8]} — still running. "
            f"Pending is not green; wait for it (`gh pr checks --watch`) and complete again."))
    if conclusion in GREEN:
        return GateResult(False, detail=f"{check_name} success on {sha[:8]} (PR #{pr.get('number')})")
    if conclusion in NEUTRAL:
        return GateResult(True, reason=(
            f"{check_name} concluded {conclusion!r} on head {sha[:8]}. A skipped or neutral "
            f"check means the suite did not run; that is not evidence of a passing build."))
    return GateResult(True, reason=(
        f"{check_name} FAILED on head {sha[:8]} (conclusion {conclusion!r}). "
        f"Read the failure with `gh pr checks` and `gh run view --log-failed`, fix, push, "
        f"and complete again once it is green."))
