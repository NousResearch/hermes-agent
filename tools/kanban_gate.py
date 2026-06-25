"""kanban_gate.py — AP-4031 + AP-4032 mechanical kanban-complete gate.

Houses the gate logic lifted from feat/ap-4032-gate-skip-verifier
(hermes-agent commit 2000c99a5) and refactored out of tools/kanban_tools.py
for surgical merge to main without conflict on the larger _handle_complete
function. Imported by tools/kanban_tools.py::_handle_complete.

Ships disabled (HERMES_KANBAN_GATE_REQUIRED=0 default per BC-9). Operators
opt-in per the 14-day observation window documented in
docs/migration-notes/ap-4031-kanban-gate-2026-06-24.md.

Trust-boundary fix (WAGS post-build review, 2026-06-24): the verifier and
the evidence-chain write path both require HERMES_PROFILE=wags before any
gate_skip_authorisation entry is trusted. See ADR section 10 risk row
"sev-2 actor-spoof".
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── Constants ───────────────────────────────────────────────────────────

# AP-4031: when HERMES_KANBAN_GATE_REQUIRED=1, _handle_complete refuses
# to close any non-wags card unless a gate_signature evidence entry exists
# in the evidence chain. Default is 0 (off) for one release cycle per
# the BC-9 additivity rules in the ADR.
GATE_REQUIRED_ENV = "HERMES_KANBAN_GATE_REQUIRED"
GATE_BYPASS_SIGNER = "wags"

# Default location of the gap_detection helper. Overridable via env so
# tests and hermes-agent-installed venvs can point at the right copy.
GAP_DETECTION_HELPER_ENV = "HERMES_GAP_DETECTION_PY"

# AP-4032: gate_skip_verifier.py implements the 3-branch verification
# path that closes the bypass on AP-4031's permissive gate_skip_reason
# short-circuit. Located the same way as gap_detection.py so workers
# running in totum-src's AOS_WORKSPACE pick up the workspace copy.
GATE_SKIP_VERIFIER_ENV = "HERMES_GATE_SKIP_VERIFIER_PY"

# Subprocess timeout for the gate check. 2s is generous (target p99
# 200ms per ADR §3.4); cap protects kanban_complete from a wedged
# subprocess wedging the worker.
_GATE_CHECK_TIMEOUT_S = 2.0


# ── Helpers ─────────────────────────────────────────────────────────────


def _gap_detection_helper() -> str:
    """Locate scripts/gap_detection.py for the subprocess gate check.

    Lookup order:
      1. HERMES_GAP_DETECTION_PY env var (explicit override)
      2. AOS_WORKSPACE/scripts/gap_detection.py (the workspace the
         dispatcher spawned us in)
      3. <hermes-agent-root>/../scripts/gap_detection.py (sibling
         repo fallback for the standard workspace layout — totum-src
         lives one directory up from hermes-agent)
    """
    explicit = os.environ.get(GAP_DETECTION_HELPER_ENV, "").strip()
    if explicit and os.path.isfile(explicit):
        return explicit
    ws = os.environ.get("AOS_WORKSPACE", "").strip()
    if ws:
        candidate = os.path.join(ws, "scripts", "gap_detection.py")
        if os.path.isfile(candidate):
            return candidate
    here = os.path.dirname(os.path.abspath(__file__))
    sibling = os.path.abspath(
        os.path.join(here, "..", "..", "scripts", "gap_detection.py")
    )
    if os.path.isfile(sibling):
        return sibling
    return ""


def _gate_skip_verifier_helper() -> str:
    """Locate scripts/gate_skip_verifier.py (AP-4032). Same lookup order
    as _gap_detection_helper()."""
    explicit = os.environ.get(GATE_SKIP_VERIFIER_ENV, "").strip()
    if explicit and os.path.isfile(explicit):
        return explicit
    ws = os.environ.get("AOS_WORKSPACE", "").strip()
    if ws:
        candidate = os.path.join(ws, "scripts", "gate_skip_verifier.py")
        if os.path.isfile(candidate):
            return candidate
    here = os.path.dirname(os.path.abspath(__file__))
    sibling = os.path.abspath(
        os.path.join(here, "..", "..", "scripts", "gate_skip_verifier.py")
    )
    if os.path.isfile(sibling):
        return sibling
    return ""


def gate_required() -> bool:
    """True iff the env-var gate is on for this process."""
    raw = os.environ.get(GATE_REQUIRED_ENV, "").strip()
    if not raw:
        return False
    return raw not in ("0", "false", "False", "no", "off")


# ── Gate pre-check ─────────────────────────────────────────────────────


def gate_pre_check(task_id: str, metadata: dict, get_task_fn) -> Optional[str]:
    """Run the AP-4031 mechanical gate, returning an error message on
    refusal or None on pass.

    Decision tree (per ADR §3.3 + AP-4032 §3.2):
      - HERMES_KANBAN_GATE_REQUIRED=0 → pass (backward compatible).
      - assignee == 'wags' → pass (wags signs his own cards).
      - metadata.gate_skip_reason non-empty → run the AP-4032 3-branch
        verifier (operator | wags | auto). Pass iff verifier accepts;
        refuse otherwise. The verifier's verdict replaces the previous
        permissive short-circuit (the bypass AP-4032 closes).
      - else: run gap_detection.py check-card-completeness. Exit 0
        passes; non-zero refuses with the prescribed error message.

    The check is intentionally lock-free and side-effect-free (it only
    reads the evidence chain and the kanban DB). Failures from the
    subprocess (timeout, missing helper, exception) DEGRADE OPEN — the
    gate refuses rather than silently passing — because a silent pass
    would defeat the purpose of the gate.

    `get_task_fn` is the caller-supplied task lookup; passed in to avoid
    a circular import on hermes_cli.kanban_db (kept inside tools/kanban_tools).
    """
    if not gate_required():
        return None
    if not task_id:
        return None
    try:
        task = get_task_fn(task_id)
    except Exception as e:
        logger.exception("kanban_complete: gate pre-check db lookup failed")
        return (
            f"cannot complete {task_id}: AP-4031 gate pre-check could not "
            f"read task row ({e}); refusing rather than bypassing"
        )
    if task is None:
        # Unknown id — let kb.complete_task surface its own error.
        return None
    if (getattr(task, "assignee", "") or "").strip() == GATE_BYPASS_SIGNER:
        return None
    skip_reason = (metadata or {}).get("gate_skip_reason") or ""
    skip_reason = str(skip_reason).strip()
    if skip_reason:
        # AP-4032: instead of short-circuiting on the bare field, run the
        # 3-branch verifier. This closes the bypass that AP-4031 left open
        # by treating any non-empty gate_skip_reason as a pass.
        return _gate_skip_check(task_id, task, metadata or {}, skip_reason)
    helper = _gap_detection_helper()
    if not helper:
        return (
            f"cannot complete {task_id}: AP-4031 gate required but "
            f"scripts/gap_detection.py not found (set "
            f"HERMES_GAP_DETECTION_PY to override); see AP-4031"
        )
    try:
        proc = subprocess.run(
            [sys.executable, helper, "check-card-completeness", task_id],
            capture_output=True,
            text=True,
            timeout=_GATE_CHECK_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return (
            f"cannot complete {task_id}: AP-4031 gate check timed out "
            f"after {_GATE_CHECK_TIMEOUT_S}s; refusing rather than bypassing"
        )
    except Exception as e:
        logger.exception("kanban_complete: gate subprocess failed")
        return (
            f"cannot complete {task_id}: AP-4031 gate subprocess error ({e}); "
            f"refusing rather than bypassing"
        )
    if proc.returncode == 0:
        return None
    if proc.returncode == 1:
        return (
            f"cannot complete {task_id}: no gate_signature evidence entry; "
            f"see AP-4031 for the gate pattern; operator override via "
            f"gate_skip_reason + gate_skip_actor fields in metadata (AP-4032)"
        )
    return (
        f"cannot complete {task_id}: gate check error (exit "
        f"{proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
    )


def _gate_skip_check(
    task_id: str, task: Any, metadata: dict, skip_reason: str
) -> Optional[str]:
    """AP-4032 3-branch gate-skip verifier dispatch.

    Spawns scripts/gate_skip_verifier.py check with the card's role + the
    skip metadata. On accept, appends a kind=gate_skip evidence entry to
    the chain (audit trail per ADR §3.2 step 6). The verifier itself
    does NOT write evidence; the caller invokes _emit_gate_skip_evidence
    after a positive verdict.
    """
    helper = _gate_skip_verifier_helper()
    if not helper:
        return (
            f"cannot complete {task_id}: AP-4032 gate_skip verifier "
            f"(scripts/gate_skip_verifier.py) not found (set "
            f"HERMES_GATE_SKIP_VERIFIER_PY to override); see AP-4032"
        )
    actor = (metadata.get("gate_skip_actor") or "").strip()
    cmd = [
        sys.executable, helper, "check",
        "--card-id", task_id,
        "--reason", skip_reason,
        "--actor", actor or "operator",
        "--card-role", (getattr(task, "assignee", "") or "").strip(),
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=_GATE_CHECK_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return (
            f"cannot complete {task_id}: AP-4032 gate_skip verifier timed "
            f"out after {_GATE_CHECK_TIMEOUT_S}s; refusing rather than bypassing"
        )
    except Exception as e:
        logger.exception("kanban_complete: gate_skip verifier subprocess failed")
        return (
            f"cannot complete {task_id}: AP-4032 gate_skip verifier "
            f"subprocess error ({e}); refusing rather than bypassing"
        )
    if proc.returncode not in (0, 1):
        return (
            f"cannot complete {task_id}: AP-4032 gate_skip verifier error "
            f"(exit {proc.returncode}): "
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )
    verdict: dict[str, Any] = {}
    try:
        out = proc.stdout.strip()
        if out:
            verdict = json.loads(out.splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        verdict = {}
    accepted = bool(verdict.get("accepted"))
    if proc.returncode == 0:
        accepted = True
    if not accepted:
        reason = verdict.get("refusal_reason") or (
            f"AP-4032 gate_skip verifier refused (exit {proc.returncode})"
        )
        return (
            f"cannot complete {task_id}: {reason}"
        )
    # Accepted. Emit the audit-trail evidence entry per ADR §3.2 step 6.
    soft_pass = bool(verdict.get("soft_pass"))
    rule_id = verdict.get("rule_id")
    emit_err = _emit_gate_skip_evidence(
        task_id=task_id,
        actor=actor or "operator",
        reason=skip_reason,
        soft_pass=soft_pass,
        rule_id=rule_id,
    )
    if emit_err:
        # Audit trail emission failure → degrade open (refuse). Without
        # the audit entry the skip is invisible to Wags's weekly CNS
        # review, which is the whole point of AP-4032.
        return (
            f"cannot complete {task_id}: AP-4032 gate_skip accepted but "
            f"audit-trail evidence entry failed to write ({emit_err}); "
            f"refusing rather than bypassing"
        )
    return None


def _emit_gate_skip_evidence(
    *,
    task_id: str,
    actor: str,
    reason: str,
    soft_pass: bool,
    rule_id: Optional[str],
) -> Optional[str]:
    """Append a kind=gate_skip entry to the evidence chain (audit trail
    per ADR §3.2 step 6). Returns None on success, or an error message
    on failure.

    Uses the same evidence_chain.py helper as everything else. The
    summary includes the soft-pass / rule_id markers so downstream
    auditors (Wags weekly CNS review) can see at a glance which skip
    path authorised this close.
    """
    ws = os.environ.get("AOS_WORKSPACE", "").strip()
    candidates: list[str] = []
    if ws:
        candidates.append(os.path.join(ws, "scripts", "evidence_chain.py"))
    here = os.path.dirname(os.path.abspath(__file__))
    candidates.append(
        os.path.abspath(
            os.path.join(here, "..", "..", "scripts", "evidence_chain.py")
        )
    )
    ev_helper = next((c for c in candidates if os.path.isfile(c)), "")
    if not ev_helper:
        return "scripts/evidence_chain.py not found in workspace or sibling path"
    summary_parts = [f"gate_skip accepted for {task_id}"]
    if soft_pass:
        summary_parts.append("[operator soft-pass: AP-NNNN-A not yet shipped]")
    if rule_id:
        summary_parts.append(f"[rule_id={rule_id}]")
    summary = " ".join(summary_parts)
    cmd = [
        sys.executable, ev_helper, "append",
        "--kind", "gate_skip",
        "--actor", actor,
        "--gate-skip-reason", reason,
        "--gate-skip-actor", actor,
        "--card-id", task_id,
        "--summary", summary[:400],
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=_GATE_CHECK_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return f"evidence_chain.py append timed out after {_GATE_CHECK_TIMEOUT_S}s"
    except Exception as e:
        return f"evidence_chain.py append subprocess error: {e}"
    if proc.returncode != 0:
        return (
            f"evidence_chain.py append failed (exit {proc.returncode}): "
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )
    return None