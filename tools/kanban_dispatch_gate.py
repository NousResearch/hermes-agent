"""kanban_dispatch_gate.py — AP-4033 verified-dispatch on kanban-create.

Symmetric mirror of kanban_gate.py (AP-4031+AP-4032) applied to the
dispatch side of the kanban DB. Every kanban_create must either be
performed by an allowed dispatcher (HERMES_PROFILE in
HERMES_KANBAN_DISPATCHERS, default ['operator']) OR carry a verified
kind=dispatch metadata field backed by a prior kind=dispatch evidence
entry signed by an allowed dispatcher.

Ships disabled (HERMES_KANBAN_DISPATCH_GATE=0 default per BC-9). Operators
opt-in per the 14-day observation window documented in
docs/migration-notes/ap-4033-kanban-dispatch-2026-06-24.md.

Trust-boundary mirror of AP-4032 (kanban_gate.py): the verifier and
the evidence-chain write path on the dispatch side require either
HERMES_PROFILE in HERMES_KANBAN_DISPATCHERS (for dispatch /
dispatch_override / spawn_refused kinds) OR HERMES_PROFILE=wags (for
dispatch_override_authorisation kind). Same actor-spoof defense.

Extracted from AP-4033 commit 643646054 (totum-src main) for surgical
merge to hermes-agent main without conflict on _handle_create.
Imported by tools/kanban_tools.py::_handle_create.
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

# AP-4033: when HERMES_KANBAN_DISPATCH_GATE=1, _handle_create refuses
# unless the caller is an allowed dispatcher (per the env-var list)
# or the card carries a verified dispatch entry. Default is 0 (off)
# for one release cycle per BC-9.
DISPATCH_GATE_ENV = "HERMES_KANBAN_DISPATCH_GATE"

# Allowed dispatcher profiles for the write-side gate. Default is
# ["operator"] only — Wags is excluded because including them in the
# dispatcher list would make their identity forgeable via the dispatch
# path, defeating the close-side HERMES_PROFILE=wags trust boundary.
# Configurable via HERMES_KANBAN_DISPATCHERS env var (comma-separated).
DISPATCHERS_ENV = "HERMES_KANBAN_DISPATCHERS"
DEFAULT_DISPATCHERS = ["operator"]

# Allowed approver profiles for the auto-rule approval check (separate
# from DISPATCHERS so operators dispatch while wags approves auto-rules).
# Configurable via HERMES_KANBAN_DISPATCH_APPROVERS env var.
APPROVERS_ENV = "HERMES_KANBAN_DISPATCH_APPROVERS"
DEFAULT_APPROVERS = ["wags"]

# Default location of the dispatch_gap_detection helper.
DISPATCH_GAP_DETECTION_HELPER_ENV = "HERMES_DISPATCH_GAP_DETECTION_PY"

# Default location of the dispatch_verifier helper.
DISPATCH_VERIFIER_HELPER_ENV = "HERMES_DISPATCH_VERIFIER_PY"

# Subprocess timeout for the gate check. 2s is generous (target p99
# 200ms); cap protects kanban_create from a wedged subprocess.
_GATE_CHECK_TIMEOUT_S = 2.0


# ── Helpers ─────────────────────────────────────────────────────────────


def dispatch_gate_required() -> bool:
    """True iff the env-var dispatch gate is on for this process."""
    raw = os.environ.get(DISPATCH_GATE_ENV, "").strip()
    if not raw:
        return False
    return raw not in ("0", "false", "False", "no", "off")


def _allowed_dispatchers() -> list[str]:
    """Return the configured list of allowed dispatcher profiles.

    Reads HERMES_KANBAN_DISPATCHERS env var (comma-separated). Falls back
    to ["operator"] per ADR §3.1 operator revisions (WAGS pre-build
    review 2026-06-25).
    """
    raw = os.environ.get(DISPATCHERS_ENV, "").strip()
    if raw:
        return [s.strip() for s in raw.split(",") if s.strip()]
    return list(DEFAULT_DISPATCHERS)


def _allowed_approvers() -> list[str]:
    """Return the configured list of allowed approver profiles.

    Reads HERMES_KANBAN_DISPATCH_APPROVERS env var (comma-separated).
    Falls back to ["wags"] — approvers approve auto-rules (different
    role from dispatchers who actually create cards).
    """
    raw = os.environ.get(APPROVERS_ENV, "").strip()
    if raw:
        return [s.strip() for s in raw.split(",") if s.strip()]
    return list(DEFAULT_APPROVERS)


def _dispatch_gap_detection_helper() -> str:
    """Locate scripts/dispatch_gap_detection.py for the subprocess gate check.

    Lookup order:
      1. HERMES_DISPATCH_GAP_DETECTION_PY env var (explicit override)
      2. AOS_WORKSPACE/scripts/dispatch_gap_detection.py (workspace copy)
      3. <hermes-agent-root>/../scripts/dispatch_gap_detection.py (sibling
         repo fallback — totum-src lives one directory up from hermes-agent)
    """
    explicit = os.environ.get(DISPATCH_GAP_DETECTION_HELPER_ENV, "").strip()
    if explicit and os.path.isfile(explicit):
        return explicit
    ws = os.environ.get("AOS_WORKSPACE", "").strip()
    if ws:
        candidate = os.path.join(ws, "scripts", "dispatch_gap_detection.py")
        if os.path.isfile(candidate):
            return candidate
    here = os.path.dirname(os.path.abspath(__file__))
    sibling = os.path.abspath(
        os.path.join(here, "..", "..", "scripts", "dispatch_gap_detection.py")
    )
    if os.path.isfile(sibling):
        return sibling
    return ""


def _dispatch_verifier_helper() -> str:
    """Locate scripts/dispatch_verifier.py (AP-4033 override path)."""
    explicit = os.environ.get(DISPATCH_VERIFIER_HELPER_ENV, "").strip()
    if explicit and os.path.isfile(explicit):
        return explicit
    ws = os.environ.get("AOS_WORKSPACE", "").strip()
    if ws:
        candidate = os.path.join(ws, "scripts", "dispatch_verifier.py")
        if os.path.isfile(candidate):
            return candidate
    here = os.path.dirname(os.path.abspath(__file__))
    sibling = os.path.abspath(
        os.path.join(here, "..", "..", "scripts", "dispatch_verifier.py")
    )
    if os.path.isfile(sibling):
        return sibling
    return ""


# ── Gate pre-check ─────────────────────────────────────────────────────


def dispatch_gate_pre_check(
    card_metadata: dict,
    caller_profile: Optional[str],
    has_dispatch_entry: bool,
) -> Optional[str]:
    """Run the AP-4033 dispatch gate pre-check.

    Decision tree (per ADR §3.4):
      - HERMES_KANBAN_DISPATCH_GATE=0 → pass (backward compatible).
      - caller_profile in HERMES_KANBAN_DISPATCHERS → pass (caller IS
        the dispatcher; the dispatch entry will be written post-create).
      - card_metadata.dispatched_by matches a verified dispatch entry
        → pass.
      - card_metadata has dispatch_override field with actor + reason
        → run the 3-branch verifier; pass iff accepted.
      - else → refuse with the prescribed error message.

    Returns None on pass, error-message string on refusal.

    `has_dispatch_entry` is the caller-supplied check: did the kanban
    DB row (or its prior entries) carry a verified dispatch evidence
    entry? Passed in to keep this module free of internal DB imports.
    """
    if not dispatch_gate_required():
        return None
    # Caller IS the dispatcher → pass; the dispatch entry will be
    # appended by the caller immediately after create_task.
    if caller_profile and caller_profile in _allowed_dispatchers():
        return None
    # Card metadata carries a dispatch_override → run the 3-branch verifier.
    override = (card_metadata or {}).get("dispatch_override")
    if override and isinstance(override, dict):
        return _dispatch_override_check(override)
    # Card has a prior verified dispatch entry → pass.
    if has_dispatch_entry:
        return None
    return (
        "AP-4033 dispatch gate: no verified dispatch entry for this card. "
        "Either (a) the caller's HERMES_PROFILE must be in "
        "HERMES_KANBAN_DISPATCHERS (default ['operator']), (b) the card "
        "metadata must carry a dispatch_override field for the 3-branch "
        "verifier, or (c) a kind=dispatch evidence entry must already "
        "exist for this card_id. See docs/migration-notes/"
        "ap-4033-kanban-dispatch-2026-06-24.md."
    )


def _dispatch_override_check(override: dict) -> Optional[str]:
    """AP-4033 dispatch-side 3-branch override verification.

    Mirrors kanban_gate.py::_gate_skip_check. Spawns
    scripts/dispatch_verifier.py check with the override metadata.
    On accept, the verifier appends a kind=dispatch_override evidence
    entry to the chain (audit trail per ADR §3.4).
    """
    helper = _dispatch_verifier_helper()
    if not helper:
        return (
            "AP-4033 dispatch_override check requires "
            "scripts/dispatch_verifier.py (set "
            "HERMES_DISPATCH_VERIFIER_PY to override); see AP-4033"
        )
    actor = (override.get("actor") or "").strip()
    reason = (override.get("reason") or "").strip()
    card_id = (override.get("card_id") or "").strip()
    cmd = [
        sys.executable, helper, "check",
        "--card-id", card_id,
        "--reason", reason,
        "--actor", actor or "operator",
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=_GATE_CHECK_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return (
            f"AP-4033 dispatch_override verifier timed out after "
            f"{_GATE_CHECK_TIMEOUT_S}s; refusing rather than bypassing"
        )
    except Exception as e:
        logger.exception("kanban_create: dispatch_override verifier subprocess failed")
        return (
            f"AP-4033 dispatch_override verifier subprocess error ({e}); "
            f"refusing rather than bypassing"
        )
    if proc.returncode not in (0, 1):
        return (
            f"AP-4033 dispatch_override verifier error "
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
        reason_str = verdict.get("refusal_reason") or (
            f"AP-4033 dispatch_override verifier refused (exit {proc.returncode})"
        )
        return f"AP-4033 dispatch_override refused: {reason_str}"
    return None