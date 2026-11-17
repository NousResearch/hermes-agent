"""Opt-in work-item claims on the existing delegation construction and settlement paths.

Ordinary context is untouched. The explicit context envelope selects a record in
its parent's scoped SessionDB; it cannot select a database or grant authority.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from hermes_cli.task_delivery.store import DeliveryError, WorkLedger, logical_session

logger = logging.getLogger(__name__)
_MARKER = "[HERMES_WORK_ITEM_V1]"
_REQUIRED = frozenset({"session_id", "task_id", "role"})
_ALLOWED = _REQUIRED | {"base", "files"}


@dataclass
class WorkClaim:
    ledger: WorkLedger
    session_id: str
    task_id: str
    owner: str
    generation: int
    context: str
    settled: bool = False
    started: bool = False
    dispatched: bool = False


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise DeliveryError("Managed work context contains a duplicate JSON field")
        result[key] = value
    return result


def _text(value, name, limit):
    if not isinstance(value, str) or not value.strip() or len(value) > limit or "\x00" in value:
        raise DeliveryError(f"Managed work {name} must be nonempty text of at most {limit} characters")
    return value


def parse_context(context):
    """Return the validated opt-in reference, or None for ordinary context."""
    if not isinstance(context, str) or not context.startswith(_MARKER):
        return None
    lines = context.split("\n", 3)
    if len(lines) < 3 or lines[0].rstrip("\r") != _MARKER or lines[2].strip():
        raise DeliveryError("Managed work context requires its marker, one JSON line, and a blank line")
    try:
        reference = json.loads(lines[1], object_pairs_hook=_unique_object)
    except (TypeError, ValueError) as exc:
        raise DeliveryError("Managed work context must contain strict JSON without duplicate fields") from exc
    if not isinstance(reference, dict) or not _REQUIRED <= reference.keys() or reference.keys() - _ALLOWED:
        raise DeliveryError("Managed work context has missing or unsupported fields")
    _text(reference["session_id"], "session_id", 512)
    _text(reference["task_id"], "task_id", 256)
    if reference["role"] not in ("implement", "review"):
        raise DeliveryError("Managed work role must be implement or review")
    if "base" in reference and (not isinstance(reference["base"], str) or len(reference["base"]) > 256):
        raise DeliveryError("Managed work base must be text of at most 256 characters")
    paths = reference.get("files", [])
    if not isinstance(paths, list) or len(paths) > 128:
        raise DeliveryError("Managed work files must be an array of at most 128 paths")
    for path in paths:
        _text(path, "file", 4096)
    return reference


def _scoped_db(parent):
    from hermes_state import AsyncSessionDB
    db = getattr(parent, "_session_db", None)
    if isinstance(db, AsyncSessionDB):
        db = db._db
    if db is None or not callable(getattr(db, "_execute_write", None)) or not callable(getattr(db, "_read_ctx", None)):
        raise DeliveryError("Managed delegation requires the parent's scoped SessionDB")
    return db


def _validate_parent(parent, db, reference, route):
    if logical_session(db, reference["session_id"]) != logical_session(db, getattr(parent, "session_id", None)):
        raise DeliveryError("Managed work session does not match the actual parent session")
    profile = getattr(route, "profile_home", None)
    if profile is not None:
        path = getattr(db, "db_path", None)
        if not isinstance(path, (str, Path)) or Path(profile).resolve() != Path(path).resolve().parent:
            raise DeliveryError("Managed delegation cannot cross the parent's profile boundary")


def _claim_commands(item):
    claim = item["claim"]
    flags = ["--session", item["session_id"], "--task", item["task_id"],
             "--owner", claim["owner"], "--generation", str(claim["generation"])]
    action = "review" if claim["role"] == "review" else "deliver"
    deliver_args = (["--file", "<review-report.json>"] if action == "review" else
                    ["--file", "<artifact-path>", "--summary", "<delivery-summary>"])
    return {"renew": ["hermes", "work", "renew", *flags],
            action: ["hermes", "work", action, *flags, *deliver_args]}


def _claim_context(original, item):
    claim = item["claim"]
    details = {"session_id": item["session_id"], "task_id": item["task_id"],
               "owner": claim["owner"], "generation": claim["generation"], "role": claim["role"],
               "expires_at": claim["expires_at"], "commands": _claim_commands(item),
               "candidate": item.get("candidate"), "acceptance_criteria": item["acceptance_criteria"]}
    # Keep the opt-in marker first so a durable continuation cannot lose this binding.
    marker, reference_line, body = original.split("\n", 2)
    return (marker + "\n" + reference_line + "\n\n[HERMES_WORK_CLAIM_V1]\n" + json.dumps(details, ensure_ascii=False)
            + "\nThe commands above are argument arrays, not shell text; replace only their file/summary placeholders. "
            "Use the role's deliver/review command to publish its concrete artifact/report. "
            "For long tasks run hermes work renew with this owner and generation before expiry "
            "(default lease: 1800 seconds). Finishing execution alone does not accept the work.\n\n"
            + body.lstrip("\r\n"))


def prepare_claims(tasks, routes, parent, *, rebuilding=False):
    """Validate the complete batch before claiming; no child may exist yet."""
    references = {i: reference for i, task in enumerate(tasks)
                  if (reference := parse_context(task.get("context"))) is not None}
    if not references:
        return {}
    if rebuilding:
        raise DeliveryError("Managed work cannot continue an old claim; reconcile it and spawn a new managed task")
    db = _scoped_db(parent)
    for index, reference in references.items():
        _validate_parent(parent, db, reference, routes[index])
    ledger, claims = WorkLedger(db), {}
    try:
        for index, reference in references.items():
            if routes[index].error:
                continue
            owner = "delegation-" + uuid.uuid4().hex
            item = ledger.claim(reference["session_id"], reference["task_id"], owner=owner,
                                role=reference["role"], request_id=owner,
                                base=reference.get("base", ""), files=reference.get("files", []))
            claim = item["claim"]
            binding = WorkClaim(ledger, item["session_id"], item["task_id"], owner,
                                claim["generation"], "")
            claims[index] = binding
            binding.context = _claim_context(tasks[index]["context"], item)
    except Exception:
        release_claims(claims.values(), "Managed delegation preflight did not start any worker")
        raise
    return claims


def release_claims(claims, reason):
    """Release only this batch's newly acquired leases; retain failures for reconciliation."""
    errors = []
    for claim in claims:
        if claim.settled:
            continue
        try:
            claim.ledger.release(claim.session_id, claim.task_id, owner=claim.owner,
                                 generation=claim.generation, reason=reason)
            claim.settled = True
        except (DeliveryError, OSError, sqlite3.Error) as exc:
            errors.append(type(exc).__name__)
    if errors:
        raise DeliveryError("Managed claim cleanup failed; reconcile retained claims before retrying: " + ", ".join(errors))


def release_unstarted(child):
    claim = getattr(child, "_delegate_work_claim", None)
    if isinstance(claim, WorkClaim):
        release_claims([claim], "Delegated worker did not start")


def begin_execution(child):
    """A queued worker must still own its live claim when it actually starts."""
    claim = getattr(child, "_delegate_work_claim", None)
    if not isinstance(claim, WorkClaim):
        return
    live = claim.ledger.get(claim.session_id, claim.task_id).get("claim")
    if (claim.settled or not live or (live["owner"], live["generation"]) != (claim.owner, claim.generation)
            or live["expires_at"] <= time.time()):
        raise DeliveryError("Managed execution claim expired or was superseded; reconcile before dispatch")
    claim.started = True


def mark_dispatched(child):
    claim = getattr(child, "_delegate_work_claim", None)
    if isinstance(claim, WorkClaim):
        claim.dispatched = True


def needs_unstarted_cleanup(child):
    claim = getattr(child, "_delegate_work_claim", None)
    return (isinstance(claim, WorkClaim) and not claim.settled
            and not claim.started and not claim.dispatched)


def finish_execution(child, entry):
    """Record execution separately from artifact delivery, review and acceptance."""
    claim = getattr(child, "_delegate_work_claim", None)
    if not isinstance(claim, WorkClaim) or claim.settled:
        return entry
    try:
        claim.ledger.finish_execution(
            claim.session_id, claim.task_id, owner=claim.owner, generation=claim.generation,
            status=entry.get("status", "error"), result=json.dumps(entry, ensure_ascii=False, default=str),
        )
        claim.settled = True
    except (ValueError, TypeError, OSError, sqlite3.Error) as exc:
        logger.warning("Managed delegation execution could not be recorded: %s", type(exc).__name__)
        return {**entry, "status": "error", "execution_status": entry.get("status"),
                "failure_reason": "work_tracking_failed",
                "error": "Managed execution requires reconciliation; its result could not be recorded"}
    return entry
