#!/usr/bin/env python3
"""Canonical Brain tools for free-Hermes operational persistence.

These tools are intentionally thin mechanical adapters. They do not decide
business meaning. The Hermes agent decides when durable operational state
exists, then calls these tools to persist canonical events or route-back state.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import datetime as dt
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import uuid
from typing import Any, Dict, Optional

try:
    from hermes_cli.config import load_config
except Exception:  # pragma: no cover - import-safe for tool discovery
    load_config = None  # type: ignore[assignment]

from tools.registry import registry, tool_error

from gateway.support_ops_team_registry import (
    SKYVISION_CONTROL_TOWER_CHANNEL_ID,
    TeamMember,
    resolve_team_member,
)

CANONICAL_BRAIN_ROOT = pathlib.Path("/opt/adventico-ai-platform/canonical-brain")
CLOUD_SQL_HELPER = CANONICAL_BRAIN_ROOT / "bin" / "cloud_sql_synthetic_write_gate.py"
EVENT_TABLE = "canonical_event_log"
MAX_ROUTE_BACK_MESSAGE_CHARS = 1900
CANONICAL_BRAIN_IO_TIMEOUT_SECONDS = 12
# A claimed route-back intent remains owned by the inserting worker while it
# completes the bounded send, live readback, receipt validation, and durable
# terminal append.  This exceeds the worst-case sum of those configured I/O
# windows; a retry must not race the worker by declaring it blocked early.
ROUTE_BACK_INTENT_LEASE_SECONDS = 180
TASK_MODEL_EVENT_TYPES = {
    "task.plan.updated",
    "task.verification.recorded",
}
PROCESS_RECEIPT_EVENT_TYPES = {
    "approval.capability.recorded",
    "capability.check.recorded",
}
ALLOWED_EVENT_TYPES = {
    "case.note",
    "handoff.created",
    "handoff.waiting",
    "resolver.reply.received",
    "route_back.required",
    "route_back.intent.created",
    "route_back.sent",
    "route_back.blocked",
    "handoff.closed",
    "operational.note.needs_review",
    "semantic_interpreter.failed",
    "semantic_interpreter.skipped",
    "semantic_event.drafted",
    "person.alias.learned",
} | TASK_MODEL_EVENT_TYPES | PROCESS_RECEIPT_EVENT_TYPES
RECEIPT_REQUIRED_EVENT_TYPES = {"route_back.sent"} | PROCESS_RECEIPT_EVENT_TYPES
TASK_STEP_STATUSES = {"pending", "in_progress", "completed", "cancelled", "blocked"}
TASK_PLAN_STATES = {"active", "completed", "blocked", "cancelled"}
TASK_VERIFICATION_OUTCOMES = {"passed", "failed", "inconclusive"}
MAX_TASK_PLAN_CHARS = 64_000
MAX_TASK_STEPS = 64
MAX_TASK_CRITERIA = 32
MAX_TASK_COLLECTION_ITEMS = 64
MAX_TASK_PLAN_GRAPH_ROWS = 256
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CANONICAL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,159}$")
_CASE_ID_RE = re.compile(r"^case:[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}$")
FORBIDDEN_ROUTE_BACK_DM_KEYS = {
    "dm_channel_id",
    "direct_message_channel_id",
    "recipient_id",
    "dm_recipient_id",
}
FORBIDDEN_ROUTE_BACK_DM_VALUES = {"dm", "direct_message", "private_dm", "user_dm"}
SECRET_MARKERS = (
    "api_key=", "apikey=", "token=", "password=", "secret=",
    "authorization: bearer", "private_key", "BEGIN PRIVATE KEY",
)
SECRET_KEY_NAMES = {
    "token",
    "access_token",
    "password",
    "secret",
    "api_key",
    "apikey",
    "private_key",
    "authorization",
    "bearer",
    "credentials",
    "payment_credential",
}


def _utc_now() -> str:
    # Preserve sub-second append order.  Projection still uses explicit plan
    # revisions as its primary task-workspace ordering key, but unrelated
    # lifecycle events written in the same second must not be left to UUID
    # ordering.
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds")


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _hash(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8", errors="replace")).hexdigest()


def _event_uuid(
    idempotency_key: str,
    event_type: str = "",
    case_id: str = "",
) -> str:
    """Deterministic event UUID scoped by case + event type + lifecycle key.

    The lifecycle idempotency key can intentionally be shared across
    route_back.required -> route_back.sent/blocked transitions, so event_type is
    part of the event UUID while the raw key remains in payload for grouping.
    """
    return str(uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"canonical-brain:{case_id}:{event_type}:{idempotency_key}",
    ))


def _load_helper() -> Any:
    if not CLOUD_SQL_HELPER.exists():
        raise RuntimeError("canonical brain Cloud SQL helper missing")
    spec = importlib.util.spec_from_file_location("canonical_brain_cloud_sql_helper", CLOUD_SQL_HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load canonical brain Cloud SQL helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _bound_socket_io(sock: Any) -> None:
    """Apply a finite read/write deadline when the helper exposes a socket."""
    setter = getattr(sock, "settimeout", None)
    if callable(setter):
        setter(CANONICAL_BRAIN_IO_TIMEOUT_SECONDS)


def _normalize_secret_key(key: Any) -> str:
    return str(key or "").strip().casefold().replace("-", "_")


def _has_structured_secret_value(value: Any) -> bool:
    """Return True when a secret-keyed field actually carries content.

    Operational safety metadata often uses boolean flags such as
    ``{"secret": False}`` or ``{"payment_credential": False}`` to state that
    no sensitive value is present. Treating the key alone as a secret caused
    harmless Canonical Brain appends to fail closed. Still block non-empty
    values under credential-shaped keys before any helper/connect.
    """
    if value is None or value is False:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    if isinstance(value, (list, tuple, set, dict)) and not value:
        return False
    return True


def _contains_secret_like(value: Any) -> bool:
    """Return True for secret-looking values or structured secret keys.

    This is deliberately mechanical, not semantic. It recursively inspects
    dict/list payloads before any Cloud SQL helper load/connect so structured
    credentials such as {"token": "..."} are blocked even when the value does
    not contain marker strings like ``token=``.
    """
    if isinstance(value, dict):
        for key, nested in value.items():
            if (
                _normalize_secret_key(key) in SECRET_KEY_NAMES
                and _has_structured_secret_value(nested)
            ):
                return True
            if _contains_secret_like(nested):
                return True
        return False
    if isinstance(value, (list, tuple, set)):
        return any(_contains_secret_like(item) for item in value)
    text = str(value or "").casefold()
    return any(marker.casefold() in text for marker in SECRET_MARKERS)


def _block_secret_like_fields(**fields: Any) -> None:
    """Fail closed before any Cloud SQL helper/connect on secret-like content.

    Hermes decides operational meaning, but the adapter must mechanically ensure
    no secret-looking values are written into source/actor/payload/receipt/status
    surfaces.  Keep this broad and field-oriented rather than business-semantic.
    """
    for name, value in fields.items():
        if _contains_secret_like(value):
            raise ValueError(f"secret_like_content_blocked:{name}")


def _normalize_dict(value: Optional[Dict[str, Any]], name: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _get_session_env(name: str, default: str = "") -> str:
    """Read gateway session context without making tool discovery depend on it."""
    try:
        from gateway.session_context import get_session_env

        return str(get_session_env(name, default) or default)
    except Exception:
        return default


def _augment_source_refs_from_session_context(source_refs: Dict[str, Any]) -> Dict[str, Any]:
    """Fill mechanical source refs from the current gateway session when omitted.

    The model should pass exact refs when it has them. In live gateway runs the
    runtime already carries platform/chat/thread/message context; use that as a
    deterministic fallback before validation so operational appends do not fail
    merely because the model forgot to copy boilerplate refs into the tool call.
    """
    refs = dict(source_refs)
    changed = False

    env_fields = {
        "platform": "HERMES_SESSION_PLATFORM",
        "chat_id": "HERMES_SESSION_CHAT_ID",
        "thread_id": "HERMES_SESSION_THREAD_ID",
        "session_id": "HERMES_SESSION_ID",
        "session_key": "HERMES_SESSION_KEY",
        "user_id": "HERMES_SESSION_USER_ID",
        "user_name": "HERMES_SESSION_USER_NAME",
    }
    for ref_key, env_key in env_fields.items():
        if refs.get(ref_key):
            continue
        value = _get_session_env(env_key, "").strip()
        if value:
            refs[ref_key] = value
            changed = True

    if not (refs.get("message_id") or refs.get("event_ref") or refs.get("manual_ref")):
        message_id = _get_session_env("HERMES_SESSION_MESSAGE_ID", "").strip()
        if message_id:
            refs["message_id"] = message_id
            changed = True
        else:
            manual_parts = [
                str(refs.get("platform") or "").strip(),
                str(refs.get("chat_id") or "").strip(),
                str(refs.get("thread_id") or "").strip(),
                str(refs.get("session_id") or refs.get("session_key") or "").strip(),
            ]
            manual_ref = ":".join(part for part in manual_parts if part)
            if manual_ref:
                refs["manual_ref"] = f"hermes_session:{manual_ref}"
                changed = True

    if changed:
        refs.setdefault("source_ref_source", "hermes_session_context")
    return refs


def _normalize_list(value: Any, name: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return value


def _required_text(mapping: Dict[str, Any], key: str, path: str, *, max_chars: int = 4000) -> str:
    value = str(mapping.get(key) or "").strip()
    if not value:
        raise ValueError(f"{path}.{key} is required")
    if len(value) > max_chars:
        raise ValueError(f"{path}.{key} exceeds {max_chars} characters")
    return value


def _required_identifier(mapping: Dict[str, Any], key: str, path: str) -> str:
    value = _required_text(mapping, key, path, max_chars=160)
    if not _CANONICAL_ID_RE.fullmatch(value):
        raise ValueError(
            f"{path}.{key} must use only letters, digits, dot, underscore, colon, slash, or hyphen"
        )
    return value


def _bounded_object_list(value: Any, path: str, *, minimum: int = 0, maximum: int = MAX_TASK_COLLECTION_ITEMS) -> list[Dict[str, Any]]:
    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        raise ValueError(f"{path} must contain {minimum}..{maximum} objects")
    if not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{path} must contain only objects")
    return value


def _bounded_string_list(value: Any, path: str, *, minimum: int = 0, maximum: int = MAX_TASK_COLLECTION_ITEMS) -> list[str]:
    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        raise ValueError(f"{path} must contain {minimum}..{maximum} strings")
    normalized = [str(item or "").strip() for item in value]
    if any(not item for item in normalized):
        raise ValueError(f"{path} cannot contain empty values")
    return normalized


def _validate_task_plan_payload(payload: Dict[str, Any]) -> None:
    plan = payload.get("plan")
    if not isinstance(plan, dict):
        raise ValueError("task.plan.updated requires payload.plan")
    if len(_stable_json(plan)) > MAX_TASK_PLAN_CHARS:
        raise ValueError(f"payload.plan exceeds {MAX_TASK_PLAN_CHARS} characters")

    _required_identifier(plan, "plan_id", "payload.plan")
    _required_text(plan, "objective", "payload.plan", max_chars=4000)
    revision = plan.get("revision")
    if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
        raise ValueError("payload.plan.revision must be a positive integer")
    supersedes_plan_id = str(plan.get("supersedes_plan_id") or "").strip()
    if supersedes_plan_id:
        if not _CANONICAL_ID_RE.fullmatch(supersedes_plan_id):
            raise ValueError("payload.plan.supersedes_plan_id is not a safe identifier")
        supersedes_revision = plan.get("supersedes_plan_revision")
        if (
            not isinstance(supersedes_revision, int)
            or isinstance(supersedes_revision, bool)
            or supersedes_revision < 1
        ):
            raise ValueError(
                "payload.plan.supersedes_plan_revision must identify the exact predecessor revision"
            )
    state = str(plan.get("state") or "").strip()
    if state not in TASK_PLAN_STATES:
        raise ValueError(f"payload.plan.state must be one of {sorted(TASK_PLAN_STATES)}")

    criteria = _bounded_object_list(
        plan.get("success_criteria"),
        "payload.plan.success_criteria",
        minimum=1,
        maximum=MAX_TASK_CRITERIA,
    )
    criterion_ids: set[str] = set()
    for index, criterion in enumerate(criteria):
        path = f"payload.plan.success_criteria[{index}]"
        criterion_id = _required_identifier(criterion, "id", path)
        if criterion_id in criterion_ids:
            raise ValueError(f"duplicate success criterion id:{criterion_id}")
        criterion_ids.add(criterion_id)
        if not str(criterion.get("content") or criterion.get("description") or "").strip():
            raise ValueError(f"{path}.content or .description is required")

    steps = _bounded_object_list(
        plan.get("steps"),
        "payload.plan.steps",
        minimum=1,
        maximum=MAX_TASK_STEPS,
    )
    step_ids: set[str] = set()
    in_progress_ids: list[str] = []
    for index, step in enumerate(steps):
        path = f"payload.plan.steps[{index}]"
        step_id = _required_identifier(step, "id", path)
        _required_text(step, "content", path, max_chars=4000)
        if step_id in step_ids:
            raise ValueError(f"duplicate task step id:{step_id}")
        step_ids.add(step_id)
        status = str(step.get("status") or "").strip()
        if status not in TASK_STEP_STATUSES:
            raise ValueError(f"{path}.status must be one of {sorted(TASK_STEP_STATUSES)}")
        if status == "in_progress":
            in_progress_ids.append(step_id)
        depends_on = step.get("depends_on", [])
        if not isinstance(depends_on, list) or not all(str(item or "").strip() for item in depends_on):
            raise ValueError(f"{path}.depends_on must be an array of non-empty step ids")
        if step_id in {str(item) for item in depends_on}:
            raise ValueError(f"{path}.depends_on cannot reference itself")
    if len(in_progress_ids) > 1:
        raise ValueError("payload.plan.steps allows at most one in_progress step")
    for index, step in enumerate(steps):
        unknown = {
            str(item)
            for item in step.get("depends_on", [])
            if str(item) not in step_ids
        }
        if unknown:
            raise ValueError(
                f"payload.plan.steps[{index}].depends_on contains unknown ids:{sorted(unknown)}"
            )

    # Dependency structure is mechanical execution state, not semantic
    # planning.  Reject cycles and impossible in-progress cursors so a restart
    # can never hydrate a plan that cannot advance.
    step_by_id = {str(step["id"]): step for step in steps}
    visiting: set[str] = set()
    visited: set[str] = set()

    def _visit(step_id: str) -> None:
        if step_id in visiting:
            raise ValueError("payload.plan.steps.depends_on contains a cycle")
        if step_id in visited:
            return
        visiting.add(step_id)
        for dependency_id in step_by_id[step_id].get("depends_on", []):
            _visit(str(dependency_id))
        visiting.remove(step_id)
        visited.add(step_id)

    for step_id in step_by_id:
        _visit(step_id)
    for step in steps:
        if step.get("status") != "in_progress":
            continue
        unfinished_dependencies = [
            str(dependency_id)
            for dependency_id in step.get("depends_on", [])
            if step_by_id[str(dependency_id)].get("status")
            not in {"completed", "cancelled"}
        ]
        if unfinished_dependencies:
            raise ValueError(
                "in_progress task step has non-terminal dependencies:"
                + ",".join(unfinished_dependencies)
            )

    current_step_id = str(plan.get("current_step_id") or "").strip()
    if current_step_id and current_step_id not in step_ids:
        raise ValueError("payload.plan.current_step_id must reference a plan step")
    if in_progress_ids and current_step_id != in_progress_ids[0]:
        raise ValueError("payload.plan.current_step_id must match the in_progress step")

    resume_cursor = plan.get("resume_cursor")
    if not isinstance(resume_cursor, dict):
        raise ValueError("payload.plan.resume_cursor is required")
    _required_text(resume_cursor, "summary", "payload.plan.resume_cursor", max_chars=2000)
    next_step_id = str(resume_cursor.get("next_step_id") or "").strip()
    if next_step_id and next_step_id not in step_ids:
        raise ValueError("payload.plan.resume_cursor.next_step_id must reference a plan step")

    if state == "active":
        if not current_step_id:
            raise ValueError("active plan requires payload.plan.current_step_id")
        for path, step_id in (
            ("payload.plan.current_step_id", current_step_id),
            ("payload.plan.resume_cursor.next_step_id", next_step_id),
        ):
            if not step_id:
                raise ValueError(f"active plan requires {path}")
            if step_by_id[step_id].get("status") not in {"pending", "in_progress"}:
                raise ValueError(f"{path} must reference a pending or in_progress step")
        if current_step_id != next_step_id:
            raise ValueError(
                "active plan current_step_id and resume_cursor.next_step_id must match"
            )
        unfinished_cursor_dependencies = [
            str(dependency_id)
            for dependency_id in step_by_id[current_step_id].get("depends_on", [])
            if step_by_id[str(dependency_id)].get("status")
            not in {"completed", "cancelled"}
        ]
        if unfinished_cursor_dependencies:
            raise ValueError(
                "active task cursor has non-terminal dependencies:"
                + ",".join(unfinished_cursor_dependencies)
            )

    for key in ("attempts", "decisions", "artifacts"):
        if key in plan:
            _bounded_object_list(plan.get(key), f"payload.plan.{key}")

    if state == "completed":
        if current_step_id or next_step_id:
            raise ValueError(
                "completed plan must clear current_step_id and resume_cursor.next_step_id"
            )
        non_terminal = [
            str(step.get("id"))
            for step in steps
            if step.get("status") not in {"completed", "cancelled"}
        ]
        if non_terminal:
            raise ValueError(f"completed plan has non-terminal steps:{non_terminal}")
        verification_ids = _bounded_string_list(
            plan.get("verification_event_ids"),
            "payload.plan.verification_event_ids",
            minimum=1,
        )
        for event_id in verification_ids:
            try:
                uuid.UUID(event_id)
            except (ValueError, TypeError, AttributeError):
                raise ValueError("payload.plan.verification_event_ids must contain UUID event ids") from None
    elif state == "blocked":
        blocker = plan.get("blocker")
        if not isinstance(blocker, dict):
            raise ValueError("blocked plan requires payload.plan.blocker")
        _required_text(blocker, "reason", "payload.plan.blocker", max_chars=2000)
        _bounded_object_list(
            blocker.get("attempts"),
            "payload.plan.blocker.attempts",
            minimum=1,
        )
        _required_text(
            blocker,
            "required_input_or_authority",
            "payload.plan.blocker",
            max_chars=2000,
        )
        _required_text(blocker, "resume_when", "payload.plan.blocker", max_chars=2000)


def _validate_task_verification_payload(payload: Dict[str, Any]) -> None:
    verification = payload.get("verification")
    if not isinstance(verification, dict):
        raise ValueError("task.verification.recorded requires payload.verification")
    _required_identifier(verification, "verification_id", "payload.verification")
    _required_identifier(verification, "plan_id", "payload.verification")
    plan_revision = verification.get("plan_revision")
    if (
        not isinstance(plan_revision, int)
        or isinstance(plan_revision, bool)
        or plan_revision < 1
    ):
        raise ValueError("payload.verification.plan_revision must be a positive integer")
    _required_text(verification, "summary", "payload.verification", max_chars=4000)
    outcome = str(verification.get("outcome") or "").strip()
    if outcome not in TASK_VERIFICATION_OUTCOMES:
        raise ValueError(
            f"payload.verification.outcome must be one of {sorted(TASK_VERIFICATION_OUTCOMES)}"
        )
    criterion_ids = _bounded_string_list(
        verification.get("criterion_ids"),
        "payload.verification.criterion_ids",
        minimum=1,
        maximum=MAX_TASK_CRITERIA,
    )
    if len(set(criterion_ids)) != len(criterion_ids):
        raise ValueError("payload.verification.criterion_ids must be unique")
    receipt = verification.get("receipt")
    if not isinstance(receipt, dict):
        raise ValueError("payload.verification.receipt is required")
    _required_text(receipt, "kind", "payload.verification.receipt", max_chars=160)
    if not any(
        str(receipt.get(key) or "").strip()
        for key in ("ref", "sha256", "message_id", "commit_sha", "deployment_sha")
    ):
        raise ValueError(
            "payload.verification.receipt requires ref, sha256, message_id, commit_sha, or deployment_sha"
        )
    reserved = {"event_id", "occurred_at", "runtime_attested"}
    if reserved.intersection(verification):
        raise ValueError(
            "payload.verification contains reserved runtime projection fields"
        )


def _validate_runtime_receipt(event_type: str, payload: Dict[str, Any]) -> None:
    key = "approval_receipt" if event_type == "approval.capability.recorded" else "capability_receipt"
    receipt = payload.get(key)
    if not isinstance(receipt, dict):
        raise ValueError(f"{event_type} requires payload.{key}")
    if {"event_id", "occurred_at", "runtime_attested"}.intersection(receipt):
        raise ValueError(f"payload.{key} contains reserved runtime projection fields")
    _required_text(receipt, "approval_id", f"payload.{key}", max_chars=160)
    _required_text(receipt, "plan_id", f"payload.{key}", max_chars=160)
    session_hash = _required_text(receipt, "session_key_sha256", f"payload.{key}", max_chars=64)
    if not _SHA256_RE.fullmatch(session_hash):
        raise ValueError(f"payload.{key}.session_key_sha256 must be a sha256 digest")
    if event_type == "approval.capability.recorded":
        approval_source_hash = _required_text(
            receipt,
            "approval_source_sha256",
            f"payload.{key}",
            max_chars=64,
        )
        if not _SHA256_RE.fullmatch(approval_source_hash):
            raise ValueError(f"payload.{key}.approval_source_sha256 must be a sha256 digest")
        hashes = _bounded_string_list(
            receipt.get("command_hashes"),
            f"payload.{key}.command_hashes",
            minimum=1,
        )
        if any(not _SHA256_RE.fullmatch(item) for item in hashes):
            raise ValueError(f"payload.{key}.command_hashes must contain sha256 digests")
        if "exact_commands" in receipt or "commands" in receipt:
            raise ValueError(f"payload.{key} must not contain raw commands")
        if str(receipt.get("state") or "") != "granted":
            raise ValueError(f"payload.{key}.state must be granted")
    else:
        command_hash = _required_text(receipt, "command_sha256", f"payload.{key}", max_chars=64)
        if not _SHA256_RE.fullmatch(command_hash):
            raise ValueError(f"payload.{key}.command_sha256 must be a sha256 digest")
        if str(receipt.get("state") or "") != "authorized":
            raise ValueError(f"payload.{key}.state must be authorized")


def _validate_append_request(
    *,
    event_type: str,
    case_id: str,
    summary: str,
    source_refs: Dict[str, Any],
    actors: Dict[str, Any],
    payload: Dict[str, Any],
    safety: Dict[str, Any],
) -> None:
    if event_type not in ALLOWED_EVENT_TYPES:
        raise ValueError(f"event_type_not_allowed:{event_type}")
    if not _CASE_ID_RE.fullmatch(str(case_id or "")):
        raise ValueError(
            "case_id must be a bounded safe identifier starting with case:"
        )
    if not source_refs.get("platform"):
        raise ValueError("source_refs.platform is required")
    if not (source_refs.get("message_id") or source_refs.get("event_ref") or source_refs.get("manual_ref")):
        raise ValueError("source_refs requires message_id, event_ref, or manual_ref")
    if bool(safety.get("contains_secret")) or bool(safety.get("contains_payment_credential")):
        raise ValueError("safety flags block append")
    _block_secret_like_fields(
        summary=summary,
        source_refs=source_refs,
        actors=actors,
        payload=payload,
        safety=safety,
    )
    if event_type in RECEIPT_REQUIRED_EVENT_TYPES:
        if event_type == "route_back.sent":
            receipt = payload.get("receipt") if isinstance(payload, dict) else None
            if not isinstance(receipt, dict):
                raise ValueError("route_back.sent requires payload.receipt")
            for key in ("message_id", "channel_id", "content_sha256"):
                if not str(receipt.get(key) or "").strip():
                    raise ValueError(f"route_back.sent requires payload.receipt.{key}")
            if not _SHA256_RE.fullmatch(str(receipt.get("content_sha256") or "")):
                raise ValueError("route_back.sent receipt.content_sha256 must be sha256")
            _validate_no_route_back_dm_refs(payload)
            route_back = payload.get("route_back")
            route_back = route_back if isinstance(route_back, dict) else {}
            target_ref = route_back.get("target_ref")
            target_ref = target_ref if isinstance(target_ref, dict) else {}
            target_channel_id = str(
                target_ref.get("channel_id") or target_ref.get("thread_id") or ""
            ).strip()
            receipt_channel_id = str(receipt.get("channel_id") or "").strip()
            if not target_channel_id or target_channel_id != receipt_channel_id:
                raise ValueError(
                    "route_back.sent target channel must exactly match receipt.channel_id"
                )
            channel_surfaces = (
                ("route_back.target_ref", target_ref),
                ("route_back.receipt", route_back.get("receipt")),
                ("receipt", receipt),
            )
            for surface_name, surface in channel_surfaces:
                if not isinstance(surface, dict):
                    continue
                for key in ("channel_id", "thread_id", "chat_id"):
                    value = str(surface.get(key) or "").strip()
                    if value and value != receipt_channel_id:
                        raise ValueError(
                            f"route_back.sent {surface_name}.{key} conflicts with receipt channel"
                        )
            execution_binding = route_back.get("execution_binding")
            if not isinstance(execution_binding, dict):
                raise ValueError("route_back.sent requires route_back.execution_binding")
            if (
                str(execution_binding.get("target_channel_id") or "").strip()
                != receipt_channel_id
            ):
                raise ValueError(
                    "route_back.sent execution binding target does not match receipt channel"
                )
            if (
                str(execution_binding.get("content_sha256") or "").strip()
                != str(receipt.get("content_sha256") or "").strip()
            ):
                raise ValueError(
                    "route_back.sent execution binding content does not match receipt"
                )
            verified = _discord_verify_message_receipt(
                channel_id=str(receipt["channel_id"]),
                message_id=str(receipt["message_id"]),
                expected_content_sha256=str(receipt["content_sha256"]),
            )
            if not isinstance(verified, dict) or verified.get("verified") is not True:
                raise ValueError("route_back.sent Discord adapter readback was not verified")
            expected_verified_fields = {
                "channel_id": str(receipt["channel_id"]),
                "message_id": str(receipt["message_id"]),
                "content_sha256": str(receipt["content_sha256"]),
            }
            for key, expected in expected_verified_fields.items():
                if str(verified.get(key) or "") != expected:
                    raise ValueError(
                        f"route_back.sent Discord adapter readback {key} mismatch"
                    )
        else:
            _validate_runtime_receipt(event_type, payload)
    if event_type == "person.alias.learned":
        if not str(payload.get("alias") or "").strip() or not str(payload.get("member_key") or "").strip():
            raise ValueError("person.alias.learned requires payload.alias and payload.member_key")
    elif event_type == "task.plan.updated":
        _validate_task_plan_payload(payload)
    elif event_type == "task.verification.recorded":
        _validate_task_verification_payload(payload)


def _contains_forbidden_dm_route_ref(value: Any) -> bool:
    """Return True when route-back target/receipt metadata points at a DM.

    Muncho's SkyVision policy allows public approved channels/threads only for
    team route-backs. A Discord user mention can still be used inside a public
    channel message, but DM channel ids or recipient-id delivery metadata must
    not be recorded as valid route_back.sent evidence.
    """
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized_key = _normalize_secret_key(key)
            if normalized_key in FORBIDDEN_ROUTE_BACK_DM_KEYS and _has_structured_secret_value(nested):
                return True
            if normalized_key in {"channel_type", "target_type", "delivery_type", "lane", "role"}:
                normalized_value = str(nested or "").strip().casefold()
                if normalized_value in FORBIDDEN_ROUTE_BACK_DM_VALUES or normalized_value.endswith("_dm"):
                    return True
            if _contains_forbidden_dm_route_ref(nested):
                return True
        return False
    if isinstance(value, (list, tuple, set)):
        return any(_contains_forbidden_dm_route_ref(item) for item in value)
    return False


def _validate_no_route_back_dm_refs(payload: Dict[str, Any]) -> None:
    route_back = payload.get("route_back") if isinstance(payload, dict) else None
    surfaces = [
        payload.get("target_ref"),
        payload.get("receipt"),
        route_back.get("target_ref") if isinstance(route_back, dict) else None,
        route_back.get("receipt") if isinstance(route_back, dict) else None,
    ]
    if any(_contains_forbidden_dm_route_ref(surface) for surface in surfaces if surface):
        raise ValueError("route_back.sent forbids direct-message/DM delivery receipts; use public approved channel/thread target or record_blocked")


def _observed_session_refs() -> Dict[str, str]:
    """Return immutable runtime-observed identity separately from model refs."""
    observed: Dict[str, str] = {}
    for key, env_name in {
        "platform": "HERMES_SESSION_PLATFORM",
        "chat_id": "HERMES_SESSION_CHAT_ID",
        "thread_id": "HERMES_SESSION_THREAD_ID",
        "message_id": "HERMES_SESSION_MESSAGE_ID",
        "session_id": "HERMES_SESSION_ID",
        "session_key_sha256": "HERMES_SESSION_KEY",
        "user_id": "HERMES_SESSION_USER_ID",
    }.items():
        value = _get_session_env(env_name, "").strip()
        if value:
            observed[key] = _hash(value) if key == "session_key_sha256" else value
    return observed


def _event_evidence(
    *,
    event_type: str,
    payload: Dict[str, Any],
    source_refs: Dict[str, Any],
) -> list[Dict[str, Any]]:
    """Separate model assertions from deterministic runtime attestation."""
    if event_type == "route_back.sent":
        receipt_surface = (
            payload.get("receipt")
            or {}
        )
        return [{
            "label": f"{event_type}.runtime_receipt",
            "verified": True,
            "attestation": "deterministic_runtime_receipt",
            "receipt_sha256": _hash(receipt_surface),
        }]
    if event_type in PROCESS_RECEIPT_EVENT_TYPES:
        receipt_surface = (
            payload.get("approval_receipt")
            or payload.get("capability_receipt")
            or {}
        )
        return [{
            "label": f"{event_type}.process_receipt",
            "verified": False,
            "attestation": "runtime_process_receipt_unverified",
            "receipt_sha256": _hash(receipt_surface),
        }]

    supplied = payload.get("evidence")
    if isinstance(supplied, list) and supplied:
        normalized = []
        for item in supplied[:MAX_TASK_COLLECTION_ITEMS]:
            if isinstance(item, dict):
                normalized.append({
                    **item,
                    "verified": False,
                    "attestation": "model_authored",
                })
        if normalized:
            return normalized
    return [{
        "label": "hermes_semantic_decision",
        "verified": False,
        "attestation": "model_authored",
        "source_refs_hash": _hash(source_refs)[:16],
    }]


def _readback_matches(
    rows: Any,
    *,
    event_id: str,
    event_type: str,
    case_id: str,
    idempotency_key: str,
    content_sha256: str,
) -> bool:
    if not isinstance(rows, list) or len(rows) != 1:
        return False
    row = rows[0]
    if isinstance(row, dict):
        actual = (
            row.get("event_id"),
            row.get("event_type"),
            row.get("case_id"),
            row.get("idempotency_key") or row.get("?column?"),
            row.get("canonical_content_sha256"),
        )
    elif isinstance(row, (list, tuple)) and len(row) >= 6:
        actual = (row[0], row[1], row[2], row[4], row[5])
    else:
        return False
    return tuple(str(value or "") for value in actual) == (
        event_id,
        event_type,
        case_id,
        idempotency_key,
        content_sha256,
    )


def _decode_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (TypeError, ValueError):
            return {}
        return dict(decoded) if isinstance(decoded, dict) else {}
    return {}


def _task_plan_graph_query_sql(where: str, *, limit: int) -> str:
    """Return latest revision + each distinct supersession edge per plan."""
    return f"""
WITH plan_events AS (
  SELECT e.event_id, e.occurred_at,
         e.payload->'plan' AS plan,
         e.payload->'plan'->>'plan_id' AS plan_id,
         e.payload->'plan'->>'revision' AS revision_text,
         COALESCE(e.payload->'plan'->>'supersedes_plan_id', '') AS supersedes_plan_id,
         COALESCE(e.payload->'plan'->>'supersedes_plan_revision', '') AS supersedes_revision_text
  FROM {EVENT_TABLE} AS e
  WHERE ({where}) AND e.event_type = 'task.plan.updated'
), ranked AS (
  SELECT p.*,
         ROW_NUMBER() OVER (
           PARTITION BY p.plan_id
           ORDER BY
             CASE WHEN p.revision_text ~ '^[1-9][0-9]*$'
                  THEN p.revision_text::numeric ELSE -1 END DESC,
             p.occurred_at DESC, p.event_id DESC
         ) AS latest_revision_rank,
         ROW_NUMBER() OVER (
           PARTITION BY p.plan_id, p.supersedes_plan_id, p.supersedes_revision_text
           ORDER BY
             CASE WHEN p.revision_text ~ '^[1-9][0-9]*$'
                  THEN p.revision_text::numeric ELSE -1 END,
             p.occurred_at, p.event_id
         ) AS edge_rank
  FROM plan_events AS p
)
SELECT event_id::text AS event_id, plan_id,
       revision_text AS revision, plan, occurred_at::text AS occurred_at
FROM ranked
WHERE latest_revision_rank = 1
   OR (supersedes_plan_id <> '' AND edge_rank = 1)
ORDER BY occurred_at DESC, event_id DESC
LIMIT {limit};
"""


def _normalize_task_plan_graph_rows(
    rows: Any,
    *,
    case_id: str,
) -> list[Dict[str, Any]]:
    normalized: list[Dict[str, Any]] = []
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, dict):
            payload = _decode_mapping(row.get("payload"))
            plan = _decode_mapping(row.get("plan")) or _decode_mapping(payload.get("plan"))
            event_id = str(row.get("event_id") or "")
            occurred_at = str(row.get("occurred_at") or "")
            projection_row = dict(row)
        elif isinstance(row, (list, tuple)):
            event_id = str(row[0] if len(row) > 0 else "")
            plan = _decode_mapping(row[3] if len(row) > 3 else None)
            occurred_at = str(row[4] if len(row) > 4 else "")
            projection_row = {}
        else:
            continue
        normalized.append({
            **projection_row,
            "event_id": event_id,
            "event_type": "task.plan.updated",
            "case_id": case_id,
            "occurred_at": occurred_at,
            "payload": {"plan": plan},
        })
    return normalized


def _load_task_plan_graph_rows(
    helper: Any,
    sock: Any,
    *,
    where: str,
    case_id: str,
) -> tuple[list[Dict[str, Any]], bool]:
    result = helper.query(
        sock,
        _task_plan_graph_query_sql(where, limit=MAX_TASK_PLAN_GRAPH_ROWS + 1),
    )
    raw_rows = result.get("rows", []) if isinstance(result, dict) else []
    rows = _normalize_task_plan_graph_rows(raw_rows, case_id=case_id)
    return rows[:MAX_TASK_PLAN_GRAPH_ROWS], len(rows) > MAX_TASK_PLAN_GRAPH_ROWS


def _latest_task_plan_record(helper: Any, sock: Any, case_id: str) -> Dict[str, Any]:
    rows, truncated = _load_task_plan_graph_rows(
        helper,
        sock,
        where=f"e.case_id = {helper.sql_quote(case_id)}",
        case_id=case_id,
    )
    if truncated:
        raise ValueError("task plan graph exceeds the bounded validation window")
    from gateway.canonical_brain_projection import select_canonical_plan_head

    head, graph_error = select_canonical_plan_head(rows)
    if graph_error:
        raise ValueError(f"canonical task plan graph is invalid:{graph_error}")
    if head is None:
        return {}
    plan = _decode_mapping(_decode_mapping(head.get("payload")).get("plan"))
    plan_id = str(plan.get("plan_id") or "")
    lineage_edges = {
        (
            str(candidate.get("supersedes_plan_id") or "").strip(),
            candidate.get("supersedes_plan_revision"),
        )
        for row in rows
        for candidate in [
            _decode_mapping(_decode_mapping(row.get("payload")).get("plan"))
        ]
        if str(candidate.get("plan_id") or "") == plan_id
        and str(candidate.get("supersedes_plan_id") or "").strip()
    }
    if len(lineage_edges) == 1:
        predecessor_id, predecessor_revision = next(iter(lineage_edges))
        plan["supersedes_plan_id"] = predecessor_id
        plan["supersedes_plan_revision"] = predecessor_revision
    return {
        "event_id": str(head.get("event_id") or ""),
        "plan_id": plan_id,
        "revision": plan.get("revision"),
        "plan": plan,
    }


def _validate_task_verification_against_plan(
    helper: Any,
    sock: Any,
    *,
    case_id: str,
    payload: Dict[str, Any],
) -> None:
    verification = payload.get("verification") if isinstance(payload, dict) else None
    if not isinstance(verification, dict):
        return
    latest = _latest_task_plan_record(helper, sock, case_id)
    plan = latest.get("plan") if isinstance(latest.get("plan"), dict) else {}
    if not latest or not plan:
        raise ValueError("task verification requires an existing active task plan")
    if str(plan.get("state") or "") != "active":
        raise ValueError("task verification requires the latest task plan to be active")
    if str(verification.get("plan_id") or "") != str(latest.get("plan_id") or ""):
        raise ValueError("task verification plan_id does not match the latest task plan")
    try:
        latest_revision = int(latest.get("revision"))
    except (TypeError, ValueError):
        raise ValueError("latest task plan has an invalid revision") from None
    if int(verification.get("plan_revision")) != latest_revision:
        raise ValueError("task verification plan_revision is stale or does not match the latest task plan")
    known_criteria = {
        str(item.get("id"))
        for item in plan.get("success_criteria") or []
        if isinstance(item, dict) and item.get("id")
    }
    unknown = sorted(
        set(str(value) for value in verification.get("criterion_ids") or [])
        - known_criteria
    )
    if unknown:
        raise ValueError(
            "task verification references unknown success criteria:" + ",".join(unknown)
        )


def _validate_completed_plan_receipts(
    helper: Any,
    sock: Any,
    *,
    case_id: str,
    payload: Dict[str, Any],
) -> None:
    plan = payload.get("plan") if isinstance(payload, dict) else None
    if not isinstance(plan, dict) or plan.get("state") != "completed":
        return
    event_ids = [str(value) for value in plan.get("verification_event_ids") or []]
    plan_id = str(plan.get("plan_id") or "")
    expected_plan_revision = int(plan.get("revision")) - 1
    if expected_plan_revision < 1:
        raise ValueError(
            "completed plan requires a prior active revision and revision-bound verification receipts"
        )
    quoted = ", ".join(f"{helper.sql_quote(value)}::uuid" for value in event_ids)
    rows = helper.query(sock, f"""
SELECT event_id::text, payload->'verification'->'criterion_ids',
       payload->'verification'->>'plan_revision'
FROM {EVENT_TABLE}
WHERE case_id = {helper.sql_quote(case_id)}
  AND event_type = 'task.verification.recorded'
  AND payload->'verification'->>'plan_id' = {helper.sql_quote(plan_id)}
  AND payload->'verification'->>'plan_revision' = {helper.sql_quote(str(expected_plan_revision))}
  AND payload->'verification'->>'outcome' = 'passed'
  AND event_id IN ({quoted});
""").get("rows", [])
    found: set[str] = set()
    covered_criteria: set[str] = set()
    for row in rows:
        if not row:
            continue
        if isinstance(row, dict):
            receipt_event_id = str(row.get("event_id") or "")
            raw_criteria = row.get("criterion_ids")
            receipt_revision = row.get("plan_revision")
        else:
            receipt_event_id = str(row[0])
            raw_criteria = row[1] if len(row) > 1 else []
            receipt_revision = row[2] if len(row) > 2 else None
        try:
            if int(receipt_revision) != expected_plan_revision:
                continue
        except (TypeError, ValueError):
            continue
        found.add(receipt_event_id)
        if isinstance(raw_criteria, str):
            try:
                raw_criteria = json.loads(raw_criteria)
            except (TypeError, ValueError):
                raw_criteria = []
        if isinstance(raw_criteria, list):
            covered_criteria.update(str(value) for value in raw_criteria if str(value).strip())
    missing = sorted(set(event_ids) - found)
    if missing:
        raise ValueError(
            "completed plan references missing or non-passing verification events:"
            + ",".join(missing)
        )
    required_criteria = {
        str(item.get("id"))
        for item in plan.get("success_criteria") or []
        if isinstance(item, dict) and item.get("id")
    }
    uncovered = sorted(required_criteria - covered_criteria)
    if uncovered:
        raise ValueError(
            "completed plan has success criteria without passing verification receipts:"
            + ",".join(uncovered)
        )


def _validate_task_plan_revision(
    helper: Any,
    sock: Any,
    *,
    case_id: str,
    event_id: str,
    payload: Dict[str, Any],
) -> str | None:
    plan = payload.get("plan") if isinstance(payload, dict) else None
    if not isinstance(plan, dict):
        return None
    latest = _latest_task_plan_record(helper, sock, case_id)
    if not latest:
        if int(plan.get("revision")) != 1:
            raise ValueError("the first task plan event must start at revision 1")
        return None
    previous_event_id = str(latest.get("event_id") or "")
    previous_plan_id = str(latest.get("plan_id") or "")
    previous_revision_raw = latest.get("revision")
    try:
        previous_revision = int(previous_revision_raw)
    except (TypeError, ValueError):
        raise ValueError("latest task plan has an invalid revision") from None
    plan_id = str(plan.get("plan_id") or "")
    revision = int(plan.get("revision"))
    if previous_plan_id == plan_id:
        if revision < previous_revision:
            raise ValueError(
                f"task plan revision regressed:{revision}<{previous_revision}"
            )
        if revision == previous_revision and previous_event_id != event_id:
            raise ValueError(
                "task plan revision must increase for a new event"
            )
        if revision > previous_revision + 1:
            raise ValueError(
                f"task plan revision must advance exactly one step:{previous_revision}->{revision}"
            )
        previous_plan = (
            latest.get("plan") if isinstance(latest.get("plan"), dict) else {}
        )
        if (
            revision > previous_revision
            and str(previous_plan.get("state") or "") in {"completed", "cancelled"}
        ):
            raise ValueError(
                "completed or cancelled task plans cannot advance under the same plan_id; "
                "use explicit plan supersession for new work"
            )
        previous_supersession = (
            str(previous_plan.get("supersedes_plan_id") or "").strip(),
            previous_plan.get("supersedes_plan_revision"),
        )
        current_supersession = (
            str(plan.get("supersedes_plan_id") or "").strip(),
            plan.get("supersedes_plan_revision"),
        )
        if current_supersession != previous_supersession:
            raise ValueError(
                "task plan supersession metadata is immutable within one plan_id"
            )
        # A revision may refine an active plan monotonically as GPT discovers
        # more work, but it must not silently erase or rewrite obligations that
        # were already approved. Removals, rewording, and dependency rewrites
        # require a new plan_id with explicit supersession.
        previous_criteria = {
            str(item.get("id")): item
            for item in previous_plan.get("success_criteria") or []
            if isinstance(item, dict) and item.get("id")
        }
        current_criteria = {
            str(item.get("id")): item
            for item in plan.get("success_criteria") or []
            if isinstance(item, dict) and item.get("id")
        }
        removed_criteria = sorted(previous_criteria.keys() - current_criteria.keys())
        changed_criteria = sorted(
            criterion_id
            for criterion_id in previous_criteria.keys() & current_criteria.keys()
            if str(
                previous_criteria[criterion_id].get("content")
                or previous_criteria[criterion_id].get("description")
                or ""
            ).strip()
            != str(
                current_criteria[criterion_id].get("content")
                or current_criteria[criterion_id].get("description")
                or ""
            ).strip()
        )
        if removed_criteria or changed_criteria:
            raise ValueError(
                "same plan_id revision must preserve prior success criteria; "
                f"removed={removed_criteria}, changed={changed_criteria}; "
                "use explicit plan supersession to remove or rewrite scope"
            )

        previous_steps = {
            str(item.get("id")): item
            for item in previous_plan.get("steps") or []
            if isinstance(item, dict) and item.get("id")
        }
        current_steps = {
            str(item.get("id")): item
            for item in plan.get("steps") or []
            if isinstance(item, dict) and item.get("id")
        }
        removed_steps = sorted(previous_steps.keys() - current_steps.keys())
        changed_steps = sorted(
            step_id
            for step_id in previous_steps.keys() & current_steps.keys()
            if (
                str(previous_steps[step_id].get("content") or "").strip()
                != str(current_steps[step_id].get("content") or "").strip()
                or {
                    str(value)
                    for value in previous_steps[step_id].get("depends_on", [])
                }
                != {
                    str(value)
                    for value in current_steps[step_id].get("depends_on", [])
                }
            )
        )
        if removed_steps or changed_steps:
            raise ValueError(
                "same plan_id revision must preserve prior steps and dependencies; "
                f"removed={removed_steps}, changed={changed_steps}; "
                "use explicit plan supersession to remove or rewrite scope"
            )
        return None
    if (
        revision != 1
        or str(plan.get("supersedes_plan_id") or "") != previous_plan_id
        or int(plan.get("supersedes_plan_revision") or 0) != previous_revision
    ):
        raise ValueError(
            "a new plan_id must start at revision 1 and explicitly supersede "
            "the exact latest plan_id/revision"
        )
    return previous_plan_id


def _validate_route_back_sent_against_intent(
    helper: Any,
    sock: Any,
    *,
    case_id: str,
    idempotency_key: str,
    payload: Dict[str, Any],
) -> None:
    """Require the sent receipt to match the durable claimed execution intent."""
    route_back = payload.get("route_back") if isinstance(payload, dict) else None
    route_back = route_back if isinstance(route_back, dict) else {}
    sent_binding = route_back.get("execution_binding")
    sent_binding = sent_binding if isinstance(sent_binding, dict) else {}
    intent_event_id = _event_uuid(
        idempotency_key,
        "route_back.intent.created",
        case_id,
    )
    rows = helper.query(sock, f"""
SELECT payload AS intent_payload
FROM {EVENT_TABLE}
WHERE case_id = {helper.sql_quote(case_id)}
  AND event_id = {helper.sql_quote(intent_event_id)}::uuid
  AND event_type = 'route_back.intent.created'
LIMIT 1;
""").get("rows", [])
    if not rows:
        raise ValueError("route_back.sent requires a matching durable execution intent")
    row = rows[0]
    raw_payload = row.get("intent_payload") if isinstance(row, dict) else row[0]
    intent_payload = _decode_mapping(raw_payload)
    intent_route_back = intent_payload.get("route_back")
    intent_route_back = intent_route_back if isinstance(intent_route_back, dict) else {}
    intent_binding = intent_route_back.get("execution_binding")
    intent_binding = intent_binding if isinstance(intent_binding, dict) else {}
    if not intent_binding or _stable_json(intent_binding) != _stable_json(sent_binding):
        raise ValueError(
            "route_back.sent receipt does not match the durable execution intent binding"
        )


def canonical_event_append_tool(
    event_type: str,
    case_id: str,
    summary: str,
    source_refs: Dict[str, Any],
    actors: Optional[Dict[str, Any]] = None,
    payload: Optional[Dict[str, Any]] = None,
    safety: Optional[Dict[str, Any]] = None,
    idempotency_key: Optional[str] = None,
) -> str:
    """Append one canonical operational event to Cloud SQL.

    The caller (Hermes) decides meaning. This function validates mechanics and
    writes a deterministic/idempotent event row.
    """
    try:
        source_refs = _normalize_dict(source_refs, "source_refs")
        source_refs = _augment_source_refs_from_session_context(source_refs)
        actors = _normalize_dict(actors, "actors")
        payload = _normalize_dict(payload, "payload")
        safety = _normalize_dict(safety, "safety")
        if {"idempotency_key", "canonical_content_sha256"}.intersection(payload):
            raise ValueError("payload contains reserved canonical append fields")
        _validate_append_request(
            event_type=event_type,
            case_id=case_id,
            summary=summary,
            source_refs=source_refs,
            actors=actors,
            payload=payload,
            safety=safety,
        )
        if not idempotency_key:
            idempotency_key = f"{case_id}:{event_type}:{_hash({'source_refs': source_refs, 'payload': payload})[:24]}"
        idempotency_key = str(idempotency_key)
        if len(idempotency_key) > 512 or any(ord(char) < 32 for char in idempotency_key):
            raise ValueError("idempotency_key must be <=512 characters without control characters")
        canonical_content_sha256 = _hash({
            "event_type": event_type,
            "case_id": case_id,
            "summary": str(summary or ""),
            "source_refs": source_refs,
            "actors": actors,
            "payload": payload,
            "safety": safety,
        })
        event_identity_key = idempotency_key
        if event_type == "task.plan.updated":
            plan_identity = payload["plan"]
            # One canonical row per exact plan revision.  This closes the
            # concurrent-writer race without interpreting plan content: two
            # writers for the same explicit revision converge on one event id,
            # and the content-hash readback detects disagreement.
            if int(plan_identity["revision"]) == 1:
                predecessor = str(plan_identity.get("supersedes_plan_id") or "")
                event_identity_key = (
                    "task-plan-transition:"
                    + (
                        f"{predecessor}:revision:"
                        f"{plan_identity.get('supersedes_plan_revision')}"
                        if predecessor
                        else "initial"
                    )
                )
            else:
                event_identity_key = (
                    f"task-plan-transition:{plan_identity['plan_id']}:"
                    f"revision:{int(plan_identity['revision']) - 1}"
                )
        event_id = _event_uuid(event_identity_key, event_type, case_id)
        occurred_at = _utc_now()
        source = {
            "system": "hermes_agent",
            "component": "canonical_brain_tool",
            "source_refs": source_refs,
            "observed_session": _observed_session_refs(),
        }
        actor = actors.get("actor") or {"type": "agent", "id": "hermes"}
        subject = actors.get("subject") or {"type": "case", "id": case_id}
        evidence = _event_evidence(
            event_type=event_type,
            payload=payload,
            source_refs=source_refs,
        )
        model_decision = payload.get("decision") if isinstance(payload.get("decision"), dict) else {}
        runtime_attested = event_type == "route_back.sent"
        process_receipt = event_type in PROCESS_RECEIPT_EVENT_TYPES
        decision = {
            **model_decision,
            "kind": "hermes_semantic_operational_persistence",
            "decided_by": (
                "deterministic_runtime_receipt"
                if runtime_attested
                else (
                    "runtime_process_receipt_unverified"
                    if process_receipt
                    else "hermes_agent_llm_reasoning"
                )
            ),
            "keyword_authority": False,
            "attestation": (
                "deterministic_runtime_receipt"
                if runtime_attested
                else (
                    "runtime_process_receipt_unverified"
                    if process_receipt
                    else "model_authored"
                )
            ),
        }
        explicit_state = event_type
        if event_type == "task.plan.updated":
            explicit_state = str(payload["plan"]["state"])
        elif event_type == "task.verification.recorded":
            explicit_state = str(payload["verification"]["outcome"])
        elif event_type == "approval.capability.recorded":
            explicit_state = str(payload["approval_receipt"]["state"])
        elif event_type == "capability.check.recorded":
            explicit_state = str(payload["capability_receipt"]["state"])
        status = {
            "state": explicit_state,
            "event_type": event_type,
            "summary": str(summary or "")[:500],
        }
        next_action = payload.get("next_action") if isinstance(payload.get("next_action"), dict) else {}
        if event_type == "task.plan.updated" and not next_action:
            cursor = payload["plan"].get("resume_cursor")
            if isinstance(cursor, dict):
                next_action = {
                    "kind": "task_resume",
                    "plan_id": payload["plan"]["plan_id"],
                    **cursor,
                }
        safety_doc = {
            "secret_value_recorded": False,
            "payment_credential_recorded": False,
            "business_mutation": False,
            "outbound": bool(payload.get("outbound", False)),
            **safety,
        }
        clean_payload = {
            **payload,
            "idempotency_key": idempotency_key,
            "summary": summary,
            "canonical_content_sha256": canonical_content_sha256,
        }
        _block_secret_like_fields(
            summary=summary,
            source_refs=source_refs,
            actors=actors,
            payload=payload,
            safety=safety,
            next_action=next_action,
            clean_payload=clean_payload,
        )
        helper = _load_helper()
        password = helper.get_secret_value()
        try:
            sock = helper.connect(password)
            _bound_socket_io(sock)
            try:
                _authorize_append_scope(helper, sock, case_id=case_id)
                superseded_plan_id: str | None = None
                if event_type == "task.plan.updated":
                    superseded_plan_id = _validate_task_plan_revision(
                        helper,
                        sock,
                        case_id=case_id,
                        event_id=event_id,
                        payload=payload,
                    )
                elif event_type == "task.verification.recorded":
                    _validate_task_verification_against_plan(
                        helper,
                        sock,
                        case_id=case_id,
                        payload=payload,
                    )
                elif event_type == "route_back.sent":
                    _validate_route_back_sent_against_intent(
                        helper,
                        sock,
                        case_id=case_id,
                        idempotency_key=idempotency_key,
                        payload=payload,
                    )
                _validate_completed_plan_receipts(
                    helper,
                    sock,
                    case_id=case_id,
                    payload=payload,
                )
                sql = f"""
INSERT INTO {EVENT_TABLE} (
  event_id, schema_version, event_type, occurred_at, case_id,
  source, actor, subject, evidence, decision, status, next_action, safety, payload
) VALUES (
  {helper.sql_quote(event_id)}::uuid,
  'canonical_event.v1',
  {helper.sql_quote(event_type)},
  {helper.sql_quote(occurred_at)}::timestamptz,
  {helper.sql_quote(case_id)},
  {helper.json_sql(source)},
  {helper.json_sql(actor)},
  {helper.json_sql(subject)},
  {helper.json_sql(evidence)},
  {helper.json_sql(decision)},
  {helper.json_sql(status)},
  {helper.json_sql(next_action)},
  {helper.json_sql(safety_doc)},
  {helper.json_sql(clean_payload)}
)
ON CONFLICT (event_id) DO NOTHING;
"""
                tag = helper.query(sock, sql)["command_tag"]
                readback = helper.query(sock, f"""
SELECT event_id::text, event_type, case_id, occurred_at::text,
       payload->>'idempotency_key' AS idempotency_key,
       payload->>'canonical_content_sha256' AS canonical_content_sha256
FROM {EVENT_TABLE}
WHERE event_id = {helper.sql_quote(event_id)}::uuid
LIMIT 1;
""")["rows"]
            finally:
                try:
                    sock.close()
                except Exception:
                    pass
        finally:
            password = ""
        readback_verified = _readback_matches(
            readback,
            event_id=event_id,
            event_type=event_type,
            case_id=case_id,
            idempotency_key=idempotency_key,
            content_sha256=canonical_content_sha256,
        )
        idempotency_conflict = tag == "INSERT 0 0" and not readback_verified
        response = {
            "success": readback_verified,
            "status": (
                "CANONICAL_EVENT_APPEND_PASS"
                if readback_verified
                else (
                    "CANONICAL_EVENT_APPEND_IDEMPOTENCY_CONFLICT"
                    if idempotency_conflict
                    else "CANONICAL_EVENT_APPEND_READBACK_FAILED"
                )
            ),
            "event_id": event_id,
            "event_type": event_type,
            "case_id": case_id,
            "idempotency_key": idempotency_key,
            "canonical_content_sha256": canonical_content_sha256,
            "command_tag": tag,
            "readback": readback,
            "readback_verified": readback_verified,
            "write_may_have_occurred": not readback_verified and tag == "INSERT 0 1",
            "inserted": tag == "INSERT 0 1",
            "deduped": tag == "INSERT 0 0",
        }
        if idempotency_conflict:
            response["error"] = (
                "idempotency key already exists with different canonical content"
            )
        if event_type == "task.plan.updated" and readback_verified:
            plan = payload.get("plan") if isinstance(payload, dict) else {}
            session_key = _get_session_env("HERMES_SESSION_KEY", "").strip()
            if session_key:
                try:
                    from tools.approval import revoke_plan_capability

                    if isinstance(plan, dict) and plan.get("state") in {
                        "completed", "blocked", "cancelled"
                    }:
                        revoke_plan_capability(session_key, str(plan.get("plan_id") or ""))
                    if superseded_plan_id:
                        revoke_plan_capability(session_key, superseded_plan_id)
                except Exception:
                    # Capability revocation is a local defense-in-depth cleanup;
                    # the durable task event remains the canonical truth.
                    pass
        if event_type == "person.alias.learned" and readback_verified:
            from gateway.support_ops_team_registry import learn_team_member_alias

            response["alias"] = learn_team_member_alias(
                str(payload.get("alias")),
                str(payload.get("member_key")),
            )
        return json.dumps(response, ensure_ascii=False, sort_keys=True)
    except Exception as exc:
        return tool_error(f"CANONICAL_EVENT_APPEND_FAIL: {exc}")


def record_plan_approval_receipt(
    *,
    case_id: str,
    receipt: Dict[str, Any],
    source_refs: Optional[Dict[str, Any]] = None,
) -> str:
    """Persist an owner-verified exact-plan capability grant.

    This function is intentionally not registered as a model tool.  The todo
    approval path calls it only after ``grant_plan_capability`` has verified
    the authenticated owner and reduced commands to hashes.
    """
    return canonical_event_append_tool(
        event_type="approval.capability.recorded",
        case_id=case_id,
        summary=f"Exact plan capability granted for {str(receipt.get('plan_id') or '')[:160]}",
        source_refs=source_refs or {},
        actors={
            "actor": {
                "type": "authenticated_owner",
                "id": str(receipt.get("approved_by_user_id") or ""),
            },
            "subject": {
                "type": "task_plan",
                "id": str(receipt.get("plan_id") or ""),
            },
        },
        payload={"approval_receipt": receipt},
        # Bind durable grant uniqueness to the observed approval turn, not the
        # ephemeral in-memory approval UUID. Replaying an old owner message
        # after a process restart dedupes and cannot mint a fresh capability.
        idempotency_key=(
            f"approval-source:{receipt.get('approval_source_sha256')}:"
            f"plan:{receipt.get('plan_id')}"
        ),
    )


def record_plan_capability_check(
    *,
    case_id: str,
    receipt: Dict[str, Any],
    source_refs: Optional[Dict[str, Any]] = None,
) -> str:
    """Persist one deterministic exact-command capability authorization."""
    return canonical_event_append_tool(
        event_type="capability.check.recorded",
        case_id=case_id,
        summary=f"Exact plan capability authorized command hash for {str(receipt.get('plan_id') or '')[:160]}",
        source_refs=source_refs or {},
        actors={
            "actor": {"type": "service", "id": "hermes_plan_capability_runtime"},
            "subject": {
                "type": "task_plan",
                "id": str(receipt.get("plan_id") or ""),
            },
        },
        payload={"capability_receipt": receipt},
        idempotency_key=(
            f"capability-check:{receipt.get('approval_id')}:"
            f"{receipt.get('command_sha256')}:{receipt.get('remaining_uses_for_command')}"
        ),
    )


def _route_back_state_impl(
    case_id: str,
    target_ref: Dict[str, Any],
    message_summary: str,
    source_refs: Dict[str, Any],
    mode: str = "record_required_only",
    receipt: Optional[Dict[str, Any]] = None,
    blocker_reason: Optional[str] = None,
    idempotency_key: Optional[str] = None,
    _internal_sent: bool = False,
    _execution_binding: Optional[Dict[str, Any]] = None,
) -> str:
    """Record route-back required/sent/blocked state.

    This tool does not infer meaning and does not secretly send Discord messages.
    It records the state Hermes decided or the delivery receipt Hermes obtained.
    """
    try:
        target_ref = _normalize_dict(target_ref, "target_ref")
        source_refs = _normalize_dict(source_refs, "source_refs")
        receipt = _normalize_dict(receipt, "receipt")
        allowed_modes = {"record_required_only", "queue_intent", "record_blocked"}
        if _internal_sent:
            allowed_modes.add("record_sent_receipt")
        if mode not in allowed_modes:
            raise ValueError(f"mode_not_allowed:{mode}")
        if not target_ref.get("id") and not target_ref.get("mention") and not target_ref.get("lane"):
            raise ValueError("target_ref requires id, mention, or lane")
        if _contains_forbidden_dm_route_ref(target_ref) or _contains_forbidden_dm_route_ref(receipt):
            raise ValueError("route_back_state forbids direct-message/DM targets; use public approved channel/thread target or record_blocked")
        base_payload = {
            "route_back": {
                "target_ref": target_ref,
                "mode": mode,
                "message_summary": message_summary,
                "receipt": receipt or None,
                "blocker_reason": blocker_reason,
            },
            "next_action": {"kind": "deliver_route_back_or_record_receipt", "target_ref": target_ref},
        }
        if _execution_binding is not None:
            if not isinstance(_execution_binding, dict):
                raise ValueError("execution binding must be an object")
            binding_channel = str(
                _execution_binding.get("target_channel_id") or ""
            ).strip()
            binding_hash = str(_execution_binding.get("content_sha256") or "").strip()
            if not binding_channel or not _SHA256_RE.fullmatch(binding_hash):
                raise ValueError(
                    "execution binding requires target_channel_id and content_sha256"
                )
            base_payload["route_back"]["execution_binding"] = {
                "target_channel_id": binding_channel,
                "content_sha256": binding_hash,
            }
        terminal_outcome = False
        required_next_step = "deliver_route_back_or_record_blocked"
        if mode == "record_sent_receipt":
            event_type = "route_back.sent"
            if not receipt.get("message_id"):
                raise ValueError("record_sent_receipt requires receipt.message_id")
            base_payload["receipt"] = receipt
            terminal_outcome = True
            required_next_step = "none"
        elif mode == "record_blocked":
            event_type = "route_back.blocked"
            if not blocker_reason:
                raise ValueError("record_blocked requires blocker_reason")
            terminal_outcome = True
            required_next_step = "none"
        elif mode == "queue_intent":
            event_type = "route_back.intent.created"
        else:
            event_type = "route_back.required"
        _block_secret_like_fields(
            target_ref=target_ref,
            receipt=receipt,
            blocker_reason=blocker_reason,
            message_summary=message_summary,
            next_action=base_payload.get("next_action"),
            clean_payload=base_payload,
        )
        result = canonical_event_append_tool(
            event_type=event_type,
            case_id=case_id,
            summary=message_summary,
            source_refs=source_refs,
            actors={"subject": {"type": "route_back", "id": target_ref.get("id") or target_ref.get("lane") or "target"}},
            payload=base_payload,
            safety={"contains_secret": False, "contains_payment_credential": False},
            idempotency_key=idempotency_key,
        )
        try:
            data = json.loads(result)
        except Exception:
            return result
        if isinstance(data, dict) and data.get("success"):
            data["route_back"] = {
                "mode": mode,
                "event_type": event_type,
                "terminal_outcome": terminal_outcome,
                "required_next_step": required_next_step,
            }
            if not terminal_outcome:
                data["route_back"]["final_answer_guard"] = (
                    "Do not present this as delivered or complete. Continue in the same turn "
                    "until the message is actually sent and record_sent_receipt is recorded, "
                    "or record_blocked is recorded with a concrete blocker."
                )
            return json.dumps(data, ensure_ascii=False, sort_keys=True)
        return result
    except Exception as exc:
        return tool_error(f"ROUTE_BACK_STATE_FAIL: {exc}")


def route_back_tool(
    case_id: str,
    target_ref: Dict[str, Any],
    message_summary: str,
    source_refs: Dict[str, Any],
    mode: str = "record_required_only",
    receipt: Optional[Dict[str, Any]] = None,
    blocker_reason: Optional[str] = None,
    idempotency_key: Optional[str] = None,
) -> str:
    """Public model-facing route-back state writer; sent state is unavailable."""
    return _route_back_state_impl(
        case_id=case_id,
        target_ref=target_ref,
        message_summary=message_summary,
        source_refs=source_refs,
        mode=mode,
        receipt=receipt,
        blocker_reason=blocker_reason,
        idempotency_key=idempotency_key,
    )


def _record_route_back_execution_intent(
    *,
    case_id: str,
    target_ref: Dict[str, Any],
    message_summary: str,
    source_refs: Dict[str, Any],
    idempotency_key: str,
    execution_binding: Dict[str, Any],
) -> str:
    return _route_back_state_impl(
        case_id=case_id,
        target_ref=target_ref,
        message_summary=message_summary,
        source_refs=source_refs,
        mode="queue_intent",
        idempotency_key=idempotency_key,
        _execution_binding=execution_binding,
    )


def _record_route_back_sent_receipt(
    *,
    case_id: str,
    target_ref: Dict[str, Any],
    message_summary: str,
    source_refs: Dict[str, Any],
    receipt: Dict[str, Any],
    idempotency_key: str,
    execution_binding: Dict[str, Any],
) -> str:
    return _route_back_state_impl(
        case_id=case_id,
        target_ref=target_ref,
        message_summary=message_summary,
        source_refs=source_refs,
        mode="record_sent_receipt",
        receipt=receipt,
        idempotency_key=idempotency_key,
        _internal_sent=True,
        _execution_binding=execution_binding,
    )


def _resolve_route_back_public_target(target_ref: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve an exact approved public route-back target.

    This intentionally uses the team/channel registry, not business-keyword
    routing.  The first supported executor target is the owner lane: references
    to Emil/owner resolve to the public control-tower channel, never a DM.
    """
    candidate_values = []
    for key in ("id", "mention", "lane", "person", "target_person", "key"):
        value = str(target_ref.get(key) or "").strip()
        if value:
            candidate_values.append(value)

    channel_id = str(target_ref.get("channel_id") or target_ref.get("thread_id") or "").strip()
    raw_id = str(target_ref.get("id") or "").strip()
    if raw_id == SKYVISION_CONTROL_TOWER_CHANNEL_ID:
        channel_id = SKYVISION_CONTROL_TOWER_CHANNEL_ID

    resolved_member: TeamMember | None = None
    for value in candidate_values:
        resolution = resolve_team_member(value)
        if resolution.status == "resolved":
            resolved_member = resolution.member
            break
        if resolution.status == "ambiguous":
            raise ValueError("route_back_execute target_ref ambiguous; ask requester to clarify the public target")

    if resolved_member is not None:
        return {
            "channel_id": resolved_member.default_channel_id,
            "channel_type": "public_channel",
            "target_kind": "member_default_public_channel",
            "target_member_key": resolved_member.key,
            "target_member_id": resolved_member.discord_user_id,
            "target_mention": resolved_member.mention,
        }

    if channel_id == SKYVISION_CONTROL_TOWER_CHANNEL_ID:
        return {
            "channel_id": SKYVISION_CONTROL_TOWER_CHANNEL_ID,
            "channel_type": "public_channel",
            "target_kind": "owner_public_channel",
            "target_member_key": "emil_lomliev",
            "target_member_id": "1279454038731264061",
            "target_mention": "<@1279454038731264061>",
        }

    if channel_id:
        from gateway.channel_directory import is_discord_public_target

        if not is_discord_public_target(channel_id):
            raise ValueError("route_back_execute requires a directory-confirmed public Discord channel/thread")
        return {
            "channel_id": channel_id,
            "channel_type": "public_channel_or_thread",
            "target_kind": "exact_public_directory_target",
            "target_member_key": None,
            "target_member_id": None,
            "target_mention": None,
        }

    raise ValueError("route_back_execute target is unresolved; ask the requester to clarify the public channel/thread")


def _configured_discord_channel_allowed(channel_id: str) -> bool:
    allowed_raw = os.getenv("DISCORD_ALLOWED_CHANNELS", "").strip()
    if not allowed_raw:
        return False
    allowed = {item.strip() for item in allowed_raw.split(",") if item.strip()}
    return "*" in allowed or str(channel_id) in allowed


def _authorize_route_back_execution(
    *,
    case_id: str,
    public_target: Dict[str, Any],
) -> None:
    """Authorize caller/case and exact public target before any outbound send."""
    if not _canonical_scope_enforced():
        return
    helper = _load_helper()
    password = helper.get_secret_value()
    try:
        sock = helper.connect(password)
        _bound_socket_io(sock)
        try:
            _authorize_existing_case_scope(helper, sock, case_id=case_id)
            target_kind = str(public_target.get("target_kind") or "")
            if target_kind in {
                "member_default_public_channel",
                "owner_public_channel",
            }:
                return
            channel_id = str(public_target.get("channel_id") or "").strip()
            if _configured_discord_channel_allowed(channel_id):
                return
            target_match = _thread_authorization_match_sql(
                helper,
                channel_id,
                alias="targeted",
            )
            rows = helper.query(sock, f"""
SELECT EXISTS (
  SELECT 1 FROM {EVENT_TABLE} AS targeted
  WHERE targeted.case_id = {helper.sql_quote(case_id)}
    AND {target_match}
) AS target_linked;
""").get("rows", [])
            row = rows[0] if rows else None
            linked = (
                _bool_cell(row.get("target_linked"))
                if isinstance(row, dict)
                else _bool_cell(row[0] if isinstance(row, (list, tuple)) and row else False)
            )
            if not linked:
                raise PermissionError(
                    "route_back_execute target is neither configured nor runtime-linked to this case"
                )
        finally:
            try:
                sock.close()
            except Exception:
                pass
    finally:
        password = ""


def _existing_route_back_terminal(
    *,
    case_id: str,
    idempotency_key: str,
) -> Dict[str, Any]:
    """Return a legacy/new terminal row for this exact execution key."""
    if not _canonical_scope_enforced():
        return {}
    helper = _load_helper()
    password = helper.get_secret_value()
    try:
        sock = helper.connect(password)
        _bound_socket_io(sock)
        try:
            _authorize_existing_case_scope(helper, sock, case_id=case_id)
            event_ids = [
                _event_uuid(idempotency_key, "route_back.sent", case_id),
                _event_uuid(idempotency_key, "route_back.blocked", case_id),
            ]
            quoted = ", ".join(
                f"{helper.sql_quote(event_id)}::uuid" for event_id in event_ids
            )
            rows = helper.query(sock, f"""
SELECT event_type, payload
FROM {EVENT_TABLE}
WHERE case_id = {helper.sql_quote(case_id)}
  AND event_id IN ({quoted})
  AND event_type IN ('route_back.sent', 'route_back.blocked')
ORDER BY occurred_at DESC, event_id DESC
LIMIT 1;
""").get("rows", [])
        finally:
            try:
                sock.close()
            except Exception:
                pass
    finally:
        password = ""
    if not rows:
        return {}
    row = rows[0]
    if isinstance(row, dict):
        event_type = str(row.get("event_type") or "")
        payload = _decode_mapping(row.get("payload"))
    else:
        event_type = str(row[0] if len(row) > 0 else "")
        payload = _decode_mapping(row[1] if len(row) > 1 else None)
    return {"event_type": event_type, "payload": payload}


def _discord_verify_message_receipt(
    *,
    channel_id: str,
    message_id: str,
    expected_content_sha256: str | None = None,
    timeout: int = 15,
) -> Dict[str, Any]:
    from gateway.config import Platform
    from gateway.run import _gateway_runner_ref

    runner = _gateway_runner_ref()
    adapter = runner.adapters.get(Platform.DISCORD) if runner is not None else None
    verifier = getattr(adapter, "verify_public_message_receipt", None)
    loop = getattr(runner, "_gateway_loop", None) if runner is not None else None
    if adapter is None or not callable(verifier) or loop is None or not loop.is_running():
        raise RuntimeError("live_discord_receipt_verifier_unavailable")
    future = asyncio.run_coroutine_threadsafe(
        verifier(
            channel_id=str(channel_id),
            message_id=str(message_id),
            expected_content_sha256=expected_content_sha256,
        ),
        loop,
    )
    try:
        result = future.result(timeout=max(1, min(int(timeout), 60)))
    except concurrent.futures.TimeoutError:
        future.cancel()
        raise RuntimeError("discord_receipt_verification_timeout") from None
    if not isinstance(result, dict) or result.get("verified") is not True:
        raise RuntimeError("discord_receipt_verification_failed")
    expected_fields = {
        "channel_id": str(channel_id),
        "message_id": str(message_id),
    }
    if expected_content_sha256:
        expected_fields["content_sha256"] = str(expected_content_sha256)
    for key, expected in expected_fields.items():
        if str(result.get(key) or "") != expected:
            raise RuntimeError(f"discord_receipt_verification_{key}_mismatch")
    return result


def _discord_expected_content_sha256(content: str) -> str:
    """Hash the one exact Discord-rendered message the adapter will send."""
    from gateway.config import Platform
    from gateway.run import _gateway_runner_ref

    runner = _gateway_runner_ref()
    adapter = runner.adapters.get(Platform.DISCORD) if runner is not None else None
    if adapter is None:
        raise RuntimeError("live_discord_adapter_unavailable")
    rendered = adapter.format_message(str(content))
    chunks = adapter.truncate_message(rendered, adapter.MAX_MESSAGE_LENGTH)
    if len(chunks) != 1:
        raise RuntimeError("discord_route_back_requires_one_rendered_message")
    return hashlib.sha256(chunks[0].encode("utf-8")).hexdigest()


def _discord_post_message(channel_id: str, content: str, *, timeout: int = 15) -> Dict[str, Any]:
    from gateway.config import Platform
    from gateway.run import _gateway_runner_ref

    runner = _gateway_runner_ref()
    adapter = runner.adapters.get(Platform.DISCORD) if runner is not None else None
    if adapter is None:
        raise RuntimeError("live_discord_adapter_unavailable")
    loop = getattr(runner, "_gateway_loop", None)
    if loop is None or not loop.is_running():
        raise RuntimeError("live_discord_gateway_loop_unavailable")
    expected_content_sha256 = _discord_expected_content_sha256(content)
    future = asyncio.run_coroutine_threadsafe(
        adapter.send(
            str(channel_id),
            content,
            metadata={"require_single_public_receipt": True},
        ),
        loop,
    )
    try:
        result = future.result(timeout=max(1, min(int(timeout), 60)))
    except concurrent.futures.TimeoutError:
        future.cancel()
        raise RuntimeError("discord_adapter_send_timeout") from None
    if not getattr(result, "success", False):
        raise RuntimeError(str(getattr(result, "error", None) or "discord_adapter_send_failed"))
    message_id = str(getattr(result, "message_id", None) or "")
    verified = _discord_verify_message_receipt(
        channel_id=str(channel_id),
        message_id=message_id,
        expected_content_sha256=expected_content_sha256,
        timeout=timeout,
    )
    return {
        "id": message_id,
        "channel_id": str(channel_id),
        "adapter_receipt": True,
        "content_sha256": str(verified.get("content_sha256") or ""),
        "receipt_readback_verified": True,
    }


def _route_back_record_blocked(
    *,
    case_id: str,
    target_ref: Dict[str, Any],
    message_summary: str,
    source_refs: Dict[str, Any],
    blocker_reason: str,
    idempotency_key: Optional[str],
) -> Dict[str, Any]:
    result = route_back_tool(
        case_id=case_id,
        target_ref=target_ref,
        message_summary=message_summary,
        source_refs=source_refs,
        mode="record_blocked",
        blocker_reason=blocker_reason,
        idempotency_key=idempotency_key,
    )
    try:
        data = json.loads(result)
    except Exception:
        data = {"raw": result}
    return data if isinstance(data, dict) else {"raw": result}


def _sanitized_blocked_target_ref(target_ref: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only public-safe metadata when persisting a blocked target."""
    if _contains_forbidden_dm_route_ref(target_ref):
        return {
            "id": f"blocked-target:{_hash(target_ref)[:16]}",
            "target_kind": "forbidden_or_unresolved_target",
            "original_target_ref_sha256": _hash(target_ref),
        }
    allowed_keys = {
        "id", "mention", "lane", "channel_id", "thread_id",
        "channel_type", "target_kind", "target_member_key",
    }
    safe = {key: value for key, value in target_ref.items() if key in allowed_keys and value}
    safe.setdefault("id", f"blocked-target:{_hash(target_ref)[:16]}")
    return safe


def _blocked_execution_result(blocker_reason: str, blocked: Dict[str, Any]) -> str:
    recorded = bool(isinstance(blocked, dict) and blocked.get("success"))
    data: Dict[str, Any] = {
        "success": recorded,
        "status": (
            "ROUTE_BACK_EXECUTE_BLOCKED"
            if recorded
            else "ROUTE_BACK_EXECUTE_BLOCKED_RECORD_FAILED"
        ),
        "blocker_reason": blocker_reason,
        "route_back_record": blocked,
    }
    if not recorded:
        data["final_answer_guard"] = (
            "Delivery was blocked and durable route_back.blocked recording also failed. "
            "Do not claim a durable terminal outcome; report both blockers clearly."
        )
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def _route_back_intent_lease_expired(
    intent: Dict[str, Any],
    *,
    now: Optional[dt.datetime] = None,
) -> bool:
    """Return True only when a verified durable intent is provably stale."""
    readback = intent.get("readback") if isinstance(intent, dict) else None
    if not isinstance(readback, list) or len(readback) != 1:
        return False
    row = readback[0]
    if isinstance(row, dict):
        raw_occurred_at = row.get("occurred_at")
    elif isinstance(row, (list, tuple)) and len(row) > 3:
        raw_occurred_at = row[3]
    else:
        return False
    try:
        occurred_at = dt.datetime.fromisoformat(
            str(raw_occurred_at or "").strip().replace("Z", "+00:00")
        )
    except (TypeError, ValueError):
        return False
    if occurred_at.tzinfo is None:
        occurred_at = occurred_at.replace(tzinfo=dt.timezone.utc)
    current = now or dt.datetime.now(dt.timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=dt.timezone.utc)
    age_seconds = (current.astimezone(dt.timezone.utc) - occurred_at.astimezone(
        dt.timezone.utc
    )).total_seconds()
    return age_seconds >= ROUTE_BACK_INTENT_LEASE_SECONDS


def route_back_execute_tool(
    case_id: str,
    target_ref: Dict[str, Any],
    message: str,
    message_summary: str,
    source_refs: Dict[str, Any],
    idempotency_key: Optional[str] = None,
) -> str:
    """Deliver an exact public route-back and record the terminal outcome.

    This is the send+receipt executor counterpart to ``route_back_state``. It
    never sends DMs. If it cannot send to an approved public target, it records
    ``route_back.blocked`` instead of leaving the case in a pending state.
    """
    try:
        target_ref = _normalize_dict(target_ref, "target_ref")
        source_refs = _normalize_dict(source_refs, "source_refs")
        message = str(message or "").strip()
        message_summary = str(message_summary or "").strip() or message[:200]
        idempotency_key = str(idempotency_key or "").strip()
        if not idempotency_key:
            raise ValueError(
                "route_back_execute requires a stable idempotency_key before outbound delivery"
            )
        if not message:
            raise ValueError("message is required")
        if len(message) > MAX_ROUTE_BACK_MESSAGE_CHARS:
            blocker_reason = f"message_too_long:{len(message)}>{MAX_ROUTE_BACK_MESSAGE_CHARS}"
            blocked = _route_back_record_blocked(
                case_id=case_id,
                target_ref=_sanitized_blocked_target_ref(target_ref),
                message_summary=message_summary,
                source_refs=source_refs,
                blocker_reason=blocker_reason,
                idempotency_key=idempotency_key,
            )
            return _blocked_execution_result(blocker_reason, blocked)

        try:
            public_target = _resolve_route_back_public_target(target_ref)
            _authorize_route_back_execution(
                case_id=case_id,
                public_target=public_target,
            )
        except Exception as exc:
            blocker_reason = f"target_not_approved_or_unresolved:{type(exc).__name__}"
            blocked = _route_back_record_blocked(
                case_id=case_id,
                target_ref=_sanitized_blocked_target_ref(target_ref),
                message_summary=message_summary,
                source_refs=source_refs,
                blocker_reason=blocker_reason,
                idempotency_key=idempotency_key,
            )
            return _blocked_execution_result(blocker_reason, blocked)

        existing_terminal = _existing_route_back_terminal(
            case_id=case_id,
            idempotency_key=idempotency_key,
        )
        if existing_terminal:
            terminal_type = existing_terminal.get("event_type")
            terminal_payload = existing_terminal.get("payload") or {}
            route_back_payload = (
                terminal_payload.get("route_back")
                if isinstance(terminal_payload, dict)
                else {}
            )
            route_back_payload = (
                route_back_payload if isinstance(route_back_payload, dict) else {}
            )
            receipt = (
                terminal_payload.get("receipt")
                or route_back_payload.get("receipt")
                or {}
            )
            return json.dumps({
                "success": terminal_type == "route_back.sent",
                "status": (
                    "ROUTE_BACK_EXECUTE_ALREADY_SENT"
                    if terminal_type == "route_back.sent"
                    else "ROUTE_BACK_EXECUTE_ALREADY_BLOCKED"
                ),
                "receipt": receipt,
                "terminal_event_type": terminal_type,
                "final_answer_guard": (
                    "The exact execution key already has a terminal Canonical Brain "
                    "outcome. Do not send again."
                ),
            }, ensure_ascii=False, sort_keys=True)

        resolved_target_ref = {
            **{
                key: value
                for key, value in target_ref.items()
                if key not in {"channel_id", "thread_id", "chat_id"}
            },
            "id": target_ref.get("id") or public_target.get("target_member_id") or public_target["channel_id"],
            "mention": target_ref.get("mention") or public_target.get("target_mention"),
            "channel_id": public_target["channel_id"],
            "channel_type": public_target["channel_type"],
            "target_kind": public_target["target_kind"],
        }

        try:
            expected_content_sha256 = _discord_expected_content_sha256(message)
        except Exception as exc:
            blocker_reason = f"discord_rendered_receipt_unavailable:{type(exc).__name__}"
            blocked = _route_back_record_blocked(
                case_id=case_id,
                target_ref=_sanitized_blocked_target_ref(resolved_target_ref),
                message_summary=message_summary,
                source_refs=source_refs,
                blocker_reason=blocker_reason,
                idempotency_key=idempotency_key,
            )
            return _blocked_execution_result(blocker_reason, blocked)
        execution_binding = {
            "target_channel_id": str(public_target["channel_id"]),
            "content_sha256": expected_content_sha256,
        }

        _block_secret_like_fields(
            target_ref=resolved_target_ref,
            message=message,
            message_summary=message_summary,
            source_refs=source_refs,
        )

        # Claim one durable execution intent before sending. Only the runtime
        # that inserts the intent may perform the outbound mutation; retries or
        # concurrent workers fail closed instead of duplicating a public post.
        intent_raw = _record_route_back_execution_intent(
            case_id=case_id,
            target_ref=resolved_target_ref,
            message_summary=message_summary,
            source_refs=source_refs,
            idempotency_key=idempotency_key,
            execution_binding=execution_binding,
        )
        try:
            intent = json.loads(intent_raw)
        except Exception:
            intent = {"success": False, "raw": intent_raw}
        if not isinstance(intent, dict) or not intent.get("success"):
            return json.dumps({
                "success": False,
                "status": "ROUTE_BACK_EXECUTE_INTENT_FAILED",
                "route_back_record": intent,
                "final_answer_guard": (
                    "No public message was sent because the durable execution intent "
                    "could not be verified. Report the blocker and do not claim delivery."
                ),
            }, ensure_ascii=False, sort_keys=True)
        if intent.get("inserted") is not True:
            if not _route_back_intent_lease_expired(intent):
                return json.dumps({
                    "success": False,
                    "status": "ROUTE_BACK_EXECUTE_ALREADY_CLAIMED_PENDING",
                    "lease_seconds": ROUTE_BACK_INTENT_LEASE_SECONDS,
                    "route_back_record": intent,
                    "final_answer_guard": (
                        "Another worker may still be completing this exact execution. "
                        "Do not send or record a terminal outcome during the intent lease. "
                        "Retry after the lease or inspect the exact terminal Canonical Brain event."
                    ),
                }, ensure_ascii=False, sort_keys=True)
            blocker_reason = (
                "route_back_execution_intent_lease_expired_outcome_uncertain"
            )
            blocked = _route_back_record_blocked(
                case_id=case_id,
                target_ref=_sanitized_blocked_target_ref(resolved_target_ref),
                message_summary=message_summary,
                source_refs=source_refs,
                blocker_reason=blocker_reason,
                idempotency_key=idempotency_key,
            )
            return _blocked_execution_result(blocker_reason, blocked)

        try:
            delivery = _discord_post_message(public_target["channel_id"], message)
        except Exception as exc:
            blocker_reason = f"discord_send_failed:{type(exc).__name__}"
            blocked = _route_back_record_blocked(
                case_id=case_id,
                target_ref=_sanitized_blocked_target_ref(resolved_target_ref),
                message_summary=message_summary,
                source_refs=source_refs,
                blocker_reason=blocker_reason,
                idempotency_key=idempotency_key,
            )
            return _blocked_execution_result(blocker_reason, blocked)

        message_id = str(delivery.get("id") or "").strip() if isinstance(delivery, dict) else ""
        if not message_id:
            blocked = _route_back_record_blocked(
                case_id=case_id,
                target_ref=resolved_target_ref,
                message_summary=message_summary,
                source_refs=source_refs,
                blocker_reason="discord_send_missing_message_id_receipt",
                idempotency_key=idempotency_key,
            )
            return _blocked_execution_result(
                "discord_send_missing_message_id_receipt",
                blocked,
            )

        receipt = {
            "platform": "discord",
            "message_id": message_id,
            "channel_id": public_target["channel_id"],
            "chat_id": public_target["channel_id"],
            "channel_type": public_target["channel_type"],
            "target_kind": public_target["target_kind"],
            "content_sha256": str(
                delivery.get("content_sha256")
                or hashlib.sha256(message.encode("utf-8")).hexdigest()
            ),
        }
        if receipt["content_sha256"] != expected_content_sha256:
            blocked = _route_back_record_blocked(
                case_id=case_id,
                target_ref=_sanitized_blocked_target_ref(resolved_target_ref),
                message_summary=message_summary,
                source_refs=source_refs,
                blocker_reason="discord_send_content_hash_mismatch",
                idempotency_key=idempotency_key,
            )
            return _blocked_execution_result(
                "discord_send_content_hash_mismatch",
                blocked,
            )
        result = _record_route_back_sent_receipt(
            case_id=case_id,
            target_ref=resolved_target_ref,
            message_summary=message_summary,
            source_refs=source_refs,
            receipt=receipt,
            idempotency_key=idempotency_key,
            execution_binding=execution_binding,
        )
        try:
            record_data = json.loads(result)
        except Exception:
            record_data = {"raw": result}
        if not isinstance(record_data, dict) or not record_data.get("success"):
            blocked = _route_back_record_blocked(
                case_id=case_id,
                target_ref=_sanitized_blocked_target_ref(resolved_target_ref),
                message_summary=message_summary,
                source_refs=source_refs,
                blocker_reason="route_back_sent_receipt_persistence_failed",
                idempotency_key=idempotency_key,
            )
            return json.dumps({
                "success": False,
                "status": (
                    "ROUTE_BACK_EXECUTE_SENT_RECEIPT_RECORD_BLOCKED"
                    if isinstance(blocked, dict) and blocked.get("success")
                    else "ROUTE_BACK_EXECUTE_SENT_RECEIPT_AND_BLOCK_RECORD_FAILED"
                ),
                "receipt": receipt,
                "route_back_record": record_data,
                "blocked_record": blocked,
                "final_answer_guard": (
                    "The public message was sent, but durable route_back.sent recording failed. "
                    "Do not resend. Report the real adapter receipt, the recording failure, "
                    "and whether route_back.blocked was durably recorded for that blocker."
                ),
            }, ensure_ascii=False, sort_keys=True)

        return json.dumps({
            "success": True,
            "status": "ROUTE_BACK_EXECUTE_SENT",
            "receipt": receipt,
            "route_back_record": record_data,
        }, ensure_ascii=False, sort_keys=True)
    except Exception as exc:
        return tool_error(f"ROUTE_BACK_EXECUTE_FAIL: {exc}")


_QUERY_COLUMNS = [
    "event_id", "schema_version", "event_type", "case_id", "occurred_at",
    "source", "actor", "subject", "evidence", "decision", "status",
    "next_action", "safety", "payload",
]


def _normalize_query_rows(rows: Any) -> list[Dict[str, Any]]:
    normalized: list[Dict[str, Any]] = []
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, dict):
            normalized.append(row)
        elif isinstance(row, (list, tuple)):
            normalized.append(dict(zip(_QUERY_COLUMNS, row)))
    return normalized


def _thread_retrieval_match_sql(helper: Any, thread_id: str, *, alias: str = "e") -> str:
    ref = helper.sql_quote(thread_id)
    return f"""(
 {alias}.source->'source_refs'->>'thread_id' = {ref}
 OR {alias}.source->'source_refs'->>'chat_id' = {ref}
 OR {alias}.payload->'route_back'->'target_ref'->>'thread_id' = {ref}
 OR {alias}.payload->'route_back'->'target_ref'->>'channel_id' = {ref}
 OR {alias}.payload->'route_back'->'receipt'->>'thread_id' = {ref}
 OR {alias}.payload->'route_back'->'receipt'->>'channel_id' = {ref}
 OR {alias}.payload->'receipt'->>'thread_id' = {ref}
 OR {alias}.payload->'receipt'->>'channel_id' = {ref}
)"""


def _thread_authorization_match_sql(
    helper: Any,
    thread_id: str,
    *,
    alias: str = "e",
) -> str:
    """Match only immutable observed or runtime-attested Discord linkage.

    Model-authored ``source_refs`` and route-back targets remain useful
    retrieval metadata, but they can never grant read/write authority.
    """
    ref = helper.sql_quote(thread_id)
    runtime_receipt = (
        f"{alias}.evidence @> "
        "'[{\"verified\":true,\"attestation\":\"deterministic_runtime_receipt\"}]'::jsonb"
    )
    return f"""(
 (
   {alias}.source->'observed_session'->>'platform' = 'discord'
   AND (
     {alias}.source->'observed_session'->>'thread_id' = {ref}
     OR {alias}.source->'observed_session'->>'chat_id' = {ref}
   )
 )
 OR (
   {alias}.event_type = 'route_back.sent'
   AND {runtime_receipt}
   AND (
     {alias}.payload->'route_back'->'target_ref'->>'thread_id' = {ref}
     OR {alias}.payload->'route_back'->'target_ref'->>'channel_id' = {ref}
     OR {alias}.payload->'route_back'->'receipt'->>'thread_id' = {ref}
     OR {alias}.payload->'route_back'->'receipt'->>'channel_id' = {ref}
     OR {alias}.payload->'receipt'->>'thread_id' = {ref}
     OR {alias}.payload->'receipt'->>'channel_id' = {ref}
   )
 )
)"""


def _configured_plan_owner_ids() -> set[str]:
    if load_config is None:
        return set()
    try:
        cfg = load_config() or {}
    except Exception:
        return set()
    approvals = cfg.get("approvals") if isinstance(cfg, dict) else {}
    if not isinstance(approvals, dict):
        return set()
    return {
        str(value).strip()
        for value in approvals.get("plan_owner_user_ids") or []
        if str(value).strip()
    }


def _canonical_scope_enforced() -> bool:
    # Any live gateway/platform context is scoped even when a transient config
    # or helper probe fails. Availability decides whether the operation can run;
    # it must never decide whether authorization applies.
    if _get_session_env("HERMES_SESSION_PLATFORM", "").strip():
        return True
    try:
        return bool(check_canonical_brain_requirements())
    except Exception:
        return False


def _current_canonical_scope() -> tuple[str, str, bool]:
    platform = _get_session_env("HERMES_SESSION_PLATFORM", "").strip().casefold()
    current_thread = (
        _get_session_env("HERMES_SESSION_THREAD_ID", "").strip()
        or _get_session_env("HERMES_SESSION_CHAT_ID", "").strip()
    )
    user_id = _get_session_env("HERMES_SESSION_USER_ID", "").strip()
    owner_ids = _configured_plan_owner_ids()
    # Configured IDs here are Discord snowflakes. Never let an equal-looking
    # identifier from another adapter inherit global owner scope.
    return platform, current_thread, bool(
        platform == "discord" and user_id and user_id in owner_ids
    )


def _bool_cell(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().casefold() in {"1", "t", "true", "yes"}


def _case_scope_status(
    helper: Any,
    sock: Any,
    *,
    case_id: str,
    thread_id: str,
) -> tuple[bool, bool]:
    authorized_match = _thread_authorization_match_sql(
        helper,
        thread_id,
        alias="scoped",
    )
    rows = helper.query(sock, f"""
SELECT
  EXISTS (
    SELECT 1 FROM {EVENT_TABLE}
    WHERE case_id = {helper.sql_quote(case_id)}
  ) AS case_exists,
  EXISTS (
    SELECT 1 FROM {EVENT_TABLE} AS scoped
    WHERE scoped.case_id = {helper.sql_quote(case_id)}
      AND {authorized_match}
  ) AS scope_linked;
""").get("rows", [])
    if not rows:
        return False, False
    row = rows[0]
    if isinstance(row, dict):
        return _bool_cell(row.get("case_exists")), _bool_cell(row.get("scope_linked"))
    return (
        _bool_cell(row[0] if len(row) > 0 else False),
        _bool_cell(row[1] if len(row) > 1 else False),
    )


def _authorize_append_scope(helper: Any, sock: Any, *, case_id: str) -> None:
    """Fail closed in the enabled private runtime before mutating a case."""
    if not _canonical_scope_enforced():
        return
    platform, current_thread, owner_scoped = _current_canonical_scope()
    if owner_scoped:
        return
    if platform != "discord" or not current_thread:
        raise PermissionError(
            "canonical brain append requires an authenticated owner or exact observed Discord thread scope"
        )
    case_exists, scope_linked = _case_scope_status(
        helper,
        sock,
        case_id=case_id,
        thread_id=current_thread,
    )
    if case_exists and not scope_linked:
        raise PermissionError(
            "canonical brain append is outside the current observed Discord case scope"
        )


def _authorize_existing_case_scope(helper: Any, sock: Any, *, case_id: str) -> None:
    if not _canonical_scope_enforced():
        return
    platform, current_thread, owner_scoped = _current_canonical_scope()
    if owner_scoped:
        return
    if platform != "discord" or not current_thread:
        raise PermissionError(
            "canonical case access requires an authenticated owner or exact observed Discord thread scope"
        )
    case_exists, scope_linked = _case_scope_status(
        helper,
        sock,
        case_id=case_id,
        thread_id=current_thread,
    )
    if not case_exists or not scope_linked:
        raise PermissionError(
            "canonical case is not linked to the current observed Discord thread"
        )


def _query_where_sql(helper: Any, *, case_id: str, thread_id: str) -> str:
    platform, current_thread, owner_scoped = _current_canonical_scope()
    scope_enforced = _canonical_scope_enforced()
    if not owner_scoped and (platform or scope_enforced):
        if platform != "discord":
            raise PermissionError(
                "canonical brain query requires an authenticated owner outside Discord"
            )
        if not current_thread:
            raise PermissionError("canonical brain Discord query requires an exact current thread scope")
        if thread_id and thread_id != current_thread:
            raise PermissionError("canonical brain thread query is outside the current Discord thread scope")

    if thread_id:
        if owner_scoped or (not platform and not scope_enforced):
            return _thread_retrieval_match_sql(helper, thread_id, alias="e")
        return _thread_authorization_match_sql(helper, thread_id, alias="e")

    case_clause = f"e.case_id = {helper.sql_quote(case_id)}"
    if owner_scoped or (not platform and not scope_enforced):
        return case_clause
    scoped_match = _thread_authorization_match_sql(
        helper,
        current_thread,
        alias="scoped",
    )
    return f"""(
 {case_clause}
 AND EXISTS (
   SELECT 1 FROM {EVENT_TABLE} AS scoped
   WHERE scoped.case_id = e.case_id AND {scoped_match}
 )
)"""


def _query_select_sql(where: str, *, extra_where: str = "", limit: int) -> str:
    extra = f" AND ({extra_where})" if extra_where else ""
    return f"""
SELECT e.event_id::text, e.schema_version, e.event_type, e.case_id, e.occurred_at::text,
       e.source, e.actor, e.subject, e.evidence, e.decision, e.status,
       e.next_action, e.safety, e.payload
FROM {EVENT_TABLE} AS e
WHERE ({where}){extra}
ORDER BY e.occurred_at DESC, e.event_id DESC
LIMIT {limit};
"""


def _query_thread_case_ids(
    helper: Any,
    sock: Any,
    *,
    thread_id: str,
    max_cases: int,
) -> tuple[list[str], bool]:
    where = _query_where_sql(helper, case_id="", thread_id=thread_id)
    result = helper.query(sock, f"""
SELECT e.case_id, MAX(e.occurred_at)::text AS latest_event_at
FROM {EVENT_TABLE} AS e
WHERE {where}
GROUP BY e.case_id
ORDER BY MAX(e.occurred_at) DESC, e.case_id
LIMIT {max_cases + 1};
""")
    rows = result.get("rows", []) if isinstance(result, dict) else []
    case_ids: list[str] = []
    for row in rows if isinstance(rows, list) else []:
        value = row.get("case_id") if isinstance(row, dict) else (row[0] if row else None)
        case_id = str(value or "").strip()
        if case_id and case_id not in case_ids:
            case_ids.append(case_id)
    return case_ids[:max_cases], len(case_ids) > max_cases


def canonical_brain_query_tool(
    *,
    case_id: str = "",
    thread_id: str = "",
    limit: int = 80,
    view: str = "summary",
) -> str:
    """Read exact Canonical events and mechanically fold bounded case state."""
    try:
        case_id = str(case_id or "").strip()
        thread_id = str(thread_id or "").strip()
        view = str(view or "summary").strip()
        if bool(case_id) == bool(thread_id):
            raise ValueError("provide exactly one of case_id or thread_id")
        if case_id and not _CASE_ID_RE.fullmatch(case_id):
            raise ValueError("case_id must be a bounded safe identifier starting with case:")
        if view not in {"summary", "resume_bundle"}:
            raise ValueError("view must be summary or resume_bundle")
        if view == "resume_bundle" and not case_id:
            raise ValueError("resume_bundle requires an exact case_id; use thread summary to discover candidates")
        limit = int(limit)
        if limit < 1 or limit > 200:
            raise ValueError("limit must be between 1 and 200")

        support_incomplete_reasons: list[str] = []
        missing_verification_event_ids: list[str] = []
        helper = _load_helper()
        password = helper.get_secret_value()
        try:
            sock = helper.connect(password)
            _bound_socket_io(sock)
            try:
                candidate_cases_truncated = False
                if thread_id:
                    case_ids, candidate_cases_truncated = _query_thread_case_ids(
                        helper,
                        sock,
                        thread_id=thread_id,
                        max_cases=max(1, min(10, limit)),
                    )
                    recent_rows = []
                    truncated = candidate_cases_truncated
                    if case_ids:
                        per_case_limit = max(1, min(40, limit // len(case_ids)))
                        for linked_case_id in case_ids:
                            linked_where = _query_where_sql(
                                helper,
                                case_id=linked_case_id,
                                thread_id="",
                            )
                            linked_result = helper.query(
                                sock,
                                _query_select_sql(
                                    linked_where,
                                    limit=per_case_limit + 1,
                                ),
                            )
                            linked_raw = (
                                linked_result.get("rows", [])
                                if isinstance(linked_result, dict)
                                else []
                            )
                            linked_rows = _normalize_query_rows(linked_raw)
                            truncated = truncated or len(linked_rows) > per_case_limit
                            recent_rows.extend(linked_rows[:per_case_limit])
                        recent_rows.sort(
                            key=lambda row: (
                                str(row.get("occurred_at") or ""),
                                str(row.get("event_id") or ""),
                            ),
                            reverse=True,
                        )
                        recent_rows = recent_rows[:limit]
                else:
                    where = _query_where_sql(helper, case_id=case_id, thread_id="")
                    recent_result = helper.query(
                        sock,
                        _query_select_sql(where, limit=limit + 1),
                    )
                    recent_raw = (
                        recent_result.get("rows", [])
                        if isinstance(recent_result, dict)
                        else []
                    )
                    recent_normalized = _normalize_query_rows(recent_raw)
                    truncated = len(recent_normalized) > limit
                    recent_rows = recent_normalized[:limit]

                support_rows: list[Dict[str, Any]] = []
                if view == "resume_bundle":
                    plan_rows, plan_graph_truncated = _load_task_plan_graph_rows(
                        helper,
                        sock,
                        where=where,
                        case_id=case_id,
                    )
                    support_rows.extend(plan_rows)
                    if plan_graph_truncated:
                        support_incomplete_reasons.append(
                            "task_plan_graph_support_truncated"
                        )
                    from gateway.canonical_brain_projection import (
                        select_canonical_plan_head,
                    )

                    plan_head, plan_graph_error = select_canonical_plan_head(plan_rows)
                    if plan_graph_error:
                        support_incomplete_reasons.append(plan_graph_error)
                    for event_type, support_limit in (
                        ("task.verification.recorded", 80),
                        ("approval.capability.recorded", 20),
                        ("capability.check.recorded", 200),
                    ):
                        support_result = helper.query(
                            sock,
                            _query_select_sql(
                                where,
                                extra_where=f"e.event_type = {helper.sql_quote(event_type)}",
                                limit=support_limit,
                            ),
                        )
                        support_rows.extend(_normalize_query_rows(
                            support_result.get("rows", [])
                            if isinstance(support_result, dict)
                            else []
                        ))

                    head_plan = _decode_mapping(
                        _decode_mapping(plan_head.get("payload")).get("plan")
                    ) if plan_head else {}
                    if head_plan.get("state") == "completed":
                        raw_ids = head_plan.get("verification_event_ids")
                        if not isinstance(raw_ids, list) or len(raw_ids) > MAX_TASK_COLLECTION_ITEMS:
                            support_incomplete_reasons.append(
                                "completed_plan_verification_id_set_invalid"
                            )
                            raw_ids = []
                        exact_ids: list[str] = []
                        for raw_id in raw_ids:
                            value = str(raw_id or "").strip()
                            try:
                                uuid.UUID(value)
                            except (ValueError, TypeError, AttributeError):
                                support_incomplete_reasons.append(
                                    "completed_plan_verification_id_set_invalid"
                                )
                                continue
                            if value not in exact_ids:
                                exact_ids.append(value)
                        if exact_ids:
                            quoted_ids = ", ".join(
                                f"{helper.sql_quote(value)}::uuid"
                                for value in exact_ids
                            )
                            exact_result = helper.query(
                                sock,
                                _query_select_sql(
                                    where,
                                    extra_where=(
                                        "e.event_type = 'task.verification.recorded' "
                                        f"AND e.event_id IN ({quoted_ids})"
                                    ),
                                    limit=len(exact_ids) + 1,
                                ),
                            )
                            exact_rows = _normalize_query_rows(
                                exact_result.get("rows", [])
                                if isinstance(exact_result, dict)
                                else []
                            )
                            support_rows.extend(exact_rows)
                            found_ids = {
                                str(row.get("event_id") or "")
                                for row in exact_rows
                            }
                            missing_verification_event_ids = sorted(
                                set(exact_ids) - found_ids
                            )
                            if missing_verification_event_ids:
                                support_incomplete_reasons.append(
                                    "completed_plan_verification_support_missing"
                                )
            finally:
                try:
                    sock.close()
                except Exception:
                    pass
        finally:
            password = ""

        from gateway.canonical_brain_projection import fold_case_events

        combined_rows = recent_rows + support_rows
        cases = fold_case_events(combined_rows, timeline_limit=min(20, limit))
        return json.dumps({
            "success": True,
            "status": "CANONICAL_BRAIN_QUERY_PASS",
            "query": {
                "case_id": case_id or None,
                "thread_id": thread_id or None,
                "limit": limit,
                "view": view,
            },
            "event_count": len(recent_rows),
            "window_event_count": len(recent_rows),
            "support_event_count": len(support_rows),
            "support_incomplete": bool(support_incomplete_reasons),
            "support": {
                "complete": not support_incomplete_reasons,
                "reasons": list(dict.fromkeys(support_incomplete_reasons)),
                "missing_verification_event_ids": missing_verification_event_ids,
            },
            "truncated": truncated,
            "candidate_cases_truncated": candidate_cases_truncated,
            "case_count": len(cases),
            "cases": cases,
        }, ensure_ascii=False, sort_keys=True)
    except Exception as exc:
        return tool_error(f"CANONICAL_BRAIN_QUERY_FAIL: {exc}")


def canonical_active_plan_matches(*, case_id: str, plan_id: str) -> bool:
    """Narrow authorized check for one exact active plan identity."""
    case_id = str(case_id or "").strip()
    plan_id = str(plan_id or "").strip()
    if not _CASE_ID_RE.fullmatch(case_id) or not _CANONICAL_ID_RE.fullmatch(plan_id):
        return False
    try:
        helper = _load_helper()
        password = helper.get_secret_value()
        try:
            sock = helper.connect(password)
            _bound_socket_io(sock)
            try:
                _authorize_existing_case_scope(helper, sock, case_id=case_id)
                latest = _latest_task_plan_record(helper, sock, case_id)
            finally:
                try:
                    sock.close()
                except Exception:
                    pass
        finally:
            password = ""
    except Exception:
        return False
    plan = latest.get("plan") if isinstance(latest.get("plan"), dict) else {}
    return bool(
        latest
        and str(latest.get("plan_id") or "") == plan_id
        and str(plan.get("state") or "") == "active"
    )


def check_canonical_brain_requirements() -> bool:
    """Expose Canonical Brain tools only for explicit private/runtime installs.

    This is not an upstream-generic tool surface: it requires the private Cloud
    SQL helper and an explicit profile config enablement under
    ``canonical_brain.audit_bridge.enabled`` or ``canonical_brain.tools_enabled``.
    """
    if not CLOUD_SQL_HELPER.exists():
        return False
    if load_config is None:
        return False
    try:
        cfg = load_config() or {}
    except Exception:
        return False
    cb = cfg.get("canonical_brain") if isinstance(cfg, dict) else None
    if not isinstance(cb, dict):
        return False
    audit = cb.get("audit_bridge")
    return bool(cb.get("tools_enabled") or (isinstance(audit, dict) and audit.get("enabled")))


CANONICAL_EVENT_APPEND_SCHEMA = {
    "name": "canonical_event_append",
    "description": (
        "Append a durable operational event to a private/runtime Canonical Brain Cloud SQL. "
        "Use when Hermes has reasoned that durable state exists (case note, handoff, "
        "route_back.required/blocked, needs_review, resolver reply, or task workspace state). "
        "For a durable complex-task checkpoint use task.plan.updated with payload.plan containing "
        "plan_id, positive revision, objective, explicit state, stable success_criteria IDs, "
        "Todo-shaped steps (id/content/status), current_step_id, attempts/decisions/artifacts, "
        "resume_cursor, and verification_event_ids when completed. Record concrete checks first "
        "as task.verification.recorded with the exact plan_revision, criterion_ids, and a receipt "
        "reference. A replacement plan starts at revision 1 and supplies both "
        "supersedes_plan_id and supersedes_plan_revision. This tool "
        "does NOT decide meaning; Hermes decides. Do not use keyword matching as authority. "
        "Verified route_back.sent and process-generated approval/capability receipt events are "
        "deliberately unavailable in the model schema."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "event_type": {"type": "string", "enum": sorted(ALLOWED_EVENT_TYPES - RECEIPT_REQUIRED_EVENT_TYPES)},
            "case_id": {"type": "string", "description": "Canonical case id, must start with case:"},
            "summary": {"type": "string", "description": "Short operational summary"},
            "source_refs": {"type": "object", "description": "Exact source refs: platform + message/thread/event/manual ref"},
            "actors": {"type": "object", "description": "Optional actor/subject/requester/target refs"},
            "payload": {"type": "object", "description": "Event payload; no secrets/payment credentials"},
            "safety": {"type": "object", "description": "Safety flags; contains_secret/payment_credential block append"},
            "idempotency_key": {"type": "string", "description": "Optional stable idempotency key"},
        },
        "required": ["event_type", "case_id", "summary", "source_refs"],
    },
}

ROUTE_BACK_SCHEMA = {
    "name": "route_back_state",
    "description": (
        "Record route-back required, queued, or blocked state in private/runtime Canonical Brain. "
        "This tool never records sent state. Use route_back_execute for idempotent public send, "
        "live Discord readback, and a receipt-coupled terminal event."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "case_id": {"type": "string"},
            "target_ref": {"type": "object", "description": "Target person/lane/mention/channel refs"},
            "message_summary": {"type": "string"},
            "source_refs": {"type": "object"},
            "mode": {"type": "string", "enum": ["record_required_only", "queue_intent", "record_blocked"], "default": "record_required_only"},
            "blocker_reason": {"type": "string", "description": "Required for record_blocked"},
            "idempotency_key": {"type": "string"},
        },
        "required": ["case_id", "target_ref", "message_summary", "source_refs"],
    },
}

ROUTE_BACK_EXECUTE_SCHEMA = {
    "name": "route_back_execute",
    "description": (
        "Execute an exact approved public route-back for private/runtime Canonical Brain cases. "
        "Use this when the route-back target is already known and is a directory-confirmed public "
        "Discord channel/thread or an exact registered teammate public lane. The tool sends first "
        "through the live adapter, then records route_back.sent with the real "
        "Discord receipt/message_id. If it cannot send safely, it records route_back.blocked and returns "
        "that terminal outcome instead of leaving route_back.required pending."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "case_id": {"type": "string"},
            "target_ref": {"type": "object", "description": "Exact public target/member/lane/channel refs; no DM refs"},
            "message": {"type": "string", "description": "The public route-back message to send"},
            "message_summary": {"type": "string", "description": "Short durable summary of the route-back"},
            "source_refs": {"type": "object", "description": "Exact source refs: platform + message/thread/event/manual ref"},
            "idempotency_key": {"type": "string", "description": "Required stable lifecycle execution key; retries never duplicate delivery"},
        },
        "required": ["case_id", "target_ref", "message", "message_summary", "source_refs", "idempotency_key"],
    },
}

CANONICAL_BRAIN_QUERY_SCHEMA = {
    "name": "canonical_brain_query",
    "description": (
        "Read exact Canonical Brain events and return a mechanical bounded state fold. Use thread_id "
        "with view=summary to discover exact case candidates; GPT chooses among multiple candidates. "
        "Then use exact case_id with view=resume_bundle to recover the latest model-authored plan, "
        "verification and runtime approval/capability receipts, next action, and bounded timeline. "
        "No keyword search, classification, prioritization, or routing is performed."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "case_id": {"type": "string", "description": "Exact canonical case id"},
            "thread_id": {"type": "string", "description": "Exact Discord source/target thread id"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 80},
            "view": {
                "type": "string",
                "enum": ["summary", "resume_bundle"],
                "default": "summary",
            },
        },
    },
}

registry.register(
    name="canonical_brain_query",
    toolset="canonical_brain",
    schema=CANONICAL_BRAIN_QUERY_SCHEMA,
    handler=lambda args, **kw: canonical_brain_query_tool(
        case_id=args.get("case_id", ""),
        thread_id=args.get("thread_id", ""),
        limit=args.get("limit", 80),
        view=args.get("view", "summary"),
    ),
    check_fn=check_canonical_brain_requirements,
    emoji="🧠",
)

registry.register(
    name="canonical_event_append",
    toolset="canonical_brain",
    schema=CANONICAL_EVENT_APPEND_SCHEMA,
    handler=lambda args, **kw: canonical_event_append_tool(
        event_type=args.get("event_type", ""),
        case_id=args.get("case_id", ""),
        summary=args.get("summary", ""),
        source_refs=args.get("source_refs") or {},
        actors=args.get("actors"),
        payload=args.get("payload"),
        safety=args.get("safety"),
        idempotency_key=args.get("idempotency_key"),
    ),
    check_fn=check_canonical_brain_requirements,
    emoji="🧠",
)

registry.register(
    name="route_back_state",
    toolset="canonical_brain",
    schema=ROUTE_BACK_SCHEMA,
    handler=lambda args, **kw: route_back_tool(
        case_id=args.get("case_id", ""),
        target_ref=args.get("target_ref") or {},
        message_summary=args.get("message_summary", ""),
        source_refs=args.get("source_refs") or {},
        mode=args.get("mode", "record_required_only"),
        receipt=args.get("receipt"),
        blocker_reason=args.get("blocker_reason"),
        idempotency_key=args.get("idempotency_key"),
    ),
    check_fn=check_canonical_brain_requirements,
    emoji="📨",
)

registry.register(
    name="route_back_execute",
    toolset="canonical_brain",
    schema=ROUTE_BACK_EXECUTE_SCHEMA,
    handler=lambda args, **kw: route_back_execute_tool(
        case_id=args.get("case_id", ""),
        target_ref=args.get("target_ref") or {},
        message=args.get("message", ""),
        message_summary=args.get("message_summary", ""),
        source_refs=args.get("source_refs") or {},
        idempotency_key=args.get("idempotency_key"),
    ),
    check_fn=check_canonical_brain_requirements,
    emoji="📨",
)
