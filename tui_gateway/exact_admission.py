"""Atomic crash admission and durable receipts for exact prompt submissions.

One record owns both facts that must become durable before dispatch: the
payload/session-bound receipt and the active-turn recovery marker. The initial
admission is one fsynced temp-file replacement, never two independently durable
writes. Clearing a concluded turn removes only the marker; the receipt remains
for reconciliation and idempotency.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SUBMISSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{7,127}$")
_HEX_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_VERSION = 1
_MAX_RECORDS = 4096
_MAX_PROMPT_CHARS = 64_000
_LOCK = threading.RLock()
_BINDING_FIELDS = (
    "submission_id",
    "connection_id",
    "profile",
    "runtime_session_id",
    "stored_session_id",
    "lineage_root_id",
    "payload_digest",
    "source_digest",
    "context_digest",
    "attachment_manifest_digest",
    "attachment_count",
)


class ExactAdmissionError(RuntimeError):
    """Base class for exact-admission persistence failures."""


class ExactAdmissionConflict(ExactAdmissionError):
    """A submission id is already bound to a different immutable request."""


class ExactAdmissionInvalid(ExactAdmissionError):
    """An identifier, binding, or persisted record is malformed."""


def _identifier(value: Any, field: str, *, maximum: int = 256) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum or "\x00" in value:
        raise ExactAdmissionInvalid(f"invalid {field}")
    return value


def _submission_id(value: Any) -> str:
    if not isinstance(value, str) or not _SUBMISSION_ID_RE.fullmatch(value):
        raise ExactAdmissionInvalid("invalid submission_id")
    return value


def validate_submission_id(value: Any) -> str:
    return _submission_id(value)


def _digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _HEX_DIGEST_RE.fullmatch(value):
        raise ExactAdmissionInvalid(f"invalid {field}")
    return value


def validate_binding(binding: Any) -> dict[str, Any]:
    if type(binding) is not dict or set(binding) != set(_BINDING_FIELDS):
        raise ExactAdmissionInvalid("invalid exact admission binding")
    clean = {
        "submission_id": _submission_id(binding.get("submission_id")),
        "connection_id": _identifier(binding.get("connection_id"), "connection_id"),
        "profile": _identifier(binding.get("profile"), "profile"),
        "runtime_session_id": _identifier(binding.get("runtime_session_id"), "runtime_session_id"),
        "stored_session_id": _identifier(binding.get("stored_session_id"), "stored_session_id"),
        "lineage_root_id": _identifier(binding.get("lineage_root_id"), "lineage_root_id"),
        "payload_digest": _digest(binding.get("payload_digest"), "payload_digest"),
        "source_digest": _digest(binding.get("source_digest"), "source_digest"),
        "context_digest": _digest(binding.get("context_digest"), "context_digest"),
        "attachment_manifest_digest": _digest(
            binding.get("attachment_manifest_digest"), "attachment_manifest_digest"
        ),
    }
    count = binding.get("attachment_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0 or count > 32:
        raise ExactAdmissionInvalid("invalid attachment_count")
    clean["attachment_count"] = count
    return clean


def validate_exact_receipt(receipt: Any, binding: dict) -> dict:
    clean = validate_binding(binding)
    if type(receipt) is not dict:
        raise ExactAdmissionInvalid("invalid exact receipt")
    state = receipt.get("state")
    expected_fields = {"version", *_BINDING_FIELDS, "state", "accepted_at"}
    if state == "rejected":
        expected_fields.add("reason")
    if set(receipt) != expected_fields or type(receipt.get("version")) is not int or receipt.get("version") != _VERSION:
        raise ExactAdmissionInvalid("invalid exact receipt fields")
    if state not in {"durably_accepted", "rejected"}:
        raise ExactAdmissionInvalid("invalid exact receipt state")
    if any(receipt.get(field) != clean[field] for field in _BINDING_FIELDS):
        raise ExactAdmissionInvalid("exact receipt binding mismatch")
    _identifier(receipt.get("accepted_at"), "accepted_at")
    if state == "rejected":
        _identifier(receipt.get("reason"), "reason", maximum=128)
    return dict(receipt)


def _record_dir(home: Path | str) -> Path:
    directory = Path(home) / "desktop" / "exact-admissions"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _record_path(home: Path | str, submission_id: str) -> Path:
    safe = _submission_id(submission_id)
    return _record_dir(home) / f"{hashlib.sha256(safe.encode('utf-8')).hexdigest()}.json"


def _atomic_replace(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _encode(record: dict) -> bytes:
    return (json.dumps(record, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")


def _validate_record(value: Any, expected_submission_id: str | None = None) -> dict:
    if not isinstance(value, dict) or set(value) != {"version", "binding", "receipt", "turn"}:
        raise ExactAdmissionInvalid("invalid stored exact admission")
    if value.get("version") != _VERSION:
        raise ExactAdmissionInvalid("unsupported exact admission version")
    binding = validate_binding(value.get("binding"))
    if expected_submission_id is not None and binding["submission_id"] != expected_submission_id:
        raise ExactAdmissionInvalid("exact admission identity mismatch")
    receipt = validate_exact_receipt(value.get("receipt"), binding)
    state = receipt["state"]
    if state == "rejected":
        if value.get("turn") is not None:
            raise ExactAdmissionInvalid("rejected receipt cannot own a turn marker")
    elif value.get("turn") is not None:
        turn = value["turn"]
        if not isinstance(turn, dict) or set(turn) != {
            "submission_id",
            "stored_session_id",
            "prompt",
            "persist_user_text",
            "attempts",
            "started_at",
        }:
            raise ExactAdmissionInvalid("invalid exact turn marker")
        if turn["submission_id"] != binding["submission_id"] or turn["stored_session_id"] != binding["stored_session_id"]:
            raise ExactAdmissionInvalid("exact turn marker binding mismatch")
    return {"version": _VERSION, "binding": binding, "receipt": receipt, "turn": value.get("turn")}


def _read_record(path: Path, expected_submission_id: str | None = None) -> dict | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ExactAdmissionInvalid("stored exact admission is unreadable") from exc
    return _validate_record(value, expected_submission_id)


def get_exact_receipt(home: Path | str, submission_id: str) -> dict | None:
    safe = _submission_id(submission_id)
    with _LOCK:
        record = _read_record(_record_path(home, safe), safe)
    return None if record is None else record["receipt"]


def _new_receipt(binding: dict, state: str, *, reason: str | None = None) -> dict:
    receipt = {
        "version": _VERSION,
        **binding,
        "state": state,
        "accepted_at": datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z"),
    }
    if reason is not None:
        receipt["reason"] = _identifier(reason, "reason", maximum=128)
    return receipt


def _check_capacity(path: Path) -> None:
    if sum(1 for candidate in path.parent.iterdir() if candidate.is_file() and candidate.suffix == ".json") >= _MAX_RECORDS:
        raise ExactAdmissionError("exact admission capacity reached")


def record_exact_admission(
    home: Path | str,
    *,
    binding: dict,
    prompt: str,
    persist_user_text: str,
    attempts: int = 0,
) -> tuple[dict, bool]:
    clean = validate_binding(binding)
    if not isinstance(prompt, str) or not prompt or len(prompt) > _MAX_PROMPT_CHARS:
        raise ExactAdmissionInvalid("invalid exact prompt")
    if not isinstance(persist_user_text, str) or len(persist_user_text) > _MAX_PROMPT_CHARS:
        raise ExactAdmissionInvalid("invalid exact persisted source")
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 0:
        raise ExactAdmissionInvalid("invalid exact attempts")
    path = _record_path(home, clean["submission_id"])

    with _LOCK:
        existing = _read_record(path, clean["submission_id"])
        if existing is not None:
            if existing["binding"] != clean:
                raise ExactAdmissionConflict("submission_id is already bound to another exact request")
            return existing["receipt"], False
        _check_capacity(path)
        now = time.time()
        receipt = _new_receipt(clean, "durably_accepted")
        record = {
            "version": _VERSION,
            "binding": clean,
            "receipt": receipt,
            "turn": {
                "submission_id": clean["submission_id"],
                "stored_session_id": clean["stored_session_id"],
                "prompt": prompt,
                "persist_user_text": persist_user_text,
                "attempts": attempts,
                "started_at": now,
            },
        }
        _atomic_replace(path, _encode(record))
        return _validate_record(record, clean["submission_id"])["receipt"], True


def record_exact_rejection(home: Path | str, *, binding: dict, reason: str) -> tuple[dict, bool]:
    clean = validate_binding(binding)
    path = _record_path(home, clean["submission_id"])
    with _LOCK:
        existing = _read_record(path, clean["submission_id"])
        if existing is not None:
            if existing["binding"] != clean:
                raise ExactAdmissionConflict("submission_id is already bound to another exact request")
            return existing["receipt"], False
        _check_capacity(path)
        receipt = _new_receipt(clean, "rejected", reason=reason)
        record = {"version": _VERSION, "binding": clean, "receipt": receipt, "turn": None}
        _atomic_replace(path, _encode(record))
        return _validate_record(record, clean["submission_id"])["receipt"], True


def read_exact_turn_marker(home: Path | str, stored_session_id: str) -> dict | None:
    stored = _identifier(stored_session_id, "stored_session_id")
    newest = None
    with _LOCK:
        directory = _record_dir(home)
        for path in directory.glob("*.json"):
            record = _read_record(path)
            turn = record and record.get("turn")
            if not turn or turn.get("stored_session_id") != stored:
                continue
            if newest is None or float(turn.get("started_at") or 0) > float(newest.get("started_at") or 0):
                newest = turn
    return None if newest is None else {
        "submission_id": newest["submission_id"],
        "prompt": newest["prompt"],
        "persist_user_text": newest["persist_user_text"],
        "attempts": newest["attempts"],
        "started_at": newest["started_at"],
    }


def clear_exact_turn_marker(home: Path | str, stored_session_id: str, submission_id: str) -> None:
    stored = _identifier(stored_session_id, "stored_session_id")
    submission = _submission_id(submission_id)
    with _LOCK:
        path = _record_path(home, submission)
        record = _read_record(path, submission)
        turn = record and record.get("turn")
        if (
            not turn
            or turn.get("stored_session_id") != stored
            or turn.get("submission_id") != submission
        ):
            return
        record["turn"] = None
        _atomic_replace(path, _encode(record))
