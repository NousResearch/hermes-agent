"""Content-specific rejection fingerprints for built-in memory proposals only."""

from __future__ import annotations

import hashlib
import json
import time

from hermes_constants import get_hermes_home
from utils import atomic_json_write, read_json_or_empty


def rejection_path():
    return get_hermes_home() / "pending" / ".rejections.json"


def proposal_identity(payload: dict) -> tuple[str, str, list]:
    operations = payload.get("operations") if payload.get("action") == "batch" else [payload]
    operations = [op for op in (operations or []) if isinstance(op, dict)]
    keys = [op.get("matched_entry") or op.get("old_text") or op.get("action") or "new entry" for op in operations]
    content = [op.get("content") if op.get("content") is not None else op.get("new_text") or "" for op in operations]
    return payload.get("target") or "memory", json.dumps(keys, ensure_ascii=False), content


def proposal_fingerprint(payload: dict) -> str:
    canonical = json.dumps(["memory", *proposal_identity(payload)], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def remember_rejection(record: dict) -> None:
    from hermes_cli.active_sessions import _FileLock

    target, entry_key, _ = proposal_identity(record["payload"])
    with _FileLock(rejection_path().with_suffix(".lock")):
        data = read_json_or_empty(rejection_path())
        data[proposal_fingerprint(record["payload"])] = {"at": time.time(), "target": target, "entry_key": entry_key}
        atomic_json_write(rejection_path(), data)


def was_rejected(payload: dict) -> bool:
    return proposal_fingerprint(payload) in read_json_or_empty(rejection_path())


def forget_rejections() -> int:
    from hermes_cli.active_sessions import _FileLock

    with _FileLock(rejection_path().with_suffix(".lock")):
        count = len(read_json_or_empty(rejection_path()))
        atomic_json_write(rejection_path(), {})
        return count
