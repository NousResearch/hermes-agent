"""Immutable data contracts for hash-bound quarantine approvals."""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any

APPROVAL_BINDING_SCHEMA = "hermes-quarantine-binding-v0.1"


@dataclass(frozen=True)
class ApprovalBinding:
    schema_version: str
    candidate_sha256: str
    source_id: str
    source_version: str
    gate_release_sha256: str
    policy_version: str
    hermes_guard_version: str
    cisco_version: str
    nvidia_version: str
    scanner_completeness: tuple[tuple[str, bool], ...]
    scanner_reason_codes: tuple[tuple[str, tuple[str, ...]], ...]
    evidence_digest: str
    decision: str = "QUARANTINE"


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize deterministically for hashing/signature coverage."""
    return json.dumps(
        _json_ready(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _is_redirect(path: Path) -> bool:
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evidence_bundle_digest(root: Path) -> str:
    root = Path(root)
    if not root.is_dir() or _is_redirect(root):
        raise ValueError("evidence root missing or redirected")
    manifest: list[dict[str, str]] = []
    for current, dirs, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in dirs:
            child = current_path / name
            if _is_redirect(child):
                raise ValueError(f"evidence contains redirected directory: {child}")
        for name in files:
            child = current_path / name
            if _is_redirect(child):
                raise ValueError(f"evidence contains redirected file: {child}")
            manifest.append({
                "path": child.relative_to(root).as_posix(),
                "sha256": _sha256_file(child),
            })
    manifest.sort(key=lambda item: item["path"])
    return hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()


def approval_scope_digest(binding: ApprovalBinding, install_attempt_id: str) -> str:
    payload = {"binding": asdict(binding), "install_attempt_id": install_attempt_id}
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
