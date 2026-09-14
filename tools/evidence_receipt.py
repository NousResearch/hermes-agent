"""Strict, bounded evidence transport receipts for finalized artifact bytes.

This is a plain library, not a model tool. A producer exclusively creates and
closes an artifact before emitting a receipt. A consumer validates provenance
before independently reading and hashing the exact on-disk bytes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RECEIPT_SCHEMA = "hermes-evidence-receipt-v2"
MAX_RECEIPT_BYTES = 2048
EXECUTION_KIND = "standalone-single-query"

VERDICT_VERIFIED = "VERIFIED"
VERDICT_UNAVAILABLE = "UNAVAILABLE"
VERDICT_INTEGRITY_FAILURE = "INTEGRITY_FAILURE"

RECEIPT_FIELDS = frozenset(
    {
        "schema",
        "status",
        "artifact_path",
        "sha256",
        "byte_length",
        "created_utc",
        "profile",
        "session_id",
        "execution_kind",
        "execution_id",
        "grant_id",
        "cell_id",
    }
)
BINDING_FIELDS = frozenset(
    {
        "profile",
        "session_id",
        "execution_kind",
        "execution_id",
        "grant_id",
        "cell_id",
    }
)

_ID_RE = re.compile(r"\A[A-Za-z0-9._-]+\Z")
_SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")
_MAX_INT64 = (1 << 63) - 1
_READ_CHUNK = 1024 * 1024
_ENV_TO_FIELD = {
    "HERMES_PROFILE": "profile",
    "HERMES_SESSION_ID": "session_id",
    "HERMES_EVIDENCE_GRANT_ID": "grant_id",
    "HERMES_EVIDENCE_CELL_ID": "cell_id",
}


class EvidenceTransportError(Exception):
    """Base class for producer-side evidence transport errors."""


class ReceiptFormatError(EvidenceTransportError):
    """The receipt or producer provenance violates the v2 contract."""


class ArtifactStateError(EvidenceTransportError):
    """The artifact cannot safely be produced or consumed."""


def _valid_identifier(value: Any, maximum: int) -> bool:
    return (
        isinstance(value, str)
        and 1 <= len(value) <= maximum
        and _ID_RE.fullmatch(value) is not None
    )


def _valid_created_utc(value: Any) -> bool:
    if not isinstance(value, str) or not 20 <= len(value) <= 40:
        return False
    if not (value.endswith("Z") or value.endswith("+00:00")):
        return False
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return False
    return parsed.utcoffset() == timezone.utc.utcoffset(parsed)


def _validate_binding(binding: Any) -> bool:
    if not isinstance(binding, dict) or set(binding) != BINDING_FIELDS:
        return False
    return (
        _valid_identifier(binding.get("profile"), 64)
        and _valid_identifier(binding.get("session_id"), 64)
        and binding.get("execution_kind") == EXECUTION_KIND
        and _valid_identifier(binding.get("execution_id"), 64)
        and binding.get("execution_id") == binding.get("session_id")
        and _valid_identifier(binding.get("grant_id"), 128)
        and _valid_identifier(binding.get("cell_id"), 16)
    )


def _validate_expected_binding(binding: Any) -> bool:
    """Validate authority shape while leaving semantic mismatches comparable."""
    if not isinstance(binding, dict) or set(binding) != BINDING_FIELDS:
        return False
    return (
        _valid_identifier(binding.get("profile"), 64)
        and _valid_identifier(binding.get("session_id"), 64)
        and _valid_identifier(binding.get("execution_kind"), 64)
        and _valid_identifier(binding.get("execution_id"), 64)
        and _valid_identifier(binding.get("grant_id"), 128)
        and _valid_identifier(binding.get("cell_id"), 16)
    )


def _validate_receipt_object(receipt: Any) -> dict[str, Any]:
    if not isinstance(receipt, dict) or set(receipt) != RECEIPT_FIELDS:
        raise ReceiptFormatError("receipt fields do not match the v2 allowlist")
    if receipt.get("schema") != RECEIPT_SCHEMA or receipt.get("status") != "ready":
        raise ReceiptFormatError("receipt schema or status is invalid")

    path = receipt.get("artifact_path")
    if (
        not isinstance(path, str)
        or not 1 <= len(path) <= 1024
        or not os.path.isabs(path)
        or any(ord(character) < 0x20 for character in path)
    ):
        raise ReceiptFormatError("artifact_path is invalid")
    if not isinstance(receipt.get("sha256"), str) or _SHA256_RE.fullmatch(
        receipt["sha256"]
    ) is None:
        raise ReceiptFormatError("sha256 is invalid")
    byte_length = receipt.get("byte_length")
    if (
        not isinstance(byte_length, int)
        or isinstance(byte_length, bool)
        or not 0 <= byte_length <= _MAX_INT64
    ):
        raise ReceiptFormatError("byte_length is invalid")
    if not _valid_created_utc(receipt.get("created_utc")):
        raise ReceiptFormatError("created_utc is invalid")
    if not _validate_binding({key: receipt.get(key) for key in BINDING_FIELDS}):
        raise ReceiptFormatError("receipt provenance binding is invalid")
    return receipt


def _canonical_json(receipt: dict[str, Any]) -> str:
    try:
        return json.dumps(
            receipt,
            separators=(",", ":"),
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ReceiptFormatError("receipt is not canonically encodable") from exc


def encode_receipt(receipt: dict[str, Any]) -> str:
    """Validate and canonically encode one complete v2 receipt."""
    validated = _validate_receipt_object(receipt)
    text = _canonical_json(validated)
    if len(text.encode("utf-8")) > MAX_RECEIPT_BYTES:
        raise ReceiptFormatError("receipt exceeds the 2048-byte limit")
    return text


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReceiptFormatError("receipt contains duplicate keys")
        result[key] = value
    return result


def _reject_json_constant(_value: str) -> None:
    raise ReceiptFormatError("non-finite JSON numbers are forbidden")


def parse_receipt(text: str) -> dict[str, Any]:
    """Parse a canonical v2 receipt, rejecting every non-contract shape."""
    if not isinstance(text, str):
        raise ReceiptFormatError("receipt text must be a string")
    try:
        encoded = text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ReceiptFormatError("receipt text is not valid UTF-8") from exc
    if len(encoded) > MAX_RECEIPT_BYTES:
        raise ReceiptFormatError("receipt exceeds the 2048-byte limit")
    if encoded.startswith(b"\xef\xbb\xbf"):
        raise ReceiptFormatError("receipt BOM is forbidden")
    try:
        receipt = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, ReceiptFormatError, TypeError, ValueError) as exc:
        if isinstance(exc, ReceiptFormatError):
            raise
        raise ReceiptFormatError("receipt is not valid JSON") from exc
    validated = _validate_receipt_object(receipt)
    if _canonical_json(validated) != text:
        raise ReceiptFormatError("receipt is not in canonical encoding")
    return validated


def _provenance_from_environment() -> dict[str, str]:
    binding: dict[str, str] = {}
    for environment_name, field_name in _ENV_TO_FIELD.items():
        try:
            binding[field_name] = os.environ[environment_name]
        except KeyError as exc:
            raise ReceiptFormatError("required producer provenance is unavailable") from exc
    binding["execution_kind"] = EXECUTION_KIND
    binding["execution_id"] = binding["session_id"]
    if not _validate_binding(binding):
        raise ReceiptFormatError("producer provenance is invalid")
    return binding


def write_artifact_exclusive(path: str | os.PathLike[str], data: bytes) -> None:
    """Exclusively create, flush, fsync, and close an artifact."""
    if not isinstance(data, bytes):
        raise ArtifactStateError("artifact data must be bytes")
    descriptor = os.open(os.fspath(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb") as artifact:
        artifact.write(data)
        artifact.flush()
        os.fsync(artifact.fileno())


def sha256_and_length(path: str | os.PathLike[str]) -> tuple[str, int]:
    """Hash exact bytes from a non-symlink regular file."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(os.fspath(path), flags)
    except OSError as exc:
        raise ArtifactStateError("artifact is unavailable") from exc
    digest = hashlib.sha256()
    byte_length = 0
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise ArtifactStateError("artifact is not a regular file")
        with os.fdopen(descriptor, "rb") as artifact:
            descriptor = -1
            while chunk := artifact.read(_READ_CHUNK):
                digest.update(chunk)
                byte_length += len(chunk)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return digest.hexdigest(), byte_length


def produce_receipt(path: str | os.PathLike[str], data: bytes) -> str:
    """Finalize a new artifact, then return its canonical ready receipt."""
    binding = _provenance_from_environment()
    absolute_path = str(Path(path).absolute())
    if len(absolute_path) > 1024 or any(ord(character) < 0x20 for character in absolute_path):
        raise ReceiptFormatError("artifact path cannot be represented in a receipt")

    write_artifact_exclusive(absolute_path, data)
    digest, byte_length = sha256_and_length(absolute_path)
    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "status": "ready",
        "artifact_path": absolute_path,
        "sha256": digest,
        "byte_length": byte_length,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        **binding,
    }
    return encode_receipt(receipt)


def verify_artifact(
    artifact_path: str | os.PathLike[str],
    expected_sha256: Any,
    expected_byte_length: Any,
) -> dict[str, Any]:
    """Independently verify a regular file against an immutable identity."""
    result = {
        "artifact_path": os.fspath(artifact_path),
        "expected_sha256": expected_sha256,
        "expected_byte_length": expected_byte_length,
        "observed_sha256": None,
        "observed_byte_length": None,
    }
    if (
        not isinstance(expected_sha256, str)
        or _SHA256_RE.fullmatch(expected_sha256) is None
        or not isinstance(expected_byte_length, int)
        or isinstance(expected_byte_length, bool)
        or not 0 <= expected_byte_length <= _MAX_INT64
    ):
        result.update(verdict=VERDICT_INTEGRITY_FAILURE, reason="expected identity is invalid")
        return result
    try:
        observed_sha256, observed_byte_length = sha256_and_length(artifact_path)
    except (ArtifactStateError, OSError, TypeError, ValueError):
        result.update(verdict=VERDICT_UNAVAILABLE, reason="artifact is unavailable")
        return result
    result["observed_sha256"] = observed_sha256
    result["observed_byte_length"] = observed_byte_length
    if observed_sha256 != expected_sha256 or observed_byte_length != expected_byte_length:
        result.update(verdict=VERDICT_INTEGRITY_FAILURE, reason="artifact identity mismatch")
    else:
        result["verdict"] = VERDICT_VERIFIED
    return result


def verify_receipt(receipt_text: Any, *, expected_binding: Any = None) -> dict[str, Any]:
    """Fail closed for all attacker-controlled receipt and binding inputs."""
    try:
        receipt = parse_receipt(receipt_text)
    except Exception:
        return {"verdict": VERDICT_UNAVAILABLE, "reason": "receipt is unavailable"}
    if not _validate_expected_binding(expected_binding):
        return {
            "verdict": VERDICT_UNAVAILABLE,
            "reason": "complete expected binding is unavailable",
            "expected_sha256": receipt["sha256"],
        }
    if any(receipt[field] != expected_binding[field] for field in BINDING_FIELDS):
        return {
            "verdict": VERDICT_INTEGRITY_FAILURE,
            "reason": "receipt provenance mismatch",
            "artifact_path": receipt["artifact_path"],
            "expected_sha256": receipt["sha256"],
            "expected_byte_length": receipt["byte_length"],
        }
    return verify_artifact(
        receipt["artifact_path"], receipt["sha256"], receipt["byte_length"]
    )
