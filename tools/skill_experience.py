"""Per-skill append-only experience memory.

A skill's ``.memory.md`` stores concise runtime lessons separately from its
stable instructions. Writes are durable and coordinated across threads and
processes; reads expose only the newest bounded tail.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

from tools.skill_sidecar_io import append_sidecar, read_sidecar

MEMORY_FILE = ".memory.md"
MEMORY_LOCK_FILE = ".memory.lock"
MAX_EXPERIENCE_ENTRY_BYTES = 8 * 1024
MAX_SKILL_MEMORY_BYTES = 32 * 1024


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def memory_path(skill_dir: Path) -> Path:
    return skill_dir / MEMORY_FILE


def append_skill_experience(
    skill_dir: Path,
    experience: str,
    *,
    idempotency_key: str | None = None,
) -> Path:
    """Append one redacted Markdown experience entry durably and idempotently."""
    if not isinstance(experience, str) or not experience.strip():
        raise ValueError("experience must be a non-empty string")
    text = experience.strip()
    size = len(text.encode("utf-8"))
    if size > MAX_EXPERIENCE_ENTRY_BYTES:
        raise ValueError(
            f"experience is {size:,} bytes (limit: {MAX_EXPERIENCE_ENTRY_BYTES:,} bytes)"
        )

    from agent.redact import redact_sensitive_for_persistence

    text = redact_sensitive_for_persistence(text)
    marker = ""
    marker_bytes = None
    if idempotency_key:
        if not idempotency_key.isalnum() or len(idempotency_key) > 64:
            raise ValueError("invalid experience idempotency key")
        marker = f"<!-- hermes-pending:{idempotency_key} -->\n"
        marker_bytes = marker.encode("utf-8")
    timestamp = _utc_now().astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    block = f"\n{marker}## {timestamp}\n{text}\n".encode("utf-8")
    return append_sidecar(
        skill_dir,
        MEMORY_FILE,
        block,
        lock_name=MEMORY_LOCK_FILE,
        dedupe_marker=marker_bytes,
    )


def read_skill_experience(
    skill_dir: Path,
    *,
    max_bytes: int = MAX_SKILL_MEMORY_BYTES,
) -> Tuple[str, bool]:
    """Return the newest bounded UTF-8 tail and whether older bytes were omitted."""
    data, truncated = read_sidecar(
        skill_dir, MEMORY_FILE, max_bytes=max_bytes, tail=True
    )
    if data is None:
        return "", False

    decoded = data.decode("utf-8", errors="ignore")
    if truncated:
        # Do not surface a partial first line after seeking into the tail.
        newline = decoded.find("\n")
        if newline >= 0:
            decoded = decoded[newline + 1 :]
        decoded = "[Earlier skill experience omitted]\n" + decoded
    return decoded, truncated
