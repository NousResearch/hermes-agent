"""Privacy helpers for Wisdom capture and logs."""

from __future__ import annotations

import hashlib
import hmac
import os
import re
from pathlib import Path

from hermes_constants import get_hermes_home


_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----", re.IGNORECASE),
    re.compile(r"\bAuthorization\s*:\s*Bearer\s+[A-Za-z0-9._~+/=-]{12,}", re.IGNORECASE),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{20,}", re.IGNORECASE),
    re.compile(r"\bCookie\s*:\s*[^;\n]+=[^;\n]+", re.IGNORECASE),
    re.compile(r"\b(?:api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token|password|passwd)\s*[:=]\s*\S{8,}", re.IGNORECASE),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bsk-proj-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b", re.IGNORECASE),
    re.compile(r"\bghp_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
)


def detect_secret_like_text(text: str | None) -> bool:
    if not text:
        return False
    return any(pattern.search(text) for pattern in _SECRET_PATTERNS)


def redact_for_log(text: str | None) -> str:
    if text is None:
        return ""
    redacted = str(text)
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    if len(redacted) > 240:
        redacted = redacted[:240] + "...[truncated]"
    return redacted


def wisdom_salt_path() -> Path:
    return get_hermes_home() / "wisdom" / "salt"


def ensure_salt(path: Path | None = None) -> bytes:
    salt_path = path or wisdom_salt_path()
    salt_path.parent.mkdir(parents=True, exist_ok=True)
    if salt_path.exists():
        return salt_path.read_bytes()
    salt = os.urandom(32)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(str(salt_path), flags, 0o600)
    try:
        os.write(fd, salt)
    finally:
        os.close(fd)
    try:
        os.chmod(salt_path, 0o600)
    except OSError:
        pass
    return salt


def stable_hash(value: object | None, *, salt: bytes | None = None, prefix: str = "") -> str | None:
    if value is None:
        return None
    text = str(value)
    if text == "":
        return None
    key = salt or ensure_salt()
    digest = hmac.new(key, text.encode("utf-8", "surrogatepass"), hashlib.sha256).hexdigest()
    return f"{prefix}{digest[:32]}" if prefix else digest[:32]
