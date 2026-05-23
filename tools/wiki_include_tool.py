"""Default-off read-only wiki include family for Memory Seam.

This tool is intentionally narrow: it can read only files named by explicit
``memory_seam.wiki.allowlist`` descriptors in ``config.yaml``.  It never crawls
wiki roots, never writes, and returns source/privacy/freshness/degraded labels
with path details redacted to a safe display path.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from tools.registry import registry

_MAX_WIKI_INCLUDE_BYTES = 128 * 1024
_ALLOWED_SUFFIXES = {".md", ".markdown", ".txt"}
_DENIED_PARTS = {
    ".env",
    ".git",
    ".hermes",
    ".ssh",
    "auth.json",
    "credentials",
    "credential",
    "private",
    "protected",
    "secret",
    "secrets",
    "session",
    "sessions",
    "state",
    "transcript",
    "transcripts",
    "logs",
}
_SECRET_VALUE_RE = re.compile(
    r"(?i)\b(api[_-]?key|token|secret|password|passwd|authorization)\b\s*[:=]\s*([^\s`'\"]+)"
)
_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}")


class WikiIncludeError(ValueError):
    """Expected user/config rejection for wiki include reads."""


def _load_memory_seam_wiki_config() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly() or {}
    except Exception:
        cfg = {}
    seam = cfg.get("memory_seam") or {}
    wiki = seam.get("wiki") or {}
    return wiki if isinstance(wiki, dict) else {}


def wiki_include_available() -> bool:
    """Tool availability check: default-off until config explicitly enables it."""

    cfg = _load_memory_seam_wiki_config()
    return bool(cfg.get("enabled") is True and cfg.get("allowlist"))


def _iter_descriptors(cfg: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    allowlist = cfg.get("allowlist") or []
    if isinstance(allowlist, dict):
        allowlist = list(allowlist.values())
    if not isinstance(allowlist, list):
        return []
    return [d for d in allowlist if isinstance(d, dict)]


def _descriptor_id(desc: Dict[str, Any]) -> str:
    return str(desc.get("id") or desc.get("include_id") or desc.get("name") or "").strip()


def _find_descriptor(include_id: str, cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    wanted = include_id.strip()
    for desc in _iter_descriptors(cfg):
        if _descriptor_id(desc) == wanted:
            return desc
    return None


def _expand_path(raw: str) -> Path:
    # Expand only local filesystem paths; no URI/network fetches in this slice.
    if "://" in raw:
        raise WikiIncludeError("wiki include rejected: descriptor path must be a local file path")
    return Path(raw).expanduser().resolve(strict=False)


def _default_wiki_root() -> Path:
    # Tests and profiles can override by using explicit descriptor roots.  This
    # fallback matches Atlas's sanitized wiki convention without requiring it.
    return Path.home() / "atlas" / "shared" / "wiki"


def _roots_for_descriptor(desc: Dict[str, Any], cfg: Dict[str, Any]) -> list[Path]:
    roots_raw = desc.get("roots") or desc.get("allowed_roots") or cfg.get("allowed_roots")
    if roots_raw is None:
        roots_raw = [_default_wiki_root()]
    if isinstance(roots_raw, (str, Path)):
        roots_raw = [roots_raw]
    roots: list[Path] = []
    for item in roots_raw if isinstance(roots_raw, list) else []:
        if item:
            roots.append(_expand_path(str(item)))
    return roots


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _safe_display_path(path: Path, roots: list[Path]) -> str:
    for root in roots:
        if _is_relative_to(path, root):
            return f"wiki:{path.relative_to(root).as_posix()}"
    return f"wiki:file:{hashlib.sha256(str(path).encode('utf-8')).hexdigest()[:12]}"


def _has_denied_part(path: Path) -> Optional[str]:
    for part in path.parts:
        lowered = part.lower()
        if lowered in _DENIED_PARTS:
            return part
        if any(marker in lowered for marker in ("private", "protected", "secret", "credential")):
            return part
    return None


def _validate_before_read(desc: Dict[str, Any], cfg: Dict[str, Any], include_id: str, subject: str) -> tuple[Path, list[Path]]:
    if (desc.get("family") or "wiki") != "wiki":
        raise WikiIncludeError("wiki include rejected: descriptor family is not wiki")

    expected_subject = str(desc.get("subject") or "").strip()
    if not expected_subject:
        raise WikiIncludeError("wiki include rejected: descriptor missing subject")
    if subject.strip() != expected_subject:
        raise WikiIncludeError("wiki include rejected: include/subject mismatch")

    path_raw = str(desc.get("path") or desc.get("file") or "").strip()
    if not path_raw:
        raise WikiIncludeError("wiki include rejected: descriptor missing path")
    path = _expand_path(path_raw)
    roots = _roots_for_descriptor(desc, cfg)
    if not roots:
        raise WikiIncludeError("wiki include rejected: descriptor has no allowed roots")
    if not any(_is_relative_to(path, root) for root in roots):
        raise WikiIncludeError("wiki include rejected: descriptor path is outside allowed wiki roots")
    relative_parts: list[str] = []
    for root in roots:
        if _is_relative_to(path, root):
            relative_parts = list(path.relative_to(root).parts)
            break
    denied = _has_denied_part(Path(*relative_parts)) if relative_parts else None
    if denied:
        raise WikiIncludeError("wiki include rejected: protected/private path component is not readable")
    if path.suffix.lower() not in _ALLOWED_SUFFIXES:
        raise WikiIncludeError("wiki include rejected: unsupported wiki file suffix")
    return path, roots


def _redact_content(text: str) -> str:
    text = _BEARER_RE.sub("Bearer [REDACTED]", text)
    text = _SECRET_VALUE_RE.sub(lambda m: f"{m.group(1)}=[REDACTED]", text)
    text = _EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    return text


def _freshness_label(st_mtime: float) -> str:
    dt = datetime.fromtimestamp(st_mtime, tz=timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def _degraded_filesystem_result(
    include_id: str,
    subject: str,
    display_path: str,
    source_label: str,
    privacy_label: str,
) -> Dict[str, Any]:
    return {
        "success": False,
        "include_id": include_id,
        "subject": subject,
        "source": source_label or "wiki",
        "privacy": privacy_label,
        "freshness": "unavailable",
        "degraded": True,
        "degraded_reason": "allowlisted wiki source is unreadable",
        "path": display_path,
        "error": "allowlisted wiki source is unreadable",
    }


def read_wiki_include(include_id: str, subject: str, *, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Read one explicitly allowlisted wiki include descriptor.

    ``config`` is injectable for tests; production reads ``memory_seam.wiki``
    from profile config.  All descriptor/subject/path privacy checks happen
    before the first filesystem read/stat.
    """

    cfg = config if config is not None else _load_memory_seam_wiki_config()
    if not bool(cfg.get("enabled") is True):
        return {
            "success": False,
            "source": "wiki",
            "privacy": "default_off",
            "freshness": "unavailable",
            "degraded": True,
            "degraded_reason": "memory_seam.wiki disabled",
            "error": "wiki include family is disabled",
        }

    desc = _find_descriptor(include_id, cfg)
    if desc is None:
        raise WikiIncludeError("wiki include rejected: include id is not allowlisted")

    path, roots = _validate_before_read(desc, cfg, include_id, subject)
    display_path = _safe_display_path(path, roots)
    source_label = str(desc.get("source") or "").strip()
    privacy_label = str(desc.get("privacy") or desc.get("privacy_label") or "least_sensitive").strip()

    try:
        exists = path.exists()
    except OSError:
        return _degraded_filesystem_result(include_id, subject, display_path, source_label, privacy_label)

    if not exists:
        return {
            "success": False,
            "include_id": include_id,
            "subject": subject,
            "source": source_label or "wiki",
            "privacy": privacy_label,
            "freshness": "unavailable",
            "degraded": True,
            "degraded_reason": "allowlisted wiki source is missing",
            "path": display_path,
            "error": "allowlisted wiki source is missing",
        }
    try:
        is_file = path.is_file()
    except OSError:
        return _degraded_filesystem_result(include_id, subject, display_path, source_label, privacy_label)

    if not is_file:
        raise WikiIncludeError("wiki include rejected: descriptor path is not a file")

    try:
        size = path.stat().st_size
    except OSError:
        return _degraded_filesystem_result(include_id, subject, display_path, source_label, privacy_label)

    if size > int(cfg.get("max_bytes") or _MAX_WIKI_INCLUDE_BYTES):
        raise WikiIncludeError("wiki include rejected: file exceeds wiki include size limit")

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        st = path.stat()
    except OSError:
        return _degraded_filesystem_result(include_id, subject, display_path, source_label, privacy_label)

    redacted = _redact_content(text)
    degraded = not bool(source_label)
    return {
        "success": True,
        "include_id": include_id,
        "subject": subject,
        "source": source_label or "wiki:no_source_label",
        "privacy": privacy_label,
        "freshness": _freshness_label(st.st_mtime),
        "degraded": degraded,
        "degraded_reason": "descriptor missing source label" if degraded else None,
        "path": display_path,
        "bytes": size,
        "content": redacted,
    }


def handle_wiki_include(args: Dict[str, Any], **_: Any) -> str:
    include_id = str(args.get("include_id") or "").strip()
    subject = str(args.get("subject") or "").strip()
    if not include_id or not subject:
        return json.dumps({"success": False, "error": "include_id and subject are required"})
    try:
        return json.dumps(read_wiki_include(include_id, subject), ensure_ascii=False)
    except WikiIncludeError as exc:
        return json.dumps(
            {
                "success": False,
                "source": "wiki",
                "privacy": "blocked_before_read",
                "freshness": "unread",
                "degraded": True,
                "degraded_reason": "policy_rejected_before_read",
                "error": str(exc),
            }
        )


WIKI_INCLUDE_SCHEMA = {
    "name": "wiki_include_read",
    "description": (
        "Read one explicitly allowlisted, sanitized Atlas wiki include. "
        "Default-off and read-only; requires include_id plus matching subject. "
        "Returns source/privacy/freshness/degraded labels and redacted content."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "include_id": {"type": "string", "description": "Allowlisted wiki descriptor id."},
            "subject": {"type": "string", "description": "Expected descriptor subject; must match before any read."},
        },
        "required": ["include_id", "subject"],
        "additionalProperties": False,
    },
}


registry.register(
    name="wiki_include_read",
    toolset="memory-seam-wiki",
    schema=WIKI_INCLUDE_SCHEMA,
    handler=handle_wiki_include,
    check_fn=wiki_include_available,
    requires_env=[],
    description="Read-only allowlisted sanitized wiki include",
    emoji="📚",
    max_result_size_chars=64_000,
)
