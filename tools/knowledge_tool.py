#!/usr/bin/env python3
"""
Knowledge Ledger Tool Module - Cold, source-attributed local knowledge.

This toolset intentionally complements, rather than replaces, Hermes memory:
- MEMORY.md / USER.md stay small hot memory and are injected at session start.
- session_search remains episodic recall over raw conversations.
- skills remain procedural memory.
- knowledge/ is a cold/on-demand ledger for source-backed decisions,
  debugging histories, skill evaluations, project state, incidents, and
  inbox candidates.

Records live under $HERMES_HOME/knowledge/*. They are never injected into the
system prompt automatically; agents must explicitly call knowledge_search and
knowledge_get when the user/task needs grounded recall.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from hermes_constants import get_hermes_home
from tools.registry import registry, tool_result


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

ALLOWED_KINDS = {
    "inbox",
    "decisions",
    "debug",
    "skill-evals",
    "projects",
    "incidents",
    "entities",
    "sources",
}

KIND_ALIASES = {
    "decision": "decisions",
    "decisions": "decisions",
    "debug": "debug",
    "debugging": "debug",
    "skill_eval": "skill-evals",
    "skill-eval": "skill-evals",
    "skill_evals": "skill-evals",
    "skill-evals": "skill-evals",
    "project": "projects",
    "projects": "projects",
    "incident": "incidents",
    "incidents": "incidents",
    "entity": "entities",
    "entities": "entities",
    "source": "sources",
    "sources": "sources",
    "inbox": "inbox",
    "candidate": "inbox",
}

CONFIDENCE_VALUES = {"candidate", "observed", "confirmed", "inferred", "experimental", "stale"}
DEFAULT_SEARCH_KINDS = [
    "decisions",
    "debug",
    "skill-evals",
    "projects",
    "incidents",
    "entities",
    "sources",
    "inbox",
]


def get_knowledge_dir() -> Path:
    """Return the profile-scoped cold knowledge ledger directory."""
    return get_hermes_home() / "knowledge"


def check_knowledge_requirements() -> bool:
    """The local knowledge ledger has no external requirements."""
    return True


# ---------------------------------------------------------------------------
# Normalization and rendering helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _normalize_kind(kind: str | None) -> str:
    raw = (kind or "inbox").strip().lower().replace("_", "-")
    normalized = KIND_ALIASES.get(raw)
    if not normalized:
        raise ValueError(
            f"Unknown knowledge kind '{kind}'. Use one of: {', '.join(sorted(ALLOWED_KINDS))}."
        )
    return normalized


def _clean_string(value: Any) -> str:
    return str(value or "").strip()


def _clean_list(values: Any) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, Iterable):
        return []
    cleaned: List[str] = []
    for value in values:
        item = _clean_string(value)
        if item and item not in cleaned:
            cleaned.append(item)
    return cleaned


def _normalize_confidence(confidence: str | None, *, has_sources: bool, kind: str) -> str:
    raw = (confidence or "").strip().lower()
    if not raw:
        return "candidate" if kind == "inbox" and not has_sources else "observed"
    if raw not in CONFIDENCE_VALUES:
        raise ValueError(
            f"Unknown confidence '{confidence}'. Use one of: {', '.join(sorted(CONFIDENCE_VALUES))}."
        )
    return raw


def _slugify(title: str, fallback: str = "record") -> str:
    slug = re.sub(r"[^a-zA-Z0-9가-힣._-]+", "-", title.strip().lower())
    slug = re.sub(r"-+", "-", slug).strip("-._")
    return slug[:80] or fallback


def _yaml_scalar(value: Any) -> str:
    text = _clean_string(value)
    # Keep frontmatter simple and grep-friendly. Quote only when needed.
    if not text:
        return '""'
    if re.search(r"[:#\[\]{}\n\r]|^[-?]|\s$|^\s", text):
        return json.dumps(text, ensure_ascii=False)
    return text


def _frontmatter(metadata: Dict[str, Any]) -> str:
    lines = ["---"]
    for key, value in metadata.items():
        if isinstance(value, list):
            if value:
                lines.append(f"{key}:")
                for item in value:
                    lines.append(f"- {_yaml_scalar(item)}")
            else:
                lines.append(f"{key}: []")
        else:
            lines.append(f"{key}: {_yaml_scalar(value)}")
    lines.append("---")
    return "\n".join(lines)


def _render_record(
    *,
    record_id: str,
    kind: str,
    title: str,
    content: str,
    sources: List[str],
    confidence: str,
    status: str,
    tags: List[str],
    timestamp: str,
) -> str:
    metadata: Dict[str, Any] = {
        "id": record_id,
        "type": kind,
        "title": title,
        "status": status,
        "confidence": confidence,
        "created_at": timestamp,
        "updated_at": timestamp,
        "tags": tags,
        "sources": sources,
    }
    sections = [
        _frontmatter(metadata),
        "",
        f"# {title}",
        "",
        "## Current State",
        content.rstrip(),
        "",
        "## Timeline",
        f"- {timestamp} — Captured as `{kind}` record.",
        "",
        "## Evidence",
    ]
    if sources:
        sections.extend(f"- {source}" for source in sources)
    else:
        sections.append("- Candidate only: no source supplied yet.")
    sections.extend([
        "",
        "## Conflicts",
        "- None recorded.",
        "",
        "## Open Threads",
        "- [ ] Review/merge/delete during knowledge digest.",
        "",
    ])
    return "\n".join(sections)


def _safe_knowledge_root(*, create: bool = False) -> Path:
    """Return the ledger root without accepting a symlinked root directory."""
    root = get_knowledge_dir()
    if root.is_symlink():
        raise ValueError("Knowledge directory cannot be a symlink.")
    if create:
        root.mkdir(parents=True, exist_ok=True)
        if root.is_symlink():
            raise ValueError("Knowledge directory cannot be a symlink.")
    return root


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.stem}_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        # Knowledge records must stay confined to $HERMES_HOME/knowledge.
        # Use os.replace directly instead of utils.atomic_replace, because that
        # helper intentionally follows target symlinks for config-file use cases.
        # For ledger records, following a symlink could write outside knowledge/.
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _safe_kind_dir(kind: str) -> Path:
    """Create/return a knowledge kind directory without following kind symlinks."""
    root = _safe_knowledge_root(create=True)
    root_resolved = root.resolve()
    directory = root / kind
    directory.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink():
        raise ValueError(f"Knowledge kind directory cannot be a symlink: {kind}")
    try:
        directory.resolve().relative_to(root_resolved)
    except (OSError, ValueError) as exc:
        raise ValueError("Knowledge kind directory resolves outside the knowledge directory.") from exc
    return directory


def _unique_record_path(kind: str, title: str, timestamp: str) -> Tuple[str, Path]:
    directory = _safe_kind_dir(kind)
    stamp = timestamp.replace(":", "").replace("-", "").replace("Z", "Z")
    stem_base = f"{stamp}-{_slugify(title)}"
    stem = stem_base
    path = directory / f"{stem}.md"
    counter = 2
    while path.exists():
        stem = f"{stem_base}-{counter}"
        path = directory / f"{stem}.md"
        counter += 1
    record_id = f"{kind}/{stem}"
    return record_id, path


def _relative_to_root(path: Path) -> str:
    return path.relative_to(get_knowledge_dir()).as_posix()


# ---------------------------------------------------------------------------
# Frontmatter parsing and path safety
# ---------------------------------------------------------------------------


def _parse_frontmatter(text: str) -> Tuple[Dict[str, Any], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end == -1:
        return {}, text
    block = text[4:end]
    body = text[end + len("\n---\n"):]
    meta: Dict[str, Any] = {}
    current_list_key: Optional[str] = None
    for raw_line in block.splitlines():
        line = raw_line.rstrip()
        if not line:
            continue
        if line.startswith("- ") and current_list_key:
            meta.setdefault(current_list_key, []).append(_unquote_scalar(line[2:].strip()))
            continue
        if ":" not in line:
            current_list_key = None
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not value:
            meta[key] = []
            current_list_key = key
        elif value == "[]":
            meta[key] = []
            current_list_key = None
        else:
            meta[key] = _unquote_scalar(value)
            current_list_key = None
    return meta, body


def _unquote_scalar(value: str) -> str:
    if not value:
        return ""
    if value[0] in {'"', "'"}:
        try:
            return str(json.loads(value))
        except Exception:
            return value.strip('"\'')
    return value


def _title_from_text(meta: Dict[str, Any], body: str, path: Path) -> str:
    if meta.get("title"):
        return str(meta["title"])
    for line in body.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return path.stem


def _safe_record_path(*, record_id: str | None = None, rel_path: str | None = None) -> Path:
    root = _safe_knowledge_root(create=False).resolve()
    if record_id:
        candidate = record_id.strip()
        if not candidate.endswith(".md"):
            candidate = f"{candidate}.md"
    elif rel_path:
        candidate = rel_path.strip()
    else:
        raise ValueError("Provide id or path.")

    path = Path(candidate)
    if path.is_absolute():
        raise ValueError("Absolute paths are not allowed for knowledge records.")
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("Path traversal outside the knowledge directory is not allowed.") from exc
    return resolved


def _iter_record_paths(kind: str | None = None) -> Iterable[Path]:
    root = _safe_knowledge_root(create=False)
    root_resolved = root.resolve()
    if kind:
        kinds = [_normalize_kind(kind)]
    else:
        kinds = DEFAULT_SEARCH_KINDS
    for k in kinds:
        directory = root / k
        for path in sorted(directory.glob("*.md")):
            try:
                path.resolve().relative_to(root_resolved)
            except (OSError, ValueError):
                # Do not follow symlinked ledger entries outside knowledge/.
                continue
            yield path


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def knowledge_capture_tool(
    kind: str = "inbox",
    title: str = "",
    content: str = "",
    sources: Optional[List[str]] = None,
    confidence: Optional[str] = None,
    tags: Optional[List[str]] = None,
    status: Optional[str] = None,
) -> str:
    """Create a new cold knowledge ledger record."""
    try:
        normalized_kind = _normalize_kind(kind)
        clean_title = _clean_string(title)
        clean_content = _clean_string(content)
        clean_sources = _clean_list(sources)
        clean_tags = _clean_list(tags)

        if not clean_title:
            return tool_result(success=False, error="title cannot be empty.")
        if not clean_content:
            return tool_result(success=False, error="content cannot be empty.")
        if normalized_kind != "inbox" and not clean_sources:
            return tool_result(
                success=False,
                error=(
                    "source attribution is required for non-inbox knowledge records. "
                    "Capture unsourced material as kind='inbox' candidate, or provide sources."
                ),
            )

        unsourced_inbox = normalized_kind == "inbox" and not clean_sources
        clean_confidence = _normalize_confidence(
            confidence,
            has_sources=bool(clean_sources),
            kind=normalized_kind,
        )
        if unsourced_inbox:
            clean_confidence = "candidate"
            clean_status = "candidate"
        else:
            clean_status = _clean_string(status) or ("candidate" if normalized_kind == "inbox" else "active")
        timestamp = _now_iso()
        record_id, path = _unique_record_path(normalized_kind, clean_title, timestamp)
        markdown = _render_record(
            record_id=record_id,
            kind=normalized_kind,
            title=clean_title,
            content=clean_content,
            sources=clean_sources,
            confidence=clean_confidence,
            status=clean_status,
            tags=clean_tags,
            timestamp=timestamp,
        )
        _atomic_write_text(path, markdown)
        return tool_result(
            success=True,
            id=record_id,
            kind=normalized_kind,
            status=clean_status,
            confidence=clean_confidence,
            path=_relative_to_root(path),
            message=(
                "Knowledge record captured in cold storage. It is not injected into the "
                "system prompt; use knowledge_search/knowledge_get for on-demand recall."
            ),
        )
    except Exception as exc:
        return tool_result(success=False, error=str(exc))


def _score_record(query_terms: List[str], title: str, body: str, meta: Dict[str, Any]) -> int:
    haystacks = {
        "title": title.lower(),
        "body": body.lower(),
        "tags": " ".join(meta.get("tags") or []).lower() if isinstance(meta.get("tags"), list) else "",
        "sources": " ".join(meta.get("sources") or []).lower() if isinstance(meta.get("sources"), list) else "",
    }
    score = 0
    for term in query_terms:
        if not term:
            continue
        if term in haystacks["title"]:
            score += 5
        if term in haystacks["tags"]:
            score += 3
        if term in haystacks["sources"]:
            score += 2
        body_hits = haystacks["body"].count(term)
        score += min(body_hits, 5)
    phrase = " ".join(query_terms).strip()
    if phrase and phrase in haystacks["body"]:
        score += 4
    if phrase and phrase in haystacks["title"]:
        score += 8
    return score


def _snippet(body: str, query_terms: List[str], max_chars: int = 280) -> str:
    lower = body.lower()
    positions = [lower.find(term) for term in query_terms if term and lower.find(term) != -1]
    if positions:
        start = max(0, min(positions) - 80)
    else:
        start = 0
    snippet = re.sub(r"\s+", " ", body[start:start + max_chars]).strip()
    if start > 0:
        snippet = "…" + snippet
    if start + max_chars < len(body):
        snippet += "…"
    return snippet


def knowledge_search_tool(
    query: str,
    kind: Optional[str] = None,
    limit: int = 5,
) -> str:
    """Lexically search the cold knowledge ledger and return short snippets."""
    try:
        clean_query = _clean_string(query)
        if not clean_query:
            return tool_result(success=False, error="query cannot be empty.")
        try:
            limit = max(1, min(int(limit), 20))
        except Exception:
            limit = 5
        terms = [t for t in re.findall(r"[\w가-힣._-]+", clean_query.lower()) if t]
        if not terms:
            terms = [clean_query.lower()]

        rows: List[Dict[str, Any]] = []
        for path in _iter_record_paths(kind):
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            meta, body = _parse_frontmatter(text)
            title = _title_from_text(meta, body, path)
            score = _score_record(terms, title, body, meta)
            if score <= 0:
                continue
            rel = _relative_to_root(path)
            rows.append({
                "id": str(meta.get("id") or rel.removesuffix(".md")),
                "kind": str(meta.get("type") or path.parent.name),
                "title": title,
                "path": rel,
                "score": score,
                "confidence": meta.get("confidence"),
                "status": meta.get("status"),
                "updated_at": meta.get("updated_at"),
                "sources": meta.get("sources") or [],
                "snippet": _snippet(body, terms),
            })

        rows.sort(key=lambda r: (r["score"], r.get("updated_at") or ""), reverse=True)
        return tool_result(success=True, query=clean_query, count=len(rows[:limit]), results=rows[:limit])
    except Exception as exc:
        return tool_result(success=False, error=str(exc))


def knowledge_get_tool(id: Optional[str] = None, path: Optional[str] = None) -> str:
    """Read a single knowledge record by id or relative path."""
    try:
        record_path = _safe_record_path(record_id=id, rel_path=path)
        if not record_path.exists():
            return tool_result(success=False, error=f"Knowledge record not found: {id or path}")
        text = record_path.read_text(encoding="utf-8")
        meta, body = _parse_frontmatter(text)
        rel = _relative_to_root(record_path)
        return tool_result(
            success=True,
            id=str(meta.get("id") or rel.removesuffix(".md")),
            kind=str(meta.get("type") or record_path.parent.name),
            title=_title_from_text(meta, body, record_path),
            path=rel,
            metadata=meta,
            content=text,
        )
    except Exception as exc:
        return tool_result(success=False, error=str(exc))


# ---------------------------------------------------------------------------
# OpenAI Function-Calling Schemas
# ---------------------------------------------------------------------------

KNOWLEDGE_CAPTURE_SCHEMA = {
    "name": "knowledge_capture",
    "description": (
        "Capture a cold, source-attributed knowledge ledger record under "
        "$HERMES_HOME/knowledge. Records are NOT injected into the system prompt "
        "or hot memory; use knowledge_search/knowledge_get for on-demand recall. "
        "Use for durable decisions, debug findings, skill evaluations, project state, "
        "incidents, entities, sources, or inbox candidates. Non-inbox records require "
        "at least one source. Unsourced/automatic extraction must go to kind='inbox' "
        "as a candidate for later human/agent review."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "enum": sorted(ALLOWED_KINDS),
                "description": "Ledger section. Non-inbox sections require sources.",
                "default": "inbox",
            },
            "title": {"type": "string", "description": "Short record title."},
            "content": {"type": "string", "description": "Record body / current state."},
            "sources": {
                "type": "array",
                "description": "Source handles, URLs, commands, files, logs, or conversation references.",
                "items": {"type": "string"},
            },
            "confidence": {
                "type": "string",
                "enum": sorted(CONFIDENCE_VALUES),
                "description": "Confidence level. Defaults to candidate for unsourced inbox, observed otherwise.",
            },
            "tags": {
                "type": "array",
                "description": "Optional short tags for lexical retrieval.",
                "items": {"type": "string"},
            },
            "status": {
                "type": "string",
                "description": "Optional lifecycle status (candidate, active, resolved, superseded, stale, etc.).",
            },
        },
        "required": ["title", "content"],
    },
}

KNOWLEDGE_SEARCH_SCHEMA = {
    "name": "knowledge_search",
    "description": (
        "Search the cold Hermes knowledge ledger on demand. Returns ranked metadata "
        "and short snippets only, not a full memory dump. Use before knowledge_get "
        "when recalling source-backed decisions, debug histories, skill evaluations, "
        "or project state."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Lexical search query."},
            "kind": {
                "type": "string",
                "enum": sorted(ALLOWED_KINDS),
                "description": "Optional section filter.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results (1-20).",
                "default": 5,
            },
        },
        "required": ["query"],
    },
}

KNOWLEDGE_GET_SCHEMA = {
    "name": "knowledge_get",
    "description": (
        "Read one cold knowledge ledger record by id or relative path after search. "
        "Paths are constrained to $HERMES_HOME/knowledge to prevent traversal."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "Record id, e.g. decisions/20260504T120000Z-title."},
            "path": {"type": "string", "description": "Relative path returned by knowledge_search/capture."},
        },
        "required": [],
    },
}


registry.register(
    name="knowledge_capture",
    toolset="knowledge",
    schema=KNOWLEDGE_CAPTURE_SCHEMA,
    handler=lambda args, **kw: knowledge_capture_tool(
        kind=args.get("kind", "inbox"),
        title=args.get("title", ""),
        content=args.get("content", ""),
        sources=args.get("sources"),
        confidence=args.get("confidence"),
        tags=args.get("tags"),
        status=args.get("status"),
    ),
    check_fn=check_knowledge_requirements,
    emoji="🗃️",
)

registry.register(
    name="knowledge_search",
    toolset="knowledge",
    schema=KNOWLEDGE_SEARCH_SCHEMA,
    handler=lambda args, **kw: knowledge_search_tool(
        query=args.get("query", ""),
        kind=args.get("kind"),
        limit=args.get("limit", 5),
    ),
    check_fn=check_knowledge_requirements,
    emoji="🔎",
)

registry.register(
    name="knowledge_get",
    toolset="knowledge",
    schema=KNOWLEDGE_GET_SCHEMA,
    handler=lambda args, **kw: knowledge_get_tool(
        id=args.get("id"),
        path=args.get("path"),
    ),
    check_fn=check_knowledge_requirements,
    emoji="📄",
)
