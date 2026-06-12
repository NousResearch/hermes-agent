#!/usr/bin/env python3
"""Unified knowledge recall tool.

Collects source-backed context for "what do we know about X?" questions across
curated memories, skills, optional Markdown vault paths, and past sessions. The
tool is deliberately deterministic: it does retrieval and returns compact source
receipts plus an answer contract for the model to synthesize from.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from hermes_constants import get_hermes_home
from tools.registry import registry, tool_error
from tools.session_search_tool import session_search

_ENTRY_DELIMITER = "\n§\n"
_TEXT_EXTENSIONS = {".md", ".markdown", ".txt", ".rst", ".org"}
_DEFAULT_CONTEXT_CHARS = 700
_MAX_FILE_BYTES = 1_000_000


def _tokens(query: str) -> List[str]:
    words = re.findall(r"[\w-]+", query.lower())
    return [w for w in words if len(w) >= 2]


def _matches(text: str, tokens: Sequence[str]) -> bool:
    low = text.lower()
    return all(token in low for token in tokens) if tokens else False


def _score(text: str, tokens: Sequence[str]) -> int:
    low = text.lower()
    return sum(low.count(token) for token in tokens)


def _clip(text: str, limit: int = _DEFAULT_CONTEXT_CHARS) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _line_number(text: str, needle: str) -> int | None:
    if not needle:
        return None
    idx = text.lower().find(needle.lower())
    if idx < 0:
        return None
    return text.count("\n", 0, idx) + 1


def _read_text(path: Path) -> str | None:
    try:
        if not path.exists() or not path.is_file() or path.stat().st_size > _MAX_FILE_BYTES:
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _search_memories(query: str, max_results: int) -> List[Dict[str, Any]]:
    tokens = _tokens(query)
    results: List[Dict[str, Any]] = []
    memories_dir = get_hermes_home() / "memories"
    for target, filename in (("memory", "MEMORY.md"), ("user", "USER.md")):
        content = _read_text(memories_dir / filename)
        if not content:
            continue
        entries = [entry.strip() for entry in content.split(_ENTRY_DELIMITER) if entry.strip()]
        for idx, entry in enumerate(entries, start=1):
            if _matches(entry, tokens):
                results.append({
                    "target": target,
                    "entry_index": idx,
                    "score": _score(entry, tokens),
                    "excerpt": _clip(entry),
                })
    results.sort(key=lambda item: item.get("score", 0), reverse=True)
    return results[:max_results]


def _iter_skill_files(skills_dir: Path) -> Iterable[Path]:
    if not skills_dir.exists():
        return []
    return skills_dir.rglob("SKILL.md")


def _parse_frontmatter_name(content: str, fallback: str) -> str:
    match = re.search(r"(?m)^name:\s*['\"]?([^'\"\n]+)['\"]?\s*$", content[:2000])
    return match.group(1).strip() if match else fallback


def _parse_frontmatter_description(content: str) -> str:
    match = re.search(r"(?m)^description:\s*['\"]?([^'\"\n]+)['\"]?\s*$", content[:4000])
    return match.group(1).strip() if match else ""


def _search_skills(query: str, max_results: int) -> List[Dict[str, Any]]:
    tokens = _tokens(query)
    results: List[Dict[str, Any]] = []
    skills_dir = get_hermes_home() / "skills"
    for path in _iter_skill_files(skills_dir):
        content = _read_text(path)
        if not content or not _matches(content, tokens):
            continue
        try:
            rel = str(path.relative_to(skills_dir))
        except ValueError:
            rel = str(path)
        first_token = tokens[0] if tokens else ""
        results.append({
            "name": _parse_frontmatter_name(content, path.parent.name),
            "description": _parse_frontmatter_description(content),
            "path": rel,
            "line": _line_number(content, first_token),
            "score": _score(content, tokens),
            "excerpt": _clip(content),
        })
    results.sort(key=lambda item: item.get("score", 0), reverse=True)
    return results[:max_results]


def _safe_vault_root(path: str) -> Path | None:
    if not path or not isinstance(path, str):
        return None
    root = Path(path).expanduser()
    try:
        return root.resolve()
    except OSError:
        return None


def _iter_vault_files(root: Path) -> Iterable[Path]:
    if not root.exists() or not root.is_dir():
        return []
    return (p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in _TEXT_EXTENSIONS)


def _search_vault(query: str, vault_paths: Sequence[str] | None, max_results: int) -> List[Dict[str, Any]]:
    tokens = _tokens(query)
    results: List[Dict[str, Any]] = []
    for raw_root in vault_paths or []:
        root = _safe_vault_root(raw_root)
        if root is None:
            continue
        for path in _iter_vault_files(root):
            content = _read_text(path)
            if not content or not _matches(content, tokens):
                continue
            first_token = tokens[0] if tokens else ""
            try:
                rel = str(path.relative_to(root))
            except ValueError:
                rel = path.name
            results.append({
                "root": str(root),
                "path": str(path),
                "relative_path": rel,
                "line": _line_number(content, first_token),
                "score": _score(content, tokens),
                "excerpt": _clip(content),
            })
            if len(results) >= max_results * max(1, len(vault_paths or [])) * 4:
                break
    results.sort(key=lambda item: item.get("score", 0), reverse=True)
    return results[:max_results]


def _search_sessions(query: str, max_results: int, db=None, current_session_id: str | None = None) -> List[Dict[str, Any]]:
    try:
        payload = json.loads(session_search(
            query=query,
            limit=max_results,
            db=db,
            current_session_id=current_session_id or "",
        ))
    except Exception as exc:
        return [{"error": f"session_search failed: {exc}"}]
    if not isinstance(payload, dict) or not payload.get("success"):
        return []
    shaped = []
    for item in payload.get("results") or []:
        shaped.append({
            "session_id": item.get("session_id"),
            "title": item.get("title"),
            "source": item.get("source"),
            "when": item.get("when") or item.get("last_active") or item.get("started_at"),
            "snippet": item.get("snippet"),
            "match_message_id": item.get("match_message_id"),
            "messages": [
                {
                    "id": msg.get("id"),
                    "role": msg.get("role"),
                    "content": _clip(msg.get("content") or "", 350),
                    **({"anchor": True} if msg.get("anchor") else {}),
                }
                for msg in (item.get("messages") or [])[:5]
            ],
        })
    return shaped[:max_results]


def _normalize_sources(include_sources: Sequence[str] | None) -> List[str]:
    allowed = ["memory", "skills", "vault", "sessions"]
    if not include_sources:
        return allowed
    requested = []
    for source in include_sources:
        if isinstance(source, str):
            source = source.strip().lower()
            if source in allowed and source not in requested:
                requested.append(source)
    return requested or allowed


def knowledge_answer(
    query: str,
    vault_paths: Sequence[str] | None = None,
    include_sources: Sequence[str] | None = None,
    max_results: int = 5,
    db=None,
    current_session_id: str | None = None,
) -> str:
    """Return source receipts for a unified knowledge answer.

    The calling model should synthesize the returned evidence into the provided
    Known / Uncertain / Missing / Next action contract instead of treating this
    tool as an opaque answer generator.
    """
    if not isinstance(query, str) or not query.strip():
        return tool_error("knowledge_answer requires a non-empty query", success=False)
    try:
        max_results = int(max_results)
    except (TypeError, ValueError):
        max_results = 5
    max_results = max(1, min(max_results, 10))

    sources: Dict[str, Any] = {}
    selected = _normalize_sources(include_sources)
    clean_query = query.strip()

    if "memory" in selected:
        sources["memory"] = _search_memories(clean_query, max_results)
    if "skills" in selected:
        sources["skills"] = _search_skills(clean_query, max_results)
    if "vault" in selected:
        sources["vault"] = _search_vault(clean_query, vault_paths, max_results)
    if "sessions" in selected:
        sources["sessions"] = _search_sessions(clean_query, max_results, db=db, current_session_id=current_session_id)

    return json.dumps({
        "success": True,
        "query": clean_query,
        "sources": sources,
        "answer_contract": ["Known", "Uncertain", "Missing", "Next action"],
        "synthesis_instruction": (
            "Use only the source receipts above to draft a concise Known / Uncertain / "
            "Missing / Next action answer. Cite the source type and path/session/entry "
            "for material claims. If evidence is thin or absent, say so under Missing."
        ),
    }, ensure_ascii=False)


def check_knowledge_requirements() -> bool:
    return True


KNOWLEDGE_ANSWER_SCHEMA = {
    "name": "knowledge_answer",
    "description": (
        "Gather source-backed context for 'what do we know about X?' questions across "
        "curated memory, skills, optional Markdown vault paths, and session history. "
        "Returns evidence receipts plus a Known/Uncertain/Missing/Next action synthesis contract; "
        "it does not perform opaque autonomous writes or external access."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Topic or question to retrieve evidence for."},
            "vault_paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional local Markdown vault/project directories to search read-only.",
            },
            "include_sources": {
                "type": "array",
                "items": {"type": "string", "enum": ["memory", "skills", "vault", "sessions"]},
                "description": "Optional subset of sources. Defaults to all four sources.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum results per source, clamped to 1-10. Default 5.",
                "default": 5,
            },
        },
        "required": ["query"],
    },
}


registry.register(
    name="knowledge_answer",
    toolset="knowledge",
    schema=KNOWLEDGE_ANSWER_SCHEMA,
    handler=lambda args, **kw: knowledge_answer(
        query=args.get("query", ""),
        vault_paths=args.get("vault_paths"),
        include_sources=args.get("include_sources"),
        max_results=args.get("max_results", 5),
        db=kw.get("db"),
        current_session_id=kw.get("current_session_id"),
    ),
    check_fn=check_knowledge_requirements,
    emoji="🧠",
    max_result_size_chars=80_000,
)
