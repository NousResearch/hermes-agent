#!/usr/bin/env python3
"""Recursive Context Tool - external corpus navigation for RLM-style workflows.

This tool gives the agent a small, deterministic substrate for Recursive
Language Model style work: keep large inputs outside prompt context, inspect them
with bounded search/read calls, and produce delegation prompts over source-line
ranges instead of pasting whole corpora into the model window.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home
from agent.file_safety import get_read_block_error
from agent.redact import redact_sensitive_text
from tools.binary_extensions import has_binary_extension
from tools.file_tools import _get_max_read_chars, _is_blocked_device
from tools.registry import registry

_STORE_NAME = "recursive_context"
_DEFAULT_CHUNK_LINES = 200
_MAX_READ_LINES = 500
_MAX_SEARCH_LIMIT = 50
_MAX_CONTEXT_LINES = 5
_MAX_MAP_CHUNKS = 100


def _json(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _error(message: str) -> str:
    return _json({"success": False, "error": message})


def _coerce_int(name: str, value: Any, default: int, minimum: int, maximum: Optional[int] = None) -> int:
    """Parse integer tool parameters with stable, parameter-named errors."""
    if value is None or value == "":
        parsed = default
    elif isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    else:
        try:
            parsed = int(value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"{name} must be an integer") from e
    parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(parsed, maximum)
    return parsed


def _validate_paths(paths: Optional[List[str]]) -> List[str]:
    if paths is None:
        return []
    if isinstance(paths, (str, bytes)) or not isinstance(paths, list):
        raise ValueError("paths must be a list of file path strings")
    if not all(isinstance(path, str) for path in paths):
        raise ValueError("paths entries must be strings")
    return paths


def _looks_binary(data: bytes) -> bool:
    if not data:
        return False
    if b"\x00" in data:
        return True
    sample = data[:4096]
    control = sum(1 for byte in sample if byte < 32 and byte not in b"\n\r\t\f\b")
    return bool(sample) and (control / len(sample)) > 0.30


def _redact_corpus_text(text: str) -> str:
    # Corpus storage is durable and searchable, so redact more aggressively
    # than source-code display. Force mode ignores global redaction opt-outs.
    return redact_sensitive_text(text, force=True, code_file=False)


def _store_dir() -> Path:
    path = get_hermes_home() / _STORE_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_slug(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", (name or "corpus").strip()).strip("-._")
    return slug[:48] or "corpus"


def _corpus_dir(corpus_id: str) -> Path:
    if not re.fullmatch(r"[a-zA-Z0-9._-]{8,96}", corpus_id or ""):
        raise ValueError("Invalid corpus_id")
    return _store_dir() / corpus_id


def _metadata_path(corpus_id: str) -> Path:
    return _corpus_dir(corpus_id) / "metadata.json"


def _text_path(corpus_id: str) -> Path:
    return _corpus_dir(corpus_id) / "corpus.txt"


def _records_path(corpus_id: str) -> Path:
    return _corpus_dir(corpus_id) / "records.jsonl"


def _load_metadata(corpus_id: str) -> Dict[str, Any]:
    try:
        path = _metadata_path(corpus_id)
    except ValueError as e:
        raise FileNotFoundError(str(e)) from e
    if not path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def _records_from_corpus_text(corpus_id: str) -> List[Dict[str, Any]]:
    path = _text_path(corpus_id)
    if not path.exists():
        raise FileNotFoundError(f"Corpus text not found: {corpus_id}")
    return [
        {"line": i + 1, "text": line, "source": None, "source_line": None}
        for i, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines())
    ]


def _load_records(corpus_id: str) -> List[Dict[str, Any]]:
    records_path = _records_path(corpus_id)
    if records_path.exists():
        records = []
        try:
            for raw in records_path.read_text(encoding="utf-8", errors="replace").splitlines():
                if raw.strip():
                    records.append(json.loads(raw))
            return records
        except (OSError, json.JSONDecodeError):
            # If the sidecar record index is corrupt but the canonical corpus
            # text exists, keep the corpus readable with text-only citations.
            return _records_from_corpus_text(corpus_id)

    # Backward-compatible fallback for corpora created by the first implementation.
    return _records_from_corpus_text(corpus_id)


def _read_paths(paths: List[str]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    corpus_line = 1
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        if _is_blocked_device(raw) or _is_blocked_device(str(path)):
            raise ValueError(f"Path is blocked because it may hang or stream indefinitely: {raw}")
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {raw}")
        if path.is_dir():
            raise IsADirectoryError(f"Path is a directory; pass files only: {raw}")
        if not path.is_file():
            raise ValueError(f"Path is not a regular file: {raw}")
        if has_binary_extension(str(path)):
            raise ValueError(f"Cannot ingest binary file: {raw}")
        block_error = get_read_block_error(str(path))
        if block_error:
            raise ValueError(block_error)
        max_chars = _get_max_read_chars()
        try:
            file_size = path.stat().st_size
        except OSError as e:
            raise OSError(f"Failed to stat {raw}: {e}") from e
        if file_size > max_chars:
            raise ValueError(f"File is too large to ingest safely ({file_size:,} bytes > {max_chars:,} byte limit): {raw}")
        try:
            raw_bytes = path.read_bytes()
        except OSError as e:
            raise OSError(f"Failed to read {raw}: {e}") from e
        if _looks_binary(raw_bytes):
            raise ValueError(f"Cannot ingest binary file content: {raw}")
        try:
            content = raw_bytes.decode("utf-8", errors="replace")
        except OSError as e:
            raise OSError(f"Failed to decode {raw}: {e}") from e
        if len(content) > max_chars:
            raise ValueError(f"File is too large to ingest safely ({len(content):,} characters > {max_chars:,} character limit): {raw}")
        content = _redact_corpus_text(content)
        normalized = content.replace("\r\n", "\n").replace("\r", "\n")
        for source_line, line in enumerate(normalized.splitlines(), start=1):
            records.append({
                "line": corpus_line,
                "text": line,
                "source": str(path),
                "source_line": source_line,
            })
            corpus_line += 1
    return records


def _records_from_text(text: str, start_line: int = 1) -> List[Dict[str, Any]]:
    normalized = _redact_corpus_text(text or "").replace("\r\n", "\n").replace("\r", "\n")
    return [
        {"line": start_line + i, "text": line, "source": None, "source_line": None}
        for i, line in enumerate(normalized.splitlines())
    ]


def _shape_line(record: Dict[str, Any]) -> Dict[str, Any]:
    shaped: Dict[str, Any] = {"line": int(record["line"]), "text": record.get("text", "")}
    if record.get("source") is not None:
        shaped["source"] = record["source"]
        shaped["source_line"] = record.get("source_line")
    return shaped


def _chunk_ranges(line_count: int, chunk_lines: int) -> List[Dict[str, int]]:
    if line_count <= 0:
        return []
    chunk_lines = _coerce_int("chunk_lines", chunk_lines, _DEFAULT_CHUNK_LINES, 1)
    ranges = []
    for start in range(1, line_count + 1, chunk_lines):
        end = min(start + chunk_lines - 1, line_count)
        ranges.append({"start_line": start, "end_line": end})
    return ranges


def _shape_corpus(metadata: Dict[str, Any]) -> Dict[str, Any]:
    # Deliberately omit raw text. The whole point is to keep corpora external.
    return {
        "corpus_id": metadata["corpus_id"],
        "name": metadata["name"],
        "line_count": metadata["line_count"],
        "char_count": metadata["char_count"],
        "chunk_lines": metadata["chunk_lines"],
        "chunk_count": metadata["chunk_count"],
        "created_at": metadata["created_at"],
        "sources": metadata.get("sources", []),
    }


def _create(name: str, text: Optional[str], paths: Optional[List[str]], chunk_lines: int) -> str:
    paths = _validate_paths(paths)
    if not text and not paths:
        return _error("create requires text or paths")
    chunk_lines = _coerce_int("chunk_lines", chunk_lines, _DEFAULT_CHUNK_LINES, 1)

    records = _records_from_text(text or "") if text else []
    sources: List[str] = []
    if paths:
        path_records = _read_paths(paths)
        if records and path_records:
            offset = len(records)
            for record in path_records:
                record["line"] = offset + int(record["line"])
        records.extend(path_records)
        sources = [str(Path(p).expanduser().resolve()) for p in paths]

    source_text = "\n".join(record["text"] for record in records)
    digest_payload = {
        "text": source_text,
        "chunk_lines": chunk_lines,
        "sources": [
            {"line": record["line"], "source": record.get("source"), "source_line": record.get("source_line")}
            for record in records
        ],
    }
    digest = hashlib.sha256(
        json.dumps(digest_payload, ensure_ascii=False, sort_keys=True).encode("utf-8", errors="replace")
    ).hexdigest()[:12]
    corpus_id = f"{_safe_slug(name)}-{digest}"
    cdir = _corpus_dir(corpus_id)
    cdir.mkdir(parents=True, exist_ok=True)
    _text_path(corpus_id).write_text(source_text, encoding="utf-8")
    _records_path(corpus_id).write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records),
        encoding="utf-8",
    )

    metadata = {
        "corpus_id": corpus_id,
        "name": name or corpus_id,
        "line_count": len(records),
        "char_count": len(source_text),
        "chunk_lines": chunk_lines,
        "chunk_count": len(_chunk_ranges(len(records), chunk_lines)),
        "created_at": int(time.time()),
        "sources": sources,
        "record_format": 1,
    }
    _metadata_path(corpus_id).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return _json({"success": True, "corpus": _shape_corpus(metadata)})


def _list() -> str:
    corpora = []
    for path in sorted(_store_dir().iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not path.is_dir():
            continue
        meta_path = path / "metadata.json"
        if not meta_path.exists():
            continue
        try:
            corpora.append(_shape_corpus(json.loads(meta_path.read_text(encoding="utf-8"))))
        except (OSError, json.JSONDecodeError, KeyError):
            continue
    return _json({"success": True, "corpora": corpora, "count": len(corpora)})


def _read(corpus_id: str, start_line: int, line_count: int) -> str:
    metadata = _load_metadata(corpus_id)
    records = _load_records(corpus_id)
    start_line = _coerce_int("start_line", start_line, 1, 1)
    line_count = _coerce_int("line_count", line_count, 80, 1, _MAX_READ_LINES)
    if not records:
        return _json({
            "success": True,
            "corpus_id": corpus_id,
            "name": metadata["name"],
            "range": {"start_line": 0, "end_line": 0},
            "lines": [],
        })
    if start_line > len(records):
        return _json({
            "success": True,
            "corpus_id": corpus_id,
            "name": metadata["name"],
            "range": {"start_line": start_line, "end_line": start_line},
            "lines": [],
        })
    start_idx = min(start_line - 1, len(records))
    end_idx = min(start_idx + line_count, len(records))
    return _json({
        "success": True,
        "corpus_id": corpus_id,
        "name": metadata["name"],
        "range": {"start_line": start_idx + 1 if records else 0, "end_line": end_idx},
        "lines": [_shape_line(records[i]) for i in range(start_idx, end_idx)],
    })


def _query_terms(query: str) -> List[str]:
    # De-dupe while preserving order so repeated words do not distort matching.
    seen = set()
    terms = []
    for term in re.findall(r"[\w'-]+", query or ""):
        normalized = term.lower()
        if len(normalized) <= 1 or normalized in seen:
            continue
        seen.add(normalized)
        terms.append(normalized)
    return terms


def _search(corpus_id: str, query: str, limit: int, context_lines: int) -> str:
    metadata = _load_metadata(corpus_id)
    records = _load_records(corpus_id)
    if not query or not query.strip():
        return _error("search requires query")
    limit = _coerce_int("limit", limit, 10, 1, _MAX_SEARCH_LIMIT)
    context_lines = _coerce_int("context_lines", context_lines, 0, 0, _MAX_CONTEXT_LINES)
    query_l = query.lower().strip()
    terms = _query_terms(query)
    matches = []
    for idx, record in enumerate(records):
        line = record.get("text", "")
        lower = line.lower()
        phrase_hit = query_l in lower
        term_hits = sum(1 for term in terms if term in lower)
        all_terms_hit = bool(terms) and term_hits == len(terms)
        if not phrase_hit and not all_terms_hit:
            continue
        start = max(0, idx - context_lines)
        end = min(len(records), idx + context_lines + 1)
        snippet = "\n".join(f"{records[i]['line']}: {records[i].get('text', '')}" for i in range(start, end))
        match: Dict[str, Any] = {
            "line": int(record["line"]),
            "text": line,
            "score": (100 if phrase_hit else 0) + term_hits,
            "snippet": snippet,
            "context": [_shape_line(records[i]) for i in range(start, end)],
        }
        if record.get("source") is not None:
            match["source"] = record["source"]
            match["source_line"] = record.get("source_line")
        matches.append(match)
    matches.sort(key=lambda m: (-m["score"], m["line"]))
    return _json({
        "success": True,
        "corpus_id": corpus_id,
        "name": metadata["name"],
        "query": query,
        "matches": matches[:limit],
        "count": min(len(matches), limit),
        "total_matches": len(matches),
    })


def _map(corpus_id: str, task: str, max_chunks: int) -> str:
    metadata = _load_metadata(corpus_id)
    if not task or not task.strip():
        return _error("map requires task")
    max_chunks = _coerce_int("max_chunks", max_chunks, 12, 1, _MAX_MAP_CHUNKS)
    ranges = _chunk_ranges(metadata["line_count"], metadata["chunk_lines"])[:max_chunks]
    tasks = []
    for n, line_range in enumerate(ranges, start=1):
        line_count = line_range["end_line"] - line_range["start_line"] + 1
        prompt = (
            "Use this exact tool call first: "
            f"recursive_context(action=\"read\", corpus_id=\"{corpus_id}\", "
            f"start_line={line_range['start_line']}, line_count={line_count}).\n"
            f"Task: {task.strip()}\n"
            "Return concise findings with citations to corpus line and, when present, "
            "source/source_line. Do not infer from lines outside this range unless "
            "explicitly instructed."
        )
        tasks.append({"chunk": n, "range": line_range, "prompt": prompt})
    return _json({
        "success": True,
        "corpus_id": corpus_id,
        "name": metadata["name"],
        "task": task,
        "tasks": tasks,
        "truncated": len(ranges) < metadata["chunk_count"],
    })


def _delete(corpus_id: str) -> str:
    cdir = _corpus_dir(corpus_id)
    if not cdir.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_id}")
    deleted_id = corpus_id
    try:
        metadata = _load_metadata(corpus_id)
        deleted_id = metadata.get("corpus_id", corpus_id)
    except (FileNotFoundError, OSError, json.JSONDecodeError, KeyError, ValueError):
        # Deletion is the recovery path for broken on-disk corpora; do not make
        # corrupt metadata undeletable.
        pass
    shutil.rmtree(cdir)
    return _json({"success": True, "deleted": deleted_id})


def recursive_context(
    action: str,
    corpus_id: Optional[str] = None,
    name: str = "corpus",
    text: Optional[str] = None,
    paths: Optional[List[str]] = None,
    query: str = "",
    start_line: int = 1,
    line_count: int = 80,
    chunk_lines: int = _DEFAULT_CHUNK_LINES,
    limit: int = 10,
    context_lines: int = 0,
    task: str = "",
    max_chunks: int = 12,
) -> str:
    """Navigate large external corpora with bounded create/list/search/read/map/delete actions."""
    try:
        action = (action or "").strip().lower()
        if action == "create":
            return _create(name=name, text=text, paths=paths, chunk_lines=chunk_lines)
        if action == "list":
            return _list()
        if action in {"read", "search", "map", "delete"}:
            if not corpus_id:
                return _error(f"{action} requires corpus_id")
            resolved_corpus_id = corpus_id
        else:
            resolved_corpus_id = ""
        if action == "read":
            return _read(resolved_corpus_id, start_line=start_line, line_count=line_count)
        if action == "search":
            return _search(resolved_corpus_id, query=query, limit=limit, context_lines=context_lines)
        if action == "map":
            return _map(resolved_corpus_id, task=task, max_chunks=max_chunks)
        if action == "delete":
            return _delete(resolved_corpus_id)
        return _error("action must be one of: create, list, search, read, map, delete")
    except Exception as e:
        return _error(str(e))


RECURSIVE_CONTEXT_SCHEMA = {
    "name": "recursive_context",
    "description": (
        "RLM-style large-context substrate. Create an external corpus from text/files, "
        "then search/read bounded source-line windows or map chunks into delegation "
        "prompts. Use this instead of pasting massive documents into context."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "list", "search", "read", "map", "delete"],
                "description": "Operation to perform.",
            },
            "corpus_id": {"type": "string", "description": "Corpus id returned by create/list."},
            "name": {"type": "string", "description": "Human-readable corpus name for create."},
            "text": {"type": "string", "description": "Raw text to externalize for create."},
            "paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Text file paths to ingest for create. Directories are rejected. Read/search results preserve source and source_line citations.",
            },
            "query": {"type": "string", "description": "Search query for action=search. All query terms must match unless the full phrase matches. Results include matched text plus structured context with source/source_line citations when available."},
            "start_line": {"type": "integer", "description": "1-indexed start line for action=read. Invalid integers return parameter-named errors.", "default": 1},
            "line_count": {"type": "integer", "description": "Lines to return for action=read; capped at 500.", "default": 80},
            "chunk_lines": {"type": "integer", "description": "Lines per chunk for create/map.", "default": _DEFAULT_CHUNK_LINES},
            "limit": {"type": "integer", "description": "Max search matches; capped at 50.", "default": 10},
            "context_lines": {"type": "integer", "description": "Neighbor lines around search hits; capped at 5. Search results include structured context lines, not only a text snippet.", "default": 0},
            "task": {"type": "string", "description": "Task instruction for action=map delegation prompts."},
            "max_chunks": {"type": "integer", "description": "Max chunk prompts for action=map; capped at 100.", "default": 12},
        },
        "required": ["action"],
    },
}


registry.register(
    name="recursive_context",
    toolset="file",
    schema=RECURSIVE_CONTEXT_SCHEMA,
    handler=lambda args, **kw: recursive_context(
        action=args.get("action", ""),
        corpus_id=args.get("corpus_id"),
        name=args.get("name", "corpus"),
        text=args.get("text"),
        paths=args.get("paths"),
        query=args.get("query", ""),
        start_line=args.get("start_line", 1),
        line_count=args.get("line_count", 80),
        chunk_lines=args.get("chunk_lines", _DEFAULT_CHUNK_LINES),
        limit=args.get("limit", 10),
        context_lines=args.get("context_lines", 0),
        task=args.get("task", ""),
        max_chunks=args.get("max_chunks", 12),
    ),
    check_fn=lambda: True,
    emoji="🧭",
)
