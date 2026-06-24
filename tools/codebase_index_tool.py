#!/usr/bin/env python3
"""Repo-aware semantic codebase index and hybrid search tools.

The lower-level ``semantic_index`` tool proves LanceDB/OpenAI plumbing for a
single payload.  This module is the agent-facing layer: discover code files,
chunk them with stable metadata, embed changed chunks into LanceDB, and search
with a keyword/semantic hybrid that degrades cleanly when embeddings are not
available.
"""

from __future__ import annotations

import ast
import fnmatch
import hashlib
import json
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from hermes_constants import get_hermes_home
from tools.semantic_index_tool import (
    DEFAULT_MODEL,
    _content_hash,
    _embedding_dimensions,
    _embed_openai,
    _import_lancedb,
    _preview,
    _secret_warning,
    _table_names,
)

logger = logging.getLogger(__name__)

CODEBASE_CORPUS = "codebase"
DEFAULT_TABLE = "codebase_chunks"
CHUNKER_VERSION = "codebase-v1"
DEFAULT_MAX_FILE_BYTES = 200_000
DEFAULT_MAX_CHUNK_CHARS = 6_000
DEFAULT_INDEX_CHUNKS = 500
DEFAULT_SEARCH_CHUNKS = 5_000
EMBED_BATCH_SIZE = 64

_EXCLUDE_DIRS = {
    ".git",
    ".hermes",
    ".hg",
    ".svn",
    ".cache",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    "coverage",
    ".next",
    ".turbo",
}

_LANGUAGE_BY_EXT = {
    ".py": "python",
    ".md": "markdown",
    ".rst": "rst",
    ".txt": "text",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".jsonl": "jsonl",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".css": "css",
    ".html": "html",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".sql": "sql",
}

_SPECIAL_TEXT_NAMES = {
    "AGENTS.md",
    "CLAUDE.md",
    "README",
    "LICENSE",
    "NOTICE",
    "Makefile",
    "Dockerfile",
}


@dataclass(frozen=True)
class CodeChunk:
    path: str
    language: str
    chunk_kind: str
    symbol: str
    start_line: int
    end_line: int
    text: str

    @property
    def content_hash(self) -> str:
        return _content_hash(self.text)


def _json(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False)


def _load_config() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        return load_config() or {}
    except Exception:
        logger.debug("codebase index config load failed", exc_info=True)
        return {}


def _codebase_config() -> Dict[str, Any]:
    return dict(_load_config().get("codebase_index") or {})


def _semantic_config() -> Dict[str, Any]:
    return dict(_load_config().get("semantic_index") or {})


def _db_path(path_override: Optional[str] = None) -> Path:
    if path_override and str(path_override).strip():
        return Path(path_override).expanduser()
    cfg = _codebase_config()
    cfg_path = str(cfg.get("path") or "").strip()
    if cfg_path:
        return Path(cfg_path).expanduser()
    semantic_path = str(_semantic_config().get("path") or "").strip()
    if semantic_path:
        return Path(semantic_path).expanduser()
    return get_hermes_home() / "semantic" / "lancedb"


def _table_name(table: Optional[str] = None) -> str:
    raw = str(table or _codebase_config().get("table") or DEFAULT_TABLE).strip()
    if not raw:
        return DEFAULT_TABLE
    return re.sub(r"[^A-Za-z0-9_]", "_", raw)[:64] or DEFAULT_TABLE


def _embedding_model(model: Optional[str] = None) -> str:
    raw = str(
        model
        or _codebase_config().get("model")
        or _semantic_config().get("model")
        or DEFAULT_MODEL
    ).strip()
    return raw or DEFAULT_MODEL


def _embedding_dims(dimensions: Optional[int] = None) -> Optional[int]:
    if dimensions is not None:
        return _embedding_dimensions(dimensions)
    cfg = _codebase_config()
    if cfg.get("dimensions") not in (None, "", 0):
        return _embedding_dimensions(cfg.get("dimensions"))
    return _embedding_dimensions(_semantic_config().get("dimensions"))


def _configured_int(name: str, default: int) -> int:
    try:
        value = int(_codebase_config().get(name) or default)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _run_git(root: Path, args: Sequence[str]) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(root),
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _repo_root(root: Optional[str] = None) -> Path:
    start = Path(root or os.getcwd()).expanduser().resolve()
    if start.is_file():
        start = start.parent
    git_root = _run_git(start, ["rev-parse", "--show-toplevel"])
    if git_root:
        return Path(git_root).resolve()
    return start


def _repo_metadata(root: Path) -> Dict[str, str]:
    commit = _run_git(root, ["rev-parse", "HEAD"]) or ""
    branch = _run_git(root, ["branch", "--show-current"]) or ""
    repo_id = hashlib.sha256(str(root).encode("utf-8", errors="replace")).hexdigest()[:16]
    return {
        "repo_root": str(root),
        "repo_id": repo_id,
        "git_commit": commit,
        "git_branch": branch,
    }


def _safe_relpath(root: Path, value: str) -> Optional[str]:
    candidate = (root / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
    try:
        rel = candidate.relative_to(root)
    except ValueError:
        return None
    return rel.as_posix()


def _walk_files(root: Path) -> List[str]:
    paths: List[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError:
            continue
        if any(part in _EXCLUDE_DIRS for part in rel.split("/")):
            continue
        paths.append(rel)
    return sorted(paths)


def _git_files(root: Path, scope: str) -> Optional[List[str]]:
    if scope == "changed":
        tracked = _run_git(root, ["diff", "--name-only", "HEAD"])
        staged = _run_git(root, ["diff", "--name-only", "--cached"])
        untracked = _run_git(root, ["ls-files", "--others", "--exclude-standard"])
        if tracked is None and staged is None and untracked is None:
            return None
        names = set()
        for text in (tracked, staged, untracked):
            if text:
                names.update(line.strip() for line in text.splitlines() if line.strip())
        return sorted(names)

    tracked = _run_git(root, ["ls-files", "--cached", "--others", "--exclude-standard"])
    if tracked is None:
        return None
    return sorted(line.strip() for line in tracked.splitlines() if line.strip())


def _candidate_paths(
    *,
    root: Path,
    scope: str,
    paths: Optional[Sequence[str]],
) -> Tuple[List[str], List[str]]:
    skipped: List[str] = []
    if scope == "paths":
        resolved = []
        seen = set()
        for raw in paths or []:
            rel = _safe_relpath(root, str(raw))
            if rel is None:
                skipped.append(f"{raw}: outside repo root")
            elif rel not in seen:
                seen.add(rel)
                resolved.append(rel)
        return resolved, skipped

    files = _git_files(root, scope)
    if files is None:
        files = _walk_files(root)
    return sorted(set(files)), skipped


def _matches_any(path: str, patterns: Optional[Sequence[str]]) -> bool:
    return bool(patterns) and any(fnmatch.fnmatch(path, pattern) for pattern in patterns or [])


def _looks_textual(data: bytes) -> bool:
    if b"\x00" in data:
        return False
    if not data:
        return True
    sample = data[:4096]
    control = sum(1 for b in sample if b < 32 and b not in (9, 10, 12, 13))
    return control / max(1, len(sample)) < 0.10


def _read_text_file(path: Path, max_file_bytes: int) -> Tuple[Optional[str], Optional[str]]:
    try:
        stat = path.stat()
    except OSError as exc:
        return None, f"stat failed: {exc}"
    if stat.st_size > max_file_bytes:
        return None, f"too large: {stat.st_size} bytes > {max_file_bytes}"
    try:
        data = path.read_bytes()
    except OSError as exc:
        return None, f"read failed: {exc}"
    if not _looks_textual(data):
        return None, "binary-looking content"
    return data.decode("utf-8", errors="replace"), None


def _language_for_path(path: str) -> str:
    rel = Path(path)
    if rel.name in _SPECIAL_TEXT_NAMES:
        if rel.suffix == ".md":
            return "markdown"
        return "text"
    return _LANGUAGE_BY_EXT.get(rel.suffix.lower(), "text")


def _is_indexable_path(
    *,
    rel_path: str,
    abs_path: Path,
    include_globs: Optional[Sequence[str]],
    exclude_globs: Optional[Sequence[str]],
) -> Tuple[bool, Optional[str]]:
    parts = rel_path.split("/")
    if any(part in _EXCLUDE_DIRS for part in parts):
        return False, "excluded directory"
    if include_globs and not _matches_any(rel_path, include_globs):
        return False, "does not match include_globs"
    if _matches_any(rel_path, exclude_globs):
        return False, "matches exclude_globs"
    if not abs_path.exists() or not abs_path.is_file():
        return False, "not a file"
    return True, None


def _line_slice(lines: List[str], start_line: int, end_line: int) -> str:
    return "\n".join(lines[start_line - 1 : end_line]).strip()


def _split_by_chars(
    *,
    rel_path: str,
    language: str,
    chunk_kind: str,
    symbol: str,
    start_line: int,
    text: str,
    max_chunk_chars: int,
) -> List[CodeChunk]:
    lines = text.splitlines()
    chunks: List[CodeChunk] = []
    current: List[str] = []
    current_start = start_line
    current_len = 0
    for idx, line in enumerate(lines, start=start_line):
        extra = len(line) + 1
        if current and current_len + extra > max_chunk_chars:
            chunks.append(
                CodeChunk(
                    path=rel_path,
                    language=language,
                    chunk_kind=chunk_kind,
                    symbol=symbol,
                    start_line=current_start,
                    end_line=idx - 1,
                    text="\n".join(current).strip(),
                )
            )
            current = []
            current_start = idx
            current_len = 0
        current.append(line)
        current_len += extra
    if current:
        chunks.append(
            CodeChunk(
                path=rel_path,
                language=language,
                chunk_kind=chunk_kind,
                symbol=symbol,
                start_line=current_start,
                end_line=start_line + len(lines) - 1,
                text="\n".join(current).strip(),
            )
        )
    return [chunk for chunk in chunks if chunk.text]


def _node_start(node: ast.AST) -> int:
    decorators = getattr(node, "decorator_list", None) or []
    lines = [getattr(node, "lineno", 1)]
    lines.extend(getattr(item, "lineno", lines[0]) for item in decorators)
    return max(1, min(lines))


def _python_chunks(rel_path: str, text: str, max_chunk_chars: int) -> List[CodeChunk]:
    lines = text.splitlines()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return _fallback_chunks(rel_path, "python", text, max_chunk_chars)

    chunks: List[CodeChunk] = []
    leading_end = 0
    for node in tree.body:
        is_doc = (
            isinstance(node, ast.Expr)
            and isinstance(getattr(node, "value", None), ast.Constant)
            and isinstance(getattr(node.value, "value", None), str)
        )
        if is_doc or isinstance(node, (ast.Import, ast.ImportFrom)):
            leading_end = max(leading_end, getattr(node, "end_lineno", getattr(node, "lineno", 1)))
            continue
        break
    if leading_end:
        module_text = _line_slice(lines, 1, leading_end)
        if module_text:
            chunks.extend(
                _split_by_chars(
                    rel_path=rel_path,
                    language="python",
                    chunk_kind="module_context",
                    symbol=Path(rel_path).stem,
                    start_line=1,
                    text=module_text,
                    max_chunk_chars=max_chunk_chars,
                )
            )

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = _node_start(node)
            end = getattr(node, "end_lineno", start)
            chunks.extend(
                _split_by_chars(
                    rel_path=rel_path,
                    language="python",
                    chunk_kind="function",
                    symbol=node.name,
                    start_line=start,
                    text=_line_slice(lines, start, end),
                    max_chunk_chars=max_chunk_chars,
                )
            )
            continue
        if isinstance(node, ast.ClassDef):
            start = _node_start(node)
            end = getattr(node, "end_lineno", start)
            class_text = _line_slice(lines, start, end)
            if len(class_text) <= max_chunk_chars:
                chunks.append(
                    CodeChunk(rel_path, "python", "class", node.name, start, end, class_text)
                )
            else:
                header_end = start
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        break
                    header_end = max(header_end, getattr(child, "end_lineno", header_end))
                header_text = _line_slice(lines, start, header_end)
                if header_text:
                    chunks.append(
                        CodeChunk(
                            rel_path,
                            "python",
                            "class_context",
                            node.name,
                            start,
                            header_end,
                            header_text,
                        )
                    )
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_start = _node_start(child)
                    method_end = getattr(child, "end_lineno", method_start)
                    chunks.extend(
                        _split_by_chars(
                            rel_path=rel_path,
                            language="python",
                            chunk_kind="method",
                            symbol=f"{node.name}.{child.name}",
                            start_line=method_start,
                            text=_line_slice(lines, method_start, method_end),
                            max_chunk_chars=max_chunk_chars,
                        )
                    )

    if not chunks:
        return _fallback_chunks(rel_path, "python", text, max_chunk_chars)
    return chunks


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def _markdown_chunks(rel_path: str, text: str, max_chunk_chars: int) -> List[CodeChunk]:
    lines = text.splitlines()
    starts: List[Tuple[int, str]] = []
    for idx, line in enumerate(lines, start=1):
        match = _HEADING_RE.match(line)
        if match:
            starts.append((idx, match.group(2).strip()))
    if not starts:
        return _fallback_chunks(rel_path, "markdown", text, max_chunk_chars)

    chunks: List[CodeChunk] = []
    if starts[0][0] > 1:
        preface = _line_slice(lines, 1, starts[0][0] - 1)
        chunks.extend(
            _split_by_chars(
                rel_path=rel_path,
                language="markdown",
                chunk_kind="doc_preface",
                symbol=Path(rel_path).name,
                start_line=1,
                text=preface,
                max_chunk_chars=max_chunk_chars,
            )
        )
    for index, (start, title) in enumerate(starts):
        end = starts[index + 1][0] - 1 if index + 1 < len(starts) else len(lines)
        section = _line_slice(lines, start, end)
        chunks.extend(
            _split_by_chars(
                rel_path=rel_path,
                language="markdown",
                chunk_kind="doc_section",
                symbol=title[:160],
                start_line=start,
                text=section,
                max_chunk_chars=max_chunk_chars,
            )
        )
    return chunks


def _fallback_chunks(rel_path: str, language: str, text: str, max_chunk_chars: int) -> List[CodeChunk]:
    return _split_by_chars(
        rel_path=rel_path,
        language=language,
        chunk_kind="file_chunk",
        symbol=Path(rel_path).name,
        start_line=1,
        text=text,
        max_chunk_chars=max_chunk_chars,
    )


def _chunk_file(rel_path: str, text: str, max_chunk_chars: int) -> List[CodeChunk]:
    language = _language_for_path(rel_path)
    if language == "python":
        return _python_chunks(rel_path, text, max_chunk_chars)
    if language == "markdown":
        return _markdown_chunks(rel_path, text, max_chunk_chars)
    return _fallback_chunks(rel_path, language, text, max_chunk_chars)


def _chunk_id(repo_id: str, chunk: CodeChunk) -> str:
    raw = (
        f"{repo_id}\n{chunk.path}\n{chunk.chunk_kind}\n{chunk.symbol}\n"
        f"{chunk.start_line}:{chunk.end_line}\n{chunk.content_hash}"
    )
    return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:32]


def _uri(root: Path, chunk: CodeChunk) -> str:
    return f"file://{root / chunk.path}#L{chunk.start_line}-L{chunk.end_line}"


def _record_for_chunk(
    *,
    root: Path,
    repo: Dict[str, str],
    chunk: CodeChunk,
    vector: Sequence[float],
    model: str,
    provider: str = "openai",
) -> Dict[str, Any]:
    return {
        "id": _chunk_id(repo["repo_id"], chunk),
        "corpus": CODEBASE_CORPUS,
        "source_type": "codebase_chunk",
        "repo_root": repo["repo_root"],
        "repo_id": repo["repo_id"],
        "git_commit": repo["git_commit"],
        "git_branch": repo["git_branch"],
        "path": chunk.path,
        "language": chunk.language,
        "chunk_kind": chunk.chunk_kind,
        "symbol": chunk.symbol,
        "start_line": int(chunk.start_line),
        "end_line": int(chunk.end_line),
        "uri": _uri(root, chunk),
        "content_hash": chunk.content_hash,
        "embedding_provider": provider,
        "embedding_model": model,
        "dimensions": len(vector),
        "text": chunk.text,
        "metadata_json": json.dumps(
            {"chunker_version": CHUNKER_VERSION},
            sort_keys=True,
            ensure_ascii=False,
        ),
        "vector": [float(v) for v in vector],
        "indexed_at": time.time(),
    }


def _sanitize_row(row: Dict[str, Any], include_text: bool = False) -> Dict[str, Any]:
    data = dict(row)
    vector = data.pop("vector", None)
    text = str(data.get("text") or "")
    if not include_text:
        data.pop("text", None)
    data["text_preview"] = _preview(text, limit=500)
    data["vector_dimensions"] = len(vector) if hasattr(vector, "__len__") else data.get("dimensions")
    return data


def _collect_chunks(
    *,
    root: Path,
    scope: str,
    paths: Optional[Sequence[str]],
    include_globs: Optional[Sequence[str]],
    exclude_globs: Optional[Sequence[str]],
    max_files: int,
    max_chunks: int,
    max_file_bytes: int,
    max_chunk_chars: int,
) -> Dict[str, Any]:
    candidates, skipped = _candidate_paths(root=root, scope=scope, paths=paths)
    files_seen = 0
    chunks: List[CodeChunk] = []
    skipped_files: List[str] = list(skipped)
    for rel_path in candidates:
        abs_path = root / rel_path
        ok, reason = _is_indexable_path(
            rel_path=rel_path,
            abs_path=abs_path,
            include_globs=include_globs,
            exclude_globs=exclude_globs,
        )
        if not ok:
            skipped_files.append(f"{rel_path}: {reason}")
            continue
        if files_seen >= max_files:
            skipped_files.append(f"{rel_path}: max_files reached")
            continue
        text, read_error = _read_text_file(abs_path, max_file_bytes)
        if read_error:
            skipped_files.append(f"{rel_path}: {read_error}")
            continue
        files_seen += 1
        file_chunks = _chunk_file(rel_path, text or "", max_chunk_chars)
        for chunk in file_chunks:
            if len(chunks) >= max_chunks:
                skipped_files.append(f"{rel_path}: max_chunks reached")
                break
            chunks.append(chunk)
    return {
        "chunks": chunks,
        "candidate_paths": candidates,
        "candidate_file_count": len(candidates),
        "indexed_file_count": files_seen,
        "skipped_files": skipped_files,
        "truncated": files_seen >= max_files or len(chunks) >= max_chunks,
    }


def _open_table(db: Any, table_name: str) -> Any:
    if table_name not in _table_names(db):
        return None
    return db.open_table(table_name)


def _sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _delete_stale_rows(table: Any, repo_id: str, paths: Iterable[str]) -> List[str]:
    warnings: List[str] = []
    for path in sorted(set(paths)):
        predicate = (
            f"corpus = {_sql_quote(CODEBASE_CORPUS)} AND "
            f"repo_id = {_sql_quote(repo_id)} AND path = {_sql_quote(path)}"
        )
        try:
            table.delete(predicate)
        except Exception as exc:
            warnings.append(f"{path}: stale-row delete failed: {exc}")
    return warnings


def _embed_openai_batch(
    texts: Sequence[str],
    *,
    model: str,
    dimensions: Optional[int],
    base_url: Optional[str],
    api_key: Optional[str],
    timeout: int,
    openai_client_cls: Any = None,
) -> Dict[str, Any]:
    if not texts:
        return {"vectors": [], "usage": {"prompt_tokens": 0, "total_tokens": 0}}
    key = (api_key or os.getenv("OPENAI_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required for codebase indexing")

    if openai_client_cls is None:
        from openai import OpenAI

        openai_client_cls = OpenAI

    client = openai_client_cls(
        api_key=key,
        base_url=(base_url or os.getenv("OPENAI_BASE_URL") or None),
        timeout=timeout,
        max_retries=0,
    )
    vectors: List[List[float]] = []
    usage = {"prompt_tokens": 0, "total_tokens": 0}
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = list(texts[start : start + EMBED_BATCH_SIZE])
        kwargs: Dict[str, Any] = {"model": model, "input": batch}
        if dimensions:
            kwargs["dimensions"] = dimensions
        response = client.embeddings.create(**kwargs)
        for item in sorted(response.data, key=lambda data: data.index):
            vectors.append([float(v) for v in item.embedding])
        response_usage = getattr(response, "usage", None)
        usage["prompt_tokens"] += int(getattr(response_usage, "prompt_tokens", 0) or 0)
        usage["total_tokens"] += int(getattr(response_usage, "total_tokens", 0) or 0)
    return {"vectors": vectors, "usage": usage}


def _index_chunks(
    *,
    root: Path,
    repo: Dict[str, str],
    chunks: Sequence[CodeChunk],
    stale_paths: Sequence[str],
    table_name: str,
    db_path: Path,
    model: str,
    dimensions: Optional[int],
    base_url: Optional[str],
    api_key: Optional[str],
    timeout: int,
    lancedb_module: Any = None,
    openai_client_cls: Any = None,
) -> Dict[str, Any]:
    safe_chunks = []
    skipped_secret = []
    for chunk in chunks:
        warning = _secret_warning(chunk.text)
        if warning:
            skipped_secret.append(chunk.path)
            continue
        safe_chunks.append(chunk)

    embedded = _embed_openai_batch(
        [chunk.text for chunk in safe_chunks],
        model=model,
        dimensions=dimensions,
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        openai_client_cls=openai_client_cls,
    )
    vectors = embedded["vectors"]
    if len(vectors) != len(safe_chunks):
        raise RuntimeError(f"embedding count mismatch: got {len(vectors)} for {len(safe_chunks)} chunks")

    records = [
        _record_for_chunk(root=root, repo=repo, chunk=chunk, vector=vector, model=model)
        for chunk, vector in zip(safe_chunks, vectors)
    ]
    lancedb_module = lancedb_module or _import_lancedb()
    db_path.mkdir(parents=True, exist_ok=True)
    db = lancedb_module.connect(str(db_path))
    table = _open_table(db, table_name)
    delete_warnings: List[str] = []
    if table is None:
        table = db.create_table(table_name, data=records) if records else None
    else:
        delete_warnings = _delete_stale_rows(table, repo["repo_id"], stale_paths)
        if records:
            table.add(records)

    row_count = int(table.count_rows()) if table is not None else 0
    return {
        "records": records,
        "row_count": row_count,
        "usage": embedded["usage"],
        "skipped_secret_count": len(skipped_secret),
        "delete_warnings": delete_warnings,
    }


def _chunk_summary(repo: Dict[str, str], root: Path, chunk: CodeChunk) -> Dict[str, Any]:
    return {
        "id": _chunk_id(repo["repo_id"], chunk),
        "path": chunk.path,
        "language": chunk.language,
        "chunk_kind": chunk.chunk_kind,
        "symbol": chunk.symbol,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "char_count": len(chunk.text),
        "content_hash": chunk.content_hash,
        "uri": _uri(root, chunk),
        "text_preview": _preview(chunk.text, limit=320),
    }


def codebase_index(
    action: str = "dry_run",
    root: str = None,
    scope: str = "repo",
    paths: List[str] = None,
    include_globs: List[str] = None,
    exclude_globs: List[str] = None,
    table: str = None,
    db_path: str = None,
    model: str = None,
    dimensions: int = None,
    max_files: int = 200,
    max_chunks: int = DEFAULT_INDEX_CHUNKS,
    max_file_bytes: int = None,
    max_chunk_chars: int = None,
    embed: bool = False,
    base_url: str = None,
    api_key: str = None,
    timeout: int = 60,
    lancedb_module: Any = None,
    openai_client_cls: Any = None,
) -> str:
    """Dry-run, index, or inspect a repository-level codebase vector table."""
    action_norm = str(action or "dry_run").strip().lower()
    if action_norm in {"dryrun", "preview"}:
        action_norm = "dry_run"
    if action_norm == "inspect":
        action_norm = "stats"
    scope_norm = str(scope or "repo").strip().lower()
    if scope_norm not in {"repo", "changed", "paths"}:
        return _json({"success": False, "error": "scope must be one of: repo, changed, paths"})

    repo_root = _repo_root(root)
    repo = _repo_metadata(repo_root)
    table_name = _table_name(table)
    path = _db_path(db_path)
    model_name = _embedding_model(model)
    dims = _embedding_dims(dimensions)

    if action_norm == "stats":
        return _json(_stats(table_name, path, lancedb_module=lancedb_module))

    max_files_int = max(1, int(max_files or 200))
    max_chunks_int = max(1, int(max_chunks or DEFAULT_INDEX_CHUNKS))
    max_file_bytes_int = int(max_file_bytes or _configured_int("max_file_bytes", DEFAULT_MAX_FILE_BYTES))
    max_chunk_chars_int = int(max_chunk_chars or _configured_int("max_chunk_chars", DEFAULT_MAX_CHUNK_CHARS))

    collected = _collect_chunks(
        root=repo_root,
        scope=scope_norm,
        paths=paths,
        include_globs=include_globs,
        exclude_globs=exclude_globs,
        max_files=max_files_int,
        max_chunks=max_chunks_int,
        max_file_bytes=max_file_bytes_int,
        max_chunk_chars=max_chunk_chars_int,
    )
    chunks: List[CodeChunk] = collected["chunks"]
    sample_chunks = [_chunk_summary(repo, repo_root, chunk) for chunk in chunks[:5]]

    if action_norm == "dry_run":
        return _json(
            {
                "success": True,
                "action": "dry_run",
                "dry_run": True,
                "would_call_openai": False,
                "would_write_lancedb": False,
                "repo": repo,
                "scope": scope_norm,
                "db_path": str(path),
                "table": table_name,
                "model": model_name,
                "dimensions": dims,
                "candidate_file_count": collected["candidate_file_count"],
                "indexed_file_count": collected["indexed_file_count"],
                "chunk_count": len(chunks),
                "truncated": collected["truncated"],
                "skipped_files_sample": collected["skipped_files"][:20],
                "sample_chunks": sample_chunks,
                "next_step": (
                    "Call codebase_index(action='index', embed=true, scope=...) "
                    "to send bounded chunks to OpenAI and store vectors in LanceDB."
                ),
            }
        )

    if action_norm != "index":
        return _json({"success": False, "error": "Unknown action. Use dry_run, index, or stats."})
    if not embed:
        return _json(
            {
                "success": False,
                "error": "index requires embed=true so external OpenAI usage is explicit",
                "dry_run_hint": "Call codebase_index(action='dry_run', scope=...) first.",
            }
        )

    try:
        indexed = _index_chunks(
            root=repo_root,
            repo=repo,
            chunks=chunks,
            stale_paths=collected["candidate_paths"],
            table_name=table_name,
            db_path=path,
            model=model_name,
            dimensions=dims,
            base_url=base_url,
            api_key=api_key,
            timeout=int(timeout or 60),
            lancedb_module=lancedb_module,
            openai_client_cls=openai_client_cls,
        )
    except Exception as exc:
        return _json({"success": False, "action": "index", "error": str(exc)})

    return _json(
        {
            "success": True,
            "action": "index",
            "dry_run": False,
            "repo": repo,
            "scope": scope_norm,
            "db_path": str(path),
            "table": table_name,
            "model": model_name,
            "dimensions": indexed["records"][0]["dimensions"] if indexed["records"] else dims,
            "candidate_file_count": collected["candidate_file_count"],
            "indexed_file_count": collected["indexed_file_count"],
            "chunk_count": len(chunks),
            "stored_chunk_count": len(indexed["records"]),
            "row_count": indexed["row_count"],
            "usage": indexed["usage"],
            "truncated": collected["truncated"],
            "skipped_files_sample": collected["skipped_files"][:20],
            "skipped_secret_count": indexed["skipped_secret_count"],
            "delete_warnings": indexed["delete_warnings"][:20],
            "sample_chunks": sample_chunks,
        }
    )


def _query_tokens(query: str) -> List[str]:
    return [token for token in re.findall(r"[A-Za-z0-9_./-]+", query.lower()) if len(token) > 1]


def _lexical_score(query: str, tokens: Sequence[str], chunk: CodeChunk) -> float:
    haystack = f"{chunk.path}\n{chunk.symbol}\n{chunk.chunk_kind}\n{chunk.text}".lower()
    score = 0.0
    phrase = query.strip().lower()
    if phrase and phrase in haystack:
        score += 8.0
    for token in tokens:
        count = haystack.count(token)
        if count:
            score += min(5.0, float(count))
        if token in chunk.path.lower():
            score += 3.0
        if token in chunk.symbol.lower():
            score += 4.0
    return min(1.0, score / 24.0)


def _keyword_results(
    *,
    root: Path,
    repo: Dict[str, str],
    query: str,
    paths: Optional[Sequence[str]],
    include_globs: Optional[Sequence[str]],
    exclude_globs: Optional[Sequence[str]],
    max_files: int,
    max_chunks: int,
    max_file_bytes: int,
    max_chunk_chars: int,
) -> Dict[str, Any]:
    tokens = _query_tokens(query)
    if not tokens and not query.strip():
        return {"results": [], "scanned_chunk_count": 0}
    collected = _collect_chunks(
        root=root,
        scope="paths" if paths else "repo",
        paths=paths,
        include_globs=include_globs,
        exclude_globs=exclude_globs,
        max_files=max_files,
        max_chunks=max_chunks,
        max_file_bytes=max_file_bytes,
        max_chunk_chars=max_chunk_chars,
    )
    results = []
    for chunk in collected["chunks"]:
        score = _lexical_score(query, tokens, chunk)
        if score <= 0:
            continue
        summary = _chunk_summary(repo, root, chunk)
        summary.update({"score": score, "keyword_score": score, "match_source": "keyword"})
        results.append(summary)
    results.sort(key=lambda item: item["score"], reverse=True)
    return {"results": results, "scanned_chunk_count": len(collected["chunks"])}


def _semantic_results(
    *,
    root: Path,
    repo: Dict[str, str],
    query: str,
    table_name: str,
    db_path: Path,
    model: str,
    dimensions: Optional[int],
    limit: int,
    base_url: Optional[str],
    api_key: Optional[str],
    timeout: int,
    lancedb_module: Any = None,
    openai_client_cls: Any = None,
) -> Dict[str, Any]:
    lancedb_module = lancedb_module or _import_lancedb()
    db = lancedb_module.connect(str(db_path))
    table = _open_table(db, table_name)
    if table is None:
        raise RuntimeError(f"table '{table_name}' does not exist")

    embedded = _embed_openai(
        query,
        model=model,
        dimensions=dimensions,
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        openai_client_cls=openai_client_cls,
    )
    raw_rows = table.search(embedded["vector"]).limit(max(limit * 5, limit)).to_list()
    results = []
    for raw in raw_rows:
        row = dict(raw)
        if row.get("corpus") != CODEBASE_CORPUS or row.get("repo_id") != repo["repo_id"]:
            continue
        distance = float(row.get("_distance", 0.0) or 0.0)
        sanitized = _sanitize_row(row)
        sanitized.update(
            {
                "score": 1.0 / (1.0 + max(0.0, distance)),
                "semantic_score": 1.0 / (1.0 + max(0.0, distance)),
                "match_source": "semantic",
            }
        )
        results.append(sanitized)
        if len(results) >= limit:
            break
    return {"results": results, "usage": embedded["usage"]}


def _merge_hybrid(keyword: Sequence[Dict[str, Any]], semantic: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for item in keyword:
        merged[item["id"]] = dict(item)
    for item in semantic:
        existing = merged.get(item["id"], {})
        combined = {**item, **existing}
        combined["semantic_score"] = float(item.get("semantic_score", 0.0) or 0.0)
        combined["keyword_score"] = float(existing.get("keyword_score", 0.0) or 0.0)
        if combined["keyword_score"] and combined["semantic_score"]:
            combined["match_source"] = "hybrid"
            combined["score"] = 0.65 * combined["semantic_score"] + 0.35 * combined["keyword_score"]
        elif combined["semantic_score"]:
            combined["score"] = combined["semantic_score"]
            combined["match_source"] = "semantic"
        else:
            combined["score"] = combined["keyword_score"]
            combined["match_source"] = "keyword"
        merged[item["id"]] = combined
    return sorted(merged.values(), key=lambda item: item.get("score", 0.0), reverse=True)


def codebase_search(
    query: str,
    root: str = None,
    mode: str = "hybrid",
    paths: List[str] = None,
    include_globs: List[str] = None,
    exclude_globs: List[str] = None,
    table: str = None,
    db_path: str = None,
    model: str = None,
    dimensions: int = None,
    limit: int = 8,
    max_files: int = 1000,
    max_chunks: int = DEFAULT_SEARCH_CHUNKS,
    max_file_bytes: int = None,
    max_chunk_chars: int = None,
    include_text: bool = False,
    base_url: str = None,
    api_key: str = None,
    timeout: int = 30,
    lancedb_module: Any = None,
    openai_client_cls: Any = None,
) -> str:
    """Search a codebase with keyword, semantic, or hybrid retrieval."""
    if not str(query or "").strip():
        return _json({"success": False, "error": "query is required"})
    mode_norm = str(mode or "hybrid").strip().lower()
    if mode_norm not in {"keyword", "semantic", "hybrid"}:
        return _json({"success": False, "error": "mode must be keyword, semantic, or hybrid"})

    repo_root = _repo_root(root)
    repo = _repo_metadata(repo_root)
    table_name = _table_name(table)
    path = _db_path(db_path)
    model_name = _embedding_model(model)
    dims = _embedding_dims(dimensions)
    limit_int = max(1, min(int(limit or 8), 30))
    max_file_bytes_int = int(max_file_bytes or _configured_int("max_file_bytes", DEFAULT_MAX_FILE_BYTES))
    max_chunk_chars_int = int(max_chunk_chars or _configured_int("max_chunk_chars", DEFAULT_MAX_CHUNK_CHARS))

    keyword_payload = {"results": [], "scanned_chunk_count": 0}
    if mode_norm in {"keyword", "hybrid"}:
        keyword_payload = _keyword_results(
            root=repo_root,
            repo=repo,
            query=query,
            paths=paths,
            include_globs=include_globs,
            exclude_globs=exclude_globs,
            max_files=max(1, int(max_files or 1000)),
            max_chunks=max(1, int(max_chunks or DEFAULT_SEARCH_CHUNKS)),
            max_file_bytes=max_file_bytes_int,
            max_chunk_chars=max_chunk_chars_int,
        )

    semantic_payload = {"results": [], "usage": None}
    semantic_error = None
    if mode_norm in {"semantic", "hybrid"}:
        try:
            semantic_payload = _semantic_results(
                root=repo_root,
                repo=repo,
                query=query,
                table_name=table_name,
                db_path=path,
                model=model_name,
                dimensions=dims,
                limit=limit_int,
                base_url=base_url,
                api_key=api_key,
                timeout=int(timeout or 30),
                lancedb_module=lancedb_module,
                openai_client_cls=openai_client_cls,
            )
        except Exception as exc:
            semantic_error = str(exc)
            if mode_norm == "semantic":
                return _json(
                    {
                        "success": False,
                        "action": "search",
                        "mode": mode_norm,
                        "repo": repo,
                        "db_path": str(path),
                        "table": table_name,
                        "error": semantic_error,
                    }
                )

    merged = _merge_hybrid(keyword_payload["results"], semantic_payload["results"])
    results = merged[:limit_int]
    if not include_text:
        for item in results:
            item.pop("text", None)

    return _json(
        {
            "success": True,
            "action": "search",
            "mode": mode_norm,
            "repo": repo,
            "db_path": str(path),
            "table": table_name,
            "model": model_name,
            "dimensions": dims,
            "count": len(results),
            "results": results,
            "semantic_available": semantic_error is None if mode_norm in {"semantic", "hybrid"} else False,
            "semantic_error": semantic_error,
            "keyword_scanned_chunk_count": keyword_payload["scanned_chunk_count"],
            "usage": semantic_payload.get("usage"),
        }
    )


def _stats(table_name: str, db_path: Path, lancedb_module: Any = None) -> Dict[str, Any]:
    try:
        lancedb_module = lancedb_module or _import_lancedb()
    except Exception as exc:
        return {
            "success": True,
            "action": "stats",
            "lancedb_available": False,
            "db_path": str(db_path),
            "table": table_name,
            "error": str(exc),
        }
    db = lancedb_module.connect(str(db_path))
    tables = _table_names(db)
    if table_name not in tables:
        return {
            "success": True,
            "action": "stats",
            "lancedb_available": True,
            "db_path": str(db_path),
            "tables": tables,
            "table": table_name,
            "table_exists": False,
            "row_count": 0,
        }
    table = db.open_table(table_name)
    sample = []
    raw = table.head(5) if hasattr(table, "head") else table.limit(5).to_list()
    rows = raw.to_pylist() if hasattr(raw, "to_pylist") else raw
    for row in rows:
        sample.append(_sanitize_row(dict(row)))
    return {
        "success": True,
        "action": "stats",
        "lancedb_available": True,
        "db_path": str(db_path),
        "tables": tables,
        "table": table_name,
        "table_exists": True,
        "row_count": int(table.count_rows()),
        "sample": sample,
    }


def check_codebase_index_requirements() -> bool:
    return get_hermes_home().exists()


CODEBASE_INDEX_SCHEMA = {
    "name": "codebase_index",
    "description": (
        "Dry-run, embed, and inspect a repository-scoped codebase index. "
        "Default dry_run never calls OpenAI or writes LanceDB. Use action='index' "
        "with embed=true after inspecting chunk counts."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["dry_run", "index", "stats"],
                "description": "dry_run previews chunks; index embeds+writes; stats inspects LanceDB.",
                "default": "dry_run",
            },
            "root": {"type": "string", "description": "Repo root or path inside a repo. Defaults to cwd."},
            "scope": {
                "type": "string",
                "enum": ["repo", "changed", "paths"],
                "description": "Repo-wide, git-changed/untracked, or explicit paths.",
                "default": "repo",
            },
            "paths": {"type": "array", "items": {"type": "string"}},
            "include_globs": {"type": "array", "items": {"type": "string"}},
            "exclude_globs": {"type": "array", "items": {"type": "string"}},
            "embed": {
                "type": "boolean",
                "description": "Required true for action='index' to make OpenAI usage explicit.",
                "default": False,
            },
            "table": {"type": "string", "description": "LanceDB table. Defaults to codebase_chunks."},
            "db_path": {"type": "string", "description": "Override LanceDB path."},
            "model": {"type": "string", "description": "OpenAI embedding model."},
            "dimensions": {"type": "integer", "description": "Optional embedding dimensions override."},
            "max_files": {"type": "integer", "default": 200},
            "max_chunks": {"type": "integer", "default": DEFAULT_INDEX_CHUNKS},
            "max_file_bytes": {"type": "integer", "default": DEFAULT_MAX_FILE_BYTES},
            "max_chunk_chars": {"type": "integer", "default": DEFAULT_MAX_CHUNK_CHARS},
        },
        "required": [],
    },
}


CODEBASE_SEARCH_SCHEMA = {
    "name": "codebase_search",
    "description": (
        "Search the current codebase with keyword, semantic, or hybrid retrieval. "
        "Hybrid uses LanceDB/OpenAI when available and falls back to keyword search "
        "without failing the agent turn."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural-language or symbol-oriented query."},
            "root": {"type": "string", "description": "Repo root or path inside a repo. Defaults to cwd."},
            "mode": {
                "type": "string",
                "enum": ["hybrid", "semantic", "keyword"],
                "description": "Retrieval mode. Hybrid is the default agent mode.",
                "default": "hybrid",
            },
            "paths": {"type": "array", "items": {"type": "string"}},
            "include_globs": {"type": "array", "items": {"type": "string"}},
            "exclude_globs": {"type": "array", "items": {"type": "string"}},
            "table": {"type": "string", "description": "LanceDB table. Defaults to codebase_chunks."},
            "db_path": {"type": "string", "description": "Override LanceDB path."},
            "model": {"type": "string", "description": "OpenAI embedding model."},
            "dimensions": {"type": "integer", "description": "Optional embedding dimensions override."},
            "limit": {"type": "integer", "description": "Max results, clamped to [1, 30].", "default": 8},
            "include_text": {
                "type": "boolean",
                "description": "Include full chunk text. Default false returns previews only.",
                "default": False,
            },
            "max_files": {"type": "integer", "default": 1000},
            "max_chunks": {"type": "integer", "default": DEFAULT_SEARCH_CHUNKS},
            "max_file_bytes": {"type": "integer", "default": DEFAULT_MAX_FILE_BYTES},
            "max_chunk_chars": {"type": "integer", "default": DEFAULT_MAX_CHUNK_CHARS},
        },
        "required": ["query"],
    },
}


from tools.registry import registry

registry.register(
    name="codebase_index",
    toolset="codebase_search",
    schema=CODEBASE_INDEX_SCHEMA,
    handler=lambda args, **kw: codebase_index(
        action=args.get("action", "dry_run"),
        root=args.get("root"),
        scope=args.get("scope", "repo"),
        paths=args.get("paths"),
        include_globs=args.get("include_globs"),
        exclude_globs=args.get("exclude_globs"),
        table=args.get("table"),
        db_path=args.get("db_path"),
        model=args.get("model"),
        dimensions=args.get("dimensions"),
        max_files=args.get("max_files", 200),
        max_chunks=args.get("max_chunks", DEFAULT_INDEX_CHUNKS),
        max_file_bytes=args.get("max_file_bytes"),
        max_chunk_chars=args.get("max_chunk_chars"),
        embed=args.get("embed", False),
        base_url=args.get("base_url"),
        api_key=args.get("api_key"),
        timeout=args.get("timeout", 60),
        lancedb_module=kw.get("lancedb_module"),
        openai_client_cls=kw.get("openai_client_cls"),
    ),
    check_fn=check_codebase_index_requirements,
    emoji="",
)

registry.register(
    name="codebase_search",
    toolset="codebase_search",
    schema=CODEBASE_SEARCH_SCHEMA,
    handler=lambda args, **kw: codebase_search(
        query=args.get("query", ""),
        root=args.get("root"),
        mode=args.get("mode", "hybrid"),
        paths=args.get("paths"),
        include_globs=args.get("include_globs"),
        exclude_globs=args.get("exclude_globs"),
        table=args.get("table"),
        db_path=args.get("db_path"),
        model=args.get("model"),
        dimensions=args.get("dimensions"),
        limit=args.get("limit", 8),
        max_files=args.get("max_files", 1000),
        max_chunks=args.get("max_chunks", DEFAULT_SEARCH_CHUNKS),
        max_file_bytes=args.get("max_file_bytes"),
        max_chunk_chars=args.get("max_chunk_chars"),
        include_text=args.get("include_text", False),
        base_url=args.get("base_url"),
        api_key=args.get("api_key"),
        timeout=args.get("timeout", 30),
        lancedb_module=kw.get("lancedb_module"),
        openai_client_cls=kw.get("openai_client_cls"),
    ),
    check_fn=check_codebase_index_requirements,
    emoji="",
)
