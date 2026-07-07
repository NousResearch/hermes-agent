from __future__ import annotations

import json
import re
from pathlib import Path

from tools.code_graph.indexer import (
    DEFAULT_MAX_FILE_SIZE_BYTES,
    _iter_source_files,
    _sha256,
)
from tools.code_graph.storage import CodeGraphStore, cache_path_for_root


_TERM_RE = re.compile(r"[A-Za-z0-9_]{3,}")


def _limit(value: int, default: int = 20, maximum: int = 100) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(1, min(parsed, maximum))


def _missing_status(store: CodeGraphStore) -> dict:
    return {
        "success": True,
        "graph_status": "missing",
        "stale_files": 0,
        "indexed_files": 0,
        "source_files_seen": 0,
        "skipped_files": 0,
        "cache_path": str(store.db_path),
    }


def _repo_id_for_query(root: Path, store: CodeGraphStore) -> int | None:
    if not store.db_path.exists():
        return None
    store.initialize()
    return store.get_repo_id(root)


def graph_status(
    root: Path,
    *,
    store: CodeGraphStore | None = None,
    max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
) -> dict:
    root = Path(root).resolve()
    store = store or CodeGraphStore(cache_path_for_root(root))
    repo_id = _repo_id_for_query(root, store)
    if repo_id is None:
        return _missing_status(store)

    stale_files = 0
    source_files_seen = 0
    skipped_files = 0
    stale_details: list[dict] = []
    current_by_path: dict[str, str] = {}
    with store.connect() as conn:
        indexed = conn.execute(
            "SELECT path, sha256 FROM files WHERE repo_id = ?",
            (repo_id,),
        ).fetchall()
        indexed_by_path = {row["path"]: row["sha256"] for row in indexed}
        if not indexed_by_path:
            return _missing_status(store)

        for path in _iter_source_files(root):
            stat = path.stat()
            if stat.st_size > max_file_size_bytes:
                skipped_files += 1
                continue
            source_files_seen += 1
            rel_path = path.relative_to(root).as_posix()
            digest = _sha256(path)
            current_by_path[rel_path] = digest
            if indexed_by_path.get(rel_path) != digest:
                stale_files += 1
                if len(stale_details) < 20:
                    stale_details.append({"path": rel_path, "reason": "changed_or_new"})

    deleted_paths = set(indexed_by_path) - set(current_by_path)
    stale_files += len(deleted_paths)
    for rel_path in sorted(deleted_paths)[: max(0, 20 - len(stale_details))]:
        stale_details.append({"path": rel_path, "reason": "deleted_or_skipped"})

    return {
        "success": True,
        "graph_status": "stale" if stale_files else "fresh",
        "stale_files": stale_files,
        "indexed_files": len(indexed_by_path),
        "source_files_seen": source_files_seen,
        "skipped_files": skipped_files,
        "stale_details": stale_details,
        "cache_path": str(store.db_path),
    }


def _compact_symbol(row) -> dict:
    data = dict(row)
    docstring = data.get("docstring")
    if docstring and len(docstring) > 300:
        data["docstring"] = docstring[:297] + "..."
    return {key: value for key, value in data.items() if value is not None}


def search_symbols(
    root: Path,
    query: str,
    *,
    store: CodeGraphStore | None = None,
    limit: int = 20,
) -> dict:
    root = Path(root).resolve()
    store = store or CodeGraphStore(cache_path_for_root(root))
    repo_id = _repo_id_for_query(root, store)
    status = graph_status(root, store=store)
    if repo_id is None:
        return {"success": True, "graph_status": "missing", "matches": []}

    limit = _limit(limit)
    query = str(query or "").strip()
    like = f"%{query}%"
    prefix = f"{query}%"
    with store.connect() as conn:
        rows = conn.execute(
            """
            SELECT s.id, s.name, s.qualname, s.kind, s.start_line, s.end_line,
                   s.signature, s.docstring, f.path
            FROM symbols s
            JOIN files f ON f.id = s.file_id
            WHERE s.repo_id = ? AND (? = '' OR s.name LIKE ? OR s.qualname LIKE ?)
            ORDER BY
                CASE
                    WHEN s.name = ? THEN 0
                    WHEN s.name LIKE ? THEN 1
                    ELSE 2
                END,
                LENGTH(s.qualname),
                s.name
            LIMIT ?
            """,
            (repo_id, query, like, like, query, prefix, limit),
        ).fetchall()
    return {
        "success": True,
        "graph_status": status["graph_status"],
        "matches": [_compact_symbol(row) for row in rows],
    }


def symbol_detail(
    root: Path,
    symbol: str,
    *,
    store: CodeGraphStore | None = None,
) -> dict:
    root = Path(root).resolve()
    store = store or CodeGraphStore(cache_path_for_root(root))
    repo_id = _repo_id_for_query(root, store)
    status = graph_status(root, store=store)
    if repo_id is None:
        return {
            "success": False,
            "graph_status": "missing",
            "error": f"symbol not found: {symbol}",
        }

    with store.connect() as conn:
        row = conn.execute(
            """
            SELECT s.id, s.file_id, s.name, s.qualname, s.kind, s.start_line,
                   s.end_line, s.signature, s.docstring, f.path
            FROM symbols s
            JOIN files f ON f.id = s.file_id
            WHERE s.repo_id = ? AND (s.name = ? OR s.qualname = ?)
            ORDER BY LENGTH(s.qualname), s.start_line
            LIMIT 1
            """,
            (repo_id, symbol, symbol),
        ).fetchone()
    if row is None:
        return {
            "success": False,
            "graph_status": status["graph_status"],
            "error": f"symbol not found: {symbol}",
        }
    return {
        "success": True,
        "graph_status": status["graph_status"],
        "symbol": _compact_symbol(row),
    }


def _parse_edge_metadata(raw: str) -> dict:
    try:
        data = json.loads(raw or "{}")
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def _chunk_references(conn, repo_id: int, name: str, limit: int) -> list[dict]:
    rows = conn.execute(
        """
        SELECT f.path, c.start_line, c.text
        FROM chunks c
        JOIN files f ON f.id = c.file_id
        WHERE c.repo_id = ? AND c.text LIKE ?
        ORDER BY f.path, c.start_line
        LIMIT ?
        """,
        (repo_id, f"%{name}%", limit * 4),
    ).fetchall()
    refs: list[dict] = []
    seen: set[tuple[str, int]] = set()
    needle = name.lower()
    for row in rows:
        for offset, line in enumerate(str(row["text"]).splitlines()):
            if needle not in line.lower():
                continue
            item = {
                "path": row["path"],
                "line": int(row["start_line"]) + offset,
                "kind": "text",
            }
            key = (item["path"], item["line"])
            if key in seen:
                continue
            seen.add(key)
            refs.append(item)
            if len(refs) >= limit:
                return refs
    return refs


def neighbors_for_symbol(
    root: Path,
    symbol: str,
    *,
    store: CodeGraphStore | None = None,
    limit: int = 50,
) -> dict:
    detail = symbol_detail(root, symbol, store=store)
    if not detail.get("success"):
        detail.setdefault("imports", [])
        detail.setdefault("references", [])
        detail.setdefault("calls", [])
        return detail

    root = Path(root).resolve()
    store = store or CodeGraphStore(cache_path_for_root(root))
    repo_id = _repo_id_for_query(root, store)
    if repo_id is None:
        return detail
    limit = _limit(limit, default=50, maximum=200)
    target = detail["symbol"]
    target_name = target["name"]

    imports: list[dict] = []
    references: list[dict] = []
    calls: list[dict] = []
    like = f'%"name": "{target_name}"%'
    with store.connect() as conn:
        import_rows = conn.execute(
            """
            SELECT e.metadata_json
            FROM edges e
            WHERE e.repo_id = ?
              AND e.source_kind = 'file'
              AND e.source_id = ?
              AND e.edge_type = 'imports'
            ORDER BY e.metadata_json
            LIMIT ?
            """,
            (repo_id, int(target["file_id"]), limit),
        ).fetchall()
        ref_rows = conn.execute(
            """
            SELECT e.edge_type, e.metadata_json, f.path
            FROM edges e
            JOIN files f ON f.id = e.source_id
            WHERE e.repo_id = ?
              AND e.source_kind = 'file'
              AND e.edge_type IN ('references', 'calls')
              AND e.metadata_json LIKE ?
            ORDER BY f.path, e.edge_type
            LIMIT ?
            """,
            (repo_id, like, limit * 4),
        ).fetchall()
        chunk_refs = _chunk_references(conn, repo_id, target_name, limit)

    for row in import_rows:
        data = _parse_edge_metadata(row["metadata_json"])
        if data.get("module"):
            imports.append({"module": data["module"]})

    seen_refs: set[tuple[str, int, str]] = set()
    for row in ref_rows:
        data = _parse_edge_metadata(row["metadata_json"])
        if data.get("name") != target_name:
            continue
        item = {
            "path": row["path"],
            "line": data.get("line"),
            "kind": data.get("kind") or row["edge_type"],
        }
        key = (item["path"], int(item["line"] or 0), row["edge_type"])
        if key in seen_refs:
            continue
        seen_refs.add(key)
        if row["edge_type"] == "calls":
            calls.append(item)
        else:
            references.append(item)
        if len(calls) + len(references) >= limit:
            break
    for item in chunk_refs:
        key = (item["path"], int(item["line"] or 0), "text")
        if key in seen_refs:
            continue
        seen_refs.add(key)
        references.append(item)
        if len(calls) + len(references) >= limit:
            break

    return {
        "success": True,
        "graph_status": detail["graph_status"],
        "symbol": target,
        "imports": imports,
        "references": references,
        "calls": calls,
    }


def _normalize_rel_path(root: Path, path_value: str) -> str:
    path = Path(str(path_value))
    if path.is_absolute():
        resolved = path.resolve()
    else:
        resolved = (root / path).resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"path is outside repository root: {path_value}") from exc


def impact_for_paths(
    root: Path,
    paths: list[str],
    *,
    store: CodeGraphStore | None = None,
    limit: int = 50,
) -> dict:
    root = Path(root).resolve()
    store = store or CodeGraphStore(cache_path_for_root(root))
    repo_id = _repo_id_for_query(root, store)
    status = graph_status(root, store=store)
    if repo_id is None:
        return {
            "success": True,
            "graph_status": "missing",
            "input_paths": [],
            "symbols": [],
            "imports": [],
            "referencing_files": [],
            "likely_tests": [],
        }

    limit = _limit(limit, default=50, maximum=200)
    norm_paths = [_normalize_rel_path(root, p) for p in paths]
    symbols: list[dict] = []
    imports: list[dict] = []
    likely_tests: set[str] = set()
    referencing_files: set[str] = set()
    symbol_names: set[str] = set()

    with store.connect() as conn:
        for rel_path in norm_paths:
            rows = conn.execute(
                """
                SELECT s.name, s.qualname, s.kind, s.start_line, s.end_line, f.path
                FROM symbols s
                JOIN files f ON f.id = s.file_id
                WHERE s.repo_id = ? AND f.path = ?
                ORDER BY s.start_line
                LIMIT ?
                """,
                (repo_id, rel_path, limit),
            ).fetchall()
            for row in rows:
                item = dict(row)
                symbols.append(item)
                symbol_names.add(item["name"])

            import_rows = conn.execute(
                """
                SELECT e.metadata_json, f.path
                FROM edges e
                JOIN files f ON f.id = e.source_id
                WHERE e.repo_id = ?
                  AND e.source_kind = 'file'
                  AND f.path = ?
                  AND e.edge_type = 'imports'
                ORDER BY e.metadata_json
                LIMIT ?
                """,
                (repo_id, rel_path, limit),
            ).fetchall()
            for row in import_rows:
                data = _parse_edge_metadata(row["metadata_json"])
                if data.get("module"):
                    imports.append({"path": row["path"], "module": data["module"]})

            stem = Path(rel_path).stem
            candidates = conn.execute(
                """
                SELECT path FROM files
                WHERE repo_id = ?
                  AND (
                    path LIKE ?
                    OR path LIKE ?
                    OR path LIKE ?
                  )
                """,
                (repo_id, f"tests/%test_{stem}.py", f"%/test_{stem}.py", f"%/{stem}_test.py"),
            ).fetchall()
            likely_tests.update(row["path"] for row in candidates)

        for name in sorted(symbol_names):
            ref_rows = conn.execute(
                """
                SELECT DISTINCT f.path, e.metadata_json
                FROM edges e
                JOIN files f ON f.id = e.source_id
                WHERE e.repo_id = ?
                  AND e.source_kind = 'file'
                  AND e.edge_type IN ('references', 'calls')
                  AND e.metadata_json LIKE ?
                LIMIT ?
                """,
                (repo_id, f'%"name": "{name}"%', limit),
            ).fetchall()
            for row in ref_rows:
                data = _parse_edge_metadata(row["metadata_json"])
                if data.get("name") == name and row["path"] not in norm_paths:
                    referencing_files.add(row["path"])
            chunk_rows = conn.execute(
                """
                SELECT DISTINCT f.path
                FROM chunks c
                JOIN files f ON f.id = c.file_id
                WHERE c.repo_id = ? AND c.text LIKE ?
                LIMIT ?
                """,
                (repo_id, f"%{name}%", limit),
            ).fetchall()
            for row in chunk_rows:
                if row["path"] not in norm_paths:
                    referencing_files.add(row["path"])

    return {
        "success": True,
        "graph_status": status["graph_status"],
        "input_paths": norm_paths,
        "symbols": symbols[:limit],
        "imports": imports[:limit],
        "referencing_files": sorted(referencing_files)[:limit],
        "likely_tests": sorted(likely_tests)[:limit],
    }


def _terms(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TERM_RE.finditer(str(text).replace("_", " "))]


def context_for_goal(
    root: Path,
    goal: str,
    *,
    store: CodeGraphStore | None = None,
    budget_chars: int = 20_000,
) -> dict:
    root = Path(root).resolve()
    store = store or CodeGraphStore(cache_path_for_root(root))
    repo_id = _repo_id_for_query(root, store)
    status = graph_status(root, store=store)
    if repo_id is None:
        return {
            "success": True,
            "graph_status": "missing",
            "goal": goal,
            "symbols": [],
            "recommended_files": [],
            "chunks": [],
        }

    terms = _terms(goal)
    budget_chars = max(1000, min(int(budget_chars or 20_000), 80_000))
    with store.connect() as conn:
        symbol_rows = conn.execute(
            """
            SELECT s.name, s.qualname, s.kind, s.start_line, s.end_line,
                   s.signature, s.docstring, f.path
            FROM symbols s
            JOIN files f ON f.id = s.file_id
            WHERE s.repo_id = ?
            LIMIT 1000
            """,
            (repo_id,),
        ).fetchall()
        chunk_rows = conn.execute(
            """
            SELECT f.path, c.start_line, c.end_line, c.text
            FROM chunks c
            JOIN files f ON f.id = c.file_id
            WHERE c.repo_id = ?
            LIMIT 1000
            """,
            (repo_id,),
        ).fetchall()

    ranked_symbols: list[dict] = []
    for row in symbol_rows:
        data = _compact_symbol(row)
        haystack = " ".join(
            str(data.get(key) or "") for key in ("name", "qualname", "docstring", "path")
        ).lower()
        score = sum(1 for term in terms if term in haystack)
        if score:
            data["score"] = score
            ranked_symbols.append(data)
    ranked_symbols.sort(key=lambda item: (-item["score"], item["path"], item["start_line"]))

    ranked_chunks: list[dict] = []
    for row in chunk_rows:
        data = dict(row)
        haystack = f"{data['path']}\n{data['text']}".lower()
        score = sum(1 for term in terms if term in haystack)
        if score:
            text = data["text"]
            if len(text) > 1000:
                text = text[:997] + "..."
            ranked_chunks.append(
                {
                    "path": data["path"],
                    "start_line": data["start_line"],
                    "end_line": data["end_line"],
                    "score": score,
                    "text": text,
                }
            )
    ranked_chunks.sort(key=lambda item: (-item["score"], item["path"], item["start_line"]))

    payload = {
        "success": True,
        "graph_status": status["graph_status"],
        "goal": goal,
        "symbols": ranked_symbols[:20],
        "chunks": ranked_chunks[:10],
        "recommended_files": sorted(
            {item["path"] for item in ranked_symbols[:20]}
            | {item["path"] for item in ranked_chunks[:10]}
        ),
    }
    while len(json.dumps(payload, ensure_ascii=False)) > budget_chars:
        if payload["chunks"]:
            payload["chunks"].pop()
        elif payload["symbols"]:
            payload["symbols"].pop()
        else:
            break
        payload["recommended_files"] = sorted(
            {item["path"] for item in payload["symbols"]}
            | {item["path"] for item in payload["chunks"]}
        )
    return payload
