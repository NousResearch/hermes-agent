#!/usr/bin/env python3
"""
Semantic Search Tool - Natural language code search using LLM re-ranking

Indexes code files by chunking, then searches using keyword pre-filtering
and LLM-based semantic re-ranking via the auxiliary client for real understanding.
"""

import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home


MAX_FILE_SIZE = 2 * 1024 * 1024


def _chunk_file(content: str, max_chars: int = 2000) -> List[str]:
    chunks = []
    lines = content.split("\n")
    current = []
    current_len = 0
    for line in lines:
        if current_len + len(line) > max_chars and current:
            chunks.append("\n".join(current))
            current = []
            current_len = 0
        current.append(line)
        current_len += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


def _classify_query(query: str) -> Dict[str, Any]:
    query_lower = query.lower()

    symbol_indicators = [r"\bclass\s+\w+", r"\bfunction\s+\w+", r"\bdef\s+\w+",
                         r"\bfn\s+\w+", r"\bfunc\s+\w+", r"\b(struct|enum|trait)\s+\w+"]
    for pat in symbol_indicators:
        if re.search(pat, query):
            match = re.search(r"\b(\w+)\s*$", query)
            return {"type": "symbol", "target": match.group(1) if match else query, "confidence": 0.9}

    error_words = {"error", "bug", "fail", "crash", "issue", "problem", "wrong"}
    if any(w in query_lower for w in error_words):
        return {"type": "error", "target": query, "confidence": 0.7}

    find_words = {"find", "search", "where", "lookup", "locate"}
    if any(w in query_lower.split() for w in find_words):
        return {"type": "search", "target": query, "confidence": 0.6}

    return {"type": "semantic", "target": query, "confidence": 0.5}


def _get_index_db(project_root: str) -> str:
    project_name = os.path.basename(os.path.abspath(project_root))
    index_dir = get_hermes_home() / "semantic-index"
    index_dir.mkdir(parents=True, exist_ok=True)
    return str(index_dir / f"{project_name}.db")


def _rerank_with_llm(query: str, candidates: List[Dict]) -> List[Dict]:
    try:
        from agent.auxiliary_client import call_llm

        items = "\n".join(
            f"[{i}] {c['file']}:{c.get('line', 1)}\n{c['snippet']}"
            for i, c in enumerate(candidates[:15])
        )

        prompt = (
            f"Given this search query: \"{query}\"\n\n"
            f"Rank the following code snippets by relevance (0=unrelated, 10=perfect match). "
            f"Return a JSON list of {{index, score, reason}}.\n\n{items}"
        )

        response = call_llm(
            task="session_search",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1000,
        )

        import json as _json
        scores = _json.loads(response)
        score_map = {s["index"]: s["score"] for s in scores if isinstance(s, dict)}

        for c in candidates:
            idx = candidates.index(c)
            c["relevance"] = score_map.get(idx, 0)
            c["score"] = round(c["relevance"] / 10.0, 4)

        candidates.sort(key=lambda c: c.get("relevance", 0), reverse=True)
    except Exception:
        pass

    return candidates


def semantic_search(
    query: str,
    project_root: str,
    mode: str = "hybrid",
    max_results: int = 20,
    file_pattern: Optional[str] = None,
    task_id: Optional[str] = None,
) -> str:  # noqa: D205
    """
    Search code by natural language using LLM-based semantic re-ranking.

    Args:
        query: Natural language search query
        project_root: Root directory of the project
        mode: hybrid (keyword + LLM re-rank), keyword, index
        max_results: Maximum results to return
        file_pattern: Filter by file pattern (e.g., *.py)

    Returns:
        JSON string with ranked search results
    """
    if not os.path.isdir(project_root):
        return json.dumps({
            "success": False,
            "error": f"Project root not found: {project_root}",
        })

    abs_root = os.path.abspath(project_root)
    intent = _classify_query(query)

    db_path = _get_index_db(abs_root)

    if mode == "index":
        conn = None
        try:
            conn = sqlite3.connect(db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT,
                    language TEXT,
                    chunk_index INTEGER,
                    text TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_file ON chunks(file_path)")
            conn.execute("PRAGMA synchronous = OFF")
            conn.execute("PRAGMA journal_mode = MEMORY")

            exclude_dirs = {".git", "__pycache__", "node_modules", ".venv", "venv",
                           ".env", ".tox", "build", "dist", ".hermes"}

            stats = {"files_processed": 0, "chunks_indexed": 0}
            for root, dirs, files in os.walk(abs_root):
                dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith(".")]
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, abs_root)
                    ext = os.path.splitext(file)[1].lower()
                    lang_map = {".py": "python", ".ts": "typescript", ".js": "javascript",
                                ".go": "go", ".rs": "rust", ".java": "java",
                                ".rb": "ruby", ".c": "c", ".cpp": "cpp", ".cs": "csharp"}
                    language = lang_map.get(ext, "unknown")
                    try:
                        stat_info = os.stat(file_path)
                        if stat_info.st_size > MAX_FILE_SIZE:
                            continue
                        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                            content = f.read()
                    except Exception:
                        continue
                    chunks = _chunk_file(content)
                    conn.execute("DELETE FROM chunks WHERE file_path = ?", (rel_path,))
                    for i, chunk in enumerate(chunks):
                        conn.execute(
                            "INSERT INTO chunks (file_path, language, chunk_index, text) VALUES (?, ?, ?, ?)",
                            (rel_path, language, i, chunk)
                        )
                        stats["chunks_indexed"] += 1
                    stats["files_processed"] += 1

            conn.commit()
            return json.dumps({
                "success": True,
                "operation": "index",
                "project": abs_root,
                "stats": stats,
                "message": f"Indexed {stats['files_processed']} files, {stats['chunks_indexed']} chunks",
            })

        except Exception as e:
            return json.dumps({"success": False, "error": f"Index build failed: {e}"})
        finally:
            if conn:
                conn.close()

    query_words = query.lower().split()

    if mode == "keyword":
        try:
            results = []
            exclude_dirs = {".git", "__pycache__", "node_modules", ".venv",
                           "venv", ".env", "build", "dist"}
            for root, dirs, files in os.walk(abs_root):
                dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith(".")]
                for file in files:
                    if file_pattern:
                        pat = file_pattern.replace("*", ".*").replace("?", ".")
                        if not re.search(pat, file):
                            continue
                    file_path = os.path.join(root, file)
                    try:
                        stat_info = os.stat(file_path)
                        if stat_info.st_size > MAX_FILE_SIZE:
                            continue
                        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                            content = f.read()
                    except Exception:
                        continue
                    score = sum(1 for w in query_words if w in content.lower()) / max(len(query_words), 1)
                    if score > 0:
                        results.append({
                            "file": os.path.relpath(file_path, abs_root),
                            "score": round(score, 3),
                            "snippet": content[:200],
                        })
            results.sort(key=lambda r: r["score"], reverse=True)
            return json.dumps({
                "success": True,
                "mode": "keyword",
                "query": query,
                "intent": intent,
                "total": len(results),
                "results": results[:max_results],
            }, ensure_ascii=False)

        except Exception as e:
            return json.dumps({"success": False, "error": f"Keyword search failed: {e}"})

    if not os.path.isfile(db_path):
        return json.dumps({
            "success": False,
            "error": "No index found. Run with mode='index' first.",
        })

    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        sql = "SELECT file_path, language, chunk_index, text FROM chunks"
        params: List[Any] = []
        if file_pattern:
            pat = file_pattern.replace("*", "%").replace("?", "_")
            sql += " WHERE file_path LIKE ?"
            params.append(pat)
        cursor.execute(sql, params)

        candidates: List[Dict] = []
        for file_path, language, chunk_idx, text in cursor.fetchall():
            kw_score = sum(1 for w in query_words if w in text.lower()) / max(len(query_words), 1)
            if kw_score > 0:
                candidates.append({
                    "file": file_path,
                    "language": language,
                    "chunk": chunk_idx,
                    "keyword_score": kw_score,
                    "snippet": text[:300],
                })

        candidates.sort(key=lambda c: c["keyword_score"], reverse=True)
        top_candidates = candidates[:max_results * 2]

        if mode == "semantic" or mode == "hybrid":
            top_candidates = _rerank_with_llm(query, top_candidates)

        for c in top_candidates:
            if "score" not in c:
                c["score"] = c["keyword_score"]

        return json.dumps({
            "success": True,
            "mode": mode,
            "query": query,
            "intent": intent,
            "total": len(top_candidates),
            "results": top_candidates[:max_results],
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})
    finally:
        if conn:
            conn.close()


def check_semantic_search_requirements() -> bool:
    try:
        from agent.auxiliary_client import call_llm
        return True
    except ImportError:
        return False


SEMANTIC_SEARCH_SCHEMA = {
    "name": "semantic_search",
    "description": (
        "Search code by natural language. Uses LLM-based semantic re-ranking "
        "via the auxiliary client to understand code intent beyond keyword matching.\n\n"
        "First run with mode='index' to build the chunk index.\n"
        "Modes:\n"
        "- hybrid (default): Keyword pre-filter + LLM semantic re-ranking\n"
        "- semantic: Keyword pre-filter + LLM re-ranking (same as hybrid)\n"
        "- keyword: Fast local keyword search (no index needed)\n"
        "- index: Build/update the chunk index\n\n"
        "Query understanding: auto-detects symbol lookups, error searches, and general queries."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language search query",
            },
            "project_root": {
                "type": "string",
                "description": "Root directory of the project",
            },
            "mode": {
                "type": "string",
                "description": "Search mode: hybrid, semantic, keyword, index",
                "enum": ["hybrid", "semantic", "keyword", "index"],
                "default": "hybrid",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum results to return",
                "default": 20,
            },
            "file_pattern": {
                "type": "string",
                "description": "Filter by file pattern (e.g., *.py)",
            },
            "task_id": {
                "type": "string",
                "description": "Optional task ID for tracking",
            },
        },
        "required": ["query", "project_root"],
    },
}


from tools.registry import registry

registry.register(
    name="semantic_search",
    toolset="code_search",
    schema=SEMANTIC_SEARCH_SCHEMA,
    handler=lambda args, **kw: semantic_search(
        query=args.get("query", ""),
        project_root=args.get("project_root", ""),
        mode=args.get("mode", "hybrid"),
        max_results=args.get("max_results", 20),
        file_pattern=args.get("file_pattern"),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_semantic_search_requirements,
    emoji="🧠",
)