#!/usr/bin/env python3
"""Symbol Search Tool - Search code symbols across indexed codebase.

Provides symbol lookup, usage finding, type hierarchy, and cross-referencing
using the code index built by code_index tool.
"""

import json
import os
import re
import sqlite3
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home
from tools.code_index import _index_db_path


def symbol_search(
    query: str,
    project_root: str,
    symbol_type: Optional[str] = None,
    language: Optional[str] = None,
    find_usages: bool = False,
    exact: bool = False,
    limit: int = 50,
    task_id: Optional[str] = None,
) -> str:
    """Search for symbols (classes, functions, variables) across the indexed codebase."""
    if not os.path.isdir(project_root):
        return json.dumps({
            "success": False,
            "error": f"Project root not found: {project_root}",
        })

    abs_root = os.path.abspath(project_root)
    db_path = _index_db_path(abs_root)

    if not os.path.isfile(db_path):
        return json.dumps({
            "success": False,
            "error": "No code index found. Run 'code_index' tool with operation='build' first.",
        })

    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT value FROM meta WHERE key = 'project_root'")
        row = cursor.fetchone()
        indexed_root = row[0] if row else abs_root

        conditions: List[str] = []
        params: List[Any] = []

        if exact:
            conditions.append("s.name = ?")
            params.append(query)
        else:
            conditions.append("s.name LIKE ?")
            params.append(f"%{query}%")

        if symbol_type:
            types = [t.strip().lower() for t in symbol_type.split(",")]
            placeholders = ",".join("?" * len(types))
            conditions.append(f"s.type IN ({placeholders})")
            params.extend(types)

        if language:
            conditions.append("f.language = ?")
            params.append(language)

        where = " AND ".join(conditions) if conditions else "1=1"

        cursor.execute(f"""
            SELECT s.name, s.type, s.file_path, s.line, f.language
            FROM symbols s
            JOIN files f ON s.file_path = f.path
            WHERE {where}
            ORDER BY s.name
            LIMIT ?
        """, params + [limit])

        results = []
        for name, sym_type, file_path, line, lang in cursor.fetchall():
            full_path = os.path.join(indexed_root, file_path)
            results.append({
                "name": name,
                "type": sym_type,
                "file": file_path,
                "full_path": full_path,
                "line": line,
                "language": lang,
            })

        # Cross-file usage search: search ALL indexed source files, not
        # just the definition's own file.
        if find_usages and results:
            cursor.execute("SELECT path FROM files")
            all_indexed = [r[0] for r in cursor.fetchall()]

            for r in results:
                usages = []
                pattern = re.compile(r"\b" + re.escape(r["name"]) + r"\b", re.MULTILINE)
                for indexed_path in all_indexed:
                    full = os.path.join(indexed_root, indexed_path)
                    try:
                        with open(full, "r", encoding="utf-8", errors="replace") as f:
                            content = f.read()
                    except Exception:
                        continue
                    for match in pattern.finditer(content):
                        line_num = content[:match.start()].count("\n") + 1
                        line_start = content.rfind("\n", 0, match.start()) + 1
                        line_end = content.find("\n", match.end())
                        line_content = content[
                            line_start : line_end if line_end != -1 else len(content)
                        ].strip()
                        usages.append({
                            "file": indexed_path,
                            "line": line_num,
                            "context": line_content[:120],
                        })
                r["usages"] = usages[:30]
                r["usage_count"] = len(usages)

        per_file: Dict[str, List[Dict]] = {}
        for r in results:
            key = r["file"]
            if key not in per_file:
                per_file[key] = []
            per_file[key].append(r)

        return json.dumps({
            "success": True,
            "query": query,
            "total": len(results),
            "results": results[:limit],
            "by_file": {k: v for k, v in list(per_file.items())[:30]},
        }, ensure_ascii=False)

    except sqlite3.Error as e:
        return json.dumps({
            "success": False,
            "error": f"Index query failed: {e}",
        })
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
        })
    finally:
        if conn:
            conn.close()


def check_symbol_search_requirements() -> bool:
    """Symbol search requires Python standard library."""
    return True


SYMBOL_SEARCH_SCHEMA = {
    "name": "symbol_search",
    "description": (
        "Search for symbols (classes, functions, variables) across the indexed codebase.\n\n"
        "Requires the code index to be built first using the 'code_index' tool with operation='build'.\n\n"
        "Parameters:\n"
        "- query: Symbol name to search for (partial match by default)\n"
        "- project_root: Root directory of the project\n"
        "- symbol_type: Filter by type: class, function, method, variable, type, struct, enum, trait, interface\n"
        "- language: Filter by language: python, typescript, javascript, go, rust\n"
        "- find_usages: Find all usages of matching symbols across all indexed files\n"
        "- exact: Exact match instead of partial\n"
        "- limit: Maximum results to return (default 50)"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Symbol name to search for (partial match by default)",
            },
            "project_root": {
                "type": "string",
                "description": "Root directory of the project",
            },
            "symbol_type": {
                "type": "string",
                "description": "Filter by type: class, function, method, variable, type, struct, enum, trait, interface",
            },
            "language": {
                "type": "string",
                "description": "Filter by language: python, typescript, javascript, go, rust",
            },
            "find_usages": {
                "type": "boolean",
                "description": "Find all usages of matching symbols across all indexed files",
                "default": False,
            },
            "exact": {
                "type": "boolean",
                "description": "Exact match instead of partial",
                "default": False,
            },
            "limit": {
                "type": "integer",
                "description": "Maximum results to return",
                "default": 50,
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
    name="symbol_search",
    toolset="code_search",
    schema=SYMBOL_SEARCH_SCHEMA,
    handler=lambda args, **kw: symbol_search(
        query=args.get("query", ""),
        project_root=args.get("project_root", ""),
        symbol_type=args.get("symbol_type"),
        language=args.get("language"),
        find_usages=args.get("find_usages", False),
        exact=args.get("exact", False),
        limit=args.get("limit", 50),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_symbol_search_requirements,
    emoji="🔍",
)
