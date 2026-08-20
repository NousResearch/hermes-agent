#!/usr/bin/env python3
"""
SQLite Query Tool - Safe, parameterized SQLite queries

Provides safe SQLite operations: execute SELECT/INSERT/UPDATE/DELETE queries
with parameterized statements to prevent injection.
"""

import json
import os
import sqlite3
from typing import Any, Dict, List, Optional


def _dict_factory(cursor: sqlite3.Cursor, row: tuple) -> dict:
    """Convert SQLite row to dictionary."""
    return {col[0]: row[i] for i, col in enumerate(cursor.description)}


def sqlite_query(
    db_path: str,
    query: str,
    params: Optional[Dict[str, Any]] = None,
    fetch: str = "all",
    timeout: float = 10.0,
    task_id: Optional[str] = None,
) -> str:  # noqa: D205
    """
    Execute safe, parameterized SQLite queries.

    Args:
        db_path: Path to SQLite database file
        query: SQL query with optional named parameters (:key)
        params: Parameter values for the query (dict with :key keys)
        fetch: Fetch mode: "all", "one", or "none"
        timeout: Query timeout in seconds

    Returns:
        JSON string with query results
    """
    if not os.path.exists(db_path):
        return json.dumps({
            "success": False,
            "error": f"Database not found: {db_path}",
        })

    if not os.path.isfile(db_path):
        return json.dumps({
            "success": False,
            "error": f"Not a file: {db_path}",
        })

    abs_path = os.path.abspath(db_path)
    if not abs_path:
        return json.dumps({
            "success": False,
            "error": "Invalid database path",
        })

    MAX_DB_SIZE = 500 * 1024 * 1024
    file_size = os.path.getsize(db_path)
    if file_size > MAX_DB_SIZE:
        return json.dumps({
            "success": False,
            "error": f"Database too large: {file_size / (1024*1024):.1f}MB (max 500MB)",
        })

    query_stripped = query.strip().upper()
    if not query_stripped:
        return json.dumps({
            "success": False,
            "error": "Empty query",
        })

    allowed_starters = ("SELECT", "INSERT", "UPDATE", "DELETE", "PRAGMA", "EXPLAIN")
    if not any(query_stripped.startswith(s) for s in allowed_starters):
        return json.dumps({
            "success": False,
            "error": f"Only SELECT, INSERT, UPDATE, DELETE, PRAGMA, EXPLAIN queries allowed. Got: {query_stripped.split()[0] if query_stripped else 'empty'}",
        })

    if fetch not in ("all", "one", "none"):
        return json.dumps({
            "success": False,
            "error": "fetch must be 'all', 'one', or 'none'",
        })

    conn = None
    try:
        conn = sqlite3.connect(abs_path, timeout=timeout)
        conn.row_factory = _dict_factory
        cursor = conn.cursor()

        cursor.execute(query, params or {})

        result: Dict[str, Any] = {"success": True}

        if query_stripped.startswith("SELECT") or query_stripped.startswith("PRAGMA") or query_stripped.startswith("EXPLAIN"):
            if fetch == "all":
                rows = cursor.fetchall()
                result["rows"] = rows
                result["count"] = len(rows)
            elif fetch == "one":
                row = cursor.fetchone()
                result["row"] = row
                result["count"] = 1 if row else 0
            else:
                result["count"] = 0
        else:
            conn.commit()
            result["changes"] = conn.total_changes
            result["last_row_id"] = cursor.lastrowid

        return json.dumps(result, ensure_ascii=False, default=str)

    except sqlite3.Error as e:
        return json.dumps({
            "success": False,
            "error": f"SQLite error: {e}",
        })
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
        })
    finally:
        if conn:
            conn.close()


def check_sqlite_query_requirements() -> bool:
    """SQLite is available in Python standard library."""
    return True


SQLITE_QUERY_SCHEMA = {
    "name": "sqlite_query",
    "description": (
        "Execute safe, parameterized SQLite queries. Supports SELECT, INSERT, UPDATE, DELETE.\n\n"
        "Parameters:\n"
        "- db_path: Path to SQLite database file\n"
        "- query: SQL query with optional named parameters (:key)\n"
        "- params: Parameter values as JSON dict (e.g., {\"key\": \"value\"})\n"
        "- fetch: Fetch mode: all, one, none\n"
        "- timeout: Query timeout in seconds"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "db_path": {
                "type": "string",
                "description": "Path to SQLite database file",
            },
            "query": {
                "type": "string",
                "description": "SQL query with optional named parameters (:key)",
            },
            "params": {
                "type": "object",
                "description": "Parameter values as JSON dict",
                "default": {},
            },
            "fetch": {
                "type": "string",
                "description": "Fetch mode: all, one, none",
                "enum": ["all", "one", "none"],
                "default": "all",
            },
            "timeout": {
                "type": "number",
                "description": "Query timeout in seconds",
                "default": 10.0,
            },
            "task_id": {
                "type": "string",
                "description": "Optional task ID for tracking",
            },
        },
        "required": ["db_path", "query"],
    },
}


from tools.registry import registry

registry.register(
    name="sqlite_query",
    toolset="database",
    schema=SQLITE_QUERY_SCHEMA,
    handler=lambda args, **kw: sqlite_query(
        db_path=args.get("db_path", ""),
        query=args.get("query", ""),
        params=args.get("params"),
        fetch=args.get("fetch", "all"),
        timeout=args.get("timeout", 10.0),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_sqlite_query_requirements,
    emoji="🗄️",
)