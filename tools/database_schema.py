#!/usr/bin/env python3
"""
Database Schema Tool - Inspect SQLite database schema

Provides schema inspection: list tables, describe columns, list indexes.
"""

import json
import os
import re
import sqlite3
from typing import Any, Dict, List, Optional


def _dict_factory(cursor: sqlite3.Cursor, row: tuple) -> dict:
    return {col[0]: row[i] for i, col in enumerate(cursor.description)}


def database_schema(
    db_path: str,
    operation: str = "list_tables",
    table_name: Optional[str] = None,
    task_id: Optional[str] = None,
) -> str:  # noqa: D205
    """
    Inspect SQLite database schema.

    Args:
        db_path: Path to SQLite database file
        operation: list_tables, describe_table, list_indexes
        table_name: Table name for describe_table/list_indexes
        task_id: Optional task ID for tracking

    Returns:
        JSON string with schema information
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

    if operation not in ("list_tables", "describe_table", "list_indexes"):
        return json.dumps({
            "success": False,
            "error": "operation must be list_tables, describe_table, or list_indexes",
        })

    if operation != "list_tables" and not table_name:
        return json.dumps({
            "success": False,
            "error": f"table_name required for {operation} operation",
        })

    if operation != "list_tables" and table_name:
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", table_name):
            return json.dumps({
                "success": False,
                "error": f"Invalid table name: must be alphanumeric and underscore only",
            })

    conn = None
    try:
        conn = sqlite3.connect(abs_path)
        conn.row_factory = _dict_factory
        cursor = conn.cursor()

        if operation == "list_tables":
            cursor.execute(
                "SELECT name AS table_name, type AS object_type "
                "FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            tables = cursor.fetchall()

            result: Dict[str, Any] = {
                "success": True,
                "operation": "list_tables",
                "database": os.path.basename(db_path),
                "tables": tables,
                "count": len(tables),
            }

        elif operation == "describe_table":
            cursor.execute(f"PRAGMA table_info(\"{table_name}\")")
            columns = cursor.fetchall()

            cursor.execute(
                "SELECT name, type FROM sqlite_master "
                f"WHERE type='table' AND name=\"{table_name}\""
            )
            table_info = cursor.fetchone()

            result = {
                "success": True,
                "operation": "describe_table",
                "database": os.path.basename(db_path),
                "table_name": table_name,
                "columns": columns,
                "column_count": len(columns),
            }

            if not columns:
                result["warning"] = f"Table '{table_name}' not found"

        elif operation == "list_indexes":
            cursor.execute(
                f"SELECT name AS index_name, origin, unique AS is_unique "
                f"FROM sqlite_master WHERE type='index' AND tbl_name=\"{table_name}\" "
                "ORDER BY name"
            )
            indexes = cursor.fetchall()

            result = {
                "success": True,
                "operation": "list_indexes",
                "database": os.path.basename(db_path),
                "table_name": table_name,
                "indexes": indexes,
                "count": len(indexes),
            }

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


def check_database_schema_requirements() -> bool:
    """SQLite is available in Python standard library."""
    return True


DATABASE_SCHEMA_SCHEMA = {
    "name": "database_schema",
    "description": (
        "Inspect SQLite database schema: list tables, describe columns, list indexes.\n\n"
        "Parameters:\n"
        "- db_path: Path to SQLite database file\n"
        "- operation: list_tables, describe_table, list_indexes\n"
        "- table_name: Required for describe_table and list_indexes"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "db_path": {
                "type": "string",
                "description": "Path to SQLite database file",
            },
            "operation": {
                "type": "string",
                "description": "Operation: list_tables, describe_table, list_indexes",
                "enum": ["list_tables", "describe_table", "list_indexes"],
                "default": "list_tables",
            },
            "table_name": {
                "type": "string",
                "description": "Table name for describe_table/list_indexes",
            },
            "task_id": {
                "type": "string",
                "description": "Optional task ID for tracking",
            },
        },
        "required": ["db_path"],
    },
}


from tools.registry import registry

registry.register(
    name="database_schema",
    toolset="database",
    schema=DATABASE_SCHEMA_SCHEMA,
    handler=lambda args, **kw: database_schema(
        db_path=args.get("db_path", ""),
        operation=args.get("operation", "list_tables"),
        table_name=args.get("table_name"),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_database_schema_requirements,
    emoji="📋",
)