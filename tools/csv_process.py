#!/usr/bin/env python3
"""
CSV Processing Tool - Read, filter, transform, aggregate, and export CSV files

Provides CSV file processing capabilities including reading, filtering rows,
selecting columns, transforming data, and exporting results.
"""

import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def _validate_csv_path(file_path: str, working_dir: Optional[str] = None) -> Optional[str]:
    """Validate path is contained within the working directory.

    Uses explicit allowed-root containment — resolved absolute paths must
    be relative to the working directory.  No blacklist; if the path escapes
    the working root, it is rejected.
    """
    if working_dir is None:
        working_dir = os.getcwd()

    try:
        abs_path = Path(file_path).resolve()
        abs_working = Path(working_dir).resolve()

        # The resolved path must be inside (or equal to) the working directory.
        # relative_to() raises ValueError when the path escapes the root.
        abs_path.relative_to(abs_working)
        return None
    except ValueError:
        return (
            f"Path validation failed: {file_path} resolves outside the "
            f"working directory ({working_dir}). Use a path within the "
            f"project root."
        )
    except Exception as e:
        return f"Invalid path: {str(e)}"


def csv_process(
    operation: str,
    file_path: str,
    columns: Optional[List[str]] = None,
    filter_condition: Optional[str] = None,
    output_path: Optional[str] = None,
    working_dir: Optional[str] = None,
    task_id: Optional[str] = None,
) -> str:
    """
    Process CSV files with various operations.

    Args:
        operation: Operation to perform (read, filter, transform, aggregate, export)
        file_path: Path to CSV file
        columns: Columns to select
        filter_condition: Filter condition (e.g., "age > 25")
        output_path: Output path for export

    Returns:
        JSON string with operation results
    """
    try:
        path_error = _validate_csv_path(file_path, working_dir)
        if path_error:
            return json.dumps({"success": False, "error": path_error})

        if not os.path.exists(file_path):
            return json.dumps({
                "success": False,
                "error": f"File not found: {file_path}",
            })

        with open(file_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if operation == "read":
            selected_rows = _select_columns(rows, columns)
            return json.dumps({
                "success": True,
                "operation": "read",
                "row_count": len(selected_rows),
                "columns": list(selected_rows[0].keys()) if selected_rows else [],
                "data": selected_rows[:1000] if selected_rows else [],
            }, ensure_ascii=False)

        elif operation == "filter":
            filtered_rows = list(_filter_rows(rows, filter_condition))
            selected_rows = _select_columns(filtered_rows, columns)
            return json.dumps({
                "success": True,
                "operation": "filter",
                "original_count": len(rows),
                "filtered_count": len(selected_rows),
                "data": selected_rows[:1000] if selected_rows else [],
            }, ensure_ascii=False)

        elif operation == "transform":
            selected_rows = _select_columns(rows, columns)
            return json.dumps({
                "success": True,
                "operation": "transform",
                "row_count": len(selected_rows),
                "columns": list(selected_rows[0].keys()) if selected_rows else [],
                "data": selected_rows[:1000] if selected_rows else [],
            }, ensure_ascii=False)

        elif operation == "aggregate":
            if not rows:
                return json.dumps({
                    "success": True,
                    "operation": "aggregate",
                    "results": {},
                    "message": "No data to aggregate",
                }, ensure_ascii=False)
            agg_result = _aggregate(rows, columns)
            return json.dumps({
                "success": True,
                "operation": "aggregate",
                "results": agg_result,
            }, ensure_ascii=False)

        elif operation == "export":
            if not output_path:
                return json.dumps({
                    "success": False,
                    "error": "output_path required for export operation",
                })

            output_error = _validate_csv_path(output_path, working_dir)
            if output_error:
                return json.dumps({"success": False, "error": f"Output path validation failed: {output_error}"})

            selected_rows = _select_columns(rows, columns)
            filtered_rows = list(_filter_rows(selected_rows, filter_condition))

            with open(output_path, "w", newline="", encoding="utf-8") as f:
                if filtered_rows:
                    writer = csv.DictWriter(f, fieldnames=filtered_rows[0].keys())
                    writer.writeheader()
                    writer.writerows(filtered_rows)

            return json.dumps({
                "success": True,
                "operation": "export",
                "output_path": output_path,
                "row_count": len(filtered_rows),
            }, ensure_ascii=False)

        else:
            return json.dumps({
                "success": False,
                "error": f"Unknown operation: {operation}",
            })

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
        })


def _select_columns(rows: List[Dict[str, str]], columns: Optional[List[str]]) -> List[Dict[str, str]]:
    """Select only the specified columns."""
    if not columns:
        return rows
    return [{col: row.get(col, "") for col in columns if col in row} for row in rows]


def _filter_rows(rows: List[Dict[str, str]], condition: Optional[str]) -> List[Dict[str, str]]:
    """Filter rows based on a condition like 'age > 25' or 'status == active'."""
    if not condition:
        return list(rows)

    try:
        result = []
        for row in rows:
            if _evaluate_condition(row, condition):
                result.append(row)
        return result
    except Exception:
        return list(rows)


def _evaluate_condition(row: Dict[str, str], condition: str) -> bool:
    """Evaluate a simple condition against a row."""
    match = re.match(r"(\w+)\s*(==|!=|>=|<=|>|<)\s*(.+)", condition.strip())
    if not match:
        return True

    column, operator, value = match.groups()
    row_value = row.get(column, "")

    try:
        if operator == "==":
            return str(row_value) == value.strip().strip("'\"")
        elif operator == "!=":
            return str(row_value) != value.strip().strip("'\"")
        elif operator == ">":
            return float(row_value) > float(value)
        elif operator == "<":
            return float(row_value) < float(value)
        elif operator == ">=":
            return float(row_value) >= float(value)
        elif operator == "<=":
            return float(row_value) <= float(value)
    except (ValueError, TypeError):
        return str(row_value) == value.strip().strip("'\"")
    return True


def _aggregate(rows: List[Dict[str, str]], columns: Optional[List[str]]) -> Dict[str, Any]:
    """Aggregate data - calculate counts, sums, averages for numeric columns."""
    if not columns:
        columns = list(rows[0].keys()) if rows else []

    result = {}
    for col in columns:
        values = [row.get(col, "") for row in rows]
        numeric_values = []

        for v in values:
            try:
                numeric_values.append(float(v))
            except (ValueError, TypeError):
                pass

        result[col] = {
            "count": len(values),
            "unique": len(set(values)),
            "numeric_count": len(numeric_values),
        }

        if numeric_values:
            result[col].update({
                "sum": sum(numeric_values),
                "avg": sum(numeric_values) / len(numeric_values),
                "min": min(numeric_values),
                "max": max(numeric_values),
            })

    return result


def check_csv_process_requirements() -> bool:
    """CSV processing tool has no external requirements -- always available."""
    return True


CSV_PROCESS_SCHEMA = {
    "name": "csv_process",
    "description": (
        "Process CSV files: read, filter, transform, aggregate, and export.\n\n"
        "Operations:\n"
        "- read: Read CSV and return data (optionally select columns)\n"
        "- filter: Filter rows by condition (e.g., \"age > 25\", \"status == active\")\n"
        "- transform: Select specific columns from data\n"
        "- aggregate: Calculate statistics (count, sum, avg, min, max)\n"
        "- export: Write filtered/selected data to a new CSV file\n\n"
        "Filter operators: ==, !=, >, <, >=, <= (for numbers)\n"
        "String values in conditions should use single quotes: status == 'active'"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "description": "Operation to perform",
                "enum": ["read", "filter", "transform", "aggregate", "export"],
            },
            "file_path": {
                "type": "string",
                "description": "Path to CSV file"
            },
            "columns": {
                "type": "array",
                "description": "Columns to select (for read, transform, export)",
                "items": {"type": "string"}
            },
            "filter_condition": {
                "type": "string",
                "description": "Filter condition (e.g., \"age > 25\", \"status == 'active'\")"
            },
            "output_path": {
                "type": "string",
                "description": "Output path for export operation"
            }
        },
        "required": ["operation", "file_path"]
    }
}


from tools.registry import registry

registry.register(
    name="csv_process",
    toolset="data",
    schema=CSV_PROCESS_SCHEMA,
    handler=lambda args, **kw: csv_process(
        operation=args.get("operation", "read"),
        file_path=args.get("file_path", ""),
        columns=args.get("columns"),
        filter_condition=args.get("filter_condition"),
        output_path=args.get("output_path"),
    ),
    check_fn=check_csv_process_requirements,
    emoji="📊",
)
