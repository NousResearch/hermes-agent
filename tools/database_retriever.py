"""Universal Database Data Retriever.

Read-only SQL retrieval backend backed by SQLGlot. The module intentionally
separates SQL parsing/validation/transpilation from execution so new database
connectors can be added without weakening the safety model.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from hermes_constants import get_hermes_home
from tools.registry import registry, tool_error, tool_result

try:
    import sqlglot
    from sqlglot import exp
except Exception:  # pragma: no cover - check_fn reports unavailable
    sqlglot = None
    exp = None

logger = logging.getLogger(__name__)

SUPPORTED_SQLGLOT_DIALECTS: tuple[str, ...] = (
    "athena",
    "bigquery",
    "clickhouse",
    "databricks",
    "doris",
    "dremio",
    "drill",
    "druid",
    "duckdb",
    "dune",
    "exasol",
    "fabric",
    "hive",
    "materialize",
    "mysql",
    "oracle",
    "postgres",
    "presto",
    "prql",
    "redshift",
    "risingwave",
    "singlestore",
    "snowflake",
    "solr",
    "spark",
    "spark2",
    "sqlite",
    "starrocks",
    "tableau",
    "teradata",
    "trino",
    "tsql",
)

ALWAYS_EXECUTABLE_DIALECTS = {"sqlite"}
OPTIONAL_EXECUTABLE_DIALECTS = {"duckdb", "postgres"}
DEFAULT_SOURCE_DIALECT = "postgres"
DEFAULT_MAX_ROWS = 100
HARD_MAX_ROWS = 1000
DEFAULT_MAX_CELL_CHARS = 4000
HARD_MAX_CELL_CHARS = 20000
SQL_GENERATION_TASK = "database_retrieve"
DEFAULT_REPAIR_ATTEMPTS = 2


@dataclass(frozen=True)
class DialectCapability:
    name: str
    can_parse: bool
    can_transpile: bool
    can_execute: bool
    supports_limit: bool
    supports_explain: bool
    connector_type: str | None


@dataclass(frozen=True)
class ColumnSchema:
    name: str
    type: str
    nullable: bool = True
    primary_key: bool = False


@dataclass(frozen=True)
class TableSchema:
    name: str
    columns: tuple[ColumnSchema, ...]


@dataclass(frozen=True)
class DatabaseSchema:
    database_id: str
    dialect: str
    tables: tuple[TableSchema, ...]


@dataclass(frozen=True)
class ValidationResult:
    sql: str
    source_dialect: str
    target_dialect: str
    referenced_tables: tuple[str, ...]
    referenced_columns: tuple[str, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class GeneratedSql:
    user_request: str
    source_dialect: str
    target_dialect: str
    raw_sql: str
    sql: str
    referenced_tables: tuple[str, ...]
    referenced_columns: tuple[str, ...]
    warnings: tuple[str, ...]
    attempts: tuple[dict[str, str], ...] = ()


@dataclass(frozen=True)
class SqlAttempt:
    sql: str
    error: str = ""

    def to_dict(self) -> dict[str, str]:
        data = {"sql": self.sql}
        if self.error:
            data["error"] = self.error
        return data


class DatabaseRetrievalError(ValueError):
    """Raised when a retrieval request fails validation or execution."""


def redact_sql_literals(sql: str) -> str:
    """Return SQL with literal values removed for audit logging."""
    if not sql or sqlglot is None:
        return sql or ""
    try:
        expression = parse_single_statement(sql)
        redacted = expression.transform(
            lambda node: exp.Placeholder()
            if isinstance(node, (exp.Literal, exp.Boolean, exp.Null))
            else node,
            copy=True,
        )
        return redacted.sql(dialect=DEFAULT_SOURCE_DIALECT)
    except Exception:
        return "[unparseable sql]"


def _redact_attempts(
    attempts: tuple[dict[str, str], ...] | list[dict[str, str]],
) -> list[dict[str, str]]:
    redacted = []
    for attempt in attempts:
        item = {"sql": redact_sql_literals(attempt.get("sql", ""))}
        if attempt.get("error"):
            item["error"] = attempt["error"]
        redacted.append(item)
    return redacted


def build_audit_event(
    *,
    action: str,
    database_id: str,
    source_dialect: str = "",
    target_dialect: str = "",
    sql: str = "",
    generated_sql: str = "",
    referenced_tables: list[str] | tuple[str, ...] = (),
    referenced_columns: list[str] | tuple[str, ...] = (),
    row_count: int | None = None,
    execution_ms: int | None = None,
    attempts: tuple[dict[str, str], ...] | list[dict[str, str]] = (),
    warnings: list[str] | tuple[str, ...] = (),
    error: str = "",
    parameter_count: int = 0,
) -> dict[str, Any]:
    event = {
        "component": "database_retriever",
        "action": action,
        "database_id": database_id,
        "source_dialect": source_dialect,
        "target_dialect": target_dialect,
        "read_only": True,
        "sql_redacted": redact_sql_literals(sql),
        "generated_sql_redacted": redact_sql_literals(generated_sql),
        "referenced_tables": list(referenced_tables),
        "referenced_columns": list(referenced_columns),
        "warnings": list(warnings),
        "attempts": _redact_attempts(attempts),
        "parameter_count": parameter_count,
    }
    if row_count is not None:
        event["row_count"] = row_count
    if execution_ms is not None:
        event["execution_ms"] = execution_ms
    if error:
        event["error"] = error
    return event


def _log_audit(event: dict[str, Any]) -> None:
    logger.info("database_retriever audit: %s", event)


def _parameter_count(query_parameters: Any) -> int:
    if query_parameters is None:
        return 0
    if isinstance(query_parameters, dict):
        return len(query_parameters)
    if isinstance(query_parameters, (list, tuple)):
        return len(query_parameters)
    return 1


def _normalize_query_parameters(query_parameters: Any) -> dict[str, Any] | tuple[Any, ...]:
    if query_parameters is None:
        return ()
    if isinstance(query_parameters, dict):
        return query_parameters
    if isinstance(query_parameters, (list, tuple)):
        return tuple(query_parameters)
    raise DatabaseRetrievalError("query_parameters must be an array or object")


def _normalize_max_cell_chars(max_cell_chars: int | None) -> int:
    return max(1, min(int(max_cell_chars or DEFAULT_MAX_CELL_CHARS), HARD_MAX_CELL_CHARS))


def _normalize_include_columns(include_columns: Any) -> tuple[str, ...] | None:
    if include_columns is None:
        return None
    if not isinstance(include_columns, (list, tuple)):
        raise DatabaseRetrievalError("include_columns must be an array")
    columns = tuple(str(column).strip() for column in include_columns if str(column).strip())
    return columns or None


def _shape_cell(value: Any, max_cell_chars: int) -> tuple[Any, bool]:
    if isinstance(value, str):
        if len(value) <= max_cell_chars:
            return value, False
        return value[:max_cell_chars] + "...", True
    if isinstance(value, bytes | bytearray | memoryview):
        byte_count = len(value)
        return f"[binary data: {byte_count} bytes]", False
    return value, False


def shape_result_rows(
    rows: list[dict[str, Any]],
    *,
    include_columns: Any = None,
    max_cell_chars: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    max_chars = _normalize_max_cell_chars(max_cell_chars)
    requested_columns = _normalize_include_columns(include_columns)
    available_columns = list(rows[0].keys()) if rows else []
    if requested_columns is not None:
        missing = [
            column
            for column in requested_columns
            if available_columns and column not in available_columns
        ]
        if missing:
            raise DatabaseRetrievalError(
                "include_columns references columns not present in result: "
                + ", ".join(missing)
            )
        output_columns = list(requested_columns)
    else:
        output_columns = available_columns

    truncated_cell_count = 0
    shaped_rows = []
    for row in rows:
        shaped_row = {}
        for column in output_columns:
            shaped_value, truncated = _shape_cell(row.get(column), max_chars)
            shaped_row[column] = shaped_value
            if truncated:
                truncated_cell_count += 1
        shaped_rows.append(shaped_row)

    metadata = {
        "columns": available_columns,
        "included_columns": output_columns,
        "omitted_column_count": max(0, len(available_columns) - len(output_columns)),
        "max_cell_chars": max_chars,
        "truncated_cell_count": truncated_cell_count,
    }
    return shaped_rows, metadata


def get_database_connections_path() -> Path:
    return get_hermes_home() / "database_connections.yaml"


def _import_duckdb():
    try:
        import duckdb
    except Exception as exc:
        raise DatabaseRetrievalError(
            "DuckDB execution requires optional dependency 'duckdb'"
        ) from exc
    return duckdb


def _import_asyncpg():
    try:
        import asyncpg
    except Exception as exc:
        raise DatabaseRetrievalError(
            "PostgreSQL execution requires optional dependency 'asyncpg'"
        ) from exc
    return asyncpg


def _duckdb_available() -> bool:
    try:
        _import_duckdb()
        return True
    except DatabaseRetrievalError:
        return False


def _postgres_available() -> bool:
    try:
        _import_asyncpg()
        return True
    except DatabaseRetrievalError:
        return False


def _can_execute_dialect(dialect: str) -> bool:
    if dialect in ALWAYS_EXECUTABLE_DIALECTS:
        return True
    if dialect == "duckdb":
        return _duckdb_available()
    if dialect == "postgres":
        return _postgres_available()
    return False


def _executable_dialects() -> tuple[str, ...]:
    return tuple(
        dialect
        for dialect in SUPPORTED_SQLGLOT_DIALECTS
        if _can_execute_dialect(dialect)
    )


def get_dialect_capabilities() -> dict[str, DialectCapability]:
    return {
        name: DialectCapability(
            name=name,
            can_parse=True,
            can_transpile=True,
            can_execute=_can_execute_dialect(name),
            supports_limit=True,
            supports_explain=name in {"sqlite", "postgres", "mysql", "duckdb"},
            connector_type=name if _can_execute_dialect(name) else None,
        )
        for name in SUPPORTED_SQLGLOT_DIALECTS
    }


def inspect_dialect_capabilities() -> dict[str, Any]:
    capabilities = get_dialect_capabilities()
    result = {
        "success": True,
        "read_only": True,
        "default_source_dialect": DEFAULT_SOURCE_DIALECT,
        "supported_dialects": list(SUPPORTED_SQLGLOT_DIALECTS),
        "executable_dialects": list(_executable_dialects()),
        "dialects": {
            name: asdict(capability)
            for name, capability in capabilities.items()
        },
    }
    result["audit"] = build_audit_event(
        action="dialects",
        database_id="",
    )
    _log_audit(result["audit"])
    return result


def _normalize_dialect(dialect: str | None) -> str:
    value = (dialect or "").strip().lower()
    if not value:
        return DEFAULT_SOURCE_DIALECT
    if value not in SUPPORTED_SQLGLOT_DIALECTS:
        raise DatabaseRetrievalError(f"Unsupported SQL dialect: {value}")
    return value


def parse_single_statement(sql: str, *, source_dialect: str = DEFAULT_SOURCE_DIALECT):
    if sqlglot is None:
        raise DatabaseRetrievalError("sqlglot is not installed")
    source = _normalize_dialect(source_dialect)
    statements = sqlglot.parse(sql, read=source)
    statements = [stmt for stmt in statements if stmt is not None]
    if len(statements) != 1:
        raise DatabaseRetrievalError("Exactly one SQL statement is allowed")
    return statements[0]


def _mutation_expression_types() -> tuple[type, ...]:
    names = (
        "Insert",
        "Update",
        "Delete",
        "Drop",
        "Create",
        "Alter",
        "TruncateTable",
        "Merge",
        "Replace",
        "Command",
        "Copy",
        "Grant",
        "Revoke",
        "Execute",
    )
    return tuple(getattr(exp, name) for name in names if hasattr(exp, name))


def _is_read_only_root(expression) -> bool:
    read_only_roots = (exp.Select, exp.Union, exp.Intersect, exp.Except)
    return isinstance(expression, read_only_roots)


def _cte_aliases(expression) -> set[str]:
    aliases: set[str] = set()
    for cte in expression.find_all(exp.CTE):
        alias = cte.alias
        if alias:
            aliases.add(alias.lower())
    return aliases


def _referenced_tables(expression) -> tuple[str, ...]:
    cte_aliases = _cte_aliases(expression)
    tables = {
        table.name.lower()
        for table in expression.find_all(exp.Table)
        if table.name and table.name.lower() not in cte_aliases
    }
    return tuple(sorted(tables))


def _table_aliases(expression) -> dict[str, str]:
    cte_aliases = _cte_aliases(expression)
    aliases: dict[str, str] = {}
    for table in expression.find_all(exp.Table):
        table_name = table.name.lower() if table.name else ""
        if not table_name or table_name in cte_aliases:
            continue
        aliases[table_name] = table_name
        if table.alias:
            aliases[table.alias.lower()] = table_name
    return aliases


def _referenced_columns(
    expression,
    *,
    table_aliases: dict[str, str] | None = None,
) -> tuple[str, ...]:
    aliases = table_aliases or {}
    columns = set()
    for column in expression.find_all(exp.Column):
        name = column.name.lower() if column.name else ""
        if not name:
            continue
        table = column.table.lower() if column.table else ""
        table = aliases.get(table, table)
        columns.add(f"{table}.{name}" if table else name)
    return tuple(sorted(columns))


def _literal_int(node) -> int | None:
    if node is None:
        return None
    expression = getattr(node, "expression", None)
    if expression is None:
        return None
    try:
        return int(expression.name)
    except (TypeError, ValueError):
        return None


def _normalize_max_rows(max_rows: int | None) -> int:
    return max(1, min(int(max_rows or DEFAULT_MAX_ROWS), HARD_MAX_ROWS))


def _enforce_limit(expression, max_rows: int) -> tuple[Any, tuple[str, ...]]:
    max_rows = _normalize_max_rows(max_rows)
    warnings: list[str] = []

    existing = _literal_int(expression.args.get("limit"))
    if existing is None:
        warnings.append(f"LIMIT {max_rows} was added")
        return expression.limit(max_rows, copy=True), tuple(warnings)
    if existing > max_rows:
        warnings.append(f"LIMIT was capped from {existing} to {max_rows}")
        return expression.limit(max_rows, copy=True), tuple(warnings)
    return expression, tuple(warnings)


def _normalize_offset(offset: int | None) -> int | None:
    if offset is None:
        return None
    normalized = int(offset)
    if normalized < 0:
        raise DatabaseRetrievalError("offset must be greater than or equal to 0")
    return normalized


def _pagination_metadata(
    *,
    max_rows: int,
    offset: int | None,
    returned_row_count: int | None = None,
) -> dict[str, Any]:
    limit = _normalize_max_rows(max_rows)
    normalized_offset = _normalize_offset(offset) or 0
    metadata: dict[str, Any] = {
        "limit": limit,
        "offset": normalized_offset,
    }
    if returned_row_count is None:
        return metadata
    metadata["returned"] = returned_row_count
    metadata["next_offset"] = (
        normalized_offset + returned_row_count
        if returned_row_count == limit
        else None
    )
    return metadata


def _enforce_offset(expression, offset: int | None) -> tuple[Any, tuple[str, ...]]:
    normalized_offset = _normalize_offset(offset)
    if normalized_offset is None:
        return expression, ()

    existing = _literal_int(expression.args.get("offset"))
    if existing == normalized_offset:
        return expression, ()

    warnings: list[str] = []
    if existing is None:
        warnings.append(f"OFFSET {normalized_offset} was added")
    else:
        warnings.append(f"OFFSET was changed from {existing} to {normalized_offset}")
    return expression.offset(normalized_offset, copy=True), tuple(warnings)


def validate_read_only_sql(
    sql: str,
    *,
    source_dialect: str = DEFAULT_SOURCE_DIALECT,
    target_dialect: str = "sqlite",
    allowed_tables: set[str] | None = None,
    allowed_columns: dict[str, set[str]] | None = None,
    max_rows: int = DEFAULT_MAX_ROWS,
    offset: int | None = None,
) -> ValidationResult:
    source = _normalize_dialect(source_dialect)
    target = _normalize_dialect(target_dialect)
    expression = parse_single_statement(sql, source_dialect=source)

    if not _is_read_only_root(expression):
        raise DatabaseRetrievalError(
            f"Only read-only SELECT queries are allowed, got {expression.key.upper()}"
        )

    blocked_types = _mutation_expression_types()
    for node in expression.walk():
        if isinstance(node, blocked_types):
            raise DatabaseRetrievalError(
                f"Blocked non-read-only SQL expression: {node.key.upper()}"
            )

    tables = _referenced_tables(expression)
    if allowed_tables is not None:
        allowed = {name.lower() for name in allowed_tables}
        unknown = sorted(set(tables) - allowed)
        if unknown:
            raise DatabaseRetrievalError(
                "Query references tables outside the allowlist: " + ", ".join(unknown)
            )

    columns = _referenced_columns(expression, table_aliases=_table_aliases(expression))
    if allowed_columns is not None:
        normalized_columns = {
            table.lower(): {column.lower() for column in cols}
            for table, cols in allowed_columns.items()
        }
        for column in columns:
            if "." not in column:
                candidate_tables = tables or tuple(normalized_columns)
                if not any(
                    column in normalized_columns.get(table, set())
                    for table in candidate_tables
                ):
                    raise DatabaseRetrievalError(
                        f"Query references column outside the allowlist: {column}"
                    )
                continue
            table, name = column.split(".", 1)
            if table in normalized_columns and name not in normalized_columns[table]:
                raise DatabaseRetrievalError(
                    f"Query references column outside the allowlist: {table}.{name}"
                )

    limited_expression, limit_warnings = _enforce_limit(expression, max_rows)
    paged_expression, offset_warnings = _enforce_offset(limited_expression, offset)
    return ValidationResult(
        sql=paged_expression.sql(dialect=target),
        source_dialect=source,
        target_dialect=target,
        referenced_tables=tables,
        referenced_columns=columns,
        warnings=limit_warnings + offset_warnings,
    )


def _load_connections(path: Path | None = None) -> dict[str, dict[str, Any]]:
    config_path = path or get_database_connections_path()
    if not config_path.exists():
        return {}
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    connections = data.get("connections", data)
    if not isinstance(connections, dict):
        raise DatabaseRetrievalError("database_connections.yaml must contain a mapping")
    return {
        str(name): details
        for name, details in connections.items()
        if isinstance(details, dict)
    }


def _redact_connection_value(key: str, value: Any) -> Any:
    lowered = key.lower()
    secret_markers = (
        "password",
        "token",
        "api_key",
        "secret",
        "dsn",
        "url",
        "connection_string",
        "private_key",
    )
    if any(marker in lowered for marker in secret_markers):
        return "[redacted]"
    if isinstance(value, dict):
        return {
            str(child_key): _redact_connection_value(str(child_key), child_value)
            for child_key, child_value in value.items()
        }
    if isinstance(value, list):
        return [_redact_connection_value(key, item) for item in value]
    return value


def list_database_connections(
    connections_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Return configured connection metadata without exposing secrets."""
    safe_connections = {}
    for database_id, config in _load_connections(connections_path).items():
        safe = {}
        for key, value in config.items():
            safe[key] = _redact_connection_value(key, value)
        safe_connections[database_id] = safe
    return safe_connections


def inspect_database_connections(
    *,
    connections_path: Path | None = None,
) -> dict[str, Any]:
    connections = list_database_connections(connections_path)
    result = {
        "success": True,
        "read_only": True,
        "connection_ids": sorted(connections),
        "connection_count": len(connections),
        "connections": connections,
    }
    result["audit"] = build_audit_event(
        action="connections",
        database_id="",
        referenced_tables=(),
    )
    _log_audit(result["audit"])
    return result


def _sqlite_schema(database_id: str, db_path: str) -> DatabaseSchema:
    uri = f"file:{Path(db_path).expanduser()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        table_rows = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        tables = []
        for (table_name,) in table_rows:
            columns = []
            for row in conn.execute(f"PRAGMA table_info({table_name!r})").fetchall():
                _, name, col_type, notnull, _, pk = row
                columns.append(
                    ColumnSchema(
                        name=str(name),
                        type=str(col_type or ""),
                        nullable=not bool(notnull),
                        primary_key=bool(pk),
                    )
                )
            tables.append(TableSchema(name=str(table_name), columns=tuple(columns)))
        return DatabaseSchema(database_id=database_id, dialect="sqlite", tables=tuple(tables))
    finally:
        conn.close()


def _configured_column(name: str, raw: Any) -> ColumnSchema:
    if isinstance(raw, dict):
        column_name = str(raw.get("name") or name)
        return ColumnSchema(
            name=column_name,
            type=str(raw.get("type") or raw.get("data_type") or ""),
            nullable=bool(raw.get("nullable", True)),
            primary_key=bool(raw.get("primary_key", raw.get("pk", False))),
        )
    return ColumnSchema(name=str(name), type=str(raw or ""))


def _configured_columns(raw_columns: Any) -> tuple[ColumnSchema, ...]:
    columns: list[ColumnSchema] = []
    if isinstance(raw_columns, dict):
        for column_name, raw_column in raw_columns.items():
            columns.append(_configured_column(str(column_name), raw_column))
    elif isinstance(raw_columns, list):
        for raw_column in raw_columns:
            if isinstance(raw_column, dict):
                column_name = raw_column.get("name")
                if column_name:
                    columns.append(_configured_column(str(column_name), raw_column))
            elif raw_column:
                columns.append(ColumnSchema(name=str(raw_column), type=""))
    return tuple(columns)


def _configured_table(name: str, raw: Any) -> TableSchema | None:
    if isinstance(raw, dict):
        table_name = str(raw.get("name") or name)
        raw_columns = raw.get("columns")
        if raw_columns is None:
            raw_columns = raw.get("fields")
    else:
        table_name = str(name)
        raw_columns = raw
    if not table_name:
        return None
    return TableSchema(name=table_name, columns=_configured_columns(raw_columns))


def _configured_schema(
    *,
    database_id: str,
    dialect: str,
    config: dict[str, Any],
) -> DatabaseSchema | None:
    raw_schema = config.get("schema")
    raw_tables = None
    if isinstance(raw_schema, dict):
        raw_tables = raw_schema.get("tables")
    elif isinstance(raw_schema, list):
        raw_tables = raw_schema
    if raw_tables is None:
        raw_tables = config.get("tables")
    if raw_tables is None:
        return None

    tables: list[TableSchema] = []
    if isinstance(raw_tables, dict):
        for table_name, raw_table in raw_tables.items():
            table = _configured_table(str(table_name), raw_table)
            if table:
                tables.append(table)
    elif isinstance(raw_tables, list):
        for raw_table in raw_tables:
            if not isinstance(raw_table, dict):
                continue
            table_name = raw_table.get("name")
            if not table_name:
                continue
            table = _configured_table(str(table_name), raw_table)
            if table:
                tables.append(table)

    return DatabaseSchema(
        database_id=database_id,
        dialect=dialect,
        tables=tuple(tables),
    )


def _configured_allowlists(
    config: dict[str, Any],
) -> tuple[set[str] | None, dict[str, set[str]] | None]:
    raw_tables = config.get("allow_tables") or config.get("allowed_tables")
    allowed_tables = None
    if isinstance(raw_tables, list):
        allowed_tables = {str(name).lower() for name in raw_tables}

    raw_columns = config.get("allow_columns") or config.get("allowed_columns")
    allowed_columns = None
    if isinstance(raw_columns, dict):
        allowed_columns = {
            str(table).lower(): {str(column).lower() for column in columns}
            for table, columns in raw_columns.items()
            if isinstance(columns, list)
        }
    return allowed_tables, allowed_columns


def _filter_schema(
    schema: DatabaseSchema,
    *,
    allowed_tables: set[str] | None = None,
    allowed_columns: dict[str, set[str]] | None = None,
) -> DatabaseSchema:
    tables = []
    for table in schema.tables:
        table_key = table.name.lower()
        if allowed_tables is not None and table_key not in allowed_tables:
            continue
        columns = table.columns
        if allowed_columns is not None and table_key in allowed_columns:
            allowed = allowed_columns[table_key]
            columns = tuple(
                column for column in columns if column.name.lower() in allowed
            )
        tables.append(TableSchema(name=table.name, columns=columns))
    return DatabaseSchema(
        database_id=schema.database_id,
        dialect=schema.dialect,
        tables=tuple(tables),
    )


def _allowed_from_schema(schema: DatabaseSchema) -> tuple[set[str], dict[str, set[str]]]:
    tables = {table.name.lower() for table in schema.tables}
    columns = {
        table.name.lower(): {column.name.lower() for column in table.columns}
        for table in schema.tables
    }
    return tables, columns


def _load_database_schema(
    database_id: str,
    *,
    connections_path: Path | None = None,
) -> tuple[DatabaseSchema, dict[str, Any]]:
    connections = _load_connections(connections_path)
    if database_id not in connections:
        raise DatabaseRetrievalError(f"Unknown database_id: {database_id}")

    config = connections[database_id]
    dialect = _normalize_dialect(config.get("dialect", "sqlite"))
    configured_schema = _configured_schema(
        database_id=database_id,
        dialect=dialect,
        config=config,
    )
    if configured_schema is not None:
        schema = configured_schema
    elif dialect == "sqlite":
        db_path = config.get("path")
        if not db_path:
            raise DatabaseRetrievalError(
                f"SQLite database '{database_id}' is missing path"
            )
        schema = _sqlite_schema(database_id, db_path)
    else:
        raise DatabaseRetrievalError(
            "Schema introspection for dialect "
            f"'{dialect}' requires configured schema metadata"
        )
    allowed_tables, allowed_columns = _configured_allowlists(config)
    schema = _filter_schema(
        schema,
        allowed_tables=allowed_tables,
        allowed_columns=allowed_columns,
    )
    return schema, config


def build_schema_context(schema: DatabaseSchema) -> str:
    """Render compact schema context for SQL generation."""
    lines = [
        f"Database ID: {schema.database_id}",
        f"Dialect: {schema.dialect}",
        "Access policy: read-only SELECT queries only. Use only listed tables and columns.",
        "Tables:",
    ]
    if not schema.tables:
        lines.append("- [no accessible tables]")
        return "\n".join(lines)

    for table in schema.tables:
        column_parts = []
        for column in table.columns:
            flags = []
            if column.primary_key:
                flags.append("primary key")
            if not column.nullable:
                flags.append("not null")
            suffix = f" ({', '.join(flags)})" if flags else ""
            column_parts.append(f"{column.name} {column.type or 'UNKNOWN'}{suffix}")
        lines.append(f"- {table.name}: " + ", ".join(column_parts))
    return "\n".join(lines)


def build_sql_generation_messages(
    *,
    user_request: str,
    schema_context: str,
    source_dialect: str = DEFAULT_SOURCE_DIALECT,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> list[dict[str, str]]:
    max_rows = _normalize_max_rows(max_rows)
    system_prompt = (
        "You generate SQL for a read-only database retrieval tool. "
        "Return exactly one SQL statement and nothing else. "
        "Do not use markdown fences, comments, prose, or explanations. "
        f"Use {source_dialect} syntax as the source dialect. "
        "Only SELECT queries are allowed, including WITH ... SELECT CTEs. "
        "Never use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE, "
        "MERGE, REPLACE, COPY, CALL, EXEC, GRANT, or REVOKE. "
        "Use only the tables and columns listed in the schema context. "
        f"Include a LIMIT no greater than {max_rows} unless the query is an aggregate "
        "that naturally returns fewer rows."
    )
    user_prompt = (
        f"{schema_context}\n\n"
        f"User request:\n{user_request}\n\n"
        "SQL:"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_sql_repair_messages(
    *,
    user_request: str,
    schema_context: str,
    previous_sql: str,
    error: str,
    source_dialect: str = DEFAULT_SOURCE_DIALECT,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> list[dict[str, str]]:
    messages = build_sql_generation_messages(
        user_request=user_request,
        schema_context=schema_context,
        source_dialect=source_dialect,
        max_rows=max_rows,
    )
    messages.append(
        {
            "role": "assistant",
            "content": previous_sql,
        }
    )
    messages.append(
        {
            "role": "user",
            "content": (
                "The previous SQL failed. Return one corrected read-only SQL "
                "statement and nothing else.\n\n"
                f"Error:\n{error}"
            ),
        }
    )
    return messages


def _strip_sql_response(text: str) -> str:
    stripped = (text or "").strip()
    if not stripped:
        raise DatabaseRetrievalError("LLM returned empty SQL")
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def _call_sql_generation_llm(messages: list[dict[str, str]]) -> str:
    from agent.auxiliary_client import call_llm, extract_content_or_reasoning

    response = call_llm(
        task=SQL_GENERATION_TASK,
        messages=messages,
        temperature=0,
        max_tokens=1200,
    )
    return extract_content_or_reasoning(response)


def generate_sql_for_request(
    *,
    database_id: str,
    user_request: str,
    source_dialect: str = DEFAULT_SOURCE_DIALECT,
    max_rows: int = DEFAULT_MAX_ROWS,
    offset: int | None = None,
    connections_path: Path | None = None,
    llm_sql_generator=None,
    repair_attempts: int = DEFAULT_REPAIR_ATTEMPTS,
) -> GeneratedSql:
    schema, _config = _load_database_schema(
        database_id,
        connections_path=connections_path,
    )
    source = _normalize_dialect(source_dialect)
    schema_context = build_schema_context(schema)
    messages = build_sql_generation_messages(
        user_request=user_request,
        schema_context=schema_context,
        source_dialect=source,
        max_rows=max_rows,
    )
    generator = llm_sql_generator or _call_sql_generation_llm
    allowed_tables, allowed_columns = _allowed_from_schema(schema)
    attempts: list[SqlAttempt] = []
    current_messages = messages
    max_attempts = 1 + max(0, int(repair_attempts or 0))
    last_error = ""

    for _attempt_index in range(max_attempts):
        raw_sql = _strip_sql_response(generator(current_messages))
        try:
            validation = validate_read_only_sql(
                raw_sql,
                source_dialect=source,
                target_dialect=schema.dialect,
                allowed_tables=allowed_tables,
                allowed_columns=allowed_columns,
                max_rows=max_rows,
                offset=offset,
            )
            attempts.append(SqlAttempt(sql=raw_sql))
            return GeneratedSql(
                user_request=user_request,
                source_dialect=source,
                target_dialect=schema.dialect,
                raw_sql=raw_sql,
                sql=validation.sql,
                referenced_tables=validation.referenced_tables,
                referenced_columns=validation.referenced_columns,
                warnings=validation.warnings,
                attempts=tuple(attempt.to_dict() for attempt in attempts),
            )
        except Exception as exc:
            last_error = str(exc)
            attempts.append(SqlAttempt(sql=raw_sql, error=last_error))
            current_messages = build_sql_repair_messages(
                user_request=user_request,
                schema_context=schema_context,
                previous_sql=raw_sql,
                error=last_error,
                source_dialect=source,
                max_rows=max_rows,
            )

    raise DatabaseRetrievalError(
        f"SQL generation failed after {max_attempts} attempt(s): {last_error}"
    )


def inspect_database(
    *,
    database_id: str,
    connections_path: Path | None = None,
) -> dict[str, Any]:
    schema, config = _load_database_schema(
        database_id,
        connections_path=connections_path,
    )
    capabilities = get_dialect_capabilities()[schema.dialect]
    result = {
        "success": True,
        "database_id": database_id,
        "dialect": schema.dialect,
        "can_execute": capabilities.can_execute,
        "schema": asdict(schema),
        "schema_context": build_schema_context(schema),
        "read_only": True,
        "allow_tables": sorted(_allowed_from_schema(schema)[0]),
        "connection": {
            "dialect": config.get("dialect", "sqlite"),
            "path": config.get("path") if schema.dialect == "sqlite" else "[redacted]",
        },
    }
    result["audit"] = build_audit_event(
        action="schema",
        database_id=database_id,
        target_dialect=schema.dialect,
        referenced_tables=result["allow_tables"],
    )
    _log_audit(result["audit"])
    return result


def _connection_safe_path(dialect: str, config: dict[str, Any]) -> str:
    if dialect in {"sqlite", "duckdb"}:
        return str(config.get("path") or "")
    return "[redacted]"


def test_database_connection(
    *,
    database_id: str,
    connections_path: Path | None = None,
) -> dict[str, Any]:
    connections = _load_connections(connections_path)
    if database_id not in connections:
        raise DatabaseRetrievalError(f"Unknown database_id: {database_id}")

    config = connections[database_id]
    dialect = _normalize_dialect(config.get("dialect", "sqlite"))
    capabilities = get_dialect_capabilities()[dialect]
    schema: DatabaseSchema | None = None
    execution_checked = False
    probe_rows: list[dict[str, Any]] = []

    if dialect == "sqlite":
        db_path = config.get("path")
        if not db_path:
            raise DatabaseRetrievalError("SQLite database connection is missing path")
        schema = _sqlite_schema(database_id, db_path)
        probe_rows = execute_sqlite_read_only(db_path, "SELECT 1 AS healthcheck_ok")
        execution_checked = True
    elif dialect == "duckdb" and _can_execute_dialect(dialect):
        db_path = config.get("path")
        if not db_path:
            raise DatabaseRetrievalError("DuckDB database connection is missing path")
        configured_schema = _configured_schema(
            database_id=database_id,
            dialect=dialect,
            config=config,
        )
        schema = configured_schema
        probe_rows = execute_duckdb_read_only(db_path, "SELECT 1 AS healthcheck_ok")
        execution_checked = True
    elif dialect == "postgres" and _can_execute_dialect(dialect):
        schema = _configured_schema(
            database_id=database_id,
            dialect=dialect,
            config=config,
        )
        probe_rows = execute_postgres_read_only(
            config,
            "SELECT 1 AS healthcheck_ok",
        )
        execution_checked = True
    else:
        schema = _configured_schema(
            database_id=database_id,
            dialect=dialect,
            config=config,
        )
        if schema is None:
            raise DatabaseRetrievalError(
                f"{dialect.upper()} connection healthcheck requires configured schema "
                "metadata until an executable connector is implemented"
            )

    table_names = [table.name for table in schema.tables] if schema else []
    result = {
        "success": True,
        "database_id": database_id,
        "dialect": dialect,
        "read_only": True,
        "status": "ok",
        "can_execute": capabilities.can_execute,
        "execution_checked": execution_checked,
        "schema_available": schema is not None,
        "table_count": len(table_names),
        "table_names": table_names,
        "probe": {
            "executed": execution_checked,
            "row_count": len(probe_rows),
        },
        "connection": {
            "dialect": dialect,
            "path": _connection_safe_path(dialect, config),
        },
    }
    result["audit"] = build_audit_event(
        action="test_connection",
        database_id=database_id,
        target_dialect=dialect,
        row_count=len(probe_rows) if execution_checked else None,
        referenced_tables=table_names,
    )
    _log_audit(result["audit"])
    return result


test_database_connection.__test__ = False


def validate_database_query(
    *,
    database_id: str,
    sql: str,
    source_dialect: str = DEFAULT_SOURCE_DIALECT,
    max_rows: int = DEFAULT_MAX_ROWS,
    offset: int | None = None,
    query_parameters: Any = None,
    connections_path: Path | None = None,
) -> dict[str, Any]:
    schema, _config = _load_database_schema(
        database_id,
        connections_path=connections_path,
    )
    allowed_tables, allowed_columns = _allowed_from_schema(schema)
    validation = validate_read_only_sql(
        sql,
        source_dialect=source_dialect,
        target_dialect=schema.dialect,
        allowed_tables=allowed_tables,
        allowed_columns=allowed_columns,
        max_rows=max_rows,
        offset=offset,
    )
    result = {
        "success": True,
        "database_id": database_id,
        "source_dialect": validation.source_dialect,
        "target_dialect": validation.target_dialect,
        "sql": validation.sql,
        "referenced_tables": list(validation.referenced_tables),
        "referenced_columns": list(validation.referenced_columns),
        "warnings": list(validation.warnings),
        "read_only": True,
        "executed": False,
        "offset": _normalize_offset(offset) or 0,
        "pagination": _pagination_metadata(max_rows=max_rows, offset=offset),
        "parameter_count": _parameter_count(query_parameters),
        "schema": asdict(schema),
        "schema_context": build_schema_context(schema),
    }
    result["audit"] = build_audit_event(
        action="validate",
        database_id=database_id,
        source_dialect=validation.source_dialect,
        target_dialect=validation.target_dialect,
        sql=validation.sql,
        referenced_tables=result["referenced_tables"],
        referenced_columns=result["referenced_columns"],
        warnings=result["warnings"],
        parameter_count=result["parameter_count"],
    )
    _log_audit(result["audit"])
    return result


def _explain_sql_for_dialect(dialect: str, sql: str) -> str:
    if dialect == "sqlite":
        return f"EXPLAIN QUERY PLAN {sql}"
    if dialect == "duckdb":
        return f"EXPLAIN {sql}"
    if dialect == "postgres":
        return f"EXPLAIN (FORMAT JSON) {sql}"
    raise DatabaseRetrievalError(
        f"Execution plan support for dialect '{dialect}' is not implemented yet"
    )


_SQLITE_READ_ONLY_ACTIONS = {
    action
    for action in (
        getattr(sqlite3, "SQLITE_SELECT", None),
        getattr(sqlite3, "SQLITE_READ", None),
        getattr(sqlite3, "SQLITE_FUNCTION", None),
        getattr(sqlite3, "SQLITE_RECURSIVE", None),
    )
    if action is not None
}


def _sqlite_read_only_authorizer(action, _arg1, _arg2, _database, _trigger) -> int:
    if action in _SQLITE_READ_ONLY_ACTIONS:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def execute_sqlite_read_only(
    db_path: str,
    sql: str,
    *,
    query_parameters: Any = None,
    timeout_seconds: float = 10.0,
):
    uri = f"file:{Path(db_path).expanduser()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=timeout_seconds)
    conn.row_factory = sqlite3.Row
    conn.set_authorizer(_sqlite_read_only_authorizer)
    try:
        rows = conn.execute(sql, _normalize_query_parameters(query_parameters)).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def execute_duckdb_read_only(db_path: str, sql: str, *, query_parameters: Any = None):
    duckdb = _import_duckdb()
    conn = duckdb.connect(str(Path(db_path).expanduser()), read_only=True)
    try:
        cursor = conn.execute(sql, _normalize_query_parameters(query_parameters))
        columns = [column[0] for column in (cursor.description or [])]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    finally:
        conn.close()


def _postgres_connection_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    for key in ("dsn", "host", "port", "user", "password", "database", "ssl"):
        if config.get(key) is not None:
            kwargs[key] = config[key]
    if "database" not in kwargs and config.get("dbname") is not None:
        kwargs["database"] = config["dbname"]
    if config.get("timeout") is not None:
        kwargs["timeout"] = float(config["timeout"])
    return kwargs


def _normalize_postgres_query_parameters(query_parameters: Any) -> tuple[Any, ...]:
    normalized = _normalize_query_parameters(query_parameters)
    if isinstance(normalized, dict):
        raise DatabaseRetrievalError(
            "PostgreSQL query_parameters must be an array matching $1, $2 placeholders"
        )
    return normalized


async def _execute_postgres_read_only_async(
    config: dict[str, Any],
    sql: str,
    *,
    query_parameters: Any = None,
) -> list[dict[str, Any]]:
    asyncpg = _import_asyncpg()
    parameters = _normalize_postgres_query_parameters(query_parameters)
    conn = await asyncpg.connect(**_postgres_connection_kwargs(config))
    try:
        await conn.execute("BEGIN READ ONLY")
        rows = await conn.fetch(sql, *parameters)
        await conn.execute("ROLLBACK")
        return [dict(row) for row in rows]
    except Exception:
        try:
            await conn.execute("ROLLBACK")
        except Exception:
            pass
        raise
    finally:
        await conn.close()


def execute_postgres_read_only(
    config: dict[str, Any],
    sql: str,
    *,
    query_parameters: Any = None,
) -> list[dict[str, Any]]:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            _execute_postgres_read_only_async(
                config,
                sql,
                query_parameters=query_parameters,
            )
        )
    raise DatabaseRetrievalError(
        "PostgreSQL execution cannot run inside an active event loop"
    )


def execute_read_only(
    *,
    dialect: str,
    config: dict[str, Any],
    sql: str,
    query_parameters: Any = None,
) -> list[dict[str, Any]]:
    if dialect == "postgres":
        return execute_postgres_read_only(
            config,
            sql,
            query_parameters=query_parameters,
        )
    db_path = config.get("path")
    if not db_path:
        raise DatabaseRetrievalError(
            f"{dialect.upper()} database connection is missing path"
    )
    if dialect == "sqlite":
        return execute_sqlite_read_only(
            db_path,
            sql,
            query_parameters=query_parameters,
        )
    if dialect == "duckdb":
        return execute_duckdb_read_only(
            db_path,
            sql,
            query_parameters=query_parameters,
        )
    raise DatabaseRetrievalError(
        f"Execution for dialect '{dialect}' is not implemented yet"
    )


def explain_database_query(
    *,
    database_id: str,
    sql: str,
    source_dialect: str = DEFAULT_SOURCE_DIALECT,
    max_rows: int = DEFAULT_MAX_ROWS,
    offset: int | None = None,
    query_parameters: Any = None,
    connections_path: Path | None = None,
) -> dict[str, Any]:
    schema, config = _load_database_schema(
        database_id,
        connections_path=connections_path,
    )
    target_dialect = schema.dialect
    capabilities = get_dialect_capabilities()[target_dialect]
    if not capabilities.supports_explain:
        raise DatabaseRetrievalError(
            f"Execution plan support for dialect '{target_dialect}' is not available"
        )
    if not _can_execute_dialect(target_dialect):
        raise DatabaseRetrievalError(
            f"Execution plan support for dialect '{target_dialect}' is not implemented yet"
        )

    allowed_tables, allowed_columns = _allowed_from_schema(schema)
    validation = validate_read_only_sql(
        sql,
        source_dialect=source_dialect,
        target_dialect=target_dialect,
        allowed_tables=allowed_tables,
        allowed_columns=allowed_columns,
        max_rows=max_rows,
        offset=offset,
    )

    explain_sql = _explain_sql_for_dialect(target_dialect, validation.sql)
    start = time.monotonic()
    plan = execute_read_only(
        dialect=target_dialect,
        config=config,
        sql=explain_sql,
        query_parameters=query_parameters,
    )
    elapsed_ms = int((time.monotonic() - start) * 1000)
    result = {
        "success": True,
        "database_id": database_id,
        "source_dialect": validation.source_dialect,
        "target_dialect": validation.target_dialect,
        "sql": validation.sql,
        "explain_sql": explain_sql,
        "referenced_tables": list(validation.referenced_tables),
        "referenced_columns": list(validation.referenced_columns),
        "warnings": list(validation.warnings),
        "read_only": True,
        "data_query_executed": False,
        "offset": _normalize_offset(offset) or 0,
        "pagination": _pagination_metadata(max_rows=max_rows, offset=offset),
        "parameter_count": _parameter_count(query_parameters),
        "plan": plan,
        "plan_row_count": len(plan),
        "execution_ms": elapsed_ms,
        "schema": asdict(schema),
        "schema_context": build_schema_context(schema),
    }
    result["audit"] = build_audit_event(
        action="explain",
        database_id=database_id,
        source_dialect=validation.source_dialect,
        target_dialect=validation.target_dialect,
        sql=validation.sql,
        referenced_tables=result["referenced_tables"],
        referenced_columns=result["referenced_columns"],
        row_count=result["plan_row_count"],
        execution_ms=result["execution_ms"],
        warnings=result["warnings"],
        parameter_count=result["parameter_count"],
    )
    _log_audit(result["audit"])
    return result


def retrieve_database(
    *,
    database_id: str,
    sql: str,
    source_dialect: str = DEFAULT_SOURCE_DIALECT,
    max_rows: int = DEFAULT_MAX_ROWS,
    offset: int | None = None,
    query_parameters: Any = None,
    include_columns: Any = None,
    max_cell_chars: int | None = None,
    connections_path: Path | None = None,
) -> dict[str, Any]:
    schema, config = _load_database_schema(
        database_id,
        connections_path=connections_path,
    )
    target_dialect = schema.dialect
    if not _can_execute_dialect(target_dialect):
        raise DatabaseRetrievalError(
            f"Execution for dialect '{target_dialect}' is not implemented yet"
        )

    allowed_tables, allowed_columns = _allowed_from_schema(schema)
    validation = validate_read_only_sql(
        sql,
        source_dialect=source_dialect,
        target_dialect=target_dialect,
        allowed_tables=allowed_tables,
        allowed_columns=allowed_columns,
        max_rows=max_rows,
        offset=offset,
    )

    start = time.monotonic()
    rows = execute_read_only(
        dialect=target_dialect,
        config=config,
        sql=validation.sql,
        query_parameters=query_parameters,
    )
    elapsed_ms = int((time.monotonic() - start) * 1000)
    shaped_rows, result_shape = shape_result_rows(
        rows,
        include_columns=include_columns,
        max_cell_chars=max_cell_chars,
    )
    pagination = _pagination_metadata(
        max_rows=max_rows,
        offset=offset,
        returned_row_count=len(rows),
    )
    result = {
        "success": True,
        "database_id": database_id,
        "source_dialect": validation.source_dialect,
        "target_dialect": validation.target_dialect,
        "sql": validation.sql,
        "referenced_tables": list(validation.referenced_tables),
        "referenced_columns": list(validation.referenced_columns),
        "warnings": list(validation.warnings),
        "parameter_count": _parameter_count(query_parameters),
        "offset": _normalize_offset(offset) or 0,
        "pagination": pagination,
        "rows": shaped_rows,
        "row_count": len(rows),
        "result_shape": result_shape,
        "execution_ms": elapsed_ms,
        "schema": asdict(schema),
        "schema_context": build_schema_context(schema),
    }
    result["audit"] = build_audit_event(
        action="query",
        database_id=database_id,
        source_dialect=validation.source_dialect,
        target_dialect=validation.target_dialect,
        sql=validation.sql,
        referenced_tables=result["referenced_tables"],
        referenced_columns=result["referenced_columns"],
        row_count=result["row_count"],
        execution_ms=result["execution_ms"],
        warnings=result["warnings"],
        parameter_count=result["parameter_count"],
    )
    _log_audit(result["audit"])
    return result


def retrieve_database_for_request(
    *,
    database_id: str,
    user_request: str,
    source_dialect: str = DEFAULT_SOURCE_DIALECT,
    max_rows: int = DEFAULT_MAX_ROWS,
    offset: int | None = None,
    connections_path: Path | None = None,
    llm_sql_generator=None,
    query_parameters: Any = None,
    include_columns: Any = None,
    max_cell_chars: int | None = None,
    repair_attempts: int = DEFAULT_REPAIR_ATTEMPTS,
) -> dict[str, Any]:
    schema, config = _load_database_schema(
        database_id,
        connections_path=connections_path,
    )
    if not _can_execute_dialect(schema.dialect):
        raise DatabaseRetrievalError(
            f"Execution for dialect '{schema.dialect}' is not implemented yet"
        )

    generator = llm_sql_generator or _call_sql_generation_llm
    generated = generate_sql_for_request(
        database_id=database_id,
        user_request=user_request,
        source_dialect=source_dialect,
        max_rows=max_rows,
        offset=offset,
        connections_path=connections_path,
        llm_sql_generator=generator,
        repair_attempts=repair_attempts,
    )

    attempts = [dict(attempt) for attempt in generated.attempts]
    execution_error = ""
    max_repairs = max(0, int(repair_attempts or 0))
    for repair_index in range(max_repairs + 1):
        start = time.monotonic()
        try:
            rows = execute_read_only(
                dialect=schema.dialect,
                config=config,
                sql=generated.sql,
                query_parameters=query_parameters,
            )
            elapsed_ms = int((time.monotonic() - start) * 1000)
            shaped_rows, result_shape = shape_result_rows(
                rows,
                include_columns=include_columns,
                max_cell_chars=max_cell_chars,
            )
            pagination = _pagination_metadata(
                max_rows=max_rows,
                offset=offset,
                returned_row_count=len(rows),
            )
            result = {
                "success": True,
                "database_id": database_id,
                "user_request": user_request,
                "source_dialect": generated.source_dialect,
                "target_dialect": generated.target_dialect,
                "generated_sql": generated.raw_sql,
                "sql": generated.sql,
                "referenced_tables": list(generated.referenced_tables),
                "referenced_columns": list(generated.referenced_columns),
                "warnings": list(generated.warnings),
                "attempts": attempts,
                "parameter_count": _parameter_count(query_parameters),
                "offset": _normalize_offset(offset) or 0,
                "pagination": pagination,
                "rows": shaped_rows,
                "row_count": len(rows),
                "result_shape": result_shape,
                "execution_ms": elapsed_ms,
                "schema": asdict(schema),
                "schema_context": build_schema_context(schema),
            }
            result["audit"] = build_audit_event(
                action="retrieve",
                database_id=database_id,
                source_dialect=generated.source_dialect,
                target_dialect=generated.target_dialect,
                sql=generated.sql,
                generated_sql=generated.raw_sql,
                referenced_tables=result["referenced_tables"],
                referenced_columns=result["referenced_columns"],
                row_count=result["row_count"],
                execution_ms=result["execution_ms"],
                attempts=attempts,
                warnings=result["warnings"],
                parameter_count=result["parameter_count"],
            )
            _log_audit(result["audit"])
            return result
        except sqlite3.Error as exc:
            execution_error = str(exc)
            if attempts:
                attempts[-1]["error"] = execution_error
            else:
                attempts.append({"sql": generated.raw_sql, "error": execution_error})
            if repair_index >= max_repairs:
                break
            repair_messages = build_sql_repair_messages(
                user_request=user_request,
                schema_context=build_schema_context(schema),
                previous_sql=generated.raw_sql,
                error=f"Execution failed: {execution_error}",
                source_dialect=generated.source_dialect,
                max_rows=max_rows,
            )
            repaired_raw_sql = _strip_sql_response(generator(repair_messages))
            allowed_tables, allowed_columns = _allowed_from_schema(schema)
            validation = validate_read_only_sql(
                repaired_raw_sql,
                source_dialect=generated.source_dialect,
                target_dialect=generated.target_dialect,
                allowed_tables=allowed_tables,
                allowed_columns=allowed_columns,
                max_rows=max_rows,
                offset=offset,
            )
            attempts.append({"sql": repaired_raw_sql})
            generated = GeneratedSql(
                user_request=user_request,
                source_dialect=generated.source_dialect,
                target_dialect=generated.target_dialect,
                raw_sql=repaired_raw_sql,
                sql=validation.sql,
                referenced_tables=validation.referenced_tables,
                referenced_columns=validation.referenced_columns,
                warnings=validation.warnings,
                attempts=tuple(attempts),
            )

    audit = build_audit_event(
        action="retrieve",
        database_id=database_id,
        source_dialect=generated.source_dialect,
        target_dialect=generated.target_dialect,
        sql=generated.sql,
        generated_sql=generated.raw_sql,
        attempts=attempts,
        error=execution_error,
    )
    _log_audit(audit)
    raise DatabaseRetrievalError(
        "SQL execution failed after "
        f"{max_repairs + 1} attempt(s): {execution_error}"
    )


def check_database_retrieve_requirements() -> bool:
    return sqlglot is not None


DATABASE_RETRIEVE_SCHEMA = {
    "name": "database_retrieve",
    "description": (
        "List supported dialects or configured database ids, inspect schema, "
        "generate SQL from a user request, dry-run validate SQL, inspect an "
        "execution plan, or run validated read-only retrieval. Call "
        "action='dialects' when you need dialect capabilities, "
        "action='connections' when you need available database ids, "
        "action='test_connection' to verify a database id, and action='schema' "
        "before generating SQL when you need table context. Queries generate "
        "no writes: only SELECT/CTE SELECT statements are accepted, SQLGlot "
        "transpiles to the target dialect, and row limits are enforced before "
        "execution."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "dialects",
                    "connections",
                    "test_connection",
                    "healthcheck",
                    "schema",
                    "validate",
                    "explain",
                    "generate",
                    "query",
                    "retrieve",
                ],
                "description": (
                    "Use 'dialects' to list supported SQLGlot dialects and "
                    "execution capability; use 'connections' to list configured database ids without "
                    "secrets; use 'test_connection' or 'healthcheck' to verify "
                    "a configured database id using read-only checks; use "
                    "'schema' to inspect tables/columns before generating SQL; "
                    "use 'validate' to parse, allowlist-check, transpile, and "
                    "enforce limits without executing; use 'explain' to "
                    "validate supplied SQL and return a read-only execution "
                    "plan without running the data query; "
                    "use 'generate' to turn user_request into validated SQL without "
                    "executing; use 'query' to validate and execute supplied SQL; "
                    "use 'retrieve' to generate and execute from user_request. "
                    "Defaults to 'connections' when database_id, sql, and "
                    "user_request are omitted; otherwise 'schema' when only "
                    "database_id is present, 'query' when sql is present, and "
                    "'retrieve' when user_request is present."
                ),
            },
            "database_id": {
                "type": "string",
                "description": "Configured database id from ~/.hermes/database_connections.yaml. Required except for action='connections' and action='dialects'.",
            },
            "sql": {
                "type": "string",
                "description": "One read-only SELECT query to validate or execute.",
            },
            "query_parameters": {
                "oneOf": [
                    {"type": "array", "items": {}},
                    {"type": "object", "additionalProperties": {}},
                ],
                "description": (
                    "Optional bound parameters for placeholders in sql. "
                    "Use an array for positional placeholders or an object "
                    "for named placeholders. Parameter values are never "
                    "included in audit metadata."
                ),
            },
            "include_columns": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional response shaping list. When provided, returned "
                    "rows include only these result columns after query execution."
                ),
            },
            "max_cell_chars": {
                "type": "integer",
                "minimum": 1,
                "maximum": HARD_MAX_CELL_CHARS,
                "description": (
                    "Maximum characters returned for each string cell. Longer "
                    "values are truncated in the tool response."
                ),
            },
            "user_request": {
                "type": "string",
                "description": "Natural-language data retrieval request used by action='generate' or action='retrieve'.",
            },
            "source_dialect": {
                "type": "string",
                "enum": list(SUPPORTED_SQLGLOT_DIALECTS),
                "description": "Dialect of the supplied SQL. Defaults to postgres.",
            },
            "max_rows": {
                "type": "integer",
                "minimum": 1,
                "maximum": HARD_MAX_ROWS,
                "description": "Maximum rows to return. Defaults to 100 and is hard-capped at 1000.",
            },
            "offset": {
                "type": "integer",
                "minimum": 0,
                "description": (
                    "Optional zero-based row offset for pagination. The offset "
                    "is validated and applied before explain/query/retrieve execution."
                ),
            },
            "repair_attempts": {
                "type": "integer",
                "minimum": 0,
                "maximum": 5,
                "description": "Maximum SQL repair attempts after generation or execution failures. Defaults to 2.",
            },
        },
        "additionalProperties": False,
    },
}


def _require_database_id(args: dict, action: str) -> str:
    database_id = str(args.get("database_id") or "").strip()
    if not database_id:
        raise DatabaseRetrievalError(f"database_id is required when action='{action}'")
    return database_id


def _handle_database_retrieve(args: dict, **kwargs) -> str:
    try:
        action = args.get("action")
        if not action:
            if args.get("sql"):
                action = "query"
            elif args.get("user_request"):
                action = "retrieve"
            elif args.get("database_id"):
                action = "schema"
            else:
                action = "connections"
        if action == "dialects":
            return tool_result(inspect_dialect_capabilities())
        if action == "connections":
            return tool_result(inspect_database_connections())
        if action in {"test_connection", "healthcheck"}:
            database_id = _require_database_id(args, action)
            return tool_result(test_database_connection(database_id=database_id))
        if action == "schema":
            database_id = _require_database_id(args, action)
            return tool_result(inspect_database(database_id=database_id))
        if action == "validate":
            database_id = _require_database_id(args, action)
            if not args.get("sql"):
                return tool_error("sql is required when action='validate'")
            result = validate_database_query(
                database_id=database_id,
                sql=args.get("sql", ""),
                source_dialect=args.get("source_dialect") or DEFAULT_SOURCE_DIALECT,
                max_rows=args.get("max_rows") or DEFAULT_MAX_ROWS,
                offset=args.get("offset"),
                query_parameters=args.get("query_parameters"),
            )
            return tool_result(result)
        if action == "explain":
            database_id = _require_database_id(args, action)
            if not args.get("sql"):
                return tool_error("sql is required when action='explain'")
            result = explain_database_query(
                database_id=database_id,
                sql=args.get("sql", ""),
                source_dialect=args.get("source_dialect") or DEFAULT_SOURCE_DIALECT,
                max_rows=args.get("max_rows") or DEFAULT_MAX_ROWS,
                offset=args.get("offset"),
                query_parameters=args.get("query_parameters"),
            )
            return tool_result(result)
        if action == "generate":
            database_id = _require_database_id(args, action)
            if not args.get("user_request"):
                return tool_error("user_request is required when action='generate'")
            generated = generate_sql_for_request(
                database_id=database_id,
                user_request=args.get("user_request", ""),
                source_dialect=args.get("source_dialect") or DEFAULT_SOURCE_DIALECT,
                max_rows=args.get("max_rows") or DEFAULT_MAX_ROWS,
                offset=args.get("offset"),
                repair_attempts=args.get("repair_attempts", DEFAULT_REPAIR_ATTEMPTS),
            )
            audit = build_audit_event(
                action="generate",
                database_id=database_id,
                source_dialect=generated.source_dialect,
                target_dialect=generated.target_dialect,
                sql=generated.sql,
                generated_sql=generated.raw_sql,
                referenced_tables=list(generated.referenced_tables),
                referenced_columns=list(generated.referenced_columns),
                attempts=list(generated.attempts),
                warnings=list(generated.warnings),
            )
            _log_audit(audit)
            return tool_result(
                success=True,
                database_id=database_id,
                user_request=generated.user_request,
                source_dialect=generated.source_dialect,
                target_dialect=generated.target_dialect,
                generated_sql=generated.raw_sql,
                sql=generated.sql,
                referenced_tables=list(generated.referenced_tables),
                referenced_columns=list(generated.referenced_columns),
                warnings=list(generated.warnings),
                attempts=list(generated.attempts),
                audit=audit,
            )
        if action == "retrieve":
            database_id = _require_database_id(args, action)
            if not args.get("user_request"):
                return tool_error("user_request is required when action='retrieve'")
            result = retrieve_database_for_request(
                database_id=database_id,
                user_request=args.get("user_request", ""),
                source_dialect=args.get("source_dialect") or DEFAULT_SOURCE_DIALECT,
                max_rows=args.get("max_rows") or DEFAULT_MAX_ROWS,
                offset=args.get("offset"),
                query_parameters=args.get("query_parameters"),
                include_columns=args.get("include_columns"),
                max_cell_chars=args.get("max_cell_chars"),
                repair_attempts=args.get("repair_attempts", DEFAULT_REPAIR_ATTEMPTS),
            )
            return tool_result(result)
        if action != "query":
            return tool_error(f"Unsupported database_retrieve action: {action}")
        database_id = _require_database_id(args, action)
        if not args.get("sql"):
            return tool_error("sql is required when action='query'")
        result = retrieve_database(
            database_id=database_id,
            sql=args.get("sql", ""),
            source_dialect=args.get("source_dialect") or DEFAULT_SOURCE_DIALECT,
            max_rows=args.get("max_rows") or DEFAULT_MAX_ROWS,
            offset=args.get("offset"),
            query_parameters=args.get("query_parameters"),
            include_columns=args.get("include_columns"),
            max_cell_chars=args.get("max_cell_chars"),
        )
        return tool_result(result)
    except Exception as exc:
        return tool_error(str(exc))


registry.register(
    name="database_retrieve",
    toolset="database",
    schema=DATABASE_RETRIEVE_SCHEMA,
    handler=_handle_database_retrieve,
    check_fn=check_database_retrieve_requirements,
    emoji="🗄️",
    max_result_size_chars=100_000,
)
