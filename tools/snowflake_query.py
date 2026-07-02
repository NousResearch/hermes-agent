#!/usr/bin/env python3
"""Run read-only Snowflake queries for the Hermes analytics skill.

Credentials are intentionally loaded from private local state, never from this
repository. Resolution order:

1. Snowflake environment variables.
2. --vars / HERMES_SNOWFLAKE_VARS / SNOWFLAKE_VARS_TOML.
3. ./vars.toml if a local untracked lab file exists.
4. ../yallaplay-analytics-agent-gpt/vars.toml for migration labs.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import snowflake.connector
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit("Missing dependency: snowflake-connector-python") from exc

try:
    import sqlglot
except ImportError:  # pragma: no cover - optional validation helper
    sqlglot = None

try:
    import sqlparse
except ImportError:  # pragma: no cover - optional formatting helper
    sqlparse = None

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

from cryptography.hazmat.primitives import serialization


PROJECT_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_DIR / "logs"
DEFAULT_SIBLING_VARS = PROJECT_DIR.parent / "yallaplay-analytics-agent-gpt" / "vars.toml"
LOCAL_VARS = PROJECT_DIR / "vars.toml"

ALLOWED_FIRST_KEYWORDS = {"SELECT", "WITH", "SHOW", "DESC", "DESCRIBE", "EXPLAIN"}
BLOCKED_KEYWORDS = {
    "ALTER",
    "CALL",
    "COPY",
    "CREATE",
    "DELETE",
    "DROP",
    "GET",
    "GRANT",
    "INSERT",
    "MERGE",
    "PUT",
    "REMOVE",
    "REVOKE",
    "TRUNCATE",
    "UPDATE",
    "USE",
}


@dataclass(frozen=True)
class SnowflakeConfig:
    account: str
    host: str | None
    user: str
    private_key: str
    warehouse: str
    database: str
    schema: str
    role: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sql", nargs="?", help="SQL query string")
    parser.add_argument("-f", "--file", type=Path, help="Path to a .sql file")
    parser.add_argument("-o", "--output", type=Path, help="Output CSV path")
    parser.add_argument("--vars", type=Path, help="Private vars.toml path")
    parser.add_argument("--dry-run", action="store_true", help="validate and log SQL without executing")
    return parser.parse_args()


def read_sql(args: argparse.Namespace) -> str:
    if args.file:
        return args.file.read_text(encoding="utf-8")
    if args.sql:
        return args.sql
    raise SystemExit("Provide a SQL string or -f <file>")


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def pick_config_source(vars_path: Path | None) -> tuple[str, dict[str, Any]]:
    env = os.environ
    env_required = ["SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PRIVATE_KEY", "SNOWFLAKE_WAREHOUSE"]
    if all(env.get(key) for key in env_required):
        return "environment", dict(env)

    candidates = [
        vars_path,
        Path(env["HERMES_SNOWFLAKE_VARS"]) if env.get("HERMES_SNOWFLAKE_VARS") else None,
        Path(env["SNOWFLAKE_VARS_TOML"]) if env.get("SNOWFLAKE_VARS_TOML") else None,
        LOCAL_VARS if LOCAL_VARS.exists() else None,
        DEFAULT_SIBLING_VARS if DEFAULT_SIBLING_VARS.exists() else None,
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return str(candidate), load_toml(candidate)
    raise SystemExit(
        "No Snowflake credentials found. Set env vars or HERMES_SNOWFLAKE_VARS "
        "to a private vars.toml path."
    )


def config_from_mapping(mapping: dict[str, Any]) -> SnowflakeConfig:
    database = mapping.get("SNOWFLAKE_DATABASE") or mapping.get("SNOWFLAKE_DATABASE_PROD")
    missing = [
        key
        for key, value in {
            "SNOWFLAKE_ACCOUNT": mapping.get("SNOWFLAKE_ACCOUNT"),
            "SNOWFLAKE_USER": mapping.get("SNOWFLAKE_USER"),
            "SNOWFLAKE_PRIVATE_KEY": mapping.get("SNOWFLAKE_PRIVATE_KEY"),
            "SNOWFLAKE_WAREHOUSE": mapping.get("SNOWFLAKE_WAREHOUSE"),
            "SNOWFLAKE_DATABASE or SNOWFLAKE_DATABASE_PROD": database,
            "SNOWFLAKE_SCHEMA": mapping.get("SNOWFLAKE_SCHEMA"),
        }.items()
        if not value
    ]
    if missing:
        raise SystemExit(f"Snowflake config missing: {', '.join(missing)}")
    return SnowflakeConfig(
        account=str(mapping["SNOWFLAKE_ACCOUNT"]),
        host=str(mapping["SNOWFLAKE_HOST"]) if mapping.get("SNOWFLAKE_HOST") else None,
        user=str(mapping["SNOWFLAKE_USER"]),
        private_key=str(mapping["SNOWFLAKE_PRIVATE_KEY"]),
        warehouse=str(mapping["SNOWFLAKE_WAREHOUSE"]),
        database=str(database),
        schema=str(mapping["SNOWFLAKE_SCHEMA"]),
        role=str(mapping["SNOWFLAKE_ROLE"]) if mapping.get("SNOWFLAKE_ROLE") else None,
    )


def strip_sql_comments(sql: str) -> str:
    if sqlparse:
        return sqlparse.format(sql, strip_comments=True).strip()
    sql = re.sub(r"--.*?$", "", sql, flags=re.MULTILINE)
    return re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL).strip()


def split_statements(sql: str) -> list[str]:
    if sqlparse:
        return [statement.strip() for statement in sqlparse.split(sql) if statement.strip()]
    return [statement.strip() for statement in sql.split(";") if statement.strip()]


def first_keyword(sql: str) -> str:
    match = re.match(r"\s*([a-zA-Z]+)", sql)
    return match.group(1).upper() if match else ""


def validate_read_only(sql: str) -> str:
    stripped = strip_sql_comments(sql).rstrip(";").strip()
    statements = split_statements(stripped)
    if len(statements) != 1:
        raise SystemExit("Only one read-only SQL statement is allowed")
    statement = statements[0]
    keyword = first_keyword(statement)
    if keyword not in ALLOWED_FIRST_KEYWORDS:
        raise SystemExit(f"Only read-only SQL is allowed; got first keyword {keyword or '<none>'}")
    blocked_pattern = r"\b(" + "|".join(sorted(BLOCKED_KEYWORDS)) + r")\b"
    blocked = re.search(blocked_pattern, statement, flags=re.IGNORECASE)
    if blocked:
        raise SystemExit(f"Blocked non-read-only keyword: {blocked.group(1).upper()}")
    if sqlglot:
        try:
            sqlglot.parse(statement, dialect="snowflake")
        except sqlglot.errors.ParseError as exc:
            raise SystemExit(f"SQL parse error:\n{exc}") from exc
    return statement


def format_sql(sql: str) -> str:
    if sqlparse:
        return sqlparse.format(sql, reindent=True, keyword_case="upper").strip()
    return sql.strip()


def make_log_path(sql: str) -> Path:
    table_match = re.search(r"\bFROM\s+([\w.\"]+)", sql, flags=re.IGNORECASE)
    parts = []
    if table_match:
        parts.append(table_match.group(1).split(".")[-1].strip('"').lower())
    parts.extend(alias.lower() for alias in re.findall(r"\bAS\s+([a-zA-Z_][\w]*)", sql, flags=re.IGNORECASE)[:3])
    slug = re.sub(r"[^a-z0-9_]+", "_", "_".join(parts) or "query").strip("_")[:60]
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    return LOG_DIR / f"{timestamp}-{slug}.sql"


def write_query_log(sql: str) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = make_log_path(sql)
    path.write_text(format_sql(sql) + "\n", encoding="utf-8")
    return path


def private_key_der(pem_text: str) -> bytes:
    private_key = serialization.load_pem_private_key(pem_text.encode("utf-8"), password=None)
    return private_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def connect(config: SnowflakeConfig):
    kwargs: dict[str, Any] = {
        "account": config.account,
        "user": config.user,
        "private_key": private_key_der(config.private_key),
        "warehouse": config.warehouse,
        "database": config.database,
        "schema": config.schema,
    }
    if config.host:
        kwargs["host"] = config.host
    if config.role:
        kwargs["role"] = config.role
    return snowflake.connector.connect(**kwargs)


def print_table(headers: list[str], rows: list[tuple[Any, ...]]) -> None:
    widths = [len(header) for header in headers]
    rendered_rows: list[list[str]] = []
    for row in rows:
        rendered = ["NULL" if value is None else str(value) for value in row]
        rendered_rows.append(rendered)
        for index, value in enumerate(rendered):
            widths[index] = max(widths[index], len(value))
    fmt = "  ".join(f"{{:<{width}}}" for width in widths)
    print(fmt.format(*headers))
    print("  ".join("-" * width for width in widths))
    for row in rendered_rows:
        print(fmt.format(*row))
    print(f"\n{len(rows)} rows")


def run_query(sql: str, config: SnowflakeConfig, output: Path | None) -> None:
    connection = connect(config)
    cursor = connection.cursor()
    try:
        cursor.execute(sql)
        headers = [description[0] for description in cursor.description or []]
        rows = cursor.fetchall()
    finally:
        cursor.close()
        connection.close()

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(headers)
            writer.writerows(rows)
        print(f"{len(rows)} rows written to {output}")
    else:
        print_table(headers, rows)


def main() -> int:
    args = parse_args()
    sql = validate_read_only(read_sql(args))
    log_path = write_query_log(sql)
    print(f"Query logged to {log_path}")
    if args.dry_run:
        print("Dry run OK; query was not executed")
        return 0
    source, mapping = pick_config_source(args.vars)
    config = config_from_mapping(mapping)
    print(f"Using Snowflake config source: {source}")
    run_query(sql, config, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
