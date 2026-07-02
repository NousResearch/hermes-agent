#!/usr/bin/env python3
"""Refresh Snowflake DDL snapshots under yallaplay-wiki/reference/warehouse/ddl/.

Read-only metadata workflow. Reuses tools/snowflake_query.py credential loading
and connection code so it remains compatible with the legacy analytics-agent
vars.toml while keeping credentials out of this repository.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Import sibling helper without requiring tools/ to be a package.
TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from snowflake_query import config_from_mapping, connect, pick_config_source  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA_DIR = PROJECT_DIR / "yallaplay-wiki" / "reference" / "warehouse" / "ddl"

# EVENTS: emit CREATE TABLE column lists only (raw source tables; lineage is less useful here).
# AGGREGATES: emit full GET_DDL including target_lag and AS <SELECT> body for lineage.
DEFAULT_TARGETS = {
    "EVENTS": "columns",
    "AGGREGATES": "full",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vars", type=Path, help="Private vars.toml path")
    parser.add_argument(
        "--schema-dir",
        type=Path,
        default=DEFAULT_SCHEMA_DIR,
        help="Output directory for DDL snapshots (default: yallaplay-wiki/reference/warehouse/ddl)",
    )
    parser.add_argument(
        "--schemas",
        nargs="+",
        default=list(DEFAULT_TARGETS),
        help="Snowflake schemas to snapshot (default: EVENTS AGGREGATES)",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "columns", "full"],
        default="auto",
        help="DDL mode for all schemas; auto uses columns for EVENTS and full for AGGREGATES",
    )
    parser.add_argument("--dry-run", action="store_true", help="List target schemas without querying Snowflake")
    return parser.parse_args()


def quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def columns_ddl(cursor, database: str, schema: str, name: str, kind: str) -> str:
    cursor.execute(
        f"DESCRIBE {kind} {quote_ident(database)}.{quote_ident(schema)}.{quote_ident(name)}"
    )
    columns = cursor.fetchall()
    col_defs = []
    for col in columns:
        col_name, col_type, nullable = col[0], col[1], col[3]
        line = f"  {quote_ident(str(col_name))} {col_type}"
        if nullable == "N":
            line += " NOT NULL"
        col_defs.append(line)
    return f"CREATE {kind} {schema}.{name} (\n" + ",\n".join(col_defs) + "\n);"


def full_ddl(cursor, database: str, schema: str, name: str, kind: str) -> str:
    cursor.execute(f"SELECT GET_DDL('{kind}', '{database}.{schema}.{name}')")
    return str(cursor.fetchone()[0]).strip()


def schema_mode(schema: str, requested: str) -> str:
    if requested != "auto":
        return requested
    return DEFAULT_TARGETS.get(schema.upper(), "columns")


def snapshot_schema(cursor, database: str, schema: str, mode: str, schema_dir: Path) -> list[str]:
    schema_dir.mkdir(parents=True, exist_ok=True)
    out_path = schema_dir / f"{schema}.sql"

    entries: list[tuple[str, str]] = []
    cursor.execute(f"SHOW TABLES IN SCHEMA {quote_ident(database)}.{quote_ident(schema)}")
    entries.extend((str(row[1]), "TABLE") for row in cursor.fetchall())
    cursor.execute(f"SHOW VIEWS IN SCHEMA {quote_ident(database)}.{quote_ident(schema)}")
    entries.extend((str(row[1]), "VIEW") for row in cursor.fetchall())
    entries.sort(key=lambda item: item[0])

    blocks = []
    names = []
    for name, kind in entries:
        if mode == "full":
            blocks.append(full_ddl(cursor, database, schema, name, kind))
        else:
            blocks.append(columns_ddl(cursor, database, schema, name, kind))
        names.append(name)

    out_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    return names


def main() -> int:
    args = parse_args()
    schemas = [schema.upper() for schema in args.schemas]
    if args.dry_run:
        for schema in schemas:
            print(f"would snapshot {schema} in {schema_mode(schema, args.mode)} mode -> {args.schema_dir / (schema + '.sql')}")
        return 0

    source, mapping = pick_config_source(args.vars)
    config = config_from_mapping(mapping)
    print(f"Using Snowflake config source: {source}")
    connection = connect(config)
    cursor = connection.cursor()
    try:
        database = str(connection.database or config.database)
        for schema in schemas:
            mode = schema_mode(schema, args.mode)
            print(f"\n=== {database}.{schema} ({mode}) ===")
            names = snapshot_schema(cursor, database, schema, mode, args.schema_dir)
            for name in names:
                print(f"  {name}")
            print(f"wrote {len(names)} objects to {args.schema_dir / (schema + '.sql')}")
    finally:
        cursor.close()
        connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
