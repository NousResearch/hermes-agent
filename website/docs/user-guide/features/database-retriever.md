---
sidebar_position: 3
title: "Database Retrieval"
description: "Read-only SQL retrieval across configured databases with SQLGlot validation and dialect transpilation"
---

# Database Retrieval

The `database` toolset exposes `database_retrieve`, a read-only backend for retrieving data from configured databases. It uses SQLGlot to parse, validate, limit, and transpile SQL before execution.

The tool is off by default. Enable it with `hermes tools` and select **Database Retrieval**, or include the `database` toolset explicitly.

:::warning Read-only only
`database_retrieve` accepts only SELECT-style read queries. Mutating roots and
expressions such as INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE,
MERGE, COPY, EXECUTE, GRANT, and REVOKE are blocked before execution.
:::

## Supported Dialects

SQLGlot parsing and transpilation support is exposed through `action="dialects"`. The current supported dialect list is:

`athena`, `bigquery`, `clickhouse`, `databricks`, `doris`, `dremio`, `drill`,
`druid`, `duckdb`, `dune`, `exasol`, `fabric`, `hive`, `materialize`, `mysql`,
`oracle`, `postgres`, `presto`, `prql`, `redshift`, `risingwave`,
`singlestore`, `snowflake`, `solr`, `spark`, `spark2`, `sqlite`, `starrocks`,
`tableau`, `teradata`, `trino`, `tsql`.

Execution support is intentionally narrower:

| Dialect | Execution status |
|---------|------------------|
| SQLite | Built in, read-only file open. |
| DuckDB | Optional, enabled when `duckdb` is installed. |
| PostgreSQL | Optional, enabled by `asyncpg` through `hermes-agent[database]`. |
| Other SQLGlot dialects | Parse, validate, transpile, and use configured schema metadata. No live execution connector yet. |

## Configuration File

Connections live in `~/.hermes/database_connections.yaml`.

```yaml
connections:
  local_app:
    dialect: sqlite
    path: /absolute/path/to/app.db
    allow_tables:
      - users
      - orders

  warehouse:
    dialect: postgres
    host: localhost
    port: 5432
    database: analytics
    user: readonly_user
    password: ${POSTGRES_READONLY_PASSWORD}
    schema:
      tables:
        users:
          columns:
            id: BIGINT
            email: TEXT
            created_at: TIMESTAMP
        orders:
          columns:
            id: BIGINT
            user_id: BIGINT
            total: NUMERIC
            created_at: TIMESTAMP
    allow_tables:
      - users
      - orders
```

There is also a repo-level example at `database_connections.yaml.example`.

For non-SQLite databases, provide `schema.tables` metadata unless the connector can inspect schema live. This keeps the LLM prompt grounded and lets the validator enforce table and column allowlists.

## Actions

| Action | Purpose |
|--------|---------|
| `dialects` | List SQLGlot dialects and execution capability. No `database_id` required. |
| `connections` | List configured database IDs with secrets redacted. No `database_id` required. |
| `test_connection` / `healthcheck` | Verify a configured database using read-only checks. |
| `schema` | Inspect accessible tables and columns and return compact schema context. |
| `validate` | Parse, allowlist-check, transpile, and apply row limits without execution. |
| `explain` | Return a read-only execution plan for executable connectors. |
| `generate` | Generate validated SQL from a natural-language request without execution. |
| `query` | Validate and execute supplied SQL. |
| `retrieve` | Generate, validate, and execute SQL from a natural-language request. |

If `action` is omitted, the tool defaults to:

- `query` when `sql` is supplied.
- `retrieve` when `user_request` is supplied.
- `schema` when only `database_id` is supplied.
- `connections` when no `database_id`, `sql`, or `user_request` is supplied.

## Safety Model

Every SQL path uses the same validation sequence:

1. Parse exactly one SQL statement with SQLGlot.
2. Require a read-only root such as SELECT, UNION, INTERSECT, or EXCEPT.
3. Block known mutation expression types anywhere in the tree.
4. Enforce configured table and column allowlists.
5. Transpile into the target dialect.
6. Add or cap `LIMIT` with a hard maximum of 1000 rows.
7. Apply optional pagination `offset`.
8. Execute through a read-only connector when execution is supported.

SQLite opens databases with `mode=ro` and an authorizer that denies non-read operations. PostgreSQL execution uses `BEGIN READ ONLY` and rolls back every query. DuckDB connects with `read_only=True`.

## Examples

List configured databases:

```json
{
  "action": "connections"
}
```

Check a database:

```json
{
  "action": "test_connection",
  "database_id": "local_app"
}
```

Inspect schema:

```json
{
  "action": "schema",
  "database_id": "local_app"
}
```

Validate SQL without execution:

```json
{
  "action": "validate",
  "database_id": "local_app",
  "sql": "SELECT id, email FROM users ORDER BY id",
  "max_rows": 25,
  "offset": 50
}
```

Execute supplied SQL with bound parameters:

```json
{
  "action": "query",
  "database_id": "warehouse",
  "sql": "SELECT id, email FROM users WHERE email = $1",
  "query_parameters": ["ada@example.com"],
  "max_rows": 10,
  "include_columns": ["id", "email"],
  "max_cell_chars": 2000
}
```

Generate and retrieve from a natural-language request:

```json
{
  "action": "retrieve",
  "database_id": "local_app",
  "user_request": "show the 10 newest orders with totals",
  "max_rows": 10
}
```

## Result Metadata

Execution results include:

- `rows`: shaped result rows.
- `row_count`: number of rows fetched before response shaping.
- `pagination`: `limit`, `offset`, `returned`, and `next_offset`.
- `result_shape`: available columns, included columns, omitted column count, max cell size, and truncated cell count.
- `audit`: redacted SQL, referenced tables/columns, parameter count, and timing.

Parameter values and SQL literals are not included in audit metadata.
