import json
import sqlite3
import sys
import types

import pytest

from tools.database_retriever import (
    DATABASE_RETRIEVE_SCHEMA,
    SUPPORTED_SQLGLOT_DIALECTS,
    DatabaseRetrievalError,
    _handle_database_retrieve,
    build_audit_event,
    build_schema_context,
    build_sql_repair_messages,
    build_sql_generation_messages,
    check_database_retrieve_requirements,
    execute_duckdb_read_only,
    execute_postgres_read_only,
    explain_database_query,
    generate_sql_for_request,
    get_dialect_capabilities,
    inspect_database,
    inspect_database_connections,
    inspect_dialect_capabilities,
    execute_sqlite_read_only,
    list_database_connections,
    redact_sql_literals,
    retrieve_database,
    retrieve_database_for_request,
    shape_result_rows,
    test_database_connection,
    validate_database_query,
    validate_read_only_sql,
)


def test_dialect_registry_covers_all_sqlglot_dialects():
    expected = {
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
    }
    assert set(SUPPORTED_SQLGLOT_DIALECTS) == expected
    registry = get_dialect_capabilities()
    assert set(registry) == expected
    assert registry["sqlite"].can_execute is True
    assert registry["postgres"].can_transpile is True


def test_inspect_dialect_capabilities_exposes_execution_status():
    result = inspect_dialect_capabilities()

    assert result["success"] is True
    assert result["read_only"] is True
    assert result["default_source_dialect"] == "postgres"
    assert result["supported_dialects"] == list(SUPPORTED_SQLGLOT_DIALECTS)
    assert "sqlite" in result["executable_dialects"]
    assert result["dialects"]["sqlite"]["can_execute"] is True
    assert result["dialects"]["postgres"]["can_transpile"] is True
    assert result["dialects"]["postgres"]["can_execute"] is (
        "postgres" in result["executable_dialects"]
    )
    assert result["dialects"]["duckdb"]["can_execute"] is (
        "duckdb" in result["executable_dialects"]
    )
    assert result["audit"]["action"] == "dialects"


def test_database_retrieve_dialects_action_does_not_require_database_id():
    result = json.loads(_handle_database_retrieve({"action": "dialects"}))

    assert result["success"] is True
    assert "sqlite" in result["executable_dialects"]
    assert "postgres" in result["supported_dialects"]


def test_database_retrieve_available_without_connection_config(monkeypatch, tmp_path):
    missing_config = tmp_path / "missing_database_connections.yaml"
    monkeypatch.setattr(
        "tools.database_retriever.get_database_connections_path",
        lambda: missing_config,
    )

    assert check_database_retrieve_requirements() is True

    result = json.loads(_handle_database_retrieve({"action": "connections"}))
    assert result["success"] is True
    assert result["connection_ids"] == []
    assert result["connection_count"] == 0


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO users VALUES (1)",
        "UPDATE users SET name = 'x'",
        "DELETE FROM users",
        "DROP TABLE users",
        "TRUNCATE TABLE users",
        "CALL refresh_users()",
        "SELECT * FROM users; SELECT * FROM orders",
    ],
)
def test_read_only_validator_rejects_mutations_and_multi_statement(sql):
    with pytest.raises(DatabaseRetrievalError):
        validate_read_only_sql(sql, allowed_tables={"users", "orders"})


def test_read_only_validator_allows_cte_select_and_adds_limit():
    result = validate_read_only_sql(
        "WITH active AS (SELECT id, name FROM users WHERE enabled = true) "
        "SELECT id, name FROM active",
        allowed_tables={"users"},
        allowed_columns={"users": {"id", "name", "enabled"}},
        max_rows=25,
    )
    assert result.target_dialect == "sqlite"
    assert result.referenced_tables == ("users",)
    assert "LIMIT 25" in result.sql
    assert "LIMIT 25 was added" in result.warnings


def test_read_only_validator_caps_existing_limit():
    result = validate_read_only_sql(
        "SELECT id FROM users LIMIT 500",
        allowed_tables={"users"},
        max_rows=20,
    )
    assert "LIMIT 20" in result.sql
    assert "LIMIT was capped from 500 to 20" in result.warnings


def test_read_only_validator_applies_offset_for_pagination():
    result = validate_read_only_sql(
        "SELECT id FROM users ORDER BY id",
        allowed_tables={"users"},
        max_rows=10,
        offset=20,
    )

    assert result.sql.startswith("SELECT id FROM users ORDER BY id")
    assert result.sql.endswith("LIMIT 10 OFFSET 20")
    assert "LIMIT 10 was added" in result.warnings
    assert "OFFSET 20 was added" in result.warnings


def test_read_only_validator_rejects_negative_offset():
    with pytest.raises(DatabaseRetrievalError, match="offset"):
        validate_read_only_sql(
            "SELECT id FROM users",
            allowed_tables={"users"},
            offset=-1,
        )


def test_read_only_validator_rejects_table_outside_allowlist():
    with pytest.raises(DatabaseRetrievalError, match="outside the allowlist"):
        validate_read_only_sql("SELECT * FROM secrets", allowed_tables={"users"})


def test_read_only_validator_rejects_aliased_column_outside_allowlist():
    with pytest.raises(DatabaseRetrievalError, match="users.secret"):
        validate_read_only_sql(
            "SELECT u.secret FROM users AS u",
            allowed_tables={"users"},
            allowed_columns={"users": {"id", "name"}},
        )


def test_read_only_validator_allows_aliased_column_inside_allowlist():
    result = validate_read_only_sql(
        "SELECT u.id FROM users AS u",
        allowed_tables={"users"},
        allowed_columns={"users": {"id", "name"}},
        max_rows=5,
    )

    assert result.referenced_columns == ("users.id",)
    assert result.sql == "SELECT u.id FROM users AS u LIMIT 5"


def test_sqlite_retrieve_executes_read_only_query(tmp_path):
    db_path = tmp_path / "sample.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, enabled INTEGER)")
    conn.executemany(
        "INSERT INTO users (name, enabled) VALUES (?, ?)",
        [("Ada", 1), ("Grace", 1), ("Linus", 0)],
    )
    conn.commit()
    conn.close()

    config_path = tmp_path / "database_connections.yaml"
    config_path.write_text(
        "connections:\n"
        "  sample:\n"
        "    dialect: sqlite\n"
        f"    path: {db_path}\n",
        encoding="utf-8",
    )

    result = retrieve_database(
        database_id="sample",
        sql="SELECT id, name FROM users WHERE enabled = 1 ORDER BY id",
        connections_path=config_path,
        max_rows=10,
    )

    assert result["success"] is True
    assert result["target_dialect"] == "sqlite"
    assert result["row_count"] == 2
    assert result["rows"] == [{"id": 1, "name": "Ada"}, {"id": 2, "name": "Grace"}]
    assert result["referenced_tables"] == ["users"]
    assert result["pagination"] == {
        "limit": 10,
        "offset": 0,
        "returned": 2,
        "next_offset": None,
    }


def test_sqlite_retrieve_executes_parameterized_query(tmp_path):
    db_path = tmp_path / "sample.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
    conn.executemany(
        "INSERT INTO users (name) VALUES (?)",
        [("Ada",), ("Grace",)],
    )
    conn.commit()
    conn.close()

    config_path = tmp_path / "database_connections.yaml"
    config_path.write_text(
        "connections:\n"
        "  sample:\n"
        "    dialect: sqlite\n"
        f"    path: {db_path}\n",
        encoding="utf-8",
    )

    result = retrieve_database(
        database_id="sample",
        sql="SELECT id, name FROM users WHERE name = ?",
        query_parameters=["Grace"],
        connections_path=config_path,
        max_rows=10,
    )

    assert result["success"] is True
    assert result["rows"] == [{"id": 2, "name": "Grace"}]
    assert result["parameter_count"] == 1
    assert "Grace" not in str(result["audit"])
    assert result["audit"]["parameter_count"] == 1


def test_sqlite_retrieve_applies_offset_for_pagination(tmp_path):
    db_path = tmp_path / "sample.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
    conn.executemany(
        "INSERT INTO users (name) VALUES (?)",
        [("Ada",), ("Grace",), ("Linus",)],
    )
    conn.commit()
    conn.close()

    config_path = tmp_path / "database_connections.yaml"
    config_path.write_text(
        "connections:\n"
        "  sample:\n"
        "    dialect: sqlite\n"
        f"    path: {db_path}\n",
        encoding="utf-8",
    )

    result = retrieve_database(
        database_id="sample",
        sql="SELECT id, name FROM users ORDER BY id",
        connections_path=config_path,
        max_rows=1,
        offset=1,
    )

    assert result["success"] is True
    assert result["sql"].startswith("SELECT id, name FROM users ORDER BY id")
    assert result["sql"].endswith("LIMIT 1 OFFSET 1")
    assert result["offset"] == 1
    assert result["pagination"] == {
        "limit": 1,
        "offset": 1,
        "returned": 1,
        "next_offset": 2,
    }
    assert result["rows"] == [{"id": 2, "name": "Grace"}]


def test_result_shaping_filters_columns_and_truncates_cells():
    rows, metadata = shape_result_rows(
        [{"id": 1, "name": "Ada", "bio": "abcdef"}],
        include_columns=["id", "bio"],
        max_cell_chars=3,
    )

    assert rows == [{"id": 1, "bio": "abc..."}]
    assert metadata == {
        "columns": ["id", "name", "bio"],
        "included_columns": ["id", "bio"],
        "omitted_column_count": 1,
        "max_cell_chars": 3,
        "truncated_cell_count": 1,
    }


def test_result_shaping_rejects_missing_include_column():
    with pytest.raises(DatabaseRetrievalError, match="not present"):
        shape_result_rows(
            [{"id": 1, "name": "Ada"}],
            include_columns=["email"],
        )


def test_result_shaping_serializes_binary_cells():
    rows, metadata = shape_result_rows(
        [{"id": 1, "payload": b"abc"}],
    )

    assert rows == [{"id": 1, "payload": "[binary data: 3 bytes]"}]
    assert metadata["truncated_cell_count"] == 0


def test_sqlite_retrieve_shapes_result_rows(tmp_path):
    db_path = tmp_path / "sample.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE docs (id INTEGER PRIMARY KEY, title TEXT, body TEXT)")
    conn.execute("INSERT INTO docs (title, body) VALUES ('Guide', 'abcdef')")
    conn.commit()
    conn.close()

    config_path = tmp_path / "database_connections.yaml"
    config_path.write_text(
        "connections:\n"
        "  sample:\n"
        "    dialect: sqlite\n"
        f"    path: {db_path}\n",
        encoding="utf-8",
    )

    result = retrieve_database(
        database_id="sample",
        sql="SELECT id, title, body FROM docs",
        connections_path=config_path,
        include_columns=["title", "body"],
        max_cell_chars=4,
    )

    assert result["row_count"] == 1
    assert result["rows"] == [{"title": "Guid...", "body": "abcd..."}]
    assert result["result_shape"]["columns"] == ["id", "title", "body"]
    assert result["result_shape"]["included_columns"] == ["title", "body"]
    assert result["result_shape"]["omitted_column_count"] == 1
    assert result["result_shape"]["truncated_cell_count"] == 2


def test_sqlite_executor_blocks_writes_even_when_called_directly(tmp_path):
    db_path = tmp_path / "sample.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
    conn.execute("INSERT INTO users (name) VALUES ('Ada')")
    conn.commit()
    conn.close()

    rows = execute_sqlite_read_only(db_path, "SELECT id, name FROM users")
    assert rows == [{"id": 1, "name": "Ada"}]

    with pytest.raises(sqlite3.DatabaseError, match="not authorized|readonly"):
        execute_sqlite_read_only(db_path, "UPDATE users SET name = 'Grace'")

    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("SELECT name FROM users").fetchone()[0] == "Ada"
    finally:
        conn.close()


def test_sqlite_executor_blocks_pragma_when_called_directly(tmp_path):
    db_path = tmp_path / "sample.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
    conn.commit()
    conn.close()

    with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
        execute_sqlite_read_only(db_path, "PRAGMA table_info(users)")


def test_duckdb_executor_uses_read_only_connection(monkeypatch, tmp_path):
    db_path = tmp_path / "sample.duckdb"
    calls = {}

    class FakeCursor:
        description = [("id",), ("name",)]

        def fetchall(self):
            return [(1, "Ada")]

    class FakeConnection:
        def execute(self, sql, parameters):
            calls["sql"] = sql
            calls["parameters"] = parameters
            return FakeCursor()

        def close(self):
            calls["closed"] = True

    def fake_connect(path, *, read_only):
        calls["path"] = path
        calls["read_only"] = read_only
        return FakeConnection()

    monkeypatch.setitem(
        sys.modules,
        "duckdb",
        types.SimpleNamespace(connect=fake_connect),
    )

    rows = execute_duckdb_read_only(
        db_path,
        "SELECT id, name FROM users WHERE name = ?",
        query_parameters=["Ada"],
    )

    assert rows == [{"id": 1, "name": "Ada"}]
    assert calls["path"] == str(db_path)
    assert calls["read_only"] is True
    assert calls["sql"] == "SELECT id, name FROM users WHERE name = ?"
    assert calls["parameters"] == ("Ada",)
    assert calls["closed"] is True


def test_duckdb_retrieve_executes_when_optional_dependency_available(
    monkeypatch, tmp_path
):
    db_path = tmp_path / "sample.duckdb"
    config_path = tmp_path / "database_connections.yaml"
    config_path.write_text(
        "connections:\n"
        "  analytics:\n"
        "    dialect: duckdb\n"
        f"    path: {db_path}\n"
        "    schema:\n"
        "      tables:\n"
        "        users:\n"
        "          columns:\n"
        "            id: INTEGER\n"
        "            name: TEXT\n",
        encoding="utf-8",
    )

    class FakeCursor:
        description = [("id",), ("name",)]

        def fetchall(self):
            return [(1, "Ada")]

    class FakeConnection:
        def execute(self, _sql, _parameters):
            return FakeCursor()

        def close(self):
            pass

    monkeypatch.setitem(
        sys.modules,
        "duckdb",
        types.SimpleNamespace(
            connect=lambda _path, *, read_only: FakeConnection(),
        ),
    )

    result = retrieve_database(
        database_id="analytics",
        sql="SELECT id, name FROM users",
        connections_path=config_path,
        max_rows=5,
    )

    assert result["success"] is True
    assert result["target_dialect"] == "duckdb"
    assert result["sql"] == "SELECT id, name FROM users LIMIT 5"
    assert result["rows"] == [{"id": 1, "name": "Ada"}]


def test_postgres_executor_uses_read_only_transaction(monkeypatch):
    calls = {"execute": []}

    class FakeConnection:
        async def execute(self, sql):
            calls["execute"].append(sql)

        async def fetch(self, sql, *parameters):
            calls["fetch_sql"] = sql
            calls["parameters"] = parameters
            return [{"id": 1, "name": "Ada"}]

        async def close(self):
            calls["closed"] = True

    async def fake_connect(**kwargs):
        calls["connect"] = kwargs
        return FakeConnection()

    monkeypatch.setitem(
        sys.modules,
        "asyncpg",
        types.SimpleNamespace(connect=fake_connect),
    )

    rows = execute_postgres_read_only(
        {
            "host": "localhost",
            "port": 5432,
            "user": "reader",
            "password": "secret",
            "database": "analytics",
        },
        "SELECT id, name FROM users WHERE name = $1",
        query_parameters=["Ada"],
    )

    assert rows == [{"id": 1, "name": "Ada"}]
    assert calls["connect"] == {
        "host": "localhost",
        "port": 5432,
        "user": "reader",
        "password": "secret",
        "database": "analytics",
    }
    assert calls["execute"] == ["BEGIN READ ONLY", "ROLLBACK"]
    assert calls["fetch_sql"] == "SELECT id, name FROM users WHERE name = $1"
    assert calls["parameters"] == ("Ada",)
    assert calls["closed"] is True


def test_postgres_executor_rejects_named_parameters(monkeypatch):
    async def fake_connect(**_kwargs):
        raise AssertionError("connect should not be called")

    monkeypatch.setitem(
        sys.modules,
        "asyncpg",
        types.SimpleNamespace(connect=fake_connect),
    )

    with pytest.raises(DatabaseRetrievalError, match="array"):
        execute_postgres_read_only(
            {"host": "localhost", "database": "analytics"},
            "SELECT id FROM users WHERE name = $1",
            query_parameters={"name": "Ada"},
        )


def test_postgres_retrieve_executes_when_asyncpg_available(monkeypatch, tmp_path):
    config_path = tmp_path / "database_connections.yaml"
    config_path.write_text(
        "connections:\n"
        "  warehouse:\n"
        "    dialect: postgres\n"
        "    host: localhost\n"
        "    database: analytics\n"
        "    user: reader\n"
        "    schema:\n"
        "      tables:\n"
        "        users:\n"
        "          columns:\n"
        "            id: INTEGER\n"
        "            name: TEXT\n",
        encoding="utf-8",
    )
    calls = {}

    class FakeConnection:
        async def execute(self, _sql):
            pass

        async def fetch(self, sql, *parameters):
            calls["sql"] = sql
            calls["parameters"] = parameters
            return [{"id": 1, "name": "Ada"}]

        async def close(self):
            pass

    async def fake_connect(**kwargs):
        calls["connect"] = kwargs
        return FakeConnection()

    monkeypatch.setitem(
        sys.modules,
        "asyncpg",
        types.SimpleNamespace(connect=fake_connect),
    )

    result = retrieve_database(
        database_id="warehouse",
        sql="SELECT id, name FROM users WHERE name = $1",
        query_parameters=["Ada"],
        connections_path=config_path,
        max_rows=5,
    )

    assert result["success"] is True
    assert result["target_dialect"] == "postgres"
    assert result["sql"] == "SELECT id, name FROM users WHERE name = $1 LIMIT 5"
    assert result["rows"] == [{"id": 1, "name": "Ada"}]
    assert calls["connect"] == {
        "host": "localhost",
        "database": "analytics",
        "user": "reader",
    }
    assert calls["parameters"] == ("Ada",)


def test_sqlite_explain_returns_plan_without_running_data_query(tmp_path):
    db_path = tmp_path / "sample.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
    conn.execute("INSERT INTO users (name) VALUES ('Ada')")
    conn.commit()
    conn.close()

    config_path = tmp_path / "database_connections.yaml"
    config_path.write_text(
        "connections:\n"
        "  sample:\n"
        "    dialect: sqlite\n"
        f"    path: {db_path}\n",
        encoding="utf-8",
    )

    result = explain_database_query(
        database_id="sample",
        sql="SELECT id FROM users WHERE name = ?",
        query_parameters=["Ada"],
        connections_path=config_path,
        max_rows=5,
    )

    assert result["success"] is True
    assert result["data_query_executed"] is False
    assert result["sql"] == "SELECT id FROM users WHERE name = ? LIMIT 5"
    assert result["explain_sql"].startswith("EXPLAIN QUERY PLAN ")
    assert result["plan_row_count"] >= 1
    assert any("users" in row["detail"] for row in result["plan"])
    assert "rows" not in result
    assert result["parameter_count"] == 1
    assert result["audit"]["action"] == "explain"
    assert result["audit"]["row_count"] == result["plan_row_count"]
    assert "Ada" not in str(result["audit"])
    assert result["pagination"] == {"limit": 5, "offset": 0}


def test_duckdb_explain_uses_read_only_connection(monkeypatch, tmp_path):
    db_path = tmp_path / "sample.duckdb"
    config_path = tmp_path / "database_connections.yaml"
    config_path.write_text(
        "connections:\n"
        "  analytics:\n"
        "    dialect: duckdb\n"
        f"    path: {db_path}\n"
        "    schema:\n"
        "      tables:\n"
        "        users:\n"
        "          columns:\n"
        "            id: INTEGER\n"
        "            name: TEXT\n",
        encoding="utf-8",
    )
    calls = {}

    class FakeCursor:
        description = [("explain_key",), ("explain_value",)]

        def fetchall(self):
            return [("physical_plan", "SEQ_SCAN users")]

    class FakeConnection:
        def execute(self, sql, parameters):
            calls["sql"] = sql
            calls["parameters"] = parameters
            return FakeCursor()

        def close(self):
            calls["closed"] = True

    def fake_connect(path, *, read_only):
        calls["path"] = path
        calls["read_only"] = read_only
        return FakeConnection()

    monkeypatch.setitem(
        sys.modules,
        "duckdb",
        types.SimpleNamespace(connect=fake_connect),
    )

    result = explain_database_query(
        database_id="analytics",
        sql="SELECT id FROM users WHERE name = ?",
        query_parameters=["Ada"],
        connections_path=config_path,
        max_rows=5,
    )

    assert result["target_dialect"] == "duckdb"
    assert result["explain_sql"] == "EXPLAIN SELECT id FROM users WHERE name = ? LIMIT 5"
    assert result["plan"] == [
        {"explain_key": "physical_plan", "explain_value": "SEQ_SCAN users"}
    ]
    assert calls["path"] == str(db_path)
    assert calls["read_only"] is True
    assert calls["parameters"] == ("Ada",)
    assert calls["closed"] is True


def test_validate_database_query_dry_runs_without_execution(tmp_path):
    db_path = tmp_path / "sample.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
    conn.execute("INSERT INTO users (name) VALUES ('Ada')")
    conn.commit()
    conn.close()

    config_path = tmp_path / "database_connections.yaml"
    config_path.write_text(
        "connections:\n"
        "  sample:\n"
        "    dialect: sqlite\n"
        f"    path: {db_path}\n",
        encoding="utf-8",
    )

    result = validate_database_query(
        database_id="sample",
        sql="SELECT id, name FROM users WHERE name = 'Ada'",
        connections_path=config_path,
        max_rows=5,
    )

    assert result["success"] is True
    assert result["executed"] is False
    assert "rows" not in result
    assert result["sql"] == "SELECT id, name FROM users WHERE name = 'Ada' LIMIT 5"
    assert result["referenced_tables"] == ["users"]
    assert result["pagination"] == {"limit": 5, "offset": 0}
    assert result["audit"]["action"] == "validate"
    assert "Ada" not in str(result["audit"])


def test_validate_database_query_reports_parameter_count_without_values(tmp_path):
    db_path = tmp_path / "sample.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
    conn.commit()
    conn.close()

    config_path = tmp_path / "database_connections.yaml"
    config_path.write_text(
        "connections:\n"
        "  sample:\n"
        "    dialect: sqlite\n"
        f"    path: {db_path}\n",
        encoding="utf-8",
    )

    result = validate_database_query(
        database_id="sample",
        sql="SELECT id FROM users WHERE name = ?",
        query_parameters=["Ada"],
        connections_path=config_path,
    )

    assert result["executed"] is False
    assert result["parameter_count"] == 1
    assert result["audit"]["parameter_count"] == 1
    assert "Ada" not in str(result["audit"])


def test_schema_inspection_returns_compact_context(tmp_path):
    db_path = tmp_path / "sample.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL)")
    conn.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, total REAL)")
    conn.commit()
    conn.close()

    config_path = tmp_path / "database_connections.yaml"
    config_path.write_text(
        "connections:\n"
        "  sample:\n"
        "    dialect: sqlite\n"
        f"    path: {db_path}\n",
        encoding="utf-8",
    )

    result = inspect_database(database_id="sample", connections_path=config_path)

    assert result["success"] is True
    assert result["read_only"] is True
    assert result["dialect"] == "sqlite"
    assert "Database ID: sample" in result["schema_context"]
    assert "- users:" in result["schema_context"]
    assert "id INTEGER (primary key)" in result["schema_context"]
    assert "name TEXT (not null)" in result["schema_context"]
    assert "- orders:" in result["schema_context"]


def test_configured_schema_supports_non_sqlite_dialects(tmp_path):
    config_path = tmp_path / "database_connections.yaml"
    config_path.write_text(
        "connections:\n"
        "  warehouse:\n"
        "    dialect: postgres\n"
        "    schema:\n"
        "      tables:\n"
        "        users:\n"
        "          columns:\n"
        "            id:\n"
        "              type: INTEGER\n"
        "              nullable: false\n"
        "              primary_key: true\n"
        "            email: TEXT\n"
        "            active: BOOLEAN\n",
        encoding="utf-8",
    )

    result = inspect_database(database_id="warehouse", connections_path=config_path)

    assert result["success"] is True
    assert result["dialect"] == "postgres"
    assert result["can_execute"] is False
    assert "id INTEGER (primary key, not null)" in result["schema_context"]
    assert "- users:" in result["schema_context"]

    validation = validate_database_query(
        database_id="warehouse",
        sql="SELECT id, email FROM users WHERE active = TRUE",
        connections_path=config_path,
        max_rows=10,
    )

    assert validation["success"] is True
    assert validation["executed"] is False
    assert validation["target_dialect"] == "postgres"
    assert validation["referenced_tables"] == ["users"]
    assert validation["referenced_columns"] == ["active", "email", "id"]
    assert "LIMIT 10" in validation["sql"]


def test_generate_sql_uses_configured_non_sqlite_schema(tmp_path):
    config_path = tmp_path / "database_connections.yaml"
    config_path.write_text(
        "connections:\n"
        "  warehouse:\n"
        "    dialect: postgres\n"
        "    schema:\n"
        "      tables:\n"
        "        - name: users\n"
        "          columns:\n"
        "            - name: id\n"
        "              type: INTEGER\n"
        "            - name: email\n"
        "              type: TEXT\n",
        encoding="utf-8",
    )

    seen_messages = []

    def fake_generator(messages):
        seen_messages.extend(messages)
        return "SELECT id, email FROM users"

    generated = generate_sql_for_request(
        database_id="warehouse",
        user_request="show user emails",
        connections_path=config_path,
        max_rows=3,
        llm_sql_generator=fake_generator,
    )

    assert generated.target_dialect == "postgres"
    assert generated.sql == "SELECT id, email FROM users LIMIT 3"
    assert "Dialect: postgres" in seen_messages[1]["content"]


def test_non_sqlite_schema_requires_configured_metadata(tmp_path):
    config_path = tmp_path / "database_connections.yaml"
    config_path.write_text(
        "connections:\n"
        "  warehouse:\n"
        "    dialect: postgres\n",
        encoding="utf-8",
    )

    with pytest.raises(DatabaseRetrievalError, match="requires configured schema"):
        inspect_database(database_id="warehouse", connections_path=config_path)


def test_configured_allowlists_filter_schema_and_validation(tmp_path):
    db_path = tmp_path / "sample.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, secret TEXT)")
    conn.execute("CREATE TABLE audit_log (id INTEGER PRIMARY KEY, event TEXT)")
    conn.execute("INSERT INTO users (name, secret) VALUES ('Ada', 'hidden')")
    conn.commit()
    conn.close()

    config_path = tmp_path / "database_connections.yaml"
    config_path.write_text(
        "connections:\n"
        "  sample:\n"
        "    dialect: sqlite\n"
        f"    path: {db_path}\n"
        "    allow_tables: [users]\n"
        "    allow_columns:\n"
        "      users: [id, name]\n",
        encoding="utf-8",
    )

    schema_result = inspect_database(database_id="sample", connections_path=config_path)
    context = schema_result["schema_context"]
    assert "- users:" in context
    assert "name TEXT" in context
    assert "secret" not in context
    assert "audit_log" not in context

    result = retrieve_database(
        database_id="sample",
        sql="SELECT id, name FROM users",
        connections_path=config_path,
    )
    assert result["row_count"] == 1
    assert result["rows"] == [{"id": 1, "name": "Ada"}]

    with pytest.raises(DatabaseRetrievalError, match="outside the allowlist"):
        retrieve_database(
            database_id="sample",
            sql="SELECT id, secret FROM users",
            connections_path=config_path,
        )


def test_list_database_connections_redacts_secrets(tmp_path):
    db_path = tmp_path / "sample.db"
    config_path = tmp_path / "database_connections.yaml"
    config_path.write_text(
        "connections:\n"
        "  analytics:\n"
        "    dialect: sqlite\n"
        f"    path: {db_path}\n"
        "    password: super-secret\n"
        "    api_key: sk-test\n"
        "    nested:\n"
        "      access_token: token-value\n"
        "      host: localhost\n",
        encoding="utf-8",
    )

    connections = list_database_connections(config_path)

    assert connections["analytics"]["dialect"] == "sqlite"
    assert connections["analytics"]["path"] == str(db_path)
    assert connections["analytics"]["password"] == "[redacted]"
    assert connections["analytics"]["api_key"] == "[redacted]"
    assert connections["analytics"]["nested"]["access_token"] == "[redacted]"
    assert connections["analytics"]["nested"]["host"] == "localhost"


def test_inspect_database_connections_returns_ids_and_audit(tmp_path):
    config_path = tmp_path / "database_connections.yaml"
    config_path.write_text(
        "connections:\n"
        "  b_db:\n"
        "    dialect: sqlite\n"
        "    path: /tmp/b.sqlite\n"
        "  a_db:\n"
        "    dialect: sqlite\n"
        "    path: /tmp/a.sqlite\n",
        encoding="utf-8",
    )

    result = inspect_database_connections(connections_path=config_path)

    assert result["success"] is True
    assert result["read_only"] is True
    assert result["connection_ids"] == ["a_db", "b_db"]
    assert result["connection_count"] == 2
    assert result["audit"]["action"] == "connections"


def test_sqlite_test_connection_opens_read_only_and_reports_schema(tmp_path):
    db_path = tmp_path / "sample.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
    conn.commit()
    conn.close()

    config_path = tmp_path / "database_connections.yaml"
    config_path.write_text(
        "connections:\n"
        "  sample:\n"
        "    dialect: sqlite\n"
        f"    path: {db_path}\n",
        encoding="utf-8",
    )

    result = test_database_connection(
        database_id="sample",
        connections_path=config_path,
    )

    assert result["success"] is True
    assert result["status"] == "ok"
    assert result["dialect"] == "sqlite"
    assert result["read_only"] is True
    assert result["can_execute"] is True
    assert result["execution_checked"] is True
    assert result["schema_available"] is True
    assert result["table_count"] == 1
    assert result["table_names"] == ["users"]
    assert result["probe"] == {"executed": True, "row_count": 1}
    assert result["connection"]["path"] == str(db_path)
    assert result["audit"]["action"] == "test_connection"
    assert result["audit"]["row_count"] == 1


def test_metadata_only_test_connection_validates_configured_schema(tmp_path):
    config_path = tmp_path / "database_connections.yaml"
    config_path.write_text(
        "connections:\n"
        "  warehouse:\n"
        "    dialect: postgres\n"
        "    host: localhost\n"
        "    schema:\n"
        "      tables:\n"
        "        users:\n"
        "          columns:\n"
        "            id: INTEGER\n",
        encoding="utf-8",
    )

    result = test_database_connection(
        database_id="warehouse",
        connections_path=config_path,
    )

    assert result["success"] is True
    assert result["dialect"] == "postgres"
    assert result["can_execute"] is False
    assert result["execution_checked"] is False
    assert result["schema_available"] is True
    assert result["table_names"] == ["users"]
    assert result["probe"] == {"executed": False, "row_count": 0}
    assert result["connection"]["path"] == "[redacted]"


def test_postgres_test_connection_executes_probe_when_asyncpg_available(
    monkeypatch, tmp_path
):
    config_path = tmp_path / "database_connections.yaml"
    config_path.write_text(
        "connections:\n"
        "  warehouse:\n"
        "    dialect: postgres\n"
        "    host: localhost\n"
        "    database: analytics\n",
        encoding="utf-8",
    )
    calls = {}

    class FakeConnection:
        async def execute(self, sql):
            calls.setdefault("execute", []).append(sql)

        async def fetch(self, sql, *parameters):
            calls["fetch_sql"] = sql
            calls["parameters"] = parameters
            return [{"healthcheck_ok": 1}]

        async def close(self):
            calls["closed"] = True

    async def fake_connect(**kwargs):
        calls["connect"] = kwargs
        return FakeConnection()

    monkeypatch.setitem(
        sys.modules,
        "asyncpg",
        types.SimpleNamespace(connect=fake_connect),
    )

    result = test_database_connection(
        database_id="warehouse",
        connections_path=config_path,
    )

    assert result["success"] is True
    assert result["dialect"] == "postgres"
    assert result["can_execute"] is True
    assert result["execution_checked"] is True
    assert result["schema_available"] is False
    assert result["probe"] == {"executed": True, "row_count": 1}
    assert result["connection"]["path"] == "[redacted]"
    assert calls["execute"] == ["BEGIN READ ONLY", "ROLLBACK"]
    assert calls["fetch_sql"] == "SELECT 1 AS healthcheck_ok"
    assert calls["parameters"] == ()
    assert calls["closed"] is True


def test_metadata_only_test_connection_requires_schema(tmp_path):
    config_path = tmp_path / "database_connections.yaml"
    config_path.write_text(
        "connections:\n"
        "  warehouse:\n"
        "    dialect: postgres\n",
        encoding="utf-8",
    )

    with pytest.raises(DatabaseRetrievalError, match="requires configured schema"):
        test_database_connection(database_id="warehouse", connections_path=config_path)


def test_database_retrieve_connections_action_does_not_require_database_id(monkeypatch):
    monkeypatch.setattr(
        "tools.database_retriever.inspect_database_connections",
        lambda: {
            "success": True,
            "connection_ids": ["sample"],
            "connection_count": 1,
            "connections": {"sample": {"dialect": "sqlite"}},
            "audit": {"action": "connections"},
        },
    )

    result = json.loads(_handle_database_retrieve({"action": "connections"}))

    assert result["success"] is True
    assert result["connection_ids"] == ["sample"]


def test_database_retrieve_defaults_to_connections_without_database_id(monkeypatch):
    monkeypatch.setattr(
        "tools.database_retriever.inspect_database_connections",
        lambda: {
            "success": True,
            "connection_ids": [],
            "connection_count": 0,
            "connections": {},
            "audit": {"action": "connections"},
        },
    )

    result = json.loads(_handle_database_retrieve({}))

    assert result["success"] is True
    assert result["audit"]["action"] == "connections"


def test_database_retrieve_healthcheck_alias_requires_database_id():
    result = json.loads(_handle_database_retrieve({"action": "healthcheck"}))

    assert "database_id is required" in result["error"]


def test_database_retrieve_test_connection_action_returns_healthcheck(monkeypatch):
    seen = {}

    def fake_test_connection(**kwargs):
        seen.update(kwargs)
        return {
            "success": True,
            "database_id": kwargs["database_id"],
            "status": "ok",
            "audit": {"action": "test_connection"},
        }

    monkeypatch.setattr(
        "tools.database_retriever.test_database_connection",
        fake_test_connection,
    )

    result = json.loads(
        _handle_database_retrieve(
            {
                "action": "healthcheck",
                "database_id": "sample",
            }
        )
    )

    assert result["success"] is True
    assert result["status"] == "ok"
    assert result["audit"]["action"] == "test_connection"
    assert seen["database_id"] == "sample"


def test_database_retrieve_schema_requires_database_id():
    result = json.loads(_handle_database_retrieve({"action": "schema"}))

    assert "database_id is required" in result["error"]


def test_database_retrieve_validate_action_returns_dry_run(monkeypatch):
    seen = {}

    def fake_validate(**kwargs):
        seen.update(kwargs)
        return {
            "success": True,
            "database_id": kwargs["database_id"],
            "sql": "SELECT id FROM users LIMIT 10",
            "executed": False,
            "parameter_count": 1,
            "offset": 5,
            "audit": {"action": "validate", "parameter_count": 1},
        }

    monkeypatch.setattr(
        "tools.database_retriever.validate_database_query",
        fake_validate,
    )

    result = json.loads(
        _handle_database_retrieve(
            {
                "action": "validate",
                "database_id": "sample",
                "sql": "SELECT id FROM users WHERE name = ?",
                "query_parameters": ["Ada"],
                "offset": 5,
            }
        )
    )

    assert result["success"] is True
    assert result["database_id"] == "sample"
    assert result["executed"] is False
    assert result["parameter_count"] == 1
    assert result["offset"] == 5
    assert seen["query_parameters"] == ["Ada"]
    assert seen["offset"] == 5


def test_database_retrieve_explain_action_returns_execution_plan(monkeypatch):
    seen = {}

    def fake_explain(**kwargs):
        seen.update(kwargs)
        return {
            "success": True,
            "data_query_executed": False,
            "plan": [{"detail": "SCAN users"}],
            "parameter_count": 1,
            "offset": 5,
            "audit": {"action": "explain", "parameter_count": 1},
        }

    monkeypatch.setattr(
        "tools.database_retriever.explain_database_query",
        fake_explain,
    )

    result = json.loads(
        _handle_database_retrieve(
            {
                "action": "explain",
                "database_id": "sample",
                "sql": "SELECT id FROM users WHERE name = ?",
                "query_parameters": ["Ada"],
                "offset": 5,
            }
        )
    )

    assert result["success"] is True
    assert result["data_query_executed"] is False
    assert result["audit"]["action"] == "explain"
    assert seen["query_parameters"] == ["Ada"]
    assert seen["offset"] == 5


def test_database_retrieve_retrieve_action_forwards_result_shaping(monkeypatch):
    seen = {}

    def fake_retrieve(**kwargs):
        seen.update(kwargs)
        return {
            "success": True,
            "rows": [{"name": "Ada"}],
            "result_shape": {
                "included_columns": ["name"],
                "max_cell_chars": 10,
            },
            "audit": {"action": "retrieve"},
        }

    monkeypatch.setattr(
        "tools.database_retriever.retrieve_database_for_request",
        fake_retrieve,
    )

    result = json.loads(
        _handle_database_retrieve(
            {
                "action": "retrieve",
                "database_id": "sample",
                "user_request": "show users",
                "include_columns": ["name"],
                "max_cell_chars": 10,
            }
        )
    )

    assert result["success"] is True
    assert result["result_shape"]["included_columns"] == ["name"]
    assert seen["include_columns"] == ["name"]
    assert seen["max_cell_chars"] == 10


def test_database_retrieve_validate_requires_sql():
    result = json.loads(
        _handle_database_retrieve({"action": "validate", "database_id": "sample"})
    )

    assert result["error"] == "sql is required when action='validate'"


def test_database_retrieve_schema_allows_connections_without_required_database_id():
    parameters = DATABASE_RETRIEVE_SCHEMA["parameters"]

    assert "required" not in parameters
    assert "connections" in parameters["properties"]["action"]["enum"]
    assert "dialects" in parameters["properties"]["action"]["enum"]
    assert "test_connection" in parameters["properties"]["action"]["enum"]
    assert "healthcheck" in parameters["properties"]["action"]["enum"]
    assert "validate" in parameters["properties"]["action"]["enum"]
    assert "explain" in parameters["properties"]["action"]["enum"]
    assert parameters["properties"]["offset"]["minimum"] == 0
    assert parameters["properties"]["include_columns"]["items"]["type"] == "string"
    assert parameters["properties"]["max_cell_chars"]["maximum"] >= 4000


def test_schema_context_handles_empty_accessible_schema():
    class DummySchema:
        database_id = "empty"
        dialect = "sqlite"
        tables = ()

    assert "[no accessible tables]" in build_schema_context(DummySchema)


def test_sql_generation_prompt_is_read_only_and_schema_grounded():
    messages = build_sql_generation_messages(
        user_request="show active users",
        schema_context="Database ID: sample\nTables:\n- users: id INTEGER, name TEXT",
        max_rows=50,
    )

    assert messages[0]["role"] == "system"
    assert "Return exactly one SQL statement" in messages[0]["content"]
    assert "Only SELECT queries are allowed" in messages[0]["content"]
    assert "LIMIT no greater than 50" in messages[0]["content"]
    assert "show active users" in messages[1]["content"]
    assert "- users:" in messages[1]["content"]


def test_generate_sql_for_request_validates_llm_output(tmp_path):
    db_path = tmp_path / "sample.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, enabled INTEGER)")
    conn.commit()
    conn.close()

    config_path = tmp_path / "database_connections.yaml"
    config_path.write_text(
        "connections:\n"
        "  sample:\n"
        "    dialect: sqlite\n"
        f"    path: {db_path}\n",
        encoding="utf-8",
    )

    seen_messages = []

    def fake_generator(messages):
        seen_messages.extend(messages)
        return "```sql\nSELECT id, name FROM users WHERE enabled = 1\n```"

    generated = generate_sql_for_request(
        database_id="sample",
        user_request="show active users",
        connections_path=config_path,
        max_rows=5,
        offset=10,
        llm_sql_generator=fake_generator,
    )

    assert seen_messages
    assert generated.raw_sql == "SELECT id, name FROM users WHERE enabled = 1"
    assert generated.sql == (
        "SELECT id, name FROM users WHERE enabled = 1 LIMIT 5 OFFSET 10"
    )
    assert generated.referenced_tables == ("users",)
    assert "LIMIT 5 was added" in generated.warnings
    assert "OFFSET 10 was added" in generated.warnings


def test_generate_sql_for_request_rejects_unsafe_llm_output(tmp_path):
    db_path = tmp_path / "sample.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
    conn.commit()
    conn.close()

    config_path = tmp_path / "database_connections.yaml"
    config_path.write_text(
        "connections:\n"
        "  sample:\n"
        "    dialect: sqlite\n"
        f"    path: {db_path}\n",
        encoding="utf-8",
    )

    with pytest.raises(DatabaseRetrievalError):
        generate_sql_for_request(
            database_id="sample",
            user_request="remove users",
            connections_path=config_path,
            llm_sql_generator=lambda _messages: "DELETE FROM users",
        )


def test_retrieve_database_for_request_generates_and_executes(tmp_path):
    db_path = tmp_path / "sample.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, enabled INTEGER)")
    conn.executemany(
        "INSERT INTO users (name, enabled) VALUES (?, ?)",
        [("Ada", 1), ("Grace", 1), ("Linus", 0)],
    )
    conn.commit()
    conn.close()

    config_path = tmp_path / "database_connections.yaml"
    config_path.write_text(
        "connections:\n"
        "  sample:\n"
        "    dialect: sqlite\n"
        f"    path: {db_path}\n",
        encoding="utf-8",
    )

    result = retrieve_database_for_request(
        database_id="sample",
        user_request="show enabled users",
        connections_path=config_path,
        max_rows=1,
        offset=1,
        llm_sql_generator=lambda _messages: (
            "SELECT id, name FROM users WHERE enabled = 1 ORDER BY id"
        ),
    )

    assert result["success"] is True
    assert result["generated_sql"] == (
        "SELECT id, name FROM users WHERE enabled = 1 ORDER BY id"
    )
    assert result["sql"].endswith("LIMIT 1 OFFSET 1")
    assert result["offset"] == 1
    assert result["pagination"] == {
        "limit": 1,
        "offset": 1,
        "returned": 1,
        "next_offset": 2,
    }
    assert result["rows"] == [{"id": 2, "name": "Grace"}]


def test_repair_prompt_includes_failed_sql_and_error():
    messages = build_sql_repair_messages(
        user_request="show users",
        schema_context="Database ID: sample\nTables:\n- users: id INTEGER",
        previous_sql="DELETE FROM users",
        error="Only SELECT allowed",
    )

    assert messages[-2]["role"] == "assistant"
    assert messages[-2]["content"] == "DELETE FROM users"
    assert messages[-1]["role"] == "user"
    assert "Only SELECT allowed" in messages[-1]["content"]
    assert "corrected read-only SQL" in messages[-1]["content"]


def test_generate_sql_for_request_repairs_invalid_llm_output(tmp_path):
    db_path = tmp_path / "sample.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
    conn.commit()
    conn.close()

    config_path = tmp_path / "database_connections.yaml"
    config_path.write_text(
        "connections:\n"
        "  sample:\n"
        "    dialect: sqlite\n"
        f"    path: {db_path}\n",
        encoding="utf-8",
    )

    responses = iter(["DELETE FROM users", "SELECT id, name FROM users"])
    seen_messages = []

    def fake_generator(messages):
        seen_messages.append(messages)
        return next(responses)

    generated = generate_sql_for_request(
        database_id="sample",
        user_request="show users",
        connections_path=config_path,
        max_rows=3,
        llm_sql_generator=fake_generator,
    )

    assert generated.sql == "SELECT id, name FROM users LIMIT 3"
    assert len(generated.attempts) == 2
    assert generated.attempts[0]["sql"] == "DELETE FROM users"
    assert "error" in generated.attempts[0]
    assert generated.attempts[1]["sql"] == "SELECT id, name FROM users"
    assert "error" not in generated.attempts[1]
    assert "previous SQL failed" in seen_messages[1][-1]["content"]


def test_generate_sql_for_request_exhausts_repair_attempts(tmp_path):
    db_path = tmp_path / "sample.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
    conn.commit()
    conn.close()

    config_path = tmp_path / "database_connections.yaml"
    config_path.write_text(
        "connections:\n"
        "  sample:\n"
        "    dialect: sqlite\n"
        f"    path: {db_path}\n",
        encoding="utf-8",
    )

    with pytest.raises(DatabaseRetrievalError, match="failed after 2 attempt"):
        generate_sql_for_request(
            database_id="sample",
            user_request="show users",
            connections_path=config_path,
            llm_sql_generator=lambda _messages: "DELETE FROM users",
            repair_attempts=1,
        )


def test_retrieve_database_for_request_repairs_execution_error(tmp_path):
    db_path = tmp_path / "sample.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
    conn.executemany("INSERT INTO users (name) VALUES (?)", [("Ada",), ("Grace",)])
    conn.commit()
    conn.close()

    config_path = tmp_path / "database_connections.yaml"
    config_path.write_text(
        "connections:\n"
        "  sample:\n"
        "    dialect: sqlite\n"
        f"    path: {db_path}\n",
        encoding="utf-8",
    )

    responses = iter(
        [
            "SELECT COUNT(*) AS total FROM users HAVING missing_column > 0",
            "SELECT COUNT(*) AS total FROM users",
        ]
    )

    result = retrieve_database_for_request(
        database_id="sample",
        user_request="count users",
        connections_path=config_path,
        llm_sql_generator=lambda _messages: next(responses),
        repair_attempts=1,
    )

    assert result["success"] is True
    assert result["generated_sql"] == "SELECT COUNT(*) AS total FROM users"
    assert result["rows"] == [{"total": 2}]
    assert len(result["attempts"]) == 2
    assert result["attempts"][0]["sql"] == (
        "SELECT COUNT(*) AS total FROM users HAVING missing_column > 0"
    )
    assert "error" in result["attempts"][0]
    assert result["attempts"][1]["sql"] == "SELECT COUNT(*) AS total FROM users"
    assert "error" not in result["attempts"][1]


def test_sql_literal_redaction_for_audit():
    redacted = redact_sql_literals(
        "SELECT id FROM users WHERE email = 'ada@example.com' "
        "AND enabled = true AND age > 42"
    )

    assert "ada@example.com" not in redacted
    assert "42" not in redacted
    assert "users" in redacted
    assert "email" in redacted


def test_audit_event_redacts_sql_and_attempts():
    audit = build_audit_event(
        action="retrieve",
        database_id="sample",
        source_dialect="postgres",
        target_dialect="sqlite",
        sql="SELECT * FROM users WHERE token = 'sk-secret-value'",
        generated_sql="SELECT * FROM users WHERE token = 'sk-secret-value'",
        attempts=[
            {
                "sql": "SELECT * FROM users WHERE token = 'sk-secret-value'",
                "error": "no such column: token",
            }
        ],
        row_count=1,
        execution_ms=2,
    )

    assert audit["component"] == "database_retriever"
    assert audit["read_only"] is True
    assert audit["row_count"] == 1
    assert audit["execution_ms"] == 2
    rendered = str(audit)
    assert "sk-secret-value" not in rendered
    assert "users" in rendered


def test_retrieve_result_contains_redacted_audit_metadata(tmp_path):
    db_path = tmp_path / "sample.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT)")
    conn.execute("INSERT INTO users (email) VALUES ('ada@example.com')")
    conn.commit()
    conn.close()

    config_path = tmp_path / "database_connections.yaml"
    config_path.write_text(
        "connections:\n"
        "  sample:\n"
        "    dialect: sqlite\n"
        f"    path: {db_path}\n",
        encoding="utf-8",
    )

    result = retrieve_database(
        database_id="sample",
        sql="SELECT id, email FROM users WHERE email = 'ada@example.com'",
        connections_path=config_path,
    )

    assert result["rows"] == [{"id": 1, "email": "ada@example.com"}]
    assert "ada@example.com" in result["sql"]
    assert "ada@example.com" not in str(result["audit"])
    assert result["audit"]["action"] == "query"
    assert result["audit"]["row_count"] == 1
