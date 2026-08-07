"""Tests for sqlite_query tool."""

import json
import os
import pytest
import tempfile
import sqlite3


class TestSqliteQuery:
    @pytest.fixture
    def temp_db(self):
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT, value INTEGER)")
        conn.execute("INSERT INTO test VALUES (1, 'alpha', 100)")
        conn.execute("INSERT INTO test VALUES (2, 'beta', 200)")
        conn.execute("INSERT INTO test VALUES (3, 'gamma', 300)")
        conn.commit()
        conn.close()
        yield db_path
        os.remove(db_path)

    def test_check_requirements(self):
        from tools.sqlite_query import check_sqlite_query_requirements
        assert check_sqlite_query_requirements() is True

    def test_select_all(self, temp_db):
        from tools.sqlite_query import sqlite_query
        output = sqlite_query(temp_db, "SELECT * FROM test")
        data = json.loads(output)
        assert data["success"] is True
        assert data["count"] == 3

    def test_select_one(self, temp_db):
        from tools.sqlite_query import sqlite_query
        output = sqlite_query(temp_db, "SELECT * FROM test WHERE id = 1", fetch="one")
        data = json.loads(output)
        assert data["success"] is True
        assert data["row"]["name"] == "alpha"

    def test_select_fetch_none(self, temp_db):
        from tools.sqlite_query import sqlite_query
        output = sqlite_query(temp_db, "SELECT * FROM test", fetch="none")
        data = json.loads(output)
        assert data["success"] is True
        assert data["count"] == 0

    def test_parameterized_query(self, temp_db):
        from tools.sqlite_query import sqlite_query
        output = sqlite_query(temp_db, "SELECT * FROM test WHERE value > :min_val", params={"min_val": 150})
        data = json.loads(output)
        assert data["success"] is True
        assert data["count"] == 2

    def test_insert(self, temp_db):
        from tools.sqlite_query import sqlite_query
        output = sqlite_query(temp_db, "INSERT INTO test (name, value) VALUES (:name, :value)",
                             params={"name": "delta", "value": 400})
        data = json.loads(output)
        assert data["success"] is True
        assert data["changes"] >= 1

    def test_update(self, temp_db):
        from tools.sqlite_query import sqlite_query
        output = sqlite_query(temp_db, "UPDATE test SET value = 999 WHERE id = 1")
        data = json.loads(output)
        assert data["success"] is True

    def test_delete(self, temp_db):
        from tools.sqlite_query import sqlite_query
        output = sqlite_query(temp_db, "DELETE FROM test WHERE id = 3")
        data = json.loads(output)
        assert data["success"] is True

    def test_db_not_found(self):
        from tools.sqlite_query import sqlite_query
        output = sqlite_query("/nonexistent/path/db.sqlite", "SELECT 1")
        data = json.loads(output)
        assert data["success"] is False

    def test_empty_query(self, temp_db):
        from tools.sqlite_query import sqlite_query
        output = sqlite_query(temp_db, "")
        data = json.loads(output)
        assert data["success"] is False
        assert "empty" in data["error"].lower()

    def test_blocked_query(self, temp_db):
        from tools.sqlite_query import sqlite_query
        output = sqlite_query(temp_db, "DROP TABLE test")
        data = json.loads(output)
        assert data["success"] is False
        assert "only" in data["error"].lower()

    def test_invalid_fetch(self, temp_db):
        from tools.sqlite_query import sqlite_query
        output = sqlite_query(temp_db, "SELECT 1", fetch="invalid")
        data = json.loads(output)
        assert data["success"] is False


class TestSqliteQuerySchema:
    def test_schema_has_required_fields(self):
        from tools.sqlite_query import SQLITE_QUERY_SCHEMA
        assert SQLITE_QUERY_SCHEMA["name"] == "sqlite_query"
        props = SQLITE_QUERY_SCHEMA["parameters"]["properties"]
        assert "db_path" in props
        assert "query" in props
        assert "params" in props
        assert "fetch" in props