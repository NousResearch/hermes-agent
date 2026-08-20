"""Tests for database_schema tool."""

import json
import os
import pytest
import tempfile
import sqlite3


class TestDatabaseSchema:
    @pytest.fixture
    def temp_db(self):
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT)")
        conn.execute("CREATE INDEX idx_users_name ON users(name)")
        conn.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, total REAL)")
        conn.commit()
        conn.close()
        yield db_path
        os.remove(db_path)

    def test_check_requirements(self):
        from tools.database_schema import check_database_schema_requirements
        assert check_database_schema_requirements() is True

    def test_list_tables(self, temp_db):
        from tools.database_schema import database_schema
        output = database_schema(temp_db, operation="list_tables")
        data = json.loads(output)
        assert data["success"] is True
        assert data["operation"] == "list_tables"
        assert data["count"] == 2
        table_names = [t["table_name"] for t in data["tables"]]
        assert "users" in table_names
        assert "orders" in table_names

    def test_describe_table(self, temp_db):
        from tools.database_schema import database_schema
        output = database_schema(temp_db, operation="describe_table", table_name="users")
        data = json.loads(output)
        assert data["success"] is True
        assert data["operation"] == "describe_table"
        assert data["table_name"] == "users"
        assert data["column_count"] >= 3

    def test_describe_nonexistent_table(self, temp_db):
        from tools.database_schema import database_schema
        output = database_schema(temp_db, operation="describe_table", table_name="nonexistent")
        data = json.loads(output)
        assert data["success"] is True
        assert "warning" in data

    def test_list_indexes(self, temp_db):
        from tools.database_schema import database_schema
        output = database_schema(temp_db, operation="list_indexes", table_name="users")
        data = json.loads(output)
        assert data["success"] is True
        assert data["operation"] == "list_indexes"
        assert data["count"] >= 1

    def test_list_indexes_no_table_name(self, temp_db):
        from tools.database_schema import database_schema
        output = database_schema(temp_db, operation="list_indexes")
        data = json.loads(output)
        assert data["success"] is False
        assert "table_name required" in data["error"]

    def test_db_not_found(self):
        from tools.database_schema import database_schema
        output = database_schema("/nonexistent/db.sqlite", operation="list_tables")
        data = json.loads(output)
        assert data["success"] is False

    def test_invalid_operation(self, temp_db):
        from tools.database_schema import database_schema
        output = database_schema(temp_db, operation="invalid_op")
        data = json.loads(output)
        assert data["success"] is False

    def test_path_validation(self):
        from tools.database_schema import database_schema
        with tempfile.TemporaryDirectory() as tmpdir:
            output = database_schema(tmpdir, operation="list_tables")
            data = json.loads(output)
            assert data["success"] is False
            assert "not a file" in data["error"].lower()


class TestDatabaseSchemaSchema:
    def test_schema_has_required_fields(self):
        from tools.database_schema import DATABASE_SCHEMA_SCHEMA
        assert DATABASE_SCHEMA_SCHEMA["name"] == "database_schema"
        props = DATABASE_SCHEMA_SCHEMA["parameters"]["properties"]
        assert "db_path" in props
        assert "operation" in props
        assert "table_name" in props
        assert "task_id" in props

    def test_schema_operation_enum(self):
        from tools.database_schema import DATABASE_SCHEMA_SCHEMA
        props = DATABASE_SCHEMA_SCHEMA["parameters"]["properties"]
        assert props["operation"]["enum"] == ["list_tables", "describe_table", "list_indexes"]