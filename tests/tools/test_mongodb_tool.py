"""Tests for the MongoDB tool module."""

import json
from unittest.mock import MagicMock, patch

import pytest

from tools.mongodb_tool import (
    _check_mongodb_available,
    _check_mongodb_write_available,
    _parse_json,
    _resolve_database,
    _validate_name,
    mongo_aggregate,
    mongo_find,
    mongo_list_collections,
    mongo_write,
)


class TestAvailability:
    def test_unavailable_without_uri(self, monkeypatch):
        monkeypatch.delenv("MONGODB_URI", raising=False)
        assert _check_mongodb_available() is False

    def test_available_with_uri(self, monkeypatch):
        monkeypatch.setenv("MONGODB_URI", "mongodb://localhost:27017/test")
        assert _check_mongodb_available() is True

    def test_write_unavailable_when_read_only(self, monkeypatch):
        monkeypatch.setenv("MONGODB_URI", "mongodb://localhost:27017/test")
        with patch("tools.mongodb_tool.load_config", return_value={"mongodb": {"read_only": True}}):
            assert _check_mongodb_write_available() is False

    def test_write_available_when_not_read_only(self, monkeypatch):
        monkeypatch.setenv("MONGODB_URI", "mongodb://localhost:27017/test")
        with patch("tools.mongodb_tool.load_config", return_value={"mongodb": {"read_only": False}}):
            assert _check_mongodb_write_available() is True


class TestValidation:
    def test_validate_name_accepts_simple(self):
        assert _validate_name("orders", "collection") is None

    def test_validate_name_rejects_empty(self):
        assert _validate_name("", "collection") == "collection is required"

    def test_validate_name_rejects_special_chars(self):
        assert "Invalid" in _validate_name("bad-name", "collection")

    def test_parse_json_empty_returns_default(self):
        value, err = _parse_json("", "filter", {})
        assert value == {}
        assert err is None

    def test_parse_json_invalid(self):
        value, err = _parse_json("{bad", "filter", {})
        assert "Invalid JSON" in err

    def test_resolve_database_requires_name(self, monkeypatch):
        monkeypatch.delenv("MONGODB_URI", raising=False)
        with patch("tools.mongodb_tool._mongodb_settings", return_value={"default_database": "", "max_rows": 100}):
            db, err = _resolve_database("")
        assert db is None
        assert "database is required" in err

    def test_resolve_database_uses_default(self):
        with patch("tools.mongodb_tool._mongodb_settings", return_value={"default_database": "app", "max_rows": 100}):
            db, err = _resolve_database("")
        assert db == "app"
        assert err is None


class TestMongoListCollections:
    def test_requires_uri(self, monkeypatch):
        monkeypatch.delenv("MONGODB_URI", raising=False)
        result = json.loads(mongo_list_collections())
        assert "error" in result

    def test_lists_databases(self, monkeypatch):
        monkeypatch.setenv("MONGODB_URI", "mongodb://localhost:27017/test")
        mock_client = MagicMock()
        mock_client.list_database_names.return_value = ["admin", "app"]
        with patch("tools.mongodb_tool._get_client", return_value=mock_client):
            result = json.loads(mongo_list_collections())
        assert result["databases"] == ["admin", "app"]
        assert result["count"] == 2

    def test_lists_collections(self, monkeypatch):
        monkeypatch.setenv("MONGODB_URI", "mongodb://localhost:27017/test")
        mock_client = MagicMock()
        mock_client.__getitem__.return_value.list_collection_names.return_value = ["users", "orders"]
        with patch("tools.mongodb_tool._get_client", return_value=mock_client):
            result = json.loads(mongo_list_collections(database="app"))
        assert result["collections"] == ["orders", "users"]
        assert result["database"] == "app"


class TestMongoFind:
    def test_find_returns_documents(self, monkeypatch):
        monkeypatch.setenv("MONGODB_URI", "mongodb://localhost:27017/test")
        mock_coll = MagicMock()
        mock_coll.find.return_value.limit.return_value = [{"_id": "1", "name": "alice"}]
        mock_db = MagicMock()
        mock_db.__getitem__.return_value = mock_coll
        mock_client = MagicMock()
        mock_client.__getitem__.return_value = mock_db

        with patch("tools.mongodb_tool._get_client", return_value=mock_client), \
             patch("tools.mongodb_tool._mongodb_settings", return_value={"default_database": "app", "max_rows": 100}), \
             patch("tools.mongodb_tool._serialize_docs", return_value=[{"_id": "1", "name": "alice"}]):
            result = json.loads(mongo_find(database="app", collection="users"))

        assert result["count"] == 1
        assert result["documents"][0]["name"] == "alice"
        mock_coll.find.assert_called_once()

    def test_find_rejects_bad_filter(self, monkeypatch):
        monkeypatch.setenv("MONGODB_URI", "mongodb://localhost:27017/test")
        result = json.loads(
            mongo_find(database="app", collection="users", filter="not-json")
        )
        assert "Invalid JSON" in result["error"]

    def test_find_enforces_max_rows(self, monkeypatch):
        monkeypatch.setenv("MONGODB_URI", "mongodb://localhost:27017/test")
        mock_coll = MagicMock()
        mock_coll.find.return_value.sort.return_value.limit.return_value = []
        mock_db = MagicMock()
        mock_db.__getitem__.return_value = mock_coll
        mock_client = MagicMock()
        mock_client.__getitem__.return_value = mock_db

        with patch("tools.mongodb_tool._get_client", return_value=mock_client), \
             patch("tools.mongodb_tool._mongodb_settings", return_value={"default_database": "app", "max_rows": 25}), \
             patch("tools.mongodb_tool._serialize_docs", return_value=[]):
            result = json.loads(
                mongo_find(database="app", collection="users", limit=999)
            )

        assert result["limit"] == 25
        mock_coll.find.return_value.limit.assert_called_once_with(25)


class TestMongoAggregate:
    def test_aggregate_returns_results(self, monkeypatch):
        monkeypatch.setenv("MONGODB_URI", "mongodb://localhost:27017/test")
        mock_coll = MagicMock()
        mock_coll.aggregate.return_value = [{"total": 3}]
        mock_db = MagicMock()
        mock_db.__getitem__.return_value = mock_coll
        mock_client = MagicMock()
        mock_client.__getitem__.return_value = mock_db

        pipeline = json.dumps([{"$match": {"status": "open"}}])
        with patch("tools.mongodb_tool._get_client", return_value=mock_client), \
             patch("tools.mongodb_tool._mongodb_settings", return_value={"default_database": "app", "max_rows": 50}), \
             patch("tools.mongodb_tool._serialize_docs", return_value=[{"total": 3}]):
            result = json.loads(
                mongo_aggregate(database="app", collection="orders", pipeline=pipeline)
            )

        assert result["count"] == 1
        assert result["results"][0]["total"] == 3


class TestMongoWrite:
    def test_insert_one(self, monkeypatch):
        monkeypatch.setenv("MONGODB_URI", "mongodb://localhost:27017/test")
        mock_result = MagicMock()
        mock_result.inserted_id = "abc123"
        mock_coll = MagicMock()
        mock_coll.insert_one.return_value = mock_result
        mock_db = MagicMock()
        mock_db.__getitem__.return_value = mock_coll
        mock_client = MagicMock()
        mock_client.__getitem__.return_value = mock_db

        with patch("tools.mongodb_tool._get_client", return_value=mock_client), \
             patch("tools.mongodb_tool._mongodb_settings", return_value={"default_database": "app", "max_rows": 100}):
            result = json.loads(
                mongo_write(
                    action="insert",
                    database="app",
                    collection="users",
                    document='{"name": "bob"}',
                )
            )

        assert result["inserted_count"] == 1
        assert result["inserted_id"] == "abc123"

    def test_update_many(self, monkeypatch):
        monkeypatch.setenv("MONGODB_URI", "mongodb://localhost:27017/test")
        mock_result = MagicMock()
        mock_result.matched_count = 2
        mock_result.modified_count = 2
        mock_coll = MagicMock()
        mock_coll.update_many.return_value = mock_result
        mock_db = MagicMock()
        mock_db.__getitem__.return_value = mock_coll
        mock_client = MagicMock()
        mock_client.__getitem__.return_value = mock_db

        with patch("tools.mongodb_tool._get_client", return_value=mock_client), \
             patch("tools.mongodb_tool._mongodb_settings", return_value={"default_database": "app", "max_rows": 100}):
            result = json.loads(
                mongo_write(
                    action="update",
                    database="app",
                    collection="users",
                    filter='{"active": false}',
                    document='{"$set": {"active": true}}',
                )
            )

        assert result["matched_count"] == 2
        assert result["modified_count"] == 2

    def test_delete_many(self, monkeypatch):
        monkeypatch.setenv("MONGODB_URI", "mongodb://localhost:27017/test")
        mock_result = MagicMock()
        mock_result.deleted_count = 1
        mock_coll = MagicMock()
        mock_coll.delete_many.return_value = mock_result
        mock_db = MagicMock()
        mock_db.__getitem__.return_value = mock_coll
        mock_client = MagicMock()
        mock_client.__getitem__.return_value = mock_db

        with patch("tools.mongodb_tool._get_client", return_value=mock_client), \
             patch("tools.mongodb_tool._mongodb_settings", return_value={"default_database": "app", "max_rows": 100}):
            result = json.loads(
                mongo_write(
                    action="delete",
                    database="app",
                    collection="users",
                    filter='{"name": "ghost"}',
                )
            )

        assert result["deleted_count"] == 1

    def test_write_blocked_when_read_only(self, monkeypatch):
        monkeypatch.setenv("MONGODB_URI", "mongodb://localhost:27017/test")
        with patch("tools.mongodb_tool.load_config", return_value={"mongodb": {"read_only": True}}):
            result = json.loads(
                mongo_write(action="insert", database="app", collection="users", document="{}")
            )
        assert "read_only" in result["error"]

    def test_rejects_invalid_action(self, monkeypatch):
        monkeypatch.setenv("MONGODB_URI", "mongodb://localhost:27017/test")
        result = json.loads(
            mongo_write(action="drop", database="app", collection="users")
        )
        assert "action must be" in result["error"]
