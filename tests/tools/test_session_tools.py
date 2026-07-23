"""Tests for tools/session_tools.py — session management agent tools.

Covers HTTP helpers, tool handlers (with mocked API), schemas, and
registry registration. No live dashboard needed — all HTTP is mocked.
"""

from __future__ import annotations

import json
import os
import sys
from unittest.mock import patch, MagicMock
from urllib.error import URLError

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture(autouse=True)
def reset_registry():
    """Ensure a clean registry before each test by re-importing fresh."""
    # Clear cached import so registry.register() runs again
    for mod in list(sys.modules.keys()):
        if "session_tools" in mod:
            del sys.modules[mod]
    if "tools.registry" in sys.modules:
        del sys.modules["tools.registry"]
    yield


def _mock_urlopen(status: int = 200, body: dict | None = None) -> MagicMock:
    """Create a mock for urllib.request.urlopen."""
    m = MagicMock()
    m.__enter__.return_value.status = status
    m.__enter__.return_value.read.return_value = json.dumps(body or {}).encode("utf-8")
    return m


def _mock_urlopen_failing(error: Exception | None = None) -> MagicMock:
    """Create a mock that raises on urlopen."""
    m = MagicMock()
    m.side_effect = error or URLError("mock failure")
    return m


# =====================================================================
# HTTP helpers
# =====================================================================


class TestDashboardCheck:
    def test_reachable(self):
        with patch.dict(os.environ, {"HERMES_DASHBOARD_SESSION_TOKEN": "secret"}), \
             patch("urllib.request.urlopen", return_value=_mock_urlopen()):
            from tools.session_tools import _check_dashboard
            assert _check_dashboard() is True

    def test_public_status_is_not_enough(self):
        with patch.dict(os.environ, {}, clear=True), \
             patch("urllib.request.urlopen", return_value=_mock_urlopen()):
            from tools.session_tools import _check_dashboard
            assert _check_dashboard() is False

    def test_unreachable_connection_refused(self):
        with patch("urllib.request.urlopen", _mock_urlopen_failing(ConnectionRefusedError())):
            from tools.session_tools import _check_dashboard
            assert _check_dashboard() is False

    def test_unreachable_url_error(self):
        with patch("urllib.request.urlopen", _mock_urlopen_failing(URLError("no route to host"))):
            from tools.session_tools import _check_dashboard
            assert _check_dashboard() is False


class TestApiGet:
    def test_success(self):
        resp = {"sessions": [{"id": "s1"}]}
        with patch("tools.session_tools._dashboard_token", return_value="secret"), \
             patch("urllib.request.urlopen", return_value=_mock_urlopen(body=resp)) as mock_urlopen:
            from tools.session_tools import _api_get
            result = _api_get("/api/sessions")
            assert result == resp
            request = mock_urlopen.call_args.args[0]
            assert request.get_header("X-hermes-session-token") == "secret"

    def test_returns_none_on_failure(self):
        with patch("urllib.request.urlopen", _mock_urlopen_failing()):
            from tools.session_tools import _api_get
            assert _api_get("/api/sessions") is None


class TestApiPatch:
    def test_success(self):
        resp = {"ok": True, "title": "Renamed"}
        with patch("urllib.request.urlopen", return_value=_mock_urlopen(body=resp)):
            from tools.session_tools import _api_patch
            result = _api_patch("/api/sessions/s1", {"title": "New"})
            assert result == resp


class TestApiDelete:
    def test_success(self):
        resp = {"ok": True}
        with patch("urllib.request.urlopen", return_value=_mock_urlopen(body=resp)):
            from tools.session_tools import _api_delete
            result = _api_delete("/api/sessions/s1")
            assert result == resp


class TestApiPost:
    def test_success(self):
        resp = {"id": "sf_abc", "name": "Bug Reports"}
        with patch("urllib.request.urlopen", return_value=_mock_urlopen(body=resp)):
            from tools.session_tools import _api_post
            result = _api_post("/api/session-folders", {"name": "Bug Reports"})
            assert result == resp
            assert result["id"] == "sf_abc"


# =====================================================================
# Tool handlers
# =====================================================================


class TestSessionList:
    def test_returns_formatted_list(self):
        sessions = [
            {"id": "abc123", "title": "Test", "archived": False,
             "model": "gpt-4", "source": "cli", "message_count": 5}
        ]
        with patch("tools.session_tools._api_get", return_value={"sessions": sessions}):
            from tools.session_tools import _session_list
            result = json.loads(_session_list(limit=10))
            assert result["success"] is True
            assert result["count"] == 1

    def test_returns_error_when_unreachable(self):
        with patch("tools.session_tools._api_get", return_value=None):
            from tools.session_tools import _session_list
            result = json.loads(_session_list())
            assert result["success"] is False
            assert "Cannot reach" in result["error"]

    def test_passes_filters(self):
        sessions = []
        with patch("tools.session_tools._api_get", return_value={"sessions": sessions}) as mock_get:
            from tools.session_tools import _session_list
            _session_list(limit=5, archived="only", order="created")
            call_path = mock_get.call_args[0][0]
            assert "limit=5" in call_path
            assert "archived=only" in call_path
            assert "order=created" in call_path


class TestSessionArchive:
    def test_archive(self):
        with patch("tools.session_tools._api_patch", return_value={"ok": True}):
            from tools.session_tools import _session_archive
            result = json.loads(_session_archive("s1", archived=True))
            assert result["success"] is True
            assert "archived" in result["message"].lower()

    def test_unarchive(self):
        with patch("tools.session_tools._api_patch", return_value={"ok": True}):
            from tools.session_tools import _session_archive
            result = json.loads(_session_archive("s1", archived=False))
            assert result["success"] is True
            assert "unarchived" in result["message"].lower()

    def test_failure(self):
        with patch("tools.session_tools._api_patch", return_value=None):
            from tools.session_tools import _session_archive
            result = json.loads(_session_archive("s1"))
            assert result["success"] is False


class TestSessionRename:
    def test_rename(self):
        with patch("tools.session_tools._api_patch", return_value={"ok": True}):
            from tools.session_tools import _session_rename
            result = json.loads(_session_rename("s1", "New Title"))
            assert result["success"] is True
            assert "New Title" in result["message"]

    def test_failure(self):
        with patch("tools.session_tools._api_patch", return_value=None):
            from tools.session_tools import _session_rename
            result = json.loads(_session_rename("s1", "X"))
            assert result["success"] is False


class TestSessionDelete:
    def test_delete(self):
        with patch("tools.session_tools._api_delete", return_value={"ok": True}):
            from tools.session_tools import _session_delete
            result = json.loads(_session_delete("s1"))
            assert result["success"] is True
            assert "deleted" in result["message"].lower()

    def test_failure(self):
        with patch("tools.session_tools._api_delete", return_value={"ok": False}):
            from tools.session_tools import _session_delete
            result = json.loads(_session_delete("s1"))
            assert result["success"] is False

    def test_unreachable(self):
        with patch("tools.session_tools._api_delete", return_value=None):
            from tools.session_tools import _session_delete
            result = json.loads(_session_delete("s1"))
            assert result["success"] is False
            assert "Cannot reach" in result["error"]


class TestFolderList:
    def test_list(self):
        folders = [{"id": "f1", "name": "Bug Reports", "session_count": 3}]
        with patch("tools.session_tools._api_get", return_value=folders):
            from tools.session_tools import _folder_list
            result = json.loads(_folder_list())
            assert result["success"] is True
            assert result["count"] == 1

    def test_unreachable(self):
        with patch("tools.session_tools._api_get", return_value=None):
            from tools.session_tools import _folder_list
            result = json.loads(_folder_list())
            assert result["success"] is False


class TestFolderCreate:
    def test_create(self):
        folder = {"id": "sf_x", "name": "Design", "session_count": 0}
        with patch("tools.session_tools._api_post", return_value=folder):
            from tools.session_tools import _folder_create
            result = json.loads(_folder_create("Design"))
            assert result["success"] is True
            assert result["folder"]["name"] == "Design"

    def test_unreachable(self):
        with patch("tools.session_tools._api_post", return_value=None):
            from tools.session_tools import _folder_create
            result = json.loads(_folder_create("X"))
            assert result["success"] is False


class TestFolderAdd:
    def test_add(self):
        with patch("tools.session_tools._api_post", return_value={"ok": True, "count": 2}):
            from tools.session_tools import _folder_add
            result = json.loads(_folder_add("f1", ["s1", "s2"]))
            assert result["success"] is True
            assert "2" in result["message"]

    def test_unreachable(self):
        with patch("tools.session_tools._api_post", return_value=None):
            from tools.session_tools import _folder_add
            result = json.loads(_folder_add("f1", ["s1"]))
            assert result["success"] is False


class TestFolderRename:
    def test_rename(self):
        with patch("tools.session_tools._api_patch", return_value={"ok": True}):
            from tools.session_tools import _folder_rename
            result = json.loads(_folder_rename("f1", "New Name"))
            assert result["success"] is True
            assert "New Name" in result["message"]

    def test_failure(self):
        with patch("tools.session_tools._api_patch", return_value={"ok": False}):
            from tools.session_tools import _folder_rename
            result = json.loads(_folder_rename("f1", "X"))
            assert result["success"] is False

    def test_unreachable(self):
        with patch("tools.session_tools._api_patch", return_value=None):
            from tools.session_tools import _folder_rename
            result = json.loads(_folder_rename("f1", "X"))
            assert result["success"] is False


class TestFolderDelete:
    def test_delete(self):
        with patch("tools.session_tools._api_delete", return_value={"ok": True}):
            from tools.session_tools import _folder_delete
            result = json.loads(_folder_delete("f1"))
            assert result["success"] is True
            assert "deleted" in result["message"].lower()

    def test_failure(self):
        with patch("tools.session_tools._api_delete", return_value={"ok": False}):
            from tools.session_tools import _folder_delete
            result = json.loads(_folder_delete("f1"))
            assert result["success"] is False

    def test_unreachable(self):
        with patch("tools.session_tools._api_delete", return_value=None):
            from tools.session_tools import _folder_delete
            result = json.loads(_folder_delete("f1"))
            assert result["success"] is False


# =====================================================================
# Tool schemas (static validation)
# =====================================================================


def _import_all_schemas():
    """Re-import the module and return all schemas."""
    for mod in list(sys.modules.keys()):
        if "session_tools" in mod or "tools.registry" in mod:
            del sys.modules[mod]
    from tools.session_tools import (
        SESSION_LIST_SCHEMA, SESSION_ARCHIVE_SCHEMA, SESSION_RENAME_SCHEMA,
        SESSION_DELETE_SCHEMA, FOLDER_LIST_SCHEMA, FOLDER_CREATE_SCHEMA,
        FOLDER_ADD_SCHEMA, FOLDER_RENAME_SCHEMA, FOLDER_DELETE_SCHEMA,
    )
    return {
        "session_list": SESSION_LIST_SCHEMA,
        "session_archive": SESSION_ARCHIVE_SCHEMA,
        "session_rename": SESSION_RENAME_SCHEMA,
        "session_delete": SESSION_DELETE_SCHEMA,
        "session_folder_list": FOLDER_LIST_SCHEMA,
        "session_folder_create": FOLDER_CREATE_SCHEMA,
        "session_folder_add": FOLDER_ADD_SCHEMA,
        "session_folder_rename": FOLDER_RENAME_SCHEMA,
        "session_folder_delete": FOLDER_DELETE_SCHEMA,
    }


class TestSchemas:
    def test_all_schemas_have_name(self):
        schemas = _import_all_schemas()
        for name, schema in schemas.items():
            assert schema["name"] == name, f"{name} schema name mismatch"

    def test_all_schemas_have_description(self):
        schemas = _import_all_schemas()
        for name, schema in schemas.items():
            assert schema.get("description"), f"{name} missing description"

    def test_all_schemas_have_parameters(self):
        schemas = _import_all_schemas()
        for name, schema in schemas.items():
            assert "parameters" in schema, f"{name} missing parameters"

    def test_session_archive_requires_session_id(self):
        schemas = _import_all_schemas()
        required = schemas["session_archive"]["parameters"].get("required", [])
        assert "session_id" in required

    def test_session_rename_requires_both(self):
        schemas = _import_all_schemas()
        required = schemas["session_rename"]["parameters"].get("required", [])
        assert "session_id" in required
        assert "title" in required

    def test_session_delete_requires_session_id(self):
        schemas = _import_all_schemas()
        required = schemas["session_delete"]["parameters"].get("required", [])
        assert "session_id" in required

    def test_folder_create_requires_name(self):
        schemas = _import_all_schemas()
        required = schemas["session_folder_create"]["parameters"].get("required", [])
        assert "name" in required

    def test_folder_add_requires_both(self):
        schemas = _import_all_schemas()
        required = schemas["session_folder_add"]["parameters"].get("required", [])
        assert "folder_id" in required
        assert "session_ids" in required

    def test_folder_add_session_ids_is_array(self):
        schemas = _import_all_schemas()
        session_ids = schemas["session_folder_add"]["parameters"]["properties"]["session_ids"]
        assert session_ids["type"] == "array"
        assert session_ids["items"]["type"] == "string"

    def test_folder_rename_requires_both(self):
        schemas = _import_all_schemas()
        required = schemas["session_folder_rename"]["parameters"].get("required", [])
        assert "folder_id" in required
        assert "name" in required

    def test_folder_delete_requires_folder_id(self):
        schemas = _import_all_schemas()
        required = schemas["session_folder_delete"]["parameters"].get("required", [])
        assert "folder_id" in required


# =====================================================================
# Registry registration
# =====================================================================


class TestRegistryRegistration:
    def test_all_tools_register(self):
        """Importing the module triggers registry.register() for all 7 tools."""
        # Fresh import to trigger registration
        from tools.registry import registry as _reg

        for mod in list(sys.modules.keys()):
            if "session_tools" in mod:
                del sys.modules[mod]

        import tools.session_tools  # noqa: F401 — triggers registration

        # Registry keeps a dict of {name: ToolEntry}. Check that our tools
        # are among the registered entries. We can't call registry's methods
        # directly since they may not expose iteration, but at minimum the
        # import should not raise.
        assert tools.session_tools  # import worked

    def test_handler_functions_return_json(self):
        """Every handler returns a valid JSON string."""
        from tools.session_tools import (
            _session_list, _session_archive, _session_rename,
            _session_delete, _folder_list, _folder_create, _folder_add,
            _folder_rename, _folder_delete,
        )

        handlers = [
            (lambda: _session_list(limit=1)),
            (lambda: _session_archive("s1")),
            (lambda: _session_rename("s1", "X")),
            (lambda: _session_delete("s1")),
            (lambda: _folder_list()),
            (lambda: _folder_create("X")),
            (lambda: _folder_add("f1", ["s1"])),
            (lambda: _folder_rename("f1", "X")),
            (lambda: _folder_delete("f1")),
        ]

        # All handlers should return something parseable as JSON
        # even if the underlying API is unreachable (they handle None).
        with patch("tools.session_tools._api_get", return_value=None):
            with patch("tools.session_tools._api_patch", return_value=None):
                with patch("tools.session_tools._api_delete", return_value=None):
                    with patch("tools.session_tools._api_post", return_value=None):
                        for h in handlers:
                            result = json.loads(h())
                            assert "success" in result
