"""search_files on an EPERM tree returns a structured skip, not a traceback."""

from __future__ import annotations

import json

from tools.file_tools import _handle_search_files, search_tool


def test_search_tool_permission_error_is_structured_skip(monkeypatch, tmp_path):
    locked = tmp_path / "locked"
    locked.mkdir()

    class _Boom:
        def search(self, **_kw):
            raise PermissionError("Operation not permitted")

    monkeypatch.setattr("tools.file_tools._get_file_ops", lambda tid="default": _Boom())
    raw = search_tool(pattern="secret", path=str(locked), task_id="eperm-test")
    data = json.loads(raw)
    assert data["skipped"] is True
    assert data["success"] is False
    assert "Permission denied" in data["error"]
    assert data["path"] == str(locked)
    assert "Traceback" not in raw
    assert "AttributeError" not in raw


def test_handle_search_files_permission_error_is_structured_skip(monkeypatch, tmp_path):
    class _Boom:
        def search(self, **_kw):
            raise PermissionError("Operation not permitted")

    monkeypatch.setattr("tools.file_tools._get_file_ops", lambda tid="default": _Boom())
    raw = _handle_search_files({"pattern": "x", "path": str(tmp_path)})
    data = json.loads(raw)
    assert data["skipped"] is True
    assert data["success"] is False
