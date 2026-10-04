"""Blank search_files roots must use the documented current-directory default.

SEARCH_FILES_SCHEMA documents path='.'. A present empty or whitespace-only
value is not the same as a missing key for dict.get, so the search backend
used to treat it as a real filesystem path and return Path not found. Direct
search_tool callers (registry, execute_code stubs, tests) share that entry.
"""

from __future__ import annotations

import json

import pytest

from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations
from tools.file_tools import _handle_search_files, search_tool


NEEDLE = "TOKEN_112424_DOCUMENTED_PATH_DEFAULT"


@pytest.fixture
def proj(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    root = tmp_path / "proj"
    root.mkdir()
    (root / "hit.py").write_text(f"{NEEDLE} = 1\n", encoding="utf-8")
    spaced = root / "dir with spaces"
    spaced.mkdir()
    (spaced / "spaced.txt").write_text(f"{NEEDLE} in spaced dir\n", encoding="utf-8")
    return root


@pytest.fixture
def bind_search_root(proj, monkeypatch):
    """Point search_tool's file-ops cwd at the fixture so '.' is the project."""
    ops = ShellFileOperations(LocalEnvironment(str(proj)))
    monkeypatch.setattr("tools.file_tools._get_file_ops", lambda task_id: ops)
    return ops


def _payload(raw: str) -> dict:
    data = json.loads(raw)
    assert isinstance(data, dict)
    return data


def _error_text(data: dict) -> str:
    err = data.get("error") or ""
    return err if isinstance(err, str) else str(err)


@pytest.mark.parametrize("blank_path", ["", " \t ", "\n"])
@pytest.mark.parametrize("target", ["content", "files"])
def test_blank_path_searches_the_same_root_as_documented_default(
    bind_search_root, blank_path, target
):
    pattern = NEEDLE if target == "content" else "hit.py"
    documented = _payload(
        search_tool(pattern, target=target, path=".", task_id=f"t-dot-{target}")
    )
    blank = _payload(
        search_tool(
            pattern, target=target, path=blank_path, task_id=f"t-blank-{target}-{id(blank_path)}"
        )
    )
    omitted = _payload(search_tool(pattern, target=target, task_id=f"t-omit-{target}"))

    assert "Path not found" not in _error_text(documented)
    assert "Path not found" not in _error_text(blank)
    assert "Path not found" not in _error_text(omitted)
    assert documented.get("total_count", 0) >= 1
    assert blank.get("total_count") == documented.get("total_count")
    assert omitted.get("total_count") == documented.get("total_count")


@pytest.mark.parametrize("blank_path", ["", "   "])
def test_handler_blank_path_matches_working_directory_search(bind_search_root, blank_path):
    suffix = repr(blank_path)
    documented = _payload(
        _handle_search_files(
            {"pattern": NEEDLE, "target": "content", "path": "."},
            task_id=f"h-dot-{suffix}",
        )
    )
    blank = _payload(
        _handle_search_files(
            {"pattern": NEEDLE, "target": "content", "path": blank_path},
            task_id=f"h-blank-{suffix}",
        )
    )
    omitted = _payload(
        _handle_search_files(
            {"pattern": NEEDLE, "target": "content"},
            task_id=f"h-omit-{suffix}",
        )
    )

    assert "Path not found" not in _error_text(blank)
    documented_count = documented.get("total_count") or 0
    blank_count = blank.get("total_count") or 0
    omitted_count = omitted.get("total_count") or 0
    assert documented_count >= 1
    assert blank_count == documented_count
    assert omitted_count == documented_count


def test_missing_nonblank_root_still_reports_path_not_found(bind_search_root, proj):
    missing = proj / "definitely-not-a-search-root-112424"
    data = _payload(
        search_tool(NEEDLE, path=str(missing), task_id="t-missing-nonblank")
    )
    assert "Path not found" in _error_text(data)
    assert data.get("total_count", 0) == 0


def test_nonblank_path_with_internal_spaces_is_not_replaced(bind_search_root, proj):
    data = _payload(
        search_tool(
            NEEDLE,
            path=str(proj / "dir with spaces"),
            task_id="t-spaced-dir",
        )
    )
    assert "Path not found" not in _error_text(data)
    assert data.get("total_count", 0) >= 1
