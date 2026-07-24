"""Regression coverage for glob-shaped content-search patterns (#66129)."""

import json
from pathlib import Path

import pytest

from tools import file_tools
from tools.environments.local import LocalEnvironment
from tools.file_operations import SearchMatch, SearchResult, ShellFileOperations


def _drive_search(monkeypatch, pattern: str, backend_result: SearchResult | None = None):
    seen: list[str] = []

    class FakeFileOps:
        def search(self, **kwargs):
            seen.append(kwargs["pattern"])
            return backend_result or SearchResult(
                matches=[
                    SearchMatch(
                        path="sample.txt",
                        line_number=1,
                        content="detector scheduler",
                    )
                ],
                total_count=1,
            )

    with file_tools._read_tracker_lock:
        file_tools._read_tracker.clear()
    monkeypatch.setattr(file_tools, "_get_file_ops", lambda _task_id: FakeFileOps())
    monkeypatch.setattr(
        file_tools,
        "_filter_read_blocked_search_results",
        lambda _result, _task_id: 0,
    )
    monkeypatch.setattr(file_tools, "get_read_block_error", lambda _path: None)
    monkeypatch.setattr(
        file_tools,
        "_resolve_path_for_task",
        lambda path, _task_id: Path(path),
    )

    response = json.loads(
        file_tools.search_tool(pattern, target="content", task_id="glob-recovery-test")
    )
    return response, seen


def test_bare_glob_pattern_is_translated_once(monkeypatch):
    response, seen = _drive_search(monkeypatch, "*detector*scheduler*")

    assert seen == [r".*detector.*scheduler.*"]
    assert "error" not in response
    assert response["total_count"] == 1
    assert "interpreted as the glob-like pattern" in response["warning"]
    assert "target='files'" in response["warning"]


def test_noncapturing_wrapper_around_glob_is_recovered(monkeypatch):
    response, seen = _drive_search(monkeypatch, "(?:*detector*scheduler*)")

    assert seen == [r".*detector.*scheduler.*"]
    assert "error" not in response
    assert "warning" in response


def test_valid_regex_is_never_rewritten(monkeypatch):
    response, seen = _drive_search(monkeypatch, r"detector.*scheduler")

    assert seen == [r"detector.*scheduler"]
    assert "error" not in response
    assert "warning" not in response


def test_invalid_non_glob_regex_keeps_error_and_adds_guidance(monkeypatch):
    backend_error = SearchResult(
        error="Search failed: rg: regex parse error: unclosed character class",
        total_count=0,
    )
    response, seen = _drive_search(monkeypatch, "[", backend_error)

    assert seen == ["["]
    assert "Content search expects a regular expression" in response["error"]
    assert "target='files'" in response["error"]
    assert "warning" not in response


def test_translation_rejects_regex_intent_and_plain_text():
    assert file_tools._globish_content_pattern_to_regex(r"\d+") is None
    assert file_tools._globish_content_pattern_to_regex("foo[bar]") is None
    assert file_tools._globish_content_pattern_to_regex("plain text") is None


@pytest.fixture
def real_local_search_backend(monkeypatch, tmp_path):
    """Run search_tool through ShellFileOperations and the installed ripgrep."""
    environment = LocalEnvironment(cwd=str(tmp_path), timeout=15)
    file_ops = ShellFileOperations(environment, cwd=str(tmp_path))

    with file_tools._read_tracker_lock:
        file_tools._read_tracker.clear()
    monkeypatch.setattr(file_tools, "_get_file_ops", lambda _task_id: file_ops)
    return tmp_path


def test_glob_recovery_reaches_real_ripgrep_through_search_tool(
    real_local_search_backend,
):
    (real_local_search_backend / "sample.txt").write_text(
        "detector scheduler\n",
        encoding="utf-8",
    )

    response = json.loads(
        file_tools.search_tool(
            "*detector*scheduler*",
            target="content",
            path=".",
            task_id="glob-recovery-real-rg",
        )
    )

    assert "error" not in response
    assert response["total_count"] == 1
    assert response["matches"][0]["content"] == "detector scheduler"
    assert "interpreted as the glob-like pattern" in response["warning"]


def test_malformed_non_glob_regex_reaches_real_ripgrep_error(
    real_local_search_backend,
):
    (real_local_search_backend / "sample.txt").write_text(
        "detector scheduler\n",
        encoding="utf-8",
    )

    response = json.loads(
        file_tools.search_tool(
            "[",
            target="content",
            path=".",
            task_id="malformed-regex-real-rg",
        )
    )

    assert "error" in response
    assert "regex parse error" in response["error"]
    assert "Content search expects a regular expression" in response["error"]
