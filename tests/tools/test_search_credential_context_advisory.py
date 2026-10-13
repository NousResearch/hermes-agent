"""``search_files`` advisory on credential-keyword patterns with context lines.

Scenario: ``search_files`` with ``context: 1`` and a pattern containing ``token`` on a project's
CLAUDE.md returns the line holding a quoted credential verbatim. The call is still answered —
the advisory tells the agent to use ``files_only``/``count`` first and never to grep docs for live
secret values.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.file_tools import search_tool

ADVISORY = "Credential-keyword pattern with context>0"


@pytest.fixture
def docs(tmp_path, monkeypatch):
    """A fake backend returning one real-shaped match: the advisory is tool-layer logic, so the
    test does not need ripgrep or a shell session."""
    import tools.file_tools as ft
    from tools.file_operations_common import SearchMatch, SearchResult

    readme = tmp_path / "README.md"
    readme.write_text("# Setup\nService auth token for the API: see .env\nnext line\n", encoding="utf-8")

    class _FakeOps:
        def search(self, **kw):
            if kw.get("output_mode") == "files_only":
                return SearchResult(files=[str(readme)], total_count=1)
            return SearchResult(matches=[SearchMatch(path=str(readme), line_number=2,
                                                     content="Service auth token for the API: see .env")],
                                total_count=1)

    monkeypatch.setattr(ft, "_get_file_ops", lambda task_id="default": _FakeOps())
    monkeypatch.setattr(ft, "_resolve_path_for_task", lambda path, task_id="default": Path(path))
    return tmp_path


def _result(**kwargs) -> dict:
    return json.loads(search_tool(task_id="search-advisory-test", **kwargs))


def test_context_plus_credential_pattern_gets_the_advisory(docs):
    out = _result(pattern="X-Auth-Token|token", path=str(docs), context=1)
    assert out["matches"], out
    assert ADVISORY in out["_warning"]


def test_context_zero_has_no_advisory(docs):
    out = _result(pattern="X-Auth-Token|token", path=str(docs), context=0)
    assert "_warning" not in out


def test_non_credential_pattern_with_context_has_no_advisory(docs):
    out = _result(pattern="Setup", path=str(docs), context=2)
    assert "_warning" not in out


def test_files_only_mode_has_no_advisory(docs):
    out = _result(pattern="token", path=str(docs), context=1, output_mode="files_only")
    assert "_warning" not in out


def test_advisory_appends_to_an_existing_warning(docs):
    for _ in range(3):
        out = _result(pattern="password", path=str(docs), context=1)
    assert "3 times consecutively" in out["_warning"] and ADVISORY in out["_warning"]
