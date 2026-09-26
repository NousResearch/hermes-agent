"""The registered search tool must paginate the combined root results."""

import json

import pytest

from tools import file_tools
from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations
from tools.registry import registry


@pytest.mark.parametrize("output_mode", ["content", "files_only"])
def test_multi_root_search_paginates_combined_results(tmp_path, monkeypatch, output_mode):
    roots = [tmp_path / "first", tmp_path / "second"]
    files = []
    for root in roots:
        root.mkdir()
        file = root / "example.txt"
        file.write_text("needle\nneedle\n" if root == roots[0] else "needle\n", encoding="utf-8")
        files.append(str(file))
    ops = ShellFileOperations(LocalEnvironment(str(tmp_path)))
    monkeypatch.setattr(file_tools, "_get_file_ops", lambda task_id: ops)

    expected = files if output_mode == "files_only" else [
        (files[0], 1), (files[0], 2), (files[1], 1),
    ]
    for offset, limit in ((0, 1), (1, 1), (1, 2), (2, 1), (3, 1)):
        response = registry.dispatch("search_files", {
            "pattern": "needle",
            "path": ", ".join(map(str, roots)),
            "target": "content",
            "output_mode": output_mode,
            "limit": limit,
            "offset": offset,
        })
        assert isinstance(response, str)
        # A truncated response appends a human-facing pagination hint after JSON.
        result, _ = json.JSONDecoder().raw_decode(response)

        assert "error" not in result
        if output_mode == "files_only":
            page = result.get("files", [])
        else:
            page = [(match["path"], match["line"]) for match in result.get("matches", [])]
        assert page == expected[offset:offset + limit]
        assert result.get("truncated", False) == (offset + limit < len(expected))
