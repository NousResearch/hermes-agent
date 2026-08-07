from __future__ import annotations

import json
import uuid

import tools.file_tools as file_tools
from tools.file_operations import SearchMatch, SearchResult


class _FakeFileOps:
    def search(self, **_kwargs):
        return SearchResult(
            matches=[
                SearchMatch(path="large.txt", line_number=i, content="needle " + ("x" * 4_000))
                for i in range(1, 26)
            ],
            total_count=25,
        )


def test_search_tool_caps_large_result_without_breaking_json(monkeypatch):
    monkeypatch.setattr(file_tools, "_get_file_ops", lambda _task_id: _FakeFileOps())

    output = file_tools.search_tool(
        pattern="needle",
        target="content",
        path=".",
        limit=25,
        task_id=f"search-cap-{uuid.uuid4()}",
    )

    assert len(output) <= 64_000
    body, _, hint = output.partition("\n\n[Hint:")
    parsed = json.loads(body)
    assert parsed["truncated"] is True
    assert parsed["limit_reason"] == "output_size"
    assert hint
