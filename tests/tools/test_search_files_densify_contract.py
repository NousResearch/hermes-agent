"""Regression for #155854: the execute_code kernel ``search_files`` helper must
always return the documented ``matches`` key.

The agent-facing path densifies >= 5 matches into ``matches_text`` (by-design
token optimisation for model context) — that contract is untouched here. But the
kernel stub's docstring promises ``matches``, and programmatic callers following
it got a silent empty list at the 5-match boundary. The stub now sends
``densify: False`` and the handler honours it; the model schema never exposes the
flag, so the model path is unchanged.
"""

import json
from unittest.mock import MagicMock, patch

from tools.file_operations_common import SearchMatch, SearchResult


def _result_with(n_matches: int) -> SearchResult:
    return SearchResult(
        matches=[SearchMatch(path=f"dir/f{i}.py", line_number=i + 1, content=f"hit {i}")
                 for i in range(n_matches)],
        total_count=n_matches,
    )


def _handle(args: dict) -> dict:
    """Dispatch through the real handler with the file-ops seam mocked."""
    from tools.file_tools import _handle_search_files
    mock_ops = MagicMock()
    mock_ops.search.return_value = _result_with(6)
    with patch("tools.file_tools._get_file_ops", return_value=mock_ops):
        raw = _handle_search_files(args, task_id="t-test")
    return json.loads(raw)


class TestKernelSearchFilesContract:

    def test_kernel_flag_returns_matches_array_at_threshold(self):
        """densify=False must keep the documented shape past the 5-match
        densify threshold: `matches` present and correct, no densified block."""
        d = _handle({"pattern": "hit", "path": "dir", "densify": False})
        assert "matches" in d, "kernel callers get no matches key at >= 5 results"
        assert "matches_text" not in d
        assert len(d["matches"]) == 6
        assert d["total_count"] == 6
        assert d["matches"][0] == {"path": "dir/f0.py", "line": 1, "content": "hit 0"}

    def test_model_path_still_densifies(self):
        """Without the internal flag the agent-facing contract is unchanged:
        >= 5 matches arrive as matches_text + matches_format, no matches key."""
        d = _handle({"pattern": "hit", "path": "dir"})
        assert "matches" not in d
        assert "matches_text" in d
        assert "matches_format" in d
        assert d["total_count"] == 6

    def test_kernel_flag_below_threshold_keeps_verbose_array(self):
        """densify=False with few matches behaves exactly like the small-result
        path always has: the verbose matches array."""
        from tools.file_tools import _handle_search_files
        mock_ops = MagicMock()
        mock_ops.search.return_value = _result_with(2)
        with patch("tools.file_tools._get_file_ops", return_value=mock_ops):
            raw = _handle_search_files({"pattern": "hit", "path": "dir",
                                        "densify": False}, task_id="t-test")
        d = json.loads(raw)
        assert len(d["matches"]) == 2

    def test_generated_kernel_stub_sends_densify_false(self):
        """The generated hermes_tools module must request the undensified shape:
        execute its real generated source with a capture fake for _call."""
        from tools.code_execution_tool import generate_hermes_tools_module
        captured: dict = {}

        def fake_call(name, payload):
            captured["name"] = name
            captured["payload"] = payload
            return {"matches": []}

        namespace = {"_call": fake_call}
        src = generate_hermes_tools_module(["search_files"])
        exec(compile(src, "hermes_tools.py", "exec"), namespace)
        # The generated header defines its own UDS _call; the stub resolves
        # _call from module globals at call time, so swap in the capture fake
        # AFTER exec to avoid touching any socket.
        namespace["_call"] = fake_call
        search_files = namespace["search_files"]
        search_files(pattern="hit", path="dir", limit=10)  # type: ignore[call-arg]  # generated stub
        assert captured["name"] == "search_files"
        assert captured["payload"]["densify"] is False
        assert captured["payload"]["pattern"] == "hit"
        assert captured["payload"]["limit"] == 10

    def test_kernel_flag_zero_results_preserves_empty_matches_array(self):
        """A zero-hit kernel search returns an empty matches list, not a missing key."""
        from tools.file_tools import _handle_search_files
        mock_ops = MagicMock()
        mock_ops.search.return_value = SearchResult(total_count=0)
        with patch("tools.file_tools._get_file_ops", return_value=mock_ops):
            raw = _handle_search_files(
                {"pattern": "__kernel_zero_hit__", "path": "dir", "densify": False},
                task_id="kernel-zero-result-contract",
            )
        result = json.loads(raw)
        assert result["total_count"] == 0
        assert result["matches"] == []
