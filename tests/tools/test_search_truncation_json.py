"""Regression tests for search_files truncation output staying pure JSON (#90322).

The truncated-results hint used to be appended as plain text after the
serialized JSON payload (``{...}\\n\\n[Hint: ...]``), so the tool result was
no longer parseable JSON — downstream tool-message handling on providers
strict about tool-content formatting could reject or mishandle it. The hint
now rides inside the payload as a structured ``_hint`` field, matching the
existing ``_omitted``/``_warning`` side-channel convention in the same
function.
"""

import json
from unittest.mock import MagicMock, patch

import pytest


class _FakeSearchResult:
    """Minimal stand-in for FileOperations.search return value."""

    def __init__(self, truncated=False):
        self.matches = []
        self._truncated = truncated

    def to_dict(self, densify=False):
        payload = {"matches": [{"file": "test.py", "line": 1, "text": "match"}]}
        if self._truncated:
            payload["truncated"] = True
            payload["total_count"] = 20
        return payload


def _make_fake_file_ops(truncated):
    fake = MagicMock()
    fake.search = lambda **kw: _FakeSearchResult(truncated=truncated)
    return fake


class TestSearchTruncationStaysJson:
    @patch("tools.file_tools._get_file_ops", return_value=_make_fake_file_ops(True))
    def test_truncated_result_is_parseable_json(self, _mock_ops):
        """The whole tool output must round-trip through json.loads — no
        text appended after the serialized payload."""
        from tools.file_tools import search_tool

        raw = search_tool("def main", offset=0, limit=20, task_id="t1")
        parsed = json.loads(raw)  # raises on main: trailing "[Hint: ...]" text
        assert parsed["truncated"] is True

    @patch("tools.file_tools._get_file_ops", return_value=_make_fake_file_ops(True))
    def test_truncation_hint_is_a_structured_field(self, _mock_ops):
        from tools.file_tools import search_tool

        parsed = json.loads(search_tool("def main", offset=0, limit=20, task_id="t1"))
        assert "_hint" in parsed
        assert "offset=20" in parsed["_hint"]

    @patch("tools.file_tools._get_file_ops", return_value=_make_fake_file_ops(False))
    def test_untruncated_result_has_no_hint(self, _mock_ops):
        from tools.file_tools import search_tool

        parsed = json.loads(search_tool("def main", task_id="t1"))
        assert "truncated" not in parsed
        assert "_hint" not in parsed
