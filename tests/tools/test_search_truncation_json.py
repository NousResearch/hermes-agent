"""search_files output is one JSON document even when truncated (#90322).

The pagination hint used to be appended as text after the serialized payload,
so json.loads on the tool result failed (execute_code RPC, strict tool-message
providers). It now rides inside the payload as ``_hint``.
"""

import json
from unittest.mock import MagicMock, patch


class _TruncatedSearch:
    matches = []

    def to_dict(self, densify=False):
        return {"matches": [{"file": "t.py", "line": 1, "text": "m"}], "truncated": True, "total_count": 20}


def test_truncated_search_result_round_trips_json():
    fake = MagicMock()
    fake.search = lambda **kw: _TruncatedSearch()
    with patch("tools.file_tools._get_file_ops", return_value=fake):
        from tools.file_tools import search_tool

        parsed = json.loads(search_tool("def main", offset=0, limit=20, task_id="t1"))
    assert parsed["truncated"] is True
    assert "offset=20" in parsed["_hint"]
