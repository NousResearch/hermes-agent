"""``skill_view`` must serialize YAML-native scalars (dates) instead of failing (#82812).

A skill's YAML front matter like ``updated: 2026-03-05`` parses to a ``datetime.date``.
``skill_view`` embeds frontmatter values (description, metadata, compatibility) in its
JSON result; without a ``default=`` handler the call failed with
"Object of type date is not JSON serializable".
"""

from __future__ import annotations

import datetime
import json

from tools.skills_tool_plugin import _json


def test_date_and_datetime_are_serialized_as_iso_strings():
    payload = {"updated": datetime.date(2026, 3, 5), "at": datetime.datetime(2026, 3, 5, 4, 30)}
    decoded = json.loads(_json(payload))
    assert decoded == {"updated": "2026-03-05", "at": "2026-03-05T04:30:00"}


def test_plain_payloads_are_unchanged():
    assert json.loads(_json({"a": 1, "b": ["x"], "c": None})) == {"a": 1, "b": ["x"], "c": None}
