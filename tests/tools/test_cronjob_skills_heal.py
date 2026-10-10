"""Regression: the cronjob tool layer must heal stringified skills lists.

The tool schema accepts ``skills: array``; model callers sometimes pass the python
repr of the list instead (``"['x']"``). The create/update paths funnel through
``_canonical_skills`` — the heal must land there too, or the corruption is stored.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.cronjob_job_args import _canonical_skills


class TestCanonicalSkillsHeals:
    def test_repr_string_in_skills(self):
        assert _canonical_skills(None, "['x']") == ["x"]

    def test_repr_string_inside_list(self):
        assert _canonical_skills(None, ['["x", "y"]']) == ["x", "y"]

    def test_legacy_skill_repr(self):
        assert _canonical_skills("['x']", None) == ["x"]

    def test_plain_string_kept(self):
        assert _canonical_skills("not-a-list", None) == ["not-a-list"]

    def test_dedupe_across_healed_entries(self):
        assert _canonical_skills(None, ["['x']", "x", "['x']"]) == ["x"]

    def test_none_and_empty(self):
        assert _canonical_skills(None, None) == []
        assert _canonical_skills(None, []) == []