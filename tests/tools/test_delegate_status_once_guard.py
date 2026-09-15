#!/usr/bin/env python3
"""Guard tests for the write-the-status-ONCE anti-babble rule.

2026-09-11 (music-master session 20260911_014738_20329d): a MiniMax-M3 parent,
told to "stop with a one-line status" after a background dispatch, serialized
five simulated re-wake/status cycles into ONE assistant message ("Pickup
continues...", "Status unchanged..."). The UI looked stuck in a loop, the user
typed "stop", and a healthy child was killed mid-analysis. The rule text these
tests pin is what forbids that pattern; if it is edited out, these fail.

Run with:  python -m pytest tests/tools/test_delegate_status_once_guard.py -v
"""

import unittest

from tools.delegate_tool import _build_top_level_description
from tools.delegate_tool_dispatch import _BACKGROUND_NOTES, _STATUS_ONCE_RULE


class TestStatusOnceGuard(unittest.TestCase):
    def test_rule_present_in_single_dispatch_note(self):
        self.assertIn(_STATUS_ONCE_RULE, _BACKGROUND_NOTES["one"])

    def test_rule_present_in_batch_dispatch_note(self):
        self.assertIn(_STATUS_ONCE_RULE, _BACKGROUND_NOTES["many"])

    def test_rule_forbids_serialized_status_updates(self):
        # The phrasing that makes the prohibition unambiguous must survive edits.
        self.assertIn("ONCE", _STATUS_ONCE_RULE)
        self.assertIn("simulated re-wakes", _STATUS_ONCE_RULE)

    def test_top_level_description_carries_the_guard(self):
        desc = _build_top_level_description()
        self.assertIn("ONCE", desc)
        self.assertIn("re-wakes", desc)


if __name__ == "__main__":
    unittest.main()
