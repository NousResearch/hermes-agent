"""Desktop tool consolidation + diet (#95681, maintainer-directed).

preview = open/close/read as one action tool (576 -> ~235); project =
create/switch/list as one (244 -> ~155). Old names are GONE from the
toolsets (desktop-only tools; no long-transcript compat needed). The
preview read action still routes through the agent-level GUI callback.
"""
import json
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))




class TestPreviewHandler(unittest.TestCase):
    def test_open_and_close_route_through_desktop_ui(self):
        from tools import preview_tool

        sent = []
        with patch("tools.desktop_ui.emit", side_effect=lambda ev, p: sent.append((ev, p)) or True):
            r = json.loads(preview_tool._handle_preview({"action": "open", "url": "www.cnn.com"}))
            self.assertTrue(r["success"])
            self.assertEqual(r["url"], "https://www.cnn.com")  # normalizer kept
            r = json.loads(preview_tool._handle_preview({"action": "close"}))
            self.assertTrue(r["success"])
        self.assertEqual([e for e, _ in sent], ["preview.open", "preview.close"])

    def test_read_outside_desktop_session_teaches(self):
        from tools import preview_tool

        r = json.loads(preview_tool._handle_preview({"action": "read"}))
        self.assertFalse(r.get("success", False))







if __name__ == "__main__":
    unittest.main()
