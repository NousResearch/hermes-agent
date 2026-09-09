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


class TestConsolidatedToolsets(unittest.TestCase):
    def test_old_names_gone_new_names_present(self):
        from toolsets import TOOLSETS

        ui = TOOLSETS["desktop_ui"]["tools"]
        self.assertIn("desktop_preview", ui)
        for dead in ("open_preview", "close_preview", "read_preview"):
            self.assertNotIn(dead, ui)
        proj = TOOLSETS["project"]["tools"]
        self.assertEqual(proj, ["desktop_project"])

    def test_registry_serves_only_new_names(self):
        """Post-#97979 the GUI surface defers by default, so assemble with
        the legacy everything-eager override (defer: []) — the contract
        pinned here is the RENAME (new names only, dead names gone), not
        the deferral policy."""
        from unittest.mock import patch as _patch

        from model_tools import get_tool_definitions
        from tools.tool_search import ToolSearchConfig

        legacy = ToolSearchConfig.from_raw({"enabled": "on", "defer": []})
        with _patch("tools.tool_search.load_config_readonly", return_value=legacy), \
             _patch("tools.tool_search.load_config", return_value=legacy):
            names = {
                t["function"]["name"]
                for t in get_tool_definitions(
                    quiet_mode=True, enabled_toolsets=["desktop_ui", "project"]
                )
            }
        self.assertIn("desktop_preview", names)
        self.assertIn("desktop_project", names)
        for dead in (
            "open_preview", "close_preview", "read_preview",
            "project_create", "project_switch", "project_list",
        ):
            self.assertNotIn(dead, names)


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

    def test_unknown_action_teaches(self):
        from tools import preview_tool

        r = json.loads(preview_tool._handle_preview({"action": "zap"}))
        self.assertIn("open, close, read", r.get("error", ""))


class TestProjectHandler(unittest.TestCase):
    def test_dispatch_shapes(self):
        import tools.project_tools as pt

        with patch.object(pt, "project_list", return_value='{"success": true}') as pl:
            pt._handle_project({"action": "list"})
            pl.assert_called_once()
        with patch.object(pt, "project_create", return_value='{"success": true}') as pc:
            pt._handle_project({"action": "create", "name": "X", "path": "C:/tmp"})
            pc.assert_called_once_with(name="X", path="C:/tmp", task_id=None)
        with patch.object(pt, "project_switch", return_value='{"success": true}') as ps:
            pt._handle_project({"action": "switch", "name": "aurora"})
            ps.assert_called_once_with(project="aurora", task_id=None)
        r = json.loads(pt._handle_project({"action": "bogus"}))
        self.assertFalse(r["success"])


class TestDietBudget(unittest.TestCase):
    def test_desktop_surface_under_budget(self):
        """The 15-tool surface serialized to ~15.5K chars (≈3,861 tok);
        consolidation + diet brought the (now 11-tool) surface to ~9.2K
        chars (≈2,293 tok). Guard: stay under 10.5K chars (≈2,600 tok) —
        a bloat regression guard, not a snapshot pin. Chars, not tokens:
        tiktoken is not a repo dependency, and the chars/tokens ratio for
        these schemas is stable (~4.0)."""
        from model_tools import get_tool_definitions

        targets = {
            "drive_preview", "tour", "annotate_preview", "setup_mcp", "tip",
            "desktop_preview", "desktop_project", "read_window_below", "apply_layout",
            "read_terminal", "focus_pane",
        }
        total = 0
        for t in get_tool_definitions(quiet_mode=True, enabled_toolsets=["desktop_ui", "project"]):
            f = t["function"]
            if f["name"] in targets:
                total += len(json.dumps(f, separators=(",", ":"), ensure_ascii=False))
        self.assertLess(total, 10_500, f"desktop tool surface regressed to {total} chars")


if __name__ == "__main__":
    unittest.main()
