"""patch V4A provider gate (#95681).

V4A is the OpenAI apply_patch dialect; the dual-mode schema taxed every
non-OpenAI session ~149 tok/call for a format their models weren't
trained on. Base schema = replace-only (mode gone, path/old/new required);
the V4A layer renders only for OpenAI-family mains. Handler accepts both
shapes from any model (replay compat; mode defaults to replace).
"""
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import tools.file_tools as ft


def _family(prov, model):
    with patch("agent.auxiliary_client._read_main_provider", return_value=prov), \
         patch("agent.auxiliary_client._read_main_model", return_value=model):
        return ft._is_openai_family_main()


def _override(prov, model):
    with patch("agent.auxiliary_client._read_main_provider", return_value=prov), \
         patch("agent.auxiliary_client._read_main_model", return_value=model):
        return ft._patch_schema_overrides()


class TestPatchV4AGate(unittest.TestCase):
    def test_family_detector(self):
        for prov, model, want in [
            ("openai", "gpt-5.2", True),
            ("openai-codex", "codex-large", True),
            ("azure-openai", "deploy-x", True),
            ("openrouter", "openai/gpt-5.2", True),
            ("nous", "openai/o5-mini", True),
            ("openrouter", "anthropic/claude-sonnet-4", False),
            ("anthropic", "claude-fable-5", False),
            ("nous", "hermes-4-405b", False),
            ("", "", False),
        ]:
            self.assertEqual(_family(prov, model), want, (prov, model))

    def test_base_schema_is_replace_only(self):
        props = ft.PATCH_SCHEMA["parameters"]["properties"]
        self.assertNotIn("mode", props)
        self.assertNotIn("patch", props)
        self.assertEqual(ft.PATCH_SCHEMA["parameters"]["required"],
                         ["path", "old_string", "new_string"])
        self.assertNotIn("V4A", ft.PATCH_SCHEMA["description"])

    def test_openai_family_gets_v4a_layer(self):
        o = _override("openai", "gpt-5.2")
        self.assertIn("V4A", o["description"])
        self.assertIn("mode", o["parameters"]["properties"])
        self.assertIn("patch", o["parameters"]["properties"])
        self.assertEqual(o["parameters"]["required"], ["mode"])

    def test_non_openai_gets_no_override(self):
        self.assertEqual(_override("anthropic", "claude-fable-5"), {})

    def test_handler_accepts_both_shapes_regardless(self):
        """Wire compat: replace works without mode; V4A applies even from
        sessions whose schema never advertised it."""
        work = tempfile.mkdtemp(prefix="v4a_t_")
        f1 = os.path.join(work, "a.txt")
        open(f1, "w").write("alpha beta\n")
        r = json.loads(ft.patch_tool(path=f1, old_string="beta", new_string="B"))
        self.assertFalse(r.get("error"), r)
        v4a = (
            "*** Begin Patch\n"
            f"*** Update File: {f1}\n@@\n-alpha B\n+A B\n"
            "*** End Patch"
        )
        r = json.loads(ft.patch_tool(mode="patch", patch=v4a))
        self.assertFalse(r.get("error"), r)
        self.assertEqual(open(f1).read().strip(), "A B")


if __name__ == "__main__":
    unittest.main()
