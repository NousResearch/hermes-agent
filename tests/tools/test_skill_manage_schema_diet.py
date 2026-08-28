"""skill_manage schema diet contract (#95681, second pass).

Pins the deduped surface: patch args defer to the `patch` tool's matching
semantics instead of re-teaching them; file_path states its skill-dir-
relative shape; no authoring curriculum or confirm-with-user coaching in
the description (maintainer-directed cuts).
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tools.skill_manager_tool import SKILL_MANAGE_SCHEMA


class TestSkillManageSchemaDiet(unittest.TestCase):
    def test_patch_args_defer_to_patch_tool(self):
        props = SKILL_MANAGE_SCHEMA["parameters"]["properties"]
        self.assertIn("patch tool", props["old_string"]["description"])
        # The uniqueness/context curriculum lives in the patch tool's schema,
        # not here.
        self.assertNotIn("unique", props["old_string"]["description"])
        self.assertNotIn("surrounding context", props["old_string"]["description"])

    def test_file_path_states_relative_shape(self):
        desc = SKILL_MANAGE_SCHEMA["parameters"]["properties"]["file_path"]["description"]
        self.assertIn("RELATIVE", desc)
        self.assertIn("references/api.md", desc)
        self.assertIn("never absolute", desc)
        # The subdirectory whitelist is a real contract — must stay.
        for sub in ("references/", "templates/", "scripts/", "assets/"):
            self.assertIn(sub, desc)

    def test_description_cuts_hold(self):
        desc = SKILL_MANAGE_SCHEMA["description"]
        # Maintainer-directed: no confirm-with-user coaching.
        self.assertNotIn("Confirm with the user", desc)
        # Authoring curriculum compressed to the 57-char trigger rule +
        # skill_view pointer; the numbered-steps/pitfalls list is gone.
        self.assertIn("57 chars", desc)
        self.assertIn("skill_view()", desc)
        self.assertNotIn("numbered steps", desc)
        # Stale action vocabulary must not return.
        self.assertNotIn("edit", SKILL_MANAGE_SCHEMA["parameters"]["properties"]["name"]["description"])

    def test_content_keeps_pre_irreversibility_warning(self):
        """The REPLACES-whole-file warning is pre-irreversibility guidance:
        an error can't teach after a successful full rewrite, so it must
        stay schema-side (maintainer call)."""
        desc = SKILL_MANAGE_SCHEMA["parameters"]["properties"]["content"]["description"]
        self.assertIn("REPLACES", desc)
        self.assertIn("skill_view()", desc)


if __name__ == "__main__":
    unittest.main()
