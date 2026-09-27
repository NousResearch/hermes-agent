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
    def _branches(self):
        return SKILL_MANAGE_SCHEMA["parameters"]["properties"]["operations"]["items"]["anyOf"]

    def _op_props(self, action):
        """Merged properties of every branch advertising ``action``."""
        return {k: v for b in self._branches() if b["properties"]["action"]["enum"] == [action]
                for k, v in b["properties"].items()}

    def test_single_call_shape(self):
        """Maintainer-directed: operations[] IS the interface — each op
        names its skill; a single edit is a list of one. Flat fields are
        handler-only compat. Every per-action branch is a flat op object
        (memory-tool pattern), not a nested sub-object."""
        props = SKILL_MANAGE_SCHEMA["parameters"]["properties"]
        self.assertEqual(sorted(props), ["operations"])
        self.assertEqual(SKILL_MANAGE_SCHEMA["parameters"]["required"], ["operations"])
        for branch in self._branches():
            self.assertEqual(branch["required"][:2], ["name", "action"])
            self.assertTrue(all(p["type"] != "object" for p in branch["properties"].values()))
        self.assertTrue(self._op_props("delete"))


def _fits(op, branch):
    """Plain-Python reading of one per-action branch (required + enum + additionalProperties:
    false) so the test does not need a JSON-Schema validator."""
    props = branch["properties"]
    return (set(branch["required"]) <= set(op) and set(op) <= set(props)
            and op.get("action") in props["action"]["enum"])


class TestSkillManagePerActionShapes(unittest.TestCase):
    """#112677 — the four text slots (content / new_string / file_content / file_path) must
    not be siblings in one flat object: a create op carrying write_file's file_content
    validated against the advertised schema, so a grammar-constrained 27B model kept
    emitting it and every batch rolled back."""

    def _matches(self, op):
        return [b for b in SKILL_MANAGE_SCHEMA["parameters"]["properties"]["operations"]["items"]["anyOf"]
                if _fits(op, b)]

    def test_another_actions_text_slot_fits_no_branch(self):
        for op in ({"name": "s", "action": "create", "file_content": "# x"},
                   {"name": "s", "action": "create", "new_string": "# x"},
                   {"name": "s", "action": "write_file", "file_path": "references/a.md", "content": "x"},
                   {"name": "s", "action": "patch", "file_content": "x"}):
            self.assertEqual(self._matches(op), [], op)

    def test_every_documented_call_shape_fits_exactly_one_branch(self):
        for op in ({"name": "s", "action": "create", "content": "# x", "category": "devops"},
                   {"name": "s", "action": "patch", "old_string": "a", "new_string": "b",
                    "file_path": "references/a.md", "replace_all": True},
                   {"name": "s", "action": "patch", "content": "# whole file"},
                   {"name": "s", "action": "write_file", "file_path": "references/a.md", "file_content": "x"},
                   {"name": "s", "action": "remove_file", "file_path": "references/a.md"},
                   {"name": "s", "action": "delete"}):
            self.assertEqual(len(self._matches(op)), 1, op)

    def test_curator_consolidation_delete_keeps_absorbed_into(self):
        """The curator's consolidation delete carries ``absorbed_into=<umbrella>``; with
        additionalProperties:false a grammar-constrained backend refuses any key the delete
        branch does not advertise, and the delete guard then fail-closes every consolidation."""
        self.assertEqual(len(self._matches({"name": "s", "action": "delete", "absorbed_into": "umbrella"})), 1)


if __name__ == "__main__":
    unittest.main()
