"""todo schema diet contract (#95681).

Pins the dedup: item shape and merge semantics are taught ONLY by the
parameter schema (types/enum/required), never re-spelled in the
description — and the load-bearing behavioral teachings survive.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tools.todo_tool import TODO_SCHEMA


class TestTodoSchemaDiet(unittest.TestCase):
    def test_description_does_not_respell_param_structure(self):
        desc = TODO_SCHEMA["description"]
        # The old '{id: string, content: string, status: ...}' spell-out
        # and the 'Writing:' merge bullets duplicated the param schema.
        self.assertNotIn("{id:", desc)
        self.assertNotIn("Writing:", desc)
        self.assertNotIn("merge=", desc)
        self.assertNotIn("pending|in_progress", desc)

    def test_param_schema_is_the_single_structure_source(self):
        item = TODO_SCHEMA["parameters"]["properties"]["todos"]["items"]
        self.assertEqual(item["required"], ["id", "content", "status"])
        self.assertEqual(
            item["properties"]["status"]["enum"],
            ["pending", "in_progress", "completed", "cancelled"],
        )
        merge_desc = TODO_SCHEMA["parameters"]["properties"]["merge"]["description"]
        self.assertIn("replace the entire list", merge_desc)
        self.assertIn("update existing items by id", merge_desc)

    def test_behavioral_teachings_survive(self):
        """The parts that fight real model failure modes must not be
        dieted away: enumerate-all-N, one in_progress, verified-done,
        cancel-and-revise."""
        desc = TODO_SCHEMA["description"]
        self.assertIn("enumerate every instance", desc)
        self.assertIn("ONE item in_progress", desc)
        self.assertIn("verified done", desc)
        self.assertIn("cancel it and add a revised item", desc)
        self.assertIn("3+ steps", desc)


if __name__ == "__main__":
    unittest.main()
