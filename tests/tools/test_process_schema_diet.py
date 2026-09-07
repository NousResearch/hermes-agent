"""process schema diet contract (#95681).

Pins the shape: enum names the verbs, description carries only
non-obvious semantics, and the write-vs-submit trap teaching (Windows
PTY: a lone newline is not a line terminator) survives with emphasis.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tools.process_registry import PROCESS_SCHEMA


class TestProcessSchemaDiet(unittest.TestCase):
    def test_write_vs_submit_trap_survives(self):
        desc = PROCESS_SCHEMA["description"]
        self.assertIn("submit appends Enter", desc)
        self.assertIn("answer prompts", desc)
        self.assertIn("no newline", desc)

    def test_nonobvious_semantics_survive(self):
        desc = PROCESS_SCHEMA["description"]
        self.assertIn("partial output on timeout", desc)
        props = PROCESS_SCHEMA["parameters"]["properties"]
        self.assertIn("unique prefix", props["session_id"]["description"])
        self.assertIn("last 200", props["offset"]["description"])

    def test_enum_is_the_verb_source(self):
        props = PROCESS_SCHEMA["parameters"]["properties"]
        self.assertEqual(
            props["action"]["enum"],
            ["list", "poll", "log", "wait", "kill", "write", "submit", "close"],
        )
        # No redundant description on the enum param.
        self.assertNotIn("description", props["action"])


if __name__ == "__main__":
    unittest.main()
