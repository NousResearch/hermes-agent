"""A staged batch must carry the call-level name into its own operations.

The tool accepts a call-level ``name`` as the per-op fallback (``skill_manage(action="batch",
name="x", operations=[{action:"patch", ...}])``). Staging used to store the operations verbatim,
so the record held ops with no name; replay of the approved write then failed inside
``_validate_batch_ops`` with "operations[0] needs a 'name'" — a write the user approved and never
saw land. These tests pin the bake-in at the point of staging, and that a fully-named batch is
left alone.
"""
import importlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

SK = (
    "---\nname: {n}\ndescription: Use when probing staged batch names. Behavior.\n---\n"
    "# Probe\nStep 1.\n"
)


class TestStagedBatchName(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="skmstage_t_")
        self._old_home = os.environ.get("HERMES_HOME")
        os.environ["HERMES_HOME"] = self.home
        os.makedirs(os.path.join(self.home, "skills", "probe"), exist_ok=True)
        with open(os.path.join(self.home, "skills", "probe", "SKILL.md"), "w") as fh:
            fh.write(SK.format(n="probe"))
        import tools.skill_manager_batch as smb
        import tools.skill_manager_tool as smt
        import tools.write_approval as wa
        self.smt = importlib.reload(smt)
        self.smb = importlib.reload(smb)
        self.wa = importlib.reload(wa)
        # Staging only happens when the gate says "neither allowed nor blocked"; the test home has
        # no write_approval config, so pin that decision rather than depend on the default.
        self._gate = patch.object(self.wa, "evaluate_gate",
                                  lambda subsystem: SimpleNamespace(allow=False, blocked=False,
                                                                    message="staged for approval"))
        self._gate.start()

    def tearDown(self):
        self._gate.stop()
        shutil.rmtree(self.home, ignore_errors=True)
        if self._old_home is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = self._old_home

    def _staged(self):
        records = self.wa.list_pending(self.wa.SKILLS)
        self.assertEqual(len(records), 1, records)
        return records[0]

    def test_call_level_name_is_written_into_staged_operations(self):
        result = json.loads(self.smt.skill_manage(
            action="batch", name="probe",
            operations=[{"action": "patch", "file_path": "SKILL.md",
                         "old_string": "Step 1.", "new_string": "Step 2."}]))
        self.assertTrue(result.get("staged"), result)
        record = self._staged()
        self.assertEqual(record["payload"]["operations"][0]["name"], "probe")

    def test_staged_record_replays_and_lands(self):
        self.smt.skill_manage(
            action="batch", name="probe",
            operations=[{"action": "patch", "file_path": "SKILL.md",
                         "old_string": "Step 1.", "new_string": "Step 2."}])
        record = self._staged()
        replayed = json.loads(self.smt.apply_skill_pending(record["payload"]))
        self.assertTrue(replayed.get("success"), replayed)
        with open(os.path.join(self.home, "skills", "probe", "SKILL.md")) as fh:
            self.assertIn("Step 2.", fh.read())

    def test_per_op_names_are_not_overwritten(self):
        self.smt.skill_manage(
            action="batch", name="probe",
            operations=[
                {"action": "patch", "name": "probe", "file_path": "SKILL.md",
                 "old_string": "Step 1.", "new_string": "Step 2."},
                {"action": "write_file", "name": "probe", "file_path": "references/a.md",
                 "file_content": "a"},
            ])
        record = self._staged()
        self.assertEqual([op["name"] for op in record["payload"]["operations"]], ["probe", "probe"])
