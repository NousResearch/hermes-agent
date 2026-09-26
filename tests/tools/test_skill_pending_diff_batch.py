"""Batch pending diff for /skills diff <id> (#123315).

Writes staged through ``skill_manage`` ``operations[]`` are ``batch`` records;
without a batch case ``skill_pending_diff`` falls through to ``(batch on '')``
and the review-before-approve affordance never shows a diff.
"""
import importlib
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

SK = (
    "---\nname: {n}\ndescription: Probe skill for pending-diff tests.\n---\n"
    "# Probe\nStep 1.\n"
)


class TestSkillPendingDiffBatch(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="skdiff_t_")
        os.environ["HERMES_HOME"] = self.home
        os.makedirs(os.path.join(self.home, "skills", "probe"), exist_ok=True)
        with open(os.path.join(self.home, "skills", "probe", "SKILL.md"), "w",
                  encoding="utf-8") as f:
            f.write(SK.format(n="probe"))
        # Re-import against the temp home (skill lookup caches SKILLS_DIR).
        import tools.skill_manager_tool as smt
        importlib.reload(smt)
        import tools.write_approval as wa
        importlib.reload(wa)
        self.wa = wa

    def tearDown(self):
        shutil.rmtree(self.home, ignore_errors=True)

    def _batch_record(self):
        return {
            "id": "abc123",
            "summary": "batch(2 ops: create, patch) on newprobe, probe",
            "payload": {
                "action": "batch",
                "operations": [
                    {"action": "create", "name": "newprobe",
                     "content": SK.format(n="newprobe")},
                    {"action": "patch", "name": "probe",
                     "old_string": "Step 1.", "new_string": "Step ONE."},
                ],
            },
        }

    def test_batch_diff_renders_per_op_content(self):
        out = self.wa.skill_pending_diff(self._batch_record())
        self.assertNotIn("(batch on", out)
        # create op shows the full new content
        self.assertIn("Step 1.", out)
        # patch op shows a unified diff against the on-disk skill
        self.assertIn("-Step 1.", out)
        self.assertIn("+Step ONE.", out)

    def test_single_op_paths_unchanged(self):
        create = self.wa.skill_pending_diff(
            {"payload": {"action": "create", "name": "newprobe",
                         "content": SK.format(n="newprobe")}})
        self.assertEqual(create, SK.format(n="newprobe"))
        patch = self.wa.skill_pending_diff(
            {"payload": {"action": "patch", "name": "probe",
                         "old_string": "Step 1.", "new_string": "Step ONE."}})
        self.assertIn("-Step 1.", patch)
        self.assertIn("+Step ONE.", patch)

    def test_diff_subcommand_end_to_end(self):
        from hermes_cli.write_approval_commands import handle_pending_subcommand
        rec = self.wa.stage_write(
            self.wa.SKILLS, self._batch_record()["payload"],
            summary="batch(2 ops) on newprobe, probe", origin="foreground")
        out = handle_pending_subcommand(self.wa.SKILLS, ["diff", rec["id"]])
        self.assertIsNotNone(out)
        assert out is not None
        self.assertIn(rec["id"], out)
        self.assertNotIn("(batch on", out)
        self.assertIn("+Step ONE.", out)


if __name__ == "__main__":
    unittest.main()
