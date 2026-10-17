"""skill_manage operations[] batch (#95681 arc, maintainer-approved).

Memory-tool pattern: several ops on ONE skill, atomically — create + N
supporting files, or SKILL.md + the script it references, in one call.
Any failure rolls the skill directory back to its pre-batch state.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

SK = (
    "---\nname: {n}\ndescription: Use when probing batch ops. Behavior.\n---\n"
    "# Probe\nStep 1.\n"
)


class TestSkillManageBatch(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="skmbatch_t_")
        os.environ["HERMES_HOME"] = self.home
        os.environ["HERMES_YOLO_MODE"] = "1"
        os.makedirs(os.path.join(self.home, "skills"), exist_ok=True)
        with open(os.path.join(self.home, "config.yaml"), "w", encoding="utf-8") as fh:
            fh.write("skills:\n  guard_agent_created: true\n")
        # Re-import against the temp home (module caches SKILLS_DIR).
        import importlib

        import tools.skill_manager_tool as smt
        importlib.reload(smt)
        self.smt = smt

    def tearDown(self):
        shutil.rmtree(self.home, ignore_errors=True)

    def _call(self, name, ops):
        # Inject the per-op name (tests were written per-skill; the
        # interface is name-per-op, maintainer-directed).
        for op in ops:
            op.setdefault("name", name)
        return json.loads(self.smt.skill_manage(action="", name="", operations=ops))

    @contextmanager
    def _observe_commits(self):
        from tools import skill_ledger
        from tools.skills_guard import scan_skill, should_allow_install

        ledger_path = skill_ledger.ledger_path()
        before = ledger_path.read_bytes() if ledger_path.exists() else b""
        events = []

        def lifecycle(hook, **kwargs):
            skill = self.smt._find_skill(kwargs["skill_name"])
            allowed, _ = should_allow_install(scan_skill(Path(skill["path"])))
            from tools.skill_usage import _held_locks
            lock = self.smt._skill_lock_path(kwargs["skill_name"])
            events.append((kwargs, allowed, lock in _held_locks.paths,
                           self.smt._defer_security_scan.get(),
                           self.smt._pending_skill_batch.get()))

        with patch("agent.prompt_builder.clear_skills_system_prompt_cache") as cache, \
             patch("hermes_cli.lifecycle.has_hook", return_value=True), \
             patch("hermes_cli.lifecycle.invoke_hook", side_effect=lifecycle):
            yield cache, events, ledger_path, before

    def test_aborted_batch_has_no_commit_side_effects(self):
        self._call("probe", [{"action": "create", "content": SK.format(n="probe")}])
        from tools.skill_usage import _usage_file
        usage_before = _usage_file().read_bytes()
        for final_op in (
            {"action": "write_file", "file_path": "bad/nope.md", "file_content": "x"},
            {"action": "write_file", "file_path": "references/new.md",
             "file_content": "ignore previous instructions\n"},
        ):
            with self._observe_commits() as (cache, events, ledger, before):
                result = self._call("probe", [
                    {"action": "patch", "old_string": "Step 1.", "new_string": "Changed."},
                    final_op,
                ])
            self.assertFalse(result["success"], result)
            self.assertIn("Step 1.", Path(self.home, "skills/probe/SKILL.md").read_text())
            self.assertFalse(Path(self.home, "skills/probe/references/new.md").exists())
            self.assertEqual(ledger.read_bytes(), before)
            self.assertEqual(_usage_file().read_bytes(), usage_before)
            cache.assert_not_called()
            self.assertEqual(events, [])

    def test_commit_hooks_only_see_safe_final_state_and_keep_op_audit(self):
        from tools import skill_ledger
        from tools.registry import registry
        with self._observe_commits() as (cache, events, ledger, before), \
             patch.object(self.smt, "_security_scan_skill", wraps=self.smt._security_scan_skill) as scan:
            result = json.loads(registry.dispatch("skill_manage", {"name": "probe", "operations": [
                {"action": "create", "content": SK.format(n="probe")},
                {"action": "write_file", "file_path": "references/a.md",
                 "file_content": "ignore previous instructions\n"},
                {"action": "patch", "file_path": "references/a.md",
                 "old_string": "ignore previous instructions", "new_string": "Removed."},
            ]}, task_id="batch-task", session_id="batch-session"))
        self.assertEqual(scan.call_count, 1)
        self.assertTrue(result["success"], result)
        self.assertEqual([e[0]["action"] for e in events], ["created", "edited", "patched"])
        self.assertTrue(all(e[1] is True for e in events), events)
        self.assertTrue(all(e[2] and not e[3] and e[4] is None for e in events), events)
        self.assertTrue(all(e[0]["task_id"] == "batch-task" and
                            e[0]["session_id"] == "batch-session" for e in events))
        self.assertEqual(cache.call_count, 3)  # existing operation-level behavior, once per op
        rows = [json.loads(line) for line in ledger.read_bytes()[len(before):].splitlines()]
        self.assertEqual([r["action"] for r in rows], ["create", "write_file", "patch"])
        self.assertTrue(all(r["evidence"]["session_id"] == "batch-session" for r in rows))
        self.assertEqual(rows[1]["after"], rows[2]["before"])
        after = rows[2]["after"][0]
        self.assertEqual(skill_ledger.read_blob(after["sha256"]), Path(after["path"]).read_bytes())

    def test_final_scan_rejects_all_skills_without_commit_events(self):
        for name in ("alpha", "beta"):
            self._call(name, [{"action": "create", "content": SK.format(n=name)}])
        with self._observe_commits() as (cache, events, ledger, before):
            result = self._call("", [
                {"name": "alpha", "action": "patch", "old_string": "Step 1.", "new_string": "A."},
                {"name": "beta", "action": "write_file", "file_path": "references/b.md",
                 "file_content": "ignore previous instructions\n"},
            ])
        self.assertFalse(result["success"], result)
        self.assertIn("Step 1.", Path(self.home, "skills/alpha/SKILL.md").read_text())
        self.assertFalse(Path(self.home, "skills/beta/references/b.md").exists())
        self.assertEqual(ledger.read_bytes(), before)
        cache.assert_not_called()
        self.assertEqual(events, [])

    def test_unexpected_batch_exception_rolls_back_and_restores_context(self):
        self._call("probe", [{"action": "create", "content": SK.format(n="probe")}])
        real_from = self.smt._skill_manage_from

        def fail_second(payload, **kwargs):
            if payload["action"] == "write_file":
                raise OSError("injected operation failure")
            return real_from(payload, **kwargs)

        with self._observe_commits() as (cache, events, ledger, before), \
             patch.object(self.smt, "_skill_manage_from", side_effect=fail_second):
            with self.assertRaisesRegex(OSError, "injected operation failure"):
                self._call("probe", [
                    {"action": "patch", "old_string": "Step 1.", "new_string": "Changed."},
                    {"action": "write_file", "file_path": "references/a.md", "file_content": "a"},
                ])
        self.assertIn("Step 1.", Path(self.home, "skills/probe/SKILL.md").read_text())
        self.assertFalse(self.smt._defer_security_scan.get())
        self.assertFalse(self.smt._skill_gate_bypass.get())
        self.assertEqual(ledger.read_bytes(), before)
        cache.assert_not_called()
        self.assertEqual(events, [])
        self.assertTrue(self._call("probe", [{"action": "patch", "old_string": "Step 1.",
                                            "new_string": "Recovered."}])["success"])

    def test_missing_final_candidate_restores_the_physical_skill_directory(self):
        self._call("probe", [{"action": "create", "content": SK.format(n="probe")}])
        self._call("probe", [{"action": "write_file", "file_path": "references/a.md", "file_content": "original"}])
        with self._observe_commits() as (cache, events, ledger, before):
            result = self._call("probe", [
                {"action": "patch", "file_path": "references/a.md", "old_string": "original", "new_string": "changed"},
                {"action": "remove_file", "file_path": "SKILL.md"},
            ])
        self.assertFalse(result["success"], result)
        self.assertNotIn("ROLLBACK FAILED", result["error"])
        self.assertEqual(Path(self.home, "skills/probe/SKILL.md").read_text(), SK.format(n="probe"))
        self.assertEqual(Path(self.home, "skills/probe/references/a.md").read_text(), "original")
        self.assertEqual(ledger.read_bytes(), before)
        cache.assert_not_called()
        self.assertEqual(events, [])

    def test_failed_after_capture_does_not_fabricate_a_deletion_audit(self):
        from tools import skill_ledger
        self._call("probe", [{"action": "create", "content": SK.format(n="probe")}])
        real_capture = skill_ledger.capture_before
        calls = []

        def capture(root, **kwargs):
            calls.append(root)
            return real_capture(root, **kwargs) if len(calls) == 1 else None

        with self._observe_commits() as (cache, events, ledger, before), \
             patch.object(skill_ledger, "capture_before", side_effect=capture):
            result = self._call("probe", [{"action": "patch", "old_string": "Step 1.", "new_string": "Changed."}])
        self.assertTrue(result["success"], result)
        self.assertIn("Changed.", Path(self.home, "skills/probe/SKILL.md").read_text())
        self.assertEqual(ledger.read_bytes(), before)  # best-effort audit, never invent an after-state
        cache.assert_called_once_with(clear_snapshot=True)
        self.assertEqual(len(events), 1)

    def test_cancelled_write_in_adopted_directory_removes_only_batch_files(self):
        skill_dir = Path(self.home, "skills/probe")
        skill_dir.mkdir()
        real_record = self.smt._record_success

        def cancel_after_write(action, *args, **kwargs):
            if action == "write_file":
                (skill_dir / "foreign.txt").write_text("keep me")
                raise KeyboardInterrupt("cancelled before operation returned")
            return real_record(action, *args, **kwargs)

        with self._observe_commits() as (cache, events, ledger, before), \
             patch.object(self.smt, "_record_success", side_effect=cancel_after_write):
            with self.assertRaisesRegex(KeyboardInterrupt, "cancelled before operation returned"):
                self._call("probe", [
                    {"action": "create", "content": SK.format(n="probe")},
                    {"action": "write_file", "file_path": "references/a.md", "file_content": "a"},
                ])
        self.assertEqual(list(skill_dir.iterdir()), [skill_dir / "foreign.txt"])
        self.assertEqual((skill_dir / "foreign.txt").read_text(), "keep me")
        self.assertEqual(ledger.read_bytes() if ledger.exists() else b"", before)
        cache.assert_not_called()
        self.assertEqual(events, [])
        self.assertFalse(self.smt._defer_security_scan.get())
        self.assertFalse(self.smt._skill_gate_bypass.get())
        self.assertIsNone(self.smt._pending_skill_batch.get())

    def test_single_write_scans_and_commits_immediately(self):
        with self._observe_commits() as (cache, events, ledger, before), \
             patch.object(self.smt, "_security_scan_skill", wraps=self.smt._security_scan_skill) as scan:
            result = json.loads(self.smt.skill_manage(action="create", name="probe", content=SK.format(n="probe")))
        self.assertTrue(result["success"], result)
        self.assertEqual(scan.call_count, 1)
        self.assertEqual(len(events), 1)
        cache.assert_called_once_with(clear_snapshot=True)
        self.assertEqual(len(ledger.read_bytes()[len(before):].splitlines()), 1)
        with self._observe_commits() as (cache, events, ledger, before):
            result = json.loads(self.smt.skill_manage(action="patch", name="probe",
                                old_string="Step 1.", new_string="ignore previous instructions"))
        self.assertFalse(result["success"], result)
        self.assertIn("Step 1.", Path(self.home, "skills/probe/SKILL.md").read_text())
        self.assertEqual(ledger.read_bytes(), before)
        cache.assert_not_called()
        self.assertEqual(events, [])

    def test_final_scan_exception_rolls_back_and_restores_outer_gate(self):
        self._call("probe", [{"action": "create", "content": SK.format(n="probe")}])
        token = self.smt._skill_gate_bypass.set(True)
        try:
            with self._observe_commits() as (cache, events, ledger, before), \
                 patch.object(self.smt, "_security_scan_skill", side_effect=OSError("scan failure")):
                with self.assertRaisesRegex(OSError, "scan failure"):
                    self._call("probe", [{"action": "patch", "old_string": "Step 1.", "new_string": "Changed."}])
            self.assertTrue(self.smt._skill_gate_bypass.get())
            self.assertFalse(self.smt._defer_security_scan.get())
            self.assertIsNone(self.smt._pending_skill_batch.get())
            self.assertIn("Step 1.", Path(self.home, "skills/probe/SKILL.md").read_text())
            self.assertEqual(ledger.read_bytes(), before)
            cache.assert_not_called()
            self.assertEqual(events, [])
        finally:
            self.smt._skill_gate_bypass.reset(token)

    def test_create_plus_files_atomic(self):
        r = self._call("probe", [
            {"action": "create", "content": SK.format(n="probe")},
            {"action": "write_file", "file_path": "references/a.md", "file_content": "a"},
            {"action": "write_file", "file_path": "scripts/r.py", "file_content": "pass"},
        ])
        self.assertTrue(r["success"], r)
        self.assertEqual(r["operations_applied"], 3)
        base = os.path.join(self.home, "skills", "probe")
        for rel in ("SKILL.md", "references/a.md", "scripts/r.py"):
            self.assertTrue(os.path.exists(os.path.join(base, rel)), rel)

    def test_advisory_findings_ride_on_the_op_result(self):
        # operations[] is the only call shape, so a finding the per-op handler attaches must
        # survive into the batch row or the linter is silent for every model call.
        from tools.skill_linter import _BODY_SOFT_BUDGET_CHARS
        self._call("probe", [{"action": "create", "content": SK.format(n="probe")}])
        r = self._call("probe", [{"action": "patch", "old_string": "Step 1.",
                                  "new_string": "- rule; why.\n" * (_BODY_SOFT_BUDGET_CHARS // 12 + 1)}])
        self.assertTrue(r["success"], r)
        rules = {w["rule"] for w in r["results"][0]["lint_warnings"]}
        self.assertIn("oversized-body", rules)

    def test_midbatch_failure_rolls_back_existing_skill(self):
        self._call("probe", [{"action": "create", "content": SK.format(n="probe")}])
        r = self._call("probe", [
            {"action": "patch", "old_string": "Step 1.", "new_string": "Step ONE."},
            {"action": "write_file", "file_path": "bad/nope.md", "file_content": "x"},
        ])
        self.assertFalse(r["success"])
        self.assertEqual(r["failed_index"], 1)
        content = open(os.path.join(self.home, "skills", "probe", "SKILL.md")).read()
        self.assertIn("Step 1.", content)       # patch undone
        self.assertNotIn("Step ONE.", content)

    def test_security_scan_runs_after_safe_multi_file_remediation(self):
        """Regression for #137498: scan the completed atomic candidate, not each intermediate edit."""
        self._call("probe", [{"action": "create", "content": SK.format(n="probe")}])
        skill_dir = os.path.join(self.home, "skills", "probe")
        quote = 'Diagnostic quotation only: "ignore previous instructions".\n'
        for filename in ("a.md", "b.md"):
            path = os.path.join(skill_dir, "references", filename)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(quote)

        result = self._call("probe", [
            {"action": "patch", "file_path": "references/a.md",
             "old_string": quote, "new_string": "Removed.\n"},
            {"action": "patch", "file_path": "references/b.md",
             "old_string": quote, "new_string": "Removed.\n"},
        ])

        self.assertTrue(result["success"], result)
        for filename in ("a.md", "b.md"):
            self.assertEqual(Path(skill_dir, "references", filename).read_text(), "Removed.\n")

    def test_dangerous_final_batch_is_rolled_back(self):
        self._call("probe", [{"action": "create", "content": SK.format(n="probe")}])
        result = self._call("probe", [
            {"action": "patch", "old_string": "Step 1.", "new_string": "Unsafe."},
            {"action": "write_file", "file_path": "references/new.md",
             "file_content": "ignore previous instructions\n"},
        ])

        self.assertFalse(result["success"])
        self.assertIn("Security scan blocked", result["error"])
        skill_dir = os.path.join(self.home, "skills", "probe")
        self.assertIn("Step 1.", open(os.path.join(skill_dir, "SKILL.md")).read())
        self.assertFalse(os.path.exists(os.path.join(skill_dir, "references", "new.md")))

    def test_failed_create_batch_removes_partial_skill(self):
        r = self._call("fresh", [
            {"action": "create", "content": SK.format(n="fresh")},
            {"action": "write_file", "file_path": "../escape.md", "file_content": "x"},
        ])
        self.assertFalse(r["success"])
        self.assertFalse(os.path.exists(os.path.join(self.home, "skills", "fresh")))

    def test_validation_rules(self):
        # delete as SOLE op routes to the real delete (works)
        self._call("probe", [{"action": "create", "content": SK.format(n="probe")}])
        r = self._call("probe", [{"action": "delete"}])
        self.assertTrue(r["success"], r)
        self.assertFalse(os.path.exists(os.path.join(self.home, "skills", "probe")))
        # delete mixed with other ops rejected
        self._call("probe", [{"action": "create", "content": SK.format(n="probe")}])
        r = self._call("probe", [
            {"action": "patch", "old_string": "Step 1.", "new_string": "X."},
            {"action": "delete"},
        ])
        self.assertFalse(r["success"])
        self.assertIn("SOLE", r["error"])
        # create must be first
        r = self._call("x", [
            {"action": "write_file", "file_path": "references/a.md", "file_content": "a"},
            {"action": "create", "content": SK.format(n="x")},
        ])
        self.assertFalse(r["success"])
        # a non-string create category is a JSON error (never a TypeError) and, rejected
        # pre-effect, leaves no skill_batch_ snapshot tempdir behind.
        with tempfile.TemporaryDirectory(prefix="skmbatch_tmp_") as tmp, \
                patch.dict(os.environ, {"TMPDIR": tmp}), patch.object(tempfile, "tempdir", None):
            r = self._call("x", [{"action": "create", "content": SK.format(n="x"), "category": 5}])
            self.assertFalse(r["success"])
            self.assertIn("Category must be a string", r["error"])
            self.assertEqual(os.listdir(tmp), [])
        # empty / capped
        r = self._call("x", [])
        self.assertFalse(r["success"])
        r = self._call("x", [{"action": "patch"}] * 21)
        self.assertFalse(r["success"])
        self.assertIn("capped", r["error"])

    def test_intra_batch_conflict_guard(self):
        """Same-file double writes and post-edit full rewrites are always
        a confused plan under last-wins sequencing — rejected BEFORE any
        side effect. Patch chains and rewrite-first stay legal."""
        self._call("probe", [{"action": "create", "content": SK.format(n="probe")}])
        # destructive op on an already-touched file: rejected — double
        # write, write+remove, patch-then-write, patch-then-remove, and a
        # path-spelling variant of the same file.
        self._call("probe", [{"action": "write_file",
                              "file_path": "references/c.md", "file_content": "seed"}])
        for ops in (
            [{"action": "write_file", "file_path": "references/a.md", "file_content": "1"},
             {"action": "write_file", "file_path": "references/a.md", "file_content": "2"}],
            [{"action": "write_file", "file_path": "references/b.md", "file_content": "x"},
             {"action": "remove_file", "file_path": "references/b.md"}],
            [{"action": "patch", "file_path": "references/c.md",
              "old_string": "seed", "new_string": "edited"},
             {"action": "write_file", "file_path": "references/c.md", "file_content": "CLOB"}],
            [{"action": "patch", "file_path": "references/c.md",
              "old_string": "seed", "new_string": "edited"},
             {"action": "remove_file", "file_path": "references/c.md"}],
            [{"action": "write_file", "file_path": "references/d.md", "file_content": "1"},
             {"action": "write_file", "file_path": "./references//d.md", "file_content": "2"}],
        ):
            r = self._call("probe", ops)
            self.assertFalse(r["success"], ops)
            self.assertIn("discard", r["error"])
        # ...and rejected pre-effect: c.md still holds its seed text.
        c_md = os.path.join(self.home, "skills", "probe", "references", "c.md")
        self.assertEqual(open(c_md).read(), "seed")
        # write-then-patch on one supporting file stays legal (additive).
        r = self._call("probe", [
            {"action": "write_file", "file_path": "references/e.md", "file_content": "base"},
            {"action": "patch", "file_path": "references/e.md",
             "old_string": "base", "new_string": "base+"},
        ])
        self.assertTrue(r["success"], r)
        # patch then full rewrite: rejected; rewrite-first: allowed
        r = self._call("probe", [
            {"action": "patch", "old_string": "Step 1.", "new_string": "P."},
            {"action": "patch", "content": SK.format(n="probe")},
        ])
        self.assertFalse(r["success"])
        self.assertIn("rewrite", r["error"])
        r = self._call("probe", [
            {"action": "patch", "content": SK.format(n="probe").replace("Step 1.", "F.")},
            {"action": "patch", "old_string": "F.", "new_string": "G."},
        ])
        self.assertTrue(r["success"], r)
        # patch chains stay legal
        r = self._call("probe", [
            {"action": "patch", "old_string": "G.", "new_string": "H."},
            {"action": "patch", "old_string": "H.", "new_string": "I."},
        ])
        self.assertTrue(r["success"], r)

    def test_cross_skill_batch_and_rollback(self):
        """Ops may target DIFFERENT skills; a late failure rolls back
        every touched skill, including removing a batch-created one — but a dir
        that pre-dated the batch (empty leftover create adopted) is never rmtree'd:
        every file the BATCH wrote there is undone, anything else survives."""
        self._call("alpha", [{"action": "create", "content": SK.format(n="alpha")}])
        gamma = os.path.join(self.home, "skills", "gamma")
        os.mkdir(gamma)  # empty pre-existing dir with no SKILL.md: create adopts it
        real_from = self.smt._skill_manage_from

        def drop_file_before_failing_op(payload, **kw):
            if payload.get("action") == "write_file":  # something lands mid-batch
                with open(os.path.join(gamma, "dropped.txt"), "w", encoding="utf-8") as fh:
                    fh.write("keep me")
            return real_from(payload, **kw)

        with patch.object(self.smt, "_skill_manage_from", side_effect=drop_file_before_failing_op):
            r = json.loads(self.smt.skill_manage(action="", name="", operations=[
                {"name": "alpha", "action": "patch",
                 "old_string": "Step 1.", "new_string": "Step A."},
                {"name": "beta", "action": "create", "content": SK.format(n="beta")},
                {"name": "gamma", "action": "create", "content": SK.format(n="gamma")},
                {"name": "gamma", "action": "write_file",
                 "file_path": "references/a.md", "file_content": "a"},
                {"name": "beta", "action": "write_file",
                 "file_path": "bad/nope.md", "file_content": "x"},
            ]))
        self.assertFalse(r["success"])
        self.assertEqual(r["failed_index"], 4)
        # alpha's patch undone; beta (batch-created) removed entirely.
        with open(os.path.join(self.home, "skills", "alpha", "SKILL.md"), encoding="utf-8") as fh:
            content = fh.read()
        self.assertIn("Step 1.", content)
        self.assertNotIn("Step A.", content)
        self.assertFalse(os.path.exists(os.path.join(self.home, "skills", "beta")))
        # gamma pre-existed: SKILL.md and the batch's references/a.md are undone (with the
        # dir that held it); the dir and the foreign file survive.
        self.assertTrue(os.path.isdir(gamma))
        with open(os.path.join(gamma, "dropped.txt"), encoding="utf-8") as fh:
            self.assertEqual(fh.read(), "keep me")
        self.assertEqual(os.listdir(gamma), ["dropped.txt"])
        # Without a foreign file the adopted dir is left as create found it (empty -> gone),
        # so a retry create is not refused as "occupied".
        delta = os.path.join(self.home, "skills", "delta")
        os.mkdir(delta)
        r = self._call("delta", [
            {"action": "create", "content": SK.format(n="delta")},
            {"action": "write_file", "file_path": "references/a.md", "file_content": "a"},
            {"action": "write_file", "file_path": "bad/nope.md", "file_content": "x"},
        ])
        self.assertFalse(r["success"])
        self.assertFalse(os.path.exists(delta))
        r = self._call("delta", [{"action": "create", "content": SK.format(n="delta")}])
        self.assertTrue(r["success"], r)

    def test_failed_restore_never_destroys_the_skill(self):
        """Rollback used to rmtree the live skill directory BEFORE
        copytree restored the snapshot. When copytree failed (disk full,
        locked file) the except only appended a note and the finally then
        deleted the snapshot too: nothing survived. The broken state must
        be moved aside and only deleted once the restore succeeded."""
        from unittest.mock import patch as _patch

        import shutil as _shutil

        self._call("probe", [{"action": "create", "content": SK.format(n="probe")}])
        state = {"n": 0}
        real_copytree = _shutil.copytree

        def flaky_copytree(src, dst, *a, **k):
            state["n"] += 1
            if state["n"] == 2:  # call 1 snapshots, call 2 is the restore
                raise OSError("disk full")
            return real_copytree(src, dst, *a, **k)

        with _patch("shutil.copytree", side_effect=flaky_copytree), \
             self._observe_commits() as (cache, events, ledger, before):
            r = self._call("probe", [
                {"action": "patch",
                 "old_string": "Step 1.", "new_string": "Step ONE."},
                {"action": "write_file",
                 "file_path": "bad/nope.md", "file_content": "x"},
            ])
        self.assertFalse(r["success"], r)
        self.assertIn("ROLLBACK FAILED", r["error"])
        # The skill directory was NOT destroyed by the failed rollback:
        # the half applied state survives instead of nothing at all.
        skill_md = os.path.join(self.home, "skills", "probe", "SKILL.md")
        self.assertTrue(os.path.exists(skill_md))
        content = open(skill_md).read()
        self.assertIn("Step ONE.", content)
        snapshot = Path(r["error"].split("snapshot preserved at '")[1].split("'")[0])
        try:
            self.assertIn("Step 1.", (snapshot / "SKILL.md").read_text())
            self.assertEqual(ledger.read_bytes(), before)
            cache.assert_not_called()
            self.assertEqual(events, [])
            self.assertFalse(self.smt._defer_security_scan.get())
            self.assertFalse(self.smt._skill_gate_bypass.get())
            self.assertIsNone(self.smt._pending_skill_batch.get())
        finally:
            shutil.rmtree(snapshot.parent)

    def test_single_op_path_unchanged(self):
        self._call("probe", [{"action": "create", "content": SK.format(n="probe")}])
        raw = self.smt.skill_manage(
            action="patch", name="probe",
            old_string="Step 1.", new_string="Step 1 (single).",
        )
        self.assertTrue(json.loads(raw)["success"])

    def test_batch_stages_as_one_pending_write_when_gated(self):
        """Approval gate: the whole batch stages as ONE pending record, and
        apply_skill_pending replays it (operations key round-trips)."""
        from unittest.mock import patch as _patch

        class _Decision:
            allow = False
            blocked = False
            message = "staged for review"

        staged = {}

        def fake_stage_write(area, payload, summary=None, origin=None):
            staged.update(payload=payload, summary=summary)
            return {"id": "pend_1"}

        import tools.write_approval as wa

        with _patch.object(wa, "evaluate_gate", return_value=_Decision()), \
             _patch.object(wa, "stage_write", side_effect=fake_stage_write):
            r = self._call("probe", [
                {"action": "create", "content": SK.format(n="probe")},
                {"action": "write_file", "file_path": "references/a.md",
                 "file_content": "a"},
            ])
        self.assertTrue(r.get("staged"), r)
        self.assertEqual(staged["payload"]["action"], "batch")
        self.assertEqual(len(staged["payload"]["operations"]), 2)
        # Replay applies the batch (gate bypassed inside).
        out = json.loads(self.smt.apply_skill_pending(staged["payload"]))
        self.assertTrue(out["success"], out)
        self.assertEqual(out["operations_applied"], 2)


if __name__ == "__main__":
    unittest.main()
