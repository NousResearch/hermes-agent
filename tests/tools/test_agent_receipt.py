"""Behavior contract tests for tools/agent_receipt — the receipt schema guard.

Asserts INVARIANTS (how the validator must behave against the schema), not snapshots.
Run: python3 -m pytest tests/tools/test_agent_receipt.py -q
"""
from __future__ import annotations

import copy
import json
import os
import unittest
from unittest import mock

from tools import agent_receipt as ar


def _good():
    return {
        "claim_id": "rcpt-001",
        "producer": "reviewer-codex",
        "task": "Review the pricing route change",
        "stop_reason": "completed",
        "sources": [{"ref": "app/Pricing.php:42", "status": "verified"}],
        "touched": ["app/Pricing.php"],
        "commands": [{"cmd": "phpunit tests/PricingTest.php", "result": "3 passed", "status": "verified"}],
        "blockers": [],
        "next_owner": "none",
    }


class TestSchemaIntegrity(unittest.TestCase):
    def test_embedded_matches_on_disk_schema(self):
        """The embedded fallback must never drift from the canonical .json — else the
        fallback path would silently enforce stale rules."""
        with open(ar.SCHEMA_PATH, "r", encoding="utf-8") as fh:
            disk = json.load(fh)
        self.assertEqual(ar.EMBEDDED_SCHEMA, disk)

    def test_load_schema_from_arbitrary_cwd(self):
        cwd = os.getcwd()
        try:
            os.chdir("/tmp")
            sch = ar.load_schema()
            self.assertEqual(sch.get("title"), "agent_receipt")
            self.assertIn("stop_reason", sch.get("required", []))
        finally:
            os.chdir(cwd)

    def test_missing_schema_file_falls_back_to_embedded(self):
        sch = ar.load_schema("/nonexistent/agent_receipt.schema.json")
        self.assertEqual(sch, ar.EMBEDDED_SCHEMA)


class TestValidate(unittest.TestCase):
    def test_well_formed_passes(self):
        ok, errs = ar.validate(_good())
        self.assertTrue(ok, errs)

    def test_empty_collections_pass(self):
        r = _good()
        r.update({"sources": [], "touched": [], "commands": [], "blockers": [], "next_owner": "none"})
        ok, errs = ar.validate(r)
        self.assertTrue(ok, errs)

    def test_each_missing_required_key_fails(self):
        for key in ar.load_schema()["required"]:
            r = _good()
            r.pop(key, None)
            ok, _ = ar.validate(r)
            self.assertFalse(ok, f"missing {key} should fail")

    def test_wrong_type_for_sources_fails(self):
        r = _good()
        r["sources"] = "nope"
        self.assertFalse(ar.validate(r)[0])

    def test_bad_stop_reason_enum_fails(self):
        r = _good()
        r["stop_reason"] = "finished"
        self.assertFalse(ar.validate(r)[0])

    def test_empty_anchor_fails(self):
        r = _good()
        r["claim_id"] = "   "
        self.assertFalse(ar.validate(r)[0])

    def test_schema_driven_not_hardcoded(self):
        """Drop stop_reason from a mutated schema's required[] -> the missing-stop_reason
        case must now pass. Proves rules come FROM the schema, not Python constants."""
        mutated = copy.deepcopy(ar.load_schema())
        mutated["required"] = [r for r in mutated["required"] if r != "stop_reason"]
        r = _good()
        r.pop("stop_reason", None)
        ok, _ = ar.validate(r, mutated)
        self.assertTrue(ok, "validator is not schema-driven — still rejected after schema dropped the field")


class TestFailOpen(unittest.TestCase):
    def test_validator_code_exception_fails_open_and_marks(self):
        with mock.patch.dict(os.environ, {"HERMES_STATE_DIR": "/tmp/artest_failopen"}):
            # reload marker dir
            ar._STATE_DIR = "/tmp/artest_failopen"  # noqa: SLF001
            try:
                with mock.patch.object(ar, "_validate_node", side_effect=RuntimeError("boom")):
                    ok, errs = ar.validate(_good())
                self.assertTrue(ok, "a validator CODE fault must fail OPEN (never brick the gate)")
                self.assertEqual(errs, [])
                self.assertTrue(os.path.exists("/tmp/artest_failopen/receipt_gate_degraded"),
                                "fail-open must write a loud degraded marker")
            finally:
                import shutil
                shutil.rmtree("/tmp/artest_failopen", ignore_errors=True)
                ar._STATE_DIR = os.path.expanduser("~/.hermes/state")  # noqa: SLF001


class TestExtract(unittest.TestCase):
    def test_extract_from_markdown(self):
        text = "work done\n\n```receipt\n" + json.dumps(_good()) + "\n```\nbye"
        obj = ar.extract_receipt(text)
        assert obj is not None
        self.assertEqual(obj["claim_id"], "rcpt-001")

    def test_no_block_returns_none(self):
        self.assertIsNone(ar.extract_receipt("just prose"))

    def test_validate_text_no_block_is_invalid(self):
        ok, _ = ar.validate_text("prose with no block")
        self.assertFalse(ok)


class TestOwesReceipt(unittest.TestCase):
    def test_kanban_always_owes(self):
        self.assertTrue(ar.owes_receipt(surface="kanban"))

    def test_delegate_orchestrator_owes(self):
        self.assertTrue(ar.owes_receipt(surface="delegate", role="orchestrator"))

    def test_delegate_leaf_with_file_write_owes(self):
        trace = [{"tool": "read_file"}, {"tool": "write_file"}]
        self.assertTrue(ar.owes_receipt(surface="delegate", role="leaf", tool_trace=trace))

    def test_delegate_leaf_with_command_owes(self):
        trace = [{"tool": "terminal"}]
        self.assertTrue(ar.owes_receipt(surface="delegate", role="leaf", tool_trace=trace))

    def test_delegate_pure_lookup_does_not_owe(self):
        trace = [{"tool": "read_file"}, {"tool": "search_files"}]
        self.assertFalse(ar.owes_receipt(surface="delegate", role="leaf", tool_trace=trace))

    def test_cron_claimy_not_silent_owes(self):
        body = "deployed commit abc123, tests pass, returned 200"
        self.assertTrue(ar.owes_receipt(surface="cron", response_text=body))

    def test_cron_silent_does_not_owe(self):
        self.assertFalse(ar.owes_receipt(surface="cron", response_text="[SILENT] all clear"))

    def test_cron_single_hint_below_threshold(self):
        self.assertFalse(ar.owes_receipt(surface="cron", response_text="just one commit mention"))


if __name__ == "__main__":
    unittest.main()
