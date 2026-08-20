import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class PivotWorkerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["HERMES_HOME"] = str(Path(self.tmp.name) / "companyintel")
        os.environ["HERMES_PROFILE"] = "companyintel"

    def tearDown(self):
        self.tmp.cleanup()

    def _call(self, payload):
        from tools.companyintel_graph_tool import companyintel_graph
        return json.loads(companyintel_graph(payload))

    def test_registry_expands_domain_to_typed_public_search(self):
        from tools.companyintel_pivots import expand_pivots

        specs = expand_pivots("domain", "example.com")
        names = {spec.pivot_type for spec in specs}
        self.assertIn("exact_search", names)
        self.assertTrue(all(spec.worker == "public_search" for spec in specs if spec.pivot_type == "exact_search"))

    def test_public_search_worker_persists_result_and_completes_task(self):
        self._call({"action": "init_run", "run_id": "run_pivot_001", "target_url": "https://example.com"})
        fixture = b'''<html><body>
          <a class="result__a" href="https://result.example/company">Example Company official</a>
          <a class="result__a" href="https://directory.example/example">Example directory profile</a>
          <a class="result__a" href="http://127.0.0.1/private">must be rejected</a>
          <a class="result__snippet">Official company contact and legal information.</a>
        </body></html>'''

        with patch("tools.companyintel_public_search._fetch_search", return_value=("text/html", fixture)):
            result = self._call({
                "action": "execute_frontier",
                "run_id": "run_pivot_001",
                "worker_id": "pivot-worker-a",
                "pivot_type": "exact_search",
                "lease_seconds": 60,
            })
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["outcome"], "COMPLETED_WITH_RESULTS")
        self.assertGreaterEqual(result["persisted_results"], 2)

        summary = self._call({"action": "summary", "run_id": "run_pivot_001"})
        self.assertGreaterEqual(summary["evidence"], 2)
        self.assertGreaterEqual(summary["search_log"], 1)
        status = self._call({"action": "frontier_status", "run_id": "run_pivot_001"})
        completed = [task for task in status["tasks"] if task["state"] == "COMPLETED"]
        self.assertEqual(len(completed), 1)

    def test_public_search_empty_result_is_explicit_zero_outcome(self):
        self._call({"action": "init_run", "run_id": "run_pivot_002", "target_url": "https://example.com"})
        with patch("tools.companyintel_public_search._fetch_search", return_value=("text/html", b"<html></html>")):
            result = self._call({
                "action": "execute_frontier",
                "run_id": "run_pivot_002",
                "worker_id": "pivot-worker-a",
                "pivot_type": "exact_search",
            })
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["outcome"], "COMPLETED_ZERO_RESULTS")
        self.assertEqual(result["persisted_results"], 0)

    def test_registered_specialized_pivots_dispatch_to_typed_workers(self):
        from tools.companyintel_pivots import get_pivot

        self.assertEqual(get_pivot("maps").worker, "maps_search")
        self.assertEqual(get_pivot("marketplaces").worker, "marketplace_search")
        self.assertEqual(get_pivot("documents").worker, "document_search")

    def test_typed_workers_execute_maps_marketplace_and_documents(self):
        self._call({"action": "init_run", "run_id": "run_typed_001", "target_url": "https://example.com"})
        phone = self._call({
            "action": "record_observation", "run_id": "run_typed_001", "node_type": "phone",
            "value": "+380501234567", "source_url": "https://example.com", "excerpt": "Public phone +380501234567",
        })
        fixture = b'''<html><body>
          <a class="result__a" href="https://maps.example/place">Map place</a>
          <a class="result__a" href="https://market.example/store">Marketplace store</a>
          <a class="result__a" href="https://docs.example/report.pdf">Public report</a>
        </body></html>'''
        with patch("tools.companyintel_public_search._fetch_search", return_value=("text/html", fixture)):
            outcomes = []
            for pivot_type in ("maps", "marketplaces", "documents"):
                result = self._call({
                    "action": "execute_frontier", "run_id": "run_typed_001", "worker_id": "typed-worker",
                    "pivot_type": pivot_type, "lease_seconds": 60,
                })
                outcomes.append(result)
        self.assertEqual([item["outcome"] for item in outcomes], ["COMPLETED_WITH_RESULTS"] * 3)
        self.assertEqual([item["pivot_type"] for item in outcomes], ["maps", "marketplaces", "documents"])

    def test_budget_stop_and_resume_are_durable(self):
        self._call({"action": "init_run", "run_id": "run_resume_001", "target_url": "https://example.com", "max_tasks": 1})
        pages = {"https://example.com/": ("text/html", b"<title>Example</title>"), "https://example.com/robots.txt": ("text/plain", b"User-agent: *")}
        with patch("tools.companyintel_inventory._fetch_url", side_effect=lambda url, limits: pages.get(url, ("", b""))), \
             patch("tools.companyintel_public_search._fetch_search", return_value=("text/html", b"<html></html>")):
            first = self._call({"action": "run_frontier", "run_id": "run_resume_001", "worker_id": "resume-worker", "max_tasks": 5})
            before = self._call({"action": "summary", "run_id": "run_resume_001"})
            resumed = self._call({"action": "resume_frontier", "run_id": "run_resume_001", "worker_id": "resume-worker", "additional_tasks": 2, "max_tasks": 2})
            after = self._call({"action": "summary", "run_id": "run_resume_001"})
        self.assertTrue(first["ok"], first)
        self.assertEqual(before["status"], "PARTIAL")
        self.assertIn("BUDGET_EXHAUSTED", before["saturation_reason"])
        self.assertTrue(resumed["ok"], resumed)
        self.assertGreater(after["usage"]["tasks_executed"], before["usage"]["tasks_executed"])

    def test_identity_candidate_scoring_persists_matching_conflicting_and_missing_dimensions(self):
        self._call({"action": "init_run", "run_id": "run_candidate_001", "target_url": "https://example.com"})
        candidate = self._call({
            "action": "record_identity_candidate", "run_id": "run_candidate_001",
            "candidate_id": "candidate-example-official", "legal_name": "Example LLC",
            "match_types": ["exact_official_legal_id"],
            "conflicting_dimensions": [], "missing_dimensions": ["director"],
            "evidence_ids": [],
        })
        self.assertTrue(candidate["ok"], candidate)
        self.assertEqual(candidate["score"], 100)
        self.assertEqual(candidate["status"], "VERIFIED")
        self.assertEqual(candidate["matching_dimensions"], ["exact_official_legal_id"])
        summary = self._call({"action": "summary", "run_id": "run_candidate_001"})
        self.assertEqual(summary["identity_candidates"][0]["candidate_id"], "candidate-example-official")
        gate = self._call({"action": "legal_identity_exhaustion", "run_id": "run_candidate_001"})
        self.assertFalse(gate["allowed_not_found"])
        self.assertIn("identity_candidate_present", gate["reason"])

        self._call({"action": "init_run", "run_id": "run_legal_gate_001", "target_url": "https://example.com"})
        result = self._call({"action": "legal_identity_exhaustion", "run_id": "run_legal_gate_001"})
        self.assertTrue(result["ok"], result)
        self.assertFalse(result["allowed_not_found"])
        self.assertEqual(len(result["coverage"]), 15)
        self.assertIn(result["coverage"]["official_registry"], {"UNAVAILABLE", "PENDING", "MISSING"})
        self.assertIn("official_registry", result["blocking_classes"])
        summary = self._call({"action": "summary", "run_id": "run_legal_gate_001"})
        self.assertIn("legal_identity_exhaustion", summary)
        self.assertEqual(summary["legal_identity_exhaustion"]["status"], "PENDING")

    def test_exhaustion_gate_blocks_not_found_when_coverage_is_missing(self):
        self._call({"action": "init_run", "run_id": "run_gate_001", "target_url": "https://example.com"})
        gate = self._call({"action": "exhaustion_gate", "run_id": "run_gate_001"})
        self.assertTrue(gate["ok"], gate)
        self.assertFalse(gate["allowed_not_found"])
        self.assertIn("coverage", gate["reason"])

    def test_run_frontier_automatically_dispatches_inventory_then_search(self):
        self._call({"action": "init_run", "run_id": "run_auto_001", "target_url": "https://example.com"})
        pages = {
            "https://example.com/": ("text/html", b'<title>Example</title><a href="/about">About</a>'),
            "https://example.com/robots.txt": ("text/plain", b"User-agent: *"),
        }
        search_fixture = b'''<html><body>
          <a class="result__a" href="https://result.example/company">Example result</a>
          <a class="result__snippet">Public company result.</a>
        </body></html>'''

        def fetch_inventory(url, limits):
            return pages.get(url, ("", b""))

        with patch("tools.companyintel_inventory._fetch_url", side_effect=fetch_inventory), \
             patch("tools.companyintel_public_search._fetch_search", return_value=("text/html", search_fixture)):
            result = self._call({
                "action": "run_frontier",
                "run_id": "run_auto_001",
                "worker_id": "auto-worker",
                "max_tasks": 2,
                "lease_seconds": 60,
            })
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["tasks_executed"], 2)
        self.assertEqual(result["outcomes"][0]["pivot_type"], "site_inventory")
        self.assertEqual(result["outcomes"][1]["pivot_type"], "exact_search")
        summary = self._call({"action": "summary", "run_id": "run_auto_001"})
        self.assertGreaterEqual(summary["search_log"], 1)
        self.assertGreaterEqual(summary["evidence"], 1)


if __name__ == "__main__":
    unittest.main()
