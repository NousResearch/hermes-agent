"""Standalone tests for tools.docker_health_tool."""
from __future__ import annotations

import shutil
import unittest
from collections import Counter
from unittest import mock

from tools.docker_health_tool import (
    DEFAULT_COMMAND_TIMEOUT,
    FAILURE_CODE,
    TOP_ISSUES_CAP,
    TOOL_NAME,
    _bounded_counter,
    _container_health_summary,
    _extract_status_token,
    _service_status_by_compose_file,
    _top_issues,
    is_available,
    run,
)
from tools.validation_helper import Success


class TestDockerHealthTool(unittest.TestCase):
    def run(self, *args, **kwargs):  # pragma: no cover - runner override
        return super().run(*args, **kwargs)

    # ------------------------------------------------------------------
    # Envelope structure and count parsing
    # ------------------------------------------------------------------
    @mock.patch("tools.docker_health_tool._probe_daemon", return_value=True)
    @mock.patch("tools.docker_health_tool._discover_compose_files")
    @mock.patch("tools.docker_health_tool._service_status_by_compose_file")
    def test_success_envelope_shape(self, mock_service_stats, mock_discover, _probe):
        mock_discover.return_value = ["compose.yaml"]
        mock_service_stats.return_value = {
            "compose.yaml": {"running": 1, "healthy": 1},
        }
        envelope = run()
        self.assertIsInstance(envelope, dict)
        self.assertEqual(envelope.get("kind"), "success")
        data = envelope.get("data") or {}
        self.assertIn("docker_daemon_reachable", data)
        self.assertIn("compose_files", data)
        self.assertIn("service_status_counts", data)
        self.assertIn("container_health_summary", data)
        self.assertIn("top_issues", data)
        self.assertTrue(data["docker_daemon_reachable"])
        self.assertEqual(data["service_status_counts"], mock_service_stats.return_value)
        health = data["container_health_summary"]
        self.assertEqual(health["running"], 1)
        self.assertEqual(health["healthy"], 1)
        self.assertEqual(health["total"], 2)

    @mock.patch("tools.docker_health_tool._probe_daemon", return_value=True)
    @mock.patch("tools.docker_health_tool._discover_compose_files")
    @mock.patch("tools.docker_health_tool._service_status_by_compose_file")
    def test_count_parsing_multistatus(self, mock_service_stats, mock_discover, _probe):
        mock_discover.return_value = ["a.yaml", "b.yaml"]
        mock_service_stats.return_value = {
            "a.yaml": {"running": 2, "unhealthy": 1},
            "b.yaml": {"exited": 3},
        }
        envelope = run()
        data = envelope["data"]
        health = data["container_health_summary"]
        self.assertEqual(health["running"], 2)
        self.assertEqual(health["stopped"], 3)
        self.assertEqual(health["unhealthy"], 1)
        self.assertEqual(health["total"], 6)
        counts = data["service_status_counts"]
        self.assertEqual(counts["a.yaml"], {"running": 2, "unhealthy": 1})
        self.assertEqual(counts["b.yaml"], {"exited": 3})

    # ------------------------------------------------------------------
    # Availability gating
    # ------------------------------------------------------------------
    @mock.patch("shutil.which", return_value=None)
    def test_is_available_missing_cli(self, _which):
        self.assertFalse(is_available())

    @mock.patch("subprocess.run")
    @mock.patch("shutil.which", return_value="/usr/bin/docker")
    def test_is_available_cli_present_deamon_down(self, _which, mock_run):
        mock_run.return_value.__enter__ = mock.Mock(return_value=mock_run.return_value)
        mock_run.return_value.__exit__ = mock.Mock(return_value=False)
        mock_run.return_value.returncode = 1
        self.assertFalse(is_available())

    @mock.patch("subprocess.run")
    @mock.patch("shutil.which", return_value="/usr/bin/docker")
    def test_is_available_cli_present_deamon_up(self, _which, mock_run):
        mock_run.return_value.__enter__ = mock.Mock(return_value=mock_run.return_value)
        mock_run.return_value.__exit__ = mock.Mock(return_value=False)
        mock_run.return_value.returncode = 0
        self.assertTrue(is_available())

    # ------------------------------------------------------------------
    # Missing docker failure envelope
    # ------------------------------------------------------------------
    @mock.patch("tools.docker_health_tool._probe_daemon", return_value=False)
    def test_missing_docker_produces_failure_envelope(self, _probe):
        envelope = run()
        self.assertIsInstance(envelope, dict)
        self.assertEqual(envelope.get("kind"), "failure")
        self.assertEqual(envelope.get("code"), FAILURE_CODE)
        self.assertIn("error", envelope)
        self.assertEqual(envelope.get("meta"), {"tool": TOOL_NAME})

    # ------------------------------------------------------------------
    # Bounded read / top issues cap
    # ------------------------------------------------------------------
    def test_bounded_counter_honors_cap(self):
        items = [str(i % 1000) for i in range(5000)]
        result = _bounded_counter(items, cap=10)
        self.assertEqual(len(result), 10)
        counts = [entry["count"] for entry in result]
        self.assertEqual(counts, sorted(counts, reverse=True))

    def test_top_issues_capped(self):
        service_stats = {
            "c.yaml": {"running": 100},
            "d.yaml": {"exited": 100, "unhealthy": 100, "restarting": 100},
        }
        top = _top_issues(service_stats)
        self.assertTrue(len(top) <= TOP_ISSUES_CAP)
        for entry in top:
            self.assertIn("item", entry)
            self.assertIn("count", entry)
            self.assertGreater(entry["count"], 0)

    # ------------------------------------------------------------------
    # Status token extraction
    # ------------------------------------------------------------------
    def test_extract_status_tokens(self):
        cases = {
            "Up 2 minutes (healthy)": "healthy",
            "Up 5 minutes (unhealthy)": "unhealthy",
            "Exited (137) 3 seconds ago": "exited",
            "Restarting (1) 2 seconds ago": "restarting",
            "Running": "running",
        }
        for raw, expected in cases.items():
            self.assertEqual(_extract_status_token(raw), expected)

    # ------------------------------------------------------------------
    # Open-file / error tolerance from compose ps
    # ------------------------------------------------------------------
    @mock.patch("tools.docker_health_tool._run")
    @mock.patch("tools.docker_health_tool._probe_daemon", return_value=True)
    @mock.patch("tools.docker_health_tool._discover_compose_files")
    def test_service_command_error_counts_as_error(self, mock_discover, _probe, mock_run):
        mock_discover.return_value = ["compose.yaml"]
        completion = mock.Mock(returncode=1, stdout="", stderr="boom")
        mock_run.return_value = completion
        raw = _service_status_by_compose_file(["compose.yaml"])
        self.assertEqual(raw["compose.yaml"], {"error": 1})

    # ------------------------------------------------------------------
    # Health summary aggregation
    # ------------------------------------------------------------------
    def test_container_health_summary_aggregation(self):
        raw = {
            "a.yaml": {"running": 2, "exited": 1},
            "b.yaml": {"healthy": 2, "unhealthy": 1},
            "c.yaml": {"error": 1},
        }
        aggregated = _container_health_summary(raw)
        self.assertEqual(aggregated["running"], 2)
        self.assertEqual(aggregated["stopped"], 1)
        self.assertEqual(aggregated["healthy"], 2)
        self.assertEqual(aggregated["unhealthy"], 1)
        self.assertEqual(aggregated["error_services"], 1)
        self.assertEqual(aggregated["total"], 7)


if __name__ == "__main__":  # pragma: no cover - direct invocation helper
    unittest.main()
