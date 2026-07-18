"""Standalone tests for ``tools.platform_connectivity_tool``.

Test scope
----------
- success envelope shape when state is present
- failure enclosure when state is missing/unreadable
- ``check_fn`` returns ``False`` when no platform state exists
- issues cap/aggregation behavior
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.platform_connectivity_tool import (  # noqa: E402
    _get_issue_cap,
    _safe_str,
    check_platform_connectivity_requirements,
    platform_connectivity_tool,
    summarize_platform_connectivity,
)


class TestFailureEnvelope:
    def test_missing_hermes_home_returns_failure(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "tools.platform_connectivity_tool.get_hermes_home",
            lambda: str(tmp_path),
        )
        result = platform_connectivity_tool({})
        assert result["kind"] == "failure"
        assert result["code"] == "platform_connectivity_error"

    def test_unreadable_gateway_state_returns_failure(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        state_path = hermes_home / "gateway_state.json"
        state_path.write_text("not valid json", encoding="utf-8")
        monkeypatch.setattr(
            "tools.platform_connectivity_tool.get_hermes_home",
            lambda: str(hermes_home),
        )
        result = platform_connectivity_tool({})
        assert result["kind"] == "failure"
        assert result["code"] == "platform_connectivity_error"
        assert "gateway state" in _safe_str(result.get("error")).lower()

    def test_check_fn_false_when_no_state(self, tmp_path, monkeypatch):
        empty_home = tmp_path / ".hermes"
        empty_home.mkdir()
        monkeypatch.setattr(
            "tools.platform_connectivity_tool.get_hermes_home",
            lambda: str(empty_home),
        )
        assert check_platform_connectivity_requirements() is False


class TestSuccessEnvelopeShape:
    def _write_minimal_runtime(self, home: Path) -> None:
        (home / "gateway_state.json").write_text(
            json.dumps({"platforms": {"telegram": {"state": "connected"}}}), encoding="utf-8"
        )

    def test_success_envelope_shape(self, tmp_path, monkeypatch):
        home = tmp_path / ".hermes"
        home.mkdir()
        self._write_minimal_runtime(home)
        monkeypatch.setattr(
            "tools.platform_connectivity_tool.get_hermes_home",
            lambda: str(home),
        )
        result = platform_connectivity_tool({})
        assert result["kind"] == "success"
        data = result["data"]
        assert isinstance(data["reachable"], dict)
        assert isinstance(data["issues"], list)
        assert isinstance(data["summary_counts"], dict)
        assert "missing_tools" in data["summary_counts"]
        assert "malformed_configs" in data["summary_counts"]
        assert "total_issues" in data["summary_counts"]
        assert data["summary_counts"]["total_issues"] == (
            data["summary_counts"]["missing_tools"] + data["summary_counts"]["malformed_configs"]
        )

    def test_platform_values_are_bool(self, tmp_path, monkeypatch):
        home = tmp_path / ".hermes"
        home.mkdir()
        self._write_minimal_runtime(home)
        monkeypatch.setattr(
            "tools.platform_connectivity_tool.get_hermes_home",
            lambda: str(home),
        )
        result = platform_connectivity_tool({})
        data = result["data"]
        for value in data["reachable"].values():
            assert isinstance(value, bool)

    def test_connected_state_reflected_in_reachable(self, tmp_path, monkeypatch):
        home = tmp_path / ".hermes"
        home.mkdir()
        (home / "gateway_state.json").write_text(
            json.dumps(
                {
                    "platforms": {
                        "telegram": {"state": "connected", "error_code": None},
                        "photon": {"state": "disconnected", "error_code": None},
                    }
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(
            "tools.platform_connectivity_tool.get_hermes_home",
            lambda: str(home),
        )
        result = platform_connectivity_tool({})
        assert result["kind"] == "success"
        data = result["data"]
        assert data["reachable"]["telegram"] is True
        assert data["reachable"]["photon"] is False
        assert any("telegram error code" in _safe_str(note).lower() for note in data["issues"]) is False


class TestIssuesCap:
    def test_issues_list_is_capped(self, tmp_path, monkeypatch):
        home = tmp_path / ".hermes"
        home.mkdir()
        (home / "gateway_state.json").write_text(
            json.dumps(
                {
                    "platforms": {
                        name: {"state": "disconnected", "error_code": f"err_{name}"}
                        for name in [f"platform_{i}" for i in range(100)]
                    }
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(
            "tools.platform_connectivity_tool.get_hermes_home",
            lambda: str(home),
        )
        result = platform_connectivity_tool({})
        issues = result["data"]["issues"]
        assert len(issues) <= _get_issue_cap()

    def test_cap_constrains_issues_not_summaries(self, tmp_path, monkeypatch):
        home = tmp_path / ".hermes"
        home.mkdir()
        (home / "gateway_state.json").write_text(
            json.dumps(
                {
                    "platforms": {
                        name: {"state": "disconnected", "error_code": f"err_{name}"}
                        for name in [f"platform_{i}" for i in range(100)]
                    }
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(
            "tools.platform_connectivity_tool.get_hermes_home",
            lambda: str(home),
        )
        result = platform_connectivity_tool({})
        summary = result["data"]["summary_counts"]
        assert summary["total_issues"] >= summary["missing_tools"] + summary["malformed_configs"]
        assert len(result["data"]["issues"]) <= _get_issue_cap()
