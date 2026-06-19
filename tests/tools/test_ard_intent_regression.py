from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "ard_intent_regression.py"
spec = importlib.util.spec_from_file_location("ard_intent_regression", SCRIPT)
assert spec is not None and spec.loader is not None
ard_intent_regression = importlib.util.module_from_spec(spec)
sys.modules["ard_intent_regression"] = ard_intent_regression
spec.loader.exec_module(ard_intent_regression)


def test_evaluate_scenarios_passes_when_expected_identifier_found() -> None:
    scenarios = [
        {
            "id": "yt",
            "query": "summarize youtube transcript",
            "mode": "local",
            "expect": {"identifier_contains": "youtube-content"},
        }
    ]

    def runner(scenario):
        return [{"identifier": "urn:ai:test:skill:youtube-content", "displayName": "youtube-content", "score": 100}]

    report = ard_intent_regression.evaluate_scenarios(scenarios, runner=runner)
    assert report["ok"] is True
    assert report["summary"]["passed"] == 1
    assert report["results"][0]["status"] == "passed"


def test_evaluate_scenarios_fails_with_evidence_when_no_match() -> None:
    scenarios = [
        {"id": "missing", "query": "browser qa", "mode": "local", "expect": {"name_contains": "dogfood"}}
    ]
    report = ard_intent_regression.evaluate_scenarios(scenarios, runner=lambda _s: [])
    assert report["ok"] is False
    assert report["summary"]["failed"] == 1
    assert report["results"][0]["top_results"] == []


def test_default_scenarios_are_non_empty_and_have_expectations() -> None:
    scenarios = ard_intent_regression.default_scenarios()
    assert len(scenarios) >= 4
    assert all(s.get("query") and s.get("expect") for s in scenarios)


def test_main_writes_json_report(tmp_path: Path, capsys) -> None:
    scenario_file = tmp_path / "scenarios.json"
    report_file = tmp_path / "report.json"
    scenario_file.write_text(json.dumps([
        {"id": "yt", "query": "youtube", "mode": "local", "expect": {"identifier_contains": "youtube"}}
    ]))
    rc = ard_intent_regression.main(["--scenarios", str(scenario_file), "--output", str(report_file), "--json"])
    assert rc in {0, 1}
    assert report_file.exists()
    out = json.loads(capsys.readouterr().out)
    assert "summary" in out
