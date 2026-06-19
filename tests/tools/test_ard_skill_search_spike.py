from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "ard_skill_search_spike.py"
spec = importlib.util.spec_from_file_location("ard_skill_search_spike", SCRIPT)
assert spec is not None and spec.loader is not None
ard_skill_search_spike = importlib.util.module_from_spec(spec)
sys.modules["ard_skill_search_spike"] = ard_skill_search_spike
spec.loader.exec_module(ard_skill_search_spike)


def test_compare_queries_reports_baseline_when_external_unavailable() -> None:
    def baseline(query: str, limit: int):
        return [{"identifier": f"urn:ai:test:skill:{query}", "displayName": query}]

    report = ard_skill_search_spike.compare_queries(
        ["youtube transcript"],
        baseline_runner=baseline,
        external_runner=None,
        limit=3,
    )
    assert report["ok"] is True
    assert report["external_available"] is False
    assert report["queries"][0]["baseline"][0]["displayName"] == "youtube transcript"


def test_compare_queries_includes_external_when_runner_is_present() -> None:
    report = ard_skill_search_spike.compare_queries(
        ["browser qa"],
        baseline_runner=lambda _q, _l: [{"identifier": "urn:ai:test:skill:dogfood"}],
        external_runner=lambda _q, _l: [{"identifier": "external:dogfood"}],
        limit=5,
    )
    assert report["external_available"] is True
    assert report["queries"][0]["external"][0]["identifier"] == "external:dogfood"


def test_external_skill_search_supports_npm_cli_shape(monkeypatch) -> None:
    calls = []

    class Proc:
        returncode = 0
        stdout = json.dumps({
            "local": [{"name": "youtube-content", "description": "YouTube transcripts"}],
            "remote": [{"name": "remote-skill", "description": "Remote skill"}],
        })
        stderr = ""

    def fake_run(argv, **_kwargs):
        calls.append(argv)
        return Proc()

    monkeypatch.setattr(ard_skill_search_spike.subprocess, "run", fake_run)

    results = ard_skill_search_spike.external_skill_search("youtube transcript", 2, command="skill-search")

    assert calls[0] == ["skill-search", "youtube transcript", "--limit", "2", "--json"]
    assert [r["name"] for r in results] == ["youtube-content", "remote-skill"]


def test_find_skill_search_command_uses_env_override(monkeypatch) -> None:
    monkeypatch.setenv("ARD_SKILL_SEARCH_COMMAND", "npx -y skill-search-cli@latest")

    assert ard_skill_search_spike.find_skill_search_command() == ["npx", "-y", "skill-search-cli@latest"]


def test_external_skill_search_accepts_command_argv(monkeypatch) -> None:
    calls = []

    class Proc:
        returncode = 0
        stdout = json.dumps({"local": [{"name": "browser-qa"}], "remote": []})
        stderr = ""

    def fake_run(argv, **_kwargs):
        calls.append(argv)
        return Proc()

    monkeypatch.setattr(ard_skill_search_spike.subprocess, "run", fake_run)

    ard_skill_search_spike.external_skill_search(
        "browser qa", 1, command=["npx", "-y", "skill-search-cli@latest"]
    )

    assert calls[0] == [
        "npx",
        "-y",
        "skill-search-cli@latest",
        "browser qa",
        "--limit",
        "1",
        "--json",
    ]


def test_main_writes_report_even_without_external_cli(tmp_path: Path, monkeypatch, capsys) -> None:
    report_path = tmp_path / "report.json"
    monkeypatch.setattr(ard_skill_search_spike, "find_skill_search_command", lambda: None)
    rc = ard_skill_search_spike.main(["--query", "youtube transcript", "--output", str(report_path), "--json"])
    assert rc == 0
    report = json.loads(report_path.read_text())
    assert report["external_available"] is False
    assert report["queries"]
    assert json.loads(capsys.readouterr().out)["schema"] == "hermes.ard.skill-search-spike.v1"
