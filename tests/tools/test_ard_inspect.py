from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "ard_inspect.py"
spec = importlib.util.spec_from_file_location("ard_inspect", SCRIPT)
assert spec is not None and spec.loader is not None
ard_inspect = importlib.util.module_from_spec(spec)
sys.modules["ard_inspect"] = ard_inspect
spec.loader.exec_module(ard_inspect)


def test_inspect_entry_reports_source_catalog_and_risk(tmp_path: Path):
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"entries": [{
        "identifier": "urn:ai:test:tool:x",
        "displayName": "x/tool",
        "type": "application/vnd.hermes.tool-candidate+json",
        "url": "https://github.com/x/tool",
        "description": "Safe utility",
        "data": {"install": {"sideEffects": []}},
    }]}))
    report = ard_inspect.inspect_identifier("urn:ai:test:tool:x", catalogs=[catalog])
    assert report["ok"] is True
    assert report["source_catalog"] == str(catalog)
    assert report["entry"]["displayName"] == "x/tool"
    assert report["risk"]["decision"] in {"allow", "review", "deny"}
    assert "next_action" in report["risk"]


def test_inspect_entry_missing_identifier_returns_not_found(tmp_path: Path):
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"entries": []}))
    report = ard_inspect.inspect_identifier("urn:ai:test:missing", catalogs=[catalog])
    assert report["ok"] is False
    assert report["error"] == "not_found"


def test_main_writes_json_report(tmp_path: Path, capsys):
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"entries": [{
        "identifier": "urn:ai:test:skill:y",
        "displayName": "y-skill",
        "type": "application/ai-skill",
        "description": "Skill",
    }]}))
    out = tmp_path / "inspect.json"
    rc = ard_inspect.main(["urn:ai:test:skill:y", "--catalog", str(catalog), "--output", str(out), "--json"])
    assert rc == 0
    assert out.exists()
    assert json.loads(capsys.readouterr().out)["entry"]["identifier"] == "urn:ai:test:skill:y"
