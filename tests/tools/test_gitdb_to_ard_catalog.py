from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "gitdb_to_ard_catalog.py"
spec = importlib.util.spec_from_file_location("gitdb_to_ard_catalog", SCRIPT)
assert spec is not None and spec.loader is not None
gitdb_to_ard_catalog = importlib.util.module_from_spec(spec)
sys.modules["gitdb_to_ard_catalog"] = gitdb_to_ard_catalog
spec.loader.exec_module(gitdb_to_ard_catalog)


def _item(bucket: str = "adopt_now") -> dict:
    return {
        "full_name": "tophant-ai/promptbeat",
        "url": "https://github.com/tophant-ai/promptbeat",
        "description": "Break your AI before they do.",
        "language": "MDX",
        "stars": 884,
        "forks": 1,
        "topics": ["agents", "ai-security"],
        "license": "apache-2.0",
        "focus_score": 4.0,
        "focus_terms": ["agent", "security"],
        "triage_bucket": bucket,
        "triage_reason": "high-signal-upgrade-candidate",
        "triage_signals": ["focus-score>=4"],
        "selected": True,
        "last_push": "2026-06-18T06:50:38+00:00",
    }


def test_gitdb_item_to_ard_entry_maps_candidate_metadata() -> None:
    entry = gitdb_to_ard_catalog.gitdb_item_to_ard_entry(_item())
    assert entry["identifier"] == "urn:ai:gitdb.local:tool-candidate:tophant-ai:promptbeat"
    assert entry["displayName"] == "tophant-ai/promptbeat"
    assert entry["type"] == "application/vnd.hermes.tool-candidate+json"
    assert entry["url"] == "https://github.com/tophant-ai/promptbeat"
    assert entry["metadata"]["gitdb"]["triage_bucket"] == "adopt_now"
    assert entry["metadata"]["gitdb"]["focus_score"] == 4.0
    assert "adopt_now" in entry["tags"]


def test_build_catalog_filters_bucket_and_is_public_safe() -> None:
    data = {"items": [_item("adopt_now"), _item("ignore")]}
    catalog = gitdb_to_ard_catalog.build_catalog(data, buckets={"adopt_now"})
    assert catalog["specVersion"] == "1.0"
    assert len(catalog["entries"]) == 1
    raw = json.dumps(catalog)
    assert "/home/" not in raw
    assert "API_TOKEN" not in raw


def test_write_catalog_outputs_summary(tmp_path: Path) -> None:
    catalog = gitdb_to_ard_catalog.build_catalog({"items": [_item()]})
    out = tmp_path / "gitdb-ard.json"
    summary = gitdb_to_ard_catalog.write_catalog(catalog, out)
    assert summary["entries"] == 1
    assert json.loads(out.read_text())["entries"][0]["displayName"] == "tophant-ai/promptbeat"
