from __future__ import annotations

import json
from uuid import uuid4
import pytest

from workstation.artifacts import ArtifactStore
from workstation.durable_tasks import DurableTaskStore
from tools.browser_tool import _process_extracted_items_durably


def test_browser_extract_items_durable_processing(tmp_path, monkeypatch):
    """Scenario: browser_extract_items processes items into ArtifactStore, creates WorkPlan, and returns compact summary."""
    task_id = f"test_extract_{uuid4().hex[:6]}"
    artifact_dir = tmp_path / "artifacts"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    # Mock raw extraction data with 10 items (9 valid, 1 anomaly with empty title)
    raw_items = [
        {"title": f"Product {i}", "url": f"https://example.com/p/{i}", "price": f"R$ {i * 10},00"}
        for i in range(1, 10)
    ]
    # Injected anomaly: item with empty title
    raw_items.append({"title": "", "url": "https://example.com/p/10", "price": "R$ 100,00"})

    raw_response = {
        "success": True,
        "runtime": "electron-chromium",
        "count": len(raw_items),
        "selector_used": "div.product-card",
        "url": "https://example.com/shop",
        "title": "Shop Page",
        "items": raw_items,
    }

    result_str = _process_extracted_items_durably(
        raw_result=raw_response,
        args={"limit": 20, "output_artifact": True},
        kw={"task_id": task_id},
    )

    summary = json.loads(result_str)
    assert summary["success"] is True
    assert summary["total_extracted"] == 10
    assert summary["valid_count"] == 9
    assert summary["anomalies_count"] == 1
    assert summary["selector_used"] == "div.product-card"

    # Verify that raw items array is NOT in the summary returned to LLM
    assert "items" not in summary
    assert len(summary["sample_preview"]) == 2

    # Verify that artifact was stored with SHA-256 reference
    artifact_ref = summary["artifact_ref"]
    assert artifact_ref.startswith("artifact://")
    store = ArtifactStore()
    stored_data = store.read_json(artifact_ref)
    assert stored_data["count"] == 10
    assert len(stored_data["items"]) == 10

    # Verify durable WorkPlan was created in kanban_db
    with DurableTaskStore() as durable_store:
        plan = durable_store.get_plan(task_id)
        assert plan is not None
        assert plan.total_items == 10
        remaining_items = durable_store.resume_plan(task_id)
        assert len(remaining_items) == 1
        assert remaining_items[0].status.value == "failed"


def test_browser_extract_items_small_raw_mode():
    """Scenario: small extractions (<= 5 items) with output_artifact=False return raw json directly."""
    raw_items = [{"title": "Item 1", "url": "https://example.com/1"}]
    raw_response = {
        "success": True,
        "count": 1,
        "selector_used": "article",
        "items": raw_items,
    }

    result_str = _process_extracted_items_durably(
        raw_result=raw_response,
        args={"limit": 5, "output_artifact": False},
        kw={"task_id": "test_small"},
    )

    data = json.loads(result_str)
    assert "items" in data
    assert len(data["items"]) == 1
