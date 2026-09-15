from __future__ import annotations

import json
from pathlib import Path
import pytest

from workstation.artifacts import ArtifactStore
from workstation.terminal_summary import summarize_terminal_output
from workstation.tool_verbosity import VerbosityLevel, format_tool_output


def test_artifacts_store_and_retrieve(tmp_path):
    """Scenario 7: Full content stored as artifact; reference returned; read on demand."""
    store = ArtifactStore(root_dir=tmp_path / "artifacts")
    task_id = "task_art_1"

    large_payload = {"items": [{"id": i, "val": "x" * 50} for i in range(100)]}

    ref = store.store(task_id, "large_data.json", large_payload, schema="test_schema")
    assert ref.ref == f"artifact://tasks/{task_id}/large_data.json"
    assert ref.size_bytes > 5000
    assert Path(ref.local_path).exists()

    # Context representation contains ref and summary, not 5KB of raw JSON
    model_ref = ref.to_model_reference(signals={"complete": True})
    assert model_ref["artifact_ref"] == ref.ref
    assert model_ref["signals"]["complete"] is True
    assert "items" not in model_ref  # Raw items withheld!

    # Read back on demand
    reloaded = store.read_json(ref.ref)
    assert len(reloaded["items"]) == 100


def test_tool_verbosity_summary_vs_full(tmp_path):
    """Scenario 9: 'summary' withholds full payload, 'full' returns complete content."""
    store = ArtifactStore(root_dir=tmp_path / "artifacts")
    task_id = "task_verb_1"

    raw_page_text = "Insights Page: " + ("Visualizações: 115839, Interações: 450. " * 30)

    # Verbosity = summary: returns compact JSON with length, signals, artifact_ref
    summary_out = format_tool_output(
        {"text": raw_page_text, "views": 115839, "likes": 450},
        verbosity=VerbosityLevel.SUMMARY,
        task_id=task_id,
        name="page_snapshot",
        artifact_store=store,
        signals={"views": True, "interactions": True},
    )
    parsed = json.loads(summary_out)
    assert parsed["status"] == "ready"
    assert "artifact_ref" in parsed
    assert parsed["signals"]["views"] is True
    assert parsed["metrics"]["views"] == 115839
    # The full 1500 chars of text is NOT in the summary!
    assert "text" not in parsed

    # Verbosity = full: returns entire raw JSON
    full_out = format_tool_output(
        {"text": raw_page_text, "views": 115839},
        verbosity=VerbosityLevel.FULL,
    )
    assert "Visualizações: 115839" in full_out
    assert len(full_out) > 1000


def test_terminal_summarization(tmp_path):
    """Terminal output summarization: parses pytest results and offloads full stream."""
    store = ArtifactStore(root_dir=tmp_path / "artifacts")
    task_id = "task_term_1"

    fake_pytest_stdout = (
        "running tests...\n"
        + ("test_case_ok ... PASSED\n" * 50)
        + "FAILED tests/test_fail.py::test_bad - AssertionError\n"
        + "==================== 50 passed, 1 failed in 3.42s ====================\n"
    )

    summary = summarize_terminal_output(
        fake_pytest_stdout,
        exit_code=1,
        task_id=task_id,
        command="pytest",
        artifact_store=store,
    )

    assert summary["exit_code"] == 1
    assert summary["test_results"]["passed"] == 50
    assert summary["test_results"]["failed"] == 1
    assert "stdout_ref" in summary
    assert Path(store.resolve_ref(summary["stdout_ref"])).exists()
