"""Tests for read-only self-improvement memory/knowledge audit."""

from __future__ import annotations

import json
from pathlib import Path

from tools.self_improvement_audit import (
    HOT_LINE_CAP,
    build_self_improvement_audit,
    format_audit_report,
    reconcile_knowledge_memory,
)


def test_reconcile_flags_knowledge_only_anchor():
    hints = reconcile_knowledge_memory(
        hot_text="unrelated content",
        knowledge_texts=["architecture freeze is active"],
    )
    by_anchor = {h.anchor: h for h in hints}
    assert "architecture freeze" in by_anchor
    assert by_anchor["architecture freeze"].in_knowledge is True
    assert by_anchor["architecture freeze"].in_hot_memory is False
    assert "Consider" in by_anchor["architecture freeze"].action


def test_build_audit_hot_over_cap_proposal(tmp_path):
    learnings = tmp_path / ".learnings"
    learnings.mkdir()
    hot = learnings / "memory.md"
    hot.write_text("\n".join(f"line {i}" for i in range(HOT_LINE_CAP + 5)), encoding="utf-8")

    audit = build_self_improvement_audit(learnings_dir=learnings)

    assert audit["hot_tier"]["over_cap"] is True
    assert any("HOT memory" in p for p in audit["proposals"])
    assert audit["writes_allowed"] is False


def test_build_audit_missing_corrections_proposal(tmp_path):
    learnings = tmp_path / ".learnings"
    learnings.mkdir()
    (learnings / "memory.md").write_text(
        "Last updated: 2026-05-01\n\n# Memory\n",
        encoding="utf-8",
    )

    artifacts = tmp_path / "artifacts" / "ops"
    latest = artifacts / "knowledge_layer" / "latest_state.md"
    latest.parent.mkdir(parents=True)
    latest.write_text("generated: 2026-05-20\n\n# State\n", encoding="utf-8")

    audit = build_self_improvement_audit(
        learnings_dir=learnings,
        latest_state_path=latest,
    )

    assert any("corrections.md" in p for p in audit["proposals"])
    assert any("HOT memory last updated" in p for p in audit["proposals"])


def test_format_audit_report_renders_without_error(tmp_path):
    learnings = tmp_path / ".learnings"
    learnings.mkdir()
    (learnings / "memory.md").write_text("# Memory\n", encoding="utf-8")

    audit = build_self_improvement_audit(learnings_dir=learnings)
    text = format_audit_report(audit)

    assert "Self-improvement audit" in text
    assert "Recursive loop:" in text
    assert "FROZEN" in text


def test_audit_json_serializable(tmp_path):
    learnings = tmp_path / ".learnings"
    learnings.mkdir()
    (learnings / "memory.md").write_text("# Memory\n", encoding="utf-8")

    audit = build_self_improvement_audit(learnings_dir=learnings)
    json.dumps(audit)
