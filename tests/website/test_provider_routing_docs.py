"""Documentation checks for cost-aware model routing guidance."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROVIDER_ROUTING_DOC = REPO_ROOT / "website" / "docs" / "user-guide" / "features" / "provider-routing.md"


def test_provider_routing_docs_include_cost_aware_checklist():
    content = PROVIDER_ROUTING_DOC.read_text(encoding="utf-8")

    assert "## Cost-aware model routing checklist" in content
    assert "gpt-5.4-mini" in content
    assert "Cheap/default lane" in content
    assert "Stronger model lane" in content
    assert "Review lane" in content
    assert "fallback_providers" in content

    for signal in [
        "trace_id",
        "parent_session_id",
        "parent_task_id",
        "parent_subagent_id",
        "subagent_id",
        "model",
        "provider",
        "api_mode",
        "token breakdowns",
        "duration_seconds",
        "cost.estimated_usd",
        "cost.status",
        "cost.source",
        "status",
        "exit_reason",
        "result",
    ]:
        assert signal in content
