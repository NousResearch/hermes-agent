"""Focused tests for Skill Governance proposal storage and Curator dry-run import."""

import json
from pathlib import Path

import pytest

from tools.skill_governance_proposals import (
    MAX_ARTIFACT_TEXT_CHARS,
    SCHEMA_VERSION,
    create_or_update_skill_governance_proposal,
    get_skill_governance_proposal,
    import_curator_dry_run,
    list_skill_governance_proposals,
    record_skill_governance_decision,
)


def _set_hermes_home(monkeypatch, tmp_path: Path) -> Path:
    hermes_home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    return hermes_home


def _sample_report() -> str:
    return """# Curator run — 2026-05-06T07:17:37.364411+00:00

No mutating actions were taken.

### 1. Create new umbrella: `hermes-dashboard-development`

I would create a broad class-level Hermes dashboard umbrella skill covering dashboard and HUDUI work.

I would:
- create `hermes-dashboard-development`
- archive the absorbed narrow skills with `absorbed_into=hermes-dashboard-development`

This would archive 10 skills.

### 2. Create new umbrella: `llm-training-workflows`

I would create a broad class-level LLM training umbrella skill covering PEFT / LoRA / QLoRA fine-tuning.

This would archive 4 skills.

## Structured summary (required)
```yaml
consolidations:
  - from: hermes-dashboard-cron-operations
    into: hermes-dashboard-development
    reason: Cron operations is one dashboard feature recipe.
  - from: hermes-dashboard-feature-fixture-layer
    into: hermes-dashboard-development
    reason: Fixture-first backend slices belong under dashboard development.
  - from: hermes-hudui-frontend-workflow
    into: hermes-dashboard-development
    reason: HUDUI frontend work is adjacent dashboard workflow.
  - from: grpo-rl-training
    into: llm-training-workflows
    reason: GRPO/RL post-training is one technique.
  - from: peft-fine-tuning
    into: llm-training-workflows
    reason: PEFT is one training strategy.
```
"""


def _write_sample_curator_run(tmp_path: Path) -> tuple[Path, Path]:
    run_dir = tmp_path / "curator" / "20260506-071737"
    run_dir.mkdir(parents=True)
    report_path = run_dir / "REPORT.md"
    run_json_path = run_dir / "run.json"
    report_path.write_text(_sample_report(), encoding="utf-8")
    run_json_path.write_text(
        json.dumps({"started_at": "2026-05-06T07:17:37.364411+00:00", "model": "gpt-5.5"}),
        encoding="utf-8",
    )
    return report_path, run_json_path


def test_create_proposal_uses_profile_safe_per_proposal_json_and_artifacts(monkeypatch, tmp_path):
    hermes_home = _set_hermes_home(monkeypatch, tmp_path)

    proposal = create_or_update_skill_governance_proposal(
        {
            "proposal_id": "curator-20260506-dashboard",
            "source": "curator_dry_run",
            "source_run_id": "20260506-071737",
            "action": "consolidate",
            "title": "Consolidate dashboard skills",
            "pm_summary": "Create one dashboard umbrella skill.",
            "target_skills": ["hermes-dashboard-cron-operations", "hermes-hudui-frontend-workflow"],
            "risk_level": "medium",
            "evidence": ["Curator dry-run identified overlap."],
            "artifact_texts": {"curator_excerpt.md": "raw curator excerpt"},
        }
    )

    assert proposal["schema_version"] == SCHEMA_VERSION
    assert proposal["decision_status"] == "pending"
    assert proposal["codex_review_status"] == "not_requested"
    assert proposal["created_at"]
    assert proposal["updated_at"]

    proposal_dir = hermes_home / "skill-governance" / "proposals" / "curator-20260506-dashboard"
    assert proposal_dir.is_dir()
    assert json.loads((proposal_dir / "proposal.json").read_text(encoding="utf-8"))["proposal_id"] == proposal["proposal_id"]
    artifact_path = Path(proposal["artifact_paths"]["curator_excerpt.md"])
    assert artifact_path == proposal_dir / "artifacts" / "curator_excerpt.md"
    assert artifact_path.read_text(encoding="utf-8") == "raw curator excerpt"


def test_create_or_update_is_idempotent_and_preserves_existing_decision(monkeypatch, tmp_path):
    _set_hermes_home(monkeypatch, tmp_path)
    base = {
        "proposal_id": "curator-20260506-dashboard",
        "source": "curator_dry_run",
        "source_run_id": "20260506-071737",
        "action": "consolidate",
        "title": "Consolidate dashboard skills",
        "pm_summary": "Initial summary.",
        "target_skills": ["hermes-dashboard-cron-operations"],
        "risk_level": "medium",
    }
    first = create_or_update_skill_governance_proposal(base)
    decided = record_skill_governance_decision(
        first["proposal_id"],
        "deferred",
        note="Review later.",
        decided_by="pm-user",
    )
    previous_decision_at = decided["decision_at"]
    previous_history = list(decided["decision_history"])

    second = create_or_update_skill_governance_proposal({**base, "pm_summary": "Updated summary."})

    assert second["proposal_id"] == first["proposal_id"]
    assert second["created_at"] == first["created_at"]
    assert second["pm_summary"] == "Updated summary."
    assert second["decision_status"] == "deferred"
    assert second["decision_by"] == "pm-user"
    assert second["decision_at"] == previous_decision_at
    assert second["decision_note"] == decided["decision_note"]
    assert second["decision_history"] == previous_history
    assert [p["proposal_id"] for p in list_skill_governance_proposals()] == [first["proposal_id"]]


def test_record_decision_enforces_mvp_state_machine(monkeypatch, tmp_path):
    _set_hermes_home(monkeypatch, tmp_path)
    proposal = create_or_update_skill_governance_proposal(
        {
            "proposal_id": "curator-20260506-dashboard",
            "source": "curator_dry_run",
            "source_run_id": "20260506-071737",
            "action": "consolidate",
            "title": "Consolidate dashboard skills",
            "pm_summary": "Create one dashboard umbrella skill.",
            "target_skills": ["hermes-dashboard-cron-operations"],
            "risk_level": "medium",
        }
    )

    updated = record_skill_governance_decision(
        proposal["proposal_id"],
        "approved",
        note="Looks like a good first test.",
        decided_by="test-user",
    )
    assert updated["decision_status"] == "approved"
    assert updated["decision_by"] == "test-user"
    assert updated["decision_note"] == "Looks like a good first test."
    assert updated["decision_at"]

    with pytest.raises(ValueError):
        record_skill_governance_decision(proposal["proposal_id"], "applied")

    with pytest.raises(ValueError):
        record_skill_governance_decision(proposal["proposal_id"], "unknown")


def test_malformed_stored_proposals_with_illegal_states_are_not_returned(monkeypatch, tmp_path):
    hermes_home = _set_hermes_home(monkeypatch, tmp_path)
    proposal_dir = hermes_home / "skill-governance" / "proposals" / "bad-state"
    proposal_dir.mkdir(parents=True)
    (proposal_dir / "proposal.json").write_text(
        json.dumps({"proposal_id": "bad-state", "decision_status": "applied"}),
        encoding="utf-8",
    )

    assert get_skill_governance_proposal("bad-state") is None
    assert list_skill_governance_proposals() == []


def test_manual_proposal_id_is_normalized_for_storage_and_api(monkeypatch, tmp_path):
    hermes_home = _set_hermes_home(monkeypatch, tmp_path)

    proposal = create_or_update_skill_governance_proposal(
        {
            "proposal_id": "../unsafe proposal/id",
            "source": "manual",
            "action": "review",
            "title": "Unsafe id should be normalized",
            "pm_summary": "Avoid awkward path-shaped ids.",
            "risk_level": "low",
        }
    )

    assert proposal["proposal_id"] == "unsafe-proposal-id"
    assert (hermes_home / "skill-governance" / "proposals" / "unsafe-proposal-id" / "proposal.json").exists()
    assert get_skill_governance_proposal("../unsafe proposal/id") is None
    assert get_skill_governance_proposal("unsafe-proposal-id") is not None


def test_artifact_names_are_sanitized_and_large_artifacts_are_truncated(monkeypatch, tmp_path):
    _set_hermes_home(monkeypatch, tmp_path)
    large_text = "x" * (MAX_ARTIFACT_TEXT_CHARS + 100)

    proposal = create_or_update_skill_governance_proposal(
        {
            "proposal_id": "artifact-safety",
            "source": "manual",
            "action": "review",
            "title": "Artifact safety",
            "pm_summary": "Artifacts should stay inside the proposal directory.",
            "artifact_texts": {"../unsafe/huge.md": large_text},
        }
    )

    artifact_path = Path(proposal["artifact_paths"]["huge.md"])
    assert artifact_path.name == "huge.md"
    assert artifact_path.parent.name == "artifacts"
    stored = artifact_path.read_text(encoding="utf-8")
    assert stored.startswith("x" * 100)
    assert len(stored) < len(large_text)
    assert "artifact truncated" in stored


def test_detail_view_reads_safe_artifact_text_and_exposes_allowed_decisions(monkeypatch, tmp_path):
    _set_hermes_home(monkeypatch, tmp_path)
    proposal = create_or_update_skill_governance_proposal(
        {
            "proposal_id": "artifact-detail",
            "source": "curator_dry_run",
            "action": "consolidate",
            "title": "Artifact detail",
            "pm_summary": "Show the dry-run excerpt in the dashboard detail view.",
            "artifact_texts": {"curator_report_excerpt.md": "dry-run excerpt"},
        }
    )

    listed = list_skill_governance_proposals()[0]
    detail = get_skill_governance_proposal(proposal["proposal_id"], include_artifacts=True)

    assert "artifact_texts" not in listed
    assert detail["artifact_texts"] == {"curator_report_excerpt.md": "dry-run excerpt"}
    assert detail["diff_text"] is None
    assert "approved" in detail["allowed_decision_statuses"]


def test_bad_test_target_detail_does_not_offer_approve(monkeypatch, tmp_path):
    _set_hermes_home(monkeypatch, tmp_path)
    proposal = create_or_update_skill_governance_proposal(
        {
            "proposal_id": "bad-target",
            "source": "curator_dry_run",
            "action": "consolidate",
            "title": "Bad validation target",
            "pm_summary": "This proposal should not be approved from the bad-test-target state.",
            "decision_status": "bad_test_target",
            "decision_by": "hermes-policy",
            "decision_note": "Not a useful first Curator validation target.",
        }
    )

    detail = get_skill_governance_proposal(proposal["proposal_id"])

    assert "approved" not in detail["allowed_decision_statuses"]
    assert set(detail["allowed_decision_statuses"]) == {"pending", "deferred", "bad_test_target", "rejected"}


def test_import_curator_dry_run_creates_idempotent_dashboard_and_bad_test_target_proposals(monkeypatch, tmp_path):
    _set_hermes_home(monkeypatch, tmp_path)
    report_path, run_json_path = _write_sample_curator_run(tmp_path)

    imported = import_curator_dry_run(report_path, run_json_path=run_json_path)
    imported_again = import_curator_dry_run(report_path, run_json_path=run_json_path)

    assert [p["proposal_id"] for p in imported_again] == [p["proposal_id"] for p in imported]
    assert len(list_skill_governance_proposals()) == 2

    by_id = {proposal["proposal_id"]: proposal for proposal in imported}
    dashboard = by_id["curator-20260506-071737-hermes-dashboard-development"]
    training = by_id["curator-20260506-071737-llm-training-workflows"]

    assert dashboard["decision_status"] == "pending"
    assert dashboard["recommended_decision"] == "review_first"
    assert "hermes-dashboard-cron-operations" in dashboard["target_skills"]
    assert "hermes-hudui-frontend-workflow" in dashboard["target_skills"]
    assert dashboard["artifact_paths"]["curator_report_excerpt.md"]

    assert training["decision_status"] == "bad_test_target"
    assert training["recommended_decision"] == "defer"
    assert "llm-training-workflows" in training["title"]
    assert "user decision" in training["decision_note"].lower()

    detail = get_skill_governance_proposal(dashboard["proposal_id"])
    assert detail is not None
    assert detail["proposal_id"] == dashboard["proposal_id"]
