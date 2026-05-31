import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tools import self_improvement_tool
from tools.registry import registry
from toolsets import resolve_toolset


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _seed_ontology_repo(tmp_path: Path, *, generated_at: str, status: str = "fresh") -> Path:
    repo = tmp_path / "ontology"
    _write_json(
        repo / "evolution" / "metrics.json",
        {
            "generated_at": generated_at,
            "status": status,
            "platform": {"total_cqs": 10, "total_answered": 10},
        },
    )
    _write_json(
        repo / "evolution" / "delta_report.json",
        {
            "generated_at": generated_at,
            "previous_metrics_generated_at": generated_at,
            "status": status,
            "current": {"platform": {"total_cqs": 10, "total_answered": 10}},
        },
    )
    (repo / "evolution").mkdir(parents=True, exist_ok=True)
    (repo / "evolution" / "daily_report.md").write_text(
        "\n".join(
            [
                "# Ontology Evolution Daily Report",
                "",
                f"Generated at: `{generated_at}`",
                "",
                f"Status: {status}",
            ]
        ),
        encoding="utf-8",
    )
    return repo


def _operator_value_entries(recent: str, *, aligned: int = 4, decision_only: int = 2) -> list[dict]:
    entries = []
    for idx in range(aligned):
        entries.append(
            {
                "id": f"aligned-{idx}",
                "occurredAt": recent,
                "summary": f"Implemented operator-value path {idx}.",
                "operatorDecisionSupport": "Operator can compare the verified change and choose the next issue.",
                "changedFiles": ["tools/self_improvement_tool.py"],
                "tests": ["pytest tests/tools/test_self_improvement_tool.py passed"],
            }
        )
    for idx in range(decision_only):
        entries.append(
            {
                "id": f"decision-only-{idx}",
                "occurredAt": recent,
                "summary": f"Prepared operator decision support note {idx}.",
                "operatorDecisionSupport": "Operator has a blocker and recommended next decision.",
            }
        )
    return entries


def test_all_fresh_cross_source_skew_keeps_reliability_gate_healthy(tmp_path):
    now = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    recent = now.isoformat()
    older_but_fresh = (now - timedelta(hours=20)).isoformat()

    journal_path = tmp_path / "journal.json"
    codex_path = tmp_path / "runs.json"
    ctx_path = tmp_path / "session_bindings.json"
    ontology_root = _seed_ontology_repo(tmp_path, generated_at=recent)

    _write_json(journal_path, {"entries": [{"occurredAt": older_but_fresh}]})
    _write_json(
        codex_path,
        {"runs": {"codex_1": {"run_id": "codex_1", "status": "completed", "completed_at": recent}}},
    )
    _write_json(
        ctx_path,
        {"sessions": {"ctx_1": {"session_id": "ctx_1", "active": False, "updated_at": recent}}},
    )

    gate = self_improvement_tool.evaluate_self_improvement_evidence(
        journal_path=journal_path,
        codex_runs_path=codex_path,
        ctx_bindings_path=ctx_path,
        ontology_root=ontology_root,
        now=now,
    )

    ctx_source = gate["sources"]["ctx_bindings"]
    assert gate["status"] == "healthy"
    assert {entry["status"] for entry in gate["sources"].values()} == {"fresh", "inactive"}
    assert ctx_source["status"] == "inactive"
    assert ctx_source["freshness_required"] is False
    assert ctx_source["active_count"] == 0
    assert ctx_source["detail"] == "No active ctx bindings; retired binding timestamps are informational."
    assert gate["warnings"] == []
    assert gate["contradictions"] == []
    assert gate["freshness_spread_hours"] == 20.0

    benchmark = self_improvement_tool.evaluate_self_improvement_benchmark(
        journal_path=journal_path,
        codex_runs_path=codex_path,
        ctx_bindings_path=ctx_path,
        ontology_root=ontology_root,
        history_path=tmp_path / "history.json",
        now=now,
        persist=False,
    )
    reliability_gate = benchmark["checks"]["reliability_gate"]
    assert reliability_gate["score"] == 1.0
    assert reliability_gate["status"] == "pass"
    assert reliability_gate["detail"] == "Reliability floor is healthy."
    assert reliability_gate["metrics"]["warning_count"] == 0
    anti_make_work = benchmark["checks"]["anti_make_work_check"]
    assert anti_make_work["score"] == 1.0
    assert anti_make_work["status"] == "pass"
    assert anti_make_work["metrics"]["assessed_work_item_count"] == 0
    assert benchmark["critical_failures"] == []


def test_ontology_scan_ignores_git_worktree_runtime_artifacts(tmp_path):
    now = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    recent = now.isoformat()

    journal_path = tmp_path / "journal.json"
    codex_path = tmp_path / "runs.json"
    ctx_path = tmp_path / "session_bindings.json"
    ontology_root = _seed_ontology_repo(tmp_path, generated_at=recent)

    _write_json(journal_path, {"entries": [{"occurredAt": recent}]})
    _write_json(
        codex_path,
        {"runs": {"codex_1": {"run_id": "codex_1", "status": "completed", "completed_at": recent}}},
    )
    _write_json(
        ctx_path,
        {"sessions": {"ctx_1": {"session_id": "ctx_1", "active": False, "updated_at": recent}}},
    )
    _write_json(
        ontology_root / ".git" / "worktrees" / "old" / "hermes-codex" / "latest.json",
        {"generated_at": recent, "status": "failed"},
    )

    gate = self_improvement_tool.evaluate_self_improvement_evidence(
        journal_path=journal_path,
        codex_runs_path=codex_path,
        ctx_bindings_path=ctx_path,
        ontology_root=ontology_root,
        now=now,
    )

    assert gate["status"] == "healthy"
    assert gate["ontology"]["status"] == "fresh"
    assert gate["ontology_alerts"] == []
    assert gate["warnings"] == []
    assert gate["contradictions"] == []


def test_required_ontology_artifacts_drive_staleness_when_unrelated_future_timestamp_exists(tmp_path):
    now = datetime(2026, 5, 22, 7, 35, 41, tzinfo=timezone.utc)
    recent = now.isoformat()
    stale = (now - timedelta(hours=168)).isoformat()

    journal_path = tmp_path / "journal.json"
    codex_path = tmp_path / "runs.json"
    ctx_path = tmp_path / "session_bindings.json"
    ontology_root = _seed_ontology_repo(tmp_path, generated_at=stale)

    _write_json(journal_path, {"entries": [{"occurredAt": recent}]})
    _write_json(
        codex_path,
        {"runs": {"codex_1": {"run_id": "codex_1", "status": "completed", "completed_at": recent}}},
    )
    _write_json(
        ctx_path,
        {"sessions": {"ctx_1": {"session_id": "ctx_1", "active": False, "updated_at": recent}}},
    )
    (ontology_root / "docs").mkdir(parents=True, exist_ok=True)
    (ontology_root / "docs" / "future-cutoff.md").write_text(
        "freshness_cutoff: 2027-05-16T17:55:00Z\n",
        encoding="utf-8",
    )

    gate = self_improvement_tool.evaluate_self_improvement_evidence(
        journal_path=journal_path,
        codex_runs_path=codex_path,
        ctx_bindings_path=ctx_path,
        ontology_root=ontology_root,
        now=now,
    )

    required = gate["ontology"]["required_artifacts"]
    assert gate["status"] == "degraded"
    assert gate["ontology"]["status"] == "stale"
    assert gate["ontology"]["age_hours"] == 168.0
    assert gate["ontology"]["ignored_future_timestamp_count"] == 1
    assert {entry["status"] for entry in required.values()} == {"stale"}
    assert gate["ontology"]["external_repair"]["required"] is True
    assert "ontology_metrics stale (168.0h)" in gate["reasons"]
    assert "ontology_intelligence evidence stale" in gate["warnings"]

    benchmark = self_improvement_tool.evaluate_self_improvement_benchmark(
        journal_path=journal_path,
        codex_runs_path=codex_path,
        ctx_bindings_path=ctx_path,
        ontology_root=ontology_root,
        history_path=tmp_path / "history.json",
        now=now,
        persist=False,
    )
    reliability_gate = benchmark["checks"]["reliability_gate"]
    assert reliability_gate["status"] == "fail"
    assert "reliability_gate" in benchmark["critical_failures"]


def test_stale_retired_ctx_bindings_are_inactive_not_degraded(tmp_path):
    now = datetime(2026, 5, 22, 15, 0, 0, tzinfo=timezone.utc)
    recent = now.isoformat()
    stale_retired = (now - timedelta(hours=144)).isoformat()

    journal_path = tmp_path / "journal.json"
    codex_path = tmp_path / "runs.json"
    ctx_path = tmp_path / "session_bindings.json"
    ontology_root = _seed_ontology_repo(tmp_path, generated_at=recent)

    _write_json(journal_path, {"entries": [{"occurredAt": recent}]})
    _write_json(
        codex_path,
        {"runs": {"codex_1": {"run_id": "codex_1", "status": "completed", "completed_at": recent}}},
    )
    _write_json(
        ctx_path,
        {
            "sessions": {
                "ctx_retired": {
                    "session_id": "ctx_retired",
                    "active": False,
                    "reason": "ctx binding retired: worktree missing",
                    "updated_at": stale_retired,
                }
            }
        },
    )

    gate = self_improvement_tool.evaluate_self_improvement_evidence(
        journal_path=journal_path,
        codex_runs_path=codex_path,
        ctx_bindings_path=ctx_path,
        ontology_root=ontology_root,
        now=now,
    )

    ctx_source = gate["sources"]["ctx_bindings"]
    assert gate["status"] == "healthy"
    assert ctx_source["status"] == "inactive"
    assert ctx_source["age_hours"] == 144.0
    assert ctx_source["freshness_required"] is False
    assert ctx_source["active_count"] == 0
    assert gate["warnings"] == []
    assert gate["contradictions"] == []
    assert gate["freshness_spread_hours"] == 0.0
    assert "No active ctx bindings" in gate["provenance"]["items"][2]["notes"]

    benchmark = self_improvement_tool.evaluate_self_improvement_benchmark(
        journal_path=journal_path,
        codex_runs_path=codex_path,
        ctx_bindings_path=ctx_path,
        ontology_root=ontology_root,
        history_path=tmp_path / "history.json",
        now=now,
        persist=False,
    )
    reliability_gate = benchmark["checks"]["reliability_gate"]
    assert reliability_gate["score"] == 1.0
    assert reliability_gate["status"] == "pass"
    assert reliability_gate["metrics"]["stale_source_count"] == 0
    assert reliability_gate["metrics"]["ctx_remediation_required"] is False
    assert benchmark["critical_failures"] == []


def test_stale_retired_ctx_status_records_are_inactive_not_degraded(tmp_path):
    now = datetime(2026, 5, 22, 15, 0, 0, tzinfo=timezone.utc)
    recent = now.isoformat()
    stale_retired = (now - timedelta(hours=144)).isoformat()

    journal_path = tmp_path / "journal.json"
    codex_path = tmp_path / "runs.json"
    ctx_path = tmp_path / "session_bindings.json"
    ontology_root = _seed_ontology_repo(tmp_path, generated_at=recent)

    _write_json(journal_path, {"entries": [{"occurredAt": recent}]})
    _write_json(
        codex_path,
        {"runs": {"codex_1": {"run_id": "codex_1", "status": "completed", "completed_at": recent}}},
    )
    _write_json(
        ctx_path,
        {
            "sessions": {
                "ctx_retired_status": {
                    "session_id": "ctx_retired_status",
                    "status": "retired",
                    "reason": "ctx binding retired: stale evidence cleanup (>12h)",
                    "updated_at": stale_retired,
                }
            }
        },
    )

    benchmark = self_improvement_tool.evaluate_self_improvement_benchmark(
        journal_path=journal_path,
        codex_runs_path=codex_path,
        ctx_bindings_path=ctx_path,
        ontology_root=ontology_root,
        history_path=tmp_path / "history.json",
        now=now,
        persist=False,
    )

    ctx_source = benchmark["gate"]["sources"]["ctx_bindings"]
    reliability_gate = benchmark["checks"]["reliability_gate"]
    assert benchmark["gate"]["status"] == "healthy"
    assert ctx_source["status"] == "inactive"
    assert ctx_source["active_count"] == 0
    assert ctx_source["freshness_required"] is False
    assert reliability_gate["status"] == "pass"
    assert reliability_gate["metrics"]["stale_source_count"] == 0
    assert reliability_gate["metrics"]["ctx_remediation_required"] is False
    assert benchmark["critical_failures"] == []


def test_active_ctx_staleness_uses_active_timestamp_when_retired_record_is_fresher(tmp_path):
    now = datetime(2026, 5, 22, 15, 0, 0, tzinfo=timezone.utc)
    recent = now.isoformat()
    stale_active = (now - timedelta(hours=144)).isoformat()

    journal_path = tmp_path / "journal.json"
    codex_path = tmp_path / "runs.json"
    ctx_path = tmp_path / "session_bindings.json"
    ontology_root = _seed_ontology_repo(tmp_path, generated_at=recent)
    live_worktree = tmp_path / "live-worktree"
    live_worktree.mkdir()

    _write_json(journal_path, {"entries": [{"occurredAt": recent}]})
    _write_json(
        codex_path,
        {"runs": {"codex_1": {"run_id": "codex_1", "status": "completed", "completed_at": recent}}},
    )
    _write_json(
        ctx_path,
        {
            "sessions": {
                "ctx_active_stale": {
                    "session_id": "ctx_active_stale",
                    "active": True,
                    "updated_at": stale_active,
                    "worktree_path": str(live_worktree),
                },
                "ctx_retired_recent": {
                    "session_id": "ctx_retired_recent",
                    "active": False,
                    "updated_at": recent,
                },
            }
        },
    )

    benchmark = self_improvement_tool.evaluate_self_improvement_benchmark(
        journal_path=journal_path,
        codex_runs_path=codex_path,
        ctx_bindings_path=ctx_path,
        ontology_root=ontology_root,
        history_path=tmp_path / "history.json",
        now=now,
        persist=False,
    )

    ctx_source = benchmark["gate"]["sources"]["ctx_bindings"]
    reliability_gate = benchmark["checks"]["reliability_gate"]
    assert benchmark["gate"]["status"] == "degraded"
    assert ctx_source["status"] == "stale"
    assert ctx_source["latest_timestamp"] == stale_active
    assert ctx_source["active_latest_timestamp"] == stale_active
    assert ctx_source["latest_record_timestamp"] == recent
    assert ctx_source["freshness_required"] is True
    assert reliability_gate["status"] == "fail"
    assert reliability_gate["metrics"]["stale_source_count"] == 1
    assert benchmark["gate"]["ctx_remediation"]["stale_active_count"] == 1
    assert "reliability_gate" in benchmark["critical_failures"]


def test_active_ctx_without_timestamp_is_degraded_not_masked_by_retired_timestamp(tmp_path):
    now = datetime(2026, 5, 22, 15, 0, 0, tzinfo=timezone.utc)
    recent = now.isoformat()

    journal_path = tmp_path / "journal.json"
    codex_path = tmp_path / "runs.json"
    ctx_path = tmp_path / "session_bindings.json"
    ontology_root = _seed_ontology_repo(tmp_path, generated_at=recent)
    live_worktree = tmp_path / "live-worktree"
    live_worktree.mkdir()

    _write_json(journal_path, {"entries": [{"occurredAt": recent}]})
    _write_json(
        codex_path,
        {"runs": {"codex_1": {"run_id": "codex_1", "status": "completed", "completed_at": recent}}},
    )
    _write_json(
        ctx_path,
        {
            "sessions": {
                "ctx_active_missing_timestamp": {
                    "session_id": "ctx_active_missing_timestamp",
                    "active": True,
                    "worktree_path": str(live_worktree),
                },
                "ctx_retired_recent": {
                    "session_id": "ctx_retired_recent",
                    "active": False,
                    "updated_at": recent,
                },
            }
        },
    )

    benchmark = self_improvement_tool.evaluate_self_improvement_benchmark(
        journal_path=journal_path,
        codex_runs_path=codex_path,
        ctx_bindings_path=ctx_path,
        ontology_root=ontology_root,
        history_path=tmp_path / "history.json",
        now=now,
        persist=False,
    )

    ctx_source = benchmark["gate"]["sources"]["ctx_bindings"]
    reliability_gate = benchmark["checks"]["reliability_gate"]
    assert benchmark["gate"]["status"] == "degraded"
    assert ctx_source["status"] == "degraded"
    assert ctx_source["latest_timestamp"] is None
    assert ctx_source["active_latest_timestamp"] is None
    assert ctx_source["latest_record_timestamp"] == recent
    assert ctx_source["detail"] == "Active ctx bindings do not include freshness timestamps."
    assert benchmark["gate"]["ctx_remediation"]["required"] is True
    assert reliability_gate["status"] == "fail"
    assert reliability_gate["metrics"]["ctx_remediation_required"] is True
    assert "reliability_gate" in benchmark["critical_failures"]


def test_missing_ctx_bindings_block_issue_selection_with_repair_guidance(tmp_path):
    now = datetime(2026, 5, 22, 15, 0, 0, tzinfo=timezone.utc)
    recent = now.isoformat()

    journal_path = tmp_path / "journal.json"
    codex_path = tmp_path / "runs.json"
    ctx_path = tmp_path / "missing" / "session_bindings.json"
    ontology_root = _seed_ontology_repo(tmp_path, generated_at=recent)

    _write_json(journal_path, {"entries": [{"occurredAt": recent}]})
    _write_json(
        codex_path,
        {"runs": {"codex_1": {"run_id": "codex_1", "status": "completed", "completed_at": recent}}},
    )

    benchmark = self_improvement_tool.evaluate_self_improvement_benchmark(
        journal_path=journal_path,
        codex_runs_path=codex_path,
        ctx_bindings_path=ctx_path,
        ontology_root=ontology_root,
        history_path=tmp_path / "history.json",
        now=now,
        persist=False,
    )

    reliability_gate = benchmark["checks"]["reliability_gate"]
    ctx_remediation = benchmark["gate"]["ctx_remediation"]
    issue_selection = benchmark["issue_selection"]
    assert reliability_gate["status"] == "fail"
    assert reliability_gate["metrics"]["ctx_remediation_required"] is True
    assert ctx_remediation["required"] is True
    assert ctx_remediation["path"] == str(ctx_path)
    assert "restore or regenerate ctx session-binding evidence" in ctx_remediation["action"]
    assert benchmark["critical_failures"] == ["reliability_gate"]
    assert issue_selection["blocked_checks"] == ["reliability_gate"]
    assert issue_selection["recommended_focus"] == "self-improvement evidence freshness repair"
    assert issue_selection["remediation_actions"] == [ctx_remediation["action"]]
    assert "Repair self-improvement evidence freshness" in issue_selection["detail"]


def test_status_language_only_work_fails_anti_make_work_check(tmp_path):
    now = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    recent = now.isoformat()

    journal_path = tmp_path / "journal.json"
    codex_path = tmp_path / "runs.json"
    ctx_path = tmp_path / "session_bindings.json"
    ontology_root = _seed_ontology_repo(tmp_path, generated_at=recent)

    _write_json(
        journal_path,
        {
            "entries": [
                {
                    "id": "status-only",
                    "occurredAt": recent,
                    "summary": "Status update: HAD-1019 is in progress and actionable.",
                    "notes": "Working on the active item; next step is to keep monitoring.",
                    "linearIssues": ["HAD-1019"],
                    "commitShas": [],
                    "reposTouched": [],
                }
            ]
        },
    )
    _write_json(
        codex_path,
        {
            "runs": {
                "codex_status": {
                    "run_id": "codex_status",
                    "status": "completed",
                    "completed_at": recent,
                    "final_message": "STATUS\nIn progress. Summary captured; no blockers.",
                    "exit_code": 0,
                }
            }
        },
    )
    _write_json(
        ctx_path,
        {"sessions": {"ctx_1": {"session_id": "ctx_1", "active": False, "updated_at": recent}}},
    )

    benchmark = self_improvement_tool.evaluate_self_improvement_benchmark(
        journal_path=journal_path,
        codex_runs_path=codex_path,
        ctx_bindings_path=ctx_path,
        ontology_root=ontology_root,
        history_path=tmp_path / "history.json",
        now=now,
        persist=False,
    )

    reliability_gate = benchmark["checks"]["reliability_gate"]
    assert reliability_gate["status"] == "pass"

    anti_make_work = benchmark["checks"]["anti_make_work_check"]
    assert anti_make_work["status"] == "fail"
    assert anti_make_work["score"] == 0.0
    assert "allowed value-category evidence" in anti_make_work["detail"]
    assert "operator decision support" in anti_make_work["detail"]
    assert "durable asset created" in anti_make_work["detail"]
    assert "control/ownership preserved" in anti_make_work["detail"]
    assert "incident risk reduced" in anti_make_work["detail"]
    assert "system capability changed" in anti_make_work["detail"]
    assert "Remediation" in anti_make_work["detail"]
    assert anti_make_work["metrics"]["assessed_work_item_count"] == 2
    assert anti_make_work["metrics"]["status_language_only_count"] == 2
    assert [
        item["label"]
        for item in anti_make_work["metrics"]["allowed_value_categories"]
    ] == [
        "operator decision support",
        "durable asset created",
        "control/ownership preserved",
        "incident risk reduced",
        "system capability changed",
    ]
    assert anti_make_work["metrics"]["value_category_counts"] == {
        "operator_decision_support": 0,
        "durable_asset_created": 0,
        "control_ownership_preserved": 0,
        "incident_risk_reduced": 0,
        "system_capability_changed": 0,
    }
    for example in anti_make_work["metrics"]["shallow_examples"]:
        assert example["value_categories"] == []
        assert example["remediation"].startswith("Add evidence for at least one allowed value category")
        assert example["issue"] == "status_language_without_value_category_evidence"
    assert "anti_make_work_check" in benchmark["critical_failures"]
    assert "reliability_gate" not in benchmark["critical_failures"]
    assert benchmark["issue_selection"]["blocked_checks"] == [
        "anti_make_work_check",
        "operator_value_alignment",
    ]
    assert benchmark["project_score"] < 100.0


def test_operator_decision_support_passes_anti_make_work_value_category(tmp_path):
    now = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    recent = now.isoformat()

    journal_path = tmp_path / "journal.json"
    codex_path = tmp_path / "runs.json"
    ctx_path = tmp_path / "session_bindings.json"
    ontology_root = _seed_ontology_repo(tmp_path, generated_at=recent)

    _write_json(
        journal_path,
        {
            "entries": [
                {
                    "id": "decision-support",
                    "occurredAt": recent,
                    "summary": "Prepared operator decision support for HAD-1100.",
                    "operatorDecisionSupport": (
                        "Operator can choose between preserving strict anti-make-work "
                        "enforcement or relaxing the check, with risk and remediation called out."
                    ),
                }
            ]
        },
    )
    _write_json(
        codex_path,
        {"runs": {"codex_1": {"run_id": "codex_1", "status": "completed", "completed_at": recent}}},
    )
    _write_json(
        ctx_path,
        {"sessions": {"ctx_1": {"session_id": "ctx_1", "active": False, "updated_at": recent}}},
    )

    benchmark = self_improvement_tool.evaluate_self_improvement_benchmark(
        journal_path=journal_path,
        codex_runs_path=codex_path,
        ctx_bindings_path=ctx_path,
        ontology_root=ontology_root,
        history_path=tmp_path / "history.json",
        now=now,
        persist=False,
    )

    anti_make_work = benchmark["checks"]["anti_make_work_check"]
    assert anti_make_work["status"] == "pass"
    assert anti_make_work["metrics"]["assessed_work_item_count"] == 1
    assert anti_make_work["metrics"]["durable_evidence_count"] == 1
    assert anti_make_work["metrics"]["value_category_counts"]["operator_decision_support"] == 1
    assert anti_make_work["metrics"]["value_category_counts"]["durable_asset_created"] == 0
    assert anti_make_work["metrics"]["shallow_work_item_count"] == 0
    assert anti_make_work["metrics"]["durable_examples"][0]["value_categories"] == [
        "operator_decision_support"
    ]
    assert anti_make_work["metrics"]["durable_examples"][0]["remediation"] is None


def test_durable_work_evidence_passes_anti_make_work_check(tmp_path):
    now = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    recent = now.isoformat()

    journal_path = tmp_path / "journal.json"
    codex_path = tmp_path / "runs.json"
    ctx_path = tmp_path / "session_bindings.json"
    ontology_root = _seed_ontology_repo(tmp_path, generated_at=recent)

    _write_json(
        journal_path,
        {
            "entries": [
                {
                    "id": "durable",
                    "occurredAt": recent,
                    "summary": "Implemented HAD-1019 benchmark hardening.",
                    "notes": (
                        "PR #1019 opened with commit abc1234; "
                        "pytest tests/tools/test_self_improvement_tool.py passed."
                    ),
                    "linearIssues": ["HAD-1019"],
                    "commitShas": ["abc1234"],
                    "operatorDecisionSupport": "Operator can compare the PR and test evidence before selecting the next issue.",
                    "pullRequests": ["https://github.com/taboularasa/hermes-agent/pull/1019"],
                    "tests": ["pytest tests/tools/test_self_improvement_tool.py passed"],
                    "reposTouched": ["hermes-agent"],
                }
            ]
        },
    )
    _write_json(
        codex_path,
        {
            "runs": {
                "codex_durable": {
                    "run_id": "codex_durable",
                    "status": "completed",
                    "completed_at": recent,
                    "final_message": (
                        "CHANGED_FILES\n"
                        "- tools/self_improvement_tool.py\n"
                        "VERIFICATION\n"
                        "- pytest tests/tools/test_self_improvement_tool.py -q passed\n"
                        "COMMIT\n"
                        "- abc1234\n"
                        "PULL_REQUEST\n"
                        "- https://github.com/taboularasa/hermes-agent/pull/1019\n"
                        "OPERATOR_DECISION_SUPPORT\n"
                        "- Operator can choose the next issue from verified PR/test evidence."
                    ),
                    "exit_code": 0,
                }
            }
        },
    )
    _write_json(
        ctx_path,
        {"sessions": {"ctx_1": {"session_id": "ctx_1", "active": False, "updated_at": recent}}},
    )

    benchmark = self_improvement_tool.evaluate_self_improvement_benchmark(
        journal_path=journal_path,
        codex_runs_path=codex_path,
        ctx_bindings_path=ctx_path,
        ontology_root=ontology_root,
        history_path=tmp_path / "history.json",
        now=now,
        persist=False,
    )

    anti_make_work = benchmark["checks"]["anti_make_work_check"]
    assert anti_make_work["status"] == "pass"
    assert anti_make_work["score"] == 1.0
    assert anti_make_work["metrics"]["assessed_work_item_count"] == 2
    assert anti_make_work["metrics"]["durable_evidence_count"] == 2
    assert anti_make_work["metrics"]["shallow_work_item_count"] == 0
    assert anti_make_work["metrics"]["value_category_counts"]["durable_asset_created"] == 2
    assert anti_make_work["metrics"]["value_category_counts"]["system_capability_changed"] == 2
    assert "durable asset created" in anti_make_work["detail"]
    assert "system capability changed" in anti_make_work["detail"]
    assert benchmark["critical_failures"] == []


def test_raw_throughput_does_not_pass_operator_value_alignment(tmp_path):
    now = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    recent = now.isoformat()

    journal_path = tmp_path / "journal.json"
    codex_path = tmp_path / "runs.json"
    ctx_path = tmp_path / "session_bindings.json"
    ontology_root = _seed_ontology_repo(tmp_path, generated_at=recent)

    _write_json(
        journal_path,
        {
            "entries": [
                {
                    "id": "volume-1",
                    "occurredAt": recent,
                    "summary": "Implemented HAD-1020 benchmark update.",
                    "changedFiles": ["tools/self_improvement_tool.py"],
                    "tests": ["pytest tests/tools/test_self_improvement_tool.py passed"],
                },
                {
                    "id": "volume-2",
                    "occurredAt": recent,
                    "summary": "Opened PR #1020 with commit abc1234.",
                    "pullRequests": ["https://github.com/taboularasa/hermes-agent/pull/1020"],
                    "commitShas": ["abc1234"],
                },
            ]
        },
    )
    _write_json(codex_path, {"runs": {"codex_1": {"run_id": "codex_1", "completed_at": recent}}})
    _write_json(ctx_path, {"sessions": {"ctx_1": {"session_id": "ctx_1", "updated_at": recent}}})

    benchmark = self_improvement_tool.evaluate_self_improvement_benchmark(
        journal_path=journal_path,
        codex_runs_path=codex_path,
        ctx_bindings_path=ctx_path,
        ontology_root=ontology_root,
        history_path=tmp_path / "history.json",
        now=now,
        persist=False,
    )

    anti_make_work = benchmark["checks"]["anti_make_work_check"]
    assert anti_make_work["status"] == "pass"
    assert anti_make_work["metrics"]["durable_evidence_count"] == 2

    operator_value = benchmark["checks"]["operator_value_alignment"]
    assert operator_value["status"] == "fail"
    assert operator_value["score"] == 0.45
    assert operator_value["metrics"]["verified_system_change_count"] == 2
    assert operator_value["metrics"]["operator_decision_support_count"] == 0
    assert "operator_value_alignment" in benchmark["critical_failures"]
    assert benchmark["issue_selection"]["quantity_guardrail_active"] is True
    assert "decision support" in benchmark["summary"]["operator_value_alignment"]


def test_execution_throughput_gap_prioritizes_journal_followthrough_without_ctx_blocker(tmp_path):
    now = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    recent = now.isoformat()

    journal_path = tmp_path / "journal.json"
    codex_path = tmp_path / "runs.json"
    ctx_path = tmp_path / "session_bindings.json"
    ontology_root = _seed_ontology_repo(tmp_path, generated_at=recent)

    _write_json(
        journal_path,
        {
            "entries": [
                {
                    "id": "journal-followthrough-1",
                    "occurredAt": recent,
                    "summary": "Implemented one self-improvement follow-through update.",
                    "operatorDecisionSupport": (
                        "Operator can inspect one completed delivery and choose whether "
                        "to backfill the rest."
                    ),
                    "changedFiles": ["tools/self_improvement_tool.py"],
                    "tests": ["pytest tests/tools/test_self_improvement_tool.py passed"],
                }
            ]
        },
    )
    _write_json(
        codex_path,
        {
            "runs": {
                f"codex_{idx}": {
                    "run_id": f"codex_{idx}",
                    "status": "completed",
                    "completed_at": recent,
                    "exit_code": 0,
                    "final_message": (
                        "CHANGED_FILES\n"
                        "- tools/self_improvement_tool.py\n"
                        "VERIFICATION\n"
                        "- pytest tests/tools/test_self_improvement_tool.py passed\n"
                        f"COMMIT\n- abc123{idx}\n"
                        f"PULL_REQUEST\n- https://github.com/taboularasa/hermes-agent/pull/{11680 + idx}"
                    ),
                }
                for idx in range(6)
            }
        },
    )
    _write_json(
        ctx_path,
        {
            "sessions": {
                "ctx_retired": {
                    "session_id": "ctx_retired",
                    "active": False,
                    "updated_at": recent,
                }
            }
        },
    )

    benchmark = self_improvement_tool.evaluate_self_improvement_benchmark(
        journal_path=journal_path,
        codex_runs_path=codex_path,
        ctx_bindings_path=ctx_path,
        ontology_root=ontology_root,
        history_path=tmp_path / "history.json",
        now=now,
        persist=False,
    )

    reliability_gate = benchmark["checks"]["reliability_gate"]
    assert reliability_gate["status"] == "pass"
    assert reliability_gate["metrics"]["ctx_remediation_required"] is False
    assert benchmark["gate"]["sources"]["ctx_bindings"]["status"] == "inactive"
    assert benchmark["gate"]["ctx_remediation"]["required"] is False

    drift = benchmark["checks"]["leading_indicator_drift"]
    remediation = drift["metrics"]["execution_throughput_remediation"]
    assert drift["status"] == "warn"
    assert remediation["required"] is True
    assert remediation["blocking_surface"] == "journal_follow_through"
    assert remediation["recent_completed_codex_count"] == 6
    assert remediation["recent_journal_work_item_count"] == 1
    assert remediation["ctx_inactivity_blocking"] is False
    assert "inactive ctx is informational" in drift["detail"]
    assert "journal entries" in remediation["actions"][0]

    issue_selection = benchmark["issue_selection"]
    assert issue_selection["recommended_focus"] == "Codex delivery journal follow-through"
    assert "Inactive ctx evidence is informational" in issue_selection["detail"]
    assert "operator_value_alignment" in issue_selection["blocked_checks"]
    assert "leading_indicator_drift" in issue_selection["blocked_checks"]
    assert "journal_follow_through" in issue_selection["execution_throughput"]["blocking_surface"]

    summary_markdown = self_improvement_tool._format_pipeline_summary(
        benchmark=benchmark,
        top_candidate=None,
    )
    compact = self_improvement_tool._pipeline_benchmark_summary(benchmark)
    assert (
        "- execution_throughput_remediation=6 completed Codex run(s), "
        "1 journal work item(s)"
    ) in summary_markdown
    assert "- execution_throughput_action=backfill journal entries" in summary_markdown
    assert (
        compact["leading_indicator_drift"]["execution_throughput_remediation"]["required"]
        is True
    )


def test_operator_value_report_preserves_decision_support_evidence(tmp_path):
    now = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    recent = now.isoformat()

    journal_path = tmp_path / "journal.json"
    codex_path = tmp_path / "runs.json"
    ctx_path = tmp_path / "session_bindings.json"
    ontology_root = _seed_ontology_repo(tmp_path, generated_at=recent)

    _write_json(
        journal_path,
        {
            "entries": [
                {
                    "id": "decision-report",
                    "occurredAt": recent,
                    "summary": "Implemented HAD-1020 operator-value reporting.",
                    "selectedWork": "HAD-1020: raise operator-value alignment in reporting.",
                    "nextDecision": "Operator chooses whether to backfill history or continue with the next benchmark issue.",
                    "blocker": "No local blocker; PR publication still depends on GitHub auth.",
                    "owner": "operator",
                    "tradeoff": "Keep verified system-change evidence strict while adding compact decision-support excerpts.",
                    "changedFiles": ["tools/self_improvement_tool.py"],
                    "tests": ["pytest tests/tools/test_self_improvement_tool.py passed"],
                }
            ]
        },
    )
    _write_json(codex_path, {"runs": {"codex_1": {"run_id": "codex_1", "completed_at": recent}}})
    _write_json(ctx_path, {"sessions": {"ctx_1": {"session_id": "ctx_1", "updated_at": recent}}})

    benchmark = self_improvement_tool.evaluate_self_improvement_benchmark(
        journal_path=journal_path,
        codex_runs_path=codex_path,
        ctx_bindings_path=ctx_path,
        ontology_root=ontology_root,
        history_path=tmp_path / "history.json",
        now=now,
        persist=False,
    )

    operator_value = benchmark["checks"]["operator_value_alignment"]
    assert operator_value["status"] == "pass"
    assert operator_value["metrics"]["operator_decision_support_count"] == 1
    assert operator_value["metrics"]["verified_system_change_count"] == 1
    evidence = operator_value["metrics"]["operator_decision_support_examples"][0]["evidence"]
    assert [item["field"] for item in evidence] == [
        "selected_work",
        "next_decision",
        "blocker",
        "owner",
        "tradeoff",
    ]
    assert "backfill history" in evidence[1]["value"]
    assert benchmark["operator_value_checks"]["operator_decision_support_evidence"] == [
        operator_value["metrics"]["operator_decision_support_examples"][0]
    ]
    assert "selected_work" in benchmark["summary"]["operator_decision_support_evidence"]
    assert "next_decision" in benchmark["summary"]["operator_decision_support_evidence"]


def test_benchmark_history_persists_operator_decision_support_snapshot(tmp_path):
    now = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    recent = now.isoformat()

    journal_path = tmp_path / "journal.json"
    codex_path = tmp_path / "runs.json"
    ctx_path = tmp_path / "session_bindings.json"
    history_path = tmp_path / "history.json"
    ontology_root = _seed_ontology_repo(tmp_path, generated_at=recent)

    _write_json(
        journal_path,
        {
            "entries": [
                {
                    "id": "history-decision-report",
                    "occurredAt": recent,
                    "summary": "Implemented HAD-1020 benchmark history reporting.",
                    "selectedWork": "HAD-1020 benchmark history snapshot.",
                    "nextDecision": "Operator can decide whether the compact report is enough for the next self-improvement issue.",
                    "changedFiles": ["tools/self_improvement_tool.py"],
                    "tests": ["pytest tests/tools/test_self_improvement_tool.py passed"],
                }
            ]
        },
    )
    _write_json(codex_path, {"runs": {"codex_1": {"run_id": "codex_1", "completed_at": recent}}})
    _write_json(ctx_path, {"sessions": {"ctx_1": {"session_id": "ctx_1", "updated_at": recent}}})

    self_improvement_tool.evaluate_self_improvement_benchmark(
        journal_path=journal_path,
        codex_runs_path=codex_path,
        ctx_bindings_path=ctx_path,
        ontology_root=ontology_root,
        history_path=history_path,
        now=now,
        persist=True,
    )

    history = json.loads(history_path.read_text(encoding="utf-8"))
    run = history["runs"][-1]
    assert run["operator_value_checks"]["operator_decision_support_rate"] == 1.0
    evidence = run["operator_value_checks"]["operator_decision_support_evidence"][0]["evidence"]
    assert [item["field"] for item in evidence] == ["selected_work", "next_decision"]
    assert "compact report" in evidence[1]["value"]
    assert run["issue_selection"]["recommended_focus"] == "normal lane selection"


def test_leading_indicator_plateau_hold_keeps_operator_value_guardrail_active(tmp_path):
    now = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    recent = now.isoformat()

    journal_path = tmp_path / "journal.json"
    codex_path = tmp_path / "runs.json"
    ctx_path = tmp_path / "session_bindings.json"
    history_path = tmp_path / "history.json"
    ontology_root = _seed_ontology_repo(tmp_path, generated_at=recent)

    _write_json(
        journal_path,
        {
            "entries": [
                {
                    "id": "volume-1",
                    "occurredAt": recent,
                    "summary": "Implemented HAD-1101 benchmark update.",
                    "changedFiles": ["tools/self_improvement_tool.py"],
                    "tests": ["pytest tests/tools/test_self_improvement_tool.py passed"],
                },
                {
                    "id": "volume-2",
                    "occurredAt": recent,
                    "summary": "Opened PR #1101 with commit abc1234.",
                    "pullRequests": ["https://github.com/taboularasa/hermes-agent/pull/1101"],
                    "commitShas": ["abc1234"],
                },
            ]
        },
    )
    _write_json(codex_path, {"runs": {"codex_1": {"run_id": "codex_1", "completed_at": recent}}})
    _write_json(ctx_path, {"sessions": {"ctx_1": {"session_id": "ctx_1", "updated_at": recent}}})
    _write_json(
        history_path,
        {
            "version": 1,
            "evaluations": [
                {
                    "evaluated_at": (now - timedelta(hours=42 - idx * 6)).isoformat(),
                    "checks": {
                        "operator_value_alignment": {
                            "score": score,
                            "status": "pass" if score >= 0.85 else "fail",
                        }
                    },
                }
                for idx, score in enumerate([0.878, 0.878, 0.878, 0.878, 0.878, 0.45, 0.45])
            ],
        },
    )

    benchmark = self_improvement_tool.evaluate_self_improvement_benchmark(
        journal_path=journal_path,
        codex_runs_path=codex_path,
        ctx_bindings_path=ctx_path,
        ontology_root=ontology_root,
        history_path=history_path,
        now=now,
        persist=False,
    )

    operator_value = benchmark["checks"]["operator_value_alignment"]
    assert operator_value["status"] == "fail"
    assert operator_value["score"] == 0.45

    drift = benchmark["checks"]["leading_indicator_drift"]
    assert drift["status"] == "pass"
    assert drift["metrics"]["triggered_harbingers"] == []
    assert drift["metrics"]["stabilization_hold"]["active"] is True
    assert "stabilization hold" in drift["detail"]
    assert "leading_indicator_drift" not in benchmark["critical_failures"]
    assert "operator_value_alignment" in benchmark["critical_failures"]
    assert benchmark["issue_selection"]["quantity_guardrail_active"] is True
    assert benchmark["issue_selection"]["suppress_raw_throughput_selection"] is True


def test_leading_indicator_drift_fails_when_operator_value_regresses(tmp_path):
    now = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    recent = now.isoformat()

    journal_path = tmp_path / "journal.json"
    codex_path = tmp_path / "runs.json"
    ctx_path = tmp_path / "session_bindings.json"
    history_path = tmp_path / "history.json"
    ontology_root = _seed_ontology_repo(tmp_path, generated_at=recent)

    entries = []
    for idx in range(4):
        entries.append(
            {
                "id": f"aligned-{idx}",
                "occurredAt": recent,
                "summary": f"Implemented operator-value path {idx}.",
                "operatorDecisionSupport": "Operator can compare the verified change and choose the next issue.",
                "changedFiles": ["tools/self_improvement_tool.py"],
                "tests": ["pytest tests/tools/test_self_improvement_tool.py passed"],
            }
        )
    for idx in range(2):
        entries.append(
            {
                "id": f"decision-only-{idx}",
                "occurredAt": recent,
                "summary": f"Prepared operator decision support note {idx}.",
                "operatorDecisionSupport": "Operator has a blocker and recommended next decision.",
            }
        )

    _write_json(journal_path, {"entries": entries})
    _write_json(codex_path, {"runs": {"codex_1": {"run_id": "codex_1", "completed_at": recent}}})
    _write_json(ctx_path, {"sessions": {"ctx_1": {"session_id": "ctx_1", "updated_at": recent}}})
    _write_json(
        history_path,
        {
            "version": 1,
            "evaluations": [
                {
                    "evaluated_at": (now - timedelta(hours=6)).isoformat(),
                    "score": 94.0,
                    "direction": "positive",
                    "checks": {"operator_value_alignment": 0.95},
                }
            ],
        },
    )

    benchmark = self_improvement_tool.evaluate_self_improvement_benchmark(
        journal_path=journal_path,
        codex_runs_path=codex_path,
        ctx_bindings_path=ctx_path,
        ontology_root=ontology_root,
        history_path=history_path,
        now=now,
        persist=False,
    )

    operator_value = benchmark["checks"]["operator_value_alignment"]
    assert operator_value["status"] == "pass"
    assert operator_value["score"] == 0.8833

    drift = benchmark["checks"]["leading_indicator_drift"]
    assert drift["status"] == "fail"
    assert drift["metrics"]["previous_operator_value_score"] == 0.95
    assert drift["metrics"]["operator_value_delta"] == -0.0667
    assert "leading_indicator_drift" in benchmark["critical_failures"]
    assert benchmark["direction"] == "negative"
    assert benchmark["trend"] == "regressing"


def test_leading_indicator_drift_reports_exact_degraded_plateau_hold():
    scores = [0.878, 0.878, 0.878, 0.878, 0.878, 0.4292, 0.4292, 0.4292]
    history = {
        "version": 1,
        "evaluations": [
            {
                "checks": {
                    "operator_value_alignment": {
                        "score": score,
                        "status": "pass" if score >= 0.85 else "fail",
                    }
                }
            }
            for score in scores[:-1]
        ],
    }
    operator_value = {"score": scores[-1], "status": "fail", "metrics": {}}

    drift = self_improvement_tool._evaluate_leading_indicator_drift_check(
        operator_value,
        history,
        {"operator_value_alignment": operator_value},
    )

    scorecard = drift["metrics"]["harbinger_scorecard"]
    assert drift["status"] == "pass"
    assert drift["score"] == 0.85
    assert drift["metrics"]["triggered_harbingers"] == []
    assert drift["metrics"]["recommended_mitigations"] == []
    assert drift["metrics"]["stabilization_hold"]["active"] is True
    assert drift["metrics"]["stabilization_hold"]["state"] == "stabilization_hold"
    assert drift["metrics"]["stabilization_hold"]["recent_scores"] == [0.4292, 0.4292, 0.4292]
    assert scorecard["critical_slowing_down"]["triggered"] is False
    assert scorecard["critical_slowing_down"]["evidence"]["active_signal"] is True
    assert scorecard["variance_explosion"]["triggered"] is False
    assert scorecard["variance_explosion"]["evidence"]["active_signal"] is True


def test_leading_indicator_drift_passes_recovered_low_variance_state():
    scores = [0.878, 0.878, 0.4292, 0.878, 0.878, 0.878]
    history = {
        "version": 1,
        "evaluations": [
            {
                "checks": {
                    "operator_value_alignment": {
                        "score": score,
                        "status": "pass" if score >= 0.85 else "fail",
                    }
                }
            }
            for score in scores[:-1]
        ],
    }
    operator_value = {"score": scores[-1], "status": "pass", "metrics": {}}

    drift = self_improvement_tool._evaluate_leading_indicator_drift_check(
        operator_value,
        history,
        {"operator_value_alignment": operator_value},
    )

    assert drift["status"] == "pass"
    assert drift["score"] == 1.0
    assert drift["metrics"]["triggered_harbingers"] == []
    assert drift["metrics"]["stabilization_hold"]["active"] is False
    assert drift["metrics"]["stabilization_hold"]["state"] == "recovered_low_variance"
    assert drift["detail"] == "Operator-value leading indicator is stable or improving."


def test_leading_indicator_drift_flags_critical_slowing_down(tmp_path):
    now = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    recent = now.isoformat()

    journal_path = tmp_path / "journal.json"
    codex_path = tmp_path / "runs.json"
    ctx_path = tmp_path / "session_bindings.json"
    history_path = tmp_path / "history.json"
    ontology_root = _seed_ontology_repo(tmp_path, generated_at=recent)

    _write_json(journal_path, {"entries": _operator_value_entries(recent)})
    _write_json(codex_path, {"runs": {"codex_1": {"run_id": "codex_1", "completed_at": recent}}})
    _write_json(ctx_path, {"sessions": {"ctx_1": {"session_id": "ctx_1", "updated_at": recent}}})
    _write_json(
        history_path,
        {
            "version": 1,
            "evaluations": [
                {
                    "evaluated_at": (now - timedelta(hours=24 - idx * 6)).isoformat(),
                    "checks": {"operator_value_alignment": {"score": score, "status": "pass"}},
                }
                for idx, score in enumerate([0.98, 0.9, 0.887, 0.884])
            ],
        },
    )

    benchmark = self_improvement_tool.evaluate_self_improvement_benchmark(
        journal_path=journal_path,
        codex_runs_path=codex_path,
        ctx_bindings_path=ctx_path,
        ontology_root=ontology_root,
        history_path=history_path,
        now=now,
        persist=False,
    )

    drift = benchmark["checks"]["leading_indicator_drift"]
    scorecard = drift["metrics"]["harbinger_scorecard"]
    assert drift["status"] == "fail"
    assert drift["metrics"]["triggered_harbingers"] == ["critical_slowing_down"]
    assert scorecard["critical_slowing_down"]["triggered"] is True
    assert scorecard["critical_slowing_down"]["evidence"]["recovery_gap"] > 0.09
    assert scorecard["critical_slowing_down"]["next_action"]
    assert "leading_indicator_drift" in benchmark["critical_failures"]


def test_leading_indicator_drift_flags_variance_explosion_and_flickering(tmp_path):
    now = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    recent = now.isoformat()

    journal_path = tmp_path / "journal.json"
    codex_path = tmp_path / "runs.json"
    ctx_path = tmp_path / "session_bindings.json"
    history_path = tmp_path / "history.json"
    ontology_root = _seed_ontology_repo(tmp_path, generated_at=recent)

    _write_json(journal_path, {"entries": _operator_value_entries(recent)})
    _write_json(codex_path, {"runs": {"codex_1": {"run_id": "codex_1", "completed_at": recent}}})
    _write_json(ctx_path, {"sessions": {"ctx_1": {"session_id": "ctx_1", "updated_at": recent}}})
    _write_json(
        history_path,
        {
            "version": 1,
            "evaluations": [
                {
                    "evaluated_at": (now - timedelta(hours=36 - idx * 6)).isoformat(),
                    "checks": {"operator_value_alignment": {"score": score, "status": status}},
                }
                for idx, (score, status) in enumerate(
                    [
                        (0.92, "pass"),
                        (0.91, "pass"),
                        (0.915, "pass"),
                        (0.52, "fail"),
                        (0.93, "pass"),
                        (0.5, "fail"),
                    ]
                )
            ],
        },
    )

    benchmark = self_improvement_tool.evaluate_self_improvement_benchmark(
        journal_path=journal_path,
        codex_runs_path=codex_path,
        ctx_bindings_path=ctx_path,
        ontology_root=ontology_root,
        history_path=history_path,
        now=now,
        persist=False,
    )

    drift = benchmark["checks"]["leading_indicator_drift"]
    scorecard = drift["metrics"]["harbinger_scorecard"]
    assert drift["status"] == "fail"
    assert set(drift["metrics"]["triggered_harbingers"]) == {"variance_explosion", "flickering"}
    assert scorecard["variance_explosion"]["evidence"]["recent_range"] > 0.4
    assert scorecard["flickering"]["evidence"]["transition_count"] >= 3
    assert len(drift["metrics"]["recommended_mitigations"]) == 2


def test_leading_indicator_drift_flags_correlation_explosion(tmp_path):
    now = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    recent = now.isoformat()

    journal_path = tmp_path / "journal.json"
    codex_path = tmp_path / "runs.json"
    ctx_path = tmp_path / "session_bindings.json"
    history_path = tmp_path / "history.json"
    ontology_root = _seed_ontology_repo(tmp_path, generated_at=recent, status="degraded")

    _write_json(
        journal_path,
        {
            "entries": [
                {
                    "id": "status-only",
                    "occurredAt": recent,
                    "summary": "Status update: selected and working on the active self-improvement item.",
                }
            ]
        },
    )
    _write_json(
        codex_path,
        {
            "runs": {
                "codex_status": {
                    "run_id": "codex_status",
                    "status": "completed",
                    "completed_at": recent,
                    "final_message": "STATUS\nWorking on active item. Next step is continued monitoring.",
                    "exit_code": 0,
                }
            }
        },
    )
    _write_json(ctx_path, {"sessions": {"ctx_1": {"session_id": "ctx_1", "updated_at": recent}}})
    _write_json(
        history_path,
        {
            "runs": [
                {
                    "generated_at": (now - timedelta(hours=6)).isoformat(),
                    "project_score": 98.0,
                    "checks": {
                        "reliability_gate": {"score": 0.96, "status": "pass"},
                        "anti_make_work_check": {"score": 0.98, "status": "pass"},
                        "operator_value_alignment": {"score": 0.95, "status": "pass"},
                    },
                }
            ]
        },
    )

    benchmark = self_improvement_tool.evaluate_self_improvement_benchmark(
        journal_path=journal_path,
        codex_runs_path=codex_path,
        ctx_bindings_path=ctx_path,
        ontology_root=ontology_root,
        history_path=history_path,
        now=now,
        persist=False,
    )

    drift = benchmark["checks"]["leading_indicator_drift"]
    scorecard = drift["metrics"]["harbinger_scorecard"]
    assert "correlation_explosion" in drift["metrics"]["triggered_harbingers"]
    assert scorecard["correlation_explosion"]["triggered"] is True
    assert scorecard["correlation_explosion"]["evidence"]["dropped_check_count"] == 3
    assert set(scorecard["correlation_explosion"]["evidence"]["check_deltas"]) == {
        "anti_make_work_check",
        "operator_value_alignment",
        "reliability_gate",
    }


def test_leading_indicator_scorecard_reports_evidence_and_mitigation_for_all_harbingers():
    history_scores = [
        (0.95, "pass"),
        (0.95, "pass"),
        (0.95, "fail"),
        (0.95, "pass"),
        (0.92, "fail"),
        (0.6, "pass"),
        (0.58, "fail"),
    ]
    history = {
        "version": 1,
        "evaluations": [
            {
                "checks": {
                    "reliability_gate": {"score": 0.95, "status": "pass"},
                    "anti_make_work_check": {"score": 0.95, "status": "pass"},
                    "operator_value_alignment": {
                        "score": score,
                        "status": status,
                    },
                }
            }
            for score, status in history_scores
        ],
    }
    current_checks = {
        "reliability_gate": {"score": 0.7, "status": "warn"},
        "anti_make_work_check": {"score": 0.7, "status": "warn"},
        "operator_value_alignment": {"score": 0.35, "status": "fail", "metrics": {}},
    }

    drift = self_improvement_tool._evaluate_leading_indicator_drift_check(
        current_checks["operator_value_alignment"],
        history,
        current_checks,
    )

    expected_harbingers = {
        "critical_slowing_down",
        "variance_explosion",
        "flickering",
        "correlation_explosion",
    }
    scorecard = drift["metrics"]["harbinger_scorecard"]
    assert set(drift["metrics"]["triggered_harbingers"]) == expected_harbingers
    assert {item["harbinger"] for item in drift["metrics"]["recommended_mitigations"]} == (
        expected_harbingers
    )
    for harbinger in expected_harbingers:
        card = scorecard[harbinger]
        assert card["triggered"] is True
        assert card["evidence"]
        assert card["evidence_summary"]
        assert card["mitigation"]
        assert card["next_action"]
        assert harbinger in drift["detail"]


def test_benchmark_history_persists_leading_indicator_scorecard_snapshot(tmp_path):
    now = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    recent = now.isoformat()

    journal_path = tmp_path / "journal.json"
    codex_path = tmp_path / "runs.json"
    ctx_path = tmp_path / "session_bindings.json"
    history_path = tmp_path / "history.json"
    ontology_root = _seed_ontology_repo(tmp_path, generated_at=recent)

    _write_json(journal_path, {"entries": _operator_value_entries(recent)})
    _write_json(codex_path, {"runs": {"codex_1": {"run_id": "codex_1", "completed_at": recent}}})
    _write_json(ctx_path, {"sessions": {"ctx_1": {"session_id": "ctx_1", "updated_at": recent}}})
    _write_json(
        history_path,
        {
            "version": 1,
            "evaluations": [
                {
                    "evaluated_at": (now - timedelta(hours=24 - idx * 6)).isoformat(),
                    "checks": {"operator_value_alignment": {"score": score, "status": "pass"}},
                }
                for idx, score in enumerate([0.98, 0.9, 0.887, 0.884])
            ],
        },
    )

    self_improvement_tool.evaluate_self_improvement_benchmark(
        journal_path=journal_path,
        codex_runs_path=codex_path,
        ctx_bindings_path=ctx_path,
        ontology_root=ontology_root,
        history_path=history_path,
        now=now,
        persist=True,
    )

    history = json.loads(history_path.read_text(encoding="utf-8"))
    drift_snapshot = history["runs"][-1]["checks"]["leading_indicator_drift"]
    critical = drift_snapshot["harbinger_scorecard"]["critical_slowing_down"]
    assert drift_snapshot["triggered_harbingers"] == ["critical_slowing_down"]
    assert drift_snapshot["recommended_mitigations"][0]["evidence_summary"]
    assert critical["evidence_summary"]
    assert critical["mitigation"]


def test_pipeline_summary_surfaces_leading_indicator_harbinger_mitigation():
    drift = self_improvement_tool._evaluate_leading_indicator_drift_check(
        {"score": 0.82, "status": "warn", "metrics": {}},
        {
            "evaluations": [
                {
                    "checks": {
                        "operator_value_alignment": {
                            "score": score,
                            "status": "pass" if score >= 0.85 else "warn",
                        }
                    }
                }
                for score in [0.97, 0.9, 0.86, 0.83]
            ]
        },
        {"operator_value_alignment": {"score": 0.82, "status": "warn", "metrics": {}}},
    )
    benchmark = {
        "score": 88.0,
        "project_score": 88.0,
        "direction": "negative",
        "trend": "regressing",
        "critical_failures": ["leading_indicator_drift"],
        "checks": {
            "reliability_gate": {"score": 1.0, "status": "pass"},
            "leading_indicator_drift": drift,
        },
    }

    summary = self_improvement_tool._format_pipeline_summary(
        benchmark=benchmark,
        top_candidate=None,
    )
    compact = self_improvement_tool._pipeline_benchmark_summary(benchmark)

    assert "- leading_indicator_harbingers=critical_slowing_down" in summary
    assert "- leading_indicator_critical_slowing_down: evidence=" in summary
    assert "mitigation=Stop expanding self-improvement scope" in summary
    assert compact["leading_indicator_drift"]["triggered_harbingers"] == [
        "critical_slowing_down"
    ]
    assert compact["leading_indicator_drift"]["recommended_mitigations"][0][
        "evidence_summary"
    ]


def test_execution_loop_warns_on_sparse_journal_follow_through_after_codex_volume(tmp_path):
    now = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    recent = now.isoformat()

    journal_path = tmp_path / "journal.json"
    codex_path = tmp_path / "runs.json"
    ctx_path = tmp_path / "session_bindings.json"
    ontology_root = _seed_ontology_repo(tmp_path, generated_at=recent)

    _write_json(journal_path, {"entries": [{"occurredAt": recent}]})
    _write_json(
        codex_path,
        {
            "runs": {
                f"codex_{idx}": {
                    "run_id": f"codex_{idx}",
                    "status": "completed",
                    "completed_at": (now - timedelta(hours=idx)).isoformat(),
                    "exit_code": 0,
                }
                for idx in range(6)
            }
        },
    )
    _write_json(
        ctx_path,
        {
            "sessions": {
                "ctx_retired": {
                    "session_id": "ctx_retired",
                    "active": False,
                    "updated_at": recent,
                }
            }
        },
    )

    benchmark = self_improvement_tool.evaluate_self_improvement_benchmark(
        journal_path=journal_path,
        codex_runs_path=codex_path,
        ctx_bindings_path=ctx_path,
        ontology_root=ontology_root,
        history_path=tmp_path / "history.json",
        now=now,
        persist=False,
    )

    execution = benchmark["checks"]["execution_loop"]
    metrics = execution["metrics"]
    assert execution["status"] == "warn"
    assert execution["score"] == 0.7
    assert metrics["completed_codex_runs_14d"] == 6
    assert metrics["active_ctx_binding_count"] == 0
    assert metrics["ctx_binding_state"] == "inactive_informational"
    assert metrics["ctx_inactivity_blocking"] is False
    assert metrics["journal_entries_14d"] == 1
    assert metrics["journal_follow_through_rate"] == 0.1667
    assert metrics["sparse_journal_follow_through"] is True
    assert "Backfill journal evidence" in metrics["next_throughput_action"]
    assert benchmark["execution_loop"]["next_throughput_action"] == metrics["next_throughput_action"]
    assert "execution_loop" in benchmark["issue_selection"]["blocked_checks"]
    assert benchmark["issue_selection"]["recommended_focus"] == "Codex delivery journal follow-through"


def test_execution_loop_passes_when_ctx_codex_and_journal_are_in_cadence(tmp_path):
    now = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    recent = now.isoformat()

    journal_path = tmp_path / "journal.json"
    codex_path = tmp_path / "runs.json"
    ctx_path = tmp_path / "session_bindings.json"
    ontology_root = _seed_ontology_repo(tmp_path, generated_at=recent)
    worktree = tmp_path / "worktree"
    worktree.mkdir()

    _write_json(journal_path, {"entries": [{"occurredAt": recent}]})
    _write_json(
        codex_path,
        {
            "runs": {
                "codex_1": {
                    "run_id": "codex_1",
                    "status": "completed",
                    "completed_at": recent,
                    "exit_code": 0,
                }
            }
        },
    )
    _write_json(
        ctx_path,
        {
            "sessions": {
                "ctx_1": {
                    "session_id": "ctx_1",
                    "active": True,
                    "updated_at": recent,
                    "worktree_path": str(worktree),
                }
            }
        },
    )

    benchmark = self_improvement_tool.evaluate_self_improvement_benchmark(
        journal_path=journal_path,
        codex_runs_path=codex_path,
        ctx_bindings_path=ctx_path,
        ontology_root=ontology_root,
        history_path=tmp_path / "history.json",
        now=now,
        persist=False,
    )

    execution = benchmark["checks"]["execution_loop"]
    metrics = execution["metrics"]
    assert execution["status"] == "pass"
    assert metrics["completed_codex_runs_14d"] == 1
    assert metrics["active_ctx_binding_count"] == 1
    assert metrics["journal_entries_14d"] == 1
    assert metrics["journal_follow_through_rate"] == 1.0
    assert metrics["ctx_inactivity_blocking"] is False
    assert metrics["next_throughput_action"].startswith("Keep converting self-improvement work")
    assert "execution_loop" not in benchmark["issue_selection"]["blocked_checks"]


def test_stale_active_ctx_still_degrades_reliability_floor(tmp_path):
    now = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    recent = now.isoformat()
    stale_active = (now - timedelta(hours=18)).isoformat()

    journal_path = tmp_path / "journal.json"
    codex_path = tmp_path / "runs.json"
    ctx_path = tmp_path / "session_bindings.json"
    ontology_root = _seed_ontology_repo(tmp_path, generated_at=recent)
    worktree = tmp_path / "worktree"
    worktree.mkdir()

    _write_json(journal_path, {"entries": [{"occurredAt": recent}]})
    _write_json(
        codex_path,
        {"runs": {"codex_1": {"run_id": "codex_1", "status": "completed", "completed_at": recent}}},
    )
    _write_json(
        ctx_path,
        {
            "sessions": {
                "ctx_stale": {
                    "session_id": "ctx_stale",
                    "active": True,
                    "updated_at": stale_active,
                    "worktree_path": str(worktree),
                },
                "ctx_status_stale": {
                    "session_id": "ctx_status_stale",
                    "status": "active",
                    "updated_at": stale_active,
                    "worktree_path": str(worktree),
                }
            }
        },
    )

    gate = self_improvement_tool.evaluate_self_improvement_evidence(
        journal_path=journal_path,
        codex_runs_path=codex_path,
        ctx_bindings_path=ctx_path,
        ontology_root=ontology_root,
        now=now,
    )

    assert gate["status"] == "degraded"
    assert gate["sources"]["ctx_bindings"]["active_count"] == 2
    assert "stale active ctx bindings detected" in gate["warnings"]
    assert "stale active ctx bindings detected" in gate["contradictions"]
    assert len(gate["stale_active_ctx"]) == 2
    assert gate["ctx_remediation"]["required"] is True
    assert gate["ctx_remediation"]["stale_active_count"] == 2
    assert "retire stale active ctx bindings" in gate["ctx_remediation"]["action"]

    benchmark = self_improvement_tool.evaluate_self_improvement_benchmark(
        journal_path=journal_path,
        codex_runs_path=codex_path,
        ctx_bindings_path=ctx_path,
        ontology_root=ontology_root,
        history_path=tmp_path / "history.json",
        now=now,
        persist=False,
    )
    reliability_gate = benchmark["checks"]["reliability_gate"]
    assert reliability_gate["status"] == "fail"
    assert reliability_gate["metrics"]["warning_count"] >= 1
    assert reliability_gate["metrics"]["ctx_remediation_required"] is True
    assert "reliability_gate" in benchmark["critical_failures"]
    assert "reliability_gate" in benchmark["issue_selection"]["blocked_checks"]
    assert benchmark["issue_selection"]["recommended_focus"] == "self-improvement evidence freshness repair"


def test_degraded_ontology_status_still_degrades_reliability_floor(tmp_path):
    now = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    recent = now.isoformat()

    journal_path = tmp_path / "journal.json"
    codex_path = tmp_path / "runs.json"
    ctx_path = tmp_path / "session_bindings.json"
    ontology_root = _seed_ontology_repo(tmp_path, generated_at=recent, status="degraded")

    _write_json(journal_path, {"entries": [{"occurredAt": recent}]})
    _write_json(
        codex_path,
        {"runs": {"codex_1": {"run_id": "codex_1", "status": "completed", "completed_at": recent}}},
    )
    _write_json(
        ctx_path,
        {"sessions": {"ctx_1": {"session_id": "ctx_1", "active": False, "updated_at": recent}}},
    )

    gate = self_improvement_tool.evaluate_self_improvement_evidence(
        journal_path=journal_path,
        codex_runs_path=codex_path,
        ctx_bindings_path=ctx_path,
        ontology_root=ontology_root,
        now=now,
    )

    assert gate["status"] == "degraded"
    assert "ontology_intelligence evidence degraded" in gate["warnings"]
    assert "ontology intelligence artifacts are stale, missing, or degraded" in gate["contradictions"]


def test_pipeline_uses_core_benchmark_contract_without_persisting(tmp_path):
    now = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    recent = now.isoformat()
    stale_retired = (now - timedelta(days=15)).isoformat()

    journal_path = tmp_path / "journal.json"
    codex_path = tmp_path / "runs.json"
    ctx_path = tmp_path / "session_bindings.json"
    history_path = tmp_path / "history.json"
    ontology_root = _seed_ontology_repo(tmp_path, generated_at=recent)

    _write_json(journal_path, {"entries": [{"occurredAt": recent}]})
    _write_json(
        codex_path,
        {"runs": {"codex_1": {"run_id": "codex_1", "status": "completed", "completed_at": recent}}},
    )
    _write_json(
        ctx_path,
        {
            "sessions": {
                "ctx_retired": {
                    "session_id": "ctx_retired",
                    "active": False,
                    "updated_at": stale_retired,
                }
            }
        },
    )

    result = json.loads(
        self_improvement_tool.self_improvement_pipeline(
            journal_path=str(journal_path),
            codex_runs_path=str(codex_path),
            ctx_bindings_path=str(ctx_path),
            ontology_root=str(ontology_root),
            history_path=str(history_path),
            now=recent,
            persist=False,
            auto_repair_linear=True,
            auto_close_resolved=True,
        )
    )
    pipeline = result["pipeline"]
    benchmark = pipeline["benchmark"]
    reliability_gate = benchmark["checks"]["reliability_gate"]
    drift = benchmark["checks"]["leading_indicator_drift"]

    assert pipeline["runtime_surface"] == "hermes-agent-core"
    assert pipeline["linear"]["available"] is False
    assert "Linear writeback is not part" in pipeline["linear"]["error"]
    assert benchmark["contract_version"] == self_improvement_tool.BENCHMARK_CONTRACT_VERSION
    assert benchmark["gate"]["sources"]["ctx_bindings"]["status"] == "inactive"
    assert reliability_gate["score"] == 1.0
    assert reliability_gate["status"] == "pass"
    assert drift["id"] == "leading_indicator_drift"
    assert drift["status"] in {"pass", "warn", "fail"}
    assert history_path.exists() is False
    assert "reliability_gate=1.0 pass" in pipeline["summary_markdown"]


def test_core_self_improvement_pipeline_owns_default_tool_surface():
    entry = registry.get_entry("self_improvement_pipeline")
    assert entry is not None
    assert entry.toolset == "self_improvement"
    handler = entry.handler

    registry.register(
        name="self_improvement_pipeline",
        toolset="hadto-self-improvement",
        schema={
            "name": "self_improvement_pipeline",
            "description": "shadow attempt",
            "parameters": {},
        },
        handler=lambda _args, **_kw: "{}",
    )

    assert registry.get_entry("self_improvement_pipeline").handler is handler
    cli_tools = resolve_toolset("hermes-cli")
    assert "self_improvement_evidence_gate" in cli_tools
    assert "self_improvement_benchmark" in cli_tools
    assert "self_improvement_pipeline" in cli_tools
