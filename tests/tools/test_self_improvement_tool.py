import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tools import self_improvement_tool


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

    assert gate["status"] == "healthy"
    assert {entry["status"] for entry in gate["sources"].values()} == {"fresh"}
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
    assert anti_make_work["metrics"]["assessed_work_item_count"] == 2
    assert anti_make_work["metrics"]["status_language_only_count"] == 2
    assert "anti_make_work_check" in benchmark["critical_failures"]
    assert benchmark["project_score"] < 100.0


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
    assert "stale active ctx bindings detected" in gate["warnings"]
    assert "stale active ctx bindings detected" in gate["contradictions"]
    assert len(gate["stale_active_ctx"]) == 1
    assert gate["ctx_remediation"]["required"] is True
    assert gate["ctx_remediation"]["stale_active_count"] == 1
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
