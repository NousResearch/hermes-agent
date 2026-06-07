from agent.codex_workflow_must_fix import build_must_fix_loop_status


def _review(*, must_fix=None, suggested_fixes=None):
    return {
        "status": "failed" if must_fix else "passed",
        "review": {
            "verdict": "failed" if must_fix else "passed",
            "must_fix": must_fix or [],
            "suggested_fixes": suggested_fixes or [],
        },
    }


def test_must_fix_loop_stops_after_max_rounds():
    result = build_must_fix_loop_status(
        review_result=_review(must_fix=["missing regression coverage"]),
        authorized=True,
        current_round=2,
    )

    assert result["status"] == "stopped"
    assert result["reason"] == "max_fix_rounds_exhausted"
    assert result["max_fix_rounds"] == 2
    assert result["must_fix_count"] == 1
    assert result["next_actions"] == []


def test_must_fix_loop_does_not_run_without_authorization():
    result = build_must_fix_loop_status(
        review_result=_review(
            must_fix=["fix validation"],
            suggested_fixes=["change validation branch"],
        ),
        authorized=False,
    )

    assert result["status"] == "authorization_required"
    assert result["authorization_required"] is True
    assert result["blocks_continuation"] is True
    assert result["next_actions"] == ["request_explicit_must_fix_loop_authorization"]
    assert result["suggested_fixes_recorded"] == ["change validation branch"]
    assert result["auto_implements_suggested_fixes"] is False


def test_must_fix_loop_stops_on_review_unavailable():
    result = build_must_fix_loop_status(
        review_result={"status": "unavailable", "reason": "review_timeout"},
        authorized=True,
    )

    assert result["status"] == "stopped"
    assert result["reason"] == "review_unavailable"
    assert result["blocks_continuation"] is True


def test_must_fix_false_positive_can_be_recorded_with_evidence():
    result = build_must_fix_loop_status(
        review_result=_review(must_fix=["incorrect review claim"]),
        authorized=True,
        finding_resolutions=[
            {
                "finding_id": "claim-1",
                "disposition": "false_positive",
                "evidence": {"review_claim_verified_against": "existing focused test covers this branch"},
            }
        ],
    )

    assert result["status"] == "ready_for_round"
    assert result["reason"] == "must_fix_round_available"
    assert result["auto_implements_suggested_fixes"] is False


def test_must_fix_true_positive_requires_regression_test_when_practical():
    result = build_must_fix_loop_status(
        review_result=_review(must_fix=["parser accepts invalid input"]),
        authorized=True,
        finding_resolutions=[
            {
                "finding_id": "claim-1",
                "disposition": "true_positive",
                "regression_test_practical": True,
            }
        ],
    )

    assert result["status"] == "stopped"
    assert result["reason"] == "true_positive_requires_regression_test"
    assert result["blocks_continuation"] is True


def test_must_fix_loop_treats_not_requested_review_as_unavailable():
    result = build_must_fix_loop_status(
        review_result={"status": "not_requested", "passed": False},
        authorized=True,
    )

    assert result["status"] == "stopped"
    assert result["reason"] == "review_unavailable"
    assert result["blocks_continuation"] is True


def test_must_fix_loop_stops_on_verification_failed():
    result = build_must_fix_loop_status(
        review_result=_review(must_fix=["fix me"]),
        authorized=True,
        verification_result={"status": "failed", "passed": False},
    )

    assert result["status"] == "stopped"
    assert result["reason"] == "verification_failed"
    assert result["verification_result"]["status"] == "failed"


def test_must_fix_loop_stops_on_dirty_overlap_or_allowlist_escape():
    dirty_overlap = build_must_fix_loop_status(
        review_result=_review(must_fix=["fix me"]),
        authorized=True,
        dirty_overlap=True,
    )
    allowlist_escape = build_must_fix_loop_status(
        review_result=_review(must_fix=["fix me"]),
        authorized=True,
        allowlist_escape=True,
    )

    assert dirty_overlap["reason"] == "dirty_overlap_or_allowlist_escape"
    assert dirty_overlap["dirty_overlap"] is True
    assert allowlist_escape["reason"] == "dirty_overlap_or_allowlist_escape"
    assert allowlist_escape["allowlist_escape"] is True


def test_must_fix_loop_stops_on_secret_risk_and_repeated_codex_flood_timeout():
    secret_risk = build_must_fix_loop_status(
        review_result=_review(must_fix=["fix me"]),
        authorized=True,
        new_secret_or_real_data_risk=True,
    )
    flood_timeout = build_must_fix_loop_status(
        review_result=_review(must_fix=["fix me"]),
        authorized=True,
        codex_flood_timeout_count=2,
    )

    assert secret_risk["reason"] == "new_secret_or_real_data_risk"
    assert flood_timeout["reason"] == "repeated_codex_flood_or_timeout"
    assert flood_timeout["codex_flood_timeout_count"] == 2


def test_must_fix_false_positive_requires_evidence():
    result = build_must_fix_loop_status(
        review_result=_review(must_fix=["incorrect review claim"]),
        authorized=True,
        finding_resolutions=[{"finding_id": "claim-1", "disposition": "false_positive"}],
    )

    assert result["status"] == "stopped"
    assert result["reason"] == "false_positive_requires_evidence"
