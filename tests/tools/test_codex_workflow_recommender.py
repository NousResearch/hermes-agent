from agent import codex_workflow_recommender as recommender


def _review(status: str = "passed") -> dict:
    return {"status": status, "verdict": status, "passed": status == "passed"}


def _verification(status: str = "passed") -> dict:
    return {
        "risk_classes": ["docs_only"],
        "hermes_verification_commands": [
            {
                "cmd_id": "diff-check",
                "argv": ["git", "diff", "--check"],
                "exit_code": 0 if status == "passed" else 1,
                "stdout": "",
                "stderr": "",
                "start_time": "2026-06-07T00:00:00Z",
                "end_time": "2026-06-07T00:00:01Z",
                "status": status,
            }
        ],
    }


def _candidate(stage_id: str = "phase12f-next-stage-recommender") -> dict:
    return {
        "stage_id": stage_id,
        "why": "review and verification passed",
        "allowed_files": ["agent/codex_workflow_recommender.py"],
        "verify_cmd_ids": ["workflow-tool-pytest", "py-compile", "diff-check"],
    }


def test_recommender_only_recommends_by_default():
    result = recommender.build_next_stage_recommendation(
        review_result=_review(),
        verification_evidence=_verification(),
        candidate=_candidate(),
    )

    assert result["status"] == "recommended"
    assert result["recommendation"]["stage_id"] == "phase12f-next-stage-recommender"
    assert result["recommendation"]["authorization_required"] is True
    assert result["recommendation"]["non_goals"] == ["commit", "push", "deploy", "restart", "force-push"]
    assert result["advance"]["requested"] is False
    assert result["advance"]["executed"] is False


def test_recommender_requires_authorization_to_advance():
    result = recommender.build_next_stage_recommendation(
        review_result=_review(),
        verification_evidence=_verification(),
        candidate=_candidate(),
        request_advance=True,
    )

    assert result["status"] == "recommended"
    assert result["advance"]["requested"] is True
    assert result["advance"]["status"] == "blocked"
    assert result["advance"]["reason"] == "authorization_required"
    assert result["advance"]["authorization_required"] is True
    assert result["advance"]["executed"] is False


def test_recommender_blocks_when_review_unavailable():
    result = recommender.build_next_stage_recommendation(
        review_result={"status": "unavailable", "reason": "timeout"},
        verification_evidence=_verification(),
        candidate=_candidate(),
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "review_unavailable"
    assert result["recommendation"] is None


def test_recommender_blocks_when_verification_failed():
    result = recommender.build_next_stage_recommendation(
        review_result=_review(),
        verification_evidence=_verification("failed"),
        candidate=_candidate(),
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "verification_failed"
    assert result["recommendation"] is None
    assert "verification_failed:diff-check" in result["verification"]["blocking_reasons"]


def test_recommender_never_recommends_deploy_restart_force_push():
    for stage_id in ("deploy-production", "restart-gateway", "force-push-branch"):
        result = recommender.build_next_stage_recommendation(
            review_result=_review(),
            verification_evidence=_verification(),
            candidate=_candidate(stage_id),
        )

        assert result["status"] == "blocked"
        assert result["reason"] == "forbidden_stage"
        assert result["recommendation"] is None
