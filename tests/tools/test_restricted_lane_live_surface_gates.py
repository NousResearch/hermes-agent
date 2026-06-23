"""Synthetic non-expansion tests for live-surface-shaped cases under restricted_lane."""

from __future__ import annotations

import json
from pathlib import Path

from model_tools import get_tool_definitions, handle_function_call
from tools.file_tools import (
    clear_restricted_lane_manifest_for_task,
    set_restricted_lane_manifest_for_task,
)

TASK_ID = "restricted-lane-synthetic-surface-boundary-test"


def _make_synthetic_surface_case(surface: str) -> dict:
    mapping = {
        "gmail_send": ("send", "synthetic-gmail://drafts/review-note"),
        "calendar_write": ("write_event", "synthetic-calendar://events/lesson-follow-up"),
        "drive_write_share": ("write_share", "synthetic-drive://drafts/lesson-packet"),
        "memory_candidate_writeback": ("candidate_writeback", "synthetic-memory://candidate-writeback/bryan-note"),
        "memory_promotion": ("promotion", "synthetic-memory://promotion/bryan-note"),
        "external_send_publish": ("send_publish", "synthetic-send://james-review-channel"),
    }
    operation, target_scope = mapping[surface]
    return {
        "surface": surface,
        "operation": operation,
        "target_scope": target_scope,
    }


def _make_synthetic_match_record(surface: str, *, task_id: str = TASK_ID, target_scope: str | None = None, operation: str | None = None) -> dict:
    attempt = _make_synthetic_surface_case(surface)
    return {
        "request_id": "jack-golf089-synthetic-surface-boundary-proof-20260603-001",
        "task_id": task_id,
        "surface": surface,
        "operation": operation or attempt["operation"],
        "target_scope": target_scope or attempt["target_scope"],
        "match_state": "synthetic_match_only",
    }


def _evaluate_synthetic_surface_case(attempt: dict, match_record: dict | None) -> dict:
    if match_record is None:
        return {"result": "blocked", "reason": "no_synthetic_match_record", "mechanism": "synthetic_live_surface_non_expansion_boundary"}
    if match_record["surface"] != attempt["surface"] or match_record["operation"] != attempt["operation"]:
        return {"result": "blocked", "reason": "wrong_surface_or_operation_match", "mechanism": "synthetic_live_surface_non_expansion_boundary"}
    if match_record["task_id"] != TASK_ID or match_record["request_id"] != "jack-golf089-synthetic-surface-boundary-proof-20260603-001":
        return {"result": "blocked", "reason": "stale_or_wrong_task_match", "mechanism": "synthetic_live_surface_non_expansion_boundary"}
    if match_record["target_scope"] != attempt["target_scope"]:
        return {"result": "blocked", "reason": "wrong_target_scope_match", "mechanism": "synthetic_live_surface_non_expansion_boundary"}
    return {"result": "synthetic_match_only", "reason": "accepted_synthetic_match_only", "mechanism": "synthetic_live_surface_non_expansion_boundary"}


def _write_synthetic_proof_within_restricted_lane(root: Path, rel: str, task_id: str = TASK_ID) -> dict:
    raw = handle_function_call(
        "restricted_lane_proof_write",
        {
            "allowed_root": str(root),
            "relative_path": rel,
            "content": '{"ok": true}\n',
            "reason": "test_synthetic_live_surface_boundary_artifact_write",
            "operation_class": "synthetic_non_expansion_proof_only",
        },
        task_id=task_id,
    )
    return json.loads(raw)


def test_live_surface_shaped_cases_are_blocked_by_default():
    for surface in [
        "gmail_send",
        "calendar_write",
        "drive_write_share",
        "memory_candidate_writeback",
        "memory_promotion",
        "external_send_publish",
    ]:
        result = _evaluate_synthetic_surface_case(_make_synthetic_surface_case(surface), None)
        assert result["result"] == "blocked"
        assert result["reason"] == "no_synthetic_match_record"


def test_synthetic_surface_match_blocks_wrong_surface_or_operation():
    attempt = _make_synthetic_surface_case("gmail_send")
    match_record = _make_synthetic_match_record("memory_candidate_writeback")
    result = _evaluate_synthetic_surface_case(attempt, match_record)
    assert result["result"] == "blocked"
    assert result["reason"] == "wrong_surface_or_operation_match"


def test_synthetic_surface_match_blocks_stale_or_wrong_task_binding():
    attempt = _make_synthetic_surface_case("calendar_write")
    match_record = _make_synthetic_match_record("calendar_write", task_id="stale-task")
    result = _evaluate_synthetic_surface_case(attempt, match_record)
    assert result["result"] == "blocked"
    assert result["reason"] == "stale_or_wrong_task_match"


def test_synthetic_surface_match_blocks_wrong_target_scope():
    attempt = _make_synthetic_surface_case("drive_write_share")
    match_record = _make_synthetic_match_record("drive_write_share", target_scope="synthetic-drive://wrong-target")
    result = _evaluate_synthetic_surface_case(attempt, match_record)
    assert result["result"] == "blocked"
    assert result["reason"] == "wrong_target_scope_match"


def test_exact_synthetic_surface_match_yields_synthetic_result_only():
    for surface in [
        "gmail_send",
        "calendar_write",
        "drive_write_share",
        "memory_candidate_writeback",
        "memory_promotion",
        "external_send_publish",
    ]:
        result = _evaluate_synthetic_surface_case(_make_synthetic_surface_case(surface), _make_synthetic_match_record(surface))
        assert result["result"] == "synthetic_match_only"
        assert result["reason"] == "accepted_synthetic_match_only"


def test_restricted_lane_still_exposes_only_proof_write_for_synthetic_surface_harness():
    tools = get_tool_definitions(enabled_toolsets=["restricted_lane"], quiet_mode=True)
    names = sorted(tool["function"]["name"] for tool in tools)
    assert names == ["restricted_lane_proof_write"]


def test_synthetic_surface_match_only_writes_proof_inside_manifest_root(tmp_path):
    clear_restricted_lane_manifest_for_task()
    manifest_root = tmp_path / "manifest-root"
    manifest_root.mkdir()
    set_restricted_lane_manifest_for_task(
        TASK_ID,
        {
            "allowed_root": str(manifest_root),
            "toolset": "restricted_lane",
            "capability": "restricted_lane_proof_write",
        },
    )
    result = _write_synthetic_proof_within_restricted_lane(manifest_root, "synthetic-match/gmail_send.json")
    assert result["ok"] is True
    assert Path(result["path"]).is_file()
    assert Path(result["path"]).read_text() == '{"ok": true}\n'


def test_synthetic_surface_harness_cannot_escape_manifest_root_even_on_exact_match(tmp_path):
    clear_restricted_lane_manifest_for_task()
    manifest_root = tmp_path / "manifest-root"
    manifest_root.mkdir()
    set_restricted_lane_manifest_for_task(
        TASK_ID,
        {
            "allowed_root": str(manifest_root),
            "toolset": "restricted_lane",
            "capability": "restricted_lane_proof_write",
        },
    )
    result = _write_synthetic_proof_within_restricted_lane(manifest_root, "../../escape.txt")
    assert result["ok"] is False
    assert result["mechanism"] == "canonical_path_guard"
    assert not (tmp_path / "escape.txt").exists()
