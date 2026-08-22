#!/usr/bin/env python3
"""Offline mocked tests for tools/vision_policy.py — Vision Orchestrator
Stage B1 inactive foundation.

Covers (from the task's required-test list):
- Vision-Need policy (NOT_NEEDED when DOM sufficient; REQUIRED for image-only)
- all five task types
- NORMAL/HIGH routing matrix
- AUTO/FAST/PRECISION/OCR mode behavior
- HIGH safety handling for explicit FAST
- legacy role aliases
- exact model-slot resolution from injected test configuration
- PASS / ESCALATE_RECOMMENDED / HUMAN_REVIEW_REQUIRED decisions
- execution failure → NOT_EVALUATED
- UI_READ missing required text
- UI_LOCATE valid / out-of-bounds coordinates, invalid bbox ordering
- crop-to-source coordinate mapping
- EXACT_OCR empty output
- EVIDENCE_VERIFY observation/inference separation
- click_authorized always false
- timeout configuration-driven
- no private endpoint hardcoded
- calibration schema validates its example

No network requests, no model calls, no live Ollama.
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.vision_policy import (  # noqa: E402
    DEFAULT_MODEL_SLOTS,
    DEFAULT_TIMEOUTS,
    CoordinateContext,
    ExecutionStatus,
    ModelSlot,
    PolicyBlockedError,
    QualityDecision,
    VisionCriticality,
    VisionMode,
    VisionNeedDecision,
    VisionRequest,
    VisionTask,
    build_result,
    decide_vision_need,
    evaluate_quality,
    plan_vision,
    resolve_model_slot,
    validate_coordinates,
    validate_normalized_point,
)


# ---------------------------------------------------------------------------
# Vision-Need policy
# ---------------------------------------------------------------------------


class TestVisionNeedPolicy:
    def test_dom_sufficient_returns_not_needed(self):
        assert (
            decide_vision_need(dom_sufficient=True)
            == VisionNeedDecision.VISION_NOT_NEEDED
        )

    def test_accessibility_sufficient_returns_not_needed(self):
        assert (
            decide_vision_need(accessibility_sufficient=True)
            == VisionNeedDecision.VISION_NOT_NEEDED
        )

    def test_image_only_content_returns_required(self):
        assert (
            decide_vision_need(image_only_content=True)
            == VisionNeedDecision.VISION_REQUIRED
        )

    def test_canvas_content_returns_required(self):
        assert (
            decide_vision_need(canvas_content=True)
            == VisionNeedDecision.VISION_REQUIRED
        )

    def test_screenshot_evidence_required(self):
        assert (
            decide_vision_need(screenshot_evidence_required=True)
            == VisionNeedDecision.VISION_REQUIRED
        )

    def test_precise_coordinates_required(self):
        assert (
            decide_vision_need(precise_coordinates_required=True)
            == VisionNeedDecision.VISION_REQUIRED
        )

    def test_explicit_image_request(self):
        assert (
            decide_vision_need(explicit_image_request=True)
            == VisionNeedDecision.VISION_REQUIRED
        )

    def test_no_inputs_defaults_to_not_needed(self):
        assert decide_vision_need() == VisionNeedDecision.VISION_NOT_NEEDED

    def test_dom_sufficient_wins_over_visual_state(self):
        # DOM sufficiency is the primary signal.
        assert (
            decide_vision_need(dom_sufficient=True, visual_state_required=True)
            == VisionNeedDecision.VISION_NOT_NEEDED
        )


# ---------------------------------------------------------------------------
# Task / criticality / mode matrix
# ---------------------------------------------------------------------------


def _req(task, criticality=VisionCriticality.NORMAL, mode=VisionMode.AUTO):
    return VisionRequest(
        request_id="t1",
        image_source="opaque://test",
        task=task,
        mode=mode,
        criticality=criticality,
    )


class TestPlannerMatrix:
    @pytest.mark.parametrize(
        "task,criticality,expected",
        [
            (VisionTask.SCENE_DESCRIBE, VisionCriticality.NORMAL, ModelSlot.FAST_VLM),
            (VisionTask.SCENE_DESCRIBE, VisionCriticality.HIGH, ModelSlot.PRECISION_VLM),
            (VisionTask.UI_READ, VisionCriticality.NORMAL, ModelSlot.FAST_VLM),
            (VisionTask.UI_READ, VisionCriticality.HIGH, ModelSlot.PRECISION_VLM),
            (VisionTask.UI_LOCATE, VisionCriticality.NORMAL, ModelSlot.FAST_VLM),
            (VisionTask.UI_LOCATE, VisionCriticality.HIGH, ModelSlot.PRECISION_VLM),
            (VisionTask.EXACT_OCR, VisionCriticality.NORMAL, ModelSlot.OCR),
            (VisionTask.EXACT_OCR, VisionCriticality.HIGH, ModelSlot.OCR),
            (VisionTask.EVIDENCE_VERIFY, VisionCriticality.NORMAL, ModelSlot.PRECISION_VLM),
            (VisionTask.EVIDENCE_VERIFY, VisionCriticality.HIGH, ModelSlot.PRECISION_VLM),
        ],
    )
    def test_auto_matrix(self, task, criticality, expected):
        plan = plan_vision(_req(task, criticality))
        assert plan.selected_slot == expected
        assert plan.max_model_calls == 1
        assert plan.click_authorized is False

    def test_explicit_precision_mode(self):
        plan = plan_vision(_req(VisionTask.SCENE_DESCRIBE, mode=VisionMode.PRECISION))
        assert plan.selected_slot == ModelSlot.PRECISION_VLM

    def test_explicit_fast_mode_normal_task(self):
        plan = plan_vision(_req(VisionTask.SCENE_DESCRIBE, mode=VisionMode.FAST))
        assert plan.selected_slot == ModelSlot.FAST_VLM

    def test_explicit_ocr_for_exact_ocr(self):
        plan = plan_vision(_req(VisionTask.EXACT_OCR, mode=VisionMode.OCR))
        assert plan.selected_slot == ModelSlot.OCR

    def test_explicit_ocr_for_text_ui_read(self):
        req = _req(VisionTask.UI_READ, mode=VisionMode.OCR)
        req.required_outputs = ["observed_text"]
        plan = plan_vision(req)
        assert plan.selected_slot == ModelSlot.OCR

    def test_high_fast_ui_read_policy_blocked(self):
        with pytest.raises(PolicyBlockedError) as exc_info:
            plan_vision(
                _req(
                    VisionTask.UI_READ,
                    criticality=VisionCriticality.HIGH,
                    mode=VisionMode.FAST,
                )
            )
        assert exc_info.value.recommended_slot == ModelSlot.PRECISION_VLM

    def test_high_fast_ui_locate_policy_blocked(self):
        with pytest.raises(PolicyBlockedError):
            plan_vision(
                _req(
                    VisionTask.UI_LOCATE,
                    criticality=VisionCriticality.HIGH,
                    mode=VisionMode.FAST,
                )
            )

    def test_high_fast_evidence_verify_policy_blocked(self):
        with pytest.raises(PolicyBlockedError):
            plan_vision(
                _req(
                    VisionTask.EVIDENCE_VERIFY,
                    criticality=VisionCriticality.HIGH,
                    mode=VisionMode.FAST,
                )
            )

    def test_high_fast_scene_describe_allowed(self):
        # SCENE_DESCRIBE is not in the precision-required set.
        plan = plan_vision(
            _req(
                VisionTask.SCENE_DESCRIBE,
                criticality=VisionCriticality.HIGH,
                mode=VisionMode.FAST,
            )
        )
        assert plan.selected_slot == ModelSlot.FAST_VLM

    def test_explicit_ocr_invalid_for_scene_describe(self):
        with pytest.raises(PolicyBlockedError):
            plan_vision(
                _req(VisionTask.SCENE_DESCRIBE, mode=VisionMode.OCR)
            )


# ---------------------------------------------------------------------------
# Legacy role aliases
# ---------------------------------------------------------------------------


class TestLegacyAliases:
    def test_fast_vision_model_alias(self):
        assert resolve_model_slot("FAST_VISION_MODEL") == ModelSlot.FAST_VLM

    def test_precision_vision_model_alias(self):
        assert resolve_model_slot("PRECISION_VISION_MODEL") == ModelSlot.PRECISION_VLM

    def test_layout_ocr_model_alias(self):
        assert resolve_model_slot("LAYOUT_OCR_MODEL") == ModelSlot.OCR

    def test_lowercase_alias(self):
        assert resolve_model_slot("fast_vision_model") == ModelSlot.FAST_VLM
        assert resolve_model_slot("precision_vision_model") == ModelSlot.PRECISION_VLM
        assert resolve_model_slot("layout_ocr_model") == ModelSlot.OCR

    def test_canonical_names_idempotent(self):
        assert resolve_model_slot("FAST_VLM") == ModelSlot.FAST_VLM
        assert resolve_model_slot("PRECISION_VLM") == ModelSlot.PRECISION_VLM
        assert resolve_model_slot("OCR") == ModelSlot.OCR
        assert resolve_model_slot("fast_vlm") == ModelSlot.FAST_VLM

    def test_unknown_returns_none(self):
        assert resolve_model_slot("NOT_A_SLOT") is None
        assert resolve_model_slot(None) is None


# ---------------------------------------------------------------------------
# Model-slot resolution from injected configuration
# ---------------------------------------------------------------------------


class TestModelSlotResolution:
    def test_default_catalog(self):
        plan = plan_vision(_req(VisionTask.SCENE_DESCRIBE))
        assert plan.selected_model_identity == DEFAULT_MODEL_SLOTS[ModelSlot.FAST_VLM]
        assert plan.selected_model_identity == "qwen2.5vl"

    def test_injected_configuration(self):
        custom = {
            ModelSlot.FAST_VLM: "my-fast:latest",
            ModelSlot.PRECISION_VLM: "my-precision:27b",
            ModelSlot.OCR: "my-ocr:1b",
        }
        plan = plan_vision(
            _req(VisionTask.UI_READ, criticality=VisionCriticality.HIGH),
            model_slots=custom,
        )
        assert plan.selected_model_identity == "my-precision:27b"

    def test_timeout_configuration_driven(self):
        timeouts = {ModelSlot.FAST_VLM: 60.0}
        plan = plan_vision(
            _req(VisionTask.SCENE_DESCRIBE), timeouts=timeouts
        )
        assert plan.timeout_seconds == 60.0

    def test_default_timeouts_are_ceilings(self):
        plan = plan_vision(
            _req(VisionTask.UI_READ, criticality=VisionCriticality.HIGH)
        )
        assert plan.timeout_seconds == DEFAULT_TIMEOUTS[ModelSlot.PRECISION_VLM]


# ---------------------------------------------------------------------------
# Quality gates
# ---------------------------------------------------------------------------


class TestQualityGates:
    def test_pass_decision_ui_read(self):
        q = evaluate_quality(
            VisionTask.UI_READ,
            response_text='{"observed_text": ["亲，请登录"]}',
            required_outputs=["observed_text"],
            extracted={"observed_text": ["亲，请登录"]},
        )
        assert q["quality_decision"] == QualityDecision.PASS.value

    def test_escalate_recommended_missing_required_text(self):
        # Task-relevant partial result: response text present but the exact
        # quoted text field is missing → ESCALATE_RECOMMENDED (a precision
        # follow-up can resolve it). Generic prose that is not JSON would be
        # classified as NOT_EVALUATED instead.
        q = evaluate_quality(
            VisionTask.UI_READ,
            response_text='{"observed_text": [], "targets": [], "evidence": "seen"}',
            required_outputs=["observed_text"],
            extracted={"observed_text": [], "targets": [], "evidence": "seen"},
        )
        assert q["quality_decision"] == QualityDecision.ESCALATE_RECOMMENDED.value

    def test_generic_prose_not_escalation_candidate(self):
        # Non-JSON generic prose → structured parse failure → NOT_EVALUATED,
        # never ESCALATE_RECOMMENDED.
        q = evaluate_quality(
            VisionTask.UI_READ,
            response_text="This is a beautiful screenshot with many colors.",
            required_outputs=["observed_text"],
            extracted={"_parse_failed": True, "raw_text": "..."},
        )
        assert q["quality_decision"] == QualityDecision.NOT_EVALUATED.value
        assert "structured_parse_failed" in q["failed_gates"]

    def test_refusal_not_escalation_candidate(self):
        q = evaluate_quality(
            VisionTask.UI_READ,
            response_text="I cannot analyze this image for you.",
            required_outputs=["observed_text"],
            extracted={"raw_text": "I cannot analyze this image for you."},
        )
        assert q["quality_decision"] == QualityDecision.HUMAN_REVIEW_REQUIRED.value
        assert "refusal_or_inability" in q["failed_gates"]

    def test_chinese_refusal_not_escalation_candidate(self):
        q = evaluate_quality(
            VisionTask.UI_READ,
            response_text="我无法分析这张图片，请提供其他输入。",
            required_outputs=["observed_text"],
            extracted={"raw_text": "我无法分析这张图片"},
        )
        assert q["quality_decision"] == QualityDecision.HUMAN_REVIEW_REQUIRED.value

    def test_duplicate_text_not_contradiction(self):
        # Diagnostic finding B: repeated identical entries are NOT
        # contradictions — UI screenshots legitimately repeat labels.
        q = evaluate_quality(
            VisionTask.UI_READ,
            response_text='{"observed_text": ["A", "A"]}',
            required_outputs=["observed_text"],
            extracted={"observed_text": ["A", "A"]},
        )
        assert "unresolved_contradiction" not in q["failed_gates"]
        assert q["quality_decision"] == QualityDecision.PASS.value

    def test_explicit_contradiction_field_triggers_human_review(self):
        # The only reliable contradiction signal is an explicit
        # model-flagged `contradiction` field (EVIDENCE_VERIFY schema).
        q = evaluate_quality(
            VisionTask.EVIDENCE_VERIFY,
            response_text='{"evidence": "x", "contradiction": "claim conflicts"}',
            required_outputs=["evidence"],
            extracted={"evidence": "x", "contradiction": "claim conflicts"},
        )
        assert "unresolved_contradiction" in q["failed_gates"]
        assert q["quality_decision"] == QualityDecision.HUMAN_REVIEW_REQUIRED.value

    def test_not_evaluated_on_empty_response(self):
        q = evaluate_quality(
            VisionTask.UI_READ,
            response_text="   ",
            required_outputs=["observed_text"],
            extracted={},
        )
        assert q["quality_decision"] == QualityDecision.NOT_EVALUATED.value
        assert "response_present" in q["failed_gates"]

    def test_ui_locate_valid_coordinates_pass(self):
        cc = CoordinateContext(source_width_px=1920, source_height_px=1080)
        q = evaluate_quality(
            VisionTask.UI_LOCATE,
            response_text='{"targets": [{"label": "login", "bbox_px": [120, 38, 246, 77], "point_px": [183, 57]}]}',
            required_outputs=["targets"],
            extracted={
                "targets": [{"label": "login", "bbox_px": [120, 38, 246, 77], "point_px": [183, 57]}]
            },
            coordinates={
                "x": 183.0,
                "y": 57.0,
                "bbox": [120.0, 38.0, 246.0, 77.0],
                "coordinate_space": "SOURCE_IMAGE_PIXELS",
            },
            coordinate_context=cc,
        )
        assert q["quality_decision"] == QualityDecision.PASS.value

    def test_ui_locate_out_of_bounds_coordinates(self):
        cc = CoordinateContext(source_width_px=1920, source_height_px=1080)
        q = evaluate_quality(
            VisionTask.UI_LOCATE,
            response_text='{"targets": [{"label": "x", "point_px": [5000, 57]}]}',
            required_outputs=["targets"],
            extracted={"targets": [{"label": "x", "point_px": [5000, 57]}]},
            coordinates={"x": 5000.0, "y": 57.0, "coordinate_space": "SOURCE_IMAGE_PIXELS"},
            coordinate_context=cc,
        )
        assert "x_out_of_bounds" in q["failed_gates"]
        assert q["quality_decision"] != QualityDecision.PASS.value

    def test_ui_locate_invalid_bbox_ordering(self):
        cc = CoordinateContext(source_width_px=1920, source_height_px=1080)
        q = evaluate_quality(
            VisionTask.UI_LOCATE,
            response_text='{"targets": [{"label": "x", "bbox_px": [300, 100, 100, 200]}]}',
            required_outputs=["targets"],
            extracted={"targets": [{"label": "x", "bbox_px": [300, 100, 100, 200]}]},
            coordinates={"bbox": [300.0, 100.0, 100.0, 200.0], "coordinate_space": "SOURCE_IMAGE_PIXELS"},
            coordinate_context=cc,
        )
        assert "bbox_left_ge_right" in q["failed_gates"]

    def test_crop_to_source_coordinate_mapping(self):
        # Crop origin (400, 200); a point at crop-local (100, 50) maps to
        # source (500, 250) — still in bounds for a 1920x1080 image.
        failures = validate_coordinates(
            x=100.0,
            y=50.0,
            width=1920,
            height=1080,
            crop_x=400,
            crop_y=200,
        )
        assert failures == []

    def test_crop_to_source_mapping_out_of_bounds(self):
        # Crop origin (2000, 0); crop-local x=100 → source 2100 > 1920.
        failures = validate_coordinates(
            x=100.0,
            y=0.0,
            width=1920,
            height=1080,
            crop_x=2000,
            crop_y=0,
        )
        assert "x_out_of_bounds" in failures

    def test_negative_crop_origin_rejected(self):
        failures = validate_coordinates(
            x=10.0, y=10.0, width=1920, height=1080,
            crop_x=-5, crop_y=0,
        )
        assert "crop_origin_negative" in failures

    def test_crop_exceeds_source_rejected(self):
        failures = validate_coordinates(
            x=10.0, y=10.0, width=1920, height=1080,
            crop_x=1900, crop_y=0, crop_width=100, crop_height=100,
        )
        assert "crop_exceeds_source_width" in failures

    def test_nonpositive_crop_dimensions_rejected(self):
        failures = validate_coordinates(
            x=10.0, y=10.0, width=1920, height=1080,
            crop_x=0, crop_y=0, crop_width=0, crop_height=100,
        )
        assert "crop_width_nonpositive" in failures

    def test_zero_area_bbox_rejected(self):
        failures = validate_coordinates(
            x=10.0, y=10.0, width=1920, height=1080,
            bbox=[100.0, 100.0, 100.0, 200.0],
        )
        assert "bbox_left_ge_right" in failures

    def test_missing_source_dimensions_with_coordinates_rejected(self):
        failures = validate_coordinates(
            x=10.0, y=10.0, width=None, height=None,
        )
        assert "coordinate_space_missing_dimensions" in failures

    def test_nonpositive_source_dimensions_rejected(self):
        failures = validate_coordinates(
            x=10.0, y=10.0, width=0, height=1080,
        )
        assert "coordinate_space_nonpositive_dimensions" in failures

    def test_boundary_point_half_open_convention(self):
        # HALF-OPEN: a point at x == width or y == height is INVALID.
        failures = validate_coordinates(
            x=1920.0, y=1080.0, width=1920, height=1080,
        )
        assert "x_out_of_bounds" in failures
        assert "y_out_of_bounds" in failures
        # The last valid pixel is (W-1, H-1).
        ok = validate_coordinates(
            x=1919.0, y=1079.0, width=1920, height=1080,
        )
        assert ok == []

    def test_full_image_bbox_half_open_valid(self):
        # [0, 0, W, H] is a valid full-image bbox under half-open edges.
        failures = validate_coordinates(
            x=100.0, y=100.0, width=1920, height=1080,
            bbox=[0.0, 0.0, 1920.0, 1080.0],
        )
        assert failures == []

    def test_bbox_right_edge_exclusive_for_point(self):
        # Point at the bbox right edge (x == right) is OUTSIDE.
        failures = validate_coordinates(
            x=246.0, y=57.0, width=1920, height=1080,
            bbox=[120.0, 38.0, 246.0, 77.0],
        )
        assert "point_outside_bbox" in failures

    def test_bbox_bottom_edge_exclusive_for_point(self):
        failures = validate_coordinates(
            x=183.0, y=77.0, width=1920, height=1080,
            bbox=[120.0, 38.0, 246.0, 77.0],
        )
        assert "point_outside_bbox" in failures

    def test_crop_ending_exactly_at_source_valid(self):
        # Crop ending exactly at W/H is valid (half-open).
        failures = validate_coordinates(
            x=0.0, y=0.0, width=1920, height=1080,
            crop_x=0, crop_y=0, crop_width=1920, crop_height=1080,
        )
        assert failures == []

    def test_crop_extending_one_unit_beyond_invalid(self):
        failures = validate_coordinates(
            x=0.0, y=0.0, width=1920, height=1080,
            crop_x=0, crop_y=0, crop_width=1921, crop_height=1080,
        )
        assert "crop_exceeds_source_width" in failures

    def test_normalized_point_half_open(self):
        assert validate_normalized_point(0.0, 0.0) == []
        assert validate_normalized_point(0.999, 0.5) == []
        assert "x_norm_out_of_bounds" in validate_normalized_point(1.0, 0.5)
        assert "y_norm_out_of_bounds" in validate_normalized_point(0.5, 1.0)

    def test_point_w_hminus1_invalid(self):
        # (W, H-1): x == W is invalid.
        failures = validate_coordinates(
            x=1920.0, y=1079.0, width=1920, height=1080,
        )
        assert "x_out_of_bounds" in failures

    def test_point_wminus1_h_invalid(self):
        # (W-1, H): y == H is invalid.
        failures = validate_coordinates(
            x=1919.0, y=1080.0, width=1920, height=1080,
        )
        assert "y_out_of_bounds" in failures

    def test_bbox_right_gt_width_invalid(self):
        failures = validate_coordinates(
            x=100.0, y=100.0, width=1920, height=1080,
            bbox=[0.0, 0.0, 1921.0, 1080.0],
        )
        assert "bbox_x_out_of_bounds" in failures

    def test_bbox_bottom_gt_height_invalid(self):
        failures = validate_coordinates(
            x=100.0, y=100.0, width=1920, height=1080,
            bbox=[0.0, 0.0, 1920.0, 1081.0],
        )
        assert "bbox_y_out_of_bounds" in failures

    def test_crop_to_source_mapping_preserves_half_open(self):
        # Crop origin (400, 200); crop-local point at the crop's last valid
        # pixel (crop_width-1, crop_height-1) maps inside the source.
        failures = validate_coordinates(
            x=1919.0, y=879.0, width=1920, height=1080,
            crop_x=400, crop_y=200, crop_width=1520, crop_height=880,
        )
        # 1919 + 400 = 2319 > 1920 → the mapped point is OUT of the source
        # (a crop-local x of 1919 exceeds the crop's valid range). The
        # half-open mapping contract rejects it.
        assert "x_out_of_bounds" in failures

    def test_exact_ocr_empty_output(self):
        q = evaluate_quality(
            VisionTask.EXACT_OCR,
            response_text='{"observed_text": ""}',
            required_outputs=["observed_text"],
            extracted={"observed_text": ""},
        )
        assert "text_non_empty" in q["failed_gates"]

    def test_evidence_verify_observation_inference_separation(self):
        q = evaluate_quality(
            VisionTask.EVIDENCE_VERIFY,
            response_text='{"observation": "login text visible", "inference": "user is not logged in", "evidence": "text at top right"}',
            required_outputs=["observation", "evidence"],
            extracted={
                "observation": "login text visible",
                "inference": "user is not logged in",
                "evidence": "text at top right",
            },
        )
        assert q["quality_decision"] == QualityDecision.PASS.value

    def test_evidence_verify_action_claimed(self):
        q = evaluate_quality(
            VisionTask.EVIDENCE_VERIFY,
            response_text='{"observation": "x", "evidence": "y", "action_claimed": true}',
            required_outputs=["observation", "evidence"],
            extracted={"observation": "x", "evidence": "y", "action_claimed": True},
        )
        assert "no_action_claimed" in q["failed_gates"]


# ---------------------------------------------------------------------------
# Build-result invariants
# ---------------------------------------------------------------------------


class TestResultInvariants:
    def test_click_authorized_always_false(self):
        r = build_result(
            execution_status=ExecutionStatus.SUCCESS,
            quality={"quality_decision": QualityDecision.PASS.value, "passed_gates": [], "failed_gates": []},
            initial_slot=ModelSlot.FAST_VLM,
            final_slot=ModelSlot.FAST_VLM,
            actual_model="qwen2.5vl",
        )
        assert r["click_authorized"] is False

    def test_result_serializes(self):
        r = build_result(
            execution_status=ExecutionStatus.SUCCESS,
            quality={"quality_decision": QualityDecision.PASS.value, "passed_gates": ["a"], "failed_gates": []},
            initial_slot=ModelSlot.FAST_VLM,
            final_slot=ModelSlot.FAST_VLM,
            actual_model="qwen2.5vl",
        )
        s = json.dumps(r, ensure_ascii=False, default=str)
        parsed = json.loads(s)
        assert parsed["click_authorized"] is False


# ---------------------------------------------------------------------------
# UI_READ canonical text-field contract
# ---------------------------------------------------------------------------


class TestUiReadTextFieldContract:
    """Canonical field is ``observed_text`` (Prompt contract + gates +
    calibration); ``visible_text`` is a legacy alias accepted ONLY through
    the explicit resolver (``_resolve_ui_read_text``)."""

    def _q(self, extracted, response_text=None):
        return evaluate_quality(
            VisionTask.UI_READ,
            response_text=response_text
            or json.dumps(extracted, ensure_ascii=False),
            required_outputs=["observed_text"],
            extracted=extracted,
        )

    def test_canonical_observed_text_nonempty_passes(self):
        q = self._q({"observed_text": ["亲，请登录"]})
        assert "required_text_present" in q["passed_gates"]
        assert q["quality_decision"] == QualityDecision.PASS.value

    def test_legacy_visible_text_alias_passes_through_explicit_resolver(self):
        # Legacy alias: model emitted visible_text only (old strict-schema
        # contract). Resolved at the one explicit boundary — gate passes.
        q = self._q({"visible_text": ["亲，请登录"]})
        assert "required_text_present" in q["passed_gates"]
        assert "text_field_conflict" not in q["failed_gates"]

    def test_missing_canonical_field_fails(self):
        q = self._q({"evidence": "seen"})
        assert "required_text_present" in q["failed_gates"]

    def test_empty_canonical_field_fails(self):
        q = self._q({"observed_text": []})
        assert "required_text_present" in q["failed_gates"]

    def test_whitespace_only_canonical_field_fails(self):
        q = self._q({"observed_text": ["   ", ""]})
        assert "required_text_present" in q["failed_gates"]

    def test_non_string_canonical_field_fails_safely(self):
        q = self._q({"observed_text": {"not": "a list"}})
        assert "required_text_present" in q["failed_gates"]
        assert "text_field_conflict" not in q["failed_gates"]

    def test_identical_dual_fields_pass(self):
        # Both fields present with identical normalized values → accept.
        q = self._q(
            {
                "observed_text": ["亲，请登录"],
                "visible_text": ["亲，请登录"],
            }
        )
        assert "required_text_present" in q["passed_gates"]
        assert "text_field_conflict" not in q["failed_gates"]

    def test_conflicting_dual_fields_are_rejected(self):
        # Materially different values under both keys → never silently
        # choose; marked contradiction, gate fails.
        q = self._q(
            {
                "observed_text": ["亲，请登录"],
                "visible_text": ["已买到的宝贝"],
            }
        )
        assert "required_text_present" in q["failed_gates"]
        assert "text_field_conflict" in q["failed_gates"]

    def test_conflicting_dual_fields_never_escalate(self):
        # Contradiction is a human-review case, not a safe escalation.
        q = self._q(
            {
                "observed_text": ["A"],
                "visible_text": ["B"],
            }
        )
        assert q["quality_decision"] == QualityDecision.HUMAN_REVIEW_REQUIRED.value
        assert "text_field_conflict" in q["failed_gates"]


class TestNoHardcodedEndpoint:
    def test_no_private_endpoint_in_repository_defaults(self):
        import hermes_cli.config_defaults as cd

        blob = json.dumps(cd.DEFAULT_CONFIG)
        assert "192.168.2.21" not in blob
        assert "11434" not in blob

    def test_no_private_endpoint_in_policy_module(self):
        import tools.vision_policy as vp

        src = Path(vp.__file__).read_text(encoding="utf-8")
        assert "192.168.2.21" not in src
        assert "11434" not in src
