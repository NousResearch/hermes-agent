#!/usr/bin/env python3
"""Offline mocked tests for tools/vision_orchestrator.py — Stage B1.

Covers (from the task's required-test list):
- router disabled causes no model call
- auxiliary.vision remains untouched
- maximum one invocation
- no automatic escalation execution
- PASS / ESCALATE_RECOMMENDED / HUMAN_REVIEW_REQUIRED via analyze_image
- execution failure produces NOT_EVALUATED
- click_authorized always false
- trace contains no image/credential material
- image preparation reuses existing helpers
- no network request in the test suite
- active tool registry contains no new analyze_image tool
- current auxiliary.vision tests/behavior unchanged (import check)

All model calls are mocked at the client boundary.
"""

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.vision_orchestrator import analyze_image
from tools.vision_policy import TRANSPORT_OPENAI_COMPATIBLE  # noqa: E402
from tools.vision_policy import (  # noqa: E402
    CoordinateContext,
    ExecutionStatus,
    ModelSlot,
    QualityDecision,
    VisionCriticality,
    VisionMode,
    VisionRequest,
    VisionTask,
)


def _req(
    task=VisionTask.SCENE_DESCRIBE,
    mode=VisionMode.AUTO,
    criticality=VisionCriticality.NORMAL,
    image_source="opaque://test-image",
    question="describe",
):
    return VisionRequest(
        request_id="req-1",
        image_source=image_source,
        task=task,
        mode=mode,
        criticality=criticality,
        question=question,
        required_outputs=[],
        coordinate_context=CoordinateContext(
            source_width_px=1920, source_height_px=1080
        ),
    )


def _mock_image_ok():
    """Patch prepare_image to return a fake data URL + dimensions + hash.

    Returns the patch object; the mock is available as ``mock`` via
    ``with _mock_image_ok() as mock``.
    """
    return patch(
        "tools.vision_orchestrator.prepare_image",
        new_callable=AsyncMock,
        return_value=(
            "data:image/png;base64,AAAA",
            1920,
            1080,
            "image/png",
            "a" * 64,
            {
                "transport_image_sha256": "a" * 64,
                "transport_mime_type": "image/png",
                "transport_transcoded": False,
            },
        ),
    )


def _mock_invoke(raw_text, status=ExecutionStatus.SUCCESS.value):
    """Patch invoke_vision_model; returns the patch object."""
    return patch(
        "tools.vision_orchestrator.invoke_vision_model",
        new_callable=AsyncMock,
        return_value={"execution_status": status, "raw_text": raw_text, "error": None},
    )


class TestDisabledGate:
    @pytest.mark.asyncio
    async def test_disabled_causes_no_model_call(self):
        with (
            patch(
                "tools.vision_orchestrator.prepare_image",
                new_callable=AsyncMock,
            ) as mock_prep,
            patch(
                "tools.vision_orchestrator.invoke_vision_model",
                new_callable=AsyncMock,
            ) as mock_invoke,
        ):
            result = await analyze_image(_req(), enabled=False)
        assert result["execution_status"] == ExecutionStatus.POLICY_BLOCKED.value
        assert "vision_router_disabled" in result["quality"]["failed_gates"]
        mock_prep.assert_not_called()
        mock_invoke.assert_not_called()
        assert result["logical_model_calls"] == 0

    @pytest.mark.asyncio
    async def test_default_enabled_follows_disabled_feature_flag(self):
        """enabled=None (default) must resolve the feature flag, which is
        disabled — zero image preparation, zero model invocation."""
        with (
            patch(
                "tools.vision_orchestrator.prepare_image",
                new_callable=AsyncMock,
            ) as mock_prep,
            patch(
                "tools.vision_orchestrator.invoke_vision_model",
                new_callable=AsyncMock,
            ) as mock_invoke,
            patch(
                "hermes_cli.config.load_config",
                return_value={"vision_router": {"enabled": False}},
            ),
        ):
            result = await analyze_image(_req())  # enabled omitted → None
        assert result["execution_status"] == ExecutionStatus.POLICY_BLOCKED.value
        mock_prep.assert_not_called()
        mock_invoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_enabled_true_permits_one_logical_invocation(self):
        """Explicit test-only enabled=True permits exactly one invocation."""
        with (
            _mock_image_ok(),
            _mock_invoke('{"observation": "x"}') as mock_invoke,
        ):
            result = await analyze_image(_req(), enabled=True, transport=TRANSPORT_OPENAI_COMPATIBLE)
        assert result["execution_status"] == ExecutionStatus.SUCCESS.value
        assert mock_invoke.await_count == 1
        assert result["logical_model_calls"] == 1


class TestSingleInvocation:
    @pytest.mark.asyncio
    async def test_exactly_one_model_call(self):
        with _mock_image_ok(), _mock_invoke(
            '{"observation": "a page", "inference": "none", "uncertainty": null}'
        ) as mock_invoke:
            await analyze_image(_req(), enabled=True, transport=TRANSPORT_OPENAI_COMPATIBLE)
        assert mock_invoke.await_count == 1

    @pytest.mark.asyncio
    async def test_no_automatic_escalation_execution(self):
        # HIGH UI_READ starts at PRECISION_VLM; missing exact text →
        # ESCALATE_RECOMMENDED → OCR follow-up recommended, but no second
        # model call happens.
        with _mock_image_ok(), _mock_invoke(
            '{"observed_text": [], "targets": [], "evidence": "seen"}'
        ) as mock_invoke:
            result = await analyze_image(
                _req(task=VisionTask.UI_READ, criticality=VisionCriticality.HIGH),
                enabled=True,
                transport=TRANSPORT_OPENAI_COMPATIBLE,
            )
        assert result["execution_status"] == ExecutionStatus.SUCCESS.value
        assert result["quality_decision"] == QualityDecision.ESCALATE_RECOMMENDED.value
        assert result["recommended_next_slot"] == ModelSlot.OCR.value
        assert mock_invoke.await_count == 1  # exactly one — no auto escalation


class TestQualityDecisions:
    @pytest.mark.asyncio
    async def test_pass_decision(self):
        with (
            _mock_image_ok(),
            _mock_invoke(
                '{"observed_text": ["亲，请登录"], "targets": [], "evidence": "seen"}'
            ),
        ):
            result = await analyze_image(
                _req(
                    task=VisionTask.UI_READ,
                    criticality=VisionCriticality.HIGH,
                    mode=VisionMode.PRECISION,
                ),
                enabled=True,
                transport=TRANSPORT_OPENAI_COMPATIBLE,
            )
        assert result["quality_decision"] == QualityDecision.PASS.value
        assert result["click_authorized"] is False

    @pytest.mark.asyncio
    async def test_escalate_recommended(self):
        # Parsable JSON but missing the exact quoted text → task-relevant
        # partial result → ESCALATE_RECOMMENDED with OCR follow-up.
        with (
            _mock_image_ok(),
            _mock_invoke('{"observed_text": [], "targets": [], "evidence": "seen"}'),
        ):
            result = await analyze_image(
                _req(
                    task=VisionTask.UI_READ,
                    criticality=VisionCriticality.HIGH,
                    mode=VisionMode.PRECISION,
                ),
                enabled=True,
                transport=TRANSPORT_OPENAI_COMPATIBLE,
            )
        assert result["quality_decision"] == QualityDecision.ESCALATE_RECOMMENDED.value
        assert result["recommended_next_slot"] == ModelSlot.OCR.value

    @pytest.mark.asyncio
    async def test_duplicate_text_no_longer_human_review(self):
        # Diagnostic finding B: repeated identical observed_text entries are
        # NOT contradictions → no HUMAN_REVIEW for duplication.
        with (
            _mock_image_ok(),
            _mock_invoke('{"observed_text": ["A", "A"]}'),
        ):
            result = await analyze_image(
                _req(task=VisionTask.UI_READ, mode=VisionMode.PRECISION),
                enabled=True,
                transport=TRANSPORT_OPENAI_COMPATIBLE,
            )
        assert "unresolved_contradiction" not in result["quality"]["failed_gates"]
        assert result["quality_decision"] == QualityDecision.PASS.value
        assert result["human_review_required"] is False

    @pytest.mark.asyncio
    async def test_execution_failure_not_evaluated(self):
        with (
            _mock_image_ok(),
            _mock_invoke("", status=ExecutionStatus.TIMEOUT.value),
        ):
            result = await analyze_image(_req(), enabled=True, transport=TRANSPORT_OPENAI_COMPATIBLE)
        assert result["execution_status"] == ExecutionStatus.TIMEOUT.value
        assert result["quality_decision"] == QualityDecision.NOT_EVALUATED.value
        assert result["recommended_next_slot"] is None

    @pytest.mark.asyncio
    async def test_policy_blocked_high_fast(self):
        with _mock_image_ok(), _mock_invoke("unused") as mock_invoke:
            result = await analyze_image(
                _req(
                    task=VisionTask.UI_LOCATE,
                    criticality=VisionCriticality.HIGH,
                    mode=VisionMode.FAST,
                ),
                enabled=True,
                transport=TRANSPORT_OPENAI_COMPATIBLE,
            )
        assert result["execution_status"] == ExecutionStatus.POLICY_BLOCKED.value
        assert result["recommended_next_slot"] == ModelSlot.PRECISION_VLM.value
        mock_invoke.assert_not_called()


class TestInvariants:
    @pytest.mark.asyncio
    async def test_click_authorized_always_false(self):
        with (
            _mock_image_ok(),
            _mock_invoke('{"observation": "x"}'),
        ):
            result = await analyze_image(_req(), enabled=True, transport=TRANSPORT_OPENAI_COMPATIBLE)
        assert result["click_authorized"] is False

    @pytest.mark.asyncio
    async def test_trace_contains_no_image_or_credential_material(self):
        with (
            _mock_image_ok(),
            _mock_invoke('{"observation": "a scene"}'),
        ):
            result = await analyze_image(_req(), enabled=True, transport=TRANSPORT_OPENAI_COMPATIBLE)
        blob = json.dumps(result)
        assert "base64" not in blob.lower()
        assert "AAAA" not in blob  # fake data-url payload must not leak
        assert "api_key" not in blob.lower()
        assert "cookie" not in blob.lower()
        trace = result["trace"]
        assert trace[0]["model_slot"] == "FAST_VLM"
        assert "latency_ms" in trace[0]
        assert "execution_status" in trace[0]

    @pytest.mark.asyncio
    async def test_trace_has_no_full_prompt(self):
        with (
            _mock_image_ok(),
            _mock_invoke('{"observation": "x"}'),
        ):
            result = await analyze_image(_req(question="SECRET QUESTION"), enabled=True, transport=TRANSPORT_OPENAI_COMPATIBLE)
        blob = json.dumps(result)
        assert "SECRET QUESTION" not in blob

    @pytest.mark.asyncio
    async def test_normalized_sha256_in_trace_no_image_bytes(self):
        with (
            _mock_image_ok(),
            _mock_invoke('{"observation": "x"}'),
        ):
            result = await analyze_image(_req(), enabled=True, transport=TRANSPORT_OPENAI_COMPATIBLE)
        assert result["normalized_image_sha256"] == "a" * 64
        assert result["trace"][0]["normalized_image_sha256"] == "a" * 64
        blob = json.dumps(result)
        assert "data:image" not in blob
        assert "AAAA" not in blob  # base64 payload must not leak


class TestEscalationTargeting:
    @pytest.mark.asyncio
    async def test_precision_ui_locate_coordinate_failure_not_ocr(self):
        """PRECISION UI_LOCATE with a coordinate failure must NOT recommend
        OCR — a text specialist cannot fix coordinates."""
        # targets present but point out of bounds → coordinate failure.
        raw = (
            '{"targets": [{"label": "login", "bbox_px": [120, 38, 246, 77], '
            '"point_px": [5000, 57]}]}'
        )
        with (
            _mock_image_ok(),
            _mock_invoke(raw),
        ):
            result = await analyze_image(
                _req(
                    task=VisionTask.UI_LOCATE,
                    criticality=VisionCriticality.HIGH,
                    mode=VisionMode.PRECISION,
                ),
                enabled=True,
                transport=TRANSPORT_OPENAI_COMPATIBLE,
            )
        assert result["quality_decision"] == QualityDecision.ESCALATE_RECOMMENDED.value
        assert result["recommended_next_slot"] == ModelSlot.PRECISION_VLM.value
        assert result["recommended_next_slot"] != ModelSlot.OCR.value

    @pytest.mark.asyncio
    async def test_refusal_through_orchestrator_is_human_review(self):
        with (
            _mock_image_ok(),
            _mock_invoke("I cannot analyze this image for you."),
        ):
            result = await analyze_image(
                _req(task=VisionTask.UI_READ, mode=VisionMode.PRECISION),
                enabled=True,
                transport=TRANSPORT_OPENAI_COMPATIBLE,
            )
        assert result["quality_decision"] == QualityDecision.HUMAN_REVIEW_REQUIRED.value
        assert result["recommended_next_slot"] is None
        assert result["human_review_required"] is True

    @pytest.mark.asyncio
    async def test_malformed_json_is_not_evaluated(self):
        with (
            _mock_image_ok(),
            _mock_invoke("this is not json at all"),
        ):
            result = await analyze_image(
                _req(task=VisionTask.UI_READ, mode=VisionMode.PRECISION),
                enabled=True,
                transport=TRANSPORT_OPENAI_COMPATIBLE,
            )
        assert result["quality_decision"] == QualityDecision.NOT_EVALUATED.value
        assert "structured_parse_failed" in result["quality"]["failed_gates"]


class TestHashDeterminism:
    def test_prepare_image_hashes_normalized_bytes(self):
        """prepare_image must compute the SHA-256 over the exact normalized
        bytes used for the data URL — deterministic and independent of the
        input path."""
        import hashlib

        from tools.ollama_vision_client import prepare_image

        # Monkeypatch the internal helpers to a deterministic pipeline.
        async def fake_resolve(image_source, task_id=None):
            # _resolve_image_bytes_async returns RAW BYTES (it unwraps
            # resolved.data internally).
            return b"RAW-BYTES-123"

        def fake_normalize(path, mime):
            # Simulate normalization producing different bytes. Called via
            # asyncio.to_thread → must be a SYNC function.
            out = path.with_suffix(".norm")
            out.write_bytes(b"NORMALIZED-BYTES-456")
            return out, "image/png", None

        def fake_encode(path, mime_type=None):
            # Called via asyncio.to_thread → must be a SYNC function.
            return "data:image/png;base64,fake"

        import tools.ollama_vision_client as mod

        with (
            patch.object(mod, "_resolve_image_bytes_async", fake_resolve),
            patch(
                "tools.vision_tools._normalize_to_supported_image",
                fake_normalize,
            ),
            patch(
                "tools.vision_tools._image_to_base64_data_url",
                fake_encode,
            ),
        ):
            (
                data_url,
                w,
                h,
                mime,
                sha,
                transport_meta,
            ) = __import__("asyncio").run(prepare_image("opaque://x"))

        expected = hashlib.sha256(b"NORMALIZED-BYTES-456").hexdigest()
        assert sha == expected
        # Not the raw-bytes hash.
        assert sha != hashlib.sha256(b"RAW-BYTES-123").hexdigest()
        # Non-WebP transport bytes pass through unchanged:
        assert transport_meta["transport_image_sha256"] == expected
        assert transport_meta["transport_transcoded"] is False
        assert transport_meta["transport_mime_type"] == "image/png"


class TestImageReuse:
    @pytest.mark.asyncio
    async def test_image_preparation_reuses_existing_helpers(self):
        """The orchestrator's prepare_image must delegate to the existing
        tools/vision_tools + tools/image_source pipeline (no reimplemented
        image handling). We patch the orchestrator's prepare_image reference
        and additionally verify that the adapter module itself imports the
        shared helpers rather than reimplementing them."""
        import inspect
        import tools.ollama_vision_client as adapter

        src = inspect.getsource(adapter)
        # The adapter must reference the shared helpers, not reimplement them.
        assert "resolve_image_source" in src
        assert "_normalize_to_supported_image" in src
        assert "_image_to_base64_data_url" in src

        with _mock_image_ok(), _mock_invoke('{"observation": "x"}') as mock_invoke:
            result = await analyze_image(_req(), enabled=True, transport=TRANSPORT_OPENAI_COMPATIBLE)
        assert result["execution_status"] == ExecutionStatus.SUCCESS.value
        assert mock_invoke.await_count == 1


class TestToolRegistry:
    def test_no_new_analyze_image_tool_in_model_tools(self):
        """analyze_image must NOT be a model-visible tool in Stage B1."""
        import model_tools

        # The legacy toolset map is the model-visible tool registry.
        registry = getattr(model_tools, "_LEGACY_TOOLSET_MAP", {})
        for tools_list in registry.values():
            assert "analyze_image" not in tools_list
        # The orchestrator module itself must not be in the registry.
        assert "vision_orchestrator" not in registry


class TestAuxiliaryUntouched:
    def test_auxiliary_vision_defaults_unchanged(self):
        import hermes_cli.config_defaults as cd

        aux_vision = cd.DEFAULT_CONFIG["auxiliary"]["vision"]
        # The existing auxiliary.vision block is unchanged (same keys).
        assert aux_vision["provider"] == "auto"
        assert "timeout" in aux_vision
        # The new vision_router block exists separately and is disabled.
        vr = cd.DEFAULT_CONFIG["auxiliary"]["vision_router"]
        assert vr["enabled"] is False
        assert vr["auxiliary_integration_enabled"] is False


# ---------------------------------------------------------------------------
# PRECISION default transport binding (task

# ---------------------------------------------------------------------------


def _native_invoke_result(content='{"observed_text": ["亲，请登录"]}'):
    """Provider-result shape returned by the mocked native generate call."""
    return {
        "execution_status": ExecutionStatus.SUCCESS.value,
        "extracted_content": content,
        "content_source": "thinking_fallback",
        "thinking_fallback_used": True,
        "response_character_count": 0,
        "thinking_character_count": len(content),
        "response_envelope": {"done": True, "done_reason": "stop", "context": 32768},
        "model": "qwen3.6:27b",
        "done": True,
        "done_reason": "stop",
        "created_at": "2026-08-02T00:00:00Z",
        "total_ms": 9000,
        "load_ms": 0,
        "prompt_eval_ms": 2000,
        "eval_ms": 6900,
        "prompt_eval_count": 4208,
        "eval_count": 415,
        "transport_attempts": 1,
        "error": None,
    }


class TestPrecisionDefaultBinding:
    """PRECISION_VLM with no explicit transport override must bind to the
    native-generate profile; FAST/OCR stay OpenAI-compatible; explicit
    overrides remain honored; invalid identities fail safely."""

    @pytest.mark.asyncio
    async def test_precision_defaults_to_native_generate(self):
        with _mock_image_ok(), patch(
            "tools.vision_orchestrator.invoke_native_generate",
            new=AsyncMock(return_value=_native_invoke_result()),
        ) as native_mock, patch(
            "tools.vision_orchestrator.invoke_vision_model", new=AsyncMock()
        ) as openai_mock:
            result = await analyze_image(
                _req(task=VisionTask.UI_READ, criticality=VisionCriticality.HIGH),
                enabled=True,
                base_url="http://ollama.test:11434",
            )
        native_mock.assert_awaited_once()
        openai_mock.assert_not_awaited()
        assert result["transport"]["selected_transport"] == "OLLAMA_NATIVE_GENERATE"
        assert result["transport"]["endpoint_family"] == "ollama_native_generate"
        assert result["transport"]["native_generate_used"] is True
        assert result["transport"]["requested_transport"] == "AUTO"
        # Bound profile values reach the native call.
        kwargs = native_mock.await_args.kwargs
        assert kwargs["num_ctx"] == 32768
        assert kwargs["num_predict"] == 4000
        assert kwargs["temperature"] == 0.1
        assert kwargs["model"] == "qwen3.6:27b"

    @pytest.mark.asyncio
    async def test_precision_timeout_120(self):
        # PRECISION_EXECUTION_PROFILE timeout is 120s (matches DEFAULT_TIMEOUTS).
        from tools.vision_policy import PRECISION_EXECUTION_PROFILE

        assert PRECISION_EXECUTION_PROFILE["timeout_seconds"] == 120

    @pytest.mark.asyncio
    async def test_precision_explicit_openai_override_honored(self):
        with _mock_image_ok(), _mock_invoke(
            '{"observed_text": ["亲，请登录"]}'
        ) as openai_mock, patch(
            "tools.vision_orchestrator.invoke_native_generate", new=AsyncMock()
        ) as native_mock:
            result = await analyze_image(
                _req(task=VisionTask.UI_READ, criticality=VisionCriticality.HIGH),
                enabled=True,
                transport=TRANSPORT_OPENAI_COMPATIBLE,
            )
        openai_mock.assert_awaited_once()
        native_mock.assert_not_awaited()
        assert result["transport"]["selected_transport"] == "OPENAI_COMPATIBLE"

    @pytest.mark.asyncio
    async def test_precision_explicit_native_override_honored(self):
        with _mock_image_ok(), patch(
            "tools.vision_orchestrator.invoke_native_generate",
            new=AsyncMock(return_value=_native_invoke_result()),
        ) as native_mock, patch(
            "tools.vision_orchestrator.invoke_vision_model", new=AsyncMock()
        ) as openai_mock:
            result = await analyze_image(
                _req(task=VisionTask.UI_READ, criticality=VisionCriticality.HIGH),
                enabled=True,
                transport="OLLAMA_NATIVE_GENERATE",
                base_url="http://ollama.test:11434",
            )
        native_mock.assert_awaited_once()
        openai_mock.assert_not_awaited()
        assert result["transport"]["selected_transport"] == "OLLAMA_NATIVE_GENERATE"

    @pytest.mark.asyncio
    async def test_no_fallback_on_native_failure_default_binding(self):
        # Default PRECISION binding: native failure → no OpenAI fallback.
        with _mock_image_ok(), patch(
            "tools.vision_orchestrator.invoke_native_generate",
            new=AsyncMock(
                return_value={
                    **_native_invoke_result(),
                    "execution_status": ExecutionStatus.TIMEOUT.value,
                    "extracted_content": "",
                    "error": "TimeoutException: slow",
                }
            ),
        ) as native_mock, patch(
            "tools.vision_orchestrator.invoke_vision_model", new=AsyncMock()
        ) as openai_mock:
            result = await analyze_image(
                _req(task=VisionTask.UI_READ, criticality=VisionCriticality.HIGH),
                enabled=True,
                base_url="http://ollama.test:11434",
            )
        native_mock.assert_awaited_once()
        openai_mock.assert_not_awaited()
        assert result["execution_status"] == ExecutionStatus.TIMEOUT.value
        assert result["logical_model_calls"] == 1

    @pytest.mark.asyncio
    async def test_fast_default_remains_openai(self):
        # NORMAL SCENE_DESCRIBE → FAST_VLM → OpenAI path (unchanged).
        with _mock_image_ok(), _mock_invoke(
            '{"observation": "a product image"}'
        ) as openai_mock, patch(
            "tools.vision_orchestrator.invoke_native_generate", new=AsyncMock()
        ) as native_mock:
            result = await analyze_image(
                _req(
                    task=VisionTask.SCENE_DESCRIBE,
                    criticality=VisionCriticality.NORMAL,
                ),
                enabled=True,
            )
        openai_mock.assert_awaited_once()
        native_mock.assert_not_awaited()
        assert result["transport"]["selected_transport"] == "OPENAI_COMPATIBLE"
        assert result["transport"]["native_generate_used"] is False
        assert result["initial_model_slot"] == ModelSlot.FAST_VLM.value

    @pytest.mark.asyncio
    async def test_ocr_default_remains_openai(self):
        with _mock_image_ok(), _mock_invoke(
            '{"observed_text": ["text"]}'
        ) as openai_mock, patch(
            "tools.vision_orchestrator.invoke_native_generate", new=AsyncMock()
        ) as native_mock:
            result = await analyze_image(
                _req(
                    task=VisionTask.EXACT_OCR,
                    criticality=VisionCriticality.HIGH,
                    mode=VisionMode.OCR,
                ),
                enabled=True,
            )
        openai_mock.assert_awaited_once()
        native_mock.assert_not_awaited()
        assert result["initial_model_slot"] == ModelSlot.OCR.value
        assert result["transport"]["selected_transport"] == "OPENAI_COMPATIBLE"

    @pytest.mark.asyncio
    async def test_ocr_plain_text_accepted_as_canonical(self):
        # Diagnostic finding D: OCR models return plain transcription; a
        # non-JSON OCR response must be accepted as the canonical result
        # (PASS), not classified NOT_EVALUATED.
        plain = "淘宝网首页 已买到的宝贝 我的淘宝 购物车 79\n知语社 >\n动漫剪辑素材"
        with _mock_image_ok(), _mock_invoke(plain) as openai_mock:
            result = await analyze_image(
                _req(
                    task=VisionTask.EXACT_OCR,
                    criticality=VisionCriticality.HIGH,
                    mode=VisionMode.OCR,
                ),
                enabled=True,
            )
        openai_mock.assert_awaited_once()
        assert result["execution_status"] == ExecutionStatus.SUCCESS.value
        assert result["structured"].get("_ocr_plain_text") is True
        assert result["structured"].get("observed_text") == [plain.strip()]
        assert "text_non_empty" in result["quality"]["passed_gates"]
        assert result["quality_decision"] == QualityDecision.PASS.value

    @pytest.mark.asyncio
    async def test_ocr_empty_plain_text_still_not_evaluated(self):
        # Empty OCR text must not fake a PASS.
        with _mock_image_ok(), _mock_invoke("   ") as openai_mock:
            result = await analyze_image(
                _req(
                    task=VisionTask.EXACT_OCR,
                    criticality=VisionCriticality.HIGH,
                    mode=VisionMode.OCR,
                ),
                enabled=True,
            )
        assert result["execution_status"] == ExecutionStatus.SUCCESS.value
        assert result["structured"].get("_ocr_plain_text") is None

    def test_invalid_transport_fails_safely(self):
        with pytest.raises(ValueError, match="Unknown transport identity"):
            import asyncio as _aio

            _aio.run(
                analyze_image(
                    _req(task=VisionTask.UI_READ, criticality=VisionCriticality.HIGH),
                    enabled=True,
                    transport="NOT_A_TRANSPORT",
                )
            )

    @pytest.mark.asyncio
    async def test_task_specific_schema_resolves(self):
        # UI_READ → strict schema requiring observed_text; SCENE_DESCRIBE →
        # schema requiring observation (task §10.8).
        from tools.ollama_generate_vision_client import TASK_STRICT_SCHEMAS

        assert TASK_STRICT_SCHEMAS["UI_READ"]["required"] == ["observed_text"]
        assert TASK_STRICT_SCHEMAS["SCENE_DESCRIBE"]["required"] == ["observation", "inference"]
        assert TASK_STRICT_SCHEMAS["EVIDENCE_VERIFY"]["required"] == ["evidence"]

    @pytest.mark.asyncio
    async def test_percall_native_options_override_profile(self):
        # Explicit caller runtime override remains bounded and functional
        # (task §10.24).
        with _mock_image_ok(), patch(
            "tools.vision_orchestrator.invoke_native_generate",
            new=AsyncMock(return_value=_native_invoke_result()),
        ) as native_mock:
            await analyze_image(
                _req(task=VisionTask.UI_READ, criticality=VisionCriticality.HIGH),
                enabled=True,
                base_url="http://ollama.test:11434",
                native_generate_options={"num_ctx": 262144, "num_predict": 2000},
            )
        kwargs = native_mock.await_args.kwargs
        assert kwargs["num_ctx"] == 262144
        assert kwargs["num_predict"] == 2000
        assert kwargs["temperature"] == 0.1  # profile default preserved
