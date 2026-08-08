#!/usr/bin/env python3
"""Offline mocked tests for the native Ollama ``/api/generate`` transport
.

Covers the 33-item offline test contract (task §14) at both layers:

- client layer (``tools/ollama_generate_vision_client``): request payload
  shape, deterministic response/thinking extraction, timing normalization,
  error classification — ``_http_post_json`` is mocked;
- orchestrator integration (``tools/vision_orchestrator.analyze_image``):
  transport selection, shared parser/quality gates, no automatic fallback /
  escalation, default transport unchanged.

All network calls are mocked. No offline test contacts Ollama.
"""
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.ollama_generate_vision_client import (  # noqa: E402
    NATIVE_GENERATE_DEFAULT_PROFILE,
    NATIVE_GENERATE_STRICT_SCHEMA,
    TRANSPORT_OLLAMA_NATIVE_GENERATE,
    TRANSPORT_OPENAI_COMPATIBLE,
    _http_post_json,
    invoke_native_generate,
)
from tools.vision_orchestrator import analyze_image  # noqa: E402
from tools.vision_policy import (  # noqa: E402
    ExecutionStatus,
    QualityDecision,
    VisionCriticality,
    VisionMode,
    VisionRequest,
    VisionTask,
)

pytestmark = pytest.mark.asyncio

# Realistic qwen3.6 thinking-field payload (gold contract of the accepted
# experiment: TAOBAO-VISION-0003, UI_READ/HIGH/observed_text — canonical
# field; visible_text retained as legacy alias with identical values).
_THINKING_JSON = json.dumps(
    {
        "observed_text": ["26年动漫MAD镜头剪辑素材", "知语社", "￥18.6"],
        "visible_text": ["26年动漫MAD镜头剪辑素材", "知语社", "￥18.6"],
        "evidence": "顶部商品标题、店铺名与价格直接可见。",
        "uncertainty": "无。",
        "inference": "无。",
    },
    ensure_ascii=False,
)

_RAW_B64 = "iVBORw0KGgoAAAANSUhEUg=="  # fake but prefix-free base64


def _env(**over):
    """Native generate response envelope (mirrors the validated real shape)."""
    env = {
        "model": "qwen3.6:27b",
        "created_at": "2026-08-02T00:00:00.000Z",
        "response": "",
        "thinking": _THINKING_JSON,
        "done": True,
        "done_reason": "stop",
        "context": 32768,
        "total_duration": 14546069900,
        "load_duration": 5034614500,
        "prompt_eval_count": 4208,
        "prompt_eval_duration": 2444031000,
        "eval_count": 525,
        "eval_duration": 6968938000,
    }
    env.update(over)
    return env


async def _invoke(**over):
    base = dict(
        model="qwen3.6:27b",
        prompt="PROMPT",
        image_raw_base64=_RAW_B64,
        num_ctx=32768,
        num_predict=4000,
        temperature=0.1,
        seed=42,
        format_spec="json",
        base_url="http://ollama.test:11434",
        timeout_seconds=120.0,
        transport_retries=0,
    )
    base.update(over)
    return await invoke_native_generate(**base)


@pytest.fixture
def mock_post():
    with patch(
        "tools.ollama_generate_vision_client._http_post_json",
        new=AsyncMock(return_value=(200, _env())),
    ) as m:
        yield m


def _last_payload(mock_post):
    call = mock_post.await_args
    assert call is not None
    return call.args[2]  # (base_url, path, payload, timeout_seconds)


# ---------------------------------------------------------------------------
# §14.1-3 — request path, stream=false, raw base64 (client layer).
# ---------------------------------------------------------------------------


class TestNativeRequestPayload:
    async def test_request_uses_generate_endpoint(self, mock_post):
        await _invoke()
        assert mock_post.await_args.args[1] == "/api/generate"

    async def test_request_stream_false(self, mock_post):
        await _invoke()
        assert _last_payload(mock_post)["stream"] is False

    async def test_request_images_raw_base64_no_prefix(self, mock_post):
        await _invoke()
        images = _last_payload(mock_post)["images"]
        assert images == [_RAW_B64]
        assert not images[0].startswith("data:")

    async def test_request_num_ctx_explicit(self, mock_post):
        await _invoke()
        assert _last_payload(mock_post)["options"]["num_ctx"] == 32768

    async def test_request_num_predict_explicit(self, mock_post):
        await _invoke()
        assert _last_payload(mock_post)["options"]["num_predict"] == 4000

    async def test_request_temperature_and_seed_explicit(self, mock_post):
        await _invoke()
        opts = _last_payload(mock_post)["options"]
        assert opts["temperature"] == 0.1
        assert opts["seed"] == 42

    async def test_request_seed_omitted_when_none(self, mock_post):
        await _invoke(seed=None)
        assert "seed" not in _last_payload(mock_post)["options"]

    async def test_request_format_json_object(self, mock_post):
        await _invoke(format_spec="json")
        assert _last_payload(mock_post)["format"] == "json"

    async def test_request_format_json_schema(self, mock_post):
        await _invoke(format_spec=NATIVE_GENERATE_STRICT_SCHEMA)
        assert _last_payload(mock_post)["format"] is NATIVE_GENERATE_STRICT_SCHEMA


# ---------------------------------------------------------------------------
# §14.11-16 — deterministic response/thinking extraction (client layer).
# ---------------------------------------------------------------------------


class TestExtraction:
    async def test_nonempty_response_is_primary(self, mock_post):
        mock_post.return_value = (
            200,
            _env(response='{"visible_text": ["from-response"]}'),
        )
        result = await _invoke()
        assert result["execution_status"] == ExecutionStatus.SUCCESS.value
        assert result["extracted_content"] == '{"visible_text": ["from-response"]}'
        assert result["content_source"] == "response"
        assert result["thinking_fallback_used"] is False

    async def test_empty_response_thinking_fallback(self, mock_post):
        result = await _invoke()  # default env: response empty, thinking set
        assert result["execution_status"] == ExecutionStatus.SUCCESS.value
        assert result["extracted_content"] == _THINKING_JSON
        assert result["content_source"] == "thinking_fallback"
        assert result["thinking_fallback_used"] is True

    async def test_response_precedence_when_both_nonempty(self, mock_post):
        mock_post.return_value = (
            200,
            _env(
                response='{"visible_text": ["resp"]}',
                thinking='{"visible_text": ["think"]}',
            ),
        )
        result = await _invoke()
        assert result["extracted_content"] == '{"visible_text": ["resp"]}'
        assert result["content_source"] == "response"
        assert result["thinking_fallback_used"] is False

    async def test_both_empty_invalid_response(self, mock_post):
        mock_post.return_value = (200, _env(response="", thinking=""))
        result = await _invoke()
        assert result["execution_status"] == ExecutionStatus.INVALID_RESPONSE.value
        assert result["error"] == "missing_response_and_thinking"

    async def test_non_string_response_rejected_safely(self, mock_post):
        mock_post.return_value = (200, _env(response={"nested": True}))
        result = await _invoke()
        assert result["execution_status"] == ExecutionStatus.INVALID_RESPONSE.value
        assert result["error"] == "non_string_response"

    async def test_non_string_thinking_rejected_safely(self, mock_post):
        mock_post.return_value = (200, _env(thinking=["not", "a", "string"]))
        result = await _invoke()
        assert result["execution_status"] == ExecutionStatus.INVALID_RESPONSE.value
        assert result["error"] == "non_string_thinking"


# ---------------------------------------------------------------------------
# §14.21-24 — timing metadata and error classification (client layer).
# ---------------------------------------------------------------------------


class TestTimingAndMetadata:
    async def test_timing_normalized_to_ms(self, mock_post):
        result = await _invoke()
        assert result["total_ms"] == 14546  # 14546069900 ns
        assert result["load_ms"] == 5034  # 5034614500 ns → int(5034.61)
        assert result["prompt_eval_ms"] == 2444
        assert result["eval_ms"] == 6968  # 6968938000 ns → int(6968.94)
        assert result["prompt_eval_count"] == 4208
        assert result["eval_count"] == 525

    async def test_done_reason_and_model_preserved(self, mock_post):
        result = await _invoke()
        assert result["done_reason"] == "stop"
        assert result["done"] is True
        assert result["model"] == "qwen3.6:27b"

    async def test_envelope_meta_excludes_private_content(self, mock_post):
        result = await _invoke()
        env = result["response_envelope"]
        assert "response" not in env
        assert "thinking" not in env
        assert env["context"] == 32768

    async def test_timeout_classified(self):
        with patch(
            "tools.ollama_generate_vision_client._http_post_json",
            new=AsyncMock(side_effect=httpx.TimeoutException("slow")),
        ):
            result = await _invoke()
        assert result["execution_status"] == ExecutionStatus.TIMEOUT.value
        assert result["transport_attempts"] == 1

    async def test_connect_error_classified_endpoint_unavailable(self):
        with patch(
            "tools.ollama_generate_vision_client._http_post_json",
            new=AsyncMock(side_effect=httpx.ConnectError("refused")),
        ):
            result = await _invoke()
        assert result["execution_status"] == ExecutionStatus.ENDPOINT_UNAVAILABLE.value

    async def test_http_500_classified_endpoint_unavailable(self, mock_post):
        mock_post.return_value = (500, {"error": "internal"})
        result = await _invoke()
        assert result["execution_status"] == ExecutionStatus.ENDPOINT_UNAVAILABLE.value

    async def test_http_404_classified_model_not_found(self, mock_post):
        mock_post.return_value = (404, {"error": "model 'qwen3.6:27b' not found"})
        result = await _invoke()
        assert result["execution_status"] == ExecutionStatus.MODEL_NOT_FOUND.value

    async def test_http_400_classified_invalid_response(self, mock_post):
        mock_post.return_value = (400, {"error": "bad request"})
        result = await _invoke()
        assert result["execution_status"] == ExecutionStatus.INVALID_RESPONSE.value

    async def test_non_json_envelope_invalid_response(self, mock_post):
        mock_post.return_value = (200, None)
        result = await _invoke()
        assert result["execution_status"] == ExecutionStatus.INVALID_RESPONSE.value
        assert result["error"] == "malformed_json_envelope"

    async def test_transport_retries_bounded_when_enabled(self):
        # Default profile keeps retries=0; when explicitly enabled, a
        # transient failure is retried exactly once then classified.
        with patch(
            "tools.ollama_generate_vision_client._http_post_json",
            new=AsyncMock(side_effect=[httpx.TimeoutException("blip"), (200, _env())]),
        ) as m:
            result = await _invoke(transport_retries=1)
        assert result["execution_status"] == ExecutionStatus.SUCCESS.value
        assert result["transport_attempts"] == 2
        assert m.await_count == 2


# ---------------------------------------------------------------------------
# Orchestrator integration: transport selection, shared gates, boundaries.
# ---------------------------------------------------------------------------


def _request():
    return VisionRequest(
        request_id="VISION-NATIVE-GENERATE-CONFIRM-0003",
        image_source="/tmp/taobao-viewport.png",
        task=VisionTask("UI_READ"),
        mode=VisionMode.AUTO,
        criticality=VisionCriticality("HIGH"),
        question="Read all visible text in this page/screenshot exactly as shown.",
        required_outputs=["observed_text"],
    )


@pytest.fixture
def mock_image():
    with patch(
        "tools.vision_orchestrator.prepare_image",
        new=AsyncMock(
            return_value=(
                f"data:image/png;base64,{_RAW_B64}",
                2148,
                3604,
                "image/png",
                "5cc05b47b33e3daf34346e8db5315ada8c0358bc9303010070998e00b00cb5ff",
                {
                    "transport_image_sha256": "5cc05b47",
                    "transport_mime_type": "image/png",
                    "transport_transcoded": False,
                },
            )
        ),
    ) as m:
        yield m


@pytest.fixture
def mock_native_post(mock_image):
    with patch(
        "tools.ollama_generate_vision_client._http_post_json",
        new=AsyncMock(return_value=(200, _env())),
    ) as m:
        yield m


async def _analyze_native(mock_native_post, **over):
    opts = {"base_url": "http://ollama.test:11434"}
    opts.update(over)
    return await analyze_image(_request(), enabled=True, transport=TRANSPORT_OLLAMA_NATIVE_GENERATE, **opts)


class TestOrchestratorNativeIntegration:
    async def test_native_flow_parses_thinking_json(
        self, mock_native_post, mock_image
    ):
        result = await _analyze_native(mock_native_post)
        assert result["execution_status"] == ExecutionStatus.SUCCESS.value
        structured = result["structured"]
        assert structured.get("observed_text") == ["26年动漫MAD镜头剪辑素材", "知语社", "￥18.6"]
        assert "required_text_present" in result["quality"]["passed_gates"]
        assert "text_field_conflict" not in result["quality"]["failed_gates"]
        assert "structured_parse_failed" not in result["quality"]["failed_gates"]
        assert result["quality_decision"] == QualityDecision.PASS.value

    async def test_native_flow_parses_fenced_thinking(
        self, mock_native_post, mock_image
    ):
        mock_native_post.return_value = (
            200,
            _env(thinking="```json\n" + _THINKING_JSON + "\n```"),
        )
        result = await _analyze_native(mock_native_post)
        assert result["execution_status"] == ExecutionStatus.SUCCESS.value
        assert result["structured"].get("observed_text") == [
            "26年动漫MAD镜头剪辑素材",
            "知语社",
            "￥18.6",
        ]

    async def test_malformed_thinking_stays_parse_failed(
        self, mock_native_post, mock_image
    ):
        mock_native_post.return_value = (200, _env(thinking="not json at all"))
        result = await _analyze_native(mock_native_post)
        assert "structured_parse_failed" in result["quality"]["failed_gates"]
        assert result["quality_decision"] == QualityDecision.NOT_EVALUATED.value

    async def test_schema_required_fields_enforced(
        self, mock_native_post, mock_image
    ):
        # Thinking JSON omits required canonical observed_text → gate fails,
        # no crash.
        bad = json.dumps({"evidence": "x"}, ensure_ascii=False)
        mock_native_post.return_value = (200, _env(thinking=bad))
        result = await _analyze_native(mock_native_post)
        assert "required_text_present" in result["quality"]["failed_gates"]

    async def test_strict_schema_requires_canonical_observed_text(self):
        # §7.10 — the corrected strict schema must require the canonical
        # field; visible_text stays a non-required legacy alias.
        assert NATIVE_GENERATE_STRICT_SCHEMA["required"] == ["observed_text"]
        assert "observed_text" in NATIVE_GENERATE_STRICT_SCHEMA["properties"]
        assert "visible_text" in NATIVE_GENERATE_STRICT_SCHEMA["properties"]

    async def test_legacy_visible_text_only_output_passes(
        self, mock_native_post, mock_image
    ):
        # §7.6-7 — legacy alias (old schema emitted visible_text only)
        # resolves through the one explicit boundary; gate passes.
        legacy = json.dumps(
            {"visible_text": ["26年动漫MAD镜头剪辑素材", "知语社", "￥18.6"]},
            ensure_ascii=False,
        )
        mock_native_post.return_value = (200, _env(thinking=legacy))
        result = await _analyze_native(mock_native_post)
        assert "required_text_present" in result["quality"]["passed_gates"]
        assert "text_field_conflict" not in result["quality"]["failed_gates"]

    async def test_conflicting_dual_fields_rejected(
        self, mock_native_post, mock_image
    ):
        # §7.9 — materially conflicting values under both keys are never
        # silently resolved; contradiction is marked and the result goes to
        # human review (never escalation).
        conflicted = json.dumps(
            {
                "observed_text": ["亲，请登录"],
                "visible_text": ["已买到的宝贝"],
            },
            ensure_ascii=False,
        )
        mock_native_post.return_value = (200, _env(thinking=conflicted))
        result = await _analyze_native(mock_native_post)
        assert "required_text_present" in result["quality"]["failed_gates"]
        assert "text_field_conflict" in result["quality"]["failed_gates"]
        assert result["quality_decision"] == QualityDecision.HUMAN_REVIEW_REQUIRED.value

    async def test_no_openai_fallback_on_native_failure(
        self, mock_native_post, mock_image
    ):
        mock_native_post.return_value = (500, {"error": "boom"})
        with patch(
            "tools.vision_orchestrator.invoke_vision_model",
            new=AsyncMock(),
        ) as m:
            result = await _analyze_native(mock_native_post)
        m.assert_not_awaited()  # no automatic OpenAI-compatible fallback
        assert result["execution_status"] == ExecutionStatus.ENDPOINT_UNAVAILABLE.value
        assert result["logical_model_calls"] == 1

    async def test_logical_model_calls_one(self, mock_native_post, mock_image):
        result = await _analyze_native(mock_native_post)
        assert result["logical_model_calls"] == 1
        assert len(result["trace"]) == 1

    async def test_no_automatic_escalation_executed(
        self, mock_native_post, mock_image
    ):
        result = await _analyze_native(mock_native_post)
        # Escalation is a recommendation only — never executed.
        assert result["click_authorized"] is False

    async def test_transport_metadata_in_result_and_trace(
        self, mock_native_post, mock_image
    ):
        result = await _analyze_native(mock_native_post)
        t = result["transport"]
        assert t["requested_transport"] == TRANSPORT_OLLAMA_NATIVE_GENERATE
        assert t["selected_transport"] == TRANSPORT_OLLAMA_NATIVE_GENERATE
        assert t["endpoint_family"] == "ollama_native_generate"
        assert t["native_generate_used"] is True
        trace_entry = result["trace"][0]
        assert trace_entry["native_generate_used"] is True
        assert trace_entry["content_source"] == "thinking_fallback"
        assert trace_entry["thinking_fallback_used"] is True
        assert trace_entry["total_ms"] == 14546
        assert trace_entry["done_reason"] == "stop"
        assert trace_entry["transport_attempts"] == 1

    async def test_image_metadata_preserved(self, mock_native_post, mock_image):
        result = await _analyze_native(mock_native_post)
        assert result["normalized_image_sha256"] == (
            "5cc05b47b33e3daf34346e8db5315ada8c0358bc9303010070998e00b00cb5ff"
        )
        assert result["transport_image_sha256"] == "5cc05b47"
        assert result["transport_mime_type"] == "image/png"
        assert result["transport_transcoded"] is False
        trace_entry = result["trace"][0]
        assert trace_entry["normalized_image_sha256"] == result["normalized_image_sha256"]

    async def test_native_requires_base_url(self, mock_native_post, mock_image):
        with pytest.raises(ValueError, match="requires base_url"):
            await _analyze_native(mock_native_post, base_url=None)

    async def test_native_options_override_profile(self, mock_native_post, mock_image):
        mock_native_post.side_effect = None
        mock_native_post.return_value = (200, _env())
        await _analyze_native(
            mock_native_post,
            native_generate_options={"num_ctx": 262144, "num_predict": 2000},
        )
        payload = _last_payload(mock_native_post)
        assert payload["options"]["num_ctx"] == 262144
        assert payload["options"]["num_predict"] == 2000

    async def test_native_default_profile_applied(self, mock_native_post, mock_image):
        await _analyze_native(mock_native_post)
        payload = _last_payload(mock_native_post)
        assert payload["options"]["num_ctx"] == NATIVE_GENERATE_DEFAULT_PROFILE["num_ctx"]
        assert payload["format"] is NATIVE_GENERATE_STRICT_SCHEMA


class TestOrchestratorDefaultTransport:
    """Fast-slot default transport remains OpenAI-compatible; native is
    invoked only for PRECISION (bound profile) or explicit override."""

    @pytest.fixture(autouse=True)
    def _default_invoke(self):
        with patch(
            "tools.vision_orchestrator.invoke_vision_model",
            new=AsyncMock(
                return_value={
                    "execution_status": ExecutionStatus.SUCCESS.value,
                    "raw_text": _THINKING_JSON,
                    "error": None,
                }
            ),
        ) as m:
            yield m

    def _fast_request(self):
        from tools.vision_policy import VisionCriticality, VisionTask

        return VisionRequest(
            request_id="FAST-DEFAULT-CHECK",
            image_source="/tmp/fast.png",
            task=VisionTask("SCENE_DESCRIBE"),
            mode=VisionMode.AUTO,
            criticality=VisionCriticality("NORMAL"),
            question="Describe the overall scene in this screenshot.",
            required_outputs=["observation"],
        )

    async def test_fast_default_transport_is_openai_compatible(
        self, mock_image, _default_invoke
    ):
        result = await analyze_image(self._fast_request(), enabled=True)
        _default_invoke.assert_awaited_once()
        assert result["transport"]["selected_transport"] == TRANSPORT_OPENAI_COMPATIBLE
        assert result["transport"]["endpoint_family"] == "openai_compatible"
        assert result["transport"]["native_generate_used"] is False
        assert result["initial_model_slot"] == "FAST_VLM"
        assert result["execution_status"] == ExecutionStatus.SUCCESS.value
        assert result["structured"]["observed_text"] == [
            "26年动漫MAD镜头剪辑素材",
            "知语社",
            "￥18.6",
        ]

    async def test_native_not_invoked_for_fast_default(
        self, mock_image, _default_invoke
    ):
        with patch(
            "tools.vision_orchestrator.invoke_native_generate",
            new=AsyncMock(),
        ) as native_mock:
            await analyze_image(self._fast_request(), enabled=True)
        native_mock.assert_not_awaited()

    async def test_disabled_router_reports_transport(self):
        result = await analyze_image(_request(), enabled=False)
        assert result["execution_status"] == ExecutionStatus.POLICY_BLOCKED.value
        assert result["transport"]["native_generate_used"] is False
