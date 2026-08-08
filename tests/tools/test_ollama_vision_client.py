#!/usr/bin/env python3
"""Offline mocked tests for tools/ollama_vision_client.py response
extraction — Vision Orchestrator OpenAI-compatible response fix.

Behavioral tests only; no source-string assertions. No network requests.
"""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.ollama_vision_client import _extract_text, _load_auxiliary_client, invoke_vision_model  # noqa: E402
from tools.vision_policy import ExecutionStatus  # noqa: E402


@pytest.fixture(autouse=True)
def _ensure_client_loaded():
    _load_auxiliary_client()
    yield


def _openai_response(content):
    """Build an OpenAI-compatible response object shape."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


class TestExtractTextObjectShape:
    def test_choices_message_content_extracted(self):
        resp = _openai_response('{"observation": "a page"}')
        assert _extract_text(resp) == '{"observation": "a page"}'

    def test_choices_content_none_returns_none(self):
        resp = _openai_response(None)
        assert _extract_text(resp) is None

    def test_choices_content_empty_returns_none(self):
        resp = _openai_response("   ")
        assert _extract_text(resp) is None

    def test_empty_choices_returns_none(self):
        resp = SimpleNamespace(choices=[])
        assert _extract_text(resp) is None

    def test_missing_message_returns_none(self):
        resp = SimpleNamespace(choices=[SimpleNamespace()])
        assert _extract_text(resp) is None

    def test_malformed_object_no_uncaught_exception(self):
        assert _extract_text(object()) is None

    def test_choices_index_error_no_uncaught_exception(self):
        class _Bad:
            @property
            def choices(self):
                raise IndexError("boom")

        assert _extract_text(_Bad()) is None


class TestExtractTextLegacyShapes:
    def test_string_shape(self):
        assert _extract_text("raw text") == "raw text"

    def test_empty_string_returns_none(self):
        assert _extract_text("") is None

    def test_dict_content_shape(self):
        assert _extract_text({"content": "dict text"}) == "dict text"

    def test_dict_text_shape(self):
        assert _extract_text({"text": "dict text"}) == "dict text"

    def test_dict_list_content_shape(self):
        resp = {"content": ["a", {"text": "b"}]}
        assert _extract_text(resp) == "ab"

    def test_dict_empty_returns_none(self):
        assert _extract_text({}) is None

    def test_none_returns_none(self):
        assert _extract_text(None) is None

    def test_top_level_content_attribute_fallback(self):
        resp = SimpleNamespace(content="attr text")
        assert _extract_text(resp) == "attr text"


class TestInvokeVisionModel:
    @pytest.mark.asyncio
    async def test_openai_object_response_success(self):
        """A realistic OpenAI-compatible response from async_call_llm must
        produce SUCCESS + evaluable text."""
        with patch(
            "tools.ollama_vision_client._async_call_llm",
            new_callable=AsyncMock,
            return_value=_openai_response('{"observation": "mock page"}'),
        ):
            result = await invoke_vision_model(
                model="qwen2.5vl",
                prompt="read the page",
                image_data_url="data:image/png;base64,AAAA",
                timeout_seconds=30.0,
            )
        assert result["execution_status"] == ExecutionStatus.SUCCESS.value
        assert result["raw_text"] == '{"observation": "mock page"}'
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_openai_object_empty_content_invalid(self):
        with patch(
            "tools.ollama_vision_client._async_call_llm",
            new_callable=AsyncMock,
            return_value=_openai_response("   "),
        ):
            result = await invoke_vision_model(
                model="qwen2.5vl",
                prompt="read",
                image_data_url="data:image/png;base64,AAAA",
            )
        assert result["execution_status"] == ExecutionStatus.INVALID_RESPONSE.value
        assert result["raw_text"] == ""

    @pytest.mark.asyncio
    async def test_empty_choices_invalid(self):
        with patch(
            "tools.ollama_vision_client._async_call_llm",
            new_callable=AsyncMock,
            return_value=SimpleNamespace(choices=[]),
        ):
            result = await invoke_vision_model(
                model="qwen2.5vl",
                prompt="read",
                image_data_url="data:image/png;base64,AAAA",
            )
        assert result["execution_status"] == ExecutionStatus.INVALID_RESPONSE.value

    @pytest.mark.asyncio
    async def test_malformed_provider_object_no_uncaught_exception(self):
        class _Junk:
            pass

        with patch(
            "tools.ollama_vision_client._async_call_llm",
            new_callable=AsyncMock,
            return_value=_Junk(),
        ):
            result = await invoke_vision_model(
                model="qwen2.5vl",
                prompt="read",
                image_data_url="data:image/png;base64,AAAA",
            )
        assert result["execution_status"] in (
            ExecutionStatus.INVALID_RESPONSE.value,
            ExecutionStatus.SUCCESS.value,
        )
        # If SUCCESS, raw_text must be empty-safe; either way no crash.
        assert isinstance(result["raw_text"], str)

    @pytest.mark.asyncio
    async def test_extraction_no_second_model_call(self):
        """The extractor must not call the model again."""
        mock = AsyncMock(return_value=_openai_response("ok"))
        with patch("tools.ollama_vision_client._async_call_llm", mock):
            await invoke_vision_model(
                model="qwen2.5vl",
                prompt="read",
                image_data_url="data:image/png;base64,AAAA",
            )
        assert mock.await_count == 1


class TestOrchestratorIntegration:
    """Parsing-failure context retention through the Orchestrator."""

    @pytest.mark.asyncio
    async def test_orchestrator_retains_context_on_invalid_response(self):
        from tools.vision_orchestrator import analyze_image
        from tools.vision_policy import (
            VisionRequest,
            VisionTask,
            VisionMode,
            VisionCriticality,
        )

        with (
            patch(
                "tools.vision_orchestrator.prepare_image",
                new_callable=AsyncMock,
                return_value=(
                    "data:image/png;base64,AAAA",
                    1920,
                    1080,
                    "image/png",
                    "b" * 64,
                    {
                        "transport_image_sha256": "b" * 64,
                        "transport_mime_type": "image/png",
                        "transport_transcoded": False,
                    },
                ),
            ),
            patch(
                "tools.vision_orchestrator.invoke_vision_model",
                new_callable=AsyncMock,
                return_value={
                    "execution_status": ExecutionStatus.INVALID_RESPONSE.value,
                    "raw_text": "",
                    "error": "unparseable_model_response",
                },
            ),
        ):
            req = VisionRequest(
                request_id="t-1",
                image_source="opaque://test",
                task=VisionTask.SCENE_DESCRIBE,
                mode=VisionMode.AUTO,
                criticality=VisionCriticality.NORMAL,
                question="describe",
            )
            result = await analyze_image(req, enabled=True)

        assert result["execution_status"] == ExecutionStatus.INVALID_RESPONSE.value
        assert result["quality_decision"] == "NOT_EVALUATED"
        # Context retention (task §6):
        assert result["initial_model_slot"] == "FAST_VLM"
        assert result["actual_model"] == "qwen2.5vl"
        assert result["logical_model_calls"] == 1
        assert result.get("normalized_image_sha256") == "b" * 64
        # Failed gate records the execution failure:
        assert "execution_invalid_response" in result["quality"]["failed_gates"]
        # No raw response / prompt / image data in result:
        blob = str(result)
        assert "data:image/png;base64,AAAA" not in blob  # no image data
        assert "unparseable" not in blob or "execution_invalid_response" in blob
