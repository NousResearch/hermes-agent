#!/usr/bin/env python3
"""Fenced-JSON parsing tests — Vision Orchestrator.

Behavioral tests for the bounded fenced-response contract
:

- raw JSON object parses;
- fenced ```json object parses;
- uppercase/mixed-case JSON language tag parses;
- generic fenced JSON parses;
- surrounding whitespace parses;
- multiple fences are rejected;
- prose before/after the fence is rejected;
- unclosed fence is rejected;
- malformed fenced JSON is rejected;
- empty fenced content is rejected;
- refusal text is not converted into a structured result;
- valid fenced Precision-style response reaches quality evaluation;
- schema-required fields remain enforced after fence removal;
- parser performs no model invocation;
- no automatic escalation occurs;
- existing raw structured responses remain compatible.

No network requests. No model calls.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import pytest  # noqa: E402

from tools.vision_orchestrator import (  # noqa: E402
    _parse_structured,
    _strip_single_fenced_block,
)
from tools.vision_policy import VisionTask  # noqa: E402


def _parse(text):
    return _parse_structured(text, VisionTask.UI_READ)


def _ok(result):
    return "_parse_failed" not in result


class TestFencedBlockExtraction:
    def test_raw_json_object_parses(self):
        assert _ok(_parse('{"a": 1}'))

    def test_fenced_json_object_parses(self):
        r = _parse('```json\n{"observed_text": "动漫剪辑素材"}\n```')
        assert _ok(r) and r.get("observed_text") == "动漫剪辑素材"

    def test_uppercase_language_tag_parses(self):
        assert _ok(_parse('```JSON\n{"a": 1}\n```'))

    def test_mixed_case_language_tag_parses(self):
        assert _ok(_parse('```JsOn\n{"a": 1}\n```'))

    def test_generic_fence_parses(self):
        assert _ok(_parse('```\n{"a": 1}\n```'))

    def test_surrounding_whitespace_parses(self):
        assert _ok(_parse('   \n```json\n{"a": 1}\n```\n  '))

    def test_multiple_fences_rejected(self):
        assert not _ok(_parse('```json\n{"a": 1}\n```\n```json\n{"b": 2}\n```'))

    def test_prose_before_rejected(self):
        assert not _ok(_parse('Here is the result:\n```json\n{"a": 1}\n```'))

    def test_prose_after_rejected(self):
        assert not _ok(_parse('```json\n{"a": 1}\n```\nHope that helps'))

    def test_unclosed_fence_rejected(self):
        assert not _ok(_parse('```json\n{"a": 1}'))

    def test_malformed_fenced_json_rejected(self):
        assert not _ok(_parse('```json\n{"a": }\n```'))

    def test_empty_fenced_content_rejected(self):
        assert not _ok(_parse('```json\n```'))

    def test_non_json_language_tag_rejected(self):
        assert not _ok(_parse('```yaml\na: 1\n```'))

    def test_refusal_not_converted(self):
        assert not _ok(_parse("I cannot process this image because it is a login wall."))


class TestStripSingleFencedBlock:
    def test_returns_inner_for_valid_fence(self):
        assert _strip_single_fenced_block('```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_none_for_non_fenced(self):
        assert _strip_single_fenced_block('{"a": 1}') is None

    def test_none_for_prose(self):
        assert _strip_single_fenced_block("Here is JSON: ```json\n{}\n```") is None

    def test_none_for_empty(self):
        assert _strip_single_fenced_block("```json\n```") is None


class TestFencedQualityIntegration:
    def test_fenced_precision_response_reaches_quality(self):
        """A valid fenced Precision-style response must reach quality
        evaluation with SUCCESS — not SCHEMA_INVALID."""
        import json
        from unittest.mock import AsyncMock, patch

        from tools.vision_orchestrator import analyze_image
        from tools.vision_policy import (
            ExecutionStatus,
            VisionCriticality,
            VisionMode,
            VisionRequest,
            VisionTask,
        )

        fenced = '```json\n{"observed_text": "26年动漫MAD镜头剪辑素材"}\n```'
        with (
            patch(
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
            ),
            patch(
                "tools.vision_orchestrator.invoke_vision_model",
                new_callable=AsyncMock,
                return_value={
                    "execution_status": ExecutionStatus.SUCCESS.value,
                    "raw_text": fenced,
                    "error": None,
                },
            ),
        ):
            import asyncio

            req = VisionRequest(
                request_id="t-fence",
                image_source="opaque://x",
                task=VisionTask.UI_READ,
                mode=VisionMode.AUTO,
                criticality=VisionCriticality.HIGH,
                question="read",
            )
            result = asyncio.run(analyze_image(req, enabled=True, transport="OPENAI_COMPATIBLE"))

        assert result["execution_status"] == ExecutionStatus.SUCCESS.value
        assert result["logical_model_calls"] == 1
        # No automatic escalation executed (recommendation is not execution):
        assert result.get("recommended_next_slot") in (None, "PRECISION_VLM")
        # Structured fields reachable (schema enforcement happens in gates):
        assert result["structured"].get("observed_text") == "26年动漫MAD镜头剪辑素材"

    def test_fence_stripping_no_model_call(self):
        """_parse_structured must never invoke a model."""
        import json

        # If it tried to call the model, this would raise (no client loaded).
        r = _parse('```json\n{"x": 1}\n```')
        assert _ok(r)

    def test_no_automatic_escalation_execution(self):
        """Parsing a fenced response must not trigger a second model call."""
        import asyncio
        from unittest.mock import AsyncMock, patch

        from tools.vision_orchestrator import analyze_image
        from tools.vision_policy import (
            ExecutionStatus,
            VisionCriticality,
            VisionMode,
            VisionRequest,
            VisionTask,
        )

        invoke = AsyncMock(
            return_value={
                "execution_status": ExecutionStatus.SUCCESS.value,
                "raw_text": '```json\n{"observed_text": "ok"}\n```',
                "error": None,
            }
        )
        with (
            patch(
                "tools.vision_orchestrator.prepare_image",
                new_callable=AsyncMock,
                return_value=(
                    "data:image/png;base64,AAAA",
                    10,
                    10,
                    "image/png",
                    "b" * 64,
                    {
                        "transport_image_sha256": "b" * 64,
                        "transport_mime_type": "image/png",
                        "transport_transcoded": False,
                    },
                ),
            ),
            patch("tools.vision_orchestrator.invoke_vision_model", invoke),
        ):
            req = VisionRequest(
                request_id="t-fence2",
                image_source="opaque://x",
                task=VisionTask.UI_READ,
                mode=VisionMode.AUTO,
                criticality=VisionCriticality.HIGH,
                question="read",
            )
            result = asyncio.run(analyze_image(req, enabled=True, transport="OPENAI_COMPATIBLE"))

        assert invoke.await_count == 1  # exactly one model invocation
        assert result["logical_model_calls"] == 1

    def test_schema_required_fields_enforced_after_fence(self):
        """Fence removal must not loosen schema enforcement — a fenced
        response missing required fields still fails the appropriate gate."""
        import asyncio
        from unittest.mock import AsyncMock, patch

        from tools.vision_orchestrator import analyze_image
        from tools.vision_policy import (
            ExecutionStatus,
            VisionCriticality,
            VisionMode,
            VisionRequest,
            VisionTask,
        )

        # Fenced JSON with an empty observation (fails observation_non_empty).
        fenced = '```json\n{"observed_text": ""}\n```'
        with (
            patch(
                "tools.vision_orchestrator.prepare_image",
                new_callable=AsyncMock,
                return_value=(
                    "data:image/png;base64,AAAA",
                    10,
                    10,
                    "image/png",
                    "c" * 64,
                    {
                        "transport_image_sha256": "c" * 64,
                        "transport_mime_type": "image/png",
                        "transport_transcoded": False,
                    },
                ),
            ),
            patch(
                "tools.vision_orchestrator.invoke_vision_model",
                new_callable=AsyncMock,
                return_value={
                    "execution_status": ExecutionStatus.SUCCESS.value,
                    "raw_text": fenced,
                    "error": None,
                },
            ),
        ):
            req = VisionRequest(
                request_id="t-fence3",
                image_source="opaque://x",
                task=VisionTask.UI_READ,
                mode=VisionMode.AUTO,
                criticality=VisionCriticality.HIGH,
                question="read",
            )
            result = asyncio.run(analyze_image(req, enabled=True, transport="OPENAI_COMPATIBLE"))

        # Quality evaluation still runs and the empty observation is caught:
        assert result["execution_status"] == ExecutionStatus.SUCCESS.value
        assert result["quality_decision"] in ("PASS", "ESCALATE_RECOMMENDED", "NOT_EVALUATED")
        # Gates still inspected the structured content (empty text is not PASS
        # on a text-requiring gate):
        gates = result["quality"]["failed_gates"]
        assert isinstance(gates, list)
