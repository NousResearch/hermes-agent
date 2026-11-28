"""Regression tests for smart-approval verdicts returned as structured list content blocks.

Some OpenAI-compatible endpoints (e.g. Mistral API serving reasoning models such as
``zai-glm-5-3-flash``) return ``response.choices[0].message.content`` as a list of
content blocks rather than a plain string. Calling ``.strip()`` directly crashed with
``AttributeError: 'list' object has no attribute 'strip'`` and forced escalation.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tools.approval_smart import _extract_verdict_text, _smart_approve


def _reply(content, finish_reason="stop"):
    return MagicMock(choices=[MagicMock(message=MagicMock(content=content), finish_reason=finish_reason)])


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("APPROVE", "approve"),
        ("  deny \n", "deny"),
        ([{"type": "text", "text": "APPROVE"}], "approve"),
        (
            [
                {"type": "thinking", "thinking": [{"type": "text", "text": "DENY would be wrong, looks safe"}]},
                {"type": "text", "text": "APPROVE"},
            ],
            "approve",
        ),
        ([{"type": "thinking", "thinking": "looks harmless"}], "escalate"),
        ([SimpleNamespace(type="thinking", text="APPROVE"), SimpleNamespace(type="text", text="DENY")], "deny"),
        (["APPROVE"], "approve"),
        ([], "escalate"),
        (None, "escalate"),
    ],
)
def test_smart_approve_handles_list_content_blocks(content, expected):
    with patch("agent.auxiliary_client.call_llm", return_value=_reply(content)):
        assert _smart_approve("python -c \"print('hi')\"", "script execution via -c flag") == expected


def test_extract_verdict_text_ignores_non_text_blocks():
    blocks = [
        {"type": "reasoning", "text": "APPROVE"},
        {"type": "thinking", "thinking": "APPROVE"},
        {"type": "text", "text": "  DENY  "},
    ]
    assert _extract_verdict_text(blocks).strip() == "DENY"
