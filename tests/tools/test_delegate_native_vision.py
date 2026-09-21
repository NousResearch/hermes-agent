"""Delegated children keep the route identity required by native vision.

Regression for the subagent path omitted when #70071 threaded named custom
provider identity through the other agent-construction surfaces.
"""

from __future__ import annotations

import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from agent.auxiliary_client import reset_runtime_main, set_runtime_main
from hermes_constants import get_hermes_home
from tools import vision_tools  # noqa: F401 - registers vision_analyze
from tools.delegate_tool_config import _resolve_child_runtime
from tools.registry import registry


_TINY_PNG = base64.b64decode(
    b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


def _parent() -> SimpleNamespace:
    return SimpleNamespace(
        model="vision-model",
        provider="custom",
        requested_provider="custom:vision-endpoint",
        base_url="https://vision.invalid/v1",
        api_mode="chat_completions",
        capabilities={},
        fallback_model=None,
        request_overrides={},
        reasoning_config=None,
        acp_args=[],
    )


def _child_runtime(parent: SimpleNamespace, *, override_base_url=None):
    return _resolve_child_runtime(
        parent,
        {},
        "test-key",
        model=None,
        override_provider=None,
        override_base_url=override_base_url,
        override_api_key=None,
        override_api_mode=None,
        override_acp_command=None,
        override_acp_args=None,
    )


def test_inherited_named_custom_child_uses_native_vision(tmp_path):
    runtime = _child_runtime(_parent())
    assert runtime["provider"] == "custom"
    assert runtime["requested_provider"] == "custom:vision-endpoint"

    image = tmp_path / "page.png"
    image.write_bytes(_TINY_PNG)
    get_hermes_home().joinpath("config.yaml").write_text(
        """\
model:
  provider: custom:vision-endpoint
  default: vision-model
providers:
  vision-endpoint:
    name: Vision Endpoint
    api: https://vision.invalid/v1
    models:
      vision-model:
        supports_vision: true
""",
        encoding="utf-8",
    )

    token = set_runtime_main(
        runtime["provider"],
        runtime["model"],
        requested_provider=runtime["requested_provider"],
        base_url=runtime["base_url"],
        api_key=runtime["api_key"],
        api_mode=runtime["api_mode"],
    )
    try:
        with patch(
            "tools.vision_tools.vision_analyze_tool", new_callable=AsyncMock
        ) as auxiliary:
            result = registry.dispatch(
                "vision_analyze", {"image_url": str(image), "question": "describe"}
            )
    finally:
        reset_runtime_main(token)

    assert isinstance(result, dict) and result["_multimodal"] is True
    auxiliary.assert_not_called()


def test_endpoint_override_does_not_borrow_parent_named_identity():
    runtime = _child_runtime(
        _parent(), override_base_url="https://different.invalid/v1"
    )

    assert runtime["provider"] == "custom"
    assert runtime["requested_provider"] == "custom"
