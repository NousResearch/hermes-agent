"""Behavior coverage for the cua-driver 0.20 public browser contract."""

import json
from typing import Any, Dict
from unittest.mock import Mock

from tools.computer_use.browser_route import CuaTypedBrowserRoute
from tools.computer_use.schema import COMPUTER_USE_SCHEMA
from tools.computer_use.tool import _dispatch


class _Driver:
    def __init__(self, responses: list[Dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, Dict[str, Any]]] = []

    def has_tool(self, _name: str) -> bool:
        return True

    def call(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        self.calls.append((name, dict(args)))
        return self.responses.pop(0)


def _route(driver: _Driver) -> CuaTypedBrowserRoute:
    return CuaTypedBrowserRoute(
        session_id="hermes-browser-contract",
        call_tool=driver.call,
        has_tool=driver.has_tool,
    )


def test_public_schema_exposes_020_state_and_type_options():
    properties = COMPUTER_USE_SCHEMA["parameters"]["properties"]

    assert properties["include_screenshot"]["type"] == "boolean"
    assert properties["replace"]["type"] == "boolean"
    assert properties["browser_type_mode"]["enum"] == ["insert_text", "keystrokes"]
    assert "approval_token" not in properties


def test_browser_state_forwards_screenshot_request_and_preserves_mcp_image():
    driver = _Driver([
        {
            "structuredContent": {
                "status": "ok",
                "target_id": "target-a",
                "binding_quality": "exact",
                "mutation_allowed": True,
                "tabs": [{"tab_id": "tab-a"}],
            },
            "images": ["/9j/browser-shot"],
            "image_mime_types": ["image/jpeg"],
        }
    ])

    result = _route(driver).observe(
        pid=101,
        window_id=202,
        include_screenshot=True,
    )

    assert driver.calls == [
        (
            "get_browser_state",
            {
                "pid": 101,
                "window_id": 202,
                "include_screenshot": True,
                "session": "hermes-browser-contract",
            },
        )
    ]
    assert result["_mcp_images"] == [
        {"data": "/9j/browser-shot", "mime_type": "image/jpeg"}
    ]


def test_browser_state_dispatch_returns_mcp_image_as_multimodal_content():
    backend = Mock()
    backend.typed_browser_state.return_value = {
        "status": "ok",
        "url": "https://example.test/",
        "_mcp_images": [
            {"data": "iVBORbrowser-shot", "mime_type": "image/png"}
        ],
    }

    result = _dispatch(
        backend,
        "cua_browser_state",
        {"tab_id": "tab-a", "include_screenshot": True},
    )

    backend.typed_browser_state.assert_called_once_with(
        tab_id="tab-a", include_screenshot=True
    )
    assert result["_multimodal"] is True
    assert json.loads(result["content"][0]["text"])["url"] == "https://example.test/"
    assert result["content"][1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,iVBORbrowser-shot"},
    }
    assert "iVBORbrowser-shot" not in result["text_summary"]


def test_browser_type_replace_reaches_typed_browser_backend():
    backend = Mock()
    backend.typed_browser_action.return_value = {"status": "ok"}

    result = _dispatch(
        backend,
        "cua_browser_type",
        {
            "tab_id": "tab-a",
            "ref": "field-a",
            "text": "replacement",
            "browser_type_mode": "keystrokes",
            "replace": True,
        },
    )

    assert json.loads(result)["status"] == "ok"
    backend.typed_browser_action.assert_called_once_with(
        "browser_type",
        tab_id="tab-a",
        args={
            "ref": "field-a",
            "text": "replacement",
            "replace": True,
            "mode": "keystrokes",
        },
    )
