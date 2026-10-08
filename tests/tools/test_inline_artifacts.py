"""Inline artifacts travel on the canonical tool-result durability fence."""
import json
import pytest


def test_publish_is_content_addressed_and_cannot_target_another_session():
    from tools.inline_artifact_tool import publish_html

    args = {"title": "Report", "html": "<h1>Hello</h1>", "fallback": "Hello"}
    first = json.loads(publish_html(args, session_id="current"))
    assert first["artifact"]["format"] == "html"
    assert first["artifact"]["html"] == args["html"]
    assert json.loads(publish_html(args, session_id="current")) == first
    assert json.loads(publish_html(args, session_id="other"))["artifact"]["id"] != first["artifact"]["id"]
    assert "error" in json.loads(publish_html({**args, "session_id": "other"}, session_id="current"))


@pytest.mark.parametrize("override", [
    {"title": ""}, {"title": "x" * 201}, {"fallback": ""}, {"fallback": "x" * 4001},
    {"html": ""}, {"html": "é" * 32769}, {"html": "\ud800"}, {"html": 42},
])
def test_invalid_artifacts_are_tool_errors(override):
    from tools.inline_artifact_tool import publish_html
    result = json.loads(publish_html({"title": "Report", "html": "<h1>Hello</h1>", "fallback": "Hello", **override},
                                     session_id="current"))
    assert "error" in result
    assert "artifact" not in result


def test_publish_requires_bound_context_and_is_discovered_as_a_core_tool():
    from tools.inline_artifact_tool import publish_html
    from model_tools import get_tool_definitions
    assert "error" in json.loads(publish_html({"title": "Report", "html": "Hi", "fallback": "Hi"}))
    definitions = get_tool_definitions(enabled_toolsets=["hermes-cli"])
    assert any(tool["function"]["name"] == "publish_html" for tool in definitions)


def test_metadata_recognition_does_not_promote_arbitrary_tool_text():
    from tools.inline_artifact_tool import publish_html
    from agent.inline_artifacts import artifact_from_result, project_artifact_metadata
    result = publish_html({"title": "Report", "html": "Hi", "fallback": "Hi"}, session_id="current")
    assert artifact_from_result("write_file", result) is None
    assert artifact_from_result("publish_html", "malformed") is None
    assert project_artifact_metadata({"inline_artifact": {"id": "invalid"}, "other": 1}) == {"other": 1}


def test_tool_complete_contract_declares_artifact_metadata():
    from tui_gateway.contracts.events import ToolCompletePayload
    assert "inline_artifact" in ToolCompletePayload.model_json_schema()["properties"]
