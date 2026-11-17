"""Viewer links from hosted artifact tool results: what non-desktop surfaces show the user."""

from __future__ import annotations

import json

import pytest
from mcp.types import CallToolResult, TextContent

from tools.hosted_artifacts import ViewerLink, viewer_link
from tools.mcp_tool_handlers import _render_call_tool_result

VIEWER = "https://portal.example.com/a/01hx"
TICKET = "https://01hx.frame.example.com/__av?t=secret-ticket"
CREATE = "mcp__hermes_artifacts__artifact_create"
LINK_TOOLS = [f"mcp__hermes_artifacts__artifact_{tool}" for tool in ("create", "update", "show", "get")]


def _handler_text(body: dict, *, error: bool = False, structured: bool = True) -> str:
    """The exact text the MCP handler hands the agent for a server result."""
    result = CallToolResult(content=[TextContent(type="text", text=json.dumps(body))],
                            structuredContent=body if structured else None, isError=error)
    return _render_call_tool_result(result, "hermes-artifacts")


@pytest.mark.parametrize("tool", LINK_TOOLS)
def test_successful_result_yields_viewer_url(tool):
    body = {"id": "01hx", "version": 2, "title": "Q3 plan", "viewer_url": VIEWER, "content_url": TICKET}
    assert viewer_link(tool, _handler_text(body)) == ViewerLink(VIEWER, "Q3 plan")


def test_structured_only_and_text_only_results_both_link():
    body = {"id": "01hx", "version": 1, "viewer_url": VIEWER, "content_url": TICKET}
    assert viewer_link(CREATE, _handler_text(body, structured=False)) == ViewerLink(VIEWER)
    structured_only = _render_call_tool_result(
        CallToolResult(content=[], structuredContent=body), "hermes-artifacts")
    assert viewer_link(CREATE, structured_only) == ViewerLink(VIEWER)


@pytest.mark.parametrize("body", [
    {"error": "stale_base", "head_version": 3, "message": "base_version 2 is not the head (3)"},
    {"error": "not_found", "message": "artifact not found"},
])
def test_tool_errors_yield_nothing(body):
    assert viewer_link(CREATE, _handler_text(body, error=True)) is None
    # A server that reports the error without flagging the result still links nothing.
    assert viewer_link(CREATE, _handler_text(body, structured=False)) is None
    assert viewer_link(CREATE, json.dumps({**body, "viewer_url": VIEWER})) is None


@pytest.mark.parametrize("result", [
    "", "not json", "[]", "null", json.dumps({"result": "plain words"}), json.dumps({"result": 7}),
    None, 42, {"error": "x", "viewer_url": VIEWER}, json.dumps({"result": json.dumps({"id": "01hx"})}),
])
def test_garbage_yields_nothing(result):
    assert viewer_link(CREATE, result) is None


@pytest.mark.parametrize("tool", [
    "mcp__hermes_artifacts__artifact_list", "mcp__hermes_artifacts__artifact_share", "artifact_create",
    "mcp__other__artifact_create", "terminal", None,
])
def test_only_the_hosted_link_tools_qualify(tool):
    body = {"id": "01hx", "version": 1, "viewer_url": VIEWER, "content_url": TICKET}
    assert viewer_link(tool, _handler_text(body)) is None


@pytest.mark.parametrize("url", [
    "javascript:alert(1)", "http://portal.example.com/a/01hx", "ftp://portal.example.com/a/01hx",
    "https://portal.example.com/a/01hx\x1b]8;;https://evil.example\x07", "https://portal.example.com/a/ 01hx",
    "/a/01hx", "",
])
def test_unsafe_or_relative_viewer_urls_are_refused(url):
    body = {"id": "01hx", "version": 1, "viewer_url": url, "content_url": TICKET}
    assert viewer_link(CREATE, _handler_text(body)) is None


def test_loopback_http_viewer_is_allowed_for_local_stacks():
    body = {"id": "01hx", "version": 1, "viewer_url": "http://portal.localhost:3000/a/01hx"}
    assert viewer_link(CREATE, _handler_text(body)) == ViewerLink("http://portal.localhost:3000/a/01hx")


def test_content_url_never_leaks():
    """No path through the helper hands out the view ticket, even when the server puts it in viewer_url."""
    swapped = {"id": "01hx", "version": 1, "viewer_url": TICKET, "content_url": TICKET}
    assert viewer_link(CREATE, _handler_text(swapped)) is None
    only_ticket = {"id": "01hx", "version": 1, "content_url": TICKET}
    assert viewer_link(CREATE, _handler_text(only_ticket)) is None
    link = viewer_link(CREATE, _handler_text({"id": "01hx", "version": 1, "title": "Plan",
                                              "viewer_url": VIEWER, "content_url": TICKET}))
    assert link is not None
    assert "secret-ticket" not in json.dumps(link.as_payload()) + link.text()


def test_title_is_flattened_and_bounded():
    body = {"id": "01hx", "version": 1, "viewer_url": VIEWER, "title": "Line\none\x1b[31m red " + "x" * 300}
    link = viewer_link(CREATE, _handler_text(body))
    assert link is not None and link.title is not None
    assert "\n" not in link.title and "\x1b" not in link.title and len(link.title) <= 120
    assert link.title.startswith("Line one")


def test_text_and_payload_shapes():
    assert ViewerLink(VIEWER, "Plan").text() == f"Artifact \u201cPlan\u201d: {VIEWER}"
    assert ViewerLink(VIEWER).text() == f"Artifact: {VIEWER}"
    assert ViewerLink(VIEWER).as_payload() == {"url": VIEWER}
    assert ViewerLink(VIEWER, "Plan").as_payload() == {"url": VIEWER, "title": "Plan"}
