"""MCP form elicitation collects the fields the server asked for on the turn's own clarify surface.

The handler runs on the MCP loop inside the tool call's contextvars snapshot; the executor binds the
agent's clarify callback for the call. Accept must carry schema-valid typed content, a headless
session must decline without asking anyone.
"""

import asyncio
import contextvars
from types import SimpleNamespace

import pytest

pytest.importorskip("mcp.types")

from tools.mcp_elicitation_form import bound_form_asker
from tools.mcp_tool_sampling import ElicitationHandler

SCHEMA = {
    "type": "object",
    "properties": {
        "project_name": {"type": "string", "title": "Project name", "minLength": 2},
        "language": {"type": "string", "enum": ["python", "rust", "go"]},
        "private": {"type": "boolean"},
        "seats": {"type": "integer", "minimum": 1},
    },
    "required": ["project_name", "language"],
}


def _call_context(asker):
    """The snapshot the MCP handler captures for an in-flight tool call dispatched by the executor."""
    holder = {}

    def capture():
        with bound_form_asker(asker):
            holder["ctx"] = contextvars.copy_context()

    capture()
    return holder["ctx"]


def _params(schema):
    return SimpleNamespace(mode="form", message="Set up the new project", requested_schema=schema)


def test_form_answers_reach_the_server_typed_and_validated():
    asked = []

    def asker(questions):
        asked.append(questions)
        if len(asked) == 1:  # name too short, seats not a number: both re-asked, the rest kept
            return {"outcome": "submitted", "answers": {"q0": "A", "q1": "rust", "q2": "Yes", "q3": "many"}}
        return {"outcome": "submitted", "answers": {"q0": "hermes-demo", "q1": "3"}}

    ctx = _call_context(asker)
    handler = ElicitationHandler("formprobe", {"timeout": 5}, call_context=lambda: ctx)
    result = asyncio.run(handler(context=None, params=_params(SCHEMA)))

    assert result.action == "accept"
    assert result.content == {"project_name": "hermes-demo", "language": "rust", "private": True, "seats": 3}
    assert [q["choices"] for q in asked[0]] == [None, ["python", "rust", "go"], ["Yes", "No"], None]
    assert "Set up the new project" in asked[0][0]["question"]
    assert len(asked[1]) == 2 and "at least 2 characters" in asked[1][0]["question"]


@pytest.mark.parametrize("reply, action", [
    ({"outcome": "cancelled"}, "cancel"),
    ({"outcome": "submitted", "answers": {}}, "decline"),  # every field skipped
])
def test_dismissed_or_skipped_form_never_accepts(reply, action):
    ctx = _call_context(lambda _questions: reply)
    handler = ElicitationHandler("formprobe", {"timeout": 5}, call_context=lambda: ctx)
    result = asyncio.run(handler(context=None, params=_params(SCHEMA)))
    assert result.action == action
    assert result.content is None


def test_headless_session_declines_without_asking(monkeypatch):
    asked = []
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    ctx = _call_context(lambda questions: asked.append(questions) or {"outcome": "submitted", "answers": {}})
    handler = ElicitationHandler("formprobe", {"timeout": 5}, call_context=lambda: ctx)
    result = asyncio.run(handler(context=None, params=_params(SCHEMA)))
    assert result.action == "decline"
    assert asked == []
