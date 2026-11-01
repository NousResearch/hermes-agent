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


@pytest.mark.parametrize("spec, answer, value", [
    ({"type": "integer"}, "9007199254740993", 9007199254740993),  # past 2**53: no float round trip
    ({"type": "integer"}, "1e3", 1000),
    ({"type": "number"}, "2.5", 2.5),
    ({"type": "string", "format": "uri"}, "mailto:user@example.com", "mailto:user@example.com"),
    ({"type": "string", "format": "uri"}, "https://example.com/x", "https://example.com/x"),
    ({"type": "string", "format": "date-time"}, "2026-10-25T09:30:00Z", "2026-10-25T09:30:00Z"),
    ({"type": "string", "format": "date-time"}, "2026-10-25T09:30:00.5-07:00", "2026-10-25T09:30:00.5-07:00"),
    ({"type": "string", "format": "date"}, "2026-10-25", "2026-10-25"),
])
def test_valid_answers_keep_their_exact_value(spec, answer, value):
    from tools.mcp_elicitation_form import coerce_answer, parse_form
    field = parse_form({"properties": {"x": spec}})[0]
    assert coerce_answer(field, answer) == (True, value, "")


@pytest.mark.parametrize("spec, answer", [
    ({"type": "number"}, "inf"),
    ({"type": "number"}, "NaN"),
    ({"type": "number"}, "1e999"),
    ({"type": "integer"}, "2.5"),
    ({"type": "string", "format": "uri"}, "not a uri"),
    ({"type": "string", "format": "date-time"}, "2026-10-25T09:30:00"),  # RFC 3339 needs an offset
    ({"type": "string", "format": "date-time"}, "2026-10-25"),
    ({"type": "string", "format": "date"}, "20261025"),
])
def test_invalid_answers_are_reasked(spec, answer):
    from tools.mcp_elicitation_form import coerce_answer, parse_form
    field = parse_form({"properties": {"x": spec}})[0]
    present, _value, error = coerce_answer(field, answer)
    assert not present and error


def test_no_question_opens_after_the_elicitation_deadline(monkeypatch):
    """The handler's wait ends (cancel sent to the server) while the user is still on the first batch:
    the worker, which cancellation cannot stop, must not open the second batch afterwards."""
    import threading
    from tools.clarify_tool import MAX_QUESTIONS
    schema = {"type": "object", "properties": {f"f{i}": {"type": "string"} for i in range(MAX_QUESTIONS + 1)}}
    asked, answered_late = [], threading.Event()

    def asker(questions):
        asked.append(questions)
        assert answered_late.wait(10)  # the user answers only after the server got its cancel
        return {"outcome": "submitted", "answers": {q["qid"]: "x" for q in questions}}

    monkeypatch.setattr(ElicitationHandler, "_OUTER_TIMEOUT_GRACE_SECONDS", 0)
    ctx = _call_context(asker)
    handler = ElicitationHandler("formprobe", {"timeout": 1}, call_context=lambda: ctx)  # 1s: the floor

    async def _elicit():
        result = await handler(context=None, params=_params(schema))
        answered_late.set()
        return result
    # asyncio.run returns only after the worker thread has finished.
    result = asyncio.run(_elicit())
    assert result.action == "cancel"
    assert len(asked) == 1


@pytest.mark.parametrize("answer, outcome", [
    ("us", (True, "us", "")),   # exact match wins over the first case-insensitive one
    ("US", (True, "US", "")),
    ("Us", (False, None, "'Us' is not one of: US, us")),  # ambiguous: re-asked, never guessed
])
def test_enum_pick_is_never_swapped_for_another_option(answer, outcome):
    from tools.mcp_elicitation_form import coerce_answer, parse_form
    field = parse_form({"properties": {"region": {"type": "string", "enum": ["US", "us"]}}})[0]
    assert coerce_answer(field, answer) == outcome


def test_skipping_every_optional_field_declines():
    schema = {"type": "object", "properties": {"note": {"type": "string"}, "rating": {"type": "integer"}}}
    ctx = _call_context(lambda _questions: {"outcome": "submitted", "answers": {}})
    handler = ElicitationHandler("formprobe", {"timeout": 5}, call_context=lambda: ctx)
    result = asyncio.run(handler(context=None, params=_params(schema)))
    assert result.action == "decline"
