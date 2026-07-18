"""Proposed compact pytest coverage for tools/validation_helper.py contract behavior."""
import json

import pytest
from pydantic import BaseModel
from pydantic_core import ValidationError

from tools.validation_helper import (
    Failure,
    Success,
    relay_failure,
    relay_success,
    validate_model_or_failure,
)


class _Payload(BaseModel):
    name: str
    count: int = 0


class TestRelayEnvelopes:
    def test_relay_success_shape(self):
        envelope = relay_success({"ok": True})
        assert envelope == {"kind": "success", "data": {"ok": True}, "meta": None}

    def test_relay_failure_shape(self):
        envelope = relay_failure("boom", code="kaboom", meta={"origin": "x"})
        assert envelope["kind"] == "failure"
        assert envelope["error"] == "boom"
        assert envelope["code"] == "kaboom"
        assert envelope["meta"] == {"origin": "x"}

    def test_success_model_is_frozen(self):
        success = Success(data={"a": 1})
        with pytest.raises(ValidationError):
            success.kind = "failure"

    def test_failure_model_is_frozen(self):
        failure = Failure(error="e", code="c")
        with pytest.raises(ValidationError):
            failure.error = "nope"


class TestValidateModelOrFailure:
    def test_happy_path(self):
        envelope = validate_model_or_failure(_Payload, {"name": "runner", "count": 2})
        assert envelope["kind"] == "success"
        assert envelope["data"] == {"name": "runner", "count": 2}

    def test_invalid_payload(self):
        envelope = validate_model_or_failure(_Payload, {"name": 123})
        assert envelope["kind"] == "failure"
        assert envelope["code"] == "validation_error"
        assert "invalid payload" in envelope["error"]
        assert "name" in envelope["error"]


class FakeTodoStore:
    def __init__(self, items):
        self._items = list(items)

    def read(self):
        return list(self._items)

    def write(self, todos, merge):
        self._items = todos if not merge else self._items + todos
        return list(self._items)


def _build_validated_todo_result(todos=None, merge=False, store=None):
    if store is None:
        return relay_failure("TodoStore not initialized", code="todo_store_unavailable")
    items = store.write(todos or [], merge) if todos is not None else store.read()
    summary = {
        "total": len(items),
        "pending": sum(1 for i in items if i["status"] == "pending"),
        "in_progress": sum(1 for i in items if i["status"] == "in_progress"),
        "completed": sum(1 for i in items if i["status"] == "completed"),
        "cancelled": sum(1 for i in items if i["status"] == "cancelled"),
    }
    try:
        return Success(data={"todos": items, "summary": summary, "validated": True}).model_dump(mode="json")
    except Exception as exc:
        return relay_failure(f"invalid todo result: {exc}", code="validation_error").model_dump(mode="json")


def _validated_todo_tool(todos=None, merge=False, store=None):
    payload = _build_validated_todo_result(todos, merge, store)
    if payload.get("kind") == "failure":
        return payload
    return payload


def _build_validated_todo_tool(store=None):
    return {
        "name": "todo_validated",
        "description": "Validated session todo list.",
        "parameters": {
            "type": "object",
            "properties": {
                "merge": {"type": "boolean", "default": False, "description": "merge behavior"},
                "todos": {
                    "type": "array",
                    "description": "Task items to write. Omit to read.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "content": {"type": "string"},
                            "status": {"type": "string", "enum": ["pending", "in_progress", "completed", "cancelled"]},
                        },
                        "required": ["id", "content", "status"],
                    },
                },
            },
            "required": [],
        },
        "handler": lambda args, **kw: _validated_todo_tool(
            todos=args.get("todos"), merge=args.get("merge", False), store=kw.get("store", store)
        ),
        "check_fn": lambda *a, **k: True,
    }


class TestValidatedTodoTool:
    def test_success_envelope_shape(self):
        store = FakeTodoStore([])
        result = _validated_todo_tool(
            todos=[{"id": "1", "content": "a", "status": "pending"}],
            merge=False,
            store=store,
        )
        assert result["kind"] == "success"
        assert result["data"]["validated"] is True
        assert len(result["data"]["todos"]) == 1
        assert result["data"]["summary"]["pending"] == 1

    def test_read_without_writing(self):
        store = FakeTodoStore([
            {"id": "1", "content": "a", "status": "pending"},
            {"id": "2", "content": "b", "status": "completed"},
        ])
        result = _validated_todo_tool(store=store)
        assert result["kind"] == "success"
        assert result["data"]["summary"]["total"] == 2
        assert result["data"]["summary"]["completed"] == 1

    def test_missing_store_returns_failure(self):
        result = _validated_todo_tool(store=None)
        assert result["kind"] == "failure"
        assert result["code"] == "todo_store_unavailable"
        assert "TodoStore not initialized" in result["error"]

    def test_json_schema_coercion_sanity(self):
        tool = _build_validated_todo_tool()
        schema = tool["parameters"]
        assert schema["properties"]["todos"]["items"]["properties"]["id"]["type"] == "string"
        assert schema["properties"]["merge"]["type"] == "boolean"
        assert schema["properties"]["todos"]["items"]["properties"]["status"]["enum"] == ["pending", "in_progress", "completed", "cancelled"]
