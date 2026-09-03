"""T1-24: structured-output schema on delegate_task.

Per-task ``output_schema`` (JSON Schema object): the child receives the
schema as an explicit output contract, the parent validates the child's
final answer with jsonschema, and on failure sends exactly ONE bounded
retry turn carrying the validation errors. Result entries gain
``schema_valid`` / ``schema_errors`` / ``schema_retries`` ONLY when a
schema was requested — schema-less calls keep a byte-identical result
shape (wire-shape pinning).

Pattern from: github/copilot-cli ctx.agent(prompt, {schema}) — PATTERN
ONLY, zero code/prompt text copied (proprietary).
"""

import json
import threading
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest

from tools.delegate_tool import (
    DELEGATE_TASK_SCHEMA,
    _run_single_child,
    delegate_task,
)
from tools.delegation_outcome import delegation_batch_icon
from tools.delegation_output_schema import (
    append_output_contract,
    build_retry_message,
    coerce_output_schema,
    validate_output,
)

ADDRESS_SCHEMA = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "zip": {"type": "string"},
    },
    "required": ["city"],
}


# ---------------------------------------------------------------------------
# Helper-module unit tests
# ---------------------------------------------------------------------------


class TestValidateOutput:
    def test_valid_json_matching_schema(self):
        ok, errors = validate_output('{"city": "Berlin"}', ADDRESS_SCHEMA)
        assert ok is True
        assert errors == []

    def test_json_violating_schema_reports_errors(self):
        ok, errors = validate_output('{"zip": "10115"}', ADDRESS_SCHEMA)
        assert ok is False
        assert errors
        assert any("city" in e for e in errors)

    def test_non_json_text_reports_parse_error(self):
        ok, errors = validate_output("I could not produce JSON, sorry.", ADDRESS_SCHEMA)
        assert ok is False
        assert errors

    def test_code_fenced_json_is_accepted(self):
        text = '```json\n{"city": "Oslo"}\n```'
        ok, errors = validate_output(text, ADDRESS_SCHEMA)
        assert ok is True
        assert errors == []

    def test_json_embedded_in_prose_is_extracted(self):
        text = 'Here is the result:\n{"city": "Lima"}\nHope that helps!'
        ok, _ = validate_output(text, ADDRESS_SCHEMA)
        assert ok is True

    def test_empty_text_is_invalid(self):
        ok, errors = validate_output("", ADDRESS_SCHEMA)
        assert ok is False
        assert errors


class TestCoerceOutputSchema:
    def test_valid_schema_passes(self):
        schema, err = coerce_output_schema(ADDRESS_SCHEMA)
        assert schema == ADDRESS_SCHEMA
        assert err is None

    def test_none_passes_through(self):
        schema, err = coerce_output_schema(None)
        assert schema is None
        assert err is None

    def test_non_dict_is_rejected(self):
        schema, err = coerce_output_schema("not a schema")
        assert schema is None
        assert err

    def test_invalid_json_schema_is_rejected(self):
        schema, err = coerce_output_schema({"type": 42})
        assert schema is None
        assert err


class TestPromptPlumbing:
    def test_contract_block_carries_schema(self):
        out = append_output_contract("base context", ADDRESS_SCHEMA)
        assert "base context" in out
        assert "OUTPUT CONTRACT" in out
        assert '"city"' in out

    def test_contract_block_without_prior_context(self):
        out = append_output_contract(None, ADDRESS_SCHEMA)
        assert "OUTPUT CONTRACT" in out

    def test_retry_message_carries_verbatim_errors(self):
        msg = build_retry_message(["'city' is a required property"])
        assert "'city' is a required property" in msg
        assert "JSON" in msg


# ---------------------------------------------------------------------------
# Tool-schema surface (one-time static field)
# ---------------------------------------------------------------------------


class TestToolSchemaSurface:
    def test_output_schema_on_task_items(self):
        item_props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"][
            "items"
        ]["properties"]
        assert "output_schema" in item_props
        assert item_props["output_schema"]["type"] == "object"
        # never required
        assert "output_schema" not in DELEGATE_TASK_SCHEMA["parameters"][
            "properties"
        ]["tasks"]["items"]["required"]

    def test_output_schema_advertised_per_task_only(self):
        """output_schema is advertised inside tasks[] items (the only spawn
        shape); the legacy top-level param stays handler-accepted but out
        of the schema."""
        props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]
        assert "output_schema" not in props
        task_props = props["tasks"]["items"]["properties"]
        assert task_props["output_schema"]["type"] == "object"


# ---------------------------------------------------------------------------
# _run_single_child validation + bounded retry
# ---------------------------------------------------------------------------


class _StubChild:
    """Minimal child agent double (mirrors test_delegate_kanban_isolation)."""

    tool_progress_callback: Callable[..., None] | None = None
    _delegate_saved_tool_names: list = []
    _credential_pool = None
    _subagent_id = None  # skip registry
    _delegate_depth = 1
    _parent_subagent_id = None
    _delegate_output_schema: dict | None = None
    model = "test-model"
    session_prompt_tokens = 0
    session_completion_tokens = 0
    session_estimated_cost_usd = 0.0
    session_reasoning_tokens = 0

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list = []

    def get_activity_summary(self):
        return {"api_call_count": 1, "max_iterations": 5, "current_tool": None}

    def run_conversation(self, user_message, task_id=None, **_kwargs):
        self.calls.append(user_message)
        text = self.responses.pop(0)
        return {
            "final_response": text,
            "completed": True,
            "api_calls": 1,
            "messages": [],
        }

    def close(self):
        return None


class _StubParent:
    _current_task_id = None
    _delegate_depth = 0

    def _touch_activity(self, _desc):
        return None


def _run(child):
    return _run_single_child(0, "produce the address", child, _StubParent())


class TestRunSingleChildSchemaValidation:
    def test_valid_first_try_no_retry(self):
        child = _StubChild(['{"city": "Berlin"}'])
        child._delegate_output_schema = ADDRESS_SCHEMA
        entry = _run(child)
        assert entry["status"] == "completed"
        assert entry["outcome"] == "unverified"
        assert entry["schema_valid"] is True
        assert "schema_errors" not in entry
        assert "error" not in entry
        assert len(child.calls) == 1

    def test_invalid_then_retry_then_valid(self):
        child = _StubChild(["not json at all", '{"city": "Oslo"}'])
        child._delegate_output_schema = ADDRESS_SCHEMA
        entry = _run(child)
        assert entry["schema_valid"] is True
        assert entry["schema_retries"] == 1
        # retry turn carried the validation errors
        assert len(child.calls) == 2
        assert "rejected" in child.calls[1] or "JSON" in child.calls[1]
        # final summary is the retried (valid) answer
        assert json.loads(entry["summary"])["city"] == "Oslo"

    def test_invalid_twice_surfaces_errors_and_stops(self):
        child = _StubChild(["nope", "still nope"])
        events = []
        child.tool_progress_callback = lambda event_type, **kwargs: events.append(
            (event_type, kwargs)
        )
        child._delegate_output_schema = ADDRESS_SCHEMA
        entry = _run(child)
        # The child loop completed, but the parent-side contract verdict is an
        # authoritative logical failure. Keep both facts in the envelope.
        assert entry["status"] == "completed"
        assert entry["outcome"] == "failed"
        assert entry["schema_valid"] is False
        assert entry["schema_errors"]
        assert entry["schema_retries"] == 1
        assert entry["error_authoritative"] is True
        assert "output_schema" in entry["error"]
        complete = next(kwargs for event, kwargs in events if event == "subagent.complete")
        assert complete["outcome"] == "failed"
        assert complete["status"] == "completed"
        assert complete["schema_valid"] is False
        assert complete["schema_errors"] == entry["schema_errors"]
        assert complete["schema_retries"] == 1
        assert complete["error_authoritative"] is True
        # exactly ONE retry — bounded
        assert len(child.calls) == 2

    def test_retry_exception_degrades_to_invalid(self):
        child = _StubChild(["nope"])
        child._delegate_output_schema = ADDRESS_SCHEMA

        original = child.run_conversation

        def flaky(user_message, task_id=None, **kw):
            if child.calls:
                raise RuntimeError("child died on retry")
            return original(user_message, task_id=task_id, **kw)

        child.run_conversation = flaky
        entry = _run(child)
        assert entry["status"] == "failed"
        assert entry["outcome"] == "failed"
        assert entry["schema_valid"] is False
        assert entry["schema_errors"]
        assert entry["exit_reason"] == "schema_retry_exception"
        assert entry["error"] == "Schema retry failed: child died on retry"

    def test_retry_runtime_failure_replaces_first_attempt_evidence(self):
        child = _StubChild([])
        child._delegate_output_schema = ADDRESS_SCHEMA
        responses = [
            {
                "final_response": "first invalid payload",
                "completed": True,
                "failed": False,
                "interrupted": False,
                "turn_exit_reason": "text_response",
                "tool_error_count": 0,
                "api_calls": 1,
                "messages": [],
            },
            {
                "final_response": "retry invalid payload",
                "completed": False,
                "failed": True,
                "interrupted": False,
                "error": "retry provider failed",
                "turn_exit_reason": "provider_error",
                "tool_error_count": 2,
                "api_calls": 1,
                "messages": [],
            },
        ]

        def staged_run(user_message, task_id=None, **_kw):
            child.calls.append(user_message)
            return responses.pop(0)

        child.run_conversation = staged_run
        entry = _run(child)

        assert entry["status"] == "failed"
        assert entry["outcome"] == "failed"
        assert entry["error"] == "retry provider failed"
        assert entry["exit_reason"] == "provider_error"
        assert entry["tool_error_count"] == 0
        assert entry["terminal_tool_error_count"] == 2
        assert entry["schema_valid"] is False
        assert entry["schema_retries"] == 1
        assert entry["api_calls"] == 2

    def test_retry_interruption_replaces_first_attempt_evidence(self):
        child = _StubChild([])
        child._delegate_output_schema = ADDRESS_SCHEMA
        responses = [
            {
                "final_response": "first invalid payload",
                "completed": True,
                "interrupted": False,
                "turn_exit_reason": "text_response",
                "tool_error_count": 0,
                "api_calls": 1,
                "messages": [],
            },
            {
                "final_response": "interrupted invalid payload",
                "completed": False,
                "interrupted": True,
                "partial": True,
                "turn_exit_reason": "interrupted",
                "tool_error_count": 1,
                "api_calls": 1,
                "messages": [],
            },
        ]

        def staged_run(user_message, task_id=None, **_kw):
            child.calls.append(user_message)
            return responses.pop(0)

        child.run_conversation = staged_run
        entry = _run(child)

        assert entry["status"] == "interrupted"
        assert entry["outcome"] == "failed"
        assert entry["exit_reason"] == "interrupted"
        assert entry["interrupted"] is True
        assert entry["tool_error_count"] == 0
        assert entry["terminal_tool_error_count"] == 1
        assert entry["schema_valid"] is False

    def test_runtime_failure_reason_survives_concurrent_schema_failure(self):
        child = _StubChild([])
        child._delegate_output_schema = ADDRESS_SCHEMA

        def failed_run(user_message, task_id=None, **kw):
            child.calls.append(user_message)
            return {
                "api_calls": 1,
                "completed": False,
                "error": "provider transport failed",
                "failed": True,
                "final_response": "still not json",
                "messages": [],
            }

        child.run_conversation = failed_run
        entry = _run(child)

        assert entry["outcome"] == "failed"
        assert entry["schema_valid"] is False
        assert entry["error"] == "provider transport failed"
        assert entry["error_authoritative"] is True
        assert "schema_retries" not in entry
        assert len(child.calls) == 1

    @pytest.mark.parametrize(
        "runtime_fields",
        [
            {"completed": False, "failed": True},
            {"completed": True, "error": "provider transport failed"},
            {
                "completed": True,
                "turn_exit_reason": "all_retries_exhausted_no_response",
            },
        ],
    )
    def test_runtime_failure_with_invalid_text_never_authorizes_schema_retry(
        self, runtime_fields
    ):
        child = _StubChild([])
        child._delegate_output_schema = ADDRESS_SCHEMA

        def failed_run(user_message, task_id=None, **_kw):
            child.calls.append(user_message)
            return {
                "api_calls": 1,
                "completed": True,
                "failed": False,
                "final_response": "invalid diagnostic text",
                "messages": [],
                **runtime_fields,
            }

        child.run_conversation = failed_run
        entry = _run(child)

        assert entry["outcome"] == "failed"
        assert entry["schema_valid"] is False
        assert "schema_retries" not in entry
        assert len(child.calls) == 1

    def test_tool_error_count_remains_aggregate_across_clean_schema_retry(self):
        child = _StubChild([])
        child._delegate_output_schema = ADDRESS_SCHEMA
        responses = [
            {
                "final_response": "first invalid payload",
                "completed": True,
                "failed": False,
                "api_calls": 1,
                "tool_error_count": 1,
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "attempt-1",
                                "function": {
                                    "name": "terminal",
                                    "arguments": "{}",
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "attempt-1",
                        "content": "Error: command failed",
                    },
                ],
            },
            {
                "final_response": '{"city": "Oslo"}',
                "completed": True,
                "failed": False,
                "api_calls": 1,
                "tool_error_count": 0,
                "messages": [],
            },
        ]

        def staged_run(user_message, task_id=None, **_kw):
            child.calls.append(user_message)
            return responses.pop(0)

        child.run_conversation = staged_run
        entry = _run(child)

        assert entry["schema_valid"] is True
        assert entry["schema_retries"] == 1
        assert entry["tool_error_count"] == 1
        assert entry["terminal_tool_error_count"] == 0
        assert any(item.get("status") == "error" for item in entry["tool_trace"])

    def test_no_schema_keeps_legacy_result_shape(self):
        """Schema-less calls must not gain new keys (wire-shape pinning)."""
        child = _StubChild(['{"city": "Berlin"}'])
        entry = _run(child)
        assert "schema_valid" not in entry
        assert "schema_errors" not in entry
        assert "schema_retries" not in entry
        assert len(child.calls) == 1

    def test_failed_child_skips_validation(self):
        """A child with no output never gets a schema retry turn."""
        child = _StubChild([""])
        child._delegate_output_schema = ADDRESS_SCHEMA
        entry = _run(child)
        assert entry["status"] == "failed"
        assert len(child.calls) == 1
        assert entry.get("schema_valid") is False

    def test_schema_failure_has_failed_outcome_without_false_green(self):
        """Regression: a final answer that still violates the declared
        output contract after the bounded retry (here the classic empty
        ``{}`` fallback) must have outcome="failed" even though its child
        lifecycle completed. The outcome-aware batch renderer must never
        print a false-green check for that envelope."""
        child = _StubChild(["not json at all", "{}"])
        child._delegate_output_schema = ADDRESS_SCHEMA
        entry = _run(child)
        assert entry["schema_valid"] is False
        assert entry["schema_errors"]
        assert entry["status"] == "completed"
        assert entry["outcome"] == "failed"
        assert delegation_batch_icon(entry) == "✗"
        # the failed entry names the schema violation, not the generic
        # "no response" error — the child DID respond, unusably
        assert "output_schema" in entry.get("error", "")
        # the invalid final text is still propagated for debugging
        assert entry["summary"] == "{}"

    def test_schema_failure_without_retry_reported_as_failed(self):
        """Same class, first-try path: retry turn raises, leaving the
        original non-JSON answer in place — status must still be failed."""
        child = _StubChild(["nope"])
        child._delegate_output_schema = ADDRESS_SCHEMA

        original = child.run_conversation

        def flaky(user_message, task_id=None, **kw):
            if child.calls:
                raise RuntimeError("child died on retry")
            return original(user_message, task_id=task_id, **kw)

        child.run_conversation = flaky
        entry = _run(child)
        assert entry["schema_valid"] is False
        assert entry["status"] == "failed"

    def test_schema_valid_entry_still_completed(self):
        """Guard: schema_valid=True keeps status="completed" untouched."""
        child = _StubChild(['{"city": "Berlin"}'])
        child._delegate_output_schema = ADDRESS_SCHEMA
        entry = _run(child)
        assert entry["status"] == "completed"
        assert "error" not in entry


# ---------------------------------------------------------------------------
# delegate_task dispatch-time schema handling
# ---------------------------------------------------------------------------


def _make_mock_parent():
    parent = MagicMock()
    parent._delegate_depth = 0
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    return parent


class TestDelegateTaskDispatch:
    def test_non_dict_output_schema_rejected(self):
        with (
            patch("tools.delegate_tool._load_config", return_value={}),
            patch(
                "tools.delegate_tool._resolve_delegation_credentials",
                return_value={
                    "provider": None,
                    "model": None,
                    "base_url": None,
                    "api_key": None,
                    "api_mode": None,
                },
            ),
        ):
            out = delegate_task(
                tasks=[
                    {"goal": "Summarize the release notes for module A", "output_schema": "not-a-dict"},
                    {"goal": "Summarize the release notes for module B"},
                ],
                parent_agent=_make_mock_parent(),
            )
        payload = json.loads(out)
        assert payload.get("error")
        assert "output_schema" in payload["error"]

    def test_invalid_json_schema_rejected_at_dispatch(self):
        with (
            patch("tools.delegate_tool._load_config", return_value={}),
            patch(
                "tools.delegate_tool._resolve_delegation_credentials",
                return_value={
                    "provider": None,
                    "model": None,
                    "base_url": None,
                    "api_key": None,
                    "api_mode": None,
                },
            ),
        ):
            out = delegate_task(
                tasks=[
                    {"goal": "Summarize the release notes for module A", "output_schema": {"type": 42}},
                    {"goal": "Summarize the release notes for module B"},
                ],
                parent_agent=_make_mock_parent(),
            )
        payload = json.loads(out)
        assert payload.get("error")
        assert "output_schema" in payload["error"]

    def test_child_receives_contract_and_schema_attr(self):
        """The built child carries the schema attr and its context gains
        the output-contract block."""
        captured = {}

        def fake_build(**kwargs):
            captured.update(kwargs)
            child = _StubChild(['{"city": "Rio"}'])
            return child

        with (
            patch("tools.delegate_tool._load_config", return_value={}),
            patch(
                "tools.delegate_tool._resolve_delegation_credentials",
                return_value={
                    "provider": None,
                    "model": None,
                    "base_url": None,
                    "api_key": None,
                    "api_mode": None,
                },
            ),
            patch(
                "tools.delegate_tool._build_child_preserving_parent_tools",
                side_effect=fake_build,
            ),
        ):
            out = delegate_task(
                goal="produce the address",
                context="base context",
                output_schema=ADDRESS_SCHEMA,
                parent_agent=_make_mock_parent(),
            )
        payload = json.loads(out)
        assert "OUTPUT CONTRACT" in (captured.get("context") or "")
        results = payload.get("results") or []
        assert results and results[0].get("schema_valid") is True
