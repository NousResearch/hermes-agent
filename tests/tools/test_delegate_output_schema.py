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
from unittest.mock import MagicMock, patch

from tools.delegate_tool import (
    DELEGATE_TASK_SCHEMA,
    _run_single_child,
    delegate_task,
)
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


class TestForgivingSchema:
    """v0.21.3: ``{}`` (or any non-constraining schema) wraps prose into a
    JSON object instead of failing. Closes the "活干了但汇报被吞" bug where
    a child agent reported in markdown prose (a valid response to the work)
    but the empty schema's strict JSON parse ate it and the parent saw
    ``status="failed"`` with ``Final answer does not satisfy the declared
    output_schema (after 1 retry)``.
    """

    def test_empty_schema_is_forgiving(self):
        from tools.delegation_output_schema import is_forgiving_schema
        assert is_forgiving_schema({}) is True
        assert is_forgiving_schema({"title": "anything"}) is True
        assert is_forgiving_schema({"description": "x", "$schema": "https://x"}) is True

    def test_constraining_keys_make_schema_strict(self):
        from tools.delegation_output_schema import is_forgiving_schema
        # Any of these keys makes the schema impose real constraints — keep strict.
        for k in ("type", "properties", "required", "items", "allOf",
                  "anyOf", "oneOf", "additionalProperties", "minProperties"):
            assert is_forgiving_schema({k: []}) is False, f"{k!r} should be strict"
        # Even a bare "type":"object" is constraining — caller wants an object.
        assert is_forgiving_schema({"type": "object"}) is False

    def test_prose_passes_forgiving_schema(self):
        ok, errs = validate_output(
            "I did the work. Here is the report:\n- step 1\n- step 2",
            {},
        )
        assert ok is True
        assert any("non_json_wrapped" in e for e in errs)

    def test_fenced_prose_passes_forgiving_schema(self):
        ok, errs = validate_output(
            "```\n## Summary\nDid it.\n```",
            {},
        )
        assert ok is True
        assert any("non_json_wrapped" in e for e in errs)

    def test_constraining_schema_still_rejects_prose(self):
        # Regression: the lenient path must NOT swallow strict-schema failures.
        ok, errs = validate_output(
            "not json, sorry",
            {"type": "object", "required": ["city"]},
        )
        assert ok is False
        assert errs
        assert not any("non_json_wrapped" in e for e in errs)

    def test_json_object_passes_forgiving_schema(self):
        # Sanity: forgiving schema still validates a proper JSON object cleanly
        # (no spurious "non_json_wrapped" warning).
        ok, errs = validate_output('{"city": "Berlin"}', {})
        assert ok is True
        assert errs == []


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

    tool_progress_callback = None
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
        assert entry["schema_valid"] is True
        assert "schema_errors" not in entry
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
        child._delegate_output_schema = ADDRESS_SCHEMA
        entry = _run(child)
        assert entry["schema_valid"] is False
        assert entry["schema_errors"]
        assert entry["schema_retries"] == 1
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
        assert entry["schema_valid"] is False
        assert entry["schema_errors"]

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

    def test_schema_failure_reported_as_failed_not_completed(self):
        """Regression: a final answer that still violates the declared
        output contract after the bounded retry (here the classic empty
        ``{}`` fallback) must be reported status="failed", not
        "completed". Otherwise the batch report prints a ✓ and
        orchestrators that read only status/icon accept an empty verdict
        — schema_valid/schema_errors carry the detail, but status must
        agree with them."""
        child = _StubChild(["not json at all", "{}"])
        child._delegate_output_schema = ADDRESS_SCHEMA
        entry = _run(child)
        assert entry["schema_valid"] is False
        assert entry["schema_errors"]
        assert entry["status"] == "failed"
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

    # ----- v0.21.3: forgiving-schema prose acceptance -----

    def test_forgiving_schema_with_prose_first_try_completed_with_warnings(self):
        """Empty schema {} + prose answer (the live-site bug):
        no retry turn, summary preserved verbatim, status="completed_with_warnings".
        """
        PROSE = "I did the work. Here is the report:\n- step 1\n- step 2"
        child = _StubChild([PROSE])
        child._delegate_output_schema = {}
        entry = _run(child)
        assert entry["status"] == "completed_with_warnings"
        assert entry["summary"] == PROSE
        assert entry["schema_valid"] is True
        assert any("non_json_wrapped" in w for w in entry["schema_warnings"])
        assert entry["answer_was_wrapped"] is True
        # Critical: no retry was sent — the work is already done.
        assert len(child.calls) == 1

    def test_forgiving_schema_skips_retry_turn_altogether(self):
        """Forgiving schema + prose: validator wraps on the first try and skips
        the bounded retry turn — re-prompting the child to "give JSON" wastes
        a turn on work that is already done. The child's only call is the
        original one (calls == 1)."""
        PROSE = "Same prose on retry."
        child = _StubChild([PROSE, PROSE])  # 2 responses queued, only 1 used
        child._delegate_output_schema = {}
        entry = _run(child)
        assert entry["status"] == "completed_with_warnings"
        assert entry["summary"] == PROSE
        assert entry["answer_was_wrapped"] is True
        # Critical efficiency win: no retry was sent — the work was already done.
        assert len(child.calls) == 1
        # schema_retries is NOT emitted (retries == 0 is falsy).
        assert "schema_retries" not in entry

    def test_forgiving_schema_with_json_object_still_completed(self):
        """Sanity: forgiving schema + proper JSON object → status="completed"
        (no spurious warning, no wrapped flag)."""
        child = _StubChild(['{"answer": 42}'])
        child._delegate_output_schema = {}
        entry = _run(child)
        assert entry["status"] == "completed"
        assert entry["schema_valid"] is True
        assert "schema_warnings" not in entry
        assert "answer_was_wrapped" not in entry
        assert len(child.calls) == 1

    def test_constraining_schema_with_prose_still_failed(self):
        """Regression: strict-schema + prose answer keeps the v0.21.2 contract
        of status="failed" with a schema violation error message. The lenient
        path must NOT swallow strict-schema failures (otherwise orchestrators
        would silently accept empty verdicts)."""
        child = _StubChild(["not json", "still not json"])
        child._delegate_output_schema = ADDRESS_SCHEMA
        entry = _run(child)
        assert entry["status"] == "failed"
        assert entry["schema_valid"] is False
        assert entry["schema_errors"]
        assert "output_schema" in entry.get("error", "")
        assert entry["summary"] == "still not json"
        assert "schema_warnings" not in entry
        assert "answer_was_wrapped" not in entry


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
