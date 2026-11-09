"""``schema_valid`` must describe the ``summary`` the parent actually receives.

``schema_valid`` / ``schema_errors`` are computed once, from the child's raw final text. The host then changes the
delivered summary: ``_apply_summary_budget`` replaces an over-budget summary with a head + tail + footer slice, and
an appended steer / sibling-write note can break the JSON extraction of an array answer. These tests pin that the
reported verdict is re-checked against the delivered text, and that entries that never carried a verdict are left
exactly as they were.

Stub children only: no model call, no network.
"""

import json
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import tools.delegate_tool as delegate
from tools.delegate_tool import _run_single_child
from tools.delegation_output_schema import revalidate_entry, validate_output

OBJECT_SCHEMA = {
    "type": "object",
    "properties": {"value": {"type": "string"}},
    "required": ["value"],
}
ARRAY_SCHEMA = {"type": "array", "items": {"type": "integer"}}
SECRET = "sk-ant-" + "syntheticschemacredential" * 3


class _StubChild:
    """Minimal child agent double (same shape as tests/tools/test_delegate_output_schema.py)."""

    tool_progress_callback = None
    _delegate_saved_tool_names: list = []
    _credential_pool = None
    _subagent_id = None
    _delegate_depth = 1
    _parent_subagent_id = None
    _delegate_output_schema: dict | None = None
    model = "test-model"
    session_prompt_tokens = 0
    session_completion_tokens = 0
    session_estimated_cost_usd = 0.0
    session_reasoning_tokens = 0

    def __init__(self, response, schema, **result_extra):
        self.response = response
        self._delegate_output_schema = schema
        self.result_extra = result_extra

    def get_activity_summary(self):
        return {"api_call_count": 1, "max_iterations": 5, "current_tool": None}

    def run_conversation(self, user_message, task_id=None, **_kwargs):
        return {"final_response": self.response, "completed": True, "api_calls": 1, "messages": [], **self.result_extra}

    def close(self):
        return None


class _StubParent:
    _current_task_id = None
    _delegate_depth = 0

    def _touch_activity(self, _desc):
        return None


def _mock_parent():
    parent = MagicMock()
    parent._delegate_depth = 0
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    return parent


def _run(child):
    return _run_single_child(0, "produce the structured answer", child, _StubParent())


def _delegate_task(child, schema, max_summary_chars):
    credentials = dict(provider=None, model=None, base_url=None, api_key=None, api_mode=None)
    with (
        patch("tools.delegate_tool._load_config", return_value={"max_summary_chars": max_summary_chars}),
        patch("tools.delegate_tool._resolve_delegation_credentials", return_value=credentials),
        patch("tools.delegate_tool._build_child_preserving_parent_tools", return_value=child),
    ):
        return json.loads(delegate.delegate_task(goal="Return a structured result", parent_agent=_mock_parent(), output_schema=schema))


def _reported_matches_delivered(entry, schema):
    return entry["schema_valid"] == validate_output(entry["summary"], schema)[0]


class TestRevalidateEntry:
    def test_valid_summary_stays_valid_and_gains_no_keys(self):
        entry = {"summary": '{"value": "ok"}', "schema_valid": True}
        revalidate_entry(entry, OBJECT_SCHEMA)
        assert entry == {"summary": '{"value": "ok"}', "schema_valid": True}

    def test_truncated_valid_answer_flips_to_invalid_with_a_note_naming_the_full_copy(self):
        entry = {"summary": '{"value": "aaaa\n... [truncated] ...', "schema_valid": True, "summary_full_path": "/cache/delegation/full.txt"}
        revalidate_entry(entry, OBJECT_SCHEMA)
        assert entry["schema_valid"] is False
        assert entry["schema_errors"]
        assert "/cache/delegation/full.txt" in entry["schema_note"]

    def test_flip_without_a_spill_file_still_explains_itself(self):
        entry = {"summary": '[1, 2]\n\n[NOTE: something appended]', "schema_valid": True}
        revalidate_entry(entry, ARRAY_SCHEMA)
        assert entry["schema_valid"] is False
        assert "shortened or annotated" in entry["schema_note"]

    @pytest.mark.parametrize("entry", [
        {"summary": "plain text", "status": "completed"},
        {"summary": None, "status": "error", "error": "boom"},
    ])
    def test_entries_without_a_verdict_are_untouched(self, entry):
        before = dict(entry)
        revalidate_entry(entry, OBJECT_SCHEMA)
        assert entry == before

    def test_missing_or_non_dict_schema_is_ignored(self):
        entry = {"summary": "not json", "schema_valid": True}
        revalidate_entry(entry, None)
        revalidate_entry(entry, "not a schema")
        assert entry["schema_valid"] is True

    def test_already_invalid_keeps_its_own_errors_and_note(self):
        entry = {"summary": "still not json", "schema_valid": False, "schema_errors": ["$: 'value' is a required property"], "schema_note": "existing note"}
        before = dict(entry)
        revalidate_entry(entry, OBJECT_SCHEMA)
        assert entry == before

    def test_host_changes_never_turn_a_failed_answer_into_a_pass(self):
        entry = {"summary": '{"value": "ok"}', "schema_valid": False, "schema_errors": ["old"], "schema_note": "old note"}
        before = dict(entry)
        revalidate_entry(entry, OBJECT_SCHEMA)
        assert entry == before

    def test_a_validator_fault_reports_invalid_without_echoing_the_exception(self, caplog):
        entry = {"summary": '{"value": "ok"}', "schema_valid": True}
        with patch("tools.delegation_output_schema.validate_output", side_effect=RuntimeError("validator exploded " + SECRET)):
            revalidate_entry(entry, OBJECT_SCHEMA)
        assert entry["schema_valid"] is False
        assert entry["schema_errors"] == ["Could not re-validate the delivered summary."]
        assert "re-validation failed (RuntimeError)" in caplog.text
        assert SECRET not in caplog.text and SECRET not in json.dumps(entry)


class TestDeliveredSummaryIsWhatIsReported:
    def test_budget_truncation_no_longer_leaves_a_stale_true(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        original = json.dumps({"value": "a" * 10000})
        assert validate_output(original, OBJECT_SCHEMA)[0]
        payload = _delegate_task(_StubChild(original, OBJECT_SCHEMA), OBJECT_SCHEMA, max_summary_chars=2000)
        entry = payload["results"][0]
        assert entry["summary_truncated"] is True, "the probe must really hit the summary budget"
        assert entry["schema_valid"] is False
        assert _reported_matches_delivered(entry, OBJECT_SCHEMA)
        assert entry["summary_full_path"] in entry["schema_note"]
        # the spilled full copy is the complete, valid answer
        assert validate_output(open(entry["summary_full_path"], encoding="utf-8-sig").read(), OBJECT_SCHEMA)[0]

    def test_an_answer_inside_the_budget_is_reported_valid(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        payload = _delegate_task(_StubChild('{"value": "small"}', OBJECT_SCHEMA), OBJECT_SCHEMA, max_summary_chars=2000)
        entry = payload["results"][0]
        assert not entry.get("summary_truncated")
        assert entry["schema_valid"] is True
        assert "schema_note" not in entry and "schema_errors" not in entry

    def test_a_missed_steer_note_that_breaks_an_array_answer_is_reported(self):
        child = _StubChild("[1, 2, 3]", ARRAY_SCHEMA, pending_steer="focus on the second item")
        entry = _run(child)
        assert "steer did not land" in entry["summary"]
        assert entry["schema_valid"] is False
        assert _reported_matches_delivered(entry, ARRAY_SCHEMA)

    def test_a_sibling_write_note_that_breaks_an_array_answer_is_reported(self):
        def _append_note(self, entry):
            entry["summary"] += "\n\n[NOTE: subagent modified files the parent previously read — re-read before editing: a.py]"

        with patch("tools.delegate_tool_child_run._ChildRun.append_sibling_write_reminder", _append_note):
            entry = _run(_StubChild("[1, 2, 3]", ARRAY_SCHEMA))
        assert entry["schema_valid"] is False
        assert _reported_matches_delivered(entry, ARRAY_SCHEMA)

    def test_json_inside_a_late_steer_does_not_certify_a_failed_answer(self):
        child = _StubChild("Sorry, still unable.", OBJECT_SCHEMA, pending_steer='Return {"value": "draft"} if stuck')
        entry = _run(child)
        assert '{"value": "draft"}' in entry["summary"]
        assert entry["schema_valid"] is False
        assert entry["schema_errors"] and entry["schema_note"]

    def test_an_object_answer_survives_a_plain_note(self):
        child = _StubChild('{"value": "ok"}', OBJECT_SCHEMA, pending_steer="focus on the second item")
        entry = _run(child)
        assert "steer did not land" in entry["summary"]
        assert entry["schema_valid"] is True
        assert _reported_matches_delivered(entry, OBJECT_SCHEMA)

    def test_schema_less_calls_keep_their_exact_wire_shape(self):
        entry = _run(_StubChild("plain answer", None))
        assert not any(key.startswith("schema") for key in entry)


class TestRedactionThenBudget:
    """Child output is redacted before the budget (tools/delegation_redaction.py), which re-checks the verdict only
    when redaction changed the summary; a truncation after that still needs its own re-check, and must not discard
    the note redaction left behind."""

    def test_redaction_keeps_the_answer_valid_and_truncation_then_flips_it(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        original = json.dumps({"value": SECRET + " " + "a" * 10000})
        payload = _delegate_task(_StubChild(original, OBJECT_SCHEMA), OBJECT_SCHEMA, max_summary_chars=2000)
        entry = payload["results"][0]
        assert entry["summary_truncated"] is True
        assert entry["schema_valid"] is False
        assert _reported_matches_delivered(entry, OBJECT_SCHEMA)
        assert "shortened or annotated" in entry["schema_note"]
        assert SECRET not in json.dumps(entry)
        spilled = open(entry["summary_full_path"], encoding="utf-8-sig").read()
        assert SECRET not in spilled and validate_output(spilled, OBJECT_SCHEMA)[0]

    def test_a_verdict_flipped_by_redaction_keeps_its_note_and_errors_through_truncation(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        schema = {
            "type": "object",
            "properties": {"value": {"const": SECRET}, "pad": {"type": "string"}},
            "required": ["value", "pad"],
        }
        original = json.dumps({"value": SECRET, "pad": "z" * 10000})
        assert validate_output(original, schema)[0]
        payload = _delegate_task(_StubChild(original, schema), schema, max_summary_chars=2000)
        entry = payload["results"][0]
        assert entry["summary_truncated"] is True
        assert entry["schema_valid"] is False
        assert "Secret redaction" in entry["schema_note"]
        assert not any("not valid JSON" in error for error in entry["schema_errors"])
        assert SECRET not in json.dumps(entry)

    def test_validator_messages_after_truncation_are_redacted_like_the_rest_of_the_parent_copy(self, tmp_path, monkeypatch):
        """jsonschema messages quote values (an enum/const from the schema, a value from the parsed candidate); the
        re-check after the budget runs after the redaction projection, so its errors must get the same projection."""
        import tools.delegate_tool_results as results
        from tests.tools.test_delegated_output_redaction import _Parent

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(delegate, "_load_config", lambda: {"max_summary_chars": 2000})
        schema = {"type": "object", "properties": {"value": {"enum": [SECRET, "ok"]}}, "required": ["value"]}
        # The tail that survives the budget still holds a parseable object, so the validator reports the enum.
        entry = {"task_index": 0, "status": "completed", "schema_valid": True,
                 "summary": "ordinary explanation\n" * 400 + '{"value": "bad"}'}
        child = SimpleNamespace(_delegate_output_schema=schema)
        results._finalize_child_results([entry], [{"goal": "g"}], [(0, {}, child)], _Parent())
        assert entry["summary_truncated"] is True
        assert entry["schema_valid"] is False and entry["schema_errors"]
        assert SECRET not in json.dumps(entry)
