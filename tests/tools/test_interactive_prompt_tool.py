"""Tests for tools/interactive_prompt_tool.py - Rich interactive prompts."""

import json
from typing import Dict, Any, List
from unittest.mock import patch

from tools.interactive_prompt_tool import (
    interactive_prompt_tool,
    check_interactive_prompt_requirements,
    MAX_OPTIONS,
    MAX_LABEL_LEN,
    MAX_VALUE_LEN,
    MAX_DESC_LEN,
    MAX_MODAL_TITLE_LEN,
    MAX_MODAL_FIELDS,
    MIN_MODAL_FIELDS,
    MAX_QUESTION_LEN,
    INTERACTIVE_PROMPT_SCHEMA,
)
from tools.human_input_gateway import HumanInputResult, ActorInfo, FileResult


def _make_options(n: int, action: str = "return") -> List[Dict[str, Any]]:
    """Build N minimal option dicts for testing."""
    opts = []
    for i in range(n):
        opt: Dict[str, Any] = {
            "label": f"Option {i}",
            "value": f"opt_{i}",
        }
        if action == "modal":
            opt["action"] = "modal"
            opt["modal"] = {
                "title": f"Modal {i}",
                "fields": [
                    {"key": "name", "label": "Name"},
                ],
            }
        opts.append(opt)
    return opts


class TestInteractivePromptBasics:
    """Basic functionality tests for interactive_prompt_tool."""

    def test_simple_return_option(self):
        """Should return JSON with the user's choice for a return option."""
        def mock_cb(q, opts, dt, ts, ap):
            return HumanInputResult(
                status="resolved",
                choice="opt_0",
                timed_out=False,
                actor=ActorInfo(platform="test", user_id="u1", display_name="Tester"),
            )

        result = json.loads(interactive_prompt_tool(
            "Pick one",
            options=_make_options(3),
            callback=mock_cb,
        ))
        assert result["status"] == "resolved"
        assert result["choice"] == "opt_0"
        assert result["timed_out"] is False
        assert result["actor"]["display_name"] == "Tester"

    def test_modal_option_with_fields(self):
        """Should return fields from a modal response."""
        def mock_cb(q, opts, dt, ts, ap):
            return HumanInputResult(
                status="resolved",
                choice="opt_0",
                timed_out=False,
                actor=ActorInfo(platform="test", user_id="u1", display_name="Tester"),
                fields={"name": "Alice"},
            )

        result = json.loads(interactive_prompt_tool(
            "Fill in",
            options=_make_options(2, action="modal"),
            callback=mock_cb,
        ))
        assert result["fields"]["name"] == "Alice"

    def test_modal_option_with_files(self):
        """Should return file metadata from a modal file upload."""
        def mock_cb(q, opts, dt, ts, ap):
            return HumanInputResult(
                status="resolved",
                choice="opt_0",
                timed_out=False,
                actor=ActorInfo(platform="test", user_id="u1", display_name="Tester"),
                files=[
                    FileResult(
                        field_key="document",
                        attachment_id="att_123",
                        filename="report.pdf",
                        content_type="application/pdf",
                        size=4096,
                        cached_path="/tmp/report.pdf",
                    )
                ],
            )

        result = json.loads(interactive_prompt_tool(
            "Upload",
            options=_make_options(1, action="modal"),
            callback=mock_cb,
        ))
        assert result["files"][0]["filename"] == "report.pdf"
        assert result["files"][0]["size"] == 4096

    def test_empty_question_returns_error(self):
        """Should return error for empty question."""
        result = json.loads(interactive_prompt_tool(
            "",
            options=_make_options(1),
            callback=lambda *a: "ignored",
        ))
        assert "error" in result
        assert "required" in result["error"].lower()

    def test_whitespace_only_question_returns_error(self):
        """Should return error for whitespace-only question."""
        result = json.loads(interactive_prompt_tool(
            "   \n\t  ",
            options=_make_options(1),
            callback=lambda *a: "ignored",
        ))
        assert "error" in result

    def test_question_too_long_returns_error(self):
        """Should reject questions exceeding MAX_QUESTION_LEN."""
        long_q = "x" * (MAX_QUESTION_LEN + 1)
        result = json.loads(interactive_prompt_tool(
            long_q,
            options=_make_options(1),
            callback=lambda *a: "ignored",
        ))
        assert "error" in result
        assert "too long" in result["error"].lower()

    def test_question_at_max_length_is_accepted(self):
        """Questions at exactly MAX_QUESTION_LEN should pass validation."""
        q = "y" * MAX_QUESTION_LEN
        result = json.loads(interactive_prompt_tool(
            q,
            options=_make_options(1),
            callback=lambda *a: HumanInputResult(
                status="selected",
                choice="opt_0",
                timed_out=False,
                actor=ActorInfo(platform="test", user_id="u1", display_name="T"),
            ),
        ))
        assert "status" in result

    def test_no_options_returns_error(self):
        """Should return error when no options provided."""
        result = json.loads(interactive_prompt_tool(
            "Question?",
            options=[],
            callback=lambda *a: "ignored",
        ))
        assert "error" in result
        assert "non-empty" in result["error"].lower()

    def test_no_callback_returns_error(self):
        """Should return error when no callback is provided."""
        result = json.loads(interactive_prompt_tool(
            "Question?",
            options=_make_options(1),
        ))
        assert "error" in result
        assert "not available" in result["error"].lower()


class TestInteractivePromptOptionsValidation:
    """Tests for options parameter validation."""

    def test_options_not_list_returns_error(self):
        """Non-list options should return error."""
        result = json.loads(interactive_prompt_tool(
            "Q?",
            options="not a list",  # type: ignore
            callback=lambda *a: "ignored",
        ))
        assert "error" in result
        assert "list" in result["error"].lower()

    def test_too_many_options_returns_error(self):
        """Options exceeding MAX_OPTIONS should return error."""
        result = json.loads(interactive_prompt_tool(
            "Pick",
            options=_make_options(MAX_OPTIONS + 1),
            callback=lambda *a: "ignored",
        ))
        assert "error" in result
        assert "too many" in result["error"].lower()

    def test_option_not_dict_returns_error(self):
        """Non-dict option should return error."""
        result = json.loads(interactive_prompt_tool(
            "Q?",
            options=["not a dict"],  # type: ignore
            callback=lambda *a: "ignored",
        ))
        assert "error" in result
        assert "must be a dict" in result["error"].lower()

    def test_option_missing_label_returns_error(self):
        """Option without label should return error."""
        result = json.loads(interactive_prompt_tool(
            "Q?",
            options=[{"value": "x"}],
            callback=lambda *a: "ignored",
        ))
        assert "error" in result
        assert "label" in result["error"].lower()

    def test_option_missing_value_returns_error(self):
        """Option without value should return error."""
        result = json.loads(interactive_prompt_tool(
            "Q?",
            options=[{"label": "My Option"}],
            callback=lambda *a: "ignored",
        ))
        assert "error" in result
        assert "value" in result["error"].lower()

    def test_option_label_too_long(self):
        """Option label exceeding MAX_LABEL_LEN should return error."""
        result = json.loads(interactive_prompt_tool(
            "Q?",
            options=[{"label": "x" * (MAX_LABEL_LEN + 1), "value": "v"}],
            callback=lambda *a: "ignored",
        ))
        assert "error" in result
        assert "label" in result["error"].lower()

    def test_option_value_too_long(self):
        """Option value exceeding MAX_VALUE_LEN should return error."""
        result = json.loads(interactive_prompt_tool(
            "Q?",
            options=[{"label": "L", "value": "x" * (MAX_VALUE_LEN + 1)}],
            callback=lambda *a: "ignored",
        ))
        assert "error" in result
        assert "value" in result["error"].lower()

    def test_option_description_too_long(self):
        """Option description exceeding MAX_DESC_LEN should return error."""
        result = json.loads(interactive_prompt_tool(
            "Q?",
            options=[{"label": "L", "value": "v", "description": "d" * (MAX_DESC_LEN + 1)}],
            callback=lambda *a: "ignored",
        ))
        assert "error" in result
        assert "description" in result["error"].lower()


class TestInteractivePromptModalValidation:
    """Tests for modal action validation."""

    def test_modal_missing_modal_object(self):
        """Modal action without modal dict should return error."""
        result = json.loads(interactive_prompt_tool(
            "Q?",
            options=[{"label": "L", "value": "v", "action": "modal"}],
            callback=lambda *a: "ignored",
        ))
        assert "error" in result
        assert "modal" in result["error"].lower()

    def test_modal_missing_title(self):
        """Modal without title should return error."""
        result = json.loads(interactive_prompt_tool(
            "Q?",
            options=[{
                "label": "L", "value": "v", "action": "modal",
                "modal": {"fields": [{"key": "k", "label": "K"}]},
            }],
            callback=lambda *a: "ignored",
        ))
        assert "error" in result
        assert "title" in result["error"].lower()

    def test_modal_title_too_long(self):
        """Modal title exceeding MAX_MODAL_TITLE_LEN should return error."""
        result = json.loads(interactive_prompt_tool(
            "Q?",
            options=[{
                "label": "L", "value": "v", "action": "modal",
                "modal": {
                    "title": "T" * (MAX_MODAL_TITLE_LEN + 1),
                    "fields": [{"key": "k", "label": "K"}],
                },
            }],
            callback=lambda *a: "ignored",
        ))
        assert "error" in result
        assert "title" in result["error"].lower()

    def test_modal_fields_not_list(self):
        """Modal with non-list fields should return error."""
        result = json.loads(interactive_prompt_tool(
            "Q?",
            options=[{
                "label": "L", "value": "v", "action": "modal",
                "modal": {"title": "T", "fields": "not a list"},
            }],
            callback=lambda *a: "ignored",
        ))
        assert "error" in result
        assert "fields" in result["error"].lower()

    def test_modal_too_many_fields(self):
        """Modal exceeding MAX_MODAL_FIELDS should return error."""
        fields = [{"key": f"f{i}", "label": f"F{i}"} for i in range(MAX_MODAL_FIELDS + 1)]
        result = json.loads(interactive_prompt_tool(
            "Q?",
            options=[{
                "label": "L", "value": "v", "action": "modal",
                "modal": {"title": "T", "fields": fields},
            }],
            callback=lambda *a: "ignored",
        ))
        assert "error" in result
        assert "fields" in result["error"].lower()

    def test_modal_too_few_fields(self):
        """Modal with fewer than MIN_MODAL_FIELDS should return error."""
        result = json.loads(interactive_prompt_tool(
            "Q?",
            options=[{
                "label": "L", "value": "v", "action": "modal",
                "modal": {"title": "T", "fields": []},
            }],
            callback=lambda *a: "ignored",
        ))
        assert "error" in result
        assert "fields" in result["error"].lower()


class TestInteractivePromptCallbackHandling:
    """Tests for callback error handling."""

    def test_callback_exception_returns_error(self):
        """Should return error if callback raises exception."""
        def failing_cb(q, opts, dt, ts, ap):
            raise RuntimeError("User cancelled")

        result = json.loads(interactive_prompt_tool(
            "Q?",
            options=_make_options(1),
            callback=failing_cb,
        ))
        assert "error" in result
        assert "Failed to get user input" in result["error"]

    def test_callback_receives_stripped_question(self):
        """Callback should receive trimmed question."""
        received = []

        def mock_cb(q, opts, dt, ts, ap):
            received.append(q)
            return HumanInputResult(status="resolved", choice="x", timed_out=False)

        interactive_prompt_tool(
            "  Question with spaces  \n",
            options=_make_options(1),
            callback=mock_cb,
        )
        assert received[0] == "Question with spaces"

    def test_callback_receives_display_type(self):
        """Callback should receive the display_type parameter."""
        received = []

        def mock_cb(q, opts, dt, ts, ap):
            received.append(dt)
            return HumanInputResult(status="resolved", choice="x", timed_out=False)

        interactive_prompt_tool(
            "Q?", options=_make_options(1),
            display_type="buttons", callback=mock_cb,
        )
        assert received[0] == "buttons"

    def test_callback_dict_result(self):
        """Should handle callback returning a plain dict (fallback path)."""
        def dict_cb(q, opts, dt, ts, ap):
            return {"status": "ok", "data": "something"}

        result = json.loads(interactive_prompt_tool(
            "Q?", options=_make_options(1), callback=dict_cb,
        ))
        assert result["status"] == "ok"
        assert result["data"] == "something"


class TestCheckRequirements:
    """Tests for the feature-flag requirements check."""

    def test_disabled_by_default(self):
        """Tool is disabled when config key is absent."""
        with patch("hermes_cli.config.load_config", return_value={}):
            assert check_interactive_prompt_requirements() is False

    def test_enabled_when_config_true(self):
        """Tool is enabled when agent.interactive_prompt_enabled=True."""
        with patch("hermes_cli.config.load_config",
                   return_value={"agent": {"interactive_prompt_enabled": True}}):
            assert check_interactive_prompt_requirements() is True

    def test_disabled_when_config_false(self):
        """Tool is disabled when agent.interactive_prompt_enabled=False."""
        with patch("hermes_cli.config.load_config",
                   return_value={"agent": {"interactive_prompt_enabled": False}}):
            assert check_interactive_prompt_requirements() is False

    def test_falls_back_on_exception(self):
        """Config load failure → disabled (safe default)."""
        with patch("hermes_cli.config.load_config", side_effect=Exception("boom")):
            assert check_interactive_prompt_requirements() is False


class TestSchema:
    """Tests for the OpenAI function-calling schema."""

    def test_schema_name(self):
        assert INTERACTIVE_PROMPT_SCHEMA["name"] == "interactive_prompt"

    def test_schema_has_description(self):
        assert "description" in INTERACTIVE_PROMPT_SCHEMA
        assert len(INTERACTIVE_PROMPT_SCHEMA["description"]) > 50

    def test_schema_question_required(self):
        assert "question" in INTERACTIVE_PROMPT_SCHEMA["parameters"]["required"]

    def test_schema_options_required(self):
        assert "options" in INTERACTIVE_PROMPT_SCHEMA["parameters"]["required"]

    def test_schema_display_type_enum(self):
        props = INTERACTIVE_PROMPT_SCHEMA["parameters"]["properties"]
        assert "buttons" in props["display_type"]["enum"]
        assert "select" not in props["display_type"]["enum"]


# ===========================================================================
# file_policy validation tests
# ===========================================================================


class TestFilePolicyValidation:
    """Validation guardrails for file_policy on file_upload fields."""

    def _call(self, options):
        from tools.interactive_prompt_tool import interactive_prompt_tool
        return interactive_prompt_tool(
            question="Upload test",
            options=options,
            callback=lambda *a, **kw: None,
        )

    def _file_upload_field(self, **policy_kwargs):
        field = {"key": "f", "label": "File", "type": "file_upload"}
        if policy_kwargs:
            field["file_policy"] = policy_kwargs
        return {
            "label": "Upload",
            "value": "upload",
            "action": "modal",
            "modal": {"title": "Upload", "fields": [field]},
        }

    def test_valid_file_policy_passes(self):
        """Valid file_policy constraints are accepted."""
        result = self._call([
            self._file_upload_field(
                max_files=5, max_bytes=10_000_000,
                allowed_extensions=[".pdf", ".png"],
            ),
        ])
        # callback returns None → "result" is json.dumps(None) = "null"
        assert json.loads(result) is None

    def test_valid_file_policy_no_policy_passes(self):
        """file_upload field with no file_policy is accepted."""
        result = self._call([
            self._file_upload_field(),
        ])
        assert json.loads(result) is None

    def test_max_files_zero_rejected(self):
        result = self._call([self._file_upload_field(max_files=0)])
        assert "max_files" in result
        assert "1–10" in result

    def test_max_files_eleven_rejected(self):
        result = self._call([self._file_upload_field(max_files=11)])
        assert "max_files" in result
        assert "1–10" in result

    def test_max_files_negative_rejected(self):
        result = self._call([self._file_upload_field(max_files=-1)])
        assert "max_files" in result

    def test_max_bytes_zero_rejected(self):
        result = self._call([self._file_upload_field(max_bytes=0)])
        assert "max_bytes" in result

    def test_max_bytes_negative_rejected(self):
        result = self._call([self._file_upload_field(max_bytes=-100)])
        assert "max_bytes" in result

    def test_allowed_extensions_without_dot_rejected(self):
        result = self._call([self._file_upload_field(allowed_extensions=["pdf"])])
        assert "allowed_extensions" in result
        assert "starting with" in result

    def test_allowed_extensions_string_rejected(self):
        result = self._call([self._file_upload_field(allowed_extensions=".pdf")])
        assert "allowed_extensions" in result

    def test_allowed_extensions_valid(self):
        result = self._call([
            self._file_upload_field(allowed_extensions=[".pdf", ".docx"]),
        ])
        assert json.loads(result) is None

    def test_file_policy_ignored_for_non_file_fields(self):
        """file_policy on a text field doesn't cause validation errors."""
        field = {
            "key": "name", "label": "Name", "type": "text",
            "file_policy": {"max_files": 999},
        }
        opt = {
            "label": "Text", "value": "text", "action": "modal",
            "modal": {"title": "T", "fields": [field]},
        }
        result = self._call([opt])
        assert json.loads(result) is None
