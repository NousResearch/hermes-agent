# tests/tools/test_clarify_rich_options.py
"""Unit tests for the rich-options extension of the clarify tool.

Covers:
  * _validate_options() — all validation paths
  * clarify_tool() — mutual exclusivity, rich callback dispatch, JSON passthrough
  * CLARIFY_SCHEMA — new properties present with correct constraints
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from tools.clarify_gateway import _ClarifyEntry, _coerce_text_response
from tools.clarify_tool import (
    CLARIFY_SCHEMA,
    MAX_OPTIONS,
    MAX_LABEL_LEN,
    MAX_VALUE_LEN,
    MAX_DESC_LEN,
    MAX_MODAL_TITLE_LEN,
    MAX_MODAL_FIELDS,
    MIN_MODAL_FIELDS,
    MAX_QUESTION_LEN,
    _validate_options,
    clarify_tool,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_option(
    label: str = "Approve",
    value: str = "approve",
    action: str = "return",
    **extra: Any,
) -> dict:
    """Return a single valid option dict."""
    opt = {"label": label, "value": value, "action": action}
    opt.update(extra)
    return opt


def _rich_callback_return(question, **kwargs):
    """Simulate a callback that returns a JSON result string."""
    return json.dumps({
        "status": "answered",
        "value": kwargs.get("options", [{}])[0].get("value", ""),
    })


# ===========================================================================
# Schema tests
# ===========================================================================

class TestClarifySchema:
    """Verify the OpenAI schema includes rich-option properties."""

    def test_schema_has_options(self):
        assert "options" in CLARIFY_SCHEMA["parameters"]["properties"]

    def test_schema_has_display_type(self):
        assert "display_type" in CLARIFY_SCHEMA["parameters"]["properties"]

    def test_schema_has_auth_policy(self):
        assert "auth_policy" in CLARIFY_SCHEMA["parameters"]["properties"]

    def test_schema_has_timeout_seconds(self):
        assert "timeout_seconds" in CLARIFY_SCHEMA["parameters"]["properties"]

    def test_options_max_items(self):
        assert CLARIFY_SCHEMA["parameters"]["properties"]["options"]["maxItems"] == MAX_OPTIONS

    def test_options_min_items(self):
        assert CLARIFY_SCHEMA["parameters"]["properties"]["options"]["minItems"] == 1

    def test_question_still_required(self):
        assert CLARIFY_SCHEMA["parameters"]["required"] == ["question"]

    def test_options_item_has_required_fields(self):
        item_schema = CLARIFY_SCHEMA["parameters"]["properties"]["options"]["items"]
        assert "label" in item_schema["required"]
        assert "value" in item_schema["required"]
        assert "action" in item_schema["required"]

    def test_timeout_seconds_minimum(self):
        ts = CLARIFY_SCHEMA["parameters"]["properties"]["timeout_seconds"]
        assert ts["minimum"] == 60

    def test_timeout_seconds_maximum(self):
        ts = CLARIFY_SCHEMA["parameters"]["properties"]["timeout_seconds"]
        assert ts["maximum"] == 3600


# ===========================================================================
# _validate_options tests
# ===========================================================================

class TestValidateOptions:
    """Exercise every validation branch in _validate_options()."""

    # -- happy paths ---------------------------------------------------------

    def test_single_valid_option(self):
        assert _validate_options([_valid_option()]) is None

    def test_25_options_ok(self):
        opts = [_valid_option(label=f"L{i}", value=f"v{i}") for i in range(MAX_OPTIONS)]
        assert _validate_options(opts) is None

    def test_option_with_description(self):
        opt = _valid_option(description="Some description")
        assert _validate_options([opt]) is None

    def test_option_with_style(self):
        for style in ("primary", "secondary", "success", "danger"):
            opt = _valid_option(style=style)
            assert _validate_options([opt]) is None

    def test_option_with_modal(self):
        opt = _valid_option(
            action="modal",
            modal={
                "title": "My Form",
                "fields": [
                    {"key": "name", "label": "Name", "type": "text"},
                ],
            },
        )
        assert _validate_options([opt]) is None

    def test_modal_with_5_fields(self):
        fields = [
            {"key": f"f{i}", "label": f"F{i}", "type": "text"}
            for i in range(MAX_MODAL_FIELDS)
        ]
        opt = _valid_option(action="modal", modal={"title": "T", "fields": fields})
        assert _validate_options([opt]) is None

    # -- failure paths -------------------------------------------------------

    def test_empty_list(self):
        err = _validate_options([])
        assert err is not None
        assert "non-empty" in err.lower()

    def test_none(self):
        err = _validate_options(None)
        assert err is not None

    def test_not_a_list(self):
        err = _validate_options("not a list")
        assert err is not None

    def test_too_many_options(self):
        opts = [_valid_option(label=f"L{i}", value=f"v{i}") for i in range(MAX_OPTIONS + 1)]
        err = _validate_options(opts)
        assert err is not None
        assert "maximum" in err.lower() or "too many" in err.lower()

    def test_option_not_dict(self):
        err = _validate_options(["not a dict"])
        assert err is not None
        assert "must be a dict" in err.lower()

    def test_missing_label(self):
        opt = _valid_option()
        del opt["label"]
        err = _validate_options([opt])
        assert err is not None
        assert "label" in err.lower()

    def test_missing_value(self):
        opt = _valid_option()
        del opt["value"]
        err = _validate_options([opt])
        assert err is not None
        assert "value" in err.lower()

    def test_empty_label(self):
        opt = _valid_option(label="   ")
        err = _validate_options([opt])
        assert err is not None
        assert "label" in err.lower()

    def test_label_too_long(self):
        opt = _valid_option(label="x" * (MAX_LABEL_LEN + 1))
        err = _validate_options([opt])
        assert err is not None
        assert "label" in err.lower() and "exceeds" in err.lower()

    def test_value_too_long(self):
        opt = _valid_option(value="x" * (MAX_VALUE_LEN + 1))
        err = _validate_options([opt])
        assert err is not None
        assert "value" in err.lower() and "exceeds" in err.lower()

    def test_description_too_long(self):
        opt = _valid_option(description="x" * (MAX_DESC_LEN + 1))
        err = _validate_options([opt])
        assert err is not None
        assert "description" in err.lower() and "exceeds" in err.lower()

    def test_invalid_style(self):
        opt = _valid_option(style="rainbow")
        err = _validate_options([opt])
        assert err is not None
        assert "style" in err.lower()

    def test_invalid_action(self):
        opt = _valid_option(action="teleport")
        err = _validate_options([opt])
        assert err is not None
        assert "action" in err.lower()

    def test_modal_action_without_modal(self):
        opt = _valid_option(action="modal")
        # modal key absent
        err = _validate_options([opt])
        assert err is not None
        assert "modal" in err.lower()

    def test_modal_missing_title(self):
        opt = _valid_option(action="modal", modal={"fields": [{"key": "k", "label": "L", "type": "text"}]})
        err = _validate_options([opt])
        assert err is not None
        assert "title" in err.lower()

    def test_modal_title_too_long(self):
        opt = _valid_option(
            action="modal",
            modal={"title": "x" * (MAX_MODAL_TITLE_LEN + 1), "fields": [{"key": "k", "label": "L", "type": "text"}]},
        )
        err = _validate_options([opt])
        assert err is not None
        assert "title" in err.lower()

    def test_modal_too_many_fields(self):
        fields = [
            {"key": f"f{i}", "label": f"F{i}", "type": "text"}
            for i in range(MAX_MODAL_FIELDS + 1)
        ]
        opt = _valid_option(action="modal", modal={"title": "T", "fields": fields})
        err = _validate_options([opt])
        assert err is not None

    def test_modal_duplicate_keys(self):
        opt = _valid_option(
            action="modal",
            modal={
                "title": "T",
                "fields": [
                    {"key": "dup", "label": "A", "type": "text"},
                    {"key": "dup", "label": "B", "type": "text"},
                ],
            },
        )
        err = _validate_options([opt])
        assert err is not None
        assert "duplicate" in err.lower()

    def test_modal_invalid_field_type(self):
        opt = _valid_option(
            action="modal",
            modal={
                "title": "T",
                "fields": [{"key": "k", "label": "L", "type": "color_picker"}],
            },
        )
        err = _validate_options([opt])
        assert err is not None
        assert "type" in err.lower()


# ===========================================================================
# clarify_tool() integration tests
# ===========================================================================

class TestClarifyToolRichPath:
    """End-to-end tests for clarify_tool() with rich options."""

    def test_mutual_exclusivity_error(self):
        """Both choices and options → error."""
        result = clarify_tool(
            question="Pick one",
            choices=["A", "B"],
            options=[_valid_option()],
            callback=lambda *a, **kw: "should not be called",
        )
        data = json.loads(result)
        assert "error" in data
        assert "both" in data["error"].lower() or "not both" in data["error"].lower()

    def test_empty_question_error(self):
        result = clarify_tool(question="", options=[_valid_option()])
        data = json.loads(result)
        assert "error" in data

    def test_whitespace_question_error(self):
        result = clarify_tool(question="   ", options=[_valid_option()])
        data = json.loads(result)
        assert "error" in data

    def test_invalid_options_returns_error(self):
        result = clarify_tool(
            question="Q?",
            options=[{"missing": "fields"}],
            callback=lambda *a, **kw: "",
        )
        data = json.loads(result)
        assert "error" in data

    def test_rich_callback_dispatched(self):
        """Rich path calls callback with options kwarg, not positional choices."""
        captured = {}

        def _cb(question, choices=None, options=None, **kw):
            captured["question"] = question
            captured["choices"] = choices
            captured["options"] = options
            return json.dumps({"status": "answered", "value": options[0]["value"]})

        result = clarify_tool(
            question="Approve?",
            options=[_valid_option()],
            callback=_cb,
        )
        assert captured["question"] == "Approve?"
        assert captured["choices"] is None
        assert captured["options"] is not None
        # JSON result should pass through
        parsed = json.loads(result)
        assert parsed["status"] == "answered"
        assert parsed["value"] == "approve"

    def test_invalid_display_type(self):
        result = clarify_tool(
            question="Q?",
            options=[_valid_option()],
            display_type="dropdown",
            callback=lambda *a, **kw: "",
        )
        data = json.loads(result)
        assert "error" in data

    def test_invalid_auth_policy(self):
        result = clarify_tool(
            question="Q?",
            options=[_valid_option()],
            auth_policy="everyone_allowed",
            callback=lambda *a, **kw: "",
        )
        data = json.loads(result)
        assert "error" in data

    def test_timeout_clamped(self):
        """timeout_seconds below 60 is clamped to 60."""
        captured = {}

        def _cb(question, choices=None, options=None, timeout_seconds=None, **kw):
            captured["timeout"] = timeout_seconds
            return json.dumps({"status": "answered", "value": "v"})

        clarify_tool(
            question="Q?",
            options=[_valid_option()],
            timeout_seconds=30,  # below minimum
            callback=_cb,
        )
        assert captured["timeout"] == 60

    def test_no_callback_returns_context_error(self):
        result = clarify_tool(question="Q?", options=[_valid_option()])
        data = json.loads(result)
        assert "error" in data
        assert "not available" in data["error"].lower() or "context" in data["error"].lower()

    def test_question_too_long(self):
        result = clarify_tool(
            question="x" * (MAX_QUESTION_LEN + 1),
            options=[_valid_option()],
        )
        data = json.loads(result)
        assert "error" in data


class TestClarifyToolSimplePathUnchanged:
    """Verify the simple choices path is unchanged after the refactor."""

    def test_simple_choices_callback(self):
        captured = {}

        def _cb(question, choices=None, **kw):
            captured["question"] = question
            captured["choices"] = choices
            return "user answer"

        result = clarify_tool(
            question="Pick",
            choices=["A", "B"],
            callback=_cb,
        )
        data = json.loads(result)
        assert data["question"] == "Pick"
        assert data["choices_offered"] == ["A", "B"]
        assert data["user_response"] == "user answer"
        # Callback should NOT receive options kwarg
        assert "options" not in captured or captured.get("options") is None

    def test_simple_open_ended(self):
        def _cb(question, choices=None, **kw):
            return "free text"

        result = clarify_tool(question="Tell me", callback=_cb)
        data = json.loads(result)
        assert data["choices_offered"] is None
        assert data["user_response"] == "free text"


# ===========================================================================
# Session-owner capture (T1 — tools/clarify_gateway.py)
# ===========================================================================

def _clear_clarify_state():
    """Reset module-level state between primitive tests."""
    from tools import clarify_gateway as cm
    with cm._lock:
        cm._entries.clear()
        cm._session_index.clear()
        cm._notify_cbs.clear()


def _read_module_source(path: str) -> str:
    """Read a source file as text with an explicit UTF-8 encoding."""
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


class TestSessionOwnerCapture:
    """T1: ``_ClarifyEntry`` carries the session owner and ``register()`` persists it.

    Foundation for Discord's ``session_owner_only`` auth fast-path (T2 reads
    this field via ``_ClarifyEntry.signature()``).  No adapter behaviour change
    here — the simple-choices / no-owner paths stay byte-identical (default None).
    """

    def setup_method(self):
        _clear_clarify_state()

    def test_register_with_session_owner_exposes_on_entry(self):
        from tools import clarify_gateway as cm

        entry = cm.register(
            "id-owner-1", "sk-owner", "Q?", None,
            session_owner_user_id="42",
        )
        assert entry.session_owner_user_id == "42"

    def test_register_with_session_owner_exposes_in_signature(self):
        from tools import clarify_gateway as cm

        entry = cm.register(
            "id-owner-2", "sk-owner", "Q?", None,
            session_owner_user_id="42",
        )
        sig = entry.signature()
        assert sig["session_owner_user_id"] == "42"

    def test_register_without_session_owner_defaults_none(self):
        """No regression for simple-choices callers that never pass the kwarg."""
        from tools import clarify_gateway as cm

        entry = cm.register("id-owner-3", "sk-owner", "Q?", ["A", "B"])
        assert entry.session_owner_user_id is None

    def test_signature_without_session_owner_is_none(self):
        from tools import clarify_gateway as cm

        entry = cm.register("id-owner-4", "sk-owner", "Q?", ["A", "B"])
        sig = entry.signature()
        assert sig["session_owner_user_id"] is None

    def test_get_owner_user_id_returns_captured_owner(self):
        """The public accessor exposes the owner without dipping into _entries."""
        from tools import clarify_gateway as cm

        cm.register(
            "id-get-owner", "sk-owner", "Q?", None,
            session_owner_user_id="42",
        )
        assert cm.get_owner_user_id("id-get-owner") == "42"

    def test_get_owner_user_id_none_when_owner_absent(self):
        from tools import clarify_gateway as cm

        cm.register("id-get-owner-none", "sk-owner", "Q?", ["A"])
        assert cm.get_owner_user_id("id-get-owner-none") is None

    def test_get_owner_user_id_none_when_entry_missing(self):
        from tools import clarify_gateway as cm

        assert cm.get_owner_user_id("never-registered") is None

    def test_closure_pattern_propagates_source_user_id(self):
        """Reproduce the ``_clarify_callback_sync`` closure pattern: capture a
        ``source`` with ``user_id="42"``, call ``register(...)`` exactly as the
        gateway does, and confirm the entry in ``_entries`` carries the owner.
        """
        from types import SimpleNamespace
        from tools import clarify_gateway as cm

        # Mimic the SessionSource the gateway holds in closure scope.
        source = SimpleNamespace(user_id="42")

        # Mirror the propagation line from gateway/run.py::_clarify_callback_sync.
        cm.register(
            clarify_id="id-closure",
            session_key="sk-closure",
            question="Q?",
            choices=None,
            options=None,
            display_type=None,
            auth_policy=None,
            session_owner_user_id=str(source.user_id) if source and source.user_id else None,
        )

        entry = cm._entries.get("id-closure")
        assert entry is not None
        assert entry.session_owner_user_id == "42"

    def test_closure_pattern_none_when_source_user_id_missing(self):
        """When ``source.user_id`` is falsy the propagation expression yields
        None — no regression for anonymous-admin platforms (Telegram)."""
        from types import SimpleNamespace
        from tools import clarify_gateway as cm

        source = SimpleNamespace(user_id=None)
        cm.register(
            clarify_id="id-closure-none",
            session_key="sk-closure",
            question="Q?",
            choices=["A"],
            session_owner_user_id=str(source.user_id) if source and source.user_id else None,
        )
        entry = cm._entries.get("id-closure-none")
        assert entry is not None
        assert entry.session_owner_user_id is None

    def test_clarify_callback_sync_wires_session_owner(self):
        """Structural invariant: the ``register(...)`` call inside
        ``_clarify_callback_sync`` (gateway/run.py) passes the
        ``session_owner_user_id`` keyword.

        ``_clarify_callback_sync`` is a deeply-nested closure with many
        closure dependencies, so it cannot be driven in isolation cheaply.
        This AST check asserts the production wiring relationship directly —
        it fails if a refactor drops or renames the kwarg at the call site.
        """
        import ast
        import gateway.run as grun

        # Find the _clarify_callback_sync FunctionDef anywhere in the module.
        tree = ast.parse(_read_module_source(grun.__file__))
        sync_defs = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_clarify_callback_sync"
        ]
        assert sync_defs, "_clarify_callback_sync not found in gateway/run.py"
        sync_body = sync_defs[0]

        # Walk its body for a call to *.register(...) carrying our keyword.
        def _has_owner_kwarg(call: ast.Call) -> bool:
            if not isinstance(call.func, ast.Attribute):
                return False
            if call.func.attr != "register":
                return False
            return any(kw.arg == "session_owner_user_id" for kw in call.keywords)

        register_calls = [
            n for n in ast.walk(sync_body)
            if isinstance(n, ast.Call) and _has_owner_kwarg(n)
        ]
        assert register_calls, (
            "_clarify_callback_sync must pass session_owner_user_id= to its "
            "register(...) call so adapters can read the session initiator."
        )


# ===========================================================================
# Typed rich-option coercion (T4 — tools/clarify_gateway.py::_coerce_text_response)
# ===========================================================================

def _rich_entry(options=None):
    """A clarify entry registered with rich options and choices=None."""
    if options is None:
        options = [
            {"label": "Approve", "value": "yes"},
            {"label": "Reject", "value": "no"},
        ]
    return _ClarifyEntry(
        clarify_id="c-rich", session_key="sk-rich", question="Q?",
        choices=None, options=options,
    )


def _simple_entry(choices=None):
    """A clarify entry registered with simple string choices."""
    if choices is None:
        choices = ["red", "green", "blue"]
    return _ClarifyEntry(
        clarify_id="c-simple", session_key="sk-simple", question="Q?",
        choices=choices,
    )


def _open_entry():
    """An open-ended clarify entry: neither choices nor options."""
    return _ClarifyEntry(
        clarify_id="c-open", session_key="sk-open", question="Q?",
        choices=None, options=None,
    )


class TestCoerceRichOptions:
    """T4: typed text is mapped to an option's ``value`` when rich options are set.

    Covers the four resolution branches (index / label / value / custom) plus
    the out-of-range and non-matching reject paths. ``entry.choices`` is None
    and ``entry.options`` is set for every case here.
    """

    # -- index branch --------------------------------------------------------

    def test_index_in_range_first(self):
        assert _coerce_text_response(_rich_entry(), "1") == "yes"

    def test_index_in_range_second(self):
        assert _coerce_text_response(_rich_entry(), "2") == "no"

    def test_index_zero_is_out_of_range(self):
        # 1-based; "0" is below range and must NOT resolve positionally.
        assert _coerce_text_response(_rich_entry(), "0") == "0"

    def test_index_above_range_returns_unchanged(self):
        assert _coerce_text_response(_rich_entry(), "3") == "3"

    def test_negative_index_returns_unchanged(self):
        assert _coerce_text_response(_rich_entry(), "-1") == "-1"

    # -- label branch --------------------------------------------------------

    def test_label_match_returns_value(self):
        assert _coerce_text_response(_rich_entry(), "Approve") == "yes"

    def test_label_match_case_insensitive(self):
        assert _coerce_text_response(_rich_entry(), "REJECT") == "no"

    def test_label_match_with_whitespace(self):
        assert _coerce_text_response(_rich_entry(), "  approve  ") == "yes"

    def test_text_not_a_label_falls_through(self):
        # "maybe" matches neither label nor value — reject path.
        assert _coerce_text_response(_rich_entry(), "maybe") == "maybe"

    # -- value branch --------------------------------------------------------

    def test_value_match_returns_value(self):
        assert _coerce_text_response(_rich_entry(), "yes") == "yes"

    def test_value_match_case_insensitive(self):
        assert _coerce_text_response(_rich_entry(), "NO") == "no"

    def test_text_not_a_value_falls_through(self):
        # Distinct from labels/values — must survive as a custom answer.
        assert _coerce_text_response(_rich_entry(), "later") == "later"

    # -- custom / free-text branch ------------------------------------------

    def test_custom_text_preserved_unchanged(self):
        # Neither an index, label, nor value: free-text "Other" semantic.
        assert _coerce_text_response(_rich_entry(), "ask me tomorrow") == "ask me tomorrow"

    def test_numeric_looking_text_with_no_index_match_still_custom(self):
        # A number that is out of range is not forced into an option.
        assert _coerce_text_response(_rich_entry(), "42") == "42"

    def test_index_wins_over_label_match(self):
        # Precedence: when an option's label is itself a number ("1"), typing
        # "1" resolves positionally (index 0) rather than to that option by
        # label.  Locks the index > label > value ordering.
        opts = [
            {"label": "first", "value": "v-first"},
            {"label": "1", "value": "v-label-one"},
        ]
        assert _coerce_text_response(_rich_entry(opts), "1") == "v-first"


class TestCoerceSimplePathUnchanged:
    """T4: the simple-choices branch (``entry.choices`` set) stays byte-identical.

    Regression guard — the new options-aware code path must not perturb the
    existing choices behaviour.
    """

    def test_index_returns_choice_text(self):
        assert _coerce_text_response(_simple_entry(), "1") == "red"

    def test_index_second_choice(self):
        assert _coerce_text_response(_simple_entry(), "2") == "green"

    def test_index_zero_out_of_range(self):
        assert _coerce_text_response(_simple_entry(), "0") == "0"

    def test_index_above_range(self):
        assert _coerce_text_response(_simple_entry(), "9") == "9"

    def test_label_match_returns_canonical_choice(self):
        assert _coerce_text_response(_simple_entry(), "GREEN") == "green"

    def test_unknown_text_returns_raw(self):
        assert _coerce_text_response(_simple_entry(), "purple") == "purple"


class TestCoerceOpenEndedUnchanged:
    """T4: open-ended clarifies (no choices, no options) are passed through."""

    def test_open_ended_returns_trimmed_text(self):
        assert _coerce_text_response(_open_entry(), "  hello world  ") == "hello world"

    def test_open_ended_numeric_passed_through(self):
        assert _coerce_text_response(_open_entry(), "42") == "42"


# ===========================================================================
# Failure-path traceback retention (#11)
# ===========================================================================

class TestGatewayDispatchTraceback:
    """#11: unexpected gateway dispatch failures must log traceback context.

    ``_clarify_callback_sync`` is a deeply-nested closure whose runtime
    invocation requires the full gateway runner context (event loop,
    SessionSource, adapter instances, etc.).  Like the existing
    ``test_clarify_callback_sync_wires_session_owner`` structural test
    above, we assert the invariant directly via AST: the ``except`` handler
    that catches ``fut.result()`` failures must pass ``exc_info=True`` to
    ``logger.warning`` so operators can diagnose the failure from logs.

    The Discord and Telegram resolution paths are covered by runtime tests
    that trigger real exceptions and assert ``caplog`` records — see
    ``test_discord_clarify_buttons.py::TestClarifyFailureTraceback`` and
    ``test_discord_interactive_views.py::TestRich*Traceback``.
    """

    def test_dispatch_except_has_exc_info(self):
        import ast
        import gateway.run as grun

        tree = ast.parse(_read_module_source(grun.__file__))
        sync_defs = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_clarify_callback_sync"
        ]
        assert sync_defs, "_clarify_callback_sync not found in gateway/run.py"
        sync_body = sync_defs[0]

        # Find the except handler that contains a logger.warning("Clarify send failed")
        # call and verify it passes exc_info=True.
        found = False
        for node in ast.walk(sync_body):
            if not isinstance(node, ast.ExceptHandler):
                continue
            for child in ast.walk(node):
                if not isinstance(child, ast.Call):
                    continue
                if not isinstance(child.func, ast.Attribute):
                    continue
                if child.func.attr != "warning":
                    continue
                # Check the first positional arg matches "Clarify send failed"
                if not child.args:
                    continue
                fmt = child.args[0]
                if not (isinstance(fmt, ast.Constant) and
                        isinstance(fmt.value, str) and
                        "Clarify send failed" in fmt.value):
                    continue
                # Must have exc_info=True keyword
                has_exc_info = any(
                    kw.arg == "exc_info" for kw in child.keywords
                )
                assert has_exc_info, (
                    "logger.warning('Clarify send failed: ...') in "
                    "_clarify_callback_sync must pass exc_info=True so "
                    "operators can diagnose unexpected dispatch failures."
                )
                found = True

        assert found, (
            "Could not locate the 'Clarify send failed' logger.warning call "
            "inside _clarify_callback_sync — the dispatch error path may have "
            "been restructured."
        )

