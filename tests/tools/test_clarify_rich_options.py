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
