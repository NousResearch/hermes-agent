"""Regression tests for ``tools.tool_search_validation`` ``{"item": x}`` envelope repair.

Frontend LLMs (Anthropic / Codex / OpenAI Responses-style tool-call parsing)
occasionally emit ``{"item": <scalar>}`` where the tool's JSON Schema declares a bare
scalar or array element — that's the JSON shape of a tool-result item, not a
tool-call argument. The validator used to reject the call outright; it now
recursively repairs unambiguous envelopes so the call matches the registered schema.

These tests pin the contract; delete or amend them if the policy changes.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

import pytest


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _register_schema(name: str, toolset: str, params: Dict[str, Any], calls: List[Dict[str, Any]]) -> None:
    from tools.registry import registry

    def _handler(args, task_id=None, **kw):
        calls.append(args)
        return json.dumps({"ok": True, "args": args})

    registry.register(
        name=name,
        handler=_handler,
        schema={"name": name, "description": f"desc {name}", "parameters": params},
        toolset=toolset,
    )


class TestRepairItemEnvelopes:
    """Direct unit tests for the repair helpers — no plugin/registry needed."""

    def test_array_position_unwraps_single_envelope(self):
        from tools.tool_search_validation import _repair_item_envelopes

        args = {"urls": [{"item": "https://x.example/"}]}
        out, changed = _repair_item_envelopes(args)
        assert changed is True
        assert out == {"urls": ["https://x.example/"]}

    def test_array_multi_envelope_unwraps(self):
        from tools.tool_search_validation import _repair_item_envelopes

        args = {"urls": [{"item": "https://a/"}, {"item": "https://b/"}]}
        out, _ = _repair_item_envelopes(args)
        assert out == {"urls": ["https://a/", "https://b/"]}

    def test_scalar_position_unwraps_single_envelope(self):
        from tools.tool_search_validation import _repair_item_envelopes

        args = {"url": {"item": "https://only.example/"}}
        out, changed = _repair_item_envelopes(args)
        assert changed is True
        assert out == {"url": "https://only.example/"}

    def test_multi_key_dict_not_unwrapped(self):
        """Only *single-key* dicts matching ``{"item": x}`` are repaired."""
        from tools.tool_search_validation import _repair_item_envelopes

        args = {"results": {"item": "data", "extra": "ignored"}}
        out, changed = _repair_item_envelopes(args)
        assert changed is False
        assert out == args

    def test_well_formed_args_unchanged(self):
        from tools.tool_search_validation import _repair_item_envelopes

        args = {"format": "markdown", "urls": ["https://x.com"]}
        out, changed = _repair_item_envelopes(args)
        assert changed is False
        assert out == args

    def test_nested_envelope_repaired(self):
        from tools.tool_search_validation import _repair_item_envelopes

        args = {"outer": {"inner": {"item": "https://x/"}, "k": "v"}}
        out, changed = _repair_item_envelopes(args)
        assert changed is True
        assert out == {"outer": {"inner": "https://x/", "k": "v"}}

    def test_idempotent(self):
        from tools.tool_search_validation import _repair_item_envelopes

        once, _ = _repair_item_envelopes({"urls": [{"item": "https://x/"}]})
        twice, changed2 = _repair_item_envelopes(once)
        assert once == {"urls": ["https://x/"]}
        assert changed2 is False

    def test_cycle_safe(self):
        from tools.tool_search_validation import _repair_item_envelopes

        node: Dict[str, Any] = {"k": "v"}
        node["self"] = node  # direct self-cycle
        # Must not infinite-loop — bounded by ``id()`` set.
        out, _ = _repair_item_envelopes({"x": node})
        assert out["x"] is node or out["x"]["self"] is node


class TestValidatorRepairsEnvelopes:
    """End-to-end through the registry's schema validator."""

    def _tool_set(self, name: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        calls: List[Dict[str, Any]] = []
        _register_schema(name, "envelope-repair-probe", params, calls)
        return calls

    def test_array_form_with_envelope_passes_validation(self):
        from tools.tool_search import validate_deferred_call_args

        calls = self._tool_set(
            "envprobe_array",
            {
                "type": "object",
                "properties": {
                    "urls": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["urls"],
            },
        )

        # Broken LLM output — single-item envelope per element. Pre-fix, this
        # raised 'arguments.urls[0] (type): {"item": "..."} is not of type "string"'.
        bad_args = {"urls": [{"item": "https://docs.example/"}]}
        err = validate_deferred_call_args("envprobe_array", bad_args)
        assert err is None, err

    def test_scalar_form_with_envelope_passes_validation(self):
        from tools.tool_search import validate_deferred_call_args

        self._tool_set(
            "envprobe_scalar",
            {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        )

        bad_args = {"url": {"item": "https://example/"}}
        err = validate_deferred_call_args("envprobe_scalar", bad_args)
        assert err is None, err

    def test_well_formed_args_still_pass(self):
        from tools.tool_search import validate_deferred_call_args

        self._tool_set(
            "envprobe_ok",
            {
                "type": "object",
                "properties": {
                    "urls": {"type": "array", "items": {"type": "string"}},
                    "format": {"type": "string"},
                },
                "required": ["urls"],
            },
        )

        err = validate_deferred_call_args(
            "envprobe_ok",
            {"urls": ["https://a/", "https://b/"], "format": "markdown"},
        )
        assert err is None

    def test_other_shapes_still_rejected(self):
        """Repair is narrow on purpose — non-envelope broken shapes still fail."""
        from tools.tool_search import validate_deferred_call_args

        self._tool_set(
            "envprobe_strict",
            {
                "type": "object",
                "properties": {
                    "urls": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["urls"],
            },
        )

        # Plain integer where an array is required — neither matches schema nor
        # is an envelope, must still fail.
        err = validate_deferred_call_args("envprobe_strict", {"urls": 42})
        assert err is not None
        assert "is not of type" in err or "valid" in err.lower()
