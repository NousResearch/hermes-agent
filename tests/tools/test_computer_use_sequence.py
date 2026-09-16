"""Tests for the computer_use `sequence` primitive (RFC #112639, reordered P0).

`sequence` is orchestration over the existing per-action path: every step re-runs the hard blocks,
per-step approval scopes, and the same backend handlers. These tests assert that contract plus the
V1 validation rules and the abort semantics.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict
from unittest.mock import patch

import pytest

from tools.computer_use.backend import ActionResult


@pytest.fixture(autouse=True)
def _reset_backend(grant_computer_use_approvals):
    from tools.computer_use.tool import reset_backend_for_tests
    reset_backend_for_tests()
    with patch.dict(os.environ, {"HERMES_COMPUTER_USE_BACKEND": "noop"}, clear=False):
        yield
    reset_backend_for_tests()


@pytest.fixture
def noop_backend():
    from tools.computer_use.tool import _get_backend
    return _get_backend()


def _run(args: Dict[str, Any], **kw: Any) -> Any:
    from tools.computer_use.tool import handle_computer_use
    out = handle_computer_use(args, **kw)
    return json.loads(out) if isinstance(out, str) else out


def _seq(*steps: Dict[str, Any], **kw: Any) -> Dict[str, Any]:
    return {"action": "sequence", "steps": list(steps), **kw}


class TestSequenceSchema:
    def test_sequence_in_action_enum(self):
        from tools.computer_use.schema import COMPUTER_USE_SCHEMA
        props = COMPUTER_USE_SCHEMA["parameters"]["properties"]
        assert "sequence" in props["action"]["enum"]
        assert props["steps"]["type"] == "array"
        assert props["verify_mode"]["enum"] == ["ax_first", "som", "vision"]


class TestSequenceValidation:
    def test_empty_steps_rejected(self):
        assert "error" in _run(_seq())

    def test_nested_sequence_rejected(self):
        out = _run(_seq({"action": "sequence", "steps": [{"action": "wait"}]}))
        assert "error" in out

    def test_capture_action_rejected_in_steps(self):
        out = _run(_seq({"action": "capture", "mode": "ax"}))
        assert "error" in out

    def test_two_grounded_steps_rejected(self):
        out = _run(_seq({"action": "click", "element": 3},
                        {"action": "click", "coordinate": [10, 20]}))
        assert "error" in out and "one snapshot-grounded" in out["error"]

    def test_per_step_capture_after_rejected(self):
        out = _run(_seq({"action": "click", "element": 3, "capture_after": True}))
        assert "error" in out

    def test_bad_verify_mode_rejected(self):
        out = _run(_seq({"action": "wait", "seconds": 0.01}, verify_mode="infrared"))
        assert "error" in out

    def test_step_count_capped(self):
        out = _run(_seq(*[{"action": "wait", "seconds": 0.01}] * 11))
        assert "error" in out


class TestSequenceExecution:
    def test_happy_path_runs_steps_in_order(self, noop_backend):
        out = _run(_seq({"action": "click", "element": 3},
                        {"action": "type", "text": "hello"},
                        {"action": "key", "keys": "return"}))
        assert out["ok"] is True
        assert [s["action"] for s in out["steps"]] == ["click", "type", "key"]
        assert all(s["ok"] for s in out["steps"])
        assert [c[0] for c in noop_backend.calls] == ["click", "type", "key"]
        m = out["sequence_metrics"]
        assert (m["actions_executed"], m["slice_length"], m["computer_use_calls"]) == (3, 3, 1)
        assert m["success"] is True and m["slice_abort_step"] is None
        assert m["task_wall_ms"] > 0 and m["tool_ms"] > 0
        assert all(s["ms"] >= 0 for s in out["steps"])

    def test_each_destructive_step_gets_own_approval(self, noop_backend):
        from tools.computer_use import tool as cu_tool
        seen = []
        cu_tool.set_approval_callback(lambda command, description, **kw: seen.append(command) or "once")
        try:
            out = _run(_seq({"action": "click", "element": 3}, {"action": "type", "text": "hi"}))
        finally:
            cu_tool.set_approval_callback(lambda command, description, **kw: "once")
        assert out["ok"] is True
        # The gate was consulted once per destructive step (the callback sees the display target).
        assert seen == ["computer_use: click element #3", "computer_use: type 'hi'"]

    def test_hard_block_applies_to_steps(self, noop_backend):
        out = _run(_seq({"action": "type", "text": "sudo rm -rf /tmp/x"},
                        {"action": "key", "keys": "return"}))
        assert out["ok"] is False and out["code"] == "sequence_aborted"
        assert out["sequence_metrics"]["slice_abort_step"] == 0
        assert out["abort_reason"] == "rejected"
        assert [c[0] for c in noop_backend.calls] == []

    def test_hard_block_fires_mid_slice(self, noop_backend):
        out = _run(_seq({"action": "click", "element": 3},
                        {"action": "type", "text": "sudo rm -rf /tmp/x"},
                        {"action": "key", "keys": "return"}))
        assert out["ok"] is False and out["code"] == "sequence_aborted"
        assert out["abort_reason"] == "rejected"
        assert out["sequence_metrics"]["slice_abort_step"] == 1
        assert out["sequence_metrics"]["actions_executed"] == 1
        assert [c[0] for c in noop_backend.calls] == ["click"]

    def test_step_failure_aborts_without_running_later_steps(self, noop_backend, monkeypatch):
        def boom(self, *pos, **kw):
            self.calls.append(("type", kw))
            return ActionResult(ok=False, action="type", message="backend exploded")
        monkeypatch.setattr(type(noop_backend), "type_text", boom)
        out = _run(_seq({"action": "click", "element": 3},
                        {"action": "type", "text": "hi"},
                        {"action": "key", "keys": "return"}))
        assert out["ok"] is False and out["code"] == "sequence_aborted"
        assert out["abort_reason"] == "action_failed"
        assert out["sequence_metrics"]["slice_abort_step"] == 1
        assert [c[0] for c in noop_backend.calls] == ["click", "type"]

    def test_suspected_noop_aborts_slice(self, noop_backend, monkeypatch):
        def noop_click(self, *pos, **kw):
            self.calls.append(("click", kw))
            return ActionResult(ok=True, action="click", effect="suspected_noop")
        monkeypatch.setattr(type(noop_backend), "click", noop_click)
        out = _run(_seq({"action": "click", "element": 3},
                        {"action": "type", "text": "hi"}))
        assert out["ok"] is False and out["abort_reason"] == "noop_verdict"
        assert [c[0] for c in noop_backend.calls] == ["click"]

    def test_approval_denial_aborts_slice(self, noop_backend):
        from tools.computer_use import tool as cu_tool
        cu_tool.set_approval_callback(lambda command, description, **kw: "deny")
        try:
            out = _run(_seq({"action": "click", "element": 3},
                            {"action": "type", "text": "hi"}))
        finally:
            cu_tool.set_approval_callback(lambda command, description, **kw: "once")
        assert out["ok"] is False and out["abort_reason"] == "approval_denied"
        assert out["sequence_metrics"]["slice_abort_step"] == 0
        assert noop_backend.calls == []


class TestSequenceFinalVerify:
    # The noop backend returns no pixels, so the final capture degrades to the text path: the sequence
    # metrics merge into the top-level payload (no multimodal envelope, no "sequence_result" key).
    def test_ax_first_falls_back_when_ax_tree_empty(self, noop_backend):
        out = _run(_seq({"action": "click", "element": 3}, capture_after=True, verify_mode="ax_first"))
        assert out["ok"] is True
        m = out["sequence_metrics"]
        assert m["verify_mode_requested"] == "ax_first"
        assert m["verify_mode_used"] == "som" and m["verify_fallback"] is True
        assert m["captures"] == 2 and m["image_captures"] == 0  # noop backend has no pixels
        assert m["verification_ms"] >= m["capture_ms"] >= 0  # verification wraps the capture(s)
        captures = [c for c in noop_backend.calls if c[0] == "capture"]
        assert [c[1]["mode"] for c in captures] == ["ax", "som"]

    def test_forced_som_skips_ax(self, noop_backend):
        out = _run(_seq({"action": "wait", "seconds": 0.01}, capture_after=True, verify_mode="som"))
        assert out["ok"] is True
        m = out["sequence_metrics"]
        assert m["verify_mode_used"] == "som" and m["verify_fallback"] is False
        assert m["captures"] == 1

    def test_multimodal_final_capture_carries_sequence_result(self, noop_backend, monkeypatch):
        from tools.computer_use import tool as cu_tool
        from tools.computer_use.backend import CaptureResult
        png10 = ("iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAIAAAACUFjqAAAAFElEQVR4nGM8ISfHgBsw4ZEbwdIARtsBGOVUa2wAAAAASUVORK5CYII=")
        def shot(self, *pos, **kw):
            self.calls.append(("capture", kw))
            return CaptureResult(mode=kw.get("mode") or "som", width=10, height=10,
                                 png_b64=png10, elements=[], app="", window_title="")
        monkeypatch.setattr(type(noop_backend), "capture", shot)
        monkeypatch.setattr(cu_tool, "_should_route_through_aux_vision", lambda: False)
        out = _run(_seq({"action": "wait", "seconds": 0.01}, capture_after=True, verify_mode="som"))
        assert out.get("_multimodal") is True
        assert out["sequence_result"]["sequence_metrics"]["image_captures"] == 1
        assert out["sequence_result"]["ok"] is True

    def test_no_capture_after_means_no_capture(self, noop_backend):
        out = _run(_seq({"action": "wait", "seconds": 0.01}))
        assert out["ok"] is True and "sequence_result" not in out
        assert out["sequence_metrics"]["captures"] == 0
        assert [c for c in noop_backend.calls if c[0] == "capture"] == []

    def test_abort_skips_final_capture(self, noop_backend):
        out = _run(_seq({"action": "type", "text": "sudo rm -rf /tmp/x"}, capture_after=True))
        assert out["ok"] is False
        assert [c for c in noop_backend.calls if c[0] == "capture"] == []
