"""Tests for the bundled tool-result-compaction plugin.

The plugin is inspired by OpenHuman-style token compression, but implemented
from scratch for Hermes' existing ``transform_tool_result`` hook.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PLUGIN_PATH = Path(__file__).resolve().parents[1] / "plugins" / "tool-result-compaction" / "__init__.py"


def _load_plugin_module():
    spec = importlib.util.spec_from_file_location("tool_result_compaction_plugin", PLUGIN_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compacts_large_terminal_json_while_preserving_failure_context():
    plugin = _load_plugin_module()
    noisy_output = "\n".join(
        ["collecting tests"]
        + [f"tests/test_many.py::{i} PASSED" for i in range(80)]
        + ["FAILED tests/test_critical.py::test_breakage - AssertionError: boom"]
        + ["Traceback (most recent call last):", "AssertionError: boom"]
    )
    raw = json.dumps({"output": noisy_output, "exit_code": 1})

    compacted = plugin.compact_tool_result(
        tool_name="terminal",
        result=raw,
        threshold_chars=350,
        head_lines=3,
        tail_lines=5,
        mode="compact",
    )

    assert compacted is not None
    assert len(compacted) < len(raw)
    payload = json.loads(compacted)
    assert payload["exit_code"] == 1
    assert payload["tool_result_compaction"]["original_chars"] == len(raw)
    assert payload["tool_result_compaction"]["mode"] == "compact"
    assert "FAILED tests/test_critical.py::test_breakage" in payload["output"]
    assert "AssertionError: boom" in payload["output"]
    assert "tests/test_many.py::79 PASSED" in payload["output"]


def test_small_or_non_terminal_results_passthrough():
    plugin = _load_plugin_module()
    raw = json.dumps({"output": "short", "exit_code": 0})

    assert plugin.compact_tool_result(
        tool_name="terminal",
        result=raw,
        threshold_chars=350,
        mode="compact",
    ) is None
    assert plugin.compact_tool_result(
        tool_name="web_extract",
        result=raw,
        threshold_chars=1,
        mode="compact",
    ) is None


def test_observe_mode_records_metadata_without_replacing_result():
    plugin = _load_plugin_module()
    raw = json.dumps({"output": "x" * 1000, "exit_code": 0})

    assert plugin.compact_tool_result(
        tool_name="terminal",
        result=raw,
        threshold_chars=100,
        mode="observe",
    ) is None


def test_compacts_large_single_line_terminal_output():
    plugin = _load_plugin_module()
    long_line = "prefix-" + ("x" * 5000) + "-ERROR-critical-failure-" + ("y" * 5000) + "-suffix"
    raw = json.dumps({"output": long_line, "exit_code": 1})

    compacted = plugin.compact_tool_result(
        tool_name="terminal",
        result=raw,
        threshold_chars=500,
        head_lines=3,
        tail_lines=5,
        mode="compact",
    )

    assert compacted is not None
    assert len(compacted) < len(raw)
    payload = json.loads(compacted)
    assert payload["exit_code"] == 1
    assert payload["tool_result_compaction"]["strategy"] == "single-line-head-diagnostics-tail"
    assert "prefix-" in payload["output"]
    assert "ERROR-critical-failure" in payload["output"]
    assert "-suffix" in payload["output"]
    assert "omitted" in payload["output"]


def test_register_wires_transform_tool_result_hook():
    plugin = _load_plugin_module()
    registered = {}

    class Ctx:
        def register_hook(self, name, fn):
            registered[name] = fn

    plugin.register(Ctx())

    assert "transform_tool_result" in registered
    assert callable(registered["transform_tool_result"])


def test_registered_hook_respects_compact_mode_settings(monkeypatch):
    plugin = _load_plugin_module()
    raw = json.dumps({"output": "\n".join(f"line {i}" for i in range(100)), "exit_code": 0})
    monkeypatch.setattr(
        plugin,
        "_load_settings",
        lambda: {
            "mode": "compact",
            "threshold_chars": 100,
            "head_lines": 2,
            "tail_lines": 2,
        },
    )

    compacted = plugin._transform_tool_result(tool_name="terminal", result=raw)

    assert compacted is not None
    payload = json.loads(compacted)
    assert payload["tool_result_compaction"]["mode"] == "compact"
    assert "line 0" in payload["output"]
    assert "line 99" in payload["output"]
