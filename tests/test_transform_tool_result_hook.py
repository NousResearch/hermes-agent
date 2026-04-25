"""Tests for the ``transform_tool_result`` plugin hook wired into
``model_tools.handle_function_call``.

Mirrors the ``transform_terminal_output`` hook tests from Phase 1 but
targets the generic tool-result seam that runs for every tool dispatch.
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import hermes_cli.plugins as plugins_mod
import model_tools


_UNSET = object()


def _run_handle_function_call(
    monkeypatch,
    *,
    tool_name="dummy_tool",
    tool_args=None,
    dispatch_result='{"output": "original"}',
    invoke_hook=_UNSET,
):
    """Drive ``handle_function_call`` with a mocked registry dispatch."""
    from tools.registry import registry

    monkeypatch.setattr(
        registry, "dispatch",
        lambda name, args, **kw: dispatch_result,
    )
    # Skip unrelated side effects (read-loop tracker).
    monkeypatch.setattr(model_tools, "_READ_SEARCH_TOOLS", frozenset())

    if invoke_hook is not _UNSET:
        # Patch the symbol actually imported inside handle_function_call.
        monkeypatch.setattr("hermes_cli.plugins.invoke_hook", invoke_hook)

    return model_tools.handle_function_call(
        tool_name,
        tool_args or {},
        task_id="t1",
        session_id="s1",
        tool_call_id="tc1",
        skip_pre_tool_call_hook=True,
    )


def test_result_unchanged_when_no_hook_registered(monkeypatch):
    # Real invoke_hook with no plugins loaded returns [].
    monkeypatch.setenv("HERMES_HOME", "/tmp/hermes_no_plugins")
    # Force a fresh plugin manager so no stale plugins pollute state.
    plugins_mod._plugin_manager = plugins_mod.PluginManager()

    out = _run_handle_function_call(monkeypatch)
    assert out == '{"output": "original"}'


def test_result_unchanged_for_none_hook_return(monkeypatch):
    out = _run_handle_function_call(
        monkeypatch,
        invoke_hook=lambda hook_name, **kw: [None],
    )
    assert out == '{"output": "original"}'


def test_result_ignores_non_string_hook_returns(monkeypatch):
    out = _run_handle_function_call(
        monkeypatch,
        invoke_hook=lambda hook_name, **kw: [{"bad": True}, 123, ["nope"]],
    )
    assert out == '{"output": "original"}'


def test_first_valid_string_return_replaces_result(monkeypatch):
    out = _run_handle_function_call(
        monkeypatch,
        invoke_hook=lambda hook_name, **kw: [None, {"x": 1}, "first", "second"],
    )
    assert out == "first"


def test_hook_receives_expected_kwargs(monkeypatch):
    captured = {}

    def _hook(hook_name, **kwargs):
        if hook_name == "transform_tool_result":
            captured.update(kwargs)
        return []

    out = _run_handle_function_call(
        monkeypatch,
        tool_name="my_tool",
        tool_args={"a": 1, "b": "x"},
        dispatch_result='{"ok": true}',
        invoke_hook=_hook,
    )
    assert out == '{"ok": true}'
    assert captured["tool_name"] == "my_tool"
    assert captured["args"] == {"a": 1, "b": "x"}
    assert captured["result"] == '{"ok": true}'
    assert captured["task_id"] == "t1"
    assert captured["session_id"] == "s1"
    assert captured["tool_call_id"] == "tc1"


def test_hook_exception_falls_back_to_original(monkeypatch):
    def _raise(*_a, **_kw):
        raise RuntimeError("boom")

    out = _run_handle_function_call(
        monkeypatch,
        invoke_hook=_raise,
    )
    assert out == '{"output": "original"}'


def test_post_tool_call_remains_observational(monkeypatch):
    """post_tool_call return values must NOT replace the result."""
    def _hook(hook_name, **kw):
        if hook_name == "post_tool_call":
            # Observers returning a string must be ignored.
            return ["observer return should be ignored"]
        return []

    out = _run_handle_function_call(
        monkeypatch,
        invoke_hook=_hook,
    )
    assert out == '{"output": "original"}'


def test_transform_tool_result_runs_after_post_tool_call(monkeypatch):
    """post_tool_call sees ORIGINAL result; transform_tool_result sees same and may replace."""
    observed = []

    def _hook(hook_name, **kw):
        if hook_name == "post_tool_call":
            observed.append(("post_tool_call", kw["result"]))
            return []
        if hook_name == "transform_tool_result":
            observed.append(("transform_tool_result", kw["result"]))
            return ["rewritten"]
        return []

    out = _run_handle_function_call(
        monkeypatch,
        dispatch_result='{"raw": "value"}',
        invoke_hook=_hook,
    )
    assert out == "rewritten"
    # Both hooks saw the ORIGINAL (untransformed) result.
    assert observed == [
        ("post_tool_call", '{"raw": "value"}'),
        ("transform_tool_result", '{"raw": "value"}'),
    ]


def test_transform_tool_result_integration_with_real_plugin(monkeypatch, tmp_path):
    """End-to-end: load a real plugin from HERMES_HOME and verify it rewrites results."""
    import yaml

    hermes_home = Path(os.environ["HERMES_HOME"])
    plugins_dir = hermes_home / "plugins"
    plugin_dir = plugins_dir / "transform_result_canon"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text("name: transform_result_canon\n", encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(
        "def register(ctx):\n"
        '    ctx.register_hook("transform_tool_result", '
        'lambda **kw: f\'CANON[{kw["tool_name"]}]\' + kw["result"])\n',
        encoding="utf-8",
    )
    # Plugins are opt-in — must be listed in plugins.enabled to load.
    cfg_path = hermes_home / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump({"plugins": {"enabled": ["transform_result_canon"]}}),
        encoding="utf-8",
    )

    # Force a fresh plugin manager so the new config is picked up.
    plugins_mod._plugin_manager = plugins_mod.PluginManager()
    plugins_mod.discover_plugins()

    out = _run_handle_function_call(
        monkeypatch,
        tool_name="some_tool",
        dispatch_result='{"payload": 42}',
    )
    assert out == 'CANON[some_tool]{"payload": 42}'
