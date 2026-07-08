import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import hermes_cli.plugins as plugins_mod
import tools.terminal_tool as terminal_tool_module


_UNSET = object()


def _make_env_config(tmp_path, **overrides):
    config = {
        "env_type": "local",
        "timeout": 30,
        "cwd": str(tmp_path),
        "host_cwd": None,
        "modal_mode": "auto",
        "docker_image": "",
        "singularity_image": "",
        "modal_image": "",
        "daytona_image": "",
    }
    config.update(overrides)
    return config


def _run_terminal(
    monkeypatch,
    tmp_path,
    *,
    output,
    returncode=0,
    invoke_hook=_UNSET,
    approval=None,
    command="echo hello",
    session_id=_UNSET,
    tool_call_id=_UNSET,
):
    mock_env = MagicMock()
    mock_env.execute.return_value = {"output": output, "returncode": returncode}

    monkeypatch.setattr(
        terminal_tool_module, "_get_env_config", lambda: _make_env_config(tmp_path)
    )
    monkeypatch.setattr(terminal_tool_module, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        terminal_tool_module,
        "_check_all_guards",
        lambda *_args, **_kwargs: approval or {"approved": True},
    )
    monkeypatch.setitem(terminal_tool_module._active_environments, "default", mock_env)
    monkeypatch.setitem(terminal_tool_module._last_activity, "default", 0.0)

    if invoke_hook is not _UNSET:
        monkeypatch.setattr("hermes_cli.plugins.invoke_hook", invoke_hook)

    call_kwargs = {"command": command}
    if session_id is not _UNSET:
        call_kwargs["session_id"] = session_id
    if tool_call_id is not _UNSET:
        call_kwargs["tool_call_id"] = tool_call_id

    result = json.loads(terminal_tool_module.terminal_tool(**call_kwargs))
    return result, mock_env


def _capturing_hook(store, *, ret=None):
    """Return an invoke_hook stand-in that records the payload kwargs.

    ``ret`` is the hook result list (default ``[]`` — no output mutation).
    """

    def _hook(hook_name, **kwargs):
        store["hook_name"] = hook_name
        store["payload"] = kwargs
        return [] if ret is None else ret

    return _hook


def test_terminal_output_unchanged_when_transform_hook_not_registered(monkeypatch, tmp_path):
    result, _mock_env = _run_terminal(monkeypatch, tmp_path, output="plain output")

    assert result["output"] == "plain output"
    assert result["exit_code"] == 0
    assert result["error"] is None


def test_terminal_output_unchanged_for_none_hook_result(monkeypatch, tmp_path):
    result, _mock_env = _run_terminal(
        monkeypatch,
        tmp_path,
        output="plain output",
        invoke_hook=lambda hook_name, **kwargs: [None],
    )

    assert result["output"] == "plain output"


def test_terminal_output_ignores_invalid_hook_results(monkeypatch, tmp_path):
    result, _mock_env = _run_terminal(
        monkeypatch,
        tmp_path,
        output="plain output",
        invoke_hook=lambda hook_name, **kwargs: [{"bad": True}, 123, ["nope"]],
    )

    assert result["output"] == "plain output"


def test_terminal_output_uses_first_valid_string_from_hooks(monkeypatch, tmp_path):
    result, _mock_env = _run_terminal(
        monkeypatch,
        tmp_path,
        output="plain output",
        invoke_hook=lambda hook_name, **kwargs: [None, {"bad": True}, "first", "second"],
    )

    assert result["output"] == "first"


def test_terminal_output_transform_still_truncates_long_replacement(monkeypatch, tmp_path):
    transformed_output = "PLUGIN-HEAD\n" + ("A" * 60000) + "\nPLUGIN-TAIL"
    result, _mock_env = _run_terminal(
        monkeypatch,
        tmp_path,
        output="short output",
        invoke_hook=lambda hook_name, **kwargs: [transformed_output],
    )

    assert "PLUGIN-HEAD" in result["output"]
    assert "PLUGIN-TAIL" in result["output"]
    assert "[OUTPUT TRUNCATED" in result["output"]
    assert transformed_output != result["output"]


def test_terminal_output_transform_still_runs_strip_and_redact(monkeypatch, tmp_path):
    # Ensure redaction is active regardless of host HERMES_REDACT_SECRETS state
    # or collection-time import order (the module snapshots env at import).
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)

    secret = "sk-proj-abc123def456ghi789jkl012mno345"
    result, _mock_env = _run_terminal(
        monkeypatch,
        tmp_path,
        output="plain output",
        invoke_hook=lambda hook_name, **kwargs: [f" \x1b[31mOPENAI_API_KEY={secret}\x1b[0m "],
    )

    assert "\x1b" not in result["output"]
    # Terminal output now passes code_file=True: ENV-assignment redaction is
    # skipped (so code constants like MAX_TOKENS=100 aren't corrupted), but a
    # real sk-/ghp_/JWT-shaped value is STILL masked by _PREFIX_RE. The full
    # secret never survives; only the leading prefix marker remains. (#33801)
    assert secret not in result["output"]
    assert "OPENAI_API_KEY=" in result["output"]
    assert "sk-pro" in result["output"]  # prefix marker from _mask_token
    assert "abc123def456" not in result["output"]  # secret body is gone


def test_terminal_output_transform_hook_exception_falls_back(monkeypatch, tmp_path):
    def _raise(*_args, **_kwargs):
        raise RuntimeError("boom")

    result, _mock_env = _run_terminal(
        monkeypatch,
        tmp_path,
        output="plain output",
        invoke_hook=_raise,
    )

    assert result["output"] == "plain output"
    assert result["exit_code"] == 0
    assert result["error"] is None


def test_terminal_output_transform_does_not_change_approval_or_exit_code_meaning(monkeypatch, tmp_path):
    approval = {
        "approved": True,
        "user_approved": True,
        "description": "dangerous command",
    }
    result, _mock_env = _run_terminal(
        monkeypatch,
        tmp_path,
        output="original output",
        returncode=1,
        approval=approval,
        command="grep foo bar",
        invoke_hook=lambda hook_name, **kwargs: ["replaced output"],
    )

    assert result["output"] == "replaced output"
    assert result["approval"] == (
        "Command required approval (dangerous command) and was approved by the user."
    )
    assert result["exit_code_meaning"] == "No matches found (not an error)"


def test_terminal_output_transform_integration_with_real_plugin(monkeypatch, tmp_path):
    import yaml

    hermes_home = Path(os.environ["HERMES_HOME"])
    plugins_dir = hermes_home / "plugins"
    plugin_dir = plugins_dir / "terminal_transform"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text("name: terminal_transform\n", encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(
        "def register(ctx):\n"
        '    ctx.register_hook("transform_terminal_output", '
        'lambda **kw: "PLUGIN-HEAD\\n" + kw["output"] + "\\nPLUGIN-TAIL")\n',
        encoding="utf-8",
    )
    # Plugins are opt-in — must be listed in plugins.enabled to load.
    cfg_path = hermes_home / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump({"plugins": {"enabled": ["terminal_transform"]}}),
        encoding="utf-8",
    )

    # Force a fresh plugin manager so the new config is picked up.
    plugins_mod._plugin_manager = plugins_mod.PluginManager()
    plugins_mod.discover_plugins()

    long_output = "X" * 60000
    result, _mock_env = _run_terminal(
        monkeypatch,
        tmp_path,
        output=long_output,
    )

    assert "PLUGIN-HEAD" in result["output"]
    assert "PLUGIN-TAIL" in result["output"]
    assert "[OUTPUT TRUNCATED" in result["output"]


def test_transform_hook_payload_carries_session_and_tool_call_id(monkeypatch, tmp_path):
    """The hook payload includes session_id and tool_call_id so observers can
    attribute terminal output to a session and a specific tool call."""
    store = {}
    result, _mock_env = _run_terminal(
        monkeypatch,
        tmp_path,
        output="plain output",
        invoke_hook=_capturing_hook(store),
        session_id="sess-REAL-123",
        tool_call_id="call-REAL-abc",
    )

    payload = store["payload"]
    assert store["hook_name"] == "transform_terminal_output"
    # The two newly-threaded fields, populated with real values.
    assert payload["session_id"] == "sess-REAL-123"
    assert payload["tool_call_id"] == "call-REAL-abc"
    # Pre-existing payload fields are unchanged (purely additive).
    assert payload["command"] == "echo hello"
    assert payload["output"] == "plain output"
    assert payload["returncode"] == 0
    assert "task_id" in payload
    assert "env_type" in payload
    # No output-mutation behavior change: returning [] leaves output as-is.
    assert result["output"] == "plain output"


def test_transform_hook_session_and_tool_call_id_default_to_empty_string(monkeypatch, tmp_path):
    """When the tool is invoked without correlation IDs, the payload carries
    empty strings (never missing keys, never None) so consumers using
    ``kwargs["session_id"]`` don't KeyError."""
    store = {}
    _run_terminal(
        monkeypatch,
        tmp_path,
        output="plain output",
        invoke_hook=_capturing_hook(store),
    )

    payload = store["payload"]
    assert payload["session_id"] == ""
    assert payload["tool_call_id"] == ""


def test_transform_hook_ids_survive_full_dispatch_chain(monkeypatch, tmp_path):
    """End-to-end guard: the ids must survive the REAL dispatch wiring, not just
    a direct terminal_tool() call.

    Exercises handle_function_call -> registry.dispatch -> _handle_terminal ->
    terminal_tool -> transform_terminal_output, with the real registry (no mock),
    proving neither model_tools' dispatch forwarding nor the _handle_terminal
    adapter silently drops session_id/tool_call_id.
    """
    import model_tools

    mock_env = MagicMock()
    mock_env.execute.return_value = {"output": "plain output", "returncode": 0}
    monkeypatch.setattr(
        terminal_tool_module, "_get_env_config", lambda: _make_env_config(tmp_path)
    )
    monkeypatch.setattr(terminal_tool_module, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        terminal_tool_module,
        "_check_all_guards",
        lambda *_args, **_kwargs: {"approved": True},
    )
    monkeypatch.setitem(terminal_tool_module._active_environments, "default", mock_env)
    monkeypatch.setitem(terminal_tool_module._last_activity, "default", 0.0)

    # Several hooks fire on this path (transform_terminal_output, post_tool_call,
    # ...); capture only the one under test and no-op the rest.
    captured = {}

    def _capture(hook_name, **kwargs):
        if hook_name == "transform_terminal_output":
            captured.update(kwargs)
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _capture)

    model_tools.handle_function_call(
        "terminal",
        {"command": "echo hello"},
        task_id="task-XYZ",
        session_id="sess-REAL-123",
        tool_call_id="call-REAL-abc",
        skip_pre_tool_call_hook=True,
        skip_tool_request_middleware=True,
    )

    assert captured, "transform_terminal_output hook never fired via dispatch"
    assert captured["session_id"] == "sess-REAL-123"
    assert captured["tool_call_id"] == "call-REAL-abc"
