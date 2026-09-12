"""Plugin discovery and sequential final-output transform contracts."""

from pathlib import Path

import yaml

import hermes_cli.plugins as plugins_mod
from hermes_cli.plugins import PluginManager, VALID_HOOKS


def _make_enabled_plugin(hermes_home: Path, name: str, register_body: str) -> Path:
    """Create a plugin under <hermes_home>/plugins/<name> and opt it in."""
    plugin_dir = hermes_home / "plugins" / name
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        yaml.safe_dump({"name": name, "version": "0.1.0"}), encoding="utf-8",
    )
    (plugin_dir / "__init__.py").write_text(
        "def register(ctx):\n"
        f"    {register_body}\n",
        encoding="utf-8",
    )
    cfg_path = hermes_home / "config.yaml"
    cfg = {}
    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
    cfg.setdefault("plugins", {}).setdefault("enabled", []).append(name)
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return plugin_dir


def test_transform_llm_output_in_valid_hooks():
    assert "transform_llm_output" in VALID_HOOKS


def test_hook_receives_expected_kwargs(tmp_path, monkeypatch):
    """Hook callback should see response_text + session_id + model + platform."""
    hermes_home = tmp_path / "hermes_test"
    hermes_home.mkdir(exist_ok=True)
    _make_enabled_plugin(
        hermes_home, "capture_hook",
        register_body=(
            'ctx.register_hook("transform_llm_output", '
            'lambda **kw: f"{kw[\'response_text\']}|{kw[\'session_id\']}|'
            '{kw[\'model\']}|{kw[\'platform\']}")'
        ),
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    mgr = PluginManager()
    mgr.discover_and_load()

    results = mgr.invoke_hook(
        "transform_llm_output",
        response_text="hello world",
        session_id="s1",
        model="anthropic/claude-sonnet-4.6",
        platform="cli",
    )
    assert results == ["hello world|s1|anthropic/claude-sonnet-4.6|cli"]






def test_hook_exception_does_not_replace_response(tmp_path, monkeypatch):
    """A plugin raising an exception must not break hook dispatch.

    PluginManager.invoke_hook catches per-callback exceptions, logs a
    warning, and continues — so a raising plugin contributes no entry
    to the results list, and the walk in run_agent.py finds nothing to
    replace with.
    """
    hermes_home = tmp_path / "hermes_test"
    hermes_home.mkdir(exist_ok=True)
    _make_enabled_plugin(
        hermes_home, "raising_hook",
        register_body=(
            'def _boom(**kw):\n'
            '        raise RuntimeError("boom")\n'
            '    ctx.register_hook("transform_llm_output", _boom)'
        ),
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    mgr = PluginManager()
    mgr.discover_and_load()

    results = mgr.invoke_hook(
        "transform_llm_output",
        response_text="keep me",
        session_id="s1",
        model="m",
        platform="cli",
    )

    final_response = "keep me"
    for _hook_result in results:
        if isinstance(_hook_result, str) and _hook_result:
            final_response = _hook_result
            break

    assert final_response == "keep me"


def test_no_plugins_returns_empty_results(tmp_path, monkeypatch):
    """With no plugins loaded, invoke_hook returns [] and the response is unchanged."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_empty"))
    plugins_mod._plugin_manager = PluginManager()

    mgr = plugins_mod._plugin_manager
    results = mgr.invoke_hook(
        "transform_llm_output",
        response_text="unchanged",
        session_id="",
        model="m",
        platform="",
    )
    assert results == []


def test_transform_pipeline_composes_in_registration_order(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes_pipeline"
    hermes_home.mkdir()
    _make_enabled_plugin(
        hermes_home, "first",
        'ctx.register_hook("transform_llm_output", lambda **kw: kw["response_text"] + "-one")',
    )
    _make_enabled_plugin(
        hermes_home, "second",
        'ctx.register_hook("transform_llm_output", lambda **kw: kw["response_text"] + "-two")',
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    mgr = PluginManager()
    mgr.discover_and_load()
    text, transformed = mgr.transform_llm_output(
        "base", session_id="s1", model="m", platform="cli"
    )
    assert text == "base-one-two"
    assert transformed is True


def test_transform_pipeline_continues_after_failure(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes_failure"
    hermes_home.mkdir()
    _make_enabled_plugin(
        hermes_home, "raising",
        'def _boom(**kw):\n        raise RuntimeError("boom")\n    ctx.register_hook("transform_llm_output", _boom)',
    )
    _make_enabled_plugin(
        hermes_home, "later",
        'ctx.register_hook("transform_llm_output", lambda **kw: "safe")',
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    mgr = PluginManager()
    mgr.discover_and_load()
    text, transformed = mgr.transform_llm_output(
        "unsafe", session_id="s1", model="m", platform="cli"
    )
    assert text == "safe"
    assert transformed is True


def test_registered_transform_disables_gateway_raw_delivery(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from gateway.run_turn_runner import TurnRunner
    from gateway.config import StreamingConfig

    _make_enabled_plugin(tmp_path, "guard",
        'ctx.register_hook("transform_llm_output", lambda **kw: "safe")')
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    manager = PluginManager()
    manager.discover_and_load()
    monkeypatch.setattr(plugins_mod, "_plugin_manager", manager)
    audio = Mock()
    status = Mock()
    runner = object.__new__(TurnRunner)
    runner._ctx = SimpleNamespace(streaming_tts_consumer_holder=[audio],
        interim_assistant_messages_enabled=True, _run_still_current=lambda: True,
        _status_adapter=status, resolve_display_setting=lambda *_args: None,
        user_config={}, source=SimpleNamespace())
    runner._runner = SimpleNamespace(config=SimpleNamespace(streaming=StreamingConfig(enabled=True)),
        _adapter_for_source=lambda _source: None)
    consumer, delta, commentary, enabled = runner._setup_stream_consumer("telegram")
    assert consumer is None and delta is None and enabled is False
    commentary("secret", already_streamed=False)
    commentary("secret", already_streamed=True)
    assert not audio.mock_calls and not status.mock_calls
