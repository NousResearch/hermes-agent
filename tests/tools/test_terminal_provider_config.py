from copy import deepcopy
import logging

import pytest

from agent.terminal_env_provider import TerminalEnvironmentProvider


def test_plugin_factory_receives_resolved_backend_config(monkeypatch):
    import hermes_cli.config as config_module
    import tools.terminal_tool_backends as terminal_tool_backends

    received = {}

    class Provider(TerminalEnvironmentProvider):
        name = "configured_box"

        def is_available(self):
            return True

        def resolve_config(self, config):
            resolved = deepcopy(config)
            resolved.setdefault("region", "us-east")
            return resolved

        def create_environment(self, **kwargs):
            received.update(kwargs)
            return object()

    raw_config = {
        "terminal": {
            "backends": {
                "configured_box": {
                    "workspace": "demo",
                }
            }
        }
    }
    monkeypatch.setattr(
        config_module,
        "read_user_config_raw",
        lambda **_kwargs: raw_config,
    )
    monkeypatch.setattr(
        terminal_tool_backends,
        "_get_plugin_env_provider",
        lambda name: Provider() if name == "configured_box" else None,
    )

    terminal_tool_backends._create_environment(
        "configured_box",
        "",
        "~",
        60,
    )

    assert received["backend_config"] == {
        "workspace": "demo",
        "region": "us-east",
    }
    received["backend_config"]["workspace"] = "changed"
    assert raw_config["terminal"]["backends"]["configured_box"]["workspace"] == "demo"


def test_plugin_factory_fails_when_profile_config_cannot_be_read(monkeypatch):
    import hermes_cli.config as config_module
    import tools.terminal_tool_backends as terminal_tool_backends

    class Provider(TerminalEnvironmentProvider):
        name = "configured_box"

        def is_available(self):
            return True

        def resolve_config(self, config):
            return {**config, "fallback": "must-not-run"}

        def create_environment(self, **kwargs):
            return object()

    monkeypatch.setattr(
        config_module,
        "read_user_config_raw",
        lambda **_kwargs: (_ for _ in ()).throw(PermissionError("config unreadable")),
    )
    monkeypatch.setattr(
        terminal_tool_backends,
        "_get_plugin_env_provider",
        lambda name: Provider() if name == "configured_box" else None,
    )

    with pytest.raises(terminal_tool_backends.PluginTerminalEnvironmentError) as exc_info:
        terminal_tool_backends._create_environment("configured_box", "", "~", 60)
    assert "config unreadable" not in str(exc_info.value)


@pytest.mark.parametrize("yaml_text", ["- ignored-root-value\n", "null\n", "false\n", "0\n", "[]\n"])
def test_plugin_factory_treats_non_mapping_yaml_root_as_empty_config(
    tmp_path, monkeypatch, yaml_text
):
    import hermes_cli.config as config_module
    import tools.terminal_tool_backends as terminal_tool_backends

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml_text, encoding="utf-8")
    received = None

    class Provider(TerminalEnvironmentProvider):
        name = "configured_box"

        def is_available(self):
            return True

        def resolve_config(self, config):
            nonlocal received
            received = config
            return config

        def create_environment(self, **kwargs):
            return object()

    monkeypatch.setattr(config_module, "get_config_path", lambda: config_path)
    monkeypatch.setattr(
        terminal_tool_backends,
        "_get_plugin_env_provider",
        lambda name: Provider() if name == "configured_box" else None,
    )

    terminal_tool_backends._create_environment("configured_box", "", "~", 60)

    assert received == {}


def test_requirements_failure_does_not_log_provider_exception_value(monkeypatch, caplog):
    import hermes_cli.config as config_module
    import tools.terminal_tool as terminal_tool
    import tools.terminal_tool_backends as terminal_tool_backends

    class Provider(TerminalEnvironmentProvider):
        name = "configured_box"

        def is_available(self):
            return True

        def create_environment(self, **kwargs):
            return object()

    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: {"env_type": "configured_box"})
    monkeypatch.setattr(
        terminal_tool_backends,
        "_get_plugin_env_provider",
        lambda name: Provider() if name == "configured_box" else None,
    )
    monkeypatch.setattr(
        config_module,
        "read_user_config_raw",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("TOP-SECRET")),
    )

    with caplog.at_level(logging.ERROR):
        assert terminal_tool.check_terminal_requirements() is False
    assert "TOP-SECRET" not in caplog.text


def test_runtime_resolver_failure_does_not_leak_provider_exception_value(monkeypatch, caplog):
    import hermes_cli.config as config_module
    import tools.terminal_tool as terminal_tool
    import tools.terminal_tool_backends as terminal_tool_backends

    class Provider(TerminalEnvironmentProvider):
        name = "configured_box"

        def is_available(self):
            return True

        def resolve_config(self, config):
            raise RuntimeError("TOP-SECRET")

        def create_environment(self, **kwargs):
            return object()

    monkeypatch.setenv("TERMINAL_ENV", "configured_box")
    monkeypatch.setattr(
        terminal_tool_backends,
        "_get_plugin_env_provider",
        lambda name: Provider() if name == "configured_box" else None,
    )
    monkeypatch.setattr(config_module, "read_user_config_raw", lambda **_kwargs: {})

    with caplog.at_level(logging.WARNING):
        assert terminal_tool.ensure_task_env("secret-test") is None
        response = terminal_tool.terminal_tool("pwd", task_id="secret-test")

    assert "TOP-SECRET" not in response
    assert "TOP-SECRET" not in caplog.text
    assert "Plugin terminal backend could not be initialized." in response
