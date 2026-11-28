"""Config-plane outages preserve the uninstrumented native dispatch contract."""
import json

import pytest


def test_observer_config_outage_preserves_config_independent_handler(monkeypatch, caplog):
    import model_tools
    from hermes_cli import config
    from hermes_cli.config_backend import ConfigBackendUnavailable

    calls = []
    def unavailable():
        raise ConfigBackendUnavailable("private backend detail")
    def handler(args, **kwargs):
        calls.append(args)
        return json.dumps({"success": True, "value": "original"})
    monkeypatch.setattr(config, "load_config_readonly", unavailable)
    monkeypatch.setattr(model_tools.registry._tools["skill_manage"], "handler", handler)
    result = model_tools.handle_function_call("skill_manage", {"operations": []})
    assert json.loads(result) == {"success": True, "value": "original"}
    assert len(calls) == 1
    assert "private backend detail" not in caplog.text


def test_handler_config_outage_still_fails_closed(monkeypatch):
    import model_tools
    from hermes_cli import config
    from hermes_cli.config_backend import ConfigBackendUnavailable

    calls = []
    def unavailable():
        raise ConfigBackendUnavailable("backend down")
    def handler(args, **kwargs):
        calls.append(args)
        config.load_config_readonly()
        return json.dumps({"success": True})
    monkeypatch.setattr(config, "load_config_readonly", unavailable)
    monkeypatch.setattr(model_tools.registry._tools["skill_manage"], "handler", handler)
    with pytest.raises(ConfigBackendUnavailable, match="backend down"):
        model_tools.handle_function_call("skill_manage", {"operations": []})
    assert len(calls) == 1


def test_real_skill_handler_config_outage_does_not_write(monkeypatch):
    import model_tools
    from hermes_cli import config
    from hermes_cli.config_backend import ConfigBackendUnavailable

    entry = model_tools.registry._tools["skill_manage"]
    original = entry.handler
    calls = []
    def actual_handler(args, **kwargs):
        calls.append(args)
        return original(args, **kwargs)
    def unavailable():
        raise ConfigBackendUnavailable("backend down")
    monkeypatch.setattr(entry, "handler", actual_handler)
    monkeypatch.setattr(config, "load_config_readonly", unavailable)
    args = {"operations": [{"action": "create", "name": "outage-test",
                            "content": "---\nname: outage-test\ndescription: Use when testing.\n---\n"}]}
    with pytest.raises(ConfigBackendUnavailable, match="backend down"):
        model_tools.handle_function_call("skill_manage", args)
    assert len(calls) == 1
    from hermes_constants import get_hermes_home
    assert not (get_hermes_home() / "skills" / "outage-test").exists()


def test_ledger_observer_config_outage_preserves_handler(monkeypatch):
    import model_tools
    from hermes_cli import config
    from hermes_cli.config_backend import ConfigBackendUnavailable
    from tools import skill_native_audit

    calls = []
    def unavailable():
        raise ConfigBackendUnavailable("backend down")
    def handler(args, **kwargs):
        calls.append(args)
        return '{"success": true}'
    monkeypatch.setattr(skill_native_audit, "_enabled", lambda: True)
    monkeypatch.setattr(config, "load_config_readonly", unavailable)
    monkeypatch.setattr(model_tools.registry._tools["skill_manage"], "handler", handler)
    assert json.loads(model_tools.handle_function_call("skill_manage", {"operations": []}))["success"]
    assert len(calls) == 1
