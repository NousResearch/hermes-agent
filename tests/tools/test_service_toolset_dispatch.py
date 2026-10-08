"""Service policy is checked at registered effects, not only schema assembly."""

import json

import pytest

from tools.registry import ToolRegistry


@pytest.fixture
def dispatch_policy(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_ALLOWED_TOOLSETS", raising=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}")
    effects = []
    registry = ToolRegistry()
    for name, toolset in (
        ("terminal", "terminal"),
        ("plugin_effect", "dispatch-test-plugin"),
        ("forbidden_effect", "dispatch-test-forbidden"),
    ):
        def handler(args, _name=name, **kwargs):
            effects.append(_name)
            return json.dumps({"executed": _name})

        registry.register(
            name=name, toolset=toolset,
            schema={"name": name, "parameters": {"type": "object"}},
            handler=handler,
        )
    registry.register_toolset_alias("dispatch-test-alias", "dispatch-test-plugin")
    # Resolve plugin aliases against the same real registry that executes them.
    monkeypatch.setattr("tools.registry.registry", registry)
    return registry, effects, config_path


@pytest.mark.parametrize("source", ["config", "environment"])
@pytest.mark.parametrize("allowed, admitted", [
    ("terminal", "terminal"),
    ("hermes-cli", "terminal"),
    ("dispatch-test-alias", "plugin_effect"),
])
def test_service_policy_blocks_effects(dispatch_policy, monkeypatch, source, allowed, admitted):
    registry, effects, config_path = dispatch_policy
    config_path.write_text(json.dumps({"agent": {"allowed_toolsets": [allowed]}}))
    if source == "environment":
        config_path.write_text(json.dumps({"agent": {"allowed_toolsets": []}}))
        monkeypatch.setenv("HERMES_ALLOWED_TOOLSETS", allowed)

    assert json.loads(registry.dispatch(admitted, {})) == {"executed": admitted}
    denied = json.loads(registry.dispatch("forbidden_effect", {}))
    assert denied["error_type"] == "toolset_not_allowed"
    assert effects == [admitted]

    # A prior successful dispatch cannot authorize effects after a policy change.
    monkeypatch.setenv("HERMES_ALLOWED_TOOLSETS", "")
    assert json.loads(registry.dispatch(admitted, {}))["error_type"] == "toolset_not_allowed"
    assert effects == [admitted]


def test_unrestricted_dispatch_keeps_existing_behavior(dispatch_policy):
    registry, effects, _ = dispatch_policy
    assert json.loads(registry.dispatch("forbidden_effect", {})) == {"executed": "forbidden_effect"}
    assert effects == ["forbidden_effect"]


def test_empty_config_policy_blocks_async_handler(dispatch_policy):
    registry, effects, config_path = dispatch_policy
    config_path.write_text(json.dumps({"agent": {"allowed_toolsets": []}}))

    async def handler(args, **kwargs):
        effects.append("async_effect")
        return json.dumps({"executed": "async_effect"})

    registry.register(
        name="async_effect", toolset="terminal",
        schema={"name": "async_effect", "parameters": {"type": "object"}},
        handler=handler,
    )
    assert json.loads(registry.dispatch("async_effect", {}))["error_type"] == "toolset_not_allowed"
    assert effects == []


def test_policy_lookup_error_cannot_execute_handler(dispatch_policy, monkeypatch):
    registry, effects, _ = dispatch_policy

    def unavailable_policy(*args, **kwargs):
        raise RuntimeError("policy unavailable")

    monkeypatch.setattr("toolsets.get_allowed_toolsets", unavailable_policy)
    result = json.loads(registry.dispatch("terminal", {}))
    assert "policy unavailable" in result["error"]
    assert effects == []
