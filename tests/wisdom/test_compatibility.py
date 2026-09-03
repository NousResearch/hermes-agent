from types import SimpleNamespace

from hermes_wisdom.compatibility import (
    LocalCapabilities,
    detect_local_capabilities,
    evaluate,
)
from hermes_wisdom.contract import (
    FilesystemRequirement,
    HermesRequirement,
    ModelRequirement,
    NetworkRequirement,
    PluginRequirement,
    RuntimeRequirement,
    SystemSpecification,
    ToolRequirement,
)


def spec(**updates):
    values = {"hermes": HermesRequirement(minimum_version="0.1.0")}
    values.update(updates)
    return SystemSpecification(**values)


def local(**updates):
    values = {
        "hermes_version": "1.0.0",
        "os": "darwin",
        "architecture": "arm64",
        "runtime": {"shell": True, "browser": False, "code": True, "sandbox": True},
    }
    values.update(updates)
    return LocalCapabilities(**values)


def test_all_four_compatibility_outcomes_are_deterministic():
    assert evaluate(spec(), local()).outcome == "compatible"
    assert (
        evaluate(
            spec(
                tools=[
                    ToolRequirement(name="git", minimum_version="2", auto_install=False)
                ]
            ),
            local(),
        ).outcome
        == "compatible_after_setup"
    )
    assert (
        evaluate(spec(known_limitations=["manual replay only"]), local()).outcome
        == "partial"
    )
    assert (
        evaluate(spec(runtime=RuntimeRequirement(browser=True)), local()).outcome
        == "blocked_pending_action"
    )


def test_skill_evaluator_is_not_a_compatibility_input():
    result = evaluate(spec(), local())
    assert not hasattr(result, "skill_evaluator")


def test_profile_detection_is_static_scoped_and_never_retains_secrets(
    monkeypatch, tmp_path
):
    readable = tmp_path / "readable.txt"
    readable.write_text("ok", encoding="utf-8")
    writable = tmp_path / "new-output.txt"
    requirement = spec(
        model=ModelRequirement(
            capabilities=["tools", "vision"], minimum_context_window=32000
        ),
        tools=[ToolRequirement(name="terminal", auto_install=False)],
        plugins=[PluginRequirement(id="example/plugin")],
        credentials=["EXAMPLE_TOKEN"],
        connections=["team-mcp", "telegram"],
        filesystem=FilesystemRequirement(read=[str(readable)], write=[str(writable)]),
        network=NetworkRequirement(destinations=["api.example.test"]),
        runtime=RuntimeRequirement(shell=True, browser=True, code=True),
    )
    monkeypatch.setattr(
        "agent.models_dev.get_model_capabilities",
        lambda *_args, **_kwargs: SimpleNamespace(
            supports_tools=True,
            supports_vision=True,
            supports_reasoning=False,
            context_window=128000,
        ),
    )
    monkeypatch.setattr(
        "hermes_cli.config.get_env_value",
        lambda name: (
            "super-secret-value"
            if name in {"EXAMPLE_TOKEN", "TELEGRAM_BOT_TOKEN"}
            else None
        ),
    )
    detected = detect_local_capabilities(
        requirement,
        config={
            "model": {"provider": "example", "default": "model"},
            "platform_toolsets": {"cli": ["terminal", "browser", "code_execution"]},
            "plugins": {"enabled": ["example/plugin"]},
            "mcp_servers": {"team-mcp": {"enabled": True}},
            "terminal": {"backend": "docker", "docker_network": False},
        },
    )

    assert detected.model_capabilities == frozenset({"tools", "vision"})
    assert detected.context_window == 128000
    assert "terminal" in detected.enabled_tools
    assert "example/plugin" in detected.plugins
    assert detected.credentials == frozenset({"EXAMPLE_TOKEN"})
    assert detected.connections == frozenset({"team-mcp", "telegram"})
    assert str(readable) in detected.filesystem_readable
    assert str(writable) in detected.filesystem_writable
    assert detected.network_access is False
    assert "super-secret-value" not in repr(detected)

    result = evaluate(requirement, detected)
    assert result.outcome == "compatible_after_setup"
    assert result.setup_actions == ("allow network access to api.example.test",)


def test_present_tool_and_plugin_with_unknown_version_satisfy_unversioned_requirement():
    result = evaluate(
        spec(
            tools=[ToolRequirement(name="git", auto_install=False)],
            plugins=[PluginRequirement(id="example/plugin")],
        ),
        local(enabled_tools={"git": None}, plugins={"example/plugin": None}),
    )
    assert result.outcome == "compatible"


def test_platform_and_architecture_aliases_match():
    result = evaluate(
        spec(platforms=["macOS"], architectures=["aarch64"]),
        local(os="darwin", architecture="arm64"),
    )
    assert result.outcome == "compatible"
