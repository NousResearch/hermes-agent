from __future__ import annotations

import json
from contextvars import Context

import pytest

from tools.execution_targets import (
    ExecutionTargetError,
    list_execution_targets,
    resolve_execution_target,
    set_execution_target_config_source,
)


@pytest.fixture(autouse=True)
def _reset_execution_target_config_source():
    try:
        yield
    finally:
        set_execution_target_config_source(None)


def _root(terminal: dict) -> dict:
    return {"terminal": terminal}


def test_legacy_omitted_and_default_select_the_flat_environment():
    config = _root({"backend": "ssh", "ssh_host": "host", "ssh_user": "user"})

    omitted = resolve_execution_target(config=config)
    explicit = resolve_execution_target("default", config=config)

    assert omitted.target == explicit.target == "default"
    assert omitted.backend == explicit.backend == "ssh"
    assert omitted.named is explicit.named is False
    assert omitted.config["ssh_host"] == "host"


def test_legacy_backend_metadata_respects_terminal_env_override(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "local")

    resolution = resolve_execution_target(
        config=_root({
            "backend": "ssh",
            "ssh_host": "configured-but-overridden",
            "ssh_user": "agent",
        }),
    )

    assert resolution.named is False
    assert resolution.target == "default"
    assert resolution.backend == "local"


def test_legacy_unknown_target_is_actionable():
    with pytest.raises(ExecutionTargetError) as excinfo:
        resolve_execution_target("devbox", config=_root({"backend": "local"}))

    message = str(excinfo.value)
    assert "devbox" in message
    assert "Available targets: 'default'" in message


def test_named_default_and_explicit_target_inherit_top_level_and_override():
    config = _root({
        "backend": "local",
        "timeout": 180,
        "container_memory": 4096,
        "default_target": "local",
        "targets": {
            "local": {"cwd": "/workspace/local"},
            "devbox": {
                "backend": "ssh",
                "ssh_host": "devbox.example.com",
                "ssh_user": "bruno",
                "cwd": "/home/bruno/project",
                "timeout": 45,
            },
        },
    })

    default = resolve_execution_target(config=config)
    devbox = resolve_execution_target("devbox", config=config)

    assert default.target == "local"
    assert default.backend == "local"
    assert default.is_default is True
    assert default.config["timeout"] == 180
    assert devbox.target == "devbox"
    assert devbox.backend == "ssh"
    assert devbox.is_default is False
    assert devbox.config["timeout"] == 45
    assert devbox.config["container_memory"] == 4096
    assert devbox.config["ssh_host"] == "devbox.example.com"
    assert "targets" not in devbox.config
    assert "default_target" not in devbox.config

    inventory = list_execution_targets(config=config)
    assert [(item.target, item.is_default) for item in inventory] == [
        ("devbox", False),
        ("local", True),
    ]


@pytest.mark.parametrize("default_target", [None, "", "missing"])
def test_named_targets_require_a_valid_default(default_target):
    terminal = {
        "backend": "local",
        "targets": {"zeta": {"backend": "local"}, "alpha": {"backend": "local"}},
    }
    if default_target is not None:
        terminal["default_target"] = default_target

    with pytest.raises(ExecutionTargetError) as excinfo:
        resolve_execution_target(config=_root(terminal))

    assert "Available targets: 'alpha', 'zeta'" in str(excinfo.value)


def test_unknown_named_target_lists_names_deterministically():
    config = _root({
        "default_target": "zeta",
        "targets": {"zeta": {"backend": "local"}, "alpha": {"backend": "local"}},
    })

    with pytest.raises(ExecutionTargetError) as excinfo:
        resolve_execution_target("other", config=config)

    assert "Available targets: 'alpha', 'zeta'" in str(excinfo.value)


@pytest.mark.parametrize(
    ("targets", "expected"),
    [
        ({"dev": "ssh"}, "must be a mapping"),
        ({"": {"backend": "local"}}, "non-empty strings"),
        ({1: {"backend": "local"}}, "non-empty strings"),
    ],
)
def test_malformed_target_entries_are_clear(targets, expected):
    with pytest.raises(ExecutionTargetError) as excinfo:
        resolve_execution_target(config=_root({"default_target": "dev", "targets": targets}))

    assert expected in str(excinfo.value)


def test_target_names_are_otherwise_arbitrary_static_strings():
    resolution = resolve_execution_target(
        "prod blue/1",
        config=_root({
            "default_target": "prod blue/1",
            "targets": {"prod blue/1": {"backend": "local"}},
        }),
    )

    assert resolution.target == "prod blue/1"


def test_tool_schemas_use_static_optional_string_target_fields():
    from tools.code_execution_tool import EXECUTE_CODE_SCHEMA
    from tools.file_tools import (
        PATCH_SCHEMA,
        READ_FILE_SCHEMA,
        SEARCH_FILES_SCHEMA,
        WRITE_FILE_SCHEMA,
    )
    from tools.terminal_tool import TERMINAL_SCHEMA

    for schema in (
        TERMINAL_SCHEMA,
        READ_FILE_SCHEMA,
        WRITE_FILE_SCHEMA,
        PATCH_SCHEMA,
        EXECUTE_CODE_SCHEMA,
    ):
        target = schema["parameters"]["properties"]["target"]
        assert target["type"] == "string"
        assert "enum" not in target
        assert "target" not in schema["parameters"].get("required", [])

    search_properties = SEARCH_FILES_SCHEMA["parameters"]["properties"]
    assert search_properties["target"]["enum"] == ["content", "files"]
    assert search_properties["execution_target"]["type"] == "string"
    assert "enum" not in search_properties["execution_target"]
    assert "compatibility" in SEARCH_FILES_SCHEMA["description"].lower()


def test_successful_terminal_result_reports_target_backend_and_cwd(monkeypatch, tmp_path):
    import tools.execution_targets as targets_mod
    import tools.terminal_tool as terminal_mod

    config = _root({
        "default_target": "local",
        "targets": {"local": {"backend": "local", "cwd": str(tmp_path)}},
    })
    monkeypatch.setattr(targets_mod, "_load_merged_config", lambda: config)
    monkeypatch.setattr(terminal_mod, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        terminal_mod,
        "_check_all_guards",
        lambda command, env_type, **kwargs: {"approved": True},
    )
    monkeypatch.setattr(terminal_mod, "_active_environments", {})
    monkeypatch.setattr(terminal_mod, "_last_activity", {})
    monkeypatch.setattr(terminal_mod, "_creation_locks", {})
    monkeypatch.setattr(terminal_mod, "_session_cwd", {})

    result = json.loads(terminal_mod.terminal_tool("pwd", task_id="session"))

    assert result["exit_code"] == 0
    assert result["target"] == "local"
    assert result["backend"] == "local"
    assert result["cwd"] == str(tmp_path)


def test_multiplex_profiles_do_not_share_environment_or_session_keys(monkeypatch):
    import tools.execution_targets as targets_mod

    config = _root({
        "default_target": "devbox",
        "targets": {"devbox": {"backend": "ssh"}},
    })
    monkeypatch.setattr(targets_mod, "_active_profile_scope", lambda: "profile-a")
    profile_a = resolve_execution_target("devbox", config=config)
    monkeypatch.setattr(targets_mod, "_active_profile_scope", lambda: "profile-b")
    profile_b = resolve_execution_target("devbox", config=config)

    assert profile_a.environment_key("default") != profile_b.environment_key("default")
    assert profile_a.session_key("chat") != profile_b.session_key("chat")


def test_classic_cli_config_override_is_context_scoped():
    ssh_config = _root({
        "default_target": "devbox",
        "targets": {"devbox": {"backend": "ssh"}},
    })
    local_config = _root({
        "default_target": "local",
        "targets": {"local": {"backend": "local"}},
    })

    def configure_and_resolve(config):
        set_execution_target_config_source(config)
        return resolve_execution_target().backend

    assert Context().run(configure_and_resolve, ssh_config) == "ssh"
    assert Context().run(configure_and_resolve, local_config) == "local"

    import threading

    set_execution_target_config_source(ssh_config)
    observed = []
    thread = threading.Thread(
        target=lambda: observed.append(resolve_execution_target().backend),
    )
    thread.start()
    thread.join(timeout=5)
    assert observed == ["ssh"]
