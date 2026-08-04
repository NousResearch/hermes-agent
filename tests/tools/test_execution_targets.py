from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
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


def test_named_target_rejects_unknown_backend():
    config = _root({
        "default_target": "dev",
        "targets": {"dev": {"backend": "telepathy"}},
    })
    with pytest.raises(ExecutionTargetError, match="unknown backend 'telepathy'"):
        resolve_execution_target(config=config)


def test_named_target_accepts_current_vercel_sandbox_backend():
    resolution = resolve_execution_target(config=_root({
        "default_target": "cloud",
        "targets": {
            "cloud": {
                "backend": "vercel_sandbox",
                "vercel_runtime": "python3.13",
                "cwd": "/vercel/sandbox",
            },
        },
    }))

    assert resolution.named is True
    assert resolution.backend == "vercel_sandbox"
    assert resolution.config["vercel_runtime"] == "python3.13"


@pytest.mark.parametrize(
    ("setting", "value", "expected"),
    [
        ("docker_network", "definitely", "expected boolean"),
        ("docker_forward_env", {"HOME": True}, "expected list"),
        ("docker_env", ["NOT_A_MAPPING"], "expected mapping"),
    ],
)
def test_named_target_rejects_malformed_backend_settings(setting, value, expected):
    config = _root({
        "default_target": "dev",
        "targets": {
            "dev": {"backend": "docker", setting: value},
        },
    })
    with pytest.raises(ExecutionTargetError, match=expected):
        resolve_execution_target(config=config)


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
        "targets": {"devbox": {
            "backend": "ssh",
            "ssh_host": "example.invalid",
            "ssh_user": "agent",
        }},
    })
    monkeypatch.setattr(targets_mod, "_active_profile_scope", lambda: "profile-a")
    profile_a = resolve_execution_target("devbox", config=config)
    monkeypatch.setattr(targets_mod, "_active_profile_scope", lambda: "profile-b")
    profile_b = resolve_execution_target("devbox", config=config)

    assert profile_a.environment_key("default") != profile_b.environment_key("default")
    assert profile_a.session_key("chat") != profile_b.session_key("chat")


def test_target_spec_fingerprint_changes_with_security_relevant_config():
    base = _root({
        "default_target": "devbox",
        "targets": {
            "devbox": {
                "backend": "ssh",
                "ssh_host": "one.example",
                "ssh_user": "agent",
            },
        },
    })
    changed = json.loads(json.dumps(base))
    changed["terminal"]["targets"]["devbox"]["ssh_host"] = "two.example"

    first = resolve_execution_target(config=base)
    second = resolve_execution_target(config=changed)

    assert first.spec_fingerprint != second.spec_fingerprint
    assert first.security_scope != second.security_scope
    assert first.environment_key("session") == second.environment_key("session")


def test_target_fingerprint_is_keyed_when_config_contains_secrets(
    monkeypatch, tmp_path,
):
    import hermes_constants
    import tools.execution_targets as targets_mod

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    targets_mod._fingerprint_keys.clear()
    resolution = resolve_execution_target(
        "alpha",
        config=_root({
            "targets": {
                "alpha": {
                    "backend": "local",
                    "sudo_password": "guessable-secret",
                },
            },
        }),
    )
    canonical = json.dumps(
        {
            "config": dict(resolution.config),
            "is_default": resolution.is_default,
        },
        sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, default=repr,
    )
    plain = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]

    assert resolution.spec_fingerprint != plain
    key_path = tmp_path / ".execution-target-fingerprint-key"
    assert key_path.stat().st_mode & 0o077 == 0


def test_default_status_changes_runtime_and_approval_identity():
    targets = {
        "alpha": {"backend": "local", "cwd": "/workspace"},
        "beta": {"backend": "local", "cwd": "/workspace"},
    }
    alpha_default = resolve_execution_target(
        "alpha", config=_root({"default_target": "alpha", "targets": targets}),
    )
    beta_default = resolve_execution_target(
        "alpha", config=_root({"default_target": "beta", "targets": targets}),
    )

    assert alpha_default.is_default is True
    assert beta_default.is_default is False
    assert alpha_default.spec_fingerprint != beta_default.spec_fingerprint
    assert alpha_default.security_scope != beta_default.security_scope


def test_backend_task_id_is_bounded_and_collision_resistant_for_named_targets():
    config = _root({
        "default_target": "alpha",
        "targets": {
            "alpha": {"backend": "docker", "docker_image": "image:a"},
            "beta": {"backend": "docker", "docker_image": "image:b"},
        },
    })
    task_id = "benchmark-" + ("x" * 500)
    alpha = resolve_execution_target("alpha", config=config).backend_task_id(task_id)
    beta = resolve_execution_target("beta", config=config).backend_task_id(task_id)

    assert len(alpha) <= 63
    assert len(beta) <= 63
    assert alpha != beta
    assert alpha.startswith("task-") and "-target-" in alpha


def test_persistent_storage_id_survives_policy_edits_but_not_owner_changes(
    monkeypatch, tmp_path,
):
    import hermes_constants
    import tools.execution_targets as targets_mod

    base = _root({
        "default_target": "alpha",
        "timeout": 30,
        "targets": {
            "alpha": {"backend": "docker", "docker_image": "image:a"},
            "beta": {"backend": "docker", "docker_image": "image:a"},
        },
    })
    changed = json.loads(json.dumps(base))
    changed["terminal"]["timeout"] = 90
    changed["terminal"]["targets"]["alpha"]["docker_image"] = "image:b"

    first = resolve_execution_target("alpha", config=base)
    edited = resolve_execution_target("alpha", config=changed)
    beta = resolve_execution_target("beta", config=base)

    assert first.backend_task_id("session") != edited.backend_task_id("session")
    assert first.storage_task_id("session") == edited.storage_task_id("session")
    assert first.storage_task_id("session") != beta.storage_task_id("session")
    assert len(first.storage_task_id("x" * 500)) <= 63

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path / "a")
    home_a = first.storage_task_id("session")
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path / "b")
    assert home_a != first.storage_task_id("session")

    monkeypatch.setattr(targets_mod, "_active_profile_scope", lambda: "other-profile")
    other_profile = resolve_execution_target("alpha", config=base)
    assert first.storage_task_id("session") != other_profile.storage_task_id("session")


def test_legacy_identity_does_not_require_fingerprint_key(monkeypatch):
    import tools.execution_targets as targets_mod
    import tools.terminal_tool as terminal_mod

    resolution = resolve_execution_target(
        config={"terminal": {"backend": "local", "cwd": "."}},
    )
    monkeypatch.setattr(
        targets_mod,
        "_target_fingerprint_key",
        lambda: (_ for _ in ()).throw(PermissionError("read-only home")),
    )

    env = SimpleNamespace()
    terminal_mod._record_environment_target(env, resolution)
    assert resolution.backend_task_id("session") == "session"
    assert resolution.metadata() == {"target": "default", "backend": "local"}
    assert env._hermes_target_scope is None


def test_classic_cli_config_override_is_context_scoped():
    ssh_config = _root({
        "default_target": "devbox",
        "targets": {"devbox": {
            "backend": "ssh",
            "ssh_host": "example.invalid",
            "ssh_user": "agent",
        }},
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


def test_unknown_target_setting_fails_with_target_name_and_suggestion():
    config = _root({
        "targets": {
            "sandbox": {
                "backend": "docker",
                "docker_netwrok": "host",
            },
        },
    })
    with pytest.raises(
        ExecutionTargetError,
        match="target 'sandbox'.*docker_netwrok.*docker_network",
    ):
        resolve_execution_target("sandbox", config=config)


def test_backend_inapplicable_target_setting_fails_before_environment_creation():
    config = _root({
        "targets": {
            "workstation": {
                "backend": "local",
                "ssh_host": "example.invalid",
            },
        },
    })
    with pytest.raises(
        ExecutionTargetError,
        match="target 'workstation'.*ssh_host.*does not apply.*local",
    ):
        resolve_execution_target("workstation", config=config)


def test_missing_required_ssh_fields_and_invalid_port_fail_in_resolver():
    missing = _root({
        "targets": {"remote": {"backend": "ssh", "ssh_user": "agent"}},
    })
    with pytest.raises(
        ExecutionTargetError,
        match="target 'remote'.*requires non-empty 'ssh_host'",
    ):
        resolve_execution_target("remote", config=missing)

    invalid_port = _root({
        "targets": {
            "remote": {
                "backend": "ssh",
                "ssh_host": "example.invalid",
                "ssh_user": "agent",
                "ssh_port": 70000,
            },
        },
    })
    with pytest.raises(
        ExecutionTargetError,
        match="target 'remote'.*ssh_port.*1 to 65535",
    ):
        resolve_execution_target("remote", config=invalid_port)
