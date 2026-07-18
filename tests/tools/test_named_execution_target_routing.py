from __future__ import annotations

import json
import os
import time

import pytest


def _named_config(cwds: dict[str, str], default: str = "alpha") -> dict:
    return {
        "terminal": {
            "timeout": 60,
            "lifetime_seconds": 3600,
            "default_target": default,
            "targets": {
                name: {"backend": "local", "cwd": cwd}
                for name, cwd in cwds.items()
            },
        },
    }


@pytest.fixture
def isolated_target_state(monkeypatch):
    import tools.file_tools as file_mod
    import tools.terminal_tool as terminal_mod

    monkeypatch.setattr(terminal_mod, "_active_environments", {})
    monkeypatch.setattr(terminal_mod, "_last_activity", {})
    monkeypatch.setattr(terminal_mod, "_creation_locks", {})
    monkeypatch.setattr(terminal_mod, "_session_cwd", {})
    monkeypatch.setattr(terminal_mod, "_task_env_overrides", {})
    monkeypatch.setattr(terminal_mod, "_active_turn_counts", {})
    monkeypatch.setattr(terminal_mod, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(file_mod, "_file_ops_cache", {})
    monkeypatch.setattr(file_mod, "_read_tracker", {})
    monkeypatch.setattr(file_mod, "_patch_failure_tracker", {})
    monkeypatch.setattr(
        terminal_mod,
        "_check_all_guards",
        lambda command, env_type, **kwargs: {"approved": True},
    )
    yield terminal_mod, file_mod
    for env in list(terminal_mod._active_environments.values()):
        try:
            env.cleanup()
        except Exception:
            pass


def test_same_task_reuses_within_target_and_isolates_across_targets(
    monkeypatch, tmp_path, isolated_target_state,
):
    import tools.execution_targets as targets_mod

    terminal_mod, _ = isolated_target_state
    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta"
    alpha.mkdir()
    beta.mkdir()
    monkeypatch.setattr(
        targets_mod, "_load_merged_config",
        lambda: _named_config({"alpha": str(alpha), "beta": str(beta)}),
    )

    first = json.loads(terminal_mod.terminal_tool("pwd", task_id="child-a"))
    second = json.loads(terminal_mod.terminal_tool("pwd", task_id="child-b", target="alpha"))
    third = json.loads(terminal_mod.terminal_tool("pwd", task_id="child-a", target="beta"))

    assert first["output"] == second["output"] == str(alpha)
    assert third["output"] == str(beta)
    assert set(terminal_mod._active_environments) == {("default", "alpha"), ("default", "beta")}


def test_ssh_target_routes_terminal_and_file_adapter_to_same_environment(
    monkeypatch, isolated_target_state,
):
    import tools.execution_targets as targets_mod

    terminal_mod, file_mod = isolated_target_state
    config = {
        "terminal": {
            "backend": "local",
            "default_target": "local",
            "targets": {
                "local": {"backend": "local", "cwd": "/workspace/local"},
                "devbox": {
                    "backend": "ssh",
                    "cwd": "/srv/project",
                    "ssh_host": "devbox.example.com",
                    "ssh_user": "agent",
                },
            },
        },
    }
    monkeypatch.setattr(targets_mod, "_load_merged_config", lambda: config)
    created = []

    class FakeSshEnvironment:
        def __init__(self, cwd):
            self.cwd = cwd
            self.is_persistent = True

        def execute(self, command, cwd="", timeout=None, **kwargs):
            return {"output": self.cwd + "\n", "returncode": 0}

        def cleanup(self):
            pass

    def fake_create(**kwargs):
        created.append(kwargs)
        return FakeSshEnvironment(kwargs["cwd"])

    monkeypatch.setattr(terminal_mod, "_create_environment", fake_create)

    result = json.loads(terminal_mod.terminal_tool(
        "pwd", task_id="session", target="devbox",
    ))
    file_ops = file_mod._get_file_ops("session", "devbox")

    assert result["target"] == "devbox"
    assert result["backend"] == "ssh"
    assert result["output"] == "/srv/project"
    assert created[0]["env_type"] == "ssh"
    assert created[0]["ssh_config"]["host"] == "devbox.example.com"
    assert created[0]["cwd"] == "/srv/project"
    assert file_ops.env is terminal_mod._active_environments[("default", "devbox")]


def test_ssh_paths_stay_remote_relative_and_ignore_host_workspace_override(
    monkeypatch, tmp_path, isolated_target_state,
):
    import tools.execution_targets as targets_mod

    terminal_mod, file_mod = isolated_target_state
    config = {
        "terminal": {
            "backend": "local",
            "default_target": "devbox",
            "targets": {
                "devbox": {
                    "backend": "ssh",
                    "cwd": ".",
                    "ssh_host": "devbox.example.com",
                    "ssh_user": "agent",
                },
                "container": {"backend": "docker", "cwd": "."},
            },
        },
    }
    monkeypatch.setattr(targets_mod, "_load_merged_config", lambda: config)
    terminal_mod.register_task_env_overrides(
        "session", {"cwd": str(tmp_path)},
    )

    assert file_mod._authoritative_workspace_root("session", "devbox") == "."
    assert file_mod._backend_operation_path(
        "relative.txt", str(tmp_path / "relative.txt"), "session", "devbox",
    ) == "relative.txt"
    assert file_mod._backend_operation_path(
        "~/notes.txt", str(tmp_path / "notes.txt"), "session", "devbox",
    ) == "~/notes.txt"
    assert file_mod._authoritative_workspace_root(
        "session", "container",
    ) == "/root"

    terminal_mod.record_session_cwd(
        "session-a", "/srv/a", target="devbox",
    )
    terminal_mod.record_session_cwd(
        "session-b", "/srv/b", target="devbox",
    )
    assert file_mod._backend_operation_path(
        "relative.txt", "/host/wrong", "session-a", "devbox",
    ) == "/srv/a/relative.txt"
    assert file_mod._backend_operation_path(
        "relative.txt", "/host/wrong", "session-b", "devbox",
    ) == "/srv/b/relative.txt"
    rewritten = file_mod._backend_v4a_patch(
        "*** Begin Patch\n*** Update File: relative.txt\n*** End Patch",
        "session-a",
        "devbox",
    )
    assert "*** Update File: /srv/a/relative.txt" in rewritten


def test_legacy_ssh_search_preserves_relative_path(monkeypatch, isolated_target_state):
    from tools.file_operations import SearchResult

    _, file_mod = isolated_target_state
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    monkeypatch.setenv("TERMINAL_CWD", "~")
    captured = {}

    class FakeOps:
        def search(self, **kwargs):
            captured.update(kwargs)
            return SearchResult(matches=[], total_count=0)

    monkeypatch.setattr(file_mod, "_get_file_ops", lambda *args, **kwargs: FakeOps())

    result = json.loads(file_mod.search_tool("needle", path="."))

    assert not result.get("error")
    assert captured["path"] == "."


def test_ssh_reads_never_use_host_mtime_dedup(
    monkeypatch, tmp_path, isolated_target_state,
):
    _, file_mod = isolated_target_state
    import tools.execution_targets as targets_mod
    from tools.file_operations import ReadResult

    host_twin = tmp_path / "same.txt"
    host_twin.write_text("host\n", encoding="utf-8")
    config = {
        "terminal": {
            "default_target": "devbox",
            "targets": {
                "devbox": {
                    "backend": "ssh",
                    "cwd": str(tmp_path),
                    "ssh_host": "devbox.example.com",
                    "ssh_user": "agent",
                },
            },
        },
    }
    monkeypatch.setattr(targets_mod, "_load_merged_config", lambda: config)

    class FakeOps:
        calls = 0

        @classmethod
        def read_file(cls, path, offset, limit):
            cls.calls += 1
            return ReadResult(
                content=f"remote-{cls.calls}", total_lines=1, file_size=8,
            )

    monkeypatch.setattr(
        file_mod, "_get_file_ops", lambda task_id, target=None: FakeOps(),
    )

    first = json.loads(file_mod.read_file_tool(
        "same.txt", task_id="session", target="devbox",
    ))
    second = json.loads(file_mod.read_file_tool(
        "same.txt", task_id="session", target="devbox",
    ))

    assert first["content"] != second["content"]
    assert second.get("status") != "unchanged"
    assert FakeOps.calls == 2


def test_target_specific_cd_does_not_cross_talk_to_terminal_or_file_tools(
    monkeypatch, tmp_path, isolated_target_state,
):
    import tools.execution_targets as targets_mod

    terminal_mod, file_mod = isolated_target_state
    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta"
    alpha_sub = alpha / "sub"
    beta_sub = beta / "sub"
    alpha_sub.mkdir(parents=True)
    beta_sub.mkdir(parents=True)
    (alpha_sub / "which.txt").write_text("alpha\n", encoding="utf-8")
    (beta / "which.txt").write_text("beta-root\n", encoding="utf-8")
    monkeypatch.setattr(
        targets_mod, "_load_merged_config",
        lambda: _named_config({"alpha": str(alpha), "beta": str(beta)}),
    )

    cd_result = json.loads(terminal_mod.terminal_tool("cd sub", task_id="session", target="alpha"))
    beta_pwd = json.loads(terminal_mod.terminal_tool("pwd", task_id="session", target="beta"))
    alpha_read = json.loads(file_mod.read_file_tool("which.txt", task_id="session", target="alpha"))
    beta_read = json.loads(file_mod.read_file_tool("which.txt", task_id="session", target="beta"))

    assert cd_result["exit_code"] == 0
    assert beta_pwd["output"] == str(beta)
    assert "alpha" in alpha_read["content"]
    assert "beta-root" in beta_read["content"]


def test_local_targets_route_write_read_patch_search_and_cache_separately(
    monkeypatch, tmp_path, isolated_target_state,
):
    import tools.execution_targets as targets_mod

    _, file_mod = isolated_target_state
    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta"
    alpha.mkdir()
    beta.mkdir()
    monkeypatch.setattr(
        targets_mod, "_load_merged_config",
        lambda: _named_config({"alpha": str(alpha), "beta": str(beta)}),
    )

    write_alpha = json.loads(file_mod.write_file_tool(
        "shared.txt", "alpha-value\n", task_id="session", target="alpha",
    ))
    write_beta = json.loads(file_mod.write_file_tool(
        "shared.txt", "beta-value\n", task_id="session", target="beta",
    ))
    patch_alpha = json.loads(file_mod.patch_tool(
        path="shared.txt", old_string="alpha-value\n", new_string="alpha-patched\n",
        task_id="session", target="alpha",
    ))
    search_alpha = json.loads(file_mod.search_tool(
        "alpha-patched", path=".", task_id="session", execution_target="alpha",
    ))
    search_beta = json.loads(file_mod.search_tool(
        "beta-value", path=".", task_id="session", execution_target="beta",
    ))

    assert not write_alpha.get("error") and not write_beta.get("error")
    assert not patch_alpha.get("error")
    assert (alpha / "shared.txt").read_text(encoding="utf-8") == "alpha-patched\n"
    assert (beta / "shared.txt").read_text(encoding="utf-8") == "beta-value\n"
    assert search_alpha["matches"] and search_beta["matches"]
    assert write_alpha["target"] == patch_alpha["target"] == "alpha"
    assert search_beta["target"] == "beta"
    assert set(file_mod._file_ops_cache) == {("default", "alpha"), ("default", "beta")}
    assert ("session", "alpha") in file_mod._read_tracker
    assert ("session", "beta") in file_mod._read_tracker


def test_real_config_loader_routes_terminal_and_files_between_two_local_targets(
    monkeypatch, tmp_path, isolated_target_state,
):
    import yaml

    import tools.file_tools as file_mod
    import tools.terminal_tool as terminal_mod

    alpha = tmp_path / "configured-alpha"
    beta = tmp_path / "configured-beta"
    alpha.mkdir()
    beta.mkdir()
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump({
            "terminal": {
                "default_target": "alpha",
                "targets": {
                    "alpha": {"backend": "local", "cwd": str(alpha)},
                    "beta": {"backend": "local", "cwd": str(beta)},
                },
            },
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("HERMES_PROFILE", raising=False)

    alpha_pwd = json.loads(terminal_mod.terminal_tool(
        "pwd", task_id="configured-session", target="alpha",
    ))
    beta_pwd = json.loads(terminal_mod.terminal_tool(
        "pwd", task_id="configured-session", target="beta",
    ))
    assert alpha_pwd["output"] == str(alpha)
    assert beta_pwd["output"] == str(beta)

    assert json.loads(file_mod.write_file_tool(
        "same.txt", "from alpha\n", task_id="configured-session", target="alpha",
    ))["target"] == "alpha"
    assert json.loads(file_mod.write_file_tool(
        "same.txt", "from beta\n", task_id="configured-session", target="beta",
    ))["target"] == "beta"
    assert "from alpha" in json.loads(file_mod.read_file_tool(
        "same.txt", task_id="configured-session", target="alpha",
    ))["content"]
    assert "from beta" in json.loads(file_mod.read_file_tool(
        "same.txt", task_id="configured-session", target="beta",
    ))["content"]


def test_workspace_override_applies_only_to_default_named_target(
    monkeypatch, tmp_path, isolated_target_state,
):
    import tools.execution_targets as targets_mod

    terminal_mod, file_mod = isolated_target_state
    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta"
    host_workspace = tmp_path / "host-workspace"
    alpha.mkdir()
    beta.mkdir()
    host_workspace.mkdir()
    monkeypatch.setattr(
        targets_mod, "_load_merged_config",
        lambda: _named_config({"alpha": str(alpha), "beta": str(beta)}),
    )

    # ACP/TUI/gateway surfaces register the host workspace as a task override.
    # It should seed the configured default target, not replace an explicit
    # remote/container target's own cwd.
    terminal_mod.register_task_env_overrides(
        "session", {"cwd": str(host_workspace)},
    )

    alpha_pwd = json.loads(terminal_mod.terminal_tool(
        "pwd", task_id="session", target="alpha",
    ))
    beta_pwd = json.loads(terminal_mod.terminal_tool(
        "pwd", task_id="session", target="beta",
    ))
    beta_write = json.loads(file_mod.write_file_tool(
        "target-only.txt", "beta\n", task_id="session", target="beta",
    ))

    assert alpha_pwd["output"] == str(host_workspace)
    assert beta_pwd["output"] == str(beta)
    assert beta_write["resolved_path"] == str(beta / "target-only.txt")
    assert (beta / "target-only.txt").read_text(encoding="utf-8") == "beta\n"
    assert not (host_workspace / "target-only.txt").exists()


def test_unknown_target_errors_are_returned_by_execution_and_file_tools(
    monkeypatch, tmp_path, isolated_target_state,
):
    import tools.code_execution_tool as code_mod
    import tools.execution_targets as targets_mod

    terminal_mod, file_mod = isolated_target_state
    alpha = tmp_path / "alpha"
    alpha.mkdir()
    monkeypatch.setattr(
        targets_mod, "_load_merged_config",
        lambda: _named_config({"alpha": str(alpha)}),
    )

    results = [
        json.loads(terminal_mod.terminal_tool("pwd", target="missing")),
        json.loads(file_mod.read_file_tool("x.txt", target="missing")),
        json.loads(file_mod.write_file_tool("x.txt", "x", target="missing")),
        json.loads(file_mod.search_tool(
            "x", path=".", execution_target="missing",
        )),
        json.loads(code_mod.execute_code("print('x')", target="missing")),
    ]

    for result in results:
        assert "missing" in result["error"]
        assert "Available targets: 'alpha'" in result["error"]
    assert not (alpha / "x.txt").exists()


def test_cleanup_without_target_removes_all_task_scopes_and_explicit_removes_one(
    monkeypatch, isolated_target_state,
):
    import tools.execution_targets as targets_mod

    terminal_mod, _ = isolated_target_state

    class FakeEnv:
        def __init__(self):
            self.cleaned = 0

        def cleanup(self):
            self.cleaned += 1

    config = _named_config({"alpha": "/a", "beta": "/b"})
    monkeypatch.setattr(targets_mod, "_load_merged_config", lambda: config)
    alpha = FakeEnv()
    beta = FakeEnv()
    terminal_mod._active_environments.update({
        ("default", "alpha"): alpha,
        ("default", "beta"): beta,
    })
    terminal_mod._last_activity.update({
        ("default", "alpha"): time.time(),
        ("default", "beta"): time.time(),
    })

    # Unregistered delegate ids collapse to the parent's "default" scope for
    # execution, but closing a delegate must preserve legacy cleanup semantics
    # and must not tear down the parent's shared environments.
    terminal_mod.cleanup_vm("child-a")
    terminal_mod.cleanup_vm("child-a", target="alpha")
    assert alpha.cleaned == beta.cleaned == 0

    terminal_mod.cleanup_vm("default", target="alpha")
    assert alpha.cleaned == 1 and beta.cleaned == 0
    assert set(terminal_mod._active_environments) == {("default", "beta")}

    terminal_mod.cleanup_vm("default")
    assert beta.cleaned == 1
    assert terminal_mod._active_environments == {}


def test_per_turn_cleanup_preserves_only_persistent_named_siblings(
    monkeypatch, isolated_target_state,
):
    import tools.execution_targets as targets_mod

    terminal_mod, file_mod = isolated_target_state

    class FakeEnv:
        def __init__(self, persistent):
            self._persistent = persistent
            self.cleaned = 0

        def cleanup(self):
            self.cleaned += 1

    monkeypatch.setattr(
        targets_mod,
        "_load_merged_config",
        lambda: _named_config({"alpha": "/a", "beta": "/b"}),
    )
    key_a = ("default", "alpha")
    key_b = ("default", "beta")
    persistent = FakeEnv(True)
    ephemeral = FakeEnv(False)
    terminal_mod._active_environments.update({
        key_a: persistent,
        key_b: ephemeral,
    })
    terminal_mod._last_activity.update({key_a: 1.0, key_b: 1.0})
    persistent_ops = object()
    ephemeral_ops = object()
    file_mod._file_ops_cache.update({
        key_a: persistent_ops,
        key_b: ephemeral_ops,
    })

    terminal_mod.register_environment_turn("turn-a")
    terminal_mod.register_environment_turn("turn-b")
    assert terminal_mod.release_environment_turn("turn-a") == 1
    terminal_mod.cleanup_vm(
        "turn-a", preserve_persistent=True, include_collapsed=False,
    )
    assert key_b in terminal_mod._active_environments

    assert terminal_mod.release_environment_turn("turn-b") == 0
    terminal_mod.cleanup_vm(
        "turn-b", preserve_persistent=True, include_collapsed=True,
    )

    assert terminal_mod._active_environments[key_a] is persistent
    assert key_b not in terminal_mod._active_environments
    assert persistent.cleaned == 0
    assert ephemeral.cleaned == 1
    assert file_mod._file_ops_cache[key_a] is persistent_ops
    assert key_b not in file_mod._file_ops_cache

    from tools.process_registry import process_registry

    active_env = FakeEnv(False)
    terminal_mod._active_environments[key_b] = active_env
    monkeypatch.setattr(
        process_registry, "has_active_processes", lambda key: key == key_b,
    )
    terminal_mod.cleanup_vm(
        "turn-c", preserve_persistent=True, include_collapsed=True,
    )
    assert terminal_mod._active_environments[key_b] is active_env
    assert active_env.cleaned == 0


def test_local_persistent_config_marks_environment(
    monkeypatch, isolated_target_state,
):
    terminal_mod, _ = isolated_target_state

    class FakeLocal:
        def __init__(self, cwd, timeout):
            self.cwd = cwd
            self.timeout = timeout

    monkeypatch.setattr(terminal_mod, "_LocalEnvironment", FakeLocal)
    env = terminal_mod._create_environment(
        "local", "", "/workspace", 30,
        local_config={"persistent": True},
    )

    assert terminal_mod._environment_is_persistent(env)


def test_process_metadata_survives_status_list_and_checkpoint(monkeypatch, tmp_path):
    import tools.process_registry as process_mod

    checkpoint = tmp_path / "processes.json"
    monkeypatch.setattr(process_mod, "CHECKPOINT_PATH", checkpoint)
    registry = process_mod.ProcessRegistry()
    session = process_mod.ProcessSession(
        id="proc_targeted",
        command="sleep 30",
        task_id="default",
        target="devbox",
        backend="ssh",
        timeout_seconds=7,
        environment_task_key="profile-a:default",
        pid=4321,
        host_start_time=99,
        started_at=time.time(),
    )
    registry._running[session.id] = session

    assert registry.has_active_processes(("profile-a:default", "devbox"))
    assert registry.poll(session.id)["target"] == "devbox"
    assert registry.list_sessions()[0]["backend"] == "ssh"
    registry._write_checkpoint()
    raw = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert raw[0]["target"] == "devbox"
    assert raw[0]["timeout_seconds"] == 7
    assert raw[0]["environment_task_key"] == "profile-a:default"
    assert raw[0]["backend"] == "ssh"

    recovered = process_mod.ProcessRegistry()
    monkeypatch.setattr(recovered, "_host_pid_is_ours", lambda pid, started: True)
    assert recovered.recover_from_checkpoint() == 1
    recovered_session = recovered.get(session.id)
    assert recovered_session is not None
    assert recovered_session.target == "devbox"
    assert recovered_session.backend == "ssh"
    assert recovered_session.timeout_seconds == 7
    assert recovered_session.environment_task_key == "profile-a:default"

    recovered_session.exited = True
    waited = recovered.wait(session.id, timeout=99)
    assert "configured limit of 7s" in waited["timeout_note"]


def test_execute_code_inherits_target_for_nested_tools_and_preserves_override(
    monkeypatch, tmp_path, isolated_target_state,
):
    import tools.code_execution_tool as code_mod
    import tools.execution_targets as targets_mod

    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta"
    alpha.mkdir()
    beta.mkdir()
    monkeypatch.setattr(
        targets_mod, "_load_merged_config",
        lambda: _named_config({"alpha": str(alpha), "beta": str(beta)}),
    )
    source = code_mod.generate_hermes_tools_module(
        ["write_file", "read_file", "search_files"],
    )
    namespace = {}
    exec(source, namespace)
    calls = []
    namespace["_call"] = lambda name, args: calls.append((name, args)) or {}

    namespace["write_file"]("nested.txt", "alpha-default\n")
    namespace["write_file"]("nested.txt", "beta-explicit\n", target="beta")
    namespace["search_files"]("needle")

    inherited = [
        (name, code_mod._inherit_execution_target(name, args, "alpha"))
        for name, args in calls
    ]
    assert inherited[0][1]["target"] == "alpha"
    assert inherited[1][1]["target"] == "beta"
    assert inherited[2][1]["execution_target"] == "alpha"

    remote_config = _named_config({"alpha": str(alpha), "beta": str(beta)})
    remote_config["terminal"]["targets"]["alpha"]["backend"] = "ssh"
    remote_config["terminal"]["targets"]["alpha"]["ssh_host"] = "example.invalid"
    monkeypatch.setattr(targets_mod, "_load_merged_config", lambda: remote_config)
    forwarded = {}
    monkeypatch.setattr(code_mod, "_get_execution_mode", lambda: "project")
    monkeypatch.setattr(
        code_mod, "_execute_remote",
        lambda code, task_id, enabled_tools, target=None, mode="strict": (
            forwarded.update(target=target, mode=mode)
            or json.dumps({"status": "success"})
        ),
    )

    result = json.loads(code_mod.execute_code("print('ok')", target="alpha"))
    assert result["status"] == "success"
    assert forwarded["target"] == "alpha"
    assert forwarded["mode"] == "project"


def test_execute_code_routes_real_nested_file_call_to_selected_local_target(
    monkeypatch, tmp_path, isolated_target_state,
):
    import tools.code_execution_tool as code_mod
    import tools.execution_targets as targets_mod

    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta"
    alpha.mkdir()
    beta.mkdir()
    monkeypatch.setattr(
        targets_mod, "_load_merged_config",
        lambda: _named_config({"alpha": str(alpha), "beta": str(beta)}),
    )

    result = json.loads(code_mod.execute_code(
        "from hermes_tools import write_file\n"
        "print(write_file('from-code.txt', 'beta-via-rpc\\n'))\n",
        task_id="execute-target-session",
        enabled_tools=["write_file"],
        target="beta",
    ))

    assert result["status"] == "success"
    assert result["target"] == "beta"
    assert result["backend"] == "local"
    assert (beta / "from-code.txt").read_text(encoding="utf-8") == "beta-via-rpc\n"
    assert not (alpha / "from-code.txt").exists()


def test_remote_project_mode_executes_from_target_session_cwd(
    monkeypatch, isolated_target_state,
):
    import tools.code_execution_tool as code_mod
    import tools.execution_targets as targets_mod

    config = _named_config({"alpha": "/srv/configured", "beta": "/b"})
    config["terminal"]["targets"]["alpha"].update({
        "backend": "ssh",
        "ssh_host": "example.invalid",
    })
    monkeypatch.setattr(targets_mod, "_load_merged_config", lambda: config)
    terminal_mod, _ = isolated_target_state
    terminal_mod.record_session_cwd(
        "remote-project", "/srv/session-project", target="alpha",
    )

    calls = []

    class FakeEnv:
        cwd = "/srv/configured"

        def execute(self, command, cwd=None, timeout=None, stdin_data=None):
            calls.append((command, cwd))
            if "command -v python3" in command:
                return {"output": "OK\n", "returncode": 0}
            return {"output": "done\n", "returncode": 0}

        def get_temp_dir(self):
            return "/tmp"

    class DummyThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def join(self, timeout=None):
            pass

    env = FakeEnv()
    monkeypatch.setattr(
        code_mod, "_get_or_create_env", lambda task_id, target=None: (env, "ssh"),
    )
    monkeypatch.setattr(code_mod, "_ship_file_to_remote", lambda *args, **kwargs: None)
    monkeypatch.setattr(code_mod.threading, "Thread", DummyThread)
    monkeypatch.setattr(
        code_mod, "_load_config", lambda: {"timeout": 30, "max_tool_calls": 5},
    )

    result = json.loads(code_mod._execute_remote(
        "print('ok')", "remote-project", [], target="alpha", mode="project",
    ))

    assert result["status"] == "success"
    script_calls = [item for item in calls if "python3 /tmp/hermes_exec_" in item[0]]
    assert script_calls
    assert script_calls[0][1] == "/srv/session-project"


def test_tool_output_persistence_uses_the_result_target(monkeypatch):
    import agent.tool_executor as executor

    default_env = object()
    beta_env = object()

    def fake_get_active_env(_task_id, target=None):
        return beta_env if target == "beta" else default_env

    monkeypatch.setattr(executor, "get_active_env", fake_get_active_env)

    result = json.dumps({"target": "beta", "output": "x"})
    assert executor._active_env_for_tool_result(
        "session", "terminal", {}, result,
    ) is beta_env

    captured = {}

    def fake_enforce(messages, env=None, env_resolver=None, config=None):
        assert callable(env_resolver)
        captured["default"] = env
        captured["selected"] = env_resolver(messages[0])

    monkeypatch.setattr(executor, "enforce_turn_budget", fake_enforce)
    messages = [{"content": "large", "tool_call_id": "call-beta"}]
    target_map = {"call-beta": "beta"}
    executor._enforce_target_aware_turn_budget(
        messages, "session", executor.DEFAULT_BUDGET, target_map,
    )

    assert captured == {"default": default_env, "selected": beta_env}
    assert target_map == {}
    assert "_execution_target" not in messages[0]

    captured.clear()
    executor._enforce_target_aware_turn_budget(
        [{"content": "web result", "tool_call_id": "call-web"}],
        "session",
        executor.DEFAULT_BUDGET,
        {},
    )
    assert captured == {"default": default_env, "selected": default_env}
    persisted = executor._append_persisted_target_hint(
        f"{executor.PERSISTED_OUTPUT_TAG}\nfull output saved",
        "beta",
    )
    assert 'Execution target for this saved output: "beta"' in persisted
    assert "`target` to `read_file`" in persisted

    monkeypatch.setattr(
        executor, "get_active_env",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad target config")),
    )
    assert executor._active_env_for_tool_result(
        "session", "terminal", {"target": "broken"},
    ) is None
    captured.clear()
    target_map = {"call-broken": "broken"}
    executor._enforce_target_aware_turn_budget(
        [{"content": "error", "tool_call_id": "call-broken"}],
        "session",
        executor.DEFAULT_BUDGET,
        target_map,
    )
    assert captured == {"default": None, "selected": None}


def test_subdirectory_hints_follow_local_target_and_skip_remote_host(
    monkeypatch, tmp_path, isolated_target_state,
):
    from types import SimpleNamespace

    import agent.tool_executor as executor
    import tools.execution_targets as targets_mod

    local_root = tmp_path / "local-target"
    subdir = local_root / "src"
    subdir.mkdir(parents=True)
    (subdir / "AGENTS.md").write_text("LOCAL TARGET HINT", encoding="utf-8")
    config = {
        "terminal": {
            "default_target": "local",
            "targets": {
                "local": {"backend": "local", "cwd": str(local_root)},
                "devbox": {
                    "backend": "ssh",
                    "cwd": "/srv/project",
                    "ssh_host": "devbox.example.com",
                    "ssh_user": "agent",
                },
            },
        },
    }
    monkeypatch.setattr(targets_mod, "_load_merged_config", lambda: config)

    class DefaultTracker:
        def __init__(self):
            self.calls = 0

        def check_tool_call(self, *args, **kwargs):
            self.calls += 1
            return "WRONG HOST HINT"

    default_tracker = DefaultTracker()
    agent = SimpleNamespace(_subdirectory_hints=default_tracker)
    local_hint = executor._target_subdirectory_hints(
        agent, "session", "read_file", {"path": "src/main.py"}, "local",
    )
    remote_hint = executor._target_subdirectory_hints(
        agent, "session", "read_file", {"path": "src/main.py"}, "devbox",
    )
    config["terminal"]["default_target"] = "devbox"
    omitted_remote_hint = executor._target_subdirectory_hints(
        agent, "session", "read_file", {"path": "src/main.py"}, None,
    )

    assert isinstance(local_hint, str)
    assert "LOCAL TARGET HINT" in local_hint
    assert executor._selected_local_target_cwd(
        "session", "write_file", {"target": "local"},
    ) == str(local_root)
    assert executor._selected_local_target_cwd(
        "session", "terminal", {"target": "devbox"},
    ) is None
    assert remote_hint is None
    assert omitted_remote_hint is None
    assert default_tracker.calls == 0


def test_targetless_intervening_tool_resets_all_target_read_trackers(
    monkeypatch, isolated_target_state,
):
    import model_tools
    import tools.execution_targets as targets_mod

    _, file_mod = isolated_target_state
    assert "memory" not in model_tools._TARGET_SELECTOR_TOOLS
    assert "terminal" in model_tools._TARGET_SELECTOR_TOOLS
    for key in ("session", ("session", "alpha"), ("session", "beta")):
        file_mod._read_tracker[key] = {
            "last_key": "same",
            "consecutive": 4,
            "dedup_hits": {"same": 2},
        }

    file_mod.notify_other_tool_call("session")

    for data in file_mod._read_tracker.values():
        assert data["last_key"] is None
        assert data["consecutive"] == 0
        assert data["dedup_hits"] == {}

    monkeypatch.setattr(targets_mod, "_active_profile_scope", lambda: "profile-a")
    profile_key = ("profile-profile-a:session", "alpha")
    file_mod._read_tracker[profile_key] = {
        "dedup": {"region": (1.0, 2, "hash")},
        "dedup_hits": {"region": 3},
    }
    file_mod.reset_file_dedup("session")
    assert file_mod._read_tracker[profile_key]["dedup"] == {}
    assert file_mod._read_tracker[profile_key]["dedup_hits"] == {}


def test_sudo_cache_and_nopasswd_probe_are_target_scoped(
    monkeypatch, isolated_target_state,
):
    terminal_mod, _ = isolated_target_state
    terminal_mod._reset_cached_sudo_passwords()

    terminal_mod._set_cached_sudo_password("local-secret", "local", "local")
    assert terminal_mod._get_cached_sudo_password("local", "local") == "local-secret"
    assert terminal_mod._get_cached_sudo_password("devbox", "ssh") == ""

    calls = []

    class Probe:
        returncode = 0

    monkeypatch.setattr(
        terminal_mod.subprocess,
        "run",
        lambda *args, **kwargs: calls.append((args, kwargs)) or Probe(),
    )
    with terminal_mod._scoped_sudo_execution("devbox", "ssh"):
        assert terminal_mod._sudo_nopasswd_works() is False
    assert calls == []

    with terminal_mod._scoped_sudo_execution("local", "local"):
        assert terminal_mod._sudo_nopasswd_works() is True
    assert len(calls) == 1

    monkeypatch.setenv("SUDO_PASSWORD", "default-secret")
    with terminal_mod._scoped_sudo_execution(
        "devbox", "ssh", named=True, sudo_password="target-secret",
    ):
        transformed, selected_password = terminal_mod._transform_sudo_command(
            "sudo id",
        )
    assert "target-secret" not in transformed
    assert "default-secret" not in transformed
    assert selected_password == "target-secret\n"

    with terminal_mod._scoped_sudo_execution("devbox", "ssh", named=True):
        terminal_mod._set_cached_sudo_password("cached-target")
        _, cached_password = terminal_mod._transform_sudo_command("sudo id")
    assert cached_password == "cached-target\n"


def test_requirements_keep_tools_registered_when_any_target_is_usable(
    monkeypatch, isolated_target_state,
):
    import tools.execution_targets as targets_mod

    terminal_mod, _ = isolated_target_state
    config = {
        "terminal": {
            "default_target": "broken-docker",
            "targets": {
                "broken-docker": {"backend": "docker", "cwd": "/workspace"},
                "local": {"backend": "local", "cwd": "/workspace/local"},
            },
        },
    }
    monkeypatch.setattr(targets_mod, "_load_merged_config", lambda: config)
    checked = []
    monkeypatch.setattr(
        terminal_mod,
        "_check_terminal_config_requirements",
        lambda cfg: checked.append(cfg["env_type"]) or cfg["env_type"] == "local",
    )

    assert terminal_mod.check_terminal_requirements() is True
    assert checked == ["local"]


def test_local_target_aliases_share_file_state_lock_namespace(
    monkeypatch, tmp_path, isolated_target_state,
):
    import tools.execution_targets as targets_mod
    from tools import file_state

    _, file_mod = isolated_target_state
    shared = tmp_path / "shared"
    shared.mkdir()
    path = shared / "same.txt"
    path.write_text("v1", encoding="utf-8")
    config = {
        "terminal": {
            "default_target": "alpha",
            "targets": {
                "alpha": {"backend": "local", "cwd": str(shared)},
                "alias": {"backend": "local", "cwd": str(shared)},
            },
        },
    }
    monkeypatch.setattr(targets_mod, "_load_merged_config", lambda: config)
    assert file_mod._file_state_namespace("reader", "alpha") is None
    assert file_mod._file_state_namespace("writer", "alias") is None

    registry = file_state.FileStateRegistry()
    monkeypatch.setattr(file_state, "_registry", registry)
    file_state.record_read(("reader", "alpha"), path, namespace=None)
    file_state.note_write(("writer", "alias"), path, namespace=None)
    warning = file_state.check_stale(("reader", "alpha"), path, namespace=None)
    assert warning is not None
    assert "writer" in warning


def test_command_approval_payload_and_observer_include_target_metadata(monkeypatch):
    from tools import approval as approval_mod

    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.setattr(approval_mod, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(approval_mod, "detect_hardline_command", lambda command: (False, None))
    monkeypatch.setattr(approval_mod, "_check_sudo_stdin_guard", lambda command: (False, None))
    monkeypatch.setattr(approval_mod, "_match_user_deny_rule", lambda command: None)
    monkeypatch.setattr(approval_mod, "_command_matches_permanent_allowlist", lambda command: False)
    monkeypatch.setattr(
        approval_mod, "detect_dangerous_command",
        lambda command: (True, "danger", "dangerous command"),
    )
    monkeypatch.setattr("tools.tirith_security.check_command_security", lambda command: {
        "action": "allow", "findings": [], "summary": "",
    })
    session_key = "approval-target-session"
    token = approval_mod.set_current_session_key(session_key)
    seen = {}
    hooks = []

    def notify(data):
        seen.update(data)
        with approval_mod._lock:
            entry = approval_mod._gateway_queues[session_key][-1]
            entry.result = "deny"
            entry.event.set()

    monkeypatch.setattr(approval_mod, "_fire_approval_hook", lambda name, **data: hooks.append((name, data)))
    approval_mod.register_gateway_notify(session_key, notify)
    try:
        approval_mod.check_all_command_guards(
            "danger", "local", execution_target="alpha", execution_backend="local",
        )
    finally:
        approval_mod.unregister_gateway_notify(session_key)
        approval_mod.reset_current_session_key(token)

    assert seen["target"] == "alpha"
    assert seen["backend"] == "local"
    assert "alpha" in seen["description"] and "local" in seen["description"]
    pre_hook = next(data for name, data in hooks if name == "pre_approval_request")
    assert pre_hook["target"] == "alpha"
    assert pre_hook["backend"] == "local"


def test_execute_code_approval_payload_includes_target_metadata(monkeypatch):
    from tools import approval as approval_mod

    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.setattr(approval_mod, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(approval_mod, "is_approved", lambda *args: False)
    monkeypatch.setattr(approval_mod, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(
        approval_mod, "is_current_session_yolo_enabled", lambda: False,
    )
    seen = {}
    monkeypatch.setattr(
        approval_mod, "_await_gateway_decision",
        lambda session_key, notify_cb, approval_data, surface: (
            seen.update(approval_data) or {"resolved": True, "choice": "once"}
        ),
    )
    session_token = approval_mod.set_current_session_key("target-approval")
    try:
        with approval_mod._lock:
            approval_mod._gateway_notify_cbs["target-approval"] = lambda data: None
        result = approval_mod.check_execute_code_guard(
            "print('ok')", "ssh", execution_target="devbox",
            execution_backend="ssh",
        )
    finally:
        approval_mod.reset_current_session_key(session_token)
        with approval_mod._lock:
            approval_mod._gateway_notify_cbs.pop("target-approval", None)

    assert result["approved"] is True
    assert seen["target"] == "devbox"
    assert seen["backend"] == "ssh"
    assert "devbox" in seen["description"]


def test_legacy_flat_config_still_uses_plain_string_keys(
    monkeypatch, tmp_path, isolated_target_state,
):
    import tools.execution_targets as targets_mod

    terminal_mod, _ = isolated_target_state
    monkeypatch.setattr(
        targets_mod, "_load_merged_config",
        lambda: {"terminal": {"backend": "local", "cwd": str(tmp_path), "timeout": 60}},
    )

    result = json.loads(terminal_mod.terminal_tool("pwd", task_id="legacy"))

    assert result["exit_code"] == 0
    assert "default" in terminal_mod._active_environments
    assert all(not isinstance(key, tuple) for key in terminal_mod._active_environments)
