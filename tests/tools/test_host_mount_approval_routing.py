"""Host-mount classification and approval policy must agree."""

import pytest


def _config(backend, extra_args):
    return {
        "env_type": backend,
        "host_cwd": None,
        "docker_mount_cwd_to_workspace": False,
        "docker_volumes": [],
        "apple_container_volumes": [],
        "docker_extra_args": extra_args if backend == "docker" else [],
        "apple_container_extra_args": extra_args if backend == "apple_container" else [],
    }


def test_docker_clustered_bind_is_host_access():
    from tools.terminal_tool import _docker_has_host_access

    assert _docker_has_host_access(_config("docker", ["-iv/tmp:/mnt"])) is True


@pytest.mark.parametrize("extra_args", [
    ["--volumes-from", "fixture-source:ro"],
    ["--volumes-from=fixture-source"],
    ["--volume-driver", "local", "-v", "fixture-volume:/data"],
    ["--volume-driver=local", "-v", "fixture-volume:/data"],
    ["--mount", "type=volume,source=fixture-volume,target=/data,volume-driver=local,"
     "volume-opt=type=none,volume-opt=o=bind,volume-opt=device=/tmp/fixture"],
    ["--mount=type=volume,target=/data,volume-opt=device=/tmp/fixture,volume-opt=o=bind"],
])
def test_declared_indirect_docker_mounts_are_host_access(extra_args):
    from tools.terminal_tool import _docker_has_host_access

    assert _docker_has_host_access(_config("docker", extra_args)) is True


@pytest.mark.parametrize("extra_args, expected", [
    (["-iv", "/tmp:/mnt"], True),
    (["-v=/tmp:/mnt:ro"], True),
    (["--volume", "/tmp/with spaces:/mnt"], True),
    (["--mount", "type=bind,source=/tmp,target=/mnt"], True),
    (["--mount=type=volume,source=fixture,target=/mnt"], True),
    (["--volume=fixture:/mnt"], True),
    ([None, 42, "--network", "none"], False),
    (["--network", "none", "--tmpfs", "/tmp"], False),
    ([], False),
])
def test_apple_extra_mount_forms_are_classified(extra_args, expected):
    from tools.terminal_tool import _docker_has_host_access

    assert _docker_has_host_access(_config("apple_container", extra_args)) is expected


@pytest.mark.parametrize("volume", [".:/data", "..:/data"])
@pytest.mark.parametrize("structured", [True, False])
def test_docker_dot_sources_are_host_access(volume, structured):
    from tools.terminal_tool import _docker_has_host_access

    config = _config("docker", [])
    if structured:
        config["docker_volumes"] = [volume]
    else:
        config["docker_extra_args"] = ["--volume", volume]
    assert _docker_has_host_access(config) is True


@pytest.mark.parametrize("backend, args, expected", [
    ("docker", ["-iv/tmp:/mnt"], True),
    ("docker", ["--mount", ' "type=bind",source=/tmp,target=/mnt'], True),
    ("docker", ["--mount", "type=bind,src=.,dst=/mnt"], True),
    ("docker", ["--volumes-from", "fixture-source"], True),
    ("docker", ["--network", "none"], False),
    ("apple_container", ["-iv", "/tmp:/mnt"], True),
    ("apple_container", ["--network", "none"], False),
])
def test_detected_access_changes_both_approval_paths(monkeypatch, backend, args, expected):
    from tools import approval
    from tools import approval_context
    from tools.terminal_tool import _docker_has_host_access

    access = _docker_has_host_access(_config(backend, args))
    assert access is expected
    monkeypatch.setattr(approval, "_user_deny_block", lambda command: None)
    monkeypatch.setattr(approval, "_yolo_active", lambda: False)
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(approval, "_presence", lambda *a, **k: (None, False, True, True))
    monkeypatch.setattr(approval, "_unattended_contexts", lambda: [])
    monkeypatch.setattr(approval, "get_current_session_key", lambda *a, **k: "mount-fixture")
    monkeypatch.setattr(approval, "is_approved", lambda *a, **k: False)
    decisions = []

    def deny(*args, **kwargs):
        decisions.append(kwargs)
        return {"approved": False, "status": "pending_approval", "message": "fixture denied"}

    monkeypatch.setattr(approval, "_human_decision", deny)
    command = approval.check_dangerous_command("rm -rf /", backend, has_host_access=access)
    assert command["approved"] is (not expected)
    decisions.clear()
    script = approval.check_execute_code_guard("pass", backend, has_host_access=access)
    assert script["approved"] is (not expected)
    assert bool(decisions) is expected


@pytest.mark.parametrize("backend, args, expected", [
    ("docker", ["-iv/tmp:/mnt"], True),
    ("docker", ["--network", "none"], False),
    ("apple_container", ["-iv", "/tmp:/mnt"], True),
    ("apple_container", ["--network", "none"], False),
])
def test_terminal_guard_dispatch_receives_detector_result(monkeypatch, backend, args, expected):
    import json
    from tools import terminal_tool as terminal

    received = []

    def deny(command, env_type, has_host_access=False):
        received.append((env_type, has_host_access))
        return {"approved": False, "message": "fixture denied"}

    monkeypatch.setattr(terminal, "_check_all_guards", deny)
    with pytest.raises(terminal._Rejected) as rejected:
        terminal._run_approval_guards("printf fixture", backend, _config(backend, args), force=False)
    assert received == [(backend, expected)]
    assert "fixture denied" in json.dumps(json.loads(rejected.value.result_json))


@pytest.mark.parametrize("backend, args, expected", [
    ("docker", ["-iv/tmp:/mnt"], True),
    ("docker", ["--network", "none"], False),
    ("apple_container", ["-iv", "/tmp:/mnt"], True),
    ("apple_container", ["--network", "none"], False),
])
def test_execute_code_dispatch_receives_detector_result(monkeypatch, backend, args, expected):
    import json
    from tools import approval, code_execution_tool, terminal_tool

    received = []
    monkeypatch.setattr(code_execution_tool, "SANDBOX_AVAILABLE", True)
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: _config(backend, args))

    def deny(code, env_type, has_host_access=False):
        received.append((env_type, has_host_access))
        return {"approved": False, "message": "fixture denied"}

    monkeypatch.setattr(approval, "check_execute_code_guard", deny)
    monkeypatch.setattr(
        code_execution_tool, "_execute_remote",
        lambda *a, **k: pytest.fail("denied script reached remote execution"),
    )
    result = json.loads(code_execution_tool.execute_code("pass", task_id="mount-fixture"))
    assert received == [(backend, expected)]
    assert "fixture denied" in json.dumps(result)


def test_other_backend_ignores_local_container_keys():
    from tools.terminal_tool import _docker_has_host_access

    config = _config("ssh", [])
    config["docker_extra_args"] = ["-v", "/tmp:/mnt"]
    config["apple_container_volumes"] = ["/tmp:/mnt"]
    assert _docker_has_host_access(config) is False


@pytest.mark.parametrize("mount", [
    "type=volume,source=fixture,target=/data,volume-driver=local",
    'type=volume,"source=unterminated',
    "type=bind,type=volume,source=fixture,target=/data",
])
def test_ambiguous_or_driver_mounts_keep_guards(mount):
    from tools.terminal_tool import _docker_has_host_access

    assert _docker_has_host_access(_config("docker", ["--mount", mount])) is True
