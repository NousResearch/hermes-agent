import json

from agent import secure_input_broker as broker
from tools.secure_input_tool import request_secure_input, set_secure_input_callback
from tools.terminal_tool import terminal_tool


def setup_function():
    broker.clear_all()
    set_secure_input_callback(None)


def teardown_function():
    broker.clear_all()
    set_secure_input_callback(None)


def test_request_secure_input_returns_only_reference():
    set_secure_input_callback(lambda purpose, prompt, metadata=None: "ghp_exampleSecretValue123456")

    result = json.loads(
        request_secure_input(
            purpose="github_token",
            title="GitHub token",
            allowed_consumers=["terminal"],
        )
    )

    assert result["success"] is True
    assert result["secret_ref"].startswith("secret://session/")
    assert result["redacted"] is True
    assert "ghp_exampleSecretValue123456" not in json.dumps(result)


def test_terminal_can_consume_stdin_secret_ref_without_echoing_secret(tmp_path, monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "local")
    info = broker.register_secret(
        "plain-secret-no-prefix",
        purpose="unit_test",
        allowed_consumers=["terminal"],
        single_use=True,
    )

    result = json.loads(
        terminal_tool(
            command="python -c 'import sys; data=sys.stdin.read(); print(len(data)); print(data)'",
            workdir=str(tmp_path),
            stdin_secret_ref=info["secret_ref"],
            timeout=30,
        )
    )

    assert result["exit_code"] == 0
    assert "22" in result["output"]
    assert "plain-secret-no-prefix" not in result["output"]
    assert "[REDACTED SECURE INPUT]" in result["output"]


def test_single_use_secret_ref_is_rejected_after_first_consume(tmp_path, monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "local")
    info = broker.register_secret(
        "reuse-secret-value",
        purpose="unit_test",
        allowed_consumers=["terminal"],
        single_use=True,
    )

    first = json.loads(
        terminal_tool(
            command="python -c 'import sys; sys.stdin.read(); print(\"ok\")'",
            workdir=str(tmp_path),
            stdin_secret_ref=info["secret_ref"],
            timeout=30,
        )
    )
    second = json.loads(
        terminal_tool(
            command="python -c 'print(\"should not run\")'",
            workdir=str(tmp_path),
            stdin_secret_ref=info["secret_ref"],
            timeout=30,
        )
    )

    assert first["exit_code"] == 0
    assert second["exit_code"] == -1
    assert "already been consumed" in second["error"]


def test_terminal_validation_does_not_consume_secret_ref(tmp_path, monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "local")
    info = broker.register_secret(
        "validation-secret-value",
        purpose="unit_test",
        allowed_consumers=["terminal"],
        single_use=True,
    )

    rejected = json.loads(
        terminal_tool(
            command="python -m http.server 8000",
            workdir=str(tmp_path),
            stdin_secret_ref=info["secret_ref"],
            timeout=30,
        )
    )
    accepted = json.loads(
        terminal_tool(
            command="python -c 'import sys; print(sys.stdin.read())'",
            workdir=str(tmp_path),
            stdin_secret_ref=info["secret_ref"],
            timeout=30,
        )
    )

    assert rejected["exit_code"] == -1
    assert "background=true" in rejected["error"]
    assert accepted["exit_code"] == 0
    assert "[REDACTED SECURE INPUT]" in accepted["output"]


def test_terminal_blocks_heredoc_backends_without_consuming_ref(tmp_path, monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "daytona")
    info = broker.register_secret(
        "heredoc-secret-value",
        purpose="unit_test",
        allowed_consumers=["terminal"],
        single_use=True,
    )

    blocked = json.loads(
        terminal_tool(
            command="python -c 'print(\"nope\")'",
            workdir=str(tmp_path),
            stdin_secret_ref=info["secret_ref"],
            timeout=30,
        )
    )

    assert blocked["exit_code"] == -1
    assert "embed stdin in command text" in blocked["error"]
    assert broker.consume_secret(info["secret_ref"], consumer="terminal") == "heredoc-secret-value"


def test_env_secret_ref_is_transient_and_redacted(tmp_path, monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "local")
    info = broker.register_secret(
        "env-secret-no-prefix",
        purpose="unit_test",
        allowed_consumers=["terminal"],
        single_use=True,
    )

    first = json.loads(
        terminal_tool(
            command="python -c 'import os; print(os.environ[\"HERMES_TEST_SECRET\"])'",
            workdir=str(tmp_path),
            env_secret_refs={"HERMES_TEST_SECRET": info["secret_ref"]},
            timeout=30,
        )
    )
    second = json.loads(
        terminal_tool(
            command="python -c 'import os; print(os.getenv(\"HERMES_TEST_SECRET\", \"missing\"))'",
            workdir=str(tmp_path),
            timeout=30,
        )
    )

    assert first["exit_code"] == 0
    assert "env-secret-no-prefix" not in first["output"]
    assert "[REDACTED SECURE INPUT]" in first["output"]
    assert second["exit_code"] == 0
    assert second["output"].strip() == "missing"


def test_env_secret_ref_blocks_external_api_backends_without_consuming_ref(tmp_path, monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "daytona")
    info = broker.register_secret(
        "env-heredoc-secret-value",
        purpose="unit_test",
        allowed_consumers=["terminal"],
        single_use=True,
    )

    blocked = json.loads(
        terminal_tool(
            command="python -c 'print(\"nope\")'",
            workdir=str(tmp_path),
            env_secret_refs={"HERMES_TEST_SECRET": info["secret_ref"]},
            timeout=30,
        )
    )

    assert blocked["exit_code"] == -1
    assert "no out-of-band env channel" in blocked["error"]
    assert broker.consume_secret(info["secret_ref"], consumer="terminal") == "env-heredoc-secret-value"


def test_workdir_rejection_does_not_consume_ref(tmp_path, monkeypatch):
    # Regression for the deferred-consumption fix: a command rejected by
    # workdir validation must leave a single-use ref usable, because
    # consumption is deferred until after workdir validation succeeds.
    monkeypatch.setenv("TERMINAL_ENV", "local")
    info = broker.register_secret(
        "workdir-secret-value",
        purpose="unit_test",
        allowed_consumers=["terminal"],
        single_use=True,
    )

    rejected = json.loads(
        terminal_tool(
            command="echo hi",
            workdir="/tmp; rm -rf /",  # disallowed characters -> blocked
            stdin_secret_ref=info["secret_ref"],
            timeout=30,
        )
    )

    assert rejected["exit_code"] == -1
    assert rejected["status"] == "blocked"
    # Ref survives the rejection and is still consumable exactly once.
    assert broker.consume_secret(info["secret_ref"], consumer="terminal") == "workdir-secret-value"


def test_env_secret_ref_background_blocked_on_non_local_without_consuming(tmp_path, monkeypatch):
    # env_secret_refs in background is only supported on the local backend.
    # Docker background goes through spawn_via_env (no secure env channel), so
    # it must be blocked and must not consume the ref.
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    info = broker.register_secret(
        "bg-env-secret-value",
        purpose="unit_test",
        allowed_consumers=["terminal"],
        single_use=True,
    )

    blocked = json.loads(
        terminal_tool(
            command="sleep 1",
            workdir=str(tmp_path),
            env_secret_refs={"HERMES_TEST_SECRET": info["secret_ref"]},
            background=True,
            timeout=30,
        )
    )

    assert blocked["exit_code"] == -1
    assert blocked["status"] == "blocked"
    assert "background" in blocked["error"]
    assert broker.consume_secret(info["secret_ref"], consumer="terminal") == "bg-env-secret-value"


def test_docker_run_bash_injects_secret_env_out_of_band(monkeypatch):
    # Regression for finding #1: Docker must forward secret env vars via
    # `docker exec -e KEY` (name only) with the value supplied through the
    # docker CLI process environment — never on argv, never in the snapshot.
    from tools.environments import docker as docker_mod

    captured = {}

    def _fake_popen_bash(cmd, stdin_data=None, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(docker_mod, "_popen_bash", _fake_popen_bash)

    env = object.__new__(docker_mod.DockerEnvironment)
    env._docker_exe = "docker"
    env._container_id = "deadbeef"
    env._init_env_args = []

    env._run_bash("echo hi", login=False, secret_env={"GH_TOKEN": "supersecret"})

    cmd = captured["cmd"]
    # Name-only -e flag present...
    assert "-e" in cmd
    assert "GH_TOKEN" in cmd
    # ...but the value must never appear on argv (host `ps` exposure).
    assert "supersecret" not in cmd
    assert "GH_TOKEN=supersecret" not in cmd
    # Value is delivered via the docker CLI process environment instead.
    assert captured["kwargs"]["env"]["GH_TOKEN"] == "supersecret"


def test_docker_advertises_secret_env_capability_and_ssh_does_not():
    from tools.environments.docker import DockerEnvironment
    from tools.environments.local import LocalEnvironment
    from tools.environments.singularity import SingularityEnvironment
    from tools.environments.ssh import SSHEnvironment

    assert DockerEnvironment.supports_secret_env is True
    assert LocalEnvironment.supports_secret_env is True
    assert SingularityEnvironment.supports_secret_env is True
    # SSH has no reliable out-of-band env channel; it must not advertise support.
    assert getattr(SSHEnvironment, "supports_secret_env", False) is False


def test_check_secure_input_requirements_reflects_callback():
    # Finding #3: the tool must only be advertised where a masked-capture
    # callback exists. With no callback registered the requirement is False so
    # the registry filters the tool from non-interactive surfaces.
    from tools import secure_input_tool

    set_secure_input_callback(None)
    assert secure_input_tool.check_secure_input_requirements() is False

    set_secure_input_callback(lambda purpose, prompt, metadata=None: "x")
    try:
        assert secure_input_tool.check_secure_input_requirements() is True
    finally:
        set_secure_input_callback(None)
