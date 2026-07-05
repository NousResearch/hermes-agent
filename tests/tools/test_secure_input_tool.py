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
    assert "external APIs" in blocked["error"]
    assert broker.consume_secret(info["secret_ref"], consumer="terminal") == "env-heredoc-secret-value"
