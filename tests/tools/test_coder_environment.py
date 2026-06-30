import json
import logging
import re
import threading
import uuid
from urllib.parse import parse_qs, urlparse
from unittest.mock import MagicMock

from hermes_state import SessionDB
from tools.environments.coder import CoderEnvironment, coder_workspace_name_for_task
import tools.terminal_tool as terminal_tool_module


class _FakeWebSocket:
    def __init__(self, messages):
        self._messages = list(messages)
        self.requested = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def recv(self, timeout=None, decode=None):
        self.requested.append({"timeout": timeout, "decode": decode})
        if not self._messages:
            raise EOFError
        message = self._messages.pop(0)
        if isinstance(message, BaseException):
            raise message
        return message


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")

    def json(self):
        return self._payload


def test_get_env_config_reads_coder_values(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "coder")
    monkeypatch.setenv("CODER_URL", "https://coder.example")
    monkeypatch.setenv("CODER_API_KEY", "secret-token")
    monkeypatch.setenv("CODER_ORGANIZATION", "acme")
    monkeypatch.setenv("CODER_WORKSPACE", "shared-dev")
    monkeypatch.setenv("CODER_TEMPLATE", "devcontainer")

    config = terminal_tool_module._get_env_config()

    assert config["env_type"] == "coder"
    assert config["coder_url"] == "https://coder.example"
    assert config["coder_api_key"] == "secret-token"
    assert config["coder_organization"] == "acme"
    assert config["coder_workspace"] == "shared-dev"
    assert config["coder_template"] == "devcontainer"
    assert config["coder_workspace_startup_timeout"] == 180


def test_get_env_config_reads_coder_workspace_startup_timeout(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "coder")
    monkeypatch.setenv("TERMINAL_CODER_WORKSPACE_STARTUP_TIMEOUT", "240")

    config = terminal_tool_module._get_env_config()

    assert config["coder_workspace_startup_timeout"] == 240


def test_coder_environment_defaults_snapshot_timeout_to_three_minutes(monkeypatch):
    monkeypatch.setattr(
        "tools.environments.coder.coder_workspace_name_for_task",
        lambda task_id, db=None: "hermes-task",
    )

    env = CoderEnvironment(
        base_url="https://coder.example",
        template_name="devcontainer",
        task_id="task-coder",
        api_key="secret-token",
        init_session=False,
    )

    assert env._snapshot_timeout == 180


def test_get_env_config_defaults_to_local_without_terminal_env(monkeypatch):
    monkeypatch.delenv("TERMINAL_ENV", raising=False)
    monkeypatch.delenv("CODER_URL", raising=False)
    monkeypatch.delenv("CODER_API_KEY", raising=False)
    monkeypatch.delenv("CODER_ORGANIZATION", raising=False)
    monkeypatch.delenv("CODER_WORKSPACE", raising=False)
    monkeypatch.delenv("CODER_TEMPLATE", raising=False)

    config = terminal_tool_module._get_env_config()

    assert config["env_type"] == "local"


def test_get_env_config_uses_env_bridged_coder_values(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "coder")
    monkeypatch.setenv("CODER_URL", "https://configured.example")
    monkeypatch.setenv("CODER_ORGANIZATION", "configured-org")
    monkeypatch.setenv("CODER_WORKSPACE", "configured-workspace")
    monkeypatch.setenv("CODER_TEMPLATE", "configured-template")
    monkeypatch.setenv("CODER_API_KEY", "secret-token")

    config = terminal_tool_module._get_env_config()

    assert config["coder_url"] == "https://configured.example"
    assert config["coder_organization"] == "configured-org"
    assert config["coder_workspace"] == "configured-workspace"
    assert config["coder_template"] == "configured-template"
    assert config["coder_api_key"] == "secret-token"
    assert config["coder_url"] == terminal_tool_module.os.getenv("CODER_URL")
    assert config["coder_organization"] == terminal_tool_module.os.getenv("CODER_ORGANIZATION")
    assert config["coder_workspace"] == terminal_tool_module.os.getenv("CODER_WORKSPACE")
    assert config["coder_template"] == terminal_tool_module.os.getenv("CODER_TEMPLATE")


def test_get_env_config_coder_discards_host_terminal_cwd(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "coder")
    monkeypatch.setenv("TERMINAL_CWD", "/Users/perqin-moego")

    config = terminal_tool_module._get_env_config()

    assert config["cwd"] == "~"


def test_get_env_config_coder_defaults_to_remote_home_not_host_home(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "coder")
    monkeypatch.delenv("TERMINAL_CWD", raising=False)

    config = terminal_tool_module._get_env_config()

    assert config["cwd"] == "~"


def test_create_environment_constructs_coder_backend(monkeypatch):
    sentinel = object()
    ctor = MagicMock(return_value=sentinel)
    monkeypatch.setattr(terminal_tool_module, "_CoderEnvironment", ctor)

    result = terminal_tool_module._create_environment(
        env_type="coder",
        image="ignored",
        cwd="/root",
        timeout=30,
        container_config={
            "coder_url": "https://coder.example",
            "coder_api_key": "secret-token",
            "coder_organization": "acme",
            "coder_workspace": "shared-dev",
            "coder_template": "devcontainer",
            "coder_forward_env": [],
            "coder_workspace_startup_timeout": 240,
        },
        task_id="task-coder",
    )

    assert result is sentinel
    ctor.assert_called_once_with(
        base_url="https://coder.example",
        template_name="devcontainer",
        task_id="task-coder",
        api_key="secret-token",
        organization_name="acme",
        workspace_name="shared-dev",
        cwd="/root",
        timeout=30,
        forward_env=[],
        workspace_startup_timeout=240,
    )


def test_coder_requirements_missing_template_logs_error(monkeypatch, caplog):
    monkeypatch.setenv("TERMINAL_ENV", "coder")
    monkeypatch.setenv("CODER_URL", "https://coder.example")
    monkeypatch.setenv("CODER_API_KEY", "secret-token")
    monkeypatch.delenv("CODER_TEMPLATE", raising=False)

    with caplog.at_level(logging.ERROR):
        ok = terminal_tool_module.check_terminal_requirements()

    assert ok is False
    assert any(
        "Coder backend selected but CODER_URL, CODER_API_KEY, and CODER_TEMPLATE must all be set"
        in record.getMessage()
        for record in caplog.records
    )


def test_coder_requirements_with_template_only_passes(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "coder")
    monkeypatch.setenv("CODER_URL", "https://coder.example")
    monkeypatch.setenv("CODER_API_KEY", "secret-token")
    monkeypatch.setenv("CODER_TEMPLATE", "devcontainer")

    assert terminal_tool_module.check_terminal_requirements() is True


def test_workspace_name_uses_lineage_root(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="20260521_173045_ab12cd", source="cli")
    db.create_session(
        session_id="20260521_180000_ef3456",
        source="cli",
        parent_session_id="20260521_173045_ab12cd",
    )

    assert (
        coder_workspace_name_for_task("20260521_180000_ef3456", db=db)
        == "hermes-20260521-173045-ab12cd"
    )


def test_workspace_name_sanitizes_non_session_task_id(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")

    workspace = coder_workspace_name_for_task("Task/ID:With Weird_Chars__AndVeryVeryLongSuffix1234567890", db=db)

    assert workspace.startswith("hermes-")
    assert len(workspace) <= 32
    assert re.fullmatch(r"[a-z0-9][a-z0-9-]{0,31}", workspace)


def test_coder_environment_initializes_session_snapshot_without_recursive_execute(monkeypatch):
    workspace_payload = {
        "id": "workspace-123",
        "name": "shared-dev",
        "latest_build": {
            "transition": "start",
            "resources": [{"agents": [{"id": "agent-123", "status": "connected", "lifecycle_state": "ready"}]}],
        },
    }
    requests_get = MagicMock(
        side_effect=[
            _FakeResponse({"workspaces": [workspace_payload]}),
            _FakeResponse(workspace_payload),
            _FakeResponse({"workspaces": [workspace_payload]}),
            _FakeResponse(workspace_payload),
        ]
    )
    connect_urls = []

    def fake_connect(url, **_kwargs):
        connect_urls.append(url)
        query = parse_qs(urlparse(url).query)
        reconnect_id = query["reconnect"][0]
        exit_marker = f"__HERMES_EXIT_{reconnect_id}__"
        command = query["command"][0]
        cwd_match = re.search(r"__HERMES_CWD_[0-9a-f]{12}__", command)
        assert cwd_match is not None
        cwd_marker = cwd_match.group(0)
        return _FakeWebSocket([f"\n{cwd_marker}/home/coder{cwd_marker}\n\n{exit_marker}0{exit_marker}\n".encode()])

    monkeypatch.setattr("tools.environments.coder.requests.get", requests_get)
    monkeypatch.setattr("tools.environments.coder.requests.post", MagicMock())
    monkeypatch.setattr("tools.environments.coder.connect", fake_connect)
    monkeypatch.setattr(
        "tools.environments.coder.coder_workspace_name_for_task",
        lambda task_id, db=None: "shared-dev",
    )
    monkeypatch.setenv("HERMES_CODER_SNAPSHOT_TEST", "forwarded-value")

    env = CoderEnvironment(
        base_url="https://coder.example",
        template_name="devcontainer",
        task_id="20260521_180000_ef3456",
        api_key="secret-token",
        workspace_name="shared-dev",
        timeout=5,
        forward_env=["HERMES_CODER_SNAPSHOT_TEST"],
    )

    assert env._snapshot_ready is True
    assert env.cwd == "/home/coder"
    assert len(connect_urls) == 1
    init_query = parse_qs(urlparse(connect_urls[0]).query)
    init_command = init_query["command"][0]
    assert init_command.startswith("bash -c ")
    assert "bash -lc" in init_command
    assert "export HERMES_CODER_SNAPSHOT_TEST=forwarded-value" in init_command
    assert "export -p" in init_command
    assert "declare -f" in init_command

    env.execute("printf $HERMES_CODER_SNAPSHOT_TEST")

    assert len(connect_urls) == 2
    followup_query = parse_qs(urlparse(connect_urls[1]).query)
    followup_command = followup_query["command"][0]
    assert f"source {env._snapshot_path}" in followup_command
    assert f"export -p > {env._snapshot_path}" in followup_command
    assert "printf $HERMES_CODER_SNAPSHOT_TEST" in followup_command


def test_coder_environment_leaves_snapshot_unready_when_init_session_fails(monkeypatch):
    workspace_payload = {
        "id": "workspace-123",
        "name": "shared-dev",
        "latest_build": {
            "transition": "start",
            "resources": [{"agents": [{"id": "agent-123", "status": "connected", "lifecycle_state": "ready"}]}],
        },
    }
    monkeypatch.setattr(
        "tools.environments.coder.requests.get",
        MagicMock(
            side_effect=[
                _FakeResponse({"workspaces": [workspace_payload]}),
                _FakeResponse(workspace_payload),
            ]
        ),
    )
    monkeypatch.setattr("tools.environments.coder.requests.post", MagicMock())
    monkeypatch.setattr(
        "tools.environments.coder.connect",
        MagicMock(return_value=_FakeWebSocket([b"init failed without exit marker\n"])),
    )
    monkeypatch.setattr(
        "tools.environments.coder.coder_workspace_name_for_task",
        lambda task_id, db=None: "shared-dev",
    )

    env = CoderEnvironment(
        base_url="https://coder.example",
        template_name="devcontainer",
        task_id="20260521_180000_ef3456",
        api_key="secret-token",
        workspace_name="shared-dev",
        timeout=5,
    )

    assert env._snapshot_ready is False


def test_coder_environment_uses_configured_workspace_without_session_derivation(monkeypatch):
    existing_workspace = {"id": "workspace-123", "name": "shared-dev"}
    requests_get = MagicMock(return_value=_FakeResponse({"workspaces": [existing_workspace]}))
    requests_post = MagicMock()

    monkeypatch.setattr("tools.environments.coder.requests.get", requests_get)
    monkeypatch.setattr("tools.environments.coder.requests.post", requests_post)
    monkeypatch.setattr(
        "tools.environments.coder.coder_workspace_name_for_task",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not derive session workspace")),
    )

    env = CoderEnvironment(
        base_url="https://coder.example",
        template_name="devcontainer",
        task_id="20260521_180000_ef3456",
        api_key="secret-token",
        workspace_name="shared-dev",
        timeout=5,
        init_session=False,
    )

    assert env.workspace == "shared-dev"
    assert env._ensure_workspace() == existing_workspace
    requests_get.assert_called_once_with(
        "https://coder.example/api/v2/workspaces",
        headers={"Coder-Session-Token": "secret-token"},
        params={"q": "owner:me name:shared-dev", "limit": 100},
        timeout=5,
    )
    requests_post.assert_not_called()


def test_coder_environment_execute_creates_workspace_then_reads_pty_until_eof(monkeypatch):
    workspace_created = {
        "id": "workspace-123",
        "name": "hermes-20260521-173045-ab12cd",
        "latest_build": {
            "transition": "start",
            "resources": [{"agents": [{"id": "agent-123", "status": "connected", "lifecycle_state": "ready"}]}],
        },
    }
    reconnect_id = uuid.UUID("12345678-1234-5678-1234-567812345678")
    exit_marker = f"__HERMES_EXIT_{reconnect_id}__"
    fake_ws = _FakeWebSocket([f"hello from coder\n\n{exit_marker}0{exit_marker}\n".encode()])
    connect_mock = MagicMock(return_value=fake_ws)
    requests_get = MagicMock(
        side_effect=[
            _FakeResponse({"workspaces": []}),
            _FakeResponse([{"id": "org-123", "is_default": True}]),
            _FakeResponse({"id": "template-uuid", "name": "devcontainer"}),
            _FakeResponse(workspace_created),
        ]
    )
    requests_post = MagicMock(return_value=_FakeResponse(workspace_created, status_code=201))

    monkeypatch.setattr("tools.environments.coder.requests.get", requests_get)
    monkeypatch.setattr("tools.environments.coder.requests.post", requests_post)
    monkeypatch.setattr("tools.environments.coder.connect", connect_mock)
    monkeypatch.setattr("tools.environments.coder.uuid.uuid4", lambda: reconnect_id)
    monkeypatch.setattr(
        "tools.environments.coder.coder_workspace_name_for_task",
        lambda task_id, db=None: "hermes-20260521-173045-ab12cd",
    )

    env = CoderEnvironment(
        base_url="https://coder.example",
        template_name="devcontainer",
        task_id="20260521_180000_ef3456",
        api_key="secret-token",
        cwd="/root",
        timeout=5,
        init_session=False,
    )

    result = env.execute("echo hello-from-hermes")

    assert result["returncode"] == 0
    assert result["output"] == "hello from coder\n"

    requests_post.assert_called_once_with(
        "https://coder.example/api/v2/organizations/org-123/members/me/workspaces",
        headers={
            "Coder-Session-Token": "secret-token",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json={"name": "hermes-20260521-173045-ab12cd", "template_id": "template-uuid"},
        timeout=5,
    )
    connect_kwargs = connect_mock.call_args.kwargs
    assert connect_kwargs["additional_headers"]["Coder-Session-Token"] == "secret-token"
    connect_url = connect_mock.call_args.args[0]
    assert "/api/v2/workspaceagents/agent-123/pty" in connect_url
    query = parse_qs(urlparse(connect_url).query)
    assert query["reconnect"] == [str(reconnect_id)]
    pty_command = query["command"][0]
    assert pty_command.startswith("bash -c ")
    assert f"{exit_marker}%s{exit_marker}" in pty_command
    assert "bash -lc" in pty_command
    assert "echo hello-from-hermes" in pty_command
    assert pty_command != "pwd"


def test_coder_environment_reconnects_same_pty_after_empty_initial_eof(monkeypatch):
    reconnect_id = uuid.UUID("22222222-3333-4444-5555-666666666666")
    exit_marker = f"__HERMES_EXIT_{reconnect_id}__"
    first_ws = _FakeWebSocket([])
    second_ws = _FakeWebSocket([f"hello after reconnect\n\n{exit_marker}0{exit_marker}\n".encode()])
    connect_mock = MagicMock(side_effect=[first_ws, second_ws])

    monkeypatch.setattr("tools.environments.coder.connect", connect_mock)
    monkeypatch.setattr("tools.environments.coder.uuid.uuid4", lambda: reconnect_id)
    monkeypatch.setattr("tools.environments.coder.time.sleep", lambda _seconds: None)
    monkeypatch.setattr(CoderEnvironment, "_resolve_agent_id", lambda self: "agent-123")
    monkeypatch.setattr(
        "tools.environments.coder.coder_workspace_name_for_task",
        lambda task_id, db=None: "hermes-20260521-173045-ab12cd",
    )

    env = CoderEnvironment(
        base_url="https://coder.example",
        template_name="devcontainer",
        task_id="20260521_180000_ef3456",
        api_key="secret-token",
        timeout=5,
        init_session=False,
    )

    result = env.execute("echo hello")

    assert result["returncode"] == 0
    assert result["output"] == "hello after reconnect\n"
    assert connect_mock.call_count == 2
    first_query = parse_qs(urlparse(connect_mock.call_args_list[0].args[0]).query)
    second_query = parse_qs(urlparse(connect_mock.call_args_list[1].args[0]).query)
    assert first_query["reconnect"] == [str(reconnect_id)]
    assert second_query["reconnect"] == [str(reconnect_id)]
    assert second_query["command"] == first_query["command"]


def test_coder_environment_reconnects_empty_eof_with_stdin_without_resending(monkeypatch):
    reconnect_id = uuid.UUID("22222222-3333-4444-5555-666666666666")
    exit_marker = f"__HERMES_EXIT_{reconnect_id}__"

    class _SendingWebSocket(_FakeWebSocket):
        def __init__(self, messages):
            super().__init__(messages)
            self.sent = []

        def send(self, message):
            self.sent.append(message)

    first_ws = _SendingWebSocket([])
    second_ws = _SendingWebSocket([f"after reconnect\n\n{exit_marker}0{exit_marker}\n".encode()])
    connect_mock = MagicMock(side_effect=[first_ws, second_ws])

    monkeypatch.setattr("tools.environments.coder.connect", connect_mock)
    monkeypatch.setattr("tools.environments.coder.uuid.uuid4", lambda: reconnect_id)
    monkeypatch.setattr(CoderEnvironment, "_resolve_agent_id", lambda self: "agent-123")
    monkeypatch.setattr(
        "tools.environments.coder.coder_workspace_name_for_task",
        lambda task_id, db=None: "hermes-20260521-173045-ab12cd",
    )

    env = CoderEnvironment(
        base_url="https://coder.example",
        template_name="devcontainer",
        task_id="20260521_180000_ef3456",
        api_key="secret-token",
        timeout=5,
        init_session=False,
    )

    result = env.execute("cat > /tmp/out.txt", stdin_data="hello stdin")

    assert result["returncode"] == 0
    assert result["output"] == "after reconnect\n"
    assert connect_mock.call_count == 2
    sent_payloads = [json.loads(frame.decode("utf-8")) for frame in first_ws.sent]
    assert sent_payloads == [{"data": "hello stdin"}, {"data": "\u0004"}]
    assert second_ws.sent == []
    first_query = parse_qs(urlparse(connect_mock.call_args_list[0].args[0]).query)
    second_query = parse_qs(urlparse(connect_mock.call_args_list[1].args[0]).query)
    assert first_query["reconnect"] == [str(reconnect_id)]
    assert second_query["reconnect"] == [str(reconnect_id)]
    assert second_query["command"] == first_query["command"]


def test_coder_environment_recv_timeout_poll_does_not_fail_silent_command(monkeypatch):
    reconnect_id = uuid.UUID("33333333-4444-5555-6666-777777777777")
    exit_marker = f"__HERMES_EXIT_{reconnect_id}__"
    fake_ws = _FakeWebSocket(
        [
            TimeoutError(),
            f"eventual output\n\n{exit_marker}0{exit_marker}\n".encode(),
        ]
    )
    connect_mock = MagicMock(return_value=fake_ws)

    monkeypatch.setattr("tools.environments.coder.connect", connect_mock)
    monkeypatch.setattr("tools.environments.coder.uuid.uuid4", lambda: reconnect_id)
    monkeypatch.setattr(CoderEnvironment, "_resolve_agent_id", lambda self: "agent-123")
    monkeypatch.setattr(
        "tools.environments.coder.coder_workspace_name_for_task",
        lambda task_id, db=None: "hermes-20260521-173045-ab12cd",
    )

    env = CoderEnvironment(
        base_url="https://coder.example",
        template_name="devcontainer",
        task_id="20260521_180000_ef3456",
        api_key="secret-token",
        timeout=5,
        init_session=False,
    )

    result = env.execute("sleep 2 && echo done", timeout=9)

    assert result["returncode"] == 0
    assert result["output"] == "eventual output\n"
    assert connect_mock.call_args.kwargs["open_timeout"] == 9
    assert fake_ws.requested[0]["timeout"] == 1.0
    assert fake_ws.requested[0]["decode"] is False


def test_coder_environment_stdin_data_uses_binary_json_frames_and_eof(monkeypatch):
    reconnect_id = uuid.UUID("11111111-2222-3333-4444-555555555555")
    exit_marker = f"__HERMES_EXIT_{reconnect_id}__"

    class _SendingWebSocket(_FakeWebSocket):
        def __init__(self, messages):
            super().__init__(messages)
            self.sent = []

        def send(self, message):
            self.sent.append(message)

    fake_ws = _SendingWebSocket([f"ok\n\n{exit_marker}0{exit_marker}\n".encode()])
    connect_mock = MagicMock(return_value=fake_ws)

    monkeypatch.setattr("tools.environments.coder.connect", connect_mock)
    monkeypatch.setattr("tools.environments.coder.uuid.uuid4", lambda: reconnect_id)
    monkeypatch.setattr(CoderEnvironment, "_resolve_agent_id", lambda self: "agent-123")
    monkeypatch.setattr(
        "tools.environments.coder.coder_workspace_name_for_task",
        lambda task_id, db=None: "hermes-20260521-173045-ab12cd",
    )

    env = CoderEnvironment(
        base_url="https://coder.example",
        template_name="devcontainer",
        task_id="20260521_180000_ef3456",
        api_key="secret-token",
        timeout=5,
        init_session=False,
    )

    result = env.execute("cat > /tmp/out.txt", stdin_data="hello stdin")

    assert result["returncode"] == 0
    sent_payloads = [json.loads(frame.decode("utf-8")) for frame in fake_ws.sent]
    assert sent_payloads == [{"data": "hello stdin"}, {"data": "\u0004"}]


def test_coder_environment_returns_nonzero_exit_code_from_pty_marker(monkeypatch):
    reconnect_id = uuid.UUID("87654321-4321-6789-4321-678987654321")
    exit_marker = f"__HERMES_EXIT_{reconnect_id}__"
    fake_ws = _FakeWebSocket([f"failure output\r\n{exit_marker}42{exit_marker}\r\n".encode()])
    connect_mock = MagicMock(return_value=fake_ws)

    monkeypatch.setattr("tools.environments.coder.connect", connect_mock)
    monkeypatch.setattr("tools.environments.coder.uuid.uuid4", lambda: reconnect_id)
    monkeypatch.setattr(CoderEnvironment, "_resolve_agent_id", lambda self: "agent-123")
    monkeypatch.setattr(
        "tools.environments.coder.coder_workspace_name_for_task",
        lambda task_id, db=None: "hermes-20260521-173045-ab12cd",
    )

    env = CoderEnvironment(
        base_url="https://coder.example",
        template_name="devcontainer",
        task_id="20260521_180000_ef3456",
        api_key="secret-token",
        timeout=5,
        init_session=False,
    )

    result = env.execute("exit 42")

    assert result["returncode"] == 42
    assert result["output"] == "failure output"
    connect_url = connect_mock.call_args.args[0]
    query = parse_qs(urlparse(connect_url).query)
    assert query["reconnect"] == [str(reconnect_id)]
    assert exit_marker in query["command"][0]


def test_coder_environment_missing_exit_marker_returns_backend_error(monkeypatch):
    fake_ws = _FakeWebSocket([b"plain output without marker\n"])
    monkeypatch.setattr("tools.environments.coder.connect", MagicMock(return_value=fake_ws))
    monkeypatch.setattr(CoderEnvironment, "_resolve_agent_id", lambda self: "agent-123")
    monkeypatch.setattr(
        "tools.environments.coder.coder_workspace_name_for_task",
        lambda task_id, db=None: "hermes-20260521-173045-ab12cd",
    )

    env = CoderEnvironment(
        base_url="https://coder.example",
        template_name="devcontainer",
        task_id="20260521_180000_ef3456",
        api_key="secret-token",
        timeout=5,
        init_session=False,
    )

    result = env.execute("echo no-marker")

    assert result["returncode"] == 1
    assert "plain output without marker" in result["output"]


def test_coder_process_kill_sends_ctrl_c_to_active_pty(monkeypatch):
    connected = threading.Event()
    closed = threading.Event()

    class _BlockingWebSocket:
        def __init__(self):
            self.sent = []
            self.closed = False

        def __enter__(self):
            connected.set()
            return self

        def __exit__(self, exc_type, exc, tb):
            self.close()
            return False

        def recv(self, timeout=None, decode=None):
            closed.wait(timeout=2)
            raise EOFError

        def send(self, message):
            self.sent.append(message)

        def close(self):
            self.closed = True
            closed.set()

    fake_ws = _BlockingWebSocket()
    monkeypatch.setattr("tools.environments.coder.connect", MagicMock(return_value=fake_ws))
    monkeypatch.setattr(CoderEnvironment, "_resolve_agent_id", lambda self: "agent-123")
    monkeypatch.setattr(
        "tools.environments.coder.coder_workspace_name_for_task",
        lambda task_id, db=None: "hermes-20260521-173045-ab12cd",
    )

    env = CoderEnvironment(
        base_url="https://coder.example",
        template_name="devcontainer",
        task_id="20260521_180000_ef3456",
        api_key="secret-token",
        timeout=5,
        init_session=False,
    )

    handle = env._run_bash("sleep 999", timeout=5)
    assert connected.wait(timeout=2)

    handle.kill()
    handle.wait(timeout=2)

    assert fake_ws.sent == [CoderEnvironment._stdin_frame("\u0003")]
    assert fake_ws.closed is True


def test_coder_environment_create_workspace_uses_configured_workspace_and_organization(monkeypatch):
    workspace_created = {"id": "workspace-123", "name": "shared-dev"}
    requests_get = MagicMock(
        side_effect=[
            _FakeResponse({"workspaces": []}),
            _FakeResponse({"id": "org-123", "name": "acme"}),
            _FakeResponse({"id": "template-uuid", "name": "devcontainer"}),
        ]
    )
    requests_post = MagicMock(return_value=_FakeResponse(workspace_created, status_code=201))

    monkeypatch.setattr("tools.environments.coder.requests.get", requests_get)
    monkeypatch.setattr("tools.environments.coder.requests.post", requests_post)
    monkeypatch.setattr(
        "tools.environments.coder.coder_workspace_name_for_task",
        lambda task_id, db=None: "hermes-20260521-173045-ab12cd",
    )

    env = CoderEnvironment(
        base_url="https://coder.example",
        template_name="devcontainer",
        task_id="20260521_180000_ef3456",
        api_key="secret-token",
        organization_name="acme",
        workspace_name="shared-dev",
        timeout=5,
        init_session=False,
    )

    assert env._ensure_workspace() == workspace_created
    called_urls = [call.args[0] for call in requests_get.call_args_list]
    assert "https://coder.example/api/v2/users/me/organizations/acme" in called_urls
    assert "https://coder.example/api/v2/organizations/org-123/templates/devcontainer" in called_urls
    requests_post.assert_called_once_with(
        "https://coder.example/api/v2/organizations/org-123/members/me/workspaces",
        headers={
            "Coder-Session-Token": "secret-token",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json={"name": "shared-dev", "template_id": "template-uuid"},
        timeout=5,
    )


def test_coder_environment_autostarts_existing_stopped_workspace(monkeypatch):
    existing_workspace = {
        "id": "workspace-123",
        "name": "hermes-20260521-173045-ab12cd",
        "latest_build": {
            "transition": "stop",
            "status": "stopped",
            "resources": [],
        },
    }
    started_workspace = {
        "id": "workspace-123",
        "name": "hermes-20260521-173045-ab12cd",
        "latest_build": {
            "transition": "start",
            "resources": [{"agents": [{"id": "agent-123", "status": "connected", "lifecycle_state": "ready"}]}],
        },
    }
    requests_get = MagicMock(
        side_effect=[
            _FakeResponse({"workspaces": [existing_workspace]}),
            _FakeResponse(existing_workspace),
            _FakeResponse({"job": {"status": "succeeded", "completed_at": "2026-05-19T10:10:00Z"}}),
            _FakeResponse({"workspaces": [started_workspace]}),
            _FakeResponse(started_workspace),
        ]
    )
    requests_post = MagicMock(return_value=_FakeResponse({"id": "build-123"}, status_code=201))

    monkeypatch.setattr("tools.environments.coder.requests.get", requests_get)
    monkeypatch.setattr("tools.environments.coder.requests.post", requests_post)
    monkeypatch.setattr("tools.environments.coder.time.sleep", lambda _seconds: None)
    monkeypatch.setattr(
        "tools.environments.coder.coder_workspace_name_for_task",
        lambda task_id, db=None: "hermes-20260521-173045-ab12cd",
    )

    env = CoderEnvironment(
        base_url="https://coder.example",
        template_name="devcontainer",
        task_id="20260521_180000_ef3456",
        api_key="secret-token",
        init_session=False,
    )

    assert env._resolve_agent_id() == "agent-123"
    requests_post.assert_called_once_with(
        "https://coder.example/api/v2/workspaces/workspace-123/builds",
        headers={"Coder-Session-Token": "secret-token"},
        json={"transition": "start"},
        timeout=60,
    )


def test_resolve_agent_id_waits_for_agent_connected_and_ready_before_returning(monkeypatch):
    base_workspace = {
        "id": "workspace-123",
        "name": "hermes-20260521-173045-ab12cd",
    }
    not_ready_workspace = {
        **base_workspace,
        "latest_build": {
            "transition": "start",
            "resources": [
                {
                    "agents": [
                        {
                            "id": "agent-123",
                            "status": "starting",
                            "lifecycle_state": "created",
                        }
                    ]
                }
            ],
        },
    }
    ready_workspace = {
        **base_workspace,
        "latest_build": {
            "transition": "start",
            "resources": [
                {
                    "agents": [
                        {
                            "id": "agent-123",
                            "status": "connected",
                            "lifecycle_state": "ready",
                        }
                    ]
                }
            ],
        },
    }

    requests_get = MagicMock(
        side_effect=[
            _FakeResponse({"workspaces": [base_workspace]}),
            _FakeResponse(not_ready_workspace),
            _FakeResponse({"workspaces": [base_workspace]}),
            _FakeResponse(ready_workspace),
        ]
    )

    monkeypatch.setattr("tools.environments.coder.requests.get", requests_get)
    monkeypatch.setattr("tools.environments.coder.requests.post", MagicMock())
    monkeypatch.setattr("tools.environments.coder.time.sleep", lambda _seconds: None)
    monkeypatch.setattr(
        "tools.environments.coder.coder_workspace_name_for_task",
        lambda task_id, db=None: "hermes-20260521-173045-ab12cd",
    )

    env = CoderEnvironment(
        base_url="https://coder.example",
        template_name="devcontainer",
        task_id="20260521_180000_ef3456",
        api_key="secret-token",
        timeout=5,
        init_session=False,
    )

    assert env._resolve_agent_id() == "agent-123"
    assert requests_get.call_count == 4


def test_terminal_tool_passes_coder_config_into_environment_factory(monkeypatch):
    class _FakeEnv:
        def execute(self, command, timeout=None, cwd=None, pty=False, **kwargs):
            assert command == "printf 'hi from coder'"
            return {"output": "hi from coder", "returncode": 0}

    create_env = MagicMock(return_value=_FakeEnv())
    monkeypatch.setattr(terminal_tool_module, "_create_environment", create_env)
    monkeypatch.setattr(terminal_tool_module, "_check_all_guards", lambda *_args, **_kwargs: {"approved": True})
    monkeypatch.setattr(terminal_tool_module, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(terminal_tool_module, "_active_environments", {})
    monkeypatch.setattr(terminal_tool_module, "_last_activity", {})
    monkeypatch.setattr(terminal_tool_module, "_creation_locks", {})
    monkeypatch.setattr(terminal_tool_module, "_task_env_overrides", {})
    monkeypatch.setenv("TERMINAL_ENV", "coder")
    monkeypatch.setenv("CODER_URL", "https://coder.example")
    monkeypatch.setenv("CODER_API_KEY", "secret-token")
    monkeypatch.setenv("CODER_ORGANIZATION", "acme")
    monkeypatch.setenv("CODER_WORKSPACE", "shared-dev")
    monkeypatch.setenv("CODER_TEMPLATE", "devcontainer")
    monkeypatch.setenv("TERMINAL_CODER_FORWARD_ENV", '["GITHUB_TOKEN"]')
    monkeypatch.setenv("TERMINAL_CODER_WORKSPACE_STARTUP_TIMEOUT", "240")

    payload = json.loads(terminal_tool_module.terminal_tool(command="printf 'hi from coder'", task_id="coder-test"))

    assert payload["exit_code"] == 0
    create_env.assert_called_once()
    assert create_env.call_args.kwargs["container_config"]["coder_url"] == "https://coder.example"
    assert create_env.call_args.kwargs["container_config"]["coder_api_key"] == "secret-token"
    assert create_env.call_args.kwargs["container_config"]["coder_organization"] == "acme"
    assert create_env.call_args.kwargs["container_config"]["coder_workspace"] == "shared-dev"
    assert create_env.call_args.kwargs["container_config"]["coder_template"] == "devcontainer"
    assert create_env.call_args.kwargs["container_config"]["coder_forward_env"] == ["GITHUB_TOKEN"]
    assert create_env.call_args.kwargs["container_config"]["coder_workspace_startup_timeout"] == 240
