from __future__ import annotations

import sys
import threading
import time
import types
from types import SimpleNamespace

import pytest


class _FakeFS:
    def __init__(self):
        self.mkdir_calls: list[tuple[tuple, dict]] = []
        self.upload_calls: list[tuple[str, str]] = []
        self.download_calls: list[tuple[str, str]] = []

    @staticmethod
    def _assert_remote_path(path: str) -> None:
        if path.startswith("/") and path != "/home/tenki" and not path.startswith("/home/tenki/"):
            raise AssertionError(f"Tenki fs path must be under /home/tenki, got {path!r}")

    def mkdir(self, path, **kwargs):
        self._assert_remote_path(str(path))
        self.mkdir_calls.append(((path,), kwargs))

    def upload(self, local_path, remote_path, **_kwargs):
        self._assert_remote_path(str(remote_path))
        self.upload_calls.append((str(local_path), str(remote_path)))

    def download(self, remote_path, local_path, **_kwargs):
        self._assert_remote_path(str(remote_path))
        self.download_calls.append((str(remote_path), str(local_path)))


class _FakeResult:
    def __init__(self, stdout: str = "", stderr: str = "", exit_code: int = 0):
        self.stdout_text = stdout
        self.stderr_text = stderr
        self.exit_code = exit_code


class _FakeProcess:
    def __init__(
        self,
        result: _FakeResult,
        *,
        stdin_data: str | None = None,
        block_until_killed: bool = False,
    ):
        self._result = result
        self.stdin_data = stdin_data
        self.closed_stdin = False
        self.killed = False
        self._block_until_killed = block_until_killed
        self._done = threading.Event()

    def close_stdin(self):
        self.closed_stdin = True

    def kill(self):
        self.killed = True
        self._done.set()

    def wait(self, *_args, **_kwargs):
        if self._block_until_killed:
            self._done.wait(timeout=5)
            return _FakeResult(stdout="", exit_code=143)
        return self._result


class _FakeSandbox:
    def __init__(
        self,
        *,
        name: str = "sb-test",
        state: str = "RUNNING",
        metadata: dict | None = None,
    ):
        self.exec_calls: list[tuple[tuple, dict]] = []
        self.start_calls: list[tuple[tuple, dict]] = []
        self.last_process: _FakeProcess | None = None
        self.snapshots: list[tuple[str | None, bool]] = []
        self.terminated = False
        self.paused = False
        self.resumed = False
        self.waited = False
        self.refreshed = False
        self.id = "sb-test"
        self.name = name
        self.state = state
        self.info = SimpleNamespace(name=name, metadata=metadata or {})
        self.fs = _FakeFS()

    @staticmethod
    def _result_for_command(args):
        command = args[-1] if args else ""
        if "echo \"$HOME\"" in command:
            return _FakeResult(stdout="/home/tenki\n")
        return _FakeResult(stdout="ran\n", exit_code=0)

    def exec(self, *args, **kwargs):
        self.exec_calls.append((args, kwargs))
        return self._result_for_command(args)

    def start(self, *args, **kwargs):
        self.start_calls.append((args, kwargs))
        command = args[-1] if args else ""
        self.last_process = _FakeProcess(
            self._result_for_command(args),
            stdin_data=kwargs.get("stdin"),
            block_until_killed="sleep infinity" in command,
        )
        return self.last_process

    def refresh(self):
        self.refreshed = True
        return self.info

    def terminate(self):
        self.terminated = True
        self.state = "TERMINATED"

    def pause(self):
        self.paused = True
        self.state = "PAUSED"

    def resume(self):
        self.resumed = True
        self.state = "RUNNING"

    def wait_ready(self, *_args, **_kwargs):
        self.waited = True

    def snapshot(self, *, name=None, wait=True):
        self.snapshots.append((name, wait))
        return SimpleNamespace(id=f"snap-{self.name}")


def _last_started_command(sandbox: _FakeSandbox) -> str:
    return sandbox.start_calls[-1][0][-1]


class _FakeSnapshotNotFoundError(Exception):
    """Mirrors tenki_sandbox.SnapshotNotFoundError for the fake SDK."""


class _FakeRegistryArtifactNotFoundError(Exception):
    """Mirrors tenki_sandbox.RegistryArtifactNotFoundError for the fake SDK."""


class _FakeSnapshotNotDurableError(Exception):
    """Mirrors tenki_sandbox.SnapshotNotDurableError for the fake SDK."""


class _FakeInvalidStateError(Exception):
    """Mirrors tenki_sandbox.InvalidStateError for the fake SDK."""


class _FakeSandboxFactory:
    created_kwargs: list[dict] = []
    failed_kwargs: list[dict] = []
    sandboxes: list[_FakeSandbox] = []
    fail_snapshot_ids: set[str] = set()
    # When a snapshot id is in fail_snapshot_ids, raise this exception type with
    # this message. Defaults to the confirmed-not-found error; tests set them to
    # a transient error / generic message to prove the pointer is preserved, or
    # to a snapshot-specific InvalidStateError to prove base-image fallback.
    snapshot_error: type[Exception] = _FakeSnapshotNotFoundError
    snapshot_error_msg: str = "restore failed"

    @classmethod
    def create(cls, **kwargs):
        if kwargs.get("snapshot_id") in cls.fail_snapshot_ids:
            cls.failed_kwargs.append(kwargs)
            raise cls.snapshot_error(cls.snapshot_error_msg)
        sandbox = _FakeSandbox(
            name=kwargs.get("name", "sb-test"),
            metadata=kwargs.get("metadata", {}),
        )
        cls.created_kwargs.append(kwargs)
        cls.sandboxes.append(sandbox)
        return sandbox


class _FakeClient:
    listed_sandboxes: list[_FakeSandbox] = []
    closed_count = 0

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.snapshots = SimpleNamespace(wait_durable=lambda *_args, **_kwargs: None)

    def create(self, **kwargs):
        return _FakeSandboxFactory.create(**kwargs)

    def list(self, **_kwargs):
        return list(self.listed_sandboxes)

    def list_project(self, *_args, **_kwargs):
        return list(self.listed_sandboxes)

    def list_workspace(self, *_args, **_kwargs):
        return list(self.listed_sandboxes)

    def close(self):
        type(self).closed_count += 1


def _install_fake_tenki(monkeypatch):
    module = types.ModuleType("tenki_sandbox")
    _FakeSandboxFactory.created_kwargs = []
    _FakeSandboxFactory.failed_kwargs = []
    _FakeSandboxFactory.sandboxes = []
    _FakeSandboxFactory.fail_snapshot_ids = set()
    _FakeSandboxFactory.snapshot_error = _FakeSnapshotNotFoundError
    _FakeSandboxFactory.snapshot_error_msg = "restore failed"
    _FakeClient.listed_sandboxes = []
    _FakeClient.closed_count = 0
    module.Client = _FakeClient
    module.Sandbox = _FakeSandboxFactory
    module.SnapshotNotFoundError = _FakeSnapshotNotFoundError
    module.RegistryArtifactNotFoundError = _FakeRegistryArtifactNotFoundError
    module.SnapshotNotDurableError = _FakeSnapshotNotDurableError
    module.InvalidStateError = _FakeInvalidStateError
    monkeypatch.setitem(sys.modules, "tenki_sandbox", module)


def _clear_tenki_auth_env(monkeypatch):
    monkeypatch.delenv("TENKI_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("TENKI_API_KEY", raising=False)


def _clear_env_passthrough_cache():
    try:
        import tools.env_passthrough as env_passthrough

        env_passthrough.clear_env_passthrough()
        env_passthrough._config_passthrough = None
    except Exception:
        pass


def test_tenki_cli_auth_token_is_normalized_for_sdk_cookie_auth(monkeypatch, tmp_path):
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text("auth_token: cli-cookie\n", encoding="utf-8")

    from tools.tenki_config import resolve_tenki_auth_token

    assert resolve_tenki_auth_token() == "cookie:cli-cookie"


def test_tenki_cli_auth_token_preserves_sdk_prefixes(monkeypatch, tmp_path):
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))

    from tools.tenki_config import resolve_tenki_auth_token

    for token in ("cookie:cli-cookie", "ory_st_session", "sk-api-key"):
        (tmp_path / "config.yaml").write_text(f"auth_token: {token}\n", encoding="utf-8")
        assert resolve_tenki_auth_token() == token


def test_tenki_cli_api_key_is_not_treated_as_cookie(monkeypatch, tmp_path):
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text("api_key: provider-key\n", encoding="utf-8")

    from tools.tenki_config import resolve_tenki_auth_token

    assert resolve_tenki_auth_token() == "provider-key"


def test_tenki_environment_uses_cli_config_and_terminates_by_default(monkeypatch, tmp_path):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text(
        "\n".join(
            [
                "api_endpoint: https://api.tenki.test",
                "current_workspace_id: ws-123",
                "current_project_id: prj-456",
                "auth_token: tok-secret",
            ]
        ),
        encoding="utf-8",
    )

    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)

    env = TenkiEnvironment(
        image="",
        task_id="session 1",
        persistent_filesystem=False,
        allow_inbound=False,
        allow_outbound=True,
    )

    kwargs = _FakeSandboxFactory.created_kwargs[0]
    # The control-plane credential is used host-side to create the sandbox...
    assert kwargs["base_url"] == "https://api.tenki.test"
    assert kwargs["workspace_id"] == "ws-123"
    assert kwargs["project_id"] == "prj-456"
    assert kwargs["auth_token"] == "cookie:tok-secret"
    # ...but is NEVER injected into the model-controlled guest environment
    # (an empty env is omitted from the create kwargs entirely).
    guest_env = kwargs.get("env", {})
    assert "TENKI_AUTH_TOKEN" not in guest_env
    assert "TENKI_API_KEY" not in guest_env
    assert "TENKI_API_ENDPOINT" not in guest_env
    assert "TENKI_WORKSPACE_ID" not in guest_env
    assert "TENKI_PROJECT_ID" not in guest_env
    assert kwargs["allow_inbound"] is False
    assert kwargs["allow_outbound"] is True
    assert kwargs["cpu_cores"] == 1
    assert "idle_timeout" not in kwargs
    assert "idle_timeout_minutes" not in kwargs
    assert "pause_retention" not in kwargs
    assert kwargs["metadata"]["hermes_backend"] == "tenki"
    assert kwargs["metadata"]["hermes_profile"]
    assert kwargs["name"].startswith("hermes-")
    assert kwargs["name"].endswith("session-1")

    output, exit_code = env._exec_raw("echo ok", timeout=5)
    assert output == "ran\n"
    assert exit_code == 0

    sandbox = _FakeSandboxFactory.sandboxes[0]
    assert "TENKI_AUTH_TOKEN" not in sandbox.exec_calls[-1][1]["env"]
    env.cleanup()
    assert sandbox.terminated is True
    assert sandbox.paused is False


def test_tenki_environment_does_not_inject_control_plane_token_by_default(monkeypatch, tmp_path):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    _clear_env_passthrough_cache()
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "missing.yaml"))
    monkeypatch.setenv("TENKI_API_KEY", "sk-test-key")

    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    env = TenkiEnvironment(task_id="api-key")

    kwargs = _FakeSandboxFactory.created_kwargs[0]
    # Host-side create still authenticates with the credential...
    assert kwargs["auth_token"] == "sk-test-key"
    # ...but the guest never receives it unless explicitly forwarded.
    guest_env = kwargs.get("env", {})
    assert "TENKI_AUTH_TOKEN" not in guest_env
    assert "TENKI_API_KEY" not in guest_env
    env.cleanup()
    _clear_env_passthrough_cache()


def test_tenki_environment_forwards_control_plane_token_only_when_opted_in(monkeypatch, tmp_path):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    _clear_env_passthrough_cache()
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "missing.yaml"))
    monkeypatch.setenv("TENKI_API_KEY", "sk-test-key")

    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    # Nested-sandbox support: the operator explicitly forwards the credential.
    env = TenkiEnvironment(task_id="api-key", forward_env=["TENKI_API_KEY"])

    kwargs = _FakeSandboxFactory.created_kwargs[0]
    assert kwargs["env"]["TENKI_API_KEY"] == "sk-test-key"
    env.cleanup()
    _clear_env_passthrough_cache()


def test_tenki_environment_honors_tenki_forward_env_from_process_env(monkeypatch, tmp_path):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    _clear_env_passthrough_cache()
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    monkeypatch.setenv("GH_TOKEN", "gho-process")
    (tmp_path / "config.yaml").write_text(
        "auth_token: tok-secret\n",
        encoding="utf-8",
    )

    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    env = TenkiEnvironment(task_id="gh-token", forward_env=["GH_TOKEN"])

    assert _FakeSandboxFactory.created_kwargs[0]["env"]["GH_TOKEN"] == "gho-process"
    env.execute("echo ok", timeout=5)
    assert env._sandbox.start_calls[-1][1]["env"]["GH_TOKEN"] == "gho-process"
    env.cleanup()
    _clear_env_passthrough_cache()


def test_tenki_environment_honors_tenki_forward_env_from_hermes_dotenv(monkeypatch, tmp_path):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    _clear_env_passthrough_cache()
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    (tmp_path / ".env").write_text("GITHUB_TOKEN=ghp-dotenv\n", encoding="utf-8")
    (tmp_path / "config.yaml").write_text(
        "auth_token: tok-secret\n",
        encoding="utf-8",
    )

    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    env = TenkiEnvironment(task_id="github-token", forward_env=["GITHUB_TOKEN"])

    assert _FakeSandboxFactory.created_kwargs[0]["env"]["GITHUB_TOKEN"] == "ghp-dotenv"
    env.execute("echo ok", timeout=5)
    assert env._sandbox.start_calls[-1][1]["env"]["GITHUB_TOKEN"] == "ghp-dotenv"
    env.cleanup()
    _clear_env_passthrough_cache()


def test_tenki_forwarded_env_prefers_profile_scope_over_process_env(monkeypatch, tmp_path):
    """Under a multiplexed profile scope, a forwarded credential must resolve
    to the active profile's value, never another profile's raw os.environ."""
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    _clear_env_passthrough_cache()
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text("auth_token: tok-secret\n", encoding="utf-8")
    # Another profile's value leaking through the process environment...
    monkeypatch.setenv("GH_TOKEN", "gho-other-profile")

    from agent import secret_scope
    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    token = secret_scope.set_secret_scope({"GH_TOKEN": "gho-this-profile"})
    try:
        monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
        env = TenkiEnvironment(task_id="scoped", forward_env=["GH_TOKEN"])
        # ...must be overridden by the active profile scope.
        assert _FakeSandboxFactory.created_kwargs[0]["env"]["GH_TOKEN"] == "gho-this-profile"
        env.cleanup()
    finally:
        secret_scope.reset_secret_scope(token)
    _clear_env_passthrough_cache()


def test_tenki_auth_token_prefers_profile_scope_over_process_env(monkeypatch, tmp_path):
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "missing.yaml"))
    monkeypatch.setenv("TENKI_AUTH_TOKEN", "tok-other-profile")

    from agent import secret_scope
    from tools.tenki_config import resolve_tenki_auth_token

    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    tok = secret_scope.set_secret_scope({"TENKI_AUTH_TOKEN": "tok-this-profile"})
    try:
        assert resolve_tenki_auth_token() == "tok-this-profile"
    finally:
        secret_scope.reset_secret_scope(tok)


def test_tenki_auth_token_fails_closed_when_multiplex_active_and_unscoped(monkeypatch, tmp_path):
    """Multiplex on + no scope installed: an os.environ token must NOT leak
    through (fail closed), rather than serving another profile's value."""
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "missing.yaml"))
    monkeypatch.setenv("TENKI_AUTH_TOKEN", "tok-leaked-from-process-env")

    from agent import secret_scope
    from tools.tenki_config import resolve_tenki_auth_token

    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    # No set_secret_scope() — this is the fail-closed branch.
    assert secret_scope.current_secret_scope() is None
    assert resolve_tenki_auth_token() == ""


def test_tenki_forwarded_env_fails_closed_when_multiplex_active_and_unscoped(monkeypatch, tmp_path):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    _clear_env_passthrough_cache()
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text("auth_token: tok-secret\n", encoding="utf-8")
    monkeypatch.setenv("GH_TOKEN", "gho-leaked-from-process-env")

    from agent import secret_scope
    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    env = TenkiEnvironment(task_id="scoped", forward_env=["GH_TOKEN"])

    # No scope installed while multiplexing → the process-env value is not leaked.
    assert "GH_TOKEN" not in _FakeSandboxFactory.created_kwargs[0].get("env", {})
    env.cleanup()
    _clear_env_passthrough_cache()


def test_tenki_control_plane_token_forwarded_from_cli_config_when_opted_in(monkeypatch, tmp_path):
    """The opt-in must work even when auth came from `tenki login` (CLI config),
    whose secret never lands in os.environ."""
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    _clear_env_passthrough_cache()
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    # Credential lives ONLY in the Tenki CLI config, not the environment.
    (tmp_path / "config.yaml").write_text("auth_token: tok-cli-login\n", encoding="utf-8")

    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    env = TenkiEnvironment(task_id="nested", forward_env=["TENKI_AUTH_TOKEN"])

    guest_env = _FakeSandboxFactory.created_kwargs[0].get("env", {})
    assert guest_env["TENKI_AUTH_TOKEN"] == "cookie:tok-cli-login"
    env.cleanup()
    _clear_env_passthrough_cache()


def test_tenki_environment_honors_safe_env_passthrough(monkeypatch, tmp_path):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    _clear_env_passthrough_cache()
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    monkeypatch.setenv("CUSTOM_TASK_ENV", "task-value")
    (tmp_path / "config.yaml").write_text(
        "auth_token: tok-secret\n"
        "terminal:\n"
        "  env_passthrough:\n"
        "    - CUSTOM_TASK_ENV\n",
        encoding="utf-8",
    )

    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    env = TenkiEnvironment(task_id="safe-passthrough")

    assert _FakeSandboxFactory.created_kwargs[0]["env"]["CUSTOM_TASK_ENV"] == "task-value"
    env.execute("echo ok", timeout=5)
    assert env._sandbox.start_calls[-1][1]["env"]["CUSTOM_TASK_ENV"] == "task-value"
    env.cleanup()
    _clear_env_passthrough_cache()


def test_tenki_environment_snapshots_when_persistent(monkeypatch, tmp_path):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text("auth_token: tok-secret\n", encoding="utf-8")

    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    env = TenkiEnvironment(task_id="persist", persistent_filesystem=True)

    sandbox = _FakeSandboxFactory.sandboxes[0]
    env.cleanup()
    assert len(sandbox.snapshots) == 1
    snap_name, snap_wait = sandbox.snapshots[0]
    assert snap_name.endswith("persist") and snap_wait is True
    assert sandbox.paused is False
    assert sandbox.terminated is True

    env = TenkiEnvironment(task_id="persist", image="base-image", persistent_filesystem=True)
    assert _FakeSandboxFactory.created_kwargs[-1]["snapshot_id"].endswith("persist")
    assert "image" not in _FakeSandboxFactory.created_kwargs[-1]
    env.cleanup()


def test_tenki_environment_falls_back_when_persistent_snapshot_is_stale(monkeypatch, tmp_path):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text("auth_token: tok-secret\n", encoding="utf-8")

    from tools.environments import tenki as tenki_module
    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    tenki_module._store_snapshot("persist", "snap-stale")
    _FakeSandboxFactory.fail_snapshot_ids = {"snap-stale"}
    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)

    env = TenkiEnvironment(task_id="persist", image="base-image", persistent_filesystem=True)

    assert _FakeSandboxFactory.failed_kwargs[0]["snapshot_id"] == "snap-stale"
    assert _FakeSandboxFactory.created_kwargs[0]["image"] == "base-image"
    assert tenki_module._get_snapshot_restore_candidate("persist") == (None, False)
    env.cleanup()


def test_tenki_environment_preserves_snapshot_on_transient_restore_error(monkeypatch, tmp_path):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text("auth_token: tok-secret\n", encoding="utf-8")

    from tools.environments import tenki as tenki_module
    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    tenki_module._store_snapshot("persist", "snap-transient")
    _FakeSandboxFactory.fail_snapshot_ids = {"snap-transient"}
    # A transient failure (not a confirmed not-found) must NOT boot a blank
    # base image or drop the recovery pointer.
    _FakeSandboxFactory.snapshot_error = RuntimeError
    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)

    with pytest.raises(RuntimeError):
        TenkiEnvironment(task_id="persist", image="base-image", persistent_filesystem=True)

    # No base-image fallback happened, and the snapshot pointer is retained.
    assert _FakeSandboxFactory.created_kwargs == []
    assert tenki_module._get_snapshot_restore_candidate("persist") == ("snap-transient", False)


def test_tenki_environment_skips_snapshot_when_not_durable(monkeypatch, tmp_path):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text("auth_token: tok-secret\n", encoding="utf-8")

    from tools.environments import tenki as tenki_module
    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    def _fail_durable(*_args, **_kwargs):
        raise RuntimeError("not durable yet")

    def _init(self, **kw):
        self.kwargs = kw
        self.snapshots = SimpleNamespace(wait_durable=_fail_durable)

    monkeypatch.setattr(_FakeClient, "__init__", _init)
    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    env = TenkiEnvironment(task_id="persist", persistent_filesystem=True)
    sandbox = _FakeSandboxFactory.sandboxes[0]

    env.cleanup()

    # Durability failed → do NOT record the snapshot and do NOT terminate the
    # live sandbox; pause it so state is preserved for recovery.
    assert sandbox.paused is True
    assert sandbox.terminated is False
    assert tenki_module._get_snapshot_restore_candidate("persist") == (None, False)


def test_tenki_environment_resumes_existing_persistent_sandbox(monkeypatch, tmp_path):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text("auth_token: tok-secret\n", encoding="utf-8")

    from tools.environments import tenki as tenki_module
    from tools.environments.tenki import TenkiEnvironment

    token = tenki_module._profile_token()
    existing = _FakeSandbox(
        name=f"hermes-{token}-persist",
        state="PAUSED",
        metadata={"hermes_task_id": "persist", "hermes_profile": token},
    )
    _FakeClient.listed_sandboxes = [existing]

    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    env = TenkiEnvironment(task_id="persist", persistent_filesystem=True)

    assert env._sandbox is existing
    assert existing.resumed is True
    assert existing.waited is True
    assert _FakeSandboxFactory.created_kwargs == []
    # The resumed guest never receives the control-plane credential either.
    assert "TENKI_AUTH_TOKEN" not in existing.exec_calls[-1][1]["env"]
    env.cleanup()


def test_tenki_environment_does_not_reuse_other_profiles_sandbox(monkeypatch, tmp_path):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text("auth_token: tok-secret\n", encoding="utf-8")

    from tools.environments.tenki import TenkiEnvironment

    # A live sandbox on the same Tenki account belonging to a DIFFERENT profile
    # (foreign token) with the same task id must never be resumed.
    foreign = _FakeSandbox(
        name="hermes-deadbeef00-persist",
        state="PAUSED",
        metadata={"hermes_task_id": "persist", "hermes_profile": "deadbeef00"},
    )
    _FakeClient.listed_sandboxes = [foreign]

    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    env = TenkiEnvironment(task_id="persist", persistent_filesystem=True)

    assert env._sandbox is not foreign
    assert foreign.resumed is False
    assert _FakeSandboxFactory.created_kwargs, "should create its own sandbox"
    env.cleanup()


def test_tenki_reuse_rejects_name_match_with_foreign_profile_metadata(monkeypatch, tmp_path):
    """Defense-in-depth: even if a candidate's NAME matches, a differing
    hermes_profile in metadata must block reuse."""
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text("auth_token: tok-secret\n", encoding="utf-8")

    from tools.environments import tenki as tenki_module
    from tools.environments.tenki import TenkiEnvironment

    token = tenki_module._profile_token()
    # Same name (as if a token collision), but metadata says a different profile.
    collider = _FakeSandbox(
        name=f"hermes-{token}-persist",
        state="PAUSED",
        metadata={"hermes_task_id": "persist", "hermes_profile": "foreign-token"},
    )
    _FakeClient.listed_sandboxes = [collider]

    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    env = TenkiEnvironment(task_id="persist", persistent_filesystem=True)

    assert env._sandbox is not collider
    assert collider.resumed is False
    env.cleanup()


def test_tenki_restore_falls_back_on_nondurable_snapshot(monkeypatch, tmp_path):
    """A snapshot that EXISTS but is permanently unusable (non-durable) must
    drop the pointer and boot the base image, not wedge the task forever."""
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text("auth_token: tok-secret\n", encoding="utf-8")

    from tools.environments import tenki as tenki_module
    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    tenki_module._store_snapshot("persist", "snap-nondurable")
    _FakeSandboxFactory.fail_snapshot_ids = {"snap-nondurable"}
    _FakeSandboxFactory.snapshot_error = _FakeSnapshotNotDurableError
    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)

    env = TenkiEnvironment(task_id="persist", image="base-image", persistent_filesystem=True)

    assert _FakeSandboxFactory.failed_kwargs[0]["snapshot_id"] == "snap-nondurable"
    assert _FakeSandboxFactory.created_kwargs[0]["image"] == "base-image"
    assert tenki_module._get_snapshot_restore_candidate("persist") == (None, False)
    env.cleanup()


def test_tenki_restore_preserves_pointer_on_invalid_state_error(monkeypatch, tmp_path):
    """InvalidStateError is a generic precondition failure, NOT snapshot-gone,
    so it must be treated as transient: preserve the pointer, do not base-boot."""
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text("auth_token: tok-secret\n", encoding="utf-8")

    from tools.environments import tenki as tenki_module
    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    tenki_module._store_snapshot("persist", "snap-invalidstate")
    _FakeSandboxFactory.fail_snapshot_ids = {"snap-invalidstate"}
    _FakeSandboxFactory.snapshot_error = _FakeInvalidStateError
    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)

    with pytest.raises(_FakeInvalidStateError):
        TenkiEnvironment(task_id="persist", image="base-image", persistent_filesystem=True)

    assert _FakeSandboxFactory.created_kwargs == []
    assert tenki_module._get_snapshot_restore_candidate("persist") == ("snap-invalidstate", False)


def test_tenki_restore_falls_back_on_snapshot_specific_invalid_state(monkeypatch, tmp_path):
    """A generic InvalidStateError whose message identifies the snapshot (the
    SDK's collapsed representation of a bad/non-durable snapshot on restore)
    IS unrecoverable → drop the pointer and boot the base image."""
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text("auth_token: tok-secret\n", encoding="utf-8")

    from tools.environments import tenki as tenki_module
    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    tenki_module._store_snapshot("persist", "snap-badstate")
    _FakeSandboxFactory.fail_snapshot_ids = {"snap-badstate"}
    _FakeSandboxFactory.snapshot_error = _FakeInvalidStateError
    _FakeSandboxFactory.snapshot_error_msg = "snapshot is not durable"
    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)

    env = TenkiEnvironment(task_id="persist", image="base-image", persistent_filesystem=True)

    assert _FakeSandboxFactory.created_kwargs[0]["image"] == "base-image"
    assert tenki_module._get_snapshot_restore_candidate("persist") == (None, False)
    env.cleanup()


def test_tenki_snapshot_store_bound_to_construction_profile(monkeypatch, tmp_path):
    """Cleanup (which may run in a background thread without the per-turn
    HERMES_HOME contextvar) must write the snapshot pointer to the profile that
    was active at construction, not whatever home is ambient at cleanup time."""
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text("auth_token: tok-secret\n", encoding="utf-8")

    from tools.environments import tenki as tenki_module
    from tools.environments.tenki import TenkiEnvironment

    home_a = tmp_path / "profiles" / "a"
    home_b = tmp_path / "profiles" / "b"
    home_a.mkdir(parents=True)
    home_b.mkdir(parents=True)

    monkeypatch.setenv("HERMES_HOME", str(home_a))
    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    env = TenkiEnvironment(task_id="persist", persistent_filesystem=True)

    # Simulate a background cleanup running under the WRONG ambient home.
    monkeypatch.setenv("HERMES_HOME", str(home_b))
    env.cleanup()

    # Pointer landed in profile A's store (construction-time), not B's.
    assert (home_a / "tenki_snapshots.json").exists()
    assert not (home_b / "tenki_snapshots.json").exists()


def test_tenki_persistent_not_terminated_when_snapshot_and_pause_both_fail(monkeypatch, tmp_path):
    """Durability failed AND pause failed: the sandbox must be left live (not
    terminated), so the only copy of un-snapshotted state is preserved."""
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text("auth_token: tok-secret\n", encoding="utf-8")

    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    def _fail_durable(*_args, **_kwargs):
        raise RuntimeError("not durable")

    def _init(self, **kw):
        self.kwargs = kw
        self.snapshots = SimpleNamespace(wait_durable=_fail_durable)

    monkeypatch.setattr(_FakeClient, "__init__", _init)
    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    env = TenkiEnvironment(task_id="persist", persistent_filesystem=True)
    sandbox = _FakeSandboxFactory.sandboxes[0]

    def _fail_pause():
        raise RuntimeError("pause unavailable")

    sandbox.pause = _fail_pause

    env.cleanup()

    # Neither snapshot durable nor pause succeeded → sandbox left live.
    assert sandbox.terminated is False


def test_tenki_environment_resumes_paused_cached_sandbox_before_execute(monkeypatch, tmp_path):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text("auth_token: tok-secret\n", encoding="utf-8")

    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    env = TenkiEnvironment(task_id="paused-cache")
    sandbox = env._sandbox
    sandbox.state = "PAUSED"

    env.execute("echo ok", timeout=5)

    assert sandbox.refreshed is True
    assert sandbox.resumed is True
    assert sandbox.waited is True
    assert env._sandbox is sandbox
    env.cleanup()


def test_tenki_environment_recreates_terminated_cached_sandbox(monkeypatch, tmp_path):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text("auth_token: tok-secret\n", encoding="utf-8")

    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    env = TenkiEnvironment(task_id="terminated-cache")
    first = env._sandbox
    first.state = "TERMINATED"

    env.execute("echo ok", timeout=5)

    assert len(_FakeSandboxFactory.sandboxes) == 2
    assert env._sandbox is _FakeSandboxFactory.sandboxes[1]
    assert env._sandbox is not first
    env.cleanup()


def test_tenki_environment_ignores_mismatched_persistent_sandbox(monkeypatch, tmp_path):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text("auth_token: tok-secret\n", encoding="utf-8")
    _FakeClient.listed_sandboxes = [
        _FakeSandbox(
            name="hermes-other",
            state="PAUSED",
            metadata={"hermes_task_id": "other"},
        )
    ]

    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    env = TenkiEnvironment(task_id="persist", persistent_filesystem=True)

    assert _FakeSandboxFactory.created_kwargs
    assert _FakeSandboxFactory.created_kwargs[0]["name"].endswith("persist")
    env.cleanup()


def test_tenki_environment_converts_idle_timeout_to_sdk_minutes(monkeypatch, tmp_path):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text("auth_token: tok-secret\n", encoding="utf-8")

    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    env = TenkiEnvironment(task_id="idle", cpu=1.2, idle_timeout=61)

    kwargs = _FakeSandboxFactory.created_kwargs[0]
    assert kwargs["cpu_cores"] == 2
    assert kwargs["idle_timeout_minutes"] == 2
    env.cleanup()


def test_tenki_environment_omits_non_positive_pause_retention(monkeypatch, tmp_path):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text("auth_token: tok-secret\n", encoding="utf-8")

    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)

    env = TenkiEnvironment(task_id="pause-default", pause_retention=0)
    kwargs = _FakeSandboxFactory.created_kwargs[0]
    assert "pause_retention" not in kwargs
    env.cleanup()

    env = TenkiEnvironment(task_id="pause-negative", pause_retention=-1)
    kwargs = _FakeSandboxFactory.created_kwargs[1]
    assert "pause_retention" not in kwargs
    env.cleanup()


def test_tenki_environment_passes_positive_pause_retention(monkeypatch, tmp_path):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text("auth_token: tok-secret\n", encoding="utf-8")

    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    env = TenkiEnvironment(task_id="pause-positive", pause_retention=3600)

    kwargs = _FakeSandboxFactory.created_kwargs[0]
    assert kwargs["pause_retention"] == 3600
    env.cleanup()


def test_tenki_sync_hermes_home_is_opt_in(monkeypatch, tmp_path):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text("auth_token: tok-secret\n", encoding="utf-8")

    from tools.environments import tenki as tenki_module
    from tools.environments.tenki import TenkiEnvironment

    calls = []

    class FakeSyncManager:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def sync(self, *, force=False):
            calls.append(("sync", force))

        def sync_back(self):
            calls.append(("sync_back", None))

    monkeypatch.setattr(tenki_module, "FileSyncManager", FakeSyncManager)
    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)

    env = TenkiEnvironment(task_id="no-sync", sync_hermes_home=False)
    assert calls == []
    env.cleanup()

    env = TenkiEnvironment(task_id="sync", sync_hermes_home=True)
    assert calls[0][0] == "init"
    assert calls[1] == ("sync", True)
    env.cleanup()
    assert ("sync_back", None) in calls


def test_tenki_bulk_sync_stages_tar_under_home_not_tmp(monkeypatch, tmp_path):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text("auth_token: tok-secret\n", encoding="utf-8")

    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    env = TenkiEnvironment(task_id="bulk-sync")
    host_file = tmp_path / "skill.md"
    host_file.write_text("content", encoding="utf-8")

    env._tenki_bulk_upload([(str(host_file), "/home/tenki/.hermes/skills/skill.md")])

    remote_tar = env._sandbox.fs.upload_calls[-1][1]
    assert remote_tar.startswith("/home/tenki/.hermes_tenki_sync.")
    assert not remote_tar.startswith("/tmp/")
    env.cleanup()


def test_tenki_bulk_sync_uses_documented_fs_root_when_home_differs(monkeypatch, tmp_path):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text("auth_token: tok-secret\n", encoding="utf-8")

    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    env = TenkiEnvironment(task_id="root-home")
    env._remote_home = "/root"

    assert env._remote_transfer_path(".hermes_tenki_sync").startswith("/home/tenki/")
    env.cleanup()


def test_tenki_cleanup_sync_back_uses_original_sandbox(monkeypatch, tmp_path):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text("auth_token: tok-secret\n", encoding="utf-8")

    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    env = TenkiEnvironment(task_id="cleanup-sync")
    original = env._sandbox
    created_before = len(_FakeSandboxFactory.created_kwargs)

    class FakeSyncManager:
        def sync_back(self):
            env._tenki_bulk_download(tmp_path / "sync-back.tar")

    env._sync_manager = FakeSyncManager()
    env.cleanup()

    assert len(_FakeSandboxFactory.created_kwargs) == created_before
    assert original.fs.download_calls
    remote_tar = original.fs.download_calls[-1][0]
    assert remote_tar.startswith("/home/tenki/.hermes_tenki_sync_back.")
    assert original.terminated is True


def test_tenki_cleanup_blocks_public_execution_while_syncing(monkeypatch, tmp_path):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text("auth_token: tok-secret\n", encoding="utf-8")

    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    env = TenkiEnvironment(task_id="cleanup-guard")
    created_before = len(_FakeSandboxFactory.created_kwargs)

    with env._lock:
        env._cleanup_in_progress = True
        env._cleanup_sandbox = env._sandbox
    try:
        try:
            env.execute("echo should-not-run", timeout=5)
        except RuntimeError as exc:
            assert "cleanup" in str(exc)
        else:
            raise AssertionError("execute should fail while cleanup is in progress")
        assert len(_FakeSandboxFactory.created_kwargs) == created_before
    finally:
        with env._lock:
            env._cleanup_in_progress = False
            env._cleanup_sandbox = None
    env.cleanup()


def test_tenki_execute_passes_stdin_natively_not_as_heredoc(monkeypatch, tmp_path):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text("auth_token: tok-secret\n", encoding="utf-8")

    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    env = TenkiEnvironment(task_id="stdin")
    large_stdin = "x" * 200_000

    env.execute("cat > /home/tenki/out.txt", stdin_data=large_stdin, timeout=5)

    sandbox = env._sandbox
    command = _last_started_command(sandbox)
    assert large_stdin not in command
    assert "HERMES_STDIN_" not in command
    assert sandbox.start_calls[-1][1]["stdin"] == large_stdin
    assert "TENKI_AUTH_TOKEN" not in sandbox.start_calls[-1][1]["env"]
    env.cleanup()


def test_tenki_cancel_kills_process_without_tearing_down_sandbox(monkeypatch, tmp_path):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text("auth_token: tok-secret\n", encoding="utf-8")

    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    env = TenkiEnvironment(task_id="cancel-process")
    sandbox = env._sandbox
    handle = env._run_bash("sleep infinity", timeout=30)
    for _ in range(100):
        if sandbox.last_process is not None:
            break
        time.sleep(0.01)

    handle.kill()
    handle.wait(timeout=1)

    assert sandbox.last_process is not None
    assert sandbox.last_process.killed is True
    assert sandbox.terminated is False
    assert sandbox.paused is False
    env.cleanup()


def test_tenki_non_sudo_command_does_not_probe_sudo(monkeypatch, tmp_path):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text("auth_token: tok-secret\n", encoding="utf-8")

    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)

    def fail_probe(self):
        raise AssertionError("sudo should not be probed for commands without sudo")

    monkeypatch.setattr(TenkiEnvironment, "_sudo_nopasswd_works", fail_probe)

    env = TenkiEnvironment(task_id="no-sudo")
    env.execute("echo ok", timeout=5)
    env.cleanup()


def test_tenki_passwordless_sudo_does_not_prompt_or_rewrite(monkeypatch, tmp_path):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text("auth_token: tok-secret\n", encoding="utf-8")

    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    monkeypatch.setattr(TenkiEnvironment, "_sudo_nopasswd_works", lambda self: True)

    def fail_prompt(*_args, **_kwargs):
        raise AssertionError("Tenki sudo should not prompt for a host password")

    monkeypatch.setattr("tools.terminal_tool._prompt_for_sudo_password", fail_prompt)

    env = TenkiEnvironment(task_id="sudo-nopasswd")
    env.execute("sudo whoami", timeout=5)

    command = _last_started_command(_FakeSandboxFactory.sandboxes[0])
    assert "sudo whoami" in command
    assert "sudo -S" not in command
    assert "sudo -n whoami" not in command
    env.cleanup()


def test_tenki_sudo_without_nopasswd_fails_fast_without_host_password(monkeypatch, tmp_path):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setenv("SUDO_PASSWORD", "host-secret")
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text("auth_token: tok-secret\n", encoding="utf-8")

    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    monkeypatch.setattr(TenkiEnvironment, "_sudo_nopasswd_works", lambda self: False)

    def fail_prompt(*_args, **_kwargs):
        raise AssertionError("Tenki sudo should not prompt for a host password")

    monkeypatch.setattr("tools.terminal_tool._prompt_for_sudo_password", fail_prompt)

    env = TenkiEnvironment(task_id="sudo-no-nopasswd")
    env.execute("sudo whoami", timeout=5)

    command = _last_started_command(_FakeSandboxFactory.sandboxes[0])
    assert "sudo -n whoami" in command
    assert "sudo -S" not in command
    assert "host-secret" not in command
    env.cleanup()
