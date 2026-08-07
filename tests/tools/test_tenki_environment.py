from __future__ import annotations

import shlex
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
        sandbox_id: str | None = "sb-test",
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
        self.id = sandbox_id
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
    """Mirrors tenki.SnapshotNotFoundError for the fake SDK."""


class _FakeRegistryImageNotFoundError(Exception):
    """Mirrors tenki.RegistryImageNotFoundError for the fake SDK.

    Named ``RegistryArtifactNotFoundError`` before tenki 0.5.
    """


class _FakeSnapshotNotDurableError(Exception):
    """Mirrors tenki.SnapshotNotDurableError for the fake SDK."""


class _FakeInvalidStateError(Exception):
    """Mirrors tenki.InvalidStateError for the fake SDK."""


class _FakeSessionNotFoundError(Exception):
    """Mirrors tenki.SessionNotFoundError for the fake SDK."""


# Sentinel so the fake records exactly which kwargs the environment passed,
# rather than the defaults it didn't.
_UNSET = object()

# Kwargs that tenki's Sandbox.create pops into the Client it builds instead of
# forwarding to Client.create.
_CLIENT_ONLY_KWARGS = ("auth_token", "base_url", "gateway_url", "cookie_name", "timeout")


class _FakeSandboxFactory:
    """Mirrors ``tenki.Sandbox``.

    ``create`` is a bare ``**kwargs`` passthrough — exactly like the real SDK's
    — that pops the client-construction kwargs and forwards the rest to
    ``Client.create``, which is the thing that actually validates names.
    """

    created_kwargs: list[dict] = []
    failed_kwargs: list[dict] = []
    sandboxes: list[_FakeSandbox] = []
    next_sandbox_id = 0
    fail_snapshot_ids: set[str] = set()
    # When a snapshot id is in fail_snapshot_ids, raise this exception type with
    # this message. Defaults to the confirmed-not-found error; tests set them to
    # a transient error / generic message to prove the pointer is preserved, or
    # to a snapshot-specific InvalidStateError to prove base-image fallback.
    snapshot_error: type[Exception] = _FakeSnapshotNotFoundError
    snapshot_error_msg: str = "restore failed"

    @classmethod
    def create(cls, **kwargs):
        client_kwargs = {key: kwargs.pop(key) for key in _CLIENT_ONLY_KWARGS if key in kwargs}
        return _FakeClient(**client_kwargs).create(**kwargs)

    @classmethod
    def _record_and_build(cls, kwargs: dict):
        if kwargs.get("snapshot_id") in cls.fail_snapshot_ids:
            cls.failed_kwargs.append(kwargs)
            raise cls.snapshot_error(cls.snapshot_error_msg)
        cls.next_sandbox_id += 1
        sandbox = _FakeSandbox(
            name=kwargs.get("name", "sb-test"),
            metadata=kwargs.get("metadata", {}),
            sandbox_id=f"sb-created-{cls.next_sandbox_id}",
        )
        cls.created_kwargs.append(kwargs)
        cls.sandboxes.append(sandbox)
        return sandbox


class _FakeClient:
    listed_sandboxes: list[_FakeSandbox] = []
    remote_sandboxes: list[_FakeSandbox] = []
    snapshot_get_results: dict[str, object] = {}
    closed_count = 0
    active_generation = 0
    deleted_snapshot_ids: list[str] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._generation = type(self).active_generation

        def get_snapshot(snapshot_id):
            if snapshot_id in _FakeSandboxFactory.fail_snapshot_ids:
                raise _FakeSandboxFactory.snapshot_error(
                    _FakeSandboxFactory.snapshot_error_msg
                )
            if snapshot_id in type(self).snapshot_get_results:
                return type(self).snapshot_get_results[snapshot_id]
            return SimpleNamespace(
                id=snapshot_id,
                state="READY",
                durability_state="DURABLE",
            )

        self.snapshots = SimpleNamespace(
            get=get_snapshot,
            wait_durable=lambda *_args, **_kwargs: None,
            delete=lambda snapshot_id: type(self).deleted_snapshot_ids.append(
                snapshot_id
            ),
        )

    # The parameter list mirrors tenki 0.5's Client.create and deliberately has
    # NO **kwargs catch-all: a name the SDK dropped (project_id, removed in 0.5)
    # raises TypeError here exactly as it would against the real client, so the
    # environment's create-kwarg filtering is genuinely under test.
    def create(
        self,
        *,
        workspace_id=_UNSET,
        name=_UNSET,
        wait=_UNSET,
        timeout=_UNSET,
        allow_inbound=_UNSET,
        allow_outbound=_UNSET,
        max_duration=_UNSET,
        idle_timeout_minutes=_UNSET,
        pause_retention=_UNSET,
        cpu_cores=_UNSET,
        memory_mb=_UNSET,
        disk_size_gb=_UNSET,
        metadata=_UNSET,
        tags=_UNSET,
        env=_UNSET,
        ssh_authorized_keys=_UNSET,
        snapshot_id=_UNSET,
        image=_UNSET,
        sticky=_UNSET,
    ):
        passed = {
            key: value
            for key, value in locals().items()
            if key != "self" and value is not _UNSET
        }
        # In the real SDK the client kwargs live on the client; merge them so
        # tests can assert against the whole create call in one dict.
        return _FakeSandboxFactory._record_and_build({**self.kwargs, **passed})

    # tenki 0.5 folded list_project/list_workspace into list(workspace_id=...).
    def list(self, *, workspace_id=None, tags=None, sticky=None):
        candidates = [*self.listed_sandboxes, *_FakeSandboxFactory.sandboxes]
        result = []
        seen: set[str] = set()
        for sandbox in candidates:
            if sandbox.id in seen:
                continue
            seen.add(sandbox.id)
            result.append(sandbox)
            if sandbox not in type(self).remote_sandboxes:
                type(self).remote_sandboxes.append(sandbox)
        return result

    def get(self, sandbox_id):
        candidates = [
            *_FakeSandboxFactory.sandboxes,
            *type(self).remote_sandboxes,
            *type(self).listed_sandboxes,
        ]
        for sandbox in candidates:
            if sandbox.id == sandbox_id:
                return sandbox
        raise _FakeSessionNotFoundError(sandbox_id)

    def close(self):
        # A delayed close from a prior test must not mutate the current test's
        # generation after _install_fake_tenki resets shared fake state.
        if getattr(self, "_generation", type(self).active_generation) == (
            type(self).active_generation
        ):
            type(self).closed_count += 1


def _install_fake_tenki(monkeypatch):
    module = types.ModuleType("tenki")
    _FakeSandboxFactory.created_kwargs = []
    _FakeSandboxFactory.failed_kwargs = []
    _FakeSandboxFactory.sandboxes = []
    _FakeSandboxFactory.fail_snapshot_ids = set()
    _FakeSandboxFactory.snapshot_error = _FakeSnapshotNotFoundError
    _FakeSandboxFactory.snapshot_error_msg = "restore failed"
    _FakeClient.listed_sandboxes = []
    _FakeClient.remote_sandboxes = []
    _FakeClient.snapshot_get_results = {}
    _FakeClient.active_generation += 1
    _FakeClient.closed_count = 0
    _FakeClient.deleted_snapshot_ids = []
    module.Client = _FakeClient
    module.Sandbox = _FakeSandboxFactory
    module.SnapshotNotFoundError = _FakeSnapshotNotFoundError
    module.RegistryImageNotFoundError = _FakeRegistryImageNotFoundError
    module.SnapshotNotDurableError = _FakeSnapshotNotDurableError
    module.InvalidStateError = _FakeInvalidStateError
    module.SessionNotFoundError = _FakeSessionNotFoundError
    monkeypatch.setitem(sys.modules, "tenki", module)


def _release_quarantined_locks_since(tenki_module, start: int) -> None:
    with tenki_module._QUARANTINED_TASK_OWNERSHIP_GUARD:
        lock_files = tenki_module._QUARANTINED_TASK_OWNERSHIP_FILES[start:]
        del tenki_module._QUARANTINED_TASK_OWNERSHIP_FILES[start:]
    for lock_file in lock_files:
        tenki_module._release_task_ownership_lock(lock_file)


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
                "auth_token: tok-secret",
            ]
        ),
        encoding="utf-8",
    )

    from tools.environments import tenki as tenki_module
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
    assert kwargs["auth_token"] == "cookie:tok-secret"
    # ...but is NEVER injected into the model-controlled guest environment
    # (an empty env is omitted from the create kwargs entirely).
    guest_env = kwargs.get("env", {})
    assert "TENKI_AUTH_TOKEN" not in guest_env
    assert "TENKI_API_KEY" not in guest_env
    assert "TENKI_API_ENDPOINT" not in guest_env
    assert "TENKI_WORKSPACE_ID" not in guest_env
    assert kwargs["allow_inbound"] is False
    assert kwargs["allow_outbound"] is True
    assert kwargs["cpu_cores"] == 1
    assert "idle_timeout" not in kwargs
    assert "idle_timeout_minutes" not in kwargs
    assert "pause_retention" not in kwargs
    assert kwargs["metadata"]["hermes_backend"] == "tenki"
    assert kwargs["metadata"]["hermes_profile"]
    assert kwargs["metadata"]["hermes_create_attempt"]
    assert kwargs["wait"] is False
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
    assert tenki_module._get_create_attempt("session 1") is None


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


def test_tenki_persistent_snapshot_retires_superseded_remote_copy(
    monkeypatch,
    tmp_path,
):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text(
        "auth_token: tok-secret\n",
        encoding="utf-8",
    )

    from tools.environments import tenki as tenki_module
    from tools.environments.tenki import TenkiEnvironment

    tenki_module._store_snapshot("persist", "snap-prior")
    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    env = TenkiEnvironment(task_id="persist", persistent_filesystem=True)
    sandbox = env._sandbox

    env.cleanup()

    replacement_id = f"snap-{sandbox.name}"
    assert tenki_module._get_snapshot_restore_candidate("persist") == (
        replacement_id,
        False,
    )
    assert _FakeClient.deleted_snapshot_ids == ["snap-prior"]
    assert sandbox.terminated is True


def test_tenki_snapshot_delete_failure_keeps_new_durable_pointer(
    monkeypatch,
    tmp_path,
):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text(
        "auth_token: tok-secret\n",
        encoding="utf-8",
    )

    from tools.environments import tenki as tenki_module
    from tools.environments.tenki import TenkiEnvironment

    tenki_module._store_snapshot("persist", "snap-prior")

    delete_attempts = []

    def flaky_delete(snapshot_id):
        delete_attempts.append(snapshot_id)
        if len(delete_attempts) == 1:
            raise RuntimeError("control plane unavailable")
        _FakeClient.deleted_snapshot_ids.append(snapshot_id)

    def _init(self, **kw):
        self.kwargs = kw
        self.snapshots = SimpleNamespace(
            wait_durable=lambda *_args, **_kwargs: None,
            delete=flaky_delete,
        )

    monkeypatch.setattr(_FakeClient, "__init__", _init)
    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    env = TenkiEnvironment(task_id="persist", persistent_filesystem=True)
    sandbox = env._sandbox

    env.cleanup()

    assert tenki_module._get_snapshot_restore_candidate("persist") == (
        f"snap-{sandbox.name}",
        False,
    )
    assert sandbox.terminated is True
    assert tenki_module._pending_snapshot_retirements() == ("snap-prior",)

    # The next lifecycle retries the durable profile-scoped journal without
    # needing the failed predecessor to remain the active restore pointer.
    retry_env = TenkiEnvironment(
        task_id="persist",
        persistent_filesystem=True,
    )

    assert delete_attempts[:2] == ["snap-prior", "snap-prior"]
    assert _FakeClient.deleted_snapshot_ids == ["snap-prior"]
    assert tenki_module._pending_snapshot_retirements() == ()
    with pytest.raises(
        tenki_module._SnapshotPointerConflict,
        match="already retired",
    ):
        tenki_module._store_snapshot("stale-writer", "snap-prior")
    retry_env.cleanup()


def test_tenki_retirement_claim_is_durable_before_remote_delete(
    monkeypatch,
    tmp_path,
):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text(
        "auth_token: tok-secret\n",
        encoding="utf-8",
    )

    from tools.environments import tenki as tenki_module
    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    env = TenkiEnvironment(task_id="retirement-claim")
    tenki_module._queue_snapshot_retirement(
        "snap-old",
        env._snapshot_store,
    )
    real_atomic_save = tenki_module._atomic_save_snapshots

    def fail_before_claim_replace(path, snapshots):
        if snapshots.get(
            tenki_module._snapshot_retired_key("snap-old")
        ) == "snap-old":
            raise OSError("ENOSPC before tombstone replace")
        return real_atomic_save(path, snapshots)

    monkeypatch.setattr(
        tenki_module,
        "_atomic_save_snapshots",
        fail_before_claim_replace,
    )

    assert env._retire_pending_snapshot_if_unreferenced(
        "snap-old",
        reason="claim failure probe",
    ) is False
    assert _FakeClient.deleted_snapshot_ids == []

    def visible_but_uncertain_claim(path, snapshots):
        real_atomic_save(path, snapshots)
        raise tenki_module._SnapshotPointerCommitUncertain(
            "directory fsync failed",
        )

    monkeypatch.setattr(
        tenki_module,
        "_atomic_save_snapshots",
        visible_but_uncertain_claim,
    )
    assert env._retire_pending_snapshot_if_unreferenced(
        "snap-old",
        reason="uncertain claim probe",
    ) is False
    assert _FakeClient.deleted_snapshot_ids == []

    # The visible uncertain claim is not acted on until its store is explicitly
    # made durable. Once durable, deletion and non-resurrection are atomic in
    # the required direction.
    monkeypatch.setattr(
        tenki_module,
        "_atomic_save_snapshots",
        real_atomic_save,
    )
    tenki_module._confirm_snapshot_store_durable(env._snapshot_store)
    assert env._retire_pending_snapshot_if_unreferenced(
        "snap-old",
        reason="claim retry",
    ) is True
    assert _FakeClient.deleted_snapshot_ids == ["snap-old"]
    with pytest.raises(
        tenki_module._SnapshotPointerConflict,
        match="already retired",
    ):
        tenki_module._store_snapshot("stale-writer", "snap-old")
    env.cleanup()


def test_tenki_windows_write_through_failure_keeps_remote_snapshot(
    monkeypatch,
    tmp_path,
):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text(
        "auth_token: tok-secret\n",
        encoding="utf-8",
    )

    from tools.environments import tenki as tenki_module
    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    env = TenkiEnvironment(task_id="windows-retirement")
    tenki_module._queue_snapshot_retirement(
        "snap-windows",
        env._snapshot_store,
    )
    monkeypatch.setattr(tenki_module, "_snapshot_platform", lambda: "nt")
    monkeypatch.setattr(
        tenki_module,
        "_windows_replace_file_write_through",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("write-through replacement failed")
        ),
    )

    assert env._retire_pending_snapshot_if_unreferenced(
        "snap-windows",
        reason="Windows durability probe",
    ) is False
    assert _FakeClient.deleted_snapshot_ids == []
    monkeypatch.setattr(tenki_module, "_snapshot_platform", lambda: "posix")
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

    # The snapshot is rejected by an authoritative preflight before the
    # non-idempotent create RPC is issued.
    assert _FakeSandboxFactory.failed_kwargs == []
    assert _FakeSandboxFactory.created_kwargs[0]["image"] == "base-image"
    assert tenki_module._get_snapshot_restore_candidate("persist") == (None, False)
    assert _FakeClient.deleted_snapshot_ids == ["snap-stale"]
    env.cleanup()


def test_tenki_snapshot_error_after_commit_never_creates_second_remote(
    monkeypatch,
    tmp_path,
):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text(
        "auth_token: tok-secret\n",
        encoding="utf-8",
    )

    from tools.environments import tenki as tenki_module
    from tools.environments.tenki import TenkiEnvironment

    task_id = "snapshot-error-after-commit"
    tenki_module._store_snapshot(task_id, "snap-existing")
    original_create = _FakeClient.create

    def commit_then_snapshot_error(self, **kwargs):
        original_create(self, **kwargs)
        raise _FakeSnapshotNotFoundError("decode failed after commit")

    monkeypatch.setattr(_FakeClient, "create", commit_then_snapshot_error)
    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)

    env = TenkiEnvironment(
        task_id=task_id,
        image="base-image",
        persistent_filesystem=True,
    )

    assert len(_FakeSandboxFactory.created_kwargs) == 1
    assert env._sandbox is _FakeSandboxFactory.sandboxes[0]
    assert tenki_module._get_snapshot_restore_candidate(task_id) == (
        "snap-existing",
        False,
    )
    assert _FakeClient.deleted_snapshot_ids == []
    env.cleanup()


@pytest.mark.parametrize(
    ("state", "durability_state"),
    [
        ("FAILED", "PROPAGATION_FAILED"),
        ("READY", "UNSPECIFIED"),
        ("DELETING", "DURABLE"),
    ],
)
def test_tenki_snapshot_preflight_rejects_returned_unusable_state(
    monkeypatch,
    tmp_path,
    state,
    durability_state,
):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text(
        "auth_token: tok-secret\n",
        encoding="utf-8",
    )

    from tools.environments import tenki as tenki_module
    from tools.environments.tenki import TenkiEnvironment

    task_id = f"bad-snapshot-{state.lower()}"
    snapshot_id = f"snap-{state.lower()}"
    tenki_module._store_snapshot(task_id, snapshot_id)
    _FakeClient.snapshot_get_results[snapshot_id] = SimpleNamespace(
        id=snapshot_id,
        state=state,
        durability_state=durability_state,
        failure_reason="injected failure",
    )
    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)

    env = TenkiEnvironment(
        task_id=task_id,
        image="base-image",
        persistent_filesystem=True,
    )

    assert len(_FakeSandboxFactory.created_kwargs) == 1
    assert "snapshot_id" not in _FakeSandboxFactory.created_kwargs[0]
    assert _FakeSandboxFactory.created_kwargs[0]["image"] == "base-image"
    assert tenki_module._get_snapshot_restore_candidate(task_id) == (
        None,
        False,
    )
    env.cleanup()


@pytest.mark.parametrize(
    ("state", "durability_state"),
    [
        ("CREATING", "UNSPECIFIED"),
        ("READY", "PROPAGATING"),
    ],
)
def test_tenki_snapshot_preflight_preserves_transition_state(
    monkeypatch,
    tmp_path,
    state,
    durability_state,
):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text(
        "auth_token: tok-secret\n",
        encoding="utf-8",
    )

    from tools.environments import tenki as tenki_module
    from tools.environments.tenki import TenkiEnvironment

    task_id = f"transition-snapshot-{state.lower()}"
    snapshot_id = f"snap-{state.lower()}"
    tenki_module._store_snapshot(task_id, snapshot_id)
    _FakeClient.snapshot_get_results[snapshot_id] = SimpleNamespace(
        id=snapshot_id,
        state=state,
        durability_state=durability_state,
    )
    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)

    with pytest.raises(RuntimeError, match="not durably restorable"):
        TenkiEnvironment(
            task_id=task_id,
            image="base-image",
            persistent_filesystem=True,
        )

    assert not _FakeSandboxFactory.created_kwargs
    assert tenki_module._get_snapshot_restore_candidate(task_id) == (
        snapshot_id,
        False,
    )
    assert _FakeClient.deleted_snapshot_ids == []


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


def test_tenki_readiness_error_keeps_exact_snapshot_remote_and_pointer(
    monkeypatch,
    tmp_path,
):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text(
        "auth_token: tok-secret\n",
        encoding="utf-8",
    )

    from tools.environments import tenki as tenki_module
    from tools.environments.tenki import TenkiEnvironment

    tenki_module._store_snapshot("persist", "snap-ready-error")

    def fail_readiness(self, *_args, **_kwargs):
        raise _FakeSnapshotNotFoundError("readiness failed after create")

    monkeypatch.setattr(_FakeSandbox, "wait_ready", fail_readiness)
    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)

    with pytest.raises(
        _FakeSnapshotNotFoundError,
        match="readiness failed after create",
    ):
        TenkiEnvironment(
            task_id="persist",
            persistent_filesystem=True,
        )

    sandbox = _FakeSandboxFactory.sandboxes[0]
    assert _FakeSandboxFactory.created_kwargs[0]["wait"] is False
    assert sandbox.paused is True
    assert sandbox.terminated is False
    assert tenki_module._get_snapshot_restore_candidate("persist") == (
        "snap-ready-error",
        False,
    )


def test_tenki_environment_skips_snapshot_when_not_durable(monkeypatch, tmp_path):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text("auth_token: tok-secret\n", encoding="utf-8")

    from tools.environments import tenki as tenki_module
    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    tenki_module._store_snapshot("persist", "snap-prior")

    def _fail_durable(*_args, **_kwargs):
        raise RuntimeError("not durable yet")

    def _init(self, **kw):
        self.kwargs = kw
        self.snapshots = SimpleNamespace(
            wait_durable=_fail_durable,
            delete=lambda snapshot_id: type(self).deleted_snapshot_ids.append(
                snapshot_id
            ),
        )

    monkeypatch.setattr(_FakeClient, "__init__", _init)
    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    env = TenkiEnvironment(task_id="persist", persistent_filesystem=True)
    sandbox = _FakeSandboxFactory.sandboxes[0]

    env.cleanup()

    # Durability failed → do NOT record the snapshot and do NOT terminate the
    # live sandbox; pause it so state is preserved for recovery.
    assert sandbox.paused is True
    assert sandbox.terminated is False
    assert tenki_module._get_snapshot_restore_candidate("persist") == (
        "snap-prior",
        False,
    )
    assert _FakeClient.deleted_snapshot_ids == [f"snap-{sandbox.name}"]


def test_tenki_snapshot_pointer_failure_preserves_prior_and_retires_new_copy(
    monkeypatch,
    tmp_path,
):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text(
        "auth_token: tok-secret\n",
        encoding="utf-8",
    )

    from tools.environments import tenki as tenki_module
    from tools.environments.tenki import TenkiEnvironment

    tenki_module._store_snapshot("persist", "snap-prior")
    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    env = TenkiEnvironment(task_id="persist", persistent_filesystem=True)
    sandbox = env._sandbox

    def fail_pointer_store(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(tenki_module, "_store_snapshot", fail_pointer_store)
    env.cleanup()

    assert sandbox.paused is True
    assert sandbox.terminated is False
    assert tenki_module._get_snapshot_restore_candidate("persist") == (
        "snap-prior",
        False,
    )
    assert _FakeClient.deleted_snapshot_ids == [f"snap-{sandbox.name}"]


def test_tenki_directory_fsync_uncertainty_keeps_both_snapshots_and_live_sandbox(
    monkeypatch,
    tmp_path,
):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text(
        "auth_token: tok-secret\n",
        encoding="utf-8",
    )

    from tools.environments import tenki as tenki_module
    from tools.environments.tenki import TenkiEnvironment

    tenki_module._store_snapshot("persist", "snap-prior")
    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    env = TenkiEnvironment(task_id="persist", persistent_filesystem=True)
    sandbox = env._sandbox
    original_fsync = tenki_module.os.fsync
    fsync_calls = 0

    def fail_directory_fsync(fd):
        nonlocal fsync_calls
        fsync_calls += 1
        if fsync_calls == 2:
            raise OSError("injected directory fsync failure")
        return original_fsync(fd)

    monkeypatch.setattr(tenki_module.os, "fsync", fail_directory_fsync)
    env.cleanup()

    replacement_id = f"snap-{sandbox.name}"
    # os.replace made the new pointer visible, but failed directory fsync
    # means a crash may reveal either pointer. Keep both remote copies and the
    # live sandbox; do not act as though the handoff committed durably.
    assert tenki_module._get_snapshot_restore_candidate("persist") == (
        replacement_id,
        False,
    )
    assert _FakeClient.deleted_snapshot_ids == []
    assert sandbox.paused is True
    assert sandbox.terminated is False


def test_tenki_environment_refuses_unmanaged_persistent_sandbox(
    monkeypatch,
    tmp_path,
):
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
    quarantine_start = len(
        tenki_module._QUARANTINED_TASK_OWNERSHIP_FILES
    )
    with pytest.raises(RuntimeError, match="unmanaged persistent sandbox"):
        TenkiEnvironment(task_id="persist", persistent_filesystem=True)

    assert existing.resumed is False
    assert existing.waited is False
    assert _FakeSandboxFactory.created_kwargs == []
    assert existing.terminated is False
    binding = tenki_module._load_snapshots()[
        tenki_module._remote_binding_key("persist")
    ]
    assert binding["conflicted"] is True
    assert binding["unmanaged"] is True
    assert binding["conflict_ids"] == [existing.id]
    _release_quarantined_locks_since(tenki_module, quarantine_start)


def test_tenki_unidentified_collision_is_durably_unresolvable(
    monkeypatch,
    tmp_path,
):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text(
        "auth_token: tok-secret\n",
        encoding="utf-8",
    )

    from tools.environments import tenki as tenki_module
    from tools.environments.tenki import TenkiEnvironment

    token = tenki_module._profile_token()
    unidentified = _FakeSandbox(
        sandbox_id=None,
        name=f"hermes-{token}-persist",
        metadata={"hermes_task_id": "persist", "hermes_profile": token},
    )
    _FakeClient.listed_sandboxes = [unidentified]
    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    quarantine_start = len(
        tenki_module._QUARANTINED_TASK_OWNERSHIP_FILES
    )

    with pytest.raises(RuntimeError, match="unmanaged persistent sandbox"):
        TenkiEnvironment(task_id="persist", persistent_filesystem=True)
    assert tenki_module._remote_binding_state("persist") == tenki_module._RemoteBinding(
        None,
        None,
        False,
        True,
        (),
        True,
    )
    assert not _FakeSandboxFactory.created_kwargs
    _release_quarantined_locks_since(tenki_module, quarantine_start)

    # A later omission cannot erase an observed branch whose exact id was
    # unavailable, so automatic recovery remains deliberately disabled.
    _FakeClient.listed_sandboxes = []
    with pytest.raises(
        RuntimeError,
        match="unresolvable durable persistent-lineage conflict",
    ):
        TenkiEnvironment(task_id="persist", persistent_filesystem=True)
    assert not _FakeSandboxFactory.created_kwargs
    _release_quarantined_locks_since(tenki_module, quarantine_start)


def test_tenki_mixed_unmanaged_collision_preserves_known_ids(
    monkeypatch,
    tmp_path,
):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text(
        "auth_token: tok-secret\n",
        encoding="utf-8",
    )

    from tools.environments import tenki as tenki_module
    from tools.environments.tenki import TenkiEnvironment

    task_id = "mixed-unmanaged"
    token = tenki_module._profile_token()
    metadata = {
        "hermes_task_id": task_id,
        "hermes_profile": token,
    }
    known = _FakeSandbox(
        sandbox_id="known-unmanaged",
        name=f"hermes-{token}-{task_id}",
        metadata=metadata,
    )
    unidentified = _FakeSandbox(
        sandbox_id=None,
        name=f"hermes-{token}-{task_id}",
        metadata=metadata,
    )
    _FakeClient.listed_sandboxes = [known, unidentified]
    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    quarantine_start = len(
        tenki_module._QUARANTINED_TASK_OWNERSHIP_FILES
    )

    with pytest.raises(RuntimeError, match="unmanaged persistent sandbox"):
        TenkiEnvironment(task_id=task_id, persistent_filesystem=True)

    assert tenki_module._remote_binding_state(task_id) == tenki_module._RemoteBinding(
        known.id,
        None,
        False,
        True,
        (known.id,),
        True,
    )
    assert not _FakeSandboxFactory.created_kwargs
    _release_quarantined_locks_since(tenki_module, quarantine_start)


def test_tenki_registration_loser_sharing_remote_releases_only_wrapper(
    monkeypatch,
    tmp_path,
):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text(
        "auth_token: tok-secret\n",
        encoding="utf-8",
    )

    from tools import terminal_tool
    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    winner = TenkiEnvironment(
        task_id="shared-remote",
        persistent_filesystem=True,
    )
    remote = winner._sandbox
    # Production cannot construct a second wrapper while this lifetime lock is
    # held. Release it deliberately to exercise only the defensive
    # same-remote registration path.
    winner._release_task_ownership()
    _FakeClient.listed_sandboxes = [remote]
    loser = TenkiEnvironment(
        task_id="shared-remote",
        persistent_filesystem=True,
    )

    assert loser._sandbox is remote
    assert loser.shares_remote_resource_with(winner) is True

    registry_key = "tenki:profile:shared-remote"
    with monkeypatch.context() as context:
        context.setattr(
            terminal_tool,
            "_active_environments",
            {registry_key: winner},
        )
        context.setattr(terminal_tool, "_env_lock", threading.Lock())
        selected = terminal_tool._register_active_environment(
            registry_key,
            loser,
        )

    assert selected is winner
    assert winner._sandbox is remote
    assert remote.terminated is False
    assert loser._sandbox is None
    assert loser._client is None
    assert loser._cleanup_complete is True
    assert _FakeClient.closed_count == 1
    winner.cleanup()


def test_tenki_persistent_list_failure_never_authorizes_creation(
    monkeypatch,
    tmp_path,
):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text(
        "auth_token: tok-secret\n",
        encoding="utf-8",
    )

    from tools.environments.tenki import TenkiEnvironment

    def fail_list(self, **_kwargs):
        raise RuntimeError("control plane unavailable")

    monkeypatch.setattr(_FakeClient, "list", fail_list)
    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)

    with pytest.raises(RuntimeError, match="could not check"):
        TenkiEnvironment(task_id="persist", persistent_filesystem=True)

    assert _FakeSandboxFactory.created_kwargs == []


def test_tenki_exact_binding_survives_list_omission_on_create_and_restart(
    monkeypatch,
    tmp_path,
):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text(
        "auth_token: tok-secret\n",
        encoding="utf-8",
    )

    from tools.environments import tenki as tenki_module
    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    monkeypatch.setattr(_FakeClient, "list", lambda self, **_kwargs: [])
    task_id = "list-omits-owned-binding"

    first = TenkiEnvironment(
        task_id=task_id,
        persistent_filesystem=True,
    )
    remote = first._sandbox
    assert len(_FakeSandboxFactory.created_kwargs) == 1
    assert tenki_module._remote_binding_state(task_id) == tenki_module._RemoteBinding(
        remote.id,
        remote.info.metadata["hermes_create_attempt"],
        True,
        False,
        (),
        False,
    )

    # Simulate a process exit: the durable exact-id binding survives, while
    # the list API still omits the live remote.
    first._release_task_ownership()
    first._sandbox = None
    first._client = None
    first._cleanup_complete = True

    successor = TenkiEnvironment(
        task_id=task_id,
        persistent_filesystem=True,
    )
    assert successor._sandbox is remote
    assert len(_FakeSandboxFactory.created_kwargs) == 1
    successor.cleanup()


def test_tenki_persistent_readiness_errors_preserve_exact_sandbox(
    monkeypatch,
    tmp_path,
):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text(
        "auth_token: tok-secret\n",
        encoding="utf-8",
    )

    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    env = TenkiEnvironment(task_id="persist", persistent_filesystem=True)
    sandbox = env._sandbox
    created_before = len(_FakeSandboxFactory.created_kwargs)

    def fail_refresh():
        raise RuntimeError("transient refresh failure")

    monkeypatch.setattr(sandbox, "refresh", fail_refresh)

    with pytest.raises(RuntimeError, match="could not refresh"):
        env._ensure_sandbox()

    assert env._sandbox is sandbox
    assert len(_FakeSandboxFactory.created_kwargs) == created_before


def test_tenki_unmanaged_persistent_remote_never_resumes_or_forks(
    monkeypatch,
    tmp_path,
):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text(
        "auth_token: tok-secret\n",
        encoding="utf-8",
    )

    from tools.environments import tenki as tenki_module
    from tools.environments.tenki import TenkiEnvironment

    token = tenki_module._profile_token()
    existing = _FakeSandbox(
        name=f"hermes-{token}-persist",
        state="PAUSED",
        metadata={"hermes_task_id": "persist", "hermes_profile": token},
    )

    def fail_resume():
        raise RuntimeError("transient resume failure")

    existing.resume = fail_resume
    _FakeClient.listed_sandboxes = [existing]
    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    quarantine_start = len(
        tenki_module._QUARANTINED_TASK_OWNERSHIP_FILES
    )

    with pytest.raises(RuntimeError, match="unmanaged persistent sandbox"):
        TenkiEnvironment(task_id="persist", persistent_filesystem=True)

    assert _FakeSandboxFactory.created_kwargs == []
    assert existing.resumed is False
    assert existing.terminated is False
    _release_quarantined_locks_since(tenki_module, quarantine_start)


def test_tenki_profile_identity_is_stable_across_runtime_modes(
    monkeypatch,
    tmp_path,
):
    from hermes_constants import (
        reset_hermes_home_override,
        set_hermes_home_override,
    )
    from tools.environments import tenki as tenki_module

    profile_home = tmp_path / "profiles" / "coder"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.setenv("HERMES_PROFILE", "coder")
    standalone_token = tenki_module._profile_token()

    token = set_hermes_home_override(profile_home)
    try:
        multiplexed_token = tenki_module._profile_token()
    finally:
        reset_hermes_home_override(token)

    assert multiplexed_token == standalone_token
    old_name_token = tenki_module._profile_token_for_basis("profile:coder")
    assert old_name_token in tenki_module._legacy_profile_tokens()


def test_tenki_migrates_legacy_profile_snapshot_key(monkeypatch, tmp_path):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    profile_home = tmp_path / "profiles" / "coder"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.setenv("HERMES_PROFILE", "coder")
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text(
        "auth_token: tok-secret\n",
        encoding="utf-8",
    )

    from tools.environments import tenki as tenki_module
    from tools.environments.tenki import TenkiEnvironment

    canonical_token = tenki_module._profile_token()
    legacy_token = tenki_module._profile_token_for_basis("profile:coder")
    canonical_task = f"tenki:{canonical_token}:default"
    legacy_task = f"tenki:{legacy_token}:default"
    tenki_module._store_snapshot(legacy_task, "snap-legacy-profile")
    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)

    env = TenkiEnvironment(
        task_id=canonical_task,
        persistent_filesystem=True,
    )

    assert _FakeSandboxFactory.created_kwargs[0]["snapshot_id"] == (
        "snap-legacy-profile"
    )
    assert tenki_module._get_snapshot_restore_candidate(
        canonical_task,
    ) == ("snap-legacy-profile", False)
    assert tenki_module._get_snapshot_restore_candidate(
        legacy_task,
    ) == (None, False)
    env.cleanup()


def test_tenki_legacy_migration_cas_never_overwrites_newer_pointer(
    monkeypatch,
    tmp_path,
):
    from tools.environments import tenki as tenki_module

    store = tmp_path / "tenki_snapshots.json"
    legacy_task = "tenki:legacy:default"
    canonical_task = "tenki:canonical:default"
    tenki_module._store_snapshot(
        legacy_task,
        "snap-legacy",
        store,
    )
    tenki_module._store_snapshot(
        canonical_task,
        "snap-newer",
        store,
    )

    with pytest.raises(
        tenki_module._SnapshotPointerConflict,
        match="advanced",
    ):
        tenki_module._migrate_snapshot_pointer(
            canonical_task,
            legacy_task,
            "snap-legacy",
            store,
        )

    assert tenki_module._get_snapshot_restore_candidate(
        canonical_task,
        store,
    ) == ("snap-newer", False)
    assert tenki_module._get_snapshot_restore_candidate(
        legacy_task,
        store,
    ) == ("snap-legacy", False)


def test_tenki_legacy_pointer_migration_failure_keeps_restored_sandbox(
    monkeypatch,
    tmp_path,
):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    profile_home = tmp_path / "profiles" / "coder"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.setenv("HERMES_PROFILE", "coder")
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text(
        "auth_token: tok-secret\n",
        encoding="utf-8",
    )

    from tools.environments import tenki as tenki_module
    from tools.environments.tenki import TenkiEnvironment

    canonical_token = tenki_module._profile_token()
    legacy_token = tenki_module._profile_token_for_basis("profile:coder")
    canonical_task = f"tenki:{canonical_token}:default"
    legacy_task = f"tenki:{legacy_token}:default"
    tenki_module._store_snapshot(legacy_task, "snap-legacy-profile")
    original_migrate = tenki_module._migrate_snapshot_pointer

    def fail_migration(*_args, **_kwargs):
        raise OSError("local pointer store unavailable")

    monkeypatch.setattr(
        tenki_module,
        "_migrate_snapshot_pointer",
        fail_migration,
    )
    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)

    env = TenkiEnvironment(
        task_id=canonical_task,
        persistent_filesystem=True,
    )
    restored = env._sandbox

    assert restored is _FakeSandboxFactory.sandboxes[0]
    assert len(_FakeSandboxFactory.created_kwargs) == 1
    assert _FakeSandboxFactory.created_kwargs[0]["snapshot_id"] == (
        "snap-legacy-profile"
    )
    assert tenki_module._get_snapshot_restore_candidate(legacy_task) == (
        "snap-legacy-profile",
        False,
    )
    assert env._snapshot_restore_task_id == legacy_task
    assert env._save_persistent_snapshot(restored) is False
    assert restored.snapshots == []

    monkeypatch.setattr(
        tenki_module,
        "_migrate_snapshot_pointer",
        original_migrate,
    )
    env._exec_raw("echo migration retry")

    assert env._sandbox is restored
    assert tenki_module._get_snapshot_restore_candidate(canonical_task) == (
        "snap-legacy-profile",
        False,
    )
    assert tenki_module._get_snapshot_restore_candidate(legacy_task) == (
        None,
        False,
    )
    env.cleanup()


def test_tenki_refuses_legacy_profile_named_sandbox(monkeypatch, tmp_path):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    profile_home = tmp_path / "profiles" / "coder"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.setenv("HERMES_PROFILE", "coder")
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text(
        "auth_token: tok-secret\n",
        encoding="utf-8",
    )

    from tools.environments import tenki as tenki_module
    from tools.environments.tenki import TenkiEnvironment

    canonical_token = tenki_module._profile_token()
    legacy_token = tenki_module._profile_token_for_basis("profile:coder")
    canonical_task = f"tenki:{canonical_token}:default"
    legacy_task = f"tenki:{legacy_token}:default"
    existing = _FakeSandbox(
        name=(
            f"hermes-{legacy_token}-"
            f"{tenki_module._safe_name(legacy_task)}"
        ),
        state="PAUSED",
        metadata={
            "hermes_task_id": legacy_task,
            "hermes_profile": legacy_token,
        },
    )
    _FakeClient.listed_sandboxes = [existing]
    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    quarantine_start = len(
        tenki_module._QUARANTINED_TASK_OWNERSHIP_FILES
    )

    with pytest.raises(RuntimeError, match="unmanaged persistent sandbox"):
        TenkiEnvironment(
            task_id=canonical_task,
            persistent_filesystem=True,
        )

    assert existing.resumed is False
    assert existing.terminated is False
    assert _FakeSandboxFactory.created_kwargs == []
    _release_quarantined_locks_since(tenki_module, quarantine_start)


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

    assert _FakeSandboxFactory.failed_kwargs == []
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


def test_tenki_sync_enumeration_and_cleanup_stay_bound_to_owner_profile(
    monkeypatch,
    tmp_path,
):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    process_home = tmp_path / "process-home"
    process_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(process_home))
    tenki_config = tmp_path / "tenki-cli.yaml"
    tenki_config.write_text("auth_token: tok-secret\n", encoding="utf-8")
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tenki_config))

    from hermes_cli import config as hermes_config
    from hermes_constants import (
        reset_hermes_home_override,
        set_hermes_home_override,
    )
    from tools import credential_files
    from tools.environments.tenki import TenkiEnvironment

    home_a = tmp_path / "profiles" / "a"
    home_b = tmp_path / "profiles" / "b"
    home_a.mkdir(parents=True)
    home_b.mkdir(parents=True)
    (home_a / "service.json").write_text("profile-a", encoding="utf-8")
    (home_a / "config.yaml").write_text(
        "terminal:\n  credential_files:\n    - service.json\n",
        encoding="utf-8",
    )
    (home_b / "config.yaml").write_text(
        "terminal:\n  credential_files: []\n",
        encoding="utf-8",
    )
    credential_files._config_files.clear()
    hermes_config._LOAD_CONFIG_CACHE.clear()
    hermes_config._RAW_CONFIG_CACHE.clear()
    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)

    token = set_hermes_home_override(home_a)
    try:
        env_a = TenkiEnvironment(task_id="sync-a", sync_hermes_home=True)
    finally:
        reset_hermes_home_override(token)
    token = set_hermes_home_override(home_b)
    try:
        env_b = TenkiEnvironment(task_id="sync-b", sync_hermes_home=True)
    finally:
        reset_hermes_home_override(token)

    files_a = env_a._sync_manager._get_files_fn()
    files_b = env_b._sync_manager._get_files_fn()
    assert any(host == str(home_a / "service.json") for host, _remote in files_a)
    assert all(host != str(home_a / "service.json") for host, _remote in files_b)

    sync_back_homes = []
    monkeypatch.setattr(
        env_a._sync_manager,
        "sync_back",
        lambda home=None: sync_back_homes.append(home),
    )
    monkeypatch.setattr(
        env_b._sync_manager,
        "sync_back",
        lambda home=None: sync_back_homes.append(home),
    )
    first = threading.Thread(target=env_a.cleanup)
    second = threading.Thread(target=env_b.cleanup)
    first.start()
    second.start()
    first.join(timeout=2)
    second.join(timeout=2)

    assert sync_back_homes == [home_a, home_b] or sync_back_homes == [home_b, home_a]
    assert process_home not in sync_back_homes


def test_tenki_snapshot_pointer_updates_are_serialized_and_atomic(
    monkeypatch,
    tmp_path,
):
    from tools.environments import tenki as tenki_module

    store = tmp_path / "tenki_snapshots.json"
    active_loads = 0
    max_active_loads = 0
    state_lock = threading.Lock()
    original_load = tenki_module._load_recovery_registry

    def tracked_load(path):
        nonlocal active_loads, max_active_loads
        with state_lock:
            active_loads += 1
            max_active_loads = max(max_active_loads, active_loads)
        time.sleep(0.02)
        try:
            return original_load(path)
        finally:
            with state_lock:
                active_loads -= 1

    monkeypatch.setattr(tenki_module, "_load_recovery_registry", tracked_load)
    start = threading.Barrier(3)

    def save(task_id, snapshot_id):
        start.wait()
        tenki_module._store_snapshot(task_id, snapshot_id, store)

    first = threading.Thread(target=save, args=("task-a", "snap-a"))
    second = threading.Thread(target=save, args=("task-b", "snap-b"))
    first.start()
    second.start()
    start.wait()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert max_active_loads == 1
    assert tenki_module._load_snapshots(store) == {
        "direct:task-a": "snap-a",
        "direct:task-b": "snap-b",
    }
    assert list(tmp_path.glob(".tenki_snapshots.json.*.tmp")) == []


def test_tenki_malformed_recovery_registry_fails_closed(
    monkeypatch,
    tmp_path,
):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text(
        "auth_token: tok-secret\n",
        encoding="utf-8",
    )
    store = tmp_path / "tenki_snapshots.json"
    malformed = '{"direct:task":"sole-snapshot"'
    store.write_text(malformed, encoding="utf-8")

    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)

    with pytest.raises(RuntimeError, match="recovery registry is unreadable"):
        TenkiEnvironment(
            task_id="task",
            persistent_filesystem=True,
        )

    assert store.read_text(encoding="utf-8") == malformed
    assert not _FakeSandboxFactory.created_kwargs


def test_tenki_snapshot_store_uses_windows_cross_process_lock(
    monkeypatch,
    tmp_path,
):
    from tools.environments import tenki as tenki_module

    lock_calls = []
    fake_msvcrt = SimpleNamespace(
        LK_LOCK=1,
        LK_UNLCK=2,
        locking=lambda fd, mode, size: lock_calls.append((fd, mode, size)),
    )
    store = tmp_path / "tenki_snapshots.json"
    monkeypatch.setattr(tenki_module, "_fcntl", None)
    monkeypatch.setattr(tenki_module, "_msvcrt", fake_msvcrt)

    tenki_module._store_snapshot("task", "snap", store)

    assert [mode for _fd, mode, _size in lock_calls] == [1, 2]
    assert all(size == 1 for _fd, _mode, size in lock_calls)
    assert tenki_module._get_snapshot_restore_candidate("task", store) == (
        "snap",
        False,
    )


def test_tenki_snapshot_store_uses_windows_write_through_replace_and_confirm(
    monkeypatch,
    tmp_path,
):
    from tools.environments import tenki as tenki_module

    store = tmp_path / "tenki_snapshots.json"
    replacements = []

    def write_through_replace(source, destination):
        replacements.append((source, destination))
        tenki_module.os.replace(source, destination)

    monkeypatch.setattr(tenki_module, "_snapshot_platform", lambda: "nt")
    monkeypatch.setattr(
        tenki_module,
        "_windows_replace_file_write_through",
        write_through_replace,
    )

    tenki_module._store_snapshot("task", "snap", store)
    assert len(replacements) == 1
    assert tenki_module._get_snapshot_restore_candidate("task", store) == (
        "snap",
        False,
    )

    tenki_module._confirm_snapshot_store_durable(store)
    assert len(replacements) == 2
    assert tenki_module._get_snapshot_restore_candidate("task", store) == (
        "snap",
        False,
    )


def test_tenki_windows_replace_requests_movefile_write_through(
    monkeypatch,
    tmp_path,
):
    import ctypes

    from tools.environments import tenki as tenki_module

    calls = []

    class FakeMoveFileEx:
        argtypes = None
        restype = None

        def __call__(self, source, destination, flags):
            calls.append((source, destination, flags))
            return True

    fake_move = FakeMoveFileEx()
    fake_kernel = SimpleNamespace(MoveFileExW=fake_move)
    monkeypatch.setattr(
        ctypes,
        "WinDLL",
        lambda *_args, **_kwargs: fake_kernel,
        raising=False,
    )

    source = str(tmp_path / "source.json")
    destination = tmp_path / "destination.json"
    tenki_module._windows_replace_file_write_through(
        source,
        destination,
    )

    assert calls == [
        (
            tenki_module.os.path.abspath(source),
            tenki_module.os.path.abspath(destination),
            0x1 | 0x8,
        )
    ]


def test_tenki_snapshot_store_fails_closed_without_os_file_lock(
    monkeypatch,
    tmp_path,
):
    from tools.environments import tenki as tenki_module

    store = tmp_path / "tenki_snapshots.json"
    monkeypatch.setattr(tenki_module, "_fcntl", None)
    monkeypatch.setattr(tenki_module, "_msvcrt", None)

    with pytest.raises(RuntimeError, match="requires fcntl or msvcrt"):
        tenki_module._store_snapshot("task", "snap", store)

    assert store.exists() is False


def test_tenki_task_ownership_lock_spans_wrapper_lifetime(
    monkeypatch,
    tmp_path,
):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text(
        "auth_token: tok-secret\n",
        encoding="utf-8",
    )

    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    owner = TenkiEnvironment(
        task_id="owned-task",
        persistent_filesystem=True,
    )

    with pytest.raises(RuntimeError, match="already active"):
        TenkiEnvironment(
            task_id="owned-task",
            persistent_filesystem=True,
        )

    assert len(_FakeSandboxFactory.created_kwargs) == 1
    owner.cleanup()

    successor = TenkiEnvironment(
        task_id="owned-task",
        persistent_filesystem=True,
    )
    assert len(_FakeSandboxFactory.created_kwargs) == 2
    successor.cleanup()


def test_tenki_failed_initialization_quiesces_before_releasing_ownership(
    monkeypatch,
    tmp_path,
):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text(
        "auth_token: tok-secret\n",
        encoding="utf-8",
    )

    from tools.environments.tenki import TenkiEnvironment

    partial_wrappers = []

    def fail_init_session(self):
        partial_wrappers.append(self)
        raise RuntimeError("session initialization failed")

    monkeypatch.setattr(TenkiEnvironment, "init_session", fail_init_session)
    with pytest.raises(RuntimeError, match="session initialization failed"):
        TenkiEnvironment(
            task_id="failed-init",
            persistent_filesystem=True,
        )

    partial = partial_wrappers[0]
    sandbox = _FakeSandboxFactory.sandboxes[0]
    assert sandbox.paused is True
    assert sandbox.terminated is False
    assert partial._sandbox is None
    assert partial._cleanup_complete is True

    # A successor can safely resume the exact quiesced remote. Delayed cleanup
    # of the failed wrapper is inert and cannot terminate it underneath the
    # successor.
    _FakeClient.listed_sandboxes = [sandbox]
    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    successor = TenkiEnvironment(
        task_id="failed-init",
        persistent_filesystem=True,
    )
    assert successor._sandbox is sandbox
    partial.cleanup()
    assert sandbox.terminated is False
    successor.cleanup()


def test_tenki_failed_initialization_quarantines_unsafe_task_lock(
    monkeypatch,
    tmp_path,
):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text(
        "auth_token: tok-secret\n",
        encoding="utf-8",
    )

    from tools.environments import tenki as tenki_module
    from tools.environments.tenki import TenkiEnvironment

    partial_wrappers = []
    quarantine_start = len(
        tenki_module._QUARANTINED_TASK_OWNERSHIP_FILES
    )

    def fail_pause(self):
        raise RuntimeError("pause unavailable")

    def fail_init_session(self):
        partial_wrappers.append(self)
        raise RuntimeError("session initialization failed")

    monkeypatch.setattr(_FakeSandbox, "pause", fail_pause)
    monkeypatch.setattr(tenki_module, "_TERMINATE_RETRY_DELAYS", ())
    monkeypatch.setattr(TenkiEnvironment, "init_session", fail_init_session)
    with pytest.raises(RuntimeError, match="session initialization failed"):
        TenkiEnvironment(
            task_id="unsafe-failed-init",
            persistent_filesystem=True,
        )

    partial = partial_wrappers[0]
    sandbox = _FakeSandboxFactory.sandboxes[0]
    assert sandbox.terminated is False
    assert partial._cleanup_complete is True
    with pytest.raises(RuntimeError, match="already active"):
        TenkiEnvironment(
            task_id="unsafe-failed-init",
            persistent_filesystem=True,
        )
    assert len(_FakeSandboxFactory.created_kwargs) == 1
    partial.cleanup()
    assert sandbox.terminated is False

    # Release this test's fail-closed quarantine explicitly; production keeps
    # it until process exit.
    with tenki_module._QUARANTINED_TASK_OWNERSHIP_GUARD:
        quarantined = tenki_module._QUARANTINED_TASK_OWNERSHIP_FILES[
            quarantine_start:
        ]
        del tenki_module._QUARANTINED_TASK_OWNERSHIP_FILES[
            quarantine_start:
        ]
    for lock_file in quarantined:
        tenki_module._release_task_ownership_lock(lock_file)


def test_tenki_ambiguous_create_is_reconciled_before_ownership_release(
    monkeypatch,
    tmp_path,
):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text(
        "auth_token: tok-secret\n",
        encoding="utf-8",
    )

    from tools.environments import tenki as tenki_module
    from tools.environments.tenki import TenkiEnvironment

    original_create = _FakeClient.create

    def commit_then_timeout(self, **kwargs):
        sandbox = original_create(self, **kwargs)
        _FakeClient.listed_sandboxes = [sandbox]
        raise TimeoutError("response lost after commit")

    monkeypatch.setattr(_FakeClient, "create", commit_then_timeout)
    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    quarantine_start = len(
        tenki_module._QUARANTINED_TASK_OWNERSHIP_FILES
    )

    with pytest.raises(TimeoutError, match="response lost"):
        TenkiEnvironment(task_id="ambiguous-create")

    first_remote = _FakeSandboxFactory.sandboxes[0]
    assert first_remote.terminated is True
    assert len(
        tenki_module._QUARANTINED_TASK_OWNERSHIP_FILES
    ) == quarantine_start

    monkeypatch.setattr(_FakeClient, "create", original_create)
    successor = TenkiEnvironment(task_id="ambiguous-create")
    assert successor._sandbox is not first_remote
    assert len(_FakeSandboxFactory.created_kwargs) == 2
    successor.cleanup()


def test_tenki_unreconciled_ambiguous_create_quarantines_task_lock(
    monkeypatch,
    tmp_path,
):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text(
        "auth_token: tok-secret\n",
        encoding="utf-8",
    )

    from tools.environments import tenki as tenki_module
    from tools.environments.tenki import TenkiEnvironment

    quarantine_start = len(
        tenki_module._QUARANTINED_TASK_OWNERSHIP_FILES
    )

    def timeout_without_visible_remote(self, **_kwargs):
        raise TimeoutError("ambiguous create")

    monkeypatch.setattr(
        _FakeClient,
        "create",
        timeout_without_visible_remote,
    )
    with pytest.raises(TimeoutError, match="ambiguous create"):
        TenkiEnvironment(task_id="unreconciled-create")

    with pytest.raises(RuntimeError, match="already active"):
        TenkiEnvironment(task_id="unreconciled-create")
    assert _FakeSandboxFactory.created_kwargs == []

    with tenki_module._QUARANTINED_TASK_OWNERSHIP_GUARD:
        quarantined = tenki_module._QUARANTINED_TASK_OWNERSHIP_FILES[
            quarantine_start:
        ]
        del tenki_module._QUARANTINED_TASK_OWNERSHIP_FILES[
            quarantine_start:
        ]
    for lock_file in quarantined:
        tenki_module._release_task_ownership_lock(lock_file)


def test_tenki_ambiguous_persistent_lineages_are_never_terminated(
    monkeypatch,
    tmp_path,
):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text(
        "auth_token: tok-secret\n",
        encoding="utf-8",
    )

    from tools.environments import tenki as tenki_module
    from tools.environments.tenki import TenkiEnvironment

    token = tenki_module._profile_token()
    task_id = "ambiguous-persistent"
    old_remote = _FakeSandbox(
        name=f"hermes-{token}-{task_id}",
        metadata={
            "hermes_task_id": task_id,
            "hermes_profile": token,
        },
    )
    original_create = _FakeClient.create
    list_calls = 0

    def eventually_consistent_list(
        self,
        *,
        workspace_id=None,
        tags=None,
        sticky=None,
    ):
        nonlocal list_calls
        list_calls += 1
        if list_calls == 1:
            return []
        return [old_remote, *_FakeSandboxFactory.sandboxes]

    def commit_then_timeout(self, **kwargs):
        original_create(self, **kwargs)
        raise TimeoutError("response lost after duplicate commit")

    monkeypatch.setattr(_FakeClient, "list", eventually_consistent_list)
    monkeypatch.setattr(_FakeClient, "create", commit_then_timeout)
    quarantine_start = len(
        tenki_module._QUARANTINED_TASK_OWNERSHIP_FILES
    )

    with pytest.raises(TimeoutError, match="duplicate commit"):
        TenkiEnvironment(
            task_id=task_id,
            persistent_filesystem=True,
        )

    new_remote = _FakeSandboxFactory.sandboxes[0]
    _FakeClient.remote_sandboxes = [old_remote]
    assert old_remote.terminated is False
    assert new_remote.terminated is False
    attempt_id = new_remote.info.metadata["hermes_create_attempt"]
    assert tenki_module._get_create_attempt(task_id) is None
    assert tenki_module._remote_binding_state(task_id) == tenki_module._RemoteBinding(
        min(old_remote.id, new_remote.id),
        attempt_id,
        False,
        True,
        tuple(sorted((old_remote.id, new_remote.id))),
        False,
    )
    assert len(_FakeSandboxFactory.created_kwargs) == 1

    # Simulate process exit releasing only the kernel lock. The durable marker
    # makes the next owner reconcile the same exact attempt; both lineages are
    # still left untouched and no third branch is created.
    with tenki_module._QUARANTINED_TASK_OWNERSHIP_GUARD:
        first_quarantine = tenki_module._QUARANTINED_TASK_OWNERSHIP_FILES[
            quarantine_start:
        ]
        del tenki_module._QUARANTINED_TASK_OWNERSHIP_FILES[
            quarantine_start:
        ]
    for lock_file in first_quarantine:
        tenki_module._release_task_ownership_lock(lock_file)

    monkeypatch.setattr(_FakeClient, "list", lambda self, **_kwargs: [])
    with pytest.raises(RuntimeError, match="durable persistent-lineage conflict"):
        TenkiEnvironment(
            task_id=task_id,
            persistent_filesystem=True,
        )
    assert len(_FakeSandboxFactory.created_kwargs) == 1
    assert old_remote.terminated is False
    assert new_remote.terminated is False

    with tenki_module._QUARANTINED_TASK_OWNERSHIP_GUARD:
        second_quarantine = tenki_module._QUARANTINED_TASK_OWNERSHIP_FILES[
            quarantine_start:
        ]
        del tenki_module._QUARANTINED_TASK_OWNERSHIP_FILES[
            quarantine_start:
        ]
    for lock_file in second_quarantine:
        tenki_module._release_task_ownership_lock(lock_file)


def test_tenki_unmanaged_collision_remains_durable_when_list_omits_remote(
    monkeypatch,
    tmp_path,
):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text(
        "auth_token: tok-secret\n",
        encoding="utf-8",
    )

    from tools.environments import tenki as tenki_module
    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    task_id = "pre-binding-persistent"
    token = tenki_module._profile_token()
    existing = _FakeSandbox(
        sandbox_id="sb-existing",
        name=f"hermes-{token}-{task_id}",
        metadata={
            "hermes_task_id": task_id,
            "hermes_profile": token,
        },
    )
    _FakeClient.listed_sandboxes = [existing]

    quarantine_start = len(
        tenki_module._QUARANTINED_TASK_OWNERSHIP_FILES
    )
    with pytest.raises(RuntimeError, match="unmanaged persistent sandbox"):
        TenkiEnvironment(
            task_id=task_id,
            persistent_filesystem=True,
        )
    assert tenki_module._get_remote_binding(task_id) == existing.id
    assert not _FakeSandboxFactory.created_kwargs
    _release_quarantined_locks_since(tenki_module, quarantine_start)

    # A later eventual-consistency omission cannot turn the collision into an
    # ownership claim or authorize a create.
    _FakeClient.listed_sandboxes = []

    with pytest.raises(RuntimeError, match="durable persistent-lineage conflict"):
        TenkiEnvironment(
            task_id=task_id,
            persistent_filesystem=True,
        )
    assert not _FakeSandboxFactory.created_kwargs
    assert existing.terminated is False
    _release_quarantined_locks_since(tenki_module, quarantine_start)


def test_tenki_known_conflict_clears_only_after_exact_remote_is_terminal(
    monkeypatch,
    tmp_path,
):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text(
        "auth_token: tok-secret\n",
        encoding="utf-8",
    )

    from tools.environments import tenki as tenki_module
    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    task_id = "resolved-unmanaged-conflict"
    token = tenki_module._profile_token()
    existing = _FakeSandbox(
        sandbox_id="sb-existing",
        name=f"hermes-{token}-{task_id}",
        metadata={
            "hermes_task_id": task_id,
            "hermes_profile": token,
        },
    )
    _FakeClient.listed_sandboxes = [existing]
    quarantine_start = len(
        tenki_module._QUARANTINED_TASK_OWNERSHIP_FILES
    )

    with pytest.raises(RuntimeError, match="unmanaged persistent sandbox"):
        TenkiEnvironment(task_id=task_id, persistent_filesystem=True)
    _release_quarantined_locks_since(tenki_module, quarantine_start)

    existing.terminate()
    _FakeClient.listed_sandboxes = []
    successor = TenkiEnvironment(
        task_id=task_id,
        persistent_filesystem=True,
    )

    assert successor._sandbox is not existing
    assert len(_FakeSandboxFactory.created_kwargs) == 1
    _binding = tenki_module._remote_binding_state(task_id)
    assert (_binding.validated, _binding.conflicted) == (True, False)
    successor.cleanup()


def test_tenki_terminal_binding_is_authoritatively_cleared_after_crash(
    monkeypatch,
    tmp_path,
):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text(
        "auth_token: tok-secret\n",
        encoding="utf-8",
    )

    from tools.environments import tenki as tenki_module
    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    task_id = "terminated-binding-crash"
    first = TenkiEnvironment(task_id=task_id)
    terminated = first._sandbox
    assert tenki_module._get_remote_binding(task_id) == terminated.id

    # The server-side termination commits, then the process dies before the
    # local binding removal. A successor proves terminal state with get(id),
    # clears that exact binding, and creates one replacement.
    terminated.terminate()
    first._release_task_ownership()
    first._sandbox = None
    first._client = None
    first._cleanup_complete = True

    successor = TenkiEnvironment(task_id=task_id)
    assert successor._sandbox is not terminated
    assert len(_FakeSandboxFactory.created_kwargs) == 2
    assert tenki_module._get_remote_binding(task_id) == successor._sandbox.id
    successor.cleanup()


def test_tenki_local_create_rejection_clears_attempt_before_retry(
    monkeypatch,
    tmp_path,
):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text(
        "auth_token: tok-secret\n",
        encoding="utf-8",
    )

    from tools.environments import tenki as tenki_module
    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    task_id = "local-create-rejection"
    original_create_kwargs = TenkiEnvironment._create_kwargs

    def invalid_create_kwargs(self):
        kwargs = original_create_kwargs(self)
        kwargs["removed_sdk_kwarg"] = True
        return kwargs

    monkeypatch.setattr(
        TenkiEnvironment,
        "_create_kwargs",
        invalid_create_kwargs,
    )
    with pytest.raises(TypeError, match="do not match the installed SDK"):
        TenkiEnvironment(task_id=task_id)

    assert tenki_module._get_create_attempt(task_id) is None
    assert tenki_module._get_remote_binding(task_id) is None
    assert not _FakeSandboxFactory.sandboxes

    monkeypatch.setattr(
        TenkiEnvironment,
        "_create_kwargs",
        original_create_kwargs,
    )
    successor = TenkiEnvironment(task_id=task_id)
    assert len(_FakeSandboxFactory.sandboxes) == 1
    successor.cleanup()


def test_tenki_post_commit_value_error_retains_attempt_and_blocks_duplicate(
    monkeypatch,
    tmp_path,
):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text(
        "auth_token: tok-secret\n",
        encoding="utf-8",
    )

    from tools.environments import tenki as tenki_module
    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    task_id = "post-commit-value-error"
    original_create = _FakeClient.create

    def commit_then_value_error(self, **kwargs):
        original_create(self, **kwargs)
        raise ValueError("response decoding failed after commit")

    monkeypatch.setattr(_FakeClient, "create", commit_then_value_error)
    monkeypatch.setattr(_FakeClient, "list", lambda self, **_kwargs: [])
    quarantine_start = len(
        tenki_module._QUARANTINED_TASK_OWNERSHIP_FILES
    )

    with pytest.raises(ValueError, match="response decoding failed"):
        TenkiEnvironment(
            task_id=task_id,
            persistent_filesystem=True,
        )

    committed = _FakeSandboxFactory.sandboxes[0]
    attempt_id = committed.info.metadata["hermes_create_attempt"]
    assert tenki_module._get_create_attempt(task_id) == attempt_id
    assert tenki_module._get_remote_binding(task_id) is None

    with tenki_module._QUARANTINED_TASK_OWNERSHIP_GUARD:
        first_quarantine = tenki_module._QUARANTINED_TASK_OWNERSHIP_FILES[
            quarantine_start:
        ]
        del tenki_module._QUARANTINED_TASK_OWNERSHIP_FILES[
            quarantine_start:
        ]
    for lock_file in first_quarantine:
        tenki_module._release_task_ownership_lock(lock_file)

    with pytest.raises(RuntimeError, match="prior create is unresolved"):
        TenkiEnvironment(
            task_id=task_id,
            persistent_filesystem=True,
        )
    assert len(_FakeSandboxFactory.created_kwargs) == 1
    assert committed.terminated is False

    with tenki_module._QUARANTINED_TASK_OWNERSHIP_GUARD:
        second_quarantine = tenki_module._QUARANTINED_TASK_OWNERSHIP_FILES[
            quarantine_start:
        ]
        del tenki_module._QUARANTINED_TASK_OWNERSHIP_FILES[
            quarantine_start:
        ]
    for lock_file in second_quarantine:
        tenki_module._release_task_ownership_lock(lock_file)


def test_tenki_expired_empty_attempt_cannot_lock_task_forever(
    monkeypatch,
    tmp_path,
):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text(
        "auth_token: tok-secret\n",
        encoding="utf-8",
    )

    from tools.environments import tenki as tenki_module
    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    task_id = "expired-empty-attempt"
    tenki_module._begin_create_attempt(
        task_id,
        "pre-rpc-crash",
        time.time() - 1,
        tmp_path / "tenki_snapshots.json",
    )

    env = TenkiEnvironment(task_id=task_id, max_duration=0)

    assert tenki_module._get_create_attempt(task_id) is None
    assert len(_FakeSandboxFactory.created_kwargs) == 1
    # A positive server-side lifetime is what makes eventual expiry
    # authoritative even when a crash happened immediately before the RPC.
    assert _FakeSandboxFactory.created_kwargs[0]["max_duration"] == 3600
    env.cleanup()


def test_tenki_terminal_list_row_requires_authoritative_get_before_clear(
    monkeypatch,
    tmp_path,
):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text(
        "auth_token: tok-secret\n",
        encoding="utf-8",
    )

    from tools.environments import tenki as tenki_module
    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    task_id = "stale-terminal-list-row"
    attempt_id = "expected-attempt"
    token = tenki_module._profile_token()
    metadata = {
        "hermes_backend": "tenki",
        "hermes_task_id": task_id,
        "hermes_profile": token,
        "hermes_create_attempt": attempt_id,
    }
    listed = _FakeSandbox(
        sandbox_id="sb-known",
        name=f"hermes-{token}-{task_id}",
        state="TERMINATED",
        metadata=metadata,
    )
    authoritative = _FakeSandbox(
        sandbox_id="sb-known",
        name=f"hermes-{token}-{task_id}",
        state="RUNNING",
        metadata=metadata,
    )
    _FakeClient.listed_sandboxes = [listed]
    _FakeClient.remote_sandboxes = [authoritative]
    tenki_module._begin_create_attempt(
        task_id,
        attempt_id,
        time.time() + 7200,
    )

    env = TenkiEnvironment(
        task_id=task_id,
        persistent_filesystem=True,
    )

    assert env._sandbox is authoritative
    assert len(_FakeSandboxFactory.created_kwargs) == 0
    _binding = tenki_module._remote_binding_state(task_id)
    assert (_binding.validated, _binding.conflicted) == (True, False)
    env.cleanup()


def test_tenki_expired_attempt_never_adopts_unmanaged_match(
    monkeypatch,
    tmp_path,
):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text(
        "auth_token: tok-secret\n",
        encoding="utf-8",
    )

    from tools.environments import tenki as tenki_module
    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    task_id = "expired-attempt-unmanaged"
    token = tenki_module._profile_token()
    unmanaged = _FakeSandbox(
        sandbox_id="external-id",
        name=f"hermes-{token}-{task_id}",
        metadata={
            "hermes_task_id": task_id,
            "hermes_profile": token,
        },
    )
    _FakeClient.listed_sandboxes = [unmanaged]
    tenki_module._begin_create_attempt(
        task_id,
        "expired-attempt",
        time.time() - 1,
    )
    quarantine_start = len(
        tenki_module._QUARANTINED_TASK_OWNERSHIP_FILES
    )

    with pytest.raises(RuntimeError, match="prior create is unresolved"):
        TenkiEnvironment(
            task_id=task_id,
            persistent_filesystem=True,
        )

    assert tenki_module._get_create_attempt(task_id) is None
    assert tenki_module._remote_binding_state(task_id) == tenki_module._RemoteBinding(
        unmanaged.id,
        "expired-attempt",
        False,
        True,
        (unmanaged.id,),
        False,
    )
    assert not _FakeSandboxFactory.created_kwargs
    assert unmanaged.resumed is False
    _release_quarantined_locks_since(tenki_module, quarantine_start)

    _FakeClient.listed_sandboxes = []
    with pytest.raises(RuntimeError, match="durable persistent-lineage conflict"):
        TenkiEnvironment(
            task_id=task_id,
            persistent_filesystem=True,
        )
    assert not _FakeSandboxFactory.created_kwargs
    assert unmanaged.resumed is False
    _release_quarantined_locks_since(tenki_module, quarantine_start)


def test_tenki_hidden_exact_conflict_preserves_visible_known_id(
    monkeypatch,
    tmp_path,
):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text(
        "auth_token: tok-secret\n",
        encoding="utf-8",
    )

    from tools.environments import tenki as tenki_module
    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    task_id = "hidden-exact-visible-other"
    token = tenki_module._profile_token()
    visible = _FakeSandbox(
        sandbox_id="visible-known",
        name=f"hermes-{token}-{task_id}",
        metadata={
            "hermes_task_id": task_id,
            "hermes_profile": token,
        },
    )
    _FakeClient.listed_sandboxes = [visible]
    tenki_module._begin_create_attempt(
        task_id,
        "hidden-exact-attempt",
        time.time() + 7200,
    )
    quarantine_start = len(
        tenki_module._QUARANTINED_TASK_OWNERSHIP_FILES
    )

    with pytest.raises(RuntimeError, match="prior create is unresolved"):
        TenkiEnvironment(task_id=task_id, persistent_filesystem=True)

    assert tenki_module._remote_binding_state(task_id) == tenki_module._RemoteBinding(
        visible.id,
        "hidden-exact-attempt",
        False,
        True,
        (visible.id,),
        True,
    )
    assert not _FakeSandboxFactory.created_kwargs
    _release_quarantined_locks_since(tenki_module, quarantine_start)


def test_tenki_unvalidated_binding_requires_durable_expected_attempt(
    monkeypatch,
    tmp_path,
):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text(
        "auth_token: tok-secret\n",
        encoding="utf-8",
    )

    from tools.environments import tenki as tenki_module
    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    task_id = "mismatched-bound-attempt"
    token = tenki_module._profile_token()
    remote = _FakeSandbox(
        sandbox_id="sb-bound",
        name=f"hermes-{token}-{task_id}",
        metadata={
            "hermes_backend": "tenki",
            "hermes_task_id": task_id,
            "hermes_profile": token,
            "hermes_create_attempt": "attacker-attempt",
        },
    )
    _FakeClient.remote_sandboxes = [remote]
    tenki_module._begin_create_attempt(
        task_id,
        "expected-attempt",
        time.time() + 7200,
    )
    tenki_module._store_remote_binding(
        task_id,
        remote.id,
        "expected-attempt",
        validated=False,
    )
    quarantine_start = len(
        tenki_module._QUARANTINED_TASK_OWNERSHIP_FILES
    )

    with pytest.raises(RuntimeError, match="cannot validate the new lineage"):
        TenkiEnvironment(
            task_id=task_id,
            persistent_filesystem=True,
        )

    assert tenki_module._remote_binding_state(task_id) == tenki_module._RemoteBinding(
        remote.id,
        "expected-attempt",
        False,
        False,
        (),
        False,
    )
    assert not _FakeSandboxFactory.created_kwargs
    assert remote.resumed is False
    _release_quarantined_locks_since(tenki_module, quarantine_start)


def test_tenki_transient_bound_id_lookup_never_falls_back_to_create(
    monkeypatch,
    tmp_path,
):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text(
        "auth_token: tok-secret\n",
        encoding="utf-8",
    )

    from tools.environments import tenki as tenki_module
    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    task_id = "transient-bound-get"
    first = TenkiEnvironment(
        task_id=task_id,
        persistent_filesystem=True,
    )
    remote = first._sandbox
    first._release_task_ownership()
    first._sandbox = None
    first._client = None
    first._cleanup_complete = True
    original_get = _FakeClient.get

    def fail_get(self, sandbox_id):
        raise TimeoutError(f"lookup timed out for {sandbox_id}")

    monkeypatch.setattr(_FakeClient, "get", fail_get)
    with pytest.raises(RuntimeError, match="could not resolve bound remote"):
        TenkiEnvironment(
            task_id=task_id,
            persistent_filesystem=True,
        )
    assert len(_FakeSandboxFactory.created_kwargs) == 1
    assert tenki_module._get_remote_binding(task_id) == remote.id

    monkeypatch.setattr(_FakeClient, "get", original_get)
    successor = TenkiEnvironment(
        task_id=task_id,
        persistent_filesystem=True,
    )
    assert successor._sandbox is remote
    assert len(_FakeSandboxFactory.created_kwargs) == 1
    successor.cleanup()


def test_tenki_unvalidated_binding_stays_fail_closed_when_list_omits_remote(
    monkeypatch,
    tmp_path,
):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text(
        "auth_token: tok-secret\n",
        encoding="utf-8",
    )

    from tools.environments import tenki as tenki_module
    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    task_id = "unvalidated-binding"
    token = tenki_module._profile_token()
    older = _FakeSandbox(
        sandbox_id="sb-older",
        name=f"hermes-{token}-{task_id}",
        metadata={
            "hermes_task_id": task_id,
            "hermes_profile": token,
        },
    )
    _FakeClient.remote_sandboxes = [older]
    list_calls = 0

    def reveal_fork_after_create(
        self,
        *,
        workspace_id=None,
        tags=None,
        sticky=None,
    ):
        nonlocal list_calls
        list_calls += 1
        if list_calls == 1:
            return []
        return [older, *_FakeSandboxFactory.sandboxes]

    monkeypatch.setattr(_FakeClient, "list", reveal_fork_after_create)
    quarantine_start = len(
        tenki_module._QUARANTINED_TASK_OWNERSHIP_FILES
    )
    with pytest.raises(RuntimeError, match="multiple active sandbox lineages"):
        TenkiEnvironment(
            task_id=task_id,
            persistent_filesystem=True,
        )

    created = _FakeSandboxFactory.sandboxes[0]
    assert tenki_module._get_create_attempt(task_id) is None
    assert tenki_module._remote_binding_state(task_id) == tenki_module._RemoteBinding(
        created.id,
        created.info.metadata["hermes_create_attempt"],
        False,
        True,
        tuple(sorted((created.id, older.id))),
        False,
    )
    assert older.terminated is False
    assert created.terminated is False

    with tenki_module._QUARANTINED_TASK_OWNERSHIP_GUARD:
        first_quarantine = tenki_module._QUARANTINED_TASK_OWNERSHIP_FILES[
            quarantine_start:
        ]
        del tenki_module._QUARANTINED_TASK_OWNERSHIP_FILES[
            quarantine_start:
        ]
    for lock_file in first_quarantine:
        tenki_module._release_task_ownership_lock(lock_file)

    # Even an empty eventual-consistency result cannot convert the exact,
    # pending binding into absence or authorize a third create.
    monkeypatch.setattr(_FakeClient, "list", lambda self, **_kwargs: [])
    with pytest.raises(RuntimeError, match="durable persistent-lineage conflict"):
        TenkiEnvironment(
            task_id=task_id,
            persistent_filesystem=True,
        )
    assert len(_FakeSandboxFactory.created_kwargs) == 1
    assert older.terminated is False
    assert created.terminated is False

    with tenki_module._QUARANTINED_TASK_OWNERSHIP_GUARD:
        second_quarantine = tenki_module._QUARANTINED_TASK_OWNERSHIP_FILES[
            quarantine_start:
        ]
        del tenki_module._QUARANTINED_TASK_OWNERSHIP_FILES[
            quarantine_start:
        ]
    for lock_file in second_quarantine:
        tenki_module._release_task_ownership_lock(lock_file)


def test_tenki_post_create_mixed_conflict_preserves_known_ids(
    monkeypatch,
    tmp_path,
):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text(
        "auth_token: tok-secret\n",
        encoding="utf-8",
    )

    from tools.environments import tenki as tenki_module
    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    task_id = "post-create-mixed-conflict"
    token = tenki_module._profile_token()
    metadata = {
        "hermes_task_id": task_id,
        "hermes_profile": token,
    }
    known = _FakeSandbox(
        sandbox_id="known-other",
        name=f"hermes-{token}-{task_id}",
        metadata=metadata,
    )
    unidentified = _FakeSandbox(
        sandbox_id=None,
        name=f"hermes-{token}-{task_id}",
        metadata=metadata,
    )
    list_calls = 0

    def reveal_mixed_conflict(self, **_kwargs):
        nonlocal list_calls
        list_calls += 1
        if list_calls == 1:
            return []
        return [*_FakeSandboxFactory.sandboxes, known, unidentified]

    monkeypatch.setattr(_FakeClient, "list", reveal_mixed_conflict)
    quarantine_start = len(
        tenki_module._QUARANTINED_TASK_OWNERSHIP_FILES
    )

    with pytest.raises(RuntimeError, match="multiple active sandbox lineages"):
        TenkiEnvironment(task_id=task_id, persistent_filesystem=True)

    created = _FakeSandboxFactory.sandboxes[0]
    assert tenki_module._remote_binding_state(task_id) == tenki_module._RemoteBinding(
        created.id,
        created.info.metadata["hermes_create_attempt"],
        False,
        True,
        tuple(sorted((created.id, known.id))),
        True,
    )
    assert not created.terminated
    assert not known.terminated
    _release_quarantined_locks_since(tenki_module, quarantine_start)


def test_tenki_runtime_recreate_reconciles_ambiguous_commit_before_retry(
    monkeypatch,
    tmp_path,
):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text(
        "auth_token: tok-secret\n",
        encoding="utf-8",
    )

    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    env = TenkiEnvironment(task_id="runtime-ambiguous-create")
    original_remote = env._sandbox
    original_remote.terminate()
    env._sandbox = None
    assert env._clear_create_attempt_marker() is True
    original_create = _FakeClient.create

    def commit_then_timeout(self, **kwargs):
        sandbox = original_create(self, **kwargs)
        _FakeClient.listed_sandboxes = [sandbox]
        raise TimeoutError("runtime create response lost")

    monkeypatch.setattr(_FakeClient, "create", commit_then_timeout)
    with pytest.raises(TimeoutError, match="response lost"):
        env._ensure_sandbox()

    ambiguous_remote = _FakeSandboxFactory.sandboxes[-1]
    assert ambiguous_remote is not original_remote
    assert ambiguous_remote.terminated is True
    assert env._create_outcome_uncertain is False

    monkeypatch.setattr(_FakeClient, "create", original_create)
    env._ensure_sandbox()
    assert env._sandbox not in (original_remote, ambiguous_remote)
    env.cleanup()


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


def test_tenki_environment_passes_sdk_resource_boundaries(monkeypatch, tmp_path):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text("auth_token: tok-secret\n", encoding="utf-8")

    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    env = TenkiEnvironment(
        task_id="resource-boundaries",
        cpu=16,
        memory=65_536,
        disk=100 * 1024,
    )

    kwargs = _FakeSandboxFactory.created_kwargs[0]
    assert kwargs["cpu_cores"] == 16
    assert kwargs["memory_mb"] == 65_536
    assert kwargs["disk_size_gb"] == 100
    env.cleanup()


def test_tenki_environment_omits_resources_outside_sdk_bounds(monkeypatch, tmp_path):
    """Shared container settings can exceed Tenki's API ranges. Omit invalid
    values so the SDK/workspace defaults apply instead of making create fail."""
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text("auth_token: tok-secret\n", encoding="utf-8")

    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    env = TenkiEnvironment(
        task_id="resource-out-of-range",
        cpu=17,
        memory=64,
        disk=4 * 1024,
    )

    kwargs = _FakeSandboxFactory.created_kwargs[0]
    assert "cpu_cores" not in kwargs
    assert "memory_mb" not in kwargs
    assert "disk_size_gb" not in kwargs
    env.cleanup()


def test_tenki_environment_omits_unaligned_sdk_memory(monkeypatch, tmp_path):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text("auth_token: tok-secret\n", encoding="utf-8")

    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    env = TenkiEnvironment(
        task_id="resource-unaligned-memory",
        memory=129,
    )

    kwargs = _FakeSandboxFactory.created_kwargs[0]
    assert "memory_mb" not in kwargs
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

        def sync_back(self, _hermes_home=None):
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
    assert (
        env._remote_transfer_path(".hermes_tenki_sync")
        != env._remote_transfer_path(".hermes_tenki_sync")
    )
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
        def sync_back(self, _hermes_home=None):
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


def test_tenki_require_sandbox_rejects_cleanup_claimed_in_capture_gap(monkeypatch, tmp_path):
    """cleanup may claim the sandbox after ensure releases the lock; the later
    capture must recheck the guard instead of returning cleanup's reference."""
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text("auth_token: tok-secret\n", encoding="utf-8")

    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    env = TenkiEnvironment(task_id="cleanup-capture-race")
    original_ensure = env._ensure_sandbox
    ensured = threading.Event()
    cleanup_claimed = threading.Event()

    def ensure_then_wait_for_cleanup():
        original_ensure()
        ensured.set()
        assert cleanup_claimed.wait(timeout=2)

    monkeypatch.setattr(env, "_ensure_sandbox", ensure_then_wait_for_cleanup)
    errors: list[BaseException] = []

    def require_sandbox():
        try:
            env._require_sandbox()
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=require_sandbox)
    worker.start()
    assert ensured.wait(timeout=2)
    with env._lock:
        env._cleanup_in_progress = True
        env._cleanup_sandbox = env._sandbox
    cleanup_claimed.set()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert "cleanup" in str(errors[0])

    monkeypatch.setattr(env, "_ensure_sandbox", original_ensure)
    with env._lock:
        env._cleanup_in_progress = False
        env._cleanup_sandbox = None
    env.cleanup()


def test_tenki_ephemeral_cleanup_closes_environment_owned_client(monkeypatch, tmp_path):
    """The default ephemeral path must not leak Sandbox.create's hidden client."""
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text("auth_token: tok-secret\n", encoding="utf-8")

    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    env = TenkiEnvironment(task_id="ephemeral-client-close")

    assert env._persistent is False
    assert isinstance(env._client, _FakeClient)
    assert _FakeClient.closed_count == 0

    env.cleanup()

    assert _FakeClient.closed_count == 1
    assert env._client is None


def test_tenki_failed_termination_remains_retryable_and_tracked(
    monkeypatch,
    tmp_path,
):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TERMINAL_ENV", "tenki")
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text("auth_token: tok-secret\n", encoding="utf-8")

    from tools import terminal_tool
    from tools.environments import tenki as tenki_module
    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    monkeypatch.setattr(tenki_module, "_TERMINATE_RETRY_DELAYS", (0, 0))
    env = TenkiEnvironment(task_id="retry-termination")
    sandbox = env._sandbox
    original_terminate = sandbox.terminate

    def fail_terminate():
        raise RuntimeError("control plane unavailable")

    monkeypatch.setattr(sandbox, "terminate", fail_terminate)
    cache_key = terminal_tool._resolve_environment_cache_key(
        "retry-termination",
        "tenki",
    )
    terminal_tool._active_environments[cache_key] = env
    terminal_tool._last_activity[cache_key] = 0

    terminal_tool._cleanup_inactive_envs(0)

    assert terminal_tool._active_environments[cache_key] is env
    assert cache_key not in terminal_tool._retiring_environments
    assert env._sandbox is sandbox
    assert env._client is not None
    assert env._cleanup_complete is False
    assert _FakeClient.closed_count == 0

    monkeypatch.setattr(sandbox, "terminate", original_terminate)
    terminal_tool.cleanup_vm("retry-termination")

    assert cache_key not in terminal_tool._active_environments
    assert sandbox.terminated is True
    assert _FakeClient.closed_count == 1


def test_tenki_concurrent_cleanup_terminates_and_closes_once(monkeypatch, tmp_path):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text("auth_token: tok-secret\n", encoding="utf-8")

    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    env = TenkiEnvironment(task_id="concurrent-cleanup")
    sandbox = env._sandbox
    terminate_entered = threading.Event()
    release_terminate = threading.Event()
    terminate_calls = []
    original_terminate = sandbox.terminate

    def blocking_terminate():
        terminate_calls.append(True)
        terminate_entered.set()
        assert release_terminate.wait(timeout=2)
        original_terminate()

    monkeypatch.setattr(sandbox, "terminate", blocking_terminate)
    cleanup_waiting = threading.Event()
    original_wait = env._lifecycle_condition.wait

    def tracked_wait(timeout=None):
        cleanup_waiting.set()
        return original_wait(timeout)

    monkeypatch.setattr(env._lifecycle_condition, "wait", tracked_wait)
    first = threading.Thread(target=env.cleanup)
    second = threading.Thread(target=env.cleanup)
    first.start()
    assert terminate_entered.wait(timeout=2)
    second.start()

    assert cleanup_waiting.wait(timeout=2)
    assert _FakeClient.closed_count == 0
    release_terminate.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert len(terminate_calls) == 1
    assert _FakeClient.closed_count == 1


@pytest.mark.parametrize("persistent", [False, True])
def test_tenki_cleanup_waits_for_cancel_before_closing_client(
    monkeypatch,
    tmp_path,
    persistent,
):
    """A detached sandbox still needs its control-plane client until cancel's
    terminate RPC completes; cleanup must not close that shared channel first."""
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text("auth_token: tok-secret\n", encoding="utf-8")

    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    env = TenkiEnvironment(
        task_id=f"cancel-cleanup-{persistent}",
        persistent_filesystem=persistent,
    )
    sandbox = env._sandbox
    # Force cancel's sandbox-level fallback instead of the process.kill fast path.
    monkeypatch.setattr(sandbox, "start", None)
    terminate_entered = threading.Event()
    release_terminate = threading.Event()
    action_name = "pause" if persistent else "terminate"
    original_action = getattr(sandbox, action_name)

    def blocking_action():
        terminate_entered.set()
        assert release_terminate.wait(timeout=2)
        original_action()

    monkeypatch.setattr(sandbox, action_name, blocking_action)
    handle = env._run_bash("echo running", timeout=5)
    cancel_thread = threading.Thread(target=handle.kill)
    cancel_thread.start()
    assert terminate_entered.wait(timeout=2)

    cleanup_thread = threading.Thread(target=env.cleanup)
    cleanup_waiting = threading.Event()
    original_wait = env._lifecycle_condition.wait

    def tracked_wait(timeout=None):
        cleanup_waiting.set()
        return original_wait(timeout)

    monkeypatch.setattr(env._lifecycle_condition, "wait", tracked_wait)
    cleanup_thread.start()
    assert cleanup_waiting.wait(timeout=2)
    assert _FakeClient.closed_count == 0

    release_terminate.set()
    cancel_thread.join(timeout=2)
    cleanup_thread.join(timeout=2)
    handle.wait(timeout=2)

    assert not cancel_thread.is_alive()
    assert not cleanup_thread.is_alive()
    assert getattr(sandbox, "paused" if persistent else "terminated") is True
    if persistent:
        assert sandbox.terminated is True
    assert _FakeClient.closed_count == 1


def test_tenki_persistent_cancel_never_terminates_when_pause_fails(
    monkeypatch,
    tmp_path,
):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text(
        "auth_token: tok-secret\n",
        encoding="utf-8",
    )

    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    env = TenkiEnvironment(
        task_id="cancel-pause-failure",
        persistent_filesystem=True,
    )
    sandbox = env._sandbox
    # Force sandbox-level cancellation before a process handle can supply a
    # working kill method, then make the preservation pause fail.
    monkeypatch.setattr(sandbox, "start", None)

    def fail_pause():
        raise RuntimeError("pause control plane unavailable")

    monkeypatch.setattr(sandbox, "pause", fail_pause)
    handle = env._run_bash("echo running", timeout=5)
    handle.kill()
    handle.wait(timeout=2)

    assert env._sandbox is sandbox
    assert sandbox.paused is False
    assert sandbox.terminated is False
    assert sandbox.state == "RUNNING"


def test_tenki_persistent_cancel_blocks_ensure_and_reuses_exact_sandbox(
    monkeypatch,
    tmp_path,
):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text(
        "auth_token: tok-secret\n",
        encoding="utf-8",
    )

    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    env = TenkiEnvironment(
        task_id="cancel-exact-reference",
        persistent_filesystem=True,
    )
    sandbox = env._sandbox
    monkeypatch.setattr(sandbox, "start", None)
    pause_entered = threading.Event()
    release_pause = threading.Event()
    original_pause = sandbox.pause

    def blocking_pause():
        pause_entered.set()
        assert release_pause.wait(timeout=2)
        original_pause()

    monkeypatch.setattr(sandbox, "pause", blocking_pause)
    handle = env._run_bash("echo running", timeout=5)
    cancel_thread = threading.Thread(target=handle.kill)
    cancel_thread.start()
    assert pause_entered.wait(timeout=2)

    # If ensure incorrectly falls back to discovery after cancel, this turns
    # that mistake into an error instead of silently finding the same sandbox.
    def fail_list(self, **_kwargs):
        raise RuntimeError("transient list miss")

    monkeypatch.setattr(_FakeClient, "list", fail_list)
    ensured = []
    ensure_errors = []

    def execute_again():
        try:
            result = env._exec_raw("echo after cancel")
            ensured.append((env._sandbox, result))
        except Exception as exc:
            ensure_errors.append(exc)

    ensure_thread = threading.Thread(target=execute_again)
    ensure_thread.start()
    time.sleep(0.05)

    assert ensure_thread.is_alive()
    assert env._sandbox is sandbox

    release_pause.set()
    cancel_thread.join(timeout=2)
    ensure_thread.join(timeout=2)
    handle.wait(timeout=2)

    assert not cancel_thread.is_alive()
    assert not ensure_thread.is_alive()
    assert ensure_errors == []
    assert ensured == [(sandbox, ("ran\n", 0))]
    assert env._sandbox is sandbox
    assert sandbox.resumed is True
    assert len(_FakeSandboxFactory.created_kwargs) == 1


def test_tenki_completed_cancel_generation_rechecks_before_lease(
    monkeypatch,
    tmp_path,
):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text(
        "auth_token: tok-secret\n",
        encoding="utf-8",
    )

    from tools.environments import tenki as tenki_module
    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    env = TenkiEnvironment(
        task_id="cancel-generation",
        persistent_filesystem=True,
    )
    sandbox = env._sandbox
    monkeypatch.setattr(sandbox, "start", None)

    class CapturedHandle:
        def __init__(self, _exec_fn, cancel_fn=None):
            self._cancel_fn = cancel_fn

        def kill(self):
            self._cancel_fn()

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(tenki_module, "_ThreadedProcessHandle", CapturedHandle)
    cancel_handle = env._run_bash("echo cancelled", timeout=5)

    original_require = env._require_sandbox
    resolved_once = threading.Event()
    release_resolved = threading.Event()
    require_calls = 0

    def require_with_gap():
        nonlocal require_calls
        resolved = original_require()
        require_calls += 1
        if require_calls == 1:
            resolved_once.set()
            assert release_resolved.wait(timeout=2)
        return resolved

    monkeypatch.setattr(env, "_require_sandbox", require_with_gap)
    leased = []
    lease_errors = []

    def take_lease():
        try:
            with env._sandbox_operation() as leased_sandbox:
                leased.append(
                    (leased_sandbox, leased_sandbox.state)
                )
        except Exception as exc:
            lease_errors.append(exc)

    lease_thread = threading.Thread(target=take_lease)
    lease_thread.start()
    assert resolved_once.wait(timeout=2)

    # Complete the entire cancel inside the resolve→publish gap. Object
    # identity is unchanged, so only the generation check can detect that the
    # resolved sandbox was paused after readiness validation.
    cancel_handle.kill()
    assert sandbox.state == "PAUSED"
    assert env._cancel_in_progress == 0
    release_resolved.set()
    lease_thread.join(timeout=2)

    assert not lease_thread.is_alive()
    assert lease_errors == []
    assert require_calls >= 2
    assert leased == [(sandbox, "RUNNING")]
    assert sandbox.resumed is True
    assert env._active_operations == 0


def test_tenki_cleanup_waits_for_running_execution(monkeypatch, tmp_path):
    """Cleanup cannot terminate the sandbox while a leased command is still
    using it, even when that command is not in the background-process registry."""
    env = _tenki_env_for_upload_race(monkeypatch, tmp_path, "cleanup-exec-lease")
    sandbox = env._sandbox
    client = env._client
    client_close_calls = []
    original_close = client.close

    def tracked_close():
        client_close_calls.append(True)
        original_close()

    monkeypatch.setattr(client, "close", tracked_close)
    handle = env._run_bash("sleep infinity", timeout=30)
    for _ in range(100):
        if sandbox.last_process is not None:
            break
        time.sleep(0.01)
    assert sandbox.last_process is not None

    cleanup_waiting = threading.Event()
    original_wait = env._lifecycle_condition.wait

    def tracked_wait(timeout=None):
        cleanup_waiting.set()
        return original_wait(timeout)

    monkeypatch.setattr(env._lifecycle_condition, "wait", tracked_wait)
    cleanup_thread = threading.Thread(target=env.cleanup)
    cleanup_thread.start()

    assert cleanup_waiting.wait(timeout=2)
    assert sandbox.terminated is False
    assert client_close_calls == []

    sandbox.last_process._done.set()
    handle.wait(timeout=2)
    cleanup_thread.join(timeout=2)

    assert not cleanup_thread.is_alive()
    assert sandbox.terminated is True
    assert client_close_calls == [True]


def test_tenki_cleanup_waits_for_running_upload(monkeypatch, tmp_path):
    """The mkdir/upload pair holds one operation lease until both calls finish."""
    env = _tenki_env_for_upload_race(monkeypatch, tmp_path, "cleanup-upload-lease")
    sandbox = env._sandbox
    host_file = tmp_path / "skill.md"
    host_file.write_text("content", encoding="utf-8")
    upload_entered = threading.Event()
    release_upload = threading.Event()
    original_upload = sandbox.fs.upload

    def blocking_upload(*args, **kwargs):
        upload_entered.set()
        assert release_upload.wait(timeout=2)
        return original_upload(*args, **kwargs)

    monkeypatch.setattr(sandbox.fs, "upload", blocking_upload)
    upload_thread = threading.Thread(
        target=env._tenki_upload,
        args=(str(host_file), "/home/tenki/.hermes/skills/skill.md"),
    )
    upload_thread.start()
    assert upload_entered.wait(timeout=2)

    cleanup_waiting = threading.Event()
    original_wait = env._lifecycle_condition.wait

    def tracked_wait(timeout=None):
        cleanup_waiting.set()
        return original_wait(timeout)

    monkeypatch.setattr(env._lifecycle_condition, "wait", tracked_wait)
    cleanup_thread = threading.Thread(target=env.cleanup)
    cleanup_thread.start()

    assert cleanup_waiting.wait(timeout=2)
    assert sandbox.terminated is False
    assert _FakeClient.closed_count == 0

    release_upload.set()
    upload_thread.join(timeout=2)
    cleanup_thread.join(timeout=2)

    assert not upload_thread.is_alive()
    assert not cleanup_thread.is_alive()
    assert sandbox.fs.upload_calls
    assert sandbox.terminated is True
    assert _FakeClient.closed_count == 1


def test_tenki_idle_reaper_waits_for_running_execution(monkeypatch, tmp_path):
    """A retirement tombstone keeps all creators from replacing the wrapper
    until its active operation and cleanup have both completed."""
    env = _tenki_env_for_upload_race(monkeypatch, tmp_path, "reaper-exec-lease")
    sandbox = env._sandbox
    handle = env._run_bash("sleep infinity", timeout=30)
    for _ in range(100):
        if sandbox.last_process is not None:
            break
        time.sleep(0.01)
    assert sandbox.last_process is not None

    from tools import terminal_tool
    from tools import code_execution_tool
    from tools import file_tools

    monkeypatch.setenv("TERMINAL_ENV", "tenki")
    cache_key = terminal_tool._resolve_environment_cache_key(
        "reaper-exec-lease",
        "tenki",
    )
    terminal_tool._active_environments[cache_key] = env
    terminal_tool._last_activity[cache_key] = 0
    cleanup_waiting = threading.Event()
    original_wait = env._lifecycle_condition.wait

    def tracked_wait(timeout=None):
        cleanup_waiting.set()
        return original_wait(timeout)

    monkeypatch.setattr(env._lifecycle_condition, "wait", tracked_wait)
    reaper_thread = threading.Thread(
        target=terminal_tool._cleanup_inactive_envs,
        args=(0,),
    )
    reaper_thread.start()

    assert cleanup_waiting.wait(timeout=2)
    assert terminal_tool._active_environments[cache_key] is env
    assert cache_key in terminal_tool._retiring_environments
    assert sandbox.terminated is False

    replacement_created = threading.Event()

    class ReplacementEnv:
        cwd = "/home/tenki"

        def cleanup(self):
            return None

    replacement = ReplacementEnv()

    def create_replacement(**_kwargs):
        assert sandbox.terminated is True
        replacement_created.set()
        return replacement

    monkeypatch.setattr(
        terminal_tool,
        "_create_environment",
        create_replacement,
    )
    cache_clear_entered = threading.Event()
    release_cache_clear = threading.Event()
    original_clear_file_ops_cache = file_tools.clear_file_ops_cache

    def blocked_clear_file_ops_cache(task_id=None):
        cache_clear_entered.set()
        assert release_cache_clear.wait(timeout=2)
        return original_clear_file_ops_cache(task_id)

    monkeypatch.setattr(
        file_tools,
        "clear_file_ops_cache",
        blocked_clear_file_ops_cache,
    )
    creator_result = []

    def create_for_code_execution():
        creator_result.append(
            code_execution_tool._get_or_create_env("reaper-exec-lease")
        )

    creator_thread = threading.Thread(target=create_for_code_execution)
    creator_thread.start()
    time.sleep(0.05)
    assert creator_thread.is_alive()
    assert replacement_created.is_set() is False

    sandbox.last_process._done.set()
    handle.wait(timeout=2)
    assert cache_clear_entered.wait(timeout=2)
    assert creator_thread.is_alive()
    assert replacement_created.is_set() is False
    release_cache_clear.set()
    reaper_thread.join(timeout=2)
    creator_thread.join(timeout=2)

    assert not reaper_thread.is_alive()
    assert not creator_thread.is_alive()
    assert sandbox.terminated is True
    assert _FakeClient.closed_count == 1
    assert replacement_created.is_set() is True
    assert creator_result == [(replacement, "tenki")]
    terminal_tool._active_environments.pop(cache_key, None)
    terminal_tool._last_activity.pop(cache_key, None)
    terminal_tool._creation_locks.pop(cache_key, None)


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

def test_exec_survives_cancel_clearing_sandbox_mid_operation(monkeypatch, tmp_path):
    """cancel() nulling self._sandbox between _ensure_sandbox() and the exec
    call must not crash the in-flight command: operations run against the
    reference captured by _require_sandbox(), and a genuinely torn-down
    sandbox surfaces a clean RuntimeError instead of an AttributeError."""
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "missing.yaml"))
    monkeypatch.setenv("TENKI_API_KEY", "sk-test-key")

    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    env = TenkiEnvironment(task_id="cancel-race")

    sandbox = _FakeSandboxFactory.sandboxes[0]
    orig_exec = sandbox.exec

    def exec_and_teardown(*args, **kwargs):
        env._sandbox = None  # what cancel() does concurrently
        return orig_exec(*args, **kwargs)

    monkeypatch.setattr(sandbox, "exec", exec_and_teardown)
    output, exit_code = env._exec_raw("echo ok", timeout=5)
    assert exit_code == 0

    # After the teardown a fruitless ensure must fail loud and typed.
    monkeypatch.setattr(env, "_ensure_sandbox", lambda: None)
    env._sandbox = None
    with pytest.raises(RuntimeError, match="torn down"):
        env._exec_raw("echo ok", timeout=5)


def _tenki_env_for_upload_race(monkeypatch, tmp_path, task_id):
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "missing.yaml"))
    monkeypatch.setenv("TENKI_API_KEY", "sk-test-key")

    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    return TenkiEnvironment(task_id=task_id)


def test_tenki_upload_targets_one_sandbox_when_cancel_races(monkeypatch, tmp_path):
    """cancel() nulls self._sandbox, so re-reading it per call could send the
    mkdir and the upload of a single file to two different sandboxes (or to
    None). Both must run against one captured reference."""
    env = _tenki_env_for_upload_race(monkeypatch, tmp_path, "upload-race")
    original = env._sandbox
    created_before = len(_FakeSandboxFactory.created_kwargs)

    orig_mkdir = original.fs.mkdir

    def mkdir_and_teardown(*args, **kwargs):
        env._sandbox = None  # what cancel() does concurrently
        return orig_mkdir(*args, **kwargs)

    monkeypatch.setattr(original.fs, "mkdir", mkdir_and_teardown)

    host_file = tmp_path / "skill.md"
    host_file.write_text("content", encoding="utf-8")
    env._tenki_upload(str(host_file), "/home/tenki/.hermes/skills/skill.md")

    assert len(_FakeSandboxFactory.created_kwargs) == created_before
    assert original.fs.mkdir_calls, "mkdir ran on the captured sandbox"
    assert original.fs.upload_calls[-1] == (
        str(host_file),
        "/home/tenki/.hermes/skills/skill.md",
    )


def test_tenki_bulk_upload_targets_one_sandbox_when_cancel_races(monkeypatch, tmp_path):
    """Same single-capture rule for the bulk flow: mkdir, tar upload, untar and
    the cleanup rm must all land on one sandbox, or the tar gets extracted
    somewhere other than where it was uploaded."""
    env = _tenki_env_for_upload_race(monkeypatch, tmp_path, "bulk-upload-race")
    original = env._sandbox
    created_before = len(_FakeSandboxFactory.created_kwargs)

    orig_exec = original.exec

    def exec_and_teardown(*args, **kwargs):
        env._sandbox = None  # cancel() lands during the mkdir
        return orig_exec(*args, **kwargs)

    monkeypatch.setattr(original, "exec", exec_and_teardown)

    host_file = tmp_path / "skill.md"
    host_file.write_text("content", encoding="utf-8")
    env._tenki_bulk_upload([(str(host_file), "/home/tenki/.hermes/skills/skill.md")])

    assert len(_FakeSandboxFactory.created_kwargs) == created_before
    remote_tar = original.fs.upload_calls[-1][1]
    commands = [call[0][-1] for call in original.exec_calls]
    assert any(cmd.startswith("mkdir ") for cmd in commands)
    # The untar and the cleanup both reference the tar this sandbox received.
    assert any(f"tar xf {shlex.quote(remote_tar)}" in cmd for cmd in commands)
    assert any(f"rm -f {shlex.quote(remote_tar)}" in cmd for cmd in commands)


def test_tenki_create_kwargs_filters_names_the_installed_sdk_dropped(monkeypatch, tmp_path):
    """``Sandbox.create`` is a bare ``**kwargs`` passthrough, so the accepted
    set has to be read off ``Client.create`` — the real validator. Filtering
    against the passthrough accepts every name, and one the SDK has dropped
    (``project_id``, gone in 0.5) reaches the client as an unexpected keyword
    and kills sandbox creation with a TypeError."""
    _install_fake_tenki(monkeypatch)
    _clear_tenki_auth_env(monkeypatch)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TENKI_CONFIG_PATH", str(tmp_path / "missing.yaml"))
    monkeypatch.setenv("TENKI_API_KEY", "sk-test-key")

    # A deliberately narrow client: no cpu_cores/memory_mb/disk_size_gb/
    # allow_*/max_duration/idle_timeout_minutes/pause_retention, and no
    # **kwargs to swallow them.
    def narrow_create(self, *, name=None, image=None, env=None, metadata=None, tags=None, wait=True):
        return _FakeSandboxFactory._record_and_build(
            {"name": name, "image": image, "env": env, "metadata": metadata, "tags": tags, "wait": wait}
        )

    monkeypatch.setattr(_FakeClient, "create", narrow_create)

    from tools.environments.tenki import TenkiEnvironment

    monkeypatch.setattr(TenkiEnvironment, "init_session", lambda self: None)
    env = TenkiEnvironment(
        task_id="narrow-sdk",
        image="base-image",
        cpu=2,
        memory=2048,
        idle_timeout=120,
        pause_retention=60,
    )

    kwargs = _FakeSandboxFactory.created_kwargs[0]
    assert kwargs["image"] == "base-image"
    for dropped in (
        "cpu_cores",
        "memory_mb",
        "disk_size_gb",
        "allow_inbound",
        "allow_outbound",
        "max_duration",
        "idle_timeout_minutes",
        "pause_retention",
    ):
        assert dropped not in kwargs
    env.cleanup()


def test_installed_tenki_sdk_exposes_adapter_contract_without_network():
    """Exercise the real optional SDK surface that the adapter introspects."""
    import inspect

    tenki = pytest.importorskip("tenki")
    client_create = inspect.signature(tenki.Client.create).parameters
    assert {
        "workspace_id",
        "snapshot_id",
        "image",
        "cpu_cores",
        "memory_mb",
        "disk_size_gb",
    } <= set(client_create)
    assert "project_id" not in client_create
    assert "workspace_id" in inspect.signature(tenki.Client.list).parameters
    assert any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in inspect.signature(
            tenki.Sandbox.create
        ).parameters.values()
    )

    # Client construction is local; no request is made until a resource
    # method is called. Verify cleanup/durability APIs without contacting
    # Tenki.
    client = tenki.Client(
        auth_token="contract-test",
        base_url="https://example.invalid",
    )
    try:
        assert callable(client.snapshots.wait_durable)
        assert callable(client.snapshots.delete)
        assert "snapshot_id" in inspect.signature(
            client.snapshots.wait_durable
        ).parameters
        assert "snapshot_id" in inspect.signature(
            client.snapshots.delete
        ).parameters
    finally:
        client.close()


def test_snapshot_unrecoverable_resolves_renamed_registry_error(monkeypatch):
    """tenki 0.5 renamed RegistryArtifactNotFoundError to
    RegistryImageNotFoundError. Importing the whole set in one statement made a
    single rename drop the isinstance check for every class in it."""
    _install_fake_tenki(monkeypatch)

    from tools.environments.tenki import TenkiEnvironment

    # 0.5 name, exported by the installed SDK.
    assert TenkiEnvironment._snapshot_unrecoverable(_FakeRegistryImageNotFoundError("gone")) is True
    assert TenkiEnvironment._snapshot_unrecoverable(_FakeSnapshotNotFoundError("gone")) is True
    assert TenkiEnvironment._snapshot_unrecoverable(_FakeSnapshotNotDurableError("nope")) is True

    # Pre-0.5 name: absent from a 0.5 SDK's exports, so only the MRO name
    # fallback can classify it. Getting this wrong silently boots a base image.
    class RegistryArtifactNotFoundError(Exception):
        pass

    assert TenkiEnvironment._snapshot_unrecoverable(RegistryArtifactNotFoundError("gone")) is True

    # Transient failures must keep the snapshot pointer for a later retry.
    assert TenkiEnvironment._snapshot_unrecoverable(RuntimeError("network blip")) is False
    assert TenkiEnvironment._snapshot_unrecoverable(_FakeInvalidStateError("workspace suspended")) is False
    # ...but an InvalidStateError that names the snapshot is unrecoverable.
    assert TenkiEnvironment._snapshot_unrecoverable(_FakeInvalidStateError("snapshot is not durable")) is True
    # A generic FAILED_PRECONDITION may mention a snapshot while describing a
    # transient workspace/policy block; that must preserve the restore pointer.
    assert (
        TenkiEnvironment._snapshot_unrecoverable(
            _FakeInvalidStateError(
                "snapshot restore temporarily blocked by workspace policy"
            )
        )
        is False
    )
