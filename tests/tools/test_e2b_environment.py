"""Unit tests for the E2B cloud sandbox environment backend."""

from __future__ import annotations

import importlib
import re
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Fake SDK objects
# ---------------------------------------------------------------------------


class _FakeCommandResult:
    def __init__(self, stdout: str = "", stderr: str = "", exit_code: int = 0):
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code


class _FakeFiles:
    def __init__(self):
        self.written: list[tuple[str, bytes | str]] = []
        self.dirs_created: list[str] = []
        self.read_results: dict[str, bytes] = {}

    def write(self, path: str, data):
        self.written.append((path, data))

    def make_dir(self, path: str):
        self.dirs_created.append(path)

    def read(self, path: str, format: str = "text"):
        return self.read_results.get(path, b"")


class _FakeCommands:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self.side_effects: list[object] = []

    def run(self, cmd: str, **kwargs) -> _FakeCommandResult:
        self.calls.append((cmd, kwargs))
        if self.side_effects:
            effect = self.side_effects.pop(0)
            if isinstance(effect, Exception):
                raise effect
            if callable(effect):
                return effect(cmd, kwargs)
            return effect
        return _FakeCommandResult()


class _FakeSandbox:
    def __init__(self, sandbox_id: str = "sb-e2b-123", home: str = "/home/user"):
        self.sandbox_id = sandbox_id
        self.home = home
        self.commands = _FakeCommands()
        self.files = _FakeFiles()
        self.killed = False
        self.paused = False
        self._is_running = True
        self.kill_calls = 0
        self.pause_calls = 0

        # Wire up home detection by default
        def _home_cmd(cmd, kwargs):
            if '"$HOME"' in cmd:
                return _FakeCommandResult(stdout=self.home)
            return _FakeCommandResult()

        self.commands.side_effects.append(_home_cmd)

    def kill(self):
        self.killed = True
        self.kill_calls += 1

    def beta_pause(self):
        self.paused = True
        self.pause_calls += 1

    def is_running(self):
        return self._is_running

    def connect(self, timeout=None):
        return self


class _FakeSDK:
    def __init__(self):
        self.create_kwargs: list[dict] = []
        self.create_side_effects: list[object] = []
        self.sandboxes: list[_FakeSandbox] = []
        self.connect_calls: list[tuple[str, dict]] = []
        self.connect_side_effects: list[object] = []

    @property
    def current(self) -> _FakeSandbox:
        return self.sandboxes[-1]

    def create(self, **kwargs) -> _FakeSandbox:
        self.create_kwargs.append(kwargs)
        if self.create_side_effects:
            effect = self.create_side_effects.pop(0)
            if isinstance(effect, Exception):
                raise effect
            if isinstance(effect, _FakeSandbox):
                self.sandboxes.append(effect)
                return effect
        sandbox = _FakeSandbox()
        self.sandboxes.append(sandbox)
        return sandbox

    def connect(self, sandbox_id: str, **kwargs) -> _FakeSandbox:
        self.connect_calls.append((sandbox_id, kwargs))
        if self.connect_side_effects:
            effect = self.connect_side_effects.pop(0)
            if isinstance(effect, Exception):
                raise effect
            if isinstance(effect, _FakeSandbox):
                self.sandboxes.append(effect)
                return effect
        sandbox = _FakeSandbox(sandbox_id=sandbox_id)
        self.sandboxes.append(sandbox)
        return sandbox


def _cwd_result(body: str = "", *, cwd: str = "/home/user", exit_code: int = 0):
    def _result(cmd: str, kwargs: dict):
        match = re.search(r"__HERMES_CWD_[A-Za-z0-9]+__", cmd)
        marker = match.group(0) if match else "__HERMES_CWD_MISSING__"
        prefix = f"{body}\n\n" if body else "\n"
        return _FakeCommandResult(
            stdout=f"{prefix}{marker}{cwd}{marker}\n", exit_code=exit_code
        )

    return _result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def e2b_sdk(monkeypatch):
    fake_sdk = _FakeSDK()

    e2b_mod = types.ModuleType("e2b")
    # Sandbox class with both static (create) and instance (connect) methods
    sandbox_cls = type(
        "Sandbox",
        (),
        {
            "create": staticmethod(fake_sdk.create),
            "connect": staticmethod(fake_sdk.connect),
        },
    )
    e2b_mod.Sandbox = sandbox_cls

    monkeypatch.setitem(sys.modules, "e2b", e2b_mod)
    return fake_sdk


@pytest.fixture()
def e2b_module(e2b_sdk, monkeypatch):
    monkeypatch.setattr("tools.environments.base.is_interrupted", lambda: False)
    monkeypatch.setattr("tools.credential_files.get_credential_file_mounts", lambda: [])
    monkeypatch.setattr("tools.credential_files.iter_skills_files", lambda **kwargs: [])
    monkeypatch.setattr("tools.credential_files.iter_cache_files", lambda **kwargs: [])

    module = importlib.import_module("tools.environments.e2b")
    return importlib.reload(module)


@pytest.fixture()
def make_env(e2b_module, e2b_sdk, request):
    envs = []

    def _cleanup():
        for env in envs:
            env._sync_manager = None
            env.cleanup()

    request.addfinalizer(_cleanup)

    def _factory(**kwargs):
        kwargs.setdefault("cwd", e2b_module.DEFAULT_E2B_CWD)
        kwargs.setdefault("timeout", 30)
        kwargs.setdefault("task_id", "task-e2b-123")
        env = e2b_module.E2BEnvironment(**kwargs)
        envs.append(env)
        return env

    return _factory


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStartup:
    def test_default_cwd_resolves_to_remote_home(self, make_env, e2b_sdk):
        sandbox = _FakeSandbox(home="/home/custom")
        e2b_sdk.create_side_effects.append(sandbox)

        env = make_env()

        assert env.cwd == "/home/custom"

    def test_tilde_cwd_resolves_to_remote_home(self, make_env, e2b_sdk):
        sandbox = _FakeSandbox(home="/home/custom")
        e2b_sdk.create_side_effects.append(sandbox)

        env = make_env(cwd="~")

        assert env.cwd == "/home/custom"

    def test_home_detection_failure_uses_default(self, make_env, e2b_sdk):
        sandbox = _FakeSandbox()
        sandbox.commands.side_effects.clear()
        sandbox.commands.side_effects.append(RuntimeError("exec failed"))
        e2b_sdk.create_side_effects.append(sandbox)

        env = make_env()

        assert env.cwd == "/home/user"

    def test_template_passed_to_create(self, make_env, e2b_sdk):
        make_env(template="code-interpreter-v1")

        assert e2b_sdk.create_kwargs[0]["template"] == "code-interpreter-v1"

    def test_no_template_omits_key(self, make_env, e2b_sdk):
        make_env()

        assert e2b_sdk.create_kwargs[0].get("template") is None


class TestPersistence:
    def test_persistent_resumes_via_connect(self, make_env, e2b_module, e2b_sdk, monkeypatch, tmp_path):
        hermes_home = tmp_path / ".hermes"
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        e2b_module._store_sandbox_id("task-e2b-123", "sb-saved")
        resumed = _FakeSandbox(sandbox_id="sb-saved", home="/home/user")
        e2b_sdk.connect_side_effects.append(resumed)

        env = make_env()

        assert e2b_sdk.connect_calls[0][0] == "sb-saved"
        assert e2b_sdk.create_kwargs == []

    def test_resume_failure_creates_fresh(self, make_env, e2b_module, e2b_sdk, monkeypatch, tmp_path):
        hermes_home = tmp_path / ".hermes"
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        e2b_module._store_sandbox_id("task-e2b-123", "sb-stale")
        e2b_sdk.connect_side_effects.append(RuntimeError("sandbox expired"))

        env = make_env()

        assert len(e2b_sdk.connect_calls) == 1
        assert len(e2b_sdk.create_kwargs) == 1
        assert e2b_module._load_sandbox_store().get("task-e2b-123") is None

    def test_non_persistent_skips_resume(self, make_env, e2b_module, e2b_sdk, monkeypatch, tmp_path):
        hermes_home = tmp_path / ".hermes"
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        e2b_module._store_sandbox_id("task-e2b-123", "sb-saved")

        env = make_env(persistent_filesystem=False)

        assert e2b_sdk.connect_calls == []
        assert len(e2b_sdk.create_kwargs) == 1


class TestCleanup:
    def test_persistent_cleanup_pauses_and_stores_id(self, make_env, e2b_module, monkeypatch, tmp_path):
        hermes_home = tmp_path / ".hermes"
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        env = make_env()
        sandbox = env._sandbox

        env.cleanup()

        assert sandbox.paused
        assert sandbox.pause_calls == 1
        assert not sandbox.killed
        assert e2b_module._load_sandbox_store().get("task-e2b-123") == "sb-e2b-123"

    def test_non_persistent_cleanup_kills(self, make_env, e2b_module, monkeypatch, tmp_path):
        hermes_home = tmp_path / ".hermes"
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        env = make_env(persistent_filesystem=False)
        sandbox = env._sandbox

        env.cleanup()

        assert sandbox.killed
        assert not sandbox.paused
        assert e2b_module._load_sandbox_store().get("task-e2b-123") is None

    def test_pause_failure_falls_back_to_kill(self, make_env, e2b_module, monkeypatch, tmp_path):
        hermes_home = tmp_path / ".hermes"
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        env = make_env()
        sandbox = env._sandbox
        sandbox.beta_pause = MagicMock(side_effect=RuntimeError("pause failed"))

        env.cleanup()

        assert sandbox.killed
        assert e2b_module._load_sandbox_store().get("task-e2b-123") is None

    def test_cleanup_idempotent(self, make_env):
        env = make_env()
        env.cleanup()
        env.cleanup()  # should not raise

    def test_cleanup_syncs_back_before_pause(self, make_env, e2b_module, monkeypatch, tmp_path):
        hermes_home = tmp_path / ".hermes"
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        env = make_env()
        sync_called = []
        env._sync_manager.sync_back = lambda: sync_called.append(True)

        env.cleanup()

        assert len(sync_called) == 1


class TestExecute:
    def test_basic_command(self, make_env, e2b_sdk):
        env = make_env()
        sandbox = e2b_sdk.current
        sandbox.commands.side_effects.append(_cwd_result("hello"))

        result = env.execute("echo hello")

        assert "hello" in result["output"]
        assert result["returncode"] == 0

    def test_nonzero_exit_code(self, make_env, e2b_sdk):
        env = make_env()
        sandbox = e2b_sdk.current
        sandbox.commands.side_effects.append(
            _cwd_result("not found", exit_code=127)
        )

        result = env.execute("bad_cmd")

        assert result["returncode"] == 127

    def test_stdin_heredoc(self, make_env, e2b_sdk):
        env = make_env()
        sandbox = e2b_sdk.current
        sandbox.commands.side_effects.append(_cwd_result("hi"))

        env.execute("python3", stdin_data="print('hi')")

        last_cmd = sandbox.commands.calls[-1][0]
        assert "HERMES_STDIN_" in last_cmd
        assert "print" in last_cmd


class TestEnsureSandboxReady:
    def test_recreates_when_not_running_and_connect_fails(self, make_env, e2b_sdk):
        env = make_env()
        original = env._sandbox
        original._is_running = False

        # connect fails, so it falls through to create
        e2b_sdk.connect_side_effects.append(RuntimeError("sandbox expired"))
        fresh = _FakeSandbox(sandbox_id="sb-fresh")
        e2b_sdk.create_side_effects.append(fresh)

        env._ensure_sandbox_ready()

        assert env._sandbox is fresh

    def test_reconnects_via_sandbox_id(self, make_env, e2b_sdk):
        env = make_env()
        original = env._sandbox
        original._is_running = False

        reconnected = _FakeSandbox(sandbox_id=original.sandbox_id)
        e2b_sdk.connect_side_effects.append(reconnected)

        env._ensure_sandbox_ready()

        assert env._sandbox is reconnected
        assert e2b_sdk.connect_calls[-1][0] == original.sandbox_id

    def test_no_op_when_running(self, make_env, e2b_sdk):
        env = make_env()
        original = env._sandbox
        original._is_running = True

        env._ensure_sandbox_ready()

        assert env._sandbox is original
        assert e2b_sdk.connect_calls == []
