"""Regression tests for sudo detection and sudo password handling."""

import json
import queue
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import cli as cli_module
import tools.terminal_tool as terminal_tool
from cli import HermesCLI
from tools.environments.base import BaseEnvironment
from tools.interrupt import is_interrupted, set_interrupt
from tools.process_registry import ProcessStartCancelled


def setup_function():
    terminal_tool._reset_cached_sudo_passwords()


def teardown_function():
    terminal_tool._reset_cached_sudo_passwords()


def test_searching_for_sudo_does_not_trigger_rewrite(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    command = "rg --line-number --no-heading --with-filename 'sudo' . | head -n 20"
    transformed, sudo_stdin = terminal_tool._transform_sudo_command(command)

    assert transformed == command
    assert sudo_stdin is None


def test_terminal_schema_advertises_persistent_env_state():
    description = terminal_tool.TERMINAL_TOOL_DESCRIPTION

    assert "exported environment variables persist between calls" in description
    assert "activate a virtualenv" in description
    assert "do not re-source the same environment before every command" in description


def test_printf_literal_sudo_does_not_trigger_rewrite(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    command = "printf '%s\\n' sudo"
    transformed, sudo_stdin = terminal_tool._transform_sudo_command(command)

    assert transformed == command
    assert sudo_stdin is None


def test_non_command_argument_named_sudo_does_not_trigger_rewrite(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    command = "grep -n sudo README.md"
    transformed, sudo_stdin = terminal_tool._transform_sudo_command(command)

    assert transformed == command
    assert sudo_stdin is None


def test_actual_sudo_command_uses_configured_password(monkeypatch):
    monkeypatch.setenv("SUDO_PASSWORD", "testpass")
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    transformed, sudo_stdin = terminal_tool._transform_sudo_command("sudo apt install -y ripgrep")

    assert transformed == "sudo -S -p '' apt install -y ripgrep"
    assert sudo_stdin == "testpass\n"


def test_actual_sudo_after_leading_env_assignment_is_rewritten(monkeypatch):
    monkeypatch.setenv("SUDO_PASSWORD", "testpass")
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    transformed, sudo_stdin = terminal_tool._transform_sudo_command("DEBUG=1 sudo whoami")

    assert transformed == "DEBUG=1 sudo -S -p '' whoami"
    assert sudo_stdin == "testpass\n"


def test_explicit_empty_sudo_password_tries_empty_without_prompt(monkeypatch):
    monkeypatch.setenv("SUDO_PASSWORD", "")
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")

    def _fail_prompt(*_args, **_kwargs):
        raise AssertionError("interactive sudo prompt should not run for explicit empty password")

    monkeypatch.setattr(terminal_tool, "_prompt_for_sudo_password", _fail_prompt)

    transformed, sudo_stdin = terminal_tool._transform_sudo_command("sudo true")

    assert transformed == "sudo -S -p '' true"
    assert sudo_stdin == "\n"


def test_cached_sudo_password_is_used_when_env_is_unset(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    terminal_tool._set_cached_sudo_password("cached-pass")

    transformed, sudo_stdin = terminal_tool._transform_sudo_command("echo ok && sudo whoami")

    assert transformed == "echo ok && sudo -S -p '' whoami"
    assert sudo_stdin == "cached-pass\n"


def test_registered_sudo_callback_is_used_without_interactive_env(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    monkeypatch.setattr(terminal_tool, "_sudo_nopasswd_works", lambda: False)

    calls = []

    def sudo_callback():
        calls.append("called")
        return "callback-pass"

    terminal_tool.set_sudo_password_callback(sudo_callback)
    try:
        transformed, sudo_stdin = terminal_tool._transform_sudo_command(
            "echo ok | sudo tee /tmp/hermes-test"
        )
    finally:
        terminal_tool.set_sudo_password_callback(None)

    assert calls == ["called"]
    assert transformed == "echo ok | sudo -S -p '' tee /tmp/hermes-test"
    assert sudo_stdin == "callback-pass\n"


def test_registered_empty_sudo_callback_preserves_skip(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    monkeypatch.setattr(terminal_tool, "_sudo_nopasswd_works", lambda: False)
    terminal_tool.set_sudo_password_callback(lambda: "")
    try:
        transformed, sudo_stdin = terminal_tool._transform_sudo_command("sudo true")
    finally:
        terminal_tool.set_sudo_password_callback(None)

    assert transformed == "sudo true"
    assert sudo_stdin is None


def test_cancelled_sudo_prompt_stops_before_command_execution(monkeypatch, tmp_path):
    task_id = "sudo-cancel-test"

    class MinimalEnvironment(BaseEnvironment):
        def __init__(self):
            super().__init__(cwd=str(tmp_path), timeout=60)
            self.run_bash_calls = 0

        def _run_bash(self, *_args, **_kwargs):
            self.run_bash_calls += 1
            return object()

        def _wait_for_process(self, *_args, **_kwargs):
            return {"output": "executed", "returncode": 0}

        def cleanup(self):
            pass

    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    monkeypatch.setattr(terminal_tool, "_sudo_nopasswd_works", lambda: False)
    monkeypatch.setattr(terminal_tool, "_resolve_container_task_id", lambda _task_id: task_id)
    monkeypatch.setattr(terminal_tool, "resolve_task_overrides", lambda _task_id: {})
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        terminal_tool,
        "_get_env_config",
        lambda: {"env_type": "local", "cwd": str(tmp_path), "timeout": 60},
    )
    environment = MinimalEnvironment()
    monkeypatch.setitem(terminal_tool._active_environments, task_id, environment)
    monkeypatch.setitem(terminal_tool._last_activity, task_id, 0.0)
    terminal_tool.set_sudo_password_callback(lambda: None)
    try:
        result = json.loads(terminal_tool.terminal_tool("sudo shutdown now", force=True))
    finally:
        terminal_tool.set_sudo_password_callback(None)

    assert result == {
        "output": "",
        "exit_code": 130,
        "error": "Command cancelled: sudo password prompt was dismissed.",
        "status": "cancelled",
    }
    assert environment.run_bash_calls == 0


@pytest.mark.parametrize("background", [False, True])
def test_nonlocal_sudo_dismissal_has_foreground_background_parity(
    monkeypatch, tmp_path, background
):
    import tools.process_registry as process_registry_module

    task_id = f"sudo-cancel-{'background' if background else 'foreground'}"
    calls = []

    class FakeEnvironment:
        cwd = str(tmp_path)

        def execute(self, *_args, **_kwargs):
            calls.append("foreground")
            raise terminal_tool.SudoPasswordPromptCancelled

    class FakeRegistry:
        def spawn_via_env(self, **_kwargs):
            calls.append("background")
            raise terminal_tool.SudoPasswordPromptCancelled

    monkeypatch.setattr(terminal_tool, "_resolve_container_task_id", lambda _task_id: task_id)
    monkeypatch.setattr(terminal_tool, "resolve_task_overrides", lambda _task_id: {})
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        terminal_tool,
        "_get_env_config",
        lambda: {
            "env_type": "docker",
            "docker_image": "inert-test-image",
            "cwd": str(tmp_path),
            "timeout": 60,
        },
    )
    monkeypatch.setitem(terminal_tool._active_environments, task_id, FakeEnvironment())
    monkeypatch.setitem(terminal_tool._last_activity, task_id, 0.0)
    monkeypatch.setattr(process_registry_module, "process_registry", FakeRegistry())

    result = json.loads(
        terminal_tool.terminal_tool(
            "sudo true",
            background=background,
            task_id=task_id,
            force=True,
        )
    )

    assert result == {
        "output": "",
        "exit_code": 130,
        "error": "Command cancelled: sudo password prompt was dismissed.",
        "status": "cancelled",
    }
    assert calls == ["background" if background else "foreground"]


@pytest.mark.parametrize("background", [False, True])
def test_nonlocal_pre_start_interrupt_has_foreground_background_parity(
    monkeypatch, tmp_path, background
):
    import tools.process_registry as process_registry_module

    task_id = f"interrupt-cancel-{'background' if background else 'foreground'}"
    calls = []

    class FakeEnvironment:
        cwd = str(tmp_path)

        def execute(self, *_args, **_kwargs):
            calls.append("foreground")
            return {
                "output": "",
                "returncode": 130,
                "_process_start_cancelled": True,
            }

    class FakeRegistry:
        def spawn_via_env(self, **_kwargs):
            calls.append("background")
            raise ProcessStartCancelled

    monkeypatch.setattr(terminal_tool, "_resolve_container_task_id", lambda _task_id: task_id)
    monkeypatch.setattr(terminal_tool, "resolve_task_overrides", lambda _task_id: {})
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        terminal_tool,
        "_get_env_config",
        lambda: {
            "env_type": "docker",
            "docker_image": "inert-test-image",
            "cwd": str(tmp_path),
            "timeout": 60,
        },
    )
    monkeypatch.setitem(terminal_tool._active_environments, task_id, FakeEnvironment())
    monkeypatch.setitem(terminal_tool._last_activity, task_id, 0.0)
    monkeypatch.setattr(process_registry_module, "process_registry", FakeRegistry())

    result = json.loads(
        terminal_tool.terminal_tool(
            "echo never-started",
            background=background,
            task_id=task_id,
            force=True,
        )
    )

    assert result == {
        "output": "[Command interrupted]",
        "exit_code": 130,
        "error": "Command cancelled before process start.",
        "status": "cancelled",
    }
    assert calls == ["background" if background else "foreground"]


@pytest.mark.parametrize(
    "marker",
    [
        "[Command interrupted]",
        "[Command interrupted - Modal sandbox exec cancelled]",
    ],
)
def test_nonlocal_background_marker_interrupt_is_not_reported_started(
    monkeypatch, tmp_path, marker
):
    import tools.process_registry as process_registry_module

    task_id = "marker-interrupt-background"

    class FakeEnvironment:
        cwd = str(tmp_path)

    class FakeRegistry:
        def spawn_via_env(self, **_kwargs):
            raise ProcessStartCancelled(marker, before_start=False)

    monkeypatch.setattr(terminal_tool, "_resolve_container_task_id", lambda _task_id: task_id)
    monkeypatch.setattr(terminal_tool, "resolve_task_overrides", lambda _task_id: {})
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        terminal_tool,
        "_get_env_config",
        lambda: {
            "env_type": "docker",
            "docker_image": "inert-test-image",
            "cwd": str(tmp_path),
            "timeout": 60,
        },
    )
    monkeypatch.setitem(terminal_tool._active_environments, task_id, FakeEnvironment())
    monkeypatch.setitem(terminal_tool._last_activity, task_id, 0.0)
    monkeypatch.setattr(process_registry_module, "process_registry", FakeRegistry())

    result = json.loads(
        terminal_tool.terminal_tool(
            "echo interrupted",
            background=True,
            task_id=task_id,
            force=True,
        )
    )

    assert result == {
        "output": marker,
        "exit_code": 130,
        "error": "Background process start was interrupted.",
        "status": "cancelled",
    }
    assert "session_id" not in result


def test_nonlocal_background_marker_with_pid_is_reported_tracked(
    monkeypatch, tmp_path
):
    import tools.process_registry as process_registry_module

    task_id = "marker-interrupt-background-with-pid"
    marker = "[Command interrupted]"

    class FakeEnvironment:
        cwd = str(tmp_path)

    class FakeRegistry:
        def spawn_via_env(self, **_kwargs):
            return SimpleNamespace(
                id="proc_interrupted",
                pid=4242,
                start_interrupted=True,
                start_interrupt_output=f"4242\n{marker}",
            )

    monkeypatch.setattr(terminal_tool, "_resolve_container_task_id", lambda _task_id: task_id)
    monkeypatch.setattr(terminal_tool, "resolve_task_overrides", lambda _task_id: {})
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        terminal_tool,
        "_get_env_config",
        lambda: {
            "env_type": "ssh",
            "cwd": str(tmp_path),
            "timeout": 60,
        },
    )
    monkeypatch.setitem(terminal_tool._active_environments, task_id, FakeEnvironment())
    monkeypatch.setitem(terminal_tool._last_activity, task_id, 0.0)
    monkeypatch.setattr(process_registry_module, "process_registry", FakeRegistry())

    result = json.loads(
        terminal_tool.terminal_tool(
            "echo interrupted",
            background=True,
            task_id=task_id,
            force=True,
        )
    )

    assert result == {
        "output": (
            "4242\n[Command interrupted]\n"
            "Background process start was interrupted; the process may still "
            "be running and remains tracked."
        ),
        "session_id": "proc_interrupted",
        "pid": 4242,
        "exit_code": 130,
        "error": (
            "Background process start was interrupted after PID capture; "
            "the process may still be running."
        ),
        "status": "running",
        "hint": (
            "background=true without notify_on_complete=true means this process "
            "runs SILENTLY — you will not be told when it exits. If this is a "
            "bounded task (test suite, build, CI poller, deploy, anything with "
            "a defined end), you almost certainly wanted notify_on_complete=true "
            "so the system pings you on exit. Re-launch with "
            "notify_on_complete=true, or call process(action='poll') / "
            "process(action='wait') yourself to learn the outcome. Only ignore "
            "this hint for genuine long-lived processes that never exit "
            "(servers, watchers, daemons)."
        ),
    }


@pytest.mark.parametrize("env_type", ["local", "docker"])
def test_approved_background_clears_stale_interrupt_once_before_spawn(
    monkeypatch, tmp_path, env_type
):
    import tools.interrupt as interrupt_module
    import tools.process_registry as process_registry_module

    task_id = f"approved-background-stale-{env_type}"
    clear_calls = []
    interrupt_states_at_spawn = []
    real_clear = interrupt_module.clear_current_thread_interrupt

    class FakeEnvironment:
        cwd = str(tmp_path)
        env = {}

    class FakeRegistry:
        pending_watchers = []

        @staticmethod
        def _session():
            return SimpleNamespace(id="proc_approved", pid=1234)

        def spawn_local(self, **_kwargs):
            interrupt_states_at_spawn.append(is_interrupted())
            return self._session()

        def spawn_via_env(self, **_kwargs):
            interrupt_states_at_spawn.append(is_interrupted())
            return self._session()

    def tracking_clear():
        clear_calls.append("clear")
        real_clear()

    config = {
        "env_type": env_type,
        "cwd": str(tmp_path),
        "timeout": 60,
    }
    if env_type == "docker":
        config["docker_image"] = "inert-test-image"

    monkeypatch.setattr(terminal_tool, "_resolve_container_task_id", lambda _task_id: task_id)
    monkeypatch.setattr(terminal_tool, "resolve_task_overrides", lambda _task_id: {})
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: config)
    monkeypatch.setattr(interrupt_module, "clear_current_thread_interrupt", tracking_clear)
    monkeypatch.setitem(terminal_tool._active_environments, task_id, FakeEnvironment())
    monkeypatch.setitem(terminal_tool._last_activity, task_id, 0.0)
    monkeypatch.setattr(process_registry_module, "process_registry", FakeRegistry())

    set_interrupt(True)
    try:
        result = json.loads(
            terminal_tool.terminal_tool(
                "echo approved",
                background=True,
                task_id=task_id,
                force=True,
            )
        )
    finally:
        set_interrupt(False)

    assert result["output"] == "Background process started"
    assert result["exit_code"] == 0
    assert interrupt_states_at_spawn == [False]
    assert clear_calls == ["clear"]


def test_approved_nonlocal_background_preserves_genuine_interrupt_after_clear(
    monkeypatch, tmp_path
):
    import tools.interrupt as interrupt_module
    import tools.process_registry as process_registry_module

    task_id = "approved-background-genuine-interrupt"
    clear_calls = []
    real_clear = interrupt_module.clear_current_thread_interrupt

    class FakeEnvironment:
        cwd = str(tmp_path)

    class FakeRegistry:
        def spawn_via_env(self, **_kwargs):
            assert not is_interrupted()
            set_interrupt(True)
            raise ProcessStartCancelled

    def tracking_clear():
        clear_calls.append("clear")
        real_clear()

    monkeypatch.setattr(terminal_tool, "_resolve_container_task_id", lambda _task_id: task_id)
    monkeypatch.setattr(terminal_tool, "resolve_task_overrides", lambda _task_id: {})
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        terminal_tool,
        "_get_env_config",
        lambda: {
            "env_type": "docker",
            "docker_image": "inert-test-image",
            "cwd": str(tmp_path),
            "timeout": 60,
        },
    )
    monkeypatch.setattr(interrupt_module, "clear_current_thread_interrupt", tracking_clear)
    monkeypatch.setitem(terminal_tool._active_environments, task_id, FakeEnvironment())
    monkeypatch.setitem(terminal_tool._last_activity, task_id, 0.0)
    monkeypatch.setattr(process_registry_module, "process_registry", FakeRegistry())

    set_interrupt(True)
    try:
        result = json.loads(
            terminal_tool.terminal_tool(
                "echo interrupted",
                background=True,
                task_id=task_id,
                force=True,
            )
        )
        assert is_interrupted()
    finally:
        set_interrupt(False)

    assert result == {
        "output": "[Command interrupted]",
        "exit_code": 130,
        "error": "Command cancelled before process start.",
        "status": "cancelled",
    }
    assert clear_calls == ["clear"]


def test_global_interrupt_after_sudo_response_dequeue_stops_process_start(
    monkeypatch, tmp_path
):
    task_id = "sudo-post-dequeue-interrupt-test"
    response_dequeued = threading.Event()
    resume_callback = threading.Event()
    real_queue = queue.Queue

    class PausingQueue(real_queue):
        def get(self, block=True, timeout=None):
            result = super().get(block=block, timeout=timeout)
            response_dequeued.set()
            assert resume_callback.wait(timeout=2), "callback did not resume"
            return result

    class MinimalEnvironment(BaseEnvironment):
        def __init__(self):
            super().__init__(cwd=str(tmp_path), timeout=60)
            self.run_bash_calls = 0

        def _run_bash(self, *_args, **_kwargs):
            self.run_bash_calls += 1
            return object()

        def _wait_for_process(self, *_args, **_kwargs):
            return {"output": "executed", "returncode": 0}

        def cleanup(self):
            pass

    cli = HermesCLI.__new__(HermesCLI)
    cli._approval_state = None
    cli._clarify_state = None
    cli._clarify_freetext = False
    cli._sudo_state = None
    cli._sudo_deadline = 0
    cli._sudo_lock = threading.Lock()
    cli._sudo_state_lock = threading.Lock()
    cli._sudo_interrupt_generation = 0
    cli._secret_state = None
    cli._modal_input_snapshot = None
    cli._app = SimpleNamespace(current_buffer=MagicMock())
    cli._paint_now = MagicMock()
    cli._capture_modal_input_snapshot = MagicMock()
    cli._restore_modal_input_snapshot = MagicMock()

    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    monkeypatch.setattr(terminal_tool, "_sudo_nopasswd_works", lambda: False)
    monkeypatch.setattr(terminal_tool, "_resolve_container_task_id", lambda _task_id: task_id)
    monkeypatch.setattr(terminal_tool, "resolve_task_overrides", lambda _task_id: {})
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        terminal_tool,
        "_get_env_config",
        lambda: {"env_type": "local", "cwd": str(tmp_path), "timeout": 60},
    )
    environment = MinimalEnvironment()
    monkeypatch.setitem(terminal_tool._active_environments, task_id, environment)
    monkeypatch.setitem(terminal_tool._last_activity, task_id, 0.0)

    result = {}

    def run_terminal():
        terminal_tool.set_sudo_password_callback(cli._sudo_password_callback)
        try:
            result["value"] = json.loads(
                terminal_tool.terminal_tool("sudo shutdown now", force=True)
            )
        finally:
            terminal_tool.set_sudo_password_callback(None)

    with patch.object(cli_module.queue, "Queue", PausingQueue), patch.object(
        cli_module, "_cprint"
    ):
        worker = threading.Thread(target=run_terminal, daemon=True)
        worker.start()

        deadline = time.time() + 2
        while cli._sudo_state is None and time.time() < deadline:
            time.sleep(0.01)
        state = cli._sudo_state
        assert state is not None
        assert cli._resolve_sudo_prompt(state, "typed-password")
        assert response_dequeued.wait(timeout=2), "sudo response was not dequeued"

        generation_before = cli._sudo_interrupt_generation
        cli._clear_active_overlays_for_interrupt()
        assert cli._sudo_interrupt_generation == generation_before + 1
        set_interrupt(True, thread_id=worker.ident)
        resume_callback.set()

        worker.join(timeout=2)
        set_interrupt(False, thread_id=worker.ident)

    assert not worker.is_alive()
    assert result["value"] == {
        "output": "[Command interrupted]",
        "exit_code": 130,
        "error": "Command cancelled before process start.",
        "status": "cancelled",
    }
    assert environment.run_bash_calls == 0


def test_cached_sudo_password_isolated_by_session_key(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    monkeypatch.setenv("HERMES_SESSION_KEY", "session-a")
    terminal_tool._set_cached_sudo_password("alpha-pass")

    monkeypatch.setenv("HERMES_SESSION_KEY", "session-b")
    assert terminal_tool._get_cached_sudo_password() == ""

    monkeypatch.setenv("HERMES_SESSION_KEY", "session-a")
    assert terminal_tool._get_cached_sudo_password() == "alpha-pass"


def test_passwordless_sudo_skips_interactive_prompt_and_rewrite(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.delenv("TERMINAL_ENV", raising=False)
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")

    def _fail_prompt(*_args, **_kwargs):
        raise AssertionError(
            "interactive sudo prompt should not run when sudo -n already works"
        )

    monkeypatch.setattr(terminal_tool, "_prompt_for_sudo_password", _fail_prompt)
    monkeypatch.setattr(terminal_tool, "_sudo_nopasswd_works", lambda: True, raising=False)

    transformed, sudo_stdin = terminal_tool._transform_sudo_command("sudo whoami")

    assert transformed == "sudo whoami"
    assert sudo_stdin is None


def test_passwordless_sudo_probe_rechecks_local_terminal(monkeypatch):
    monkeypatch.delenv("TERMINAL_ENV", raising=False)
    calls = []

    class Result:
        def __init__(self, returncode):
            self.returncode = returncode

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return Result(0 if len(calls) == 1 else 1)

    monkeypatch.setattr(terminal_tool.subprocess, "run", fake_run)

    assert terminal_tool._sudo_nopasswd_works() is True
    assert terminal_tool._sudo_nopasswd_works() is False
    assert len(calls) == 2
    assert calls[0][0] == ["sudo", "-n", "true"]
    assert calls[1][0] == ["sudo", "-n", "true"]


def test_passwordless_sudo_probe_is_disabled_for_nonlocal_terminal_env(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "docker")

    def _fail_run(*_args, **_kwargs):
        raise AssertionError("host sudo probe must not run for non-local terminal envs")

    monkeypatch.setattr(terminal_tool.subprocess, "run", _fail_run)

    assert terminal_tool._sudo_nopasswd_works() is False


def test_validate_workdir_allows_windows_drive_paths():
    assert terminal_tool._validate_workdir(r"C:\Users\Alice\project") is None
    assert terminal_tool._validate_workdir("C:/Users/Alice/project") is None


def test_validate_workdir_allows_windows_unc_paths():
    assert terminal_tool._validate_workdir(r"\\server\share\project") is None


def test_validate_workdir_blocks_shell_metacharacters_in_windows_paths():
    assert terminal_tool._validate_workdir(r"C:\Users\Alice\project; rm -rf /")
    assert terminal_tool._validate_workdir(r"C:\Users\Alice\project$(whoami)")
    assert terminal_tool._validate_workdir("C:\\Users\\Alice\\project\nwhoami")


def test_get_env_config_ignores_bad_docker_json_for_local_backend(monkeypatch):
    """Docker-only JSON env vars must not break the default local backend."""
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_DOCKER_VOLUMES", "None")
    monkeypatch.setenv("TERMINAL_DOCKER_ENV", "not-json")
    monkeypatch.setenv("TERMINAL_DOCKER_FORWARD_ENV", "not-json")
    monkeypatch.setenv("TERMINAL_DOCKER_EXTRA_ARGS", "not-json")

    config = terminal_tool._get_env_config()

    assert config["env_type"] == "local"
    assert config["docker_volumes"] == []
    assert config["docker_env"] == {}
    assert config["docker_forward_env"] == []
    assert config["docker_extra_args"] == []


def test_get_env_config_ignores_bad_docker_json_for_ssh_backend(monkeypatch):
    """Non-container remote backends should also ignore Docker-only JSON."""
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    monkeypatch.setenv("TERMINAL_DOCKER_VOLUMES", "None")
    monkeypatch.setenv("TERMINAL_DOCKER_ENV", "not-json")

    config = terminal_tool._get_env_config()

    assert config["env_type"] == "ssh"
    assert config["docker_volumes"] == []
    assert config["docker_env"] == {}


def test_get_env_config_preserves_ssh_tilde_cwd(monkeypatch):
    """SSH cwd '~' is expanded by the remote shell, not the Hermes host."""
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    monkeypatch.setenv("TERMINAL_CWD", "~")
    monkeypatch.setenv("HOME", "/opt/data")

    config = terminal_tool._get_env_config()

    assert config["env_type"] == "ssh"
    assert config["cwd"] == "~"


def test_get_env_config_preserves_ssh_tilde_child_cwd(monkeypatch):
    """SSH cwd '~/x' must not become the local/container HOME path."""
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    monkeypatch.setenv("TERMINAL_CWD", "~/project")
    monkeypatch.setenv("HOME", "/opt/data")

    config = terminal_tool._get_env_config()

    assert config["env_type"] == "ssh"
    assert config["cwd"] == "~/project"


def test_get_env_config_still_rejects_bad_docker_json_for_docker_backend(monkeypatch):
    """Selecting Docker should keep the existing actionable config error."""
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_DOCKER_VOLUMES", "None")

    try:
        terminal_tool._get_env_config()
    except ValueError as exc:
        assert "TERMINAL_DOCKER_VOLUMES" in str(exc)
    else:
        raise AssertionError("Docker backend must validate TERMINAL_DOCKER_VOLUMES")


def test_sudo_wrong_password_failure_detects_rejection_output():
    output = (
        "sudo: Authentication failed, try again.\n\n"
        "sudo: maximum 3 incorrect authentication attempts\n"
    )
    assert terminal_tool._sudo_wrong_password_failure(output) is True


def test_sudo_wrong_password_failure_ignores_tty_required_message():
    output = "sudo: a terminal is required to authenticate"
    assert terminal_tool._sudo_wrong_password_failure(output) is False


def test_invalidate_cached_sudo_on_auth_failure_clears_session_cache(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    terminal_tool._set_cached_sudo_password("wrong-pass")

    cleared = terminal_tool._invalidate_cached_sudo_on_auth_failure(
        "sudo apt install fprintd",
        "sudo: Authentication failed, try again.",
    )

    assert cleared is True
    assert terminal_tool._get_cached_sudo_password() == ""


def test_invalidate_cached_sudo_on_auth_failure_keeps_env_password(monkeypatch):
    monkeypatch.setenv("SUDO_PASSWORD", "from-env")
    terminal_tool._set_cached_sudo_password("wrong-pass")

    cleared = terminal_tool._invalidate_cached_sudo_on_auth_failure(
        "sudo true",
        "sudo: Authentication failed, try again.",
    )

    assert cleared is False
    assert terminal_tool._get_cached_sudo_password() == "wrong-pass"


def test_transform_sudo_command_pipes_one_password_line_per_invocation(monkeypatch):
    monkeypatch.setenv("SUDO_PASSWORD", "testpass")
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    transformed, sudo_stdin = terminal_tool._transform_sudo_command(
        "sudo true && sudo whoami"
    )

    assert transformed == "sudo -S -p '' true && sudo -S -p '' whoami"
    assert sudo_stdin == "testpass\ntestpass\n"


def test_count_real_sudo_invocations_ignores_mentions(monkeypatch):
    assert terminal_tool._count_real_sudo_invocations("grep sudo README.md") == 0
    assert terminal_tool._count_real_sudo_invocations("sudo a; sudo b") == 2
