"""Regression coverage for the remote-kernel spawn broken by #68948.

The fake environment inherits the shared BaseEnvironment.execute() path while
replacing only its shell/process boundary. No Docker daemon, SSH connection,
Modal client, subprocess, or POSIX shell is required.
"""

import json
import shlex
from types import SimpleNamespace
from unittest.mock import patch

import tools.code_kernel_remote as remote_kernel
from tools.environments.base import BaseEnvironment


class _SharedExecuteEnv(BaseEnvironment):
    """Dependency-light remote whose commands still traverse shared execute()."""

    def __init__(self, *, spawn_success=True):
        self.timeout = 30
        self.cwd = ""
        self._stdin_mode = "none"
        self._snapshot_ready = True
        self._prefer_nonlogin = False

        self.spawn_success = spawn_success
        self.background_pid = "4242" if spawn_success else None
        self.runner_alive = False
        self.shell_commands = []
        self.shipped = {}
        self.cell_codes = []
        self.cell_payloads = []
        self.namespace = {}
        self.execution_count = 0
        self.spawn_attempts = 0
        self.spawn_returncodes = []

    def get_temp_dir(self):
        return "/tmp"

    def ship_file(self, path, content):
        self.shipped[path] = content
        if "/cells/cell_req_" not in path:
            return

        request = json.loads(content)
        code = request["code"]
        self.cell_codes.append(code)
        self.execution_count += 1

        status = "ok"
        stdout = ""
        trace = ""
        if code == "counter = 41":
            self.namespace["counter"] = 41
        elif code == "print(counter)" and "counter" in self.namespace:
            stdout = f"{self.namespace['counter']}\n"
        else:
            status = "error"
            trace = "NameError: unsupported fake cell"

        self.cell_payloads.append({
            "id": request["id"],
            "status": status,
            "stdout": stdout,
            "stderr": "",
            "stdout_clipped": False,
            "stderr_clipped": False,
            "traceback": trace,
            "execution_count": self.execution_count,
        })

    def _before_execute(self):
        pass

    def _prepare_command(self, command):
        return command, None

    def _wrap_command(self, command, cwd):
        return command

    def _run_bash(self, command, *, login=False, timeout=None, stdin_data=None):
        self.shell_commands.append(command)
        return command

    def _wait_for_process(
        self, proc, *, timeout=None, bounded_capture=False, watch_interrupt_tid=None
    ):
        command = proc

        if "nohup env " in command:
            self.spawn_attempts += 1
            if self.spawn_success:
                self.runner_alive = True
                self.spawn_returncodes.append(0)
                return {
                    "output": f"PID:{self.background_pid}\n",
                    "returncode": 0,
                }
            self.spawn_returncodes.append(1)
            return {"output": "sh: cannot fork\n", "returncode": 1}

        if command.startswith("kill -0 "):
            if self.runner_alive:
                return {"output": "ALIVE\n", "returncode": 0}
            return {"output": "", "returncode": 1}

        if command.startswith("cat ") and "cell_res_" in command:
            if self.cell_payloads:
                return {
                    "output": json.dumps(self.cell_payloads.pop(0)),
                    "returncode": 0,
                }
            return {"output": "", "returncode": 1}

        return {"output": "", "returncode": 0}

    def _update_cwd(self, result):
        pass

    def cleanup(self):
        pass


def _ship_to_fake(env, path, content):
    env.ship_file(path, content)


def _execute_code(env, code, *, task):
    return remote_kernel.execute_in_remote_kernel(
        code,
        env=env,
        env_type="ssh",
        task_env_id=task,
        sandbox_tools=frozenset({"read_file"}),
        timeout=10,
        max_tool_calls=5,
        reset=False,
    )


def test_remote_spawn_command_pid_persistence_and_failure():
    remote_kernel.shutdown_all_remote_kernels()
    healthy = _SharedExecuteEnv()
    failed_env = _SharedExecuteEnv(spawn_success=False)
    fixed_uuid = SimpleNamespace(hex="0123456789abcdef0123456789abcdef")
    fixed_token = "fixed-token"

    try:
        with patch.object(
            remote_kernel.uuid,
            "uuid4",
            return_value=fixed_uuid,
        ), patch(
            "secrets.token_urlsafe",
            return_value=fixed_token,
        ), patch(
            "tools.code_execution_tool._ship_file_to_remote",
            side_effect=_ship_to_fake,
        ), patch(
            "tools.code_execution_tool._rpc_poll_loop",
            new=lambda *args, **kwargs: None,
        ):
            first = _execute_code(
                healthy,
                "counter = 41",
                task="spawn-regression",
            )
            second = _execute_code(
                healthy,
                "print(counter)",
                task="spawn-regression",
            )

            runner_path = next(
                path for path in healthy.shipped
                if path.endswith("/kernel_runner.py")
            )
            kernel_dir = runner_path.rsplit("/", 1)[0]
            q_dir = shlex.quote(kernel_dir)
            env_prefix = (
                f"HERMES_KERNEL_DIR={q_dir} "
                f"HERMES_RPC_DIR={shlex.quote(kernel_dir + '/rpc')} "
                f"HERMES_RPC_TOKEN={shlex.quote(fixed_token)} "
                f"PYTHONDONTWRITEBYTECODE=1 PYTHONPATH={q_dir}"
            )
            expected_spawn = (
                f"cd {q_dir} && nohup env {env_prefix} "
                f"python3 kernel_runner.py > {q_dir}/runner.log 2>&1 "
                f"& echo PID:$!"
            )
            spawn_commands = [
                command for command in healthy.shell_commands
                if "nohup env " in command
            ]

            # Command identity is independent of PID parsing and registration.
            assert spawn_commands == [expected_spawn]
            assert spawn_commands[0].endswith(" & echo PID:$!")
            assert "&& { nohup" not in spawn_commands[0]
            assert "& } echo PID:$!" not in spawn_commands[0]

            registered = [
                kernel for kernel in remote_kernel._REMOTE_KERNELS.values()
                if kernel.env is healthy
            ]
            assert len(registered) == 1
            assert registered[0].pid is not None
            assert registered[0].pid == healthy.background_pid
            assert registered[0].pid == "4242"

            # Persistence evidence is separate from command and PID evidence.
            assert first is not None
            assert first["status"] == "success"
            assert first["kernel"]["reused"] is False
            assert second is not None
            assert second["status"] == "success"
            assert second["kernel"]["reused"] is True
            assert second["stdout"] == "41\n"
            assert second["kernel"]["execution_count"] == 2
            assert healthy.cell_codes == ["counter = 41", "print(counter)"]
            assert healthy.spawn_attempts == 1

            remote_kernel.shutdown_all_remote_kernels()

            failed = _execute_code(
                failed_env,
                "counter = 41",
                task="failed-spawn-regression",
            )

            # A real nonzero spawn with no PID marker must fail open and must
            # never become indistinguishable from the registered healthy run.
            assert failed_env.spawn_attempts == 1
            assert failed_env.spawn_returncodes == [1]
            assert failed is None
            assert not [
                kernel for kernel in remote_kernel._REMOTE_KERNELS.values()
                if kernel.env is failed_env
            ]
            assert len(remote_kernel._REMOTE_KERNELS) == 0
    finally:
        remote_kernel.shutdown_all_remote_kernels()
