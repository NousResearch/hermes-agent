"""Docker execution environment wrapping mini-swe-agent's DockerEnvironment."""

import os
import subprocess

from tools.environments.base import BaseEnvironment


class DockerEnvironment(BaseEnvironment):
    """Docker container execution via mini-swe-agent.

    Wraps the upstream DockerEnvironment and adds non-blocking stdin
    and sudo -S support.
    """

    def __init__(self, image: str, cwd: str = "/", timeout: int = 60):
        super().__init__(cwd=cwd, timeout=timeout)
        from minisweagent.environments.docker import DockerEnvironment as _Docker
        self._inner = _Docker(image=image, cwd=cwd, timeout=timeout)

    def execute(self, command: str, cwd: str = "", *,
                timeout: int | None = None,
                stdin_data: str | None = None) -> dict:
        exec_command = self._prepare_command(command)
        work_dir = cwd or self.cwd
        effective_timeout = timeout or self.timeout

        assert self._inner.container_id, "Container not started"
        cmd = [self._inner.config.executable, "exec"]
        if stdin_data is not None:
            cmd.append("-i")
        cmd.extend(["-w", work_dir])
        for key in self._inner.config.forward_env:
            if (value := os.getenv(key)) is not None:
                cmd.extend(["-e", f"{key}={value}"])
        for key, value in self._inner.config.env.items():
            cmd.extend(["-e", f"{key}={value}"])
        cmd.extend([self._inner.container_id, "bash", "-lc", exec_command])

        try:
            result = subprocess.run(cmd, **self._build_run_kwargs(timeout, stdin_data))
            return {"output": result.stdout, "returncode": result.returncode}
        except subprocess.TimeoutExpired:
            return self._timeout_result(effective_timeout)

    def cleanup(self):
        self._inner.cleanup()
