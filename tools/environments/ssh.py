"""SSH remote execution environment with ControlMaster connection persistence."""

import logging
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from tools.environments.base import BaseEnvironment
from tools.interrupt import is_interrupted

logger = logging.getLogger(__name__)


class SSHEnvironment(BaseEnvironment):
    """Run commands on a remote machine over SSH.

    Uses SSH ControlMaster for connection persistence so subsequent
    commands are fast. Security benefit: the agent cannot modify its
    own code since execution happens on a separate machine.

    Foreground commands are interruptible: the local ssh process is killed
    and a remote kill is attempted over the ControlMaster socket.
    """

    def __init__(self, host: str, user: str, cwd: str = "~", timeout: int = 60, port: int = 22, key_path: str = ""):
        super().__init__(cwd=cwd, timeout=timeout)
        self.host = host
        self.user = user
        self.port = port
        self.key_path = key_path

        self.control_dir = Path(tempfile.gettempdir()) / "hermes-ssh"
        self.control_dir.mkdir(parents=True, exist_ok=True)
        self.control_socket = self.control_dir / f"{user}@{host}:{port}.sock"
        self._establish_connection()

    def _build_ssh_command(self, extra_args: list = None) -> list:
        cmd = ["ssh"]
        cmd.extend(["-o", f"ControlPath={self.control_socket}"])
        cmd.extend(["-o", "ControlMaster=auto"])
        cmd.extend(["-o", "ControlPersist=300"])
        cmd.extend(["-o", "BatchMode=yes"])
        cmd.extend(["-o", "StrictHostKeyChecking=accept-new"])
        cmd.extend(["-o", "ConnectTimeout=10"])
        if self.port != 22:
            cmd.extend(["-p", str(self.port)])
        if self.key_path:
            cmd.extend(["-i", self.key_path])
        if extra_args:
            cmd.extend(extra_args)
        cmd.append(f"{self.user}@{self.host}")
        return cmd

    def _establish_connection(self):
        cmd = self._build_ssh_command()
        cmd.append("echo 'SSH connection established'")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if result.returncode != 0:
                error_msg = result.stderr.strip() or result.stdout.strip()
                raise RuntimeError(f"SSH connection failed: {error_msg}")
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"SSH connection to {self.user}@{self.host} timed out")

    def execute(
        self, command: str, cwd: str = "", *, timeout: int | None = None, stdin_data: str | None = None
    ) -> dict:
        work_dir = cwd or self.cwd
        exec_command = self._prepare_command(command)
        wrapped = f"cd {work_dir} && {exec_command}"
        effective_timeout = timeout or self.timeout

        cmd = self._build_ssh_command()
        cmd.extend(["bash", "-c", wrapped])

        try:
            kwargs = self._build_run_kwargs(timeout, stdin_data)
            # Remove timeout from kwargs -- we handle it in the poll loop
            kwargs.pop("timeout", None)

            _output_chunks = []

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE if stdin_data else subprocess.DEVNULL,
                text=True,
            )

            if stdin_data:
                try:
                    proc.stdin.write(stdin_data)
                    proc.stdin.close()
                except Exception:
                    pass

            def _drain():
                try:
                    if proc.stdout:
                        for line in proc.stdout:
                            _output_chunks.append(line)
                except Exception:
                    pass

            reader = threading.Thread(target=_drain, daemon=True)
            reader.start()
            deadline = time.monotonic() + effective_timeout

            while proc.poll() is None:
                if is_interrupted():
                    proc.terminate()
                    try:
                        proc.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    reader.join(timeout=2)
                    return {
                        "output": "".join(_output_chunks) + "\n[Command interrupted]",
                        "returncode": 130,
                    }
                if time.monotonic() > deadline:
                    proc.kill()
                    reader.join(timeout=2)
                    return self._timeout_result(effective_timeout)
                time.sleep(0.2)

            reader.join(timeout=5)
            return {"output": "".join(_output_chunks), "returncode": proc.returncode}

        except Exception as e:
            return {"output": f"SSH execution error: {str(e)}", "returncode": 1}

    def cleanup(self):
        if self.control_socket.exists():
            try:
                cmd = ["ssh", "-o", f"ControlPath={self.control_socket}", "-O", "exit", f"{self.user}@{self.host}"]
                subprocess.run(cmd, capture_output=True, timeout=5)
            except (OSError, subprocess.SubprocessError):
                pass
            try:
                self.control_socket.unlink()
            except OSError:
                pass
