"""SSH remote execution environment with ControlMaster connection persistence."""

import logging
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from tools.environments.base import BaseEnvironment
from tools.environments.persistent_shell import PersistentShellMixin
from tools.interrupt import is_interrupted

logger = logging.getLogger(__name__)


def _ensure_ssh_available() -> None:
    """Fail fast with a clear error when the SSH client is unavailable."""
    if not shutil.which("ssh"):
        raise RuntimeError(
            "SSH is not installed or not in PATH. Install OpenSSH client: apt install openssh-client"
        )


class SSHEnvironment(PersistentShellMixin, BaseEnvironment):
    """Run commands on a remote machine over SSH.

    Uses SSH ControlMaster for connection persistence so subsequent
    commands are fast. Security benefit: the agent cannot modify its
    own code since execution happens on a separate machine.

    Foreground commands are interruptible: the local ssh process is killed
    and a remote kill is attempted over the ControlMaster socket.

    When ``persistent=True``, a single long-lived bash shell is kept alive
    over SSH and state (cwd, env vars, shell variables) persists across
    ``execute()`` calls.  Output capture uses file-based IPC on the remote
    host (stdout/stderr/exit-code written to temp files, polled via fast
    ControlMaster one-shot reads).
    """

    def __init__(self, host: str, user: str, cwd: str = "~",
                 timeout: int = 60, port: int = 22, key_path: str = "",
                 persistent: bool = False):
        super().__init__(cwd=cwd, timeout=timeout)
        self.host = host
        self.user = user
        self.port = port
        self.key_path = key_path
        self.persistent = persistent

        self.control_dir = Path(tempfile.gettempdir()) / "hermes-ssh"
        self.control_dir.mkdir(parents=True, exist_ok=True)
        self.control_socket = self.control_dir / f"{user}@{host}:{port}.sock"
        _ensure_ssh_available()
        self._establish_connection()
        self._remote_home = self._detect_remote_home()
        self._sync_skills_and_credentials()

        if self.persistent:
            self._init_persistent_shell()

    def _build_ssh_command(self, extra_args: list | None = None) -> list:
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

    def _detect_remote_home(self) -> str:
        """Detect the remote user's home directory."""
        try:
            cmd = self._build_ssh_command()
            cmd.append("echo $HOME")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            home = result.stdout.strip()
            if home and result.returncode == 0:
                logger.debug("SSH: remote home = %s", home)
                return home
        except Exception:
            pass
        # Fallback: guess from username
        if self.user == "root":
            return "/root"
        return f"/home/{self.user}"

    def _sync_skills_and_credentials(self) -> None:
        """Rsync skills directory and credential files to the remote host."""
        try:
            container_base = f"{self._remote_home}/.hermes"
            from tools.credential_files import get_credential_file_mounts, get_skills_directory_mount

            rsync_base = ["rsync", "-az", "--timeout=30", "--safe-links"]
            ssh_opts = f"ssh -o ControlPath={self.control_socket} -o ControlMaster=auto"
            if self.port != 22:
                ssh_opts += f" -p {self.port}"
            if self.key_path:
                ssh_opts += f" -i {self.key_path}"
            rsync_base.extend(["-e", ssh_opts])
            dest_prefix = f"{self.user}@{self.host}"

            # Sync individual credential files (remap /root/.hermes to detected home)
            for mount_entry in get_credential_file_mounts():
                remote_path = mount_entry["container_path"].replace("/root/.hermes", container_base, 1)
                parent_dir = str(Path(remote_path).parent)
                mkdir_cmd = self._build_ssh_command()
                mkdir_cmd.append(f"mkdir -p {parent_dir}")
                subprocess.run(mkdir_cmd, capture_output=True, text=True, timeout=10)
                cmd = rsync_base + [mount_entry["host_path"], f"{dest_prefix}:{remote_path}"]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    logger.info("SSH: synced credential %s -> %s", mount_entry["host_path"], remote_path)
                else:
                    logger.debug("SSH: rsync credential failed: %s", result.stderr.strip())

            # Sync skill directories (local + external, remap to detected home)
            for skills_mount in get_skills_directory_mount(container_base=container_base):
                remote_path = skills_mount["container_path"]
                mkdir_cmd = self._build_ssh_command()
                mkdir_cmd.append(f"mkdir -p {remote_path}")
                subprocess.run(mkdir_cmd, capture_output=True, text=True, timeout=10)
                cmd = rsync_base + [
                    skills_mount["host_path"].rstrip("/") + "/",
                    f"{dest_prefix}:{remote_path}/",
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                if result.returncode == 0:
                    logger.info("SSH: synced skills dir %s -> %s", skills_mount["host_path"], remote_path)
                else:
                    logger.debug("SSH: rsync skills dir failed: %s", result.stderr.strip())
        except Exception as e:
            logger.debug("SSH: could not sync skills/credentials: %s", e)

    def execute(self, command: str, cwd: str = "", *,
                timeout: int | None = None,
                stdin_data: str | None = None) -> dict:
        # Incremental sync before each command so mid-session credential
        # refreshes and skill updates are picked up.
        self._sync_skills_and_credentials()
        return super().execute(command, cwd, timeout=timeout, stdin_data=stdin_data)

    _poll_interval_start: float = 0.15  # SSH: higher initial interval (150ms) for network latency

    @property
    def _temp_prefix(self) -> str:
        return f"/tmp/hermes-ssh-{self._session_id}"

    def _spawn_shell_process(self) -> subprocess.Popen:
        cmd = self._build_ssh_command()
        cmd.append("bash -l")
        return subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )

    def _read_temp_files(self, *paths: str) -> list[str]:
        if len(paths) == 1:
            cmd = self._build_ssh_command()
            cmd.append(f"cat {paths[0]} 2>/dev/null")
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=10,
                )
                return [result.stdout]
            except (subprocess.TimeoutExpired, OSError):
                return [""]

        delim = f"__HERMES_SEP_{self._session_id}__"
        script = "; ".join(
            f"cat {p} 2>/dev/null; echo '{delim}'" for p in paths
        )
        cmd = self._build_ssh_command()
        cmd.append(script)
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=10,
            )
            parts = result.stdout.split(delim + "\n")
            return [parts[i] if i < len(parts) else "" for i in range(len(paths))]
        except (subprocess.TimeoutExpired, OSError):
            return [""] * len(paths)

    def _kill_shell_children(self):
        if self._shell_pid is None:
            return
        cmd = self._build_ssh_command()
        cmd.append(f"pkill -P {self._shell_pid} 2>/dev/null; true")
        try:
            subprocess.run(cmd, capture_output=True, timeout=5)
        except (subprocess.TimeoutExpired, OSError):
            pass

    def _cleanup_temp_files(self):
        cmd = self._build_ssh_command()
        cmd.append(f"rm -f {self._temp_prefix}-*")
        try:
            subprocess.run(cmd, capture_output=True, timeout=5)
        except (subprocess.TimeoutExpired, OSError):
            pass

    def _execute_oneshot(self, command: str, cwd: str = "", *,
                         timeout: int | None = None,
                         stdin_data: str | None = None) -> dict:
        work_dir = cwd or self.cwd
        exec_command, sudo_stdin = self._prepare_command(command)
        # Keep ~ unquoted (for shell expansion) and quote only the subpath.
        if work_dir == "~":
            wrapped = f'cd ~ && {exec_command}'
        elif work_dir.startswith("~/"):
            wrapped = f'cd ~/{shlex.quote(work_dir[2:])} && {exec_command}'
        else:
            wrapped = f'cd {shlex.quote(work_dir)} && {exec_command}'
        effective_timeout = timeout or self.timeout

        if sudo_stdin is not None and stdin_data is not None:
            effective_stdin = sudo_stdin + stdin_data
        elif sudo_stdin is not None:
            effective_stdin = sudo_stdin
        else:
            effective_stdin = stdin_data

        cmd = self._build_ssh_command()
        cmd.append(wrapped)

        kwargs = self._build_run_kwargs(timeout, effective_stdin)
        kwargs.pop("timeout", None)
        _output_chunks = []
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE if effective_stdin else subprocess.DEVNULL,
            text=True,
        )

        if effective_stdin:
            try:
                proc.stdin.write(effective_stdin)
                proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass

        def _drain():
            try:
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

    def cleanup(self):
        super().cleanup()
        if self.control_socket.exists():
            try:
                cmd = ["ssh", "-o", f"ControlPath={self.control_socket}",
                       "-O", "exit", f"{self.user}@{self.host}"]
                subprocess.run(cmd, capture_output=True, timeout=5)
            except (OSError, subprocess.SubprocessError):
                pass
            try:
                self.control_socket.unlink()
            except OSError:
                pass
