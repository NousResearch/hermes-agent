"""SSH remote execution environment with ControlMaster connection persistence."""

import logging
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

from tools.environments.base import BaseEnvironment, _popen_bash

logger = logging.getLogger(__name__)


def _ensure_ssh_available() -> None:
    """Fail fast with a clear error when the SSH client is unavailable."""
    if not shutil.which("ssh"):
        raise RuntimeError(
            "SSH is not installed or not in PATH. Install OpenSSH client: apt install openssh-client"
        )


class SSHEnvironment(BaseEnvironment):
    """Run commands on a remote machine over SSH.

    Spawn-per-call: every execute() spawns a fresh ``ssh ... bash -c`` process.
    Session snapshot preserves env vars across calls.
    CWD persists via in-band stdout markers.
    Uses SSH ControlMaster for connection reuse.
    """

    def __init__(self, host: str, user: str, cwd: str = "~",
                 timeout: int = 60, port: int = 22, key_path: str = ""):
        super().__init__(cwd=cwd, timeout=timeout)
        self.host = host
        self.user = user
        self.port = port
        self.key_path = key_path

        self.control_dir = Path(tempfile.gettempdir()) / "hermes-ssh"
        self.control_dir.mkdir(parents=True, exist_ok=True)
        self.control_socket = self.control_dir / f"{user}@{host}:{port}.sock"
        _ensure_ssh_available()
        self._establish_connection()
        self._remote_home = self._detect_remote_home()
        self._last_sync_time: float = 0  # guarantees first _before_execute syncs
        self._sync_files()

        self.init_session()

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
        if self.user == "root":
            return "/root"
        return f"/home/{self.user}"

    def _sync_files(self) -> None:
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

    def _run_bash(self, cmd_string: str, *, login: bool = False,
                  timeout: int = 120,
                  stdin_data: str | None = None) -> subprocess.Popen:
        """Spawn an SSH process that runs bash on the remote host."""
        cmd = self._build_ssh_command()
        if login:
            cmd.extend(["bash", "-l", "-c", shlex.quote(cmd_string)])
        else:
            cmd.extend(["bash", "-c", shlex.quote(cmd_string)])

        return _popen_bash(cmd, stdin_data)

    def cleanup(self):
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
