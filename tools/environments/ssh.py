"""SSH remote execution environment with ControlMaster connection persistence."""

import contextlib
import hashlib
import logging
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

from tools.environments.base import BaseEnvironment, EnvironmentConnectionError
from tools.environments.base_output import _popen_bash
from tools.environments.file_sync import (
    FileSyncManager, iter_sync_files, quoted_mkdir_command, quoted_rm_command, unique_parent_dirs)
from tools.environments.remote_common import bash_argv, run_capture

logger = logging.getLogger(__name__)

# Windows OpenSSH has no Unix-socket ControlMaster: ControlPath/ControlMaster options
# fail the connection outright ('getsockname failed: Not a socket'). Skip multiplexing there.
# Skip multiplexing there; each command pays a fresh connection but the backend works. See #73927.
_SSH_MULTIPLEX = os.name != "nt"


def _ensure_ssh_available() -> None:
    """Fail fast with a clear error when the SSH client is unavailable."""
    for tool in ("ssh", "scp"):
        if not shutil.which(tool):
            raise RuntimeError(f"{tool.upper()} is not installed or not in PATH. "
                               "Install OpenSSH client: apt install openssh-client")


def _sync_error(reason: str, subject: str, what: str = "the SSH connection") -> EnvironmentConnectionError:
    return EnvironmentConnectionError(
        reason, retry_hint=f"{subject} failed — verify {what} is healthy, then retry.")


class SSHEnvironment(BaseEnvironment):
    """Run commands on a remote machine over SSH.

    Spawn-per-call: every execute() spawns a fresh ``ssh ... bash -c`` process.
    Session snapshot preserves env vars across calls; CWD persists via in-band
    stdout markers. Uses SSH ControlMaster for connection reuse.
    """

    def __init__(self, host: str, user: str, cwd: str = "~",
                 timeout: int = 60, port: int = 22, key_path: str = ""):
        super().__init__(cwd=cwd, timeout=timeout)
        self.host, self.user, self.port, self.key_path = host, user, port, key_path
        self.control_dir = Path(tempfile.gettempdir()) / "hermes-ssh"
        self.control_dir.mkdir(parents=True, exist_ok=True)
        # Short, deterministic socket name: the path must stay under macOS's 104-byte sun_path
        # limit (raw user@host:port + SSH's 16-byte suffix under a deep $TMPDIR exceeds it), and
        # stability across reconnects keeps ControlMaster reuse working.
        _socket_id = hashlib.sha256(f"{user}@{host}:{port}".encode()).hexdigest()[:16]
        self.control_socket = self.control_dir / f"{_socket_id}.sock"
        _ensure_ssh_available()
        self._establish_connection()
        self._remote_home = self._detect_remote_home()
        self._ensure_remote_dirs()
        self._sync_manager = FileSyncManager(
            get_files_fn=lambda: iter_sync_files(f"{self._remote_home}/.hermes"),
            upload_fn=self._scp_upload, delete_fn=self._ssh_delete,
            bulk_upload_fn=self._ssh_bulk_upload, bulk_download_fn=self._ssh_bulk_download)
        self._sync_manager.sync(force=True)
        self.init_session()

    def _target_flags(self, port_flag: str) -> list:
        """Port/key flags shared by ssh (``-p``) and scp (``-P``)."""
        flags = [port_flag, str(self.port)] if self.port != 22 else []
        return flags + (["-i", self.key_path] if self.key_path else [])

    def _build_ssh_command(self, extra_args: list | None = None) -> list:
        cmd = ["ssh"]
        if _SSH_MULTIPLEX:
            cmd.extend(["-o", f"ControlPath={self.control_socket}",
                        "-o", "ControlMaster=auto", "-o", "ControlPersist=300"])
        cmd.extend(["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=10"])
        cmd.extend(self._target_flags("-p"))
        cmd.extend(extra_args or [])
        cmd.append(f"{self.user}@{self.host}")
        return cmd

    def _run_ssh(self, remote_cmd: str, timeout: float) -> subprocess.CompletedProcess:
        """Run one remote shell command over the multiplexed connection, capturing output."""
        return run_capture(self._build_ssh_command() + [remote_cmd], timeout=timeout)

    def _run_ssh_checked(self, remote_cmd: str, timeout: float, reason: str, subject: str) -> None:
        result = self._run_ssh(remote_cmd, timeout=timeout)
        if result.returncode != 0:
            raise _sync_error(f"{reason}: {result.stderr.strip()}", subject)

    def _establish_connection(self):
        try:
            result = self._run_ssh("echo 'SSH connection established'", timeout=15)
        except subprocess.TimeoutExpired:
            raise EnvironmentConnectionError(
                f"SSH connection to {self.user}@{self.host} timed out",
                retry_hint=(f"Check network connectivity to {self.host}:{self.port} "
                            "and that sshd is accepting connections, then retry."))
        if result.returncode != 0:
            error_msg = result.stderr.strip() or result.stdout.strip()
            raise EnvironmentConnectionError(
                f"SSH connection failed: {error_msg}",
                retry_hint=(f"Verify {self.user}@{self.host}:{self.port} is reachable "
                            "(host up, sshd running, key/agent auth working), then "
                            "retry — the connection is re-established automatically."))

    def _detect_remote_home(self) -> str:
        """Detect the remote user's home directory."""
        with contextlib.suppress(Exception):
            result = self._run_ssh("echo $HOME", timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                logger.debug("SSH: remote home = %s", result.stdout.strip())
                return result.stdout.strip()
        return "/root" if self.user == "root" else f"/home/{self.user}"

    def _ensure_remote_dirs(self) -> None:
        """Create base ~/.hermes directory tree on remote in one SSH call."""
        base = f"{self._remote_home}/.hermes"
        self._run_ssh(quoted_mkdir_command([base, f"{base}/skills", f"{base}/credentials", f"{base}/cache"]),
                      timeout=10)

    def _scp_upload(self, host_path: str, remote_path: str) -> None:
        """Upload a single file via scp over ControlMaster."""
        self._run_ssh(f"mkdir -p {shlex.quote(str(Path(remote_path).parent))}", timeout=10)
        scp_cmd = ["scp"] + (["-o", f"ControlPath={self.control_socket}"] if _SSH_MULTIPLEX else [])
        scp_cmd += self._target_flags("-P") + [host_path, f"{self.user}@{self.host}:{remote_path}"]
        result = run_capture(scp_cmd, timeout=30)
        if result.returncode != 0:
            raise _sync_error(f"scp failed: {result.stderr.strip()}", f"File sync to {self.user}@{self.host}")

    def _ssh_bulk_upload(self, files: list[tuple[str, str]]) -> None:
        """Upload many files in one tar-over-SSH stream: local ``tar c`` piped through one SSH
        connection to remote ``tar x``, after a single batched ``mkdir -p``."""
        if not files:
            return
        base = f"{self._remote_home}/.hermes"
        parents = unique_parent_dirs(files)
        if parents:
            self._run_ssh_checked(quoted_mkdir_command(parents), 30, "remote mkdir failed",
                                  f"Remote directory setup on {self.host}")

        # Symlink staging avoids fragile GNU tar --transform rules. On Windows
        # without Developer Mode symlink creation raises OSError winerror 1314;
        # only that case falls back to a plain copy, other OSErrors re-raise.
        with tempfile.TemporaryDirectory(prefix="hermes-ssh-bulk-") as staging:
            for host_path, remote_path in files:
                try:
                    rel_remote = os.path.relpath(remote_path, base)
                except ValueError as exc:
                    raise RuntimeError(f"remote path {remote_path!r} is not under sync base {base!r}") from exc
                if rel_remote == "." or rel_remote.startswith("../"):
                    raise RuntimeError(f"remote path {remote_path!r} escapes sync base {base!r}")
                staged = os.path.join(staging, rel_remote)
                os.makedirs(os.path.dirname(staged), exist_ok=True)
                try:
                    os.symlink(os.path.abspath(host_path), staged)
                except OSError as e:
                    if getattr(e, "winerror", None) != 1314:
                        raise
                    shutil.copy2(host_path, staged)

            # --no-overwrite-dir keeps tar from stamping the staging dir's mode onto
            # existing dirs (e.g. /home/<user>); a umask-002 0775 home breaks sshd StrictModes.
            ssh_cmd = self._build_ssh_command() + [f"tar xf - --no-overwrite-dir -C {shlex.quote(base)}"]
            tar_proc = subprocess.Popen(["tar", "-chf", "-", "-C", staging, "."], stdin=subprocess.DEVNULL,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                ssh_proc = subprocess.Popen(ssh_cmd, stdin=tar_proc.stdout,
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except Exception:
                tar_proc.kill()
                tar_proc.wait()
                raise
            tar_proc.stdout.close()  # let tar_proc receive SIGPIPE if ssh_proc exits early
            try:
                _, ssh_stderr = ssh_proc.communicate(timeout=120)
                # communicate() (not wait()) drains stderr so tar can't deadlock on >PIPE_BUF errors.
                if tar_proc.poll() is None:
                    _, tar_stderr_raw = tar_proc.communicate(timeout=10)
                else:
                    tar_stderr_raw = tar_proc.stderr.read() if tar_proc.stderr else b""
            except subprocess.TimeoutExpired:
                for proc in (tar_proc, ssh_proc):
                    proc.kill()
                for proc in (tar_proc, ssh_proc):
                    proc.wait()  # kill both first, then reap: never wait on one while the other blocks
                raise EnvironmentConnectionError(
                    "SSH bulk upload timed out",
                    retry_hint=f"Bulk file sync to {self.host} timed out — check the connection and retry.")
            if tar_proc.returncode != 0:
                raise RuntimeError(f"tar create failed (rc={tar_proc.returncode}): "
                                   f"{tar_stderr_raw.decode(errors='replace').strip()}")
            if ssh_proc.returncode != 0:
                raise _sync_error(f"tar extract over SSH failed (rc={ssh_proc.returncode}): "
                                  f"{ssh_stderr.decode(errors='replace').strip()}",
                                  f"File sync over SSH to {self.host}", what="the connection")
        logger.debug("SSH: bulk-uploaded %d file(s) via tar pipe", len(files))

    def _ssh_bulk_download(self, dest: Path) -> None:
        """Download remote .hermes/ as a tar archive."""
        # Tar from / with the full path so archive entries keep absolute paths
        # (home/user/.hermes/skills/f.py), matching _pushed_hashes keys.
        rel_base = f"{self._remote_home}/.hermes".lstrip("/")
        ssh_cmd = self._build_ssh_command() + [f"tar cf - -C / {shlex.quote(rel_base)}"]
        with open(dest, "wb") as f:
            result = subprocess.run(ssh_cmd, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.PIPE, timeout=120)
        if result.returncode != 0:
            raise _sync_error(f"SSH bulk download failed: {result.stderr.decode(errors='replace').strip()}",
                              f"File sync from {self.host}")

    def _ssh_delete(self, remote_paths: list[str]) -> None:
        self._run_ssh_checked(quoted_rm_command(remote_paths), 10, "remote rm failed",
                              f"Remote file cleanup on {self.host}")

    def _before_execute(self) -> None:
        self._sync_manager.sync()  # rate-limited internally

    def _run_bash(self, cmd_string: str, *, login: bool = False, timeout: int = 120,
                  stdin_data: str | None = None) -> subprocess.Popen:
        return _popen_bash(self._build_ssh_command() + bash_argv(shlex.quote(cmd_string), login), stdin_data)

    def cleanup(self):
        if self._sync_manager:
            logger.info("SSH: syncing files from sandbox...")
            self._sync_manager.sync_back()
        if self.control_socket.exists():
            with contextlib.suppress(OSError, subprocess.SubprocessError):
                cmd = ["ssh", "-o", f"ControlPath={self.control_socket}", "-O", "exit", f"{self.user}@{self.host}"]
                subprocess.run(cmd, capture_output=True, timeout=5, stdin=subprocess.DEVNULL)
            with contextlib.suppress(OSError):
                self.control_socket.unlink()
