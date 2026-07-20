"""SSH remote execution environment using PowerShell on Windows hosts."""

import base64
import hashlib
import logging
import os
import shlex
import subprocess
import tempfile
from pathlib import Path

from tools.environments.base import BaseEnvironment, _popen_bash
from tools.environments.file_sync import (
    FileSyncManager,
    iter_sync_files,
)
from tools.environments.ssh import (
    SSHEnvironment,
    _ensure_ssh_available,
)

logger = logging.getLogger(__name__)


def _decode_ssh_output(data: bytes) -> str:
    if not data:
        return ""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        pass
    try:
        return data.decode("gbk")
    except (UnicodeDecodeError, LookupError):
        pass
    return data.decode("latin-1")


class SSHPwshEnvironment(SSHEnvironment):
    """Run commands on a Windows remote over SSH using PowerShell.

    Extends SSHEnvironment — reuses SSH transport (ControlMaster, scp,
    encoding). Overrides shell-related methods to use ``pwsh`` /
    ``powershell`` instead of ``bash``.

    Uses ``-EncodedCommand`` (base64 UTF-16LE) to pass scripts through
    cmd.exe (the typical SSH server default shell on Windows) without
    quoting issues.
    """

    def __init__(self, host: str, user: str, cwd: str = "~",
                 timeout: int = 60, port: int = 22, key_path: str = ""):
        self.host = host
        self.user = user
        self.port = port
        self.key_path = key_path

        self.control_dir = Path(tempfile.gettempdir()) / "hermes-ssh"
        self.control_dir.mkdir(parents=True, exist_ok=True)

        _socket_id = hashlib.sha256(
            f"{user}@{host}:{port}".encode()
        ).hexdigest()[:16]
        self.control_socket = self.control_dir / f"{_socket_id}.sock"

        _ensure_ssh_available()
        self._detect_shell()
        self._remote_home = self._detect_remote_home()
        self._remote_temp = self._detect_remote_temp()

        # Translate Linux-style cwd to Windows path
        if cwd == "~" or cwd == "/root" or cwd.startswith("/home/"):
            cwd = self._remote_home

        BaseEnvironment.__init__(self, cwd=cwd, timeout=timeout)

        self._ensure_remote_dirs()

        self._sync_manager = FileSyncManager(
            get_files_fn=lambda: iter_sync_files(
                f"{self._remote_home}\\.hermes"
            ),
            upload_fn=self._scp_upload,
            delete_fn=self._ssh_delete,
            bulk_upload_fn=self._ssh_bulk_upload,
            bulk_download_fn=self._ssh_bulk_download,
        )
        # Skip forced sync on init - too slow for Windows remotes with many files
        # File sync will happen on-demand during execute() via _before_execute()
        self.init_session()

    def get_temp_dir(self) -> str:
        return getattr(self, "_remote_temp", "/tmp")

    def _encode_pwsh_command(self, pwsh_script: str) -> str:
        """Encode PowerShell script as base64 UTF-16LE for EncodedCommand."""
        return base64.b64encode(pwsh_script.encode("utf-16-le")).decode("ascii")

    def _run_pwsh(self, pwsh_script: str, timeout: int = 10, shell: str | None = None) -> subprocess.CompletedProcess:
        """Run PowerShell script on remote via EncodedCommand."""
        encoded = self._encode_pwsh_command(pwsh_script)
        cmd = self._build_ssh_command()
        shell_cmd = shell or self._pwsh_cmd
        cmd.extend([shell_cmd, "-NoProfile", "-EncodedCommand", encoded])
        return subprocess.run(
            cmd, capture_output=True, timeout=timeout,
            stdin=subprocess.DEVNULL,
        )

    def _detect_shell(self) -> None:
        for shell in ("pwsh", "powershell"):
            try:
                result = self._run_pwsh("Write-Output 'ok'", timeout=15, shell=shell)
                if result.returncode == 0:
                    self._pwsh_cmd = shell
                    logger.debug("SSH pwsh: using %s on %s", shell, self.host)
                    return
            except subprocess.TimeoutExpired:
                continue
        raise RuntimeError(
            f"pwsh/PowerShell not found on remote {self.host}. "
            "Install PowerShell 7 (pwsh) or use ssh backend with bash."
        )

    def _detect_remote_home(self) -> str:
        try:
            result = self._run_pwsh("Write-Output $env:USERPROFILE")
            home = _decode_ssh_output(result.stdout).strip().rstrip("\r\n")
            if home and result.returncode == 0:
                logger.debug("SSH pwsh: remote home = %s", home)
                return home
        except Exception:
            pass
        return f"C:\\Users\\{self.user}"

    def _detect_remote_temp(self) -> str:
        try:
            result = self._run_pwsh("Write-Output $env:TEMP")
            temp = _decode_ssh_output(result.stdout).strip().rstrip("\r\n")
            if temp and result.returncode == 0:
                return temp.replace("\\", "/")
        except Exception:
            pass
        return f"C:/Users/{self.user}/AppData/Local/Temp"

    def _ensure_remote_dirs(self) -> None:
        base = f"{self._remote_home}\\.hermes"
        dirs = [base, f"{base}\\skills", f"{base}\\credentials", f"{base}\\cache"]
        dirs_str = ", ".join(f"'{d}'" for d in dirs)
        script = f"foreach ($d in @({dirs_str})) {{ New-Item -ItemType Directory -Force -Path $d | Out-Null }}"
        try:
            self._run_pwsh(script, timeout=30)
        except Exception as e:
            logger.warning("SSH pwsh: failed to create remote dirs: %s", e)

    def _scp_upload(self, host_path: str, remote_path: str) -> None:
        """Upload a single file via scp over ControlMaster (Windows-aware)."""
        import shlex as _shlex
        parent = str(Path(remote_path).parent)
        # Use PowerShell to create parent directory (not bash mkdir -p)
        try:
            self._run_pwsh(
                f"New-Item -ItemType Directory -Force -Path '{parent}' | Out-Null",
                timeout=30,
            )
        except Exception as e:
            logger.warning("SSH pwsh: failed to create parent dir %s: %s", parent, e)

        scp_cmd = ["scp", "-o", f"ControlPath={self.control_socket}"]
        if self.port != 22:
            scp_cmd.extend(["-P", str(self.port)])
        if self.key_path:
            scp_cmd.extend(["-i", self.key_path])
        scp_cmd.extend([host_path, f"{self.user}@{self.host}:{remote_path}"])
        result = subprocess.run(
            scp_cmd,
            capture_output=True,
            timeout=30,
            stdin=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            raise RuntimeError(f"scp failed: {_decode_ssh_output(result.stderr).strip()}")

    def _ssh_delete(self, remote_paths: list[str]) -> None:
        paths_str = ", ".join(f"'{p}'" for p in remote_paths)
        script = f"Remove-Item -Force -Path @({paths_str}) -ErrorAction SilentlyContinue"
        try:
            result = self._run_pwsh(script)
            if result.returncode != 0:
                raise RuntimeError(
                    f"remote rm failed: {_decode_ssh_output(result.stderr).strip()}"
                )
        except subprocess.TimeoutExpired:
            raise RuntimeError("remote rm timed out")

    def _ssh_bulk_upload(self, files: list[tuple[str, str]]) -> None:
        """Upload many files via a single zip archive.

        Creates a zip locally, scp's it to the remote, and extracts via
        PowerShell ``Expand-Archive``.  This avoids the per-file scp/mkdir
        overhead that makes sequential uploads impractically slow on
        Windows remotes (issue #7467 — tar-based bulk transfer is not
        available because Windows lacks ``tar``).
        """
        if not files:
            return

        import zipfile

        base = f"{self._remote_home}\\.hermes"

        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as zf:
            zip_path = zf.name
        try:
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
                for host_path, remote_path in files:
                    rel = os.path.relpath(remote_path, base).replace("\\", "/")
                    archive.write(host_path, rel)

            remote_zip = f"{self._remote_home}\\hermes_sync_{os.getpid()}.zip"
            scp_cmd = ["scp", "-o", f"ControlPath={self.control_socket}"]
            if self.port != 22:
                scp_cmd.extend(["-P", str(self.port)])
            if self.key_path:
                scp_cmd.extend(["-i", self.key_path])
            scp_cmd.extend([zip_path, f"{self.user}@{self.host}:{remote_zip}"])
            result = subprocess.run(
                scp_cmd, capture_output=True, timeout=120,
                stdin=subprocess.DEVNULL,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"scp zip failed: {_decode_ssh_output(result.stderr).strip()}"
                )

            extract_script = (
                f"Expand-Archive -Path '{remote_zip}' "
                f"-DestinationPath '{base}' -Force; "
                f"Remove-Item '{remote_zip}' -Force -ErrorAction SilentlyContinue"
            )
            self._run_pwsh(extract_script, timeout=60)
        finally:
            try:
                os.unlink(zip_path)
            except OSError:
                pass

    def _ssh_bulk_download(self, dest: Path) -> None:
        """Download remote ``.hermes/`` as a tar archive via zip.

        Uses PowerShell ``Compress-Archive`` on the remote to create a zip,
        scp's it back, then converts to tar locally (the caller expects a
        tar file — see ``FileSyncManager._sync_back_impl``).
        """
        import io
        import tarfile
        import zipfile

        remote_zip = f"{self._remote_home}\\hermes_sync_{os.getpid()}.zip"
        hermes_dir = f"{self._remote_home}\\.hermes"

        compress_script = (
            f"Compress-Archive -Path '{hermes_dir}\\*' "
            f"-DestinationPath '{remote_zip}' -Force"
        )
        self._run_pwsh(compress_script, timeout=120)

        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as zf:
            zip_path = zf.name
        try:
            scp_cmd = ["scp", "-o", f"ControlPath={self.control_socket}"]
            if self.port != 22:
                scp_cmd.extend(["-P", str(self.port)])
            if self.key_path:
                scp_cmd.extend(["-i", self.key_path])
            scp_cmd.extend([f"{self.user}@{self.host}:{remote_zip}", zip_path])
            result = subprocess.run(
                scp_cmd, capture_output=True, timeout=120,
                stdin=subprocess.DEVNULL,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"scp zip download failed: "
                    f"{_decode_ssh_output(result.stderr).strip()}"
                )

            with zipfile.ZipFile(zip_path) as zf_in, \
                    tarfile.open(dest, "w") as tf_out:
                for info in zf_in.infolist():
                    if info.is_dir():
                        continue
                    data = zf_in.read(info.filename)
                    tar_info = tarfile.TarInfo(name=info.filename)
                    tar_info.size = len(data)
                    tf_out.addfile(tar_info, io.BytesIO(data))
        finally:
            try:
                os.unlink(zip_path)
            except OSError:
                pass
            self._run_pwsh(
                f"Remove-Item '{remote_zip}' -Force -ErrorAction SilentlyContinue",
                timeout=10,
            )

    def _run_bash(self, cmd_string: str, *, login: bool = False,
                  timeout: int = 120,
                  stdin_data: str | None = None) -> subprocess.Popen:
        encoded = self._encode_pwsh_command(cmd_string)
        cmd = self._build_ssh_command()
        cmd.extend([self._pwsh_cmd, "-NoProfile", "-EncodedCommand", encoded])
        return _popen_bash(cmd, stdin_data)

    def _before_execute(self) -> None:
        """Sync files to remote via FileSyncManager (rate-limited internally)."""
        self._sync_manager.sync()

    def _wrap_command(self, command: str, cwd: str) -> str:
        escaped = command.replace("'", "''")
        _quoted_snap = shlex.quote(self._snapshot_path)
        _snap_tmp = shlex.quote(self._snapshot_path + ".tmp.") + "$PID"

        parts = []

        if self._snapshot_ready:
            parts.append(f". {_quoted_snap} 2>$null")

        parts.append(f"Set-Location -LiteralPath {shlex.quote(cwd)}")
        parts.append("if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) { exit 126 }")

        parts.append(f"Invoke-Expression '{escaped}'")
        parts.append("$script:__hermes_ec = $LASTEXITCODE")

        if self._snapshot_ready:
            # Atomic snapshot replacement (issue #38249): write to a
            # per-writer temp file, then Move-Item -Force so concurrent
            # source() calls never read a half-written snapshot.
            parts.append(
                "Get-ChildItem Env: | ForEach-Object { "
                "$val = $_.Value -replace \"'\", \"''\"; "
                "\"`$env:$($_.Name) = '$val'\" "
                f"}} | Set-Content -Encoding UTF8 {_snap_tmp}"
            )
            parts.append(
                f"try {{ Move-Item -Force {_snap_tmp} {_quoted_snap} }} "
                f"catch {{ Remove-Item -Force {_snap_tmp} -ErrorAction SilentlyContinue }}"
            )

        parts.append(
            f'Write-Output "`n{self._cwd_marker}$((Get-Location).Path){self._cwd_marker}"'
        )
        parts.append("exit $script:__hermes_ec")

        return "\n".join(parts)

    def init_session(self):
        _quoted_cwd = shlex.quote(self.cwd)
        _quoted_snap = shlex.quote(self._snapshot_path)
        _snap_tmp = shlex.quote(self._snapshot_path + ".tmp.") + "$PID"

        bootstrap_parts = [
            f"Get-ChildItem Env: | ForEach-Object {{ $val = $_.Value -replace \"'\", \"''\"; \"`$env:$($_.Name) = '$val'\" }} | Set-Content -Encoding UTF8 {_snap_tmp}",
            f"Move-Item -Force {_snap_tmp} {_quoted_snap}",
            f"Set-Location -LiteralPath {_quoted_cwd}",
            f'Write-Output "`n{self._cwd_marker}$((Get-Location).Path){self._cwd_marker}"',
        ]
        bootstrap = "\n".join(bootstrap_parts)

        try:
            proc = self._run_bash(bootstrap, login=True,
                                  timeout=self._snapshot_timeout)
            result = self._wait_for_process(proc,
                                            timeout=self._snapshot_timeout)
            self._snapshot_ready = True
            self._update_cwd(result)
            logger.info(
                "SSH pwsh: session snapshot created (session=%s, cwd=%s)",
                self._session_id, self.cwd,
            )
        except Exception as exc:
            logger.warning(
                "SSH pwsh: init_session failed (session=%s): %s — "
                "falling back to direct pwsh per command",
                self._session_id, exc,
            )
            self._snapshot_ready = False
