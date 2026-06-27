"""SSH remote execution environment for native Windows OpenSSH hosts."""

import base64
import hashlib
import json
import logging
import os
import posixpath
import subprocess
import tempfile
import zipfile
from pathlib import Path

from tools.environments.base import BaseEnvironment, _popen_bash
from tools.environments.file_sync import FileSyncManager, iter_sync_files
from tools.environments.ssh import _ensure_ssh_available, encode_powershell_command

logger = logging.getLogger(__name__)


def _ps_single_quote(value: str) -> str:
    """Return a PowerShell single-quoted string literal."""
    return "'" + value.replace("'", "''") + "'"


def _normalize_windows_path(path: str) -> str:
    """Use forward slashes so PowerShell and Python path helpers agree."""
    return path.strip().replace("\\", "/")


class WindowsSSHEnvironment(BaseEnvironment):
    """Run commands on a native Windows machine over SSH using PowerShell.

    This backend is selected only when the SSH target probes as Windows. It
    avoids the POSIX assumptions in SSHEnvironment: no remote bash, mkdir -p,
    rm, tar, or /tmp are required.
    """

    _powershell_preamble = """
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$ProgressPreference = 'SilentlyContinue'
"""

    def get_temp_dir(self) -> str:
        return getattr(self, "_remote_temp", "C:/Windows/Temp")

    def __init__(
        self,
        host: str,
        user: str,
        cwd: str = "~",
        timeout: int = 60,
        port: int = 22,
        key_path: str = "",
    ):
        super().__init__(cwd=cwd, timeout=timeout)
        self.host = host
        self.user = user
        self.port = port
        self.key_path = key_path

        self.control_dir = Path(tempfile.gettempdir()) / "hermes-ssh"
        self.control_dir.mkdir(parents=True, exist_ok=True)
        socket_id = hashlib.sha256(f"{user}@{host}:{port}".encode()).hexdigest()[:16]
        self.control_socket = self.control_dir / f"{socket_id}.sock"

        _ensure_ssh_available()
        self._establish_connection()
        self._remote_home = self._detect_remote_home()
        self._remote_temp = self._detect_remote_temp()
        if not cwd or cwd == "~":
            self.cwd = self._remote_home
        elif cwd.startswith("~/"):
            self.cwd = f"{self._remote_home}/{cwd[2:]}"
        else:
            self.cwd = self._normalize_cwd(cwd)

        self._ensure_remote_dirs()
        self._sync_manager = FileSyncManager(
            get_files_fn=lambda: iter_sync_files(f"{self._remote_home}/.hermes"),
            upload_fn=self._powershell_upload,
            delete_fn=self._powershell_delete,
            bulk_upload_fn=self._powershell_bulk_upload,
            bulk_download_fn=None,
        )
        self._sync_manager.sync(force=True)
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

    def _build_scp_command(self, host_path: str, remote_path: str) -> list:
        cmd = ["scp", "-o", f"ControlPath={self.control_socket}"]
        if self.port != 22:
            cmd.extend(["-P", str(self.port)])
        if self.key_path:
            cmd.extend(["-i", self.key_path])
        cmd.extend([host_path, f"{self.user}@{self.host}:{_normalize_windows_path(remote_path)}"])
        return cmd

    def _build_powershell_command(self, script: str) -> list[str]:
        cmd = self._build_ssh_command()
        cmd.extend([
            "powershell",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-OutputFormat",
            "Text",
            "-EncodedCommand",
            encode_powershell_command(self._powershell_preamble + "\n" + script),
        ])
        return cmd

    def _run_powershell_script(
        self,
        script: str,
        *,
        timeout: int = 30,
        input_data: str | None = None,
    ) -> subprocess.CompletedProcess:
        kwargs = {
            "capture_output": True,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "timeout": timeout,
        }
        if input_data is None:
            kwargs["stdin"] = subprocess.DEVNULL
        else:
            kwargs["input"] = input_data
        return subprocess.run(self._build_powershell_command(script), **kwargs)

    def _establish_connection(self) -> None:
        result = self._run_powershell_script(
            "Write-Output 'SSH connection established'",
            timeout=15,
        )
        if result.returncode != 0:
            error = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"SSH connection failed: {error}")

    def _detect_remote_home(self) -> str:
        script = """
$homePath = $HOME
if (-not $homePath) { $homePath = $env:USERPROFILE }
Write-Output $homePath
"""
        try:
            result = self._run_powershell_script(script, timeout=10)
            home = _normalize_windows_path(result.stdout.strip().splitlines()[-1])
            if home and result.returncode == 0:
                logger.debug("SSH Windows: remote home = %s", home)
                return home
        except Exception:
            pass
        return f"C:/Users/{self.user}"

    def _detect_remote_temp(self) -> str:
        script = """
$tempPath = $env:TEMP
if (-not $tempPath) { $tempPath = [System.IO.Path]::GetTempPath() }
Write-Output $tempPath
"""
        try:
            result = self._run_powershell_script(script, timeout=10)
            temp = _normalize_windows_path(result.stdout.strip().splitlines()[-1])
            if temp and result.returncode == 0:
                return temp.rstrip("/")
        except Exception:
            pass
        return f"{self._remote_home}/AppData/Local/Temp"

    def _normalize_cwd(self, cwd: str) -> str:
        cwd = _normalize_windows_path(cwd or self._remote_home)
        if cwd in {"~", "~/"}:
            return self._remote_home
        if cwd.startswith("~/"):
            return f"{self._remote_home}/{cwd[2:]}"
        if cwd.startswith("/Users/") or cwd.startswith("/home/"):
            return self._remote_home
        return cwd

    def _ensure_remote_dirs(self) -> None:
        base = f"{self._remote_home}/.hermes"
        dirs = [base, f"{base}/skills", f"{base}/credentials", f"{base}/cache"]
        self._powershell_mkdir(dirs)

    def _powershell_mkdir(self, dirs: list[str]) -> None:
        if not dirs:
            return
        script = """
$paths = [Console]::In.ReadToEnd() | ConvertFrom-Json
foreach ($path in $paths) {
  if ($path) {
    New-Item -ItemType Directory -Force -Path ([string]$path) | Out-Null
  }
}
"""
        result = self._run_powershell_script(
            script,
            timeout=30,
            input_data=json.dumps([_normalize_windows_path(d) for d in dirs]),
        )
        if result.returncode != 0:
            raise RuntimeError(f"remote mkdir failed: {result.stderr.strip() or result.stdout.strip()}")

    def _powershell_upload(self, host_path: str, remote_path: str) -> None:
        with open(host_path, "rb") as f:
            content_b64 = base64.b64encode(f.read()).decode("ascii")
        payload = {
            "path": _normalize_windows_path(remote_path),
            "content": content_b64,
        }
        script = """
$payload = [Console]::In.ReadToEnd() | ConvertFrom-Json
$path = [string]$payload.path
$parent = Split-Path -Parent -Path $path
if ($parent) {
  New-Item -ItemType Directory -Force -Path $parent | Out-Null
}
[System.IO.File]::WriteAllBytes($path, [Convert]::FromBase64String([string]$payload.content))
"""
        result = self._run_powershell_script(
            script,
            timeout=60,
            input_data=json.dumps(payload),
        )
        if result.returncode != 0:
            raise RuntimeError(f"PowerShell upload failed: {result.stderr.strip() or result.stdout.strip()}")

    def _powershell_bulk_upload(self, files: list[tuple[str, str]]) -> None:
        if not files:
            return

        base = f"{self._remote_home}/.hermes"
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            zip_path = tmp.name

        try:
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for host_path, remote_path in files:
                    remote_norm = _normalize_windows_path(remote_path)
                    try:
                        rel_remote = posixpath.relpath(remote_norm, base)
                    except ValueError as exc:
                        raise RuntimeError(
                            f"remote path {remote_path!r} is not under sync base {base!r}"
                        ) from exc
                    if rel_remote == "." or rel_remote.startswith("../"):
                        raise RuntimeError(
                            f"remote path {remote_path!r} escapes sync base {base!r}"
                        )
                    zf.write(host_path, rel_remote)

            remote_zip = f"{self._remote_temp}/hermes-sync-{os.getpid()}.zip"
            zip_size = os.path.getsize(zip_path)
            scp_timeout = max(120, min(900, 120 + (zip_size // (1024 * 1024)) * 60))
            scp_result = subprocess.run(
                self._build_scp_command(zip_path, remote_zip),
                capture_output=True,
                text=True,
                timeout=scp_timeout,
                stdin=subprocess.DEVNULL,
            )
            if scp_result.returncode != 0:
                raise RuntimeError(f"scp zip upload failed: {scp_result.stderr.strip()}")

            payload = {
                "zip_path": _normalize_windows_path(remote_zip),
                "destination": base,
            }
            script = """
$payload = [Console]::In.ReadToEnd() | ConvertFrom-Json
$zipPath = [string]$payload.zip_path
$destination = [string]$payload.destination
New-Item -ItemType Directory -Force -Path $destination | Out-Null
try {
  Add-Type -AssemblyName System.IO.Compression.FileSystem
  $zip = [System.IO.Compression.ZipFile]::OpenRead($zipPath)
  try {
    foreach ($entry in $zip.Entries) {
      if ([string]::IsNullOrEmpty($entry.Name)) { continue }
      $target = Join-Path $destination $entry.FullName
      $targetParent = Split-Path -Parent -Path $target
      if ($targetParent) {
        New-Item -ItemType Directory -Force -Path $targetParent | Out-Null
      }
      if (Test-Path -LiteralPath $target) {
        Remove-Item -LiteralPath $target -Force
      }
      [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $target)
    }
  } finally {
    $zip.Dispose()
  }
} finally {
  Remove-Item -LiteralPath $zipPath -Force -ErrorAction SilentlyContinue
}
"""
            timeout = max(120, min(900, 120 + (zip_size // (1024 * 1024)) * 60))
            result = self._run_powershell_script(
                script,
                timeout=timeout,
                input_data=json.dumps(payload),
            )
            if result.returncode != 0:
                raise RuntimeError(f"PowerShell bulk upload failed: {result.stderr.strip() or result.stdout.strip()}")
        finally:
            try:
                os.unlink(zip_path)
            except OSError:
                pass

    def _powershell_delete(self, remote_paths: list[str]) -> None:
        if not remote_paths:
            return
        script = """
$paths = [Console]::In.ReadToEnd() | ConvertFrom-Json
foreach ($path in $paths) {
  if ($path) {
    Remove-Item -LiteralPath ([string]$path) -Force -ErrorAction SilentlyContinue
  }
}
"""
        result = self._run_powershell_script(
            script,
            timeout=30,
            input_data=json.dumps([_normalize_windows_path(p) for p in remote_paths]),
        )
        if result.returncode != 0:
            raise RuntimeError(f"remote rm failed: {result.stderr.strip() or result.stdout.strip()}")

    def init_session(self) -> None:
        self._snapshot_ready = True
        logger.info("Windows SSH session initialized (session=%s, cwd=%s)", self._session_id, self.cwd)

    def _before_execute(self) -> None:
        self._sync_manager.sync()

    def _wrap_powershell_command(self, command: str, cwd: str) -> str:
        quoted_cwd = _ps_single_quote(self._normalize_cwd(cwd))
        return f"""
$__hermes_cwd = {quoted_cwd}
try {{
  Set-Location -LiteralPath $__hermes_cwd
}} catch {{
  Write-Error ("cd failed: " + $__hermes_cwd + " - " + $_.Exception.Message)
  exit 126
}}
$global:LASTEXITCODE = $null
. {{
{command}
}}
$__hermes_success = $?
if ($global:LASTEXITCODE -is [int]) {{
  $__hermes_ec = $global:LASTEXITCODE
}} elseif ($__hermes_success) {{
  $__hermes_ec = 0
}} else {{
  $__hermes_ec = 1
}}
$__hermes_pwd = (Get-Location).ProviderPath
Write-Output ""
Write-Output ("{self._cwd_marker}" + $__hermes_pwd + "{self._cwd_marker}")
exit $__hermes_ec
"""

    def _run_powershell(
        self,
        script: str,
        *,
        stdin_data: str | None = None,
    ) -> subprocess.Popen:
        return _popen_bash(self._build_powershell_command(script), stdin_data=stdin_data)

    def execute(
        self,
        command: str,
        cwd: str = "",
        *,
        timeout: int | None = None,
        stdin_data: str | None = None,
        rewrite_compound_background: bool = True,
    ) -> dict:
        self._before_execute()
        effective_timeout = timeout or self.timeout
        effective_cwd = self._normalize_cwd(cwd or self.cwd)
        wrapped = self._wrap_powershell_command(command, effective_cwd)
        proc = self._run_powershell(wrapped, stdin_data=stdin_data)
        result = self._wait_for_process(proc, timeout=effective_timeout)
        self._update_cwd(result)
        return result

    def cleanup(self) -> None:
        sync_manager = getattr(self, "_sync_manager", None)
        if sync_manager:
            logger.info("SSH Windows: syncing files from sandbox...")
            sync_manager.sync_back()

        control_socket = getattr(self, "control_socket", None)
        if control_socket and control_socket.exists():
            try:
                cmd = [
                    "ssh",
                    "-o",
                    f"ControlPath={control_socket}",
                    "-O",
                    "exit",
                    f"{self.user}@{self.host}",
                ]
                subprocess.run(
                    cmd,
                    capture_output=True,
                    timeout=5,
                    stdin=subprocess.DEVNULL,
                )
            except (OSError, subprocess.SubprocessError):
                pass
            try:
                control_socket.unlink()
            except OSError:
                pass
