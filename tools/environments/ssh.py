"""SSH remote execution environment with ControlMaster connection persistence."""

import contextlib
import hashlib
import logging
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Iterable

from hermes_constants import get_hermes_home
from tools.environments.base import BaseEnvironment, EnvironmentConnectionError
from tools.environments.base_output import _popen_bash
from tools.environments.file_sync import (
    FileSyncManager, iter_sync_files, quoted_mkdir_command, quoted_rm_command, unique_parent_dirs)
from tools.environments.remote_common import (
    bash_argv, client_env_with, load_hermes_env_vars, prepend_unset, resolve_passthrough_env, run_capture)

logger = logging.getLogger(__name__)

# Windows OpenSSH has no Unix-socket ControlMaster: ControlPath/ControlMaster options
# fail the connection outright ('getsockname failed: Not a socket'). Skip multiplexing there.
# Skip multiplexing there; each command pays a fresh connection but the backend works. See #73927.
_SSH_MULTIPLEX = os.name != "nt"

# Module-level binding: tests patch ``ssh._load_hermes_env_vars`` to fake the .env file.
_load_hermes_env_vars = load_hermes_env_vars

# GNU tar's --keep-newer-files skip notice, e.g. ``tar: Current ‘./skills/x/SKILL.md’ is newer
# or same age``. Non-English locales change the wording; the skip itself is what protects the
# file — this only surfaces it.
_NEWER_SKIPPED_RE = re.compile(rb"Current (.+?) is newer or same age")


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

    # Passthrough values are re-forwarded on every command (see _run_bash), so like docker/local
    # they stay out of the remote snapshot under multiplex.
    _profile_scoped_passthrough = True
    _sudo_nopasswd_probe_supported = True

    def __init__(self, host: str, user: str, cwd: str = "~",
                 timeout: int = 60, port: int = 22, key_path: str = "",
                 probe_only: bool = False):
        super().__init__(cwd=cwd, timeout=timeout)
        self.host, self.user, self.port, self.key_path = host, user, port, key_path
        self.control_dir = Path(tempfile.gettempdir()) / "hermes-ssh"
        self.control_dir.mkdir(parents=True, exist_ok=True)
        # Short, deterministic socket name: the path must stay under macOS's 104-byte sun_path
        # limit (raw user@host:port + SSH's 16-byte suffix under a deep $TMPDIR exceeds it), and
        # stability across reconnects keeps ControlMaster reuse working. A probe gets its own
        # per-instance socket so its cleanup() can never close the agent's shared master.
        socket_key = f"{user}@{host}:{port}"
        if probe_only:
            socket_key = f"{socket_key}:probe:{self._session_id}"
        _socket_id = hashlib.sha256(socket_key.encode()).hexdigest()[:16]
        self.control_socket = self.control_dir / f"{_socket_id}.sock"
        _ensure_ssh_available()
        self._establish_connection()
        if probe_only:
            self._sync_manager = None
            return
        self._remote_home = self._detect_remote_home()
        self._ensure_remote_dirs()
        self._remote_gnu_tar: bool | None = None  # lazy, see _keep_newer_extract_flag
        if self._sync_target_is_local():
            # ssh to the machine/account the agent itself runs as: the "remote" ~/.hermes IS
            # ours. Syncing then only rewrites live files with bytes a tar read earlier, so a
            # push stalled behind a busy multiplexed channel silently reverts newer writes
            # (2026-09-19 skills-revert incident). Nothing to sync — disable the manager.
            logger.info("SSH: %s@%s resolves to this host's own hermes home — file sync disabled",
                        self.user, self.host)
            self._sync_manager = None
        else:
            self._sync_manager = FileSyncManager(
                get_files_fn=lambda: iter_sync_files(f"{self._remote_home}/.hermes"),
                upload_fn=self._scp_upload, delete_fn=self._ssh_delete,
                bulk_upload_fn=self._ssh_bulk_upload, bulk_download_fn=self._ssh_bulk_download)
            self._sync_manager.sync(force=True)
        self.init_session()

    def _control_socket_for(self, send_env: tuple[str, ...]) -> Path:
        """One ControlMaster per SendEnv name-set, beside the plain target socket: a mux master only
        relays the env names it was itself started with and silently drops the rest, so a passthrough
        command must ride a master that knows its names. scp/sync/probes keep the plain socket."""
        plain = Path(self.control_socket)
        if not send_env:
            return plain
        # <target-id[:8]><names-hash[:8]>.sock: same length as the plain socket (macOS's 104-byte
        # sun_path cap) and prefix-globbable so cleanup() finds every sibling without extra state.
        digest = hashlib.sha256(" ".join(send_env).encode()).hexdigest()[:8]
        return plain.with_name(f"{plain.stem[:8]}{digest}.sock")

    def _control_sockets(self) -> list[Path]:
        """The plain socket plus every SendEnv-set sibling (shared 8-char target prefix)."""
        plain = Path(self.control_socket)
        siblings = sorted(plain.parent.glob(f"{plain.stem[:8]}*.sock")) if plain.parent.is_dir() else []
        return [plain, *(s for s in siblings if s != plain)]

    def _target_flags(self, port_flag: str) -> list:
        """Port/key flags shared by ssh (``-p``) and scp (``-P``)."""
        flags = [port_flag, str(self.port)] if self.port != 22 else []
        return flags + (["-i", self.key_path] if self.key_path else [])

    def _build_ssh_command(self, extra_args: list | None = None, send_env: Iterable[str] = ()) -> list:
        send_env = tuple(sorted(send_env))
        cmd = ["ssh"]
        if _SSH_MULTIPLEX:
            cmd.extend(["-o", f"ControlPath={self._control_socket_for(send_env)}",
                        "-o", "ControlMaster=auto", "-o", "ControlPersist=300"])
        cmd.extend(["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=10"])
        # Names only; values ride the ssh client's own environment (never the remote command text).
        cmd.extend(arg for name in send_env for arg in ("-o", f"SendEnv={name}"))
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
                retry_hint=(f"Check that {self.host} is up and reachable on port {self.port} "
                            "and that sshd is running, then retry."))
        if result.returncode != 0:
            error_msg = result.stderr.strip() or result.stdout.strip()
            raise EnvironmentConnectionError(
                f"SSH connection failed: {error_msg}",
                retry_hint=(f"Check that {self.host} is up, sshd is running on port {self.port}, and "
                            f"{self.user} can log in with the configured key, then retry — "
                            "the connection is re-established automatically."))

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

    def _sync_target_is_local(self) -> bool:
        """True when the SSH target reads a tree this host also writes directly (same machine).

        The sync maps the remote ``{remote_home}/.hermes`` onto that SAME local path — the
        ``iter_sync_files(f"{self._remote_home}/.hermes")`` source wired up in ``__init__`` — so
        that path is the tree whose identity decides whether a sync is a self-sync. A profile
        runs with ``~/.hermes/profiles/<name>``: probing only the profile home reads a file the
        mapped root does not hold, so a same-disk target never matched and the self-sync stayed
        on (2026-09-19: profile ``trader`` → ``cameron@rei.taila6a102.ts.net``, its own host —
        263 failed sync_backs, 7.2 GB tars stranded in /tmp). The launching profile's home is
        probed as well, for the case where it *is* that mapped root under another spelling (the
        default profile, a ``HERMES_HOME`` override, a symlink): both answer the same question.

        Each candidate root gets a nonce file written into it and that name read back from the
        remote ``{remote_home}/.hermes``. Only a root that holds the very file the remote reads
        can return the nonce; a different tree never does, however identical its paths look. A
        missing or unreadable root, a failed ``cat`` or any doubt returns False, leaving sync
        enabled exactly as before — a real remote must keep its workspace.
        """
        remote_base = f"{self._remote_home}/.hermes"
        seen: set[Path] = set()
        for root in (Path(remote_base), Path(get_hermes_home())):
            if root in seen:
                continue
            seen.add(root)
            try:
                if self._probe_root_is_shared(root, remote_base):
                    return True
            except Exception:
                logger.debug("SSH: same-tree sync probe failed for %s", root, exc_info=True)
        return False

    def _probe_root_is_shared(self, local_root: Path, remote_base: str) -> bool:
        """Nonce round-trip: *local_root* is the remote's *remote_base* iff the nonce comes back.

        Never raises for a merely unusable root (missing, or a ``stat`` the process may not
        perform — ``/root/.hermes`` from an unprivileged user raises PermissionError).
        """
        if not local_root.is_dir():
            return False
        nonce = uuid.uuid4().hex
        probe = local_root / f".sync-probe-{nonce}"
        probe.write_text(nonce, encoding="utf-8")
        try:
            remote = f"{remote_base}/{probe.name}"
            result = self._run_ssh(f"cat {shlex.quote(remote)} 2>/dev/null", timeout=10)
            same = result.returncode == 0 and (result.stdout or "").strip() == nonce
            if same:
                logger.debug("SSH: same-tree probe matched (%s)", remote)
            return same
        finally:
            with contextlib.suppress(OSError):
                probe.unlink()

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
            staged_names: list[str] = []
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
                staged_names.append("./" + rel_remote.replace(os.sep, "/"))

            # The archive carries FILES ONLY (explicit member list, no directory entries). A
            # file-only stream is what lets the GNU-tar extract use --keep-newer-files: GNU tar
            # treats --keep-newer-files and --no-overwrite-dir as mutually exclusive overwrite
            # policies, so the old ``tar c .``-style archive plus keep-newer made tar exit 2 and
            # fail the whole upload. Existing directories are never touched either way (missing
            # parents are created by the extract itself), so the historical protection
            # (--no-overwrite-dir kept tar from stamping the staging dir's mode onto existing
            # dirs, e.g. /home/<user>, breaking sshd StrictModes) is structural now; the flag
            # stays on the non-GNU path where --keep-newer-files is unavailable.
            # --keep-newer-files makes the extract skip any member whose destination is newer,
            # so bytes tarred before a concurrent write can never revert that write (a stalled
            # multiplexed channel can delay the extract minutes after the read).
            keep_newer = self._keep_newer_extract_flag()
            overwrite_flags = keep_newer or " --no-overwrite-dir"
            ssh_cmd = self._build_ssh_command() + [
                f"tar xf -{overwrite_flags} -C {shlex.quote(base)}"]
            listing = os.path.join(staging, ".push-names")
            with open(listing, "w", encoding="utf-8", newline="") as fh:
                for name in staged_names:
                    fh.write(name + "\0")
            tar_proc = subprocess.Popen(["tar", "-chf", "-", "-C", staging, "--null", "-T", listing],
                                        stdin=subprocess.DEVNULL,
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
            self._warn_extract_skipped_newer(ssh_stderr)
        logger.debug("SSH: bulk-uploaded %d file(s) via tar pipe", len(files))

    def _keep_newer_extract_flag(self) -> str:
        """`` --keep-newer-files`` when the remote tar is GNU tar, else ``""``.

        bsdtar and friends reject the flag outright and a rejected flag fails the whole
        upload, so the capability is probed once per environment and cached. Checked lazily:
        environments that never sync pay nothing.
        """
        if self._remote_gnu_tar is None:
            try:
                result = self._run_ssh("tar --version 2>&1 | head -1", timeout=10)
                self._remote_gnu_tar = "GNU tar" in (result.stdout or "")
                logger.debug("SSH: remote tar is %sGNU", "" if self._remote_gnu_tar else "not ")
            except Exception:
                logger.debug("SSH: remote tar version probe failed", exc_info=True)
                self._remote_gnu_tar = False
        return " --keep-newer-files" if self._remote_gnu_tar else ""

    @staticmethod
    def _warn_extract_skipped_newer(ssh_stderr: bytes) -> None:
        """Surface every push skip caused by --keep-newer-files.

        Each skipped file is one whose destination was rewritten after the tar read it; the
        skip is the fix working, but silence would hide the loss the way it hid the original
        incident, so say it loudly with the paths.
        """
        skipped = [m.group(1).decode("utf-8", "replace").strip().strip("\u2018\u2019'\"")
                   for m in _NEWER_SKIPPED_RE.finditer(ssh_stderr or b"")]
        if not skipped:
            return
        shown = ", ".join(skipped[:5])
        more = f" (+{len(skipped) - 5} more)" if len(skipped) > 5 else ""
        logger.warning(
            "file_sync: push kept %d newer local file(s) instead of reverting them to older "
            "pushed bytes (stale-push protection): %s%s", len(skipped), shown, more)

    def _ssh_bulk_download(self, dest: Path) -> None:
        """Download remote .hermes/ as a tar archive."""
        # Tar from / with the full path so archive entries keep absolute paths
        # (home/user/.hermes/skills/f.py), matching _pushed_hashes keys.
        rel_base = f"{self._remote_home}/.hermes".lstrip("/")
        # Live sockets inside .hermes (gateway.sock and friends) cannot be archived: tar prints
        # "socket ignored" and some builds exit 2, which failed every sync-back and left a
        # multi-GB temp tar behind on each retry. Exclude them up front.
        ssh_cmd = self._build_ssh_command() + [
            f"tar cf - --exclude='*.sock' -C / {shlex.quote(rel_base)}"]
        with open(dest, "wb") as f:
            result = subprocess.run(ssh_cmd, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.PIPE, timeout=120)
        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace").strip()
            # A socket not named *.sock is the only rc=2 we knowingly accept, and only when
            # nothing else was reported — anchored to the diagnostic suffix so a filename merely
            # containing "socket ignored" cannot sneak through. Every other status still fails.
            diagnostic_lines = [line for line in stderr.splitlines() if line.strip()]
            tolerated = result.returncode == 2 and bool(diagnostic_lines) and all(
                line.endswith(": socket ignored") for line in diagnostic_lines)
            if not tolerated:
                raise _sync_error(f"SSH bulk download failed: {stderr}",
                                  f"File sync from {self.host}")

    def _ssh_delete(self, remote_paths: list[str]) -> None:
        self._run_ssh_checked(quoted_rm_command(remote_paths), 10, "remote rm failed",
                              f"Remote file cleanup on {self.host}")

    def _before_execute(self) -> None:
        if self._sync_manager is not None:
            self._sync_manager.sync()  # rate-limited internally

    def _run_bash(self, cmd_string: str, *, login: bool = False, timeout: int = 120,
                  stdin_data: str | None = None) -> subprocess.Popen:
        """Forward the passthrough allowlist (skill ``required_environment_variables`` +
        ``terminal.env_passthrough``) the way docker does: ``SendEnv`` carries the names, the ssh
        client's env carries the values, so secrets never enter the remote ``bash -c`` argv. The
        remote sshd must ``AcceptEnv`` them (#14091). Profile-scoped names missing from the active
        scope are unset remotely so a shared host cannot serve another profile's value."""
        values, unset_names = resolve_passthrough_env(hermes_env_loader=_load_hermes_env_vars)
        cmd = self._build_ssh_command(send_env=values) + bash_argv(shlex.quote(prepend_unset(cmd_string, unset_names)), login)
        client_env = client_env_with(values)
        return _popen_bash(cmd, stdin_data, env=client_env) if client_env is not None else _popen_bash(cmd, stdin_data)

    def cleanup(self):
        if self._sync_manager:
            logger.info("SSH: syncing files from sandbox...")
            self._sync_manager.sync_back()
        for socket in self._control_sockets():
            if not socket.exists():
                continue
            with contextlib.suppress(OSError, subprocess.SubprocessError):
                cmd = ["ssh", "-o", f"ControlPath={socket}", "-O", "exit", f"{self.user}@{self.host}"]
                subprocess.run(cmd, capture_output=True, timeout=5, stdin=subprocess.DEVNULL)
            with contextlib.suppress(OSError):
                socket.unlink()
