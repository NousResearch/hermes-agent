"""Agent Sandbox execution environment.

Uses the k8s-agent-sandbox Python SDK to run commands in cloud sandboxes.
Supports persistent sandboxes: when enabled, sandboxes are stopped on cleanup
and resumed on next creation, preserving the filesystem across sessions.
"""

import io
import tarfile
import uuid
import logging
import os
import shlex
import threading
from typing import TypedDict
from pathlib import Path

from tools.environments.base import (
    BaseEnvironment,
    _ThreadedProcessHandle,
)
from tools.environments.file_sync import (
    FileSyncManager,
    iter_sync_files,
    quoted_rm_command,
)

logger = logging.getLogger(__name__)


class ConnectionConfigParams(TypedDict):
    name: str
    port_forward_ready_timeout: int
    server_port: int
    router_namespace: str
    api_url: str
    gateway_name: str
    gateway_namespace: str
    gateway_ready_timeout: int
    use_pod_ip: bool


class AgentSandboxBackend(BaseEnvironment):
    """K8s-agent-sandbox cloud sandbox execution backend.

    Spawn-per-call via _ThreadedProcessHandle wrapping blocking SDK calls.
    Shell timeout wrapper preserved (SDK timeout unreliable).
    """
    def __init__(
        self,
        cwd: str,
        warmpool: str,
        connection_config_args: ConnectionConfigParams,
        timeout: int = 60,
        task_id: str = "default",
        namespace: str = "default",
        persistent_filesystem: bool = False,
        _stdin_mode: str = "heredoc",
    ):
        super().__init__(cwd=cwd, timeout=timeout)

        try:
            from tools.lazy_deps import ensure as _lazy_ensure
        except ImportError:
            _lazy_ensure = None
        except Exception as e:
            raise ImportError(str(e))

        if _lazy_ensure:
            _lazy_ensure("terminal.agent_sandbox", prompt=False)

        from k8s_agent_sandbox import (
            SandboxClient,
        )
        from k8s_agent_sandbox.models import (
            SandboxDirectConnectionConfig,
            SandboxGatewayConnectionConfig,
            SandboxLocalTunnelConnectionConfig,
            SandboxInClusterConnectionConfig,
        )

        config_name = connection_config_args.get("name", "SandboxLocalTunnelConnectionConfig")
        if config_name == "SandboxLocalTunnelConnectionConfig":
            connection_config = SandboxLocalTunnelConnectionConfig(
                port_forward_ready_timeout=connection_config_args.get("port_forward_ready_timeout", 30),
                server_port=connection_config_args.get("server_port", 8888),
                router_namespace=connection_config_args.get("router_namespace", "agent-sandbox-system"),
            )
        elif config_name == "SandboxDirectConnectionConfig":
            connection_config = SandboxDirectConnectionConfig(
                api_url=connection_config_args.get("api_url", ""),
                server_port=connection_config_args.get("server_port", 8888),
            )
        elif config_name == "SandboxGatewayConnectionConfig":
            connection_config = SandboxGatewayConnectionConfig(
                gateway_name=connection_config_args.get("gateway_name", ""),
                gateway_namespace=connection_config_args.get("gateway_namespace", "default"),
                gateway_ready_timeout=connection_config_args.get("gateway_ready_timeout", 100),
                server_port=connection_config_args.get("server_port", 8888),
            )
        elif config_name == "SandboxInClusterConnectionConfig":
            connection_config = SandboxInClusterConnectionConfig(
                server_port=connection_config_args.get("server_port", 8888),
                use_pod_ip=connection_config_args.get("use_pod_ip", False),
            )
        else:
            raise ValueError(f"Not allowed connection config name: \"{config_name}\"")

        self._timeout = timeout
        self._task_id = task_id
        self._lock = threading.Lock()
        self._persistent = persistent_filesystem
        self._sandbox = None
        self.client = SandboxClient(connection_config=connection_config)

        if self._persistent:
            try:
                claim_name = self.client.list_all_sandboxes(label_selector=f"hermes_task_id={task_id}")[0]
                self._sandbox = self.client.get_sandbox(claim_name)
            except IndexError:
                logger.info("agent-sandbox: The requested sandbox with label_selector=\"hermes_task_id=%s\" wasn't found.", task_id)
                self._sandbox = None
            except Exception as e:
                logger.warning("agent-sandbox: Error: %s\nhermes_task_id=%s", e, task_id)
                self._sandbox = None
        if self._sandbox is None:
            self._sandbox = self.client.create_sandbox(
                warmpool=warmpool,
                namespace=namespace,
                labels={"hermes_task_id": task_id},
            )
        self._hermes_path = f"{self.cwd}/.hermes"

        self._sync_manager = FileSyncManager(
            get_files_fn=lambda: iter_sync_files(self._hermes_path),
            upload_fn=self._agent_sandbox_upload,
            delete_fn=self._agent_sandbox_delete,
            bulk_upload_fn=self._agent_sandbox_bulk_upload,
            bulk_download_fn=self._agent_sandbox_bulk_download,
        )
        self._sync_manager.sync(force=True)
        self.init_session()

    def _agent_sandbox_upload(self, host_path: str, remote_path: str):
        """Upload a single file via k8s-agent-sandbox Python SDK."""
        with open(host_path, "rb") as fi:
            content = fi.read()
        self._sandbox.files.write(path=remote_path, content=content)

    def _agent_sandbox_bulk_upload(self, files: list[tuple[str, str]]) -> None:
        """Upload many files in 2 network hops: 1 tar upload + 1 tar extract."""
        if not files:
            return

        # 1. Create an in-memory tar.gz archive
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
            for host_path, remote_path in files:
                arcname = remote_path.lstrip("/")
                tar.add(name=host_path, arcname=arcname)

        # 2. Upload the single tarball to a temporary path via SDK's HTTP write
        tmp_tar_path = f"tmp/bundle_{uuid.uuid4().hex}.tar.gz"
        self._sandbox.files.write(path=tmp_tar_path, content=tar_buffer.getvalue())

        extract_cmd = f"tar -xzf {tmp_tar_path} /"
        self._sandbox.commands.run(command=extract_cmd, timeout=self._timeout)
        self._agent_sandbox_delete([tmp_tar_path])

    def _agent_sandbox_bulk_download(self, dest: Path):
        """Download remote .hermes/ dir as a tar archive."""
        rel_base = ".hermes"
        rel_remote_tar = f"{rel_base}_sync.{os.getpid()}.tar"
        self._sandbox.commands.run(
            command=f"tar cf {shlex.quote(rel_remote_tar)} {self._hermes_path}",
            timeout=self._timeout
        )
        content = self._sandbox.files.read(rel_remote_tar)
        with open(dest, "wb") as fo:
            fo.write(content)
        try:
            self._sandbox.commands.run(
                command=f"bash -c \"rm -f {shlex.quote(rel_remote_tar)}\"",
                timeout=self._timeout
            )
        except Exception:
            pass

    def _agent_sandbox_delete(self, remote_paths: list[str]):
        self._sandbox.commands.run(
            command=quoted_rm_command(remote_paths),
            timeout=self._timeout
        )

    def _before_execute(self):
        """Syncs files via FileSyncManager."""
        self._sync_manager.sync()

    def _run_bash(
        self, cmd_string: str,
        *,
        login: bool = False,
        timeout: int = 120,
        stdin_data: str | None = None
    ):
        sandbox = self._sandbox

        if login:
            shell_cmd = f"bash -l -c {shlex.quote(cmd_string)}"
        else:
            shell_cmd = f"bash -c {shlex.quote(cmd_string)}"

        def exec_fn() -> tuple[str, int]:
            response = sandbox.commands.run(command=shell_cmd, timeout=timeout)
            return (response.stdout or "") + (response.stderr or ""), response.exit_code
        return _ThreadedProcessHandle(exec_fn=exec_fn)

    def cleanup(self):
        with self._lock:
            if self._sandbox is None:
                return

            if self._sync_manager:
                logger.info("agent-sandbox: syncing files from sandbox...")
                try:
                    self._sync_manager.sync_back()
                except Exception as e:
                    logger.warning("agent-sandbox: sync_back failed: %s", e)

            try:
                if not self._persistent:
                    claim_name = self._sandbox.claim_name
                    self._sandbox.terminate()
                    logger.info("agent-sandbox: deleted sandbox with claim name '%s'", claim_name)
                else:
                    self._sandbox.close_connection()
                self._sandbox = None
                logger.info(f"agent-sandbox: clean up succeeded")
            except Exception as e:
                logger.warning("agent-sandbox: cleanup failed: %s", e)
