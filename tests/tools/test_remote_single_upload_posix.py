"""Call-site guards for single-file remote upload parent directories.

Reverting Daytona, Modal, or SSH to ``str(Path(remote_path).parent)`` must
fail even if ``remote_parent_dir`` itself still passes. ``Path.parent`` is
patched to a backslash path so a host-OS-dependent parent cannot sneak through.
"""

import shlex
from unittest.mock import MagicMock, patch

REMOTE_PATH = "/root/.hermes/credentials/token.json"
EXPECTED_PARENT = "/root/.hermes/credentials"


class _WindowsParent:
    def __str__(self) -> str:
        return "\\root\\.hermes\\credentials"


class _HostPath:
    """Path stand-in: file reads work, but .parent is a Windows path."""

    def __init__(self, value):
        self._value = str(value)

    def read_bytes(self):
        return b"payload"

    @property
    def parent(self):
        return _WindowsParent()


def test_daytona_upload_mkdir_uses_posix_parent():
    from tools.environments import daytona

    env = daytona.DaytonaEnvironment.__new__(daytona.DaytonaEnvironment)
    env._sandbox = MagicMock()
    with patch.object(daytona, "Path", _HostPath):
        env._daytona_upload("/tmp/local.json", REMOTE_PATH)

    mkdir_arg = env._sandbox.process.exec.call_args[0][0]
    assert "\\" not in mkdir_arg
    assert mkdir_arg == f"mkdir -p {shlex.quote(EXPECTED_PARENT)}"


def test_modal_upload_cmd_uses_posix_parent():
    from tools.environments import modal

    env = modal.ModalEnvironment.__new__(modal.ModalEnvironment)
    env._exec = MagicMock()
    with patch.object(modal, "Path", _HostPath):
        env._modal_upload("/tmp/local.json", REMOTE_PATH)

    cmd = env._exec.call_args[0][0]
    assert "\\" not in cmd
    assert cmd.startswith(f"mkdir -p {shlex.quote(EXPECTED_PARENT)} && ")
    assert shlex.quote(REMOTE_PATH) in cmd
    assert env._exec.call_args.kwargs["stdin"] == "cGF5bG9hZA=="


def test_scp_upload_mkdir_uses_posix_parent():
    from tools.environments import ssh

    env = ssh.SSHEnvironment.__new__(ssh.SSHEnvironment)
    env.user = "root"
    env.host = "example.com"
    env.port = 22
    env.key_path = ""
    env.control_socket = "/tmp/sock"
    env._run_ssh = MagicMock()
    ok = MagicMock(returncode=0, stderr="")
    with (
        patch.object(ssh, "Path", _HostPath),
        patch.object(ssh, "run_capture", return_value=ok),
    ):
        env._scp_upload("/tmp/local.json", REMOTE_PATH)

    mkdir_cmd = env._run_ssh.call_args[0][0]
    assert "\\" not in mkdir_cmd
    assert mkdir_cmd == f"mkdir -p {shlex.quote(EXPECTED_PARENT)}"
