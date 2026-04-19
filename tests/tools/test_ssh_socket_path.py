"""Test that SSH ControlMaster socket paths stay under macOS 104-char limit."""

import hashlib
import tempfile
from pathlib import Path


MACOS_UNIX_SOCKET_LIMIT = 104


def _build_control_socket(user: str, host: str, port: int) -> Path:
    """Reproduce the socket path logic from SSHEnvironment.__init__."""
    control_dir = Path(tempfile.gettempdir()) / "hermes-ssh"
    socket_id = hashlib.sha256(f"{user}@{host}:{port}".encode()).hexdigest()[:12]
    return control_dir / f"{socket_id}.sock"


def test_socket_path_short_for_ipv6():
    long_ipv6 = "2001:0db8:85a3:0000:0000:8a2e:0370:7334"
    sock = _build_control_socket("deployuser", long_ipv6, 22)
    assert len(str(sock)) < MACOS_UNIX_SOCKET_LIMIT, f"path too long: {sock}"


def test_socket_path_short_for_long_username():
    sock = _build_control_socket("a" * 64, "example.com", 22)
    assert len(str(sock)) < MACOS_UNIX_SOCKET_LIMIT, f"path too long: {sock}"


def test_different_hosts_get_different_sockets():
    s1 = _build_control_socket("user", "host1", 22)
    s2 = _build_control_socket("user", "host2", 22)
    assert s1 != s2
