"""Tests for symlink bypass prevention in the write deny list.

Ensures that _is_write_denied resolves symlinks before checking the deny list
so that ``/tmp/link -> ~/.ssh/authorized_keys`` is correctly blocked.
"""

import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from tools.file_operations import (
    _is_write_denied,
    ShellFileOperations,
)


_HOME = str(Path.home())


# =========================================================================
# _is_write_denied — symlink resolution
# =========================================================================


class TestSymlinkDenyList:
    """Symlinks pointing to denied paths must be blocked."""

    def test_symlink_to_ssh_authorized_keys(self, tmp_path):
        target = os.path.join(_HOME, ".ssh", "authorized_keys")
        link = tmp_path / "innocent.txt"
        link.symlink_to(target)
        assert _is_write_denied(str(link)) is True

    def test_symlink_to_ssh_id_rsa(self, tmp_path):
        target = os.path.join(_HOME, ".ssh", "id_rsa")
        link = tmp_path / "my_key"
        link.symlink_to(target)
        assert _is_write_denied(str(link)) is True

    def test_symlink_to_ssh_config(self, tmp_path):
        target = os.path.join(_HOME, ".ssh", "config")
        link = tmp_path / "cfg"
        link.symlink_to(target)
        assert _is_write_denied(str(link)) is True

    def test_symlink_to_netrc(self, tmp_path):
        target = os.path.join(_HOME, ".netrc")
        link = tmp_path / "creds"
        link.symlink_to(target)
        assert _is_write_denied(str(link)) is True

    def test_symlink_to_aws_dir(self, tmp_path):
        """Symlink into a denied prefix directory."""
        target = os.path.join(_HOME, ".aws", "credentials")
        link = tmp_path / "aws_creds"
        link.symlink_to(target)
        assert _is_write_denied(str(link)) is True

    def test_symlink_to_kube_dir(self, tmp_path):
        target = os.path.join(_HOME, ".kube", "config")
        link = tmp_path / "k8s"
        link.symlink_to(target)
        assert _is_write_denied(str(link)) is True

    def test_symlink_to_etc_shadow(self, tmp_path):
        target = "/etc/shadow"
        link = tmp_path / "shadow_link"
        link.symlink_to(target)
        assert _is_write_denied(str(link)) is True

    def test_symlink_to_bashrc(self, tmp_path):
        target = os.path.join(_HOME, ".bashrc")
        link = tmp_path / "my_rc"
        link.symlink_to(target)
        assert _is_write_denied(str(link)) is True

    def test_chained_symlinks(self, tmp_path):
        """A chain of symlinks ultimately resolving to a denied path."""
        target = os.path.join(_HOME, ".ssh", "authorized_keys")
        link1 = tmp_path / "link1"
        link1.symlink_to(target)
        link2 = tmp_path / "link2"
        link2.symlink_to(str(link1))
        assert _is_write_denied(str(link2)) is True

    def test_symlink_to_safe_path_allowed(self, tmp_path):
        """Symlink to a non-denied path must still be allowed."""
        target = tmp_path / "real_file.txt"
        target.write_text("safe content")
        link = tmp_path / "link_to_safe"
        link.symlink_to(str(target))
        assert _is_write_denied(str(link)) is False

    def test_relative_symlink_to_denied_path(self, tmp_path):
        """Relative symlink that resolves to a denied path."""
        target = os.path.join(_HOME, ".ssh", "authorized_keys")
        link = tmp_path / "rel_link"
        # Create a relative symlink — os.path.relpath computes relative form
        try:
            rel_target = os.path.relpath(target, str(tmp_path))
        except ValueError:
            # On Windows, relpath fails across drives; skip
            pytest.skip("Cannot compute relative path across drives")
        link.symlink_to(rel_target)
        assert _is_write_denied(str(link)) is True


# =========================================================================
# ShellFileOperations — symlink bypass blocked in write/patch/delete/move
# =========================================================================


class TestShellFileOpsSymlinkDenied:
    """High-level operations must reject symlinks pointing to denied paths."""

    @pytest.fixture()
    def mock_env(self):
        env = MagicMock()
        env.cwd = "/tmp/test"
        env.execute.return_value = {"output": "", "returncode": 0}
        return env

    @pytest.fixture()
    def file_ops(self, mock_env):
        return ShellFileOperations(mock_env)

    def test_write_file_via_symlink_denied(self, file_ops, tmp_path):
        target = os.path.join(_HOME, ".ssh", "authorized_keys")
        link = tmp_path / "sneaky.txt"
        link.symlink_to(target)
        result = file_ops.write_file(str(link), "evil key")
        assert result.error is not None
        assert "denied" in result.error.lower()

    def test_patch_replace_via_symlink_denied(self, file_ops, tmp_path):
        target = os.path.join(_HOME, ".ssh", "authorized_keys")
        link = tmp_path / "sneaky_patch"
        link.symlink_to(target)
        result = file_ops.patch_replace(str(link), "old", "new")
        assert result.error is not None
        assert "denied" in result.error.lower()

    def test_delete_file_via_symlink_denied(self, file_ops, tmp_path):
        target = os.path.join(_HOME, ".ssh", "authorized_keys")
        link = tmp_path / "sneaky_del"
        link.symlink_to(target)
        result = file_ops.delete_file(str(link))
        assert result.error is not None
        assert "denied" in result.error.lower()

    def test_move_file_dst_symlink_denied(self, file_ops, tmp_path):
        target = os.path.join(_HOME, ".ssh", "authorized_keys")
        link = tmp_path / "sneaky_mv"
        link.symlink_to(target)
        result = file_ops.move_file("/tmp/src.txt", str(link))
        assert result.error is not None
        assert "denied" in result.error.lower()

    def test_move_file_src_symlink_denied(self, file_ops, tmp_path):
        target = os.path.join(_HOME, ".ssh", "id_rsa")
        link = tmp_path / "sneaky_src"
        link.symlink_to(target)
        result = file_ops.move_file(str(link), "/tmp/dest.txt")
        assert result.error is not None
        assert "denied" in result.error.lower()
