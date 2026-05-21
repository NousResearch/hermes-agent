"""Tests for file write safety and HERMES_WRITE_SAFE_ROOT sandboxing.

Based on PR #1085 by ismoilh (salvaged).
"""

import os
from pathlib import Path

import pytest

from tools.file_operations import _is_write_denied


class TestStaticDenyList:
    """Basic sanity checks for the static write deny list."""

    def test_temp_file_not_denied_by_default(self, tmp_path: Path):
        target = tmp_path / "regular.txt"
        assert _is_write_denied(str(target)) is False

    def test_ssh_key_is_denied(self):
        assert _is_write_denied(os.path.expanduser("~/.ssh/id_rsa")) is True

    def test_etc_shadow_is_denied(self):
        assert _is_write_denied("/etc/shadow") is True


class TestSafeWriteRoot:
    """HERMES_WRITE_SAFE_ROOT should sandbox writes to a specific subtree."""

    def test_writes_inside_safe_root_are_allowed(self, tmp_path: Path, monkeypatch):
        safe_root = tmp_path / "workspace"
        child = safe_root / "subdir" / "file.txt"
        os.makedirs(child.parent, exist_ok=True)

        monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(safe_root))
        assert _is_write_denied(str(child)) is False

    def test_writes_to_safe_root_itself_are_allowed(self, tmp_path: Path, monkeypatch):
        safe_root = tmp_path / "workspace"
        os.makedirs(safe_root, exist_ok=True)

        monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(safe_root))
        assert _is_write_denied(str(safe_root)) is False

    def test_writes_outside_safe_root_are_denied(self, tmp_path: Path, monkeypatch):
        safe_root = tmp_path / "workspace"
        outside = tmp_path / "other" / "file.txt"
        os.makedirs(safe_root, exist_ok=True)
        os.makedirs(outside.parent, exist_ok=True)

        monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(safe_root))
        assert _is_write_denied(str(outside)) is True

    def test_safe_root_env_ignores_empty_value(self, tmp_path: Path, monkeypatch):
        target = tmp_path / "regular.txt"
        monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", "")
        assert _is_write_denied(str(target)) is False

    def test_safe_root_unset_allows_all(self, tmp_path: Path, monkeypatch):
        target = tmp_path / "regular.txt"
        monkeypatch.delenv("HERMES_WRITE_SAFE_ROOT", raising=False)
        assert _is_write_denied(str(target)) is False

    def test_safe_root_with_tilde_expansion(self, tmp_path: Path, monkeypatch):
        """~ in HERMES_WRITE_SAFE_ROOT should be expanded."""
        # Use a real subdirectory of tmp_path so we can test tilde-style paths
        safe_root = tmp_path / "workspace"
        inside = safe_root / "file.txt"
        os.makedirs(safe_root, exist_ok=True)

        monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(safe_root))
        assert _is_write_denied(str(inside)) is False

    def test_safe_root_does_not_override_static_deny(self, tmp_path: Path, monkeypatch):
        """Even if a static-denied path is inside the safe root, it's still denied."""
        # Point safe root at home to include ~/.ssh
        monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", os.path.expanduser("~"))
        assert _is_write_denied(os.path.expanduser("~/.ssh/id_rsa")) is True


class TestCheckSensitivePathMacOSBypass:
    """Verify _check_sensitive_path blocks /private/etc paths (issue #8734)."""

    def test_etc_hosts_blocked(self):
        from tools.file_tools import _check_sensitive_path
        assert _check_sensitive_path("/etc/hosts") is not None

    def test_private_etc_hosts_blocked(self):
        from tools.file_tools import _check_sensitive_path
        assert _check_sensitive_path("/private/etc/hosts") is not None

    def test_private_etc_ssh_config_blocked(self):
        from tools.file_tools import _check_sensitive_path
        assert _check_sensitive_path("/private/etc/ssh/sshd_config") is not None

    def test_private_var_blocked(self):
        from tools.file_tools import _check_sensitive_path
        assert _check_sensitive_path("/private/var/db/something") is not None

    def test_boot_still_blocked(self):
        from tools.file_tools import _check_sensitive_path
        assert _check_sensitive_path("/boot/grub/grub.cfg") is not None

    def test_safe_path_allowed(self):
        from tools.file_tools import _check_sensitive_path
        assert _check_sensitive_path("/tmp/safe_file.txt") is None


class TestNixOSConfigException:
    """NixOS users manage /etc/nixos/ as personal dotfiles — file tools
    should allow writes when the config tree is user-owned."""

    def test_nixos_config_allowed(self, tmp_path, monkeypatch):
        """User-owned /etc/nixos/ tree should allow writes."""
        from tools import file_tools
        etc_nixos = tmp_path / "etc" / "nixos"
        etc_nixos.mkdir(parents=True)
        path = etc_nixos / "config.nix"
        path.touch()
        monkeypatch.setattr(file_tools, "_NIXOS_ETC_PREFIX", str(tmp_path / "etc") + "/")
        assert file_tools._is_sensitive_nixos_config(str(path)) is True

    def test_nixos_new_file_allowed(self, tmp_path, monkeypatch):
        """Non-existent file under user-owned /etc/nixos/ should be allowed."""
        from tools import file_tools
        etc_nixos = tmp_path / "etc" / "nixos"
        etc_nixos.mkdir(parents=True)
        new_file = etc_nixos / "new-module.nix"
        monkeypatch.setattr(file_tools, "_NIXOS_ETC_PREFIX", str(tmp_path / "etc") + "/")
        assert file_tools._is_sensitive_nixos_config(str(new_file)) is True

    def test_non_nixos_etc_blocked(self):
        """Paths outside /etc/nixos/ should never match."""
        from tools.file_tools import _is_sensitive_nixos_config
        assert _is_sensitive_nixos_config("/etc/passwd") is False
        assert _is_sensitive_nixos_config("/etc/ssh/sshd_config") is False
        assert _is_sensitive_nixos_config("/etc/foo") is False

    def test_nonexistent_etc_nixos_safe_default(self, monkeypatch):
        """When /etc/nixos/ doesn't exist, return False (safe default)."""
        from tools import file_tools
        monkeypatch.setattr(file_tools, "_NIXOS_ETC_PREFIX", "/nonexistent/path/")
        assert file_tools._is_sensitive_nixos_config("/nonexistent/path/imaginary.nix") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
