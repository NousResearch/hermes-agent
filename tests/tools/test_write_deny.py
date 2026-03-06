"""Tests for _is_write_denied() — verifies deny list blocks sensitive paths on all platforms."""

import os
import pytest
from pathlib import Path

from tools.file_operations import _is_write_denied


class TestWriteDenyExactPaths:
    def test_etc_shadow(self):
        assert _is_write_denied("/etc/shadow") is True

    def test_etc_passwd(self):
        assert _is_write_denied("/etc/passwd") is True

    def test_etc_sudoers(self):
        assert _is_write_denied("/etc/sudoers") is True

    def test_ssh_authorized_keys(self):
        assert _is_write_denied("~/.ssh/authorized_keys") is True

    def test_ssh_id_rsa(self):
        path = os.path.join(str(Path.home()), ".ssh", "id_rsa")
        assert _is_write_denied(path) is True

    def test_ssh_id_ed25519(self):
        path = os.path.join(str(Path.home()), ".ssh", "id_ed25519")
        assert _is_write_denied(path) is True

    def test_netrc(self):
        path = os.path.join(str(Path.home()), ".netrc")
        assert _is_write_denied(path) is True

    def test_hermes_env(self):
        path = os.path.join(str(Path.home()), ".hermes", ".env")
        assert _is_write_denied(path) is True

    def test_shell_profiles(self):
        home = str(Path.home())
        for name in [".bashrc", ".zshrc", ".profile", ".bash_profile", ".zprofile"]:
            assert _is_write_denied(os.path.join(home, name)) is True, f"{name} should be denied"

    def test_package_manager_configs(self):
        home = str(Path.home())
        for name in [".npmrc", ".pypirc", ".pgpass"]:
            assert _is_write_denied(os.path.join(home, name)) is True, f"{name} should be denied"


class TestWriteDenyPrefixes:
    def test_ssh_prefix(self):
        path = os.path.join(str(Path.home()), ".ssh", "some_key")
        assert _is_write_denied(path) is True

    def test_aws_prefix(self):
        path = os.path.join(str(Path.home()), ".aws", "credentials")
        assert _is_write_denied(path) is True

    def test_gnupg_prefix(self):
        path = os.path.join(str(Path.home()), ".gnupg", "secring.gpg")
        assert _is_write_denied(path) is True

    def test_kube_prefix(self):
        path = os.path.join(str(Path.home()), ".kube", "config")
        assert _is_write_denied(path) is True

    def test_sudoers_d_prefix(self):
        assert _is_write_denied("/etc/sudoers.d/custom") is True

    def test_systemd_prefix(self):
        assert _is_write_denied("/etc/systemd/system/evil.service") is True


class TestWriteAllowed:
    def test_tmp_file(self):
        assert _is_write_denied("/tmp/safe_file.txt") is False

    def test_project_file(self):
        assert _is_write_denied("/home/user/project/main.py") is False

    def test_hermes_config_not_env(self):
        path = os.path.join(str(Path.home()), ".hermes", "config.yaml")
        assert _is_write_denied(path) is False
