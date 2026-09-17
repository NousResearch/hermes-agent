"""Tests for _is_write_denied() — verifies deny list blocks sensitive paths on all platforms."""

import json
import os

from pathlib import Path
from unittest.mock import patch

import pytest

from agent.file_safety import is_write_denied as _is_write_denied


class TestWriteDenyExactPaths:
    def test_etc_shadow(self):
        assert _is_write_denied("/etc/shadow") is True


    def test_ssh_authorized_keys(self):
        assert _is_write_denied("~/.ssh/authorized_keys") is True


    def test_ssh_id_ed25519(self):
        path = os.path.join(str(Path.home()), ".ssh", "id_ed25519")
        assert _is_write_denied(path) is True


    def test_hermes_root_env_when_running_under_profile(self, tmp_path, monkeypatch):
        """Top-level ``<root>/.env`` stays write-denied even when running under
        a profile (#15981).

        Before the fix, ``build_write_denied_paths`` only added
        ``<active_profile>/.env`` to the deny list, so the global
        ``~/.hermes/.env`` (whose credentials are inherited by every profile)
        could be silently overwritten by ``write_file`` while a profile was
        active.
        """
        root = tmp_path / "hermes_root"
        profile_home = root / "profiles" / "coder"
        profile_home.mkdir(parents=True)
        global_env = root / ".env"
        global_env.write_text("OPENAI_API_KEY=sk-real\n")

        monkeypatch.setenv("HERMES_HOME", str(profile_home))

        # Sanity check: HERMES_HOME does point to the profile dir, not the root.
        from hermes_constants import get_hermes_home, get_default_hermes_root
        assert get_hermes_home() == profile_home
        assert get_default_hermes_root() == root

        assert _is_write_denied(str(global_env)) is True

    def test_shell_profiles_are_writable(self):
        home = str(Path.home())
        for name in [".bashrc", ".zshrc", ".profile", ".bash_profile", ".zprofile"]:
            assert _is_write_denied(os.path.join(home, name)) is False, f"{name} should be writable"

    def test_credential_config_files_denied(self):
        home = str(Path.home())
        for name in [".netrc", ".pgpass", ".npmrc", ".pypirc"]:
            assert _is_write_denied(os.path.join(home, name)) is True, f"{name} should be denied"


class TestWriteDenyPrefixes:
    def test_ssh_prefix(self):
        path = os.path.join(str(Path.home()), ".ssh", "some_key")
        assert _is_write_denied(path) is True


    def test_systemd_prefix(self, tmp_path):
        # On NixOS, /etc/systemd is a symlink into /nix/store, so
        # realpath() resolves it to a store path that doesn't match
        # the /etc/systemd/ prefix.  Build a real directory tree so
        # realpath is a no-op and prefix matching works.
        fake_etc = tmp_path / "etc" / "systemd" / "system"
        fake_etc.mkdir(parents=True)
        target = str(fake_etc / "evil.service")
        # Patch the prefix builder to include our tmp_path prefix
        import agent.file_safety as _fs
        _orig = _fs.build_write_denied_prefixes
        _extra_prefix = str(tmp_path / "etc" / "systemd") + os.sep
        def _patched(home):
            return _orig(home) + [_extra_prefix]
        with patch.object(_fs, "build_write_denied_prefixes", _patched):
            assert _is_write_denied(target) is True


class TestWriteAllowed:
    def test_tmp_file(self):
        assert _is_write_denied("/tmp/safe_file.txt") is False


    def test_hermes_control_files_requested_writable(self):
        from hermes_constants import get_hermes_home

        home = get_hermes_home()
        for name in ["auth.json", "config.yaml", "webhook_subscriptions.json"]:
            assert _is_write_denied(str(home / name)) is False, f"{name} should be writable"


class TestProfileHomeE2E:
    """End-to-end through ``write_file_tool``: with the process HOME pinned to
    ``{HERMES_HOME}/home`` (TERMINAL_HOME_MODE=profile / container / spawned worker),
    a write aimed at the OS user's real home must still hit the credential guards —
    before the fix, absolute real-home paths sailed through untouched."""

    @pytest.fixture()
    def pinned_profile_home(self, tmp_path, monkeypatch):
        profile = tmp_path / "profile"
        (profile / "home").mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(profile))
        monkeypatch.setenv("HOME", str(profile / "home"))
        return profile

    @staticmethod
    def _real_home() -> Path:
        import pwd

        return Path(pwd.getpwuid(os.getuid()).pw_dir)

    def test_absolute_real_home_write_denied_end_to_end(self, pinned_profile_home):
        from tools.file_tools import write_file_tool

        target = self._real_home() / ".ssh" / "e2e_guard_probe_key"
        assert not target.exists()
        try:
            result = json.loads(write_file_tool(str(target), "not-a-key"))
            assert result.get("error"), f"write slipped through: {result}"
            assert not target.exists()
        finally:
            target.unlink(missing_ok=True)

    def test_tilde_write_denied_end_to_end(self, pinned_profile_home):
        """``~`` resolves to the real home via _expand_tilde's repair — and is denied."""
        from tools.file_tools import write_file_tool

        target = self._real_home() / ".aws" / "e2e_guard_probe_credentials"
        assert not target.exists()
        try:
            result = json.loads(write_file_tool("~/.aws/e2e_guard_probe_credentials", "x"))
            assert result.get("error"), f"write slipped through: {result}"
            assert not target.exists()
        finally:
            target.unlink(missing_ok=True)

    def test_named_user_tilde_denied(self, pinned_profile_home):
        """``~root/...`` resolves to another account's home — still a credential write."""
        import agent.file_safety as fs
        assert fs.is_write_denied("~root/.ssh/authorized_keys") is True
        assert fs.is_write_denied("~nosuchuser-hopefully/.ssh/authorized_keys") is False

    def test_benign_write_still_lands(self, pinned_profile_home, tmp_path):
        from tools.file_tools import write_file_tool

        target = tmp_path / "scratch" / "ok.txt"
        target.parent.mkdir(parents=True)
        result = json.loads(write_file_tool(str(target), "hello"))
        assert not result.get("error"), result
        assert target.read_text() == "hello"
