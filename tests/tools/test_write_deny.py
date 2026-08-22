"""Tests for _is_write_denied() — verifies deny list blocks sensitive paths on all platforms."""

import os

from pathlib import Path
from unittest.mock import patch

import pytest

from tools.file_operations import _is_write_denied

# Credential stores that get_read_block_error() refuses to read, and which
# must therefore also be write-denied — a writable read-blocked credential
# file lets a prompt-injected write plant tokens it can never be caught
# reading back.
READ_BLOCKED_CREDENTIAL_FILES = [
    os.path.join("auth", "google_oauth.json"),
    os.path.join("cache", "bws_cache.json"),
]

# LIVE Google Workspace OAuth stores at the HERMES_HOME root: the token
# consumed by gws_bridge.py/google_api.py and the in-flight exchange state
# consumed by setup.py. Written by the skill's own scripts via direct file
# IO (never through the model-facing guard), so denying them here cannot
# break the trusted setup/refresh flow.
GOOGLE_WORKSPACE_TOKEN_FILES = [
    "google_token.json",
    "google_oauth_pending.json",
]

WRITE_DENIED_CREDENTIAL_FILES = (
    READ_BLOCKED_CREDENTIAL_FILES + GOOGLE_WORKSPACE_TOKEN_FILES
)


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

    @pytest.mark.parametrize("rel_path", WRITE_DENIED_CREDENTIAL_FILES)
    def test_active_home_credential_stores_denied(
        self, tmp_path, monkeypatch, rel_path
    ):
        """Credential stores under the active ``HERMES_HOME`` are write-denied.

        ``auth/google_oauth.json`` and ``cache/bws_cache.json`` are
        read-blocked by ``get_read_block_error()``; the write side only
        covered the encrypted Bitwarden cache (``cache/bws_cache.enc.json``),
        leaving the plaintext cache (still live — see
        ``agent.secret_sources.bitwarden._DISK_CACHE``) overwritable.
        ``google_token.json`` and ``google_oauth_pending.json`` are the LIVE
        Google Workspace token and pending-exchange state: left writable, a
        model-directed write can plant attacker-controlled token material
        that gws_bridge.py/google_api.py consume on the next run.
        """
        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))

        assert _is_write_denied(str(home / rel_path)) is True

    @pytest.mark.parametrize("rel_path", WRITE_DENIED_CREDENTIAL_FILES)
    def test_hermes_root_credential_stores_when_running_under_profile(
        self, tmp_path, monkeypatch, rel_path
    ):
        """The same stores stay write-denied at ``<root>/`` while a profile
        is active — same shape as the ``<root>/.env`` widening above (#15981).

        Every profile inherits the root credentials, so a root-level write is
        strictly worse than a per-profile one.
        """
        root = tmp_path / "hermes_root"
        profile_home = root / "profiles" / "coder"
        profile_home.mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(profile_home))

        from hermes_constants import get_hermes_home, get_default_hermes_root

        assert get_hermes_home() == profile_home
        assert get_default_hermes_root() == root

        assert _is_write_denied(str(root / rel_path)) is True
        assert _is_write_denied(str(profile_home / rel_path)) is True

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
