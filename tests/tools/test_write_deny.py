"""Tests for _is_write_denied() — verifies deny list blocks sensitive paths on all platforms."""

import os

from pathlib import Path
from unittest.mock import patch

from tools.file_operations import _is_write_denied


class TestWriteDenyExactPaths:
    def test_etc_shadow(self):
        assert _is_write_denied("/etc/shadow") is True


    def test_ssh_authorized_keys(self):
        assert _is_write_denied("~/.ssh/authorized_keys") is True


    def test_ssh_id_ed25519(self):
        path = os.path.join(str(Path.home()), ".ssh", "id_ed25519")
        assert _is_write_denied(path) is True


    def test_hermes_env(self):
        # ``.env`` under the active HERMES_HOME (profile-aware, not just
        # ``~/.hermes``) must be write-denied. The hermetic test conftest
        # points HERMES_HOME at a tempdir — resolve via get_hermes_home()
        # to match the denylist.
        from hermes_constants import get_hermes_home
        path = str(get_hermes_home() / ".env")
        assert _is_write_denied(path) is True

    def test_plaintext_bitwarden_cache(self):
        from hermes_constants import get_hermes_home

        path = get_hermes_home() / "cache" / "bws_cache.json"
        assert _is_write_denied(str(path)) is True

    def test_encrypted_bitwarden_cache(self):
        from hermes_constants import get_hermes_home

        path = get_hermes_home() / "cache" / "bws_cache.enc.json"
        assert _is_write_denied(str(path)) is True

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
        for name in ["config.yaml"]:
            assert _is_write_denied(str(home / name)) is False, f"{name} should be writable"


class TestWriteDenyCredentialStore:
    """auth.json is the provider credential store (OAuth refresh tokens, API
    keys, the whole credential_pool). It was already read-denied via
    get_read_block_error()'s credential_file_names tuple, but missing from
    build_write_denied_paths() meant write_file/patch/delete/move could
    destroy it with no read-based warning first (#70942)."""

    def test_auth_json_denied(self):
        from hermes_constants import get_hermes_home

        path = get_hermes_home() / "auth.json"
        assert _is_write_denied(str(path)) is True

    def test_auth_lock_denied(self):
        from hermes_constants import get_hermes_home

        path = get_hermes_home() / "auth.lock"
        assert _is_write_denied(str(path)) is True

    def test_google_oauth_json_denied(self):
        from hermes_constants import get_hermes_home

        path = get_hermes_home() / "auth" / "google_oauth.json"
        assert _is_write_denied(str(path)) is True

    def test_webhook_subscriptions_json_denied(self):
        from hermes_constants import get_hermes_home

        path = get_hermes_home() / "webhook_subscriptions.json"
        assert _is_write_denied(str(path)) is True

    def test_auth_json_denied_at_root_when_running_under_profile(self, tmp_path, monkeypatch):
        """Top-level <root>/auth.json stays write-denied even when running
        under a profile, mirroring the existing .env root-widening (#15981)."""
        root = tmp_path / "hermes_root"
        profile_home = root / "profiles" / "coder"
        profile_home.mkdir(parents=True)
        global_auth = root / "auth.json"
        global_auth.write_text('{"anthropic-oauth": {"refresh_token": "rt-real"}}\n')

        monkeypatch.setenv("HERMES_HOME", str(profile_home))

        from hermes_constants import get_hermes_home, get_default_hermes_root
        assert get_hermes_home() == profile_home
        assert get_default_hermes_root() == root

        assert _is_write_denied(str(global_auth)) is True

    def test_plaintext_bitwarden_cache_denied_at_root_when_running_under_profile(
        self, tmp_path, monkeypatch
    ):
        """Top-level <root>/cache/bws_cache.json stores plaintext Bitwarden
        secret values and stays write-denied even when a profile is active."""
        root = tmp_path / "hermes_root"
        profile_home = root / "profiles" / "coder"
        profile_home.mkdir(parents=True)
        global_cache = root / "cache" / "bws_cache.json"
        global_cache.parent.mkdir(parents=True)
        global_cache.write_text('{"item": "PLAINTEXT_SECRET_VALUE"}\n')

        monkeypatch.setenv("HERMES_HOME", str(profile_home))

        from hermes_constants import get_hermes_home, get_default_hermes_root
        assert get_hermes_home() == profile_home
        assert get_default_hermes_root() == root

        assert _is_write_denied(str(global_cache)) is True
