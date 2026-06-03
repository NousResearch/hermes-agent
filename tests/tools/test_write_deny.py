"""Tests for _is_write_denied() — verifies deny list blocks sensitive paths on all platforms."""

import os

from pathlib import Path
from unittest.mock import patch

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
        # ``.env`` under the active HERMES_HOME (profile-aware, not just
        # ``~/.hermes``) must be write-denied. The hermetic test conftest
        # points HERMES_HOME at a tempdir — resolve via get_hermes_home()
        # to match the denylist.
        from hermes_constants import get_hermes_home
        path = str(get_hermes_home() / ".env")
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


class TestCredentialStoreWriteParity:
    """Every HERMES_HOME credential store blocked from *reading* must also be
    blocked from *writing*.

    Regression: ``auth.lock``, ``auth/google_oauth.json`` and
    ``cache/bws_cache.json`` were read-blocked (``get_read_block_error``) but
    writable, so a prompt-injected ``write_file`` could overwrite the Gemini
    OAuth token store / Bitwarden secret cache, or clobber the auth lock the
    agent never legitimately writes. The fix shares one ``CREDENTIAL_FILE_NAMES``
    list between the read and write guards.
    """

    def test_auth_lock_write_denied(self):
        from hermes_constants import get_hermes_home
        path = str(get_hermes_home() / "auth.lock")
        assert _is_write_denied(path) is True

    def test_google_oauth_token_write_denied(self):
        from hermes_constants import get_hermes_home
        path = str(get_hermes_home() / "auth" / "google_oauth.json")
        assert _is_write_denied(path) is True

    def test_bitwarden_cache_write_denied(self):
        from hermes_constants import get_hermes_home
        path = str(get_hermes_home() / "cache" / "bws_cache.json")
        assert _is_write_denied(path) is True

    def test_every_read_blocked_credential_is_write_denied(self):
        """Invariant: ``CREDENTIAL_FILE_NAMES`` (the read block) is a subset of
        the write deny list. Adding a new credential store to the read block
        automatically protects it from overwrites — this contract is what keeps
        the two guards from drifting apart again."""
        from hermes_constants import get_hermes_home
        from agent.file_safety import CREDENTIAL_FILE_NAMES

        home = get_hermes_home()
        for name in CREDENTIAL_FILE_NAMES:
            path = str(home / name)
            assert _is_write_denied(path) is True, (
                f"{name} is read-blocked but still writable"
            )

    def test_root_credential_write_denied_under_profile(self, tmp_path, monkeypatch):
        """Under a profile, the global ``<root>/auth.lock`` (and other root
        credential stores every profile inherits) stays write-denied — same
        widening as the root ``.env`` case (#15981)."""
        root = tmp_path / "hermes_root"
        profile_home = root / "profiles" / "coder"
        profile_home.mkdir(parents=True)
        (root / "auth.lock").write_text(" ")

        monkeypatch.setenv("HERMES_HOME", str(profile_home))

        assert _is_write_denied(str(root / "auth.lock")) is True


class TestWriteAllowed:
    def test_tmp_file(self):
        assert _is_write_denied("/tmp/safe_file.txt") is False

    def test_project_file(self):
        assert _is_write_denied("/home/user/project/main.py") is False

    def test_hermes_config_not_env(self):
        path = os.path.join(str(Path.home()), ".hermes", "config.yaml")
        assert _is_write_denied(path) is False
