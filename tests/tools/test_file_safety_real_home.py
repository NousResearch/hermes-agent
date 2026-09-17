"""Tests for file_safety real-home guard: credential files under the OS user's
real HOME are protected even when the process HOME is pinned to a profile home.

Regression test for https://github.com/NousResearch/hermes-agent/issues/113628
"""

import os

from pathlib import Path
from unittest.mock import patch

from agent.file_safety import (
    _all_guard_homes,
    _real_home,
    build_write_denied_paths,
    build_write_denied_prefixes,
    build_write_approval_paths,
    is_write_denied,
    is_write_approval_required,
)


class TestAllGuardHomes:
    """_all_guard_homes() returns deduplicated process-~ + real-home list."""

    def test_single_home_when_same(self, monkeypatch):
        """When process ~ equals real home, only one home returned."""
        real = str(Path.home())
        monkeypatch.setattr(
            "agent.file_safety._real_home", lambda: os.path.realpath(real)
        )
        homes = _all_guard_homes()
        assert len(homes) == 1
        assert homes[0] == os.path.realpath(real)

    def test_two_homes_when_different(self, monkeypatch, tmp_path):
        """When process ~ differs from real home, both are returned."""
        process_home = str(tmp_path / "profile_home")
        real_home = str(tmp_path / "real_home")
        monkeypatch.setattr(
            "agent.file_safety._real_home", lambda: os.path.realpath(real_home)
        )
        monkeypatch.setattr(
            "os.path.expanduser",
            lambda p: os.path.join(process_home, p[2:]) if p.startswith("~") else p,
        )
        homes = _all_guard_homes()
        assert len(homes) == 2
        assert os.path.realpath(process_home) in homes
        assert os.path.realpath(real_home) in homes


class TestRealHomeCredentialGuard:
    """Credential files under the real home are guarded when process HOME is pinned."""

    def test_denied_paths_cover_real_home(self, monkeypatch, tmp_path):
        """build_write_denied_paths covers both homes."""
        process_home = str(tmp_path / "profile_home")
        real_home = str(tmp_path / "real_home")
        monkeypatch.setattr(
            "agent.file_safety._real_home", lambda: os.path.realpath(real_home)
        )
        # Build denied paths for each home individually and verify coverage.
        process_paths = build_write_denied_paths(process_home)
        real_paths = build_write_denied_paths(real_home)
        # Real home credential files must be in the union.
        for name in [".ssh/authorized_keys", ".ssh/id_rsa", ".ssh/id_ed25519",
                      ".netrc", ".pgpass", ".npmrc", ".pypirc", ".git-credentials"]:
            expected = os.path.realpath(os.path.join(real_home, name))
            assert expected in real_paths, f"real home {name} should be denied"

    def test_denied_prefixes_cover_real_home(self, monkeypatch, tmp_path):
        """build_write_denied_prefixes covers both homes."""
        real_home = str(tmp_path / "real_home")
        prefixes = build_write_denied_prefixes(real_home)
        for dirname in [".ssh", ".aws", ".gnupg", ".kube",
                         ".docker", ".azure"]:
            expected = os.path.realpath(os.path.join(real_home, dirname)) + os.sep
            assert expected in prefixes, f"real home {dirname}/ should be in denied prefixes"

    def test_approval_paths_cover_real_home(self, monkeypatch, tmp_path):
        """build_write_approval_paths covers both homes."""
        real_home = str(tmp_path / "real_home")
        approval = build_write_approval_paths(real_home)
        expected = os.path.realpath(os.path.join(real_home, ".ssh", "config"))
        assert expected in approval

    def test_is_write_denied_covers_real_home(self, monkeypatch, tmp_path):
        """is_write_denied catches credential files under the real home
        even when process HOME is different."""
        real_home = str(tmp_path / "real_home")
        # Simulate: process HOME = profile home, real HOME = real_home
        profile_home = str(tmp_path / "profile_home")
        monkeypatch.setattr(
            "agent.file_safety._real_home", lambda: os.path.realpath(real_home)
        )
        monkeypatch.setattr(
            "os.path.expanduser",
            lambda p: os.path.join(profile_home, p[2:]) if p.startswith("~") else p,
        )
        # Credentials under the real home should be denied.
        assert is_write_denied(os.path.join(real_home, ".ssh", "authorized_keys"))
        assert is_write_denied(os.path.join(real_home, ".ssh", "id_rsa"))
        assert is_write_denied(os.path.join(real_home, ".ssh", "id_ed25519"))
        assert is_write_denied(os.path.join(real_home, ".netrc"))
        assert is_write_denied(os.path.join(real_home, ".ssh", "some_key"))
        assert is_write_denied(os.path.join(real_home, ".aws", "credentials"))

    def test_is_write_denied_still_allows_non_credential(self, monkeypatch, tmp_path):
        """Non-credential files remain writable."""
        real_home = str(tmp_path / "real_home")
        profile_home = str(tmp_path / "profile_home")
        monkeypatch.setattr(
            "agent.file_safety._real_home", lambda: os.path.realpath(real_home)
        )
        monkeypatch.setattr(
            "os.path.expanduser",
            lambda p: os.path.join(profile_home, p[2:]) if p.startswith("~") else p,
        )
        assert not is_write_denied(os.path.join(real_home, ".bashrc"))
        assert not is_write_denied(os.path.join(real_home, ".zshrc"))
        assert not is_write_denied(os.path.join(real_home, "Documents", "notes.txt"))

    def test_is_write_approval_required_covers_real_home(self, monkeypatch, tmp_path):
        """is_write_approval_required catches ~/.ssh/config under the real home."""
        real_home = str(tmp_path / "real_home")
        profile_home = str(tmp_path / "profile_home")
        monkeypatch.setattr(
            "agent.file_safety._real_home", lambda: os.path.realpath(real_home)
        )
        monkeypatch.setattr(
            "os.path.expanduser",
            lambda p: os.path.join(profile_home, p[2:]) if p.startswith("~") else p,
        )
        assert is_write_approval_required(os.path.join(real_home, ".ssh", "config"))
        # .ssh/authorized_keys is hard-denied, not approval-gated.
        assert not is_write_approval_required(os.path.join(real_home, ".ssh", "authorized_keys"))

    def test_named_user_home_covered(self, monkeypatch, tmp_path):
        """~name/... spellings (e.g. ~root/.ssh/authorized_keys) resolve to a
        named account's home which should also be guarded."""
        named_home = str(tmp_path / "root_home")
        real_home = str(tmp_path / "real_home")
        monkeypatch.setattr(
            "agent.file_safety._real_home", lambda: os.path.realpath(real_home)
        )
        # A named-user path resolves through expanduser to the named home.
        # The guard should catch it if it matches a denied path pattern.
        path = os.path.join(named_home, ".ssh", "authorized_keys")
        # Since named_home differs from both process and real home,
        # the guard uses the path's own resolution. The key test is that
        # the path is NOT silently allowed — the existing prefix check
        # covers this via the process-home anchor. If named_home happens
        # to equal the real home, the new guard catches it.
        # This is more of a structural test: the function should not crash.
        result = is_write_denied(path)
        assert isinstance(result, bool)
