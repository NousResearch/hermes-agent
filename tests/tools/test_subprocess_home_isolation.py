"""Tests for subprocess HOME handling in profile mode.

Rabbit state stays profile-scoped through RABBIT_HOME. Host subprocesses should
keep the user's real HOME by default so external CLIs find existing credentials.
Containers still use the profile home for persistence, and users can explicitly
opt into profile HOME isolation on the host.

See: https://github.com/seven0070/Rabbit-/issues/25114
See: https://github.com/seven0070/Rabbit-/issues/36144
See: https://github.com/seven0070/Rabbit-/issues/29015
"""

from pathlib import Path

import rabbit_constants



# ---------------------------------------------------------------------------
# get_subprocess_home()
# ---------------------------------------------------------------------------

class TestGetSubprocessHome:
    """Unit tests for rabbit_constants.get_subprocess_home()."""

    def _host_mode(self, monkeypatch):
        monkeypatch.setattr(rabbit_constants, "is_container", lambda: False)
        monkeypatch.delenv("TERMINAL_HOME_MODE", raising=False)
        monkeypatch.delenv("RABBIT_REAL_HOME", raising=False)

    def _container_mode(self, monkeypatch):
        monkeypatch.setattr(rabbit_constants, "is_container", lambda: True)
        monkeypatch.delenv("TERMINAL_HOME_MODE", raising=False)
        monkeypatch.delenv("RABBIT_REAL_HOME", raising=False)



    def test_host_auto_keeps_real_home_when_profile_home_exists(self, tmp_path, monkeypatch):
        """Host installs should not hide real ~/.ssh, ~/.gitconfig, ~/.azure, etc."""
        self._host_mode(monkeypatch)
        real_home = tmp_path / "real-home"
        rabbit_home = real_home / ".rabbit" / "profiles" / "coder"
        profile_home = rabbit_home / "home"
        profile_home.mkdir(parents=True)
        monkeypatch.setenv("HOME", str(real_home))
        monkeypatch.setenv("RABBIT_HOME", str(rabbit_home))
        from rabbit_constants import get_subprocess_home
        assert get_subprocess_home() is None

    def test_host_auto_repairs_missing_home(self, tmp_path, monkeypatch):
        """A systemd system unit with no HOME at all should still get real HOME repaired."""
        self._host_mode(monkeypatch)
        real_home = tmp_path / "real-home"
        rabbit_home = real_home / ".rabbit" / "profiles" / "coder"
        profile_home = rabbit_home / "home"
        profile_home.mkdir(parents=True)
        monkeypatch.delenv("HOME", raising=False)
        monkeypatch.setenv("RABBIT_REAL_HOME", str(real_home))
        monkeypatch.setenv("RABBIT_HOME", str(rabbit_home))
        from rabbit_constants import get_subprocess_home
        assert get_subprocess_home() == str(real_home)

    def test_terminal_child_env_carries_home_when_host_has_none(self, tmp_path, monkeypatch):
        """A host with no HOME at all (systemd system unit without User=) must still hand
        terminal children a HOME; without it a `set -u` script dies on its first `$HOME` (#116081).

        Drives the production spawn seam, tools.environments.local.build_subprocess_env, not the
        resolver directly.
        """
        self._host_mode(monkeypatch)
        real_home = tmp_path / "real-home"
        real_home.mkdir()
        monkeypatch.delenv("HOME", raising=False)
        monkeypatch.setenv("RABBIT_REAL_HOME", str(real_home))
        monkeypatch.setenv("RABBIT_HOME", str(tmp_path / ".rabbit"))
        from tools.environments.local import build_subprocess_env
        assert build_subprocess_env()["HOME"] == str(real_home)

    def test_container_auto_uses_profile_home_when_home_dir_exists(self, tmp_path, monkeypatch):
        self._container_mode(monkeypatch)
        rabbit_home = tmp_path / ".rabbit"
        profile_home = rabbit_home / "home"
        profile_home.mkdir(parents=True)
        monkeypatch.setenv("RABBIT_HOME", str(rabbit_home))
        from rabbit_constants import get_subprocess_home
        assert get_subprocess_home() == str(profile_home)

    def test_returns_profile_specific_path(self, tmp_path, monkeypatch):
        """Explicit profile mode keeps the old per-profile HOME behavior."""
        self._host_mode(monkeypatch)
        profile_dir = tmp_path / ".rabbit" / "profiles" / "coder"
        profile_dir.mkdir(parents=True)
        profile_home = profile_dir / "home"
        profile_home.mkdir()
        monkeypatch.setenv("TERMINAL_HOME_MODE", "profile")
        monkeypatch.setenv("RABBIT_HOME", str(profile_dir))
        from rabbit_constants import get_subprocess_home
        assert get_subprocess_home() == str(profile_home)

    def test_real_mode_repairs_parent_home_already_pointing_at_profile(self, tmp_path, monkeypatch):
        self._host_mode(monkeypatch)
        profile_dir = tmp_path / ".rabbit" / "profiles" / "coder"
        profile_home = profile_dir / "home"
        profile_home.mkdir(parents=True)
        real_home = tmp_path / "real-home"
        real_home.mkdir()
        monkeypatch.setenv("TERMINAL_HOME_MODE", "real")
        monkeypatch.setenv("RABBIT_HOME", str(profile_dir))
        monkeypatch.setenv("HOME", str(profile_home))
        monkeypatch.setenv("RABBIT_REAL_HOME", str(real_home))

        from rabbit_constants import get_subprocess_home, get_real_home

        assert get_real_home() == str(real_home)
        assert get_subprocess_home() == str(real_home)


    def test_two_profiles_get_different_homes(self, tmp_path, monkeypatch):
        self._container_mode(monkeypatch)
        base = tmp_path / ".rabbit" / "profiles"
        for name in ("alpha", "beta"):
            p = base / name
            p.mkdir(parents=True)
            (p / "home").mkdir()

        from rabbit_constants import get_subprocess_home

        monkeypatch.setenv("RABBIT_HOME", str(base / "alpha"))
        home_a = get_subprocess_home()

        monkeypatch.setenv("RABBIT_HOME", str(base / "beta"))
        home_b = get_subprocess_home()

        assert home_a is not None
        assert home_b is not None
        assert home_a != home_b
        assert Path(home_a).parts[-2:] == ("alpha", "home")
        assert Path(home_b).parts[-2:] == ("beta", "home")



# ---------------------------------------------------------------------------
# _make_run_env() injection
# ---------------------------------------------------------------------------

class TestMakeRunEnvHomeInjection:
    """Verify _make_run_env() applies the subprocess HOME policy."""

    def test_host_auto_preserves_real_home_when_profile_home_exists(self, tmp_path, monkeypatch):
        rabbit_home = tmp_path / "rabbit"
        rabbit_home.mkdir()
        (rabbit_home / "home").mkdir()
        real_home = tmp_path / "real-home"
        real_home.mkdir()
        monkeypatch.setattr(rabbit_constants, "is_container", lambda: False)
        monkeypatch.setenv("RABBIT_HOME", str(rabbit_home))
        monkeypatch.setenv("HOME", str(real_home))
        monkeypatch.setenv("PATH", "/usr/bin:/bin")

        from tools.environments.local import _make_run_env
        result = _make_run_env({})

        assert result["HOME"] == str(real_home)
        assert result["RABBIT_REAL_HOME"] == str(real_home)

    def test_profile_mode_injects_profile_home_when_profile_home_exists(self, tmp_path, monkeypatch):
        rabbit_home = tmp_path / "rabbit"
        rabbit_home.mkdir()
        (rabbit_home / "home").mkdir()
        real_home = tmp_path / "real-home"
        real_home.mkdir()
        monkeypatch.setattr(rabbit_constants, "is_container", lambda: False)
        monkeypatch.setenv("TERMINAL_HOME_MODE", "profile")
        monkeypatch.setenv("RABBIT_HOME", str(rabbit_home))
        monkeypatch.setenv("HOME", str(real_home))
        monkeypatch.setenv("PATH", "/usr/bin:/bin")

        from tools.environments.local import _make_run_env
        result = _make_run_env({})

        assert result["HOME"] == str(rabbit_home / "home")
        assert result["RABBIT_REAL_HOME"] == str(real_home)

    def test_no_injection_when_home_dir_missing(self, tmp_path, monkeypatch):
        rabbit_home = tmp_path / "rabbit"
        rabbit_home.mkdir()
        # No home/ subdirectory
        monkeypatch.setenv("RABBIT_HOME", str(rabbit_home))
        monkeypatch.setenv("HOME", "/root")
        monkeypatch.setenv("PATH", "/usr/bin:/bin")

        from tools.environments.local import _make_run_env
        result = _make_run_env({})

        assert result["HOME"] == "/root"

    def test_no_injection_when_rabbit_home_unset(self, monkeypatch):
        monkeypatch.delenv("RABBIT_HOME", raising=False)
        monkeypatch.setenv("HOME", "/home/user")
        monkeypatch.setenv("PATH", "/usr/bin:/bin")

        from tools.environments.local import _make_run_env
        result = _make_run_env({})

        assert result["HOME"] == "/home/user"



# ---------------------------------------------------------------------------
# _sanitize_subprocess_env() injection
# ---------------------------------------------------------------------------

class TestSanitizeSubprocessEnvHomeInjection:
    """Verify _sanitize_subprocess_env() applies the subprocess HOME policy."""

    def test_host_auto_preserves_real_home_when_profile_home_exists(self, tmp_path, monkeypatch):
        rabbit_home = tmp_path / "rabbit"
        rabbit_home.mkdir()
        (rabbit_home / "home").mkdir()
        real_home = tmp_path / "real-home"
        real_home.mkdir()
        monkeypatch.setattr(rabbit_constants, "is_container", lambda: False)
        monkeypatch.setenv("RABBIT_HOME", str(rabbit_home))

        base_env = {"HOME": str(real_home), "PATH": "/usr/bin", "USER": "root"}
        from tools.environments.local import _sanitize_subprocess_env
        result = _sanitize_subprocess_env(base_env)

        assert result["HOME"] == str(real_home)
        assert result["RABBIT_REAL_HOME"] == str(real_home)

    def test_profile_mode_injects_profile_home_when_profile_home_exists(self, tmp_path, monkeypatch):
        rabbit_home = tmp_path / "rabbit"
        rabbit_home.mkdir()
        (rabbit_home / "home").mkdir()
        real_home = tmp_path / "real-home"
        real_home.mkdir()
        monkeypatch.setattr(rabbit_constants, "is_container", lambda: False)
        monkeypatch.setenv("TERMINAL_HOME_MODE", "profile")
        monkeypatch.setenv("RABBIT_HOME", str(rabbit_home))

        base_env = {"HOME": str(real_home), "PATH": "/usr/bin", "USER": "root"}
        from tools.environments.local import _sanitize_subprocess_env
        result = _sanitize_subprocess_env(base_env)

        assert result["HOME"] == str(rabbit_home / "home")
        assert result["RABBIT_REAL_HOME"] == str(real_home)

    def test_no_injection_when_home_dir_missing(self, tmp_path, monkeypatch):
        rabbit_home = tmp_path / "rabbit"
        rabbit_home.mkdir()
        monkeypatch.setenv("RABBIT_HOME", str(rabbit_home))

        base_env = {"HOME": "/root", "PATH": "/usr/bin"}
        from tools.environments.local import _sanitize_subprocess_env
        result = _sanitize_subprocess_env(base_env)

        assert result["HOME"] == "/root"



# ---------------------------------------------------------------------------
# Profile bootstrap
# ---------------------------------------------------------------------------

class TestProfileBootstrap:
    """Verify new profiles get a home/ subdirectory."""


    def test_create_profile_bootstraps_home_dir(self, tmp_path, monkeypatch):
        """create_profile() should create home/ inside the profile dir."""
        home = tmp_path / ".rabbit"
        home.mkdir()
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("RABBIT_HOME", str(home))

        from rabbit_cli.profiles import create_profile
        profile_dir = create_profile("testbot", no_alias=True)
        assert (profile_dir / "home").is_dir()


# ---------------------------------------------------------------------------
# Python process HOME unchanged
# ---------------------------------------------------------------------------

