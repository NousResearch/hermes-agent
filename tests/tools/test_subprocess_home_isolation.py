"""Tests for subprocess HOME handling in profile mode.

Hermes state stays profile-scoped through HERMES_HOME. Host subprocesses should
keep the user's real HOME by default so external CLIs find existing credentials.
Containers still use the profile home for persistence, and users can explicitly
opt into profile HOME isolation on the host.

See: https://github.com/NousResearch/hermes-agent/issues/25114
See: https://github.com/NousResearch/hermes-agent/issues/36144
See: https://github.com/NousResearch/hermes-agent/issues/29015
"""

import os
import sys
import threading
from pathlib import Path

import hermes_constants
import pytest



# ---------------------------------------------------------------------------
# get_subprocess_home()
# ---------------------------------------------------------------------------

class TestGetSubprocessHome:
    """Unit tests for hermes_constants.get_subprocess_home()."""

    def _host_mode(self, monkeypatch):
        monkeypatch.setattr(hermes_constants, "is_container", lambda: False)
        monkeypatch.delenv("TERMINAL_HOME_MODE", raising=False)
        monkeypatch.delenv("HERMES_REAL_HOME", raising=False)

    def _container_mode(self, monkeypatch):
        monkeypatch.setattr(hermes_constants, "is_container", lambda: True)
        monkeypatch.delenv("TERMINAL_HOME_MODE", raising=False)
        monkeypatch.delenv("HERMES_REAL_HOME", raising=False)



    def test_host_auto_keeps_real_home_when_profile_home_exists(self, tmp_path, monkeypatch):
        """Host installs should not hide real ~/.ssh, ~/.gitconfig, ~/.azure, etc."""
        self._host_mode(monkeypatch)
        real_home = tmp_path / "real-home"
        hermes_home = real_home / ".hermes" / "profiles" / "coder"
        profile_home = hermes_home / "home"
        profile_home.mkdir(parents=True)
        monkeypatch.setenv("HOME", str(real_home))
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        from hermes_constants import get_subprocess_home
        assert get_subprocess_home() is None

    def test_container_auto_uses_profile_home_when_home_dir_exists(self, tmp_path, monkeypatch):
        self._container_mode(monkeypatch)
        hermes_home = tmp_path / ".hermes"
        profile_home = hermes_home / "home"
        profile_home.mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        from hermes_constants import get_subprocess_home
        assert get_subprocess_home() == str(profile_home)

    def test_returns_profile_specific_path(self, tmp_path, monkeypatch):
        """Explicit profile mode keeps the old per-profile HOME behavior."""
        self._host_mode(monkeypatch)
        profile_dir = tmp_path / ".hermes" / "profiles" / "coder"
        profile_dir.mkdir(parents=True)
        profile_home = profile_dir / "home"
        profile_home.mkdir()
        monkeypatch.setenv("TERMINAL_HOME_MODE", "profile")
        monkeypatch.setenv("HERMES_HOME", str(profile_dir))
        from hermes_constants import get_subprocess_home
        assert get_subprocess_home() == str(profile_home)

    def test_real_mode_repairs_parent_home_already_pointing_at_profile(self, tmp_path, monkeypatch):
        self._host_mode(monkeypatch)
        profile_dir = tmp_path / ".hermes" / "profiles" / "coder"
        profile_home = profile_dir / "home"
        profile_home.mkdir(parents=True)
        real_home = tmp_path / "real-home"
        real_home.mkdir()
        monkeypatch.setenv("TERMINAL_HOME_MODE", "real")
        monkeypatch.setenv("HERMES_HOME", str(profile_dir))
        monkeypatch.setenv("HOME", str(profile_home))
        monkeypatch.setenv("HERMES_REAL_HOME", str(real_home))

        from hermes_constants import get_subprocess_home, get_real_home

        assert get_real_home() == str(real_home)
        assert get_subprocess_home() == str(real_home)


    def test_two_profiles_get_different_homes(self, tmp_path, monkeypatch):
        self._container_mode(monkeypatch)
        base = tmp_path / ".hermes" / "profiles"
        for name in ("alpha", "beta"):
            p = base / name
            p.mkdir(parents=True)
            (p / "home").mkdir()

        from hermes_constants import get_subprocess_home

        monkeypatch.setenv("HERMES_HOME", str(base / "alpha"))
        home_a = get_subprocess_home()

        monkeypatch.setenv("HERMES_HOME", str(base / "beta"))
        home_b = get_subprocess_home()

        assert home_a is not None
        assert home_b is not None
        assert home_a != home_b
        assert home_a.endswith("alpha/home")
        assert home_b.endswith("beta/home")



# ---------------------------------------------------------------------------
# _make_run_env() injection
# ---------------------------------------------------------------------------

class TestMakeRunEnvHomeInjection:
    """Verify _make_run_env() applies the subprocess HOME policy."""

    def test_host_auto_preserves_real_home_when_profile_home_exists(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        (hermes_home / "home").mkdir()
        real_home = tmp_path / "real-home"
        real_home.mkdir()
        monkeypatch.setattr(hermes_constants, "is_container", lambda: False)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("HOME", str(real_home))
        monkeypatch.setenv("PATH", "/usr/bin:/bin")

        from tools.environments.local import _make_run_env
        result = _make_run_env({})

        assert result["HOME"] == str(real_home)
        assert result["HERMES_REAL_HOME"] == str(real_home)

    def test_profile_mode_injects_profile_home_when_profile_home_exists(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        (hermes_home / "home").mkdir()
        real_home = tmp_path / "real-home"
        real_home.mkdir()
        monkeypatch.setattr(hermes_constants, "is_container", lambda: False)
        monkeypatch.setenv("TERMINAL_HOME_MODE", "profile")
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("HOME", str(real_home))
        monkeypatch.setenv("PATH", "/usr/bin:/bin")

        from tools.environments.local import _make_run_env
        result = _make_run_env({})

        assert result["HOME"] == str(hermes_home / "home")
        assert result["HERMES_REAL_HOME"] == str(real_home)

    def test_no_injection_when_home_dir_missing(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        # No home/ subdirectory
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("HOME", "/root")
        monkeypatch.setenv("PATH", "/usr/bin:/bin")

        from tools.environments.local import _make_run_env
        result = _make_run_env({})

        assert result["HOME"] == "/root"

    def test_no_injection_when_hermes_home_unset(self, monkeypatch):
        monkeypatch.delenv("HERMES_HOME", raising=False)
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
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        (hermes_home / "home").mkdir()
        real_home = tmp_path / "real-home"
        real_home.mkdir()
        monkeypatch.setattr(hermes_constants, "is_container", lambda: False)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        base_env = {"HOME": str(real_home), "PATH": "/usr/bin", "USER": "root"}
        from tools.environments.local import _sanitize_subprocess_env
        result = _sanitize_subprocess_env(base_env)

        assert result["HOME"] == str(real_home)
        assert result["HERMES_REAL_HOME"] == str(real_home)

    def test_profile_mode_injects_profile_home_when_profile_home_exists(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        (hermes_home / "home").mkdir()
        real_home = tmp_path / "real-home"
        real_home.mkdir()
        monkeypatch.setattr(hermes_constants, "is_container", lambda: False)
        monkeypatch.setenv("TERMINAL_HOME_MODE", "profile")
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        base_env = {"HOME": str(real_home), "PATH": "/usr/bin", "USER": "root"}
        from tools.environments.local import _sanitize_subprocess_env
        result = _sanitize_subprocess_env(base_env)

        assert result["HOME"] == str(hermes_home / "home")
        assert result["HERMES_REAL_HOME"] == str(real_home)

    def test_no_injection_when_home_dir_missing(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        base_env = {"HOME": "/root", "PATH": "/usr/bin"}
        from tools.environments.local import _sanitize_subprocess_env
        result = _sanitize_subprocess_env(base_env)

        assert result["HOME"] == "/root"


@pytest.mark.parametrize("builder_name", ["build", "hermes"])
@pytest.mark.parametrize("empty_policy_keys", [False, True])
def test_explicit_subprocess_base_never_reads_late_home_policy(
    builder_name, empty_policy_keys, tmp_path, monkeypatch
):
    """A captured child base treats missing HOME-policy keys as authoritative absence."""
    launch_home = tmp_path / "launch"
    (launch_home / "home").mkdir(parents=True)
    real_home = tmp_path / "real-home"
    real_home.mkdir()
    late_real_home = tmp_path / "secondary-poison"
    late_real_home.mkdir()
    monkeypatch.setattr(hermes_constants, "is_container", lambda: False)
    monkeypatch.setenv("HERMES_REAL_HOME", str(late_real_home))
    monkeypatch.setenv("TERMINAL_HOME_MODE", "profile")
    base = {
        "HERMES_HOME": str(launch_home),
        "HOME": str(real_home),
        "PATH": "/usr/bin",
    }
    if empty_policy_keys:
        base.update(HERMES_REAL_HOME="", TERMINAL_HOME_MODE="")

    from tools.environments.local import build_subprocess_env, hermes_subprocess_env

    if builder_name == "build":
        result = build_subprocess_env(base=base, scrub_secrets=False)
    else:
        result = hermes_subprocess_env(base=base, inherit_credentials=True)

    assert result["HERMES_REAL_HOME"] == str(real_home)
    assert result["HOME"] == str(real_home)

    # Snapshot-owned variables used inside HERMES_HOME must not be expanded from
    # the later process environment on the strict child path.
    snapshot_root = tmp_path / "snapshot-root"
    snapshot_profile = snapshot_root / "profile"
    (snapshot_profile / "home").mkdir(parents=True)
    late_root = tmp_path / "late-root"
    (late_root / "profile" / "home").mkdir(parents=True)
    monkeypatch.setenv("PROFILE_ROOT", str(late_root))
    snapshot_env = {
        "HERMES_HOME": "$PROFILE_ROOT/profile",
        "PROFILE_ROOT": str(snapshot_root),
        "HOME": str(real_home),
        "TERMINAL_HOME_MODE": "profile",
    }
    assert hermes_constants.get_subprocess_home(
        snapshot_env, allow_process_fallback=False
    ) == str(snapshot_profile / "home")
    assert hermes_constants.get_process_hermes_home(snapshot_env) == snapshot_profile

    # The canonical default launch home is part of the same snapshot authority:
    # resolving it must not consult later HOME/LOCALAPPDATA process mutations.
    monkeypatch.setattr(sys, "platform", "linux")
    assert hermes_constants.get_process_hermes_home(
        {"HOME": str(real_home)}
    ) == real_home / ".hermes"
    monkeypatch.setattr(sys, "platform", "win32")
    assert hermes_constants.get_process_hermes_home(
        {"LOCALAPPDATA": str(real_home)}
    ) == real_home / "hermes"
    monkeypatch.setattr(Path, "home", lambda: late_real_home)
    for platform, empty_snapshot in (
        ("linux", {}),
        ("linux", {"HOME": ""}),
        ("win32", {}),
        ("win32", {"LOCALAPPDATA": "", "USERPROFILE": "", "HOME": ""}),
    ):
        monkeypatch.setattr(sys, "platform", platform)
        with pytest.raises(ValueError, match="platform home"):
            hermes_constants.get_process_hermes_home(empty_snapshot)


def test_explicit_home_lookup_never_reads_os_account(monkeypatch):
    """Strict snapshot mode treats an absent real-home value as absent."""
    pwd = pytest.importorskip("pwd")

    class _Entry:
        pw_dir = "/ambient-account-home"

    monkeypatch.setattr(pwd, "getpwuid", lambda _uid: _Entry())

    assert hermes_constants.get_real_home(
        {}, allow_process_fallback=False
    ) == "/tmp"


def test_strict_home_normalization_never_reads_ambient_home(tmp_path, monkeypatch):
    """A relative snapshot home is anchored to its captured PWD, not live cwd."""
    launch_cwd = tmp_path / "launch-cwd"
    (launch_cwd / "home").mkdir(parents=True)
    late_cwd = tmp_path / "late-cwd"
    late_cwd.mkdir()
    monkeypatch.setattr(hermes_constants, "is_container", lambda: False)
    snapshot = {
        "HERMES_HOME": ".",
        "HOME": "home",
        "PWD": str(launch_cwd),
        "TERMINAL_HOME_MODE": "profile",
    }

    monkeypatch.chdir(launch_cwd)
    assert hermes_constants.get_subprocess_home(
        snapshot, allow_process_fallback=False
    ) == str(launch_cwd / "home")
    assert hermes_constants.get_process_hermes_home(snapshot) == launch_cwd
    monkeypatch.chdir(late_cwd)
    assert hermes_constants.get_subprocess_home(
        snapshot, allow_process_fallback=False
    ) == str(launch_cwd / "home")
    assert hermes_constants.get_process_hermes_home(snapshot) == launch_cwd


def test_explicit_named_tilde_does_not_use_live_expanduser(monkeypatch):
    """An explicit mapping never resolves ``~user`` through ambient OS state."""
    def _unexpected_expanduser(_path):
        raise AssertionError("explicit snapshot called live expanduser")

    monkeypatch.setattr(os.path, "expanduser", _unexpected_expanduser)
    assert hermes_constants.get_process_hermes_home(
        {"HERMES_HOME": "~other/profile", "HOME": "/snapshot-home"}
    ) == Path("~other/profile")


@pytest.mark.parametrize("raw_home", ["$MISSING/profile", "~other/profile"])
def test_strict_unresolved_profile_home_does_not_probe_live_cwd(
    raw_home, tmp_path, monkeypatch
):
    """Literal unresolved homes remain stable even if a same-named cwd path exists."""
    first_cwd = tmp_path / "first"
    second_cwd = tmp_path / "second"
    (first_cwd / raw_home / "home").mkdir(parents=True)
    second_cwd.mkdir()
    snapshot = {
        "HERMES_HOME": raw_home,
        "HOME": "/snapshot-home",
        "PWD": str(tmp_path),
        "TERMINAL_HOME_MODE": "profile",
    }
    expected = str(Path(raw_home) / "home")

    monkeypatch.chdir(first_cwd)
    assert hermes_constants.get_subprocess_home(
        snapshot, allow_process_fallback=False
    ) == expected
    monkeypatch.chdir(second_cwd)
    assert hermes_constants.get_subprocess_home(
        snapshot, allow_process_fallback=False
    ) == expected


def test_strict_subprocess_home_ignores_late_container_env(tmp_path, monkeypatch):
    """Container policy for an explicit snapshot cannot read a later Kubernetes marker."""
    profile = tmp_path / "profile"
    (profile / "home").mkdir(parents=True)
    real_home = tmp_path / "real-home"
    real_home.mkdir()
    snapshot = {
        "HERMES_HOME": str(profile),
        "HOME": str(real_home),
        "PWD": str(tmp_path),
    }
    original_exists = os.path.exists
    monkeypatch.setattr(
        os.path,
        "exists",
        lambda path: False
        if path in {"/.dockerenv", "/run/.containerenv"}
        else original_exists(path),
    )
    monkeypatch.setattr(hermes_constants, "_proc_file_has_marker", lambda *_args: False)
    monkeypatch.setattr(hermes_constants, "_root_mount_has_marker", lambda *_args: False)
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    monkeypatch.setattr(hermes_constants, "_container_detected", None)
    assert hermes_constants.get_subprocess_home(
        snapshot, allow_process_fallback=False
    ) is None

    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "late-poison")
    monkeypatch.setattr(hermes_constants, "_container_detected", None)
    assert hermes_constants.get_subprocess_home(
        snapshot, allow_process_fallback=False
    ) is None



# ---------------------------------------------------------------------------
# Profile bootstrap
# ---------------------------------------------------------------------------

class TestProfileBootstrap:
    """Verify new profiles get a home/ subdirectory."""

    def test_profile_dirs_includes_home(self):
        from hermes_cli.profiles import _PROFILE_DIRS
        assert "home" in _PROFILE_DIRS

    def test_create_profile_bootstraps_home_dir(self, tmp_path, monkeypatch):
        """create_profile() should create home/ inside the profile dir."""
        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(home))

        from hermes_cli.profiles import create_profile
        profile_dir = create_profile("testbot", no_alias=True)
        assert (profile_dir / "home").is_dir()


# ---------------------------------------------------------------------------
# Python process HOME unchanged
# ---------------------------------------------------------------------------

