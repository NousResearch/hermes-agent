"""Tests for the durable lazy-install target (immutable Docker images).

These cover the mechanism that lets opt-in backends lazy-install on the
sealed-venv Docker image without being able to break the agent core:
installs are redirected to a writable dir on the data volume, and that dir
is appended to the END of ``sys.path`` so the core venv always wins name
collisions.

The headline invariant — *a package in the durable store can never shadow
a core module* — is proved with a REAL install into a temp target (no
mocked pip), exercising the actual ``--target`` + sys.path-append path.
That E2E test is guarded by network availability; everything else is pure
unit logic with no network.
"""

from __future__ import annotations

import os
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

from tools import lazy_deps as ld


# ---------------------------------------------------------------------------
# Target resolution + gating
# ---------------------------------------------------------------------------


class TestTargetResolution:
    def test_no_target_when_env_unset(self, monkeypatch):
        monkeypatch.delenv(ld._LAZY_TARGET_ENV, raising=False)
        assert ld._lazy_install_target() is None

    def test_no_target_when_env_blank(self, monkeypatch):
        monkeypatch.setenv(ld._LAZY_TARGET_ENV, "   ")
        assert ld._lazy_install_target() is None

    def test_target_resolved_when_set(self, monkeypatch, tmp_path):
        monkeypatch.setenv(ld._LAZY_TARGET_ENV, str(tmp_path / "lazy"))
        assert ld._lazy_install_target() == tmp_path / "lazy"


class TestGatingWithTarget:
    """``HERMES_DISABLE_LAZY_INSTALLS=1`` must STOP blocking once a durable
    target is configured — the redirect is the safe path — but the config
    kill switch still wins in every mode."""

    def test_disable_env_blocks_without_target(self, monkeypatch):
        monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")
        monkeypatch.delenv(ld._LAZY_TARGET_ENV, raising=False)
        # config unreadable → fails open on the config check, but the sealed
        # env var with no target still blocks.
        monkeypatch.setattr(
            "hermes_cli.config.load_config", lambda: {}, raising=False
        )
        assert ld._allow_lazy_installs() is False

    def test_disable_env_allows_with_target(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")
        monkeypatch.setenv(ld._LAZY_TARGET_ENV, str(tmp_path))
        monkeypatch.setattr(
            "hermes_cli.config.load_config", lambda: {}, raising=False
        )
        assert ld._allow_lazy_installs() is True

    def test_config_killswitch_wins_even_with_target(self, monkeypatch, tmp_path):
        # Explicit opt-out must disable installs even when a target exists.
        monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")
        monkeypatch.setenv(ld._LAZY_TARGET_ENV, str(tmp_path))
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"security": {"allow_lazy_installs": False}},
            raising=False,
        )
        assert ld._allow_lazy_installs() is False

    def test_normal_mode_unaffected(self, monkeypatch):
        # No sealed env, no target → default allow (unchanged behaviour).
        monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
        monkeypatch.delenv(ld._LAZY_TARGET_ENV, raising=False)
        monkeypatch.setattr(
            "hermes_cli.config.load_config", lambda: {}, raising=False
        )
        assert ld._allow_lazy_installs() is True


# ---------------------------------------------------------------------------
# ABI stamp / durable-store rebuild safety
# ---------------------------------------------------------------------------


class TestAbiStamp:
    def test_creates_dir_and_stamp(self, tmp_path):
        target = tmp_path / "lazy"
        err = ld._ensure_target_ready(target)
        assert err is None
        assert target.is_dir()
        stamp = target / ld._TARGET_STAMP_NAME
        assert stamp.read_text().strip() == ld._python_abi_tag()

    def test_matching_stamp_preserves_contents(self, tmp_path):
        target = tmp_path / "lazy"
        ld._ensure_target_ready(target)
        # Drop a fake installed package.
        (target / "somepkg").mkdir()
        (target / "somepkg" / "__init__.py").write_text("x = 1\n")
        # Re-run with the SAME abi → contents must survive.
        err = ld._ensure_target_ready(target)
        assert err is None
        assert (target / "somepkg" / "__init__.py").exists()

    def test_mismatched_stamp_wipes_contents(self, tmp_path):
        target = tmp_path / "lazy"
        ld._ensure_target_ready(target)
        (target / "stalepkg").mkdir()
        (target / "stalepkg" / "mod.py").write_text("x = 1\n")
        # Simulate an image rebuild onto a different interpreter ABI.
        (target / ld._TARGET_STAMP_NAME).write_text("2.7:old-abi-tag")
        err = ld._ensure_target_ready(target)
        assert err is None
        # Stale package wiped; stamp refreshed to current ABI.
        assert not (target / "stalepkg").exists()
        assert (target / ld._TARGET_STAMP_NAME).read_text().strip() == ld._python_abi_tag()

    def test_readonly_target_reports_error(self, tmp_path):
        # A path under a non-writable parent should surface a clean error,
        # not raise.
        ro_parent = tmp_path / "ro"
        ro_parent.mkdir()
        os.chmod(ro_parent, 0o500)
        try:
            err = ld._ensure_target_ready(ro_parent / "lazy")
            assert err is not None
            assert "not writable" in err
        finally:
            os.chmod(ro_parent, 0o700)  # let pytest clean up


# ---------------------------------------------------------------------------
# sys.path append ordering (the core-wins invariant, unit level)
# ---------------------------------------------------------------------------


class TestSysPathAppend:
    def test_target_appended_not_prepended(self, tmp_path, monkeypatch):
        target = tmp_path / "lazy"
        target.mkdir()
        saved = list(sys.path)
        try:
            ld._activate_target_on_syspath(target)
            assert str(target) in sys.path
            # Must be at/after every pre-existing entry — i.e. core wins.
            idx = sys.path.index(str(target))
            assert idx >= len(saved), (
                "durable target must be appended after all core entries"
            )
        finally:
            sys.path[:] = saved

    def test_activation_idempotent(self, tmp_path, monkeypatch):
        target = tmp_path / "lazy"
        target.mkdir()
        saved = list(sys.path)
        try:
            ld._activate_target_on_syspath(target)
            ld._activate_target_on_syspath(target)
            assert sys.path.count(str(target)) == 1
        finally:
            sys.path[:] = saved


# ---------------------------------------------------------------------------
# E2E: a REAL install into a durable target cannot shadow core.
# ---------------------------------------------------------------------------


def _network_available() -> bool:
    try:
        import urllib.request
        urllib.request.urlopen("https://pypi.org/simple/", timeout=5)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _network_available(), reason="needs PyPI network access")
class TestRealInstallCoreWins:
    """Install a real package into a durable target and prove:

    1. It lands in the target dir, NOT the core venv.
    2. It is importable via the appended sys.path entry.
    3. A package name that ALSO exists in core resolves to the CORE copy,
       never the durable-store copy (the structural anti-shadow guarantee).
    """

    def test_install_lands_in_target_and_imports(self, tmp_path, monkeypatch):
        target = tmp_path / "lazy-packages"
        monkeypatch.setenv(ld._LAZY_TARGET_ENV, str(target))
        # 'isodate' is tiny, pure-python, and not shipped in the core venv,
        # so a successful import must resolve to the durable target.
        result = ld._venv_pip_install(("isodate==0.7.2",))
        assert result.success, f"install failed: {result.stderr}"
        # Landed in the durable target, not the core venv.
        installed = list(target.glob("isodate*"))
        assert installed, f"isodate not found under target {target}: {list(target.iterdir())}"
        # Importable now that the target is on sys.path.
        import importlib
        importlib.invalidate_caches()
        mod = importlib.import_module("isodate")
        assert mod.__file__ is not None
        assert Path(mod.__file__).is_relative_to(target)

    def test_core_package_is_not_shadowed(self, tmp_path, monkeypatch):
        """Force-install an OLD version of a package the core already ships
        into the durable target, then assert the running interpreter still
        imports the CORE version — proving append-ordering protects core.

        We use 'packaging', which is always present in the venv (transitive
        of pip/build tooling). We install a deliberately old pin into the
        target and check the resolved module path + version is core's.
        """
        import packaging  # core copy
        core_path = Path(packaging.__file__).parent
        core_version = __import__("importlib.metadata", fromlist=["version"]).version(
            "packaging"
        )

        target = tmp_path / "lazy-packages"
        monkeypatch.setenv(ld._LAZY_TARGET_ENV, str(target))
        # Install an old packaging into the target WITHOUT the core
        # constraints file (bypass the tidy resolver) so a shadow copy
        # genuinely exists on disk in the target — the worst case.
        ld._ensure_target_ready(target)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--target", str(target),
             "--no-deps", "packaging==20.9"],
            check=True, capture_output=True, text=True,
        )
        assert list(target.glob("packaging*")), "shadow copy should exist on disk"

        # Activate the target (append) and re-resolve.
        ld._activate_target_on_syspath(target)
        import importlib
        importlib.invalidate_caches()
        importlib.reload(packaging)
        # Core path + version must still win.
        assert Path(packaging.__file__).parent == core_path, (
            "durable-store copy shadowed the core module — append ordering broke"
        )
        new_version = __import__("importlib.metadata", fromlist=["version"]).version(
            "packaging"
        )
        assert new_version == core_version, (
            f"metadata resolved to shadow version {new_version}, expected core {core_version}"
        )
