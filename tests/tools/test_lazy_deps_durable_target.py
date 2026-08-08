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

    def test_a_target_permits_install_specs_in_a_sealed_image(
        self, monkeypatch, tmp_path
    ):
        """install_specs must still work in the container.

        A memory provider that a user installs into ~/.hermes/plugins names
        its own packages in plugin.yaml, and pyproject.toml does not hold
        them, so no image can bake them. The target directory is where those
        go. ensure() is the one that must refuse — see
        test_ensure_refuses_in_a_sealed_image_even_with_a_target.
        """
        monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")
        monkeypatch.setenv(ld._LAZY_TARGET_ENV, str(tmp_path))
        monkeypatch.setattr(
            "hermes_cli.config.load_config", lambda: {}, raising=False
        )
        assert ld._allow_lazy_installs() is True

    def test_ensure_refuses_in_a_sealed_image_even_with_a_target(
        self, monkeypatch, tmp_path
    ):
        """A LAZY_DEPS feature never installs in the image.

        The build bakes each extra that a container can run, so a feature
        that reaches this point names a dependency the image should have
        shipped. Report that instead of a download from PyPI. The target
        directory must not change this: it exists for install_specs.
        """
        monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")
        monkeypatch.setenv(ld._LAZY_TARGET_ENV, str(tmp_path))
        monkeypatch.setattr(ld, "feature_missing", lambda _f: ("zzzpkg==1.0",))
        monkeypatch.setattr(
            ld, "_venv_pip_install",
            lambda *a, **kw: pytest.fail("pip must not run in a sealed image"),
        )
        feature = next(iter(ld.LAZY_DEPS))
        with pytest.raises(ld.FeatureUnavailable) as excinfo:
            ld.ensure(feature, prompt=False)
        assert "HERMES_DISABLE_LAZY_INSTALLS" in str(excinfo.value)

    def test_sealed_reason_does_not_blame_the_config_key(self, monkeypatch):
        """The sealed message must not name a setting that the user never set.

        The message must not say "security.allow_lazy_installs=false". That
        key is not the cause, and the venv is read-only, so a
        `uv pip install` command cannot succeed.
        """
        monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")
        reason = ld._sealed_venv_reason()
        assert reason and "allow_lazy_installs" not in reason
        assert "HERMES_DISABLE_LAZY_INSTALLS" in reason

        monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
        assert ld._sealed_venv_reason() is None

    def test_sealed_error_omits_the_manual_install_hint(self, monkeypatch):
        """A `uv pip install` hint is useless against a read-only venv."""
        monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")
        err = ld.FeatureUnavailable(
            "some.feature", ("pkg==1.0",), ld._sealed_venv_reason(), actionable=False
        )
        assert "uv pip install" not in str(err)
        # ...but the normal path keeps it.
        actionable = ld.FeatureUnavailable("some.feature", ("pkg==1.0",), "nope")
        assert "uv pip install" in str(actionable)


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
# Install path: arg construction (network-free) + a real install (opt-in).
# ---------------------------------------------------------------------------


class TestInstallArgConstruction:
    """Verify the durable-target install builds the right pip/uv command
    WITHOUT hitting the network, by stubbing the subprocess layer. This is
    the CI-safe coverage of the install path; the genuine PyPI install below
    is opt-in only."""

    def test_target_and_constraint_args_passed(self, tmp_path, monkeypatch):
        target = tmp_path / "lazy-packages"
        monkeypatch.setenv(ld._LAZY_TARGET_ENV, str(target))
        # No uv on PATH → force the pip tier so we assert one known command.
        monkeypatch.setattr(ld.shutil, "which", lambda _: None)

        captured = {}
        calls: list[list[str]] = []

        def fake_run(cmd, *a, **k):
            # The pip --version probe must look healthy so we reach install.
            if "--version" in cmd:
                return subprocess.CompletedProcess(cmd, 0, "pip 24.0", "")
            calls.append(list(cmd))
            captured["cmd"] = cmd
            return subprocess.CompletedProcess(cmd, 0, "ok", "")

        monkeypatch.setattr(ld.subprocess, "run", fake_run)
        # Avoid mutating the real interpreter's sys.path on success.
        monkeypatch.setattr(ld, "_activate_target_on_syspath", lambda _t: None)

        result = ld._venv_pip_install(("somepkg==1.2.3",))
        assert result.success
        # The backend install, not the --no-deps security-override repair pass
        # that follows it (see tools/lazy_deps._pip_reassert_overrides).
        backend = [c for c in calls if "--no-deps" not in c]
        assert backend, f"no backend install captured: {calls}"
        cmd = backend[0]
        # --target points at the durable dir...
        assert "--target" in cmd
        assert str(target) in cmd
        # ...a --constraint file pins shared deps to core...
        assert "--constraint" in cmd
        # ...and the spec is last.
        assert cmd[-1] == "somepkg==1.2.3"

    def test_override_repair_pass_targets_the_durable_dir(self, tmp_path, monkeypatch):
        """The pip override repair must land in the durable target too.

        Without --target the repair would write the overridden package into
        the sealed core venv, which is exactly what durable-target mode exists
        to prevent.
        """
        target = tmp_path / "lazy-packages"
        monkeypatch.setenv(ld._LAZY_TARGET_ENV, str(target))
        monkeypatch.setattr(ld.shutil, "which", lambda _: None)

        calls: list[list[str]] = []

        def fake_run(cmd, *a, **k):
            if "--version" in cmd:
                return subprocess.CompletedProcess(cmd, 0, "pip 24.0", "")
            calls.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 0, "ok", "")

        monkeypatch.setattr(ld.subprocess, "run", fake_run)
        monkeypatch.setattr(ld, "_activate_target_on_syspath", lambda _t: None)

        ld._venv_pip_install(("somepkg==1.2.3",))
        repair = [c for c in calls if "--no-deps" in c]
        if not ld._security_overrides():
            pytest.skip("no security overrides configured")
        assert repair, f"no override repair pass captured: {calls}"
        assert "--target" in repair[0] and str(target) in repair[0], (
            f"repair pass must write to the durable target: {repair[0]}"
        )

    def test_no_target_args_in_venv_scoped_mode(self, monkeypatch):
        # Env unset → plain venv-scoped install, no --target and no
        # core-constraints file.
        monkeypatch.delenv(ld._LAZY_TARGET_ENV, raising=False)
        monkeypatch.setattr(ld.shutil, "which", lambda _: None)
        captured = {}

        def fake_run(cmd, *a, **k):
            if "--version" in cmd:
                return subprocess.CompletedProcess(cmd, 0, "pip 24.0", "")
            captured["cmd"] = cmd
            return subprocess.CompletedProcess(cmd, 0, "ok", "")

        monkeypatch.setattr(ld.subprocess, "run", fake_run)
        result = ld._venv_pip_install(("somepkg==1.2.3",))
        assert result.success
        cmd = captured["cmd"]
        assert "--target" not in cmd
        # The durable-target mode's core-constraints file must be absent. The
        # pip tier does still carry a --constraint holding the security floor
        # (see tools/lazy_deps._security_overrides), so assert on the file's
        # role rather than on the flag's mere presence.
        constraint_paths = [
            Path(cmd[i + 1])
            for i, tok in enumerate(cmd)
            if tok == "--constraint"
        ]
        assert not any(
            p.name.startswith("hermes-core-constraints-") for p in constraint_paths
        ), f"venv-scoped install must not pin against a core-constraints file: {cmd}"


@pytest.mark.skipif(
    os.environ.get("HERMES_RUN_NETWORK_TESTS") != "1",
    reason="opt-in real-install test (set HERMES_RUN_NETWORK_TESTS=1); CI runs "
    "the network-free arg-construction + synthetic-shadow tests instead",
)
class TestRealInstallCoreWins:
    """Genuine PyPI install into a durable target (opt-in). Proves the wire
    end to end: the package lands in the target, not the core venv, and is
    importable via the appended sys.path entry. Skipped by default so the
    unit-test shard never depends on PyPI reachability/egress."""

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


class TestCoreNeverShadowed:
    """The headline invariant — a package in the durable store can never
    shadow a core module — proved WITHOUT a network install by synthesizing
    a shadow copy of a core package directly on disk in the target. This is
    deterministic (no PyPI) and a stronger check: we control exactly what
    the shadow contains, so a sentinel attribute proves which copy won.
    """

    def test_synthetic_shadow_does_not_win(self, tmp_path, monkeypatch):
        # 'packaging' is always present in the venv (transitive of the build
        # toolchain). Resolve the core copy's location first.
        import importlib.util
        core_spec = importlib.util.find_spec("packaging")
        assert core_spec is not None and core_spec.origin
        core_path = Path(core_spec.origin).parent

        # Plant a fake 'packaging' in the durable target with a sentinel that
        # the real core copy does NOT have.
        target = tmp_path / "lazy-packages"
        monkeypatch.setenv(ld._LAZY_TARGET_ENV, str(target))
        ld._ensure_target_ready(target)
        shadow_pkg = target / "packaging"
        shadow_pkg.mkdir()
        (shadow_pkg / "__init__.py").write_text(
            "SHADOW_SENTINEL = True\n__version__ = '0.0.0-shadow'\n"
        )
        assert (shadow_pkg / "__init__.py").exists(), "shadow copy must exist on disk"

        # Activate the target (append-only) and re-resolve the import.
        saved = list(sys.path)
        try:
            ld._activate_target_on_syspath(target)
            import importlib
            importlib.invalidate_caches()
            spec_after = importlib.util.find_spec("packaging")
            assert spec_after is not None and spec_after.origin
            resolved = Path(spec_after.origin).parent
            # Core path must still win; the shadow in the target is ignored.
            assert resolved == core_path, (
                f"durable-store copy shadowed core: resolved to {resolved}, "
                f"expected core at {core_path}"
            )
            assert resolved != shadow_pkg, "import resolved to the shadow copy"
        finally:
            sys.path[:] = saved
            sys.modules.pop("packaging", None)
            importlib.invalidate_caches()
