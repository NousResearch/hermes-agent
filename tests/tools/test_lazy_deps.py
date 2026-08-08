"""Behaviour of tools/lazy_deps.py that is not pyproject.toml.

The specs come from the extras now, and uv.lock pins them, so nothing here
restates a package or a version. What is left is the code around the lookup:

* the allowlist — only a key in LAZY_DEPS may install
* the gate — security.allow_lazy_installs and the sealed-image flag
* ensure() — no-op when satisfied, and a clear error when pip lies
* active_features / refresh_active_features — the `hermes update` pass
* install_specs — the path for a package that no extra can hold

tests/tools/test_lazy_deps_extras_mapping.py covers the map and the reader.
"""
from __future__ import annotations


import pytest

import tools.lazy_deps as ld


def _register_fake_feature(monkeypatch, feature: str, specs: tuple[str, ...]) -> str:
    """Register a synthetic feature + backing extra for a test.

    Specs live in pyproject.toml's ``[project.optional-dependencies]``, so a
    test feature needs both halves: an entry in ``LAZY_DEPS`` mapping it to
    an extra name, and that extra in the (cached) pyproject table. Returns the
    generated extra name.
    """
    extra = f"__test-{feature.replace('.', '-')}"
    monkeypatch.setitem(ld.LAZY_DEPS, feature, extra)
    table = dict(ld._optional_dependencies())
    table[extra] = tuple(specs)
    monkeypatch.setattr(ld, "_optional_dependencies", lambda: table)
    return extra


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
        _register_fake_feature(monkeypatch, "test.feat", ("packageX>=1.0,<2",))
        monkeypatch.setattr(ld, "_is_satisfied", lambda spec: False)
        monkeypatch.setattr(ld, "_allow_lazy_installs", lambda: False)
        monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
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
        _register_fake_feature(monkeypatch, "test.satisfied", ("zzzfake>=1",))
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
        _register_fake_feature(monkeypatch, "test.cache", ("zzzfake>=1",))
        monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
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
        _register_fake_feature(monkeypatch, "test.miss", ("zzzfake>=1",))
        monkeypatch.setattr(ld, "_is_satisfied", lambda spec: False)
        assert ld.is_available("test.miss") is False


class TestMarkersDecideWhetherThereIsWork:
    """A spec for another platform must not become an install attempt.

    [wake-tflite] pins ai-edge-litert for macOS. On Linux there is no such
    wheel, so an install gets an error and not a package.
    """

    def test_a_spec_for_another_platform_counts_as_satisfied(self):
        assert ld._is_satisfied("zzzfake==1.0; sys_platform == 'nonesuch'")

    def test_a_spec_for_this_platform_still_installs(self, monkeypatch):
        import sys as _sys

        here = f"zzzfake==1.0; sys_platform == '{_sys.platform}'"
        assert ld._is_satisfied(here) is False

    def test_the_feature_is_available_when_its_marker_is_false(
        self, monkeypatch
    ):
        _register_fake_feature(
            monkeypatch, "test.elsewhere",
            ("zzzfake==1.0; sys_platform == 'nonesuch'",),
        )
        assert ld.feature_missing("test.elsewhere") == ()
        assert ld.is_available("test.elsewhere") is True


# ---------------------------------------------------------------------------
# active_features + refresh_active_features (Piece A — hermes update wiring)
# ---------------------------------------------------------------------------


class TestActiveFeatures:
    def test_no_packages_installed_returns_empty(self, monkeypatch):
        monkeypatch.setattr(ld, "_is_present", lambda spec: False)
        assert ld.active_features() == []


    def test_shared_dependency_does_not_activate_feature(self, monkeypatch):
        # asyncpg is a generic dependency that may be installed for unrelated
        # reasons. Even with Matrix in the record (used once, then removed),
        # asyncpg's presence must not stand in for the Matrix anchor
        # (mautrix) on hermes update.
        ld._write_feature_record({"platform.matrix"})
        monkeypatch.setattr(
            ld, "_is_present",
            lambda spec: ld._pkg_name_from_spec(spec) == "asyncpg",
        )
        assert "platform.matrix" not in ld.active_features()

    def test_a_composed_helper_does_not_activate_its_siblings(self, monkeypatch):
        """sounddevice is in every audio extra, via [audio-io].

        The regression: extra_specs expands references first, so specs[0]
        of [voice] and of each wake engine was sounddevice, and one local
        STT install marked all of them active. `hermes update` then
        installed ~500MB of wake engines the user never asked for.

        Recording every feature makes the point sharper: even with each
        audio feature in the record, only the one whose anchor is installed
        counts as active.
        """
        ld._write_feature_record(
            {"stt.faster_whisper", "wake.openwakeword", "wake.sherpa",
             "wake.porcupine"}
        )
        monkeypatch.setattr(
            ld, "_is_present",
            lambda spec: ld._pkg_name_from_spec(spec) in {
                "sounddevice", "numpy", "faster-whisper",
            },
        )
        active = ld.active_features()
        assert "stt.faster_whisper" in active
        assert [f for f in active if f.startswith("wake.")] == []

    def test_ensure_records_the_feature(self, monkeypatch):
        """A satisfied ensure() must land the feature in the record file.

        The record is the primary signal: it says which backends the user
        runs, where package presence can only say which packages exist.
        """
        _register_fake_feature(monkeypatch, "test.recorded", ("pkgx==1.0",))
        monkeypatch.setattr(ld, "_is_satisfied", lambda spec: True)
        ld.ensure("test.recorded", prompt=False)
        assert "test.recorded" in ld._read_feature_record()

    def test_a_recorded_feature_needs_its_anchor_installed(self, monkeypatch):
        """The record alone must not resurrect an uninstalled backend."""
        _register_fake_feature(monkeypatch, "test.gone", ("pkgy==1.0",))
        ld._write_feature_record({"test.gone"})
        monkeypatch.setattr(ld, "_is_present", lambda spec: False)
        assert "test.gone" not in ld.active_features()

    def test_an_absent_record_means_nothing_is_active(self, monkeypatch):
        """No seeding. An install that predates the record refreshes nothing
        on its first update; ensure() at backend start repairs stale pins
        and records the feature, so the next update covers it.
        """
        monkeypatch.setattr(
            ld, "_is_present",
            lambda spec: ld._pkg_name_from_spec(spec) == "mautrix",
        )
        assert not ld._feature_record_path().exists()
        assert ld.active_features() == []
        # Reading must not create the file either.
        assert not ld._feature_record_path().exists()

    def test_a_corrupt_record_counts_as_empty(self, monkeypatch):
        ld._feature_record_path().parent.mkdir(parents=True, exist_ok=True)
        ld._feature_record_path().write_text("not json", encoding="utf-8")
        monkeypatch.setattr(
            ld, "_is_present",
            lambda spec: ld._pkg_name_from_spec(spec) == "mautrix",
        )
        assert ld.active_features() == []


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


    def test_mixed_results_returns_per_feature_status(self, monkeypatch):
        monkeypatch.setattr(ld, "active_features", lambda: ["a.ok", "b.fail"])
        _register_fake_feature(monkeypatch, "a.ok", ("pkga==1.0",))
        _register_fake_feature(monkeypatch, "b.fail", ("pkgb==1.0",))
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

    def test_the_sealed_gate_runs_before_the_installer(self, monkeypatch):
        """A sealed deployment must stop the install, whatever the specs are."""
        monkeypatch.setattr(ld, "_allow_lazy_installs", lambda: False)
        monkeypatch.setattr(
            ld, "_venv_pip_install",
            lambda *a, **kw: pytest.fail("pip should not be called"),
        )
        result = ld.install_specs(["zzzpkg==1.0"])
        assert result.ok is False
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


class TestUvSyncTier:
    def test_uv_sync_names_the_project_directory(self, monkeypatch, tmp_path):
        """`uv sync` discovers the project from cwd, and the agent's cwd is
        the user's working directory — not the install tree. Without
        --project the sync errors out of the wrong directory on every real
        deployment and the tier silently never fires.
        """
        root = tmp_path / "checkout"
        root.mkdir()
        (root / "pyproject.toml").write_text("[project]\nname='x'\n")
        (root / "uv.lock").write_text("")
        monkeypatch.setattr(ld, "_project_root", lambda: root)
        monkeypatch.delenv(ld._LAZY_TARGET_ENV, raising=False)
        monkeypatch.setattr(ld.shutil, "which", lambda _n: "/usr/bin/uv")

        seen = {}

        def fake_run(cmd, *a, **kw):
            seen["cmd"] = list(cmd)

            class R:
                returncode = 0
                stdout = ""
                stderr = ""

            return R()

        monkeypatch.setattr(ld.subprocess, "run", fake_run)
        result = ld._uv_sync_extra("provider.anthropic")
        assert result is not None and result.success
        cmd = seen["cmd"]
        assert "--project" in cmd, f"uv sync must name the project: {cmd}"
        assert cmd[cmd.index("--project") + 1] == str(root)


class TestInstallSpecsManagedGuard:
    def test_a_managed_install_is_blocked_with_the_real_reason(self, monkeypatch):
        """A Nix install's venv is in the read-only store. The pip ladder
        can only burn time and surface EROFS — report the Nix remedy."""
        monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
        monkeypatch.delenv(ld._LAZY_TARGET_ENV, raising=False)
        monkeypatch.setattr(ld, "_managed_system", lambda: "NixOS")
        monkeypatch.setattr(
            ld, "_venv_pip_install",
            lambda *a, **kw: pytest.fail("must not attempt a store write"),
        )
        result = ld.install_specs(["some-plugin-sdk==1.0"])
        assert result.blocked is True
        assert "extraDependencyGroups" in result.reason

    def test_a_durable_target_overrides_the_managed_guard(self, monkeypatch, tmp_path):
        """The NixOS container module sets HERMES_MANAGED=true AND a
        writable target; install_specs must still work there."""
        monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
        monkeypatch.setenv(ld._LAZY_TARGET_ENV, str(tmp_path))
        monkeypatch.setattr(ld, "_managed_system", lambda: "NixOS")
        monkeypatch.setattr(
            ld, "_venv_pip_install",
            lambda *a, **kw: ld._InstallResult(True, "ok", ""),
        )
        result = ld.install_specs(["some-plugin-sdk==1.0"])
        assert result.ok is True
