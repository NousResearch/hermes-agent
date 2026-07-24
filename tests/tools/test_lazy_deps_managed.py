"""Managed-install coverage for lazy_deps (issue #48628)."""

from __future__ import annotations

import subprocess

import pytest

from tools import lazy_deps as ld


class TestManagedPipBootstrap:
    """Managed installs skip only ensurepip, while preserving useful errors."""

    def test_managed_install_still_reaches_installer(self, monkeypatch):
        checks = iter([("anthropic==0.87.0",), ()])
        monkeypatch.setattr(ld, "feature_missing", lambda _feature: next(checks))
        monkeypatch.setattr(ld, "_allow_lazy_installs", lambda: True)
        calls = []

        def fake_install(specs, **kwargs):
            calls.append((specs, kwargs))
            return ld._InstallResult(True, "ok", "")

        monkeypatch.setattr(ld, "_venv_pip_install", fake_install)
        monkeypatch.setenv("HERMES_MANAGED", "nixos")

        ld.ensure("provider.anthropic", prompt=False)

        assert calls == [
            (("anthropic==0.87.0",), {"feature": "provider.anthropic"})
        ]

    def test_nixos_skips_ensurepip_and_names_dependency_group(self, monkeypatch):
        monkeypatch.setattr(ld.shutil, "which", lambda _name: None)
        monkeypatch.setenv("HERMES_MANAGED", "nixos")
        calls = []

        def fake_run(cmd, *args, **kwargs):
            calls.append(cmd)
            if cmd[-1] == "--version":
                return subprocess.CompletedProcess(cmd, 1, "", "No module named pip")
            pytest.fail(f"managed install must not run bootstrap/install: {cmd}")

        monkeypatch.setattr(ld.subprocess, "run", fake_run)

        result = ld._venv_pip_install(
            ("anthropic==0.87.0",), feature="provider.anthropic"
        )

        assert not result.success
        assert "ensurepip bootstrap was skipped" in result.stderr
        assert 'extraDependencyGroups = [ "anthropic" ]' in result.remediation
        assert len(calls) == 1

    def test_ensure_surfaces_nix_remediation_without_pip_hint(self, monkeypatch):
        monkeypatch.setattr(
            ld,
            "feature_missing",
            lambda _feature: ("anthropic==0.87.0",),
        )
        monkeypatch.setattr(ld, "_allow_lazy_installs", lambda: True)
        monkeypatch.setattr(ld.shutil, "which", lambda _name: None)
        monkeypatch.setenv("HERMES_MANAGED", "true")

        def fake_run(cmd, *args, **kwargs):
            if cmd[-1] == "--version":
                return subprocess.CompletedProcess(cmd, 1, "", "No module named pip")
            pytest.fail(f"managed install must not run bootstrap/install: {cmd}")

        monkeypatch.setattr(ld.subprocess, "run", fake_run)

        with pytest.raises(ld.FeatureUnavailable) as exc_info:
            ld.ensure("provider.anthropic", prompt=False)

        message = str(exc_info.value)
        assert 'extraDependencyGroups = [ "anthropic" ]' in message
        assert "sudo nixos-rebuild switch" in message
        assert "uv pip install" not in message

    def test_readonly_nix_prefix_detected_without_managed_env(self, monkeypatch):
        monkeypatch.delenv("HERMES_MANAGED", raising=False)
        monkeypatch.setattr(
            "hermes_cli.config.get_managed_system", lambda: None
        )
        monkeypatch.setattr(ld.sys, "prefix", "/nix/store/example-hermes-venv")
        monkeypatch.setattr(ld.os, "access", lambda _path, _mode: False)

        assert ld._managed_install_system() == "NixOS"

    def test_writable_unmanaged_prefix_still_bootstraps(self, monkeypatch):
        monkeypatch.setattr(ld.shutil, "which", lambda _name: None)
        monkeypatch.setattr(
            "hermes_cli.config.get_managed_system", lambda: None
        )
        monkeypatch.setattr(ld.os, "access", lambda _path, _mode: True)
        calls = []

        def fake_run(cmd, *args, **kwargs):
            calls.append(cmd)
            if cmd[-1] == "--version":
                return subprocess.CompletedProcess(cmd, 1, "", "No module named pip")
            return subprocess.CompletedProcess(cmd, 0, "ok", "")

        monkeypatch.setattr(ld.subprocess, "run", fake_run)

        result = ld._venv_pip_install(("some-pkg==1.0",), feature="test.feature")

        assert result.success
        assert any("ensurepip" in cmd for cmd in calls)
        assert any("install" in cmd and "some-pkg==1.0" in cmd for cmd in calls)

    def test_homebrew_bootstrap_error_uses_package_manager(self, monkeypatch):
        monkeypatch.setattr(ld.shutil, "which", lambda _name: None)
        monkeypatch.setenv("HERMES_MANAGED", "homebrew")

        def fake_run(cmd, *args, **kwargs):
            if cmd[-1] == "--version":
                return subprocess.CompletedProcess(cmd, 1, "", "No module named pip")
            pytest.fail(f"managed install must not run bootstrap/install: {cmd}")

        monkeypatch.setattr(ld.subprocess, "run", fake_run)

        result = ld._venv_pip_install(
            ("anthropic==0.87.0",), feature="provider.anthropic"
        )

        assert not result.success
        assert "Homebrew" in result.stderr
        assert "Homebrew package/formula" in result.remediation
