"""Tests for tools.host_profiles."""

import pytest

from tools.host_profiles import get_host_profile, register_workstation_profile


class TestDefaultResolution:
    def test_none_resolves_to_nipogi(self):
        profile = get_host_profile(None)
        assert profile.name == "nipogi"
        assert profile.enabled is True

    def test_nipogi_explicit(self):
        profile = get_host_profile("nipogi")
        assert profile.enabled is True

    def test_local_alias(self):
        profile = get_host_profile("local")
        assert profile.enabled is True

    def test_unknown_profile_disabled_not_crashed(self):
        profile = get_host_profile("some-unknown-host")
        assert profile.enabled is False
        assert "unknown host profile" in profile.disabled_reason


class TestWorkstationDefaultDisabled:
    def test_workstation_disabled_by_default(self):
        profile = get_host_profile("workstation")
        assert profile.enabled is False
        assert "not enabled by default" in profile.disabled_reason.lower() or "not configured" in profile.disabled_reason.lower()

    def test_disabled_profile_run_raises(self):
        profile = get_host_profile("workstation")
        with pytest.raises(RuntimeError, match="disabled"):
            profile.run("ls", timeout=5)


class TestNipogiExecutesLocally:
    def test_local_run_executes(self):
        profile = get_host_profile("nipogi")
        result = profile.run("echo rob-host-profile-test", timeout=5)
        assert result.returncode == 0
        assert "rob-host-profile-test" in result.stdout


class TestWorkstationActivation:
    def test_register_enables_profile(self):
        register_workstation_profile("someuser@100.0.0.1")
        profile = get_host_profile("workstation")
        assert profile.enabled is True
        # Restore disabled state so this test doesn't leak into others —
        # module-level registry is process-global.
        from tools import host_profiles as hp

        hp._PROFILES["workstation"] = hp.HostProfile(
            name="workstation", enabled=False, disabled_reason="reset after test"
        )
