"""Driver overrides belong to the active profile, not the gateway's launch env."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent import secret_scope
from tools.computer_use.cua_backend_driver import resolve_cua_driver_cmd


@pytest.fixture
def driver(tmp_path):
    binary = tmp_path / "fake-driver.exe"
    binary.write_bytes(b"unused: resolution only\n")
    binary.chmod(0o755)
    return str(binary)


@pytest.mark.parametrize("multiplex", [False, True], ids=["routed", "multiplex"])
@pytest.mark.parametrize("launch_override", [None, "/launch-profile/driver"])
def test_resolver_uses_profile_override(driver, tmp_path, monkeypatch, multiplex, launch_override):
    import pm

    monkeypatch.delenv("HERMES_CUA_DRIVER_CMD", raising=False)
    if launch_override is not None:
        monkeypatch.setenv("HERMES_CUA_DRIVER_CMD", launch_override)
    installed = Mock(return_value=SimpleNamespace(binary="/pm-managed/cua-driver"))
    monkeypatch.setattr(pm, "installed_package", installed)
    mode = secret_scope.set_multiplex_context(multiplex)
    scope = secret_scope.set_secret_scope(
        {"HERMES_CUA_DRIVER_CMD": driver}, profile_home=str(tmp_path / "routed-profile"))
    try:
        assert secret_scope.serves_routed_profile()
        assert resolve_cua_driver_cmd() == driver
        installed.assert_not_called()
    finally:
        secret_scope.reset_secret_scope(scope)
        secret_scope.reset_multiplex_context(mode)


@pytest.mark.parametrize("multiplex", [False, True], ids=["env-only", "unscoped-probe"])
def test_resolver_preserves_unscoped_env_override(driver, monkeypatch, multiplex):
    monkeypatch.setenv("HERMES_CUA_DRIVER_CMD", driver)
    scope = secret_scope.set_secret_scope(None)
    mode = secret_scope.set_multiplex_context(multiplex)
    try:
        assert secret_scope.is_multiplex_active() == multiplex
        assert resolve_cua_driver_cmd() == driver
    finally:
        secret_scope.reset_multiplex_context(mode)
        secret_scope.reset_secret_scope(scope)


@pytest.mark.parametrize("configured", [None, "", "missing-driver"])
def test_scoped_miss_never_borrows_launch_override(driver, tmp_path, monkeypatch, configured):
    import pm

    monkeypatch.setenv("HERMES_CUA_DRIVER_CMD", driver)
    installed = Mock(return_value=SimpleNamespace(binary="/pm-managed/cua-driver"))
    monkeypatch.setattr(pm, "installed_package", installed)
    mapping = {} if configured is None else {"HERMES_CUA_DRIVER_CMD": configured}
    mode = secret_scope.set_multiplex_context(True)
    scope = secret_scope.set_secret_scope(mapping, profile_home=str(tmp_path / "routed-profile"))
    try:
        assert resolve_cua_driver_cmd() == (None if configured else "/pm-managed/cua-driver")
        if configured:
            installed.assert_not_called()
    finally:
        secret_scope.reset_secret_scope(scope)
        secret_scope.reset_multiplex_context(mode)


@pytest.mark.parametrize("ready", [False, True], ids=["invalid-contract", "ready"])
def test_start_does_not_acquire_profile_override(driver, tmp_path, monkeypatch, ready):
    import pm
    from tools.computer_use import cua_backend

    monkeypatch.delenv("HERMES_CUA_DRIVER_CMD", raising=False)
    acquire = Mock()
    monkeypatch.setattr(pm, "ensure", acquire)
    monkeypatch.setattr(pm, "ensure_import", lambda name: None)
    monkeypatch.setattr(cua_backend, "sandbox_mcp_invocation", lambda: None)
    monkeypatch.setattr(cua_backend, "cua_driver_runtime_contract_status", lambda: {
        "ready": ready, "reason": "invalid test manifest"})
    backend = cua_backend.CuaDriverBackend()
    started = Mock()
    monkeypatch.setattr(backend._session, "start", started)
    monkeypatch.setattr(backend._session, "call_tool", lambda *args: None)
    scope = secret_scope.set_secret_scope(
        {"HERMES_CUA_DRIVER_CMD": driver}, profile_home=str(tmp_path / "routed-profile"))
    try:
        if ready:
            backend.start()
            started.assert_called_once()
        else:
            with pytest.raises(RuntimeError, match="Update the binary selected by HERMES_CUA_DRIVER_CMD"):
                backend.start()
            started.assert_not_called()
        acquire.assert_not_called()
    finally:
        secret_scope.reset_secret_scope(scope)
