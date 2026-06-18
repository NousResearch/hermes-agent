import importlib
import logging


terminal_tool_module = importlib.import_module("tools.terminal_tool")


def _clear_terminal_env(monkeypatch):
    """Remove terminal env vars that could affect requirements checks."""
    keys = [
        "TERMINAL_ENV",
        "TERMINAL_CONTAINER_CPU",
        "TERMINAL_CONTAINER_DISK",
        "TERMINAL_CONTAINER_MEMORY",
        "TERMINAL_DOCKER_FORWARD_ENV",
        "TERMINAL_DOCKER_VOLUMES",
        "TERMINAL_LIFETIME_SECONDS",
        "TERMINAL_MODAL_MODE",
        "TERMINAL_SSH_HOST",
        "TERMINAL_SSH_PORT",
        "TERMINAL_SSH_USER",
        "TERMINAL_TIMEOUT",
        "MODAL_TOKEN_ID",
        "MODAL_TOKEN_SECRET",
        "BL_API_KEY",
        "BL_WORKSPACE",
        "BL_REGION",
        "HOME",
        "USERPROFILE",
    ]
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    # Default: no Nous subscription — patch both the terminal_tool local
    # binding and tool_backend_helpers (used by resolve_modal_backend_state).
    monkeypatch.setattr(terminal_tool_module, "managed_nous_tools_enabled", lambda: False)
    import tools.tool_backend_helpers as _tbh
    monkeypatch.setattr(_tbh, "managed_nous_tools_enabled", lambda: False)


def test_local_terminal_requirements(monkeypatch, caplog):
    """Local backend uses Hermes' own LocalEnvironment wrapper."""
    _clear_terminal_env(monkeypatch)
    monkeypatch.setenv("TERMINAL_ENV", "local")

    with caplog.at_level(logging.ERROR):
        ok = terminal_tool_module.check_terminal_requirements()

    assert ok is True
    assert "Terminal requirements check failed" not in caplog.text


def test_unknown_terminal_env_logs_error_and_returns_false(monkeypatch, caplog):
    _clear_terminal_env(monkeypatch)
    monkeypatch.setenv("TERMINAL_ENV", "unknown-backend")

    with caplog.at_level(logging.ERROR):
        ok = terminal_tool_module.check_terminal_requirements()

    assert ok is False
    assert any(
        "Unknown TERMINAL_ENV 'unknown-backend'" in record.getMessage()
        for record in caplog.records
    )


def test_ssh_backend_without_host_or_user_logs_and_returns_false(monkeypatch, caplog):
    _clear_terminal_env(monkeypatch)
    monkeypatch.setenv("TERMINAL_ENV", "ssh")

    with caplog.at_level(logging.ERROR):
        ok = terminal_tool_module.check_terminal_requirements()

    assert ok is False
    assert any(
        "SSH backend selected but TERMINAL_SSH_HOST and TERMINAL_SSH_USER" in record.getMessage()
        for record in caplog.records
    )


def test_modal_backend_without_token_or_config_logs_specific_error(monkeypatch, caplog, tmp_path):
    _clear_terminal_env(monkeypatch)
    monkeypatch.setenv("TERMINAL_ENV", "modal")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setattr(terminal_tool_module, "is_managed_tool_gateway_ready", lambda _vendor: False)
    monkeypatch.setattr(terminal_tool_module.importlib.util, "find_spec", lambda _name: object())

    with caplog.at_level(logging.ERROR):
        ok = terminal_tool_module.check_terminal_requirements()

    assert ok is False
    assert any(
        "Modal backend selected but no direct Modal credentials/config was found" in record.getMessage()
        for record in caplog.records
    )


def test_modal_backend_with_managed_gateway_does_not_require_direct_creds_or_minisweagent(monkeypatch, tmp_path):
    _clear_terminal_env(monkeypatch)
    monkeypatch.setattr(terminal_tool_module, "managed_nous_tools_enabled", lambda: True)
    import tools.tool_backend_helpers as _tbh
    monkeypatch.setattr(_tbh, "managed_nous_tools_enabled", lambda: True)
    monkeypatch.setenv("TERMINAL_ENV", "modal")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("TERMINAL_MODAL_MODE", "managed")
    monkeypatch.setattr(terminal_tool_module, "is_managed_tool_gateway_ready", lambda _vendor: True)
    monkeypatch.setattr(
        terminal_tool_module.importlib.util,
        "find_spec",
        lambda _name: (_ for _ in ()).throw(AssertionError("should not be called")),
    )

    assert terminal_tool_module.check_terminal_requirements() is True


def test_modal_backend_auto_mode_prefers_managed_gateway_over_direct_creds(monkeypatch, tmp_path):
    _clear_terminal_env(monkeypatch)
    monkeypatch.setattr(terminal_tool_module, "managed_nous_tools_enabled", lambda: True)
    import tools.tool_backend_helpers as _tbh
    monkeypatch.setattr(_tbh, "managed_nous_tools_enabled", lambda: True)
    monkeypatch.setenv("TERMINAL_ENV", "modal")
    monkeypatch.setenv("MODAL_TOKEN_ID", "tok-id")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "tok-secret")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setattr(terminal_tool_module, "is_managed_tool_gateway_ready", lambda _vendor: True)
    monkeypatch.setattr(
        terminal_tool_module.importlib.util,
        "find_spec",
        lambda _name: (_ for _ in ()).throw(AssertionError("should not be called")),
    )

    assert terminal_tool_module.check_terminal_requirements() is True


def test_modal_backend_direct_mode_does_not_fall_back_to_managed(monkeypatch, caplog, tmp_path):
    _clear_terminal_env(monkeypatch)
    monkeypatch.setenv("TERMINAL_ENV", "modal")
    monkeypatch.setenv("TERMINAL_MODAL_MODE", "direct")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setattr(terminal_tool_module, "is_managed_tool_gateway_ready", lambda _vendor: True)

    with caplog.at_level(logging.ERROR):
        ok = terminal_tool_module.check_terminal_requirements()

    assert ok is False
    assert any(
        "TERMINAL_MODAL_MODE=direct" in record.getMessage()
        for record in caplog.records
    )


def test_modal_backend_managed_mode_does_not_fall_back_to_direct(monkeypatch, caplog, tmp_path):
    _clear_terminal_env(monkeypatch)
    monkeypatch.setenv("TERMINAL_ENV", "modal")
    monkeypatch.setenv("TERMINAL_MODAL_MODE", "managed")
    monkeypatch.setenv("MODAL_TOKEN_ID", "tok-id")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "tok-secret")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setattr(terminal_tool_module, "is_managed_tool_gateway_ready", lambda _vendor: False)

    with caplog.at_level(logging.ERROR):
        ok = terminal_tool_module.check_terminal_requirements()

    assert ok is False
    assert any(
        "Nous Tool Gateway access is not currently available" in record.getMessage()
        for record in caplog.records
    )


def test_modal_backend_managed_mode_without_feature_flag_logs_clear_error(monkeypatch, caplog, tmp_path):
    _clear_terminal_env(monkeypatch)
    monkeypatch.setenv("TERMINAL_ENV", "modal")
    monkeypatch.setenv("TERMINAL_MODAL_MODE", "managed")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setattr(terminal_tool_module, "is_managed_tool_gateway_ready", lambda _vendor: False)

    with caplog.at_level(logging.ERROR):
        ok = terminal_tool_module.check_terminal_requirements()

    assert ok is False
    assert any(
        "Nous Tool Gateway access is not currently available" in record.getMessage()
        for record in caplog.records
    )


def test_blaxel_backend_without_api_key_logs_specific_error(monkeypatch, caplog):
    _clear_terminal_env(monkeypatch)
    monkeypatch.setenv("TERMINAL_ENV", "blaxel")
    monkeypatch.setenv("BL_WORKSPACE", "workspace")
    monkeypatch.setattr(terminal_tool_module.importlib.util, "find_spec", lambda _name: object())

    with caplog.at_level(logging.ERROR):
        ok = terminal_tool_module.check_terminal_requirements()

    assert ok is False
    assert any(
        "BL_API_KEY is not set" in record.getMessage()
        for record in caplog.records
    )


def test_blaxel_backend_without_workspace_logs_specific_error(monkeypatch, caplog):
    _clear_terminal_env(monkeypatch)
    monkeypatch.setenv("TERMINAL_ENV", "blaxel")
    monkeypatch.setenv("BL_API_KEY", "key")
    monkeypatch.setattr(terminal_tool_module.importlib.util, "find_spec", lambda _name: object())

    with caplog.at_level(logging.ERROR):
        ok = terminal_tool_module.check_terminal_requirements()

    assert ok is False
    assert any(
        "BL_WORKSPACE is not set" in record.getMessage()
        for record in caplog.records
    )


def test_blaxel_backend_accepts_auth_and_sdk(monkeypatch):
    _clear_terminal_env(monkeypatch)
    monkeypatch.setenv("TERMINAL_ENV", "blaxel")
    monkeypatch.setenv("BL_API_KEY", "key")
    monkeypatch.setenv("BL_WORKSPACE", "workspace")
    monkeypatch.setattr(terminal_tool_module.importlib.util, "find_spec", lambda _name: object())

    assert terminal_tool_module.check_terminal_requirements() is True


def test_blaxel_backend_invalidates_import_caches_after_lazy_install(monkeypatch):
    _clear_terminal_env(monkeypatch)
    monkeypatch.setenv("TERMINAL_ENV", "blaxel")
    monkeypatch.setenv("BL_API_KEY", "key")
    monkeypatch.setenv("BL_WORKSPACE", "workspace")

    find_results = iter([None, object()])

    def fake_find_spec(name):
        assert name == "blaxel"
        return next(find_results)

    invalidated = []
    monkeypatch.setattr(terminal_tool_module.importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(
        terminal_tool_module.importlib,
        "invalidate_caches",
        lambda: invalidated.append(True),
    )
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *args, **kwargs: None)

    assert terminal_tool_module.check_terminal_requirements() is True
    assert invalidated == [True]


def test_blaxel_backend_defaults_to_4gb_memory(monkeypatch):
    _clear_terminal_env(monkeypatch)
    monkeypatch.setenv("TERMINAL_ENV", "blaxel")

    assert terminal_tool_module._get_env_config()["container_memory"] == 4096


def test_blaxel_backend_defaults_to_10gb_volume(monkeypatch):
    _clear_terminal_env(monkeypatch)
    monkeypatch.setenv("TERMINAL_ENV", "blaxel")

    assert terminal_tool_module._get_env_config()["container_disk"] == 10240


def test_blaxel_backend_respects_memory_override(monkeypatch):
    _clear_terminal_env(monkeypatch)
    monkeypatch.setenv("TERMINAL_ENV", "blaxel")
    monkeypatch.setenv("TERMINAL_CONTAINER_MEMORY", "2048")

    assert terminal_tool_module._get_env_config()["container_memory"] == 2048


def test_blaxel_create_environment_preserves_fractional_cpu(monkeypatch):
    captured = {}

    class FakeBlaxelEnvironment:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import tools.environments.blaxel as blaxel_mod

    monkeypatch.setattr(blaxel_mod, "BlaxelEnvironment", FakeBlaxelEnvironment)

    terminal_tool_module._create_environment(
        "blaxel",
        "blaxel/base-image:latest",
        "/blaxel",
        60,
        container_config={
            "container_cpu": 0.5,
            "container_memory": 4096,
            "container_disk": 10240,
            "container_persistent": True,
        },
        task_id="fractional-cpu",
    )

    assert captured["cpu"] == 0.5


def test_blaxel_backend_without_sdk_reports_lazy_install_failure(
    monkeypatch, caplog,
):
    _clear_terminal_env(monkeypatch)
    monkeypatch.setenv("TERMINAL_ENV", "blaxel")
    monkeypatch.setenv("BL_API_KEY", "key")
    monkeypatch.setenv("BL_WORKSPACE", "workspace")
    monkeypatch.setattr(terminal_tool_module.importlib.util, "find_spec", lambda _name: None)

    def fail_lazy_install(_feature, prompt=False):
        raise RuntimeError("offline")

    monkeypatch.setattr("tools.lazy_deps.ensure", fail_lazy_install)

    with caplog.at_level(logging.ERROR):
        ok = terminal_tool_module.check_terminal_requirements()

    assert ok is False
    assert any(
        "blaxel is required for the Blaxel terminal backend" in record.getMessage()
        for record in caplog.records
    )
