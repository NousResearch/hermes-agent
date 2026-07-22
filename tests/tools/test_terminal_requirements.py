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


def test_unknown_terminal_env_returns_false_silently(monkeypatch, caplog):
    _clear_terminal_env(monkeypatch)
    monkeypatch.setenv("TERMINAL_ENV", "unknown-backend")

    with caplog.at_level(logging.WARNING):
        ok = terminal_tool_module.check_terminal_requirements()

    assert ok is False
    warnings_and_errors = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings_and_errors == [], (
        f"check_terminal_requirements must be silent (no WARNING/ERROR logs), "
        f"but emitted: {[r.getMessage() for r in warnings_and_errors]}"
    )


def test_ssh_backend_without_host_or_user_returns_false_silently(monkeypatch, caplog):
    _clear_terminal_env(monkeypatch)
    monkeypatch.setenv("TERMINAL_ENV", "ssh")

    with caplog.at_level(logging.WARNING):
        ok = terminal_tool_module.check_terminal_requirements()

    assert ok is False
    warnings_and_errors = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings_and_errors == [], (
        f"check_terminal_requirements must be silent (no WARNING/ERROR logs), "
        f"but emitted: {[r.getMessage() for r in warnings_and_errors]}"
    )


def test_modal_backend_without_token_or_config_returns_false_silently(monkeypatch, caplog, tmp_path):
    _clear_terminal_env(monkeypatch)
    monkeypatch.setenv("TERMINAL_ENV", "modal")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setattr(terminal_tool_module, "is_managed_tool_gateway_ready", lambda _vendor: False)
    monkeypatch.setattr(terminal_tool_module.importlib.util, "find_spec", lambda _name: object())

    with caplog.at_level(logging.WARNING):
        ok = terminal_tool_module.check_terminal_requirements()

    assert ok is False
    warnings_and_errors = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings_and_errors == [], (
        f"check_terminal_requirements must be silent (no WARNING/ERROR logs), "
        f"but emitted: {[r.getMessage() for r in warnings_and_errors]}"
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

    with caplog.at_level(logging.WARNING):
        ok = terminal_tool_module.check_terminal_requirements()

    assert ok is False
    warnings_and_errors = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings_and_errors == [], (
        f"check_terminal_requirements must be silent (no WARNING/ERROR logs), "
        f"but emitted: {[r.getMessage() for r in warnings_and_errors]}"
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

    with caplog.at_level(logging.WARNING):
        ok = terminal_tool_module.check_terminal_requirements()

    assert ok is False
    warnings_and_errors = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings_and_errors == [], (
        f"check_terminal_requirements must be silent (no WARNING/ERROR logs), "
        f"but emitted: {[r.getMessage() for r in warnings_and_errors]}"
    )


def test_modal_backend_managed_mode_without_feature_flag_returns_false_silently(monkeypatch, caplog, tmp_path):
    _clear_terminal_env(monkeypatch)
    monkeypatch.setenv("TERMINAL_ENV", "modal")
    monkeypatch.setenv("TERMINAL_MODAL_MODE", "managed")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setattr(terminal_tool_module, "is_managed_tool_gateway_ready", lambda _vendor: False)

    with caplog.at_level(logging.WARNING):
        ok = terminal_tool_module.check_terminal_requirements()

    assert ok is False
    warnings_and_errors = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings_and_errors == [], (
        f"check_terminal_requirements must be silent (no WARNING/ERROR logs), "
        f"but emitted: {[r.getMessage() for r in warnings_and_errors]}"
    )
