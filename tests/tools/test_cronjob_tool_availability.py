"""Regression tests for cronjob availability across gateway context changes."""

import pytest

import model_tools


_AVAILABILITY_ENV = (
    "HERMES_INTERACTIVE",
    "HERMES_GATEWAY_SESSION",
    "HERMES_EXEC_ASK",
    "HERMES_SESSION_PLATFORM",
)


@pytest.fixture(autouse=True)
def _isolated_availability_context(monkeypatch):
    """Keep each assertion independent of process and ContextVar state."""
    from gateway.session_context import reset_session_vars
    from tools.registry import invalidate_check_fn_cache

    for name in _AVAILABILITY_ENV:
        monkeypatch.delenv(name, raising=False)
    reset_session_vars()
    model_tools._clear_tool_defs_cache()
    invalidate_check_fn_cache()
    yield
    reset_session_vars()
    model_tools._clear_tool_defs_cache()
    invalidate_check_fn_cache()


def _cronjob_names() -> set[str]:
    return {
        tool["function"]["name"]
        for tool in model_tools.get_tool_definitions(
            enabled_toolsets=["cronjob"],
            quiet_mode=True,
            skip_tool_search_assembly=True,
        )
    }


def test_cached_missing_cronjob_refreshes_after_gateway_env(monkeypatch):
    """A pre-gateway lookup cannot hide cronjob after approval mode appears."""
    assert "cronjob" not in _cronjob_names()

    monkeypatch.setenv("HERMES_EXEC_ASK", "1")

    assert "cronjob" in _cronjob_names()


def test_cached_missing_cronjob_refreshes_after_gateway_session():
    """Task-local gateway identity enables cronjob without a global env flag."""
    from gateway.session_context import clear_session_vars, set_session_vars

    assert "cronjob" not in _cronjob_names()

    tokens = set_session_vars(platform="matrix", chat_id="!room:example.org")
    try:
        assert "cronjob" in _cronjob_names()
    finally:
        clear_session_vars(tokens)
