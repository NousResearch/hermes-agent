"""Boolean ``TERMINAL_*`` reads: shared truthy set, and a blank value means "unset"."""

import pytest

from tools.terminal_tool import _get_env_config
from tools.terminal_tool_config import _tenv_bool


@pytest.mark.parametrize("value", ["", "  "])
@pytest.mark.parametrize("default, expected", [("true", True), ("false", False)])
def test_blank_value_yields_default(monkeypatch, value, default, expected):
    monkeypatch.setenv("TERMINAL_TEST_FLAG", value)

    assert _tenv_bool("TERMINAL_TEST_FLAG", default) is expected


@pytest.mark.parametrize("value, expected", [
    ("on", True), (" TRUE ", True), ("1", True), ("yes", True),
    ("off", False), ("false", False), ("0", False), ("no", False),
])
def test_values_use_shared_truthy_set(monkeypatch, value, expected):
    monkeypatch.setenv("TERMINAL_TEST_FLAG", value)

    assert _tenv_bool("TERMINAL_TEST_FLAG", "true") is expected
    assert _tenv_bool("TERMINAL_TEST_FLAG", "false") is expected


def test_unset_yields_default(monkeypatch):
    monkeypatch.delenv("TERMINAL_TEST_FLAG", raising=False)

    assert _tenv_bool("TERMINAL_TEST_FLAG", "true") is True
    assert _tenv_bool("TERMINAL_TEST_FLAG", "false") is False


def test_blank_env_keeps_default_on_settings_on(monkeypatch):
    for key in ("TERMINAL_CONTAINER_PERSISTENT", "TERMINAL_DOCKER_NETWORK"):
        monkeypatch.setenv(key, "")

    config = _get_env_config()

    assert config["container_persistent"] is True
    assert config["docker_network"] is True
