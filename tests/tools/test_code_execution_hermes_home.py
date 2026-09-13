"""execute_code child env honors the multiplexed per-turn HERMES_HOME override (#110303).

Under a multiplexed Desktop/Dashboard connection one server process serves several
profiles, binding a context-local HERMES_HOME override per turn. ``_build_child_env``
scrubs the server process's ``os.environ`` — which carries the machine-default
HERMES_HOME — so without the rewrite below, skill scripts run via ``execute_code``
silently read/write the wrong profile's directory.
"""

import sys

import pytest

from hermes_constants import (
    get_hermes_home_override,
    reset_hermes_home_override,
    set_hermes_home_override,
)
from tools.code_execution_env import _build_child_env


@pytest.fixture
def home_override():
    tokens = []

    def _set(path):
        tokens.append(set_hermes_home_override(path))
        return str(path)

    yield _set
    for token in reversed(tokens):
        reset_hermes_home_override(token)


def _child_env():
    return _build_child_env(
        rpc_endpoint="sock",
        rpc_token="tok",
        tmpdir="/tmp/hermes-test",
        child_python=sys.executable,
    )


class TestMultiplexedHermesHome:
    def test_override_rewrites_stale_server_default(self, monkeypatch, home_override, tmp_path):
        """The reported bug: child must see the active profile's home, not the server default."""
        monkeypatch.setenv("HERMES_HOME", "/machine/default/.hermes")
        alpha_home = home_override(tmp_path / "profiles" / "alpha")

        assert _child_env()["HERMES_HOME"] == alpha_home

    def test_no_override_leaves_inherited_value_untouched(self, monkeypatch):
        """Dedicated per-profile processes (no override): zero behavior change."""
        assert get_hermes_home_override() is None
        monkeypatch.setenv("HERMES_HOME", "/machine/default/.hermes")

        assert _child_env()["HERMES_HOME"] == "/machine/default/.hermes"

    def test_override_applies_when_parent_env_unset(self, monkeypatch, home_override, tmp_path):
        monkeypatch.delenv("HERMES_HOME", raising=False)
        beta_home = home_override(tmp_path / "profiles" / "beta")

        assert _child_env()["HERMES_HOME"] == beta_home

    def test_override_does_not_leak_between_profiles(self, monkeypatch, home_override, tmp_path):
        """Sequential turns for different profiles each see their own home."""
        monkeypatch.setenv("HERMES_HOME", "/machine/default/.hermes")
        home_override(tmp_path / "profiles" / "alpha")
        first = _child_env()["HERMES_HOME"]

        home_override(tmp_path / "profiles" / "beta")
        second = _child_env()["HERMES_HOME"]

        assert first == str(tmp_path / "profiles" / "alpha")
        assert second == str(tmp_path / "profiles" / "beta")
