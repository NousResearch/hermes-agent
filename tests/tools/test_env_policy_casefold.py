"""Regression tests for case-insensitive credential environment names.

Regression for #116149: Windows environment lookup treats names case-insensitively,
so every credential policy surface must reject casing variants consistently.
"""

import pytest

from tools.env_passthrough import (
    clear_env_passthrough,
    is_env_passthrough,
    register_env_passthrough,
)
from tools.environments.docker_egress import (
    check_forward_env_collisions,
    _extra_args_egress_collisions,
)
from tools.environments.local import _sanitize_subprocess_env
from tools.environments.remote_common import resolve_passthrough_env


@pytest.fixture(autouse=True)
def _clean_passthrough():
    clear_env_passthrough()
    yield
    clear_env_passthrough()


def test_case_variant_provider_names_are_rejected_and_scrubbed(monkeypatch):
    """Registration and local child scrubbing use the same folded policy."""
    monkeypatch.setenv("OPENAI_API_KEY", "redacted-provider-secret")
    register_env_passthrough(["openai_api_key"])

    assert not is_env_passthrough("openai_api_key")
    sanitized = _sanitize_subprocess_env({"openai_api_key": "redacted-provider-secret"})
    assert "openai_api_key" not in sanitized


def test_case_variant_provider_names_are_blocked_from_remote_and_docker_egress():
    """Remote forwarding and Docker collision guards reject casing variants."""
    values, unset = resolve_passthrough_env(
        explicit_forward=("openai_api_key",),
        hermes_env_loader=lambda: {"openai_api_key": "redacted-provider-secret"},
    )
    assert "openai_api_key" not in values
    assert "openai_api_key" not in unset

    with pytest.raises(RuntimeError, match="egress-protected"):
        check_forward_env_collisions(["openai_api_key"], {"OPENAI_API_KEY"}, True)
    assert "openai_api_key" in _extra_args_egress_collisions(
        ["--env", "openai_api_key=redacted-provider-secret"], {"OPENAI_API_KEY"})
