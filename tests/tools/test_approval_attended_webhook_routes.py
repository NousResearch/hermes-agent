"""approvals.attended_webhook_routes: webhook sessions from a listed route keep the normal
gateway approval flow instead of being resolved by approvals.unattended_mode.

Since webhook sessions became unattended (#100532), a plugin that relays approvals for one
route to a chat platform through the ``pre_approval_request`` hook never sees the request: the
command is denied before any pending approval exists. Drives the real config loader against a
temp HERMES_HOME and the real session context.
"""
import base64
import json

import pytest

import hermes_cli.config as hc
from gateway.session_context import clear_session_vars, set_session_vars
from tools import approval_context as ac


def _webhook_key(route: str) -> str:
    token = base64.urlsafe_b64encode(
        json.dumps(["default", route, "delivery-1"], separators=(",", ":")).encode()
    ).decode().rstrip("=")
    return f"agent:main:webhook:webhook:webhook:v2:{token}:webhook:{route}"


@pytest.fixture
def write_config(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    def _write(approvals_yaml: str) -> None:
        (home / "config.yaml").write_text(f"approvals:\n  unattended_mode: deny\n{approvals_yaml}")
        hc._LOAD_CONFIG_CACHE.clear()

    yield _write
    hc._LOAD_CONFIG_CACHE.clear()


@pytest.fixture
def session():
    tokens = []

    def _bind(platform: str, key: str) -> None:
        tokens.append(set_session_vars(platform=platform, session_key=key))

    yield _bind
    for t in reversed(tokens):
        clear_session_vars(t)


@pytest.mark.parametrize("routes_yaml,route,unattended", [
    ("", "relay-route", True),                                             # default: every webhook unattended
    ("  attended_webhook_routes: [relay-route]\n", "relay-route", False),
    ("  attended_webhook_routes: [relay-route]\n", "github-push", True),   # other routes stay strict
    ("  attended_webhook_routes: [relay]\n", "relay-route", True),         # whole route name only
    ("  attended_webhook_routes: relay-route\n", "relay-route", False),    # a bare string works too
])
def test_webhook_route_classification(write_config, session, routes_yaml, route, unattended):
    write_config(routes_yaml)
    session("webhook", _webhook_key(route))
    assert ac._is_unattended_platform_approval_context() is unattended
    assert ac._is_gateway_approval_context() is (not unattended)


def test_only_webhook_platform_is_exempted(write_config, session):
    write_config("  attended_webhook_routes: [relay-route]\n")
    session("api_server", _webhook_key("relay-route"))
    assert ac._is_unattended_platform_approval_context() is True
