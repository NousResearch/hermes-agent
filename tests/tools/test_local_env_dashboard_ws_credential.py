"""The dashboard hands its PTY child two WebSocket URLs that carry a live dashboard
credential in the query string (``?token=<_SESSION_TOKEN>`` on a loopback dashboard,
``?internal=<internal_ws_credential()>`` when the dashboard is gated):
``HERMES_TUI_SIDECAR_URL`` and ``HERMES_TUI_GATEWAY_URL``. The TUI process reads them
once at start-up. The agent's own children never need them, and the scrub denylisted
only the bare ``HERMES_DASHBOARD_SESSION_TOKEN``, so ``terminal("env")`` showed a
working ``/api/ws`` login."""

import pytest

from tests.tools._child_env_fixtures import child_env, observe_child  # child_env: fixture
from tools.environments import local

TUI_URL_VARS = ["HERMES_TUI_SIDECAR_URL", "HERMES_TUI_GATEWAY_URL"]
URL = "ws://127.0.0.1:9119/api/pub?token=0123456789abcdef0123456789abcdef&channel=c1"
# Other HERMES_TUI_* knobs are plain settings and keep flowing to children.
KEPT = {"HERMES_TUI_MAX_TURNS": "7", "MY_CUSTOM_VAR": "x"}


@pytest.mark.parametrize("builder", [
    pytest.param(lambda: local._make_run_env({}), id="terminal"),
    pytest.param(lambda: local.build_subprocess_env(scrub_secrets=True), id="background"),
    pytest.param(lambda: local.hermes_subprocess_env(), id="helper"),
    pytest.param(lambda: local.hermes_subprocess_env(inherit_credentials=True), id="inherit_credentials"),
])
def test_dashboard_ws_credential_urls_never_reach_spawned_children(child_env, monkeypatch, builder):
    for name in TUI_URL_VARS:
        monkeypatch.setenv(name, URL)
    for name, value in KEPT.items():
        monkeypatch.setenv(name, value)
    observed = observe_child(builder(), [*TUI_URL_VARS, *KEPT])
    assert {k: observed[k] for k in TUI_URL_VARS} == dict.fromkeys(TUI_URL_VARS)
    assert {k: observed[k] for k in KEPT} == KEPT
