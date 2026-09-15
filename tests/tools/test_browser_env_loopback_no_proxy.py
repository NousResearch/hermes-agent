"""Loopback CDP dials must never go through a proxy (#110565).

``websockets>=14`` connects with ``proxy=True`` and resolves the proxy via
``urllib.request.getproxies()``, which on macOS reads the *system* proxy even with no
``*_proxy`` env vars. The browser child env therefore carries loopback ``NO_PROXY`` entries
(both casings, operator entries kept), and the in-process CDP dials pass ``proxy=None``
for loopback URLs only.
"""

import pytest

import tools.browser_tool as bt
from agent.proxy_bypass import loopback_connect_kwargs


@pytest.fixture
def stub_sanitized_env(monkeypatch):
    """Replace the credential-scrub layer with a fixed dict so the test sees exactly what
    ``_build_browser_env`` adds on top."""
    import tools.environments.local as local
    holder = {}
    monkeypatch.setattr(local, "hermes_subprocess_env", lambda inherit_credentials=False: dict(holder))
    return holder


def test_browser_env_appends_loopback_to_operator_no_proxy(stub_sanitized_env):
    stub_sanitized_env["NO_PROXY"] = "git.internal, 10.0.0.0/8"
    env = bt._build_browser_env()
    assert env["NO_PROXY"] == "git.internal,10.0.0.0/8,127.0.0.1,localhost,::1"
    assert env["no_proxy"] == "127.0.0.1,localhost,::1"


@pytest.mark.parametrize("url, expected", [
    ("ws://127.0.0.1:9222/devtools/browser/abc", {"proxy": None}),
    ("ws://localhost:9222/devtools/browser/abc", {"proxy": None}),
    ("ws://[::1]:9222/devtools/browser/abc", {"proxy": None}),
    ("wss://connect.browserbase.com/cdp?apiKey=x", {}),
])
def test_in_process_cdp_dial_disables_proxy_only_for_loopback(url, expected):
    assert loopback_connect_kwargs(url) == expected
