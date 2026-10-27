"""What the metadata floor refuses, and host forms WHATWG clients normalise away.

``is_always_blocked_url`` guards local browser backends that skip ``is_safe_url``, so it refuses
only what can hide a different host. A templated port (``{port}``, ``%d``, ``${port}``) leaves the
host as written. Chromium and Node drop a trailing C0 control and map IDNA dots (``。``, ``．``,
``｡``) to ``.``, so both checks must judge the host those clients dial, even when DNS fails.
"""

import socket
from unittest.mock import patch

import pytest

from tools import browser_session_cloud, browser_url_policy, browser_use_cli
from tools.url_safety import is_always_blocked_url, is_safe_url

_LOCALHOST = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0))]


def _localhost_only(host, *args, **kwargs):
    """``localhost`` resolves to loopback; every other name fails like an unresolvable name."""
    if host == "localhost":
        return _LOCALHOST
    raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")


@pytest.fixture(autouse=True)
def _dns():
    with patch("socket.getaddrinfo", side_effect=_localhost_only):
        yield


@pytest.fixture
def local_browser(monkeypatch):
    """A local Chromium backend: ``is_safe_url`` is skipped, only the metadata floor applies."""
    monkeypatch.setattr(browser_session_cloud, "is_local_backend", lambda: True)
    monkeypatch.setattr(browser_session_cloud, "allow_private_urls", lambda: False)
    monkeypatch.setattr(browser_url_policy, "hybrid_routes_locally", lambda url: False)
    monkeypatch.setattr(browser_url_policy, "check_website_access", lambda url: None)


@pytest.fixture
def proxy(monkeypatch):
    """A configured proxy: ``is_safe_url`` hands a DNS failure on a name to the proxy."""
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:3128")


TEMPLATED_CODE = [
    'port = 8000\nnew_tab(f"http://localhost:{port}/")',
    'port = 8000\nnew_tab("http://localhost:%d/" % port)',
    'js("fetch(`http://localhost:${port}/api`)")',
]

class TestFloorAcceptsTemplatedUrlLiterals:
    @pytest.mark.parametrize("code", TEMPLATED_CODE)
    def test_local_browser_runs_templated_dev_server_urls(self, local_browser, code):
        assert browser_use_cli._code_url_verdict(code).blocked is None

    @pytest.mark.parametrize("url", [
        "http://localhost:{port}/", "http://localhost:%d/", "http://localhost:${port}/api",
        "http://{host}:8080/", "http://%s/", "http://user:pass@localhost:{port}/",
    ])
    def test_floor_ignores_unparseable_port_and_host_placeholders(self, url):
        assert is_always_blocked_url(url) is False

    @pytest.mark.parametrize("url", ["http://localhost:{port}/", "http://localhost:%d/"])
    def test_is_safe_url_keeps_refusing_them(self, url):
        assert is_safe_url(url) is False


class TestFloorJudgesPercentEncodedHostDecoded:
    @pytest.mark.parametrize("url", [
        "http://%31%36%39.254.169.254/",
        "HTTP://%31%36%39.254.169.254:80/latest",
        "http://metadata.google.%69nternal/computeMetadata/v1/",
    ])
    def test_decoded_metadata_host_blocked(self, url):
        assert is_always_blocked_url(url) is True

    def test_decoded_metadata_host_blocked_through_browser_scan(self, local_browser):
        verdict = browser_use_cli._code_url_verdict('new_tab("http://%31%36%39.254.169.254/")')
        assert verdict.blocked is not None and "cloud metadata" in verdict.blocked

    @pytest.mark.parametrize("url", ["http://%6c%6f%63%61%6c%68%6f%73%74/", "http://exa%20mple.com/"])
    def test_decoded_ordinary_host_not_blocked(self, url):
        assert is_always_blocked_url(url) is False
