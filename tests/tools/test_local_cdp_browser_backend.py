"""Tests for the fake-transport LocalCdpBrowserBackend safety wrapper."""

from __future__ import annotations


class FakeCdpTransport:
    def __init__(self, *, url: str = "http://localhost/app?secret=hide", ok: bool = True) -> None:
        self.url = url
        self.ok = ok
        self.calls = []

    def call(self, method, params=None):
        self.calls.append((method, params or {}))
        if method == "Browser.getVersion":
            return {"ok": self.ok, "product": "FakeChrome/0"}
        if method == "Hermes.getActivePage":
            return {"ok": True, "url": self.url, "tab_id_hash": "tab123"}
        return {"ok": True, "method": method, "params": params or {}}


def test_local_cdp_backend_requires_loopback_allowlist_and_transport():
    from tools.computer_use.browser_input import LocalCdpBrowserBackend

    assert not LocalCdpBrowserBackend(endpoint="http://127.0.0.1:9222", domain_allowlist=("localhost",)).is_available()
    assert not LocalCdpBrowserBackend(endpoint="http://192.168.1.2:9222", domain_allowlist=("localhost",), transport=FakeCdpTransport()).is_available()
    assert not LocalCdpBrowserBackend(endpoint="http://127.0.0.1:9222", domain_allowlist=(), transport=FakeCdpTransport()).is_available()
    assert LocalCdpBrowserBackend(endpoint="http://127.0.0.1:9222", domain_allowlist=("localhost",), transport=FakeCdpTransport()).is_available()


def test_local_cdp_backend_redacts_url_and_routes_fake_click():
    from tools.computer_use.browser_input import LocalCdpBrowserBackend

    transport = FakeCdpTransport(url="http://localhost/app?token=secret#frag")
    backend = LocalCdpBrowserBackend(endpoint="http://127.0.0.1:9222", domain_allowlist=("localhost",), transport=transport)

    result = backend.click({"selector": "#ok", "role": "button", "name": "OK", "metadata": {"label": "continue"}})

    assert result["ok"] is True
    assert result["domain"] == "localhost"
    assert result["url"] == "http://localhost/app"
    assert transport.calls[-1][0] == "Hermes.click"
    assert transport.calls[-1][1]["target"]["selector"] == "#ok"


def test_local_cdp_backend_blocks_disallowed_domain():
    from tools.computer_use.browser_input import LocalCdpBrowserBackend

    backend = LocalCdpBrowserBackend(endpoint="http://127.0.0.1:9222", domain_allowlist=("localhost",), transport=FakeCdpTransport(url="https://example.com/account"))

    result = backend.scroll({"selector": "main"}, delta_y=100)

    assert result["ok"] is False
    assert result["error"] == "domain_not_allowed"


def test_local_cdp_backend_blocks_sensitive_target_and_text():
    from tools.computer_use.browser_input import LocalCdpBrowserBackend

    backend = LocalCdpBrowserBackend(endpoint="http://127.0.0.1:9222", domain_allowlist=("localhost",), transport=FakeCdpTransport())

    password_result = backend.type_text({"selector": "#password", "role": "textbox", "name": "Password"}, "hello")
    token_result = backend.type_text({"selector": "#search", "role": "textbox", "name": "Search"}, "oauth token 123")

    assert password_result["ok"] is False
    assert "password" in password_result["error"].lower()
    assert token_result["ok"] is False
    assert "sensitive" in token_result["error"].lower()


def test_browser_input_executor_can_use_local_cdp_backend_with_fake_transport():
    from tools.computer_use.browser_input import BrowserInputExecutor, LocalCdpBrowserBackend
    from tools.computer_use.proposals import BrowserActionProposal, ElementRef

    transport = FakeCdpTransport()
    backend = LocalCdpBrowserBackend(endpoint="http://127.0.0.1:9222", domain_allowlist=("localhost",), transport=transport)
    executor = BrowserInputExecutor(backend=backend)
    proposal = BrowserActionProposal(action="scroll", target=ElementRef(selector="main"), origin="http://localhost/app")

    result = executor.execute(proposal, platform="win32")

    assert result.ok is True
    assert result.executed is True
    assert result.backend_result["action"] == "scroll"
    assert any(call[0] == "Hermes.scroll" for call in transport.calls)
