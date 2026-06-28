import pytest

from gateway.platforms import yuanbao_media


@pytest.mark.asyncio
async def test_download_url_rejects_unsafe_url_before_http(monkeypatch):
    monkeypatch.setattr(yuanbao_media.url_safety, "is_safe_url", lambda url: False)
    client_cls = MockAsyncClientFactory()
    monkeypatch.setattr(yuanbao_media.httpx, "AsyncClient", client_cls)

    with pytest.raises(ValueError, match="Blocked unsafe URL"):
        await yuanbao_media.download_url("http://127.0.0.1:8000/private.png")

    assert client_cls.calls == []


@pytest.mark.asyncio
async def test_download_url_installs_redirect_ssrf_guard(monkeypatch):
    monkeypatch.setattr(yuanbao_media.url_safety, "is_safe_url", lambda url: True)
    client_cls = MockAsyncClientFactory(enter_exception=StopAfterClientCreated)
    monkeypatch.setattr(yuanbao_media.httpx, "AsyncClient", client_cls)

    with pytest.raises(StopAfterClientCreated):
        await yuanbao_media.download_url("https://example.com/image.png")

    assert len(client_cls.calls) == 1
    kwargs = client_cls.calls[0]
    assert kwargs["follow_redirects"] is True
    assert kwargs["event_hooks"] == {"response": [yuanbao_media._ssrf_redirect_guard]}


class StopAfterClientCreated(Exception):
    pass


class MockAsyncClientFactory:
    def __init__(self, enter_exception=None):
        self.calls = []
        self.enter_exception = enter_exception

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return MockAsyncClient(self.enter_exception)


class MockAsyncClient:
    def __init__(self, enter_exception=None):
        self.enter_exception = enter_exception

    async def __aenter__(self):
        if self.enter_exception:
            raise self.enter_exception()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False
