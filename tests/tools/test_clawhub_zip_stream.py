"""Bound the archive before extraction, without buffering response.content."""

from contextlib import contextmanager
import io
import zipfile

import httpx
import pytest

from tools import skills_hub_clawhub as clawhub


def _archive():
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr("SKILL.md", "# A normal skill")
    return data.getvalue()


def _mock_download(monkeypatch, data, headers):
    responses = []

    @contextmanager
    def stream(*args, **kwargs):
        response = httpx.Response(200, headers=headers, stream=httpx.ByteStream(data))
        responses.append(response)
        try:
            yield response
        finally:
            response.close()

    monkeypatch.setattr(clawhub, "_guarded_http_stream", stream, raising=False)
    monkeypatch.setattr(clawhub.httpx, "get", lambda *a, **k: httpx.Response(200, headers=headers, content=data))
    return responses


@pytest.mark.parametrize("declared", [None, "1", "invalid", "-1"])
def test_actual_stream_size_enforces_cap_despite_header(monkeypatch, declared):
    data = _archive()
    _mock_download(monkeypatch, data, {} if declared is None else {"content-length": declared})
    monkeypatch.setattr(clawhub.ClawHubSource, "ZIP_DOWNLOAD_MAX_BYTES", len(data) - 1, raising=False)
    assert clawhub.ClawHubSource()._download_zip("example", "1") == {}


def test_exact_limit_stream_extracts_without_content_access_and_closes(monkeypatch):
    data = _archive()
    responses = _mock_download(monkeypatch, data, {})
    monkeypatch.setattr(clawhub.ClawHubSource, "ZIP_DOWNLOAD_MAX_BYTES", len(data), raising=False)
    assert clawhub.ClawHubSource()._download_zip("example", "1") == {"SKILL.md": "# A normal skill"}
    assert len(responses) == 1 and responses[0].is_closed


def test_declared_oversize_does_not_read_body(monkeypatch):
    responses = _mock_download(monkeypatch, b"", {"content-length": "101"})
    monkeypatch.setattr(clawhub.ClawHubSource, "ZIP_DOWNLOAD_MAX_BYTES", 100)
    monkeypatch.setattr(httpx.Response, "iter_bytes", lambda *a, **k: pytest.fail("read oversized body"))
    assert clawhub.ClawHubSource()._download_zip("example", "1") == {}
    assert responses[0].is_closed


def test_rate_limit_exhaustion_closes_responses_and_sleeps_only_between_attempts(monkeypatch):
    responses = []
    delays = []

    @contextmanager
    def stream(*args, **kwargs):
        response = httpx.Response(429, headers={"retry-after": ["-1", "1000", "invalid"][len(responses)]})
        responses.append(response)
        try:
            yield response
        finally:
            response.close()

    monkeypatch.setattr(clawhub, "_guarded_http_stream", stream)
    monkeypatch.setattr(clawhub.time, "sleep", delays.append)
    assert clawhub.ClawHubSource()._download_zip("example", "1") == {}
    assert delays == [0, 15]
    assert len(responses) == 3 and all(response.is_closed for response in responses)


@pytest.mark.parametrize("destination", ["https://download.example/bundle", "http://127.0.0.1/private", "https://blocked.example/bundle"])
def test_stream_rechecks_redirect_policy_and_closes_clients(monkeypatch, destination):
    from tools import skills_hub, url_safety

    requests = []
    clients = []

    def respond(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(302, headers={"location": destination})
        return httpx.Response(200, content=b"bundle")

    def client(**kwargs):
        result = httpx.Client(transport=httpx.MockTransport(respond), **kwargs)
        clients.append(result)
        return result

    monkeypatch.setattr(url_safety, "create_ssrf_safe_client", client)
    monkeypatch.setattr(skills_hub, "is_safe_url", lambda url: "127.0.0.1" not in url)
    monkeypatch.setattr(skills_hub, "check_website_access", lambda url: {"host": "blocked.example", "rule": "test"} if "blocked.example" in url else None)
    with skills_hub._guarded_http_stream("https://api.example/download", params={"slug": "secret"}) as response:
        if "download.example" in destination:
            assert response.read() == b"bundle"
            assert str(requests[1].url) == destination
        else:
            assert response is None
            assert len(requests) == 1
    assert requests[0].url.params["slug"] == "secret"
    assert clients and all(client.is_closed for client in clients)
