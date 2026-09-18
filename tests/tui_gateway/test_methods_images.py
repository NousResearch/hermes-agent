"""``image.generate`` RPC: the provider-result ``image`` field may be a bare remote
URL (cache_url_best_effort falls back to it), so ``_image_to_data_url`` fetches it
server-side — it must route through the SSRF guard like provider_media.save_url.
"""

import http.server
import socketserver
import threading

import httpx
import pytest

from tui_gateway import methods_images

_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000d49444154789c6300010000000500010d0a2db40000000049454e44"
    "ae426082"
)


def test_image_to_data_url_refuses_metadata_url():
    assert methods_images._image_to_data_url("http://169.254.169.254/latest/meta-data", 1024) is None


def test_image_to_data_url_refuses_file_scheme():
    assert methods_images._image_to_data_url("file:///etc/passwd", 1024) is None


def test_image_to_data_url_fetches_safe_url(monkeypatch):
    monkeypatch.setattr("tools.url_safety.is_safe_url", lambda url: True)
    monkeypatch.setattr(
        "tools.url_safety.create_ssrf_safe_client",
        lambda **kw: httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, content=_PNG, headers={"content-type": "image/png"}, request=request)),
            **kw))

    data_url = methods_images._image_to_data_url("https://example.com/i.png", 1024)

    assert data_url is not None
    assert data_url.startswith("data:image/png;base64,")


def test_image_to_data_url_caps_remote_body(monkeypatch):
    monkeypatch.setattr("tools.url_safety.is_safe_url", lambda url: True)
    monkeypatch.setattr(
        "tools.url_safety.create_ssrf_safe_client",
        lambda **kw: httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, content=_PNG * 8, headers={"content-type": "image/png"}, request=request)),
            **kw))

    assert methods_images._image_to_data_url("https://example.com/i.png", 64) is None


class _ImageHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path == "/ok.png":
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.end_headers()
            self.wfile.write(_PNG)
        elif self.path == "/redirect-metadata":
            self.send_response(302)
            self.send_header("Location", "http://169.254.169.254/latest/meta-data/")
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args, **kw):
        return


@pytest.fixture
def image_server(monkeypatch):
    """Real local HTTP server — same e2e shape as tests/agent/test_save_url_image."""
    monkeypatch.setenv("HERMES_ALLOW_PRIVATE_URLS", "1")
    from tools import url_safety
    url_safety._reset_allow_private_cache()
    httpd = socketserver.TCPServer(("127.0.0.1", 0), _ImageHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    monkeypatch.delenv("HERMES_ALLOW_PRIVATE_URLS", raising=False)
    url_safety._reset_allow_private_cache()


def test_image_to_data_url_e2e_real_fetch(image_server):
    """End-to-end: real guarded client, real socket, real cache path."""
    data_url = methods_images._image_to_data_url(f"{image_server}/ok.png", 1 << 20)
    assert data_url is not None and data_url.startswith("data:image/png;base64,")


def test_image_to_data_url_e2e_redirect_to_metadata_blocked(image_server):
    assert methods_images._image_to_data_url(f"{image_server}/redirect-metadata", 1 << 20) is None
