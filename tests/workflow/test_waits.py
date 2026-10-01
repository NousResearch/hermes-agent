"""A poll wait's URL is authored in a model-editable workflow, so its probe obeys Hermes's one
outbound policy (``tools.url_safety``): private targets need the explicit opt-in, cloud metadata
is refused even with it, and a redirect is re-validated before it is followed."""

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from tools.url_safety import _reset_allow_private_cache
from workflow.waits import http_ok


@pytest.fixture
def server():
    hits: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            if self.path == "/to-metadata":
                self.send_response(302)
                self.send_header("Location", "http://169.254.169.254/latest/meta-data/")
            else:
                self.send_response(200)
            self.end_headers()

        def log_message(self, *_args):
            return

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", hits
    httpd.shutdown()
    _reset_allow_private_cache()


def test_poll_refuses_private_targets_without_the_opt_in(server, monkeypatch):
    base, hits = server
    monkeypatch.setenv("HERMES_ALLOW_PRIVATE_URLS", "false")
    _reset_allow_private_cache()
    assert http_ok(f"{base}/ready") is False
    assert http_ok("http://10.0.0.1/ready") is False
    assert hits == []  # refused before any connection was made


def test_metadata_stays_refused_under_the_opt_in_even_through_a_redirect(server, monkeypatch):
    base, hits = server
    monkeypatch.setenv("HERMES_ALLOW_PRIVATE_URLS", "true")
    _reset_allow_private_cache()
    assert http_ok(f"{base}/ready") is True  # the opt-in admits a private target
    assert http_ok("http://169.254.169.254/latest/meta-data/") is False
    assert http_ok(f"{base}/to-metadata") is False  # redirect hop re-validated at connect
    assert hits == ["/ready", "/to-metadata"]
