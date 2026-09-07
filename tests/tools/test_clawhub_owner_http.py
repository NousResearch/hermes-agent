"""Owner-qualified ClawHub fetches retain identity across actual HTTP requests."""

import io
import json
import threading
import zipfile
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from tools.skills_hub_clawhub import ClawHubSource


@contextmanager
def registry(*, fallback=False, mismatch=False):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            url = urlsplit(self.path)
            query = parse_qs(url.query)
            requests.append((url.path, query))
            status = 200
            if query.get("owner") != ["alice"]:
                status, payload = 409, {"code": "AMBIGUOUS_SKILL_SLUG"}
            elif url.path.endswith("/download"):
                if fallback:
                    status, payload = 404, {}
                else:
                    buf = io.BytesIO()
                    with zipfile.ZipFile(buf, "w") as archive:
                        archive.writestr("SKILL.md", "# Alice fixture")
                    payload = buf.getvalue()
            elif url.path.endswith("/versions"):
                payload = {"items": [{"version": "1.0"}]} if fallback else [{"version": "1.0"}]
            elif url.path.endswith("/versions/1.0"):
                payload = {"files": {"SKILL.md": "# Alice fixture"}}
            else:
                # Explicit owner must survive even when metadata omits it.
                payload = {"skill": {"slug": "collision"}}
                if mismatch:
                    payload["owner"] = {"handle": "bob"}
            body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    source = ClawHubSource()
    source.BASE_URL = f"http://127.0.0.1:{server.server_port}/api/v1"
    try:
        yield source, requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_qualified_owner_survives_metadata_versions_and_download():
    for fallback in (False, True):
        with registry(fallback=fallback) as (source, requests):
            for identifier in ("@alice/collision", "clawhub/@alice/collision", "alice/skills/collision"):
                meta = source.inspect(identifier)
                assert meta is not None
                assert meta.identifier == "@alice/collision"
                bundle = source.fetch(identifier)
                assert bundle is not None
                assert bundle.identifier == "@alice/collision"
                assert bundle.files == {"SKILL.md": "# Alice fixture"}
            assert all(query.get("owner") == ["alice"] for _, query in requests)
            assert any(path.endswith("/versions") for path, _ in requests)
            assert any(path.endswith("/download") for path, _ in requests)
            if fallback:
                assert any(path.endswith("/versions/1.0") for path, _ in requests)


def test_ambiguous_or_mismatched_owner_never_downloads():
    with registry() as (source, requests):
        assert source.fetch("collision") is None
        assert source.fetch("github-owner/repository/collision") is None
        assert len(requests) == 1
    with registry(mismatch=True) as (source, requests):
        assert source.fetch("@alice/collision") is None
        assert len(requests) == 1
