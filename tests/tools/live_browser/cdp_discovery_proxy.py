"""Forward a real Chrome discovery response behind an explicit delivery barrier."""
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

import requests


class DiscoveryHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, proxy, **kwargs):
        self.proxy = proxy
        super().__init__(*args, **kwargs)

    def do_GET(self):
        assert self.path == '/json/version'
        with requests.Session() as client:
            client.trust_env = False
            response = client.get(self.proxy.upstream + self.path, timeout=10)
            response.raise_for_status()
        self.proxy.discovered.set()
        assert self.proxy.release.wait(timeout=30), 'Release the real Chrome discovery response'
        self.send_response(response.status_code)
        self.send_header('Content-Type', response.headers['Content-Type'])
        self.send_header('Content-Length', str(len(response.content)))
        self.end_headers()
        self.wfile.write(response.content)


class HoldingDiscoveryProxy:
    def __init__(self, upstream):
        self.upstream = upstream
        self.discovered = threading.Event()
        self.release = threading.Event()
        self.server = ThreadingHTTPServer(
            ('127.0.0.1', 0), partial(DiscoveryHandler, proxy=self)
        )
        self.root = f'http://127.0.0.1:{self.server.server_port}'
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=10)
        assert not self.thread.is_alive()
