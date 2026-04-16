#!/usr/bin/env python3
"""Mock Lightpanda binary for integration tests.

Simulates the real lightpanda binary's CDP server mode:
  mock_lightpanda.py serve --host HOST --port PORT

Starts a minimal HTTP server that responds to /json/version with a valid
CDP discovery payload, exactly as the real Lightpanda binary would.

Usage:
    LIGHTPANDA_PATH=<path-to-this-script> python -m pytest ...
"""

import argparse
import json
import signal
import sys
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    serve_parser = subparsers.add_parser("serve")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=9222)

    args = parser.parse_args()

    if args.command != "serve":
        print("mock_lightpanda: unknown command", file=sys.stderr)
        sys.exit(1)

    host = args.host
    port = args.port
    browser_id = uuid.uuid4().hex

    class CDPHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/json/version":
                payload = {
                    "Browser": "MockLightpanda/1.0.0",
                    "Protocol-Version": "1.3",
                    "webSocketDebuggerUrl": f"ws://{host}:{port}/devtools/browser/{browser_id}",
                }
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, fmt, *args):
            # Suppress access log to keep test output clean
            pass

    server = HTTPServer((host, port), CDPHandler)

    def _shutdown(signum, frame):
        server.server_close()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    server.serve_forever()


if __name__ == "__main__":
    main()
