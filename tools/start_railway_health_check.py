#!/usr/bin/env python3
"""Start Hermes and wait for its health endpoint before returning.

This wrapper is suitable for Railway/process supervisor startup because
the Railway HTTP probe can reach the health endpoint even when the
gateway is still initializing.
"""

import os
import subprocess
import sys
import time


def _health_url() -> str:
    host = os.environ.get("RAILWAY_PUBLIC_DOMAIN") or os.environ.get("RAILWAY_SERVICE_HERMES_Z0RV_URL") or os.environ.get("HERMES_PUBLIC_URL", "")
    port = os.environ.get("PORT") or os.environ.get("HERMES_PORT") or "8080"
    if not host:
        host = f"127.0.0.1:{port}"
        return f"http://{host}/health"
    if "://" not in host:
        host = f"http://{host}"
    if host.endswith("/"):
        host = host[:-1]
    return f"{host}/health"


def main() -> int:
    cmd = [
        sys.executable,
        "-m",
        "hermes_cli.main",
        "gateway",
        "run",
        "--no-supervise",
    ]
    proc = subprocess.Popen(cmd)
    try:
        timeout = float(os.environ.get("HERMES_HEALTH_TIMEOUT", "180"))
        deadline = time.time() + timeout
        last = None
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            try:
                import urllib.request
                req = urllib.request.Request(_health_url(), method="GET")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    if resp.status == 200:
                        sys.stdout.write(f"hermes health ready: {_health_url()}\n")
                        sys.stdout.flush()
                        return proc.wait()
            except Exception as exc:
                last = exc
            time.sleep(2)
        sys.stderr.write(f"hermes health probe failed: timeout / last_error={last}\n")
        sys.stderr.flush()
        return proc.wait()
    finally:
        if proc.poll() is None:
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
