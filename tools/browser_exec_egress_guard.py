"""Parent-side egress guard plumbing for ``browser_exec`` (Region B).

Generates/verifies the stdlib-only child interposer pair
(``sitecustomize.py`` + ``browser_exec_egress_guard.py``) into a Hermes-owned
cache dir, injects it into the CLI subprocess env (PYTHONPATH prepend +
per-spawn nonce + policy snapshot), parses the child's tamper-evident
markers after the run (fail-closed: any ``:block:``/``:tamper:``/
``:disabled:`` marker, or a missing/nonce-mismatched ``:installed:`` marker,
withholds output), and pins a Hermes filtering forward proxy so
proxy-honoring spawned tools (curl/wget) get the same URL policy.

Files are sha256-verified and regenerated on every spawn (same-UID tamper
between spawns is repaired; a forged marker cannot carry the next spawn's
nonce). ``HERMES_BROWSER_EXEC_EGRESS_GUARD=0`` (config
``browser.exec_egress_guard: off``) disables the guard: the parent logs a
warning, installs nothing, and does no marker checks.

Threat model — coverage boundary (accepted residual, see the child
interposer docstring for the full note): the interposer covers ONLY the CLI
process and the proxy pinning covers only proxy-honoring spawned tools.
Model-spawned env-stripped or native children (no PYTHONPATH → no
sitecustomize → no socket guard; proxy vars dropped; no markers emitted)
bypass both. OS-level egress enforcement (firewall, seccomp, network
namespaces) is out of scope for this PR.
"""

import hashlib
import json
import logging
import os
import secrets
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_GUARD_PREFIX = "__HERMES_EGRESS_GUARD__:"

# Files that make up the child interposer (source of truth: the checked-in
# bootstrap package under tools/browser_use_guard_bootstrap/).
_CHILD_FILES = ("sitecustomize.py", "browser_exec_egress_guard.py")


def _egress_guard_dir() -> Path:
    """Hermes-owned cache dir holding the generated child interposer pair."""
    from hermes_constants import get_hermes_home

    return Path(get_hermes_home()) / "cache" / "browser-use" / "egress-guard"


def _egress_guard_source() -> dict:
    """Return {filename: content} for the child interposer files."""
    pkg = Path(__file__).parent / "browser_use_guard_bootstrap"
    out: dict = {}
    for name in _CHILD_FILES:
        out[name] = (pkg / name).read_text(encoding="utf-8")
    return out


def _file_sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _verify_or_regenerate(guard_dir: Path) -> None:
    """Rewrite any child file that is absent or hash-mismatched (F4 fix)."""
    source = _egress_guard_source()
    guard_dir.mkdir(parents=True, exist_ok=True)
    for name, content in source.items():
        path = guard_dir / name
        try:
            existing = path.read_bytes()
            if _file_sha256(existing) == _file_sha256(content.encode("utf-8")):
                continue
        except OSError:
            pass
        try:
            path.write_text(content, encoding="utf-8")
            logger.debug("egress guard: regenerated %s", path)
        except OSError as e:  # pragma: no cover — cache dir unwritable
            logger.warning("egress guard: cannot write %s: %s", path, e)


def _policy_snapshot(allow_private: Optional[bool] = None, nonce: str = "") -> dict:
    """Frozen JSON-able policy snapshot rendered from tools.url_safety."""
    from tools.url_safety import (
        _ALWAYS_BLOCKED_IPS,
        _ALWAYS_BLOCKED_NETWORKS,
        _BLOCKED_HOSTNAMES,
        _CGNAT_NETWORK,
        _global_allow_private_urls,
    )

    return {
        "nonce": nonce,
        "allow_private": _global_allow_private_urls() if allow_private is None else allow_private,
        "blocked_hostnames": sorted(str(h) for h in _BLOCKED_HOSTNAMES),
        "always_blocked_ips": sorted(str(ip) for ip in _ALWAYS_BLOCKED_IPS),
        "always_blocked_networks": sorted(str(net) for net in _ALWAYS_BLOCKED_NETWORKS),
        "cgnat_network": str(_CGNAT_NETWORK),
        "allow_hosts": [],
    }


def _egress_guard_enabled() -> bool:
    """Operator kill-switch: config ``browser.exec_egress_guard: off``."""
    try:
        from tools.browser_use_cli import _read_browser_cfg

        cfg = _read_browser_cfg()
        value = cfg.get("exec_egress_guard")
        if value is False or str(value or "").strip().lower() == "off":
            return False
    except Exception:
        pass
    if os.environ.get("HERMES_BROWSER_EXEC_EGRESS_GUARD", "") == "0":
        return False
    return True


def _install_egress_guard(env: dict) -> bool:
    """Inject the child interposer into ``env`` (Region B §2.1/§2.6.1).

    Returns True when the guard is armed, False when the operator disabled
    it (in which case nothing is installed and no marker checks apply).
    """
    if not _egress_guard_enabled():
        logger.warning(
            "browser_exec egress guard disabled by config "
            "(browser.exec_egress_guard: off) — CLI subprocess egress is "
            "not interposed."
        )
        return False

    guard_dir = _egress_guard_dir()
    _verify_or_regenerate(guard_dir)

    nonce = secrets.token_hex(8)
    env["HERMES_BROWSER_EXEC_EGRESS_GUARD"] = "1"
    env["HERMES_BROWSER_EXEC_EGRESS_GUARD_NONCE"] = nonce

    policy = _policy_snapshot(nonce=nonce)
    try:
        from tools.browser_use_cli import _read_browser_cfg

        allow_cfg = _read_browser_cfg().get("exec_egress_allow") or ""
        if isinstance(allow_cfg, (list, tuple)):
            allow_cfg = ",".join(str(x) for x in allow_cfg)
        if allow_cfg:
            env["HERMES_BROWSER_EXEC_EGRESS_ALLOW"] = str(allow_cfg)
            policy["allow_hosts"] = [h.strip() for h in str(allow_cfg).split(",") if h.strip()]
    except Exception:
        pass
    env["HERMES_BROWSER_EXEC_EGRESS_POLICY"] = json.dumps(policy, sort_keys=True)

    prepend = str(guard_dir)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = prepend + (os.pathsep + existing if existing else "")

    _pin_egress_proxy(env)
    return True


_MARKER_RE = None


def _marker_re():
    import re

    global _MARKER_RE
    if _MARKER_RE is None:
        _MARKER_RE = re.compile(
            r"^" + re.escape(_GUARD_PREFIX) + r"(installed|block|tamper|disabled):(.*)$"
        )
    return _MARKER_RE


def _parse_guard_markers(stderr: str, nonce: str) -> Optional[str]:
    """Inspect untruncated CLI stderr for guard markers (Region B §2.6.2).

    Returns a withhold reason string, or None when the guard verified clean.
    Called ONLY when the guard is enabled. ``:installed:`` must be present
    with the exact per-spawn nonce; any ``:block:``/``:tamper:``/
    ``:disabled:`` marker is an immediate withhold.
    """
    installed_ok = False
    for line in (stderr or "").splitlines():
        m = _marker_re().match(line.strip())
        if not m:
            continue
        kind, payload = m.group(1), m.group(2).strip()
        if kind == "installed":
            installed_ok = payload == nonce
        elif kind == "block":
            return (
                "Blocked: browser_exec attempted a direct connection to a "
                f"private or internal address ({payload}); output withheld."
            )
        elif kind == "tamper":
            return (
                "Blocked: the browser_exec egress guard detected binding "
                f"tamper ({payload}); output withheld."
            )
        elif kind == "disabled":
            return (
                f"Blocked: the browser_exec egress guard disabled itself "
                f"({payload}); output withheld."
            )
    if not installed_ok:
        return (
            "Blocked: the browser_exec egress guard did not report a "
            "verified install (missing or nonce-mismatched :installed: "
            "marker); output withheld."
        )
    return None


def _strip_guard_markers(stderr: str) -> str:
    """Remove guard marker lines from stderr before it reaches the model."""
    if _GUARD_PREFIX not in (stderr or ""):
        return stderr
    kept = [ln for ln in stderr.splitlines() if _GUARD_PREFIX not in ln]
    return "\n".join(kept)


# ── L2: Hermes filtering forward proxy (complementary to the interposer) ──

_PROXY_LOCK = threading.Lock()
_PROXY_INSTANCE = None  # (server, port)


class _EgressFilterProxyHandler(BaseHTTPRequestHandler):
    """Minimal forwarding proxy that re-runs the URL policy per request.

    Handles absolute-form HTTP requests and CONNECT tunnels. Blocked targets
    get a 403. Redirects are followed manually (each hop re-checked).
    """

    protocol_version = "HTTP/1.1"

    # Silence BaseHTTPRequestHandler's default stderr logging.
    def log_message(self, *args):  # pragma: no cover — noisy by default
        pass

    def _blocked(self, url: str) -> bool:
        try:
            from tools.browser_tool import _url_block_reason

            return _url_block_reason(url) is not None
        except Exception:
            return True  # fail closed if the policy module is unavailable

    def _reply(self, code: int, body: bytes, content_type: str = "text/plain") -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 — http.server API
        self._forward_absolute()

    def do_POST(self):  # noqa: N802
        self._forward_absolute()

    def do_PUT(self):  # noqa: N802
        self._forward_absolute()

    def do_DELETE(self):  # noqa: N802
        self._forward_absolute()

    def do_HEAD(self):  # noqa: N802
        self._forward_absolute()

    def _forward_absolute(self) -> None:
        import urllib.request

        target = self.path
        if not target.lower().startswith(("http://", "https://")):
            self._reply(400, b"only absolute-form HTTP requests are proxied")
            return
        if self._blocked(target):
            self._reply(403, b"Blocked by Hermes browser_exec egress guard")
            return
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        for _hop in range(6):
            if self._blocked(target):
                self._reply(403, b"Blocked by Hermes browser_exec egress guard")
                return
            try:
                req = urllib.request.Request(
                    target, data=body, headers=dict(self.headers), method=self.command
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    if resp.status in (301, 302, 303, 307, 308):
                        location = resp.headers.get("Location")
                        if not location:
                            self._reply(502, b"redirect without Location")
                            return
                        from urllib.parse import urljoin

                        target = urljoin(target, location)
                        body = None
                        continue
                    payload = resp.read()
                    self.send_response(resp.status)
                    for key, value in resp.headers.items():
                        if key.lower() in ("transfer-encoding", "connection"):
                            continue
                        self.send_header(key, value)
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
            except Exception:
                self._reply(502, b"upstream failure")
                return
        self._reply(502, b"too many redirects")

    def do_CONNECT(self):  # noqa: N802
        import ipaddress

        host_port = self.path
        host, _, port_s = host_port.rpartition(":")
        if not host or not port_s.isdigit():
            self._reply(400, b"malformed CONNECT target")
            return
        port = int(port_s)
        # Re-run the URL policy on the tunnel target.
        try:
            from tools.url_safety import ip_is_blocked

            host_ip = host.strip("[]")
            try:
                blocked, _ = ip_is_blocked(host_ip)
            except Exception:
                blocked = False
            if not blocked and not _is_literal(host_ip):
                import socket as _socket

                try:
                    infos = _socket.getaddrinfo(host, port, _socket.AF_UNSPEC, _socket.SOCK_STREAM)
                    blocked = any(
                        _ip_blocked_for_connect(sa[0]) for *_x, sa in infos
                    )
                except _socket.gaierror:
                    blocked = True
        except Exception:
            blocked = True
        if blocked:
            self._reply(403, b"Blocked by Hermes browser_exec egress guard")
            return
        try:
            upstream = socket.create_connection((host, port), timeout=30)
        except OSError:
            self._reply(502, b"upstream connect failed")
            return
        self.send_response(200, "Connection Established")
        self.send_header("Content-Length", "0")
        self.end_headers()
        try:
            _pump(self.connection, upstream)
        except Exception:
            pass
        finally:
            try:
                upstream.close()
            except Exception:
                pass


def _is_literal(host: str) -> bool:
    import ipaddress

    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def _ip_blocked_for_connect(ip_str: str) -> bool:
    import ipaddress

    from tools.url_safety import ip_is_blocked

    try:
        blocked, _ = ip_is_blocked(ip_str)
        return blocked
    except Exception:
        return True


def _pump(client: socket.socket, upstream: socket.socket) -> None:
    import select

    client.setblocking(False)
    upstream.setblocking(False)
    try:
        while True:
            readable, _, _ = select.select([client, upstream], [], [], 1.0)
            if not readable:
                continue
            for sock in readable:
                try:
                    data = sock.recv(65536)
                except OSError:
                    return
                if not data:
                    return
                peer = upstream if sock is client else client
                try:
                    peer.sendall(data)
                except OSError:
                    return
    finally:
        client.setblocking(True)


class _EgressFilterProxy:
    """Daemon-thread stdlib HTTP proxy that filters by URL policy."""

    def __init__(self) -> None:
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> Optional[int]:
        if self._server is not None:
            return self._server.server_address[1]
        try:
            self._server = ThreadingHTTPServer(("127.0.0.1", 0), _EgressFilterProxyHandler)
        except OSError as e:
            logger.warning("egress proxy could not start: %s", e)
            return None
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="hermes-egress-filter-proxy",
            daemon=True,
        )
        self._thread.start()
        return self._server.server_address[1]


def _pin_egress_proxy(env: dict) -> None:
    """Point HTTP(S)/ALL proxy vars at the filtering proxy (Region B §2.6.5).

    Fail-closed: when the egress guard is armed and the filtering proxy
    cannot be started, raise — the caller withholds before the CLI spawns
    (proxy-honoring spawned tools would otherwise be unfiltered).

    ``NO_PROXY=127.0.0.1,localhost,::1`` keeps loopback CDP/daemon traffic
    off the proxy so local-browser flows keep working.
    """
    global _PROXY_INSTANCE
    with _PROXY_LOCK:
        if _PROXY_INSTANCE is None:
            port = _EgressFilterProxy().start()
            if port is None:
                raise RuntimeError(
                    "browser_exec egress guard could not start its filtering "
                    "proxy; proxy-honoring spawned tools would be unfiltered "
                    "(fail-closed)"
                )
            _PROXY_INSTANCE = ("127.0.0.1", port)
        host, port = _PROXY_INSTANCE
    proxy_url = f"http://{host}:{port}"
    for var in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy",
                "ALL_PROXY", "all_proxy"):
        env[var] = proxy_url
    env["NO_PROXY"] = "127.0.0.1,localhost,::1"
    env.setdefault("no_proxy", "127.0.0.1,localhost,::1")
