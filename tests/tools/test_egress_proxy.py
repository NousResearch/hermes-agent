"""Tests for the sandbox egress allowlist (security-critical host matching)."""

import contextlib
import http.client
import socket
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tools.environments import egress_proxy
from tools.environments import egress_proxy_server as eps


# ── host_allowed: the default-deny matching core ──────────────────


def test_empty_allowlist_denies_everything():
    assert eps.host_allowed("github.com", []) is False
    assert eps.host_allowed("", ["github.com"]) is False


def test_exact_match():
    assert eps.host_allowed("github.com", ["github.com"]) is True
    assert eps.host_allowed("GitHub.com", ["github.com"]) is True  # case-insensitive


def test_bare_domain_authorizes_subdomains():
    assert eps.host_allowed("api.github.com", ["github.com"]) is True
    assert eps.host_allowed("a.b.github.com", ["github.com"]) is True


def test_wildcard_matches_subdomains_not_apex():
    al = ["*.githubusercontent.com"]
    assert eps.host_allowed("objects.githubusercontent.com", al) is True
    assert eps.host_allowed("githubusercontent.com", al) is False


def test_suffix_confusion_is_rejected():
    """A host that merely ends with the allowed string must NOT match."""
    al = ["example.com"]
    assert eps.host_allowed("notexample.com", al) is False
    assert eps.host_allowed("evilexample.com", al) is False
    # Attacker-controlled parent domain must not be authorized by a child entry.
    assert eps.host_allowed("example.com.evil.com", al) is False


def test_trailing_dot_normalized():
    assert eps.host_allowed("github.com.", ["github.com"]) is True


def test_ip_literal_requires_exact_match():
    """A literal IP entry authorizes only itself — no subdomain/suffix logic."""
    al = ["1.2.3.4"]
    assert eps.host_allowed("1.2.3.4", al) is True
    assert eps.host_allowed("x.1.2.3.4", al) is False
    assert eps.host_allowed("11.2.3.4", al) is False
    assert eps.host_allowed("1.2.3.4.evil.com", al) is False


def test_ipv6_literal_exact_match():
    assert eps.host_allowed("::1", ["::1"]) is True
    assert eps.host_allowed("::2", ["::1"]) is False


# ── parse_allowlist ───────────────────────────────────────────────


def test_parse_allowlist_splits_and_dedupes():
    assert eps.parse_allowlist("github.com, pypi.org\nGITHUB.COM") == ["github.com", "pypi.org"]
    assert eps.parse_allowlist("") == []
    assert eps.parse_allowlist(None) == []


# ── orchestration naming is deterministic & allowlist-sensitive ───


def test_names_are_deterministic_and_order_insensitive():
    n1, p1 = egress_proxy._names(["github.com", "pypi.org"])
    n2, p2 = egress_proxy._names(["pypi.org", "github.com"])  # reordered
    assert (n1, p1) == (n2, p2)
    assert n1.startswith("hermes-egress-")
    assert p1.startswith("hermes-egress-proxy-")


def test_different_allowlists_get_different_networks():
    n1, _ = egress_proxy._names(["github.com"])
    n2, _ = egress_proxy._names(["github.com", "pypi.org"])
    assert n1 != n2


def test_proxy_env_points_at_alias():
    env = egress_proxy._proxy_env()
    assert env["HTTPS_PROXY"] == f"http://{egress_proxy.PROXY_ALIAS}:{egress_proxy.PROXY_PORT}"
    assert env["http_proxy"] == env["HTTP_PROXY"]
    assert "127.0.0.1" in env["NO_PROXY"]


# ── orchestration: provisioning / dual-homing / prune (mocked docker) ──


def _cp(rc: int = 0, out: str = "", err: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=["docker"], returncode=rc, stdout=out, stderr=err)


def test_ensure_network_creates_internal_with_label(monkeypatch):
    calls = []

    def fake(*args, **kw):
        calls.append(args)
        if args[:2] == ("network", "inspect"):
            return _cp(1, err="not found")
        return _cp(0)

    monkeypatch.setattr(egress_proxy, "_docker", fake)
    egress_proxy._ensure_network("hermes-egress-test")
    create = next(c for c in calls if c[:2] == ("network", "create"))
    assert "--internal" in create
    assert "--label" in create and f"{egress_proxy._EGRESS_LABEL}=1" in create


def test_ensure_network_lost_create_race_is_ok(monkeypatch):
    """create failing because another process won the race must not raise."""
    inspects = {"n": 0}

    def fake(*args, **kw):
        if args[:2] == ("network", "inspect"):
            inspects["n"] += 1
            return _cp(0 if inspects["n"] > 1 else 1)
        if args[:2] == ("network", "create"):
            return _cp(1, err="network already exists")
        return _cp(0)

    monkeypatch.setattr(egress_proxy, "_docker", fake)
    egress_proxy._ensure_network("hermes-egress-test")  # no raise


def test_ensure_network_real_failure_raises(monkeypatch):
    def fake(*args, **kw):
        if args[:2] == ("network", "inspect"):
            return _cp(1)
        if args[:2] == ("network", "create"):
            return _cp(1, err="permission denied")
        return _cp(0)

    monkeypatch.setattr(egress_proxy, "_docker", fake)
    with pytest.raises(RuntimeError, match="permission denied"):
        egress_proxy._ensure_network("hermes-egress-test")


def test_start_proxy_noop_when_already_running(monkeypatch):
    calls = []

    def fake(*args, **kw):
        calls.append(args)
        if args[0] == "inspect" and "{{.State.Running}}" in args:
            return _cp(0, out="true\n")
        return _cp(0)

    monkeypatch.setattr(egress_proxy, "_docker", fake)
    egress_proxy._start_proxy("net-x", "proxy-x", ["github.com"])
    assert not any(c[0] == "run" for c in calls)


def test_start_proxy_restarts_stopped_container(monkeypatch):
    calls = []
    running = iter(["false", "true"])  # before start / after start

    def fake(*args, **kw):
        calls.append(args)
        if args[0] == "inspect" and "{{.State.Running}}" in args:
            return _cp(0, out=next(running))
        if args[0] == "inspect" and "{{.State.Status}}" in args:
            return _cp(0, out="exited")
        return _cp(0)

    monkeypatch.setattr(egress_proxy, "_docker", fake)
    egress_proxy._start_proxy("net-x", "proxy-x", [])
    assert ("start", "proxy-x") in calls
    assert not any(c[0] == "run" for c in calls)


def test_start_proxy_creates_labeled_container_and_dual_homes(monkeypatch):
    calls = []

    def fake(*args, **kw):
        calls.append(args)
        if args[0] == "inspect":
            return _cp(1)  # doesn't exist yet
        return _cp(0)

    monkeypatch.setattr(egress_proxy, "_docker", fake)
    monkeypatch.setattr(egress_proxy, "find_docker", lambda: "/usr/local/bin/docker")
    egress_proxy._start_proxy("net-x", "proxy-x", ["github.com"])
    run = next(c for c in calls if c[0] == "run")
    assert "--label" in run and f"{egress_proxy._EGRESS_LABEL}=1" in run
    assert run[run.index("--network") + 1] == "net-x"
    assert "EGRESS_ALLOWLIST=github.com" in run
    assert ("network", "connect", "bridge", "proxy-x") in calls


def test_start_proxy_connect_failure_fails_closed(monkeypatch):
    """No outbound connectivity would mean 502s forever — raise + clean up."""
    calls = []

    def fake(*args, **kw):
        calls.append(args)
        if args[0] == "inspect":
            return _cp(1)
        if args[:2] == ("network", "connect"):
            return _cp(1, err="no such network: bridge")
        return _cp(0)

    monkeypatch.setattr(egress_proxy, "_docker", fake)
    monkeypatch.setattr(egress_proxy, "find_docker", lambda: "/usr/local/bin/docker")
    with pytest.raises(RuntimeError, match="no such network"):
        egress_proxy._start_proxy("net-x", "proxy-x", [])
    assert ("rm", "-f", "proxy-x") in calls


def test_start_proxy_already_connected_is_ok(monkeypatch):
    def fake(*args, **kw):
        if args[0] == "inspect":
            return _cp(1)
        if args[:2] == ("network", "connect"):
            return _cp(1, err="endpoint proxy-x already exists in network bridge")
        return _cp(0)

    monkeypatch.setattr(egress_proxy, "_docker", fake)
    monkeypatch.setattr(egress_proxy, "find_docker", lambda: "/usr/local/bin/docker")
    egress_proxy._start_proxy("net-x", "proxy-x", [])  # no raise


def test_outbound_network_resolution(monkeypatch):
    monkeypatch.setattr(egress_proxy, "find_docker", lambda: "/usr/local/bin/docker")
    assert egress_proxy._outbound_network() == "bridge"
    monkeypatch.setattr(egress_proxy, "find_docker", lambda: "/opt/homebrew/bin/podman")
    assert egress_proxy._outbound_network() == "podman"
    monkeypatch.setattr(egress_proxy, "find_docker", lambda: None)
    assert egress_proxy._outbound_network() == "bridge"


def test_ensure_allowlisted_network_happy_path(monkeypatch):
    def fake(*args, **kw):
        if args[0] == "inspect" or args[:2] == ("network", "inspect"):
            return _cp(1)  # nothing exists yet
        return _cp(0)

    monkeypatch.setattr(egress_proxy, "_docker", fake)
    monkeypatch.setattr(egress_proxy, "find_docker", lambda: "/usr/local/bin/docker")
    result = egress_proxy.ensure_allowlisted_network(["github.com"])
    expected_network, _ = egress_proxy._names(["github.com"])
    assert result.network == expected_network
    assert result.proxy_env["HTTP_PROXY"].endswith(f":{egress_proxy.PROXY_PORT}")


def test_prune_removes_unattached_proxy(monkeypatch):
    calls = []

    def fake(*args, **kw):
        calls.append(args)
        if args[:2] == ("network", "ls"):
            return _cp(0, out="hermes-egress-abc123\n")
        if args[:2] == ("network", "inspect"):
            return _cp(0, out="hermes-egress-proxy-abc123 ")
        return _cp(0)

    monkeypatch.setattr(egress_proxy, "_docker", fake)
    assert egress_proxy.prune_egress_proxies() == 1
    assert ("rm", "-f", "hermes-egress-proxy-abc123") in calls
    assert ("network", "rm", "hermes-egress-abc123") in calls


def test_prune_keeps_networks_still_in_use(monkeypatch):
    calls = []

    def fake(*args, **kw):
        calls.append(args)
        if args[:2] == ("network", "ls"):
            return _cp(0, out="hermes-egress-abc123\n")
        if args[:2] == ("network", "inspect"):
            return _cp(0, out="hermes-egress-proxy-abc123 hermes-sandbox-1 ")
        return _cp(0)

    monkeypatch.setattr(egress_proxy, "_docker", fake)
    assert egress_proxy.prune_egress_proxies() == 0
    assert not any(c[0] == "rm" for c in calls)
    assert not any(c[:2] == ("network", "rm") for c in calls)


def test_prune_survives_ls_failure(monkeypatch):
    monkeypatch.setattr(egress_proxy, "_docker", lambda *a, **k: _cp(1, err="daemon down"))
    assert egress_proxy.prune_egress_proxies() == 0


# ── proxy server integration (real HTTP server, loopback only) ────


@contextlib.contextmanager
def _running_proxy(allowlist):
    """Run _ProxyHandler on an ephemeral loopback port with a given allowlist."""
    old = eps._ALLOWLIST
    eps._ALLOWLIST = allowlist
    server = ThreadingHTTPServer(("127.0.0.1", 0), eps._ProxyHandler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        eps._ALLOWLIST = old


def _connect_via_proxy(proxy_port: int, authority: str) -> bytes:
    """Send a raw CONNECT and return the first response chunk."""
    with socket.create_connection(("127.0.0.1", proxy_port), timeout=10) as s:
        s.sendall(f"CONNECT {authority} HTTP/1.1\r\nHost: {authority}\r\n\r\n".encode())
        return s.recv(4096)


def test_proxy_denies_connect_to_unlisted_host():
    with _running_proxy(["github.com"]) as port:
        resp = _connect_via_proxy(port, "evil.com:443")
    assert resp.split(b"\r\n", 1)[0].split(b" ")[1] == b"403"


def test_proxy_denies_plain_http_to_unlisted_host():
    with _running_proxy(["github.com"]) as port:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        conn.request("GET", "http://evil.com/data")
        resp = conn.getresponse()
        assert resp.status == 403
        conn.close()


def test_proxy_tunnels_connect_to_allowed_host():
    """CONNECT to an allowlisted host must blind-tunnel bytes both ways."""
    echo = socket.socket()
    echo.bind(("127.0.0.1", 0))
    echo.listen(1)
    echo_port = echo.getsockname()[1]

    def _serve_one():
        conn, _ = echo.accept()
        with conn:
            conn.sendall(conn.recv(1024))

    threading.Thread(target=_serve_one, daemon=True).start()
    try:
        with _running_proxy(["127.0.0.1"]) as port:
            with socket.create_connection(("127.0.0.1", port), timeout=10) as s:
                s.sendall(f"CONNECT 127.0.0.1:{echo_port} HTTP/1.1\r\n\r\n".encode())
                established = s.recv(4096)
                assert b" 200 " in established.split(b"\r\n", 1)[0] + b" "
                s.sendall(b"ping-through-tunnel")
                assert s.recv(1024) == b"ping-through-tunnel"
    finally:
        echo.close()


def test_proxy_forwards_plain_http_to_allowed_host():
    class _Upstream(BaseHTTPRequestHandler):
        def do_GET(self):
            body = b"hello-from-upstream"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    upstream.daemon_threads = True
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    up_port = upstream.server_address[1]
    try:
        with _running_proxy(["127.0.0.1"]) as port:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
            conn.request("GET", f"http://127.0.0.1:{up_port}/")
            resp = conn.getresponse()
            assert resp.status == 200
            assert resp.read() == b"hello-from-upstream"
            conn.close()
    finally:
        upstream.shutdown()
        upstream.server_close()


def test_proxy_returns_502_when_upstream_unreachable():
    # Grab a port that nothing is listening on.
    tmp = socket.socket()
    tmp.bind(("127.0.0.1", 0))
    closed_port = tmp.getsockname()[1]
    tmp.close()

    with _running_proxy(["127.0.0.1"]) as port:
        resp = _connect_via_proxy(port, f"127.0.0.1:{closed_port}")
    assert resp.split(b"\r\n", 1)[0].split(b" ")[1] == b"502"


def test_expected_network_name_matches_provisioning():
    """expected_network_name must be pure and agree with the provisioning
    naming (same canonicalization + hash), so the Docker reuse guard can
    validate a persisted container without touching docker."""
    assert egress_proxy.expected_network_name(["B.com", " a.com "]) == (
        egress_proxy.expected_network_name(["a.com", "b.com"])
    )
    name = egress_proxy.expected_network_name(["github.com"])
    assert name.startswith(egress_proxy.EGRESS_NETWORK_PREFIX)
    assert name == egress_proxy._names(["github.com"])[0]
    # Different allowlists must map to different networks (stale-container detection).
    assert name != egress_proxy.expected_network_name(["pypi.org"])


# ── config wiring: the settings must actually reach the sandbox layer ──


def test_get_env_config_reads_egress_settings(monkeypatch):
    import tools.terminal_tool as terminal_tool

    monkeypatch.setenv("TERMINAL_CONTAINER_NETWORK", "allowlist")
    monkeypatch.setenv("TERMINAL_CONTAINER_NETWORK_ALLOWLIST", '["github.com", "pypi.org"]')
    cfg = terminal_tool._get_env_config()
    assert cfg["container_network"] == "allowlist"
    assert cfg["container_network_allowlist"] == ["github.com", "pypi.org"]


def test_get_env_config_egress_defaults(monkeypatch):
    import tools.terminal_tool as terminal_tool

    monkeypatch.delenv("TERMINAL_CONTAINER_NETWORK", raising=False)
    monkeypatch.delenv("TERMINAL_CONTAINER_NETWORK_ALLOWLIST", raising=False)
    cfg = terminal_tool._get_env_config()
    assert cfg["container_network"] == "on"
    assert cfg["container_network_allowlist"] == []


# ── creation paths: all three sandbox creators must wire egress settings ──


def _drive_creation_path(monkeypatch, invoke):
    """Run *invoke* with egress env vars set and ``_DockerEnvironment`` faked,
    returning the kwargs that actually reached the DockerEnvironment class.

    Exercises the real ``_get_env_config()`` → ``container_config`` →
    ``_create_environment`` chain, so a creation path that drops the egress
    keys from its ``container_config`` dict fails here with
    ``network_mode=None`` instead of shipping a full-network sandbox.
    """
    import tools.terminal_tool as tt

    captured = {}

    class _DummyEnv:
        cwd = "/root"
        cwd_owner = None

        def execute(self, *args, **kwargs):
            return {"output": "", "exit_code": 0}

    def _fake_docker_env(**kwargs):
        captured.update(kwargs)
        return _DummyEnv()

    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_CONTAINER_NETWORK", "allowlist")
    monkeypatch.setenv("TERMINAL_CONTAINER_NETWORK_ALLOWLIST", '["github.com"]')
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    monkeypatch.setattr(tt, "_DockerEnvironment", _fake_docker_env)
    monkeypatch.setattr(tt, "_maybe_reap_docker_orphans", lambda cc: None)
    monkeypatch.setattr(tt, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(tt, "_active_environments", {})
    monkeypatch.setattr(tt, "_last_activity", {})
    invoke()
    assert captured, "creation path never reached DockerEnvironment"
    return captured


def test_terminal_path_egress_reaches_docker_environment(monkeypatch):
    import tools.terminal_tool as tt

    captured = _drive_creation_path(
        monkeypatch,
        lambda: tt.terminal_tool(command="pwd", task_id="egress-term", force=True),
    )
    assert captured["network_mode"] == "allowlist"
    assert captured["network_allowlist"] == ["github.com"]


def test_code_execution_path_egress_reaches_docker_environment(monkeypatch):
    from tools.code_execution_tool import _get_or_create_env

    captured = _drive_creation_path(
        monkeypatch, lambda: _get_or_create_env("egress-exec"),
    )
    assert captured["network_mode"] == "allowlist"
    assert captured["network_allowlist"] == ["github.com"]


def test_file_tools_path_egress_reaches_docker_environment(monkeypatch):
    import tools.file_tools as ft

    monkeypatch.setattr(ft, "_file_ops_cache", {})
    captured = _drive_creation_path(
        monkeypatch, lambda: ft._get_file_ops("egress-files"),
    )
    assert captured["network_mode"] == "allowlist"
    assert captured["network_allowlist"] == ["github.com"]


def test_cli_bridge_exports_egress_env_vars(tmp_path, monkeypatch):
    """Regression guard for the config.yaml → env-var bridge (CLI startup).

    The sandbox layer once shipped fully unit-tested but inert, because the
    startup bridges never mapped container_network*. Drive the real
    ``cli.load_cli_config()`` and assert the env vars terminal_tool reads
    actually get set (the gateway bridge has its own behavioral test in
    tests/gateway/test_config_env_bridge_authority.py).
    """
    import json
    import os
    from unittest import mock

    import yaml

    import cli

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(yaml.safe_dump({
        "terminal": {
            "backend": "docker",
            "container_network": "allowlist",
            "container_network_allowlist": ["github.com", "pypi.org"],
        },
    }))

    # load_cli_config bridges many terminal keys into os.environ; snapshot
    # and restore the whole environment so nothing leaks into other tests.
    with mock.patch.dict(os.environ):
        os.environ["HERMES_HOME"] = str(hermes_home)
        os.environ.pop("TERMINAL_CONTAINER_NETWORK", None)
        os.environ.pop("TERMINAL_CONTAINER_NETWORK_ALLOWLIST", None)
        monkeypatch.setattr(cli, "_hermes_home", hermes_home)
        cli.load_cli_config()

        assert os.environ["TERMINAL_CONTAINER_NETWORK"] == "allowlist"
        assert json.loads(os.environ["TERMINAL_CONTAINER_NETWORK_ALLOWLIST"]) == [
            "github.com", "pypi.org",
        ]
