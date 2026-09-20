"""Offline qualification for the quota-coordinated SearXNG plugin candidate.

These tests exercise real Hermes plugin discovery and native ``web_search``
dispatch.  The test process creates the loopback listener; product/plugin code
never binds.  The canonical runner is launched with a task-owned HOME that
contains the frozen plugin and reviewed quota source under fixed, non-live
paths.  No credential, DNS, public network, or live Hermes profile is used.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import socket
import sqlite3
import ssl
import subprocess
import threading
from collections.abc import Generator
from typing import Any

import pytest
import yaml


PLUGIN_NAME = "quota-coordinator"
PROFILE_NAME = "coding-hermes"
SERVER_NAME = "searxng.quota.invalid"
OPENSSL = Path("/opt/homebrew/bin/openssl")
CANDIDATE_PLUGIN = Path.home() / ".hermes-quota-offline-candidate" / PLUGIN_NAME
REVIEWED_SOURCE = Path.home() / ".hermes-quota-reviewed-source"

pytestmark = pytest.mark.skipif(
    not (CANDIDATE_PLUGIN.is_dir() and REVIEWED_SOURCE.is_dir()),
    reason="task-owned offline quota-provider qualification inputs are absent",
)


@dataclass(frozen=True)
class CertificateSet:
    ca_pem: bytes
    server_cert: Path
    server_key: Path


@dataclass
class Lab:
    profile_home: Path
    run_root: Path
    listener: socket.socket
    manager: Any
    provider: Any | None


class SyntheticTLSServer:
    """Serve a fixed sequence over a pre-bound loopback listener."""

    def __init__(
        self,
        listener: socket.socket,
        certificates: CertificateSet,
        responses: list[bytes],
    ) -> None:
        self.listener = listener
        self.certificates = certificates
        self.responses = list(responses)
        self.requests: list[bytes] = []
        self.connections = 0
        self.error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="quota-provider-offline-tls",
            daemon=True,
        )

    def start(self) -> "SyntheticTLSServer":
        self._thread.start()
        return self

    def _run(self) -> None:
        try:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            context.load_cert_chain(
                certfile=str(self.certificates.server_cert),
                keyfile=str(self.certificates.server_key),
            )
            self.listener.settimeout(5.0)
            for response in self.responses:
                raw, _peer = self.listener.accept()
                self.connections += 1
                with raw:
                    with context.wrap_socket(raw, server_side=True) as tls:
                        request = bytearray()
                        while b"\r\n\r\n" not in request:
                            chunk = tls.recv(4096)
                            if not chunk:
                                break
                            request.extend(chunk)
                            if len(request) > 65536:
                                raise AssertionError("synthetic request exceeded bound")
                        self.requests.append(bytes(request))
                        tls.sendall(response)
        except BaseException as exc:  # surfaced deterministically by join()
            self.error = exc

    def join(self, timeout: float = 8.0) -> None:
        self._thread.join(timeout)
        assert not self._thread.is_alive(), "synthetic TLS server did not stop"
        if self.error is not None:
            raise AssertionError(f"synthetic TLS server failed: {self.error!r}")


def _openssl(root: Path, *args: str) -> None:
    completed = subprocess.run(
        [str(OPENSSL), *args],
        cwd=root,
        env={"HOME": str(root), "PATH": "/usr/bin:/bin"},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=20,
        check=False,
    )
    if completed.returncode:
        raise AssertionError(
            "synthetic certificate generation failed: "
            + completed.stderr.decode("utf-8", "replace")[-1000:]
        )


@pytest.fixture(scope="module")
def certificates(tmp_path_factory) -> Generator[CertificateSet, None, None]:
    assert OPENSSL.is_file(), "offline OpenSSL executable unavailable"
    root = tmp_path_factory.mktemp("quota-provider-certs")
    os.chmod(root, 0o700)
    ca_key = root / "ca.key"
    ca_pem = root / "ca.pem"
    server_key = root / "server.key"
    server_csr = root / "server.csr"
    server_cert = root / "server.pem"
    server_ext = root / "server.ext"
    try:
        _openssl(
            root,
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-sha256",
            "-days",
            "1",
            "-subj",
            "/CN=Hermes Offline Quota Test CA",
            "-addext",
            "basicConstraints=critical,CA:TRUE",
            "-addext",
            "keyUsage=critical,keyCertSign,cRLSign",
            "-keyout",
            ca_key.name,
            "-out",
            ca_pem.name,
        )
        server_ext.write_text(
            "basicConstraints=critical,CA:FALSE\n"
            "keyUsage=critical,digitalSignature,keyEncipherment\n"
            "extendedKeyUsage=serverAuth\n"
            f"subjectAltName=DNS:{SERVER_NAME}\n",
            encoding="utf-8",
        )
        _openssl(
            root,
            "req",
            "-new",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-sha256",
            "-subj",
            f"/CN={SERVER_NAME}",
            "-keyout",
            server_key.name,
            "-out",
            server_csr.name,
        )
        _openssl(
            root,
            "x509",
            "-req",
            "-sha256",
            "-days",
            "1",
            "-in",
            server_csr.name,
            "-CA",
            ca_pem.name,
            "-CAkey",
            ca_key.name,
            "-set_serial",
            "1001",
            "-extfile",
            server_ext.name,
            "-out",
            server_cert.name,
        )
        ca_bytes = ca_pem.read_bytes()
        ca_key.unlink()
        server_csr.unlink()
        server_ext.unlink()
        yield CertificateSet(ca_bytes, server_cert, server_key)
    finally:
        for key in root.glob("*.key"):
            key.unlink(missing_ok=True)


def _reset_runtime_state() -> None:
    import hermes_constants
    from agent import web_required_provider, web_search_registry
    from hermes_cli import plugins

    plugins._reset_plugin_managers_for_tests()
    web_search_registry._reset_for_tests()
    web_required_provider._bindings.clear()
    hermes_constants.reset_hermes_home_key_cache()
    hermes_constants._default_hermes_root_memo = None


def _response(body: bytes) -> bytes:
    return (
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: application/json; charset=utf-8\r\n"
        + f"Content-Length: {len(body)}\r\n".encode("ascii")
        + b"Connection: close\r\n\r\n"
        + body
    )


def _success_body(count: int = 2) -> bytes:
    return json.dumps(
        {
            "query": "synthetic",
            "number_of_results": count,
            "results": [
                {
                    "title": f"Result {index}",
                    "url": f"https://example.com/result-{index}",
                    "content": f"Synthetic snippet {index}",
                    "engine": "offline-fixture",
                }
                for index in range(1, count + 1)
            ],
            "answers": [],
            "corrections": [],
            "infoboxes": [],
            "suggestions": [],
            "unresponsive_engines": [],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _expected_result(count: int = 2) -> dict[str, Any]:
    return {
        "success": True,
        "data": {
            "web": [
                {
                    "title": f"Result {index}",
                    "url": f"https://example.com/result-{index}",
                    "description": f"Synthetic snippet {index}",
                    "position": index,
                }
                for index in range(1, count + 1)
            ]
        },
    }


def _install(
    tmp_path: Path,
    monkeypatch,
    certificates: CertificateSet,
    *,
    profile_name: str = PROFILE_NAME,
    quota_source: Path = REVIEWED_SOURCE,
    configured_port: int | None = None,
) -> Lab:
    assert CANDIDATE_PLUGIN.is_dir(), CANDIDATE_PLUGIN
    assert REVIEWED_SOURCE.is_dir(), REVIEWED_SOURCE

    _reset_runtime_state()
    default_root = Path.home() / ".hermes"
    profile_home = default_root / "profiles" / profile_name
    if profile_home.exists():
        shutil.rmtree(profile_home)
    profile_home.mkdir(mode=0o700, parents=True)
    plugins_root = profile_home / "plugins"
    plugins_root.mkdir(mode=0o700)
    shutil.copytree(CANDIDATE_PLUGIN, plugins_root / PLUGIN_NAME)

    run_root = profile_home / "offline-quota-run"
    run_root.mkdir(mode=0o700)
    ca_path = run_root / "ca.pem"
    ca_path.write_bytes(certificates.ca_pem)
    os.chmod(ca_path, 0o600)

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)
    actual_port = listener.getsockname()[1]
    port = actual_port if configured_port is None else configured_port

    empty_bundled = tmp_path / "empty-bundled"
    empty_bundled.mkdir()
    config = {
        "plugins": {
            "enabled": [PLUGIN_NAME],
            "entries": {
                PLUGIN_NAME: {
                    "settings": {
                        "mode": "offline-synthetic-loopback-v1",
                        "quota_source_root": str(quota_source),
                        "run_root": str(run_root),
                        "ca_pem_path": str(ca_path),
                        "listener_fd": listener.fileno(),
                        "listener_port": port,
                        "request_timeout": 1.5,
                        "max_response_bytes": 16384,
                        "max_results": 5,
                    }
                }
            },
        },
        "web": {
            "required_provider": PLUGIN_NAME,
            "backend": PLUGIN_NAME,
            "cache_enabled": True,
            "keyless_rescue": True,
            "keyless_fallback": True,
        },
    }
    (profile_home / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=True), encoding="utf-8"
    )

    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(empty_bundled))
    monkeypatch.setenv("HERMES_ENABLE_PROJECT_PLUGINS", "0")
    _reset_runtime_state()

    from agent import web_search_registry
    from hermes_cli.plugins import get_plugin_manager

    manager = get_plugin_manager()
    manager.discover_and_load()
    provider = web_search_registry.get_provider(PLUGIN_NAME)
    return Lab(profile_home, run_root, listener, manager, provider)


def _close_lab(lab: Lab) -> None:
    try:
        lab.manager.unload()
    finally:
        lab.listener.close()
        _reset_runtime_state()
        shutil.rmtree(lab.profile_home, ignore_errors=True)


def _plugin_error(lab: Lab) -> str:
    loaded = lab.manager._plugins.get(PLUGIN_NAME)
    assert loaded is not None, lab.manager._plugins
    return str(loaded.error or "")


def test_real_discovery_native_dispatch_and_durable_settlement(
    tmp_path, monkeypatch, certificates
):
    from tools.web_tools import check_web_api_key, web_search_tool

    lab = _install(tmp_path, monkeypatch, certificates)
    try:
        assert lab.provider is not None, _plugin_error(lab)
        # The real Hermes registry uses one readiness gate for web_search and
        # web_extract.  This search-only provider must expose web_search while
        # unsupported extraction remains fail-closed.
        assert check_web_api_key() is True
        raw_response = _response(_success_body())
        server = SyntheticTLSServer(
            lab.listener, certificates, [raw_response, raw_response]
        ).start()

        first = json.loads(web_search_tool("mountain boots", limit=2))
        second = json.loads(web_search_tool("mountain boots", limit=2))
        server.join()

        assert first == second == _expected_result()
        assert server.connections == 2
        assert len(server.requests) == 2
        expected_line = (
            b"GET /search?q=mountain+boots&format=json&language=en&safesearch=1 "
            b"HTTP/1.1"
        )
        assert all(
            request.split(b"\r\n", 1)[0] == expected_line for request in server.requests
        )
        assert all(
            f"Host: {SERVER_NAME}\r\n".encode() in request
            for request in server.requests
        )

        with sqlite3.connect(lab.run_root / "ledger.db") as connection:
            request_rows = connection.execute(
                "SELECT execution,delivery,dispatch_count FROM requests ORDER BY rowid"
            ).fetchall()
            permit_rows = connection.execute(
                "SELECT phase,accounting,settlement FROM permits ORDER BY rowid"
            ).fetchall()
        assert request_rows == [("succeeded", "delivered", 1)] * 2
        # ``phase`` is immutable evidence that the effect started; durable closure
        # is represented by accounting+settlement, not by erasing that evidence.
        assert permit_rows == [("started", "settled", "success")] * 2
        assert list(lab.profile_home.rglob("*.key")) == []

        assert lab.manager.unload(PLUGIN_NAME)
        dns_calls: list[tuple[Any, ...]] = []

        def refuse_dns(*args, **kwargs):
            dns_calls.append(args)
            raise AssertionError("required-provider failure must precede DNS")

        monkeypatch.setattr(socket, "getaddrinfo", refuse_dns)
        missing = json.loads(web_search_tool("must not escape", limit=1))
        assert missing.get("success") is not True and missing.get("error"), missing
        assert "required_web_provider" in missing["error"]
        assert dns_calls == []
    finally:
        _close_lab(lab)


class _SyntheticUnderlying:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.closed = False

    def is_available(self) -> bool:
        return not self.closed

    def search(self, query: str, limit: int) -> Any:
        return self.response

    def cancel_active(self) -> bool:
        return False

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize(
    ("response", "expected_error"),
    [
        (
            {
                "success": True,
                "results": [{"title": "missing fields"}],
                "source": PLUGIN_NAME,
            },
            "outcome_unknown",
        ),
        (
            {
                "success": False,
                "error": "provider_busy",
                "results": [],
                "source": PLUGIN_NAME,
            },
            "provider_busy",
        ),
    ],
)
def test_wrapper_preserves_only_reviewed_envelopes(
    tmp_path, monkeypatch, certificates, response, expected_error
):
    from tools.web_tools import web_search_tool

    lab = _install(tmp_path, monkeypatch, certificates)
    try:
        assert lab.provider is not None, _plugin_error(lab)
        original = lab.provider._runtime.underlying
        original.close()
        lab.provider._runtime.underlying = _SyntheticUnderlying(response)

        result = json.loads(web_search_tool("synthetic", limit=1))
        assert result == {"success": False, "error": expected_error}
    finally:
        _close_lab(lab)


def test_tampered_reviewed_source_is_rejected_before_dispatch(
    tmp_path, monkeypatch, certificates
):
    tampered = tmp_path / "tampered-source"
    shutil.copytree(REVIEWED_SOURCE, tampered)
    routing = tampered / "quota_v1" / "routing.py"
    routing.write_bytes(routing.read_bytes() + b"\n")

    lab = _install(
        tmp_path,
        monkeypatch,
        certificates,
        quota_source=tampered,
    )
    try:
        assert lab.provider is None
        assert "source_untrusted" in _plugin_error(lab)
    finally:
        _close_lab(lab)


def test_unpinned_bytecode_cache_is_rejected_before_dispatch(
    tmp_path, monkeypatch, certificates
):
    tampered = tmp_path / "bytecode-source"
    shutil.copytree(REVIEWED_SOURCE, tampered)
    bytecode = tampered / "quota_v1" / "__pycache__" / "routing.cpython-311.pyc"
    bytecode.parent.mkdir(mode=0o700)
    bytecode.write_bytes(b"unpinned-bytecode-must-not-load")

    lab = _install(
        tmp_path,
        monkeypatch,
        certificates,
        quota_source=tampered,
    )
    try:
        assert lab.provider is None
        assert "source_untrusted" in _plugin_error(lab)
    finally:
        _close_lab(lab)


@pytest.mark.parametrize("fault", ["profile", "listener"])
def test_scope_and_listener_pins_fail_closed(
    tmp_path, monkeypatch, certificates, fault
):
    if fault == "profile":
        lab = _install(
            tmp_path,
            monkeypatch,
            certificates,
            profile_name="not-coding-hermes",
        )
        expected = "profile_scope_mismatch"
    else:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.bind(("127.0.0.1", 0))
            wrong_port = probe.getsockname()[1]
        finally:
            probe.close()
        lab = _install(
            tmp_path,
            monkeypatch,
            certificates,
            configured_port=wrong_port,
        )
        expected = "listener_untrusted"
    try:
        assert lab.provider is None
        assert expected in _plugin_error(lab)
    finally:
        _close_lab(lab)
