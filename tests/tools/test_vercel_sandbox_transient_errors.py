"""Vercel sandbox retries treat transport failures from the SDK's HTTP stack as transient.

vercel >= 0.12 raises httpx2 exceptions, which do not subclass httpx's.
"""

import socket

import httpx
import pytest

from tools.environments.vercel_sandbox import _is_transient_vercel_error, _retry_vercel_call

httpx2 = pytest.importorskip("httpx2")


def _refused_url() -> str:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}/"


@pytest.mark.parametrize("client_mod", [httpx, httpx2], ids=["httpx", "httpx2"])
def test_connection_refused_is_transient(client_mod):
    with pytest.raises(Exception) as caught, client_mod.Client(timeout=5) as client:
        client.get(_refused_url())

    assert _is_transient_vercel_error(caught.value)


def test_wrapped_httpx2_transport_error_is_transient():
    try:
        try:
            raise httpx2.ReadError("connection reset")
        except httpx2.ReadError as inner:
            raise RuntimeError("sandbox create failed") from inner
    except RuntimeError as outer:
        assert _is_transient_vercel_error(outer)


def test_httpx2_connect_error_is_retried():
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise httpx2.ConnectError("connection refused")
        return "ok"

    assert _retry_vercel_call("sandbox create", flaky, attempts=3) == "ok"
    assert len(calls) == 3


def test_non_transport_error_is_not_retried():
    calls = []

    def broken():
        calls.append(1)
        raise ValueError("bad request payload")

    with pytest.raises(ValueError):
        _retry_vercel_call("sandbox create", broken, attempts=3)
    assert len(calls) == 1
