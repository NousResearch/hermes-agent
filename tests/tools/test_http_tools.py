"""Tests for tools/http_tools.py retry helpers."""

import urllib.error

import httpx
import pytest

from tools.http_tools import retryable_get


# -----------------------------------------------------------------------
# Dummy clients / helpers
# -----------------------------------------------------------------------

class DummyClient:
    """Very small stub that mimics enough of httpx.Client for these tests."""
    def __init__(self, status=200, fail_exc=None, json_body=None):
        self.status = status
        self.fail_exc = fail_exc
        self.json_body = json_body or {}
        self.get_calls = []

    def get(self, url, **kwargs):
        self.get_calls.append((url, kwargs))
        if self.fail_exc:
            raise self.fail_exc
        return _DummyResponse(self.status, self.json_body)


class _DummyResponse:
    def __init__(self, status, json_body):
        self.status_code = status
        self._json = json_body

    def json(self):
        return self._json

    def raise_for_status(self):
        if 400 <= self.status_code < 600:
            raise httpx.HTTPStatusError(
                "bad status", request=None, response=self
            )


class TestRetryableGet:
    def test_success_first_try(self):
        client = DummyClient(status=200, json_body={"ok": True})
        resp = retryable_get(lambda: client.get("http://x"))
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}
        assert len(client.get_calls) == 1

    def test_retryable_exception_retries(self, monkeypatch):
        excs = [
            httpx.ConnectError("fail1"),
            httpx.ConnectError("fail2"),
            httpx.ConnectError("fail3"),
        ]
        calls = []
        sleeps = []

        def fn():
            idx = len(calls)
            calls.append(idx)
            if idx < 3:
                raise excs[idx]
            return _DummyResponse(200, {"recovered": True})

        monkeypatch.setattr("tools.http_tools.time.sleep", sleeps.append)
        resp = retryable_get(
            fn,
            max_attempts=5,
            base_delay=0.01,
            backoff_factor=2.0,
            max_delay=1.0,
        )

        assert resp.status_code == 200
        assert resp.json() == {"recovered": True}
        assert len(calls) == 4
        assert sleeps == [0.01, 0.02, 0.04]

    def test_non_retryable_exception_no_retry(self):
        calls = 0

        def fn():
            nonlocal calls
            calls += 1
            raise ValueError("non-retryable")

        with pytest.raises(ValueError, match="non-retryable"):
            retryable_get(fn, max_attempts=5, base_delay=0)
        assert calls == 1

    def test_max_attempts_exhausted(self):
        calls = 0

        def fn():
            nonlocal calls
            calls += 1
            raise httpx.ReadTimeout("timeout")

        with pytest.raises(httpx.ReadTimeout):
            retryable_get(fn, max_attempts=2, base_delay=0)
        assert calls == 2

    def test_http_status_error_is_not_retried(self):
        calls = 0

        def fn():
            nonlocal calls
            calls += 1
            raise httpx.HTTPStatusError("bad status", request=None, response=None)

        with pytest.raises(httpx.HTTPStatusError):
            retryable_get(fn, max_attempts=3, base_delay=0)
        assert calls == 1

    def test_urllib_http_error_is_not_retried(self):
        calls = 0

        def fn():
            nonlocal calls
            calls += 1
            raise urllib.error.HTTPError(
                "https://example.com",
                503,
                "unavailable",
                {},
                None,
            )

        with pytest.raises(urllib.error.HTTPError):
            retryable_get(fn, max_attempts=3, base_delay=0)
        assert calls == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
