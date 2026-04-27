"""
Tests for tools/http_tools.py retry helpers.
"""
import time
import httpx
import pytest

from tools.http_tools import (
    retryable_http_call,
    retryable_get,
    retryable_post,
    DEFAULT_RETRY_CONFIG,
)


# -----------------------------------------------------------------------
# Dummy clients / helpers
# -----------------------------------------------------------------------

class DummyClient:
    """Very small stub that mimics enough of httpx.Client for these tests."""
    def __init__(self, status=200, fail_exc=None, json_body=None):
        self.status = status
        self.fail_exc = fail_exc
        self.json_body = json_body or {}
        self.post_calls = []
        self.get_calls = []

    def post(self, url, **kwargs):
        self.post_calls.append((url, kwargs))
        if self.fail_exc:
            raise self.fail_exc
        return _dummy_resp(self.status, self.json_body)

    def get(self, url, **kwargs):
        self.get_calls.append((url, kwargs))
        if self.fail_exc:
            raise self.fail_exc
        return _dummy_resp(self.status, self.json_body)


class _dummy_resp:
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


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------

class TestRetryableHttpCall:
    def test_success_first_try(self):
        client = DummyClient(status=200, json_body={"ok": True})
        resp = retryable_http_call(lambda: client.post("http://x"))
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}
        assert len(client.post_calls) == 1

    def test_retryable_exception_retries(self):
        # Simulate three ConnectErrors then success
        excs = [
            httpx.ConnectError("fail1"),
            httpx.ConnectError("fail2"),
            httpx.ConnectError("fail3"),
        ]
        calls = []

        def fn():
            idx = len(calls)
            calls.append(idx)
            if idx < 3:
                raise excs[idx]
            return _dummy_resp(200, {"recovered": True})

        start = time.time()
        resp = retryable_http_call(
            fn,
            max_attempts=5,
            base_delay=0.01,
            backoff_factor=2.0,
            max_delay=1.0,
        )
        elapsed = time.time() - start

        assert resp.status_code == 200
        assert resp.json() == {"recovered": True}
        # 3 failures + 1 success = 4 calls
        assert len(calls) == 4
        # Should have slept between failures (elapsed > tiny)
        assert elapsed > 0.02

    def test_non_retryable_exception_no_retry(self):
        def fn():
            raise ValueError("non-retryable")

        with pytest.raises(ValueError, match="non-retryable"):
            retryable_http_call(fn, max_attempts=5, base_delay=0.01)

    def test_max_attempts_exhausted(self):
        def fn():
            raise httpx.ReadTimeout("timeout")

        with pytest.raises(httpx.ReadTimeout):
            retryable_http_call(fn, max_attempts=2, base_delay=0.01)


class TestRetryableGet:
    def test_passes_kwargs_through(self):
        client = DummyClient(status=200, json_body={"x": 1})
        resp = retryable_get(
            client,
            "http://example.com",
            params={"q": "test"},
            headers={"X-Custom": "val"},
            timeout=2.5,
            max_attempts=1,
        )
        assert resp.status_code == 200
        assert len(client.get_calls) == 1
        _, kwargs = client.get_calls[0]
        assert kwargs["params"] == {"q": "test"}
        assert kwargs["headers"] == {"X-Custom": "val"}
        assert kwargs["timeout"] == 2.5


class TestRetryablePost:
    def test_data_vs_json_prefer_data(self):
        client = DummyClient(status=201, json_body={"id": 5})
        resp = retryable_post(
            client,
            "http://example.com",
            json={"a": 1},
            data={"b": 2},
            max_attempts=1,
        )
        assert resp.status_code == 201
        # Should pick 'data' when both provided
        _, kwargs = client.post_calls[0]
        assert "data" in kwargs
        assert kwargs.get("json") is None

    def test_headers_timeout_omitted_when_none(self):
        client = DummyClient(status=200)
        retryable_post(
            client,
            "http://example.com",
            data={"x": 1},
            max_attempts=1,
        )
        _, kwargs = client.post_calls[0]
        # headers/timeout should not be passed if None
        assert "headers" not in kwargs
        assert "timeout" not in kwargs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
