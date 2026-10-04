"""429 rate-limit retry in the Discord REST tool."""

import json
import sys
import types
import urllib.error
from io import BytesIO
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import tools.discord_tool as dt


def _http_error(status, body, headers=None):
    return urllib.error.HTTPError(
        url="https://discord.com/api/v10/x", code=status, msg="err",
        hdrs=types.SimpleNamespace(get=lambda k, d=None: (headers or {}).get(k, d)),
        fp=BytesIO(json.dumps(body).encode() if isinstance(body, dict) else body.encode()))


class _Resp:
    def __init__(self, payload, status=200):
        self.status = status
        self._b = json.dumps(payload).encode()

    def read(self, n=-1):
        return self._b[:n] if n and n > 0 else self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture(autouse=True)
def _reset_global_cooldown():
    dt._global_rate_limit_until[0] = 0.0
    yield
    dt._global_rate_limit_until[0] = 0.0


class TestRetryAfterParsing:
    def test_prefers_json_body_over_header(self):
        # The header rounds to whole seconds; the body is sub-second precise.
        hdrs = types.SimpleNamespace(get=lambda k, d=None: {"Retry-After": "1"}.get(k, d))
        assert dt._parse_retry_after(hdrs, '{"retry_after": 0.15}') == pytest.approx(0.15)

    def test_falls_back_to_header_when_body_unparseable(self):
        hdrs = types.SimpleNamespace(get=lambda k, d=None: {"Retry-After": "2"}.get(k, d))
        assert dt._parse_retry_after(hdrs, "not json") == pytest.approx(2.0)

    def test_returns_none_when_neither_present(self):
        hdrs = types.SimpleNamespace(get=lambda k, d=None: d)
        assert dt._parse_retry_after(hdrs, "") is None

    def test_negative_delay_clamped_to_zero(self):
        assert dt._parse_retry_after(None, '{"retry_after": -5}') == 0.0


class TestRateLimitRetry:
    def test_retries_after_429_then_succeeds(self, monkeypatch):
        calls = []
        slept = []
        monkeypatch.setattr(dt.time, "sleep", lambda s: slept.append(s))

        def fake_open(req, timeout=None):
            calls.append(req.full_url)
            if len(calls) == 1:
                raise _http_error(429, {"retry_after": 0.25, "global": False})
            return _Resp({"ok": True})

        monkeypatch.setattr(dt.urllib.request, "urlopen", fake_open)
        assert dt._discord_request("GET", "/x", "tok") == {"ok": True}
        assert len(calls) == 2
        assert slept == [pytest.approx(0.25)]

    def test_gives_up_after_max_retries(self, monkeypatch):
        monkeypatch.setattr(dt.time, "sleep", lambda s: None)
        monkeypatch.setattr(
            dt.urllib.request, "urlopen",
            lambda req, timeout=None: (_ for _ in ()).throw(_http_error(429, {"retry_after": 0.01})))
        with pytest.raises(dt.DiscordAPIError) as exc:
            dt._discord_request("GET", "/x", "tok")
        assert exc.value.status == 429
        assert "after 3 retries" in exc.value.body

    def test_delay_above_cap_is_not_slept(self, monkeypatch):
        """A 600s thread-rename cooldown must surface as an error, not stall the turn."""
        slept = []
        monkeypatch.setattr(dt.time, "sleep", lambda s: slept.append(s))
        monkeypatch.setattr(
            dt.urllib.request, "urlopen",
            lambda req, timeout=None: (_ for _ in ()).throw(_http_error(429, {"retry_after": 600})))
        with pytest.raises(dt.DiscordAPIError) as exc:
            dt._discord_request("PATCH", "/channels/1", "tok")
        assert "above the" in exc.value.body
        assert slept == []

    def test_non_429_error_is_not_retried(self, monkeypatch):
        calls = []

        def fake_open(req, timeout=None):
            calls.append(1)
            raise _http_error(403, {"message": "Missing Permissions"})

        monkeypatch.setattr(dt.urllib.request, "urlopen", fake_open)
        with pytest.raises(dt.DiscordAPIError) as exc:
            dt._discord_request("GET", "/x", "tok")
        assert exc.value.status == 403
        assert len(calls) == 1

    def test_global_429_sets_process_wide_cooldown(self, monkeypatch):
        monkeypatch.setattr(dt.time, "sleep", lambda s: None)
        seq = [_http_error(429, {"retry_after": 0.2, "global": True}), _Resp({"ok": True})]

        def fake_open(req, timeout=None):
            item = seq.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        monkeypatch.setattr(dt.urllib.request, "urlopen", fake_open)
        dt._discord_request("GET", "/x", "tok")
        assert dt._global_rate_limit_until[0] > 0

    def test_body_is_resent_on_retry(self, monkeypatch):
        """The payload must survive the retry; a consumed buffer would send an empty body."""
        bodies = []
        monkeypatch.setattr(dt.time, "sleep", lambda s: None)

        def fake_open(req, timeout=None):
            bodies.append(req.data)
            if len(bodies) == 1:
                raise _http_error(429, {"retry_after": 0.01})
            return _Resp({"ok": True})

        monkeypatch.setattr(dt.urllib.request, "urlopen", fake_open)
        dt._discord_request("POST", "/x", "tok", body={"name": "hi"})
        assert bodies[0] == bodies[1] == b'{"name": "hi"}'
