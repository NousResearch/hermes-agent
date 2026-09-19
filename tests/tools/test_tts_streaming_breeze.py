"""Tests for the Breeze local streaming-TTS provider.

Breeze is a self-hosted warm-GPU server (``breeze-tts``) that serves chunked
``audio/pcm`` from ``POST /v1/audio/speech``. These tests exercise the provider
in isolation — registration/resolution, health-gated availability, the chunked
read path, multipart-vs-urlencoded body selection, and the 409 busy-retry —
with the network fully mocked, so they run without a live server.
"""

import io
import urllib.error

import pytest

import tools.tts_streaming as ts


class _FakeResp:
    """Minimal urlopen() response: context manager + chunked ``read(n)``."""

    def __init__(self, chunks, headers=None):
        self._buf = io.BytesIO(b"".join(chunks))
        self.headers = headers or {}

    def read(self, n=-1):
        return self._buf.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _breeze(section=None):
    return ts.BreezeStreamer(tts_config={}, section=section or {})


# ---------------------------------------------------------------------------
# Registration / resolution
# ---------------------------------------------------------------------------

def test_breeze_is_registered():
    assert ts._REGISTRY.get("breeze") is ts.BreezeStreamer


def test_pinned_breeze_resolves_when_healthy(monkeypatch):
    monkeypatch.setattr(ts, "_breeze_health", lambda url: (True, 24000))
    prov = ts.resolve_streaming_provider({"streaming": {"provider": "breeze"}})
    assert isinstance(prov, ts.BreezeStreamer)


def test_pinned_breeze_declines_when_unreachable(monkeypatch):
    monkeypatch.setattr(ts, "_breeze_health", lambda url: (False, None))
    prov = ts.resolve_streaming_provider({"streaming": {"provider": "breeze"}})
    assert prov is None


def test_breeze_not_in_auto_priority():
    # ``auto`` walks cloud providers only; a local box shouldn't be picked for
    # everyone who opts into "best available".
    assert "breeze" not in ts._PROVIDER_PRIORITY


# ---------------------------------------------------------------------------
# Availability (health-gated)
# ---------------------------------------------------------------------------

def test_available_reflects_health(monkeypatch):
    monkeypatch.setattr(ts, "_breeze_base_url", lambda: "http://x")
    monkeypatch.setattr(ts, "_breeze_health", lambda url: (True, 24000))
    assert ts.BreezeStreamer.available() is True
    monkeypatch.setattr(ts, "_breeze_health", lambda url: (False, None))
    assert ts.BreezeStreamer.available() is False


def test_sample_rate_adopts_server_when_unset(monkeypatch):
    monkeypatch.setattr(ts, "_breeze_health", lambda url: (True, 16000))
    s = _breeze({"base_url": "http://x"})
    assert s.sample_rate == 16000


def test_sample_rate_config_overrides_server(monkeypatch):
    monkeypatch.setattr(ts, "_breeze_health", lambda url: (True, 16000))
    s = _breeze({"base_url": "http://x", "sample_rate": 48000})
    assert s.sample_rate == 48000


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

def test_stream_yields_pcm_chunks(monkeypatch):
    captured = {}

    def _fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["ctype"] = req.headers.get("Content-type")
        captured["body"] = req.data
        return _FakeResp([b"\x01\x02" * 100, b"\x03\x04" * 100])

    monkeypatch.setattr(ts, "_breeze_health", lambda url: (True, 24000))
    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)

    s = _breeze({"base_url": "http://breeze:7860"})
    out = b"".join(s.stream("Hello there."))

    assert len(out) == 400
    assert captured["url"] == "http://breeze:7860/v1/audio/speech"
    # No ref_audio configured → form-urlencoded, text carried in the body.
    assert captured["ctype"] == "application/x-www-form-urlencoded"
    assert b"Hello" in captured["body"]


def test_stream_uses_multipart_with_ref_audio(monkeypatch, tmp_path):
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"RIFFfake")
    captured = {}

    def _fake_urlopen(req, timeout=None):
        captured["ctype"] = req.headers.get("Content-type")
        captured["body"] = req.data
        return _FakeResp([b"\x00\x00" * 10])

    monkeypatch.setattr(ts, "_breeze_health", lambda url: (True, 24000))
    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)

    s = _breeze({
        "base_url": "http://breeze:7860",
        "ref_audio": str(ref),
        "ref_text": "reference transcript",
    })
    list(s.stream("Clone my voice."))

    assert captured["ctype"].startswith("multipart/form-data; boundary=")
    assert b'name="ref_audio"' in captured["body"]
    assert b"reference transcript" in captured["body"]
    assert b"RIFFfake" in captured["body"]


def test_ref_audio_without_ref_text_falls_back_to_urlencoded(monkeypatch, tmp_path):
    # Breeze rejects one without the other; we send text-only rather than error.
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"RIFFfake")
    captured = {}

    def _fake_urlopen(req, timeout=None):
        captured["ctype"] = req.headers.get("Content-type")
        return _FakeResp([b"\x00\x00"])

    monkeypatch.setattr(ts, "_breeze_health", lambda url: (True, 24000))
    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)

    s = _breeze({"base_url": "http://x", "ref_audio": str(ref)})  # no ref_text
    list(s.stream("Hi."))
    assert captured["ctype"] == "application/x-www-form-urlencoded"


def test_stream_retries_on_409_busy(monkeypatch):
    calls = {"n": 0}

    def _fake_urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError(req.full_url, 409, "busy", {}, None)
        return _FakeResp([b"\x07\x08" * 50])

    monkeypatch.setattr(ts, "_breeze_health", lambda url: (True, 24000))
    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    monkeypatch.setattr("time.sleep", lambda *_: None)

    s = _breeze({"base_url": "http://x"})
    out = b"".join(s.stream("Retry me."))
    assert calls["n"] == 2
    assert len(out) == 100


def test_stream_raises_when_unreachable(monkeypatch):
    def _fake_urlopen(req, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(ts, "_breeze_health", lambda url: (True, 24000))
    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)

    s = _breeze({"base_url": "http://x"})
    with pytest.raises(RuntimeError, match="unreachable"):
        list(s.stream("Nobody home."))
