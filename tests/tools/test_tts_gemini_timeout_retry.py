"""Regression tests for Gemini TTS request timeout and transient-error retry.

Long-form Gemini TTS requests (~3-4k characters) routinely spend more than
60s server-side before the single non-streaming ``generateContent`` response
comes back, so the previously hard-coded ``timeout=60`` turned a healthy
generation into ``TTS generation failed (gemini): ... Read timed out``.

These tests pin the two behaviours that fix it:
  * the read timeout is configurable via ``tts.gemini.timeout`` and defaults
    to something well above 60s;
  * exactly one bounded retry (with backoff) covers timeouts, 429 and 5xx,
    while 4xx errors still fail fast.
"""

import base64
from unittest.mock import MagicMock, patch

import pytest

import requests


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in (
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_BASE_URL",
        "HERMES_SESSION_PLATFORM",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")


@pytest.fixture
def fake_pcm_bytes():
    # 0.1s of silence at 24kHz mono 16-bit = 4800 bytes
    return b"\x00" * 4800


def _ok_response(pcm: bytes) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": "audio/L16;codec=pcm;rate=24000",
                                "data": base64.b64encode(pcm).decode(),
                            }
                        }
                    ]
                }
            }
        ]
    }
    return resp


def _error_response(status: int, message: str = "boom") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = {"error": {"message": message}}
    resp.iter_content.return_value = iter(
        [b'{"error": {"message": "%s"}}' % message.encode()]
    )
    return resp


class TestGeminiTimeoutResolution:
    def test_default_timeout_is_well_above_sixty_seconds(self):
        from tools.tts_tool import (
            DEFAULT_GEMINI_TTS_TIMEOUT_SECONDS,
            _resolve_gemini_tts_timeout,
        )

        assert DEFAULT_GEMINI_TTS_TIMEOUT_SECONDS > 60
        assert _resolve_gemini_tts_timeout({}) == float(
            DEFAULT_GEMINI_TTS_TIMEOUT_SECONDS
        )

    @pytest.mark.parametrize("key", ["timeout", "timeout_seconds"])
    def test_config_override_is_honoured(self, key):
        from tools.tts_tool import _resolve_gemini_tts_timeout

        assert _resolve_gemini_tts_timeout({key: 240}) == 240.0

    @pytest.mark.parametrize("bad", ["nope", None, 0, -5, [1]])
    def test_invalid_timeout_falls_back_to_default(self, bad):
        from tools.tts_tool import (
            DEFAULT_GEMINI_TTS_TIMEOUT_SECONDS,
            _resolve_gemini_tts_timeout,
        )

        assert _resolve_gemini_tts_timeout({"timeout": bad}) == float(
            DEFAULT_GEMINI_TTS_TIMEOUT_SECONDS
        )

    def test_request_uses_resolved_timeout_not_hardcoded_sixty(
        self, tmp_path, fake_pcm_bytes
    ):
        from tools.tts_tool import _generate_gemini_tts

        out = tmp_path / "out.wav"
        with patch("requests.post", return_value=_ok_response(fake_pcm_bytes)) as post:
            _generate_gemini_tts("hi", str(out), {"gemini": {"timeout": 245}})

        assert post.call_args.kwargs["timeout"] == 245.0

    def test_request_default_timeout_is_not_sixty(self, tmp_path, fake_pcm_bytes):
        from tools.tts_tool import (
            DEFAULT_GEMINI_TTS_TIMEOUT_SECONDS,
            _generate_gemini_tts,
        )

        out = tmp_path / "out.wav"
        with patch("requests.post", return_value=_ok_response(fake_pcm_bytes)) as post:
            _generate_gemini_tts("hi", str(out), {})

        assert post.call_args.kwargs["timeout"] == float(
            DEFAULT_GEMINI_TTS_TIMEOUT_SECONDS
        )


class TestGeminiRetry:
    def test_read_timeout_is_retried_once_and_succeeds(
        self, tmp_path, fake_pcm_bytes
    ):
        """The reported failure mode: first attempt read-times-out, retry works."""
        from tools.tts_tool import _generate_gemini_tts

        out = tmp_path / "out.wav"
        responses = [
            requests.exceptions.ReadTimeout(
                "HTTPSConnectionPool(host='generativelanguage.googleapis.com', "
                "port=443): Read timed out. (read timeout=60)"
            ),
            _ok_response(fake_pcm_bytes),
        ]

        def _post(*_args, **_kwargs):
            item = responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        with patch("requests.post", side_effect=_post) as post, \
                patch("time.sleep") as sleep:
            result = _generate_gemini_tts("hi", str(out), {})

        assert result == str(out)
        assert out.exists() and out.stat().st_size > 0
        assert post.call_count == 2
        # Backoff, not a busy retry.
        assert sleep.call_count == 1
        assert sleep.call_args.args[0] > 0

    def test_retries_are_bounded_and_final_timeout_propagates(self, tmp_path):
        """No infinite retry: attempts are capped and the error surfaces."""
        from tools.tts_tool import GEMINI_TTS_MAX_ATTEMPTS, _generate_gemini_tts

        assert GEMINI_TTS_MAX_ATTEMPTS == 2

        out = tmp_path / "out.wav"
        with patch(
            "requests.post",
            side_effect=requests.exceptions.ReadTimeout("Read timed out."),
        ) as post, patch("time.sleep"):
            with pytest.raises(requests.exceptions.Timeout):
                _generate_gemini_tts("hi", str(out), {})

        assert post.call_count == GEMINI_TTS_MAX_ATTEMPTS

    @pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
    def test_retryable_status_codes_are_retried_once(
        self, tmp_path, fake_pcm_bytes, status
    ):
        from tools.tts_tool import _generate_gemini_tts

        out = tmp_path / "out.wav"
        responses = [_error_response(status), _ok_response(fake_pcm_bytes)]

        with patch("requests.post", side_effect=responses) as post, \
                patch("time.sleep"):
            _generate_gemini_tts("hi", str(out), {})

        assert post.call_count == 2
        assert out.exists()

    @pytest.mark.parametrize("status", [400, 401, 403, 404])
    def test_client_errors_are_not_retried(self, tmp_path, status):
        from tools.tts_tool import _generate_gemini_tts

        out = tmp_path / "out.wav"
        with patch(
            "requests.post", side_effect=[_error_response(status, "bad request")]
        ) as post, patch("time.sleep") as sleep:
            with pytest.raises(RuntimeError, match=f"HTTP {status}"):
                _generate_gemini_tts("hi", str(out), {})

        assert post.call_count == 1
        assert sleep.call_count == 0

    def test_persistent_retryable_status_still_raises_api_error(self, tmp_path):
        from tools.tts_tool import GEMINI_TTS_MAX_ATTEMPTS, _generate_gemini_tts

        out = tmp_path / "out.wav"
        with patch(
            "requests.post",
            side_effect=[_error_response(503) for _ in range(GEMINI_TTS_MAX_ATTEMPTS)],
        ) as post, patch("time.sleep"):
            with pytest.raises(RuntimeError, match="HTTP 503"):
                _generate_gemini_tts("hi", str(out), {})

        assert post.call_count == GEMINI_TTS_MAX_ATTEMPTS

    def test_retry_disabled_by_config(self, tmp_path):
        """Users can opt out of the retry entirely."""
        from tools.tts_tool import _generate_gemini_tts

        out = tmp_path / "out.wav"
        with patch(
            "requests.post",
            side_effect=requests.exceptions.ReadTimeout("Read timed out."),
        ) as post, patch("time.sleep"):
            with pytest.raises(requests.exceptions.Timeout):
                _generate_gemini_tts("hi", str(out), {"gemini": {"retry": False}})

        assert post.call_count == 1
