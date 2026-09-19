"""Behaviour contract for ``tts.request_timeout``.

Generation time scales with script length, so long-form speech can exceed a
fixed 60s read timeout and fail with ``Read timed out`` *after* the provider has
already done (and billed) the work. The timeout must therefore be configurable —
and must stay enabled when the config value is nonsense, since a disabled
timeout hangs a turn forever.
"""

from tools.tts_tool_providers import (
    DEFAULT_TTS_REQUEST_TIMEOUT,
    _resolve_request_timeout,
)


def test_configured_timeout_reaches_the_request():
    """A positive ``tts.request_timeout`` overrides the default."""
    assert _resolve_request_timeout({"request_timeout": 180}) == 180
    assert _resolve_request_timeout({}) == DEFAULT_TTS_REQUEST_TIMEOUT


def test_unusable_values_keep_a_live_timeout():
    """Garbage must fall back to the default, never to "no timeout".

    ``True`` is included deliberately: ``isinstance(True, int)`` is True in
    Python, so a bare int check would accept it and set a 1-second timeout.
    """
    for bad in (0, -5, True, "180", None, 1.5):
        resolved = _resolve_request_timeout({"request_timeout": bad})
        assert resolved == DEFAULT_TTS_REQUEST_TIMEOUT, bad
        assert resolved > 0
