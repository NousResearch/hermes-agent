"""Tests for the local whisper model idle-unload mechanism.

The local faster-whisper model singleton (``_local_model``) is loaded once
and never released — hundreds of MB of RAM/VRAM sit idle between voice
messages on long-running gateway processes. These tests verify the
config-driven idle unload: after ``unload_after_idle_seconds`` of no
transcription activity, a daemon thread sets ``_local_model = None`` so the
memory can be reclaimed. The next transcription reloads the model
transparently.

Contract under test:

1. Default is 0 (never unload) — no thread is started.
2. After a successful transcription with a non-zero timeout, the watcher
   thread is started and eventually unloads the model.
3. ``_touch_transcription_time`` resets the idle timer.
4. ``_unload_local_model`` is safe to call when the model is already None.
5. Config resolution handles garbage values gracefully.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from tools.transcription_tools import (
    _get_idle_unload_seconds,
    _touch_transcription_time,
    _unload_local_model,
    _start_idle_unload_watcher,
    _IDLE_UNLOAD_CHECK_INTERVAL,
)
import tools.transcription_tools as tt


# ============================================================================
# Config resolution
# ============================================================================


class TestGetIdleUnloadSeconds:
    def test_default_is_zero(self):
        assert _get_idle_unload_seconds({}) == 0

    def test_explicit_value(self):
        assert _get_idle_unload_seconds({"unload_after_idle_seconds": 300}) == 300

    def test_zero_means_never(self):
        assert _get_idle_unload_seconds({"unload_after_idle_seconds": 0}) == 0

    def test_negative_clamped_to_zero(self):
        assert _get_idle_unload_seconds({"unload_after_idle_seconds": -5}) == 0

    def test_garbage_falls_back_to_zero(self):
        assert _get_idle_unload_seconds({"unload_after_idle_seconds": "never"}) == 0

    def test_none_falls_back_to_zero(self):
        assert _get_idle_unload_seconds({"unload_after_idle_seconds": None}) == 0


# ============================================================================
# _unload_local_model
# ============================================================================


class TestUnloadLocalModel:
    def test_unloads_when_model_present(self):
        mock_model = MagicMock(name="whisper_model")
        with patch.object(tt, "_local_model", mock_model), \
             patch.object(tt, "_local_model_name", "base"):
            _unload_local_model()
        assert tt._local_model is None
        assert tt._local_model_name is None

    def test_safe_when_already_none(self):
        with patch.object(tt, "_local_model", None), \
             patch.object(tt, "_local_model_name", None):
            # Must not raise
            _unload_local_model()
        assert tt._local_model is None

    def test_acquires_model_lock(self):
        """The unload must hold _local_model_lock to prevent races with
        concurrent transcriptions that are mid-load."""
        mock_model = MagicMock(name="whisper_model")
        with patch.object(tt, "_local_model", mock_model), \
             patch.object(tt, "_local_model_name", "base"), \
             patch.object(tt, "_local_model_lock") as mock_lock:
            _unload_local_model()
            mock_lock.__enter__.assert_called()
            mock_lock.__exit__.assert_called()


# ============================================================================
# _touch_transcription_time
# ============================================================================


class TestTouchTranscriptionTime:
    def test_sets_timestamp(self):
        original = tt._last_transcription_time
        try:
            tt._last_transcription_time = 0.0
            _touch_transcription_time()
            assert tt._last_transcription_time > 0
        finally:
            tt._last_transcription_time = original


# ============================================================================
# Watcher thread (with mocked time)
# ============================================================================


class TestIdleUnloadWatcher:
    def test_default_zero_does_not_start_thread(self):
        """When unload_after_idle_seconds is 0, no watcher should be started."""
        with patch.object(tt, "_local_model", MagicMock()), \
             patch.object(tt, "_start_idle_unload_watcher") as mock_start:
            # Simulate _transcribe_local's behavior: only start if > 0
            idle_timeout = _get_idle_unload_seconds({"unload_after_idle_seconds": 0})
            if idle_timeout > 0:
                _start_idle_unload_watcher(idle_timeout)
        mock_start.assert_not_called()

    def test_watcher_unloads_after_timeout(self):
        """The watcher unloads the model after the configured idle period."""
        mock_model = MagicMock(name="whisper_model")
        # Use a very short timeout and patch the check interval to 0.01s
        # so the test runs in < 1 second.
        with patch.object(tt, "_local_model", mock_model), \
             patch.object(tt, "_local_model_name", "base"), \
             patch.object(tt, "_IDLE_UNLOAD_CHECK_INTERVAL", 0.01), \
             patch.object(tt, "_last_transcription_time", time.monotonic() - 100):
            _start_idle_unload_watcher(timeout_seconds=1)
            # Wait for the watcher to fire
            for _ in range(100):
                if tt._local_model is None:
                    break
                time.sleep(0.02)
        assert tt._local_model is None

    def test_watcher_does_not_unload_within_timeout(self):
        """The watcher does NOT unload when the model was recently used."""
        mock_model = MagicMock(name="whisper_model")
        original_model = tt._local_model
        original_name = tt._local_model_name
        original_interval = tt._IDLE_UNLOAD_CHECK_INTERVAL
        original_ts = tt._last_transcription_time
        try:
            tt._local_model = mock_model
            tt._local_model_name = "base"
            tt._IDLE_UNLOAD_CHECK_INTERVAL = 0.01
            tt._last_transcription_time = time.monotonic()
            _start_idle_unload_watcher(timeout_seconds=100)
            # Give it a few check cycles — model must survive
            time.sleep(0.1)
            assert tt._local_model is not None
        finally:
            tt._local_model = original_model
            tt._local_model_name = original_name
            tt._IDLE_UNLOAD_CHECK_INTERVAL = original_interval
            tt._last_transcription_time = original_ts

    def test_watcher_exits_when_model_already_none(self):
        """If the model was unloaded by another path, the watcher exits."""
        with patch.object(tt, "_local_model", None), \
             patch.object(tt, "_IDLE_UNLOAD_CHECK_INTERVAL", 0.01), \
             patch.object(tt, "_last_transcription_time", 0.0):
            _start_idle_unload_watcher(timeout_seconds=1)
            time.sleep(0.05)
        # No crash, no hang — watcher detected _local_model is None and exited

    def test_watcher_stopped_on_new_start(self):
        """Starting a new watcher stops the previous one."""
        mock_model = MagicMock(name="whisper_model")
        with patch.object(tt, "_local_model", mock_model), \
             patch.object(tt, "_local_model_name", "base"), \
             patch.object(tt, "_IDLE_UNLOAD_CHECK_INTERVAL", 0.01), \
             patch.object(tt, "_last_transcription_time", time.monotonic()):
            _start_idle_unload_watcher(timeout_seconds=100)
            first_thread = tt._idle_unload_thread
            assert first_thread is not None and first_thread.is_alive()
            _start_idle_unload_watcher(timeout_seconds=200)
            second_thread = tt._idle_unload_thread
            assert second_thread is not first_thread
            # First thread should have been stopped
            time.sleep(0.05)
            assert not first_thread.is_alive()
