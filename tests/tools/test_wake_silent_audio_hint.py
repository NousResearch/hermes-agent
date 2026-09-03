"""Tests for wake_word.silent_audio_hint's client-capture branches.

Under ``wake_word.capture: client`` the microphone is on the DESKTOP machine;
the backend host has none. The hint is the only guidance a user gets when the
ear arms but hears nothing, so pointing it at the backend's mic permission
sends people to debug a machine that was never involved.
"""

from unittest.mock import patch

from tools.wake_word import silent_audio_hint


DETAILS = {"name": "Some Input", "index": 3}


class TestClientCapture:
    def test_no_frames_blames_the_desktop_not_the_backend(self):
        """No PCM arrived: the desktop never opened its mic."""
        hint = silent_audio_hint(DETAILS, external_audio=True, frames_seen=False)
        assert "desktop" in hint.lower()
        assert "not streaming" in hint.lower()
        assert "hermes backend" not in hint.lower()

    def test_silent_frames_blames_the_desktop_input_choice(self):
        """PCM arrived but was silent: the desktop opened the wrong input."""
        hint = silent_audio_hint(DETAILS, external_audio=True, frames_seen=True)
        assert "desktop" in hint.lower()
        assert "silence" in hint.lower()
        assert "hermes backend" not in hint.lower()

    def test_the_two_client_failures_give_different_advice(self):
        """Distinguishable causes must not collapse into one message."""
        no_frames = silent_audio_hint(DETAILS, external_audio=True, frames_seen=False)
        silent = silent_audio_hint(DETAILS, external_audio=True, frames_seen=True)
        assert no_frames != silent

    def test_names_the_virtual_loopbacks_that_cause_this(self):
        hint = silent_audio_hint(DETAILS, external_audio=True, frames_seen=True)
        assert "BlackHole" in hint

    def test_client_capture_never_cites_the_backend_host_platform(self):
        """The backend's OS is irrelevant when the mic is on another machine."""
        for platform in ("darwin", "win32", "linux"):
            with patch("tools.wake_word.sys.platform", platform):
                hint = silent_audio_hint(DETAILS, external_audio=True, frames_seen=True)
            assert "desktop" in hint.lower()


class TestLocalCapture:
    def test_macos_points_at_the_backend_mic_permission(self):
        """Without client capture the mic IS on this host — old behaviour stands."""
        with patch("tools.wake_word.sys.platform", "darwin"):
            hint = silent_audio_hint(DETAILS)
        assert "Hermes backend" in hint
        assert "Privacy & Security" in hint

    def test_windows_points_at_the_input_device_setting(self):
        with patch("tools.wake_word.sys.platform", "win32"):
            hint = silent_audio_hint(DETAILS)
        assert "wake_word.input_device" in hint

    def test_local_capture_is_the_default(self):
        """Callers that don't pass the flag must get the local-host advice."""
        with patch("tools.wake_word.sys.platform", "darwin"):
            assert silent_audio_hint(DETAILS) == silent_audio_hint(
                DETAILS, external_audio=False
            )
