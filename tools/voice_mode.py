"""Voice Mode -- Push-to-talk audio recording and playback for the CLI.

Provides audio capture via sounddevice, WAV encoding via stdlib wave,
STT dispatch via tools.transcription_tools, and TTS playback via
sounddevice or system audio players.

Dependencies (optional):
    pip install sounddevice numpy
    or: pip install hermes-agent[voice]
"""

import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy audio imports -- never imported at module level to avoid crashing
# in headless environments (SSH, Docker, WSL, no PortAudio).
# ---------------------------------------------------------------------------

def _import_audio():
    """Lazy-import sounddevice and numpy.  Returns (sd, np).

    Raises ImportError or OSError if the libraries are not available
    (e.g. PortAudio missing on headless servers).
    """
    import sounddevice as sd
    import numpy as np
    return sd, np


def _audio_available() -> bool:
    """Return True if audio libraries can be imported."""
    try:
        _import_audio()
        return True
    except (ImportError, OSError):
        return False


from hermes_constants import is_termux as _is_termux_environment


def _voice_capture_install_hint() -> str:
    if _is_termux_environment():
        return "pkg install python-numpy portaudio && python -m pip install sounddevice"
    return "pip install sounddevice numpy"


def _termux_microphone_command() -> Optional[str]:
    if not _is_termux_environment():
        return None
    return shutil.which("termux-microphone-record")



def _termux_api_app_installed() -> bool:
    if not _is_termux_environment():
        return False
    try:
        result = subprocess.run(
            ["pm", "list", "packages", "com.termux.api"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return "package:com.termux.api" in (result.stdout or "")
    except Exception:
        return False


def _termux_voice_capture_available() -> bool:
    return _termux_microphone_command() is not None and _termux_api_app_installed()


def detect_audio_environment() -> dict:
    """Detect if the current environment supports audio I/O.

    Returns dict with 'available' (bool), 'warnings' (list of hard-fail
    reasons that block voice mode), and 'notices' (list of informational
    messages that do NOT block voice mode).
    """
    warnings = []   # hard-fail: these block voice mode
    notices = []     # informational: logged but don't block
    termux_mic_cmd = _termux_microphone_command()
    termux_app_installed = _termux_api_app_installed()
    termux_capture = bool(termux_mic_cmd and termux_app_installed)

    # SSH detection
    if any(os.environ.get(v) for v in ('SSH_CLIENT', 'SSH_TTY', 'SSH_CONNECTION')):
        warnings.append("Running over SSH -- no audio devices available")

    # Docker/Podman container detection
    from hermes_constants import is_container
    if is_container():
        warnings.append("Running inside Docker container -- no audio devices")

    # WSL detection — PulseAudio bridge makes audio work in WSL.
    # Only block if PULSE_SERVER is not configured.
    try:
        with open('/proc/version', 'r') as f:
            if 'microsoft' in f.read().lower():
                if os.environ.get('PULSE_SERVER'):
                    notices.append("Running in WSL with PulseAudio bridge")
                else:
                    warnings.append(
                        "Running in WSL -- audio requires PulseAudio bridge.\n"
                        "  1. Set PULSE_SERVER=unix:/mnt/wslg/PulseServer\n"
                        "  2. Create ~/.asoundrc pointing ALSA at PulseAudio\n"
                        "  3. Verify with: arecord -d 3 /tmp/test.wav && aplay /tmp/test.wav"
                    )
    except (FileNotFoundError, PermissionError, OSError):
        pass

    # Check audio libraries
    try:
        sd, _ = _import_audio()
        try:
            devices = sd.query_devices()
            if not devices:
                if termux_capture:
                    notices.append("No PortAudio devices detected, but Termux:API microphone capture is available")
                else:
                    warnings.append("No audio input/output devices detected")
        except Exception:
            # In WSL with PulseAudio, device queries can fail even though
            # recording/playback works fine. Don't block if PULSE_SERVER is set.
            if os.environ.get('PULSE_SERVER'):
                notices.append("Audio device query failed but PULSE_SERVER is set -- continuing")
            elif termux_capture:
                notices.append("PortAudio device query failed, but Termux:API microphone capture is available")
            else:
                warnings.append("Audio subsystem error (PortAudio cannot query devices)")
    except ImportError:
        if termux_capture:
            notices.append("Termux:API microphone recording available (sounddevice not required)")
        elif termux_mic_cmd and not termux_app_installed:
            warnings.append(
                "Termux:API Android app is not installed. Install/update the Termux:API app to use termux-microphone-record."
            )
        else:
            warnings.append(f"Audio libraries not installed ({_voice_capture_install_hint()})")
    except OSError:
        if termux_capture:
            notices.append("Termux:API microphone recording available (PortAudio not required)")
        elif termux_mic_cmd and not termux_app_installed:
            warnings.append(
                "Termux:API Android app is not installed. Install/update the Termux:API app to use termux-microphone-record."
            )
        elif _is_termux_environment():
            warnings.append(
                "PortAudio system library not found -- install it first:\n"
                "  Termux: pkg install portaudio\n"
                "Then retry /voice on."
            )
        else:
            warnings.append(
                "PortAudio system library not found -- install it first:\n"
                "  Linux:  sudo apt-get install libportaudio2\n"
                "  macOS:  brew install portaudio\n"
                "Then retry /voice on."
            )

    return {
        "available": not warnings,
        "warnings": warnings,
        "notices": notices,
    }

# ---------------------------------------------------------------------------
# Recording parameters
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000  # Whisper native rate
CHANNELS = 1  # Mono
DTYPE = "int16"  # 16-bit PCM
SAMPLE_WIDTH = 2  # bytes per sample (int16)

# Silence detection defaults
SILENCE_RMS_THRESHOLD = 200  # RMS below this = silence (int16 range 0-32767)
SILENCE_DURATION_SECONDS = 3.0  # Seconds of continuous silence before auto-stop

# Temp directory for voice recordings
_TEMP_DIR = os.path.join(tempfile.gettempdir(), "hermes_voice")


# ============================================================================
# Audio cues (beep tones)
# ============================================================================
def play_beep(frequency: int = 880, duration: float = 0.12, count: int = 1) -> None:
    """Play a short beep tone using numpy + sounddevice.

    Args:
        frequency: Tone frequency in Hz (default 880 = A5).
        duration: Duration of each beep in seconds.
        count: Number of beeps to play (with short gap between).
    """
    try:
        sd, np = _import_audio()
    except (ImportError, OSError):
        return
    try:
        gap = 0.06  # seconds between beeps
        samples_per_beep = int(SAMPLE_RATE * duration)
        samples_per_gap = int(SAMPLE_RATE * gap)

        parts = []
        for i in range(count):
            t = np.linspace(0, duration, samples_per_beep, endpoint=False)
            # Apply fade in/out to avoid click artifacts
            tone = np.sin(2 * np.pi * frequency * t)
            fade_len = min(int(SAMPLE_RATE * 0.01), samples_per_beep // 4)
            tone[:fade_len] *= np.linspace(0, 1, fade_len)
            tone[-fade_len:] *= np.linspace(1, 0, fade_len)
            parts.append((tone * 0.3 * 32767).astype(np.int16))
            if i < count - 1:
                parts.append(np.zeros(samples_per_gap, dtype=np.int16))

        audio = np.concatenate(parts)
        sd.play(audio, samplerate=SAMPLE_RATE)
        # sd.wait() calls Event.wait() without timeout — hangs forever if the
        # audio device stalls.  Poll with a 2s ceiling and force-stop.
        deadline = time.monotonic() + 2.0
        while sd.get_stream() and sd.get_stream().active and time.monotonic() < deadline:
            time.sleep(0.01)
        sd.stop()
    except Exception as e:
        logger.debug("Beep playback failed: %s", e)


# ============================================================================
# Termux Audio Recorder
# ============================================================================
class TermuxAudioRecorder:
    """Recorder backend that uses Termux:API microphone capture commands."""

    supports_silence_autostop = False

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._recording = False
        self._start_time = 0.0
        self._recording_path: Optional[str] = None
        self._current_rms = 0

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def elapsed_seconds(self) -> float:
        if not self._recording:
            return 0.0
        return time.monotonic() - self._start_time

    @property
    def current_rms(self) -> int:
        return self._current_rms

    def start(self, on_silence_stop=None) -> None:
        del on_silence_stop  # Termux:API does not expose live silence callbacks.
        mic_cmd = _termux_microphone_command()
        if not mic_cmd:
            raise RuntimeError(
                "Termux voice capture requires the termux-api package and app.\n"
                "Install with: pkg install termux-api\n"
                "Then install/update the Termux:API Android app."
            )
        if not _termux_api_app_installed():
            raise RuntimeError(
                "Termux voice capture requires the Termux:API Android app.\n"
                "Install/update the Termux:API app, then retry /voice on."
            )

        with self._lock:
            if self._recording:
                return
            os.makedirs(_TEMP_DIR, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self._recording_path = os.path.join(_TEMP_DIR, f"recording_{timestamp}.aac")

        command = [
            mic_cmd,
            "-f", self._recording_path,
            "-l", "0",
            "-e", "aac",
            "-r", str(SAMPLE_RATE),
            "-c", str(CHANNELS),
        ]
        try:
            subprocess.run(command, capture_output=True, text=True, timeout=15, check=True)
        except subprocess.CalledProcessError as e:
            details = (e.stderr or e.stdout or str(e)).strip()
            raise RuntimeError(f"Termux microphone start failed: {details}") from e
        except Exception as e:
            raise RuntimeError(f"Termux microphone start failed: {e}") from e

        with self._lock:
            self._start_time = time.monotonic()
            self._recording = True
            self._current_rms = 0
        logger.info("Termux voice recording started")

    def _stop_termux_recording(self) -> None:
        mic_cmd = _termux_microphone_command()
        if not mic_cmd:
            return
        subprocess.run([mic_cmd, "-q"], capture_output=True, text=True, timeout=15, check=False)

    def stop(self) -> Optional[str]:
        with self._lock:
            if not self._recording:
                return None
            self._recording = False
            path = self._recording_path
            self._recording_path = None
            started_at = self._start_time
            self._current_rms = 0

        self._stop_termux_recording()
        if not path or not os.path.isfile(path):
            return None
        if time.monotonic() - started_at < 0.3:
            try:
                os.unlink(path)
            except OSError:
                pass
            return None
        if os.path.getsize(path) <= 0:
            try:
                os.unlink(path)
            except OSError:
                pass
            return None
        logger.info("Termux voice recording stopped: %s", path)
        return path

    def cancel(self) -> None:
        with self._lock:
            path = self._recording_path
            self._recording = False
            self._recording_path = None
            self._current_rms = 0
        try:
            self._stop_termux_recording()
        except Exception:
            pass
        if path and os.path.isfile(path):
            try:
                os.unlink(path)
            except OSError:
                pass
        logger.info("Termux voice recording cancelled")

    def shutdown(self) -> None:
        self.cancel()


# ============================================================================
# AudioRecorder
# ============================================================================
class AudioRecorder:
    """Thread-safe audio recorder using sounddevice.InputStream.

    Usage::

        recorder = AudioRecorder()
        recorder.start(on_silence_stop=my_callback)
        # ... user speaks ...
        wav_path = recorder.stop()   # returns path to WAV file
        # or
        recorder.cancel()            # discard without saving

    If ``on_silence_stop`` is provided, recording automatically stops when
    the user is silent for ``silence_duration`` seconds and calls the callback.
    """

    supports_silence_autostop = True

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stream: Any = None
        self._frames: List[Any] = []
        self._recording = False
        self._start_time: float = 0.0
        # Silence detection state
        self._has_spoken = False
        self._speech_start: float = 0.0  # When speech attempt began
        self._dip_start: float = 0.0  # When current below-threshold dip began
        self._min_speech_duration: float = 0.3  # Seconds of speech needed to confirm
        self._max_dip_tolerance: float = 0.3  # Max dip duration before resetting speech
        self._silence_start: float = 0.0
        self._resume_start: float = 0.0  # Tracks sustained speech after silence starts
        self._resume_dip_start: float = 0.0  # Dip tolerance tracker for resume detection
        self._on_silence_stop = None
        self._silence_threshold: int = SILENCE_RMS_THRESHOLD
        self._silence_duration: float = SILENCE_DURATION_SECONDS
        self._max_wait: float = 15.0  # Max seconds to wait for speech before auto-stop
        # Peak RMS seen during recording (for speech presence check in stop())
        self._peak_rms: int = 0
        # Live audio level (read by UI for visual feedback)
        self._current_rms: int = 0

    # -- public properties ---------------------------------------------------

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def elapsed_seconds(self) -> float:
        if not self._recording:
            return 0.0
        return time.monotonic() - self._start_time

    @property
    def current_rms(self) -> int:
        """Current audio input RMS level (0-32767). Updated each audio chunk."""
        return self._current_rms

    @property
    def is_recording(self) -> bool:
        """Whether audio recording is currently active."""
        return self._recording

    # -- public methods ------------------------------------------------------

    def _ensure_stream(self) -> None:
        """Create the audio InputStream once and keep it alive.

        The stream stays open for the lifetime of the recorder.  Between
        recordings the callback simply discards audio chunks (``_recording``
        is ``False``).  This avoids the CoreAudio bug where closing and
        re-opening an ``InputStream`` hangs indefinitely on macOS.
        """
        if self._stream is not None:
            return  # already alive

        sd, np = _import_audio()

        def _callback(indata, frames, time_info, status):  # noqa: ARG001
            if status:
                logger.debug("sounddevice status: %s", status)
            # When not recording the stream is idle — discard audio.
            if not self._recording:
                return
            self._frames.append(indata.copy())

            # Compute RMS for level display and silence detection
            rms = int(np.sqrt(np.mean(indata.astype(np.float64) ** 2)))
            self._current_rms = rms
            if rms > self._peak_rms:
                self._peak_rms = rms

            # Silence detection
            if self._on_silence_stop is not None:
                now = time.monotonic()
                elapsed = now - self._start_time

                if rms > self._silence_threshold:
                    # Audio is above threshold -- this is speech (or noise).
                    self._dip_start = 0.0  # Reset dip tracker
                    if self._speech_start == 0.0:
                        self._speech_start = now
                    elif not self._has_spoken and now - self._speech_start >= self._min_speech_duration:
                        self._has_spoken = True
                        logger.debug("Speech confirmed (%.2fs above threshold)",
                                     now - self._speech_start)
                    # After speech is confirmed, only reset silence timer if
                    # speech is sustained (>0.3s above threshold).  Brief
                    # spikes from ambient noise should NOT reset the timer.
                    if not self._has_spoken:
                        self._silence_start = 0.0
                    else:
                        # Track resumed speech with dip tolerance.
                        # Brief dips below threshold are normal during speech,
                        # so we mirror the initial speech detection pattern:
                        # start tracking, tolerate short dips, confirm after 0.3s.
                        self._resume_dip_start = 0.0  # Above threshold — no dip
                        if self._resume_start == 0.0:
                            self._resume_start = now
                        elif now - self._resume_start >= self._min_speech_duration:
                            self._silence_start = 0.0
                            self._resume_start = 0.0
                elif self._has_spoken:
                    # Below threshold after speech confirmed.
                    # Use dip tolerance before resetting resume tracker —
                    # natural speech has brief dips below threshold.
                    if self._resume_start > 0:
                        if self._resume_dip_start == 0.0:
                            self._resume_dip_start = now
                        elif now - self._resume_dip_start >= self._max_dip_tolerance:
                            # Sustained dip — user actually stopped speaking
                            self._resume_start = 0.0
                            self._resume_dip_start = 0.0
                elif self._speech_start > 0:
                    # We were in a speech attempt but RMS dipped.
                    # Tolerate brief dips (micro-pauses between syllables).
                    if self._dip_start == 0.0:
                        self._dip_start = now
                    elif now - self._dip_start >= self._max_dip_tolerance:
                        # Dip lasted too long -- genuine silence, reset
                        logger.debug("Speech attempt reset (dip lasted %.2fs)",
                                     now - self._dip_start)
                        self._speech_start = 0.0
                        self._dip_start = 0.0

                # Fire silence callback when:
                # 1. User spoke then went silent for silence_duration, OR
                # 2. No speech detected at all for max_wait seconds
                should_fire = False
                if self._has_spoken and rms <= self._silence_threshold:
                    # User was speaking and now is silent
                    if self._silence_start == 0.0:
                        self._silence_start = now
                    elif now - self._silence_start >= self._silence_duration:
                        logger.info("Silence detected (%.1fs), auto-stopping",
                                    self._silence_duration)
                        should_fire = True
                elif not self._has_spoken and elapsed >= self._max_wait:
                    logger.info("No speech within %.0fs, auto-stopping",
                                self._max_wait)
                    should_fire = True

                if should_fire:
                    with self._lock:
                        cb = self._on_silence_stop
                        self._on_silence_stop = None  # fire only once
                    if cb:
                        def _safe_cb():
                            try:
                                cb()
                            except Exception as e:
                                logger.error("Silence callback failed: %s", e, exc_info=True)
                        threading.Thread(target=_safe_cb, daemon=True).start()

        # Create stream — may block on CoreAudio (first call only).
        stream = None
        try:
            stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                callback=_callback,
            )
            stream.start()
        except Exception as e:
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
            raise RuntimeError(
                f"Failed to open audio input stream: {e}. "
                "Check that a microphone is connected and accessible."
            ) from e
        self._stream = stream

    def start(self, on_silence_stop=None) -> None:
        """Start capturing audio from the default input device.

        The underlying InputStream is created once and kept alive across
        recordings.  Subsequent calls simply reset detection state and
        toggle frame collection via ``_recording``.

        Args:
            on_silence_stop: Optional callback invoked (in a daemon thread) when
                silence is detected after speech. The callback receives no arguments.
                Use this to auto-stop recording and trigger transcription.

        Raises ``RuntimeError`` if sounddevice/numpy are not installed
        or if a recording is already in progress.
        """
        try:
            _import_audio()
        except (ImportError, OSError) as e:
            raise RuntimeError(
                "Voice mode requires sounddevice and numpy.\n"
                f"Install with: {sys.executable} -m pip install sounddevice numpy"
            ) from e

        with self._lock:
            if self._recording:
                return  # already recording

            self._frames = []
            self._start_time = time.monotonic()
            self._has_spoken = False
            self._speech_start = 0.0
            self._dip_start = 0.0
            self._silence_start = 0.0
            self._resume_start = 0.0
            self._resume_dip_start = 0.0
            self._peak_rms = 0
            self._current_rms = 0
            self._on_silence_stop = on_silence_stop

        # Ensure the persistent stream is alive (no-op after first call).
        self._ensure_stream()

        with self._lock:
            self._recording = True
        logger.info("Voice recording started (rate=%d, channels=%d)", SAMPLE_RATE, CHANNELS)

    def _close_stream_with_timeout(self, timeout: float = 3.0) -> None:
        """Close the audio stream with a timeout to prevent CoreAudio hangs."""
        if self._stream is None:
            return

        stream = self._stream
        self._stream = None

        def _do_close():
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

        t = threading.Thread(target=_do_close, daemon=True)
        t.start()
        # Poll in short intervals so Ctrl+C is not blocked
        deadline = __import__("time").monotonic() + timeout
        while t.is_alive() and __import__("time").monotonic() < deadline:
            t.join(timeout=0.1)
        if t.is_alive():
            logger.warning("Audio stream close timed out after %.1fs — forcing ahead", timeout)

    def stop(self) -> Optional[str]:
        """Stop recording and write captured audio to a WAV file.

        The underlying stream is kept alive for reuse — only frame
        collection is stopped.

        Returns:
            Path to the WAV file, or ``None`` if no audio was captured.
        """
        with self._lock:
            if not self._recording:
                return None

            self._recording = False
            self._current_rms = 0
            # Stream stays alive — no close needed.

            if not self._frames:
                return None

            # Concatenate frames and write WAV
            _, np = _import_audio()
            audio_data = np.concatenate(self._frames, axis=0)
            self._frames = []

            elapsed = time.monotonic() - self._start_time
            logger.info("Voice recording stopped (%.1fs, %d samples)", elapsed, len(audio_data))

            # Skip very short recordings (< 0.3s of audio)
            min_samples = int(SAMPLE_RATE * 0.3)
            if len(audio_data) < min_samples:
                logger.debug("Recording too short (%d samples), discarding", len(audio_data))
                return None

            # Skip silent recordings using peak RMS (not overall average, which
            # gets diluted by silence at the end of the recording).
            if self._peak_rms < SILENCE_RMS_THRESHOLD:
                logger.info("Recording too quiet (peak RMS=%d < %d), discarding",
                            self._peak_rms, SILENCE_RMS_THRESHOLD)
                return None

            return self._write_wav(audio_data)

    def cancel(self) -> None:
        """Stop recording and discard all captured audio.

        The underlying stream is kept alive for reuse.
        """
        with self._lock:
            self._recording = False
            self._frames = []
            self._on_silence_stop = None
            self._current_rms = 0
        logger.info("Voice recording cancelled")

    def shutdown(self) -> None:
        """Release the audio stream.  Call when voice mode is disabled."""
        with self._lock:
            self._recording = False
            self._frames = []
            self._on_silence_stop = None
        # Close stream OUTSIDE the lock to avoid deadlock with audio callback
        self._close_stream_with_timeout()
        logger.info("AudioRecorder shut down")

    # -- private helpers -----------------------------------------------------

    @staticmethod
    def _write_wav(audio_data) -> str:
        """Write numpy int16 audio data to a WAV file.

        Returns the file path.
        """
        os.makedirs(_TEMP_DIR, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        wav_path = os.path.join(_TEMP_DIR, f"recording_{timestamp}.wav")

        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_data.tobytes())

        file_size = os.path.getsize(wav_path)
        logger.info("WAV written: %s (%d bytes)", wav_path, file_size)
        return wav_path


def create_audio_recorder() -> AudioRecorder | TermuxAudioRecorder:
    """Return the best recorder backend for the current environment."""
    if _termux_voice_capture_available():
        return TermuxAudioRecorder()
    return AudioRecorder()


# ============================================================================
# Whisper hallucination filter
# ============================================================================
# Whisper commonly hallucinates these phrases on silent/near-silent audio.
WHISPER_HALLUCINATIONS = {
    "thank you.",
    "thank you",
    "thanks for watching.",
    "thanks for watching",
    "subscribe to my channel.",
    "subscribe to my channel",
    "like and subscribe.",
    "like and subscribe",
    "please subscribe.",
    "please subscribe",
    "thank you for watching.",
    "thank you for watching",
    "bye.",
    "bye",
    "you",
    "the end.",
    "the end",
    # Non-English hallucinations (common on silence)
    "продолжение следует",
    "продолжение следует...",
    "sous-titres",
    "sous-titres réalisés par la communauté d'amara.org",
    "sottotitoli creati dalla comunità amara.org",
    "untertitel von stephanie geiges",
    "amara.org",
    "www.mooji.org",
    "ご視聴ありがとうございました",
}

# Regex patterns for repetitive hallucinations (e.g. "Thank you. Thank you. Thank you.")
_HALLUCINATION_REPEAT_RE = re.compile(
    r'^(?:thank you|thanks|bye|you|ok|okay|the end|\.|\s|,|!)+$',
    flags=re.IGNORECASE,
)


def is_whisper_hallucination(transcript: str) -> bool:
    """Check if a transcript is a known Whisper hallucination on silence."""
    cleaned = transcript.strip().lower()
    if not cleaned:
        return True
    # Exact match against known phrases
    if cleaned.rstrip('.!') in WHISPER_HALLUCINATIONS or cleaned in WHISPER_HALLUCINATIONS:
        return True
    # Repetitive patterns (e.g. "Thank you. Thank you. Thank you. you")
    if _HALLUCINATION_REPEAT_RE.match(cleaned):
        return True
    return False


# ============================================================================
# STT dispatch
# ============================================================================
def transcribe_recording(wav_path: str, model: Optional[str] = None) -> Dict[str, Any]:
    """Transcribe a WAV recording using the existing Whisper pipeline.

    Delegates to ``tools.transcription_tools.transcribe_audio()``.
    Filters out known Whisper hallucinations on silent audio.

    Args:
        wav_path: Path to the WAV file.
        model: Whisper model name (default: from config or ``whisper-1``).

    Returns:
        Dict with ``success``, ``transcript``, and optionally ``error``.
    """
    from tools.transcription_tools import transcribe_audio

    result = transcribe_audio(wav_path, model=model)

    # Filter out Whisper hallucinations (common on silent/near-silent audio)
    if result.get("success") and is_whisper_hallucination(result.get("transcript", "")):
        logger.info("Filtered Whisper hallucination: %r", result["transcript"])
        return {"success": True, "transcript": "", "filtered": True}

    return result


# ============================================================================
# Audio playback (interruptable)
# ============================================================================

# Global reference to the active playback process so it can be interrupted.
_active_playback: Optional[subprocess.Popen] = None
_playback_lock = threading.Lock()


def stop_playback() -> None:
    """Interrupt the currently playing audio (if any)."""
    global _active_playback
    with _playback_lock:
        proc = _active_playback
        _active_playback = None
    if proc and proc.poll() is None:
        try:
            proc.terminate()
            logger.info("Audio playback interrupted")
        except Exception:
            pass
    # Also stop sounddevice playback if active
    try:
        sd, _ = _import_audio()
        sd.stop()
    except Exception:
        pass


def play_audio_file(file_path: str) -> bool:
    """Play an audio file through the default output device.

    Strategy:
    1. WAV files via ``sounddevice.play()`` when available.
    2. System commands: ``afplay`` (macOS), ``ffplay`` (cross-platform),
       ``aplay`` (Linux ALSA).

    Playback can be interrupted by calling ``stop_playback()``.

    Returns:
        ``True`` if playback succeeded, ``False`` otherwise.
    """
    global _active_playback

    if not os.path.isfile(file_path):
        logger.warning("Audio file not found: %s", file_path)
        return False

    # Try sounddevice for WAV files
    if file_path.endswith(".wav"):
        try:
            sd, np = _import_audio()
            with wave.open(file_path, "rb") as wf:
                frames = wf.readframes(wf.getnframes())
                audio_data = np.frombuffer(frames, dtype=np.int16)
                sample_rate = wf.getframerate()

            sd.play(audio_data, samplerate=sample_rate)
            # sd.wait() calls Event.wait() without timeout — hangs forever if
            # the audio device stalls.  Poll with a ceiling and force-stop.
            duration_secs = len(audio_data) / sample_rate
            deadline = time.monotonic() + duration_secs + 2.0
            while sd.get_stream() and sd.get_stream().active and time.monotonic() < deadline:
                time.sleep(0.01)
            sd.stop()
            return True
        except (ImportError, OSError):
            pass  # audio libs not available, fall through to system players
        except Exception as e:
            logger.debug("sounddevice playback failed: %s", e)

    # Fall back to system audio players (using Popen for interruptability)
    system = platform.system()
    players = []

    if system == "Darwin":
        players.append(["afplay", file_path])
    players.append(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", file_path])
    if system == "Linux":
        players.append(["aplay", "-q", file_path])

    for cmd in players:
        exe = shutil.which(cmd[0])
        if exe:
            try:
                proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                with _playback_lock:
                    _active_playback = proc
                proc.wait(timeout=300)
                with _playback_lock:
                    _active_playback = None
                return True
            except subprocess.TimeoutExpired:
                logger.warning("System player %s timed out, killing process", cmd[0])
                proc.kill()
                proc.wait()
                with _playback_lock:
                    _active_playback = None
            except Exception as e:
                logger.debug("System player %s failed: %s", cmd[0], e)
                with _playback_lock:
                    _active_playback = None

    logger.warning("No audio player available for %s", file_path)
    return False


# ============================================================================
# Requirements check
# ============================================================================
def check_voice_requirements() -> Dict[str, Any]:
    """Check if all voice mode requirements are met.

    Returns:
        Dict with ``available``, ``audio_available``, ``stt_available``,
        ``missing_packages``, and ``details``.
    """
    # Determine STT provider availability
    from tools.transcription_tools import _get_provider, _load_stt_config, is_stt_enabled
    stt_config = _load_stt_config()
    stt_enabled = is_stt_enabled(stt_config)
    stt_provider = _get_provider(stt_config)
    stt_available = stt_enabled and stt_provider != "none"

    missing: List[str] = []
    termux_capture = _termux_voice_capture_available()
    has_audio = _audio_available() or termux_capture

    if not has_audio:
        missing.extend(["sounddevice", "numpy"])

    # Environment detection
    env_check = detect_audio_environment()

    available = has_audio and stt_available and env_check["available"]
    details_parts = []

    if termux_capture:
        details_parts.append("Audio capture: OK (Termux:API microphone)")
    elif has_audio:
        details_parts.append("Audio capture: OK")
    else:
        details_parts.append(f"Audio capture: MISSING ({_voice_capture_install_hint()})")

    if not stt_enabled:
        details_parts.append("STT provider: DISABLED in config (stt.enabled: false)")
    elif stt_provider == "local":
        details_parts.append("STT provider: OK (local faster-whisper)")
    elif stt_provider == "groq":
        details_parts.append("STT provider: OK (Groq)")
    elif stt_provider == "openai":
        details_parts.append("STT provider: OK (OpenAI)")
    else:
        details_parts.append(
            "STT provider: MISSING (pip install faster-whisper, "
            "or set GROQ_API_KEY / VOICE_TOOLS_OPENAI_KEY)"
        )

    for warning in env_check["warnings"]:
        details_parts.append(f"Environment: {warning}")
    for notice in env_check.get("notices", []):
        details_parts.append(f"Environment: {notice}")

    return {
        "available": available,
        "audio_available": has_audio,
        "stt_available": stt_available,
        "missing_packages": missing,
        "details": "\n".join(details_parts),
        "environment": env_check,
    }


# ============================================================================
# Temp file cleanup
# ============================================================================
def cleanup_temp_recordings(max_age_seconds: int = 3600) -> int:
    """Remove old temporary voice recording files.

    Args:
        max_age_seconds: Delete files older than this (default: 1 hour).

    Returns:
        Number of files deleted.
    """
    if not os.path.isdir(_TEMP_DIR):
        return 0

    deleted = 0
    now = time.time()

    for entry in os.scandir(_TEMP_DIR):
        if entry.is_file() and entry.name.startswith("recording_") and entry.name.endswith(".wav"):
            try:
                age = now - entry.stat().st_mtime
                if age > max_age_seconds:
                    os.unlink(entry.path)
                    deleted += 1
            except OSError:
                pass

    if deleted:
        logger.debug("Cleaned up %d old voice recordings", deleted)
    return deleted
