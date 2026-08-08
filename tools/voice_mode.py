"""Voice Mode -- Push-to-talk audio recording and playback for the CLI.

Provides audio capture via sounddevice, WAV encoding via stdlib wave,
STT dispatch via tools.transcription_tools, and TTS playback via
sounddevice or system audio players.

Dependencies (optional):
    pip install sounddevice numpy
    or: uv sync --extra voice
"""

import difflib
import logging
import math
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
import tempfile
import threading
import time
import wave
from typing import Any, Callable, Dict, List, Optional

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


def _import_numpy():
    """Lazy-import numpy only (no sounddevice).  Returns the module.

    Used where we need to synthesize/convert audio samples but must NOT
    import sounddevice — see _sounddevice_output_allowed.
    """
    import numpy as np
    return np


def _sounddevice_output_allowed() -> bool:
    """Whether sounddevice may be used for audio OUTPUT.

    Returns False on macOS: importing/initializing sounddevice
    (PortAudio/CoreAudio) for output triggers a kTCCServiceMediaLibrary
    permission prompt, even though playback needs no media-library access.
    On macOS all output is routed through ``afplay`` instead. This does NOT
    affect audio *input* (recording), which legitimately needs microphone
    permission. See PR #62601 / #13291.
    """
    return platform.system() != "Darwin"


def _play_int16_via_tempfile(audio, sample_rate: int) -> None:
    """Write int16 mono PCM to a temp WAV and play it via play_audio_file.

    Used on macOS so tone/beep output goes through ``afplay`` instead of
    sounddevice (avoids the TCC media-library prompt).
    """
    tmp_path = None
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        with wave.open(tmp, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(audio.tobytes())
        play_audio_file(tmp_path)
    except Exception as e:
        logger.debug("Tone tempfile playback failed: %s", e)
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _audio_available() -> bool:
    """Return True if audio libraries can be imported."""
    try:
        _import_audio()
        return True
    except (ImportError, OSError):
        return False


def _default_input_samplerate(sd) -> int:
    """Return the preferred capture rate for the default input device.

    Falls back to the Whisper-friendly 16 kHz constant when the backend does
    not expose a numeric default rate.
    """
    try:
        info = sd.query_devices(None, "input")
        rate = info.get("default_samplerate") if isinstance(info, dict) else getattr(info, "default_samplerate", None)
        if isinstance(rate, (int, float)) and rate > 0:
            return int(round(rate))
    except Exception:
        pass
    return SAMPLE_RATE


from hermes_constants import is_termux as _is_termux_environment


def _voice_capture_install_hint() -> str:
    if _is_termux_environment():
        return "pkg install python-numpy portaudio && python -m pip install sounddevice"
    # If we're running inside a venv (e.g. the bundled Hermes venv at
    # ~/.hermes/profiles/<name>/hermes-agent/venv/), `pip install` on the
    # user's PATH won't reach the right site-packages — the bare hint sends
    # them off to whichever Python their shell resolves first, which on macOS
    # is often a system Python under Rosetta with a totally separate wheel
    # index. Point them at the actual interpreter pip is sitting next to.
    try:
        if sys.prefix != getattr(sys, "base_prefix", sys.prefix):
            pip_in_venv = Path(sys.prefix) / "bin" / "pip"
            if pip_in_venv.exists():
                return f"{pip_in_venv} install sounddevice numpy"
    except Exception:
        pass
    return "pip install sounddevice numpy"


def _termux_microphone_command() -> Optional[str]:
    if not _is_termux_environment():
        return None
    return shutil.which("termux-microphone-record")



# Probes used to detect whether the Termux:API Android app is installed.
# `pm list packages` is the canonical Android lookup but is unreliable on
# some devices: on certain ROMs / Android API levels `pm` itself isn't on
# Termux's PATH while `cmd package` is, on others `pm` returns nothing for
# the calling user even when the app is present.  We try both before
# concluding that the app is genuinely missing (issue #31015).
_TERMUX_API_PACKAGE_PROBES = (
    ("pm", "list", "packages", "com.termux.api"),
    ("cmd", "package", "list", "packages", "com.termux.api"),
)


def _termux_api_app_installed() -> bool:
    """Return True iff the Termux:API Android app is installed.

    Strategy (issue #31015):

    1. Try each probe in ``_TERMUX_API_PACKAGE_PROBES`` and look for
       ``package:com.termux.api`` in stdout.  Any positive hit is
       authoritative — return True.
    2. If every probe is *inconclusive* (binary missing, permission
       denied, timeout, non-zero exit) we cannot honestly say the app
       is missing; fall back to trusting the ``termux-microphone-record``
       binary on PATH.  The binary ships with the ``termux-api`` package
       and is only useful when the Android app is installed; users who
       installed the package deliberately almost always have the app
       too.  A false negative on this gate blocks ``/voice on``
       outright (the symptom reported in #31015), while a false
       positive only surfaces a precise runtime error from the binary
       itself — strictly more actionable.
    3. If at least one probe ran cleanly and definitively did not
       mention the package, treat the app as missing and return False
       — that's the genuine "Termux:API CLI installed without the app"
       case the existing warning was written for.
    """
    if not _is_termux_environment():
        return False

    inconclusive = False
    for cmd in _TERMUX_API_PACKAGE_PROBES:
        try:
            result = subprocess.run(
                list(cmd),
                capture_output=True,
                text=True, encoding='utf-8', errors='replace',
                timeout=5,
                check=False,
                stdin=subprocess.DEVNULL,
            )
        except (FileNotFoundError, PermissionError, OSError):
            inconclusive = True
            continue
        except subprocess.TimeoutExpired:
            inconclusive = True
            continue
        if result.returncode != 0:
            inconclusive = True
            continue
        if "package:com.termux.api" in (result.stdout or "").lower():
            return True

    if inconclusive and shutil.which("termux-microphone-record") is not None:
        logger.debug(
            "Termux package-manager probes inconclusive; trusting "
            "termux-microphone-record binary on PATH (issue #31015)."
        )
        return True
    return False


def _termux_voice_capture_available() -> bool:
    return _termux_microphone_command() is not None and _termux_api_app_installed()


def _pulse_socket_reachable() -> bool:
    """Return True if a PulseAudio/PipeWire socket is reachable on disk.

    Covers the common case where a sound server runs locally (e.g. on a
    remote SSH host) without ``PULSE_SERVER``/``PIPEWIRE_REMOTE`` being set --
    the client just connects to the default socket under the runtime dir.
    We look at ``PULSE_SERVER`` unix paths, ``PULSE_RUNTIME_PATH``, and
    ``XDG_RUNTIME_DIR`` for a ``pulse/native`` or ``pipewire-0`` socket
    (issue #35622).
    """
    import socket
    import stat

    candidates: List[str] = []

    pulse_server = os.environ.get('PULSE_SERVER', '')
    # PULSE_SERVER may be "unix:/path", "unix:/path;..." or a bare path.
    for part in pulse_server.split(';'):
        part = part.strip()
        if part.startswith('unix:'):
            candidates.append(part[len('unix:'):])

    pulse_runtime = os.environ.get('PULSE_RUNTIME_PATH')
    if pulse_runtime:
        candidates.append(os.path.join(pulse_runtime, 'native'))

    xdg_runtime = os.environ.get('XDG_RUNTIME_DIR')
    if xdg_runtime:
        candidates.append(os.path.join(xdg_runtime, 'pulse', 'native'))
        candidates.append(os.path.join(xdg_runtime, 'pipewire-0'))

    for path in candidates:
        if not path:
            continue
        try:
            if not stat.S_ISSOCK(os.stat(path).st_mode):
                continue
        except OSError:
            continue
        # Confirm the socket actually accepts a connection -- a stale socket
        # file left by a dead server should not count as reachable.
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.settimeout(0.5)
            sock.connect(path)
            return True
        except OSError:
            continue
        finally:
            sock.close()
    return False


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
    has_forwarded_audio = bool(
        os.environ.get('PULSE_SERVER')
        or os.environ.get('PIPEWIRE_REMOTE')
        or _pulse_socket_reachable()
    )

    # SSH detection -- normally no audio devices, but honor a reachable
    # sound server (PulseAudio/PipeWire socket or forwarding env vars), which
    # works fine over SSH (issue #35622).
    if any(os.environ.get(v) for v in ('SSH_CLIENT', 'SSH_TTY', 'SSH_CONNECTION')):
        if has_forwarded_audio:
            notices.append("Running over SSH with a reachable PulseAudio/PipeWire sound server")
        else:
            warnings.append(
                "Running over SSH -- no audio devices available.\n"
                "  If a sound server (PulseAudio/PipeWire) is running on this host,\n"
                "  point Hermes at it, e.g.:\n"
                "    export XDG_RUNTIME_DIR=/run/user/$(id -u)\n"
                "    # or: export PULSE_SERVER=unix:$XDG_RUNTIME_DIR/pulse/native"
            )

    # Docker/Podman container detection — honor host audio forwarding.
    # When the user mounts a PulseAudio/PipeWire socket into the container
    # and points PULSE_SERVER / PIPEWIRE_REMOTE at it, audio works fine
    # (issue #21203).  Only block when no forwarding is configured.
    from hermes_constants import is_container
    if is_container():
        if has_forwarded_audio:
            notices.append("Running inside container (Docker/Podman/LXC) with host audio forwarding")
        else:
            warnings.append(
                "Running inside container (Docker/Podman/LXC) -- no audio devices.\n"
                "  Forward host audio with one of (substitute $XDG_RUNTIME_DIR for your runtime dir,\n"
                "  typically /run/user/$UID):\n"
                "    PulseAudio:  -v $XDG_RUNTIME_DIR/pulse/native:$XDG_RUNTIME_DIR/pulse/native \\\n"
                "                 -e PULSE_SERVER=unix:$XDG_RUNTIME_DIR/pulse/native\n"
                "    PipeWire:    -e PIPEWIRE_REMOTE=$XDG_RUNTIME_DIR/pipewire-0"
            )

    # WSL detection — a reachable sound server makes audio work in WSL.
    # Honor any forwarding (PulseAudio bridge OR a forwarded PipeWire/Pulse
    # socket), mirroring the SSH and container blocks above. When no
    # forwarding is configured, only hard-block if the WSL2 PowerShell TTS
    # fallback (Media.SoundPlayer via powershell.exe, see play_audio_file)
    # isn't available either. The PowerShell path only covers OUTPUT (TTS
    # playback) -- microphone recording genuinely still needs the
    # PulseAudio bridge -- so when it's the only thing available we
    # downgrade to a notice (keeps the same recording guidance visible,
    # but doesn't block /voice on for TTS-only usage).
    try:
        with open('/proc/version', 'r', encoding="utf-8") as f:
            if 'microsoft' in f.read().lower():
                if has_forwarded_audio:
                    notices.append("Running in WSL with a reachable PulseAudio/PipeWire sound server")
                elif _wsl_powershell_tts_available():
                    notices.append(
                        "Running in WSL without a PulseAudio bridge -- TTS playback "
                        "will use the PowerShell/Media.SoundPlayer fallback. "
                        "Voice INPUT (recording) still requires a PulseAudio bridge:\n"
                        "  1. Set PULSE_SERVER=unix:/mnt/wslg/PulseServer\n"
                        "  2. Create ~/.asoundrc pointing ALSA at PulseAudio\n"
                        "  3. Verify with: arecord -d 3 /tmp/test.wav && aplay /tmp/test.wav"
                    )
                else:
                    warnings.append(
                        "Running in WSL -- audio requires a forwarded sound server.\n"
                        "  PulseAudio: export PULSE_SERVER=unix:/mnt/wslg/PulseServer\n"
                        "  PipeWire:   export PIPEWIRE_REMOTE=$XDG_RUNTIME_DIR/pipewire-0\n"
                        "  Then verify: arecord -d 3 /tmp/test.wav && aplay /tmp/test.wav"
                    )
    except (FileNotFoundError, PermissionError, OSError):
        pass

    # Check audio libraries
    try:
        sd, _ = _import_audio()
        try:
            devices = sd.query_devices()
            if not devices:
                if has_forwarded_audio:
                    notices.append(
                        "No PortAudio devices detected but host audio forwarding is configured -- continuing"
                    )
                elif termux_capture:
                    notices.append("No PortAudio devices detected, but Termux:API microphone capture is available")
                else:
                    warnings.append("No audio input/output devices detected")
        except Exception:
            # In WSL with PulseAudio, device queries can fail even though
            # recording/playback works fine. Don't block if host audio
            # forwarding is configured.
            if has_forwarded_audio:
                notices.append(
                    "Audio device query failed but host audio forwarding is configured -- continuing"
                )
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
_DEFAULT_BEEP_VOLUME = 0.3   # Backward-compatible default (matches prior hardcoded value)


def _get_beep_volume() -> float:
    """Read ``voice.beep_volume`` from config.yaml; clamps to 0.0-1.0.

    Defaults to 0.3 when the key is missing, invalid, or when the config
    system can't be imported (e.g. broken ~/.hermes/config.yaml during a
    partial install). Failures fall back silently so the audio cue never
    breaks the voice loop on a degenerate config.
    """
    try:
        from hermes_cli.config import load_config
        voice_cfg = load_config().get("voice", {})
        if not isinstance(voice_cfg, dict):
            return _DEFAULT_BEEP_VOLUME
        raw = voice_cfg.get("beep_volume", _DEFAULT_BEEP_VOLUME)
    except Exception:
        return _DEFAULT_BEEP_VOLUME
    try:
        volume = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_BEEP_VOLUME
    if isinstance(raw, bool) or volume < 0.0 or volume > 1.0 or _is_nan(volume):
        return _DEFAULT_BEEP_VOLUME
    return volume


def _is_nan(value: float) -> bool:
    try:
        return math.isnan(value)
    except Exception:
        return False


def play_beep(frequency: int = 880, duration: float = 0.12, count: int = 1) -> None:
    """Play a short beep tone using numpy + sounddevice.

    Args:
        frequency: Tone frequency in Hz (default 880 = A5).
        duration: Duration of each beep in seconds.
        count: Number of beeps to play (with short gap between).
    """
    # Synthesize the tone with numpy only (no sounddevice import yet, so the
    # macOS TCC prompt is not triggered on the synthesis step).
    try:
        np = _import_numpy()
    except ImportError:
        return
    try:
        gap = 0.06  # seconds between beeps
        samples_per_beep = int(SAMPLE_RATE * duration)
        samples_per_gap = int(SAMPLE_RATE * gap)

        beep_volume = _get_beep_volume()
        parts = []
        for i in range(count):
            t = np.linspace(0, duration, samples_per_beep, endpoint=False)
            # Apply fade in/out to avoid click artifacts
            tone = np.sin(2 * np.pi * frequency * t)
            fade_len = min(int(SAMPLE_RATE * 0.01), samples_per_beep // 4)
            tone[:fade_len] *= np.linspace(0, 1, fade_len)
            tone[-fade_len:] *= np.linspace(1, 0, fade_len)
            parts.append((tone * beep_volume * 32767).astype(np.int16))
            if i < count - 1:
                parts.append(np.zeros(samples_per_gap, dtype=np.int16))

        audio = np.concatenate(parts)

        # On macOS, route the tone through afplay instead of sounddevice.
        if not _sounddevice_output_allowed():
            _play_int16_via_tempfile(audio, SAMPLE_RATE)
            return

        try:
            sd, _ = _import_audio()
        except (ImportError, OSError):
            return
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
# Thinking sound — calm ambient "blub blub" while the agent works
# ============================================================================
# During a voice conversation the agent can think / run tools for minutes with
# zero audio, which reads as "it died". A quiet, repeating pair of soft water-
# bubble blips fills that gap. Fully synthesized with numpy (no binary asset),
# volume-scaled by voice.beep_volume, gated by voice.thinking_sound (default
# on), and macOS-TCC-safe: sounddevice OUTPUT is gated there
# (_sounddevice_output_allowed), and spawning afplay every second would churn
# subprocesses, so on macOS the thinking sound is skipped silently.

# The host's *should_play* callback decides when blips are allowed; the
# module-level output ref-count below tracks when real audio (TTS sentences,
# file playback) is actually flowing so hosts have an accurate signal.

_audio_output_active_count = 0
_audio_output_lock = threading.Lock()


def mark_audio_output_active(active: bool) -> None:
    """Reference-count real audio output (TTS/file playback).

    Playback paths bracket their work with ``mark_audio_output_active(True)``
    / ``(False)`` so ``is_audio_output_active()`` reflects whether speech
    audio is leaving the speakers RIGHT NOW — unlike the per-turn TTS-done
    events, which stay 'busy' for a whole turn even while the pipeline is
    silently waiting for text.
    """
    global _audio_output_active_count
    with _audio_output_lock:
        _audio_output_active_count = max(
            0, _audio_output_active_count + (1 if active else -1)
        )


def is_audio_output_active() -> bool:
    """True while TTS/file audio is actually playing on the speakers."""
    with _audio_output_lock:
        return _audio_output_active_count > 0


_thinking_lock = threading.Lock()
_thinking_stop: Optional[threading.Event] = None


def thinking_sound_enabled() -> bool:
    """Config gate: ``voice.thinking_sound`` (default True)."""
    try:
        from hermes_cli.config import load_config
        from utils import is_truthy_value

        voice_cfg = load_config().get("voice", {})
        if isinstance(voice_cfg, dict):
            return is_truthy_value(
                voice_cfg.get("thinking_sound", True), default=True
            )
    except Exception:
        pass
    return True


def _synth_thinking_blip(np, frequency: float) -> "Any":
    """One soft 'blub': short sine with a gentle downward pitch glide and a
    smooth attack/decay envelope (no clicks), low-volume."""
    duration = 0.16
    n = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    # Downward glide (water-drop feel): freq → 0.72*freq over the blip.
    glide = np.linspace(1.0, 0.72, n)
    phase = 2 * np.pi * np.cumsum(frequency * glide) / SAMPLE_RATE
    tone = np.sin(phase)
    # Soften harmonics (cheap low-pass feel): add a quieter octave-down sine.
    tone = 0.8 * tone + 0.2 * np.sin(phase / 2.0)
    # Envelope: quick-but-smooth attack, long exponential-ish decay.
    attack = int(0.02 * SAMPLE_RATE)
    env = np.ones(n)
    env[:attack] = np.linspace(0.0, 1.0, attack)
    env *= np.exp(-t * 14.0)
    volume = _get_beep_volume() * 0.5  # deliberately quieter than the beeps
    return (tone * env * volume * 32767).astype(np.int16)


def _thinking_sound_loop(stop: threading.Event, should_play) -> None:
    """Daemon loop: play alternating-pitch blips every ~0.8-1.2s until *stop*.

    Skips a blip (without stopping) whenever *should_play* returns False —
    e.g. TTS audio started flowing or the mic re-armed. macOS: sounddevice
    output is TCC-gated, and per-second afplay subprocess churn is worse
    than silence, so the loop exits immediately there.
    """
    if not _sounddevice_output_allowed():
        return
    try:
        sd, np = _import_audio()
    except (ImportError, OSError):
        return

    import random

    pitches = (392.0, 329.6)  # G4 / E4 — calm, low, alternating
    blips = [_synth_thinking_blip(np, p) for p in pitches]
    i = 0
    while not stop.is_set():
        try:
            if should_play is None or should_play():
                blip = blips[i % len(blips)]
                sd.play(blip, samplerate=SAMPLE_RATE)
                stop.wait(len(blip) / SAMPLE_RATE + 0.02)
                sd.stop()
                i += 1
        except Exception as e:
            logger.debug("Thinking sound blip failed: %s", e)
            return
        stop.wait(0.8 + random.random() * 0.4)


def start_thinking_sound(should_play=None) -> bool:
    """Start the ambient thinking sound (idempotent).

    *should_play* is polled before each blip; return False to skip while
    speech audio flows or the mic is capturing. Returns True when the loop
    was started (or already running), False when disabled/unavailable.
    """
    global _thinking_stop
    if not thinking_sound_enabled():
        return False
    with _thinking_lock:
        if _thinking_stop is not None and not _thinking_stop.is_set():
            return True  # already running
        stop = threading.Event()
        _thinking_stop = stop
    threading.Thread(
        target=_thinking_sound_loop,
        args=(stop, should_play),
        daemon=True,
        name="voice-thinking-sound",
    ).start()
    return True


def stop_thinking_sound() -> None:
    """Stop the ambient thinking sound instantly (idempotent)."""
    global _thinking_stop
    with _thinking_lock:
        stop, _thinking_stop = _thinking_stop, None
    if stop is not None:
        stop.set()


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
            subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=15, check=True, stdin=subprocess.DEVNULL)
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
        subprocess.run([mic_cmd, "-q"], capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=15, check=False, stdin=subprocess.DEVNULL)

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
        self._sample_rate: int = SAMPLE_RATE
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
        # Hard cap on total recording length, wired from voice.max_recording_seconds
        # by the CLI before each recording. 0 (or unset) = no cap (previous behaviour).
        self._max_recording_seconds: float = 0.0
        # Peak RMS seen during recording (for speech presence check in stop())
        self._peak_rms: int = 0
        # Live audio level (read by UI for visual feedback)
        self._current_rms: int = 0

    def _max_duration_reached(self, elapsed: float) -> bool:
        """Whether the configured hard recording-length cap has elapsed.

        ``voice.max_recording_seconds`` is applied by the CLI before each
        recording (see ``HermesCLI._voice_start_recording``). A value <= 0
        (or unset) disables the cap, preserving the previous unbounded
        behaviour.
        """
        cap = self._max_recording_seconds
        return bool(cap and cap > 0 and elapsed >= cap)

    # -- public properties ---------------------------------------------------

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
            self._peak_rms = max(self._peak_rms, rms)

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

                # 3. Hard cap on total recording length (voice.max_recording_seconds).
                #    Independent of speech/silence so a continuous speaker past the
                #    configured limit still auto-stops instead of recording forever.
                if not should_fire and self._max_duration_reached(elapsed):
                    logger.info("Max recording length reached (%.0fs), auto-stopping",
                                self._max_recording_seconds)
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
                samplerate=self._sample_rate,
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
            sd, _ = _import_audio()
        except OSError as e:
            # sounddevice imports but PortAudio's shared library is missing —
            # a pip install can't fix that; point at the system package
            # instead of misreporting missing Python packages (#18432).
            if _is_termux_environment():
                portaudio_hint = "  Termux: pkg install portaudio"
            else:
                portaudio_hint = (
                    "  Linux:  sudo apt-get install libportaudio2\n"
                    "  macOS:  brew install portaudio"
                )
            raise RuntimeError(
                "PortAudio system library not found -- install it first:\n"
                f"{portaudio_hint}\n"
                "Then retry /voice on."
            ) from e
        except ImportError as e:
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
        self._sample_rate = _default_input_samplerate(sd)
        self._ensure_stream()

        with self._lock:
            self._recording = True
        logger.info("Voice recording started (rate=%d, channels=%d)", self._sample_rate, CHANNELS)

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
            min_samples = int(self._sample_rate * 0.3)
            if len(audio_data) < min_samples:
                logger.debug("Recording too short (%d samples), discarding", len(audio_data))
                return None

            # Skip silent recordings using peak RMS (not overall average, which
            # gets diluted by silence at the end of the recording).
            if self._peak_rms < SILENCE_RMS_THRESHOLD:
                logger.info("Recording too quiet (peak RMS=%d < %d), discarding",
                            self._peak_rms, SILENCE_RMS_THRESHOLD)
                return None

            return self._write_wav(audio_data, sample_rate=self._sample_rate)

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
    def _write_wav(audio_data, *, sample_rate: int = SAMPLE_RATE) -> str:
        """Write numpy int16 audio data to a WAV file.

        Returns the file path.
        """
        os.makedirs(_TEMP_DIR, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        wav_path = os.path.join(_TEMP_DIR, f"recording_{timestamp}.wav")

        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(sample_rate)
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
# Voice-chat stop phrases
# ============================================================================

DEFAULT_VOICE_STOP_PHRASES = ("stop",)


def _load_voice_stop_phrases() -> tuple:
    """Return the configured ``voice.stop_phrases`` list (default: ("stop",)).

    Malformed config (scalar, dict, list of non-strings) falls back to the
    default rather than crashing the voice loop.
    """
    try:
        from hermes_cli.config import load_config
        voice_cfg = load_config().get("voice", {})
        if isinstance(voice_cfg, dict):
            raw = voice_cfg.get("stop_phrases", DEFAULT_VOICE_STOP_PHRASES)
            if isinstance(raw, str):
                raw = [raw]
            if isinstance(raw, (list, tuple)):
                phrases = tuple(
                    str(p).strip().lower() for p in raw
                    if isinstance(p, (str, int, float)) and str(p).strip()
                )
                return phrases  # empty tuple = feature disabled
    except Exception:
        pass
    return DEFAULT_VOICE_STOP_PHRASES


def is_voice_stop_phrase(transcript: str, stop_phrases: Optional[tuple] = None) -> bool:
    """Return True when *transcript* is EXACTLY a configured stop phrase.

    Ends the voice conversation when the user says "stop" (or another
    configured phrase) and nothing else. Deliberately strict: the whole
    utterance — after lowercasing and stripping surrounding punctuation —
    must equal a phrase, so "stop doing that and try again" still reaches
    the agent. Configure via ``voice.stop_phrases`` in config.yaml
    (set ``[]`` to disable).
    """
    if not transcript:
        return False
    cleaned = transcript.strip().lower().strip(".,!?;: \t\n\"'")
    if not cleaned:
        return False
    if stop_phrases is None:
        stop_phrases = _load_voice_stop_phrases()
    return cleaned in stop_phrases


# Similarity ratio (difflib.SequenceMatcher, 0..1) above which a
# playback-phase barge transcript is treated as a self-capture of Hermes'
# own just-spoken TTS rather than genuine user speech. See #75780: the
# full-duplex listener has no acoustic echo cancellation, so speaker bleed
# on the mic can trip the barge trigger and get transcribed nearly
# verbatim from the TTS text, creating a TTS -> STT -> TTS feedback loop.
DEFAULT_TTS_ECHO_SIMILARITY_THRESHOLD = 0.6

# Minimum normalized-transcript length (in characters) required before the
# fragment sliding-window fallback runs. Below this, any same-length window
# of `spoken_text` that happens to contain the transcript verbatim (e.g. a
# genuine one-word barge-in like "yes" landing inside a longer reply that
# also says "yes") scores a trivial 1.0 ratio and would otherwise be
# misread as a self-capture. A real self-capture fragment spans at least
# the pre-roll buffer plus time-to-silence, so it is normally well above
# this length; a short genuine interjection is not (#75792 review).
MIN_FRAGMENT_LENGTH_FOR_ECHO = 10


def _normalize_for_echo_compare(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def is_tts_echo(
    transcript: str,
    spoken_text: str,
    threshold: float = DEFAULT_TTS_ECHO_SIMILARITY_THRESHOLD,
) -> bool:
    """Return True when *transcript* looks like a self-capture of *spoken_text*.

    Compares a playback-phase barge-in transcript against the TTS text
    Hermes just spoke using a character-level similarity ratio, which works
    across languages without word-tokenization. A genuine user interjection
    is very unlikely to closely match Hermes' own words, so a high ratio is
    a strong signal of speaker-bleed self-capture (fail-closed guard for the
    playback-phase full-duplex listener, which has no acoustic echo
    cancellation; see #75780).

    The playback-phase capture is cut immediately when the barge trigger
    fires and only spans the pre-roll buffer plus time-to-silence, so for
    any spoken reply longer than a clause the transcript is a short
    FRAGMENT of `spoken_text`, not a near-verbatim repeat of the whole
    thing. A whole-string ratio dilutes towards 0 as `spoken_text` grows
    past the fragment's length, so when the whole-string check misses, we
    also slide a window sized to the transcript's character length across
    `spoken_text` and compare against each window, catching a short
    fragment echoed from within a much longer multi-sentence reply. This
    windowing is character-based (not word-split), so it also works for
    languages without whitespace between words. Transcripts shorter than
    `MIN_FRAGMENT_LENGTH_FOR_ECHO` skip this fallback entirely, since a
    short genuine interjection can trivially match an equally short window
    of unrelated spoken text.
    """
    if not transcript or not spoken_text:
        return False
    a = _normalize_for_echo_compare(transcript)
    b = _normalize_for_echo_compare(spoken_text)
    if not a or not b:
        return False
    if difflib.SequenceMatcher(None, a, b).ratio() >= threshold:
        return True
    if len(a) < MIN_FRAGMENT_LENGTH_FOR_ECHO or len(a) >= len(b):
        return False
    for start in range(0, len(b) - len(a) + 1):
        window = b[start : start + len(a)]
        if difflib.SequenceMatcher(None, a, window).ratio() >= threshold:
            return True
    return False


def voice_stop_hint() -> str:
    """One-line 'Say "stop" to end the voice chat.' hint for voice-mode start.

    Sources the phrase from ``voice.stop_phrases`` (first entry) so a custom
    phrase renders correctly; returns "" when stop phrases are disabled
    (``stop_phrases: []``) so surfaces show no hint at all. Every surface
    that announces voice-mode start (CLI /voice on, TUI, desktop) uses this
    one owner instead of hardcoding the wording.
    """
    phrases = _load_voice_stop_phrases()
    if not phrases:
        return ""
    return f'Say "{phrases[0]}" to end the voice chat.'


# ============================================================================
# STT dispatch
# ============================================================================
def _bounded_stt_metadata(value: Any, default: str) -> str:
    """Normalize STT metadata without forwarding free-form provider output."""
    if not isinstance(value, str):
        return default
    normalized = value.strip().lower()
    if not normalized or len(normalized) > 64:
        return default
    if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", normalized):
        return default
    return normalized


def _stt_failure_metadata(result: Any) -> tuple[str, str]:
    if not isinstance(result, dict):
        return "unknown", "provider_failure"
    provider = _bounded_stt_metadata(result.get("provider"), "unknown")
    error_code = result.get("error_code") or result.get("error_type")
    error = result.get("error")
    if not error_code and isinstance(error, dict):
        error_code = error.get("code") or error.get("type")
    return provider, _bounded_stt_metadata(error_code, "provider_failure")


def _safe_stt_failure_result(provider: str, error_code: str) -> Dict[str, Any]:
    return {
        "success": False,
        "transcript": "",
        "provider": _bounded_stt_metadata(provider, "unknown"),
        "error_code": _bounded_stt_metadata(error_code, "provider_failure"),
        "error": "Transcription failed",
    }


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
    from tools.transcription_tools import MAX_FILE_SIZE, transcribe_audio

    try:
        result = transcribe_audio(wav_path, model=model)
    except Exception as exc:
        logger.error(
            "Voice STT failure provider=unknown stage=transcribe "
            "error_code=stt_exception exception_type=%s",
            type(exc).__name__,
        )
        return _safe_stt_failure_result("unknown", "stt_exception")

    # Only chunk when the provider itself reports "File too large" —
    # local providers (faster-whisper, whisper.cpp, etc.) have no upload
    # cap so ``transcribe_audio`` will never return this error for them.
    raw_error = result.get("error", "")
    is_file_too_large = (
        isinstance(raw_error, str) and "File too large" in raw_error
    ) or result.get("error_code") == "file_too_large"
    if not result.get("success") and is_file_too_large:
        result = _transcribe_wav_in_chunks(wav_path, model=model, max_file_size=MAX_FILE_SIZE)

    # Filter out Whisper hallucinations (common on silent/near-silent audio).
    # A configured voice-chat stop phrase is checked FIRST and always survives:
    # phrases like "bye" or "okay" overlap the hallucination blocklist/repeat
    # regex, and swallowing them here would make saying "bye" (when configured
    # as a stop phrase) silently fail to end the voice chat.
    if result.get("success"):
        raw_transcript = result.get("transcript", "")
        if is_whisper_hallucination(raw_transcript) and not is_voice_stop_phrase(
            raw_transcript
        ):
            logger.info(
                "Filtered Whisper hallucination: stage=hallucination_filter "
                "transcript_chars=%d transcript_bytes=%d",
                len(raw_transcript),
                len(raw_transcript.encode("utf-8")),
            )
            return {"success": True, "transcript": "", "filtered": True}

    # Providers that flag no_speech (empty transcript) failed to hear words,
    # not to transcribe — treat like silence so the voice loop re-listens
    # quietly instead of surfacing "Transcription failed".
    if result.get("no_speech"):
        return {"success": True, "transcript": "", "no_speech": True}

    if not result.get("success"):
        provider, error_code = _stt_failure_metadata(result)
        logger.info(
            "Voice STT failure provider=%s stage=transcribe error_code=%s",
            provider,
            error_code,
        )
        return _safe_stt_failure_result(provider, error_code)

    return result


def _should_chunk_for_transcription(file_path: str, max_file_size: int) -> bool:
    """Return whether a CLI WAV recording needs to be split before STT."""
    if not file_path.lower().endswith(".wav"):
        return False
    try:
        return os.path.getsize(file_path) > max_file_size
    except OSError:
        return False


def _transcribe_wav_in_chunks(
    wav_path: str,
    *,
    model: Optional[str],
    max_file_size: int,
) -> Dict[str, Any]:
    """Split an oversized WAV into provider-sized chunks and join transcripts."""
    from tools.transcription_tools import transcribe_audio

    chunk_paths: List[str] = []
    transcripts: List[str] = []

    try:
        chunk_paths = _split_wav_for_transcription(wav_path, max_file_size=max_file_size)
        if not chunk_paths:
            logger.info(
                "Voice STT failure provider=unknown stage=chunk error_code=no_chunks"
            )
            return _safe_stt_failure_result("unknown", "no_chunks")

        logger.info("Transcribing oversized WAV: stage=chunk chunks=%d", len(chunk_paths))
        for chunk_path in chunk_paths:
            result = transcribe_audio(chunk_path, model=model)
            if not result.get("success"):
                provider, error_code = _stt_failure_metadata(result)
                logger.info(
                    "Voice STT failure provider=%s stage=chunk error_code=%s",
                    provider,
                    error_code,
                )
                return _safe_stt_failure_result(provider, error_code)

            transcript = result.get("transcript", "").strip()
            if transcript and not is_whisper_hallucination(transcript):
                transcripts.append(transcript)

        return {
            "success": True,
            "transcript": " ".join(transcripts).strip(),
            "provider": result.get("provider"),
            "chunks": len(chunk_paths),
        }
    except Exception as exc:
        logger.error(
            "Voice STT failure provider=unknown stage=chunk "
            "error_code=stt_exception exception_type=%s",
            type(exc).__name__,
        )
        return _safe_stt_failure_result("unknown", "stt_exception")
    finally:
        for chunk_path in chunk_paths:
            try:
                if os.path.isfile(chunk_path):
                    os.unlink(chunk_path)
            except OSError:
                pass


def _split_wav_for_transcription(wav_path: str, *, max_file_size: int) -> List[str]:
    """Write WAV chunks small enough to pass the shared STT file-size gate."""
    os.makedirs(_TEMP_DIR, exist_ok=True)
    chunk_paths: List[str] = []
    header_reserve = 64 * 1024

    with wave.open(wav_path, "rb") as source:
        params = source.getparams()
        block_align = max(1, params.nchannels * params.sampwidth)
        max_data_bytes = max_file_size - header_reserve
        if max_data_bytes < block_align:
            raise ValueError("STT max_file_size is too small for WAV chunking")

        frames_per_chunk = max(1, max_data_bytes // block_align)
        index = 0
        while True:
            frames = source.readframes(frames_per_chunk)
            if not frames:
                break

            index += 1
            temp = tempfile.NamedTemporaryFile(
                prefix=f"{os.path.splitext(os.path.basename(wav_path))[0]}_chunk{index:03d}_",
                suffix=".wav",
                dir=_TEMP_DIR,
                delete=False,
            )
            chunk_path = temp.name
            temp.close()

            try:
                with wave.open(chunk_path, "wb") as chunk:
                    chunk.setnchannels(params.nchannels)
                    chunk.setsampwidth(params.sampwidth)
                    chunk.setframerate(params.framerate)
                    chunk.setcomptype(params.comptype, params.compname)
                    chunk.writeframes(frames)
                chunk_paths.append(chunk_path)
            except Exception:
                try:
                    os.unlink(chunk_path)
                except OSError:
                    pass
                raise

    return chunk_paths


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


def _is_wsl() -> bool:
    """True when running inside Windows Subsystem for Linux."""
    try:
        with open("/proc/version", "r", encoding="utf-8", errors="replace") as f:
            return "microsoft" in f.read().lower()
    except Exception:
        return False


def _is_wsl2_env() -> bool:
    """Return True when running inside WSL2 (Windows Subsystem for Linux 2).

    Reads /proc/version and checks for the Microsoft kernel signature.
    Returns False on any error (non-WSL Linux, Docker, SSH, etc.).
    Extracted as a module-level function so tests can patch it directly
    without fighting builtins.open patching complexity.
    """
    try:
        with open("/proc/version", encoding="utf-8", errors="replace") as _fv:
            return "microsoft" in _fv.read().lower()
    except OSError:
        return False


def _wsl_powershell_tts_available() -> bool:
    """Return True when the WSL2 PowerShell TTS playback fallback can be used.

    This only covers OUTPUT (TTS playback via Media.SoundPlayer on the
    Windows host) -- it does NOT make microphone recording work. A caller
    using this to relax the audio-environment gate must still surface the
    existing PulseAudio-bridge guidance for recording/STT.
    """
    return bool(
        _is_wsl2_env()
        and shutil.which("powershell.exe")
        and shutil.which("ffmpeg")
    )


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
    # Ref-count real speaker output for the whole call so the thinking-sound
    # loop (and any other ambient cue) knows audio is flowing right now.
    mark_audio_output_active(True)
    try:
        return _play_audio_file_impl(file_path)
    finally:
        mark_audio_output_active(False)


def _play_audio_file_impl(file_path: str) -> bool:
    global _active_playback

    if not os.path.isfile(file_path):
        logger.warning("Audio file not found: %s", file_path)
        return False

    # Skip sounddevice for output where it is not allowed (macOS): PortAudio/
    # CoreAudio init triggers a kTCCServiceMediaLibrary permission prompt even
    # though playback needs no media-library access. afplay (added to the
    # system-player list below) handles all formats natively instead.
    if file_path.endswith(".wav") and _sounddevice_output_allowed():
        try:
            sd, np = _import_audio()
            with wave.open(file_path, "rb") as wf:
                frames = wf.readframes(wf.getnframes())
                audio_data = np.frombuffer(frames, dtype=np.int16)
                sample_rate = wf.getframerate()

            # WSLg RDP audio needs a warmup to avoid crackling at the start.
            # The RDP virtual-channel connection takes ~100 ms to stabilise,
            # and small default blocksize exasperates timing jitter caused by
            # systemd-timesyncd clock adjustments (microsoft/wslg#1257).
            if _is_wsl():
                silence_samples = int(0.1 * sample_rate)
                fade_samples = int(0.1 * sample_rate)
                fade = np.linspace(0.0, 1.0, fade_samples, dtype=np.float64)
                audio_float = audio_data.astype(np.float64)
                audio_float[:fade_samples] *= fade
                tail = np.zeros(int(0.05 * sample_rate), dtype=np.int16)
                audio_data = np.concatenate([
                    np.zeros(silence_samples, dtype=np.int16),
                    audio_float.astype(np.int16),
                    tail,
                ])
                blocksize = 4096
            else:
                blocksize = 0  # default (auto)

            sd.play(audio_data, samplerate=sample_rate, blocksize=blocksize)
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

    # WSL2 PowerShell fallback: when running in WSL without a PulseAudio
    # bridge, ffplay and aplay have no audio device. If powershell.exe and
    # ffmpeg are available, convert the audio to a uniquely-named WAV in the
    # Windows %TEMP% directory and play it via Media.SoundPlayer -- which
    # always has a working audio device on the Windows host (#17608).
    # A unique suffix prevents concurrent Hermes TTS calls from colliding on
    # the same filename. The WAV is deleted in the shell pipeline
    # unconditionally (success or failure), and the ORIGINAL ffmpeg/
    # powershell exit status is preserved past that cleanup so the player
    # loop below can correctly fall through to ffplay/aplay on failure.
    if system == "Linux" and shutil.which("powershell.exe") and shutil.which("ffmpeg"):
        if _is_wsl2_env():
            try:
                import uuid
                _win_tmp_raw = subprocess.check_output(
                    ["cmd.exe", "/c", "echo %TEMP%"],
                    stderr=subprocess.DEVNULL, timeout=3,
                ).decode(errors="replace").strip()
                _win_tmp_wsl = subprocess.check_output(
                    ["wslpath", "-u", _win_tmp_raw],
                    stderr=subprocess.DEVNULL, timeout=3,
                ).decode(errors="replace").strip()
                if _win_tmp_wsl:
                    # Unique suffix prevents concurrent TTS playback collision.
                    _unique = uuid.uuid4().hex[:8]
                    _wsl_wav = os.path.join(_win_tmp_wsl, f"hermes-tts-{_unique}.wav")
                    _win_wav = subprocess.check_output(
                        ["wslpath", "-w", _wsl_wav],
                        stderr=subprocess.DEVNULL, timeout=3,
                    ).decode(errors="replace").strip()
                    if _win_wav:
                        _win_wav_safe = _win_wav.replace("'", "''")
                        _ps_script = (
                            f"(New-Object Media.SoundPlayer '{_win_wav_safe}').PlaySync()"
                        )
                        _ps_cmd = " && ".join([
                            shlex.join(["ffmpeg", "-i", file_path, "-f", "wav",
                                        _wsl_wav, "-loglevel", "quiet", "-y"]),
                            shlex.join(["powershell.exe", "-NoProfile", "-Command",
                                        _ps_script]),
                        ])
                        _cleanup = shlex.join(["rm", "-f", _wsl_wav])
                        # Capture the (ffmpeg && powershell) exit status into
                        # $rc BEFORE cleanup runs, then exit with that status
                        # instead of rm -f's (rm -f always exits 0, which
                        # would otherwise mask a conversion/playback failure
                        # and prevent falling through to the next player).
                        _full_cmd = f"( {_ps_cmd} ); rc=$?; {_cleanup}; exit $rc"
                        # Use full path so the which(cmd[0]) check in the player loop passes.
                        players.insert(0, ["/bin/sh", "-c", _full_cmd])
            except Exception:
                pass  # WSL path resolution failed; fall through to ffplay/aplay

    players.append(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", file_path])
    if system == "Linux":
        players.append(["aplay", "-q", file_path])

    for cmd in players:
        exe = shutil.which(cmd[0])
        if exe:
            try:
                # Sibling of TTS/STT credential scrub (#70342 / #56332): system
                # audio players must not inherit gateway tokens / API keys.
                from tools.environments.local import hermes_subprocess_env

                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                    env=hermes_subprocess_env(inherit_credentials=False),
                )
                with _playback_lock:
                    _active_playback = proc
                proc.wait(timeout=300)
                rc = proc.returncode
                with _playback_lock:
                    _active_playback = None
                if rc == 0:
                    return True
                # Non-zero exit: player failed (e.g. WSL ffplay/aplay with no
                # audio device, or the PowerShell fallback's ffmpeg/playback
                # step failing). Fall through to the next player in the list.
                logger.debug("System player %s exited with code %d, trying next", cmd[0], rc)
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
# Barge-in — detect the user speaking over TTS playback
# ============================================================================
def listen_for_speech(
    should_stop: Callable[[], bool],
    threshold: Optional[int] = None,
    sustained_ms: int = 300,
    calibration_ms: int = 400,
    capture: bool = False,
    on_trigger: Optional[Callable[[], None]] = None,
    pre_roll_ms: int = 1200,
    endpoint_silence_ms: int = 1250,
    max_utterance_ms: int = 30_000,
):
    """Block until sustained speech is heard on the mic, or *should_stop*.

    Barge-in monitor: run in a side thread while TTS is playing. Without
    *capture* it returns ``True`` when the user started talking (cut playback).
    With ``capture=True`` it ALSO records the interruption — a rolling
    *pre_roll_ms* buffer means the utterance is kept from its first syllable,
    not from the moment detection tripped — and keeps rolling until the user
    goes quiet for *endpoint_silence_ms*, then returns the WAV path (or
    ``None`` if speech never tripped). *on_trigger* fires at the moment of
    detection so the caller can stop playback while capture continues.

    The noise floor is calibrated from the first *calibration_ms* of input —
    playback is already audible then, so speaker bleed is baked into the
    floor and only louder-than-playback speech trips the trigger. Requiring
    *sustained_ms* of consecutive above-threshold blocks filters out coughs,
    keyboard thumps, and playback transients.
    """
    try:
        sd, np = _import_audio()
    except (ImportError, OSError):
        return None if capture else False

    from collections import deque

    block = int(SAMPLE_RATE * 0.03)  # 30ms blocks
    calib_blocks = max(1, calibration_ms // 30)
    trip_blocks = max(1, sustained_ms // 30)
    endpoint_blocks = max(1, endpoint_silence_ms // 30)
    max_blocks = max(1, max_utterance_ms // 30)

    # Rolling floor window: continuously tracks TTS speaker-bleed volume
    # throughout playback, not just the first calibration_ms.  This is the
    # key fix for false barge-in — a one-shot calibration freezes a floor
    # from the opening TTS passage, but later louder passages exceed the
    # stale floor and false-trigger.  The rolling window keeps the floor
    # current so only genuinely louder-than-playback speech trips the VAD.
    floor_window: "deque[float]" = deque(maxlen=max(calib_blocks, 100))  # ~3s rolling
    pre_roll: deque = deque(maxlen=max(1, pre_roll_ms // 30))
    consecutive = 0
    min_floor = 0.0  # baseline from initial calibration; floor never drops below this
    block_idx = 0  # block counter for diagnostic logging

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=block) as stream:
            while not should_stop():
                data, _ = stream.read(block)
                rms = float(np.sqrt(np.mean(data.astype(np.float64) ** 2)))
                if capture:
                    pre_roll.append(data.copy())
                block_idx += 1

                # Wait for at least calib_blocks before evaluating.  During
                # the initial warmup we always feed the window so calibration
                # has data to work with.
                if len(floor_window) < calib_blocks:
                    floor_window.append(rms)
                    continue

                # Lock a minimum floor from the initial calibration samples.
                # During inter-sentence pauses the rolling window can flush
                # with near-silence, collapsing the 90th-percentile floor
                # toward zero and false-triggering on the next rising
                # sentence.  min_floor keeps the trigger from ever dropping
                # below the baseline TTS playback level established during
                # the initial calibration_ms window.
                #
                # If the grace period ended during an inter-sentence gap the
                # calibration samples near-silence.  Locking a near-zero
                # floor sets the trigger so low that TTS blocks exceed it,
                # are excluded from the rolling window (rms >= trigger), and
                # the floor freezes — guaranteeing a false trigger the moment
                # TTS resumes.  Clamp min_floor to SILENCE_RMS_THRESHOLD * 2
                # (400 RMS) so the 8x multiplier yields a trigger of at least
                # (500-2000 RMS) stays below it and feeds the rolling window,
                # while genuine speech (3000-8000 RMS) can still trip it.
                if min_floor == 0.0 and len(floor_window) >= calib_blocks:
                    _pct90 = float(np.percentile(list(floor_window), 90))
                    min_floor = max(_pct90, SILENCE_RMS_THRESHOLD * 2)
                else:
                    _pct90 = float(np.percentile(list(floor_window), 90))

                # Use the 90th percentile of the ROLLING window for the
                # noise floor so the trigger reflects the loudest parts of
                # recent playback — not a frozen snapshot from TTS onset.
                _floor = max(_pct90, min_floor)
                # 8.0x multiplier: TTS speaker bleed has wide
                # volume variation between sentences and within sentences.
                # At 5x, louder TTS passages exceed the trigger, get
                # excluded from the floor window, and create a low-stale
                # floor that false-triggers on the next loud passage.
                # 8x gives enough headroom for TTS dynamics to stay below
                # the trigger and get absorbed into the rolling floor.
                trigger = max(float(threshold or SILENCE_RMS_THRESHOLD * 2), _floor * 8.0)
                # Ceiling: never let the trigger exceed 4000 RMS, otherwise
                # a very loud TTS passage would push the trigger so high
                # that genuine speech (which is typically 3000–8000 RMS)
                # couldn't trip it.
                trigger = min(trigger, 4000.0)

                # Only feed the floor window with blocks that are NOT above
                # the current trigger — speech blocks would inflate the floor
                # and make the trigger unreachable.
                if rms < trigger:
                    floor_window.append(rms)

                consecutive = consecutive + 1 if rms >= trigger else 0
                if consecutive > 0:
                    logger.debug(
                        "VAD above-trigger: block=%d rms=%.0f floor=%.0f trigger=%.0f "
                        "consec=%d/%d min_floor=%.0f window_len=%d",
                        block_idx, rms, _floor, trigger, consecutive,
                        trip_blocks, min_floor, len(floor_window),
                    )
                if consecutive < trip_blocks:
                    continue

                # Tripped — the user is talking over playback.
                logger.info(
                    "VAD TRIPPED: block=%d rms=%.0f floor=%.0f trigger=%.0f "
                    "consec=%d min_floor=%.0f — cutting TTS playback",
                    block_idx, rms, _floor, trigger, consecutive, min_floor,
                )
                if on_trigger:
                    try:
                        on_trigger()
                    except Exception as e:
                        logger.debug("Barge-in trigger callback failed: %s", e)
                if not capture:
                    return True

                # Keep rolling until the user goes quiet. Playback is stopped
                # now, so plain silence endpointing (recorder threshold) works.
                frames: List[Any] = list(pre_roll)
                quiet = 0
                for _ in range(max_blocks):
                    data, _ = stream.read(block)
                    frames.append(data.copy())
                    rms = float(np.sqrt(np.mean(data.astype(np.float64) ** 2)))
                    quiet = quiet + 1 if rms < SILENCE_RMS_THRESHOLD else 0
                    if quiet >= endpoint_blocks:
                        break
                return AudioRecorder._write_wav(np.concatenate(frames, axis=0))
    except Exception as e:
        logger.debug("Barge-in listener failed: %s", e)
    return None if capture else False


# ============================================================================
# Full-duplex agent-turn listener
# ============================================================================
#
# One listener for the WHOLE agent turn in continuous voice mode: armed the
# moment an utterance is submitted, disarmed when the turn is fully done
# (response + TTS finished). It replaces the scattered per-playback barge
# monitors, which had two class-level failures:
#
#   1. HALF-DUPLEX GAP: the monitor only spawned when TTS playback started,
#      so during LLM generation (seconds to minutes) there was NO microphone
#      listener at all — the user could not interject by voice.
#   2. PLAYBACK DEAFNESS: the monitor calibrated its noise floor WHILE the
#      speaker was already blasting TTS, baking speaker bleed into the floor;
#      with an 8x multiplier the trigger became unreachable for normal speech,
#      and requiring a full second of strictly CONSECUTIVE 30ms blocks above
#      trigger meant any intra-word dip reset the counter.
#
# This listener calibrates against the QUIET room at turn start (before any
# playback exists), freezes that baseline through playback (never calibrating
# against its own speaker bleed), and trips on a windowed majority of blocks
# instead of a strict consecutive run.

# Minimum trigger while TTS audio is flowing. Speaker bleed reaching the mic
# through air at conversational volume is typically well under this (a few
# hundred RMS at arm's length; ~1000-1400 with loud speakers close to the
# mic), while direct speech at normal distance measures 3000-8000 RMS.
PLAYBACK_MIN_TRIGGER = 1500.0

# Absolute trigger ceiling — a noisy room must never push the trigger above
# what normal speech (3000-8000 RMS) can reach.
TRIGGER_CEILING = 4000.0

# Default trigger multiplier over the quiet-room floor. The old 8x default
# only made sense when the floor was (wrongly) calibrated against active TTS
# bleed; against a genuine quiet-room baseline (typically 50-300 RMS) the
# synthetic-frame tests show 3x separates speech from ambient cleanly while
# staying reachable (300 RMS floor * 3 = 900 trigger vs 3000+ RMS speech).
DEFAULT_BARGE_MULTIPLIER = 3.0


def _voice_debug_enabled() -> bool:
    return os.environ.get("HERMES_VOICE_DEBUG", "").strip() == "1"


def _vad_log(msg: str) -> None:
    """VAD decision-point diagnostic — always logger.debug, plus stderr when
    HERMES_VOICE_DEBUG=1 so live hardware tuning doesn't need a log tail."""
    logger.debug(msg)
    if _voice_debug_enabled():
        try:
            print(f"[voice-vad] {msg}", file=sys.stderr, flush=True)
        except Exception:
            pass


def full_duplex_listen(
    should_stop: Callable[[], bool],
    is_playing: Optional[Callable[[], bool]] = None,
    on_trigger: Optional[Callable[[str], None]] = None,
    multiplier: Optional[float] = None,
    sustained_ms: int = 300,
    calibration_ms: int = 450,
    grace_ms: int = 500,
    pre_roll_ms: int = 1200,
    endpoint_silence_ms: int = 1250,
    max_utterance_ms: int = 30_000,
) -> Optional[str]:
    """Listen across an ENTIRE agent turn; return the captured interruption.

    Runs from utterance-submit to turn-complete. Two phases, decided per
    30ms block by *is_playing* (usually ``is_audio_output_active``):

    * ``generation`` — no TTS audio flowing. The room is quiet; the first
      *calibration_ms* establish the noise floor (pre-playback calibration).
      Trigger = quiet_floor x *multiplier* (clamped to a sane minimum), i.e.
      ordinary speech detection.
    * ``playback`` — TTS audio flowing. The quiet baseline is HELD (never
      recalibrated against speaker bleed); the trigger is additionally
      clamped up to ``PLAYBACK_MIN_TRIGGER`` so bleed alone can't trip it,
      and a *grace_ms* window after playback first starts suppresses trips
      from the playback onset transient.

    Detection is a windowed majority — >=80% of the last *sustained_ms*
    worth of blocks above trigger (with the current block above) — so
    intra-word energy dips don't reset progress the way the old strictly-
    consecutive counter did.

    On detection ``on_trigger(phase)`` fires (cut TTS / interrupt the turn),
    then capture continues from the rolling *pre_roll_ms* buffer until
    *endpoint_silence_ms* of quiet, and the WAV path is returned. Returns
    ``None`` when *should_stop* ends the turn without speech.
    """
    try:
        sd, np = _import_audio()
    except (ImportError, OSError):
        return None

    from collections import deque

    block = int(SAMPLE_RATE * 0.03)  # 30ms blocks
    calib_blocks = max(1, calibration_ms // 30)
    trip_blocks = max(1, sustained_ms // 30)
    trip_needed = max(1, int(round(trip_blocks * 0.8)))
    grace_blocks = max(0, grace_ms // 30)
    endpoint_blocks = max(1, endpoint_silence_ms // 30)
    max_blocks = max(1, max_utterance_ms // 30)
    mult = float(multiplier) if multiplier else DEFAULT_BARGE_MULTIPLIER

    ambient: "deque[float]" = deque(maxlen=100)  # ~3s of quiet-phase RMS
    pre_roll: deque = deque(maxlen=max(1, pre_roll_ms // 30))
    recent_above: "deque[bool]" = deque(maxlen=trip_blocks)
    quiet_floor = float(SILENCE_RMS_THRESHOLD)
    floor_locked = False
    playing_prev = False
    playback_seen = False
    grace_remaining = 0
    blocks_since_playback = 10_000
    block_idx = 0

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=block
        ) as stream:
            while not should_stop():
                data, _ = stream.read(block)
                rms = float(np.sqrt(np.mean(data.astype(np.float64) ** 2)))
                pre_roll.append(data.copy())
                block_idx += 1

                playing = bool(is_playing()) if is_playing is not None else False

                # Pre-playback calibration: the listener arms at utterance
                # submit, before any TTS exists, so the first calib_blocks
                # sample the actual quiet room — NOT speaker bleed.
                if not floor_locked:
                    if not playing:
                        ambient.append(rms)
                    if len(ambient) >= calib_blocks or playing:
                        pct90 = (
                            float(np.percentile(list(ambient), 90))
                            if ambient
                            else float(SILENCE_RMS_THRESHOLD)
                        )
                        quiet_floor = max(pct90, float(SILENCE_RMS_THRESHOLD))
                        floor_locked = True
                        _vad_log(
                            f"calibrated quiet floor={quiet_floor:.0f} "
                            f"(pct90={pct90:.0f}, {len(ambient)} blocks, "
                            f"mult={mult:g})"
                        )
                    if not floor_locked:
                        continue

                # Playback phase transitions: grace only when playback starts
                # after a real gap (>=1s), so inter-sentence flapping of the
                # audio-active flag can't chain grace windows together and
                # swallow a genuine interjection.
                if playing and not playing_prev:
                    if not playback_seen or blocks_since_playback > 33:
                        grace_remaining = grace_blocks
                        _vad_log(
                            f"playback started (block={block_idx}) — "
                            f"grace {grace_blocks * 30}ms"
                        )
                    playback_seen = True
                playing_prev = playing
                blocks_since_playback = 0 if playing else blocks_since_playback + 1

                # Trigger: quiet baseline x multiplier, phase-clamped.
                trigger = quiet_floor * mult
                if playing:
                    trigger = max(trigger, PLAYBACK_MIN_TRIGGER)
                else:
                    trigger = max(trigger, float(SILENCE_RMS_THRESHOLD) * 2)
                trigger = min(trigger, TRIGGER_CEILING)

                # Keep the quiet floor current with ambient drift — but ONLY
                # while nothing is playing (never absorb speaker bleed) and
                # the block isn't speech.
                if not playing and rms < trigger:
                    ambient.append(rms)
                    if ambient:
                        quiet_floor = max(
                            float(np.percentile(list(ambient), 90)),
                            float(SILENCE_RMS_THRESHOLD),
                        )

                above = rms >= trigger
                if above and grace_remaining > 0:
                    _vad_log(
                        f"grace suppression: block={block_idx} rms={rms:.0f} "
                        f"trigger={trigger:.0f} ({grace_remaining} blocks left)"
                    )
                    above = False
                if grace_remaining > 0:
                    grace_remaining -= 1

                recent_above.append(above)
                if rms >= trigger * 0.5:
                    _vad_log(
                        f"block={block_idx} rms={rms:.0f} floor={quiet_floor:.0f} "
                        f"trigger={trigger:.0f} above={above} "
                        f"window={sum(recent_above)}/{trip_needed} "
                        f"phase={'playback' if playing else 'generation'}"
                    )

                if not (above and sum(recent_above) >= trip_needed):
                    continue

                phase = "playback" if playing else "generation"
                _vad_log(
                    f"TRIPPED ({phase}): block={block_idx} rms={rms:.0f} "
                    f"floor={quiet_floor:.0f} trigger={trigger:.0f} "
                    f"window={sum(recent_above)}/{len(recent_above)}"
                )
                if on_trigger:
                    try:
                        on_trigger(phase)
                    except Exception as e:
                        logger.debug("full-duplex trigger callback failed: %s", e)

                # Capture until the user goes quiet. Playback was cut by
                # on_trigger, so plain silence endpointing works.
                frames: List[Any] = list(pre_roll)
                quiet = 0
                for _ in range(max_blocks):
                    data, _ = stream.read(block)
                    frames.append(data.copy())
                    rms = float(np.sqrt(np.mean(data.astype(np.float64) ** 2)))
                    quiet = quiet + 1 if rms < SILENCE_RMS_THRESHOLD else 0
                    if quiet >= endpoint_blocks:
                        break
                return AudioRecorder._write_wav(np.concatenate(frames, axis=0))
    except Exception as e:
        logger.debug("Full-duplex listener failed: %s", e)
    return None


# ============================================================================
# Requirements check
# ============================================================================
def _check_plugin_stt_provider(provider: str) -> bool:
    """Return True when *provider* resolves to an available STT plugin."""
    if not provider:
        return False
    key = provider.lower().strip()
    if key == "none":
        return False
    try:
        from agent.transcription_registry import get_provider
        from hermes_cli.plugins import _ensure_plugins_discovered

        _ensure_plugins_discovered()
        plugin_provider = get_provider(key)
        if plugin_provider is None:
            # Match the transcription dispatcher: long-lived processes may
            # need one refresh after plugins or configuration change.
            _ensure_plugins_discovered(force=True)
            plugin_provider = get_provider(key)
    except Exception as exc:  # noqa: BLE001 - discovery failure is non-fatal
        logger.debug(
            "STT plugin requirements check skipped for '%s': %s", key, exc,
        )
        return False

    if plugin_provider is None:
        return False

    try:
        return bool(plugin_provider.is_available())
    except Exception as exc:  # noqa: BLE001 - plugins must not break status
        logger.warning(
            "STT plugin provider '%s' is_available() raised during requirements "
            "check: %s - treating as unavailable",
            key,
            exc,
            exc_info=True,
        )
        return False


def check_voice_requirements() -> Dict[str, Any]:
    """Check if all voice mode requirements are met.

    Returns:
        Dict with ``available``, ``audio_available``, ``stt_available``,
        ``missing_packages``, and ``details``.
    """
    # Determine STT provider availability
    from tools.transcription_tools import (
        _get_provider,
        _load_stt_config,
        _resolve_command_stt_provider_config,
        is_stt_enabled,
    )
    stt_config = _load_stt_config()
    stt_enabled = is_stt_enabled(stt_config)
    stt_provider = _get_provider(stt_config)
    native_stt_available = stt_provider in {
        "local",
        "local_command",
        "groq",
        "openai",
        "mistral",
        "xai",
        "elevenlabs",
    }
    command_stt_config = None
    plugin_stt_available = False
    if stt_enabled and not native_stt_available:
        command_stt_config = _resolve_command_stt_provider_config(
            stt_provider, stt_config,
        )
        if command_stt_config is None:
            plugin_stt_available = _check_plugin_stt_provider(stt_provider)
    stt_available = stt_enabled and (
        native_stt_available
        or command_stt_config is not None
        or plugin_stt_available
    )

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
    elif stt_provider == "local_command":
        details_parts.append("STT provider: OK (local command)")
    elif stt_provider == "groq":
        details_parts.append("STT provider: OK (Groq)")
    elif stt_provider == "openai":
        details_parts.append("STT provider: OK (OpenAI)")
    elif stt_provider == "mistral":
        details_parts.append("STT provider: OK (Mistral Voxtral)")
    elif stt_provider == "xai":
        details_parts.append("STT provider: OK (xAI Grok STT)")
    elif stt_provider == "elevenlabs":
        details_parts.append("STT provider: OK (ElevenLabs Scribe)")
    elif command_stt_config is not None:
        details_parts.append(f"STT provider: OK (command: {stt_provider})")
    elif plugin_stt_available:
        details_parts.append(f"STT provider: OK (plugin: {stt_provider})")
    else:
        details_parts.append(
            "STT provider: MISSING (uv pip install faster-whisper — "
            "`pip install faster-whisper` also works if pip is on PATH, "
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
