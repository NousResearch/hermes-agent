"""VOICEVOX TTS integration tool.

Calls a locally-running VOICEVOX engine to synthesise Japanese speech.
Default engine URL: http://127.0.0.1:50021

VOICEVOX must be running before using this tool.
Download: https://voicevox.hiroshiba.jp/

Environment variables:
  VOICEVOX_URL      Engine base URL  (default: http://127.0.0.1:50021)
  VOICEVOX_SPEAKER  Speaker/style ID (default: 8)

Speaker IDs (common):
  1  - 四国めたん (あまあま)
  2  - 四国めたん (ノーマル)
  3  - ずんだもん (あまあま)
  8  - 春日部つむぎ (ノーマル)  ← はくあ default
  9  - 雨晴はう (ノーマル)
  10 - 波音リツ (ノーマル)
  13 - 青山龍星 (ノーマル)
  14 - 冥鳴ひまり (ノーマル)
  Run `voicevox_list_speakers()` to see all available speakers.
"""

import io
import logging
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

_VOICEVOX_URL = os.environ.get("VOICEVOX_URL", "http://127.0.0.1:50021").rstrip("/")
_DEFAULT_SPEAKER = int(os.environ.get("VOICEVOX_SPEAKER", "8"))

# Lock to prevent concurrent synthesis (VOICEVOX engine is single-threaded)
_synthesis_lock = threading.Lock()


def _engine_url() -> str:
    return os.environ.get("VOICEVOX_URL", _VOICEVOX_URL).rstrip("/")


def voicevox_speak(
    text: str,
    speaker: Optional[int] = None,
    blocking: bool = True,
    output_device: str | int | None = None,
) -> dict:
    """Synthesise text with VOICEVOX and play through the system audio output.

    Args:
        text:     Text to speak (Japanese recommended; max ~200 chars per call).
        speaker:  Speaker/style ID. Defaults to VOICEVOX_SPEAKER env var (default: 8).
        blocking: If True, wait until playback finishes before returning.
        output_device: Optional sounddevice output device index or name substring.
                       Use this to route speech into a virtual cable selected as
                       VRChat's microphone input.

    Returns:
        {"success": True, "duration_ms": <int>}  or  {"success": False, "error": "..."}
    """
    if not text or not text.strip():
        return {"success": False, "error": "text cannot be empty"}

    speaker_id = speaker if speaker is not None else _DEFAULT_SPEAKER
    base = _engine_url()

    with _synthesis_lock:
        try:
            # Step 1: create audio query
            r = requests.post(
                f"{base}/audio_query",
                params={"speaker": speaker_id, "text": text},
                timeout=15,
            )
            r.raise_for_status()
            query = r.json()
        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "error": (
                    f"Cannot reach VOICEVOX engine at {base}. "
                    "Is VOICEVOX running? Download: https://voicevox.hiroshiba.jp/"
                ),
            }
        except Exception as exc:
            return {"success": False, "error": f"audio_query failed: {exc}"}

        try:
            # Step 2: synthesise WAV
            r2 = requests.post(
                f"{base}/synthesis",
                params={"speaker": speaker_id},
                json=query,
                timeout=30,
            )
            r2.raise_for_status()
            wav_bytes = r2.content
        except Exception as exc:
            return {"success": False, "error": f"synthesis failed: {exc}"}

    # Step 3: play the WAV
    try:
        result = _play_wav(wav_bytes, blocking=blocking, output_device=output_device)
        return result
    except Exception as exc:
        return {"success": False, "error": f"playback failed: {exc}"}


def voicevox_synthesise(text: str, speaker: Optional[int] = None) -> dict:
    """Synthesise text and return raw WAV bytes (does NOT play audio).

    Useful for saving to file or forwarding to another audio system
    (e.g. the Live2D companion via its HTTP control API).

    Args:
        text:    Text to synthesise.
        speaker: Speaker/style ID (default: VOICEVOX_SPEAKER env var).

    Returns:
        {"success": True, "wav_bytes": <bytes>, "size_bytes": <int>}
        or {"success": False, "error": "..."}
    """
    if not text or not text.strip():
        return {"success": False, "error": "text cannot be empty"}

    speaker_id = speaker if speaker is not None else _DEFAULT_SPEAKER
    base = _engine_url()

    try:
        r = requests.post(
            f"{base}/audio_query",
            params={"speaker": speaker_id, "text": text},
            timeout=15,
        )
        r.raise_for_status()
        query = r.json()

        r2 = requests.post(
            f"{base}/synthesis",
            params={"speaker": speaker_id},
            json=query,
            timeout=30,
        )
        r2.raise_for_status()
        wav_bytes = r2.content
        return {"success": True, "wav_bytes": wav_bytes, "size_bytes": len(wav_bytes)}
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": f"Cannot reach VOICEVOX engine at {base}.",
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def voicevox_list_speakers() -> dict:
    """Return all available speakers from the VOICEVOX engine.

    Returns:
        {"success": True, "speakers": [{"name": ..., "styles": [{"id": ..., "name": ...}]}]}
    """
    base = _engine_url()
    try:
        r = requests.get(f"{base}/speakers", timeout=10)
        r.raise_for_status()
        return {"success": True, "speakers": r.json()}
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": f"Cannot reach VOICEVOX engine at {base}.",
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def voicevox_status() -> dict:
    """Check whether the VOICEVOX engine is reachable."""
    base = _engine_url()
    try:
        r = requests.get(f"{base}/version", timeout=5)
        r.raise_for_status()
        return {"reachable": True, "url": base, "version": r.text.strip('"')}
    except Exception as exc:
        return {"reachable": False, "url": base, "error": str(exc)}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _play_wav(
    wav_bytes: bytes,
    blocking: bool = True,
    output_device: str | int | None = None,
) -> dict:
    """Play WAV bytes through the system audio output."""
    if output_device is not None and str(output_device).strip():
        return _play_wav_to_output_device(wav_bytes, output_device, blocking=blocking)

    import wave

    # Measure duration from WAV header
    try:
        with wave.open(io.BytesIO(wav_bytes)) as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration_ms = int(frames / rate * 1000)
    except Exception:
        duration_ms = 0

    # Write to a temp file then play
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        tmp_path = f.name

    try:
        if sys.platform == "win32":
            import winsound
            flag = winsound.SND_FILENAME
            if not blocking:
                flag |= winsound.SND_ASYNC
            winsound.PlaySound(tmp_path, flag)
        elif sys.platform == "darwin":
            cmd = ["afplay", tmp_path]
            if blocking:
                subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL)
            else:
                subprocess.Popen(cmd, stdin=subprocess.DEVNULL)
        else:
            # Linux: try paplay, then aplay, then ffplay
            for player in ("paplay", "aplay", "ffplay"):
                if _cmd_exists(player):
                    extra = [] if player != "ffplay" else ["-nodisp", "-autoexit"]
                    cmd = [player, *extra, tmp_path]
                    if blocking:
                        subprocess.run(
                            cmd,
                            check=True,
                            capture_output=True,
                            stdin=subprocess.DEVNULL,
                        )
                    else:
                        subprocess.Popen(
                            cmd,
                            stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                    break
            else:
                return {"success": False, "error": "No audio player found (paplay/aplay/ffplay)"}
    finally:
        # Clean up temp file after (slight delay for async)
        if blocking:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass
        else:
            def _cleanup():
                import time
                time.sleep(max(duration_ms / 1000 + 1, 3))
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except Exception:
                    pass
            threading.Thread(target=_cleanup, daemon=True).start()

    return {"success": True, "duration_ms": duration_ms}


def _play_wav_to_output_device(
    wav_bytes: bytes,
    output_device: str | int,
    *,
    blocking: bool = True,
) -> dict:
    """Play WAV bytes through a named/indexed sounddevice output."""
    try:
        import sounddevice as sd
    except Exception as exc:
        return {
            "success": False,
            "error": "sounddevice_unavailable",
            "detail": str(exc),
            "output_device": output_device,
        }

    try:
        data, samplerate, duration_ms = _wav_bytes_to_float32(wav_bytes)
        device_index, device_name = _resolve_output_device(sd, output_device)
        sd.play(data, samplerate, device=device_index, blocking=blocking)
        return {
            "success": True,
            "duration_ms": duration_ms,
            "output_device": device_name,
            "output_device_index": device_index,
        }
    except Exception as exc:
        return {
            "success": False,
            "error": "output_device_playback_failed",
            "detail": str(exc),
            "output_device": output_device,
        }


def _wav_bytes_to_float32(wav_bytes: bytes) -> tuple[Any, int, int]:
    import struct
    import wave

    with wave.open(io.BytesIO(wav_bytes)) as wf:
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        samplerate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        frame_count = wf.getnframes()

    try:
        import numpy as np

        if width == 2:
            data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        elif width == 4:
            data = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"unsupported WAV sample width: {width}")

        if channels > 1:
            data = data.reshape(-1, channels)
    except ImportError:
        if width == 2:
            raw = struct.unpack(f"<{len(frames) // 2}h", frames)
            data = [sample / 32768.0 for sample in raw]
        elif width == 4:
            raw = struct.unpack(f"<{len(frames) // 4}i", frames)
            data = [sample / 2147483648.0 for sample in raw]
        else:
            raise ValueError(f"unsupported WAV sample width: {width}") from None
        if channels > 1:
            data = [
                data[index:index + channels]
                for index in range(0, len(data), channels)
            ]

    duration_ms = int(frame_count / samplerate * 1000) if samplerate else 0
    return data, samplerate, duration_ms


def _resolve_output_device(sd: Any, output_device: str | int) -> tuple[int, str]:
    devices = sd.query_devices()
    if isinstance(output_device, int) or str(output_device).strip().isdigit():
        index = int(output_device)
        try:
            device = devices[index]
        except (IndexError, TypeError) as exc:
            raise ValueError(f"output device index not found: {index}") from exc
        name = str(device.get("name", index))
        channels = int(device.get("max_output_channels", 0) or 0)
        if channels <= 0:
            raise ValueError(f"device has no output channels: {name}")
        return index, name

    needle = str(output_device).strip().casefold()
    for index, device in enumerate(devices):
        name = str(device.get("name", ""))
        channels = int(device.get("max_output_channels", 0) or 0)
        if channels > 0 and needle in name.casefold():
            return index, name

    raise ValueError(f"output device not found: {output_device}")


def _cmd_exists(cmd: str) -> bool:
    import shutil
    return shutil.which(cmd) is not None


def _check_voicevox_requirements() -> dict:
    status = voicevox_status()
    if status["reachable"]:
        return {"available": True}
    return {
        "available": False,
        "reason": (
            f"VOICEVOX engine not reachable at {status['url']}. "
            "Please start VOICEVOX: https://voicevox.hiroshiba.jp/"
        ),
    }


# ---------------------------------------------------------------------------
# Registry — makes these tools callable by the Hermes AI agent
# ---------------------------------------------------------------------------
from tools.registry import registry  # noqa: E402

registry.register(
    name="voicevox_speak",
    toolset="voicevox",
    schema={
        "name": "voicevox_speak",
        "description": (
            "Speak text aloud using VOICEVOX Japanese TTS engine. "
            "Produces high-quality Japanese voice output through the system speakers. "
            "Use this to give はくあ an audible voice in the real world. "
            "VOICEVOX must be running locally (http://127.0.0.1:50021)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to speak aloud (Japanese recommended, max ~200 chars per call)",
                },
                "speaker": {
                    "type": "integer",
                    "description": (
                        "VOICEVOX speaker/style ID. Default: 8 (春日部つむぎ). "
                        "Use voicevox_list_speakers to see all options."
                    ),
                },
                "blocking": {
                    "type": "boolean",
                    "description": "Wait for playback to finish before returning. Default: true",
                },
                "output_device": {
                    "description": (
                        "Optional sounddevice output device index or name substring. "
                        "Use a virtual cable output device when VRChat is listening to the matching cable input."
                    ),
                },
            },
            "required": ["text"],
        },
    },
    handler=lambda args, **kw: voicevox_speak(
        text=args["text"],
        speaker=args.get("speaker"),
        blocking=args.get("blocking", True),
        output_device=args.get("output_device"),
    ),
    check_fn=_check_voicevox_requirements,
    emoji="🔊",
)

registry.register(
    name="voicevox_list_speakers",
    toolset="voicevox",
    schema={
        "name": "voicevox_list_speakers",
        "description": "List all available VOICEVOX speakers and their style IDs.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    handler=lambda args, **kw: voicevox_list_speakers(),
    check_fn=_check_voicevox_requirements,
    emoji="🎤",
)

registry.register(
    name="voicevox_status",
    toolset="voicevox",
    schema={
        "name": "voicevox_status",
        "description": "Check whether the VOICEVOX engine is running and reachable.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    handler=lambda args, **kw: voicevox_status(),
    emoji="🔍",
)
