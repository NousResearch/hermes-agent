"""Bounded native-audio payload preparation; never run speech-to-text preprocessing."""

from pathlib import Path
import subprocess
import tempfile

from hermes_platform.resolver import locate_command
from tools.audio_container import sniff_container
from tools.environments.local import build_subprocess_env

# Leave space for the prompt and JSON within a 20 MiB inline request.
MAX_AUDIO_BYTES = 14 * 1024 * 1024


def prepare_audio(data: bytes, supported_formats=("wav", "mp3")) -> tuple[bytes, str]:
    """Keep supported audio bytes; decode other audio containers to PCM without downmix/resampling.

    The shared source resolver owns ingress/SSRF/sandbox limits. Conversion owns a separate
    output cap: a small compressed upload may decode to hours of PCM. Never analyze a prefix
    silently when ffmpeg reaches that cap.
    """
    container = sniff_container(data)
    if container is None:
        raise ValueError("Unrecognized audio container")
    if container in supported_formats:
        if len(data) > MAX_AUDIO_BYTES:
            raise ValueError("Audio exceeds the 14 MiB inline limit; trim it and retry")
        return data, container
    command = locate_command("ffmpeg").command
    if not command:
        raise ValueError("Install ffmpeg to analyze M4A, AAC, FLAC, Ogg or WebM audio")
    with tempfile.TemporaryDirectory(prefix="hermes-audio-") as directory:
        source, output = Path(directory) / f"input.{container}", Path(directory) / "output.wav"
        source.write_bytes(data)
        try:
            result = subprocess.run(
                [*command, "-nostdin", "-hide_banner", "-loglevel", "error",
                 "-protocol_whitelist", "file,pipe", "-i", str(source),
                 "-map", "0:a:0", "-vn", "-threads", "1", "-c:a", "pcm_s16le",
                 "-fs", str(MAX_AUDIO_BYTES), str(output)],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=90,
                env=build_subprocess_env(), check=False)
        except subprocess.TimeoutExpired as exc:
            raise ValueError("Audio conversion exceeded 90 seconds; trim it and retry") from exc
        if result.returncode or not output.is_file():
            raise ValueError("Cannot decode an audio stream from this file")
        if output.stat().st_size >= MAX_AUDIO_BYTES:
            raise ValueError("Decoded audio exceeds 14 MiB; trim it and retry (no partial analysis sent)")
        return output.read_bytes(), "wav"
