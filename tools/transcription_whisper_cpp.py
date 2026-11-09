"""Managed whisper.cpp command and first-use, verified model acquisition."""

from __future__ import annotations

import logging
import shlex
from pathlib import Path

from hermes_constants import get_hermes_home

logger = logging.getLogger("tools.transcription_tools")

# Keys include the profile's absolute file path; a served profile never inherits
# another profile's verification. Changed/replaced files must be checked again.
_verified_files: dict[Path, tuple] = {}

# Model artifacts stay outside PM's bundled tool closure: only first transcription
# downloads them, into writable profile state even when the program is in WindowsApps.
_MODEL_FILES = (
    ("ggml-base.bin",
     "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-base.bin",
     "60ed5bc3dd14eea856493d334349b405782ddcaf0028d4b5df4088345fba2efe"),
    ("ggml-silero-v6.2.0.bin",
     "https://huggingface.co/ggml-org/whisper-vad/resolve/9ffd54a1e1ee413ddf265af9913beaf518d1639b/ggml-silero-v6.2.0.bin",
     "2aa269b785eeb53a82983a20501ddf7c1d9c48e33ab63a41391ac6c9f7fb6987"),
)


def _model_dir() -> Path:
    return get_hermes_home() / "cache" / "whisper.cpp"


def _file_signature(path: Path, digest: str) -> tuple | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return (digest, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def whisper_cpp_command() -> str | None:
    import pm

    # Passive: an engine the install does not carry is absent, never fetched here.
    installed = pm.installed_package("whispercpp-cpu")
    if installed is None:
        return None
    binary = installed.binary
    models = _model_dir()
    # The existing command adapter tokenizes with shlex, including on Windows.
    quote = lambda path: shlex.quote(path.as_posix())
    return (f"{quote(binary)} -m {quote(models / 'ggml-base.bin')} "
            "-f {input_path} -l {language} -t 4 -bs 5 -bo 5 -ng -nt -otxt "
            "-of {output_dir}/transcript --vad "
            f"-vm {quote(models / 'ggml-silero-v6.2.0.bin')} -vsd 500")


def ensure_whisper_cpp_models(model_name: str) -> None:
    if model_name != "base":
        raise ValueError("Windows ARM64 whisper.cpp currently supports stt.local.model: base; "
                         f"requested {model_name!r}")
    from pm.downloader import Download, Source

    destination = _model_dir()
    sources = [Source(url, destination / name, digest) for name, url, digest in _MODEL_FILES]
    pending = [source for source in sources
               if (signature := _file_signature(source.dest, source.sha256)) is None
               or _verified_files.get(source.dest) != signature]
    if not pending:
        return
    logger.info("Preparing whisper.cpp Base and VAD models (first use downloads about 149 MB)")
    # PM owns resumable partials, concurrent-download locking, SHA256 verification,
    # and atomic publication. A complete cache is checked without network access.
    Download(pending, partials_dir=destination / "partials").run()
    for source in pending:
        signature = _file_signature(source.dest, source.sha256)
        if signature is not None:
            _verified_files[source.dest] = signature
