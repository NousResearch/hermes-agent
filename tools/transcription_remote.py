"""Remote STT audio fetching, URL validation, and bounded chunk fallback."""

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

from hermes_cli._subprocess_compat import windows_hide_flags
from tools.transcription_command import _apply_pre_transcription_hook, _enforce_prompt_length_limit
from tools.transcription_audio import _find_ffmpeg_binary, _find_ffprobe_binary, _validate_audio_file
from tools.transcription_common import DEFAULT_GROQ_STT_MODEL, GROQ_BASE_URL, MAX_FILE_SIZE, OPENAI_MODELS, SUPPORTED_FORMATS, _get_stt_section
from tools.url_safety import is_safe_url

logger = logging.getLogger("tools.transcription_tools")
DEFAULT_REMOTE_AUDIO_MAX_DOWNLOAD_SIZE = 250 * 1024 * 1024
DEFAULT_REMOTE_AUDIO_CHUNK_SIZE = 20 * 1024 * 1024
REMOTE_AUDIO_TIMEOUT = (5, 30)
REMOTE_AUDIO_MAX_REDIRECTS = 10
REMOTE_AUDIO_REDIRECT_STATUSES = {301, 302, 303, 307, 308}


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _remote_audio_limits(stt_config: Optional[Dict[str, Any]] = None) -> tuple[int, int]:
    from tools.transcription_tools import _load_stt_config

    if stt_config is None:
        stt_config = _load_stt_config()

    remote_cfg = _get_stt_section(stt_config, "remote_audio")
    max_download_bytes = _positive_int(
        remote_cfg.get("max_download_bytes"),
        DEFAULT_REMOTE_AUDIO_MAX_DOWNLOAD_SIZE,
    )
    chunk_bytes = min(
        _positive_int(remote_cfg.get("chunk_bytes"), DEFAULT_REMOTE_AUDIO_CHUNK_SIZE),
        MAX_FILE_SIZE - 1024 * 1024,
    )
    return max_download_bytes, chunk_bytes


def _is_remote_audio_url(value: str) -> bool:
    parsed = urlparse(str(value).strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _parse_content_length(headers: Any) -> Optional[int]:
    raw_length = headers.get("Content-Length") if headers else None
    if raw_length:
        try:
            return int(raw_length)
        except (TypeError, ValueError):
            pass

    content_range = headers.get("Content-Range") if headers else None
    if content_range and "/" in content_range:
        _, _, total = content_range.rpartition("/")
        if total and total != "*":
            try:
                return int(total)
            except ValueError:
                return None
    return None


def _looks_like_audio_content_type(content_type: str) -> bool:
    normalized = content_type.split(";", 1)[0].strip().lower()
    return (
        normalized.startswith("audio/")
        or normalized in {"video/mp4", "video/mpeg", "video/webm", "application/ogg", "application/octet-stream"}
    )


def _guess_remote_audio_suffix(url: str, content_type: str = "") -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in SUPPORTED_FORMATS:
        return suffix

    normalized_type = content_type.split(";", 1)[0].strip().lower()
    content_type_suffixes = {
        "audio/aac": ".aac",
        "audio/flac": ".flac",
        "audio/m4a": ".m4a",
        "audio/mp4": ".m4a",
        "audio/mpeg": ".mp3",
        "audio/mp3": ".mp3",
        "audio/ogg": ".ogg",
        "audio/wav": ".wav",
        "audio/wave": ".wav",
        "audio/webm": ".webm",
        "video/mp4": ".mp4",
        "video/webm": ".webm",
        "application/ogg": ".ogg",
    }
    return content_type_suffixes.get(normalized_type, ".mp3")


def _validate_remote_audio_metadata(info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not info.get("success"):
        return {
            "success": False,
            "transcript": "",
            "error": info.get("error") or "Failed to resolve remote audio URL",
        }

    final_url = str(info.get("url") or "")
    if not _is_remote_audio_url(final_url):
        return {
            "success": False,
            "transcript": "",
            "error": "Remote audio redirects must resolve to an http(s) URL",
        }

    suffix = Path(urlparse(final_url).path).suffix.lower()
    content_type = str(info.get("content_type") or "")
    if suffix and suffix not in SUPPORTED_FORMATS and not _looks_like_audio_content_type(content_type):
        return {
            "success": False,
            "transcript": "",
            "error": f"Unsupported remote audio format: {suffix}. Supported: {', '.join(sorted(SUPPORTED_FORMATS))}",
        }
    if content_type and not suffix and not _looks_like_audio_content_type(content_type):
        return {
            "success": False,
            "transcript": "",
            "error": f"Remote URL does not look like audio (Content-Type: {content_type})",
        }
    return None


def _blocked_remote_audio_url_error(url: str) -> str:
    return f"Remote audio URL is blocked by URL safety policy: {url}"


def _safe_remote_audio_request(
    session: Any,
    method: str,
    audio_url: str,
    **kwargs: Any,
) -> tuple[Optional[Any], Optional[str]]:
    """Run a remote audio request while validating each redirect target."""
    current_url = audio_url
    request_method = getattr(session, method.lower())
    for _ in range(REMOTE_AUDIO_MAX_REDIRECTS + 1):
        if not _is_remote_audio_url(current_url):
            return None, "Remote audio redirects must resolve to an http(s) URL"
        if not is_safe_url(current_url):
            return None, _blocked_remote_audio_url_error(current_url)

        response = request_method(current_url, allow_redirects=False, **kwargs)
        final_url = str(getattr(response, "url", current_url) or current_url)
        if not is_safe_url(final_url):
            response.close()
            return None, _blocked_remote_audio_url_error(final_url)

        location = response.headers.get("Location") if response.headers else None
        if response.status_code in REMOTE_AUDIO_REDIRECT_STATUSES and location:
            next_url = urljoin(final_url, location)
            response.close()
            current_url = next_url
            continue

        return response, None

    return None, "Remote audio redirects exceeded safe redirect limit"


def _probe_remote_audio_url(audio_url: str) -> Dict[str, Any]:
    """Resolve redirects and gather cheap metadata for an http(s) audio URL."""
    if not _is_remote_audio_url(audio_url):
        return {
            "success": False,
            "url": audio_url,
            "error": "Remote audio URL must use http or https",
        }

    try:
        import requests
    except ImportError:
        return {
            "success": False,
            "url": audio_url,
            "error": "requests package not installed",
        }

    session = requests.Session()
    last_error = ""
    try:
        try:
            response, safety_error = _safe_remote_audio_request(
                session,
                "HEAD",
                audio_url,
                timeout=REMOTE_AUDIO_TIMEOUT,
            )
            if safety_error:
                return {"success": False, "url": audio_url, "error": safety_error}
            if 200 <= response.status_code < 400:
                try:
                    return {
                        "success": True,
                        "url": response.url,
                        "content_type": response.headers.get("Content-Type", ""),
                        "content_length": _parse_content_length(response.headers),
                    }
                finally:
                    response.close()
            last_error = f"HEAD returned HTTP {response.status_code}"
            response.close()
        except requests.RequestException as exc:
            last_error = str(exc)

        response, safety_error = _safe_remote_audio_request(
            session,
            "GET",
            audio_url,
            headers={"Range": "bytes=0-0"},
            stream=True,
            timeout=REMOTE_AUDIO_TIMEOUT,
        )
        if safety_error:
            return {"success": False, "url": audio_url, "error": safety_error}
        try:
            if response.status_code not in {200, 206}:
                detail = response.text[:200] if response.text else last_error
                return {
                    "success": False,
                    "url": audio_url,
                    "error": f"Remote audio probe failed (HTTP {response.status_code}): {detail}",
                }

            return {
                "success": True,
                "url": response.url,
                "content_type": response.headers.get("Content-Type", ""),
                "content_length": _parse_content_length(response.headers),
            }
        finally:
            response.close()
    except requests.RequestException as exc:
        detail = f"{last_error}; {exc}" if last_error else str(exc)
        return {
            "success": False,
            "url": audio_url,
            "error": f"Remote audio probe failed: {detail}",
        }
    finally:
        session.close()


def _download_remote_audio(
    audio_url: str,
    work_dir: str,
    *,
    size_hint: Optional[int] = None,
    content_type: str = "",
    max_download_bytes: Optional[int] = None,
) -> tuple[Optional[str], Optional[str]]:
    if max_download_bytes is None:
        max_download_bytes, _ = _remote_audio_limits()

    if size_hint is not None and size_hint > max_download_bytes:
        return (
            None,
            (
                f"Remote audio is too large to download safely: "
                f"{size_hint / (1024 * 1024):.1f}MB "
                f"(max {max_download_bytes / (1024 * 1024):.0f}MB)"
            ),
        )

    try:
        import requests
    except ImportError:
        return None, "requests package not installed"

    session = requests.Session()
    response = None
    output_path: Optional[Path] = None
    try:
        response, safety_error = _safe_remote_audio_request(
            session,
            "GET",
            audio_url,
            stream=True,
            timeout=REMOTE_AUDIO_TIMEOUT,
        )
        if safety_error:
            return None, safety_error
        if response.status_code < 200 or response.status_code >= 300:
            detail = response.text[:200] if response.text else ""
            return None, f"Remote audio download failed (HTTP {response.status_code}): {detail}"

        final_url = response.url
        if not _is_remote_audio_url(final_url):
            return None, "Remote audio redirects must resolve to an http(s) URL"

        response_length = _parse_content_length(response.headers)
        if response_length is not None and response_length > max_download_bytes:
            return (
                None,
                (
                    f"Remote audio is too large to download safely: "
                    f"{response_length / (1024 * 1024):.1f}MB "
                    f"(max {max_download_bytes / (1024 * 1024):.0f}MB)"
                ),
            )

        response_type = response.headers.get("Content-Type", "") or content_type
        suffix = _guess_remote_audio_suffix(final_url, response_type)
        output_path = Path(work_dir) / f"remote-audio{suffix}"
        total = 0
        with open(output_path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                total += len(chunk)
                if total > max_download_bytes:
                    return (
                        None,
                        (
                            f"Remote audio exceeded safe download limit "
                            f"({max_download_bytes / (1024 * 1024):.0f}MB)"
                        ),
                    )
                handle.write(chunk)

        return str(output_path), None
    except requests.RequestException as exc:
        return None, f"Remote audio download failed: {exc}"
    except OSError as exc:
        return None, f"Failed to write remote audio download: {exc}"
    finally:
        if response is not None:
            response.close()
        session.close()
        if output_path is not None and output_path.exists() and output_path.stat().st_size == 0:
            output_path.unlink(missing_ok=True)


def _probe_audio_duration_for_chunk(file_path: str) -> tuple[Optional[float], Optional[str]]:
    ffprobe = _find_ffprobe_binary()
    if not ffprobe:
        return None, "Chunk fallback requires ffprobe, but ffprobe was not found"

    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        file_path,
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
            stdin=subprocess.DEVNULL,
            creationflags=windows_hide_flags(),
        )
        duration = float(result.stdout.strip())
        if duration <= 0:
            return None, "Could not determine a positive audio duration for chunk fallback"
        return duration, None
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError) as exc:
        return None, f"Failed to inspect audio duration for chunk fallback: {exc}"


def _split_audio_file_for_stt(
    file_path: str,
    work_dir: str,
    *,
    max_chunk_bytes: int = DEFAULT_REMOTE_AUDIO_CHUNK_SIZE,
) -> tuple[List[str], Optional[str]]:
    audio_path = Path(file_path)
    try:
        file_size = audio_path.stat().st_size
    except OSError as exc:
        return [], f"Failed to inspect audio file for chunk fallback: {exc}"

    if file_size <= MAX_FILE_SIZE:
        return [file_path], None

    ffmpeg = _find_ffmpeg_binary()
    if not ffmpeg:
        return [], "Chunk fallback requires ffmpeg, but ffmpeg was not found"

    duration, duration_error = _probe_audio_duration_for_chunk(file_path)
    if duration_error:
        return [], duration_error

    target_bytes = min(max_chunk_bytes, MAX_FILE_SIZE - 1024 * 1024)
    segment_seconds = max(10, int(duration * target_bytes / file_size * 0.85))
    suffix = audio_path.suffix.lower() if audio_path.suffix.lower() in SUPPORTED_FORMATS else ".mp3"

    for attempt in range(4):
        chunks_dir = Path(work_dir) / f"chunks-{attempt}"
        chunks_dir.mkdir(parents=True, exist_ok=True)
        output_pattern = str(chunks_dir / f"chunk-%03d{suffix}")
        command = [
            ffmpeg,
            "-y",
            "-i",
            file_path,
            "-map",
            "0:a:0",
            "-vn",
            "-f",
            "segment",
            "-segment_time",
            str(segment_seconds),
            "-reset_timestamps",
            "1",
            "-c",
            "copy",
            output_pattern,
        ]
        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=300,
                stdin=subprocess.DEVNULL,
                creationflags=windows_hide_flags(),
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            stderr = getattr(exc, "stderr", "") or ""
            stdout = getattr(exc, "stdout", "") or ""
            details = stderr.strip() or stdout.strip() or str(exc)
            return [], f"Failed to split audio for chunk fallback: {details}"

        chunks = sorted(str(path) for path in chunks_dir.glob(f"*{suffix}"))
        if not chunks:
            return [], "Chunk fallback did not produce any audio chunks"

        largest_chunk = max(Path(path).stat().st_size for path in chunks)
        if largest_chunk <= MAX_FILE_SIZE:
            return chunks, None

        segment_seconds = max(5, int(segment_seconds * target_bytes / largest_chunk * 0.8))

    return [], "Chunk fallback could not produce chunks below the provider size limit"


def _looks_like_provider_size_limit_error(message: str) -> bool:
    lowered = str(message).lower()
    return any(
        marker in lowered
        for marker in (
            "too large",
            "size limit",
            "maximum file",
            "maximum content",
            "payload too large",
            "413",
        )
    )


def _transcribe_audio_file_chunks(
    file_path: str,
    provider: str,
    stt_config: dict,
    model: Optional[str],
    *,
    max_chunk_bytes: Optional[int] = None,
    source: Optional[str] = None,
) -> Dict[str, Any]:
    from tools.transcription_tools import _transcribe_prepared_audio

    with tempfile.TemporaryDirectory(prefix="hermes-stt-chunks-") as chunk_dir:
        if max_chunk_bytes is None:
            _, max_chunk_bytes = _remote_audio_limits(stt_config)
        chunks, chunk_error = _split_audio_file_for_stt(
            file_path,
            chunk_dir,
            max_chunk_bytes=max_chunk_bytes,
        )
        if chunk_error:
            return {"success": False, "transcript": "", "error": chunk_error}

        transcripts: List[str] = []
        for index, chunk_path in enumerate(chunks, start=1):
            result = _transcribe_prepared_audio(chunk_path, model, source)
            if not result.get("success"):
                return {
                    "success": False,
                    "transcript": "\n".join(transcripts).strip(),
                    "provider": provider,
                    "error": (
                        f"Chunk {index}/{len(chunks)} transcription failed: "
                        f"{result.get('error', 'unknown error')}"
                    ),
                }
            transcript = str(result.get("transcript") or "").strip()
            if transcript:
                transcripts.append(transcript)

        return {
            "success": True,
            "transcript": "\n".join(transcripts).strip(),
            "provider": provider,
            "chunks": len(chunks),
        }


def _transcribe_remote_audio_url(
    audio_url: str, model: Optional[str] = None, source: Optional[str] = None) -> Dict[str, Any]:
    """Resolve and transcribe a remote http(s) audio URL."""
    from tools.transcription_tools import _get_provider, _load_stt_config, _transcribe_prepared_audio, is_stt_enabled, _no_provider_error

    stt_config = _load_stt_config()
    if not is_stt_enabled(stt_config):
        return {
            "success": False,
            "transcript": "",
            "error": "STT is disabled in config.yaml (stt.enabled: false).",
        }

    provider = _get_provider(stt_config)
    if provider == "none":
        return _no_provider_error(provider, stt_config)

    probe = _probe_remote_audio_url(audio_url)
    metadata_error = _validate_remote_audio_metadata(probe)
    if metadata_error:
        return metadata_error

    final_url = str(probe["url"])
    content_type = str(probe.get("content_type") or "")
    content_length = probe.get("content_length")
    max_download_bytes, max_chunk_bytes = _remote_audio_limits(stt_config)

    if provider == "groq":
        if content_length is None or content_length <= MAX_FILE_SIZE:
            from tools.transcription_tools import _builtin_model_name, _resolve_stt_language

            prompt = stt_config.get("prompt")
            prompt = prompt if isinstance(prompt, str) and prompt.strip() else None
            direct_model, language, prompt = _apply_pre_transcription_hook(
                file_path=final_url, provider=provider, model=model,
                language=_get_stt_section(stt_config, provider).get("language"),
                prompt=prompt, source=source,
            )
            model_name = _builtin_model_name(provider, stt_config, direct_model)
            language = language or _resolve_stt_language(provider, stt_config)
            prompt = _enforce_prompt_length_limit(prompt, provider)
            direct_result = _transcribe_groq_remote_url(
                final_url, model_name, language=language, prompt=prompt)
            if direct_result.get("success"):
                direct_result["source_url"] = final_url
                return direct_result
            if not _looks_like_provider_size_limit_error(str(direct_result.get("error") or "")):
                return direct_result

    with tempfile.TemporaryDirectory(prefix="hermes-stt-remote-") as work_dir:
        downloaded_path, download_error = _download_remote_audio(
            final_url,
            work_dir,
            size_hint=content_length if isinstance(content_length, int) else None,
            content_type=content_type,
            max_download_bytes=max_download_bytes,
        )
        if download_error:
            return {"success": False, "transcript": "", "error": download_error}
        if downloaded_path is None:
            return {"success": False, "transcript": "", "error": "Remote audio download failed"}

        try:
            downloaded_size = Path(downloaded_path).stat().st_size
        except OSError as exc:
            return {"success": False, "transcript": "", "error": f"Failed to inspect downloaded audio: {exc}"}

        if downloaded_size > MAX_FILE_SIZE:
            return _transcribe_audio_file_chunks(
                downloaded_path,
                provider,
                stt_config,
                model,
                max_chunk_bytes=max_chunk_bytes,
                source=source,
            )

        validation_error = _validate_audio_file(downloaded_path)
        if validation_error:
            return validation_error
        result = _transcribe_prepared_audio(downloaded_path, model, source)
        if result.get("success"):
            result["source_url"] = final_url
        return result


def _transcribe_groq_remote_url(
    audio_url: str, model_name: str, *, language: Optional[str] = None, prompt: Optional[str] = None) -> Dict[str, Any]:
    """Transcribe a resolved remote audio URL using Groq's URL input."""
    from tools.transcription_tools import _resolve_provider_key

    api_key = _resolve_provider_key("GROQ_API_KEY", "groq")
    if not api_key:
        return {"success": False, "transcript": "", "error": "GROQ_API_KEY not set"}

    if model_name in OPENAI_MODELS:
        logger.info("Model %s not available on Groq, using %s", model_name, DEFAULT_GROQ_STT_MODEL)
        model_name = DEFAULT_GROQ_STT_MODEL

    try:
        import requests
    except ImportError:
        return {"success": False, "transcript": "", "error": "requests package not installed"}

    fields = {
        "model": (None, model_name),
        "url": (None, audio_url),
        "response_format": (None, "text"),
    }
    if language:
        fields["language"] = (None, language)
    if prompt:
        fields["prompt"] = (None, prompt)
    try:
        response = requests.post(
            f"{GROQ_BASE_URL.rstrip('/')}/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            files=fields,
            timeout=120,
        )
        if response.status_code != 200:
            detail = ""
            try:
                error_body = response.json()
                detail = error_body.get("error", {}).get("message", "") or response.text[:300]
            except Exception:
                detail = response.text[:300]
            return {
                "success": False,
                "transcript": "",
                "error": f"Groq STT API error (HTTP {response.status_code}): {detail}",
            }

        transcript_text = response.text.strip()
        logger.info(
            "Transcribed remote URL via Groq API (%s, %d chars)",
            model_name,
            len(transcript_text),
        )

        return {"success": True, "transcript": transcript_text, "provider": "groq"}

    except requests.ConnectionError as e:
        return {"success": False, "transcript": "", "error": f"Connection error: {e}"}
    except requests.Timeout as e:
        return {"success": False, "transcript": "", "error": f"Request timeout: {e}"}
    except requests.RequestException as e:
        return {"success": False, "transcript": "", "error": f"API error: {e}"}
    except Exception as e:
        logger.error("Groq remote URL transcription failed: %s", e, exc_info=True)
        return {"success": False, "transcript": "", "error": f"Transcription failed: {e}"}
