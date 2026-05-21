#!/usr/bin/env python3
"""YouTube transcript tool for Hermes.

The tool intentionally focuses on transcript retrieval rather than doing the
summary itself.  That keeps the side-effect surface small: the model can call
``youtube_transcript`` to ground itself in the video's subtitles, then transform
that transcript into a summary, chapters, quotes, or any other requested format.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable, Optional
from urllib.parse import parse_qs, quote, urlparse
from urllib.request import urlopen

from tools.registry import registry, tool_result

_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
DEFAULT_MAX_CHARS = 20_000
OEMBED_TIMEOUT_SECONDS = 5


class YouTubeTranscriptDependencyError(RuntimeError):
    """Raised when youtube-transcript-api cannot be imported or lazily installed."""


ERROR_MESSAGES = {
    "INVALID_URL": "Expected a YouTube URL or 11-character video ID",
    "DEPENDENCY_MISSING": (
        "youtube-transcript-api is not installed. Enable the youtube extra "
        "or run: pip install youtube-transcript-api"
    ),
    "TRANSCRIPTS_DISABLED": "Transcripts are disabled for this video.",
    "NO_TRANSCRIPT_FOUND": (
        "No transcript could be retrieved for this video. Try another language "
        "or verify subtitles are available."
    ),
    "VIDEO_UNAVAILABLE": "The YouTube video is unavailable or private.",
    "AGE_RESTRICTED": "The YouTube video appears to be age restricted.",
    "IP_BLOCKED": "YouTube blocked transcript access from this network/IP.",
    "UNKNOWN_ERROR": "Failed to retrieve the YouTube transcript.",
}


def extract_video_id(url_or_id: str) -> str:
    """Extract a YouTube 11-character video ID from common URL forms.

    Supports normal watch URLs, youtu.be short links, Shorts, embeds, live
    links, and raw IDs.  Raises ``ValueError`` when no valid ID can be found.
    """
    candidate = (url_or_id or "").strip()
    if _VIDEO_ID_RE.fullmatch(candidate):
        return candidate

    parsed = urlparse(candidate)
    host = parsed.netloc.lower()
    path = parsed.path.strip("/")

    if host.endswith("youtube.com") or host.endswith("youtube-nocookie.com"):
        query_id = parse_qs(parsed.query).get("v", [None])[0]
        if query_id and _VIDEO_ID_RE.fullmatch(query_id):
            return query_id

        parts = [part for part in path.split("/") if part]
        for marker in ("shorts", "embed", "live"):
            if marker in parts:
                idx = parts.index(marker)
                if idx + 1 < len(parts) and _VIDEO_ID_RE.fullmatch(parts[idx + 1]):
                    return parts[idx + 1]

    if host.endswith("youtu.be"):
        first = path.split("/", 1)[0]
        if _VIDEO_ID_RE.fullmatch(first):
            return first

    # Last-resort regex for pasted URLs with extra wrapping text.
    match = re.search(r"(?:v=|youtu\.be/|shorts/|embed/|live/)([A-Za-z0-9_-]{11})", candidate)
    if match:
        return match.group(1)

    raise ValueError(ERROR_MESSAGES["INVALID_URL"])


def format_timestamp(seconds: float) -> str:
    """Convert seconds to ``M:SS`` or ``H:MM:SS``."""
    total = max(0, int(seconds or 0))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _coerce_languages(languages: Optional[str | Iterable[str]]) -> Optional[list[str]]:
    if languages is None:
        return None
    if isinstance(languages, str):
        items = languages.split(",")
    else:
        items = list(languages)
    cleaned = [str(item).strip() for item in items if str(item).strip()]
    return cleaned or None


def _coerce_max_chars(max_chars: Optional[int]) -> Optional[int]:
    """Normalize max_chars.

    ``None`` disables truncation for internal callers.  Public schema defaults to
    ``DEFAULT_MAX_CHARS`` to protect context windows for long videos.
    """
    if max_chars is None:
        return None
    try:
        value = int(max_chars)
    except (TypeError, ValueError):
        return DEFAULT_MAX_CHARS
    if value <= 0:
        return DEFAULT_MAX_CHARS
    return value


def _load_youtube_transcript_api():
    """Import youtube-transcript-api, lazily installing the pinned extra if allowed."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        return YouTubeTranscriptApi
    except ImportError:
        try:
            from tools.lazy_deps import ensure
            ensure("skill.youtube", prompt=False)
            from youtube_transcript_api import YouTubeTranscriptApi
            return YouTubeTranscriptApi
        except Exception as exc:  # pragma: no cover - exact exception depends on env/config
            raise YouTubeTranscriptDependencyError(ERROR_MESSAGES["DEPENDENCY_MISSING"]) from exc


def _snippet_to_dict(segment: Any) -> dict[str, Any]:
    """Normalize dict snippets and youtube-transcript-api objects."""
    if isinstance(segment, dict):
        text = segment.get("text", "")
        start = segment.get("start", 0.0)
        duration = segment.get("duration", 0.0)
    else:
        text = getattr(segment, "text", "")
        start = getattr(segment, "start", 0.0)
        duration = getattr(segment, "duration", 0.0)
    return {
        "text": str(text).replace("\n", " ").strip(),
        "start": float(start or 0.0),
        "duration": float(duration or 0.0),
    }


def _transcript_metadata(transcript: Any) -> dict[str, Any]:
    """Extract stable metadata from transcript-api transcript/list objects."""
    return {
        "language": getattr(transcript, "language", None),
        "language_code": getattr(transcript, "language_code", None),
        "is_generated": getattr(transcript, "is_generated", None),
        "is_translatable": getattr(transcript, "is_translatable", None),
    }


def _available_languages(video_id: str) -> list[dict[str, Any]]:
    """Return available transcript languages when the API supports listing.

    Best effort only: callers should treat an empty list as "unknown", not as a
    proof that no languages exist.
    """
    try:
        api_cls = _load_youtube_transcript_api()
        api = api_cls()
        transcript_list = api.list(video_id)
    except TypeError:
        return []
    except Exception:
        return []

    languages: list[dict[str, Any]] = []
    try:
        iterator = iter(transcript_list)
    except TypeError:
        return languages
    for transcript in iterator:
        info = _transcript_metadata(transcript)
        if info.get("language_code"):
            languages.append(info)
    return languages


def _fetch_segments(
    video_id: str,
    languages: Optional[list[str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fetch and normalize transcript segments.

    Handles youtube-transcript-api v1.x (``YouTubeTranscriptApi().fetch``) and
    older class/static APIs used by some tests and existing environments.
    """
    api_cls = _load_youtube_transcript_api()
    fetched: Any

    try:
        api = api_cls()
        fetched = api.fetch(video_id, languages=languages) if languages else api.fetch(video_id)
    except TypeError:
        # Older youtube-transcript-api exposed get_transcript as a static/class method.
        fetched = api_cls.get_transcript(video_id, languages=languages) if languages else api_cls.get_transcript(video_id)

    metadata = _transcript_metadata(fetched)
    segments = [_snippet_to_dict(segment) for segment in fetched]
    return segments, metadata


def _truncate_text(text: str, max_chars: Optional[int]) -> tuple[str, bool, int, int]:
    """Return ``(returned_text, truncated, original_chars, returned_chars)``."""
    original_chars = len(text)
    if max_chars is None or original_chars <= max_chars:
        return text, False, original_chars, original_chars
    marker = "\n...[truncated]"
    if max_chars <= len(marker):
        truncated = text[:max_chars]
    else:
        truncated = text[: max_chars - len(marker)].rstrip() + marker
    return truncated, True, original_chars, len(truncated)


def _classify_error(exc: Exception) -> tuple[str, str]:
    """Map transcript API failures to stable error codes and user-safe messages."""
    if isinstance(exc, YouTubeTranscriptDependencyError):
        return "DEPENDENCY_MISSING", ERROR_MESSAGES["DEPENDENCY_MISSING"]

    raw = str(exc) or type(exc).__name__
    lower = raw.lower()
    if "disabled" in lower:
        return "TRANSCRIPTS_DISABLED", ERROR_MESSAGES["TRANSCRIPTS_DISABLED"]
    if "no transcript" in lower or "could not retrieve" in lower or "notranscript" in lower:
        return "NO_TRANSCRIPT_FOUND", ERROR_MESSAGES["NO_TRANSCRIPT_FOUND"]
    if "unavailable" in lower or "private" in lower or "video unavailable" in lower:
        return "VIDEO_UNAVAILABLE", ERROR_MESSAGES["VIDEO_UNAVAILABLE"]
    if "age" in lower and "restrict" in lower:
        return "AGE_RESTRICTED", ERROR_MESSAGES["AGE_RESTRICTED"]
    if "ip" in lower and ("block" in lower or "ban" in lower):
        return "IP_BLOCKED", ERROR_MESSAGES["IP_BLOCKED"]
    if "too many requests" in lower or "429" in lower:
        return "IP_BLOCKED", ERROR_MESSAGES["IP_BLOCKED"]
    return "UNKNOWN_ERROR", raw


def _error_result(
    code: str,
    message: str,
    *,
    video_id: Optional[str] = None,
    available_languages: Optional[list[dict[str, Any]]] = None,
) -> str:
    payload: dict[str, Any] = {
        "success": False,
        "error": message,
        "error_code": code,
        "error_message": message,
    }
    if video_id is not None:
        payload["video_id"] = video_id
    if available_languages is not None:
        payload["available_languages"] = available_languages
    return tool_result(payload)


def _fetch_oembed_metadata(video_id: str) -> Optional[dict[str, Any]]:
    """Fetch lightweight public YouTube metadata via keyless oEmbed.

    Failure is non-fatal because the transcript is the primary evidence source.
    """
    canonical_url = f"https://www.youtube.com/watch?v={video_id}"
    endpoint = f"https://www.youtube.com/oembed?url={quote(canonical_url, safe='')}&format=json"
    try:
        with urlopen(endpoint, timeout=OEMBED_TIMEOUT_SECONDS) as response:  # noqa: S310 - fixed HTTPS endpoint
            data = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None

    return {
        "title": data.get("title"),
        "author_name": data.get("author_name"),
        "author_url": data.get("author_url"),
        "provider_name": data.get("provider_name"),
        "provider_url": data.get("provider_url"),
        "thumbnail_url": data.get("thumbnail_url"),
    }


def youtube_transcript_tool(
    url: str,
    languages: Optional[str | Iterable[str]] = None,
    include_timestamps: bool = True,
    include_segments: bool = False,
    max_chars: Optional[int] = DEFAULT_MAX_CHARS,
    include_metadata: bool = True,
) -> str:
    """Fetch a YouTube transcript as JSON for downstream model analysis."""
    try:
        video_id = extract_video_id(url)
    except ValueError as exc:
        return _error_result("INVALID_URL", str(exc))

    language_list = _coerce_languages(languages)
    normalized_max_chars = _coerce_max_chars(max_chars)
    try:
        segments, transcript_metadata = _fetch_segments(video_id, language_list)
    except Exception as exc:
        code, message = _classify_error(exc)
        return _error_result(
            code,
            message,
            video_id=video_id,
            available_languages=_available_languages(video_id),
        )

    full_text_raw = " ".join(segment["text"] for segment in segments if segment["text"]).strip()
    timestamped_text_raw = "\n".join(
        f"{format_timestamp(segment['start'])} {segment['text']}"
        for segment in segments
        if segment["text"]
    )
    full_text, full_truncated, full_chars, returned_full_chars = _truncate_text(
        full_text_raw, normalized_max_chars)
    timestamped_text, timestamps_truncated, timestamp_chars, returned_timestamp_chars = _truncate_text(
        timestamped_text_raw, normalized_max_chars)

    duration_seconds = 0.0
    if segments:
        last = segments[-1]
        duration_seconds = float(last["start"]) + float(last["duration"])

    language_code = transcript_metadata.get("language_code")
    result: dict[str, Any] = {
        "success": True,
        "video_id": video_id,
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "analysis_source": "youtube_transcript",
        "source_limitations": [
            "Transcript-based understanding only; video frames and audio are not analyzed.",
            "Automatically generated captions may contain recognition errors.",
        ],
        "language": language_code,
        "requested_languages": language_list,
        "is_generated": transcript_metadata.get("is_generated"),
        "is_translatable": transcript_metadata.get("is_translatable"),
        "segment_count": len(segments),
        "duration": format_timestamp(duration_seconds),
        "duration_seconds": duration_seconds,
        "full_text": full_text,
        "truncated": full_truncated or timestamps_truncated,
        "text_stats": {
            "max_chars": normalized_max_chars,
            "full_text_chars": full_chars,
            "returned_full_text_chars": returned_full_chars,
            "full_text_truncated": full_truncated,
            "timestamped_text_chars": timestamp_chars,
            "returned_timestamped_text_chars": returned_timestamp_chars,
            "timestamped_text_truncated": timestamps_truncated,
        },
    }
    if transcript_metadata.get("language") is not None:
        result["language_name"] = transcript_metadata.get("language")
    if include_metadata:
        result["metadata"] = _fetch_oembed_metadata(video_id)
    if include_timestamps:
        result["timestamped_text"] = timestamped_text
    if include_segments:
        result["segments"] = segments
    return tool_result(result)


YOUTUBE_TRANSCRIPT_SCHEMA = {
    "name": "youtube_transcript",
    "description": (
        "Fetch the transcript/subtitles for a YouTube video URL or video ID so the "
        "assistant can understand and summarize the video. Use this before answering "
        "requests about YouTube content. It returns transcript text, stable metadata, "
        "truncation stats, and optionally timestamped lines or raw segments; the "
        "assistant should then produce the requested summary, chapters, quotes, or analysis."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "A YouTube URL (watch, youtu.be, shorts, embed, live) or raw 11-character video ID.",
            },
            "languages": {
                "type": "string",
                "description": "Optional comma-separated transcript language preference/fallback chain, e.g. 'ko,en' or 'en'. Leave empty for YouTube's default selection.",
            },
            "include_timestamps": {
                "type": "boolean",
                "description": "Whether to include timestamped transcript lines. Defaults to true.",
            },
            "include_segments": {
                "type": "boolean",
                "description": "Whether to include raw segment objects with start/duration. Defaults to false to keep output compact.",
            },
            "max_chars": {
                "type": "integer",
                "description": "Maximum characters to return for each transcript text field. Defaults to 20000 to protect context windows; set higher for long-video analysis.",
            },
            "include_metadata": {
                "type": "boolean",
                "description": "Whether to fetch lightweight public video metadata via YouTube oEmbed. Defaults to true.",
            },
        },
        "required": ["url"],
    },
}


registry.register(
    name="youtube_transcript",
    toolset="web",
    schema=YOUTUBE_TRANSCRIPT_SCHEMA,
    handler=lambda args, **kw: youtube_transcript_tool(
        url=args.get("url", ""),
        languages=args.get("languages"),
        include_timestamps=args.get("include_timestamps", True),
        include_segments=args.get("include_segments", False),
        max_chars=args.get("max_chars", DEFAULT_MAX_CHARS),
        include_metadata=args.get("include_metadata", True),
    ),
    emoji="▶️",
    max_result_size_chars=120_000,
)
