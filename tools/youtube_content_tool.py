#!/usr/bin/env python3
"""
YouTube Content Tool -- transcript to structured content formats.

Fetches a YouTube transcript and transforms it into creator-ready content
formats (chapters, summary, thread, blog, quotes, etc.) using Hermes'
auxiliary LLM client.
"""

import json
import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from agent.auxiliary_client import get_text_auxiliary_client

logger = logging.getLogger(__name__)

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api import _errors as yta_errors
    _YTA_IMPORT_ERROR: Optional[str] = None
except Exception as exc:  # pragma: no cover - import gate
    YouTubeTranscriptApi = None
    yta_errors = None
    _YTA_IMPORT_ERROR = str(exc)


VALID_OUTPUT_FORMATS = {
    "chapters",
    "summary",
    "chapter_summaries",
    "thread",
    "blog",
    "quotes",
    "all",
}

MAX_TRANSCRIPT_CHARS = 120_000


def _format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _extract_video_id(url: str) -> Tuple[Optional[str], Optional[str]]:
    if not url or not isinstance(url, str):
        return None, "URL is required."

    url = url.strip()
    if not url:
        return None, "URL is required."

    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):
        return url, None

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return None, "Invalid YouTube URL. Use http(s)://youtube.com/... or https://youtu.be/..."

    host = (parsed.netloc or "").lower()
    path = (parsed.path or "").strip("/")
    query = parse_qs(parsed.query)

    video_id = None
    if host in {"youtu.be", "www.youtu.be"}:
        video_id = path.split("/")[0] if path else None
    elif host.endswith("youtube.com") or host.endswith("youtube-nocookie.com"):
        if path == "watch":
            video_id = (query.get("v") or [None])[0]
        elif path.startswith("shorts/"):
            video_id = path.split("/", 1)[1].split("/")[0]
        elif path.startswith("embed/"):
            video_id = path.split("/", 1)[1].split("/")[0]
        elif path.startswith("live/"):
            video_id = path.split("/", 1)[1].split("/")[0]

    if not video_id or not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        return None, "Could not extract a valid YouTube video ID from the URL."
    return video_id, None


def _normalize_language_preference(language: str) -> List[str]:
    if not language or language == "auto":
        return []
    raw = [p.strip() for p in str(language).replace(";", ",").split(",")]
    langs = [p for p in raw if p]
    return langs or []


def _snippet_to_dict(item: Any) -> Dict[str, Any]:
    if isinstance(item, dict):
        return {
            "text": str(item.get("text", "")).strip(),
            "start": float(item.get("start", 0.0) or 0.0),
            "duration": float(item.get("duration", 0.0) or 0.0),
        }
    return {
        "text": str(getattr(item, "text", "")).strip(),
        "start": float(getattr(item, "start", 0.0) or 0.0),
        "duration": float(getattr(item, "duration", 0.0) or 0.0),
    }


def _fetch_transcript(video_id: str, language: str = "auto") -> Dict[str, Any]:
    if YouTubeTranscriptApi is None:
        return {
            "success": False,
            "error": f"youtube-transcript-api is not installed ({_YTA_IMPORT_ERROR}).",
        }

    api = YouTubeTranscriptApi()
    requested_languages = _normalize_language_preference(language)

    try:
        transcript_obj = None
        transcript_meta = {
            "video_id": video_id,
            "language": None,
            "language_code": None,
            "is_generated": None,
        }

        if requested_languages:
            fetched = api.fetch(video_id, languages=requested_languages, preserve_formatting=False)
        else:
            transcript_list = api.list(video_id)
            preferred_auto = ["tr", "en", "en-US", "en-GB"]
            finders = (
                transcript_list.find_manually_created_transcript,
                transcript_list.find_generated_transcript,
                transcript_list.find_transcript,
            )

            last_err = None
            for finder in finders:
                try:
                    transcript_obj = finder(preferred_auto)
                    break
                except Exception as exc:
                    last_err = exc

            if transcript_obj is None:
                try:
                    transcript_obj = next(iter(transcript_list))
                except StopIteration:
                    if last_err:
                        raise last_err
                    raise

            fetched = transcript_obj.fetch()

        if transcript_obj is not None:
            transcript_meta["language"] = getattr(transcript_obj, "language", None)
            transcript_meta["language_code"] = getattr(transcript_obj, "language_code", None)
            transcript_meta["is_generated"] = getattr(transcript_obj, "is_generated", None)

        if hasattr(fetched, "snippets"):
            snippets = [_snippet_to_dict(s) for s in getattr(fetched, "snippets", [])]
            transcript_meta["language"] = transcript_meta["language"] or getattr(fetched, "language", None)
            transcript_meta["language_code"] = transcript_meta["language_code"] or getattr(fetched, "language_code", None)
            transcript_meta["is_generated"] = transcript_meta["is_generated"]
            if transcript_meta["is_generated"] is None:
                transcript_meta["is_generated"] = getattr(fetched, "is_generated", None)
        elif hasattr(fetched, "to_raw_data"):
            snippets = [_snippet_to_dict(s) for s in fetched.to_raw_data()]
        else:
            snippets = [_snippet_to_dict(s) for s in list(fetched)]

        snippets = [s for s in snippets if s.get("text")]
        if not snippets:
            return {"success": False, "error": "Transcript was fetched but contained no usable text."}

        lines = [f"{_format_timestamp(s['start'])} {s['text']}" for s in snippets]
        transcript_text = "\n".join(lines)
        truncated = False
        if len(transcript_text) > MAX_TRANSCRIPT_CHARS:
            head = transcript_text[: int(MAX_TRANSCRIPT_CHARS * 0.75)]
            tail = transcript_text[-int(MAX_TRANSCRIPT_CHARS * 0.2):]
            transcript_text = (
                head
                + "\n\n[...transcript truncated for token budget...]\n\n"
                + tail
            )
            truncated = True

        return {
            "success": True,
            "video_id": video_id,
            "snippets": snippets,
            "transcript_text": transcript_text,
            "transcript_line_count": len(lines),
            "transcript_chars": len(transcript_text),
            "truncated": truncated,
            "language": transcript_meta.get("language"),
            "language_code": transcript_meta.get("language_code"),
            "is_generated": transcript_meta.get("is_generated"),
        }

    except Exception as exc:
        err_name = type(exc).__name__
        err_text = str(exc).strip() or err_name

        if yta_errors is not None:
            if isinstance(exc, getattr(yta_errors, "InvalidVideoId", tuple())):
                return {"success": False, "error": "Invalid YouTube video ID or URL."}
            if isinstance(exc, getattr(yta_errors, "TranscriptsDisabled", tuple())):
                return {"success": False, "error": "Transcripts are disabled for this video."}
            if isinstance(exc, getattr(yta_errors, "NoTranscriptFound", tuple())):
                return {"success": False, "error": "No transcript found for this video (requested language may be unavailable)."}
            if isinstance(exc, getattr(yta_errors, "VideoUnavailable", tuple())):
                return {"success": False, "error": "Video is unavailable (deleted, private, or restricted)."}
            if isinstance(exc, getattr(yta_errors, "VideoUnplayable", tuple())):
                return {"success": False, "error": "Video is not playable (possibly private or restricted)."}
            if isinstance(exc, getattr(yta_errors, "AgeRestricted", tuple())):
                return {"success": False, "error": "Video is age-restricted and transcript could not be retrieved."}
            if isinstance(exc, getattr(yta_errors, "CouldNotRetrieveTranscript", tuple())):
                return {"success": False, "error": f"Could not retrieve transcript: {err_text}"}

        logger.warning("youtube_content transcript fetch failed for %s: %s", video_id, err_text)
        return {"success": False, "error": f"Failed to fetch transcript ({err_name}): {err_text}"}


def _language_instruction(language: str, transcript_language: Optional[str], transcript_language_code: Optional[str]) -> str:
    if language and language != "auto":
        return (
            f"Write the output in language '{language}'. "
            "If that exact locale is not natural, use the closest language variant."
        )
    if transcript_language:
        label = transcript_language
        if transcript_language_code:
            label += f" ({transcript_language_code})"
        return f"Write the output in the same language as the transcript: {label}."
    return "Write the output in the transcript's original language."


def _build_user_prompt(
    *,
    output_format: str,
    url: str,
    transcript_text: str,
    language_instruction: str,
    transcript_meta: Dict[str, Any],
) -> str:
    meta_block = (
        f"URL: {url}\n"
        f"Video ID: {transcript_meta.get('video_id')}\n"
        f"Transcript language: {transcript_meta.get('language') or 'unknown'}\n"
        f"Language code: {transcript_meta.get('language_code') or 'unknown'}\n"
        f"Generated transcript: {transcript_meta.get('is_generated')}\n"
        f"Transcript lines: {transcript_meta.get('transcript_line_count')}\n"
        f"Transcript truncated: {transcript_meta.get('truncated')}\n"
    )

    prompts = {
        "chapters": (
            "Create chapter timestamps from the transcript.\n"
            "Output format rules:\n"
            "- One line per chapter\n"
            "- Format exactly: MM:SS Chapter Title (or HH:MM:SS if needed)\n"
            "- Use concise titles\n"
            "- Cover the whole video without tiny over-segmentation\n"
            "- No bullets, no numbering, no extra commentary"
        ),
        "summary": (
            "Write a 5-10 sentence summary of the video.\n"
            "- Focus on key arguments, takeaways, and conclusions\n"
            "- No timestamps unless essential\n"
            "- No bullet list unless the transcript is very list-like"
        ),
        "chapter_summaries": (
            "Create chapter timestamps and a short summary for each chapter.\n"
            "Output format rules:\n"
            "- For each chapter, first line: MM:SS Chapter Title\n"
            "- Then exactly 2 sentences summarizing that chapter\n"
            "- Blank line between chapters\n"
            "- Cover the full video"
        ),
        "thread": (
            "Convert the transcript into a Twitter/X thread.\n"
            "Output format rules:\n"
            "- Create a strong hook tweet first\n"
            "- Then a threaded sequence of concise posts\n"
            "- Keep each post reasonably short and readable\n"
            "- Preserve the speaker's core claims without hype inflation\n"
            "- Include a final takeaway/CTA post\n"
            "- Number tweets like '1/' '2/' etc."
        ),
        "blog": (
            "Write a full blog post based on the transcript.\n"
            "Output format rules:\n"
            "- Include a strong title\n"
            "- Include section headings\n"
            "- Organize the content into a coherent article, not a transcript rewrite\n"
            "- Preserve important specifics and examples\n"
            "- End with a short conclusion"
        ),
        "quotes": (
            "Extract the most important quotes from the transcript.\n"
            "Output format rules:\n"
            "- Only include quotes supported by the transcript text provided\n"
            "- Prefer verbatim phrases when possible\n"
            "- Include timestamps\n"
            "- Format each item as: MM:SS \"Quote text\" - Why it matters\n"
            "- Return 8-20 quotes depending on transcript richness"
        ),
        "all": (
            "Produce all output formats in one response using the exact section order below:\n"
            "## Chapters\n"
            "## Summary\n"
            "## Chapter Summaries\n"
            "## Thread\n"
            "## Blog\n"
            "## Quotes\n\n"
            "Apply the formatting rules implied by each format (chapters timestamps, 5-10 sentence summary, "
            "chapter titles + 2-sentence summaries, numbered X thread, titled/sectioned blog post, timestamped quotes)."
        ),
    }

    return (
        f"{prompts[output_format]}\n\n"
        f"{language_instruction}\n\n"
        "Video metadata:\n"
        f"{meta_block}\n"
        "Transcript (timestamped lines):\n"
        f"{transcript_text}"
    )


def _max_tokens_for_format(output_format: str) -> int:
    return {
        "chapters": 1200,
        "summary": 900,
        "chapter_summaries": 2200,
        "thread": 1800,
        "blog": 3200,
        "quotes": 1800,
        "all": 7000,
    }.get(output_format, 1600)


def youtube_content(url: str, output_format: str = "chapters", language: str = "auto") -> Dict[str, Any]:
    """
    Convert a YouTube transcript into structured content.

    Returns a dict. The registry handler wraps it in JSON for the model.
    """
    output_format = (output_format or "chapters").strip().lower()
    language = (language or "auto").strip()

    if output_format not in VALID_OUTPUT_FORMATS:
        return {
            "success": False,
            "error": (
                f"Invalid output_format '{output_format}'. "
                f"Use one of: {', '.join(sorted(VALID_OUTPUT_FORMATS))}"
            ),
        }

    video_id, err = _extract_video_id(url)
    if err:
        return {"success": False, "error": err}

    transcript = _fetch_transcript(video_id, language=language)
    if not transcript.get("success"):
        return transcript

    client, model = get_text_auxiliary_client()
    if client is None or model is None:
        return {
            "success": False,
            "error": (
                "No auxiliary text model is available for transcript transformation. "
                "Configure OpenRouter, Nous Portal, or a custom OpenAI-compatible endpoint."
            ),
        }

    transcript_meta = {
        "video_id": video_id,
        "language": transcript.get("language"),
        "language_code": transcript.get("language_code"),
        "is_generated": transcript.get("is_generated"),
        "transcript_line_count": transcript.get("transcript_line_count"),
        "truncated": transcript.get("truncated"),
    }
    language_instruction = _language_instruction(
        language,
        transcript.get("language"),
        transcript.get("language_code"),
    )
    user_prompt = _build_user_prompt(
        output_format=output_format,
        url=url,
        transcript_text=transcript["transcript_text"],
        language_instruction=language_instruction,
        transcript_meta=transcript_meta,
    )

    system_prompt = (
        "You transform YouTube transcripts into structured content outputs. "
        "Stay faithful to the transcript, preserve key claims and details, and avoid inventing facts. "
        "When timestamps are requested, use transcript timestamps that align with the provided text."
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=_max_tokens_for_format(output_format),
            timeout=90.0,
        )
        content = (response.choices[0].message.content or "").strip()
        if not content:
            return {"success": False, "error": "LLM returned an empty response."}

        return {
            "success": True,
            "url": url,
            "video_id": video_id,
            "output_format": output_format,
            "language_requested": language,
            "transcript_language": transcript.get("language"),
            "transcript_language_code": transcript.get("language_code"),
            "transcript_is_generated": transcript.get("is_generated"),
            "transcript_line_count": transcript.get("transcript_line_count"),
            "transcript_chars": transcript.get("transcript_chars"),
            "transcript_truncated": transcript.get("truncated"),
            "model_used": model,
            "content": content,
        }
    except Exception as exc:
        logger.warning("youtube_content LLM transform failed for %s: %s", video_id, exc)
        return {"success": False, "error": f"Failed to transform transcript with LLM: {exc}"}


def check_youtube_content_requirements() -> bool:
    """Tool requires youtube-transcript-api and an auxiliary text model."""
    if YouTubeTranscriptApi is None:
        return False
    client, model = get_text_auxiliary_client()
    return client is not None and model is not None


YOUTUBE_CONTENT_SCHEMA = {
    "name": "youtube_content",
    "description": (
        "Fetch a YouTube video's transcript and convert it into structured content "
        "formats (chapters, summary, chapter summaries, thread, blog, quotes, or all). "
        "Use this for turning long videos into creator-ready outputs."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "YouTube video URL (youtube.com/watch?v=..., youtu.be/..., shorts URL) or a raw 11-char video ID.",
            },
            "output_format": {
                "type": "string",
                "enum": sorted(VALID_OUTPUT_FORMATS),
                "default": "chapters",
                "description": "Desired output format for the transcript transformation.",
            },
            "language": {
                "type": "string",
                "default": "auto",
                "description": (
                    "Transcript language preference. Use 'auto' to auto-select. "
                    "You can also pass a language code (e.g. 'tr', 'en') or comma-separated fallbacks (e.g. 'tr,en')."
                ),
            },
        },
        "required": ["url"],
    },
}


# --- Registry ---
from tools.registry import registry

registry.register(
    name="youtube_content",
    toolset="web",
    schema=YOUTUBE_CONTENT_SCHEMA,
    handler=lambda args, **kw: json.dumps(
        youtube_content(
            url=args.get("url", ""),
            output_format=args.get("output_format", "chapters"),
            language=args.get("language", "auto"),
        ),
        ensure_ascii=False,
    ),
    check_fn=check_youtube_content_requirements,
)

