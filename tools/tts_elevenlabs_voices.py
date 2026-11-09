"""ElevenLabs voice search behind ``text_to_speech``'s ``find_voices`` argument.

The argument (and ``voice_id``) only exist in the ElevenLabs variant of the ``text_to_speech``
schema, so other providers never see voice search. Two sources share one paginated shape:
``library`` (``GET /v1/shared-voices``: the public Voice Library, the default) and ``account``
(``GET /v2/voices``: premade + the user's saved, cloned and designed voices). ``next_page`` is
opaque to the model: a page number for the library and a cursor token for the account.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

import httpx

from tools.registry import tool_error

logger = logging.getLogger(__name__)

DEFAULT_ELEVENLABS_BASE_URL = "https://api.elevenlabs.io"
SOURCES = ("library", "account")
# GET /v1/shared-voices ``sort`` keys; the default ranks by a year of real usage.
LIBRARY_SORTS = ("usage_character_count_1y", "trending", "cloned_by_count", "created_date")
_DEFAULT_PAGE_SIZE = 20
_MAX_PAGE_SIZE = 100
_DESCRIPTION_MAX_CHARS = 240
_TIMEOUT_SECONDS = 20.0
_FILTERS = ("gender", "age", "accent", "language")

FIND_VOICES_SCHEMA = {
    "type": "object",
    "description": (
        "Search voices instead of speaking (text is then not needed). Returns a page of voices "
        "with voice_id, labels, description and preview_url; pass next_page back as page. "
        "source 'library' (default) = the public Voice Library, thousands of community voices, "
        "most-used first, usable directly by voice_id; 'account' = the user's own voices. The "
        "library search matches words in names and descriptions literally: when a role word "
        "('dwarf', 'blacksmith') finds nothing, describe the sound instead ('gravelly old', "
        "'deep growl') with gender/age and use_case 'characters_animation'. You can't hear "
        "voices: choose from labels and description, and give the user preview_url links when "
        "the pick matters."),
    "properties": {
        "query": {"type": "string", "description": "Free-text search, e.g. 'orc' or 'gravelly old'."},
        "source": {"type": "string", "enum": list(SOURCES)},
        "gender": {"type": "string", "description": "e.g. male, female, neutral."},
        "age": {"type": "string", "description": "e.g. young, middle_aged, old."},
        "accent": {"type": "string", "description": "e.g. british, american, scottish."},
        "language": {"type": "string", "description": "ISO 639-1 code, e.g. en."},
        "use_case": {"type": "string", "description": (
            "Library only, e.g. characters_animation, narrative_story, conversational.")},
        "sort": {"type": "string", "enum": list(LIBRARY_SORTS),
                 "description": "Library only; default usage_character_count_1y (most used)."},
        "page": {"type": "string", "description": "next_page from the previous result."},
        "page_size": {"type": "integer", "description": "1-100, default 20."},
    },
}


def _clip(text: Any) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= _DESCRIPTION_MAX_CHARS else text[: _DESCRIPTION_MAX_CHARS - 1] + "…"


def _voice_row(voice: dict[str, Any]) -> dict[str, Any]:
    """One compact voice: library rows carry the traits at top level, account rows under ``labels``."""
    raw_labels = voice.get("labels")
    labels: dict[str, Any] = raw_labels if isinstance(raw_labels, dict) else {}
    row = {"voice_id": voice.get("voice_id"), "name": voice.get("name")}
    for key in ("gender", "age", "accent", "language", "descriptive", "use_case"):
        value = voice.get(key) or labels.get(key) or labels.get(key.replace("_", " "))
        if value:
            row[key] = value
    if voice.get("category"):
        row["category"] = voice["category"]
    if voice.get("description"):
        row["description"] = _clip(voice["description"])
    if voice.get("rate") not in (None, 1, 1.0):  # library voice billed at a custom credit multiplier
        row["credit_rate"] = voice["rate"]
    if voice.get("preview_url"):
        row["preview_url"] = voice["preview_url"]
    return row


def _request_params(source: str, query: str, filters: dict[str, str], use_case: str, sort: str,
                    page: str, page_size: int) -> tuple[str, dict[str, Any]]:
    """``(path, params)`` for one source; raises ValueError on a bad cursor or sort."""
    params: dict[str, Any] = {"page_size": page_size, **filters}
    if query:
        params["search"] = query
    if source == "account":
        params["include_total_count"] = "false"
        if page:
            params["next_page_token"] = page
        return "/v2/voices", params
    if page and not page.isdigit():
        raise ValueError(f"Invalid library page {page!r}: pass next_page from the previous result.")
    if sort and sort not in LIBRARY_SORTS:
        raise ValueError(f"find_voices.sort must be one of {', '.join(LIBRARY_SORTS)}")
    params["sort"] = sort or LIBRARY_SORTS[0]
    if use_case:
        params["use_cases"] = [use_case]
    params["page"] = int(page or 0)
    return "/v1/shared-voices", params


def _next_page(source: str, body: dict[str, Any], page: str) -> Optional[str]:
    if not body.get("has_more"):
        return None
    if source == "account":
        return body.get("next_page_token") or None
    return str(int(page or 0) + 1)


def search_voices(find: dict[str, Any], tts_config: dict[str, Any], api_key: str) -> str:
    """Search one ElevenLabs voice source and return a compact, paginated JSON page."""
    def arg(name: str) -> str:
        return str(find.get(name) or "").strip()

    source = (arg("source") or SOURCES[0]).lower()
    if source not in SOURCES:
        return tool_error(f"find_voices.source must be one of {', '.join(SOURCES)}", success=False)
    try:
        size = max(1, min(_MAX_PAGE_SIZE, int(find.get("page_size") or _DEFAULT_PAGE_SIZE)))
    except (TypeError, ValueError):
        size = _DEFAULT_PAGE_SIZE
    filters = {name: arg(name).lower() for name in _FILTERS if arg(name)}
    page = arg("page")
    try:
        path, params = _request_params(source, arg("query"), filters, arg("use_case").lower(),
                                       arg("sort").lower(), page, size)
    except ValueError as exc:
        return tool_error(str(exc), success=False)
    el_config = tts_config.get("elevenlabs") or {}
    base_url = (el_config.get("base_url") or DEFAULT_ELEVENLABS_BASE_URL).rstrip("/")
    try:
        response = httpx.get(base_url + path, params=params, timeout=_TIMEOUT_SECONDS,
                             headers={"xi-api-key": api_key, "accept": "application/json"})
    except httpx.HTTPError as exc:
        return tool_error(f"ElevenLabs voice search failed: {exc}", success=False)
    if response.status_code != 200:
        logger.warning("ElevenLabs %s returned HTTP %s", path, response.status_code)
        return tool_error(f"ElevenLabs voice search failed (HTTP {response.status_code}): "
                          f"{response.text[:300]}", success=False)
    try:
        body = response.json()
    except ValueError:
        return tool_error("ElevenLabs voice search returned a non-JSON body", success=False)
    voices = [_voice_row(v) for v in body.get("voices") or [] if isinstance(v, dict)]
    result: dict[str, Any] = {
        "success": True, "source": source, "count": len(voices), "voices": voices,
        "next_page": _next_page(source, body, page),
        "configured_voice_id": el_config.get("voice_id") or None,
    }
    if source == "library" and body.get("total_count"):
        result["total"] = body["total_count"]
    return json.dumps(result, ensure_ascii=False)
