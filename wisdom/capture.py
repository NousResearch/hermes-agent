"""Capture operations for Hermes Wisdom Kernel."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from wisdom.classify import classify_capture, detect_explicit_trigger
from wisdom.config import load_wisdom_config
from wisdom.db import WisdomDB
from wisdom.models import (
    CaptureOutcome,
    Category,
    SourceType,
    VALID_CATEGORIES,
    VALID_SOURCE_TYPES,
    WisdomConfig,
)
from wisdom.redaction import detect_secret_like_text, ensure_salt, stable_hash


def effective_enabled(db: WisdomDB, config: WisdomConfig) -> bool:
    setting = db.get_setting("enabled")
    if setting is None:
        return config.enabled
    return setting.lower() in {"1", "true", "yes", "on"}


def effective_capture_mode(db: WisdomDB, config: WisdomConfig) -> str:
    setting = db.get_setting("capture_mode")
    mode = (setting or config.capture_mode or "explicit").lower()
    return "explicit" if mode not in {"off", "explicit"} else mode


def capture_text(
    text: str,
    *,
    channel: str = "gateway",
    source_kind: str = "text",
    session_key: object | None = None,
    message_ref: object | None = None,
    metadata: dict[str, Any] | None = None,
    config: WisdomConfig | None = None,
    db: WisdomDB | None = None,
    cleaned_text: str | None = None,
    category: Category | str | None = None,
    source_type: SourceType | str | None = None,
    context_note: str | None = None,
    require_enabled: bool = True,
) -> CaptureOutcome:
    config = config or load_wisdom_config()
    db = db or WisdomDB(config.db_path)
    db.init()

    if require_enabled and not effective_enabled(db, config):
        return CaptureOutcome("disabled", message="Wisdom is off.")
    if detect_secret_like_text(text):
        return CaptureOutcome("blocked_secret", message="Capture blocked because the text looks like it contains a secret.")
    if context_note and detect_secret_like_text(str(context_note)):
        return CaptureOutcome("blocked_secret", message="Capture blocked because the context looks like it contains a secret.")

    trigger = detect_explicit_trigger(text)
    cleaned = cleaned_text if cleaned_text is not None else (trigger.cleaned_text if trigger else text.strip())
    classification = classify_capture(text, cleaned, trigger)
    extracted_metadata, inferred_source_type = _source_context_metadata(text)
    category_override = _valid_category(category)
    source_type_override = _valid_source_type(source_type)
    metadata_source_type = _valid_source_type(inferred_source_type)
    chosen_source_type = source_type_override or metadata_source_type
    if category_override or chosen_source_type:
        classification = replace(
            classification,
            category=category_override or classification.category,
            source_type=chosen_source_type or classification.source_type,
            confidence=max(classification.confidence, 0.82),
        )
    salt = ensure_salt()
    raw_metadata = _safe_metadata(metadata or {})
    if trigger:
        raw_metadata["trigger"] = trigger.prefix
    for key, value in extracted_metadata.items():
        raw_metadata.setdefault(key, value)
    if context_note:
        safe_context_note = _metadata_value(context_note)
        raw_metadata["context_note"] = safe_context_note

    capture_metadata: dict[str, Any] = {"capture_version": 2}
    capture_metadata.update(extracted_metadata)
    if context_note:
        capture_metadata["context_note"] = _metadata_value(context_note)

    record = db.create_capture(
        original_text=text,
        cleaned_text=cleaned,
        classification=classification,
        channel=channel,
        source_kind=source_kind,
        session_key_hash=stable_hash(session_key, salt=salt, prefix="sess_"),
        message_ref_hash=stable_hash(message_ref, salt=salt, prefix="msg_"),
        raw_metadata=raw_metadata,
        capture_metadata=capture_metadata,
    )
    return CaptureOutcome("captured", capture=record)


def _safe_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    allowed: dict[str, Any] = {}
    for key, value in metadata.items():
        key_text = str(key)
        if key_text.lower() in {"chat_id", "user_id", "message_id", "thread_id", "phone", "platform_id"}:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            allowed[key_text] = value
    return allowed


def _source_context_metadata(text: str) -> tuple[dict[str, str], SourceType | None]:
    metadata: dict[str, str] = {}
    source_type: SourceType | None = None
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    for line in lines[:5]:
        label_match = re.match(r"^(source|context)\s*:\s*(.+)$", line, flags=re.IGNORECASE)
        if label_match:
            key = label_match.group(1).lower()
            value = _metadata_value(label_match.group(2))
            if value:
                metadata[key] = value
            continue

        note_match = re.match(
            r"^(podcast|book|article|meeting|conversation|quote)(?:\s+(?:note|idea))?\s*:\s*(.+)$",
            line,
            flags=re.IGNORECASE,
        )
        if note_match:
            kind = note_match.group(1).lower()
            source_type = _valid_source_type(kind)
            source_text, context_text = _split_inline_context(note_match.group(2))
            if source_text and "source" not in metadata:
                metadata["source"] = source_text
            if context_text and "context" not in metadata:
                metadata["context"] = context_text

    return metadata, source_type


def _split_inline_context(value: str) -> tuple[str, str | None]:
    parts = re.split(r"\bcontext\s*:\s*", value, maxsplit=1, flags=re.IGNORECASE)
    source = _metadata_value(parts[0])
    context = _metadata_value(parts[1]) if len(parts) > 1 else None
    return source, context


def _metadata_value(value: object) -> str:
    compact = " ".join(str(value or "").strip().split())
    return compact.rstrip(".")[:500]


def _valid_category(value: Category | str | None) -> Category | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text if text in VALID_CATEGORIES else None


def _valid_source_type(value: SourceType | str | None) -> SourceType | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text if text in VALID_SOURCE_TYPES else None
