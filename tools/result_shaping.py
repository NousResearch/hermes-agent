"""Shared helpers for compacting large model-facing tool outputs."""

from __future__ import annotations

import os
from dataclasses import dataclass


_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
_DISABLE_RESULT_COMPACTION_ENV = "HERMES_DISABLE_RESULT_COMPACTION"


def normalize_result_mode(value: str | None) -> str:
    if isinstance(value, str) and value.lower() in {"auto", "full", "preview"}:
        return value.lower()
    return "auto"


def result_compaction_disabled() -> bool:
    return os.getenv(_DISABLE_RESULT_COMPACTION_ENV, "").strip().lower() in _TRUE_ENV_VALUES


@dataclass(frozen=True)
class CompactedText:
    text: str
    metadata: dict


def compact_text_output(
    text: str,
    *,
    result_mode: str = "auto",
    field_name: str = "output",
    threshold_chars: int = 20_000,
    preview_chars: int = 8_000,
    head_ratio: float = 0.45,
    hint: str = "",
) -> CompactedText:
    """Return head/tail text plus metadata when output should be compacted.

    The input text is never stored in metadata. Counts and a continuation hint
    are safe to expose even when the omitted content may contain sensitive data.
    """
    result_mode = normalize_result_mode(result_mode)
    if result_compaction_disabled() or result_mode == "full" or not isinstance(text, str):
        return CompactedText(text=text, metadata={})

    if preview_chars < 200:
        preview_chars = 200
    should_compact = result_mode == "preview" or len(text) > threshold_chars
    if not should_compact or len(text) <= preview_chars:
        return CompactedText(text=text, metadata={})

    head_chars = max(1, int(preview_chars * head_ratio))
    tail_chars = max(1, preview_chars - head_chars)
    omitted_chars = max(len(text) - head_chars - tail_chars, 0)
    if omitted_chars <= 0:
        return CompactedText(text=text, metadata={})

    total_lines = text.count("\n") + (1 if text else 0)
    omitted_text = text[head_chars:-tail_chars]
    omitted_lines = omitted_text.count("\n")
    notice = (
        f"\n\n... [{field_name.upper()} COMPACTED - {omitted_chars} chars"
        f" omitted from {len(text)} total"
    )
    if total_lines:
        notice += f"; {omitted_lines} lines omitted from {total_lines} total"
    notice += "] ...\n\n"

    metadata = {
        "compacted": True,
        f"{field_name}_original_chars": len(text),
        f"{field_name}_preview_chars": len(text[:head_chars] + notice + text[-tail_chars:]),
        f"{field_name}_omitted_chars": omitted_chars,
        f"{field_name}_total_lines": total_lines,
        f"{field_name}_omitted_lines": omitted_lines,
    }
    if hint:
        metadata["_hint"] = hint

    return CompactedText(
        text=text[:head_chars] + notice + text[-tail_chars:],
        metadata=metadata,
    )
