"""Prompt echo guard for hosted whisper-family STT.

Whisper reads ``prompt`` as the transcript just before the audio, so a weak decode can return the
prompt itself as an ordinary success. A near-verbatim copy is retried once with the prompt withheld.
"""

from __future__ import annotations

import difflib
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Optional

from tools.transcription_command import _WHISPER_PROMPT_CAPPED_PROVIDERS
from tools.transcription_common import CLOUD_STT_PROVIDERS

logger = logging.getLogger("tools.transcription_tools")

# Hosted APIs that condition on the prompt. The other backends ignore it, and a local retry would
# keep ``stt.local.initial_prompt`` (faster-whisper also drops low confidence segments itself).
_GUARDED_PROVIDERS = _WHISPER_PROMPT_CAPPED_PROVIDERS & CLOUD_STT_PROVIDERS
# A short reply may legitimately use the prompt's vocabulary.
_MIN_CHARS = 25
# An echo is a near-verbatim copy: nearly all of the prompt, in order, and almost nothing else.
# Corrupted echoes measured 0.97 or more on both; speech adding even one word ("and") 0.86 or less.
_PROMPT_COVERAGE = 0.9
_TRANSCRIPT_COVERAGE = 0.9


def _normalize(text: str) -> str:
    """Lowercase, accent stripped, letters and digits of any script (so non-Latin text still compares)."""
    folded = unicodedata.normalize("NFKD", text.lower())
    kept = [c if c.isalnum() else " " for c in folded if not unicodedata.combining(c)]
    return re.sub(r"\s+", " ", "".join(kept)).strip()


def is_prompt_echo(transcript: Optional[str], prompt: Optional[str]) -> bool:
    """True when *transcript* is a near-verbatim copy of *prompt*."""
    spoken, biased = _normalize(transcript or ""), _normalize(prompt or "")
    if not biased or len(spoken) < _MIN_CHARS or len(spoken) * _TRANSCRIPT_COVERAGE > len(biased):
        return False
    # autojunk would discard every common letter of a prompt over 200 characters and match nothing.
    matcher = difflib.SequenceMatcher(None, spoken, biased, autojunk=False)
    reproduced = sum(block.size for block in matcher.get_matching_blocks())
    return reproduced >= _PROMPT_COVERAGE * len(biased) and reproduced >= _TRANSCRIPT_COVERAGE * len(spoken)


def transcribe_with_prompt_echo_guard(
    file_path: str, provider: str, stt_config: dict[str, Any], model: Optional[str],
    language: Optional[str], prompt: Optional[str]) -> dict[str, Any]:
    """Route to *provider*; when a guarded backend returns a copy of the prompt, retry once with only the
    prompt withheld. That decode never saw the prompt, so its result stands as is, even if it matches."""
    from tools.transcription_tools import _route_stt_provider
    result = _route_stt_provider(file_path, provider, stt_config, model, language, prompt)
    if provider not in _GUARDED_PROVIDERS or not is_prompt_echo(result.get("transcript"), prompt):
        return result
    logger.warning("STT provider '%s' returned its own prompt as the transcript for %s; retrying once "
                   "without the prompt", provider, Path(file_path).name)
    return _route_stt_provider(file_path, provider, stt_config, model, language, None)
