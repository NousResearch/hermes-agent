"""``transform_spoken_text`` plugin seam shared by every path where Hermes speaks a reply.

Each surface calls :func:`apply_spoken_text_hook` after its TTS normalizer and before synthesis.
The first ``str`` result wins, empty string included (that surface then speaks nothing through
its existing empty-text path); ``None`` and any other value leave the text unchanged. Fail-open:
a raising, hung or broken plugin never suppresses the voice reply.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def apply_spoken_text_hook(
    text: str, *, surface: str, platform: str = "", session_id: str = "", index: int = 0,
) -> str:
    """Return the text to synthesize for one spoken reply (or one streamed sentence).

    ``surface`` is one of ``voice_reply``, ``auto_tts``, ``cli_voice``, ``read_aloud``,
    ``streaming``; ``index`` is the sentence's position in the reply on ``streaming`` and 0 on
    whole-reply surfaces."""
    try:
        from hermes_cli.lifecycle import has_hook, invoke_hook
        if not text or not has_hook("transform_spoken_text"):
            return text
        results = invoke_hook(
            "transform_spoken_text", text=text, surface=surface, platform=platform or "",
            session_id=session_id or "", index=index)
        return next((r for r in results if isinstance(r, str)), text)
    except Exception as hook_err:
        logger.debug("transform_spoken_text hook error: %s", hook_err, exc_info=True)
        return text
