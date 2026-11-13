"""Plugin seams for text Hermes speaks: ``transform_spoken_text`` and ``transform_spoken_display_text``.

Every speech path calls :func:`apply_spoken_text_hook` after its TTS normalizer and before
synthesis. The gateway calls :func:`apply_spoken_display_text_hook` on the chat text / caption it
sends alongside a reply it also spoke. Both hooks: the first ``str`` result wins, empty string
included (nothing is spoken / no text is sent); ``None`` and any other value leave the text
unchanged. Fail-open: a raising, hung or broken plugin never suppresses the reply.
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
    return _first_str_result(
        "transform_spoken_text", text, surface=surface, platform=platform or "",
        session_id=session_id or "", index=index)


def apply_spoken_display_text_hook(
    text: str, *, surface: str, platform: str = "", session_id: str = "",
) -> str:
    """Return the chat text / caption to send with a reply Hermes also spoke on ``surface``
    (``voice_reply`` or ``auto_tts``). Display only: the spoken text is not affected."""
    return _first_str_result(
        "transform_spoken_display_text", text, surface=surface, platform=platform or "",
        session_id=session_id or "", index=0)


def _first_str_result(hook_name: str, text: str, **kwargs) -> str:
    try:
        from hermes_cli.lifecycle import has_hook, invoke_hook
        if not text or not has_hook(hook_name):
            return text
        results = invoke_hook(hook_name, text=text, **kwargs)
        return next((r for r in results if isinstance(r, str)), text)
    except Exception as hook_err:
        logger.debug("%s hook error: %s", hook_name, hook_err, exc_info=True)
        return text
