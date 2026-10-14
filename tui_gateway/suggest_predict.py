"""Model-predicted next user message for the composer ghost (opt-in).

Inspired by ChatGPT/Codex "Composer predictions" (OpenAI release notes, Oct 9 2026): after the
agent replies, a suggested next message appears in the empty composer; Tab puts it in the input
for review and it is never sent automatically. Hermes runs it on the ``composer_predictions``
auxiliary task so it can be pointed at a cheap model, and only when the free deterministic
extractor in ``suggest.py`` found nothing. It is off by default (``display.composer_predictions``)
because every completed turn would otherwise bill an extra request.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

TASK = "composer_predictions"
MAX_PREDICTION_CHARS = 200
_CONTEXT_MESSAGES = 6
_PER_MESSAGE_CHARS = 1500
_NONE = "NONE"
# Models echo the transcript's role labels back ("USER: ...").
_ROLE_PREFIX_RE = re.compile(r"^(?:user|assistant)\s*:\s*", re.IGNORECASE)

_PROMPT = (
    "You predict the next message a user will type to their AI agent. Read the recent "
    "conversation and write the single message the user is most likely to send next, in their "
    "own voice, language and register: a short reply, follow-up request or next step, at most "
    "25 words. Output only the message text: no role label, no quotes, no explanation, and never "
    f"the assistant's answer. If no next message is clearly likely, output exactly {_NONE}."
)


def _transcript(history: list[dict], coerce) -> list[str]:
    turns: list[str] = []
    for msg in history:
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue
        text = (coerce(msg.get("content")) or "").strip()
        if text:
            turns.append(f"{role.upper()}: {text[-_PER_MESSAGE_CHARS:]}")
    return turns[-_CONTEXT_MESSAGES:]


def predict_next_message(history: list[dict], coerce, *, main_runtime: dict | None = None) -> str:
    """One prediction, or ``""`` when the model abstains, fails, or the transcript has no reply yet."""
    turns = _transcript(history, coerce)
    # Predict only right after an agent reply; a trailing user turn means the reply never landed.
    if not turns or not turns[-1].startswith("ASSISTANT:"):
        return ""
    from agent.auxiliary_client import call_llm, extract_content_or_reasoning

    try:
        response: Any = call_llm(
            task=TASK,
            messages=[{"role": "system", "content": _PROMPT}, {"role": "user", "content": "\n\n".join(turns)}],
            max_tokens=80, temperature=None, main_runtime=main_runtime,
            reasoning_config={"enabled": False},
        )
    except Exception:  # health: allow BLE001 -- decorative: any provider/transport failure means "no ghost"
        logger.debug("composer prediction failed", exc_info=True)
        return ""
    text = " ".join((extract_content_or_reasoning(response) or "").split()).strip().strip("\"'")
    text = _ROLE_PREFIX_RE.sub("", text).strip("\"'")
    if not text or text.upper().rstrip(".") == _NONE:
        return ""
    return text[:MAX_PREDICTION_CHARS]
