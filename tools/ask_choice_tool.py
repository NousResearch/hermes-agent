"""On-screen choice dialog: ask the user to pick one of a few short options in a
small, always-on-top window on their desktop — instead of in the chat — and
block until they answer (or the request times out).

The agent calls ``ask_choice``; this handler writes a small request file next
to the other Hermes state (HERMES_HOME root, ``%LOCALAPPDATA%\\hermes`` on
Windows) and polls for a matching answer file. The desktop app's
``ask-choice`` window polls the request, renders a compact card with one button
per option (centered on the primary display), and writes the answer back when
the user clicks. This is the same file-poll bridge the scanline sweep uses for
its scope file — no new gateway IPC, and it works from any backend the desktop
app owns.

The dialog is a *separate* surface from the chat on purpose: a short decision
question shouldn't live in the long conversation window. Callers should SKIP
this tool entirely when the user has already named their choice in prose (e.g.
"analyze the right screen") — asking then would be noise.
"""

import json
import os
import time
import uuid
from typing import Optional

from hermes_constants import get_hermes_home
from tools.registry import registry, tool_error

# Request/response filenames, written directly in the HERMES_HOME root (the
# desktop main process resolves the same directory via resolveHermesHome()).
REQUEST_FILENAME = "ask-choice-request.json"
ANSWER_FILENAME = "ask-choice-answer.json"

# Poll cadence for the answer file — tight enough to feel instant, cheap enough
# to ignore.
ANSWER_POLL_S = 0.1
# How long to wait for the user before giving up. On timeout the dialog closes
# and this returns status="expired" so the caller can say so in chat.
DEFAULT_TIMEOUT_S = 90.0
MAX_TIMEOUT_S = 300.0
MIN_TIMEOUT_S = 5.0
MAX_OPTIONS = 4
MAX_OPTION_CHARS = 64
MAX_QUESTION_CHARS = 240


def _request_path() -> str:
    return os.path.join(get_hermes_home(), REQUEST_FILENAME)


def _answer_path() -> str:
    return os.path.join(get_hermes_home(), ANSWER_FILENAME)


def _write_json_atomic(path: str, payload: dict) -> None:
    """Write JSON via a temp file + replace so the desktop poller never reads a
    partial write (a half-written file would just parse-fail and be ignored,
    but atomicity keeps the round-trip clean)."""
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _read_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _cleanup_request() -> None:
    try:
        os.remove(_request_path())
    except OSError:
        pass


def _cleanup_answer() -> None:
    # Drop our answer file once the round-trip is done. Any file here is ours to
    # clear (the desktop only writes one, keyed to the current request id), so a
    # stale leftover from a prior ask is litter worth removing too.
    try:
        os.remove(_answer_path())
    except OSError:
        pass


def _read_answer(request_id: str):
    """Return the outcome for this request, or None if not (yet) answered.
    An answer that doesn't name this request's id is stale from a prior ask and
    is ignored. Returns ``{"answered": <choice>}`` on a click or
    ``{"cancelled": True}`` when the user dismissed the dialog (Esc)."""
    data = _read_json(_answer_path())
    if not isinstance(data, dict):
        return None
    if data.get("request_id") != request_id:
        return None
    if data.get("cancelled"):
        return {"cancelled": True}
    return {"answered": data.get("choice")}


def ask_choice(question: str = "", options: Optional[list] = None,
               timeout_seconds: Optional[float] = None, **_kwargs) -> str:
    """Show a small always-on-top dialog on the user's desktop asking them to
    pick one of a few short options, and wait for their answer."""
    question = (question or "").strip()
    if not question:
        return tool_error("ask_choice needs a non-empty 'question'.")
    question = question[:MAX_QUESTION_CHARS]

    raw_options = options or []
    if not isinstance(raw_options, list) or not raw_options:
        return tool_error("ask_choice needs a non-empty 'options' array (2–4 strings).")
    if len(raw_options) > MAX_OPTIONS:
        return tool_error(f"ask_choice accepts at most {MAX_OPTIONS} options.")

    cleaned = []
    for opt in raw_options:
        text = str(opt).strip()
        if not text:
            return tool_error("ask_choice options must be non-empty strings.")
        cleaned.append(text[:MAX_OPTION_CHARS])
    if len(cleaned) < 2:
        return tool_error("ask_choice needs at least 2 options.")

    try:
        timeout = float(timeout_seconds) if timeout_seconds is not None else DEFAULT_TIMEOUT_S
    except (TypeError, ValueError):
        timeout = DEFAULT_TIMEOUT_S
    timeout = max(MIN_TIMEOUT_S, min(MAX_TIMEOUT_S, timeout))

    request_id = uuid.uuid4().hex
    started = time.monotonic()
    _write_json_atomic(_request_path(), {
        "request_id": request_id,
        "question": question,
        "options": cleaned,
        "ts": int(time.time() * 1000),
    })

    try:
        while True:
            outcome = _read_answer(request_id)
            if outcome is not None:
                if outcome.get("cancelled"):
                    return json.dumps({
                        "status": "cancelled",
                        "choice": None,
                        "question": question,
                        "options": cleaned,
                    }, ensure_ascii=False)
                return json.dumps({
                    "status": "answered",
                    "choice": outcome.get("answered"),
                    "question": question,
                    "options": cleaned,
                }, ensure_ascii=False)
            if time.monotonic() - started >= timeout:
                return json.dumps({
                    "status": "expired",
                    "choice": None,
                    "question": question,
                    "options": cleaned,
                }, ensure_ascii=False)
            time.sleep(ANSWER_POLL_S)
    finally:
        # Drop both files so the dialog (which watches the request file) fades
        # out and the next ask starts clean.
        _cleanup_request()
        _cleanup_answer()


ASK_CHOICE_SCHEMA = {
    "name": "ask_choice",
    "description": (
        "Ask the user to pick one of a few short options in a small, always-on-top "
        "dialog window on their desktop (NOT in this chat), and wait for their "
        "answer. Use for short decisions where a separate little on-screen prompt "
        "is better than a message in the conversation (e.g. which screen to "
        "analyze). SKIP this entirely if the user already named their choice in "
        "their message — don't ask what they just told you. Pass a short "
        "'question' (one line) and 2–4 short 'options' (button labels). Blocks "
        "until they click (or 'expired' after the timeout, default 90s)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "Short one-line question shown above the options.",
            },
            "options": {
                "type": "array",
                "items": {"type": "string"},
                "description": "2–4 short option labels, each a button in the dialog.",
            },
            "timeout_seconds": {
                "type": "number",
                "description": "Optional seconds to wait (default 90). 5–300.",
            },
        },
        "required": ["question", "options"],
    },
}


registry.register(
    name="ask_choice",
    toolset="desktop_ui",
    schema=ASK_CHOICE_SCHEMA,
    handler=lambda args, **kw: ask_choice(
        question=args.get("question", ""),
        options=args.get("options"),
        timeout_seconds=args.get("timeout_seconds"),
        **kw,
    ),
    emoji="🔘",
)
