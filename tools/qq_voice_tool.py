"""Send synthesized speech as a native QQ voice message via OneBot.

Registers one LLM-callable tool, ``qq_send_voice``, which turns text into
speech with Hermes's existing ``text_to_speech`` backend and delivers it as
a QQ voice message. It can also send an already-recorded local audio file.

Delivery goes through a running OneBot v11 implementation using the
``send_msg`` action with a ``record`` message segment. The audio is embedded
inline as a ``base64://`` payload, so Hermes and the OneBot client do not
need to share a filesystem.
"""

import base64
import json
import logging
import os

from tools.onebot_client import onebot_call, onebot_configured
from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

_AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".amr", ".silk", ".m4a", ".flac", ".aac"}
_MAX_AUDIO_BYTES = 30 * 1024 * 1024


def _coerce_qq_id(value, field: str) -> int:
    """Coerce a QQ user/group id to a positive integer."""
    try:
        qid = int(str(value).strip())
    except (TypeError, ValueError):
        raise ValueError(f"'{field}' must be a numeric QQ id, got {value!r}") from None
    if qid <= 0:
        raise ValueError(f"'{field}' must be a positive QQ id, got {qid}")
    return qid


def _build_record_message(audio_b64: str) -> list:
    """Build the OneBot v11 message array for a voice record segment."""
    return [{"type": "record", "data": {"file": f"base64://{audio_b64}"}}]


def _build_send_params(message: list, user_id, group_id) -> dict:
    """Build OneBot ``send_msg`` params for a private or group target."""
    if group_id is not None:
        return {
            "message_type": "group",
            "group_id": int(group_id),
            "message": message,
        }
    return {
        "message_type": "private",
        "user_id": int(user_id),
        "message": message,
    }


def _read_audio_file(path: str) -> bytes:
    """Read a local audio file, raising ``ValueError`` for invalid input."""
    resolved = os.path.expanduser(str(path))
    if not os.path.isfile(resolved):
        raise ValueError("file not found")
    ext = os.path.splitext(resolved)[1].lower()
    if ext not in _AUDIO_EXTS:
        raise ValueError(
            f"unsupported audio type '{ext}' (allowed: {sorted(_AUDIO_EXTS)})"
        )
    size = os.path.getsize(resolved)
    if size == 0:
        raise ValueError("file is empty")
    if size > _MAX_AUDIO_BYTES:
        raise ValueError(f"audio too large ({size} bytes; max {_MAX_AUDIO_BYTES})")
    with open(resolved, "rb") as fh:
        return fh.read()


def _synthesize_speech(text: str) -> str:
    """Synthesize text to an audio file via Hermes's configured TTS backend."""
    from tools.tts_tool import text_to_speech_tool  # noqa: PLC0415

    raw = text_to_speech_tool(text=text)
    try:
        result = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as e:
        raise RuntimeError(
            f"text_to_speech returned an unparseable result: {str(raw)[:200]}"
        ) from e

    if not result.get("success"):
        raise RuntimeError(result.get("error") or "speech synthesis failed")
    path = result.get("file_path")
    if not path or not os.path.isfile(path):
        raise RuntimeError("text_to_speech reported success but produced no file")
    return path


def _handle_qq_send_voice(args: dict, **kw) -> str:
    """Handler for the qq_send_voice tool."""
    text = (args.get("text") or "").strip()
    audio_file = (args.get("audio_file") or "").strip()

    if not text and not audio_file:
        return tool_error(
            "qq_send_voice requires 'text' (to synthesize) or 'audio_file'."
        )
    if text and audio_file:
        return tool_error(
            "qq_send_voice: provide either 'text' or 'audio_file', not both."
        )

    raw_user = args.get("user_id")
    raw_group = args.get("group_id")
    has_user = raw_user not in (None, "")
    has_group = raw_group not in (None, "")
    if has_user == has_group:
        return tool_error(
            "qq_send_voice requires exactly one target: 'user_id' (private "
            "chat) or 'group_id' (group chat)."
        )

    try:
        if has_group:
            group_id: int | None = _coerce_qq_id(raw_group, "group_id")
            user_id: int | None = None
        else:
            user_id = _coerce_qq_id(raw_user, "user_id")
            group_id = None
    except ValueError as e:
        return tool_error(f"qq_send_voice: {e}")

    if audio_file:
        try:
            audio_bytes = _read_audio_file(audio_file)
        except ValueError as e:
            return tool_error(f"Audio file '{audio_file}': {e}")
        synthesized = False
    else:
        try:
            audio_path = _synthesize_speech(text)
        except Exception as e:  # noqa: BLE001
            logger.error("qq_send_voice: speech synthesis failed: %s", e)
            return tool_error(f"Speech synthesis failed: {e}")
        try:
            audio_bytes = _read_audio_file(audio_path)
        except ValueError as e:
            return tool_error(f"Synthesized audio unusable: {e}")
        synthesized = True

    audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
    params = _build_send_params(
        _build_record_message(audio_b64),
        user_id,
        group_id,
    )

    try:
        data = onebot_call("send_msg", params)
    except Exception as e:  # noqa: BLE001
        logger.error("qq_send_voice: OneBot send_msg failed: %s", e)
        return tool_error(f"Could not send the voice message via OneBot: {e}")

    target = (
        {"type": "group", "id": group_id}
        if group_id is not None
        else {"type": "private", "id": user_id}
    )
    return json.dumps(
        {
            "success": True,
            "message_id": data.get("message_id"),
            "target": target,
            "synthesized": synthesized,
            "audio_bytes": len(audio_bytes),
            "message": "Voice message sent to QQ.",
        },
        ensure_ascii=False,
    )


QQ_SEND_VOICE_SCHEMA = {
    "name": "qq_send_voice",
    "description": (
        "Send a native QQ voice message to a QQ user or group. Give 'text' "
        "to synthesize speech with the configured TTS provider/voice, or "
        "'audio_file' to send an existing local audio file. Delivered via a "
        "running OneBot (NapCat / Lagrange) instance. Exactly one of "
        "'user_id' or 'group_id' is required."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": (
                    "Text to speak. Provide this or 'audio_file', not both."
                ),
            },
            "audio_file": {
                "type": "string",
                "description": (
                    "Path to an existing local audio file: mp3, wav, ogg, "
                    "amr, silk, m4a, flac, or aac. Provide this or 'text', "
                    "not both."
                ),
            },
            "user_id": {
                "type": "string",
                "description": (
                    "Target QQ number for a private-chat voice message. "
                    "Provide either 'user_id' or 'group_id', not both."
                ),
            },
            "group_id": {
                "type": "string",
                "description": (
                    "Target QQ group number for a group-chat voice message. "
                    "Provide either 'user_id' or 'group_id', not both."
                ),
            },
        },
        "required": [],
    },
}


registry.register(
    name="qq_send_voice",
    toolset="qq_voice",
    schema=QQ_SEND_VOICE_SCHEMA,
    handler=_handle_qq_send_voice,
    check_fn=onebot_configured,
    requires_env=[],
)
