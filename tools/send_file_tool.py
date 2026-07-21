"""send_file tool — deliver a file from the terminal environment to the user.

Thin explicit wrapper over the MEDIA: delivery pipeline (#466). The MEDIA:
directive already just works — including transparent fetch from remote
terminal backends (see gateway/media_fetch.py) — but it is fire-and-forget:
the agent never learns whether the file was extractable or under the size
cap. This tool runs the same validation/fetch eagerly and reports the
outcome, returning a ``media_tag`` for the agent to include in its reply
(the same contract as text_to_speech).
"""

import json
import logging
import os

logger = logging.getLogger(__name__)


def send_file_tool(path: str, message: str = None, task_id: str = None) -> str:
    """Validate/fetch *path* for platform delivery and return a MEDIA tag."""
    path = (path or "").strip()
    if not path:
        return tool_error("send_file requires a non-empty 'path'")

    backend = (os.getenv("TERMINAL_ENV") or "local").strip().lower()
    from agent.prompt_builder import _REMOTE_TERMINAL_BACKENDS

    if backend in _REMOTE_TERMINAL_BACKENDS:
        from gateway.media_fetch import fetch_remote_media

        local_path, reason = fetch_remote_media(path, task_id=task_id)
        if not local_path:
            return tool_error(
                f"Could not deliver {os.path.basename(path)!r} from the "
                f"{backend} backend: {reason}"
            )
    else:
        from gateway.platforms.base import validate_media_delivery_path

        local_path = validate_media_delivery_path(path)
        if not local_path:
            if not os.path.isfile(os.path.expanduser(path)):
                return tool_error(
                    f"File not found on the host: {path!r}. Pass an absolute "
                    f"path to an existing file."
                )
            return tool_error(
                f"The path {path!r} is not allowed for delivery "
                f"(credential/system location)."
            )

    media_tag = f"MEDIA:{local_path}"
    if message:
        media_tag = f"{message}\n{media_tag}"
    return json.dumps({
        "success": True,
        "file_path": local_path,
        "media_tag": media_tag,
        "note": (
            "File is staged for delivery. Include the media_tag line "
            "verbatim in your reply so the platform attaches it."
        ),
    }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
from tools.registry import registry, tool_error

SEND_FILE_SCHEMA = {
    "name": "send_file",
    "description": (
        "Send a file from the terminal environment to the user as a native "
        "attachment. Works on every terminal backend: when the terminal runs "
        "in a remote sandbox (ssh, docker, modal, daytona, singularity) the "
        "file is fetched out of the sandbox first. Returns a media_tag to "
        "include verbatim in your reply, or a clear error when the file is "
        "missing, too large, or in a protected location. Plain "
        "MEDIA:/absolute/path directives do the same implicitly — use this "
        "tool when you want confirmation that the file is deliverable."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Absolute path of the file inside the terminal "
                    "environment (the sandbox filesystem when the backend "
                    "is remote)."
                ),
            },
            "message": {
                "type": "string",
                "description": "Optional caption to send along with the file.",
            },
        },
        "required": ["path"],
    },
}

registry.register(
    name="send_file",
    toolset="file",
    schema=SEND_FILE_SCHEMA,
    handler=lambda args, **kw: send_file_tool(
        path=args.get("path", ""),
        message=args.get("message"),
        task_id=kw.get("task_id")),
    emoji="📎",
)
