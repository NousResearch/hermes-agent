"""Send File Tool -- transfer files from sandbox to user.

Extracts a file from the active execution environment (local, Docker, SSH,
Modal, Singularity, Daytona) and delivers it to the user. In CLI mode the
file is copied to the current working directory. In gateway mode a FILE: tag
is emitted so the platform adapter can send it as a native document.
"""

import json
import logging
import os
import shutil

logger = logging.getLogger(__name__)


SEND_FILE_SCHEMA = {
    "name": "send_file",
    "description": (
        "Send a file from the terminal environment to the user. "
        "The file is extracted from the sandbox (Docker, SSH, Modal, etc.) "
        "and delivered as a downloadable document. Use this after creating "
        "files like reports, charts, exports, or any artifacts the user needs."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute path to the file inside the terminal environment",
            },
            "message": {
                "type": "string",
                "description": "Optional caption or description for the file",
            },
        },
        "required": ["path"],
    },
}


def send_file_tool(args, **kw):
    """Handle send_file tool calls."""
    path = args.get("path", "")
    message = args.get("message", "")
    task_id = kw.get("task_id", "default")

    if not path:
        return json.dumps({"error": "Missing required parameter: path"})

    from tools.file_transfer import extract_file_from_sandbox, validate_file_path

    # Validate path
    error = validate_file_path(path)
    if error:
        return json.dumps({"error": error})

    # Extract from sandbox
    result = extract_file_from_sandbox(path, task_id)
    if not result.get("success"):
        return json.dumps({"error": result.get("error", "Extraction failed")})

    host_path = result["host_path"]
    filename = result["filename"]
    mime_type = result["mime_type"]
    size = result["size"]

    # Check if we're running in gateway mode or CLI mode
    is_gateway = os.getenv("HERMES_GATEWAY", "").lower() in ("1", "true", "yes")

    if is_gateway:
        # Gateway mode: emit FILE: tag for the platform adapter to pick up
        caption = message if message else f"Here's the file: {filename}"
        response = {
            "success": True,
            "filename": filename,
            "mime_type": mime_type,
            "size": size,
            "delivered_via": "gateway",
        }
        # The FILE: tag will be extracted by BasePlatformAdapter.extract_files()
        # Use angle-bracket delimiters to support paths with spaces.
        # Pipe separator carries an optional caption for the platform adapter.
        # Escape \ and > in both path and caption so the FILE tag parser
        # can distinguish content from the closing delimiter.
        safe_path = host_path.replace("\\", "\\\\").replace(">", "\\>")
        safe_caption = caption.replace("\\", "\\\\").replace(">", "\\>")
        return json.dumps(response) + f"\nFILE:<{safe_path}|{safe_caption}>"
    else:
        # CLI mode: copy to current working directory
        cwd = os.getcwd()
        dest = os.path.join(cwd, filename)

        # Avoid overwriting: add suffix if file exists
        if os.path.exists(dest):
            base, ext = os.path.splitext(filename)
            counter = 1
            while os.path.exists(dest):
                dest = os.path.join(cwd, f"{base}_{counter}{ext}")
                counter += 1

        shutil.copy2(host_path, dest)
        return json.dumps({
            "success": True,
            "filename": os.path.basename(dest),
            "path": dest,
            "mime_type": mime_type,
            "size": size,
            "delivered_via": "cli",
            "message": message or f"File saved to {dest}",
        })


# --- Registry ---
from tools.registry import registry

registry.register(
    name="send_file",
    toolset="file_transfer",
    schema=SEND_FILE_SCHEMA,
    handler=lambda args, **kw: send_file_tool(args, **kw),
)
