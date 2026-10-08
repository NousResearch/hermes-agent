"""Publish static HTML through the calling conversation's normal tool result."""
import hashlib
import json

from tools.registry import registry, tool_error

MAX_HTML_BYTES = 64 * 1024


def publish_html(args: dict, **kwargs) -> str:
    """No caller-supplied session selector or filesystem access. Persistence belongs to the executor."""
    if set(args) - {"title", "html", "fallback"}:
        return tool_error("Only title, html and fallback are accepted; publication belongs to this conversation.")
    session_id = kwargs.get("session_id") or kwargs.get("task_id")
    if not session_id:
        return tool_error("Inline HTML publication requires a bound conversation.")
    title, html, fallback = (args.get(key) for key in ("title", "html", "fallback"))
    if not isinstance(title, str) or not title.strip() or len(title) > 200:
        return tool_error("title must be a non-empty string of at most 200 characters.")
    if not isinstance(fallback, str) or not fallback.strip() or len(fallback) > 4000:
        return tool_error("fallback must be non-empty plain text of at most 4000 characters.")
    try:
        if not isinstance(html, str) or not html.strip() or len(html.encode("utf-8")) > MAX_HTML_BYTES:
            return tool_error("html must be non-empty UTF-8 HTML of at most 65536 bytes.")
        identity = json.dumps([str(session_id), title, html, fallback], ensure_ascii=False).encode("utf-8")
    except UnicodeEncodeError:
        return tool_error("Artifact strings must be valid UTF-8.")
    artifact = {"version": 1, "id": "html_" + hashlib.sha256(identity).hexdigest(),
                "title": title, "format": "html", "html": html, "fallback": fallback}
    return json.dumps({"artifact": artifact, "text": fallback}, ensure_ascii=False)


registry.register(
    name="publish_html", toolset="artifacts", handler=publish_html,
    schema={"name": "publish_html", "description": (
        "Publish static HTML inline in this conversation. Scripts, network access, forms and privileged "
        "APIs are disabled. Supply plain-text fallback for clients without artifact support. Publication "
        "is saved with this tool result; repeat identical content to retry without changing its ID."),
        "parameters": {"type": "object", "properties": {
            "title": {"type": "string", "maxLength": 200},
            "html": {"type": "string", "description": "Static HTML, maximum 65536 UTF-8 bytes."},
            "fallback": {"type": "string", "maxLength": 4000}},
            "required": ["title", "html", "fallback"], "additionalProperties": False}},
)
