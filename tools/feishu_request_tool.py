"""Read-only Feishu Open API gateway with fail-closed path validation.

Exposes a single ``feishu_request`` tool for ad-hoc Feishu/Lark reads. The
path is checked against a curated GET-only endpoint map *before* dispatch;
unknown paths are rejected with a "did you mean" hint instead of 404ing at
Feishu (which the agent often misreads as a permissions problem).

Why this exists: agents without a folder-listing tool invent paths like
``GET /open-apis/drive/v1/files/{folder_token}/children`` (does not exist).
The correct call is ``GET /open-apis/drive/v1/files?folder_token=xxx``.

Scope: GET only. This tool is wired into ``feishu_drive`` (comment-agent
path with lark client injection). Writes stay on typed tools and the
handler's controlled ``deliver_comment_reply`` path — not this gateway.

Why not write+approval here?
    Hermes approval is not a generic "Open API DELETE" gate. Existing
    surfaces are: shell dangerous-command patterns (``tools.approval`` +
    platform ``send_exec_approval`` on Feishu/WhatsApp/Teams/…), plugin
    ``pre_tool_call`` → ``request_tool_approval``, ACP edit approval for
    ``write_file``/``patch``, and optional ``write_approval`` for memory/
    skills. Platform tools such as Discord ``delete_message`` do not go
    through that gate. Comment-agent is output-only, so admitting writes
    (even after a shell-style approval click) would still bypass
    ``deliver_comment_reply``. Fail-closed GET-only is the matching
    control for this toolset placement.

Validation is pure Python (no SDK import). Dispatch reuses
``feishu_drive_tool._do_request``.
"""

import difflib
import logging
import threading

from tools.feishu_drive_tool import _do_request
from tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)

# Thread-local lark client, injected per-request by the feishu_comment handler.
_local = threading.local()


def set_client(client):
    """Store a lark client for the current thread (called by feishu_comment)."""
    _local.client = client


def get_client():
    """Return the lark client for the current thread, or None."""
    return getattr(_local, "client", None)


def _check_feishu():
    # Mirror feishu_doc_tool._check_feishu -- find_spec keeps CLI startup fast.
    import importlib.util
    try:
        return importlib.util.find_spec("lark_oapi") is not None
    except (ImportError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Read-only navigation map: (METHOD, path_template).
#
# ``:segment`` matches one concrete path segment (token / id).
# Fail-closed: unlisted paths never leave the process. GET only — no
# POST/PUT/DELETE (comment writes, Bitable mutations, create_folder, etc.).
# Templates match official Feishu HTTP URLs verbatim (open.feishu.cn).
# ---------------------------------------------------------------------------
_FEISHU_ENDPOINTS = [
    # Drive — list is ?folder_token=, NOT /:token/children
    # https://open.feishu.cn/document/server-docs/docs/drive-v1/folder/list
    ("GET", "/open-apis/drive/v1/files"),
    # https://open.feishu.cn/document/uAjLw4CM/ukTMukTMukTM/reference/drive-v1/file-statistics/get
    ("GET", "/open-apis/drive/v1/files/:file_token/statistics"),
    # https://open.feishu.cn/document/server-docs/docs/drive-v1/folder/get-root-folder-meta
    ("GET", "/open-apis/drive/explorer/v2/root_folder/meta"),
    # https://open.feishu.cn/document/uAjLw4CM/ukTMukTMukTM/reference/drive-v1/file-comment/list
    ("GET", "/open-apis/drive/v1/files/:file_token/comments"),
    # https://open.feishu.cn/document/uAjLw4CM/ukTMukTMukTM/reference/drive-v1/file-comment-reply/list
    ("GET", "/open-apis/drive/v1/files/:file_token/comments/:comment_id/replies"),
    # Docx
    # https://open.feishu.cn/document/server-docs/docs/docs/docx-v1/document/get
    ("GET", "/open-apis/docx/v1/documents/:document_id"),
    # https://open.feishu.cn/document/server-docs/docs/docs/docx-v1/document/raw_content
    ("GET", "/open-apis/docx/v1/documents/:document_id/raw_content"),
    # https://open.feishu.cn/document/ukTMukTMukTM/uUDN04SN0QjL1QDN/document-docx/docx-v1/document-block/list
    ("GET", "/open-apis/docx/v1/documents/:document_id/blocks"),
    # Bitable (reads only)
    # https://open.feishu.cn/document/server-docs/docs/bitable-v1/app/get
    ("GET", "/open-apis/bitable/v1/apps/:app_token"),
    # https://open.feishu.cn/document/server-docs/docs/bitable-v1/app-table/list
    ("GET", "/open-apis/bitable/v1/apps/:app_token/tables"),
    # https://open.feishu.cn/document/server-docs/docs/bitable-v1/app-table-field/list
    ("GET", "/open-apis/bitable/v1/apps/:app_token/tables/:table_id/fields"),
    # https://open.feishu.cn/document/server-docs/docs/bitable-v1/app-table-record/list
    ("GET", "/open-apis/bitable/v1/apps/:app_token/tables/:table_id/records"),
    # Sheets
    # https://open.feishu.cn/document/server-docs/docs/sheets-v3/spreadsheet/get
    ("GET", "/open-apis/sheets/v3/spreadsheets/:spreadsheet_token"),
    # https://open.feishu.cn/document/server-docs/docs/sheets-v3/spreadsheet-sheet/query
    ("GET", "/open-apis/sheets/v3/spreadsheets/:spreadsheet_token/sheets/query"),
    # Wiki
    # https://open.feishu.cn/document/server-docs/docs/wiki-v2/space/list
    ("GET", "/open-apis/wiki/v2/spaces"),
    # https://open.feishu.cn/document/server-docs/docs/wiki-v2/space-node/list
    ("GET", "/open-apis/wiki/v2/spaces/:space_id/nodes"),
]

_ALLOWED_METHODS = {"GET"}


def _normalize_path(path: str) -> str:
    """Reduce any caller-supplied form to a bare ``/open-apis/...`` path.

    Agents often paste full URLs (scheme+host). Slice from ``/open-apis/`` so
    the host is not treated as path segments.
    """
    path = (path or "").strip()
    idx = path.find("/open-apis/")
    return path[idx:] if idx > 0 else path


def _split(path: str) -> list:
    """Strip query string and slashes, return path segments."""
    path = _normalize_path(path).split("?", 1)[0].strip("/")
    return path.split("/") if path else []


def _looks_like_token(seg: str) -> bool:
    """Heuristic: is this segment a concrete id/token (vs a static path word)?"""
    return len(seg) >= 16 or (len(seg) >= 10 and any(c.isdigit() for c in seg))


def validate_endpoint(method: str, path: str):
    """Check ``method``+``path`` against the navigation map.

    Returns ``(template, paths)`` when valid -- ``template`` is the canonical
    URI for the lark BaseRequest and ``paths`` maps each ``:param`` to the
    concrete value pulled from *path*.  Returns ``(None, suggestions)`` when
    invalid, where ``suggestions`` is a list of the closest valid templates.
    """
    method = str(method).upper()
    segments = _split(path)

    # Prefer the most specific structural match (fewest wildcards) so a
    # literal segment like ``raw_content`` or ``query`` is not captured as
    # a ``:token`` by a looser template.
    matches = []
    for m, template in _FEISHU_ENDPOINTS:
        if m != method:
            continue
        t_segs = template.strip("/").split("/")
        if len(t_segs) != len(segments):
            continue
        paths, wildcards, ok = {}, 0, True
        for tseg, seg in zip(t_segs, segments):
            if tseg.startswith(":"):
                paths[tseg[1:]] = seg
                wildcards += 1
            elif tseg != seg:
                ok = False
                break
        if ok:
            matches.append((wildcards, template, paths))
    if matches:
        matches.sort(key=lambda x: x[0])
        return matches[0][1], matches[0][2]

    # No match -> "did you mean" suggestions. Normalize concrete tokens to
    # ``:x`` so fuzzy matching compares shapes, not tokens.
    norm = "/" + "/".join(":x" if _looks_like_token(s) else s for s in segments)
    candidates = [t for (m, t) in _FEISHU_ENDPOINTS if m == method]
    if not candidates:
        candidates = [t for (_, t) in _FEISHU_ENDPOINTS]
    norm_candidates = {
        "/" + "/".join(":x" if s.startswith(":") else s for s in _split(t)): t
        for t in candidates
    }
    close = difflib.get_close_matches(norm, list(norm_candidates), n=3, cutoff=0.5)
    suggestions = [norm_candidates[c] for c in close]
    if not suggestions:
        # Fall back to longest shared path-prefix.
        def _prefix_len(t):
            ts = _split(t)
            n = 0
            for a, b in zip(ts, segments):
                if a.startswith(":") or a == b:
                    n += 1
                else:
                    break
            return n
        suggestions = sorted(candidates, key=_prefix_len, reverse=True)[:3]
    return None, suggestions


# ---------------------------------------------------------------------------
# feishu_request
# ---------------------------------------------------------------------------

FEISHU_REQUEST_SCHEMA = {
    "name": "feishu_request",
    "description": (
        "Call a validated Feishu/Lark Open API endpoint (GET / read-only). "
        "The path is checked against a known-good map BEFORE the call — an "
        "unknown path is rejected with the correct one suggested, instead of "
        "404ing. Use for Drive/Docx/Bitable/Sheets/Wiki reads. "
        "Writes are not supported here. "
        "List items in a folder: GET /open-apis/drive/v1/files with query "
        "folder_token=xxx (there is NO /files/{token}/children path)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "method": {
                "type": "string",
                "description": "HTTP method (GET only — this tool is read-only).",
                "enum": ["GET"],
            },
            "path": {
                "type": "string",
                "description": (
                    "API path starting with /open-apis/, with concrete tokens "
                    "inline, e.g. /open-apis/drive/v1/files or "
                    "/open-apis/docx/v1/documents/DOC_TOKEN/raw_content."
                ),
            },
            "query": {
                "type": "object",
                "description": "Optional query parameters as a flat object.",
            },
        },
        "required": ["method", "path"],
    },
}


def _handle_feishu_request(args: dict, **kwargs) -> str:
    method = str(args.get("method", "")).strip().upper()
    path = str(args.get("path", "")).strip()
    query = args.get("query") or {}

    if not method or not path:
        return tool_error("method and path are required")
    if method not in _ALLOWED_METHODS:
        return tool_error(
            f"Unsupported method '{method}'. feishu_request is read-only "
            f"(GET only). Writes stay on typed Feishu tools / delivery path."
        )

    template, result = validate_endpoint(method, path)
    if template is None:
        logger.warning(
            "[Feishu-Request] REJECTED %s %s — not in endpoint map; suggesting %s",
            method, path, result,
        )
        return tool_error(
            f"Invalid Feishu endpoint: {method} {path}. This path is not in the "
            f"validated endpoint map and would 404. Did you mean: {result}? "
            f"(Reminder: list items in a folder is "
            f"GET /open-apis/drive/v1/files?folder_token=xxx -- query param, "
            f"NOT /files/{{token}}/children.)",
            valid_suggestions=result,
        )

    client = get_client()
    if client is None:
        return tool_error("Feishu client not available (not in a Feishu context)")

    # Expand list/tuple values into repeated key=value pairs (some Feishu
    # endpoints take repeated query params); scalars stringify directly.
    queries = []
    for k, v in query.items():
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            queries.extend((k, str(x)) for x in v)
        else:
            queries.append((k, str(v)))

    logger.info("[Feishu-Request] %s %s paths=%s", method, template, result or {})
    code, msg, data = _do_request(
        client, method, template,
        paths=result or None,
        queries=queries or None,
    )
    if code != 0:
        logger.warning("[Feishu-Request] %s %s failed: code=%s msg=%s",
                       method, template, code, msg)
        return tool_error(f"Feishu request failed: code={code} msg={msg}")

    return tool_result({"success": True, "data": data})


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

registry.register(
    name="feishu_request",
    toolset="feishu_drive",
    schema=FEISHU_REQUEST_SCHEMA,
    handler=_handle_feishu_request,
    check_fn=_check_feishu,
    requires_env=[],
    is_async=False,
    description="Call a validated Feishu Open API endpoint (GET / read-only)",
    emoji="\U0001f5fa",
)
