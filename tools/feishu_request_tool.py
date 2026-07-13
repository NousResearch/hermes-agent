"""Feishu generic request tool -- one validated entry point for the whole API.

Rather than shipping one typed tool per Feishu endpoint (there are hundreds),
this exposes a single ``feishu_request`` tool whose ``path`` is checked against a
curated "navigation map" of known-good endpoint templates *before* the call is
made.  An unknown path is rejected up front with a "did you mean ..." hint,
instead of being sent to Feishu and bouncing back as a confusing 404.

Why this exists: the agent has no folder-listing tool, so it used to fall back
to the raw code sandbox and hand-craft URLs like
``GET /open-apis/drive/v1/files/{folder_token}/children`` -- which does not
exist (the correct call is ``GET /open-apis/drive/v1/files?folder_token=xxx``)
and only failed at runtime.  Routing all ad-hoc calls through this tool turns
that runtime 404 into a pre-flight rejection that names the correct endpoint.

Validation is pure Python (no SDK import), so it is independently testable.
Dispatch reuses ``feishu_drive_tool._do_request`` for response parsing.
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
# The navigation map: (METHOD, path_template).
#
# A ``:segment`` matches any single concrete path segment (a token / id).
# This list is the source of truth for what the agent is allowed to call --
# fail-closed: anything not listed here is rejected.  It is intentionally
# curated (not auto-generated) so every entry is a verified-correct endpoint.
# Extend it as new endpoints are needed; keep templates exactly as Feishu
# documents them (https://open.feishu.cn/document/server-docs).
# ---------------------------------------------------------------------------
_FEISHU_ENDPOINTS = [
    # --- Drive: files & folders ---------------------------------------------
    # List items in a folder == the call the agent kept getting wrong.
    # It is a QUERY param (?folder_token=), NOT a /:token/children path.
    ("GET", "/open-apis/drive/v1/files"),
    ("POST", "/open-apis/drive/v1/files/create_folder"),
    ("GET", "/open-apis/drive/v1/files/:file_token/statistics"),
    ("POST", "/open-apis/drive/v1/metas/batch_query"),
    # root_folder/meta is not in the typed lark SDK; verified against the
    # official docs (GET .../drive/explorer/v2/root_folder/meta).
    ("GET", "/open-apis/drive/explorer/v2/root_folder/meta"),
    ("GET", "/open-apis/drive/v1/files/:file_token/comments"),
    # new_comments mirrors feishu_drive_tool._ADD_COMMENT_URI (shipping code);
    # it is the whole-document-comment endpoint, distinct from .../comments.
    ("POST", "/open-apis/drive/v1/files/:file_token/new_comments"),
    ("GET", "/open-apis/drive/v1/files/:file_token/comments/:comment_id/replies"),
    ("POST", "/open-apis/drive/v1/files/:file_token/comments/:comment_id/replies"),
    # --- Docx ----------------------------------------------------------------
    ("GET", "/open-apis/docx/v1/documents/:document_id"),
    ("GET", "/open-apis/docx/v1/documents/:document_id/raw_content"),
    ("GET", "/open-apis/docx/v1/documents/:document_id/blocks"),
    # --- Bitable (multi-dimensional tables) ----------------------------------
    ("GET", "/open-apis/bitable/v1/apps/:app_token"),
    ("GET", "/open-apis/bitable/v1/apps/:app_token/tables"),
    ("GET", "/open-apis/bitable/v1/apps/:app_token/tables/:table_id/fields"),
    ("GET", "/open-apis/bitable/v1/apps/:app_token/tables/:table_id/records"),
    ("POST", "/open-apis/bitable/v1/apps/:app_token/tables/:table_id/records"),
    ("POST", "/open-apis/bitable/v1/apps/:app_token/tables/:table_id/records/search"),
    ("PUT", "/open-apis/bitable/v1/apps/:app_token/tables/:table_id/records/:record_id"),
    ("DELETE", "/open-apis/bitable/v1/apps/:app_token/tables/:table_id/records/:record_id"),
    # --- Sheets --------------------------------------------------------------
    ("GET", "/open-apis/sheets/v3/spreadsheets/:spreadsheet_token"),
    ("GET", "/open-apis/sheets/v3/spreadsheets/:spreadsheet_token/sheets/query"),
    # --- Wiki ----------------------------------------------------------------
    ("GET", "/open-apis/wiki/v2/spaces"),
    ("GET", "/open-apis/wiki/v2/spaces/:space_id/nodes"),
]

_ALLOWED_METHODS = {"GET", "POST", "PUT", "DELETE", "PATCH"}


def _normalize_path(path: str) -> str:
    """Reduce any caller-supplied form to a bare ``/open-apis/...`` path.

    The agent often pastes a full URL (the original incident used
    ``https://open.feishu.cn/open-apis/drive/v1/files?page_size=50``). Slice
    from ``/open-apis/`` so scheme+host don't get mistaken for path segments
    and reject an otherwise-valid call.
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

    # Collect every structural match, then prefer the most specific one
    # (fewest wildcards). This stops a literal segment like ``create_folder``
    # or ``search`` from being captured as a ``:token`` by a looser template.
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

    # No match -> build "did you mean" suggestions. Normalise the caller's
    # concrete tokens to ``:x`` so fuzzy matching compares shapes, not tokens.
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
        "Call any Feishu/Lark Open API endpoint. The path is validated against "
        "a map of known-good endpoints BEFORE the call -- an unknown path is "
        "rejected with the correct one suggested, instead of 404ing. "
        "Use this for Drive/Docx/Bitable/Sheets/Wiki reads and writes. "
        "Note: list items in a folder is "
        "GET /open-apis/drive/v1/files with query folder_token=xxx "
        "(query param -- there is NO /files/{token}/children path)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "method": {
                "type": "string",
                "description": "HTTP method.",
                "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"],
            },
            "path": {
                "type": "string",
                "description": (
                    "API path starting with /open-apis/, with concrete tokens "
                    "inline, e.g. /open-apis/bitable/v1/apps/APP_TOKEN/tables."
                ),
            },
            "query": {
                "type": "object",
                "description": "Optional query parameters as a flat object.",
            },
            "body": {
                "type": "object",
                "description": "Optional JSON body for POST/PUT/PATCH requests.",
            },
        },
        "required": ["method", "path"],
    },
}


def _handle_feishu_request(args: dict, **kwargs) -> str:
    method = str(args.get("method", "")).strip().upper()
    path = str(args.get("path", "")).strip()
    query = args.get("query") or {}
    body = args.get("body")

    if not method or not path:
        return tool_error("method and path are required")
    if method not in _ALLOWED_METHODS:
        return tool_error(
            f"Unsupported method '{method}'. Use one of: "
            f"{', '.join(sorted(_ALLOWED_METHODS))}."
        )

    template, result = validate_endpoint(method, path)
    if template is None:
        # Log the rejection so a guessed endpoint is auditable in agent.log
        # rather than silently bounced — mirrors feishu_comment's API logging.
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
        body=body,
    )
    if code != 0:
        logger.warning("[Feishu-Request] %s %s failed: code=%s msg=%s",
                       method, template, code, msg)
        return tool_error(f"Feishu request failed: code={code} msg={msg}")

    return tool_result(success=True, data=data)


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
    description="Call a validated Feishu Open API endpoint",
    emoji="\U0001f5fa️",
)
