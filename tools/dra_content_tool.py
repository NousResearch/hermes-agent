"""DRA Content — Hermes' publishing surface for browser-renderable artifacts.

Toolset: ``dra_content``. Enabled as of Stage 14 for the ``telegram`` and
``api_server`` platforms via ``platform_toolsets`` in config.yaml. Note
that a toolset name must appear in BOTH ``toolsets.py``'s TOOLSETS catalog
and ``hermes_cli/tools_config.py``'s CONFIGURABLE_TOOLSETS -- platform
resolution silently drops any name missing from the latter, with no error
anywhere, which cost a full debugging cycle in Stage 11.

Every handler is a thin translation layer over the DRA Content API: build a
JSON body, call dra_content_client, translate the response into a small
model-facing result. Authorization, versioning, storage, search and the
default-private-ACL invariant all live in the API itself -- nothing here
re-implements or second-guesses them. In particular, no handler may set an
artifact's ACL to anything broader than VIEWER, and none accepts a
"make this public" argument, because the API has no such capability to
call in the first place (DRA Content Stage 4/9).

``content_publish`` was built and proven first, alone, per the project's
repetitive-task rule; the other nine follow its exact shape.
"""
from __future__ import annotations

import base64
import os
import json
import logging

from tools import dra_content_client as client
from tools.dra_content_client import DRAContentError
from tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)

TOOLSET = "dra_content"
EMOJI = "\U0001F4C4"

# Files a model can plausibly generate as UTF-8 text in a tool call.
TEXT_EXTENSIONS = (".html", ".htm", ".css", ".js", ".mjs", ".json", ".csv", ".txt", ".md", ".svg")

# Binary assets arrive by reference, never through the model's context: the
# model names a file already on disk (a generated render, a downloaded chart)
# and this tool reads and encodes the bytes itself. A model cannot paste a
# megabyte of base64 into a tool call, which is why the text-only path could
# never carry an image -- and why images were being parked in Drive and
# inlined as data: URIs instead, storing every asset twice and producing
# multi-megabyte pages that cannot be cached per image.
BINARY_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".avif", ".ico",
                     ".woff", ".woff2", ".ttf", ".otf")

# Standalone documents go to Google Drive, never here. An artifact is a
# website -- a page and the assets it renders. A PDF, Word file, spreadsheet
# or slide deck is a document in its own right: Drive already files it,
# permissions it and can edit it, and publishing a second copy here would put
# one file under two permission systems with this one unable to see the
# other's. Rejected by name rather than by omission so the error can say where
# the file actually belongs.
DRIVE_ONLY_EXTENSIONS = {
    ".pdf": "PDF", ".doc": "Word document", ".docx": "Word document",
    ".xls": "spreadsheet", ".xlsx": "spreadsheet",
    ".ppt": "slide deck", ".pptx": "slide deck",
    ".zip": "ZIP archive",
}

# content-api allows 25 MiB per version, but the request is JSON and base64
# inflates by 4/3, and nginx caps the body at 30 MB. 20 MiB of real bytes is
# the largest that reliably fits: 20 * 4/3 = 26.7 MB of base64.
MAX_SOURCE_BYTES = 20 * 1024 * 1024

ARTIFACT_TYPES = ["document", "report", "presentation", "dashboard", "analysis",
                  "proposal", "comparison", "brief", "memo", "specification",
                  "marketing", "other"]

FILES_SCHEMA = {
    "type": "array",
    "description": (
        "The artifact's files. Must include an 'index.html' path. CSS goes "
        "in a separate file (e.g. styles.css) referenced with <link "
        "rel=stylesheet>; JS in a separate file (e.g. app.js) referenced "
        "with <script src=...>. Inline <script> tags will NOT execute -- "
        "the rendering origin's Content-Security-Policy is script-src "
        "'self', external files only. Images and other binary assets are "
        "uploaded by naming their path on disk in 'source_path' -- never "
        "as a data: URI and never via Drive."
    ),
    "items": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "e.g. index.html, styles.css, img/hero.webp"},
            "content": {"type": "string", "description": "The file's full text content. Use this for text files."},
            "source_path": {
                "type": "string",
                "description": (
                    "Absolute path to an existing file on disk to upload as-is, "
                    "instead of 'content'. Use this for images and any other "
                    "binary asset -- e.g. a generated render. Do not base64 it "
                    "yourself and do not inline it as a data: URI."
                ),
            },
        },
        "required": ["path"],
    },
    "minItems": 1,
}


def _coerce_files(files):
    """Normalise whatever the model actually sent into a list of dicts.

    Models do not reliably emit a native JSON array for an array-typed
    parameter. Some emit the whole array as a JSON-encoded *string*; when
    that happens Hermes' own coerce_tool_args cannot parse it either and
    falls back to wrapping the bare string in a single-element list. The
    first production call of this tool hit exactly that, and the old code
    -- which assumed every element was a dict -- died with
    "'str' object has no attribute 'get'", an internal error the model
    could do nothing useful with.

    So: accept a JSON string at either level, and if it still is not
    usable, fail with a message that tells the model the exact shape to
    send rather than leaking an AttributeError.
    """
    shape_hint = (
        'files must be an array of objects, each {"path": "index.html", '
        '"content": "<html>..."} -- not a JSON-encoded string'
    )

    if isinstance(files, str):
        try:
            files = json.loads(files)
        except (ValueError, TypeError):
            raise ValueError(shape_hint)

    if isinstance(files, dict):
        files = [files]

    if not isinstance(files, list) or not files:
        raise ValueError(shape_hint)

    out = []
    for item in files:
        if isinstance(item, str):
            try:
                item = json.loads(item)
            except (ValueError, TypeError):
                raise ValueError(shape_hint)
        if not isinstance(item, dict):
            raise ValueError(shape_hint)
        out.append(item)

    # A JSON-encoded array nested one level down: [[{...}, {...}]]
    if len(out) == 1 and isinstance(out[0], list):
        out = out[0]
    return out


def _read_source(path_on_disk: str, artifact_path: str) -> bytes:
    """Read a binary asset the model named, with the checks that matter.

    The model already has file tools, so this adds no ability to reach a file
    it could not otherwise read. What it does add is a route from a file to a
    URL other people can open, so the extension allowlist is the control that
    matters: it is what stops a credential file or a database dump being
    published as an artifact asset.
    """
    resolved = os.path.realpath(os.path.expanduser(path_on_disk))
    if not os.path.isfile(resolved):
        raise ValueError(
            "file %r: source_path %r does not exist or is not a regular file"
            % (artifact_path, path_on_disk)
        )
    size = os.path.getsize(resolved)
    if size == 0:
        raise ValueError("file %r: source_path %r is empty" % (artifact_path, path_on_disk))
    if size > MAX_SOURCE_BYTES:
        raise ValueError(
            "file %r is %.1f MB; the limit is %d MB per file"
            % (artifact_path, size / 1048576.0, MAX_SOURCE_BYTES // 1048576)
        )
    with open(resolved, "rb") as handle:
        return handle.read()


def _encode_files(files) -> list:
    """Normalise each file to {path, content_b64}.

    Text arrives as 'content' -- what the model typed. Binary arrives as
    'source_path' -- a file on disk this function reads itself, so the bytes
    never pass through the model's context. Extensions are validated here,
    against the same allowlist content-api enforces, so a bad request fails
    with a clear message rather than a generic 422.
    """
    out = []
    total = 0
    for f in _coerce_files(files):
        path = str(f.get("path", "")).strip()
        if not path:
            raise ValueError("every file needs a non-empty path")
        content = f.get("content")
        source_path = f.get("source_path")
        ext = "." + path.rsplit(".", 1)[-1].lower() if "." in path else ""

        if ext in DRIVE_ONLY_EXTENSIONS:
            raise ValueError(
                "file %r is a %s. Standalone documents are not published as "
                "artifacts -- upload it to Google Drive instead (see the "
                "draas-drive-organization skill) and hand over the Drive link. "
                "DRA Content holds websites: HTML plus the CSS, JavaScript, "
                "images, fonts and data files that page renders."
                % (path, DRIVE_ONLY_EXTENSIONS[ext])
            )

        if source_path:
            if ext not in BINARY_EXTENSIONS and ext not in TEXT_EXTENSIONS:
                raise ValueError(
                    "file %r: extension %s is not allowed (%s)"
                    % (path, ext or "(none)", ", ".join(BINARY_EXTENSIONS + TEXT_EXTENSIONS))
                )
            blob = _read_source(str(source_path), path)
        else:
            if content is None:
                raise ValueError(
                    "file %r needs either 'content' (text) or 'source_path' "
                    "(a file on disk, for images and other binary assets)" % path
                )
            if ext in BINARY_EXTENSIONS:
                raise ValueError(
                    "file %r is a binary type (%s); pass it as 'source_path' "
                    "naming the file on disk, not as inline 'content'" % (path, ext)
                )
            if ext not in TEXT_EXTENSIONS:
                raise ValueError(
                    "file %r: extension %s is not a supported text type (%s)"
                    % (path, ext or "(none)", ", ".join(TEXT_EXTENSIONS))
                )
            blob = str(content).encode("utf-8")

        total += len(blob)
        if total > MAX_SOURCE_BYTES:
            raise ValueError(
                "the version totals more than %d MB; publish fewer or smaller assets"
                % (MAX_SOURCE_BYTES // 1048576)
            )
        out.append({"path": path, "content_b64": base64.b64encode(blob).decode("ascii")})
    return out


def _entry_file(paths: set) -> str:
    return "index.html" if "index.html" in paths else next(iter(sorted(paths)))


# =============================================================================
# content_publish
# =============================================================================

PUBLISH_SCHEMA = {
    "name": "content_publish",
    "description": (
        "Publish a NEW DRA Content artifact -- the default destination for "
        "document-like deliverables (reports, analyses, comparisons, "
        "proposals, briefs, specifications). Returns a permanent artifact "
        "ID and canonical URL. New artifacts are PRIVATE by default: only "
        "the platform administrators can see them until you call "
        "content_share. Do not use this for a simple conversational answer "
        "-- only for a reusable document-like deliverable. If the user "
        "explicitly asked for a Google Doc, Word/DOCX, PDF, Excel/XLSX or "
        "Google Sheet, use that workflow instead, not this tool. If you are "
        "updating something you (or another session) already published, "
        "use content_update instead -- search with content_find first if "
        "you are not certain whether this already exists."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Artifact title, shown in the admin UI and browser tab"},
            "files": FILES_SCHEMA,
            "description": {"type": "string", "description": "One or two sentence summary, used in search"},
            "artifact_type": {"type": "string", "enum": ARTIFACT_TYPES, "description": "Default 'document' if unsure"},
            "project": {"type": "string", "description": "Resolve via entity_resolver first; do not invent a name"},
            "entity": {"type": "string", "description": "Resolve via entity_resolver first; do not invent a name"},
            "category": {"type": "string", "description": "e.g. Legal, Construction, Marketing"},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["title", "files"],
    },
}


async def _handle_publish(args, **kw):
    args = args or {}
    title = str(args.get("title", "")).strip()
    files = args.get("files")
    if not title:
        return tool_error("title is required")
    if not files:
        return tool_error("files is required and must be a non-empty list")

    try:
        encoded = _encode_files(files)
    except ValueError as exc:
        return tool_error(str(exc))

    paths = set(f["path"] for f in encoded)
    entry_file = _entry_file(paths)
    if entry_file != "index.html":
        logger.info("content_publish: no index.html supplied, using %r as entry file", entry_file)

    body = {
        "title": title,
        "files": [{"path": f["path"], "content_b64": f["content_b64"]} for f in encoded],
        "entry_file": entry_file,
        "description": args.get("description"),
        "artifact_type": args.get("artifact_type") or "document",
        "project": args.get("project"),
        "entity": args.get("entity"),
        "category": args.get("category"),
        "tags": args.get("tags") or [],
        # dispatch() is called with task_id=... / user_task=..., never a
        # "session_id" kwarg -- there is no separate conversation-session id
        # surfaced to tool handlers in this codebase. task_id is the closest
        # available identifier (this specific tool-call/turn) and is stored
        # as-is rather than invented.
        "source_session_id": kw.get("task_id"),
        "change_summary": "Initial version",
    }
    body = dict((k, v) for k, v in body.items() if v is not None)

    try:
        result = await client.post("/api/artifacts", json_body=body)
    except DRAContentError as exc:
        return tool_error("could not publish artifact: %s" % exc.detail, status_code=exc.status_code)

    return tool_result({
        "artifact_id": result["human_id"],
        "title": result["title"],
        "version": result["current_version"],
        "url": result["url"],
        "access": "Platform administrators only (default). Use content_share to grant access.",
    })


# =============================================================================
# content_update
# =============================================================================

UPDATE_SCHEMA = {
    "name": "content_update",
    "description": (
        "Publish a new version of an EXISTING artifact. The artifact ID stays "
        "the same, the canonical URL stays the same, and the previous version "
        "remains permanently retrievable -- this is the correct tool when the "
        "user asks you to revise, update, or add to something already "
        "published. Use content_find first if you are not certain which "
        "artifact they mean; never guess an artifact ID."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "artifact_id": {"type": "string", "description": "e.g. DRA-ART-2026-000184"},
            "files": FILES_SCHEMA,
            "change_summary": {"type": "string", "description": "What changed in this version, e.g. 'Added ABC's revised quote'"},
        },
        "required": ["artifact_id", "files", "change_summary"],
    },
}


async def _handle_update(args, **kw):
    args = args or {}
    artifact_id = str(args.get("artifact_id", "")).strip()
    files = args.get("files")
    change_summary = str(args.get("change_summary", "")).strip()
    if not artifact_id:
        return tool_error("artifact_id is required")
    if not files:
        return tool_error("files is required and must be a non-empty list")
    if not change_summary:
        return tool_error("change_summary is required -- say what changed in this version")

    try:
        encoded = _encode_files(files)
    except ValueError as exc:
        return tool_error(str(exc))

    entry_file = _entry_file(set(f["path"] for f in encoded))
    body = {
        "files": [{"path": f["path"], "content_b64": f["content_b64"]} for f in encoded],
        "entry_file": entry_file,
        "change_summary": change_summary,
        "source_session": kw.get("task_id"),
    }
    try:
        result = await client.post("/api/artifacts/%s/versions" % artifact_id, json_body=body)
    except DRAContentError as exc:
        return tool_error("could not update %s: %s" % (artifact_id, exc.detail), status_code=exc.status_code)

    return tool_result({
        "artifact_id": artifact_id,
        "version": result["version_number"],
        "change_summary": result["change_summary"],
    })


# =============================================================================
# content_find
# =============================================================================

FIND_SCHEMA = {
    "name": "content_find",
    "description": (
        "Search DRA Content artifacts by title, project, tag, description, or "
        "artifact ID -- fuzzy matching included, so a half-remembered name "
        "like 'the Amber contractor comparison' works. Returns ranked "
        "candidates and a 'confident' flag: when confident is true there is "
        "exactly one clear match and you may act on it directly (e.g. before "
        "calling content_update); when false, several plausible candidates "
        "exist and you should present them and ask which one is meant, "
        "rather than guessing."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search text or an exact artifact ID"},
            "project": {"type": "string", "description": "Optional: narrow to one project"},
            "include_archived": {"type": "boolean", "description": "Default false"},
        },
        "required": ["query"],
    },
}


async def _handle_find(args, **kw):
    args = args or {}
    query = str(args.get("query", "")).strip()
    if not query:
        return tool_error("query is required")

    params = {"q": query, "limit": 10}
    if args.get("project"):
        params["project"] = args["project"]
    if args.get("include_archived"):
        params["include_archived"] = "true"

    try:
        result = await client.get("/api/search", params=params)
    except DRAContentError as exc:
        return tool_error("search failed: %s" % exc.detail, status_code=exc.status_code)

    return tool_result({
        "confident": result["confident"],
        "count": result["count"],
        "results": [
            {"artifact_id": r["human_id"], "title": r["title"], "project": r.get("project"),
             "type": r["artifact_type"], "version": r["current_version"], "url": r["url"],
             "match": r["match"], "score": r["score"]}
            for r in result["results"]
        ],
    })


# =============================================================================
# content_get
# =============================================================================

GET_SCHEMA = {
    "name": "content_get",
    "description": "Fetch one artifact's current metadata by its exact ID.",
    "parameters": {
        "type": "object",
        "properties": {"artifact_id": {"type": "string", "description": "e.g. DRA-ART-2026-000184"}},
        "required": ["artifact_id"],
    },
}


async def _handle_get(args, **kw):
    args = args or {}
    artifact_id = str(args.get("artifact_id", "")).strip()
    if not artifact_id:
        return tool_error("artifact_id is required")
    try:
        result = await client.get("/api/artifacts/%s" % artifact_id)
    except DRAContentError as exc:
        return tool_error("could not fetch %s: %s" % (artifact_id, exc.detail), status_code=exc.status_code)
    keep = ("uuid", "human_id", "title", "description", "artifact_type",
            "project", "entity", "category", "tags", "status",
            "current_version", "archived", "url", "created_at", "updated_at")
    return tool_result(dict((k, result[k]) for k in keep if k in result))


# =============================================================================
# content_list_versions
# =============================================================================

LIST_VERSIONS_SCHEMA = {
    "name": "content_list_versions",
    "description": "List every version of an artifact, newest first, with each version's change summary.",
    "parameters": {
        "type": "object",
        "properties": {"artifact_id": {"type": "string"}},
        "required": ["artifact_id"],
    },
}


async def _handle_list_versions(args, **kw):
    args = args or {}
    artifact_id = str(args.get("artifact_id", "")).strip()
    if not artifact_id:
        return tool_error("artifact_id is required")
    try:
        result = await client.get("/api/artifacts/%s/versions" % artifact_id)
    except DRAContentError as exc:
        return tool_error("could not list versions of %s: %s" % (artifact_id, exc.detail), status_code=exc.status_code)
    return tool_result({"artifact_id": artifact_id, "versions": [
        {"version": v["version_number"], "change_summary": v["change_summary"],
         "status": v["publication_status"], "created_at": v["created_at"]}
        for v in result
    ]})


# =============================================================================
# content_share
# =============================================================================

SHARE_SCHEMA = {
    "name": "content_share",
    "description": (
        "Grant a person access to an artifact. DEFAULTS, and they matter: "
        "omitting permission grants VIEWER (never grant ADMIN unless the "
        "user explicitly asked to make someone an administrator of this "
        "artifact); omitting days grants access for 365 days. A permanent "
        "grant (days=0) may only be requested by a platform administrator "
        "and will be refused otherwise. Never call this to make something "
        "public -- there is no public/anonymous access level in this system."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "artifact_id": {"type": "string"},
            "email": {"type": "string", "description": "Email address to grant access to"},
            "permission": {"type": "string", "enum": ["VIEWER", "ADMIN"], "description": "Default VIEWER"},
            "days": {"type": "integer", "description": "Default 365. 0 means permanent (admins only)."},
            "reason": {"type": "string", "description": "Optional context for the audit log"},
        },
        "required": ["artifact_id", "email"],
    },
}


async def _handle_share(args, **kw):
    args = args or {}
    artifact_id = str(args.get("artifact_id", "")).strip()
    email = str(args.get("email", "")).strip()
    if not artifact_id or not email:
        return tool_error("artifact_id and email are required")

    body = {"email": email}
    if args.get("permission"):
        body["permission"] = args["permission"]
    if "days" in args and args["days"] is not None:
        body["days"] = args["days"]
    if args.get("reason"):
        body["reason"] = args["reason"]

    try:
        result = await client.post("/api/artifacts/%s/permissions" % artifact_id, json_body=body)
    except DRAContentError as exc:
        return tool_error("could not share %s: %s" % (artifact_id, exc.detail), status_code=exc.status_code)

    expires = result.get("expires_at")
    return tool_result({
        "artifact_id": artifact_id, "email": email, "permission": result["permission"],
        "expires": expires or "never",
    })


# =============================================================================
# content_revoke
# =============================================================================

REVOKE_SCHEMA = {
    "name": "content_revoke",
    "description": "Remove a person's access to an artifact immediately.",
    "parameters": {
        "type": "object",
        "properties": {
            "artifact_id": {"type": "string"},
            "email": {"type": "string"},
        },
        "required": ["artifact_id", "email"],
    },
}


async def _handle_revoke(args, **kw):
    args = args or {}
    artifact_id = str(args.get("artifact_id", "")).strip()
    email = str(args.get("email", "")).strip().lower()
    if not artifact_id or not email:
        return tool_error("artifact_id and email are required")

    try:
        perms = await client.get("/api/artifacts/%s/permissions" % artifact_id)
    except DRAContentError as exc:
        return tool_error("could not look up access for %s: %s" % (artifact_id, exc.detail), status_code=exc.status_code)

    match = None
    for p in perms:
        if p["subject_kind"] == "user" and p["subject"].lower() == email and p["status"] == "ACTIVE":
            match = p
            break
    if match is None:
        return tool_error("%s does not currently have active access to %s" % (email, artifact_id))

    try:
        await client.delete("/api/artifacts/%s/permissions/%s" % (artifact_id, match["id"]))
    except DRAContentError as exc:
        return tool_error("could not revoke access: %s" % exc.detail, status_code=exc.status_code)

    return tool_result({"artifact_id": artifact_id, "email": email, "status": "revoked"})


# =============================================================================
# content_access_list
# =============================================================================

ACCESS_LIST_SCHEMA = {
    "name": "content_access_list",
    "description": "Show everyone who currently has access to an artifact, their role, and when it expires.",
    "parameters": {
        "type": "object",
        "properties": {"artifact_id": {"type": "string"}},
        "required": ["artifact_id"],
    },
}


async def _handle_access_list(args, **kw):
    args = args or {}
    artifact_id = str(args.get("artifact_id", "")).strip()
    if not artifact_id:
        return tool_error("artifact_id is required")
    try:
        perms = await client.get("/api/artifacts/%s/permissions" % artifact_id)
    except DRAContentError as exc:
        return tool_error("could not list access for %s: %s" % (artifact_id, exc.detail), status_code=exc.status_code)
    return tool_result({"artifact_id": artifact_id, "access": [
        {"subject": p["subject"], "kind": p["subject_kind"], "permission": p["permission"],
         "status": p["status"], "expires": p.get("expires_at") or "never",
         "granted_by": p.get("granted_by_email")}
        for p in perms
    ]})


# =============================================================================
# content_user_access
# =============================================================================

USER_ACCESS_SCHEMA = {
    "name": "content_user_access",
    "description": (
        "Show every artifact a specific person can currently access, and why "
        "(direct grant, group membership, ownership, or platform admin). "
        "Answers questions like 'what can abc@example.com see?'. A non-"
        "administrator asking about someone other than themselves will be "
        "refused by the API -- only platform administrators may query "
        "another person's access."
    ),
    "parameters": {
        "type": "object",
        "properties": {"email": {"type": "string"}},
        "required": ["email"],
    },
}


async def _handle_user_access(args, **kw):
    args = args or {}
    email = str(args.get("email", "")).strip().lower()
    if not email:
        return tool_error("email is required")
    try:
        result = await client.get("/api/principals/%s/permissions" % email)
    except DRAContentError as exc:
        return tool_error("could not look up access for %s: %s" % (email, exc.detail), status_code=exc.status_code)

    if not result.get("known"):
        return tool_result({"email": email, "known": False, "artifacts": []})

    return tool_result({"email": email, "known": True, "count": result["count"], "artifacts": [
        {"artifact_id": a["human_id"], "title": a["title"], "permission": a["permission"],
         "source": a["source"], "expires": a.get("expires_at") or "never"}
        for a in result["artifacts"]
    ]})


# =============================================================================
# content_archive
# =============================================================================

ARCHIVE_SCHEMA = {
    "name": "content_archive",
    "description": (
        "Archive an artifact (soft delete). It stops appearing in normal "
        "search and the dashboard, but its ID, history and files are kept "
        "permanently -- the human-readable ID is never reused."
    ),
    "parameters": {
        "type": "object",
        "properties": {"artifact_id": {"type": "string"}},
        "required": ["artifact_id"],
    },
}


async def _handle_archive(args, **kw):
    args = args or {}
    artifact_id = str(args.get("artifact_id", "")).strip()
    if not artifact_id:
        return tool_error("artifact_id is required")
    try:
        result = await client.post("/api/artifacts/%s/archive" % artifact_id)
    except DRAContentError as exc:
        return tool_error("could not archive %s: %s" % (artifact_id, exc.detail), status_code=exc.status_code)
    return tool_result({"artifact_id": artifact_id, "status": result["status"]})


# =============================================================================
# content_export
# =============================================================================

EXPORT_SCHEMA = {
    "name": "content_export",
    "description": (
        "Export an artifact to Google Drive as a Word document (DOCX), a "
        "PDF, or a native Google Doc. Use this ONLY when the user explicitly "
        "asks for one of those formats or to save something to Drive -- "
        "otherwise the artifact already lives in DRA Content and that URL is "
        "the right thing to share. Interactive elements (scripts, charts, "
        "diagrams) do not survive this conversion; only text, headings, "
        "tables and images do. The destination Drive folder is resolved "
        "automatically from the artifact's project/category -- if none is "
        "configured, this will fail with a clear message rather than "
        "guessing a folder."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "artifact_id": {"type": "string"},
            "format": {"type": "string", "enum": ["docx", "pdf", "gdoc"]},
            "version": {"type": "integer", "description": "Omit to export the current version"},
        },
        "required": ["artifact_id", "format"],
    },
}


def _resolve_export_credentials():
    """Load the session user's own Google credentials, the same way every
    other per-user GWS tool in this codebase does. Returns (None, reason)
    when there is no session context or no token -- the caller turns that
    into a clear, model-facing error rather than a stack trace."""
    try:
        from tools import gws_auth
        from tools.gws_ops_tools import _default_service_name, _vault_available
    except Exception:
        return None, "GWS tooling is not available in this environment"

    if not _vault_available():
        return None, "the credential vault is not reachable"

    service_name = _default_service_name()
    try:
        if not gws_auth.has_token(service_name):
            return None, (
                "no Google account is connected for this session (service %r). "
                "The user needs to connect their Google account before export "
                "to Drive is possible." % service_name
            )
        creds = gws_auth.load_credentials(service_name)
    except Exception as exc:
        return None, "could not load Google credentials: %s" % exc
    return creds, None


async def _handle_export(args, **kw):
    args = args or {}
    artifact_id = str(args.get("artifact_id", "")).strip()
    fmt = str(args.get("format", "")).strip().lower()
    if not artifact_id:
        return tool_error("artifact_id is required")
    if fmt not in ("docx", "pdf", "gdoc"):
        return tool_error("format must be one of: docx, pdf, gdoc")

    body = {"format": fmt}
    if args.get("version") is not None:
        body["version"] = args["version"]

    try:
        prepared = await client.post("/api/artifacts/%s/export" % artifact_id, json_body=body)
    except DRAContentError as exc:
        return tool_error("could not prepare export for %s: %s" % (artifact_id, exc.detail), status_code=exc.status_code)

    creds, err = _resolve_export_credentials()
    if creds is None:
        return tool_error("export prepared but could not upload to Drive: %s" % err)

    try:
        from tools.gws_drive_export import export_html_as
        uploaded = export_html_as(
            creds,
            name=prepared["suggested_filename"],
            html=prepared["html"],
            target_format=fmt,
            parent_id=prepared["drive_folder_id"],
        )
    except Exception as exc:
        logger.exception("content_export: Drive upload failed for %s", artifact_id)
        return tool_error("Drive upload failed: %s" % exc)

    try:
        await client.post("/api/artifacts/%s/export/complete" % artifact_id, json_body={
            "version": prepared["version_number"], "format": fmt,
            "drive_file_id": uploaded["id"], "drive_file_url": uploaded.get("url") or "",
            "drive_folder_id": prepared["drive_folder_id"],
        })
    except DRAContentError:
        # The export itself succeeded; a failure to record it is logged but
        # must not be reported to the user as an export failure.
        logger.warning("content_export: export succeeded but /export/complete failed for %s", artifact_id)

    return tool_result({
        "artifact_id": artifact_id, "format": fmt, "version": prepared["version_number"],
        "drive_url": uploaded.get("url"), "drive_file_id": uploaded["id"],
    })


# =============================================================================
# registration
# =============================================================================

registry.register(name="content_publish", toolset=TOOLSET, schema=PUBLISH_SCHEMA,
                  handler=_handle_publish, is_async=True, emoji=EMOJI)
registry.register(name="content_update", toolset=TOOLSET, schema=UPDATE_SCHEMA,
                  handler=_handle_update, is_async=True, emoji=EMOJI)
registry.register(name="content_find", toolset=TOOLSET, schema=FIND_SCHEMA,
                  handler=_handle_find, is_async=True, emoji="\U0001F50E")
registry.register(name="content_get", toolset=TOOLSET, schema=GET_SCHEMA,
                  handler=_handle_get, is_async=True, emoji=EMOJI)
registry.register(name="content_list_versions", toolset=TOOLSET, schema=LIST_VERSIONS_SCHEMA,
                  handler=_handle_list_versions, is_async=True, emoji=EMOJI)
registry.register(name="content_share", toolset=TOOLSET, schema=SHARE_SCHEMA,
                  handler=_handle_share, is_async=True, emoji="\U0001F517")
registry.register(name="content_revoke", toolset=TOOLSET, schema=REVOKE_SCHEMA,
                  handler=_handle_revoke, is_async=True, emoji="\U0001F6AB")
registry.register(name="content_access_list", toolset=TOOLSET, schema=ACCESS_LIST_SCHEMA,
                  handler=_handle_access_list, is_async=True, emoji=EMOJI)
registry.register(name="content_user_access", toolset=TOOLSET, schema=USER_ACCESS_SCHEMA,
                  handler=_handle_user_access, is_async=True, emoji=EMOJI)
registry.register(name="content_archive", toolset=TOOLSET, schema=ARCHIVE_SCHEMA,
                  handler=_handle_archive, is_async=True, emoji="\U0001F5C4")
registry.register(name="content_export", toolset=TOOLSET, schema=EXPORT_SCHEMA,
                  handler=_handle_export, is_async=True, emoji="\U0001F4E4")
