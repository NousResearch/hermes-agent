"""Bitwarden CLI tools for secret-safe vault access.

The native Bitwarden toolset intentionally avoids returning secret values in tool
results. Secret retrieval writes to a mode-600 temp file and returns only a file
reference so another local command can consume it without the secret entering the
LLM transcript.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from tools.registry import registry


BW_TOOLSET = "bitwarden"
_BW_BIN = os.getenv("BW_BIN", "bw")
_SESSION_GLOBS = ("/tmp/hermes_bw_session*",)
_SECRET_DIR = Path(os.getenv("HERMES_BW_SECRET_DIR", tempfile.gettempdir()))
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class BitwardenError(RuntimeError):
    """Raised for expected Bitwarden CLI failures."""


def _json_result(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _run_bw(args: List[str], *, session: Optional[str] = None, input_text: str | None = None, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    cmd = [_BW_BIN, *args]
    if session:
        cmd.extend(["--session", session])
    env = os.environ.copy()
    # Keep BW_SESSION out of child env unless explicitly passed as --session so
    # command behavior is auditable and does not accidentally depend on globals.
    env.pop("BW_SESSION", None)
    return subprocess.run(
        cmd,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
        env=env,
    )


def _check_bitwarden_requirements() -> bool:
    try:
        result = subprocess.run([_BW_BIN, "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5, check=False)
        return result.returncode == 0
    except Exception:
        return False


def _parse_json(text: str, fallback: Any = None) -> Any:
    try:
        return json.loads(text or "")
    except Exception:
        return fallback


def _status_for_session(session: Optional[str] = None) -> Dict[str, Any]:
    result = _run_bw(["status", "--raw"], session=session, timeout=10)
    data = _parse_json(result.stdout, {}) if result.returncode == 0 else {}
    return {
        "ok": result.returncode == 0,
        "status": data.get("status") or "unknown",
        "user_email": data.get("userEmail") or None,
        "last_sync": data.get("lastSync") or None,
        "stderr": (result.stderr or "").strip(),
    }


def _iter_session_files() -> List[Path]:
    files: List[Path] = []
    for pattern in _SESSION_GLOBS:
        files.extend(Path("/tmp").glob(Path(pattern).name))
    return sorted({p for p in files if p.is_file()}, key=lambda p: p.stat().st_mtime, reverse=True)


def _find_unlocked_session() -> tuple[Optional[str], str, Dict[str, Any]]:
    env_session = os.getenv("BW_SESSION")
    if env_session:
        status = _status_for_session(env_session)
        if status.get("status") == "unlocked":
            return env_session, "env", status

    for path in _iter_session_files():
        try:
            mode = stat.S_IMODE(path.stat().st_mode)
            if mode & 0o077:
                # Session files should not be group/world-readable.
                continue
            session = path.read_text(encoding="utf-8").strip()
        except Exception:
            continue
        if not session:
            continue
        status = _status_for_session(session)
        if status.get("status") == "unlocked":
            return session, str(path), status

    status = _status_for_session(None)
    return None, "none", status


def _require_unlocked() -> tuple[str, str, Dict[str, Any]]:
    session, source, status = _find_unlocked_session()
    if session:
        return session, source, status
    raise BitwardenError(
        "Bitwarden is locked or unauthenticated. Run `bw unlock`/`bw login`, "
        "or provide a task-scoped /tmp/hermes_bw_session* file."
    )


def _redact_email(email: Optional[str]) -> Optional[str]:
    if not email or "@" not in email:
        return email
    local, domain = email.split("@", 1)
    if len(local) <= 2:
        redacted_local = local[:1] + "*"
    else:
        redacted_local = local[:2] + "***" + local[-1:]
    return f"{redacted_local}@{domain}"


def _item_type_name(item_type: Any) -> str:
    return {1: "login", 2: "secure_note", 3: "card", 4: "identity"}.get(item_type, str(item_type or "unknown"))


def _item_metadata(item: Dict[str, Any]) -> Dict[str, Any]:
    login = item.get("login") or {}
    uris = login.get("uris") or []
    return {
        "name": item.get("name"),
        "type": _item_type_name(item.get("type")),
        "username": login.get("username") or None,
        "has_password": bool(login.get("password")),
        "has_totp": bool(login.get("totp")),
        "has_notes": bool(item.get("notes")),
        "uris": [u.get("uri") for u in uris if u.get("uri")],
        "revision_date": item.get("revisionDate"),
    }


def _list_items(search: str, session: str, limit: int) -> List[Dict[str, Any]]:
    args = ["list", "items", "--raw"]
    if search:
        args.extend(["--search", search])
    result = _run_bw(args, session=session, timeout=45)
    if result.returncode != 0:
        raise BitwardenError((result.stderr or "Bitwarden item search failed").strip())
    items = _parse_json(result.stdout, []) or []
    return items[: max(1, min(limit, 50))]


def _find_exact_item(name: str, session: str) -> Dict[str, Any]:
    items = _list_items(name, session, limit=50)
    matches = [item for item in items if item.get("name") == name]
    if not matches:
        raise BitwardenError(f"No Bitwarden item found with exact name: {name}")
    if len(matches) > 1:
        raise BitwardenError(f"Multiple Bitwarden items found with exact name: {name}")
    return matches[0]


def bitwarden_status() -> str:
    """Return non-secret Bitwarden CLI status."""
    try:
        version = subprocess.run([_BW_BIN, "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5, check=False)
        session, source, status = _find_unlocked_session()
        return _json_result({
            "success": True,
            "available": version.returncode == 0,
            "version": (version.stdout or "").strip() or None,
            "status": status.get("status"),
            "unlocked": bool(session),
            "session_source": source if session else None,
            "user_email": _redact_email(status.get("user_email")),
            "last_sync": status.get("last_sync"),
        })
    except Exception as exc:
        return _json_result({"success": False, "error": str(exc), "available": False})


def bitwarden_search(query: str = "", limit: int = 10) -> str:
    """Search Bitwarden and return item metadata only — never secrets."""
    try:
        session, source, _ = _require_unlocked()
        items = _list_items(str(query or ""), session, int(limit or 10))
        return _json_result({
            "success": True,
            "query": query,
            "count": len(items),
            "session_source": source,
            "items": [_item_metadata(item) for item in items],
        })
    except Exception as exc:
        return _json_result({"success": False, "error": str(exc), "needs_unlock": True})


def _extract_secret(item: Dict[str, Any], field: str, session: str) -> str:
    login = item.get("login") or {}
    if field == "password":
        return login.get("password") or ""
    if field == "username":
        return login.get("username") or ""
    if field == "notes":
        return item.get("notes") or ""
    if field == "totp":
        item_id = item.get("id")
        if not item_id:
            return ""
        result = _run_bw(["get", "totp", item_id, "--raw"], session=session, timeout=20)
        if result.returncode != 0:
            raise BitwardenError((result.stderr or "Failed to get TOTP").strip())
        return (result.stdout or "").strip()
    raise BitwardenError(f"Unsupported field: {field}")


def bitwarden_get_secret_ref(item_name: str, field: str = "password", env_var: str = "BW_SECRET_VALUE") -> str:
    """Write a selected secret to a mode-600 temp env file and return only the path."""
    try:
        if field not in {"password", "username", "notes", "totp"}:
            raise BitwardenError("field must be one of: password, username, notes, totp")
        if not _ENV_NAME_RE.match(env_var or ""):
            raise BitwardenError("env_var must be a valid shell environment variable name")
        session, source, _ = _require_unlocked()
        item = _find_exact_item(item_name, session)
        secret = _extract_secret(item, field, session)
        if not secret:
            raise BitwardenError(f"Selected field is empty for item: {item_name}")

        _SECRET_DIR.mkdir(parents=True, exist_ok=True)
        path = _SECRET_DIR / f"hermes_bw_secret_{uuid4().hex}.env"
        content = f"export {env_var}={shlex.quote(secret)}\n"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        return _json_result({
            "success": True,
            "item": _item_metadata(item),
            "field": field,
            "env_var": env_var,
            "env_file": str(path),
            "session_source": source,
            "secret_returned": False,
            "cleanup_hint": f"Delete {path} after use.",
        })
    except Exception as exc:
        return _json_result({"success": False, "error": str(exc), "needs_unlock": isinstance(exc, BitwardenError)})


def _encode_payload(payload: Dict[str, Any], session: str) -> str:
    result = _run_bw(["encode"], session=None, input_text=json.dumps(payload), timeout=20)
    if result.returncode != 0:
        raise BitwardenError((result.stderr or "bw encode failed").strip())
    return (result.stdout or "").strip()


def bitwarden_upsert_login(name: str, username: str = "", secret: str = "", uri: str = "", notes: str = "") -> str:
    """Create or update a login item. The secret is never echoed in the result."""
    try:
        if not name:
            raise BitwardenError("name is required")
        if not secret:
            raise BitwardenError("secret is required")
        session, source, _ = _require_unlocked()
        existing = None
        try:
            existing = _find_exact_item(name, session)
        except BitwardenError:
            existing = None

        login: Dict[str, Any] = {"username": username or "", "password": secret}
        if uri:
            login["uris"] = [{"uri": uri, "match": 0}]
        payload: Dict[str, Any]
        mode: str
        if existing:
            payload = dict(existing)
            payload["type"] = 1
            payload["name"] = name
            payload["login"] = login
            if notes:
                payload["notes"] = notes
            encoded = _encode_payload(payload, session)
            result = _run_bw(["edit", "item", existing["id"], encoded], session=session, timeout=45)
            mode = "updated"
        else:
            payload = {"type": 1, "name": name, "login": login}
            if notes:
                payload["notes"] = notes
            encoded = _encode_payload(payload, session)
            result = _run_bw(["create", "item", encoded], session=session, timeout=45)
            mode = "created"
        if result.returncode != 0:
            raise BitwardenError((result.stderr or "Bitwarden upsert failed").strip())
        _run_bw(["sync"], session=session, timeout=60)
        saved = _parse_json(result.stdout, {}) or payload
        return _json_result({
            "success": True,
            "mode": mode,
            "session_source": source,
            "item": _item_metadata(saved),
            "secret_returned": False,
            "synced": True,
        })
    except Exception as exc:
        return _json_result({"success": False, "error": str(exc), "needs_unlock": isinstance(exc, BitwardenError)})


BITWARDEN_STATUS_SCHEMA = {
    "name": "bitwarden_status",
    "description": "Check Bitwarden CLI availability and lock status. Never returns secrets.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

BITWARDEN_SEARCH_SCHEMA = {
    "name": "bitwarden_search",
    "description": "Search Bitwarden vault items and return metadata only, never passwords/tokens/notes.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search string passed to Bitwarden."},
            "limit": {"type": "integer", "description": "Maximum metadata results, 1-50.", "default": 10},
        },
        "required": [],
    },
}

BITWARDEN_GET_SECRET_REF_SCHEMA = {
    "name": "bitwarden_get_secret_ref",
    "description": "Retrieve a secret from an exact Bitwarden item name into a mode-600 temp env file. Returns only a file path, never the secret value.",
    "parameters": {
        "type": "object",
        "properties": {
            "item_name": {"type": "string", "description": "Exact Bitwarden item name."},
            "field": {"type": "string", "enum": ["password", "username", "notes", "totp"], "default": "password"},
            "env_var": {"type": "string", "description": "Environment variable name to write in the temp env file.", "default": "BW_SECRET_VALUE"},
        },
        "required": ["item_name"],
    },
}

BITWARDEN_UPSERT_LOGIN_SCHEMA = {
    "name": "bitwarden_upsert_login",
    "description": "Create or update a Bitwarden login/API-token item. Does not echo the supplied secret.",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Bitwarden item name."},
            "username": {"type": "string", "description": "Login username or token label."},
            "secret": {"type": "string", "description": "Password/token to save. This is never returned."},
            "uri": {"type": "string", "description": "Optional login URI."},
            "notes": {"type": "string", "description": "Optional non-public note to store with the item."},
        },
        "required": ["name", "secret"],
    },
}


def _handle_status(args, **kwargs):
    return bitwarden_status()


def _handle_search(args, **kwargs):
    return bitwarden_search(query=args.get("query", ""), limit=args.get("limit", 10))


def _handle_get_secret_ref(args, **kwargs):
    return bitwarden_get_secret_ref(
        item_name=args.get("item_name", ""),
        field=args.get("field", "password"),
        env_var=args.get("env_var", "BW_SECRET_VALUE"),
    )


def _handle_upsert_login(args, **kwargs):
    return bitwarden_upsert_login(
        name=args.get("name", ""),
        username=args.get("username", ""),
        secret=args.get("secret", ""),
        uri=args.get("uri", ""),
        notes=args.get("notes", ""),
    )


registry.register(name="bitwarden_status", toolset=BW_TOOLSET, schema=BITWARDEN_STATUS_SCHEMA, handler=_handle_status, check_fn=_check_bitwarden_requirements, emoji="🔐")
registry.register(name="bitwarden_search", toolset=BW_TOOLSET, schema=BITWARDEN_SEARCH_SCHEMA, handler=_handle_search, check_fn=_check_bitwarden_requirements, emoji="🔎")
registry.register(name="bitwarden_get_secret_ref", toolset=BW_TOOLSET, schema=BITWARDEN_GET_SECRET_REF_SCHEMA, handler=_handle_get_secret_ref, check_fn=_check_bitwarden_requirements, emoji="🔑")
registry.register(name="bitwarden_upsert_login", toolset=BW_TOOLSET, schema=BITWARDEN_UPSERT_LOGIN_SCHEMA, handler=_handle_upsert_login, check_fn=_check_bitwarden_requirements, emoji="💾")
