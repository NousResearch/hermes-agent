"""Import selected Firefox cookies into one Hermes-managed Camofox profile."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Iterable


DEFAULT_CAMOFOX_PROFILE_ROOT = Path("~/.camofox/profiles")


def _profile_state_path(profile_root: Path, user_id: str) -> Path:
    digest = hashlib.sha256(str(user_id).encode()).hexdigest()[:32]
    return Path(profile_root).expanduser() / digest / "storage-state.json"


def _read_firefox_cookies(profile: Path, domains: set[str]) -> list[dict]:
    source = profile.expanduser() / "cookies.sqlite"
    if not source.is_file():
        raise FileNotFoundError(f"Firefox cookies database not found under {source.parent}")
    with tempfile.TemporaryDirectory(prefix="hermes-camofox-auth-") as tmp:
        copied = Path(tmp) / "cookies.sqlite"
        shutil.copy2(source, copied)
        for suffix in ("-wal", "-shm"):
            companion = Path(f"{source}{suffix}")
            if companion.is_file():
                shutil.copy2(companion, Path(f"{copied}{suffix}"))
        conn = sqlite3.connect(f"file:{copied}?mode=ro", uri=True)
        try:
            rows = conn.execute(
                "SELECT name, value, host, path, expiry, isSecure, isHttpOnly, sameSite "
                "FROM moz_cookies"
            ).fetchall()
        finally:
            conn.close()

    selected = []
    same_site = {0: "None", 1: "Lax", 2: "Strict"}
    for name, value, host, path, expiry, secure, http_only, site in rows:
        normalized_host = str(host or "").lower().lstrip(".")
        if not any(normalized_host == d or normalized_host.endswith(f".{d}") for d in domains):
            continue
        selected.append({
            "name": str(name), "value": str(value), "domain": str(host),
            "path": str(path or "/"), "expires": int(expiry) if int(expiry or 0) > 0 else -1,
            "secure": bool(secure), "httpOnly": bool(http_only),
            "sameSite": same_site.get(int(site or 0), "None"),
        })
    return selected


def _atomic_merge_storage_state(path: Path, cookies: list[dict]) -> None:
    state = {"cookies": [], "origins": []}
    if path.is_file():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict) or not isinstance(loaded.get("cookies"), list):
            raise ValueError("Existing Camofox storage state is invalid")
        state = loaded
        state.setdefault("origins", [])
    replacements = {(c["name"], c["domain"], c.get("path", "/")) for c in cookies}
    state["cookies"] = [
        c for c in state["cookies"]
        if (c.get("name"), c.get("domain"), c.get("path", "/")) not in replacements
    ] + cookies
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(state, handle, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
        os.chmod(path, 0o600)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        Path(tmp_name).unlink(missing_ok=True)
        raise


def import_firefox_cookies(
    firefox_profile: Path,
    *,
    task_id: str,
    domains: Iterable[str],
    required_names: Iterable[str],
) -> int:
    """Merge selected cookies into exactly one stopped, deterministic profile."""
    scope = str(task_id or "").strip()
    if not scope:
        raise ValueError("A non-empty browser scope is required")
    allowed = {str(domain).strip().lower().lstrip(".") for domain in domains}
    allowed.discard("")
    required = {str(name).strip() for name in required_names}
    required.discard("")
    if not allowed or not required:
        raise ValueError("Explicit domains and required cookie names are required")
    selected = _read_firefox_cookies(Path(firefox_profile), allowed)
    present = {cookie["name"] for cookie in selected}
    missing = required - present
    if missing:
        raise ValueError(f"Required authentication cookies are absent ({len(missing)} missing)")
    if not selected:
        raise ValueError("No cookies matched the requested domains")

    from tools import browser_camofox
    user_id = browser_camofox.camofox_identity_for_scope(scope)["user_id"]
    cfg = browser_camofox._get_camofox_config()
    root = Path(os.getenv("CAMOFOX_PROFILE_DIR") or cfg.get("profile_dir") or DEFAULT_CAMOFOX_PROFILE_ROOT)
    if browser_camofox._per_thread_instances_mode(cfg):
        pool = browser_camofox._get_instance_pool(cfg)
        with pool.scope_lifecycle(scope):
            browser_camofox.stop_camofox_scope(scope)
            _atomic_merge_storage_state(_profile_state_path(root, user_id), selected)
    else:
        browser_camofox.stop_camofox_scope(scope)
        _atomic_merge_storage_state(_profile_state_path(root, user_id), selected)
    return len(selected)


def _handle_import(args, **kwargs):
    source_profile = str(args.get("source_profile") or "").strip()
    if not source_profile:
        raise ValueError("An explicit Firefox source_profile is required")
    count = import_firefox_cookies(
        Path(source_profile),
        task_id=kwargs.get("task_id"),
        domains=args.get("domains") or (),
        required_names=args.get("required_names") or (),
    )
    return json.dumps({"success": True, "imported": count, "values_returned": False})


def _check_import_available() -> bool:
    from tools.browser_camofox import is_camofox_mode
    return is_camofox_mode()


IMPORT_COOKIES_SCHEMA = {
    "name": "browser_import_cookies",
    "description": "Securely import explicitly selected Firefox cookies into the current scoped Camofox browser profile. Stops only this browser scope; the next browser_navigate starts it with a fresh tab. Cookie values are never returned.",
    "parameters": {
        "type": "object",
        "properties": {
            "source_profile": {"type": "string", "minLength": 1},
            "domains": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "required_names": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        },
        "required": ["source_profile", "domains", "required_names"],
        "additionalProperties": False,
    },
}


from tools.registry import registry
registry.register(
    name="browser_import_cookies", toolset="browser", schema=IMPORT_COOKIES_SCHEMA,
    handler=_handle_import, check_fn=_check_import_available, emoji="🔐",
)
