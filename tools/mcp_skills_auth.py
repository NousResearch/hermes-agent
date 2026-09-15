"""Private Skills eligibility bound to stored and adopted OAuth authority."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home
from tools.mcp_oauth import _safe_filename
from tools.mcp_skills_fs import secure_read_json, validate_managed_root


def _without_expiry(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key not in ("expires_in", "expires_at")}


def authorization_context(server: str, home: Path | str | None = None, *,
                          oauth_provider: Any = None, require_oauth: bool = False) -> str:
    """Digest actual authority; an opaque refreshed token cannot prove continuity.

    Rotation conservatively invalidates held Skills, but expiry bookkeeping does
    not. Read disk directly: cached delivery has no HTTP auth flow/disk watcher.
    For a connected OAuth source, also prove that the exact transport provider
    adopted this state. Missing files do not clear an initialized SDK provider.
    """
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", server) is None:
        raise ValueError("MCP skill authorization context has an invalid server label")
    root = validate_managed_root(Path(home or get_hermes_home()).expanduser()).absolute()
    state: dict[str, Any] = {"profile": str(root), "server": server}
    for suffix in (".json", ".client.json", ".meta.json"):
        try:
            payload = secure_read_json(root, Path("mcp-tokens") / f"{_safe_filename(server)}{suffix}")
        except FileNotFoundError:
            payload = None
        except (OSError, ValueError, TypeError):
            raise RuntimeError("MCP skill authorization context is unavailable; reconnect or start a new session") from None
        if payload is not None:
            if not isinstance(payload, dict):
                raise RuntimeError("MCP skill authorization context is invalid; reconnect or start a new session")
            if suffix == ".json":
                payload = _without_expiry(payload)
        state[suffix] = payload
    if require_oauth:
        context = getattr(oauth_provider, "context", None)
        adopted = {}
        for suffix, attr in ((".json", "current_tokens"), (".client.json", "client_info"), (".meta.json", "oauth_metadata")):
            model = getattr(context, attr, None)
            payload = model.model_dump(mode="json", exclude_none=True) if model is not None else None
            adopted[suffix] = _without_expiry(payload) if suffix == ".json" and payload is not None else payload
        if (not state[".json"] or not adopted[".json"]
                or any(state[key] != value for key, value in adopted.items())
                or Path(getattr(oauth_provider, "_hermes_home", "") or ".").absolute() != root):
            raise RuntimeError("MCP skill authorization context is not adopted by the connected OAuth source; reconnect")
    raw = json.dumps(state, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def source_authorization_context(server: str, source: Any, home=None) -> str:
    provider = getattr(source, "_oauth_provider", None)
    config = getattr(source, "_config", {}) or {}
    return authorization_context(server, home, oauth_provider=provider,
                                 require_oauth=provider is not None or config.get("auth") == "oauth")


def require_authorization_context(server: str, expected: str | None, home=None) -> None:
    if not expected or authorization_context(server, home) != expected:
        raise RuntimeError("MCP skill authorization context changed or is unbound; start a new session after reconnect")
