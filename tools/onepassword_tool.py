"""1Password in-session tools for Hermes Agent.

Uses the ``onepassword-sdk`` Python SDK for daemon-free vault access.
Requires ``OP_SERVICE_ACCOUNT_TOKEN`` in the environment.

Registered tools:
  - onepassword_list_vaults    — list all accessible vaults
  - onepassword_list_items     — list items in a vault
  - onepassword_get_item       — get full item details (including field values)
  - onepassword_resolve_field  — resolve op://vault/item/field shorthand

All tools share the same disk cache as the startup secret source
(``agent/secret_sources/onepassword.py``), so a cold start that
fetches vaults for injection also warms the tool cache and vice versa.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tools.registry import registry, tool_error, tool_result

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_TTL = 300
_DISK_CACHE_BASENAME = "onepassword_cache.json"

# ---------------------------------------------------------------------------
# Lazy SDK client
# ---------------------------------------------------------------------------

_ONEPASSWORD_CLIENT = None


async def _get_client():
    """Return an authenticated 1Password SDK client, or None if unavailable."""
    global _ONEPASSWORD_CLIENT
    if _ONEPASSWORD_CLIENT is not None:
        return _ONEPASSWORD_CLIENT
    try:
        from onepassword import Client
    except ImportError:
        return None

    token = os.environ.get("OP_SERVICE_ACCOUNT_TOKEN", "")
    if not token:
        return None

    _ONEPASSWORD_CLIENT = await Client.authenticate(
        auth=token,
        integration_name="Hermes Agent",
        integration_version="1.0.0",
    )
    return _ONEPASSWORD_CLIENT


def check_requirements() -> bool:
    """Tool is available when the SDK is installed and the token is set."""
    try:
        import onepassword  # noqa: F401
    except ImportError:
        return False
    return bool(os.environ.get("OP_SERVICE_ACCOUNT_TOKEN"))


# ---------------------------------------------------------------------------
# Token / cache-key helpers
# ---------------------------------------------------------------------------


def _token_fingerprint(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]


def _token_fp_from_env() -> str:
    token = os.environ.get("OP_SERVICE_ACCOUNT_TOKEN", "")
    return _token_fingerprint(token) if token else "unknown"


# ---------------------------------------------------------------------------
# In-process cache (L1 — instant hits within one agent session)
# ---------------------------------------------------------------------------

_TOOL_CACHE: Dict[str, Any] = {}
_TOOL_EXPIRY: Dict[str, float] = {}


def _get_hermes_home() -> Path:
    from hermes_constants import get_hermes_home as _ghh
    return _ghh()


def _disk_cache_path() -> Path:
    return _get_hermes_home() / "cache" / _DISK_CACHE_BASENAME


def _read_raw_disk_cache() -> dict:
    """Read the entire disk cache JSON.  Returns empty dict on any error."""
    path = _disk_cache_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return payload
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def _write_raw_disk_cache(payload: dict) -> None:
    """Atomically write the disk cache JSON.  Best-effort."""
    import tempfile

    path = _disk_cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            prefix=".onepassword_cache_", suffix=".tmp", dir=str(path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            os.chmod(tmp, 0o600)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except OSError:
        pass


def _tool_cache_get(cache_key: str, ttl: float = _DEFAULT_TTL) -> Optional[Any]:
    """L1 (in-process) then L2 (disk).  Returns None if expired or missing."""
    now = time.time()

    # L1: in-process
    if cache_key in _TOOL_EXPIRY and now < _TOOL_EXPIRY[cache_key]:
        return _TOOL_CACHE.get(cache_key)

    # L2: disk
    disk = _read_raw_disk_cache()
    entry = disk.get(cache_key)
    if entry is None:
        return None
    fetched_at = entry.get("fetched_at", 0)
    if now - fetched_at >= ttl:
        return None
    value = entry.get("value")
    # Promote to L1
    _TOOL_CACHE[cache_key] = value
    _TOOL_EXPIRY[cache_key] = now + ttl * 0.9
    return value


def _tool_cache_set(cache_key: str, value: Any, ttl: float = _DEFAULT_TTL) -> None:
    """Set a value in L1 (in-process) and L2 (disk)."""
    now = time.time()
    _TOOL_CACHE[cache_key] = value
    _TOOL_EXPIRY[cache_key] = now + ttl * 0.9

    disk = _read_raw_disk_cache()
    disk[cache_key] = {"value": value, "fetched_at": now}
    _write_raw_disk_cache(disk)


# ---------------------------------------------------------------------------
# Cache key namespaces
# ---------------------------------------------------------------------------


def _vaults_cache_key(fp: str) -> str:
    return f"tools:{fp}:vaults"


def _items_cache_key(fp: str, vault_id: str) -> str:
    return f"tools:{fp}:items:{vault_id}"


def _item_cache_key(fp: str, vault_id: str, item_id: str) -> str:
    return f"tools:{fp}:item:{vault_id}:{item_id}"


def _vault_contents_key(fp: str, vault_id: str) -> str:
    """Key for the full vault contents cache (ALL items with ALL fields).

    This is the most important cache entry for rate-limit avoidance —
    it holds a dict mapping item_title → {field_title: value} so
    ``onepassword_resolve_field`` can resolve in O(1) with zero API
    calls when the cache is warm.
    """
    return f"tools:{fp}:vault_contents:{vault_id}"


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def onepassword_list_vaults(task_id: str = None) -> str:
    """List all vaults the service account can access."""
    try:
        result = asyncio.run(_list_vaults_async())
        return result
    except Exception as exc:
        return tool_error(str(exc))


async def _list_vaults_async() -> str:
    client = await _get_client()
    if client is None:
        return tool_error(
            "1Password SDK not available. Install with: "
            "uv pip install --python <hermes-python> onepassword-sdk"
        )

    fp = _token_fp_from_env()
    cache_key = _vaults_cache_key(fp)
    cached = _tool_cache_get(cache_key)
    if cached is not None:
        return tool_result({"count": len(cached), "vaults": cached, "cached": True})

    from onepassword.types import VaultGetParams

    vaults = await client.vaults.list()
    vault_data = []
    for v in vaults:
        full = await client.vaults.get(vault_id=v.id, vault_params=VaultGetParams())
        vault_data.append({
            "id": full.id,
            "name": full.title,
            "description": full.description,
            "item_count": full.active_item_count,
        })

    _tool_cache_set(cache_key, vault_data)
    return tool_result({"count": len(vault_data), "vaults": vault_data, "cached": False})


def onepassword_list_items(
    vault_id: str = "",
    vault_name: str = "",
    task_id: str = None,
) -> str:
    """List items in a vault, specified by ID or name."""
    try:
        result = asyncio.run(_list_items_async(vault_id, vault_name))
        return result
    except Exception as exc:
        return tool_error(str(exc))


async def _list_items_async(vault_id: str, vault_name: str) -> str:
    client = await _get_client()
    if client is None:
        return tool_error("1Password SDK not available.")

    fp = _token_fp_from_env()

    resolved = await _resolve_vault(client, vault_id, vault_name, fp)
    if resolved is None:
        return tool_error(f"Vault not found: id={vault_id}, name={vault_name}")
    target_id, target_name = resolved

    cache_key = _items_cache_key(fp, target_id)
    cached = _tool_cache_get(cache_key)
    if cached is not None:
        return tool_result({
            "vault_id": target_id,
            "vault_name": target_name,
            "count": len(cached),
            "items": cached,
            "cached": True,
        })

    items = await client.items.list(vault_id=target_id)
    item_data = []
    for item in items:
        item_data.append({
            "id": item.id,
            "title": item.title,
            "category": str(item.category),
        })

    _tool_cache_set(cache_key, item_data)
    return tool_result({
        "vault_id": target_id,
        "vault_name": target_name,
        "count": len(item_data),
        "items": item_data,
        "cached": False,
    })


def onepassword_get_item(
    vault_id: str = "",
    vault_name: str = "",
    item_id: str = "",
    task_id: str = None,
) -> str:
    """Get full item details including all field values.

    Use onepassword_list_items first to find the item_id.
    Use onepassword_resolve_field for targeted field resolution.
    """
    if not item_id:
        return tool_error(
            "item_id is required. Use onepassword_list_items first "
            "to find the item ID."
        )
    try:
        result = asyncio.run(_get_item_async(vault_id, vault_name, item_id))
        return result
    except Exception as exc:
        return tool_error(str(exc))


async def _get_item_async(vault_id: str, vault_name: str, item_id: str) -> str:
    client = await _get_client()
    if client is None:
        return tool_error("1Password SDK not available.")

    fp = _token_fp_from_env()

    resolved = await _resolve_vault(client, vault_id, vault_name, fp)
    if resolved is None:
        return tool_error(f"Vault not found: id={vault_id}, name={vault_name}")
    target_id, target_name = resolved

    item_cache_key = _item_cache_key(fp, target_id, item_id)
    cached = _tool_cache_get(item_cache_key)
    if cached is not None:
        return tool_result({**cached, "cached": True})

    try:
        item = await client.items.get(vault_id=target_id, item_id=item_id)
    except Exception as exc:
        return tool_error(f"Item not found: {exc}")

    fields_data = []
    for field in item.fields:
        val = field.value or ""
        ftype = str(field.field_type) if field.field_type else "Text"
        fields_data.append({
            "id": field.id,
            "label": field.title or field.id,
            "type": ftype,
            "value": val,
            "section_id": field.section_id or "",
        })

    sections_data = []
    if item.sections:
        for section in item.sections:
            sections_data.append({"id": section.id, "title": section.title})

    item_data = {
        "id": item.id,
        "title": item.title,
        "category": str(item.category),
        "vault_id": item.vault_id,
        "version": item.version,
        "notes": item.notes or "",
        "tags": list(item.tags) if item.tags else [],
        "fields": fields_data,
        "sections": sections_data,
        "vault_name": target_name,
    }

    _tool_cache_set(item_cache_key, item_data)
    return tool_result({**item_data, "cached": False})


def onepassword_resolve_field(ref: str = "", task_id: str = None) -> str:
    """Resolve an op://vault/item/field reference to its value.

    Example: op://Personal/My API Key/credential

    Rate-limit optimisation: on first access to a vault, this fetches
    and caches ALL items+fields from that vault.  Subsequent resolves
    (within the TTL) are dictionary lookups — zero API calls.
    """
    if not ref:
        return tool_error("ref is required. Format: op://vault/item/field")

    if not ref.startswith("op://"):
        return tool_error("ref must start with op://")

    # Strip op:// prefix and match content
    inner = ref[5:]
    if not re.match(r"^[\w\s'().,-]+/[\w\s'().,-]+/[\w\s'().,-]+$", inner):
        return tool_error(
            "ref must be op://vault/item/field with alphanumeric names"
        )

    try:
        result = asyncio.run(_resolve_field_async(ref))
        return result
    except Exception as exc:
        return tool_error(str(exc))


async def _resolve_field_async(ref: str) -> str:
    client = await _get_client()
    if client is None:
        return tool_error("1Password SDK not available.")

    parts = ref[5:].split("/")
    if len(parts) < 3:
        return tool_error(
            f"ref must be op://vault/item/field (saw {len(parts)} parts)"
        )

    vault_name, item_name, field_label = parts[0], parts[1], parts[2]
    fp = _token_fp_from_env()

    # Step 1: resolve vault name → ID (cached)
    resolved = await _resolve_vault(client, "", vault_name, fp)
    if resolved is None:
        return tool_error(f"Vault '{vault_name}' not found")
    target_id, target_name = resolved

    # Step 2: use or populate the vault-contents cache
    contents_cache_key = _vault_contents_key(fp, target_id)
    vault_contents = _tool_cache_get(contents_cache_key)

    if vault_contents is None:
        vault_contents = await _build_vault_contents(client, target_id)
        _tool_cache_set(contents_cache_key, vault_contents)

    # Step 3: lookup in cached contents
    item_lower = item_name.lower()
    item_data = vault_contents.get(item_lower)
    if item_data is None:
        return tool_error(
            f"Item '{item_name}' not found in vault '{vault_name}'. "
            f"Available items: {', '.join(sorted(vault_contents.keys()))}"
        )

    # Step 4: find the field by label (case-insensitive)
    field_lower = field_label.lower()
    for field_title, field_value in item_data.get("fields", {}).items():
        if field_title.lower() == field_lower:
            return tool_result({
                "ref": ref,
                "value": field_value,
                "cached": vault_contents is not None,
            })

    return tool_error(
        f"Field '{field_label}' not found in item '{item_name}'. "
        f"Available fields: {', '.join(sorted(item_data['fields'].keys()))}"
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


async def _resolve_vault(
    client,
    vault_id: str,
    vault_name: str,
    fp: str,
) -> Optional[Tuple[str, str]]:
    """Resolve a vault by ID or name.  Returns (vault_id, vault_name)."""
    vaults_cache_key = _vaults_cache_key(fp)
    cached_vaults = _tool_cache_get(vaults_cache_key)

    if cached_vaults is not None:
        if vault_id:
            for v in cached_vaults:
                if v["id"] == vault_id:
                    return (v["id"], v["name"])
        if vault_name:
            vn = vault_name.lower()
            for v in cached_vaults:
                if v["name"].lower() == vn or v["id"].lower() == vn:
                    return (v["id"], v["name"])

    from onepassword.types import VaultGetParams

    vaults = await client.vaults.list()
    vault_data = []
    for v in vaults:
        full = await client.vaults.get(vault_id=v.id, vault_params=VaultGetParams())
        vault_data.append({
            "id": full.id,
            "name": full.title,
            "description": full.description,
            "item_count": full.active_item_count,
        })
        if vault_id and full.id == vault_id:
            _tool_cache_set(vaults_cache_key, vault_data)
            return (full.id, full.title)
        if vault_name and (
            full.title.lower() == vault_name.lower()
            or full.id == vault_name
        ):
            _tool_cache_set(vaults_cache_key, vault_data)
            return (full.id, full.title)

    _tool_cache_set(vaults_cache_key, vault_data)
    return None


async def _build_vault_contents(client, vault_id: str) -> Dict[str, Any]:
    """Fetch ALL items+fields from a vault and return a lookup dict.

    Returns: {item_title_lowercase: {"title": str, "id": str, "fields": {field_title: value}}}
    """
    items = await client.items.list(vault_id=vault_id)
    vault_contents: Dict[str, Dict[str, Any]] = {}

    for item_overview in items:
        try:
            item = await client.items.get(vault_id=vault_id, item_id=item_overview.id)
        except Exception:
            continue

        fields: Dict[str, str] = {}
        for field in item.fields:
            label = field.title or field.id or ""
            val = field.value or ""
            fields[label] = val

        vault_contents[item.title.lower()] = {
            "title": item.title,
            "id": item.id,
            "fields": fields,
        }

    return vault_contents


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

_LIST_VAULTS_SCHEMA = {
    "name": "onepassword_list_vaults",
    "description": (
        "List all 1Password vaults accessible via the service account. "
        "Returns vault names and IDs."
    ),
    "parameters": {"type": "object", "properties": {}},
}

_LIST_ITEMS_SCHEMA = {
    "name": "onepassword_list_items",
    "description": (
        "List items in a 1Password vault by ID or name. "
        "Use onepassword_list_vaults first to discover vaults."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "vault_id": {
                "type": "string",
                "description": (
                    "Vault ID (from onepassword_list_vaults). "
                    "Optional if vault_name is provided."
                ),
            },
            "vault_name": {
                "type": "string",
                "description": (
                    "Vault name (e.g. 'Private'). "
                    "Optional if vault_id is provided."
                ),
            },
        },
    },
}

_GET_ITEM_SCHEMA = {
    "name": "onepassword_get_item",
    "description": (
        "Get full item details from a 1Password vault, including all "
        "field values (credential tokens, API keys, passwords, etc.). "
        "Use onepassword_list_items first to find the item_id."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "vault_id": {
                "type": "string",
                "description": (
                    "Vault ID (from onepassword_list_vaults). "
                    "Optional if vault_name is provided."
                ),
            },
            "vault_name": {
                "type": "string",
                "description": (
                    "Vault name (e.g. 'Private'). "
                    "Optional if vault_id is provided."
                ),
            },
            "item_id": {
                "type": "string",
                "description": (
                    "Item ID (from onepassword_list_items). Required."
                ),
            },
        },
        "required": ["item_id"],
    },
}

_RESOLVE_FIELD_SCHEMA = {
    "name": "onepassword_resolve_field",
    "description": (
        "Resolve an op://vault/item/field reference to its actual value. "
        "Example: op://Personal/My API Key/credential returns the "
        "credential field value."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "ref": {
                "type": "string",
                "description": (
                    "The op:// reference to resolve. "
                    "Format: op://vault_name/item_name/field_label"
                ),
            },
        },
        "required": ["ref"],
    },
}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

registry.register(
    name="onepassword_list_vaults",
    toolset="onepassword",
    schema=_LIST_VAULTS_SCHEMA,
    handler=lambda args, **kw: onepassword_list_vaults(task_id=kw.get("task_id")),
    check_fn=check_requirements,
    requires_env=["OP_SERVICE_ACCOUNT_TOKEN"],
)

registry.register(
    name="onepassword_list_items",
    toolset="onepassword",
    schema=_LIST_ITEMS_SCHEMA,
    handler=lambda args, **kw: onepassword_list_items(
        vault_id=args.get("vault_id", ""),
        vault_name=args.get("vault_name", ""),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_requirements,
    requires_env=["OP_SERVICE_ACCOUNT_TOKEN"],
)

registry.register(
    name="onepassword_get_item",
    toolset="onepassword",
    schema=_GET_ITEM_SCHEMA,
    handler=lambda args, **kw: onepassword_get_item(
        vault_id=args.get("vault_id", ""),
        vault_name=args.get("vault_name", ""),
        item_id=args.get("item_id", ""),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_requirements,
    requires_env=["OP_SERVICE_ACCOUNT_TOKEN"],
)

registry.register(
    name="onepassword_resolve_field",
    toolset="onepassword",
    schema=_RESOLVE_FIELD_SCHEMA,
    handler=lambda args, **kw: onepassword_resolve_field(
        ref=args.get("ref", ""),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_requirements,
    requires_env=["OP_SERVICE_ACCOUNT_TOKEN"],
)
