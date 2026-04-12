"""Feishu identity/chat lookup tool with official API first, local fallback.

Primary resolution uses the official Feishu APIs for:
- user identity lookup by open_id/user_id/union_id
- chat lookup by chat_id
- chat search by name/query
- chat member enumeration

When official API credentials are unavailable, or an authoritative lookup does
not succeed, the tool falls back to Hermes' locally observed session metadata
from ``~/.hermes/sessions/sessions.json``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_SESSION_KEY_ACCOUNT_RE = re.compile(r":feishu\[([^\]]+)\]:")

_ID_PREFIX_MAP = {
    "ou_": "open_id",
    "on_": "union_id",
    "u_": "user_id",
}

FEISHU_ID_SCHEMA = {
    "name": "feishu_id",
    "description": (
        "Inspect Feishu identities, chats, and session routes for Hermes. "
        "Uses official Feishu APIs when credentials are configured, and falls back "
        "to Hermes' local observed session metadata when needed. Supports chat_id/name "
        "lookup, open_id/user_id/union_id lookup, group member lookup, and session-key resolution."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "resolve_user",
                    "whois_user",
                    "search_chats",
                    "whois_chat",
                    "members",
                    "session_lookup",
                    "my_chats",
                ],
                "description": "Lookup action to perform.",
            },
            "query": {
                "type": "string",
                "description": "Free-text query for name-based lookup (user or chat).",
            },
            "chat_id": {
                "type": "string",
                "description": "Feishu chat ID such as oc_xxx.",
            },
            "session_key": {
                "type": "string",
                "description": "Hermes gateway session key to resolve.",
            },
            "session_id": {
                "type": "string",
                "description": "Hermes runtime session_id to resolve.",
            },
            "open_id": {
                "type": "string",
                "description": "Feishu open_id (ou_xxx).",
            },
            "user_id": {
                "type": "string",
                "description": "Feishu tenant-scoped user_id (u_xxx).",
            },
            "union_id": {
                "type": "string",
                "description": "Feishu union_id (on_xxx).",
            },
            "account_id": {
                "type": "string",
                "description": "Preferred account scope for the query. Defaults to the current Feishu account when applicable.",
            },
            "as_account_id": {
                "type": "string",
                "description": "Explicit target account for cross-account lookup. From Feishu sessions this is allowed only for configured admins.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return. Default 10, max 50.",
            },
        },
        "required": ["action"],
    },
}


def _sessions_path() -> Path:
    return get_hermes_home() / "sessions" / "sessions.json"


def _check_feishu_id() -> bool:
    from gateway.session_context import get_session_env

    platform = get_session_env("HERMES_SESSION_PLATFORM", "").strip().lower()
    if platform in {"", "local", "feishu"} and _sessions_path().exists():
        return True

    try:
        from gateway.config import Platform, load_gateway_config

        config = load_gateway_config()
        return Platform.FEISHU in config.get_connected_platforms()
    except Exception:
        return _sessions_path().exists()


def _get_config_callable(config, name):
    try:
        instance_attr = vars(config).get(name)
    except Exception:
        instance_attr = None
    if callable(instance_attr):
        return instance_attr

    class_attr = getattr(type(config), name, None)
    if callable(class_attr):
        return getattr(config, name)
    return None


def _load_gateway_config_safe():
    try:
        from gateway.config import load_gateway_config

        return load_gateway_config()
    except Exception as exc:
        logger.debug("feishu_id: failed to load gateway config: %s", exc)
        return None


def _resolve_default_account_id(config) -> Optional[str]:
    if config is None:
        return None
    try:
        from gateway.config import Platform

        getter = _get_config_callable(config, "get_default_account_id")
        if getter:
            return getter(Platform.FEISHU)
    except Exception:
        pass
    return None


def _resolve_platform_config(config, account_id: Optional[str] = None):
    if config is None:
        return None
    try:
        from gateway.config import Platform

        getter = _get_config_callable(config, "get_platform_config")
        if getter:
            try:
                return getter(Platform.FEISHU, account_id=account_id)
            except TypeError:
                return getter(Platform.FEISHU)

        platforms = getattr(config, "platforms", {}) or {}
        return platforms.get(Platform.FEISHU)
    except Exception:
        return None


def _iter_feishu_accounts(config) -> List[str]:
    if config is None:
        return []
    try:
        from gateway.config import Platform

        iterator = _get_config_callable(config, "iter_platform_account_configs")
        if iterator:
            return [
                account_id
                for account_id, _platform_config in iterator(Platform.FEISHU)
                if str(account_id or "").strip()
            ]
    except Exception:
        pass
    return []


def _extract_admin_ids(account_config) -> set[str]:
    ids: set[str] = set()
    if not account_config:
        return ids
    extra = getattr(account_config, "extra", {}) or {}
    raw_admins = extra.get("admins", []) or []
    for item in raw_admins:
        if isinstance(item, str):
            text = item.strip()
            if text:
                ids.add(text)
            continue
        if isinstance(item, dict):
            for field in ("user_id", "open_id", "union_id"):
                value = str(item.get(field) or "").strip()
                if value:
                    ids.add(value)
    return ids


def _classify_id(value: Optional[str]) -> Optional[Tuple[str, str]]:
    text = str(value or "").strip()
    if not text:
        return None
    for prefix, id_type in _ID_PREFIX_MAP.items():
        if text.startswith(prefix):
            return id_type, text
    return None, text


def _normalize_text(value: Optional[str]) -> str:
    return str(value or "").strip().lower()


def _extract_account_from_session_key(session_key: str) -> Optional[str]:
    match = _SESSION_KEY_ACCOUNT_RE.search(session_key or "")
    if match:
        return match.group(1)
    return None


def _safe_sort_strings(values) -> List[str]:
    return sorted({str(v).strip() for v in values if str(v).strip()})


def _read_session_entries() -> List[Dict[str, Any]]:
    from gateway.session import SessionEntry

    path = _sessions_path()
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("feishu_id: failed to read %s: %s", path, exc)
        return []

    entries: List[Dict[str, Any]] = []
    for session_key, entry_data in raw.items():
        try:
            entry = SessionEntry.from_dict(entry_data)
        except Exception:
            logger.debug("feishu_id: failed to parse session entry %s", session_key, exc_info=True)
            continue
        origin = entry.origin
        if not origin or getattr(origin.platform, "value", None) != "feishu":
            continue
        entries.append(
            {
                "session_key": session_key,
                "session_id": entry.session_id,
                "created_at": entry.created_at.isoformat(),
                "updated_at": entry.updated_at.isoformat(),
                "origin": origin.to_dict(),
            }
        )
    return entries


def _ensure_user_record(
    users_by_key: Dict[str, Dict[str, Any]],
    id_lookup: Dict[Tuple[Optional[str], str, str], str],
    *,
    account_id: Optional[str],
    identifiers: Dict[str, str],
    name: Optional[str],
) -> Optional[str]:
    if not identifiers and not str(name or "").strip():
        return None

    existing_keys = {
        id_lookup[(account_id, id_type, id_value)]
        for id_type, id_value in identifiers.items()
        if (account_id, id_type, id_value) in id_lookup
    }

    if existing_keys:
        record_key = sorted(existing_keys)[0]
    else:
        seed = identifiers.get("union_id") or identifiers.get("user_id") or identifiers.get("open_id") or name or "unknown"
        record_key = f"{account_id or '_'}:{seed}"
        users_by_key.setdefault(
            record_key,
            {
                "account_id": account_id,
                "ids": {"open_id": set(), "user_id": set(), "union_id": set(), "unknown": set()},
                "names": set(),
                "chat_ids": set(),
                "session_keys": set(),
                "session_ids": set(),
            },
        )

    record = users_by_key[record_key]
    if name:
        record["names"].add(name)
    for id_type, id_value in identifiers.items():
        bucket = id_type if id_type in record["ids"] else "unknown"
        record["ids"][bucket].add(id_value)
        id_lookup[(account_id, id_type, id_value)] = record_key

    # Merge split observations if the same person was first seen under different IDs.
    for other_key in sorted(existing_keys - {record_key}):
        other = users_by_key.pop(other_key, None)
        if not other:
            continue
        for bucket, values in other["ids"].items():
            record["ids"][bucket].update(values)
            for id_value in values:
                id_lookup[(account_id, bucket, id_value)] = record_key
        record["names"].update(other["names"])
        record["chat_ids"].update(other["chat_ids"])
        record["session_keys"].update(other["session_keys"])
        record["session_ids"].update(other["session_ids"])

    return record_key


def _build_observed_index(config) -> Dict[str, Any]:
    default_account_id = _resolve_default_account_id(config)
    home_channels = {}
    for account_id in _iter_feishu_accounts(config):
        account_config = _resolve_platform_config(config, account_id=account_id)
        home = getattr(account_config, "home_channel", None)
        if home and getattr(home, "chat_id", None):
            home_channels[(account_id, str(home.chat_id))] = getattr(home, "name", None)

    users_by_key: Dict[str, Dict[str, Any]] = {}
    user_id_lookup: Dict[Tuple[Optional[str], str, str], str] = {}
    chats: Dict[Tuple[Optional[str], str], Dict[str, Any]] = {}
    sessions_by_key: Dict[str, Dict[str, Any]] = {}

    for item in _read_session_entries():
        origin = item["origin"]
        session_key = item["session_key"]
        raw_account_id = str(origin.get("account_id") or "").strip() or None
        inferred_account_id = False
        account_id = raw_account_id or _extract_account_from_session_key(session_key)
        if not account_id and default_account_id:
            account_id = default_account_id
            inferred_account_id = True

        identifiers: Dict[str, str] = {}
        for raw_value in (origin.get("user_id"), origin.get("user_id_alt")):
            classified = _classify_id(raw_value)
            if not classified:
                continue
            id_type, id_value = classified
            bucket = id_type or "unknown"
            identifiers[bucket] = id_value

        user_key = _ensure_user_record(
            users_by_key,
            user_id_lookup,
            account_id=account_id,
            identifiers=identifiers,
            name=origin.get("user_name"),
        )
        if user_key:
            user_record = users_by_key[user_key]
            user_record["chat_ids"].add(str(origin.get("chat_id") or ""))
            user_record["session_keys"].add(session_key)
            user_record["session_ids"].add(item["session_id"])

        chat_id = str(origin.get("chat_id") or "").strip()
        if not chat_id:
            continue

        chat_key = (account_id, chat_id)
        chat_record = chats.setdefault(
            chat_key,
            {
                "account_id": account_id,
                "chat_id": chat_id,
                "chat_types": set(),
                "names": set(),
                "session_keys": set(),
                "session_ids": set(),
                "observed_member_keys": set(),
                "updated_at": item["updated_at"],
                "home_channel_name": home_channels.get((account_id, chat_id)),
                "account_id_inferred": inferred_account_id,
            },
        )
        chat_record["chat_types"].add(str(origin.get("chat_type") or "dm"))
        if origin.get("chat_name"):
            chat_record["names"].add(str(origin["chat_name"]))
        if origin.get("user_name") and str(origin.get("chat_type") or "") == "dm":
            chat_record["names"].add(str(origin["user_name"]))
        chat_record["session_keys"].add(session_key)
        chat_record["session_ids"].add(item["session_id"])
        if item["updated_at"] > chat_record["updated_at"]:
            chat_record["updated_at"] = item["updated_at"]
        if user_key:
            chat_record["observed_member_keys"].add(user_key)

        sessions_by_key[session_key] = {
            "session_key": session_key,
            "session_id": item["session_id"],
            "account_id": account_id,
            "account_id_inferred": inferred_account_id,
            "chat_id": chat_id,
            "chat_name": origin.get("chat_name"),
            "chat_type": origin.get("chat_type"),
            "user_name": origin.get("user_name"),
            "identifiers": identifiers,
            "created_at": item["created_at"],
            "updated_at": item["updated_at"],
        }

    users = []
    for record in users_by_key.values():
        users.append(
            {
                "account_id": record["account_id"],
                "names": _safe_sort_strings(record["names"]),
                "open_ids": _safe_sort_strings(record["ids"]["open_id"]),
                "user_ids": _safe_sort_strings(record["ids"]["user_id"]),
                "union_ids": _safe_sort_strings(record["ids"]["union_id"]),
                "unknown_ids": _safe_sort_strings(record["ids"]["unknown"]),
                "chat_ids": _safe_sort_strings(record["chat_ids"]),
                "session_keys": _safe_sort_strings(record["session_keys"]),
                "session_ids": _safe_sort_strings(record["session_ids"]),
            }
        )

    normalized_chats = []
    for chat_record in chats.values():
        members = []
        for user_key in sorted(chat_record["observed_member_keys"]):
            user_record = users_by_key.get(user_key)
            if not user_record:
                continue
            members.append(
                {
                    "names": _safe_sort_strings(user_record["names"]),
                    "open_ids": _safe_sort_strings(user_record["ids"]["open_id"]),
                    "user_ids": _safe_sort_strings(user_record["ids"]["user_id"]),
                    "union_ids": _safe_sort_strings(user_record["ids"]["union_id"]),
                }
            )
        normalized_chats.append(
            {
                "account_id": chat_record["account_id"],
                "account_id_inferred": chat_record["account_id_inferred"],
                "chat_id": chat_record["chat_id"],
                "chat_types": _safe_sort_strings(chat_record["chat_types"]),
                "names": _safe_sort_strings(chat_record["names"]),
                "session_keys": _safe_sort_strings(chat_record["session_keys"]),
                "session_ids": _safe_sort_strings(chat_record["session_ids"]),
                "updated_at": chat_record["updated_at"],
                "is_home_channel": bool(chat_record["home_channel_name"]),
                "home_channel_name": chat_record["home_channel_name"],
                "observed_members": members,
            }
        )

    sessions = list(sessions_by_key.values())
    sessions.sort(key=lambda item: item["updated_at"], reverse=True)
    users.sort(key=lambda item: ((item["account_id"] or ""), item["names"][0] if item["names"] else item["open_ids"][0] if item["open_ids"] else ""))
    normalized_chats.sort(key=lambda item: ((item["account_id"] or ""), item["chat_id"]))

    return {
        "users": users,
        "chats": normalized_chats,
        "sessions": sessions,
    }


def _resolve_secret(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if not text.startswith("env:"):
        return text
    return os.getenv(text[4:].strip(), "").strip()


def _candidate_account_ids(config, account_id: Optional[str]) -> List[Optional[str]]:
    if account_id is not None:
        return [account_id]

    account_ids = _iter_feishu_accounts(config)
    if account_ids:
        return list(account_ids)

    if _resolve_platform_config(config) is not None:
        return [None]
    return []


def _home_channel_name(config, account_id: Optional[str], chat_id: str) -> Optional[str]:
    account_config = _resolve_platform_config(config, account_id=account_id)
    home = getattr(account_config, "home_channel", None) if account_config else None
    if not home:
        return None
    if str(getattr(home, "chat_id", "") or "").strip() != str(chat_id or "").strip():
        return None
    return str(getattr(home, "name", "") or "").strip() or None


def _response_succeeded(response: Any) -> bool:
    return bool(response and getattr(response, "success", lambda: False)())


def _response_error(response: Any, default_message: str) -> str:
    if response is None:
        return default_message
    code = getattr(response, "code", "unknown")
    msg = getattr(response, "msg", default_message)
    return f"[{code}] {msg}"


def _merge_lists(*values: List[str]) -> List[str]:
    merged: List[str] = []
    for items in values:
        merged.extend(items or [])
    return _safe_sort_strings(merged)


def _map_chat_type(raw_chat_type: Optional[str]) -> str:
    raw = str(raw_chat_type or "").strip().lower()
    if raw in {"p2p", "single"}:
        return "dm"
    if raw in {"group", "topic"}:
        return "group"
    return raw or "dm"


def _observed_chat_for_merge(index: Dict[str, Any], account_id: Optional[str], chat_id: str) -> Optional[Dict[str, Any]]:
    return _find_chat(index, account_id, chat_id)


def _observed_user_for_merge(
    index: Dict[str, Any],
    account_id: Optional[str],
    *,
    open_id: Optional[str] = None,
    user_id: Optional[str] = None,
    union_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    identifiers = {}
    if open_id:
        identifiers["open_id"] = open_id
    if user_id:
        identifiers["user_id"] = user_id
    if union_id:
        identifiers["union_id"] = union_id
    return _find_user_by_ids(index, account_id, identifiers)


def _serialize_official_user(user: Any, account_id: Optional[str]) -> Dict[str, Any]:
    return {
        "account_id": account_id,
        "names": _safe_sort_strings(
            [
                getattr(user, "name", None),
                getattr(user, "nickname", None),
                getattr(user, "en_name", None),
            ]
        ),
        "open_ids": _safe_sort_strings([getattr(user, "open_id", None)]),
        "user_ids": _safe_sort_strings([getattr(user, "user_id", None)]),
        "union_ids": _safe_sort_strings([getattr(user, "union_id", None)]),
        "unknown_ids": [],
        "chat_ids": [],
        "session_keys": [],
        "session_ids": [],
        "email": str(getattr(user, "email", "") or "").strip() or None,
        "enterprise_email": str(getattr(user, "enterprise_email", "") or "").strip() or None,
        "mobile": str(getattr(user, "mobile", "") or "").strip() or None,
        "job_title": str(getattr(user, "job_title", "") or "").strip() or None,
        "department_ids": _safe_sort_strings(getattr(user, "department_ids", None) or []),
    }


def _merge_user_payload(official_user: Dict[str, Any], observed_user: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not observed_user:
        return official_user

    merged = dict(official_user)
    for key in (
        "names",
        "open_ids",
        "user_ids",
        "union_ids",
        "unknown_ids",
        "chat_ids",
        "session_keys",
        "session_ids",
    ):
        merged[key] = _merge_lists(official_user.get(key, []), observed_user.get(key, []))
    return merged


def _serialize_official_chat(
    chat_id: str,
    data: Any,
    account_id: Optional[str],
    *,
    home_channel_name: Optional[str],
) -> Dict[str, Any]:
    raw_chat_type = (
        str(getattr(data, "chat_type", "") or "").strip().lower()
        or str(getattr(data, "type", "") or "").strip().lower()
    )

    user_count_raw = str(getattr(data, "user_count", "") or "").strip()
    bot_count_raw = str(getattr(data, "bot_count", "") or "").strip()

    return {
        "account_id": account_id,
        "account_id_inferred": False,
        "chat_id": chat_id,
        "chat_types": _safe_sort_strings([_map_chat_type(raw_chat_type)]),
        "raw_type": raw_chat_type or None,
        "names": _safe_sort_strings([getattr(data, "name", None)]),
        "description": str(getattr(data, "description", "") or "").strip() or None,
        "owner_id": str(
            getattr(data, "owner_id", None)
            or getattr(data, "owner_user_id", None)
            or ""
        ).strip() or None,
        "owner_id_type": str(getattr(data, "owner_id_type", "") or "").strip() or None,
        "chat_status": str(getattr(data, "chat_status", "") or "").strip() or None,
        "external": getattr(data, "external", None),
        "tenant_key": str(getattr(data, "tenant_key", "") or "").strip() or None,
        "labels": _safe_sort_strings(getattr(data, "labels", None) or []),
        "member_count": int(user_count_raw) if user_count_raw.isdigit() else None,
        "bot_count": int(bot_count_raw) if bot_count_raw.isdigit() else None,
        "session_keys": [],
        "session_ids": [],
        "updated_at": None,
        "is_home_channel": bool(home_channel_name),
        "home_channel_name": home_channel_name,
        "observed_members": [],
    }


def _merge_chat_payload(official_chat: Dict[str, Any], observed_chat: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not observed_chat:
        return official_chat

    merged = dict(official_chat)
    merged["chat_types"] = _merge_lists(official_chat.get("chat_types", []), observed_chat.get("chat_types", []))
    merged["names"] = _merge_lists(official_chat.get("names", []), observed_chat.get("names", []))
    merged["session_keys"] = _merge_lists(official_chat.get("session_keys", []), observed_chat.get("session_keys", []))
    merged["session_ids"] = _merge_lists(official_chat.get("session_ids", []), observed_chat.get("session_ids", []))
    merged["observed_members"] = observed_chat.get("observed_members", [])
    merged["updated_at"] = observed_chat.get("updated_at") or official_chat.get("updated_at")
    merged["account_id_inferred"] = bool(observed_chat.get("account_id_inferred"))
    merged["is_home_channel"] = bool(official_chat.get("is_home_channel") or observed_chat.get("is_home_channel"))
    merged["home_channel_name"] = (
        official_chat.get("home_channel_name")
        or observed_chat.get("home_channel_name")
    )
    if not merged.get("description"):
        merged["description"] = observed_chat.get("description")
    if not merged.get("chat_status"):
        merged["chat_status"] = observed_chat.get("chat_status")
    if merged.get("member_count") is None:
        observed_count = len(observed_chat.get("observed_members", []) or [])
        merged["member_count"] = observed_count or None
    return merged


def _build_official_client(config, account_id: Optional[str]) -> Tuple[Optional[Any], Optional[str]]:
    try:
        import lark_oapi as lark
        from lark_oapi.core.const import FEISHU_DOMAIN, LARK_DOMAIN
    except Exception as exc:
        return None, f"lark_oapi unavailable: {exc}"

    account_config = _resolve_platform_config(config, account_id=account_id)
    if account_config is None:
        return None, f"No Feishu config found for account_id={account_id!r}."

    extra = getattr(account_config, "extra", {}) or {}
    app_id = _resolve_secret(extra.get("app_id"))
    app_secret = _resolve_secret(extra.get("app_secret"))
    domain_name = str(extra.get("domain") or "feishu").strip().lower() or "feishu"
    if not app_id or not app_secret:
        return None, f"Feishu official API credentials missing for account_id={account_id!r}."

    domain = FEISHU_DOMAIN if domain_name != "lark" else LARK_DOMAIN
    client = (
        lark.Client.builder()
        .app_id(app_id)
        .app_secret(app_secret)
        .domain(domain)
        .log_level(lark.LogLevel.WARNING)
        .build()
    )
    return client, None


def _official_get_user(
    config,
    index: Dict[str, Any],
    account_id: Optional[str],
    identifiers: Dict[str, str],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not identifiers:
        return None, "Official Feishu user lookup requires an identifier."

    client, error = _build_official_client(config, account_id)
    if not client:
        return None, error

    try:
        from lark_oapi.api.contact.v3 import GetUserRequest
    except Exception as exc:
        return None, f"GetUserRequest unavailable: {exc}"

    for id_type in ("user_id", "open_id", "union_id"):
        id_value = identifiers.get(id_type)
        if not id_value:
            continue
        try:
            request = (
                GetUserRequest.builder()
                .user_id(id_value)
                .user_id_type(id_type)
                .department_id_type("open_department_id")
                .build()
            )
            response = client.contact.v3.user.get(request)
        except Exception as exc:
            return None, f"Official Feishu user lookup failed for account_id={account_id!r}: {exc}"
        if not _response_succeeded(response):
            continue
        user = getattr(getattr(response, "data", None), "user", None)
        if not user:
            continue
        payload = _serialize_official_user(user, account_id)
        observed = _observed_user_for_merge(
            index,
            account_id,
            open_id=payload["open_ids"][0] if payload["open_ids"] else None,
            user_id=payload["user_ids"][0] if payload["user_ids"] else None,
            union_id=payload["union_ids"][0] if payload["union_ids"] else None,
        )
        return _merge_user_payload(payload, observed), None

    return None, "No official Feishu user matched the provided identifiers."


def _official_get_chat(
    config,
    index: Dict[str, Any],
    account_id: Optional[str],
    chat_id: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    client, error = _build_official_client(config, account_id)
    if not client:
        return None, error

    try:
        from lark_oapi.api.im.v1 import GetChatRequest
    except Exception as exc:
        return None, f"GetChatRequest unavailable: {exc}"

    try:
        request = GetChatRequest.builder().chat_id(chat_id).user_id_type("open_id").build()
        response = client.im.v1.chat.get(request)
    except Exception as exc:
        return None, f"Official Feishu chat lookup failed for account_id={account_id!r}: {exc}"
    if not _response_succeeded(response):
        return None, _response_error(response, "Official Feishu chat lookup failed.")

    data = getattr(response, "data", None)
    payload = _serialize_official_chat(
        chat_id,
        data,
        account_id,
        home_channel_name=_home_channel_name(config, account_id, chat_id),
    )
    observed = _observed_chat_for_merge(index, account_id, chat_id)
    return _merge_chat_payload(payload, observed), None


def _official_get_members(
    config,
    index: Dict[str, Any],
    account_id: Optional[str],
    chat_id: str,
    limit: int,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    client, error = _build_official_client(config, account_id)
    if not client:
        return None, error

    try:
        from lark_oapi.api.contact.v3 import BatchUserRequest
        from lark_oapi.api.im.v1 import GetChatMembersRequest
    except Exception as exc:
        return None, f"Feishu member APIs unavailable: {exc}"

    try:
        request = (
            GetChatMembersRequest.builder()
            .chat_id(chat_id)
            .member_id_type("open_id")
            .page_size(limit)
            .build()
        )
        response = client.im.v1.chat_members.get(request)
    except Exception as exc:
        return None, f"Official Feishu member lookup failed for account_id={account_id!r}: {exc}"
    if not _response_succeeded(response):
        return None, _response_error(response, "Official Feishu member lookup failed.")

    data = getattr(response, "data", None)
    raw_members = list(getattr(data, "items", None) or [])
    open_ids = _safe_sort_strings([getattr(item, "member_id", None) for item in raw_members])

    enriched_by_open_id: Dict[str, Dict[str, Any]] = {}
    batch_error: Optional[str] = None
    if open_ids:
        try:
            batch_request = (
                BatchUserRequest.builder()
                .user_ids(open_ids)
                .user_id_type("open_id")
                .department_id_type("open_department_id")
                .build()
            )
            batch_response = client.contact.v3.user.batch(batch_request)
            if _response_succeeded(batch_response):
                for user in list(getattr(getattr(batch_response, "data", None), "items", None) or []):
                    serialized = _serialize_official_user(user, account_id)
                    open_id = serialized["open_ids"][0] if serialized["open_ids"] else None
                    if open_id:
                        observed_user = _observed_user_for_merge(
                            index,
                            account_id,
                            open_id=open_id,
                            user_id=serialized["user_ids"][0] if serialized["user_ids"] else None,
                            union_id=serialized["union_ids"][0] if serialized["union_ids"] else None,
                        )
                        enriched_by_open_id[open_id] = _merge_user_payload(serialized, observed_user)
            else:
                batch_error = _response_error(batch_response, "Official Feishu batch user lookup failed.")
        except Exception as exc:
            batch_error = f"Official Feishu batch user lookup failed: {exc}"

    members = []
    for item in raw_members:
        open_id = str(getattr(item, "member_id", "") or "").strip() or None
        if open_id and open_id in enriched_by_open_id:
            member = dict(enriched_by_open_id[open_id])
        else:
            member = {
                "account_id": account_id,
                "names": _safe_sort_strings([getattr(item, "name", None)]),
                "open_ids": _safe_sort_strings([open_id]),
                "user_ids": [],
                "union_ids": [],
                "unknown_ids": [],
                "chat_ids": [],
                "session_keys": [],
                "session_ids": [],
            }
            observed_user = _observed_user_for_merge(index, account_id, open_id=open_id)
            member = _merge_user_payload(member, observed_user)
        member["tenant_key"] = str(getattr(item, "tenant_key", "") or "").strip() or None
        members.append(member)

    chat_payload, chat_error = _official_get_chat(config, index, account_id, chat_id)
    if chat_error:
        observed_chat = _observed_chat_for_merge(index, account_id, chat_id)
        if not observed_chat:
            return None, chat_error
        chat_payload = observed_chat

    return {
        "success": True,
        "source": "official_api",
        "authoritative": True,
        "members_only": True,
        "chat": chat_payload,
        "members": members,
        "member_total": int(str(getattr(data, "member_total", "") or "0") or "0") or len(members),
        "has_more": bool(getattr(data, "has_more", False)),
        "page_token": str(getattr(data, "page_token", "") or "").strip() or None,
        "official_member_enrichment_error": batch_error,
    }, None


def _official_search_chats(
    config,
    index: Dict[str, Any],
    account_id: Optional[str],
    query: str,
    limit: int,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    candidate_accounts = _candidate_account_ids(config, account_id)
    if not candidate_accounts:
        return None, "No Feishu accounts are configured for official chat search."

    try:
        from lark_oapi.api.im.v1 import SearchChatRequest
    except Exception as exc:
        return None, f"SearchChatRequest unavailable: {exc}"

    matches_by_key: Dict[Tuple[Optional[str], str], Dict[str, Any]] = {}
    errors: List[str] = []
    for candidate_account_id in candidate_accounts:
        client, error = _build_official_client(config, candidate_account_id)
        if not client:
            errors.append(str(error))
            continue
        try:
            request = (
                SearchChatRequest.builder()
                .query(query)
                .user_id_type("open_id")
                .page_size(limit)
                .build()
            )
            response = client.im.v1.chat.search(request)
        except Exception as exc:
            errors.append(f"Official Feishu chat search failed for account_id={candidate_account_id!r}: {exc}")
            continue
        if not _response_succeeded(response):
            errors.append(_response_error(response, "Official Feishu chat search failed."))
            continue

        items = list(getattr(getattr(response, "data", None), "items", None) or [])
        for item in items:
            chat_id = str(getattr(item, "chat_id", "") or "").strip()
            if not chat_id:
                continue
            serialized = _serialize_official_chat(
                chat_id,
                item,
                candidate_account_id,
                home_channel_name=_home_channel_name(config, candidate_account_id, chat_id),
            )
            observed_chat = _observed_chat_for_merge(index, candidate_account_id, chat_id)
            matches_by_key[(candidate_account_id, chat_id)] = _merge_chat_payload(serialized, observed_chat)

    if matches_by_key:
        matches = list(matches_by_key.values())
        matches.sort(
            key=lambda item: _search_score(query, item["names"] + [item["chat_id"]])
        )
        return {
            "success": True,
            "source": "official_api",
            "authoritative": True,
            "matches": matches[:limit],
        }, None

    if errors:
        return None, "; ".join(errors)
    return None, "No official Feishu chats matched the provided query."


def _current_session_principal() -> Dict[str, Optional[str]]:
    from gateway.session_context import get_session_env

    return {
        "platform": get_session_env("HERMES_SESSION_PLATFORM") or None,
        "account_id": get_session_env("HERMES_SESSION_ACCOUNT_ID") or None,
        "user_id": get_session_env("HERMES_SESSION_USER_ID") or None,
        "user_id_alt": get_session_env("HERMES_SESSION_USER_ID_ALT") or None,
        "user_name": get_session_env("HERMES_SESSION_USER_NAME") or None,
        "chat_id": get_session_env("HERMES_SESSION_CHAT_ID") or None,
    }


def _authorize_account_scope(config, requested_account_id: Optional[str]) -> Optional[str]:
    if not requested_account_id:
        return None

    current = _current_session_principal()
    current_platform = str(current.get("platform") or "").lower()
    current_account_id = current.get("account_id")
    if current_platform in {"", "local"}:
        return None
    if current_platform != "feishu":
        return None
    if not current_account_id or current_account_id == requested_account_id:
        return None

    current_ids = {
        str(current.get("user_id") or "").strip(),
        str(current.get("user_id_alt") or "").strip(),
    }
    current_ids.discard("")
    if not current_ids:
        return "Cross-account feishu_id query requires current sender identity in session context."

    current_config = _resolve_platform_config(config, account_id=current_account_id)
    admins = _extract_admin_ids(current_config)
    if current_ids & admins:
        return None
    return (
        f"Cross-account feishu_id query denied for current Feishu session. "
        f"Requested account_id={requested_account_id!r}, current account_id={current_account_id!r}. "
        "Configure the current sender as an admin on the current account to allow this."
    )


def _effective_account_scope(args: Dict[str, Any], config) -> Optional[str]:
    requested = str(args.get("as_account_id") or args.get("account_id") or "").strip() or None
    if requested:
        return requested
    current = _current_session_principal()
    if str(current.get("platform") or "").lower() == "feishu" and current.get("account_id"):
        return str(current["account_id"])
    return None


def _normalize_limit(value: Any, default: int = 10) -> int:
    try:
        limit = int(value if value is not None else default)
    except Exception:
        limit = default
    return max(1, min(limit, 50))


def _search_score(query: str, values: List[str]) -> Tuple[int, str]:
    query_l = _normalize_text(query)
    best = 99
    exemplar = ""
    for value in values:
        current = _normalize_text(value)
        if current == query_l:
            return 0, value
        if current.startswith(query_l):
            best = min(best, 1)
            exemplar = exemplar or value
        elif query_l in current:
            best = min(best, 2)
            exemplar = exemplar or value
    return best, exemplar


def _match_account(item: Dict[str, Any], account_id: Optional[str]) -> bool:
    return account_id is None or item.get("account_id") == account_id


def _find_user_by_ids(index: Dict[str, Any], account_id: Optional[str], identifiers: Dict[str, str]) -> Optional[Dict[str, Any]]:
    if not identifiers:
        return None
    for user in index["users"]:
        if not _match_account(user, account_id):
            continue
        if any(value in user[f"{id_type}s"] for id_type, value in identifiers.items()):
            return user
    return None


def _find_chat(index: Dict[str, Any], account_id: Optional[str], chat_id: str) -> Optional[Dict[str, Any]]:
    for chat in index["chats"]:
        if not _match_account(chat, account_id):
            continue
        if chat["chat_id"] == chat_id:
            return chat
    return None


def _handle_whois_chat(args: Dict[str, Any], index: Dict[str, Any], account_id: Optional[str], config):
    current = _current_session_principal()
    chat_id = str(args.get("chat_id") or current.get("chat_id") or "").strip()
    if not chat_id:
        return None, "chat_id is required for whois_chat outside an active Feishu session."
    official_accounts = _candidate_account_ids(config, account_id)
    official_errors: List[str] = []
    for candidate_account_id in official_accounts:
        chat, official_error = _official_get_chat(config, index, candidate_account_id, chat_id)
        if chat:
            return {
                "success": True,
                "source": "official_api",
                "authoritative": True,
                "chat": chat,
            }, None
        if official_error:
            official_errors.append(official_error)

    chat = _find_chat(index, account_id, chat_id)
    if not chat:
        if official_errors:
            return None, "; ".join(official_errors)
        return None, f"No observed Feishu chat found for chat_id={chat_id!r}."
    payload = {
        "success": True,
        "source": "observed_local_index",
        "authoritative": False,
        "note": "Chat names and members are derived from local observed Hermes sessions only.",
        "chat": chat,
    }
    if official_errors:
        payload["official_error"] = "; ".join(official_errors)
    return payload, None


def _handle_members(args: Dict[str, Any], index: Dict[str, Any], account_id: Optional[str], config):
    current = _current_session_principal()
    chat_id = str(args.get("chat_id") or current.get("chat_id") or "").strip()
    if not chat_id:
        return None, "chat_id is required for members outside an active Feishu session."

    limit = _normalize_limit(args.get("limit"))
    official_accounts = _candidate_account_ids(config, account_id)
    official_errors: List[str] = []
    for candidate_account_id in official_accounts:
        payload, official_error = _official_get_members(config, index, candidate_account_id, chat_id, limit)
        if payload:
            return payload, None
        if official_error:
            official_errors.append(official_error)

    chat_payload, error = _handle_whois_chat(args, index, account_id, config)
    if error:
        return None, error
    chat_payload["members_only"] = True
    chat_payload["members"] = chat_payload["chat"].get("observed_members", [])
    chat_payload["note"] = "Members are only the Feishu users Hermes has locally observed in this chat."
    if official_errors:
        chat_payload["official_error"] = "; ".join(official_errors)
    return chat_payload, None


def _handle_search_chats(args: Dict[str, Any], index: Dict[str, Any], account_id: Optional[str], config):
    query = str(args.get("query") or "").strip()
    if not query:
        return None, "query is required for search_chats."
    limit = _normalize_limit(args.get("limit"))

    official_payload, official_error = _official_search_chats(config, index, account_id, query, limit)
    if official_payload:
        return official_payload, None

    matches = []
    for chat in index["chats"]:
        if not _match_account(chat, account_id):
            continue
        score, exemplar = _search_score(query, chat["names"] + [chat["chat_id"]])
        if score >= 99:
            continue
        matches.append((score, exemplar or chat["chat_id"], chat))
    matches.sort(key=lambda item: (item[0], item[1]))
    payload = {
        "success": True,
        "source": "observed_local_index",
        "authoritative": False,
        "matches": [item[2] for item in matches[:limit]],
    }
    if official_error:
        payload["official_error"] = official_error
    return payload, None


def _handle_whois_user(args: Dict[str, Any], index: Dict[str, Any], account_id: Optional[str], config):
    current = _current_session_principal()
    identifiers = {}
    for field in ("open_id", "user_id", "union_id"):
        value = str(args.get(field) or "").strip()
        if value:
            identifiers[field] = value
    if not identifiers and current.get("platform") == "feishu":
        classified = _classify_id(current.get("user_id"))
        if classified and classified[0]:
            identifiers[classified[0]] = classified[1]
        classified_alt = _classify_id(current.get("user_id_alt"))
        if classified_alt and classified_alt[0]:
            identifiers[classified_alt[0]] = classified_alt[1]
    if not identifiers:
        return None, "whois_user requires open_id, user_id, or union_id."

    official_accounts = _candidate_account_ids(config, account_id)
    official_errors: List[str] = []
    for candidate_account_id in official_accounts:
        user, official_error = _official_get_user(config, index, candidate_account_id, identifiers)
        if user:
            return {
                "success": True,
                "source": "official_api",
                "authoritative": True,
                "user": user,
            }, None
        if official_error:
            official_errors.append(official_error)

    user = _find_user_by_ids(index, account_id, identifiers)
    if not user:
        if official_errors:
            return None, "; ".join(official_errors)
        return None, "No observed Feishu user matched the provided identifiers."
    payload = {
        "success": True,
        "source": "observed_local_index",
        "authoritative": False,
        "user": user,
    }
    if official_errors:
        payload["official_error"] = "; ".join(official_errors)
    return payload, None


def _handle_resolve_user(args: Dict[str, Any], index: Dict[str, Any], account_id: Optional[str], config):
    query = str(args.get("query") or "").strip()
    if not query:
        return None, "query is required for resolve_user."
    limit = _normalize_limit(args.get("limit"))
    matches = []
    for user in index["users"]:
        if not _match_account(user, account_id):
            continue
        score, exemplar = _search_score(query, user["names"])
        if score >= 99:
            continue
        matches.append((score, exemplar or (user["names"][0] if user["names"] else ""), user))
    matches.sort(key=lambda item: (item[0], item[1]))
    return {
        "success": True,
        "source": "observed_local_index",
        "authoritative": False,
        "matches": [item[2] for item in matches[:limit]],
    }, None


def _handle_session_lookup(args: Dict[str, Any], index: Dict[str, Any], account_id: Optional[str], config):
    session_key = str(args.get("session_key") or "").strip()
    session_id = str(args.get("session_id") or "").strip()
    if session_key:
        session = next((item for item in index["sessions"] if item["session_key"] == session_key), None)
    elif session_id:
        session = next((item for item in index["sessions"] if item["session_id"] == session_id), None)
    else:
        return None, "session_lookup requires session_key or session_id."

    if not session or not _match_account(session, account_id):
        return None, "No observed Feishu session matched the provided key."
    return {
        "success": True,
        "source": "observed_local_index",
        "authoritative": False,
        "session": session,
    }, None


def _handle_my_chats(args: Dict[str, Any], index: Dict[str, Any], account_id: Optional[str], config):
    current = _current_session_principal()
    identifiers = {}
    for field in ("open_id", "user_id", "union_id"):
        value = str(args.get(field) or "").strip()
        if value:
            identifiers[field] = value
    if not identifiers and current.get("platform") == "feishu":
        classified = _classify_id(current.get("user_id"))
        if classified and classified[0]:
            identifiers[classified[0]] = classified[1]
        classified_alt = _classify_id(current.get("user_id_alt"))
        if classified_alt and classified_alt[0]:
            identifiers[classified_alt[0]] = classified_alt[1]
    if not identifiers:
        return None, "my_chats requires a Feishu sender context or explicit open_id/user_id/union_id."

    user = _find_user_by_ids(index, account_id, identifiers)
    if not user:
        return None, "No observed Feishu user matched the requested principal."

    chat_ids = set(user["chat_ids"])
    chats = [chat for chat in index["chats"] if chat["chat_id"] in chat_ids and _match_account(chat, account_id)]
    chats.sort(key=lambda item: ((item.get("account_id") or ""), item["chat_id"]))
    return {
        "success": True,
        "source": "observed_local_index",
        "authoritative": False,
        "user": user,
        "chats": chats[:_normalize_limit(args.get("limit"))],
    }, None


_ACTION_HANDLERS = {
    "whois_chat": _handle_whois_chat,
    "members": _handle_members,
    "search_chats": _handle_search_chats,
    "whois_user": _handle_whois_user,
    "resolve_user": _handle_resolve_user,
    "session_lookup": _handle_session_lookup,
    "my_chats": _handle_my_chats,
}


def feishu_id_tool(args: Dict[str, Any], **_kw) -> str:
    from tools.registry import tool_error, tool_result

    action = str(args.get("action") or "").strip()
    handler = _ACTION_HANDLERS.get(action)
    if not handler:
        return tool_error(f"Unknown action: {action}")

    config = _load_gateway_config_safe()
    account_id = _effective_account_scope(args, config)
    auth_error = _authorize_account_scope(config, account_id)
    if auth_error:
        return tool_error(auth_error)

    index = _build_observed_index(config)
    payload, error = handler(args, index, account_id, config)
    if error:
        return tool_error(error)
    return tool_result(payload)


from tools.registry import registry

registry.register(
    name="feishu_id",
    toolset="messaging",
    schema=FEISHU_ID_SCHEMA,
    handler=feishu_id_tool,
    check_fn=_check_feishu_id,
    emoji="🪪",
)
