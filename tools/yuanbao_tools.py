"""yuanbao_tools.py - 元宝平台工具集 (the "hermes-yuanbao" toolset).

get_group_info / query_group_members / search_sticker / send_sticker / send_dm. Sticker
flow mirrors chatbot-web's sticker-search/sticker-send: the LLM should search_sticker for
a sticker_id (or pass the Chinese name), then send_sticker — never bare Unicode emoji.
The active adapter singleton lives in ``gateway.platforms.yuanbao.YuanbaoAdapter.get_active``.
"""

from __future__ import annotations

import functools
import logging
from contextlib import suppress
from pathlib import Path
from typing import Tuple

from tools.registry import registry, tool_result

logger = logging.getLogger(__name__)

_USER_TYPE_LABEL = {0: "unknown", 1: "user", 2: "yuanbao_ai", 3: "bot"}

MENTION_HINT = (
    'To @mention a user, you MUST use the format: '
    'space + @ + nickname + space (e.g. " @Alice ").'
)

# Image extensions for media dispatch (mirrors MessageSender.IMAGE_EXTS)
_IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"})


class _YbError(Exception):
    """Tool-level failure; ``payload`` is the error envelope returned to the model."""

    def __init__(self, msg: str, **extra):
        super().__init__(msg)
        self.payload = {"success": False, "error": msg, **extra}


def _yb_tool(label: str):
    """Handler decorator: ``fn(args)`` → tool_result; ``_YbError`` → its envelope; else logged envelope."""
    def deco(fn):
        @functools.wraps(fn)
        async def handler(args, **kw):
            try:
                return tool_result(await fn(args))
            except _YbError as exc:
                return tool_result(exc.payload)
            except Exception as exc:
                logger.exception("[yuanbao_tools] %s error", label)
                return tool_result(_YbError(str(exc)).payload)
        return handler
    return deco


def _get_active_adapter():
    """Lazy import to avoid ImportError when gateway.platforms.yuanbao is unavailable."""
    with suppress(ImportError):
        from gateway.platforms.yuanbao import YuanbaoAdapter
        return YuanbaoAdapter.get_active()
    return None


def _adapter():
    if (adapter := _get_active_adapter()) is None:
        raise _YbError("Yuanbao adapter is not connected")
    return adapter


def _session_env(name: str) -> str:
    with suppress(Exception):
        from gateway.session_context import get_session_env
        return get_session_env(name, "")
    return ""


def _nick(m: dict, default: str = "") -> str:
    return m.get("nickname", m.get("nick_name", default))


async def _members(adapter, group_code: str) -> list:
    raw = await adapter.get_group_member_list(group_code)
    if raw is None:
        raise _YbError("get_group_member_list returned None")
    return raw.get("members", [])


async def _resolve_dm_recipient(adapter, group_code: str, name: str) -> Tuple[str, str]:
    """Resolve ``name`` to (user_id, nickname) via the group member list; >1 partial match raises
    with ``candidates`` for disambiguation instead of guessing."""
    if not group_code:
        raise _YbError("group_code is required when user_id is not provided")
    if not name:
        raise _YbError("name is required when user_id is not provided")
    filt = name.strip().lower()
    matched = [m for m in await _members(adapter, group_code)
               if filt in (m.get("nickname") or m.get("nick_name") or "").lower()]
    if not matched:
        raise _YbError(f'No member matching "{name}" found in group {group_code}.')
    if len(matched) > 1:
        raise _YbError(
            f'Multiple members match "{name}". Please specify which one.',
            candidates=[{"user_id": m.get("user_id", ""), "nickname": _nick(m)} for m in matched],
        )
    m = matched[0]
    return m.get("user_id", ""), _nick(m, name)


@_yb_tool("get_group_info")
async def get_group_info(args) -> dict:
    """查询群基本信息（群名、群主、成员数）。"""
    group_code = args.get("group_code", "")
    if not group_code:
        raise _YbError("group_code is required")
    gi = await _adapter().query_group_info(group_code)
    if gi is None:
        raise _YbError("query_group_info returned None")
    return {
        "success": True, "group_code": group_code, "group_name": gi.get("group_name", ""),
        "member_count": gi.get("member_count", 0),
        "owner": {"user_id": gi.get("owner_id", ""), "nickname": gi.get("owner_nickname", "")},
        "note": 'The group is called "派 (Pai)" in the app.',
    }


@_yb_tool("query_group_members")
async def query_group_members(args) -> dict:
    """统一的群成员查询（对齐 TS query_session_members）。
    action: find (按昵称模糊搜索; 无 name 时等同 list_all) / list_bots / list_all (默认)."""
    group_code, name = args.get("group_code", ""), args.get("name", "")
    action = args.get("action", "list_all")
    if not group_code:
        raise _YbError("group_code is required")
    all_members = [
        {"user_id": m.get("user_id", ""), "nickname": _nick(m),
         "role": _USER_TYPE_LABEL.get(m.get("user_type", m.get("role", 0)), "unknown")}
        for m in await _members(_adapter(), group_code)
    ]
    if not all_members:
        raise _YbError("No members found in this group.")

    hint = {"mention_hint": MENTION_HINT} if args.get("mention", False) else {}

    def _listing(ok: bool, msg: str, members: list) -> dict:
        return {"success": ok, "msg": msg, "members": members, **hint}

    if action == "list_bots":
        bots = [m for m in all_members if m["role"] in {"yuanbao_ai", "bot"}]
        if not bots:
            raise _YbError("No bots found in this group.")
        return _listing(True, f"Found {len(bots)} bot(s).", bots)

    if action == "find" and name:
        filt = name.strip().lower()
        matched = [m for m in all_members if filt in m["nickname"].lower()]
        if matched:
            return _listing(True, f'Found {len(matched)} member(s) matching "{name}".', matched)
        return _listing(False, f'No match for "{name}". All members listed below.', all_members)

    return _listing(True, f"Found {len(all_members)} member(s).", all_members)


@_yb_tool("search_sticker")
async def search_sticker(args) -> dict:
    """在内置贴纸表中按关键词模糊搜索，返回 Top-N 候选（空 query 返回前 N 条）。"""
    from gateway.platforms.yuanbao_sticker import search_stickers

    query, limit = args.get("query", ""), args.get("limit", 10)
    try:
        safe_limit = max(1, min(50, int(limit) if limit else 10))
    except (TypeError, ValueError):
        safe_limit = 10
    matches = search_stickers(query or "", limit=safe_limit)
    return {
        "success": True, "query": query or "", "count": len(matches),
        "results": [{k: s.get(k, "") for k in ("sticker_id", "name", "description", "package_id")}
                    for s in matches],
    }


@_yb_tool("send_sticker")
async def send_sticker(args) -> dict:
    """向 chat_id（缺省取当前会话 HERMES_SESSION_CHAT_ID）发送一张内置贴纸（TIMFaceElem）。
    ``sticker``: 名称（如 "六六六"）或 sticker_id（如 "278"）；为空时随机发送。
    ``chat_id``: ``direct:{account_id}`` / ``group:{group_code}`` / 裸 account_id。"""
    from gateway.platforms.yuanbao_sticker import get_sticker_by_id, get_sticker_by_name, get_random_sticker

    target = (args.get("chat_id", "") or "").strip() or _session_env("HERMES_SESSION_CHAT_ID")
    if not target:
        raise _YbError("chat_id is required (no active yuanbao session detected)")
    adapter = _adapter()

    raw = (args.get("sticker", "") or "").strip()
    if not raw:
        sticker_obj = get_random_sticker()
    else:
        sticker_obj = (get_sticker_by_id(raw) if raw.isdigit() else None) or get_sticker_by_name(raw)
    if sticker_obj is None:
        raise _YbError(f"Sticker not found: {raw!r}. Use search_sticker first to discover available stickers.")

    result = await adapter.send_sticker(chat_id=target, sticker_name=sticker_obj.get("name", ""),
                                        reply_to=args.get("reply_to", "") or None)
    if not getattr(result, "success", False):
        raise _YbError(getattr(result, "error", "send_sticker failed"))
    return {
        "success": True, "chat_id": target,
        "sticker": {"sticker_id": sticker_obj.get("sticker_id", ""), "name": sticker_obj.get("name", "")},
        "message_id": getattr(result, "message_id", None),
        "note": "Sticker delivered to the chat. If you have additional text to say, reply now; otherwise end your turn without generating text.",
    }


@_yb_tool("send_dm")
async def send_dm(args) -> dict:
    """Send a DM to a group member, with optional media. group_code defaults to the session's
    "group:<code>" chat_id. Without ``user_id`` the member list is searched by ``name`` (partial,
    case-insensitive; >1 match returns candidates). media_files items are {"path", "is_voice"} dicts or
    (path, is_voice) pairs; ``MEDIA:<path>`` tags in the text count too. Partial media failures are
    reported in ``note``, not as failure."""
    group_code = args.get("group_code", "")
    if not group_code and (chat_id := _session_env("HERMES_SESSION_CHAT_ID")).startswith("group:"):
        group_code = chat_id.split(":", 1)[1]

    media_files = []
    for item in args.get("media_files") or []:
        if isinstance(item, dict):
            media_files.append((item.get("path", ""), bool(item.get("is_voice", False))))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            media_files.append((str(item[0]), bool(item[1])))

    from gateway.platforms.base import BasePlatformAdapter
    embedded_media, message = BasePlatformAdapter.extract_media(args.get("message", ""))
    media_files = BasePlatformAdapter.filter_media_delivery_paths(media_files + list(embedded_media or []))

    if not message and not media_files:
        raise _YbError("message or media_files is required")
    adapter = _adapter()

    name = args.get("name", "")
    resolved_user_id, resolved_nickname = (args.get("user_id", "") or "").strip(), name.strip()
    if not resolved_user_id:
        resolved_user_id, resolved_nickname = await _resolve_dm_recipient(adapter, group_code, name)
    if not resolved_user_id:
        raise _YbError("Could not resolve user_id")

    chat_id = f"direct:{resolved_user_id}"
    last_result = None
    errors: list[str] = []
    if message and message.strip():
        last_result = await adapter.send_dm(resolved_user_id, message, group_code=group_code)
        if not last_result.success:
            errors.append(last_result.error or "text send failed")

    for media_path, _is_voice in media_files:
        send = adapter.send_image_file if Path(media_path).suffix.lower() in _IMAGE_EXTS else adapter.send_document
        last_result = await send(chat_id, media_path, group_code=group_code)
        if not last_result.success:
            errors.append(last_result.error or "media send failed")

    if last_result is None:
        raise _YbError("No deliverable text or media remained")
    if errors and not last_result.success:
        raise _YbError("; ".join(errors))

    note = f'DM sent to "{resolved_nickname}" successfully.'
    if errors:
        note += f" (partial failure: {'; '.join(errors)})"
    return {"success": True, "user_id": resolved_user_id, "nickname": resolved_nickname,
            "message_id": last_result.message_id, "note": note}


def _check_yuanbao():
    """Toolset availability check — True when running in a yuanbao gateway session."""
    return _session_env("HERMES_SESSION_PLATFORM") == "yuanbao" or _get_active_adapter() is not None


# (schema, handler, emoji); the tool name is schema["name"].
_TOOLS = (
    (
        {
            "name": "yb_query_group_info",
            "description": (
                "Query basic info about a group (called '派/Pai' in the app), "
                "including group name, owner, and member count."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "group_code": {
                        "type": "string",
                        "description": "The unique group identifier (group_code).",
                    },
                },
                "required": ["group_code"],
            },
        },
        get_group_info,
        "👥",
    ),
    (
        {
            "name": "yb_query_group_members",
            "description": (
                "Query members of a group (called '派/Pai' in the app). "
                "Use this tool when you need to @mention someone, find a user by name, "
                "list bots (including Yuanbao AI), or list all members. "
                "IMPORTANT: You MUST call this tool before @mentioning any user, "
                "because you need the exact nickname to construct the @mention format."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "group_code": {
                        "type": "string",
                        "description": "The unique group identifier (group_code).",
                    },
                    "action": {
                        "type": "string",
                        "enum": ["find", "list_bots", "list_all"],
                        "description": (
                            "find — search a user by name (use when you need to @mention or look up someone); "
                            "list_bots — list bots and Yuanbao AI assistants; "
                            "list_all — list all members."
                        ),
                    },
                    "name": {
                        "type": "string",
                        "description": (
                            "User name to search (partial match, case-insensitive). "
                            "Required for 'find'. Use the name the user mentioned in the conversation."
                        ),
                    },
                    "mention": {
                        "type": "boolean",
                        "description": (
                            "Set to true when you need to @mention/at someone in your reply. "
                            "The response will include the exact @mention format to use."
                        ),
                    },
                },
                "required": ["group_code", "action"],
            },
        },
        query_group_members,
        "📋",
    ),
    (
        {
            "name": "yb_send_dm",
            "description": (
                "Send a private/direct message (DM) to a user in a group, with optional media files. "
                "This tool automatically looks up the user by name in the group member list "
                "and sends the message. Use this when someone asks to privately message / 私信 / DM a user. "
                "Supports text, images, and file attachments. "
                "You can also provide user_id directly if already known."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "group_code": {
                        "type": "string",
                        "description": (
                            "The group where the target user belongs. "
                            "Extract from chat_id: 'group:328306697' → '328306697'. "
                            "Required when user_id is not provided."
                        ),
                    },
                    "name": {
                        "type": "string",
                        "description": (
                            "Target user's display name (partial match, case-insensitive). "
                            "Required when user_id is not provided."
                        ),
                    },
                    "message": {
                        "type": "string",
                        "description": "The message text to send as a DM. Can be empty if only sending media.",
                    },
                    "user_id": {
                        "type": "string",
                        "description": (
                            "Target user's account ID. If provided, skips the member lookup. "
                            "Usually obtained from a previous yb_query_group_members call."
                        ),
                    },
                    "media_files": {
                        "type": "array",
                        "description": (
                            "Optional list of media files to send along with the DM. "
                            "Images (.jpg/.png/.gif/.webp/.bmp) are sent as image messages; "
                            "other files are sent as document attachments."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {
                                    "type": "string",
                                    "description": "Absolute local file path of the media to send.",
                                },
                                "is_voice": {
                                    "type": "boolean",
                                    "description": "Whether this file is a voice message (default false).",
                                },
                            },
                            "required": ["path"],
                        },
                    },
                },
                "required": [],
            },
        },
        send_dm,
        "✉️",
    ),
    (
        {
            "name": "yb_search_sticker",
            "description": (
                "Search the built-in Yuanbao sticker (TIM face / 表情包) catalogue by keyword. "
                "Returns the top matching candidates with sticker_id, name, and description. "
                "Use this BEFORE yb_send_sticker to discover the right sticker_id. "
                "Sticker = 贴纸 = TIM face — NOT a message reaction. "
                "Prefer sending a sticker over bare Unicode emoji when reacting/expressing emotion."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Search keyword (Chinese or English, e.g. '666', '比心', 'cool', '吃瓜'). "
                            "Empty string returns the first N stickers."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max number of candidates to return (default 10, max 50).",
                    },
                },
                "required": [],
            },
        },
        search_sticker,
        "🔍",
    ),
    (
        {
            "name": "yb_send_sticker",
            "description": (
                "Send a built-in sticker (TIMFaceElem / 贴纸表情) to the current Yuanbao chat. "
                "Call yb_search_sticker first if you don't know the sticker_id/name. "
                "Sticker = 贴纸 = TIM face — NOT a message reaction. "
                "CRITICAL: Whenever the user asks you to send a sticker / 贴纸 / 表情包, you MUST "
                "use this tool. DO NOT draw a PNG via execute_code / Pillow / matplotlib and "
                "then call send_image_file — that produces a fake 'sticker' image instead of a "
                "real TIM face and is the WRONG path. If no suitable sticker_id is known, call "
                "yb_search_sticker first. When the recent thread shows users sending stickers, "
                "prefer matching that tone by replying with a sticker instead of (or in "
                "addition to) text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sticker": {
                        "type": "string",
                        "description": (
                            "Sticker name (e.g. '六六六', '比心', 'ok') or numeric sticker_id "
                            "(e.g. '278'). Empty string sends a random built-in sticker."
                        ),
                    },
                    "chat_id": {
                        "type": "string",
                        "description": (
                            "Target chat. Defaults to the current session. "
                            "Format: 'direct:{account_id}', 'group:{group_code}', or bare account_id."
                        ),
                    },
                    "reply_to": {
                        "type": "string",
                        "description": "Optional ref_msg_id to quote-reply (group chat only).",
                    },
                },
                "required": [],
            },
        },
        send_sticker,
        "🎨",
    ),
)

for _schema, _handler, _emoji in _TOOLS:
    registry.register(
        name=_schema["name"], toolset="hermes-yuanbao", schema=_schema, handler=_handler,
        check_fn=_check_yuanbao, is_async=True, emoji=_emoji,
    )


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import List  # noqa: F401,E402
from typing import Optional  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
