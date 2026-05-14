"""Native Tlon management tool for Hermes.

This exposes the operational surface from the Tlon OpenClaw plugin and
tlon-skill CLI through Hermes tools: group/channel management, message
history, contacts, settings, activity, expose controls, and raw scry/poke
escape hatches.
"""

from __future__ import annotations

import json
import mimetypes
import os
import random
import re
import string
import time
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from agent.redact import redact_sensitive_text
from gateway.platforms.tlon import (
    _extract_message_text,
    _format_ud,
    _normalize_ship,
    _text_to_story,
)
from gateway.platforms.tlon_media import format_blob_annotations, parse_blob_data
from tools.registry import registry


_ADMIN_ROLE_ID = "admin"
_CHANNEL_KINDS = {"chat", "diary", "heap"}
_GROUP_PRIVACY = {"public", "private", "secret"}
_SETTINGS_DESK = "moltbot"
_SETTINGS_BUCKET = "tlon"
_MAX_RESULT_CHARS = 60000


TLON_SCHEMA = {
    "name": "tlon",
    "description": (
        "Manage a connected Tlon/Urbit ship. Use this for Tlon groups, "
        "channels, roles/admins, invites, contacts, settings, message history, "
        "post reactions/edits/deletes, activity, expose controls, and raw "
        "scry/poke/thread calls. For ordinary message sending, use send_message; "
        "for creating a group or channel, inviting ships, or making someone admin, "
        "use this tool."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "raw_scry",
                    "raw_poke",
                    "raw_thread",
                    "groups_list",
                    "group_info",
                    "group_create",
                    "group_create_owned",
                    "group_invite",
                    "group_leave",
                    "group_join",
                    "group_request_invite",
                    "group_accept_invite",
                    "group_reject_invite",
                    "group_cancel_join",
                    "group_rescind_request",
                    "group_revoke_invite",
                    "group_delete",
                    "group_update",
                    "group_kick",
                    "group_ban",
                    "group_unban",
                    "group_add_role",
                    "group_delete_role",
                    "group_update_role",
                    "group_assign_role",
                    "group_remove_role",
                    "group_promote",
                    "group_demote",
                    "group_set_privacy",
                    "group_accept_join",
                    "group_reject_join",
                    "channel_create",
                    "channel_info",
                    "channel_update",
                    "channel_delete",
                    "channel_add_writers",
                    "channel_remove_writers",
                    "channel_add_readers",
                    "channel_remove_readers",
                    "channel_join",
                    "channel_leave",
                    "channels_list",
                    "messages_history",
                    "messages_search",
                    "message_context",
                    "message_get",
                    "post_react",
                    "post_unreact",
                    "post_edit",
                    "post_delete",
                    "dm_accept",
                    "dm_decline",
                    "dm_react",
                    "dm_unreact",
                    "dm_delete",
                    "notebook_post",
                    "upload_file",
                    "hook_template",
                    "hook_list",
                    "hook_get",
                    "hook_add",
                    "hook_edit",
                    "hook_delete",
                    "hook_order",
                    "hook_config",
                    "hook_cron",
                    "hook_rest",
                    "blocked_list",
                    "block_ship",
                    "unblock_ship",
                    "contacts_list",
                    "contact_self",
                    "contact_get",
                    "contact_sync",
                    "contact_add",
                    "contact_remove",
                    "contact_update",
                    "profile_update",
                    "settings_get",
                    "settings_set",
                    "settings_delete",
                    "settings_add_to_array",
                    "settings_remove_from_array",
                    "settings_set_channel_rule",
                    "settings_allow_dm",
                    "settings_remove_dm",
                    "settings_allow_channel",
                    "settings_remove_channel",
                    "settings_open_channel",
                    "settings_restrict_channel",
                    "settings_authorize_ship",
                    "settings_deauthorize_ship",
                    "settings_allow_group_inviter",
                    "settings_remove_group_inviter",
                    "settings_set_owner",
                    "settings_set_bool",
                    "owner_listen_status",
                    "owner_listen_set",
                    "owner_listen_channel_set",
                    "activity",
                    "unreads",
                    "expose_list",
                    "expose_show",
                    "expose_hide",
                    "expose_check",
                    "expose_url",
                ],
                "description": "Tlon operation to perform.",
            },
            "app": {"type": "string", "description": "Gall app for raw_scry/raw_poke, e.g. groups, channels, chat, settings."},
            "path": {"type": "string", "description": "Scry path for raw_scry, without the app prefix, e.g. /v2/groups."},
            "mark": {"type": "string", "description": "Mark for raw_poke."},
            "json": {"description": "JSON body for raw_poke."},
            "desk": {"type": "string", "description": "Desk for raw_thread, e.g. groups."},
            "input_mark": {"type": "string", "description": "Thread input mark for raw_thread."},
            "output_mark": {"type": "string", "description": "Thread output mark for raw_thread."},
            "thread_name": {"type": "string", "description": "Thread name for raw_thread."},
            "body": {"description": "Thread body for raw_thread."},
            "group_id": {"type": "string", "description": "Group flag, e.g. ~host/group-slug."},
            "channel_id": {"type": "string", "description": "Channel nest or DM id, e.g. chat/~host/general or ~ship."},
            "kind": {"type": "string", "enum": ["chat", "diary", "heap"], "description": "Channel kind."},
            "title": {"type": "string", "description": "Group/channel/role title."},
            "description": {"type": "string", "description": "Group/channel/role description."},
            "image": {"type": "string", "description": "Image URL for metadata/profile."},
            "cover": {"type": "string", "description": "Cover URL for metadata/profile."},
            "ship": {"type": "string", "description": "Single ship, e.g. ~zod."},
            "ships": {"type": "array", "items": {"type": "string"}, "description": "Ship list."},
            "role_id": {"type": "string", "description": "Group role id."},
            "privacy": {"type": "string", "enum": ["public", "private", "secret"], "description": "Group privacy."},
            "post_id": {"type": "string", "description": "Post id, dotted or bare @ud. For DMs, include ~author/id when possible."},
            "parent_id": {"type": "string", "description": "Parent post id for reply operations."},
            "author_id": {"type": "string", "description": "Author ship for DM/post operations when needed."},
            "message": {"type": "string", "description": "Text/markdown for post_edit."},
            "source": {"type": "string", "description": "Inline hook source, notebook Story JSON, or upload URL depending on action."},
            "hook_id": {"type": "string", "description": "Hook id for hook_* actions."},
            "hook_ids": {"type": "array", "items": {"type": "string"}, "description": "Ordered hook id list for hook_order."},
            "schedule": {"type": "string", "description": "Hook cron schedule in @dr format, e.g. ~h1 or ~m30."},
            "content_type": {"type": "string", "description": "MIME type for upload_file."},
            "file_name": {"type": "string", "description": "File name for upload_file."},
            "emoji": {"type": "string", "description": "Emoji reaction."},
            "query": {"type": "string", "description": "Search query."},
            "limit": {"type": "integer", "description": "Maximum records to return.", "default": 20},
            "mode": {"type": "string", "description": "List mode or activity bucket. For channels_list: all, dms, group_dms, groups. For activity: all, mentions, replies."},
            "key": {"type": "string", "description": "Settings key."},
            "value": {"description": "Settings value or contact/profile field value."},
            "resolve_blobs": {"type": "boolean", "description": "Include readable blob attachment summaries in message results.", "default": True},
            "include_replies": {"type": "boolean", "description": "Include replies when supported.", "default": True},
        },
        "required": ["action"],
    },
}


class TlonToolError(Exception):
    """User-facing Tlon tool failure."""


class TlonHttpClient:
    def __init__(self, *, ship_url: str, ship_name: str, ship_code: str):
        self.ship_url = ship_url.rstrip("/")
        self.ship_name = _normalize_ship(ship_name)
        self.ship_code = ship_code
        self._session = None
        self._cookie = ""

    async def __aenter__(self) -> "TlonHttpClient":
        try:
            await self.authenticate()
        except Exception:
            if self._session:
                await self._session.close()
                self._session = None
            raise
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    async def authenticate(self) -> None:
        try:
            import aiohttp
        except ImportError as exc:
            raise TlonToolError("aiohttp is required for the Tlon tool") from exc

        self._session = aiohttp.ClientSession()
        async with self._session.post(
            f"{self.ship_url}/~/login",
            data={"password": self.ship_code},
            allow_redirects=False,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            if resp.status not in (200, 204, 302, 303, 307):
                raise TlonToolError(f"Tlon auth failed: HTTP {resp.status}")
            cookie = resp.headers.get("set-cookie", "")
            if not cookie:
                for c in self._session.cookie_jar:
                    if c.key.startswith("urbauth"):
                        cookie = f"{c.key}={c.value}"
                        break
            if not cookie:
                raise TlonToolError("Tlon auth failed: no urbauth cookie received")
            self._cookie = cookie.split(";", 1)[0]

    @property
    def ship_no_sig(self) -> str:
        return self.ship_name.lstrip("~")

    async def scry(self, app: str, path: str, *, timeout: int = 60) -> Any:
        import aiohttp

        if not app or not path:
            raise TlonToolError("raw_scry requires app and path")
        if not path.startswith("/"):
            path = "/" + path
        suffix = path if path.endswith(".json") else f"{path}.json"
        async with self._session.get(
            f"{self.ship_url}/~/scry/{app}{suffix}",
            headers={"Cookie": self._cookie},
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise TlonToolError(f"Scry {app}{path} failed: HTTP {resp.status} - {text[:300]}")
            return await resp.json()

    async def poke(self, app: str, mark: str, json_data: Any, *, timeout: int = 30) -> Dict[str, Any]:
        import aiohttp

        if not app or not mark:
            raise TlonToolError("raw_poke requires app and mark")
        channel_id = f"tlon-tool-{int(time.time())}-{uuid.uuid4().hex[:8]}"
        channel_url = f"{self.ship_url}/~/channel/{channel_id}"
        action = {
            "id": 1,
            "action": "poke",
            "ship": self.ship_no_sig,
            "app": app,
            "mark": mark,
            "json": json_data,
        }
        headers = {"Content-Type": "application/json", "Cookie": self._cookie}
        async with self._session.put(
            channel_url,
            json=[action],
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            if resp.status not in (200, 204):
                text = await resp.text()
                raise TlonToolError(f"Poke {app}/{mark} failed: HTTP {resp.status} - {text[:300]}")
        try:
            async with self._session.delete(
                channel_url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=5),
            ):
                pass
        except Exception:
            pass
        return {"success": True, "app": app, "mark": mark}

    async def thread(
        self,
        *,
        desk: str,
        input_mark: str,
        thread_name: str,
        output_mark: str,
        body: Any,
        timeout: int = 90,
    ) -> Any:
        import aiohttp

        if not all([desk, input_mark, thread_name, output_mark]):
            raise TlonToolError("raw_thread requires desk, input_mark, thread_name, and output_mark")
        output = output_mark if output_mark.endswith(".json") else f"{output_mark}.json"
        url = f"{self.ship_url}/spider/{desk}/{input_mark}/{thread_name}/{output}"
        async with self._session.post(
            url,
            json=body,
            headers={"Content-Type": "application/json", "Cookie": self._cookie},
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            text = await resp.text()
            if not (200 <= resp.status < 300):
                raise TlonToolError(f"Thread {desk}/{thread_name} failed: HTTP {resp.status} - {text[:300]}")
            if not text:
                return {"success": True}
            try:
                return json.loads(text)
            except ValueError:
                return text


def tlon_tool(args: Dict[str, Any], **_kw: Any) -> str:
    try:
        from model_tools import _run_async

        result = _run_async(_tlon_tool_async(args or {}))
        return _json_result(result)
    except TlonToolError as exc:
        return _json_result({"error": _sanitize(str(exc))})
    except Exception as exc:
        return _json_result({"error": _sanitize(f"Tlon tool failed: {exc}")})


async def _tlon_tool_async(args: Dict[str, Any]) -> Dict[str, Any]:
    cfg = _load_tlon_config()
    action = str(args.get("action", "")).strip()
    if not action:
        raise TlonToolError("action is required")

    async with TlonHttpClient(**cfg) as client:
        if action == "raw_scry":
            return _ok(action, result=await client.scry(_required(args, "app"), _required(args, "path")))
        if action == "raw_poke":
            return _ok(
                action,
                result=await client.poke(_required(args, "app"), _required(args, "mark"), args.get("json")),
            )
        if action == "raw_thread":
            return _ok(
                action,
                result=await client.thread(
                    desk=_required(args, "desk"),
                    input_mark=_required(args, "input_mark"),
                    thread_name=_required(args, "thread_name"),
                    output_mark=_required(args, "output_mark"),
                    body=args.get("body"),
                ),
            )

        groups = TlonGroups(client)
        channels = TlonChannels(client)
        messages = TlonMessages(client)
        contacts = TlonContacts(client)
        settings = TlonSettingsTool(client)
        hooks = TlonHooks(client)
        misc = TlonMisc(client)

        if action.startswith("group_") or action == "groups_list":
            return await groups.handle(action, args)
        if action.startswith("channel_") or action == "channels_list":
            return await channels.handle(action, args)
        if action.startswith("message") or action.startswith("post_") or action.startswith("dm_") or action == "notebook_post":
            return await messages.handle(action, args)
        if action.startswith("hook_"):
            return await hooks.handle(action, args)
        if action.startswith("contact") or action == "profile_update":
            return await contacts.handle(action, args)
        if action.startswith("settings_") or action.startswith("owner_listen_"):
            return await settings.handle(action, args)
        if action in {"activity", "unreads", "upload_file"} or action.startswith("expose_") or action in {"blocked_list", "block_ship", "unblock_ship"}:
            return await misc.handle(action, args)

    raise TlonToolError(f"Unknown Tlon action: {action}")


class TlonGroups:
    def __init__(self, client: TlonHttpClient):
        self.client = client

    async def handle(self, action: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if action == "groups_list":
            groups = await _best_effort_scry(
                self.client,
                [
                    ("groups", "/v2/groups"),
                    ("groups-ui", "/v7/init"),
                    ("groups-ui", "/v8/init"),
                ],
            )
            return _ok(action, groups=groups)
        if action == "group_info":
            group_id = _required(args, "group_id")
            return _ok(action, group_id=group_id, group=await self.client.scry("groups", f"/v2/ui/groups/{group_id}"))
        if action in {"group_create", "group_create_owned"}:
            owner = _normalize_ship(str(args.get("ship") or args.get("owner") or os.getenv("TLON_OWNER_SHIP", "")))
            members = _ships(args)
            if action == "group_create_owned" and owner:
                members = _unique_ships([owner, *members])
            created = await self.create_group(
                title=_required(args, "title"),
                description=str(args.get("description") or ""),
                member_ids=members,
            )
            if action == "group_create_owned" and owner:
                await self.ensure_admin_role(created["group_id"])
                await self.assign_role(created["group_id"], _ADMIN_ROLE_ID, [owner])
                created["owner_ship"] = owner
                created["admin_role"] = _ADMIN_ROLE_ID
            return _ok(action, **created)
        if action == "group_invite":
            group_id = _required(args, "group_id")
            ships = _required_ships(args)
            await self.invite(group_id, ships)
            return _ok(action, group_id=group_id, invited=ships)
        if action == "group_leave":
            group_id = _required(args, "group_id")
            await self.client.poke("groups", "group-leave", group_id)
            return _ok(action, group_id=group_id)
        if action == "group_join":
            group_id = _required(args, "group_id")
            await self.client.poke("groups", "group-join", {"flag": group_id, "join-all": True})
            return _ok(action, group_id=group_id)
        if action == "group_request_invite":
            group_id = _required(args, "group_id")
            await self.client.poke("groups", "group-knock", group_id)
            return _ok(action, group_id=group_id)
        if action == "group_accept_invite":
            group_id = _required(args, "group_id")
            await self.client.poke("groups", "group-join", {"flag": group_id, "join-all": True})
            return _ok(action, group_id=group_id)
        if action == "group_reject_invite":
            group_id = _required(args, "group_id")
            await self.client.poke("groups", "group-reject", group_id)
            return _ok(action, group_id=group_id)
        if action == "group_cancel_join":
            group_id = _required(args, "group_id")
            await self.client.poke("groups", "group-cancel", group_id)
            return _ok(action, group_id=group_id)
        if action == "group_rescind_request":
            group_id = _required(args, "group_id")
            await self.client.poke("groups", "group-rescind", group_id)
            return _ok(action, group_id=group_id)
        if action == "group_revoke_invite":
            return await self._group_action(action, args, {"entry": {"pending": {"ships": _required_ships(args), "a-pending": {"del": None}}}})
        if action == "group_delete":
            return await self._group_action(action, args, {"delete": None})
        if action == "group_update":
            group_id = _required(args, "group_id")
            current = await self.client.scry("groups", f"/v2/ui/groups/{group_id}")
            meta = {
                "title": args.get("title") or _get_in(current, ["meta", "title"], ""),
                "description": args.get("description") if args.get("description") is not None else _get_in(current, ["meta", "description"], ""),
                "image": args.get("image") if args.get("image") is not None else _get_in(current, ["meta", "image"], ""),
                "cover": args.get("cover") if args.get("cover") is not None else _get_in(current, ["meta", "cover"], ""),
            }
            await self._poke_group(group_id, {"meta": meta})
            return _ok(action, group_id=group_id, meta=meta)
        if action == "group_kick":
            return await self._group_action(action, args, {"seat": {"ships": _required_ships(args), "a-seat": {"del": None}}})
        if action == "group_ban":
            return await self._group_action(action, args, {"entry": {"ban": {"add-ships": _required_ships(args)}}})
        if action == "group_unban":
            return await self._group_action(action, args, {"entry": {"ban": {"del-ships": _required_ships(args)}}})
        if action == "group_add_role":
            group_id = _required(args, "group_id")
            role_id = _required(args, "role_id")
            await self.add_role(group_id, role_id, str(args.get("title") or role_id), str(args.get("description") or ""))
            return _ok(action, group_id=group_id, role_id=role_id)
        if action == "group_delete_role":
            return await self._role_action(action, args, {"del": None})
        if action == "group_update_role":
            group_id = _required(args, "group_id")
            role_id = _required(args, "role_id")
            group = await self.client.scry("groups", f"/v2/ui/groups/{group_id}")
            current = _get_in(group, ["roles", role_id], {}) or {}
            meta = {
                "title": args.get("title") or current.get("title") or role_id,
                "description": args.get("description") if args.get("description") is not None else current.get("description", ""),
                "image": args.get("image") if args.get("image") is not None else current.get("image", ""),
                "cover": args.get("cover") if args.get("cover") is not None else current.get("cover", ""),
            }
            await self._poke_group(group_id, {"role": {"roles": [role_id], "a-role": {"edit": meta}}})
            return _ok(action, group_id=group_id, role_id=role_id, meta=meta)
        if action in {"group_assign_role", "group_promote"}:
            group_id = _required(args, "group_id")
            role_id = _ADMIN_ROLE_ID if action == "group_promote" else _required(args, "role_id")
            await self.assign_role(group_id, role_id, _required_ships(args))
            return _ok(action, group_id=group_id, role_id=role_id, ships=_required_ships(args))
        if action in {"group_remove_role", "group_demote"}:
            group_id = _required(args, "group_id")
            role_id = _ADMIN_ROLE_ID if action == "group_demote" else _required(args, "role_id")
            ships = _required_ships(args)
            await self._poke_group(group_id, {"seat": {"ships": ships, "a-seat": {"del-roles": [role_id]}}})
            return _ok(action, group_id=group_id, role_id=role_id, ships=ships)
        if action == "group_set_privacy":
            privacy = str(args.get("privacy") or "")
            if privacy not in _GROUP_PRIVACY:
                raise TlonToolError("privacy must be public, private, or secret")
            return await self._group_action(action, args, {"entry": {"privacy": privacy}})
        if action == "group_accept_join":
            return await self._group_action(action, args, {"entry": {"ask": {"ships": _required_ships(args), "a-ask": "approve"}}})
        if action == "group_reject_join":
            return await self._group_action(action, args, {"entry": {"ask": {"ships": _required_ships(args), "a-ask": "deny"}}})

        raise TlonToolError(f"Unsupported group action: {action}")

    async def create_group(self, *, title: str, description: str = "", member_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        slug = _slug()
        group_id = f"{self.client.ship_name}/{slug}"
        channel_id = f"chat/{self.client.ship_name}/{slug}-general"
        body = {
            "groupId": group_id,
            "meta": {
                "title": title,
                "description": description,
                "image": "",
                "cover": "",
            },
            "guestList": member_ids or [],
            "channels": [
                {
                    "channelId": channel_id,
                    "meta": {
                        "title": "General",
                        "description": "General chat",
                        "image": "",
                        "cover": "",
                    },
                }
            ],
        }
        result = await self.client.thread(
            desk="groups",
            input_mark="group-create-thread",
            thread_name="group-create-1",
            output_mark="group-ui-2",
            body=body,
        )
        return {
            "group_id": group_id,
            "channel_id": channel_id,
            "title": title,
            "description": description,
            "thread_result": result,
        }

    async def invite(self, group_id: str, ships: List[str]) -> None:
        await self.client.poke(
            "groups",
            "group-action-4",
            {
                "invite": {
                    "flag": group_id,
                    "ships": ships,
                    "a-invite": {"token": None, "note": None},
                }
            },
        )

    async def ensure_admin_role(self, group_id: str) -> None:
        group = await self.client.scry("groups", f"/v2/ui/groups/{group_id}")
        roles = group.get("roles") or {}
        admins = group.get("admins") or []
        if _ADMIN_ROLE_ID not in roles:
            await self.add_role(group_id, _ADMIN_ROLE_ID, "Admin", "Group administrator")
        if _ADMIN_ROLE_ID not in admins:
            await self._poke_group(group_id, {"role": {"roles": [_ADMIN_ROLE_ID], "a-role": {"set-admin": None}}})

    async def add_role(self, group_id: str, role_id: str, title: str, description: str) -> None:
        await self._poke_group(
            group_id,
            {
                "role": {
                    "roles": [role_id],
                    "a-role": {
                        "add": {
                            "title": title,
                            "description": description,
                            "image": "",
                            "cover": "",
                        }
                    },
                }
            },
        )

    async def assign_role(self, group_id: str, role_id: str, ships: List[str]) -> None:
        await self._poke_group(group_id, {"seat": {"ships": ships, "a-seat": {"add-roles": [role_id]}}})

    async def _role_action(self, action: str, args: Dict[str, Any], role_action: Dict[str, Any]) -> Dict[str, Any]:
        group_id = _required(args, "group_id")
        role_id = _required(args, "role_id")
        await self._poke_group(group_id, {"role": {"roles": [role_id], "a-role": role_action}})
        return _ok(action, group_id=group_id, role_id=role_id)

    async def _group_action(self, action: str, args: Dict[str, Any], group_action: Dict[str, Any]) -> Dict[str, Any]:
        group_id = _required(args, "group_id")
        await self._poke_group(group_id, group_action)
        out = {"group_id": group_id}
        if "ships" in args:
            out["ships"] = _ships(args)
        return _ok(action, **out)

    async def _poke_group(self, group_id: str, group_action: Dict[str, Any]) -> None:
        await self.client.poke(
            "groups",
            "group-action-4",
            {"group": {"flag": group_id, "a-group": group_action}},
        )


class TlonChannels:
    def __init__(self, client: TlonHttpClient):
        self.client = client

    async def handle(self, action: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if action == "channels_list":
            mode = str(args.get("mode") or "all")
            if mode == "groups":
                return _ok(action, mode=mode, groups=await self.client.scry("groups", "/v2/groups"))
            if mode in {"all", "group_channels"}:
                init = await _best_effort_scry(self.client, [("groups-ui", "/v7/init"), ("groups-ui", "/v8/init")])
                return _ok(action, mode=mode, init=init)
            if mode in {"dms", "group_dms"}:
                init = await _best_effort_scry(self.client, [("groups-ui", "/v7/init"), ("groups-ui", "/v8/init")])
                return _ok(action, mode=mode, channels=_filter_init_channels(init, mode))
            raise TlonToolError("channels_list mode must be all, groups, dms, or group_dms")
        if action == "channel_info":
            channel_id = _required(args, "channel_id")
            group_id = args.get("group_id")
            if group_id:
                group = await self.client.scry("groups", f"/v2/ui/groups/{group_id}")
                return _ok(action, channel_id=channel_id, group_id=group_id, channel=_find_channel(group, channel_id), group=group)
            groups = await self.client.scry("groups", "/v2/groups")
            return _ok(action, channel_id=channel_id, match=_find_channel_in_groups(groups, channel_id))
        if action == "channel_create":
            group_id = _required(args, "group_id")
            title = _required(args, "title")
            kind = str(args.get("kind") or "chat")
            if kind not in _CHANNEL_KINDS:
                raise TlonToolError("kind must be chat, diary, or heap")
            name = _slug()
            channel_id = f"{kind}/{self.client.ship_name}/{name}"
            await self.client.poke(
                "channels",
                os.getenv("TLON_CHANNEL_MANAGE_MARK", "channel-action-2"),
                {
                    "create": {
                        "kind": kind,
                        "group": group_id,
                        "name": name,
                        "title": title,
                        "description": str(args.get("description") or ""),
                        "meta": None,
                        "readers": [],
                        "writers": [],
                    }
                },
            )
            return _ok(action, group_id=group_id, channel_id=channel_id, title=title, kind=kind)
        if action == "channel_update":
            group_id = _required(args, "group_id")
            channel_id = _required(args, "channel_id")
            group = await self.client.scry("groups", f"/v2/ui/groups/{group_id}")
            channel = _find_channel(group, channel_id) or {}
            description = args.get("description") if args.get("description") is not None else channel.get("description", "")
            update = {
                "added": channel.get("addedToGroupAt") or channel.get("added") or int(time.time() * 1000),
                "meta": {
                    "title": args.get("title") or channel.get("title") or "",
                    "description": json.dumps(
                        {
                            "description": description,
                            "channelContentConfiguration": channel.get("contentConfiguration"),
                        }
                    ),
                    "image": args.get("image") if args.get("image") is not None else channel.get("iconImage", ""),
                    "cover": args.get("cover") if args.get("cover") is not None else channel.get("coverImage", ""),
                },
                "section": _find_channel_section(group, channel_id),
                "readers": [r.get("roleId") for r in channel.get("readerRoles", []) if isinstance(r, dict) and r.get("roleId")],
                "join": channel.get("currentUserIsMember", True),
            }
            await self._poke_group(group_id, {"channel": {"nest": channel_id, "a-channel": {"edit": update}}})
            return _ok(action, group_id=group_id, channel_id=channel_id, update=update)
        if action == "channel_delete":
            group_id = _required(args, "group_id")
            channel_id = _required(args, "channel_id")
            await self._poke_group(group_id, {"channel": {"nest": channel_id, "a-channel": {"del": None}}})
            return _ok(action, group_id=group_id, channel_id=channel_id)
        if action == "channel_add_writers":
            return await self._channel_action(action, args, {"add-writers": _roles(args)})
        if action == "channel_remove_writers":
            return await self._channel_action(action, args, {"del-writers": _roles(args)})
        if action == "channel_add_readers":
            return await self._reader_action(action, args, "add-readers")
        if action == "channel_remove_readers":
            return await self._reader_action(action, args, "del-readers")
        if action == "channel_join":
            return await self._channel_action(action, args, {"join": _required(args, "group_id")})
        if action == "channel_leave":
            return await self._channel_action(action, args, {"leave": None})
        raise TlonToolError(f"Unsupported channel action: {action}")

    async def _channel_action(self, action: str, args: Dict[str, Any], channel_action: Dict[str, Any]) -> Dict[str, Any]:
        channel_id = _required(args, "channel_id")
        await self.client.poke(
            "channels",
            os.getenv("TLON_CHANNEL_MANAGE_MARK", "channel-action-2"),
            {"channel": {"nest": channel_id, "action": channel_action}},
        )
        return _ok(action, channel_id=channel_id)

    async def _reader_action(self, action: str, args: Dict[str, Any], op: str) -> Dict[str, Any]:
        group_id = _required(args, "group_id")
        channel_id = _required(args, "channel_id")
        roles = _roles(args)
        await self._poke_group(group_id, {"channel": {"nest": channel_id, "a-channel": {op: roles}}})
        return _ok(action, group_id=group_id, channel_id=channel_id, roles=roles)

    async def _poke_group(self, group_id: str, group_action: Dict[str, Any]) -> None:
        await self.client.poke(
            "groups",
            "group-action-4",
            {"group": {"flag": group_id, "a-group": group_action}},
        )


class TlonMessages:
    def __init__(self, client: TlonHttpClient):
        self.client = client

    async def handle(self, action: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if action == "messages_history":
            channel_id = _required(args, "channel_id")
            limit = _limit(args)
            data = await self._posts(channel_id, count=limit, include_replies=_bool(args, "include_replies", True))
            posts = _extract_posts_from_response(data)
            return _ok(action, channel_id=channel_id, posts=self._format_posts(posts, args))
        if action == "messages_search":
            channel_id = _required(args, "channel_id")
            query = _required(args, "query")
            path = _search_path(channel_id, query, args.get("cursor"), _limit(args, default=500))
            app = "channels" if _is_group_channel(channel_id) else "chat"
            data = await self.client.scry(app, path)
            posts = _extract_search_posts(data)
            return _ok(action, channel_id=channel_id, query=query, posts=self._format_posts(posts, args), raw=data)
        if action == "message_context":
            channel_id = _required(args, "channel_id")
            post_id = _format_post_id(_required(args, "post_id"))
            data = await self._posts(channel_id, mode="around", cursor=post_id, count=_limit(args, default=10), include_replies=True)
            posts = _extract_posts_from_response(data)
            return _ok(action, channel_id=channel_id, post_id=post_id, posts=self._format_posts(posts, args))
        if action == "message_get":
            channel_id = _required(args, "channel_id")
            raw_post_id = _required(args, "post_id")
            if _is_group_channel(channel_id):
                post_id = _format_post_id(raw_post_id)
                path = f"/v5/{channel_id}/posts/post/{post_id}"
                data = await self.client.scry("channels", path)
            else:
                author, bare = _split_writ_id(raw_post_id, args.get("author_id") or args.get("ship") or self.client.ship_name)
                chat_type = "dm" if channel_id.startswith("~") else "club"
                data = await self.client.scry("chat", f"/v4/{chat_type}/{channel_id}/writs/writ/id/{author}/{bare}")
                post_id = bare
            return _ok(action, channel_id=channel_id, post_id=post_id, post=data)
        if action == "post_react":
            return await self._react(args, add=True, dm=False)
        if action == "post_unreact":
            return await self._react(args, add=False, dm=False)
        if action == "post_edit":
            channel_id = _required(args, "channel_id")
            post_id = _format_post_id(_required(args, "post_id"))
            story = _text_to_story(_required(args, "message"))
            await self.client.poke(
                "channels",
                os.getenv("TLON_CHANNEL_MANAGE_MARK", "channel-action-2"),
                {
                    "channel": {
                        "nest": channel_id,
                        "action": {
                            "post": {
                                "edit": {
                                    "id": post_id,
                                    "essay": {
                                        "content": story,
                                        "author": self.client.ship_name,
                                        "sent": int(time.time() * 1000),
                                        "kind": _kind_for_channel(channel_id),
                                        "meta": _post_meta(args),
                                        "blob": None,
                                    },
                                }
                            }
                        },
                    }
                },
            )
            return _ok(action, channel_id=channel_id, post_id=post_id)
        if action == "post_delete":
            channel_id = _required(args, "channel_id")
            post_id = _format_post_id(_required(args, "post_id"))
            await self.client.poke(
                "channels",
                os.getenv("TLON_CHANNEL_MANAGE_MARK", "channel-action-2"),
                {"channel": {"nest": channel_id, "action": {"post": {"del": post_id}}}},
            )
            return _ok(action, channel_id=channel_id, post_id=post_id)
        if action == "dm_accept":
            ship = _normalize_ship(_required(args, "ship"))
            await self.client.poke("chat", "chat-dm-rsvp", {"ship": ship.lstrip("~"), "accept": True})
            return _ok(action, ship=ship)
        if action == "dm_decline":
            ship = _normalize_ship(_required(args, "ship"))
            await self.client.poke("chat", "chat-dm-rsvp", {"ship": ship.lstrip("~"), "accept": False})
            return _ok(action, ship=ship)
        if action == "dm_react":
            return await self._react(args, add=True, dm=True)
        if action == "dm_unreact":
            return await self._react(args, add=False, dm=True)
        if action == "dm_delete":
            ship = _normalize_ship(_required(args, "ship"))
            post_id = _required(args, "post_id")
            author, bare = _split_writ_id(post_id, args.get("author_id") or self.client.ship_name)
            await self.client.poke(
                "chat",
                os.getenv("TLON_DM_MANAGE_MARK", "chat-dm-action-2"),
                {"ship": ship, "diff": {"id": f"{author}/{bare}", "delta": {"del": None}}},
            )
            return _ok(action, ship=ship, post_id=post_id)
        if action == "notebook_post":
            channel_id = _required(args, "channel_id")
            if not channel_id.startswith("diary/"):
                raise TlonToolError("notebook_post requires a diary channel_id")
            title = _required(args, "title")
            sent_at = int(time.time() * 1000)
            content = _story_from_args(args)
            await self.client.poke(
                "channels",
                os.getenv("TLON_CHANNEL_ACTION_MARK", "channel-action-1"),
                {
                    "channel": {
                        "nest": channel_id,
                        "action": {
                            "post": {
                                "add": {
                                    "content": content,
                                    "author": self.client.ship_name,
                                    "sent": sent_at,
                                    "kind": "/diary",
                                    "meta": {
                                        "title": title,
                                        "description": str(args.get("description") or ""),
                                        "image": str(args.get("image") or ""),
                                        "cover": str(args.get("cover") or ""),
                                    },
                                    "blob": None,
                                }
                            }
                        },
                    }
                },
            )
            return _ok(action, channel_id=channel_id, title=title, sent_at=sent_at)
        raise TlonToolError(f"Unsupported message action: {action}")

    async def _posts(self, channel_id: str, *, mode: str = "newest", cursor: Optional[str] = None, count: int = 20, include_replies: bool = True) -> Any:
        if _is_group_channel(channel_id):
            path = f"/v5/{channel_id}/posts/{mode}"
            if cursor:
                path += f"/{cursor}"
            path += f"/{count}"
            path += "/post" if include_replies else "/outline"
            return await self.client.scry("channels", path)
        chat_type = "dm" if channel_id.startswith("~") else "club"
        target = channel_id if not channel_id.startswith("~") else _normalize_ship(channel_id)
        path = f"/v4/{chat_type}/{target}/writs/{mode}"
        if cursor:
            path += f"/{cursor}"
        path += f"/{count}"
        path += "/heavy" if include_replies else "/light"
        return await self.client.scry("chat", path)

    def _format_posts(self, posts: List[Any], args: Dict[str, Any]) -> List[Dict[str, Any]]:
        resolve_blobs = _bool(args, "resolve_blobs", True)
        return [_format_post(post, resolve_blobs=resolve_blobs) for post in posts]

    async def _react(self, args: Dict[str, Any], *, add: bool, dm: bool) -> Dict[str, Any]:
        emoji = args.get("emoji")
        op = "add" if add else "del"
        if dm:
            ship = _normalize_ship(_required(args, "ship"))
            author, bare = _split_writ_id(_required(args, "post_id"), args.get("author_id") or ship)
            delta = (
                {"add-react": {"react": emoji, "author": self.client.ship_name}}
                if add
                else {"del-react": self.client.ship_name}
            )
            await self.client.poke(
                "chat",
                os.getenv("TLON_DM_MANAGE_MARK", "chat-dm-action-2"),
                {"ship": ship, "diff": {"id": f"{author}/{bare}", "delta": delta}},
            )
            return _ok("dm_react" if add else "dm_unreact", ship=ship, post_id=bare)

        channel_id = _required(args, "channel_id")
        post_id = _format_post_id(_required(args, "post_id"))
        if args.get("parent_id"):
            react_action = {
                "reply": {
                    "id": _format_post_id(str(args["parent_id"])),
                    "action": {
                        ("add-react" if add else "del-react"): (
                            {"id": post_id, "react": emoji, "ship": self.client.ship_name}
                            if add
                            else {"id": post_id, "ship": self.client.ship_name}
                        )
                    },
                }
            }
        else:
            react_action = {
                "add-react" if add else "del-react": (
                    {"id": post_id, "react": emoji, "ship": self.client.ship_name}
                    if add
                    else {"id": post_id, "ship": self.client.ship_name}
                )
            }
        await self.client.poke(
            "channels",
            os.getenv("TLON_CHANNEL_MANAGE_MARK", "channel-action-2"),
            {"channel": {"nest": channel_id, "action": {"post": react_action}}},
        )
        return _ok("post_react" if add else "post_unreact", channel_id=channel_id, post_id=post_id)


class TlonContacts:
    def __init__(self, client: TlonHttpClient):
        self.client = client

    async def handle(self, action: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if action == "contacts_list":
            return _ok(action, contacts=await self.client.scry("contacts", "/v1/all"))
        if action == "contact_self":
            return _ok(action, profile=await self.client.scry("contacts", "/v1/self"))
        if action == "contact_get":
            ship = _normalize_ship(_required(args, "ship"))
            contacts = await self.client.scry("contacts", "/v1/all")
            return _ok(action, ship=ship, contact=_find_contact(contacts, ship))
        if action == "contact_sync":
            ships = _required_ships(args)
            await self.client.poke("contacts", "contact-action", {"sync": ships})
            return _ok(action, ships=ships)
        if action == "contact_add":
            ship = _normalize_ship(_required(args, "ship"))
            await self.client.poke("contacts", "contact-action-1", {"page": {"kip": ship, "contact": {}}})
            return _ok(action, ship=ship)
        if action == "contact_remove":
            ship = _normalize_ship(_required(args, "ship"))
            await self.client.poke("contacts", "contact-action-1", {"wipe": [ship]})
            return _ok(action, ship=ship)
        if action == "contact_update":
            ship = _normalize_ship(_required(args, "ship"))
            meta: Dict[str, Any] = {}
            if args.get("title") is not None or args.get("value") is not None:
                meta["nickname"] = args.get("title") if args.get("title") is not None else args.get("value")
            if args.get("image") is not None:
                meta["avatarImage"] = args.get("image")
            await self.client.poke("contacts", "contact-action", {"edit-contact": {"ship": ship, "metadata": meta}})
            return _ok(action, ship=ship, metadata=meta)
        if action == "profile_update":
            edits = []
            field_map = {
                "title": "nickname",
                "description": "bio",
                "status": "status",
                "image": "avatar",
                "cover": "cover",
            }
            for source, target in field_map.items():
                if args.get(source) is not None:
                    edits.append({target: args.get(source)})
            if args.get("key") and args.get("value") is not None:
                edits.append({str(args["key"]): args["value"]})
            if not edits:
                raise TlonToolError("profile_update requires a profile field")
            await self.client.poke("contacts", "contact-action", {"edit": edits})
            return _ok(action, edits=edits)
        raise TlonToolError(f"Unsupported contact action: {action}")


class TlonSettingsTool:
    def __init__(self, client: TlonHttpClient):
        self.client = client

    async def handle(self, action: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if action == "settings_get":
            return _ok(action, settings=await self.get_settings())
        if action == "settings_set":
            key = _required(args, "key")
            await self.put_entry(key, args.get("value"))
            return _ok(action, key=key)
        if action == "settings_delete":
            key = _required(args, "key")
            await self.client.poke(
                "settings",
                "settings-event",
                {"del-entry": {"desk": _SETTINGS_DESK, "bucket-key": _SETTINGS_BUCKET, "entry-key": key}},
            )
            return _ok(action, key=key)
        if action in {"settings_add_to_array", "settings_remove_from_array"}:
            key = _required(args, "key")
            item = args.get("value") if args.get("value") is not None else args.get("ship") or args.get("channel_id")
            if item is None:
                raise TlonToolError(f"{action} requires value, ship, or channel_id")
            settings = await self.get_settings()
            current = list(settings.get(key) or [])
            normalized = _normalize_ship(str(item)) if key in {"dmAllowlist", "defaultAuthorizedShips"} else str(item)
            if action == "settings_add_to_array":
                updated = current if normalized in current else [*current, normalized]
            else:
                updated = [x for x in current if x != normalized]
            await self.put_entry(key, updated)
            return _ok(action, key=key, value=normalized, updated=updated)
        if action == "settings_set_channel_rule":
            channel_id = _required(args, "channel_id")
            mode = str(args.get("mode") or "restricted")
            if mode not in {"open", "restricted"}:
                raise TlonToolError("settings_set_channel_rule mode must be open or restricted")
            settings = await self.get_settings()
            rules = _parse_jsonish(settings.get("channelRules"), {})
            rule: Dict[str, Any] = {"mode": mode}
            ships = _ships(args)
            if mode == "restricted" and ships:
                rule["allowedShips"] = ships
            rules[channel_id] = rule
            await self.put_entry("channelRules", json.dumps(rules, separators=(",", ":")))
            return _ok(action, channel_id=channel_id, rule=rule)
        if action == "settings_allow_dm":
            return await self._array_ship_action(action, "dmAllowlist", args, add=True)
        if action == "settings_remove_dm":
            return await self._array_ship_action(action, "dmAllowlist", args, add=False)
        if action == "settings_allow_channel":
            return await self._array_value_action(action, "groupChannels", _required(args, "channel_id"), add=True)
        if action == "settings_remove_channel":
            return await self._array_value_action(action, "groupChannels", _required(args, "channel_id"), add=False)
        if action == "settings_authorize_ship":
            return await self._array_ship_action(action, "defaultAuthorizedShips", args, add=True)
        if action == "settings_deauthorize_ship":
            return await self._array_ship_action(action, "defaultAuthorizedShips", args, add=False)
        if action == "settings_allow_group_inviter":
            return await self._array_ship_action(action, "groupInviteAllowlist", args, add=True)
        if action == "settings_remove_group_inviter":
            return await self._array_ship_action(action, "groupInviteAllowlist", args, add=False)
        if action == "settings_open_channel":
            channel_id = _required(args, "channel_id")
            rule = await self.set_channel_rule(channel_id, {"mode": "open"})
            return _ok(action, channel_id=channel_id, rule=rule)
        if action == "settings_restrict_channel":
            channel_id = _required(args, "channel_id")
            rule: Dict[str, Any] = {"mode": "restricted"}
            ships = _ships(args)
            if ships:
                rule["allowedShips"] = ships
            await self.set_channel_rule(channel_id, rule)
            return _ok(action, channel_id=channel_id, rule=rule)
        if action == "settings_set_owner":
            ship = _normalize_ship(_required(args, "ship"))
            await self.put_entry("ownerShip", ship)
            return _ok(action, owner_ship=ship)
        if action == "settings_set_bool":
            key = _required(args, "key")
            if key not in {
                "autoDiscover",
                "autoAcceptDmInvites",
                "autoAcceptGroupInvites",
                "showModelSig",
                "ownerListenEnabled",
            }:
                raise TlonToolError("settings_set_bool key must be an OpenClaw boolean setting")
            value = _bool(args, "value", False)
            await self.put_entry(key, value)
            return _ok(action, key=key, value=value)
        if action == "owner_listen_status":
            settings = await self.get_settings()
            disabled = list(settings.get("ownerListenDisabledChannels") or [])
            return _ok(
                action,
                enabled=settings.get("ownerListenEnabled", True),
                disabled_channels=disabled,
            )
        if action == "owner_listen_set":
            enabled = _bool(args, "value", True)
            await self.put_entry("ownerListenEnabled", enabled)
            return _ok(action, enabled=enabled)
        if action == "owner_listen_channel_set":
            channel_id = _required(args, "channel_id")
            enabled = _bool(args, "value", True)
            return await self._array_value_action(
                action,
                "ownerListenDisabledChannels",
                channel_id,
                add=not enabled,
                extra={"enabled": enabled},
            )
        raise TlonToolError(f"Unsupported settings action: {action}")

    async def get_settings(self) -> Dict[str, Any]:
        try:
            raw = await self.client.scry("settings", "/all")
        except TlonToolError as exc:
            if "HTTP 404" in str(exc):
                return {}
            raise
        return _get_in(raw, ["all", _SETTINGS_DESK, _SETTINGS_BUCKET], {}) or {}

    async def put_entry(self, key: str, value: Any) -> None:
        await self.client.poke(
            "settings",
            "settings-event",
            {
                "put-entry": {
                    "desk": _SETTINGS_DESK,
                    "bucket-key": _SETTINGS_BUCKET,
                    "entry-key": key,
                    "value": value,
                }
            },
        )

    async def set_channel_rule(self, channel_id: str, rule: Dict[str, Any]) -> Dict[str, Any]:
        settings = await self.get_settings()
        rules = _parse_jsonish(settings.get("channelRules"), {})
        rules[channel_id] = rule
        await self.put_entry("channelRules", json.dumps(rules, separators=(",", ":")))
        return rule

    async def _array_ship_action(self, action: str, key: str, args: Dict[str, Any], *, add: bool) -> Dict[str, Any]:
        ship = _normalize_ship(_required(args, "ship"))
        return await self._array_value_action(action, key, ship, add=add, extra={"ship": ship})

    async def _array_value_action(
        self,
        action: str,
        key: str,
        item: str,
        *,
        add: bool,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        settings = await self.get_settings()
        current = [str(x) for x in (settings.get(key) or []) if isinstance(x, str)]
        if add:
            updated = current if item in current else [*current, item]
        else:
            updated = [x for x in current if x != item]
        await self.put_entry(key, updated)
        return _ok(action, key=key, value=item, updated=updated, **(extra or {}))


class TlonHooks:
    def __init__(self, client: TlonHttpClient):
        self.client = client

    async def handle(self, action: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if action == "hook_template":
            kind = str(args.get("kind") or "on-post")
            name = str(args.get("title") or args.get("value") or "Hermes hook")
            return _ok(action, kind=kind, source=_hook_template(name, kind))
        if action == "hook_list":
            return _ok(action, hooks=await self.client.scry("channels-server", "/v0/hooks"))
        if action == "hook_get":
            hook_id = _required(args, "hook_id")
            hooks = await self.client.scry("channels-server", "/v0/hooks")
            return _ok(action, hook_id=hook_id, hook=_get_in(hooks, ["hooks", hook_id]))
        if action == "hook_add":
            title = _required(args, "title")
            src = _source_from_args(args)
            result = await self._poke({"add": {"name": title, "src": src}})
            return _ok(action, title=title, result=result)
        if action == "hook_edit":
            hook_id = _required(args, "hook_id")
            hooks = await self.client.scry("channels-server", "/v0/hooks")
            existing = _get_in(hooks, ["hooks", hook_id], {}) or {}
            edit = {
                "id": hook_id,
                "name": args.get("title") or existing.get("name") or hook_id,
                "meta": existing.get("meta") or {},
                "src": _source_from_args(args, default=existing.get("src") or ""),
            }
            result = await self._poke({"edit": edit})
            return _ok(action, hook_id=hook_id, result=result)
        if action == "hook_delete":
            hook_id = _required(args, "hook_id")
            result = await self._poke({"del": hook_id})
            return _ok(action, hook_id=hook_id, result=result)
        if action == "hook_order":
            channel_id = _required(args, "channel_id")
            ids = _string_list_arg(args.get("hook_ids") or args.get("value"))
            if not ids:
                raise TlonToolError("hook_order requires hook_ids")
            result = await self._poke({"order": {"nest": channel_id, "seq": ids}})
            return _ok(action, channel_id=channel_id, hook_ids=ids, result=result)
        if action == "hook_config":
            hook_id = _required(args, "hook_id")
            channel_id = _required(args, "channel_id")
            config = args.get("json") if isinstance(args.get("json"), dict) else args.get("value")
            if not isinstance(config, dict):
                raise TlonToolError("hook_config requires json/value as a config object; values must be hook-encoded strings")
            result = await self._poke({"config": {"id": hook_id, "nest": channel_id, "config": config}})
            return _ok(action, hook_id=hook_id, channel_id=channel_id, result=result)
        if action == "hook_cron":
            hook_id = _required(args, "hook_id")
            result = await self._poke(
                {
                    "cron": {
                        "id": hook_id,
                        "origin": args.get("channel_id") or None,
                        "schedule": _required(args, "schedule"),
                        "config": args.get("json") if isinstance(args.get("json"), dict) else {},
                    }
                }
            )
            return _ok(action, hook_id=hook_id, result=result)
        if action == "hook_rest":
            hook_id = _required(args, "hook_id")
            result = await self._poke({"rest": {"id": hook_id, "origin": args.get("channel_id") or None}})
            return _ok(action, hook_id=hook_id, result=result)
        raise TlonToolError(f"Unsupported hook action: {action}")

    async def _poke(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        return await self.client.poke("channels-server", "hook-action-0", json_data)


class TlonMisc:
    def __init__(self, client: TlonHttpClient):
        self.client = client

    async def handle(self, action: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if action == "blocked_list":
            return _ok(action, blocked=await self.client.scry("chat", "/blocked"))
        if action == "block_ship":
            ship = _normalize_ship(_required(args, "ship"))
            await self.client.poke("chat", "chat-block-ship", {"ship": ship})
            return _ok(action, ship=ship)
        if action == "unblock_ship":
            ship = _normalize_ship(_required(args, "ship"))
            await self.client.poke("chat", "chat-unblock-ship", {"ship": ship})
            return _ok(action, ship=ship)
        if action == "activity":
            bucket = str(args.get("mode") or "all")
            if bucket not in {"all", "mentions", "replies"}:
                raise TlonToolError("activity mode must be all, mentions, or replies")
            limit = _limit(args, default=30)
            raw = await self.client.scry("activity", f"/v5/feed/init/{limit}")
            events = raw.get(bucket, []) if isinstance(raw, dict) else []
            return _ok(action, mode=bucket, events=events)
        if action == "unreads":
            raw = await self.client.scry("activity", "/v4/activity")
            return _ok(action, unreads=raw)
        if action == "upload_file":
            uploaded = await self.upload_file(args)
            return _ok(action, **uploaded)
        if action == "expose_list":
            exposed = await _best_effort_scry_or_default(self.client, [("expose", "/show")], [])
            return _ok(action, exposed=_format_exposed_list(exposed), raw=exposed)
        if action in {"expose_show", "expose_hide", "expose_check", "expose_url"}:
            cite = _expand_cite_path(_cite_path(args))
            if action == "expose_check":
                try:
                    exposed = await self.client.scry("expose", f"/show{cite}")
                except TlonToolError as exc:
                    if "HTTP 404" in str(exc):
                        exposed = False
                    else:
                        raise
                return _ok(action, cite=cite, exposed=bool(exposed))
            if action == "expose_url":
                return _ok(action, cite=cite, url=f"{self.client.ship_url}/expose{_cite_to_url_path(cite)}")
            path_parts = [part for part in cite.split("/") if part]
            await self.client.poke(
                "expose",
                "json",
                {"show": path_parts} if action == "expose_show" else {"hide": path_parts},
            )
            return _ok(action, cite=cite)
        raise TlonToolError(f"Unsupported Tlon action: {action}")

    async def upload_file(self, args: Dict[str, Any]) -> Dict[str, Any]:
        data, file_name, content_type = await _read_upload_input(args)
        file_name = str(args.get("file_name") or file_name or f"upload-{int(time.time())}")
        file_name = _safe_filename(file_name)
        content_type = str(args.get("content_type") or content_type or mimetypes.guess_type(file_name)[0] or "application/octet-stream")
        file_key = f"{self.client.ship_no_sig}/{int(time.time() * 1000)}-{file_name}"
        try:
            token = await self.client.scry("genuine", "/secret")
        except TlonToolError as exc:
            raise TlonToolError(f"upload_file needs hosted Memex/genuine storage support: {exc}") from exc
        import aiohttp

        endpoint = f"https://memex.tlon.network/v1/{self.client.ship_no_sig}/upload"
        async with self.client._session.put(
            endpoint,
            json={
                "token": token,
                "contentLength": len(data),
                "contentType": content_type,
                "fileName": file_key,
            },
            headers={"Content-Type": "application/json"},
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise TlonToolError(f"Memex upload request failed: HTTP {resp.status} - {text[:300]}")
            upload_info = await resp.json()
        upload_url = upload_info.get("url")
        hosted_url = upload_info.get("filePath")
        if not upload_url or not hosted_url:
            raise TlonToolError("Memex upload request returned no upload URL")
        async with self.client._session.put(
            upload_url,
            data=data,
            headers={"Cache-Control": "public, max-age=3600", "Content-Type": content_type},
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            if not (200 <= resp.status < 300):
                text = await resp.text()
                raise TlonToolError(f"Memex file PUT failed: HTTP {resp.status} - {text[:300]}")
        return {
            "url": hosted_url,
            "file_name": file_name,
            "content_type": content_type,
            "size": len(data),
        }


def _load_tlon_config() -> Dict[str, str]:
    try:
        from dotenv import load_dotenv

        home = os.getenv("HERMES_HOME", os.path.expanduser("~/.hermes"))
        load_dotenv(os.path.join(home, ".env"), override=False)
    except Exception:
        pass
    ship_url = os.getenv("TLON_SHIP_URL", "").strip()
    ship_name = os.getenv("TLON_SHIP_NAME", "").strip()
    ship_code = os.getenv("TLON_SHIP_CODE", "").strip()
    if not all([ship_url, ship_name, ship_code]):
        raise TlonToolError("Tlon not configured: TLON_SHIP_URL, TLON_SHIP_NAME, and TLON_SHIP_CODE are required")
    return {"ship_url": ship_url, "ship_name": ship_name, "ship_code": ship_code}


def _check_tlon_tool() -> bool:
    try:
        _load_tlon_config()
        import aiohttp  # noqa: F401

        return True
    except Exception:
        return False


def _ok(action: str, **data: Any) -> Dict[str, Any]:
    return {"success": True, "action": action, **data}


def _json_result(result: Any) -> str:
    text = json.dumps(result, ensure_ascii=False, indent=2, default=str)
    if len(text) > _MAX_RESULT_CHARS:
        text = text[:_MAX_RESULT_CHARS] + "\n... [truncated]"
    return redact_sensitive_text(text)


def _sanitize(text: str) -> str:
    return redact_sensitive_text(text)


def _required(args: Dict[str, Any], key: str) -> str:
    value = args.get(key)
    if value is None or str(value).strip() == "":
        raise TlonToolError(f"{key} is required")
    return str(value).strip()


def _ships(args: Dict[str, Any]) -> List[str]:
    values: List[str] = []
    if isinstance(args.get("ships"), list):
        values.extend(str(s) for s in args["ships"] if str(s).strip())
    if args.get("ship"):
        values.append(str(args["ship"]))
    return _unique_ships(values)


def _required_ships(args: Dict[str, Any]) -> List[str]:
    ships = _ships(args)
    if not ships:
        raise TlonToolError("ship or ships is required")
    return ships


def _unique_ships(ships: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for ship in ships:
        normalized = _normalize_ship(str(ship))
        if normalized and normalized not in seen:
            out.append(normalized)
            seen.add(normalized)
    return out


def _roles(args: Dict[str, Any]) -> List[str]:
    roles = args.get("roles")
    if isinstance(roles, list):
        out = [str(r).strip() for r in roles if str(r).strip()]
    elif args.get("role_id"):
        out = [str(args["role_id"]).strip()]
    else:
        value = args.get("value")
        out = [str(value).strip()] if value is not None and str(value).strip() else []
    if not out:
        raise TlonToolError("role_id or roles is required")
    return out


def _limit(args: Dict[str, Any], default: int = 20) -> int:
    try:
        return max(1, min(int(args.get("limit") or default), 500))
    except (TypeError, ValueError):
        return default


def _bool(args: Dict[str, Any], key: str, default: bool) -> bool:
    value = args.get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "on"}


def _slug(length: int = 8) -> str:
    chars = string.ascii_lowercase
    rest = string.ascii_lowercase + string.digits
    return random.choice(chars) + "".join(random.choice(rest) for _ in range(length - 1))


def _get_in(value: Any, path: List[str], default: Any = None) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


def _parse_jsonish(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return default
    return default


async def _best_effort_scry(client: TlonHttpClient, candidates: List[tuple[str, str]]) -> Any:
    last_error: Optional[Exception] = None
    for app, path in candidates:
        try:
            return await client.scry(app, path)
        except Exception as exc:
            last_error = exc
    if last_error:
        raise last_error
    return None


async def _best_effort_scry_or_default(client: TlonHttpClient, candidates: List[tuple[str, str]], default: Any) -> Any:
    try:
        return await _best_effort_scry(client, candidates)
    except Exception:
        return default


def _filter_init_channels(init: Any, mode: str) -> List[Any]:
    channels = init.get("channels", []) if isinstance(init, dict) else []
    if not isinstance(channels, list):
        return []
    wanted = "dm" if mode == "dms" else "groupDm"
    return [ch for ch in channels if isinstance(ch, dict) and ch.get("type") == wanted]


def _find_channel(group: Any, channel_id: str) -> Optional[Dict[str, Any]]:
    channels = group.get("channels") if isinstance(group, dict) else None
    if isinstance(channels, list):
        for channel in channels:
            if isinstance(channel, dict) and channel.get("id") == channel_id:
                return channel
    if isinstance(channels, dict):
        data = channels.get(channel_id)
        if isinstance(data, dict):
            return {"id": channel_id, **data}
    return None


def _find_channel_in_groups(groups: Any, channel_id: str) -> Optional[Dict[str, Any]]:
    iterable = groups.values() if isinstance(groups, dict) else groups
    if not isinstance(iterable, Iterable):
        return None
    for group in iterable:
        if not isinstance(group, dict):
            continue
        channel = _find_channel(group, channel_id)
        if channel:
            return {"group_id": group.get("id") or group.get("flag"), "group": group, "channel": channel}
    return None


def _find_channel_section(group: Any, channel_id: str) -> str:
    sections = group.get("navSections") or group.get("zone") or []
    if isinstance(sections, list):
        for section in sections:
            if not isinstance(section, dict):
                continue
            for channel in section.get("channels") or []:
                if isinstance(channel, dict) and channel.get("channelId") == channel_id:
                    return section.get("sectionId") or "default"
    return "default"


def _is_group_channel(channel_id: str) -> bool:
    return channel_id.startswith(("chat/", "diary/", "heap/"))


def _kind_for_channel(channel_id: str) -> str:
    if channel_id.startswith("diary/"):
        return "/diary"
    if channel_id.startswith("heap/"):
        return "/heap"
    return "/chat"


def _format_post_id(post_id: str) -> str:
    raw = str(post_id).split("/", 1)[-1].replace(".", "")
    if raw.isdigit():
        return _format_ud(int(raw))
    return str(post_id)


def _split_writ_id(post_id: str, default_author: Any) -> tuple[str, str]:
    if "/" in str(post_id):
        author, bare = str(post_id).split("/", 1)
    else:
        author, bare = str(default_author or ""), str(post_id)
    return _normalize_ship(author), _format_post_id(bare)


def _search_path(channel_id: str, query: str, cursor: Any, depth: int) -> str:
    encoded = _encode_cord(query)
    cursor_part = _format_post_id(str(cursor)) if cursor else ""
    if _is_group_channel(channel_id):
        return f"/v5/{channel_id}/search/bounded/text/{cursor_part}/{depth}/{encoded}"
    chat_type = "dm" if channel_id.startswith("~") else "club"
    return f"/{chat_type}/{channel_id}/search/bounded/text/{cursor_part}/{depth}/{encoded}"


def _encode_cord(text: str) -> str:
    # Approximate @t URL-safe rendering used by @tloncorp/api's encodeString:
    # lowercase-safe text stays readable, spaces become dots, and other bytes
    # are hex escaped with a leading tilde.
    encoded = ["~."]
    for char in text:
        if ("a" <= char <= "z") or ("0" <= char <= "9") or char == "-":
            encoded.append(char)
        elif char == " ":
            encoded.append(".")
        else:
            encoded.append("~" + format(ord(char), "x") + ".")
    return "".join(encoded)


def _extract_posts_from_response(data: Any) -> List[Any]:
    if isinstance(data, dict):
        for key in ("posts", "writs", "messages"):
            value = data.get(key)
            if isinstance(value, list):
                return value
            if isinstance(value, dict):
                return [v for v in value.values() if v is not None]
        if "post" in data:
            return [data["post"]]
    if isinstance(data, list):
        return data
    return []


def _extract_search_posts(data: Any) -> List[Any]:
    scan = data.get("scan") if isinstance(data, dict) else None
    if not isinstance(scan, list):
        return _extract_posts_from_response(data)
    posts = []
    for item in scan:
        if not isinstance(item, dict):
            continue
        for key in ("post", "writ"):
            if key in item:
                posts.append(item[key])
        if "reply" in item:
            posts.append(item["reply"])
    return posts


def _format_post(post: Any, *, resolve_blobs: bool) -> Dict[str, Any]:
    if not isinstance(post, dict):
        return {"raw": post}
    essay = post.get("essay") or post.get("memo") or post.get("content") or post
    content = essay.get("content") if isinstance(essay, dict) else post.get("content")
    blob = post.get("blob") or (essay.get("blob") if isinstance(essay, dict) else None)
    formatted = {
        "id": post.get("id") or _get_in(post, ["seal", "id"]),
        "parent_id": post.get("parentId") or post.get("parent-id") or _get_in(post, ["seal", "parent-id"]),
        "author": post.get("authorId") or post.get("author") or (essay.get("author") if isinstance(essay, dict) else None),
        "sent": post.get("sentAt") or post.get("sent") or (essay.get("sent") if isinstance(essay, dict) else None),
        "text": _extract_message_text(content),
    }
    if resolve_blobs and blob:
        entries = parse_blob_data(blob)
        formatted["blob"] = blob
        formatted["blob_annotations"] = format_blob_annotations(entries)
    return formatted


def _post_meta(args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not any(args.get(k) is not None for k in ("title", "description", "image", "cover")):
        return None
    return {
        "title": args.get("title") or "",
        "description": args.get("description") or "",
        "image": args.get("image") or "",
        "cover": args.get("cover") or "",
    }


def _story_from_args(args: Dict[str, Any]) -> List[Any]:
    value = args.get("json")
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]
    for key in ("source", "value"):
        raw = args.get(key)
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict):
            return [raw]
        if isinstance(raw, str) and raw.strip():
            stripped = raw.strip()
            if stripped.startswith("[") or stripped.startswith("{"):
                try:
                    parsed = json.loads(stripped)
                    return parsed if isinstance(parsed, list) else [parsed]
                except ValueError:
                    pass
            return _text_to_story(raw)
    return _text_to_story(str(args.get("message") or ""))


def _source_from_args(args: Dict[str, Any], default: str = "") -> str:
    for key in ("source", "message", "value"):
        value = args.get(key)
        if isinstance(value, str) and value:
            return value
    path = args.get("path")
    if isinstance(path, str) and path:
        source_path = Path(path).expanduser()
        if source_path.exists() and source_path.is_file():
            return source_path.read_text(encoding="utf-8")
    if default:
        return default
    raise TlonToolError("hook source is required via source, message, value, or local path")


def _string_list_arg(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [part.strip() for part in re.split(r"[\s,]+", value) if part.strip()]
    return []


def _hook_template(name: str, kind: str) -> str:
    safe_name = re.sub(r"[^a-zA-Z0-9-_ ]", "", name).strip() or "hermes-hook"
    if kind == "cron":
        return (
            f":: {safe_name} (cron)\n"
            ":: Runs on a configured cron schedule.\n"
            "|=  [=event:h =bowl:h]\n"
            "^-  outcome:h\n"
            "?.  ?=(%cron -.event)  &+[[[%allowed event] ~] state.hook.bowl]\n"
            "&+[[[%allowed event] ~] state.hook.bowl]\n"
        )
    if kind == "moderation":
        return (
            f":: {safe_name} (moderation)\n"
            ":: Starter moderation hook.\n"
            "|=  [=event:h =bowl:h]\n"
            "^-  outcome:h\n"
            "&+[[[%allowed event] ~] state.hook.bowl]\n"
        )
    if kind == "bare":
        return (
            f":: {safe_name} (bare)\n"
            "|=  [=event:h =bowl:h]\n"
            "^-  outcome:h\n"
            "&+[[[%allowed event] ~] state.hook.bowl]\n"
        )
    return (
        f":: {safe_name} (on-post)\n"
        ":: Reacts to post events; edit before installing.\n"
        "|=  [=event:h =bowl:h]\n"
        "^-  outcome:h\n"
        "&+[[[%allowed event] ~] state.hook.bowl]\n"
    )


def _find_contact(contacts: Any, ship: str) -> Any:
    if isinstance(contacts, dict):
        return contacts.get(ship) or contacts.get(ship.lstrip("~"))
    if isinstance(contacts, list):
        for contact in contacts:
            if isinstance(contact, dict) and contact.get("id") == ship:
                return contact
    return None


def _cite_path(args: Dict[str, Any]) -> str:
    cite = args.get("path") or args.get("channel_id")
    post_id = args.get("post_id")
    if cite and post_id and not str(cite).endswith(str(post_id)):
        return f"{cite}/{post_id}"
    if not cite:
        raise TlonToolError("expose actions require path or channel_id plus post_id")
    return str(cite).lstrip("/")


def _expand_cite_path(cite: str) -> str:
    raw = str(cite).strip()
    if raw.startswith("/1/"):
        return raw
    if raw.startswith("1/"):
        return "/" + raw
    parts = raw.strip("/").split("/")
    if len(parts) < 4:
        raise TlonToolError("cite path must be kind/~host/channel/post-id or /1/chan/kind/~host/channel/type/post-id")
    kind, host, channel = parts[0], parts[1], parts[2]
    if kind not in _CHANNEL_KINDS:
        raise TlonToolError("cite path kind must be chat, diary, or heap")
    content_type = {"chat": "msg", "diary": "note", "heap": "curio"}[kind]
    post_id = "/".join(parts[3:])
    return f"/1/chan/{kind}/{host}/{channel}/{content_type}/{post_id}"


def _cite_to_url_path(cite: str) -> str:
    full = _expand_cite_path(cite)
    if full.startswith("/1/"):
        return full[2:]
    if full.startswith("1/"):
        return "/" + full[2:]
    return full


def _format_exposed_list(exposed: Any) -> List[str]:
    if isinstance(exposed, list):
        return [_format_cite(item) for item in exposed]
    if isinstance(exposed, dict):
        return [_format_cite(item) for item in exposed.values()]
    return []


def _format_cite(cite: Any) -> str:
    if isinstance(cite, str):
        return cite
    if isinstance(cite, dict) and isinstance(cite.get("chan"), dict):
        chan = cite["chan"]
        nest = chan.get("nest")
        wer = chan.get("wer")
        if isinstance(nest, list) and len(nest) == 2:
            kind = nest[0]
            host_name = nest[1]
            if isinstance(host_name, list) and len(host_name) == 2:
                host, name = host_name
                host = _normalize_ship(str(host))
                wer_path = "/".join(str(x) for x in wer) if isinstance(wer, list) else str(wer or "")
                return f"/1/chan/{kind}/{host}/{name}/{wer_path}"
    return json.dumps(cite, default=str, separators=(",", ":"))


async def _read_upload_input(args: Dict[str, Any]) -> tuple[bytes, str, str]:
    source = str(args.get("source") or args.get("path") or "").strip()
    if not source:
        raise TlonToolError("upload_file requires source or path")
    if source.startswith(("http://", "https://")):
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.get(source, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                if resp.status != 200:
                    raise TlonToolError(f"Failed to fetch upload source: HTTP {resp.status}")
                data = await resp.read()
                content_type = resp.headers.get("content-type", "").split(";", 1)[0].strip()
        parsed = urlparse(source)
        file_name = Path(parsed.path).name or "upload"
        return data, file_name, content_type
    path = Path(source).expanduser()
    if not path.exists() or not path.is_file():
        raise TlonToolError(f"Upload file not found: {path}")
    return path.read_bytes(), path.name, mimetypes.guess_type(path.name)[0] or ""


def _safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(name).name).strip(".-")
    return cleaned or f"upload-{int(time.time())}"


registry.register(
    name="tlon",
    toolset="tlon",
    schema=TLON_SCHEMA,
    handler=tlon_tool,
    check_fn=_check_tlon_tool,
    emoji="T",
    max_result_size_chars=_MAX_RESULT_CHARS,
)
