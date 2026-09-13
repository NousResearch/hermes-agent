"""Recipient-bound peer inputs adapted from #98072 / mapped 5425e0fa.

Only source-store resolution lives here; the existing peer HTTP client owns
bounded upload, grant enforcement and dispatch.
"""
from __future__ import annotations

import hashlib
from typing import Any

from gateway.hosted_room_attachments import HostedRoomAttachmentStore
from gateway.hosted_room_driver import validate_bound_task_manifest
from gateway.hosted_room_peer import canonical_attachment_manifest


def bound_attachment_payloads(
    store: HostedRoomAttachmentStore | None,
    room_id: str,
    member_id: str,
    attachments: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Resolve immutable task references through the member's committed ACL."""
    if not attachments:
        return []
    manifest = validate_bound_task_manifest(attachments)
    if store is None:
        raise ValueError("Group Chat attachment storage is unavailable")
    result = []
    for item in manifest:
        saved = store.read(room_id=room_id, event_id=item["event_id"],
            attachment_id=item["attachment_id"], recipient_member_id=member_id)
        if any(saved.attachment[key] != item[key] for key in ("kind", "name", "mime", "size")):
            raise ValueError("Group Chat attachment identity changed")
        result.append({**{key: item[key] for key in ("attachment_id", "kind", "name", "mime", "size")},
            "sha256": hashlib.sha256(saved.data).hexdigest(), "data": saved.data})
    canonical_attachment_manifest([
        {key: value for key, value in item.items() if key != "data"} for item in result])
    return result
