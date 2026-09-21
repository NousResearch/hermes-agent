"""Canonical Files preparation and exact root-local output acknowledgement."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable, Mapping
from contextlib import closing
from pathlib import Path
from typing import Any, Protocol

from gateway.hosted_room_attachments import AttachmentData, HostedRoomAttachmentStore
from gateway.hosted_room_artifacts import (
    RoomArtifactError, RoomArtifactScope, validate_terminal_artifact_manifest,
)


class PublishedAttachmentStore(Protocol):
    """Read-only publication proof consumed before source acknowledgement."""

    db_path: Path

    def _require_viewer_room(
        self, conn: sqlite3.Connection, *, room_id: str,
        authority_gateway_id: str, authority_epoch: int,
    ) -> object: ...

    def read_viewer(
        self, *, room_id: Any, attachment_id: Any, event_id: Any | None = None,
        recipient_member_id: Any = None, authority_gateway_id: Any,
        authority_epoch: Any,
    ) -> AttachmentData: ...


def _snapshot(store: PublishedAttachmentStore):
    conn = sqlite3.connect(store.db_path.resolve().as_uri() + "?mode=ro", uri=True, timeout=2)
    conn.row_factory = sqlite3.Row
    return conn


def _terminal_binding(store: PublishedAttachmentStore, scope, manifest, recipient_member_ids=None):
    scope = RoomArtifactScope.from_mapping(scope.as_mapping())
    items = validate_terminal_artifact_manifest(manifest)
    if not items or len({item["artifact_id"] for item in items}) != len(items):
        raise RoomArtifactError("output manifest must contain distinct artifacts")
    with closing(_snapshot(store)) as conn:
        conn.execute("BEGIN")
        store._require_viewer_room(
            conn, room_id=scope.room_id, authority_gateway_id=scope.authority_gateway_id,
            authority_epoch=scope.authority_epoch,
        )
        room = conn.execute("SELECT members_json FROM hosted_rooms WHERE room_id=?", (scope.room_id,)).fetchone()
        members = json.loads(room["members_json"])
        if not any(member.get("member_id") == scope.member_id and member.get("profile") == scope.target_profile
                   for member in members):
            raise RoomArtifactError("output participant is no longer a room member")
        if recipient_member_ids is not None:
            member_ids = {member["member_id"] for member in members}
            if (not isinstance(recipient_member_ids, list) or not recipient_member_ids
                    or any(not isinstance(value, str) for value in recipient_member_ids)
                    or len(set(recipient_member_ids)) != len(recipient_member_ids)
                    or not set(recipient_member_ids).issubset(member_ids)):
                raise RoomArtifactError("output recipients must be actual room members")
        row = conn.execute(
            "SELECT payload_json, result_json, status, execution_generation "
            "FROM hosted_room_driver_tasks WHERE room_id=? AND task_id=?",
            (scope.room_id, scope.task_id),
        ).fetchone()
        if row is None:
            raise RoomArtifactError("output has no retained task")
        payload, result = json.loads(row["payload_json"]), json.loads(row["result_json"] or "null")
        frozen = payload.get("recipient_member_ids")
        if (not isinstance(frozen, list) or not frozen or scope.member_id not in frozen
                or any(not isinstance(value, str) for value in frozen)
                or len(set(frozen)) != len(frozen)
                or (recipient_member_ids is not None and recipient_member_ids != frozen)):
            raise RoomArtifactError("output requires admission-frozen recipients")
        if (row["status"] != "settled" or row["execution_generation"] != scope.execution_generation
                or payload.get("target_profile") != scope.target_profile
                or payload.get("target_member_id", payload.get("target_profile")) != scope.member_id
                or not isinstance(result, dict) or result.get("artifact_scope") != scope.as_mapping()
                or result.get("artifacts") != manifest):
            raise RoomArtifactError("output does not match the retained terminal attempt")
    return items


def _upload_id(scope, item):
    digest = hashlib.sha256((scope.key + "\0" + item["artifact_id"]).encode()).hexdigest()
    return "bot-output:" + digest


def prepare_output(
    store: HostedRoomAttachmentStore,
    *,
    scope: RoomArtifactScope,
    manifest: Mapping,
    recipient_member_ids: list[str],
    read_artifact: Callable,
) -> dict:
    """Verify explicit output bytes and stage an immutable event-bound manifest.

    read_artifact(scope, artifact_id) returns exact metadata and bytes through
    the caller's authenticated producer route, never a supplied host path.
    No source ACK is issued here, even when preparation is repeated.
    """
    items = _terminal_binding(store, scope, manifest, recipient_member_ids)
    canonical = []
    for item in items:
        metadata, data = read_artifact(scope, item["artifact_id"])
        if (metadata != item or not isinstance(data, bytes) or len(data) != item["size"]
                or hashlib.sha256(data).hexdigest() != item["sha256"]):
            raise RoomArtifactError("output bytes do not match the terminal manifest")
        saved = store.put(room_id=scope.room_id, upload_id=_upload_id(scope, item),
                          name=item["name"], kind=item["kind"], mime=item["mime"], data=data)
        canonical.append({key: saved[key] for key in ("attachment_id", "kind", "name", "size", "mime")})
    _terminal_binding(store, scope, manifest, recipient_member_ids)
    event_id = "dmessage:" + scope.task_id.removeprefix("dtask:")
    attachments = store.commit_message(
        room_id=scope.room_id, event_id=event_id, manifest=canonical,
        recipient_member_ids=recipient_member_ids, viewer_access=True, hold_until_event=True,
    )
    return {
        "event_id": event_id,
        "payload": {
            "task_id": scope.task_id,
            "execution_generation": scope.execution_generation,
            "artifact_scope_sha256": scope.key,
            "artifact_manifest_digest": manifest["manifest_digest"],
            "attachments": attachments,
        },
    }


def acknowledge_published(
    store: PublishedAttachmentStore,
    *,
    scope: RoomArtifactScope,
    manifest: Mapping,
    acknowledge: Callable,
):
    """ACK exact source custody only after canonical publication and byte proof.

    acknowledge(scope, artifact_ids, message_event_id=...) must use the same
    authenticated source as preparation. It is deliberately retryable by the
    caller if a response is lost; canonical bytes survive that retry.
    """
    items = _terminal_binding(store, scope, manifest)
    event_id = "dmessage:" + scope.task_id.removeprefix("dtask:")
    with closing(_snapshot(store)) as conn:
        event = conn.execute(
            "SELECT actor_json, payload_json, authority_epoch FROM hosted_room_events "
            "WHERE room_id=? AND event_id=? AND kind='message.member'",
            (scope.room_id, event_id),
        ).fetchone()
        if event is None:
            raise RoomArtifactError("output event is not published")
        payload, actor = json.loads(event["payload_json"]), json.loads(event["actor_json"])
        if (event["authority_epoch"] != scope.authority_epoch
                or actor.get("kind") != "member" or actor.get("id") != scope.member_id
                or actor.get("profile") != scope.target_profile
                or payload.get("task_id") != scope.task_id
                or ("execution_generation" in payload and payload["execution_generation"] != scope.execution_generation)
                or ("artifact_scope_sha256" in payload and payload["artifact_scope_sha256"] != scope.key)
                or ("artifact_manifest_digest" in payload and payload["artifact_manifest_digest"] != manifest["manifest_digest"])):
            raise RoomArtifactError("output publication commitment changed")
        task = conn.execute(
            "SELECT payload_json FROM hosted_room_driver_tasks WHERE room_id=? AND task_id=?",
            (scope.room_id, scope.task_id),
        ).fetchone()
        frozen = json.loads(task["payload_json"]).get("recipient_member_ids") if task else None
        if not frozen or payload.get("recipient_member_ids") != frozen:
            raise RoomArtifactError("output publication recipients changed")
        attachments = payload.get("attachments")
        if not isinstance(attachments, list) or len(attachments) != len(items):
            raise RoomArtifactError("output publication manifest changed")
        for item, attachment in zip(items, attachments):
            row = conn.execute(
                "SELECT upload_id FROM hosted_room_attachments WHERE room_id=? AND attachment_id=?",
                (scope.room_id, attachment.get("attachment_id")),
            ).fetchone()
            if row is None or row["upload_id"] != _upload_id(scope, item):
                raise RoomArtifactError("output publication source changed")
    for item, attachment in zip(items, attachments):
        saved = store.read_viewer(
            room_id=scope.room_id, event_id=event_id, attachment_id=attachment["attachment_id"],
            authority_gateway_id=scope.authority_gateway_id, authority_epoch=scope.authority_epoch,
        )
        if (any(saved.attachment[key] != item[key] for key in ("kind", "name", "size", "mime"))
                or hashlib.sha256(saved.data).hexdigest() != item["sha256"]):
            raise RoomArtifactError("published output bytes changed")
    return acknowledge(scope, [item["artifact_id"] for item in items], message_event_id=event_id)
