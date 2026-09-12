"""Trusted TUI/Desktop issuance for model-visible hosted-room references."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

from agent.shared_discovery import SharedDiscoveryScope, build_local_discovery_scope


_UNKNOWN = "Unknown or unavailable discovery reference."
_ROOM_ACTIONS = frozenset({"inspect"})


@dataclass(frozen=True)
class RoomReferencePin:
    room_id: str
    actions: tuple[str, ...]
    participants: tuple[str, ...]
    authority_gateway_id: str
    authority_epoch: int


def _normalized_policy(cfg: Mapping[str, Any]) -> tuple[tuple[dict[str, Any], ...], str]:
    orchestration = cfg.get("orchestration", {}) if isinstance(cfg, Mapping) else {}
    discovery = orchestration.get("discovery", {}) if isinstance(orchestration, Mapping) else {}
    rows = discovery.get("rooms", []) if isinstance(discovery, Mapping) else []
    if rows is None:
        rows = []
    if not isinstance(rows, list):
        raise ValueError("orchestration.discovery.rooms must be a list")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Each discovery room grant must be a mapping")
        room_id = str(row.get("id") or "").strip()
        if not room_id or room_id in seen:
            raise ValueError("Each discovery room grant needs a unique nonempty id")
        raw_actions = row.get("actions", ["inspect"])
        raw_participants = row.get("participants", [])
        if not isinstance(raw_actions, list) or not isinstance(raw_participants, list):
            raise ValueError("Discovery room actions and participants must be lists")
        actions = tuple(sorted({str(item).strip() for item in raw_actions if str(item).strip()}))
        participants = tuple(sorted({str(item).strip() for item in raw_participants if str(item).strip()}))
        if not set(actions).issubset(_ROOM_ACTIONS):
            raise ValueError("Discovery room actions currently support only inspect")
        normalized.append({"id": room_id, "actions": list(actions), "participants": list(participants)})
        seen.add(room_id)
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return tuple(normalized), hashlib.sha256(encoded.encode()).hexdigest()


def _participant_names(room: Mapping[str, Any]) -> set[str]:
    names: set[str] = set()
    for member in room.get("members", ()):
        if not isinstance(member, Mapping):
            continue
        for key in ("profile", "handle", "member_id"):
            value = str(member.get(key) or "").strip()
            if value:
                names.add(value)
    return names


class GatewayRoomDiscoveryProvider:
    """A frozen grant whose validation remains owned by the live gateway record."""

    def __init__(
        self, server: ModuleType, *, sid: str, profile_name: str, profile_home: str,
        service: Any, db_path: str, policy_digest: str, pins: tuple[RoomReferencePin, ...],
    ) -> None:
        self.server = server
        self.sid = sid
        self.profile_name = profile_name
        self.profile_home = profile_home
        self.service = service
        self.db_path = db_path
        self.policy_digest = policy_digest
        self.pins = pins

    def _current_record(self, agent: Any) -> Mapping[str, Any]:
        record = self.server._current_runtime_session_record.get()
        with self.server._sessions_lock:
            if record is None or self.server._sessions.get(self.sid) is not record:
                raise PermissionError(_UNKNOWN)
            scope = getattr(agent, "_shared_discovery_scope", None)
            if record.get("agent") is not agent or record.get("discovery_scope") is not scope:
                raise PermissionError(_UNKNOWN)
            if not isinstance(scope, SharedDiscoveryScope) or scope.room_provider is not self:
                raise PermissionError(_UNKNOWN)
            if record.get("_compute_host_active") or record.get("source") == "bot_room":
                raise PermissionError(_UNKNOWN)
        return record

    def _validate(self, agent: Any, pin: RoomReferencePin) -> Mapping[str, Any]:
        self._current_record(agent)
        from hermes_constants import get_hermes_home
        if (self.server._current_profile_name() or "default") != self.profile_name:
            raise PermissionError(_UNKNOWN)
        if str(Path(get_hermes_home()).resolve()) != self.profile_home:
            raise PermissionError(_UNKNOWN)
        _rows, digest = _normalized_policy(self.server._load_cfg())
        if digest != self.policy_digest:
            raise PermissionError(_UNKNOWN)
        from tui_gateway.methods_groups import get_hosted_room_service
        if get_hosted_room_service() is not self.service:
            raise PermissionError(_UNKNOWN)
        if str(Path(getattr(self.service, "db_path", "")).resolve()) != self.db_path:
            raise PermissionError(_UNKNOWN)
        from gateway.hosted_rooms import local_authority_gateway_id_existing, room_state_existing
        room = room_state_existing(self.db_path, room_id=pin.room_id)
        if room is None:
            raise PermissionError(_UNKNOWN)
        if (str(room.get("authority_gateway_id")) != pin.authority_gateway_id
                or int(room.get("authority_epoch") or 0) != pin.authority_epoch
                or local_authority_gateway_id_existing() != pin.authority_gateway_id):
            raise PermissionError(_UNKNOWN)
        if not set(pin.participants).issubset(_participant_names(room)):
            raise PermissionError(_UNKNOWN)
        return room

    @staticmethod
    def _item(pin: RoomReferencePin, room: Mapping[str, Any]) -> Mapping[str, Any]:
        return {
            "reference": f"room:{pin.room_id}", "kind": "room", "label": str(room.get("name") or "Room"),
            "availability": "available", "freshness": "live", "actions": list(pin.actions),
            "authority_epoch": pin.authority_epoch, "participants": list(pin.participants),
            "scope": {"kind": "gateway_room_grant"},
        }

    def resolve(self, agent: Any, reference: str | None) -> Mapping[str, Any]:
        if reference:
            room_id = reference.partition(":")[2]
            pin = next((item for item in self.pins if item.room_id == room_id and "inspect" in item.actions), None)
            if pin is None:
                raise PermissionError(_UNKNOWN)
            return {"reference": self._item(pin, self._validate(agent, pin))}
        references = []
        for pin in self.pins:
            try:
                room = self._validate(agent, pin)
            except PermissionError:
                continue
            references.append(self._item(pin, room))
        return {"references": references}


def build_gateway_discovery_scope(
    server: ModuleType, *, sid: str, cfg: Mapping[str, Any], source: str,
) -> SharedDiscoveryScope:
    """Issue a room grant from trusted session construction, or a local-only scope."""
    base = build_local_discovery_scope()
    with server._sessions_lock:
        record = server._sessions.get(sid)
        if source == "bot_room" or (record is not None and record.get("_compute_host_active")):
            return base
    rows, digest = _normalized_policy(cfg)
    if not rows:
        return base
    from tui_gateway.methods_groups import get_hosted_room_service
    service = get_hosted_room_service()
    if service is None:
        return base
    from gateway.hosted_rooms import list_rooms_existing, local_authority_gateway_id_existing
    db_path = str(Path(service.db_path).resolve())
    gateway_id = local_authority_gateway_id_existing()
    if not gateway_id:
        return base
    rooms = {room["room_id"]: room for room in list_rooms_existing(db_path, room_ids=tuple(row["id"] for row in rows))}
    pins = []
    for row in rows:
        room = rooms.get(row["id"])
        if room is None or str(room.get("authority_gateway_id")) != gateway_id:
            continue
        participants = tuple(row["participants"])
        if not set(participants).issubset(_participant_names(room)):
            continue
        pins.append(RoomReferencePin(
            row["id"], tuple(row["actions"]), participants, gateway_id,
            int(room["authority_epoch"]),
        ))
    if not pins:
        return base
    provider = GatewayRoomDiscoveryProvider(
        server, sid=sid, profile_name=base.profile_name, profile_home=base.profile_home,
        service=service, db_path=db_path, policy_digest=digest, pins=tuple(pins),
    )
    return replace(base, room_provider=provider)
