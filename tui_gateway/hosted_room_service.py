"""Production coordinator for same-gateway hosted Discussion rooms."""

from __future__ import annotations

import contextlib
import os
import threading
import time
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

from gateway import hosted_room_discussion as discussion
from gateway import hosted_room_driver as driver
from gateway import hosted_room_links
from gateway import hosted_rooms
from gateway.hosted_room_policy_checkpoint import HostedRoomPolicyCheckpoint, PolicySnapshot
from gateway.hosted_room_peer import (
    GatewayRoomCatalog, HostedMemberDispatch, PROTOCOL_VERSION, room_grant_needs_dispatch_refresh)
from tui_gateway.hosted_room_driver import HostedRoomBinding, HostedRoomRuntime
from tui_gateway.hosted_room_server_rpc import HostedRoomServerRPC
from tui_gateway.hosted_room_peer_http import (
    PeerRunsHTTPClient, PeerRunsHTTPError, digest_reauthorization_error)
from tui_gateway.hosted_room_peer_transport import (
    HostedRoomPeerClient, PeerHostedRoomTransport, PeerMemberRoute, build_member_dispatch)

_HOSTED_ROOM_IDLE_FALLBACK_SECONDS = 5.0
_HOSTED_ROOM_ACTIVE_POLL_SECONDS = 0.25
_HOSTED_ROOM_TERMINAL_GRACE_SECONDS = 30.0

_TERMINAL_STATUSES = ("deferred", "settled", "failed", "cancelled")
_LIVE_STATUSES = ("queued", "running", "stopping")
_STOPPABLE_STATUSES = ("queued", "running", "indeterminate", "deferred", "stopping")
_RETRYABLE_STATUSES = ("indeterminate", "deferred")


def _hosted_room_turn_timeout_seconds() -> float:
    try:
        agent_timeout = float(os.getenv("HERMES_AGENT_TIMEOUT", "1800"))
    except (TypeError, ValueError):
        agent_timeout = 0.0
    return (agent_timeout if agent_timeout > 0 else 1800.0) + _HOSTED_ROOM_TERMINAL_GRACE_SECONDS


def _grant_revoke_is_terminal(exc: PeerRunsHTTPError) -> bool:
    """Return whether the peer proves the scoped grant is already unusable."""
    return exc.status_code in {401, 403} and exc.error_code in {
        "invalid_room_grant", "room_reauthorization_required"}


def _hook(obj: Any, name: str):
    """Optional callable attribute of a duck-typed peer client, or None."""
    value = getattr(obj, name, None)
    return value if callable(value) else None


def _authority(room: Mapping[str, Any]) -> tuple[str, int]:
    return str(room["authority_gateway_id"]), int(room["authority_epoch"])


class HostedRoomService:
    """Own the hosted Discussion policy and its transport-free worker."""

    def __init__(
        self, server: ModuleType, *, db_path: Path | str | None = None,
        peer_routes: Mapping[tuple[str, str], PeerMemberRoute] | None = None,
        peer_clients: Mapping[Any, HostedRoomPeerClient] | None = None) -> None:
        self.server, self.db_path = server, Path(db_path or hosted_rooms.default_db_path())
        hosted_rooms.prune_disbanded_rooms(self.db_path)
        self._policy_lock = threading.RLock()
        self._pending_actions: dict[tuple[str, str], dict[str, Any]] = {}
        self.policy_checkpoint = HostedRoomPolicyCheckpoint(self.db_path)
        self.rpc = HostedRoomServerRPC(server)
        self._link_load_error = None
        self._peer_route_status: dict[tuple[str, str], str] = {}
        self.peer_routes: dict[tuple[str, str], PeerMemberRoute] = {}
        self.peer_clients: dict[tuple[str, str], Any] = {}
        try:
            self._load_stored_links()
        except Exception as exc:
            self._link_load_error = str(exc)
        supplied_clients = dict(peer_clients or {})
        for key, route in dict(peer_routes or {}).items():
            self.peer_routes[key] = route
            client = supplied_clients.get(key, supplied_clients.get(route.target_install_id))
            if client is not None:
                self.peer_clients[key] = client
        self.runtime = HostedRoomRuntime(
            db_path=self.db_path, rooms=self.bindings, rpc=self.rpc,
            transport_resolver=self._resolve_member_transport, turn_lock=self._turn_lock,
            prepare_room=self.prepare_room, publish_terminal=self.publish_terminal,
            pending_action=self._set_pending_action,
            poll_interval_seconds=_HOSTED_ROOM_IDLE_FALLBACK_SECONDS,
            active_poll_interval_seconds=_HOSTED_ROOM_ACTIVE_POLL_SECONDS,
            turn_timeout_seconds=_hosted_room_turn_timeout_seconds())

    def _load_stored_links(self) -> None:
        """Rehydrate persisted peer routes; collect per-link errors into one string."""
        stored_links, load_errors = hosted_room_links.load_room_links_tolerant(self.db_path)
        errors = list(load_errors)
        for stored in stored_links:
            key, catalog = (stored.room_id, stored.member_id), stored.catalog
            if PROTOCOL_VERSION not in catalog.protocol_versions:
                errors.append(f"{stored.room_id}:{stored.member_id}:protocol-upgrade-required")
                continue
            self.peer_routes[key] = PeerMemberRoute(
                home_install_id=hosted_rooms.local_authority_gateway_id(),
                member_id=stored.member_id, target_install_id=catalog.installation_id,
                target_profile=stored.target_profile, capability_digest=catalog.catalog_digest,
                execution_policy_digest=catalog.execution_policy.policy_digest,
                cancellation_scope_id=stored.cancellation_scope_id, trace_id=stored.trace_id,
                grant=stored.grant)
            self.peer_clients[key] = PeerRunsHTTPClient(
                base_url=stored.target_url, api_key="", receipt_db_path=self.db_path)
            self._peer_route_status[key] = stored.status
        if errors:
            self._link_load_error = ",".join(errors)

    @property
    def root(self) -> Path:
        return self.db_path.parent

    def local_profiles(self) -> tuple[str, ...]:
        profiles, profiles_dir = {"default"}, self.root / "profiles"
        if profiles_dir.is_dir():
            profiles.update(path.name for path in profiles_dir.iterdir() if path.is_dir())
        return tuple(sorted(profiles))

    def bindings(self) -> tuple[HostedRoomBinding, ...]:
        local_gateway_id = hosted_rooms.local_authority_gateway_id()
        return tuple(
            HostedRoomBinding(str(room["room_id"]), *_authority(room))
            for room in hosted_rooms.list_rooms(self.db_path)
            if str(room["authority_gateway_id"]) == local_gateway_id)

    def _room(self, room_id: str) -> dict[str, Any]:
        return hosted_rooms.room_state(self.db_path, room_id=room_id)

    def _owned_authority(self, room_id: str) -> tuple[str, int]:
        """(gateway_id, epoch) of a room this gateway owns; conflict error otherwise."""
        gateway_id, epoch = _authority(self._room(room_id))
        if gateway_id != hosted_rooms.local_authority_gateway_id():
            raise hosted_rooms.AuthorityConflictError(
                "This Group Chat is managed by another gateway.")
        return gateway_id, epoch

    def _turn_lock(self, profile: str) -> contextlib.AbstractContextManager[Path]:
        from tools.bot_relay import acquire_turn_lock
        return acquire_turn_lock(self.root, profile)

    def start(self) -> None:
        self.runtime.start()

    def stop(self, *, timeout: float = 5.0) -> bool:
        return self.runtime.stop(timeout=timeout)

    def wakeup(self) -> None:
        self.runtime.wakeup()

    def _list_tasks(self, room_id: str, statuses) -> Iterator[Mapping[str, Any]]:
        for status in statuses:
            yield from driver.list_tasks(self.db_path, room_id=room_id, status=status)

    def _save_link(self, **link: Any) -> None:
        """Persist one stored link (``make_stored_link`` keyword fields)."""
        hosted_room_links.save_room_link(self.db_path, hosted_room_links.make_stored_link(**link))

    def register_peer_route(
        self, *, room_id: str, member_id: str, route: PeerMemberRoute,
        client: HostedRoomPeerClient, target_url: str | None = None,
        catalog: GatewayRoomCatalog | None = None) -> None:
        """Register one verified route and optionally persist its scoped grant."""
        bind_store = _hook(client, "bind_receipt_store")
        if bind_store is not None:
            bind_store(self.db_path)
        if catalog is not None:
            if not route.execution_policy_digest:
                route = replace(
                    route, execution_policy_digest=catalog.execution_policy.policy_digest)
            if (
                route.capability_digest != catalog.catalog_digest
                or route.execution_policy_digest != catalog.execution_policy.policy_digest):
                raise ValueError("peer route does not match its target catalog")
            if target_url is not None:
                self._save_link(
                    room_id=room_id, member_id=member_id, target_url=target_url,
                    target_profile=route.target_profile, grant=route.grant, catalog=catalog,
                    cancellation_scope_id=route.cancellation_scope_id, trace_id=route.trace_id)
        # Persistence is the publication boundary: a failed disk write must never
        # leave a process-local route that disappears after restart.
        self._publish_route((room_id, member_id), route, client)
        self.runtime.wakeup()

    def _publish_route(self, key: tuple[str, str], route: PeerMemberRoute, client=None) -> None:
        """Make a persisted route live as ``ready`` (and bind its client when given)."""
        with self._policy_lock:
            self.peer_routes[key], self._peer_route_status[key] = route, "ready"
            if client is not None:
                self.peer_clients[key] = client

    def revoke_room_routes(self, room_id: str) -> int:
        """Revoke and forget every scoped peer route for one room; an unreachable target
        leaves the room intact for retry rather than a false disband with a live grant."""
        with self._policy_lock:
            routes = [(key, route) for key, route in self.peer_routes.items() if key[0] == room_id]
        for key, route in routes:
            revoke = _hook(self.peer_clients.get(key), "revoke_grant")
            if revoke is None:
                raise RuntimeError("peer room grant cannot be revoked safely")
            try:
                revoke(grant=route.grant)
            except PeerRunsHTTPError as exc:
                if not _grant_revoke_is_terminal(exc):
                    raise
        hosted_rooms.delete_room_link_records(self.db_path, room_id=room_id)
        with self._policy_lock:
            for key, _route in routes:
                for table in (self.peer_routes, self._peer_route_status, self.peer_clients):
                    table.pop(key, None)
        return len(routes)

    def _resolve_member_transport(self, binding: HostedRoomBinding, task: Mapping[str, Any]):
        payload = task.get("payload", {})
        member_id = str(payload.get("target_member_id") or payload.get("target_profile") or "")
        key = (binding.room_id, member_id)
        route = self.peer_routes.get(key)
        if route is None:
            if self._member_is_peer(binding.room_id, member_id):
                raise RuntimeError("peer room route is unavailable")
            return self.rpc
        client = self.peer_clients.get(key)
        if client is None:
            raise RuntimeError("peer room client is unavailable")
        identity = task.get("identity")
        execution_generation = int(task.get("execution_generation") or 0)
        bind_observation = _hook(client, "bind_observation")
        if (
            bind_observation is not None and isinstance(identity, driver.TaskIdentity)
            and execution_generation > 0):
            bind_observation(task_id=identity.task_id, execution_generation=execution_generation)

        def set_status(status: str):
            return lambda: self._set_route_status(*key, status)
        tracked_client = _RouteStatusPeerClient(
            client, on_ready=set_status("ready"),
            on_reauthorization=set_status("needs_reauthorization"),
            on_unavailable=set_status("unavailable"),
            on_refreshed=lambda grant, catalog=None: self._rotate_route_grant(
                *key, grant, catalog))
        self._recover_peer_admission(binding, task, route, tracked_client)
        return PeerHostedRoomTransport(
            binding=binding, route=route, client=tracked_client,
            source_event_seq=int(payload.get("source_event_seq") or 0),
            task_id=getattr(identity, "task_id", None), execution_generation=execution_generation)

    def _recover_peer_admission(
        self, binding: HostedRoomBinding, task: Mapping[str, Any], route: PeerMemberRoute,
        client: Any) -> None:
        """Rediscover an admitted peer run without advancing its generation."""
        recover = _hook(client, "recover_dispatch")
        identity, payload = task.get("identity"), task.get("payload")
        execution_generation = int(task.get("execution_generation") or 0)
        if (
            recover is None or not isinstance(identity, driver.TaskIdentity)
            or not isinstance(payload, Mapping) or execution_generation < 1
            or task.get("status") not in {"running", "indeterminate", "stopping"}):
            return
        prompt = payload.get("prompt")
        source_event_seq = int(payload.get("source_event_seq") or 0)
        if not isinstance(prompt, str) or source_event_seq < 1 or not route.trace_id:
            raise RuntimeError("peer room admission identity is unavailable for recovery")
        dispatch = build_member_dispatch(
            binding=binding, route=route, room_id=identity.room_id, task_id=identity.task_id,
            target_profile=route.target_profile, execution_generation=execution_generation,
            source_event_seq=source_event_seq, prompt=prompt, trace_id=route.trace_id)
        recover(dispatch=dispatch.as_mapping(), grant=route.grant)

    def _member_is_peer(self, room_id: str, member_id: str) -> bool:
        for m in self._room(room_id).get("members") or []:
            if isinstance(m, Mapping) and str(
                    m.get("member_id") or m.get("profile") or "") == member_id:
                target = m.get("target")
                return isinstance(target, Mapping) and target.get("kind") == "peer"
        return False

    def _set_route_status(self, room_id: str, member_id: str, status: str) -> None:
        with self._policy_lock:
            if self._peer_route_status.get((room_id, member_id)) == status:
                return
            self._peer_route_status[(room_id, member_id)] = status
        hosted_room_links.mark_room_link_status(
            self.db_path, room_id=room_id, member_id=member_id, status=status)

    def _set_pending_action(
        self, room_id: str, member_id: str, action: Mapping[str, Any] | None) -> None:
        with self._policy_lock:
            if action is None:
                self._pending_actions.pop((room_id, member_id), None)
            else:
                self._pending_actions[(room_id, member_id)] = {**action, "member_id": member_id}

    def _rotate_route_grant(
        self, room_id: str, member_id: str, grant: str, catalog: GatewayRoomCatalog | None = None
    ) -> None:
        """Persist a target-refreshed scoped grant before publishing it live."""
        key = (room_id, member_id)
        route = self.peer_routes.get(key)
        if route is None:
            raise RuntimeError("peer room route is unavailable")
        stored = next((
            l for l in hosted_room_links.load_room_links(self.db_path)
            if (l.room_id, l.member_id) == key), None)
        if stored is None:
            raise RuntimeError("peer room route cannot be renewed before persistence")
        digests = {}
        if catalog is not None:
            if (
                catalog.installation_id != route.target_install_id
                or catalog.execution_policy.target_profile != route.target_profile
                or PROTOCOL_VERSION not in catalog.protocol_versions
                or "direct" not in catalog.link_modes or not catalog.text
                or catalog.execution_policy.policy_digest != route.execution_policy_digest):
                self._set_route_status(room_id, member_id, "needs_reauthorization")
                raise RuntimeError(
                    "peer room execution policy changed; reauthorization is required")
            digests = {
                "capability_digest": catalog.catalog_digest,
                "execution_policy_digest": catalog.execution_policy.policy_digest}
        self._save_link(
            room_id=room_id, member_id=member_id, target_url=stored.target_url,
            target_profile=stored.target_profile, grant=grant, catalog=catalog or stored.catalog,
            cancellation_scope_id=stored.cancellation_scope_id, trace_id=stored.trace_id)
        self._publish_route(key, replace(route, grant=grant, **digests))

    def _route_statuses(self, room_id: str | None = None) -> list[dict[str, str]]:
        with self._policy_lock:
            rows = sorted(self._peer_route_status.items())
        return [
            {"room_id": key[0], "member_id": key[1], "status": status}
            for key, status in rows if room_id is None or key[0] == room_id]

    def _events(self, room_id: str) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        cursor = 0
        while True:
            page = hosted_rooms.read_events(
                self.db_path, room_id=room_id, since_seq=cursor, limit=hosted_rooms.MAX_LOG_LIMIT)
            rows = page.get("events")
            if isinstance(rows, list):
                events.extend(row for row in rows if isinstance(row, dict))
            next_cursor = int(page.get("cursor") or cursor)
            if not page.get("has_more"):
                return events
            if next_cursor <= cursor:
                raise RuntimeError("hosted room replay cursor did not advance")
            cursor = next_cursor

    def _policy_snapshot(self, room: Mapping[str, Any]) -> PolicySnapshot:
        return self.policy_checkpoint.snapshot(
            room_id=str(room["room_id"]), latest_seq=int(room["latest_seq"]))

    def _publish_terminal_tasks(self, room: Mapping[str, Any]) -> bool:
        changed, room_id, local_profiles = False, str(room["room_id"]), self.local_profiles()
        for task in self._list_tasks(room_id, _TERMINAL_STATUSES):
            status, execution_generation = task["status"], int(task["execution_generation"])
            if self.policy_checkpoint.publication_exists(
                room_id=room_id, task_id=task["identity"].task_id, status=status,
                execution_generation=execution_generation):
                continue
            task_events = self.policy_checkpoint.events_for_task(
                room_id=room_id, source_event_seq=int(task["payload"]["source_event_seq"]))
            plan = discussion.reconstruct_task_plan(
                room, task_events, task, local_profiles=local_profiles)
            publication = discussion.plan_publication(
                room, task_events, plan, status=status, result=task.get("result"),
                execution_generation=execution_generation if status == "deferred" else None,
                local_profiles=local_profiles)
            for event in publication.events:
                hosted_rooms.append_event(self.db_path, **event.append_kwargs(room_id))
            changed = True
        return changed

    def _append_room_status(
        self, room: Mapping[str, Any], decision: discussion.DiscussionDecision) -> None:
        if decision.discussion_event_id is None:
            return
        gateway_id, epoch = _authority(room)
        hosted_rooms.append_event(
            self.db_path, room_id=str(room["room_id"]),
            event_id=f"dactivity:{decision.discussion_event_id}:{decision.reason}",
            kind="room.activity", actor={"kind": "gateway", "id": gateway_id},
            payload={
                "status": decision.status, "reason_code": decision.reason,
                "thread_id": decision.thread_id,
                "discussion_event_id": decision.discussion_event_id},
            authority_gateway_id=gateway_id, authority_epoch=epoch)

    def prepare_room(self, binding: HostedRoomBinding) -> None:
        with self._policy_lock:
            room = self._room(binding.room_id)
            snapshot = self._policy_snapshot(room)  # sync() side effect feeds the publish below
            if self._publish_terminal_tasks(room):
                room = self._room(binding.room_id)
                snapshot = self._policy_snapshot(room)
            self.policy_checkpoint.compact_completed(room_id=binding.room_id)
            driver.prune_published_terminal_tasks(
                self.db_path, room_id=binding.room_id, clock=self.runtime.clock)
            if next(iter(self._list_tasks(binding.room_id, _LIVE_STATUSES)), None) is not None:
                return
            decision = discussion.plan_next_task(
                room, list(snapshot.events), local_profiles=self.local_profiles(),
                initial_watermarks=snapshot.watermarks)
            if decision.status == "task" and decision.task is not None:
                driver.admit_task(
                    self.db_path, decision.task.identity, payload=decision.task.payload,
                    clock=time.time)
                # A stop can race the policy read from another process: re-read after admission
                # and cancel a task whose source event is now behind the room stop fence.
                fence = self._policy_snapshot(self._room(binding.room_id)).stopped_through_seq
                if decision.source_event_seq is not None and decision.source_event_seq < fence:
                    self.runtime.cancel(decision.task.identity, cancel_id=f"stop-fence:{fence}")
            elif decision.status in {"settled", "bounded"}:
                self._append_room_status(room, decision)

    def publish_terminal(self, binding: HostedRoomBinding, _task: Mapping[str, Any]) -> None:
        self.prepare_room(binding)
        self.runtime.wakeup()

    def create_room(self, *, room_id: str, name: str, members: Any) -> dict[str, Any]:
        normalized = discussion.validate_roster(members, local_profiles=self.local_profiles())
        room = hosted_rooms.create_room(
            self.db_path, room_id=room_id, name=name,
            members=[
                {
                    "member_id": member.member_id, "profile": member.profile,
                    "handle": member.handle, "target": dict(member.target or {}),
                    **({"display_name": member.display_name} if member.display_name else {})}
                for member in normalized],
            authority_gateway_id=hosted_rooms.local_authority_gateway_id())
        self.runtime.wakeup()
        return room

    def send(self, *, room_id: str, event_id: str, payload: Any) -> dict[str, Any]:
        normalized = discussion.validate_user_payload(payload)
        gateway_id, epoch = self._owned_authority(room_id)
        event = hosted_rooms.append_event(
            self.db_path, room_id=room_id, event_id=event_id, kind="message.user",
            actor={"kind": "user", "id": "desktop"}, payload=normalized,
            authority_gateway_id=gateway_id, authority_epoch=epoch)
        binding = next((b for b in self.bindings() if b.room_id == room_id), None)
        if binding is None:
            raise hosted_rooms.RoomNotFoundError("hosted room not found")
        self.prepare_room(binding)
        self.runtime.wakeup()
        return event

    def stop_room(
        self, room_id: str, *, cancel_id: str, require_acknowledged: bool = False) -> int:
        gateway_id, epoch = self._owned_authority(room_id)
        hosted_rooms.request_room_stop(
            self.db_path, room_id=room_id, cancel_id=cancel_id, expected_gateway_id=gateway_id,
            expected_epoch=epoch)
        pending = 0
        with self._policy_lock:
            tasks = {
                (task["identity"].room_id, task["identity"].task_id): task
                for task in self._list_tasks(room_id, _STOPPABLE_STATUSES)}
            for task in tasks.values():
                own_cancel_id = (
                    task.get("status") == "stopping" and str(task.get("cancel_id") or ""))
                result = self.runtime.cancel(task["identity"], cancel_id=own_cancel_id or cancel_id)
                if result["status"] == "stopping":
                    pending += 1
        if require_acknowledged and pending:
            raise RuntimeError("room work is still stopping; retry deletion after Stop completes")
        self.runtime.wakeup()
        return len(tasks)

    def retry_room_task(self, room_id: str, *, task_id: str) -> dict[str, Any]:
        """Retry one uncertain or deferred task only after explicit user action."""
        candidates = self._list_tasks(room_id, _RETRYABLE_STATUSES)
        task = next((c for c in candidates if c["identity"].task_id == task_id), None)
        if task is None:
            raise driver.InvalidTaskTransitionError("no retryable room task matches task_id")
        return self.runtime.retry_indeterminate(task["identity"])

    def approve_room_task(
        self, room_id: str, *, member_id: str, task_id: str, execution_generation: int,
        choice: str, request_id: str | None = None) -> Mapping[str, Any]:
        """Resolve one exact local or peer approval and wake room observation."""
        key = (room_id, member_id)
        route, client = self.peer_routes.get(key), self.peer_clients.get(key)
        with self._policy_lock:
            action = self._pending_actions.get(key)
        requested_approval_id = str(request_id or "")

        def matches(pending: Mapping[str, Any] | None) -> bool:
            return pending is not None and (
                str(pending.get("request_id") or ""), pending.get("task_id"),
                int(pending.get("execution_generation") or 0),
            ) == (requested_approval_id, task_id, execution_generation)
        if not requested_approval_id or not matches(action):
            raise RuntimeError("room approval is no longer pending")
        if choice not in {"once", "deny"}:
            raise RuntimeError("room approval choice must be once or deny")
        approve = _hook(client, "approve_receipt")
        if route is not None and approve is not None:
            result = approve(
                task_id=task_id, execution_generation=execution_generation,
                request_id=requested_approval_id, choice=choice, grant=route.grant)
        else:
            session_id = str(action.get("session_id") or "")
            if not session_id:
                raise RuntimeError("local room approval identity is unavailable")
            result = self.rpc.approve(
                session_id=session_id, request_id=requested_approval_id, choice=choice)
        if result is None:
            raise RuntimeError("room approval target is unavailable")
        with self._policy_lock:
            if matches(self._pending_actions.get(key)):
                self._pending_actions.pop(key, None)
        self.runtime.wakeup()
        return result

    def status(self, room_id: str | None = None) -> dict[str, Any]:
        runtime = {**self.runtime.status(), "peer_routes": self._route_statuses(room_id)}
        if self._link_load_error:
            runtime["link_load_error"] = self._link_load_error
        if room_id is None:
            return runtime
        tasks = driver.list_tasks(self.db_path, room_id=room_id)
        counts = Counter(str(task["status"]) for task in tasks)
        pending_actions = [
            {"kind": "retry", "task_id": task["identity"].task_id}
            for task in tasks if task["status"] in _RETRYABLE_STATUSES]
        with self._policy_lock:
            pending_actions.extend(
                dict(action) for (action_room_id, _member_id), action
                in self._pending_actions.items() if action_room_id == room_id)
        return {
            "running": runtime["running"], "working": any(counts.get(s) for s in _LIVE_STATUSES),
            "blocked": room_id in runtime["blocked_rooms"]
            or bool(counts.get("indeterminate") or counts.get("stopping")),
            "counts": dict(counts), "pending_actions": pending_actions,
            "peer_routes": self._route_statuses(room_id)}


class _RouteStatusPeerClient:
    """Classify scoped-auth failures without exposing route credentials."""

    def __init__(
        self, client, *, on_ready, on_reauthorization, on_unavailable, on_refreshed) -> None:
        self._client, self._on_ready, self._on_refreshed = client, on_ready, on_refreshed
        self._on_reauthorization, self._on_unavailable = on_reauthorization, on_unavailable

    def _refresh_grant(self, kwargs: dict) -> dict:
        """Rotate an expiring grant before dispatch; return the kwargs to send. Refresh
        failures escalate to reauthorization only when the peer says so or the grant is
        past its hard expiry; otherwise the original grant is tried as-is. A refreshed
        catalog whose digests drift from the dispatch is a policy change: refused."""
        grant = kwargs["grant"]
        if not room_grant_needs_dispatch_refresh(grant):
            return kwargs
        checked = HostedMemberDispatch.from_mapping(kwargs["dispatch"])
        refresh = _hook(self._client, "refresh_grant")
        if refresh is None:
            return kwargs
        try:
            refreshed = refresh(
                grant=grant, capability_digest=checked.capability_digest,
                execution_policy_digest=checked.execution_policy_digest)
        except Exception as exc:
            if getattr(exc, "needs_reauthorization", False) or (
                room_grant_needs_dispatch_refresh(grant, leeway_seconds=0)):
                self._on_reauthorization()
                raise
            return kwargs
        replacement = str(refreshed.get("grant") or "")
        if not replacement:
            raise RuntimeError("peer returned no refreshed room grant")
        refreshed_catalog = None
        if refreshed.get("catalog") is not None:
            refreshed_catalog = GatewayRoomCatalog.from_mapping(refreshed.get("catalog"))
            drift = digest_reauthorization_error(
                refreshed_catalog, capability_digest=checked.capability_digest,
                execution_policy_digest=checked.execution_policy_digest)
            if drift is not None:
                self._on_reauthorization()
                raise drift
        self._on_refreshed(replacement, refreshed_catalog)
        return {**kwargs, "grant": replacement}

    def __getattr__(self, name):
        value = getattr(self._client, name)
        if not callable(value):
            return value

        def tracked(*args, **kwargs):
            if name in {"dispatch", "recover_dispatch"} and "grant" in kwargs:
                kwargs = self._refresh_grant(kwargs)
            try:
                result = value(*args, **kwargs)
            except Exception as exc:
                if getattr(exc, "needs_reauthorization", False):
                    self._on_reauthorization()
                elif getattr(exc, "not_admitted", False):
                    self._on_unavailable()
                raise
            if name != "prepare":
                self._on_ready()
            return result
        return tracked


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import hashlib  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
